"""Horizontally partitioned bloom filter spanning multiple physical files.

A single BloomFilter, as implement in filter.py, is entirely adequate
until the operational cost of checkpointing it becomes the bottleneck
rather than the memory it occupies. Persisting one large filter means
rewriting its complete bit array on every checkpoint, even when only a
small fraction of it changed since the previous save. A ShardedBloomFilter
addresses that specific cost, not memory, by partitioning one logical
filter across several independent BloomFilter instances, each backed by
its own file, and tracking which of those shards were actually touched
since the last checkpoint so that only the changed ones are rewritten.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

from vstg.core import shard_index
from vstg.filter import BloomFilter
from vstg.persist import PersistPolicy, load, save, should_checkpoint


class ShardedBloomFilter:
  """A logically single bloom filter partitioned across multiple shard files.

  Every member is routed to exactly one underlying shard through
  shard_index, and every operation, insertion, or membership testing,
  is delegated entirely to that one shard. From the perspective of
  correctness, this behaves identically to one large BloomFilter, the
  same asymmetric guarantee applies, a negative answer from the
  correct shard is certain, a positive answer is probable. What
  differs in persistence: each shard checkpoints to its own file, and
  a checkpoint only never rewrites shards that received an insertion
  since the previous one.

  Instances are not safe for concurrent use from multiple threads
  without external synchronization, for the same reasons the
  underlying BloomFilter and ManagedBloomFilter are not.
  """

  def __init__(
      self,
      shards: tuple[BloomFilter, ...],
      directory: Path,
      policy: PersistPolicy,
      clock: Callable[[], float] = time.monotonic,
  ) -> None:
    """Bind an already constructed set of shards to a directory and policy.
    
    Most callers should prefer the open classmethod over calling this
    constructor directly, since open additionally handles resuming
    each shard from its own checkpoint file if one exists.

    Args:
      shards: The BloomFilter instances composing this sharded filter,
              ordered such that a member routed to shard index i belongs
              to shards[i].
      directory: The directory each shard's checkpoint file resides within.
      policy: The rules governing when an automatic checkpoint occurs.
      clock: A zero argument callable returning the current time, in seconds.
    """
    self._shards = shards
    self._directory = directory
    self._policy = policy
    self._clock = clock
    self._dirty_shard_indices: set[int] = set()
    self._last_checkpoint_at = clock()
    self._inserts_since_checkpoint = 0

  @staticmethod
  def _shard_path(directory: Path, index: int) -> Path:
    """Compute the deterministic file path a given shard index resides at.
    
    Zero padding to four digits keeps filenames lexicographically
    sortable up to 9_999 shards, which comfortably exceeds any
    realistic shard count.

    Args:
      directory: The directory shard files reside within.
      index: The shard index whose path is being computed.

    Returns:
      The file path the given shard index's checkpoint occupies.
    """
    return directory / f"shard-{index:04d}.bloom"

  @classmethod
  def open(
    cls,
    directory: Path,
    shard_count: int,
    capacity: int,
    error_rate: float,
    policy: PersistPolicy,
    clock: Callable[[], float] = time.monotonic,
  ) -> ShardedBloomFilter:
    """Resume each shard from its existing checkpoint, or begin fresh where none exists.
    
    The requested total capacity is divided evenly across the shard
    count, rounding upward so that the combined capacity of every
    shard never falls short of what was requested, at the cost of
    occasionally exceeding it slightly.

    Args:
      directory: The directory shard checkpoint files reside within,
                 or should be created within if it does not yet exist.
      shard_count: The number of independent shards to partition across.
      capacity: The total member capacity across all shards combined.
      error_rate: The false positive rate each individual shard is
                  sized to honor.
      policy: The rules governing when an automatic checkpoint occurs.
      clock: A zero argument callable returning the current time, in seconds.

    Returns:
      A fully initialized ShardedBloomFilter, with each shard either
      resumed from disk or freshly allocated.

    Raises:
      ValueError: If shard_count is not positive.
    """
    if shard_count <= 0:
      raise ValueError("shard_count must be a positive integer")

    directory.mkdir(parents=True, exist_ok=True)
    capacity_per_shard = math.ceil(capacity / shard_count)

    shards: list[BloomFilter] = []

    for index in range(shard_count):
      shard_path = cls._shard_path(directory, index)

      if shard_path.exists():
        shards.append(load(shard_path))
      else:
        shards.append(BloomFilter(capacity_per_shard, error_rate))

    return cls(tuple(shards), directory, policy, clock)

  @property
  def shard_count(self) -> int:
    """The number of independent shards composing this filter."""
    return len(self._shards)

  @property
  def dirty_shard_count(self) -> int:
    """The number of shards that have received an insertion since the last checkpoint."""
    return len(self._dirty_shard_indices)

  def add(self, member: bytes) -> None:
    """Insert a member into its routed shard, then checkpoint automatically if warranted.

    Args:
      member: The raw bytes of the element being inserted.
    """
    index = shard_index(member, len(self._shards))
    self._shards[index].add(member)
    self._dirty_shard_indices.add(index)
    self._inserts_since_checkpoint += 1

    current_time = self._clock()

    if should_checkpoint(
      self._policy,
      self._last_checkpoint_at,
      current_time,
      self._inserts_since_checkpoint,
    ):
      self.checkpoint()

  def might_contain(self, member: bytes) -> bool:
    """Test whether a member may have been previously inserted into its routed shard.

    Args:
      member: The raw bytes of the element being tested.

    Returns:
      False if the member is certainly absent from its routed shard.
      True if the member is probably present.
    """
    index = shard_index(member, len(self._shards))

    return self._shards[index].might_contain(member)

  def __contains__(self, member: object) -> bool:
    """Enable idiomatic membership testing through the in operator.

    Args:
      member: The candidate object being tested for membership.

    Returns:
      False if the member is certainly absent. True if the member is
      probably present.

    Raises:
      TypeError: If member is not an instance of bytes.
    """
    if not isinstance(member, bytes):
      raise TypeError(
        f"ShardedBloomFilter membership testing requires bytes, received {type(member).__name__}"
      )

    return self.might_contain(member)

  def checkpoint(self) -> None:
    """Persist every shard modified since the previous checkpoint, and no others.

    This is the specific efficiency guarantee sharding exists to
    provide, a checkpoint's cost scales with how much activity
    occurred since the last one, not with the total size of the
    filter.
    """
    for index in self._dirty_shard_indices:
      save(self._shards[index], self._shard_path(self._directory, index))

    self._dirty_shard_indices.clear()
    self._last_checkpoint_at = self._clock()
    self._inserts_since_checkpoint = 0

  def close(self) -> None:
    """Perform a final checkpoint of any remaining dirty shards, if configured to."""
    if self._policy.checkpoint_on_shutdown:
      self.checkpoint()

  def __enter__(self) -> ShardedBloomFilter:
    """Support use as a context manager, guaranteeing close on exit."""
    return self

  def __exit__(self, *exc_info: object) -> None:
    """Invoke close automatically when the context manager block exits."""
    self.close()

  def __repr__(self) -> str:
    """Return an unambiguous representation for debugging and logging."""
    return (
      f"ShardedBloomFilter(shard_count={len(self._shards)!r}, "
      f"dirty_shard_count={len(self._dirty_shard_indices)!r})"
    )
