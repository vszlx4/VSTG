"""Automatic checkpoint orchestration binding a live filter to its policy.

The prior modules deliberately stop short of connecting a running filter
to its persistence rules: filter.py knows nothing about disk, and
persist.py's should_checkpoint is a pure decision that nobody calls on
its own. This module is where those pieces are wired together into
something a caller can actually operate without manually tracking
elapsed time and insertion counts themselves. A ManagedBloomFilter is
the only entry point most integrations should need, insert members
through it, and checkpointing happens transparently whenever the
configured policy determines it is due.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from vstg.filter import BloomFilter
from vstg.persist import PersistPolicy, load, save, should_checkpoint


class ManagedBloomFilter:
  """A BloomFilter paired with the state needed to checkpoint it automatically.

  This class owns three things a bare BloomFilter does not: the
  destination path a checkpoint should be written to, the policy
  governing when that should happen, and the bookkeeping, elapsed
  time and accumulated insertions since the last checkpoint, required
  to evaluate that policy on every insertion without the caller having
  to track any of it themselves.

  Instances are not safe for concurrent use from multiple threads
  without external synchronization, for the same reason the underlying
  BloomFilter is not: bit flips and checkpoint bookkeeping are not
  atomic against one another.
  """

  def __init__(
    self,
    bloom_filter: BloomFilter,
    destination: Path,
    policy: PersistPolicy,
    clock: Callable[[], float] = time.monotonic,
  ) -> None:
    """Bind an existing filter to a destination path and checkpoint policy.

    Most callers should prefer the open classmethod over calling this
    constructor directly, since open additionally handles resuming
    from an existing checkpoint on disk. This constructor is exposed
    separately for callers who have already constructed or restored a
    BloomFilter through some other path and merely need to attach
    checkpoint management to it.

    Args:
      bloom_filter: The filter instance to manage.
      destination: The file path automatic and manual checkpoints
                   should be written to.
      policy: The rules governing when an automatic checkpoint occurs.
      clock: A zero argument callable returning the current time, in
             seconds. Defaults to time.monotonic rather than time.time,
             since checkpoint timing depends only on elapsed duration and
             must not be disturbed by system clock adjustments. Tests may
             substitute a deterministic callable in its place.
    """
    self._bloom_filter = bloom_filter
    self._destination = destination
    self._policy = policy
    self._clock = clock
    self._last_checkpoint_at = clock()
    self._inserts_since_checkpoint = 0

  @classmethod
  def open(
    cls,
    destination: Path,
    capacity: int,
    error_rate: float,
    policy: PersistPolicy,
    clock: Callable[[], float] = time.monotonic,
  ) -> ManagedBloomFilter:
    """Resume from an existing checkpoint, or begin fresh if none exists.

    This is the intended entry point for the common case: a backend
    process starting up, which should transparently inherit whatever
    state a previous process instance had checkpointed, without the
    caller having to write the existence check themselves.

    Args:
      destination: The file path a prior checkpoint may already occupy,
                   and future checkpoints will be written to.
      capacity: The member capacity to construct a fresh filter with, if
                no checkpoint is found. Ignored if a checkpoint is loaded,
                since the restored filter carries its own original capacity.
      error_rate: The false positive rate to construct a fresh filter with,
                  if no checkpoint is found. Ignored if a checkpoint is
                  loaded, for the same reason as capacity.
      policy: The rules governing when an automatic checkpoint occurs.
      clock: A zero argument callable returning the current time, in
             seconds.

    Returns:
      A fully initialized ManagedBloomFilter, either resumed from disk
      or freshly allocated.
    """
    if destination.exists():  # noqa: SIM108 (ternary would exceed the line-length limit)
      bloom_filter = load(destination)
    else:
      bloom_filter = BloomFilter(capacity, error_rate)

    return cls(bloom_filter, destination, policy, clock)

  @property
  def bloom_filter(self) -> BloomFilter:
    """The underlying filter instance being managed."""
    return self._bloom_filter

  def might_contain(self, member: bytes) -> bool:
    """Test whether a member may have been previously inserted.

    Delegates directly to the underlying filter, preserving its
    asymmetric correctness guarantee unchanged.

    Args:
      member: The raw bytes of the element being tested.

    Returns:
      False if the member is certainly absent. True if the member is
      probably present.
    """
    return self._bloom_filter.might_contain(member)

  def __contains__(self, member: object) -> bool:
    """Enable idiomatic membership testing through the in operator.

    Unlike might_contain, this method accepts an object of any type,
    matching the signature the language itself requires of anything
    implementing the in operator. A non-bytes argument is rejected
    explicitly here, rather than allowed to fail unpredictably deeper
    inside the underlying filter.

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
        f"ManagedBloomFilter membership testing requires bytes, received {type(member).__name__}"
      )

    return self.might_contain(member)

  def add(self, member: bytes) -> None:
    """Insert a member, then checkpoint automatically if the policy demands it.

    This is the sole method most integrations need to call on the
    write path. The bookkeeping the constructor initialized is updated
    here on every call, and should_checkpoint is consulted immediately
    afterward, the caller never evaluates the policy directly.

    Args:
      member: The raw bytes of the element being inserted.
    """
    self._bloom_filter.add(member)
    self._inserts_since_checkpoint += 1

    current_time = self._clock()

    if should_checkpoint(
      self._policy,
      self._last_checkpoint_at,
      current_time,
      self._inserts_since_checkpoint,
    ):
      self.checkpoint()

  def checkpoint(self) -> None:
    """Force an immediate checkpoint, bypassing the policy's own timing.

    Automatic checkpoints call this internally once should_checkpoint
    returns True, but it is equally valid for a caller to invoke this
    directly, for instance, from an operator command or a health
    endpoint, without waiting for the policy's own conditions to be
    met.
    """
    save(self._bloom_filter, self._destination)
    self._last_checkpoint_at = self._clock()
    self._inserts_since_checkpoint = 0

  def close(self) -> None:
    """Perform a final checkpoint on orderly shutdown, if configured to.

    Whether this actually writes anything depends entirely on the
    policy's checkpoint_on_shutdown flag, allowing a caller who has
    already arranged their own shutdown checkpointing to disable this
    behavior without changing how the rest of the class is used.
    """
    if self._policy.checkpoint_on_shutdown:
      self.checkpoint()

  def __enter__(self) -> ManagedBloomFilter:
    """Support use as a context manager, guaranteeing close on exit."""
    return self

  def __exit__(self, *exc_info: object) -> None:
    """Invoke close automatically when the context manager block exits."""
    self.close()
