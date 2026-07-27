"""Named collection of independently configured bloom filters.

A single application commonly needs more than one bloom filter
serving entirely different purposes, one guarding username
availability, another guarding email verification tokens, a third
guarding payment idempotency keys, each with its own capacity and its
own tolerated error rate, since the cost of a false positive differs
depending on what the filter is protecting. This module lets every
such filter be configured once, by name, at process startup, and
retrieved from anywhere else in the codebase without threading filter
instances through every layer of a call stack by hand.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from vstg.managed import ManagedBloomFilter
from vstg.persist import PersistPolicy
from vstg.shard import ShardedBloomFilter


class PersistentBloomFilter(Protocol):
  """The common interface shared by ManagedBloomFilter and ShardedBloomFilter.

  BloomFilterRegistry holds filters polymorphically through this
  protocol rather than a concrete class, since a registered filter may
  be backed by either a single checkpoint file or a directory of
  shards, and the registry itself has no need to distinguish between
  the two once a filter is registered. Both concrete classes already
  satisfy this protocol structurally, without inheriting from it
  explicitly.
  """

  def might_contain(self, member: bytes) -> bool:
    """Test whether a member may have been previously inserted."""
    ...

  def add(self, member: bytes) -> None:
    """Insert a member, checkpointing automatically if its policy demands it."""
    ...

  def checkpoint(self) -> None:
    """Force an immediate checkpoint, bypassing the policy's own timing."""
    ...

  def close(self) -> None:
    """Perform a final checkpoint on orderly shutdown, if configured to."""
    ...


class BloomFilterRegistry:
  """A named collection of independently configured bloom filters.
  
  Every filter registered here checkpoints to a location beneath a
  single base directory, keeping an application's entire bloom filter
  state discoverable in one place on disk rather than scattered
  across wherever each individual filter happened to be constructed.

  Registration is intended to happen exactly once per name, during
  application startup. Nothing in this class prevents calling
  register from elsewhere, but doing so defeats the guarantee that a
  filter's capacity, error rate, and policy are fixed for the entire
  lifetime of the process, which is the property that makes automatic
  checkpointing safe to reason about in the first place.
  """

  def __init__(self, directory: Path) -> None:
    """Create a registry rooted at the given base directory.
    
    The directory is created immediately if it does not already
    exist, so that registration can proceed without every caller
    separately handling its absence.

    Args:
      directory: The base directory every registered filter's checkpoint
                 file or shard subdirectory will reside beneath.
    """
    directory.mkdir(parents=True, exist_ok=True)
    self._directory = directory
    self._filters: dict[str, PersistentBloomFilter] = {}

  def register(
      self,
      name: str,
      capacity: int,
      error_rate: float,
      policy: PersistPolicy | None = None,
      shard_count: int | None = None,
      clock: Callable[[], float] = time.monotonic,
  ) -> None:
    """Configure and open a new named filter, resuming from disk if a checkpoint exists.

    Args:
      name: A unique identifier this filter will be retrieved under
            elsewhere in the codebase, such as "usernames" or
            "email_verification_tokens".
      capacity: The number of members this filter is expected to
                hold. Distinct named filters are free to specify entirely
                different capacities from one another.
      error_rate: The false positive probability this filter should honor. 
                  Distinct named filters are free to specify entirely different 
                  error rates from one another, reflecting that a false positive
                  protecting a username check and one protecting a payment
                  idempotency check do not carry equal cost.
      policy: The checkpoint policy this filter should follow.
              Defaults to a policy relying on checkpoint_on_shutdown alone,
              with no automatic time or count trigger, if left unspecified.
      shard_count: If provided, the filter is partitioned across this
                   many shard files rather than a single checkpoint file,
                   appropriate for filters expected to grow large enough that
                   checkpoint cost becomes a concern. Left as None, a single
                   ManagedBloomFilter backed by one file is registered instead.
      clock: A zero argument callable returning the current time, in
             seconds, used for policy timing. Tests may substitute a
             deterministic callable in its place.

    Raises:
      ValueError: If name has already been registered.
    """
    if name in self._filters:
      raise ValueError(
        f"a filter is already registered under name {name!r}; choose a "
        f"distinct name or unregister the existing one before replacing it"
      )

    effective_policy = policy if policy is not None else PersistPolicy()

    if shard_count is not None:
      self._filters[name] = ShardedBloomFilter.open(
        self._directory / name,
        shard_count,
        capacity,
        error_rate,
        effective_policy,
        clock,
      )
    else:
      self._filters[name] = ManagedBloomFilter.open(
        self._directory / f"{name}.bloom",
        capacity,
        error_rate,
        effective_policy,
        clock,
      )

  def get(self, name: str) -> PersistentBloomFilter:
    """Retrieve a previously registered filter by name.

    Args:
      name: The identifier the filter was registered under.

    Returns:
      The registered filter instance, either a ManagedBloomFilter or
      a ShardedBloomFilter depending on how it was registered.

    Raises:
      KeyError: If no filter has been registered under name. The error
                message lists the names that are actually available, since
                hitting this is almost always the result of a typo or a missing
                registration at startup, not a case a caller needs to handle
                gracefully at runtime.
    """
    try:
      return self._filters[name]
    except KeyError:
      raise KeyError(
        f"no filter registered under name {name!r}; registered names "
        f"are {sorted(self._filters)!r}"
      ) from None

  def might_contain(self, name: str, member: bytes) -> bool:
    """Test membership against a named filter without holding a reference to it.

    A convenience shorthand for get(name).might_contain(member).

    Args:
      name: The identifier the filter was registered under.
      member: The raw bytes of the element being tested.

    Returns:
      False if the member is certainly absent. True if the member is
      probably present.
    """
    return self.get(name).might_contain(member)

  def add(self, name: str, member: bytes) -> None:
    """Insert a member into a named filter without holding a reference to it.

    A convenience shorthand for get(name).add(member).

    Args:
      name: The identifier the filter was registered under.
      member: The raw bytes of the element being inserted.
    """
    self.get(name).add(member)

  def checkpoint_all(self) -> None:
    """Force an immediate checkpoint of every registered filter.

    Useful for an operator command or a health endpoint that wants to
    guarantee every filter's state is durable at a specific moment,
    independent of any individual filter's own policy timing.
    """
    for bloom_filter in self._filters.values():
      bloom_filter.checkpoint()

  def close_all(self) -> None:
    """Perform a final checkpoint of every registered filter, per each one's own policy.

    Intended to be called once, during an orderly application
    shutdown. Whether any individual filter actually writes anything
    depends entirely on its own checkpoint_on_shutdown setting.
    """
    for bloom_filter in self._filters.values():
      bloom_filter.close()

  def __enter__(self) -> BloomFilterRegistry:
    """Support use as a context manager, guaranteeing close_all on exit."""
    return self

  def __exit__(self, *exc_info: object) -> None:
    """Invoke close_all automatically when the context manager block exits."""
    self.close_all()

  def __repr__(self) -> str:
    """Return an unambiguous representation for debugging and logging."""
    return f"BloomFilterRegistry(directory={self._directory!r}, names={sorted(self._filters)!r})"
