"""Top level convenience API for the vstg bloom filter package.

Every prior module in this package exposes composable primitives
meant to be assembled deliberately: BloomFilter for a single filter,
ManagedBloomFilter for automatic checkpointing, ShardedBloomFilter for
partitioned persistence, and BloomFilterRegistry for holding several
named filters in one place. This module exists for the common case
that does not need any of that spelled out explicitly at the call
site: configure a set of named filters once, during application
startup, then check and insert against them from anywhere else in the
codebase through two plain functions. A caller writing
vstg.might_contain("usernames", member) before a database query
should never need to know a registry, a policy, or a checkpoint
exists underneath it.

The class based API remains fully available and unaffected by this
module for callers who want direct control over multiple independent
registries, custom clocks, or anything this convenience layer does
not anticipate.
"""

from __future__ import annotations

from pathlib import Path

from vstg.filter import BloomFilter
from vstg.managed import ManagedBloomFilter
from vstg.persist import PersistPolicy
from vstg.registry import BloomFilterRegistry
from vstg.shard import ShardedBloomFilter

__all__ = [
  "BloomFilter",
  "ManagedBloomFilter",
  "ShardedBloomFilter",
  "BloomFilterRegistry",
  "PersistPolicy",
  "init",
  "register",
  "might_contain",
  "add",
  "checkpoint_all",
  "close_all",
]


_registry: BloomFilterRegistry | None = None


def init(directory: Path) -> None:
  """Configure the default registry used by every module level convenience function.

  Intended to be called exactly once, during application startup,
  before any call to register, might_contain, or add. Calling it a
  second time within the same process is treated as a configuration
  error rather than silently replacing the existing registry, since a
  registry being swapped out mid run would invalidate whatever a
  caller had already registered against it, and the policies this
  package establishes are meant to be fixed at launch, not mutated
  during operation.

  Args:
    directory: The base directory every filter registered against the
               default registry will checkpoint beneath.

  Raises:
    RuntimeError: If init has already been called in this process.
  """
  global _registry

  if _registry is not None:
    raise RuntimeError(
      "vstg.init has already been called in this process; the default "
      "registry is configured exactly once at startup"
    )

  _registry = BloomFilterRegistry(directory)


def _require_registry() -> BloomFilterRegistry:
  """Return the configured default registry, or raise a clear error if init was never called.

  Centralizing this check in one place ensures every convenience
  function in this module fails with the identical, actionable
  message rather than each one independently guessing how to explain
  a missing registry.

  Returns:
    The default registry configured by init.

  Raises:
    RuntimeError: If init has not yet been called in this process.
  """
  if _registry is None:
    raise RuntimeError(
      "vstg has not been initialized; call vstg.init(directory) once "
      "during application startup before using this function"
    )

  return _registry


def register(
  name: str,
  capacity: int,
  error_rate: float,
  policy: PersistPolicy | None = None,
  shard_count: int | None = None,
) -> None:
  """Register a named filter against the default registry.

  A thin delegation to BloomFilterRegistry.register. See that method
  for the full description of each parameter.

  Args:
    name: A unique identifier this filter will be retrieved under.
    capacity: The number of members this filter is expected to hold.
    error_rate: The false positive probability this filter should
                honor.
    policy: The checkpoint policy this filter should follow. Defaults
            to a policy relying on checkpoint_on_shutdown alone if left
            unspecified.
    shard_count: If provided, partitions the filter across this many
                 shard files instead of a single checkpoint file.

  Raises:
    RuntimeError: If init has not yet been called.
    ValueError: If name has already been registered.
  """
  _require_registry().register(name, capacity, error_rate, policy, shard_count)


def might_contain(name: str, member: bytes) -> bool:
  """Test membership against a named filter registered with vstg.register.

  Args:
    name: The identifier the filter was registered under.
    member: The raw bytes of the element being tested.

  Returns:
    False if the member is certainly absent. True if the member is
    probably present.

  Raises:
    RuntimeError: If init has not yet been called.
    KeyError: If name was never registered.
  """
  return _require_registry().might_contain(name, member)


def add(name: str, member: bytes) -> None:
  """Insert a member into a named filter registered with vstg.register.

  Args:
    name: The identifier the filter was registered under.
    member: The raw bytes of the element being inserted.

  Raises:
    RuntimeError: If init has not yet been called.
    KeyError: If name was never registered.
  """
  _require_registry().add(name, member)


def checkpoint_all() -> None:
  """Force an immediate checkpoint of every filter in the default registry.

  Raises:
    RuntimeError: If init has not yet been called.
  """
  _require_registry().checkpoint_all()


def close_all() -> None:
  """Perform a final checkpoint of every filter in the default registry, per each one's own policy.

  Intended to be called once, during an orderly application shutdown.

  Raises:
    RuntimeError: If init has not yet been called.
  """
  _require_registry().close_all()


def _reset_for_testing() -> None:
  """Clear the default registry, permitting init to be called again.

  This exists solely to let the test suite exercise init's guard
  behavior and its success path repeatedly within the same process,
  without each test needing to run in a separate interpreter. It is
  not part of the public API and must never be called from
  application code, calling it while a registry holds unsaved state
  discards that state without checkpointing it first.
  """
  global _registry

  _registry = None
