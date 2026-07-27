"""Unit tests for the named filter collection in vstg.registry.

These tests validate that BloomFilterRegistry correctly registers both
plain and sharded filters under independent names, that each named
filter genuinely operates in isolation from every other, that
duplicate registration is rejected, and that the checkpoint_all,
close_all, and shorthand accessor methods delegate correctly to the
underlying filters without the caller needing to hold a direct
reference to any of them.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vstg.managed import ManagedBloomFilter
from vstg.persist import PersistPolicy
from vstg.registry import BloomFilterRegistry
from vstg.shard import ShardedBloomFilter


class RegistryConstructionTests(unittest.TestCase):
  """Validate registry construction and its handling of the base directory."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._base_directory = Path(self._temporary_directory.name) / "bloom_state"

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_base_directory_is_created_when_absent(self) -> None:
    """The base directory itself should not need to exist beforehand."""
    self.assertFalse(self._base_directory.exists())

    BloomFilterRegistry(self._base_directory)

    self.assertTrue(self._base_directory.exists())

  def test_repr_reflects_an_empty_registry_before_any_registration(self) -> None:
    """A freshly constructed registry must report no registered names."""
    registry = BloomFilterRegistry(self._base_directory)

    self.assertIn("names=[]", repr(registry))


class RegisterTests(unittest.TestCase):
  """Validate the registration of both plain and sharded named filters."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and registry for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._registry = BloomFilterRegistry(Path(self._temporary_directory.name) / "bloom_state")

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_register_without_shard_count_creates_a_managed_filter(self) -> None:
    """Omitting shard_count must register a single-file ManagedBloomFilter."""
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)

    self.assertIsInstance(self._registry.get("usernames"), ManagedBloomFilter)

  def test_register_with_shard_count_creates_a_sharded_filter(self) -> None:
    """Providing shard_count must register a ShardedBloomFilter instead."""
    self._registry.register("email_tokens", capacity=1_000, error_rate=0.01, shard_count=4)

    self.assertIsInstance(self._registry.get("email_tokens"), ShardedBloomFilter)

  def test_register_rejects_a_duplicate_name(self) -> None:
    """Registering the same name twice must fail rather than silently replace the first."""
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)

    with self.assertRaises(ValueError):
      self._registry.register("usernames", capacity=2_000, error_rate=0.02)

  def test_distinct_names_may_carry_independent_capacity_and_error_rate(self) -> None:
    """Two named filters must retain the specific configuration each was registered with."""
    self._registry.register("usernames", capacity=10_000, error_rate=0.001)
    self._registry.register("email_tokens", capacity=500, error_rate=0.05)

    usernames_filter = self._registry.get("usernames")
    email_tokens_filter = self._registry.get("email_tokens")

    assert isinstance(usernames_filter, ManagedBloomFilter)
    assert isinstance(email_tokens_filter, ManagedBloomFilter)

    self.assertEqual(usernames_filter.bloom_filter.capacity, 10_000)
    self.assertEqual(usernames_filter.bloom_filter.error_rate, 0.001)
    self.assertEqual(email_tokens_filter.bloom_filter.capacity, 500)
    self.assertEqual(email_tokens_filter.bloom_filter.error_rate, 0.05)

  def test_register_defaults_to_a_policy_relying_on_shutdown_alone(self) -> None:
    """Omitting policy must not silently disable checkpoint_on_shutdown."""
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)
    self._registry.add("usernames", b"a-member")

    self._registry.close_all()

    checkpoint_path = Path(self._temporary_directory.name) / "bloom_state" / "usernames.bloom"
    self.assertTrue(checkpoint_path.exists())


class GetTests(unittest.TestCase):
  """Validate retrieval of registered filters by name, including the failure path."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and registry for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._registry = BloomFilterRegistry(Path(self._temporary_directory.name) / "bloom_state")

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_get_returns_the_exact_instance_created_by_register(self) -> None:
    """Retrieving a filter twice by the same name must return the same live instance."""
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)

    self.assertIs(self._registry.get("usernames"), self._registry.get("usernames"))

  def test_get_raises_key_error_for_an_unregistered_name(self) -> None:
    """Requesting a name that was never registered must fail loudly, not return None."""
    with self.assertRaises(KeyError):
      self._registry.get("never-registered")

  def test_key_error_message_lists_the_names_that_are_actually_available(self) -> None:
    """The failure message must help the caller notice a typo or a missing registration."""
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)
    self._registry.register("email_tokens", capacity=1_000, error_rate=0.01)

    with self.assertRaises(KeyError) as raised:
      self._registry.get("typo_ed_name")

    message = str(raised.exception)
    self.assertIn("usernames", message)
    self.assertIn("email_tokens", message)


class ShorthandTests(unittest.TestCase):
  """Validate the might_contain and add shorthand methods against the get-based equivalent."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and registry for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._registry = BloomFilterRegistry(Path(self._temporary_directory.name) / "bloom_state")
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_add_shorthand_is_reflected_by_might_contain_shorthand(self) -> None:
    """Inserting through the shorthand must be visible through the shorthand check."""
    self._registry.add("usernames", b"registered-member")

    self.assertTrue(self._registry.might_contain("usernames", b"registered-member"))

  def test_might_contain_shorthand_reports_absence_for_a_never_inserted_member(self) -> None:
    """A member never inserted through either path must report as certainly absent."""
    self.assertFalse(self._registry.might_contain("usernames", b"never-inserted-member"))

  def test_distinct_named_filters_do_not_cross_contaminate(self) -> None:
    """A member inserted into one named filter must not appear present in a different one."""
    self._registry.register("email_tokens", capacity=1_000, error_rate=0.01)
    self._registry.add("usernames", b"shared-looking-member")

    self.assertTrue(self._registry.might_contain("usernames", b"shared-looking-member"))
    self.assertFalse(self._registry.might_contain("email_tokens", b"shared-looking-member"))


class CheckpointAllTests(unittest.TestCase):
  """Validate that checkpoint_all persists every registered filter, plain and sharded alike."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and registry for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._base_directory = Path(self._temporary_directory.name) / "bloom_state"
    self._registry = BloomFilterRegistry(self._base_directory)

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_checkpoint_all_writes_both_a_managed_and_a_sharded_filter(self) -> None:
    """A single checkpoint_all call must persist every registered filter, regardless of kind."""
    self._registry.register("usernames", capacity=1_000, error_rate=0.01)
    self._registry.register("email_tokens", capacity=1_000, error_rate=0.01, shard_count=4)
    self._registry.add("usernames", b"a-member")
    self._registry.add("email_tokens", b"another-member")

    self._registry.checkpoint_all()

    self.assertTrue((self._base_directory / "usernames.bloom").exists())
    self.assertNotEqual(list((self._base_directory / "email_tokens").iterdir()), [])


class CloseAllTests(unittest.TestCase):
  """Validate that close_all respects each filter's own checkpoint_on_shutdown setting."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and registry for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._base_directory = Path(self._temporary_directory.name) / "bloom_state"
    self._registry = BloomFilterRegistry(self._base_directory)

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_close_all_writes_a_filter_configured_to_checkpoint_on_shutdown(self) -> None:
    """A filter registered with checkpoint_on_shutdown enabled must be persisted by close_all."""
    self._registry.register(
      "usernames", capacity=1_000, error_rate=0.01, policy=PersistPolicy(checkpoint_on_shutdown=True)
    )
    self._registry.add("usernames", b"a-member")

    self._registry.close_all()

    self.assertTrue((self._base_directory / "usernames.bloom").exists())

  def test_close_all_skips_a_filter_configured_not_to_checkpoint_on_shutdown(self) -> None:
    """A filter registered with checkpoint_on_shutdown disabled must be left unsaved by close_all."""
    self._registry.register(
      "usernames", capacity=1_000, error_rate=0.01, policy=PersistPolicy(checkpoint_on_shutdown=False)
    )
    self._registry.add("usernames", b"a-member")

    self._registry.close_all()

    self.assertFalse((self._base_directory / "usernames.bloom").exists())

  def test_close_all_treats_each_registered_filter_independently(self) -> None:
    """One filter's shutdown policy must not influence whether another filter gets persisted."""
    self._registry.register(
      "usernames", capacity=1_000, error_rate=0.01, policy=PersistPolicy(checkpoint_on_shutdown=True)
    )
    self._registry.register(
      "email_tokens",
      capacity=1_000,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=False),
    )
    self._registry.add("usernames", b"a-member")
    self._registry.add("email_tokens", b"another-member")

    self._registry.close_all()

    self.assertTrue((self._base_directory / "usernames.bloom").exists())
    self.assertFalse((self._base_directory / "email_tokens.bloom").exists())


class ContextManagerTests(unittest.TestCase):
  """Validate that the registry supports use as a context manager."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._base_directory = Path(self._temporary_directory.name) / "bloom_state"

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_context_manager_closes_all_on_exit(self) -> None:
    """Exiting a with block must invoke close_all automatically."""
    with BloomFilterRegistry(self._base_directory) as registry:
      registry.register(
        "usernames", capacity=1_000, error_rate=0.01, policy=PersistPolicy(checkpoint_on_shutdown=True)
      )
      registry.add("usernames", b"a-member")

    self.assertTrue((self._base_directory / "usernames.bloom").exists())


class ReprTests(unittest.TestCase):
  """Validate that the debugging representation surfaces registered names sorted and readably."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and registry for each test."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._registry = BloomFilterRegistry(Path(self._temporary_directory.name) / "bloom_state")

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_repr_lists_registered_names_in_sorted_order(self) -> None:
    """Names must appear sorted, regardless of the order they were registered in."""
    self._registry.register("zebra_filter", capacity=1_000, error_rate=0.01)
    self._registry.register("apple_filter", capacity=1_000, error_rate=0.01)

    representation = repr(self._registry)

    self.assertIn("'apple_filter', 'zebra_filter'", representation)


if __name__ == "__main__":
  unittest.main()
