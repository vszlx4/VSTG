"""Unit tests for the top level convenience API in vstg's package init.

These tests validate that init correctly configures the default
registry exactly once, that every convenience function fails with a
clear error before init has been called, and that register,
might_contain, add, checkpoint_all, and close_all all delegate
correctly once a registry is in place. Since the default registry is
process level global state, _reset_for_testing is invoked before and
after every test to guarantee no state leaks between them.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import vstg
from vstg import _reset_for_testing


class InitTests(unittest.TestCase):
  """Validate the configuration and one time guard behavior of init."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory and reset the default registry."""
    _reset_for_testing()
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._base_directory = Path(self._temporary_directory.name) / "bloom_state"

  def tearDown(self) -> None:
    """Reset the default registry and release the temporary directory."""
    _reset_for_testing()
    self._temporary_directory.cleanup()

  def test_init_creates_the_base_directory(self) -> None:
    """The base directory itself should not need to exist beforehand."""
    self.assertFalse(self._base_directory.exists())

    vstg.init(self._base_directory)

    self.assertTrue(self._base_directory.exists())

  def test_init_called_twice_raises_runtime_error(self) -> None:
    """A second call to init within the same process must be rejected."""
    vstg.init(self._base_directory)

    with self.assertRaises(RuntimeError):
      vstg.init(self._base_directory)


class UninitializedGuardTests(unittest.TestCase):
  """Validate that every convenience function fails clearly before init is called."""

  def setUp(self) -> None:
    """Reset the default registry so each test starts genuinely uninitialized."""
    _reset_for_testing()

  def tearDown(self) -> None:
    """Reset the default registry after each test."""
    _reset_for_testing()

  def test_register_raises_before_init(self) -> None:
    """Registering a filter before init must fail rather than register against nothing."""
    with self.assertRaises(RuntimeError):
      vstg.register("usernames", capacity=1_000, error_rate=0.01)

  def test_might_contain_raises_before_init(self) -> None:
    """Checking membership before init must fail rather than silently return False."""
    with self.assertRaises(RuntimeError):
      vstg.might_contain("usernames", b"a-member")

  def test_add_raises_before_init(self) -> None:
    """Inserting a member before init must fail rather than silently discard it."""
    with self.assertRaises(RuntimeError):
      vstg.add("usernames", b"a-member")

  def test_checkpoint_all_raises_before_init(self) -> None:
    """Checkpointing before init must fail rather than silently do nothing."""
    with self.assertRaises(RuntimeError):
      vstg.checkpoint_all()

  def test_close_all_raises_before_init(self) -> None:
    """Closing before init must fail rather than silently do nothing."""
    with self.assertRaises(RuntimeError):
      vstg.close_all()


class ConvenienceFunctionTests(unittest.TestCase):
  """Validate register, might_contain, add, checkpoint_all, and close_all once initialized."""

  def setUp(self) -> None:
    """Reset the default registry, initialize a fresh one, and register a working filter."""
    _reset_for_testing()
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._base_directory = Path(self._temporary_directory.name) / "bloom_state"
    vstg.init(self._base_directory)

  def tearDown(self) -> None:
    """Reset the default registry and release the temporary directory."""
    _reset_for_testing()
    self._temporary_directory.cleanup()

  def test_add_and_might_contain_round_trip(self) -> None:
    """A member inserted through add must be reported present through might_contain."""
    vstg.register("usernames", capacity=1_000, error_rate=0.01)
    vstg.add("usernames", b"registered-member")

    self.assertTrue(vstg.might_contain("usernames", b"registered-member"))

  def test_might_contain_reports_absence_for_a_never_inserted_member(self) -> None:
    """A member never inserted must always report as certainly absent."""
    vstg.register("usernames", capacity=1_000, error_rate=0.01)

    self.assertFalse(vstg.might_contain("usernames", b"never-inserted-member"))

  def test_might_contain_raises_key_error_for_an_unregistered_name(self) -> None:
    """Checking a never registered name must fail loudly, matching the registry's behavior."""
    with self.assertRaises(KeyError):
      vstg.might_contain("never-registered", b"a-member")

  def test_register_accepts_shard_count_for_a_sharded_filter(self) -> None:
    """The shard_count parameter must pass through to the underlying registry correctly."""
    vstg.register("email_tokens", capacity=1_000, error_rate=0.01, shard_count=4)
    vstg.add("email_tokens", b"a-token")

    self.assertTrue(vstg.might_contain("email_tokens", b"a-token"))

  def test_checkpoint_all_persists_a_registered_filter(self) -> None:
    """checkpoint_all must write every registered filter to disk immediately."""
    vstg.register("usernames", capacity=1_000, error_rate=0.01)
    vstg.add("usernames", b"a-member")

    vstg.checkpoint_all()

    self.assertTrue((self._base_directory / "usernames.bloom").exists())

  def test_close_all_persists_a_filter_configured_to_checkpoint_on_shutdown(self) -> None:
    """close_all must persist a filter whose policy enables checkpoint_on_shutdown."""
    vstg.register(
      "usernames",
      capacity=1_000,
      error_rate=0.01,
      policy=vstg.PersistPolicy(checkpoint_on_shutdown=True),
    )
    vstg.add("usernames", b"a-member")

    vstg.close_all()

    self.assertTrue((self._base_directory / "usernames.bloom").exists())


class PublicApiSurfaceTests(unittest.TestCase):
  """Validate that the module exposes exactly the intended public surface."""

  def test_all_expected_names_are_exported(self) -> None:
    """Every class and function meant for public use must appear in __all__."""
    expected_names = {
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
    }

    self.assertEqual(set(vstg.__all__), expected_names)

  def test_private_reset_helper_is_not_exported(self) -> None:
    """The testing only reset helper must remain absent from the public surface."""
    self.assertNotIn("_reset_for_testing", vstg.__all__)


if __name__ == "__main__":
  unittest.main()
