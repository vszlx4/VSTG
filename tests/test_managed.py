"""Unit tests for the checkpoint orchestration in vstg.managed.

These tests validate that ManagedBloomFilter correctly resumes from an
existing checkpoint or begins fresh, that automatic checkpointing fires
precisely when the configured policy demands it and not before, and
that manual checkpointing and shutdown behavior operate independently
of the automatic path. A deterministic fake clock is used throughout
rather than the real system clock, since time based checkpoint logic
tested against actual elapsed wall time would be slow and flaky.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vstg.filter import BloomFilter
from vstg.managed import ManagedBloomFilter
from vstg.persist import PersistPolicy


class _FakeClock:
  """A deterministic, manually advanced stand in for a wall clock, used only in tests.

  Production code always receives time.monotonic as its clock, which
  cannot be advanced or rewound on demand. This substitute exists
  solely so that interval based checkpoint behavior can be tested by
  advancing simulated time explicitly, rather than by waiting on real
  elapsed seconds.
  """

  def __init__(self, initial_time: float = 0.0) -> None:
    """Initialize the fake clock at a given starting time, in seconds."""
    self._current_time = initial_time

  def advance(self, seconds: float) -> None:
    """Move the fake clock forward by the given number of seconds."""
    self._current_time += seconds

  def __call__(self) -> float:
    """Return the current fake time, satisfying the clock callable protocol."""
    return self._current_time


class ManagedBloomFilterTests(unittest.TestCase):
  """Validate ManagedBloomFilter against a temporary directory and a fake clock."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory for each test to write checkpoints into."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._checkpoint_path = Path(self._temporary_directory.name) / "checkpoint.bloom"

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_open_creates_a_fresh_filter_when_no_checkpoint_exists(self) -> None:
    """With no prior checkpoint on disk, open must allocate a new, empty filter."""
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertFalse(managed_filter.might_contain(b"anything"))

  def test_open_resumes_from_an_existing_checkpoint(self) -> None:
    """A prior checkpoint on disk must be loaded rather than discarded in favor of a fresh filter."""
    original_managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path,
      capacity=100,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=True),
    )
    original_managed_filter.add(b"member-before-restart")
    original_managed_filter.close()

    resumed_managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertTrue(resumed_managed_filter.might_contain(b"member-before-restart"))

  def test_add_triggers_automatic_checkpoint_at_insert_threshold(self) -> None:
    """Reaching the configured insert threshold must write a checkpoint without a manual call."""
    policy = PersistPolicy(checkpoint_insert_threshold=3)
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=policy, clock=_FakeClock()
    )

    managed_filter.add(b"first-member")
    managed_filter.add(b"second-member")
    self.assertFalse(self._checkpoint_path.exists())

    managed_filter.add(b"third-member")
    self.assertTrue(self._checkpoint_path.exists())

  def test_add_triggers_automatic_checkpoint_after_configured_interval(self) -> None:
    """Elapsing past the configured interval must write a checkpoint on the next insertion."""
    fake_clock = _FakeClock()
    policy = PersistPolicy(checkpoint_interval_seconds=60)
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=policy, clock=fake_clock
    )

    managed_filter.add(b"first-member")
    self.assertFalse(self._checkpoint_path.exists())

    fake_clock.advance(61)
    managed_filter.add(b"second-member")
    self.assertTrue(self._checkpoint_path.exists())

  def test_checkpoint_forces_an_immediate_save_regardless_of_policy(self) -> None:
    """A manual checkpoint call must write to disk even when no automatic trigger has fired."""
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=PersistPolicy()
    )
    managed_filter.add(b"unsaved-member")
    self.assertFalse(self._checkpoint_path.exists())

    managed_filter.checkpoint()

    self.assertTrue(self._checkpoint_path.exists())

  def test_close_writes_a_final_checkpoint_when_configured_to(self) -> None:
    """With checkpoint_on_shutdown enabled, close must write a checkpoint unconditionally."""
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path,
      capacity=100,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=True),
    )
    managed_filter.add(b"final-member")

    managed_filter.close()

    self.assertTrue(self._checkpoint_path.exists())

  def test_close_does_not_write_when_configured_not_to(self) -> None:
    """With checkpoint_on_shutdown disabled, close must not write a checkpoint at all."""
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path,
      capacity=100,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=False),
    )
    managed_filter.add(b"unsaved-member")

    managed_filter.close()

    self.assertFalse(self._checkpoint_path.exists())

  def test_context_manager_closes_on_exit(self) -> None:
    """Exiting a with block must invoke close automatically, writing a final checkpoint."""
    policy = PersistPolicy(checkpoint_on_shutdown=True)

    with ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=policy
    ) as managed_filter:
      managed_filter.add(b"member-inside-context")

    self.assertTrue(self._checkpoint_path.exists())

  def test_might_contain_and_in_operator_reflect_insertions(self) -> None:
    """Both might_contain and the in operator must agree on what has actually been inserted."""
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=PersistPolicy()
    )
    managed_filter.add(b"present-member")

    self.assertTrue(managed_filter.might_contain(b"present-member"))
    self.assertIn(b"present-member", managed_filter)
    self.assertNotIn(b"absent-member", managed_filter)

  def test_bloom_filter_property_exposes_the_underlying_instance(self) -> None:
    """The bloom_filter property must expose a genuine BloomFilter instance, not a copy or proxy."""
    managed_filter = ManagedBloomFilter.open(
      self._checkpoint_path, capacity=100, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertIsInstance(managed_filter.bloom_filter, BloomFilter)


if __name__ == "__main__":
  unittest.main()
