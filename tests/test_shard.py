"""Unit tests for the sharded bloom filter in vstg.shard.

These tests validate that a ShardedBloomFilter behaves, from the
perspective of correctness, identically to a single BloomFilter, while
verifying the property that actually justifies sharding in the first
place: a checkpoint only ever writes the shard files that received an
insertion since the previous checkpoint, never the full set. A
deterministic fake clock stands in for the real system clock throughout,
for the same reason it does in test_managed.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vstg.persist import PersistPolicy
from vstg.shard import ShardedBloomFilter


class _FakeClock:
  """A deterministic, manually advanced stand in for a wall clock, used only in tests."""

  def __init__(self, initial_time: float = 0.0) -> None:
    """Initialize the fake clock at a given starting time, in seconds."""
    self._current_time = initial_time

  def advance(self, seconds: float) -> None:
    """Move the fake clock forward by the given number of seconds."""
    self._current_time += seconds

  def __call__(self) -> float:
    """Return the current fake time, satisfying the clock callable protocol."""
    return self._current_time


class ShardedBloomFilterTests(unittest.TestCase):
  """Validate ShardedBloomFilter against a temporary directory and a fake clock."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory for each test to write shard files into."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._shard_directory = Path(self._temporary_directory.name) / "shards"

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_open_creates_the_shard_directory_when_absent(self) -> None:
    """The directory itself should not need to exist beforehand."""
    self.assertFalse(self._shard_directory.exists())

    ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertTrue(self._shard_directory.exists())

  def test_open_rejects_non_positive_shard_count(self) -> None:
    """A shard count of zero or below describes no meaningful partitioning."""
    with self.assertRaises(ValueError):
      ShardedBloomFilter.open(
        self._shard_directory, shard_count=0, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
      )

  def test_no_shard_files_exist_before_any_checkpoint(self) -> None:
    """A freshly opened filter, before any insertion or checkpoint, must write nothing to disk."""
    ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertEqual(list(self._shard_directory.iterdir()), [])

  def test_add_and_might_contain_agree_for_an_inserted_member(self) -> None:
    """Insertion followed immediately by a check must never report absence."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )
    sharded_filter.add(b"registered-member")

    self.assertTrue(sharded_filter.might_contain(b"registered-member"))

  def test_might_contain_reports_absence_for_a_never_inserted_member(self) -> None:
    """A member that was never added must always report as certainly absent."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertFalse(sharded_filter.might_contain(b"never-inserted-member"))

  def test_checkpoint_after_a_single_insertion_writes_exactly_one_shard_file(self) -> None:
    """A single insertion can only ever mark one shard dirty, regardless of shard count."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=8, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )
    sharded_filter.add(b"single-member")

    sharded_filter.checkpoint()

    written_files = list(self._shard_directory.iterdir())
    self.assertEqual(len(written_files), 1)

  def test_checkpoint_eventually_writes_every_shard_once_each_has_an_insertion(self) -> None:
    """With enough distinct members, every shard should eventually receive at least one insertion."""
    shard_count = 4
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory,
      shard_count=shard_count,
      capacity=1_000,
      error_rate=0.01,
      policy=PersistPolicy(),
    )

    for index in range(500):
      sharded_filter.add(f"member-{index}".encode("utf-8"))

    sharded_filter.checkpoint()

    written_files = list(self._shard_directory.iterdir())
    self.assertEqual(len(written_files), shard_count)

  def test_checkpoint_clears_dirty_tracking(self) -> None:
    """A checkpoint must reset the dirty count, since everything pending has now been saved."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )
    sharded_filter.add(b"first-member")
    self.assertEqual(sharded_filter.dirty_shard_count, 1)

    sharded_filter.checkpoint()

    self.assertEqual(sharded_filter.dirty_shard_count, 0)

  def test_dirty_shard_count_does_not_double_count_the_same_shard(self) -> None:
    """Repeated insertions routed to the same shard must not inflate the dirty count."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=1, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )
    sharded_filter.add(b"first-member")
    sharded_filter.add(b"second-member")
    sharded_filter.add(b"third-member")

    self.assertEqual(sharded_filter.dirty_shard_count, 1)

  def test_open_resumes_an_inserted_member_after_a_simulated_restart(self) -> None:
    """A member checkpointed before a restart must still be found by a freshly opened instance."""
    original_filter = ShardedBloomFilter.open(
      self._shard_directory,
      shard_count=4,
      capacity=1_000,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=True),
    )
    original_filter.add(b"member-before-restart")
    original_filter.close()

    resumed_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertTrue(resumed_filter.might_contain(b"member-before-restart"))

  def test_add_triggers_automatic_checkpoint_at_insert_threshold(self) -> None:
    """Reaching the configured insert threshold must write pending shards without a manual call."""
    policy = PersistPolicy(checkpoint_insert_threshold=3)
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory,
      shard_count=4,
      capacity=1_000,
      error_rate=0.01,
      policy=policy,
      clock=_FakeClock(),
    )

    sharded_filter.add(b"first-member")
    sharded_filter.add(b"second-member")
    self.assertEqual(list(self._shard_directory.iterdir()), [])

    sharded_filter.add(b"third-member")
    self.assertNotEqual(list(self._shard_directory.iterdir()), [])

  def test_add_triggers_automatic_checkpoint_after_configured_interval(self) -> None:
    """Elapsing past the configured interval must write pending shards on the next insertion."""
    fake_clock = _FakeClock()
    policy = PersistPolicy(checkpoint_interval_seconds=60)
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory,
      shard_count=4,
      capacity=1_000,
      error_rate=0.01,
      policy=policy,
      clock=fake_clock,
    )

    sharded_filter.add(b"first-member")
    self.assertEqual(list(self._shard_directory.iterdir()), [])

    fake_clock.advance(61)
    sharded_filter.add(b"second-member")
    self.assertNotEqual(list(self._shard_directory.iterdir()), [])

  def test_close_writes_pending_shards_when_configured_to(self) -> None:
    """With checkpoint_on_shutdown enabled, close must persist any remaining dirty shards."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory,
      shard_count=4,
      capacity=1_000,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=True),
    )
    sharded_filter.add(b"final-member")

    sharded_filter.close()

    self.assertNotEqual(list(self._shard_directory.iterdir()), [])

  def test_close_does_not_write_when_configured_not_to(self) -> None:
    """With checkpoint_on_shutdown disabled, close must leave dirty shards unsaved."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory,
      shard_count=4,
      capacity=1_000,
      error_rate=0.01,
      policy=PersistPolicy(checkpoint_on_shutdown=False),
    )
    sharded_filter.add(b"unsaved-member")

    sharded_filter.close()

    self.assertEqual(list(self._shard_directory.iterdir()), [])

  def test_context_manager_closes_on_exit(self) -> None:
    """Exiting a with block must invoke close automatically, writing pending shards."""
    policy = PersistPolicy(checkpoint_on_shutdown=True)

    with ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=policy
    ) as sharded_filter:
      sharded_filter.add(b"member-inside-context")

    self.assertNotEqual(list(self._shard_directory.iterdir()), [])

  def test_contains_operator_reflects_insertions(self) -> None:
    """Both might_contain and the in operator must agree on what has actually been inserted."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )
    sharded_filter.add(b"present-member")

    self.assertIn(b"present-member", sharded_filter)
    self.assertNotIn(b"absent-member", sharded_filter)

  def test_contains_operator_rejects_non_bytes_argument(self) -> None:
    """The in operator carries no static guarantee of its argument type and must validate it."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )

    with self.assertRaises(TypeError):
      sharded_filter.__contains__("not-bytes")  # type: ignore[arg-type]

  def test_shard_count_property_reflects_the_requested_count(self) -> None:
    """The shard_count property must reflect exactly what open was asked to allocate."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=6, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )

    self.assertEqual(sharded_filter.shard_count, 6)

  def test_repr_contains_shard_count_and_dirty_count(self) -> None:
    """A useful repr must let an engineer identify partitioning and pending activity at a glance."""
    sharded_filter = ShardedBloomFilter.open(
      self._shard_directory, shard_count=4, capacity=1_000, error_rate=0.01, policy=PersistPolicy()
    )
    sharded_filter.add(b"a-member")
    representation = repr(sharded_filter)

    self.assertIn("shard_count=4", representation)
    self.assertIn("dirty_shard_count=1", representation)


if __name__ == "__main__":
  unittest.main()
