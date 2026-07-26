"""Unit tests for the persistence layer in vstg.persist.

These tests cover the PersistPolicy dataclass and its validation, the
pure should_checkpoint decision function, the serialize and deserialize
round trip, and the save and load functions that perform actual disk
interaction through a temporary directory. The malformed payload tests
exist to confirm deserialize fails loudly and specifically rather than
silently misinterpreting a corrupted or unrelated file.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from vstg.filter import BloomFilter
from vstg.persist import (
  PersistPolicy,
  _FILE_SIGNATURE,
  _FORMAT_VERSION,
  _HEADER_LAYOUT,
  _HEADER_LENGTH,
  deserialize,
  load,
  save,
  serialize,
  should_checkpoint,
)


class PersistPolicyTests(unittest.TestCase):
  """Validate construction and field validation of the PersistPolicy dataclass."""

  def test_defaults_permit_construction_with_no_arguments(self) -> None:
    """A policy with no automatic triggers configured must still construct successfully."""
    policy = PersistPolicy()

    self.assertIsNone(policy.checkpoint_interval_seconds)
    self.assertIsNone(policy.checkpoint_insert_threshold)
    self.assertTrue(policy.checkpoint_on_shutdown)

  def test_accepts_both_triggers_configured_simultaneously(self) -> None:
    """Both a time based and count based trigger may be configured together."""
    policy = PersistPolicy(checkpoint_interval_seconds=300, checkpoint_insert_threshold=1_000)

    self.assertEqual(policy.checkpoint_interval_seconds, 300)
    self.assertEqual(policy.checkpoint_insert_threshold, 1_000)

  def test_rejects_non_positive_interval(self) -> None:
    """An interval of zero or below describes no meaningful waiting period."""
    with self.assertRaises(ValueError):
      PersistPolicy(checkpoint_interval_seconds=0)

    with self.assertRaises(ValueError):
      PersistPolicy(checkpoint_interval_seconds=-5)

  def test_rejects_non_positive_insert_threshold(self) -> None:
    """A threshold of zero or below describes no meaningful accumulation."""
    with self.assertRaises(ValueError):
      PersistPolicy(checkpoint_insert_threshold=0)

    with self.assertRaises(ValueError):
      PersistPolicy(checkpoint_insert_threshold=-1)

  def test_policy_is_immutable(self) -> None:
    """A frozen dataclass must reject attribute assignment after construction."""
    policy = PersistPolicy(checkpoint_interval_seconds=60)

    with self.assertRaises(AttributeError):
      policy.checkpoint_interval_seconds = 120  # type: ignore[misc]


class ShouldCheckpointTests(unittest.TestCase):
  """Validate the pure decision function governing automatic checkpoint timing."""

  def test_fires_when_configured_interval_has_elapsed(self) -> None:
    """An elapsed duration meeting or exceeding the interval must trigger a checkpoint."""
    policy = PersistPolicy(checkpoint_interval_seconds=60)

    self.assertTrue(
      should_checkpoint(policy, last_checkpoint_at=0.0, current_time=60.0, inserts_since_checkpoint=0)
    )

  def test_does_not_fire_before_configured_interval_elapses(self) -> None:
    """An elapsed duration short of the interval must not trigger a checkpoint."""
    policy = PersistPolicy(checkpoint_interval_seconds=60)

    self.assertFalse(
      should_checkpoint(policy, last_checkpoint_at=0.0, current_time=30.0, inserts_since_checkpoint=0)
    )

  def test_fires_when_insert_threshold_is_reached(self) -> None:
    """An insertion count meeting or exceeding the threshold must trigger a checkpoint."""
    policy = PersistPolicy(checkpoint_insert_threshold=100)

    self.assertTrue(
      should_checkpoint(policy, last_checkpoint_at=0.0, current_time=1.0, inserts_since_checkpoint=100)
    )

  def test_does_not_fire_before_insert_threshold_is_reached(self) -> None:
    """An insertion count short of the threshold must not trigger a checkpoint."""
    policy = PersistPolicy(checkpoint_insert_threshold=100)

    self.assertFalse(
      should_checkpoint(policy, last_checkpoint_at=0.0, current_time=1.0, inserts_since_checkpoint=50)
    )

  def test_either_configured_trigger_is_independently_sufficient(self) -> None:
    """With both triggers configured, satisfying only one must still fire a checkpoint."""
    policy = PersistPolicy(checkpoint_interval_seconds=60, checkpoint_insert_threshold=1_000)

    self.assertTrue(
      should_checkpoint(policy, last_checkpoint_at=0.0, current_time=60.0, inserts_since_checkpoint=1)
    )

  def test_no_configured_trigger_never_fires(self) -> None:
    """A policy with neither trigger configured must never request an automatic checkpoint."""
    policy = PersistPolicy()

    self.assertFalse(
      should_checkpoint(
        policy, last_checkpoint_at=0.0, current_time=1_000_000.0, inserts_since_checkpoint=1_000_000
      )
    )


class SerializeDeserializeTests(unittest.TestCase):
  """Validate the in memory serialization round trip, independent of disk interaction."""

  def test_round_trip_preserves_configuration_fields(self) -> None:
    """Capacity, error rate, size, and hash count must survive a serialize and deserialize cycle."""
    original_filter = BloomFilter(capacity=1_000, error_rate=0.01)
    restored_filter = deserialize(serialize(original_filter))

    self.assertEqual(restored_filter.capacity, original_filter.capacity)
    self.assertEqual(restored_filter.error_rate, original_filter.error_rate)
    self.assertEqual(restored_filter.size, original_filter.size)
    self.assertEqual(restored_filter.hash_count, original_filter.hash_count)

  def test_round_trip_preserves_membership_answers(self) -> None:
    """Every membership answer the original filter gives must be reproduced after the round trip."""
    original_filter = BloomFilter(capacity=1_000, error_rate=0.01)
    original_filter.add(b"first-member")
    original_filter.add(b"second-member")

    restored_filter = deserialize(serialize(original_filter))

    self.assertTrue(restored_filter.might_contain(b"first-member"))
    self.assertTrue(restored_filter.might_contain(b"second-member"))
    self.assertFalse(restored_filter.might_contain(b"never-inserted-member"))


class DeserializeErrorTests(unittest.TestCase):
  """Validate that malformed payloads are rejected explicitly rather than silently misread."""

  def test_rejects_payload_shorter_than_the_header(self) -> None:
    """A payload too short to contain a complete header must raise ValueError."""
    with self.assertRaises(ValueError):
      deserialize(b"too-short")

  def test_rejects_payload_with_unrecognized_signature(self) -> None:
    """A payload beginning with a different four byte signature must be rejected outright."""
    original_filter = BloomFilter(capacity=100, error_rate=0.01)
    payload = bytearray(serialize(original_filter))
    payload[0:4] = b"XXXX"

    with self.assertRaises(ValueError):
      deserialize(bytes(payload))

  def test_rejects_payload_with_unsupported_version(self) -> None:
    """A payload declaring a format version this module does not recognize must be rejected."""
    header = struct.pack(_HEADER_LAYOUT, _FILE_SIGNATURE, _FORMAT_VERSION + 1, 100, 0.01, 800, 5)
    bit_length = (800 + 7) // 8
    payload = header + bytes(bit_length)

    with self.assertRaises(ValueError):
      deserialize(payload)

  def test_rejects_payload_with_bit_length_mismatch(self) -> None:
    """A payload whose bit array length disagrees with its own header must be rejected."""
    original_filter = BloomFilter(capacity=100, error_rate=0.01)
    payload = serialize(original_filter)
    truncated_payload = payload[:_HEADER_LENGTH] + payload[_HEADER_LENGTH:-1]

    with self.assertRaises(ValueError):
      deserialize(truncated_payload)


class SaveLoadTests(unittest.TestCase):
  """Validate save and load against an actual temporary directory on disk."""

  def setUp(self) -> None:
    """Allocate a fresh temporary directory for each test to write checkpoints into."""
    self._temporary_directory = tempfile.TemporaryDirectory()
    self._checkpoint_path = Path(self._temporary_directory.name) / "checkpoint.bloom"

  def tearDown(self) -> None:
    """Release the temporary directory and everything written into it."""
    self._temporary_directory.cleanup()

  def test_load_reproduces_a_saved_filter(self) -> None:
    """A filter written with save and read back with load must answer membership identically."""
    original_filter = BloomFilter(capacity=500, error_rate=0.01)
    original_filter.add(b"persisted-member")
    save(original_filter, self._checkpoint_path)

    restored_filter = load(self._checkpoint_path)

    self.assertTrue(restored_filter.might_contain(b"persisted-member"))
    self.assertFalse(restored_filter.might_contain(b"never-persisted-member"))

  def test_save_leaves_no_temporary_file_behind(self) -> None:
    """The atomic rename must leave only the final destination, never its temporary sibling."""
    bloom_filter = BloomFilter(capacity=500, error_rate=0.01)
    save(bloom_filter, self._checkpoint_path)

    temporary_sibling = self._checkpoint_path.with_suffix(self._checkpoint_path.suffix + ".tmp")

    self.assertTrue(self._checkpoint_path.exists())
    self.assertFalse(temporary_sibling.exists())

  def test_save_overwrites_a_previous_checkpoint(self) -> None:
    """A second save to the same destination must fully replace the first, not merge with it."""
    first_filter = BloomFilter(capacity=500, error_rate=0.01)
    first_filter.add(b"member-from-first-checkpoint")
    save(first_filter, self._checkpoint_path)

    second_filter = BloomFilter(capacity=500, error_rate=0.01)
    second_filter.add(b"member-from-second-checkpoint")
    save(second_filter, self._checkpoint_path)

    restored_filter = load(self._checkpoint_path)

    self.assertFalse(restored_filter.might_contain(b"member-from-first-checkpoint"))
    self.assertTrue(restored_filter.might_contain(b"member-from-second-checkpoint"))


if __name__ == "__main__":
  unittest.main()
