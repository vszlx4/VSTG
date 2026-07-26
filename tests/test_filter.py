"""Unit tests for the mutable BloomFilter structure in vstg.filter.

These tests cover the two deterministic guarantees a bloom filter must
uphold unconditionally, that every inserted member is always reported
present and that a filter with nothing inserted always reports absence,
alongside the probabilistic guarantee that can only be validated
empirically, that the observed false positive rate across a large
sample of unrelated members stays within a reasonable bound of the
rate the filter was configured to honor.
"""

from __future__ import annotations

import unittest

from vstg.filter import BloomFilter


class BasicOperationTests(unittest.TestCase):
  """Validate the two unconditional correctness guarantees of the filter."""

  def test_added_member_is_always_reported_present(self) -> None:
    """Insertion followed immediately by a check must never report absence."""
    bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
    bloom_filter.add(b"registered-member")

    self.assertTrue(bloom_filter.might_contain(b"registered-member"))

  def test_empty_filter_reports_every_member_absent(self) -> None:
    """With no bits set at all, every possible member must report absence."""
    bloom_filter = BloomFilter(capacity=100, error_rate=0.01)

    self.assertFalse(bloom_filter.might_contain(b"never-inserted"))

  def test_multiple_distinct_members_are_all_retained(self) -> None:
    """Inserting several members must not cause earlier insertions to be lost."""
    bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
    bloom_filter.add(b"first-member")
    bloom_filter.add(b"second-member")
    bloom_filter.add(b"third-member")

    self.assertTrue(bloom_filter.might_contain(b"first-member"))
    self.assertTrue(bloom_filter.might_contain(b"second-member"))
    self.assertTrue(bloom_filter.might_contain(b"third-member"))


class IdempotencyTests(unittest.TestCase):
  """Validate that repeated insertion of the same member is a no-op beyond the first."""

  def test_repeated_insertion_leaves_bits_unchanged(self) -> None:
    """Setting an already-set bit must not alter the backing array further."""
    bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
    bloom_filter.add(b"repeated-member")
    bits_after_first_insertion = bloom_filter.bits

    bloom_filter.add(b"repeated-member")
    bloom_filter.add(b"repeated-member")

    self.assertEqual(bits_after_first_insertion, bloom_filter.bits)


class ContainsOperatorTests(unittest.TestCase):
  """Validate that the in operator delegates faithfully to might_contain."""

  def test_in_operator_reflects_insertion(self) -> None:
    """The in operator must agree with an explicit might_contain call."""
    bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
    bloom_filter.add(b"present-member")

    self.assertIn(b"present-member", bloom_filter)
    self.assertNotIn(b"absent-member", bloom_filter)


class PropertyTests(unittest.TestCase):
  """Validate that constructor arguments and derived sizing are exposed accurately."""

  def test_capacity_and_error_rate_reflect_constructor_arguments(self) -> None:
    """The properties must return exactly what the caller supplied, unmodified."""
    bloom_filter = BloomFilter(capacity=5_000, error_rate=0.02)

    self.assertEqual(bloom_filter.capacity, 5_000)
    self.assertEqual(bloom_filter.error_rate, 0.02)

  def test_size_and_hash_count_are_positive(self) -> None:
    """Derived sizing must always produce usable, strictly positive values."""
    bloom_filter = BloomFilter(capacity=5_000, error_rate=0.02)

    self.assertGreater(bloom_filter.size, 0)
    self.assertGreater(bloom_filter.hash_count, 0)


class BitsSnapshotTests(unittest.TestCase):
  """Validate that the bits property exposes state without permitting mutation."""

  def test_bits_snapshot_does_not_grow_stale_or_leak_a_live_reference(self) -> None:
    """A previously fetched snapshot must remain frozen even as the filter changes further."""
    bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
    bloom_filter.add(b"first-member")
    earlier_snapshot = bloom_filter.bits

    bloom_filter.add(b"second-member")
    later_snapshot = bloom_filter.bits

    self.assertIsInstance(earlier_snapshot, bytes)
    self.assertNotEqual(earlier_snapshot, later_snapshot)


class RestoreTests(unittest.TestCase):
  """Validate that a filter reconstructed through _restore behaves identically to the original."""

  def test_restored_filter_agrees_with_original_on_membership(self) -> None:
    """Every membership answer the original gives must be reproduced exactly after restoration."""
    original_filter = BloomFilter(capacity=200, error_rate=0.01)
    original_filter.add(b"restored-member-one")
    original_filter.add(b"restored-member-two")

    restored_filter = BloomFilter._restore(
      original_filter.capacity,
      original_filter.error_rate,
      original_filter.size,
      original_filter.hash_count,
      original_filter.bits,
    )

    self.assertTrue(restored_filter.might_contain(b"restored-member-one"))
    self.assertTrue(restored_filter.might_contain(b"restored-member-two"))
    self.assertFalse(restored_filter.might_contain(b"never-inserted-member"))

  def test_restored_filter_preserves_configuration_fields(self) -> None:
    """Capacity, error rate, size, and hash count must survive restoration unchanged."""
    original_filter = BloomFilter(capacity=200, error_rate=0.01)

    restored_filter = BloomFilter._restore(
      original_filter.capacity,
      original_filter.error_rate,
      original_filter.size,
      original_filter.hash_count,
      original_filter.bits,
    )

    self.assertEqual(restored_filter.capacity, original_filter.capacity)
    self.assertEqual(restored_filter.error_rate, original_filter.error_rate)
    self.assertEqual(restored_filter.size, original_filter.size)
    self.assertEqual(restored_filter.hash_count, original_filter.hash_count)


class ReprTests(unittest.TestCase):
  """Validate that the debugging representation surfaces the fields that matter."""

  def test_repr_contains_class_name_and_configuration_fields(self) -> None:
    """A useful repr must let an engineer identify both the type and its configuration at a glance."""
    bloom_filter = BloomFilter(capacity=300, error_rate=0.05)
    representation = repr(bloom_filter)

    self.assertIn("BloomFilter", representation)
    self.assertIn("capacity=300", representation)
    self.assertIn("error_rate=0.05", representation)


class EmpiricalFalsePositiveRateTests(unittest.TestCase):
  """Validate the probabilistic guarantee the filter cannot prove analytically at runtime.

  Correctness of the sizing formula in core.py is established by
  OptimalSizeTests and OptimalHashCountTests against their known
  mathematical properties. What those tests cannot establish is that
  the filter, as actually assembled from that sizing and the hashing
  in bit_indices, behaves as the formula predicts once real data is
  pushed through it. This class inserts a known population, tests an
  entirely disjoint population of the same size, and confirms the
  observed false positive rate stays within a generous bound of the
  configured target, since the underlying hash outputs for any given
  input are fixed and not reseeded, an implementation error here would
  manifest as a reproducible, not merely occasional, violation.
  """

  def test_observed_false_positive_rate_stays_within_bound_of_configured_target(self) -> None:
    """A correct implementation should not exceed several times its configured error rate."""
    configured_error_rate = 0.01
    population_size = 2_000
    bloom_filter = BloomFilter(capacity=population_size, error_rate=configured_error_rate)

    for index in range(population_size):
      bloom_filter.add(f"inserted-member-{index}".encode("utf-8"))

    sample_size = 20_000
    false_positive_count = 0

    for index in range(sample_size):
      candidate = f"disjoint-member-{index}".encode("utf-8")
      if bloom_filter.might_contain(candidate):
        false_positive_count += 1

    observed_rate = false_positive_count / sample_size
    acceptable_upper_bound = configured_error_rate * 3

    self.assertLess(observed_rate, acceptable_upper_bound)

  def test_no_false_negatives_occur_across_the_entire_inserted_population(self) -> None:
    """Unlike the false positive rate, this guarantee must hold with zero exceptions."""
    population_size = 2_000
    bloom_filter = BloomFilter(capacity=population_size, error_rate=0.01)

    for index in range(population_size):
      bloom_filter.add(f"inserted-member-{index}".encode("utf-8"))

    for index in range(population_size):
      member = f"inserted-member-{index}".encode("utf-8")
      self.assertTrue(bloom_filter.might_contain(member))


if __name__ == "__main__":
  unittest.main()
