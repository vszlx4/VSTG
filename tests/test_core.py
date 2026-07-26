"""Unit tests for the pure mathematical primitives in vstg.core.

These tests exercise optimal_size, optimal_hash_count, and bit_indices
in isolation, since all three are fully deterministic and free of any
mutable state. Coverage here is intended to be exhaustive precisely
because these functions form the foundation every other module in the
package depends on. An error in the sizing formula or the index
derivation would silently corrupt the correctness guarantees of every
filter built on top of it.
"""

from __future__ import annotations

import unittest

from vstg.core import bit_indices, optimal_hash_count, optimal_size


class OptimalSizeTests(unittest.TestCase):
  """Validate the bit array sizing formula against its known properties."""

  def test_size_grows_with_capacity(self) -> None:
    """A larger expected capacity must never yield a smaller bit array."""
    smaller = optimal_size(capacity=1_000, error_rate=0.01)
    larger = optimal_size(capacity=10_000, error_rate=0.01)

    self.assertLess(smaller, larger)

  def test_size_grows_as_error_rate_tightens(self) -> None:
    """A stricter error rate must never yield a smaller bit array."""
    lenient = optimal_size(capacity=10_000, error_rate=0.10)
    strict = optimal_size(capacity=10_000, error_rate=0.001)

    self.assertLess(lenient, strict)

  def test_result_is_always_a_positive_integer(self) -> None:
    """The formula involves floating point division and must be rounded up."""
    size = optimal_size(capacity=500, error_rate=0.05)

    self.assertIsInstance(size, int)
    self.assertGreater(size, 0)

  def test_rejects_non_positive_capacity(self) -> None:
    """A capacity of zero or below describes no meaningful filter."""
    with self.assertRaises(ValueError):
      optimal_size(capacity=0, error_rate=0.01)

    with self.assertRaises(ValueError):
      optimal_size(capacity=-10, error_rate=0.01)

  def test_rejects_error_rate_outside_open_interval(self) -> None:
    """An error rate of exactly zero or one is not a meaningful probability."""
    with self.assertRaises(ValueError):
      optimal_size(capacity=1_000, error_rate=0.0)

    with self.assertRaises(ValueError):
      optimal_size(capacity=1_000, error_rate=1.0)


class OptimalHashCountTests(unittest.TestCase):
  """Validate the hash round count formula against its known properties."""

  def test_never_returns_less_than_one(self) -> None:
    """Even a poorly proportioned size and capacity must yield at least one round."""
    hash_count = optimal_hash_count(size=10, capacity=1_000)

    self.assertGreaterEqual(hash_count, 1)

  def test_result_is_always_an_integer(self) -> None:
    """The formula involves a floating point ratio and must be rounded."""
    hash_count = optimal_hash_count(size=9_586, capacity=1_000)

    self.assertIsInstance(hash_count, int)

  def test_matches_the_reference_case(self) -> None:
    """A well known capacity and error rate pair should reproduce its textbook hash count."""
    size = optimal_size(capacity=1_000, error_rate=0.01)
    hash_count = optimal_hash_count(size, capacity=1_000)

    self.assertEqual(hash_count, 7)

  def test_rejects_non_positive_size(self) -> None:
    """A bit array of zero or negative length cannot be indexed into."""
    with self.assertRaises(ValueError):
      optimal_hash_count(size=0, capacity=1_000)

  def test_rejects_non_positive_capacity(self) -> None:
    """A capacity of zero or below describes no meaningful filter."""
    with self.assertRaises(ValueError):
      optimal_hash_count(size=1_000, capacity=0)


class BitIndicesTests(unittest.TestCase):
  """Validate the deterministic derivation of bit positions for a member."""

  def test_same_member_always_yields_identical_positions(self) -> None:
    """Determinism is the property every other guarantee in the filter relies on."""
    first_attempt = bit_indices(b"consistent-member", size=1_024, hash_count=5)
    second_attempt = bit_indices(b"consistent-member", size=1_024, hash_count=5)

    self.assertEqual(first_attempt, second_attempt)

  def test_distinct_members_typically_diverge(self) -> None:
    """Two unrelated members should not collide across every derived position."""
    first_member = bit_indices(b"apple", size=1_024, hash_count=5)
    second_member = bit_indices(b"banana", size=1_024, hash_count=5)

    self.assertNotEqual(first_member, second_member)

  def test_returned_count_matches_hash_count(self) -> None:
    """The number of positions returned must equal the requested hash count."""
    positions = bit_indices(b"member", size=2_048, hash_count=9)

    self.assertEqual(len(positions), 9)

  def test_every_position_falls_within_bounds(self) -> None:
    """No returned position may fall outside the addressable bit array."""
    positions = bit_indices(b"member", size=256, hash_count=12)

    for position in positions:
      self.assertGreaterEqual(position, 0)
      self.assertLess(position, 256)

  def test_rejects_non_positive_size(self) -> None:
    """A bit array of zero or negative length cannot be indexed into."""
    with self.assertRaises(ValueError):
      bit_indices(b"member", size=0, hash_count=5)

  def test_rejects_non_positive_hash_count(self) -> None:
    """Zero or fewer hash rounds would derive no positions at all."""
    with self.assertRaises(ValueError):
      bit_indices(b"member", size=1_024, hash_count=0)


if __name__ == "__main__":
  unittest.main()
