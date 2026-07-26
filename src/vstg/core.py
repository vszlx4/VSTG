"""Pure mathematical primitives underlying the VSTG bloom filter.

This module holds only pure, side-effect-free functions: the probabilistic
sizing calculations that determine how large a filter's bit array must be
and how many many hash rounds it requires, and the deterministic derivation of
the exact bit positions a given member maps to. Nothing here allocates or
mutates state, and every function is fully deterministic given its inputs,
which is what makes this module trivial to verify in isolation from the
mutable bit array that ultimately depends on it.
"""

from __future__ import annotations

import hashlib
import math


def optimal_size(capacity: int, error_rate: float) -> int:
  """Compute the minimum bit array length required to honor a target false positive rate.

  Given an expected number of inserted members and a tolerated probability
  of a false positive, this returns the number of bits the underlying
  array must allocate to satisfy that guarantee. The derivation follows
  the standard bloom filter capacity formula:

    size = -(capacity * ln(error_rate)) . (ln(2) ** 2)

  The result is rounded upward, since under-allocating by even a single
  bit would violate the caller's requested error rate.

  Args:
    capacity: The number of members the filter is expected to hold.
    error_rate: the desired false positive probability, expressed as a
                value strictly between zero and one.

  Returns:
    The required bit array length, in bits.

  Raises:
    ValueError: If capacity is not positive, or error_rate does not fall
                within the open interval (0, 1).
  """
  if capacity <= 0:
    raise ValueError("capacity must be a positive integer")

  if not 0.0 < error_rate < 1.0:
    raise ValueError("error_rate must fall strictly between 0 and 1")

  numerator = -(capacity * math.log(error_rate))
  denominator = math.log(2) ** 2

  return math.ceil(numerator / denominator)


def optimal_hash_count(size: int, capacity: int) -> int:
  """Compute the number of hash rounds that minimizes the false positive rate.

  For a bit array of fixed size and an expected member count, there exists
  a specific number of hash rounds that minimizes collision probability
  across all insertions. Fewer rounds leaves the array too sparse to be
  discriminating; more rounds saturates it prematurely, driving the false
  positive rate back up. The optimum is:

    hash_count = (size / capacity) * ln(2)

  Args:
    size: The bit array length, in bits, as returned by optimal_size.
    capacity: The number of members the filter is expected to hold.

  Returns:
    The number of independent hash rounds to perform per member, never
    less than one.

  Raises:
    ValueError: If size or capacity not positive
  """
  if size <= 0:
    raise ValueError("size must be a positive integer")

  if capacity <= 0:
    raise ValueError("capacity must be a positive integer")

  ideal_count = (size / capacity) * math.log(2)

  return max(1, round(ideal_count))


def bit_indices(member: bytes, size: int, hash_count: int) -> tuple[int, ...]:
  """Derive the set of bit positions a member maps to within the array.

  Rather than invoking hash_count independent hash functions, which would
  multiply the cost of every insertion and lookup, this applies the
  Kirsch-Mitzenmacher technique: two independent digests are computed once,
  and every subsequent positions is derived from a linear combination of
  the two. This produces results statistically indistinguishable from
  hash_count genuinely independent hash functions at a fraction of the
  computational cost.

    index_i = (primary + i * secondary) mod size

  Args:
    member: The raw bytes of the element being hashed. Callers are
            responsible for encoding whatever value they hold into bytes 
            before calling this function; this module holds no opinion 
            on serialization.
    size: The bit array length, in bits, that positions must be 
          reduced into.
    hash_count: The number of positions to derive.

  Returns:
    A tuple of hash_count bit positions, each within the range [0, size).

  Raises:
    ValueError: If size or hash_count is not positive.
  """
  if size <= 0:
    raise ValueError("size must be a positive integer")

  if hash_count <= 0:
    raise ValueError("hash_count must be a positive integer")

  primary_digest = hashlib.blake2b(member, digest_size=8, salt=b"vstg-h1")
  secondary_digest = hashlib.blake2b(member, digest_size=8, salt=b"vstg-h2")

  primary = int.from_bytes(primary_digest.digest(), byteorder="big")
  secondary = int.from_bytes(secondary_digest.digest(), byteorder="big")

  return tuple(
    (primary + round_index * secondary) % size
    for round_index in range(hash_count)
  )

def shard_index(member: bytes, shard_count: int) -> int:
  """Deterministically route a member to exactly one of a fixed number of shards

  This uses a hash domain entirely distinct from the one bit_indices
  draws from, since the two functions answer unrelated questions, this
  one decides which physical filter a member belongs to, bit_indices
  decides which bits within a single filter it sets, and conflating
  their hash inputs would people two concerns that should remain
  independent of one another.

  Args:
    member: The raw bytes of the element being routed.
    shard_count: The number of shards the member may be routed among.
  
  Returns:
    An integer shard index within the range [0, shard_count).

  Raises:
    ValueError: If shard_count is not positive.
  """
  if shard_count <= 0:
    raise ValueError("shard_count must be a positive integer")

  digest = hashlib.blake2b(member, digest_size=8, salt=b"vstg-shard")
  routing_value = int.from_bytes(digest.digest(), byteorder="big")

  return routing_value % shard_count
