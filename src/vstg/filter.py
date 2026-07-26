"""Mutable bit array implementation of the vstg probabilistic set.

Where core.py holds the pure mathematics governing sizing and index
derivation, this module holds the one piece of the library that is
deliberately, unavoidably mutable: the bit array itself. Flipping a bit
in place, without copying the array on every insertion, is not a style
preference but a hard requirement, a bloom filter that reconstructed
its entire backing array on every add would defeat the memory and
performance guarantees that justify its existence in the first place.
Mutability is therefore confined entirely to this module and never
leaks into core.py, which remains pure and independently verifiable.
"""

from __future__ import annotations

import math

from vstg.core import bit_indices, optimal_hash_count, optimal_size


class BloomFilter:
  """A probabilistic set membership structure backed by a fixed-size bit array.

  A BloomFilter answers a single question, has this member possibly
  been inserted before, using a fixed amount of memory regardless of
  how large or numerous the inserted members are. The structure offers
  an asymmetric correctness guarantee: a negative answer is always
  certain, since insertion only ever sets bits and never clears them,
  while a positive answer is only probable, since unrelated members can
  coincidentally set the same combination of bits. The rate at which
  that coincidence occurs is bounded by the error_rate supplied at
  construction and enforced through the sizing derived in core.py.

  Instances are not safe for concurrent mutation from multiple threads
  without external synchronization, since bit flips are not atomic
  across the underlying bytearray.
  """

  def __init__(self, capacity: int, error_rate: float) -> None:
    """Allocate a bit array sized for the requested capacity and error rate.

    Args:
      capacity: The number of members this filter is expected to hold.
                Exceeding this figure does not corrupt the filter, but degrades
                its actual false positive rate beyond what was requested.
      error_rate: The desired false positive probability, expressed as
                  a value strictly between zero and one.

    Raises:
      ValueError: If capacity or error_rate falls outside the domain
                  accepted by optimal_size and optimal_hash_count.
    """
    size = optimal_size(capacity, error_rate)
    hash_count = optimal_hash_count(size, capacity)

    self._capacity = capacity
    self._error_rate = error_rate
    self._size = size
    self._hash_count = hash_count
    self._bits = bytearray(math.ceil(size / 8))

  @classmethod
  def _restore(
    cls,
    capacity: int,
    error_rate: float,
    size: int,
    hash_count: int,
    bits: bytes,
  ) -> BloomFilter:
    """Reconstruct a filter instance from previously persisted state.

    This bypasses the normal constructor entirely, since __init__
    always allocates a fresh, zeroed bit array from scratch and
    recomputes sizing from capacity and error_rate. Restoration
    instead needs to install an already-populated bit array exactly as
    it was serialized, without re-deriving or re-zeroing anything.
    This method is intended for use by the persistence layer and is
    not part of the public construction interface.

    Args:
      capacity: The originally configured member capacity.
      error_rate: The originally configured false positive rate.
      size: The bit array length, in bits, recorded at serialization
            time.
      hash_count: The number of hash rounds recorded at serialization
                  time.
      bits: The raw bit array contents, as bytes.

    Returns:
      A fully initialized filter instance reflecting the restored
      state.
    """
    instance = cls.__new__(cls)
    instance._capacity = capacity
    instance._error_rate = error_rate
    instance._size = size
    instance._hash_count = hash_count
    instance._bits = bytearray(bits)

    return instance

  @property
  def capacity(self) -> int:
    """The number of members this filter was sized to hold."""
    return self._capacity

  @property
  def error_rate(self) -> float:
    """The false positive probability this filter was sized to honor."""
    return self._error_rate

  @property
  def size(self) -> int:
    """The length of the backing bit array, in bits."""
    return self._size

  @property
  def hash_count(self) -> int:
    """The number of derived bit positions consulted per member."""
    return self._hash_count

  @property
  def bits(self) -> bytes:
    """An immutable snapshot of the backing bit array.

    Returned as a fresh bytes copy rather than a reference to the
    internal bytearray, preventing callers from mutating filter state
    through a channel that bypasses the hashing logic in add.
    """
    return bytes(self._bits)

  def add(self, member: bytes) -> None:
    """Insert a member by setting each of its derived bit positions.

    This operation is idempotent: inserting the same member repeatedly
    has no effect beyond the first insertion, since the same positions
    are derived deterministically every time and setting an already-set
    bit is a no-op.

    Args:
      member: The raw bytes of the element being inserted. Callers are
              responsible for encoding whatever logical value they hold into
              a stable byte representation before calling this method.
    """
    for position in bit_indices(member, self._size, self._hash_count):
      byte_index, bit_offset = divmod(position, 8)
      self._bits[byte_index] |= 1 << bit_offset

  def might_contain(self, member: bytes) -> bool:
    """Test whether a member may have been previously inserted.

    The check short circuits on the first cleared bit it encounters,
    since a single cleared bit is sufficient proof that the member was
    never inserted. Only a member whose every derived position is set
    reaches the end of the loop and is reported as probably present.

    Args:
      member: The raw bytes of the element being tested, encoded
              identically to how it was encoded at insertion time.

    Returns:
      False if the member is certainly absent. True if the member is
      probably present, subject to the false positive rate the filter
      was constructed with.
    """
    for position in bit_indices(member, self._size, self._hash_count):
      byte_index, bit_offset = divmod(position, 8)
      if not (self._bits[byte_index] >> bit_offset) & 1:
        return False

    return True

  def __contains__(self, member: object) -> bool:
    """Enable idiomatic membership testing through the in operator.

    Unlike might_contain, this method accepts an object of any type,
    matching the signature the language itself requires of anything
    implementing the in operator. A non-bytes argument is rejected
    explicitly here, rather than allowed to fail unpredictably deeper
    inside bit_indices, since callers of the in operator have no
    static guarantee they are passing bytes.

    Args:
      member: The candidate object being tested for membership.

    Returns:
      False if the member is certainly absent. True if the member is
      probably present.

    Raises:
      TypeError: If member is not an instance of bytes.
    """
    if not isinstance(member, bytes):
      raise TypeError(
        f"BloomFilter membership testing requires bytes, received {type(member).__name__}"
      )

    return self.might_contain(member)

  def __repr__(self) -> str:
    """Return an unambiguous representation for debugging and logging."""
    return (
      f"BloomFilter(capacity={self._capacity!r}, "
      f"error_rate={self._error_rate!r}, "
      f"size={self._size!r}, "
      f"hash_count={self._hash_count!r})"
    )
