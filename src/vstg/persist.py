"""Durable checkpointing for the in-memory VSTG bloom filter.

A BloomFilter exists entirely in memory during normal operation, for
the same reason any performance-sensitive structure does: disk access
on every mutation would be several orders of magnitude slower than the
bit-flipping the structure was designed to make cheap. This module
provides the mechanism by which that in-memory state is periodically
made durable, a compact binary serialization format, a pure decision
function for when a checkpoint is warranted, and the atomic file
operations that write and restore that format without risk of leaving
a corrupted checkpoint behind.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from vstg.filter import BloomFilter


_FILE_SIGNATURE = b"VSTG"
_FORMAT_VERSION = 1
_HEADER_LAYOUT = ">4sBQdQI"
_HEADER_LENGTH = struct.calcsize(_HEADER_LAYOUT)


@dataclass(frozen=True)
class PersistPolicy:
  """Configuration governing when a running filter checkpoints to disk.

  A policy expresses up to two independent triggers for an automatic
  checkpoint, either of which is sufficient on its own to warrant one.
  A policy leaving both unset performs no automatic checkpointing and
  relies entirely on manual invocation or on checkpoint_on_shutdown.

  Attributes:
    checkpoint_interval_seconds: The maximum duration, in seconds,
                                 permitted to elapse between checkpoints. None disables
                                 time-based checkpointing.
    checkpoint_insert_threshold: The maximum number of insertions
                                 permitted to accumulate between checkpoints. None disables
                                 count-based checkpointing.
    checkpoint_on_shutdown: Whether a final checkpoint should be
                            written during an orderly shutdown, independent of how much
                            time or how many insertions have elapsed since the previous one.
  """

  checkpoint_interval_seconds: float | None = None
  checkpoint_insert_threshold: int | None = None
  checkpoint_on_shutdown: bool = True

  def __post_init__(self) -> None:
    """Reject non-positive thresholds before they can silently misbehave."""
    if self.checkpoint_interval_seconds is not None and self.checkpoint_interval_seconds <= 0:
      raise ValueError("checkpoint_interval_seconds must be positive when set")

    if self.checkpoint_insert_threshold is not None and self.checkpoint_insert_threshold <= 0:
      raise ValueError("checkpoint_insert_threshold must be positive when set")


def should_checkpoint(
  policy: PersistPolicy,
  last_checkpoint_at: float,
  current_time: float,
  inserts_since_checkpoint: int,
) -> bool:
  """Determine whether accumulated activity warrants an immediate checkpoint.

  This function accepts explicit timestamps rather than reading a
  clock internally, which keeps the decision deterministic and
  trivially testable without waiting on real elapsed time or mocking
  the system clock.

  Args:
    policy: The checkpoint policy currently in effect.
    last_checkpoint_at: The timestamp, in seconds, at which the most
                        recent checkpoint was written.
    current_time: The current timestamp, in seconds, measured against
                  last_checkpoint_at to determine elapsed time.
    inserts_since_checkpoint: The number of insertions that have
                              occurred since the most recent checkpoint.

  Returns:
    True if either the configured interval or insertion threshold has
    been reached or exceeded; False otherwise.
  """
  if policy.checkpoint_interval_seconds is not None:
    elapsed_seconds = current_time - last_checkpoint_at
    if elapsed_seconds >= policy.checkpoint_interval_seconds:
      return True

  if policy.checkpoint_insert_threshold is not None:
    if inserts_since_checkpoint >= policy.checkpoint_insert_threshold:
      return True

  return False


def serialize(bloom_filter: BloomFilter) -> bytes:
  """Encode a filter's complete state into a portable binary payload.

  The payload begins with a fixed-width header identifying the format
  and recording the parameters required to reconstruct the filter,
  followed immediately by the raw bit array contents. The header opens
  with a four-byte signature and a version byte specifically so that a
  future format revision, or an entirely unrelated file handed to this
  function by mistake, is rejected explicitly during deserialization
  rather than silently misinterpreted.

  Args:
    bloom_filter: The filter instance to encode.

  Returns:
    The complete serialized payload, header followed by bit array.
  """
  header = struct.pack(
    _HEADER_LAYOUT,
    _FILE_SIGNATURE,
    _FORMAT_VERSION,
    bloom_filter.capacity,
    bloom_filter.error_rate,
    bloom_filter.size,
    bloom_filter.hash_count,
  )

  return header + bloom_filter.bits


def deserialize(payload: bytes) -> BloomFilter:
  """Reconstruct a filter instance from a previously serialized payload.

  Args:
    payload: The complete binary payload, as produced by serialize.

  Returns:
    A filter instance whose state exactly matches the one originally
    serialized.

  Raises:
    ValueError: If the payload is too short to contain a complete
                header, carries an unrecognized signature or unsupported format
                version, or the bit array length does not match what the header
                declares.
  """
  if len(payload) < _HEADER_LENGTH:
    raise ValueError("payload is truncated: incomplete header")

  signature, version, capacity, error_rate, size, hash_count = struct.unpack(
    _HEADER_LAYOUT, payload[:_HEADER_LENGTH]
  )

  if signature != _FILE_SIGNATURE:
    raise ValueError(f"unrecognized file signature: {signature!r}")

  if version != _FORMAT_VERSION:
    raise ValueError(f"unsupported format version: {version}")

  bits = payload[_HEADER_LENGTH:]
  expected_byte_length = math.ceil(size / 8)

  if len(bits) != expected_byte_length:
    raise ValueError(
      f"bit array length mismatch: header declares {expected_byte_length} "
      f"bytes, payload contains {len(bits)}"
    )

  return BloomFilter._restore(capacity, error_rate, size, hash_count, bits)


def save(bloom_filter: BloomFilter, destination: Path) -> None:
  """Persist a filter to disk atomically.

  The payload is first written in full to a temporary sibling file and
  only then moved into place with an atomic rename, this guarantees
  that a process interrupted mid-write, whether by a crash or a forced
  shutdown, can never leave a corrupted, partially-written file at the
  destination, the destination holds either the previous complete
  checkpoint or the new one, never a mixture of both.

  Args:
    bloom_filter: The filter instance to serialize and persist.
    destination: The file path the checkpoint should occupy once
                 written.
  """
  payload = serialize(bloom_filter)
  temporary_path = destination.with_suffix(destination.suffix + ".tmp")
  temporary_path.write_bytes(payload)
  temporary_path.replace(destination)


def load(source: Path) -> BloomFilter:
  """Load a previously persisted filter from disk.

  Args:
    source: The file path a checkpoint was previously written to.

  Returns:
    A filter instance whose state exactly matches the one that was
    checkpointed.
  """
  payload = source.read_bytes()

  return deserialize(payload)
