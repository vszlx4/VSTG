# Changelog

All notable changes to this project are documented here.

This project follows [Semantic Versioning](https://semver.org/), and
this file follows the [Keep a Changelog](https://keepachangelog.com/)
format.

## [Unreleased]

## [0.1.0] - 2026-07-27

### Added
- `BloomFilter`, the core in-memory bit array with `add` and `might_contain`.
- `core.py` sizing math, `optimal_size`, `optimal_hash_count`, `bit_indices`, `shard_index`.
- `PersistPolicy` and atomic `save`/`load` checkpointing to disk.
- `ManagedBloomFilter`, automatic checkpointing bound to a policy.
- `ShardedBloomFilter`, checkpointing partitioned across multiple files.
- `BloomFilterRegistry`, named filters with independent configuration.
- Module level convenience API, `vstg.init`, `register`, `might_contain`, `add`.