# Bwtree

Latch-free Bw-tree, written from scratch in C++20, hooked into Postgres 16 as a custom index access method.

## What is this

A Bw-tree is an ordered index that never takes a lock. Instead of locking a page to update it, you install a small "delta" record onto it with one atomic CAS. Readers just walk the delta chain when they hit a page. Once a chain gets long it gets folded into a fresh base node. No latches anywhere.

I'm not claiming this beats everything. A sharded hash map will out-throughput it on plain point lookups, because sharding gives up ordering to get there. What the Bw-tree gets you is being ordered *and* latch-free at the same time, which almost nothing else in this comparison can say, plus way more stable tail latency once threads start fighting over the same pages.

## Layout

```
src/core/        the actual tree (no Postgres dependency, builds standalone)
src/postgres/    the Postgres access method + C ABI shim
bench.cpp        YCSB-style benchmark harness
overnight.sh     runs the full benchmark battery, regenerates figures
```

Core files, if you're poking around:
- `bwtree.h` / `bwtree_core.cpp` — insert/lookup/remove, splits, consolidation
- `mapping_table.h` — the logical page id -> physical pointer indirection that makes the CAS trick work
- `epoch.h` — epoch-based reclamation so old nodes get freed without anyone yanking memory out from under a reader
- `node.h` — base nodes + delta records

One thing I want to flag rather than bury: merges (on delete) aren't latch-free yet. An underflowing leaf just sits there until it gets consolidated again. Doesn't break correctness, just wastes a bit of space. Listed as future work, not hidden.

## Building

```bash
g++ -O2 -std=c++20 -pthread -I. bench.cpp bwtree_core.cpp -o bench
```

Postgres extension build is in `src/postgres/`, needs PG16 headers.

The Postgres side is "Tier 1" right now: tree lives backend-local, gets rebuilt from the heap on a fresh backend. Passes 14/14 correctness checks (point + range + negative cases on a unique bigint index). Real cross-backend concurrency through PG shared memory is Tier 2, not built yet. So the concurrency numbers below come from the standalone harness driving the tree directly, not from inside a live Postgres session.

## Benchmarks

Compares against `stdmap` (single thread, just a reference point), `mutexmap`, `sharedmap`, and `shardedmap` (64 shards, scales great on point ops, can't do a correct ordered scan across shards).

```bash
./overnight.sh
```

or run one sweep directly, e.g.:

```bash
./bench results_writeratio.csv --readratios "1.0 0.9 0.75 0.5 0.25 0.1 0.0" --threads "8" --trials 5
```

At 8 threads, uniform keys, 1M keys:

read-only: bwtree does 5.05 Mops/s, sharded does 7.36, sharedmap 2.68, mutex 0.94
balanced (50/50): bwtree 3.73, sharded 6.17, sharedmap 0.37, mutex 1.05

So yeah, sharded wins on throughput. But p99.9 tail latency on read-only tells a different story: bwtree sits at 10.7us, sharded at 48.5, sharedmap at 50, mutex blows up to 219. bwtree's p99 barely moves from 1 to 8 threads. Everything else degrades 20x+ over the same range.

Full numbers in `RESULTS.md`, figures in `figures/`.

Correctness: ran a concurrent oracle test (random concurrent ops, check final state against ground truth) and ThreadSanitizer comes back clean, 0 data races.

## Not done yet

- Tier 2 Postgres integration (shared memory, actual cross-backend concurrency)
- latch-free merges on delete
- MVCC-aware range scans
- head to head against something like ART or Masstree, not just the toy baselines here

Built this as a class project, so scope is intentionally cut where it is.
