//
// concurrency_stress.cpp
//
// Two tests:
//   test_basic()  — single-threaded sanity: insert / overwrite / lookup / delete /
//                   absence, plus a key-count check that the structure stays
//                   consistent across many consolidations and splits.
//   test_stress() — the concurrency test the assignment asks for: M threads doing
//                   MIXED insert + lookup on heavily OVERLAPPING key ranges, then a
//                   single-threaded FINAL VALIDATION pass cross-checking presence
//                   AND absence against a deterministic ground-truth set.
//
// Determinism trick that makes validation airtight despite racing writers:
//   * every insert of key k writes the SAME value vfor(k), so whichever writer
//     "wins" a contended key, the final value is vfor(k) regardless of ordering;
//   * each thread, before exiting, inserts its ENTIRE band, so the set of present
//     keys at the end is exactly the union of all bands = [0, SPACE), independent
//     of how the random phase happened to interleave.
// So after join, ground truth is: every key in [0, SPACE) present with value
// vfor(k); every key outside [0, SPACE) absent.
//
#include "bwtree/bwtree.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace bwtree;

static inline Value vfor(Key k) { return k * 2 + 1; }

// --------------------------------------------------------------------------
static void test_basic() {
  BwTree tree;
  auto ctx = tree.register_thread();

  const int N = 5000;                       // >> split thresholds: forces many
  for (int i = 0; i < N; ++i)               // leaf splits, inner splits, root growth
    tree.insert(ctx, i, vfor(i));

  for (int i = 0; i < N; ++i) {
    Value v = 0;
    bool f = tree.lookup(ctx, i, &v);
    assert(f && v == vfor(i));
  }
  for (int i = N; i < N + 200; ++i) {       // never inserted => absent
    Value v = 0;
    assert(!tree.lookup(ctx, i, &v));
  }

  tree.insert(ctx, 42, 999);                // overwrite
  { Value v = 0; assert(tree.lookup(ctx, 42, &v) && v == 999); }

  assert(tree.remove(ctx, 42));             // delete present
  { Value v = 0; assert(!tree.lookup(ctx, 42, &v)); }
  assert(!tree.remove(ctx, 10 * N));        // delete absent => false

  size_t cnt = tree.debug_count_keys();     // N inserted, 1 deleted
  std::printf("[basic] count = %zu (expected %d)\n", cnt, N - 1);
  assert(cnt == static_cast<size_t>(N - 1));

  std::printf("[basic] PASSED\n");
}

// --------------------------------------------------------------------------
static void test_stress() {
  BwTree tree;

  const int M     = 4;       // worker threads
  const Key SPACE = 4000;    // key space [0, SPACE)
  const int OPS   = 15000;   // mixed ops per thread before the band-fill

  // Band for thread t: width = 75% of the space, slid so that consecutive bands
  // overlap heavily and the union of all bands is exactly [0, SPACE).
  auto band = [&](int t) -> std::pair<Key, Key> {
    Key width = SPACE * 3 / 4;
    Key start = (SPACE - width) * t / (M > 1 ? (M - 1) : 1);
    return {start, start + width};
  };

  std::atomic<bool> go{false};
  std::vector<std::thread> threads;

  for (int t = 0; t < M; ++t) {
    threads.emplace_back([&, t] {
      auto ctx = tree.register_thread();
      auto range = band(t);
      Key lo = range.first, hi = range.second;

      std::mt19937_64 rng(1000u + static_cast<unsigned>(t));
      std::uniform_int_distribution<Key> pick(lo, hi - 1);    // within my band
      std::uniform_int_distribution<Key> any(0, SPACE - 1);   // anywhere

      while (!go.load(std::memory_order_acquire)) { /* spin to maximize overlap */ }

      // Mixed phase: ~75% inserts (contending on overlapping keys), ~25% lookups
      // (exercising the read path concurrently with other threads' writes).
      for (int i = 0; i < OPS; ++i) {
        if ((i & 3) != 0) {
          Key k = pick(rng);
          tree.insert(ctx, k, vfor(k));
        } else {
          Key k = any(rng);
          Value v = 0;
          if (tree.lookup(ctx, k, &v))
            assert(v == vfor(k) && "a present key must carry its deterministic value");
        }
      }

      // Deterministic coverage: make sure my whole band ends up inserted.
      for (Key k = lo; k < hi; ++k) tree.insert(ctx, k, vfor(k));
    });
  }

  go.store(true, std::memory_order_release);
  for (auto& th : threads) th.join();

  // ---- FINAL VALIDATION (single-threaded) ---------------------------------
  auto vctx = tree.register_thread();

  // Presence + correct value for every key in the ground-truth union [0, SPACE).
  for (Key k = 0; k < SPACE; ++k) {
    Value v = 0;
    bool f = tree.lookup(vctx, k, &v);
    assert(f && "every key in the union must be present");
    assert(v == vfor(k) && "value must match the deterministic ground truth");
  }
  // Absence for keys that were never inserted by anyone.
  for (Key k = SPACE; k < SPACE + 500; ++k) {
    Value v = 0; assert(!tree.lookup(vctx, k, &v));
  }
  for (Key k = -500; k < 0; ++k) {
    Value v = 0; assert(!tree.lookup(vctx, k, &v));
  }

  size_t cnt = tree.debug_count_keys();
  std::printf("[stress] M=%d space=%lld ops/thread=%d  final count=%zu (expected %lld)\n",
              M, static_cast<long long>(SPACE), OPS, cnt, static_cast<long long>(SPACE));
  assert(cnt == static_cast<size_t>(SPACE));

  std::printf("[stress] PASSED\n");
}

// --------------------------------------------------------------------------
int main() {
  test_basic();
  test_stress();
  std::printf("ALL TESTS PASSED\n");
  return 0;
}
