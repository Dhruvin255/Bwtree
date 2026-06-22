#pragma once
//
// epoch.h — Epoch-Based Reclamation (EBR).
//
// ============================================================================
//  WHY IMMEDIATE free() IS UNSAFE  (the use-after-free we are preventing)
// ============================================================================
//  Reads in a Bw-tree are latch-free: a thread loads a raw `Node*` head from the
//  mapping table and walks the chain WITHOUT taking any lock. Meanwhile a writer
//  may, with a single CAS on that same mapping-table slot:
//     - consolidate the page (swap in a fresh base, unlinking the whole old chain), or
//     - retire an old base after a split.
//  The two threads overlap in time. If the writer were to `delete` the unlinked
//  nodes the instant the CAS succeeds, the reader — still parked on a node in the
//  middle of that chain — would dereference freed memory. That is a use-after-free:
//  undefined behavior, a heap-corruption bug, and a data race (a read concurrent
//  with the deallocation, with no happens-before between them). AddressSanitizer
//  would flag it as heap-use-after-free; ThreadSanitizer as a data race on the
//  freed object.
//
//  Per-node reference counting would "fix" this but re-introduces an atomic STORE
//  on every node touched during a read — turning the read path back into a stream
//  of contended writes (cache-line ping-pong) and defeating the point of being
//  latch-free. EBR keeps reads write-free and merely DEFERS frees until provably
//  safe.
//
// ============================================================================
//  THE PROTOCOL
// ============================================================================
//  * One monotonically increasing GLOBAL EPOCH counter.
//  * Each thread owns a slot with an "announce" word.
//      - enter(): announce = current global epoch   (I am active in this epoch)
//      - exit():  announce = 0                       (I am quiescent)
//  * A node is RETIRED only AFTER the CAS that unlinks it (so no new traversal can
//    reach it). retire() tags it with the global epoch at retire time and pushes
//    it onto the retiring thread's OWN garbage list — single-producer, so the push
//    needs no synchronization.
//  * To reclaim, a thread computes
//        min_active = min over all currently-active threads of their announce
//    (quiescent threads do not constrain anything). Any garbage tagged with an
//    epoch STRICTLY LESS THAN min_active is safe to free.
//
//  WHY THAT BOUND IS CORRECT.  Take a retired node P tagged `g`, and suppose
//  min_active > g at reclaim time. Consider any thread T that might still hold P:
//    (1) T is active now, announcing a_T > g. The global epoch only reaches g+1
//        AFTER P was tagged g, which was AFTER P was unlinked. T announced a_T>=g+1,
//        i.e. it read the global epoch after it had already advanced past g, i.e.
//        T entered its current operation AFTER P became unreachable. A traversal
//        that starts after P is unlinked can never load P. So T does not hold P.
//    (2) T is quiescent (announce==0). It holds nothing.
//    (3) T was active at some epoch <= g. Then min_active <= g, contradicting our
//        premise. So this case cannot occur.
//  Hence no live thread holds P, and freeing it is safe. QED.
//
//  MEMORY ORDERING.  Every atomic here uses seq_cst. With a single total order
//  over all atomic ops, the temporal argument above carries real happens-before
//  weight: T's announce-store (seq_cst) that a reclaimer observes synchronizes the
//  reclaimer's later free with T's earlier reads of P, so the free is ordered
//  after those reads and ThreadSanitizer sees no race. Correctness, not speed, is
//  the priority; relaxing these orderings later is a separate, careful exercise.
// ============================================================================
//
#include <atomic>
#include <cstdint>
#include <vector>
#include <new>
#include "bwtree/node.h"

namespace bwtree {

inline constexpr uint64_t kInactive = 0;   // announce value meaning "not in any epoch"

class EpochManager {
 public:
  // A per-thread handle. `id` indexes this thread's slot. Operations construct an
  // EpochGuard from their Context; retire() routes garbage to ctx.id's local list.
  struct Context {
    EpochManager* mgr = nullptr;
    int           id  = -1;
  };

  // RAII bracket around a single tree operation.
  //   ctor -> enter(): publish the current global epoch (this thread is now active).
  //   dtor -> exit():  publish quiescent, then opportunistically try to reclaim.
  // Holding the guard for the whole operation is exactly what guarantees that any
  // node the operation loads cannot be freed underneath it.
  class EpochGuard {
   public:
    EpochGuard(EpochManager& m, int id) : mgr_(m), id_(id) { mgr_.enter(id_); }
    ~EpochGuard() { mgr_.exit(id_); }
    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;
   private:
    EpochManager& mgr_;
    int           id_;
  };

  explicit EpochManager(int max_threads = kDefaultMaxThreads)
      : slots_(static_cast<size_t>(max_threads)), max_threads_(max_threads) {}

  ~EpochManager() { reclaim_all(); }

  // Register the calling thread once; returns a Context it keeps for its lifetime.
  Context register_thread() {
    int id = next_id_.fetch_add(1, std::memory_order_seq_cst);
    if (id >= max_threads_) {
      // Bounded registry. Raise max_threads in the ctor if you need more workers.
      std::abort();
    }
    slots_[static_cast<size_t>(id)].announce.store(kInactive, std::memory_order_seq_cst);
    return Context{this, id};
  }

  // ---- hot path -----------------------------------------------------------

  // Publish that thread `id` is now active in the current global epoch. The
  // seq_cst store both announces liveness and orders this thread's subsequent
  // mapping-table loads after the announcement, so a reclaimer that misses the
  // announcement also could not have freed anything this op will touch.
  void enter(int id) {
    uint64_t e = global_epoch_.load(std::memory_order_seq_cst);
    slots_[static_cast<size_t>(id)].announce.store(e, std::memory_order_seq_cst);
  }

  // Publish quiescent, then (cheaply) advance the epoch periodically and try to
  // free this thread's own backlog. Reclamation here is what keeps memory bounded
  // during a long run; the final reclaim_all() mops up whatever is left.
  void exit(int id) {
    slots_[static_cast<size_t>(id)].announce.store(kInactive, std::memory_order_seq_cst);
    Slot& s = slots_[static_cast<size_t>(id)];
    if (++s.ops_since_bump >= kBumpInterval) {
      s.ops_since_bump = 0;
      global_epoch_.fetch_add(1, std::memory_order_seq_cst);  // keep epochs moving
    }
    try_reclaim(id);
  }

  // Hand a just-unlinked node to EBR. Single-producer push onto this thread's own
  // list — no lock, no atomics on the list itself. Tagging with the current global
  // epoch (>= this thread's announce) guarantees the node will not be freed until
  // AFTER this very operation finishes, so a thread can never free a node it just
  // retired but might still read within the same op.
  void retire(int id, Node* n) {
    if (n == nullptr) return;
    uint64_t g = global_epoch_.load(std::memory_order_seq_cst);
    slots_[static_cast<size_t>(id)].garbage.push_back(Retired{g, n});
  }

  // Free everything this thread retired whose tag is provably safe (tag < the
  // smallest epoch any thread is currently active in). Quiescent threads do not
  // constrain us. Runs entirely on this thread's own list.
  void try_reclaim(int id) {
    uint64_t safe = min_active_epoch();
    Slot& s = slots_[static_cast<size_t>(id)];
    auto& g = s.garbage;
    size_t w = 0;
    for (size_t r = 0; r < g.size(); ++r) {
      if (g[r].epoch < safe) {
        delete g[r].node;        // provably no live thread can reach this node
      } else {
        g[w++] = g[r];           // not yet safe — keep it for a later pass
      }
    }
    g.resize(w);
  }

  // Called once, after ALL worker threads have joined. With no active threads,
  // min_active_epoch() is +inf, so every retired node is safe to free.
  void reclaim_all() {
    for (auto& s : slots_) {
      for (auto& r : s.garbage) delete r.node;
      s.garbage.clear();
    }
  }

  uint64_t global_epoch() const { return global_epoch_.load(std::memory_order_seq_cst); }

 private:
  struct Retired { uint64_t epoch; Node* node; };

  // Padded to a cache line so neighbouring threads' announce words and garbage
  // lists do not false-share.
  struct alignas(64) Slot {
    std::atomic<uint64_t> announce{kInactive};
    std::vector<Retired>  garbage;
    uint32_t              ops_since_bump = 0;
    char                  pad[64];
  };

  // Smallest epoch announced by any active thread; +inf if all are quiescent.
  uint64_t min_active_epoch() const {
    uint64_t m = std::numeric_limits<uint64_t>::max();
    for (auto& s : slots_) {
      uint64_t a = s.announce.load(std::memory_order_seq_cst);
      if (a != kInactive && a < m) m = a;
    }
    return m;
  }

  static constexpr int      kDefaultMaxThreads = 256;
  static constexpr uint32_t kBumpInterval      = 16;  // ops between global-epoch bumps

  std::vector<Slot>     slots_;
  std::atomic<uint64_t> global_epoch_{1};
  std::atomic<int>      next_id_{0};
  int                   max_threads_;
};

}  // namespace bwtree
