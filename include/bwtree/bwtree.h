#pragma once
//
// bwtree.h — public API.
//
// Concurrency model: a bounded set of threads, each of which calls
// register_thread() ONCE and reuses the returned Context for every operation.
// The Context carries the thread's epoch slot (for liveness) and its private
// garbage list (for reclamation). insert / lookup / remove are latch-free for
// concurrent execution; see the per-method notes and src/bwtree.cpp.
//
#include <vector>
#include "bwtree/types.h"
#include "bwtree/node.h"
#include "bwtree/mapping_table.h"
#include "bwtree/epoch.h"

namespace bwtree {

class BwTree {
 public:
  using Context = EpochManager::Context;

  explicit BwTree(size_t mapping_capacity = (size_t{1} << 18));
  ~BwTree();

  BwTree(const BwTree&) = delete;
  BwTree& operator=(const BwTree&) = delete;

  // Each worker thread calls this once and keeps the Context for its lifetime.
  Context register_thread() { return epoch_.register_thread(); }

  // Insert or overwrite key->value. Latch-free: prepends a delta via CAS with a
  // retry loop, then performs best-effort structural maintenance (consolidate /
  // split). Safe under concurrent insert + lookup.
  void insert(const Context& ctx, Key key, Value value);

  // Point lookup. Latch-free and read-only (writes nothing to shared nodes).
  // Returns true and sets *out_value if present.
  bool lookup(const Context& ctx, Key key, Value* out_value);

  // Delete key (tombstone delta). The delta path is latch-free like insert; node
  // MERGES on underflow are deliberately NOT implemented latch-free — see the big
  // comment in src/bwtree.cpp. Underflowing pages simply stay (possibly empty)
  // until a future consolidation drops the tombstones.
  bool remove(const Context& ctx, Key key);

  // --- diagnostics / single-threaded helpers --------------------------------
  // Force-collapse a page's delta chain. Exposed for tests; normally triggered
  // automatically by insert/remove maintenance.
  void debug_consolidate(const Context& ctx, PID pid);
  // Reclaim everything retired so far. Call only when no other thread is active
  // (e.g. between phases of a single-threaded test, or after all joins).
  void quiesce_and_reclaim() { epoch_.reclaim_all(); }
  // Count live keys by walking the structure (single-threaded use only).
  size_t debug_count_keys();

 private:
  // ---- traversal -----------------------------------------------------------
  // Descend from the root to the leaf page responsible for `key`, following side
  // links across in-progress splits and cooperatively posting missing index terms
  // to parents. If `path` is non-null it is filled root-first with the PIDs
  // visited (used by maintenance to know each node's parent).
  PID descend_to_leaf(const Context& ctx, Key key, std::vector<PID>* path);

  // Within page `head`, decide whether `key` has moved right (this page split and
  // `key >= split_key`). Returns the sibling PID to follow, or kInvalidPID if the
  // key still belongs here.
  PID split_redirect(Node* head, Key key) const;

  // Route `key` to a child PID within an inner page (index-entry deltas override
  // the base, newest first; then the base separators).
  PID route_inner(Node* head, Key key) const;

  // ---- structural maintenance (best effort) --------------------------------
  void maintain_node(const Context& ctx, PID pid, PID parent);
  Node* consolidate_leaf(const Context& ctx, PID pid);
  Node* consolidate_inner(const Context& ctx, PID pid);
  void  split_leaf(const Context& ctx, PID pid, PID parent, LeafNode* base);
  void  split_inner(const Context& ctx, PID pid, PID parent, InnerNode* base);
  void  try_post_index_term(const Context& ctx, PID parent,
                            Key sep_low, bool has_sep_high, Key sep_high, PID child);
  void  grow_root(const Context& ctx, PID old_root,
                  Key split_key, bool has_high, Key high_key, PID sibling);

  // Retire every node in the chain [head .. base] (used after consolidation).
  void retire_chain(const Context& ctx, Node* head);

  MappingTable          table_;
  EpochManager          epoch_;
  std::atomic<PID>      root_pid_;
};

}  // namespace bwtree
