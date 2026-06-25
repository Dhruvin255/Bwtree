#pragma once
// bwtree.h - public API.
//
// Each thread calls register_thread() once and keeps the returned Context for
// its lifetime. The Context carries the thread's epoch slot and private garbage
// list. insert, lookup, and remove are all latch-free for concurrent execution.

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

  Context register_thread() { return epoch_.register_thread(); }

  // Insert or overwrite key->value. Prepends a delta via CAS with a retry
  // loop, then does best-effort consolidation and splitting.
  void insert(const Context& ctx, Key key, Value value);

  // Point lookup. Read-only; never writes to any shared node.
  bool lookup(const Context& ctx, Key key, Value* out_value);

  // Delete key via tombstone delta. Latch-free on the write path.
  // Node merges on underflow are not implemented; see bwtree.cpp.
  bool remove(const Context& ctx, Key key);

  // Force-collapse a page's delta chain. Exposed for tests.
  void debug_consolidate(const Context& ctx, PID pid);

  // Reclaim all retired nodes. Only safe when no other thread is active.
  void quiesce_and_reclaim() { epoch_.reclaim_all(); }

  // Count live keys by walking the whole structure. Single-threaded only.
  size_t debug_count_keys();

  // Returns the current root PID (changes when the root splits).
  PID debug_root_pid() const { return root_pid_.load(std::memory_order_seq_cst); }

  // Returns the chain_len of the current head of page `pid`.
  // 0 means the page just consolidated down to a clean base.
  uint32_t debug_chain_len(PID pid) const {
    Node* h = table_.load(pid);
    return h ? h->chain_len : 0;
  }

 private:
  // Descend from root to the leaf responsible for `key`. Follows side links
  // across in-progress splits and cooperatively posts any missing index terms
  // to parents. If `path` is non-null it is filled root-first with visited PIDs.
  PID descend_to_leaf(const Context& ctx, Key key, std::vector<PID>* path);

  // Returns the sibling PID to follow if `key` has moved past this page's
  // high fence, or kInvalidPID if the key still belongs here.
  PID split_redirect(Node* head, Key key) const;

  // Route `key` to a child PID within an inner page.
  PID route_inner(Node* head, Key key) const;

  void maintain_node(const Context& ctx, PID pid, PID parent);
  Node* consolidate_leaf(const Context& ctx, PID pid);
  Node* consolidate_inner(const Context& ctx, PID pid);
  void  split_leaf(const Context& ctx, PID pid, PID parent, LeafNode* base);
  void  split_inner(const Context& ctx, PID pid, PID parent, InnerNode* base);
  void  try_post_index_term(const Context& ctx, PID parent,
                            Key sep_low, bool has_sep_high, Key sep_high, PID child);
  void  grow_root(const Context& ctx, PID old_root,
                  Key split_key, bool has_high, Key high_key, PID sibling);

  void retire_chain(const Context& ctx, Node* head);

  MappingTable          table_;
  EpochManager          epoch_;
  std::atomic<PID>      root_pid_;
};

}  // namespace bwtree