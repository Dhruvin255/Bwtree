#pragma once
//
// node.h — the node hierarchy.
//
// A logical page is a singly-linked "delta chain": zero or more delta records
// prepended on top of exactly one base node. The mapping table points at the
// HEAD of the chain (the newest delta). Traversal walks head -> ... -> base.
//
// CRUCIAL INVARIANT (this is what makes the whole thing latch-free and race-free):
//   *** A node is immutable once it is published. ***
//   We never write to a node's fields after it has been installed into the
//   mapping table. Updates only ever (a) prepend a brand-new delta whose `next`
//   points at the old head, or (b) build a brand-new base and CAS it in. Because
//   published nodes are read-only, concurrent readers never race with a writer on
//   a node's fields, and the only synchronization needed is on the atomic head
//   pointer in the mapping table. Field reads therefore need no atomics; the
//   acquire load of the head pointer + release CAS that published it provide the
//   necessary happens-before for everything reachable from that head.
//
#include <vector>
#include <utility>
#include "bwtree/types.h"

namespace bwtree {

// Base class. `next` chains deltas toward the base (nullptr at the base).
struct Node {
  NodeType type;
  bool     leaf_level;   // true if this node belongs to a leaf page (delta or base)
  uint32_t chain_len;    // number of deltas above the base; base has 0. Drives consolidation.
  Node*    next;         // next (older) record in the chain; nullptr iff this is a base

  Node(NodeType t, bool leaf, uint32_t len, Node* nxt)
      : type(t), leaf_level(leaf), chain_len(len), next(nxt) {}
  virtual ~Node() = default;   // virtual so `delete (Node*)p` frees the right subtype
};

inline bool is_base(const Node* n) {
  return n->type == NodeType::LeafBase || n->type == NodeType::InnerBase;
}

// ---- Base nodes -----------------------------------------------------------

// A leaf base owns a sorted, de-duplicated run of (key,value) pairs and the
// logical key range [low_key, high_key) it is responsible for, plus a side link
// (right sibling PID at the leaf level).
struct LeafNode : Node {
  std::vector<std::pair<Key, Value>> data;  // sorted ascending by key, unique keys
  bool has_low = false;  Key low_key = 0;
  bool has_high = false; Key high_key = 0;  // exclusive upper bound
  PID  side_link = kInvalidPID;

  LeafNode() : Node(NodeType::LeafBase, /*leaf=*/true, /*len=*/0, /*next=*/nullptr) {}
};

// An inner base routes keys to child PIDs. `leftmost_child` handles keys strictly
// below seps[0].first; otherwise the child is the last sep whose key <= search key.
struct InnerNode : Node {
  PID leftmost_child = kInvalidPID;
  std::vector<std::pair<Key, PID>> seps;  // sorted ascending by key; (boundary_key -> child)
  bool has_low = false;  Key low_key = 0;
  bool has_high = false; Key high_key = 0;
  PID  side_link = kInvalidPID;

  InnerNode() : Node(NodeType::InnerBase, /*leaf=*/false, /*len=*/0, /*next=*/nullptr) {}
};

// ---- Delta records --------------------------------------------------------

struct LeafInsertDelta : Node {
  Key   key;
  Value value;
  LeafInsertDelta(Key k, Value v, uint32_t len, Node* nxt)
      : Node(NodeType::LeafInsert, true, len, nxt), key(k), value(v) {}
};

struct LeafDeleteDelta : Node {
  Key key;
  LeafDeleteDelta(Key k, uint32_t len, Node* nxt)
      : Node(NodeType::LeafDelete, true, len, nxt), key(k) {}
};

// Installed by the FIRST phase of a split. Logically truncates this page to
// [.., split_key) and redirects keys in [split_key, high_key) to `sibling`. It
// doubles as the side link a concurrent searcher follows when the parent index
// term has not been posted yet. Carries the old high so the matching index term
// can be reconstructed by anyone (the splitter or a cooperative searcher).
struct SplitDelta : Node {
  Key  split_key;
  PID  sibling;
  bool has_high;   // old high of the splitting page (false => +infinity)
  Key  high_key;
  SplitDelta(Key sk, PID sib, bool hh, Key hk, bool leaf, uint32_t len, Node* nxt)
      : Node(NodeType::Split, leaf, len, nxt),
        split_key(sk), sibling(sib), has_high(hh), high_key(hk) {}
};

// Installed by the SECOND phase of a split, onto the PARENT inner page. Routes
// keys in [sep_low, sep_high) directly to the new `child`, so searchers no longer
// have to take the side link.
struct IndexEntryDelta : Node {
  Key  sep_low;
  bool has_sep_high;
  Key  sep_high;
  PID  child;
  IndexEntryDelta(Key lo, bool hh, Key hi, PID c, uint32_t len, Node* nxt)
      : Node(NodeType::IndexEntry, /*leaf=*/false, len, nxt),
        sep_low(lo), has_sep_high(hh), sep_high(hi), child(c) {}
};

}  // namespace bwtree
