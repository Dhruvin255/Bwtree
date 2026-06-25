// bwtree.cpp - latch-free Bw-tree: insert, lookup, remove, with consolidation,
// two-phase splits, B-link side-link traversal, and epoch-based reclamation.
//
// How correctness works:
//   1. Indirection. Every sibling/child link is a PID. Replacing a page's
//      chain head is a single compare_exchange on one mapping-table slot.
//   2. Immutability. Published nodes are never mutated. Updates prepend a
//      fresh delta or CAS in a fresh base. Readers never race with writers.
//   3. CAS retry. Writers build a private node, then CAS. On failure they
//      delete it and rebuild against the new head.
//   4. B-link side links. A split publishes the sibling in one CAS (phase 1)
//      before telling the parent (phase 2). Searchers follow the side link
//      until the parent is updated.
//   5. Epoch reclamation. Unlinked nodes are retired and only freed once no
//      thread can still be traversing them.
//
// Note on merges: remove() posts a tombstone delta on the latch-free path
// just like insert. Node merging on underflow is not implemented; it requires
// coordinating three pages and is out of scope. Underflowing pages stay in
// place and their tombstones are dropped on the next consolidation.

#include "bwtree/bwtree.h"

#include <map>
#include <vector>

namespace bwtree {

// File-local helpers
namespace {

// Returns the first split delta in a chain, or nullptr.
SplitDelta* find_split_delta(Node* head) {
  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::Split) return static_cast<SplitDelta*>(n);
    if (is_base(n)) break;
  }
  return nullptr;
}

// Leftmost child of an inner page; always lives on the base node.
PID inner_leftmost(Node* head) {
  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::InnerBase) return static_cast<InnerNode*>(n)->leftmost_child;
  }
  return kInvalidPID;
}

// Logical contents of a leaf page, folded from its delta chain over its base.
// Used by consolidate_leaf and debug_count_keys.
struct LeafFold {
  std::map<Key, Value> live;
  bool has_low = false;  Key low_key = 0;
  bool has_high = false; Key high_key = 0;
  PID  side_link = kInvalidPID;
};

bool fold_leaf(Node* head, LeafFold& f) {
  std::vector<Node*> deltas;          // head .. (just above base), newest first
  LeafNode* base = nullptr;
  bool has_split = false;
  Key  split_key = 0;
  PID  sd_sib = kInvalidPID;

  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::LeafBase) { base = static_cast<LeafNode*>(n); break; }
    if (n->type == NodeType::Split) {
      has_split = true;
      auto* sd = static_cast<SplitDelta*>(n);
      split_key = sd->split_key;
      sd_sib    = sd->sibling;
    }
    deltas.push_back(n);
  }
  if (base == nullptr) return false;

  // Seed from the base; skip keys that moved to the sibling on a split.
  for (auto& kv : base->data) {
    if (has_split && kv.first >= split_key) continue;
    f.live[kv.first] = kv.second;
  }
  // Replay deltas oldest to newest so later writes win.
  for (auto it = deltas.rbegin(); it != deltas.rend(); ++it) {
    Node* d = *it;
    if (d->type == NodeType::LeafInsert) {
      auto* x = static_cast<LeafInsertDelta*>(d);
      if (!(has_split && x->key >= split_key)) f.live[x->key] = x->value;
    } else if (d->type == NodeType::LeafDelete) {
      auto* x = static_cast<LeafDeleteDelta*>(d);
      f.live.erase(x->key);
    }
    // Split deltas are accounted for via the truncation above.
  }

  f.has_low = base->has_low;
  f.low_key = base->low_key;
  if (has_split) {                    // fold the split: high becomes split_key,
    f.has_high  = true;               // side link becomes the sibling
    f.high_key  = split_key;
    f.side_link = sd_sib;
  } else {
    f.has_high  = base->has_high;
    f.high_key  = base->high_key;
    f.side_link = base->side_link;
  }
  return true;
}

}  // namespace

// Construction / destruction

BwTree::BwTree(size_t mapping_capacity)
    : table_(mapping_capacity), epoch_(), root_pid_(kInvalidPID) {
  // Start with a single empty leaf owning the entire key space.
  PID root = table_.allocate();
  LeafNode* leaf = new LeafNode();
  table_.store(root, leaf);
  root_pid_.store(root, std::memory_order_seq_cst);
}

BwTree::~BwTree() {
  // Walk every allocated slot and free the chain. Each physical node belongs
  // to exactly one slot so nothing gets double-freed.
  PID wm = table_.watermark();
  for (PID pid = 0; pid < wm; ++pid) {
    Node* n = table_.load(pid);
    while (n != nullptr) {
      Node* nxt = n->next;
      delete n;
      n = nxt;
    }
  }
  epoch_.reclaim_all();   // all workers have joined, everything retired is safe
}

// Traversal primitives

// Returns the sibling PID to follow if `key` has moved past this page's high
// fence (via an in-flight SplitDelta or a consolidated base with a side link),
// or kInvalidPID if the key still belongs here.
PID BwTree::split_redirect(Node* head, Key key) const {
  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::Split) {
      auto* sd = static_cast<SplitDelta*>(n);
      if (key >= sd->split_key) return sd->sibling;   // moved to the sibling
      // else: still below the split key; keep scanning toward the base.
    } else if (n->type == NodeType::LeafBase) {
      auto* b = static_cast<LeafNode*>(n);
      if (b->has_high && key >= b->high_key && b->side_link != kInvalidPID)
        return b->side_link;                          // folded split: take side link
      return kInvalidPID;
    } else if (n->type == NodeType::InnerBase) {
      auto* b = static_cast<InnerNode*>(n);
      if (b->has_high && key >= b->high_key && b->side_link != kInvalidPID)
        return b->side_link;
      return kInvalidPID;
    }
  }
  return kInvalidPID;
}

// Route `key` to a child within an inner page. Newest index-entry delta that
// brackets the key wins; otherwise the base separators decide.
PID BwTree::route_inner(Node* head, Key key) const {
  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::IndexEntry) {
      auto* ie = static_cast<IndexEntryDelta*>(n);
      if (key >= ie->sep_low && (!ie->has_sep_high || key < ie->sep_high))
        return ie->child;
    } else if (n->type == NodeType::InnerBase) {
      auto* b = static_cast<InnerNode*>(n);
      // Largest separator whose key <= search key; if none, the leftmost child.
      PID child = b->leftmost_child;
      int lo = 0, hi = static_cast<int>(b->seps.size()) - 1, idx = -1;
      while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (b->seps[mid].first <= key) { idx = mid; lo = mid + 1; }
        else                           { hi = mid - 1; }
      }
      if (idx >= 0) child = b->seps[idx].second;
      return child;
    }
    // Split deltas on an inner page are handled by split_redirect during descent.
  }
  return kInvalidPID;   // unreachable on a well-formed inner page
}

// Descend from root to the leaf responsible for `key`. Follows side links
// across in-progress splits at every level and cooperatively posts any missing
// index term to the parent. If `path` is given it is filled root-first.
PID BwTree::descend_to_leaf(const Context& ctx, Key key, std::vector<PID>* path) {
  if (path) path->clear();
  PID cur    = root_pid_.load(std::memory_order_seq_cst);
  PID parent = kInvalidPID;   // parent PID of cur's level, for help-posting

  while (true) {
    Node* head = table_.load(cur);

    // Did this page split with `key` now to the right? Take the side link.
    PID sib = split_redirect(head, key);
    if (sib != kInvalidPID) {
      // Cooperative phase 2: push the split's index term to the parent.
      if (parent != kInvalidPID) {
        if (SplitDelta* sd = find_split_delta(head)) {
          try_post_index_term(ctx, parent, sd->split_key,
                              sd->has_high, sd->high_key, sd->sibling);
        }
      }
      cur = sib;        // move sideways; same level, same parent; re-evaluate
      continue;         // (deliberately do NOT push the too-small page onto path)
    }

    // Reached the leaf level.
    if (head->leaf_level) {
      if (path) path->push_back(cur);
      return cur;
    }

    // Inner page that owns `key` at this level: record it, descend.
    if (path) path->push_back(cur);
    parent = cur;
    cur    = route_inner(head, key);
  }
}

// Public operations

bool BwTree::lookup(const Context& ctx, Key key, Value* out_value) {
  // Epoch guard held for the whole operation; no node we load can be freed
  // until we return.
  EpochManager::EpochGuard guard(epoch_, ctx.id);

  PID leaf = descend_to_leaf(ctx, key, nullptr);

  // The leaf may split again between descent and our read; chase side links.
  while (true) {
    Node* head = table_.load(leaf);
    PID sib = split_redirect(head, key);
    if (sib != kInvalidPID) { leaf = sib; continue; }

    // Scan newest-first; the first record that decides `key` wins.
    for (Node* n = head; n != nullptr; n = n->next) {
      switch (n->type) {
        case NodeType::LeafInsert: {
          auto* d = static_cast<LeafInsertDelta*>(n);
          if (d->key == key) { if (out_value) *out_value = d->value; return true; }
          break;
        }
        case NodeType::LeafDelete: {
          auto* d = static_cast<LeafDeleteDelta*>(n);
          if (d->key == key) return false;   // tombstone: definitively absent
          break;
        }
        case NodeType::LeafBase: {
          auto* b = static_cast<LeafNode*>(n);
          int lo = 0, hi = static_cast<int>(b->data.size()) - 1;
          while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if      (b->data[mid].first == key) {
              if (out_value) *out_value = b->data[mid].second;
              return true;
            } else if (b->data[mid].first < key) lo = mid + 1;
            else                                  hi = mid - 1;
          }
          return false;                       // not in the base => absent
        }
        default: break;                       // Split delta: handled above
      }
    }
    return false;
  }
}

void BwTree::insert(const Context& ctx, Key key, Value value) {
  EpochManager::EpochGuard guard(epoch_, ctx.id);

  std::vector<PID> path;
  PID leaf = descend_to_leaf(ctx, key, &path);

  // CAS retry loop: prepend a LeafInsertDelta.
  while (true) {
    Node* head = table_.load(leaf);

    // If the leaf split since descent and `key` moved right, follow the side link.
    PID sib = split_redirect(head, key);
    if (sib != kInvalidPID) {
      leaf = sib;
      if (!path.empty()) path.back() = sib;
      continue;
    }

    // Build the new head pointing at the current head. It is private until
    // the CAS publishes it.
    Node* delta = new LeafInsertDelta(key, value, head->chain_len + 1, head);

    Node* expected = head;
    if (table_.cas(leaf, expected, delta)) break;   // published; done

    // Lost the race; our delta was never visible, so delete it and retry.
    delete delta;
  }

  // Best-effort structural maintenance, bottom-up.
  // path is root-first; process leaf to root.
  for (size_t i = path.size(); i-- > 0;) {
    PID node   = path[i];
    PID parent = (i == 0) ? kInvalidPID : path[i - 1];
    maintain_node(ctx, node, parent);
  }
}

bool BwTree::remove(const Context& ctx, Key key) {
  EpochManager::EpochGuard guard(epoch_, ctx.id);

  std::vector<PID> path;
  PID leaf = descend_to_leaf(ctx, key, &path);

  while (true) {
    Node* head = table_.load(leaf);

    PID sib = split_redirect(head, key);
    if (sib != kInvalidPID) {
      leaf = sib;
      if (!path.empty()) path.back() = sib;
      continue;
    }

    // Check presence against the current head.
    bool present = false;
    for (Node* n = head; n != nullptr; n = n->next) {
      if (n->type == NodeType::LeafInsert) {
        auto* d = static_cast<LeafInsertDelta*>(n);
        if (d->key == key) { present = true; break; }
      } else if (n->type == NodeType::LeafDelete) {
        auto* d = static_cast<LeafDeleteDelta*>(n);
        if (d->key == key) { present = false; break; }
      } else if (n->type == NodeType::LeafBase) {
        auto* b = static_cast<LeafNode*>(n);
        int lo = 0, hi = static_cast<int>(b->data.size()) - 1;
        present = false;
        while (lo <= hi) {
          int mid = (lo + hi) / 2;
          if      (b->data[mid].first == key) { present = true; break; }
          else if (b->data[mid].first <  key) lo = mid + 1;
          else                                hi = mid - 1;
        }
        break;
      }
    }
    if (!present) return false;   // nothing to delete

    Node* delta = new LeafDeleteDelta(key, head->chain_len + 1, head);
    Node* expected = head;
    if (table_.cas(leaf, expected, delta)) break;   // tombstone published

    delete delta;                 // lost the race; re-evaluate against new head
  }

  for (size_t i = path.size(); i-- > 0;) {
    PID node   = path[i];
    PID parent = (i == 0) ? kInvalidPID : path[i - 1];
    maintain_node(ctx, node, parent);
  }
  return true;
}

// Structural maintenance (all best-effort; correctness never depends on it)

void BwTree::maintain_node(const Context& ctx, PID pid, PID parent) {
  Node* head = table_.load(pid);
  if (head == nullptr) return;

  // Long delta chain: consolidate into a fresh base.
  if (head->chain_len >= kConsolidateThreshold) {
    head = head->leaf_level ? consolidate_leaf(ctx, pid)
                            : consolidate_inner(ctx, pid);
    if (head == nullptr) return;   // lost the race; let another op retry
  }

  // Oversized base with no in-flight split: split it. Only splitting a bare
  // base guarantees we never stack two SplitDeltas on one page.
  if (is_base(head)) {
    if (head->leaf_level) {
      auto* b = static_cast<LeafNode*>(head);
      if (b->data.size() > kLeafSplitThreshold) split_leaf(ctx, pid, parent, b);
    } else {
      auto* b = static_cast<InnerNode*>(head);
      if (b->seps.size() > kInnerSplitThreshold) split_inner(ctx, pid, parent, b);
    }
  }
}

Node* BwTree::consolidate_leaf(const Context& ctx, PID pid) {
  Node* head = table_.load(pid);
  if (head == nullptr) return nullptr;
  if (is_base(head)) return head;   // already consolidated by someone else

  LeafFold f;
  if (!fold_leaf(head, f)) return nullptr;

  LeafNode* nb = new LeafNode();
  nb->data.reserve(f.live.size());
  for (auto& kv : f.live) nb->data.emplace_back(kv.first, kv.second);
  nb->has_low   = f.has_low;   nb->low_key  = f.low_key;
  nb->has_high  = f.has_high;  nb->high_key = f.high_key;
  nb->side_link = f.side_link;
  // (chain_len = 0, next = nullptr already from the LeafNode ctor)

  // Publish the new base. On success the old chain is unlinked and handed
  // to EBR; it will not be freed until no thread can still be walking it.
  Node* expected = head;
  if (table_.cas(pid, expected, nb)) {
    retire_chain(ctx, head);
    return nb;
  }
  delete nb;                         // lost the race; our base was never visible
  return nullptr;
}

Node* BwTree::consolidate_inner(const Context& ctx, PID pid) {
  Node* head = table_.load(pid);
  if (head == nullptr) return nullptr;
  if (is_base(head)) return head;

  // Fold the inner chain, honoring any in-flight split.
  std::vector<Node*> deltas;
  InnerNode* base = nullptr;
  bool has_split = false;
  Key  split_key = 0;
  PID  sd_sib = kInvalidPID;

  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::InnerBase) { base = static_cast<InnerNode*>(n); break; }
    if (n->type == NodeType::Split) {
      has_split = true;
      auto* sd = static_cast<SplitDelta*>(n);
      split_key = sd->split_key;
      sd_sib    = sd->sibling;
    }
    deltas.push_back(n);
  }
  if (base == nullptr) return nullptr;

  std::map<Key, PID> seps;
  for (auto& s : base->seps) {
    if (has_split && s.first >= split_key) continue;   // separator moved to sibling
    seps[s.first] = s.second;
  }
  for (auto it = deltas.rbegin(); it != deltas.rend(); ++it) {
    Node* d = *it;
    if (d->type == NodeType::IndexEntry) {
      auto* ie = static_cast<IndexEntryDelta*>(d);
      if (!(has_split && ie->sep_low >= split_key)) seps[ie->sep_low] = ie->child;
    }
  }

  InnerNode* nb = new InnerNode();
  nb->leftmost_child = base->leftmost_child;
  nb->seps.reserve(seps.size());
  for (auto& s : seps) nb->seps.emplace_back(s.first, s.second);
  nb->has_low = base->has_low; nb->low_key = base->low_key;
  if (has_split) {
    nb->has_high  = true; nb->high_key = split_key; nb->side_link = sd_sib;
  } else {
    nb->has_high  = base->has_high; nb->high_key = base->high_key;
    nb->side_link = base->side_link;
  }

  Node* expected = head;
  if (table_.cas(pid, expected, nb)) {
    retire_chain(ctx, head);
    return nb;
  }
  delete nb;
  return nullptr;
}

void BwTree::split_leaf(const Context& ctx, PID pid, PID parent, LeafNode* base) {
  size_t n = base->data.size();
  if (n < 2) return;
  size_t mid = n / 2;
  Key split_key = base->data[mid].first;     // first key that migrates right

  // Sibling takes the upper half and inherits the old high fence and side link.
  LeafNode* sib = new LeafNode();
  sib->data.assign(base->data.begin() + mid, base->data.end());
  sib->has_low   = true;            sib->low_key  = split_key;
  sib->has_high  = base->has_high;  sib->high_key = base->high_key;
  sib->side_link = base->side_link;

  // Plain store is correct: this PID is brand new and unreachable until
  // phase 1 links it via the SplitDelta.
  PID q = table_.allocate();
  table_.store(q, sib);

  // Phase 1: CAS a SplitDelta onto this page. The instant it lands, searchers
  // with keys >= split_key follow the side link to the sibling.
  Node* sd = new SplitDelta(split_key, q, base->has_high, base->high_key,
                            /*leaf=*/true, /*len=*/1, /*next=*/base);
  Node* expected = base;
  if (!table_.cas(pid, expected, sd)) {
    // Lost the race. The sibling was never linked, so roll it back.
    // Its slot leaks (never reused); a PID free list is a later optimization.
    table_.store(q, nullptr);
    delete sd;
    delete sib;
    return;
  }

  // Phase 2: tell the parent so future searchers route directly to the sibling.
  if (parent == kInvalidPID) {
    grow_root(ctx, pid, split_key, base->has_high, base->high_key, q);
  } else {
    try_post_index_term(ctx, parent, split_key, base->has_high, base->high_key, q);
  }
}

void BwTree::split_inner(const Context& ctx, PID pid, PID parent, InnerNode* base) {
  size_t n = base->seps.size();
  if (n < 2) return;
  size_t mid = n / 2;
  Key split_key = base->seps[mid].first;     // median separator goes up to parent

  // Sibling takes separators (mid, n); the child at `mid` becomes its leftmost,
  // because keys in [seps[mid].first, seps[mid+1].first) route to seps[mid].second,
  // and seps[mid].first == split_key is exactly the sibling's low fence.
  InnerNode* sib = new InnerNode();
  sib->leftmost_child = base->seps[mid].second;
  sib->seps.assign(base->seps.begin() + mid + 1, base->seps.end());
  sib->has_low   = true;            sib->low_key  = split_key;
  sib->has_high  = base->has_high;  sib->high_key = base->high_key;
  sib->side_link = base->side_link;

  PID q = table_.allocate();
  table_.store(q, sib);

  Node* sd = new SplitDelta(split_key, q, base->has_high, base->high_key,
                            /*leaf=*/false, /*len=*/1, /*next=*/base);
  Node* expected = base;
  if (!table_.cas(pid, expected, sd)) {
    table_.store(q, nullptr);
    delete sd;
    delete sib;
    return;
  }

  if (parent == kInvalidPID) {
    grow_root(ctx, pid, split_key, base->has_high, base->high_key, q);
  } else {
    try_post_index_term(ctx, parent, split_key, base->has_high, base->high_key, q);
  }
}

void BwTree::try_post_index_term(const Context& ctx, PID parent,
                                 Key sep_low, bool has_sep_high, Key sep_high,
                                 PID child) {
  (void)ctx;   // no node is retired here

  // Bounded retry; this is purely an optimization. Correctness comes from
  // the side link, so we don't need to spin forever.
  for (int attempt = 0; attempt < 4; ++attempt) {
    Node* head = table_.load(parent);
    if (head == nullptr) return;

    // Skip if an equivalent term is already posted (by us or a cooperative searcher).
    bool present = false;
    for (Node* n = head; n != nullptr; n = n->next) {
      if (n->type == NodeType::IndexEntry) {
        auto* ie = static_cast<IndexEntryDelta*>(n);
        if (ie->sep_low == sep_low && ie->child == child) { present = true; break; }
      } else if (n->type == NodeType::InnerBase) {
        auto* b = static_cast<InnerNode*>(n);
        for (auto& s : b->seps)
          if (s.first == sep_low && s.second == child) { present = true; break; }
        break;
      }
    }
    if (present) return;

    // If the parent itself has since split and this separator now belongs to
    // the parent's sibling, bail; a later traversal will help-post there.
    if (split_redirect(head, sep_low) != kInvalidPID) return;

    Node* ie = new IndexEntryDelta(sep_low, has_sep_high, sep_high, child,
                                   head->chain_len + 1, head);
    Node* expected = head;
    if (table_.cas(parent, expected, ie)) return;   // posted

    delete ie;   // lost the race; retry against the new head
  }
  // Gave up; correctness preserved by the side link.
}

void BwTree::grow_root(const Context& ctx, PID old_root,
                       Key split_key, bool has_high, Key high_key, PID sibling) {
  (void)ctx; (void)has_high; (void)high_key;

  // New root: keys < split_key go to old_root, keys >= split_key go to sibling.
  InnerNode* nr = new InnerNode();
  nr->leftmost_child = old_root;
  nr->seps.emplace_back(split_key, sibling);
  // Root spans the whole key space: no fences, no side link.

  PID new_root_pid = table_.allocate();
  table_.store(new_root_pid, nr);

  // Swing the root pointer. If we somehow lose, the sibling is still reachable
  // via the side link so correctness is preserved.
  PID expected = old_root;
  if (!root_pid_.compare_exchange_strong(expected, new_root_pid,
                                         std::memory_order_seq_cst,
                                         std::memory_order_seq_cst)) {
    table_.store(new_root_pid, nullptr);
    delete nr;
  }
}

void BwTree::retire_chain(const Context& ctx, Node* head) {
  // Hand head through base (inclusive) to EBR. In-flight readers are protected
  // by their epoch guards and will finish before any of these nodes are freed.
  Node* n = head;
  while (n != nullptr) {
    Node* nxt = n->next;
    epoch_.retire(ctx.id, n);
    n = nxt;
  }
}

// Diagnostics (single-threaded use)

void BwTree::debug_consolidate(const Context& ctx, PID pid) {
  EpochManager::EpochGuard guard(epoch_, ctx.id);
  Node* head = table_.load(pid);
  if (head == nullptr) return;
  if (head->leaf_level) consolidate_leaf(ctx, pid);
  else                  consolidate_inner(ctx, pid);
}

size_t BwTree::debug_count_keys() {
  // Walk to the leftmost leaf, then follow side links across all leaves.
  // fold_leaf truncates at any pending split so keys are never counted twice.
  PID cur = root_pid_.load(std::memory_order_seq_cst);
  while (true) {
    Node* head = table_.load(cur);
    if (head == nullptr) return 0;
    if (head->leaf_level) break;
    cur = inner_leftmost(head);
    if (cur == kInvalidPID) return 0;
  }

  size_t count = 0;
  size_t hops  = 0;
  size_t limit = static_cast<size_t>(table_.watermark()) + 1;   // cycle guard
  while (cur != kInvalidPID) {
    Node* head = table_.load(cur);
    if (head == nullptr) break;
    LeafFold f;
    if (!fold_leaf(head, f)) break;
    count += f.live.size();
    cur = f.side_link;
    if (++hops > limit) break;
  }
  return count;
}

}  // namespace bwtree