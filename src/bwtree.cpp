//
// bwtree.cpp — latch-free Bw-tree: insert / lookup / remove, with consolidation,
// two-phase splits, B-link side-link traversal, and epoch-based reclamation.
//
// ============================================================================
//  HOW CONCURRENCY CORRECTNESS IS ACHIEVED  (read this first)
// ============================================================================
//  1. INDIRECTION. Every sibling/child link is a PID, not a raw Node*. A page's
//     physical chain head lives in one std::atomic<Node*> mapping-table slot.
//     Replacing a page's representation (consolidate or split) is therefore a
//     SINGLE compare_exchange on that one slot — nobody else holds a pointer that
//     needs fixing.
//
//  2. IMMUTABILITY. A published node is never mutated. Updates only (a) prepend a
//     fresh delta whose `next` is the old head, or (b) build a fresh base and CAS
//     it in. So readers never race a writer on a node's fields; the only sync
//     point is the atomic slot, and the seq_cst load/CAS on it supply all the
//     happens-before we need.
//
//  3. CAS RETRY. Every writer reads the current head, builds a new (private,
//     not-yet-visible) node pointing at it, then CASes. If it loses the race it
//     deletes its private node and rebuilds against the new head. The losing
//     thread's node was never visible, so deleting it is always safe.
//
//  4. B-LINK SIDE LINKS. A split publishes the sibling and a side link in ONE CAS
//     (phase 1) BEFORE telling the parent (phase 2). Until the parent learns the
//     new boundary, a searcher that lands on the now-too-small page simply follows
//     the side link. Reads stay correct the instant phase 1 lands; phase 2 is a
//     pure routing optimization.
//
//  5. EPOCH-BASED RECLAMATION. A node unlinked by a CAS is *retired*, not freed,
//     and only reclaimed once no thread can still be mid-traversal on it. This is
//     what prevents the use-after-free that immediate free() would cause. EBR also
//     incidentally rules out ABA on the slot CAS: a retired node is never
//     re-installed and cannot be freed+reallocated while any thread holds the
//     epoch guard, so a slot never silently returns to a stale-but-reused pointer.
//     The full argument lives in epoch.h.
//
// ============================================================================
//  MERGES ARE DELIBERATELY NOT LATCH-FREE  (scope decision, 1-week project)
// ============================================================================
//  remove() posts a tombstone (LeafDeleteDelta) on the latch-free write path just
//  like insert. But node MERGING on underflow — the inverse of split, requiring a
//  RemoveNodeDelta + NodeMergeDelta + a parent IndexTermDeleteDelta coordinated
//  across three pages — is NOT implemented concurrently here. It is the trickiest
//  part of the Bw-tree to get right under contention and is out of scope for one
//  week. Consequence: an underflowing (even empty) leaf simply stays in place; its
//  tombstones are dropped the next time the page consolidates, but the page itself
//  is not reclaimed and not folded into a neighbor. This costs some space, never
//  correctness. A single-threaded compaction pass could be added later to merge
//  underfull pages while no other thread is active.
// ============================================================================
//
#include "bwtree/bwtree.h"

#include <map>
#include <vector>

namespace bwtree {

// ---------------------------------------------------------------------------
//  file-local helpers
// ---------------------------------------------------------------------------
namespace {

// First (topmost) split delta in a chain, or nullptr. There is at most one,
// because we only ever split a freshly consolidated bare base.
SplitDelta* find_split_delta(Node* head) {
  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::Split) return static_cast<SplitDelta*>(n);
    if (is_base(n)) break;
  }
  return nullptr;
}

// Leftmost child of an inner page. Splits only ever peel keys off the RIGHT and
// index-entry deltas only ever add separators at/after an existing boundary, so
// the global leftmost child is always the base's leftmost_child.
PID inner_leftmost(Node* head) {
  for (Node* n = head; n != nullptr; n = n->next) {
    if (n->type == NodeType::InnerBase) return static_cast<InnerNode*>(n)->leftmost_child;
  }
  return kInvalidPID;
}

// The "logical contents" of a leaf page, obtained by folding its delta chain over
// its base. Shared by consolidate_leaf (to build the new base) and
// debug_count_keys (to enumerate live keys). Applying a fold here — rather than
// trusting the base alone — is what lets a reader see an as-yet-unconsolidated
// page correctly.
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

  // Seed from the base. If a split delta is present, keys at/above split_key now
  // belong to the sibling and must NOT be reported here.
  for (auto& kv : base->data) {
    if (has_split && kv.first >= split_key) continue;
    f.live[kv.first] = kv.second;
  }
  // Replay deltas oldest -> newest so later writes win.
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

// ---------------------------------------------------------------------------
//  construction / destruction
// ---------------------------------------------------------------------------

BwTree::BwTree(size_t mapping_capacity)
    : table_(mapping_capacity), epoch_(), root_pid_(kInvalidPID) {
  // The tree starts as a single empty leaf that owns the entire key space
  // (no low fence, no high fence).
  PID root = table_.allocate();
  LeafNode* leaf = new LeafNode();
  table_.store(root, leaf);
  root_pid_.store(root, std::memory_order_seq_cst);
}

BwTree::~BwTree() {
  // Free every LIVE chain by walking each allocated slot head -> base. Sibling and
  // child links are PIDs (not Node*), so each physical node belongs to exactly one
  // slot's chain and is deleted exactly once. Retired (already-unlinked) nodes are
  // disjoint from all live chains and are freed separately below — no double free.
  PID wm = table_.watermark();
  for (PID pid = 0; pid < wm; ++pid) {
    Node* n = table_.load(pid);
    while (n != nullptr) {
      Node* nxt = n->next;
      delete n;
      n = nxt;
    }
  }
  epoch_.reclaim_all();   // all workers have joined => everything retired is safe
}

// ---------------------------------------------------------------------------
//  traversal primitives
// ---------------------------------------------------------------------------

// B-link redirect test. Returns the sibling PID to follow if `key` has moved past
// this page's high fence (either because an in-flight SplitDelta says so, or
// because a consolidated base already carries the post-split high + side link), or
// kInvalidPID if `key` still belongs on this page.
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

// Route `key` to a child within an inner page: newest index-entry delta that
// brackets `key` wins; otherwise the base separators decide.
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

// Descend from the root to the leaf responsible for `key`. Follows side links
// across in-progress splits at every level and cooperatively posts any missing
// index term to the parent so later searchers route directly. If `path` is given
// it is filled root-first with the PID actually responsible for `key` at each
// level (used afterwards by maintenance to learn each node's parent).
PID BwTree::descend_to_leaf(const Context& ctx, Key key, std::vector<PID>* path) {
  if (path) path->clear();
  PID cur    = root_pid_.load(std::memory_order_seq_cst);
  PID parent = kInvalidPID;   // parent PID of cur's level, for help-posting

  while (true) {
    Node* head = table_.load(cur);

    // (1) Did this page split with `key` now to the right? Take the side link.
    PID sib = split_redirect(head, key);
    if (sib != kInvalidPID) {
      // Cooperative phase-2: if we can still see the split delta, push its index
      // term to the parent. Best-effort — pure optimization, never correctness.
      if (parent != kInvalidPID) {
        if (SplitDelta* sd = find_split_delta(head)) {
          try_post_index_term(ctx, parent, sd->split_key,
                              sd->has_high, sd->high_key, sd->sibling);
        }
      }
      cur = sib;        // move sideways; same level, same parent; re-evaluate
      continue;         // (deliberately do NOT push the too-small page onto path)
    }

    // (2) Reached the leaf level.
    if (head->leaf_level) {
      if (path) path->push_back(cur);
      return cur;
    }

    // (3) Inner page that owns `key` at this level: record it, descend.
    if (path) path->push_back(cur);
    parent = cur;
    cur    = route_inner(head, key);
  }
}

// ---------------------------------------------------------------------------
//  public operations
// ---------------------------------------------------------------------------

bool BwTree::lookup(const Context& ctx, Key key, Value* out_value) {
  // The epoch guard is held for the WHOLE operation: any node we load below cannot
  // be freed under us until the guard is dropped at function exit.
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

  // ---- CAS retry loop: prepend a LeafInsertDelta -------------------------
  while (true) {
    Node* head = table_.load(leaf);

    // If the leaf split since descent and `key` moved right, follow the side link
    // and retry on the sibling. Keep `path` pointing at the leaf we actually land
    // on, so maintenance below targets the right page.
    PID sib = split_redirect(head, key);
    if (sib != kInvalidPID) {
      leaf = sib;
      if (!path.empty()) path.back() = sib;
      continue;
    }

    // Build the new head pointing at the current head. It is PRIVATE — no other
    // thread can see it until the CAS publishes it.
    Node* delta = new LeafInsertDelta(key, value, head->chain_len + 1, head);

    Node* expected = head;
    if (table_.cas(leaf, expected, delta)) break;   // published; done

    // Lost the race: someone changed the head first. Our delta was never visible,
    // so we delete it and rebuild against whatever is there now. (`expected` was
    // refreshed by compare_exchange, but we simply re-load at the top of the loop.)
    delete delta;
  }

  // ---- best-effort structural maintenance, bottom-up ---------------------
  // path is root-first; process leaf -> ... -> root. Each node's parent is the
  // preceding path entry (root has none).
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

    // Decide presence against THIS head (consistent with what we will CAS onto).
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

// ---------------------------------------------------------------------------
//  structural maintenance (all best-effort; correctness never depends on it)
// ---------------------------------------------------------------------------

void BwTree::maintain_node(const Context& ctx, PID pid, PID parent) {
  Node* head = table_.load(pid);
  if (head == nullptr) return;

  // (A) Long delta chain -> consolidate into a fresh base.
  if (head->chain_len >= kConsolidateThreshold) {
    head = head->leaf_level ? consolidate_leaf(ctx, pid)
                            : consolidate_inner(ctx, pid);
    if (head == nullptr) return;   // lost the consolidation race; let another op retry
  }

  // (B) Oversized base with no in-flight split -> split. We only split when the
  // head is a bare base, which guarantees we never stack two SplitDeltas on a page.
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

  // Publish the new base. On success the entire old chain (head .. base inclusive)
  // is unlinked and handed to EBR — NOT freed now, because other threads may still
  // be walking it.
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

  // Fold the inner chain: collect deltas + base, honoring an in-flight split.
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
  // Precondition (guaranteed by maintain_node): `base` is believed to be the
  // current head and is a bare LeafBase. Split its sorted run in half.
  size_t n = base->data.size();
  if (n < 2) return;
  size_t mid = n / 2;
  Key split_key = base->data[mid].first;     // first key that migrates right

  // Build the sibling (upper half). It inherits the old high fence and the old
  // right-neighbor side link.
  LeafNode* sib = new LeafNode();
  sib->data.assign(base->data.begin() + mid, base->data.end());
  sib->has_low   = true;            sib->low_key  = split_key;
  sib->has_high  = base->has_high;  sib->high_key = base->high_key;
  sib->side_link = base->side_link;

  // Publish the sibling into a fresh slot. A plain store is correct: this PID is
  // brand new and unreachable until phase 1 links it via the SplitDelta.
  PID q = table_.allocate();
  table_.store(q, sib);

  // ---- PHASE 1 ----------------------------------------------------------
  // Atomically cap this page at split_key AND expose the side link to the sibling,
  // by CASing a SplitDelta onto the page. The instant this lands, every reader is
  // correct: keys >= split_key follow the side link to `sib`.
  Node* sd = new SplitDelta(split_key, q, base->has_high, base->high_key,
                            /*leaf=*/true, /*len=*/1, /*next=*/base);
  Node* expected = base;
  if (!table_.cas(pid, expected, sd)) {
    // Lost the race (head changed). The sibling was never linked from anywhere,
    // so roll it back. Its slot leaks (never reused) — acceptable; a PID free list
    // is a later optimization.
    table_.store(q, nullptr);
    delete sd;
    delete sib;
    return;
  }

  // ---- PHASE 2 (best effort) -------------------------------------------
  // Tell the parent about the new boundary so future searchers route directly to
  // `sib` instead of taking the side link. If this page was the root, grow a new
  // root above it instead.
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
  (void)ctx;   // no node is retired here (a lost CAS deletes an unpublished delta)

  // Bounded CAS retry. This is purely an optimization, so we do NOT loop forever:
  // if we keep losing, the side link still routes searchers correctly.
  for (int attempt = 0; attempt < 4; ++attempt) {
    Node* head = table_.load(parent);
    if (head == nullptr) return;

    // Skip if an equivalent term is already posted (by us earlier, or by a
    // cooperative searcher), whether still a delta or already folded into the base.
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

    // If the parent ITSELF has since split and this separator now belongs to the
    // parent's sibling, don't post here. Bail; a later traversal will help-post at
    // the correct parent. (Side links keep everything correct meanwhile.)
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

  // New inner root: keys < split_key -> old_root; keys >= split_key -> sibling.
  InnerNode* nr = new InnerNode();
  nr->leftmost_child = old_root;
  nr->seps.emplace_back(split_key, sibling);
  // The root spans the whole key space: no fences, no side link.

  PID new_root_pid = table_.allocate();
  table_.store(new_root_pid, nr);

  // Swing the root pointer. Only the thread that won this page's phase-1 split
  // reaches here for `old_root`, and the root pointer is changed by nobody else,
  // so this CAS effectively always succeeds. If it somehow fails, drop our new
  // root: the split's sibling is still reachable via the side link, so results
  // stay correct.
  PID expected = old_root;
  if (!root_pid_.compare_exchange_strong(expected, new_root_pid,
                                         std::memory_order_seq_cst,
                                         std::memory_order_seq_cst)) {
    table_.store(new_root_pid, nullptr);
    delete nr;
  }
}

void BwTree::retire_chain(const Context& ctx, Node* head) {
  // Hand head .. base (inclusive) to EBR. These were just unlinked by a successful
  // CAS, so no NEW traversal can reach them; any in-flight reader is protected by
  // its epoch guard until it finishes, after which reclamation frees them.
  Node* n = head;
  while (n != nullptr) {
    Node* nxt = n->next;
    epoch_.retire(ctx.id, n);
    n = nxt;
  }
}

// ---------------------------------------------------------------------------
//  diagnostics (single-threaded use)
// ---------------------------------------------------------------------------

void BwTree::debug_consolidate(const Context& ctx, PID pid) {
  EpochManager::EpochGuard guard(epoch_, ctx.id);
  Node* head = table_.load(pid);
  if (head == nullptr) return;
  if (head->leaf_level) consolidate_leaf(ctx, pid);
  else                  consolidate_inner(ctx, pid);
}

size_t BwTree::debug_count_keys() {
  // Walk to the leftmost leaf, then follow leaf-level side links to the right,
  // folding each page and counting its live keys. fold_leaf truncates at any
  // pending split, so keys that have migrated to a sibling are counted exactly
  // once (on the sibling), never twice.
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
