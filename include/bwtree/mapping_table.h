#pragma once
//
// mapping_table.h — the indirection layer.
//
// Slot i holds std::atomic<Node*>, the physical head of logical page i's delta
// chain. Every page mutation is published by a compare_exchange on the relevant
// slot. Readers do a seq_cst load. There are NO locks on the read or write path —
// the atomic slot IS the synchronization point, and the immutability of published
// nodes (see node.h) means nothing else has to be synchronized.
//
// Why indirection at all: because sibling/child links are PIDs, swapping a page's
// physical form (consolidate or split) is a single CAS on one slot. No other node
// holds a raw pointer to the page, so nothing else needs fixing up. That is the
// structural trick that makes single-CAS page replacement possible.
//
#include <atomic>
#include <cstdlib>
#include "bwtree/types.h"
#include "bwtree/node.h"

namespace bwtree {

class MappingTable {
 public:
  explicit MappingTable(size_t capacity)
      : capacity_(capacity),
        slots_(new std::atomic<Node*>[capacity]) {
    for (size_t i = 0; i < capacity_; ++i)
      slots_[i].store(nullptr, std::memory_order_relaxed);
  }
  ~MappingTable() { delete[] slots_; }

  size_t capacity() const { return capacity_; }

  // Allocate a fresh logical page id. A simple monotonic bump allocator; aborted
  // splits "leak" their slot (set back to nullptr, never reused). A free list to
  // recycle ids is a straightforward extension and not needed for correctness.
  PID allocate() {
    PID id = next_pid_.fetch_add(1, std::memory_order_seq_cst);
    if (id >= capacity_) std::abort();  // raise capacity in the BwTree ctor
    return id;
  }

  // Acquire-or-stronger load of a page head.
  Node* load(PID pid) const {
    return slots_[pid].load(std::memory_order_seq_cst);
  }

  // Plain store — only legitimate for a freshly allocated PID that no other thread
  // can observe yet (e.g. a split's brand-new sibling before it is linked in).
  void store(PID pid, Node* n) {
    slots_[pid].store(n, std::memory_order_seq_cst);
  }

  // The one and only way to publish an update to an already-visible page.
  // Returns true on success. On failure, `expected` is updated to the current head
  // (standard compare_exchange_strong semantics) so the caller's retry loop can
  // rebuild against the value it lost the race to.
  bool cas(PID pid, Node*& expected, Node* desired) {
    return slots_[pid].compare_exchange_strong(
        expected, desired,
        std::memory_order_seq_cst,   // success: publish `desired` (release side)
        std::memory_order_seq_cst);  // failure: reload `expected` (acquire side)
  }

  // Highest allocated PID + 1; used by the destructor to walk live chains.
  PID watermark() const { return next_pid_.load(std::memory_order_seq_cst); }

 private:
  size_t                 capacity_;
  std::atomic<Node*>*    slots_;
  std::atomic<PID>       next_pid_{0};
};

}  // namespace bwtree
