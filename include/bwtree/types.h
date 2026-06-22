#pragma once
//
// types.h — fundamental types, constants, and node tags for the Bw-tree.
//
// Keys/values are concrete int64 to keep the concurrency logic front-and-center
// for the write-up. Templatizing on <Key,Value,Comparator> is mechanical and does
// not change anything about the latch-free protocol or the epoch reclamation.
//
#include <cstdint>
#include <limits>

namespace bwtree {

using Key   = int64_t;
using Value = int64_t;

// A PID (page id) is a *logical* pointer. The mapping table translates a PID into
// the *physical* head pointer of that page's delta chain. All sibling / child
// links in the tree are PIDs, never raw Node*. This indirection is what lets us
// swap a page's physical representation (consolidate, split) with a single CAS
// without having to find-and-fix pointers held by other nodes.
using PID = uint64_t;

inline constexpr PID kInvalidPID = std::numeric_limits<PID>::max();

// ---- Tuning knobs ---------------------------------------------------------
// Kept small so that even modest test sizes exercise consolidation, leaf splits,
// inner splits, and root growth. Bump them up for throughput once correctness is
// established.
inline constexpr uint32_t kConsolidateThreshold = 8;   // delta-chain length trigger
inline constexpr uint32_t kLeafSplitThreshold   = 32;  // max entries in a leaf base
inline constexpr uint32_t kInnerSplitThreshold  = 16;  // max separators in an inner base

// Distinguishes the physical kind of a Node. Both base nodes and delta records
// are Nodes so the mapping table can hold a single std::atomic<Node*> per slot.
enum class NodeType : uint8_t {
  LeafBase,        // sorted [Key -> Value] run, the consolidated leaf
  InnerBase,       // sorted separators -> child PIDs, the consolidated inner

  LeafInsert,      // delta: insert/overwrite (Key -> Value) on a leaf
  LeafDelete,      // delta: tombstone (Key) on a leaf

  Split,           // delta: this page split; [split_key, high) now lives at `sibling`
  IndexEntry,      // delta (inner): route keys in [sep_low, sep_high) -> `child`
};

}  // namespace bwtree
