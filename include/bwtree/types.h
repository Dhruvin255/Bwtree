#pragma once
// types.h - fundamental types and constants for the Bw-tree.
//
// Keys and values are int64 to keep the concurrency logic front and center.
// Templatizing on <Key,Value,Comparator> is mechanical and changes nothing
// about the latch-free protocol or epoch reclamation.

#include <cstdint>
#include <limits>

namespace bwtree {

using Key   = int64_t;
using Value = int64_t;

// A PID is a logical page id. The mapping table translates it to the physical
// head pointer of that page's delta chain. All sibling/child links use PIDs
// instead of raw pointers, so swapping a page's form (consolidate or split)
// takes a single CAS on the mapping table slot, nothing else.
using PID = uint64_t;

inline constexpr PID kInvalidPID = std::numeric_limits<PID>::max();

// Kept small so even modest test sizes exercise consolidation, leaf splits,
// inner splits, and root growth.
inline constexpr uint32_t kConsolidateThreshold = 8;
inline constexpr uint32_t kLeafSplitThreshold   = 32;
inline constexpr uint32_t kInnerSplitThreshold  = 16;

// Physical kind of a Node. Base nodes and delta records both live behind a
// single atomic<Node*> in the mapping table.
enum class NodeType : uint8_t {
  LeafBase,
  InnerBase,

  LeafInsert,
  LeafDelete,

  Split,
  IndexEntry,
};

}  // namespace bwtree