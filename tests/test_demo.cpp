// test_demo.cpp
// Run this during the presentation. Each section maps to one talking point.

#include "bwtree/bwtree.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

using namespace bwtree;

static std::mutex io_mtx;

static void line(const std::string& s) {
    std::lock_guard<std::mutex> lk(io_mtx);
    std::cout << s << "\n";
}

static void check(bool ok, const std::string& label) {
    std::string mark = ok ? "  \u2713" : "  \u2717 FAIL";
    line("    " + label + mark);
    if (!ok) std::exit(1);
}

static void section(int n, const std::string& title) {
    std::cout << "\n[" << n << "] " << title << "\n";
}

int main() {
    std::cout << "=== Bw-Tree Demo ===\n";

    // -------------------------------------------------------------------------
    // 1. Basic ops: insert, lookup, remove
    // -------------------------------------------------------------------------
    section(1, "Basic operations");
    {
        BwTree tree;
        auto ctx = tree.register_thread();

        tree.insert(ctx, 1, 10);
        tree.insert(ctx, 2, 20);
        tree.insert(ctx, 3, 30);
        line("    inserted keys 1, 2, 3");

        Value v = 0;
        check(tree.lookup(ctx, 2, &v) && v == 20, "lookup(2)        -> 20");
        check(!tree.lookup(ctx, 9, &v),            "lookup(9)        -> not found");

        tree.remove(ctx, 1);
        check(!tree.lookup(ctx, 1, &v),            "remove(1); lookup(1) -> not found  (tombstone delta)");
    }

    // -------------------------------------------------------------------------
    // 2. Delta chain growing and automatic consolidation
    //    This is the core structural mechanism: watch chain_len climb to the
    //    threshold, then snap back to 0 when maintain_node consolidates.
    // -------------------------------------------------------------------------
    section(2, "Delta chain  (consolidate threshold = " +
               std::to_string(kConsolidateThreshold) + ")");
    {
        BwTree tree;
        auto ctx = tree.register_thread();
        PID root = tree.debug_root_pid();   // PID 0 on a fresh tree; won't change here

        for (int k = 1; k <= 10; ++k) {
            tree.insert(ctx, k, k * 10);
            uint32_t len = tree.debug_chain_len(root);
            std::ostringstream oss;
            oss << "    after insert(" << std::setw(2) << k << ")  chain_len = " << len;
            if (len == 0 && k > 1) oss << "  <- threshold hit, consolidated";
            line(oss.str());
        }
    }

    // -------------------------------------------------------------------------
    // 3. Leaf splits
    //    Insert enough keys to push a leaf past kLeafSplitThreshold.
    //    The tree grows invisibly; debug_count_keys walks the full leaf level.
    // -------------------------------------------------------------------------
    section(3, "Leaf splits  (split threshold = " +
               std::to_string(kLeafSplitThreshold) + ")");
    {
        BwTree tree;
        auto ctx = tree.register_thread();

        line("    inserting 200 keys...");
        for (int k = 0; k < 200; ++k) tree.insert(ctx, k, k);

        tree.quiesce_and_reclaim();
        size_t n = tree.debug_count_keys();
        check(n == 200, "live keys across all pages: " + std::to_string(n) + " (expected 200)");
    }

    // -------------------------------------------------------------------------
    // 4. Concurrent inserts
    //    4 threads, disjoint key ranges, no coordination except the tree itself.
    //    Final count must be exactly nthreads * per.
    // -------------------------------------------------------------------------
    section(4, "Concurrent  (4 threads x 250 keys = 1000 total)");
    {
        BwTree tree;
        constexpr int nthreads = 4, per = 250;

        std::vector<BwTree::Context> ctxs;
        for (int t = 0; t < nthreads; ++t) ctxs.push_back(tree.register_thread());

        std::vector<std::thread> workers;
        for (int t = 0; t < nthreads; ++t) {
            workers.emplace_back([&, t]() {
                for (int k = t * per; k < (t + 1) * per; ++k)
                    tree.insert(ctxs[t], k, k);

                std::ostringstream oss;
                oss << "    thread " << t << " done: inserted ["
                    << std::setw(4) << t * per << ", "
                    << std::setw(4) << (t + 1) * per << ")";
                line(oss.str());
            });
        }
        for (auto& w : workers) w.join();
        tree.quiesce_and_reclaim();

        size_t n = tree.debug_count_keys();
        check(n == nthreads * per,
              "live keys in tree: " + std::to_string(n) +
              " (expected " + std::to_string(nthreads * per) + ")");
    }

    std::cout << "\n=== all checks passed ===\n";
    return 0;
}