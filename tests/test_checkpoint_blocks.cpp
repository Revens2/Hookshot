// checkpoint_blocks::next (gpu/vk/checkpoint_blocks.h): a recording thread's marker numbers, reserved 64 at a time.
// tests/test_checkpoint_blocks.py
#include "../gpu/vk/checkpoint_blocks.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

int main() {
    // (one thread: count up from 1, a block of 64 reserved every 64 markers)
    {
        std::atomic<uint32_t> n{0};
        uint32_t cp_next = 0, cp_end = 0;
        bool counts_up = true;
        for (uint32_t i = 1; i <= 100; i++) counts_up = counts_up && checkpoint_blocks::next(cp_next, cp_end, n) == i;
        expect(counts_up, "one thread: the markers count up from 1");
        expect(n.load() == 128, "one thread, 100 markers: two blocks of 64 reserved");
    }
    // (five threads, 100 markers each: every marker unique, a block one thread's own, a thread's own markers count up
    // inside each block it reserved)
    {
        const int threads = 5, markers = 100;
        std::atomic<uint32_t> n{0};
        std::vector<std::vector<uint32_t>> got(threads);
        std::vector<std::thread> workers;
        for (int t = 0; t < threads; t++)
            workers.emplace_back([&, t] {
                uint32_t cp_next = 0, cp_end = 0;
                for (int i = 0; i < markers; i++) got[t].push_back(checkpoint_blocks::next(cp_next, cp_end, n));
            });
        for (auto& w : workers) w.join();
        std::vector<uint32_t> all;
        for (const auto& g : got) all.insert(all.end(), g.begin(), g.end());
        std::sort(all.begin(), all.end());
        expect(all.front() >= 1, "no marker number 0");
        expect(std::adjacent_find(all.begin(), all.end()) == all.end(), "every marker number unique");
        expect(n.load() == uint32_t((markers + 63) / 64 * 64 * threads), "each thread reserved two blocks of 64");
        std::vector<int> owner((all.back() - 1) / 64 + 1, -1);   // (which thread reserved each block)
        bool alone = true, counts_up = true;
        for (int t = 0; t < threads; t++) {
            const auto& g = got[t];
            for (uint32_t s : g) {
                const int b = (int)((s - 1) / 64);
                if (owner[b] != -1 && owner[b] != t) alone = false;
                owner[b] = t;
            }
            // (within a block a thread's numbers count up; the jump to its next block is the reservation's own)
            for (size_t i = 1; i < g.size(); i++) counts_up = counts_up && ((g[i] == g[i - 1] + 1) || (g[i - 1] % 64 == 0));
        }
        expect(alone, "a block of 64 markers is one thread's own");
        expect(counts_up, "a thread's own markers count up in each block");
    }
    // (a block left over while the other recorders reserved more than a ring: given up for a fresh one, so the numbers
    // it has left cannot write over an entry a newer marker uses. A block still inside the ring is kept and used up.)
    {
        std::atomic<uint32_t> n{0};
        uint32_t a_next = 0, a_end = 0, b_next = 0, b_end = 0;
        expect(checkpoint_blocks::next(a_next, a_end, n) == 1, "resume: the thread's first block starts at 1");
        for (int i = 0; i < 200000; i++) checkpoint_blocks::next(b_next, b_end, n);   // (other recorders, over two rings)
        const uint32_t reserved = n.load();
        const uint32_t s = checkpoint_blocks::next(a_next, a_end, n);
        expect(s == reserved + 1, "resume: a block a ring behind is given up, a fresh one reserved");
        expect(a_next == s && a_end == s + 63, "resume: the fresh block is 64 numbers");
        uint32_t c_next = 0, c_end = 0;
        std::atomic<uint32_t> m{0};
        expect(checkpoint_blocks::next(c_next, c_end, m) == 1, "keep: the thread's first block starts at 1");
        for (int i = 0; i < 10; i++) checkpoint_blocks::next(b_next, b_end, m);
        expect(checkpoint_blocks::next(c_next, c_end, m) == 2, "keep: a block inside the ring is used up");
        // (the edge: a number one ring behind n is the last safe one; two behind, the block is given up)
        uint32_t d_next = 0, d_end = 0;
        std::atomic<uint32_t> r{0};
        expect(checkpoint_blocks::next(d_next, d_end, r) == 1, "edge: the thread's first block starts at 1");
        while (r.load() < (1u << 16) - 1) checkpoint_blocks::next(b_next, b_end, r);   // (n - d_next == ring - 1)
        expect(checkpoint_blocks::next(d_next, d_end, r) == 2, "edge: inside the ring by one, the block is kept");
        while (r.load() < (1u << 16) + 1) checkpoint_blocks::next(b_next, b_end, r);   // (n - d_next == ring + 1)
        const uint32_t edge_reserved = r.load();
        expect(checkpoint_blocks::next(d_next, d_end, r) == edge_reserved + 1,
               "edge: one past the ring, the block is given up for a fresh one");
    }
    if (failures) return EXIT_FAILURE;
    std::printf("checkpoint blocks: all checks passed\n");
    return EXIT_SUCCESS;
}