// The page write generations against a brute-force table (gpu/vk/gen_blocks.h): single pages, ranges and whole regions
// stamped at increasing generations, then written_since and written_runs asked at random ranges and generations, with
// the 64-page block skip on and off, which must give the same answers. A negative control forgets one block update and
// checks that the block skip then loses the written page: the parity checks below are what catches such a bug.
//
// The window straddles a chunk boundary (and the lazy allocation of the second chunk), and its pages cross the region
// and block edges (512 and 64 pages), and the start is not a block boundary. No GPU, no game.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "gen_blocks.h"

namespace {

constexpr uint64_t kBasePage = (1ull << 20) - 32;   // (mid-block, 32 pages before a chunk boundary)
constexpr uint64_t kWindowPages = gen_blocks::kChunkPages + 64;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)

struct Case {
    gen_blocks::Table table;
    std::vector<uint32_t> brute = std::vector<uint32_t>(kWindowPages, 0u);
    uint32_t gen = 0;

    void stamp(uint64_t first, uint64_t last) {
        CHECK(first <= last && last < kWindowPages);
        ++gen;
        table.stamp_range(kBasePage + first, kBasePage + last, gen);
        for (uint64_t p = first; p <= last; p++) brute[p] = gen;
    }
};

bool brute_since(const std::vector<uint32_t>& brute, uint64_t first, uint64_t last, uint32_t gen) {
    for (uint64_t p = first; p <= last; p++)
        if (brute[p] > gen) return true;
    return false;
}

std::vector<std::pair<uint64_t, uint64_t>> brute_runs(const std::vector<uint32_t>& brute, uint64_t first, uint64_t last,
                                                      uint64_t guest, uint64_t size, uint32_t gen) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    for (uint64_t p = first; p <= last; p++) {
        if (brute[p] <= gen) continue;
        const uint64_t at = (kBasePage + p) << gen_blocks::kPageShift;
        const uint64_t a = std::max(at, guest) - guest;
        const uint64_t b = std::min(at + 4096u, guest + size) - guest;
        if (!out.empty() && out.back().second == a) out.back().second = b;
        else out.push_back({a, b});
    }
    return out;
}

void check_window(Case& c) {
    for (uint64_t p = 0; p < kWindowPages; p++) {
        const uint32_t got = c.table.page_gen(kBasePage + p);
        if (got != c.brute[p]) {
            std::fprintf(stderr, "FAIL: page %llu: gen %u, brute force %u\n", (unsigned long long)p, got, c.brute[p]);
            std::exit(1);
        }
    }
}

void check_queries(Case& c, std::mt19937_64& rng, int count) {
    const uint64_t gbase = kBasePage << gen_blocks::kPageShift;
    for (int i = 0; i < count; i++) {
        const uint64_t first = 1 + rng() % (kWindowPages - 3);
        const uint64_t last = first + rng() % (kWindowPages - first - 1);
        const uint32_t synced = c.gen ? (uint32_t)(rng() % (c.gen + 1)) : 0u;
        const uint64_t off = rng() % 4096;
        const uint64_t guest = gbase + (first << gen_blocks::kPageShift) + off;
        const uint64_t size = ((last - first + 1) << gen_blocks::kPageShift) - off;
        const bool want = brute_since(c.brute, first, last, synced);
        CHECK(c.table.written_since(guest, size, synced, false) == want);
        CHECK(c.table.written_since(guest, size, synced, true) == want);
        auto want_runs = brute_runs(c.brute, first, last, guest, size, synced);
        auto got_blocks = std::vector<std::pair<uint64_t, uint64_t>>{};
        c.table.written_runs(guest, size, synced, got_blocks, true);
        if (got_blocks != want_runs) {
            std::fprintf(stderr, "FAIL: written_runs(blocks): pages %llu..%llu, gen %u, guest %#llx +%#llx\n",
                         (unsigned long long)first, (unsigned long long)last, synced,
                         (unsigned long long)guest, (unsigned long long)size);
            for (const auto& r : got_blocks)
                std::fprintf(stderr, "  got  %llu..%llu\n", (unsigned long long)r.first, (unsigned long long)r.second);
            for (const auto& r : want_runs)
                std::fprintf(stderr, "  want %llu..%llu\n", (unsigned long long)r.first, (unsigned long long)r.second);
            std::exit(1);
        }
        auto got_plain = std::vector<std::pair<uint64_t, uint64_t>>{};
        c.table.written_runs(guest, size, synced, got_plain, false);
        CHECK(got_plain == want_runs);
        CHECK(c.table.written_since(guest, 0, synced, true) == false);
        auto empty = std::vector<std::pair<uint64_t, uint64_t>>{};
        c.table.written_runs(guest, 0, synced, empty, true);
        CHECK(empty.empty());
    }
}

void check_forgotten_block() {
    gen_blocks::Table t;
    const uint64_t page = kBasePage + 3 * gen_blocks::kBlockPages + 7;   // (a block past the first)
    t.stamp_page(page, 5);
    const uint64_t block_begin = page - page % gen_blocks::kBlockPages;
    const uint64_t guest = block_begin << gen_blocks::kPageShift;   // (the whole block, so the skip can trigger)
    const uint64_t size = gen_blocks::kBlockPages << gen_blocks::kPageShift;
    CHECK(t.written_since(guest, size, 4, true));
    CHECK(t.written_since(guest, size, 4, false));
    t.blocks_[page / gen_blocks::kChunkPages][(page % gen_blocks::kChunkPages) / gen_blocks::kBlockPages] = 0;   // (the update forgotten)
    CHECK(!t.written_since(guest, size, 4, true));   // (the block's summary is load-bearing: a stale one loses the page)
    CHECK(t.written_since(guest, size, 4, false));
    t.blocks_[page / gen_blocks::kChunkPages][(page % gen_blocks::kChunkPages) / gen_blocks::kBlockPages] = 5;
    CHECK(t.written_since(guest, size, 4, true));
}

} // namespace

int main() {
    std::mt19937_64 rng(0xC0FFEEull);
    Case c;
    for (int round = 0; round < 40; round++) {
        for (int i = 0; i < 20; i++) {
            const uint64_t kind = rng() % 4;
            const uint64_t first = rng() % kWindowPages;
            if (kind == 0) {
                c.stamp(first, first);
            } else if (kind == 1) {
                c.stamp(first, std::min(kWindowPages - 1, first + rng() % 200));
            } else if (kind == 2) {   // (a whole 2 MB region)
                const uint64_t a = first - first % gen_blocks::kRegionPages;
                c.stamp(a, a + gen_blocks::kRegionPages - 1);
            } else {   // (a range across regions and blocks)
                c.stamp(first, std::min(kWindowPages - 1, first + rng() % 5000));
            }
        }
        check_window(c);
        check_queries(c, rng, 40);
    }
    CHECK(c.gen > 0);
    check_forgotten_block();
    std::printf("PASS: generations, written_since and written_runs match the brute-force table, blocks on and off\n");
    return 0;
}
