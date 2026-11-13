// The page write generations behind the mirror syncs and the texture checks (vk_memory.cpp).
//
// Every write to guest memory raises its pages' generation, and the checks that ask "was any of this written since
// generation X" (`mirror_sync`, `memory_written_since`, `memory_written_runs`) walk the written ranges. Two levels kept
// those walks short already: a per-page generation in chunks of 64K pages, made when a page of the chunk is first
// stamped, and the newest generation of each 2 MB region (512 pages), which skips a clean region whole. A region with
// one dirty page made every check of it probe all 512; a 64-page (256 KB) block level skips its clean blocks.
//
// Invariant: a block's generation is at least the generation of every page in it, which holds because the only stamp of
// a page (stamp_page) raises the page's own generation and its block's together. A whole-region stamp (region_full)
// covers all 512 pages and is checked beside the block; it need not raise the blocks. Generations only grow, so a block
// no newer than a check's generation holds no page the check has to see.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace gen_blocks {

constexpr uint64_t kPageShift = 12;
constexpr uint64_t kChunkPages = 1u << 16;   // (64K pages: 256 MB of guest memory)
constexpr uint64_t kRegionPages = 512;       // (2 MB)
constexpr uint64_t kBlockPages = 64;         // (256 KB: the write-back diff's shadow granularity)
constexpr uint64_t kBlocksPerChunk = kChunkPages / kBlockPages;
constexpr uint64_t kPages = 1ull << 27;      // (512 GB of guest memory)

inline uint32_t load_u32(uint32_t& x) {
    return std::atomic_ref<uint32_t>(x).load(std::memory_order_acquire);
}
inline void store_u32(uint32_t& x, uint32_t v) {
    std::atomic_ref<uint32_t>(x).store(v, std::memory_order_release);
}

/// The tables. The member arrays are public so a check can corrupt one on purpose (the negative control: a forgotten
/// block update must lose a written page); production code goes through the methods.
class Table {
public:
    Table() : pages_(kPages / kChunkPages), regions_(kPages / kRegionPages), full_(kPages / kRegionPages),
              blocks_(kPages / kChunkPages) {}

    /// The newest generation page p was written at: its own, or a whole-region stamp's over it
    uint32_t page_gen(uint64_t p) {
        const auto& c = pages_[p / kChunkPages];
        return std::max(c ? load_u32(c[p % kChunkPages]) : 0u, load_u32(full_[p / kRegionPages]));
    }
    /// The newest generation any page of the 64-page block holding p was stamped at (a whole-region stamp need not
    /// count here: region_full is checked beside it)
    uint32_t block_gen(uint64_t p) {
        const auto& b = blocks_[p / kChunkPages];
        return b ? load_u32(b[(p % kChunkPages) / kBlockPages]) : 0u;
    }
    /// The newest generation any page of 2 MB region r was stamped at (a skip's summary)
    uint32_t region_gen(uint64_t r) {
        return load_u32(regions_[r]);
    }
    /// The generation every page of region r was stamped at, at least (a whole-region stamp)
    uint32_t region_full_gen(uint64_t r) {
        return load_u32(full_[r]);
    }

    /// Page p was written at generation g
    void stamp_page(uint64_t p, uint32_t g) {
        if (p >= kPages) return;
        auto& c = pages_[p / kChunkPages];
        if (!c) {
            c.reset(new uint32_t[kChunkPages]);
            std::fill(c.get(), c.get() + kChunkPages, 0u);
        }
        store_u32(c[p % kChunkPages], g);
        auto& b = blocks_[p / kChunkPages];
        if (!b) {
            b.reset(new uint32_t[kBlocksPerChunk]);
            std::fill(b.get(), b.get() + kBlocksPerChunk, 0u);
        }
        auto& block = b[(p % kChunkPages) / kBlockPages];
        if (g > load_u32(block)) store_u32(block, g);
        auto& region = regions_[p / kRegionPages];
        if (g > load_u32(region)) store_u32(region, g);
    }

    /// Pages first_page..last_page (inclusive) were written at generation g: whole regions through region_full, the
    /// rest a page at a time
    void stamp_range(uint64_t first_page, uint64_t last_page, uint32_t g) {
        if (first_page > last_page || first_page >= kPages) return;
        last_page = std::min(last_page, kPages - 1);
        for (uint64_t p = first_page; p <= last_page;) {
            if (p % kRegionPages == 0 && p + kRegionPages - 1 <= last_page) {   // (a whole region)
                store_u32(full_[p / kRegionPages], g);
                auto& region = regions_[p / kRegionPages];
                if (g > load_u32(region)) store_u32(region, g);
                p += kRegionPages;
                continue;
            }
            stamp_page(p++, g);
        }
    }

    /// Whether [guest, guest + size) holds a page written at a generation newer than gen. blocks: also skip its clean
    /// 64-page blocks (a clean layout's cost, not a different answer)
    bool written_since(uint64_t guest, uint64_t size, uint32_t gen, bool blocks) {
        if (!size) return false;
        const uint64_t p0 = guest >> kPageShift, p1 = std::min((guest + size - 1) >> kPageShift, kPages - 1);
        for (uint64_t r = p0 / kRegionPages; r <= p1 / kRegionPages; r++) {
            if (region_gen(r) <= gen) continue;
            if (region_full_gen(r) > gen) return true;
            const uint64_t a = std::max(p0, r * kRegionPages), b = std::min(p1, r * kRegionPages + kRegionPages - 1);
            for (uint64_t p = a; p <= b; p++) {
                if (blocks && p % kBlockPages == 0 && p + kBlockPages - 1 <= b &&
                    std::max(block_gen(p), region_full_gen(r)) <= gen) {
                    p += kBlockPages - 1;
                    continue;
                }
                if (page_gen(p) > gen) return true;
            }
        }
        return false;
    }

    /// The runs of [guest, guest + size) that a page written after gen touches, as (offset, end) from guest, merged
    /// where contiguous. blocks: as written_since's
    void written_runs(uint64_t guest, uint64_t size, uint32_t gen, std::vector<std::pair<uint64_t, uint64_t>>& out,
                      bool blocks) {
        if (!size) return;
        const uint64_t p0 = guest >> kPageShift, p1 = std::min((guest + size - 1) >> kPageShift, kPages - 1);
        for (uint64_t r = p0 / kRegionPages; r <= p1 / kRegionPages; r++) {
            if (region_gen(r) <= gen) continue;
            const uint64_t a = std::max(p0, r * kRegionPages), b = std::min(p1, r * kRegionPages + kRegionPages - 1);
            for (uint64_t p = a; p <= b; p++) {
                if (blocks && p % kBlockPages == 0 && p + kBlockPages - 1 <= b &&
                    std::max(block_gen(p), region_full_gen(r)) <= gen) {
                    p += kBlockPages - 1;
                    continue;
                }
                if (page_gen(p) <= gen) continue;
                const uint64_t lo = std::max(p << kPageShift, guest) - guest;
                const uint64_t hi = std::min((p + 1) << kPageShift, guest + size) - guest;
                if (!out.empty() && out.back().second == lo) out.back().second = hi;
                else out.push_back({lo, hi});
            }
        }
    }

    // (public for the negative control: a forgotten block update must lose a written page)
    std::vector<std::unique_ptr<uint32_t[]>> pages_;
    std::vector<uint32_t> regions_;
    std::vector<uint32_t> full_;
    std::vector<std::unique_ptr<uint32_t[]>> blocks_;
};

} // namespace gen_blocks
