/* How full VRAM really is, for the rules that give memory back near the driver's limit (vk_memory.cpp's write-back shadows, and
   the texture budget that leaves the rest of the process its room: vk_texture.cpp's texture_frame). No Vulkan here.
   tests/test_vram_pressure.py */
#pragma once

#include <algorithm>
#include <cstdint>

namespace mkwvk::vram_pressure {

/// What the process really holds: the driver's count less the free space inside VMA's blocks. A destroyed image's or
/// buffer's memory stays in its block, and the driver counts the block whole until VMA frees it
inline uint64_t allocated(uint64_t used, uint64_t block_free) {
    return used - std::min(used, block_free);
}

/// Whether the write-back's shadows are given back: what is really allocated past 95% of what the driver allows, or
/// the driver's own count past 99%. The count alone passes 95% whenever the texture cache is full, with gigabytes of
/// it free space inside the blocks that no eviction gives back.
/// (A shadow over 32 MB is an allocation of its own, not a share of a block: VMA's blocks are 64 MB and anything over
/// half of one gets its own memory. So the shadows going lowers the count by their whole size, and the lines at which
/// they are made again, below, are 2% and 5% under these: with the count's at 98% they came back at once and went again
/// at the next 1%, five to seven times in a five-minute run, each time read whole again)
inline bool shadows_go(uint64_t used, uint64_t allowed, uint64_t block_free) {
    if (!allowed) return false;
    return allocated(used, block_free) > allowed / 100 * 95 || used > allowed / 100 * 99;
}

/// Whether a shadow of n bytes may be made: with it, what is really allocated does not pass 90% nor the driver's own
/// count 97%. The textures come first: their budget gives way past 90% really allocated and is squeezed past a 97%
/// count, and a shadow adds its whole size to both (its memory is its own: the blocks' free space is of no use to it),
/// so one made above these lines could be what starts textures being evicted, which does not lower the count. Not a
/// step lower: the count sits at 95-98% for minutes with a full cache, and at 96% and 89% the shadows that a moment at
/// 99% took were not back by the end of either of two five-minute runs (96 MB and none of ~400, missing in 27 of 101
/// and 9 of 114 windows of 120 presents); at these lines 384 MB in both (missing in 7 of 105 and 10 of 120)
inline bool shadow_fits(uint64_t n, uint64_t used, uint64_t allowed, uint64_t block_free) {
    if (!allowed) return true;
    return allocated(used, block_free) + n <= allowed / 100 * 90 && used + n <= allowed / 100 * 97;
}

/// The textures' budget that leaves the rest of the process's VRAM (the imports, the pools' mirrors, the render targets, the
/// buffers, the pipelines) its room: `resident` is what the process really allocated (allocated(), or with compaction what
/// the textures' pool's free space adds back), so everything but the textures is `resident - total`. The textures' own may
/// then be what is left under 90% of what the driver allows; a budget taken once, at the first look, knows nothing of a
/// rest that grows later (#2156: 8 GB card, a frozen 3.2 GB of textures beside 3.0 GB of other, the driver paged VRAM and
/// a frame took 793 ms). Cards with room keep theirs: 90% of a 12-16 GB card's allowance less 3.5 GB is above the 6 GB the
/// budget is capped at anyway
inline uint64_t texture_budget(uint64_t budget, uint64_t total, uint64_t resident, uint64_t allowed) {
    if (!allowed) return budget;
    const uint64_t other = resident > total ? resident - total : 0;
    const uint64_t room = allowed / 100 * 90;
    return std::min(budget, room > other ? room - other : 0);
}

} // namespace mkwvk::vram_pressure
