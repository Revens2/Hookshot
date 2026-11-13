// Whether a texture uploaded from guest memory that the game has finalized has been left behind (vk_texture.cpp's eviction
// pass). tests/test_finalized_upload.py
#pragma once
#include <cstdint>
#include <cstdlib>

#include "compat/hookshot_env.h"

namespace finalized_upload {

// (hookshot) the eviction pass let a texture go only over its budget, 6 GB on a 12 GB card, or when the GPU had drawn over
// it. So the textures of the areas left behind, which the game finalized, stayed until the cache was full: on an RTX 5070,
// Hyrule Field -> Kakariko -> the graveyard, 4.8-5.8 GB of textures cached while the last 8 presents used 0.2-0.3 GB. With the
// render targets, mirrors and imports beside them the blocks reached what the driver allows, and the GPU's frame went from
// 3.5 ms to 19-30 ms. A texture uploaded from guest memory has its texels there: made again, it is uploaded again from there,
// as after any eviction. Not one the GPU drew (its image can be the only copy: keep_rendered.h and the finalized ones kept)
struct State {
    bool gpu_written;          // the GPU rendered, copied or stored into it
    bool texels_in_memory;     // its image holds what guest memory holds (uploaded from it, or written back): Texture::texels_in_memory
    uint64_t finalized_frame;  // the present the game finalized it in (0: not finalized)
    uint64_t last_frame;       // the present it was last used in
};

// Presents a finalized upload stays unused before it goes: RX_VK_FINALIZED_UPLOADS=<presents> (1800 by default, ~20-30 s;
// 0/off/false/no: never, as before; a number under 60, or on: the default). Long enough that the terrain's chunks finalized and
// made again as Link walks to and fro keep theirs
constexpr uint64_t kDefaultPresents = 1800;
inline uint64_t presents_from_env(const char* e) {
    if (!e) return kDefaultPresents;
    if (env_value_off(e)) return 0;
    const long v = std::strtol(e, nullptr, 10);
    return v >= 60 ? (uint64_t)v : kDefaultPresents;
}

// The game sampled it again after finalizing it (more than a few presents later: the frames in flight when it was finalized)
// keeps it: that is a texture made again over the same storage, still in use
inline bool left_behind(const State& s, uint64_t frame_no, uint64_t presents) {
    if (!presents || s.gpu_written || !s.texels_in_memory || !s.finalized_frame) return false;
    if (s.last_frame > s.finalized_frame + 8) return false;
    return frame_no > s.last_frame && frame_no - s.last_frame >= presents;
}

// What one pass lets go of at most, so a whole area's textures do not go in one burst of destroys
constexpr uint64_t kPassBytes = 256ull << 20;

// Whether a texture still fits in a pass that has let go of used bytes: hard, so one bigger than the whole cap (a course map,
// 349 MB) is not taken alone and the pass never goes over it. That one waits for the budget's rule
inline bool fits_in_pass(uint64_t used, uint64_t bytes) { return bytes <= kPassBytes && used <= kPassBytes - bytes; }

} // namespace finalized_upload
