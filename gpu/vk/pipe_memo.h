// (hookshot) What a graphics pipeline was built from, and the slot a thread's memo of its recent pipelines keeps it in
// (vk_draw.cpp: graphics_pipeline).
// tests/test_pipe_memo.py
#pragma once
#include <cstdint>

namespace pipe_memo {

// The state a pipeline entry was built from: what pipe_same used to compare, one argument at a time, on every draw.
// key is the program's own (Program::key): a freed program's address can come back as another program's, and the address
// alone would take the new one for the remembered one.
struct From {
    uint64_t prog = 0, variant = 0, key = 0;
    int topo = 0, n = 0, nvattr = 0;
    uint32_t samples = 0, depth = 0, blend_en = 0, chmask = 0, poly = 0, clamp = 0, a2c = 0;
    uint32_t formats[8]{}, vattr[16]{}, div[16]{};
    uint64_t blend[8]{};
};

// Which of a thread's memo slots a pipeline goes to: direct-mapped by its program's address (the low four bits are the
// type's, all zero for one), which variant of the program, the topology class and the number of colour targets.
inline unsigned slot(uintptr_t prog, uint64_t variant, int topo, int n) {
    return (unsigned)(((prog >> 4) ^ variant ^ ((uint64_t)topo << 3) ^ (uint64_t)n) & 31);
}

// The full compare a memo hit needs: everything baked into the pipeline.
inline bool same(const From& a, const From& b) {
    if (a.prog != b.prog || a.variant != b.variant || a.topo != b.topo || a.samples != b.samples || a.n != b.n ||
        a.depth != b.depth || a.nvattr != b.nvattr || a.blend_en != b.blend_en || a.chmask != b.chmask || a.poly != b.poly ||
        a.clamp != b.clamp || a.a2c != b.a2c)
        return false;
    for (int i = 0; i < 8; i++)
        if (a.formats[i] != b.formats[i] || a.blend[i] != b.blend[i]) return false;
    for (int i = 0; i < 16; i++)
        if (a.vattr[i] != b.vattr[i] || a.div[i] != b.div[i]) return false;
    return true;
}

// Whether the memo holds this pipeline: the full compare and the program's key.
inline bool hit(const From& want, const From& memo) {
    return want.key == memo.key && same(want, memo);
}

} // namespace pipe_memo
