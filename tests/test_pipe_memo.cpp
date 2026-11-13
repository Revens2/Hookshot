// pipe_memo (gpu/vk/pipe_memo.h): a thread's memo of its pipelines, hit by its own full compare + the program's key.
// tests/test_pipe_memo.py
#include "../gpu/vk/pipe_memo.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

static pipe_memo::From make(uintptr_t prog, uint64_t key) {
    pipe_memo::From f;
    f.prog = prog;
    f.key = key;
    f.variant = 7;
    f.topo = 3;
    f.n = 2;
    f.nvattr = 5;
    f.samples = 1;
    f.depth = 81;
    f.blend_en = 3;
    f.chmask = 0xF;
    f.poly = 2;
    f.clamp = 0;
    f.a2c = 1;
    for (int i = 0; i < 8; i++) f.formats[i] = (uint32_t)(10 + i), f.blend[i] = 0x11 + i;
    for (int i = 0; i < 16; i++) f.vattr[i] = (uint32_t)(i * 3), f.div[i] = (uint32_t)(i % 4);
    return f;
}

// One thread's memo: lookup is the slot, then pipe_memo::hit; a miss fills the slot (vk_draw.cpp's graphics_pipeline).
struct Memo {
    pipe_memo::From slot[32]{};
    bool on[32]{};
    int hits = 0, misses = 0;
    const pipe_memo::From* find(const pipe_memo::From& want) {
        const unsigned i = pipe_memo::slot(want.prog, want.variant, want.topo, want.n);
        if (on[i] && pipe_memo::hit(want, slot[i])) {
            hits++;
            return &slot[i];
        }
        misses++;
        slot[i] = want;
        on[i] = true;
        return nullptr;
    }
};

int main() {
    const pipe_memo::From a = make(0x10000, 0xA1), b = make(0x10010, 0xB1);
    // (the two slots differ: the alternation below is the memo's job, not the pair fallback's)
    expect(pipe_memo::slot(a.prog, a.variant, a.topo, a.n) != pipe_memo::slot(b.prog, b.variant, b.topo, b.n),
           "two programs: two slots");

    // A B A B ...: the first two fill their slots, the rest hit
    Memo m;
    for (int i = 0; i < 8; i++) {
        const pipe_memo::From& want = (i & 1) ? b : a;
        const pipe_memo::From* hit = m.find(want);
        expect((hit != nullptr) == (i >= 2), "alternating A B: hit after the first two");
    }
    expect(m.hits == 6 && m.misses == 2, "alternating A B: six hits, two misses");
    expect(m.find(make(0x10000, 0xA1)) != nullptr, "a fresh copy of the same state hits");

    // a changed state misses, even with the same address and key
    pipe_memo::From c = a;
    c.blend[0] ^= 1;
    expect(!pipe_memo::same(a, c) && !pipe_memo::hit(a, c), "a changed blend misses");
    c = a;
    c.formats[3] += 1;
    expect(!pipe_memo::same(a, c), "a changed target format misses");
    c = a;
    c.vattr[9] += 1;
    expect(!pipe_memo::same(a, c), "a changed vertex attribute misses");
    c = a;
    c.div[9] += 1;
    expect(!pipe_memo::same(a, c), "a changed divisor misses");

    // a freed program's address reused: the slot and the whole state match, the key does not
    const pipe_memo::From reused = make(0x10000, 0xA2);
    expect(pipe_memo::same(a, reused), "same address and state: pipe_same alone says yes");
    expect(!pipe_memo::hit(a, reused), "same address, another program key: misses");
    Memo m2;
    m2.find(a);
    expect(m2.find(reused) == nullptr, "the reused address takes the memo miss path");

    if (failures) return EXIT_FAILURE;
    std::printf("pipe memo: all checks passed\n");
    return EXIT_SUCCESS;
}
