// The rules that judge VRAM near the limit (gpu/vk/vram_pressure.h): the shadows' and the texture budget's. No game, no GPU.
#include "../gpu/vk/vram_pressure.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)

using mkwvk::vram_pressure::allocated;
using mkwvk::vram_pressure::shadow_fits;
using mkwvk::vram_pressure::shadows_go;
using mkwvk::vram_pressure::texture_budget;

int main() {
    const uint64_t MB = 1 << 20;
    CHECK(allocated(10462 * MB, 3472 * MB) == 6990 * MB);
    CHECK(allocated(100 * MB, 300 * MB) == 0);                // (never below nothing)
    CHECK(allocated(500 * MB, 0) == 500 * MB);
    // (a player's log, an RTX 4070 SUPER with the texture cache full: 10,462 of 10,732 MB used, 97.5% by the count)
    CHECK(!shadows_go(10462 * MB, 10732 * MB, 3472 * MB));    // (3.5 GB of it free inside the blocks: they stay)
    CHECK(!shadows_go(10462 * MB, 10732 * MB, 300 * MB));     // (94.7% really allocated)
    CHECK(shadows_go(10462 * MB, 10732 * MB, 0));             // (97.5% really allocated: they go)
    CHECK(shadows_go(10650 * MB, 10732 * MB, 3472 * MB));     // (the driver's own count past 99%: they go whatever is free)
    CHECK(!shadows_go(10462 * MB, 0, 0));                     // (no budget known: nothing to judge by)
    CHECK(!shadows_go(0, 10732 * MB, 0));
    // (a shadow is made only where, with it, the textures' own lines are not passed: 90% really allocated, a 97% count)
    CHECK(shadow_fits(32 * MB, 9700 * MB, 10732 * MB, 1600 * MB));     // (75.8% really allocated, a 90.7% count)
    CHECK(shadow_fits(32 * MB, 9500 * MB, 10732 * MB, 0));             // (88.8% really allocated with it)
    CHECK(!shadow_fits(32 * MB, 9700 * MB, 10732 * MB, 0));            // (90.7% really allocated with it: the textures' line)
    CHECK(!shadow_fits(32 * MB, 10400 * MB, 10732 * MB, 3472 * MB));   // (the count would pass 97%)
    CHECK(shadow_fits(32 * MB, 10300 * MB, 10732 * MB, 3472 * MB));    // (96.3% with it)
    CHECK(shadow_fits(32 * MB, 10462 * MB, 0, 0));                     // (no budget known)
    // (given back and made again do not chase each other: what goes at 95% really allocated only fits again at 90%)
    const uint64_t allowed = 10000 * MB, shadows = 400 * MB;
    CHECK(shadows_go(9600 * MB, allowed, 0));
    CHECK(!shadow_fits(shadows, 9600 * MB - shadows, allowed, 0));     // (9,200 + 400 MB: back at 96%)
    CHECK(!shadow_fits(shadows, 8700 * MB, allowed, 0));               // (and not at 91% either)
    CHECK(shadow_fits(shadows, 8600 * MB, allowed, 0));
    // (nor by the count: the shadows are allocations of their own, so their going takes their whole size off the count.
    // What goes at a 99% count only fits again at 97%: not at once, and not at the 98% it used to)
    CHECK(shadows_go(9950 * MB, allowed, 3000 * MB));
    CHECK(!shadow_fits(shadows, 9950 * MB - shadows, allowed, 3000 * MB));   // (9,550 + 400 MB: back over 99%)
    CHECK(!shadow_fits(shadows, 9400 * MB, allowed, 3000 * MB));             // (9,800 MB: 98%, where they came back before)
    CHECK(shadow_fits(shadows, 9300 * MB, allowed, 3000 * MB));              // (9,700 MB: 97%)
    CHECK(!shadows_go(9700 * MB, allowed, 3000 * MB));                       // (and there they stay)
    // (the textures' budget leaves the rest of the process's VRAM its room: 90% of the allowance less what it holds.
    // An 8 GB card, 7,000 MB allowed, 5,900 MB really allocated of which the textures are 2,810: the rest at 3,090)
    CHECK(texture_budget(3213 * MB, 2810 * MB, 5900 * MB, 7000 * MB) == 3210 * MB);
    // (the same budget on a card with room: 90% of 15,500 MB less 3,500 MB is above the 6 GB cap the budget has)
    CHECK(texture_budget(6144 * MB, 3000 * MB, 6500 * MB, 15500 * MB) == 6144 * MB);
    // (nothing changes while the rest and the budget fit under the line, so the cache may still grow to its budget)
    CHECK(texture_budget(3213 * MB, 2810 * MB, 5000 * MB, 7000 * MB) == 3213 * MB);
    // (the rest alone over the line: what is left for the textures, and not below nothing)
    CHECK(texture_budget(3213 * MB, 1000 * MB, 6500 * MB, 7000 * MB) == 800 * MB);
    CHECK(texture_budget(3213 * MB, 500 * MB, 7000 * MB, 7000 * MB) == 0);
    // (no allowance known, or more textures counted than resident: unchanged)
    CHECK(texture_budget(3213 * MB, 2810 * MB, 5900 * MB, 0) == 3213 * MB);
    CHECK(texture_budget(3213 * MB, 2500 * MB, 2000 * MB, 7000 * MB) == 3213 * MB);
    std::printf("PASS: the shadows go by what is really allocated\n");
    return 0;
}
