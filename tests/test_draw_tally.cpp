// draw_tally.h (gpu/vk): a recording thread's counts in its own lines, added to the totals by tally_fold, and
// RX_VK_SHARED_COUNTERS=1 adding straight to the totals (the behaviour before the tally). tests/test_draw_tally.py
#include "../gpu/vk/draw_tally.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace mkwvk;

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

int main() {
    const bool shared = shared_counters();
    tally_bump(&DrawTally::waits, st_waits, 5);
    expect(st_waits.load() == (shared ? 5 : 0), "this thread's count: in the total at once when shared, at its fold otherwise");
    tally_fold();
    expect(st_waits.load() == 5, "fold: the thread's count reaches the total");
    expect(tally.waits == 0, "fold: the thread's own line is spent");
    tally_fold();
    expect(st_waits.load() == 5, "fold: folding again adds nothing");
    const int threads = 4, per = 10000;
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; t++)
        workers.emplace_back([] {
            for (int i = 0; i < per / 2; i++) {
                tally_bump(&DrawTally::draws, st_draws);
                tally_bump(&DrawTally::barriers, st_barriers);
                tally_bump(&DrawTally::dummy_textures, st_dummy_textures, 0);   // (a zero count adds nothing)
            }
            tally_fold();   // (a segment's end)
            for (int i = 0; i < per / 2; i++) {
                tally_bump(&DrawTally::draws, st_draws);
                tally_bump(&DrawTally::barriers, st_barriers);
            }
            tally_fold();
        });
    for (auto& w : workers) w.join();
    expect(st_draws.load() == int64_t(threads) * per, "the totals hold every thread's counts, once");
    expect(st_barriers.load() == int64_t(threads) * per, "each counter on its own line is complete");
    expect(st_dummy_textures.load() == 0, "a zero count adds nothing");
    par_locks.fetch_add(1, std::memory_order_relaxed);
    expect(par_locks.load() == 1, "par locks stay shared (a thread outside a segment never folds)");
    if (failures) return EXIT_FAILURE;
    std::printf("draw tally: all checks passed (%s)\n", shared ? "shared" : "tallied");
    return EXIT_SUCCESS;
}
