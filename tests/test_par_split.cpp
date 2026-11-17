// par_split::count (gpu/vk/par_split.h): how many segments a parallel replay cuts a command buffer into. tests/test_par_split.py
#include "../gpu/vk/par_split.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

int main() {
    using par_split::count;
    const int64_t per = 160, min_seg = par_split::kMinSeg;   // (RX_VK_PARALLEL_SEG's default; 96)
    // (RX_VK_PARALLEL=4: four workers and the replay thread)
    expect(count(430, per, 5, min_seg, 6) == 4, "430 draws, 5 recorders: 4 segments of ~107 (was 2 of 215)");
    expect(count(300, per, 5, min_seg, 6) == 3, "300: 3 segments of 100 (each keeps 96)");
    expect(count(320, per, 5, min_seg, 6) == 3, "320: 3, not 2 as before");
    expect(count(100, per, 5, min_seg, 6) == 1, "100: one (too little for two of 96)");
    expect(count(2000, per, 5, min_seg, 6) == 6, "2000: the batch cap");
    expect(count(900, per, 9, min_seg, 6) == 6, "RX_VK_PARALLEL=8: still the cap of 6");
    // (one recorder: the rule before, total / per)
    for (int64_t t = 0; t <= 3000; t += 7) {
        const int64_t before = t / per < 1 ? 1 : t / per > 6 ? 6 : t / per;
        if (count(t, per, 1, min_seg, 6) != before) {
            std::printf("FAIL: one recorder, %lld draws: %d, the rule before gives %lld\n", (long long)t,
                        count(t, per, 1, min_seg, 6), (long long)before);
            failures++;
            break;
        }
    }
    // (never fewer segments than before, and never more than the batch cap; monotone in the work)
    int prev = 0;
    for (int64_t t = 0; t <= 3000; t++) {
        const int n = count(t, per, 5, min_seg, 6);
        const int64_t before = t / per < 1 ? 1 : t / per > 6 ? 6 : t / per;
        if (n < before || n > 6 || n < prev) {
            std::printf("FAIL: %lld draws: %d segments (before %lld, previous %d)\n", (long long)t, n, (long long)before, prev);
            failures++;
            break;
        }
        prev = n;
    }
    // (a segment keeps min_seg draws whenever there is more than one)
    for (int64_t t = 0; t <= 3000; t++) {
        const int n = count(t, per, 5, min_seg, 6);
        if (n > 1 && t / n < min_seg && n > t / per) {
            std::printf("FAIL: %lld draws in %d segments: under %lld each\n", (long long)t, n, (long long)min_seg);
            failures++;
            break;
        }
    }
    expect(count(-5, per, 5, min_seg, 6) == 1 && count(500, 0, 5, min_seg, 6) >= 1 && count(500, per, 0, min_seg, 6) >= 1,
           "degenerate inputs: at least one segment");
    if (failures) return EXIT_FAILURE;
    std::printf("par split: all checks passed\n");
    return EXIT_SUCCESS;
}
