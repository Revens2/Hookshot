// finalized_upload (gpu/vk/finalized_upload.h): when an uploaded texture the game finalized is an area left behind.
// tests/test_finalized_upload.py
#include "../gpu/vk/finalized_upload.h"

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
    using finalized_upload::fits_in_pass;
    using finalized_upload::kPassBytes;
    using finalized_upload::left_behind;
    using finalized_upload::presents_from_env;
    using finalized_upload::State;
    const uint64_t P = 1800;
    const uint64_t MB = 1ull << 20;

    // (finalized at 1000, last used at 1002: the frames in flight then)
    const State done{false, true, 1000, 1002};
    expect(!left_behind(done, 1002 + P - 1, P), "unused one present short of the time: kept");
    expect(left_behind(done, 1002 + P, P), "unused for the time: left behind");
    expect(left_behind(done, 1002 + 10 * P, P), "long after: left behind");

    // (sampled again after the finalize: made again over the same storage, in use)
    expect(!left_behind({false, true, 1000, 1009}, 1009 + 10 * P, P), "sampled 9 presents after its finalize: kept");
    expect(left_behind({false, true, 1000, 1008}, 1008 + P, P), "within the 8 presents in flight: still left behind");

    expect(!left_behind({true, true, 1000, 1000}, 1000 + 10 * P, P), "one the GPU drew: never (its image can be the only copy)");
    expect(!left_behind({false, false, 1000, 1000}, 1000 + 10 * P, P), "not uploaded from memory: never");
    expect(!left_behind({false, true, 0, 1000}, 1000 + 10 * P, P), "not finalized: never (the budget's rule)");
    expect(!left_behind(done, 1002 + 10 * P, 0), "switched off: never");
    expect(!left_behind({false, true, 4995, 5000}, 4000, P), "a last use after now (no wrapped difference): kept");

    // (the switch: hookshot_env.h's off values, a number of presents, else the default)
    expect(presents_from_env(nullptr) == finalized_upload::kDefaultPresents, "unset: the default");
    expect(presents_from_env("0") == 0, "0: off");
    expect(presents_from_env("off") == 0, "off: off");
    expect(presents_from_env("No") == 0, "No: off");
    expect(presents_from_env("") == 0, "empty: off (hookshot_env.h)");
    expect(presents_from_env("3600") == 3600, "a number of presents");
    expect(presents_from_env("1") == finalized_upload::kDefaultPresents, "1 (on): the default");
    expect(presents_from_env("on") == finalized_upload::kDefaultPresents, "on: the default");

    // (the pass's cap: hard, a texture that would push it over is not taken, even alone)
    expect(!fits_in_pass(0, 349 * MB), "a 349 MB course map over the cap alone: never in one pass");
    expect(fits_in_pass(0, kPassBytes), "a texture of exactly the cap, alone: taken");
    expect(fits_in_pass(100 * MB, 156 * MB), "filling the pass to exactly the cap: taken");
    expect(!fits_in_pass(100 * MB, 156 * MB + 1), "one byte over the cap: not taken");
    expect(!fits_in_pass(kPassBytes, 1), "the pass already full: nothing more");
    expect(!fits_in_pass(1, kPassBytes), "a full-cap texture with the pass not empty: not taken");

    if (failures) return EXIT_FAILURE;
    std::printf("finalized upload: all checks passed\n");
    return EXIT_SUCCESS;
}
