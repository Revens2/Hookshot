// How many segments a parallel replay cuts a command buffer into (vk_replay.cpp replay_handle). tests/test_par_split.py
#pragma once
#include <algorithm>
#include <cstdint>

namespace par_split {

// (hookshot) The rule was total / RX_VK_PARALLEL_SEG (160) whatever the threads: at Kakariko's graveyard (RTX 5070, i5-14600KF,
// 1095 draws and 100 dispatches a frame) a parallel command buffer of ~430 draws and dispatches made 2-3 segments of 160-319
// with RX_VK_PARALLEL at 4, 6 or 8 alike (2.6-2.7 segments, 0.97-1.01 ms a command buffer), so the threads past the third did
// nothing and the replay thread waited the whole time. Now: at most `per` a segment as before, but at least as many segments
// as there are recorders (the workers and the replay thread), while each keeps `min_seg` draws and dispatches: a segment's
// first draws pay a fresh recording's state (full descriptor push, dynamic state, vertex binds) and each segment one submit
constexpr int64_t kMinSeg = 96;

inline int count(int64_t total, int64_t per, int recorders, int64_t min_seg, int max_batch) {
    if (total <= 0 || max_batch < 1) return 1;
    int64_t n = total / std::max<int64_t>(per, 1);   // (the rule before)
    const int64_t spread = std::min<int64_t>(std::max(recorders, 1), total / std::max<int64_t>(min_seg, 1));
    n = std::max(n, spread);
    return (int)std::clamp<int64_t>(n, 1, max_batch);
}

} // namespace par_split
