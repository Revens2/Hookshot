/* Which snapshot holds a late page's pre-diff bytes, and which of a submission's parts filled it last (vk_memory.cpp's
   late copy merge). No Vulkan here: the pick is the real one the merge runs. tests/test_capture_late_merge.py */
#pragma once

#include <cstdint>

namespace mkwvk::late_snap {

/// A snapshot's range: the pool at lo, the fill's bytes [at, at + len) of it, copied at off in the entry
struct Range {
    uint64_t lo, at, len, off;
};

/// The chosen snapshot: the entry's fill order (0 = none) and where in that entry the page starts
struct Hit {
    uint64_t seq, off;
};

/// Whether the range holds the page [at, at + page) of the pool at lo
inline bool range_holds(const Range& r, uint64_t lo, uint64_t at, uint64_t page) {
    return r.lo == lo && at >= r.at && at + page <= r.at + r.len;
}

/// The newest of two snapshots of one page: a submission's parts share a value and keep an entry each, so a page filled
/// again in a later part resolves to that part's fill (the one before the last diff), not the first's
inline bool seq_newer(uint64_t seq, uint64_t best) { return seq > best; }

/// The index of the newest snapshot holding the page [at, at + page) of the pool at lo, among the entries keep accepts
/// (-1 when none), and its offset: a submission's parts keep an entry each, ordered by the fill they took
template <typename Entries, typename Keep>
inline int pick(const Entries& entries, Keep&& keep, uint64_t lo, uint64_t at, uint64_t page, Hit& hit) {
    int best = -1;
    hit = {0, 0};
    int i = 0;
    for (const auto& e : entries) {
        if (keep(e))
            for (const auto& r : e.ranges) {
                if (!range_holds(r, lo, at, page)) continue;
                if (best >= 0 && !seq_newer(e.seq, hit.seq)) continue;
                best = i;
                hit = {e.seq, r.off + (at - r.at)};
            }
        i++;
    }
    return best;
}

} // namespace mkwvk::late_snap
