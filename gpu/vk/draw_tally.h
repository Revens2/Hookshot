// (hookshot) The replay's counters as a thread's own. Under the parallel replay five threads record segments at once,
// and a counter in one shared line was written by every draw and every bind: the threads took that line in turn, ~60-120
// ns a write on the owner's CPU, and lost increments. Each recording thread keeps its tally in lines of its own and
// adds it to the totals once a segment (tally_fold), so a draw touches only its thread's lines; the dry pass and the
// serial replay fold the same way. The totals are atomics that the frame log and the statistics' readers read; a serial
// replay's counts reach them at its next fold (the frame log folds first), so a reader on another thread may be a frame
// behind, which the logs do not mind. par_locks and par_lock_wait_ns stay shared: ParLock is taken outside the replay
// threads too (texture registration, uploads), where no segment ends to fold a tally.
// RX_VK_SHARED_COUNTERS=1: every thread adds to the totals itself, as before the tally (for an A/B of this change).
#pragma once
#include <atomic>
#include <cstdint>

#include "compat/hookshot_env.h"

namespace mkwvk {

/// One recording thread's share of the counters: lines of its own, written by that thread only (aligned so none is shared)
struct alignas(64) DrawTally {
    int64_t draws = 0, skipped = 0, dispatches = 0, passes = 0, mesh = 0, pending_skips = 0, waits = 0;
    int64_t desc_reused = 0, dummy_textures = 0;
    int64_t bg_prog = 0, bg_bufs = 0, bg_ubo = 0, bg_ssbo = 0, bg_tex = 0, bg_samp = 0, bg_img = 0;
    int64_t barriers = 0, ub_updates = 0, ub_overlays = 0;
};
inline thread_local DrawTally tally;

// (the totals every thread's tally is added to: what the frame log and draw_stats read)
inline std::atomic<int64_t> st_draws{0}, st_skipped{0}, st_dispatches{0}, st_passes{0}, st_mesh{0}, st_pending_skips{0}, st_waits{0};
inline std::atomic<int64_t> st_desc_reused{0}, st_dummy_textures{0};
inline std::atomic<int64_t> st_bg_prog{0}, st_bg_bufs{0}, st_bg_ubo{0}, st_bg_ssbo{0}, st_bg_tex{0}, st_bg_samp{0}, st_bg_img{0};
inline std::atomic<int64_t> st_barriers{0}, st_ub_updates{0}, st_ub_overlays{0};
inline std::atomic<int64_t> par_locks{0}, par_lock_wait_ns{0};

/// RX_VK_SHARED_COUNTERS=1: a counted total written by whichever thread counts it, as before the tally
inline bool shared_counters() {
    static const bool on = env_on("RX_VK_SHARED_COUNTERS");
    return on;
}

/// One count: into the thread's own lines, or into the shared total with RX_VK_SHARED_COUNTERS
inline void tally_bump(int64_t DrawTally::*field, std::atomic<int64_t>& total, int64_t n = 1) {
    if (shared_counters()) total.fetch_add(n, std::memory_order_relaxed);
    else tally.*field += n;
}

// (one field of this thread's tally into its total, then zero: the count is not lost when the next one starts)
inline void tally_add(int64_t& local, std::atomic<int64_t>& total) {
    const int64_t v = local;
    if (v) total.fetch_add(v, std::memory_order_relaxed), local = 0;
}

/// This thread's tally into the totals: once a segment's end, and before a log reads them on the replay thread
inline void tally_fold() {
    tally_add(tally.draws, st_draws);
    tally_add(tally.skipped, st_skipped);
    tally_add(tally.dispatches, st_dispatches);
    tally_add(tally.passes, st_passes);
    tally_add(tally.mesh, st_mesh);
    tally_add(tally.pending_skips, st_pending_skips);
    tally_add(tally.waits, st_waits);
    tally_add(tally.desc_reused, st_desc_reused);
    tally_add(tally.dummy_textures, st_dummy_textures);
    tally_add(tally.bg_prog, st_bg_prog);
    tally_add(tally.bg_bufs, st_bg_bufs);
    tally_add(tally.bg_ubo, st_bg_ubo);
    tally_add(tally.bg_ssbo, st_bg_ssbo);
    tally_add(tally.bg_tex, st_bg_tex);
    tally_add(tally.bg_samp, st_bg_samp);
    tally_add(tally.bg_img, st_bg_img);
    tally_add(tally.barriers, st_barriers);
    tally_add(tally.ub_updates, st_ub_updates);
    tally_add(tally.ub_overlays, st_ub_overlays);
}

}  // namespace mkwvk
