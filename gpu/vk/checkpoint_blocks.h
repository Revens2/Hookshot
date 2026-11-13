// (hookshot) The checkpoints' marker numbers, reserved in blocks by each recording thread. ++checkpoint_n was one locked
// read-modify-write per draw and dispatch from five threads on one line, and the Checkpoint entries the threads wrote
// next to each other shared their lines too. A block of 64 numbers is the reserving thread's own: 64 * sizeof(Checkpoint)
// (24) = 1536 bytes, a multiple of a cache line, so the block is exactly 24 lines shared with no other thread.
// A block left over after its thread went quiet is given up once what is left of it lies a whole ring behind the
// reservations: those numbers would name slots newer markers use, and writing them would take those entries away. The
// reports also check an entry's sequence number against the marker they print it for.
// tests/test_checkpoint_blocks.py.
#pragma once
#include <atomic>
#include <cstdint>

namespace checkpoint_blocks {

constexpr uint32_t kBlock = 64;

/// This thread's next marker number: a block reserved from n when the previous one is used up, or given up when what is
/// left of it lies a whole ring or more behind the reservations (its numbers would then share their slots with newer
/// markers). Checked every call: recorders keep using a block across a long quiet stretch otherwise. ring: the numbers
/// the entries and marker words hold.
inline uint32_t next(uint32_t& cp_next, uint32_t& cp_end, std::atomic<uint32_t>& n, uint32_t ring = 1u << 16) {
    // (a number x shares its slot with a marker already reserved once x + ring <= n; the next number is cp_next + 1, so
    // the block is given up when n - cp_next > ring. An n reset to 0 lands here too: the subtraction wraps to no block)
    if (cp_end != 0 && uint32_t(n.load(std::memory_order_relaxed) - cp_next) > ring) cp_next = cp_end;
    if (cp_next == cp_end) {
        cp_next = n.fetch_add(kBlock, std::memory_order_relaxed);
        cp_end = cp_next + kBlock;
    }
    return ++cp_next;
}

}  // namespace checkpoint_blocks
