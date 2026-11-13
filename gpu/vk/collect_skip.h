/* (hookshot) Ranges set arithmetic for the replay's collection (gpu/vk/vk_memory.cpp, host/rx_backend.c): a's ranges
   sorted and disjoint, minus b's (sorted, disjoint; b's may reach past a's), into out. out gets at most na + 2 * nb
   ranges, sorted and disjoint: -1 when max cannot hold them. (tests/test_collect_skip_captured.py: against a page set,
   and the collection's old and new walks side by side) */
#pragma once
#include <stdint.h>

static inline int64_t ranges_minus(const uint64_t (*a)[2], int na, const uint64_t (*b)[2], int nb,
                                   uint64_t (*out)[2], int max) {
    int n = 0, j = 0;
    for (int i = 0; i < na; i++) {
        uint64_t lo = a[i][0], hi = a[i][1];
        while (j < nb && b[j][1] <= lo) j++;
        for (int k = j; k < nb && b[k][0] < hi; k++) {
            if (b[k][0] > lo) {
                if (n == max) return -1;
                out[n][0] = lo, out[n][1] = b[k][0], n++;
            }
            if (b[k][1] > lo) lo = b[k][1];
            if (lo >= hi) break;
        }
        if (lo < hi) {
            if (n == max) return -1;
            out[n][0] = lo, out[n][1] = hi, n++;
        }
    }
    return n;
}