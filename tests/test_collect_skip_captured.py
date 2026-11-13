"""The replay's collection leaves the ranges the captures own to the captures (P1, RX_VK_COLLECT_SKIP_CAPTURED).

memory_collect_written walked every GPU-visible range, took each marked page, and put the captured pages back
(mark_written) for the next capture: the find was the capture's own walk done a second time, and the report only fed
the write-back diff's shadows. Now the host walks the GPU ranges less the captured ranges less the shadowed pools' (the
shadowed pages still need the report), and the captured pages' marks stay set for their next capture.

1. gpu/vk/collect_skip.h's ranges_minus: against a page set (random lists, the gaps and the touching ranges included).
2. host/rx_backend.c's collect_in, old flow against new: the same dirty marks and the same pages outside the captures,
   and the shadowed captured pages found still.
3. The source: the switch read through env_value_off, the exception call only under it, the skip list keyed on both
   versions and published where a shadow is made, given back or reaped (tests/test_wb_shadow_remake.py cuts those
   functions; its fixture stubs the publish).
python -I -B tests/test_collect_skip_captured.py [--cxx g++]
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def check_source(memory, api, backend):
    switch = block(memory, "bool collect_skip_captured() {")
    if "RX_VK_COLLECT_SKIP_CAPTURED" not in switch or "env_value_off(" not in switch:
        raise SystemExit("vk_memory.cpp: the skip switch is not read through env_value_off (RX_VK_COLLECT_SKIP_CAPTURED)")
    collect = block(memory, "void memory_collect_written() {")
    try:
        at_switch = collect.index("collect_skip_captured()")
        at_except = collect.index("host->collect_written_except")
    except ValueError:
        raise SystemExit("vk_memory.cpp: memory_collect_written no longer takes the skip switch and the exception call")
    if at_switch > at_except:
        raise SystemExit("vk_memory.cpp: memory_collect_written uses collect_written_except before the switch")
    on_arm = collect[collect.index("if (skip_on) {"):collect.index("} else {")]
    if "host->collect_written_except" not in on_arm:
        raise SystemExit("vk_memory.cpp: collect_written_except is not under the skip switch")
    for want in ("collect_skip_ranges(owned)", "host->collect_written(&pages)"):
        if want not in collect:
            raise SystemExit(f"vk_memory.cpp: memory_collect_written lost {want}")
    if not collect.count("collect_written_except") or memory.count("collect_written_except") != collect.count("collect_written_except"):
        raise SystemExit("vk_memory.cpp: collect_written_except is used outside memory_collect_written, or not at all")
    skip = block(memory, "const std::vector<std::pair<uint64_t, uint64_t>>& collect_skip_ranges("
                         "const std::vector<std::pair<uint64_t, uint64_t>>& owned) {")
    for want in ("ranges_minus(", "shadow_ranges", "wb_diff_on()", "capture_ranges_seen", "shadow_ranges_ver"):
        if want not in skip:
            raise SystemExit(f"vk_memory.cpp: collect_skip_ranges is not keyed on {want} (or does not cut the shadows out)")
    if "shadow_ranges_ver++" not in block(memory, "void shadow_ranges_publish() {"):
        raise SystemExit("vk_memory.cpp: shadow_ranges_publish does not bump the skip list's version")
    for name in ("void shadow_free(Pool& p) {", "bool shadow_make(Pool& p) {", "void shadows_reap() {"):
        if "shadow_ranges_publish()" not in block(memory, name):
            raise SystemExit(f"vk_memory.cpp: {name} does not publish the skip list")
    version = re.search(r"#define MKWVK_API_VERSION\s+(\d+)", api)
    if not version or int(version.group(1)) < 23:
        raise SystemExit("vk_api.h: MkwVkHost gained a function, MKWVK_API_VERSION must go to 23")
    if "collect_written_except" not in api:
        raise SystemExit("vk_api.h: MkwVkHost.collect_written_except is missing")
    if backend.count("static int64_t host_collect_except(") != 2:
        raise SystemExit("rx_backend.c: host_collect_except is not defined for both the write-track and the other builds")
    if "#include \"../gpu/vk/collect_skip.h\"" not in backend or "ranges_minus(" not in backend:
        raise SystemExit("rx_backend.c: the walk does not cut the skip ranges with gpu/vk/collect_skip.h")
    table = backend[backend.index("static const MkwVkHost vk_host = {"):]
    if "host_collect_except" not in table[:table.index("};")]:
        raise SystemExit("rx_backend.c: the host table does not hand over collect_written_except")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    memory = (ROOT / "gpu" / "vk" / "vk_memory.cpp").read_text(encoding="utf-8")
    api = (ROOT / "gpu" / "vk" / "vk_api.h").read_text(encoding="utf-8")
    backend = (ROOT / "host" / "rx_backend.c").read_text(encoding="utf-8")
    check_source(memory, api, backend)
    cut = block(backend, "static int cmp_range_lo(") + "\n" + block(backend, "static int64_t collect_in(")
    # (C's void* assignments are C++ errors: the fixture casts the two output allocations and changes nothing else)
    cut = cut.replace("uint32_t *out = malloc(", "uint32_t *out = (uint32_t *)malloc(")
    cut = cut.replace("out = realloc(out,", "out = (uint32_t *)realloc(out,")
    if "uint32_t *)malloc(" not in cut or "uint32_t *)realloc(" not in cut:
        raise SystemExit("rx_backend.c: collect_in's allocations moved (the fixture's C++ casts)")
    fixture = r'''
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include "gpu/vk/collect_skip.h"

#define CHECK(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); std::exit(1); } } while (0)

#define A64_DIRTY_PAGES (1u << 20)
static uint8_t a64_dirty[A64_DIRTY_PAGES];
static uint64_t guest_pages = A64_DIRTY_PAGES;
static uint64_t hos_guest_size(void) { return guest_pages << 12; }

''' + "CUT_HERE" + r'''

using Range = std::pair<uint64_t, uint64_t>;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s ^= s << 13, s ^= s >> 7, s ^= s << 17; return s; }
    uint64_t below(uint64_t n) { return next() % n; }
};

static bool page_in(const std::vector<Range>& rs, uint64_t page) {
    for (const Range& r : rs)
        if (page >= (r.first >> 12) && page < (r.second >> 12)) return true;
    return false;
}

static std::vector<Range> random_ranges(Rng& rng, uint64_t pages, int count) {
    std::vector<Range> out;
    uint64_t at = rng.below(4);
    for (int i = 0; i < count && at < pages; i++) {
        at += rng.below(8);
        if (at >= pages) break;
        const uint64_t n = 1 + rng.below(24);
        out.push_back({at << 12, std::min(at + n, pages) << 12});
        at += n;
    }
    return out;
}

static std::vector<Range> clip_to(const std::vector<Range>& rs, const std::vector<Range>& within) {
    std::vector<Range> out;
    for (const Range& r : rs)
        for (const Range& w : within) {
            const uint64_t lo = std::max(r.first, w.first), hi = std::min(r.second, w.second);
            if (lo < hi) out.push_back({lo, hi});
        }
    std::sort(out.begin(), out.end());
    return out;
}

static bool disjoint_sorted(const std::vector<Range>& rs) {
    for (size_t i = 0; i < rs.size(); i++) {
        if (rs[i].first >= rs[i].second) return false;
        if (i && rs[i].first < rs[i - 1].second) return false;
    }
    return true;
}

static std::vector<Range> coalesce(const std::vector<Range>& rs) {
    std::vector<Range> out;
    for (const Range& r : rs) {
        if (!out.empty() && r.first <= out.back().second) out.back().second = std::max(out.back().second, r.second);
        else out.push_back(r);
    }
    return out;
}

static std::vector<Range> minus(const std::vector<Range>& a, const std::vector<Range>& b, int64_t* got = nullptr) {
    std::vector<std::array<uint64_t, 2>> buf(a.size() + 2 * b.size() + 8);
    const int64_t n = ranges_minus((const uint64_t(*)[2])a.data(), (int)a.size(), (const uint64_t(*)[2])b.data(), (int)b.size(),
                                   (uint64_t(*)[2])buf.data(), (int)buf.size());
    CHECK(n >= 0);
    std::vector<Range> out;
    for (int64_t i = 0; i < n; i++) out.push_back({buf[(size_t)i][0], buf[(size_t)i][1]});
    if (got) *got = n;
    return out;
}

static void ranges_minus_against_pages() {
    Rng rng(20261113);
    for (int round = 0; round < 400; round++) {
        const uint64_t pages = 1 + rng.below(300);
        guest_pages = pages;
        const std::vector<Range> a = random_ranges(rng, pages, 1 + (int)rng.below(5));
        const std::vector<Range> b = random_ranges(rng, pages, (int)rng.below(5));
        CHECK(disjoint_sorted(a) && disjoint_sorted(b));
        int64_t n = -1;
        const std::vector<Range> got = minus(a, b, &n);
        CHECK(disjoint_sorted(got));
        std::vector<Range> want;
        for (uint64_t p = 0; p < pages; p++)
            if (page_in(a, p) && !page_in(b, p)) {
                if (!want.empty() && want.back().second == (p << 12)) want.back().second = (p + 1) << 12;
                else want.push_back({p << 12, (p + 1) << 12});
            }
        CHECK(coalesce(got) == want);   // (adjacent ranges may stay apart, the page set is what matters)
        CHECK(n <= (int64_t)(a.size() + 2 * b.size()));
        // (a's ranges with none of b's before or in them come out unchanged; touching ranges do; empty lists stay empty)
        CHECK(minus(a, {}) == a);
        CHECK(minus({}, b).empty());
        uint64_t out[2];
        CHECK(ranges_minus((const uint64_t(*)[2])a.data(), (int)a.size(), (const uint64_t(*)[2])b.data(), (int)b.size(),
                           (uint64_t(*)[2])out, 1) == (got.size() > 1 ? -1 : (int64_t)got.size()));
    }
    // (a range b reaches past, and b spanning a whole range and the gap after it: [0,100) less [90,200) and [200,300))
    const std::vector<Range> a = {{0, 100 << 12}, {150 << 12, 300 << 12}};
    const std::vector<Range> b = {{90 << 12, 200 << 12}};
    CHECK(minus(a, b) == std::vector<Range>({{0, 90 << 12}, {200 << 12, 300 << 12}}));
    std::printf("PASS: ranges_minus matches a page set over 400 random lists\n");
}

static uint64_t initial_dirty[A64_DIRTY_PAGES];

static std::vector<uint32_t> collect(const std::vector<Range>& ranges) {
    uint32_t* pages = nullptr;
    const int64_t n = collect_in((const uint64_t(*)[2])ranges.data(), (int)ranges.size(), &pages);
    std::vector<uint32_t> out(pages, pages + (n > 0 ? (size_t)n : 0));
    free(pages);
    return out;
}

static void mark_put_back(const std::vector<uint32_t>& pages, const std::vector<Range>& owned) {
    for (uint32_t page : pages)
        if (page_in(owned, page)) a64_dirty[page] = 1;
}

static std::vector<uint32_t> without_pages(const std::vector<uint32_t>& pages, const std::vector<Range>& owned) {
    std::vector<uint32_t> out;
    for (uint32_t page : pages)
        if (!page_in(owned, page)) out.push_back(page);
    return out;
}

static std::vector<uint32_t> within_pages(const std::vector<uint32_t>& pages, const std::vector<Range>& owned) {
    std::vector<uint32_t> out;
    for (uint32_t page : pages)
        if (page_in(owned, page)) out.push_back(page);
    return out;
}

struct FlowResult {
    size_t left, walked;   // (a captured page left to its capture; a shadowed captured page still walked)
};

static FlowResult flow_case(const std::vector<Range>& gpu, const std::vector<Range>& owned,
                            const std::vector<Range>& shadowed, const std::vector<uint32_t>& dirty) {
    const std::vector<Range> skip = minus(owned, shadowed);   // (the collection's skip list)
    const std::vector<Range> keep = minus(gpu, skip);
    CHECK(disjoint_sorted(gpu) && disjoint_sorted(owned) && disjoint_sorted(skip) && disjoint_sorted(keep));
    std::memset(a64_dirty, 0, sizeof a64_dirty);
    for (uint32_t page : dirty) a64_dirty[page] = 1;
    std::memcpy(initial_dirty, a64_dirty, sizeof a64_dirty);
    // (the old flow: every GPU range walked, every captured page put back)
    const std::vector<uint32_t> old_pages = collect(gpu);
    mark_put_back(old_pages, owned);
    const std::vector<uint8_t> old_dirty(a64_dirty, a64_dirty + guest_pages);
    // (the new flow, from the same marks: the GPU ranges less the skip list, the shadowed captured pages put back)
    std::memcpy(a64_dirty, initial_dirty, sizeof a64_dirty);
    const std::vector<uint32_t> new_pages = collect(keep);
    mark_put_back(new_pages, owned);
    const std::vector<uint32_t> old_captured = within_pages(old_pages, owned);
    const std::vector<uint32_t> new_captured = within_pages(new_pages, owned);
    CHECK(std::vector<uint8_t>(a64_dirty, a64_dirty + guest_pages) == old_dirty);
    CHECK(without_pages(old_pages, owned) == without_pages(new_pages, owned));
    CHECK(new_captured == within_pages(old_pages, shadowed));   // (the pages left are the ones only a shadow reads)
    return {old_captured.size() - within_pages(old_pages, shadowed).size(), new_captured.size()};
}

static void old_flow_against_new() {
    // (one case with each kind of page: outside the GPU ranges, in one, in a captured range the skip list leaves, and in a
    // shadowed captured range: page 10 is a captured page only the old flow finds, 20 is the one both find)
    guest_pages = 128;
    const FlowResult one = flow_case({{0, 64 << 12}}, {{8 << 12, 40 << 12}}, {{16 << 12, 24 << 12}}, {4, 10, 20, 50, 100});
    CHECK(one.left == 1 && one.walked == 1);   // (page 10 left, page 20 walked: a flow that walks all, or none, fails above)
    // (no shadow over a captured range: the whole captured range is left, nothing put back)
    const FlowResult bare = flow_case({{0, 64 << 12}}, {{8 << 12, 40 << 12}}, {}, {4, 10, 20, 50, 100});
    CHECK(bare.left == 2 && bare.walked == 0);
    // (nothing captured yet: the whole walk, the marks included)
    const FlowResult none = flow_case({{0, 64 << 12}}, {}, {}, {4, 10, 20, 50, 100});
    CHECK(none.left == 0 && none.walked == 0);
    // (and whatever the lists, 300 random rounds: the same marks, and the same pages outside the captures)
    Rng rng(424242);
    std::vector<uint32_t> dirty;
    size_t left = 0, walked = 0;
    for (int round = 0; round < 300; round++) {
        const uint64_t pages = 64 + rng.below(600);
        guest_pages = pages;
        const std::vector<Range> gpu = random_ranges(rng, pages, 2 + (int)rng.below(5));
        const std::vector<Range> owned = clip_to(random_ranges(rng, pages, 1 + (int)rng.below(4)), gpu);
        std::vector<Range> shadowed;
        if (!owned.empty()) shadowed = clip_to(random_ranges(rng, pages, (int)rng.below(3)), owned);
        dirty.clear();
        for (uint64_t p = 0; p < pages; p++)
            if (rng.below(3) == 0) dirty.push_back((uint32_t)p);
        const FlowResult r = flow_case(gpu, owned, shadowed, dirty);
        left += r.left ? 1 : 0, walked += r.walked ? 1 : 0;
    }
    std::printf("PASS: the old and new collection leave the same dirty marks and the same pages outside the captures (300 "
                "rounds, %zu with a captured page left, %zu with a shadowed one walked)\n", left, walked);
}

int main() {
    ranges_minus_against_pages();
    old_flow_against_new();
}
'''
    cpp_text = fixture.replace("CUT_HERE", cut)
    with tempfile.TemporaryDirectory(prefix="collect-skip-") as tmp:
        cpp = Path(tmp) / "fixture.cpp"
        exe = Path(tmp) / "fixture"
        cpp.write_text(cpp_text, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-D_CRT_SECURE_NO_WARNINGS",
                        "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()