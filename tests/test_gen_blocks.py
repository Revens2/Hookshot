"""The page write generations (gpu/vk/gen_blocks.h) answer exactly as a brute-force page table: random single pages,
ranges and whole 2 MB regions stamped at increasing generations, then written_since and written_runs asked at random
ranges and generations, with the 64-page block level on and off (same answers, fewer probes). The blocks are what skip a
dirty region's clean parts: a negative control forgets one block update and checks the skip then loses the written page.

The source checks keep the invariant that makes a skip safe: every stamp of a page raises its block's generation, a
whole 2 MB region goes through region_full (checked beside the block), memory_stamp uses the range stamp, and
mirror_sync skips by block only behind RX_VK_SYNC_BLOCKS, finds a capture's buffer once a generation, and reports the
CPU's writes to wb_learn once a dirty run instead of once a page.

python -I -B tests/test_gen_blocks.py [--cxx g++] [--source /path/to/source/tree]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def need(where, text, needles):
    for needle in needles:
        if needle not in text:
            raise SystemExit(where + " no longer has: " + needle)


def check_source(source):
    header = (source / "gpu/vk/gen_blocks.h").read_text(encoding="utf-8")
    mem = (source / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    # (the pages and their 64-page block are stamped together: a block no newer than a sync's generation holds no
    # newer page, so mirror_sync and memory_written_* may skip it)
    stamp = block(header, "void stamp_page(uint64_t p, uint32_t g) {")
    need("gen_blocks.h: stamp_page()", stamp, ["blocks_", "kBlockPages", "store_u32"])
    rng = block(header, "void stamp_range(uint64_t first_page, uint64_t last_page, uint32_t g) {")
    need("gen_blocks.h: stamp_range()", rng, ["full_[", "stamp_page("])
    since = block(header, "bool written_since(")
    need("gen_blocks.h: written_since()", since, ["region_full_gen(r)", "block_gen(p)", "page_gen(p)"])
    runs = block(header, "void written_runs(")
    need("gen_blocks.h: written_runs()", runs, ["region_full_gen(r)", "block_gen(p)", "page_gen(p)"])
    # (memory_stamp: a whole-region stamp goes through region_full, the rest a page at a time)
    st = block(mem, "void memory_stamp(uint64_t guest, uint64_t size) {")
    need("vk_memory.cpp: memory_stamp()", st, ["gen_table.stamp_range("])
    # (RX_VK_SYNC_BLOCKS=0: the dirty region's pages all examined, as before; on unless an off value)
    switch = block(mem, "bool sync_blocks_on() {")
    need("vk_memory.cpp: sync_blocks_on()", switch, ['getenv("RX_VK_SYNC_BLOCKS")', "env_value_off"])
    # (the sync: block skip behind the switch, one capture-buffer lookup a generation, one wb_learn note a dirty run)
    sync = block(mem, "void mirror_sync(Pool& p) {")
    need("vk_memory.cpp: mirror_sync()", sync,
         ["sync_blocks_on()", "gen_table.block_gen(page)", "gen_table.region_full_gen(region)",
          "memo_buf = cap_bufs.find(e.gen)", "memory_note_cpu_access(note_lo, note_hi - note_lo, true)"])
    if "memory_note_cpu_access(page << kPageShift, 1ull << kPageShift, true)" in sync:
        raise SystemExit("vk_memory.cpp: mirror_sync() reports the CPU's writes once a page again, not once a dirty run")
    # (the counter that decides the block level in play: examined pages against dirty regions, and the lookups)
    report = block(mem, "void memory_end_submission(VkCommandBuffer cmd, uint64_t part) {")
    need("vk_memory.cpp: memory_end_submission()", report,
         ["st_sync_examined", "st_sync_dirty_regions", "st_sync_blocks_skipped", "st_sync_cap_lookups"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--source", type=Path, default=ROOT, help="source tree whose generation tables are tested")
    a = p.parse_args()
    source = a.source.resolve()
    with tempfile.TemporaryDirectory(prefix="gen-blocks-") as tmp:
        exe = Path(tmp) / "gen_blocks"
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(source / "gpu/vk"),
                        str(source / "tests/test_gen_blocks.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    check_source(source)


if __name__ == "__main__":
    main()
