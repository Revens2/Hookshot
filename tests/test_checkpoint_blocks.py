"""Compile and run the checkpoints' marker numbers reserved in blocks by each recording thread
(gpu/vk/checkpoint_blocks.h; no game/GPU)."""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    """checkpoint() reserves blocks instead of one locked RMW a draw, and the counters go to a thread's own line"""
    src = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    for needle in ('#include "checkpoint_blocks.h"', "checkpoint_blocks::next(cp_next, cp_end, checkpoint_n)"):
        if needle not in src:
            raise SystemExit("vk_draw.cpp no longer has: " + needle)
    if "++checkpoint_n" in src:
        raise SystemExit("vk_draw.cpp still takes the locked RMW per checkpoint: ++checkpoint_n")
    tally = (ROOT / "gpu/vk/draw_tally.h").read_text(encoding="utf-8")
    for needle in ('env_on("RX_VK_SHARED_COUNTERS")', "tally_bump(", "tally_fold()", "thread_local DrawTally tally"):
        if needle not in tally:
            raise SystemExit("draw_tally.h no longer has: " + needle)
    replay = (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")
    for needle in ("tally_fold();", "tally_bump(&DrawTally::barriers, st_barriers)",
                   "tally_bump(&DrawTally::bg_prog, st_bg_prog)"):
        if needle not in replay:
            raise SystemExit("vk_replay.cpp no longer has: " + needle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="checkpoint-blocks-") as tmp:
        exe = Path(tmp) / "checkpoint-blocks"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_checkpoint_blocks.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
