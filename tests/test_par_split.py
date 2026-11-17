"""Compile and run the parallel replay's segment count (gpu/vk/par_split.h; no game/GPU), and check replay_handle uses it."""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    """replay_handle counts its segments with the rule, reads the switch, and records segment 0 on the replay thread"""
    src = (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")
    for needle in ('#include "par_split.h"', 'getenv("RX_VK_PARALLEL_BALANCE")',
                   "par_split::count(total, per, threads + 1, par_split::kMinSeg, kMaxBatch)",
                   "if (balance && j->index == 0) continue;", "run_job(jobs[0].get());", "ub_gen++;",
                   "longest segment %.2f ms"):
        if needle not in src:
            raise SystemExit("vk_replay.cpp no longer has: " + needle)
    # (segment 0 is recorded here only after the workers were woken: it must not delay their start)
    if src.index("par_pool.cv.notify_all();") > src.index("run_job(jobs[0].get());"):
        raise SystemExit("vk_replay.cpp: segment 0 is recorded before the workers are woken")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="par-split-") as tmp:
        exe = Path(tmp) / "par-split"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_par_split.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
