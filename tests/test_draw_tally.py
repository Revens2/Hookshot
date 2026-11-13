"""Compile and run draw_tally.h's per-thread counters (gpu/vk/draw_tally.h; no game/GPU): threads count in their own
lines and tally_fold adds them once, and RX_VK_SHARED_COUNTERS=1 adds straight to the totals as before the tally."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="draw-tally-") as tmp:
        exe = Path(tmp) / "draw-tally"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_draw_tally.cpp"), "-o", str(exe)], check=True)
        for shared in (False, True):
            env = os.environ.copy()
            env.pop("RX_VK_SHARED_COUNTERS", None)
            if shared:
                env["RX_VK_SHARED_COUNTERS"] = "1"
            out = subprocess.run([str(exe)], env=env, capture_output=True, text=True, check=True).stdout
            want = "shared" if shared else "tallied"
            if f"draw tally: all checks passed ({want})" not in out:
                raise SystemExit("draw tally, RX_VK_SHARED_COUNTERS=" + ("1" if shared else "unset") + ":\n" + out)
    print("pass: the tally is per thread and folds once a segment; RX_VK_SHARED_COUNTERS=1 adds straight to the totals")


if __name__ == "__main__":
    main()
