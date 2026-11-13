"""Compile and run the per-thread pipeline memo's compare and slot rule (gpu/vk/pipe_memo.h; no game/GPU)."""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    """graphics_pipeline takes its memo from the header, and the switch back is still read"""
    src = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    for needle in ('#include "pipe_memo.h"', "pipe_memo::slot((uintptr_t)&p, p.variant, topo_c, at.n)",
                   "pipe_memo::hit(from, slot->from)", "pipe_memo::hit(from, f.from)",
                   'getenv("RX_VK_PIPE_MEMO")', "env_value_off(e)"):
        if needle not in src:
            raise SystemExit("vk_draw.cpp no longer has: " + needle)
    if "bool pipe_same(" in src:
        raise SystemExit("vk_draw.cpp still defines pipe_same: the compare moved to gpu/vk/pipe_memo.h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="pipe-memo-") as tmp:
        exe = Path(tmp) / "pipe-memo"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_pipe_memo.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
