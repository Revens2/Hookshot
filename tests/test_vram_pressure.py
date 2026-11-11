"""Compile and run the rules that judge VRAM near the driver's limit (gpu/vk/vram_pressure.h; no game/GPU required): the
write-back shadows' and the texture budget that leaves the rest of the process its room; and check that vk_memory.cpp still
judges the shadows by what is really allocated, with the switch back to the driver's count alone, and that vk_texture.cpp
takes its budget through the helper."""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    src = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    for needle in ('#include "vram_pressure.h"', 'getenv("RX_VK_SHADOW_RAW")',
                   "vram_pressure::shadow_fits(n, used, allowed, block_free)",
                   "vram_pressure::shadows_go(used, allowed, block_free)"):
        if needle not in src:
            raise SystemExit("vk_memory.cpp no longer has: " + needle)
    tex = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
    for needle in ('#include "vram_pressure.h"', "vram_pressure::texture_budget(budget, total_bytes, pressure, allowed)"):
        if needle not in tex:
            raise SystemExit("vk_texture.cpp no longer has: " + needle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="vram-pressure-") as tmp:
        exe = Path(tmp) / "vram-pressure"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_vram_pressure.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
