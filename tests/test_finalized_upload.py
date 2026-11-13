"""Compile and run the rule for when an uploaded texture the game finalized is let go (gpu/vk/finalized_upload.h; no game/GPU)."""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    """The finalize stamps the uploads, the eviction pass reads the rule under the budget, and the switch is still read"""
    src = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
    for needle in ('#include "finalized_upload.h"',
                   'finalized_upload::presents_from_env(getenv("RX_VK_FINALIZED_UPLOADS"))',
                   "it->second->finalized_frame = frame_no;",
                   "finalized_upload::left_behind(",
                   "if (sweep && !overwritten && !behind) continue;",
                   "!upload_never(*tex);",
                   "if (behind) fin_bytes += tex->bytes + tex->pad",
                   "finalized_upload::fits_in_pass(fin_bytes, tex->bytes + tex->pad)"):
        if needle not in src:
            raise SystemExit("vk_texture.cpp no longer has: " + needle)
    # (a re-upload or a GPU write clears the stamp: two sites, next to released = false)
    if src.count("tex->finalized_frame = 0;") < 2:
        raise SystemExit("vk_texture.cpp: the upload and the GPU write no longer clear finalized_frame")
    # (only texels that are in guest memory count: set by an upload and a write-back, cleared where a texture is made over
    # storage the GPU rendered and not uploaded (7b774bae's terrain tiles), by upload_never and by a GPU write)
    if src.count("texels_in_memory = true;") < 2 or src.count("texels_in_memory = false;") < 3:
        raise SystemExit("vk_texture.cpp: texels_in_memory is no longer set at the upload/write-back/not-uploaded/GPU-write sites")
    for needle in ("if (!it->second->gpu_written && it->second->texels_in_memory) {",
                   "{tex->gpu_written, tex->texels_in_memory, tex->finalized_frame, tex->last_frame}"):
        if needle not in src:
            raise SystemExit("vk_texture.cpp no longer has: " + needle)
    hdr = (ROOT / "gpu/vk/vk_texture.h").read_text(encoding="utf-8")
    for needle in ("uint64_t finalized_frame = 0;", "bool texels_in_memory = false;"):
        if needle not in hdr:
            raise SystemExit("vk_texture.h: Texture has no " + needle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="finalized-upload-") as tmp:
        exe = Path(tmp) / "finalized-upload"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_finalized_upload.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
