"""The 97% driver's-limit squeeze waits while VMA's blocks hold room, but not past the allowance.

At the driver's limit, texture_frame set the texture budget below what the textures held every 10 frames. Evicting frees
space inside VMA's blocks, which the driver still counts, so the count stayed at the limit and the budget shrank again: a
ratchet. On a 4 GB RX 5500M a paused game (its scene draws nothing) lost 1700 MB of textures in seconds, made them again on
unpausing (a long wait), and the HUD's GPU-drawn hearts came back as black glyphs until an area change. The rule now waits
while the blocks hold an eighth of the allowance free; RX_VK_SQUEEZE_ALWAYS=1 squeezes as before.

But only while the count is within the allowance: past it the driver pages VRAM out to system memory whatever room the blocks
hold, and the squeeze must run (an RTX 5050 8 GB at 7602 MB used of 6982 with 1649 MB free in the blocks paged 4.5 GB and
dropped to 14-25 fps, #2156).

python -I -B tests/test_texture_squeeze.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"


def main():
    tex = (VK / "vk_texture.cpp").read_text(encoding="utf-8")
    if "const bool squeeze_held" not in tex:
        raise SystemExit("the 97% squeeze runs whatever room VMA's blocks hold (a ratchet that drains a paused game's textures)")
    squeeze = tex[tex.index("const bool squeeze_held"):]
    squeeze = squeeze[:squeeze.index("budget = std::min")]
    if "vram_pressure::squeeze_waits(used, allowed, block_free, squeeze_always)" not in squeeze:
        raise SystemExit("the squeeze does not judge the blocks' room and the allowance together")
    if "&& !squeeze_held)" not in squeeze:
        raise SystemExit("the 97% squeeze does not wait for squeeze_waits")
    if "squeeze_held ?" not in tex:
        raise SystemExit("the VRAM log line does not say when the squeeze waited")
    header = (VK / "vram_pressure.h").read_text(encoding="utf-8")
    if "used <= allowed && block_free >= allowed / 8" not in header:
        raise SystemExit("squeeze_waits does not keep the squeeze past the allowance (the driver pages VRAM out there)")
    if 'getenv("RX_VK_SQUEEZE_ALWAYS")' not in tex:
        raise SystemExit("RX_VK_SQUEEZE_ALWAYS=1 does not bring the old squeeze back")
    print("the driver's-limit squeeze waits while VMA's blocks hold room, within the allowance: ok")


if __name__ == "__main__":
    main()
