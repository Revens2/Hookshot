"""The draw hot path reads what it already resolved: colour output types with the targets, and the draw-stats switch
once a run. Source checks, no game: each value must still come from the same function it came from before
(color_types_of, targets_persistent, stats_dir), so a draw does the same thing it did.

- Targets carries the colour types and targets() fills them with color_types_of before caching the set;
- draw() reads them from the cached targets and does not classify the formats itself;
- the persistence gates still call targets_persistent() themselves (no frozen answer: frame_no moves at present);
- stats_tick() returns on a cached switch before it reads the frame.
python -I -B tests/test_draw_hot_path_cache.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def body_of(text, signature):
    start = text.index(signature)
    brace = text.index("{", start)
    end = brace + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[brace:end]


def main():
    src = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    failures = []

    def check(ok, label):
        print(("PASS " if ok else "FAIL ") + label)
        if not ok:
            failures.append(label)

    targets_struct = body_of(src, "struct Targets {")
    check("uint8_t color_type[8]" in targets_struct, "Targets carries the colour output types")

    resolve = body_of(src, "Targets targets(NvnState& s)")
    check("color_types_of(at, at.color_type)" in resolve, "targets() resolves the types with color_types_of")
    check(resolve.index("color_types_of(at, at.color_type)") < resolve.index("cached_targets = at"),
          "the types are resolved before the set is cached")

    draw = body_of(src, "void draw(VkCommandBuffer cmd, NvnState& s, const DrawArgs& d)")
    check("const uint8_t* types = at.color_type;" in draw, "draw() reads the cached types")
    check("color_types_of(at, types)" not in draw, "draw() does not classify the formats itself")
    check("targets_hold" not in draw, "draw() keeps no frozen persistence answer")
    for gate in ("if (legacy_translation_skip && targets_persistent(at))",
                 "if (!always_wait && translate_wait_ms >= 0 && targets_persistent(at)) {",
                 "const bool may_skip = !always_wait && targets_persistent(at) && !writes_memory(p);"):
        check(gate in draw, "gate: " + gate)

    tick = body_of(src, "void stats_tick()")
    check("static const bool enabled = stats_dir() != nullptr;" in tick, "stats_tick() caches the switch")
    check(tick.index("if (!enabled) return;") < tick.index("texture_frame_no()"),
          "the switch is answered before the frame is read")

    # the same answers as before: the classification and persistence functions themselves are untouched
    check("types[i] = (uint8_t)Shader::AttributeType::UnsignedInt;" in src, "color_types_of keeps its classification")
    check("bool targets_persistent(const Targets& at)" in src, "targets_persistent keeps its own answer")
    if failures:
        raise SystemExit(1)
    print("PASS: the draw hot path reads resolved values, from the same functions as before")


if __name__ == "__main__":
    main()
