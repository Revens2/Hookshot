"""A draw keeps the last program and pipeline when the material has not changed.

python -I -B tests/test_same_draw.py [--cxx g++]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    shader = (ROOT / "gpu" / "vk" / "vk_shader.cpp").read_text(encoding="utf-8")
    draw = (ROOT / "gpu" / "vk" / "vk_draw.cpp").read_text(encoding="utf-8")
    prog = block(shader, "bool prog_same(")
    pipe = (ROOT / "gpu" / "vk" / "pipe_memo.h").read_text(encoding="utf-8")
    pipe = "\n".join(line for line in pipe.splitlines() if not line.startswith("#pragma once"))   # (pasted into a .cpp)
    choose = block(shader, "const Program* program_for(")
    if choose.index("prog_same(") > choose.index("picks[") :
        raise SystemExit("program_for hashes the inputs before checking the last answer")
    pipes = block(draw, "PipeEntry* graphics_pipeline(")
    if pipes.index("pipe_memo::hit(") > pipes.index("mine.find("):
        raise SystemExit("graphics_pipeline hashes the key before checking the last pipeline")
    fixture = f'''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do {{ if (!(x)) {{ std::fprintf(stderr, "FAIL:%d %s\\n", __LINE__, #x); std::exit(1); }} }} while (0)
{prog}
{pipe}
int main() {{
    uint32_t va[16]{{}}, vb[16]{{}};
    uint8_t ct[8]{{}}, cu[8]{{}};
    uint32_t fm[8]{{}}, fn[8]{{}};
    uint32_t dv[16]{{}}, dw[16]{{}};
    uint64_t bl[8]{{}}, bm[8]{{}};
    va[3] = 7; vb[3] = 7; ct[1] = 2; cu[1] = 2; fm[0] = 9; fn[0] = 9; dv[1] = 4; dw[1] = 4; bl[0] = 11; bm[0] = 11;
    CHECK(prog_same(1, 5, 9, 3, 4, 8, 2, va, ct, 1, 5, 9, 3, 4, 8, 2, vb, cu));
    CHECK(!prog_same(1, 5, 9, 3, 4, 8, 2, va, ct, 1, 5, 9, 99, 4, 8, 2, vb, cu));
    CHECK(!prog_same(1, 5, 9, 3, 4, 8, 2, va, ct, 1, 5, 9, 3, 4, 1, 2, vb, cu));
    cu[1] = 3;
    CHECK(!prog_same(1, 5, 9, 3, 4, 8, 2, va, ct, 1, 5, 9, 3, 4, 8, 2, vb, cu));
    pipe_memo::From pa, pb;
    auto fill = [](pipe_memo::From& f, const uint32_t* fm, const uint32_t* va, const uint32_t* dv, const uint64_t* bl) {{
        f.prog = 1, f.variant = 2, f.topo = 3, f.samples = 4, f.n = 1, f.depth = 5, f.nvattr = 6;
        f.blend_en = 7, f.chmask = 8, f.poly = 9, f.clamp = 0, f.a2c = 1;
        for (int i = 0; i < 8; i++) f.formats[i] = fm[i], f.blend[i] = bl[i];
        for (int i = 0; i < 16; i++) f.vattr[i] = va[i], f.div[i] = dv[i];
    }};
    fill(pa, fm, va, dv, bl);
    fill(pb, fn, vb, dw, bm);
    CHECK(pipe_memo::same(pa, pb));
    pb.prog = 99;
    CHECK(!pipe_memo::same(pa, pb));
    pb.prog = 1;
    pb.depth = 1;
    CHECK(!pipe_memo::same(pa, pb));
    pb.depth = 5;
    fn[0] = 10;
    fill(pb, fn, vb, dw, bm);
    CHECK(!pipe_memo::same(pa, pb));
    std::printf("PASS: the same material keeps its program and pipeline\\n");
}}
'''
    with tempfile.TemporaryDirectory(prefix="same-draw-") as tmp:
        cpp = Path(tmp) / "fixture.cpp"
        exe = Path(tmp) / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
