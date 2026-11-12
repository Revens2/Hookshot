"""Apex Neural Rendering FG as a frame generation maker (gpu/vk/vk_anrfg.cpp), game-free and without Apex's DLL.

The provider is a DLSS-G DLL built by MSVC, and it reads the host's parameters through NGX's C++ interface. The renderer
is built by clang for MinGW, whose virtual tables are laid out otherwise, so vk_anrfg.cpp writes the table out by hand.
A slot in the wrong place is not a build error: the provider would read a float where the host put an integer, or call
Reset for a Get. So:

- the table's slots are in MSVC's order for NVIDIA's declaration (overloads of one name together, the last declared first);
- the table and its map, cut out of the source and built here, give back through each slot what was set through another
  (a number as any number type, a count past 2^53 exactly, a pointer as a pointer, nothing after Reset);
- the file includes nothing of NVIDIA's SDK and names the provider's Vulkan exports it calls;
- framegen_record asks ANR FG before DLSS-G, does not ask DLSS-G for a set ANR FG made, and names the set's maker.

    python -I -B tests/test_anrfg.py [--cxx g++]
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parent.parent

# NVSDK_NGX_Parameter's virtual functions as NVIDIA's header declares them (nvsdk_ngx_params.h): Set then Get, each over
# these types in this order, then Reset
DECLARED = ["ull", "float", "double", "uint", "int", "d3d11", "d3d12", "pointer"]

FIXTURE = r"""
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
/* ACTUAL */
int main() {
    Params p{&kParamsVtbl};
    const ParamsVtbl* t = p.vtbl;
    // (as the provider calls: through the table, by slot)
    t->set_uint(&p, "DLSSG.MultiFrameCount", 3u);
    unsigned u = 0; int i = 0; float f = 0; double d = 0; unsigned long long q = 0;
    CHECK(t->get_uint(&p, "DLSSG.MultiFrameCount", &u) == kNgxSuccess && u == 3);
    CHECK(t->get_int(&p, "DLSSG.MultiFrameCount", &i) == kNgxSuccess && i == 3);
    CHECK(t->get_float(&p, "DLSSG.MultiFrameCount", &f) == kNgxSuccess && f == 3.0f);
    CHECK(t->get_double(&p, "DLSSG.MultiFrameCount", &d) == kNgxSuccess && d == 3.0);
    CHECK(t->get_ull(&p, "DLSSG.MultiFrameCount", &q) == kNgxSuccess && q == 3);
    t->set_float(&p, "DLSSG.CameraNear", 0.25f);
    CHECK(t->get_float(&p, "DLSSG.CameraNear", &f) == kNgxSuccess && f == 0.25f);
    CHECK(t->get_uint(&p, "DLSSG.CameraNear", &u) == kNgxSuccess && u == 0);
    t->set_double(&p, "d", 2.5);
    CHECK(t->get_double(&p, "d", &d) == kNgxSuccess && d == 2.5);
    t->set_int(&p, "i", -7);
    CHECK(t->get_int(&p, "i", &i) == kNgxSuccess && i == -7 && t->get_float(&p, "i", &f) == kNgxSuccess && f == -7.0f);
    const unsigned long long big = (1ull << 60) + 1;   // (a frame's count past what a double holds)
    t->set_ull(&p, "DLSSG.BackbufferFrameID", big);
    CHECK(t->get_ull(&p, "DLSSG.BackbufferFrameID", &q) == kNgxSuccess && q == big);
    int thing = 0;
    void* out = nullptr;
    t->set_pointer(&p, "DLSSG.Backbuffer", &thing);
    CHECK(t->get_pointer(&p, "DLSSG.Backbuffer", &out) == kNgxSuccess && out == &thing);
    CHECK(t->get_d3d12(&p, "DLSSG.Backbuffer", &out) == kNgxSuccess && out == &thing);
    CHECK(ngx_failed(t->get_uint(&p, "DLSSG.Backbuffer", &u)));   // (a pointer is no number)
    CHECK(ngx_failed(t->get_pointer(&p, "DLSSG.CameraNear", &out)));   // (nor a number a pointer)
    t->set_pointer(&p, "DLSSG.HUDLess", nullptr);
    out = &thing;
    CHECK(t->get_pointer(&p, "DLSSG.HUDLess", &out) == kNgxSuccess && out == nullptr);   // (given as none)
    CHECK(ngx_failed(t->get_uint(&p, "never set", &u)) && ngx_failed(t->get_uint(&p, nullptr, &u)));
    CHECK(ngx_failed(t->get_uint(&p, "DLSSG.MultiFrameCount", nullptr)));
    t->reset(&p);
    CHECK(ngx_failed(t->get_uint(&p, "DLSSG.MultiFrameCount", &u)));
    CHECK(sizeof(ParamsVtbl) == 17 * sizeof(void*));
    return 0;
}
"""


def fail(why):
    sys.exit("FAIL: " + why)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    src = (ROOT / "gpu" / "vk" / "vk_anrfg.cpp").read_text(encoding="utf-8")

    # ---- the table's order ----
    table = src[src.index("struct ParamsVtbl {"):src.index("};", src.index("struct ParamsVtbl {"))]
    slots = re.findall(r"\(\*(\w+)\)", table)
    want = ["set_" + t for t in reversed(DECLARED)] + ["get_" + t for t in reversed(DECLARED)] + ["reset"]
    if slots != want:
        fail("ParamsVtbl is not in MSVC's order for NVSDK_NGX_Parameter:\n  has  %s\n  want %s" % (slots, want))
    filled = src[src.index("const ParamsVtbl kParamsVtbl = {"):src.index("static_assert(offsetof(Params, vtbl) == 0")]
    kinds = re.findall(r"\[\]\((?:const )?Params\* p(?:, const char\* n, ([^)]*?) v)?\)", filled)
    want_types = ["void*", "void*", "void*", "int", "unsigned", "double", "float", "unsigned long long",
                  "void**", "void**", "void**", "int*", "unsigned*", "double*", "float*", "unsigned long long*", ""]
    if kinds != want_types:
        fail("kParamsVtbl's functions do not take each slot's type:\n  has  %s\n  want %s" % (kinds, want_types))

    # ---- the table and its map, as built ----
    actual = src[src.index("using NgxResult = uint32_t;"):src.index("constexpr int kNgxVersionApi")] + \
        src[src.index("struct Params;"):src.index("// (the provider's exports")]
    with tempfile.TemporaryDirectory(prefix="anrfg-") as temp:
        cpp, exe = Path(temp) / "fixture.cpp", Path(temp) / "fixture.exe"
        cpp.write_text(FIXTURE.replace("/* ACTUAL */", actual), encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-invalid-offsetof",
                        "-Wno-missing-field-initializers", "-pthread",
                        str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=30)

    # ---- nothing of NVIDIA's SDK, and the provider's own exports ----
    if re.search(r'#include\s*[<"]nvsdk', src) or "HOOKSHOT_NGX" in src:
        fail("vk_anrfg.cpp takes NVIDIA's NGX SDK: the provider is called without it, in every build")
    for name in ("NVSDK_NGX_VULKAN_Init_Ext2", "NVSDK_NGX_VULKAN_Init", "NVSDK_NGX_VULKAN_PopulateParameters_Impl",
                 "NVSDK_NGX_VULKAN_CreateFeature1", "NVSDK_NGX_VULKAN_EvaluateFeature", "NVSDK_NGX_VULKAN_ReleaseFeature",
                 "NVSDK_NGX_VULKAN_Shutdown1"):
        if '"%s"' % name not in src:
            fail("vk_anrfg.cpp does not look for the provider's %s" % name)
    if "CP_ACP" in src:
        fail("vk_anrfg.cpp: a path through the ANSI code page (tests/test_utf8_paths.py)")

    # ---- who is asked first, and who is named ----
    fg = (ROOT / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    record = fg[fg.index("bool framegen_record("):fg.index("void framegen_queue(")]
    if "anr_fg_frames(" not in record or record.index("anr_fg_frames(") > record.index("dlss_fg_frames("):
        fail("framegen_record does not ask ANR FG before DLSS-G")
    if not re.search(r"if \(!by_anr\)\s*set->made = dlss_fg_frames\(", record):
        fail("framegen_record asks DLSS-G for a set ANR FG made")
    if "by_anr ? FgMaker::Anr" not in record:
        fail("framegen_record does not note ANR FG as the maker of its sets")
    head = (ROOT / "gpu" / "vk" / "vk_framegen.h").read_text(encoding="utf-8")
    makers = re.search(r"enum class FgMaker \{([^}]*)\}", head).group(1).replace(" ", "").split(",")
    if "Anr" not in makers:
        fail("vk_framegen.h: FgMaker has no Anr")
    if "maker_count[%d]" % len(makers) not in fg:
        fail("vk_framegen.cpp: maker_count does not hold a count for each of FgMaker's %d makers" % len(makers))
    menu = (ROOT / "gpu" / "vk" / "vk_menu.cpp").read_text(encoding="utf-8")
    if 'case FgMaker::Anr: return "ANR FG";' not in menu:
        fail("vk_menu.cpp: the Frame generation by row and the FPS counter do not name ANR FG")
    # ---- the row's own value: 2, ANR FG without DLSS-G, with the reason when another makes the frames ----
    if '"Frame generation by", &opts.framegen_by, 0, 2, 1, tx_framegen_by,' not in menu:
        fail("vk_menu.cpp: the Frame generation by row does not go up to 2 (ANR FG)")
    if 'if (v == 2) return "ANR FG";' not in menu or "if (v == 2) return hn_framegen_by_anr(variant);" not in menu:
        fail("vk_menu.cpp: the row's value 2 is not named ANR FG, or has no line of its own")
    if "opts.framegen_by == 0 || opts.framegen_by == 2" not in src:
        fail("vk_anrfg.cpp: ANR FG is not asked at the row's Auto and ANR FG")
    dlss = (ROOT / "gpu" / "vk" / "vk_dlss.cpp").read_text(encoding="utf-8")
    if "(forced || opts.framegen_by == 0)" not in dlss:
        fail("vk_dlss.cpp: DLSS-G is asked at other than the row's Auto (ANR FG is to run without it)")
    head_anr = (ROOT / "gpu" / "vk" / "vk_anrfg.h").read_text(encoding="utf-8")
    reasons = [r for r in re.findall(r"\b(ANR_SKIP_\w+)", head_anr[head_anr.index("enum AnrSkip"):head_anr.index("int anr_fg_skip")])
               if r not in ("ANR_SKIP_NONE", "ANR_SKIP_COUNT")]
    text = menu[menu.index("const char* framegen_anr_skip_text("):menu.index("const char* hn_framegen_by_anr(")]
    for r in reasons + ["ANR_SKIP_COUNT"]:
        if "case %s:" % r not in text:
            fail("vk_menu.cpp: the row has no text for %s" % r)
    frames = src[src.index("int anr_fg_frames("):src.index("void anr_fg_idle()")]
    for r in reasons:
        if r != "ANR_SKIP_PLATFORM" and "skip(%s" % r not in frames and "? %s" % r not in frames and ": %s" % r not in frames:
            fail("anr_fg_frames: a set not made for %s does not say so (anr_fg_skip)" % r)
    if not re.search(r"g\.made\+\+;\s*skip\(ANR_SKIP_NONE\);\s*return count;", frames):
        fail("anr_fg_frames: a set ANR FG made does not clear the reason the menu shows")
    for name, text, call in (("vk_temporal.cpp", None, "anr_fg_inputs("), ("vk_replay.cpp", None, "anr_fg_idle()"),
                             ("vk_main.cpp", None, "anr_fg_close()")):
        if call not in (ROOT / "gpu" / "vk" / name).read_text(encoding="utf-8"):
            fail("%s does not call %s" % (name, call.rstrip("(")))
    print("PASS: ANR FG: NGX's parameter table in MSVC's order and round-tripped by slot; asked before DLSS-G and named")


if __name__ == "__main__":
    main()
