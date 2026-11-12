// (hookshot) Apex Neural Rendering frame generation (ANR FG, github.com/apex-compute/ANR) as one of frame generation's
// makers (vk_anrfg.h). ANR FG is a DLSS-G provider: an nvngx_dlssg.dll with NGX's own entry points, which hosts such as
// OptiScaler load in NVIDIA's place. It runs on NVIDIA, AMD and Intel GPUs. Here the renderer is the host: the DLL is loaded
// by its path and called through its NVSDK_NGX_VULKAN_* exports, on the renderer's own device, with no NGX core, no
// NVIDIA SDK and no driver part in between. The renderer's frame generation (vk_framegen.cpp) keeps showing and pacing the
// frames; ANR FG makes them, from the window's picture and the temporal chain's depth, motion and camera, as DLSS-G does
// (vk_dlss.cpp), which this file follows step by step.
//
// Apex's files are not part of Hookshot and are not fetched by it: the player takes a release's package and puts the
// contents of its "Apex Neural Rendering FG\dlssg" folder (nvngx_dlssg.dll with the folders beside it) where it is looked
// for: RX_VK_ANR_FG_DIR=<folder> when set, the game folder's anr-fg, the game folder's "Apex Neural Rendering FG\dlssg",
// then %LOCALAPPDATA%\Hookshot\anr-fg. With the DLL there and the Insert menu's Frame generation by row at Auto or ANR FG
// (ANR FG: DLSS-G is not asked), it makes
// the sets at 2x and up before DLSS-G is asked; RX_VK_ANR_FG=0 leaves it out for a run. Where it does not make a set, the
// next maker does (DLSS-G, FSR 3.1's interpolation, the renderer's own), and the log says every 600 presents how many sets
// it made and why it did not make the others.
//
// What is taken for granted of the provider, none of it written down by Apex for Vulkan hosts, so each is said in the log
// when it turns out otherwise and has a switch for checks:
//  - NGX's parameter object is a C++ interface of MSVC's layout. This file is built by clang for MinGW, whose virtual
//    tables are laid out otherwise, so the table is written out by hand in MSVC's order (ParamsVtbl): overloads of one
//    name together, the last declared first.
//  - The images come in the layouts OptiScaler's DLSS-G replacements get them in (ANR's record/capture/capture.cpp): the
//    backbuffer SHADER_READ_ONLY_OPTIMAL, the depth and the motion TRANSFER_DST_OPTIMAL, the outputs GENERAL, and are in
//    them again after the call. RX_VK_ANR_FG_LAYOUTS=0: everything GENERAL, as the renderer keeps its images.
//  - The motion is in render pixels (scale 1), as DLSS-G takes it here. RX_VK_ANR_FG_MV_SCALE=<x> multiplies it.
#include "vk_anrfg.h"

#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "dlss_math.h"
#include "vk_menu.h"
#include "vk_temporal.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {

#ifdef _WIN32

namespace {

namespace fs = std::filesystem;

int env_switch(const char* name) {   // (-1: unset)
    const char* e = getenv(name);
    return e && *e ? atoi(e) : -1;
}

// ---- NGX's interface, as much of it as a DLSS-G provider is called with (NVIDIA's public headers' names and layouts) ----

using NgxResult = uint32_t;
constexpr NgxResult kNgxSuccess = 0x1, kNgxFail = 0xBAD00000u;
bool ngx_failed(NgxResult r) { return (r & 0xFFF00000u) == kNgxFail; }
constexpr int kNgxVersionApi = 0x15;          // (NVSDK_NGX_Version_API)
constexpr int kNgxFrameGeneration = 11;       // (NVSDK_NGX_Feature_FrameGeneration)
constexpr unsigned long long kAppId = 1;      // (any: a provider maps none; ANR's own replay host gives 1)

struct NgxHandle {
    unsigned id;
};
struct NgxImageView {   // (NVSDK_NGX_ImageViewInfo_VK)
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
};
struct NgxBuffer {      // (NVSDK_NGX_BufferInfo_VK)
    VkBuffer buffer;
    unsigned size;
};
struct NgxResource {    // (NVSDK_NGX_Resource_VK)
    union {
        NgxImageView image;
        NgxBuffer buffer;
    } resource;
    int type;           // (0: an image view)
    bool read_write;
};
NgxResource resource(VkImageView view, VkImage image, VkFormat format, uint32_t w, uint32_t h, bool write) {
    NgxResource r{};
    r.resource.image = {view, image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, format, w, h};
    r.type = 0;
    r.read_write = write;
    return r;
}

/// NVSDK_NGX_Parameter as the provider sees it: an object whose first word points at a table of functions, in the order
/// MSVC lays out
///     virtual void Set(const char*, unsigned long long / float / double / unsigned / int / ID3D11Resource* /
///                      ID3D12Resource* / void*);
///     virtual NVSDK_NGX_Result Get(const char*, the same as pointers) const;
///     virtual void Reset();
/// which is each name's overloads together, the last declared first.
struct Params;
struct ParamsVtbl {
    void (*set_pointer)(Params*, const char*, void*);
    void (*set_d3d12)(Params*, const char*, void*);
    void (*set_d3d11)(Params*, const char*, void*);
    void (*set_int)(Params*, const char*, int);
    void (*set_uint)(Params*, const char*, unsigned);
    void (*set_double)(Params*, const char*, double);
    void (*set_float)(Params*, const char*, float);
    void (*set_ull)(Params*, const char*, unsigned long long);
    NgxResult (*get_pointer)(const Params*, const char*, void**);
    NgxResult (*get_d3d12)(const Params*, const char*, void**);
    NgxResult (*get_d3d11)(const Params*, const char*, void**);
    NgxResult (*get_int)(const Params*, const char*, int*);
    NgxResult (*get_uint)(const Params*, const char*, unsigned*);
    NgxResult (*get_double)(const Params*, const char*, double*);
    NgxResult (*get_float)(const Params*, const char*, float*);
    NgxResult (*get_ull)(const Params*, const char*, unsigned long long*);
    void (*reset)(Params*);
};
/// A value by its name: a number readable as any number type (NGX's parameters have no declared type), or a pointer
struct Value {
    bool pointer = false, whole = false;
    double number = 0;
    unsigned long long integer = 0;   // (whole: the number exactly, a frame's count past 2^53 too)
    void* address = nullptr;
};
struct Params {
    const ParamsVtbl* vtbl;   // (the first word: what the provider calls through)
    mutable std::mutex mu;    // (the provider may read from a thread of its own)
    std::unordered_map<std::string, Value> values;

    void put(const char* name, const Value& v) {
        if (!name) return;
        std::lock_guard lk(mu);
        values[name] = v;
    }
    template <typename T>
    void number(const char* name, T v) {
        Value x;
        x.number = (double)v;
        if constexpr (std::is_integral_v<T>) x.whole = true, x.integer = (unsigned long long)(long long)v;
        put(name, x);
    }
    void address(const char* name, const void* p) {
        Value x;
        x.pointer = true, x.address = (void*)p;
        put(name, x);
    }
    template <typename T>
    NgxResult read(const char* name, T* out) const {
        if (!name || !out) return kNgxFail;
        std::lock_guard lk(mu);
        const auto it = values.find(name);
        if (it == values.end() || it->second.pointer) return kNgxFail;
        const Value& v = it->second;
        if constexpr (std::is_integral_v<T>) *out = v.whole ? (T)v.integer : (T)(long long)v.number;
        else *out = (T)v.number;
        return kNgxSuccess;
    }
    NgxResult read_address(const char* name, void** out) const {
        if (!name || !out) return kNgxFail;
        std::lock_guard lk(mu);
        const auto it = values.find(name);
        if (it == values.end() || !it->second.pointer) return kNgxFail;
        *out = it->second.address;
        return kNgxSuccess;
    }
};
const ParamsVtbl kParamsVtbl = {
    [](Params* p, const char* n, void* v) { p->address(n, v); },
    [](Params* p, const char* n, void* v) { p->address(n, v); },
    [](Params* p, const char* n, void* v) { p->address(n, v); },
    [](Params* p, const char* n, int v) { p->number(n, v); },
    [](Params* p, const char* n, unsigned v) { p->number(n, v); },
    [](Params* p, const char* n, double v) { p->number(n, v); },
    [](Params* p, const char* n, float v) { p->number(n, v); },
    [](Params* p, const char* n, unsigned long long v) { p->number(n, v); },
    [](const Params* p, const char* n, void** v) { return p->read_address(n, v); },
    [](const Params* p, const char* n, void** v) { return p->read_address(n, v); },
    [](const Params* p, const char* n, void** v) { return p->read_address(n, v); },
    [](const Params* p, const char* n, int* v) { return p->read(n, v); },
    [](const Params* p, const char* n, unsigned* v) { return p->read(n, v); },
    [](const Params* p, const char* n, double* v) { return p->read(n, v); },
    [](const Params* p, const char* n, float* v) { return p->read(n, v); },
    [](const Params* p, const char* n, unsigned long long* v) { return p->read(n, v); },
    [](Params* p) {
        std::lock_guard lk(p->mu);
        p->values.clear();
    },
};
static_assert(offsetof(Params, vtbl) == 0, "the provider reads the table's address from the object's first word");

// (the provider's exports: NGX's Vulkan entry points as a host calls them, ANR's record/capture/capture.cpp)
using PfnInit = NgxResult (*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
                              PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, const void* info, int version);
using PfnInitExt2 = NgxResult (*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
                                  PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int version, const void* info);
using PfnPopulate = NgxResult (*)(Params*);
using PfnCreate1 = NgxResult (*)(VkDevice, VkCommandBuffer, int feature, Params*, NgxHandle**);
using PfnEvaluate = NgxResult (*)(VkCommandBuffer, const NgxHandle*, Params*, void* progress);
using PfnRelease = NgxResult (*)(NgxHandle*);
using PfnShutdown1 = NgxResult (*)(VkDevice);

constexpr const wchar_t* kRuntimeName = L"nvngx_dlssg.dll";
constexpr int kRefusalsOff = 3;   // (sets refused in a row, or creations failed, before it is given up for the run)

/// The game folder (vk_dlss.cpp's rule: bin\hookshot.exe beside image.bin is the folder above bin)
fs::path game_dir() {
    std::wstring exe(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, exe.data(), (DWORD)exe.size());
    if (!n || n >= exe.size()) return {};
    exe.resize(n);
    fs::path dir = fs::path(exe).parent_path();
    if (!_wcsicmp(dir.filename().c_str(), L"bin") && GetFileAttributesW((dir.parent_path() / L"image.bin").c_str()) != INVALID_FILE_ATTRIBUTES)
        dir = dir.parent_path();
    return dir;
}
fs::path env_path(const wchar_t* name) {
    const wchar_t* e = _wgetenv(name);
    return e && *e ? fs::path(e) : fs::path();
}
/// A path for our log (UTF-8)
std::string shown(const fs::path& p) {
    const std::wstring& w = p.native();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
/// The provider's DLL, looked for once: empty when the player put none there (ANR FG is then not asked at all)
const fs::path& runtime() {
    static const fs::path found = [] {
        std::vector<fs::path> dirs;
        if (fs::path e = env_path(L"RX_VK_ANR_FG_DIR"); !e.empty()) dirs.push_back(e);
        if (const fs::path game = game_dir(); !game.empty() && game.is_absolute()) {
            dirs.push_back(game / L"anr-fg");
            dirs.push_back(game / L"Apex Neural Rendering FG" / L"dlssg");
        }
        if (fs::path l = env_path(L"LOCALAPPDATA"); !l.empty()) dirs.push_back(l / L"Hookshot" / L"anr-fg");
        for (const auto& d : dirs) {
            std::error_code ec;
            if (d.is_absolute() && fs::is_regular_file(d / kRuntimeName, ec)) return d / kRuntimeName;
        }
        return fs::path();
    }();
    return found;
}

struct FgImage {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct Fg {
    std::mutex mu;
    bool tried = false, ok = false;   // (the provider's start: once)
    HMODULE lib = nullptr;
    PfnCreate1 create = nullptr;
    PfnEvaluate evaluate = nullptr;
    PfnRelease release = nullptr;
    PfnShutdown1 shutdown = nullptr;
    Params params{&kParamsVtbl};
    unsigned most = 1;                // (the most frames it makes between two of the game's: MultiFrameCountMax)
    NgxHandle* feature = nullptr;
    uint32_t fw = 0, fh = 0;          // (the picture its feature was made for)
    int create_failures = 0, refusals = 0;
    bool closed = false;
    // (read without mu, each present: a feature held (anr_fg_idle), and given up for the run (anr_fg_wanted))
    std::atomic<bool> held{false}, off{false};
    // this frame's inputs (anr_fg_inputs): the depth and motion copied, the camera NGX's way
    FgImage depth, motion;
    uint32_t rw = 0, rh = 0;
    float view_to_clip[4][4], clip_to_view[4][4], to_lens[4][4], to_prev[4][4], from_prev[4][4];
    float pos[3], up[3], right[3], fwd[3];
    float jitter[2]{}, near_z = 0.1f, far_z = 1000.0f, fov = 1.0f, aspect = 16.0f / 9.0f;
    bool inputs_reset = true, depth_reversed = false;
    float last_vp[16]{};
    bool have_last = false;
    uint64_t inputs = 0, used = 0;    // (how many inputs were taken, and the last one a present used)
    uint64_t last_sequence = 0;       // (the present sequence it last made a set at: one between them is a new history)
    uint64_t presents = 0, made = 0, no_inputs = 0, refused = 0, too_many = 0, not_blended = 0, resets = 0;
};
Fg& fgs() {
    static Fg f;
    return f;
}

/// Whether it may run in this process at all: its DLL there, and not RX_VK_ANR_FG=0
bool possible() {
    static const bool yes = env_switch("RX_VK_ANR_FG") != 0 && !runtime().empty();
    return yes;
}
/// Whether it is asked for now: the Frame generation by row at Auto or ANR FG (read each time: it changes while the game runs)
bool asked() { return possible() && (opts.framegen_by == 0 || opts.framegen_by == 2); }

// Why the last set it was asked for went to the next maker (anr_fg_skip); none when it made it. A set with nothing to blend
// from (a loading hitch) leaves it as it was
std::atomic<int> fg_skip{ANR_SKIP_NONE};
std::atomic<unsigned> fg_skip_most{0};
int skip(int why) {
    fg_skip.store(why, std::memory_order_relaxed);
    return 0;
}

constexpr float kIdentity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
/// A 4x4 inverse in doubles (dlss_math's Gauss-Jordan: identity * inverse(m)); false when m is singular
bool inverse(const float* m, float* out) { return dlss_math::reprojection(m, kIdentity, out); }
void mul(const float* a, const float* b, float* out) {   // (row major: out = a * b)
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double sum = 0;
            for (int k = 0; k < 4; k++) sum += (double)a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = (float)sum;
        }
}
/// NGX's matrices are rows of a row-vector convention (position * M, as Streamline's float4x4): the game's, transposed
void put(float out[4][4], const float* m) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out[r][c] = m[c * 4 + r];
}

/// The provider loaded and started on the renderer's device, once (under g.mu). The reason is logged when not
bool start(Fg& g) {
    if (g.tried) return g.ok;
    g.tried = true;
    g.off = true;   // (until it has started, below)
    const fs::path& dll = runtime();
    if (!dev.instance || !dev.physical || !dev.device) return false;
    // (its own folder first in the search: the files it reads lie beside it. It needs no other DLL than Windows' own)
    g.lib = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g.lib) {
        logf("vk: ANR FG: %s not loaded (Windows error %lu): off", shown(dll).c_str(), (unsigned long)GetLastError());
        return false;
    }
    const auto get = [&g](const char* name) { return (void*)GetProcAddress(g.lib, name); };
    const auto init = (PfnInit)get("NVSDK_NGX_VULKAN_Init");
    const auto init2 = (PfnInitExt2)get("NVSDK_NGX_VULKAN_Init_Ext2");
    const auto populate = (PfnPopulate)get("NVSDK_NGX_VULKAN_PopulateParameters_Impl");
    g.create = (PfnCreate1)get("NVSDK_NGX_VULKAN_CreateFeature1");
    g.evaluate = (PfnEvaluate)get("NVSDK_NGX_VULKAN_EvaluateFeature");
    g.release = (PfnRelease)get("NVSDK_NGX_VULKAN_ReleaseFeature");
    g.shutdown = (PfnShutdown1)get("NVSDK_NGX_VULKAN_Shutdown1");
    if ((!init && !init2) || !g.create || !g.evaluate || !g.release) {
        logf("vk: ANR FG: %s lacks NGX's Vulkan entry points (Init, CreateFeature1, EvaluateFeature, ReleaseFeature): off",
             shown(dll).c_str());
        return false;
    }
    // (the renderer's own entry points, as a game hands them to NGX: the provider makes its calls on this device)
    const std::wstring data = dll.parent_path().native();
    const NgxResult r = init2 ? init2(kAppId, data.c_str(), dev.instance, dev.physical, dev.device, instance_proc(),
                                      vkGetDeviceProcAddr, kNgxVersionApi, nullptr)
                              : init(kAppId, data.c_str(), dev.instance, dev.physical, dev.device, instance_proc(),
                                     vkGetDeviceProcAddr, nullptr, kNgxVersionApi);
    if (ngx_failed(r)) {
        logf("vk: ANR FG: its Init failed (%#x; its own log, nr_fg-*.log beside the DLL, says why): off", (unsigned)r);
        return false;
    }
    if (populate) populate(&g.params);
    unsigned most = 0;
    g.params.read("DLSSG.MultiFrameCountMax", &most);
    g.most = most ? most : 1;
    // (RX_VK_ANR_FG_MOST=<n>, for checks: no more than n)
    if (const char* cap = getenv("RX_VK_ANR_FG_MOST"); cap && atoi(cap) >= 1 && (unsigned)atoi(cap) < g.most)
        g.most = (unsigned)atoi(cap);
    logf("vk: ANR FG: %s started, up to %u made frames between two of the game's%s", shown(dll).c_str(), g.most,
         most ? "" : " (it did not say how many: 1)");
    g.ok = true;
    g.off = false;
    return true;
}

/// One of its own images (the depth or motion copy), made again at another size; the old one freed once work in flight is done
bool fg_image(VkCommandBuffer cmd, FgImage& im, VkFormat format, uint32_t w, uint32_t h) {
    if (im.image && im.w == w && im.h == h && im.format == format) return true;
    if (im.image) {
        const FgImage old = im;
        temporal_later([old] {
            vkDestroyImageView(dev.device, old.view, nullptr);
            destroy_image(old.image, old.alloc);
        });
        im = {};
    }
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!create_image(ci, im.image, im.alloc)) return im = {}, false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev.device, &vi, nullptr, &im.view) != VK_SUCCESS) {
        destroy_image(im.image, im.alloc);
        return im = {}, false;
    }
    im.w = w, im.h = h, im.format = format;
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = vi.subresourceRange;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
    return true;
}

void fg_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 m{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    m.srcStageMask = m.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    m.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    m.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &m;
    vkCmdPipelineBarrier2(cmd, &di);
}

/// The layouts the provider takes its inputs in (the top of this file), and back: the renderer keeps its images GENERAL
void layouts(VkCommandBuffer cmd, const VkImage* images, const VkImageLayout* theirs, int n, bool back) {
    VkImageMemoryBarrier2 b[3]{};
    for (int i = 0; i < n; i++) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b[i].srcStageMask = b[i].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b[i].srcAccessMask = b[i].dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        b[i].oldLayout = back ? theirs[i] : VK_IMAGE_LAYOUT_GENERAL;
        b[i].newLayout = back ? VK_IMAGE_LAYOUT_GENERAL : theirs[i];
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].image = images[i];
        b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = (uint32_t)n, di.pImageMemoryBarriers = b;
    vkCmdPipelineBarrier2(cmd, &di);
}

/// The feature let go: the queue idle first (it may be in a set the display thread has not shown yet)
void release(Fg& g) {
    if (!g.feature) return;
    {
        std::lock_guard qk(dev.present_mu);
        device_wait_idle_locked();
    }
    g.release(g.feature);
    g.feature = nullptr;
    g.held = false;
    g.fw = g.fh = 0;
}

/// Given up for the run: anr_fg_wanted says no from here, and the next maker makes the sets
void give_up(Fg& g, const char* why, NgxResult r) {
    g.off = true;
    logf("vk: ANR FG: %s (%#x): off for this run, the next maker makes the frames (its own log, nr_fg-*.log beside the DLL, "
         "says why)", why, (unsigned)r);
}

}  // namespace

bool anr_fg_wanted() { return asked() && !fgs().off.load(std::memory_order_relaxed); }

int anr_fg_skip(unsigned* most) {
    if (most) *most = fg_skip_most.load(std::memory_order_relaxed);
    return fg_skip.load(std::memory_order_relaxed);
}

void anr_fg_inputs(VkCommandBuffer cmd, const TemporalFrame& f, const float projection[16], const float view_projection[16],
                   const float clip_jitter[2], bool motion) {
    if (!anr_fg_wanted()) return;
    Fg& g = fgs();
    std::lock_guard lk(g.mu);
    if (g.closed) return;
    if (opts.framegen < 2) return void(g.have_last = false);   // (not a fixed multiplier: the next inputs start a history)
    const uint32_t rw = f.render_w, rh = f.render_h;
    if (!f.depth.image || !f.motion.image || !rw || !rh) return;
    // the camera without the jitter: view = P^-1 * VP, the camera's frame its inverse
    float p[16], vp[16], ip[16], view[16], cam[16];
    std::memcpy(p, projection, sizeof p), std::memcpy(vp, view_projection, sizeof vp);
    dlss_math::unjitter(p, clip_jitter), dlss_math::unjitter(vp, clip_jitter);
    if (!inverse(p, ip)) return;
    mul(ip, vp, view);
    if (!inverse(view, cam)) return;
    const bool history = g.have_last && motion && !f.reset;
    float to_prev[16], from_prev[16];
    if (!history || !dlss_math::reprojection(vp, g.last_vp, to_prev)) std::memcpy(to_prev, kIdentity, sizeof kIdentity);
    if (!inverse(to_prev, from_prev)) return;
    std::memcpy(g.last_vp, vp, sizeof vp), g.have_last = true;
    put(g.view_to_clip, p), put(g.clip_to_view, ip), put(g.to_lens, kIdentity);
    put(g.to_prev, to_prev), put(g.from_prev, from_prev);
    for (int i = 0; i < 3; i++) {   // (the view's inverse: OpenGL's camera, looking down -z)
        g.pos[i] = cam[i * 4 + 3];
        g.right[i] = cam[i * 4 + 0];
        g.up[i] = cam[i * 4 + 1];
        g.fwd[i] = -cam[i * 4 + 2];
    }
    g.jitter[0] = f.jitter[0], g.jitter[1] = f.jitter[1];
    g.depth_reversed = f.depth_reversed;
    g.near_z = f.near_z > 0 ? f.near_z : 0.1f;
    g.far_z = f.far_z > g.near_z ? f.far_z : 100000.0f;
    g.fov = f.fov_y > 0 ? f.fov_y : 1.0f;
    g.aspect = p[0] != 0 ? std::fabs(p[5] / p[0]) : 16.0f / 9.0f;
    g.inputs_reset |= !history;
    // the depth and the motion, the frame's part: copied, so the present reads this frame's whatever the chain does next
    if (!fg_image(cmd, g.depth, VK_FORMAT_R32_SFLOAT, rw, rh) || !fg_image(cmd, g.motion, VK_FORMAT_R16G16_SFLOAT, rw, rh)) return;
    fg_barrier(cmd);
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                        {rw, rh, 1}};
    vkCmdCopyImage(cmd, f.depth.image, VK_IMAGE_LAYOUT_GENERAL, g.depth.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vkCmdCopyImage(cmd, f.motion.image, VK_IMAGE_LAYOUT_GENERAL, g.motion.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    fg_barrier(cmd);
    g.rw = rw, g.rh = rh;
    g.inputs++;
    static bool told;
    if (!told) {
        told = true;
        logf("vk: ANR FG: inputs %ux%u (near %.3f far %.1f fov %.1f deg)", rw, rh, g.near_z, g.far_z, g.fov * 57.29578f);
    }
}

int anr_fg_frames(VkCommandBuffer cmd, VkImage color, VkImageView color_view, uint32_t w, uint32_t h, int count,
                  const VkImage* out, const VkImageView* out_views, bool blend, uint64_t sequence) {
    if (!possible()) return skip(env_switch("RX_VK_ANR_FG") == 0 ? ANR_SKIP_OFF : ANR_SKIP_NO_DLL);
    Fg& g = fgs();
    std::lock_guard lk(g.mu);
    if (g.closed) return 0;
    if (!asked()) {   // (turned off in the menu: its feature let go, its memory back)
        if (g.feature) release(g), logf("vk: ANR FG: off (Frame generation by): its feature let go");
        return 0;
    }
    if (++g.presents % 600 == 0) {
        logf("vk: ANR FG: of the last 600 presents %llu made by ANR FG (%llu told to reset); the next maker made the rest: "
             "%llu without new inputs (no temporal Upscaler this frame), %llu after a gap, %llu asked more than %u frames, "
             "%llu refused",
             (unsigned long long)g.made, (unsigned long long)g.resets, (unsigned long long)g.no_inputs,
             (unsigned long long)g.not_blended, (unsigned long long)g.too_many, g.most, (unsigned long long)g.refused);
        g.made = g.no_inputs = g.not_blended = g.too_many = g.refused = g.resets = 0;
    }
    if (count < 1 || !start(g) || g.off.load(std::memory_order_relaxed)) {
        if (g.feature) release(g);   // (given up: the refused sets' feature and its memory, let go once)
        return skip(g.ok ? ANR_SKIP_REFUSED : ANR_SKIP_START);
    }
    if (!blend) return g.not_blended++, 0;
    if ((unsigned)count > g.most) {
        static bool told = false;   // (once a run: the 600-present lines count them)
        if (!told) {
            told = true;
            logf("vk: ANR FG: %d made frames asked, but it makes %u at most (max_generated in its nr_fg.ini): the next maker "
                 "makes them", count, g.most);
        }
        fg_skip_most.store(g.most, std::memory_order_relaxed);
        return g.too_many++, skip(ANR_SKIP_TOO_MANY);
    }
    if (g.inputs == g.used || !g.rw) return g.no_inputs++, skip(ANR_SKIP_INPUTS);
    g.used = g.inputs;
    Params& p = g.params;
    if (g.feature && (g.fw != w || g.fh != h)) release(g);
    if (!g.feature) {
        p.number("CreationNodeMask", 1u), p.number("VisibilityNodeMask", 1u);
        p.number("Width", w), p.number("Height", h);
        p.number("DLSSG.Width", w), p.number("DLSSG.Height", h);
        p.number("DLSSG.BackbufferFormat", (unsigned)VK_FORMAT_R8G8B8A8_UNORM);   // (the window's picture, vk_framegen.cpp's frames)
        p.number("DLSSG.InternalWidth", g.rw), p.number("DLSSG.InternalHeight", g.rh);
        p.number("DLSSG.DynamicResolution", 0u);
        ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (the provider records into cmd itself)
        const NgxResult r = g.create(dev.device, cmd, kNgxFrameGeneration, &p, &g.feature);
        note_work();
        if (ngx_failed(r) || !g.feature) {
            g.feature = nullptr;
            logf("vk: ANR FG: the feature was not made for %ux%u (%#x)", w, h, (unsigned)r);
            if (++g.create_failures >= kRefusalsOff) give_up(g, "its feature was not made three times", r);
            return g.refused++, skip(ANR_SKIP_REFUSED);
        }
        g.fw = w, g.fh = h;
        g.held = true;
        g.last_sequence = 0;
        logf("vk: ANR FG: feature made for %ux%u from %ux%u inputs, %d made frame(s) a set", w, h, g.rw, g.rh, count);
    }
    NgxResource back = resource(color_view, color, VK_FORMAT_R8G8B8A8_UNORM, w, h, false);
    NgxResource depth = resource(g.depth.view, g.depth.image, g.depth.format, g.rw, g.rh, false);
    NgxResource mvecs = resource(g.motion.view, g.motion.image, g.motion.format, g.rw, g.rh, false);
    p.address("DLSSG.Backbuffer", &back), p.address("DLSSG.Depth", &depth), p.address("DLSSG.MVecs", &mvecs);
    for (const char* none : {"DLSSG.HUDLess", "DLSSG.UI", "DLSSG.UIAlpha", "DLSSG.BidirectionalDistortionField",
                             "DLSSG.OutputReal", "DLSSG.OutputDisableInterpolation"})
        p.address(none, nullptr);
    p.address("DLSSG.CameraViewToClip", g.view_to_clip), p.address("DLSSG.ClipToCameraView", g.clip_to_view);
    p.address("DLSSG.ClipToLensClip", g.to_lens);
    p.address("DLSSG.ClipToPrevClip", g.to_prev), p.address("DLSSG.PrevClipToClip", g.from_prev);
    p.number("DLSSG.JitterOffsetX", g.jitter[0]), p.number("DLSSG.JitterOffsetY", g.jitter[1]);
    static const float mv_scale = getenv("RX_VK_ANR_FG_MV_SCALE") ? (float)atof(getenv("RX_VK_ANR_FG_MV_SCALE")) : 1.0f;
    p.number("DLSSG.MvecScaleX", mv_scale), p.number("DLSSG.MvecScaleY", mv_scale);
    p.number("DLSSG.CameraPinholeOffsetX", 0.0f), p.number("DLSSG.CameraPinholeOffsetY", 0.0f);
    p.number("DLSSG.CameraPosX", g.pos[0]), p.number("DLSSG.CameraPosY", g.pos[1]), p.number("DLSSG.CameraPosZ", g.pos[2]);
    p.number("DLSSG.CameraUpX", g.up[0]), p.number("DLSSG.CameraUpY", g.up[1]), p.number("DLSSG.CameraUpZ", g.up[2]);
    p.number("DLSSG.CameraRightX", g.right[0]), p.number("DLSSG.CameraRightY", g.right[1]);
    p.number("DLSSG.CameraRightZ", g.right[2]);
    p.number("DLSSG.CameraFwdX", g.fwd[0]), p.number("DLSSG.CameraFwdY", g.fwd[1]), p.number("DLSSG.CameraFwdZ", g.fwd[2]);
    p.number("DLSSG.CameraNear", g.near_z), p.number("DLSSG.CameraFar", g.far_z);
    p.number("DLSSG.CameraFOV", g.fov), p.number("DLSSG.CameraAspectRatio", g.aspect);
    p.number("DLSSG.ColorBuffersHDR", 0u);
    p.number("DLSSG.DepthInverted", g.depth_reversed ? 1u : 0u);   // (the chain's depth: reversed for FSR 3.1 on the chain)
    p.number("DLSSG.CameraMotionIncluded", 1u);                    // (its motion is the whole of it: camera and actors)
    // (a new history after a present it did not make, a cut the chain saw, or a new feature)
    const bool reset = g.inputs_reset || !g.last_sequence || sequence != g.last_sequence + 1;
    g.resets += reset;
    p.number("DLSSG.Reset", reset ? 1u : 0u);
    p.number("DLSSG.AutomodeOverrideReset", 0u), p.number("DLSSG.NotRenderingGameFrames", 0u);
    p.number("DLSSG.OrthoProjection", 0u);
    p.number("DLSSG.MvecInvalidValue", FLT_MAX), p.number("DLSSG.MvecDilated", 0u);
    p.number("DLSSG.MenuDetectionEnabled", 0u);
    // (each subrect the whole of its image, by its size: the provider takes a subrect of 0 x 0 as it is, an empty picture.
    // With zeros (what NVIDIA's own helper leaves for "all of it") it compared empty pictures, called every frame a scene
    // cut and dropped every made frame: nr_fg's log, "subrects: backbuffer 0,0 0x0", 2026-11-12)
    const auto rect = [&p](const char* name, uint32_t rw_, uint32_t rh_) {
        const std::string n(name);
        p.number((n + "BaseX").c_str(), 0u), p.number((n + "BaseY").c_str(), 0u);
        p.number((n + "Width").c_str(), rw_), p.number((n + "Height").c_str(), rh_);
    };
    rect("DLSSG.InputBackbufferSubrect", w, h), rect("DLSSG.BackbufferSubrect", w, h);
    rect("DLSSG.OutputInterpolatedSubrect", w, h);
    rect("DLSSG.MVecsSubrect", g.rw, g.rh), rect("DLSSG.DepthSubrect", g.rw, g.rh);
    p.number("DLSSG.MultiFrameCount", (unsigned)count);
    p.number("DLSSG.BackbufferFrameID", (unsigned long long)sequence);
    static const bool their_layouts = env_switch("RX_VK_ANR_FG_LAYOUTS") != 0;
    const VkImage inputs[3] = {color, g.depth.image, g.motion.image};
    static const VkImageLayout theirs[3] = {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL};
    fg_barrier(cmd);
    // (each made frame starts as the game's own picture: a slot the provider drops (a scene cut it sees, with no
    // OutputDisableInterpolation here to say so) then shows this frame again rather than what the image held before)
    for (int k = 0; k < count; k++) {
        const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                            {w, h, 1}};
        vkCmdCopyImage(cmd, color, VK_IMAGE_LAYOUT_GENERAL, out[k], VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    fg_barrier(cmd);
    if (their_layouts) layouts(cmd, inputs, theirs, 3, false);
    int made = 0;
    NgxResult r = kNgxSuccess;
    for (int k = 1; k <= count; k++) {
        NgxResource frame = resource(out_views[k - 1], out[k - 1], VK_FORMAT_R8G8B8A8_UNORM, w, h, true);
        p.address("DLSSG.OutputInterpolated", &frame);
        p.number("DLSSG.MultiFrameIndex", (unsigned)k);
        ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (the provider records into cmd itself)
        r = g.evaluate(cmd, g.feature, &p, nullptr);
        note_work();
        if (ngx_failed(r)) break;
        made = k;
    }
    // (the resources above are this call's: none of their addresses is left in the map for a later read)
    for (const char* mine : {"DLSSG.Backbuffer", "DLSSG.Depth", "DLSSG.MVecs", "DLSSG.OutputInterpolated"}) p.address(mine, nullptr);
    if (their_layouts) layouts(cmd, inputs, theirs, 3, true);
    fg_barrier(cmd);   // (its writes done before the next maker, or the present, reads or writes the same images)
    if (made != count) {
        static int told = 0;
        if (told++ < 8) logf("vk: ANR FG: frame %d of %d was refused (%#x)", made + 1, count, (unsigned)r);
        g.last_sequence = 0;
        if (++g.refusals >= kRefusalsOff) give_up(g, "three sets in a row refused", r);
        return g.refused++, skip(ANR_SKIP_REFUSED);   // (a half-made set: the next maker makes all of it)
    }
    g.refusals = 0;
    g.inputs_reset = false;
    g.last_sequence = sequence;
    g.made++;
    skip(ANR_SKIP_NONE);
    return count;
}

void anr_fg_idle() {
    Fg& g = fgs();
    if (!g.held.load(std::memory_order_relaxed)) return;
    std::lock_guard lk(g.mu);
    if (g.feature) release(g), logf("vk: ANR FG: frame generation Off or Auto: its feature let go");
}

void anr_fg_close() {
    if (!possible()) return;
    Fg& g = fgs();
    std::lock_guard lk(g.mu);
    if (g.closed) return;
    g.closed = true;
    if (!g.ok) return;
    // (the swapchain's lock for the queue's idle, waited for a while: a present stuck at the exit leaves the provider to the
    // process's end rather than hanging the exit, as DLSS-G's close does)
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    std::unique_lock qk(dev.present_mu, std::defer_lock);
    while (!qk.try_lock() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!qk.owns_lock()) {
        logf("vk: ANR FG: the queue stayed busy at the exit: its feature left to the process's end");
        return;
    }
    device_wait_idle_locked();
    if (g.feature) g.release(g.feature), g.feature = nullptr;
    if (g.shutdown) g.shutdown(dev.device);
    logf("vk: ANR FG: let go for the exit");
}

#else  // (not Windows: the provider is a Windows DLL)

bool anr_fg_wanted() { return false; }
int anr_fg_skip(unsigned* most) {
    if (most) *most = 0;
    return ANR_SKIP_PLATFORM;
}
void anr_fg_inputs(VkCommandBuffer, const TemporalFrame&, const float*, const float*, const float*, bool) {}
int anr_fg_frames(VkCommandBuffer, VkImage, VkImageView, uint32_t, uint32_t, int, const VkImage*, const VkImageView*, bool,
                  uint64_t) {
    return 0;
}
void anr_fg_idle() {}
void anr_fg_close() {}

#endif

} // namespace mkwvk
