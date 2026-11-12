// (hookshot) Apex Neural Rendering frame generation (ANR FG, github.com/apex-compute/ANR), a DLSS-G provider on any GPU: its
// nvngx_dlssg.dll, which the player puts in the game folder, loaded and called through its own NGX Vulkan exports
// (vk_anrfg.cpp). Nothing of Apex's or NVIDIA's is part of Hookshot. Windows only (the provider is a Windows DLL); elsewhere,
// and without the DLL, these say it is not there and the other makers go on as before.
#pragma once

#include "vk_internal.h"

namespace mkwvk {

struct TemporalFrame;

/// Whether it is asked for now: its DLL found, the Insert menu's Frame generation by row at Auto or ANR FG, RX_VK_ANR_FG
/// not 0, and not given up for the run (a creation or three sets in a row refused)
bool anr_fg_wanted();
/// Why the last set it was asked for went to the next maker, for the Frame generation by row at ANR FG (its texts are the
/// menu's, to be translated): ANR_SKIP_NONE when it made the set or nothing is known. most: the most made frames it makes a
/// set (ANR_SKIP_TOO_MANY)
enum AnrSkip {
    ANR_SKIP_NONE,
    ANR_SKIP_OFF,        // (RX_VK_ANR_FG=0 for the run)
    ANR_SKIP_NO_DLL,     // (the player put no nvngx_dlssg.dll where it is looked for)
    ANR_SKIP_START,      // (the DLL was not loaded, lacks the entry points, or its Init failed)
    ANR_SKIP_TOO_MANY,   // (more made frames than it makes)
    ANR_SKIP_INPUTS,     // (no picture from a temporal Upscaler this frame)
    ANR_SKIP_REFUSED,    // (it refused the set, or was given up after three)
    ANR_SKIP_PLATFORM,   // (not Windows: the provider is a Windows DLL)
    ANR_SKIP_COUNT
};
int anr_fg_skip(unsigned* most = nullptr);
/// The chain's frame (vk_temporal.cpp, beside dlss_fg_inputs): its depth and motion copied, its camera kept, for the next
/// present. Nothing unless it is asked for
void anr_fg_inputs(VkCommandBuffer cmd, const TemporalFrame& f, const float projection[16], const float view_projection[16],
                   const float clip_jitter[2], bool motion);
/// At the present (vk_framegen.cpp framegen_record, before DLSS-G): `count` frames between the last picture and this one
/// (color, the window's, w x h RGBA8) into out[0..count). Returns count, or 0 when it did not make them (the next maker does)
int anr_fg_frames(VkCommandBuffer cmd, VkImage color, VkImageView color_view, uint32_t w, uint32_t h, int count,
                  const VkImage* out, const VkImageView* out_views, bool blend, uint64_t sequence);
/// Frame generation at Off or Auto (no fixed multiplier): its feature, if made, let go. Each such present; nothing to do is
/// one load
void anr_fg_idle();
/// Before the process exits: its feature let go with the queue idle, and the provider shut down
void anr_fg_close();

} // namespace mkwvk
