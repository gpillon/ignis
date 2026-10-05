// ignis kernel leaf -- the Flash-Next GDN layer core (spec flash-next/04, GitHub #302, slice S2):
// OURS (ADR 0043), no port claim. fn_gdn_layer (flash_next_internal.h) is the program's entry
// point; this header names the stage under it so the kernel-leaf tests drive it on their own state
// buffers, without a seq pool.
//
// The oracle is transformers' Qwen4ExpTextGatedDeltaNet (models/qwen4_exp/modeling_qwen4_exp.py):
//
//   qkv = in_proj_qkv(x), z = in_proj_z(x), a = in_proj_a(x), b = in_proj_b(x)   BF16 each
//   qkv = silu(causal_conv1d(qkv))            depthwise, kernel 4, over the 10240 channels
//   q, k, v = split(qkv, [2048, 2048, 6144])  16 / 16 / 48 heads of 128
//   beta = bf16(sigmoid(b));  g = -exp(A_log) * softplus(a + dt_bias)          fp32
//   o = gated_delta_rule(l2norm(q), l2norm(k), v, g, beta) / sqrt(128)        value head h reads
//                                                                             q/k head h / 3
//   o = rmsnorm_gated(o, z): bf16(bf16(weight * bf16(o * rsqrt(mean(o^2) + eps))) * sigmoid(z))
//   y = out_proj(o)
//
// The projections are fn_linear; the convolution and the recurrence are the vendored ninfer ops,
// whose geometry is the 27B's (16 / 48 heads of 128, kernel 4); the gating and the sigmoid-gated
// norm are ours (the 27B's gating takes FP32 A_log / dt_bias and its norm gates with SiLU).
//
// A call is either one token per lane (decode, and a one-token prefill chunk), run on the
// vendored snapshot forms, which read and write each lane's state at its slot from device memory
// (graph-safe); or one lane of several tokens (prefill, eager), whose state is gathered from its
// slot into scratch, advanced by the plain forms and scattered back.

#pragma once

#include "flash_next_internal.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::flash_next::gdn {

// The geometry this stage is written for; anything else is refused by name.
inline constexpr int32_t kQkHeads = 16;
inline constexpr int32_t kValueHeads = 48;
inline constexpr int32_t kHeadDim = 128;
inline constexpr int32_t kConvKernel = 4;
inline constexpr int32_t kKeyWidth = kQkHeads * kHeadDim;        // 2048
inline constexpr int32_t kValueWidth = kValueHeads * kHeadDim;   // 6144
inline constexpr int32_t kConvChannels = 2 * kKeyWidth + kValueWidth;  // 10240
inline constexpr int32_t kConvStateTaps = kConvKernel - 1;       // 3
// The vendored snapshot forms take at most this many lanes at one token each.
inline constexpr int32_t kMaxLanes = 8;

// One GDN layer's lane states, every slot (the seq pool's gdn_pool layer, or a test's buffers).
struct State {
  void *conv = nullptr;       // BF16 [slots][kConvStateTaps][kConvChannels], oldest tap first
  float *recurrent = nullptr; // FP32 [slots][kValueHeads][kHeadDim][kHeadDim]
  int32_t slots = 0;
};

// The geometry check fn_gdn_layer makes; null, or what it refuses.
const char *check_geometry(const Geometry &g);

// y = the layer core of x for the batch, advancing the lanes' states at batch.slots. x, y: BF16
// [rows][hidden]. 0, or -1 with fn_set_error.
int32_t run(const Geometry &g, const State &state, const GdnWeights &w, const Batch &batch,
            const void *x, void *y, ninfer::DeviceArena &scratch, cudaStream_t stream);

}  // namespace ignis::flash_next::gdn
