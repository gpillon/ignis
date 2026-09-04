/* ignis kernel leaf - Ticket 28 (kernel-abi 06, GitHub #28): GDN causal conv.
 *
 * Provenance: faithful adaptation of the reference GDN projected-input
 * causal convolution
 *   F:/ai/q38/ninfer/src/ops/gdn_input_proj/gdn_projected_conv.cu
 *   (gdn_projected_conv_kernel: one thread per channel; the rolling 3-tap
 *   state s0,s1,s2 + the current tap p; the 4-tap fma chain
 *   w0*s0 + w1*s1 + w2*s2 + w3*p; the SiLU epilogue in
 *   ops/common/math.cuh: `x / (1 + exp(-x))`). The reference kernel is a
 *   per-(channel, batch) machine driven by the engine's Tensor dispatch
 *   (separate query/key/value output tensors, a per-batch valid-column
 *   count, and a slot-indexed state pool). That dispatch layer is C++
 *   state, which the flat C ABI of ADR 0001 forbids across the boundary,
 *   so it is not ported; the per-channel rolling conv math (the fma
 *   chain, the SiLU, the 3-tap state shift) is ported 1:1 to the flat
 *   C ABI geometry (the single-sequence [tokens][channels] projection,
 *   the per-channel state [channels][3]).
 *
 * Contract (see kernel/include/ignis_kernel.h, ticket 28):
 *   projected   : bf16 [tokens][channels] (the projected q/k/v rows, the
 *                 GEMM output — the `z` rows are NOT part of `channels`,
 *                 they bypass the conv entirely).
 *   conv_weight : bf16 [4][channels] (the 4 taps w0..w3, tap-major — the
 *                 artifact's `gdn/convolution` tensor {4, channels}).
 *   state_in    : bf16 [channels][3] (the rolling 3-tap state s0,s1,s2
 *                 per channel, channel-major).
 *   state_out   : bf16 [channels][3] (the updated rolling state after the
 *                 chunk — the last 3 consumed taps; `state_in` may alias
 *                 `state_out` (all state reads happen before any write)).
 *   out         : bf16 [tokens][channels] (the conv'd + SiLU'd q/k/v).
 *
 * One thread per channel (the spec's "one thread per channel"); the fp32
 * fma chain + SiLU + bf16 rounding is ported 1:1 from the reference (the
 * FMA baseline is the v1 starting point, ADR 0005 — the bf16x2 /
 * tensor-core variants are the later performance-gate material, ADR 0007).
 *
 * Device-only header. Instantiated by gdn_conv_rope_surface.cu.
 */
#ifndef IGNIS_GDN_CAUSAL_CONV_CUH
#define IGNIS_GDN_CAUSAL_CONV_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ignis {

// SiLU (the reference's `ops/common/math.cuh`: `x / (1 + exp(-x))`), the
// fp32 device form the conv epilogue uses.
__device__ __forceinline__ float gdn_conv_silu(float x) {
  return x / (1.0f + expf(-x));
}

// GDN 4-tap depthwise causal conv + SiLU, one thread per channel. The
// rolling 3-tap state (s0, s1, s2) is loaded from `state_in` (channel-
// major [channels][3]); per token t the current tap p = projected[t][c]
// feeds the fma chain and shifts the state (s0,s1,s2) = (s1,s2,p); the
// conv'd + SiLU'd value is stored to out[t][c] (bf16-rounded). After the
// chunk, `state_out` receives the updated rolling state (the last 3
// consumed taps).
__global__ void gdn_causal_conv_kernel(const __nv_bfloat16* __restrict__ projected,
                                       const __nv_bfloat16* __restrict__ conv_weight,
                                       const __nv_bfloat16* __restrict__ state_in,
                                       __nv_bfloat16* __restrict__ state_out,
                                       __nv_bfloat16* __restrict__ out, int tokens,
                                       int channels) {
  const int c = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (c >= channels) return;

  // The rolling 3-tap state (channel-major [channels][3]) + the 4 per-
  // channel taps (tap-major [4][channels]). All state reads happen before
  // any state write, so `state_in` may alias `state_out`.
  const std::int64_t s_base = static_cast<std::int64_t>(c) * 3;
  float s0 = __bfloat162float(state_in[s_base]);
  float s1 = __bfloat162float(state_in[s_base + 1]);
  float s2 = __bfloat162float(state_in[s_base + 2]);
  const float w0 = __bfloat162float(conv_weight[static_cast<std::int64_t>(0) * channels + c]);
  const float w1 = __bfloat162float(conv_weight[static_cast<std::int64_t>(1) * channels + c]);
  const float w2 = __bfloat162float(conv_weight[static_cast<std::int64_t>(2) * channels + c]);
  const float w3 = __bfloat162float(conv_weight[static_cast<std::int64_t>(3) * channels + c]);

  for (int t = 0; t < tokens; ++t) {
    const std::int64_t col = static_cast<std::int64_t>(t) * channels + c;
    const float p = __bfloat162float(projected[col]);

    // The 4-tap fma chain (the reference's accumulation order 1:1).
    float conv = fmaf(w0, s0, 0.0f);
    conv = fmaf(w1, s1, conv);
    conv = fmaf(w2, s2, conv);
    conv = fmaf(w3, p, conv);
    out[col] = __float2bfloat16_rn(gdn_conv_silu(conv));

    // The 3-tap state shift (the reference's `s0 = s1; s1 = s2; s2 = p;`).
    s0 = s1;
    s1 = s2;
    s2 = p;
  }

  // The updated rolling state after the chunk = the last 3 consumed taps
  // (for tokens >= 3; shorter chunks keep the earlier `state_in` taps).
  state_out[s_base] = __float2bfloat16_rn(s0);
  state_out[s_base + 1] = __float2bfloat16_rn(s1);
  state_out[s_base + 2] = __float2bfloat16_rn(s2);
}

}  // namespace ignis

#endif  // IGNIS_GDN_CAUSAL_CONV_CUH