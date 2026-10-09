// ignis kernel leaf: the MoE router's selection of one token by one warp -- OURS (see
// kernel/include/ignis_moe.h). Leaf-internal: shared by the router's select launch
// (moe_router.cu) and residency's routed demand step (residency.cu, GitHub #306 step 5), so the
// two make the same selection from the same logits, bit for bit.
//
// The 512 logits rounded to BF16 (the transformers module's F.linear output dtype), each packed
// with its expert into one 32-bit key ordered by (value descending, expert ascending), ten rounds
// of a one-instruction warp max over the keys, and a softmax over the ten in fp32 rounded to BF16
// as the module returns it. A NaN logit ranks as -inf, so a token whose input is not finite still
// gets ten distinct ids in range (0..9 when every logit is NaN) and NaN weights.
#ifndef IGNIS_MOE_ROUTER_SELECT_CUH
#define IGNIS_MOE_ROUTER_SELECT_CUH

#include "ignis_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis_moe_select {

constexpr int kExperts = IGNIS_MOE_EXPERTS;
constexpr int kTopK = IGNIS_MOE_TOP_K;

// The ordering key of a logit: its BF16 value (fp32 rounded to nearest even, as the
// checkpoint's router returns it; NaN as -inf; -0 as +0, which compares equal) mapped
// monotonically onto an unsigned 16-bit number, then 511 - expert -- so a larger key is a larger
// value or, on equal values, the lower expert. Every key is above 0, the mark of a taken one.
__device__ __forceinline__ uint32_t pick_key(float logit, int expert) {
  float v = isnan(logit) ? -INFINITY : logit;
  if (v == 0.0f) v = 0.0f;
  uint32_t b = __bfloat16_as_ushort(__float2bfloat16_rn(v));
  b = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
  return (b << 16) | static_cast<uint32_t>(kExperts - 1 - expert);
}

__device__ __forceinline__ float key_value(uint32_t key) {
  const uint32_t b = key >> 16;
  const uint16_t bits = static_cast<uint16_t>((b & 0x8000u) ? (b & 0x7FFFu) : (~b & 0xFFFFu));
  return __bfloat162float(__ushort_as_bfloat16(bits));
}

// Token t's selection from its fp32 logits [kExperts], by the whole calling warp (`lane` its lane):
// ids[t][0..9] and weights[t][0..9].
__device__ __forceinline__ void select_token(const float *__restrict__ logits, int t, int lane,
                                             int32_t *__restrict__ ids, float *__restrict__ weights) {
  constexpr int kPerLane = kExperts / 32;
  uint32_t keys[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    keys[i] = pick_key(logits[static_cast<size_t>(t) * kExperts + i * 32 + lane], i * 32 + lane);
  }
  float chosen[kTopK];
#pragma unroll
  for (int r = 0; r < kTopK; ++r) {
    uint32_t best = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) best = max(best, keys[i]);
    const uint32_t top = __reduce_max_sync(0xFFFFFFFFu, best);
    // Keys are distinct (they carry the expert), so only the owner's matches.
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    chosen[r] = key_value(top);
    if (lane == 0) ids[static_cast<size_t>(t) * kTopK + r] = kExperts - 1 - static_cast<int>(top & 0xFFFFu);
  }
  if (lane == 0) {
    float e[kTopK];
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < kTopK; ++r) {
      e[r] = expf(chosen[r] - chosen[0]);
      sum += e[r];
    }
#pragma unroll
    for (int r = 0; r < kTopK; ++r) {
      weights[static_cast<size_t>(t) * kTopK + r] = __bfloat162float(__float2bfloat16_rn(e[r] / sum));
    }
  }
}

}  // namespace ignis_moe_select

#endif  // IGNIS_MOE_ROUTER_SELECT_CUH
