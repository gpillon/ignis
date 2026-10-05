// ignis kernel leaf: Flash-Next's MoE router -- OURS (kernel/include/ignis_moe.h).
//
// Two launches, both graph-capturable and allocation-free:
//
//   logits   grid (token tiles of 16, expert groups). A CTA stages its tokens' BF16 rows in
//            shared memory; each warp takes whole experts, each lane a fixed set of 80 of the
//            2560 inputs (ten 16-byte chunks), accumulating BF16 x BF16 products (exact in
//            fp32) in a fixed order, then a fixed butterfly over the lanes. A logit's value
//            therefore depends only on its token and expert, never on T or the grid: the
//            decode and prefill shapes produce the same bits for the same token.
//   select   one warp per token: the 512 logits rounded to BF16 (the transformers module's
//            F.linear output dtype, measured by kernel/tests/fixtures/flash_next/record.py),
//            ten rounds of a warp arg-max ordered by (value descending, expert ascending), and
//            a softmax over the ten in fp32 rounded to BF16 as the module returns it. A NaN
//            logit ranks as -inf, so a token whose input is not finite still gets ten distinct
//            ids in range (0..9 when every logit is NaN) and NaN weights, on which the expert
//            ops trap; nothing downstream indexes with a garbage id.
//
// At decode the weight (2.6 MB) is the whole cost; groups of 4 experts give 128 CTAs so the read
// spreads over the card. Wide calls use groups of 32 so each token tile is staged fewer times.

#include "moe_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ignis_moe {
namespace {

constexpr int kTokenTile = 16;
constexpr int kChunks = kHidden / 8 / 32;  // 16-byte chunks per lane: 10

__device__ __forceinline__ void fma_chunk(float &acc, const uint4 &x, const uint4 &w) {
  const __nv_bfloat162 *xp = reinterpret_cast<const __nv_bfloat162 *>(&x);
  const __nv_bfloat162 *wp = reinterpret_cast<const __nv_bfloat162 *>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xf = __bfloat1622float2(xp[i]);
    const float2 wf = __bfloat1622float2(wp[i]);
    acc = fmaf(xf.x, wf.x, acc);
    acc = fmaf(xf.y, wf.y, acc);
  }
}

__global__ void router_logits_kernel(const __nv_bfloat16 *__restrict__ x, int tokens,
                                     const __nv_bfloat16 *__restrict__ w, int experts_per_cta,
                                     float *__restrict__ logits) {
  extern __shared__ uint4 xs[];  // [rows][kHidden / 8]
  const int t0 = blockIdx.x * kTokenTile;
  const int rows = min(kTokenTile, tokens - t0);
  const uint4 *xg = reinterpret_cast<const uint4 *>(x + static_cast<size_t>(t0) * kHidden);
  for (int i = threadIdx.x; i < rows * (kHidden / 8); i += blockDim.x) xs[i] = xg[i];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  for (int e = blockIdx.y * experts_per_cta + warp; e < (blockIdx.y + 1) * experts_per_cta; e += warps) {
    const uint4 *wg = reinterpret_cast<const uint4 *>(w + static_cast<size_t>(e) * kHidden);
    uint4 wr[kChunks];
#pragma unroll
    for (int i = 0; i < kChunks; ++i) wr[i] = __ldg(wg + i * 32 + lane);
    for (int t = 0; t < rows; ++t) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < kChunks; ++i) fma_chunk(acc, xs[t * (kHidden / 8) + i * 32 + lane], wr[i]);
      acc = warp_sum(acc);
      if (lane == 0) logits[static_cast<size_t>(t0 + t) * kExperts + e] = acc;
    }
  }
}

struct Pick {
  float value;
  int id;
};

// Larger value first; on equal values the lower expert id.
__device__ __forceinline__ bool better(const Pick &a, const Pick &b) {
  return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__global__ void router_select_kernel(const float *__restrict__ logits, int tokens,
                                     int32_t *__restrict__ ids, float *__restrict__ weights) {
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (t >= tokens) return;
  constexpr int kPerLane = kExperts / 32;
  float v[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    // BF16, as the checkpoint's router returns its logits: fp32 rounded to nearest even.
    const float l = logits[static_cast<size_t>(t) * kExperts + i * 32 + lane];
    v[i] = isnan(l) ? -INFINITY : __bfloat162float(__float2bfloat16_rn(l));
  }
  uint32_t taken = 0;
  float chosen[kTopK];
#pragma unroll
  for (int r = 0; r < kTopK; ++r) {
    Pick best{-INFINITY, 0x7FFFFFFF};
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const Pick p{v[i], i * 32 + lane};
      if (!(taken & (1u << i)) && better(p, best)) best = p;
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
      const Pick o{__shfl_xor_sync(0xFFFFFFFFu, best.value, m), __shfl_xor_sync(0xFFFFFFFFu, best.id, m)};
      if (better(o, best)) best = o;
    }
    // best.id is always a real expert here (512 candidates, ten rounds, no NaN left).
    if (best.id < kExperts && (best.id & 31) == lane) taken |= 1u << (best.id >> 5);
    chosen[r] = best.value;
    if (lane == 0) ids[static_cast<size_t>(t) * kTopK + r] = best.id;
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

}  // namespace

int32_t prepare_router() {
  const cudaError_t err = cudaFuncSetAttribute(router_logits_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               kTokenTile * kHidden * 2);
  if (err != cudaSuccess) return fail(std::string("ignis_moe_prepare (router): ") + cudaGetErrorString(err));
  return 0;
}

}  // namespace ignis_moe

using namespace ignis_moe;

extern "C" int32_t ignis_moe_router(const void *x, uint32_t tokens, const void *w_router,
                                    int32_t *ids, float *weights, float *logits, void *stream) {
  if (x == nullptr || w_router == nullptr || ids == nullptr || weights == nullptr) {
    return fail("ignis_moe_router: NULL pointer");
  }
  if (logits == nullptr) {
    return fail("ignis_moe_router: the fp32 logits buffer [tokens][512] is required");
  }
  if (tokens == 0) return fail("ignis_moe_router: tokens must be at least 1");
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w_router)) & 15) {
    return fail("ignis_moe_router: x and w_router must be 16-byte aligned");
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int experts_per_cta = tokens <= kTokenTile ? 4 : 32;
  const int threads = experts_per_cta == 4 ? 128 : 256;
  const dim3 grid((tokens + kTokenTile - 1) / kTokenTile, kExperts / experts_per_cta);
  const int rows = tokens < static_cast<uint32_t>(kTokenTile) ? static_cast<int>(tokens) : kTokenTile;
  const size_t smem = static_cast<size_t>(rows) * kHidden * 2;
  if (require_prepared("ignis_moe_router", nullptr) != 0) return -1;
  router_logits_kernel<<<grid, threads, smem, s>>>(static_cast<const __nv_bfloat16 *>(x),
                                                   static_cast<int>(tokens),
                                                   static_cast<const __nv_bfloat16 *>(w_router),
                                                   experts_per_cta, logits);
  if (check_launch("ignis_moe_router (logits)") != 0) return -1;
  const int warps = 8;
  router_select_kernel<<<(tokens + warps - 1) / warps, warps * 32, 0, s>>>(
      logits, static_cast<int>(tokens), ids, weights);
  return check_launch("ignis_moe_router (select)");
}
