// ignis kernel leaf: Flash-Next's MoE router -- OURS (kernel/include/ignis_moe.h).
//
// Two launches, both graph-capturable and allocation-free (ignis_moe_router_logits is the first
// alone; residency's routed demand step runs the second's per-token selection itself):
//
//   logits   grid (token tiles of 16, expert groups). A CTA stages its tokens' BF16 rows in
//            shared memory; each warp takes whole experts, each lane a fixed set of 80 of the
//            2560 inputs (ten 16-byte chunks), accumulating BF16 x BF16 products (exact in
//            fp32) in a fixed order, then a fixed butterfly over the lanes. A logit's value
//            therefore depends only on its token and expert, never on T or the grid: the
//            decode and prefill shapes produce the same bits for the same token.
//   select   one warp per token: the 512 logits rounded to BF16 (the transformers module's
//            F.linear output dtype, measured by kernel/tests/fixtures/flash_next/record.py),
//            each packed with its expert into one 32-bit key ordered by (value descending,
//            expert ascending), ten rounds of a one-instruction warp max over the keys, and a
//            softmax over the ten in fp32 rounded to BF16 as the module returns it. A NaN logit
//            ranks as -inf, so a token whose input is not finite still gets ten distinct ids in
//            range (0..9 when every logit is NaN) and NaN weights, on which the expert ops trap;
//            nothing downstream indexes with a garbage id. It is a programmatic dependent launch:
//            its CTAs are scheduled while the logits run and wait for them on the device.
//
// At decode the weight (2.6 MB) is the whole cost; groups of 4 experts give 128 CTAs so the read
// spreads over the card, and each warp's weight loads are issued before the tokens are staged.
// Wide calls use groups of 32 so each token tile is staged fewer times.

#include "moe_common.cuh"
#include "moe_router_select.cuh"

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
  // The select launch may be scheduled now; it waits for this grid on the device.
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int e_first = blockIdx.y * experts_per_cta + warp;
  const int e_end = (blockIdx.y + 1) * experts_per_cta;
  // The warp's first expert row is in flight while the tokens are staged.
  uint4 wr[kChunks];
  auto load_row = [&](int expert) {
    const uint4 *wg = reinterpret_cast<const uint4 *>(w + static_cast<size_t>(expert) * kHidden);
#pragma unroll
    for (int i = 0; i < kChunks; ++i) wr[i] = __ldg(wg + i * 32 + lane);
  };
  if (e_first < e_end) load_row(e_first);

  const int t0 = blockIdx.x * kTokenTile;
  const int rows = min(kTokenTile, tokens - t0);
  const uint4 *xg = reinterpret_cast<const uint4 *>(x + static_cast<size_t>(t0) * kHidden);
#pragma unroll 4
  for (int i = threadIdx.x; i < rows * (kHidden / 8); i += blockDim.x) xs[i] = __ldg(xg + i);
  __syncthreads();

  for (int e = e_first; e < e_end; e += warps) {
    if (e != e_first) load_row(e);
    for (int t = 0; t < rows; ++t) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < kChunks; ++i) fma_chunk(acc, xs[t * (kHidden / 8) + i * 32 + lane], wr[i]);
      acc = warp_sum(acc);
      if (lane == 0) logits[static_cast<size_t>(t0 + t) * kExperts + e] = acc;
    }
  }
}

// One warp per token: moe_router_select.cuh's select_token, which residency's routed demand step
// (GitHub #306) runs too.
__global__ void router_select_kernel(const float *__restrict__ logits, int tokens,
                                     int32_t *__restrict__ ids, float *__restrict__ weights) {
  // Launched as a programmatic dependent of the logits kernel: wait for its results here.
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (t >= tokens) return;
  ignis_moe_select::select_token(logits, t, lane, ids, weights);
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

namespace {

// The logits launch, after the checks both entry points make.
int32_t router_logits(const char *op, const void *x, uint32_t tokens, const void *w_router, float *logits,
                      cudaStream_t s) {
  if (x == nullptr || w_router == nullptr) return fail(std::string(op) + ": NULL pointer");
  if (logits == nullptr) return fail(std::string(op) + ": the fp32 logits buffer [tokens][512] is required");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be at least 1");
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w_router)) & 15) {
    return fail(std::string(op) + ": x and w_router must be 16-byte aligned");
  }
  const int experts_per_cta = tokens <= kTokenTile ? 4 : 32;
  const int threads = experts_per_cta == 4 ? 128 : 256;
  const dim3 grid((tokens + kTokenTile - 1) / kTokenTile, kExperts / experts_per_cta);
  const int rows = tokens < static_cast<uint32_t>(kTokenTile) ? static_cast<int>(tokens) : kTokenTile;
  const size_t smem = static_cast<size_t>(rows) * kHidden * 2;
  if (require_prepared(op, nullptr) != 0) return -1;
  router_logits_kernel<<<grid, threads, smem, s>>>(static_cast<const __nv_bfloat16 *>(x),
                                                   static_cast<int>(tokens),
                                                   static_cast<const __nv_bfloat16 *>(w_router),
                                                   experts_per_cta, logits);
  return check_launch((std::string(op) + " (logits)").c_str());
}

}  // namespace

extern "C" int32_t ignis_moe_router_logits(const void *x, uint32_t tokens, const void *w_router, float *logits,
                                           void *stream) {
  return router_logits("ignis_moe_router_logits", x, tokens, w_router, logits, static_cast<cudaStream_t>(stream));
}

extern "C" int32_t ignis_moe_router(const void *x, uint32_t tokens, const void *w_router,
                                    int32_t *ids, float *weights, float *logits, void *stream) {
  if (ids == nullptr || weights == nullptr) return fail("ignis_moe_router: NULL pointer");
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (router_logits("ignis_moe_router", x, tokens, w_router, logits, s) != 0) return -1;
  const int warps = 8;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((tokens + warps - 1) / warps);
  cfg.blockDim = dim3(warps * 32);
  cfg.stream = s;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, router_select_kernel, static_cast<const float *>(logits),
                                             static_cast<int>(tokens), ids, weights);
  if (err != cudaSuccess) return fail(std::string("ignis_moe_router (select): ") + cudaGetErrorString(err));
  return check_launch("ignis_moe_router (select)");
}
