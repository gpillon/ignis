/* ignis kernel leaf - Ticket 06 (kernel-abi-02): RMSNorm / LayerNorm.
 *
 * Provenance: faithful adaptation of the reference RMSNorm math
 *   (F:/ai/q38/ninfer/src/ops/kernel/rmsnorm.cuh: the per-vector
 *   sum-of-squares -> rsqrtf(mean + eps) -> scaled-epilogue pipeline) and
 *   its LayerNorm sibling (ops/kernel/layer_norm.cuh: the center-then-
 *   normalize form). The reference kernels are row-geometry machines
 *   (bf16x2-aligned dispatch, PDL-synced, shape-specific variants driven
 *   by the engine's Tensor dispatch). That dispatch layer is C++ state,
 *   which the flat C ABI of ADR 0001 forbids across the boundary, so it
 *   is not ported; the PDL sync/publish calls are likewise stripped in
 *   favor of plain <<<>>> launches. The normalization math -- the sum of
 *   squares, the rsqrtf(mean + eps) reciprocal, the out = value * inv *
 *   weight epilogue -- is ported 1:1 for the ABI's 1-D vector contract:
 *   x / out are bf16 [n]; center (nullable) centers first (the LayerNorm
 *   mode, base[i] = x[i] - center[i]); weight (nullable) is 1.0 where
 *   absent; eps is pre-resolved to a positive value by the surface.
 *
 * Kernel: one block of 1024 threads, grid-stride over n, fp32 internal
 * math, bf16 in/out. The reference's bf16x2 / warp-tiled row variants are
 * the performance-gate material (ADR 0007). Device-only header.
 * Instantiated by norms_sampling_surface.cu.
 */
#ifndef IGNIS_RMSNORM_CUH
#define IGNIS_RMSNORM_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis {

// RMSNorm (center == null) / LayerNorm (center != null, "centered
// first"), one vector of n elements. base[i] = center ? x[i] - center[i]
// : x[i]; inv = rsqrtf(mean(base^2) + eps); out[i] = base[i] * inv *
// weight[i] (weight is 1.0 where the ABI passes a null pointer). One
// block; each thread accumulates its grid-stride slice of the sum of
// squares, a two-level block reduce (warp shuffles + the warp-0
// aggregation, ported from the reference's block_reduce_sum) yields the
// total, and the epilogue re-reads the (recomputed) base values.
__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ weight,
                               const __nv_bfloat16* __restrict__ center,
                               __nv_bfloat16* __restrict__ out, int n,
                               float eps) {
  constexpr int kThreads = 1024;
  constexpr int kWarpSize = 32;
  constexpr int kWarps = kThreads / kWarpSize;

  __shared__ float warp_sums[kWarps];
  __shared__ float total_sums;

  // The warp-0 aggregation below shuffles across a FULL warp (all 32
  // lanes of thread indices [0, 32)), so the second stage needs
  // kWarps == kWarpSize exactly (a full-warp block, kThreads = 1024).
  static_assert(kWarps == kWarpSize,
                "the warp-0 aggregation needs a full-warp block (1024 threads)");

  const int tid = static_cast<int>(threadIdx.x);

  // Phase 1: grid-stride sum of squares of the (optionally centered)
  // vector, fp32 internal math (the reference's per-row sum-of-squares
  // accumulation, ported to the 1-D vector contract).
  float sumsq = 0.0f;
  for (int i = tid; i < n; i += kThreads) {
    const float xv = __bfloat162float(x[i]);
    // center is nullptr in the RMSNorm mode (the guard never traps: a
    // null dereference is avoided by the ternary).
    const float cv = center != nullptr ? __bfloat162float(center[i]) : 0.0f;
    const float base = xv - cv;
    sumsq += base * base;
  }

  // Block reduce of the per-thread partials (port of the reference's
  // block_reduce_sum<1024>: warp shuffles, then the warp-0 aggregation of
  // the per-warp partials, broadcast via the shared slot).
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset);
  }
  if ((tid & (kWarpSize - 1)) == 0) {
    warp_sums[tid / kWarpSize] = sumsq;
  }
  __syncthreads();
  if (tid < kWarps) {
    float v = warp_sums[tid];
    for (int offset = kWarps / 2; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (tid == 0) {
      total_sums = v;
    }
  }
  __syncthreads();
  const float total = total_sums;
  const float mean = total / static_cast<float>(n);
  const float inv = rsqrtf(mean + eps);

  // Phase 2: the reference's plain epilogue (value * inv * weight), with
  // weight = 1.0 where the ABI passes a null pointer.
  for (int i = tid; i < n; i += kThreads) {
    const float xv = __bfloat162float(x[i]);
    const float cv = center != nullptr ? __bfloat162float(center[i]) : 0.0f;
    const float base = xv - cv;
    const float w = weight != nullptr ? __bfloat162float(weight[i]) : 1.0f;
    out[i] = __float2bfloat16(base * inv * w);
  }
}

}  // namespace ignis

#endif  // IGNIS_RMSNORM_CUH