/* ignis kernel leaf - Ticket 28 (kernel-abi 06, GitHub #28): GQA RoPE.
 *
 * Provenance: faithful adaptation of the reference GQA split-half NeoX RoPE
 * rotation
 *   F:/ai/q38/ninfer/src/ops/kernel/rope.cuh
 *   (rope.cuh: apply_rope_head — the split-half pair rotation
 *   out[p] = a*cos - b*sin, out[p + R/2] = b*cos + a*sin, with a = x[p],
 *   b = x[p + R/2]; the per-block cos/sin shared cache, the reference's
 *   fixed-sincos pattern; the legacy unscaled route — attention_factor
 *   1.0 — is the fp32 product + sincosf, bit-stable engine history) and
 *   its inv_freq table
 *   F:/ai/q38/ninfer/src/ops/wrapper/rope.cpp
 *   (rope_linear_frequencies: inv_freq[p] = θ^(-2p/rotary_dim)). The
 *   reference fuses rope(rmsnorm(x)) (ops/launcher/qk_norm_rope.cu); the
 *   v1 does the ignis_rmsnorm (kernel-abi 02) + this RoPE as two steps
 *   (the fused kernel is the later performance item, ADR 0005). The
 *   reference's Tensor dispatch (the RopeKernelMode / QHeads / KHeads
 *   template dispatch, the PDL syncs, the engine's Tensor geometry) is
 *   C++ state, which the flat C ABI of ADR 0001 forbids across the
 *   boundary, so it is not ported; the pair-rotation math (the split-
 *   half NeoX rotation, the fp32 unscaled sincos route) is ported 1:1.
 *
 * Contract (see kernel/include/ignis_kernel.h, ticket 28):
 *   q         : bf16 [batch][seq][num_q_heads][head_dim] (in-place).
 *   k         : bf16 [batch][seq][num_kv_heads][head_dim] (in-place).
 *   inv_freq  : fp32 [rotary_dim/2] (the per-pair frequencies, the
 *               reference's `rope_linear_frequencies` table — θ = 1e7,
 *               rotary_dim = 64 in v1; computed once at construction,
 *               host-side, a deterministic table).
 *   rotary_dim: the R (rotary width, the Qwen 3.8-27B GQA geometry:
 *               R = 64 of head_dim = 256 — 32 pairs). For each pair
 *               p in [0, R/2), with a = x[p], b = x[p + R/2]:
 *                 out[p]           = a*cos - b*sin
 *                 out[p + R/2]     = b*cos + a*sin
 *               where (cos, sin) = sincosf(pos * inv_freq[p]) (the fp32
 *               unscaled route, the reference's attention_factor 1.0
 *               bit-stable path — v1 is unscaled, factor 1.0). The dims
 *               [R, head_dim) are never written (the un-rotated dims).
 *
 * One warp per (batch, seq, head) row; each lane strided-covers the
 * rotary pairs (lane, lane+32, ...). A per-block shared cos/sin cache
 * (the reference's per-token shared-table pattern; the single scalar
 * `pos` makes it a per-block cache).
 *
 * Device-only header. Instantiated by gdn_conv_rope_surface.cu.
 */
#ifndef IGNIS_ROPE_QK_CUH
#define IGNIS_ROPE_QK_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ignis {

// The GQA split-half NeoX RoPE (in-place on q/k). For each (batch, seq,
// head) row and each rotary pair p in [0, rotary_dim/2), with a = x[p],
// b = x[p + rotary_dim/2], (cos, sin) = sincosf(pos * inv_freq[p]):
//   x[p]               = a*cos - b*sin
//   x[p + rotary_dim/2] = b*cos + a*sin
// The un-rotated dims [rotary_dim, head_dim) are left unchanged. One
// warp per row; each lane strided-covers the pairs (lane, lane+32, ...).
// The per-block shared cos/sin cache is the reference's fixed-sincos
// pattern (a single scalar `pos` makes it a per-block cache).
__global__ void rope_qk_kernel(__nv_bfloat16* __restrict__ q,
                               __nv_bfloat16* __restrict__ k,
                               const float* __restrict__ inv_freq, int batch, int seq,
                               int num_q_heads, int num_kv_heads, int head_dim,
                               int rotary_dim, int pos) {
  constexpr int kMaxPairs = 128;  // the head_dim/2 bound (the rotary_dim/2 cap)
  __shared__ float cos_cache[kMaxPairs];
  __shared__ float sin_cache[kMaxPairs];

  const int pairs = rotary_dim / 2;
  for (int p = static_cast<int>(threadIdx.x); p < pairs; p += static_cast<int>(blockDim.x)) {
    // The fp32 product (the reference's unscaled route, bit-stable).
    const float angle = static_cast<float>(pos) * inv_freq[p];
    sincosf(angle, &sin_cache[p], &cos_cache[p]);
  }
  __syncthreads();

  const int total_heads = num_q_heads + num_kv_heads;
  const int rows = batch * seq * total_heads;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int rows_per_block = static_cast<int>(blockDim.x) >> 5;
  const int row_stride = static_cast<int>(gridDim.x) * rows_per_block;

  for (int row = static_cast<int>(blockIdx.x) * rows_per_block + warp; row < rows;
       row += row_stride) {
    // Decompose the flat row into (batch, seq, head). The q rows come
    // first (heads 0..num_q_heads), then the k rows (heads num_q_heads..
    // total_heads) — the GQA layout.
    const int h = row % total_heads;
    const int s_idx = (row / total_heads) % seq;
    const int b_idx = row / (seq * total_heads);

    // The row's head vector (the q row if h < num_q_heads, else the k row).
    __nv_bfloat16* base;
    if (h < num_q_heads) {
      base = q + (static_cast<std::int64_t>(b_idx * seq + s_idx) * num_q_heads + h) * head_dim;
    } else {
      base = k + (static_cast<std::int64_t>(b_idx * seq + s_idx) * num_kv_heads +
                  (h - num_q_heads)) *
                    head_dim;
    }

    // Each lane strided-covers the rotary pairs (lane, lane+32, ...) —
    // the split-half NeoX rotation (the reference's apply_rope_head pair
    // math, ported 1:1).
    for (int p = lane; p < pairs; p += 32) {
      const float a = __bfloat162float(base[p]);
      const float b = __bfloat162float(base[p + pairs]);
      const float c = cos_cache[p];
      const float s = sin_cache[p];
      base[p] = __float2bfloat16_rn(a * c - b * s);
      base[p + pairs] = __float2bfloat16_rn(b * c + a * s);
    }
  }
}

}  // namespace ignis

#endif  // IGNIS_ROPE_QK_CUH