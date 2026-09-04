// ignis kernel leaf - Ticket 29 (kernel-abi 10, GitHub #29): bf16 GEMM (the
// logits path for the W8-dequantized lm_head).
//
// Provenance: this kernel follows the proven rowsplit FMA pattern of the
// ticket-22 NVFP4 prefill GEMM (nvfp4_gemm_prefill.cuh — the rowsplit
// "rows-of-W x tokens" tiling with fp32 FMA accumulation, ADR 0005's "for
// now" baseline), with the NVFP4 dequant staging replaced by a plain bf16
// weight read. There is no separate reference kernel for this path: the
// reference engine consumes dequantized weights through its own GEMM
// dispatch (C++ state the flat C ABI of ADR 0001 does not carry), so this
// is a new self-contained kernel in the leaf's established style. The
// tensor-core MMA (the later performance-gate material, ADR 0005/0007) is
// deliberately not used: v1 is a correctness-first scalar-FMA baseline.
//
// Device-only header. Instantiated by bf16_gemm_surface.cu.

#ifndef IGNIS_BF16_GEMM_CUH
#define IGNIS_BF16_GEMM_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis {

// Multi-token bf16 GEMM (the logits path for the W8-dequantized lm_head):
//   out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]
//   act       : bf16 [tokens][k] (row-major, k contiguous per token).
//   wt        : bf16 [m][k] (row-major, k contiguous per m-row — the
//               W8-dequantized lm_head, produced by the A1 artifact dequant).
//   bias      : bf16 [m] (nullable).
//   out       : bf16 [tokens][m] (row-major, m contiguous per token).
//
// Rowsplit tiling (the proven rowsplit FMA baseline, ADR 0005): the grid is
// split along (m-rows, tokens); each block (kTileT x kTileM threads) computes
// a [kTileT tokens] x [kTileM m-rows] output tile and accumulates over k in
// fp32, chunking the reduction through shared memory (kTileK k-elements at a
// time), then rounds to bf16. No tensor cores, no cuBLASLt (ADR 0001); the
// tensor-core MMA is the later performance-gate material (ADR 0007). `tokens
// == 1` is the GEMV special case (the decode logits path); `tokens > 1` is
// the batched-prefill logits path (B1). k needs no alignment constraint
// (plain bf16 planes — no NVFP4 group scales); any k >= 1 works, with the
// tail chunk staged in-bounds (clamped reads, bounds-guarded store).
__global__ void bf16_gemm_kernel(const __nv_bfloat16* __restrict__ act,
                                 const __nv_bfloat16* __restrict__ wt,
                                 const __nv_bfloat16* __restrict__ bias,
                                 __nv_bfloat16* __restrict__ out,
                                 std::int32_t tokens, std::int32_t m,
                                 std::int32_t k) {
  // Block tile: one thread per output element (kTileM m-rows x kTileT tokens).
  constexpr int kTileM = 16;  // m-rows per block
  constexpr int kTileT = 16;  // tokens per block
  constexpr int kTileK = 32;  // k-elements per shared-memory chunk
  constexpr int kThreads = kTileM * kTileT;  // 256

  const int m0 = static_cast<int>(blockIdx.x) * kTileM;  // first m-row
  const int t0 = static_cast<int>(blockIdx.y) * kTileT;  // first token
  const int tid =
      static_cast<int>(threadIdx.x) + static_cast<int>(threadIdx.y) * kTileM;

  // Shared tiles, staged as fp32 so the FMA accumulator stays fp32 (the bf16
  // values are converted on stage-in, matching the ticket-22 prefill GEMM).
  __shared__ float sA[kTileT][kTileK];  // activation tile
  __shared__ float sB[kTileM][kTileK];  // weight tile

  float acc = 0.0f;
  for (std::int32_t kk = 0; kk < k; kk += kTileK) {
    const std::int32_t kEnd = (kk + kTileK < k) ? kTileK : (k - kk);
    // Stage the activation tile: sA[tt][jj] = act[t0 + tt][kk + jj]. Out-of-
    // range token rows (t0 + tt >= tokens) are clamped to the last valid row
    // so the staging stays in-bounds; their outputs are never written (the
    // final store is bounds-guarded), so the clamped values are discarded.
    for (std::int32_t i = tid; i < kTileT * kEnd; i += kThreads) {
      const std::int32_t tt = i / kEnd;
      const std::int32_t jj = i - tt * kEnd;
      const std::int32_t trow = (t0 + tt < tokens) ? (t0 + tt) : (tokens - 1);
      sA[tt][jj] = __bfloat162float(act[static_cast<std::int64_t>(trow) * k + (kk + jj)]);
    }
    // Stage the weight tile: sB[mm][jj] = W[m0 + mm][kk + jj] (bf16 -> fp32).
    // Out-of-range m-rows (m0 + mm >= m) are clamped to the last valid row so
    // the staging stays in-bounds; their outputs are never written.
    for (std::int32_t i = tid; i < kTileM * kEnd; i += kThreads) {
      const std::int32_t mm = i / kEnd;
      const std::int32_t jj = i - mm * kEnd;
      const std::int32_t grow = (m0 + mm < m) ? (m0 + mm) : (m - 1);  // clamped m-row
      sB[mm][jj] = __bfloat162float(wt[static_cast<std::int64_t>(grow) * k + (kk + jj)]);
    }
    __syncthreads();
    // This thread's output element: out[t0 + ty][m0 + tx] += sum_j sA[ty][j]
    // * sB[tx][j] over this k-chunk.
    float partial = 0.0f;
    for (std::int32_t j = 0; j < kEnd; ++j) {
      partial += sA[static_cast<int>(threadIdx.y)][j] * sB[static_cast<int>(threadIdx.x)][j];
    }
    acc += partial;
    __syncthreads();  // all threads done reading before the next chunk overwrites
  }

  const int t = t0 + static_cast<int>(threadIdx.y);
  const int mr = m0 + static_cast<int>(threadIdx.x);
  if (t < tokens && mr < m) {
    float total = acc;
    if (bias != nullptr) {
      total += __bfloat162float(bias[mr]);
    }
    out[static_cast<std::int64_t>(t) * m + mr] = __float2bfloat16_rn(total);
  }
}

}  // namespace ignis

#endif  // IGNIS_BF16_GEMM_CUH