/* ignis kernel leaf - Ticket 22 (kernel-abi 05, GitHub #22): multi-token NVFP4
 * GEMM (the prefill / FFN-projection path).
 *
 * Provenance: faithful adaptation of the reference bf16-activation small-tokens
 * NVFP4 GEMM (F:/ai/q38/ninfer/src/ops/linear/nvfp4/nvfp4_small_t.cuh — the
 * rowsplit "rows-of-W × tokens" tiling with fp32 FMA accumulation) and its
 * large-tokens rowsplit grid structure (nvfp4_w4a4_mma.cuh: grid.x over m-rows,
 * grid.y over token tiles). The reference kernels are driven by the engine's
 * Geometry/Schedule/RowPolicy/Epilogue dispatch and use Programmatic Dependent
 * Launch (PDL); the large-tokens path is a tensor-core W4A4 MMA that requires
 * FP4-quantized *activations*. That dispatch layer is C++ state, which the flat
 * C ABI of ADR 0001 forbids across the boundary, so it is not ported; the PDL
 * syncs are stripped in favor of plain <<<>>> launches; and the tensor-core W4A4
 * MMA (which needs FP4 activations, not the bf16 activations of this ABI) is
 * deferred to the performance gate (ADR 0007).
 *
 * What IS ported 1:1 is the math: bf16 activations x dequantized NVFP4 weights
 * (E2M1 codes x E4M3 per-group-16 scales), fp32 FMA accumulation, the rowsplit
 * grid (output m-rows x tokens), and the plain row-major code/scale plane
 * layouts ([m][k/2] and [m][k/16]) that the rest of the leaf already uses
 * (ticket 03). The E2M1/E4M3 decode helpers are ported 1:1 (inlined in this
 * header with a `gemm_prefill_` prefix, to keep it self-contained and avoid
 * pulling the ticket-03 GEMV kernel into this translation unit), so a
 * `tokens == 1` call produces the same dequantized weights as the single-token
 * GEMV. This is a faithful, self-contained, non-tensor-core multi-token GEMM
 * that compiles standalone in the leaf.
 *
 * Device-only header. Instantiated by nvfp4_gemm_prefill_surface.cu.
 */
#ifndef IGNIS_NVFP4_GEMM_PREFILL_CUH
#define IGNIS_NVFP4_GEMM_PREFILL_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis {

// E2M1 (FP4) decode: 1 sign bit (0x8), 3 magnitude bits. The 8 unsigned
// magnitude codes map to {0, 0.5, 1, 1.5, 2, 3, 4, 6} (NVIDIA FP4 E2M1).
// Ported 1:1 from nvfp4_gemm_decode.cuh (the ticket-03 GEMV) so the dequant is
// identical to the single-token GEMV. Inlined (no device-visible constant
// array, which nvcc cannot place in device code) and given a `gemm_prefill_`
// prefix to keep this header self-contained (it must not pull the GEMV kernel
// into this translation unit).
__device__ __forceinline__ float gemm_prefill_decode_e2m1(std::uint8_t code) {
  const int mag3 = code & 0x7;
  float mag;
  switch (mag3) {
    case 0: mag = 0.0f; break;
    case 1: mag = 0.5f; break;
    case 2: mag = 1.0f; break;
    case 3: mag = 1.5f; break;
    case 4: mag = 2.0f; break;
    case 5: mag = 3.0f; break;
    case 6: mag = 4.0f; break;
    default: mag = 6.0f; break;
  }
  return (code & 0x8) ? -mag : mag;
}

// E4M3 (FP8) decode: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7, no
// infinity (max 448), subnormals for exp == 0. Ported 1:1 from
// nvfp4_gemm_decode.cuh (the ticket-03 GEMV).
__device__ __forceinline__ float gemm_prefill_decode_e4m3(std::uint8_t code) {
  const int sign = (code & 0x80) ? -1 : 1;
  const int exp  = (code >> 3) & 0xF;
  const int man  = code & 0x7;
  float mag;
  if (exp == 0) {
    // Subnormal: (m/8) * 2^(1-bias) with bias 7 -> (m/8) * 2^-6.
    mag = (static_cast<float>(man) / 8.0f) * 0.015625f;
  } else {
    mag = (1.0f + static_cast<float>(man) / 8.0f) * ldexpf(1.0f, exp - 7);
  }
  return sign * mag;
}

// Multi-token NVFP4 GEMM (prefill / FFN-projection path):
//   out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]
// where W[m][k] = e2m1(code[m][k]) * e4m3(scale[m][k/16]) is the dequantized
// NVFP4 weight.
//   act       : bf16 [tokens][k] (row-major, k contiguous per token).
//   wt_codes  : E2M1 codes, 2 packed per byte, [m][k/2] (row-major).
//   wt_scales : E4M3 per-group-16 scales, [m][k/16] (row-major).
//   bias      : bf16 [m] (nullable).
//   out       : bf16 [tokens][m] (row-major, m contiguous per token).
//
// Rowsplit tiling: the grid is split along (m-rows, tokens); each block
// (kTileM x kTileT threads) computes a [kTileT tokens] x [kTileM m-rows] output
// tile and accumulates over k in fp32, chunking the reduction through shared
// memory (kTileK k-elements at a time), then rounds to bf16. No tensor cores,
// no cuBLASLt (ADR 0001); the tensor-core W4A4 MMA is the later
// performance-gate material (ADR 0007). k must be a multiple of 16.
__global__ void nvfp4_gemm_prefill_kernel(const __nv_bfloat16* __restrict__ act,
                                          const std::uint8_t* __restrict__ wt_codes,
                                          const std::uint8_t* __restrict__ wt_scales,
                                          const __nv_bfloat16* __restrict__ bias,
                                          __nv_bfloat16* __restrict__ out,
                                          std::int32_t tokens, std::int32_t m,
                                          std::int32_t k) {
  // Block tile: one thread per output element (kTileM m-rows x kTileT tokens).
  constexpr int kTileM = 16;  // m-rows per block
  constexpr int kTileT = 16;  // tokens per block
  constexpr int kTileK = 32;  // k-elements per shared-memory chunk

  const int m0 = static_cast<int>(blockIdx.x) * kTileM;  // first m-row
  const int t0 = static_cast<int>(blockIdx.y) * kTileT;  // first token
  const int tid = static_cast<int>(threadIdx.x) + static_cast<int>(threadIdx.y) * kTileM;
  constexpr int kThreads = kTileM * kTileT;  // 256

  // Shared tiles, staged as fp32 so the FMA accumulator stays fp32 (the bf16
  // values are converted on stage-in, matching the ticket-03 GEMV's fp32 math).
  __shared__ float sA[kTileT][kTileK];  // activation tile
  __shared__ float sB[kTileM][kTileK];  // dequantized weight tile

  float acc = 0.0f;
  for (std::int32_t kk = 0; kk < k; kk += kTileK) {
    const std::int32_t kEnd = (kk + kTileK < k) ? kTileK : (k - kk);
    // Stage the activation tile: sA[tt][jj] = act[t0 + tt][kk + jj]. Out-of-
    //range token rows (t0 + tt >= tokens) are clamped to the last valid row so
    //the staging stays in-bounds; their outputs are never written (the final
    //store is bounds-guarded), so the clamped values are discarded.
    for (std::int32_t i = tid; i < kTileT * kEnd; i += kThreads) {
      const std::int32_t tt = i / kEnd;
      const std::int32_t jj = i - tt * kEnd;
      const std::int32_t trow = (t0 + tt < tokens) ? (t0 + tt) : (tokens - 1);
      sA[tt][jj] = __bfloat162float(act[static_cast<std::int64_t>(trow) * k + (kk + jj)]);
    }
    // Stage the dequantized weight tile: sB[mm][jj] = e2m1(code) * e4m3(scale)
    // for W[m0 + mm][kk + jj] (the NVFP4 dequant, 1:1 with the ticket-03 GEMV).
    // Out-of-range m-rows (m0 + mm >= m) are clamped to the last valid row so
    //the staging stays in-bounds; their outputs are never written.
    for (std::int32_t i = tid; i < kTileM * kEnd; i += kThreads) {
      const std::int32_t mm = i / kEnd;
      const std::int32_t jj = i - mm * kEnd;
      const std::int32_t grow = (m0 + mm < m) ? (m0 + mm) : (m - 1);  // clamped m-row
      const std::int32_t gk = kk + jj;   // global k-element
      const std::uint8_t codeByte =
          wt_codes[static_cast<std::int64_t>(grow) * (k / 2) + (gk >> 1)];
      const std::uint8_t code = (codeByte >> ((gk & 1) * 4)) & 0xF;
      const std::uint8_t scale =
          wt_scales[static_cast<std::int64_t>(grow) * (k / 16) + (gk / 16)];
      sB[mm][jj] = gemm_prefill_decode_e2m1(code) * gemm_prefill_decode_e4m3(scale);
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

#endif  // IGNIS_NVFP4_GEMM_PREFILL_CUH