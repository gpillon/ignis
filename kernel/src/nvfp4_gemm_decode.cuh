/* ignis kernel leaf - Ticket 03: NVFP4 decode GEMM (GEMV path).
 *
 * Provenance: faithful adaptation of the reference NVFP4 decode GEMV kernel
 *   F:/ai/q38/ninfer/src/ops/linear/nvfp4/nvfp4_gemv.cuh
 *   (+ nvfp4_codec.cuh decode helpers, nvfp4_output.cuh epilogue).
 * The reference kernel is a warp-MMA, shared-staged rowsplit machine templated
 * on a per-shape Geometry/Schedule and driven by the engine's Tensor/Weight
 * dispatch. That dispatch layer is C++ state, which the flat C ABI of ADR 0001
 * forbids across the boundary, so it is not ported. The NVFP4 math (E2M1 codes,
 * per-group-16 E4M3 scales, group-scaled dot product) is ported 1:1; the
 * code/scale plane layouts are simplified to plain row-major [m][k/2] and
 * [m][k/16] (the reference's swizzled 512-element tile layouts belong to the
 * stripped dispatch layer); the tensor-core micro-optimization is deferred
 * (parity-gate ticket). This is a faithful, self-contained decode GEMV that
 * compiles standalone in the leaf.
 *
 * Device-only header. Instantiated by decode_surface.cu.
 */
#ifndef IGNIS_NVFP4_GEMM_DECODE_CUH
#define IGNIS_NVFP4_GEMM_DECODE_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis {

// E2M1 (FP4) decode: 1 sign bit (0x8), 3 magnitude bits. The 8 unsigned
// magnitude codes map to {0, 0.5, 1, 1.5, 2, 3, 4, 6} (NVIDIA FP4 E2M1).
// Inlined (no device-visible constant array, which nvcc cannot place in device
// code) to stay self-contained.
__device__ __forceinline__ float decode_nvfp4_e2m1(std::uint8_t code) {
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

// E4M3 (FP8) decode: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7,
// no infinity (max 448), subnormals for exp == 0.
__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t code) {
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

// NVFP4 decode GEMV (single-token decode): out[m] = bias[m] + sum_k x[k]*W[m,k].
//   x       : bf16 activation vector, length k.
//   wt_codes: E2M1 codes, 2 packed per byte, [m][k/2].
//   wt_scales: E4M3 per-group-16 scales, [m][k/16].
//   bias    : bf16, length m (nullable).
//   out     : bf16, length m.
//
// One block (256 threads) per output row; threads grid-stride over the k
// elements, each thread accumulates a partial group-scaled dot, then a block
// reduce produces the row total.
__global__ void nvfp4_gemm_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                         const std::uint8_t* __restrict__ wt_codes,
                                         const std::uint8_t* __restrict__ wt_scales,
                                         const __nv_bfloat16* __restrict__ bias,
                                         __nv_bfloat16* __restrict__ out, int m, int k) {
  const int row      = static_cast<int>(blockIdx.x);
  if (row >= m) return;
  const int tid      = static_cast<int>(threadIdx.x);
  const int threads  = static_cast<int>(blockDim.x);
  const int code_row = row * (k / 2);          // [m][k/2] code bytes
  const int scale_row = row * (k / 16);        // [m][k/16] scale bytes
  const __nv_bfloat16* xrow = x;               // single decode token: length k
  float acc = 0.0f;
  for (int i = tid; i < k; i += threads) {
    const int code_byte = code_row + (i >> 1);
    const int bit       = (i & 1) * 4;
    const std::uint8_t code = (wt_codes[code_byte] >> bit) & 0xF;
    const int group        = i / 16;
    const float w          = decode_nvfp4_e2m1(code) * decode_nvfp4_e4m3(wt_scales[scale_row + group]);
    acc += w * __bfloat162float(xrow[i]);
  }
  // Block reduce over threads.
  __shared__ float warp_red[32];
  const int warp   = tid >> 5;
  const int lane   = tid & 31;
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off, 32);
  }
  if (lane == 0) warp_red[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    const int nwarps = (threads + 31) / 32;
    for (int w = 0; w < nwarps; ++w) total += warp_red[w];
    total += (bias != nullptr) ? __bfloat162float(bias[row]) : 0.0f;
    out[row] = __float2bfloat16_rn(total);
  }
}

}  // namespace ignis

#endif  // IGNIS_NVFP4_GEMM_DECODE_CUH