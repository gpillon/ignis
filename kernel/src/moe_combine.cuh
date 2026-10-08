// ignis kernel leaf: the MoE combine's gate and value -- OURS (see kernel/include/ignis_moe.h).
// Leaf-internal: shared by ignis_moe_combine's kernel (moe_shared.cu) and the hyper-connection mix
// that folds the combine into its first launch (flash_next/hc.cu, GitHub #306 step 2), so the two
// compute the same bits from the same operands.
#ifndef IGNIS_MOE_COMBINE_CUH
#define IGNIS_MOE_COMBINE_CUH

#include "moe_common.cuh"

#include <cuda_bf16.h>

namespace ignis_moe {

// The CTA the gate's reduction is written for: its order depends on the thread count.
constexpr int kCombineThreads = 256;

// sigmoid(x . w_gate) of one token's row by the whole CTA (kCombineThreads threads): each thread's
// strided fmaf chain over the kHidden inputs, the warp butterfly, the warps' partials summed in
// order by thread 0, broadcast through `slot`. `partial` holds kCombineThreads / 32 floats.
__device__ __forceinline__ float combine_gate(const __nv_bfloat16 *__restrict__ x,
                                              const __nv_bfloat16 *__restrict__ w_gate, float *partial,
                                              float *slot) {
  float dot = 0.0f;
  for (int k = threadIdx.x; k < kHidden; k += kCombineThreads) {
    dot = fmaf(__bfloat162float(x[k]), __bfloat162float(w_gate[k]), dot);
  }
  dot = warp_sum(dot);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = dot;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kCombineThreads / 32; ++w) s += partial[w];
    *slot = 1.0f / (1.0f + expf(-s));
  }
  __syncthreads();
  return *slot;
}

// One combined output: the shared expert's fp32 value gated, plus the routed experts' fixed-point
// sum, rounded once to BF16.
__device__ __forceinline__ __nv_bfloat16 combine_value(float gate, float shared, long long routed) {
  return __float2bfloat16_rn(fmaf(gate, shared, from_fixed(routed)));
}

}  // namespace ignis_moe

#endif  // IGNIS_MOE_COMBINE_CUH
