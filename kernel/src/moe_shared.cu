// ignis kernel leaf: Flash-Next's shared expert and the MoE combine -- OURS
// (kernel/include/ignis_moe.h).
//
// The shared expert is the checkpoint's Qwen4ExpTextMLP at intermediate 640 on the FP8 row-scale
// linear: gate/up with the SwiGLU epilogue (h in BF16, as the linear's activations are), then
// down into fp32. The combine adds it, gated by sigmoid(x . w_gate) over the BF16
// shared_expert_gate, to the routed experts' fixed-point sum, rounds once to BF16, and zeroes the
// accumulator it read for the next call. One CTA per token; the gate's dot product is a fixed
// block reduction, so the result is deterministic.

#include "moe_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ignis_moe {
namespace {

constexpr int kCombineThreads = 256;

__global__ void __launch_bounds__(kCombineThreads) combine_kernel(long long *__restrict__ acc, const float *__restrict__ shared,
                                                                  const __nv_bfloat16 *__restrict__ x,
                                                                  const __nv_bfloat16 *__restrict__ w_gate,
                                                                  __nv_bfloat16 *__restrict__ out) {
  __shared__ float partial[kCombineThreads / 32];
  __shared__ float gate;
  const int t = blockIdx.x;
  const __nv_bfloat16 *xr = x + static_cast<size_t>(t) * kHidden;
  float dot = 0.0f;
  for (int k = threadIdx.x; k < kHidden; k += kCombineThreads) {
    dot = fmaf(__bfloat162float(xr[k]), __bfloat162float(w_gate[k]), dot);
  }
  dot = warp_sum(dot);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = dot;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kCombineThreads / 32; ++w) s += partial[w];
    gate = 1.0f / (1.0f + expf(-s));
  }
  __syncthreads();
  const float g = gate;
  long long *ar = acc + static_cast<size_t>(t) * kHidden;
  const float *sr = shared + static_cast<size_t>(t) * kHidden;
  __nv_bfloat16 *orow = out + static_cast<size_t>(t) * kHidden;
  for (int k = threadIdx.x; k < kHidden; k += kCombineThreads) {
    const float routed = from_fixed(ar[k]);
    orow[k] = __float2bfloat16_rn(fmaf(g, sr[k], routed));
    ar[k] = 0;
  }
}

}  // namespace
}  // namespace ignis_moe

using namespace ignis_moe;

extern "C" int32_t ignis_moe_shared_expert(const void *gate, const void *up, const void *down, const void *x,
                                           uint32_t tokens, void *h, float *shared, void *stream) {
  if (down == nullptr || h == nullptr || shared == nullptr) return fail("ignis_moe_shared_expert: NULL pointer");
  if (ignis_fp8_linear_swiglu(gate, up, kInter, kHidden, x, tokens, h, stream) != 0) return -1;
  return ignis_fp8_linear(down, kHidden, kInter, h, tokens, shared, 1, stream);
}

extern "C" int32_t ignis_moe_combine(int64_t *acc, const float *shared, const void *x, const void *w_gate,
                                     uint32_t tokens, void *out, void *stream) {
  if (acc == nullptr || shared == nullptr || x == nullptr || w_gate == nullptr || out == nullptr) {
    return fail("ignis_moe_combine: NULL pointer");
  }
  if (tokens == 0) return fail("ignis_moe_combine: tokens must be at least 1");
  combine_kernel<<<tokens, kCombineThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<long long *>(acc), shared, static_cast<const __nv_bfloat16 *>(x),
      static_cast<const __nv_bfloat16 *>(w_gate), static_cast<__nv_bfloat16 *>(out));
  return check_launch("ignis_moe_combine");
}
