// ignis kernel leaf: Flash-Next's shared expert and the MoE combine -- OURS
// (kernel/include/ignis_moe.h).
//
// The shared expert is the checkpoint's Qwen4ExpTextMLP at intermediate 640 on the FP8 row-scale
// linear: gate/up with the SwiGLU epilogue (h in BF16, as the linear's activations are), then
// down into fp32. The combine adds it, gated by sigmoid(x . w_gate) over the BF16
// shared_expert_gate, to the routed experts' fixed-point sum, rounds once to BF16, and zeroes the
// accumulator it read for the next call. One CTA per token -- at decode widths one per slice of a
// token's row, each computing the token's gate itself; the gate's dot product is a fixed block
// reduction in every CTA, so the result is deterministic and the same at every width.

#include "moe_combine.cuh"
#include "moe_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ignis_moe {
namespace {

// A decode-width call spreads each row over this many CTAs (one element per thread).
constexpr int kCombineSlices = kHidden / kCombineThreads;
static_assert(kCombineSlices * kCombineThreads == kHidden, "a slice is one element per thread");

// The gate and the value are moe_combine.cuh's, which the hyper-connection mix's folded combine
// runs too (GitHub #306).
__global__ void __launch_bounds__(kCombineThreads) combine_kernel(long long *__restrict__ acc, const float *__restrict__ shared,
                                                                  const __nv_bfloat16 *__restrict__ x,
                                                                  const __nv_bfloat16 *__restrict__ w_gate,
                                                                  __nv_bfloat16 *__restrict__ out) {
  __shared__ float partial[kCombineThreads / 32];
  __shared__ float gate;
  const int t = blockIdx.x;
  const float g = combine_gate(x + static_cast<size_t>(t) * kHidden, w_gate, partial, &gate);
  long long *ar = acc + static_cast<size_t>(t) * kHidden;
  const float *sr = shared + static_cast<size_t>(t) * kHidden;
  __nv_bfloat16 *orow = out + static_cast<size_t>(t) * kHidden;
  const int span = kHidden / gridDim.y;
  const int end = (blockIdx.y + 1) * span;
  for (int k = blockIdx.y * span + threadIdx.x; k < end; k += kCombineThreads) {
    orow[k] = combine_value(g, sr[k], ar[k]);
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
  if (require_prepared("ignis_moe_combine", nullptr) != 0) return -1;
  const dim3 grid(tokens, tokens <= IGNIS_MOE_DECODE_MAX_TOKENS ? kCombineSlices : 1);
  combine_kernel<<<grid, kCombineThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<long long *>(acc), shared, static_cast<const __nv_bfloat16 *>(x),
      static_cast<const __nv_bfloat16 *>(w_gate), static_cast<__nv_bfloat16 *>(out));
  return check_launch("ignis_moe_combine");
}
