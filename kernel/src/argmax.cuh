/* ignis kernel leaf - Ticket 06 (kernel-abi-02): greedy argmax sampling.
 *
 * Provenance: faithful adaptation of the reference argmax kernel
 *   (F:/ai/q38/ninfer/src/ops/kernel/argmax.cuh: argmax_kernel -- the
 *   per-row block reduce with the value/lower-index winner rule, plus the
 *   shared-memory argmax_block_reduce aggregation). The reference reads
 *   bf16 logits with a physical/valid row pair for padded rows, and the
 *   launcher dispatches a tiled atomic-cas route for wide vocabularies;
 *   the ABI contract is an f32 [batch][vocab] matrix with no padding
 *   (vocab == the valid row count), so the single-tile finite route is
 *   ported 1:1 with the input typed to f32, and the tiled-atomic route is
 *   deferred to the performance gate (ADR 0007). The winner rule
 *   (value > best || (value == best && index < best_index) -- ties
 *   resolve to the lowest index, the v1 determinism floor) is ported
 *   1:1.
 *
 * Kernel: one block per row (512 threads, the reference's kArgmaxBlock),
 * grid-stride over the row's vocab; the shared-memory block reduce picks
 * the winner. Device-only header. Instantiated by
 * norms_sampling_surface.cu.
 */
#ifndef IGNIS_ARGMAX_CUH
#define IGNIS_ARGMAX_CUH

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ignis {

// Winner rule (port of the reference's argmax_better): a candidate wins on
// a strictly higher value, or -- on an exact tie -- on the lower index
// (the deterministic v1 floor, ADR 0007: greedy + fixed seed).
__device__ __forceinline__ bool argmax_better(float value, std::int32_t index,
                                              float best_value,
                                              std::int32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

// Warp-level reduce of the (value, index) pair (port of the reference's
// argmax_warp_reduce: shfl_down butterfly over the winner rule).
__device__ __forceinline__ void argmax_warp_reduce(float& value,
                                                   std::int32_t& index) {
  constexpr unsigned int kMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(kMask, value, offset);
    const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
    if (argmax_better(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
    }
  }
}

// Shared-memory block reduce (port of the reference's argmax_block_reduce):
// per-warp winners land in shared slots, warp 0 aggregates them (padding
// slots carry -inf / INT32_MAX so they can never win), and the result is
// broadcast from the head of warp 0. Requires a 512-thread block (16
// warp slots, the reference's kArgmaxBlock).
__device__ __forceinline__ void argmax_block_reduce(float& value,
                                                    std::int32_t& index) {
  __shared__ float warp_values[16];
  __shared__ std::int32_t warp_indices[16];

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  argmax_warp_reduce(value, index);
  if (lane == 0) {
    warp_values[warp] = value;
    warp_indices[warp] = index;
  }
  __syncthreads();

  value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
  index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
  if (warp == 0) {
    argmax_warp_reduce(value, index);
  }
}

// One block per row (512 threads, the reference's kArgmaxBlock). Each
// thread grid-strides its slice of the row's vocab, initializing the
// per-thread winner with element 0 (logits[base], index 0 -- always a
// valid candidate since vocab >= 1); the block reduce picks the row's
// winner.
__global__ void argmax_kernel(const float* __restrict__ logits,
                              std::int32_t* __restrict__ out,
                              std::int32_t batch, std::int32_t vocab) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x);
  const std::int64_t base = static_cast<std::int64_t>(t) * vocab;

  float best_value = logits[base];
  std::int32_t best_index = 0;
  for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < vocab;
       v += blockDim.x) {
    const float value = logits[base + v];
    if (argmax_better(value, v, best_value, best_index)) {
      best_value = value;
      best_index = v;
    }
  }

  argmax_block_reduce(best_value, best_index);
  if (threadIdx.x == 0) {
    out[t] = best_index;
  }
}

}  // namespace ignis

#endif  // IGNIS_ARGMAX_CUH