/* ignis kernel leaf - Ticket 06 (kernel-abi-02): dense embedding gather.
 *
 * Provenance: 1:1 port of the reference dense embedding gather
 *   (F:/ai/q38/ninfer/src/ops/kernel/embed_gather.cuh:
 *   embed_gather_dense_kernel -- the grid-stride row copy
 *   out[i] = table[ids[i/d] * d + i % d]). The quantized table variants
 *   (Q6 / W8 / FP8 code + scale planes, the grouped per-token launchers)
 *   and the shape-specific route dispatch are not part of the v1 ABI
 *   contract -- the ABI table is a dense bf16 [vocab][hidden] matrix, so
 *   only the dense gather's math (the row selection and the grid-stride
 *   copy) is ported; the quantized paths are later-ticket material.
 *
 * Kernel: grid-stride over the [batch][hidden] output (one element per
 * iteration, row = ids[i / hidden]); block size 128 with a div_up(n, 128)
 * grid, exactly the reference's grid_for(n) launcher helper. Device-only
 * header. Instantiated by norms_sampling_surface.cu.
 */
#ifndef IGNIS_EMBED_GATHER_CUH
#define IGNIS_EMBED_GATHER_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis {

// Dense embedding gather: out[t * d + k] = table[ids[t] * d + k] for
// t in [0, batch), k in [0, d) (the reference's embed_gather_dense_kernel,
// ported 1:1 with __restrict__ qualifiers added to the leaf style).
__global__ void embed_gather_kernel(const std::int32_t* __restrict__ ids,
                                    const __nv_bfloat16* __restrict__ table,
                                    __nv_bfloat16* __restrict__ out,
                                    std::int32_t d, std::int32_t batch) {
  const std::int64_t n = static_cast<std::int64_t>(d) * batch;
  const std::int64_t start =
      blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = start; i < n; i += stride) {
    const std::int32_t t = static_cast<std::int32_t>(i / d);
    const std::int32_t k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
    out[i] = table[static_cast<std::int64_t>(ids[t]) * d + k];
  }
}

}  // namespace ignis

#endif  // IGNIS_EMBED_GATHER_CUH