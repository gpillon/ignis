// ignis kernel leaf: what Flash-Next's two routed-expert decode kernels share -- OURS
// (kernel/include/ignis_moe.h; moe_decode.cu, moe_decode_cluster.cu). Leaf-internal.
#ifndef IGNIS_MOE_DECODE_COMMON_CUH
#define IGNIS_MOE_DECODE_COMMON_CUH

#include "moe_common.cuh"
#include "moe_workspace.cuh"

#include <cstdint>

namespace ignis_moe {

// The distinct experts of the call in order of first appearance (token-major, rank order),
// with each token's routing weight for them and the mask of tokens that selected them, into the
// kernel's shared state `s`, which has n_unique, unique_id[kDecodeMaxUnique],
// unique_sel[kDecodeMaxUnique], unique_w[kDecodeMaxUnique][kDecodeMaxTokens] and the scratch
// first[] / slot_of[] [kDecodeMaxTokens * kTopK]. Every thread of the block takes part.
template <typename S>
__device__ void build_unique(S &s, const int32_t *ids, const float *weights, int tokens) {
  const int n = tokens * kTopK;
  const int i = threadIdx.x;
  for (int j = threadIdx.x; j < kDecodeMaxUnique * kDecodeMaxTokens; j += blockDim.x) {
    (&s.unique_w[0][0])[j] = 0.0f;
  }
  if (i < kDecodeMaxUnique) s.unique_sel[i] = 0u;
  int e = -1;
  if (i < n) {
    e = ids[i];
    int f = i;
    for (int j = 0; j < i; ++j) {
      if (ids[j] == e) {
        f = j;
        break;
      }
    }
    s.first[i] = f;
  }
  __syncthreads();
  if (i < n && s.first[i] == i) {
    int slot = 0;
    for (int j = 0; j < i; ++j) slot += s.first[j] == j;
    s.slot_of[i] = slot;
    s.unique_id[slot] = e;
  }
  if (i == 0) {
    int count = 0;
    for (int j = 0; j < n; ++j) count += s.first[j] == j;
    s.n_unique = count;
  }
  __syncthreads();
  if (i < n) {
    const int u = s.slot_of[s.first[i]];
    s.unique_w[u][i / kTopK] = weights[i];
    atomicOr(&s.unique_sel[u], 1u << (i / kTopK));
  }
  __syncthreads();
}

}  // namespace ignis_moe

#endif  // IGNIS_MOE_DECODE_COMMON_CUH
