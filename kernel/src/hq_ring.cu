// ignis kernel leaf - GitHub #257 (spec runtime/06): the apply kernel for
// the hq-e8-2b recent ring's validity words. See kernel/include/ignis_hq_ring.h
// for the rule and why the masks are computed on the host.

#include "ignis_hq_ring.h"

#include <cuda_runtime.h>

namespace {

// One thread per word: the whole row is 16 words, one block.
__global__ void ignis_hq_ring_apply_kernel(std::uint32_t *words, ignis_hq_ring_words and_mask,
                                           ignis_hq_ring_words or_mask) {
  const int i = static_cast<int>(threadIdx.x);
  words[i]    = (words[i] & and_mask.w[i]) | or_mask.w[i];
}

} // namespace

cudaError_t ignis_hq_ring_apply(std::uint32_t *words, const ignis_hq_ring_words &and_mask,
                                const ignis_hq_ring_words &or_mask, cudaStream_t stream) {
  if (words == nullptr) {
    return cudaSuccess;
  }
  ignis_hq_ring_apply_kernel<<<1, kIgnisHqRingWords, 0, stream>>>(words, and_mask, or_mask);
  return cudaGetLastError();
}
