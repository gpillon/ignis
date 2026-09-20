#include "permitted_tokens.h"

#include <cuda_bf16.h>
#include <math.h>

namespace {

/// What a masked column reads as. See the header for why it is not `-inf`.
constexpr float kMasked = -1.0e30f;

/// The mask. One block row per lane, a grid-stride walk of the vocabulary
/// inside it; membership is a linear scan of at most `max_permitted` ids,
/// which at 32 entries is cheaper in registers than any set structure would
/// be to build.
__global__ void permit_mask_kernel(__nv_bfloat16 *logits, int32_t vocab, const int32_t *permitted,
                                   const int32_t *counts, int32_t max_permitted) {
  const int lane = blockIdx.y;
  const int32_t count = counts[lane];
  if (count <= 0) {
    return;
  }
  const int32_t *ids = permitted + static_cast<size_t>(lane) * max_permitted;
  __nv_bfloat16 *row = logits + static_cast<size_t>(lane) * vocab;
  const int stride = blockDim.x * gridDim.x;
  for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += stride) {
    bool permitted_column = false;
    for (int32_t k = 0; k < count; ++k) {
      if (ids[k] == v) {
        permitted_column = true;
        break;
      }
    }
    if (!permitted_column) {
      row[v] = __float2bfloat16(kMasked);
    }
  }
}

/// The per-lane restricted probability, one thread per lane. At most
/// `max_permitted` reads each, so a single small block covers every lane of
/// the widest round.
__global__ void permit_probability_kernel(const __nv_bfloat16 *logits, int32_t vocab,
                                          uint32_t lanes, const int32_t *permitted,
                                          const int32_t *counts, int32_t max_permitted,
                                          const int32_t *chosen, float *out) {
  const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= lanes) {
    return;
  }
  const int32_t count = counts[lane];
  if (count <= 0) {
    out[lane] = 0.0f;
    return;
  }
  const int32_t *ids = permitted + static_cast<size_t>(lane) * max_permitted;
  const __nv_bfloat16 *row = logits + static_cast<size_t>(lane) * vocab;
  const int32_t picked = chosen[lane];

  float highest = -INFINITY;
  for (int32_t k = 0; k < count; ++k) {
    const int32_t id = ids[k];
    if (id < 0 || id >= vocab) {
      continue;
    }
    const float value = __bfloat162float(row[id]);
    if (value > highest) {
      highest = value;
    }
  }
  if (!isfinite(highest)) {
    out[lane] = 0.0f;
    return;
  }
  float total = 0.0f;
  float own = 0.0f;
  for (int32_t k = 0; k < count; ++k) {
    const int32_t id = ids[k];
    if (id < 0 || id >= vocab) {
      continue;
    }
    const float weight = expf(__bfloat162float(row[id]) - highest);
    total += weight;
    if (id == picked) {
      own = weight;
    }
  }
  // A draw that landed outside the set would leave `own` at zero, which is
  // the honest reading: the probability the set gave what was committed.
  out[lane] = (total > 0.0f) ? (own / total) : 0.0f;
}

int32_t launched(const char *what, cudaStream_t stream) {
  (void)what;
  (void)stream;
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // namespace

int32_t ignis_permit_mask(void *logits, int32_t vocab, uint32_t lanes, const int32_t *permitted,
                          const int32_t *counts, int32_t max_permitted, cudaStream_t stream) {
  if (logits == nullptr || permitted == nullptr || counts == nullptr || lanes == 0 || vocab <= 0) {
    return -1;
  }
  constexpr int kThreads = 256;
  // Enough blocks to cover the vocabulary once without a long grid-stride
  // loop, capped so a wide round stays a single launch.
  const int columns = (vocab + kThreads - 1) / kThreads;
  const dim3 grid(static_cast<unsigned>(columns > 512 ? 512 : columns), lanes);
  permit_mask_kernel<<<grid, kThreads, 0, stream>>>(static_cast<__nv_bfloat16 *>(logits), vocab,
                                                    permitted, counts, max_permitted);
  return launched("ignis_permit_mask", stream);
}

int32_t ignis_permit_probability(const void *logits, int32_t vocab, uint32_t lanes,
                                 const int32_t *permitted, const int32_t *counts,
                                 int32_t max_permitted, const int32_t *chosen, float *out,
                                 cudaStream_t stream) {
  if (logits == nullptr || permitted == nullptr || counts == nullptr || chosen == nullptr ||
      out == nullptr || lanes == 0 || vocab <= 0) {
    return -1;
  }
  constexpr int kThreads = 64;
  const unsigned blocks = (lanes + kThreads - 1) / kThreads;
  permit_probability_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(logits), vocab, lanes, permitted, counts, max_permitted,
      chosen, out);
  return launched("ignis_permit_probability", stream);
}
