// The attention readout (GitHub #260, ADR 0038). See attention_readout.h.
//
// One warp per key: eight coordinates per lane against a query held in shared
// memory, reduced across the warp, scaled once. The query is rotated into the
// codec's frame first under hq-e8-2b -- the same orthonormal rotation the
// test-only tap's host mirror applies (`hq_rotate`, crates/core/src/attn_tap.rs)
// and that the hq prompt route's key planes are stored in -- so the score is
// the frame-free `q . k / sqrt(head_dim)` either way.
#include "attention_readout.h"

#include "ops/kernel/hq_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

namespace {

constexpr int kHeadDim = 256;
constexpr int kKvHeads = 4;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;
static_assert(kThreads == kHeadDim, "one thread per query coordinate while it is rotated");

// `kHq`: keys from the hq prompt route's plane (`keys` = its base for the KV
// head, rows at absolute positions); otherwise from the BF16 pages.
template <bool kHq>
__global__ void attention_readout_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *keys,
                                         const __nv_bfloat16 *k_pages,
                                         const std::int32_t *block_table, std::int32_t kv_head,
                                         std::int64_t key_begin, std::int64_t key_count,
                                         float scale, float *scores) {
  __shared__ float q[kHeadDim];
  const int t = static_cast<int>(threadIdx.x);
  float value = __bfloat162float(query[t]);
  if (kHq) {
    // R = H * diag(signs) / 16: signs, then the natural-order Walsh-Hadamard
    // butterfly -- (a, b) -> (a + b, a - b) over pairs `len` apart -- then
    // 1/sqrt(256). Orthonormal, so R q . R k == q . k.
    q[t] = value * ninfer::ops::hq_engine_sign(t);
    __syncthreads();
    for (int len = 1; len < kHeadDim; len <<= 1) {
      const float a = q[t];
      const float b = q[t ^ len];
      __syncthreads();
      q[t] = (t & len) ? (b - a) : (a + b);
      __syncthreads();
    }
    value = q[t] * (1.0F / 16.0F);
    __syncthreads();
  }
  q[t] = value;
  __syncthreads();

  const int warp = t / 32;
  const int lane = t % 32;
  for (std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       k < key_count; k += static_cast<std::int64_t>(gridDim.x) * kWarpsPerBlock) {
    const std::int64_t position = key_begin + k;
    const __nv_bfloat16 *row =
        kHq ? keys + position * kHeadDim
            : k_pages + ninfer::ops::paged_kv_element_offset<kHeadDim, kKvHeads>(
                            block_table, kv_head, static_cast<std::int32_t>(position), 0);
    float acc = 0.0F;
#pragma unroll
    for (int j = 0; j < kHeadDim / 32; ++j) {
      const int d = lane + 32 * j;
      acc += q[d] * __bfloat162float(row[d]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
      scores[k] = acc * scale;
    }
  }
}

} // namespace

int32_t ignis_attention_readout_run(const AttentionReadoutTarget &target,
                                    const void *rotated_query, int32_t q_heads, int32_t kv_heads,
                                    int32_t tokens, int64_t visible_keys,
                                    const ninfer::PagedKVBatchLayerView &cache,
                                    bool hq_prompt_scratch, const void *attention_workspace,
                                    int64_t span, float scale, cudaStream_t stream,
                                    const char **error) {
  if (kv_heads != kKvHeads || q_heads % kv_heads != 0 || target.query_head < 0 ||
      target.query_head >= q_heads || tokens <= 0) {
    *error = "attention readout: the layer's head geometry is not the one it was built for";
    return -1;
  }
  const std::int64_t key_end = target.key_begin + target.key_count;
  // Keys past the history are not keys attention read.
  if (target.key_begin < 0 || target.key_count <= 0 || key_end > visible_keys) {
    return 0;
  }
  const bool hq = cache.dtype == ninfer::DType::U8;
  // Under hq-e8-2b only the prompt route's single band holds them: a small-T
  // route decodes in registers and a banded prompt keeps only its last band.
  if (hq && (!hq_prompt_scratch || attention_workspace == nullptr || key_end > span)) {
    return 0;
  }
  const std::int32_t kv_head = target.query_head / (q_heads / kv_heads);
  const auto *query = static_cast<const __nv_bfloat16 *>(rotated_query) +
                      (static_cast<std::int64_t>(tokens) - 1) * q_heads * kHeadDim +
                      static_cast<std::int64_t>(target.query_head) * kHeadDim;
  const std::int64_t wanted_blocks = (target.key_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const unsigned blocks = static_cast<unsigned>(wanted_blocks < 1024 ? wanted_blocks : 1024);
  if (hq) {
    const auto *keys = static_cast<const __nv_bfloat16 *>(attention_workspace) +
                       static_cast<std::int64_t>(kv_head) * span * kHeadDim;
    attention_readout_kernel<true><<<blocks, kThreads, 0, stream>>>(
        query, keys, nullptr, nullptr, kv_head, target.key_begin, target.key_count, scale,
        target.device_scores);
  } else {
    attention_readout_kernel<false><<<blocks, kThreads, 0, stream>>>(
        query, nullptr, static_cast<const __nv_bfloat16 *>(cache.k_pages.data),
        static_cast<const std::int32_t *>(cache.block_tables.data), kv_head, target.key_begin,
        target.key_count, scale, target.device_scores);
  }
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    *error = cudaGetErrorString(launched);
    return -1;
  }
  *target.read = true;
  return 0;
}
