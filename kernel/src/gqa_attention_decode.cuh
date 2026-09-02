/* ignis kernel leaf - Ticket 03: GQA attention decode (paged bf16 KV).
 *
 * Provenance: faithful adaptation of the reference decode attention
 *   F:/ai/q38/ninfer/src/ops/kernel/gqa_attention_decode_bf16.cuh
 * (shared scaffolding in gqa_attention_decode.cuh, paged addressing in
 * paged_kv_address.cuh). The reference kernel is a tensor-core split-KV
 * machine templated on a KV-source policy (bf16 / hq-e8-2b codec) that pulls
 * in the hq codec and prefill-HQ header (out of ticket 03 scope) plus the
 * engine's C++ launch state. Those are stripped here. The GQA head grouping,
 * the paged-KV element offset math (paged_kv_element_offset) and the
 * online-softmax split-reduce structure are ported 1:1; this is a faithful,
 * self-contained (non-tensor-core, non-split-KV) decode attention that
 * compiles standalone in the leaf.
 *
 * KV layout: `kv_cache` is a single buffer of TWO paged planes (K first, V
 * second), each paged as [num_blocks][num_kv_heads][block_size][head_dim]
 * (kv_head-major within a page; head_dim fastest). The V plane starts
 * `num_blocks*num_kv_heads*block_size*head_dim` bf16 elements after the K
 * plane base.
 *
 * Device-only header. Instantiated by decode_surface.cu.
 */
#ifndef IGNIS_GQA_ATTENTION_DECODE_CUH
#define IGNIS_GQA_ATTENTION_DECODE_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ignis {

// Paged bf16 KV element offset (1:1 with the reference paged_kv_element_offset):
//   element_offset = head_dim * block_size * (kv_head + num_kv_heads * page)
//                  + head_dim * block_offset + d
// i.e. [physical_page][kv_head][block_offset][d] with d fastest. `page` is a
// physical page id taken from the block table (logical block -> physical page).
__device__ __forceinline__ std::int64_t paged_kv_element_offset(std::int64_t head_dim,
                                                                std::int64_t num_kv_heads,
                                                                std::int64_t block_size,
                                                                int physical_page, int kv_head,
                                                                int block_offset, int d) {
  return head_dim * block_size *
             (static_cast<std::int64_t>(kv_head) +
              static_cast<std::int64_t>(num_kv_heads) * physical_page) +
         static_cast<std::int64_t>(head_dim) * block_offset + d;
}

// GQA decode attention, single token. One block (head_dim threads, one thread
// per head_dim element) per q head. q head h maps to kv head h / (num_q_heads /
// num_kv_heads). Per key the q.k dot is a block reduce; an online softmax then
// accumulates the value-weighted sum for the thread's element.
__global__ void gqa_attention_decode_kernel(const __nv_bfloat16* __restrict__ kv,
                                            const std::int32_t* __restrict__ block_table,
                                            const __nv_bfloat16* __restrict__ q,
                                            __nv_bfloat16* __restrict__ out, int num_q_heads,
                                            int num_kv_heads, int head_dim, int seq_len,
                                            int block_size, int num_blocks, float softmax_scale) {
  const int h = static_cast<int>(blockIdx.x);
  if (h >= num_q_heads) return;
  const int d = static_cast<int>(threadIdx.x);
  if (d >= head_dim) return;

  const int group   = (num_q_heads / num_kv_heads) > 0 ? (num_q_heads / num_kv_heads) : 1;
  const int kv_head = h / group;
  // K plane base is the buffer start; V plane follows the K plane.
  const std::int64_t plane_elems =
      static_cast<std::int64_t>(num_blocks) * block_size * num_kv_heads * head_dim;
  const __nv_bfloat16* v_base = kv + plane_elems;
  const float qd = __bfloat162float(q[static_cast<std::int64_t>(h) * head_dim + d]);

  __shared__ float block_red[256];
  __shared__ float score;
  float m = -1e30f, l = 0.0f, acc = 0.0f;

  for (int key = 0; key < seq_len; ++key) {
    const int block    = key / block_size;
    const int offset   = key % block_size;
    const int page     = block_table[block];
    const std::int64_t k_off =
        paged_kv_element_offset(head_dim, num_kv_heads, block_size, page, kv_head, offset, d);
    const std::int64_t v_off =
        paged_kv_element_offset(head_dim, num_kv_heads, block_size, page, kv_head, offset, d);
    const float local = qd * __bfloat162float(kv[k_off]);

    // Per-key q.k dot product via a block reduce (one slot per head_dim element).
    block_red[d] = local;
    __syncthreads();
    if (d == 0) {
      float s = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        s += block_red[i];
      }
      score = s * softmax_scale;
    }
    __syncthreads();
    const float sc = score;

    // Online softmax (per-element, value-weighted); m/l converge identically
    // across threads since the per-key score is a shared scalar.
    const float m_new = fmaxf(m, sc);
    const float alpha = expf(m - m_new);
    const float p     = expf(sc - m_new);
    const float vd    = __bfloat162float(v_base[v_off]);
    l = alpha * l + p;
    acc = alpha * acc + p * vd;
    m   = m_new;
  }

  const std::int64_t o = static_cast<std::int64_t>(h) * head_dim + d;
  out[o] = (l > 0.0f) ? __float2bfloat16_rn(acc / l) : __float2bfloat16_rn(0.0f);
}

}  // namespace ignis

#endif  // IGNIS_GQA_ATTENTION_DECODE_CUH