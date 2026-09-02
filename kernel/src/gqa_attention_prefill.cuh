/* ignis kernel leaf - Ticket 05 (kernel-abi-01): GQA prefill attention (batched).
 *
 * Provenance: faithful adaptation of the reference causal GQA prefill
 *   F:/ai/q38/ninfer/src/ops/kernel/gqa_attention_prefill_bf16.cuh (shared
 *   scaffolding in gqa_attention_prefill_common.cuh, paged addressing in
 *   paged_kv_address.cuh). The reference kernel is a tensor-core
 *   FlashAttention-2 forward (cp.async K/V staging, m16n8k16 MMA, online
 *   softmax in exp2, the hq-e8-2b rotated-frame variant) templated on the
 *   engine's Geometry/Metadata dispatch. That dispatch layer is C++ state,
 *   which the flat C ABI of ADR 0001 forbids across the boundary, so it is not
 *   ported; the tensor-core micro-optimization is likewise deferred to the
 *   performance gate (ADR 0007). The bottom-right causal mask, the paged-KV
 *   element-offset math (paged_kv_element_offset, shared with ticket 03) and
 *   the online-softmax (value-weighted running max/sum) are ported 1:1; this is
 *   a faithful, self-contained (non-tensor-core) causal prefill that compiles
 *   standalone in the leaf.
 *
 * Layout (batched, 1:1 extension of the ticket-03 decode layout):
 *   q  : bf16 [batch][seq_len][num_q_heads][head_dim] (head_dim fastest).
 *   kv : bf16, two paged planes (K first, V second), each
 *        [batch][num_blocks][num_kv_heads][block_size][head_dim] (kv_head-major
 *        within a page; head_dim fastest). The V plane
 *        starts `batch * plane_elems` bf16 elements after the K plane base,
 *        where plane_elems = num_blocks * block_size * num_kv_heads * head_dim
 *        (per batch). So batch b's K plane is `kv + b*plane_elems` and its V
 *        plane is `kv + (batch + b)*plane_elems`.
 *   block_table: i32 [batch][num_blocks] (logical block -> physical page, per
 *        batch row).
 *   out: bf16 [batch][seq_len][num_q_heads][head_dim].
 *
 * Causal: query position i (0-indexed within the batch's seq_len) attends to
 * keys [0, i] (bottom-right alignment, base_pos = 0 -- the fresh-prompt
 * prefill path).
 *
 * Device-only header. Instantiated by prefill_gdn_surface.cu.
 */
#ifndef IGNIS_GQA_ATTENTION_PREFILL_CUH
#define IGNIS_GQA_ATTENTION_PREFILL_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ignis {

// Paged bf16 KV element offset within a batch's plane (1:1 with ticket 03's
// paged_kv_element_offset, see gqa_attention_decode.cuh). `page` is a physical
// page id taken from the batch's block table (logical block -> physical page).
__device__ __forceinline__ std::int64_t prefill_paged_offset(std::int64_t head_dim,
                                                             std::int64_t num_kv_heads,
                                                             std::int64_t block_size,
                                                             int physical_page, int kv_head,
                                                             int block_offset, int d) {
  return head_dim * block_size *
             (static_cast<std::int64_t>(kv_head) +
              static_cast<std::int64_t>(num_kv_heads) * physical_page) +
         static_cast<std::int64_t>(head_dim) * block_offset + d;
}

// Causal GQA prefill attention. One block per (q head, position, batch); each
// thread owns one head_dim element. Per key a block reduce yields the q.k score
// (a shared scalar across the block's head_dim threads); an online softmax
// (m/l/acc) then accumulates the value-weighted sum. Query position `pos`
// attends to keys [0, pos] (bottom-right causal, base_pos = 0).
__global__ void gqa_attention_prefill_kernel(const __nv_bfloat16* __restrict__ kv,
                                             const std::int32_t* __restrict__ block_table,
                                             const __nv_bfloat16* __restrict__ q,
                                             __nv_bfloat16* __restrict__ out, int batch,
                                             int seq_len, int num_q_heads, int num_kv_heads,
                                             int head_dim, int block_size, int num_blocks,
                                             float softmax_scale) {
  const int h   = static_cast<int>(blockIdx.x);  // q head
  const int pos = static_cast<int>(blockIdx.y);  // position in [0, seq_len)
  const int b   = static_cast<int>(blockIdx.z);  // batch
  const int d   = static_cast<int>(threadIdx.x);  // head_dim element

  if (h >= num_q_heads || pos >= seq_len || b >= batch || d >= head_dim) return;

  const int group   = (num_q_heads / num_kv_heads) > 0 ? (num_q_heads / num_kv_heads) : 1;
  const int kv_head = h / group;
  const std::int64_t plane_elems =
      static_cast<std::int64_t>(num_blocks) * block_size * num_kv_heads * head_dim;
  // Batch b's K plane base and V plane base (the V plane follows ALL batches' K).
  const __nv_bfloat16* k_plane = kv + static_cast<std::int64_t>(b) * plane_elems;
  const __nv_bfloat16* v_plane = kv + static_cast<std::int64_t>(batch + b) * plane_elems;
  const std::int32_t* b_table  = block_table + static_cast<std::int64_t>(b) * num_blocks;

  // q row for (batch b, position pos, head h), element d.
  const std::int64_t q_off =
      (static_cast<std::int64_t>(b) * seq_len + pos) *
          (static_cast<std::int64_t>(num_q_heads) * head_dim) +
      static_cast<std::int64_t>(h) * head_dim + d;
  const float qd = __bfloat162float(q[q_off]);

  extern __shared__ float s_red[];  // [head_dim] scratch for the per-key block reduce
  __shared__ float s_score;

  float m = -1e30f, l = 0.0f, acc = 0.0f;

  // Causal: position pos attends to keys [0, pos] (pos+1 keys, base_pos = 0).
  for (int key = 0; key <= pos; ++key) {
    const int blk  = key / block_size;
    const int off  = key % block_size;
    const int page = b_table[blk];
    const std::int64_t paged =
        prefill_paged_offset(head_dim, num_kv_heads, block_size, page, kv_head, off, d);

    // Per-key q.k dot product via a block reduce (one slot per head_dim element).
    s_red[d] = qd * __bfloat162float(k_plane[paged]);
    __syncthreads();
    if (d == 0) {
      float s = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        s += s_red[i];
      }
      s_score = s * softmax_scale;
    }
    __syncthreads();
    const float sc = s_score;

    // Online softmax (per-element value-weighted); m/l/acc converge identically
    // across the block since the per-key score is a shared scalar.
    const float m_new = fmaxf(m, sc);
    const float alpha = expf(m - m_new);
    const float p     = expf(sc - m_new);
    const float vd    = __bfloat162float(v_plane[paged]);
    l   = alpha * l + p;
    acc = alpha * acc + p * vd;
    m   = m_new;
  }

  const std::int64_t o_off =
      (static_cast<std::int64_t>(b) * seq_len + pos) *
          (static_cast<std::int64_t>(num_q_heads) * head_dim) +
      static_cast<std::int64_t>(h) * head_dim + d;
  out[o_off] = (l > 0.0f) ? __float2bfloat16_rn(acc / l) : __float2bfloat16_rn(0.0f);
}

}  // namespace ignis

#endif  // IGNIS_GQA_ATTENTION_PREFILL_CUH