// ignis kernel leaf -- the Flash-Next QSA attention sublayer (spec flash-next/04, GitHub #302,
// slice S2): OURS (ADR 0043), no port claim. fn_qsa_attention (flash_next_internal.h) is the
// program's entry point; this header names the stages under it so the kernel-leaf tests drive
// them on their own page tables, without a seq pool.
//
// The oracle is transformers' Qwen4ExpTextAttention (models/qwen4_exp/modeling_qwen4_exp.py):
//
//   q_h | gate_h = q_proj(x) per head, [256 | 256] of each 512   (view(.., -1, 512).chunk(2))
//   q = rope(q_norm(q_h)), k = rope(k_norm(k_proj(x))), v = v_proj(x)
//                    (1 + w) RMSNorm in fp32 rounded once to BF16; rope on dims [0, 64) of each
//                    head, pairs (i, i + 32), in torch's BF16 arithmetic (cos and sin rounded to
//                    BF16, each product rounded, then their sum) -- the indexer's convention
//   o = softmax(q k^T / 16) v over the row's visible tokens (causal), query head h on KV head
//       h / 12 (repeat_kv); or over the indexer's selection (S3's sparse route)
//   y = o_proj(bf16(o * bf16(sigmoid(gate))))
//
// One call, in order: the three projections (fn_linear), `prepare` (norms, rope), under hq-e8-2b
// S3's decode of the rows the call reads into a plain-frame BF16 scratch (it must precede the
// append: the residual window's ring rows of the keys before the call are overwritten by it),
// `append` (the call's K/V into the lanes' pages and, under hq, the residual window), the
// attention (`attend_dense` when the selection is dense -- prefill, one lane, at most
// dense_threshold() keys -- else S3's sparse::attend), `gate`, and o_proj.

#pragma once

#include "flash_next_internal.h"
#include "indexer.h"
#include "qsa_sparse.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::flash_next::qsa {

// The geometry these kernels are written for; anything else is refused by name.
inline constexpr int32_t kQHeads = 24;
inline constexpr int32_t kKvHeads = 2;
inline constexpr int32_t kHeadDim = 256;
inline constexpr int32_t kRotaryDim = 64;
inline constexpr int32_t kGroup = kQHeads / kKvHeads;          // 12
inline constexpr int32_t kQProjWidth = kQHeads * 2 * kHeadDim;  // 12288: [q 256 | gate 256] per head
inline constexpr int32_t kKvWidth = kKvHeads * kHeadDim;        // 512
inline constexpr int32_t kOutWidth = kQHeads * kHeadDim;        // 6144
// The most lanes a decode call (one token each) takes: S3's listed hq decode is sized by it.
inline constexpr int32_t kMaxDecodeLanes = 8;

using Status = const char *;

// One attention layer's K/V in the lanes' pages (the seq pool's planes of the layer, or a test's).
struct Kv {
  int32_t kv_format = 0;                  // enum ignis_kv_format
  const int32_t *block_tables = nullptr;  // DEVICE [slots][logical_pages]: physical page ids
  int32_t logical_pages = 0;
  int32_t slots = 0;                      // block-table rows; a lane's slot outside them traps
  // BF16: the K and V planes, page-major [page][kv_heads][64][256]. hq-e8-2b: the code planes
  // [page][kv_heads][64][64 bytes] and the metadata planes [page][kv_heads][64][8 bytes].
  void *k = nullptr;
  void *v = nullptr;
  void *k_meta = nullptr;
  void *v_meta = nullptr;
  // hq-e8-2b's residual window: side planes [slot][32 + 512][kv_heads][256] (rotated frame) and
  // ring words [slot][512 / 32]. Null on BF16.
  __nv_bfloat16 *residual_k = nullptr;
  __nv_bfloat16 *residual_v = nullptr;
  uint32_t *ring = nullptr;
};

// The geometry check fn_qsa_attention makes.
Status check_geometry(const Geometry &g);

// qg: the q_proj output, BF16 [rows][kQHeads][512]; k: the k_proj output, BF16 [rows][kKvHeads]
// [256], normed and roped in place; q: BF16 [rows][kQHeads][256], the normed, roped queries.
// Graph-safe.
Status prepare(const Geometry &g, const indexer::Rope &rope, const void *q_norm, const void *k_norm,
               const Batch &batch, const __nv_bfloat16 *qg, __nv_bfloat16 *q, __nv_bfloat16 *k,
               cudaStream_t stream);

// The call's K/V rows (BF16 [rows][kKvHeads][256], k roped) into the lanes' pages at their
// positions; under hq-e8-2b encoded with the vendored codec, and the rows the residual window
// keeps (sink keys, and the last 512 of the lane's frontier after the call) written to the side
// planes with their ring bits. Graph-safe.
Status append(const Geometry &g, const Kv &kv, const Batch &batch, const __nv_bfloat16 *k,
              const __nv_bfloat16 *v, cudaStream_t stream);

// Causal attention for a call of ONE lane: row t (position positions[0] + t) over the keys
// [0, positions[0] + t] read from `source` (S3's KvSource: the BF16 pages, Paged, whose lane slot
// must be below `slots`, or a decoded scratch, ByPosition). q: BF16 [tokens][kQHeads][256]; out:
// BF16 [tokens][kQHeads][256], before the gate. Requires batch.max_visible <= dense_threshold().
Status attend_dense(const Geometry &g, const sparse::KvSource &source, int32_t slots, const Batch &batch,
                    const __nv_bfloat16 *q, __nv_bfloat16 *out, cudaStream_t stream);

// out = bf16(out * bf16(sigmoid(gate))) in place, gate the second half of each head in qg.
Status gate(const Batch &batch, const __nv_bfloat16 *qg, __nv_bfloat16 *out, cudaStream_t stream);

// The whole sublayer for one call on the given K/V -- fn_qsa_attention's body, which only builds
// `kv` from the seq pool. x, y: BF16 [rows][hidden]. 0, or -1 with fn_set_error.
int32_t run(const Geometry &g, const Kv &kv, const indexer::Rope &rope, const QsaWeights &w, const Batch &batch,
            const void *x, const Selection &selection, void *y, ninfer::DeviceArena &scratch, cudaStream_t stream);

}  // namespace ignis::flash_next::qsa
