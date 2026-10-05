// ignis kernel leaf -- the Flash-Next QSA sparse attention (spec flash-next/04, GitHub #302,
// slice S3): OURS (ADR 0043), no port claim. Above dense_threshold() visible tokens every query
// row attends only to the tokens its indexer selection lists (indexer.h); this is that
// attention, called by S2's fn_qsa_attention when the Selection is not dense, after its q/k
// norms, rope and the K/V append, and before its sigmoid gate and o_proj (coordinator,
// 2026-10-05: one gathered kernel for prefill rows and decode lanes).
//
// The math is the checkpoint's attention restricted to the selected tokens (eager / sdpa with
// the indexer's mask): for query head h of row r, over the row's listed tokens j,
//   out[r][h] = sum_j softmax_j(q[r][h] . k[j][h / group] / sqrt(head_dim)) v[j][h / group].
//
// One CTA per (row, KV head, split): the group's 12 query heads are the M of BF16 m16n8k16 tensor
// core tiles (padded to 16), the row's list is walked in 32-token tiles whose K and V rows are
// gathered by position through the block table, softmax runs online in fp32 (exp2, the 1/16
// scale folded in). Decode spreads a row's list over several splits and merges them in a second
// kernel; prefill rows run one split each. Graph-safe: grids depend on the row count only, and
// each row's list length is read on the device.
//
// KV sources: the paged BF16 planes (ninfer page-major [head_dim][64][kv_heads][pages]), or a
// BF16 scratch the hq-e8-2b route decodes the selected rows into, read by list index.

#pragma once

#include "flash_next_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::flash_next::sparse {

inline constexpr int32_t kHeadDim = 256;
inline constexpr int32_t kGroup = 12;  // query heads per KV head (24 / 2)
inline constexpr int32_t kTileTokens = 32;

using Status = const char *;

// Where a row's listed tokens' K and V rows are read from.
struct KvSource {
  // Paged (by_index false): BF16 pages of one attention layer, element (page, head, offset, d) at
  // ((page * kv_heads + head) * 64 + offset) * head_dim + d; a lane's page table row is its slot's.
  const __nv_bfloat16 *k = nullptr;
  const __nv_bfloat16 *v = nullptr;
  const int32_t *block_tables = nullptr;
  int32_t logical_pages = 0;
  int32_t kv_heads = 0;
  // By index (by_index true): k/v are BF16 [rows][selection_width][kv_heads][head_dim], row r's
  // i-th listed token at (r * selection_width + i); block_tables is unused.
  bool by_index = false;
};

// The geometry this kernel is written for; anything else is refused by name.
Status check_geometry(const Geometry &g);

// Splits of a row's list a call of `rows` rows uses (1 for wide prefill calls).
int32_t splits_for(const Geometry &g, int32_t rows);

// q: BF16 [rows][q_heads][head_dim] (normed, roped); out: BF16 [rows][q_heads][head_dim].
// selection must not be dense. `partials` holds splits_for > 1's partial results
// (partial_bytes), else may be null.
Status attend(const Geometry &g, const KvSource &kv, const Batch &batch, const __nv_bfloat16 *q,
              const Selection &selection, __nv_bfloat16 *out, void *partials, cudaStream_t stream);
std::size_t partial_bytes(const Geometry &g, int32_t rows);

}  // namespace ignis::flash_next::sparse
