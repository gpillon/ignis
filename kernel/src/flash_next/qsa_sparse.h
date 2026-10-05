// ignis kernel leaf -- the Flash-Next QSA sparse attention (spec flash-next/04, GitHub #302,
// slice S3): OURS (ADR 0043), no port claim. Above dense_threshold() visible tokens every query
// row attends only to the tokens its indexer selection lists (indexer.h); this is that
// attention, called by S2's fn_qsa_attention when the Selection is not dense, after its q/k
// norms and rope, and before its sigmoid gate and o_proj (coordinator, 2026-10-05: one gathered
// kernel for prefill rows and decode lanes). BF16 KV: attend after the call's K/V append (it reads
// the pages). hq-e8-2b KV: decode_listed_hq / decode_visible_hq BEFORE the append, with the call's
// own rows as fresh (see HqSource), then the append, then attend.
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
// BF16 scratch the hq-e8-2b route decodes rows into: a decode call's listed rows (read by list
// index), or a prefill lane's visible rows once per layer (read by position).

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
  enum class Mode : int32_t {
    // BF16 pages of one attention layer: element (page, head, offset, d) at
    // ((page * kv_heads + head) * 64 + offset) * head_dim + d; a lane's page table row is its slot's.
    Paged = 0,
    // BF16 [rows][selection_width][kv_heads][head_dim]: row r's i-th listed token at
    // r * selection_width + i (decode_listed_hq's output).
    ByIndex = 1,
    // BF16 [positions][kv_heads][head_dim] of the call's one lane, by absolute position
    // (decode_visible_hq's output).
    ByPosition = 2,
  };
  const __nv_bfloat16 *k = nullptr;
  const __nv_bfloat16 *v = nullptr;
  const int32_t *block_tables = nullptr;  // Paged only
  int32_t logical_pages = 0;              // Paged only
  int32_t kv_heads = 0;
  Mode mode = Mode::Paged;
};

// hq-e8-2b K/V of one attention layer (ADR 0022), and where its rows are read EXACTLY instead of
// decoded -- the 27B's rules (vendored gqa_attention_*_hq), so the sparse route reads the same
// values as S2's dense hq route:
//   1. fresh: positions at or past the lane's first position in the call come from the call's own
//      BF16 K/V [rows][kv_heads][head_dim] (plain frame), when given;
//   2. residual window: positions below kGqaHqSinkKeys, or in the recent window with their ring
//      bit set, come from the side planes (rotated frame, [slot][sink + recent][kv_heads]
//      [head_dim], ring words [slot][recent / 32]), when given. With fresh rows the window is the
//      kGqaHqRecentKeys BEFORE the call's first position, and the decode must run BEFORE the
//      call's append: the append writes the call's keys into the ring slots of keys
//      [first - recent, end - recent), which this would read as theirs (the vendored prompt route's
//      has_fresh rule, attend-first since #258). Without fresh rows the window is the call's last
//      kGqaHqRecentKeys, its own keys among them, read AFTER the append (the vendored decode rule);
//   3. otherwise the codec row (code plane [64][64][kv_heads][pages], metadata [8][64][...]),
//      decoded with the vendored hq_codec.cuh device functions (dither seed (head, position,
//      role), the engine sign diagonal).
// The output rows are in the PLAIN frame (rotated rows un-rotated once, rounded once to BF16), so
// attend is format-agnostic.
struct HqSource {
  const uint8_t *k_codes = nullptr, *k_meta = nullptr, *v_codes = nullptr, *v_meta = nullptr;
  const int32_t *block_tables = nullptr;
  int32_t logical_pages = 0;
  int32_t kv_heads = 0;
  const __nv_bfloat16 *fresh_k = nullptr, *fresh_v = nullptr;
  const __nv_bfloat16 *residual_k = nullptr, *residual_v = nullptr;
  const uint32_t *ring_valid = nullptr;
};

// Decode lanes: every listed token of every row into k/v [rows][selection_width][kv_heads][head_dim]
// (listed_hq_bytes each); `out` becomes the ByIndex source over them. Graph-safe.
Status decode_listed_hq(const Geometry &g, const HqSource &hq, const Batch &batch, const Selection &selection,
                        __nv_bfloat16 *k, __nv_bfloat16 *v, KvSource *out, cudaStream_t stream);
std::size_t listed_hq_bytes(const Geometry &g, int32_t rows);

// Prefill (one lane, batch.max_visible exact): positions [0, batch.max_visible) into k/v
// [max_visible][kv_heads][head_dim] (visible_hq_bytes each: 1 KiB per position and role at
// Flash-Next's geometry, 256 MiB for both roles at 128K); `out` becomes the ByPosition source.
Status decode_visible_hq(const Geometry &g, const HqSource &hq, const Batch &batch, __nv_bfloat16 *k,
                         __nv_bfloat16 *v, KvSource *out, cudaStream_t stream);
std::size_t visible_hq_bytes(const Geometry &g, int32_t max_visible);

// Everything a QSA layer call of `rows` rows (`tokens` per lane) needs from the arena for the
// sparse route and the hq decode, at the load's kv_format (enum ignis_kv_format): decode splits'
// partials, plus under hq-e8-2b both roles' decoded rows -- listed rows for a decode call
// (tokens == 1), the lane's visible rows up to max_visible for a prefill call (S2's dense hq
// prefill reads the same). The plan sums the larger of its prefill and decode calls.
std::size_t scratch_bytes(const Geometry &g, int32_t kv_format, int32_t rows, int32_t tokens, int32_t max_visible);

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
