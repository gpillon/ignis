// ignis kernel leaf -- the Flash-Next QSA indexer's stages (spec flash-next/04, GitHub #302,
// slice S3): OURS (ADR 0043), no port claim. fn_indexer_select (flash_next_internal.h) is the
// program's entry point; this header names the stages under it so the kernel-leaf tests can
// drive them on synthetic page tables without a seq pool.
//
// The oracle is transformers' Qwen4ExpTextQSAIndexer (models/qwen4_exp/modeling_qwen4_exp.py),
// verified op for op by kernel/tests/fixtures/flash_next_indexer/record_indexer.py:
//
//   qk = index_qk_proj(x)                       [rows][(heads + 1) * head_dim]: q heads, then the key
//   q_h = rope(q_layernorm(q_h), position)      (1 + w) RMSNorm, then rope on the first rotary_dim
//   block b of a sequence = its tokens 4b..4b+3 (blocks start at position 0)
//   k_b = rope(k_layernorm(bf16(mean_fp32(raw keys of b))), 4b)       rope at the block's FIRST token
//   score(row, b) = sum_h relu(q_h . k_b) / sqrt(head_dim)             for every complete visible b
//   selection(row) = the min(512, blocks) best blocks + the row's incomplete block's tokens
//
// GitHub #558: a token's "position" in rope() is its rotation, the attention's (qsa.h): on a span
// of a multimodal prompt its three axes (pair i of the rotary quarter at axis i % 3), after an
// image its KV position plus the lane's rope delta, else its KV position ("4b" is the block's first
// token's). A block's key rotates at its first token's, which may sit in an earlier call: a vision
// load keeps that token's three axes in the block's own key slot until the key replaces it
// (AxisRecord below). Scoring, selection and causality count KV positions.
//
// Numerics kept from the checkpoint: every RMSNorm in fp32 rounded once to BF16; rope in BF16
// arithmetic as torch runs it (cos/sin times the table's YaRN attention factor rounded to BF16,
// each product rounded to BF16, then their sum), with the checkpoint's fp32 frequency table (1.0f / (float)theta^(2i/rotary), which is not
// (float) of the double table for 15 of the 32 pairs) and phi = fp32(position) * inv_freq in fp32;
// the pooled key's mean in fp32 rounded to BF16; dot products exact BF16 products summed in fp32;
// the 1/sqrt(head_dim) applied as torch CUDA applies a division by a host scalar (a multiplication
// by its fp32 reciprocal).
//
// The tie rule (documented, kernel/tests/fixtures/flash_next_indexer/): the k largest scores, and
// among scores equal to the k-th the LOWEST block indices -- torch CUDA topk's gather order, the
// one the converter's references ran. Exact-zero scores (relu) make such ties common.

#pragma once

#include "flash_next_internal.h"

#include "core/paged_kv_cache.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::flash_next::indexer {

// The geometry these kernels are written for; anything else is refused by name.
inline constexpr int32_t kHeadDim = 128;
inline constexpr int32_t kRotaryDim = 64;
inline constexpr int32_t kCompress = 4;
inline constexpr int32_t kPageTokens = ninfer::kPagedKVPageSize;
inline constexpr int32_t kBlocksPerPage = kPageTokens / kCompress;

// One attention layer's indexer section, as the kernels address it.
struct Paged {
  const int32_t *block_tables = nullptr;  // DEVICE [table rows][logical_pages]: physical KV page ids
  int32_t logical_pages = 0;              // a block-table row's length
  __nv_bfloat16 *block_keys = nullptr;    // [physical page][kBlocksPerPage][kHeadDim]
  __nv_bfloat16 *tail_keys = nullptr;     // [slot][kCompress - 1][kHeadDim]
  // GitHub #558: a vision load's section. An incomplete block's key slot holds its first token's
  // AxisRecord -- written by the call that appends that token, read and then replaced by the key
  // in the call that completes the block -- so a block whose first token sits in an earlier call
  // rotates at that token's three axes. Off (a text load): nothing is written there, and every
  // block rotates at its first token's KV position, as before.
  bool axis_records = false;
};

// GitHub #558: what an incomplete block's key slot holds on a vision load, in its first bytes,
// until the block completes. Nothing reads a key slot before its block completes (a row's scores
// stop at its complete blocks), and the slot moves with its KV page wherever the page goes (a
// checkpoint's copied tail page, a snapshot's or a spill's indexer keys), so the record does too.
// Each attention layer keeps its own, in its own section: a call reads a layer's record before
// that layer's key replaces it. A slot that holds none -- its magic or its block's first position
// is not this one's, as in a section written before records existed -- rotates the block at that
// KV position on all three axes, a text history's rotation.
struct AxisRecord {
  int32_t magic = 0;  // kAxisRecordMagic
  int32_t first = 0;  // the block's first token's KV position (4b)
  int32_t axes[3] = {};
};
inline constexpr int32_t kAxisRecordMagic = 0x33584931;  // "1IX3" little-endian
static_assert(sizeof(AxisRecord) == 5 * sizeof(int32_t) && sizeof(AxisRecord) <= kHeadDim * 2,
              "an AxisRecord is five words, inside one block key's slot");

// The checkpoint's rope frequencies, fp32, one per rotary pair, and the YaRN attention factor
// that multiplies cos and sin on queries and keys alike (HF's `attention_scaling`, ADR 0059):
// 1.0f, an exact multiply, without scaling.
struct Rope {
  float inv_freq[kRotaryDim / 2] = {};
  float scale = 1.0F;
};

// The checkpoint's fp32 table from the load's double table: 1.0f / (float)(1 / inv_frequency[i]),
// and its attention_factor as the scale.
Rope rope_from(const ninfer::ops::RopeFrequencies &frequencies);

// A stage returns nullptr, or a static message naming what it refused or what failed.
using Status = const char *;

// The geometry check every stage makes.
Status check_geometry(const Geometry &g);

// Every call, dense or not: for each lane, the keys of the blocks this call completes (their
// earlier tokens' raw keys from the lane's tail) written to their pages, then the raw keys of its
// incomplete last block written to the tail -- and with paged.axis_records, when this call begins
// that block, its first token's AxisRecord to the block's key slot. A block key rotates at its
// first token's rotation: the batch's source (its three axes, its positions plus its rope deltas,
// or its positions) for a token of this call, the block's record for an earlier one. qk: BF16
// [rows][(heads + 1) * kHeadDim], the key at column heads * kHeadDim. A batch naming both rope
// sources is refused. Graph-safe: grids from batch.lanes / batch.tokens only.
Status append_keys(const Geometry &g, const Paged &paged, const Rope &rope, const void *k_norm,
                   const Batch &batch, const void *qk, cudaStream_t stream);

// The row-local parts of selection for rows [row_begin, row_begin + rows) of the batch, each
// written at its local row index (0 .. rows):
//   prepare_queries: q [rows][heads][kHeadDim] BF16, normed and roped at the row's rotation (the
//          batch's three axes, its position plus its lane's rope delta, or its position);
//   score: scores [rows][score_stride] fp32 for every complete visible block of the row (columns
//          past the row's block count are not written); max_blocks bounds the grid (host);
//   select_blocks: the row's token list into tokens[row][selection_width] / counts[row]: every
//          visible token when it has at most budget / compress blocks, else the selected blocks'
//          tokens then its incomplete block's, ascending; -1 past counts[row].
// Rows of one score call must not straddle lanes unless batch.tokens == 1.
Status prepare_queries(const Geometry &g, const Rope &rope, const void *q_norm, const Batch &batch,
                       int32_t row_begin, int32_t rows, const void *qk, __nv_bfloat16 *q,
                       cudaStream_t stream);
Status score(const Geometry &g, const Paged &paged, const Batch &batch, int32_t row_begin, int32_t rows,
             const __nv_bfloat16 *q, int32_t max_blocks, float *scores, int32_t score_stride,
             cudaStream_t stream);
Status select_blocks(const Geometry &g, const Batch &batch, int32_t row_begin, int32_t rows,
                     const float *scores, int32_t score_stride, int32_t *tokens, int32_t *counts,
                     cudaStream_t stream);

// Prepare, score and select every row of the batch, in waves of at most kWaveRows rows through
// scratch; out.tokens / out.counts are written at the batch's global rows. A call whose
// batch.max_visible is at most dense_threshold() (prefill only) sets out.dense and launches nothing.
inline constexpr int32_t kWaveRows = 256;
Status select(const Geometry &g, const Paged &paged, const Rope &rope, const void *q_norm,
              const Batch &batch, const void *qk, Selection &out, ninfer::DeviceArena &scratch,
              cudaStream_t stream);
std::size_t select_scratch_bytes(const Geometry &g, int32_t rows, int32_t max_visible);

}  // namespace ignis::flash_next::indexer
