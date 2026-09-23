// The attention readout (GitHub #260, ADR 0038): one query head of one GQA
// layer, at a prefill chunk's last position, dotted with the keys of one span
// exactly as that layer's attention read them -- and, with a head set
// (GitHub #263, ADR 0039), every head of the set that layer holds, each
// reduced on the device to the one key it peaks on.
//
// Internal, not part of the flat C ABI: the ABI half is
// `ignis_prefill_options`'s attention fields (kernel/include/ignis_step.h).
// `kernel/src/step.cu` arms a readout for the span's last chunk and hands
// every armed GQA layer an `AttentionReadoutTarget`; `run_gqa_layer`
// (kernel/src/gqa_layer.cu) calls `ignis_attention_readout_run` right after
// the layer's attention, while the keys it read are still where it read
// them.
#pragma once

#include "core/paged_kv_cache.h"
#include "ignis_gqa_workspace.h"
#include "ignis_seq_internal.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

// The GQA layers a readout may arm: all of them.
constexpr int kReadoutGqaLayers = kIgnisGqaLayerCount;
// The most heads of a set one layer can hold: all of its query heads.
constexpr int kReadoutLayerHeads = kIgnisGqaQHeads;
// The most heads a whole set may name -- every query head of every GQA
// layer -- and so what a vision load reserves the set's results for.
constexpr int kReadoutMaxSetHeads = kReadoutGqaLayers * kReadoutLayerHeads;
// The most keys a set's argmax may skip (the fallback cells), carried as
// launch arguments rather than memory.
constexpr int kReadoutMaxExcluded = 32;

// GitHub #264: the host's inverse of the fused kernel's order-preserving
// float -> uint packing (`pack_score`), so the peak's own score can be read
// off the packed argmax instead of copied again.
inline float ignis_attention_unpack_score(std::uint32_t packed) {
  const std::uint32_t bits = (packed & 0x80000000u) ? (packed & 0x7fffffffu) : ~packed;
  float score = 0.0F;
  static_assert(sizeof(score) == sizeof(bits), "a float is four bytes");
  std::memcpy(&score, &bits, sizeof(score));
  return score;
}

struct AttentionReadoutTarget {
  // The pointing head, when this layer holds it: its score row is written to
  // `device_scores`. -1 on a layer that holds only heads of the set.
  int32_t query_head = 0;
  // Absolute positions [key_begin, key_begin + key_count) of the keys read.
  int64_t key_begin = 0;
  int64_t key_count = 0;
  // Device F32 [key_count], allocated by the chunk and copied out by it:
  // one pre-softmax score per key, `q . k / sqrt(head_dim)`. Null on a layer
  // that holds only heads of the set.
  float *device_scores = nullptr;
  // GitHub #263: the heads of the set this layer holds (none: the single-head
  // kernel of #260, unchanged), each with its slot in `device_set_best` --
  // one packed (score, key index) per head of the whole set, zeroed by the
  // chunk before the first armed layer and raised with `atomicMax`, so the
  // largest packed value is the argmax (the larger index on a tie).
  int32_t set_heads = 0;
  int32_t set_query_head[kReadoutLayerHeads] = {};
  int32_t set_slot[kReadoutLayerHeads] = {};
  unsigned long long *device_set_best = nullptr;
  // GitHub #264: where this layer's heads' neighbour scores go -- four floats
  // a head of the *whole* set, at `4 * set_slot[i]`, NaN for a neighbour off
  // the image grid -- and that grid's columns, which say which exist. Both
  // are needed whenever `set_heads` is nonzero.
  float *device_set_neighbours = nullptr;
  int32_t grid_cols = 0;
  // Span-relative keys no head of the set may peak on. The pointing head's
  // row still covers them.
  int32_t excluded_count = 0;
  int32_t excluded[kReadoutMaxExcluded] = {};
  // Host flag the layer sets when it wrote every score and argmax it holds
  // from the keys its attention read. Left false when those keys were not
  // there to read.
  bool *read = nullptr;
};

// Score `target`'s span for the chunk whose attention just ran.
//
// `rotated_query` is the layer's BF16 [q_heads * 256, tokens] query after its
// norm and rotary embedding; the query is its column `tokens - 1`. The keys:
//
// - under hq-e8-2b (`cache.dtype == U8`), the prompt route's materialized
//   key plane at the base of `attention_workspace` -- BF16 [kv_heads][span]
//   [256] in the codec's rotated frame, rows at absolute positions (one band,
//   `span` = the band's row count) -- read only when `hq_prompt_scratch` says
//   the prompt route ran and materialized a single band covering the span.
//   The query is rotated into the same frame first (signs, a natural-order
//   256-point Walsh-Hadamard transform, 1/16: the codec's orthonormal
//   rotation), so the dot product is unchanged by the frame;
// - under BF16, the cache's own pages through `cache.block_tables` row 0,
//   which the layer's attention appended before it read them.
//
// A target with no heads of a set launches the single-head kernel (#260).
// One with heads of a set launches one fused kernel: a grid of key blocks by
// the four KV heads, each block rotating the armed query heads of its KV head
// once, each warp scoring one key row against all of them (the row is read
// once per layer, not once per head), each block publishing one `atomicMax`
// per head.
//
// Returns 0 and sets `*target.read` when it launched the scores; returns 0
// with `*target.read` untouched when the keys were not there to read; a
// negative value (with `error` set) for a layer whose head geometry is not
// the one it was built for, or a launch failure.
int32_t ignis_attention_readout_run(const AttentionReadoutTarget &target,
                                    const void *rotated_query, int32_t q_heads, int32_t kv_heads,
                                    int32_t tokens, int64_t visible_keys,
                                    const ninfer::PagedKVBatchLayerView &cache,
                                    bool hq_prompt_scratch, const void *attention_workspace,
                                    int64_t span, float scale, cudaStream_t stream,
                                    const char **error);
