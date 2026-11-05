// The hq-e8-2b prompt attention without prompt planes (ADR 0061, spec 27b-16gb/03, GitHub
// #335): an ADAPTED op (ADR 0051), kernel/src/hq_prompt_attention.cu.
//
// What it replaces: the vendored A1 Prompt route under hq-e8-2b, which decodes the layer's whole
// visible history into BF16 prompt planes (4,096 bytes per context token, up to 262,144 keys)
// before its FA2 kernel attends. This op reads the history in two passes, both on the tensor
// cores, and its workspace stops growing with the context at one band:
//   - the residual window -- the 32 sinks, the 512-key ring before the chunk and the chunk
//     itself -- in BF16, its rows exact as before (the ring slots a rollback cleared through the
//     codec, as before);
//   - every other key, which only the codec holds, decoded once per chunk and layer into a fixed
//     band of `band_keys` keys in 8 bits (2,048 bytes per key: K in INT8, V in E4M3) and
//     attended on the INT8 MMA (Q K^T) and the block-scaled FP8 MMA (P V), band after band.
// Without the residual window every key takes the FP8 pass. The integer band (kInt8 below) runs
// both passes on one 8-bit kernel of its own.
//
// The call is A1's for one sequence (B = 1) on an hq-e8-2b cache, with the same order of append
// and attention: with the residual window on, the chunk attends first (its own rows exact from
// `k`/`v`) and is appended after (ADR 0037, GitHub #258); without it the chunk is appended first
// and read back through the codec. The output takes A1's sigmoid gate.
//
// Not part of the flat C ABI: `run_gqa_layer` (kernel/src/gqa_layer.cu) and the leaf's own tests
// call it.
#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

// The keys one band holds: 2,048 bytes each (INT8 K and E4M3 V over 4 KV heads), so 64 MiB.
inline constexpr std::uint32_t kIgnisHqPromptBandKeys = 32768;
// The most key splits a launch runs; each keeps its own online-softmax state across bands.
inline constexpr std::int32_t kIgnisHqPromptMaxSplits = 5;
// The keys one tile holds: the band, the splits and the scales are cut at this grain.
inline constexpr std::uint32_t kIgnisHqPromptTileKeys = 32;

struct IgnisHqPromptKeys;

// How the band pass multiplies (spec 27b-16gb/03 Solution 6): in 8 bits, the default -- Q K^T on
// the INT8 MMA, exact in int32, P V in E4M3 on the block-scaled FP8 MMA, sm_120a's; or the
// fallback, FP16 operands with FP16 accumulate inside each 32-key tile, folded into FP32 -- its
// band holds half the keys in the same bytes. The window pass is BF16 under both.
//
// kInt8, the integer band (GitHub #510, spec #503): the 8-bit band on integer MMAs alone, which
// every sm_80 card runs at full rate, and the window pass with it. Q K^T as kFp8's; P V on the
// u8 x s8 MMA, P = rn(255 p), V^T in s8 with one amax / 127 scale per 128-key group and KV head,
// exact in int32 across the group and folded into FP32 once a group, O rescaled only when a row's
// max grows past 2^8. Its key splits, unless `splits` names them: 2 up to 16,384 visible keys,
// 3 above. The residual window's rows stay BF16 in the cache for decode: only this pass rounds
// them to 8 bits.
enum class IgnisHqPromptPrecision { kFp8, kFp16, kInt8 };

// Where one call's split states are, in its workspace, after the passes and the merge: the state
// each split carried from pass to pass. Device pointers, valid until the workspace's next use.
struct IgnisHqPromptCarry {
  const void *o = nullptr;   // FP16 [split][token][24][256], rotated frame, O over the split's l
  const float *m = nullptr;  // [split][token][24], log2 units of the scaled score
  const float *l = nullptr;  // [split][token][24], in units of 256 p at m
  std::int32_t splits = 0;
  std::int32_t width = 0;
};

struct IgnisHqPromptOptions {
  // Keys per band, a multiple of `kIgnisHqPromptTileKeys`. Tests narrow it to walk several
  // bands at a small context; the engine keeps the default.
  std::uint32_t band_keys = kIgnisHqPromptBandKeys;
  // Key splits, 1..kIgnisHqPromptMaxSplits; 0 lets the op pick them (kInt8: its own rule).
  std::int32_t splits = 0;
  IgnisHqPromptPrecision precision = IgnisHqPromptPrecision::kFp8;
  // Runs on the host after the attention is enqueued and before an attend-first append, while
  // every key the attention read is still where it read it, and is told where they are (the
  // attention readout, GitHub #260, and the test-only tap).
  std::function<void(const IgnisHqPromptKeys &)> after_attention;
  // Runs on the host after the merge is enqueued, told where the split states are: the leaf's
  // tests read the carried state there. Unset in the engine.
  std::function<void(const IgnisHqPromptCarry &)> after_merge;
};

// Where one chunk's keys are, as the attention reads them: the codec planes, the residual window
// (null when it is off) and, when the chunk attends before its append, the chunk's own rows.
// Plain pointers, passed by value into kernels; `ignis_hq_prompt_keys` fills it from the view.
struct IgnisHqPromptKeys {
  const void *codes_k = nullptr;
  const void *codes_v = nullptr;
  const void *meta_k = nullptr;
  const void *meta_v = nullptr;
  const std::int32_t *block_table = nullptr;
  const void *residual_k = nullptr;
  const void *residual_v = nullptr;
  const std::uint32_t *ring_valid = nullptr;
  const void *new_k = nullptr;  // BF16 [256, 4, width], or null
  const void *new_v = nullptr;
  std::int32_t start = 0;  // the chunk's first cache position
  std::int32_t width = 0;
};

// The keys of the chunk `k`/`v` at `start_position` over `cache`. The chunk's own rows are read
// from `k`/`v` exactly when the view carries the residual window, which is when the chunk
// attends before its append.
IgnisHqPromptKeys ignis_hq_prompt_keys(const ninfer::PagedKVLayerView &cache,
                                       const ninfer::Tensor &k, const ninfer::Tensor &v,
                                       std::int32_t start_position);

// Writes the rows of keys [key_begin, key_begin + count) -- K, or V with `values` -- into `dst`,
// BF16 [4 KV heads][count][256] in the codec's rotated frame: the values the attention reads
// before its band rounds them to 8 bits. Enqueued on `stream`; call it before an attend-first
// append, while the ring still holds the keys before the chunk.
void ignis_hq_prompt_fetch_rows(const IgnisHqPromptKeys &keys, bool values,
                                std::int32_t key_begin, std::int32_t count, void *dst,
                                cudaStream_t stream);

// The transient arena one call of `width` tokens bumps when `visible_keys` keys (the chunk's
// included) are visible, in bytes, under either precision. It depends on the context only up to
// one band.
std::size_t ignis_hq_prompt_workspace_bytes(std::int32_t width, std::uint32_t visible_keys,
                                            std::uint32_t band_keys = kIgnisHqPromptBandKeys);

// A1 at B = 1 on an hq-e8-2b cache: appends `k`/`v` at `positions`, attends `q` over keys
// [0, start_position + T), and writes `out` = attention * sigmoid(gate).
//
// q, gate, out: BF16 [256, 24, T]; k, v: BF16 [256, 4, T]; positions: I32 [T], holding
// start_position .. start_position + T - 1; `cache`: one sequence's layer view
// (ignis_kv_layer_view). `workspace` must hold `ignis_hq_prompt_workspace_bytes(T,
// start_position + T, options.band_keys)`. Throws std::invalid_argument on a shape or format it
// does not serve.
void ignis_hq_prompt_attention(const ninfer::Tensor &q, const ninfer::Tensor &k,
                               const ninfer::Tensor &v, const ninfer::Tensor &positions,
                               std::int32_t start_position, const ninfer::Tensor &gate,
                               float scale, const ninfer::PagedKVLayerView &cache,
                               ninfer::WorkspaceArena &workspace, ninfer::Tensor &out,
                               cudaStream_t stream, const IgnisHqPromptOptions &options = {});
