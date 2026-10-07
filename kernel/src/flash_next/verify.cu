// ignis kernel leaf -- the Flash-Next verify round's device side (spec flash-next/07 phase C,
// GitHub #307; OURS, ADR 0043). See verify.h for the round and what each half does.
//
// The fold is ours -- our addressing of Flash-Next's per-layer GDN state (the vendored
// gdn_replay_fold admits only the 27B's all-layer view and registered layer counts), our records,
// the commit counts read from the device so the commit replays from a graph -- around the vendored
// recurrence's own per-token step (recurrent.cuh's recurrent_bf16_body in Fold mode). That step is
// the one the decode round's snapshot form and the pass's replay record run, so c folded columns
// leave each lane's state exactly as c one-token rounds over the same inputs would.

#include "verify.h"

#include "gdn.h"
#include "ignis_seq_internal.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/kernel/hq_codec.cuh"
// The header also defines one non-template kernel, which the vendored library compiles itself:
// renamed in this object so the two libraries do not both define it.
#define recurrent_fp32_kernel ignis_fn_verify_unused_recurrent_fp32_kernel
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"
#undef recurrent_fp32_kernel

#include <cuda_bf16.h>

#include <algorithm>
#include <exception>
#include <string>

namespace ignis::flash_next::verify {

namespace {

namespace gd = ninfer::ops::detail::gated_delta_net;

constexpr std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

constexpr int32_t kRingWords = static_cast<int32_t>(ninfer::ops::kGqaHqRecentKeys / 32);
constexpr int32_t kRowDim = 256;           // an hq side row: one KV head's head_dim
constexpr int32_t kMaxNgramColumns = 16;   // the n-gram conv's past columns a restore holds in registers
constexpr int32_t kThreads = 256;

// The hq side planes' addressing at Flash-Next's KV head count.
struct RingGeometry {
  static constexpr int KVHeads = 2;
};

// The widest round a load runs: its rows and its columns per lane, over the widths that draft.
uint32_t max_rows_of(uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget) {
  uint32_t rows = 0;
  for (uint32_t w = 1; w <= lanes; ++w) {
    const uint32_t k = window_for(draft_tokens, row_budget, w);
    if (k != 0) rows = std::max(rows, w * (k + 1));
  }
  return rows;
}
uint32_t max_columns_of(uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget) {
  uint32_t columns = 0;
  for (uint32_t w = 1; w <= lanes; ++w) {
    const uint32_t k = window_for(draft_tokens, row_budget, w);
    if (k != 0) columns = std::max(columns, k + 1);
  }
  return columns;
}
// The positions a lane's saved ring rows cover, at the widest window.
uint32_t max_ring_columns_of(uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget, bool mtp) {
  uint32_t columns = 0;
  for (uint32_t w = 1; w <= lanes; ++w) {
    const uint32_t k = window_for(draft_tokens, row_budget, w);
    if (k != 0) columns = std::max(columns, written_positions(k, mtp));
  }
  return columns;
}

// One GDN layer's records at `rows` rows, plane by plane.
struct GdnRecordLayout {
  std::size_t key = 0, value = 0, gate = 0, conv = 0;
  std::size_t layer() const { return key + value + gate + conv; }
};
GdnRecordLayout gdn_layout(uint32_t rows) {
  const auto r = static_cast<std::size_t>(rows);
  return {aligned(r * gdn::kKeyWidth * sizeof(__nv_bfloat16)), aligned(r * gdn::kValueWidth * sizeof(__nv_bfloat16)),
          aligned(r * gdn::kValueHeads * sizeof(uint2)), aligned(r * gdn::kConvChannels * sizeof(__nv_bfloat16))};
}

// What the pass saves, per lane: every attention layer's tail, the n-gram conv columns, the ring
// words and, per attention layer, column and role, the side rows the column's position overwrites.
struct SavedLayout {
  std::size_t tails = 0, ngram = 0, words = 0, rows = 0;
  int32_t tail_elements = 0;  // per (layer, lane)
  int32_t ngram_elements = 0; // per lane
  int32_t columns = 0;        // the rows section's columns per lane
  std::size_t bytes() const { return tails + ngram + words + rows; }
};
SavedLayout saved_layout(const Geometry &g, int32_t kv_format, uint32_t lanes, uint32_t columns,
                         int32_t attention_layers) {
  SavedLayout s;
  const auto n = static_cast<std::size_t>(lanes);
  s.tail_elements = (g.compress_ratio - 1) * g.indexer_kv_heads * g.indexer_head_dim;
  s.ngram_elements = g.ngram_conv_state_columns() * g.residual_width();
  s.columns = static_cast<int32_t>(columns);
  s.tails = aligned(static_cast<std::size_t>(attention_layers) * n * s.tail_elements * sizeof(__nv_bfloat16));
  s.ngram = aligned(n * s.ngram_elements * sizeof(__nv_bfloat16));
  if (kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    s.words = aligned(n * kRingWords * sizeof(uint32_t));
    s.rows = aligned(static_cast<std::size_t>(attention_layers) * n * columns * 2 * RingGeometry::KVHeads * kRowDim *
                     sizeof(__nv_bfloat16));
  }
  return s;
}

// ----------------------------------------------------------------------------------------------
// Save (the pass's first step).

struct SaveArgs {
  Sections s;
  const int32_t *slots;
  const int32_t *positions;
  __nv_bfloat16 *tails;
  __nv_bfloat16 *ngram;
  uint32_t *words;
  __nv_bfloat16 *rows;
  int32_t lanes;
  int32_t columns;      // the positions this width's round writes past the frontier
  int32_t row_columns;  // the rows section's columns per lane
  int32_t tail_elements;
  int32_t ngram_elements;
};

// One CTA per (attention layer, lane): the lane's tail keys.
__global__ void save_tails_kernel(SaveArgs a) {
  const int32_t layer = static_cast<int32_t>(blockIdx.x);
  const int32_t lane = static_cast<int32_t>(blockIdx.y);
  const auto *src = static_cast<const __nv_bfloat16 *>(a.s.tails[layer]) +
                    static_cast<int64_t>(a.slots[lane]) * a.s.tail_slot_elements;
  __nv_bfloat16 *dst = a.tails + (static_cast<int64_t>(layer) * a.lanes + lane) * a.tail_elements;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < a.tail_elements; i += blockDim.x) dst[i] = src[i];
}

// The lane's n-gram conv columns, 16 bytes per thread.
__global__ void save_ngram_kernel(SaveArgs a) {
  const int32_t lane = static_cast<int32_t>(blockIdx.y);
  const int64_t vectors = a.ngram_elements / 8;
  const auto *src = reinterpret_cast<const uint4 *>(static_cast<const __nv_bfloat16 *>(a.s.ngram_conv) +
                                                    static_cast<int64_t>(a.slots[lane]) * a.ngram_elements);
  auto *dst = reinterpret_cast<uint4 *>(a.ngram + static_cast<int64_t>(lane) * a.ngram_elements);
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < vectors;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    dst[i] = src[i];
  }
}

// One CTA per (attention layer, column, role) and lane, 32 threads per KV head: the side row the
// column's position names (its ring slot's current row, or its sink row), and the ring words.
__global__ void save_ring_kernel(SaveArgs a) {
  const int32_t unit = static_cast<int32_t>(blockIdx.x);
  const int32_t lane = static_cast<int32_t>(blockIdx.y);
  const int32_t slot = a.slots[lane];
  if (unit == 0 && threadIdx.x < kRingWords) {
    a.words[lane * kRingWords + threadIdx.x] = a.s.ring[static_cast<int64_t>(slot) * kRingWords + threadIdx.x];
  }
  const int32_t layer = unit / (a.columns * 2);
  const int32_t column = (unit / 2) % a.columns;
  const bool role_v = (unit & 1) != 0;
  const int32_t head = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t chunk = static_cast<int32_t>(threadIdx.x) % 32;
  const auto *plane = static_cast<const __nv_bfloat16 *>(role_v ? a.s.residual_v[layer] : a.s.residual_k[layer]);
  const __nv_bfloat16 *src =
      ninfer::ops::hq_residual_row<RingGeometry>(plane, slot, head, a.positions[lane] + column) + chunk * 8;
  __nv_bfloat16 *dst =
      a.rows +
      ((((static_cast<int64_t>(layer) * a.lanes + lane) * a.row_columns + column) * 2 + (role_v ? 1 : 0)) *
           RingGeometry::KVHeads +
       head) *
          kRowDim +
      chunk * 8;
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// ----------------------------------------------------------------------------------------------
// Restore (the commit's lane-state half).

struct RestoreArgs {
  Sections s;
  const int32_t *slots;
  const int32_t *positions;
  const int32_t *commit;
  const __nv_bfloat16 *tails;
  const __nv_bfloat16 *ngram;
  const uint32_t *words;
  const __nv_bfloat16 *rows;
  const __nv_bfloat16 *keys;  // the recorded raw indexer keys, layer-major
  int64_t key_layer_elements;
  int32_t key_dim;
  int32_t compress;
  int32_t lanes;
  int32_t columns;       // this width's k + 1: the recorded keys' columns per lane
  int32_t ring_columns;  // the ring-row columns restored per lane: the verify's written positions, the head's window + 1
  int32_t layer_base;    // the first attention section restored
  int32_t row_columns;
  int32_t tail_elements;
  int32_t ngram_columns;
  int32_t ngram_channels;
};

// One CTA per (attention layer, lane), one thread per key element: the tail of the new frontier
// f = p + c holds the raw keys of [f / compress * compress, f): the saved tail's below p, the
// call's recorded keys from p on.
__global__ void restore_tails_kernel(RestoreArgs a) {
  const int32_t layer = a.layer_base + static_cast<int32_t>(blockIdx.x);
  const int32_t lane = static_cast<int32_t>(blockIdx.y);
  const int32_t d = static_cast<int32_t>(threadIdx.x);
  if (d >= a.key_dim) return;
  const int32_t p = a.positions[lane];
  const int32_t f = p + a.commit[lane];
  const int32_t first_new = f / a.compress * a.compress;
  const int32_t first_old = p / a.compress * a.compress;
  const __nv_bfloat16 *saved = a.tails + (static_cast<int64_t>(layer) * a.lanes + lane) * a.tail_elements;
  const __nv_bfloat16 *keys = a.keys + layer * a.key_layer_elements;
  auto *tail = static_cast<__nv_bfloat16 *>(a.s.tails[layer]) + static_cast<int64_t>(a.slots[lane]) * a.s.tail_slot_elements;
  for (int32_t q = first_new; q < f; ++q) {
    const __nv_bfloat16 v = q < p ? saved[(q - first_old) * a.key_dim + d]
                                  : keys[(static_cast<int64_t>(lane) * a.columns + (q - p)) * a.key_dim + d];
    tail[(q - first_new) * a.key_dim + d] = v;
  }
}

// One thread per (lane, channel): the n-gram conv columns tail_n(saved || inputs[0, c)). The pass
// left tail_n(saved || inputs[0, columns)), whose last `columns` entries are the inputs (columns
// <= n), so input i is post[n - columns + i].
__global__ void restore_ngram_kernel(RestoreArgs a) {
  const int32_t lane = static_cast<int32_t>(blockIdx.y);
  const int32_t ch = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ch >= a.ngram_channels) return;
  const int32_t n = a.ngram_columns;
  const int32_t c = a.commit[lane];
  auto *state = static_cast<__nv_bfloat16 *>(a.s.ngram_conv) + static_cast<int64_t>(a.slots[lane]) * n * a.ngram_channels + ch;
  const __nv_bfloat16 *saved = a.ngram + static_cast<int64_t>(lane) * n * a.ngram_channels + ch;
  __nv_bfloat16 post[kMaxNgramColumns];
  for (int32_t m = 0; m < n; ++m) post[m] = state[static_cast<int64_t>(m) * a.ngram_channels];
  for (int32_t m = 0; m < n; ++m) {
    state[static_cast<int64_t>(m) * a.ngram_channels] =
        m + c < n ? saved[static_cast<int64_t>(m + c) * a.ngram_channels] : post[m + c - a.columns];
  }
}

// The side rows of every rejected column (j >= c) back, and the ring words: the saved words plus
// the committed positions' bits.
__global__ void restore_ring_kernel(RestoreArgs a) {
  const int32_t unit = static_cast<int32_t>(blockIdx.x);
  const int32_t lane = static_cast<int32_t>(blockIdx.y);
  const int32_t slot = a.slots[lane];
  const int32_t p = a.positions[lane];
  const int32_t c = a.commit[lane];
  if (unit == 0 && threadIdx.x < kRingWords) {
    const int32_t word = static_cast<int32_t>(threadIdx.x);
    uint32_t bits = a.words[lane * kRingWords + word];
    for (int32_t q = p; q < p + c; ++q) {
      if (q < static_cast<int32_t>(ninfer::ops::kGqaHqSinkKeys)) continue;
      const int32_t r = q & (static_cast<int32_t>(ninfer::ops::kGqaHqRecentKeys) - 1);
      if ((r >> 5) == word) bits |= 1U << (r & 31);
    }
    a.s.ring[static_cast<int64_t>(slot) * kRingWords + word] = bits;
  }
  const int32_t layer = a.layer_base + unit / (a.ring_columns * 2);
  const int32_t column = (unit / 2) % a.ring_columns;
  if (column < c) return;
  const bool role_v = (unit & 1) != 0;
  const int32_t head = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t chunk = static_cast<int32_t>(threadIdx.x) % 32;
  auto *plane = static_cast<__nv_bfloat16 *>(role_v ? a.s.residual_v[layer] : a.s.residual_k[layer]);
  __nv_bfloat16 *dst = ninfer::ops::hq_residual_row<RingGeometry>(plane, slot, head, p + column) + chunk * 8;
  const __nv_bfloat16 *src =
      a.rows +
      ((((static_cast<int64_t>(layer) * a.lanes + lane) * a.row_columns + column) * 2 + (role_v ? 1 : 0)) *
           RingGeometry::KVHeads +
       head) *
          kRowDim +
      chunk * 8;
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// ----------------------------------------------------------------------------------------------
// The fold (the commit's GDN half).

struct FoldArgs {
  float *recurrent[kMaxGdnLayers];
  __nv_bfloat16 *conv[kMaxGdnLayers];
  const __nv_bfloat16 *key;
  const __nv_bfloat16 *value;
  const uint2 *gate;
  const __nv_bfloat16 *conv_record;
  int64_t record_layer_bytes;
  const int32_t *slots;
  const int32_t *commit;
  int32_t width;  // columns per lane
};

// The vendored step's Access for Flash-Next: grid (value head, lane, layer * state tile), the
// lane's slot and commit count from the device, each GDN layer's own state and records.
struct FoldAccess {
  const FoldArgs *a;

  __device__ __forceinline__ gd::RecurrentCoordinates coordinates() const {
    const int32_t layer_tile = static_cast<int32_t>(blockIdx.z);
    const int32_t state_tile = layer_tile % (gd::kStateDim / gd::kBlockDv);
    const auto value_head = static_cast<uint32_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    return {lane,
            warp,
            static_cast<int32_t>(blockIdx.y),
            layer_tile / (gd::kStateDim / gd::kBlockDv),
            state_tile,
            value_head,
            value_head / static_cast<uint32_t>(gdn::kValueHeads / gdn::kQkHeads),
            static_cast<uint32_t>(state_tile * gd::kBlockDv + warp * gd::kDvPerWarp),
            static_cast<uint32_t>(lane * gd::kQkPerLane)};
  }

  __device__ __forceinline__ int32_t active_columns(const gd::RecurrentCoordinates &coord) const {
    return a->commit[coord.batch];
  }

  template <typename T>
  __device__ __forceinline__ const T *record(const void *plane, const gd::RecurrentCoordinates &coord) const {
    return reinterpret_cast<const T *>(static_cast<const unsigned char *>(plane) + coord.layer * a->record_layer_bytes);
  }

  __device__ __forceinline__ int64_t column(const gd::RecurrentCoordinates &coord, int32_t token) const {
    return static_cast<int64_t>(coord.batch) * a->width + token;
  }

  __device__ __forceinline__ float *state_read_base(const gd::RecurrentCoordinates &coord) const {
    constexpr int64_t kSlot = static_cast<int64_t>(gdn::kValueHeads) * gd::kStateDim * gd::kStateDim;
    return a->recurrent[coord.layer] + static_cast<int64_t>(a->slots[coord.batch]) * kSlot +
           static_cast<int64_t>(coord.value_head) * gd::kStateDim * gd::kStateDim;
  }

  __device__ __forceinline__ const __nv_bfloat16 *key_ptr(const gd::RecurrentCoordinates &coord, int32_t token) const {
    return record<__nv_bfloat16>(a->key, coord) + (column(coord, token) * gdn::kQkHeads + coord.qk_head) * gd::kStateDim;
  }

  __device__ __forceinline__ const __nv_bfloat16 *value_ptr(const gd::RecurrentCoordinates &coord,
                                                            int32_t token) const {
    return record<__nv_bfloat16>(a->value, coord) +
           (column(coord, token) * gdn::kValueHeads + coord.value_head) * gd::kStateDim;
  }

  __device__ __forceinline__ gd::RawGatePair load_gate(const gd::RecurrentCoordinates &coord, int32_t token) const {
    return gd::load_record_gate(record<uint2>(a->gate, coord), column(coord, token) * gdn::kValueHeads + coord.value_head);
  }

  __device__ __forceinline__ void store_final_state(const gd::RecurrentCoordinates &coord,
                                                    const float (&state)[gd::kDvPerWarp][gd::kQkPerLane]) const {
    float *destination = state_read_base(coord);
#pragma unroll
    for (int r = 0; r < gd::kDvPerWarp; ++r) {
      gd::store_qk_lane(state[r], destination + static_cast<int64_t>(coord.dv_base + r) * gd::kStateDim, coord.dqk_base);
    }
  }

  // The conv taps tail3(taps || inputs[0, commit)): one thread per channel over the first
  // kConvChannels / 128 (value head, state tile) CTAs of the layer.
  __device__ __forceinline__ void publish_final_conv_history(const gd::RecurrentCoordinates &coord,
                                                             int32_t commit) const {
    constexpr int32_t kC = gdn::kConvChannels;
    const int32_t tile_block = static_cast<int32_t>(coord.value_head) * (gd::kStateDim / gd::kBlockDv) + coord.state_tile;
    if (tile_block >= kC / 128) return;
    const int32_t channel = tile_block * 128 + coord.warp * ninfer::ops::kWarpSize + coord.lane;
    __nv_bfloat16 *taps = a->conv[coord.layer] + static_cast<int64_t>(a->slots[coord.batch]) * (gdn::kConvStateTaps * kC) + channel;
    const __nv_bfloat16 *inputs = record<__nv_bfloat16>(a->conv_record, coord) + column(coord, 0) * kC + channel;
    __nv_bfloat16 h0, h1, h2;
    if (commit == 1) {
      h0 = taps[kC];
      h1 = taps[2 * kC];
      h2 = inputs[0];
    } else if (commit == 2) {
      h0 = taps[2 * kC];
      h1 = inputs[0];
      h2 = inputs[kC];
    } else {
      h0 = inputs[static_cast<int64_t>(commit - 3) * kC];
      h1 = inputs[static_cast<int64_t>(commit - 2) * kC];
      h2 = inputs[static_cast<int64_t>(commit - 1) * kC];
    }
    taps[0] = h0;
    taps[kC] = h1;
    taps[2 * kC] = h2;
  }
};

static_assert(ninfer::ops::kWarpSize * gd::kNumWarps == 128, "one fold CTA spans 128 conv channels");
static_assert(gdn::kConvStateTaps == 3, "the fold's conv publish holds three taps");

__global__ void __launch_bounds__(ninfer::ops::kWarpSize *gd::kNumWarps, 2) fold_kernel(const __grid_constant__ FoldArgs args) {
  const FoldAccess access{&args};
  const gd::RecurrentCoordinates coord = access.coordinates();
  gd::recurrent_bf16_body<gd::RecurrentMode::Fold, true>(access, coord, args.width, access.active_columns(coord));
}

// What the restore kernels read, for sections [layer_base, ...) and ring rows over `ring_columns`
// positions past each lane's frontier.
RestoreArgs restore_args(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes,
                         uint32_t window, const int32_t *slots, const int32_t *positions, int32_t ring_columns,
                         int32_t layer_base) {
  const uint32_t row_columns = max_ring_columns_of(state.lanes, state.draft_tokens, state.row_budget, state.mtp);
  const SavedLayout layout = saved_layout(g, sections.ring != nullptr ? IGNIS_KV_FORMAT_HQ_E8_2B : IGNIS_KV_FORMAT_BF16,
                                          state.lanes, row_columns, sections.attention_layers);
  const auto *saved = static_cast<const unsigned char *>(state.saved->p);
  return RestoreArgs{sections,
                     slots,
                     positions,
                     static_cast<const int32_t *>(state.commit->p),
                     reinterpret_cast<const __nv_bfloat16 *>(saved),
                     reinterpret_cast<const __nv_bfloat16 *>(saved + layout.tails),
                     reinterpret_cast<const uint32_t *>(saved + layout.tails + layout.ngram),
                     reinterpret_cast<const __nv_bfloat16 *>(saved + layout.tails + layout.ngram + layout.words),
                     static_cast<const __nv_bfloat16 *>(state.records.indexer_keys),
                     static_cast<int64_t>(state.records.indexer_layer_bytes / sizeof(__nv_bfloat16)),
                     g.indexer_head_dim,
                     g.compress_ratio,
                     static_cast<int32_t>(lanes),
                     static_cast<int32_t>(window + 1),
                     ring_columns,
                     layer_base,
                     static_cast<int32_t>(row_columns),
                     layout.tail_elements,
                     sections.ngram_columns,
                     sections.ngram_channels};
}

bool launched(const char *what, std::string *error) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return true;
  *error = std::string(what) + ": " + cudaGetErrorString(err);
  return false;
}

// Whether a call of `lanes` lanes at window k fits what `state` was sized for -- its records'
// rows, its saved lanes and widest window's ring rows, its accept's drafts -- else *error.
bool fits(const State &state, uint32_t lanes, uint32_t window, const char *what, std::string *error) {
  const auto rows = static_cast<uint32_t>(state.records.rows);
  const uint32_t widest = state.window(1);
  if (lanes >= 1 && lanes <= state.lanes && window >= 1 && window <= widest && lanes * (window + 1) <= rows) {
    return true;
  }
  *error = std::string(what) + ": " + std::to_string(lanes) + " lanes x " + std::to_string(window + 1) +
           " columns overrun this load's round (1.." + std::to_string(state.lanes) + " lanes, a window of 1.." +
           std::to_string(widest) + ", " + std::to_string(rows) + " rows)";
  return false;
}

}  // namespace

uint32_t written_positions(uint32_t window, bool mtp) { return mtp ? 2 * window : window + 1; }

uint32_t window_for(uint32_t draft_tokens, uint32_t row_budget, uint32_t lanes) {
  if (draft_tokens == 0 || lanes == 0) return 0;
  const uint32_t budget = row_budget == 0 || row_budget > kMaxRows ? kMaxRows : row_budget;
  const uint32_t columns = budget / lanes;
  return columns <= 1 ? 0 : std::min(draft_tokens, columns - 1);
}

uint32_t decode_rows(uint32_t decode_lanes, uint32_t draft_tokens, uint32_t row_budget) {
  return std::max(decode_lanes, max_rows_of(decode_lanes, draft_tokens, row_budget));
}

Plan plan(const Geometry &g, int32_t kv_format, uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget,
          int32_t attention_layers, int32_t gdn_layers, bool mtp) {
  Plan p;
  const uint32_t rows = max_rows_of(lanes, draft_tokens, row_budget);
  const uint32_t ring_columns = max_ring_columns_of(lanes, draft_tokens, row_budget, mtp);
  if (rows == 0) return p;
  const std::size_t lane_i32 = aligned(static_cast<std::size_t>(lanes) * sizeof(int32_t));
  const std::size_t drafts = aligned(static_cast<std::size_t>(lanes) * draft_tokens * sizeof(int32_t));
  p.staging = 7 * lane_i32 + drafts + 2 * aligned(static_cast<std::size_t>(rows) * sizeof(int32_t));
  if (mtp) {
    // picks, chain tokens, chain stacks, chain positions, drafts out.
    p.staging += aligned(static_cast<std::size_t>(rows) * sizeof(int32_t)) + lane_i32 +
                 aligned(static_cast<std::size_t>(lanes) * g.residual_width() * 2) +
                 aligned(static_cast<std::size_t>(lanes) * std::max<uint32_t>(draft_tokens - 1, 1) * sizeof(int32_t)) +
                 drafts;
  }
  p.accept = aligned(ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                 g.vocab, 1, static_cast<int32_t>(draft_tokens), 1, static_cast<int32_t>(lanes))) +
             256;
  p.gdn_records = static_cast<std::size_t>(gdn_layers) * gdn_layout(rows).layer();
  p.indexer_records = static_cast<std::size_t>(attention_layers) *
                      aligned(static_cast<std::size_t>(rows) * g.indexer_kv_heads * g.indexer_head_dim * sizeof(__nv_bfloat16));
  p.saved = saved_layout(g, kv_format, lanes, ring_columns, attention_layers).bytes();
  return p;
}

std::size_t State::device_bytes() const {
  std::size_t bytes = accept_workspace != nullptr ? accept_workspace->capacity() : 0;
  for (const auto *buffer : {&valid_columns, &extents, &lengths, &anchors, &drafts, &commit, &target_tokens, &licensed,
                             &licensed_counts, &accepted, &gdn_records, &indexer_records, &saved, &picks,
                             &chain_tokens, &chain_stack, &chain_positions, &drafts_out}) {
    bytes += *buffer != nullptr ? (*buffer)->bytes : 0;
  }
  return bytes;
}

uint32_t State::ready_mask() const {
  uint32_t mask = 0;
  for (uint32_t w = 0; w < kMaxLanes; ++w) mask |= ready[w] ? 1U << w : 0U;
  return mask;
}

State::~State() {
  for (auto &exec : pass_exec) {
    if (exec != nullptr) cudaGraphExecDestroy(exec);
  }
  for (auto &exec : commit_exec) {
    if (exec != nullptr) cudaGraphExecDestroy(exec);
  }
}

std::string refusal(const Geometry &g, uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget,
                    int32_t attention_layers, int32_t gdn_layers) {
  if (draft_tokens == 0 || draft_tokens > kMaxWindow) {
    return "a draft window of " + std::to_string(draft_tokens) + " tokens (1.." + std::to_string(kMaxWindow) + ")";
  }
  if (lanes == 0 || lanes > kMaxLanes) {
    return std::to_string(lanes) + " decode lanes for the verify round (1.." + std::to_string(kMaxLanes) + ")";
  }
  if (row_budget > kMaxRows || window_for(draft_tokens, row_budget, 1) == 0) {
    return "a draft row budget of " + std::to_string(row_budget) + " (0 for the decode route's " +
           std::to_string(kMaxRows) + ", or 2.." + std::to_string(kMaxRows) + ")";
  }
  if (attention_layers <= 0 || attention_layers > kMaxAttentionLayers || gdn_layers <= 0 ||
      gdn_layers > kMaxGdnLayers) {
    return "the verify round addresses at most " + std::to_string(kMaxAttentionLayers) + " attention and " +
           std::to_string(kMaxGdnLayers) + " GDN layers";
  }
  if (g.ngram_conv_state_columns() > kMaxNgramColumns ||
      max_columns_of(lanes, draft_tokens, row_budget) > static_cast<uint32_t>(g.ngram_conv_state_columns())) {
    return "the n-gram conv keeps " + std::to_string(g.ngram_conv_state_columns()) +
           " past columns: a verify round's columns must fit in them, and they in the restore's " +
           std::to_string(kMaxNgramColumns);
  }
  if (g.indexer_kv_heads != 1 || g.compress_ratio < 2) {
    return "the verify round's tail restore is written for one indexer key head and blocks of 2 or more tokens";
  }
  return {};
}

std::unique_ptr<State> create(const Geometry &g, int32_t kv_format, uint32_t lanes, uint32_t draft_tokens,
                              uint32_t row_budget, int32_t attention_layers, int32_t gdn_layers, bool mtp,
                              std::string *error) {
  if (std::string why = refusal(g, lanes, draft_tokens, row_budget, attention_layers, gdn_layers);
      !why.empty()) {
    *error = why;
    return nullptr;
  }
  auto st = std::make_unique<State>();
  st->draft_tokens = draft_tokens;
  st->row_budget = row_budget;
  st->lanes = lanes;
  st->attention_layers = attention_layers;
  st->gdn_layers = gdn_layers;
  st->mtp = mtp;
  st->sizes = plan(g, kv_format, lanes, draft_tokens, row_budget, attention_layers, gdn_layers, mtp);
  const uint32_t rows = max_rows_of(lanes, draft_tokens, row_budget);
  try {
    const auto buffer = [](std::size_t bytes) { return std::make_unique<ninfer::DeviceBuffer>(aligned(bytes)); };
    const std::size_t lane_i32 = static_cast<std::size_t>(lanes) * sizeof(int32_t);
    st->valid_columns = buffer(lane_i32);
    st->extents = buffer(lane_i32);
    st->lengths = buffer(lane_i32);
    st->anchors = buffer(lane_i32);
    st->commit = buffer(lane_i32);
    st->licensed_counts = buffer(lane_i32);
    st->accepted = buffer(lane_i32);
    st->drafts = buffer(static_cast<std::size_t>(lanes) * draft_tokens * sizeof(int32_t));
    st->target_tokens = buffer(static_cast<std::size_t>(rows) * sizeof(int32_t));
    st->licensed = buffer(static_cast<std::size_t>(rows) * sizeof(int32_t));
    st->accept_workspace = std::make_unique<ninfer::DeviceArena>(st->sizes.accept);
    st->gdn_records = buffer(st->sizes.gdn_records);
    st->indexer_records = buffer(st->sizes.indexer_records);
    st->saved = buffer(st->sizes.saved);
    if (mtp) {
      st->picks = buffer(static_cast<std::size_t>(rows) * sizeof(int32_t));
      st->chain_tokens = buffer(lane_i32);
      st->chain_stack = buffer(static_cast<std::size_t>(lanes) * g.residual_width() * 2);
      st->chain_positions = buffer(static_cast<std::size_t>(lanes) * std::max<uint32_t>(draft_tokens - 1, 1) *
                                   sizeof(int32_t));
      st->drafts_out = buffer(static_cast<std::size_t>(lanes) * draft_tokens * sizeof(int32_t));
    }
  } catch (const std::exception &e) {
    *error = std::string("the verify round's reservations: ") + e.what();
    return nullptr;
  }
  const GdnRecordLayout layout = gdn_layout(rows);
  auto *records = static_cast<unsigned char *>(st->gdn_records->p);
  st->records.valid_columns = static_cast<const int32_t *>(st->valid_columns->p);
  st->records.rows = static_cast<int32_t>(rows);
  st->records.gdn_key = records;
  st->records.gdn_value = records + layout.key;
  st->records.gdn_gate = records + layout.key + layout.value;
  st->records.gdn_conv = records + layout.key + layout.value + layout.gate;
  st->records.gdn_layer_bytes = layout.layer();
  st->records.indexer_keys = st->indexer_records->p;
  st->records.indexer_layer_bytes = st->sizes.indexer_records / static_cast<std::size_t>(attention_layers);
  return st;
}

bool sections_of(const ignis_seq_pool &pool, const Geometry &g, int32_t gdn_layers, Sections *out,
                 std::string *error) {
  Sections s;
  if (!pool.has_indexer() || !pool.has_ngram_conv()) {
    *error = "the pool has no indexer or n-gram conv section";
    return false;
  }
  if (pool.kv_num_layers > kMaxAttentionLayers || gdn_layers > kMaxGdnLayers) {
    *error = "the pool has more layers than the verify round addresses";
    return false;
  }
  s.attention_layers = pool.kv_num_layers;
  s.gdn_layers = gdn_layers;
  s.tail_slot_elements = static_cast<int64_t>(pool.indexer_tail_slot_bytes / sizeof(__nv_bfloat16));
  for (int32_t a = 0; a < s.attention_layers; ++a) {
    s.tails[a] = pool.indexer_tail_keys(a);
    if (pool.has_hq_residual()) {
      s.residual_k[a] = pool.hq_residual_plane(false, a, 0);
      s.residual_v[a] = pool.hq_residual_plane(true, a, 0);
    }
  }
  if (pool.has_hq_residual()) s.ring = pool.hq_ring_words(0);
  s.ngram_conv = pool.ngram_conv->p;
  s.ngram_channels = g.residual_width();
  s.ngram_columns = g.ngram_conv_state_columns();
  if (pool.ngram_conv_slot_bytes != static_cast<std::uint64_t>(s.ngram_columns) * s.ngram_channels * 2) {
    *error = "the pool's n-gram conv slots are not this model's";
    return false;
  }
  try {
    for (int32_t l = 0; l < gdn_layers; ++l) {
      const auto layer = static_cast<std::uint32_t>(l);
      s.recurrent[l] = static_cast<float *>(pool.gdn_pool.recurrent_slot(layer, 0).data);
      s.conv[l] = pool.gdn_pool.conv_slot(layer, 0).data;
    }
  } catch (const std::exception &e) {
    *error = std::string("the pool's GDN state: ") + e.what();
    return false;
  }
  *out = s;
  return true;
}

int32_t save(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes, uint32_t window,
             const int32_t *slots, const int32_t *positions, cudaStream_t stream, std::string *error) {
  if (!fits(state, lanes, window, "verify save", error)) return -1;
  const uint32_t row_columns = max_ring_columns_of(state.lanes, state.draft_tokens, state.row_budget, state.mtp);
  const SavedLayout layout = saved_layout(g, sections.ring != nullptr ? IGNIS_KV_FORMAT_HQ_E8_2B : IGNIS_KV_FORMAT_BF16,
                                          state.lanes, row_columns, sections.attention_layers);
  const auto ring_columns = static_cast<int32_t>(written_positions(window, state.mtp));
  auto *saved = static_cast<unsigned char *>(state.saved->p);
  SaveArgs a{sections,
             slots,
             positions,
             reinterpret_cast<__nv_bfloat16 *>(saved),
             reinterpret_cast<__nv_bfloat16 *>(saved + layout.tails),
             reinterpret_cast<uint32_t *>(saved + layout.tails + layout.ngram),
             reinterpret_cast<__nv_bfloat16 *>(saved + layout.tails + layout.ngram + layout.words),
             static_cast<int32_t>(lanes),
             ring_columns,
             static_cast<int32_t>(row_columns),
             layout.tail_elements,
             layout.ngram_elements};
  const auto n = static_cast<unsigned>(lanes);
  save_tails_kernel<<<dim3(static_cast<unsigned>(sections.attention_layers), n), kThreads, 0, stream>>>(a);
  if (!launched("verify save: tails", error)) return -1;
  save_ngram_kernel<<<dim3(32, n), kThreads, 0, stream>>>(a);
  if (!launched("verify save: n-gram conv", error)) return -1;
  if (sections.ring != nullptr) {
    const auto units = static_cast<unsigned>(sections.attention_layers * ring_columns * 2);
    save_ring_kernel<<<dim3(units, n), RingGeometry::KVHeads * 32, 0, stream>>>(a);
    if (!launched("verify save: hq ring", error)) return -1;
  }
  return 0;
}

int32_t accept(State &state, const Geometry &g, uint32_t lanes, uint32_t window, const void *logits,
               const void *configs, cudaStream_t stream, std::string *error) {
  if (!fits(state, lanes, window, "verify accept", error)) return -1;
  const auto b = static_cast<int32_t>(lanes);
  const auto t = static_cast<int32_t>(window + 1);
  const auto k = static_cast<int32_t>(window);
  try {
    void *rows = const_cast<void *>(logits);
    const ninfer::Tensor all(rows, ninfer::DType::BF16, {g.vocab, b * t, 1, 1});
    ninfer::Tensor targets(state.target_tokens->p, ninfer::DType::I32, {b * t, 1, 1, 1});
    ninfer::ops::argmax(all, targets, g.vocab, stream);
    const ninfer::Tensor targets_rows(state.target_tokens->p, ninfer::DType::I32, {t, b, 1, 1});
    const ninfer::Tensor logits_rows(rows, ninfer::DType::BF16, {g.vocab, t, b, 1});
    const ninfer::Tensor drafts(state.drafts->p, ninfer::DType::I32, {k, b, 1, 1});
    const ninfer::Tensor extents(state.extents->p, ninfer::DType::I32, {b, 1, 1, 1});
    ninfer::Tensor lengths(state.lengths->p, ninfer::DType::I32, {b, 1, 1, 1});
    ninfer::Tensor anchors(state.anchors->p, ninfer::DType::I32, {b, 1, 1, 1});
    ninfer::Tensor licensed(state.licensed->p, ninfer::DType::I32, {t, b, 1, 1});
    ninfer::Tensor counts(state.licensed_counts->p, ninfer::DType::I32, {b, 1, 1, 1});
    ninfer::Tensor accepted(state.accepted->p, ninfer::DType::I32, {b, 1, 1, 1});
    ninfer::DeviceArena::Scope scope = state.accept_workspace->scope();
    ninfer::ops::speculative_accept_greedy_drafts(targets_rows, logits_rows, drafts, extents, lengths, anchors,
                                                  licensed, counts, accepted, g.vocab,
                                                  static_cast<const ninfer::ops::SamplingConfig *>(configs),
                                                  *state.accept_workspace, stream);
  } catch (const std::exception &e) {
    *error = std::string("verify accept: ") + e.what();
    return -1;
  }
  return 0;
}

int32_t fold(const State &state, const Sections &sections, uint32_t lanes, uint32_t window, const int32_t *slots,
             cudaStream_t stream, std::string *error) {
  if (!fits(state, lanes, window, "verify commit: GDN fold", error)) return -1;
  FoldArgs f{};
  for (int32_t l = 0; l < sections.gdn_layers; ++l) {
    f.recurrent[l] = sections.recurrent[l];
    f.conv[l] = static_cast<__nv_bfloat16 *>(sections.conv[l]);
  }
  f.key = static_cast<const __nv_bfloat16 *>(state.records.gdn_key);
  f.value = static_cast<const __nv_bfloat16 *>(state.records.gdn_value);
  f.gate = static_cast<const uint2 *>(state.records.gdn_gate);
  f.conv_record = static_cast<const __nv_bfloat16 *>(state.records.gdn_conv);
  f.record_layer_bytes = static_cast<int64_t>(state.records.gdn_layer_bytes);
  f.slots = slots;
  f.commit = static_cast<const int32_t *>(state.commit->p);
  f.width = static_cast<int32_t>(window + 1);
  const dim3 fold_grid(static_cast<unsigned>(gdn::kValueHeads), static_cast<unsigned>(lanes),
                       static_cast<unsigned>(sections.gdn_layers * (gd::kStateDim / gd::kBlockDv)));
  fold_kernel<<<fold_grid, dim3(ninfer::ops::kWarpSize, gd::kNumWarps, 1), 0, stream>>>(f);
  return launched("verify commit: GDN fold", error) ? 0 : -1;
}

int32_t restore(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes, uint32_t window,
                const int32_t *slots, const int32_t *positions, cudaStream_t stream, std::string *error) {
  if (!fits(state, lanes, window, "verify commit: restore", error)) return -1;
  const auto n = static_cast<unsigned>(lanes);
  const auto ring_columns = static_cast<int32_t>(written_positions(window, state.mtp));
  const RestoreArgs r = restore_args(state, sections, g, lanes, window, slots, positions, ring_columns, 0);
  restore_tails_kernel<<<dim3(static_cast<unsigned>(sections.attention_layers), n), static_cast<unsigned>(g.indexer_head_dim), 0,
                         stream>>>(r);
  if (!launched("verify commit: indexer tails", error)) return -1;
  restore_ngram_kernel<<<dim3(static_cast<unsigned>((sections.ngram_channels + kThreads - 1) / kThreads), n), kThreads, 0,
                         stream>>>(r);
  if (!launched("verify commit: n-gram conv", error)) return -1;
  if (sections.ring != nullptr) {
    const auto units = static_cast<unsigned>(sections.attention_layers * ring_columns * 2);
    restore_ring_kernel<<<dim3(units, n), RingGeometry::KVHeads * 32, 0, stream>>>(r);
    if (!launched("verify commit: hq ring", error)) return -1;
  }
  return 0;
}

int32_t restore_head(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes, uint32_t window,
                     const int32_t *slots, const int32_t *positions, cudaStream_t stream, std::string *error) {
  if (!state.mtp) {
    *error = "verify restore_head: this load has no MTP head";
    return -1;
  }
  if (!fits(state, lanes, window, "verify restore_head", error)) return -1;
  const auto n = static_cast<unsigned>(lanes);
  const auto columns = static_cast<int32_t>(window + 1);
  const RestoreArgs r = restore_args(state, sections, g, lanes, window, slots, positions, columns,
                                     sections.attention_layers - 1);
  restore_tails_kernel<<<dim3(1, n), static_cast<unsigned>(g.indexer_head_dim), 0, stream>>>(r);
  if (!launched("verify drafting: the head's indexer tail", error)) return -1;
  if (sections.ring != nullptr) {
    restore_ring_kernel<<<dim3(static_cast<unsigned>(columns * 2), n), RingGeometry::KVHeads * 32, 0, stream>>>(r);
    if (!launched("verify drafting: the head's hq ring", error)) return -1;
  }
  return 0;
}

}  // namespace ignis::flash_next::verify
