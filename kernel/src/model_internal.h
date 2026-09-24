// ignis kernel leaf - the `ignis_model` handle's real definition, shared
// between model.cu (P1-17, GitHub #53: load / stats / free) and step.cu
// (P1-18, GitHub #54: the degenerate embed -> norm -> head -> argmax
// program). Never exposed across the C ABI (kernel/include/ignis_model.h
// keeps `struct ignis_model;` opaque) -- this header is leaf-internal.

#pragma once

#include "ignis_model.h"
#include "ignis_step.h"

#include "rope_scaling.h"

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// One decoder layer's leaf-crossing weights (the reference's per-layer
// weight struct shape, e.g. targets/qwen3_6 FullLayerW / GdnLayerW --
// written by us: ADR 0009, the program layer is ours, not vendored).
struct GqaLayerWeights {
  ninfer::Weight input_norm;
  ninfer::Weight query_key_gate_value;
  ninfer::Weight query_norm;
  ninfer::Weight key_norm;
  ninfer::Weight output;
  ninfer::Weight post_attention_norm;
  ninfer::Weight mlp_gate_up;
  ninfer::Weight mlp_down;
};

struct GdnLayerWeights {
  ninfer::Weight input_norm;
  ninfer::Weight a_log;
  ninfer::Weight dt_bias;
  ninfer::Weight convolution;
  ninfer::Weight a_b_projection;
  ninfer::Weight query_key_value_z;
  ninfer::Weight norm;
  ninfer::Weight output;
  ninfer::Weight post_attention_norm;
  ninfer::Weight mlp_gate_up;
  ninfer::Weight mlp_down;
};

// P5-02 (GitHub #150): the DFlash2 drafter's weights, bound only under
// IGNIS_SPECULATIVE_DFLASH2. Each attention and MLP block carries its own
// two-tap dynamic conv (a base kernel and its projection).
struct Dflash2LayerWeights {
  ninfer::Weight input_norm;
  ninfer::Weight query_key_value;
  ninfer::Weight query_norm;
  ninfer::Weight key_norm;
  ninfer::Weight output;
  ninfer::Weight attention_conv_base;
  ninfer::Weight attention_conv_proj;
  ninfer::Weight post_attention_norm;
  ninfer::Weight mlp_gate_up;
  ninfer::Weight mlp_down;
  ninfer::Weight mlp_conv_base;
  ninfer::Weight mlp_conv_proj;
};

inline constexpr std::size_t kDflash2Layers = 5;

// P5-03 (GitHub #152): the target layers whose outputs the drafter consumes,
// in the order their features are concatenated (5 x 5120 = the
// `feature_projection` input width), and the constants the context append
// (kernel/src/step.cu) projects them with -- the reference's `DFlash2Config`.
inline constexpr std::array<std::uint32_t, 5> kDflash2TapLayers{5, 19, 33, 47, 61};
inline constexpr std::int64_t kDflash2QuerySize = 4096; // 32 query heads x 128
inline constexpr float kDflash2RmsEps          = 1.0e-6F;
inline constexpr float kDflash2RopeTheta       = 1.0e7F;
// P5-05 (GitHub #155): the drafter's forward (kernel/src/dflash2_drafter.cu)
// -- the mask token its query block pads with, the per-column candidates its
// selector scores, and its attention scale (1/sqrt(128), the one the
// vendored `swa` admits).
inline constexpr std::int32_t kDflash2MaskToken    = 248070;
inline constexpr std::int32_t kDflash2SelectorTopK = 16;
inline constexpr float kDflash2AttentionScale      = 0.08838834764831845F;

struct Dflash2Weights {
  ninfer::Weight feature_projection;
  ninfer::Weight context_norm;
  std::array<Dflash2LayerWeights, kDflash2Layers> layers{};
  ninfer::Weight final_norm;
  ninfer::Weight selector_hidden;
  ninfer::Weight selector_predecessor;
  ninfer::Weight selector_successor;
};

// GitHub #177: the vision tower's geometry (the reference's
// `VisionBackboneConfig`, `targets/qwen3_6/export/.../vision.h`) and its
// weights, named after the artifact's `vision/*` objects.
inline constexpr std::int32_t kVisionLayers = 27;
inline constexpr std::int32_t kVisionHidden = 1152;
inline constexpr std::int32_t kVisionIntermediate = 4304;
inline constexpr std::int32_t kVisionHeads = 16;
inline constexpr std::int32_t kVisionPatchDim = 3 * 2 * 16 * 16;
inline constexpr std::int32_t kVisionMergeUnit = 4;
inline constexpr std::int32_t kVisionMergerHidden = kVisionHidden * kVisionMergeUnit;
inline constexpr std::int32_t kVisionPositionEmbeddings = 48 * 48;
// The frontend's per-request segment bound the workspace is sized for (the
// reference's `kFrontendSegmentLimit`, 768 / 2).
inline constexpr std::int32_t kVisionMaxSegments = 768 / 2;

struct VisionLayerWeights {
  ninfer::Weight qkv;
  ninfer::Weight qkv_bias;
  ninfer::Weight output;
  ninfer::Weight output_bias;
  ninfer::Weight fc1;
  ninfer::Weight fc1_bias;
  ninfer::Weight fc2;
  ninfer::Weight fc2_bias;
  ninfer::Weight norm1_weight;
  ninfer::Weight norm1_bias;
  ninfer::Weight norm2_weight;
  ninfer::Weight norm2_bias;
};

struct VisionWeights {
  ninfer::Weight patch_embedding;
  ninfer::Weight patch_embedding_bias;
  ninfer::Weight position_embedding;
  std::array<VisionLayerWeights, kVisionLayers> layers{};
  ninfer::Weight merger_fc1;
  ninfer::Weight merger_fc1_bias;
  ninfer::Weight merger_fc2;
  ninfer::Weight merger_fc2_bias;
  ninfer::Weight merger_norm_weight;
  ninfer::Weight merger_norm_bias;
};

struct ignis_media_embedding;

// GitHub #243: the media embedding pool -- one device reservation carved
// into fixed-width column pages.
//
// Paging rather than slots because an embedding's width spans two orders of
// magnitude: at 5120 hidden a 320x240 thumbnail is 80 columns (one page) and
// a 4096x4096 screenshot is 16,384 (128 pages). Slots wide enough for the
// second waste almost all of themselves on the first, and the whole point of
// holding embeddings past their encode is to hold *several*.
//
// The page width is the one thing the rest of the leaf has to agree on: the
// encoder writes an item's columns page by page (kernel/src/vision_encode.cu)
// and a prefill chunk scatters them back page by page
// (kernel/src/step.cu). Both are plain column ranges over the ops that
// already existed -- no kernel reads a page table.
struct VisionEmbeddingPool {
  std::unique_ptr<ninfer::DeviceBuffer> buffer;
  // `[hidden]` BF16 columns per page, and so the bytes of one.
  std::int32_t page_columns = 0;
  std::size_t page_bytes = 0;
  // Per page: the embedding holding it, or nullptr. A raw pointer and not a
  // bool because a release names its embedding, not its pages.
  std::vector<const ignis_media_embedding *> owner;

  bool present() const { return buffer != nullptr; }
  std::int32_t pages() const { return static_cast<std::int32_t>(owner.size()); }
  std::int32_t free_pages() const {
    std::int32_t n = 0;
    for (const ignis_media_embedding *o : owner) {
      n += (o == nullptr) ? 1 : 0;
    }
    return n;
  }
  // The pages `columns` merged columns need.
  std::int32_t pages_for(std::int32_t columns) const {
    return page_columns > 0 ? (columns + page_columns - 1) / page_columns : 0;
  }
  std::uint8_t *page_ptr(std::int32_t page) const {
    return static_cast<std::uint8_t *>(buffer->p) + static_cast<std::size_t>(page) * page_bytes;
  }
  // Claim `count` pages for `owner_of`, or return empty. The pages need not
  // be adjacent -- that is the whole point of paging them.
  std::vector<std::int32_t> take(std::int32_t count, const ignis_media_embedding *owner_of) {
    std::vector<std::int32_t> taken;
    if (count > free_pages()) {
      return taken;
    }
    taken.reserve(static_cast<std::size_t>(count));
    for (std::int32_t page = 0; page < pages() && taken.size() < static_cast<std::size_t>(count);
         ++page) {
      if (owner[static_cast<std::size_t>(page)] == nullptr) {
        owner[static_cast<std::size_t>(page)] = owner_of;
        taken.push_back(page);
      }
    }
    return taken;
  }
  void give(const std::vector<std::int32_t> &pages_held) {
    for (const std::int32_t page : pages_held) {
      owner[static_cast<std::size_t>(page)] = nullptr;
    }
  }
};

struct LayerWeights {
  ignis_layer_kind kind = IGNIS_LAYER_GDN;
  GqaLayerWeights gqa{};
  GdnLayerWeights gdn{};
};

// P5-04 (GitHub #153): the verify round's substrate, built at load when the
// options fixed a draft window (`draft_tokens > 0`, under DFLASH2 or
// VERIFY_ONLY) and absent otherwise, so a load without a window reserves
// nothing here.
//
// Everything below is a stable device address the verify graphs bake in
// (ADR 0019 / 0020): the round refreshes the *_staging buffers in place by
// one H2D copy each before a replay, the traversal writes the rest. `B` is
// `IGNIS_DECODE_MAX_BATCH`; `k` is the window; a `[k+1, B]` matrix is
// contiguous with column `b`'s `k+1` entries adjacent, which is both the
// vendored `speculative_*` ops' `[K+1,B]` and the traversal's lane-major
// column order (lane b's columns are `b*(k+1) .. b*(k+1)+k`).
struct IgnisVerifyRound {
  uint32_t window = 0; // k

  // Host-refreshed per round.
  std::unique_ptr<ninfer::DeviceBuffer> anchors;        // I32 [B]: each lane's pending token
  std::unique_ptr<ninfer::DeviceBuffer> drafts;         // I32 [k, B]: the proposals
  std::unique_ptr<ninfer::DeviceBuffer> base_positions; // I32 [B]: each lane's frontier
  std::unique_ptr<ninfer::DeviceBuffer> extents;        // I32 [B]: draft columns this round
  std::unique_ptr<ninfer::DeviceBuffer> valid_columns;  // I32 [B]: extent + 1
  std::unique_ptr<ninfer::DeviceBuffer> lengths;        // I32 [B]: the accept RNG's position base
  // GitHub #195: a vision load's verify columns rotate at `position +
  // rope_delta`, the way its decode rounds do (`decode_rope_positions`).
  // Present only on a load with vision; null everywhere else, where the
  // round rotates at `positions` itself. I32 [k+1, B], staged whole by the
  // round -- `base + min(j, extent) + rope_delta` per lane -- rather than
  // derived on the device, because the delta is per lane and the vendored
  // `offset_i32_positions` takes one scalar for the whole matrix.
  std::unique_ptr<ninfer::DeviceBuffer> rope_positions;

  // Written by the traversal (device-only).
  std::unique_ptr<ninfer::DeviceBuffer> verify_ids;      // I32 [k+1, B]
  std::unique_ptr<ninfer::DeviceBuffer> positions;       // I32 [k+1, B]: base + min(j, extent)
  std::unique_ptr<ninfer::DeviceBuffer> target_tokens;   // I32 [k+1, B]: argmax per column
  std::unique_ptr<ninfer::DeviceBuffer> logits;          // BF16 [vocab, k+1, B]
  std::unique_ptr<ninfer::DeviceBuffer> hidden;          // BF16 [hidden, k+1, B]: final residual
  std::unique_ptr<ninfer::DeviceBuffer> licensed_tokens; // I32 [k+1, B]: accept output
  std::unique_ptr<ninfer::DeviceBuffer> licensed_counts; // I32 [B]
  std::unique_ptr<ninfer::DeviceBuffer> accepted;        // I32 [B]
  std::unique_ptr<ninfer::DeviceBuffer> selectors;       // I32 [B]: committed - 1
  std::unique_ptr<ninfer::DeviceBuffer> selected_hidden; // BF16 [hidden, B]: for the drafter

  // The ReplaySSM records every GDN layer writes for every lane's `k+1`
  // columns (conv input, key, value, {g, beta}), folded into each lane's
  // slot after accept. Record row b is lane b.
  std::unique_ptr<ninfer::DeviceBuffer> records_backing;
  ninfer::GdnReplayRecordLayout records_layout;
  ninfer::GdnReplayRecords records;

  // The accept kernel's own transient scratch (its partial-top-k route at
  // temperature > 0), sized once for `k` drafts across `B` lanes.
  std::unique_ptr<ninfer::DeviceArena> accept_workspace;

  // P5-05 (GitHub #155): the DFlash2 drafter's round buffers, present only on
  // a load with the drafter (null under VERIFY_ONLY, whose drafts come per
  // call). `features` holds the traversal's feature taps -- the target's
  // layer-5/19/33/47/61 outputs for every verify column, concatenated per
  // column in kDflash2TapLayers order, lane-major like `positions` -- which
  // the commit appends to each lane's window for its committed columns;
  // `append_counts` is that column count per lane (0 at extent 0).
  // `drafter_scratch` backs the drafter's forward (inside the verify graph)
  // and that append (after it); a graph bakes the forward's addresses, so it
  // is reserved once here and every call walks it from the same base.
  std::unique_ptr<ninfer::DeviceBuffer> features;      // BF16 [5 * hidden, (k+1) * B]
  std::unique_ptr<ninfer::DeviceBuffer> append_counts; // I32 [B]
  std::unique_ptr<ninfer::DeviceArena> drafter_scratch;

  // One verify graph per exact width, captured by
  // `ignis_decode_graph_capture` after the decode graphs.
  std::array<cudaGraphExec_t, IGNIS_DECODE_MAX_BATCH> graph_exec{};
  std::array<bool, IGNIS_DECODE_MAX_BATCH> graph_ready{};

  std::size_t device_bytes() const {
    std::size_t bytes = 0;
    for (const auto *buffer :
         {anchors.get(), drafts.get(), base_positions.get(), extents.get(), valid_columns.get(),
          lengths.get(), rope_positions.get(), verify_ids.get(), positions.get(),
          target_tokens.get(), logits.get(), hidden.get(), licensed_tokens.get(),
          licensed_counts.get(), accepted.get(), selectors.get(), selected_hidden.get(),
          records_backing.get(), features.get(), append_counts.get()}) {
      if (buffer != nullptr) {
        bytes += buffer->bytes;
      }
    }
    for (const auto *arena : {accept_workspace.get(), drafter_scratch.get()}) {
      if (arena != nullptr) {
        bytes += arena->capacity();
      }
    }
    return bytes;
  }
};

// The opaque loaded-model handle (never dereferenced across the boundary).
struct ignis_model {
  ninfer::Weight token_embedding;
  ninfer::Weight final_norm;
  ninfer::Weight output_head;
  std::vector<LayerWeights> layers;
  uint64_t vram_bytes = 0;
  uint64_t bound_tensor_count = 0;

  // Program-layer resources (ADR 0009, GitHub #54): the step ABI's stream
  // and scratch arena for degenerate-program intermediates (embedding /
  // norm / logits / argmax buffers). Owned by the model handle so Rust
  // never sees a stream (the spec: "streams are internal to the leaf").
  // `hidden` / `vocab` / `rms_norm_eps` are copied from the topology
  // descriptor at load time so the step ABI does not need it again.
  uint64_t hidden = 0;
  uint64_t vocab = 0;
  float rms_norm_eps = 0.0F;
  // GitHub #227: the text rotary table every GQA layer rotates at, built
  // once at load from the load's `--rope-scaling` (kernel/src/rope_scaling.h)
  // -- the linear table without it, the YaRN one with it. Held here rather
  // than rebuilt per layer call, which is what the GQA layers did before the
  // table had anything to say.
  ninfer::ops::RopeFrequencies text_rope =
      ninfer::ops::rope_linear_frequencies(ignis::kTextRopeTheta, ignis::kTextRotaryDim);
  cudaStream_t stream = nullptr;
  // GitHub #212: with vision, media encode runs out of this arena too,
  // between prefill steps; it is sized for the larger of the two.
  std::unique_ptr<ninfer::DeviceArena> scratch;
  // GitHub #212: what one prefill chunk alone needs of `scratch` -- its
  // capacity without vision.
  uint64_t prefill_scratch_bytes = 0;
  uint64_t last_step_micros = 0;
  uint64_t last_step_kernel_count = 0;

  // P2-02 (GitHub #84): the chunk width the scratch arena above was sized
  // for at load (P2-01, GitHub #83). The chunked prefill route cuts a span
  // into chunks of this width; the last chunk of a span may be narrower.
  uint32_t prefill_chunk_tokens = 0;

  // P3-03 (GitHub #99): device-side sampling's staging buffers. Separate
  // from `scratch` above (which resets every call) because a decode round's
  // configs/positions must sit at addresses a future decode CUDA graph
  // (P3-05/#102) can replay reading -- the host writes this round's values
  // into these same buffers every call, it never reallocates them.
  // `sampling_single_configs`/`sampling_single_positions` back the
  // one-sequence-at-a-time calls (`ignis_prefill`, `ignis_decode`'s
  // per-degenerate-step loop, `ignis_program_prefill`);
  // `sampling_decode_configs`/`sampling_decode_positions` are sized for
  // `IGNIS_DECODE_MAX_BATCH` lanes and back `ignis_program_decode`.
  // `sampling_workspace` is `ninfer::ops::sample`'s own transient scratch
  // (candidate selection, not the caller's inputs above), sized once for the
  // widest lane count and reset via its own Scope every call -- it carries
  // no cross-call state, so it does not need a stable address.
  // `*_out` are the device I32 destinations `ninfer::ops::sample` writes
  // picked ids into -- distinct buffers from `*_positions` above (the op's
  // contract forbids `out` aliasing `logical_positions`).
  std::unique_ptr<ninfer::DeviceBuffer> sampling_single_configs;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_single_positions;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_single_out;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_configs;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_positions;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_out;
  // The decode round's batched logits: BF16 [vocab, IGNIS_DECODE_MAX_BATCH]
  // -- lane i's forward pass copies its own single-token logits into column
  // i (device-to-device, still inside its own scratch scope) so the round's
  // sampling is one `ninfer::ops::sample` call over every lane, not one call
  // per lane.
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_logits;
  // P6-06 (GitHub #242): the round's per-lane permitted token sets, staged
  // like the configs above and at stable addresses for the same reason.
  // `permitted` is I32 [IGNIS_DECODE_MAX_BATCH][IGNIS_MAX_PERMITTED_TOKENS],
  // `permitted_counts` is I32 [lanes] (0 = that lane is unconstrained), and
  // `permitted_probs` is the F32 [lanes] the post-sample kernel writes the
  // committed token's restricted probability into.
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_permitted;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_permitted_counts;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_permitted_probs;
  std::unique_ptr<ninfer::DeviceArena> sampling_workspace;

  // P3-05 (GitHub #102, ADR 0019): the decode CUDA graphs' own resources,
  // reserved once at load (scratch/staging) and captured once after the
  // sequence pool exists (`ignis_decode_graph_capture`). `max_context_tokens`
  // is copied from `ignis_model_load`'s argument so the graph's fixed,
  // conservative `GqaExecutionEnvelope` (every replay, regardless of a
  // lane's actual position) needs no second parameter threaded through
  // capture.
  uint32_t max_context_tokens = 0;
  // P4-05 (GitHub #123): the KV storage format (`enum ignis_kv_format`) the
  // two scratch arenas above were reserved for. Both attention workspace
  // queries are asked under this format's cache dtype, and the GQA layer
  // entry points refuse a sequence pool built in the other one -- the format
  // is fixed for the life of a load (ADR 0022), and here that is not a
  // convention but an arena-sizing fact.
  int32_t kv_format = IGNIS_KV_FORMAT_BF16;
  // Separate from `scratch` above: a graph replays fixed device addresses,
  // and a prefill chunk (which only ever uses `scratch`) landing between two
  // replays must never alias what a replay rereads. Sized once for one
  // lane's per-layer peak at T=1 (mirrors `scratch`'s own sizing at
  // `prefill_chunk_tokens`, kernel/src/model.cu) and reused, via its own
  // `Scope`, sequentially across a graph's lanes -- the same reuse pattern
  // the eager per-lane decode loop already applies to `scratch`.
  std::unique_ptr<ninfer::DeviceArena> decode_graph_scratch;
  // This round's token id per lane (I32 x IGNIS_DECODE_MAX_BATCH),
  // refreshed by one H2D copy before a replay; a captured graph's embedding
  // step reads lane i's id directly from column i, no per-lane device copy.
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_token_ids;
  // This round's physical pool slot per lane (I32 x IGNIS_DECODE_MAX_BATCH),
  // refreshed the same way. The single value at column i serves both GQA's
  // `kv_table_rows` (selecting a row of the pool-wide block-table matrix)
  // and GDN's `initial_state_slots`/`snapshot_base_slots` (in place: same
  // buffer for both, since a physical slot is a physical slot) -- read by
  // the kernels at replay time, never baked at capture time (ADR 0019).
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_slots;
  std::array<cudaGraphExec_t, IGNIS_DECODE_MAX_BATCH> decode_graph_exec{};
  std::array<bool, IGNIS_DECODE_MAX_BATCH> decode_graph_ready{};
  // Set by the most recent `ignis_program_decode` call: 1 if it replayed a
  // graph, 0 if it ran the eager loop (`ignis_program_stats`'s
  // `graph_launches`).
  uint64_t last_step_graph_launches = 0;

  // P5-02 (GitHub #150): speculation, chosen at load. Under
  // IGNIS_SPECULATIVE_DFLASH2, `dflash2` is bound and the prefill scratch
  // carries the drafter's context append. The drafter's window is
  // per-sequence state, so it lives in the sequence pool, one lane per slot
  // (P5-03, GitHub #152, `ignis_seq_pool`); a pool
  // built without the same backend is refused by the program entry points.
  int32_t speculative_backend = IGNIS_SPECULATIVE_NONE;
  uint32_t draft_tokens = 0;
  Dflash2Weights dflash2{};

  // GitHub #177: the vision tower, bound in its stored formats when the load
  // names a vision envelope (`vision_max_tokens > 0`), and one item's output
  // transient, sized once for the envelope. The encoder's workspace is
  // `scratch` above, grown to fit one item of `vision_item_max_tokens`
  // (GitHub #212) -- the load's item bound, never above the envelope.
  uint32_t vision_max_tokens = 0;
  uint32_t vision_item_max_tokens = 0;
  VisionWeights vision{};
  // GitHub #243: the embedding pool, carved into fixed-width column pages.
  // Where the reference (and GitHub #178 after it) keeps one output
  // transient, this holds as many embeddings as their own columns fit --
  // which is what lets an embedding outlive the encode that made it.
  VisionEmbeddingPool vision_pool{};
  // GitHub #178: the decode round's per-lane rope positions (I32 x
  // IGNIS_DECODE_MAX_BATCH), `position + rope_delta`, staged beside
  // `sampling_decode_positions` and read by the graphs from this stable
  // address. Only a vision load has one: a text load's rounds rotate at the
  // positions themselves, exactly as before.
  std::unique_ptr<ninfer::DeviceBuffer> decode_rope_positions;
  // What vision adds beside a text load: the embedding pool, and what the
  // encoder's workspace grew `scratch` by.
  uint64_t vision_reserved_bytes() const {
    const uint64_t shared = scratch ? scratch->capacity() : 0;
    return (shared > prefill_scratch_bytes ? shared - prefill_scratch_bytes : 0) +
           (vision_pool.buffer ? vision_pool.buffer->bytes : 0);
  }

  // P5-04 (GitHub #153): the verify round's substrate, present exactly when
  // `draft_tokens > 0`. Its traversal runs out of `decode_graph_scratch`,
  // which a windowed load sizes for `k+1` columns per lane instead of one.
  std::unique_ptr<IgnisVerifyRound> verify;
};

// GitHub #178: a media embedding -- the `[hidden, columns]` BF16 encoder
// output, live until released. GitHub #243: its columns sit in the pages
// `pages` names, in order, `page_columns` of them per page and the last one
// short; the embedding is contiguous in column space and not in memory.
struct ignis_media_embedding {
  ignis_model *model = nullptr;
  std::int32_t columns = 0;
  std::vector<std::int32_t> pages;
};

// GitHub #178: the encoder workspace for `tokens` merged tokens over
// `segments` segments -- the reference's `build_workspace_layout`, shared by
// the load's reservation (kernel/src/model.cu) and the encode that runs out
// of it (kernel/src/vision_encode.cu), so the two cannot drift.
std::size_t ignis_vision_workspace_bytes(std::int32_t tokens, std::int32_t segments);
