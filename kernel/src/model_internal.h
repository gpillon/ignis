// ignis kernel leaf - the `ignis_model` handle's real definition, shared
// between model.cu (P1-17, GitHub #53: load / stats / free) and step.cu
// (P1-18, GitHub #54: the degenerate embed -> norm -> head -> argmax
// program). Never exposed across the C ABI (kernel/include/ignis_model.h
// keeps `struct ignis_model;` opaque) -- this header is leaf-internal.

#pragma once

#include "drafter.h"
#include "gdn_gating_home.h"
#include "ignis_model.h"
#include "ignis_step.h"
#include "ignis_trellis_linear.h"

#include "prefill_swiglu.h"
#include "process_environment.h"
#include "rope_scaling.h"
#include "round_layout.h"
#include "round_shape.h"
#include "round_timing.h"
#include "tma_descriptor_cache.h"
#include "w4a4_activation_layout.h"

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// GitHub #339 (spec 27b-16gb/05a): a trellis body's linears, the 16 GB
// artifact's split parts (spec 27b-16gb/01's As built) -- bound instead of a
// layer's NVFP4 planes below, never beside them (kernel/src/trellis_body.h).
//
// Both kinds of layer end the same way after their token mixer: the output
// projection's residual add and the MLP (`ignis::trellis_body::output_and_mlp`).
struct TrellisTail {
  ignis_trellis_weight output;
  ignis_trellis_weight gate_proj;
  ignis_trellis_weight up_proj;
  ignis_trellis_weight down;
};

struct GqaTrellisWeights {
  ignis_trellis_weight q_proj;  // [24 heads x (q 256 | gate 256)] x hidden
  ignis_trellis_weight k_proj;
  ignis_trellis_weight v_proj;
  TrellisTail tail;
};

struct GdnTrellisWeights {
  ignis_trellis_weight in_proj_qkv;  // [q 2048 | k 2048 | v 6144] x hidden: the conv input's channels
  ignis_trellis_weight in_proj_z;
  TrellisTail tail;
};

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
  // GitHub #339: the trellis body's linears; empty on the NVFP4 body, where
  // the planes above are bound.
  GqaTrellisWeights trellis{};
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
  // GitHub #339: as GqaLayerWeights::trellis.
  GdnTrellisWeights trellis{};
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

// GitHub #620 (spec #618): the Qwen 3.8-27B's MTP head -- the checkpoint's one
// MTP layer, the container's 12 `mtp/` objects, its matrices W8 on every 27B
// artifact (the 16 GB ones included) and its gains BF16. The head has no
// embedding and no output head of its own: it reads the target's
// `token_embedding` and scores with the target's output head. Bound only
// under IGNIS_SPECULATIVE_MTP (kernel/src/mtp_head.h runs it).
struct MtpHeadWeights {
  ninfer::Weight input_projection;      // W8 [hidden, 2 hidden]: the stem's fc over [embedding; hidden]
  ninfer::Weight embedding_norm;        // BF16 [hidden]
  ninfer::Weight hidden_norm;           // BF16 [hidden]
  ninfer::Weight input_norm;            // BF16 [hidden]: the layer's
  ninfer::Weight query_key_gate_value;  // W8 [2 q + 2 kv, hidden]: rows query, key, gate, value
  ninfer::Weight query_norm;            // BF16 [head_dim]
  ninfer::Weight key_norm;              // BF16 [head_dim]
  ninfer::Weight output;                // W8 [hidden, q]
  ninfer::Weight post_attention_norm;   // BF16 [hidden]
  ninfer::Weight mlp_gate_up;           // W8 [2 ffn, hidden]: gate rows, then up rows
  ninfer::Weight mlp_down;              // W8 [hidden, ffn]
  ninfer::Weight final_norm;            // BF16 [hidden]: the head's own, before the output head
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

// GitHub #414 (ADR 0075): under IGNIS_VISION_RESIDENCY_HOST the tower is
// copied into the prefill scratch for each media encode in these stages, in
// the order the encode reads them: the pre-block weights (the patch
// embedding, its bias and the position table, ~6.7 MB), each of the 27 blocks
// (~8.94 MB), the merger's norm and fc1 (~22.6 MB), then its fc2 (~25.1 MB,
// the largest) on the 27B; Flash-Next's companion, every matrix W8 (GitHub
// #556), ~7.2 MB, ~16.4 MB a block, ~22.6 MB (its largest) and ~12.6 MB.
// `visit(stage, weight)` once per tensor, in stage order.
inline constexpr std::size_t kVisionStagingStages = 1 + kVisionLayers + 2;

template <class Weights, class Visit> void visit_vision_stages(Weights &w, Visit visit) {
  visit(std::size_t{0}, w.patch_embedding);
  visit(std::size_t{0}, w.patch_embedding_bias);
  visit(std::size_t{0}, w.position_embedding);
  for (std::size_t b = 0; b < w.layers.size(); ++b) {
    auto &l = w.layers[b];
    for (auto *weight : {&l.qkv, &l.qkv_bias, &l.output, &l.output_bias, &l.fc1, &l.fc1_bias, &l.fc2,
                         &l.fc2_bias, &l.norm1_weight, &l.norm1_bias, &l.norm2_weight, &l.norm2_bias}) {
      visit(1 + b, *weight);
    }
  }
  const std::size_t merger = 1 + w.layers.size();
  visit(merger, w.merger_norm_weight);
  visit(merger, w.merger_norm_bias);
  visit(merger, w.merger_fc1);
  visit(merger, w.merger_fc1_bias);
  visit(merger + 1, w.merger_fc2);
  visit(merger + 1, w.merger_fc2_bias);
}

// GitHub #414: the staging ring's own stream and events, created at load
// under IGNIS_VISION_RESIDENCY_HOST and absent otherwise. `copy` is
// non-blocking and the encode's alone: the next stage copies on it while the
// model's stream computes the current one, ordered by `copied` (a slot holds
// its stage) and `consumed` (a slot's stage has been read, so the stage three
// on may overwrite it); `head` marks the work queued on the scratch before the
// encode, which the first copies wait for.
struct VisionStaging {
  std::size_t slot_bytes = 0;
  cudaStream_t copy = nullptr;
  cudaEvent_t head = nullptr;
  std::array<cudaEvent_t, IGNIS_VISION_STAGING_SLOTS> copied{};
  std::array<cudaEvent_t, IGNIS_VISION_STAGING_SLOTS> consumed{};

  VisionStaging() = default;
  VisionStaging(const VisionStaging &) = delete;
  VisionStaging &operator=(const VisionStaging &) = delete;
  // Whichever of them the load made, on any of its exits and at free; a copy
  // still running finishes first.
  ~VisionStaging() {
    if (copy != nullptr) {
      cudaStreamSynchronize(copy);
      cudaStreamDestroy(copy);
    }
    for (cudaEvent_t event : copied) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    for (cudaEvent_t event : consumed) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    if (head != nullptr) {
      cudaEventDestroy(head);
    }
  }

  bool staged() const { return copy != nullptr; }
};

struct ignis_media_embedding;

// The columns `ignis_program_prefill`'s span-logits readout scores at a time
// (kernel/src/step.cu): the widest call it makes of the output head, which
// the trellis body's workspace is sized for (GitHub #339, kernel/src/model.cu).
inline constexpr std::int32_t kSpanLogitsBlockColumns = 32;

// GitHub #302: a Flash-Next load's program (kernel/src/flash_next/
// program.h), complete only there; its deleter is defined beside it so this
// header need not see the type.
namespace ignis::flash_next {
struct FlashNextModel;
struct FlashNextModelDeleter {
  void operator()(FlashNextModel *model) const;
};
}  // namespace ignis::flash_next

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
//
// GitHub #415 (ADR 0075): under IGNIS_VISION_RESIDENCY_HOST the same pages
// are one mapped, portable pinned host allocation (`mapped`) instead of a
// device buffer: the encoder writes each page through a device page and copies
// it there, and the prefill chunk's scatter reads it in place through its
// device alias, as ADR 0062's token embedding is read. Its holds, eviction and
// sizing are the device pool's.
struct MappedHostRegion {
  void *host = nullptr;
  // The pointer the device reads and writes it through.
  void *device = nullptr;
  std::size_t bytes = 0;

  // cudaHostAlloc(Mapped | Portable) of `bytes`; throws naming the bytes and
  // the CUDA error, holding nothing.
  explicit MappedHostRegion(std::size_t size_bytes);
  MappedHostRegion(const MappedHostRegion &) = delete;
  MappedHostRegion &operator=(const MappedHostRegion &) = delete;
  ~MappedHostRegion() {
    if (host != nullptr) {
      cudaFreeHost(host);
    }
  }
};

struct VisionEmbeddingPool {
  // Exactly one of the two, by the load's vision residency: VRAM under
  // DEVICE, mapped host memory under HOST.
  std::unique_ptr<ninfer::DeviceBuffer> buffer;
  std::unique_ptr<MappedHostRegion> mapped;
  // `[hidden]` BF16 columns per page, and so the bytes of one.
  std::int32_t page_columns = 0;
  std::size_t page_bytes = 0;
  // Per page: the embedding holding it, or nullptr. A raw pointer and not a
  // bool because a release names its embedding, not its pages.
  std::vector<const ignis_media_embedding *> owner;

  bool present() const { return buffer != nullptr || mapped != nullptr; }
  // The pool's bytes where each placement counts them: VRAM, host-mapped.
  std::uint64_t device_bytes() const { return buffer != nullptr ? buffer->bytes : 0; }
  std::uint64_t host_mapped_bytes() const { return mapped != nullptr ? mapped->bytes : 0; }
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
  // A page as the device addresses it, wherever the pool lives.
  std::uint8_t *page_ptr(std::int32_t page) const {
    void *const base = buffer != nullptr ? buffer->p : mapped->device;
    return static_cast<std::uint8_t *>(base) + static_cast<std::size_t>(page) * page_bytes;
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

// GitHub #676 (spec #672): a DFlash2 load's verify rounds stage their inputs
// and take their accept back through host-mapped memory
// (kernel/src/deferred_verify.cu, kernel/src/verify_return.h): one kernel
// copies a round's staged inputs to their device buffers, and one copies its
// accept into the slot and then writes the ticket the host waits on. Made at
// load, so nothing page-locks host memory while serving, and freed with the
// model: a model loaded again in the same process stages through memory of
// its own.
//
// Two slots, one a round, alternating: a round writes its slot only once the
// round that used it before has run past its last kernel reading it
// (`used_by`, an event recorded after that round's work), so a slot is never
// rewritten under a kernel still to read it.
struct VerifyStaging {
  static constexpr std::size_t kSlotBytes = 16384;
  static constexpr std::uint32_t kMaxCopies = 12;
  struct Copy {
    void *dst;
    const void *src;
    std::uint32_t bytes;
  };
  // Copies one kernel makes, in order: passed to it by value.
  struct Copies {
    Copy at[kMaxCopies];
    std::uint32_t count = 0;
  };

  // The two slots, then the ticket.
  MappedHostRegion region{2 * kSlotBytes + 64};
  std::array<cudaEvent_t, 2> used_by{};
  // This round's slot, the bytes of it written so far, and the copies staged
  // there and not yet made.
  int slot = 0;
  std::size_t used = 0;
  Copies pending;
  // The last ticket handed out.
  std::uint64_t ticket = 0;

  // Zeroes the region and creates the events; throws naming what failed.
  VerifyStaging();
  VerifyStaging(const VerifyStaging &) = delete;
  VerifyStaging &operator=(const VerifyStaging &) = delete;
  ~VerifyStaging();
};

// P5-04 (GitHub #153): the verify round's substrate, built at load when the
// options fixed a draft window (`draft_tokens > 0`, under DFLASH2 or
// VERIFY_ONLY) and absent otherwise, so a load without a window reserves
// nothing here.
//
// Everything below is a stable device address the verify graphs bake in
// (ADR 0019 / 0020): the round refreshes the *_staging buffers in place by
// one H2D copy each before a replay (GitHub #676: a DFlash2 load's round, all
// of them by one kernel from `staging`), the traversal writes the rest. `B` is
// the load's `decode_lanes` (GitHub #336); `k` is the window; a `[k+1, B]` matrix is
// contiguous with column `b`'s `k+1` entries adjacent, which is both the
// vendored `speculative_*` ops' `[K+1,B]` and the traversal's lane-major
// column order (lane b's columns are `b*(k+1) .. b*(k+1)+k`).
//
// GitHub #686 (spec #684): one round shape's verify state. A verify round, its
// graph capture and its replay are handed the shape they run at
// (`run_verify_round`, `ignis_verify_graph_run_batch`), and everything they
// call takes it as an argument: nothing inside them reads a shape from the
// model (`ignis_model::base_shape`).
//
// GitHub #687: a DFlash2 load past IGNIS_BASE_SHAPE_DRAFT_TOKENS holds two
// shapes (`ignis_model::extended_shape`) on one set of device buffers: every
// buffer and arena below is held by both, allocated once at the larger of the
// two shapes' sizes (kernel/src/model.cu `plan_load_sizes`), and each shape
// lays its own window and lanes over them -- its strides, its ReplaySSM
// records' layout -- with host-mapped staging and graphs of its own. One round
// runs at a time on the model's one stream and each stages or writes every
// buffer before it reads it, so a round of one shape never reads what a round
// of the other left, but for the kept scores (`kept_epoch`).
struct IgnisVerifyRound {
  uint32_t window = 0; // k
  // GitHub #687: the most lanes a round at this shape runs, and the widths its
  // graphs are captured at: the load's decode lanes for its base shape, at most
  // IGNIS_EXTENDED_SHAPE_MAX_LANES for an extended one.
  uint32_t lanes = 0;
  // Whether this is a load's extended shape: more than the base shape's 8
  // columns a lane.
  bool extended() const { return window > IGNIS_BASE_SHAPE_DRAFT_TOKENS; }

  // Host-refreshed per round.
  std::shared_ptr<ninfer::DeviceBuffer> anchors;        // I32 [B]: each lane's pending token
  std::shared_ptr<ninfer::DeviceBuffer> drafts;         // I32 [k, B]: the proposals
  std::shared_ptr<ninfer::DeviceBuffer> base_positions; // I32 [B]: each lane's frontier
  std::shared_ptr<ninfer::DeviceBuffer> extents;        // I32 [B]: draft columns this round
  std::shared_ptr<ninfer::DeviceBuffer> valid_columns;  // I32 [B]: extent + 1
  std::shared_ptr<ninfer::DeviceBuffer> lengths;        // I32 [B]: the accept RNG's position base
  // GitHub #195: a vision load's verify columns rotate at `position +
  // rope_delta`, the way its decode rounds do (`decode_rope_positions`).
  // Present only on a load with vision; null everywhere else, where the
  // round rotates at `positions` itself. I32 [k+1, B], staged whole by the
  // round -- `base + min(j, extent) + rope_delta` per lane -- rather than
  // derived on the device, because the delta is per lane and the vendored
  // `offset_i32_positions` takes one scalar for the whole matrix.
  std::shared_ptr<ninfer::DeviceBuffer> rope_positions;

  // Written by the traversal (device-only).
  std::shared_ptr<ninfer::DeviceBuffer> verify_ids;      // I32 [k+1, B]
  std::shared_ptr<ninfer::DeviceBuffer> positions;       // I32 [k+1, B]: base + min(j, extent)
  std::shared_ptr<ninfer::DeviceBuffer> target_tokens;   // I32 [k+1, B]: argmax per column
  std::shared_ptr<ninfer::DeviceBuffer> logits;          // BF16 [vocab, k+1, B]
  std::shared_ptr<ninfer::DeviceBuffer> hidden;          // BF16 [hidden, k+1, B]: final residual
  std::shared_ptr<ninfer::DeviceBuffer> licensed_tokens; // I32 [k+1, B]: accept output
  std::shared_ptr<ninfer::DeviceBuffer> licensed_counts; // I32 [B]
  std::shared_ptr<ninfer::DeviceBuffer> accepted;        // I32 [B]
  std::shared_ptr<ninfer::DeviceBuffer> selectors;       // I32 [B]: committed - 1
  std::shared_ptr<ninfer::DeviceBuffer> selected_hidden; // BF16 [hidden, B]: for the drafter

  // The ReplaySSM records every GDN layer writes for every lane's `k+1`
  // columns (conv input, key, value, {g, beta}), folded into each lane's
  // slot after accept. Record row b is lane b.
  std::shared_ptr<ninfer::DeviceBuffer> records_backing;
  ninfer::GdnReplayRecordLayout records_layout;
  ninfer::GdnReplayRecords records;

  // The accept kernel's own transient scratch (its partial-top-k route at
  // temperature > 0), sized once for `k` drafts across `B` lanes.
  std::shared_ptr<ninfer::DeviceArena> accept_workspace;

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
  std::shared_ptr<ninfer::DeviceBuffer> features;      // BF16 [5 * hidden, (k+1) * B]
  std::shared_ptr<ninfer::DeviceBuffer> append_counts; // I32 [B]
  std::shared_ptr<ninfer::DeviceArena> drafter_scratch;

  // GitHub #576 (spec #574, ADR 0090): the load's draft rule. Under the
  // sampled draft a sampling lane draws its drafts from the drafter's
  // distribution q (`ignis_sampled_draft_draw`, inside the drafter's
  // forward) and the round's accept is ignis's (`ignis_sampled_accept`); a
  // greedy lane keeps the walk's drafts and takes them as under the fixed
  // draft. Under the fixed draft, the rule of earlier releases, the round is
  // exactly what it was: the walk's drafts and the vendored accept. The
  // drafter's temperature is the lane's times `drafter_temperature_factor`,
  // its top-p `drafter_top_p`. With the drafter, `draft_candidates` and
  // `draft_q` keep each sampling lane's q -- its candidates' ids and
  // probabilities -- from the draw to the accept; counted on the drafter's
  // plan line. Null under the fixed draft and on a
  // verify-only load, whose drafts come per call and whose q is one-hot on
  // them.
  bool sampled_draft = false;
  float drafter_temperature_factor = 0.0f;
  float drafter_top_p = 1.0f;
  std::shared_ptr<ninfer::DeviceBuffer> draft_candidates; // I32 [16, k, B]
  std::shared_ptr<ninfer::DeviceBuffer> draft_q;          // F32 [16, k, B]

  // GitHub #577 (test-only): during a teacher-forced round under the sampled
  // draft, the device copy of its recorded tokens (I32 [k + 1, B]), which the
  // draw drafts along and the accept takes by the forced verdict. Set by that
  // round alone (kernel/src/step.cu), which runs eagerly and clears it before
  // it returns; null in every serving round and while the verify graphs are
  // captured, so no graph carries it.
  const std::int32_t *forced_tokens = nullptr;

  // GitHub #692 (test-only): during a teacher-forced round that names its
  // drafts (`ignis_decode_options::forced_drafts`), their host copy at this
  // shape's stride (I32 [k, B]), which the pass writes over the drafter's
  // proposal (kernel/src/decode_graph.cu). Set by that round alone, which
  // runs eagerly and clears it before it returns; null in every serving round
  // and while the verify graphs are captured, so no graph carries it.
  const std::int32_t *forced_drafts = nullptr;

  // GitHub #590 (spec #287): `logits` is also where a repair reads the
  // scores of a lane whose commit the host cut (`ignis_kept_scores`, source
  // kVerifyLogits). Every round writes the whole buffer, so each round stamps
  // it once, from the model's `kept_epoch` counter, and a repair refuses a
  // kept record whose stamp is no longer this one. GitHub #687: on a load of
  // two shapes the buffer is both shapes', and a record is current only while
  // no round of either has written it since (kernel/src/step.cu
  // `kept_scores_current`): one stamp across both, read at the repair.
  std::uint64_t kept_epoch = 0;

  // One verify graph per exact width 1..lanes, captured by
  // `ignis_decode_graph_capture` after the decode graphs; the tables stay
  // the most a round serves wide (GitHub #336).
  std::array<cudaGraphExec_t, IGNIS_DECODE_MAX_BATCH> graph_exec{};
  std::array<bool, IGNIS_DECODE_MAX_BATCH> graph_ready{};
  // GitHub #572: each captured graph's node count, 0 for a width not captured
  // (`ignis_program_stats`'s `verify_graph_nodes`).
  std::array<uint32_t, IGNIS_DECODE_MAX_BATCH> graph_nodes{};
  // GitHub #680: how many of those nodes are copies (`ignis_program_stats`'s
  // `verify_graph_copy_nodes`).
  std::array<uint32_t, IGNIS_DECODE_MAX_BATCH> graph_copy_nodes{};

  // GitHub #676: IGNIS_DEFER_VERIFY as the latest `ignis_decode_graph_capture`
  // read it, off before a load's first capture, as the copy-free round and the
  // GDN gating's home are (earlier releases' round until then); the staging a DFlash2 load's rounds take
  // under it, made at load (null on every other load); the drafter's append
  // after the commit as one graph per width, captured with the verify graphs
  // under it; and the rounds that returned with their fold in flight
  // (`ignis_program_stats`).
  bool defer = false;
  std::unique_ptr<VerifyStaging> staging;
  std::array<cudaGraphExec_t, IGNIS_DECODE_MAX_BATCH> append_exec{};
  // GitHub #712: each append graph's node count, 0 for a width not captured
  // (`ignis_program_stats`'s round shapes' `drafter_append_graph_nodes`).
  std::array<uint32_t, IGNIS_DECODE_MAX_BATCH> append_nodes{};
  std::uint64_t folds_in_flight = 0;
  // GitHub #687: the verify rounds run at this shape, and how many of them ran
  // eagerly, with no graph (`ignis_program_stats`'s round shapes).
  std::uint64_t rounds = 0;
  std::uint64_t eager_rounds = 0;

  // The device bytes this shape holds; GitHub #687: with `shared_with`, only
  // the buffers it does not hold too, so a load's two shapes count their one
  // set of buffers once.
  std::size_t device_bytes(const IgnisVerifyRound *shared_with = nullptr) const {
    const auto buffers = [](const IgnisVerifyRound &shape) {
      return std::array<const ninfer::DeviceBuffer *, 22>{
          shape.anchors.get(),         shape.drafts.get(),          shape.base_positions.get(),
          shape.extents.get(),         shape.valid_columns.get(),   shape.lengths.get(),
          shape.rope_positions.get(),  shape.verify_ids.get(),      shape.positions.get(),
          shape.target_tokens.get(),   shape.logits.get(),          shape.hidden.get(),
          shape.licensed_tokens.get(), shape.licensed_counts.get(), shape.accepted.get(),
          shape.selectors.get(),       shape.selected_hidden.get(), shape.records_backing.get(),
          shape.features.get(),        shape.append_counts.get(),   shape.draft_candidates.get(),
          shape.draft_q.get()};
    };
    const auto arenas = [](const IgnisVerifyRound &shape) {
      return std::array<const ninfer::DeviceArena *, 2>{shape.accept_workspace.get(), shape.drafter_scratch.get()};
    };
    const auto own = buffers(*this);
    const auto own_arenas = arenas(*this);
    std::array<const ninfer::DeviceBuffer *, 22> other{};
    std::array<const ninfer::DeviceArena *, 2> other_arenas{};
    if (shared_with != nullptr) {
      other = buffers(*shared_with);
      other_arenas = arenas(*shared_with);
    }
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < own.size(); ++i) {
      if (own[i] != nullptr && own[i] != other[i]) {
        bytes += own[i]->bytes;
      }
    }
    for (std::size_t i = 0; i < own_arenas.size(); ++i) {
      if (own_arenas[i] != nullptr && own_arenas[i] != other_arenas[i]) {
        bytes += own_arenas[i]->capacity();
      }
    }
    return bytes;
  }
};

// GitHub #674: one kind of round's timing under the measurement switch IGNIS_ROUND_TIMING
// (kernel/src/step.cu): an event recorded on the model's stream just before each replayed round's
// graph and one just after it, on the GPU's clock. A ring of them, so a round is read once its own
// events and the next round's launch have completed, without waiting on the stream. GitHub #689:
// one ring for both of a load's round shapes, each slot naming its round's, so that a round's
// period runs to the next round of its kind whatever shape that one is
// (kernel/src/round_timing.h).
struct RoundTiming {
  static constexpr std::uint64_t kRing = 16;
  std::array<cudaEvent_t, kRing> pre{};
  std::array<cudaEvent_t, kRing> post{};
  // Each slot's round shape: 0, the base shape, for every plain round and every verify round of a
  // load of one shape; 1 for an extended round; GitHub #713, `kMixedBlock` for a mixed one.
  std::array<std::uint8_t, kRing> shape{};
  // Rounds whose two events were recorded, and rounds read so far.
  std::uint64_t recorded = 0;
  std::uint64_t collected = 0;
  // The current blocks' samples, one block per round shape.
  ignis::round_timing::Blocks blocks;
};

// The opaque loaded-model handle (never dereferenced across the boundary).
struct ignis_model {
  ninfer::Weight token_embedding;
  ninfer::Weight final_norm;
  ninfer::Weight output_head;
  std::vector<LayerWeights> layers;
  // GitHub #339 (spec 27b-16gb/05a): the body the load bound, by its head's
  // format (kernel/src/model.cu): the NVFP4 planes and `output_head` above,
  // or the 16 GB artifact's trellis matrices -- each layer's `trellis`
  // weights and `trellis_head` -- which every linear and head site then
  // dispatches to (kernel/src/trellis_body.h). Never both.
  bool is_trellis_body = false;
  ignis_trellis_weight trellis_head{};
  // The trellis op's one workspace, reserved at load at a stable address the
  // round graphs bake (its plan line, `trellis_workspace_bytes`) and zeroed
  // once: every trellis call runs on `stream` below, one at a time, and
  // leaves its counters zero. Null on the NVFP4 body.
  std::unique_ptr<ninfer::DeviceBuffer> trellis_workspace_buffer;
  ignis_trellis_workspace trellis_workspace{};
  // GitHub #509: `ignis_model_load_options::prefill_linear` (enum
  // ignis_prefill_linear). Under DENSE every body call wider than the decode
  // route runs the dense prefill (kernel/src/trellis_body.cu): its INT8
  // activations in `trellis_workspace`, each matrix decoded in turn into
  // this one buffer (the plan line `trellis_dense_bytes`), null otherwise.
  uint32_t prefill_linear = IGNIS_PREFILL_LINEAR_FUSED;
  std::unique_ptr<ninfer::DeviceBuffer> trellis_dense_buffer;
  // GitHub #650: on Windows, the W4A4 TMA products' descriptor slots serving
  // `stream` (kernel/src/tma_descriptor_cache.h; the plan line
  // `tma_descriptor_bytes`); null where the load plans none.
  std::unique_ptr<ignis::tma_descriptors::Cache> tma_descriptors;
  // GitHub #652 (spec #646): the layout the W4A4 prefill products read their
  // activations in, decided once at load (model.cu
  // `w4a4_activation_layout_route`), none where the prefill runs no W4A4 TMA
  // product (the C ABI's IGNIS_W4A4_ACTIVATION_LAYOUT_NONE), and that layout
  // named for `stream` (kernel/src/w4a4_activation_layout.h), which every
  // launcher of the products asks; null without a layout.
  std::optional<ignis::w4a4_activation_layout::Layout> w4a4_activation_layout;
  std::unique_ptr<ignis::w4a4_activation_layout::ForStream> w4a4_activation_layout_stream;
  // GitHub #514: the GDN prefill's route, decided once at load (model.cu
  // `gdn_prefill_route`): our full-rate kernels (kernel/src/gdn_prefill.h)
  // when true, the vendored chunked ones otherwise; read by every GDN
  // prefill call of at least ignis::gdn_prefill::kMinTokens tokens
  // (kernel/src/gdn_layer.cu `run_gdn_layer`).
  bool gdn_prefill_ours = false;
  // GitHub #572 (spec #570): the attention route of the decode and verify
  // rounds, decided once at load (model.cu `hq_decode_attention_route`):
  // ignis's warp-specialized route over hq-e8-2b when true
  // (kernel/src/hq_decode_attention.h), the vendored kernel's otherwise; read
  // by every call of the batch body (kernel/src/gqa_layer.cu
  // `run_gqa_layer_batch`), so every graph captured on this load bakes it in.
  bool hq_decode_attention_ours = false;
  // GitHub #688 (spec #684): what IGNIS_EXTENDED_ATTENTION named at the latest
  // `ignis_decode_graph_capture` (kernel/src/extended_attention.h), unset
  // before a load's first: whether an extended round's chunks take the route
  // above where the load takes it, or the vendored chunks. Read by every call
  // of an extended round (kernel/src/gqa_layer.cu), replayed or eager.
  ignis::RouteSwitch extended_attention = ignis::RouteSwitch::Unset;
  // GitHub #627 (spec #618): the MTP head's matrices, decided once at an MTP
  // load (model.cu `mtp_head_linear_route`): ours on an RTX 30 or RTX 40 card
  // (kernel/src/w8_imma.h) when true, the vendored `ops::linear` otherwise;
  // read by every head call (kernel/src/mtp_head.cu `head_linear`), so every
  // graph captured on this load bakes it in. False on a load without the head.
  bool mtp_head_linear_ours = false;
  // GitHub #651, #653 (spec #646): the route of the prefill's MLP gate and up product, decided once
  // at load (model.cu `validate_and_bind`): the fused op (kernel/src/prefill_swiglu.h) at every
  // width the W4A4 TMA route takes, its NVFP4 epilogue where the down product runs on that route
  // (Quant) or its BF16 one throughout (Fused), or the vendored dispatch; read by every MLP call of
  // the NVFP4 body (kernel/src/gdn_layer.cu and gqa_layer.cu `output_and_mlp`). The trellis body
  // runs none.
  ignis::prefill_swiglu::Route prefill_swiglu = ignis::prefill_swiglu::Route::Split;
  uint64_t vram_bytes = 0;
  // ADR 0062: what of the bound tensors lives in mapped host memory.
  uint64_t host_mapped_bytes = 0;
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
  // the load's `decode_lanes` lanes and back `ignis_program_decode`.
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
  // The decode round's batched logits: BF16 [vocab, decode_lanes]
  // -- lane i's forward pass copies its own single-token logits into column
  // i (device-to-device, still inside its own scratch scope) so the round's
  // sampling is one `ninfer::ops::sample` call over every lane, not one call
  // per lane.
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_logits;
  // GitHub #586 (spec #287): the same buffer is where a repair reads the
  // scores a draw came from (`ignis_kept_scores`, ignis_seq_internal.h): a
  // plain round leaves each lane's row in its lane column, and a 27B
  // prefill's draw copies its row into the column of its sequence's slot.
  // Every write stamps the column with the next epoch, so a sequence whose
  // column has been written since its draw knows its scores are gone. No
  // allocation of its own: the rows a round already leaves here are the
  // scores, and a slot's column is free between the rounds that use it.
  std::uint64_t kept_epoch = 0;
  std::array<std::uint64_t, IGNIS_DECODE_MAX_BATCH> kept_column_epochs{};
  // P6-06 (GitHub #242): the round's per-lane permitted token sets, staged
  // like the configs above and at stable addresses for the same reason.
  // `permitted` is I32 [decode_lanes][IGNIS_MAX_PERMITTED_TOKENS],
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
  // GitHub #335: `ignis_model_load_options::prefill_attention` (enum
  // ignis_prefill_attention), for the hq-e8-2b prompt attention's band pass.
  uint32_t prefill_attention = IGNIS_PREFILL_ATTENTION_FP8;
  // Separate from `scratch` above: a graph replays fixed device addresses,
  // and a prefill chunk (which only ever uses `scratch`) landing between two
  // replays must never alias what a replay rereads. Sized once for one
  // lane's per-layer peak at T=1 (mirrors `scratch`'s own sizing at
  // `prefill_chunk_tokens`, kernel/src/model.cu) and reused, via its own
  // `Scope`, sequentially across a graph's lanes -- the same reuse pattern
  // the eager per-lane decode loop already applies to `scratch`.
  std::unique_ptr<ninfer::DeviceArena> decode_graph_scratch;
  // This round's token id per lane (I32 x decode_lanes),
  // refreshed by one H2D copy before a replay; a captured graph's embedding
  // step reads lane i's id directly from column i, no per-lane device copy.
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_token_ids;
  // This round's physical pool slot per lane (I32 x decode_lanes),
  // refreshed the same way. The single value at column i serves both GQA's
  // `kv_table_rows` (selecting a row of the pool-wide block-table matrix)
  // and GDN's `initial_state_slots`/`snapshot_base_slots` (in place: same
  // buffer for both, since a physical slot is a physical slot) -- read by
  // the kernels at replay time, never baked at capture time (ADR 0019).
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_slots;
  // One decode graph per exact width 1..decode_lanes; the tables stay the
  // most a round serves wide.
  std::array<cudaGraphExec_t, IGNIS_DECODE_MAX_BATCH> decode_graph_exec{};
  std::array<bool, IGNIS_DECODE_MAX_BATCH> decode_graph_ready{};
  // GitHub #572: each captured graph's node count, 0 for a width not captured
  // (`ignis_program_stats`'s `decode_graph_nodes`).
  std::array<uint32_t, IGNIS_DECODE_MAX_BATCH> decode_graph_nodes{};
  // GitHub #677: how many of those nodes are copies (`ignis_program_stats`'s
  // `decode_graph_copy_nodes`).
  std::array<uint32_t, IGNIS_DECODE_MAX_BATCH> decode_graph_copy_nodes{};
  // GitHub #677 (spec #672): the decode widths whose traversal runs copy-free,
  // bit w-1 (kernel/src/decode_graph_copies.h), as the latest
  // `ignis_decode_graph_capture` decided them from the card and
  // IGNIS_DECODE_GRAPH_COPIES; every round after it reads them, replayed or
  // eager. None before a load's first capture: a round then copies, as the
  // round of earlier releases does.
  uint32_t decode_copy_free_widths = 0;
  // GitHub #680: whether the verify round's traversal runs copy-free at every
  // width, decided at the same capture by the same rule; false before it.
  bool verify_copy_free = false;
  // GitHub #681 (spec #672): where each GDN layer's gating runs in the decode
  // and verify rounds (kernel/src/gdn_gating_home.h), as the latest
  // `ignis_decode_graph_capture` decided it from the card and
  // IGNIS_GDN_GATING; the chain before a load's first capture.
  ignis::gdn_gating::Home gdn_gating_home = ignis::gdn_gating::Home::Chain;
  // Its branch's stream and events, made by the first capture that takes the
  // branch and destroyed with the model.
  ignis::flash_next::SideBranch gdn_gating_branch;
  // Set by the most recent `ignis_program_decode` call: 1 if it replayed a
  // graph, 0 if it ran the eager loop (`ignis_program_stats`'s
  // `graph_launches`).
  uint64_t last_step_graph_launches = 0;
  // GitHub #674: IGNIS_ROUND_TIMING as the latest `ignis_decode_graph_capture` read it, and the
  // plain and verify rounds' timings it turns on. Their events are made by the first capture that
  // turns it on and destroyed with the model.
  bool round_timing = false;
  RoundTiming plain_timing;
  RoundTiming verify_timing;

  // P5-02 (GitHub #150): speculation, chosen at load. Under
  // IGNIS_SPECULATIVE_DFLASH2, `dflash2` is bound and the prefill scratch
  // carries the drafter's context append. The drafter's window is
  // per-sequence state, so it lives in the sequence pool, one lane per slot
  // (P5-03, GitHub #152, `ignis_seq_pool`); a pool
  // built without the same backend is refused by the program entry points.
  int32_t speculative_backend = IGNIS_SPECULATIVE_NONE;
  // GitHub #619: the drafter that backend names (kernel/src/drafter.h), set
  // where the load binds its weights; every choice the load makes for its
  // drafter follows it.
  IgnisDrafter drafter = IgnisDrafter::kNone;
  uint32_t draft_tokens = 0;
  // GitHub #336: the 27B's decode lanes, a load option (1..IGNIS_DECODE_MAX_BATCH):
  // the widest decode or verify round, what every round buffer above and
  // below is sized for, and the widths the round graphs are captured at. A
  // round wider than it is refused. Flash-Next keeps its own
  // (`FlashNextModel::decode_lanes`).
  uint32_t decode_lanes = IGNIS_DECODE_MAX_BATCH;
  Dflash2Weights dflash2{};
  // GitHub #620: the 27B's MTP head, bound under IGNIS_SPECULATIVE_MTP.
  MtpHeadWeights mtp{};
  // The drafter's shortlist proposal head (`text/draft_head`, Q4 over the
  // 131,072 most frequent tokens) and the token id each of its rows stands
  // for (`text/draft_head_token_ids`), bound when the load hands both over;
  // `proposal_token_ids` is null otherwise and the drafter scores its draft
  // columns with `output_head`. Only what is drafted changes: the verify
  // round always scores with the full head.
  ninfer::Weight proposal_head{};
  const std::int32_t *proposal_token_ids = nullptr;

  // GitHub #177: the vision tower, bound in its stored formats when the load
  // names a vision envelope (`vision_max_tokens > 0`), and one item's output
  // transient, sized once for the envelope. The encoder's workspace is
  // `scratch` above, grown to fit one item of `vision_item_max_tokens`
  // (GitHub #212) -- the load's item bound, never above the envelope.
  uint32_t vision_max_tokens = 0;
  uint32_t vision_item_max_tokens = 0;
  // GitHub #275 (ADR 0041): the most keys of a text span an attention
  // readout reads -- the load's `attention_text_max_keys`, capped by the
  // context -- and so the room `scratch` holds for its scores and rows.
  uint32_t attention_text_max_keys = 0;
  VisionWeights vision{};
  // GitHub #243: the embedding pool, carved into fixed-width column pages.
  // Where the reference (and GitHub #178 after it) keeps one output
  // transient, this holds as many embeddings as their own columns fit --
  // which is what lets an embedding outlive the encode that made it.
  VisionEmbeddingPool vision_pool{};
  // GitHub #414 (ADR 0075): set under IGNIS_VISION_RESIDENCY_HOST, where
  // `vision` points into mapped host memory and each encode stages it into
  // `scratch`; `staged()` is false on a device load.
  VisionStaging vision_staging{};
  // GitHub #558 (ADR 0087): on a Flash-Next vision load, the expert residency
  // whose staging ring an encode may take beside `scratch` -- idle between
  // forwards (ignis_residency.h), and counted on by the item bound -- borrowed
  // as the load borrows it; null on the 27B and when the residency holds every
  // expert, which stages nothing and has no ring.
  struct ignis_residency *vision_ring_lender = nullptr;
  // GitHub #178: the decode round's per-lane rope positions (I32 x
  // decode_lanes), `position + rope_delta`, staged beside
  // `sampling_decode_positions` and read by the graphs from this stable
  // address. Only a vision load has one: a text load's rounds rotate at the
  // positions themselves, exactly as before.
  std::unique_ptr<ninfer::DeviceBuffer> decode_rope_positions;
  // What vision adds beside a text load in VRAM: the embedding pool where it
  // is on the device (GitHub #415: a host-mapped one counts in the stats'
  // `host_mapped_bytes`), and what the encoder's workspace grew `scratch` by.
  uint64_t vision_reserved_bytes() const {
    const uint64_t shared = scratch ? scratch->capacity() : 0;
    return (shared > prefill_scratch_bytes ? shared - prefill_scratch_bytes : 0) + vision_pool.device_bytes();
  }

  // P5-04 (GitHub #153): the verify round's substrate, present exactly when
  // `draft_tokens > 0`. Its traversal runs out of `decode_graph_scratch`,
  // which a windowed load sizes for `k+1` columns per lane instead of one.
  // GitHub #686 (spec #684): the load's base shape, its one round shape. Read
  // here only where a round's shape is chosen (`ignis_program_decode`,
  // `ignis_decode_graph_capture`) and outside the rounds (the load, its
  // reservations and teardown, the stats, the readouts, a repair's kept
  // scores).
  std::unique_ptr<IgnisVerifyRound> base_shape;
  // GitHub #687: on a DFlash2 load past IGNIS_BASE_SHAPE_DRAFT_TOKENS drafts,
  // its extended shape -- the load's window at up to
  // IGNIS_EXTENDED_SHAPE_MAX_LANES lanes, over `base_shape`'s buffers, whose
  // window is then IGNIS_BASE_SHAPE_DRAFT_TOKENS -- and null on every other
  // load. Read where `base_shape` is.
  std::unique_ptr<IgnisVerifyRound> extended_shape;
  // GitHub #689: the shape rule's thresholds a load with an extended shape
  // takes (`ignis_model_load_options::shape_enter_accepted`,
  // `shape_stay_accepted`), read where a verify round's shape is chosen and
  // after it; the defaults, never read, on every other load.
  ignis::round_shape::ShapeRule shape_rule{};
  // GitHub #690: and the shape policy (`ignis_model_load_options::shape_policy`,
  // `shape_estimate_rounds`, `shape_period_ratios`), read where a verify
  // round's shape is chosen.
  ignis::round_shape::ShapePolicy shape_policy{};
  // GitHub #713 (spec #711): a mixed round's base records on a load of two round shapes, the base
  // shape's record width laid at the extended shape's stride over the one backing both shapes'
  // records share (kernel/src/round_layout.h `mixed_base_record_capacity`): bound at load where
  // such a layout exists, `mixed_base_capacity` of its rows, holding no byte of its own; 0 rows
  // where none does, and no mixed round runs there.
  ninfer::GdnReplayRecords mixed_base_records;
  std::uint32_t mixed_base_capacity = 0;
  // Each mix's verify graph and drafter append graph (`ignis::round_layout::kMixTable`'s order), as
  // the latest capture made them under IGNIS_MIXED_ROUNDS=1, and their node counts
  // (`ignis_program_stats`); and the branch a mixed round's base group runs on beside its extended
  // group, made by the first mixed capture or round and destroyed with the model.
  std::array<cudaGraphExec_t, ignis::round_layout::kMixes> mixed_verify_exec{};
  std::array<cudaGraphExec_t, ignis::round_layout::kMixes> mixed_append_exec{};
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed_verify_nodes{};
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed_append_nodes{};
  ignis::flash_next::SideBranch mixed_branch;
  // GitHub #714 (spec #711): the mixes whose graphs the load planned (kernel/src/mixed_graphs.h
  // `planned`, bit m for kMixTable[m]), the plan's `mixed_graphs` line; whether the latest capture
  // asked for the mixed graphs, the mixes it tried and could not capture (a mix is its verify graph
  // and, where the rounds defer, its append graph: a failure of either leaves it neither), the
  // device bytes each captured mix's graphs took (the free memory before its capture less after);
  // the rounds of a mix with no graph that ran at the uniform choice instead, by reason
  // (IGNIS_MIXED_FALLBACK_*); and the test seam's mix whose next capture fails, -1 for none.
  std::uint32_t mixed_planned_mask = 0;
  bool mixed_capture_asked = false;
  std::uint32_t mixed_failed_mask = 0;
  std::array<std::uint64_t, ignis::round_layout::kMixes> mixed_graph_bytes{};
  std::array<std::uint64_t, IGNIS_MIXED_FALLBACK_REASONS> mixed_fallback_rounds{};
  int mixed_capture_fault = -1;
  // GitHub #719: the mixed rounds run, by mix, from the load's start (a round at a mix with no graph
  // that ran at the uniform choice is a fallback above, not counted here).
  std::array<std::uint64_t, ignis::round_layout::kMixes> mixed_rounds{};
  // The kept-scores stamp (`IgnisVerifyRound::kept_epoch`) of the latest mixed round, which ran on
  // the base shape's buffers at each group's stride, so that a read of them at one stride refuses.
  std::uint64_t mixed_kept_epoch = 0;
  // GitHub #718 (spec #711): the mixed graph's build, as the latest capture read it from
  // IGNIS_MIXED_GRAPH_BUILD (kernel/src/mixed_graphs.h `built`; on before a load's first capture):
  // a mixed round's first op runs one group after the other on the round's stream, so that its
  // graph has one root; its target argmax zeroes its output by a kernel (kernel/src/tiled_argmax.h),
  // so that its graph has no memset node; and its extended group runs on `mixed_priority_branch`,
  // which a capture that reads the build makes at the device's greatest stream priority, its graphs
  // instantiated with their nodes' priorities (`ignis_mixed_priorities`). Off: each as it was. And
  // each captured mix's verify and append graph's root nodes and memset nodes
  // (`ignis_program_stats`), 0 for a mix with no graph.
  bool mixed_graph_build = true;
  ignis::flash_next::SideBranch mixed_priority_branch;
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed_verify_roots{};
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed_verify_memsets{};
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed_append_roots{};
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed_append_memsets{};
  // GitHub #302: set on a Flash-Next load, whose program the step ABI's
  // entry points dispatch to. It shares this handle's stream, both scratch
  // arenas, the sampling staging, the decode staging and the decode graphs;
  // none of the 27B's fields above it is bound.
  std::unique_ptr<ignis::flash_next::FlashNextModel, ignis::flash_next::FlashNextModelDeleter> flash_next;
};

// GitHub #718 (spec #711): whether a round of `layout` runs its extended group at the device's
// greatest stream priority, and a graph of it is instantiated with its nodes' priorities
// (cudaGraphInstantiateFlagUseNodePriority): a mixed round under the mixed graph's build, on a load
// whose capture made the branch for it.
inline bool ignis_mixed_priorities(const ignis_model &model, const ignis::round_layout::Layout &layout) {
  return !layout.uniform() && model.mixed_graph_build && model.mixed_priority_branch.stream != nullptr;
}

// GitHub #713 (spec #711): `call(group, stream)` for each group of `layout` that holds lanes: a uniform
// round's one group on the model's stream, as every round of one shape runs; a mixed round's two
// side by side, the extended group on the model's stream and the base group on `mixed_branch`,
// forked from it and joined back (in a capture, edges of the graph and no nodes). The branch is
// joined whatever `call` throws; a mixed round with no branch made throws. GitHub #718: where the
// round takes the mixed graph's priorities (`ignis_mixed_priorities`), the extended group on
// `mixed_priority_branch` instead, at the device's greatest stream priority, and the base group on
// `mixed_branch`, both forked from the model's stream and joined back.
template <class Call>
void ignis_each_group(ignis_model &model, const ignis::round_layout::Layout &layout, Call &&call) {
  if (layout.uniform()) {
    for (const ignis::round_layout::Group &group : layout.groups) {
      if (group.lanes > 0) {
        call(group, model.stream);
      }
    }
    return;
  }
  const auto check = [](cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("the mixed round's branch (") + what + "): " + cudaGetErrorString(e));
    }
  };
  if (model.mixed_branch.stream == nullptr) {
    throw std::runtime_error("a mixed round on a load whose mixed branch was not made");
  }
  if (ignis_mixed_priorities(model, layout)) {
    check(cudaEventRecord(model.mixed_branch.fork, model.stream), "fork");
    check(cudaStreamWaitEvent(model.mixed_priority_branch.stream, model.mixed_branch.fork, 0), "fork");
    check(cudaStreamWaitEvent(model.mixed_branch.stream, model.mixed_branch.fork, 0), "fork");
    struct JoinBoth {
      ignis_model &model;
      ~JoinBoth() {
        (void)cudaEventRecord(model.mixed_priority_branch.join, model.mixed_priority_branch.stream);
        (void)cudaEventRecord(model.mixed_branch.join, model.mixed_branch.stream);
        (void)cudaStreamWaitEvent(model.stream, model.mixed_priority_branch.join, 0);
        (void)cudaStreamWaitEvent(model.stream, model.mixed_branch.join, 0);
      }
    } join{model};
    call(layout.groups[0], model.mixed_priority_branch.stream);
    call(layout.groups[1], model.mixed_branch.stream);
    return;
  }
  check(cudaEventRecord(model.mixed_branch.fork, model.stream), "fork");
  check(cudaStreamWaitEvent(model.mixed_branch.stream, model.mixed_branch.fork, 0), "fork");
  struct Join {
    ignis_model &model;
    ~Join() {
      (void)cudaEventRecord(model.mixed_branch.join, model.mixed_branch.stream);
      (void)cudaStreamWaitEvent(model.stream, model.mixed_branch.join, 0);
    }
  } join{model};
  call(layout.groups[0], model.stream);
  call(layout.groups[1], model.mixed_branch.stream);
}

// GitHub #713 (spec #711): why this load cannot run a mixed round of `layout` over a pool of
// `slot_count` slots, or null where it can: a DFlash2 load of two round shapes on the NVFP4 body;
// one of the twelve mixes, within the load's lanes and its extended shape's; each group's GDN
// records over the round's columns in the records of its shape (a window of 15,
// kernel/src/round_layout.h `records_hold`); and the slots the base group's fold takes, its leading
// rows' beside its lanes'.
inline const char *ignis_mixed_round_refusal(const ignis_model &model, std::int64_t slot_count,
                                             const ignis::round_layout::Layout &layout) {
  if (model.drafter != IgnisDrafter::kDflash2 || model.base_shape == nullptr || model.extended_shape == nullptr) {
    return "a mixed round runs on a DFlash2 load of two round shapes";
  }
  if (model.is_trellis_body) {
    return "a trellis body runs no mixed round";
  }
  if (ignis::round_layout::mix_index(layout) < 0) {
    return "its lanes make none of the twelve mixes";
  }
  if (layout.lanes() > model.decode_lanes || layout.groups[0].lanes > model.extended_shape->lanes) {
    return "its lanes are more than the load's, or its extended lanes more than the extended shape's";
  }
  if (model.mixed_base_capacity == 0 ||
      !ignis::round_layout::records_hold(
          layout, static_cast<std::uint32_t>(model.extended_shape->records_layout.spec.record_capacity),
          model.mixed_base_capacity)) {
    return "the load's GDN records hold no mixed round at its window (at 15 drafts alone a group's record rows "
           "are its columns)";
  }
  if (slot_count < static_cast<std::int64_t>(layout.groups[1].first_record_row + layout.groups[1].lanes)) {
    return "the pool has too few slots for the base group's fold behind its leading rows";
  }
  return nullptr;
}

// GitHub #692 (spec #684): the load's round shape at a draft window of `window` -- its base shape, or
// on a DFlash2 load of two its extended shape -- or null where no shape of the load has that window.
// The shape a test seam is asked for by its window: a teacher-forced round's `forced_shape_window`,
// the readouts' `window` (kernel/include/ignis_dflash2_readout.h).
inline IgnisVerifyRound *ignis_round_shape_at(const ignis_model &model, uint32_t window) {
  if (model.base_shape != nullptr && model.base_shape->window == window) {
    return model.base_shape.get();
  }
  if (model.extended_shape != nullptr && model.extended_shape->window == window) {
    return model.extended_shape.get();
  }
  return nullptr;
}

// GitHub #692: the round shape whose verify round wrote the logits the load's shapes share last, or
// null before its first verify round. Each round stamps its shape from the model's one counter
// (`IgnisVerifyRound::kept_epoch`), so the later stamp is the later round's.
inline const IgnisVerifyRound *ignis_latest_verify_shape(const ignis_model &model) {
  const IgnisVerifyRound *latest = model.base_shape.get();
  if (latest == nullptr) {
    return nullptr;
  }
  if (model.extended_shape != nullptr && model.extended_shape->kept_epoch > latest->kept_epoch) {
    latest = model.extended_shape.get();
  }
  return latest->kept_epoch == 0 ? nullptr : latest;
}

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

// GitHub #414 (ADR 0075): one staging slot's bytes under
// IGNIS_VISION_RESIDENCY_HOST -- the largest stage's bound, from the bound
// tensors' sizes alone (each tensor's payload and the padding its host span
// may add), so the plan, which sees no pointer, and the encode size it alike.
// The encoder's share of the scratch is its workspace, aligned, then
// IGNIS_VISION_STAGING_SLOTS of these (ignis_vision_staged_share).
std::size_t ignis_vision_staging_slot_bytes(const VisionWeights &weights);

// GitHub #414: the encoder's share of the scratch under
// IGNIS_VISION_RESIDENCY_HOST, for an encode workspace of `workspace_bytes`:
// the ring of IGNIS_VISION_STAGING_SLOTS slots of `slot_bytes` starts at
// `ring_offset`, the workspace aligned, and the share ends with it at `bytes`.
// Shared by the load's reservation (kernel/src/model.cu) and the encode that
// lays its share out (kernel/src/vision_encode.cu), so the two cannot drift.
struct VisionStagedShare {
  std::size_t ring_offset = 0;
  std::size_t bytes = 0;
};
VisionStagedShare ignis_vision_staged_share(std::size_t workspace_bytes, std::size_t slot_bytes);

// GitHub #558 (ADR 0087): where one item's encode lays its two parts -- the
// encoder's workspace (`workspace_bytes`) and under IGNIS_VISION_RESIDENCY_HOST
// the tower's staging slots (`slots_bytes`, all IGNIS_VISION_STAGING_SLOTS of
// them; 0 under device) -- over the two areas a Flash-Next load holds idle
// while an item encodes: its prefill scratch (`scratch_bytes`) and its expert
// staging ring (`ring_bytes`; 0 when the residency holds every expert and
// stages nothing). Tried in this order: both in the scratch (the workspace
// aligned, then the slots, as ignis_vision_staged_share lays them), the
// workspace in the scratch and the slots in the ring, the workspace in the
// ring and the slots in the scratch, both in the ring; `fits` is false when
// none holds them. An area holding both lays them out as the scratch does.
// Shared by the item bound the load and its plan compute (kernel/src/model.cu)
// and the encode that lays its share out (kernel/src/vision_encode.cu), so the
// two cannot drift.
struct VisionEncodePlacement {
  bool fits = false;
  bool workspace_in_ring = false;
  bool slots_in_ring = false;
};
VisionEncodePlacement ignis_vision_encode_placement(std::size_t scratch_bytes, std::size_t ring_bytes,
                                                    std::size_t workspace_bytes, std::size_t slots_bytes);

// GitHub #414: throws, naming the stage, unless every stage of `weights` --
// as the binder laid it out in host memory -- fits a slot of `slot_bytes`,
// with each tensor's planes inside its payload. The load asks once, so an
// encode never meets a stage it cannot copy.
void ignis_vision_staging_check(const VisionWeights &weights, std::size_t slot_bytes);
