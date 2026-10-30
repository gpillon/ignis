// ignis kernel leaf - GitHub #178: the media encode step. One media item's
// BF16 patch rows run through the vision tower -- patch projection, the
// bilinear position-table add, 27 blocks (LayerNorm, QKV + bias, 2-D vision
// RoPE, segmented attention, projection + bias + residual, LayerNorm, fc1 +
// bias, GELU-tanh, fc2 + bias + residual), then the merger (LayerNorm, 2x2
// view, fc1 + bias, GELU-exact, fc2 + bias) -- into the load's
// the model's embedding pool (ADR 0035). The program layer is ours (ADR 0009); every op is the ADR
// 0010 vendored one, called in the reference's order
// (`impl/runtime/vision_context_impl.h`, `VisionContext::encode`).
//
// Everything runs out of the reservations `ignis_model_load` made: the load's
// scratch arena, sized for one item of the load's item bound, for the
// intermediates, and the embedding pool for the result. No allocation
// happens here. GitHub #212: that arena is the one prefill steps use, sized
// for the larger of the two; an encode runs between prefill steps, never
// inside one, so the two never hold it at once.
//
// GitHub #320: the image model's tower (Qwen3-VL-8B, ignis_image_vision.h)
// is this program, parametrised rather than forked: the merger's output
// width and pages, several segments in one call, and the DeepStack mergers,
// which tap the residual between blocks. The 27B's call passes its own width
// and pool pages and no DeepStack, so it runs what it ran before.
//
// GitHub #414 (ADR 0075): under the host vision residency the 27B's tower
// lives in mapped pinned host memory, and an encode stages it into the same
// scratch, after its workspace, through a ring of three slots (TowerStager
// below). The ops and their order are unchanged; only the weights' addresses
// are the slots'. A device load passes no stager and runs exactly as before.
// GitHub #415: the same residency's embedding pool is host-mapped, and the
// merger's fc2 writes each page through a device page in the ring, then copies
// it there (`VisionEncodeCall::page_bounce`).

#include "ignis_image_vision.h"
#include "ignis_step.h"

#include "model_internal.h"

#include "core/layout.h"

#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;
thread_local std::string g_image_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

constexpr std::size_t kVisionWorkspaceAlignment = 256;
constexpr std::int32_t kVisionHeadDim = kVisionHidden / kVisionHeads;
constexpr float kVisionRopeTheta = 10'000.0F;
constexpr float kVisionNormEps = 1.0e-6F;
// The image model's tower takes at most this many patches (32,768 merged tokens): the W8 dispatch's
// K 4608 arm (the mergers' fc2) throws above that, after the blocks have been enqueued.
constexpr std::uint32_t kImageVisionMaxPatches = 131072;

struct VisionWorkspaceLayout {
  ninfer::TensorRegion position_ids;
  ninfer::TensorRegion cu_seqlens;
  ninfer::TensorRegion pos_indices;
  ninfer::TensorRegion pos_weights;
  ninfer::TensorRegion x;
  ninfer::TensorRegion patch_bf16;
  ninfer::TensorRegion attended;
  ninfer::TensorRegion qkv;
  ninfer::TensorRegion attention_norm;
  std::optional<ninfer::LayoutRegion> attention_workspace;
  ninfer::TensorRegion projected;
  ninfer::TensorRegion mlp_up;
  ninfer::TensorRegion mlp_norm;
  ninfer::TensorRegion mlp_down;
  ninfer::TensorRegion normalized;
  ninfer::TensorRegion merger_hidden;
  std::size_t bytes = 0;
};

// Region for region and scope for scope the reference's
// `build_workspace_layout`, with one departure: the BF16 patches get a scope
// of their own. The reference keeps them live for the whole encode, but the
// patch embedding is their only reader, so every block after it may reuse
// their bytes -- 1,536 BF16 per patch, 402,653,184 bytes at a 32,768-token
// item, on the encode's one stream.
VisionWorkspaceLayout build_workspace_layout(std::int32_t patches, std::int32_t tokens,
                                             std::int32_t segments) {
  using ninfer::DType;
  ninfer::LayoutBuilder builder;
  VisionWorkspaceLayout out;
  const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape, const char *label) {
    return builder.add_tensor(dtype, shape, kVisionWorkspaceAlignment, label);
  };
  out.position_ids = add(DType::I32, {patches, 2}, "vision position ids");
  out.cu_seqlens = add(DType::I32, {segments + 1}, "vision segment bounds");
  out.pos_indices = add(DType::I32, {4, patches}, "vision position indices");
  out.pos_weights = add(DType::FP32, {4, patches}, "vision position weights");
  out.x = add(DType::BF16, {kVisionHidden, patches}, "vision residual");
  {
    auto patch_scope = builder.scope();
    out.patch_bf16 = add(DType::BF16, {kVisionPatchDim, patches}, "vision BF16 patches");
  }
  {
    auto attention_scope = builder.scope();
    out.attended = add(DType::BF16, {kVisionHidden, patches}, "vision attended");
    {
      auto qkv_scope = builder.scope();
      out.qkv = add(DType::BF16, {3 * kVisionHidden, patches}, "vision QKV");
      {
        auto norm_scope = builder.scope();
        out.attention_norm = add(DType::BF16, {kVisionHidden, patches}, "vision attention norm");
      }
      const std::size_t attention_bytes = ninfer::ops::vision_attention_workspace_capacity_bytes(
          patches, patches, segments, segments);
      if (attention_bytes != 0) {
        out.attention_workspace =
            builder.add(attention_bytes, kVisionWorkspaceAlignment, "vision attention workspace");
      }
    }
    out.projected = add(DType::BF16, {kVisionHidden, patches}, "vision projected");
  }
  {
    auto mlp_scope = builder.scope();
    out.mlp_up = add(DType::BF16, {kVisionIntermediate, patches}, "vision MLP up");
    {
      auto norm_scope = builder.scope();
      out.mlp_norm = add(DType::BF16, {kVisionHidden, patches}, "vision MLP norm");
    }
    out.mlp_down = add(DType::BF16, {kVisionHidden, patches}, "vision MLP down");
  }
  out.normalized = add(DType::BF16, {kVisionHidden, patches}, "vision merger norm");
  out.merger_hidden = add(DType::BF16, {kVisionMergerHidden, tokens}, "vision merger hidden");
  out.bytes = builder.finish(1, "vision workspace");
  return out;
}

// A bound BF16 vector (a bias or a norm weight) as the tensor the ops take.
ninfer::Tensor bf16_vector(const ninfer::Weight &weight, std::int32_t length) {
  return ninfer::Tensor(const_cast<void *>(weight.qdata), ninfer::DType::BF16, {length});
}

void copy_host(const void *src, ninfer::Tensor &dst, cudaStream_t stream, const char *what) {
  const cudaError_t err =
      cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync(") + what + ") failed: " +
                             cudaGetErrorString(err));
  }
}

// ---------------------------------------------------------------------------
// GitHub #414 (ADR 0075): the tower staged into the prefill scratch.
// ---------------------------------------------------------------------------

// The binder gives each tensor a span of its own in the mapped allocation,
// 4 KiB aligned at both ends, in directory order: a stage's tensors are
// separated by less than two pages of padding each. A stage is copied as runs
// of tensors with only that padding between them, each run whole, so a
// tensor's planes keep their offsets from its payload; a run starts at its
// first payload aligned down to the scratch's alignment and lands at that
// alignment in its slot, so every tensor keeps its own (mod 256).
constexpr std::size_t kStagingSpanSlack = 2 * 4096 + kVisionWorkspaceAlignment;

std::size_t align_up(std::size_t bytes, std::size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

// One contiguous host range of a stage and where it lands in the stage's slot.
struct StagedRun {
  const unsigned char *host = nullptr;
  std::size_t bytes = 0;
  std::size_t offset = 0;
};

// A stage laid out in its slot: its runs, and the slot bytes they take.
struct StageCopy {
  std::vector<StagedRun> runs;
  std::size_t bytes = 0;
};

// Lay out the stage `tensors` (sorted here by host address) as runs in a
// slot. Each tensor's code plane is its payload's start and its other planes
// lie inside the payload (model.cu's `to_weight`), which is what lets a run
// carry them; a tensor that is not so is refused rather than half copied.
StageCopy plan_stage(std::vector<ninfer::Weight *> &tensors) {
  for (const ninfer::Weight *w : tensors) {
    const auto *lo = static_cast<const unsigned char *>(w->payload);
    const auto *hi = lo + w->payload_bytes;
    if (lo == nullptr || w->qdata != w->payload) {
      throw std::runtime_error("a tower tensor's code plane is not its payload's start");
    }
    for (const void *plane : {w->qhigh, w->scales}) {
      const auto *q = static_cast<const unsigned char *>(plane);
      if (q != nullptr && (q < lo || q >= hi)) {
        throw std::runtime_error("a tower tensor's plane lies outside its payload");
      }
    }
  }
  std::sort(tensors.begin(), tensors.end(),
            [](const ninfer::Weight *a, const ninfer::Weight *b) { return a->payload < b->payload; });
  StageCopy copy;
  for (const ninfer::Weight *w : tensors) {
    const auto *lo = static_cast<const unsigned char *>(w->payload);
    const auto *hi = lo + w->payload_bytes;
    if (!copy.runs.empty()) {
      StagedRun &run = copy.runs.back();
      const unsigned char *end = run.host + run.bytes;
      if (lo >= end && static_cast<std::size_t>(lo - end) < kStagingSpanSlack) {
        run.bytes = static_cast<std::size_t>(hi - run.host);
        continue;
      }
    }
    const auto *start = reinterpret_cast<const unsigned char *>(
        reinterpret_cast<std::uintptr_t>(lo) & ~static_cast<std::uintptr_t>(kVisionWorkspaceAlignment - 1));
    copy.runs.push_back({start, static_cast<std::size_t>(hi - start), 0});
  }
  for (StagedRun &run : copy.runs) {
    run.offset = align_up(copy.bytes, kVisionWorkspaceAlignment);
    copy.bytes = run.offset + run.bytes;
  }
  return copy;
}

// Point each of `tensors` at its bytes in `slot`, where `copy` lands them --
// the pointer rebase #407's `visit_weights` does for its staged tower.
void rebase_stage(const std::vector<ninfer::Weight *> &tensors, const StageCopy &copy, unsigned char *slot) {
  for (ninfer::Weight *w : tensors) {
    const auto *lo = static_cast<const unsigned char *>(w->payload);
    const auto run = std::find_if(copy.runs.begin(), copy.runs.end(), [&](const StagedRun &r) {
      return lo >= r.host && lo < r.host + r.bytes;
    });
    const auto rebase = [&](const void *p) -> const void * {
      return p == nullptr ? nullptr : slot + run->offset + (static_cast<const unsigned char *>(p) - run->host);
    };
    w->qhigh = rebase(w->qhigh);
    w->scales = rebase(w->scales);
    w->qdata = rebase(w->qdata);
    w->payload = rebase(w->payload);
  }
}

// Each stage's tensors of `weights`, in stage order.
template <class Weights> auto stage_tensors(Weights &weights) {
  using Weight = std::remove_reference_t<decltype(weights.patch_embedding)>;
  std::array<std::vector<Weight *>, kVisionStagingStages> stages;
  visit_vision_stages(weights, [&](std::size_t stage, Weight &w) { stages[stage].push_back(&w); });
  return stages;
}

void check_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("staging the vision tower: ") + what + ": " + cudaGetErrorString(err));
  }
}

// One encode's side of the staging. Stage t lands in slot t % 3: its copy
// runs on the ring's copy stream once stage t - 3 has been read from that
// slot, and the model's stream waits for it before the stage's first op. The
// encode asks the next two stages to copy before each stage computes, so two
// copies stay ahead of the compute and the merger's 25 MB fc2 is in its slot
// before the last block ends.
class TowerStager {
 public:
  // `weights` is the encode's own copy of the tower, rebased here onto the
  // ring at `ring`: IGNIS_VISION_STAGING_SLOTS slots of the staging's
  // `slot_bytes`.
  TowerStager(const VisionStaging &staging, VisionWeights &weights, unsigned char *ring) : staging_(staging) {
    auto stages = stage_tensors(weights);
    for (std::size_t s = 0; s < kVisionStagingStages; ++s) {
      copies_[s] = plan_stage(stages[s]);
      if (copies_[s].bytes > staging_.slot_bytes) {
        throw std::runtime_error("vision tower stage " + std::to_string(s) + " takes " +
                                 std::to_string(copies_[s].bytes) + " bytes, a staging slot holds " +
                                 std::to_string(staging_.slot_bytes));
      }
      rebase_stage(stages[s], copies_[s], ring + (s % IGNIS_VISION_STAGING_SLOTS) * staging_.slot_bytes);
      slots_[s] = ring + (s % IGNIS_VISION_STAGING_SLOTS) * staging_.slot_bytes;
    }
  }

  TowerStager(const TowerStager &) = delete;
  TowerStager &operator=(const TowerStager &) = delete;

  // Every copy lands before the scratch is given back: on an encode that
  // completed the model's stream has waited for all of them already, and on
  // one that threw midway this keeps a late copy out of the next step's bytes.
  ~TowerStager() { cudaStreamSynchronize(staging_.copy); }

  // Before the item's uploads: nothing may land in the scratch before the
  // work already queued on it, then the first two stages start copying.
  void begin(cudaStream_t stream) {
    check_cuda(cudaEventRecord(staging_.head, stream), "record the encode's head");
    check_cuda(cudaStreamWaitEvent(staging_.copy, staging_.head, 0), "wait for the encode's head");
    issue_through(1);
  }

  // Before stage `stage`'s first op: keep two copies ahead, then wait for its
  // own. At stage 0 the item's uploads are queued: the third slot's first copy
  // waits for them, so it does not share the link with them and the first
  // stage starts as early as with two slots.
  void before(std::size_t stage, cudaStream_t stream) {
    if (stage == 0) {
      check_cuda(cudaEventRecord(staging_.head, stream), "record the item's uploads");
    }
    issue_through(stage + IGNIS_VISION_STAGING_SLOTS - 1);
    check_cuda(cudaStreamWaitEvent(stream, staging_.copied[stage % IGNIS_VISION_STAGING_SLOTS], 0),
               "wait for a stage's copy");
  }

  // Stage `stage`'s slot in the ring.
  unsigned char *slot(std::size_t stage) const { return slots_[stage]; }

  // After stage `stage`'s last op: its slot may take the stage three on.
  void after(std::size_t stage, cudaStream_t stream) {
    check_cuda(cudaEventRecord(staging_.consumed[stage % IGNIS_VISION_STAGING_SLOTS], stream),
               "record a stage read");
  }

 private:
  // Issue every copy through stage `last` not issued yet. Stage t's slot was
  // last read by stage t - 3, whose `consumed` record precedes this call in
  // host order, so the wait names that read.
  void issue_through(std::size_t last) {
    while (issued_ < kVisionStagingStages && issued_ <= last) {
      const std::size_t t = issued_++;
      const std::size_t slot = t % IGNIS_VISION_STAGING_SLOTS;
      if (t >= IGNIS_VISION_STAGING_SLOTS) {
        check_cuda(cudaStreamWaitEvent(staging_.copy, staging_.consumed[slot], 0), "wait for a slot");
      } else if (t >= 2) {
        check_cuda(cudaStreamWaitEvent(staging_.copy, staging_.head, 0), "wait for the item's uploads");
      }
      for (const StagedRun &run : copies_[t].runs) {
        check_cuda(cudaMemcpyAsync(slots_[t] + run.offset, run.host, run.bytes, cudaMemcpyHostToDevice,
                                   staging_.copy),
                   "copy a stage");
      }
      check_cuda(cudaEventRecord(staging_.copied[slot], staging_.copy), "record a stage's copy");
    }
  }

  const VisionStaging &staging_;
  std::array<StageCopy, kVisionStagingStages> copies_{};
  std::array<unsigned char *, kVisionStagingStages> slots_{};
  std::size_t issued_ = 0;
};

// A merger: the tower's, whose LayerNorm runs over the hidden before the
// 2x2 view, or a DeepStack one (GitHub #320), whose LayerNorm runs over the
// merged width after it (transformers' `use_postshuffle_norm`).
struct VisionMergerWeights {
  ninfer::Weight fc1;
  ninfer::Weight fc1_bias;
  ninfer::Weight fc2;
  ninfer::Weight fc2_bias;
  ninfer::Weight norm_weight;
  ninfer::Weight norm_bias;
};

// One encode's inputs and outputs beside the tower's weights.
struct VisionEncodeCall {
  // The host arrays `ignis_media_encode_input` carries, over every segment.
  const void *patches = nullptr;
  const std::int32_t *position_ids = nullptr;
  const std::int32_t *cu_seqlens = nullptr;
  const std::int32_t *position_table_indices = nullptr;
  const float *position_table_weights = nullptr;
  // The merger's output: `out_hidden` wide (the model's hidden), in pages
  // of `page_columns` merged columns, in column order (GitHub #243). The
  // image model's one page is its whole output.
  std::int32_t out_hidden = 0;
  std::int32_t page_columns = 0;
  std::vector<void *> pages;
  // GitHub #415 (ADR 0075): a device page the merger's fc2 and its bias write
  // each page into first, copied to its page after on the same stream -- set
  // where the pages are host-mapped, whose in-place writes cross the bus at a
  // fraction of a device copy's speed, and `add_bias` reads them back. Null:
  // the merger writes the pages in place, as on the device.
  void *page_bounce = nullptr;
  // GitHub #320: DeepStack merger i taps the residual after block
  // `deepstack_layers[i]` into `deepstack[i]` ([out_hidden, tokens]; null:
  // it does not run). The 27B has none.
  std::int32_t deepstack_count = 0;
  const VisionMergerWeights *deepstack_mergers = nullptr;
  const std::uint32_t *deepstack_layers = nullptr;
  void *const *deepstack = nullptr;
};

// A merger's LayerNorm, its 2x2 view and fc1 + bias + GELU-exact, from the
// residual `x` into the layout's merger hidden, which it returns. Between
// blocks the layout's merger regions hold nothing live (they share bytes
// with the blocks' scopes only), so a DeepStack merger runs there too.
ninfer::Tensor merger_hidden(ninfer::Tensor &x, const VisionMergerWeights &m, bool post_shuffle,
                             const VisionWorkspaceLayout &layout,
                             const ninfer::DeviceSpan &backing, std::int32_t tokens,
                             cudaStream_t stream) {
  ninfer::Tensor normalized = layout.normalized.bind(backing);
  if (post_shuffle) {
    const ninfer::Tensor shuffled = x.view({kVisionMergerHidden, tokens});
    ninfer::Tensor out = normalized.view({kVisionMergerHidden, tokens});
    ninfer::ops::layer_norm(shuffled, bf16_vector(m.norm_weight, kVisionMergerHidden),
                            bf16_vector(m.norm_bias, kVisionMergerHidden), kVisionNormEps, out,
                            stream);
  } else {
    ninfer::ops::layer_norm(x, bf16_vector(m.norm_weight, kVisionHidden),
                            bf16_vector(m.norm_bias, kVisionHidden), kVisionNormEps, normalized,
                            stream);
  }
  const ninfer::Tensor merged = normalized.view({kVisionMergerHidden, tokens});
  ninfer::Tensor hidden = layout.merger_hidden.bind(backing);
  ninfer::ops::linear(merged, m.fc1, hidden, stream);
  ninfer::ops::add_bias(bf16_vector(m.fc1_bias, kVisionMergerHidden), hidden, stream);
  ninfer::ops::gelu(hidden, ninfer::ops::GeluMode::Exact, stream);
  return hidden;
}

// The encoder over one call, enqueued on `stream` into the merger's pages
// (GitHub #243) and the DeepStack outputs, its intermediates in `backing` as
// `layout` places them (VisionContext::encode). GitHub #414: with a
// `stager`, `w` is its rebased copy and each stage's ops wait for its slot.
void encode(const VisionWeights &w, const VisionEncodeCall &call,
            const VisionWorkspaceLayout &layout, const ninfer::DeviceSpan &backing,
            std::int32_t patches, std::int32_t tokens, cudaStream_t stream,
            TowerStager *stager = nullptr) {
  const auto before = [&](std::size_t stage) {
    if (stager != nullptr) {
      stager->before(stage, stream);
    }
  };
  const auto after = [&](std::size_t stage) {
    if (stager != nullptr) {
      stager->after(stage, stream);
    }
  };
  if (stager != nullptr) {
    stager->begin(stream);
  }
  ninfer::Tensor position_ids = layout.position_ids.bind(backing);
  ninfer::Tensor cu_seqlens = layout.cu_seqlens.bind(backing);
  ninfer::Tensor pos_indices = layout.pos_indices.bind(backing);
  ninfer::Tensor pos_weights = layout.pos_weights.bind(backing);
  copy_host(call.position_ids, position_ids, stream, "position ids");
  copy_host(call.cu_seqlens, cu_seqlens, stream, "segment bounds");
  copy_host(call.position_table_indices, pos_indices, stream, "position indices");
  copy_host(call.position_table_weights, pos_weights, stream, "position weights");

  ninfer::Tensor x = layout.x.bind(backing);
  ninfer::Tensor patch_bf16 = layout.patch_bf16.bind(backing);
  copy_host(call.patches, patch_bf16, stream, "patches");
  before(0);
  ninfer::ops::linear(patch_bf16, w.patch_embedding, x, stream);
  ninfer::ops::add_bias(bf16_vector(w.patch_embedding_bias, kVisionHidden), x, stream);
  // The artifact's table is [rows, hidden] row-major, which is the tensor
  // convention's [hidden, rows] read in place -- a view, not a transpose.
  const ninfer::Tensor position_table(const_cast<void *>(w.position_embedding.qdata),
                                      ninfer::DType::BF16,
                                      {kVisionHidden, kVisionPositionEmbeddings});
  ninfer::ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
  after(0);

  static const ninfer::ops::RopeFrequencies frequencies =
      ninfer::ops::rope_vision_frequencies(kVisionRopeTheta);
  for (std::size_t b = 0; b < w.layers.size(); ++b) {
    const VisionLayerWeights &block = w.layers[b];
    before(1 + b);
    {
      ninfer::Tensor attended = layout.attended.bind(backing);
      {
        ninfer::Tensor qkv = layout.qkv.bind(backing);
        {
          ninfer::Tensor h = layout.attention_norm.bind(backing);
          ninfer::ops::layer_norm(x, bf16_vector(block.norm1_weight, kVisionHidden),
                                  bf16_vector(block.norm1_bias, kVisionHidden), kVisionNormEps, h,
                                  stream);
          ninfer::ops::linear(h, block.qkv, qkv, stream);
        }
        ninfer::ops::add_bias(bf16_vector(block.qkv_bias, 3 * kVisionHidden), qkv, stream);
        const std::size_t plane_bytes = static_cast<std::size_t>(kVisionHidden) * 2;
        ninfer::Tensor q(qkv.data, ninfer::DType::BF16, {kVisionHeadDim, kVisionHeads, patches});
        ninfer::Tensor k(static_cast<unsigned char *>(qkv.data) + plane_bytes, ninfer::DType::BF16,
                         {kVisionHeadDim, kVisionHeads, patches});
        ninfer::Tensor v(static_cast<unsigned char *>(qkv.data) + 2 * plane_bytes,
                         ninfer::DType::BF16, {kVisionHeadDim, kVisionHeads, patches});
        q.nb[2] = qkv.nb[1];
        k.nb[2] = qkv.nb[1];
        v.nb[2] = qkv.nb[1];
        ninfer::ops::rope(position_ids, kVisionHeadDim, frequencies, q, k, stream);
        ninfer::Tensor attended_heads = attended.view({kVisionHeadDim, kVisionHeads, patches});
        const ninfer::DeviceSpan attention_backing =
            layout.attention_workspace ? layout.attention_workspace->bind(backing) : backing;
        ninfer::WorkspaceArena attention_workspace(attention_backing);
        ninfer::ops::vision_attention(q, k, v, cu_seqlens, attention_workspace, attended_heads,
                                      stream);
      }
      ninfer::Tensor projected = layout.projected.bind(backing);
      ninfer::ops::linear(attended, block.output, projected, stream);
      ninfer::ops::add_bias(bf16_vector(block.output_bias, kVisionHidden), projected, stream);
      ninfer::ops::residual_add(projected, x, stream);
    }
    {
      ninfer::Tensor down = layout.mlp_down.bind(backing);
      ninfer::Tensor up = layout.mlp_up.bind(backing);
      {
        ninfer::Tensor h = layout.mlp_norm.bind(backing);
        ninfer::ops::layer_norm(x, bf16_vector(block.norm2_weight, kVisionHidden),
                                bf16_vector(block.norm2_bias, kVisionHidden), kVisionNormEps, h,
                                stream);
        ninfer::ops::linear(h, block.fc1, up, stream);
      }
      ninfer::ops::add_bias(bf16_vector(block.fc1_bias, kVisionIntermediate), up, stream);
      ninfer::ops::gelu(up, ninfer::ops::GeluMode::Tanh, stream);
      ninfer::ops::linear(up, block.fc2, down, stream);
      ninfer::ops::add_bias(bf16_vector(block.fc2_bias, kVisionHidden), down, stream);
      ninfer::ops::residual_add(down, x, stream);
    }
    after(1 + b);
    // GitHub #320: the DeepStack mergers that tap the residual after this
    // block (transformers' `deepstack_visual_indexes`).
    for (std::int32_t i = 0; i < call.deepstack_count; ++i) {
      if (call.deepstack[i] == nullptr || call.deepstack_layers[i] != b) {
        continue;
      }
      const VisionMergerWeights &m = call.deepstack_mergers[i];
      const ninfer::Tensor hidden = merger_hidden(x, m, true, layout, backing, tokens, stream);
      ninfer::Tensor output(call.deepstack[i], ninfer::DType::BF16, {call.out_hidden, tokens});
      ninfer::ops::linear(hidden, m.fc2, output, stream);
      ninfer::ops::add_bias(bf16_vector(m.fc2_bias, call.out_hidden), output, stream);
    }
  }

  const VisionMergerWeights merger{w.merger_fc1,  w.merger_fc1_bias,   w.merger_fc2,
                                   w.merger_fc2_bias, w.merger_norm_weight, w.merger_norm_bias};
  const std::size_t merger_stage = 1 + w.layers.size();
  before(merger_stage);
  const ninfer::Tensor hidden = merger_hidden(x, merger, false, layout, backing, tokens, stream);
  after(merger_stage);
  before(merger_stage + 1);
  // GitHub #243: the merger's second projection writes the item's columns
  // straight into its pool pages, one call per page. The columns are the
  // GEMM's N, so a page is a plain column range of the same
  // `linear` + `add_bias` a single output transient took -- the pages are
  // never gathered and no kernel here reads a page table. The alternative,
  // encoding contiguously and copying into pages, would cost a second
  // envelope-sized reservation to encode into.
  for (std::size_t p = 0; p < call.pages.size(); ++p) {
    const auto first = static_cast<std::int32_t>(p) * call.page_columns;
    const std::int32_t take = std::min(call.page_columns, tokens - first);
    const ninfer::Tensor hidden_page = hidden.slice(1, first, take);
    void *const written = call.page_bounce != nullptr ? call.page_bounce : call.pages[p];
    ninfer::Tensor output(written, ninfer::DType::BF16, {call.out_hidden, take});
    ninfer::ops::linear(hidden_page, w.merger_fc2, output, stream);
    ninfer::ops::add_bias(bf16_vector(w.merger_fc2_bias, call.out_hidden), output, stream);
    if (call.page_bounce != nullptr) {
      const cudaError_t err = cudaMemcpyAsync(call.pages[p], call.page_bounce, output.bytes(), cudaMemcpyDefault, stream);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("copying a pool page from its device page: ") + cudaGetErrorString(err));
      }
    }
  }
  after(merger_stage + 1);
}

// GitHub #320: a W8G32 matrix of the image model's tower as `ops::linear`
// takes it -- what model.cu's `to_weight` makes of a bound row-split W8
// tensor, its K padded to 128 (`row-split-k128-v1`).
ninfer::Weight image_w8(const ignis_image_vision_w8 &m, std::int32_t rows, std::int32_t cols) {
  constexpr std::int32_t kRowSplitKAlignment = 128;
  const std::int32_t padded = (cols + kRowSplitKAlignment - 1) / kRowSplitKAlignment * kRowSplitKAlignment;
  ninfer::Weight w{};
  w.qtype = ninfer::QType::W8G32_F16S;
  w.layout = ninfer::QuantLayout::RowSplit;
  w.qdata = m.codes;
  w.scales = m.scales;
  w.payload = m.codes;
  w.payload_bytes = static_cast<std::uint64_t>(rows) * padded + static_cast<std::uint64_t>(rows) * (padded / 32) * 2;
  w.ndim = 2;
  w.shape[0] = rows;
  w.shape[1] = cols;
  w.padded_shape[0] = rows;
  w.padded_shape[1] = padded;
  w.n = rows;
  w.k = cols;
  w.group_size = 32;
  w.group = 32;
  w.scale_dtype = ninfer::DType::FP16;
  return w;
}

// A BF16 vector (a bias, a norm, the position table): only its data is read.
ninfer::Weight image_bf16(const void *data) {
  ninfer::Weight w{};
  w.qdata = data;
  return w;
}

VisionMergerWeights image_merger(const ignis_image_vision_merger &m) {
  return {image_w8(m.fc1, kVisionMergerHidden, kVisionMergerHidden), image_bf16(m.fc1_bias),
          image_w8(m.fc2, IGNIS_IMAGE_VISION_OUT_HIDDEN, kVisionMergerHidden), image_bf16(m.fc2_bias),
          image_bf16(m.norm_weight), image_bf16(m.norm_bias)};
}

// The tower's weights as the program reads them, every matrix through image_w8.
VisionWeights image_weights(const ignis_image_vision_weights &in) {
  VisionWeights w;
  w.patch_embedding = image_w8(in.patch_embedding, kVisionHidden, kVisionPatchDim);
  w.patch_embedding_bias = image_bf16(in.patch_embedding_bias);
  w.position_embedding = image_bf16(in.position_embedding);
  for (std::size_t b = 0; b < w.layers.size(); ++b) {
    const ignis_image_vision_block &s = in.blocks[b];
    VisionLayerWeights &d = w.layers[b];
    d.qkv = image_w8(s.qkv, 3 * kVisionHidden, kVisionHidden);
    d.qkv_bias = image_bf16(s.qkv_bias);
    d.output = image_w8(s.output, kVisionHidden, kVisionHidden);
    d.output_bias = image_bf16(s.output_bias);
    d.fc1 = image_w8(s.fc1, kVisionIntermediate, kVisionHidden);
    d.fc1_bias = image_bf16(s.fc1_bias);
    d.fc2 = image_w8(s.fc2, kVisionHidden, kVisionIntermediate);
    d.fc2_bias = image_bf16(s.fc2_bias);
    d.norm1_weight = image_bf16(s.norm1_weight);
    d.norm1_bias = image_bf16(s.norm1_bias);
    d.norm2_weight = image_bf16(s.norm2_weight);
    d.norm2_bias = image_bf16(s.norm2_bias);
  }
  const VisionMergerWeights merger = image_merger(in.merger);
  w.merger_fc1 = merger.fc1;
  w.merger_fc1_bias = merger.fc1_bias;
  w.merger_fc2 = merger.fc2;
  w.merger_fc2_bias = merger.fc2_bias;
  w.merger_norm_weight = merger.norm_weight;
  w.merger_norm_bias = merger.norm_bias;
  return w;
}

bool aligned(const void *p, std::uintptr_t alignment) {
  return (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

bool merger_complete(const ignis_image_vision_merger &m) {
  return m.fc1.codes != nullptr && m.fc1.scales != nullptr && m.fc1_bias != nullptr &&
         m.fc2.codes != nullptr && m.fc2.scales != nullptr && m.fc2_bias != nullptr &&
         m.norm_weight != nullptr && m.norm_bias != nullptr;
}

bool block_complete(const ignis_image_vision_block &b) {
  for (const ignis_image_vision_w8 *m : {&b.qkv, &b.output, &b.fc1, &b.fc2}) {
    if (m->codes == nullptr || m->scales == nullptr) {
      return false;
    }
  }
  return b.qkv_bias != nullptr && b.output_bias != nullptr && b.fc1_bias != nullptr &&
         b.fc2_bias != nullptr && b.norm1_weight != nullptr && b.norm1_bias != nullptr &&
         b.norm2_weight != nullptr && b.norm2_bias != nullptr;
}

// The image tower's argument checks: the reason for the first refusal, or empty.
std::string image_vision_refusal(const ignis_image_vision_weights *weights,
                                 const ignis_image_vision_input *input, const void *embeds,
                                 void *const *deepstack, const void *workspace,
                                 std::uint64_t workspace_bytes) {
  if (weights == nullptr || input == nullptr || embeds == nullptr || workspace == nullptr) {
    return "null argument";
  }
  if (weights->block_count != IGNIS_IMAGE_VISION_BLOCKS || weights->blocks == nullptr) {
    return "the tower has " + std::to_string(weights->block_count) + " blocks, the leaf runs " +
           std::to_string(IGNIS_IMAGE_VISION_BLOCKS);
  }
  if (weights->patch_embedding.codes == nullptr || weights->patch_embedding.scales == nullptr ||
      weights->patch_embedding_bias == nullptr || weights->position_embedding == nullptr ||
      !merger_complete(weights->merger)) {
    return "a null weight pointer";
  }
  for (std::uint32_t b = 0; b < weights->block_count; ++b) {
    if (!block_complete(weights->blocks[b])) {
      return "block " + std::to_string(b) + " has a null weight pointer";
    }
  }
  for (std::uint32_t i = 0; i < IGNIS_IMAGE_VISION_DEEPSTACK; ++i) {
    const std::uint32_t layer = weights->deepstack_layers[i];
    if (layer >= weights->block_count || (i > 0 && layer <= weights->deepstack_layers[i - 1])) {
      return "deepstack_layers must ascend below the block count";
    }
    if (deepstack != nullptr && deepstack[i] != nullptr && !merger_complete(weights->deepstack[i])) {
      return "DeepStack merger " + std::to_string(i) + " has a null weight pointer";
    }
  }
  if (input->segments == 0 || input->segments > IGNIS_IMAGE_VISION_MAX_SEGMENTS) {
    return "segments must be 1 .. " + std::to_string(IGNIS_IMAGE_VISION_MAX_SEGMENTS);
  }
  if (input->patches == 0 || input->patches % IGNIS_IMAGE_VISION_MERGE_UNIT != 0 ||
      input->patches > kImageVisionMaxPatches) {
    return "patches must be a positive multiple of " +
           std::to_string(IGNIS_IMAGE_VISION_MERGE_UNIT) + ", at most " +
           std::to_string(kImageVisionMaxPatches);
  }
  if (input->patch_rows == nullptr || input->position_ids == nullptr ||
      input->cu_seqlens == nullptr || input->position_table_indices == nullptr ||
      input->position_table_weights == nullptr) {
    return "null input array";
  }
  if (input->cu_seqlens[0] != 0 ||
      input->cu_seqlens[input->segments] != static_cast<std::int32_t>(input->patches)) {
    return "cu_seqlens must run from 0 to patches";
  }
  for (std::uint32_t s = 0; s < input->segments; ++s) {
    const std::int32_t length = input->cu_seqlens[s + 1] - input->cu_seqlens[s];
    if (length <= 0 || length % static_cast<std::int32_t>(IGNIS_IMAGE_VISION_MERGE_UNIT) != 0) {
      return "segment " + std::to_string(s) + " is not a positive multiple of " +
             std::to_string(IGNIS_IMAGE_VISION_MERGE_UNIT) + " patches";
    }
  }
  const std::uint32_t tokens = input->patches / IGNIS_IMAGE_VISION_MERGE_UNIT;
  const std::uint64_t need = ignis_image_vision_workspace_bytes(tokens, input->segments);
  if (workspace_bytes < need || !aligned(workspace, kVisionWorkspaceAlignment)) {
    return "the workspace must hold " + std::to_string(need) + " bytes, 256-byte aligned";
  }
  // The outputs and the workspace: 16-byte aligned, no two overlapping.
  const std::uint64_t out_bytes =
      static_cast<std::uint64_t>(tokens) * IGNIS_IMAGE_VISION_OUT_HIDDEN * 2;
  std::vector<std::pair<std::uintptr_t, std::uint64_t>> ranges{
      {reinterpret_cast<std::uintptr_t>(workspace), workspace_bytes},
      {reinterpret_cast<std::uintptr_t>(embeds), out_bytes}};
  for (std::uint32_t i = 0; deepstack != nullptr && i < IGNIS_IMAGE_VISION_DEEPSTACK; ++i) {
    if (deepstack[i] != nullptr) {
      ranges.emplace_back(reinterpret_cast<std::uintptr_t>(deepstack[i]), out_bytes);
    }
  }
  for (std::size_t a = 1; a < ranges.size(); ++a) {
    if (!aligned(reinterpret_cast<const void *>(ranges[a].first), 16)) {
      return "the outputs must be 16-byte aligned";
    }
    for (std::size_t b = 0; b < a; ++b) {
      if (ranges[a].first < ranges[b].first + ranges[b].second &&
          ranges[b].first < ranges[a].first + ranges[a].second) {
        return "the outputs overlap each other or the workspace";
      }
    }
  }
  return {};
}

} // namespace

std::size_t ignis_vision_workspace_bytes(std::int32_t tokens, std::int32_t segments) {
  return build_workspace_layout(tokens * kVisionMergeUnit, tokens, segments).bytes;
}

std::size_t ignis_vision_staging_slot_bytes(const VisionWeights &weights) {
  std::array<std::size_t, kVisionStagingStages> bound{};
  visit_vision_stages(weights, [&](std::size_t stage, const ninfer::Weight &w) {
    bound[stage] += w.payload_bytes + kStagingSpanSlack;
  });
  return align_up(*std::max_element(bound.begin(), bound.end()), kVisionWorkspaceAlignment);
}

void ignis_vision_staging_check(const VisionWeights &weights, std::size_t slot_bytes) {
  VisionWeights copy = weights;
  auto stages = stage_tensors(copy);
  for (std::size_t s = 0; s < kVisionStagingStages; ++s) {
    const StageCopy stage = plan_stage(stages[s]);
    if (stage.bytes > slot_bytes) {
      throw std::runtime_error("vision tower stage " + std::to_string(s) + " takes " + std::to_string(stage.bytes) +
                               " bytes, a staging slot holds " + std::to_string(slot_bytes));
    }
  }
}

extern "C" int32_t ignis_media_encode(struct ignis_model *model,
                                      const struct ignis_media_encode_input *input,
                                      struct ignis_media_embedding **out_embedding) {
  if (out_embedding != nullptr) {
    *out_embedding = nullptr;
  }
  if (model == nullptr || input == nullptr || out_embedding == nullptr) {
    set_error("ignis_media_encode: null argument");
    return -1;
  }
  if (input->size != sizeof(struct ignis_media_encode_input)) {
    set_error("ignis_media_encode: unrecognized ignis_media_encode_input size " +
              std::to_string(input->size));
    return -1;
  }
  if (!model->vision_pool.present()) {
    set_error("ignis_media_encode: the model was loaded without vision");
    return -1;
  }
  if (input->patches == nullptr || input->position_ids == nullptr || input->cu_seqlens == nullptr ||
      input->position_table_indices == nullptr || input->position_table_weights == nullptr) {
    set_error("ignis_media_encode: null input array");
    return -1;
  }
  const std::uint64_t t = input->grid_t;
  const std::uint64_t h = input->grid_h;
  const std::uint64_t w = input->grid_w;
  if (t == 0 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0) {
    set_error("ignis_media_encode: grid " + std::to_string(t) + "x" + std::to_string(h) + "x" +
              std::to_string(w) + " is not positive and merge-aligned");
    return -1;
  }
  const std::uint64_t tokens = t * h * w / kVisionMergeUnit;
  // The workspace holds one item of the load's item bound (never above the
  // envelope), so that is the bound an item is checked against.
  const std::uint64_t item_bound =
      std::min(model->vision_item_max_tokens, model->max_context_tokens);
  if (tokens > item_bound) {
    set_error("ignis_media_encode: an item of " + std::to_string(tokens) +
              " merged tokens exceeds the load's vision item bound of " + std::to_string(item_bound));
    return -1;
  }
  if (t > static_cast<std::uint64_t>(std::min<std::uint64_t>(tokens, kVisionMaxSegments))) {
    set_error("ignis_media_encode: " + std::to_string(t) + " segments exceed the load's bound");
    return -1;
  }
  try {
    const auto patches = static_cast<std::int32_t>(tokens * kVisionMergeUnit);
    const VisionWorkspaceLayout layout = build_workspace_layout(
        patches, static_cast<std::int32_t>(tokens), static_cast<std::int32_t>(t));
    // GitHub #414: a staged tower's ring follows the workspace, as the load
    // sized the scratch for (`plan_load_sizes`).
    const VisionStaging &staging = model->vision_staging;
    const std::size_t ring_offset = align_up(layout.bytes, kVisionWorkspaceAlignment);
    const std::size_t need =
        staging.staged() ? ring_offset + IGNIS_VISION_STAGING_SLOTS * staging.slot_bytes : layout.bytes;
    if (need > model->scratch->capacity()) {
      throw std::runtime_error("the item needs " + std::to_string(need) +
                               " workspace bytes, the load reserved " +
                               std::to_string(model->scratch->capacity()));
    }
    // GitHub #243: the item's pages, claimed before any kernel runs. The
    // embedding owns them, so it is built first and named as their owner.
    auto embedding = std::make_unique<ignis_media_embedding>();
    embedding->model = model;
    embedding->columns = static_cast<std::int32_t>(tokens);
    VisionEmbeddingPool &pool = model->vision_pool;
    const std::int32_t want = pool.pages_for(embedding->columns);
    embedding->pages = pool.take(want, embedding.get());
    if (embedding->pages.empty() && want > 0) {
      // The item bound check above already refused an item the pool could
      // never hold, so this is only "not right now": the caller releases
      // something and calls again.
      set_error("ignis_media_encode: an item of " + std::to_string(tokens) +
                " merged tokens needs " + std::to_string(want) + " of the pool's " +
                std::to_string(pool.pages()) + " pages, and only " +
                std::to_string(pool.free_pages()) + " are free -- release an embedding first");
      return IGNIS_MEDIA_ENCODE_POOL_FULL;
    }
    // Any exit but the successful one gives the pages back. An encoder op
    // that throws must not leave the pool holding room for an embedding
    // nobody will ever be handed -- there is no other owner to release it,
    // since only a returned handle reaches ignis_media_embedding_release.
    bool handed_over = false;
    struct PageGuard {
      VisionEmbeddingPool &pool;
      const std::vector<std::int32_t> &pages;
      const bool &handed_over;
      ~PageGuard() {
        if (!handed_over) {
          pool.give(pages);
        }
      }
    } guard{pool, embedding->pages, handed_over};
    // Held until the stream has drained: nothing else may take these bytes
    // while the encoder's kernels still run on them.
    ninfer::DeviceArena::Scope scope = model->scratch->scope();
    const ninfer::DeviceSpan backing = model->scratch->alloc_bytes(need, kVisionWorkspaceAlignment);
    // GitHub #414: the encode's own copy of the tower's weights, rebased onto
    // the ring. Destroyed before `scope`, so every copy has landed before the
    // scratch is given back.
    VisionWeights staged_weights;
    std::optional<TowerStager> stager;
    if (staging.staged()) {
      staged_weights = model->vision;
      stager.emplace(staging, staged_weights, static_cast<unsigned char *>(backing.data) + ring_offset);
    }
    VisionEncodeCall call;
    call.patches = input->patches;
    call.position_ids = input->position_ids;
    call.cu_seqlens = input->cu_seqlens;
    call.position_table_indices = input->position_table_indices;
    call.position_table_weights = input->position_table_weights;
    call.out_hidden = static_cast<std::int32_t>(model->hidden);
    call.page_columns = pool.page_columns;
    for (const std::int32_t page : embedding->pages) {
      call.pages.push_back(pool.page_ptr(page));
    }
    // GitHub #415 (ADR 0075): a host-mapped pool, which only the host
    // residency has, is written through a device page in the slot the
    // merger's first half was read from: free while its fc2 runs, since no
    // later stage lands there (the load checked a page fits a slot).
    static_assert(IGNIS_VISION_STAGING_SLOTS >= 2, "the merger's halves take two slots");
    if (pool.mapped != nullptr) {
      if (!stager) {
        throw std::runtime_error("a host-mapped embedding pool without the staged tower's ring");
      }
      call.page_bounce = stager->slot(kVisionStagingStages - 2);
    }
    encode(stager ? staged_weights : model->vision, call, layout, backing, patches,
           static_cast<std::int32_t>(tokens), model->stream, stager ? &*stager : nullptr);
    const cudaError_t err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_media_encode: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    handed_over = true;
    *out_embedding = embedding.release();
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_media_encode: ") + e.what());
    return -1;
  }
}

extern "C" uint32_t ignis_media_embedding_columns(const struct ignis_media_embedding *embedding) {
  return embedding == nullptr ? 0 : static_cast<uint32_t>(embedding->columns);
}

extern "C" int32_t ignis_media_embedding_read(const struct ignis_media_embedding *embedding, void *out,
                                              uint64_t out_bytes) {
  if (embedding == nullptr || embedding->model == nullptr || out == nullptr) {
    set_error("ignis_media_embedding_read: null argument");
    return -1;
  }
  const VisionEmbeddingPool &pool = embedding->model->vision_pool;
  const std::size_t column_bytes = static_cast<std::size_t>(embedding->model->hidden) * sizeof(std::uint16_t);
  const std::size_t bytes = static_cast<std::size_t>(embedding->columns) * column_bytes;
  if (out_bytes != bytes) {
    set_error("ignis_media_embedding_read: the embedding is " + std::to_string(bytes) + " bytes, the output " +
              std::to_string(out_bytes));
    return -1;
  }
  for (std::size_t p = 0; p < embedding->pages.size(); ++p) {
    const auto first = static_cast<std::int32_t>(p) * pool.page_columns;
    const std::int32_t take = std::min(pool.page_columns, embedding->columns - first);
    const cudaError_t err = cudaMemcpy(static_cast<unsigned char *>(out) + static_cast<std::size_t>(first) * column_bytes,
                                       pool.page_ptr(embedding->pages[p]),
                                       static_cast<std::size_t>(take) * column_bytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_media_embedding_read: ") + cudaGetErrorString(err));
      return -1;
    }
  }
  return 0;
}

extern "C" void ignis_media_embedding_release(struct ignis_media_embedding *embedding) {
  if (embedding == nullptr) {
    return;
  }
  if (embedding->model != nullptr) {
    embedding->model->vision_pool.give(embedding->pages);
  }
  delete embedding;
}

extern "C" const char *ignis_media_last_error(void) {
  return g_last_error.c_str();
}

extern "C" uint64_t ignis_image_vision_workspace_bytes(uint32_t tokens, uint32_t segments) {
  if (tokens == 0 || segments == 0 || tokens > kImageVisionMaxPatches / static_cast<std::uint32_t>(kVisionMergeUnit) ||
      segments > IGNIS_IMAGE_VISION_MAX_SEGMENTS) {
    return 0;
  }
  try {
    return build_workspace_layout(static_cast<std::int32_t>(tokens * kVisionMergeUnit),
                                  static_cast<std::int32_t>(tokens),
                                  static_cast<std::int32_t>(segments))
        .bytes;
  } catch (...) {
    return 0;
  }
}

// GitHub #320: the image model's tower is the program above with its own
// parameters -- the merger 4096 wide into one page that is the whole output,
// several images as attention segments, and the three DeepStack mergers.
extern "C" int32_t ignis_image_vision_forward(const struct ignis_image_vision_weights *weights,
                                              const struct ignis_image_vision_input *input,
                                              void *embeds, void *const *deepstack,
                                              void *workspace, uint64_t workspace_bytes,
                                              void *stream) {
  const std::string refusal =
      image_vision_refusal(weights, input, embeds, deepstack, workspace, workspace_bytes);
  if (!refusal.empty()) {
    g_image_last_error = "ignis_image_vision_forward: " + refusal;
    return -1;
  }
  try {
    const auto patches = static_cast<std::int32_t>(input->patches);
    const std::int32_t tokens = patches / kVisionMergeUnit;
    const auto segments = static_cast<std::int32_t>(input->segments);
    const VisionWorkspaceLayout layout = build_workspace_layout(patches, tokens, segments);
    const ninfer::DeviceSpan backing{workspace, layout.bytes};
    const VisionWeights w = image_weights(*weights);
    std::vector<VisionMergerWeights> mergers;
    for (const ignis_image_vision_merger &m : weights->deepstack) {
      mergers.push_back(image_merger(m));
    }
    void *const none[IGNIS_IMAGE_VISION_DEEPSTACK] = {nullptr, nullptr, nullptr};
    VisionEncodeCall call;
    call.patches = input->patch_rows;
    call.position_ids = input->position_ids;
    call.cu_seqlens = input->cu_seqlens;
    call.position_table_indices = input->position_table_indices;
    call.position_table_weights = input->position_table_weights;
    call.out_hidden = static_cast<std::int32_t>(IGNIS_IMAGE_VISION_OUT_HIDDEN);
    call.page_columns = tokens;
    call.pages.push_back(embeds);
    call.deepstack_count = static_cast<std::int32_t>(IGNIS_IMAGE_VISION_DEEPSTACK);
    call.deepstack_mergers = mergers.data();
    call.deepstack_layers = weights->deepstack_layers;
    call.deepstack = deepstack != nullptr ? deepstack : none;
    encode(w, call, layout, backing, patches, tokens, static_cast<cudaStream_t>(stream));
    return 0;
  } catch (const std::exception &e) {
    g_image_last_error = std::string("ignis_image_vision_forward: ") + e.what();
    return -1;
  }
}

extern "C" const char *ignis_image_vision_last_error(void) {
  return g_image_last_error.c_str();
}
