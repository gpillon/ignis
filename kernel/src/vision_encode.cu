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
// Everything runs out of the reservations `ignis_model_load` made for the
// envelope: the load's scratch arena for the intermediates, the output
// transient for the result. No allocation happens here. GitHub #212: that
// arena is the one prefill steps use, sized for the larger of the two; an
// encode runs between prefill steps, never inside one, so the two never
// hold it at once.

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
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

constexpr std::size_t kVisionWorkspaceAlignment = 256;
constexpr std::int32_t kVisionHeadDim = kVisionHidden / kVisionHeads;
constexpr float kVisionRopeTheta = 10'000.0F;
constexpr float kVisionNormEps = 1.0e-6F;

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
// `build_workspace_layout`.
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
  out.patch_bf16 = add(DType::BF16, {kVisionPatchDim, patches}, "vision BF16 patches");
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

// The encoder over one item, enqueued on the model's stream into the pool
// pages `pages` names (GitHub #243), its intermediates in `backing` as
// `layout` places them (VisionContext::encode).
void encode(ignis_model &model, const ignis_media_encode_input &input,
            const VisionWorkspaceLayout &layout, const ninfer::DeviceSpan &backing,
            std::int32_t patches, std::int32_t tokens,
            const std::vector<std::int32_t> &pages) {
  const VisionWeights &w = model.vision;
  const cudaStream_t stream = model.stream;

  ninfer::Tensor position_ids = layout.position_ids.bind(backing);
  ninfer::Tensor cu_seqlens = layout.cu_seqlens.bind(backing);
  ninfer::Tensor pos_indices = layout.pos_indices.bind(backing);
  ninfer::Tensor pos_weights = layout.pos_weights.bind(backing);
  copy_host(input.position_ids, position_ids, stream, "position ids");
  copy_host(input.cu_seqlens, cu_seqlens, stream, "segment bounds");
  copy_host(input.position_table_indices, pos_indices, stream, "position indices");
  copy_host(input.position_table_weights, pos_weights, stream, "position weights");

  ninfer::Tensor x = layout.x.bind(backing);
  ninfer::Tensor patch_bf16 = layout.patch_bf16.bind(backing);
  copy_host(input.patches, patch_bf16, stream, "patches");
  ninfer::ops::linear(patch_bf16, w.patch_embedding, x, stream);
  ninfer::ops::add_bias(bf16_vector(w.patch_embedding_bias, kVisionHidden), x, stream);
  // The artifact's table is [rows, hidden] row-major, which is the tensor
  // convention's [hidden, rows] read in place -- a view, not a transpose.
  const ninfer::Tensor position_table(const_cast<void *>(w.position_embedding.qdata),
                                      ninfer::DType::BF16,
                                      {kVisionHidden, kVisionPositionEmbeddings});
  ninfer::ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);

  static const ninfer::ops::RopeFrequencies frequencies =
      ninfer::ops::rope_vision_frequencies(kVisionRopeTheta);
  for (const VisionLayerWeights &block : w.layers) {
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
  }

  ninfer::Tensor normalized = layout.normalized.bind(backing);
  ninfer::ops::layer_norm(x, bf16_vector(w.merger_norm_weight, kVisionHidden),
                          bf16_vector(w.merger_norm_bias, kVisionHidden), kVisionNormEps, normalized,
                          stream);
  const ninfer::Tensor merged = normalized.view({kVisionMergerHidden, tokens});
  ninfer::Tensor hidden = layout.merger_hidden.bind(backing);
  ninfer::ops::linear(merged, w.merger_fc1, hidden, stream);
  ninfer::ops::add_bias(bf16_vector(w.merger_fc1_bias, kVisionMergerHidden), hidden, stream);
  ninfer::ops::gelu(hidden, ninfer::ops::GeluMode::Exact, stream);
  const auto out_hidden = static_cast<std::int32_t>(model.hidden);
  // GitHub #243: the merger's second projection writes the item's columns
  // straight into its pool pages, one call per page. The columns are the
  // GEMM's N, so a page is a plain column range of the same
  // `linear` + `add_bias` a single output transient took -- the pages are
  // never gathered and no kernel here reads a page table. The alternative,
  // encoding contiguously and copying into pages, would cost a second
  // envelope-sized reservation to encode into.
  const VisionEmbeddingPool &pool = model.vision_pool;
  for (std::size_t p = 0; p < pages.size(); ++p) {
    const auto first = static_cast<std::int32_t>(p) * pool.page_columns;
    const std::int32_t take = std::min(pool.page_columns, tokens - first);
    const ninfer::Tensor hidden_page = hidden.slice(1, first, take);
    ninfer::Tensor output(pool.page_ptr(pages[p]), ninfer::DType::BF16, {out_hidden, take});
    ninfer::ops::linear(hidden_page, w.merger_fc2, output, stream);
    ninfer::ops::add_bias(bf16_vector(w.merger_fc2_bias, out_hidden), output, stream);
  }
}

} // namespace

std::size_t ignis_vision_workspace_bytes(std::int32_t tokens, std::int32_t segments) {
  return build_workspace_layout(tokens * kVisionMergeUnit, tokens, segments).bytes;
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
  const std::uint64_t envelope = std::min(model->vision_max_tokens, model->max_context_tokens);
  if (tokens > envelope) {
    set_error("ignis_media_encode: an item of " + std::to_string(tokens) +
              " merged tokens exceeds the load's vision envelope of " + std::to_string(envelope));
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
    if (layout.bytes > model->scratch->capacity()) {
      throw std::runtime_error("the item needs " + std::to_string(layout.bytes) +
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
      // The envelope check above already refused an item the pool could
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
    const ninfer::DeviceSpan backing =
        model->scratch->alloc_bytes(layout.bytes, kVisionWorkspaceAlignment);
    encode(*model, *input, layout, backing, patches, static_cast<std::int32_t>(tokens),
           embedding->pages);
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
