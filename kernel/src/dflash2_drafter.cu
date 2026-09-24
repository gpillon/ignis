// ignis kernel leaf - P5-05 (GitHub #155): the DFlash2 drafter in the
// verify round -- see kernel/src/dflash2_drafter.h for the contract.

#include "dflash2_drafter.h"

#include "ignis_dflash2_topk.h"

#include "ninfer/ops/cast.h"
#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_selector_predecessors.h"
#include "ninfer/ops/dflash2_selector_scores.h"
#include "ninfer/ops/dflash2_selector_walk.h"
#include "ninfer/ops/dflash2_topk.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace {

// A BF16 weight (a norm, a conv base) as the tensor the ops take.
ninfer::Tensor bf16_tensor(const ninfer::Weight &weight, std::initializer_list<std::int32_t> shape) {
  return ninfer::Tensor(const_cast<void *>(weight.qdata), ninfer::DType::BF16, shape);
}

// `dst` ([rows, T]) receives rows [first_row, first_row + rows) of every
// column of `src` ([*, T]), device to device, in one strided copy.
void copy_bf16_rows(ninfer::Tensor &dst, const ninfer::Tensor &src, std::int32_t first_row,
                    cudaStream_t stream) {
  const auto *from = static_cast<const std::uint8_t *>(src.data) +
                     static_cast<std::size_t>(first_row) * sizeof(std::uint16_t);
  const cudaError_t err = cudaMemcpy2DAsync(
      dst.data, static_cast<std::size_t>(dst.nb[1]), from, static_cast<std::size_t>(src.nb[1]),
      static_cast<std::size_t>(dst.ne[0]) * sizeof(std::uint16_t),
      static_cast<std::size_t>(dst.ne[1]), cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy2DAsync(bf16 rows) failed: ") +
                             cudaGetErrorString(err));
  }
}

// The drafter's own unscaled base-1e7 RoPE table, shared by the context
// append (keys) and the forward (queries and keys).
const ninfer::ops::RopeFrequencies &drafter_rope() {
  static const ninfer::ops::RopeFrequencies frequencies =
      ninfer::ops::rope_linear_frequencies(kDflash2RopeTheta, kIgnisDflash2HeadDim);
  return frequencies;
}

// The NVFP4 A16 `linear_swiglu` route is registered through T=16 (the
// reference slices the drafter's MLP the same way, dflash2_impl.h); the
// drafter's block carries (k+1) x width columns, up to 64.
constexpr std::int32_t kSwiGluColumnLimit = 16;

} // namespace

void ignis_dflash2_append_context(ignis_model *model, ignis_seq_pool *pool,
                                  ninfer::DeviceArena &arena, const ninfer::Tensor &features,
                                  const ninfer::Tensor &positions, const ninfer::Tensor &counts,
                                  const ninfer::Tensor &lanes,
                                  ninfer::ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
  const cudaStream_t stream     = model->stream;
  const auto hidden             = static_cast<std::int32_t>(model->hidden);
  const std::int32_t head_dim   = kIgnisDflash2HeadDim;
  const std::int32_t kv_heads   = kIgnisDflash2KvHeads;
  const std::int32_t kv_width   = head_dim * kv_heads;
  const auto query_size         = static_cast<std::int32_t>(kDflash2QuerySize);
  const std::int32_t width      = positions.ne[0];
  const std::int32_t batch      = positions.ne[1];
  const std::int32_t columns    = width * batch;
  const Dflash2Weights &weights = model->dflash2;

  ninfer::Tensor projected = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
  ninfer::ops::linear(features, weights.feature_projection, projected, stream);
  ninfer::Tensor context = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
  ninfer::ops::rmsnorm(projected, bf16_tensor(weights.context_norm, {hidden, 1, 1, 1}),
                       kDflash2RmsEps, /*unit_offset=*/false, context, stream);

  for (std::size_t layer = 0; layer < kDflash2Layers; ++layer) {
    ninfer::DeviceArena::Scope layer_scope = arena.scope();
    const Dflash2LayerWeights &w = weights.layers[layer];
    ninfer::Tensor qkv = arena.alloc(ninfer::DType::BF16, {query_size + 2 * kv_width, columns, 1, 1});
    ninfer::ops::linear(context, w.query_key_value, qkv, stream);
    ninfer::Tensor key_raw = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor key     = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor value   = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    copy_bf16_rows(key_raw, qkv, query_size, stream);
    copy_bf16_rows(value, qkv, query_size + kv_width, stream);

    ninfer::Tensor key_heads = key.view({head_dim, kv_heads, columns});
    ninfer::ops::rmsnorm(key_raw.view({head_dim, kv_heads, columns}),
                         bf16_tensor(w.key_norm, {head_dim, 1, 1, 1}), kDflash2RmsEps,
                         /*unit_offset=*/false, key_heads, stream);
    ninfer::ops::rope(positions.view({columns}), head_dim, drafter_rope(), key_heads,
                      ninfer::ops::RopeSide::Key, stream);
    ninfer::ops::kv_cache_append_prefix(
        key.view({head_dim, kv_heads, width, batch}), value.view({head_dim, kv_heads, width, batch}),
        positions, counts, lanes, envelope,
        pool->dflash2_window->layer_view(static_cast<std::uint32_t>(layer)), stream);
  }
}

void ignis_dflash2_propose(ignis_model *model, ignis_seq_pool *pool, std::uint32_t width) {
  IgnisVerifyRound &verify    = *model->verify;
  ninfer::DeviceArena &arena  = *verify.drafter_scratch;
  ninfer::DeviceArena::Scope scope = arena.scope();
  const cudaStream_t stream   = model->stream;
  const Dflash2Weights &weights = model->dflash2;
  const auto hidden           = static_cast<std::int32_t>(model->hidden);
  const auto vocab            = static_cast<std::int32_t>(model->vocab);
  const auto k                = static_cast<std::int32_t>(verify.window);
  const std::int32_t block    = k + 1;
  const auto batch            = static_cast<std::int32_t>(width);
  const std::int32_t columns  = block * batch;
  const std::int32_t draft_columns = k * batch;
  const std::int32_t head_dim = kIgnisDflash2HeadDim;
  const std::int32_t kv_heads = kIgnisDflash2KvHeads;
  const std::int32_t kv_width = head_dim * kv_heads;
  const auto query_size       = static_cast<std::int32_t>(kDflash2QuerySize);
  const std::int32_t query_heads = query_size / head_dim;
  const std::int32_t top_k    = kDflash2SelectorTopK;
  const auto rank             = static_cast<std::int32_t>(weights.selector_hidden.n);

  const ninfer::Tensor anchors(verify.anchors->p, ninfer::DType::I32, {batch, 1, 1, 1});
  const ninfer::Tensor lengths(verify.base_positions->p, ninfer::DType::I32, {batch, 1, 1, 1});
  const ninfer::Tensor valid_columns(verify.valid_columns->p, ninfer::DType::I32, {batch, 1, 1, 1});
  const ninfer::Tensor lanes(model->decode_graph_slots->p, ninfer::DType::I32, {batch, 1, 1, 1});

  // Each lane's query block: its anchor, then mask tokens, at `base + j`
  // (the invalid tail repeats the last valid position).
  ninfer::Tensor ids       = arena.alloc(ninfer::DType::I32, {block, batch, 1, 1});
  ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {block, batch, 1, 1});
  ninfer::ops::prepare_masked_block(anchors, lengths, valid_columns, kDflash2MaskToken, ids,
                                    positions, stream);
  ninfer::Tensor residual = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
  ninfer::ops::embedding(ids.view({columns}), model->token_embedding, residual, stream);

  // The whole context the load admits: a fixed launch profile, so a captured
  // graph replays at any lane frontier (the decode graphs' own conservative
  // envelope, ADR 0019).
  const ninfer::ops::SwaContextExecutionEnvelope swa_envelope{0, model->max_context_tokens};
  for (std::size_t layer = 0; layer < kDflash2Layers; ++layer) {
    const Dflash2LayerWeights &w = weights.layers[layer];
    const ninfer::Tensor attention_base = bf16_tensor(w.attention_conv_base, {2, 2, hidden, 1});
    const ninfer::Tensor mlp_base       = bf16_tensor(w.mlp_conv_base, {2, 2, hidden, 1});
    {
      ninfer::DeviceArena::Scope attention_scope = arena.scope();
      ninfer::Tensor normed = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::rmsnorm(residual, bf16_tensor(w.input_norm, {hidden, 1, 1, 1}), kDflash2RmsEps,
                           /*unit_offset=*/false, normed, stream);
      ninfer::Tensor dynamic = arena.alloc(
          ninfer::DType::BF16, {static_cast<std::int32_t>(w.attention_conv_proj.n), columns, 1, 1});
      ninfer::ops::linear(normed, w.attention_conv_proj, dynamic, stream);
      ninfer::Tensor conv_hidden = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::dflash2_dynamic_conv(normed, dynamic, attention_base, 0, block, conv_hidden,
                                        stream);
      ninfer::Tensor qkv =
          arena.alloc(ninfer::DType::BF16, {query_size + 2 * kv_width, columns, 1, 1});
      ninfer::ops::linear(conv_hidden, w.query_key_value, qkv, stream);
      ninfer::Tensor query_raw = arena.alloc(ninfer::DType::BF16, {query_size, columns, 1, 1});
      ninfer::Tensor key_raw   = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
      ninfer::Tensor value     = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
      copy_bf16_rows(query_raw, qkv, 0, stream);
      copy_bf16_rows(key_raw, qkv, query_size, stream);
      copy_bf16_rows(value, qkv, query_size + kv_width, stream);
      ninfer::Tensor query = arena.alloc(ninfer::DType::BF16, {query_size, columns, 1, 1});
      ninfer::Tensor key   = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
      ninfer::Tensor query_normed = query.view({head_dim, query_heads, columns});
      ninfer::ops::rmsnorm(query_raw.view({head_dim, query_heads, columns}),
                           bf16_tensor(w.query_norm, {head_dim, 1, 1, 1}), kDflash2RmsEps,
                           /*unit_offset=*/false, query_normed, stream);
      ninfer::Tensor key_normed = key.view({head_dim, kv_heads, columns});
      ninfer::ops::rmsnorm(key_raw.view({head_dim, kv_heads, columns}),
                           bf16_tensor(w.key_norm, {head_dim, 1, 1, 1}), kDflash2RmsEps,
                           /*unit_offset=*/false, key_normed, stream);
      ninfer::ops::rope(positions.view({columns}), head_dim, drafter_rope(), query_normed,
                        key_normed, stream);
      ninfer::Tensor attention = arena.alloc(ninfer::DType::BF16, {query_size, columns, 1, 1});
      ninfer::Tensor attention_blocks = attention.view({head_dim, query_heads, block, batch});
      ninfer::ops::swa(query.view({head_dim, query_heads, block, batch}),
                       key.view({head_dim, kv_heads, block, batch}),
                       value.view({head_dim, kv_heads, block, batch}), positions, valid_columns,
                       lanes, kDflash2AttentionScale,
                       pool->dflash2_window->layer_view(static_cast<std::uint32_t>(layer)),
                       swa_envelope, arena, attention_blocks, stream);
      ninfer::Tensor projected = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::linear(attention, w.output, projected, stream);
      ninfer::Tensor conv_attention = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::dflash2_dynamic_conv(projected, dynamic, attention_base, 1, block, conv_attention,
                                        stream);
      ninfer::ops::residual_add(conv_attention, residual, stream);
    }
    {
      ninfer::DeviceArena::Scope mlp_scope = arena.scope();
      ninfer::Tensor normed = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::rmsnorm(residual, bf16_tensor(w.post_attention_norm, {hidden, 1, 1, 1}),
                           kDflash2RmsEps, /*unit_offset=*/false, normed, stream);
      ninfer::Tensor dynamic = arena.alloc(
          ninfer::DType::BF16, {static_cast<std::int32_t>(w.mlp_conv_proj.n), columns, 1, 1});
      ninfer::ops::linear(normed, w.mlp_conv_proj, dynamic, stream);
      ninfer::Tensor conv_hidden = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::dflash2_dynamic_conv(normed, dynamic, mlp_base, 0, block, conv_hidden, stream);
      const auto intermediate_rows = static_cast<std::int32_t>(w.mlp_down.k);
      ninfer::Tensor intermediate =
          arena.alloc(ninfer::DType::BF16, {intermediate_rows, columns, 1, 1});
      for (std::int32_t begin = 0; begin < columns; begin += kSwiGluColumnLimit) {
        const std::int32_t span = std::min(kSwiGluColumnLimit, columns - begin);
        const ninfer::Tensor gate_up_in = conv_hidden.slice(1, begin, span);
        ninfer::Tensor gate_up_out      = intermediate.slice(1, begin, span);
        ninfer::ops::linear_swiglu(gate_up_in, w.mlp_gate_up, gate_up_out,
                                   ninfer::ops::LinearPolicy::A16Only, arena, stream);
      }
      ninfer::Tensor projected = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::linear(intermediate, w.mlp_down, projected, stream);
      ninfer::Tensor conv_projected = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::dflash2_dynamic_conv(projected, dynamic, mlp_base, 1, block, conv_projected,
                                        stream);
      ninfer::ops::residual_add(conv_projected, residual, stream);
    }
  }

  // The draft columns 1..k of every lane, packed lane-major.
  ninfer::Tensor packed = arena.alloc(ninfer::DType::BF16, {hidden, draft_columns, 1, 1});
  const std::size_t element_bytes = sizeof(std::uint16_t);
  const std::size_t row_bytes = static_cast<std::size_t>(hidden) * k * element_bytes;
  const std::size_t source_pitch = static_cast<std::size_t>(hidden) * block * element_bytes;
  const auto *source = static_cast<const std::uint8_t *>(residual.data) +
                       static_cast<std::size_t>(hidden) * element_bytes;
  const cudaError_t err = cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                            static_cast<std::size_t>(batch),
                                            cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy2DAsync(draft columns) failed: ") +
                             cudaGetErrorString(err));
  }
  ninfer::Tensor proposal_hidden = arena.alloc(ninfer::DType::BF16, {hidden, draft_columns, 1, 1});
  ninfer::ops::rmsnorm(packed, bf16_tensor(weights.final_norm, {hidden, 1, 1, 1}), kDflash2RmsEps,
                       /*unit_offset=*/false, proposal_hidden, stream);

  // The proposal head over the draft columns, their top-k, and the selector
  // lattice over those candidates, walked into the round's drafts. The head
  // is the target's own unless the load bound the shortlist one, whose rows
  // are the most frequent tokens: its top-k are row indices, mapped back to
  // token ids before the selector reads them, so everything past this point
  // sees token ids either way.
  const bool shortlist = model->proposal_token_ids != nullptr;
  const ninfer::Weight &head = shortlist ? model->proposal_head : model->output_head;
  const std::int32_t head_rows = shortlist ? head.n : vocab;
  ninfer::Tensor logits = arena.alloc(ninfer::DType::BF16, {head_rows, draft_columns, 1, 1});
  ninfer::ops::linear(proposal_hidden, head, logits, stream);
  ninfer::Tensor candidate_ids    = arena.alloc(ninfer::DType::I32, {top_k, draft_columns, 1, 1});
  ninfer::Tensor candidate_values = arena.alloc(ninfer::DType::BF16, {top_k, draft_columns, 1, 1});
  // Ours rather than the vendored op (kernel/include/ignis_dflash2_topk.h): the
  // vendored one gives a single warp to a column and spends 3.1 ms of every
  // decode round selecting 16 rows of 248,046. Same contract and the same
  // answer bit-for-bit; a shape it does not specialize it forwards.
  const std::size_t topk_workspace_bytes =
      ignis_dflash2_topk_workspace_bytes(head_rows, draft_columns, top_k);
  const ninfer::DeviceSpan topk_workspace =
      arena.alloc_bytes(std::max<std::size_t>(topk_workspace_bytes, 1));
  ignis_dflash2_topk(logits, top_k, candidate_ids, candidate_values, topk_workspace.data,
                     topk_workspace.bytes, stream);
  if (shortlist) {
    // The op takes one contiguous vector: every candidate of every column.
    ninfer::Tensor flat_ids = candidate_ids.view({top_k * draft_columns});
    ninfer::ops::proposal_remap_token_ids(flat_ids, model->proposal_token_ids, head_rows, stream);
  }
  const ninfer::Tensor candidates = candidate_ids.view({top_k, k, batch});
  ninfer::Tensor unary = arena.alloc(ninfer::DType::FP32, {top_k, k, batch, 1});
  ninfer::ops::cast_bf16_to_fp32(candidate_values.view({top_k, k, batch}), unary, stream);
  ninfer::Tensor predecessors = arena.alloc(ninfer::DType::I32, {top_k, k, batch, 1});
  ninfer::ops::dflash2_selector_predecessors(candidates, anchors, predecessors, stream);
  ninfer::Tensor hidden_proj = arena.alloc(ninfer::DType::BF16, {rank, draft_columns, 1, 1});
  ninfer::ops::linear(proposal_hidden, weights.selector_hidden, hidden_proj, stream);
  ninfer::Tensor hidden_proj_f32 = arena.alloc(ninfer::DType::FP32, {rank, k, batch, 1});
  ninfer::ops::cast_bf16_to_fp32(hidden_proj.view({rank, k, batch}), hidden_proj_f32, stream);
  ninfer::Tensor scores = arena.alloc(ninfer::DType::FP32, {top_k, top_k, k, batch});
  ninfer::ops::dflash2_selector_scores(candidates, predecessors, unary, hidden_proj_f32,
                                       weights.selector_successor, weights.selector_predecessor,
                                       scores, stream);
  ninfer::Tensor drafts(verify.drafts->p, ninfer::DType::I32, {k, batch, 1, 1});
  ninfer::ops::dflash2_selector_walk(scores, candidates, drafts, stream);
}

void ignis_dflash2_append_round(ignis_model *model, ignis_seq_pool *pool, std::uint32_t width) {
  IgnisVerifyRound &verify   = *model->verify;
  ninfer::DeviceArena &arena = *verify.drafter_scratch;
  ninfer::DeviceArena::Scope scope = arena.scope();
  const auto hidden          = static_cast<std::int32_t>(model->hidden);
  const auto block           = static_cast<std::int32_t>(verify.window + 1);
  const auto batch           = static_cast<std::int32_t>(width);
  const auto taps            = static_cast<std::int32_t>(kDflash2TapLayers.size());
  const ninfer::Tensor features(verify.features->p, ninfer::DType::BF16,
                                {taps * hidden, block * batch, 1, 1});
  const ninfer::Tensor positions(verify.positions->p, ninfer::DType::I32, {block, batch, 1, 1});
  const ninfer::Tensor counts(verify.append_counts->p, ninfer::DType::I32, {batch, 1, 1, 1});
  const ninfer::Tensor lanes(model->decode_graph_slots->p, ninfer::DType::I32, {batch, 1, 1, 1});
  ignis_dflash2_append_context(model, pool, arena, features, positions, counts, lanes,
                               {0, static_cast<std::uint32_t>(block)});
}
