// ignis kernel leaf - P5-05 (GitHub #155): the DFlash2 drafter in the
// verify round -- see kernel/src/dflash2_drafter.h for the contract.

#include "dflash2_drafter.h"
#include "dflash2_small_t.h"
#include "sampled_draft.h"
#include "trellis_body.h"

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
#include <array>
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

// The append's projections over `columns` columns of `features` at `positions` (I32 `[columns]`),
// once, then each drafter layer's normalized, roped keys and its values handed to `append(layer,
// key, value)`, `[kv_width, columns]` each, which lands them in the lanes' windows.
template <class Append>
void project_and_append(ignis_model *model, ninfer::DeviceArena &arena, const ninfer::Tensor &features,
                        const ninfer::Tensor &positions, std::int32_t columns, IgnisA16Call call, Append &&append) {
  const cudaStream_t stream     = model->stream;
  const auto hidden             = static_cast<std::int32_t>(model->hidden);
  const std::int32_t head_dim   = kIgnisDflash2HeadDim;
  const std::int32_t kv_heads   = kIgnisDflash2KvHeads;
  const std::int32_t kv_width   = head_dim * kv_heads;
  const auto query_size         = static_cast<std::int32_t>(kDflash2QuerySize);
  const Dflash2Weights &weights = model->dflash2;

  ninfer::Tensor projected = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
  ignis_linear_a16(features, weights.feature_projection, projected, call, stream);
  ninfer::Tensor context = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
  ninfer::ops::rmsnorm(projected, bf16_tensor(weights.context_norm, {hidden, 1, 1, 1}),
                       kDflash2RmsEps, /*unit_offset=*/false, context, stream);

  for (std::size_t layer = 0; layer < kDflash2Layers; ++layer) {
    ninfer::DeviceArena::Scope layer_scope = arena.scope();
    const Dflash2LayerWeights &w = weights.layers[layer];
    // GitHub #694: the append keeps only the key and value rows of the q | k | v product; where
    // IGNIS_DFLASH2_APPEND_KV_ONLY takes the call it projects those rows alone, the same bytes
    // (kernel/src/dflash2_small_t.h). `key_row` is where they start in `qkv`.
    const bool key_value_only =
        ignis::dflash2_small_t::projects_key_value_only(w.query_key_value, columns, call);
    const std::int32_t key_row = key_value_only ? 0 : query_size;
    ninfer::Tensor qkv = arena.alloc(ninfer::DType::BF16, {key_row + 2 * kv_width, columns, 1, 1});
    if (key_value_only) {
      ignis::dflash2_small_t::key_value_linear(context, w.query_key_value, qkv, stream);
    } else {
      ignis_linear_a16(context, w.query_key_value, qkv, call, stream);
    }
    ninfer::Tensor key_raw = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor key     = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor value   = arena.alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    copy_bf16_rows(key_raw, qkv, key_row, stream);
    copy_bf16_rows(value, qkv, key_row + kv_width, stream);

    ninfer::Tensor key_heads = key.view({head_dim, kv_heads, columns});
    ninfer::ops::rmsnorm(key_raw.view({head_dim, kv_heads, columns}),
                         bf16_tensor(w.key_norm, {head_dim, 1, 1, 1}), kDflash2RmsEps,
                         /*unit_offset=*/false, key_heads, stream);
    ninfer::ops::rope(positions.view({columns}), head_dim, drafter_rope(), key_heads,
                      ninfer::ops::RopeSide::Key, stream);
    append(static_cast<std::uint32_t>(layer), key, value);
  }
}

} // namespace

void ignis_dflash2_append_context(ignis_model *model, ignis_seq_pool *pool,
                                  ninfer::DeviceArena &arena, const ninfer::Tensor &features,
                                  const ninfer::Tensor &positions, const ninfer::Tensor &counts,
                                  const ninfer::Tensor &lanes,
                                  ninfer::ops::KVCacheAppendPrefixExecutionEnvelope envelope,
                                  IgnisA16Call call) {
  const std::int32_t head_dim = kIgnisDflash2HeadDim;
  const std::int32_t kv_heads = kIgnisDflash2KvHeads;
  const std::int32_t width    = positions.ne[0];
  const std::int32_t batch    = positions.ne[1];
  project_and_append(model, arena, features, positions, width * batch, call,
                     [&](std::uint32_t layer, ninfer::Tensor &key, ninfer::Tensor &value) {
                       ninfer::ops::kv_cache_append_prefix(
                           key.view({head_dim, kv_heads, width, batch}), value.view({head_dim, kv_heads, width, batch}),
                           positions, counts, lanes, envelope, pool->dflash2_window->layer_view(layer), model->stream);
                     });
}

void ignis_dflash2_propose(ignis_model *model, ignis_seq_pool *pool, IgnisVerifyRound &verify,
                           const ignis::round_layout::Layout &layout, const Dflash2ProposalReadout *readout) {
  ninfer::DeviceArena &arena  = *verify.drafter_scratch;
  ninfer::DeviceArena::Scope scope = arena.scope();
  const cudaStream_t stream   = model->stream;
  const Dflash2Weights &weights = model->dflash2;
  const auto hidden           = static_cast<std::int32_t>(model->hidden);
  const auto vocab            = static_cast<std::int32_t>(model->vocab);
  const auto columns          = static_cast<std::int32_t>(layout.columns());
  const auto draft_columns    = static_cast<std::int32_t>(layout.drafts());
  const std::int32_t head_dim = kIgnisDflash2HeadDim;
  const std::int32_t kv_heads = kIgnisDflash2KvHeads;
  const std::int32_t kv_width = head_dim * kv_heads;
  const auto query_size       = static_cast<std::int32_t>(kDflash2QuerySize);
  const std::int32_t query_heads = query_size / head_dim;
  const std::int32_t top_k    = kDflash2SelectorTopK;
  const auto rank             = static_cast<std::int32_t>(weights.selector_hidden.n);

  if (readout != nullptr && !layout.uniform()) {
    throw std::invalid_argument("the drafter's readout is a uniform round's");
  }
  // GitHub #713: a group's rows of a per-lane buffer, and its view of a flat column-major tensor
  // from its first column (or draft), each in `shape`.
  using Group = ignis::round_layout::Group;
  const auto lane_rows = [](const ninfer::DeviceBuffer &buffer, const Group &group) {
    return ninfer::Tensor(static_cast<std::int32_t *>(buffer.p) + group.first_lane, ninfer::DType::I32,
                          {static_cast<std::int32_t>(group.lanes), 1, 1, 1});
  };
  const auto from = [](const ninfer::Tensor &flat, std::size_t first, std::initializer_list<std::int32_t> shape) {
    const std::size_t column_bytes = flat.nb[1];
    return ninfer::Tensor(static_cast<std::uint8_t *>(flat.data) + first * column_bytes, flat.dtype, shape);
  };

  // Each lane's query block: its anchor, then mask tokens, at `base + j`
  // (the invalid tail repeats the last valid position).
  ninfer::Tensor ids       = arena.alloc(ninfer::DType::I32, {1, columns, 1, 1});
  ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {1, columns, 1, 1});
  const auto masked_block = [&](const Group &group, cudaStream_t on) {
    const auto block = static_cast<std::int32_t>(group.lane_columns());
    const auto lanes = static_cast<std::int32_t>(group.lanes);
    ninfer::Tensor group_ids       = from(ids, group.first_column, {block, lanes, 1, 1});
    ninfer::Tensor group_positions = from(positions, group.first_column, {block, lanes, 1, 1});
    ninfer::ops::prepare_masked_block(lane_rows(*verify.anchors, group), lane_rows(*verify.base_positions, group),
                                      lane_rows(*verify.valid_columns, group), kDflash2MaskToken, group_ids,
                                      group_positions, on);
  };
  // GitHub #718: the round's first op; a mixed round's under the mixed graph's build one group after
  // the other on the round's stream, so that its graph has one root where a branch forked at the
  // capture's start would begin a second one.
  if (!layout.uniform() && model->mixed_graph_build) {
    for (const Group &group : layout.groups) {
      masked_block(group, stream);
    }
  } else {
    ignis_each_group(*model, layout, masked_block);
  }
  ninfer::Tensor residual = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
  ninfer::ops::embedding(ids.view({columns}), model->token_embedding, residual, stream);

  // The whole context the load admits: a fixed launch profile, so a captured
  // graph replays at any lane frontier (the decode graphs' own conservative
  // envelope, ADR 0019).
  const ninfer::ops::SwaContextExecutionEnvelope swa_envelope{0, model->max_context_tokens};
  // GitHub #713: the two-tap dynamic conv over each group's blocks.
  const auto dynamic_conv = [&](const ninfer::Tensor &in, const ninfer::Tensor &dynamic, const ninfer::Tensor &base,
                                std::int32_t side, ninfer::Tensor &out) {
    ignis_each_group(*model, layout, [&](const Group &group, cudaStream_t on) {
      const auto first = static_cast<std::int32_t>(group.first_column);
      const auto count = static_cast<std::int32_t>(group.columns());
      ninfer::Tensor group_out = out.slice(1, first, count);
      ninfer::ops::dflash2_dynamic_conv(in.slice(1, first, count), dynamic.slice(1, first, count), base, side,
                                        static_cast<std::int32_t>(group.lane_columns()), group_out, on);
    });
  };
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
      // GitHub #694: the conv projections and q | k | v on 4 warps a CTA at 5 to 16 columns unless
      // IGNIS_DFLASH2_SMALL_T_FILL=0 (kernel/src/dflash2_small_t.h); the same bytes.
      ignis::dflash2_small_t::forward_linear(normed, w.attention_conv_proj, dynamic, stream);
      ninfer::Tensor conv_hidden = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      dynamic_conv(normed, dynamic, attention_base, 0, conv_hidden);
      ninfer::Tensor qkv =
          arena.alloc(ninfer::DType::BF16, {query_size + 2 * kv_width, columns, 1, 1});
      ignis::dflash2_small_t::forward_linear(conv_hidden, w.query_key_value, qkv, stream);
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
      // GitHub #713: each group's window attention over its blocks; a mixed round's two on
      // workspaces of their own, so that they run side by side, a uniform round's on the arena.
      std::array<ninfer::DeviceSpan, 2> swa_workspaces{};
      if (!layout.uniform()) {
        for (std::size_t i = 0; i < layout.groups.size(); ++i) {
          const auto block = static_cast<std::int32_t>(layout.groups[i].lane_columns());
          swa_workspaces[i] = arena.alloc_bytes(std::max<std::size_t>(
              ninfer::ops::swa_workspace_capacity_bytes(swa_envelope, block, block,
                                                        static_cast<std::int32_t>(layout.groups[i].lanes)),
              1));
        }
      }
      ignis_each_group(*model, layout, [&](const Group &group, cudaStream_t on) {
        const auto block = static_cast<std::int32_t>(group.lane_columns());
        const auto lanes = static_cast<std::int32_t>(group.lanes);
        ninfer::Tensor group_attention = from(attention, group.first_column, {head_dim, query_heads, block, lanes});
        const auto attend = [&](ninfer::DeviceArena &workspace) {
          ninfer::ops::swa(from(query, group.first_column, {head_dim, query_heads, block, lanes}),
                           from(key, group.first_column, {head_dim, kv_heads, block, lanes}),
                           from(value, group.first_column, {head_dim, kv_heads, block, lanes}),
                           from(positions, group.first_column, {block, lanes, 1, 1}),
                           lane_rows(*verify.valid_columns, group), lane_rows(*model->decode_graph_slots, group),
                           kDflash2AttentionScale, pool->dflash2_window->layer_view(static_cast<std::uint32_t>(layer)),
                           swa_envelope, workspace, group_attention, on);
        };
        if (layout.uniform()) {
          attend(arena);
        } else {
          ninfer::DeviceArena workspace(swa_workspaces[group.extended ? 0 : 1]);
          attend(workspace);
        }
      });
      ninfer::Tensor projected = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::linear(attention, w.output, projected, stream);
      ninfer::Tensor conv_attention = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      dynamic_conv(projected, dynamic, attention_base, 1, conv_attention);
      ninfer::ops::residual_add(conv_attention, residual, stream);
    }
    {
      ninfer::DeviceArena::Scope mlp_scope = arena.scope();
      ninfer::Tensor normed = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      ninfer::ops::rmsnorm(residual, bf16_tensor(w.post_attention_norm, {hidden, 1, 1, 1}),
                           kDflash2RmsEps, /*unit_offset=*/false, normed, stream);
      ninfer::Tensor dynamic = arena.alloc(
          ninfer::DType::BF16, {static_cast<std::int32_t>(w.mlp_conv_proj.n), columns, 1, 1});
      ignis::dflash2_small_t::forward_linear(normed, w.mlp_conv_proj, dynamic, stream);
      ninfer::Tensor conv_hidden = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
      dynamic_conv(normed, dynamic, mlp_base, 0, conv_hidden);
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
      dynamic_conv(projected, dynamic, mlp_base, 1, conv_projected);
      ninfer::ops::residual_add(conv_projected, residual, stream);
    }
  }

  // The draft columns 1..k of every lane, packed lane-major (GitHub #713: a group's at its draft
  // base).
  ninfer::Tensor packed = arena.alloc(ninfer::DType::BF16, {hidden, draft_columns, 1, 1});
  const std::size_t element_bytes = sizeof(std::uint16_t);
  const std::size_t hidden_bytes = static_cast<std::size_t>(hidden) * element_bytes;
  ignis_each_group(*model, layout, [&](const Group &group, cudaStream_t on) {
    const std::size_t row_bytes = hidden_bytes * group.window;
    const std::size_t source_pitch = hidden_bytes * group.lane_columns();
    const auto *source = static_cast<const std::uint8_t *>(residual.data) + (group.first_column + 1) * hidden_bytes;
    auto *target = static_cast<std::uint8_t *>(packed.data) + group.first_draft * hidden_bytes;
    const cudaError_t err = cudaMemcpy2DAsync(target, row_bytes, source, source_pitch, row_bytes,
                                              static_cast<std::size_t>(group.lanes), cudaMemcpyDeviceToDevice, on);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy2DAsync(draft columns) failed: ") +
                               cudaGetErrorString(err));
    }
  });
  ninfer::Tensor proposal_hidden = arena.alloc(ninfer::DType::BF16, {hidden, draft_columns, 1, 1});
  ninfer::ops::rmsnorm(packed, bf16_tensor(weights.final_norm, {hidden, 1, 1, 1}), kDflash2RmsEps,
                       /*unit_offset=*/false, proposal_hidden, stream);

  // The proposal head over the draft columns, their top-k, and the selector
  // lattice over those candidates, walked into the round's drafts. The head
  // is the target's own unless the load bound the shortlist one, whose rows
  // are the most frequent tokens: its top-k are row indices, mapped back to
  // token ids before the selector reads them, so everything past this point
  // sees token ids either way. GitHub #339: the target's own head is the
  // trellis one on a trellis body (the shortlist head is the served Q4 one on
  // either body).
  const bool shortlist = model->proposal_token_ids != nullptr;
  const std::int32_t head_rows = shortlist ? model->proposal_head.n : vocab;
  ninfer::Tensor logits = arena.alloc(ninfer::DType::BF16, {head_rows, draft_columns, 1, 1});
  if (shortlist) {
    ninfer::ops::linear(proposal_hidden, model->proposal_head, logits, stream);
  } else {
    ignis::trellis_body::output_head(*model, proposal_hidden, logits, stream);
  }
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
  // GitHub #713: the selector's per-column operands, flat over the draft columns, each group's
  // lanes viewed from its first draft.
  ninfer::Tensor unary = arena.alloc(ninfer::DType::FP32, {top_k, draft_columns, 1, 1});
  ninfer::ops::cast_bf16_to_fp32(candidate_values, unary, stream);
  ninfer::Tensor predecessors = arena.alloc(ninfer::DType::I32, {top_k, draft_columns, 1, 1});
  const auto group_drafts = [&](const ninfer::Tensor &flat, const Group &group, std::int32_t rows) {
    return from(flat, group.first_draft, {rows, static_cast<std::int32_t>(group.window),
                                          static_cast<std::int32_t>(group.lanes), 1});
  };
  ignis_each_group(*model, layout, [&](const Group &group, cudaStream_t on) {
    ninfer::Tensor group_predecessors = group_drafts(predecessors, group, top_k);
    ninfer::ops::dflash2_selector_predecessors(group_drafts(candidate_ids, group, top_k),
                                               lane_rows(*verify.anchors, group), group_predecessors, on);
  });
  ninfer::Tensor hidden_proj = arena.alloc(ninfer::DType::BF16, {rank, draft_columns, 1, 1});
  ninfer::ops::linear(proposal_hidden, weights.selector_hidden, hidden_proj, stream);
  ninfer::Tensor hidden_proj_f32 = arena.alloc(ninfer::DType::FP32, {rank, draft_columns, 1, 1});
  ninfer::ops::cast_bf16_to_fp32(hidden_proj, hidden_proj_f32, stream);
  ninfer::Tensor scores = arena.alloc(ninfer::DType::FP32, {top_k * top_k, draft_columns, 1, 1});
  ignis_each_group(*model, layout, [&](const Group &group, cudaStream_t on) {
    const auto k = static_cast<std::int32_t>(group.window);
    const auto lanes = static_cast<std::int32_t>(group.lanes);
    const ninfer::Tensor candidates = group_drafts(candidate_ids, group, top_k);
    ninfer::Tensor group_scores = from(scores, group.first_draft, {top_k, top_k, k, lanes});
    ninfer::ops::dflash2_selector_scores(candidates, group_drafts(predecessors, group, top_k),
                                         group_drafts(unary, group, top_k), group_drafts(hidden_proj_f32, group, rank),
                                         weights.selector_successor, weights.selector_predecessor, group_scores, on);
    ninfer::Tensor drafts(static_cast<std::int32_t *>(verify.drafts->p) + group.first_draft, ninfer::DType::I32,
                          {k, lanes, 1, 1});
    ninfer::ops::dflash2_selector_walk(group_scores, candidates, drafts, on);
    // GitHub #576 (ADR 0090): under the sampled draft each sampling lane draws
    // its drafts from the drafter's distribution over the same lattice instead,
    // and keeps that distribution for the round's accept; a greedy lane keeps
    // the walk's. GitHub #577: the proposal readout draws as the round does,
    // from the lane's config it staged, so it reads the drafts the round would
    // verify; a teacher-forced round (test-only, eager) drafts along its
    // recording (`forced_tokens`, null otherwise; GitHub #713: from the
    // group's first column).
    if (verify.sampled_draft) {
      ignis_sampled_draft_draw(
          group_scores, candidates,
          static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_decode_configs->p) + group.first_lane,
          static_cast<const std::int32_t *>(verify.base_positions->p) + group.first_lane,
          verify.drafter_temperature_factor, verify.drafter_top_p, drafts,
          static_cast<std::int32_t *>(verify.draft_candidates->p) + group.first_draft * top_k,
          static_cast<float *>(verify.draft_q->p) + group.first_draft * top_k, on,
          verify.forced_tokens != nullptr ? verify.forced_tokens + group.first_column : nullptr);
    }
  });

  // GitHub #267: the measurement's readout, device to host on this stream.
  // Every tensor is contiguous with the lane outermost, so each is one copy.
  if (readout != nullptr) {
    const auto copy_out = [&](void *host, const ninfer::Tensor &from, const char *what) {
      if (host == nullptr) {
        return;
      }
      const cudaError_t copied =
          cudaMemcpyAsync(host, from.data, from.bytes(), cudaMemcpyDeviceToHost, stream);
      if (copied != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync(readout ") + what +
                                 ") failed: " + cudaGetErrorString(copied));
      }
    };
    copy_out(readout->candidate_ids, candidate_ids, "candidates");
    copy_out(readout->candidate_logits, unary, "candidate logits");
    copy_out(readout->scores, scores, "scores");
    copy_out(readout->head_logits, logits, "head logits");
    if (verify.sampled_draft) {
      // The draw's q: a sampling lane's rows; a greedy lane's are stale (the
      // readout's caller writes its own one-hot rows for one).
      const ninfer::Tensor kept_q(verify.draft_q->p, ninfer::DType::FP32, {top_k, draft_columns, 1, 1});
      copy_out(readout->q, kept_q, "drafter distribution");
    }
  }
}

void ignis_dflash2_append_round(ignis_model *model, ignis_seq_pool *pool, const IgnisVerifyRound &verify,
                                const ignis::round_layout::Layout &layout) {
  ninfer::DeviceArena &arena = *verify.drafter_scratch;
  ninfer::DeviceArena::Scope scope = arena.scope();
  const auto hidden          = static_cast<std::int32_t>(model->hidden);
  const auto taps            = static_cast<std::int32_t>(kDflash2TapLayers.size());
  const std::size_t column_bytes =
      static_cast<std::size_t>(taps) * static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
  // GitHub #713: a mixed round's projections once over its columns, then each group's keys and
  // values into its lanes' windows.
  if (!layout.uniform()) {
    const auto columns = static_cast<std::int32_t>(layout.columns());
    const ninfer::Tensor features(verify.features->p, ninfer::DType::BF16, {taps * hidden, columns, 1, 1});
    const ninfer::Tensor positions(verify.positions->p, ninfer::DType::I32, {columns, 1, 1, 1});
    const std::int32_t head_dim = kIgnisDflash2HeadDim;
    const std::int32_t kv_heads = kIgnisDflash2KvHeads;
    project_and_append(
        model, arena, features, positions, columns, IgnisA16Call::General,
        [&](std::uint32_t layer, ninfer::Tensor &key, ninfer::Tensor &value) {
          ignis_each_group(*model, layout, [&](const ignis::round_layout::Group &group, cudaStream_t on) {
            const auto block = static_cast<std::int32_t>(group.lane_columns());
            const auto batch = static_cast<std::int32_t>(group.lanes);
            const auto first = static_cast<std::int32_t>(group.first_column);
            const auto count = static_cast<std::int32_t>(group.columns());
            ninfer::ops::kv_cache_append_prefix(
                key.slice(1, first, count).view({head_dim, kv_heads, block, batch}),
                value.slice(1, first, count).view({head_dim, kv_heads, block, batch}),
                ninfer::Tensor(static_cast<std::int32_t *>(verify.positions->p) + group.first_column,
                               ninfer::DType::I32, {block, batch, 1, 1}),
                ninfer::Tensor(static_cast<std::int32_t *>(verify.append_counts->p) + group.first_lane,
                               ninfer::DType::I32, {batch, 1, 1, 1}),
                ninfer::Tensor(static_cast<std::int32_t *>(model->decode_graph_slots->p) + group.first_lane,
                               ninfer::DType::I32, {batch, 1, 1, 1}),
                {0, static_cast<std::uint32_t>(block)}, pool->dflash2_window->layer_view(layer), on);
          });
        });
    return;
  }
  for (const ignis::round_layout::Group &group : layout.groups) {
    if (group.lanes == 0) {
      continue;
    }
    const auto block = static_cast<std::int32_t>(group.lane_columns());
    const auto batch = static_cast<std::int32_t>(group.lanes);
    const ninfer::Tensor features(static_cast<std::uint8_t *>(verify.features->p) + group.first_column * column_bytes,
                                  ninfer::DType::BF16, {taps * hidden, block * batch, 1, 1});
    const ninfer::Tensor positions(static_cast<std::int32_t *>(verify.positions->p) + group.first_column,
                                   ninfer::DType::I32, {block, batch, 1, 1});
    const ninfer::Tensor counts(static_cast<std::int32_t *>(verify.append_counts->p) + group.first_lane,
                                ninfer::DType::I32, {batch, 1, 1, 1});
    const ninfer::Tensor lanes(static_cast<std::int32_t *>(model->decode_graph_slots->p) + group.first_lane,
                               ninfer::DType::I32, {batch, 1, 1, 1});
    ignis_dflash2_append_context(model, pool, arena, features, positions, counts, lanes,
                                 {0, static_cast<std::uint32_t>(block)}, IgnisA16Call::General);
  }
}
