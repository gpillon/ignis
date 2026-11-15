// ignis kernel leaf - GitHub #620 (spec #618): the Qwen 3.8-27B's MTP head --
// OURS. What it computes and which rows each step writes: mtp_head.h.

#include "mtp_head.h"

#include "hq_decode_attention.h"
#include "ignis_gqa_workspace.h"
#include "ignis_hq_prompt_attention.h"
#include "ignis_seq_internal.h"
#include "model_internal.h"
#include "trellis_body.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/qk_norm_rope.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/silu_mul.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::mtp {
namespace {

// The scratch arenas' allocation alignment (`kArenaAlign`, kernel/src/model.cu):
// every buffer below starts on a block, so it takes its bytes rounded up.
constexpr std::size_t kAlign = 256;

std::size_t aligned(std::size_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

std::size_t bf16_bytes(std::int64_t elements) {
  return aligned(static_cast<std::size_t>(elements) * sizeof(std::uint16_t));
}

std::size_t i32_bytes(std::int64_t elements) {
  return aligned(static_cast<std::size_t>(elements) * sizeof(std::int32_t));
}

void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
  }
}

ninfer::Tensor gain(const ninfer::Weight &weight, std::int32_t width) {
  return ninfer::Tensor(const_cast<void *>(weight.qdata), ninfer::DType::BF16, {width, 1, 1, 1});
}

// `columns` columns of BF16 [width, ...] starting at column `first`.
ninfer::Tensor columns_of(const ninfer::Tensor &matrix, std::int32_t width, std::int32_t first,
                          std::int32_t columns) {
  auto *base = static_cast<unsigned char *>(matrix.data) +
               static_cast<std::size_t>(first) * static_cast<std::size_t>(width) * sizeof(std::uint16_t);
  return ninfer::Tensor(base, ninfer::DType::BF16, {width, columns, 1, 1});
}

// Rows [first, first + count) of every one of `columns` columns of `from`
// (BF16 [from_rows, columns]) into the contiguous `to` (BF16 [count, columns]).
void copy_rows(const ninfer::Tensor &from, std::int32_t from_rows, std::int32_t first, ninfer::Tensor &to,
               std::int32_t count, std::int32_t columns, cudaStream_t stream) {
  check(cudaMemcpy2DAsync(to.data, static_cast<std::size_t>(count) * sizeof(std::uint16_t),
                          static_cast<const unsigned char *>(from.data) +
                              static_cast<std::size_t>(first) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(from_rows) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(count) * sizeof(std::uint16_t), static_cast<std::size_t>(columns),
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync(the head's split)");
}

// `count` rows into rows [first, first + count) of every column of `to`
// (BF16 [to_rows, columns]).
void copy_into_rows(const ninfer::Tensor &from, std::int32_t count, ninfer::Tensor &to, std::int32_t to_rows,
                    std::int32_t first, std::int32_t columns, cudaStream_t stream) {
  check(cudaMemcpy2DAsync(static_cast<unsigned char *>(to.data) + static_cast<std::size_t>(first) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(to_rows) * sizeof(std::uint16_t), from.data,
                          static_cast<std::size_t>(count) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(count) * sizeof(std::uint16_t), static_cast<std::size_t>(columns),
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync(the head's stem input)");
}

// out[b] = in[b] - 1: the head's row of a lane that consumes position in[b].
__global__ void previous_positions_kernel(const std::int32_t *in, std::int32_t *out, std::int32_t count) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    out[i] = in[i] - 1;
  }
}

// out[:, b] = table[:, slots[b]]: each lane's continuation hidden, by its
// slot read on the device (graph-safe).
__global__ void gather_columns_kernel(const std::uint16_t *table, std::int32_t rows, const std::int32_t *slots,
                                      std::uint16_t *out, std::int32_t columns) {
  const std::int32_t b = static_cast<std::int32_t>(blockIdx.y);
  if (b >= columns) {
    return;
  }
  const std::uint16_t *column = table + static_cast<std::size_t>(slots[b]) * static_cast<std::size_t>(rows);
  std::uint16_t *to = out + static_cast<std::size_t>(b) * static_cast<std::size_t>(rows);
  for (std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x); i < rows;
       i += static_cast<std::int32_t>(gridDim.x * blockDim.x)) {
    to[i] = column[i];
  }
}

// GitHub #621: chain step `step`'s row and mask per lane, and (GitHub #624)
// with `rope` its rotation (mtp_head.h `detail::chain_rows`).
__global__ void chain_rows_kernel(const std::int32_t *base, const std::int32_t *extents, std::int32_t step,
                                  std::int32_t *positions, std::int32_t *valid, std::int32_t count,
                                  const std::int32_t *rope, std::int32_t lane_columns, std::int32_t *rotation) {
  const std::int32_t b = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (b < count) {
    const bool needed = step < extents[b];
    positions[b] = needed ? base[b] - 1 + step : 0;
    valid[b] = needed ? 1 : 0;
    if (rope != nullptr) {
      rotation[b] = needed ? rope[static_cast<std::size_t>(b) * static_cast<std::size_t>(lane_columns)] - 1 + step : 0;
    }
  }
}

// drafts[b * window + step] = tokens[b].
__global__ void write_drafts_kernel(const std::int32_t *tokens, std::int32_t *drafts, std::int32_t window,
                                    std::int32_t step, std::int32_t count) {
  const std::int32_t b = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (b < count) {
    drafts[static_cast<std::size_t>(b) * static_cast<std::size_t>(window) + static_cast<std::size_t>(step)] = tokens[b];
  }
}

// The commit's input hiddens (mtp_head.h `detail::commit_hiddens`): column
// (b, 0) from the continuation table by slot, column (b, j > 0) the round's
// normalized column (b, j - 1).
__global__ void commit_hiddens_kernel(const std::uint16_t *normalized, const std::uint16_t *table,
                                      const std::int32_t *slots, std::uint16_t *out, std::int32_t rows,
                                      std::int32_t lane_columns) {
  const std::int32_t column = static_cast<std::int32_t>(blockIdx.y);
  const std::int32_t lane = column / lane_columns;
  const std::int32_t j = column % lane_columns;
  const std::uint16_t *from =
      j == 0 ? table + static_cast<std::size_t>(slots[lane]) * static_cast<std::size_t>(rows)
             : normalized + static_cast<std::size_t>(column - 1) * static_cast<std::size_t>(rows);
  std::uint16_t *to = out + static_cast<std::size_t>(column) * static_cast<std::size_t>(rows);
  for (std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x); i < rows;
       i += static_cast<std::int32_t>(gridDim.x * blockDim.x)) {
    to[i] = from[i];
  }
}

// The head's stem and attention over `columns` rows: `hidden` the target's
// post-norm hiddens (or, in a chain, the head's own), `embedded` the next
// tokens' embeddings (both BF16 [hidden, columns]), `rotation` the positions
// the rows rotate at (I32 [columns]); `attend(rq, rk, v, gate, out)` appends
// the rows' keys and values and attends, over flat [256, heads, columns]
// views. Returns the stem's residual and the attention's output. Every buffer
// is `call_bytes`'s first two groups, in its order; the attention bumps its
// own workspace after them. A row's key and value are this half's alone, so
// a caller that keeps only the rows (`commit_rows`) stops here.
struct AttendedRows {
  ninfer::Tensor x;
  ninfer::Tensor attention;
};

template <class Attend>
AttendedRows attend_rows(ignis_model &model, const ninfer::Tensor &hidden, const ninfer::Tensor &embedded,
                         const ninfer::Tensor &rotation, std::int32_t columns, ninfer::DeviceArena &arena,
                         Attend &&attend) {
  const MtpHeadWeights &w = model.mtp;
  const auto h = static_cast<std::int32_t>(model.hidden);
  const float eps = model.rms_norm_eps;
  const cudaStream_t stream = model.stream;
  const auto alloc = [&](std::int32_t rows) { return arena.alloc(ninfer::DType::BF16, {rows, columns, 1, 1}); };

  // The stem: [RMSNorm(e); RMSNorm(h)] -> the input projection.
  ninfer::Tensor embedding_normed = alloc(h);
  ninfer::Tensor hidden_normed = alloc(h);
  ninfer::Tensor stem = alloc(2 * h);
  ninfer::Tensor x = alloc(h);
  ninfer::ops::rmsnorm(embedded, gain(w.embedding_norm, h), eps, /*unit_offset=*/true, embedding_normed, stream);
  ninfer::ops::rmsnorm(hidden, gain(w.hidden_norm, h), eps, /*unit_offset=*/true, hidden_normed, stream);
  copy_into_rows(embedding_normed, h, stem, 2 * h, 0, columns, stream);
  copy_into_rows(hidden_normed, h, stem, 2 * h, h, columns, stream);
  ninfer::ops::linear(stem, w.input_projection, x, stream);

  // The attention: the fused projection's rows query | key | gate | value.
  ninfer::Tensor attention_in = alloc(h);
  ninfer::Tensor projected = alloc(kProjectionRows);
  ninfer::Tensor query = alloc(kQueryWidth);
  ninfer::Tensor key = alloc(kKvWidth);
  ninfer::Tensor gate = alloc(kQueryWidth);
  ninfer::Tensor value = alloc(kKvWidth);
  ninfer::Tensor rotated_query = alloc(kQueryWidth);
  ninfer::Tensor rotated_key = alloc(kKvWidth);
  ninfer::Tensor attention = alloc(kQueryWidth);
  ninfer::ops::rmsnorm(x, gain(w.input_norm, h), eps, /*unit_offset=*/true, attention_in, stream);
  ninfer::ops::linear(attention_in, w.query_key_gate_value, projected, stream);
  copy_rows(projected, kProjectionRows, 0, query, kQueryWidth, columns, stream);
  copy_rows(projected, kProjectionRows, kQueryWidth, key, kKvWidth, columns, stream);
  copy_rows(projected, kProjectionRows, kQueryWidth + kKvWidth, gate, kQueryWidth, columns, stream);
  copy_rows(projected, kProjectionRows, 2 * kQueryWidth + kKvWidth, value, kKvWidth, columns, stream);
  ninfer::Tensor rotated_query_heads = rotated_query.view({kHeadDim, kQueryHeads, columns, 1});
  ninfer::Tensor rotated_key_heads = rotated_key.view({kHeadDim, kKvHeads, columns, 1});
  ninfer::ops::qk_norm_rope(query.view({kHeadDim, kQueryHeads, columns, 1}), key.view({kHeadDim, kKvHeads, columns, 1}),
                            gain(w.query_norm, kHeadDim), gain(w.key_norm, kHeadDim), eps, rotation, model.text_rope,
                            rotated_query_heads, rotated_key_heads, stream);
  ninfer::Tensor attention_heads = attention.view({kHeadDim, kQueryHeads, columns, 1});
  attend(rotated_query_heads, rotated_key_heads, value.view({kHeadDim, kKvHeads, columns, 1}),
         gate.view({kHeadDim, kQueryHeads, columns, 1}), attention_heads);
  return AttendedRows{x, attention};
}

// The head's forward over `columns` rows (`attend_rows`, then the output
// projection's residual add, the MLP's and the final norm): leaves the head's
// hidden in `out` (BF16 [hidden, columns]). Every buffer is `call_bytes`'s,
// in its order.
template <class Attend>
void forward(ignis_model &model, const ninfer::Tensor &hidden, const ninfer::Tensor &embedded,
             const ninfer::Tensor &rotation, std::int32_t columns, ninfer::DeviceArena &arena, ninfer::Tensor &out,
             Attend &&attend) {
  const MtpHeadWeights &w = model.mtp;
  const auto h = static_cast<std::int32_t>(model.hidden);
  const float eps = model.rms_norm_eps;
  const cudaStream_t stream = model.stream;
  const auto alloc = [&](std::int32_t rows) { return arena.alloc(ninfer::DType::BF16, {rows, columns, 1, 1}); };
  AttendedRows rows = attend_rows(model, hidden, embedded, rotation, columns, arena, attend);
  ninfer::Tensor &x = rows.x;
  const ninfer::Tensor &attention = rows.attention;

  // The output projection's residual add, then the MLP's.
  ninfer::Tensor projected_out = alloc(h);
  ninfer::Tensor mlp_in = alloc(h);
  ninfer::Tensor gate_up = alloc(2 * kFfn);
  ninfer::Tensor activated = alloc(kFfn);
  ninfer::Tensor down = alloc(h);
  ninfer::ops::linear(attention, w.output, projected_out, stream);
  ninfer::ops::residual_add(projected_out, x, stream);
  ninfer::ops::rmsnorm(x, gain(w.post_attention_norm, h), eps, /*unit_offset=*/true, mlp_in, stream);
  ninfer::ops::linear(mlp_in, w.mlp_gate_up, gate_up, stream);
  ninfer::ops::silu_mul(gate_up.slice(0, 0, kFfn), gate_up.slice(0, kFfn, kFfn), activated, stream);
  ninfer::ops::linear(activated, w.mlp_down, down, stream);
  ninfer::ops::residual_add(down, x, stream);
  ninfer::ops::rmsnorm(x, gain(w.final_norm, h), eps, /*unit_offset=*/true, out, stream);
}

// The head's rows `start .. start + columns - 1` of one sequence, appended
// and attended over its own rows before them, as a target GQA layer's
// prefill chunk is (kernel/src/gqa_layer.cu `run_gqa_layer`): the prompt
// route of ignis's hq attention over hq-e8-2b, the vendored A1 otherwise.
// `axes` (host I32 axis-major, `axes_pitch` entries an axis, or null) are the
// rows' three MRoPE axes on a multimodal chunk (GitHub #624).
void sequence_rows(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, const ninfer::Tensor &hidden,
                   const ninfer::Tensor &embedded, std::int32_t start, std::int32_t columns, ninfer::DeviceArena &arena,
                   ninfer::Tensor &out, const std::int32_t *axes = nullptr, std::int32_t axes_pitch = 0) {
  const cudaStream_t stream = model.stream;
  ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {columns, 1, 1, 1});
  ninfer::ops::fill_i32_positions(positions, start, stream);
  // The rows rotate where the target's hidden of the same position did: a
  // multimodal chunk's axes, else the position plus the sequence's delta.
  ninfer::Tensor rotation;
  if (axes != nullptr) {
    rotation = arena.alloc(ninfer::DType::I32, {columns, 3, 1, 1});
    check(cudaMemcpy2DAsync(rotation.data, static_cast<std::size_t>(columns) * sizeof(std::int32_t), axes,
                            static_cast<std::size_t>(axes_pitch) * sizeof(std::int32_t),
                            static_cast<std::size_t>(columns) * sizeof(std::int32_t), 3, cudaMemcpyHostToDevice,
                            stream),
          "cudaMemcpy2DAsync(the head's rope axes)");
  } else {
    rotation = arena.alloc(ninfer::DType::I32, {columns, 1, 1, 1});
    ninfer::ops::fill_i32_positions(rotation, start + seq.rope_delta, stream);
  }
  const ninfer::Tensor kv_table_rows = arena.alloc(ninfer::DType::I32, {1, 1, 1, 1});
  check(cudaMemsetAsync(kv_table_rows.data, 0, sizeof(std::int32_t), stream), "cudaMemsetAsync(kv table rows)");
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const auto attend = [&](const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
                          const ninfer::Tensor &g, ninfer::Tensor &o) {
    ninfer::PagedKVBatchLayerView cache = ignis_kv_batch_layer_view(&pool, pool.mtp_head_layer());
    const ninfer::Tensor block_table = seq.kv.block_table();
    cache.block_tables = block_table.view({block_table.ne[0], 1});
    ignis_kv_fill_residual(cache, &pool, pool.mtp_head_layer(), seq.slot, 1);
    ignis_mtp_head_ring(cache, &pool, seq.slot, 1);
    const ninfer::ops::GqaExecutionEnvelope envelope{
        .min_visible_keys = 1,
        .max_visible_keys = static_cast<std::uint32_t>(start + columns),
    };
    const std::size_t workspace_bytes = ignis_gqa_attention_workspace_bytes(cache.dtype, envelope, columns);
    const ninfer::DeviceSpan storage = arena.alloc_bytes(std::max<std::size_t>(workspace_bytes, 1));
    if (ignis_gqa_workspace_needs_zeroing(cache.dtype)) {
      check(cudaMemsetAsync(storage.data, 0, storage.bytes, stream), "cudaMemsetAsync(attention workspace)");
    }
    ninfer::DeviceArena workspace(storage);
    const bool hq_prompt =
        cache.dtype == ninfer::DType::U8 &&
        ninfer::ops::detail::gqa_attention_resolve_route(kQueryHeads, columns, 1, cache.dtype, envelope) ==
            ninfer::ops::detail::GqaAttentionRoute::Prompt;
    if (hq_prompt) {
      IgnisHqPromptOptions options;
      options.precision = model.prefill_attention == IGNIS_PREFILL_ATTENTION_FP16   ? IgnisHqPromptPrecision::kFp16
                          : model.prefill_attention == IGNIS_PREFILL_ATTENTION_INT8 ? IgnisHqPromptPrecision::kInt8
                                                                                    : IgnisHqPromptPrecision::kFp8;
      ignis_hq_prompt_attention(q, k, v, positions, start, g, scale, ignis_mtp_head_layer_view(&pool, &seq), workspace,
                                o, stream, options);
    } else {
      ninfer::ops::gqa_attention(q, k, v, positions, /*valid_columns=*/ninfer::Tensor{}, kv_table_rows, g, scale, cache,
                                 envelope, workspace, o, stream);
    }
  };
  forward(model, hidden, embedded, rotation, columns, arena, out, attend);
}

// The append-and-attend of a round's rows, `lane_columns` a lane over the
// round's `batch` lanes, as a target GQA layer's batch body is
// (kernel/src/gqa_layer.cu `run_gqa_layer_batch`): the pool-wide tables, each
// lane's slot from `model.decode_graph_slots`, positions I32 [lane_columns,
// batch]; `valid_columns` I32 [batch] the masked form's each lane's valid
// prefix, or an empty tensor (every column). Graph-safe: every choice is the
// load's and the call's shape.
struct BatchAttend {
  ignis_model &model;
  ignis_seq_pool &pool;
  ninfer::DeviceArena &arena;
  std::int32_t lane_columns;
  std::int32_t batch;
  const void *positions;
  ninfer::Tensor valid_columns;

  void operator()(const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v, const ninfer::Tensor &g,
                  ninfer::Tensor &o) const {
    const cudaStream_t stream = model.stream;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const ninfer::Tensor slots(model.decode_graph_slots->p, ninfer::DType::I32, {batch, 1, 1, 1});
    const ninfer::Tensor positions_rows(const_cast<void *>(positions), ninfer::DType::I32,
                                        {lane_columns, batch, 1, 1});
    const ninfer::PagedKVBatchLayerView cache = ignis_mtp_head_batch_view(&pool);
    const ninfer::ops::GqaExecutionEnvelope envelope{
        .min_visible_keys = 1,
        .max_visible_keys = model.max_context_tokens,
    };
    const std::size_t workspace_bytes = ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kQueryHeads, cache.dtype, envelope, batch, lane_columns, lane_columns);
    const ninfer::DeviceSpan storage = arena.alloc_bytes(std::max<std::size_t>(workspace_bytes, 1));
    if (ignis_gqa_workspace_needs_zeroing(cache.dtype)) {
      check(cudaMemsetAsync(storage.data, 0, storage.bytes, stream), "cudaMemsetAsync(attention workspace)");
    }
    ninfer::DeviceArena workspace(storage);
    const ninfer::Tensor q_rows = q.view({kHeadDim, kQueryHeads, lane_columns, batch});
    const ninfer::Tensor k_rows = k.view({kHeadDim, kKvHeads, lane_columns, batch});
    const ninfer::Tensor v_rows = v.view({kHeadDim, kKvHeads, lane_columns, batch});
    const ninfer::Tensor g_rows = g.view({kHeadDim, kQueryHeads, lane_columns, batch});
    ninfer::Tensor o_rows = o.view({kHeadDim, kQueryHeads, lane_columns, batch});
    if (model.hq_decode_attention_ours &&
        ignis::hq_decode_attention::serves(kQueryHeads, lane_columns, batch, cache.dtype, envelope)) {
      ignis::hq_decode_attention::attend(q_rows, k_rows, v_rows, positions_rows, valid_columns, slots, g_rows, scale,
                                         cache, envelope, workspace, o_rows, stream);
    } else {
      ninfer::ops::gqa_attention(q_rows, k_rows, v_rows, positions_rows, valid_columns, slots, g_rows, scale, cache,
                                 envelope, workspace, o_rows, stream);
    }
  }
};

std::int32_t continuation_slots(const ignis_seq_pool &pool) {
  return static_cast<std::int32_t>(pool.mtp_continuation->bytes / pool.mtp_continuation_slot_bytes());
}

// GitHub #624: the media columns a call's next tokens took in the target's
// embedding, into `embedded` (BF16 [hidden, count], column i the chunk's
// column `first + i`): each placeholder's image column in place of its id's
// row, copied in runs of consecutive columns within one pool page, from
// wherever the load's vision residency put the pool.
void embed_media_columns(const ignis_model &model, const ChunkMultimodal &multimodal, std::int32_t first,
                         std::int32_t count, ninfer::Tensor &embedded, cudaStream_t stream) {
  if (multimodal.count == 0) {
    return;
  }
  const VisionEmbeddingPool &vision = model.vision_pool;
  const std::size_t column_bytes = static_cast<std::size_t>(model.hidden) * sizeof(std::uint16_t);
  const std::int32_t *const end_of_media = multimodal.scatter + multimodal.count;
  const std::int32_t *begin = std::lower_bound(multimodal.scatter, end_of_media, first);
  const std::int32_t *const end = std::lower_bound(begin, end_of_media, first + count);
  while (begin < end) {
    const auto at = multimodal.first_column + static_cast<std::int32_t>(begin - multimodal.scatter);
    const std::int32_t within = at % vision.page_columns;
    std::int32_t take = 1;
    while (begin + take < end && begin[take] == begin[0] + take && within + take < vision.page_columns) {
      ++take;
    }
    check(cudaMemcpyAsync(static_cast<unsigned char *>(embedded.data) +
                              static_cast<std::size_t>(begin[0] - first) * column_bytes,
                          vision.page_ptr(multimodal.media->pages[static_cast<std::size_t>(at / vision.page_columns)]) +
                              static_cast<std::size_t>(within) * column_bytes,
                          static_cast<std::size_t>(take) * column_bytes, cudaMemcpyDefault, stream),
          "cudaMemcpyAsync(the head's media columns)");
    begin += take;
  }
}

void launch_gather_columns(const ignis_seq_pool &pool, const ninfer::Tensor &slots, ninfer::Tensor &out,
                           std::int32_t hidden, std::int32_t batch, cudaStream_t stream) {
  gather_columns_kernel<<<dim3(static_cast<unsigned>((hidden + 255) / 256), static_cast<unsigned>(batch)), 256, 0,
                          stream>>>(static_cast<const std::uint16_t *>(pool.mtp_continuation->p), hidden,
                                    static_cast<const std::int32_t *>(slots.data),
                                    static_cast<std::uint16_t *>(out.data), batch);
  check(cudaGetLastError(), "the continuation hidden's gather");
}

}  // namespace

namespace detail {

void chain_rows(const std::int32_t *base_positions, const std::int32_t *extents, std::int32_t step,
                std::int32_t *positions, std::int32_t *valid, std::int32_t batch, cudaStream_t stream,
                const std::int32_t *rope_positions, std::int32_t lane_columns, std::int32_t *rotation) {
  chain_rows_kernel<<<1, 32, 0, stream>>>(base_positions, extents, step, positions, valid, batch, rope_positions,
                                          lane_columns, rotation);
  check(cudaGetLastError(), "the chain's rows");
}

void write_drafts(const std::int32_t *tokens, std::int32_t *drafts, std::int32_t window, std::int32_t step,
                  std::int32_t batch, cudaStream_t stream) {
  write_drafts_kernel<<<1, 32, 0, stream>>>(tokens, drafts, window, step, batch);
  check(cudaGetLastError(), "the chain's drafts");
}

void commit_hiddens(const std::uint16_t *normalized, const std::uint16_t *table, const std::int32_t *slots,
                    std::uint16_t *out, std::int32_t hidden, std::int32_t lane_columns, std::int32_t batch,
                    cudaStream_t stream) {
  commit_hiddens_kernel<<<dim3(static_cast<unsigned>((hidden + 255) / 256),
                               static_cast<unsigned>(lane_columns * batch)),
                          256, 0, stream>>>(normalized, table, slots, out, hidden, lane_columns);
  check(cudaGetLastError(), "the commit's hiddens");
}

}  // namespace detail

std::size_t call_bytes(std::int64_t hidden, std::int64_t rows) {
  std::size_t bytes = 0;
  // The stem: both normed inputs, their concatenation and the residual.
  bytes += 2 * bf16_bytes(hidden * rows) + bf16_bytes(2 * hidden * rows) + bf16_bytes(hidden * rows);
  // The attention's input, the fused projection, its four parts, the two
  // rotated ones and the attention's output.
  bytes += bf16_bytes(hidden * rows) + bf16_bytes(static_cast<std::int64_t>(kProjectionRows) * rows);
  bytes += 3 * bf16_bytes(static_cast<std::int64_t>(kQueryWidth) * rows) +
           3 * bf16_bytes(static_cast<std::int64_t>(kKvWidth) * rows) +
           bf16_bytes(static_cast<std::int64_t>(kQueryWidth) * rows);
  // The output projection, the MLP's input, gate and up, the SwiGLU and down.
  bytes += 2 * bf16_bytes(hidden * rows) + bf16_bytes(2 * static_cast<std::int64_t>(kFfn) * rows) +
           bf16_bytes(static_cast<std::int64_t>(kFfn) * rows) + bf16_bytes(hidden * rows);
  return bytes;
}

std::size_t prefill_scratch_bytes(std::int64_t hidden, std::int64_t chunk, bool vision) {
  const std::int64_t rows = std::min<std::int64_t>(chunk, kPrefillRows);
  // The call's input hiddens, embeddings and head hiddens; its positions,
  // rotation positions (three axes a row on a vision load's multimodal
  // chunk) and table row; the call itself.
  return 3 * bf16_bytes(hidden * rows) + i32_bytes(rows) + i32_bytes((vision ? 3 : 1) * rows) + i32_bytes(1) +
         call_bytes(hidden, rows);
}

std::size_t round_scratch_bytes(std::int64_t hidden, std::int64_t columns, bool vision) {
  // The round's continuation hiddens, embeddings and head hiddens, its
  // positions (and on a vision load their rotation), and the call.
  return 3 * bf16_bytes(hidden * columns) + (vision ? 2 : 1) * i32_bytes(columns) + call_bytes(hidden, columns);
}

int32_t prefill_chunk(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, const ninfer::Tensor &ids,
                      const ninfer::Tensor &final_residual, std::int32_t tokens, const ChunkMultimodal &multimodal,
                      std::uint16_t *span_hidden, std::string *error) {
  try {
    const auto hidden = static_cast<std::int32_t>(model.hidden);
    const cudaStream_t stream = model.stream;
    const ninfer::Tensor final_norm = gain(model.final_norm, hidden);
    const std::size_t column_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    // The chunk's rows are the head's at the positions before each of its
    // tokens: the first, at the position before the chunk, built from the
    // continuation hidden -- a prompt's first chunk has none, so one row
    // fewer -- and the last at the chunk's last position but one.
    const std::int64_t chunk_start = static_cast<std::int64_t>(seq.position);
    const std::int32_t first_column = chunk_start > 0 ? -1 : 0;
    const std::int32_t rows = tokens - 1 - first_column;
    for (std::int32_t done = 0; done < rows; done += kPrefillRows) {
      const std::int32_t count = std::min(kPrefillRows, rows - done);
      ninfer::DeviceArena::Scope scope = model.scratch->scope();
      // The target's hidden of each row's position: column `j0 + i` of the
      // chunk, column -1 being the continuation hidden.
      const std::int32_t j0 = first_column + done;
      ninfer::Tensor hidden_in = model.scratch->alloc(ninfer::DType::BF16, {hidden, count, 1, 1});
      if (j0 < 0) {
        check(cudaMemcpyAsync(hidden_in.data, pool.mtp_continuation_of(seq.slot), column_bytes,
                              cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync(continuation hidden)");
        if (count > 1) {
          ninfer::Tensor rest = columns_of(hidden_in, hidden, 1, count - 1);
          ninfer::ops::rmsnorm(columns_of(final_residual, hidden, 0, count - 1), final_norm, model.rms_norm_eps,
                               /*unit_offset=*/true, rest, stream);
        }
      } else {
        ninfer::ops::rmsnorm(columns_of(final_residual, hidden, j0, count), final_norm, model.rms_norm_eps,
                             /*unit_offset=*/true, hidden_in, stream);
      }
      if (span_hidden != nullptr) {
        // The chunk's own columns among them, for the readout.
        const std::int32_t skip = j0 < 0 ? 1 : 0;
        if (count > skip) {
          check(cudaMemcpyAsync(span_hidden + static_cast<std::size_t>(j0 + skip) * static_cast<std::size_t>(hidden),
                                static_cast<const unsigned char *>(hidden_in.data) + skip * column_bytes,
                                static_cast<std::size_t>(count - skip) * column_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync(span hidden)");
        }
      }
      // The token after each row's position: chunk column `j0 + 1 + i`,
      // embedded as the target's embedding took it -- an image placeholder
      // as its image's column (GitHub #624).
      const ninfer::Tensor next_ids(static_cast<std::int32_t *>(ids.data) + (j0 + 1), ninfer::DType::I32,
                                    {count, 1, 1, 1});
      ninfer::Tensor embedded = model.scratch->alloc(ninfer::DType::BF16, {hidden, count, 1, 1});
      ninfer::ops::embedding(next_ids, model.token_embedding, embedded, stream);
      embed_media_columns(model, multimodal, j0 + 1, count, embedded, stream);
      ninfer::Tensor out = model.scratch->alloc(ninfer::DType::BF16, {hidden, count, 1, 1});
      // A multimodal chunk's rows rotate at the axes of chunk columns j0 ..
      // j0 + count - 1, entries j0 + 1 .. of the chunk's axes.
      sequence_rows(model, pool, seq, hidden_in, embedded, static_cast<std::int32_t>(chunk_start + j0), count,
                    *model.scratch, out, multimodal.axes == nullptr ? nullptr : multimodal.axes + (j0 + 1),
                    tokens + 1);
    }
    // The chunk's last column is the sequence's continuation hidden, read by
    // the head's next row (the next chunk's first, or the next round's).
    ninfer::Tensor continuation(pool.mtp_continuation_of(seq.slot), ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(columns_of(final_residual, hidden, tokens - 1, 1), final_norm, model.rms_norm_eps,
                         /*unit_offset=*/true, continuation, stream);
    if (span_hidden != nullptr) {
      check(cudaMemcpyAsync(span_hidden + static_cast<std::size_t>(tokens - 1) * static_cast<std::size_t>(hidden),
                            continuation.data, column_bytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(span hidden)");
    }
    return 0;
  } catch (const std::exception &e) {
    *error = std::string("the MTP head: ") + e.what();
    return -1;
  }
}

int32_t decode_round(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, const ninfer::Tensor &normalized,
                     std::string *error) {
  try {
    const auto hidden = static_cast<std::int32_t>(model.hidden);
    const auto batch = static_cast<std::int32_t>(width);
    const cudaStream_t stream = model.stream;
    ninfer::DeviceArena &arena = *model.decode_graph_scratch;
    const ninfer::Tensor slots(model.decode_graph_slots->p, ninfer::DType::I32, {batch, 1, 1, 1});
    const ninfer::Tensor ids(model.decode_graph_token_ids->p, ninfer::DType::I32, {batch, 1, 1, 1});

    // Each lane's row is the head's at its position less one: from its
    // continuation hidden and the token this round consumes.
    ninfer::Tensor hidden_in = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    launch_gather_columns(pool, slots, hidden_in, hidden, batch, stream);
    ninfer::Tensor embedded = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::ops::embedding(ids, model.token_embedding, embedded, stream);
    ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
    previous_positions_kernel<<<1, 32, 0, stream>>>(static_cast<const std::int32_t *>(model.sampling_decode_positions->p),
                                                    static_cast<std::int32_t *>(positions.data), batch);
    check(cudaGetLastError(), "the head's positions");
    // GitHub #624: a vision load's rows rotate at the round's own rotation
    // positions less one (position + rope delta); a text load's at their
    // positions.
    ninfer::Tensor rotation = positions;
    if (model.decode_rope_positions != nullptr) {
      rotation = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
      previous_positions_kernel<<<1, 32, 0, stream>>>(static_cast<const std::int32_t *>(model.decode_rope_positions->p),
                                                      static_cast<std::int32_t *>(rotation.data), batch);
      check(cudaGetLastError(), "the head's rotation");
    }
    ninfer::Tensor out = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});

    // A1's batched shape, one row per lane, every lane's row valid.
    forward(model, hidden_in, embedded, rotation, batch, arena, out,
            BatchAttend{model, pool, arena, 1, batch, positions.data, ninfer::Tensor{}});

    // Each lane's continuation hidden is now its hidden at the position it
    // consumed.
    ninfer::Tensor table(pool.mtp_continuation->p, ninfer::DType::BF16, {hidden, continuation_slots(pool), 1, 1});
    ninfer::ops::scatter(normalized, slots, table, stream);
    return 0;
  } catch (const std::exception &e) {
    *error = std::string("the MTP head: ") + e.what();
    return -1;
  }
}

int32_t frontier_row(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, std::int32_t next_token,
                     ninfer::DeviceArena &arena, ninfer::Tensor &out_hidden, std::string *error) {
  try {
    if (seq.position == 0) {
      throw std::invalid_argument("the sequence holds no position yet");
    }
    const auto hidden = static_cast<std::int32_t>(model.hidden);
    const cudaStream_t stream = model.stream;
    ninfer::Tensor hidden_in(pool.mtp_continuation_of(seq.slot), ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::Tensor id = arena.alloc(ninfer::DType::I32, {1, 1, 1, 1});
    check(cudaMemcpyAsync(id.data, &next_token, sizeof(next_token), cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync(next token)");
    ninfer::Tensor embedded = arena.alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::embedding(id, model.token_embedding, embedded, stream);
    sequence_rows(model, pool, seq, hidden_in, embedded, static_cast<std::int32_t>(seq.position - 1), 1, arena,
                  out_hidden);
    return 0;
  } catch (const std::exception &e) {
    *error = std::string("the MTP head: ") + e.what();
    return -1;
  }
}

RoundRows round_rows(std::uint64_t base, std::uint32_t extent, std::uint32_t committed) {
  // The chain's steps 0 .. e - 1 write rows P - 1 .. P + e - 2; the
  // commit's c rows start at P - 1 too.
  RoundRows rows;
  rows.first = base - 1;
  rows.chain_end = base - 1 + extent;
  rows.committed_end = base - 1 + committed;
  return rows;
}

int32_t propose(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, std::string *error,
                std::uint16_t *step_scores) {
  try {
    IgnisVerifyRound &verify = *model.verify;
    const auto window = static_cast<std::int32_t>(verify.window);
    const auto hidden = static_cast<std::int32_t>(model.hidden);
    const auto vocab = static_cast<std::int32_t>(model.vocab);
    const auto batch = static_cast<std::int32_t>(width);
    const cudaStream_t stream = model.stream;
    ninfer::DeviceArena &arena = *model.decode_graph_scratch;
    // Its own scope: released before the pass's buffers are bumped.
    ninfer::DeviceArena::Scope scope = arena.scope();
    const ninfer::Tensor slots(model.decode_graph_slots->p, ninfer::DType::I32, {batch, 1, 1, 1});
    const ninfer::Tensor anchors(verify.anchors->p, ninfer::DType::I32, {batch, 1, 1, 1});
    const auto *base_positions = static_cast<const std::int32_t *>(verify.base_positions->p);
    const auto *extents = static_cast<const std::int32_t *>(verify.extents->p);
    auto *drafts = static_cast<std::int32_t *>(verify.drafts->p);

    // A step's input hidden and its output, swapped after each step: the
    // first step's is the continuation hidden.
    ninfer::Tensor step_in = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::Tensor step_out = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    launch_gather_columns(pool, slots, step_in, hidden, batch, stream);
    ninfer::Tensor chosen = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor valid = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
    // GitHub #624: a vision load's rows rotate at each lane's rotation
    // positions (base + rope delta, the round's own) less one, plus the step;
    // a text load's at their positions.
    const auto *rope_positions =
        verify.rope_positions != nullptr ? static_cast<const std::int32_t *>(verify.rope_positions->p) : nullptr;
    ninfer::Tensor rotation =
        rope_positions != nullptr ? arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1}) : positions;
    // The step's scores: the round's own logits, which the pass after the
    // chain writes whole -- or (GitHub #626) the readout's block of the step.
    ninfer::Tensor logits(verify.logits->p, ninfer::DType::BF16, {vocab, batch, 1, 1});
    for (std::int32_t step = 0; step < window; ++step) {
      ninfer::DeviceArena::Scope step_scope = arena.scope();
      if (step_scores != nullptr) {
        logits = ninfer::Tensor(step_scores + static_cast<std::size_t>(step) * static_cast<std::size_t>(vocab) *
                                                  static_cast<std::size_t>(batch),
                                ninfer::DType::BF16, {vocab, batch, 1, 1});
      }
      detail::chain_rows(base_positions, extents, step, static_cast<std::int32_t *>(positions.data),
                         static_cast<std::int32_t *>(valid.data), batch, stream, rope_positions, window + 1,
                         static_cast<std::int32_t *>(rotation.data));
      ninfer::Tensor embedded = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
      ninfer::ops::embedding(step == 0 ? anchors : chosen, model.token_embedding, embedded, stream);
      forward(model, step_in, embedded, rotation, batch, arena, step_out,
              BatchAttend{model, pool, arena, 1, batch, positions.data, valid});
      ignis::trellis_body::output_head(model, step_out, logits, stream);
      ninfer::ops::argmax(logits, chosen, vocab, stream);
      detail::write_drafts(static_cast<const std::int32_t *>(chosen.data), drafts, window, step, batch, stream);
      std::swap(step_in, step_out);
    }
    return 0;
  } catch (const std::exception &e) {
    *error = std::string("the MTP head's chain: ") + e.what();
    return -1;
  }
}

int32_t commit_rows(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, const std::int32_t *positions,
                    const std::int32_t *rotation, const std::int32_t *committed, std::string *error) {
  try {
    IgnisVerifyRound &verify = *model.verify;
    const auto lane_columns = static_cast<std::int32_t>(verify.window + 1);
    const auto hidden = static_cast<std::int32_t>(model.hidden);
    const auto batch = static_cast<std::int32_t>(width);
    const std::int32_t columns = lane_columns * batch;
    const cudaStream_t stream = model.stream;
    ninfer::DeviceArena &arena = *model.decode_graph_scratch;
    ninfer::DeviceArena::Scope scope = arena.scope();

    // The target's post-final-norm hidden of every column of the round: the
    // pass's own normalization, again, of the residual it kept.
    const ninfer::Tensor residual(verify.hidden->p, ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::Tensor normalized = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::ops::rmsnorm(residual, gain(model.final_norm, hidden), model.rms_norm_eps, /*unit_offset=*/true,
                         normalized, stream);
    // Row P - 1 + j is built from the hidden at P - 1 + j -- the continuation
    // hidden at j = 0, the round's column j - 1 after it -- and the token at
    // P + j, the round's own column j: the run's tokens.
    ninfer::Tensor hidden_in = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    detail::commit_hiddens(static_cast<const std::uint16_t *>(normalized.data),
                           static_cast<const std::uint16_t *>(pool.mtp_continuation->p),
                           static_cast<const std::int32_t *>(model.decode_graph_slots->p),
                           static_cast<std::uint16_t *>(hidden_in.data), hidden, lane_columns, batch, stream);
    const ninfer::Tensor ids(verify.verify_ids->p, ninfer::DType::I32, {columns, 1, 1, 1});
    ninfer::Tensor embedded = arena.alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::ops::embedding(ids, model.token_embedding, embedded, stream);
    ninfer::Tensor rows = arena.alloc(ninfer::DType::I32, {columns, 1, 1, 1});
    ninfer::Tensor valid = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
    check(cudaMemcpyAsync(rows.data, positions, static_cast<std::size_t>(columns) * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync(the commit's rows)");
    check(cudaMemcpyAsync(valid.data, committed, static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync(the commit's counts)");
    // GitHub #624: a vision load's rows rotate at their positions plus the
    // lane's rope delta, as the chain's did.
    ninfer::Tensor turn = rows;
    if (rotation != nullptr) {
      turn = arena.alloc(ninfer::DType::I32, {columns, 1, 1, 1});
      check(cudaMemcpyAsync(turn.data, rotation, static_cast<std::size_t>(columns) * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(the commit's rotation)");
    }
    // Each lane's committed prefix of its columns, appended and attended in
    // the masked form; the rows are what stays, the attention is no draft's.
    (void)attend_rows(model, hidden_in, embedded, turn, columns, arena,
                      BatchAttend{model, pool, arena, lane_columns, batch, rows.data, valid});
    return 0;
  } catch (const std::exception &e) {
    *error = std::string("the MTP head's commit: ") + e.what();
    return -1;
  }
}

int32_t commit_continuation(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, std::string *error) {
  try {
    const auto hidden = static_cast<std::int32_t>(model.hidden);
    const auto batch = static_cast<std::int32_t>(width);
    ninfer::DeviceArena &arena = *model.decode_graph_scratch;
    ninfer::DeviceArena::Scope scope = arena.scope();
    const ninfer::Tensor selected(model.verify->selected_hidden->p, ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::Tensor normalized = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    // One CTA a column at this width, the pass's own normalization: the bits
    // of the column whose scores drew the lane's pending token.
    ninfer::ops::rmsnorm(selected, gain(model.final_norm, hidden), model.rms_norm_eps, /*unit_offset=*/true,
                         normalized, model.stream);
    const ninfer::Tensor slots(model.decode_graph_slots->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor table(pool.mtp_continuation->p, ninfer::DType::BF16, {hidden, continuation_slots(pool), 1, 1});
    ninfer::ops::scatter(normalized, slots, table, model.stream);
    return 0;
  } catch (const std::exception &e) {
    *error = std::string("the MTP head's continuation: ") + e.what();
    return -1;
  }
}

}  // namespace ignis::mtp
