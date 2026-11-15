// ignis kernel leaf - GitHub #620 (spec #618): the Qwen 3.8-27B's MTP head --
// OURS. What it computes and which rows each step writes: mtp_head.h.

#include "mtp_head.h"

#include "hq_decode_attention.h"
#include "ignis_gqa_workspace.h"
#include "ignis_hq_prompt_attention.h"
#include "ignis_seq_internal.h"
#include "model_internal.h"

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

// The head's forward over `columns` rows: `hidden` the target's post-norm
// hiddens, `embedded` the next tokens' embeddings (both BF16 [hidden,
// columns]), `rotation` the positions the rows rotate at (I32 [columns]);
// `attend(rq, rk, v, gate, out)` appends the rows' keys and values and
// attends, over flat [256, heads, columns] views. Leaves the head's hidden in
// `out` (BF16 [hidden, columns]). Every buffer is `call_bytes`'s, in its
// order; the attention bumps its own workspace after them.
template <class Attend>
void forward(ignis_model &model, const ninfer::Tensor &hidden, const ninfer::Tensor &embedded,
             const ninfer::Tensor &rotation, std::int32_t columns, ninfer::DeviceArena &arena, ninfer::Tensor &out,
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
void sequence_rows(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, const ninfer::Tensor &hidden,
                   const ninfer::Tensor &embedded, std::int32_t start, std::int32_t columns, ninfer::DeviceArena &arena,
                   ninfer::Tensor &out) {
  const cudaStream_t stream = model.stream;
  ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {columns, 1, 1, 1});
  ninfer::ops::fill_i32_positions(positions, start, stream);
  // The rows rotate where the target's hidden of the same position did.
  ninfer::Tensor rotation = arena.alloc(ninfer::DType::I32, {columns, 1, 1, 1});
  ninfer::ops::fill_i32_positions(rotation, start + seq.rope_delta, stream);
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

}  // namespace

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

std::size_t prefill_scratch_bytes(std::int64_t hidden, std::int64_t chunk) {
  const std::int64_t rows = std::min<std::int64_t>(chunk, kPrefillRows);
  // The call's input hiddens, embeddings and head hiddens; its positions,
  // rotation positions and table row; the call itself.
  return 3 * bf16_bytes(hidden * rows) + 2 * i32_bytes(rows) + i32_bytes(1) + call_bytes(hidden, rows);
}

std::size_t round_scratch_bytes(std::int64_t hidden, std::int64_t columns) {
  // The round's continuation hiddens, embeddings and head hiddens, its
  // positions, and the call.
  return 3 * bf16_bytes(hidden * columns) + i32_bytes(columns) + call_bytes(hidden, columns);
}

int32_t prefill_chunk(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, const ninfer::Tensor &ids,
                      const ninfer::Tensor &final_residual, std::int32_t tokens, std::uint16_t *span_hidden,
                      std::string *error) {
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
      // The token after each row's position: chunk column `j0 + 1 + i`.
      const ninfer::Tensor next_ids(static_cast<std::int32_t *>(ids.data) + (j0 + 1), ninfer::DType::I32,
                                    {count, 1, 1, 1});
      ninfer::Tensor embedded = model.scratch->alloc(ninfer::DType::BF16, {hidden, count, 1, 1});
      ninfer::ops::embedding(next_ids, model.token_embedding, embedded, stream);
      ninfer::Tensor out = model.scratch->alloc(ninfer::DType::BF16, {hidden, count, 1, 1});
      sequence_rows(model, pool, seq, hidden_in, embedded, static_cast<std::int32_t>(chunk_start + j0), count,
                    *model.scratch, out);
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
    const auto state_slots =
        static_cast<std::int32_t>(pool.mtp_continuation->bytes / pool.mtp_continuation_slot_bytes());

    // Each lane's row is the head's at its position less one: from its
    // continuation hidden and the token this round consumes.
    ninfer::Tensor hidden_in = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    gather_columns_kernel<<<dim3(static_cast<unsigned>((hidden + 255) / 256), static_cast<unsigned>(batch)), 256, 0,
                            stream>>>(static_cast<const std::uint16_t *>(pool.mtp_continuation->p), hidden,
                                      static_cast<const std::int32_t *>(slots.data),
                                      static_cast<std::uint16_t *>(hidden_in.data), batch);
    check(cudaGetLastError(), "the continuation hidden's gather");
    ninfer::Tensor embedded = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::ops::embedding(ids, model.token_embedding, embedded, stream);
    ninfer::Tensor positions = arena.alloc(ninfer::DType::I32, {batch, 1, 1, 1});
    previous_positions_kernel<<<1, 32, 0, stream>>>(static_cast<const std::int32_t *>(model.sampling_decode_positions->p),
                                                    static_cast<std::int32_t *>(positions.data), batch);
    check(cudaGetLastError(), "the head's positions");
    ninfer::Tensor out = arena.alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});

    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const ninfer::Tensor positions_rows(positions.data, ninfer::DType::I32, {1, batch, 1, 1});
    const auto attend = [&](const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
                            const ninfer::Tensor &g, ninfer::Tensor &o) {
      // A1's batched shape, one row per lane (kernel/src/gqa_layer.cu
      // `run_gqa_layer_batch`): the pool-wide tables, each lane's slot.
      const ninfer::PagedKVBatchLayerView cache = ignis_mtp_head_batch_view(&pool);
      const ninfer::ops::GqaExecutionEnvelope envelope{
          .min_visible_keys = 1,
          .max_visible_keys = model.max_context_tokens,
      };
      const std::size_t workspace_bytes =
          ninfer::ops::gqa_attention_workspace_capacity_bytes(kQueryHeads, cache.dtype, envelope, batch, 1, 1);
      const ninfer::DeviceSpan storage = arena.alloc_bytes(std::max<std::size_t>(workspace_bytes, 1));
      if (ignis_gqa_workspace_needs_zeroing(cache.dtype)) {
        check(cudaMemsetAsync(storage.data, 0, storage.bytes, stream), "cudaMemsetAsync(attention workspace)");
      }
      ninfer::DeviceArena workspace(storage);
      const ninfer::Tensor q_rows = q.view({kHeadDim, kQueryHeads, 1, batch});
      const ninfer::Tensor k_rows = k.view({kHeadDim, kKvHeads, 1, batch});
      const ninfer::Tensor v_rows = v.view({kHeadDim, kKvHeads, 1, batch});
      const ninfer::Tensor g_rows = g.view({kHeadDim, kQueryHeads, 1, batch});
      ninfer::Tensor o_rows = o.view({kHeadDim, kQueryHeads, 1, batch});
      if (model.hq_decode_attention_ours &&
          ignis::hq_decode_attention::serves(kQueryHeads, 1, batch, cache.dtype, envelope)) {
        ignis::hq_decode_attention::attend(q_rows, k_rows, v_rows, positions_rows, ninfer::Tensor{}, slots, g_rows,
                                           scale, cache, envelope, workspace, o_rows, stream);
      } else {
        ninfer::ops::gqa_attention(q_rows, k_rows, v_rows, positions_rows, ninfer::Tensor{}, slots, g_rows, scale,
                                   cache, envelope, workspace, o_rows, stream);
      }
    };
    // A text load's rows rotate at their positions (an MTP load has no vision).
    forward(model, hidden_in, embedded, positions, batch, arena, out, attend);

    // Each lane's continuation hidden is now its hidden at the position it
    // consumed.
    ninfer::Tensor table(pool.mtp_continuation->p, ninfer::DType::BF16, {hidden, state_slots, 1, 1});
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

}  // namespace ignis::mtp
