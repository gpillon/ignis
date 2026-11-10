// ignis kernel leaf - P1-21 (GitHub #57): one GQA layer in the device-resident
// program. The program layer is ours (ADR 0009); every dispatched op is the
// ADR 0010 vendored reference implementation. The order is the Qwen text
// context: input norm -> fused Q/K/gate/V projection -> q/k norm + RoPE ->
// KV append -> attention + sigmoid output gate -> output residual -> MLP tail.

#include "attention_readout.h"
#include "ignis_attn_tap.h"
#include "ignis_gqa_layer.h"

#include "ignis_gqa_workspace.h"
#include "ignis_hq_prompt_attention.h"
#include "ignis_seq_internal.h"
#include "layer_internal.h"
#include "model_internal.h"
#include "trellis_body.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/qk_norm_rope.h"
#include "ninfer/ops/rmsnorm.h"

#include "ignis_step.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

ninfer::Tensor weight_tensor(const ninfer::Weight &weight, ninfer::DType dtype,
                             std::initializer_list<std::int32_t> shape) {
  return ninfer::Tensor(const_cast<void *>(weight.qdata), dtype, shape);
}

// GitHub #339: the layer's query, output gate, key and value on the body the
// load bound -- the vendored fused projection under the call's mode policy,
// or the trellis body's three matrices: `q_proj`'s per-head [q | gate] rows
// split into the query and the gate (kernel/src/trellis_body.h), `k_proj`
// and `v_proj` straight into theirs (GitHub #509: on the dense prefill, one
// quantized input for the three and the split in the epilogue). The trellis
// op has no policy to resolve; `mode` is the NVFP4 arm's alone.
void input_projection(ignis_model &model, const GqaLayerWeights &weights, const ninfer::Tensor &normalized,
                      ninfer::Tensor &query, ninfer::Tensor &gate, ninfer::Tensor &key, ninfer::Tensor &value,
                      std::int32_t heads, std::int32_t head_dim, ninfer::DeviceArena &arena, LinearPolicyMode mode,
                      cudaStream_t stream) {
  if (model.is_trellis_body) {
    const GqaTrellisWeights &t = weights.trellis;
    ignis::trellis_body::attention_projection(model, t.q_proj, t.k_proj, t.v_proj, normalized, heads, head_dim, arena,
                                              query, gate, key, value, stream);
    return;
  }
  ninfer::ops::attn_input_proj(normalized, weights.query_key_gate_value, query, gate, key, value,
                               ignis_policy_for(weights.query_key_gate_value.qtype, mode), arena, stream);
}

// The layer's input RMSNorm of `input` into `normalized`, then the input
// projection above. GitHub #511: on the dense prefill the norm writes the
// three matrices' INT8 input instead (kernel/src/trellis_body.h,
// fused_producers), and `normalized`, which nothing past the projection
// reads, is not written.
void norm_and_input_projection(ignis_model &model, const GqaLayerWeights &weights, const ninfer::Tensor &input,
                               const ninfer::Tensor &input_norm, ninfer::Tensor &normalized, ninfer::Tensor &query,
                               ninfer::Tensor &gate, ninfer::Tensor &key, ninfer::Tensor &value, std::int32_t heads,
                               std::int32_t head_dim, ninfer::DeviceArena &arena, LinearPolicyMode mode,
                               cudaStream_t stream) {
  if (model.is_trellis_body && ignis::trellis_body::fused_producers(model, input.ne[1])) {
    const GqaTrellisWeights &t = weights.trellis;
    ignis::trellis_body::norm_attention_projection(model, input, input_norm, t.q_proj, t.k_proj, t.v_proj, heads,
                                                   head_dim, query, gate, key, value, stream);
    return;
  }
  ninfer::ops::rmsnorm(input, input_norm, model.rms_norm_eps, /*unit_offset=*/true, normalized, stream);
  input_projection(model, weights, normalized, query, gate, key, value, heads, head_dim, arena, mode, stream);
}

// GitHub #339: the output projection's residual add and the MLP tail on the
// body the load bound (see `input_projection`).
void output_and_mlp(ignis_model &model, const GqaLayerWeights &weights, const ninfer::Tensor &attention,
                    ninfer::Tensor &residual, ninfer::Tensor &post, ninfer::Tensor &fused,
                    ninfer::DeviceArena &arena, LinearPolicyMode mode, std::int32_t tokens, cudaStream_t stream) {
  const auto hidden = static_cast<std::int32_t>(model.hidden);
  const ninfer::Tensor post_norm =
      weight_tensor(weights.post_attention_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
  if (model.is_trellis_body) {
    ignis::trellis_body::output_and_mlp(model, weights.trellis.tail, attention, post_norm, residual, post, fused,
                                        stream);
    return;
  }
  // P2-03 (GitHub #85): the residual add takes the call's mode policy
  // (AllowA4 on the NVFP4 output projection under the engine default, so
  // its W4A4 route is the vendored dispatch's own decision) or A16Only
  // under the override.
  ninfer::ops::linear_add(attention, weights.output, residual, ignis_policy_for(weights.output.qtype, mode), arena,
                          stream);
  ninfer::ops::rmsnorm(residual, post_norm, model.rms_norm_eps, /*unit_offset=*/true, post, stream);
  // P2-03 (GitHub #85): the MLP tail takes the call's mode policy too.
  // The reference's text-model policy is AllowA4 on the NVFP4 gate_up at
  // every width under the engine default, and the route (A16 small-T vs
  // the W4A4/TMA large-T) is the vendored plan's own decision per its
  // token thresholds (GitHub #85's acceptance: the engine encodes no
  // threshold of its own there). Under the test-only A16 override the
  // width-aware `linear_swiglu` counterpart (kernel/src/layer_internal.h)
  // forces the A16 route where it is registered (T<=16) and the op's only
  // runnable route above its A16 cap -- the pre-#85 behavior. The scratch
  // was already reserved for AllowA4 at any T (P2-01, GitHub #83), so a
  // chunk's A4 route needs nothing new.
  ninfer::ops::linear_swiglu(post, weights.mlp_gate_up, fused,
                             ignis_linear_swiglu_policy_for(weights.mlp_gate_up.qtype, mode, tokens), arena,
                             stream);
  ninfer::ops::linear_add(fused, weights.mlp_down, residual, ignis_policy_for(weights.mlp_down.qtype, mode), arena,
                          stream);
}

int32_t run_gqa_layer(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                      uint32_t layer, void *in_residual, void *out_residual,
                      uint32_t gqa_layer, uint64_t num_tokens, LinearPolicyMode mode,
                      const void *rope_positions, const AttentionReadoutTarget *attention_readout) {
  constexpr std::int32_t kHeadDim = 256;
  constexpr std::int32_t kQHeads = kIgnisGqaQHeads;
  constexpr std::int32_t kKvHeads = 4;

  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto tokens = static_cast<std::int32_t>(num_tokens);
  const auto q_width = kHeadDim * kQHeads;
  const auto kv_width = kHeadDim * kKvHeads;
  const auto ffn = ignis::trellis_body::mlp_width(*model, model->layers[layer].gqa.trellis.tail,
                                                  model->layers[layer].gqa.mlp_gate_up);
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const GqaLayerWeights &weights = model->layers[layer].gqa;
  const auto stream = model->stream;
  const auto start_position = static_cast<std::int32_t>(seq->gqa_positions[gqa_layer]);

  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    ninfer::Tensor normalized = model->scratch->alloc(ninfer::DType::BF16, {hidden, tokens, 1, 1});
    ninfer::Tensor query = model->scratch->alloc(ninfer::DType::BF16, {q_width, tokens, 1, 1});
    ninfer::Tensor key = model->scratch->alloc(ninfer::DType::BF16, {kv_width, tokens, 1, 1});
    ninfer::Tensor gate = model->scratch->alloc(ninfer::DType::BF16, {q_width, tokens, 1, 1});
    ninfer::Tensor value = model->scratch->alloc(ninfer::DType::BF16, {kv_width, tokens, 1, 1});
    ninfer::Tensor rotated_query = model->scratch->alloc(ninfer::DType::BF16, {q_width, tokens, 1, 1});
    ninfer::Tensor rotated_key = model->scratch->alloc(ninfer::DType::BF16, {kv_width, tokens, 1, 1});
    ninfer::Tensor positions = model->scratch->alloc(ninfer::DType::I32, {tokens, 1, 1, 1});
    ninfer::Tensor attention = model->scratch->alloc(ninfer::DType::BF16, {q_width, tokens, 1, 1});
    ninfer::Tensor post = model->scratch->alloc(ninfer::DType::BF16, {hidden, tokens, 1, 1});
    ninfer::Tensor fused = model->scratch->alloc(ninfer::DType::BF16, {ffn, tokens, 1, 1});

    const ninfer::Tensor input(in_residual, ninfer::DType::BF16, {hidden, tokens, 1, 1});
    const ninfer::Tensor input_norm =
        weight_tensor(weights.input_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});

    // The input RMSNorm, then the projection (GitHub #511: one fused pass on
    // the dense prefill). P2-03 (GitHub #85): the reference's own activation-quantization policy
    // under the call's mode (ADR 0016): AllowA4 on the NVFP4 projection when
    // the mode is the engine default -- the W4A4 (MMA/TMA) route is then the
    // vendored dispatch's own decision per its token thresholds, not ours --
    // or A16Only when the override forces it. The transient workspace comes
    // from the model's scratch (reserved at load for the widest policy,
    // P2-01, GitHub #83), so a chunk's A4 route needs nothing new. GitHub
    // #339: the trellis body's three matrices instead.
    norm_and_input_projection(*model, weights, input, input_norm, normalized, query, gate, key, value, kQHeads,
                              kHeadDim, *model->scratch, mode, stream);

    // The future full prefill/decode step supplies the same offset to every
    // GQA layer for a token, so positions advance per token, never per layer.
    ninfer::ops::fill_i32_positions(positions, start_position, stream);
    // GitHub #178: what the span rotates at -- a multimodal chunk's three
    // axes, a multimodal sequence's positions shifted by its rope delta, or
    // (every text sequence) the cache positions above, as before.
    ninfer::Tensor rotation_positions = positions;
    if (rope_positions != nullptr) {
      rotation_positions = ninfer::Tensor(const_cast<void *>(rope_positions), ninfer::DType::I32,
                                          {tokens, 3, 1, 1});
    } else if (seq->rope_delta != 0) {
      rotation_positions = model->scratch->alloc(ninfer::DType::I32, {tokens, 1, 1, 1});
      ninfer::ops::fill_i32_positions(rotation_positions, start_position + seq->rope_delta, stream);
    }
    const ninfer::Tensor q_norm =
        weight_tensor(weights.query_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    const ninfer::Tensor k_norm =
        weight_tensor(weights.key_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    // GitHub #227: the load's own table -- linear, or YaRN when the
    // operator scaled the rotary envelope. Built once at load
    // (kernel/src/rope_scaling.h), not rebuilt per layer per call.
    const ninfer::ops::RopeFrequencies &rope = model->text_rope;
    ninfer::Tensor query_heads = query.view({kHeadDim, kQHeads, tokens, 1});
    ninfer::Tensor key_heads = key.view({kHeadDim, kKvHeads, tokens, 1});
    ninfer::Tensor rotated_query_heads = rotated_query.view({kHeadDim, kQHeads, tokens, 1});
    ninfer::Tensor rotated_key_heads = rotated_key.view({kHeadDim, kKvHeads, tokens, 1});
    ninfer::ops::qk_norm_rope(query_heads, key_heads, q_norm, k_norm, model->rms_norm_eps,
                              rotation_positions, rope, rotated_query_heads, rotated_key_heads,
                              stream);

    // Test-only attention-input tap (kernel/include/ignis_attn_tap.h): one
    // flag load when disarmed, which is the only way production ever runs.
    if (ignis_attn_tap_record(gqa_layer, start_position, tokens, rotated_query.data,
                              rotated_key.data, stream) != 0) {
      set_error(std::string("ignis_gqa_layer: ") + ignis_attn_tap_last_error());
      return -1;
    }

    // P2-04 (GitHub #86): the fused append-and-attend entry point (A1)
    // replaces the two-pass A2 (gqa_kv_append) + A3 (gqa_attention_cached)
    // composition: the chunk's keys and values are appended to the paged
    // cache at the tokens' absolute positions and attended in one pass,
    // with the sigmoid output gate applied inside the op exactly as A3 did.
    // The op's own route resolver picks the kernel from the width and the
    // cache's declared dtype: the prompt route above the small-T width
    // (every chunked prefill width, B=1), the small-T route for the T=1
    // decode and per-token calls -- the engine encodes no threshold of its
    // own, and no format branch either. Declaring the cache U8 is what
    // selects the hq prefill/decode kernels over the BF16 ones (P4-05,
    // GitHub #123); the small-T tile is 8 tokens under hq against BF16's 6,
    // which is the resolver's business, not this layer's. The shared
    // numerical contract of A1/A2/A3
    // (kernel/vendor/include/ninfer/ops/gqa_attention.h) keeps the layer's
    // output unchanged in form under either.
    // The pool stores one K/V plane run per GQA layer and allocates one
    // block-table row per sequence slot; `ignis_kv_batch_layer_view`
    // (kernel/include/ignis_seq_internal.h) builds the view from the pool's
    // own KV format (GitHub #122/#123) and is shared with the leaf's own
    // append and route-agreement tests, so the hq plane selection -- and the
    // U8 + quant_group 32 declaration that routes A1 to the hq kernels at
    // all (P4-05, GitHub #123) -- is the one under test rather than a second
    // copy of it.
    //
    // A1 takes the batched cache view. This path is single-sequence: the
    // sequence's own block-table row replaces the pool-wide matrix as the
    // complete [logical_pages, 1] table, and table row 0 selects it (the
    // kv_table_rows buffer below), so a chunk can only ever address its own
    // pages.
    ninfer::PagedKVBatchLayerView batch_cache =
        ignis_kv_batch_layer_view(pool, static_cast<std::int32_t>(gqa_layer));
    const ninfer::Tensor seq_block_table = seq->kv.block_table();
    batch_cache.block_tables = seq_block_table.view({seq_block_table.ne[0], 1});
    // GitHub #257: the hq residual window narrows the same way. It is indexed
    // by the table row the kernels select, and that row is 0 of this
    // one-row table, so the window is pre-sliced to the sequence's own slot:
    // a chunk reads and writes its own sink and ring rows, never lane 0's.
    ignis_kv_fill_residual(batch_cache, pool, static_cast<std::int32_t>(gqa_layer), seq->slot, 1);
    // The B=1 table-row selector: one device I32 holding 0, bumped from the
    // load-time reservation (an arena bump like every other per-layer
    // buffer here, not an allocation; the arena scope resets after the
    // layer).
    const ninfer::Tensor kv_table_rows = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
    cudaError_t error = cudaMemsetAsync(kv_table_rows.data, 0, sizeof(std::int32_t), stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_step: cudaMemsetAsync(kv table rows) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    ninfer::Tensor value_heads = value.view({kHeadDim, kKvHeads, tokens, 1});
    const ninfer::ops::GqaExecutionEnvelope envelope{
        .min_visible_keys = 1,
        .max_visible_keys = static_cast<std::uint32_t>(start_position + tokens),
    };
    // The small-T attention route reserves a fixed number of partial split
    // slots, while only the active prefix is written for a short history: a
    // fresh, zeroed workspace per layer invocation keeps stale inactive
    // partials from being reduced on the next call (P2-02, GitHub #84) --
    // on the formats whose kernels leave those slots untouched, which is the
    // question `ignis_gqa_workspace_needs_zeroing` answers below. At
    // a chunk-sized width the resolver's B=1 prompt route legally needs none
    // ("a legal B=1 prompt route may return zero for BF16/INT8 caches"),
    // but neither `DeviceArena` constructor admits zero capacity, so at
    // least one byte is reserved regardless -- the op only ever touches
    // what its own query reported.
    // Sized from the cache's *declared* dtype, never a constant: an hq
    // prefill chunk takes ignis's prompt attention (below), whose band and
    // split states BF16 has no counterpart of, so asking as BF16 here would
    // under-reserve out of the arena `ignis_model_load` already sized for
    // this format (P4-05, GitHub #123). It goes through
    // `ignis_gqa_attention_workspace_bytes` (kernel/include/
    // ignis_gqa_workspace.h), which answers for whichever op the width
    // takes.
    const std::size_t attention_workspace_bytes =
        ignis_gqa_attention_workspace_bytes(batch_cache.dtype, envelope, tokens);
    const ninfer::DeviceSpan attention_workspace_storage =
        model->scratch->alloc_bytes(std::max<std::size_t>(attention_workspace_bytes, 1));
    if (ignis_gqa_workspace_needs_zeroing(batch_cache.dtype)) {
      error = cudaMemsetAsync(attention_workspace_storage.data, 0,
                              attention_workspace_storage.bytes, stream);
      if (error != cudaSuccess) {
        set_error(
            std::string("ignis_gqa_layer_step: cudaMemsetAsync(attention workspace) failed: ") +
            cudaGetErrorString(error));
        return -1;
      }
    }
    ninfer::DeviceArena attention_workspace(attention_workspace_storage);
    ninfer::Tensor attention_heads = attention.view({kHeadDim, kQHeads, tokens, 1});
    // GitHub #335 (ADR 0061): every hq-e8-2b chunk the resolver sends to the
    // Prompt route takes ignis's prompt attention, which reads the history
    // without prompt planes (kernel/include/ignis_hq_prompt_attention.h). It
    // takes the sequence's own layer view and keeps A1's order of append and
    // attention. The vendored route keeps every other call: BF16 caches and
    // the small-T widths.
    const bool hq_prompt =
        batch_cache.dtype == ninfer::DType::U8 &&
        ninfer::ops::detail::gqa_attention_resolve_route(kQHeads, tokens, 1, batch_cache.dtype,
                                                         envelope) ==
            ninfer::ops::detail::GqaAttentionRoute::Prompt;
    // GitHub #260 (ADR 0038): the attention readout, scored here and nowhere
    // else -- after this layer's attention and inside its scope, while the
    // keys that attention read are still where it read them: under hq before
    // the chunk's attend-first append (the ring still holds the keys before
    // the chunk), which is why ignis's attention runs it; under BF16 after
    // A1, which appended the pages before it read them. The test-only tap
    // (kernel/include/ignis_attn_tap.h) captures the consumed keys at the
    // same point.
    const char *readout_error = nullptr;
    bool readout_failed = false;
    const auto readout = [&](const IgnisHqPromptKeys *hq_keys) {
      if (ignis_attn_tap_record_consumed(gqa_layer, start_position, tokens, hq_keys, stream) != 0) {
        readout_error = ignis_attn_tap_last_error();
        readout_failed = true;
        return;
      }
      if (attention_readout != nullptr &&
          ignis_attention_readout_run(
              *attention_readout, rotated_query.data, kQHeads, kKvHeads, tokens,
              static_cast<std::int64_t>(envelope.max_visible_keys), batch_cache, hq_keys,
              attention_scale, stream, &readout_error) != 0) {
        readout_failed = true;
      }
    };
    if (hq_prompt) {
      IgnisHqPromptOptions options;
      options.precision = model->prefill_attention == IGNIS_PREFILL_ATTENTION_FP16   ? IgnisHqPromptPrecision::kFp16
                          : model->prefill_attention == IGNIS_PREFILL_ATTENTION_INT8 ? IgnisHqPromptPrecision::kInt8
                                                                                     : IgnisHqPromptPrecision::kFp8;
      options.after_attention = [&](const IgnisHqPromptKeys &keys) { readout(&keys); };
      ignis_hq_prompt_attention(rotated_query_heads, rotated_key_heads, value_heads, positions,
                                start_position, gate.view({kHeadDim, kQHeads, tokens, 1}),
                                attention_scale,
                                ignis_kv_layer_view(pool, seq, static_cast<std::int32_t>(gqa_layer)),
                                attention_workspace, attention_heads, stream, options);
    } else {
      ninfer::ops::gqa_attention(
          rotated_query_heads, rotated_key_heads, value_heads, positions,
          /*valid_columns=*/ninfer::Tensor{}, kv_table_rows,
          gate.view({kHeadDim, kQHeads, tokens, 1}), attention_scale, batch_cache, envelope,
          attention_workspace, attention_heads, stream);
      readout(nullptr);
    }
    if (readout_failed) {
      set_error(std::string("ignis_gqa_layer: ") + readout_error);
      return -1;
    }

    ninfer::Tensor residual(out_residual, ninfer::DType::BF16, {hidden, tokens, 1, 1});
    error = cudaMemcpyAsync(out_residual, in_residual,
                            static_cast<std::size_t>(hidden) * tokens * sizeof(uint16_t),
                            cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_step: cudaMemcpyAsync(residual) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    output_and_mlp(*model, weights, attention, residual, post, fused, *model->scratch, mode, tokens, stream);

    // No stream synchronization here (P2-01, GitHub #83): the layer body
    // only enqueues work, so a later chunk loop can run every layer as one
    // pipelined unit. `ignis_gqa_layer_step` below synchronizes -- and only
    // then advances `gqa_positions` -- once the layer body returns, so a
    // failed synchronize never leaves the position counter ahead of
    // confirmed device work.
    return 0;
  } catch (const std::exception &error) {
    set_error(std::string("ignis_gqa_layer_step: ") + error.what());
    return -1;
  }
}

// GitHub #111: the graph-safe counterpart of `run_gqa_layer` for a
// whole decode round -- `width` lanes at one token each, traversed once as a
// [.., width] batch, not `width` times at batch 1. Every address either
// belongs to the model/pool for their lifetime or is a stable staging
// buffer the round refreshes in place, so the same kernel launches are
// correct at capture time (arbitrary staged content) and at every later
// replay or direct eager execution (this round's real content, refreshed by
// the caller) -- see the declaration in layer_internal.h and ADR 0019 for
// the full argument.
//
// P5-04 (GitHub #153) generalizes the body from one token per lane to
// `lane_tokens` columns per lane: the verify round traverses every lane's
// `k+1` verify columns (anchor + drafts, the tail padded with the anchor)
// as a [.., lane_tokens, batch] batch. `positions` is then the round's
// staged I32 [lane_tokens, batch] matrix (`base + min(j, extent)` per
// column) and `valid_columns` its I32 [batch] extents (extent + 1), so A1
// appends and attends exactly a lane's valid prefix and writes zero for the
// padded tail -- the masked form of A1's own contract. The decode round is
// the `lane_tokens == 1` case with the dense (empty) mask, unchanged.
int32_t run_gqa_layer_batch(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                            uint32_t width, std::int32_t lane_tokens, const void *positions_ptr,
                            const void *rope_positions_ptr,
                            const void *valid_columns_ptr, void *in_residual, void *out_residual,
                            uint32_t gqa_layer, LinearPolicyMode mode, const char *op) {
  constexpr std::int32_t kHeadDim = 256;
  constexpr std::int32_t kQHeads = kIgnisGqaQHeads;
  constexpr std::int32_t kKvHeads = 4;

  // The batch's rows are the lanes; every row carries `lane_tokens`
  // columns, lane-major, so the activations are `columns` wide.
  const auto batch = static_cast<std::int32_t>(width);
  const std::int32_t columns = lane_tokens * batch;
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto q_width = kHeadDim * kQHeads;
  const auto kv_width = kHeadDim * kKvHeads;
  const auto ffn = ignis::trellis_body::mlp_width(*model, model->layers[layer].gqa.trellis.tail,
                                                  model->layers[layer].gqa.mlp_gate_up);
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const GqaLayerWeights &weights = model->layers[layer].gqa;
  const auto stream = model->stream;

  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    ninfer::Tensor normalized =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::Tensor query =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, columns, 1, 1});
    ninfer::Tensor key =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor gate =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, columns, 1, 1});
    ninfer::Tensor value =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor rotated_query =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, columns, 1, 1});
    ninfer::Tensor rotated_key =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {kv_width, columns, 1, 1});
    ninfer::Tensor attention =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, columns, 1, 1});
    ninfer::Tensor post = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::Tensor fused = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {ffn, columns, 1, 1});

    const ninfer::Tensor input(in_residual, ninfer::DType::BF16, {hidden, columns, 1, 1});
    const ninfer::Tensor input_norm =
        weight_tensor(weights.input_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(input, input_norm, model->rms_norm_eps, /*unit_offset=*/true,
                         normalized, stream);

    input_projection(*model, weights, normalized, query, gate, key, value, kQHeads, kHeadDim,
                     *model->decode_graph_scratch, mode, stream);

    // Graph-safe positions (ADR 0019): the round's staged absolute
    // positions, read from device memory at replay -- `fill_i32_positions`'s
    // host-scalar `start` would freeze this round's positions into the graph
    // forever. The lanes' positions are unrelated to one another, so this is
    // read as-is rather than derived from a base; it is the same contiguous
    // I32 [lane_tokens, batch] buffer both `qk_norm_rope` (positions [T],
    // read per column) and `gqa_attention` (positions [W,B]) want, and both
    // only read it. Two shapes over the same buffer, because the ops name
    // the batch differently.
    const ninfer::Tensor positions(const_cast<void *>(positions_ptr), ninfer::DType::I32,
                                   {columns, 1, 1, 1});
    const ninfer::Tensor positions_rows(const_cast<void *>(positions_ptr), ninfer::DType::I32,
                                        {lane_tokens, batch, 1, 1});
    // GitHub #178: what the columns rotate at, staged at its own stable
    // address on a vision load (`position + rope_delta`); the positions
    // above otherwise.
    const ninfer::Tensor rotation_positions(const_cast<void *>(rope_positions_ptr),
                                            ninfer::DType::I32, {columns, 1, 1, 1});
    // The masked form (P5-04): each lane's valid prefix, or the dense form
    // (an empty tensor) for the decode round.
    const ninfer::Tensor valid_columns =
        valid_columns_ptr == nullptr
            ? ninfer::Tensor{}
            : ninfer::Tensor(const_cast<void *>(valid_columns_ptr), ninfer::DType::I32,
                             {batch, 1, 1, 1});

    const ninfer::Tensor q_norm =
        weight_tensor(weights.query_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    const ninfer::Tensor k_norm =
        weight_tensor(weights.key_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    // GitHub #227: the load's own table -- linear, or YaRN when the
    // operator scaled the rotary envelope. Built once at load
    // (kernel/src/rope_scaling.h), not rebuilt per layer per call.
    const ninfer::ops::RopeFrequencies &rope = model->text_rope;
    // Two views of the same contiguous storage, because the two ops name
    // the batch differently: `qk_norm_rope` takes flat `[256,Hq|Hkv,T]` and
    // reads one position per column, so the round's columns are simply its
    // T; A1 below takes `[256,Hq|Hkv,W,B]` and needs the lanes on the batch
    // axis. The element order is identical either way -- lane-major, a
    // lane's columns adjacent -- so this is a reshape, not a copy.
    ninfer::Tensor query_tokens = query.view({kHeadDim, kQHeads, columns, 1});
    ninfer::Tensor key_tokens = key.view({kHeadDim, kKvHeads, columns, 1});
    ninfer::Tensor rotated_query_tokens = rotated_query.view({kHeadDim, kQHeads, columns, 1});
    ninfer::Tensor rotated_key_tokens = rotated_key.view({kHeadDim, kKvHeads, columns, 1});
    ninfer::ops::qk_norm_rope(query_tokens, key_tokens, q_norm, k_norm, model->rms_norm_eps,
                              rotation_positions, rope, rotated_query_tokens, rotated_key_tokens,
                              stream);
    ninfer::Tensor rotated_query_heads = rotated_query.view({kHeadDim, kQHeads, lane_tokens, batch});
    ninfer::Tensor rotated_key_heads = rotated_key.view({kHeadDim, kKvHeads, lane_tokens, batch});

    // ADR 0019: the pool-wide block-table matrix (every physical slot, a
    // fixed address independent of round composition) with `kv_table_rows`
    // read from this round's staged physical slots -- unlike the eager
    // per-token path, which narrows the same view to one sequence's own row.
    // Row b of the A1 batch selects lane b's own KV pages, which is what
    // keeps the lanes' caches isolated inside one call. Built by the one
    // format-aware view builder (kernel/include/ignis_seq_internal.h), so an
    // hq round routes to the hq decode kernels here exactly as the eager
    // path does (P4-05, GitHub #123), and every address it names still
    // belongs to the pool for its lifetime -- which is what keeps the
    // capture safe.
    const ninfer::PagedKVBatchLayerView batch_cache =
        ignis_kv_batch_layer_view(pool, static_cast<std::int32_t>(gqa_layer));
    const ninfer::Tensor kv_table_rows(model->decode_graph_slots->p, ninfer::DType::I32,
                                       {batch, 1, 1, 1});
    ninfer::Tensor value_heads = value.view({kHeadDim, kKvHeads, lane_tokens, batch});
    // A fixed, conservative envelope (ADR 0019): `max_visible_keys` is a
    // host launch-resource promise for workspace sizing and kernel-route
    // selection, not the causal mask (the mask comes from `positions`
    // above, read per row inside the kernel) -- so the sequence pool's
    // configured cap is always a safe over-approximation, regardless of any
    // lane's actual current position, and lets one captured graph serve
    // every round at this width.
    const ninfer::ops::GqaExecutionEnvelope envelope{
        .min_visible_keys = 1,
        .max_visible_keys = model->max_context_tokens,
    };
    // The cache's own declared dtype, as on the eager path: at one token per
    // lane both formats resolve to the small-T route, but the reservation
    // `ignis_model_load` made for `decode_graph_scratch` was made under this
    // format and the query has to be asked the same way (GitHub #123). At
    // the verify width the resolver's own chunking decides the route -- one
    // hq width-8 tile, or BF16's two passes of six -- not this layer.
    const std::size_t attention_workspace_bytes = ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, batch_cache.dtype, envelope, batch, lane_tokens, lane_tokens);
    const ninfer::DeviceSpan attention_workspace_storage =
        model->decode_graph_scratch->alloc_bytes(std::max<std::size_t>(attention_workspace_bytes, 1));
    // Graph-safe (ADR 0019): the predicate is the pool's KV format, fixed for
    // the life of the load (ADR 0022), so which nodes this body records does
    // not depend on anything a replay can change.
    cudaError_t error = cudaSuccess;
    if (ignis_gqa_workspace_needs_zeroing(batch_cache.dtype)) {
      error = cudaMemsetAsync(attention_workspace_storage.data, 0,
                              attention_workspace_storage.bytes, stream);
      if (error != cudaSuccess) {
        set_error(std::string(op) + ": cudaMemsetAsync(attention workspace) failed: " +
                  cudaGetErrorString(error));
        return -1;
      }
    }
    ninfer::DeviceArena attention_workspace(attention_workspace_storage);
    ninfer::Tensor attention_heads = attention.view({kHeadDim, kQHeads, lane_tokens, batch});
    ninfer::ops::gqa_attention(
        rotated_query_heads, rotated_key_heads, value_heads, positions_rows, valid_columns,
        kv_table_rows, gate.view({kHeadDim, kQHeads, lane_tokens, batch}), attention_scale,
        batch_cache, envelope, attention_workspace, attention_heads, stream);

    ninfer::Tensor residual(out_residual, ninfer::DType::BF16, {hidden, columns, 1, 1});
    error = cudaMemcpyAsync(out_residual, in_residual,
                            static_cast<std::size_t>(hidden) * columns * sizeof(uint16_t),
                            cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      set_error(std::string(op) + ": cudaMemcpyAsync(residual) failed: " +
                cudaGetErrorString(error));
      return -1;
    }
    output_and_mlp(*model, weights, attention, residual, post, fused, *model->decode_graph_scratch, mode, columns,
                   stream);
    return 0;
  } catch (const std::exception &error) {
    set_error(std::string(op) + ": " + error.what());
    return -1;
  }
}

// The decode round's shape of the body above: one token per lane, positions
// from `sampling_decode_positions`, the dense mask.
int32_t run_gqa_layer_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                            uint32_t width, void *in_residual, void *out_residual,
                            uint32_t gqa_layer, LinearPolicyMode mode) {
  return run_gqa_layer_batch(model, pool, layer, width, /*lane_tokens=*/1,
                             model->sampling_decode_positions->p,
                             model->decode_rope_positions != nullptr
                                 ? model->decode_rope_positions->p
                                 : model->sampling_decode_positions->p,
                             /*valid_columns_ptr=*/nullptr,
                             in_residual, out_residual, gqa_layer, mode, "ignis_gqa_layer_graph");
}

// P4-05 (GitHub #123): both formats serve, and which routes a layer takes is
// the pool's format alone -- but the arena those routes allocate their
// attention workspace from was reserved once, at `ignis_model_load`, for the
// format that call named (kernel/src/model.cu). The hq prompt route's
// rotated-frame scratch planes are the largest single item in that
// reservation, so a pool whose format differs from the load's would bump an
// arena sized for the other one. That is a caller error with a quiet failure
// mode (an arena throw deep inside a chunk, or a silent over-reservation), so
// it is refused by name here, before any device work.
bool gqa_format_matches_load(ignis_model *model, ignis_seq_pool *pool, const char *op) {
  if (pool->kv_format == model->kv_format) {
    return true;
  }
  set_error(std::string(op) + ": the sequence pool's KV format is " +
            ignis_kv_format_name(pool->kv_format) + ", not the " +
            ignis_kv_format_name(model->kv_format) +
            " this model was loaded with; the format is fixed for the life of a load (ADR 0022)");
  return false;
}

}  // namespace

// P2-02 (GitHub #84): the validated body a chunk loop dispatches directly
// (kernel/src/layer_internal.h) -- every check `ignis_gqa_layer_step` did,
// minus the synchronization and position advance a per-chunk caller defers
// until its whole chunk's dispatches succeed. `mode` is the call's
// compute-policy mode (P2-03, GitHub #85): every NVFP4 projection in the
// body is dispatched under the policy `ignis_policy_for` resolves for it.
int32_t ignis_gqa_layer_run_body(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                 uint32_t layer, const void *in_residual, void *out_residual,
                                 uint64_t num_tokens, LinearPolicyMode mode,
                                 const void *rope_positions,
                                 const AttentionReadoutTarget *attention_readout) {
  // Errors here are prefixed `ignis_gqa_layer` (not `..._step`): this body
  // is now dispatched both by `ignis_gqa_layer_step` and directly by a
  // chunk loop (kernel/src/step.cu), so a message naming the ABI wrapper
  // would misattribute a chunked-prefill failure.
  if (model == nullptr || pool == nullptr || seq == nullptr || in_residual == nullptr ||
      out_residual == nullptr) {
    set_error("ignis_gqa_layer: null argument");
    return -1;
  }
  if (num_tokens == 0) {
    set_error("ignis_gqa_layer: num_tokens must be positive");
    return -1;
  }
  if (!gqa_format_matches_load(model, pool, "ignis_gqa_layer")) {
    return -1;
  }
  if (layer >= model->layers.size() || model->layers[layer].kind != IGNIS_LAYER_GQA) {
    set_error("ignis_gqa_layer: layer " + std::to_string(layer) + " is not a GQA layer");
    return -1;
  }
  if (layer < 3 || (layer + 1) % 4 != 0) {
    set_error("ignis_gqa_layer: GQA layer index is not in the Qwen 3.8 topology");
    return -1;
  }
  const uint32_t gqa_layer = ignis_gqa_relative_layer(layer);
  const uint64_t start_position = seq->gqa_positions[gqa_layer];
  if (num_tokens > static_cast<uint64_t>(INT32_MAX) - start_position ||
      start_position + num_tokens > ignis_seq_token_capacity(*seq)) {
    set_error("ignis_gqa_layer: token positions exceed the sequence KV capacity");
    return -1;
  }
  return run_gqa_layer(model, pool, seq, layer, const_cast<void *>(in_residual), out_residual,
                       gqa_layer, num_tokens, mode, rope_positions, attention_readout);
}

// P2-03 (GitHub #85): `ignis_gqa_layer_step`'s synchronous contract (body +
// one stream synchronization + the deferred `gqa_positions` advance) with a
// non-default compute-policy mode -- the program's per-token route
// (kernel/src/step.cu) threads ADR 0016's `compute_policy` override
// through it. The flat C ABI entry point below is this same contract with
// `kEngineDefault`, so the ABI callers keep seeing the engine's own
// policy, not an override.
int32_t ignis_gqa_layer_step_mode(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens, LinearPolicyMode mode) {
  const int32_t rc =
      ignis_gqa_layer_run_body(model, pool, seq, layer, in_residual, out_residual, num_tokens,
                               mode);
  if (rc != 0) {
    return rc;
  }
  const uint32_t gqa_layer = ignis_gqa_relative_layer(layer);
  // P2-01 (GitHub #83): the sync moved here from the layer body so a direct
  // call through this ABI entry point (this function's own GPU tests) keeps
  // seeing a synchronous result, while a caller that dispatches the body
  // directly (the program's per-chunk loop) can pipeline every layer.
  const cudaError_t error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess) {
    set_error(std::string("ignis_gqa_layer_step: cudaStreamSynchronize failed: ") +
              cudaGetErrorString(error));
    return -1;
  }
  // Only advance the position counter once the synchronize above confirms
  // the layer's device work actually completed -- moving the sync out of
  // the layer body must not let this counter get ahead of reality on a
  // failed synchronize.
  seq->gqa_positions[gqa_layer] += static_cast<std::uint32_t>(num_tokens);
  return 0;
}

// GitHub #111: validates and dispatches `run_gqa_layer_graph` --
// called once per GQA layer per decode round (not per lane) by
// `kernel/src/decode_graph.cu`, either while a graph is being captured or
// when the round runs eagerly at a width whose capture failed. It does no
// host-visible branching on device data and no stream synchronization, so
// replaying the captured launches is equivalent to running them directly --
// exactly what `run_gqa_layer_graph` above upholds.
int32_t ignis_gqa_layer_run_body_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                                       uint32_t width, const void *in_residual, void *out_residual,
                                       LinearPolicyMode mode) {
  if (model == nullptr || pool == nullptr || in_residual == nullptr || out_residual == nullptr) {
    set_error("ignis_gqa_layer_graph: null argument");
    return -1;
  }
  if (layer >= model->layers.size() || model->layers[layer].kind != IGNIS_LAYER_GQA) {
    set_error("ignis_gqa_layer_graph: layer " + std::to_string(layer) + " is not a GQA layer");
    return -1;
  }
  // GitHub #336: the load's lanes, which every round buffer is sized for.
  if (width == 0 || width > model->decode_lanes) {
    set_error("ignis_gqa_layer_graph: width " + std::to_string(width) + " is not in 1.." +
              std::to_string(model->decode_lanes) + ", the load's decode lanes");
    return -1;
  }
  if (!gqa_format_matches_load(model, pool, "ignis_gqa_layer_graph")) {
    return -1;
  }
  const uint32_t gqa_layer = ignis_gqa_relative_layer(layer);
  return run_gqa_layer_graph(model, pool, layer, width, const_cast<void *>(in_residual), out_residual,
                             gqa_layer, mode);
}

// P5-04 (GitHub #153): the verify round's GQA body -- `width` lanes at the
// load's `k+1` verify columns each, positions and valid extents from the
// verify substrate's staging (model_internal.h `IgnisVerifyRound`), dispatched
// once per GQA layer per round by `ignis_verify_graph_run_batch`
// (kernel/src/decode_graph.cu) under capture, replay or eager alike.
int32_t ignis_gqa_layer_run_body_verify(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                                        uint32_t width, const void *in_residual, void *out_residual,
                                        LinearPolicyMode mode) {
  if (model == nullptr || pool == nullptr || in_residual == nullptr || out_residual == nullptr) {
    set_error("ignis_gqa_layer_verify: null argument");
    return -1;
  }
  if (model->verify == nullptr) {
    set_error("ignis_gqa_layer_verify: the model was loaded without a draft window");
    return -1;
  }
  if (layer >= model->layers.size() || model->layers[layer].kind != IGNIS_LAYER_GQA) {
    set_error("ignis_gqa_layer_verify: layer " + std::to_string(layer) + " is not a GQA layer");
    return -1;
  }
  if (width == 0 || width > model->decode_lanes) {
    set_error("ignis_gqa_layer_verify: width " + std::to_string(width) + " is not in 1.." +
              std::to_string(model->decode_lanes) + ", the load's decode lanes");
    return -1;
  }
  if (!gqa_format_matches_load(model, pool, "ignis_gqa_layer_verify")) {
    return -1;
  }
  const uint32_t gqa_layer = ignis_gqa_relative_layer(layer);
  const IgnisVerifyRound &verify = *model->verify;
  // GitHub #195: a vision load rotates the round's columns at `position +
  // rope_delta` from its own staging -- the decode round's rule, at the
  // verify round's column shape. Without vision there is no such buffer and
  // the columns rotate at their cache positions, exactly as before.
  const void *rotation =
      verify.rope_positions != nullptr ? verify.rope_positions->p : verify.positions->p;
  return run_gqa_layer_batch(model, pool, layer, width, static_cast<std::int32_t>(verify.window + 1),
                             verify.positions->p, rotation, verify.valid_columns->p,
                             const_cast<void *>(in_residual), out_residual, gqa_layer, mode,
                             "ignis_gqa_layer_verify");
}

extern "C" int32_t ignis_gqa_layer_step(struct ignis_model *model, struct ignis_seq_pool *pool,
                                         struct ignis_seq *seq, uint32_t layer,
                                         const void *in_residual, void *out_residual,
                                         uint64_t num_tokens) {
  return ignis_gqa_layer_step_mode(model, pool, seq, layer, in_residual, out_residual, num_tokens,
                                   LinearPolicyMode::kEngineDefault);
}

extern "C" const char *ignis_gqa_layer_last_error(void) {
  return g_last_error.c_str();
}
