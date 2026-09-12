// ignis kernel leaf - P1-21 (GitHub #57): one GQA layer in the device-resident
// program. The program layer is ours (ADR 0009); every dispatched op is the
// ADR 0010 vendored reference implementation. The order is the Qwen text
// context: input norm -> fused Q/K/gate/V projection -> q/k norm + RoPE ->
// KV append -> attention + sigmoid output gate -> output residual -> MLP tail.

#include "ignis_gqa_layer.h"

#include "ignis_gqa_workspace.h"
#include "ignis_seq_internal.h"
#include "layer_internal.h"
#include "model_internal.h"

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

int32_t run_gqa_layer(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                      uint32_t layer, void *in_residual, void *out_residual,
                      uint32_t gqa_layer, uint64_t num_tokens, LinearPolicyMode mode) {
  constexpr std::int32_t kHeadDim = 256;
  constexpr std::int32_t kQHeads = kIgnisGqaQHeads;
  constexpr std::int32_t kKvHeads = 4;
  constexpr std::int32_t kRotaryDim = 64;
  constexpr float kRopeTheta = 10'000'000.0F;

  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto tokens = static_cast<std::int32_t>(num_tokens);
  const auto q_width = kHeadDim * kQHeads;
  const auto kv_width = kHeadDim * kKvHeads;
  const auto ffn = model->layers[layer].gqa.mlp_gate_up.n / 2;
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
    ninfer::ops::rmsnorm(input, input_norm, model->rms_norm_eps, /*unit_offset=*/true,
                         normalized, stream);

    // P2-03 (GitHub #85): the reference's own activation-quantization policy
    // under the call's mode (ADR 0016): AllowA4 on the NVFP4 projection when
    // the mode is the engine default -- the W4A4 (MMA/TMA) route is then the
    // vendored dispatch's own decision per its token thresholds, not ours --
    // or A16Only when the override forces it. The transient workspace comes
    // from the model's scratch (reserved at load for the widest policy,
    // P2-01, GitHub #83), so a chunk's A4 route needs nothing new.
    ninfer::ops::attn_input_proj(normalized, weights.query_key_gate_value, query, gate, key,
                                 value, ignis_policy_for(weights.query_key_gate_value.qtype, mode),
                                 *model->scratch, stream);

    // The future full prefill/decode step supplies the same offset to every
    // GQA layer for a token, so positions advance per token, never per layer.
    ninfer::ops::fill_i32_positions(positions, start_position, stream);
    const ninfer::Tensor q_norm =
        weight_tensor(weights.query_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    const ninfer::Tensor k_norm =
        weight_tensor(weights.key_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    const ninfer::ops::RopeFrequencies rope =
        ninfer::ops::rope_linear_frequencies(kRopeTheta, kRotaryDim);
    ninfer::Tensor query_heads = query.view({kHeadDim, kQHeads, tokens, 1});
    ninfer::Tensor key_heads = key.view({kHeadDim, kKvHeads, tokens, 1});
    ninfer::Tensor rotated_query_heads = rotated_query.view({kHeadDim, kQHeads, tokens, 1});
    ninfer::Tensor rotated_key_heads = rotated_key.view({kHeadDim, kKvHeads, tokens, 1});
    ninfer::ops::qk_norm_rope(query_heads, key_heads, q_norm, k_norm, model->rms_norm_eps,
                              positions, rope, rotated_query_heads, rotated_key_heads, stream);

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
    // partials from being reduced on the next call (P2-02, GitHub #84). At
    // a chunk-sized width the resolver's B=1 prompt route legally needs none
    // ("a legal B=1 prompt route may return zero for BF16/INT8 caches"),
    // but neither `DeviceArena` constructor admits zero capacity, so at
    // least one byte is reserved regardless -- the op only ever touches
    // what its own query reported.
    // Sized from the cache's *declared* dtype, never a constant: the hq
    // prompt route additionally materializes the envelope's visible history
    // into two rotated-frame BF16 scratch planes, so asking as BF16 here
    // would under-reserve out of the arena `ignis_model_load` already sized
    // for this format (P4-05, GitHub #123). It goes through
    // `ignis_gqa_attention_workspace_bytes` (kernel/include/
    // ignis_gqa_workspace.h) rather than the vendored query directly, for the
    // one correction that header documents -- a narrow hq prompt width is
    // under-reported, and an under-reserved arena is an exception thrown
    // mid-layer.
    const std::size_t attention_workspace_bytes =
        ignis_gqa_attention_workspace_bytes(batch_cache.dtype, envelope, tokens);
    const ninfer::DeviceSpan attention_workspace_storage =
        model->scratch->alloc_bytes(std::max<std::size_t>(attention_workspace_bytes, 1));
    error = cudaMemsetAsync(attention_workspace_storage.data, 0,
                            attention_workspace_storage.bytes, stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_step: cudaMemsetAsync(attention workspace) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    ninfer::DeviceArena attention_workspace(attention_workspace_storage);
    ninfer::Tensor attention_heads = attention.view({kHeadDim, kQHeads, tokens, 1});
    ninfer::ops::gqa_attention(
        rotated_query_heads, rotated_key_heads, value_heads, positions,
        /*valid_columns=*/ninfer::Tensor{}, kv_table_rows,
        gate.view({kHeadDim, kQHeads, tokens, 1}), attention_scale, batch_cache, envelope,
        attention_workspace, attention_heads, stream);

    ninfer::Tensor residual(out_residual, ninfer::DType::BF16, {hidden, tokens, 1, 1});
    error = cudaMemcpyAsync(out_residual, in_residual,
                            static_cast<std::size_t>(hidden) * tokens * sizeof(uint16_t),
                            cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_step: cudaMemcpyAsync(residual) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    // P2-03 (GitHub #85): the residual add takes the call's mode policy
    // (AllowA4 on the NVFP4 output projection under the engine default, so
    // its W4A4 route is the vendored dispatch's own decision) or A16Only
    // under the override.
    ninfer::ops::linear_add(attention, weights.output, residual,
                            ignis_policy_for(weights.output.qtype, mode), *model->scratch, stream);

    const ninfer::Tensor post_norm =
        weight_tensor(weights.post_attention_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(residual, post_norm, model->rms_norm_eps, /*unit_offset=*/true, post,
                         stream);
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
    ninfer::ops::linear_swiglu(
        post, weights.mlp_gate_up, fused,
        ignis_linear_swiglu_policy_for(weights.mlp_gate_up.qtype, mode, tokens), *model->scratch,
        stream);
    ninfer::ops::linear_add(fused, weights.mlp_down, residual,
                            ignis_policy_for(weights.mlp_down.qtype, mode), *model->scratch,
                            stream);

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
int32_t run_gqa_layer_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                            uint32_t width, void *in_residual, void *out_residual,
                            uint32_t gqa_layer, LinearPolicyMode mode) {
  constexpr std::int32_t kHeadDim = 256;
  constexpr std::int32_t kQHeads = kIgnisGqaQHeads;
  constexpr std::int32_t kKvHeads = 4;
  constexpr std::int32_t kRotaryDim = 64;
  constexpr float kRopeTheta = 10'000'000.0F;
  // A decode round is exactly one token per lane: the batch's rows are the
  // lanes (`batch` below), and every row's width is one.
  constexpr std::int32_t kLaneTokens = 1;

  const auto batch = static_cast<std::int32_t>(width);
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto q_width = kHeadDim * kQHeads;
  const auto kv_width = kHeadDim * kKvHeads;
  const auto ffn = model->layers[layer].gqa.mlp_gate_up.n / 2;
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const GqaLayerWeights &weights = model->layers[layer].gqa;
  const auto stream = model->stream;

  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    ninfer::Tensor normalized =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::Tensor query =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, batch, 1, 1});
    ninfer::Tensor key =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {kv_width, batch, 1, 1});
    ninfer::Tensor gate =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, batch, 1, 1});
    ninfer::Tensor value =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {kv_width, batch, 1, 1});
    ninfer::Tensor rotated_query =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, batch, 1, 1});
    ninfer::Tensor rotated_key =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {kv_width, batch, 1, 1});
    ninfer::Tensor attention =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {q_width, batch, 1, 1});
    ninfer::Tensor post = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::Tensor fused = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {ffn, batch, 1, 1});

    const ninfer::Tensor input(in_residual, ninfer::DType::BF16, {hidden, batch, 1, 1});
    const ninfer::Tensor input_norm =
        weight_tensor(weights.input_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(input, input_norm, model->rms_norm_eps, /*unit_offset=*/true,
                         normalized, stream);

    ninfer::ops::attn_input_proj(normalized, weights.query_key_gate_value, query, gate, key,
                                 value, ignis_policy_for(weights.query_key_gate_value.qtype, mode),
                                 *model->decode_graph_scratch, stream);

    // Graph-safe positions (ADR 0019): the round's staged per-lane absolute
    // positions, read from device memory at replay -- `fill_i32_positions`'s
    // host-scalar `start` would freeze this round's positions into the graph
    // forever. The lanes' positions are unrelated to one another, so this is
    // read as-is rather than derived from a base; it is the same contiguous
    // I32 [width] buffer both `qk_norm_rope` (positions [T], read per token)
    // and `gqa_attention` (positions [W,B] with W=1) want, and both only
    // read it.
    // Two shapes over the same buffer, again because the ops name the batch
    // differently: `qk_norm_rope` wants a flat I32 [T], A1 wants I32 [W,B]
    // with W=1.
    const ninfer::Tensor positions(model->sampling_decode_positions->p, ninfer::DType::I32,
                                   {batch, 1, 1, 1});
    const ninfer::Tensor positions_rows(model->sampling_decode_positions->p, ninfer::DType::I32,
                                        {kLaneTokens, batch, 1, 1});

    const ninfer::Tensor q_norm =
        weight_tensor(weights.query_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    const ninfer::Tensor k_norm =
        weight_tensor(weights.key_norm, ninfer::DType::BF16, {kHeadDim, 1, 1, 1});
    const ninfer::ops::RopeFrequencies rope =
        ninfer::ops::rope_linear_frequencies(kRopeTheta, kRotaryDim);
    // Two views of the same contiguous storage, because the two ops name
    // the batch differently: `qk_norm_rope` takes flat `[256,Hq|Hkv,T]` and
    // reads one position per column, so the round's lanes are simply its T;
    // A1 below takes `[256,Hq|Hkv,W,B]` and needs the lanes on the batch
    // axis with W=1. The element order is identical either way -- one token
    // per lane, lane-major -- so this is a reshape, not a copy.
    ninfer::Tensor query_tokens = query.view({kHeadDim, kQHeads, batch, 1});
    ninfer::Tensor key_tokens = key.view({kHeadDim, kKvHeads, batch, 1});
    ninfer::Tensor rotated_query_tokens = rotated_query.view({kHeadDim, kQHeads, batch, 1});
    ninfer::Tensor rotated_key_tokens = rotated_key.view({kHeadDim, kKvHeads, batch, 1});
    ninfer::ops::qk_norm_rope(query_tokens, key_tokens, q_norm, k_norm, model->rms_norm_eps,
                              positions, rope, rotated_query_tokens, rotated_key_tokens, stream);
    ninfer::Tensor rotated_query_heads = rotated_query.view({kHeadDim, kQHeads, kLaneTokens, batch});
    ninfer::Tensor rotated_key_heads = rotated_key.view({kHeadDim, kKvHeads, kLaneTokens, batch});

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
    ninfer::Tensor value_heads = value.view({kHeadDim, kKvHeads, kLaneTokens, batch});
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
    // format and the query has to be asked the same way (GitHub #123).
    const std::size_t attention_workspace_bytes = ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, batch_cache.dtype, envelope, batch, kLaneTokens, kLaneTokens);
    const ninfer::DeviceSpan attention_workspace_storage =
        model->decode_graph_scratch->alloc_bytes(std::max<std::size_t>(attention_workspace_bytes, 1));
    cudaError_t error = cudaMemsetAsync(attention_workspace_storage.data, 0,
                                        attention_workspace_storage.bytes, stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_graph: cudaMemsetAsync(attention workspace) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    ninfer::DeviceArena attention_workspace(attention_workspace_storage);
    ninfer::Tensor attention_heads = attention.view({kHeadDim, kQHeads, kLaneTokens, batch});
    ninfer::ops::gqa_attention(
        rotated_query_heads, rotated_key_heads, value_heads, positions_rows,
        /*valid_columns=*/ninfer::Tensor{}, kv_table_rows,
        gate.view({kHeadDim, kQHeads, kLaneTokens, batch}), attention_scale, batch_cache, envelope,
        attention_workspace, attention_heads, stream);

    ninfer::Tensor residual(out_residual, ninfer::DType::BF16, {hidden, batch, 1, 1});
    error = cudaMemcpyAsync(out_residual, in_residual,
                            static_cast<std::size_t>(hidden) * batch * sizeof(uint16_t),
                            cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_graph: cudaMemcpyAsync(residual) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    ninfer::ops::linear_add(attention, weights.output, residual,
                            ignis_policy_for(weights.output.qtype, mode), *model->decode_graph_scratch,
                            stream);

    const ninfer::Tensor post_norm =
        weight_tensor(weights.post_attention_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(residual, post_norm, model->rms_norm_eps, /*unit_offset=*/true, post,
                         stream);
    ninfer::ops::linear_swiglu(
        post, weights.mlp_gate_up, fused,
        ignis_linear_swiglu_policy_for(weights.mlp_gate_up.qtype, mode, batch),
        *model->decode_graph_scratch, stream);
    ninfer::ops::linear_add(fused, weights.mlp_down, residual,
                            ignis_policy_for(weights.mlp_down.qtype, mode), *model->decode_graph_scratch,
                            stream);
    return 0;
  } catch (const std::exception &error) {
    set_error(std::string("ignis_gqa_layer_graph: ") + error.what());
    return -1;
  }
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
                                 uint64_t num_tokens, LinearPolicyMode mode) {
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
      start_position + num_tokens > seq->kv.mapped_token_capacity()) {
    set_error("ignis_gqa_layer: token positions exceed the sequence KV capacity");
    return -1;
  }
  return run_gqa_layer(model, pool, seq, layer, const_cast<void *>(in_residual), out_residual,
                       gqa_layer, num_tokens, mode);
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
  if (width == 0 || width > IGNIS_DECODE_MAX_BATCH) {
    set_error("ignis_gqa_layer_graph: width " + std::to_string(width) +
              " is not in 1..IGNIS_DECODE_MAX_BATCH");
    return -1;
  }
  if (!gqa_format_matches_load(model, pool, "ignis_gqa_layer_graph")) {
    return -1;
  }
  const uint32_t gqa_layer = ignis_gqa_relative_layer(layer);
  return run_gqa_layer_graph(model, pool, layer, width, const_cast<void *>(in_residual), out_residual,
                             gqa_layer, mode);
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
