// ignis kernel leaf - P1-21 (GitHub #57): one GQA layer in the device-resident
// program. The program layer is ours (ADR 0009); every dispatched op is the
// ADR 0010 vendored reference implementation. The order is the Qwen text
// context: input norm -> fused Q/K/gate/V projection -> q/k norm + RoPE ->
// KV append -> attention + sigmoid output gate -> output residual -> MLP tail.

#include "ignis_gqa_layer.h"

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

// The P1-19 pool stores one BF16 K,V pair for every GQA layer and allocates
// one block-table row per sequence slot. This view is non-owning: the
// sequence's allocation keeps the mapping and pages alive for the full call.
ninfer::PagedKVLayerView cache_view(ignis_seq_pool *pool, ignis_seq *seq, uint32_t gqa_layer) {
  ninfer::PagedKVLayerView view;
  view.k_pages = pool->kv_pool.plane(2 * gqa_layer);
  view.v_pages = pool->kv_pool.plane(2 * gqa_layer + 1);
  view.block_table = seq->kv.block_table();
  view.head_dim = 256;
  view.num_kv_heads = 4;
  view.dtype = ninfer::DType::BF16;
  return view;
}

int32_t run_gqa_layer(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                      uint32_t layer, void *in_residual, void *out_residual,
                      uint32_t gqa_layer, uint64_t num_tokens, LinearPolicyMode mode) {
  constexpr std::int32_t kHeadDim = 256;
  constexpr std::int32_t kQHeads = 24;
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

    const ninfer::PagedKVLayerView cache = cache_view(pool, seq, gqa_layer);
    ninfer::Tensor value_heads = value.view({kHeadDim, kKvHeads, tokens, 1});
    ninfer::ops::gqa_kv_append(rotated_key_heads, value_heads, positions, cache, stream);
    const ninfer::ops::GqaExecutionEnvelope envelope{
        .min_visible_keys = 1,
        .max_visible_keys = static_cast<std::uint32_t>(start_position + tokens),
    };
    // The BF16 small-T attention route reserves a fixed number of partial
    // split slots, while only the active prefix is written for a short
    // history. Give it a fresh, zeroed workspace per layer invocation: stale
    // inactive partials would otherwise be reduced on the next T=1 call. At
    // a chunk-sized T the resolver's B=1 prompt route legally needs none
    // (kernel/vendor/include/ninfer/ops/gqa_attention.h: "a legal B=1
    // prompt route may return zero for BF16/INT8 caches") -- but neither
    // `DeviceArena` constructor (kernel/vendor/src/core/arena.h/.cu) admits
    // zero capacity, and `WorkspaceArena` is `DeviceArena` verbatim, so
    // there is no zero-sized arena to hand the op. P2-02 (GitHub #84):
    // reserve at least one byte regardless -- the op only ever touches what
    // its own query reported (zero, here), so the extra byte is never read
    // or written.
    const std::size_t attention_workspace_bytes = ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, ninfer::DType::BF16, envelope, /*batch_size=*/1, tokens, tokens);
    const ninfer::DeviceSpan attention_workspace_storage =
        model->scratch->alloc_bytes(std::max<std::size_t>(attention_workspace_bytes, 1));
    cudaError_t error = cudaMemsetAsync(attention_workspace_storage.data, 0,
                                        attention_workspace_storage.bytes, stream);
    if (error != cudaSuccess) {
      set_error(std::string("ignis_gqa_layer_step: cudaMemsetAsync(attention workspace) failed: ") +
                cudaGetErrorString(error));
      return -1;
    }
    ninfer::DeviceArena attention_workspace(attention_workspace_storage);
    ninfer::Tensor attention_heads = attention.view({kHeadDim, kQHeads, tokens, 1});
    ninfer::ops::gqa_attention_cached(
        rotated_query_heads, positions, gate.view({kHeadDim, kQHeads, tokens, 1}), attention_scale, cache, envelope,
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
