// ignis kernel leaf - P1-22 (GitHub #58): one GDN layer in the program
// (ADR 0009, ADR 0010). The layer's full attention + MLP tail, composed from
// the vendored ops (ninfer::ops) on top of the sequence's GDN state (the conv
// taps and the fp32 recurrent slot, GitHub #55) and the layer's per-layer
// weights (GitHub #53). The program layer is ours (the reference's program is
// not vendored); this dispatches to the reference's ops in the reference's
// text-context order:
//
//   input RMSNorm -> fused GDN input projection + causal conv (rolling taps
//   on the sequence's conv state) -> GDN gating projection + gating ->
//   per-head fp32 recurrence on the sequence's GDN slot -> gated RMSNorm with
//   z -> output projection + residual -> MLP tail.
//
// Verified against the P1-20 f64 layer reference (crates/artifact
// f64_reference.rs, evaluate_layer on the GDN layer). The layer's GDN state
// (conv taps + recurrent slot) is drawn from the sequence's slot and carries
// across the `num_tokens` tokens; releasing and re-allocating the sequence
// resets it (a fresh slot reads zero, ignis_seq.h).

#include "ignis_gdn_layer.h"

#include "ignis_seq_internal.h"
#include "layer_internal.h"
#include "model_internal.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"

#include "ignis_step.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace {

// The last error message on this thread (ignis_gdn_layer_last_error) -- a
// separate channel from the model / step / seq ABIs (each ABI surface owns
// its own, model.cu / step.cu / seq.cu's convention).
thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

// A non-owning BF16/FP32 view of a loaded dense GDN weight's device payload
// (the small per-layer GDN weights cross the ops as plain Tensors: the input
// and post attention norms, the gdn gated-norm weight, the conv weight, and
// the fp32 A_log / dt_bias gating constants).
ninfer::Tensor weight_tensor(const ninfer::Weight &w, ninfer::DType dtype,
                             std::initializer_list<std::int32_t> shape) {
  return ninfer::Tensor(const_cast<void *>(w.qdata), dtype, shape);
}

// Runs one GDN layer for `num_tokens` sequential tokens of one sequence
// (GitHub #58). `slot` is the sequence's state-pool slot (conv taps + fp32
// recurrent state); `layer` is the *model's* zero-based layer index (0..63,
// used for `model->layers[layer]`'s weights); `gdn_layer` is this GDN
// layer's position among GDN layers only (0..47, used for the state pool,
// which is sized/addressed by GDN layer count, GitHub #55) -- the same
// absolute-vs-relative split `ignis_gqa_layer_step`'s `gqa_layer` already
// makes for its own (16-count) pool addressing. `in_residual` /
// `out_residual` are device BF16 `[hidden, num_tokens]` feature-major
// buffers (out_residual is written in place and receives the final
// residual). Returns 0 on success, -1 on error (message set via set_error).
int32_t run_gdn_layer(ignis_model *model, ignis_seq_pool *pool, int32_t slot, uint32_t layer,
                      uint32_t gdn_layer, void *in_residual, void *out_residual,
                      uint64_t num_tokens, LinearPolicyMode mode) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto T = static_cast<std::int32_t>(num_tokens);
  const auto stream = model->stream;
  const GdnLayerWeights &w = model->layers[layer].gdn;

  // The GDN geometry, derived from the sequence-state pool's spec (the pool is
  // sized by the same per-model GDN geometry the model handle's layers were
  // bound with): the value-head count and the (square) head dim, and the
  // conv-channel count (query + key + value widths).
  const auto &spec = pool->gdn_pool.spec;
  const auto value_heads = static_cast<std::int32_t>(spec.value_heads);
  const auto head_dim = static_cast<std::int32_t>(spec.key_head_dim);
  const auto conv_channels = static_cast<std::int32_t>(spec.conv_channels);
  const auto value_width = value_heads * head_dim;      // 48 * 128 = 6144
  const auto qk_width = (conv_channels - value_width) / 2;  // (10240 - 6144) / 2 = 2048
  const auto qk_heads = qk_width / head_dim;            // 2048 / 128 = 16
  // The MLP intermediate (the SwiGLU gate_up row count / 2), from the loaded
  // weight's row count.
  const auto ffn = w.mlp_gate_up.n / 2;                 // 34816 / 2 = 17408
  // The GDN readout scale (1 / sqrt of the (square) state dim), matching the
  // f64 reference's 1/sqrt(GDN_HEAD_DIM).
  const float readout_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    // --- per-layer intermediates (the layer's scratch for this step) ---
    ninfer::Tensor h = model->scratch->alloc(ninfer::DType::BF16, {hidden, T, 1, 1});
    ninfer::Tensor qkv = model->scratch->alloc(ninfer::DType::BF16, {conv_channels, T, 1, 1});
    ninfer::Tensor qkv_conv = model->scratch->alloc(ninfer::DType::BF16, {conv_channels, T, 1, 1});
    ninfer::Tensor query = model->scratch->alloc(ninfer::DType::BF16, {qk_width, T, 1, 1});
    ninfer::Tensor key = model->scratch->alloc(ninfer::DType::BF16, {qk_width, T, 1, 1});
    ninfer::Tensor value = model->scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor zbuf = model->scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor g = model->scratch->alloc(ninfer::DType::FP32, {value_heads, T, 1, 1});
    ninfer::Tensor beta = model->scratch->alloc(ninfer::DType::FP32, {value_heads, T, 1, 1});
    ninfer::Tensor recurrent = model->scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor gated = model->scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor post = model->scratch->alloc(ninfer::DType::BF16, {hidden, T, 1, 1});
    ninfer::Tensor fused = model->scratch->alloc(ninfer::DType::BF16, {ffn, T, 1, 1});

    // The sequence's persistent GDN state views (conv taps + fp32 recurrent
    // slot), addressed directly by `slot` -- not the snapshot-indexed forms
    // (those spend one pool slot per processed token, sized for ReplaySSM
    // prefix-window capture, not for one sequence's in-place carry; the
    // pool here is sized by concurrency, GitHub #55). These views alias the
    // pool's persistent storage: the ops below update them in place, and a
    // released/re-allocated sequence's fresh slot reads zero (ignis_seq.h).
    ninfer::Tensor conv_state = pool->gdn_pool.conv_slot(gdn_layer, slot);
    ninfer::Tensor ssm_state = pool->gdn_pool.recurrent_slot(gdn_layer, slot);

    // The input residual viewed as the layer's [hidden, T] feature-major input.
    const ninfer::Tensor in(in_residual, ninfer::DType::BF16, {hidden, T, 1, 1});

    // --- input RMSNorm (the layer's pre-attention norm) ---
    const ninfer::Tensor input_norm =
        weight_tensor(w.input_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(in, input_norm, model->rms_norm_eps, /*unit_offset=*/true, h, stream);

    // --- GDN input projection (NVFP4, kernel/src/gdn_input_proj.cu) -> the
    // combined pre-conv q/k/v plane (query/key/value channel order) and z
    // (bypasses the conv) -- then the causal conv + SiLU over the combined
    // plane, in place on the sequence's rolling conv taps. The fused
    // projection+conv+split op (gdn_input_proj_conv_snapshot) has no leaf
    // implementation yet (kernel/src/gdn_input_proj.cu only wires the plain
    // projection); composing the two vendored ops here is equivalent for
    // T within the conv's dense (unmasked) domain, which every caller of
    // this step ABI uses (no mid-batch padding).
    const ninfer::Tensor conv_weight = weight_tensor(
        w.convolution, ninfer::DType::BF16, {conv_channels, kIgnisGdnConvKernel, 1, 1});
    // P2-03 (GitHub #85): the GDN input projection under the call's mode
    // policy (ADR 0016): AllowA4 on the NVFP4 query_key_value_z parent
    // (the W4A4 route is then the vendored dispatch's own decision per its
    // token thresholds) or A16Only under the override. The transient
    // workspace comes from the model's scratch (reserved at load for the
    // widest policy, P2-01, GitHub #83), so a chunk's A4 route needs
    // nothing new.
    ninfer::ops::gdn_input_proj(h, w.query_key_value_z, qkv, zbuf,
                                ignis_policy_for(w.query_key_value_z.qtype, mode), *model->scratch,
                                stream);
    ninfer::ops::causal_conv1d_silu(qkv, conv_weight, conv_state, qkv_conv, stream);

    // The convolved query/key/value channel ranges are strided sub-views of
    // `qkv_conv` (a [conv_channels, T] feature-major buffer: for a fixed
    // token column, channels are contiguous), so a per-range device-to-device
    // strided copy (not a Tensor::slice view) lands each range in its own
    // contiguous buffer -- the per-head views below require contiguous
    // storage.
    constexpr std::size_t kElemBytes = 2;  // BF16
    auto copy_channel_range = [&](std::int32_t offset, std::int32_t width,
                                  ninfer::Tensor &dst) -> cudaError_t {
      const auto *src = static_cast<const std::uint8_t *>(qkv_conv.data) +
                        static_cast<std::size_t>(offset) * kElemBytes;
      return cudaMemcpy2DAsync(dst.data, static_cast<std::size_t>(width) * kElemBytes, src,
                               static_cast<std::size_t>(conv_channels) * kElemBytes,
                               static_cast<std::size_t>(width) * kElemBytes,
                               static_cast<std::size_t>(T), cudaMemcpyDeviceToDevice, stream);
    };
    cudaError_t err = copy_channel_range(0, qk_width, query);
    if (err == cudaSuccess) { err = copy_channel_range(qk_width, qk_width, key); }
    if (err == cudaSuccess) { err = copy_channel_range(2 * qk_width, value_width, value); }
    if (err != cudaSuccess) {
      set_error(std::string("ignis_gdn_layer_step: cudaMemcpy2DAsync(qkv split) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    // --- GDN gating projection + gating -> g (decay) and beta (update gate) ---
    const ninfer::Tensor a_log =
        weight_tensor(w.a_log, ninfer::DType::FP32, {value_heads, 1, 1, 1});
    const ninfer::Tensor dt_bias =
        weight_tensor(w.dt_bias, ninfer::DType::FP32, {value_heads, 1, 1, 1});
    ninfer::ops::gdn_gating_proj(h, w.a_b_projection, a_log, dt_bias, *model->scratch, g, beta,
                                 stream);

    // --- P2-04 (GitHub #86): the distinct-state recurrence entry point.
    // The sequence's own slot is supplied as BOTH the input and the output
    // state: the op runs the vendored chunked tensor-core kernels
    // (prepare_wy_wu, state_passing, output) over whole 64-token chunks and
    // the recurrent kernel over the tail, then publishes the post-T state
    // into exactly the pool slot it read (the spec's "the state after a
    // chunk is unchanged" contract). With the two state arguments aliased to
    // one storage the launches are the same kernel instantiations the
    // in-place overload resolves to (its T>1 form delegates to this exact
    // call; its T=1 form is this call's tail with both state pointers
    // aliased), so decode's T=1 path is numerically unchanged. The chunked
    // workspace comes from the load-time reservation (P2-01, GitHub #83),
    // sized by the vendored capacity query at the configured chunk width --
    // no allocation on the prefill path.
    // The recurrence's output (the per-token readout) is the
    // `[128, value_heads, T]` view of the `recurrent` buffer (the op writes
    // it in place, so it must be an lvalue, not a temporary view).
    ninfer::Tensor recurrent_out = recurrent.view({head_dim, value_heads, T, 1});
    ninfer::ops::gated_delta_net(
        query.view({head_dim, qk_heads, T, 1}), key.view({head_dim, qk_heads, T, 1}),
        value.view({head_dim, value_heads, T, 1}), g, beta, readout_scale, /*normalize_qk=*/true,
        *model->scratch, ssm_state, ssm_state, recurrent_out, stream);

    // --- gated RMSNorm with z (per-head norm * SiLU(z)) ---
    const ninfer::Tensor gdn_norm =
        weight_tensor(w.norm, ninfer::DType::BF16, {head_dim, 1, 1, 1});
    ninfer::Tensor gated_out = gated.view({head_dim, value_heads, T, 1});
    ninfer::ops::gated_rmsnorm(recurrent_out, gdn_norm, zbuf.view({head_dim, value_heads, T, 1}),
                               model->rms_norm_eps, gated_out, stream);

    // --- output projection + residual (the layer's attention residual add; the
    // layer-4 BF16 output arm, exercised by the test). `residual_view` is not
    // const: the linear_adds update the residual in place.
    ninfer::Tensor residual_view(out_residual, ninfer::DType::BF16, {hidden, T, 1, 1});
    err = cudaMemcpyAsync(out_residual, in_residual, static_cast<std::size_t>(hidden) * T * 2,
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_gdn_layer_step: cudaMemcpyAsync(residual) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    // P2-03 (GitHub #85): the output projection's residual add under the
    // call's mode policy (AllowA4 on the NVFP4 output parent, or A16Only
    // under the override).
    ninfer::ops::linear_add(gated, w.output, residual_view,
                            ignis_policy_for(w.output.qtype, mode), *model->scratch, stream);

    // --- MLP tail: post-attention norm -> SwiGLU (gate_up + SiLU-mul) -> down +
    // residual ---
    const ninfer::Tensor post_norm =
        weight_tensor(w.post_attention_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(residual_view, post_norm, model->rms_norm_eps, /*unit_offset=*/true,
                         post, stream);
    // P2-03 (GitHub #85): the MLP tail under the call's mode policy too
    // (AllowA4 on the NVFP4 gate_up/down parents under the engine default,
    // or A16 under the override). The width-aware `linear_swiglu`
    // counterpart of the policy helper (kernel/src/layer_internal.h):
    // under the override the A16 route is forced where it is registered
    // (T<=16) and the op's only runnable route above its A16 cap (the pre-
    // #85 behavior); under the engine default AllowA4 at every width, the
    // vendored dispatch's own decision per its token thresholds, GitHub
    // #85's acceptance.
    ninfer::ops::linear_swiglu(post, w.mlp_gate_up, fused,
                               ignis_linear_swiglu_policy_for(w.mlp_gate_up.qtype, mode, T),
                               *model->scratch, stream);
    ninfer::ops::linear_add(fused, w.mlp_down, residual_view,
                            ignis_policy_for(w.mlp_down.qtype, mode), *model->scratch, stream);

    // No stream synchronization here (P2-01, GitHub #83): the layer body
    // only enqueues work, so a later chunk loop can run every layer as one
    // pipelined unit. `ignis_gdn_layer_step` below synchronizes once the
    // layer body returns, keeping this function's own callers' contract
    // (its GPU tests) unchanged.
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_gdn_layer_step: ") + e.what());
    return -1;
  }
}

// P3-06 (GitHub #111): the graph-safe counterpart of `run_gdn_layer` for a
// whole decode round -- `width` lanes at one token each, traversed once as a
// [.., width] batch rather than `width` times at batch 1.
// `causal_conv1d_silu_snapshot` / `gated_delta_net_snapshot` replace the
// direct `conv_slot`/`recurrent_slot` calls: both read the pool slot to
// update for row b from `model->decode_graph_slots[b]` (device memory, this
// round's real value at replay time) instead of a host `slot` int baked at
// capture time, and both write back to that same slot in place
// (`snapshot_base_slots == initial_state_slots`), which is what keeps the
// lanes' conv taps and recurrent states isolated inside one call.
// `ssm_states` / `conv_states` span the whole pool (every slot, fixed
// address) rather than one sequence's own view -- see layer_internal.h's
// declaration and ADR 0019.
int32_t run_gdn_layer_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                            uint32_t width, void *in_residual, void *out_residual,
                            uint32_t gdn_layer, LinearPolicyMode mode) {
  // A decode round is exactly one token per lane: the batch's rows are the
  // lanes, so the activation buffers below carry `T` == `width` columns and
  // every snapshot op sees W=1 per row.
  const auto T = static_cast<std::int32_t>(width);
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto stream = model->stream;
  const GdnLayerWeights &w = model->layers[layer].gdn;

  const auto &spec = pool->gdn_pool.spec;
  const auto value_heads = static_cast<std::int32_t>(spec.value_heads);
  const auto head_dim = static_cast<std::int32_t>(spec.key_head_dim);
  const auto conv_channels = static_cast<std::int32_t>(spec.conv_channels);
  const auto value_width = value_heads * head_dim;
  const auto qk_width = (conv_channels - value_width) / 2;
  const auto qk_heads = qk_width / head_dim;
  const auto ffn = w.mlp_gate_up.n / 2;
  const auto slot_count = pool->gdn_pool.slot_count();
  const float readout_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    ninfer::Tensor h = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, T, 1, 1});
    ninfer::Tensor qkv = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {conv_channels, T, 1, 1});
    ninfer::Tensor qkv_conv = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {conv_channels, T, 1, 1});
    ninfer::Tensor query = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {qk_width, T, 1, 1});
    ninfer::Tensor key = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {qk_width, T, 1, 1});
    ninfer::Tensor value = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor zbuf = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor g = model->decode_graph_scratch->alloc(ninfer::DType::FP32, {value_heads, T, 1, 1});
    ninfer::Tensor beta = model->decode_graph_scratch->alloc(ninfer::DType::FP32, {value_heads, T, 1, 1});
    ninfer::Tensor recurrent = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor gated = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {value_width, T, 1, 1});
    ninfer::Tensor post = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, T, 1, 1});
    ninfer::Tensor fused = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {ffn, T, 1, 1});

    // The whole-pool state planes (every slot, fixed address) -- slot 0's
    // view supplies the base pointer; the pool's own per-slot stride
    // (identical to what `conv_slot`/`recurrent_slot` step by) gives the
    // rest of the shape.
    const ninfer::Tensor conv_states(pool->gdn_pool.conv_slot(gdn_layer, 0).data,
                                     ninfer::DType::BF16,
                                     {conv_channels, kIgnisGdnConvStateWidth, slot_count, 1});
    ninfer::Tensor ssm_states(pool->gdn_pool.recurrent_slot(gdn_layer, 0).data, ninfer::DType::FP32,
                              {head_dim, head_dim, value_heads, slot_count});
    // This round's physical pool slot per lane, contiguous I32 [width]: both
    // snapshot ops' `initial_state_slots` and `snapshot_base_slots`. The
    // slots are distinct physical slots, so the ops' requirement that the
    // rows' [base, base+W) reservations be disjoint holds by construction.
    const ninfer::Tensor lane_slots(model->decode_graph_slots->p, ninfer::DType::I32,
                                    {T, 1, 1, 1});

    const ninfer::Tensor in(in_residual, ninfer::DType::BF16, {hidden, T, 1, 1});

    const ninfer::Tensor input_norm =
        weight_tensor(w.input_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(in, input_norm, model->rms_norm_eps, /*unit_offset=*/true, h, stream);

    const ninfer::Tensor conv_weight = weight_tensor(
        w.convolution, ninfer::DType::BF16, {conv_channels, kIgnisGdnConvKernel, 1, 1});
    ninfer::ops::gdn_input_proj(h, w.query_key_value_z, qkv, zbuf,
                                ignis_policy_for(w.query_key_value_z.qtype, mode),
                                *model->decode_graph_scratch, stream);
    // In-place snapshot form: each row's real physical slot (read from
    // `lane_slots` at replay time) is both the window that row reads and the
    // window it writes back. `x`/`out` are [C,W,B] with W=1, which is the
    // [conv_channels, width] buffer above viewed one column per lane.
    ninfer::Tensor conv_states_mut = conv_states;
    ninfer::Tensor qkv_rows = qkv.view({conv_channels, 1, T, 1});
    ninfer::Tensor qkv_conv_rows = qkv_conv.view({conv_channels, 1, T, 1});
    ninfer::ops::causal_conv1d_silu_snapshot(qkv_rows, conv_weight, conv_states_mut,
                                             /*valid_columns=*/ninfer::Tensor{}, lane_slots,
                                             lane_slots, qkv_conv_rows, stream);

    constexpr std::size_t kElemBytes = 2;
    auto copy_channel_range = [&](std::int32_t offset, std::int32_t width,
                                  ninfer::Tensor &dst) -> cudaError_t {
      const auto *src = static_cast<const std::uint8_t *>(qkv_conv.data) +
                        static_cast<std::size_t>(offset) * kElemBytes;
      return cudaMemcpy2DAsync(dst.data, static_cast<std::size_t>(width) * kElemBytes, src,
                               static_cast<std::size_t>(conv_channels) * kElemBytes,
                               static_cast<std::size_t>(width) * kElemBytes,
                               static_cast<std::size_t>(T), cudaMemcpyDeviceToDevice, stream);
    };
    cudaError_t err = copy_channel_range(0, qk_width, query);
    if (err == cudaSuccess) { err = copy_channel_range(qk_width, qk_width, key); }
    if (err == cudaSuccess) { err = copy_channel_range(2 * qk_width, value_width, value); }
    if (err != cudaSuccess) {
      set_error(std::string("ignis_gdn_layer_graph: cudaMemcpy2DAsync(qkv split) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    const ninfer::Tensor a_log =
        weight_tensor(w.a_log, ninfer::DType::FP32, {value_heads, 1, 1, 1});
    const ninfer::Tensor dt_bias =
        weight_tensor(w.dt_bias, ninfer::DType::FP32, {value_heads, 1, 1, 1});
    ninfer::ops::gdn_gating_proj(h, w.a_b_projection, a_log, dt_bias, *model->decode_graph_scratch,
                                 g, beta, stream);

    // The snapshot form's [.., W, B] layout with W=1: one token per lane,
    // `width` lanes -- the same contiguous buffers the projections above
    // wrote, re-viewed with the batch on the last axis.
    ninfer::Tensor recurrent_out = recurrent.view({head_dim, value_heads, 1, T});
    ninfer::ops::gated_delta_net_snapshot(
        query.view({head_dim, qk_heads, 1, T}), key.view({head_dim, qk_heads, 1, T}),
        value.view({head_dim, value_heads, 1, T}), g.view({value_heads, 1, T, 1}),
        beta.view({value_heads, 1, T, 1}), readout_scale, /*normalize_qk=*/true, ssm_states,
        /*valid_columns=*/ninfer::Tensor{}, lane_slots, lane_slots, recurrent_out, stream);

    const ninfer::Tensor gdn_norm =
        weight_tensor(w.norm, ninfer::DType::BF16, {head_dim, 1, 1, 1});
    ninfer::Tensor gated_out = gated.view({head_dim, value_heads, 1, T});
    ninfer::ops::gated_rmsnorm(recurrent_out, gdn_norm, zbuf.view({head_dim, value_heads, 1, T}),
                               model->rms_norm_eps, gated_out, stream);

    ninfer::Tensor residual_view(out_residual, ninfer::DType::BF16, {hidden, T, 1, 1});
    err = cudaMemcpyAsync(out_residual, in_residual, static_cast<std::size_t>(hidden) * T * 2,
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_gdn_layer_graph: cudaMemcpyAsync(residual) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    ninfer::ops::linear_add(gated, w.output, residual_view,
                            ignis_policy_for(w.output.qtype, mode), *model->decode_graph_scratch,
                            stream);

    const ninfer::Tensor post_norm =
        weight_tensor(w.post_attention_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(residual_view, post_norm, model->rms_norm_eps, /*unit_offset=*/true,
                         post, stream);
    ninfer::ops::linear_swiglu(post, w.mlp_gate_up, fused,
                               ignis_linear_swiglu_policy_for(w.mlp_gate_up.qtype, mode, T),
                               *model->decode_graph_scratch, stream);
    ninfer::ops::linear_add(fused, w.mlp_down, residual_view,
                            ignis_policy_for(w.mlp_down.qtype, mode), *model->decode_graph_scratch,
                            stream);
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_gdn_layer_graph: ") + e.what());
    return -1;
  }
}

} // namespace

// P2-02 (GitHub #84): the validated body a chunk loop dispatches directly
// (kernel/src/layer_internal.h) -- every check `ignis_gdn_layer_step` did,
// minus the synchronization a per-chunk caller defers until its whole
// chunk's dispatches succeed. `mode` is the call's compute-policy mode
// (P2-03, GitHub #85): every NVFP4 projection in the body is dispatched
// under the policy `ignis_policy_for` resolves for it.
int32_t ignis_gdn_layer_run_body(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                 uint32_t layer, const void *in_residual, void *out_residual,
                                 uint64_t num_tokens, LinearPolicyMode mode) {
  // Errors here are prefixed `ignis_gdn_layer` (not `..._step`): this body
  // is now dispatched both by `ignis_gdn_layer_step` and directly by a
  // chunk loop (kernel/src/step.cu), so a message naming the ABI wrapper
  // would misattribute a chunked-prefill failure.
  if (model == nullptr || pool == nullptr || seq == nullptr || in_residual == nullptr ||
      out_residual == nullptr) {
    set_error("ignis_gdn_layer: null argument");
    return -1;
  }
  if (num_tokens == 0) {
    set_error("ignis_gdn_layer: num_tokens must be positive");
    return -1;
  }
  if (layer >= model->layers.size() || model->layers[layer].kind != IGNIS_LAYER_GDN) {
    set_error("ignis_gdn_layer: layer " + std::to_string(layer) + " is not a GDN layer");
    return -1;
  }
  // This GDN layer's position among GDN layers only (0..47): the Qwen 3.8
  // topology's GQA layers sit at index 3, 7, 11, ... (every 4th, validated
  // by `ignis_gqa_layer_step`'s own check), so the count of GQA layers at
  // or before `layer` is `(layer + 1) / 4` -- subtracting it out of the
  // absolute index gives the GDN-relative one the state pool is sized and
  // addressed by (GitHub #55), mirroring `ignis_gqa_layer_step`'s
  // `gqa_layer = (layer - 3) / 4` for its own (16-count) pool.
  const uint32_t gdn_layer = layer - (layer + 1) / 4;
  return run_gdn_layer(model, pool, seq->slot, layer, gdn_layer, const_cast<void *>(in_residual),
                       out_residual, num_tokens, mode);
}

// P2-03 (GitHub #85): `ignis_gdn_layer_step`'s synchronous contract (body +
// one stream synchronization; no position advance -- the GDN layer's state
// is updated in place by the enqueued work) with a non-default
// compute-policy mode -- the program's per-token route (kernel/src/step.cu)
// threads ADR 0016's `compute_policy` override through it. The flat C ABI
// entry point below is this same contract with `kEngineDefault`.
int32_t ignis_gdn_layer_step_mode(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens, LinearPolicyMode mode) {
  const int32_t rc =
      ignis_gdn_layer_run_body(model, pool, seq, layer, in_residual, out_residual, num_tokens,
                               mode);
  if (rc != 0) {
    return rc;
  }
  // P2-01 (GitHub #83): the sync moved here from the layer body so a direct
  // call through this ABI entry point (this function's own GPU tests) keeps
  // seeing a synchronous result, while a caller that dispatches the body
  // directly (the program's per-chunk loop) can pipeline every layer.
  const cudaError_t error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess) {
    set_error(std::string("ignis_gdn_layer_step: cudaStreamSynchronize failed: ") +
              cudaGetErrorString(error));
    return -1;
  }
  return 0;
}

// P3-06 (GitHub #111): validates and dispatches `run_gdn_layer_graph` --
// called once per GDN layer per decode round (not per lane) by
// `kernel/src/decode_graph.cu`, either while a graph is being captured or
// when the round runs eagerly at a width whose capture failed.
int32_t ignis_gdn_layer_run_body_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                                       uint32_t width, const void *in_residual, void *out_residual,
                                       LinearPolicyMode mode) {
  if (model == nullptr || pool == nullptr || in_residual == nullptr || out_residual == nullptr) {
    set_error("ignis_gdn_layer_graph: null argument");
    return -1;
  }
  if (layer >= model->layers.size() || model->layers[layer].kind != IGNIS_LAYER_GDN) {
    set_error("ignis_gdn_layer_graph: layer " + std::to_string(layer) + " is not a GDN layer");
    return -1;
  }
  if (width == 0 || width > IGNIS_DECODE_MAX_BATCH) {
    set_error("ignis_gdn_layer_graph: width " + std::to_string(width) +
              " is not in 1..IGNIS_DECODE_MAX_BATCH");
    return -1;
  }
  const uint32_t gdn_layer = layer - (layer + 1) / 4;
  return run_gdn_layer_graph(model, pool, layer, width, const_cast<void *>(in_residual), out_residual,
                             gdn_layer, mode);
}

extern "C" int32_t ignis_gdn_layer_step(struct ignis_model *model, struct ignis_seq_pool *pool,
                                         struct ignis_seq *seq, uint32_t layer,
                                         const void *in_residual, void *out_residual,
                                         uint64_t num_tokens) {
  return ignis_gdn_layer_step_mode(model, pool, seq, layer, in_residual, out_residual, num_tokens,
                                   LinearPolicyMode::kEngineDefault);
}

extern "C" const char *ignis_gdn_layer_last_error(void) {
  return g_last_error.c_str();
}
