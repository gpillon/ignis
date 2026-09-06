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
#include "model_internal.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"

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
                      uint64_t num_tokens) {
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
    ninfer::ops::gdn_input_proj(h, w.query_key_value_z, qkv, zbuf, stream);
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

    // --- per-head fp32 recurrence on the sequence's GDN slot (q/k L2-normalized,
    // 1/sqrt(head_dim) readout; state published to the slot after all T tokens) ---
    // The recurrence's output (the per-token readout) is the `[128, value_heads, T]`
    // view of the `recurrent` buffer (the op writes it in place, so it must be an
    // lvalue, not a temporary view).
    ninfer::Tensor recurrent_out = recurrent.view({head_dim, value_heads, T, 1});
    ninfer::ops::gated_delta_net(
        query.view({head_dim, qk_heads, T, 1}), key.view({head_dim, qk_heads, T, 1}),
        value.view({head_dim, value_heads, T, 1}), g, beta, readout_scale, /*normalize_qk=*/true,
        *model->scratch, ssm_state, recurrent_out, stream);

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
    ninfer::ops::linear_add(gated, w.output, residual_view, *model->scratch, stream);

    // --- MLP tail: post-attention norm -> SwiGLU (gate_up + SiLU-mul) -> down +
    // residual ---
    const ninfer::Tensor post_norm =
        weight_tensor(w.post_attention_norm, ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(residual_view, post_norm, model->rms_norm_eps, /*unit_offset=*/true,
                         post, stream);
    ninfer::ops::linear_swiglu(post, w.mlp_gate_up, fused, *model->scratch, stream);
    ninfer::ops::linear_add(fused, w.mlp_down, residual_view, *model->scratch, stream);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_gdn_layer_step: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_gdn_layer_step: ") + e.what());
    return -1;
  }
}

} // namespace

extern "C" int32_t ignis_gdn_layer_step(struct ignis_model *model, struct ignis_seq_pool *pool,
                                         struct ignis_seq *seq, uint32_t layer,
                                         const void *in_residual, void *out_residual,
                                         uint64_t num_tokens) {
  if (model == nullptr || pool == nullptr || seq == nullptr || in_residual == nullptr ||
      out_residual == nullptr) {
    set_error("ignis_gdn_layer_step: null argument");
    return -1;
  }
  if (num_tokens == 0) {
    set_error("ignis_gdn_layer_step: num_tokens must be positive");
    return -1;
  }
  if (layer >= model->layers.size() || model->layers[layer].kind != IGNIS_LAYER_GDN) {
    set_error("ignis_gdn_layer_step: layer " + std::to_string(layer) + " is not a GDN layer");
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
  return run_gdn_layer(model, pool, seq->slot, layer, gdn_layer,
                       const_cast<void *>(in_residual), out_residual, num_tokens);
}

extern "C" const char *ignis_gdn_layer_last_error(void) {
  return g_last_error.c_str();
}