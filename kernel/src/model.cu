// ignis kernel leaf - P1-17 (GitHub #53): model-load flat C ABI (ADR 0009).
//
// Builds the leaf's per-layer weight structures from the bound-tensor +
// topology descriptors Rust hands across the boundary. No device work here
// (no kernel launch, no cudaMalloc): the artifact crate already placed every
// tensor in its device arena (crates/artifact) -- this file is pure
// host-side bookkeeping that matches each bound tensor's name against the
// topology-derived per-layer schema and rejects (loudly, all-or-nothing) a
// missing, extra, or mis-shaped one.
//
// The `*_input_scale_divisor` objects never get their own bound-tensor
// descriptor (Rust binds and validates them against the artifact, ADR
// 0002, but the leaf's per-layer schema below only lists the weight
// tensors), but each one's value crosses on its paired NVFP4 weight's
// `ignis_bound_tensor::input_scale_divisor` field (GitHub #58): the
// reference's NVFP4 `Weight` validation requires a finite, positive
// divisor regardless of compute policy, even though the W4A4 path that
// multiplies by it is still G2.
//
// Style follows the ticket-04 leaf (device.cu): explicit pointers + sizes,
// int32 return codes (0 = ok, -1 = error), no C++ types across the boundary.

#include "ignis_model.h"

#include "model_internal.h"

#include "core/weight.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// The last error message on this thread (ignis_model_last_error).
thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

ninfer::Weight to_weight(const ignis_bound_tensor &t) {
  ninfer::Weight w{};
  w.qtype = static_cast<ninfer::QType>(t.qtype);
  w.layout = static_cast<ninfer::QuantLayout>(t.layout);
  w.qdata = t.qdata;
  w.qhigh = t.qhigh;
  w.scales = t.scales;
  // The packed payload's base address: every layout's low/code plane sits at
  // its start (NVFP4/FP8's own format validation requires `qdata == payload`
  // and `scales == payload + scale_plane_offset`, kernel/vendor/src/ops/
  // linear/{nvfp4,fp8}/*_format.cpp), and `bytes` is the layout's exact
  // encoded length (ignis_model.h).
  w.payload = t.qdata;
  w.payload_bytes = t.bytes;
  w.ndim = t.ndim;
  for (uint32_t i = 0; i < 4; ++i) {
    w.shape[i] = t.shape[i];
    w.padded_shape[i] = t.padded_shape[i];
  }
  w.weight_scale_divisor = t.weight_scale_divisor;
  w.input_scale_divisor = t.input_scale_divisor;
  // A 2-D weight's row/column counts: `ninfer::Weight` carries these
  // separately from `shape` (ops::linear / ops::embedding read n/k
  // directly, e.g. w8_dispatch's launch-table lookup) -- GitHub #54 is the
  // first caller to drive a bound weight through either op, so this is the
  // first gap that would otherwise surface as n/k == 0.
  if (t.ndim == 2) {
    w.n = static_cast<int32_t>(t.shape[0]);
    w.k = static_cast<int32_t>(t.shape[1]);
  }
  // The W8G32_F16S group geometry + scale dtype (constant for the qtype,
  // not carried by `ignis_bound_tensor`): required by
  // ninfer::ops::embedding's W8 metadata validation (the two W8G32 text
  // endpoints, token_embedding and output_head).
  if (w.qtype == ninfer::QType::W8G32_F16S) {
    w.group_size = 32;
    w.group = 32;
    w.scale_dtype = ninfer::DType::FP16;
  }
  // The NVFP4 blockscale group geometry + scale dtype (constant for the
  // qtype, not carried by `ignis_bound_tensor`): required by
  // ninfer::ops::detail::validate_nvfp4_weight (GitHub #58 is the first
  // caller to drive a bound NVFP4 weight through an NVFP4 op).
  if (w.qtype == ninfer::QType::NVFP4) {
    w.group_size = 16;
    w.group = 16;
    w.scale_dtype = ninfer::DType::FP8_E4M3FN;
  }
  return w;
}

bool shape_matches(const ignis_bound_tensor &t, std::initializer_list<int64_t> want) {
  if (t.ndim != want.size()) {
    return false;
  }
  uint32_t i = 0;
  for (int64_t dim : want) {
    if (static_cast<int64_t>(t.shape[i]) != dim) {
      return false;
    }
    ++i;
  }
  return true;
}

std::string shape_str(std::initializer_list<int64_t> want) {
  std::string s = "[";
  bool first = true;
  for (int64_t dim : want) {
    if (!first) {
      s += ",";
    }
    first = false;
    s += std::to_string(dim);
  }
  s += "]";
  return s;
}

// The name -> index map + "consumed" bitmap every field pull goes through.
// An entry left unconsumed after every field is bound is an extra bound
// tensor -- the topology never asked for it.
class ModelBinder {
 public:
  ModelBinder(const ignis_bound_tensor *tensors, uint64_t count)
      : tensors_(tensors), used_(count, false) {
    index_.reserve(count * 2);
  }

  bool build_index(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      if (tensors_[i].name == nullptr) {
        set_error("ignis_model_load: bound tensor " + std::to_string(i) + " has a null name");
        return false;
      }
      auto result = index_.emplace(tensors_[i].name, i);
      if (!result.second) {
        set_error(std::string("ignis_model_load: duplicate bound tensor: ") + tensors_[i].name);
        return false;
      }
    }
    return true;
  }

  // Bind a required field: look up `name`, check its shape, and fill `out`.
  // False (error set) on a missing or mis-shaped tensor.
  bool bind(const std::string &name, std::initializer_list<int64_t> want_shape,
            ninfer::Weight &out) {
    auto it = index_.find(name);
    if (it == index_.end()) {
      set_error("ignis_model_load: missing bound tensor: " + name);
      return false;
    }
    const ignis_bound_tensor &t = tensors_[it->second];
    if (!shape_matches(t, want_shape)) {
      set_error("ignis_model_load: " + name + " has an unexpected shape (want " +
                shape_str(want_shape) + ")");
      return false;
    }
    used_[it->second] = true;
    out = to_weight(t);
    return true;
  }

  bool require_no_extras() const {
    for (uint64_t i = 0; i < used_.size(); ++i) {
      if (!used_[i]) {
        set_error(std::string("ignis_model_load: extra bound tensor: ") + tensors_[i].name);
        return false;
      }
    }
    return true;
  }

 private:
  const ignis_bound_tensor *tensors_;
  std::vector<bool> used_;
  std::unordered_map<std::string, uint64_t> index_;
};

// Every geometry the per-layer schema's expected shapes are derived from
// (mirrors crates/core/src/compute.rs `ModelConfig`'s derivations exactly --
// keep the two in step).
struct Geometry {
  int64_t hidden;
  int64_t vocab;
  int64_t gqa_width;
  int64_t gqa_kv_width;
  int64_t head_dim;
  int64_t ffn_intermediate;
  int64_t gdn_conv_channels;
  int64_t gdn_in_proj_m;
  int64_t gdn_norm_width;
  int64_t gdn_ab_width;
  int64_t gdn_state_rows;
  int64_t gdn_num_layers;

  static Geometry from(const ignis_topology &t) {
    Geometry g{};
    g.hidden = static_cast<int64_t>(t.hidden);
    g.vocab = static_cast<int64_t>(t.vocab);
    g.gqa_width = static_cast<int64_t>(t.num_q_heads * t.head_dim);
    g.gqa_kv_width = static_cast<int64_t>(t.num_kv_heads * t.head_dim);
    g.head_dim = static_cast<int64_t>(t.head_dim);
    g.ffn_intermediate = static_cast<int64_t>(t.ffn_intermediate);
    g.gdn_conv_channels = static_cast<int64_t>(t.gdn_q_width + t.gdn_state_cols + t.gdn_state_rows);
    g.gdn_in_proj_m =
        static_cast<int64_t>(t.gdn_q_width + t.gdn_state_cols + t.gdn_state_rows + t.gdn_z_width);
    g.gdn_ab_width = static_cast<int64_t>(t.gdn_ab_width);
    g.gdn_state_rows = static_cast<int64_t>(t.gdn_state_rows);
    g.gdn_num_layers = static_cast<int64_t>(t.gdn_num_layers);
    // The GDN gated RMSNorm's per-head width (state_rows / value heads); the
    // caller already rejected gdn_num_layers == 0 with gdn_state_rows != 0.
    g.gdn_norm_width = t.gdn_num_layers == 0 ? 0 : g.gdn_state_rows / g.gdn_num_layers;
    return g;
  }
};

// The GDN causal-conv kernel width (the reference's `gdn_conv_kernel`
// model constant -- not per-model config, so it is not on the topology
// descriptor).
constexpr int64_t kGdnConvKernel = 4;

bool bind_gqa_layer(ModelBinder &binder, const std::string &prefix, const Geometry &g,
                     GqaLayerWeights &w) {
  return binder.bind(prefix + "input_norm", {g.hidden}, w.input_norm) &&
         binder.bind(prefix + "attention/query_key_gate_value",
                     {2 * g.gqa_width + 2 * g.gqa_kv_width, g.hidden}, w.query_key_gate_value) &&
         binder.bind(prefix + "attention/query_norm", {g.head_dim}, w.query_norm) &&
         binder.bind(prefix + "attention/key_norm", {g.head_dim}, w.key_norm) &&
         binder.bind(prefix + "attention/output", {g.hidden, g.gqa_width}, w.output) &&
         binder.bind(prefix + "post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind(prefix + "mlp/gate_up", {2 * g.ffn_intermediate, g.hidden}, w.mlp_gate_up) &&
         binder.bind(prefix + "mlp/down", {g.hidden, g.ffn_intermediate}, w.mlp_down);
}

bool bind_gdn_layer(ModelBinder &binder, const std::string &prefix, const Geometry &g,
                     GdnLayerWeights &w) {
  return binder.bind(prefix + "input_norm", {g.hidden}, w.input_norm) &&
         binder.bind(prefix + "gdn/a_log", {g.gdn_num_layers}, w.a_log) &&
         binder.bind(prefix + "gdn/dt_bias", {g.gdn_num_layers}, w.dt_bias) &&
         binder.bind(prefix + "gdn/convolution", {kGdnConvKernel, g.gdn_conv_channels},
                     w.convolution) &&
         binder.bind(prefix + "gdn/a_b_projection", {g.gdn_ab_width, g.hidden}, w.a_b_projection) &&
         binder.bind(prefix + "gdn/query_key_value_z", {g.gdn_in_proj_m, g.hidden},
                     w.query_key_value_z) &&
         binder.bind(prefix + "gdn/norm", {g.gdn_norm_width}, w.norm) &&
         binder.bind(prefix + "gdn/output", {g.hidden, g.gdn_state_rows}, w.output) &&
         binder.bind(prefix + "post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind(prefix + "mlp/gate_up", {2 * g.ffn_intermediate, g.hidden}, w.mlp_gate_up) &&
         binder.bind(prefix + "mlp/down", {g.hidden, g.ffn_intermediate}, w.mlp_down);
}

// ---------------------------------------------------------------------------
// P2-01 (GitHub #83): the program scratch reservation.
//
// Sized once at load for a prefill chunk of `prefill_chunk_tokens` tokens,
// from the layer's own plain activation allocations (the same shapes
// kernel/src/gqa_layer.cu / gdn_layer.cu allocate) plus each dispatched
// vendored op's own workspace-capacity query over the token interval
// [1, prefill_chunk_tokens] -- under the widest compute policy each weight's
// own qtype admits (`widest_policy_for` below: AllowA4 for NVFP4, A16Only
// for everything else -- the real artifact mixes NVFP4 with a few BF16
// exception arms, GitHub #83's own gate run), so turning AllowA4 on for a
// weight that already admits it (G2, GitHub #85) changes no reservation.
// This is a load-time host-arithmetic mirror of those files' allocation
// sequence, not a dry run: no device call, no sequence-state pool (it does
// not exist yet at model-load time), and no kernel dispatch.
//
// The GDN recurrence's fixed per-head state dimension for this project's
// single model family (ADR 0001): value_head_dim == key_head_dim == 128
// (mirrors the sequence-state pool's spec, GitHub #55, which the model
// handle's GDN layers are bound against).
constexpr int64_t kGdnHeadDim = 128;

constexpr std::size_t kArenaAlign = 256;

// Mirrors `ninfer::DeviceArena::alloc_bytes`'s alignment rounding
// (kernel/vendor/src/core/arena.cu) so the reservation matches what the
// arena will actually consume per allocation.
std::size_t round_up_arena_align(std::size_t bytes) {
  return (bytes + (kArenaAlign - 1)) & ~(kArenaAlign - 1);
}

std::size_t bf16_bytes(int64_t elements) {
  return round_up_arena_align(static_cast<std::size_t>(elements) * 2);
}

std::size_t fp32_bytes(int64_t elements) {
  return round_up_arena_align(static_cast<std::size_t>(elements) * 4);
}

std::size_t i32_bytes(int64_t elements) {
  return round_up_arena_align(static_cast<std::size_t>(elements) * 4);
}

// The widest compute policy `qtype`'s own registered arm admits (the real
// 27B artifact mixes NVFP4 with a few documented BF16 exception arms --
// GQA layer 3's attention input and output, GDN layer 4's output,
// docs/layer-reference-fixture.md -- and BF16_CTRL / W8G32_F16S admit only
// LinearPolicy::A16Only; only NVFP4 admits AllowA4). Sizing every op at its
// own weight's real qtype (not a blanket AllowA4) is what "widest policy
// the engine may adopt" means per weight.
ninfer::ops::LinearPolicy widest_policy_for(ninfer::QType qtype) {
  return qtype == ninfer::QType::NVFP4 ? ninfer::ops::LinearPolicy::AllowA4
                                        : ninfer::ops::LinearPolicy::A16Only;
}

// One GQA layer's peak scratch at `T` tokens (kernel/src/gqa_layer.cu's
// `run_gqa_layer`): every plain activation buffer it allocates, plus the
// attention / attn_input_proj / linear_add / linear_swiglu workspace
// queries over [1, T].
std::size_t gqa_layer_scratch_bytes(const ignis_topology &topology, const GqaLayerWeights &w,
                                     std::int32_t T, uint32_t max_context_tokens) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto q_width = static_cast<std::int32_t>(topology.num_q_heads * topology.head_dim);
  const auto kv_width = static_cast<std::int32_t>(topology.num_kv_heads * topology.head_dim);
  const auto ffn = static_cast<std::int32_t>(topology.ffn_intermediate);
  const auto q_heads = static_cast<std::int32_t>(topology.num_q_heads);

  std::size_t bytes = 0;
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * T);   // normalized
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * T);  // query
  bytes += bf16_bytes(static_cast<int64_t>(kv_width) * T); // key
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * T);  // gate
  bytes += bf16_bytes(static_cast<int64_t>(kv_width) * T); // value
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * T);  // rotated_query
  bytes += bf16_bytes(static_cast<int64_t>(kv_width) * T); // rotated_key
  bytes += i32_bytes(T);                                   // positions
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * T);  // attention
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * T);   // post
  bytes += bf16_bytes(static_cast<int64_t>(ffn) * T);      // fused

  const ninfer::ops::GqaExecutionEnvelope envelope{
      /*min_visible_keys=*/1, /*max_visible_keys=*/max_context_tokens};
  bytes += round_up_arena_align(ninfer::ops::gqa_attention_workspace_capacity_bytes(
      q_heads, ninfer::DType::BF16, envelope, /*batch_size=*/1, /*min_width=*/1,
      /*max_width=*/T));
  bytes += round_up_arena_align(ninfer::ops::attn_input_proj_workspace_capacity_bytes(
      w.query_key_gate_value.qtype, w.query_key_gate_value.n, w.query_key_gate_value.k,
      widest_policy_for(w.query_key_gate_value.qtype), 1, T));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.output.qtype, w.output.n, w.output.k, widest_policy_for(w.output.qtype), 1, T));
  bytes += round_up_arena_align(ninfer::ops::linear_swiglu_workspace_capacity_bytes(
      w.mlp_gate_up.qtype, w.mlp_gate_up.n, w.mlp_gate_up.k,
      widest_policy_for(w.mlp_gate_up.qtype), 1, T));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.mlp_down.qtype, w.mlp_down.n, w.mlp_down.k, widest_policy_for(w.mlp_down.qtype), 1, T));
  return bytes;
}

// One GDN layer's peak scratch at `T` tokens (kernel/src/gdn_layer.cu's
// `run_gdn_layer`): every plain activation buffer it allocates, plus the
// gdn_input_proj / gdn_gating_proj / gated_delta_net / linear_add /
// linear_swiglu workspace queries over [1, T].
std::size_t gdn_layer_scratch_bytes(const ignis_topology &topology, const GdnLayerWeights &w,
                                     std::int32_t T) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto ffn = static_cast<std::int32_t>(topology.ffn_intermediate);
  const auto state_rows = static_cast<std::int32_t>(topology.gdn_state_rows);
  const auto state_cols = static_cast<std::int32_t>(topology.gdn_state_cols);
  const auto q_width = static_cast<std::int32_t>(topology.gdn_q_width);
  const auto conv_channels = q_width + state_cols + state_rows;
  const auto value_width = state_rows;
  const auto qk_width = (q_width + state_cols) / 2;
  const auto value_heads = value_width / static_cast<std::int32_t>(kGdnHeadDim);
  const auto qk_heads = qk_width / static_cast<std::int32_t>(kGdnHeadDim);

  std::size_t bytes = 0;
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * T);        // h
  bytes += bf16_bytes(static_cast<int64_t>(conv_channels) * T); // qkv
  bytes += bf16_bytes(static_cast<int64_t>(conv_channels) * T); // qkv_conv
  bytes += bf16_bytes(static_cast<int64_t>(qk_width) * T);      // query
  bytes += bf16_bytes(static_cast<int64_t>(qk_width) * T);      // key
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * T);   // value
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * T);   // zbuf
  bytes += fp32_bytes(static_cast<int64_t>(value_heads) * T);   // g
  bytes += fp32_bytes(static_cast<int64_t>(value_heads) * T);   // beta
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * T);   // recurrent
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * T);   // gated
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * T);        // post
  bytes += bf16_bytes(static_cast<int64_t>(ffn) * T);           // fused

  bytes += round_up_arena_align(ninfer::ops::gdn_input_proj_workspace_capacity_bytes(
      w.query_key_value_z.qtype, w.query_key_value_z.n, w.query_key_value_z.k,
      widest_policy_for(w.query_key_value_z.qtype), 1, T));
  bytes += round_up_arena_align(
      ninfer::ops::gdn_gating_proj_workspace_capacity_bytes(value_heads, hidden, 1, T));
  bytes += round_up_arena_align(ninfer::ops::gated_delta_net_workspace_capacity_bytes(
      qk_heads, value_heads, /*normalize_qk=*/true, 1, T));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.output.qtype, w.output.n, w.output.k, widest_policy_for(w.output.qtype), 1, T));
  bytes += round_up_arena_align(ninfer::ops::linear_swiglu_workspace_capacity_bytes(
      w.mlp_gate_up.qtype, w.mlp_gate_up.n, w.mlp_gate_up.k,
      widest_policy_for(w.mlp_gate_up.qtype), 1, T));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.mlp_down.qtype, w.mlp_down.n, w.mlp_down.k, widest_policy_for(w.mlp_down.qtype), 1, T));
  return bytes;
}

// The outer program scope's own allocations at `chunk` tokens
// (kernel/src/step.cu's `run_program_token`): the per-chunk token-id
// staging and residual pair (widened here so P2-02's chunk loop needs no
// new allocation), plus the single-position final-norm / output-head /
// argmax stage (only the span's last position feeds the output head,
// GitHub #72 -- independent of the chunk width).
std::size_t program_outer_scratch_bytes(const ignis_topology &topology, std::int32_t chunk) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto vocab = static_cast<std::int32_t>(topology.vocab);

  std::size_t bytes = 0;
  bytes += i32_bytes(chunk);
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * chunk); // left
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * chunk); // right
  bytes += bf16_bytes(hidden);                               // normalized
  bytes += bf16_bytes(vocab);                                 // logits
  bytes += i32_bytes(1);                                      // argmax_out
  return bytes;
}

// The program scratch arena's total reservation: the outer scope's own
// allocations plus the widest single decoder layer's peak (layers run
// sequentially, one nested scope at a time, so only one layer's scratch is
// ever live alongside the outer scope's).
std::size_t compute_program_scratch_bytes(const ignis_model &model, const ignis_topology &topology,
                                          std::int32_t chunk, uint32_t max_context_tokens) {
  std::size_t layer_peak = 0;
  for (const auto &layer : model.layers) {
    const std::size_t layer_bytes = layer.kind == IGNIS_LAYER_GQA
        ? gqa_layer_scratch_bytes(topology, layer.gqa, chunk, max_context_tokens)
        : gdn_layer_scratch_bytes(topology, layer.gdn, chunk);
    layer_peak = std::max(layer_peak, layer_bytes);
  }
  return program_outer_scratch_bytes(topology, chunk) + layer_peak;
}

} // namespace

extern "C" int32_t ignis_model_load(const struct ignis_bound_tensor *tensors, uint64_t count,
                                     const struct ignis_topology *topology,
                                     uint32_t prefill_chunk_tokens, uint32_t max_context_tokens,
                                     struct ignis_model **out_model) {
  if (out_model != nullptr) {
    *out_model = nullptr;
  }
  if (tensors == nullptr || topology == nullptr || out_model == nullptr) {
    set_error("ignis_model_load: null argument");
    return -1;
  }
  if (topology->num_layers > 0 && topology->layer_kinds == nullptr) {
    set_error("ignis_model_load: topology.layer_kinds is null");
    return -1;
  }
  if (topology->gdn_num_layers == 0) {
    set_error("ignis_model_load: topology.gdn_num_layers must be positive");
    return -1;
  }
  if (topology->gdn_state_rows % topology->gdn_num_layers != 0) {
    set_error("ignis_model_load: gdn_state_rows is not a multiple of gdn_num_layers");
    return -1;
  }
  // P2-01 (GitHub #83): the prefill chunk width the caller will hand
  // ignis_program_prefill, validated against the reference's own alignment
  // rule (also the alignment the GDN chunked kernels' 64-token chunk
  // divides evenly).
  if (prefill_chunk_tokens == 0 || prefill_chunk_tokens % 128 != 0) {
    set_error("ignis_model_load: prefill_chunk_tokens must be a nonzero multiple of 128");
    return -1;
  }
  if (max_context_tokens == 0) {
    set_error("ignis_model_load: max_context_tokens must be positive");
    return -1;
  }
  // A chunk wider than the sequence pool's own context bound can never be
  // prefilled anyway, and the GQA attention workspace query needs
  // max_visible_keys >= the query width it is sized for.
  if (prefill_chunk_tokens > max_context_tokens) {
    set_error("ignis_model_load: prefill_chunk_tokens (" + std::to_string(prefill_chunk_tokens) +
              ") must not exceed max_context_tokens (" + std::to_string(max_context_tokens) + ")");
    return -1;
  }

  ModelBinder binder(tensors, count);
  if (!binder.build_index(count)) {
    return -1;
  }

  const Geometry g = Geometry::from(*topology);
  auto model = std::make_unique<ignis_model>();

  if (!binder.bind("text/token_embedding", {g.vocab, g.hidden}, model->token_embedding) ||
      !binder.bind("text/final_norm", {g.hidden}, model->final_norm) ||
      !binder.bind("text/output_head", {g.vocab, g.hidden}, model->output_head)) {
    return -1;
  }

  model->layers.resize(topology->num_layers);
  for (uint32_t i = 0; i < topology->num_layers; ++i) {
    const std::string prefix = "text/layers/" + std::to_string(i) + "/";
    const auto kind = static_cast<ignis_layer_kind>(topology->layer_kinds[i]);
    model->layers[i].kind = kind;
    const bool ok = (kind == IGNIS_LAYER_GQA)
                        ? bind_gqa_layer(binder, prefix, g, model->layers[i].gqa)
                        : bind_gdn_layer(binder, prefix, g, model->layers[i].gdn);
    if (!ok) {
      return -1;
    }
  }

  if (!binder.require_no_extras()) {
    return -1;
  }

  uint64_t vram_bytes = 0;
  for (uint64_t i = 0; i < count; ++i) {
    vram_bytes += tensors[i].bytes;
  }
  model->bound_tensor_count = count;
  model->vram_bytes = vram_bytes;

  // The step ABI's geometry + program resources (ADR 0009, GitHub #54): a
  // dedicated stream and a small scratch arena for step intermediates
  // (embedding / norm / logits / argmax buffers). Owned by the model handle
  // so no stream crosses the ABI.
  model->hidden = g.hidden;
  model->vocab = g.vocab;
  model->rms_norm_eps = topology->rms_norm_eps;

  const cudaError_t stream_err = cudaStreamCreate(&model->stream);
  if (stream_err != cudaSuccess) {
    set_error(std::string("ignis_model_load: cudaStreamCreate failed: ") +
              cudaGetErrorString(stream_err));
    return -1;
  }

  // P2-01 (GitHub #83): reserve the scratch once, sized for a
  // `prefill_chunk_tokens`-wide chunk (see compute_program_scratch_bytes
  // above) -- never at the first long prompt. A chunk whose reservation
  // does not fit the device's free memory fails the load right here, with
  // a message naming the shortfall.
  std::size_t scratch_bytes = 0;
  try {
    scratch_bytes = compute_program_scratch_bytes(
        *model, *topology, static_cast<std::int32_t>(prefill_chunk_tokens), max_context_tokens);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: prefill scratch sizing failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (mem_err == cudaSuccess && scratch_bytes > free_bytes) {
    set_error("ignis_model_load: a " + std::to_string(prefill_chunk_tokens) +
              "-token prefill chunk needs a " + std::to_string(scratch_bytes) +
              "-byte scratch reservation, but only " + std::to_string(free_bytes) +
              " bytes are free -- pick a smaller prefill chunk width");
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  try {
    model->scratch = std::make_unique<ninfer::DeviceArena>(scratch_bytes);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: scratch arena allocation failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  *out_model = model.release();
  return 0;
}

extern "C" int32_t ignis_model_stats(const struct ignis_model *model,
                                      struct ignis_model_stats *out_stats) {
  if (model == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->vram_bytes = model->vram_bytes;
  out_stats->bound_tensor_count = model->bound_tensor_count;
  return 0;
}

extern "C" void ignis_model_free(struct ignis_model *model) {
  if (model != nullptr && model->stream != nullptr) {
    cudaStreamDestroy(model->stream);
  }
  delete model;
}

extern "C" const char *ignis_model_last_error(void) {
  return g_last_error.c_str();
}
