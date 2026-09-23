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
#include "ignis_step.h"

#include "attention_readout.h"
#include "layer_internal.h"
#include "model_internal.h"

#include "ignis_dflash2_topk.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/weight.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"
#include "ninfer/ops/vision_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
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
  // GitHub #178: a row-split row is stored K-padded to a multiple of 128
  // (`row-split-k128-v1`), and the row-split kernels step rows by that padded
  // width (the reference's `row_split_weight`: `padded_shape[1] =
  // padded_columns`). Every text-scope row-split weight is already 128-aligned,
  // so only the vision tower's `mlp/fc2` (K = 4304, stored at 4352) differs --
  // read at 4304 it came out as noise.
  if (t.ndim == 2 && w.layout == ninfer::QuantLayout::RowSplit) {
    constexpr int32_t kRowSplitKAlignment = 128;
    w.padded_shape[1] = (w.k + kRowSplitKAlignment - 1) / kRowSplitKAlignment * kRowSplitKAlignment;
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
  // GitHub #177: the vision tower's Q4/Q5/Q6 row-split G64 weights, F16
  // scales (the reference's `materialized_weight` for these qtypes).
  if (w.qtype == ninfer::QType::Q4G64_F16S || w.qtype == ninfer::QType::Q5G64_F16S ||
      w.qtype == ninfer::QType::Q6G64_F16S) {
    w.group_size = 64;
    w.group = 64;
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
// P5-02 (GitHub #150): the DFlash2 drafter's weights (the reference's
// `docs/maintainer/qwen3.8-27b-artifact.md` §15). Its hidden and MLP widths
// are the target's; the rest are the module's own constants.
constexpr int64_t kDflash2FeatureTaps = 5;   // target layers 5, 19, 33, 47, 61
constexpr int64_t kDflash2QueryHeads = 32;
constexpr int64_t kDflash2KvHeads = 8;
constexpr int64_t kDflash2HeadDim = 128;
constexpr int64_t kDflash2ConvTaps = 2;
constexpr int64_t kDflash2ConvProjRows = 1280; // 320 groups x 2 taps x 2
constexpr int64_t kDflash2SelectorRank = 256;
constexpr int64_t kDflash2WindowTokens = 2048;

bool bind_dflash2_layer(ModelBinder &binder, const std::string &prefix, const Geometry &g,
                        Dflash2LayerWeights &w) {
  const int64_t q_width = kDflash2QueryHeads * kDflash2HeadDim;
  const int64_t kv_width = kDflash2KvHeads * kDflash2HeadDim;
  const std::initializer_list<int64_t> conv_base{kDflash2ConvTaps, kDflash2ConvTaps, g.hidden};
  return binder.bind(prefix + "input_norm", {g.hidden}, w.input_norm) &&
         binder.bind(prefix + "attention/query_key_value", {q_width + 2 * kv_width, g.hidden},
                     w.query_key_value) &&
         binder.bind(prefix + "attention/query_norm", {kDflash2HeadDim}, w.query_norm) &&
         binder.bind(prefix + "attention/key_norm", {kDflash2HeadDim}, w.key_norm) &&
         binder.bind(prefix + "attention/output", {g.hidden, q_width}, w.output) &&
         binder.bind(prefix + "attention/conv_base", conv_base, w.attention_conv_base) &&
         binder.bind(prefix + "attention/conv_proj", {kDflash2ConvProjRows, g.hidden},
                     w.attention_conv_proj) &&
         binder.bind(prefix + "post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind(prefix + "mlp/gate_up", {2 * g.ffn_intermediate, g.hidden}, w.mlp_gate_up) &&
         binder.bind(prefix + "mlp/down", {g.hidden, g.ffn_intermediate}, w.mlp_down) &&
         binder.bind(prefix + "mlp/conv_base", conv_base, w.mlp_conv_base) &&
         binder.bind(prefix + "mlp/conv_proj", {kDflash2ConvProjRows, g.hidden}, w.mlp_conv_proj);
}

bool bind_dflash2(ModelBinder &binder, const Geometry &g, Dflash2Weights &w) {
  if (!binder.bind("dflash2/feature_projection", {g.hidden, kDflash2FeatureTaps * g.hidden},
                   w.feature_projection) ||
      !binder.bind("dflash2/context_norm", {g.hidden}, w.context_norm)) {
    return false;
  }
  for (std::size_t l = 0; l < kDflash2Layers; ++l) {
    const std::string prefix = "dflash2/layers/" + std::to_string(l) + "/";
    if (!bind_dflash2_layer(binder, prefix, g, w.layers[l])) {
      return false;
    }
  }
  return binder.bind("dflash2/final_norm", {g.hidden}, w.final_norm) &&
         binder.bind("dflash2/selector/hidden", {kDflash2SelectorRank, g.hidden},
                     w.selector_hidden) &&
         binder.bind("dflash2/selector/predecessor", {g.vocab, kDflash2SelectorRank},
                     w.selector_predecessor) &&
         binder.bind("dflash2/selector/successor", {g.vocab, kDflash2SelectorRank},
                     w.selector_successor);
}

// P5-03 (GitHub #152): the drafter's window lives in the sequence pool, one
// lane per slot (kernel/src/seq.cu), and the context append in step.cu writes
// the query/key/value widths below into it -- the three places must agree.
static_assert(kDflash2Layers == kIgnisDflash2Layers &&
                  kDflash2WindowTokens == kIgnisDflash2WindowTokens &&
                  kDflash2KvHeads == kIgnisDflash2KvHeads &&
                  kDflash2HeadDim == kIgnisDflash2HeadDim,
              "the drafter geometry bound here has drifted from the sequence pool's window");
static_assert(kDflash2FeatureTaps == static_cast<int64_t>(kDflash2TapLayers.size()) &&
                  kDflash2QueryHeads * kDflash2HeadDim == kDflash2QuerySize,
              "the drafter geometry bound here has drifted from step.cu's context append");

// ---------------------------------------------------------------------------
// P2-01 (GitHub #83): the program scratch reservation.
//
// Sized once at load for a prefill chunk of `prefill_chunk_tokens` tokens,
// from the layer's own plain activation allocations (the same shapes
// kernel/src/gqa_layer.cu / gdn_layer.cu allocate) plus each dispatched
// vendored op's own workspace-capacity query over the token interval
// [1, prefill_chunk_tokens] -- under the widest compute policy each weight's
// own qtype admits (`ignis_widest_linear_policy_for`, kernel/src/layer_internal.h:
// AllowA4 for NVFP4, A16Only for everything else -- the real artifact mixes
// NVFP4 with a few BF16 exception arms, GitHub #83's own gate run), so a
// dispatch site already using that policy (P2-02, GitHub #84's
// `linear_swiglu` call) reserves nothing new.
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

// One GQA layer's peak scratch at `T` tokens per row and `batch` rows
// (kernel/src/gqa_layer.cu's `run_gqa_layer` at batch 1, its
// `run_gqa_layer_graph` at T=1 and a decode round's width): every plain
// activation buffer it allocates -- all of them `T * batch` columns wide --
// plus the attention / attn_input_proj / linear_add / linear_swiglu
// workspace queries. The attention query is per-row (`batch_size`, widths
// over [1, T]); the projections see one matrix of `T * batch` columns and
// are queried that way.
//
// `cache_dtype` is the KV format's own declared element type (P4-05, GitHub
// #123): the layer asks the attention query under exactly this dtype, and
// the hq (U8) prompt route's answer is far larger than BF16's -- it
// materializes the envelope's visible history into two rotated-frame BF16
// scratch planes before the shared FA2 kernel runs over it. Sizing the
// arena as BF16 and then dispatching hq is the one way this reservation can
// be wrong, so the format reaches here rather than being assumed.
std::size_t gqa_layer_scratch_bytes(const ignis_topology &topology, const GqaLayerWeights &w,
                                     std::int32_t T, uint32_t max_context_tokens,
                                     ninfer::DType cache_dtype, std::int32_t batch) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto q_width = static_cast<std::int32_t>(topology.num_q_heads * topology.head_dim);
  const auto kv_width = static_cast<std::int32_t>(topology.num_kv_heads * topology.head_dim);
  const auto ffn = static_cast<std::int32_t>(topology.ffn_intermediate);
  const auto q_heads = static_cast<std::int32_t>(topology.num_q_heads);
  const std::int32_t columns = T * batch;

  std::size_t bytes = 0;
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);   // normalized
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * columns);  // query
  bytes += bf16_bytes(static_cast<int64_t>(kv_width) * columns); // key
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * columns);  // gate
  bytes += bf16_bytes(static_cast<int64_t>(kv_width) * columns); // value
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * columns);  // rotated_query
  bytes += bf16_bytes(static_cast<int64_t>(kv_width) * columns); // rotated_key
  bytes += i32_bytes(columns);                                   // positions
  bytes += i32_bytes(batch);                                     // kv_table_rows (P2-04, GitHub #86)
  bytes += bf16_bytes(static_cast<int64_t>(q_width) * columns);  // attention
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);   // post
  bytes += bf16_bytes(static_cast<int64_t>(ffn) * columns);      // fused

  const ninfer::ops::GqaExecutionEnvelope envelope{
      /*min_visible_keys=*/1, /*max_visible_keys=*/max_context_tokens};
  bytes += round_up_arena_align(ninfer::ops::gqa_attention_workspace_capacity_bytes(
      q_heads, cache_dtype, envelope, /*batch_size=*/batch, /*min_width=*/1,
      /*max_width=*/T));
  bytes += round_up_arena_align(ninfer::ops::attn_input_proj_workspace_capacity_bytes(
      w.query_key_gate_value.qtype, w.query_key_gate_value.n, w.query_key_gate_value.k,
      ignis_widest_linear_policy_for(w.query_key_gate_value.qtype), 1, columns));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.output.qtype, w.output.n, w.output.k, ignis_widest_linear_policy_for(w.output.qtype), 1,
      columns));
  bytes += round_up_arena_align(ninfer::ops::linear_swiglu_workspace_capacity_bytes(
      w.mlp_gate_up.qtype, w.mlp_gate_up.n, w.mlp_gate_up.k,
      ignis_widest_linear_policy_for(w.mlp_gate_up.qtype), 1, columns));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.mlp_down.qtype, w.mlp_down.n, w.mlp_down.k, ignis_widest_linear_policy_for(w.mlp_down.qtype),
      1, columns));
  return bytes;
}

// One GDN layer's peak scratch at `T` tokens per row and `batch` rows
// (kernel/src/gdn_layer.cu's `run_gdn_layer` at batch 1, its
// `run_gdn_layer_graph` at T=1 and a decode round's width): every plain
// activation buffer it allocates -- all of them `T * batch` columns wide --
// plus the gdn_input_proj / gdn_gating_proj / gated_delta_net / linear_add /
// linear_swiglu workspace queries over that column count. The snapshot form
// of the recurrence uses no arena at all, so its query is the conservative
// one either way.
std::size_t gdn_layer_scratch_bytes(const ignis_topology &topology, const GdnLayerWeights &w,
                                     std::int32_t T, std::int32_t batch) {
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
  const std::int32_t columns = T * batch;

  std::size_t bytes = 0;
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);        // h
  bytes += bf16_bytes(static_cast<int64_t>(conv_channels) * columns); // qkv
  bytes += bf16_bytes(static_cast<int64_t>(conv_channels) * columns); // qkv_conv
  bytes += bf16_bytes(static_cast<int64_t>(qk_width) * columns);      // query
  bytes += bf16_bytes(static_cast<int64_t>(qk_width) * columns);      // key
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * columns);   // value
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * columns);   // zbuf
  bytes += fp32_bytes(static_cast<int64_t>(value_heads) * columns);   // g
  bytes += fp32_bytes(static_cast<int64_t>(value_heads) * columns);   // beta
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * columns);   // recurrent
  bytes += bf16_bytes(static_cast<int64_t>(value_width) * columns);   // gated
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);        // post
  bytes += bf16_bytes(static_cast<int64_t>(ffn) * columns);           // fused

  bytes += round_up_arena_align(ninfer::ops::gdn_input_proj_workspace_capacity_bytes(
      w.query_key_value_z.qtype, w.query_key_value_z.n, w.query_key_value_z.k,
      ignis_widest_linear_policy_for(w.query_key_value_z.qtype), 1, columns));
  bytes += round_up_arena_align(
      ninfer::ops::gdn_gating_proj_workspace_capacity_bytes(value_heads, hidden, 1, columns));
  bytes += round_up_arena_align(ninfer::ops::gated_delta_net_workspace_capacity_bytes(
      qk_heads, value_heads, /*normalize_qk=*/true, 1, columns));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.output.qtype, w.output.n, w.output.k, ignis_widest_linear_policy_for(w.output.qtype), 1,
      columns));
  bytes += round_up_arena_align(ninfer::ops::linear_swiglu_workspace_capacity_bytes(
      w.mlp_gate_up.qtype, w.mlp_gate_up.n, w.mlp_gate_up.k,
      ignis_widest_linear_policy_for(w.mlp_gate_up.qtype), 1, columns));
  bytes += round_up_arena_align(ninfer::ops::linear_add_workspace_capacity_bytes(
      w.mlp_down.qtype, w.mlp_down.n, w.mlp_down.k, ignis_widest_linear_policy_for(w.mlp_down.qtype),
      1, columns));
  return bytes;
}

// The outer program scope's own allocations at `chunk` tokens per row and
// `batch` rows (kernel/src/step.cu's `run_program_token` at batch 1;
// kernel/src/decode_graph.cu's `ignis_decode_graph_run_batch` at chunk 1 and
// a decode round's width): the token-id staging and residual pair (widened
// here so P2-02's chunk loop needs no new allocation), plus the final-norm /
// output-head / argmax stage. Prefill feeds only the span's last position to
// the output head (GitHub #72), a decode round feeds one column per lane, so
// that stage is sized by `batch` rather than by the whole column count --
// except for the verify round (P5-04, GitHub #153, `head_every_column`),
// which norms every one of its `k+1` columns per lane before the output
// head (its logits and argmax land in model-owned buffers, not here).
std::size_t program_outer_scratch_bytes(const ignis_topology &topology, std::int32_t chunk,
                                        std::int32_t batch, bool head_every_column) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto vocab = static_cast<std::int32_t>(topology.vocab);
  const std::int32_t columns = chunk * batch;
  const std::int32_t head_columns = head_every_column ? columns : batch;

  std::size_t bytes = 0;
  bytes += i32_bytes(columns);
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);      // left
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);      // right
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * head_columns); // normalized
  bytes += bf16_bytes(static_cast<int64_t>(vocab) * batch);         // logits
  bytes += i32_bytes(1);                                            // argmax_out
  return bytes;
}

// The program scratch arena's total reservation: the outer scope's own
// allocations plus the widest single decoder layer's peak (layers run
// sequentially, one nested scope at a time, so only one layer's scratch is
// ever live alongside the outer scope's).
std::size_t compute_program_scratch_bytes(const ignis_model &model, const ignis_topology &topology,
                                          std::int32_t chunk, uint32_t max_context_tokens,
                                          ninfer::DType cache_dtype, std::int32_t batch,
                                          bool head_every_column = false) {
  std::size_t layer_peak = 0;
  for (const auto &layer : model.layers) {
    const std::size_t layer_bytes = layer.kind == IGNIS_LAYER_GQA
        ? gqa_layer_scratch_bytes(topology, layer.gqa, chunk, max_context_tokens, cache_dtype, batch)
        : gdn_layer_scratch_bytes(topology, layer.gdn, chunk, batch);
    layer_peak = std::max(layer_peak, layer_bytes);
  }
  return program_outer_scratch_bytes(topology, chunk, batch, head_every_column) + layer_peak;
}

// ---------------------------------------------------------------------------
// The ReplaySSM record geometry of the verify round at window `k`.
ninfer::GdnReplayRecordSpec verify_record_spec(const ignis_topology &topology, uint32_t window) {
  std::int32_t gdn_layers = 0;
  for (uint32_t i = 0; i < topology.num_layers; ++i) {
    if (static_cast<ignis_layer_kind>(topology.layer_kinds[i]) == IGNIS_LAYER_GDN) {
      ++gdn_layers;
    }
  }
  ninfer::GdnReplayRecordSpec spec{};
  spec.layers = gdn_layers;
  spec.record_capacity = IGNIS_DECODE_MAX_BATCH;
  spec.width = static_cast<std::int32_t>(window) + 1;
  spec.conv_channels =
      static_cast<std::int32_t>(topology.gdn_q_width + topology.gdn_state_cols + topology.gdn_state_rows);
  spec.qk_heads = static_cast<std::int32_t>(topology.gdn_q_width / kGdnHeadDim);
  spec.value_heads = static_cast<std::int32_t>(topology.gdn_state_rows / kGdnHeadDim);
  spec.key_dim = static_cast<std::int32_t>(kGdnHeadDim);
  spec.value_dim = static_cast<std::int32_t>(kGdnHeadDim);
  return spec;
}

// GitHub #210: every byte `build_verify_round` below allocates, without
// allocating -- the VRAM plan's verify line. `IgnisVerifyRound::device_bytes`
// reads the same total back off a built round, and the load checks the two
// agree.
std::size_t verify_round_bytes(const ignis_topology &topology, uint32_t window, bool vision) {
  const auto k = static_cast<std::size_t>(window);
  const std::size_t columns = k + 1;
  const std::size_t lanes = IGNIS_DECODE_MAX_BATCH;
  const auto hidden = static_cast<std::size_t>(topology.hidden);
  const auto vocab = static_cast<std::size_t>(topology.vocab);
  const std::size_t i32 = sizeof(int32_t);
  const std::size_t bf16 = sizeof(std::uint16_t);

  std::size_t bytes = 0;
  // anchors, base_positions, extents, valid_columns, lengths,
  // licensed_counts, accepted, selectors
  bytes += 8 * lanes * i32;
  bytes += k * lanes * i32;                          // drafts
  bytes += (vision ? 5 : 4) * columns * lanes * i32; // verify_ids, positions, target_tokens,
                                                     // licensed_tokens, rope_positions
  bytes += vocab * columns * lanes * bf16;           // logits
  bytes += hidden * columns * lanes * bf16;          // hidden
  bytes += hidden * lanes * bf16;                    // selected_hidden
  ninfer::LayoutBuilder builder;
  (void)ninfer::plan_gdn_replay_records(builder, verify_record_spec(topology, window));
  bytes += builder.finish(kArenaAlign, "GDN replay records");
  bytes += std::max<std::size_t>(ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                     static_cast<std::int32_t>(vocab), static_cast<std::int32_t>(k),
                                     static_cast<std::int32_t>(k), 1, static_cast<std::int32_t>(lanes)),
                                 kArenaAlign);
  return bytes;
}

// P5-04 (GitHub #153): the verify round's substrate (model_internal.h's
// `IgnisVerifyRound`) at window `k`, for `IGNIS_DECODE_MAX_BATCH` lanes. The
// ReplaySSM record geometry is the model's GDN geometry -- the layer count
// from the topology's kinds, the head counts from the state widths -- and
// the vendored planner (`plan_gdn_replay_records`) lays the four planes out;
// the fold only admits its registered all-layer geometries, so a wrong
// count here fails at the first round, by name, not silently.
//
// GitHub #195: `vision` adds the round's rope-position staging, so a
// multimodal sequence's verify columns rotate at `position + rope_delta` the
// way its decode rounds do. A load without vision allocates nothing for it
// and every round rotates at `positions` itself, exactly as before.
std::unique_ptr<IgnisVerifyRound> build_verify_round(const ignis_topology &topology,
                                                     uint32_t window, bool vision) {
  auto verify = std::make_unique<IgnisVerifyRound>();
  verify->window = window;
  const auto k = static_cast<std::int32_t>(window);
  const std::int32_t columns = k + 1;
  const std::int32_t lanes = IGNIS_DECODE_MAX_BATCH;
  const auto hidden = static_cast<std::size_t>(topology.hidden);
  const auto vocab = static_cast<std::int32_t>(topology.vocab);

  const auto i32 = [](std::int64_t n) {
    return std::make_unique<ninfer::DeviceBuffer>(static_cast<std::size_t>(n) * sizeof(int32_t));
  };
  const auto bf16 = [](std::size_t n) {
    return std::make_unique<ninfer::DeviceBuffer>(n * sizeof(std::uint16_t));
  };
  verify->anchors = i32(lanes);
  verify->drafts = i32(static_cast<std::int64_t>(k) * lanes);
  verify->base_positions = i32(lanes);
  verify->extents = i32(lanes);
  verify->valid_columns = i32(lanes);
  verify->lengths = i32(lanes);
  verify->verify_ids = i32(static_cast<std::int64_t>(columns) * lanes);
  verify->positions = i32(static_cast<std::int64_t>(columns) * lanes);
  verify->target_tokens = i32(static_cast<std::int64_t>(columns) * lanes);
  verify->logits = bf16(static_cast<std::size_t>(vocab) * columns * lanes);
  verify->hidden = bf16(hidden * columns * lanes);
  verify->licensed_tokens = i32(static_cast<std::int64_t>(columns) * lanes);
  verify->licensed_counts = i32(lanes);
  verify->accepted = i32(lanes);
  verify->selectors = i32(lanes);
  verify->selected_hidden = bf16(hidden * lanes);
  if (vision) {
    verify->rope_positions = i32(static_cast<std::int64_t>(columns) * lanes);
  }
  // The staging a capture reads before any round refreshed it must still be
  // a legal input (a valid extent, a real slot): zero is one. A lane's
  // anchor column is always valid -- the drafter's query block (P5-05)
  // admits no fewer than one -- so the valid-column count starts at 1.
  for (auto *buffer : {verify->anchors.get(), verify->drafts.get(), verify->base_positions.get(),
                       verify->extents.get(), verify->lengths.get(),
                       verify->rope_positions.get()}) {
    if (buffer != nullptr) {
      buffer->fill(0);
    }
  }
  const std::vector<std::int32_t> anchor_only(static_cast<std::size_t>(lanes), 1);
  verify->valid_columns->copy_from_host(anchor_only.data(), anchor_only.size() * sizeof(std::int32_t));

  ninfer::LayoutBuilder builder;
  verify->records_layout =
      ninfer::plan_gdn_replay_records(builder, verify_record_spec(topology, window));
  const std::size_t record_bytes = builder.finish(kArenaAlign, "GDN replay records");
  verify->records_backing = std::make_unique<ninfer::DeviceBuffer>(record_bytes);
  verify->records = ninfer::GdnReplayRecords(
      ninfer::DeviceSpan{verify->records_backing->p, verify->records_backing->bytes},
      verify->records_layout);

  const std::size_t accept_bytes = ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
      vocab, k, k, 1, lanes);
  verify->accept_workspace =
      std::make_unique<ninfer::DeviceArena>(std::max<std::size_t>(accept_bytes, kArenaAlign));
  return verify;
}

// P5-03 (GitHub #152): what one prefill chunk adds to the program scratch on
// a load with the DFlash2 drafter -- the feature taps and the context append
// kernel/src/step.cu runs over them (`append_dflash2_context`), in the order
// it allocates them. A chunk taps at most the span's last window of
// positions, so no buffer here is wider than kIgnisDflash2WindowTokens
// columns. The taps are live across the target layers and the rest after
// them, so the sum is conservative rather than exact; one drafter layer's
// buffers are live at a time.
std::size_t dflash2_prefill_scratch_bytes(const ignis_topology &topology, std::int32_t chunk) {
  const auto hidden = static_cast<int64_t>(topology.hidden);
  const int64_t columns =
      std::min<int64_t>(chunk, static_cast<int64_t>(kIgnisDflash2WindowTokens));
  const int64_t kv_width = kDflash2KvHeads * kDflash2HeadDim;

  std::size_t bytes = 0;
  bytes += bf16_bytes(kDflash2FeatureTaps * hidden * columns); // features
  bytes += i32_bytes(columns);                                 // positions
  bytes += i32_bytes(1);                                       // commit count
  bytes += i32_bytes(1);                                       // lane
  bytes += bf16_bytes(hidden * columns);                       // projected
  bytes += bf16_bytes(hidden * columns);                       // context
  bytes += bf16_bytes((kDflash2QuerySize + 2 * kv_width) * columns); // query_key_value
  bytes += bf16_bytes(kv_width * columns);                     // key_raw
  bytes += bf16_bytes(kv_width * columns);                     // key
  bytes += bf16_bytes(kv_width * columns);                     // value
  return bytes;
}

// P5-05 (GitHub #155): the fixed allowance the drafter's round scratch
// carries for the two vendored workspaces its forward calls into (`swa`,
// `linear_swiglu`). A constant rather than their queried sizes, so the Rust
// side (`Speculation::round_scratch_bytes`) states the reservation to the
// byte; the load refuses if the queries ever outgrow it. `swa`'s split-KV
// tiling saturates at 32 key tiles from a ~1k context on, where the two
// queries total just over 16 MiB: 32 MiB covers every context.
constexpr std::size_t kDflash2RoundWorkspaceBytes = 32 * 1024 * 1024;

// P5-05 (GitHub #155): the activations the drafter's round allocates from
// `drafter_scratch` (kernel/src/dflash2_drafter.cu) at window `k` across
// IGNIS_DECODE_MAX_BATCH lanes, in the order it allocates them: its forward
// -- both blocks of a layer counted together, so the sum is conservative --
// or the round's context append, whichever is larger.
std::size_t dflash2_round_activation_bytes(const ignis_topology &topology, std::int32_t window) {
  const auto hidden = static_cast<int64_t>(topology.hidden);
  const auto vocab = static_cast<int64_t>(topology.vocab);
  const auto intermediate = static_cast<int64_t>(topology.ffn_intermediate);
  const int64_t columns = static_cast<int64_t>(window + 1) * IGNIS_DECODE_MAX_BATCH;
  const int64_t drafts = static_cast<int64_t>(window) * IGNIS_DECODE_MAX_BATCH;
  const int64_t kv_width = kDflash2KvHeads * kDflash2HeadDim;
  const int64_t top_k = kDflash2SelectorTopK;

  std::size_t forward = 0;
  forward += i32_bytes(columns) * 2;                                // ids, positions
  forward += bf16_bytes(hidden * columns);                          // residual
  forward += bf16_bytes(hidden * columns) * 2;                      // attention: normed, conv
  forward += bf16_bytes(kDflash2ConvProjRows * columns);            // dynamic
  forward += bf16_bytes((kDflash2QuerySize + 2 * kv_width) * columns); // query_key_value
  forward += bf16_bytes(kDflash2QuerySize * columns) * 3;           // query_raw, query, attention
  forward += bf16_bytes(kv_width * columns) * 3;                    // key_raw, value, key
  forward += bf16_bytes(hidden * columns) * 2;                      // projected, conv_attention
  forward += bf16_bytes(hidden * columns) * 2;                      // mlp: normed, conv_hidden
  forward += bf16_bytes(kDflash2ConvProjRows * columns);            // dynamic
  forward += bf16_bytes(intermediate * columns);                    // intermediate
  forward += bf16_bytes(hidden * columns) * 2;                      // projected, conv_projected
  forward += bf16_bytes(hidden * drafts) * 2;                       // packed, proposal_hidden
  forward += bf16_bytes(vocab * drafts);                            // logits
  forward += i32_bytes(top_k * drafts) + bf16_bytes(top_k * drafts); // candidate ids, values
  // The row-split partials our own top-k merges (kernel/include/ignis_dflash2_topk.h),
  // asked of the same function the call site bumps with; a shape it forwards
  // to the vendored op reports zero, and the call site still bumps one byte
  // because the arena admits no empty allocation.
  forward += std::max<std::size_t>(
      ignis_dflash2_topk_workspace_bytes(static_cast<std::int32_t>(vocab),
                                         static_cast<std::int32_t>(drafts),
                                         static_cast<std::int32_t>(top_k)),
      1);                                                             // top-k partials
  forward += fp32_bytes(top_k * drafts) + i32_bytes(top_k * drafts); // unary, predecessors
  forward += bf16_bytes(kDflash2SelectorRank * drafts);             // hidden_proj
  forward += fp32_bytes(kDflash2SelectorRank * drafts);             // hidden_proj_f32
  forward += fp32_bytes(top_k * top_k * drafts);                    // scores

  std::size_t append = 0;
  append += bf16_bytes(hidden * columns) * 2;                       // projected, context
  append += bf16_bytes((kDflash2QuerySize + 2 * kv_width) * columns); // query_key_value
  append += bf16_bytes(kv_width * columns) * 3;                     // key_raw, key, value
  return std::max(forward, append);
}

// GitHub #177: the vision tower's 333 objects, shapes from the reference's
// `impl/vision/bindings.cpp` and (merger fc2, the 27B's own out width)
// `qwen3_6_27b/impl/load/bindings.cpp`.
bool bind_vision(ModelBinder &binder, const Geometry &g, VisionWeights &w) {
  const int64_t h = kVisionHidden;
  const int64_t inter = kVisionIntermediate;
  if (!binder.bind("vision/patch_embedding", {h, kVisionPatchDim}, w.patch_embedding) ||
      !binder.bind("vision/patch_embedding_bias", {h}, w.patch_embedding_bias) ||
      !binder.bind("vision/position_embedding", {kVisionPositionEmbeddings, h},
                   w.position_embedding)) {
    return false;
  }
  for (int32_t l = 0; l < kVisionLayers; ++l) {
    const std::string p = "vision/layers/" + std::to_string(l) + "/";
    VisionLayerWeights &layer = w.layers[l];
    if (!binder.bind(p + "attention/qkv", {3 * h, h}, layer.qkv) ||
        !binder.bind(p + "attention/qkv_bias", {3 * h}, layer.qkv_bias) ||
        !binder.bind(p + "attention/output", {h, h}, layer.output) ||
        !binder.bind(p + "attention/output_bias", {h}, layer.output_bias) ||
        !binder.bind(p + "mlp/fc1", {inter, h}, layer.fc1) ||
        !binder.bind(p + "mlp/fc1_bias", {inter}, layer.fc1_bias) ||
        !binder.bind(p + "mlp/fc2", {h, inter}, layer.fc2) ||
        !binder.bind(p + "mlp/fc2_bias", {h}, layer.fc2_bias) ||
        !binder.bind(p + "norm1/weight", {h}, layer.norm1_weight) ||
        !binder.bind(p + "norm1/bias", {h}, layer.norm1_bias) ||
        !binder.bind(p + "norm2/weight", {h}, layer.norm2_weight) ||
        !binder.bind(p + "norm2/bias", {h}, layer.norm2_bias)) {
      return false;
    }
  }
  return binder.bind("vision/merger/fc1", {kVisionMergerHidden, kVisionMergerHidden},
                     w.merger_fc1) &&
         binder.bind("vision/merger/fc1_bias", {kVisionMergerHidden}, w.merger_fc1_bias) &&
         binder.bind("vision/merger/fc2", {g.hidden, kVisionMergerHidden}, w.merger_fc2) &&
         binder.bind("vision/merger/fc2_bias", {g.hidden}, w.merger_fc2_bias) &&
         binder.bind("vision/merger/norm/weight", {h}, w.merger_norm_weight) &&
         binder.bind("vision/merger/norm/bias", {h}, w.merger_norm_bias);
}

constexpr std::size_t kVisionWorkspaceAlignment = 256;

// GitHub #177: one item's `[hidden, tokens]` BF16 encoder output (the
// reference's `VisionContext::output_transient_bytes`).
std::size_t vision_output_transient_bytes(std::int64_t hidden, std::int32_t tokens) {
  ninfer::LayoutBuilder layout;
  (void)layout.add_tensor(ninfer::DType::BF16, {static_cast<std::int32_t>(hidden), tokens},
                          kVisionWorkspaceAlignment, "vision item output transient");
  return layout.finish(kVisionWorkspaceAlignment, "vision item output transient layout");
}

// The load options `ignis_model_load` and `ignis_model_plan_reservations`
// both run under, once validated.
struct LoadOptions {
  ninfer::DType cache_dtype = ninfer::DType::BF16;
  int32_t speculative_backend = IGNIS_SPECULATIVE_NONE;
  uint32_t draft_tokens = 0;
  uint32_t vision_max_tokens = 0;
  // GitHub #243: the embedding pool's bytes, already floored at the
  // envelope's own output by `validate_and_bind`.
  uint64_t vision_embedding_pool_bytes = 0;
  // GitHub #227: the text rotary table's scaling; the default is no scaling,
  // the linear table.
  ignis::RopeScaling rope_scaling{};
};

// GitHub #243: one embedding pool page, in bytes -- the width every side of
// the leaf agrees on (kernel/include/ignis_step.h).
std::size_t vision_embedding_page_bytes(std::int64_t hidden) {
  return static_cast<std::size_t>(hidden) * IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS *
         sizeof(std::uint16_t);
}

// The pool's bytes, rounded up to whole pages and floored at the envelope's
// own output: an item that fits the envelope must always fit the pool once
// everything else is released, or a caller told to "release and retry" would
// never make progress.
std::size_t vision_embedding_pool_bytes(std::int64_t hidden, std::int32_t envelope_tokens,
                                        std::uint64_t requested) {
  const std::size_t page = vision_embedding_page_bytes(hidden);
  const std::size_t floor_bytes = vision_output_transient_bytes(hidden, envelope_tokens);
  const std::size_t want = std::max<std::size_t>(static_cast<std::size_t>(requested), floor_bytes);
  return ((want + page - 1) / page) * page;
}

// The argument checks and the tensor binding of a load, shared by the load
// itself and by its plan (GitHub #210) so the two refuse the same calls with
// the same messages. Allocates nothing on the device. Null (error set) on
// any refusal.
std::unique_ptr<ignis_model> validate_and_bind(const struct ignis_bound_tensor *tensors,
                                               uint64_t count, const struct ignis_topology *topology,
                                               uint32_t prefill_chunk_tokens,
                                               uint32_t max_context_tokens, int32_t kv_format,
                                               const struct ignis_model_load_options *options,
                                               LoadOptions &out) {
  if (topology->num_layers > 0 && topology->layer_kinds == nullptr) {
    set_error("ignis_model_load: topology.layer_kinds is null");
    return nullptr;
  }
  if (topology->gdn_num_layers == 0) {
    set_error("ignis_model_load: topology.gdn_num_layers must be positive");
    return nullptr;
  }
  if (topology->gdn_state_rows % topology->gdn_num_layers != 0) {
    set_error("ignis_model_load: gdn_state_rows is not a multiple of gdn_num_layers");
    return nullptr;
  }
  // P2-01 (GitHub #83): the prefill chunk width the caller will hand
  // ignis_program_prefill, validated against the reference's own alignment
  // rule (also the alignment the GDN chunked kernels' 64-token chunk
  // divides evenly).
  if (prefill_chunk_tokens == 0 || prefill_chunk_tokens % 128 != 0) {
    set_error("ignis_model_load: prefill_chunk_tokens must be a nonzero multiple of 128");
    return nullptr;
  }
  if (max_context_tokens == 0) {
    set_error("ignis_model_load: max_context_tokens must be positive");
    return nullptr;
  }
  // A chunk wider than the sequence pool's own context bound can never be
  // prefilled anyway, and the GQA attention workspace query needs
  // max_visible_keys >= the query width it is sized for.
  if (prefill_chunk_tokens > max_context_tokens) {
    set_error("ignis_model_load: prefill_chunk_tokens (" + std::to_string(prefill_chunk_tokens) +
              ") must not exceed max_context_tokens (" + std::to_string(max_context_tokens) + ")");
    return nullptr;
  }
  // P4-05 (GitHub #123): the format both scratch reservations below are
  // sized for. Refused here rather than defaulted to BF16 -- an unrecognized
  // value would otherwise reserve one format's arena and let the layers run
  // the other's routes out of it.
  if (kv_format != IGNIS_KV_FORMAT_BF16 && kv_format != IGNIS_KV_FORMAT_HQ_E8_2B) {
    set_error("ignis_model_load: kv_format " + std::to_string(kv_format) +
              " is not an ignis_kv_format");
    return nullptr;
  }
  out.cache_dtype = ignis_kv_cache_dtype(kv_format);

  // P5-02 (GitHub #150, ADR 0016): the load options, validated before any
  // binding -- a NULL pointer is the production default, no speculation.
  int32_t speculative_backend = IGNIS_SPECULATIVE_NONE;
  uint32_t draft_tokens = 0;
  uint32_t vision_max_tokens = 0;
  uint64_t vision_embedding_pool_request = 0;
  ignis::RopeScaling rope_scaling{};
  if (options != nullptr) {
    if (options->size != sizeof(struct ignis_model_load_options)) {
      set_error("ignis_model_load: options.size " + std::to_string(options->size) +
                " is not a recognized ignis_model_load_options size");
      return nullptr;
    }
    speculative_backend = options->speculative_backend;
    draft_tokens = options->draft_tokens;
    vision_max_tokens = options->vision_max_tokens;
    vision_embedding_pool_request = options->vision_embedding_pool_bytes;
    rope_scaling.factor = options->rope_scaling_factor;
    rope_scaling.temperature = options->rope_scaling_temperature;
    rope_scaling.beta_fast = options->rope_scaling_beta_fast;
    rope_scaling.beta_slow = options->rope_scaling_beta_slow;
  }
  // GitHub #227: a scaling that cannot build a table is refused by name --
  // the alternative is a load that silently rotates at a different one.
  if (const std::string rejection = ignis::rope_scaling_rejection(rope_scaling);
      !rejection.empty()) {
    set_error("ignis_model_load: " + rejection);
    return nullptr;
  }
  if (vision_max_tokens > IGNIS_VISION_MAX_TOKENS_LIMIT) {
    set_error("ignis_model_load: vision_max_tokens " + std::to_string(vision_max_tokens) +
              " exceeds " + std::to_string(IGNIS_VISION_MAX_TOKENS_LIMIT));
    return nullptr;
  }
  // GitHub #243: a pool asked for without vision is a caller that thinks it
  // configured something. The floor is applied where the geometry is known;
  // here only the contradiction is refused.
  if (vision_max_tokens == 0 && vision_embedding_pool_request != 0) {
    set_error("ignis_model_load: vision_embedding_pool_bytes " +
              std::to_string(vision_embedding_pool_request) + " needs vision_max_tokens");
    return nullptr;
  }
  if (speculative_backend != IGNIS_SPECULATIVE_NONE &&
      speculative_backend != IGNIS_SPECULATIVE_DFLASH2 &&
      speculative_backend != IGNIS_SPECULATIVE_VERIFY_ONLY) {
    set_error("ignis_model_load: speculative_backend " + std::to_string(speculative_backend) +
              " is not an ignis_speculative_backend");
    return nullptr;
  }
  if (speculative_backend == IGNIS_SPECULATIVE_NONE && draft_tokens != 0) {
    set_error("ignis_model_load: draft_tokens " + std::to_string(draft_tokens) +
              " needs a speculative_backend");
    return nullptr;
  }
  // P5-04 (GitHub #153): VERIFY_ONLY takes the same window rule -- the
  // verify graphs are captured at it, and the round is refused at any other.
  if (speculative_backend != IGNIS_SPECULATIVE_NONE &&
      (draft_tokens < 1 || draft_tokens > IGNIS_DFLASH2_MAX_DRAFT_TOKENS)) {
    set_error("ignis_model_load: draft_tokens " + std::to_string(draft_tokens) +
              " must be in 1.." + std::to_string(IGNIS_DFLASH2_MAX_DRAFT_TOKENS));
    return nullptr;
  }
  // GitHub #195: vision and a speculative backend are two independent load
  // options. #178's fence stood until the verify round learned the sequence's
  // `rope_delta` (`IgnisVerifyRound::rope_positions`, staged per round); the
  // drafter's context append needed nothing, because it consumes the span's
  // KV positions, which is what the reference's own prefill sink captures on
  // a multimodal span too.
  out.speculative_backend = speculative_backend;
  out.draft_tokens = draft_tokens;
  out.vision_max_tokens = vision_max_tokens;
  out.vision_embedding_pool_bytes = vision_embedding_pool_request;
  out.rope_scaling = rope_scaling;

  ModelBinder binder(tensors, count);
  if (!binder.build_index(count)) {
    return nullptr;
  }

  const Geometry g = Geometry::from(*topology);
  auto model = std::make_unique<ignis_model>();
  // GitHub #227: the table the text layers will rotate at, resolved once
  // here -- the linear one without scaling, the YaRN one with it.
  model->text_rope = ignis::text_rope_frequencies(rope_scaling);

  if (!binder.bind("text/token_embedding", {g.vocab, g.hidden}, model->token_embedding) ||
      !binder.bind("text/final_norm", {g.hidden}, model->final_norm) ||
      !binder.bind("text/output_head", {g.vocab, g.hidden}, model->output_head)) {
    return nullptr;
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
      return nullptr;
    }
  }

  // Without the option the `dflash2/*` tensors are not asked for, so a caller
  // that hands them over anyway fails on `require_no_extras` below.
  if (speculative_backend == IGNIS_SPECULATIVE_DFLASH2 && !bind_dflash2(binder, g, model->dflash2)) {
    return nullptr;
  }
  // GitHub #177: likewise the `vision/*` tensors, only with an envelope.
  if (vision_max_tokens > 0 && !bind_vision(binder, g, model->vision)) {
    return nullptr;
  }

  if (!binder.require_no_extras()) {
    return nullptr;
  }
  return model;
}

// The drafter round's vendored workspaces (`swa`, `linear_swiglu`), checked
// against their fixed allowance; see kDflash2RoundWorkspaceBytes.
void check_dflash2_round_workspace(const ignis_model &model, const Geometry &g,
                                   uint32_t max_context_tokens, uint32_t draft_tokens) {
  const auto block = static_cast<std::int32_t>(draft_tokens + 1);
  const std::size_t workspace_bytes =
      ninfer::ops::swa_workspace_capacity_bytes({0, max_context_tokens}, 1, block,
                                                IGNIS_DECODE_MAX_BATCH) +
      ninfer::ops::linear_swiglu_workspace_capacity_bytes(
          model.dflash2.layers[0].mlp_gate_up.qtype, static_cast<std::int32_t>(2 * g.ffn_intermediate),
          static_cast<std::int32_t>(g.hidden), ninfer::ops::LinearPolicy::A16Only, 1, 16);
  if (workspace_bytes > kDflash2RoundWorkspaceBytes) {
    throw std::runtime_error("the drafter's vendored workspaces need " +
                             std::to_string(workspace_bytes) + " bytes, past the " +
                             std::to_string(kDflash2RoundWorkspaceBytes) + "-byte allowance");
  }
}

// The arenas and buffers a load reserves beside the weights, sized from a
// bound model (GitHub #210): the one computation both `ignis_model_load`,
// which allocates from it, and `ignis_model_plan_reservations`, which only
// reports it, run. Throws on a sizing failure.
struct LoadSizes {
  std::size_t prefill_scratch = 0;
  // GitHub #260 (ADR 0038): the attention readout's scores, which a prefill
  // chunk takes from the same arena beside `prefill_scratch`. Kept apart from
  // it because only a vision load has any: what vision adds to the arena is
  // counted as vision's (`vision_reserved_bytes`), readout included.
  std::size_t attention_readout = 0;
  std::size_t vision_workspace = 0;
  std::size_t media_embedding = 0;
  std::size_t sampling_workspace = 0;
  std::size_t decode_graph_scratch = 0;
  std::size_t verify_round = 0;
  std::size_t drafter_features = 0;
  std::size_t drafter_scratch = 0;
  std::size_t sampling_logits = 0;

  // GitHub #212: the one arena prefill chunks and media encode share. A
  // media encode runs between prefill steps, never inside one, so the two
  // are never live at once -- ninfer sizes its single workspace the same
  // way.
  std::size_t workspace() const {
    return std::max(prefill_scratch + attention_readout, vision_workspace);
  }

  ignis_model_reservations reservations() const {
    const std::size_t lanes = IGNIS_DECODE_MAX_BATCH;
    const bool vision = media_embedding > 0;
    ignis_model_reservations out{};
    out.workspace_bytes = workspace();
    out.media_embedding_bytes = media_embedding;
    // sampling_single_{configs, positions, out}, sampling_decode_{configs,
    // positions, out, logits} and the workspace, as the load allocates them.
    // GitHub #242 adds the permitted-set staging to the same line: ids
    // [lanes][cap], counts [lanes] and the committed probabilities [lanes].
    out.sampling_bytes = sizeof(ninfer::ops::SamplingConfig) + 2 * sizeof(int32_t) +
                         sizeof(ninfer::ops::SamplingConfig) * lanes + 2 * sizeof(int32_t) * lanes +
                         sizeof(int32_t) * lanes * IGNIS_MAX_PERMITTED_TOKENS +
                         sizeof(int32_t) * lanes + sizeof(float) * lanes +
                         sampling_logits + sampling_workspace;
    // decode_graph_{scratch, token_ids, slots}, and decode_rope_positions
    // with vision.
    out.decode_graph_bytes =
        decode_graph_scratch + (vision ? 3 : 2) * sizeof(int32_t) * lanes;
    out.verify_round_bytes = verify_round;
    out.drafter_round_bytes =
        drafter_scratch == 0 ? 0 : drafter_features + sizeof(std::int32_t) * lanes + drafter_scratch;
    return out;
  }
};

LoadSizes plan_load_sizes(const ignis_model &model, const ignis_topology &topology,
                          uint32_t prefill_chunk_tokens, uint32_t max_context_tokens,
                          const LoadOptions &options) {
  const Geometry g = Geometry::from(topology);
  LoadSizes sizes;

  // P2-01 (GitHub #83): the scratch, sized for a `prefill_chunk_tokens`-wide
  // chunk (see compute_program_scratch_bytes above).
  sizes.prefill_scratch = compute_program_scratch_bytes(
      model, topology, static_cast<std::int32_t>(prefill_chunk_tokens), max_context_tokens,
      options.cache_dtype, /*batch=*/1);
  if (options.speculative_backend == IGNIS_SPECULATIVE_DFLASH2) {
    sizes.prefill_scratch +=
        dflash2_prefill_scratch_bytes(topology, static_cast<std::int32_t>(prefill_chunk_tokens));
  }

  // GitHub #177: the encoder workspace for the envelope's merged tokens
  // (capped by the context), and GitHub #243 the embedding pool.
  if (options.vision_max_tokens > 0) {
    const auto tokens =
        static_cast<std::int32_t>(std::min(options.vision_max_tokens, max_context_tokens));
    // GitHub #260 (ADR 0038): a head point's attention readout scores one
    // image's placeholder span from the prefill chunk's own scope -- one F32
    // per key, and an image holds at most the envelope's tokens. Only a
    // vision load can be asked one (the span is an image's), so a text load
    // reserves nothing for it.
    //
    // GitHub #263 (ADR 0039): and beside them the head set's results, one
    // packed (score, key) per head, reserved for the largest set a readout
    // may name -- under 4 KB.
    sizes.attention_readout =
        fp32_bytes(tokens) +
        round_up_arena_align(static_cast<std::size_t>(kReadoutMaxSetHeads) * sizeof(unsigned long long));
    sizes.vision_workspace =
        ignis_vision_workspace_bytes(tokens, std::min(tokens, kVisionMaxSegments));
    sizes.media_embedding =
        vision_embedding_pool_bytes(g.hidden, tokens, options.vision_embedding_pool_bytes);
  }

  // P3-03 (GitHub #99): device sampling's workspace and its decode logits.
  const auto vocab = static_cast<std::int32_t>(g.vocab);
  sizes.sampling_logits =
      static_cast<std::size_t>(vocab) * IGNIS_DECODE_MAX_BATCH * sizeof(std::uint16_t);
  sizes.sampling_workspace = std::max<std::size_t>(
      ninfer::ops::sampling_workspace_capacity_bytes(vocab, 1, IGNIS_DECODE_MAX_BATCH), kArenaAlign);

  // P3-05 (GitHub #102) / P5-04 (GitHub #153): the round scratch, one column
  // per lane, or `k+1` with a draft window.
  const bool windowed = options.draft_tokens > 0;
  const auto round_columns = static_cast<std::int32_t>(windowed ? options.draft_tokens + 1 : 1);
  sizes.decode_graph_scratch = compute_program_scratch_bytes(
      model, topology, /*chunk=*/round_columns, max_context_tokens, options.cache_dtype,
      /*batch=*/IGNIS_DECODE_MAX_BATCH, /*head_every_column=*/windowed);

  if (windowed) {
    sizes.verify_round = verify_round_bytes(topology, options.draft_tokens, options.vision_max_tokens > 0);
  }

  // P5-05 (GitHub #155): the drafter's round buffers.
  if (options.speculative_backend == IGNIS_SPECULATIVE_DFLASH2) {
    check_dflash2_round_workspace(model, g, max_context_tokens, options.draft_tokens);
    const auto block = static_cast<std::size_t>(options.draft_tokens + 1);
    sizes.drafter_features = kDflash2TapLayers.size() * static_cast<std::size_t>(g.hidden) * block *
                             IGNIS_DECODE_MAX_BATCH * sizeof(std::uint16_t);
    sizes.drafter_scratch =
        dflash2_round_activation_bytes(topology, static_cast<std::int32_t>(options.draft_tokens)) +
        kDflash2RoundWorkspaceBytes;
  }
  return sizes;
}

// GitHub #210: what a loaded model holds beside its weights, read off its own
// buffers -- the other half of `LoadSizes::reservations`, which the caller
// compares against its plan.
ignis_model_reservations reserved_of(const ignis_model &model) {
  const auto bytes = [](const std::unique_ptr<ninfer::DeviceBuffer> &buffer) -> uint64_t {
    return buffer != nullptr ? buffer->bytes : 0;
  };
  const auto capacity = [](const std::unique_ptr<ninfer::DeviceArena> &arena) -> uint64_t {
    return arena != nullptr ? arena->capacity() : 0;
  };
  ignis_model_reservations out{};
  out.workspace_bytes = capacity(model.scratch);
  out.media_embedding_bytes = bytes(model.vision_pool.buffer);
  out.sampling_bytes = bytes(model.sampling_single_configs) + bytes(model.sampling_single_positions) +
                       bytes(model.sampling_single_out) + bytes(model.sampling_decode_configs) +
                       bytes(model.sampling_decode_positions) + bytes(model.sampling_decode_out) +
                       bytes(model.sampling_decode_logits) +
                       bytes(model.sampling_decode_permitted) +
                       bytes(model.sampling_decode_permitted_counts) +
                       bytes(model.sampling_decode_permitted_probs) +
                       capacity(model.sampling_workspace);
  out.decode_graph_bytes = capacity(model.decode_graph_scratch) + bytes(model.decode_graph_token_ids) +
                           bytes(model.decode_graph_slots) + bytes(model.decode_rope_positions);
  if (model.verify != nullptr) {
    out.drafter_round_bytes = bytes(model.verify->features) + bytes(model.verify->append_counts) +
                              capacity(model.verify->drafter_scratch);
    out.verify_round_bytes = model.verify->device_bytes() - out.drafter_round_bytes;
  }
  return out;
}

} // namespace

extern "C" int32_t ignis_model_plan_reservations(const struct ignis_bound_tensor *tensors,
                                                 uint64_t count,
                                                 const struct ignis_topology *topology,
                                                 uint32_t prefill_chunk_tokens,
                                                 uint32_t max_context_tokens, int32_t kv_format,
                                                 const struct ignis_model_load_options *options,
                                                 struct ignis_model_reservations *out) {
  if (out != nullptr) {
    *out = ignis_model_reservations{};
  }
  if (tensors == nullptr || topology == nullptr || out == nullptr) {
    set_error("ignis_model_plan_reservations: null argument");
    return -1;
  }
  LoadOptions load{};
  const auto model = validate_and_bind(tensors, count, topology, prefill_chunk_tokens,
                                       max_context_tokens, kv_format, options, load);
  if (model == nullptr) {
    return -1;
  }
  try {
    *out = plan_load_sizes(*model, *topology, prefill_chunk_tokens, max_context_tokens, load)
               .reservations();
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_plan_reservations: ") + e.what());
    return -1;
  }
  return 0;
}

extern "C" int32_t ignis_model_load(const struct ignis_bound_tensor *tensors, uint64_t count,
                                     const struct ignis_topology *topology,
                                     uint32_t prefill_chunk_tokens, uint32_t max_context_tokens,
                                     int32_t kv_format,
                                     const struct ignis_model_load_options *options,
                                     struct ignis_model **out_model) {
  if (out_model != nullptr) {
    *out_model = nullptr;
  }
  if (tensors == nullptr || topology == nullptr || out_model == nullptr) {
    set_error("ignis_model_load: null argument");
    return -1;
  }
  LoadOptions load{};
  auto model = validate_and_bind(tensors, count, topology, prefill_chunk_tokens,
                                 max_context_tokens, kv_format, options, load);
  if (model == nullptr) {
    return -1;
  }
  const Geometry g = Geometry::from(*topology);
  const int32_t speculative_backend = load.speculative_backend;
  const uint32_t draft_tokens = load.draft_tokens;
  const uint32_t vision_max_tokens = load.vision_max_tokens;
  model->speculative_backend = speculative_backend;
  model->draft_tokens = draft_tokens;

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
  model->prefill_chunk_tokens = prefill_chunk_tokens;
  // P3-05 (GitHub #102, ADR 0019): kept for the decode graphs' fixed,
  // conservative GqaExecutionEnvelope (every capture and every replay uses
  // this cap, never a lane's actual current position).
  model->max_context_tokens = max_context_tokens;
  // P4-05 (GitHub #123): what the two arenas below are reserved for, and
  // what `kernel/src/gqa_layer.cu` checks a sequence pool against.
  model->kv_format = kv_format;

  // GitHub #210: every reservation below is sized here, once, by the same
  // computation `ignis_model_plan_reservations` reports.
  LoadSizes sizes;
  try {
    sizes = plan_load_sizes(*model, *topology, prefill_chunk_tokens, max_context_tokens, load);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: reservation sizing failed: ") + e.what());
    return -1;
  }

  const cudaError_t stream_err = cudaStreamCreate(&model->stream);
  if (stream_err != cudaSuccess) {
    set_error(std::string("ignis_model_load: cudaStreamCreate failed: ") +
              cudaGetErrorString(stream_err));
    return -1;
  }

  // P2-01 (GitHub #83): reserve the scratch once, sized for a
  // `prefill_chunk_tokens`-wide chunk (see compute_program_scratch_bytes
  // above) -- never at the first long prompt. GitHub #212: with vision, the
  // same arena is the encoder's workspace, so it is sized for the larger of
  // the two. A reservation that does not fit the device's free memory fails
  // the load right here, with a message naming the shortfall and whichever
  // of the two sized it.
  const std::size_t scratch_bytes = sizes.workspace();
  const auto vision_tokens = std::min(vision_max_tokens, max_context_tokens);
  const bool encoder_sized = sizes.vision_workspace > sizes.prefill_scratch;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (mem_err == cudaSuccess && scratch_bytes > free_bytes) {
    const std::string sized_by =
        encoder_sized ? "the vision encoder's workspace for a " + std::to_string(vision_tokens) +
                            "-token envelope"
                      : "a " + std::to_string(prefill_chunk_tokens) + "-token prefill chunk";
    set_error("ignis_model_load: " + sized_by + " needs a " + std::to_string(scratch_bytes) +
              "-byte scratch reservation, but only " + std::to_string(free_bytes) +
              " bytes are free -- " +
              (encoder_sized ? "lower --vision-max-tokens" : "pick a smaller prefill chunk width"));
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  try {
    model->scratch = std::make_unique<ninfer::DeviceArena>(scratch_bytes);
    model->prefill_scratch_bytes = sizes.prefill_scratch;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: scratch arena allocation failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  // GitHub #177: the vision reservation, taken here at load -- before the
  // caller builds its sequence pool -- so enabling vision can never OOM a
  // later request: GitHub #243's embedding pool, beside the scratch above
  // that already fits the encoder's workspace (GitHub #212). A reservation
  // that does not fit the free memory fails the load naming it.
  if (vision_max_tokens > 0) {
    model->vision_max_tokens = vision_max_tokens;
    try {
      const std::size_t output_bytes = sizes.media_embedding;
      std::size_t vision_free = 0;
      std::size_t vision_total = 0;
      if (cudaMemGetInfo(&vision_free, &vision_total) == cudaSuccess && output_bytes > vision_free) {
        throw std::runtime_error("a " + std::to_string(vision_tokens) +
                                 "-token vision envelope's embedding pool needs " +
                                 std::to_string(output_bytes) + " bytes, but only " +
                                 std::to_string(vision_free) +
                                 " are free -- lower --vision-embedding-pool or"
                                 " --vision-max-tokens");
      }
      const std::size_t page_bytes = vision_embedding_page_bytes(g.hidden);
      model->vision_pool.buffer = std::make_unique<ninfer::DeviceBuffer>(output_bytes);
      model->vision_pool.page_columns = IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS;
      model->vision_pool.page_bytes = page_bytes;
      model->vision_pool.owner.assign(output_bytes / page_bytes, nullptr);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: vision reservation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }

  // P3-03 (GitHub #99): device-side sampling's stable staging buffers and its
  // own transient candidate-selection workspace, sized once at load. See
  // model_internal.h's field comments for why these are separate from
  // `scratch` above.
  try {
    model->sampling_single_configs =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(ninfer::ops::SamplingConfig));
    model->sampling_single_positions = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t));
    model->sampling_single_out = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t));
    model->sampling_decode_configs = std::make_unique<ninfer::DeviceBuffer>(
        sizeof(ninfer::ops::SamplingConfig) * IGNIS_DECODE_MAX_BATCH);
    model->sampling_decode_positions =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    // GitHub #178: allocated before the decode graphs are captured, which
    // bake its address into every GQA layer's rotation.
    if (vision_max_tokens > 0) {
      model->decode_rope_positions =
          std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    }
    model->sampling_decode_out =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    model->sampling_decode_logits = std::make_unique<ninfer::DeviceBuffer>(sizes.sampling_logits);
    // GitHub #242: three small per-lane rows, allocated whether or not any
    // request ever constrains a draw -- 4 KB against a plan in gigabytes,
    // and an allocation that depends on traffic is one the VRAM plan cannot
    // state at load (ADR 0030).
    model->sampling_decode_permitted = std::make_unique<ninfer::DeviceBuffer>(
        sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH * IGNIS_MAX_PERMITTED_TOKENS);
    model->sampling_decode_permitted_counts =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    model->sampling_decode_permitted_probs =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(float) * IGNIS_DECODE_MAX_BATCH);
    model->sampling_workspace = std::make_unique<ninfer::DeviceArena>(sizes.sampling_workspace);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: sampling buffer allocation failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  // P3-05 (GitHub #102, ADR 0019) / GitHub #111: the decode rounds'
  // own scratch and staging buffers, reserved once here and never touched by
  // prefill -- so a chunk running between two replays can never alias what a
  // replay rereads. Sized for the widest round the leaf admits: one token
  // per lane (`chunk=1`) across `IGNIS_DECODE_MAX_BATCH` lanes, since #111
  // made a round one batch-wide traversal instead of a per-lane loop, and
  // every width 1..IGNIS_DECODE_MAX_BATCH shares this one reservation.
  // Capture itself (`ignis_decode_graph_capture`) happens later, once the
  // sequence pool exists.
  //
  // P5-04 (GitHub #153): a load with a draft window sizes the same arena for
  // the verify round instead -- `k+1` columns per lane, the head over every
  // column -- which bounds today's one-column round too, so both rounds
  // share it.
  try {
    model->decode_graph_scratch = std::make_unique<ninfer::DeviceArena>(sizes.decode_graph_scratch);
    model->decode_graph_token_ids =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    model->decode_graph_slots =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: decode graph buffer allocation failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  // P5-04 (GitHub #153): the verify round's substrate, for either windowed
  // backend.
  if (draft_tokens > 0) {
    try {
      model->verify = build_verify_round(*topology, draft_tokens, vision_max_tokens > 0);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: verify round allocation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }

  // P5-05 (GitHub #155): the drafter's round buffers (model_internal.h), on
  // a load with the drafter only. `plan_load_sizes` has already checked the
  // vendored workspaces against their allowance.
  if (speculative_backend == IGNIS_SPECULATIVE_DFLASH2) {
    try {
      const auto lanes = static_cast<std::size_t>(IGNIS_DECODE_MAX_BATCH);
      model->verify->features = std::make_unique<ninfer::DeviceBuffer>(sizes.drafter_features);
      model->verify->append_counts = std::make_unique<ninfer::DeviceBuffer>(lanes * sizeof(std::int32_t));
      model->verify->drafter_scratch = std::make_unique<ninfer::DeviceArena>(sizes.drafter_scratch);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: drafter round allocation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
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
  out_stats->vision_reserved_bytes = model->vision_reserved_bytes();
  out_stats->reserved = reserved_of(*model);
  return 0;
}

extern "C" void ignis_model_free(struct ignis_model *model) {
  if (model != nullptr) {
    // P3-05 (GitHub #102, ADR 0019): destroy every captured decode graph
    // before the stream/scratch it was captured against goes away.
    for (auto &exec : model->decode_graph_exec) {
      if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
      }
    }
    if (model->verify != nullptr) {
      for (auto &exec : model->verify->graph_exec) {
        if (exec != nullptr) {
          cudaGraphExecDestroy(exec);
          exec = nullptr;
        }
      }
    }
    if (model->stream != nullptr) {
      cudaStreamDestroy(model->stream);
    }
  }
  delete model;
}

extern "C" const char *ignis_model_last_error(void) {
  return g_last_error.c_str();
}
