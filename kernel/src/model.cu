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
#include "ignis_fp8_linear.h"
#include "ignis_residency.h"
#include "ignis_image_gemm.h"
#include "ignis_step.h"

#include "attention_readout.h"
#include "draft_window.h"
#include "extended_attention.h"
#include "fp8_exception.h"
#include "ignis_dense_prefill.h"
#include "gdn_prefill.h"
#include "hq_decode_attention.h"
#include "layer_internal.h"
#include "model_internal.h"
#include "mtp_head.h"
#include "sampled_draft.h"
#include "trellis_body.h"
#include "w4a4_activation_layout.h"
#include "w4a4_tma_route.h"
#include "w8_imma.h"

#include "ignis_dflash2_topk.h"

#include "flash_next/bind.h"
#include "flash_next/program.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/weight.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ignis_gqa_workspace.h"
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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
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

// ADR 0062: whether `data` is page-locked host memory -- a mapped table the
// kernels read in place over the bus -- rather than device memory. A pointer
// the runtime does not know (none of the leaf's own) reads as not host.
bool in_host_memory(const void *data) {
  if (data == nullptr) {
    return false;
  }
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, data) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeHost;
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
    // GitHub #339: a trellis code is the leaf's own, past ninfer::QType -- it
    // never becomes a ninfer::Weight.
    if (ignis::trellis_body::is_trellis_qtype(t.qtype)) {
      set_error("ignis_model_load: " + name + " is a trellis tensor where the schema binds a stored " +
                "weight: only a trellis body's linears and head are trellis (text/output_head decides " +
                "the body), and a load binds one body, never a mix");
      return false;
    }
    if (!shape_matches(t, want_shape)) {
      set_error("ignis_model_load: " + name + " has an unexpected shape (want " +
                shape_str(want_shape) + ")");
      return false;
    }
    used_[it->second] = true;
    out = to_weight(t);
    return true;
  }

  // GitHub #339 (spec 27b-16gb/05a): bind a trellis matrix of the 16 GB
  // artifact, `[out, in]` in its stored (Hugging Face) orientation, into the
  // op's weight. The record is `[tiles | suh fp16[in] | svh fp16[out]]`, and
  // its `qhigh` / `scales` are where its `suh` / `svh` sit -- checked against
  // the tiles' own length when the descriptor carries addresses (a plan
  // carries none).
  bool bind_trellis(const std::string &name, std::initializer_list<int64_t> want_shape,
                    ignis_trellis_weight &out) {
    auto it = index_.find(name);
    if (it == index_.end()) {
      set_error("ignis_model_load: missing bound tensor: " + name);
      return false;
    }
    const ignis_bound_tensor &t = tensors_[it->second];
    if (!ignis::trellis_body::is_trellis_qtype(t.qtype) || t.layout != IGNIS_LAYOUT_TRELLIS_TILE16) {
      set_error("ignis_model_load: " + name + " is qtype " + std::to_string(t.qtype) + ", layout " +
                std::to_string(t.layout) + ", not a trellis tensor, but the body is trellis " +
                "(text/output_head is): a load binds one body, never a mix");
      return false;
    }
    if (!shape_matches(t, want_shape)) {
      set_error("ignis_model_load: " + name + " has an unexpected shape (want " +
                shape_str(want_shape) + ")");
      return false;
    }
    ignis_trellis_weight w{};
    w.k2 = ignis::trellis_body::k2_of(t.qtype);
    w.out = static_cast<uint32_t>(t.shape[0]);
    w.in = static_cast<uint32_t>(t.shape[1]);
    w.tile_stride = 0;
    w.trellis = t.qdata;
    w.suh = t.qhigh;
    w.svh = t.scales;
    const uint64_t tiles = ignis::trellis_body::tile_bytes(w);
    if (t.bytes < tiles + 2ull * (w.in + w.out)) {
      set_error("ignis_model_load: " + name + "'s " + std::to_string(t.bytes) +
                " bytes cannot hold its tiles and scales");
      return false;
    }
    const auto *base = static_cast<const std::uint8_t *>(t.qdata);
    if (t.qdata != nullptr && (t.qhigh != base + tiles || t.scales != base + tiles + 2ull * w.in)) {
      set_error("ignis_model_load: " + name + "'s suh and svh are not where its record puts them");
      return false;
    }
    used_[it->second] = true;
    out = w;
    return true;
  }

  bool has(const std::string &name) const { return index_.find(name) != index_.end(); }

  // GitHub #339: whether `name` is bound, in a trellis code.
  bool is_trellis(const std::string &name) const {
    auto it = index_.find(name);
    return it != index_.end() && ignis::trellis_body::is_trellis_qtype(tensors_[it->second].qtype);
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
  // The GDN output gate's width: `gdn/in_proj_z`'s rows on the trellis body.
  int64_t gdn_z_width;
  int64_t gdn_ab_width;
  int64_t gdn_state_rows;
  int64_t gdn_value_heads;
  // The GDN head dim: one value head's width, and so the gated RMSNorm's.
  int64_t gdn_head_dim;
  int64_t gdn_conv_kernel;

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
    g.gdn_z_width = static_cast<int64_t>(t.gdn_z_width);
    g.gdn_ab_width = static_cast<int64_t>(t.gdn_ab_width);
    g.gdn_state_rows = static_cast<int64_t>(t.gdn_state_rows);
    g.gdn_value_heads = static_cast<int64_t>(t.gdn_value_heads);
    g.gdn_head_dim = static_cast<int64_t>(t.gdn_head_dim);
    g.gdn_conv_kernel = static_cast<int64_t>(t.gdn_conv_kernel);
    return g;
  }
};

// The GDN geometry the vendored ops run (GitHub #302): the causal conv at
// kernel width 4 (its state pool keeps 3 taps, kIgnisGdnConvStateWidth) and
// the recurrence at head dim 128. Both are topology fields; a topology that
// asks for another value is refused at load, not run against these.
constexpr int64_t kGdnConvKernel = 4;
constexpr int64_t kGdnHeadDim = 128;

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
         binder.bind(prefix + "gdn/a_log", {g.gdn_value_heads}, w.a_log) &&
         binder.bind(prefix + "gdn/dt_bias", {g.gdn_value_heads}, w.dt_bias) &&
         binder.bind(prefix + "gdn/convolution", {g.gdn_conv_kernel, g.gdn_conv_channels},
                     w.convolution) &&
         binder.bind(prefix + "gdn/a_b_projection", {g.gdn_ab_width, g.hidden}, w.a_b_projection) &&
         binder.bind(prefix + "gdn/query_key_value_z", {g.gdn_in_proj_m, g.hidden},
                     w.query_key_value_z) &&
         binder.bind(prefix + "gdn/norm", {g.gdn_head_dim}, w.norm) &&
         binder.bind(prefix + "gdn/output", {g.hidden, g.gdn_state_rows}, w.output) &&
         binder.bind(prefix + "post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind(prefix + "mlp/gate_up", {2 * g.ffn_intermediate, g.hidden}, w.mlp_gate_up) &&
         binder.bind(prefix + "mlp/down", {g.hidden, g.ffn_intermediate}, w.mlp_down);
}

// GitHub #339 (spec 27b-16gb/05a): the trellis body's layers -- the 16 GB
// artifact's names and row orders (spec 27b-16gb/01's As built): the fused
// planes' split parts, each its own matrix with its own scales and K, and
// every other tensor the served artifact's. `attention/q_proj` is 24 heads of
// [q | gate] interleaved per head; `gdn/in_proj_qkv` is the conv input's
// [q | k | v] channels.
bool bind_gqa_layer_trellis(ModelBinder &binder, const std::string &prefix, const Geometry &g,
                            GqaLayerWeights &w) {
  GqaTrellisWeights &t = w.trellis;
  return binder.bind(prefix + "input_norm", {g.hidden}, w.input_norm) &&
         binder.bind_trellis(prefix + "attention/q_proj", {2 * g.gqa_width, g.hidden}, t.q_proj) &&
         binder.bind_trellis(prefix + "attention/k_proj", {g.gqa_kv_width, g.hidden}, t.k_proj) &&
         binder.bind_trellis(prefix + "attention/v_proj", {g.gqa_kv_width, g.hidden}, t.v_proj) &&
         binder.bind(prefix + "attention/query_norm", {g.head_dim}, w.query_norm) &&
         binder.bind(prefix + "attention/key_norm", {g.head_dim}, w.key_norm) &&
         binder.bind_trellis(prefix + "attention/output", {g.hidden, g.gqa_width}, t.tail.output) &&
         binder.bind(prefix + "post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind_trellis(prefix + "mlp/gate_proj", {g.ffn_intermediate, g.hidden}, t.tail.gate_proj) &&
         binder.bind_trellis(prefix + "mlp/up_proj", {g.ffn_intermediate, g.hidden}, t.tail.up_proj) &&
         binder.bind_trellis(prefix + "mlp/down", {g.hidden, g.ffn_intermediate}, t.tail.down);
}

bool bind_gdn_layer_trellis(ModelBinder &binder, const std::string &prefix, const Geometry &g,
                            GdnLayerWeights &w) {
  GdnTrellisWeights &t = w.trellis;
  return binder.bind(prefix + "input_norm", {g.hidden}, w.input_norm) &&
         binder.bind(prefix + "gdn/a_log", {g.gdn_value_heads}, w.a_log) &&
         binder.bind(prefix + "gdn/dt_bias", {g.gdn_value_heads}, w.dt_bias) &&
         binder.bind(prefix + "gdn/convolution", {g.gdn_conv_kernel, g.gdn_conv_channels},
                     w.convolution) &&
         binder.bind(prefix + "gdn/a_b_projection", {g.gdn_ab_width, g.hidden}, w.a_b_projection) &&
         binder.bind_trellis(prefix + "gdn/in_proj_qkv", {g.gdn_conv_channels, g.hidden}, t.in_proj_qkv) &&
         binder.bind_trellis(prefix + "gdn/in_proj_z", {g.gdn_z_width, g.hidden}, t.in_proj_z) &&
         binder.bind(prefix + "gdn/norm", {g.gdn_head_dim}, w.norm) &&
         binder.bind_trellis(prefix + "gdn/output", {g.hidden, g.gdn_state_rows}, t.tail.output) &&
         binder.bind(prefix + "post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind_trellis(prefix + "mlp/gate_proj", {g.ffn_intermediate, g.hidden}, t.tail.gate_proj) &&
         binder.bind_trellis(prefix + "mlp/up_proj", {g.ffn_intermediate, g.hidden}, t.tail.up_proj) &&
         binder.bind_trellis(prefix + "mlp/down", {g.hidden, g.ffn_intermediate}, t.tail.down);
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
// The shortlist proposal head's rows (`text/draft_head`): the 131,072 most
// frequent tokens, each mapped back by `text/draft_head_token_ids`.
constexpr int64_t kProposalHeadRows = 131072;
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

// GitHub #620 (spec #618): the 27B's MTP head, the container's 12 `mtp/`
// objects: its matrices W8 and its gains BF16 on every 27B artifact, the 16
// GB ones included (the head is never trellis), at the target's widths.
bool bind_mtp_head(ModelBinder &binder, const Geometry &g, MtpHeadWeights &w) {
  return binder.bind("mtp/input_projection", {g.hidden, 2 * g.hidden}, w.input_projection) &&
         binder.bind("mtp/embedding_norm", {g.hidden}, w.embedding_norm) &&
         binder.bind("mtp/hidden_norm", {g.hidden}, w.hidden_norm) &&
         binder.bind("mtp/layer/input_norm", {g.hidden}, w.input_norm) &&
         binder.bind("mtp/layer/attention/query_key_gate_value", {2 * g.gqa_width + 2 * g.gqa_kv_width, g.hidden},
                     w.query_key_gate_value) &&
         binder.bind("mtp/layer/attention/query_norm", {g.head_dim}, w.query_norm) &&
         binder.bind("mtp/layer/attention/key_norm", {g.head_dim}, w.key_norm) &&
         binder.bind("mtp/layer/attention/output", {g.hidden, g.gqa_width}, w.output) &&
         binder.bind("mtp/layer/post_attention_norm", {g.hidden}, w.post_attention_norm) &&
         binder.bind("mtp/layer/mlp/gate_up", {2 * g.ffn_intermediate, g.hidden}, w.mlp_gate_up) &&
         binder.bind("mtp/layer/mlp/down", {g.hidden, g.ffn_intermediate}, w.mlp_down) &&
         binder.bind("mtp/final_norm", {g.hidden}, w.final_norm);
}

// The head runs on the W8 linear at its five registered geometries, so a
// matrix the load hands over in another format would reach no route; refused
// by name at load rather than at its first prefill.
bool check_mtp_head_formats(const MtpHeadWeights &w) {
  for (const auto *matrix : {&w.input_projection, &w.query_key_gate_value, &w.output, &w.mlp_gate_up, &w.mlp_down}) {
    if (matrix->qtype != ninfer::QType::W8G32_F16S) {
      set_error("ignis_model_load: the MTP head's matrices are W8G32 on every 27B artifact; one is qtype " +
                std::to_string(static_cast<int>(matrix->qtype)));
      return false;
    }
  }
  return true;
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

// Which of the load's prefill routes an arena's layers run (the review of spec
// #503): the prefill chunk's arena runs every one the load chose, a decode
// round's none -- its calls are a round's columns at most, which take the fused
// op's decode route, and its GDN layers the vendored conv and recurrence forms.
struct LayerRoutes {
  // GitHub #507: the GDN prefill's conv writes q, k and v split, so the layer
  // allocates no [conv_channels, T] plane.
  bool split_conv = false;
  // GitHub #509: the dense prefill, which takes a call wider than
  // IGNIS_TRELLIS_DECODE_MAX_TOKENS and writes the query and the gate split.
  bool dense = false;
  // GitHub #514: the GDN prefill on our kernels (`gdn_prefill_route`), from
  // ignis::gdn_prefill::kMinTokens tokens, with their own workspace.
  bool gdn_ours = false;
};

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
// the hq (U8) prompt widths take ignis's prompt attention (GitHub #335),
// whose band and split states BF16 has no counterpart of. Sizing the arena
// as BF16 and then dispatching hq is the one way this reservation can be
// wrong, so the format reaches here rather than being assumed.
//
// GitHub #339: on the trellis body (`trellis`) the projections take no
// vendored workspace -- the trellis op's is its own plan line -- and the
// query projection lands in a `[2 q_width, columns]` buffer before its heads'
// halves move into `query` and `gate` (kernel/src/trellis_body.h). Under the
// dense prefill (`routes.dense`, GitHub #509) only a call the fused op takes
// does, IGNIS_TRELLIS_DECODE_MAX_TOKENS columns at most.
std::size_t gqa_layer_scratch_bytes(const ignis_topology &topology, const GqaLayerWeights &w,
                                     std::int32_t T, uint32_t max_context_tokens,
                                     ninfer::DType cache_dtype, std::int32_t batch, bool trellis,
                                     LayerRoutes routes) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto q_width = static_cast<std::int32_t>(topology.num_q_heads * topology.head_dim);
  const auto kv_width = static_cast<std::int32_t>(topology.num_kv_heads * topology.head_dim);
  const auto ffn = static_cast<std::int32_t>(topology.ffn_intermediate);
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
  bytes += round_up_arena_align(ignis_gqa_attention_capacity_bytes(
      cache_dtype, envelope, /*batch_size=*/batch, /*min_width=*/1, /*max_width=*/T));
  if (trellis) {
    const std::int64_t fused_columns =
        routes.dense ? std::min<std::int64_t>(columns, IGNIS_TRELLIS_DECODE_MAX_TOKENS) : columns;
    return bytes + bf16_bytes(2 * static_cast<int64_t>(q_width) * fused_columns);  // interleaved q | gate
  }
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
// one either way. GitHub #339: on the trellis body (`trellis`) the
// projections take no vendored workspace, and the activations are the same
// buffers -- the two input projections write `qkv` and `zbuf` themselves.
// The prefill's `routes` (the review of spec #503): its conv writes q, k and v
// split, so there is no `qkv_conv` plane (GitHub #507), and on our GDN
// prefill a call of ignis::gdn_prefill::kMinTokens or more takes our
// recurrence's workspace, a narrower one the vendored op's (GitHub #514).
std::size_t gdn_layer_scratch_bytes(const ignis_topology &topology, const GdnLayerWeights &w,
                                     std::int32_t T, std::int32_t batch, bool trellis, LayerRoutes routes) {
  const auto hidden = static_cast<std::int32_t>(topology.hidden);
  const auto ffn = static_cast<std::int32_t>(topology.ffn_intermediate);
  const auto state_rows = static_cast<std::int32_t>(topology.gdn_state_rows);
  const auto state_cols = static_cast<std::int32_t>(topology.gdn_state_cols);
  const auto q_width = static_cast<std::int32_t>(topology.gdn_q_width);
  const auto conv_channels = q_width + state_cols + state_rows;
  const auto value_width = state_rows;
  const auto qk_width = (q_width + state_cols) / 2;
  const auto value_heads = static_cast<std::int32_t>(topology.gdn_value_heads);
  const auto qk_heads = qk_width / static_cast<std::int32_t>(topology.gdn_head_dim);
  const std::int32_t columns = T * batch;

  std::size_t bytes = 0;
  bytes += bf16_bytes(static_cast<int64_t>(hidden) * columns);        // h
  bytes += bf16_bytes(static_cast<int64_t>(conv_channels) * columns); // qkv
  if (!routes.split_conv) {
    bytes += bf16_bytes(static_cast<int64_t>(conv_channels) * columns); // qkv_conv
  }
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

  bytes += round_up_arena_align(
      ninfer::ops::gdn_gating_proj_workspace_capacity_bytes(value_heads, hidden, 1, columns));
  if (routes.gdn_ours && columns >= ignis::gdn_prefill::kMinTokens) {
    // Ours grows with the call's tokens, so the chunk's bounds every call it
    // takes; below kMinTokens the vendored op runs, at most that wide.
    bytes += std::max(round_up_arena_align(ignis::gdn_prefill::workspace_bytes(qk_heads, value_heads, columns)),
                      round_up_arena_align(ninfer::ops::gated_delta_net_workspace_capacity_bytes(
                          qk_heads, value_heads, /*normalize_qk=*/true, 1, ignis::gdn_prefill::kMinTokens - 1)));
  } else {
    bytes += round_up_arena_align(ninfer::ops::gated_delta_net_workspace_capacity_bytes(
        qk_heads, value_heads, /*normalize_qk=*/true, 1, columns));
  }
  if (trellis) {
    return bytes;
  }
  bytes += round_up_arena_align(ninfer::ops::gdn_input_proj_workspace_capacity_bytes(
      w.query_key_value_z.qtype, w.query_key_value_z.n, w.query_key_value_z.k,
      ignis_widest_linear_policy_for(w.query_key_value_z.qtype), 1, columns));
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
// ever live alongside the outer scope's). `routes`: the prefill routes the
// arena's layers run, none for a decode round's.
std::size_t compute_program_scratch_bytes(const ignis_model &model, const ignis_topology &topology,
                                          std::int32_t chunk, uint32_t max_context_tokens,
                                          ninfer::DType cache_dtype, std::int32_t batch,
                                          bool head_every_column, LayerRoutes routes) {
  std::size_t layer_peak = 0;
  for (const auto &layer : model.layers) {
    const std::size_t layer_bytes = layer.kind == IGNIS_LAYER_GQA
        ? gqa_layer_scratch_bytes(topology, layer.gqa, chunk, max_context_tokens, cache_dtype, batch,
                                  model.is_trellis_body, routes)
        : gdn_layer_scratch_bytes(topology, layer.gdn, chunk, batch, model.is_trellis_body, routes);
    layer_peak = std::max(layer_peak, layer_bytes);
  }
  return program_outer_scratch_bytes(topology, chunk, batch, head_every_column) + layer_peak;
}

// ---------------------------------------------------------------------------
// The ReplaySSM record geometry of the verify round at window `k`, one
// record row per lane of the load's `lanes` (GitHub #336).
ninfer::GdnReplayRecordSpec verify_record_spec(const ignis_topology &topology, uint32_t window,
                                               uint32_t lanes) {
  std::int32_t gdn_layers = 0;
  for (uint32_t i = 0; i < topology.num_layers; ++i) {
    if (static_cast<ignis_layer_kind>(topology.layer_kinds[i]) == IGNIS_LAYER_GDN) {
      ++gdn_layers;
    }
  }
  ninfer::GdnReplayRecordSpec spec{};
  spec.layers = gdn_layers;
  spec.record_capacity = static_cast<std::int32_t>(lanes);
  spec.width = static_cast<std::int32_t>(window) + 1;
  spec.conv_channels =
      static_cast<std::int32_t>(topology.gdn_q_width + topology.gdn_state_cols + topology.gdn_state_rows);
  // A topology without GDN layers carries no head dim: it records nothing.
  spec.qk_heads = topology.gdn_head_dim == 0
                      ? 0
                      : static_cast<std::int32_t>(topology.gdn_q_width / topology.gdn_head_dim);
  spec.value_heads = static_cast<std::int32_t>(topology.gdn_value_heads);
  spec.key_dim = static_cast<std::int32_t>(topology.gdn_head_dim);
  spec.value_dim = static_cast<std::int32_t>(topology.gdn_head_dim);
  return spec;
}

// GitHub #210: every byte `build_verify_round` below allocates, without
// allocating -- the VRAM plan's verify line. `IgnisVerifyRound::device_bytes`
// reads the same total back off a built round, and the load checks the two
// agree. GitHub #336: for the load's `decode_lanes`, IGNIS_DECODE_MAX_BATCH before.
//
// GitHub #687: by buffer, the buffers of one extent under one field, because
// a load's two round shapes hold one set of buffers, each the larger of the
// two shapes' (`larger_of`). At 8 lanes both shapes top out at 64 columns (8
// x 8 and 4 x 16), so the set is the base shape's -- the line of a load at 7
// drafts -- but for the drafts: 15 x 4 words against 7 x 8.
struct VerifyRoundSizes {
  // anchors, base_positions, extents, valid_columns, lengths,
  // licensed_counts, accepted, selectors: I32 [B] each
  std::size_t lane_words = 0;
  std::size_t drafts = 0; // I32 [k, B]
  // verify_ids, positions, target_tokens, licensed_tokens, and
  // rope_positions with `vision`: I32 [k+1, B] each
  std::size_t column_words = 0;
  std::size_t logits = 0;          // BF16 [vocab, k+1, B]
  std::size_t hidden = 0;          // BF16 [hidden, k+1, B]
  std::size_t selected_hidden = 0; // BF16 [hidden, B]
  std::size_t records = 0;         // the ReplaySSM records' backing
  std::size_t accept = 0;          // the accept workspace
  bool vision = false;

  std::size_t total() const {
    return 8 * lane_words + drafts + (vision ? 5 : 4) * column_words + logits + hidden + selected_hidden + records +
           accept;
  }
};

VerifyRoundSizes verify_round_sizes(const ignis_topology &topology, uint32_t window, bool vision,
                                    uint32_t decode_lanes) {
  const auto k = static_cast<std::size_t>(window);
  const std::size_t columns = k + 1;
  const std::size_t lanes = decode_lanes;
  const auto hidden = static_cast<std::size_t>(topology.hidden);
  const auto vocab = static_cast<std::size_t>(topology.vocab);
  const std::size_t i32 = sizeof(int32_t);
  const std::size_t bf16 = sizeof(std::uint16_t);

  VerifyRoundSizes sizes;
  sizes.vision = vision;
  sizes.lane_words = lanes * i32;
  sizes.drafts = k * lanes * i32;
  sizes.column_words = columns * lanes * i32;
  sizes.logits = vocab * columns * lanes * bf16;
  sizes.hidden = hidden * columns * lanes * bf16;
  sizes.selected_hidden = hidden * lanes * bf16;
  ninfer::LayoutBuilder builder;
  (void)ninfer::plan_gdn_replay_records(builder, verify_record_spec(topology, window, decode_lanes));
  sizes.records = builder.finish(kArenaAlign, "GDN replay records");
  sizes.accept = std::max<std::size_t>(ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                           static_cast<std::int32_t>(vocab), static_cast<std::int32_t>(k),
                                           static_cast<std::int32_t>(k), 1, static_cast<std::int32_t>(lanes)),
                                       kArenaAlign);
  return sizes;
}

// GitHub #687: each buffer at the larger of two round shapes'.
VerifyRoundSizes larger_of(const VerifyRoundSizes &a, const VerifyRoundSizes &b) {
  VerifyRoundSizes sizes;
  sizes.lane_words = std::max(a.lane_words, b.lane_words);
  sizes.drafts = std::max(a.drafts, b.drafts);
  sizes.column_words = std::max(a.column_words, b.column_words);
  sizes.logits = std::max(a.logits, b.logits);
  sizes.hidden = std::max(a.hidden, b.hidden);
  sizes.selected_hidden = std::max(a.selected_hidden, b.selected_hidden);
  sizes.records = std::max(a.records, b.records);
  sizes.accept = std::max(a.accept, b.accept);
  sizes.vision = a.vision || b.vision;
  return sizes;
}

// The ReplaySSM records of `verify`'s window across its lanes, laid out by
// the vendored planner over its backing. GitHub #687: a backing two round
// shapes share is the larger layout's, so each shape binds its own planes
// over as many of its bytes as its layout takes.
void bind_verify_records(IgnisVerifyRound &verify, const ignis_topology &topology) {
  ninfer::LayoutBuilder builder;
  verify.records_layout =
      ninfer::plan_gdn_replay_records(builder, verify_record_spec(topology, verify.window, verify.lanes));
  const std::size_t record_bytes = builder.finish(kArenaAlign, "GDN replay records");
  if (record_bytes > verify.records_backing->bytes) {
    throw std::runtime_error("the verify round's records outgrow their backing");
  }
  verify.records =
      ninfer::GdnReplayRecords(ninfer::DeviceSpan{verify.records_backing->p, record_bytes}, verify.records_layout);
}

// P5-04 (GitHub #153): the verify round's substrate (model_internal.h's
// `IgnisVerifyRound`) at window `k`, for the load's `decode_lanes` lanes
// (GitHub #336; IGNIS_DECODE_MAX_BATCH before). The
// ReplaySSM record geometry is the model's GDN geometry -- the layer count
// from the topology's kinds, the head counts from the state widths -- and
// the vendored planner (`plan_gdn_replay_records`) lays the four planes out;
// the fold only admits its registered all-layer geometries, so a wrong
// count here fails at the first round, by name, not silently.
//
// GitHub #195: a vision load's `sizes` add the round's rope-position staging,
// so a multimodal sequence's verify columns rotate at `position + rope_delta`
// the way its decode rounds do. A load without vision allocates nothing for
// it and every round rotates at `positions` itself, exactly as before.
//
// GitHub #687: every buffer is allocated at `sizes`, the round's own
// (`verify_round_sizes` at this window and these lanes) or, on a load of two
// round shapes, the larger of the two shapes'.
std::unique_ptr<IgnisVerifyRound> build_verify_round(const ignis_topology &topology, uint32_t window,
                                                     uint32_t decode_lanes, const VerifyRoundSizes &sizes) {
  auto verify = std::make_unique<IgnisVerifyRound>();
  verify->window = window;
  verify->lanes = decode_lanes;
  const auto lanes = static_cast<std::int32_t>(decode_lanes);

  const auto device = [](std::size_t bytes) { return std::make_shared<ninfer::DeviceBuffer>(bytes); };
  verify->anchors = device(sizes.lane_words);
  verify->drafts = device(sizes.drafts);
  verify->base_positions = device(sizes.lane_words);
  verify->extents = device(sizes.lane_words);
  verify->valid_columns = device(sizes.lane_words);
  verify->lengths = device(sizes.lane_words);
  verify->verify_ids = device(sizes.column_words);
  verify->positions = device(sizes.column_words);
  verify->target_tokens = device(sizes.column_words);
  verify->logits = device(sizes.logits);
  verify->hidden = device(sizes.hidden);
  verify->licensed_tokens = device(sizes.column_words);
  verify->licensed_counts = device(sizes.lane_words);
  verify->accepted = device(sizes.lane_words);
  verify->selectors = device(sizes.lane_words);
  verify->selected_hidden = device(sizes.selected_hidden);
  if (sizes.vision) {
    verify->rope_positions = device(sizes.column_words);
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

  verify->records_backing = device(sizes.records);
  bind_verify_records(*verify, topology);

  verify->accept_workspace = std::make_shared<ninfer::DeviceArena>(sizes.accept);
  return verify;
}

// GitHub #687: a load's extended round shape at window `window` across
// `lanes`, over `base`'s device memory: every buffer and arena is the base
// shape's own, which the load sized for the larger of the two shapes
// (`plan_load_sizes`), the drafter's included, and the ReplaySSM records take
// the extended shape's layout over the same backing. Nothing here is
// allocated on the device; the host-mapped staging and the graphs are the
// shape's own.
std::unique_ptr<IgnisVerifyRound> build_extended_shape(const ignis_topology &topology, uint32_t window,
                                                       uint32_t lanes, const IgnisVerifyRound &base) {
  auto verify = std::make_unique<IgnisVerifyRound>();
  verify->window = window;
  verify->lanes = lanes;
  verify->anchors = base.anchors;
  verify->drafts = base.drafts;
  verify->base_positions = base.base_positions;
  verify->extents = base.extents;
  verify->valid_columns = base.valid_columns;
  verify->lengths = base.lengths;
  verify->rope_positions = base.rope_positions;
  verify->verify_ids = base.verify_ids;
  verify->positions = base.positions;
  verify->target_tokens = base.target_tokens;
  verify->logits = base.logits;
  verify->hidden = base.hidden;
  verify->licensed_tokens = base.licensed_tokens;
  verify->licensed_counts = base.licensed_counts;
  verify->accepted = base.accepted;
  verify->selectors = base.selectors;
  verify->selected_hidden = base.selected_hidden;
  verify->records_backing = base.records_backing;
  verify->accept_workspace = base.accept_workspace;
  verify->features = base.features;
  verify->append_counts = base.append_counts;
  verify->drafter_scratch = base.drafter_scratch;
  verify->draft_candidates = base.draft_candidates;
  verify->draft_q = base.draft_q;
  bind_verify_records(*verify, topology);
  verify->sampled_draft = base.sampled_draft;
  verify->drafter_temperature_factor = base.drafter_temperature_factor;
  verify->drafter_top_p = base.drafter_top_p;
  if (verify->sampled_draft &&
      ignis_sampled_accept_workspace_bytes(static_cast<std::int32_t>(topology.vocab), static_cast<std::int32_t>(window),
                                           static_cast<std::int32_t>(lanes)) > verify->accept_workspace->capacity()) {
    throw std::runtime_error("the sampled draft's accept at the extended shape outgrows the accept workspace");
  }
  verify->staging = std::make_unique<VerifyStaging>();
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
// `drafter_scratch` (kernel/src/dflash2_drafter.cu) at window `k` across the
// load's `lanes` (GitHub #336; IGNIS_DECODE_MAX_BATCH before), in the order
// it allocates them: its forward -- both blocks of a layer counted together,
// so the sum is conservative -- or the round's context append, whichever is
// larger.
//
// GitHub #687: the forward in its two phases, for the extended round shape's
// bound (`dflash2_extended_round_scratch_bytes`): the query block's ids,
// positions and residual with one layer's blocks, then the draft columns'
// head, top-k and selector, which the forward allocates once every layer's
// blocks are released (their scopes close first, dflash2_drafter.cu).
struct Dflash2RoundActivations {
  std::size_t query = 0;  // ids, positions, residual: live through the forward
  std::size_t layers = 0; // one layer's attention and MLP blocks together
  std::size_t head = 0;   // packed .. scores
  std::size_t append = 0; // the round's context append
};

Dflash2RoundActivations dflash2_round_activations(const ignis_topology &topology, std::int32_t window,
                                                  uint32_t lanes) {
  const auto hidden = static_cast<int64_t>(topology.hidden);
  const auto vocab = static_cast<int64_t>(topology.vocab);
  const auto intermediate = static_cast<int64_t>(topology.ffn_intermediate);
  const int64_t columns = static_cast<int64_t>(window + 1) * lanes;
  const int64_t drafts = static_cast<int64_t>(window) * lanes;
  const int64_t kv_width = kDflash2KvHeads * kDflash2HeadDim;
  const int64_t top_k = kDflash2SelectorTopK;

  std::size_t forward = 0;
  forward += i32_bytes(columns) * 2;                                // ids, positions
  forward += bf16_bytes(hidden * columns);                          // residual
  const std::size_t query = forward;
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
  const std::size_t layers = forward - query;
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
  return Dflash2RoundActivations{query, layers, forward - query - layers, append};
}

// P5-05 (GitHub #155): the drafter's round scratch at window `k` across
// `lanes`: the forward's activations summed, or the append's, with the
// allowance for the vendored workspaces (kDflash2RoundWorkspaceBytes).
std::size_t dflash2_round_scratch_bytes(const ignis_topology &topology, std::int32_t window, uint32_t lanes) {
  const Dflash2RoundActivations bytes = dflash2_round_activations(topology, window, lanes);
  return std::max(bytes.query + bytes.layers + bytes.head, bytes.append) + kDflash2RoundWorkspaceBytes;
}

// GitHub #687: what an extended round asks of the scratch the two round
// shapes share, by the forward's phases: while a layer runs, the query block,
// its blocks and the vendored workspaces; after the last one, the query block
// and the head's buffers, where the layers' were. The extended forward's head
// is larger than the base one's at 8 lanes (4 x 15 draft columns against 8 x
// 7), yet it fits what the base shape's sum reserves, so the shared scratch
// stays the base shape's at every lane count and window.
std::size_t dflash2_extended_round_scratch_bytes(const ignis_topology &topology, std::int32_t window,
                                                 uint32_t lanes) {
  const Dflash2RoundActivations bytes = dflash2_round_activations(topology, window, lanes);
  return std::max(bytes.query + std::max(bytes.layers + kDflash2RoundWorkspaceBytes, bytes.head),
                  bytes.append + kDflash2RoundWorkspaceBytes);
}

// GitHub #576: what the sampled draft keeps of the drafter's q from the draw
// to the accept at window `k` across `lanes`: each draft column's candidate
// ids (I32) and their probabilities (F32), two buffers of
// `[kDflash2SelectorTopK, k, lanes]`.
std::size_t dflash2_draw_bytes(uint32_t window, uint32_t lanes) {
  return 2 * static_cast<std::size_t>(kDflash2SelectorTopK) * window * lanes * sizeof(std::int32_t);
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

// GitHub #558 (ADR 0087): the expert staging ring of the residency a load
// borrows (ignis_model_load_options::residency) -- 0 without one, and when it
// holds every expert and stages nothing -- read before the options are
// validated, so only from options of a recognized size.
uint64_t lent_staging_ring_bytes(const struct ignis_model_load_options *options) {
  if (options == nullptr || options->size != sizeof(struct ignis_model_load_options) ||
      options->residency == nullptr) {
    return 0;
  }
  ignis_residency_layout layout{};
  return ignis_residency_get_layout(options->residency, &layout) == 0 ? layout.ring_bytes : 0;
}

// GitHub #557, #558 (ADR 0087): the most merged tokens one media item may
// hold when its encode runs in what a Flash-Next load lends it, idle while
// the item encodes -- `scratch_bytes` of its prefill scratch and its expert
// staging ring of `ring_bytes` (0 when the residency holds every expert) --
// the encoder's workspace for the item (ignis_vision_workspace_bytes, at the
// segments the encode checks) and with a staged tower (`staging_slot_bytes`
// nonzero) its slots, laid out as ignis_vision_encode_placement lays them, at
// most `cap`; 0 when not even a one-token item fits. The share grows with the
// item, so the largest that fits is found by bisection.
uint32_t vision_item_bound(std::size_t scratch_bytes, std::size_t ring_bytes, std::size_t staging_slot_bytes,
                           uint32_t cap) {
  const auto fits_in = [&](uint32_t tokens) {
    const auto t = static_cast<std::int32_t>(tokens);
    const std::size_t workspace = ignis_vision_workspace_bytes(t, std::min(t, kVisionMaxSegments));
    return ignis_vision_encode_placement(scratch_bytes, ring_bytes, workspace,
                                         IGNIS_VISION_STAGING_SLOTS * staging_slot_bytes)
        .fits;
  };
  uint32_t fits = 0;
  uint32_t past = cap + 1;
  while (past - fits > 1) {
    const uint32_t mid = fits + (past - fits) / 2;
    if (fits_in(mid)) {
      fits = mid;
    } else {
      past = mid;
    }
  }
  return fits;
}

// The load options `ignis_model_load` and `ignis_model_plan_reservations`
// both run under, once validated.
struct LoadOptions {
  ninfer::DType cache_dtype = ninfer::DType::BF16;
  int32_t speculative_backend = IGNIS_SPECULATIVE_NONE;
  uint32_t draft_tokens = 0;
  uint32_t vision_max_tokens = 0;
  // The most merged tokens one item may hold, already resolved against the
  // envelope by `validate_and_bind` (never above it; the envelope itself
  // when the caller named no bound). 0 without vision.
  uint32_t vision_item_max_tokens = 0;
  // GitHub #243: the embedding pool's bytes, already floored at the
  // envelope's own output by `validate_and_bind`.
  uint64_t vision_embedding_pool_bytes = 0;
  // GitHub #227: the text rotary table's scaling; the default is no scaling,
  // the linear table.
  ignis::RopeScaling rope_scaling{};
  // GitHub #275: the most keys of a text span an attention readout reads.
  uint32_t attention_text_max_keys = 0;
  // GitHub #335: the prompt attention's precision (enum ignis_prefill_attention).
  uint32_t prefill_attention = IGNIS_PREFILL_ATTENTION_FP8;
  // GitHub #509: the trellis body's prefill route (enum ignis_prefill_linear).
  uint32_t prefill_linear = IGNIS_PREFILL_LINEAR_FUSED;
  // GitHub #336: the 27B's decode lanes, resolved (0 is IGNIS_DECODE_MAX_BATCH):
  // the widest round, and every round reservation's width.
  uint32_t decode_lanes = IGNIS_DECODE_MAX_BATCH;
  // GitHub #414 (ADR 0075): where the vision tower lives (enum
  // ignis_vision_residency); HOST only with vision.
  uint32_t vision_residency = IGNIS_VISION_RESIDENCY_DEVICE;
  // GitHub #514: the GDN prefill's route (`gdn_prefill_route`).
  bool gdn_prefill_ours = false;
  // GitHub #572 (spec #570): what the measurement switch IGNIS_HQ_DECODE_ATTENTION names; the
  // load decides the route from it and the card (`hq_decode_attention_route`).
  ignis::hq_decode_attention::Switch hq_decode_attention = ignis::hq_decode_attention::Switch::Unset;
  // GitHub #627 (spec #618): the MTP head's matrices on ours (`mtp_head_linear_route`), which an
  // MTP load runs.
  bool mtp_head_linear_ours = false;
  // GitHub #650: whether the load keeps its W4A4 TMA products' descriptors on the device
  // (`tma_descriptor_cache_route`): on Windows, on a card the route runs on, unless the
  // measurement switch IGNIS_TMA_DESCRIPTOR_CACHE=0 names upstream's per-launch upload.
  bool tma_descriptor_cache = false;
  // GitHub #651, #653 (spec #646): the route of the prefill's MLP gate and up product and its
  // epilogue, what the measurement switch IGNIS_PREFILL_SWIGLU names
  // (`ignis::prefill_swiglu::route_for_load`).
  ignis::prefill_swiglu::Route prefill_swiglu = ignis::prefill_swiglu::Route::Quant;
  // GitHub #652 (spec #646): what the measurement switch IGNIS_W4A4_ACTIVATION_LAYOUT names; the
  // load decides from it, the body, the chunk and the card the layout its W4A4 prefill products
  // read (`w4a4_activation_layout_route`).
  ignis::w4a4_activation_layout::Layout w4a4_activation_layout = ignis::w4a4_activation_layout::Layout::KTile;
  // GitHub #576 (ADR 0090): the verify round's draft rule (enum
  // ignis_draft_rule) and, under the sampled draft, the drafter's
  // temperature factor and top-p.
  uint32_t draft_rule = IGNIS_DRAFT_RULE_FIXED;
  float drafter_temperature_factor = 0.0f;
  float drafter_top_p = 1.0f;
};

// The current device's compute capability major, which a load's route choices read: false (error
// set, naming `what` the generation was for) when the device cannot say it.
bool device_major(const char *what, int &major) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_model_load: the GPU generation for ") + what + ": " + cudaGetErrorString(err));
    return false;
  }
  return true;
}

// Spec #646: a measurement switch read once per load by `read`, which throws
// std::invalid_argument naming the variable on a value it does not know, so that value refuses
// the plan and the load by name. False (error set) on a refusal.
template <class Read, class Value> bool load_switch(Read read, Value &named) {
  try {
    named = read();
  } catch (const std::invalid_argument &e) {
    set_error(std::string("ignis_model_load: ") + e.what());
    return false;
  }
  return true;
}

// GitHub #514: the GDN prefill on our full-rate kernels (kernel/src/gdn_prefill.h)
// or on the vendored chunked ones, decided once per load: the card's generation
// decides (ours on 8.x and on Blackwell, `ignis::gdn_prefill::runs_by_default`)
// unless the measurement switch IGNIS_GDN_PREFILL names a route
// (`ignis::gdn_prefill::route_for_load`);
// any other value is refused by name. False (error set) on a refusal or when the
// device cannot say its generation.
bool gdn_prefill_route(bool &ours) {
  int major = 0;
  if (!device_major("the GDN prefill", major)) {
    return false;
  }
  try {
    ours = ignis::gdn_prefill::route_for_load(major);
  } catch (const std::invalid_argument &e) {
    set_error(std::string("ignis_model_load: ") + e.what());
    return false;
  }
  return true;
}

// GitHub #627 (spec #618): the MTP head's matrices on ours (kernel/src/w8_imma.h) or on the
// vendored `ops::linear`, decided once per load: ours on every card below compute capability 12,
// the vendored routes on Blackwell unless the measurement switch IGNIS_MTP_HEAD_LINEAR names ours
// (`ignis::w8_imma::route_for_load`); any other value is refused by name. False (error set) on a
// refusal or when the device cannot say its generation.
bool mtp_head_linear_route(bool &ours) {
  int major = 0;
  if (!device_major("the MTP head's matrices", major)) {
    return false;
  }
  try {
    ours = ignis::w8_imma::route_for_load(major);
  } catch (const std::invalid_argument &e) {
    set_error(std::string("ignis_model_load: ") + e.what());
    return false;
  }
  return true;
}

// GitHub #650: whether the load keeps the W4A4 TMA products' descriptor blocks in device slots
// (kernel/src/tma_descriptor_cache.h), decided once per load: where the launchers take them
// through a pointer (Windows), on a card the products' sm_120a code runs on (compute capability
// 12), unless the measurement switch IGNIS_TMA_DESCRIPTOR_CACHE=0 names upstream's per-launch
// upload; any other value is refused by name. False (error set) on a refusal or when the device
// cannot say its generation. The card is asked only where the slots could be planned, so a Linux
// load makes no call it did not make before.
bool tma_descriptor_cache_route(bool &cache) {
  cache = false;
  bool enabled = false;
  if (!load_switch(ignis::tma_descriptors::enabled_for_load, enabled)) {
    return false;
  }
  if (!ignis::tma_descriptors::kThroughPointer || !enabled) {
    return true;
  }
  int major = 0;
  if (!device_major("the TMA descriptor slots", major)) {
    return false;
  }
  cache = ignis::w4a4_tma::runs_on(major);
  return true;
}

// GitHub #652 (spec #646): the layout the load's W4A4 prefill products read their activations in,
// decided once per load: the one the switch names where the prefill runs the W4A4 TMA route
// (kernel/src/w4a4_tma_route.h) -- the NVFP4 body, chunks of at least 1,024 tokens, a card the
// products' sm_120a code runs on (compute capability 12) -- and none elsewhere. False (error set)
// when the device cannot say its generation; the card is asked only where the route could run.
bool w4a4_activation_layout_route(ignis::w4a4_activation_layout::Layout named, bool trellis_body,
                                  uint32_t prefill_chunk_tokens,
                                  std::optional<ignis::w4a4_activation_layout::Layout> &layout) {
  layout.reset();
  if (trellis_body || !ignis::w4a4_tma::reaches(prefill_chunk_tokens)) {
    return true;
  }
  int major = 0;
  if (!device_major("the W4A4 activation layout", major)) {
    return false;
  }
  if (ignis::w4a4_tma::runs_on(major)) {
    layout = named;
  }
  return true;
}

// GitHub #572 (spec #570): the attention route of the 27B's decode and verify rounds, decided once
// per load, after the plan and outside any capture (kernel/src/hq_decode_attention.h): ignis's
// warp-specialized route over hq-e8-2b on a Blackwell card whose image this binary holds, unless
// the switch names today's route; today's (the vendored kernel's) everywhere else. False (error
// set) when the device cannot say its generation. The image probe runs only on a card the route
// runs on, so an RTX 30 or 40 load makes no call it did not make before the route existed.
bool hq_decode_attention_route(ignis::hq_decode_attention::Switch named, int32_t kv_format, bool &ours) {
  ours = false;
  if (kv_format != IGNIS_KV_FORMAT_HQ_E8_2B || named == ignis::hq_decode_attention::Switch::Vendored) {
    return true;
  }
  int major = 0;
  if (!device_major("the decode attention", major)) {
    return false;
  }
  const bool image = ignis::hq_decode_attention::runs_on(major) && ignis::hq_decode_attention::image_here();
  ours = ignis::hq_decode_attention::takes_ours(named, true, major, image);
  return true;
}

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

// GitHub #302: the first Flash-Next block a topology carries, by name, or
// null when it carries none (every field zero).
const char *flash_next_block_in(const ignis_topology &t) {
  const ignis_moe_topology &moe = t.moe;
  if ((moe.num_experts | moe.experts_per_token | moe.expert_intermediate |
       moe.shared_expert_intermediate) != 0) {
    return "MoE";
  }
  if ((t.hyper.streams | t.hyper.rank) != 0) {
    return "hyper-connection";
  }
  const ignis_indexer_topology &indexer = t.indexer;
  if ((indexer.heads | indexer.head_dim | indexer.kv_heads | indexer.compress_ratio |
       indexer.budget) != 0) {
    return "indexer";
  }
  const ignis_ngram_topology &ngram = t.ngram;
  if ((ngram.ngram_size | ngram.heads_per_ngram | ngram.embed_dim | ngram.conv_kernel |
       ngram.layer) != 0) {
    return "n-gram";
  }
  return nullptr;
}

// The argument checks and the tensor binding of a load, shared by the load
// itself and by its plan (GitHub #210) so the two refuse the same calls with
// the same messages. Allocates nothing on the device. Null (error set) on
// any refusal. GitHub #558 (ADR 0087): `staging_ring_bytes` is a Flash-Next
// load's expert staging ring, which its item bound counts on -- the plan's,
// or at load the residency's -- and null where no bound is asked for (a
// reservation plan, whose lines the bound does not move); the 27B's bound is
// its own scratch's, so it reads none.
std::unique_ptr<ignis_model> validate_and_bind(const struct ignis_bound_tensor *tensors,
                                               uint64_t count, const struct ignis_topology *topology,
                                               uint32_t prefill_chunk_tokens,
                                               uint32_t max_context_tokens, int32_t kv_format,
                                               const struct ignis_model_load_options *options,
                                               LoadOptions &out, const uint64_t *staging_ring_bytes) {
  if (topology->num_layers > 0 && topology->layer_kinds == nullptr) {
    set_error("ignis_model_load: topology.layer_kinds is null");
    return nullptr;
  }
  // GitHub #302: the GDN layer count and the GDN value-head count are two
  // numbers (48 and 48 on the 27B, 36 and 48 on Flash-Next). The layer count
  // is the layer kinds'; the value heads size every per-head GDN parameter.
  uint64_t gdn_layers = 0;
  for (uint32_t i = 0; i < topology->num_layers; ++i) {
    const int32_t kind = topology->layer_kinds[i];
    if (kind != IGNIS_LAYER_GDN && kind != IGNIS_LAYER_GQA) {
      set_error("ignis_model_load: layer " + std::to_string(i) + "'s kind " + std::to_string(kind) +
                " is not an ignis_layer_kind");
      return nullptr;
    }
    gdn_layers += kind == IGNIS_LAYER_GDN ? 1 : 0;
  }
  if (topology->gdn_num_layers != gdn_layers) {
    set_error("ignis_model_load: topology.gdn_num_layers (" + std::to_string(topology->gdn_num_layers) +
              ") is not the topology's GDN layer count (" + std::to_string(gdn_layers) + ")");
    return nullptr;
  }
  // ADR 0043: the family names the program.
  if (topology->family != IGNIS_MODEL_FAMILY_QWEN38_27B &&
      topology->family != IGNIS_MODEL_FAMILY_FLASH_NEXT) {
    set_error("ignis_model_load: topology.family " + std::to_string(topology->family) +
              " is not an ignis_model_family");
    return nullptr;
  }
  const bool flash_next = topology->family == IGNIS_MODEL_FAMILY_FLASH_NEXT;
  // The heads every GDN parameter is sized by, checked against the widths
  // the projections are sized by, and the geometry the vendored conv and
  // recurrence run. A topology without GDN layers needs none of it, and
  // Flash-Next's GDN layers are its own program's to check.
  if (gdn_layers > 0 && !flash_next) {
    if (topology->gdn_value_heads == 0) {
      set_error("ignis_model_load: topology.gdn_value_heads must be positive");
      return nullptr;
    }
    if (topology->gdn_head_dim != static_cast<uint64_t>(kGdnHeadDim)) {
      set_error("ignis_model_load: topology.gdn_head_dim " + std::to_string(topology->gdn_head_dim) +
                " is not the " + std::to_string(kGdnHeadDim) + " the GDN recurrence runs");
      return nullptr;
    }
    if (topology->gdn_conv_kernel != static_cast<uint64_t>(kGdnConvKernel)) {
      set_error("ignis_model_load: topology.gdn_conv_kernel " + std::to_string(topology->gdn_conv_kernel) +
                " is not the " + std::to_string(kGdnConvKernel) + " the GDN causal conv runs");
      return nullptr;
    }
    if (topology->gdn_state_rows != topology->gdn_value_heads * topology->gdn_head_dim) {
      set_error("ignis_model_load: gdn_state_rows (" + std::to_string(topology->gdn_state_rows) +
                ") is not gdn_value_heads x gdn_head_dim (" + std::to_string(topology->gdn_value_heads) +
                " x " + std::to_string(topology->gdn_head_dim) + ")");
      return nullptr;
    }
  }
  // Flash-Next's blocks are all zero in a 27B topology.
  if (const char *block = flash_next ? nullptr : flash_next_block_in(*topology); block != nullptr) {
    set_error(std::string("ignis_model_load: a Qwen 3.8-27B topology carries no ") + block +
              " block");
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
  uint32_t vision_item_max_tokens = 0;
  uint64_t vision_embedding_pool_request = 0;
  uint32_t attention_text_max_keys = 0;
  uint32_t prefill_attention = 0;
  uint32_t prefill_linear = IGNIS_PREFILL_LINEAR_FUSED;
  uint32_t vision_residency = IGNIS_VISION_RESIDENCY_DEVICE;
  uint32_t draft_rule = IGNIS_DRAFT_RULE_FIXED;
  float drafter_temperature_factor = 0.0f;
  float drafter_top_p = 1.0f;
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
    vision_item_max_tokens = options->vision_item_max_tokens;
    vision_embedding_pool_request = options->vision_embedding_pool_bytes;
    rope_scaling.factor = options->rope_scaling_factor;
    rope_scaling.temperature = options->rope_scaling_temperature;
    rope_scaling.beta_fast = options->rope_scaling_beta_fast;
    rope_scaling.beta_slow = options->rope_scaling_beta_slow;
    attention_text_max_keys = options->attention_text_max_keys;
    prefill_attention = options->prefill_attention;
    prefill_linear = options->prefill_linear;
    vision_residency = options->vision_residency;
    draft_rule = options->draft_rule;
    drafter_temperature_factor = options->drafter_temperature_factor;
    drafter_top_p = options->drafter_top_p;
  }
  // GitHub #335: a value the leaf does not know is refused rather than run as
  // one of the three.
  if (prefill_attention != IGNIS_PREFILL_ATTENTION_FP8 && prefill_attention != IGNIS_PREFILL_ATTENTION_FP16 &&
      prefill_attention != IGNIS_PREFILL_ATTENTION_INT8) {
    set_error("ignis_model_load: prefill_attention " + std::to_string(prefill_attention) +
              " is not 0 (8-bit), 1 (FP16 accumulate) or 2 (integer)");
    return nullptr;
  }
  // GitHub #505 (spec #503): both routes are named by the option surface. The
  // integer band runs (GitHub #510) and so does the dense prefill (GitHub
  // #509): an NVFP4 body refuses the dense one below, once its head says which
  // body it is.
  if (prefill_linear != IGNIS_PREFILL_LINEAR_FUSED && prefill_linear != IGNIS_PREFILL_LINEAR_DENSE) {
    set_error("ignis_model_load: prefill_linear " + std::to_string(prefill_linear) +
              " is not 0 (fused) or 1 (dense)");
    return nullptr;
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
  // The same contradiction for the item bound: it bounds items of an
  // envelope, and without one there are none.
  if (vision_max_tokens == 0 && vision_item_max_tokens != 0) {
    set_error("ignis_model_load: vision_item_max_tokens " + std::to_string(vision_item_max_tokens) +
              " needs vision_max_tokens");
    return nullptr;
  }
  // GitHub #414: a residency the leaf does not know is refused rather than run
  // as one of the two, and a host tower without a tower is the same
  // contradiction as the pool's.
  if (vision_residency != IGNIS_VISION_RESIDENCY_DEVICE && vision_residency != IGNIS_VISION_RESIDENCY_HOST) {
    set_error("ignis_model_load: vision_residency " + std::to_string(vision_residency) +
              " is not 0 (device) or 1 (host)");
    return nullptr;
  }
  if (vision_max_tokens == 0 && vision_residency != IGNIS_VISION_RESIDENCY_DEVICE) {
    set_error("ignis_model_load: vision_residency " + std::to_string(vision_residency) + " needs vision_max_tokens");
    return nullptr;
  }
  if (speculative_backend != IGNIS_SPECULATIVE_NONE &&
      speculative_backend != IGNIS_SPECULATIVE_DFLASH2 &&
      speculative_backend != IGNIS_SPECULATIVE_VERIFY_ONLY && speculative_backend != IGNIS_SPECULATIVE_MTP) {
    set_error("ignis_model_load: speculative_backend " + std::to_string(speculative_backend) +
              " is not an ignis_speculative_backend");
    return nullptr;
  }
  if (speculative_backend == IGNIS_SPECULATIVE_NONE && draft_tokens != 0) {
    set_error("ignis_model_load: draft_tokens " + std::to_string(draft_tokens) +
              " needs a speculative_backend");
    return nullptr;
  }
  // P5-04 (GitHub #153): VERIFY_ONLY takes a window too -- the verify graphs
  // are captured at it, and the round is refused at any other. GitHub #685:
  // each backend's range is its own, and the refusal names it.
  if (speculative_backend != IGNIS_SPECULATIVE_NONE &&
      (draft_tokens < 1 || draft_tokens > ignis_max_draft_tokens(speculative_backend))) {
    set_error("ignis_model_load: draft_tokens " + std::to_string(draft_tokens) + " must be in 1.." +
              std::to_string(ignis_max_draft_tokens(speculative_backend)) + " under speculative_backend " +
              std::to_string(speculative_backend) + " (" + ignis_speculative_backend_name(speculative_backend) +
              ")");
    return nullptr;
  }
  // GitHub #576 (ADR 0090): the draft rule decides what a verify round's
  // sampling lanes draft and accept, so it needs a draft window; the
  // sampled draft's two drafter settings are read only under it, and a value
  // the draw cannot use is refused rather than run as some other one.
  if (draft_rule != IGNIS_DRAFT_RULE_FIXED && draft_rule != IGNIS_DRAFT_RULE_SAMPLED) {
    set_error("ignis_model_load: draft_rule " + std::to_string(draft_rule) + " is not 0 (fixed) or 1 (sampled)");
    return nullptr;
  }
  if (draft_rule == IGNIS_DRAFT_RULE_SAMPLED) {
    if (draft_tokens == 0) {
      set_error("ignis_model_load: draft_rule 1 (sampled) needs a draft window");
      return nullptr;
    }
    if (!std::isfinite(drafter_temperature_factor) || drafter_temperature_factor < 0.0f) {
      set_error("ignis_model_load: drafter_temperature_factor " + std::to_string(drafter_temperature_factor) +
                " is not a finite factor of at least 0");
      return nullptr;
    }
    if (!std::isfinite(drafter_top_p) || drafter_top_p < 0.0f || drafter_top_p > 1.0f) {
      set_error("ignis_model_load: drafter_top_p " + std::to_string(drafter_top_p) + " is not in 0..1");
      return nullptr;
    }
    if (!ignis_sampled_accept_serves(static_cast<std::int32_t>(topology->vocab),
                                     static_cast<std::int32_t>(draft_tokens))) {
      set_error("ignis_model_load: draft_rule 1 (sampled) builds the target's distribution on the sampler's "
                "multi-block route, which a vocabulary of " + std::to_string(topology->vocab) + " does not take");
      return nullptr;
    }
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
  // An item never holds more than its request may: no bound, or one past
  // the envelope, is the envelope.
  out.vision_item_max_tokens = vision_item_max_tokens == 0
                                   ? vision_max_tokens
                                   : std::min(vision_item_max_tokens, vision_max_tokens);
  out.vision_embedding_pool_bytes = vision_embedding_pool_request;
  out.rope_scaling = rope_scaling;
  out.attention_text_max_keys = std::min(attention_text_max_keys, max_context_tokens);
  out.prefill_attention = prefill_attention;
  out.vision_residency = vision_residency;
  out.prefill_linear = prefill_linear;
  out.draft_rule = draft_rule;
  out.drafter_temperature_factor = drafter_temperature_factor;
  out.drafter_top_p = drafter_top_p;
  // The routes a load decides from the card and a measurement switch, and the switches it reads
  // on their own, in this order, so a value a switch does not know refuses the plan and the load by
  // name: IGNIS_HQ_DECODE_ATTENTION (GitHub #572), IGNIS_PREFILL_SWIGLU (GitHub #651, #653: unset
  // or `quant`, `fused`, `split`), IGNIS_W4A4_ACTIVATION_LAYOUT (GitHub #652: unset or `ktile`,
  // `token`) and IGNIS_FP8_GEMV_PAIRED (GitHub #678: unset or `1`, `0`; each call of the FP8 q | k |
  // gate | v projection reads it again, kernel/src/fp8_exception.h).
  bool fp8_gemv_paired = true;
  if (!gdn_prefill_route(out.gdn_prefill_ours) ||
      !load_switch(ignis::hq_decode_attention::switch_for_load, out.hq_decode_attention) ||
      !mtp_head_linear_route(out.mtp_head_linear_ours) ||
      !tma_descriptor_cache_route(out.tma_descriptor_cache) ||
      !load_switch(ignis::prefill_swiglu::route_for_load, out.prefill_swiglu) ||
      !load_switch(ignis::w4a4_activation_layout::layout_for_load, out.w4a4_activation_layout) ||
      !load_switch(ignis::fp8_exception::gemv_paired, fp8_gemv_paired)) {
    return nullptr;
  }

  // GitHub #302: Flash-Next binds its own schema (kernel/src/flash_next/
  // bind.cu) into its own program (kernel/src/flash_next/program.h), and has
  // no drafter or attention readouts. Its decode lanes and its residency are
  // its options alone. GitHub #557: with vision, its embedding pool is the
  // 27B's rule -- floored at the envelope's own output, whole pages, in VRAM
  // or mapped host memory by the vision residency -- at its own hidden width;
  // its encoder is the vision companion's (ADR 0087): the `vision/*` tensors
  // bind into the 27B's tower schema at the merger's 2,560 rows, and an item
  // encodes in the prefill scratch and the expert staging ring (GitHub #558),
  // so the item bound is what those hold.
  if (flash_next) {
    if (prefill_attention != IGNIS_PREFILL_ATTENTION_FP8) {
      set_error("ignis_model_load: prefill_attention is the 27B's; Qwen3.8-Flash-Next's attention is QSA");
      return nullptr;
    }
    if (prefill_linear != IGNIS_PREFILL_LINEAR_FUSED) {
      set_error("ignis_model_load: prefill_linear is the 27B's; Qwen3.8-Flash-Next has its own prefill");
      return nullptr;
    }
    if (draft_rule != IGNIS_DRAFT_RULE_FIXED) {
      set_error("ignis_model_load: draft_rule is the 27B's; Qwen3.8-Flash-Next's MTP speculation keeps its own rule");
      return nullptr;
    }
    if (attention_text_max_keys != 0) {
      set_error("ignis_model_load: Qwen3.8-Flash-Next has no attention readouts");
      return nullptr;
    }
    const uint64_t vision_pool_bytes =
        vision_max_tokens == 0
            ? 0
            : vision_embedding_pool_bytes(static_cast<std::int64_t>(topology->hidden),
                                          static_cast<std::int32_t>(std::min(vision_max_tokens, max_context_tokens)),
                                          vision_embedding_pool_request);
    // GitHub #307 (spec flash-next/07): MTP, or the verify round alone for a
    // test's drafts; DFlash2 is the 27B's.
    if (speculative_backend == IGNIS_SPECULATIVE_DFLASH2) {
      set_error("ignis_model_load: DFlash2 is the 27B's drafter; Qwen3.8-Flash-Next drafts with its MTP head");
      return nullptr;
    }
    // Without vision a `vision/*` tensor stays among the program's, whose
    // binder refuses it as an extra.
    std::vector<ignis_bound_tensor> text;
    std::vector<ignis_bound_tensor> tower;
    for (uint64_t i = 0; i < count; ++i) {
      const bool is_tower = vision_max_tokens > 0 && tensors[i].name != nullptr &&
                            std::strncmp(tensors[i].name, "vision/", 7) == 0;
      (is_tower ? tower : text).push_back(tensors[i]);
    }
    std::string error;
    auto program = ignis::flash_next::bind_model(text.data(), text.size(), *topology, prefill_chunk_tokens,
                                                 max_context_tokens, kv_format, options, rope_scaling,
                                                 vision_pool_bytes,
                                                 vision_residency == IGNIS_VISION_RESIDENCY_HOST, &error);
    if (program == nullptr) {
      set_error("ignis_model_load: " + error);
      return nullptr;
    }
    auto model = std::make_unique<ignis_model>();
    model->flash_next.reset(program.release());
    if (vision_max_tokens > 0) {
      ModelBinder binder(tower.data(), tower.size());
      if (!binder.build_index(tower.size()) ||
          !bind_vision(binder, Geometry::from(*topology), model->vision) || !binder.require_no_extras()) {
        return nullptr;
      }
      model->vision_max_tokens = vision_max_tokens;
      // GitHub #558 (ADR 0087): the encode also takes the expert staging
      // ring, idle between forwards (ignis_residency.h), so the item bound is
      // what the two hold.
      if (staging_ring_bytes != nullptr) {
        const std::size_t scratch = ignis::flash_next::reservations(*model->flash_next).workspace_bytes;
        const std::size_t ring = *staging_ring_bytes;
        const std::size_t slot =
            vision_residency == IGNIS_VISION_RESIDENCY_HOST ? ignis_vision_staging_slot_bytes(model->vision) : 0;
        const uint32_t bound =
            vision_item_bound(scratch, ring, slot, std::min(out.vision_item_max_tokens, max_context_tokens));
        if (bound == 0) {
          set_error("ignis_model_load: Qwen3.8-Flash-Next's prefill scratch of " + std::to_string(scratch) +
                    " bytes" +
                    (ring == 0 ? std::string(" holds")
                               : " and its expert staging ring of " + std::to_string(ring) + " bytes hold") +
                    " no media item's encode" +
                    (slot == 0 ? std::string() : " with the vision tower's staging slots") +
                    " -- a wider --model-prefill-chunk" +
                    (slot == 0 ? std::string() : ", or the tower in VRAM (--vision-residency device)"));
          return nullptr;
        }
        out.vision_item_max_tokens = bound;
        model->vision_item_max_tokens = bound;
      }
    }
    return model;
  }
  if (options != nullptr && (options->residency != nullptr || options->draft_row_budget != 0 ||
                             options->mtp_expert_slots != nullptr)) {
    set_error("ignis_model_load: residency, draft_row_budget and mtp_expert_slots are "
              "Qwen3.8-Flash-Next's options");
    return nullptr;
  }
  // GitHub #336: the 27B's decode lanes are a load option too -- the widest
  // round, which every round reservation below is sized for and every round
  // graph is captured up to. 0 is the most a round serves.
  const uint32_t decode_lanes =
      options != nullptr && options->decode_lanes != 0 ? options->decode_lanes : IGNIS_DECODE_MAX_BATCH;
  if (decode_lanes > IGNIS_DECODE_MAX_BATCH) {
    set_error("ignis_model_load: decode_lanes " + std::to_string(decode_lanes) + " is more than the " +
              std::to_string(IGNIS_DECODE_MAX_BATCH) + " a Qwen 3.8-27B round serves");
    return nullptr;
  }
  out.decode_lanes = decode_lanes;
  // GitHub #620: the 27B's own MTP head, whose draft rule is the fixed
  // draft's. GitHub #624: it binds beside vision as DFlash2 does
  // (kernel/src/mtp_head.h says how its rows rotate and embed on a vision
  // load).
  if (speculative_backend == IGNIS_SPECULATIVE_MTP && draft_rule != IGNIS_DRAFT_RULE_FIXED) {
    set_error("ignis_model_load: the 27B's MTP head drafts its best token at every temperature, the fixed "
              "draft; draft_rule 1 (sampled) is DFlash2's");
    return nullptr;
  }

  ModelBinder binder(tensors, count);
  if (!binder.build_index(count)) {
    return nullptr;
  }

  const Geometry g = Geometry::from(*topology);
  auto model = std::make_unique<ignis_model>();
  // GitHub #227: the table the text layers will rotate at, resolved once
  // here -- the linear one without scaling, the YaRN one with it.
  model->text_rope = ignis::text_rope_frequencies(rope_scaling);

  // GitHub #339 (spec 27b-16gb/05a): the body is the head's. The 16 GB
  // artifact's head is trellis like every one of its body's linears, the
  // served artifact's W8 beside its NVFP4 planes; the whole body binds under
  // the one schema, so a tensor of the other body fails naming itself.
  model->is_trellis_body = binder.is_trellis("text/output_head");
  if (!binder.bind("text/token_embedding", {g.vocab, g.hidden}, model->token_embedding) ||
      !binder.bind("text/final_norm", {g.hidden}, model->final_norm) ||
      !(model->is_trellis_body
            ? binder.bind_trellis("text/output_head", {g.vocab, g.hidden}, model->trellis_head)
            : binder.bind("text/output_head", {g.vocab, g.hidden}, model->output_head))) {
    return nullptr;
  }
  // GitHub #509: the dense prefill decodes trellis matrices; an NVFP4 body has none.
  if (prefill_linear == IGNIS_PREFILL_LINEAR_DENSE && !model->is_trellis_body) {
    set_error("ignis_model_load: prefill_linear 1 (the dense prefill) runs the trellis body's matrices, and this "
              "artifact's body is NVFP4");
    return nullptr;
  }

  model->layers.resize(topology->num_layers);
  for (uint32_t i = 0; i < topology->num_layers; ++i) {
    const std::string prefix = "text/layers/" + std::to_string(i) + "/";
    const auto kind = static_cast<ignis_layer_kind>(topology->layer_kinds[i]);
    model->layers[i].kind = kind;
    const bool ok =
        model->is_trellis_body
            ? ((kind == IGNIS_LAYER_GQA) ? bind_gqa_layer_trellis(binder, prefix, g, model->layers[i].gqa)
                                         : bind_gdn_layer_trellis(binder, prefix, g, model->layers[i].gdn))
            : ((kind == IGNIS_LAYER_GQA) ? bind_gqa_layer(binder, prefix, g, model->layers[i].gqa)
                                         : bind_gdn_layer(binder, prefix, g, model->layers[i].gdn));
    if (!ok) {
      return nullptr;
    }
  }

  // GitHub #619: the weights the load's drafter binds. Without the option
  // the `dflash2/*` tensors are not asked for, so a caller that hands them
  // over anyway fails on `require_no_extras` below.
  model->drafter = ignis_drafter_of(speculative_backend);
  switch (model->drafter) {
  case IgnisDrafter::kDflash2:
    if (!bind_dflash2(binder, g, model->dflash2)) {
      return nullptr;
    }
    // The shortlist proposal head is the caller's choice (the artifact always
    // carries it): bound when either half is handed over, so a load that sends
    // one without the other fails naming the missing one, and one without a
    // drafter fails on `require_no_extras`.
    if (binder.has("text/draft_head") || binder.has("text/draft_head_token_ids")) {
      ninfer::Weight token_ids{};
      if (!binder.bind("text/draft_head", {kProposalHeadRows, g.hidden}, model->proposal_head) ||
          !binder.bind("text/draft_head_token_ids", {kProposalHeadRows}, token_ids)) {
        return nullptr;
      }
      model->proposal_token_ids = static_cast<const std::int32_t *>(token_ids.qdata);
    }
    break;
  case IgnisDrafter::kMtp:
    // GitHub #620: the head's 12 objects. Without the option a caller that
    // hands them over fails on `require_no_extras` below.
    if (!bind_mtp_head(binder, g, model->mtp) || !check_mtp_head_formats(model->mtp)) {
      return nullptr;
    }
    break;
  case IgnisDrafter::kHost:
  case IgnisDrafter::kNone:
    break;
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

// The drafter round's vendored workspaces (`swa`, `linear_swiglu`) at the
// load's `lanes`, checked against their fixed allowance; see
// kDflash2RoundWorkspaceBytes.
void check_dflash2_round_workspace(const ignis_model &model, const Geometry &g,
                                   uint32_t max_context_tokens, uint32_t draft_tokens,
                                   uint32_t lanes) {
  const auto block = static_cast<std::int32_t>(draft_tokens + 1);
  const std::size_t workspace_bytes =
      ninfer::ops::swa_workspace_capacity_bytes({0, max_context_tokens}, 1, block,
                                                static_cast<std::int32_t>(lanes)) +
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
  // GitHub #275 (ADR 0041): the attention readout's room for a text span --
  // its scores, the set's results and IGNIS_ATTENTION_MAX_ROW_HEADS whole
  // rows over the load's `attention_text_max_keys` -- which a prefill chunk
  // takes from the same arena beside `prefill_scratch`. A text load's own,
  // so counted with the prefill scratch rather than as vision's.
  std::size_t text_readout = 0;
  // GitHub #260 (ADR 0038): what a vision load's readout needs beyond the
  // text room: the scores of one item's span, and the set's results when
  // there is no text room to share them with. Kept apart from the prefill
  // scratch because only a vision load has it: what vision adds to the
  // arena is counted as vision's (`vision_reserved_bytes`), readout
  // included.
  std::size_t attention_readout = 0;
  std::size_t vision_workspace = 0;
  // The embedding pool's bytes, wherever the vision residency puts it: a
  // VRAM line under DEVICE, host-mapped and no VRAM line under HOST (GitHub
  // #415). `vision` says the load has vision whatever the pool's place.
  std::size_t media_embedding = 0;
  bool media_embedding_host = false;
  bool vision = false;
  std::size_t sampling_workspace = 0;
  std::size_t decode_graph_scratch = 0;
  // The verify round's buffers; all zero without a draft window. GitHub #687:
  // on a load of two round shapes, each the larger of the two shapes'.
  VerifyRoundSizes verify_round;
  std::size_t drafter_features = 0;
  // One count per lane: the columns its committed run appends to its window.
  std::size_t drafter_append_counts = 0;
  std::size_t drafter_scratch = 0;
  // GitHub #576: under the sampled draft, the drafter's q kept from the draw
  // to the accept -- its candidates' ids and probabilities (0 otherwise).
  std::size_t drafter_draws = 0;
  // GitHub #687: the load's base shape's window, and its extended shape's
  // window and lanes (0 on a load with one shape). Its rounds run in the base
  // shape's buffers, so `verify_round` and the drafter's lines above are each
  // buffer at the larger of the two shapes'.
  uint32_t base_window = 0;
  uint32_t extended_window = 0;
  uint32_t extended_lanes = 0;
  std::size_t sampling_logits = 0;
  // GitHub #339: the trellis body's one workspace (0 on the NVFP4 body).
  std::size_t trellis_workspace = 0;
  // GitHub #509: the dense prefill's one buffer (0 unless the load runs it).
  std::size_t trellis_dense = 0;
  // GitHub #650: the W4A4 TMA products' descriptor slots (0 unless the load keeps them).
  std::size_t tma_descriptors = 0;
  // GitHub #336: the load's decode lanes, which every round line is sized
  // for.
  std::size_t lanes = IGNIS_DECODE_MAX_BATCH;

  // GitHub #212: the one arena prefill chunks and media encode share. A
  // media encode runs between prefill steps, never inside one, so the two
  // are never live at once -- ninfer sizes its single workspace the same
  // way.
  std::size_t workspace() const {
    return std::max(prefill_scratch + text_readout + attention_readout, vision_workspace);
  }

  ignis_model_reservations reservations() const {
    ignis_model_reservations out{};
    out.workspace_bytes = workspace();
    out.media_embedding_bytes = media_embedding_host ? 0 : media_embedding;
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
    out.verify_round_bytes = verify_round.total();
    // The load's drafter's round buffers, as `plan_load_sizes` chose them
    // (GitHub #619): 0 without a drafter that has any.
    out.drafter_round_bytes = drafter_features + drafter_append_counts + drafter_draws + drafter_scratch;
    out.trellis_workspace_bytes = trellis_workspace;
    out.trellis_dense_bytes = trellis_dense;
    out.tma_descriptor_bytes = tma_descriptors;
    return out;
  }
};

// GitHub #339 (spec 27b-16gb/05a): the trellis body's workspace -- the widest
// call its linears and its head make, asked of the op on this device
// (`ignis_trellis_workspace_bytes`: the decode route splits its inputs for the
// device's resident CTAs, so the line is the device's own; ADR 0030). Every
// call runs on the model's one stream, so one workspace serves them all.
//
// A body linear runs at any width a prefill chunk or a round has: 1..
// IGNIS_TRELLIS_DECODE_MAX_TOKENS on the decode route -- a short prompt, or a
// span's last chunk, is a chunk that narrow -- and up to the load's chunk on
// the prefill route, whose workspace grows with the width. The head runs at a
// round's columns (a decode round's lanes, a verify round's `k+1` per lane,
// the drafter's `k` per lane), one span-logits block, or the one column a
// prefill samples from: never past `head_tokens`. Asking every width the
// route splits differently at is what makes the line an upper bound; a call
// past it fails by name (the op checks its workspace) rather than overrun it.
//
// The head is asked on the decode route alone, which holds only while its
// widest call fits that route: a verify round's `lanes x (k + 1)` is at most
// 8 x 8 = 64 by the load's own limits (GitHub #687: an extended round's 4 x 16
// at most too), and a span-logits block 32. Both are
// pinned below at compile time, and a load whose head would still be wider is
// refused here, naming why, rather than its first such round failing mid-
// decode on a workspace sized for a narrower call.
//
// GitHub #509: under the dense prefill (`dense`) no body call wider than the
// decode route reaches the fused op, so the prefill width asks the dense op
// instead: the INT8 activations of each shared input's matrices together
// (`ignis_dense_prefill_workspace_bytes`), past the fused op's counters.
// GitHub #687: a round at the base shape runs every lane count, at most the
// base shape's window (the MTP head's and the verify substrate's caps among
// it); one at the extended shape its fewer lanes, at most DFlash2's cap.
static_assert(IGNIS_MTP_MAX_DRAFT_TOKENS <= IGNIS_BASE_SHAPE_DRAFT_TOKENS &&
                  IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS <= IGNIS_BASE_SHAPE_DRAFT_TOKENS,
              "a backend without an extended shape drafts no more than the base shape at every lane count");
static_assert(IGNIS_DECODE_MAX_BATCH * (IGNIS_BASE_SHAPE_DRAFT_TOKENS + 1) <= IGNIS_TRELLIS_DECODE_MAX_TOKENS,
              "a verify round's head columns (lanes x (k + 1)) must stay on the trellis op's decode route");
static_assert(IGNIS_EXTENDED_SHAPE_MAX_LANES * (IGNIS_DFLASH2_MAX_DRAFT_TOKENS + 1) <= IGNIS_TRELLIS_DECODE_MAX_TOKENS,
              "an extended verify round's head columns (lanes x (k + 1)) must stay on the trellis op's decode route");
static_assert(kSpanLogitsBlockColumns <= IGNIS_TRELLIS_DECODE_MAX_TOKENS,
              "a span-logits block's head columns must stay on the trellis op's decode route");
std::size_t trellis_workspace_bytes(const ignis_model &model, uint32_t prefill_chunk_tokens,
                                    uint32_t head_tokens, bool dense) {
  if (head_tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS) {
    throw std::runtime_error("the trellis head's widest call, " + std::to_string(head_tokens) +
                             " columns (a round's lanes x (k + 1), or a span-logits block), is past the trellis " +
                             "op's decode route (" + std::to_string(IGNIS_TRELLIS_DECODE_MAX_TOKENS) +
                             " tokens), the only route the head's workspace is sized for");
  }
  if (ignis_trellis_prepare() != 0) {
    // Kernel attributes and launch geometry only: no allocation the plan
    // would have to state (the SM probe runs under the debug cap alone).
    throw std::runtime_error(std::string("preparing the trellis linear: ") + ignis_trellis_last_error());
  }
  // `group`: the matrices that read the call's input, which the dense
  // prefill quantizes together (trellis_body.h).
  struct Call {
    uint32_t in, out, weights, max_decode_tokens, prefill_tokens, group;
  };
  std::vector<Call> calls;
  const auto body = [&](const ignis_trellis_weight &w, uint32_t weights, uint32_t group) {
    calls.push_back({w.in, w.out, weights, IGNIS_TRELLIS_DECODE_MAX_TOKENS, prefill_chunk_tokens, group});
  };
  // The tail's output and down projections, and its gate and up as one SwiGLU.
  const auto tail = [&](const TrellisTail &t) {
    body(t.output, 1, 1);
    body(t.down, 1, 1);
    body(t.gate_proj, 2, 2);
  };
  for (const LayerWeights &layer : model.layers) {
    if (layer.kind == IGNIS_LAYER_GQA) {
      const GqaTrellisWeights &t = layer.gqa.trellis;
      for (const ignis_trellis_weight *w : {&t.q_proj, &t.k_proj, &t.v_proj}) {
        body(*w, 1, 3);
      }
      tail(t.tail);
    } else {
      const GdnTrellisWeights &t = layer.gdn.trellis;
      for (const ignis_trellis_weight *w : {&t.in_proj_qkv, &t.in_proj_z}) {
        body(*w, 1, 2);
      }
      tail(t.tail);
    }
  }
  calls.push_back({model.trellis_head.in, model.trellis_head.out, 1, head_tokens, 0, 1});
  std::size_t line = 0;
  for (std::size_t i = 0; i < calls.size(); ++i) {
    const Call &c = calls[i];
    // The 64 layers repeat a handful of shapes: ask each once.
    bool seen = false;
    for (std::size_t j = 0; j < i && !seen; ++j) {
      seen = calls[j].in == c.in && calls[j].out == c.out && calls[j].weights == c.weights &&
             calls[j].max_decode_tokens == c.max_decode_tokens && calls[j].prefill_tokens == c.prefill_tokens &&
             calls[j].group == c.group;
    }
    if (seen) {
      continue;
    }
    const auto ask = [&](uint32_t tokens) {
      const uint64_t bytes = ignis_trellis_workspace_bytes(c.in, c.out, c.weights, tokens);
      if (bytes == 0) {
        throw std::runtime_error("the trellis linear refuses [" + std::to_string(c.out) + " x " +
                                 std::to_string(c.in) + "] at " + std::to_string(tokens) + " tokens: " +
                                 ignis_trellis_last_error());
      }
      line = std::max<std::size_t>(line, bytes);
    };
    for (uint32_t tokens = 1; tokens <= c.max_decode_tokens; ++tokens) {
      ask(tokens);
    }
    if (c.prefill_tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS && !dense) {
      ask(c.prefill_tokens);
    }
    if (c.prefill_tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS && dense) {
      const uint64_t bytes = ignis_dense_prefill_workspace_bytes(c.in, c.group, c.prefill_tokens);
      if (bytes == 0) {
        throw std::runtime_error("the dense prefill refuses " + std::to_string(c.group) + " matrices of " +
                                 std::to_string(c.in) + " inputs at " + std::to_string(c.prefill_tokens) + " tokens");
      }
      line = std::max<std::size_t>(line, bytes);
    }
  }
  return line;
}

// GitHub #509: the dense prefill's one buffer -- the INT8 copy of the body's
// largest trellis matrix, which each matrix is decoded into in turn (on the
// 27B the MLP's gate, up and down, 5120 x 17408 bytes). Never the head's: the
// head runs on the fused op's decode route.
std::size_t trellis_dense_bytes(const ignis_model &model) {
  std::size_t line = 0;
  const auto body = [&](const ignis_trellis_weight &w) {
    line = std::max<std::size_t>(line, ignis_dense_prefill_weight_bytes(w.in, w.out));
  };
  for (const LayerWeights &layer : model.layers) {
    const TrellisTail &tail = layer.kind == IGNIS_LAYER_GQA ? layer.gqa.trellis.tail : layer.gdn.trellis.tail;
    if (layer.kind == IGNIS_LAYER_GQA) {
      body(layer.gqa.trellis.q_proj);
      body(layer.gqa.trellis.k_proj);
      body(layer.gqa.trellis.v_proj);
    } else {
      body(layer.gdn.trellis.in_proj_qkv);
      body(layer.gdn.trellis.in_proj_z);
    }
    body(tail.output);
    body(tail.down);
    body(tail.gate_proj);
    body(tail.up_proj);
  }
  return line;
}

// GitHub #650: the W4A4 TMA products' descriptor slots (kernel/src/tma_descriptor_cache.h) -- a
// slot for each product a prefill traversal runs on the route, its NVFP4 parents (the query, key,
// gate and value or the GDN input projection, the output projection, the MLP's gate and up, one
// block a width on either SwiGLU route (GitHub #651), and its down; an exception tensor held in
// FP8 or BF16 runs elsewhere), at every width the route takes up to the chunk. 0 on a load that
// keeps none (`tma_descriptor_cache_route`) and on the trellis body, which runs no W4A4 product.
std::size_t tma_descriptor_bytes(const ignis_model &model, uint32_t prefill_chunk_tokens,
                                 const LoadOptions &options) {
  if (!options.tma_descriptor_cache || model.is_trellis_body) {
    return 0;
  }
  std::size_t products = 0;
  const auto parent = [&](const ninfer::Weight &w) { products += w.qtype == ninfer::QType::NVFP4 ? 1 : 0; };
  for (const LayerWeights &layer : model.layers) {
    if (layer.kind == IGNIS_LAYER_GQA) {
      parent(layer.gqa.query_key_gate_value);
      parent(layer.gqa.output);
      parent(layer.gqa.mlp_gate_up);
      parent(layer.gqa.mlp_down);
    } else {
      parent(layer.gdn.query_key_value_z);
      parent(layer.gdn.output);
      parent(layer.gdn.mlp_gate_up);
      parent(layer.gdn.mlp_down);
    }
  }
  return ignis::tma_descriptors::planned_slots(products, prefill_chunk_tokens) * ignis::tma_descriptors::kSlotBytes;
}

// The attention readout's room in a prefill chunk's arena (GitHub #260,
// #263, #264, #275): one F32 score per key of a span of `score_keys`, one
// packed (score, key) and four neighbour scores per head of the largest set
// a readout may name, and -- over a text span of `row_keys` -- every head's
// row for the most heads a readout reads in rows. `step.cu` takes all of it
// from this arena. None without a span to read.
std::size_t attention_readout_bytes(std::int32_t score_keys, std::int32_t row_keys) {
  if (score_keys <= 0) {
    return 0;
  }
  return fp32_bytes(score_keys) +
         round_up_arena_align(static_cast<std::size_t>(kReadoutMaxSetHeads) * sizeof(unsigned long long)) +
         round_up_arena_align(static_cast<std::size_t>(kReadoutMaxSetHeads) * 4 * sizeof(float)) +
         (row_keys > 0 ? fp32_bytes(static_cast<std::int64_t>(kReadoutMaxRowHeads) * row_keys) : 0);
}

LoadSizes plan_load_sizes(const ignis_model &model, const ignis_topology &topology,
                          uint32_t prefill_chunk_tokens, uint32_t max_context_tokens,
                          const LoadOptions &options) {
  const Geometry g = Geometry::from(topology);
  LoadSizes sizes;
  // GitHub #336: every round line below is the load's lanes wide.
  const uint32_t lanes = options.decode_lanes;
  sizes.lanes = lanes;

  // P2-01 (GitHub #83): the scratch, sized for a `prefill_chunk_tokens`-wide
  // chunk (see compute_program_scratch_bytes above), for the routes the
  // load's prefill runs (the review of spec #503).
  LayerRoutes prefill_routes;
  prefill_routes.split_conv = true;
  prefill_routes.dense = options.prefill_linear == IGNIS_PREFILL_LINEAR_DENSE;
  prefill_routes.gdn_ours = options.gdn_prefill_ours;
  sizes.prefill_scratch = compute_program_scratch_bytes(
      model, topology, static_cast<std::int32_t>(prefill_chunk_tokens), max_context_tokens,
      options.cache_dtype, /*batch=*/1, /*head_every_column=*/false, prefill_routes);

  // GitHub #177: the encoder workspace for one item's merged tokens, and
  // GitHub #243 the embedding pool for the envelope's -- both capped by the
  // context. The encoder only ever holds one item, and an item is bounded by
  // the load's item bound, never past the envelope.
  if (options.vision_max_tokens > 0) {
    const auto tokens =
        static_cast<std::int32_t>(std::min(options.vision_max_tokens, max_context_tokens));
    const auto item_tokens =
        static_cast<std::int32_t>(std::min(options.vision_item_max_tokens, max_context_tokens));
    // GitHub #260 (ADR 0038): a head point's attention readout scores one
    // image's placeholder span from the prefill chunk's own scope -- one F32
    // per key, and an image holds at most one item's tokens. Only a vision
    // load can be asked one (the span is an image's), so a text load
    // reserves nothing for it.
    //
    // GitHub #263 (ADR 0039): and beside them the head set's results, one
    // packed (score, key) per head, reserved for the largest set a readout
    // may name -- under 4 KB -- and (GitHub #264) the four neighbour scores
    // per head that ride with it, 6 KB more. `step.cu` takes all three from
    // this arena; the neighbours went unreserved until the encoder stopped
    // sizing the arena past the prefill scratch, which had hidden them.
    //
    // GitHub #275: beyond the text room, if the load has one -- the two
    // readouts never run at once, so the item's scores need only what the
    // text scores do not already hold.
    const auto text_keys = static_cast<std::int32_t>(options.attention_text_max_keys);
    sizes.attention_readout = attention_readout_bytes(std::max(item_tokens, text_keys), text_keys) -
                              attention_readout_bytes(text_keys, text_keys);
    sizes.vision_workspace =
        ignis_vision_workspace_bytes(item_tokens, std::min(item_tokens, kVisionMaxSegments));
    // GitHub #414 (ADR 0075): a tower in host memory is staged through a ring
    // after the workspace, in the encoder's share of the scratch; the scratch
    // is still the larger of the two shares (ADR 0030), so the ring costs
    // VRAM only where the encoder's share sets the scratch's size.
    if (options.vision_residency == IGNIS_VISION_RESIDENCY_HOST) {
      sizes.vision_workspace =
          ignis_vision_staged_share(sizes.vision_workspace, ignis_vision_staging_slot_bytes(model.vision)).bytes;
    }
    sizes.media_embedding =
        vision_embedding_pool_bytes(g.hidden, tokens, options.vision_embedding_pool_bytes);
    // GitHub #415: and the pool, the same pages and the same bytes, is
    // host-mapped, so it leaves the VRAM plan.
    sizes.media_embedding_host = options.vision_residency == IGNIS_VISION_RESIDENCY_HOST;
    sizes.vision = true;
  }

  // GitHub #275 (ADR 0041): a `locate`'s readout over a text span, reserved
  // at load whether or not the load has vision (ADR 0030) -- a text load's
  // own, so counted with the prefill scratch.
  sizes.text_readout = attention_readout_bytes(static_cast<std::int32_t>(options.attention_text_max_keys),
                                               static_cast<std::int32_t>(options.attention_text_max_keys));

  // P3-03 (GitHub #99): device sampling's workspace and its decode logits.
  const auto vocab = static_cast<std::int32_t>(g.vocab);
  sizes.sampling_logits = static_cast<std::size_t>(vocab) * lanes * sizeof(std::uint16_t);
  sizes.sampling_workspace = std::max<std::size_t>(
      ninfer::ops::sampling_workspace_capacity_bytes(vocab, 1, static_cast<std::int32_t>(lanes)), kArenaAlign);

  // GitHub #687: past the base shape's window a DFlash2 load runs two round
  // shapes out of one set of buffers: its base shape (the load's round below,
  // at IGNIS_BASE_SHAPE_DRAFT_TOKENS and every lane count) and its extended
  // shape (the load's window, at no more than IGNIS_EXTENDED_SHAPE_MAX_LANES).
  const bool two_shapes =
      model.drafter == IgnisDrafter::kDflash2 && options.draft_tokens > IGNIS_BASE_SHAPE_DRAFT_TOKENS;
  sizes.base_window = two_shapes ? IGNIS_BASE_SHAPE_DRAFT_TOKENS : options.draft_tokens;
  if (two_shapes) {
    sizes.extended_window = options.draft_tokens;
    sizes.extended_lanes = std::min<uint32_t>(lanes, IGNIS_EXTENDED_SHAPE_MAX_LANES);
  }
  const bool vision = options.vision_max_tokens > 0;

  // P3-05 (GitHub #102) / P5-04 (GitHub #153): the round scratch, one column
  // per lane, or `k+1` with a draft window.
  const bool windowed = options.draft_tokens > 0;
  const auto round_columns = static_cast<std::int32_t>(windowed ? sizes.base_window + 1 : 1);
  sizes.decode_graph_scratch = compute_program_scratch_bytes(
      model, topology, /*chunk=*/round_columns, max_context_tokens, options.cache_dtype,
      /*batch=*/static_cast<std::int32_t>(lanes), /*head_every_column=*/windowed, LayerRoutes{});
  if (two_shapes) {
    // GitHub #687: and the extended round's, asked at two lanes or more: one
    // lane's attention runs in chunks of the small-T tile (kernel/src/gqa_layer.cu,
    // kernel/src/round_shape.h), which a two-lane query bounds, where a
    // one-lane query would size the prompt route's planes for the context.
    const auto extended_batch = static_cast<std::int32_t>(std::max<uint32_t>(2, sizes.extended_lanes));
    sizes.decode_graph_scratch =
        std::max(sizes.decode_graph_scratch,
                 compute_program_scratch_bytes(model, topology, static_cast<std::int32_t>(sizes.extended_window + 1),
                                               max_context_tokens, options.cache_dtype, extended_batch,
                                               /*head_every_column=*/true, LayerRoutes{}));
  }

  if (windowed) {
    sizes.verify_round = verify_round_sizes(topology, sizes.base_window, vision, lanes);
  }
  if (two_shapes) {
    sizes.verify_round = larger_of(
        sizes.verify_round, verify_round_sizes(topology, sizes.extended_window, vision, sizes.extended_lanes));
  }

  // GitHub #619: the load's drafter's lines, chosen here once -- what its
  // prefill adds to a chunk's scratch, and its round buffers. The verify
  // substrate alone has neither: its drafts come from its caller.
  switch (model.drafter) {
  case IgnisDrafter::kDflash2: {
    // P5-03 (GitHub #152): the chunk's feature taps and their context append.
    sizes.prefill_scratch +=
        dflash2_prefill_scratch_bytes(topology, static_cast<std::int32_t>(prefill_chunk_tokens));
    // P5-05 (GitHub #155): the drafter's round buffers.
    check_dflash2_round_workspace(model, g, max_context_tokens, sizes.base_window, lanes);
    const auto block = static_cast<std::size_t>(sizes.base_window + 1);
    sizes.drafter_features = kDflash2TapLayers.size() * static_cast<std::size_t>(g.hidden) * block *
                             lanes * sizeof(std::uint16_t);
    sizes.drafter_append_counts = sizeof(std::int32_t) * lanes;
    sizes.drafter_scratch =
        dflash2_round_scratch_bytes(topology, static_cast<std::int32_t>(sizes.base_window), lanes);
    if (options.draft_rule == IGNIS_DRAFT_RULE_SAMPLED) {
      sizes.drafter_draws = dflash2_draw_bytes(sizes.base_window, lanes);
    }
    // GitHub #687: the extended shape drafts out of the same buffers, each the
    // larger of the two shapes' (its lanes are never more than the base
    // shape's, so the append counts are the base shape's).
    if (two_shapes) {
      check_dflash2_round_workspace(model, g, max_context_tokens, sizes.extended_window, sizes.extended_lanes);
      sizes.drafter_features =
          std::max(sizes.drafter_features, kDflash2TapLayers.size() * static_cast<std::size_t>(g.hidden) *
                                               (sizes.extended_window + 1) * sizes.extended_lanes *
                                               sizeof(std::uint16_t));
      sizes.drafter_scratch = std::max(
          sizes.drafter_scratch, dflash2_extended_round_scratch_bytes(
                                     topology, static_cast<std::int32_t>(sizes.extended_window), sizes.extended_lanes));
      if (options.draft_rule == IGNIS_DRAFT_RULE_SAMPLED) {
        sizes.drafter_draws =
            std::max(sizes.drafter_draws, dflash2_draw_bytes(sizes.extended_window, sizes.extended_lanes));
      }
    }
    break;
  }
  case IgnisDrafter::kMtp:
    // GitHub #620: the head's rows in every prefill chunk, at most
    // ignis::mtp::kPrefillRows a call, and in every round, over each of its
    // columns. Their attention takes the workspace the target's GQA layers
    // already hold at the same widths, so only the head's own buffers are
    // added (kernel/src/mtp_head.h); its weights are on the weights line and
    // its state is the pool's. GitHub #624: on a vision load, its rows'
    // rotation beside their positions.
    sizes.prefill_scratch += ignis::mtp::prefill_scratch_bytes(
        g.hidden, static_cast<std::int64_t>(prefill_chunk_tokens), options.vision_max_tokens > 0);
    sizes.decode_graph_scratch += ignis::mtp::round_scratch_bytes(
        g.hidden, static_cast<std::int64_t>(lanes) * round_columns, options.vision_max_tokens > 0);
    break;
  case IgnisDrafter::kHost:
  case IgnisDrafter::kNone:
    break;
  }

  // GitHub #339: the trellis body's workspace, whose head runs at most a
  // round's columns across the load's lanes or one span-logits block.
  if (model.is_trellis_body) {
    // GitHub #687: an extended round's, at its fewer lanes, too.
    const uint32_t round_head_columns =
        std::max(lanes * static_cast<uint32_t>(round_columns), sizes.extended_lanes * (sizes.extended_window + 1));
    const bool dense = options.prefill_linear == IGNIS_PREFILL_LINEAR_DENSE;
    sizes.trellis_workspace = trellis_workspace_bytes(
        model, prefill_chunk_tokens,
        std::max<uint32_t>(round_head_columns, static_cast<uint32_t>(kSpanLogitsBlockColumns)), dense);
    if (dense) {
      sizes.trellis_dense = trellis_dense_bytes(model);
    }
  }
  sizes.tma_descriptors = tma_descriptor_bytes(model, prefill_chunk_tokens, options);
  return sizes;
}

// GitHub #210: what a loaded model holds beside its weights, read off its own
// buffers -- the other half of `LoadSizes::reservations`, which the caller
// compares against its plan.
ignis_model_reservations reserved_of(const ignis_model &model) {
  // Of a unique or a shared pointer (GitHub #687: the round shapes' buffers
  // are shared).
  const auto bytes = [](const auto &buffer) -> uint64_t { return buffer != nullptr ? buffer->bytes : 0; };
  const auto capacity = [](const auto &arena) -> uint64_t { return arena != nullptr ? arena->capacity() : 0; };
  ignis_model_reservations out{};
  out.workspace_bytes = capacity(model.scratch);
  // GitHub #415: a host-mapped pool holds no VRAM.
  out.media_embedding_bytes = model.vision_pool.device_bytes();
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
  out.trellis_workspace_bytes = bytes(model.trellis_workspace_buffer);
  out.trellis_dense_bytes = bytes(model.trellis_dense_buffer);
  out.tma_descriptor_bytes = model.tma_descriptors != nullptr ? model.tma_descriptors->bytes() : 0;
  // GitHub #687: a load's extended shape holds the base shape's buffers and
  // none of its own, so the base shape's are every byte of both.
  if (model.base_shape != nullptr) {
    out.drafter_round_bytes = bytes(model.base_shape->features) + bytes(model.base_shape->append_counts) +
                              bytes(model.base_shape->draft_candidates) + bytes(model.base_shape->draft_q) +
                              capacity(model.base_shape->drafter_scratch);
    out.verify_round_bytes = model.base_shape->device_bytes() - out.drafter_round_bytes;
  }
  return out;
}

} // namespace

MappedHostRegion::MappedHostRegion(std::size_t size_bytes) {
  cudaError_t err = cudaHostAlloc(&host, size_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
  if (err != cudaSuccess) {
    host = nullptr;
    throw std::runtime_error("cudaHostAlloc(mapped) of " + std::to_string(size_bytes) +
                             " bytes failed: " + cudaGetErrorString(err));
  }
  err = cudaHostGetDevicePointer(&device, host, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host);
    host = nullptr;
    throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") + cudaGetErrorString(err));
  }
  bytes = size_bytes;
}

// GitHub #414 (ADR 0075): a loaded vision tower must be where the residency
// says -- the plan sized the scratch by the option, and a tower read from
// host memory in place would be a load that neither plan describes -- and,
// under host, every stage must fit a slot of the plan's ring before an encode
// needs it. Then the ring's own copy stream and events. Both models' loads
// (GitHub #557: Flash-Next's tower is its vision companion's), after the
// embedding pool; throws naming what failed.
static void prepare_vision_tower(ignis_model &model, bool host) {
  std::size_t tensors = 0;
  std::size_t in_host = 0;
  visit_vision_stages(model.vision, [&](std::size_t, const ninfer::Weight &w) {
    ++tensors;
    in_host += in_host_memory(w.qdata) ? 1 : 0;
  });
  if (in_host != (host ? tensors : 0)) {
    throw std::runtime_error(std::string("the vision residency is ") + (host ? "host" : "device") + " but " +
                             std::to_string(in_host) + " of the tower's " + std::to_string(tensors) +
                             " tensors are bound in host memory: bind the tower where the residency puts it");
  }
  if (!host) {
    return;
  }
  VisionStaging &staging = model.vision_staging;
  staging.slot_bytes = ignis_vision_staging_slot_bytes(model.vision);
  ignis_vision_staging_check(model.vision, staging.slot_bytes);
  // GitHub #415: the merger writes each host-mapped pool page through a
  // device page in a slot of the ring (vision_encode.cu).
  if (model.vision_pool.page_bytes > staging.slot_bytes) {
    throw std::runtime_error("an embedding pool page of " + std::to_string(model.vision_pool.page_bytes) +
                             " bytes does not fit a staging slot of " + std::to_string(staging.slot_bytes));
  }
  const auto event = [](cudaEvent_t &e) {
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
      throw std::runtime_error("creating a staging event failed");
    }
  };
  event(staging.head);
  for (std::size_t slot = 0; slot < IGNIS_VISION_STAGING_SLOTS; ++slot) {
    event(staging.copied[slot]);
    event(staging.consumed[slot]);
  }
  if (cudaStreamCreateWithFlags(&staging.copy, cudaStreamNonBlocking) != cudaSuccess) {
    staging.copy = nullptr;
    throw std::runtime_error("creating the staging copy stream failed");
  }
}

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
  // No item bound: it moves no line of the plan (ADR 0087).
  const auto model = validate_and_bind(tensors, count, topology, prefill_chunk_tokens,
                                       max_context_tokens, kv_format, options, load, nullptr);
  if (model == nullptr) {
    return -1;
  }
  if (model->flash_next) {
    *out = ignis::flash_next::reservations(*model->flash_next);
    return 0;
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

extern "C" int32_t ignis_model_plan_vision_item_max_tokens(const struct ignis_bound_tensor *tensors,
                                                           uint64_t count,
                                                           const struct ignis_topology *topology,
                                                           uint32_t prefill_chunk_tokens,
                                                           uint32_t max_context_tokens, int32_t kv_format,
                                                           const struct ignis_model_load_options *options,
                                                           uint64_t staging_ring_bytes, uint32_t *out_tokens) {
  if (out_tokens != nullptr) {
    *out_tokens = 0;
  }
  if (tensors == nullptr || topology == nullptr || out_tokens == nullptr) {
    set_error("ignis_model_plan_vision_item_max_tokens: null argument");
    return -1;
  }
  LoadOptions load{};
  const auto model = validate_and_bind(tensors, count, topology, prefill_chunk_tokens,
                                       max_context_tokens, kv_format, options, load, &staging_ring_bytes);
  if (model == nullptr) {
    return -1;
  }
  if (!model->flash_next && staging_ring_bytes != 0) {
    set_error("ignis_model_plan_vision_item_max_tokens: an expert staging ring is Qwen3.8-Flash-Next's; the 27B's "
              "encode has its own share of the scratch");
    return -1;
  }
  // What the load would hold: the bound as validated (on Flash-Next, what
  // its prefill scratch and its expert staging ring hold), within the
  // context, as ignis_model_vision_item_max_tokens reports it.
  if (load.vision_max_tokens > 0) {
    *out_tokens = std::min(load.vision_item_max_tokens, max_context_tokens);
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
  // GitHub #558 (ADR 0087): the expert staging ring of the residency a
  // Flash-Next load borrows, which its encode borrows in turn.
  const uint64_t staging_ring_bytes = lent_staging_ring_bytes(options);
  auto model = validate_and_bind(tensors, count, topology, prefill_chunk_tokens,
                                 max_context_tokens, kv_format, options, load, &staging_ring_bytes);
  if (model == nullptr) {
    return -1;
  }
  const Geometry g = Geometry::from(*topology);
  const int32_t speculative_backend = load.speculative_backend;
  const uint32_t draft_tokens = load.draft_tokens;
  const uint32_t vision_max_tokens = load.vision_max_tokens;
  model->speculative_backend = speculative_backend;
  model->draft_tokens = draft_tokens;
  // GitHub #336: the widest round this load serves.
  model->decode_lanes = load.decode_lanes;
  const std::size_t lanes = load.decode_lanes;

  // ADR 0062: a tensor bound from mapped pinned host memory (the 27B's
  // token embedding) takes no VRAM; the runtime says where each one lives.
  uint64_t vram_bytes = 0;
  uint64_t host_mapped_bytes = 0;
  for (uint64_t i = 0; i < count; ++i) {
    (in_host_memory(tensors[i].qdata) ? host_mapped_bytes : vram_bytes) += tensors[i].bytes;
  }
  model->bound_tensor_count = count;
  model->vram_bytes = vram_bytes;
  model->host_mapped_bytes = host_mapped_bytes;

  // GitHub #302: a Flash-Next load reserves and prepares its own program;
  // GitHub #557: and with vision, its companion's tower as the 27B's is.
  if (model->flash_next) {
    std::string error;
    int32_t rc = ignis::flash_next::finish_load(*model, &error);
    if (rc == 0 && vision_max_tokens > 0) {
      try {
        prepare_vision_tower(*model, load.vision_residency == IGNIS_VISION_RESIDENCY_HOST);
      } catch (const std::exception &e) {
        error = std::string("the vision tower's residency: ") + e.what();
        rc = -1;
      }
    }
    if (rc != 0) {
      set_error("ignis_model_load: " + error);
      if (model->stream != nullptr) {
        cudaStreamDestroy(model->stream);
        model->stream = nullptr;
      }
      return -1;
    }
    // GitHub #558 (ADR 0087): an encode may also take the expert staging
    // ring, which the item bound above counted on.
    if (vision_max_tokens > 0 && staging_ring_bytes > 0) {
      model->vision_ring_lender = options->residency;
    }
    *out_model = model.release();
    return 0;
  }

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
  // GitHub #335: the prompt attention's precision, read by every hq-e8-2b
  // prefill chunk (`kernel/src/gqa_layer.cu`).
  model->prefill_attention = load.prefill_attention;
  // GitHub #509: the trellis body's prefill route, read by every body call
  // (kernel/src/trellis_body.cu).
  model->prefill_linear = load.prefill_linear;
  // GitHub #514: the GDN prefill's route, read by every GDN prefill call
  // (kernel/src/gdn_layer.cu).
  model->gdn_prefill_ours = load.gdn_prefill_ours;
  // GitHub #651, #653 (spec #646): the MLP gate and up product's route and epilogue, read by every
  // MLP call of the NVFP4 body (kernel/src/gdn_layer.cu and gqa_layer.cu `output_and_mlp`).
  model->prefill_swiglu = load.prefill_swiglu;
  // GitHub #627 (spec #618): the MTP head's matrices, read by every head call
  // (kernel/src/mtp_head.cu) and so baked into every graph captured after this load.
  model->mtp_head_linear_ours = speculative_backend == IGNIS_SPECULATIVE_MTP && load.mtp_head_linear_ours;
  // GitHub #572 (spec #570): the decode and verify rounds' attention route, read by every call of
  // the batch body (kernel/src/gqa_layer.cu `run_gqa_layer_batch`) and so baked into every graph
  // captured after this load.
  if (!hq_decode_attention_route(load.hq_decode_attention, kv_format, model->hq_decode_attention_ours)) {
    return -1;
  }
  // GitHub #652 (spec #646): the layout the W4A4 prefill products read, named for the model's stream
  // once it exists (below).
  if (!w4a4_activation_layout_route(load.w4a4_activation_layout, model->is_trellis_body, prefill_chunk_tokens,
                                    model->w4a4_activation_layout)) {
    return -1;
  }

  // GitHub #341: the BF16 exception tensors a load holds in FP8 run on the FP8
  // linear and the W8A8 GEMM (kernel/src/fp8_exception.h), which refuse an
  // unprepared device. Prepared here, outside any capture, whichever way the
  // tensors are held: it only sets kernel attributes. The W8A8 GEMM compiles
  // for the sm_120a set alone (GitHub #456): a device with no image of it
  // leaves it unprepared, so its route refuses there, and the host never has
  // a load on such a card hold the tensors in FP8.
  const auto no_image_here = [] {
    const cudaError_t err = cudaGetLastError();
    return err == cudaErrorNoKernelImageForDevice || err == cudaErrorInvalidDeviceFunction;
  };
  if (ignis_fp8_linear_prepare() != 0 || (ignis_image_gemm_prepare() != 0 && !no_image_here())) {
    set_error(std::string("ignis_model_load: preparing the FP8 exception kernels: ") +
              ignis_fp8_linear_last_error() + " / " + ignis_image_gemm_last_error());
    return -1;
  }

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
  const auto vision_item_tokens = std::min(load.vision_item_max_tokens, max_context_tokens);
  const bool encoder_sized = sizes.vision_workspace > sizes.prefill_scratch;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (mem_err == cudaSuccess && scratch_bytes > free_bytes) {
    const std::string sized_by =
        encoder_sized ? "the vision encoder's workspace for a " + std::to_string(vision_item_tokens) +
                            "-token item"
                      : "a " + std::to_string(prefill_chunk_tokens) + "-token prefill chunk";
    set_error("ignis_model_load: " + sized_by + " needs a " + std::to_string(scratch_bytes) +
              "-byte scratch reservation, but only " + std::to_string(free_bytes) +
              " bytes are free -- " +
              (encoder_sized ? "lower --vision-max-tokens below the item bound"
                             : "pick a smaller prefill chunk width"));
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  try {
    model->scratch = std::make_unique<ninfer::DeviceArena>(scratch_bytes);
    model->prefill_scratch_bytes = sizes.prefill_scratch + sizes.text_readout;
    model->attention_text_max_keys = load.attention_text_max_keys;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: scratch arena allocation failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  // GitHub #339 (spec 27b-16gb/05a): the trellis body's one workspace, at
  // the address every trellis call -- eager or replayed from a round graph --
  // runs against, zeroed once on the model's stream before any of them (each
  // call leaves its counters zero). `plan_load_sizes` prepared the device.
  if (model->is_trellis_body) {
    try {
      model->trellis_workspace_buffer = std::make_unique<ninfer::DeviceBuffer>(sizes.trellis_workspace);
      model->trellis_workspace.base = model->trellis_workspace_buffer->p;
      model->trellis_workspace.bytes = model->trellis_workspace_buffer->bytes;
      if (ignis_trellis_workspace_init(&model->trellis_workspace, model->stream) != 0) {
        throw std::runtime_error(ignis_trellis_last_error());
      }
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: trellis workspace allocation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }
  // GitHub #509: the dense prefill's kernels prepared (its GEMM's shared
  // memory and occupancy), and its one buffer, at a stable address like the
  // workspace's.
  if (model->prefill_linear == IGNIS_PREFILL_LINEAR_DENSE) {
    try {
      if (ignis_dense_prefill_prepare() != 0) {
        throw std::runtime_error(ignis_trellis_last_error());
      }
      model->trellis_dense_buffer = std::make_unique<ninfer::DeviceBuffer>(sizes.trellis_dense);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: the dense prefill's buffer: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }
  // GitHub #650: the W4A4 TMA products' descriptor slots, serving the model's stream from here
  // on, so a first prefill finds them in place and allocates nothing a later one does not.
  if (sizes.tma_descriptors > 0) {
    try {
      model->tma_descriptors = std::make_unique<ignis::tma_descriptors::Cache>(
          sizes.tma_descriptors / ignis::tma_descriptors::kSlotBytes, model->stream);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: the TMA descriptor slots: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }
  // GitHub #652 (spec #646): every W4A4 TMA product launched on the model's stream from here on
  // reads its activations in the layout the load decided, which its quantizer then writes.
  if (model->w4a4_activation_layout.has_value()) {
    try {
      model->w4a4_activation_layout_stream = std::make_unique<ignis::w4a4_activation_layout::ForStream>(
          model->stream, *model->w4a4_activation_layout);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: the W4A4 activation layout: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }

  // GitHub #177: the vision reservation, taken here at load -- before the
  // caller builds its sequence pool -- so enabling vision can never OOM a
  // later request: GitHub #243's embedding pool, beside the scratch above
  // that already fits the encoder's workspace (GitHub #212). A reservation
  // that does not fit the free memory fails the load naming it.
  if (vision_max_tokens > 0) {
    model->vision_max_tokens = vision_max_tokens;
    model->vision_item_max_tokens = load.vision_item_max_tokens;
    try {
      const std::size_t output_bytes = sizes.media_embedding;
      std::size_t vision_free = 0;
      std::size_t vision_total = 0;
      if (!sizes.media_embedding_host && cudaMemGetInfo(&vision_free, &vision_total) == cudaSuccess &&
          output_bytes > vision_free) {
        throw std::runtime_error("a " + std::to_string(vision_tokens) +
                                 "-token vision envelope's embedding pool needs " +
                                 std::to_string(output_bytes) + " bytes, but only " +
                                 std::to_string(vision_free) +
                                 " are free -- lower --vision-embedding-pool or"
                                 " --vision-max-tokens");
      }
      const std::size_t page_bytes = vision_embedding_page_bytes(g.hidden);
      if (sizes.media_embedding_host) {
        // GitHub #415 (ADR 0075): the same pages in mapped pinned host
        // memory, which the host plan counted; a host that will not pin them
        // fails the load, never moving the pool to the device.
        try {
          model->vision_pool.mapped = std::make_unique<MappedHostRegion>(output_bytes);
        } catch (const std::exception &e) {
          throw std::runtime_error(std::string("the embedding pool in mapped host memory: ") + e.what() +
                                   " -- keep it in VRAM (--vision-residency device) or turn vision off");
        }
      } else {
        model->vision_pool.buffer = std::make_unique<ninfer::DeviceBuffer>(output_bytes);
      }
      model->vision_pool.page_columns = IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS;
      model->vision_pool.page_bytes = page_bytes;
      model->vision_pool.owner.assign(output_bytes / page_bytes, nullptr);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: vision reservation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
    // GitHub #414 (ADR 0075): the tower where the residency says, and under
    // host its staging ring (prepare_vision_tower).
    try {
      prepare_vision_tower(*model, load.vision_residency == IGNIS_VISION_RESIDENCY_HOST);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: the vision tower's residency: ") + e.what());
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
    model->sampling_decode_configs =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(ninfer::ops::SamplingConfig) * lanes);
    model->sampling_decode_positions = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes);
    // GitHub #178: allocated before the decode graphs are captured, which
    // bake its address into every GQA layer's rotation.
    if (vision_max_tokens > 0) {
      model->decode_rope_positions = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes);
    }
    model->sampling_decode_out = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes);
    model->sampling_decode_logits = std::make_unique<ninfer::DeviceBuffer>(sizes.sampling_logits);
    // GitHub #242: three small per-lane rows, allocated whether or not any
    // request ever constrains a draw -- 4 KB against a plan in gigabytes,
    // and an allocation that depends on traffic is one the VRAM plan cannot
    // state at load (ADR 0030).
    model->sampling_decode_permitted =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes * IGNIS_MAX_PERMITTED_TOKENS);
    model->sampling_decode_permitted_counts = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes);
    model->sampling_decode_permitted_probs = std::make_unique<ninfer::DeviceBuffer>(sizeof(float) * lanes);
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
  // replay rereads. Sized for the widest round the load admits: one token
  // per lane (`chunk=1`) across its `decode_lanes` lanes (GitHub #336;
  // IGNIS_DECODE_MAX_BATCH before), since #111 made a round one batch-wide
  // traversal instead of a per-lane loop, and every width 1..decode_lanes
  // shares this one reservation.
  // Capture itself (`ignis_decode_graph_capture`) happens later, once the
  // sequence pool exists.
  //
  // P5-04 (GitHub #153): a load with a draft window sizes the same arena for
  // the verify round instead -- `k+1` columns per lane, the head over every
  // column -- which bounds today's one-column round too, so both rounds
  // share it.
  try {
    model->decode_graph_scratch = std::make_unique<ninfer::DeviceArena>(sizes.decode_graph_scratch);
    model->decode_graph_token_ids = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes);
    model->decode_graph_slots = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * lanes);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_model_load: decode graph buffer allocation failed: ") + e.what());
    cudaStreamDestroy(model->stream);
    model->stream = nullptr;
    return -1;
  }

  // P5-04 (GitHub #153): the verify round's substrate, for either windowed
  // backend: GitHub #686, the load's base shape. GitHub #687: in buffers sized
  // for both of a load's shapes (`sizes.verify_round`); the extended shape's
  // view of them is made below, once the drafter's exist.
  if (draft_tokens > 0) {
    try {
      model->base_shape = build_verify_round(*topology, sizes.base_window, load.decode_lanes, sizes.verify_round);
      // GitHub #576 (ADR 0090): the round's draft rule. The sampled draft's
      // accept takes the same workspace the vendored one does.
      model->base_shape->sampled_draft = load.draft_rule == IGNIS_DRAFT_RULE_SAMPLED;
      model->base_shape->drafter_temperature_factor = load.drafter_temperature_factor;
      model->base_shape->drafter_top_p = load.drafter_top_p;
      if (model->base_shape->sampled_draft &&
          ignis_sampled_accept_workspace_bytes(static_cast<std::int32_t>(topology->vocab),
                                               static_cast<std::int32_t>(sizes.base_window),
                                               static_cast<std::int32_t>(load.decode_lanes)) >
              model->base_shape->accept_workspace->capacity()) {
        throw std::runtime_error("the sampled draft's accept outgrows the accept workspace");
      }
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: verify round allocation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
  }

  // P5-05 (GitHub #155): the drafter's round buffers (model_internal.h), as
  // the load's drafter has them (GitHub #619). `plan_load_sizes` has already
  // checked the vendored workspaces against their allowance.
  switch (model->drafter) {
  case IgnisDrafter::kDflash2:
    try {
      model->base_shape->features = std::make_unique<ninfer::DeviceBuffer>(sizes.drafter_features);
      model->base_shape->append_counts = std::make_unique<ninfer::DeviceBuffer>(sizes.drafter_append_counts);
      model->base_shape->drafter_scratch = std::make_unique<ninfer::DeviceArena>(sizes.drafter_scratch);
      // GitHub #576: under the sampled draft, the drafter's q from the draw
      // to the accept, one half of `drafter_draws` each.
      if (sizes.drafter_draws != 0) {
        model->base_shape->draft_candidates = std::make_unique<ninfer::DeviceBuffer>(sizes.drafter_draws / 2);
        model->base_shape->draft_q = std::make_unique<ninfer::DeviceBuffer>(sizes.drafter_draws / 2);
      }
      // GitHub #676: the host-mapped staging its verify rounds take, page-locked
      // here rather than while serving.
      model->base_shape->staging = std::make_unique<VerifyStaging>();
      // GitHub #687: the extended shape, a view of every buffer above with
      // host-mapped staging of its own (two 16 KiB slots and a ticket, no
      // device memory).
      if (sizes.extended_window != 0) {
        model->extended_shape =
            build_extended_shape(*topology, sizes.extended_window, sizes.extended_lanes, *model->base_shape);
      }
    } catch (const std::exception &e) {
      set_error(std::string("ignis_model_load: drafter round allocation failed: ") + e.what());
      cudaStreamDestroy(model->stream);
      model->stream = nullptr;
      return -1;
    }
    break;
  case IgnisDrafter::kMtp:
    // GitHub #620: the MTP head has no drafter round buffers; its buffers are
    // in the decode graph's scratch (`plan_load_sizes`).
  case IgnisDrafter::kHost:
  case IgnisDrafter::kNone:
    break;
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
  // GitHub #415: and the embedding pool, where the host vision residency
  // maps it.
  out_stats->host_mapped_bytes = model->host_mapped_bytes + model->vision_pool.host_mapped_bytes();
  out_stats->bound_tensor_count = model->bound_tensor_count;
  out_stats->vision_reserved_bytes = model->vision_reserved_bytes();
  out_stats->reserved = model->flash_next ? ignis::flash_next::reserved(*model) : reserved_of(*model);
  return 0;
}

extern "C" uint32_t ignis_model_vision_item_max_tokens(const struct ignis_model *model) {
  if (model == nullptr || model->vision_max_tokens == 0) {
    return 0;
  }
  return std::min(model->vision_item_max_tokens, model->max_context_tokens);
}

extern "C" uint32_t ignis_model_gdn_prefill(const struct ignis_model *model) {
  if (model == nullptr || model->flash_next) {
    return IGNIS_GDN_PREFILL_VENDORED;
  }
  return model->gdn_prefill_ours ? IGNIS_GDN_PREFILL_OURS : IGNIS_GDN_PREFILL_VENDORED;
}

extern "C" uint32_t ignis_model_hq_decode_attention(const struct ignis_model *model) {
  if (model == nullptr || model->flash_next) {
    return IGNIS_HQ_DECODE_ATTENTION_VENDORED;
  }
  return model->hq_decode_attention_ours ? IGNIS_HQ_DECODE_ATTENTION_OURS : IGNIS_HQ_DECODE_ATTENTION_VENDORED;
}

// GitHub #688: the rule an extended round's call takes (kernel/src/gqa_layer.cu `run_gqa_layer_batch`), at
// the shape's columns a lane: a load whose rounds take ignis's route is an hq-e8-2b load.
extern "C" uint32_t ignis_model_extended_attention(const struct ignis_model *model) {
  if (model == nullptr || model->flash_next || model->extended_shape == nullptr) {
    return IGNIS_EXTENDED_ATTENTION_NONE;
  }
  const auto columns = static_cast<std::int32_t>(model->extended_shape->window + 1);
  const bool serves = model->hq_decode_attention_ours &&
                      ignis::hq_decode_attention::serves_chunked(kIgnisGqaQHeads, columns, ninfer::DType::U8);
  return ignis::extended_attention::route_of(model->hq_decode_attention_ours, model->extended_attention, serves) ==
                 ignis::extended_attention::Route::kOurs
             ? IGNIS_EXTENDED_ATTENTION_OURS
             : IGNIS_EXTENDED_ATTENTION_VENDORED;
}

extern "C" uint32_t ignis_model_mtp_head_linear(const struct ignis_model *model) {
  if (model == nullptr || model->flash_next || model->speculative_backend != IGNIS_SPECULATIVE_MTP) {
    return IGNIS_MTP_HEAD_LINEAR_NONE;
  }
  return model->mtp_head_linear_ours ? IGNIS_MTP_HEAD_LINEAR_OURS : IGNIS_MTP_HEAD_LINEAR_VENDORED;
}

extern "C" uint32_t ignis_model_prefill_swiglu(const struct ignis_model *model) {
  // None where the load's prefill runs no W4A4 TMA product, the trellis body among them: the
  // decision the layout's answer below reports (`w4a4_activation_layout_route`).
  if (model == nullptr || model->flash_next || !model->w4a4_activation_layout.has_value()) {
    return IGNIS_PREFILL_SWIGLU_NONE;
  }
  switch (model->prefill_swiglu) {
  case ignis::prefill_swiglu::Route::Quant:
    return IGNIS_PREFILL_SWIGLU_QUANT;
  case ignis::prefill_swiglu::Route::Fused:
    return IGNIS_PREFILL_SWIGLU_FUSED;
  case ignis::prefill_swiglu::Route::Split:
    break;
  }
  return IGNIS_PREFILL_SWIGLU_SPLIT;
}

extern "C" uint32_t ignis_model_w4a4_activation_layout(const struct ignis_model *model) {
  if (model == nullptr || model->flash_next || !model->w4a4_activation_layout.has_value()) {
    return IGNIS_W4A4_ACTIVATION_LAYOUT_NONE;
  }
  return *model->w4a4_activation_layout == ignis::w4a4_activation_layout::Layout::Token
             ? IGNIS_W4A4_ACTIVATION_LAYOUT_TOKEN
             : IGNIS_W4A4_ACTIVATION_LAYOUT_KTILE;
}

extern "C" void ignis_model_free(struct ignis_model *model) {
  if (model != nullptr) {
    // GitHub #676: a verify round's commit may still be on the stream, reading
    // the buffers and writing the host-mapped staging freed below.
    if (model->stream != nullptr) {
      cudaStreamSynchronize(model->stream);
    }
    // P3-05 (GitHub #102, ADR 0019): destroy every captured decode graph
    // before the stream/scratch it was captured against goes away.
    for (auto &exec : model->decode_graph_exec) {
      if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
      }
    }
    // GitHub #687: each of a load's round shapes.
    for (IgnisVerifyRound *shape : {model->base_shape.get(), model->extended_shape.get()}) {
      if (shape == nullptr) {
        continue;
      }
      for (auto &exec : shape->graph_exec) {
        if (exec != nullptr) {
          cudaGraphExecDestroy(exec);
          exec = nullptr;
        }
      }
      // GitHub #676: and the drafter's append graphs.
      for (auto &exec : shape->append_exec) {
        if (exec != nullptr) {
          cudaGraphExecDestroy(exec);
          exec = nullptr;
        }
      }
    }
    // GitHub #674: the round timing's events, made by a capture under IGNIS_ROUND_TIMING=1.
    for (RoundTiming *timing : {&model->plain_timing, &model->verify_timing}) {
      for (std::uint64_t i = 0; i < RoundTiming::kRing; ++i) {
        for (cudaEvent_t event : {timing->pre[i], timing->post[i]}) {
          if (event != nullptr) {
            cudaEventDestroy(event);
          }
        }
      }
    }
    // GitHub #650: the descriptor slots stop serving the stream before it goes away.
    model->tma_descriptors.reset();
    // GitHub #652: and the stream's layout is no longer named, so a stream created at its address
    // starts from the default.
    model->w4a4_activation_layout_stream.reset();
    // GitHub #681: the gating's branch, idle once the stream is (every round joins it).
    model->gdn_gating_branch.destroy();
    if (model->stream != nullptr) {
      cudaStreamDestroy(model->stream);
    }
  }
  delete model;
}

extern "C" const char *ignis_model_last_error(void) {
  return g_last_error.c_str();
}
