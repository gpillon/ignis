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
// The `*_input_scale_divisor` objects (the W4A4 activation-quant path, G2)
// do not cross this ABI yet -- Rust binds and validates them against the
// artifact (ADR 0002), but the leaf's per-layer schema below only lists the
// fields the program layer consumes today.
//
// Style follows the ticket-04 leaf (device.cu): explicit pointers + sizes,
// int32 return codes (0 = ok, -1 = error), no C++ types across the boundary.

#include "ignis_model.h"

#include "model_internal.h"

#include "core/weight.h"

#include <cuda_runtime.h>

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

} // namespace

extern "C" int32_t ignis_model_load(const struct ignis_bound_tensor *tensors, uint64_t count,
                                     const struct ignis_topology *topology,
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
  // embed/norm/logits/argmax scratch at 27B geometry is ~505 KiB (vocab
  // dominates); headroom is cheap next to the 15+ GiB weight upload.
  constexpr std::size_t kScratchBytes = 2ull * 1024 * 1024;
  try {
    model->scratch = std::make_unique<ninfer::DeviceArena>(kScratchBytes);
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
