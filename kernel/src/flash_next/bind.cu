// ignis kernel leaf -- the Flash-Next binder (spec flash-next/04, GitHub
// #302; OURS, ADR 0043). See bind.h.

#include "bind.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ignis::flash_next {

namespace {

std::string shape_text(std::initializer_list<int64_t> shape) {
  std::string s = "[";
  for (int64_t d : shape) {
    s += (s.size() > 1 ? "," : "") + std::to_string(d);
  }
  return s + "]";
}

// The descriptors by name, and which the schema consumed.
class Schema {
 public:
  Schema(const ignis_bound_tensor *tensors, uint64_t count, std::string *error)
      : tensors_(tensors), used_(count, false), error_(error) {}

  bool index(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      if (tensors_[i].name == nullptr) {
        return fail("bound tensor " + std::to_string(i) + " has a null name");
      }
      if (!names_.emplace(tensors_[i].name, i).second) {
        return fail(std::string("duplicate bound tensor: ") + tensors_[i].name);
      }
    }
    return true;
  }

  // A linear weight [rows, cols], FP8 row-scale or BF16.
  bool linear(const std::string &name, int64_t rows, int64_t cols, Linear &out) {
    const ignis_bound_tensor *t = take(name, {rows, cols});
    if (t == nullptr) {
      return false;
    }
    if (t->qtype == IGNIS_QTYPE_FP8_E4M3FN_ROW_BF16S && t->layout == IGNIS_LAYOUT_ROW_SCALE) {
      out = {t->qdata, static_cast<int32_t>(rows), static_cast<int32_t>(cols), WeightFormat::Fp8RowScale};
      return true;
    }
    if (t->qtype == IGNIS_QTYPE_BF16_CTRL && t->layout == IGNIS_LAYOUT_CONTIGUOUS) {
      out = {t->qdata, static_cast<int32_t>(rows), static_cast<int32_t>(cols), WeightFormat::Bf16};
      return true;
    }
    return fail(name + " is neither FP8 row-scale nor BF16 (qtype " + std::to_string(t->qtype) +
                ", layout " + std::to_string(t->layout) + ")");
  }

  // A BF16 tensor of exactly `shape` (norms, vectors, the conv weights, the
  // router, the block-inject weights).
  bool bf16(const std::string &name, std::initializer_list<int64_t> shape, const void *&out) {
    const ignis_bound_tensor *t = take(name, shape);
    if (t == nullptr) {
      return false;
    }
    if (t->qtype != IGNIS_QTYPE_BF16_CTRL || t->layout != IGNIS_LAYOUT_CONTIGUOUS) {
      return fail(name + " is not BF16 (qtype " + std::to_string(t->qtype) + ")");
    }
    out = t->qdata;
    return true;
  }

  bool no_extras() {
    for (uint64_t i = 0; i < used_.size(); ++i) {
      if (!used_[i]) {
        return fail(std::string("extra bound tensor: ") + tensors_[i].name);
      }
    }
    return true;
  }

  bool fail(std::string message) {
    *error_ = "bind_flash_next: " + std::move(message);
    return false;
  }

 private:
  const ignis_bound_tensor *take(const std::string &name, std::initializer_list<int64_t> shape) {
    const auto it = names_.find(name);
    if (it == names_.end()) {
      fail("missing bound tensor: " + name);
      return nullptr;
    }
    const ignis_bound_tensor &t = tensors_[it->second];
    bool ok = t.ndim == shape.size();
    uint32_t i = 0;
    for (int64_t d : shape) {
      ok = ok && i < 4 && t.shape[i] == d;
      ++i;
    }
    if (!ok) {
      fail(name + " has an unexpected shape (want " + shape_text(shape) + ")");
      return nullptr;
    }
    used_[it->second] = true;
    return &t;
  }

  const ignis_bound_tensor *tensors_;
  std::vector<bool> used_;
  std::unordered_map<std::string, uint64_t> names_;
  std::string *error_;
};

bool bind_hc(Schema &s, const std::string &prefix, const Geometry &g, bool inject, HcWeights &w) {
  const int64_t hc = g.residual_width();
  return (!inject || s.bf16(prefix + ".block_inject_weight.weight", {g.streams, hc}, w.block_inject)) &&
         s.bf16(prefix + ".hc_norm.weight", {hc}, w.hc_norm) &&
         s.linear(prefix + ".input_mix_weight_down.weight", g.hc_rank, hc, w.mix_down) &&
         s.linear(prefix + ".input_mix_weight_up.weight", hc, g.hc_rank, w.mix_up);
}

bool bind_gdn(Schema &s, const std::string &p, const Geometry &g, GdnWeights &w) {
  const int64_t key_dim = static_cast<int64_t>(g.gdn_qk_heads) * g.gdn_head_dim;
  const int64_t value_dim = static_cast<int64_t>(g.gdn_value_heads) * g.gdn_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  return s.linear(p + "linear_attn.in_proj_qkv.weight", conv_dim, g.hidden, w.in_proj_qkv) &&
         s.linear(p + "linear_attn.in_proj_z.weight", value_dim, g.hidden, w.in_proj_z) &&
         s.linear(p + "linear_attn.in_proj_a.weight", g.gdn_value_heads, g.hidden, w.in_proj_a) &&
         s.linear(p + "linear_attn.in_proj_b.weight", g.gdn_value_heads, g.hidden, w.in_proj_b) &&
         s.linear(p + "linear_attn.out_proj.weight", g.hidden, value_dim, w.out_proj) &&
         s.bf16(p + "linear_attn.conv1d.weight", {conv_dim, 1, g.gdn_conv_kernel}, w.conv) &&
         s.bf16(p + "linear_attn.A_log", {g.gdn_value_heads}, w.a_log) &&
         s.bf16(p + "linear_attn.dt_bias", {g.gdn_value_heads}, w.dt_bias) &&
         s.bf16(p + "linear_attn.norm.weight", {g.gdn_head_dim}, w.norm);
}

bool bind_qsa(Schema &s, const std::string &p, const Geometry &g, QsaWeights &w) {
  const int64_t q_width = static_cast<int64_t>(g.q_heads) * g.head_dim;
  const int64_t kv_width = static_cast<int64_t>(g.kv_heads) * g.head_dim;
  const int64_t index_rows =
      static_cast<int64_t>(g.indexer_heads + g.indexer_kv_heads) * g.indexer_head_dim;
  return s.linear(p + "self_attn.q_proj.weight", 2 * q_width, g.hidden, w.q_proj) &&
         s.linear(p + "self_attn.k_proj.weight", kv_width, g.hidden, w.k_proj) &&
         s.linear(p + "self_attn.v_proj.weight", kv_width, g.hidden, w.v_proj) &&
         s.linear(p + "self_attn.o_proj.weight", g.hidden, q_width, w.o_proj) &&
         s.bf16(p + "self_attn.q_norm.weight", {g.head_dim}, w.q_norm) &&
         s.bf16(p + "self_attn.k_norm.weight", {g.head_dim}, w.k_norm) &&
         s.linear(p + "self_attn.indexer.index_qk_proj.weight", index_rows, g.hidden, w.indexer.qk_proj) &&
         s.bf16(p + "self_attn.indexer.q_layernorm.weight", {g.indexer_head_dim}, w.indexer.q_norm) &&
         s.bf16(p + "self_attn.indexer.k_layernorm.weight", {g.indexer_head_dim}, w.indexer.k_norm);
}

bool bind_moe(Schema &s, const std::string &p, const Geometry &g, MoeWeights &w) {
  Linear gate, up, down;
  if (!s.bf16(p + "mlp.gate.weight", {g.experts, g.hidden}, w.router) ||
      !s.linear(p + "mlp.shared_expert.gate_proj.weight", g.shared_intermediate, g.hidden, gate) ||
      !s.linear(p + "mlp.shared_expert.up_proj.weight", g.shared_intermediate, g.hidden, up) ||
      !s.linear(p + "mlp.shared_expert.down_proj.weight", g.hidden, g.shared_intermediate, down) ||
      !s.bf16(p + "mlp.shared_expert_gate.weight", {1, g.hidden}, w.shared_expert_gate)) {
    return false;
  }
  // kern's shared expert runs FP8 row-scale weights only.
  for (const Linear *l : {&gate, &up, &down}) {
    if (l->format != WeightFormat::Fp8RowScale) {
      return s.fail(p + "mlp.shared_expert: the shared expert runs FP8 row-scale weights only");
    }
  }
  w.shared_gate = gate.data;
  w.shared_up = up.data;
  w.shared_down = down.data;
  return true;
}

bool bind_ngram(Schema &s, const std::string &p, const Geometry &g, NgramWeights &w) {
  const int64_t hc = g.residual_width();
  return s.bf16(p + "ple.conv1d.weight", {hc, 1, g.ngram_conv_kernel}, w.conv) &&
         s.linear(p + "ple.key_proj.weight", hc, g.ngram_embed_dim, w.key_proj) &&
         s.bf16(p + "ple.norm_conv.weight", {hc}, w.norm_conv) &&
         s.bf16(p + "ple.norm_key.weight", {hc}, w.norm_key) &&
         s.bf16(p + "ple.norm_query.weight", {hc}, w.norm_query) &&
         s.linear(p + "ple.value_proj.weight", g.hidden, g.ngram_embed_dim, w.value_proj);
}

// The geometry facts the schema relies on; a topology that breaks one is not
// Flash-Next's, whatever its family says.
std::string geometry_fault(const Geometry &g) {
  if (g.streams <= 0 || g.hc_rank <= 0) return "no hyper-connection streams";
  if (g.experts <= 0 || g.experts_per_token <= 0 || g.shared_intermediate <= 0) return "no MoE block";
  if (g.indexer_heads <= 0 || g.indexer_head_dim <= 0 || g.compress_ratio <= 0) return "no indexer";
  if (g.ngram_heads <= 0 || g.ngram_embed_dim <= 0 || g.ngram_layer < 0 || g.ngram_layer >= g.layers) {
    return "no n-gram embedding inside the layers";
  }
  if (g.gdn_head_dim <= 0 || g.gdn_qk_heads <= 0) return "no GDN head geometry";
  return "";
}

}  // namespace

std::unique_ptr<Weights> bind_flash_next(const ignis_bound_tensor *tensors, uint64_t count,
                                         const ignis_topology &topology, std::string *error) {
  std::string scratch_error;
  if (error == nullptr) {
    error = &scratch_error;
  }
  Schema s(tensors, count, error);
  const Geometry g = Geometry::from(topology);
  if (const std::string fault = geometry_fault(g); !fault.empty()) {
    s.fail("not a Flash-Next topology: " + fault);
    return nullptr;
  }
  if (topology.num_layers > 0 && topology.layer_kinds == nullptr) {
    s.fail("topology.layer_kinds is null");
    return nullptr;
  }
  if (!s.index(count)) {
    return nullptr;
  }
  auto w = std::make_unique<Weights>();
  if (!s.linear("embed_tokens.weight", g.vocab, g.hidden, w->embed) ||
      !s.linear("lm_head.weight", g.vocab, g.hidden, w->head) ||
      !bind_hc(s, "hyper_connection_mixer", g, /*inject=*/false, w->final_mixer)) {
    return nullptr;
  }
  w->layers.resize(static_cast<std::size_t>(g.layers));
  for (int32_t l = 0; l < g.layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    LayerWeights &layer = w->layers[static_cast<std::size_t>(l)];
    layer.attention = topology.layer_kinds[l] == IGNIS_LAYER_GQA;
    if (!bind_hc(s, p + "attn_hyper_connection", g, true, layer.attn_hc) ||
        !bind_hc(s, p + "mlp_hyper_connection", g, true, layer.mlp_hc) ||
        !(layer.attention ? bind_qsa(s, p, g, layer.qsa) : bind_gdn(s, p, g, layer.gdn)) ||
        !bind_moe(s, p, g, layer.moe) ||
        (l == g.ngram_layer && !bind_ngram(s, p, g, w->ngram))) {
      return nullptr;
    }
  }
  if (!s.no_extras()) {
    return nullptr;
  }
  return w;
}

}  // namespace ignis::flash_next
