// GitHub #302 (spec flash-next/04): the GDN layer count and the GDN value-head
// count are two topology numbers -- OURS, not vendored.
//
// The 27B has 48 GDN layers and 48 GDN value heads, so a binder that sized the
// per-head GDN parameters (`gdn/a_log`, `gdn/dt_bias`, the gated norm's width)
// by the layer count bound it anyway. Flash-Next has 36 GDN layers of 48 value
// heads, and the same binder refused it. This pins the binder to the
// value-head count (`ignis_topology::gdn_value_heads`), with the layer count
// checked against the layer kinds, and the family's refusals beside them.
//
// Host-only by construction: `ignis_model_plan_reservations` allocates nothing,
// and every arm below fails at binding -- the arms that bind every layer prove
// it by failing on an extra tensor appended after them. The descriptors carry
// shapes and no planes.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so nothing here can
// read as a skip.

#include "ignis_model.h"
#include "ignis_seq.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

constexpr int32_t kGdnHeadDim = 128;
constexpr int32_t kConvKernel = 4;
constexpr const char *kExtra = "test/extra";

// A GDN geometry: the layer pattern (every `attention_every`-th layer is GQA,
// 0 = none) and the GDN head counts, at the 27B program's per-layer schema.
struct Geometry {
  uint32_t num_layers;
  uint32_t attention_every;
  uint64_t gdn_layers;
  uint64_t value_heads;
  uint64_t key_heads;
  uint64_t hidden;
};

// Flash-Next's: 48 layers as 12 x (3 GDN + 1 QSA), so 36 GDN layers, each of
// 16 key and 48 value heads of 128, at hidden 2560 (the checkpoint's config).
constexpr Geometry kFlashNext{48, 4, 36, 48, 16, 2560};

// A small one where the layer count divides the value-head width, so a binder
// that confuses the two gets past the divisibility check and fails on shapes.
constexpr Geometry kTwoLayersFourHeads{2, 0, 2, 4, 2, 256};

constexpr uint64_t kVocab = 1024;
constexpr uint64_t kQHeads = 24;
constexpr uint64_t kKvHeads = 2;
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kFfn = 640;

struct Topology {
  std::vector<int32_t> kinds;
  ignis_topology raw{};
};

Topology topology_of(const Geometry &g) {
  Topology t;
  for (uint32_t i = 0; i < g.num_layers; ++i) {
    const bool attention = g.attention_every != 0 && (i + 1) % g.attention_every == 0;
    t.kinds.push_back(attention ? IGNIS_LAYER_GQA : IGNIS_LAYER_GDN);
  }
  t.raw.num_layers       = g.num_layers;
  t.raw.layer_kinds      = t.kinds.data();
  t.raw.hidden           = g.hidden;
  t.raw.vocab            = kVocab;
  t.raw.num_q_heads      = kQHeads;
  t.raw.num_kv_heads     = kKvHeads;
  t.raw.head_dim         = kHeadDim;
  t.raw.rotary_dim       = 64;
  t.raw.rope_theta       = 1e7;
  t.raw.gdn_state_rows   = g.value_heads * kGdnHeadDim;
  t.raw.gdn_state_cols   = g.key_heads * kGdnHeadDim;
  t.raw.gdn_num_layers   = g.gdn_layers;
  t.raw.gdn_value_heads  = g.value_heads;
  t.raw.gdn_q_width      = g.key_heads * kGdnHeadDim;
  t.raw.gdn_z_width      = g.value_heads * kGdnHeadDim;
  t.raw.gdn_ab_width     = 2 * g.value_heads;
  t.raw.ffn_intermediate = kFfn;
  t.raw.rms_norm_eps     = 1e-6f;
  return t;
}

struct Named {
  std::string name;
  std::vector<int32_t> shape;
};

// The tensors the 27B program's schema asks of `g`, with every per-head GDN
// parameter shaped by the value heads, as the checkpoint stores them.
std::vector<Named> tensors_of(const Geometry &g) {
  const auto h = static_cast<int32_t>(g.hidden);
  const auto v = static_cast<int32_t>(kVocab);
  const auto f = static_cast<int32_t>(kFfn);
  const auto heads = static_cast<int32_t>(g.value_heads);
  const auto q = static_cast<int32_t>(kQHeads * kHeadDim);
  const auto kv = static_cast<int32_t>(kKvHeads * kHeadDim);
  const auto state_rows = heads * kGdnHeadDim;
  const auto key_width = static_cast<int32_t>(g.key_heads) * kGdnHeadDim;
  const auto conv_channels = 2 * key_width + state_rows;
  std::vector<Named> out{{"text/token_embedding", {v, h}},
                         {"text/final_norm", {h}},
                         {"text/output_head", {v, h}}};
  for (uint32_t i = 0; i < g.num_layers; ++i) {
    const std::string p = "text/layers/" + std::to_string(i) + "/";
    const bool attention = g.attention_every != 0 && (i + 1) % g.attention_every == 0;
    out.push_back({p + "input_norm", {h}});
    if (attention) {
      out.push_back({p + "attention/query_key_gate_value", {2 * q + 2 * kv, h}});
      out.push_back({p + "attention/query_norm", {static_cast<int32_t>(kHeadDim)}});
      out.push_back({p + "attention/key_norm", {static_cast<int32_t>(kHeadDim)}});
      out.push_back({p + "attention/output", {h, q}});
    } else {
      out.push_back({p + "gdn/a_log", {heads}});
      out.push_back({p + "gdn/dt_bias", {heads}});
      out.push_back({p + "gdn/convolution", {kConvKernel, conv_channels}});
      out.push_back({p + "gdn/a_b_projection", {2 * heads, h}});
      out.push_back({p + "gdn/query_key_value_z", {conv_channels + state_rows, h}});
      out.push_back({p + "gdn/norm", {kGdnHeadDim}});
      out.push_back({p + "gdn/output", {h, state_rows}});
    }
    out.push_back({p + "post_attention_norm", {h}});
    out.push_back({p + "mlp/gate_up", {2 * f, h}});
    out.push_back({p + "mlp/down", {h, f}});
  }
  return out;
}

ignis_bound_tensor descriptor(const Named &n) {
  ignis_bound_tensor t{};
  t.name = n.name.c_str();
  t.ndim = static_cast<uint32_t>(n.shape.size());
  for (std::size_t i = 0; i < 4; ++i) {
    t.shape[i] = i < n.shape.size() ? n.shape[i] : 1;
    t.padded_shape[i] = t.shape[i];
  }
  return t;
}

// Plans `named` against `topology` and returns the leaf's last-error message.
// Every arm here must fail at binding.
std::string plan_error(const ignis_topology &topology, const std::vector<Named> &named) {
  std::vector<ignis_bound_tensor> tensors;
  for (const Named &n : named) {
    tensors.push_back(descriptor(n));
  }
  ignis_model_reservations out{};
  const int32_t rc = ignis_model_plan_reservations(tensors.data(), tensors.size(), &topology,
                                                   /*prefill_chunk_tokens=*/128,
                                                   /*max_context_tokens=*/128,
                                                   IGNIS_KV_FORMAT_BF16, /*options=*/nullptr, &out);
  if (rc == 0) {
    std::fprintf(stderr, "FATAL: every plan here must fail at binding\n");
    std::exit(EXIT_FAILURE);
  }
  return ignis_model_last_error();
}

// A topology whose per-head GDN parameters are shaped by the value heads binds
// every layer: the only complaint left is the extra tensor after them.
void binds_every_layer(const Geometry &g, const std::string &label) {
  const Topology topology = topology_of(g);
  std::vector<Named> named = tensors_of(g);
  named.push_back({kExtra, {1}});
  const std::string message = plan_error(topology.raw, named);
  check(message.find(std::string("extra bound tensor: ") + kExtra) != std::string::npos,
        label + ": every layer binds and only the extra tensor is refused: got \"" + message +
            "\"");
}

// A topology the leaf refuses before binding anything, with a message holding
// every one of `wanted`.
void refused(const Topology &topology, const std::vector<std::string> &wanted,
             const std::string &label) {
  std::vector<Named> named = tensors_of(kFlashNext);
  const std::string message = plan_error(topology.raw, named);
  for (const std::string &part : wanted) {
    check(message.find(part) != std::string::npos,
          label + ": refused naming \"" + part + "\": got \"" + message + "\"");
  }
}

} // namespace

int main() {
  binds_every_layer(kFlashNext, "36 GDN layers of 48 value heads (Flash-Next)");
  binds_every_layer(kTwoLayersFourHeads, "2 GDN layers of 4 value heads");

  // The inverse conflation: the value heads handed over as the layer count.
  Topology swapped = topology_of(kFlashNext);
  swapped.raw.gdn_num_layers = kFlashNext.value_heads;
  refused(swapped, {"gdn_num_layers (48)", "GDN layer count (36)"}, "48 GDN layers claimed of 36");

  Topology uneven = topology_of(kFlashNext);
  uneven.raw.gdn_value_heads = 5;
  refused(uneven, {"gdn_state_rows is not a multiple of gdn_value_heads"}, "5 value heads");

  Topology headless = topology_of(kFlashNext);
  headless.raw.gdn_value_heads = 0;
  refused(headless, {"gdn_value_heads must be positive"}, "no value heads");

  // ADR 0043: the family names the program. Flash-Next's is not built yet,
  // and a 27B topology carries none of its blocks.
  Topology flash_next = topology_of(kFlashNext);
  flash_next.raw.family = IGNIS_MODEL_FAMILY_FLASH_NEXT;
  refused(flash_next, {"Flash-Next program is not built yet"}, "the Flash-Next family");

  Topology unknown = topology_of(kFlashNext);
  unknown.raw.family = 7;
  refused(unknown, {"family 7"}, "an unknown family");

  Topology dense_with_experts = topology_of(kFlashNext);
  dense_with_experts.raw.moe.num_experts = 512;
  refused(dense_with_experts, {"carries no MoE block"}, "a 27B topology with experts");

  Topology dense_with_ngram = topology_of(kFlashNext);
  dense_with_ngram.raw.ngram.layer = 1;
  refused(dense_with_ngram, {"carries no n-gram block"}, "a 27B topology with an n-gram layer");

  if (g_failed != 0) {
    std::fprintf(stderr, "model load GDN geometry test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("model load GDN geometry test: ok\n");
  return 0;
}
