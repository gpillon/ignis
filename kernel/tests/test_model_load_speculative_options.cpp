// P5-02 (GitHub #150): `ignis_model_load`'s speculation option -- OURS, not
// vendored.
//
// Two things are pinned here, both host-only:
//
// 1. The options struct is validated before anything is bound (ADR 0016): an
//    unrecognized size, an unknown backend, a draft window outside 1..7, and
//    a window with no backend are each refused by name.
// 2. The leaf's drafter schema: under IGNIS_SPECULATIVE_DFLASH2 every one of
//    the 66 `dflash2/*` tensors is asked for, and a missing or mis-shaped one
//    fails the load naming it; without the option the same tensors are extras.
//
// Host-only by construction: a topology with no decoder layers needs three
// text tensors, and every check below fails at binding -- ahead of the
// stream and the scratch reservation (the drafter's prefill scratch
// included), which are the load's only device calls; the drafter's window
// lives in the sequence pool (P5-03, GitHub #152).
// The descriptors therefore carry shapes and no planes.
// A load that *succeeds* would touch the device, so no arm here lets one;
// the "all 66 bound" arm proves the binding completed by failing on an
// extra tensor appended after them instead.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so nothing here can
// read as a skip.

#include "ignis_model.h"
#include "ignis_seq.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
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

constexpr uint64_t kHidden = 5120;
constexpr uint64_t kVocab = 248320;
constexpr uint64_t kFfn = 17408;

ignis_topology no_layer_topology() {
  ignis_topology topology{};
  topology.num_layers       = 0;
  topology.layer_kinds      = nullptr;
  topology.hidden           = kHidden;
  topology.vocab            = kVocab;
  topology.ffn_intermediate = kFfn;
  topology.gdn_num_layers   = 1;
  return topology;
}

struct Named {
  std::string name;
  std::vector<int32_t> shape;
};

// The three text tensors a no-layer topology binds, then the drafter's 66 in
// the artifact's order (the reference's qwen3.8-27b-artifact.md §15.2).
std::vector<Named> text_tensors() {
  const auto h = static_cast<int32_t>(kHidden);
  const auto v = static_cast<int32_t>(kVocab);
  return {{"text/token_embedding", {v, h}}, {"text/final_norm", {h}}, {"text/output_head", {v, h}}};
}

std::vector<Named> dflash2_tensors() {
  const auto h = static_cast<int32_t>(kHidden);
  const auto v = static_cast<int32_t>(kVocab);
  const auto f = static_cast<int32_t>(kFfn);
  std::vector<Named> out{{"dflash2/feature_projection", {h, 5 * h}}, {"dflash2/context_norm", {h}}};
  for (int l = 0; l < 5; ++l) {
    const std::string p = "dflash2/layers/" + std::to_string(l) + "/";
    for (Named n : std::initializer_list<Named>{
             {"input_norm", {h}},
             {"attention/query_key_value", {6144, h}},
             {"attention/query_norm", {128}},
             {"attention/key_norm", {128}},
             {"attention/output", {h, 4096}},
             {"attention/conv_base", {2, 2, h}},
             {"attention/conv_proj", {1280, h}},
             {"post_attention_norm", {h}},
             {"mlp/gate_up", {2 * f, h}},
             {"mlp/down", {h, f}},
             {"mlp/conv_base", {2, 2, h}},
             {"mlp/conv_proj", {1280, h}},
         }) {
      n.name = p + n.name;
      out.push_back(n);
    }
  }
  out.push_back({"dflash2/final_norm", {h}});
  out.push_back({"dflash2/selector/hidden", {256, h}});
  out.push_back({"dflash2/selector/predecessor", {v, 256}});
  out.push_back({"dflash2/selector/successor", {v, 256}});
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

// Loads `named` under `options` and returns the leaf's last-error message.
// Every arm here must fail -- a success would have reached the device.
std::string load_error(const std::vector<Named> &named, const ignis_model_load_options *options) {
  const ignis_topology topology = no_layer_topology();
  std::vector<ignis_bound_tensor> tensors;
  for (const Named &n : named) {
    tensors.push_back(descriptor(n));
  }
  if (tensors.empty()) {
    tensors.push_back(ignis_bound_tensor{});
  }
  ignis_model *model = nullptr;
  const int32_t rc = ignis_model_load(tensors.data(), named.size(), &topology,
                                      /*prefill_chunk_tokens=*/128, /*max_context_tokens=*/128,
                                      IGNIS_KV_FORMAT_BF16, options, &model);
  if (rc == 0 || model != nullptr) {
    std::fprintf(stderr, "FATAL: this load must fail before any device work\n");
    if (model != nullptr) { ignis_model_free(model); }
    std::exit(EXIT_FAILURE);
  }
  return ignis_model_last_error();
}

ignis_model_load_options dflash2(uint32_t draft_tokens) {
  ignis_model_load_options options{};
  options.size = sizeof(options);
  options.speculative_backend = IGNIS_SPECULATIVE_DFLASH2;
  options.draft_tokens = draft_tokens;
  return options;
}

bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

std::vector<Named> concat(std::vector<Named> a, const std::vector<Named> &b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

} // namespace

int main() {
  const std::vector<Named> text = text_tensors();
  const std::vector<Named> drafter = dflash2_tensors();
  check(drafter.size() == 66, "the drafter schema is 66 tensors");

  // --- 1. the options struct ------------------------------------------------
  {
    ignis_model_load_options bad = dflash2(7);
    bad.size = sizeof(bad) + 4;
    const std::string m = load_error({}, &bad);
    check(contains(m, "options.size"), "an unrecognized options size is refused: " + m);
  }
  for (const int32_t backend : {-1, 2, 9}) {
    ignis_model_load_options bad = dflash2(7);
    bad.speculative_backend = backend;
    const std::string m = load_error({}, &bad);
    check(contains(m, "speculative_backend") && contains(m, std::to_string(backend)),
          "backend " + std::to_string(backend) + " is refused by name: " + m);
  }
  for (const uint32_t window : {0u, 8u, 15u}) {
    const ignis_model_load_options bad = dflash2(window);
    const std::string m = load_error({}, &bad);
    check(contains(m, "draft_tokens") && contains(m, "1..7") && contains(m, std::to_string(window)),
          "window " + std::to_string(window) + " is refused naming the range: " + m);
  }
  {
    ignis_model_load_options bad{};
    bad.size = sizeof(bad);
    bad.draft_tokens = 3;
    const std::string m = load_error({}, &bad);
    check(contains(m, "draft_tokens") && contains(m, "speculative_backend"),
          "a window with no backend is refused: " + m);
  }
  // The control arm for the checks above: every valid window gets past them
  // and fails on binding instead.
  for (uint32_t window = 1; window <= IGNIS_DFLASH2_MAX_DRAFT_TOKENS; ++window) {
    const ignis_model_load_options ok = dflash2(window);
    const std::string m = load_error({}, &ok);
    check(contains(m, "text/token_embedding"),
          "window " + std::to_string(window) + " reaches the binding: " + m);
  }

  // --- 2. the drafter schema ------------------------------------------------
  const ignis_model_load_options spec = dflash2(7);
  {
    // All 66 bound: the load gets past them and fails on the one extra.
    const std::string m = load_error(concat(concat(text, drafter), {{"dflash2/extra", {1}}}), &spec);
    check(contains(m, "extra bound tensor: dflash2/extra"), "every dflash2 tensor binds: " + m);
  }
  for (std::size_t drop = 0; drop < drafter.size(); ++drop) {
    std::vector<Named> partial = text;
    for (std::size_t i = 0; i < drafter.size(); ++i) {
      if (i != drop) { partial.push_back(drafter[i]); }
    }
    const std::string m = load_error(partial, &spec);
    check(contains(m, "missing bound tensor: " + drafter[drop].name),
          "a missing " + drafter[drop].name + " is named: " + m);
  }
  for (std::size_t bend = 0; bend < drafter.size(); ++bend) {
    std::vector<Named> bent = drafter;
    bent[bend].shape[0] += 1;
    const std::string m = load_error(concat(text, bent), &spec);
    check(contains(m, bent[bend].name + " has an unexpected shape"),
          "a mis-shaped " + bent[bend].name + " is named: " + m);
  }
  {
    // Without the option the leaf asks for none of them.
    const std::string m = load_error(concat(text, drafter), nullptr);
    check(contains(m, "extra bound tensor: dflash2/feature_projection"),
          "without the option the drafter's tensors are extras: " + m);
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "model load speculative options test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("model load speculative options test: ok\n");
  return 0;
}
