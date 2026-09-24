// GitHub #177: `ignis_model_load`'s vision envelope -- OURS, not vendored.
//
// Two things are pinned here, both host-only:
//
// 1. The envelope is validated before anything is bound: one past
//    IGNIS_VISION_MAX_TOKENS_LIMIT is refused by name, the limit itself is not.
// 2. The leaf's vision schema: with an envelope every one of the 333
//    `vision/*` tensors is asked for, and a missing or mis-shaped one fails the
//    load naming it; without it the same tensors are extras.
// 3. GitHub #195: the envelope and a speculative backend are independent
//    options -- a load asking for both reaches the binding and asks for both
//    scopes, where #178 refused it outright.
// 4. The item bound: `vision_item_max_tokens` sizes the encoder's workspace
//    for one item of that many merged tokens (never past the envelope) and
//    leaves the embedding pool at the envelope's floor; without an envelope
//    it is refused. Planned through `ignis_model_plan_reservations`, which
//    binds and sizes but never reaches the device.
//
// Host-only by construction, the same way as
// test_model_load_speculative_options.cpp: a topology with no decoder layers
// needs three text tensors, and every check below fails at binding -- ahead
// of the stream, the scratch and the vision reservation, which are the load's
// only device calls. The "all 333 bound" arm proves the binding completed by
// failing on an extra tensor appended after them.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE.

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

std::vector<Named> text_tensors() {
  const auto h = static_cast<int32_t>(kHidden);
  const auto v = static_cast<int32_t>(kVocab);
  return {{"text/token_embedding", {v, h}}, {"text/final_norm", {h}}, {"text/output_head", {v, h}}};
}

// The tower's 333 tensors in the artifact's order (the reference's
// qwen3.8-27b-artifact.md, vision section).
std::vector<Named> vision_tensors() {
  const auto out = static_cast<int32_t>(kHidden);
  std::vector<Named> t{{"vision/patch_embedding", {1152, 1536}},
                       {"vision/patch_embedding_bias", {1152}},
                       {"vision/position_embedding", {2304, 1152}}};
  for (int b = 0; b < 27; ++b) {
    const std::string p = "vision/layers/" + std::to_string(b) + "/";
    for (Named n : std::initializer_list<Named>{
             {"attention/qkv", {3456, 1152}},
             {"attention/qkv_bias", {3456}},
             {"attention/output", {1152, 1152}},
             {"attention/output_bias", {1152}},
             {"mlp/fc1", {4304, 1152}},
             {"mlp/fc1_bias", {4304}},
             {"mlp/fc2", {1152, 4304}},
             {"mlp/fc2_bias", {1152}},
             {"norm1/weight", {1152}},
             {"norm1/bias", {1152}},
             {"norm2/weight", {1152}},
             {"norm2/bias", {1152}},
         }) {
      n.name = p + n.name;
      t.push_back(n);
    }
  }
  t.push_back({"vision/merger/fc1", {4608, 4608}});
  t.push_back({"vision/merger/fc1_bias", {4608}});
  t.push_back({"vision/merger/fc2", {out, 4608}});
  t.push_back({"vision/merger/fc2_bias", {out}});
  t.push_back({"vision/merger/norm/weight", {1152}});
  t.push_back({"vision/merger/norm/bias", {1152}});
  return t;
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

ignis_model_load_options vision(uint32_t max_tokens) {
  ignis_model_load_options options{};
  options.size = sizeof(options);
  options.speculative_backend = IGNIS_SPECULATIVE_NONE;
  options.vision_max_tokens = max_tokens;
  return options;
}

// Plans `named` under `options` at a serving-sized context, so the envelope
// is never capped by it. Every arm here must succeed: a plan binds but never
// touches the device.
ignis_model_reservations plan(const std::vector<Named> &named, const ignis_model_load_options &options) {
  const ignis_topology topology = no_layer_topology();
  std::vector<ignis_bound_tensor> tensors;
  for (const Named &n : named) {
    tensors.push_back(descriptor(n));
  }
  ignis_model_reservations out{};
  const int32_t rc = ignis_model_plan_reservations(tensors.data(), named.size(), &topology,
                                                   /*prefill_chunk_tokens=*/128,
                                                   /*max_context_tokens=*/262144,
                                                   IGNIS_KV_FORMAT_BF16, &options, &out);
  if (rc != 0) {
    std::fprintf(stderr, "FATAL: this plan must succeed: %s\n", ignis_model_last_error());
    std::exit(EXIT_FAILURE);
  }
  return out;
}

ignis_model_load_options vision_items(uint32_t max_tokens, uint32_t item_max_tokens) {
  ignis_model_load_options options = vision(max_tokens);
  options.vision_item_max_tokens = item_max_tokens;
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
  const std::vector<Named> tower = vision_tensors();
  check(tower.size() == 333, "the vision schema is 333 tensors");

  // --- 1. the envelope --------------------------------------------------------
  {
    const ignis_model_load_options bad = vision(IGNIS_VISION_MAX_TOKENS_LIMIT + 1);
    const std::string m = load_error({}, &bad);
    check(contains(m, "vision_max_tokens") &&
              contains(m, std::to_string(IGNIS_VISION_MAX_TOKENS_LIMIT + 1)),
          "an envelope past the limit is refused by name: " + m);
  }
  for (const uint32_t max_tokens : {1u, 32768u, IGNIS_VISION_MAX_TOKENS_LIMIT}) {
    const ignis_model_load_options ok = vision(max_tokens);
    const std::string m = load_error({}, &ok);
    check(contains(m, "text/token_embedding"),
          "envelope " + std::to_string(max_tokens) + " reaches the binding: " + m);
  }

  // --- 2. the vision schema ---------------------------------------------------
  const ignis_model_load_options on = vision(32768);
  {
    const std::string m = load_error(concat(concat(text, tower), {{"vision/extra", {1}}}), &on);
    check(contains(m, "extra bound tensor: vision/extra"), "every vision tensor binds: " + m);
  }
  for (std::size_t drop = 0; drop < tower.size(); ++drop) {
    std::vector<Named> partial = text;
    for (std::size_t i = 0; i < tower.size(); ++i) {
      if (i != drop) { partial.push_back(tower[i]); }
    }
    const std::string m = load_error(partial, &on);
    check(contains(m, "missing bound tensor: " + tower[drop].name),
          "a missing " + tower[drop].name + " is named: " + m);
  }
  for (std::size_t bend = 0; bend < tower.size(); ++bend) {
    std::vector<Named> bent = tower;
    bent[bend].shape[0] += 1;
    const std::string m = load_error(concat(text, bent), &on);
    check(contains(m, bent[bend].name + " has an unexpected shape"),
          "a mis-shaped " + bent[bend].name + " is named: " + m);
  }
  {
    // Without an envelope the leaf asks for none of them -- a NULL options
    // pointer and an explicit 0 alike.
    const std::string m = load_error(concat(text, tower), nullptr);
    check(contains(m, "extra bound tensor: vision/patch_embedding"),
          "without the option the vision tensors are extras: " + m);
    const ignis_model_load_options off = vision(0);
    const std::string m0 = load_error(concat(text, tower), &off);
    check(contains(m0, "extra bound tensor: vision/patch_embedding"),
          "with a zero envelope the vision tensors are extras: " + m0);
  }

  // --- 3. vision with speculation (GitHub #195) -------------------------------
  // #178 fenced the two options apart until the verify round learned the
  // sequence's rope delta. They are two independent scopes again: neither
  // backend is refused ahead of binding, DFlash2's load asks for the drafter
  // beside the tower, and a windowed load still asks for the whole tower.
  {
    ignis_model_load_options both = vision(32768);
    both.speculative_backend = IGNIS_SPECULATIVE_DFLASH2;
    both.draft_tokens = 4;
    const std::string m = load_error(concat(text, tower), &both);
    check(contains(m, "missing bound tensor: dflash2/"),
          "vision with DFlash2 binds the drafter beside the tower: " + m);
  }
  {
    ignis_model_load_options both = vision(32768);
    both.speculative_backend = IGNIS_SPECULATIVE_VERIFY_ONLY;
    both.draft_tokens = 4;
    const std::string m = load_error(text, &both);
    check(contains(m, "missing bound tensor: vision/patch_embedding"),
          "vision with the verify-only backend still asks for the tower: " + m);
  }

  // --- 4. the item bound ------------------------------------------------------
  // The envelope caps a request's vision tokens, but the encoder only ever
  // holds one item, and the processor's `smart_resize` caps an item at its
  // `longest_edge` over 32 x 32 pixels a token (16,384 for this artifact) --
  // half the default envelope. With `vision_item_max_tokens` the encoder's
  // workspace is sized for one item of that bound, exactly as a load whose
  // whole envelope were that small, while the embedding pool keeps the
  // envelope's floor: what an item costs to encode is not what a request may
  // hold.
  {
    const std::vector<Named> both = concat(text, tower);
    const ignis_model_reservations envelope = plan(both, vision(32768));
    const ignis_model_reservations small = plan(both, vision(16384));
    const ignis_model_reservations bounded = plan(both, vision_items(32768, 16384));
    check(bounded.workspace_bytes == small.workspace_bytes,
          "an item bound sizes the workspace like an envelope that small");
    check(bounded.workspace_bytes < envelope.workspace_bytes,
          "an item bound below the envelope shrinks the workspace");
    check(bounded.media_embedding_bytes == envelope.media_embedding_bytes,
          "an item bound leaves the embedding pool at the envelope's floor");

    check(plan(both, vision_items(32768, 0)).workspace_bytes == envelope.workspace_bytes,
          "no item bound is the envelope");
    const ignis_model_reservations wide = plan(both, vision(8192));
    check(plan(both, vision_items(8192, 16384)).workspace_bytes == wide.workspace_bytes,
          "an item bound past the envelope is the envelope");
  }
  {
    const ignis_model_load_options stray = vision_items(0, 16384);
    const std::string m = load_error(text, &stray);
    check(contains(m, "vision_item_max_tokens"),
          "an item bound without a vision envelope is refused by name: " + m);
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "model load vision options test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("model load vision options test: ok\n");
  return 0;
}
