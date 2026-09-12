// P4-05 (GitHub #123): `ignis_model_load` refuses a KV format it does not
// recognize -- OURS, not vendored.
//
// The format is a load argument because the GQA attention workspace is part
// of the scratch reservation and its size depends on it (kernel/src/model.cu:
// the hq-e8-2b prompt route materializes the envelope's visible history into
// two rotated-frame BF16 scratch planes, which BF16's own prompt route has no
// counterpart for). An unrecognized value that fell through to a BF16 default
// would reserve one format's arena and then let the layers run whatever the
// pool says -- the exact silent mismatch `ignis_gqa_layer`'s own refusal
// exists to prevent, except caught nowhere.
//
// It is a CTest rather than a Rust test because the Rust binding cannot
// express the bug: `ignis_core::KvFormat` is a two-variant enum, so
// `abi_code()` only ever produces 0 or 1. This guard is only reachable from a
// caller writing the flat C ABI directly, which is what this file does.
//
// Host-only by construction: every check `ignis_model_load` makes before the
// kv_format one is pure argument validation, and the function returns at the
// kv_format check without creating a stream, binding a tensor, or touching
// the device. That is deliberate -- it is why the empty bound-tensor list
// below is enough, and it is itself worth pinning: the control arm asserts a
// *recognized* format gets past this check and fails on something else.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so nothing here can
// read as a skip.

#include "ignis_model.h"
#include "ignis_seq.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

// The smallest topology that gets past every argument check ahead of the
// kv_format one: no layers (so `layer_kinds` may be null), one GDN layer so
// the `gdn_state_rows % gdn_num_layers` check divides.
ignis_topology minimal_topology() {
  ignis_topology topology{};
  topology.num_layers     = 0;
  topology.layer_kinds    = nullptr;
  topology.gdn_num_layers = 1;
  return topology;
}

// Loads with `kv_format` and returns the leaf's own last-error message. The
// load always fails here -- with no bound tensors it could not do otherwise.
// What the two arms differ on is *why*.
std::string load_error_for(int32_t kv_format) {
  const ignis_topology topology = minimal_topology();
  // Non-null, count 0: the null check is about the pointer, and the format
  // check happens before anything reads through it.
  ignis_bound_tensor tensors[1]{};
  ignis_model *model = nullptr;
  const int32_t rc   = ignis_model_load(tensors, /*count=*/0, &topology,
                                        /*prefill_chunk_tokens=*/128,
                                        /*max_context_tokens=*/128, kv_format, &model);
  if (rc == 0 || model != nullptr) {
    std::fprintf(stderr, "FATAL: a load with no bound tensors must not succeed\n");
    if (model != nullptr) { ignis_model_free(model); }
    std::exit(EXIT_FAILURE);
  }
  return ignis_model_last_error();
}

} // namespace

int main() {
  // An unrecognized format is refused by name, and the message says which
  // value it was so an operator (or a future third format) can see what the
  // ABI was handed.
  for (const int32_t bogus : {-1, 2, 7}) {
    const std::string message = load_error_for(bogus);
    check(message.find("kv_format") != std::string::npos,
         "kv_format " + std::to_string(bogus) + " is refused by name: got \"" + message + "\"");
    check(message.find(std::to_string(bogus)) != std::string::npos,
         "the refusal names the value it was handed (" + std::to_string(bogus) + "): got \"" +
             message + "\"");
  }

  // The control arm, and the reason the checks above are not vacuous: a
  // recognized format gets *past* the kv_format check and fails on the first
  // missing bound tensor instead. Both real formats, so neither is the one
  // that happens to be a passthrough.
  for (const int32_t known : {IGNIS_KV_FORMAT_BF16, IGNIS_KV_FORMAT_HQ_E8_2B}) {
    const std::string message = load_error_for(known);
    check(message.find("kv_format") == std::string::npos,
         "a recognized kv_format (" + std::to_string(known) +
             ") must get past the format check: got \"" + message + "\"");
    check(message.find("text/token_embedding") != std::string::npos,
         "a recognized kv_format reaches the bound-tensor binding: got \"" + message + "\"");
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "model load kv_format test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("model load kv_format test: ok\n");
  return 0;
}
