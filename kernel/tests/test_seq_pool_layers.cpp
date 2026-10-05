// GitHub #302 (spec flash-next/04): a sequence pool holds the K/V of the
// topology's attention layers -- `ignis_seq_pool_spec::kv_num_layers`, 16 on
// Qwen 3.8-27B and 12 on Flash-Next -- rather than a compiled-in 16 -- OURS.
//
// Through `ignis_seq_pool_plan`, which lays a pool out without building it:
// the KV arena and the hq-e8-2b residual window grow by exactly a layer's
// planes per layer, and a count outside 1..16 is refused by name.
//
// Host-only (the plan allocates nothing). ADR 0006: no SKIP_RETURN_CODE.

#include "ignis_model.h"
#include "ignis_seq.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

// Flash-Next's attention geometry (2 KV heads of 256), 3 lanes, 8 pages.
ignis_seq_pool_spec spec_of(int32_t kv_format, uint32_t kv_num_layers) {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = 2;
  spec.head_dim = 256;
  spec.kv_format = kv_format;
  spec.kv_page_group_count = 8;
  spec.max_context_tokens = 512;
  spec.slot_count = 3;
  spec.gdn_num_layers = 2;
  spec.gdn_conv_channels = 10240;
  spec.gdn_value_heads = 48;
  spec.gdn_head_dim = 128;
  spec.vocab = 1024;
  spec.kv_num_layers = kv_num_layers;
  return spec;
}

bool plan(const ignis_seq_pool_spec &spec, struct ignis_seq_pool_plan &out) {
  out = {};
  return ignis_seq_pool_plan(&spec, &out) == 0;
}

}  // namespace

int main() {
  // BF16: a layer's K and V planes are 2 x (2 B x 256 x 64 tokens x 2 heads)
  // = 131,072 bytes per page; 4 layers x 8 pages more for 16 than for 12.
  struct ignis_seq_pool_plan twelve{}, sixteen{};
  check(plan(spec_of(IGNIS_KV_FORMAT_BF16, 12), twelve), std::string("a 12-layer BF16 pool plans: ") + ignis_seq_last_error());
  check(plan(spec_of(IGNIS_KV_FORMAT_BF16, 16), sixteen), "a 16-layer BF16 pool plans");
  check(sixteen.kv_bytes - twelve.kv_bytes == 4ull * 8 * 131072,
        "BF16 KV: 4 more layers are 4 x 8 pages x 131,072 bytes: got " +
            std::to_string(sixteen.kv_bytes - twelve.kv_bytes));
  check(twelve.lane_state_bytes == sixteen.lane_state_bytes,
        "the attention layers move no lane state (GDN, penalties) on a BF16 pool");

  // hq-e8-2b: the residual window is 2 roles x layers x slots planes of
  // 256 x 2 heads x 544 rows x 2 bytes, plus 16 ring words per slot.
  struct ignis_seq_pool_plan hq12{}, hq16{};
  check(plan(spec_of(IGNIS_KV_FORMAT_HQ_E8_2B, 12), hq12), "a 12-layer hq pool plans");
  check(plan(spec_of(IGNIS_KV_FORMAT_HQ_E8_2B, 16), hq16), "a 16-layer hq pool plans");
  const uint64_t plane = 256ull * 2 * 544 * 2;
  check(hq12.hq_residual_bytes == 2 * 12 * 3 * plane + 3 * 16 * 4,
        "hq residual window at 12 layers: got " + std::to_string(hq12.hq_residual_bytes));
  check(hq16.hq_residual_bytes == 2 * 16 * 3 * plane + 3 * 16 * 4,
        "hq residual window at 16 layers: got " + std::to_string(hq16.hq_residual_bytes));
  // hq KV: 72 bytes a row (64 code + 8 meta) per role per head.
  check(hq16.kv_bytes - hq12.kv_bytes == 4ull * 8 * 2 * 72 * 64 * 2,
        "hq KV: 4 more layers are 4 x 8 pages x 2 roles x 72 B x 64 tokens x 2 heads: got " +
            std::to_string(hq16.kv_bytes - hq12.kv_bytes));

  // Outside 1..16, refused by name.
  for (uint32_t layers : {0u, 17u}) {
    struct ignis_seq_pool_plan out{};
    const bool planned = plan(spec_of(IGNIS_KV_FORMAT_BF16, layers), out);
    const std::string message = ignis_seq_last_error();
    check(!planned && message.find("kv_num_layers " + std::to_string(layers)) != std::string::npos,
          "kv_num_layers " + std::to_string(layers) + " is refused by name: got \"" + message + "\"");
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "seq pool layers test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("seq pool layers test: ok\n");
  return 0;
}
