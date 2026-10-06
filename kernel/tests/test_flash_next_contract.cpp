// GitHub #302 (spec flash-next/04): the Flash-Next program's internal
// contract (kernel/src/flash_next/flash_next_internal.h) -- OURS, not vendored.
//
// The geometry every op family of the program is sized by is read once off
// the load's `ignis_topology`. This pins `Geometry::from` against
// Flash-Next's own numbers (the checkpoint's config.json text_config, the
// same values crates/core/src/compute.rs ModelConfig::qwen38_flash_next
// carries) and the derived widths the op slices size their buffers by.
//
// Host-only: the header is compiled and its inline arithmetic run, nothing
// touches the device. ADR 0006: no SKIP_RETURN_CODE.

#include "flash_next/flash_next_internal.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void expect_eq(long long got, long long want, const std::string &label) {
  if (got != want) {
    std::fprintf(stderr, "  FAIL: %s: got %lld, want %lld\n", label.c_str(), got, want);
    ++g_failed;
  }
}

// The topology Rust hands the leaf for Flash-Next (ModelConfig::topology_abi).
ignis_topology flash_next_topology(std::vector<int32_t> &kinds) {
  kinds.clear();
  for (int i = 0; i < 48; ++i) {
    kinds.push_back((i + 1) % 4 == 0 ? IGNIS_LAYER_GQA : IGNIS_LAYER_GDN);
  }
  ignis_topology t{};
  t.num_layers = 48;
  t.layer_kinds = kinds.data();
  t.hidden = 2560;
  t.vocab = 248320;
  t.num_q_heads = 24;
  t.num_kv_heads = 2;
  t.head_dim = 256;
  t.rotary_dim = 64;
  t.rope_theta = 1e7;
  t.gdn_state_rows = 6144;
  t.gdn_state_cols = 2048;
  t.gdn_num_layers = 36;
  t.gdn_q_width = 2048;
  t.gdn_z_width = 6144;
  t.gdn_ab_width = 96;
  t.ffn_intermediate = 0;
  t.rms_norm_eps = 1e-6f;
  t.family = IGNIS_MODEL_FAMILY_FLASH_NEXT;
  t.gdn_value_heads = 48;
  t.gdn_head_dim = 128;
  t.gdn_conv_kernel = 4;
  t.moe = {512, 10, 640, 640};
  t.hyper = {4, 320};
  t.indexer = {4, 128, 1, 4, 2048};
  t.ngram = {3, 8, 2560, 4, 1};
  return t;
}

}  // namespace

int main() {
  std::vector<int32_t> kinds;
  const auto g = ignis::flash_next::Geometry::from(flash_next_topology(kinds));

  expect_eq(g.layers, 48, "layers");
  expect_eq(g.hidden, 2560, "hidden");
  expect_eq(g.vocab, 248320, "vocab");
  expect_eq(g.q_heads, 24, "q_heads");
  expect_eq(g.kv_heads, 2, "kv_heads");
  expect_eq(g.head_dim, 256, "head_dim");
  expect_eq(g.rotary_dim, 64, "rotary_dim");
  // linear_num_key_heads 16, linear_num_value_heads 48, of 128; kernel 4.
  expect_eq(g.gdn_qk_heads, 16, "gdn_qk_heads");
  expect_eq(g.gdn_value_heads, 48, "gdn_value_heads");
  expect_eq(g.gdn_head_dim, 128, "gdn_head_dim");
  expect_eq(g.gdn_conv_kernel, 4, "gdn_conv_kernel");
  expect_eq(g.experts, 512, "experts");
  expect_eq(g.experts_per_token, 10, "experts_per_token");
  expect_eq(g.expert_intermediate, 640, "expert_intermediate");
  expect_eq(g.shared_intermediate, 640, "shared_intermediate");
  expect_eq(g.streams, 4, "streams");
  expect_eq(g.hc_rank, 320, "hc_rank");
  expect_eq(g.indexer_heads, 4, "indexer_heads");
  expect_eq(g.indexer_head_dim, 128, "indexer_head_dim");
  expect_eq(g.indexer_kv_heads, 1, "indexer_kv_heads");
  expect_eq(g.compress_ratio, 4, "compress_ratio");
  expect_eq(g.indexer_budget, 2048, "indexer_budget");
  expect_eq(g.ngram_size, 3, "ngram_size");
  expect_eq(g.ngram_heads, 16, "ngram_heads");
  expect_eq(g.ngram_embed_dim, 2560, "ngram_embed_dim");
  expect_eq(g.ngram_conv_kernel, 4, "ngram_conv_kernel");
  expect_eq(g.ngram_layer, 1, "ngram_layer");

  // The derived widths the slices size their buffers by.
  expect_eq(g.residual_width(), 10240, "residual_width: 4 streams of 2560");
  expect_eq(g.dense_threshold(), 2051, "dense_threshold: the checkpoint's 2048 + 4 - 1");
  expect_eq(g.selection_width(), 2051, "selection_width");
  expect_eq(g.ngram_conv_state_columns(), 9, "ngram conv state: (4 - 1) x dilation 3");
  expect_eq(g.ngram_head_dim(), 160, "ngram_head_dim: 2560 / 16");
  expect_eq(g.ngram_row_bytes(), 90, "ngram_row_bytes: 80 code bytes + 5 fp16 scales");

  ignis::flash_next::Batch batch;
  batch.lanes = 3;
  batch.tokens = 1;
  expect_eq(batch.rows(), 3, "a decode batch of 3 lanes is 3 rows");

  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next contract test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next contract test: ok\n");
  return 0;
}
