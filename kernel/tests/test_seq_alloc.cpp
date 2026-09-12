// Leaf-level sequence handle test (P1-19, GitHub #55).
//
// Ours, not vendored: exercises the public flat ABI (ignis_seq.h) the way
// Rust does (pool create -> alloc -> stats -> release -> exhaust -> re-alloc)
// plus one check the flat ABI deliberately cannot make from Rust -- that a
// released-then-re-allocated slot's KV pages and GDN state actually read
// back as zero -- using ignis_seq_internal.h's struct definitions to reach
// the vendored pools' device memory directly (the same access
// test_kv_cache.cpp / test_state_store.cpp use).
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE is set on this test
// (kernel/tests/CMakeLists.txt), so a missing/busy GPU fails it, never skips.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
  }
}

void expect_rc(int32_t rc, int32_t want, const char *label) {
  if (rc != want) {
    std::fprintf(stderr, "FAIL: %s (rc=%d, want %d: %s)\n", label, rc, want,
                 ignis_seq_last_error());
    ++failures;
  }
}

bool cuda_unavailable(cudaError_t err) {
  return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

bool tensor_is_zero(const ninfer::Tensor &t) {
  std::vector<unsigned char> host(t.bytes());
  CUDA_CHECK(cudaMemcpy(host.data(), t.data, host.size(), cudaMemcpyDeviceToHost));
  for (unsigned char b : host) {
    if (b != 0) {
      return false;
    }
  }
  return true;
}

bool page_is_zero(const ninfer::Tensor &plane, std::int32_t page_id) {
  std::vector<unsigned char> host(static_cast<std::size_t>(plane.nb[3]));
  const auto *base = static_cast<const unsigned char *>(plane.data);
  CUDA_CHECK(cudaMemcpy(host.data(), base + static_cast<std::int64_t>(page_id) * plane.nb[3],
                       host.size(), cudaMemcpyDeviceToHost));
  for (unsigned char b : host) {
    if (b != 0) {
      return false;
    }
  }
  return true;
}

void dirty_page(const ninfer::Tensor &plane, std::int32_t page_id) {
  auto *base = static_cast<unsigned char *>(plane.data);
  CUDA_CHECK(cudaMemset(base + static_cast<std::int64_t>(page_id) * plane.nb[3], 0xab,
                        static_cast<std::size_t>(plane.nb[3])));
}

void dirty_slot(ignis_seq_pool &pool, std::uint32_t layers, std::int32_t slot) {
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    ninfer::Tensor conv = pool.gdn_pool.conv_slot(layer, slot);
    ninfer::Tensor rec  = pool.gdn_pool.recurrent_slot(layer, slot);
    CUDA_CHECK(cudaMemset(conv.data, 0xcd, conv.bytes()));
    CUDA_CHECK(cudaMemset(rec.data, 0xcd, rec.bytes()));
  }
  // P3-03 (GitHub #99): this slot's presence/frequency penalty-count row.
  CUDA_CHECK(cudaMemset(pool.token_counts_for(slot), 0xcd,
                        static_cast<std::size_t>(pool.vocab) * sizeof(std::int32_t)));
}

bool slot_is_zero(ignis_seq_pool &pool, std::uint32_t layers, std::int32_t slot) {
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    if (!tensor_is_zero(pool.gdn_pool.conv_slot(layer, slot)) ||
        !tensor_is_zero(pool.gdn_pool.recurrent_slot(layer, slot))) {
      return false;
    }
  }
  std::vector<std::int32_t> counts(static_cast<std::size_t>(pool.vocab));
  CUDA_CHECK(cudaMemcpy(counts.data(), pool.token_counts_for(slot),
                        counts.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
  for (std::int32_t count : counts) {
    if (count != 0) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  int count                   = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&count);
  if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
  }
  if (count_err != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
    return 1;
  }

  // A small, fast geometry: every GQA layer has a K,V pair of head_dim=8 x 2
  // heads, kPagedKVPageSize=64 tokens/page; a 2-layer GDN state of 2 value
  // heads x 4x4 fp32 each. slot_count=3 lets one test drive both the "no free
  // KV pages" and the "no free slot" exhaustion paths independently.
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = 2;
  spec.head_dim            = 8;
  spec.kv_page_group_count = 4;
  spec.max_context_tokens  = 128; // pages_for_tokens(128) == 2
  spec.slot_count          = 3;
  spec.gdn_num_layers      = 2;
  spec.gdn_conv_channels   = 6;
  spec.gdn_value_heads     = 2;
  spec.gdn_head_dim        = 4;
  spec.vocab               = 32; // P3-03 (GitHub #99): small, fast penalty-count geometry

  ignis_seq_pool *pool = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "pool create");

  struct ignis_seq_pool_stats stats{};
  expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "pool stats (fresh)");
  expect(stats.kv_page_group_count == 4, "fresh: page group count");
  expect(stats.kv_entitled_pages == 0, "fresh: entitled pages");
  expect(stats.kv_free_pages == 4, "fresh: free pages");
  expect(stats.logical_page_capacity == 2, "fresh: logical page capacity");
  expect(stats.slot_count == 3, "fresh: slot count");
  expect(stats.free_slot_count == 3, "fresh: free slot count");

  // Two sequences of 2 pages each exhaust the 4-page pool while one slot
  // stays free -- the next alloc must fail on pages, not on slots.
  ignis_seq *seq_a = nullptr;
  expect_rc(ignis_seq_alloc(pool, 128, &seq_a), 0, "alloc A");
  ignis_seq *seq_b = nullptr;
  expect_rc(ignis_seq_alloc(pool, 128, &seq_b), 0, "alloc B");

  struct ignis_seq_stats a_stats{};
  expect_rc(ignis_seq_stats(seq_a, &a_stats), 0, "seq A stats");
  expect(a_stats.page_entitlement == 2, "seq A page entitlement");
  expect(a_stats.mapped_pages == 2, "seq A mapped pages");
  expect(a_stats.token_capacity == 128, "seq A token capacity");
  expect(a_stats.slot >= 0 && a_stats.slot < 3, "seq A slot in range");

  expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "pool stats (2 live)");
  expect(stats.kv_entitled_pages == 4, "2 live: entitled pages");
  expect(stats.kv_free_pages == 0, "2 live: free pages");
  expect(stats.free_slot_count == 1, "2 live: free slot count");

  ignis_seq *seq_c = nullptr;
  expect_rc(ignis_seq_alloc(pool, 64, &seq_c), -1, "alloc C exhausts KV pages, not slots");
  expect(seq_c == nullptr, "alloc C produced no handle");

  // Pool is left unchanged by the failed alloc.
  expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "pool stats (after failed alloc)");
  expect(stats.kv_entitled_pages == 4, "after failed alloc: entitled pages unchanged");
  expect(stats.free_slot_count == 1, "after failed alloc: free slot count unchanged");

  // Dirty A's pages and slot before releasing it, so re-allocation can prove
  // the fresh handle observes zero, not the previous occupant's bytes.
  const std::int32_t a_slot = a_stats.slot;
  for (std::size_t plane = 0; plane < pool->kv_pool.plane_count(); ++plane) {
    dirty_page(pool->kv_pool.plane(plane), 0);
  }
  dirty_slot(*pool, spec.gdn_num_layers, a_slot);
  expect(!page_is_zero(pool->kv_pool.plane(0), 0), "sanity: dirtied page reads non-zero");
  expect(!slot_is_zero(*pool, spec.gdn_num_layers, a_slot), "sanity: dirtied slot reads non-zero");

  ignis_seq_release(pool, seq_a);
  expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "pool stats (after release A)");
  expect(stats.kv_entitled_pages == 2, "after release A: entitled pages");
  expect(stats.kv_free_pages == 2, "after release A: free pages");
  expect(stats.free_slot_count == 2, "after release A: free slot count");

  // Re-allocate: the only free physical pages are the ones A just returned
  // (0 and 1, from take_pages' lowest-first policy), and the only slot in
  // A's row range that was freed is a_slot re-entering the free list --
  // whichever slot this lands on, its KV pages and GDN state must read
  // zero, not the dirtied bytes above.
  ignis_seq *seq_d = nullptr;
  expect_rc(ignis_seq_alloc(pool, 64, &seq_d), 0, "alloc D (re-alloc)");
  struct ignis_seq_stats d_stats{};
  expect_rc(ignis_seq_stats(seq_d, &d_stats), 0, "seq D stats");
  for (std::int32_t page_id : seq_d->kv.page_ids()) {
    for (std::size_t plane = 0; plane < pool->kv_pool.plane_count(); ++plane) {
      if (!page_is_zero(pool->kv_pool.plane(plane), page_id)) {
        ++failures;
        std::fprintf(stderr, "FAIL: re-allocated sequence's KV page %d plane %zu is not zero\n",
                     page_id, plane);
      }
    }
  }
  expect(slot_is_zero(*pool, spec.gdn_num_layers, d_stats.slot),
        "re-allocated sequence's GDN slot is zero");

  // Snapshot / restore live in their own test (kernel/tests/test_seq_snapshot.cu,
  // P4-06 / GitHub #124): they need a dirtied sequence and a second one to
  // restore into, which is more setup than this file's alloc/release
  // accounting.

  ignis_seq_release(pool, seq_d);
  ignis_seq_release(pool, seq_b);

  expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "pool stats (all released)");
  expect(stats.kv_entitled_pages == 0, "all released: entitled pages");
  expect(stats.kv_free_pages == 4, "all released: free pages");
  expect(stats.free_slot_count == 3, "all released: free slot count");

  // A dedicated slot_count=1 pool isolates "no free slot" from "no free KV
  // pages" (the pool above always had spare pages relative to slots at the
  // point every slot was in use).
  ignis_seq_pool_spec tight_spec = spec;
  tight_spec.kv_page_group_count = 100;
  tight_spec.slot_count          = 1;
  ignis_seq_pool *tight_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&tight_spec, &tight_pool), 0, "tight pool create");
  ignis_seq *seq_x = nullptr;
  expect_rc(ignis_seq_alloc(tight_pool, 64, &seq_x), 0, "tight alloc X");
  ignis_seq *seq_y = nullptr;
  expect_rc(ignis_seq_alloc(tight_pool, 64, &seq_y), -1, "tight alloc Y exhausts slots, not pages");
  expect(seq_y == nullptr, "tight alloc Y produced no handle");
  ignis_seq_release(tight_pool, seq_x);
  ignis_seq_pool_free(tight_pool);

  // Bad arguments are rejected, not silently accepted, and produce no handle.
  ignis_seq_pool *rejected_pool = nullptr;
  expect_rc(ignis_seq_pool_create(nullptr, &rejected_pool), -1, "null spec is rejected");
  expect(rejected_pool == nullptr, "null spec produced no handle");

  ignis_seq *rejected_seq = nullptr;
  expect_rc(ignis_seq_alloc(pool, 0, &rejected_seq), -1, "zero context_tokens is rejected");
  expect(rejected_seq == nullptr, "zero context_tokens produced no handle");

  ignis_seq_pool_spec bad_spec = spec;
  bad_spec.head_dim            = 0;
  ignis_seq_pool *bad_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&bad_spec, &bad_pool), -1, "non-positive geometry is rejected");
  expect(bad_pool == nullptr, "non-positive geometry produced no handle");

  ignis_seq_pool_spec bad_vocab_spec = spec;
  bad_vocab_spec.vocab               = 0;
  ignis_seq_pool *bad_vocab_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&bad_vocab_spec, &bad_vocab_pool), -1,
           "zero vocab is rejected (P3-03, GitHub #99)");
  expect(bad_vocab_pool == nullptr, "zero vocab produced no handle");

  // ---- the KV format as a load option (P4-04, GitHub #122) ----------------
  //
  // The same byte-for-byte pool geometry under each format, at the real 27B
  // head shape (4 KV heads of 256), so the reported per-token cost is the
  // one the capacity finding recorded: 65,536 bytes under BF16 and 9,216
  // under hq-e8-2b. Both numbers come out of the planes the pool actually
  // planned, not from a per-format constant, which is what makes a token
  // capacity derived rather than configured.
  ignis_seq_pool_spec format_spec   = spec;
  format_spec.num_kv_heads          = 4;
  format_spec.head_dim              = 256;
  format_spec.kv_page_group_count   = 8;
  format_spec.max_context_tokens    = 128;

  ignis_seq_pool_spec bf16_spec = format_spec;
  bf16_spec.kv_format           = IGNIS_KV_FORMAT_BF16;
  ignis_seq_pool *bf16_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&bf16_spec, &bf16_pool), 0, "bf16 pool create");
  struct ignis_seq_pool_stats bf16_stats{};
  expect_rc(ignis_seq_pool_stats(bf16_pool, &bf16_stats), 0, "bf16 pool stats");
  expect(bf16_stats.kv_format == IGNIS_KV_FORMAT_BF16, "bf16: reported format");
  expect(bf16_stats.kv_bytes_per_token == 65536, "bf16: 65,536 bytes per sequence-token");
  expect(bf16_stats.kv_token_capacity == 8 * 64, "bf16: capacity is pages x page tokens");
  expect(bf16_pool->kv_pool.plane_count() == 2 * kIgnisGqaLayerCount,
        "bf16: two planes per GQA layer");

  ignis_seq_pool_spec hq_spec = format_spec;
  hq_spec.kv_format           = IGNIS_KV_FORMAT_HQ_E8_2B;
  ignis_seq_pool *hq_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&hq_spec, &hq_pool), 0, "hq pool create");
  struct ignis_seq_pool_stats hq_stats{};
  expect_rc(ignis_seq_pool_stats(hq_pool, &hq_stats), 0, "hq pool stats");
  expect(hq_stats.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B, "hq: reported format");
  expect(hq_stats.kv_bytes_per_token == 9216, "hq: 9,216 bytes per sequence-token");
  expect(hq_stats.kv_token_capacity == 8 * 64, "hq: capacity is pages x page tokens");
  expect(hq_pool->kv_pool.plane_count() == 4 * kIgnisGqaLayerCount,
        "hq: a code and a metadata plane per role per GQA layer");
  // The same page count is 7.11x the tokens per byte, which is the whole
  // reason this format is a load option at all.
  expect(hq_stats.kv_page_bytes * 7 < bf16_stats.kv_page_bytes,
        "hq pages are more than 7x denser than BF16 pages");
  // Each layer's planes carry the codec's fixed row budgets, addressed
  // page-major exactly like the BF16 value planes beside them.
  const ninfer::Tensor &hq_codes =
      hq_pool->kv_pool.plane(ignis_kv_plane_index(IGNIS_KV_FORMAT_HQ_E8_2B, 3, IGNIS_KV_PLANE_V));
  const ninfer::Tensor &hq_meta = hq_pool->kv_pool.plane(
      ignis_kv_plane_index(IGNIS_KV_FORMAT_HQ_E8_2B, 3, IGNIS_KV_PLANE_V_META));
  expect(hq_codes.ne[0] == kIgnisHqCodeRowBytes, "hq: 64-byte code rows");
  expect(hq_meta.ne[0] == kIgnisHqMetaRowBytes, "hq: 8-byte metadata rows");
  expect(hq_codes.ne[1] == ninfer::kPagedKVPageSize && hq_meta.ne[1] == ninfer::kPagedKVPageSize,
        "hq: both planes are page-major");

  // A sequence draws pages from either pool identically -- the format
  // changes what a page holds, never how many a context needs.
  ignis_seq *hq_seq = nullptr;
  expect_rc(ignis_seq_alloc(hq_pool, 128, &hq_seq), 0, "hq alloc");
  struct ignis_seq_stats hq_seq_stats{};
  expect_rc(ignis_seq_stats(hq_seq, &hq_seq_stats), 0, "hq seq stats");
  expect(hq_seq_stats.token_capacity == 128, "hq: a 128-token sequence maps 128 tokens");
  ignis_seq_release(hq_pool, hq_seq);

  ignis_seq_pool_free(hq_pool);
  ignis_seq_pool_free(bf16_pool);

  // An unknown format is refused rather than quietly treated as BF16.
  ignis_seq_pool_spec bad_format_spec = format_spec;
  bad_format_spec.kv_format           = 7;
  ignis_seq_pool *bad_format_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&bad_format_spec, &bad_format_pool), -1,
           "an unknown kv_format is rejected");
  expect(bad_format_pool == nullptr, "an unknown kv_format produced no handle");

  // The codec's row budget is defined for a 256-dimension row only, so an hq
  // pool at any other head_dim would plan planes its append path cannot fill.
  ignis_seq_pool_spec bad_hq_spec = format_spec;
  bad_hq_spec.kv_format           = IGNIS_KV_FORMAT_HQ_E8_2B;
  bad_hq_spec.head_dim            = 128;
  ignis_seq_pool *bad_hq_pool     = nullptr;
  expect_rc(ignis_seq_pool_create(&bad_hq_spec, &bad_hq_pool), -1,
           "hq at a head_dim other than 256 is rejected");
  expect(bad_hq_pool == nullptr, "a bad hq head_dim produced no handle");

  ignis_seq_pool_free(pool);

  if (failures != 0) {
    std::fprintf(stderr, "sequence handle test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("sequence handle test: ok\n");
  return 0;
}
