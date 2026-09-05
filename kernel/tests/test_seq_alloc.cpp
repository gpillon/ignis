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
}

bool slot_is_zero(ignis_seq_pool &pool, std::uint32_t layers, std::int32_t slot) {
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    if (!tensor_is_zero(pool.gdn_pool.conv_slot(layer, slot)) ||
        !tensor_is_zero(pool.gdn_pool.recurrent_slot(layer, slot))) {
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

  // A small, fast geometry: 2 KV planes (K, V) of head_dim=8 x 2 heads,
  // kPagedKVPageSize=64 tokens/page; a 2-layer GDN state of 2 value heads x
  // 4x4 fp32 each. slot_count=3 lets one test drive both the "no free KV
  // pages" and the "no free slot" exhaustion paths independently.
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
  dirty_page(pool->kv_pool.plane(0), 0);
  dirty_page(pool->kv_pool.plane(1), 0);
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
    if (!page_is_zero(pool->kv_pool.plane(0), page_id) ||
        !page_is_zero(pool->kv_pool.plane(1), page_id)) {
      ++failures;
      std::fprintf(stderr, "FAIL: re-allocated sequence's KV page %d is not zero\n", page_id);
    }
  }
  expect(slot_is_zero(*pool, spec.gdn_num_layers, d_stats.slot),
        "re-allocated sequence's GDN slot is zero");

  // Snapshot/restore: declared, not implemented (G4) -- exercise on the
  // still-live handle before releasing it.
  unsigned char scratch[16];
  expect_rc(ignis_seq_snapshot(seq_d, scratch, sizeof(scratch)), IGNIS_SEQ_ERR_NOT_IMPLEMENTED,
           "snapshot is not-implemented");
  expect_rc(ignis_seq_restore(seq_d, scratch, sizeof(scratch)), IGNIS_SEQ_ERR_NOT_IMPLEMENTED,
           "restore is not-implemented");

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

  ignis_seq_pool_free(pool);

  if (failures != 0) {
    std::fprintf(stderr, "sequence handle test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("sequence handle test: ok\n");
  return 0;
}
