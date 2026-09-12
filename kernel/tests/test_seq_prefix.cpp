// Leaf-level device prefix reuse test (P4-10, GitHub #126, ADR 0024).
//
// Ours, not vendored. The flat ABI can say a claim succeeded; it cannot say
// whether the two mechanisms ADR 0024 describes actually happened, and those
// are what this file checks:
//
//   1. **the pages are shared, not copied.** A claimant's block-table row
//      addresses the publisher's own physical pages for the head and its own
//      for the tail, and the pool is charged for the shared pages exactly
//      once however many claimants hold them;
//   2. **the mutable state is cloned device-to-device.** A claimant's GDN
//      slot, conv taps and penalty-count row come out byte-identical to the
//      publisher's state at the prefix's end, without a host round trip;
//   3. **a page is freed only when the last holder releases it.** Releasing
//      one claimant returns its tail and nothing else, and the shared pages
//      come back when the publisher's handle and every claimant are gone;
//   4. **every refusal refuses**, leaving the sequence and the pool as they
//      were: a partial page, a frontier that is not the prefix, a mid-chunk
//      publisher, a second publish, a reservation with no page of its own,
//      and a state transfer of a sequence whose history is not all its own.
//
// It also reports the measured device-to-device clone cost at the real 27B
// geometry -- the other half of ADR 0024's cost asymmetry, whose host-side
// half GitHub #124 measured. The numbers print on every run.
//
// What this file deliberately does not claim is that a claimant *generates*
// what a sibling that prefilled the prefix itself generates. That needs a
// real model and real tokens, and it lives one level up in
// crates/core/tests/prefix_reuse_gpu.rs.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE is set on this test
// (kernel/tests/CMakeLists.txt), so a missing/busy GPU fails it, never skips.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
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

// A deterministic byte pattern, so a cloned section can be compared against
// the one it came from without a second host copy of each.
std::vector<unsigned char> pattern(std::size_t bytes, std::uint32_t seed) {
  std::vector<unsigned char> host(bytes);
  std::uint32_t x = seed | 1U;
  for (unsigned char &b : host) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    b = static_cast<unsigned char>(x);
  }
  return host;
}

void fill_device(void *dst, std::size_t bytes, std::uint32_t seed) {
  if (bytes == 0) {
    return;
  }
  const std::vector<unsigned char> host = pattern(bytes, seed);
  CUDA_CHECK(cudaMemcpy(dst, host.data(), bytes, cudaMemcpyHostToDevice));
}

std::vector<unsigned char> read_device(const void *src, std::size_t bytes) {
  std::vector<unsigned char> host(bytes);
  if (bytes != 0) {
    CUDA_CHECK(cudaMemcpy(host.data(), src, bytes, cudaMemcpyDeviceToHost));
  }
  return host;
}

// Put `seq` at a completed chunk boundary `tokens` in: the program frontier
// and every layer's own, GDN included.
void set_frontier(ignis_seq &seq, std::uint64_t tokens) {
  seq.position = tokens;
  for (std::uint32_t &frontier : seq.gqa_positions) {
    frontier = static_cast<std::uint32_t>(tokens);
  }
  for (std::uint32_t &frontier : seq.gdn_positions) {
    frontier = static_cast<std::uint32_t>(tokens);
  }
}

// Give `seq` a history of `tokens` tokens: a chunk boundary there, a pending
// token, a known pattern in the KV pages its history occupies, and a known
// pattern in every mutable section.
void give_history(ignis_seq_pool &pool, ignis_seq &seq, std::uint64_t tokens,
                  std::uint32_t seed) {
  set_frontier(seq, tokens);
  seq.pending_token = static_cast<std::int32_t>(1000 + seed);

  std::uint32_t salt = seed;
  const std::uint32_t written = ninfer::pages_for_tokens(static_cast<std::uint32_t>(tokens));
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    auto *base                  = static_cast<unsigned char *>(plane.data);
    const auto page_ids         = seq.kv.page_ids();
    for (std::uint32_t page = 0; page < written && page < page_ids.size(); ++page) {
      fill_device(base + static_cast<std::int64_t>(page_ids[page]) * plane.nb[3],
                  static_cast<std::size_t>(plane.nb[3]), ++salt);
    }
  }
  for (std::uint32_t layer = 0; layer < pool.gdn_pool.layer_count(); ++layer) {
    const ninfer::Tensor conv = pool.gdn_pool.conv_slot(layer, seq.slot);
    const ninfer::Tensor rec  = pool.gdn_pool.recurrent_slot(layer, seq.slot);
    fill_device(conv.data, conv.bytes(), ++salt);
    fill_device(rec.data, rec.bytes(), ++salt);
  }
  fill_device(pool.token_counts_for(seq.slot),
              static_cast<std::size_t>(pool.vocab) * sizeof(std::int32_t), ++salt);
}

// A sequence's block-table row as the kernels read it: `count` physical page
// ids, in logical page order.
std::vector<std::int32_t> row_of(const ignis_seq_pool &pool, std::int32_t slot,
                                 std::uint32_t count) {
  const ninfer::Tensor row = pool.kv_pool.block_table_row(slot);
  std::vector<std::int32_t> ids(count);
  if (count != 0) {
    CUDA_CHECK(cudaMemcpy(ids.data(), row.data, count * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
  }
  return ids;
}

// A slot's whole mutable state, as one host image: the conv taps, then the
// recurrent matrices, then the penalty-count row. Two slots carrying the same
// image are carrying the same state, which is what a clone has to produce.
std::vector<unsigned char> mutable_image_of(const ignis_seq_pool &pool, std::int32_t slot) {
  const std::size_t conv_bytes   = pool.gdn_pool.conv_host_image_bytes();
  const std::size_t rec_bytes    = pool.gdn_pool.recurrent_host_image_bytes();
  const std::size_t counts_bytes = static_cast<std::size_t>(pool.vocab) * sizeof(std::int32_t);
  std::vector<unsigned char> image(conv_bytes + rec_bytes + counts_bytes);
  pool.gdn_pool.pack_slot_to_host(slot, image.data(), image.data() + conv_bytes, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(image.data() + conv_bytes + rec_bytes, pool.token_counts_for(slot),
                             counts_bytes, cudaMemcpyDeviceToHost, nullptr));
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  return image;
}

// Every plane's bytes for one physical page -- the KV history itself, used to
// show that a shared page is never zeroed or rewritten by a claim.
std::vector<unsigned char> page_image_of(const ignis_seq_pool &pool, std::int32_t page_id) {
  std::vector<unsigned char> image;
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    const auto *base            = static_cast<const unsigned char *>(plane.data);
    const std::vector<unsigned char> bytes =
        read_device(base + static_cast<std::int64_t>(page_id) * plane.nb[3],
                    static_cast<std::size_t>(plane.nb[3]));
    image.insert(image.end(), bytes.begin(), bytes.end());
  }
  return image;
}

struct ignis_seq_prefix_stats stats_of(const ignis_seq_prefix *prefix, const char *label) {
  struct ignis_seq_prefix_stats stats{};
  expect_rc(ignis_seq_prefix_stats(prefix, &stats), 0, label);
  return stats;
}

struct ignis_seq_stats seq_stats_of(const ignis_seq *seq, const char *label) {
  struct ignis_seq_stats stats{};
  expect_rc(ignis_seq_stats(seq, &stats), 0, label);
  return stats;
}

// A pool big enough for a publisher and three claimants at 384-token
// contexts (6 pages each), with room to show that the shared pages are
// charged once: 32 physical pages could not hold four unshared 6-page
// sequences plus their tails otherwise.
ignis_seq_pool_spec small_spec() {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = 2;
  spec.head_dim            = 8;
  spec.kv_format           = IGNIS_KV_FORMAT_BF16;
  spec.kv_page_group_count = 32;
  spec.max_context_tokens  = 384; // pages_for_tokens(384) == 6
  spec.slot_count          = 4;
  spec.gdn_num_layers      = 2;
  spec.gdn_conv_channels   = 6;
  spec.gdn_value_heads     = 2;
  spec.gdn_head_dim        = 4;
  spec.vocab               = 32;
  return spec;
}

// The real Qwen 3.8-27B sequence geometry (crates/core/src/compute.rs's
// `ModelConfig::qwen38_27b`) under hq-e8-2b, the serving default -- what
// makes the cost figure below the engine's own rather than a scaled proxy.
ignis_seq_pool_spec qwen38_27b_spec(std::uint32_t context_tokens, std::uint32_t slot_count) {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = 4;
  spec.head_dim            = 256;
  spec.kv_format           = IGNIS_KV_FORMAT_HQ_E8_2B;
  spec.kv_page_group_count = ninfer::pages_for_tokens(context_tokens) * slot_count;
  spec.max_context_tokens  = context_tokens;
  spec.slot_count          = slot_count;
  spec.gdn_num_layers      = 48;
  spec.gdn_conv_channels   = 10240; // q 2048 + k 2048 + v 6144
  spec.gdn_value_heads     = 48;
  spec.gdn_head_dim        = 128;
  spec.vocab               = 248320;
  return spec;
}

constexpr std::uint32_t kPageTokens = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
constexpr std::uint32_t kContext    = 384;
constexpr std::uint32_t kPrefix     = 2 * kPageTokens; // 128 tokens == 2 pages

// ---- 1. the pages are shared, and charged once ----------------------------

void check_publish_shares_pages_and_charges_once() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "share: pool create");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "share: alloc publisher");
  give_history(*pool, *publisher, kPrefix, 0x21u);

  const std::vector<std::int32_t> before_row = row_of(*pool, publisher->slot, 6);
  const std::vector<std::int32_t> head_ids(before_row.begin(), before_row.begin() + 2);
  const std::vector<unsigned char> head_image = page_image_of(*pool, head_ids[0]);

  struct ignis_seq_pool_stats pool_before{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_before), 0, "share: pool stats before");

  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, &prefix), 0, "share: publish");

  struct ignis_seq_pool_stats pool_after{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_after), 0, "share: pool stats after");
  // Publishing does not cost a page: the prefix takes over pages the
  // publisher already held, and the publisher re-reserves exactly the tail it
  // gave back.
  expect(pool_after.kv_free_pages == pool_before.kv_free_pages,
         "share: publishing a prefix charges the pool nothing new");

  const struct ignis_seq_prefix_stats prefix_stats = stats_of(prefix, "share: prefix stats");
  expect(prefix_stats.tokens == kPrefix, "share: the prefix covers the tokens it was published at");
  expect(prefix_stats.pages == 2, "share: 128 tokens over 64-token pages is 2 pages");
  expect(prefix_stats.refcount == 2,
         "share: the returned handle and the publishing sequence both hold the prefix");
  expect(prefix_stats.clone_image_bytes > 0,
         "share: the prefix keeps a device image of the mutable sections");
  expect(prefix_stats.clone_count == 0, "share: publishing is not a claim");

  // The publisher's own row is unchanged where it matters: the head still
  // addresses the same physical pages, which is the whole point of taking
  // them over rather than copying them.
  const std::vector<std::int32_t> after_row = row_of(*pool, publisher->slot, 6);
  expect(std::equal(head_ids.begin(), head_ids.end(), after_row.begin()),
         "share: the publisher still addresses the same physical head pages");
  expect(page_image_of(*pool, head_ids[0]) == head_image,
         "share: publishing does not rewrite a shared page");

  const struct ignis_seq_stats publisher_stats = seq_stats_of(publisher, "share: publisher seq stats");
  expect(publisher_stats.shared_pages == 2, "share: the publisher now shares its own head");
  expect(publisher_stats.mapped_pages == 6,
         "share: the publisher's history is still its whole reservation");
  expect(publisher_stats.token_capacity == 6ULL * kPageTokens,
         "share: capacity counts the shared pages the row addresses");

  // Three claimants, each costing only its own tail (4 pages), never the
  // shared head. Unshared, four 6-page sequences would need 24 pages; here
  // the pool spends 6 + 3 * 4 == 18.
  std::vector<ignis_seq *> claimants;
  for (int i = 0; i < 3; ++i) {
    ignis_seq *claimant = nullptr;
    expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "share: claim");
    claimants.push_back(claimant);
  }
  struct ignis_seq_pool_stats pool_claimed{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_claimed), 0, "share: pool stats claimed");
  expect(pool_before.kv_free_pages - pool_claimed.kv_free_pages == 3 * 4,
         "share: three claimants cost three tails, not three histories");

  const std::vector<std::int32_t> claimant_row = row_of(*pool, claimants[0]->slot, 6);
  expect(std::equal(head_ids.begin(), head_ids.end(), claimant_row.begin()),
         "share: a claimant's row addresses the publisher's physical head pages");
  expect(claimant_row[2] != head_ids[0] && claimant_row[2] != head_ids[1],
         "share: a claimant's tail is its own");
  expect(page_image_of(*pool, head_ids[0]) == head_image,
         "share: claiming does not zero or rewrite a shared page");

  for (ignis_seq *claimant : claimants) {
    ignis_seq_release(pool, claimant);
  }
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  struct ignis_seq_pool_stats pool_end{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_end), 0, "share: pool stats end");
  expect(pool_end.kv_free_pages == pool_end.kv_page_group_count,
         "share: every page is back once the last holder releases");
  expect(pool_end.free_slot_count == spec.slot_count, "share: every slot is back");
  ignis_seq_pool_free(pool);
}

// ---- 2. the mutable state is cloned device-to-device ----------------------

void check_a_claimant_receives_the_mutable_state() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "clone: pool create");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "clone: alloc publisher");
  give_history(*pool, *publisher, kPrefix, 0x37u);
  const std::vector<unsigned char> at_boundary = mutable_image_of(*pool, publisher->slot);

  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, &prefix), 0, "clone: publish");

  // The publisher keeps going: its mutable state moves past the prefix, which
  // is exactly why the prefix had to capture its own copy.
  fill_device(pool->token_counts_for(publisher->slot),
              static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t), 0xBEEFu);
  for (std::uint32_t layer = 0; layer < pool->gdn_pool.layer_count(); ++layer) {
    const ninfer::Tensor rec = pool->gdn_pool.recurrent_slot(layer, publisher->slot);
    fill_device(rec.data, rec.bytes(), 0xC0DEu + layer);
  }
  set_frontier(*publisher, kPrefix + kPageTokens);
  expect(mutable_image_of(*pool, publisher->slot) != at_boundary,
         "clone: the publisher has moved past the boundary it published");

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "clone: claim");
  expect(mutable_image_of(*pool, claimant->slot) == at_boundary,
         "clone: a claimant's GDN state, conv taps and penalty counts are the publisher's at the "
         "prefix's end");
  expect(claimant->position == kPrefix, "clone: a claimant stands where the prefix ends");
  expect(claimant->pending_token == publisher->pending_token,
         "clone: a claimant carries the pending token the prefix ended on");
  expect(ignis_seq_at_chunk_boundary(*claimant),
         "clone: every layer's frontier is the prefix's end");

  const struct ignis_seq_prefix_stats after = stats_of(prefix, "clone: prefix stats");
  expect(after.clone_count == 1, "clone: the claim is counted");
  expect(after.last_clone_micros > 0.0, "clone: the claim's cost is measured, not assumed");
  expect(after.refcount == 3, "clone: handle, publisher and claimant");

  // A second claimant is cloned from the prefix, not from the first
  // claimant: releasing one claimant must leave the others alone.
  ignis_seq *second = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &second), 0, "clone: second claim");
  fill_device(pool->token_counts_for(claimant->slot),
              static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t), 0x1234u);
  const std::vector<unsigned char> second_before = mutable_image_of(*pool, second->slot);
  ignis_seq_release(pool, claimant);
  expect(mutable_image_of(*pool, second->slot) == second_before,
         "clone: releasing one claimant leaves another claimant's state untouched");
  expect(second_before == at_boundary, "clone: the second claimant got the same state");

  ignis_seq_release(pool, second);
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_pool_free(pool);
}

// ---- 3. a page is freed only when the last holder releases ----------------

void check_the_last_holder_frees_the_pages() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "lifetime: pool create");
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "lifetime: pool stats empty");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "lifetime: alloc publisher");
  give_history(*pool, *publisher, kPrefix, 0x44u);
  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, &prefix), 0, "lifetime: publish");
  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "lifetime: claim");

  const std::vector<std::int32_t> shared = row_of(*pool, claimant->slot, 2);

  // The publisher goes first, and its handle too: the prefix's pages stay
  // out of the free list because a claimant still holds them.
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  struct ignis_seq_pool_stats with_claimant{};
  expect_rc(ignis_seq_pool_stats(pool, &with_claimant), 0, "lifetime: pool stats with claimant");
  expect(empty.kv_free_pages - with_claimant.kv_free_pages == 2 + 4,
         "lifetime: the shared pages and the claimant's tail are still out of the pool");
  // And the claimant still reads them: a freed page could be handed to the
  // next allocation, so this is the check that the refcount is load-bearing.
  expect(row_of(*pool, claimant->slot, 2) == shared,
         "lifetime: the surviving claimant still addresses the shared pages");
  expect(ignis_seq_alloc(pool, kContext, &publisher) == 0, "lifetime: another sequence allocates");
  const std::vector<std::int32_t> other = row_of(*pool, publisher->slot, 6);
  for (std::int32_t page : other) {
    expect(page != shared[0] && page != shared[1],
           "lifetime: a shared page is never handed to another sequence");
  }
  ignis_seq_release(pool, publisher);

  ignis_seq_release(pool, claimant);
  struct ignis_seq_pool_stats after{};
  expect_rc(ignis_seq_pool_stats(pool, &after), 0, "lifetime: pool stats after");
  expect(after.kv_free_pages == after.kv_page_group_count,
         "lifetime: the last holder's release returns the shared pages");
  ignis_seq_pool_free(pool);
}

// ---- 4. the refusals ------------------------------------------------------

void check_refusals() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "refuse: pool create");
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "refuse: pool stats empty");

  ignis_seq *seq = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &seq), 0, "refuse: alloc");
  give_history(*pool, *seq, kPrefix, 0x55u);

  ignis_seq_prefix *prefix = nullptr;
  // A prefix is whole pages: a partial page would be written by its owner
  // and read by its claimants.
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix + 1, &prefix), -1,
            "refuse: a partial page is not a prefix");
  expect(prefix == nullptr, "refuse: nothing is published on a refusal");
  expect_rc(ignis_seq_prefix_publish(pool, seq, 0, &prefix), -1, "refuse: zero tokens");
  // The frontier must be exactly the prefix: the state a claimant gets is
  // the state at the prefix's end.
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPageTokens, &prefix),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: a frontier past the prefix");
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix + kPageTokens, &prefix),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: a frontier short of the prefix");
  // Mid-chunk: one layer ahead of the rest.
  seq->gqa_positions[0] = static_cast<std::uint32_t>(kPrefix + 1);
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, &prefix), IGNIS_SEQ_ERR_NOT_AT_BOUNDARY,
            "refuse: a mid-chunk publisher");
  seq->gqa_positions[0] = static_cast<std::uint32_t>(kPrefix);

  struct ignis_seq_pool_stats refused{};
  expect_rc(ignis_seq_pool_stats(pool, &refused), 0, "refuse: pool stats after refusals");
  expect(refused.kv_free_pages == empty.kv_free_pages - 6,
         "refuse: a refused publish charges the pool nothing");

  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, &prefix), 0, "refuse: publish");
  // A sequence claims at most one prefix, and a claimed head is not its own
  // to publish.
  ignis_seq_prefix *second = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, &second), -1,
            "refuse: a second publish from a sequence that already shares its head");
  expect(second == nullptr, "refuse: nothing is published on a second publish");

  // A reservation that leaves the claimant no page of its own is refused:
  // it would have nowhere to write without touching a shared page.
  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kPrefix, prefix, &claimant), -1,
            "refuse: a reservation with no page of its own");
  expect(claimant == nullptr, "refuse: nothing is allocated on a refusal");
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "refuse: claim");

  // A sequence whose history is not all its own cannot be moved as one blob
  // (P4-10 against P4-06): the leading pages belong to the prefix.
  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, claimant, &bytes), IGNIS_SEQ_ERR_SHARED_PREFIX,
            "refuse: a claimant has no whole-sequence snapshot");
  std::vector<unsigned char> scratch(4096);
  expect_rc(ignis_seq_snapshot(pool, claimant, scratch.data(), scratch.size()),
            IGNIS_SEQ_ERR_SHARED_PREFIX, "refuse: a claimant is not snapshotted");
  expect_rc(ignis_seq_restore(pool, claimant, scratch.data(), scratch.size()),
            IGNIS_SEQ_ERR_SHARED_PREFIX, "refuse: a claimant is not restored into");
  // The publisher is a claimant of its own prefix, so the same holds for it.
  expect_rc(ignis_seq_snapshot_size(pool, seq, &bytes), IGNIS_SEQ_ERR_SHARED_PREFIX,
            "refuse: a publisher shares its own head too");

  ignis_seq_release(pool, claimant);
  ignis_seq_release(pool, seq);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_pool_free(pool);
}

// ---- 5. what the clone costs ---------------------------------------------

void report_clone_cost() {
  // One publisher plus one claimant at full context. The clone's size does
  // not depend on the prefix length -- the GDN slot, the conv taps and the
  // penalty-count row are the whole of it -- so one cell is the measurement,
  // and the prefix length below only decides how much prefill it skips.
  constexpr std::uint32_t kFullContext = 40960;
  const ignis_seq_pool_spec spec       = qwen38_27b_spec(kFullContext, 2);
  ignis_seq_pool *pool                 = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0) {
    std::fprintf(stderr, "FAIL: cost: pool create: %s\n", ignis_seq_last_error());
    ++failures;
    return;
  }

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kFullContext, &publisher), 0, "cost: alloc publisher");
  set_frontier(*publisher, kFullContext / 2);
  publisher->pending_token = 7;

  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kFullContext / 2, &prefix), 0,
            "cost: publish");
  const struct ignis_seq_prefix_stats published = stats_of(prefix, "cost: prefix stats");

  // One untimed claim first (the allocator and the driver both warm up),
  // then the mean of three.
  constexpr int kReps = 3;
  double total        = 0.0;
  for (int rep = 0; rep <= kReps; ++rep) {
    ignis_seq *claimant = nullptr;
    expect_rc(ignis_seq_alloc_shared(pool, kFullContext, prefix, &claimant), 0, "cost: claim");
    const struct ignis_seq_prefix_stats stats = stats_of(prefix, "cost: claim stats");
    if (rep > 0) {
      total += stats.last_clone_micros;
    }
    ignis_seq_release(pool, claimant);
  }

  const double mean_micros = total / kReps;
  const double mib = static_cast<double>(published.clone_image_bytes) / (1024.0 * 1024.0);
  std::printf("[prefix clone cost] qwen3.8-27b, hq-e8-2b, %u-token prefix\n", published.tokens);
  std::printf("  shared KV pages   : %u (%.2f MiB of history, not copied)\n", published.pages,
              static_cast<double>(published.pages) *
                  static_cast<double>(
                      ninfer::paged_kv_host_image_bytes(pool->kv_pool, 1)) /
                  (1024.0 * 1024.0));
  std::printf("  cloned state      : %.2f MiB device to device\n", mib);
  std::printf("  clone             : %.3f ms (mean of %d)\n", mean_micros / 1000.0, kReps);
  std::printf("  effective         : %.1f GB/s\n",
              static_cast<double>(published.clone_image_bytes) / (mean_micros * 1e-6) / 1e9);
  std::fflush(stdout);

  expect(mean_micros > 0.0, "cost: the clone is measured");
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_pool_free(pool);
}

} // namespace

int main() {
  int device_count            = 0;
  const cudaError_t available = cudaGetDeviceCount(&device_count);
  if (cuda_unavailable(available) || device_count == 0) {
    // ADR 0006: a missing GPU is a failure here, never a skip.
    std::fprintf(stderr, "FAIL: no CUDA device (%s)\n", cudaGetErrorString(available));
    return 1;
  }

  check_publish_shares_pages_and_charges_once();
  check_a_claimant_receives_the_mutable_state();
  check_the_last_holder_frees_the_pages();
  check_refusals();
  report_clone_cost();

  if (failures != 0) {
    std::cerr << failures << " prefix reuse check(s) failed\n";
    return 1;
  }
  std::cout << "device prefix reuse: ok\n";
  return 0;
}
