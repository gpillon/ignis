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
//   3b. **a chained publish extends a claimed head** (GitHub #187): a
//      sequence standing on a prefix publishes the pages it warmed past it,
//      over the ones it claimed — one reference per link, the row in logical
//      page order, and the whole chain freed in one cascade;
//   4. **every refusal refuses**, leaving the sequence and the pool as they
//      were: a partial page, a frontier that is not the prefix, a mid-chunk
//      publisher, a chained publish that reaches no further than what is
//      already shared, a reservation with no page of its own, and a state
//      restore into a target whose history is still shared. A snapshot of a
//      claimant materializes that history since GitHub #190;
//   5. **the image lives in a retained slot the caller names** (GitHub #215):
//      a publish into a slot out of range or still held is refused, the slot
//      comes back with the publish handle, and a prefix whose handle is gone
//      is never cloned from again.
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

#include "ignis_model.h"
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
  spec.retained_slot_count = 4;
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
  spec.retained_slot_count = 1;
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
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, 0, &prefix), 0, "share: publish");

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
  expect(prefix_stats.clone_image_bytes == pool->slot_state_bytes,
         "share: the prefix keeps an image of the mutable sections in one retained slot");
  expect(pool->retained_held[0], "share: and holds that slot");
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
  publisher->rope_delta = -55; // a multimodal publisher (GitHub #194)
  const std::vector<unsigned char> at_boundary = mutable_image_of(*pool, publisher->slot);

  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, 0, &prefix), 0, "clone: publish");

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
  // GitHub #194: a snapshot carries the rope delta, a clone does not -- the
  // publisher's is its whole prompt's, and a claimant's comes from its own
  // prefill span.
  expect(publisher->rope_delta != 0 && claimant->rope_delta == 0,
         "clone: a claimant does not inherit the publisher's rope delta");
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

// P5-03 (GitHub #152): on a pool with the DFlash2 drafter, the device clone
// carries the drafter's window and its checkpoint like any other CLONE
// section -- a sibling's window is the publisher's at the prefix's end.
std::vector<unsigned char> lane_image_of(const ninfer::CyclicKVCache &cache, std::int32_t slot) {
  std::vector<unsigned char> image(cache.lane_host_bytes());
  cache.copy_lane_to_host(slot, image.data(), nullptr);
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  return image;
}

void fill_lane(ninfer::CyclicKVCache &cache, std::int32_t slot, std::uint32_t seed) {
  const std::vector<unsigned char> host = pattern(cache.lane_host_bytes(), seed);
  cache.copy_lane_from_host(host.data(), slot, nullptr);
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void check_a_claimant_receives_the_drafter_window() {
  ignis_seq_pool_spec spec  = small_spec();
  spec.speculative_backend  = IGNIS_SPECULATIVE_DFLASH2;
  ignis_seq_pool *pool      = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "drafter clone: pool create");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "drafter clone: alloc publisher");
  give_history(*pool, *publisher, kPrefix, 0x61u);
  fill_lane(*pool->dflash2_window, publisher->slot, 0x62u);
  fill_lane(*pool->dflash2_checkpoint, publisher->slot, 0x63u);
  publisher->dflash2_position = kPrefix;
  const std::vector<unsigned char> window     = lane_image_of(*pool->dflash2_window, publisher->slot);
  const std::vector<unsigned char> checkpoint = lane_image_of(*pool->dflash2_checkpoint, publisher->slot);
  const std::vector<unsigned char> at_boundary = mutable_image_of(*pool, publisher->slot);

  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, 0, &prefix), 0,
            "drafter clone: publish");
  const std::uint64_t lane_bytes = pool->dflash2_lane_bytes();
  expect(stats_of(prefix, "drafter clone: prefix stats").clone_image_bytes >= 2 * lane_bytes,
         "drafter clone: the prefix's device image holds both drafter lanes");

  // The publisher's drafter moves on past the prefix.
  fill_lane(*pool->dflash2_window, publisher->slot, 0x71u);
  fill_lane(*pool->dflash2_checkpoint, publisher->slot, 0x72u);
  publisher->dflash2_position = kPrefix + kPageTokens;

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "drafter clone: claim");
  expect(lane_image_of(*pool->dflash2_window, claimant->slot) == window,
         "drafter clone: the sibling's window equals the publisher's at the prefix's end");
  expect(lane_image_of(*pool->dflash2_checkpoint, claimant->slot) == checkpoint,
         "drafter clone: the sibling's checkpoint equals the publisher's at the prefix's end");
  expect(mutable_image_of(*pool, claimant->slot) == at_boundary,
         "drafter clone: the rest of the mutable state still clones");
  expect(claimant->dflash2_position == kPrefix,
         "drafter clone: the sibling's drafter frontier is the prefix's end");

  ignis_seq_release(pool, claimant);
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
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, 0, &prefix), 0, "lifetime: publish");
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

// ---- 3b. a chained publish extends a claimed head (GitHub #187) -----------

// Warm `own_pages` of `seq`'s own pages and put it at the boundary they end
// on. Unlike `give_history` this writes only pages the sequence owns, which
// is what a claimant standing on a prefix actually has to write.
void extend_history(ignis_seq_pool &pool, ignis_seq &seq, std::uint32_t own_pages,
                    std::uint64_t tokens, std::uint32_t seed) {
  set_frontier(seq, tokens);
  seq.pending_token = static_cast<std::int32_t>(1000 + seed);
  std::uint32_t salt = seed;
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    auto *base                  = static_cast<unsigned char *>(plane.data);
    const auto page_ids         = seq.kv.page_ids();
    for (std::uint32_t page = 0; page < own_pages && page < page_ids.size(); ++page) {
      fill_device(base + static_cast<std::int64_t>(page_ids[page]) * plane.nb[3],
                  static_cast<std::size_t>(plane.nb[3]), ++salt);
    }
  }
}

void check_a_chained_publish_extends_a_claimed_head() {
  // GitHub #187. A sequence that resumed from retained state and prefilled
  // past it has no head of its own to publish -- the pages below its own
  // generation opener are partly the entry it claimed. It publishes a
  // *chained* entry: the pages it warmed itself, over the ones it claimed.
  // That is what puts its own opener inside a page it alone writes, which is
  // exactly what ignis_seq_checkpoint_capture demands, and it is why every
  // iteration of an agent's tool loop can leave a checkpoint instead of only
  // the first.
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "chain: pool create");
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "chain: pool stats empty");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "chain: alloc publisher");
  give_history(*pool, *publisher, kPrefix, 0x71u);
  ignis_seq_prefix *parent = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, 0, &parent), 0, "chain: publish parent");

  // Turn N+1: it claims the parent, prefills one page past it, and publishes
  // what it now covers -- three pages, only one of them its own.
  ignis_seq *turn2 = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, parent, &turn2), 0, "chain: claim parent");
  const std::vector<std::int32_t> parent_ids = row_of(*pool, turn2->slot, 2);
  const std::uint32_t kChained               = 3 * kPageTokens;
  extend_history(*pool, *turn2, 1, kChained, 0x72u);
  const std::int32_t own_page = turn2->kv.page_ids()[0];

  struct ignis_seq_pool_stats before{};
  expect_rc(ignis_seq_pool_stats(pool, &before), 0, "chain: pool stats before");
  ignis_seq_prefix *child = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, turn2, kChained, 1, &child), 0, "chain: publish chained");
  struct ignis_seq_pool_stats after{};
  expect_rc(ignis_seq_pool_stats(pool, &after), 0, "chain: pool stats after");
  expect(after.kv_free_pages == before.kv_free_pages,
         "chain: a chained publish charges the pool nothing new either");

  const struct ignis_seq_prefix_stats child_stats = stats_of(child, "chain: child stats");
  expect(child_stats.tokens == kChained, "chain: the entry covers the whole head");
  expect(child_stats.pages == 1, "chain: but owns only the page it warmed itself");
  expect(child_stats.refcount == 2, "chain: the handle and the publishing sequence");
  const struct ignis_seq_prefix_stats parent_stats = stats_of(parent, "chain: parent stats");
  expect(parent_stats.refcount == 3,
         "chain: the publisher, the returned handle, and the child -- the claimant's own "
         "reference moved to the child rather than a new one being taken");

  const struct ignis_seq_stats turn2_stats = seq_stats_of(turn2, "chain: turn 2 seq stats");
  expect(turn2_stats.shared_pages == 3, "chain: it now shares the whole three-page head");
  expect(turn2_stats.mapped_pages == 6, "chain: and still maps its whole reservation");
  // Logical page order: the parent's two pages, then the child's one, then
  // the tail the publisher re-reserved. A row that put the child's page first
  // would answer from history in the wrong order, silently.
  const std::vector<std::int32_t> row = row_of(*pool, turn2->slot, 3);
  expect(row[0] == parent_ids[0] && row[1] == parent_ids[1],
         "chain: the parent's pages are still logical pages 0 and 1");
  expect(row[2] == own_page, "chain: and the child's page is logical page 2");

  // Turn N+2 claims the chained entry and shares all three pages, the
  // parent's included.
  ignis_seq *turn3 = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, child, &turn3), 0, "chain: claim the chain");
  const struct ignis_seq_stats turn3_stats = seq_stats_of(turn3, "chain: turn 3 seq stats");
  expect(turn3_stats.shared_pages == 3, "chain: a claimant shares the whole chain");
  const std::vector<std::int32_t> claimed_row = row_of(*pool, turn3->slot, 3);
  expect(claimed_row == row, "chain: over the same physical pages, in the same order");

  // Lifetime: the parent outlives its own publisher, and the whole chain
  // comes back in one release when the last holder of the child lets go.
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, parent);
  ignis_seq_release(pool, turn2);
  struct ignis_seq_pool_stats held{};
  expect_rc(ignis_seq_pool_stats(pool, &held), 0, "chain: pool stats with the chain held");
  expect(empty.kv_free_pages - held.kv_free_pages == 3 + 3,
         "chain: the three shared pages and turn 3's tail are still out of the pool");
  expect(row_of(*pool, turn3->slot, 3) == row,
         "chain: and the survivor still addresses every page of the chain");

  ignis_seq_release(pool, turn3);
  ignis_seq_prefix_release(pool, child);
  struct ignis_seq_pool_stats freed{};
  expect_rc(ignis_seq_pool_stats(pool, &freed), 0, "chain: pool stats after");
  expect(freed.kv_free_pages == freed.kv_page_group_count,
         "chain: the last release returns the child's page and cascades to the parent's");
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
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix + 1, 0, &prefix), -1,
            "refuse: a partial page is not a prefix");
  expect(prefix == nullptr, "refuse: nothing is published on a refusal");
  expect_rc(ignis_seq_prefix_publish(pool, seq, 0, 0, &prefix), -1, "refuse: zero tokens");
  // The frontier must be exactly the prefix: the state a claimant gets is
  // the state at the prefix's end.
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPageTokens, 0, &prefix),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: a frontier past the prefix");
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix + kPageTokens, 0, &prefix),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: a frontier short of the prefix");
  // Mid-chunk: one layer ahead of the rest.
  seq->gqa_positions[0] = static_cast<std::uint32_t>(kPrefix + 1);
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, 0, &prefix), IGNIS_SEQ_ERR_NOT_AT_BOUNDARY,
            "refuse: a mid-chunk publisher");
  seq->gqa_positions[0] = static_cast<std::uint32_t>(kPrefix);

  struct ignis_seq_pool_stats refused{};
  expect_rc(ignis_seq_pool_stats(pool, &refused), 0, "refuse: pool stats after refusals");
  expect(refused.kv_free_pages == empty.kv_free_pages - 6,
         "refuse: a refused publish charges the pool nothing");

  expect(!pool->retained_held[0], "refuse: a refused publish holds no retained slot");
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, 0, &prefix), 0, "refuse: publish");
  // GitHub #187 made a second publish legal — as a *chained* entry over what
  // the sequence already shares. One that reaches no further covers nothing of
  // its own, and would be a second entry over pages it does not own, so it is
  // refused where a second publish used to be refused outright.
  ignis_seq_prefix *second = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, 1, &second), -1,
            "refuse: a chained publish that reaches no further than what is already shared");
  expect(second == nullptr, "refuse: nothing is published on it");

  // A reservation that leaves the claimant no page of its own is refused:
  // it would have nowhere to write without touching a shared page.
  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kPrefix, prefix, &claimant), -1,
            "refuse: a reservation with no page of its own");
  expect(claimant == nullptr, "refuse: nothing is allocated on a refusal");
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "refuse: claim");

  // GitHub #190: a sequence whose leading pages belong to a prefix is
  // snapshotted as a self-contained blob. The target of a restore still may
  // not share pages, because overwriting them would corrupt other claimants.
  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, claimant, &bytes), 0,
            "materialize: size a claimant's whole-sequence snapshot");
  std::vector<unsigned char> scratch(bytes);
  expect_rc(ignis_seq_snapshot(pool, claimant, scratch.data(), scratch.size()), 0,
            "materialize: snapshot a claimant including its shared head");
  expect_rc(ignis_seq_restore(pool, claimant, scratch.data(), scratch.size()),
            IGNIS_SEQ_ERR_SHARED_PREFIX, "refuse: a claimant is not restored into");
  // The publisher is a claimant of its own prefix and materializes too.
  expect_rc(ignis_seq_snapshot_size(pool, seq, &bytes), 0,
            "materialize: a publisher shares its own head too");

  ignis_seq_release(pool, claimant);
  ignis_seq_release(pool, seq);
  ignis_seq *restored = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &restored), 0, "materialize: alloc restore target");
  expect_rc(ignis_seq_restore(pool, restored, scratch.data(), scratch.size()), 0,
            "materialize: restore without the shared handle");
  std::uint64_t restored_bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, restored, &restored_bytes), 0,
            "materialize: size restored state");
  std::vector<unsigned char> again(restored_bytes);
  expect_rc(ignis_seq_snapshot(pool, restored, again.data(), again.size()), 0,
            "materialize: re-snapshot restored state");
  expect(again == scratch, "materialize: the standalone round trip is byte-exact");
  ignis_seq_release(pool, restored);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_pool_free(pool);
}

// ---- 5. the image lives in a retained slot (GitHub #215) -----------------

void check_the_image_lives_in_a_retained_slot() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "slot: pool create");

  ignis_seq *first = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &first), 0, "slot: alloc first");
  give_history(*pool, *first, kPrefix, 0xA1u);
  ignis_seq *second = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &second), 0, "slot: alloc second");
  give_history(*pool, *second, kPrefix, 0xA2u);

  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, first, kPrefix, spec.retained_slot_count, &prefix), -1,
            "slot: a retained slot past the pool's is refused");
  expect(prefix == nullptr && first->prefix == nullptr, "slot: and nothing was published");

  // The image is the publisher's state, in the retained slot it was named.
  const std::vector<unsigned char> at_boundary = mutable_image_of(*pool, first->slot);
  expect_rc(ignis_seq_prefix_publish(pool, first, kPrefix, 3, &prefix), 0, "slot: publish");
  expect(mutable_image_of(*pool, ignis_seq_retained_pool_slot(*pool, 3)) == at_boundary,
         "slot: the image is the publisher's state, held in retained slot 3");

  // A second publish may not write over it, nor may a plain store.
  ignis_seq_prefix *other = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, second, kPrefix, 3, &other), -1,
            "slot: a publish into a slot another prefix holds is refused");
  expect(other == nullptr && second->prefix == nullptr, "slot: and changes nothing");
  expect_rc(ignis_seq_retained_store(pool, second, 3), -1,
            "slot: a store into a held slot is refused");
  expect(mutable_image_of(*pool, ignis_seq_retained_pool_slot(*pool, 3)) == at_boundary,
         "slot: the held image is untouched");

  // The pages outlive the handle; the image does not.
  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &claimant), 0, "slot: claim");
  ignis_seq_release(pool, first);
  ignis_seq_prefix_release(pool, prefix);
  expect(!pool->retained_held[3], "slot: the slot came back with the handle");
  expect(stats_of(prefix, "slot: stats after release").clone_image_bytes == 0,
         "slot: the prefix holds no image any more");
  ignis_seq *late = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, prefix, &late), -1,
            "slot: a prefix whose handle is gone is not cloned from");
  expect(late == nullptr, "slot: and allocates nothing");
  expect_rc(ignis_seq_prefix_publish(pool, second, kPrefix, 3, &other), 0,
            "slot: the freed slot takes the next image");
  expect(mutable_image_of(*pool, claimant->slot) == at_boundary,
         "slot: the claimant built before the release kept its own copy");

  ignis_seq_release(pool, claimant);
  ignis_seq_release(pool, second);
  ignis_seq_prefix_release(pool, other);
  struct ignis_seq_pool_stats end{};
  expect_rc(ignis_seq_pool_stats(pool, &end), 0, "slot: pool stats end");
  expect(end.kv_free_pages == end.kv_page_group_count, "slot: every page is back");
  ignis_seq_pool_free(pool);
}

// ---- 6. what the clone costs ---------------------------------------------

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
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kFullContext / 2, 0, &prefix), 0,
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
  // And that it never left the card. This host runs the 5090 on a PCIe Gen 3
  // x16 link measured at ~12 GB/s in either direction
  // (docs/findings/2026-09-12-sequence-snapshot-transfer-cost.md), so
  // 147.76 MiB could not cross it in under ~12 ms one way, let alone make the
  // round trip a restore-from-a-sibling's-snapshot would need. A clone that
  // came in under a third of one crossing cannot have taken that route --
  // which is the acceptance criterion "no prefix-reuse path performs a host
  // round-trip", checked rather than asserted in a comment.
  const double one_pcie_crossing_micros =
      static_cast<double>(published.clone_image_bytes) / 12e9 * 1e6;
  expect(mean_micros < one_pcie_crossing_micros / 3.0,
         "cost: the clone is far too fast to have crossed PCIe");
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_pool_free(pool);
}

} // namespace

// ---- 3c. a retained prefix spills to a blob and comes back (GitHub #190) --
//
// What leaves the device is the blob its publisher would have written while
// standing on the prefix. What comes back is a *published prefix* again:
// restored into a fresh sequence, published there, and claimable -- with the
// same bytes, so a burst member that claims it stands exactly where one that
// claimed the original stood.

void check_a_spilled_prefix_comes_back_as_the_same_prefix(bool dflash2) {
  ignis_seq_pool_spec spec = small_spec();
  if (dflash2) {
    spec.speculative_backend = IGNIS_SPECULATIVE_DFLASH2;
  }
  ignis_seq_pool *pool = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "spill: pool create");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "spill: alloc publisher");
  give_history(*pool, *publisher, kPrefix, 0x91u);
  publisher->rope_delta = -77; // a multimodal publisher (GitHub #194)
  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, 2, &prefix), 0, "spill: publish");

  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_prefix_snapshot_size(pool, prefix, &bytes), 0, "spill: size");
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_prefix_snapshot(pool, prefix, blob.data(), bytes), 0, "spill: snapshot");
  std::uint64_t publisher_bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, publisher, &publisher_bytes), 0, "spill: publisher size");
  std::vector<unsigned char> own(static_cast<std::size_t>(publisher_bytes));
  expect_rc(ignis_seq_snapshot(pool, publisher, own.data(), publisher_bytes), 0,
            "spill: publisher snapshot");
  // GitHub #194: the one word the two differ in is the rope delta, which a
  // prefix does not carry -- its publisher's is the whole prompt's.
  if (own.size() == blob.size()) {
    ignis_seq_snapshot_header header{};
    std::memcpy(&header, own.data(), sizeof(header));
    std::uint64_t at = 0;
    for (const ignis_seq_section &section : ignis_seq_section_table(*pool, header.kv_page_count)) {
      if (section.kind == IGNIS_SEQ_SECTION_PROGRESS) {
        at = section.offset + offsetof(ignis_seq_progress_image, rope_delta);
      }
    }
    std::int32_t own_delta  = 0;
    std::int32_t blob_delta = 0;
    std::memcpy(&own_delta, own.data() + at, sizeof(own_delta));
    std::memcpy(&blob_delta, blob.data() + at, sizeof(blob_delta));
    expect(own_delta == -77 && blob_delta == 0,
           "spill: the publisher's blob carries its rope delta, the prefix's does not");
    std::memset(own.data() + at, 0, sizeof(own_delta));
  }
  expect(own == blob,
         "spill: the prefix's blob is its publisher's, standing on it, but for the rope delta");

  // Every device holder goes: nothing of the prefix is left on the card.
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);

  // Back up: restore into a fresh sequence, publish there, let it go.
  ignis_seq *carrier = nullptr;
  expect_rc(ignis_seq_alloc(pool, kPrefix + kPageTokens, &carrier), 0, "spill: alloc carrier");
  expect_rc(ignis_seq_restore(pool, carrier, blob.data(), bytes), 0, "spill: restore");
  ignis_seq_prefix *returned = nullptr;
  // Into the very slot the original held: it came back with the handle.
  expect_rc(ignis_seq_prefix_publish(pool, carrier, kPrefix, 2, &returned), 0,
            "spill: publish the restored head");
  ignis_seq_release(pool, carrier);

  std::uint64_t again_bytes = 0;
  expect_rc(ignis_seq_prefix_snapshot_size(pool, returned, &again_bytes), 0, "spill: size again");
  std::vector<unsigned char> again(static_cast<std::size_t>(again_bytes));
  expect_rc(ignis_seq_prefix_snapshot(pool, returned, again.data(), again_bytes), 0,
            "spill: snapshot again");
  expect(again == blob, "spill: the prefix that came back is the prefix that left");

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, returned, &claimant), 0,
            "spill: a burst member claims the returned prefix");
  std::uint64_t claimant_bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, claimant, &claimant_bytes), 0, "spill: claimant size");
  std::vector<unsigned char> claimed(static_cast<std::size_t>(claimant_bytes));
  expect_rc(ignis_seq_snapshot(pool, claimant, claimed.data(), claimant_bytes), 0,
            "spill: claimant snapshot");
  expect(claimed == blob, "spill: and stands exactly where the original's claimant stood");

  ignis_seq_release(pool, claimant);
  ignis_seq_prefix_release(pool, returned);
  ignis_seq_pool_free(pool);
}

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
  check_a_claimant_receives_the_drafter_window();
  check_the_last_holder_frees_the_pages();
  check_a_chained_publish_extends_a_claimed_head();
  check_a_spilled_prefix_comes_back_as_the_same_prefix(false);
  check_a_spilled_prefix_comes_back_as_the_same_prefix(true);
  check_refusals();
  check_the_image_lives_in_a_retained_slot();
  report_clone_cost();

  if (failures != 0) {
    std::cerr << failures << " prefix reuse check(s) failed\n";
    return 1;
  }
  std::cout << "device prefix reuse: ok\n";
  return 0;
}
