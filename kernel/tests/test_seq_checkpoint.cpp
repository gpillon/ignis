// Leaf-level prompt checkpoint test (GitHub #186, ADR 0029).
//
// Ours, not vendored. `test_seq_prefix.cpp` proves what a *shared prefix*
// does; this file proves the three things a checkpoint adds, none of which
// the flat ABI can report on its own:
//
//   1. **the state is captured at the opener, not at the page below it.** A
//      claimant's GDN slot, conv taps and penalty-count row come out
//      byte-identical to the capturing sequence's state at the opener — which
//      is up to 63 tokens further on than the prefix's own image.
//   2. **the partial tail page is copied, and every whole page below it is
//      shared.** The claimant's first own physical page carries the capturing
//      sequence's bytes for that page, while its head addresses the same
//      physical pages the prefix holds and the pool is charged for them once.
//   3. **the capture perturbs nothing, and the claim consumes nothing.** The
//      capturing sequence's row, reservation and state are what they were,
//      and N claimants all succeed — a retry, a regenerate and two forks of
//      one history all hit.
//
// It also asserts the acceptance criterion the ABI is the only place to check
// it: **the penalty-count row captured at the opener is all zeros**, because
// nothing has been sampled at that point in a real request's life.
//
// What this file deliberately does not claim is that a claimant *generates*
// what a cold prefill split at the same opener generates. That needs a real
// model and real tokens, and it lives one level up in
// crates/core/tests/prompt_checkpoint_gpu.rs.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE is set on this test
// (kernel/tests/CMakeLists.txt), so a missing/busy GPU fails it, never skips.

#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_seq_checkpoint_internal.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
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

void fill_lane(ninfer::CyclicKVCache &cache, std::int32_t slot, std::uint32_t seed) {
  const std::vector<unsigned char> host = pattern(cache.lane_host_bytes(), seed);
  cache.copy_lane_from_host(host.data(), slot, nullptr);
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
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

// Write a known pattern into every page of `seq`'s own allocation that its
// history reaches, and into every mutable section of its slot.
void dirty_state(ignis_seq_pool &pool, ignis_seq &seq, std::uint32_t own_pages,
                 std::uint32_t seed) {
  std::uint32_t salt   = seed;
  const auto page_ids  = seq.kv.page_ids();
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    auto *base                  = static_cast<unsigned char *>(plane.data);
    for (std::uint32_t page = 0; page < own_pages && page < page_ids.size(); ++page) {
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
  if (pool.has_dflash2()) {
    fill_lane(*pool.dflash2_window, seq.slot, ++salt);
    fill_lane(*pool.dflash2_checkpoint, seq.slot, ++salt);
    seq.dflash2_position = seq.position;
  }
}

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

// A slot's whole mutable state, as one host image.
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

struct ignis_seq_checkpoint_stats stats_of(const ignis_seq_checkpoint *checkpoint,
                                           const char *label) {
  struct ignis_seq_checkpoint_stats stats{};
  expect_rc(ignis_seq_checkpoint_stats(checkpoint, &stats), 0, label);
  return stats;
}

// A small pool in `kv_format`. The head geometry follows the format: hq-e8-2b
// stores fixed-budget rows at the codec's own 256-wide head (the engine's real
// one), and it lays a page out over *four* planes per GQA layer against BF16's
// two -- which is exactly why the claim check below runs against both. A tail
// page that is copied plane by plane is where a plane set can be got wrong.
ignis_seq_pool_spec small_spec(int32_t kv_format = IGNIS_KV_FORMAT_BF16) {
  const bool hq            = kv_format == IGNIS_KV_FORMAT_HQ_E8_2B;
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = hq ? 4 : 2;
  spec.head_dim            = hq ? 256 : 8;
  spec.kv_format           = kv_format;
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

constexpr std::uint32_t kPageTokens = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
constexpr std::uint32_t kContext    = 384;
// The shared prefix: two whole pages. The generation opener sits 40 tokens
// into the sequence's own third page — the shape a rendered prompt has, where
// the opener is wherever `<|im_start|>assistant\n` happens to end.
constexpr std::uint32_t kPrefix = 2 * kPageTokens;
constexpr std::uint32_t kOpener = kPrefix + 40;

// A publisher standing at the opener, with a prefix of `kPrefix` under it and
// a known pattern everywhere. `*out_prefix` receives the publish handle.
ignis_seq *publisher_at_opener(ignis_seq_pool *pool, ignis_seq_prefix **out_prefix,
                               const char *label) {
  ignis_seq *seq = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &seq), 0, label);
  // Stand at the prefix boundary, dirty the head, publish, then walk on to
  // the opener and dirty what the sequence now owns — exactly the order a
  // real prefill takes: the chunk that lands on the publish point, then the
  // short chunk that lands on the opener.
  set_frontier(*seq, kPrefix);
  dirty_state(*pool, *seq, 2, 0x31u);
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, out_prefix), 0, label);
  set_frontier(*seq, kOpener);
  seq->pending_token = 4242;
  seq->rope_delta    = -42;
  dirty_state(*pool, *seq, 1, 0x57u);
  // Note what is NOT done here: the penalty-count row is left exactly as the
  // sequence's own life left it. `ignis_seq_alloc` zeroed it and nothing has
  // sampled since, which is the state a real request is in at its generation
  // opener -- the scheduler only ever asks for a capture from an intermediate,
  // greedy chunk. Zeroing it here would stage the property the test asserts.
  return seq;
}

// ---- 1. the capture reads, and changes nothing ---------------------------

void check_capture_perturbs_nothing() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "capture: pool create");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "capture: publisher");

  const std::vector<std::int32_t> row_before = row_of(*pool, publisher->slot, 6);
  const std::vector<unsigned char> state_before = mutable_image_of(*pool, publisher->slot);
  struct ignis_seq_pool_stats pool_before{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_before), 0, "capture: pool stats before");

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "capture: capture at the opener");

  expect(row_of(*pool, publisher->slot, 6) == row_before,
         "capture: the capturing sequence's block-table row is untouched");
  expect(mutable_image_of(*pool, publisher->slot) == state_before,
         "capture: and so is every byte of its mutable state");
  expect(publisher->position == kOpener, "capture: and its frontier");
  struct ignis_seq_pool_stats pool_after{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_after), 0, "capture: pool stats after");
  expect(pool_after.kv_free_pages == pool_before.kv_free_pages,
         "capture: a capture costs the KV pool no page");

  const struct ignis_seq_checkpoint_stats stats = stats_of(checkpoint, "capture: stats");
  expect(stats.tokens == kOpener, "capture: the checkpoint reaches the opener, not the page");
  expect(stats.pages == 2, "capture: over the two whole pages the prefix holds");
  expect(stats.claim_count == 0, "capture: capturing is not a claim");
  std::uint64_t image_bytes = 0;
  expect_rc(ignis_seq_checkpoint_image_bytes(pool, &image_bytes), 0, "capture: image bytes query");
  expect(stats.image_bytes == image_bytes,
         "capture: a checkpoint costs exactly what the pool said one costs");

  // The prefix now has three holders: the publish handle, the publishing
  // sequence, and the checkpoint. That third one is what keeps the pages
  // alive once the request is gone.
  struct ignis_seq_prefix_stats prefix_stats{};
  expect_rc(ignis_seq_prefix_stats(prefix, &prefix_stats), 0, "capture: prefix stats");
  expect(prefix_stats.refcount == 3, "capture: the checkpoint holds the prefix too");

  ignis_seq_checkpoint_release(pool, checkpoint);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);
  ignis_seq_pool_free(pool);
}

// ---- 2. a claimant stands exactly where the capture stood ----------------

void check_claim_reproduces_the_state_at_the_opener(int32_t kv_format) {
  const ignis_seq_pool_spec spec = small_spec(kv_format);
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "claim: pool create");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "claim: publisher");

  const std::vector<std::int32_t> publisher_row = row_of(*pool, publisher->slot, 6);
  const std::vector<unsigned char> state_at_opener = mutable_image_of(*pool, publisher->slot);
  // The page the opener ends inside: the publisher's own first page.
  const std::int32_t publisher_own_page = publisher->kv.page_ids()[0];
  const std::vector<unsigned char> tail_page = page_image_of(*pool, publisher_own_page);

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "claim: capture");

  // The capturing request goes on and moves its state on -- including
  // rewriting the very physical page the checkpoint copied from, which is
  // exactly why that page has to be *copied* and not shared, and why the
  // capture had to happen at the opener rather than be read back later.
  set_frontier(*publisher, kOpener + 17);
  dirty_state(*pool, *publisher, 1, 0x99u);

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &claimant), 0,
            "claim: alloc from checkpoint");

  expect(claimant->position == kOpener, "claim: the claimant stands at the opener");
  expect(claimant->pending_token == 4242, "claim: with the capture's pending token");
  expect(claimant->rope_delta == 0,
         "claim: without the capturing sequence's rope delta (GitHub #194)");
  expect(claimant->shared_pages == 2, "claim: sharing the two whole pages below it");
  expect(mutable_image_of(*pool, claimant->slot) == state_at_opener,
         "claim: its mutable state is the capture's, byte for byte");

  // The head addresses the publisher's own physical pages; the partial tail
  // page is the claimant's own, carrying the capture's bytes.
  const std::vector<std::int32_t> claimant_row = row_of(*pool, claimant->slot, 6);
  expect(std::equal(publisher_row.begin(), publisher_row.begin() + 2, claimant_row.begin()),
         "claim: the head shares the same physical pages");
  expect(claimant_row[2] != publisher_row[2],
         "claim: the partial tail page is the claimant's own, not the publisher's");
  expect(page_image_of(*pool, claimant_row[2]) == tail_page,
         "claim: and carries the capture's bytes for that page");

  // Non-consuming: a second and a third claimant hit the same entry.
  ignis_seq *second = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &second), 0,
            "claim: a second claimant");
  expect(mutable_image_of(*pool, second->slot) == state_at_opener,
         "claim: the second claimant gets the same state");
  const struct ignis_seq_checkpoint_stats stats = stats_of(checkpoint, "claim: stats");
  expect(stats.claim_count == 2, "claim: both claims counted");
  expect(stats.last_claim_micros > 0.0, "claim: and the transfer was measured");

  ignis_seq_release(pool, second);
  ignis_seq_release(pool, claimant);
  ignis_seq_checkpoint_release(pool, checkpoint);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);
  ignis_seq_pool_free(pool);
}

// ---- 3. the penalty-count row at the opener is zero ----------------------

void check_penalty_counts_are_zero_at_the_opener() {
  // The acceptance criterion, checked where it is checkable: a request that
  // has not sampled anything has an all-zero count row, so a claimant that
  // resumes at the opener starts its own sampling from zero rather than from
  // someone else's history.
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "counts: pool create");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "counts: publisher");

  // The property first, on the sequence itself and before anything is
  // captured: a request that has reached its opener has sampled nothing, so
  // its count row is untouched from allocation. Nothing in this test put it
  // there.
  const std::size_t counts_bytes =
      static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t);
  const std::vector<unsigned char> at_opener =
      read_device(pool->token_counts_for(publisher->slot), counts_bytes);
  expect(std::all_of(at_opener.begin(), at_opener.end(),
                     [](unsigned char b) { return b == 0; }),
         "counts: a sequence standing at its opener has sampled nothing");

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "counts: capture");
  // The capturing request now samples, which is what a real one does the
  // moment its prompt is warm: the checkpoint must not have picked that up.
  fill_device(pool->token_counts_for(publisher->slot), counts_bytes, 0xB1u);

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &claimant), 0,
            "counts: claim");
  const std::vector<unsigned char> counts =
      read_device(pool->token_counts_for(claimant->slot), counts_bytes);
  expect(std::all_of(counts.begin(), counts.end(), [](unsigned char b) { return b == 0; }),
         "counts: the penalty-count row a claimant receives is all zeros");

  ignis_seq_release(pool, claimant);
  ignis_seq_checkpoint_release(pool, checkpoint);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);
  ignis_seq_pool_free(pool);
}

// ---- 4. the pages come back only when the last holder lets go ------------

void check_lifetime() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "lifetime: pool create");

  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "lifetime: pool stats empty");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "lifetime: publisher");
  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "lifetime: capture");

  // The request ends: its sequence and the publish handle both go. The
  // checkpoint is the only holder left, and the shared pages stay.
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  struct ignis_seq_pool_stats retained{};
  expect_rc(ignis_seq_pool_stats(pool, &retained), 0, "lifetime: pool stats retained");
  expect(retained.kv_free_pages == empty.kv_free_pages - 2,
         "lifetime: the two shared pages are still held, and nothing else is");

  // And a claimant long after the request is gone still stands up on them.
  ignis_seq *late = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &late), 0,
            "lifetime: a claim after the request ended");
  expect(late->position == kOpener, "lifetime: at the opener, as always");
  ignis_seq_release(pool, late);

  ignis_seq_checkpoint_release(pool, checkpoint);
  struct ignis_seq_pool_stats released{};
  expect_rc(ignis_seq_pool_stats(pool, &released), 0, "lifetime: pool stats released");
  expect(released.kv_free_pages == empty.kv_free_pages,
         "lifetime: the last holder's release returns every page");
  ignis_seq_pool_free(pool);
}

// ---- 5. a spilled checkpoint is a self-contained host blob ---------------

void check_materialized_blob_outlives_every_device_handle(bool dflash2) {
  ignis_seq_pool_spec spec = small_spec(IGNIS_KV_FORMAT_HQ_E8_2B);
  spec.slot_count = 2;
  if (dflash2) {
    spec.speculative_backend = IGNIS_SPECULATIVE_DFLASH2;
  }
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "materialize: pool create");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "materialize: publisher");
  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "materialize: capture");

  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_checkpoint_snapshot_size(pool, checkpoint, &bytes), 0,
            "materialize: size");
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_checkpoint_snapshot(pool, checkpoint, blob.data(), bytes), 0,
            "materialize: snapshot");

  // The host blob, not a prefix/checkpoint handle, is now the only retained
  // state. Releasing every device owner must therefore return all source
  // pages without making the blob unusable.
  ignis_seq_checkpoint_release(pool, checkpoint);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);

  ignis_seq *restored = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &restored), 0, "materialize: target");
  expect_rc(ignis_seq_restore(pool, restored, blob.data(), bytes), 0,
            "materialize: restore after handles released");
  std::uint64_t restored_bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, restored, &restored_bytes), 0,
            "materialize: restored size");
  std::vector<unsigned char> round_trip(static_cast<std::size_t>(restored_bytes));
  expect_rc(ignis_seq_snapshot(pool, restored, round_trip.data(), restored_bytes), 0,
            "materialize: restored snapshot");
  expect(restored_bytes == bytes && round_trip == blob,
         "materialize: restored state is byte-identical to the spilled blob");

  ignis_seq_release(pool, restored);
  ignis_seq_pool_free(pool);
}

// ---- 5b. a checkpoint on a prefix chain materializes every link ----------
//
// GitHub #187 made the pages under a checkpoint a *chain* -- the block a burst
// shares, then the pages a later turn warmed itself -- and #190's blob has to
// carry all of them in block-table order. The reference is the sequence that
// took the checkpoint, snapshotted where it stands: the two blobs describe the
// same history and have to be the same bytes.

void check_a_chained_checkpoint_blob_is_the_capturing_sequences_own(bool dflash2) {
  ignis_seq_pool_spec spec = small_spec(IGNIS_KV_FORMAT_HQ_E8_2B);
  spec.slot_count          = 3;
  if (dflash2) {
    spec.speculative_backend = IGNIS_SPECULATIVE_DFLASH2;
  }
  ignis_seq_pool *pool = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "chain blob: pool create");

  // Turn 1 publishes the block: two pages.
  ignis_seq *turn1 = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &turn1), 0, "chain blob: alloc turn 1");
  set_frontier(*turn1, kPrefix);
  dirty_state(*pool, *turn1, 2, 0x61u);
  ignis_seq_prefix *block = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, turn1, kPrefix, &block), 0, "chain blob: publish block");

  // Turn 2 claims it, warms a page of its own, chains it over the block, and
  // walks on to an opener inside the page after.
  const std::uint32_t chained = 3 * kPageTokens;
  const std::uint32_t opener  = chained + 40;
  ignis_seq *turn2            = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, block, &turn2), 0, "chain blob: claim block");
  set_frontier(*turn2, chained);
  dirty_state(*pool, *turn2, 1, 0x62u);
  ignis_seq_prefix *link = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, turn2, chained, &link), 0, "chain blob: publish link");
  set_frontier(*turn2, opener);
  turn2->pending_token = 777;
  dirty_state(*pool, *turn2, 1, 0x63u);

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, turn2, opener, &checkpoint), 0,
            "chain blob: capture on the chain");
  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_checkpoint_snapshot_size(pool, checkpoint, &bytes), 0, "chain blob: size");
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_checkpoint_snapshot(pool, checkpoint, blob.data(), bytes), 0,
            "chain blob: snapshot the checkpoint");

  std::uint64_t own_bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, turn2, &own_bytes), 0, "chain blob: sequence size");
  std::vector<unsigned char> own(static_cast<std::size_t>(own_bytes));
  expect_rc(ignis_seq_snapshot(pool, turn2, own.data(), own_bytes), 0,
            "chain blob: snapshot the capturing sequence");
  expect(own_bytes == bytes && own == blob,
         "chain blob: the checkpoint's blob is the capturing sequence's own, link for link");

  // Nothing on the device is needed to bring it back.
  ignis_seq_checkpoint_release(pool, checkpoint);
  ignis_seq_prefix_release(pool, link);
  ignis_seq_prefix_release(pool, block);
  ignis_seq_release(pool, turn2);
  ignis_seq_release(pool, turn1);
  ignis_seq *restored = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &restored), 0, "chain blob: restore target");
  expect_rc(ignis_seq_restore(pool, restored, blob.data(), bytes), 0,
            "chain blob: restore with every link released");
  std::vector<unsigned char> again(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_snapshot(pool, restored, again.data(), bytes), 0,
            "chain blob: re-snapshot the restored sequence");
  expect(again == blob, "chain blob: the round trip is byte-exact");
  ignis_seq_release(pool, restored);
  ignis_seq_pool_free(pool);
}

// ---- 6. every refusal refuses, and costs nothing -------------------------

void check_refusals() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "refuse: pool create");

  ignis_seq_checkpoint *out = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(nullptr, nullptr, kOpener, &out), -1,
            "refuse: null arguments");

  // A sequence with no shared prefix under it: the opener would fall inside a
  // page nothing else owns, but the whole pages below it are nobody's to
  // share, so there is no checkpoint to build.
  ignis_seq *bare = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &bare), 0, "refuse: alloc bare");
  set_frontier(*bare, kOpener);
  expect_rc(ignis_seq_checkpoint_capture(pool, bare, kOpener, &out), -1,
            "refuse: a sequence holding no shared prefix");
  expect(out == nullptr, "refuse: and hands back no handle");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "refuse: publisher");

  // Off the frontier, in both directions.
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener - 1, &out),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: an opener behind the frontier");
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener + 1, &out),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: an opener ahead of the frontier");

  // Mid-chunk: the sections are not consistent with one another.
  const std::uint32_t gqa0 = publisher->gqa_positions[0];
  publisher->gqa_positions[0] = gqa0 - 1;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &out),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: a mid-chunk sequence");
  publisher->gqa_positions[0] = gqa0;

  // An opener whose whole pages are not the ones this sequence shares: the
  // #187 lineage case — a request that resumed from an earlier checkpoint and
  // prefilled past it. Its own first page is not the opener's.
  set_frontier(*publisher, 4 * kPageTokens + 8);
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, 4 * kPageTokens + 8, &out), -1,
            "refuse: an opener outside the sequence's own first page");
  set_frontier(*publisher, kOpener);

  // A capture still works afterwards: every refusal above left the sequence
  // and the pool exactly as they were.
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &out), 0,
            "refuse: a valid capture after every refusal");
  ignis_seq_checkpoint_release(pool, out);

  ignis_seq_release(pool, bare);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);
  ignis_seq_pool_free(pool);
}

} // namespace

int main() {
  int device_count      = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                 "FAIL: no usable CUDA device (%s). ADR 0006: a GPU test fails here, it does "
                 "not skip.\n",
                 cuda_unavailable(err) ? cudaGetErrorString(err) : "device count is zero");
    return 1;
  }
  check_capture_perturbs_nothing();
  // Both KV formats: BF16 is the oracle (ADR 0022), hq-e8-2b is what the
  // owner actually serves, and only hq exercises the four-plane page layout
  // the tail-page copy walks.
  check_claim_reproduces_the_state_at_the_opener(IGNIS_KV_FORMAT_BF16);
  check_claim_reproduces_the_state_at_the_opener(IGNIS_KV_FORMAT_HQ_E8_2B);
  check_penalty_counts_are_zero_at_the_opener();
  check_lifetime();
  check_materialized_blob_outlives_every_device_handle(false);
  check_materialized_blob_outlives_every_device_handle(true);
  check_a_chained_checkpoint_blob_is_the_capturing_sequences_own(false);
  check_a_chained_checkpoint_blob_is_the_capturing_sequences_own(true);
  check_refusals();
  if (failures != 0) {
    std::fprintf(stderr, "%d checkpoint check(s) failed\n", failures);
    return 1;
  }
  std::cout << "ignis_kernel_seq_checkpoint_test: ok\n";
  return 0;
}
