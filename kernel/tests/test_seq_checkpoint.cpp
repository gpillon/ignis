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
//   4. **nothing is allocated** (GitHub #215): the image goes into the
//      retained slot the caller names and the partial page into one KV page
//      of the pool — none for an opener on a page boundary — and both come
//      back with the checkpoint.
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

// GitHub #281: the second pass of main() builds every pool with its
// retained slots on the host (`ignis_seq_pool_spec::retained_host_slot_count`)
// instead of the device, and runs the same checks.
bool g_host_slots = false;

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
    seq.dflash2_position = seq.position;
  }
  // GitHub #257: an hq pool's residual window -- every layer's side planes
  // and the ring words.
  if (pool.has_hq_residual()) {
    for (const bool role_v : {false, true}) {
      for (std::int32_t layer = 0; layer < kIgnisGqaLayerCount; ++layer) {
        fill_device(pool.hq_residual_plane(role_v, layer, seq.slot),
                    static_cast<std::size_t>(pool.hq_residual_plane_bytes()), ++salt);
      }
    }
    fill_device(pool.hq_ring_words(seq.slot), kIgnisHqRingWords * sizeof(std::uint32_t), ++salt);
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

// A slot's whole mutable state, as one host image -- on an hq pool its
// residual window included (GitHub #257), so a claim that dropped the window
// shows up as a changed image.
std::vector<unsigned char> mutable_image_of(const ignis_seq_pool &pool, std::int32_t slot) {
  const std::size_t conv_bytes   = pool.gdn_pool.conv_host_image_bytes();
  const std::size_t rec_bytes    = pool.gdn_pool.recurrent_host_image_bytes();
  const std::size_t counts_bytes = static_cast<std::size_t>(pool.vocab) * sizeof(std::int32_t);
  const std::size_t window_bytes = static_cast<std::size_t>(pool.hq_residual_slot_bytes());
  std::vector<unsigned char> image(conv_bytes + rec_bytes + counts_bytes + window_bytes);
  pool.gdn_pool.pack_slot_to_host(slot, image.data(), image.data() + conv_bytes, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(image.data() + conv_bytes + rec_bytes, pool.token_counts_for(slot),
                             counts_bytes, cudaMemcpyDeviceToHost, nullptr));
  if (window_bytes != 0) {
    ignis_seq_copy_hq_residual(pool, slot, image.data() + conv_bytes + rec_bytes + counts_bytes,
                               cudaMemcpyDeviceToHost);
  }
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
  spec.kv_num_layers = kIgnisGqaLayerCount;
  spec.gdn_conv_channels   = 6;
  spec.gdn_value_heads     = 2;
  spec.gdn_head_dim        = 4;
  spec.vocab               = 32;
  spec.retained_slot_count = 4;
  if (g_host_slots) {
    spec.retained_host_slot_count = spec.retained_slot_count;
    spec.retained_slot_count      = 0;
  }
  return spec;
}

constexpr std::uint32_t kPageTokens = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
constexpr std::uint32_t kContext    = 384;
// The shared prefix: two whole pages. The generation opener sits 40 tokens
// into the sequence's own third page — the shape a rendered prompt has, where
// the opener is wherever `<|im_start|>assistant\n` happens to end.
constexpr std::uint32_t kPrefix = 2 * kPageTokens;
constexpr std::uint32_t kOpener = kPrefix + 40;
// The retained slots the publisher's prefix and its checkpoint take (GitHub
// #215).
constexpr std::uint32_t kPrefixSlot     = 0;
constexpr std::uint32_t kCheckpointSlot = 1;

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
  expect_rc(ignis_seq_prefix_publish(pool, seq, kPrefix, kPrefixSlot, out_prefix), 0, label);
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
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &checkpoint), 0,
            "capture: capture at the opener");

  expect(row_of(*pool, publisher->slot, 6) == row_before,
         "capture: the capturing sequence's block-table row is untouched");
  expect(mutable_image_of(*pool, publisher->slot) == state_before,
         "capture: and so is every byte of its mutable state");
  expect(publisher->position == kOpener, "capture: and its frontier");
  struct ignis_seq_pool_stats pool_after{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_after), 0, "capture: pool stats after");
  expect(pool_after.kv_free_pages == pool_before.kv_free_pages - 1,
         "capture: a capture costs the KV pool the one page its opener ends inside");
  expect(pool->retained_held[kCheckpointSlot], "capture: and holds its retained slot");

  const struct ignis_seq_checkpoint_stats stats = stats_of(checkpoint, "capture: stats");
  expect(stats.tokens == kOpener, "capture: the checkpoint reaches the opener, not the page");
  expect(stats.pages == 2, "capture: over the two whole pages the prefix holds");
  expect(stats.claim_count == 0, "capture: capturing is not a claim");
  expect(stats.image_bytes == pool->slot_state_bytes + ignis_seq_checkpoint_page_bytes(*pool),
         "capture: a checkpoint holds one slot's state and one page");

  // The prefix now has three holders: the publish handle, the publishing
  // sequence, and the checkpoint. That third one is what keeps the pages
  // alive once the request is gone.
  struct ignis_seq_prefix_stats prefix_stats{};
  expect_rc(ignis_seq_prefix_stats(prefix, &prefix_stats), 0, "capture: prefix stats");
  expect(prefix_stats.refcount == 3, "capture: the checkpoint holds the prefix too");

  ignis_seq_checkpoint_release(pool, checkpoint);
  expect(!pool->retained_held[kCheckpointSlot], "capture: the release gives the slot back");
  struct ignis_seq_pool_stats pool_released{};
  expect_rc(ignis_seq_pool_stats(pool, &pool_released), 0, "capture: pool stats released");
  expect(pool_released.kv_free_pages == pool_before.kv_free_pages,
         "capture: and the page");
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);
  ignis_seq_pool_free(pool);
}

// An opener on a page boundary ends inside no page: the capture copies none
// and costs none, and a claimant stands on the prefix's pages alone.
void check_a_page_aligned_opener_takes_no_page() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "aligned: pool create");

  ignis_seq *publisher = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &publisher), 0, "aligned: alloc");
  set_frontier(*publisher, kPrefix);
  dirty_state(*pool, *publisher, 2, 0x21u);
  publisher->pending_token = 99;
  ignis_seq_prefix *prefix = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, publisher, kPrefix, kPrefixSlot, &prefix), 0,
            "aligned: publish");
  const std::vector<unsigned char> state = mutable_image_of(*pool, publisher->slot);

  struct ignis_seq_pool_stats before{};
  expect_rc(ignis_seq_pool_stats(pool, &before), 0, "aligned: pool stats before");
  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kPrefix, kCheckpointSlot, &checkpoint),
            0, "aligned: capture on the publish point");
  struct ignis_seq_pool_stats after{};
  expect_rc(ignis_seq_pool_stats(pool, &after), 0, "aligned: pool stats after");
  expect(after.kv_free_pages == before.kv_free_pages, "aligned: no page is taken");
  expect(stats_of(checkpoint, "aligned: stats").image_bytes == pool->slot_state_bytes,
         "aligned: the checkpoint holds one slot's state and nothing else");

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &claimant), 0,
            "aligned: claim");
  expect(claimant->position == kPrefix && claimant->pending_token == 99,
         "aligned: the claimant stands at the opener");
  expect(mutable_image_of(*pool, claimant->slot) == state, "aligned: with the capture's state");

  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_checkpoint_snapshot_size(pool, checkpoint, &bytes), 0, "aligned: size");
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_checkpoint_snapshot(pool, checkpoint, blob.data(), bytes), 0,
            "aligned: snapshot");
  std::uint64_t own_bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, claimant, &own_bytes), 0, "aligned: claimant size");
  std::vector<unsigned char> own(static_cast<std::size_t>(own_bytes));
  expect_rc(ignis_seq_snapshot(pool, claimant, own.data(), own_bytes), 0, "aligned: claimant snapshot");
  expect(own == blob, "aligned: the checkpoint's blob is its claimant's, standing on it");

  ignis_seq_release(pool, claimant);
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
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &checkpoint), 0,
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
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &checkpoint), 0,
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
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &checkpoint), 0,
            "lifetime: capture");

  // The request ends: its sequence and the publish handle both go. The
  // checkpoint is the only holder left, and the shared pages stay.
  ignis_seq_release(pool, publisher);
  ignis_seq_prefix_release(pool, prefix);
  struct ignis_seq_pool_stats retained{};
  expect_rc(ignis_seq_pool_stats(pool, &retained), 0, "lifetime: pool stats retained");
  expect(retained.kv_free_pages == empty.kv_free_pages - 3,
         "lifetime: the two shared pages and the checkpoint's own page are still held, and "
         "nothing else is");

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
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &checkpoint), 0,
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
  expect_rc(ignis_seq_prefix_publish(pool, turn1, kPrefix, 0, &block), 0, "chain blob: publish block");

  // Turn 2 claims it, warms a page of its own, chains it over the block, and
  // walks on to an opener inside the page after.
  const std::uint32_t chained = 3 * kPageTokens;
  const std::uint32_t opener  = chained + 40;
  ignis_seq *turn2            = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, block, &turn2), 0, "chain blob: claim block");
  set_frontier(*turn2, chained);
  dirty_state(*pool, *turn2, 1, 0x62u);
  ignis_seq_prefix *link = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, turn2, chained, 1, &link), 0, "chain blob: publish link");
  set_frontier(*turn2, opener);
  turn2->pending_token = 777;
  dirty_state(*pool, *turn2, 1, 0x63u);

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, turn2, opener, 2, &checkpoint), 0,
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

// ---- 6. a capture lends the pages below the opener to a link (GitHub #306) -
//
// ADR 0029 as amended 2026-10-07: a sequence whose whole pages below the
// opener are not yet a prefix -- one standing on nothing, or on a prefix that
// stops short of the opener's page -- is no longer refused. The capture lends
// those pages to a **pages-only link** chained over whatever the sequence
// stands on, and moves nothing: the sequence keeps them in its allocation and
// its block-table row, and goes on exactly as it was. The link has no image,
// so it is never claimed on its own, and no handle: the sequence and the
// checkpoint hold it, and it takes the pages when the sequence is released.

// A sequence standing at `opener` over `shared` pages of an imaged prefix (or
// none), its own pages up to the opener patterned. `*out_prefix` receives the
// publish handle when `shared` is not 0.
ignis_seq *standing_at(ignis_seq_pool *pool, std::uint32_t shared, std::uint32_t opener,
                       ignis_seq_prefix **out_prefix, const char *label) {
  ignis_seq *seq = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &seq), 0, label);
  if (shared != 0) {
    set_frontier(*seq, shared * kPageTokens);
    dirty_state(*pool, *seq, shared, 0x31u);
    expect_rc(ignis_seq_prefix_publish(pool, seq, shared * kPageTokens, kPrefixSlot, out_prefix), 0, label);
  }
  set_frontier(*seq, opener);
  seq->pending_token = 4242;
  dirty_state(*pool, *seq, (opener + kPageTokens - 1) / kPageTokens - shared, 0x41u);
  return seq;
}

void check_a_capture_lends_the_pages_below_to_a_link(int32_t kv_format, std::uint32_t shared,
                                                     std::uint32_t opener) {
  const ignis_seq_pool_spec spec = small_spec(kv_format);
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "link: pool create");
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "link: pool stats empty");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *seq           = standing_at(pool, shared, opener, &prefix, "link: the capturing sequence");
  const std::uint32_t below   = opener / kPageTokens;
  const bool partial          = opener % kPageTokens != 0;
  const std::uint32_t covered = below + (partial ? 1 : 0);
  const std::vector<std::int32_t> row_before = row_of(*pool, seq->slot, 6);
  std::vector<std::vector<unsigned char>> history;
  for (std::uint32_t page = 0; page < covered; ++page) {
    history.push_back(page_image_of(*pool, row_before[page]));
  }
  const std::vector<unsigned char> state = mutable_image_of(*pool, seq->slot);
  struct ignis_seq_pool_stats before{};
  expect_rc(ignis_seq_pool_stats(pool, &before), 0, "link: pool stats before");

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, seq, opener, kCheckpointSlot, &checkpoint), 0,
            "link: a capture over pages that are not yet a prefix");
  if (checkpoint == nullptr) {
    ignis_seq_release(pool, seq);
    ignis_seq_prefix_release(pool, prefix);
    ignis_seq_pool_free(pool);
    return;
  }

  ignis_seq_prefix *link = seq->lent_to;
  expect(link != nullptr && link->lender == seq && link->lent.size() == below - shared,
         "link: the sequence lent the pages it warmed past what it shared");
  expect(link != nullptr && link->retained_slot < 0 && link->parent == prefix && !link->kv.valid(),
         "link: to a pages-only link over what it stands on, which owns nothing yet");
  expect(link != nullptr && link->refcount == 2,
         "link: held by its lender and the checkpoint -- there is no handle");
  expect(seq->prefix == prefix && seq->shared_pages == shared,
         "link: the sequence still stands where it stood");
  expect(row_of(*pool, seq->slot, 6) == row_before,
         "link: and its block-table row is untouched -- the capture moved no page");
  expect(mutable_image_of(*pool, seq->slot) == state && seq->position == opener &&
             seq->pending_token == 4242,
         "link: nor any byte of its state");
  struct ignis_seq_pool_stats after{};
  expect_rc(ignis_seq_pool_stats(pool, &after), 0, "link: pool stats after");
  expect(after.kv_free_pages == before.kv_free_pages - (partial ? 1 : 0),
         "link: the checkpoint's tail page is all the capture takes");
  const struct ignis_seq_checkpoint_stats stats = stats_of(checkpoint, "link: stats");
  expect(stats.pages == below && stats.tokens == opener, "link: the checkpoint stands on the whole chain");

  ignis_seq *refused = nullptr;
  expect_rc(ignis_seq_alloc_shared(pool, kContext, link, &refused), -1,
            "link: a link is never claimed on its own -- there is no state at its end");

  // A second capture at the same opener stands on the loan; a capture at a
  // later opener would lend twice, and is refused.
  ignis_seq_checkpoint *again = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, seq, opener, kCheckpointSlot + 1, &again), 0,
            "link: a second capture at the same opener");
  expect(again != nullptr && link->refcount == 3 && seq->lent_to == link,
         "link: stands on the same link, lending nothing more");
  ignis_seq_checkpoint_release(pool, again);

  // The sequence goes on, writing the page its opener ends inside; a claimant
  // of the checkpoint still stands exactly where the capture stood.
  set_frontier(*seq, opener + 17);
  dirty_state(*pool, *seq, 0, 0x99u);
  if (partial) {
    for (std::size_t plane_index = 0; plane_index < pool->kv_pool.plane_count(); ++plane_index) {
      const ninfer::Tensor &plane = pool->kv_pool.plane(plane_index);
      fill_device(static_cast<unsigned char *>(plane.data) +
                      static_cast<std::int64_t>(row_before[below]) * plane.nb[3],
                  static_cast<std::size_t>(plane.nb[3]), 0x9Au + static_cast<std::uint32_t>(plane_index));
    }
  }
  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &claimant), 0, "link: claim");
  if (claimant != nullptr) {
    const std::vector<std::int32_t> claimant_row = row_of(*pool, claimant->slot, 6);
    expect(std::equal(row_before.begin(), row_before.begin() + below, claimant_row.begin()),
           "link: the claimant addresses the chain's physical pages, lent ones included");
    for (std::uint32_t page = 0; page < covered; ++page) {
      expect(page_image_of(*pool, claimant_row[page]) == history[page],
             "link: and reads the capture's history, the partial page included");
    }
    expect(mutable_image_of(*pool, claimant->slot) == state && claimant->position == opener,
           "link: with the state at the opener");
    ignis_seq_release(pool, claimant);
  }

  // The request ends: the link takes the lent pages, where they are, and the
  // checkpoint alone holds the chain; its release returns every page.
  ignis_seq_release(pool, seq);
  ignis_seq_prefix_release(pool, prefix);
  expect(link->lender == nullptr && link->lent.empty() && link->kv.valid() &&
             std::equal(row_before.begin() + shared, row_before.begin() + below,
                        link->kv.page_ids().begin()) &&
             link->kv.page_ids().size() == below - shared,
         "link: the lender's release hands the link exactly the pages it lent");
  struct ignis_seq_pool_stats retained{};
  expect_rc(ignis_seq_pool_stats(pool, &retained), 0, "link: pool stats retained");
  expect(retained.kv_free_pages == empty.kv_free_pages - below - (partial ? 1 : 0),
         "link: the checkpoint holds the chain's pages and its tail page, nothing else");
  ignis_seq_checkpoint_release(pool, checkpoint);
  struct ignis_seq_pool_stats released{};
  expect_rc(ignis_seq_pool_stats(pool, &released), 0, "link: pool stats released");
  expect(released.kv_free_pages == empty.kv_free_pages, "link: the last holder returns every page");
  ignis_seq_pool_free(pool);
}

// ---- 7. a capture that fails changes nothing (GitHub #306) ---------------
//
// A fault injected at the capture's commit point -- after every fallible step
// has run, before anything is moved -- must leave the sequence, the prefix
// under it and the pool as they were, whatever the capture was about to do:
// lend pages, stand on a prefix, or stand on an earlier loan. One point
// covers every failure the capture can have, because everything before it
// only reads the sequence and nothing after it can fail.

struct Observed {
  std::vector<std::int32_t> row;
  std::vector<unsigned char> state;
  std::uint64_t position;
  ignis_seq_prefix *prefix;
  // Its holders as well as its identity: a capture that took a reference on
  // what the sequence stands on and then failed would leave it held forever.
  std::uint32_t prefix_refcount;
  std::uint32_t shared_pages;
  ignis_seq_prefix *lent_to;
  std::uint32_t lent_refcount;
  std::uint32_t free_pages;
  bool slot_held;
};

Observed observe(ignis_seq_pool *pool, const ignis_seq *seq, std::uint32_t retained_slot) {
  struct ignis_seq_pool_stats stats{};
  expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "fault: pool stats");
  return Observed{row_of(*pool, seq->slot, 6),
                  mutable_image_of(*pool, seq->slot),
                  seq->position,
                  seq->prefix,
                  seq->prefix == nullptr ? 0 : seq->prefix->refcount,
                  seq->shared_pages,
                  seq->lent_to,
                  seq->lent_to == nullptr ? 0 : seq->lent_to->refcount,
                  stats.kv_free_pages,
                  pool->retained_held[retained_slot]};
}

bool same(const Observed &a, const Observed &b) {
  return a.row == b.row && a.state == b.state && a.position == b.position && a.prefix == b.prefix &&
         a.prefix_refcount == b.prefix_refcount && a.shared_pages == b.shared_pages &&
         a.lent_to == b.lent_to && a.lent_refcount == b.lent_refcount &&
         a.free_pages == b.free_pages && a.slot_held == b.slot_held;
}

void expect_a_failed_capture_changes_nothing(ignis_seq_pool *pool, ignis_seq *seq, std::uint32_t opener,
                                             std::uint32_t retained_slot, const char *label) {
  const Observed before = observe(pool, seq, retained_slot);
  ignis_seq_inject_capture_fault();
  ignis_seq_checkpoint *out = nullptr;
  const int32_t rc          = ignis_seq_checkpoint_capture(pool, seq, opener, retained_slot, &out);
  expect(rc == -1 && out == nullptr, label);
  expect(same(observe(pool, seq, retained_slot), before), label);
  // And the very same capture succeeds once nothing fails.
  expect_rc(ignis_seq_checkpoint_capture(pool, seq, opener, retained_slot, &out), 0, label);
  ignis_seq_checkpoint_release(pool, out);
}

void check_a_failed_capture_changes_nothing() {
  const ignis_seq_pool_spec spec = small_spec(IGNIS_KV_FORMAT_HQ_E8_2B);
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "fault: pool create");
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "fault: pool stats empty");

  // About to lend: a sequence standing on nothing.
  ignis_seq_prefix *none = nullptr;
  ignis_seq *lender      = standing_at(pool, 0, kOpener, &none, "fault: a lender");
  expect_a_failed_capture_changes_nothing(pool, lender, kOpener, kCheckpointSlot,
                                          "fault: a failed capture lends nothing and changes nothing");
  // About to stand on that loan: the capture above did lend, at last.
  expect(lender->lent_to != nullptr, "fault: the retried capture lent its pages");
  expect_a_failed_capture_changes_nothing(pool, lender, kOpener, kCheckpointSlot,
                                          "fault: a failed capture on a loan changes nothing");
  ignis_seq_release(pool, lender);

  // About to lend over a prefix, which the link would take a reference on.
  ignis_seq_prefix *under = nullptr;
  ignis_seq *over         = standing_at(pool, 2, 4 * kPageTokens + 8, &under, "fault: a lender over a prefix");
  expect_a_failed_capture_changes_nothing(pool, over, 4 * kPageTokens + 8, kCheckpointSlot,
                                          "fault: a failed capture over a prefix takes no reference on it");
  ignis_seq_prefix_release(pool, under);
  ignis_seq_release(pool, over);

  // About to stand on a prefix: the #186 shape, nothing to lend.
  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "fault: a publisher");
  expect_a_failed_capture_changes_nothing(pool, publisher, kOpener, kCheckpointSlot,
                                          "fault: a failed capture on a prefix changes nothing");
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);

  struct ignis_seq_pool_stats end{};
  expect_rc(ignis_seq_pool_stats(pool, &end), 0, "fault: pool stats end");
  expect(end.kv_free_pages == empty.kv_free_pages, "fault: and every page came back");
  ignis_seq_pool_free(pool);
}

// ---- 7b. a lender: what it refuses, and what outlives it (GitHub #306) -----
//
// Until it is released, a lender's first own pages are a link's. Nothing may
// give them a second owner -- a publish, a restore over them -- and the link
// is no handle a caller could release, snapshot or read. And when the lender
// goes, a claimant standing on the link keeps addressing the very pages it
// lent, whether the lender was released outright or evicted to a blob first.

bool last_error_names(const char *what) {
  return std::string(ignis_seq_last_error()).find(what) != std::string::npos;
}

void check_a_lender_publishes_and_restores_nothing() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "lender: pool create");

  // An opener on a page boundary: the lender stands exactly where a publish
  // of the pages it lent would be cut, so nothing but the loan refuses one.
  const std::uint32_t opener = 3 * kPageTokens;
  ignis_seq_prefix *none     = nullptr;
  ignis_seq *lender          = standing_at(pool, 0, opener, &none, "lender: the capturing sequence");
  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, lender, opener, kCheckpointSlot, &checkpoint), 0,
            "lender: the capture lends");
  expect(lender->lent_to != nullptr, "lender: and the sequence is a lender");
  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, lender, &bytes), 0, "lender: blob size");
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_snapshot(pool, lender, blob.data(), bytes), 0, "lender: a blob to restore");

  const Observed before   = observe(pool, lender, kPrefixSlot);
  ignis_seq_prefix *again = nullptr;
  expect_rc(ignis_seq_prefix_publish(pool, lender, opener, kPrefixSlot, &again), -1,
            "lender: a publish of the pages it lent is refused");
  expect(again == nullptr && last_error_names("lent"), "lender: naming the loan");
  expect(same(observe(pool, lender, kPrefixSlot), before),
         "lender: and changes nothing -- the publish's retained slot is still free");
  expect_rc(ignis_seq_restore(pool, lender, blob.data(), bytes), IGNIS_SEQ_ERR_SHARED_PREFIX,
            "lender: a restore over the pages it lent is refused");
  expect(last_error_names("lent"), "lender: naming the loan too");
  expect(same(observe(pool, lender, kPrefixSlot), before), "lender: and changes nothing either");

  ignis_seq_release(pool, lender);
  ignis_seq_checkpoint_release(pool, checkpoint);
  ignis_seq_pool_free(pool);
}

void check_a_link_on_loan_has_no_handle() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "no handle: pool create");
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(pool, &empty), 0, "no handle: pool stats empty");

  ignis_seq_prefix *none = nullptr;
  ignis_seq *lender      = standing_at(pool, 0, kOpener, &none, "no handle: a lender");
  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, lender, kOpener, kCheckpointSlot, &checkpoint), 0,
            "no handle: the capture lends");
  ignis_seq_prefix *link = lender->lent_to;
  if (link == nullptr) {
    expect(false, "no handle: a link to try");
    ignis_seq_release(pool, lender);
    ignis_seq_checkpoint_release(pool, checkpoint);
    ignis_seq_pool_free(pool);
    return;
  }
  struct ignis_seq_pool_stats before{};
  expect_rc(ignis_seq_pool_stats(pool, &before), 0, "no handle: pool stats before");

  ignis_seq_prefix_release(pool, link);
  expect(link->refcount == 2 && last_error_names("pages-only link"),
         "no handle: a release of the link is refused, naming it");
  ignis_seq_prefix_release(nullptr, link);
  expect(link->refcount == 2 && last_error_names("pages-only link"),
         "no handle: with no pool to check it against, too");
  std::uint64_t bytes = 1;
  expect_rc(ignis_seq_prefix_snapshot_size(pool, link, &bytes), -1, "no handle: no blob size");
  expect(bytes == 0 && last_error_names("pages-only link"), "no handle: naming the link");
  std::vector<unsigned char> blob(64);
  expect_rc(ignis_seq_prefix_snapshot(pool, link, blob.data(), blob.size()), -1, "no handle: no blob");
  expect(last_error_names("pages-only link"), "no handle: naming the link");
  struct ignis_seq_prefix_stats stats{};
  expect_rc(ignis_seq_prefix_stats(link, &stats), -1, "no handle: no stats");
  expect(last_error_names("pages-only link"), "no handle: naming the link");
  struct ignis_seq_pool_stats after{};
  expect_rc(ignis_seq_pool_stats(pool, &after), 0, "no handle: pool stats after");
  expect(after.kv_free_pages == before.kv_free_pages && link->lender == lender,
         "no handle: and nothing moved");

  ignis_seq_release(pool, lender);
  ignis_seq_checkpoint_release(pool, checkpoint);
  struct ignis_seq_pool_stats end{};
  expect_rc(ignis_seq_pool_stats(pool, &end), 0, "no handle: pool stats end");
  expect(end.kv_free_pages == empty.kv_free_pages, "no handle: every page came back");
  ignis_seq_pool_free(pool);
}

// A claimant of the checkpoint over the lender's link, standing at the opener
// with the pages it addresses and their history. `shared` pages under the
// lender are an imaged prefix's (or none), released with the lender.
struct LoanFixture {
  ignis_seq_pool *pool;
  ignis_seq_prefix *prefix;
  ignis_seq *lender;
  ignis_seq_checkpoint *checkpoint;
  ignis_seq *claimant;
  ignis_seq_prefix *link;
  std::uint32_t free_pages;
  std::uint32_t covered;
  // The lender's row and the history on it, the opener's partial page
  // included, and its state: what the claimant reads.
  std::vector<std::int32_t> lender_row;
  std::vector<std::vector<unsigned char>> history;
  std::vector<unsigned char> state;
  std::vector<std::int32_t> claimant_row;
};

LoanFixture a_claimant_on_a_loan(std::uint32_t shared, std::uint32_t opener, const char *label) {
  const ignis_seq_pool_spec spec = small_spec(IGNIS_KV_FORMAT_HQ_E8_2B);
  LoanFixture f{};
  expect_rc(ignis_seq_pool_create(&spec, &f.pool), 0, label);
  struct ignis_seq_pool_stats empty{};
  expect_rc(ignis_seq_pool_stats(f.pool, &empty), 0, label);
  f.free_pages = empty.kv_free_pages;
  f.lender  = standing_at(f.pool, shared, opener, &f.prefix, label);
  f.covered = (opener + kPageTokens - 1) / kPageTokens;
  f.lender_row = row_of(*f.pool, f.lender->slot, f.covered);
  for (std::uint32_t page = 0; page < f.covered; ++page) {
    f.history.push_back(page_image_of(*f.pool, f.lender_row[page]));
  }
  f.state = mutable_image_of(*f.pool, f.lender->slot);
  expect_rc(ignis_seq_checkpoint_capture(f.pool, f.lender, opener, kCheckpointSlot, &f.checkpoint), 0,
            label);
  f.link = f.lender->lent_to;
  expect(f.link != nullptr, label);
  expect_rc(ignis_seq_alloc_from_checkpoint(f.pool, kContext, f.checkpoint, &f.claimant), 0, label);
  if (f.claimant != nullptr) {
    f.claimant_row = row_of(*f.pool, f.claimant->slot, f.covered);
  }
  return f;
}

// The claimant still addresses the chain's pages, reads the lender's history
// and its state at the opener, and no page it addresses is `other`'s.
void expect_the_claimant_untouched(const LoanFixture &f, const ignis_seq *other, const char *label) {
  expect(row_of(*f.pool, f.claimant->slot, f.covered) == f.claimant_row, label);
  for (std::uint32_t page = 0; page < f.covered; ++page) {
    expect(page_image_of(*f.pool, f.claimant_row[page]) == f.history[page], label);
  }
  expect(mutable_image_of(*f.pool, f.claimant->slot) == f.state, label);
  const std::vector<std::int32_t> theirs = row_of(*f.pool, other->slot, 6);
  for (const std::int32_t page : f.claimant_row) {
    expect(std::find(theirs.begin(), theirs.end(), page) == theirs.end(), label);
  }
}

void release_the_loan_fixture(LoanFixture &f, ignis_seq *other, const char *label) {
  ignis_seq_release(f.pool, other);
  ignis_seq_release(f.pool, f.claimant);
  ignis_seq_checkpoint_release(f.pool, f.checkpoint);
  struct ignis_seq_pool_stats end{};
  expect_rc(ignis_seq_pool_stats(f.pool, &end), 0, label);
  expect(end.kv_free_pages == f.free_pages, label);
  ignis_seq_pool_free(f.pool);
}

void check_a_claimant_outlives_its_lender(std::uint32_t shared, std::uint32_t opener) {
  LoanFixture f = a_claimant_on_a_loan(shared, opener, "outlive: a claimant on the loan");
  if (f.link == nullptr || f.claimant == nullptr) {
    ignis_seq_pool_free(f.pool);
    return;
  }
  const std::uint32_t below = opener / kPageTokens;

  // The lender goes, and the handle on what it stood on, while the claimant
  // stands on the link: the link takes the very pages it was lent.
  ignis_seq_release(f.pool, f.lender);
  ignis_seq_prefix_release(f.pool, f.prefix);
  expect(f.link->lender == nullptr && f.link->kv.valid() && f.link->refcount == 2 &&
             f.link->kv.page_ids().size() == below - shared &&
             std::equal(f.lender_row.begin() + shared, f.lender_row.begin() + below,
                        f.link->kv.page_ids().begin()),
         "outlive: the link holds what was lent, for the checkpoint and the claimant");

  // A fresh sequence takes whatever the lender gave back and writes all of it.
  ignis_seq *fresh = nullptr;
  expect_rc(ignis_seq_alloc(f.pool, kContext, &fresh), 0, "outlive: a fresh sequence");
  if (fresh != nullptr) {
    dirty_state(*f.pool, *fresh, 6, 0xA1u);
    expect_the_claimant_untouched(f, fresh, "outlive: the claimant's row, history and state are its own");
  }
  release_the_loan_fixture(f, fresh, "outlive: every page came back");
}

void check_a_lender_evicted_while_lent_comes_back_whole(std::uint32_t shared, std::uint32_t opener) {
  LoanFixture f = a_claimant_on_a_loan(shared, opener, "evict: a claimant on the loan");
  if (f.link == nullptr || f.claimant == nullptr) {
    ignis_seq_pool_free(f.pool);
    return;
  }

  // Evicted while lent: its blob carries the lent pages as history of its
  // own, as the checkpoint's blob does -- the same bytes.
  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_snapshot_size(f.pool, f.lender, &bytes), 0, "evict: lender blob size");
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_snapshot(f.pool, f.lender, blob.data(), bytes), 0, "evict: lender blob");
  std::uint64_t checkpoint_bytes = 0;
  expect_rc(ignis_seq_checkpoint_snapshot_size(f.pool, f.checkpoint, &checkpoint_bytes), 0,
            "evict: checkpoint blob size");
  std::vector<unsigned char> reference(static_cast<std::size_t>(checkpoint_bytes));
  expect_rc(ignis_seq_checkpoint_snapshot(f.pool, f.checkpoint, reference.data(), checkpoint_bytes), 0,
            "evict: checkpoint blob");
  expect(bytes == checkpoint_bytes && blob == reference,
         "evict: the lender's blob is its whole history, lent pages included");

  ignis_seq_release(f.pool, f.lender);
  ignis_seq_prefix_release(f.pool, f.prefix);

  // Restored into a fresh sequence while the claimant stands on the link:
  // pages of its own, the same history, the same state.
  ignis_seq *restored = nullptr;
  expect_rc(ignis_seq_alloc(f.pool, kContext, &restored), 0, "evict: restore target");
  if (restored != nullptr) {
    expect_rc(ignis_seq_restore(f.pool, restored, blob.data(), bytes), 0, "evict: restore");
    const std::vector<std::int32_t> row = row_of(*f.pool, restored->slot, f.covered);
    for (std::uint32_t page = 0; page < f.covered; ++page) {
      expect(page_image_of(*f.pool, row[page]) == f.history[page],
             "evict: the restored sequence reads the lender's history");
    }
    expect(mutable_image_of(*f.pool, restored->slot) == f.state && restored->position == opener &&
               restored->prefix == nullptr && restored->lent_to == nullptr,
           "evict: from its own pages, with the lender's state, lending nothing");
    std::vector<unsigned char> again(static_cast<std::size_t>(bytes));
    expect_rc(ignis_seq_snapshot(f.pool, restored, again.data(), bytes), 0, "evict: re-snapshot");
    expect(again == blob, "evict: the round trip is byte-exact");
    // And writing past the opener, as it goes on, touches nothing the
    // claimant reads.
    dirty_state(*f.pool, *restored, 6, 0xB1u);
    expect_the_claimant_untouched(f, restored, "evict: the claimant's row, history and state are its own");
  }
  release_the_loan_fixture(f, restored, "evict: every page came back");
}

// ---- 8. every refusal refuses, and costs nothing -------------------------

void check_refusals() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "refuse: pool create");

  ignis_seq_checkpoint *out = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(nullptr, nullptr, kOpener, kCheckpointSlot, &out), -1,
            "refuse: null arguments");

  // A sequence standing on nothing, its opener inside its first page: there
  // is no whole page below the opener for a checkpoint to hold (GitHub #306
  // hands the pages below over, and there are none).
  ignis_seq *bare = nullptr;
  expect_rc(ignis_seq_alloc(pool, kContext, &bare), 0, "refuse: alloc bare");
  set_frontier(*bare, 40);
  expect_rc(ignis_seq_checkpoint_capture(pool, bare, 40, kCheckpointSlot, &out), -1,
            "refuse: an opener in the first page of a sequence holding no shared prefix");
  expect(out == nullptr, "refuse: and hands back no handle");

  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "refuse: publisher");

  // Off the frontier, in both directions.
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener - 1, kCheckpointSlot, &out),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: an opener behind the frontier");
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener + 1, kCheckpointSlot, &out),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: an opener ahead of the frontier");

  // Mid-chunk: the sections are not consistent with one another.
  const std::uint32_t gqa0 = publisher->gqa_positions[0];
  publisher->gqa_positions[0] = gqa0 - 1;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &out),
            IGNIS_SEQ_ERR_NOT_AT_BOUNDARY, "refuse: a mid-chunk sequence");
  publisher->gqa_positions[0] = gqa0;

  // An opener inside the pages the sequence shares: its partial page is one
  // other holders own, so there is no page of its own to copy (the case of a
  // claim reaching past its opener's page).
  set_frontier(*publisher, kPrefix - 8);
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kPrefix - 8, kCheckpointSlot, &out), -1,
            "refuse: an opener inside the shared pages");
  set_frontier(*publisher, kOpener);

  // GitHub #215: a slot out of range, or one the prefix under it holds.
  struct ignis_seq_pool_stats before_slots{};
  expect_rc(ignis_seq_pool_stats(pool, &before_slots), 0, "refuse: pool stats before slots");
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener,
                                         spec.retained_slot_count + spec.retained_host_slot_count, &out),
            -1, "refuse: a retained slot past the pool's");
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kPrefixSlot, &out), -1,
            "refuse: the retained slot the prefix's image is in");
  expect(out == nullptr, "refuse: and hands back no handle");
  struct ignis_seq_pool_stats after_slots{};
  expect_rc(ignis_seq_pool_stats(pool, &after_slots), 0, "refuse: pool stats after slots");
  expect(after_slots.kv_free_pages == before_slots.kv_free_pages,
         "refuse: a refused capture takes no page");

  // A capture still works afterwards: every refusal above left the sequence
  // and the pool exactly as they were.
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, kCheckpointSlot, &out), 0,
            "refuse: a valid capture after every refusal");
  ignis_seq_checkpoint_release(pool, out);

  ignis_seq_release(pool, bare);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_release(pool, publisher);
  ignis_seq_pool_free(pool);
}

void run_all() {
  check_capture_perturbs_nothing();
  check_a_page_aligned_opener_takes_no_page();
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
  // GitHub #306, in both formats: a sequence standing on nothing, one on a
  // prefix two pages short of its opener's page (#187's lineage case), and
  // an opener on a page boundary, which has no partial page to copy.
  for (const int32_t format : {IGNIS_KV_FORMAT_BF16, IGNIS_KV_FORMAT_HQ_E8_2B}) {
    check_a_capture_lends_the_pages_below_to_a_link(format, 0, kOpener);
    check_a_capture_lends_the_pages_below_to_a_link(format, 2, 4 * kPageTokens + 8);
    check_a_capture_lends_the_pages_below_to_a_link(format, 0, 3 * kPageTokens);
  }
  check_a_failed_capture_changes_nothing();
  check_a_lender_publishes_and_restores_nothing();
  check_a_link_on_loan_has_no_handle();
  // A claimant standing on the link across its lender's release, and across
  // its eviction and restore: a lender standing on nothing, one on a prefix,
  // and one at a page-aligned opener, which copies no partial page.
  check_a_claimant_outlives_its_lender(0, kOpener);
  check_a_claimant_outlives_its_lender(2, 4 * kPageTokens + 8);
  check_a_claimant_outlives_its_lender(0, 3 * kPageTokens);
  check_a_lender_evicted_while_lent_comes_back_whole(0, kOpener);
  check_a_lender_evicted_while_lent_comes_back_whole(2, 4 * kPageTokens + 8);
  check_refusals();
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
  run_all();
  // GitHub #281: every check again with the retained slots on the host --
  // a checkpoint captured into one, claimed from one and spilled from one.
  g_host_slots = true;
  run_all();
  if (failures != 0) {
    std::fprintf(stderr, "%d checkpoint check(s) failed\n", failures);
    return 1;
  }
  std::cout << "ignis_kernel_seq_checkpoint_test: ok\n";
  return 0;
}
