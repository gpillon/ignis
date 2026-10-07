// GitHub #302, #303 (spec flash-next/05): a sequence pool built with
// Flash-Next's indexer and n-gram sections -- OURS.
//
// On the device, the sections are carried by every way a sequence's state
// moves (ADR 0024's "by all or by none"):
// - a slot's indexer tails and n-gram conv columns come back zeroed to its
//   next sequence;
// - a prefix publish into a device retained slot, and its claim, hand the
//   claimant the publisher's tails and conv state at the prefix, and the
//   prefix's pages -- with their block keys -- in place;
// - a checkpoint capture into a host retained slot, and its claim, hand over
//   the state at the opener and a copy of the partial page's block keys;
// - a snapshot (of a sequence standing on a prefix) and a materialized
//   checkpoint blob restore into a fresh sequence with every page's keys,
//   the tails and the conv state, and its blobs carry Flash-Next's own layout
//   version.
// The program writes these sections; here the test writes known bytes into
// them and moves the frontier by hand, as a program's chunk would.
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

std::vector<unsigned char> read(const void *device, std::size_t bytes) {
  std::vector<unsigned char> host(bytes);
  cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost);
  return host;
}

constexpr int32_t kLayers = 12;

// A slot's Flash-Next state, as the pool lays it out.
struct SlotState {
  std::vector<unsigned char> tails, conv;
};

SlotState state_of(const ignis_seq_pool &pool, int32_t slot) {
  SlotState s;
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    const auto tail = read(pool.indexer_slot_tail(layer, slot), pool.indexer_tail_slot_bytes);
    s.tails.insert(s.tails.end(), tail.begin(), tail.end());
  }
  s.conv = read(pool.ngram_slot_conv(slot), pool.ngram_conv_slot_bytes);
  return s;
}

void write_state(const ignis_seq_pool &pool, int32_t slot, unsigned char value) {
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    cudaMemset(pool.indexer_slot_tail(layer, slot), value + layer, pool.indexer_tail_slot_bytes);
  }
  cudaMemset(pool.ngram_slot_conv(slot), value, pool.ngram_conv_slot_bytes);
}

std::vector<unsigned char> keys_of(const ignis_seq_pool &pool, const std::vector<int32_t> &pages) {
  std::vector<unsigned char> out;
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    for (int32_t page : pages) {
      const auto keys = read(pool.indexer_page_keys(layer, page), pool.indexer_page_bytes());
      out.insert(out.end(), keys.begin(), keys.end());
    }
  }
  return out;
}

void write_keys(const ignis_seq_pool &pool, int32_t page, unsigned char value) {
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    cudaMemset(pool.indexer_page_keys(layer, page), value + 3 * layer, pool.indexer_page_bytes());
  }
}

// What a program's completed chunk leaves: every frontier at `position`.
void stand_at(ignis_seq &seq, uint64_t position) {
  seq.position = position;
  seq.pending_token = 5;
  for (auto &f : seq.gqa_positions) f = static_cast<uint32_t>(position);
  for (auto &f : seq.gdn_positions) f = static_cast<uint32_t>(position);
}

std::vector<int32_t> pages_of(const ignis_seq &seq, uint32_t count) {
  std::vector<int32_t> pages = ignis_seq_prefix_chain_page_ids(seq.prefix);
  const auto own = seq.kv.page_ids();
  pages.insert(pages.end(), own.begin(), own.end());
  pages.resize(count);
  return pages;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }

  // Flash-Next's geometry at a small scale: 12 attention layers of 2 KV heads,
  // 2 GDN layers, 16 pages, 3 lanes, a device retained slot and a host one.
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = 2;
  spec.head_dim = 256;
  spec.kv_format = IGNIS_KV_FORMAT_HQ_E8_2B;
  spec.kv_page_group_count = 16;
  spec.max_context_tokens = 256;
  spec.slot_count = 3;
  spec.gdn_num_layers = 2;
  spec.gdn_conv_channels = 10240;
  spec.gdn_value_heads = 48;
  spec.gdn_head_dim = 128;
  spec.vocab = 1024;
  spec.kv_num_layers = kLayers;
  spec.indexer_key_dim = 128;
  spec.indexer_compress_tokens = 4;
  spec.ngram_conv_columns = 9;
  spec.ngram_conv_channels = 10240;
  spec.retained_slot_count = 1;
  spec.retained_host_slot_count = 1;

  struct ignis_seq_pool_plan plan{};
  check(ignis_seq_pool_plan(&spec, &plan) == 0, std::string("plans: ") + ignis_seq_last_error());
  ignis_seq_pool *pool = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0) {
    std::fprintf(stderr, "FATAL: ignis_seq_pool_create: %s\n", ignis_seq_last_error());
    return 1;
  }
  struct ignis_seq_pool_stats stats{};
  check(ignis_seq_pool_stats(pool, &stats) == 0, "stats");
  check(stats.indexer_bytes == plan.indexer_bytes && stats.ngram_conv_bytes == plan.ngram_conv_bytes &&
            stats.retained_host_bytes == plan.retained_host_bytes,
        "the pool holds what its plan named");
  check(ignis_seq_pool_snapshot_format_version(pool) == kIgnisSeqSnapshotFormatVersionFlashNext,
        "a Flash-Next pool writes its own family's layout version");

  // --- a slot's sections are zero for its next sequence ---------------------
  ignis_seq *first = nullptr;
  check(ignis_seq_alloc(pool, 256, &first) == 0, std::string("alloc: ") + ignis_seq_last_error());
  const int32_t reused = first->slot;
  write_state(*pool, reused, 0xA5);
  cudaDeviceSynchronize();
  ignis_seq_release(pool, first);
  ignis_seq *a = nullptr;
  check(ignis_seq_alloc(pool, 256, &a) == 0, std::string("alloc A: ") + ignis_seq_last_error());
  check(a->slot == reused, "the pool hands the freed slot back (LIFO)");
  const SlotState zero = state_of(*pool, a->slot);
  bool zeroed = true;
  for (unsigned char b : zero.tails) zeroed = zeroed && b == 0;
  for (unsigned char b : zero.conv) zeroed = zeroed && b == 0;
  check(zeroed, "the slot's indexer tails and n-gram conv columns are zero for its next sequence");

  // --- A writes its first page and publishes it as a prefix -----------------
  const std::vector<int32_t> a_own = [&] {
    const auto ids = a->kv.page_ids();
    return std::vector<int32_t>(ids.begin(), ids.end());
  }();
  write_keys(*pool, a_own[0], 0x10);
  write_state(*pool, a->slot, 0x20);
  stand_at(*a, 64);
  cudaDeviceSynchronize();
  const SlotState at_prefix = state_of(*pool, a->slot);
  ignis_seq_prefix *prefix = nullptr;
  check(ignis_seq_prefix_publish(pool, a, 64, 0, &prefix) == 0,
        std::string("publish 64 tokens into the device retained slot: ") + ignis_seq_last_error());

  // A goes on to 100 tokens: its second page's keys, new tails and conv.
  const std::vector<int32_t> a_pages = pages_of(*a, 2);
  write_keys(*pool, a_pages[1], 0x30);
  write_state(*pool, a->slot, 0x40);
  stand_at(*a, 100);
  cudaDeviceSynchronize();
  const SlotState at_opener = state_of(*pool, a->slot);
  const std::vector<unsigned char> a_keys = keys_of(*pool, a_pages);
  ignis_seq_checkpoint *checkpoint = nullptr;
  check(ignis_seq_checkpoint_capture(pool, a, 100, 1, &checkpoint) == 0,
        std::string("capture at 100 into the host retained slot: ") + ignis_seq_last_error());

  // --- the prefix's claimant: the state at 64, the prefix's page in place ---
  ignis_seq *c = nullptr;
  check(ignis_seq_alloc_shared(pool, 256, prefix, &c) == 0, std::string("claim the prefix: ") + ignis_seq_last_error());
  if (c != nullptr) {
    const SlotState got = state_of(*pool, c->slot);
    check(got.tails == at_prefix.tails && got.conv == at_prefix.conv,
          "the prefix's claimant holds the publisher's tails and conv state at the prefix");
    check(pages_of(*c, 1)[0] == a_own[0], "the claimant reads the prefix's page in place");
    check(c->position == 64, "the claimant stands at the prefix");
    ignis_seq_release(pool, c);
  }

  // --- the checkpoint's claimant: the state at 100, its own copy of page 1 --
  ignis_seq *d = nullptr;
  check(ignis_seq_alloc_from_checkpoint(pool, 256, checkpoint, &d) == 0,
        std::string("claim the checkpoint: ") + ignis_seq_last_error());
  if (d != nullptr) {
    const SlotState got = state_of(*pool, d->slot);
    check(got.tails == at_opener.tails && got.conv == at_opener.conv,
          "the checkpoint's claimant holds the tails and conv state at the opener (through the host slot)");
    const std::vector<int32_t> d_pages = pages_of(*d, 2);
    check(d_pages[0] == a_pages[0] && d_pages[1] != a_pages[1], "page 0 shared, page 1 the claimant's own");
    check(keys_of(*pool, d_pages) == a_keys, "the claimant's copy of the partial page carries its block keys");
    ignis_seq_release(pool, d);
  }

  // --- a snapshot of A (standing on the prefix) into a fresh sequence --------
  uint64_t bytes = 0;
  check(ignis_seq_snapshot_size(pool, a, &bytes) == 0, std::string("snapshot size: ") + ignis_seq_last_error());
  std::vector<unsigned char> blob(bytes);
  check(ignis_seq_snapshot(pool, a, blob.data(), blob.size()) == 0, std::string("snapshot: ") + ignis_seq_last_error());
  ignis_seq_snapshot_header header{};
  std::memcpy(&header, blob.data(), sizeof(header));
  check(header.format_version == kIgnisSeqSnapshotFormatVersionFlashNext, "the blob names Flash-Next's layout");
  ignis_seq *e = nullptr;
  check(ignis_seq_alloc(pool, 256, &e) == 0, std::string("alloc E: ") + ignis_seq_last_error());
  check(ignis_seq_restore(pool, e, blob.data(), blob.size()) == 0, std::string("restore: ") + ignis_seq_last_error());
  {
    const SlotState got = state_of(*pool, e->slot);
    check(got.tails == at_opener.tails && got.conv == at_opener.conv, "the restored tails and conv state are A's");
    check(keys_of(*pool, pages_of(*e, 2)) == a_keys, "the restored pages carry A's block keys");
    check(e->position == 100, "the restored sequence stands at 100");
  }
  ignis_seq_release(pool, e);

  // --- the checkpoint materialized and restored ------------------------------
  uint64_t ckpt_bytes = 0;
  check(ignis_seq_checkpoint_snapshot_size(pool, checkpoint, &ckpt_bytes) == 0,
        std::string("checkpoint blob size: ") + ignis_seq_last_error());
  std::vector<unsigned char> ckpt_blob(ckpt_bytes);
  check(ignis_seq_checkpoint_snapshot(pool, checkpoint, ckpt_blob.data(), ckpt_blob.size()) == 0,
        std::string("checkpoint blob: ") + ignis_seq_last_error());
  ignis_seq *f = nullptr;
  check(ignis_seq_alloc(pool, 256, &f) == 0, std::string("alloc F: ") + ignis_seq_last_error());
  check(ignis_seq_restore(pool, f, ckpt_blob.data(), ckpt_blob.size()) == 0,
        std::string("restore the checkpoint blob: ") + ignis_seq_last_error());
  {
    const SlotState got = state_of(*pool, f->slot);
    check(got.tails == at_opener.tails && got.conv == at_opener.conv, "the checkpoint blob carries the opener's state");
    check(keys_of(*pool, pages_of(*f, 2)) == a_keys, "and both pages' block keys");
  }
  ignis_seq_release(pool, f);

  ignis_seq_checkpoint_release(pool, checkpoint);

  // --- GitHub #306: a capture standing on nothing hands page 0 over ---------
  // The pages below the opener become a pages-only link; the sequence's new
  // first page carries the opener's partial page, block keys included, and a
  // claimant of the checkpoint reads both pages' keys and the opener's state.
  struct ignis_seq_pool_stats before_g{};
  check(ignis_seq_pool_stats(pool, &before_g) == 0, "stats before G");
  ignis_seq *g = nullptr;
  check(ignis_seq_alloc(pool, 256, &g) == 0, std::string("alloc G: ") + ignis_seq_last_error());
  const std::vector<int32_t> g_pages = pages_of(*g, 2);
  write_keys(*pool, g_pages[0], 0x50);
  write_keys(*pool, g_pages[1], 0x60);
  write_state(*pool, g->slot, 0x70);
  stand_at(*g, 100);
  cudaDeviceSynchronize();
  const SlotState g_state = state_of(*pool, g->slot);
  const std::vector<unsigned char> g_keys = keys_of(*pool, g_pages);
  ignis_seq_checkpoint *handed = nullptr;
  check(ignis_seq_checkpoint_capture(pool, g, 100, 1, &handed) == 0,
        std::string("capture at 100 over a page that is no prefix yet: ") + ignis_seq_last_error());
  if (handed != nullptr) {
    const std::vector<int32_t> g_after = pages_of(*g, 2);
    check(g_after[0] == g_pages[0] && g->shared_pages == 1, "page 0 is a link's now, in place");
    check(keys_of(*pool, g_after) == g_keys, "the sequence's new first page carries the partial page's block keys");
    check(state_of(*pool, g->slot).tails == g_state.tails && state_of(*pool, g->slot).conv == g_state.conv,
          "and its tails and conv state are untouched");
    ignis_seq *h = nullptr;
    check(ignis_seq_alloc_from_checkpoint(pool, 256, handed, &h) == 0,
          std::string("claim the checkpoint over the link: ") + ignis_seq_last_error());
    if (h != nullptr) {
      const SlotState got = state_of(*pool, h->slot);
      check(got.tails == g_state.tails && got.conv == g_state.conv, "the claimant holds the opener's tails and conv");
      check(keys_of(*pool, pages_of(*h, 2)) == g_keys, "and reads both pages' block keys");
      ignis_seq_release(pool, h);
    }
    ignis_seq_checkpoint_release(pool, handed);
  }
  ignis_seq_release(pool, g);
  struct ignis_seq_pool_stats after_g{};
  check(ignis_seq_pool_stats(pool, &after_g) == 0, "stats after G");
  check(after_g.kv_free_pages == before_g.kv_free_pages, "the link went with its last holder, and every page came back");

  ignis_seq_release(pool, a);
  ignis_seq_prefix_release(pool, prefix);
  ignis_seq_pool_free(pool);
  if (g_failed != 0) {
    std::fprintf(stderr, "seq flash-next sections test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("seq flash-next sections test: ok\n");
  return 0;
}
