// GitHub #302 (spec flash-next/04): a sequence pool built with Flash-Next's
// indexer and n-gram sections -- OURS.
//
// On the device: the pool holds the sections its plan named, byte for byte;
// a slot's indexer tails and n-gram conv columns come back zeroed to its next
// sequence; and every entry point that snapshots, restores or clones a
// sequence refuses by name, because the section table those carry does not
// list the new sections yet (spec flash-next/05) -- a clone without them
// would hand a claimant the wrong state.
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
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

bool all_zero(const void *device, std::size_t bytes) {
  std::vector<unsigned char> host(bytes);
  if (cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }
  for (unsigned char b : host) {
    if (b != 0) return false;
  }
  return true;
}

bool refused(int32_t rc, const std::string &what) {
  const std::string message = ignis_seq_last_error();
  const bool named = message.find("Flash-Next indexer and n-gram sections") != std::string::npos;
  check(rc != 0 && named, what + " is refused by name: rc " + std::to_string(rc) + ", \"" + message + "\"");
  return rc != 0;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }

  // Flash-Next's geometry at a small scale: 12 attention layers of 2 KV heads,
  // 2 GDN layers, 8 pages, 3 lanes.
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = 2;
  spec.head_dim = 256;
  spec.kv_format = IGNIS_KV_FORMAT_HQ_E8_2B;
  spec.kv_page_group_count = 8;
  spec.max_context_tokens = 128;
  spec.slot_count = 3;
  spec.gdn_num_layers = 2;
  spec.gdn_conv_channels = 10240;
  spec.gdn_value_heads = 48;
  spec.gdn_head_dim = 128;
  spec.vocab = 1024;
  spec.kv_num_layers = 12;
  spec.indexer_key_dim = 128;
  spec.indexer_compress_tokens = 4;
  spec.ngram_conv_columns = 9;
  spec.ngram_conv_channels = 10240;

  struct ignis_seq_pool_plan plan{};
  check(ignis_seq_pool_plan(&spec, &plan) == 0, std::string("plans: ") + ignis_seq_last_error());
  ignis_seq_pool *pool = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0) {
    std::fprintf(stderr, "FATAL: ignis_seq_pool_create: %s\n", ignis_seq_last_error());
    return 1;
  }
  struct ignis_seq_pool_stats stats{};
  check(ignis_seq_pool_stats(pool, &stats) == 0, "stats");
  check(stats.indexer_bytes == plan.indexer_bytes && stats.indexer_bytes != 0,
        "the indexer section is the planned " + std::to_string(plan.indexer_bytes) + " bytes: got " +
            std::to_string(stats.indexer_bytes));
  check(stats.ngram_conv_bytes == plan.ngram_conv_bytes && stats.ngram_conv_bytes != 0,
        "the n-gram conv state is the planned " + std::to_string(plan.ngram_conv_bytes) + " bytes: got " +
            std::to_string(stats.ngram_conv_bytes));
  check(pool->has_indexer() && pool->has_ngram_conv(), "the pool names both sections");

  // A slot's tails and conv columns, dirtied by one sequence, are zero for
  // the next one drawn into the same slot.
  ignis_seq *seq = nullptr;
  check(ignis_seq_alloc(pool, 128, &seq) == 0, std::string("alloc: ") + ignis_seq_last_error());
  const int32_t slot = seq->slot;
  auto tail_of = [&](int32_t layer) {
    return static_cast<unsigned char *>(pool->indexer_tail_keys(layer)) +
           static_cast<std::uint64_t>(slot) * pool->indexer_tail_slot_bytes;
  };
  auto conv_of = [&]() {
    return static_cast<unsigned char *>(pool->ngram_conv->p) +
           static_cast<std::uint64_t>(slot) * pool->ngram_conv_slot_bytes;
  };
  for (int32_t layer = 0; layer < 12; ++layer) {
    cudaMemset(tail_of(layer), 0xA5, pool->indexer_tail_slot_bytes);
  }
  cudaMemset(conv_of(), 0x5A, pool->ngram_conv_slot_bytes);
  cudaDeviceSynchronize();

  // Every way a sequence's state leaves its slot or arrives in another one.
  uint64_t bytes = 0;
  refused(ignis_seq_snapshot_size(pool, seq, &bytes), "a snapshot's size");
  std::vector<unsigned char> blob(1 << 20);
  refused(ignis_seq_snapshot(pool, seq, blob.data(), blob.size()), "a snapshot");
  refused(ignis_seq_restore(pool, seq, blob.data(), blob.size()), "a restore");
  ignis_seq_prefix *prefix = nullptr;
  refused(ignis_seq_prefix_publish(pool, seq, 64, 0, &prefix), "a prefix publish");
  ignis_seq_checkpoint *checkpoint = nullptr;
  refused(ignis_seq_checkpoint_capture(pool, seq, 32, 0, &checkpoint), "a checkpoint capture");

  ignis_seq_release(pool, seq);
  ignis_seq *next = nullptr;
  check(ignis_seq_alloc(pool, 128, &next) == 0, std::string("re-alloc: ") + ignis_seq_last_error());
  check(next->slot == slot, "the pool hands the freed slot back (LIFO)");
  bool tails_zero = true;
  for (int32_t layer = 0; layer < 12; ++layer) {
    tails_zero = tails_zero && all_zero(tail_of(layer), pool->indexer_tail_slot_bytes);
  }
  check(tails_zero, "the slot's indexer tails are zero for its next sequence");
  check(all_zero(conv_of(), pool->ngram_conv_slot_bytes), "its n-gram conv columns are zero");

  ignis_seq_release(pool, next);
  ignis_seq_pool_free(pool);
  if (g_failed != 0) {
    std::fprintf(stderr, "seq flash-next sections test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("seq flash-next sections test: ok\n");
  return 0;
}
