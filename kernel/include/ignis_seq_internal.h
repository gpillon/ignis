/* ignis kernel leaf: `ignis_seq_pool` / `ignis_seq` struct definitions
 * (GitHub #55).
 *
 * Not part of the public flat C ABI (ignis_seq.h keeps both types opaque
 * across the Rust boundary, ADR 0009) -- this header exists so
 * kernel/src/seq.cu and the leaf's own CTest (kernel/tests/test_seq_alloc.cpp)
 * can share one definition: the CTest verifies zero-state directly against
 * the vendored pools' device memory, which the flat ABI deliberately never
 * exposes a pointer to.
 */
#ifndef IGNIS_SEQ_INTERNAL_H
#define IGNIS_SEQ_INTERNAL_H

#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"

#include <array>
#include <cstdint>
#include <vector>

/* The GDN causal-conv kernel width (mirrors kernel/src/model.cu's
 * `kGdnConvKernel` -- the model's fixed causal-conv width, not per-model
 * config, so it is not on `ignis_seq_pool_spec` either). */
inline constexpr int32_t kIgnisGdnConvKernel = 4;

/* The conv-state history width: the 4-tap causal conv keeps width-1 = 3 past
 * taps in the state pool. The reference's LinearAttentionStatePool requires
 * conv_width == 3 (vendor/tests/test_state_store.cpp,
 * gated_delta_net/replay.cpp), NOT the kernel width above -- the conv_snapshot
 * op's slot stride is `channels * 3`, so the pool must carry 3 taps. */
inline constexpr int32_t kIgnisGdnConvStateWidth = kIgnisGdnConvKernel - 1;

/* The Qwen 3.8 text backbone has one full-attention layer every four layers:
 * 3, 7, ..., 63. Each keeps independent K/V history, so the paged pool owns
 * one K/V-plane pair and one frontier per GQA layer. */
inline constexpr int32_t kIgnisGqaLayerCount = 16;

struct ignis_seq_pool {
  ninfer::DeviceArena kv_arena;
  ninfer::PagedKVPool kv_pool;
  ninfer::DeviceArena gdn_arena;
  ninfer::LinearAttentionStatePool gdn_pool;
  std::uint64_t kv_page_bytes = 0;
  std::vector<std::int32_t> free_slots;

  // P3-03 (GitHub #99): one int32 occurrence count per vocab entry, per slot
  // -- device-side presence/frequency penalties read and atomically update
  // this row for the sampling call's sequence. A flat DeviceBuffer, not a
  // DeviceArena: every slot's region is a fixed `slot * vocab` offset (no
  // suballocation bookkeeping needed), and the whole buffer is one owning
  // cudaMalloc, mirroring kv_arena/gdn_arena's "one pool, sized once at
  // create" shape. Not zeroed here: like the KV pages and GDN state above,
  // it is zeroed per slot at `ignis_seq_alloc`, not for the whole pool at
  // creation.
  ninfer::DeviceBuffer sampling_counts;
  std::int32_t vocab = 0;

  ignis_seq_pool(std::size_t kv_bytes, const ninfer::PagedKVPoolLayout &kv_layout,
                 std::size_t gdn_bytes, const ninfer::LinearAttentionStatePoolLayout &gdn_layout,
                 std::size_t sampling_counts_bytes, std::int32_t vocab_size)
      : kv_arena(kv_bytes), kv_pool({kv_arena.base(), kv_arena.capacity()}, kv_layout),
        gdn_arena(gdn_bytes), gdn_pool({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout),
        sampling_counts(sampling_counts_bytes), vocab(vocab_size) {}

  // This slot's penalty-count row: `vocab` int32 entries, zeroed at every
  // ignis_seq_alloc of this slot.
  std::int32_t *token_counts_for(std::int32_t slot) {
    auto *base = static_cast<std::int32_t *>(sampling_counts.p);
    return base + static_cast<std::ptrdiff_t>(slot) * vocab;
  }
};

struct ignis_seq {
  ninfer::PagedKVAllocation kv;
  // Also addresses this sequence's presence/frequency penalty-count row in
  // the pool's sampling_counts buffer (`pool->token_counts_for(slot)`,
  // P3-03/#99) -- one more state section this slot owns, alongside its KV
  // pages and GDN state. No RNG state lives on `ignis_seq`: the vendored
  // sampler's RNG is a pure function of (seed, position, purpose), carried
  // entirely by the caller's per-round sampling params and this handle's own
  // `position` below, so there is nothing to snapshot beyond what a restore
  // (G4) already needs for those two.
  std::int32_t slot = -1;
  std::array<std::uint32_t, kIgnisGqaLayerCount> gqa_positions{};
  // The token which is ready to be emitted on the next decode round.  Prefill
  // consumes the complete prompt and computes this greedy successor; decode
  // returns it while consuming it to prepare the following round.
  std::int32_t pending_token = -1;
  // One program-wide frontier, distinct from the per-GQA cache frontiers
  // above.  It pins span prefill's start_position contract even for GDN-only
  // prefixes.
  std::uint64_t position = 0;
};

#endif /* IGNIS_SEQ_INTERNAL_H */
