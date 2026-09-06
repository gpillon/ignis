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

struct ignis_seq_pool {
  ninfer::DeviceArena kv_arena;
  ninfer::PagedKVPool kv_pool;
  ninfer::DeviceArena gdn_arena;
  ninfer::LinearAttentionStatePool gdn_pool;
  std::uint64_t kv_page_bytes = 0;
  std::vector<std::int32_t> free_slots;

  ignis_seq_pool(std::size_t kv_bytes, const ninfer::PagedKVPoolLayout &kv_layout,
                 std::size_t gdn_bytes, const ninfer::LinearAttentionStatePoolLayout &gdn_layout)
      : kv_arena(kv_bytes), kv_pool({kv_arena.base(), kv_arena.capacity()}, kv_layout),
        gdn_arena(gdn_bytes), gdn_pool({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout) {}
};

struct ignis_seq {
  ninfer::PagedKVAllocation kv;
  std::int32_t slot = -1;
};

#endif /* IGNIS_SEQ_INTERNAL_H */
