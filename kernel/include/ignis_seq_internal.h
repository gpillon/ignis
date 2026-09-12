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

#include "ignis_seq.h"

#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"

#include <array>
#include <cassert>
#include <cstddef>
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

/* The other 48 of the 64 backbone layers: 0, 1, 2, 4, 5, 6, 8, ... Each keeps
 * its own conv taps and recurrent state in the linear-attention state pool,
 * so the pool is sized by this count and `ignis_seq::gdn_positions` carries
 * one frontier per layer, exactly as `gqa_positions` does for the 16 above. */
inline constexpr int32_t kIgnisGdnLayerCount = 48;

/* The hq-e8-2b per-row plane extents (`ops/kernel/hq_codec.cuh`'s
 * kHqRowBudgetBytes / kHqMetaBytes, restated here so this header stays
 * free of the codec's CUDA includes) and the quant_group the vendored
 * gqa_attention wrapper requires an hq cache view to declare. */
inline constexpr int32_t kIgnisHqCodeRowBytes = 64;
inline constexpr int32_t kIgnisHqMetaRowBytes = 8;
inline constexpr int32_t kIgnisHqHeadDim      = 256;
inline constexpr int32_t kIgnisHqQuantGroup   = 32;

/* Planes one GQA layer's K/V history occupies, per format: BF16 stores one
 * plane per role, hq-e8-2b a code plane and a metadata plane per role. The
 * plane order is (K..., V...) in both, so a layer's planes are a contiguous
 * run and `ignis_kv_plane_index` below is the only place that knows the
 * stride. */
inline constexpr int32_t kIgnisKvPlanesPerLayerBf16 = 2;
inline constexpr int32_t kIgnisKvPlanesPerLayerHq   = 4;

inline constexpr int32_t ignis_kv_planes_per_layer(int32_t kv_format) {
  return kv_format == IGNIS_KV_FORMAT_HQ_E8_2B ? kIgnisKvPlanesPerLayerHq
                                               : kIgnisKvPlanesPerLayerBf16;
}

/* Plane roles inside one layer's run, in allocation order. Under BF16 only
 * the two value planes exist; under hq each role's value plane is the code
 * plane and is followed by its metadata plane. */
enum ignis_kv_plane_role {
  IGNIS_KV_PLANE_K       = 0,
  IGNIS_KV_PLANE_K_META  = 1,
  IGNIS_KV_PLANE_V       = 2,
  IGNIS_KV_PLANE_V_META  = 3
};

/* The pool plane index of one GQA layer's `role` plane.
 *
 * BF16 has no metadata planes at all, so its V plane sits at offset 1, not
 * 2, and asking a BF16 pool for a metadata role is a caller bug: there is no
 * plane that could answer, so it asserts rather than handing back a
 * plausible wrong plane. Check the format first, the way
 * `ignis_kv_layer_view` below does. */
inline std::size_t ignis_kv_plane_index(int32_t kv_format, int32_t gqa_layer,
                                        ignis_kv_plane_role role) {
  const bool wants_meta = role == IGNIS_KV_PLANE_K_META || role == IGNIS_KV_PLANE_V_META;
  assert((!wants_meta || kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) &&
         "a BF16 KV pool has no metadata planes");
  const bool is_v      = role == IGNIS_KV_PLANE_V || role == IGNIS_KV_PLANE_V_META;
  const int32_t within = kv_format == IGNIS_KV_FORMAT_HQ_E8_2B ? static_cast<int32_t>(role)
                                                               : (is_v ? 1 : 0);
  return static_cast<std::size_t>(gqa_layer) *
             static_cast<std::size_t>(ignis_kv_planes_per_layer(kv_format)) +
         static_cast<std::size_t>(within);
}

struct ignis_seq_pool {
  ninfer::DeviceArena kv_arena;
  ninfer::PagedKVPool kv_pool;
  ninfer::DeviceArena gdn_arena;
  ninfer::LinearAttentionStatePool gdn_pool;
  std::uint64_t kv_page_bytes = 0;
  /* One of enum ignis_kv_format: what every plane above stores, fixed for
   * the life of this pool (ADR 0022). */
  std::int32_t kv_format = IGNIS_KV_FORMAT_BF16;
  /* The head geometry the pool was built with. Kept because it cannot be
   * read back off the planes under every format: an hq code plane's leading
   * extent is the codec's row budget, not head_dim. */
  std::int32_t kv_head_dim     = 0;
  std::int32_t kv_num_kv_heads = 0;
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

  // The same row for a read-only caller: `ignis_seq_snapshot` takes the pool
  // by const pointer, because capturing a sequence must not be able to
  // change one (P4-06, GitHub #124).
  const std::int32_t *token_counts_for(std::int32_t slot) const {
    const auto *base = static_cast<const std::int32_t *>(sampling_counts.p);
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
  // The GDN layers' own frontiers (P4-06, GitHub #124). Unlike `gqa_positions`
  // no kernel reads these -- a GDN layer's state is updated in place and
  // carries its own history -- but something has to be able to say whether
  // every layer has consumed the same tokens, and for the 48 GDN layers
  // nothing could. `ignis_seq_at_chunk_boundary` needs both arrays: a
  // sequence stepped one GDN layer at a time has state ahead of its KV, and
  // a snapshot taken there would restore into a subtly wrong sequence.
  std::array<std::uint32_t, kIgnisGdnLayerCount> gdn_positions{};
  // The token which is ready to be emitted on the next decode round.  Prefill
  // consumes the complete prompt and computes this greedy successor; decode
  // returns it while consuming it to prepare the following round.
  std::int32_t pending_token = -1;
  // One program-wide frontier, distinct from the per-GQA cache frontiers
  // above.  It pins span prefill's start_position contract even for GDN-only
  // prefixes.
  std::uint64_t position = 0;
};

/* The single-sequence cache view one GQA layer's ops take, built from the
 * pool's own KV format (P4-04, GitHub #122).
 *
 * Lives here rather than in kernel/src/gqa_layer.cu so the one function that
 * decides which planes and which declared dtype an hq view carries is the
 * one under test (kernel/tests/test_kv_append_format.cu) -- a second copy in
 * the test would leave a bug in this one invisible.
 *
 * Under hq-e8-2b the value planes carry the codec's 64-byte code rows and
 * the `*_scale_pages` slots carry its 8-byte metadata rows (the slots the
 * vendored gqa_attention wrapper reads hq metadata from), with quant_group
 * 32, which that wrapper requires an hq view to declare. Page addressing is
 * the same `paged_kv_element_offset` in both formats; only a plane's leading
 * extent differs, which is what keeps capacity math format-independent.
 *
 * The view is non-owning: `seq`'s allocation keeps the mapping and pages
 * alive for as long as the caller uses it. The residual planes stay empty --
 * the hq residual window is not a feature this engine has opted into. */
inline ninfer::PagedKVLayerView ignis_kv_layer_view(ignis_seq_pool *pool, ignis_seq *seq,
                                                    std::int32_t gqa_layer) {
  ninfer::PagedKVLayerView view;
  view.k_pages =
      pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_K));
  view.v_pages =
      pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_V));
  if (pool->kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    view.k_scale_pages = pool->kv_pool.plane(
        ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_K_META));
    view.v_scale_pages = pool->kv_pool.plane(
        ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_V_META));
    view.dtype       = ninfer::DType::U8;
    view.quant_group = kIgnisHqQuantGroup;
  } else {
    view.dtype = ninfer::DType::BF16;
  }
  view.block_table  = seq->kv.block_table();
  view.head_dim     = pool->kv_head_dim;
  view.num_kv_heads = pool->kv_num_kv_heads;
  return view;
}

#endif /* IGNIS_SEQ_INTERNAL_H */
