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

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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

/* The DFlash2 drafter's per-sequence window geometry (P5-03, GitHub #152;
 * the reference's `DFlash2Config` and `qwen3.8-27b-artifact.md` §15.1):
 * five sliding-attention layers, each keeping BF16 K and V for the last
 * 2048 positions of 8 KV heads of 128. kernel/src/model.cu binds the
 * drafter's weights against the same numbers and checks it. */
inline constexpr std::uint32_t kIgnisDflash2Layers       = 5;
inline constexpr std::uint32_t kIgnisDflash2WindowTokens = 2048;
inline constexpr std::int32_t kIgnisDflash2KvHeads       = 8;
inline constexpr std::int32_t kIgnisDflash2HeadDim       = 128;

/* The hq-e8-2b per-row plane extents (`ops/kernel/hq_codec.cuh`'s
 * kHqRowBudgetBytes / kHqMetaBytes, restated here so this header stays
 * free of the codec's CUDA includes) and the quant_group the vendored
 * gqa_attention wrapper requires an hq cache view to declare. */
inline constexpr int32_t kIgnisHqCodeRowBytes = 64;
inline constexpr int32_t kIgnisHqMetaRowBytes = 8;
inline constexpr int32_t kIgnisHqHeadDim      = 256;
inline constexpr int32_t kIgnisHqQuantGroup   = 32;

/* The hq-e8-2b residual window's geometry (GitHub #257, spec runtime/06):
 * the vendored `kGqaHqSinkKeys` sink rows and `kGqaHqRecentKeys` recent-ring
 * rows one slot keeps exact per (GQA layer, KV head), and the ring's
 * validity words -- one bit per ring slot, `kGqaHqRecentKeys / 32` words per
 * slot row. Restated like the codec's byte budgets above and checked against
 * the vendored constants in kernel/src/seq.cu. */
inline constexpr int32_t kIgnisHqSinkKeys     = 32;
inline constexpr int32_t kIgnisHqRecentKeys   = 512;
inline constexpr int32_t kIgnisHqResidualRows = kIgnisHqSinkKeys + kIgnisHqRecentKeys;
inline constexpr int32_t kIgnisHqRingWords    = kIgnisHqRecentKeys / 32;

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
 * `ignis_kv_fill_layer_planes` below does. */
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
  /* Lane slots only, `0..slot_count`: the retained slots past them are never
   * listed here (GitHub #211). */
  std::vector<std::int32_t> free_slots;
  /* Retained slots, at slot indices `slot_count..slot_count +
   * retained_slot_count` of the GDN pool, the penalty counts and the drafter
   * lanes (GitHub #211), and what one slot's state occupies. */
  std::uint32_t retained_slot_count = 0;
  std::uint64_t slot_state_bytes = 0;
  /* Which retained slots hold a published prefix's or a captured checkpoint's
   * image (GitHub #215), one flag per slot. The caller decides which slot a
   * publish or a capture takes (`ignis_core::RetainedSlotLedger`); this is the
   * leaf refusing to write an image over one that is still claimable. */
  std::vector<bool> retained_held;

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

  // P5-03 (GitHub #152): the DFlash2 drafter's per-sequence state, present
  // only when the pool was built with IGNIS_SPECULATIVE_DFLASH2. One cyclic
  // lane per slot -- the lane index IS the slot, as it is for the GDN pool --
  // in the window the drafter attends over and in its rewrite checkpoint.
  // Both are carved from `dflash2_arena`; all three stay null without the
  // drafter, so a plain pool costs nothing and lists no drafter section.
  std::int32_t speculative_backend = 0;
  std::unique_ptr<ninfer::DeviceArena> dflash2_arena;
  std::unique_ptr<ninfer::CyclicKVCache> dflash2_window;
  std::unique_ptr<ninfer::CyclicKVCache> dflash2_checkpoint;

  bool has_dflash2() const { return dflash2_window != nullptr; }

  // Bytes one slot's lane of the window occupies (the checkpoint's is the
  // same): every layer's K and V over the whole ring. 0 without the drafter.
  std::uint64_t dflash2_lane_bytes() const {
    return has_dflash2() ? static_cast<std::uint64_t>(dflash2_window->lane_host_bytes()) : 0;
  }

  // GitHub #257 (spec runtime/06): the hq-e8-2b residual window -- the exact
  // BF16 sink and recent-ring rows the vendored hq attention kernels read
  // instead of decoding, in the codec's rotated frame, plus the ring's
  // validity words. Present only on an hq pool; a BF16 pool leaves all three
  // null, which is what keeps its views on the plain route.
  //
  // Indexed by **slot**, not by page, exactly like the reference's
  // `decoder_state.cpp`: plane (layer, slot) is dim-3 index
  // `layer * hq_residual_slots + slot` of `[256, kv_heads, 544, 16 *
  // hq_residual_slots]`, and the ring is `[16, hq_residual_slots]` words shared
  // by every layer (an append is position-driven, so the layers agree). The
  // slots are every state slot -- the lanes, then the retained slots past them
  // (GitHub #211) -- so a retained image carries its rows like any other CLONE
  // section; the views name only the lanes' leading rows. One cudaMalloc,
  // sized at create and never per request (ADR 0030).
  std::unique_ptr<ninfer::DeviceBuffer> hq_residual;
  void *hq_residual_k       = nullptr;
  void *hq_residual_v       = nullptr;
  std::uint32_t *hq_ring    = nullptr;
  std::int32_t hq_residual_slots = 0;

  bool has_hq_residual() const { return hq_residual != nullptr; }

  // One (layer, slot) side plane: every KV head's 544 rows.
  std::uint64_t hq_residual_plane_bytes() const {
    return static_cast<std::uint64_t>(kIgnisHqHeadDim) * static_cast<std::uint64_t>(kv_num_kv_heads) *
           kIgnisHqResidualRows * 2u;
  }

  // Slot `slot`'s side plane of GQA layer `gqa_layer`, role K or V.
  void *hq_residual_plane(bool role_v, std::int32_t gqa_layer, std::int32_t slot) const {
    auto *base = static_cast<unsigned char *>(role_v ? hq_residual_v : hq_residual_k);
    return base + (static_cast<std::uint64_t>(gqa_layer) * static_cast<std::uint64_t>(hq_residual_slots) +
                   static_cast<std::uint64_t>(slot)) *
                      hq_residual_plane_bytes();
  }

  // Slot `slot`'s ring validity words.
  std::uint32_t *hq_ring_words(std::int32_t slot) const {
    return hq_ring + static_cast<std::ptrdiff_t>(slot) * kIgnisHqRingWords;
  }

  // Bytes one slot's window occupies, on the device and in its host image
  // alike: every layer's K plane, then every layer's V plane, then its ring
  // words. 0 on a BF16 pool.
  std::uint64_t hq_residual_slot_bytes() const {
    return has_hq_residual() ? 2u * kIgnisGqaLayerCount * hq_residual_plane_bytes() +
                                   kIgnisHqRingWords * sizeof(std::uint32_t)
                             : 0;
  }

  // What one retained image holds on the device: a slot's state and, on an
  // hq pool, its residual window (GitHub #257) -- the window rides every
  // clone, but the VRAM plan counts it on its own line rather than in
  // `slot_state_bytes`.
  std::uint64_t retained_image_bytes() const { return slot_state_bytes + hq_residual_slot_bytes(); }

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

/* The shared prefix a sequence may hold (P4-10, GitHub #126). Defined in
 * ignis_seq_prefix_internal.h, which needs the state-section table and so
 * includes this header -- a pointer is all `ignis_seq` needs. */
struct ignis_seq_prefix;

struct ignis_seq {
  /* This sequence's OWN KV pages: the tail it writes itself. A sequence
   * holding a shared prefix does not own its leading pages, so this
   * allocation is not its whole history -- `ignis_seq_logical_page_count`
   * below is, and the block-table row addresses both halves in order. */
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
  // GitHub #178: the rope delta a multimodal prompt left (its largest
  // position + 1 - its length). Every GQA rotation after the prompt is at
  // `position + rope_delta`; `position` stays the KV index and the sampler's
  // key. 0 for a text sequence. A progress scalar (GitHub #194): snapshot and
  // restore carry it; a clone's capture records 0, since a claimant's delta
  // comes from its own tail span.
  std::int32_t rope_delta = 0;
  // P5-03 (GitHub #152): the drafter window's own frontier -- one past the
  // last absolute position whose context K/V it holds (the ring keeps the
  // 2048 before it). Only a pool with the drafter moves it: a prefill writes
  // its span's tail into the window and leaves this at the span's end.
  // Decode does not move it yet, so it may trail `position`; that is why it
  // is carried in the progress image rather than derived from `position`.
  std::uint64_t dflash2_position = 0;
  // GitHub #157: a verify round at extent 0 commits its anchor and leaves
  // the window untouched (#155 AC 3), so the frontier stays at the anchor.
  // The anchor's feature taps -- BF16 [5 x hidden], one column -- are kept
  // here instead (the reference's `pending_features`), and whatever
  // continues the sequence (its next round, a prefill) appends them at
  // `dflash2_position` before anything else. Host-only and never a section:
  // a snapshot or a prefix clone taken here carries the frontier but not
  // the taps, and its sequence resumes with the hole this closes -- an
  // acceptance loss, never a different text. The scheduler never snapshots
  // such a sequence: extent 0 is a request's last round.
  bool dflash2_pending = false;
  std::vector<std::uint8_t> dflash2_pending_features;
  // The shared prefix this sequence claims (P4-10, GitHub #126), or null.
  // One reference is held for as long as this handle lives; `shared_pages`
  // restates its page count so every capacity question here can be answered
  // without the prefix's own definition.
  ignis_seq_prefix *prefix  = nullptr;
  std::uint32_t shared_pages = 0;
};

/* Whether `seq` was actually drawn from `pool`.
 *
 * Every call that hands a sequence and a pool to the vendored pools together
 * needs this first: the vendored side's own mismatch check aborts the
 * process (CUDA_CHECK / std::invalid_argument out of a noexcept path), so
 * the pairing is refused here as a plain bad argument instead.
 *
 * A sequence's slot is a lane's, below the KV block-table rows: the GDN pool
 * holds the retained slots past them (GitHub #211), and no sequence stands on
 * one. */
inline bool ignis_seq_belongs_to(const ignis_seq_pool &pool, const ignis_seq &seq) {
  return seq.kv.valid() && seq.kv.belongs_to(pool.kv_pool) && seq.slot >= 0 &&
         seq.slot < pool.kv_pool.table_row_count();
}

/* Count one allocation of `bytes` (`alloc`) or one free of `kind` (enum
 * ignis_alloc_kind), for ignis_alloc_counts (GitHub #211). Defined in
 * kernel/src/seq.cu. */
void ignis_alloc_count_record(std::int32_t kind, bool alloc, std::uint64_t bytes);

/* --- retained slots (GitHub #211, #215, ADR 0030) -------------------------- */

/* The pool slot index of retained slot `retained_slot`: past every lane. */
inline std::int32_t ignis_seq_retained_pool_slot(const ignis_seq_pool &pool,
                                                 std::uint32_t retained_slot) {
  return pool.kv_pool.table_row_count() + static_cast<std::int32_t>(retained_slot);
}

/* Why an image cannot be written into retained slot `retained_slot`, or an
 * empty string when it can: a slot past the pool's retained slots, or one
 * still holding a claimable image. */
inline std::string ignis_seq_retained_slot_refusal(const ignis_seq_pool &pool,
                                                   std::uint32_t retained_slot) {
  if (retained_slot >= pool.retained_slot_count) {
    return "retained slot " + std::to_string(retained_slot) + " is out of range; this pool holds " +
           std::to_string(pool.retained_slot_count);
  }
  if (pool.retained_held[retained_slot]) {
    return "retained slot " + std::to_string(retained_slot) +
           " still holds a prefix's or a checkpoint's image; it comes back when that handle is "
           "released";
  }
  return {};
}

/* Copy every mutable state section of pool slot `src` over pool slot `dst`,
 * device to device, and synchronize: a lane into a retained slot, a retained
 * slot into a lane. Walks the CLONE sections of the state-section table, so a
 * section added there without a case here throws rather than being silently
 * left behind (ADR 0024's "carried by all or by none"). Defined in
 * kernel/src/seq.cu. */
void ignis_seq_copy_slot_state(ignis_seq_pool &pool, std::int32_t src, std::int32_t dst);

/* GitHub #257: pool slot `slot`'s hq residual window to (`kind` =
 * cudaMemcpyDeviceToHost) or from (cudaMemcpyHostToDevice) `host`, laid out
 * as the IGNIS_SEQ_SECTION_HQ_RESIDUAL payload. Enqueued on the default
 * stream; the caller synchronizes. Throws on a failed copy. Defined in
 * kernel/src/seq.cu. */
void ignis_seq_copy_hq_residual(const ignis_seq_pool &pool, std::int32_t slot, void *host,
                                cudaMemcpyKind kind);

/* Zero pool slot `slot`'s hq residual window -- every side row and every ring
 * bit -- so a slot handed to a new sequence never serves its previous
 * occupant's rows: the ring bits because they carry no position, and the
 * sink rows because the kernels read them with no bit at all. A no-op on a
 * BF16 pool. Throws on a failed memset. */
void ignis_seq_zero_hq_residual(ignis_seq_pool &pool, std::int32_t slot);

/* The leaf's thread-local last-error slot -- the one `ignis_seq_last_error`
 * reports. Defined in kernel/src/seq.cu and written by kernel/src/seq_prefix.cu
 * too, so one error surface answers for every sequence entry point rather
 * than one per translation unit. */
void ignis_seq_set_last_error(std::string message);

/* The pages of a sequence's history: the prefix's, which it shares, plus its
 * own. **Not** `kv.mapped_page_count()`, which is only the half it owns --
 * every capacity question (how many tokens fit, where the next one lands)
 * is about this number, because the block-table row addresses both halves in
 * order. */
inline std::uint32_t ignis_seq_logical_page_count(const ignis_seq &seq) {
  return seq.shared_pages + seq.kv.mapped_page_count();
}

/* Tokens the sequence's mapped history can hold, prefix included. The
 * sibling of `ninfer::PagedKVAllocation::mapped_token_capacity` for a
 * sequence that may not own all of its pages. */
inline std::uint64_t ignis_seq_token_capacity(const ignis_seq &seq) {
  return static_cast<std::uint64_t>(ignis_seq_logical_page_count(seq)) *
         static_cast<std::uint64_t>(ninfer::kPagedKVPageSize);
}

/* The cache element type a format's rows are declared as, and the
 * quant_group that declaration has to carry (P4-05, GitHub #123).
 *
 * These two are what route the vendored `gqa_attention` family: its wrapper
 * dispatches on `cache.dtype`, so declaring U8 with quant_group 32 IS the
 * act of selecting the hq attention kernels over the BF16 ones. Stated once
 * here, beside the plane mapping, so no call site can select the hq planes
 * and the BF16 route (or the reverse) by editing only half of a view. */
inline ninfer::DType ignis_kv_cache_dtype(std::int32_t kv_format) {
  return kv_format == IGNIS_KV_FORMAT_HQ_E8_2B ? ninfer::DType::U8 : ninfer::DType::BF16;
}

inline std::int32_t ignis_kv_quant_group(std::int32_t kv_format) {
  /* A BF16 cache must declare none at all; the hq wrapper requires 32. */
  return kv_format == IGNIS_KV_FORMAT_HQ_E8_2B ? kIgnisHqQuantGroup : 0;
}

/* The operator-facing spelling of a format, for a message a human reads.
 * The same two names `ignis_core::KvFormat::as_str`, the `--kv-format` flag
 * and `CONTEXT.md` use -- a leaf error that printed the raw enum ordinal
 * would make the reader translate it back. `"unknown"` covers a value the
 * ABI rejected, which is the only way one gets this far. */
inline const char *ignis_kv_format_name(std::int32_t kv_format) {
  switch (kv_format) {
  case IGNIS_KV_FORMAT_BF16:
    return "bf16";
  case IGNIS_KV_FORMAT_HQ_E8_2B:
    return "hq-e8-2b";
  default:
    return "unknown";
  }
}

/* Fills the plane set, dtype and quant_group `view` needs for one GQA
 * layer's history under `pool`'s format. Shared by the single-sequence and
 * the batched view builders below, which differ only in how they name the
 * block table -- one sequence's row, or the pool-wide matrix.
 *
 * Under hq-e8-2b the value planes carry the codec's 64-byte code rows and
 * the `*_scale_pages` slots carry its 8-byte metadata rows (the slots the
 * vendored gqa_attention wrapper reads hq metadata from). Page addressing is
 * the same `paged_kv_element_offset` in both formats; only a plane's leading
 * extent differs, which is what keeps capacity math format-independent.
 *
 * The residual window is not filled here: it is indexed by slot row rather
 * than by page, so the two builders name it differently
 * (`ignis_kv_fill_residual` below). */
template <class View>
inline void ignis_kv_fill_layer_planes(View &view, ignis_seq_pool *pool, std::int32_t gqa_layer) {
  view.k_pages =
      pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_K));
  view.v_pages =
      pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_V));
  if (pool->kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    view.k_scale_pages = pool->kv_pool.plane(
        ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_K_META));
    view.v_scale_pages = pool->kv_pool.plane(
        ignis_kv_plane_index(pool->kv_format, gqa_layer, IGNIS_KV_PLANE_V_META));
  }
  view.dtype        = ignis_kv_cache_dtype(pool->kv_format);
  view.quant_group  = ignis_kv_quant_group(pool->kv_format);
  view.head_dim     = pool->kv_head_dim;
  view.num_kv_heads = pool->kv_num_kv_heads;
}

/* The hq-e8-2b residual window (GitHub #257, spec runtime/06) of GQA layer
 * `gqa_layer` over `slots` consecutive slot rows from `first_slot`: the
 * exact side planes `[256, kv_heads, 544, slots]` and the ring words `[16,
 * slots]` the vendored hq kernels read when a view carries them. Every hq
 * kernel guards the window on a null pointer, so a BF16 pool -- which
 * allocates none -- leaves all three empty and keeps the plain route.
 *
 * The tensors are built over the pool's own storage rather than sliced from
 * a whole-pool tensor: the ring words of `slots` rows are `slots * 64`
 * contiguous bytes, and a slice of a wider `[16, state_slots]` tensor would
 * keep the wider row stride the vendored contiguity check refuses. */
template <class View>
inline void ignis_kv_fill_residual(View &view, const ignis_seq_pool *pool, std::int32_t gqa_layer,
                                   std::int32_t first_slot, std::int32_t slots) {
  if (!pool->has_hq_residual()) {
    return;
  }
  const std::initializer_list<std::int32_t> plane = {kIgnisHqHeadDim, pool->kv_num_kv_heads,
                                                     kIgnisHqResidualRows, slots};
  view.residual_k = ninfer::Tensor(pool->hq_residual_plane(false, gqa_layer, first_slot),
                                   ninfer::DType::BF16, plane);
  view.residual_v = ninfer::Tensor(pool->hq_residual_plane(true, gqa_layer, first_slot),
                                   ninfer::DType::BF16, plane);
  view.ring_valid = ninfer::Tensor(pool->hq_ring_words(first_slot), ninfer::DType::I32,
                                   {kIgnisHqRingWords, slots, 1, 1});
}

/* The single-sequence cache view, for the ops that take one: A2
 * (`gqa_kv_append`) and A3 (`gqa_attention_cached`).
 *
 * No production caller today, and that is not an oversight. P2-04 (GitHub
 * #86) replaced the layer's A2+A3 composition with the fused A1, which takes
 * the *batched* view below, so the only callers left are the leaf's own
 * append and route-agreement tests. It is kept because those tests must
 * reach the plane mapping through the one function that decides it
 * (kernel/tests/test_kv_append_format.cu,
 * kernel/tests/test_hq_route_agreement.cu) -- each building its own view
 * would leave a bug in the shared mapping invisible, which is the whole
 * reason the mapping lives in this header.
 *
 * The view is non-owning: `seq`'s allocation keeps the mapping and pages
 * alive for as long as the caller uses it. */
inline ninfer::PagedKVLayerView ignis_kv_layer_view(ignis_seq_pool *pool, ignis_seq *seq,
                                                    std::int32_t gqa_layer) {
  ninfer::PagedKVLayerView view;
  ignis_kv_fill_layer_planes(view, pool, gqa_layer);
  // A layer view arrives pre-sliced to its sequence's slot row (the vendored
  // `GqaPrefillDirectMetadata::residual_slot` is 0).
  ignis_kv_fill_residual(view, pool, gqa_layer, seq->slot, 1);
  view.block_table = seq->kv.block_table();
  return view;
}

/* The batched counterpart (P4-05, GitHub #123): the same planes over the
 * pool-wide block-table matrix, which is what a decode round's A1 call takes
 * -- row `b` of it is lane `b`'s own page list, selected by the round's
 * `kv_table_rows`. No `ignis_seq*`: every address here belongs to the pool
 * for its lifetime, which is what lets a captured decode graph replay it
 * (ADR 0019). */
inline ninfer::PagedKVBatchLayerView ignis_kv_batch_layer_view(ignis_seq_pool *pool,
                                                               std::int32_t gqa_layer) {
  ninfer::PagedKVBatchLayerView view;
  ignis_kv_fill_layer_planes(view, pool, gqa_layer);
  // Every lane's row, in block-table order: the kernels offset the window by
  // the table row they already selected, and `validate_residual` wants
  // exactly as many rows as the block tables have. The retained slots past
  // the lanes hold images, never a row a round names.
  ignis_kv_fill_residual(view, pool, gqa_layer, 0, pool->kv_pool.table_row_count());
  view.block_tables = pool->kv_pool.block_tables();
  return view;
}

#endif /* IGNIS_SEQ_INTERNAL_H */
