// ignis kernel leaf - P1-19 (GitHub #55): sequence handle flat C ABI (ADR 0009).
//
// Builds the two device-resident pools (paged KV, GDN state) once at
// ignis_seq_pool_create and hands out slots from them: ignis_seq_alloc
// reserves + materializes KV pages, binds the block-table row, and zeroes
// both the fresh KV pages and the GDN slot before returning a handle;
// ignis_seq_release returns everything (ninfer::PagedKVAllocation's
// destructor unbinds the row and frees the pages; the slot itself returns
// to this file's own free list, since the vendored pools carry no slot
// allocator of their own -- core/paged_kv_cache.h's row_in_use_ bitmap and
// core/linear_attention_state.h's slot addressing are both private/caller
// driven).
//
// Style follows model.cu: explicit pointers + sizes, int32 return codes (0
// = ok, -1 = error, IGNIS_SEQ_ERR_NOT_IMPLEMENTED for the snapshot/restore
// stubs), no C++ types across the boundary.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "ops/kernel/hq_codec.cuh"

#include <cuda_runtime.h>

// `ignis_seq_internal.h` restates the codec's per-row byte budgets so that
// header stays free of the codec's CUDA includes. This translation unit has
// both, so it is where the restatement is checked: VENDOR.md's `verify`
// catches an edited vendored file, never a constant copied out of one.
static_assert(kIgnisHqCodeRowBytes == ninfer::ops::kHqRowBudgetBytes,
              "kIgnisHqCodeRowBytes has drifted from the vendored kHqRowBudgetBytes");
static_assert(kIgnisHqMetaRowBytes == ninfer::ops::kHqMetaBytes,
              "kIgnisHqMetaRowBytes has drifted from the vendored kHqMetaBytes");
static_assert(kIgnisHqHeadDim == ninfer::ops::kHqHeadDim,
              "kIgnisHqHeadDim has drifted from the vendored kHqHeadDim");

#include <memory>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

bool positive(uint32_t v) {
  return v > 0;
}

// One GQA layer's storage planes under `kv_format`, appended in the order
// `ignis_kv_plane_index` addresses them (K, [K meta,] V, [V meta]).
//
// This is the only place that says what a KV format is made of. BF16 stores
// the rows themselves; hq-e8-2b stores a fixed 64-byte code row plus an
// 8-byte metadata row per (token, kv_head), which is why its planes are U8
// with the codec's byte budgets as their leading extents rather than
// head_dim. Both keep the paged-KV contract's fixed-bytes-per-token
// property, so `plan_paged_kv_pool` sizes them identically in shape and the
// capacity math below never branches on the format.
void push_layer_planes(ninfer::PagedKVPoolSpec &spec, int32_t kv_format, uint32_t num_kv_heads,
                       uint32_t head_dim) {
  const auto heads = static_cast<std::int32_t>(num_kv_heads);
  if (kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    spec.planes.push_back({ninfer::DType::U8, kIgnisHqCodeRowBytes, heads});
    spec.planes.push_back({ninfer::DType::U8, kIgnisHqMetaRowBytes, heads});
    spec.planes.push_back({ninfer::DType::U8, kIgnisHqCodeRowBytes, heads});
    spec.planes.push_back({ninfer::DType::U8, kIgnisHqMetaRowBytes, heads});
    return;
  }
  spec.planes.push_back({ninfer::DType::BF16, static_cast<std::int32_t>(head_dim), heads});
  spec.planes.push_back({ninfer::DType::BF16, static_cast<std::int32_t>(head_dim), heads});
}

bool known_kv_format(int32_t kv_format) {
  return kv_format == IGNIS_KV_FORMAT_BF16 || kv_format == IGNIS_KV_FORMAT_HQ_E8_2B;
}

} // namespace

extern "C" int32_t ignis_seq_pool_create(const struct ignis_seq_pool_spec *spec,
                                          struct ignis_seq_pool **out_pool) {
  if (out_pool != nullptr) {
    *out_pool = nullptr;
  }
  if (spec == nullptr || out_pool == nullptr) {
    set_error("ignis_seq_pool_create: null argument");
    return -1;
  }
  if (!positive(spec->num_kv_heads) || !positive(spec->head_dim) ||
      !positive(spec->kv_page_group_count) || !positive(spec->max_context_tokens) ||
      !positive(spec->slot_count) || !positive(spec->gdn_num_layers) ||
      !positive(spec->gdn_conv_channels) || !positive(spec->gdn_value_heads) ||
      !positive(spec->gdn_head_dim) || !positive(spec->vocab)) {
    set_error("ignis_seq_pool_create: every geometry field must be positive");
    return -1;
  }
  if (!known_kv_format(spec->kv_format)) {
    set_error("ignis_seq_pool_create: kv_format " + std::to_string(spec->kv_format) +
              " is not an ignis_kv_format");
    return -1;
  }
  // The codec's row budget is defined for a 256-dimension row only
  // (kHqHeadDim); a pool of any other head_dim would plan planes the hq
  // append path cannot write.
  if (spec->kv_format == IGNIS_KV_FORMAT_HQ_E8_2B &&
      spec->head_dim != static_cast<std::uint32_t>(kIgnisHqHeadDim)) {
    set_error("ignis_seq_pool_create: hq-e8-2b KV requires head_dim " +
              std::to_string(kIgnisHqHeadDim) + ", got " + std::to_string(spec->head_dim));
    return -1;
  }

  try {
    const auto logical_page_capacity = ninfer::pages_for_tokens(spec->max_context_tokens);

    ninfer::LayoutBuilder kv_builder;
    ninfer::PagedKVPoolSpec kv_spec;
    kv_spec.page_group_count      = spec->kv_page_group_count;
    kv_spec.logical_page_capacity = logical_page_capacity;
    kv_spec.table_rows            = static_cast<std::int32_t>(spec->slot_count);
    kv_spec.plane_order           = ninfer::PagedKVPlaneOrder::PageMajor;
    // One K/V plane run per full-attention layer. The GQA layer program
    // selects its own run (`ignis_kv_plane_index`), so a layer's K/V history
    // never aliases another layer's pages.
    kv_spec.planes.reserve(static_cast<std::size_t>(ignis_kv_planes_per_layer(spec->kv_format)) *
                           kIgnisGqaLayerCount);
    for (int32_t layer = 0; layer < kIgnisGqaLayerCount; ++layer) {
      push_layer_planes(kv_spec, spec->kv_format, spec->num_kv_heads, spec->head_dim);
    }
    const ninfer::PagedKVPoolLayout kv_layout = ninfer::plan_paged_kv_pool(kv_builder, kv_spec);
    const std::size_t kv_bytes                = kv_builder.finish(256);
    const std::uint64_t kv_page_bytes = static_cast<std::uint64_t>(kv_layout.payload_bytes()) /
                                        spec->kv_page_group_count;

    ninfer::LayoutBuilder gdn_builder;
    ninfer::LinearAttentionStatePoolSpec gdn_spec;
    gdn_spec.layers         = spec->gdn_num_layers;
    gdn_spec.conv_channels  = static_cast<std::int32_t>(spec->gdn_conv_channels);
    // The conv STATE (history) is width-1 = 3 taps, not the kernel width: the
    // conv_snapshot op's slot stride is `channels * 3` and the reference's
    // state pool requires conv_width == 3 (GitHub #58, GDN layer).
    gdn_spec.conv_width     = kIgnisGdnConvStateWidth;
    gdn_spec.value_heads    = static_cast<std::int32_t>(spec->gdn_value_heads);
    gdn_spec.value_head_dim = static_cast<std::int32_t>(spec->gdn_head_dim);
    gdn_spec.key_head_dim   = static_cast<std::int32_t>(spec->gdn_head_dim);
    gdn_spec.slot_count     = static_cast<std::int32_t>(spec->slot_count);
    gdn_spec.conv_dtype     = ninfer::DType::BF16;
    const ninfer::LinearAttentionStatePoolLayout gdn_layout =
        ninfer::plan_linear_attention_state_pool(gdn_builder, gdn_spec);
    const std::size_t gdn_bytes = gdn_builder.finish(256);

    // P3-03 (GitHub #99): one int32 penalty count per vocab entry, per slot.
    const std::size_t sampling_counts_bytes = static_cast<std::size_t>(spec->slot_count) *
                                              static_cast<std::size_t>(spec->vocab) *
                                              sizeof(std::int32_t);

    auto pool = std::make_unique<ignis_seq_pool>(kv_bytes, kv_layout, gdn_bytes, gdn_layout,
                                                 sampling_counts_bytes,
                                                 static_cast<std::int32_t>(spec->vocab));
    pool->kv_page_bytes   = kv_page_bytes;
    pool->kv_format       = spec->kv_format;
    pool->kv_head_dim     = static_cast<std::int32_t>(spec->head_dim);
    pool->kv_num_kv_heads = static_cast<std::int32_t>(spec->num_kv_heads);
    pool->free_slots.reserve(spec->slot_count);
    for (std::uint32_t i = 0; i < spec->slot_count; ++i) {
      pool->free_slots.push_back(static_cast<std::int32_t>(i));
    }

    *out_pool = pool.release();
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_seq_pool_create: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_pool_stats(const struct ignis_seq_pool *pool,
                                         struct ignis_seq_pool_stats *out_stats) {
  if (pool == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->kv_page_group_count  = pool->kv_pool.page_group_count();
  out_stats->kv_entitled_pages    = pool->kv_pool.entitled_pages();
  out_stats->kv_free_pages        = pool->kv_pool.free_pages();
  out_stats->kv_page_bytes        = pool->kv_page_bytes;
  out_stats->logical_page_capacity = pool->kv_pool.logical_page_capacity();
  out_stats->slot_count           = static_cast<std::uint32_t>(pool->kv_pool.table_row_count());
  out_stats->free_slot_count      = static_cast<std::uint32_t>(pool->free_slots.size());
  out_stats->kv_format            = pool->kv_format;
  // GitHub #122: both derived from the pool the leaf actually planned --
  // `kv_page_bytes` came out of `plan_paged_kv_pool` over this format's
  // planes, and the page size is the vendored `kPagedKVPageSize`. Nothing
  // here is a per-format constant, so a format (or budget) change shows up
  // as a different number rather than as a surprise under load.
  out_stats->kv_bytes_per_token =
      pool->kv_page_bytes / static_cast<std::uint64_t>(ninfer::kPagedKVPageSize);
  out_stats->kv_token_capacity = static_cast<std::uint64_t>(pool->kv_pool.page_group_count()) *
                                 static_cast<std::uint64_t>(ninfer::kPagedKVPageSize);
  return 0;
}

extern "C" void ignis_seq_pool_free(struct ignis_seq_pool *pool) {
  delete pool;
}

extern "C" int32_t ignis_seq_alloc(struct ignis_seq_pool *pool, uint32_t context_tokens,
                                    struct ignis_seq **out_seq) {
  if (out_seq != nullptr) {
    *out_seq = nullptr;
  }
  if (pool == nullptr || out_seq == nullptr) {
    set_error("ignis_seq_alloc: null argument");
    return -1;
  }
  if (context_tokens == 0) {
    set_error("ignis_seq_alloc: context_tokens must be positive");
    return -1;
  }
  if (pool->free_slots.empty()) {
    set_error("ignis_seq_alloc: sequence pool exhausted (no free slot)");
    return -1;
  }
  const std::uint32_t pages_needed = ninfer::pages_for_tokens(context_tokens);
  if (!pool->kv_pool.can_reserve(pages_needed)) {
    set_error("ignis_seq_alloc: sequence pool exhausted (KV pages)");
    return -1;
  }

  // Peek (not pop) the candidate slot: on any failure below, `seq`'s
  // destructor unwinds the KV reservation (PagedKVAllocation::release --
  // return pages, unbind the row) automatically, so `free_slots` must stay
  // untouched until every step has actually succeeded.
  const std::int32_t slot = pool->free_slots.back();
  try {
    auto seq        = std::make_unique<ignis_seq>();
    seq->kv         = pool->kv_pool.reserve(pages_needed);
    seq->kv.materialize_pages(pages_needed);
    seq->kv.bind_row(slot);
    pool->kv_pool.zero_pages(seq->kv.page_ids());
    pool->gdn_pool.zero_slot(slot);
    // P3-03 (GitHub #99): zero this slot's penalty-count row so a
    // re-allocated slot never inherits another request's counts.
    const cudaError_t err =
        cudaMemset(pool->token_counts_for(slot), 0,
                  static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t));
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemset(sampling counts) failed: ") +
                               cudaGetErrorString(err));
    }
    seq->slot = slot;

    pool->free_slots.pop_back();
    *out_seq = seq.release();
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_seq_alloc: ") + e.what());
    return -1;
  }
}

extern "C" void ignis_seq_release(struct ignis_seq_pool *pool, struct ignis_seq *seq) {
  if (seq == nullptr) {
    return;
  }
  const std::int32_t slot = seq->slot;
  delete seq; // ~PagedKVAllocation: unbind the row, return the KV pages.
  if (pool != nullptr && slot >= 0) {
    pool->free_slots.push_back(slot);
  }
}

extern "C" int32_t ignis_seq_stats(const struct ignis_seq *seq, struct ignis_seq_stats *out_stats) {
  if (seq == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->slot             = seq->slot;
  out_stats->page_entitlement = seq->kv.page_entitlement();
  out_stats->mapped_pages     = seq->kv.mapped_page_count();
  out_stats->token_capacity   = seq->kv.mapped_token_capacity();
  return 0;
}

extern "C" int32_t ignis_seq_snapshot(const struct ignis_seq *seq, void *dst, uint64_t dst_bytes) {
  (void)seq;
  (void)dst;
  (void)dst_bytes;
  return IGNIS_SEQ_ERR_NOT_IMPLEMENTED;
}

extern "C" int32_t ignis_seq_restore(struct ignis_seq *seq, const void *src, uint64_t src_bytes) {
  (void)seq;
  (void)src;
  (void)src_bytes;
  return IGNIS_SEQ_ERR_NOT_IMPLEMENTED;
}

extern "C" const char *ignis_seq_last_error(void) {
  return g_last_error.c_str();
}
