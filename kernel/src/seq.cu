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
      !positive(spec->gdn_head_dim)) {
    set_error("ignis_seq_pool_create: every geometry field must be positive");
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
    kv_spec.planes                = {
        {ninfer::DType::BF16, static_cast<std::int32_t>(spec->head_dim),
         static_cast<std::int32_t>(spec->num_kv_heads)},
        {ninfer::DType::BF16, static_cast<std::int32_t>(spec->head_dim),
         static_cast<std::int32_t>(spec->num_kv_heads)},
    };
    const ninfer::PagedKVPoolLayout kv_layout = ninfer::plan_paged_kv_pool(kv_builder, kv_spec);
    const std::size_t kv_bytes                = kv_builder.finish(256);
    const std::uint64_t kv_page_bytes = static_cast<std::uint64_t>(kv_layout.payload_bytes()) /
                                        spec->kv_page_group_count;

    ninfer::LayoutBuilder gdn_builder;
    ninfer::LinearAttentionStatePoolSpec gdn_spec;
    gdn_spec.layers         = spec->gdn_num_layers;
    gdn_spec.conv_channels  = static_cast<std::int32_t>(spec->gdn_conv_channels);
    gdn_spec.conv_width     = kIgnisGdnConvKernel;
    gdn_spec.value_heads    = static_cast<std::int32_t>(spec->gdn_value_heads);
    gdn_spec.value_head_dim = static_cast<std::int32_t>(spec->gdn_head_dim);
    gdn_spec.key_head_dim   = static_cast<std::int32_t>(spec->gdn_head_dim);
    gdn_spec.slot_count     = static_cast<std::int32_t>(spec->slot_count);
    gdn_spec.conv_dtype     = ninfer::DType::BF16;
    const ninfer::LinearAttentionStatePoolLayout gdn_layout =
        ninfer::plan_linear_attention_state_pool(gdn_builder, gdn_spec);
    const std::size_t gdn_bytes = gdn_builder.finish(256);

    auto pool = std::make_unique<ignis_seq_pool>(kv_bytes, kv_layout, gdn_bytes, gdn_layout);
    pool->kv_page_bytes = kv_page_bytes;
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
