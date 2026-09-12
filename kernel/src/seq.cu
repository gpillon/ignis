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
// P4-06 (GitHub #124, ADR 0024) adds state transfer on top: a snapshot size
// query, a blob format version, and the two whole-sequence transfers
// themselves. Everything about what a sequence is made of comes from one
// table (ignis_seq_sections.h) -- this file only moves bytes and decides
// what to refuse.
//
// Style follows model.cu: explicit pointers + sizes, int32 return codes (0
// = ok, -1 = error, IGNIS_SEQ_ERR_NOT_AT_BOUNDARY / _BAD_SNAPSHOT for the
// two refusals state transfer makes), no C++ types across the boundary.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

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

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  ignis_seq_set_last_error(std::move(message));
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

// ---- state transfer (P4-06, GitHub #124, ADR 0024) -----------------------

// Why `seq` cannot be moved as one blob, or nullptr if it can.
//
// A sequence that claims a shared prefix (P4-10, GitHub #126) does not own
// its leading KV pages: they are the prefix's, refcounted and addressed by
// every claimant's block-table row. A snapshot of it would either copy
// another request's history into this request's blob, or leave a hole a
// restore would read as zeroed attention -- so the transfer is refused
// rather than approximated. Releasing the sequence drops the claim, and a
// re-prefill is the way back on.
const char *shared_prefix_refusal(const ignis_seq &seq) {
  if (seq.prefix == nullptr) {
    return nullptr;
  }
  return "the sequence claims a shared prefix, so its KV history is not all "
         "its own; release it and re-prefill instead";
}

// The offset of `kind` in `sections`. The table always carries every kind
// (ignis_seq_section_table builds it unconditionally), so a miss is a
// programming error in this file rather than a caller's.
std::uint64_t section_offset(const std::vector<ignis_seq_section> &sections, int32_t kind) {
  for (const ignis_seq_section &section : sections) {
    if (section.kind == kind) {
      return section.offset;
    }
  }
  throw std::logic_error(std::string("state-section table has no ") +
                         ignis_seq_section_name(kind) + " section");
}


// Zero the bytes of `dst` that no section's payload covers: the gap after
// the header and records, and each section's alignment padding.
//
// Only a few hundred bytes in total, but they are part of the blob's extent,
// so a blob written into a reused buffer would otherwise carry whatever the
// previous occupant left in its gaps. Two snapshots of the same sequence
// have to be the same bytes -- the leaf's own round-trip test says so by
// comparing them -- and that has to be true of a recycled host region as
// much as of a fresh one.
void zero_blob_gaps(unsigned char *base, const std::vector<ignis_seq_section> &sections,
                    std::uint64_t total_bytes) {
  std::uint64_t cursor = sizeof(ignis_seq_snapshot_header) +
                         sections.size() * sizeof(ignis_seq_section);
  for (const ignis_seq_section &section : sections) {
    if (section.offset > cursor) {
      std::memset(base + cursor, 0, static_cast<std::size_t>(section.offset - cursor));
    }
    cursor = section.offset + section.bytes;
  }
  if (total_bytes > cursor) {
    std::memset(base + cursor, 0, static_cast<std::size_t>(total_bytes - cursor));
  }
}

// One host<->device copy on the default stream, as a checked error rather
// than the vendored CUDA_CHECK's abort.
void checked_memcpy_async(void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind,
                          const char *what) {
  if (bytes == 0) {
    return;
  }
  const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync(") + what +
                             ") failed: " + cudaGetErrorString(err));
  }
}

// Why `header` cannot be restored into `seq` of `pool`, or an empty string
// if it can.
//
// Every check here is a pure host-side comparison, which is what lets
// ignis_seq_restore run all of them before it writes a single byte: a
// refused restore leaves the target sequence exactly as it was (ADR 0024).
std::string snapshot_refusal(const ignis_seq_pool &pool, const ignis_seq &seq,
                             const ignis_seq_snapshot_header &header,
                             const std::vector<ignis_seq_section> &records,
                             std::uint64_t src_bytes) {
  const auto mismatch = [](const char *field, std::uint64_t blob, std::uint64_t target) {
    return std::string("snapshot ") + field + " is " + std::to_string(blob) + ", this pool's is " +
           std::to_string(target);
  };

  if (header.magic != kIgnisSeqSnapshotMagic) {
    return "buffer is not an ignis sequence snapshot (bad magic)";
  }
  if (header.format_version != kIgnisSeqSnapshotFormatVersion) {
    return "snapshot format version " + std::to_string(header.format_version) +
           ", this leaf writes and accepts " + std::to_string(kIgnisSeqSnapshotFormatVersion);
  }
  if (header.header_bytes != sizeof(ignis_seq_snapshot_header) ||
      header.section_record_bytes != sizeof(ignis_seq_section)) {
    return "snapshot header/record sizes do not match this leaf's";
  }
  ignis_seq_snapshot_header bare = header;
  bare.header_checksum           = 0;
  if (ignis_seq_fnv1a(&bare, sizeof(bare)) != header.header_checksum) {
    return "snapshot header checksum mismatch";
  }
  if (header.total_bytes != src_bytes) {
    return mismatch("size", header.total_bytes, src_bytes);
  }

  // The geometry as one struct, compared whole: `memcmp` decides, and the
  // field names only improve the message, so a geometry field added to the
  // blob is checked here whether or not anyone remembered to name it.
  const ignis_seq_snapshot_geometry target = ignis_seq_snapshot_geometry_of(pool);
  if (const char *field = ignis_seq_snapshot_geometry_names(header.geometry, target)) {
    return std::string("snapshot ") + field + " does not match this pool's";
  }

  // The one requirement on the target beyond matching geometry: it must
  // have room for the history the blob carries. A larger reservation is
  // fine -- a restored sequence keeps generating into its own.
  if (header.kv_page_count > seq.kv.mapped_page_count()) {
    return "snapshot carries " + std::to_string(header.kv_page_count) +
           " KV pages, the target sequence maps " + std::to_string(seq.kv.mapped_page_count());
  }

  // The blob's own layout against the table this leaf would build for it.
  // This is what refuses a blob from a build whose section table differed
  // without the version having been bumped: the version says "same format",
  // the records say otherwise, and the records win.
  const std::vector<ignis_seq_section> expected =
      ignis_seq_section_table(pool, header.kv_page_count);
  if (records.size() != expected.size()) {
    return "snapshot lists " + std::to_string(records.size()) + " state sections, this leaf has " +
           std::to_string(expected.size());
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (records[i].kind != expected[i].kind || records[i].transfer != expected[i].transfer ||
        records[i].offset != expected[i].offset || records[i].bytes != expected[i].bytes) {
      return std::string("snapshot section ") + ignis_seq_section_name(records[i].kind) +
             " does not match this leaf's " + ignis_seq_section_name(expected[i].kind) +
             " section";
    }
  }
  if (ignis_seq_snapshot_bytes(expected) != header.total_bytes) {
    return mismatch("total size", header.total_bytes, ignis_seq_snapshot_bytes(expected));
  }
  return {};
}

} // namespace

// Not in the anonymous namespace above: kernel/src/seq_prefix.cu writes this
// same slot (declared in ignis_seq_internal.h), so the two translation units
// share one `ignis_seq_last_error`.
void ignis_seq_set_last_error(std::string message) {
  g_last_error = std::move(message);
}

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
  // P4-10 (GitHub #126): this sequence holds one reference to its shared
  // prefix. Drop it before the handle goes, so the prefix's pages return to
  // the pool exactly when the last holder -- claimant or publisher's handle
  // -- lets go, and not a moment earlier.
  ignis_seq_prefix_drop_reference(seq->prefix);
  seq->prefix       = nullptr;
  seq->shared_pages = 0;
  delete seq; // ~PagedKVAllocation: unbind the row, return the sequence's own KV pages.
  if (pool != nullptr && slot >= 0) {
    pool->free_slots.push_back(slot);
  }
}

extern "C" int32_t ignis_seq_stats(const struct ignis_seq *seq, struct ignis_seq_stats *out_stats) {
  if (seq == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->slot = seq->slot;
  // P4-10 (GitHub #126): a sequence that claims a shared prefix owns only
  // its tail, but its history -- and so its entitlement, its mapped pages
  // and its capacity -- is the prefix's pages plus its own. Reporting the
  // allocation alone would say a claimant has less context than it can
  // actually address.
  out_stats->page_entitlement = seq->shared_pages + seq->kv.page_entitlement();
  out_stats->mapped_pages     = ignis_seq_logical_page_count(*seq);
  out_stats->token_capacity   = ignis_seq_token_capacity(*seq);
  out_stats->shared_pages     = seq->shared_pages;
  return 0;
}

extern "C" uint32_t ignis_seq_snapshot_format_version(void) {
  return kIgnisSeqSnapshotFormatVersion;
}

extern "C" int32_t ignis_seq_snapshot_size(const struct ignis_seq_pool *pool,
                                            const struct ignis_seq *seq, uint64_t *out_bytes) {
  if (out_bytes != nullptr) {
    *out_bytes = 0;
  }
  if (pool == nullptr || seq == nullptr || out_bytes == nullptr) {
    set_error("ignis_seq_snapshot_size: null argument");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error("ignis_seq_snapshot_size: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = shared_prefix_refusal(*seq)) {
    set_error(std::string("ignis_seq_snapshot_size: ") + refusal);
    return IGNIS_SEQ_ERR_SHARED_PREFIX;
  }
  if (!ignis_seq_at_chunk_boundary(*seq)) {
    set_error("ignis_seq_snapshot_size: sequence slot " + std::to_string(seq->slot) +
              " is mid-chunk (program frontier " + std::to_string(seq->position) +
              "); a snapshot is taken only at a chunk boundary");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }
  try {
    const std::uint32_t pages = ignis_seq_snapshot_page_count(*seq);
    *out_bytes = ignis_seq_snapshot_bytes(ignis_seq_section_table(*pool, pages));
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_seq_snapshot_size: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_snapshot(const struct ignis_seq_pool *pool,
                                       const struct ignis_seq *seq, void *dst,
                                       uint64_t dst_bytes) {
  if (pool == nullptr || seq == nullptr || dst == nullptr) {
    set_error("ignis_seq_snapshot: null argument");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error("ignis_seq_snapshot: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = shared_prefix_refusal(*seq)) {
    set_error(std::string("ignis_seq_snapshot: ") + refusal);
    return IGNIS_SEQ_ERR_SHARED_PREFIX;
  }
  if (!ignis_seq_at_chunk_boundary(*seq)) {
    set_error("ignis_seq_snapshot: sequence slot " + std::to_string(seq->slot) +
              " is mid-chunk (program frontier " + std::to_string(seq->position) +
              "); its state sections are not consistent with one another");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }

  // Every copy below runs on the default stream and is confirmed by this
  // call's own synchronize before it returns. That is sufficient without
  // knowing the model's stream: the program's entry points each synchronize
  // before returning (kernel/src/step.cu), so a caller can only reach this
  // function with the sequence's device work already complete -- which is
  // the same fact the chunk-boundary check above rests on.
  try {
    const std::uint32_t pages = ignis_seq_snapshot_page_count(*seq);
    const std::vector<ignis_seq_section> sections = ignis_seq_section_table(*pool, pages);
    const ignis_seq_snapshot_header header = ignis_seq_snapshot_header_for(*pool, pages, sections);
    if (dst_bytes < header.total_bytes) {
      set_error("ignis_seq_snapshot: destination holds " + std::to_string(dst_bytes) +
                " bytes, this snapshot is " + std::to_string(header.total_bytes));
      return -1;
    }

    auto *base = static_cast<unsigned char *>(dst);
    std::memcpy(base, &header, sizeof(header));
    std::memcpy(base + sizeof(header), sections.data(),
                sections.size() * sizeof(ignis_seq_section));
    zero_blob_gaps(base, sections, header.total_bytes);
    const std::uint64_t recurrent_at = section_offset(sections, IGNIS_SEQ_SECTION_GDN_RECURRENT);

    for (const ignis_seq_section &section : sections) {
      unsigned char *at = base + section.offset;
      switch (section.kind) {
      case IGNIS_SEQ_SECTION_KV_PAGES:
        ninfer::pack_paged_kv_allocation_to_host(seq->kv, pool->kv_pool, pages, at, nullptr);
        break;
      case IGNIS_SEQ_SECTION_GDN_CONV:
        // The vendored state pool moves a slot's conv taps and recurrent
        // matrices in one call into two destinations, so this case writes
        // both sections and the recurrent case below writes none. They stay
        // two rows of the table because they are separately sized and
        // separately classified -- the table describes the state, not the
        // memcpy that happens to move it.
        pool->gdn_pool.pack_slot_to_host(seq->slot, at, base + recurrent_at, nullptr);
        break;
      case IGNIS_SEQ_SECTION_GDN_RECURRENT:
        break;
      case IGNIS_SEQ_SECTION_PENALTY_COUNTS:
        checked_memcpy_async(at, pool->token_counts_for(seq->slot),
                             static_cast<std::size_t>(section.bytes), cudaMemcpyDeviceToHost,
                             "penalty counts");
        break;
      case IGNIS_SEQ_SECTION_PROGRESS: {
        const ignis_seq_progress_image image = ignis_seq_progress_of(*seq);
        std::memcpy(at, &image, sizeof(image));
        break;
      }
      default:
        // ADR 0024's "carried by all three or by none": a section added to
        // the table but not to this switch is a loud failure here rather
        // than a silently unsnapshotted piece of a sequence.
        throw std::logic_error(std::string("state section ") +
                               ignis_seq_section_name(section.kind) +
                               " has no snapshot implementation");
      }
    }

    const cudaError_t err = cudaStreamSynchronize(nullptr);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_seq_snapshot: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_seq_snapshot: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_restore(struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                      const void *src, uint64_t src_bytes) {
  if (pool == nullptr || seq == nullptr || src == nullptr) {
    set_error("ignis_seq_restore: null argument");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error("ignis_seq_restore: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = shared_prefix_refusal(*seq)) {
    set_error(std::string("ignis_seq_restore: ") + refusal +
              "; the target sequence is unchanged");
    return IGNIS_SEQ_ERR_SHARED_PREFIX;
  }
  if (src_bytes < sizeof(ignis_seq_snapshot_header)) {
    set_error("ignis_seq_restore: " + std::to_string(src_bytes) +
              " bytes is smaller than a snapshot header");
    return IGNIS_SEQ_ERR_BAD_SNAPSHOT;
  }

  // Read the header and the records out of the caller's buffer before
  // believing any of it: the buffer is host memory of unknown provenance
  // and unknown alignment, so nothing is accessed in place.
  const auto *base = static_cast<const unsigned char *>(src);
  ignis_seq_snapshot_header header{};
  std::memcpy(&header, base, sizeof(header));

  // Read the records only if the header's own account of them fits inside
  // the buffer the caller actually passed -- that bound comes from
  // `src_bytes`, a real allocation, so a header claiming a preposterous
  // section count allocates nothing here. Left empty otherwise, which
  // `snapshot_refusal` then refuses by count.
  std::vector<ignis_seq_section> records;
  if (header.section_record_bytes == sizeof(ignis_seq_section) && header.section_count != 0) {
    const std::uint64_t records_end =
        sizeof(ignis_seq_snapshot_header) +
        static_cast<std::uint64_t>(header.section_count) * sizeof(ignis_seq_section);
    if (records_end <= src_bytes) {
      records.resize(header.section_count);
      std::memcpy(records.data(), base + sizeof(ignis_seq_snapshot_header),
                  records.size() * sizeof(ignis_seq_section));
    }
  }

  try {
    const std::string refusal = snapshot_refusal(*pool, *seq, header, records, src_bytes);
    if (!refusal.empty()) {
      set_error("ignis_seq_restore: " + refusal + "; the target sequence is unchanged");
      return IGNIS_SEQ_ERR_BAD_SNAPSHOT;
    }

    const std::uint64_t recurrent_at = section_offset(records, IGNIS_SEQ_SECTION_GDN_RECURRENT);
    ignis_seq_progress_image image{};
    for (const ignis_seq_section &section : records) {
      const unsigned char *at = base + section.offset;
      switch (section.kind) {
      case IGNIS_SEQ_SECTION_KV_PAGES:
        ninfer::unpack_paged_kv_allocation_from_host(seq->kv, pool->kv_pool, at,
                                                     header.kv_page_count, header.kv_page_count,
                                                     nullptr);
        break;
      case IGNIS_SEQ_SECTION_GDN_CONV:
        // Packed together with the recurrent section (see ignis_seq_snapshot).
        pool->gdn_pool.unpack_slot_from_host(seq->slot, at, base + recurrent_at, nullptr);
        break;
      case IGNIS_SEQ_SECTION_GDN_RECURRENT:
        break;
      case IGNIS_SEQ_SECTION_PENALTY_COUNTS:
        checked_memcpy_async(pool->token_counts_for(seq->slot), at,
                             static_cast<std::size_t>(section.bytes), cudaMemcpyHostToDevice,
                             "penalty counts");
        break;
      case IGNIS_SEQ_SECTION_PROGRESS:
        std::memcpy(&image, at, sizeof(image));
        break;
      default:
        throw std::logic_error(std::string("state section ") +
                               ignis_seq_section_name(section.kind) +
                               " has no restore implementation");
      }
    }

    const cudaError_t err = cudaStreamSynchronize(nullptr);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_seq_restore: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    // The progress scalars last, once the synchronize above has confirmed
    // every section actually landed -- the same ordering the chunk loop
    // uses for its own position advance (kernel/src/step.cu).
    ignis_seq_apply_progress(*seq, image);
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_seq_restore: ") + e.what());
    return -1;
  }
}

extern "C" const char *ignis_seq_last_error(void) {
  return g_last_error.c_str();
}
