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
// P4-07 (GitHub #125) adds the transfer's own host-side memory: a pinned
// (page-locked) alloc/free pair the host KV-RAM tier calls to get the
// region ignis_seq_snapshot writes into and ignis_seq_restore reads from.
// GitHub #213 (ADR 0030) moves the page-locking to the load: the pair now
// places blobs in one process-wide vendored HostPinnedArena of
// --kv-host-pool-bytes, pinned once and held for the life of the process,
// so nothing calls cudaHostAlloc while serving and Windows reports a fixed
// shared-GPU-memory figure for the process.
//
// Style follows model.cu: explicit pointers + sizes, int32 return codes (0
// = ok, -1 = error, and an IGNIS_SEQ_ERR_* code per refusal that is not a
// caller mistake -- _NOT_AT_BOUNDARY, _BAD_SNAPSHOT and _SHARED_PREFIX for
// state transfer, _NO_HOST_ROOM for a KV-RAM arena with no span long enough),
// no C++ types across the boundary.

#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include "core/arena.h"
#include "ninfer/ops/gqa_attention.h"
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
static_assert(kIgnisHqSinkKeys == static_cast<int32_t>(ninfer::ops::kGqaHqSinkKeys),
              "kIgnisHqSinkKeys has drifted from the vendored kGqaHqSinkKeys");
static_assert(kIgnisHqRecentKeys == static_cast<int32_t>(ninfer::ops::kGqaHqRecentKeys),
              "kIgnisHqRecentKeys has drifted from the vendored kGqaHqRecentKeys");

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
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

// Zero `slot`'s lane of the drafter window and of its checkpoint (P5-03,
// GitHub #152), so a re-allocated slot never attends over another request's
// context. A no-op on a pool without the drafter.
void zero_dflash2_lane(ignis_seq_pool &pool, std::int32_t slot) {
  if (!pool.has_dflash2()) {
    return;
  }
  for (const ninfer::CyclicKVCache *cache : {pool.dflash2_window.get(),
                                             pool.dflash2_checkpoint.get()}) {
    for (std::uint32_t layer = 0; layer < cache->layer_count(); ++layer) {
      const ninfer::CyclicKVCacheLayerView view = cache->layer_view(layer);
      for (const ninfer::Tensor *plane : {&view.k, &view.v}) {
        const ninfer::Tensor lane = plane->slice(3, slot, 1);
        const cudaError_t err     = cudaMemset(lane.data, 0, lane.bytes());
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemset(dflash2 window lane) failed: ") +
                                   cudaGetErrorString(err));
        }
      }
    }
  }
}

// ---- state transfer (P4-06, GitHub #124, ADR 0024) -----------------------

// Why a restore target cannot already hold shared history, or nullptr if it
// can.
//
// A snapshot of a sequence holding a shared prefix materializes the prefix's
// pages into the blob (GitHub #190), so taking one needs no refusal. Writing
// a blob back is different: the target's leading pages would be the prefix's,
// refcounted and read by every other claimant, and a restore would overwrite
// their history. The way back on for such a sequence is a fresh one.
const char *shared_prefix_refusal(const ignis_seq &seq) {
  if (seq.prefix == nullptr) {
    return nullptr;
  }
  return "the target sequence claims a shared prefix, whose pages a restore would "
         "overwrite for every other claimant; restore into a fresh sequence";
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

} // namespace

// GitHub #257: slot `slot`'s residual window against `host`, laid out as the
// IGNIS_SEQ_SECTION_HQ_RESIDUAL payload -- every GQA layer's K plane, then
// every layer's V plane, then the ring words. One 2D copy per role: a slot's
// plane sits at the same offset of every layer's run, one layer pitch apart.
// On the default stream, like every other section copy here.
void ignis_seq_copy_hq_residual(const ignis_seq_pool &pool, std::int32_t slot, void *host,
                                cudaMemcpyKind kind) {
  const bool to_host        = kind == cudaMemcpyDeviceToHost;
  const std::size_t plane   = static_cast<std::size_t>(pool.hq_residual_plane_bytes());
  const std::size_t pitch   = pool.hq_residual_layer_pitch();
  auto *image               = static_cast<unsigned char *>(host);
  for (const bool role_v : {false, true}) {
    void *device        = pool.hq_residual_plane(role_v, 0, slot);
    unsigned char *here = image + (role_v ? kIgnisGqaLayerCount * plane : 0);
    const cudaError_t err =
        to_host ? cudaMemcpy2DAsync(here, plane, device, pitch, plane, kIgnisGqaLayerCount, kind, nullptr)
                : cudaMemcpy2DAsync(device, pitch, here, plane, plane, kIgnisGqaLayerCount, kind, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy2DAsync(hq residual ") +
                               (role_v ? "v" : "k") + ") failed: " + cudaGetErrorString(err));
    }
  }
  unsigned char *ring    = image + 2 * kIgnisGqaLayerCount * plane;
  const std::size_t ring_bytes = kIgnisHqRingWords * sizeof(std::uint32_t);
  checked_memcpy_async(to_host ? static_cast<void *>(ring) : static_cast<void *>(pool.hq_ring_words(slot)),
                       to_host ? static_cast<const void *>(pool.hq_ring_words(slot)) : ring, ring_bytes,
                       kind, "hq ring words");
}

void ignis_seq_zero_hq_residual(ignis_seq_pool &pool, std::int32_t slot) {
  if (!pool.has_hq_residual()) {
    return;
  }
  const std::size_t plane = static_cast<std::size_t>(pool.hq_residual_plane_bytes());
  const std::size_t pitch = pool.hq_residual_layer_pitch();
  for (const bool role_v : {false, true}) {
    const cudaError_t err =
        cudaMemset2D(pool.hq_residual_plane(role_v, 0, slot), pitch, 0, plane, kIgnisGqaLayerCount);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemset2D(hq residual) failed: ") +
                               cudaGetErrorString(err));
    }
  }
  const cudaError_t err =
      cudaMemset(pool.hq_ring_words(slot), 0, kIgnisHqRingWords * sizeof(std::uint32_t));
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemset(hq ring words) failed: ") +
                             cudaGetErrorString(err));
  }
}

namespace {

// The KV_PAGES payload of a snapshot of `seq`: its first `count` logical
// pages.
//
// A sequence that owns its whole history keeps the vendored pack, exactly as
// before GitHub #190. One holding a shared prefix is materialized: the
// prefix chain's pages first, then its own, in block-table order -- the same
// bytes the vendored pack would have written had the sequence owned them.
void pack_logical_pages(const ignis_seq_pool &pool, const ignis_seq &seq, std::uint32_t count,
                        void *dst) {
  if (seq.prefix == nullptr) {
    ninfer::pack_paged_kv_allocation_to_host(seq.kv, pool.kv_pool, count, dst, nullptr);
    return;
  }
  std::vector<std::int32_t> pages = ignis_seq_prefix_chain_page_ids(seq.prefix);
  const auto own = seq.kv.page_ids();
  pages.insert(pages.end(), own.begin(), own.end());
  if (count > pages.size()) {
    throw std::logic_error("snapshot extent exceeds the sequence's logical pages");
  }
  pages.resize(count);
  ignis_seq_pack_pages_to_host(pool, pages, dst);
}

// Write pool slot `slot`'s payload of the device-resident CLONE section
// `section` into a blob at `base`, laid out by `sections`. False for a
// section this does not write -- KV pages and progress, which the caller
// writes for itself.
//
// Shared by a live sequence's snapshot (its lane) and a retained object's
// materialized blob (its retained slot, GitHub #215), so the two cannot lay a
// section out differently.
bool pack_slot_section_to_host(const ignis_seq_pool &pool, std::int32_t slot,
                               const ignis_seq_section &section,
                               const std::vector<ignis_seq_section> &sections,
                               unsigned char *base) {
  unsigned char *at = base + section.offset;
  switch (section.kind) {
  case IGNIS_SEQ_SECTION_GDN_CONV:
    // The vendored state pool moves a slot's conv taps and recurrent
    // matrices in one call into two destinations, so this case writes both
    // sections and the recurrent case below writes none. They stay two rows
    // of the table because they are separately sized and separately
    // classified -- the table describes the state, not the memcpy that
    // happens to move it.
    pool.gdn_pool.pack_slot_to_host(
        slot, at, base + ignis_seq_section_offset(sections, IGNIS_SEQ_SECTION_GDN_RECURRENT),
        nullptr);
    return true;
  case IGNIS_SEQ_SECTION_GDN_RECURRENT:
    return true;
  case IGNIS_SEQ_SECTION_PENALTY_COUNTS:
    checked_memcpy_async(at, pool.token_counts_for(slot), static_cast<std::size_t>(section.bytes),
                         cudaMemcpyDeviceToHost, "penalty counts");
    return true;
  case IGNIS_SEQ_SECTION_DFLASH_WINDOW:
    pool.dflash2_window->copy_lane_to_host(slot, at, nullptr);
    return true;
  case IGNIS_SEQ_SECTION_DFLASH_CHECKPOINT:
    pool.dflash2_checkpoint->copy_lane_to_host(slot, at, nullptr);
    return true;
  case IGNIS_SEQ_SECTION_HQ_RESIDUAL:
    ignis_seq_copy_hq_residual(pool, slot, at, cudaMemcpyDeviceToHost);
    return true;
  default:
    return false;
  }
}

} // namespace

// Not in the anonymous namespace: kernel/src/seq_checkpoint.cu materializes
// a checkpoint into the same blob layout (GitHub #190), and one definition of
// that layout is what keeps a spilled checkpoint restorable by
// ignis_seq_restore.

std::uint64_t ignis_seq_section_offset(const std::vector<ignis_seq_section> &sections,
                                       int32_t kind) {
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
void ignis_seq_zero_blob_gaps(unsigned char *base, const std::vector<ignis_seq_section> &sections,
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

void ignis_seq_copy_to_host(void *dst, const void *src, std::size_t bytes, const char *what) {
  checked_memcpy_async(dst, src, bytes, cudaMemcpyDeviceToHost, what);
}

std::vector<std::int32_t> ignis_seq_prefix_chain_page_ids(const ignis_seq_prefix *head) {
  std::vector<const ignis_seq_prefix *> chain;
  for (const ignis_seq_prefix *at = head; at != nullptr; at = at->parent) {
    chain.push_back(at);
  }
  std::vector<std::int32_t> pages;
  for (auto at = chain.rbegin(); at != chain.rend(); ++at) {
    const auto ids = (*at)->kv.page_ids();
    pages.insert(pages.end(), ids.begin(), ids.end());
  }
  return pages;
}

void ignis_seq_pack_pages_to_host(const ignis_seq_pool &pool,
                                  const std::vector<std::int32_t> &pages, void *dst) {
  if (pool.kv_pool.plane_order() != ninfer::PagedKVPlaneOrder::PageMajor) {
    throw std::logic_error("materialized snapshots require a PageMajor KV pool");
  }
  auto *out = static_cast<unsigned char *>(dst);
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    const std::size_t bytes     = static_cast<std::size_t>(plane.nb[3]);
    const auto *base            = static_cast<const unsigned char *>(plane.data);
    // One copy per run of consecutive pages, the way the vendored pack
    // coalesces them: a chain and a sequence's own allocation are each
    // usually one run, so a materialized blob costs a handful of copies per
    // plane rather than one per page.
    std::size_t begin = 0;
    while (begin < pages.size()) {
      std::size_t end = begin + 1;
      while (end < pages.size() && pages[end] == pages[end - 1] + 1) {
        ++end;
      }
      checked_memcpy_async(out, base + static_cast<std::int64_t>(pages[begin]) * plane.nb[3],
                           (end - begin) * bytes, cudaMemcpyDeviceToHost, "materialized KV pages");
      out += (end - begin) * bytes;
      begin = end;
    }
  }
}

std::uint64_t ignis_seq_materialized_blob_bytes(const ignis_seq_pool &pool, std::uint32_t pages) {
  return ignis_seq_snapshot_bytes(ignis_seq_section_table(pool, pages));
}

void ignis_seq_write_materialized_blob(const ignis_seq_pool &pool, const ignis_seq_prefix *chain,
                                       std::int32_t tail_page, std::uint32_t retained_slot,
                                       const ignis_seq_progress_image &progress,
                                       std::uint32_t pages, void *dst, std::uint64_t dst_bytes) {
  const std::vector<ignis_seq_section> sections = ignis_seq_section_table(pool, pages);
  const ignis_seq_snapshot_header header = ignis_seq_snapshot_header_for(pool, pages, sections);
  if (dst_bytes < header.total_bytes) {
    throw std::invalid_argument("destination holds " + std::to_string(dst_bytes) +
                                " bytes, this blob is " + std::to_string(header.total_bytes));
  }
  if (retained_slot >= pool.retained_slot_count) {
    throw std::logic_error("materialization names retained slot " + std::to_string(retained_slot) +
                           " of a pool holding " + std::to_string(pool.retained_slot_count));
  }
  // The chain's whole pages, then the checkpoint's own page -- the layout a
  // sequence standing at the same point would have packed.
  std::vector<std::int32_t> history = ignis_seq_prefix_chain_page_ids(chain);
  if (tail_page >= 0) {
    history.push_back(tail_page);
  }
  if (pages != history.size()) {
    throw std::logic_error("materialization extent does not match the chain and its tail");
  }
  auto *base = static_cast<unsigned char *>(dst);
  std::memcpy(base, &header, sizeof(header));
  std::memcpy(base + sizeof(header), sections.data(), sections.size() * sizeof(ignis_seq_section));
  ignis_seq_zero_blob_gaps(base, sections, header.total_bytes);
  const std::int32_t slot = ignis_seq_retained_pool_slot(pool, retained_slot);
  for (const ignis_seq_section &section : sections) {
    unsigned char *at = base + section.offset;
    switch (section.kind) {
    case IGNIS_SEQ_SECTION_KV_PAGES:
      ignis_seq_pack_pages_to_host(pool, history, at);
      break;
    case IGNIS_SEQ_SECTION_PROGRESS:
      std::memcpy(at, &progress, sizeof(progress));
      break;
    default:
      // Every other section is device-resident state, held in the retained
      // slot exactly as a lane holds it.
      if (!pack_slot_section_to_host(pool, slot, section, sections, base)) {
        throw std::logic_error(std::string("state section ") +
                               ignis_seq_section_name(section.kind) +
                               " has no materialization implementation");
      }
      break;
    }
  }
  const cudaError_t err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize after materializing failed: ") +
                             cudaGetErrorString(err));
  }
}

namespace {

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

namespace {

// The spec checks `ignis_seq_pool_create` and `ignis_seq_pool_plan` share
// (GitHub #210), naming the entry point `fn` in the message. False (error
// set) on a refusal.
bool validate_pool_spec(const struct ignis_seq_pool_spec &spec, const std::string &fn) {
  if (!positive(spec.num_kv_heads) || !positive(spec.head_dim) ||
      !positive(spec.kv_page_group_count) || !positive(spec.max_context_tokens) ||
      !positive(spec.slot_count) || !positive(spec.gdn_num_layers) ||
      !positive(spec.gdn_conv_channels) || !positive(spec.gdn_value_heads) ||
      !positive(spec.gdn_head_dim) || !positive(spec.vocab)) {
    set_error(fn + ": every geometry field must be positive");
    return false;
  }
  if (!known_kv_format(spec.kv_format)) {
    set_error(fn + ": kv_format " + std::to_string(spec.kv_format) +
              " is not an ignis_kv_format");
    return false;
  }
  // The codec's row budget is defined for a 256-dimension row only
  // (kHqHeadDim); a pool of any other head_dim would plan planes the hq
  // append path cannot write.
  if (spec.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B &&
      spec.head_dim != static_cast<std::uint32_t>(kIgnisHqHeadDim)) {
    set_error(fn + ": hq-e8-2b KV requires head_dim " + std::to_string(kIgnisHqHeadDim) +
              ", got " + std::to_string(spec.head_dim));
    return false;
  }
  if (spec.speculative_backend != IGNIS_SPECULATIVE_NONE &&
      spec.speculative_backend != IGNIS_SPECULATIVE_DFLASH2 &&
      spec.speculative_backend != IGNIS_SPECULATIVE_VERIFY_ONLY) {
    set_error(fn + ": speculative_backend " + std::to_string(spec.speculative_backend) +
              " is not an ignis_speculative_backend");
    return false;
  }
  return true;
}

// Every layout a pool is built from, and the bytes each arena takes: planned
// once, allocated by `ignis_seq_pool_create`, reported by
// `ignis_seq_pool_plan` (GitHub #210). Throws on a planning failure.
struct PoolLayout {
  ninfer::PagedKVPoolLayout kv_layout;
  std::size_t kv_bytes = 0;
  std::uint64_t kv_page_bytes = 0;
  ninfer::LinearAttentionStatePoolLayout gdn_layout;
  std::size_t gdn_bytes = 0;
  std::size_t sampling_counts_bytes = 0;
  bool dflash2 = false;
  ninfer::CyclicKVCacheLayout dflash2_window_layout;
  ninfer::CyclicKVCacheLayout dflash2_checkpoint_layout;
  std::size_t dflash2_bytes = 0;
  /* One slot's share of the GDN arena, the penalty counts and the drafter
   * arena (GitHub #211). */
  std::uint64_t slot_state_bytes = 0;
  /* The hq residual window (GitHub #257): every state slot's K planes, then
   * their V planes, then their ring words, in one buffer. 0 on a BF16 pool. */
  std::size_t hq_residual_bytes = 0;
  std::size_t hq_residual_plane_bytes = 0;
};

// Every slot the state arenas hold: the lanes, then the retained slots past
// them (GitHub #211). The KV block tables hold the lanes' rows only.
std::uint32_t state_slot_count(const struct ignis_seq_pool_spec &spec) {
  return spec.slot_count + spec.retained_slot_count;
}

std::uint64_t per_slot_bytes(const std::vector<ninfer::LayoutRegion> &regions,
                             std::uint32_t slots);
std::uint64_t cyclic_lane_bytes(const ninfer::CyclicKVCacheLayout &layout);

PoolLayout plan_pool_layout(const struct ignis_seq_pool_spec &spec) {
  PoolLayout out;
  const auto logical_page_capacity = ninfer::pages_for_tokens(spec.max_context_tokens);
  const std::uint32_t state_slots  = state_slot_count(spec);

  ninfer::LayoutBuilder kv_builder;
  ninfer::PagedKVPoolSpec kv_spec;
  kv_spec.page_group_count      = spec.kv_page_group_count;
  kv_spec.logical_page_capacity = logical_page_capacity;
  kv_spec.table_rows            = static_cast<std::int32_t>(spec.slot_count);
  kv_spec.plane_order           = ninfer::PagedKVPlaneOrder::PageMajor;
  // One K/V plane run per full-attention layer. The GQA layer program
  // selects its own run (`ignis_kv_plane_index`), so a layer's K/V history
  // never aliases another layer's pages.
  kv_spec.planes.reserve(static_cast<std::size_t>(ignis_kv_planes_per_layer(spec.kv_format)) *
                         kIgnisGqaLayerCount);
  for (int32_t layer = 0; layer < kIgnisGqaLayerCount; ++layer) {
    push_layer_planes(kv_spec, spec.kv_format, spec.num_kv_heads, spec.head_dim);
  }
  out.kv_layout     = ninfer::plan_paged_kv_pool(kv_builder, kv_spec);
  out.kv_bytes      = kv_builder.finish(256);
  out.kv_page_bytes = static_cast<std::uint64_t>(out.kv_layout.payload_bytes()) /
                      spec.kv_page_group_count;

  ninfer::LayoutBuilder gdn_builder;
  ninfer::LinearAttentionStatePoolSpec gdn_spec;
  gdn_spec.layers         = spec.gdn_num_layers;
  gdn_spec.conv_channels  = static_cast<std::int32_t>(spec.gdn_conv_channels);
  // The conv STATE (history) is width-1 = 3 taps, not the kernel width: the
  // conv_snapshot op's slot stride is `channels * 3` and the reference's
  // state pool requires conv_width == 3 (GitHub #58, GDN layer).
  gdn_spec.conv_width     = kIgnisGdnConvStateWidth;
  gdn_spec.value_heads    = static_cast<std::int32_t>(spec.gdn_value_heads);
  gdn_spec.value_head_dim = static_cast<std::int32_t>(spec.gdn_head_dim);
  gdn_spec.key_head_dim   = static_cast<std::int32_t>(spec.gdn_head_dim);
  gdn_spec.slot_count     = static_cast<std::int32_t>(state_slots);
  gdn_spec.conv_dtype     = ninfer::DType::BF16;
  out.gdn_layout = ninfer::plan_linear_attention_state_pool(gdn_builder, gdn_spec);
  out.gdn_bytes  = gdn_builder.finish(256);

  // P3-03 (GitHub #99): one int32 penalty count per vocab entry, per slot.
  out.sampling_counts_bytes = static_cast<std::size_t>(state_slots) *
                              static_cast<std::size_t>(spec.vocab) * sizeof(std::int32_t);

  // P5-03 (GitHub #152): the drafter's window and its rewrite checkpoint,
  // one cyclic lane per slot, planned by the vendored cache itself so the
  // lane layout the drafter's kernels address is the one sized here.
  if (spec.speculative_backend == IGNIS_SPECULATIVE_DFLASH2) {
    ninfer::LayoutBuilder dflash2_builder;
    const auto lanes = static_cast<std::int32_t>(state_slots);
    out.dflash2 = true;
    out.dflash2_window_layout = ninfer::plan_cyclic_kv_cache(
        dflash2_builder, kIgnisDflash2Layers, kIgnisDflash2WindowTokens, kIgnisDflash2KvHeads,
        kIgnisDflash2HeadDim, lanes);
    out.dflash2_checkpoint_layout = ninfer::plan_cyclic_kv_cache(
        dflash2_builder, kIgnisDflash2Layers, kIgnisDflash2WindowTokens, kIgnisDflash2KvHeads,
        kIgnisDflash2HeadDim, lanes);
    out.dflash2_bytes = dflash2_builder.finish(256);
  }
  out.slot_state_bytes = per_slot_bytes(out.gdn_layout.conv, state_slots) +
                         per_slot_bytes(out.gdn_layout.recurrent, state_slots) +
                         static_cast<std::uint64_t>(spec.vocab) * sizeof(std::int32_t);
  if (out.dflash2) {
    out.slot_state_bytes += cyclic_lane_bytes(out.dflash2_window_layout) +
                            cyclic_lane_bytes(out.dflash2_checkpoint_layout);
  }
  // GitHub #257 (spec runtime/06): the hq residual window, for every state
  // slot. Reserved here, at load, like every other line of the plan (ADR
  // 0030): its bytes come off the KV pool's budget, never out of a request.
  // Each (layer, slot) plane is 1.06 MiB at 4 KV heads, a multiple of the
  // 256-byte alignment, so the three regions pack without padding.
  if (spec.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    out.hq_residual_plane_bytes = static_cast<std::size_t>(kIgnisHqHeadDim) * spec.num_kv_heads *
                                  kIgnisHqResidualRows * sizeof(std::uint16_t);
    const std::size_t planes = static_cast<std::size_t>(kIgnisGqaLayerCount) * state_slots;
    out.hq_residual_bytes = 2 * planes * out.hq_residual_plane_bytes +
                            static_cast<std::size_t>(state_slots) * kIgnisHqRingWords *
                                sizeof(std::uint32_t);
  }
  return out;
}

// The state arenas' bytes the lanes hold: every arena less the retained
// slots' share (GitHub #210, #211).
std::uint64_t lane_state_bytes_of(const PoolLayout &layout,
                                  const struct ignis_seq_pool_spec &spec) {
  return layout.gdn_bytes + layout.sampling_counts_bytes + layout.dflash2_bytes -
         spec.retained_slot_count * layout.slot_state_bytes;
}

// One slot's share of every region in `regions`.
std::uint64_t per_slot_bytes(const std::vector<ninfer::LayoutRegion> &regions,
                             std::uint32_t slots) {
  std::uint64_t bytes = 0;
  for (const auto &region : regions) {
    bytes += region.bytes / slots;
  }
  return bytes;
}

std::uint64_t cyclic_lane_bytes(const ninfer::CyclicKVCacheLayout &layout) {
  std::uint64_t bytes = 0;
  const auto lanes = static_cast<std::uint64_t>(layout.lane_capacity);
  for (std::size_t layer = 0; layer < layout.k.size(); ++layer) {
    bytes += layout.k[layer].region.bytes / lanes + layout.v[layer].region.bytes / lanes;
  }
  return bytes;
}

} // namespace

extern "C" int32_t ignis_seq_pool_plan(const struct ignis_seq_pool_spec *spec,
                                        struct ignis_seq_pool_plan *out) {
  if (out != nullptr) {
    *out = {};
  }
  if (spec == nullptr || out == nullptr) {
    set_error("ignis_seq_pool_plan: null argument");
    return -1;
  }
  if (!validate_pool_spec(*spec, "ignis_seq_pool_plan")) {
    return -1;
  }
  try {
    const PoolLayout layout = plan_pool_layout(*spec);
    out->kv_bytes = layout.kv_bytes;
    out->lane_state_bytes = lane_state_bytes_of(layout, *spec);
    out->slot_state_bytes = layout.slot_state_bytes;
    out->retained_state_bytes = spec->retained_slot_count * layout.slot_state_bytes;
    out->hq_residual_bytes = layout.hq_residual_bytes;
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_seq_pool_plan: ") + e.what());
    return -1;
  }
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
  if (!validate_pool_spec(*spec, "ignis_seq_pool_create")) {
    return -1;
  }

  try {
    const PoolLayout layout = plan_pool_layout(*spec);
    auto pool = std::make_unique<ignis_seq_pool>(layout.kv_bytes, layout.kv_layout,
                                                 layout.gdn_bytes, layout.gdn_layout,
                                                 layout.sampling_counts_bytes,
                                                 static_cast<std::int32_t>(spec->vocab));
    pool->kv_page_bytes   = layout.kv_page_bytes;
    pool->kv_format       = spec->kv_format;
    pool->kv_head_dim     = static_cast<std::int32_t>(spec->head_dim);
    pool->kv_num_kv_heads = static_cast<std::int32_t>(spec->num_kv_heads);

    if (layout.dflash2) {
      pool->dflash2_arena = std::make_unique<ninfer::DeviceArena>(layout.dflash2_bytes);
      const ninfer::DeviceSpan backing{pool->dflash2_arena->base(),
                                       pool->dflash2_arena->capacity()};
      pool->dflash2_window =
          std::make_unique<ninfer::CyclicKVCache>(backing, layout.dflash2_window_layout);
      pool->dflash2_checkpoint =
          std::make_unique<ninfer::CyclicKVCache>(backing, layout.dflash2_checkpoint_layout);
    }
    // Named on every pool, VERIFY_ONLY included (P5-04, GitHub #153): that
    // backend owns no per-slot state, but the program entry points still pair
    // a pool with a model of the same backend, one rule for every backend.
    pool->speculative_backend = spec->speculative_backend;
    pool->retained_slot_count = spec->retained_slot_count;
    pool->slot_state_bytes    = layout.slot_state_bytes;
    pool->retained_held.assign(spec->retained_slot_count, false);

    if (layout.hq_residual_bytes != 0) {
      const std::size_t planes =
          static_cast<std::size_t>(kIgnisGqaLayerCount) * state_slot_count(*spec);
      pool->hq_residual = std::make_unique<ninfer::DeviceBuffer>(layout.hq_residual_bytes);
      // Zeroed once here and per slot at every alloc: no bit is set and no
      // side row holds anything until an append writes it.
      pool->hq_residual->fill(0);
      auto *base              = static_cast<unsigned char *>(pool->hq_residual->p);
      pool->hq_residual_k     = base;
      pool->hq_residual_v     = base + planes * layout.hq_residual_plane_bytes;
      pool->hq_ring           = reinterpret_cast<std::uint32_t *>(
          base + 2 * planes * layout.hq_residual_plane_bytes);
      pool->hq_residual_slots = static_cast<std::int32_t>(state_slot_count(*spec));
      if (pool->hq_residual_plane_bytes() != layout.hq_residual_plane_bytes) {
        throw std::logic_error("the hq residual plane the pool addresses is not the one it planned");
      }
    }

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
  // GitHub #210: the device bytes the pool holds, read off its arenas, for
  // the load to check against its VRAM plan.
  out_stats->kv_arena_bytes = pool->kv_arena.capacity();
  out_stats->retained_slot_count  = pool->retained_slot_count;
  out_stats->slot_state_bytes     = pool->slot_state_bytes;
  out_stats->retained_state_bytes = pool->retained_slot_count * pool->slot_state_bytes;
  out_stats->lane_state_bytes = pool->gdn_arena.capacity() + pool->sampling_counts.bytes +
                                (pool->has_dflash2() ? pool->dflash2_arena->capacity() : 0) -
                                out_stats->retained_state_bytes;
  out_stats->hq_residual_bytes = pool->has_hq_residual() ? pool->hq_residual->bytes : 0;
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
    zero_dflash2_lane(*pool, slot);
    // GitHub #257: nothing of the slot's previous occupant stays readable as
    // an exact row -- neither a ring bit, which names no position, nor a sink
    // row, which the kernels read with no bit at all.
    ignis_seq_zero_hq_residual(*pool, slot);
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
  out_stats->position         = seq->position;
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
    ignis_seq_zero_blob_gaps(base, sections, header.total_bytes);

    for (const ignis_seq_section &section : sections) {
      unsigned char *at = base + section.offset;
      switch (section.kind) {
      case IGNIS_SEQ_SECTION_KV_PAGES:
        pack_logical_pages(*pool, *seq, pages, at);
        break;
      case IGNIS_SEQ_SECTION_PROGRESS: {
        const ignis_seq_progress_image image = ignis_seq_progress_of(*seq);
        std::memcpy(at, &image, sizeof(image));
        break;
      }
      default:
        // ADR 0024's "carried by all three or by none": a section added to
        // the table but not to the slot packer is a loud failure here rather
        // than a silently unsnapshotted piece of a sequence.
        if (!pack_slot_section_to_host(*pool, seq->slot, section, sections, base)) {
          throw std::logic_error(std::string("state section ") +
                                 ignis_seq_section_name(section.kind) +
                                 " has no snapshot implementation");
        }
        break;
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

    const std::uint64_t recurrent_at = ignis_seq_section_offset(records, IGNIS_SEQ_SECTION_GDN_RECURRENT);
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
      case IGNIS_SEQ_SECTION_DFLASH_WINDOW:
        pool->dflash2_window->copy_lane_from_host(at, seq->slot, nullptr);
        break;
      case IGNIS_SEQ_SECTION_DFLASH_CHECKPOINT:
        pool->dflash2_checkpoint->copy_lane_from_host(at, seq->slot, nullptr);
        break;
      case IGNIS_SEQ_SECTION_HQ_RESIDUAL:
        // GitHub #257: the blob's own rows and bits over the target slot's,
        // whatever sequence held that slot before (the reference's
        // program_impl.h:1126 records the bug a restore without this is).
        ignis_seq_copy_hq_residual(*pool, seq->slot, const_cast<unsigned char *>(at),
                                   cudaMemcpyHostToDevice);
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

// ---- retained slots (GitHub #211, ADR 0030) ------------------------------

// Walks the CLONE sections a retained slot carries
// (`ignis_seq_prefix_clone_layout`), so a section added to the table without a
// case here throws rather than being silently left behind -- ADR 0024's
// "carried by all or by none". Since GitHub #215 this is the one
// device-to-device move of a sequence's mutable state: a prefix publish and a
// checkpoint capture copy a lane into a retained slot, and a claim copies it
// back.
void ignis_seq_copy_slot_state(ignis_seq_pool &pool, std::int32_t src, std::int32_t dst) {
  const auto copy = [](void *to, const void *from, std::size_t bytes, const char *what) {
    const cudaError_t err = cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToDevice, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync(") + what +
                               ", slot to slot) failed: " + cudaGetErrorString(err));
    }
  };
  // One 2D copy per GDN section: a slot's state is the same region of every
  // layer's tensor, at the pool's own layer pitch on both sides. It matters:
  // at the 27B geometry the GDN sections are 96 layer-slots, and 96 separate
  // copies of 3 MiB are launch-bound rather than bandwidth-bound (measured:
  // 1.51 ms against ~0.25 ms for two 2D copies --
  // docs/findings/2026-09-12-device-prefix-clone-cost.md). The pitch is read
  // off the pool's own layer tensors, and a single-layer pool has none to
  // read, which is why it takes the linear path.
  const auto copy_gdn = [&](bool recurrent, const char *what) {
    const std::uint32_t layers = pool.gdn_pool.layer_count();
    const std::size_t per_layer =
        recurrent ? pool.gdn_pool.recurrent_slot_bytes() : pool.gdn_pool.conv_slot_bytes();
    const std::vector<ninfer::Tensor> &planes =
        recurrent ? pool.gdn_pool.recurrent : pool.gdn_pool.conv;
    const auto slot_at = [&](std::int32_t slot) {
      return recurrent ? pool.gdn_pool.recurrent_slot(0, slot).data
                       : pool.gdn_pool.conv_slot(0, slot).data;
    };
    if (layers <= 1) {
      copy(slot_at(dst), slot_at(src), per_layer, what);
      return;
    }
    const std::size_t pitch = static_cast<std::size_t>(
        static_cast<const unsigned char *>(planes[1].data) -
        static_cast<const unsigned char *>(planes[0].data));
    const cudaError_t err = cudaMemcpy2DAsync(slot_at(dst), pitch, slot_at(src), pitch, per_layer,
                                              layers, cudaMemcpyDeviceToDevice, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy2DAsync(") + what +
                               ", slot to slot) failed: " + cudaGetErrorString(err));
    }
  };

  for (const ignis_seq_section &section : ignis_seq_prefix_clone_layout(pool)) {
    const char *what = ignis_seq_section_name(section.kind);
    switch (section.kind) {
    case IGNIS_SEQ_SECTION_GDN_CONV:
      copy_gdn(false, what);
      break;
    case IGNIS_SEQ_SECTION_GDN_RECURRENT:
      copy_gdn(true, what);
      break;
    case IGNIS_SEQ_SECTION_PENALTY_COUNTS:
      copy(pool.token_counts_for(dst), pool.token_counts_for(src),
           static_cast<std::size_t>(section.bytes), what);
      break;
    case IGNIS_SEQ_SECTION_DFLASH_WINDOW:
    case IGNIS_SEQ_SECTION_DFLASH_CHECKPOINT: {
      const ninfer::CyclicKVCache &cache = section.kind == IGNIS_SEQ_SECTION_DFLASH_WINDOW
                                               ? *pool.dflash2_window
                                               : *pool.dflash2_checkpoint;
      for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        const ninfer::CyclicKVCacheLayerView view = cache.layer_view(layer);
        for (const ninfer::Tensor *plane : {&view.k, &view.v}) {
          const ninfer::Tensor from = plane->slice(3, src, 1);
          const ninfer::Tensor to   = plane->slice(3, dst, 1);
          copy(to.data, from.data, from.bytes(), what);
        }
      }
      break;
    }
    case IGNIS_SEQ_SECTION_HQ_RESIDUAL: {
      // GitHub #257: the window is slot-indexed, so a claimant does not see
      // the rows of the image it cloned unless they are copied to its own
      // slot. One 2D copy per role, as for the GDN sections, plus the words.
      const std::size_t plane = static_cast<std::size_t>(pool.hq_residual_plane_bytes());
      const std::size_t pitch = pool.hq_residual_layer_pitch();
      for (const bool role_v : {false, true}) {
        const cudaError_t err = cudaMemcpy2DAsync(
            pool.hq_residual_plane(role_v, 0, dst), pitch, pool.hq_residual_plane(role_v, 0, src),
            pitch, plane, kIgnisGqaLayerCount, cudaMemcpyDeviceToDevice, nullptr);
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy2DAsync(") + what +
                                   ", slot to slot) failed: " + cudaGetErrorString(err));
        }
      }
      copy(pool.hq_ring_words(dst), pool.hq_ring_words(src),
           kIgnisHqRingWords * sizeof(std::uint32_t), what);
      break;
    }
    default:
      throw std::logic_error(std::string("state section ") + what +
                             " has no slot-to-slot copy implementation");
    }
  }
  const cudaError_t err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize after a slot copy failed: ") +
                             cudaGetErrorString(err));
  }
}

namespace {

// The checks ignis_seq_retained_store and _load share, naming `fn` in the
// message. 0 when the call may proceed.
int32_t retained_refusal(const ignis_seq_pool *pool, const ignis_seq *seq,
                         std::uint32_t retained_slot, const char *fn) {
  if (pool == nullptr || seq == nullptr) {
    set_error(std::string(fn) + ": null argument");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error(std::string(fn) + ": the sequence was not drawn from this pool");
    return -1;
  }
  if (retained_slot >= pool->retained_slot_count) {
    set_error(std::string(fn) + ": " + ignis_seq_retained_slot_refusal(*pool, retained_slot));
    return -1;
  }
  if (!ignis_seq_at_chunk_boundary(*seq)) {
    set_error(std::string(fn) + ": sequence slot " + std::to_string(seq->slot) +
              " is mid-chunk (program frontier " + std::to_string(seq->position) +
              "); its state sections are not consistent with one another");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }
  return 0;
}

} // namespace

extern "C" int32_t ignis_seq_retained_store(struct ignis_seq_pool *pool,
                                             const struct ignis_seq *seq,
                                             uint32_t retained_slot) {
  const char *const fn = "ignis_seq_retained_store";
  if (const int32_t rc = retained_refusal(pool, seq, retained_slot, fn); rc != 0) {
    return rc;
  }
  // GitHub #215: a slot a prefix or a checkpoint still holds is not written
  // over -- a claimant of that object would clone this sequence instead.
  if (const std::string refusal = ignis_seq_retained_slot_refusal(*pool, retained_slot);
      !refusal.empty()) {
    set_error(std::string(fn) + ": " + refusal);
    return -1;
  }
  try {
    ignis_seq_copy_slot_state(*pool, seq->slot, ignis_seq_retained_pool_slot(*pool, retained_slot));
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string(fn) + ": " + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_retained_load(struct ignis_seq_pool *pool, uint32_t retained_slot,
                                            struct ignis_seq *seq) {
  const char *const fn = "ignis_seq_retained_load";
  if (const int32_t rc = retained_refusal(pool, seq, retained_slot, fn); rc != 0) {
    return rc;
  }
  try {
    ignis_seq_copy_slot_state(*pool, ignis_seq_retained_pool_slot(*pool, retained_slot), seq->slot);
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string(fn) + ": " + e.what());
    return -1;
  }
}

// ---- device allocation counter (GitHub #211) -----------------------------

namespace {

struct alloc_counter {
  std::atomic<std::uint64_t> allocs{0};
  std::atomic<std::uint64_t> frees{0};
  std::atomic<std::uint64_t> alloc_bytes{0};
};

std::array<alloc_counter, IGNIS_ALLOC_KIND_COUNT> g_alloc_counters;

} // namespace

void ignis_alloc_count_record(std::int32_t kind, bool alloc, std::uint64_t bytes) {
  if (kind < 0 || kind >= IGNIS_ALLOC_KIND_COUNT) {
    return;
  }
  alloc_counter &counter = g_alloc_counters[static_cast<std::size_t>(kind)];
  if (alloc) {
    counter.allocs.fetch_add(1, std::memory_order_relaxed);
    counter.alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    counter.frees.fetch_add(1, std::memory_order_relaxed);
  }
}

extern "C" int32_t ignis_alloc_counts(int32_t kind, struct ignis_alloc_count *out) {
  if (out == nullptr || kind < 0 || kind >= IGNIS_ALLOC_KIND_COUNT) {
    return -1;
  }
  const alloc_counter &counter = g_alloc_counters[static_cast<std::size_t>(kind)];
  out->allocs      = counter.allocs.load(std::memory_order_relaxed);
  out->frees       = counter.frees.load(std::memory_order_relaxed);
  out->alloc_bytes = counter.alloc_bytes.load(std::memory_order_relaxed);
  return 0;
}

// ---- the KV-RAM arena (GitHub #213, ADR 0030) ----------------------------

namespace {

// The one pinned region the host tier places every blob in. Process-wide
// because the tier is: a handle would only be a pointer every caller hands
// back unchanged. Absent when --kv-host-pool-bytes is 0, which disables the
// tier, and absent before the load creates it.
//
// The vendored arena carries no lock of its own, and blobs are freed from
// whichever thread drops the map that held them (the three snapshot maps in
// RuntimeCompute lock independently), so every entry point below takes this
// one. Only the create makes a CUDA call under it; the rest is a free-list
// walk.
std::mutex g_host_pool_mutex;
std::unique_ptr<ninfer::HostPinnedArena> g_host_pool;

} // namespace

extern "C" int32_t ignis_host_pinned_pool_create(uint64_t bytes) {
  const std::lock_guard<std::mutex> guard(g_host_pool_mutex);
  if (g_host_pool) {
    set_error("ignis_host_pinned_pool_create: a KV-RAM arena of " +
              std::to_string(g_host_pool->capacity()) +
              " bytes already exists; destroy it before creating another");
    return -1;
  }
  if (bytes == 0) {
    // The tier is off. The arena's constructor refuses a zero capacity, and
    // there would be nothing to place in it anyway.
    return 0;
  }
  try {
    g_host_pool = std::make_unique<ninfer::HostPinnedArena>(static_cast<std::size_t>(bytes));
  } catch (const std::exception &e) {
    set_error("ignis_host_pinned_pool_create: pinning " + std::to_string(bytes) +
              " bytes of KV-RAM failed: " + e.what());
    return -1;
  }
  ignis_alloc_count_record(IGNIS_ALLOC_KV_RAM_ARENA, true, bytes);
  return 0;
}

extern "C" void ignis_host_pinned_pool_destroy(void) {
  const std::lock_guard<std::mutex> guard(g_host_pool_mutex);
  if (g_host_pool) {
    ignis_alloc_count_record(IGNIS_ALLOC_KV_RAM_ARENA, false, 0);
    g_host_pool.reset();
  }
}

extern "C" int32_t ignis_host_pinned_pool_stats(uint64_t *out_capacity, uint64_t *out_used) {
  if (out_capacity == nullptr || out_used == nullptr) {
    set_error("ignis_host_pinned_pool_stats: null argument");
    return -1;
  }
  const std::lock_guard<std::mutex> guard(g_host_pool_mutex);
  *out_capacity = g_host_pool ? static_cast<uint64_t>(g_host_pool->capacity()) : 0;
  *out_used     = g_host_pool ? static_cast<uint64_t>(g_host_pool->used()) : 0;
  return 0;
}

extern "C" int32_t ignis_host_pinned_can_alloc(uint64_t bytes, int32_t *out_fits) {
  if (out_fits == nullptr) {
    set_error("ignis_host_pinned_can_alloc: null out_fits");
    return -1;
  }
  *out_fits = 0;
  if (bytes == 0) {
    set_error("ignis_host_pinned_can_alloc: bytes must be positive");
    return -1;
  }
  const std::lock_guard<std::mutex> guard(g_host_pool_mutex);
  if (!g_host_pool) {
    return 0;
  }
  try {
    *out_fits = g_host_pool->can_alloc(static_cast<std::size_t>(bytes)) ? 1 : 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_host_pinned_can_alloc: ") + e.what());
    return -1;
  }
  return 0;
}

extern "C" int32_t ignis_host_pinned_alloc(uint64_t bytes, void **out_ptr) {
  if (out_ptr == nullptr) {
    set_error("ignis_host_pinned_alloc: null out_ptr");
    return -1;
  }
  *out_ptr = nullptr;
  if (bytes == 0) {
    set_error("ignis_host_pinned_alloc: bytes must be positive");
    return -1;
  }
  const std::lock_guard<std::mutex> guard(g_host_pool_mutex);
  if (!g_host_pool) {
    set_error("ignis_host_pinned_alloc: no KV-RAM arena -- --kv-host-pool-bytes 0 disables the "
              "tier, and the load pins the arena otherwise");
    return -1;
  }
  void *ptr = nullptr;
  try {
    ptr = g_host_pool->try_alloc(static_cast<std::size_t>(bytes));
  } catch (const std::exception &e) {
    set_error(std::string("ignis_host_pinned_alloc: ") + e.what());
    return -1;
  }
  if (ptr == nullptr) {
    set_error("ignis_host_pinned_alloc: no free span of " + std::to_string(bytes) +
              " bytes in the " + std::to_string(g_host_pool->capacity()) +
              "-byte KV-RAM arena (" + std::to_string(g_host_pool->used()) + " bytes held)");
    return IGNIS_SEQ_ERR_NO_HOST_ROOM;
  }
  *out_ptr = ptr;
  return 0;
}

extern "C" void ignis_host_pinned_free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  const std::lock_guard<std::mutex> guard(g_host_pool_mutex);
  if (!g_host_pool) {
    std::fprintf(stderr, "ignis_host_pinned_free: no KV-RAM arena to return %p to\n", ptr);
    return;
  }
  try {
    g_host_pool->free(ptr);
  } catch (const std::exception &e) {
    // A free has nowhere to return a code, and the arena refuses a pointer it
    // never handed out rather than corrupting its own free list.
    std::fprintf(stderr, "ignis_host_pinned_free: %s\n", e.what());
  }
}

extern "C" const char *ignis_seq_last_error(void) {
  return g_last_error.c_str();
}
