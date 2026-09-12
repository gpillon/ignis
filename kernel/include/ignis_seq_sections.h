/* ignis kernel leaf: the sequence state-section table (P4-06, GitHub #124,
 * ADR 0024).
 *
 * **This is the one place in the leaf that says what a sequence is made
 * of.** Every section of a sequence's state is listed in
 * `ignis_seq_section_table` below with its size in bytes and whether it is
 * shareable read-only history or mutable state that must be cloned per
 * sequence. Three consumers read that one description: snapshot to host,
 * restore from host, and (P4-10, GitHub #126) the device-to-device clone
 * behind prefix reuse -- so a section is carried by all three or by none.
 *
 * Not part of the public flat C ABI. ADR 0024 is explicit that the layout
 * does not cross the C boundary: `ignis_seq.h` exposes a snapshot *size*
 * and a *format version*, and the blob itself is opaque and carries its own
 * header. A consumer that wanted the layout would be computing buffer
 * geometry the leaf already owns, and would pin this file into a contract
 * every future section change breaks.
 *
 * **Adding a section is an edit to `ignis_seq_section_table`, and that edit
 * is the act that re-earns the snapshot-point permission.** The G3 session
 * established that the sections below are mutually consistent at a completed
 * chunk boundary and recorded the caveat that a *new* section does not
 * inherit that permission (`.scratch/DEFERRED-DECISIONS.md` item 5). A new
 * section must therefore come with its own answer to "at which points is
 * this consistent with the others?", and must bump
 * `kIgnisSeqSnapshotFormatVersion` so blobs written before it are refused
 * rather than reinterpreted.
 */
#ifndef IGNIS_SEQ_SECTIONS_H
#define IGNIS_SEQ_SECTIONS_H

#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

/* The snapshot blob format this leaf writes and accepts. Bump it whenever
 * the section table, a section's own payload layout, or the header below
 * changes: a blob does not survive a version change, which is the intent --
 * a snapshot taken before a layout change is unusable and must be rejected,
 * not reinterpreted (ADR 0024). */
inline constexpr std::uint32_t kIgnisSeqSnapshotFormatVersion = 1;

/* 'IGNISSNP' little-endian: the first thing a restore checks, so a foreign
 * buffer is refused before any of its fields are believed. */
inline constexpr std::uint64_t kIgnisSeqSnapshotMagic = 0x504E5353494E4749ULL;

/* Every section payload starts at a multiple of this, matching the device
 * arenas' own alignment so a host->device copy of a section never starts
 * mid-word. */
inline constexpr std::uint64_t kIgnisSeqSectionAlign = 256;

/* The sections a sequence is made of, in blob order.
 *
 * The order is part of the format: a restore checks that the blob's records
 * are exactly the table this leaf would build, kind for kind, so a blob from
 * a build with a different table is refused by its layout and not only by
 * its version. */
enum ignis_seq_section_kind {
  /* The sequence's KV history: `kv_page_count` physical pages packed across
   * every plane of every GQA layer, in the pool's own plane order
   * (`ninfer::pack_paged_kv_allocation_to_host`). */
  IGNIS_SEQ_SECTION_KV_PAGES = 0,
  /* The GDN causal-conv taps: `gdn_num_layers` x `conv_slot_bytes`. */
  IGNIS_SEQ_SECTION_GDN_CONV = 1,
  /* The GDN recurrent state matrices: `gdn_num_layers` x
   * `recurrent_slot_bytes` (144 MiB at the 27B geometry -- the snapshot's
   * floor, paid regardless of prompt length). */
  IGNIS_SEQ_SECTION_GDN_RECURRENT = 2,
  /* The presence/frequency penalty counts (P3-03, GitHub #99): one int32
   * per vocab entry. */
  IGNIS_SEQ_SECTION_PENALTY_COUNTS = 3,
  /* The host-side progress scalars (`ignis_seq_progress_image`): the
   * program frontier, the pending token and every GQA layer's own frontier.
   * A section like any other, so that "what a sequence is made of" has no
   * footnotes. */
  IGNIS_SEQ_SECTION_PROGRESS = 4
};

/* How a second sequence may come to hold a section (ADR 0024).
 *
 * This is the field prefix reuse reads: shareable sections are pointed at
 * by refcount and never copied, clone sections are copied device-to-device.
 * It is recorded here rather than at the copy site so that one description
 * answers for the snapshot path and the clone path both. */
enum ignis_seq_section_transfer {
  /* Read-only history: a second sequence may reference the same physical
   * bytes under a refcount rather than receive a copy. */
  IGNIS_SEQ_SECTION_SHAREABLE = 0,
  /* Mutable per-sequence state: a second sequence must receive its own
   * copy, because it will write to it from the first step onwards. */
  IGNIS_SEQ_SECTION_CLONE = 1
};

/* One row of the table: what the section is, how it transfers, where it
 * sits in the blob, and how many bytes it occupies there. */
struct ignis_seq_section {
  std::int32_t kind;     /* enum ignis_seq_section_kind */
  std::int32_t transfer; /* enum ignis_seq_section_transfer */
  std::uint64_t offset;  /* from the start of the blob */
  std::uint64_t bytes;
};

/* How many rows `ignis_seq_section_table` returns. Named so a caller can
 * reserve for it and a test can assert against it, and so that adding a
 * section is one edit in one place rather than a literal to chase. */
inline constexpr std::size_t kIgnisSeqSectionCount = 5;

/* The progress scalars, as the IGNIS_SEQ_SECTION_PROGRESS payload. Fixed
 * width and explicitly padded: it is written to a host buffer that another
 * process-lifetime may read back, so its size is pinned below rather than
 * left to the compiler. */
struct ignis_seq_progress_image {
  std::uint64_t position;
  std::int32_t pending_token;
  std::int32_t reserved;
  std::uint32_t gqa_positions[kIgnisGqaLayerCount];
  std::uint32_t gdn_positions[kIgnisGdnLayerCount];
};
static_assert(sizeof(ignis_seq_progress_image) ==
                  16 + 4 * (kIgnisGqaLayerCount + kIgnisGdnLayerCount),
              "ignis_seq_progress_image has gained padding; bump "
              "kIgnisSeqSnapshotFormatVersion and restate the size");

/* The pool geometry a blob was taken at, and which a restore's target pool
 * must match.
 *
 * One struct rather than loose header fields so that the geometry is built
 * once and compared *whole*: a field added here is carried into the blob and
 * into the comparison by the same edit, where two field-by-field lists would
 * let a new field be written and never checked.
 *
 * Deliberately the pools' *planned* sizes and not only their spec fields:
 * `kv_page_bytes`, `gdn_conv_slot_bytes` and `gdn_recurrent_slot_bytes` come
 * out of what the pools actually planned, so a vendored layout change that
 * leaves every spec field equal still reads as a mismatch. */
struct ignis_seq_snapshot_geometry {
  std::int32_t kv_format;
  std::uint32_t kv_num_kv_heads;
  std::uint32_t kv_head_dim;
  std::uint32_t kv_page_size;
  std::uint32_t kv_plane_count;
  std::uint32_t gqa_layer_count;
  std::uint64_t kv_page_bytes;
  std::uint32_t gdn_num_layers;
  std::uint32_t gdn_conv_channels;
  std::uint32_t gdn_conv_width;
  std::uint32_t gdn_value_heads;
  std::uint32_t gdn_head_dim;
  std::uint32_t vocab;
  std::uint64_t gdn_conv_slot_bytes;
  std::uint64_t gdn_recurrent_slot_bytes;
};
static_assert(sizeof(ignis_seq_snapshot_geometry) == 72,
              "the snapshot geometry layout changed; bump kIgnisSeqSnapshotFormatVersion, "
              "restate the size, and add the new field to ignis_seq_snapshot_geometry_names");

/* The blob's own header: identity, the geometry a restore must agree with,
 * and the shape of the record array that follows it. Everything a restore
 * validates before it touches the target sequence is here. */
struct ignis_seq_snapshot_header {
  std::uint64_t magic;
  std::uint32_t format_version;
  std::uint32_t header_bytes;
  std::uint32_t section_record_bytes;
  std::uint32_t section_count;
  std::uint64_t total_bytes;

  struct ignis_seq_snapshot_geometry geometry;

  /* --- what this particular blob holds ---------------------------------- */
  /* Physical KV pages captured: `pages_for_tokens(position)`, the history
   * the sequence has actually written, not its whole reservation. */
  std::uint32_t kv_page_count;
  /* Explicit, so the struct has no implicit padding to reason about: the
   * checksum below runs over the raw bytes of this header. */
  std::uint32_t reserved0;

  /* FNV-1a 64 over this header with the field itself zeroed. Cheap, and
   * enough to refuse a buffer that is neither a snapshot nor empty. */
  std::uint64_t header_checksum;
};
static_assert(sizeof(ignis_seq_snapshot_header) == 120,
              "the snapshot header layout changed; bump kIgnisSeqSnapshotFormatVersion "
              "and restate the size");
static_assert(sizeof(ignis_seq_section) == 24,
              "the section record layout changed; bump kIgnisSeqSnapshotFormatVersion "
              "and restate the size");

inline std::uint64_t ignis_seq_align_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

/* FNV-1a 64 over `bytes` of `data`. */
inline std::uint64_t ignis_seq_fnv1a(const void *data, std::size_t bytes) {
  const auto *p = static_cast<const unsigned char *>(data);
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

/* A sequence is at a chunk boundary when **every** layer's frontier -- all
 * 16 GQA and all 48 GDN -- has caught up with the program frontier (ADR
 * 0018).
 *
 * `kernel/src/step.cu` advances all of them only after the chunk's single
 * `cudaStreamSynchronize` has confirmed the whole chunk's device work, so
 * their agreeing is exactly the "completed chunk" the G3 session established
 * the sections are mutually consistent at. They disagree while a caller
 * drives a per-layer entry point itself (`ignis_gqa_layer_step`,
 * `ignis_gdn_layer_step`), which is the mid-chunk state a snapshot must
 * refuse rather than capture.
 *
 * Both arrays, not just the GQA one: a GDN layer's conv taps and recurrent
 * state are updated in place, so a sequence stepped one GDN layer at a time
 * has GDN state ahead of its KV with nothing about the KV pages to show it.
 * That is the quietest version of the inconsistency, not the absent one. */
inline bool ignis_seq_at_chunk_boundary(const ignis_seq &seq) {
  for (std::uint32_t frontier : seq.gqa_positions) {
    if (static_cast<std::uint64_t>(frontier) != seq.position) {
      return false;
    }
  }
  for (std::uint32_t frontier : seq.gdn_positions) {
    if (static_cast<std::uint64_t>(frontier) != seq.position) {
      return false;
    }
  }
  return true;
}

/* The geometry of the pool `pool`'s sequences live in. */
inline ignis_seq_snapshot_geometry ignis_seq_snapshot_geometry_of(const ignis_seq_pool &pool) {
  const ninfer::LinearAttentionStatePoolSpec &gdn = pool.gdn_pool.spec;
  ignis_seq_snapshot_geometry geometry{};
  geometry.kv_format       = pool.kv_format;
  geometry.kv_num_kv_heads = static_cast<std::uint32_t>(pool.kv_num_kv_heads);
  geometry.kv_head_dim     = static_cast<std::uint32_t>(pool.kv_head_dim);
  geometry.kv_page_size    = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  geometry.kv_plane_count  = static_cast<std::uint32_t>(pool.kv_pool.plane_count());
  geometry.gqa_layer_count = static_cast<std::uint32_t>(kIgnisGqaLayerCount);
  geometry.kv_page_bytes   = pool.kv_page_bytes;
  geometry.gdn_num_layers  = gdn.layers;
  geometry.gdn_conv_channels        = static_cast<std::uint32_t>(gdn.conv_channels);
  geometry.gdn_conv_width           = static_cast<std::uint32_t>(gdn.conv_width);
  geometry.gdn_value_heads          = static_cast<std::uint32_t>(gdn.value_heads);
  geometry.gdn_head_dim             = static_cast<std::uint32_t>(gdn.value_head_dim);
  geometry.vocab                    = static_cast<std::uint32_t>(pool.vocab);
  geometry.gdn_conv_slot_bytes      = pool.gdn_pool.conv_slot_bytes();
  geometry.gdn_recurrent_slot_bytes = pool.gdn_pool.recurrent_slot_bytes();
  return geometry;
}

/* The name of the first field in which `blob` and `target` differ, or
 * nullptr if they are identical.
 *
 * Equality is decided by one `memcmp` over the whole struct, so no field can
 * escape the check by being missing from the list below -- the list only
 * makes the message name the field. A field added to the struct and not to
 * the list is still refused, and says so. */
inline const char *ignis_seq_snapshot_geometry_names(const ignis_seq_snapshot_geometry &blob,
                                                     const ignis_seq_snapshot_geometry &target) {
  if (std::memcmp(&blob, &target, sizeof(blob)) == 0) {
    return nullptr;
  }
  if (blob.kv_format != target.kv_format) return "kv_format";
  if (blob.kv_num_kv_heads != target.kv_num_kv_heads) return "num_kv_heads";
  if (blob.kv_head_dim != target.kv_head_dim) return "head_dim";
  if (blob.kv_page_size != target.kv_page_size) return "kv page size";
  if (blob.kv_plane_count != target.kv_plane_count) return "kv plane count";
  if (blob.gqa_layer_count != target.gqa_layer_count) return "gqa layer count";
  if (blob.kv_page_bytes != target.kv_page_bytes) return "kv page bytes";
  if (blob.gdn_num_layers != target.gdn_num_layers) return "gdn layer count";
  if (blob.gdn_conv_channels != target.gdn_conv_channels) return "gdn conv channels";
  if (blob.gdn_conv_width != target.gdn_conv_width) return "gdn conv width";
  if (blob.gdn_value_heads != target.gdn_value_heads) return "gdn value heads";
  if (blob.gdn_head_dim != target.gdn_head_dim) return "gdn head dim";
  if (blob.vocab != target.vocab) return "vocab";
  if (blob.gdn_conv_slot_bytes != target.gdn_conv_slot_bytes) return "gdn conv slot bytes";
  if (blob.gdn_recurrent_slot_bytes != target.gdn_recurrent_slot_bytes) {
    return "gdn recurrent slot bytes";
  }
  return "an unnamed geometry field";
}

/* The physical KV pages a snapshot of `seq` captures: the pages its written
 * history occupies, not its whole reservation. A sequence reserves its
 * entire context up front, so capturing the reservation would price a
 * 100-token sequence like a 40,960-token one. */
inline std::uint32_t ignis_seq_snapshot_page_count(const ignis_seq &seq) {
  const std::uint64_t tokens = seq.position;
  if (tokens > static_cast<std::uint64_t>(UINT32_MAX)) {
    return seq.kv.mapped_page_count();
  }
  const std::uint32_t pages = ninfer::pages_for_tokens(static_cast<std::uint32_t>(tokens));
  return pages < seq.kv.mapped_page_count() ? pages : seq.kv.mapped_page_count();
}

/* The state-section table of a sequence of `pool`'s geometry holding
 * `kv_page_count` pages of history -- **the one description of what a
 * sequence is made of** (ADR 0024).
 *
 * Offsets are assigned relative to the start of the blob, after the header
 * and the record array, each section aligned to `kIgnisSeqSectionAlign`. The
 * table is a pure function of (pool geometry, page count), which is what
 * lets a restore rebuild the table it *expects* from the blob's own header
 * and compare it record for record against the blob's.
 *
 * A zero-byte section (an unwritten sequence's KV history) shares its
 * offset with the section after it. That is deliberate: the offset is
 * computed the same way on the write side and the validate side, so the two
 * agree, and no bytes are written at it either way.
 *
 * Throws (`std::overflow_error`) only out of the vendored page-image
 * arithmetic; callers at the ABI boundary catch it. */
inline std::vector<ignis_seq_section> ignis_seq_section_table(const ignis_seq_pool &pool,
                                                              std::uint32_t kv_page_count) {
  std::vector<ignis_seq_section> sections;
  sections.reserve(kIgnisSeqSectionCount);
  const auto push = [&sections](ignis_seq_section_kind kind, ignis_seq_section_transfer transfer,
                                std::uint64_t bytes) {
    sections.push_back({static_cast<std::int32_t>(kind), static_cast<std::int32_t>(transfer), 0,
                        bytes});
  };

  // KV pages are read-only history: a sibling that matches this prefix is
  // handed the same physical pages under a refcount rather than a copy
  // (ADR 0024, GitHub #126). Every section below is written from the first
  // step a restored sequence takes, so each must be cloned per sequence.
  push(IGNIS_SEQ_SECTION_KV_PAGES, IGNIS_SEQ_SECTION_SHAREABLE,
       ninfer::paged_kv_host_image_bytes(pool.kv_pool, kv_page_count));
  push(IGNIS_SEQ_SECTION_GDN_CONV, IGNIS_SEQ_SECTION_CLONE,
       pool.gdn_pool.conv_host_image_bytes());
  push(IGNIS_SEQ_SECTION_GDN_RECURRENT, IGNIS_SEQ_SECTION_CLONE,
       pool.gdn_pool.recurrent_host_image_bytes());
  push(IGNIS_SEQ_SECTION_PENALTY_COUNTS, IGNIS_SEQ_SECTION_CLONE,
       static_cast<std::uint64_t>(pool.vocab) * sizeof(std::int32_t));
  push(IGNIS_SEQ_SECTION_PROGRESS, IGNIS_SEQ_SECTION_CLONE, sizeof(ignis_seq_progress_image));
  assert(sections.size() == kIgnisSeqSectionCount &&
         "kIgnisSeqSectionCount has drifted from the table above");

  std::uint64_t cursor = ignis_seq_align_up(
      sizeof(ignis_seq_snapshot_header) + sections.size() * sizeof(ignis_seq_section),
      kIgnisSeqSectionAlign);
  for (ignis_seq_section &section : sections) {
    section.offset = cursor;
    cursor = ignis_seq_align_up(cursor + section.bytes, kIgnisSeqSectionAlign);
  }
  return sections;
}

/* Bytes a blob of `sections` occupies: the end of the last section's
 * payload, its tail padding included, so the reported size and the written
 * size are the same number by construction. `ignis_seq_section_table`
 * assigns offsets in order, so the last section ends last. */
inline std::uint64_t ignis_seq_snapshot_bytes(const std::vector<ignis_seq_section> &sections) {
  const std::uint64_t records_end = sizeof(ignis_seq_snapshot_header) +
                                    sections.size() * sizeof(ignis_seq_section);
  if (sections.empty()) {
    return ignis_seq_align_up(records_end, kIgnisSeqSectionAlign);
  }
  const ignis_seq_section &last = sections.back();
  return ignis_seq_align_up(last.offset + last.bytes, kIgnisSeqSectionAlign);
}

/* A section's name, for the messages a refusal sets. */
inline const char *ignis_seq_section_name(std::int32_t kind) {
  switch (kind) {
  case IGNIS_SEQ_SECTION_KV_PAGES:
    return "kv_pages";
  case IGNIS_SEQ_SECTION_GDN_CONV:
    return "gdn_conv";
  case IGNIS_SEQ_SECTION_GDN_RECURRENT:
    return "gdn_recurrent";
  case IGNIS_SEQ_SECTION_PENALTY_COUNTS:
    return "penalty_counts";
  case IGNIS_SEQ_SECTION_PROGRESS:
    return "progress";
  default:
    return "unknown";
  }
}

/* The header describing a blob of `sections` over `pool`'s geometry, with
 * its checksum already computed. */
inline ignis_seq_snapshot_header
ignis_seq_snapshot_header_for(const ignis_seq_pool &pool, std::uint32_t kv_page_count,
                              const std::vector<ignis_seq_section> &sections) {
  ignis_seq_snapshot_header header{};
  header.magic                = kIgnisSeqSnapshotMagic;
  header.format_version       = kIgnisSeqSnapshotFormatVersion;
  header.header_bytes         = static_cast<std::uint32_t>(sizeof(ignis_seq_snapshot_header));
  header.section_record_bytes = static_cast<std::uint32_t>(sizeof(ignis_seq_section));
  header.section_count        = static_cast<std::uint32_t>(sections.size());
  header.total_bytes          = ignis_seq_snapshot_bytes(sections);

  header.geometry      = ignis_seq_snapshot_geometry_of(pool);
  header.kv_page_count = kv_page_count;

  header.header_checksum = 0;
  header.header_checksum = ignis_seq_fnv1a(&header, sizeof(header));
  return header;
}

#endif /* IGNIS_SEQ_SECTIONS_H */
