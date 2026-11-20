/* ignis kernel leaf: the sequence state-section table (P4-06, GitHub #124,
 * ADR 0024).
 *
 * **This is the one place in the leaf that says what a sequence is made
 * of.** Every section of a sequence's state is listed in
 * `ignis_seq_section_table` below with its size in bytes and whether it is
 * shareable read-only history or mutable state that must be cloned per
 * sequence. Every consumer reads that one description: snapshot to host,
 * restore from host, and (P4-10, GitHub #126) the device-to-device clone
 * behind prefix reuse and prompt checkpoints -- since GitHub #211 and #215 a
 * slot-to-slot copy into and out of a retained slot -- so a section is
 * carried by all of them or by none.
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
 * inherit that permission (`docs/archive/DEFERRED-DECISIONS.md` item 5). A new
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
/* 2 (P5-03, GitHub #152): the drafter's window and checkpoint sections, the
 * drafter frontier in the progress image, and the drafter's lane size in the
 * geometry.
 * 3 (GitHub #194): the progress image's reserved word is the sequence's
 * multimodal `rope_delta`. Same size, new meaning: a version-2 blob has 0
 * there, so it is refused rather than restored at a delta it never recorded.
 * 4 (GitHub #257): an hq-e8-2b pool's residual-window section. A version-3
 * hq blob carries no exact rows and no ring words, and a sequence restored
 * from one would read whatever its slot's previous occupant left there.
 * 5: the drafter's checkpoint section is gone. It held the reference's
 * rewrite checkpoint of the window, which nothing in ignis ever read; a
 * version-4 drafter blob lists one section more than this leaf's table.
 *
 * Still 5 since GitHub #622 (ADR 0095): a pool built for the 27B's MTP head
 * lists one section more, the continuation hidden, and its hq residual
 * window carries the head's ring words after the target's. Nothing else
 * changed, and a pool without the head writes the bytes it wrote before, so
 * no blob that exists is reread: an MTP pool refused every move of a
 * sequence until this table listed the head's state, so no version-5 blob of
 * one was ever written, and a blob of one pool kind restored into the other
 * is refused by name ("the MTP head's state") before its geometry is read.
 * Keeping the number lets the blob identity name the drafter, the field that
 * differs, rather than the layout version.
 * 6 (GitHub #689): the progress image ends with the sequence's ask of the
 * shape rule and a reserved word. A version-5 progress image is 8 bytes
 * shorter and says nothing of the ask, so it is refused rather than restored
 * at the base shape the request may not have been at. */
inline constexpr std::uint32_t kIgnisSeqSnapshotFormatVersion = 6;

/* GitHub #303 (spec flash-next/05): a Flash-Next pool's blobs number their
 * own layout, per model family -- family 1 in the high byte, its version in
 * the low -- so a blob of one family is never read as the other's whatever
 * either side's version is. 1: the 27B's sections plus the indexer's keys,
 * its tail and the n-gram conv state. The Flash-Next leaf (Rust,
 * flash_next_leaf.rs) puts its n-gram hashing context in a 256-byte block
 * before these bytes and names this version in its blob identity: changing
 * that block bumps it. 2 (GitHub #689): the progress image the two families
 * share grew by the 27B's ask of the shape rule, always 0 here. */
inline constexpr std::uint32_t kIgnisSeqSnapshotFormatVersionFlashNext = 0x102;

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
  IGNIS_SEQ_SECTION_PROGRESS = 4,
  /* The DFlash2 drafter's sliding K/V window for this slot's lane (P5-03,
   * GitHub #152): every drafter layer's K then V over the whole ring
   * (`ninfer::CyclicKVCache::copy_lane_to_host` order). Listed only by a pool
   * built with the drafter, and placed before the progress section in blob
   * order. Mutable: the drafter rewrites it every round. */
  IGNIS_SEQ_SECTION_DFLASH_WINDOW = 5,
  /* 6 was the window's rewrite checkpoint, retired in format 5 (the
   * reference restores it when a later turn rewrites text past its chat
   * template's boundary; ignis claims a checkpoint image taken at the
   * opener instead, which carries the window as it stood there). The value
   * is not reused, so an old kind never names a new section. */
  /* The hq-e8-2b residual window of this slot (GitHub #257, spec
   * runtime/06): every GQA layer's exact K side plane, then every layer's V
   * side plane (`ignis_seq_pool::hq_residual_plane_bytes` each, sink rows
   * then ring rows), then the 16 ring validity words. On a pool built for the
   * MTP head (GitHub #622) the head's layer is the last layer of both plane
   * runs and its own 16 ring words follow the target's. Listed only by an hq
   * pool, and placed before the progress section in blob order. Mutable:
   * every append rewrites ring rows and bits. Slot-indexed, not
   * page-indexed, which is why it is a CLONE section rather than riding the
   * shareable KV pages -- a claimant or a restored sequence addresses its
   * own slot row, never the one the image was taken from. */
  IGNIS_SEQ_SECTION_HQ_RESIDUAL = 7,
  /* GitHub #303 (spec flash-next/05): the QSA indexer's complete blocks'
   * keys, paged like the KV -- every attention layer's keys of the
   * `kv_page_count` pages, layer by layer, page by page in the sequence's
   * logical order. SHAREABLE: a block key is written once, when the block
   * completes, and a claimant of a prefix reads its pages' keys in place. */
  IGNIS_SEQ_SECTION_INDEXER_KEYS = 8,
  /* GitHub #303: the indexer's incomplete block -- its raw keys, up to
   * compress - 1 per attention layer -- slot-indexed, so CLONE, carried
   * with the image (spec flash-next/05). */
  IGNIS_SEQ_SECTION_INDEXER_TAIL = 9,
  /* GitHub #303: the n-gram embedding's dilated conv state, its past input
   * columns: CLONE, rewritten every step. */
  IGNIS_SEQ_SECTION_NGRAM_CONV = 10,
  /* GitHub #622 (spec #618, ADR 0095): the 27B MTP head's continuation hidden
   * of this slot, one BF16 [5,120] vector -- the target's post-final-norm
   * hidden at the sequence's last consumed position, half of the head's next
   * row. Listed only by a pool built for the head, and placed before the
   * progress section in blob order. CLONE: every step rewrites it, and a
   * claimant or a restored sequence builds its first head row from it. The
   * head's attention state rides the sections above: its layer is the last
   * plane run of every KV page (SHAREABLE, like the target's) and, under
   * hq-e8-2b, the last layer of the residual window with its own ring words. */
  IGNIS_SEQ_SECTION_MTP_CONTINUATION = 11
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

/* How many rows `ignis_seq_section_table` returns: every pool's sections,
 * plus the drafter's window on a pool built with it, plus the residual
 * window's one on an hq-e8-2b pool (GitHub #257). Named so a caller can
 * reserve for it and a test can assert against it, and so that adding a
 * section is one edit in one place rather than a literal to chase. */
inline constexpr std::size_t kIgnisSeqSectionCount = 5;
inline constexpr std::size_t kIgnisSeqDflash2SectionCount = 1;
inline constexpr std::size_t kIgnisSeqHqResidualSectionCount = 1;
/* GitHub #303: the indexer's keys and tail, and the n-gram conv state. */
inline constexpr std::size_t kIgnisSeqIndexerSectionCount = 2;
inline constexpr std::size_t kIgnisSeqNgramSectionCount = 1;
/* GitHub #622: the MTP head's continuation hidden. */
inline constexpr std::size_t kIgnisSeqMtpSectionCount = 1;

inline std::size_t ignis_seq_section_count(const ignis_seq_pool &pool) {
  return kIgnisSeqSectionCount + (pool.has_dflash2() ? kIgnisSeqDflash2SectionCount : 0) +
         (pool.has_hq_residual() ? kIgnisSeqHqResidualSectionCount : 0) +
         (pool.has_indexer() ? kIgnisSeqIndexerSectionCount : 0) +
         (pool.has_ngram_conv() ? kIgnisSeqNgramSectionCount : 0) +
         (pool.has_mtp_head() ? kIgnisSeqMtpSectionCount : 0);
}

/* The blob layout version a pool of this family writes and accepts (GitHub
 * #303): the 27B's, or Flash-Next's own numbering. */
inline std::uint32_t ignis_seq_snapshot_format_version_of(const ignis_seq_pool &pool) {
  return pool.has_flash_next_sections() ? kIgnisSeqSnapshotFormatVersionFlashNext
                                        : kIgnisSeqSnapshotFormatVersion;
}

/* The progress scalars, as the IGNIS_SEQ_SECTION_PROGRESS payload. Fixed
 * width and explicitly padded: it is written to a host buffer that another
 * process-lifetime may read back, so its size is pinned below rather than
 * left to the compiler. */
struct ignis_seq_progress_image {
  std::uint64_t position;
  std::int32_t pending_token;
  /* GitHub #194: the multimodal rope delta (`ignis_seq::rope_delta`); 0 for
   * a text sequence. Consistent with the scalars around it at a completed
   * chunk boundary -- the one point a snapshot is taken: `step.cu` assigns
   * it once a prefill span's chunks have run and synchronized, beside the
   * frontier advance, and every span of a multimodal prompt assigns the
   * same prompt-wide value, so a sequence evicted between its chunks
   * already holds the delta its later chunks would set. A clone's capture
   * zeroes it (`ignis_seq_capture_state`): a claimant takes its delta from
   * its own prefill span. */
  std::int32_t rope_delta;
  /* The drafter window's frontier (`ignis_seq::dflash2_position`); 0 on a
   * pool without the drafter. */
  std::uint64_t dflash2_position;
  std::uint32_t gqa_positions[kIgnisGqaLayerCount];
  std::uint32_t gdn_positions[kIgnisGdnLayerCount];
  /* GitHub #689: the sequence's ask of the shape rule
   * (`ignis_seq::asks_extended_shape`), 1 for the extended shape; 0 on a
   * pool whose load has one round shape, and in a clone's capture. */
  std::uint32_t asks_extended_shape;
  std::uint32_t reserved;
};
static_assert(sizeof(ignis_seq_progress_image) ==
                  32 + 4 * (kIgnisGqaLayerCount + kIgnisGdnLayerCount),
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
  /* One slot's lane of the drafter window, as the pool planned it; 0 on a
   * pool without the drafter. This is the field that refuses a blob taken
   * with the drafter on a pool without it, and the reverse (P5-03, GitHub
   * #152). */
  std::uint64_t dflash2_window_lane_bytes;
};
static_assert(sizeof(ignis_seq_snapshot_geometry) == 80,
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
static_assert(sizeof(ignis_seq_snapshot_header) == 128,
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
  geometry.gqa_layer_count = static_cast<std::uint32_t>(pool.kv_num_layers);
  geometry.kv_page_bytes   = pool.kv_page_bytes;
  geometry.gdn_num_layers  = gdn.layers;
  geometry.gdn_conv_channels        = static_cast<std::uint32_t>(gdn.conv_channels);
  geometry.gdn_conv_width           = static_cast<std::uint32_t>(gdn.conv_width);
  geometry.gdn_value_heads          = static_cast<std::uint32_t>(gdn.value_heads);
  geometry.gdn_head_dim             = static_cast<std::uint32_t>(gdn.value_head_dim);
  geometry.vocab                    = static_cast<std::uint32_t>(pool.vocab);
  geometry.gdn_conv_slot_bytes      = pool.gdn_pool.conv_slot_bytes();
  geometry.gdn_recurrent_slot_bytes = pool.gdn_pool.recurrent_slot_bytes();
  geometry.dflash2_window_lane_bytes = pool.dflash2_lane_bytes();
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
  if (blob.dflash2_window_lane_bytes != target.dflash2_window_lane_bytes) {
    return blob.dflash2_window_lane_bytes == 0 || target.dflash2_window_lane_bytes == 0
               ? "dflash2 drafter state (one side was built without the drafter)"
               : "dflash2 window lane bytes";
  }
  return "an unnamed geometry field";
}

/* `seq`'s IGNIS_SEQ_SECTION_PROGRESS payload, and the same payload applied
 * back to a sequence.
 *
 * Here rather than beside one consumer because all three read it: snapshot
 * writes it into a blob, restore reads it back out of one, and a prefix
 * clone hands it from a publisher to a claimant without a blob at all
 * (P4-10, GitHub #126). One pair of functions is what keeps a scalar added
 * to `ignis_seq_progress_image` from reaching two of them and not the
 * third. */
inline ignis_seq_progress_image ignis_seq_progress_of(const ignis_seq &seq) {
  ignis_seq_progress_image image{};
  image.position      = seq.position;
  image.pending_token = seq.pending_token;
  image.rope_delta    = seq.rope_delta;
  image.dflash2_position = seq.dflash2_position;
  for (std::size_t i = 0; i < static_cast<std::size_t>(kIgnisGqaLayerCount); ++i) {
    image.gqa_positions[i] = seq.gqa_positions[i];
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(kIgnisGdnLayerCount); ++i) {
    image.gdn_positions[i] = seq.gdn_positions[i];
  }
  image.asks_extended_shape = seq.asks_extended_shape ? 1 : 0;
  return image;
}

inline void ignis_seq_apply_progress(ignis_seq &seq, const ignis_seq_progress_image &image) {
  seq.position      = image.position;
  seq.pending_token = image.pending_token;
  seq.rope_delta    = image.rope_delta;
  seq.dflash2_position = image.dflash2_position;
  // GitHub #157: the carried anchor taps are the handle's, not the image's;
  // taps left from this handle's previous life belong to another position.
  seq.dflash2_pending = false;
  seq.dflash2_pending_features.clear();
  // GitHub #586: and so are the kept scores of its pending draw -- the image's
  // pending token was drawn from scores no column holds for this handle.
  seq.kept = ignis_kept_scores{};
  for (std::size_t i = 0; i < static_cast<std::size_t>(kIgnisGqaLayerCount); ++i) {
    seq.gqa_positions[i] = image.gqa_positions[i];
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(kIgnisGdnLayerCount); ++i) {
    seq.gdn_positions[i] = image.gdn_positions[i];
  }
  // GitHub #689: the image's ask, never the handle's own from a previous life.
  seq.asks_extended_shape = image.asks_extended_shape != 0;
  // GitHub #690: and no recent rounds, which the image does not carry: the
  // handle's own are another life's.
  seq.shape_rounds = {};
  seq.shape_rounds_recorded = 0;
}

/* The physical KV pages a snapshot of `seq` captures: the pages its written
 * history occupies, not its whole reservation. A sequence reserves its
 * entire context up front, so capturing the reservation would price a
 * 100-token sequence like a 40,960-token one. */
inline std::uint32_t ignis_seq_snapshot_page_count(const ignis_seq &seq) {
  const std::uint64_t tokens = seq.position;
  if (tokens > static_cast<std::uint64_t>(UINT32_MAX)) {
    return ignis_seq_logical_page_count(seq);
  }
  const std::uint32_t pages = ninfer::pages_for_tokens(static_cast<std::uint32_t>(tokens));
  return pages < ignis_seq_logical_page_count(seq) ? pages : ignis_seq_logical_page_count(seq);
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
  sections.reserve(ignis_seq_section_count(pool));
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
  // GitHub #303: the indexer's block keys ride with the KV pages they sit
  // beside, and are shared the same way.
  if (pool.has_indexer()) {
    push(IGNIS_SEQ_SECTION_INDEXER_KEYS, IGNIS_SEQ_SECTION_SHAREABLE,
         static_cast<std::uint64_t>(pool.kv_num_layers) * kv_page_count * pool.indexer_page_bytes());
  }
  push(IGNIS_SEQ_SECTION_GDN_CONV, IGNIS_SEQ_SECTION_CLONE,
       pool.gdn_pool.conv_host_image_bytes());
  push(IGNIS_SEQ_SECTION_GDN_RECURRENT, IGNIS_SEQ_SECTION_CLONE,
       pool.gdn_pool.recurrent_host_image_bytes());
  push(IGNIS_SEQ_SECTION_PENALTY_COUNTS, IGNIS_SEQ_SECTION_CLONE,
       static_cast<std::uint64_t>(pool.vocab) * sizeof(std::int32_t));
  // P5-03 (GitHub #152): the drafter's window. CLONE, not SHAREABLE: unlike
  // KV pages the window is rewritten in place every round, so a claimant
  // writes it from its first step. Consistent with the sections above at the
  // same completed chunk boundary, because the prefill chunk that writes it
  // synchronizes once for both (and advances `dflash2_position` with the
  // rest of the progress scalars).
  if (pool.has_dflash2()) {
    push(IGNIS_SEQ_SECTION_DFLASH_WINDOW, IGNIS_SEQ_SECTION_CLONE, pool.dflash2_lane_bytes());
  }
  // GitHub #257: the hq residual window. Consistent with the sections above
  // at a completed chunk boundary for the same reason the KV pages are: the
  // append that writes a side row and its ring bit is the same launch that
  // writes the row's code, inside the chunk the one synchronize confirms. A
  // verify round clears the bits of its rejected columns before it returns
  // (kernel/src/step.cu), so no boundary a snapshot can see holds a bit for
  // a position the sequence did not keep.
  if (pool.has_hq_residual()) {
    push(IGNIS_SEQ_SECTION_HQ_RESIDUAL, IGNIS_SEQ_SECTION_CLONE, pool.hq_residual_slot_bytes());
  }
  // GitHub #303 (spec flash-next/05): Flash-Next's slot state. Consistent
  // with the sections above at a completed chunk boundary: the chunk that
  // appends a token's raw key and its conv column is the one the program's
  // synchronize confirms (kernel/src/flash_next/program.cu).
  if (pool.has_indexer()) {
    push(IGNIS_SEQ_SECTION_INDEXER_TAIL, IGNIS_SEQ_SECTION_CLONE,
         static_cast<std::uint64_t>(pool.kv_num_layers) * pool.indexer_tail_slot_bytes);
  }
  if (pool.has_ngram_conv()) {
    push(IGNIS_SEQ_SECTION_NGRAM_CONV, IGNIS_SEQ_SECTION_CLONE, pool.ngram_conv_slot_bytes);
  }
  // GitHub #622 (ADR 0095): the MTP head's continuation hidden. The head's
  // state earns the snapshot point at the two points a snapshot is taken,
  // and holds there what `ignis_mtp_head_frontier` derives from the progress
  // scalars -- no frontier of its own to disagree:
  // - a completed prefill chunk: the chunk's head call writes the rows
  //   [start - 1, end - 1) from the hidden the step before left and the
  //   chunk's own, then leaves its last column as the continuation hidden,
  //   inside the chunk the one synchronize confirms -- every row below the
  //   head's frontier, none at it, the hidden at the last consumed position;
  // - a committed round: a plain round writes its lanes' rows before their
  //   positions and the round's hidden; a verify round's commit rewrites the
  //   committed run's rows and moves the hidden to the run's last column as
  //   its last device step, and under hq-e8-2b clears the head's ring bits of
  //   every chain row past the commit, so rows past the head's frontier are
  //   never read before a later step writes them. A round that fails before
  //   that step leaves the frontier and the hidden together.
  if (pool.has_mtp_head()) {
    push(IGNIS_SEQ_SECTION_MTP_CONTINUATION, IGNIS_SEQ_SECTION_CLONE, pool.mtp_continuation_slot_bytes());
  }
  push(IGNIS_SEQ_SECTION_PROGRESS, IGNIS_SEQ_SECTION_CLONE, sizeof(ignis_seq_progress_image));
  assert(sections.size() == ignis_seq_section_count(pool) &&
         "ignis_seq_section_count has drifted from the table above");

  // The payload starts after room for every section this leaf knows, not
  // only the ones this pool lists: more records would otherwise push the
  // first payload across an alignment boundary, and a drafter pool's blob
  // would outgrow a plain one's by more than its lane (P5-03, GitHub #152).
  // The unused records' bytes are zeroed like any other gap.
  // GitHub #303: the Flash-Next sections' records fit the same aligned
  // room (128 + 10 x 24 bytes still round to 512), so a 27B blob's payload
  // starts where it always did. GitHub #622: and so does the MTP head's
  // record (128 + 11 x 24 = 392), which keeps every blob of a pool without
  // the head the bytes it was.
  std::uint64_t cursor = ignis_seq_align_up(
      sizeof(ignis_seq_snapshot_header) +
          (kIgnisSeqSectionCount + kIgnisSeqDflash2SectionCount + kIgnisSeqHqResidualSectionCount +
           kIgnisSeqIndexerSectionCount + kIgnisSeqNgramSectionCount + kIgnisSeqMtpSectionCount) *
              sizeof(ignis_seq_section),
      kIgnisSeqSectionAlign);
  static_assert((sizeof(ignis_seq_snapshot_header) + 7 * sizeof(ignis_seq_section) + kIgnisSeqSectionAlign - 1) /
                        kIgnisSeqSectionAlign ==
                    (sizeof(ignis_seq_snapshot_header) + 11 * sizeof(ignis_seq_section) + kIgnisSeqSectionAlign - 1) /
                        kIgnisSeqSectionAlign,
                "the Flash-Next or MTP records moved a 27B blob's first payload");
  static_assert(kIgnisSeqSectionCount + kIgnisSeqDflash2SectionCount + kIgnisSeqHqResidualSectionCount +
                        kIgnisSeqIndexerSectionCount + kIgnisSeqNgramSectionCount + kIgnisSeqMtpSectionCount ==
                    11,
                "the record room above counts every section kind this leaf knows");
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
  case IGNIS_SEQ_SECTION_DFLASH_WINDOW:
    return "dflash_window";
  case IGNIS_SEQ_SECTION_HQ_RESIDUAL:
    return "hq_residual";
  case IGNIS_SEQ_SECTION_INDEXER_KEYS:
    return "indexer_keys";
  case IGNIS_SEQ_SECTION_INDEXER_TAIL:
    return "indexer_tail";
  case IGNIS_SEQ_SECTION_NGRAM_CONV:
    return "ngram_conv";
  case IGNIS_SEQ_SECTION_MTP_CONTINUATION:
    return "mtp_continuation";
  default:
    return "unknown";
  }
}

/* GitHub #622: why a blob listing `records` cannot be restored into a pool
 * whose head presence is `pool_has_head`, or nullptr. A blob taken on a pool
 * built for the MTP head lists the continuation hidden, and one taken on any
 * other pool does not; the two kinds of pool also differ in their KV pages
 * (17 attention layers against 16), which the geometry would name as a plane
 * count. This names the drafter state that actually differs, ahead of the
 * geometry, the way the DFlash2 window's lane size names the other drafter. */
inline const char *ignis_seq_mtp_head_refusal(const std::vector<ignis_seq_section> &records, bool pool_has_head) {
  bool blob_has_head = false;
  for (const ignis_seq_section &record : records) {
    blob_has_head = blob_has_head || record.kind == IGNIS_SEQ_SECTION_MTP_CONTINUATION;
  }
  if (blob_has_head == pool_has_head) {
    return nullptr;
  }
  return blob_has_head ? "the MTP head's state (the blob was taken on a pool built for the MTP head, this pool has "
                         "none)"
                       : "the MTP head's state (this pool is built for the MTP head, the blob was taken on a pool "
                         "without it)";
}

/* The header describing a blob of `sections` over `pool`'s geometry, with
 * its checksum already computed. */
inline ignis_seq_snapshot_header
ignis_seq_snapshot_header_for(const ignis_seq_pool &pool, std::uint32_t kv_page_count,
                              const std::vector<ignis_seq_section> &sections) {
  ignis_seq_snapshot_header header{};
  header.magic                = kIgnisSeqSnapshotMagic;
  header.format_version       = ignis_seq_snapshot_format_version_of(pool);
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
