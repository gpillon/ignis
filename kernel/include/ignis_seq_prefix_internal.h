/* ignis kernel leaf: the shared-prefix entry and the device-to-device
 * transfer of a sequence's mutable state (P4-10, GitHub #126, ADR 0024).
 *
 * A shared prefix is two things at once, and ADR 0024 names both:
 *
 *   - **the read-only half is shared in place.** The entry owns the physical
 *     KV pages of a prompt head, and every sequence that claims it addresses
 *     those same pages through its own block-table row. Nothing is copied,
 *     and the pages are charged to the pool exactly once however many
 *     claimants hold them.
 *   - **the mutable half is cloned device-to-device.** The GDN recurrent
 *     state, the conv taps and the penalty-count row are written from the
 *     first step a claimant takes, so each claimant needs its own copy. The
 *     entry keeps one image of them, captured at the prefix's end into a
 *     **retained slot** of the pool (GitHub #215, ADR 0030) -- reserved at
 *     load, so a publish allocates nothing -- and a claim copies it into the
 *     claimant's slot without the bytes ever leaving the card.
 *
 * Both halves are described by the **same state-section table** the snapshot
 * path walks (`ignis_seq_sections.h`): `IGNIS_SEQ_SECTION_SHAREABLE` is the
 * first bullet and `IGNIS_SEQ_SECTION_CLONE` is the second. That is what
 * makes "a section is carried by all three or by none" checkable rather than
 * remembered -- a section added to the table with no case in
 * `ignis_seq_copy_slot_state` throws, exactly as it does in
 * ignis_seq_snapshot and ignis_seq_restore.
 *
 * Not part of the public flat C ABI: `ignis_seq.h` keeps `ignis_seq_prefix`
 * opaque and exposes only publish / claim / release / stats.
 */
#ifndef IGNIS_SEQ_PREFIX_INTERNAL_H
#define IGNIS_SEQ_PREFIX_INTERNAL_H

#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_sections.h"

#include "core/arena.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/* A published prefix: the shared pages, the cloned state, and who holds it.
 *
 * Lifetime is one refcount over three kinds of holder. The handle
 * ignis_seq_prefix_publish returns is one (released by
 * ignis_seq_prefix_release), every sequence allocated against the prefix is
 * one more (released by ignis_seq_release), and since GitHub #187 a *chained*
 * prefix published on top of this one is one more again. The entry -- and
 * with it the KV pages -- is destroyed when the count reaches zero, which is
 * the leaf's answer to "a page is freed only when the last holder releases
 * it". Destroying a chained entry drops one reference from its parent, so a
 * whole run of the chain can go at once. */
struct ignis_seq_prefix {
  /* The shared KV pages this entry itself owns. Owned here and nowhere else:
   * a claiming sequence's own `ignis_seq::kv` covers only the tail it writes
   * itself, and this allocation is never bound to a block-table row -- rows
   * are per sequence, pages are not.
   *
   * GitHub #187: its OWN pages, not its whole history's. A chained entry
   * covers `parent`'s pages too, and those stay the parent's allocation --
   * charged to the pool once and freed once, whichever link of the chain a
   * sequence happens to hold. */
  ninfer::PagedKVAllocation kv;
  /* The prefix this one extends (GitHub #187), or null for one that owns
   * every page of the head it covers.
   *
   * A sequence that resumed from retained state and prefilled past it has no
   * head of its own to publish: the pages below its generation opener are
   * partly the entry it claimed. It publishes a chained entry instead -- its
   * own new pages, plus the reference on the parent that the sequence held
   * until the publish, which *moves* here rather than being taken afresh. So
   * the chain is held by exactly one reference per link, and every iteration
   * of an agent's tool loop can leave a prompt checkpoint instead of only the
   * first (ADR 0029). */
  ignis_seq_prefix *parent = nullptr;
  /* The retained slot holding the image of every device-resident CLONE
   * section, one copy per prefix, not per claimant (GitHub #215). Held from
   * the publish until the publish handle is released -- nothing can claim the
   * prefix without that handle, so its image is unreachable from then on and
   * the slot goes back, while the pages live on under the sequences still
   * standing on them. -1 once released. */
  std::int32_t retained_slot = -1;
  /* What that image occupies: one slot's state and its hq residual window,
   * `ignis_seq_pool::retained_image_bytes`. */
  std::uint64_t image_bytes = 0;
  /* The IGNIS_SEQ_SECTION_PROGRESS payload: host scalars, so they live here
   * rather than in the retained slot above (the snapshot path writes them
   * with a plain memcpy for the same reason). */
  ignis_seq_progress_image progress{};
  /* Tokens of history the prefix covers, its chain included -- always
   * `ignis_seq_prefix_total_pages * kPagedKVPageSize`, which is what makes a
   * claimant's first write land on a page it owns. */
  std::uint32_t tokens = 0;
  /* Live holders: the publisher's handle, every claiming sequence, and a
   * chained child (GitHub #187). */
  std::uint32_t refcount = 0;
  /* What the clone actually cost, rather than what it was assumed to cost
   * (ADR 0024). Reported through ignis_seq_prefix_stats. */
  std::uint64_t clone_count      = 0;
  double last_clone_micros       = 0.0;
};

/* The KV pages the head `prefix` covers: its own and every ancestor's
 * (GitHub #187).
 *
 * This is what a claimant shares and what `ignis_seq::shared_pages` counts —
 * "how much history is warm", which the chain answers together. Who gives
 * which page back is a different question, answered by each link's own
 * `kv.mapped_page_count()`. */
inline std::uint32_t ignis_seq_prefix_total_pages(const ignis_seq_prefix &prefix) {
  std::uint32_t pages = 0;
  for (const ignis_seq_prefix *at = &prefix; at != nullptr; at = at->parent) {
    pages += at->kv.mapped_page_count();
  }
  return pages;
}

/* The CLONE sections a retained slot carries: those of `pool`'s state-section
 * table, in table order, each aligned to `kIgnisSeqSectionAlign`, with
 * offsets assigned from the start of a packed image.
 *
 * Derived from `ignis_seq_section_table` rather than restated, so a section
 * added there is carried without a second edit. Two rows of the table do not
 * appear:
 *
 *   - IGNIS_SEQ_SECTION_KV_PAGES, because it is SHAREABLE -- it is the half
 *     that is not copied at all;
 *   - IGNIS_SEQ_SECTION_PROGRESS, because its payload is host scalars
 *     (`ignis_seq_prefix::progress`) and a slot has no place for them.
 *
 * Every other CLONE section is device-resident by construction: it is state
 * a sequence's kernels write. A new one therefore lands here automatically,
 * and `ignis_seq_copy_slot_state` refuses it until it is given a case. */
inline std::vector<ignis_seq_section> ignis_seq_prefix_clone_layout(const ignis_seq_pool &pool) {
  std::vector<ignis_seq_section> image;
  std::uint64_t cursor = 0;
  for (const ignis_seq_section &section : ignis_seq_section_table(pool, 0)) {
    if (section.transfer != IGNIS_SEQ_SECTION_CLONE ||
        section.kind == IGNIS_SEQ_SECTION_PROGRESS) {
      continue;
    }
    ignis_seq_section placed = section;
    placed.offset            = cursor;
    image.push_back(placed);
    cursor = ignis_seq_align_up(cursor + placed.bytes, kIgnisSeqSectionAlign);
  }
  return image;
}

/* Capture `seq`'s mutable state into retained slot `retained_slot`, and its
 * progress scalars into `progress` (GitHub #215).
 *
 * One function for a prefix publish and a checkpoint capture, so the two
 * cannot drift apart about what a sequence is made of. Device to device and
 * synchronized: nothing crosses PCIe (ADR 0024), and nothing is allocated --
 * the slot was reserved at load (ADR 0030). The caller has checked the slot
 * with `ignis_seq_retained_slot_refusal`. */
inline void ignis_seq_capture_state(ignis_seq_pool &pool, const ignis_seq &seq,
                                    std::uint32_t retained_slot,
                                    ignis_seq_progress_image &progress) {
  ignis_seq_copy_slot_state(pool, seq.slot, ignis_seq_retained_pool_slot(pool, retained_slot));
  progress = ignis_seq_progress_of(seq);
  // GitHub #194: a clone carries no rope delta. The publisher's is its whole
  // prompt's, not the head's, and a text request claiming a multimodal
  // publisher's text-only head to its prompt's end runs no prefill span that
  // could set its own. A multimodal claimant always prefills a tail (ADR
  // 0029), and that span sets the delta.
  progress.rope_delta = 0;
}

/* The reverse: retained slot `retained_slot`'s state and `progress` become
 * `seq`'s -- what a claimant of a prefix or a checkpoint starts from. */
inline void ignis_seq_clone_state(ignis_seq_pool &pool, std::uint32_t retained_slot,
                                  const ignis_seq_progress_image &progress, ignis_seq &seq) {
  ignis_seq_copy_slot_state(pool, ignis_seq_retained_pool_slot(pool, retained_slot), seq.slot);
  ignis_seq_apply_progress(seq, progress);
}

/* Drop one reference to `prefix`, destroying it -- and returning its pages to
 * the pool it was published from -- when the last holder lets go. A null
 * `prefix` is a no-op.
 *
 * Defined in kernel/src/seq_prefix.cu and declared here because
 * ignis_seq_release has to call it: a released sequence is one holder fewer,
 * and nothing else in the leaf knows that. */
void ignis_seq_prefix_drop_reference(ignis_seq_prefix *prefix);

/* --- the materialized blob (GitHub #190) ----------------------------------
 *
 * A snapshot of a sequence holding a shared prefix, and a retained prompt
 * checkpoint spilled to KV-RAM, are both written as the blob ignis_seq_snapshot
 * writes for a sequence that owns its history: the shared pages are copied
 * into it rather than referenced. Defined in kernel/src/seq.cu and declared
 * here because kernel/src/seq_checkpoint.cu writes that blob too, and two
 * copies of its layout could drift apart without either failing. */

/* The offset of `kind` in `sections`. The table always carries every kind
 * (ignis_seq_section_table builds it unconditionally), so a miss is a
 * programming error rather than a caller's. */
std::uint64_t ignis_seq_section_offset(const std::vector<ignis_seq_section> &sections,
                                       int32_t kind);

/* Zero the bytes of a blob no section's payload covers, so two blobs of the
 * same state are the same bytes whatever buffer they were written into. */
void ignis_seq_zero_blob_gaps(unsigned char *base, const std::vector<ignis_seq_section> &sections,
                              std::uint64_t total_bytes);

/* One checked device-to-host copy on the default stream. */
void ignis_seq_copy_to_host(void *dst, const void *src, std::size_t bytes, const char *what);

/* The physical pages of a prefix chain, root first: the block-table order a
 * claimant of `head` addresses them in. Empty for a null `head`. */
std::vector<std::int32_t> ignis_seq_prefix_chain_page_ids(const ignis_seq_prefix *head);

/* Pack `pages` of `pool` into `dst` in the vendored snapshot layout (plane by
 * plane, page by page). Consecutive pages go in one copy. */
void ignis_seq_pack_pages_to_host(const ignis_seq_pool &pool,
                                  const std::vector<std::int32_t> &pages, void *dst);

/* Bytes a materialized blob of `pages` KV pages occupies. */
std::uint64_t ignis_seq_materialized_blob_bytes(const ignis_seq_pool &pool, std::uint32_t pages);

/* Write the blob of retained state that stands on `chain` -- plus
 * `tail_page`, the pool page holding a checkpoint's copy of the page its opener
 * ends inside, when not negative -- with retained slot `retained_slot`'s state
 * and `progress`. The layout ignis_seq_snapshot writes for a sequence standing
 * at the same point, so ignis_seq_restore takes it back. Throws on a short
 * `dst`, an extent that does not match the chain, or a failed device copy. */
void ignis_seq_write_materialized_blob(const ignis_seq_pool &pool, const ignis_seq_prefix *chain,
                                       std::int32_t tail_page, std::uint32_t retained_slot,
                                       const ignis_seq_progress_image &progress,
                                       std::uint32_t pages, void *dst, std::uint64_t dst_bytes);

#endif /* IGNIS_SEQ_PREFIX_INTERNAL_H */
