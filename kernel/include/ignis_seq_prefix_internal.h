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
 *     entry keeps one device-resident image of them, captured at the
 *     prefix's end, and a claim copies it into the claimant's slot without
 *     the bytes ever leaving the card.
 *
 * Both halves are described by the **same state-section table** the snapshot
 * path walks (`ignis_seq_sections.h`): `IGNIS_SEQ_SECTION_SHAREABLE` is the
 * first bullet and `IGNIS_SEQ_SECTION_CLONE` is the second. That is what
 * makes "a section is carried by all three or by none" checkable rather than
 * remembered -- a section added to the table with no case in
 * `ignis_seq_prefix_transfer` below throws, exactly as it does in
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
  /* The device-resident image of every device-resident CLONE section, laid
   * out by `ignis_seq_prefix_clone_layout`. One copy per prefix, not per
   * claimant. */
  ignis_counted_device_buffer clone_image{IGNIS_ALLOC_PREFIX_IMAGE};
  /* The IGNIS_SEQ_SECTION_PROGRESS payload: host scalars, so they live here
   * rather than in the device image above (the snapshot path writes them
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

/* The device image's layout: the CLONE sections of `pool`'s state-section
 * table, in table order, each aligned to `kIgnisSeqSectionAlign`, with
 * offsets assigned from the start of the image.
 *
 * Derived from `ignis_seq_section_table` rather than restated, so a section
 * added there is sized and placed here without a second edit. Two rows of
 * the table do not appear:
 *
 *   - IGNIS_SEQ_SECTION_KV_PAGES, because it is SHAREABLE -- it is the half
 *     that is not copied at all;
 *   - IGNIS_SEQ_SECTION_PROGRESS, because its payload is host scalars
 *     (`ignis_seq_prefix::progress`) and a device image of them would be
 *     bytes nothing reads.
 *
 * Every other CLONE section is device-resident by construction: it is state
 * a sequence's kernels write. A new one therefore lands here automatically,
 * and `ignis_seq_prefix_transfer` refuses it until it is given a case. */
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

/* Bytes `ignis_seq_prefix_clone_layout` occupies. */
inline std::uint64_t ignis_seq_prefix_clone_bytes(const std::vector<ignis_seq_section> &image) {
  if (image.empty()) {
    return 0;
  }
  const ignis_seq_section &last = image.back();
  return ignis_seq_align_up(last.offset + last.bytes, kIgnisSeqSectionAlign);
}

/* Which way `ignis_seq_prefix_transfer` moves the mutable state. */
enum ignis_seq_prefix_direction {
  /* Capture: the publishing sequence's state becomes the prefix's image. */
  IGNIS_SEQ_PREFIX_CAPTURE = 0,
  /* Clone: the prefix's image becomes a claiming sequence's state. */
  IGNIS_SEQ_PREFIX_CLONE = 1
};

/* Move every mutable state section between `seq`'s slot and a device
 * `image` laid out by `ignis_seq_prefix_clone_layout`, in `direction`, and
 * the host-side progress scalars with it.
 *
 * The image is a plain pointer rather than a shared prefix because a prompt
 * checkpoint (GitHub #186, ADR 0029) captures the same sections into an
 * image of its own, at the generation opener rather than at the prefix's
 * page boundary. One function, so the two cannot drift apart about what a
 * sequence is made of.
 *
 * Both directions are device-to-device for the device-resident sections:
 * nothing here touches pinned host memory or crosses PCIe, which is the
 * whole point of cloning rather than restoring a sibling's snapshot (ADR
 * 0024). The copies are issued on the default stream; the caller
 * synchronizes.
 *
 * Throws `std::logic_error` for a CLONE section with no case below -- the
 * same loud failure ignis_seq_snapshot and ignis_seq_restore make, so that a
 * section added to the table is carried by all three or by none. */
inline void ignis_seq_state_transfer(ignis_seq_pool &pool, unsigned char *image,
                                     ignis_seq_progress_image &progress, ignis_seq &seq,
                                     ignis_seq_prefix_direction direction) {
  const bool capture = direction == IGNIS_SEQ_PREFIX_CAPTURE;
  const auto copy    = [&](void *device_state, void *image_at, std::size_t bytes,
                        const char *what) {
    if (bytes == 0) {
      return;
    }
    void *dst             = capture ? image_at : device_state;
    const void *src       = capture ? device_state : image_at;
    const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync(") + what +
                               ", device to device) failed: " + cudaGetErrorString(err));
    }
  };
  /* One 2D copy per section rather than one per layer.
   *
   * A slot's state is the same region of every layer's tensor, so the pool
   * side is `layer_count` rows of `per_layer` bytes at the pool's own layer
   * stride, and the image side is those rows packed -- exactly the shape
   * `LinearAttentionStatePool::pack_slot_to_host` moves to host memory, run
   * device-to-device here. It matters: at the 27B geometry the GDN sections
   * are 96 layer-slots, and 96 separate copies of 3 MiB are launch-bound
   * rather than bandwidth-bound (measured: 1.51 ms against ~0.25 ms for the
   * two 2D copies below -- see
   * docs/findings/2026-09-12-device-prefix-clone-cost.md).
   *
   * The layer stride is read off the pool's own layer tensors rather than
   * assumed: two adjacent layers' base pointers are exactly one stride
   * apart, and a single-layer pool has no stride to read, which is why it
   * takes the linear path. */
  const auto copy_gdn = [&](const ignis_seq_section &section, bool recurrent) {
    const std::uint32_t layers = pool.gdn_pool.layer_count();
    const std::size_t per_layer =
        recurrent ? pool.gdn_pool.recurrent_slot_bytes() : pool.gdn_pool.conv_slot_bytes();
    const std::vector<ninfer::Tensor> &planes =
        recurrent ? pool.gdn_pool.recurrent : pool.gdn_pool.conv;
    void *first = recurrent ? pool.gdn_pool.recurrent_slot(0, seq.slot).data
                            : pool.gdn_pool.conv_slot(0, seq.slot).data;
    unsigned char *packed = image + section.offset;
    if (layers <= 1) {
      copy(first, packed, per_layer, ignis_seq_section_name(section.kind));
      return;
    }
    const std::size_t layer_pitch = static_cast<std::size_t>(
        static_cast<const unsigned char *>(planes[1].data) -
        static_cast<const unsigned char *>(planes[0].data));
    void *dst          = capture ? static_cast<void *>(packed) : first;
    const void *src    = capture ? first : static_cast<const void *>(packed);
    const std::size_t dst_pitch = capture ? per_layer : layer_pitch;
    const std::size_t src_pitch = capture ? layer_pitch : per_layer;
    const cudaError_t err = cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, per_layer, layers,
                                              cudaMemcpyDeviceToDevice, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy2DAsync(") +
                               ignis_seq_section_name(section.kind) +
                               ", device to device) failed: " + cudaGetErrorString(err));
    }
  };

  for (const ignis_seq_section &section : ignis_seq_prefix_clone_layout(pool)) {
    switch (section.kind) {
    case IGNIS_SEQ_SECTION_GDN_CONV:
      copy_gdn(section, false);
      break;
    case IGNIS_SEQ_SECTION_GDN_RECURRENT:
      copy_gdn(section, true);
      break;
    case IGNIS_SEQ_SECTION_PENALTY_COUNTS:
      copy(pool.token_counts_for(seq.slot), image + section.offset,
           static_cast<std::size_t>(section.bytes), ignis_seq_section_name(section.kind));
      break;
    case IGNIS_SEQ_SECTION_DFLASH_WINDOW:
    case IGNIS_SEQ_SECTION_DFLASH_CHECKPOINT: {
      /* P5-03 (GitHub #152): the slot's lane of every drafter layer's K and
       * V, packed in the order `CyclicKVCache::copy_lane_to_host` packs it,
       * so this image and a snapshot blob lay the section out the same way.
       * Ten 4 MiB copies per section at the 27B geometry. */
      const ninfer::CyclicKVCache &cache = section.kind == IGNIS_SEQ_SECTION_DFLASH_WINDOW
                                               ? *pool.dflash2_window
                                               : *pool.dflash2_checkpoint;
      std::size_t cursor = 0;
      for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        const ninfer::CyclicKVCacheLayerView view = cache.layer_view(layer);
        for (const ninfer::Tensor *plane : {&view.k, &view.v}) {
          const ninfer::Tensor lane = plane->slice(3, seq.slot, 1);
          copy(lane.data, image + section.offset + cursor, lane.bytes(),
               ignis_seq_section_name(section.kind));
          cursor += lane.bytes();
        }
      }
      break;
    }
    default:
      /* ADR 0024's "carried by all three or by none", for the third
       * consumer. A CLONE section added to the table without a case here is
       * state a claimant would silently not receive. */
      throw std::logic_error(std::string("state section ") +
                             ignis_seq_section_name(section.kind) +
                             " has no device-to-device clone implementation");
    }
  }

  /* The progress scalars, whichever way we are going. They are host state,
   * so they are the one section this function moves with an assignment. */
  if (capture) {
    progress = ignis_seq_progress_of(seq);
    // GitHub #194: a clone carries no rope delta. The publisher's is its
    // whole prompt's, not the head's, and a text request claiming a
    // multimodal publisher's text-only head to its prompt's end runs no
    // prefill span that could set its own. A multimodal claimant always
    // prefills a tail (ADR 0029), and that span sets the delta.
    progress.rope_delta = 0;
  } else {
    ignis_seq_apply_progress(seq, progress);
  }
}

/* [`ignis_seq_state_transfer`] against a shared prefix's own image and
 * progress -- the original call, kept so every prefix call site reads
 * exactly as it did before a second consumer existed. */
inline void ignis_seq_prefix_transfer(ignis_seq_pool &pool, ignis_seq_prefix &prefix,
                                      ignis_seq &seq, ignis_seq_prefix_direction direction) {
  ignis_seq_state_transfer(pool, static_cast<unsigned char *>(prefix.clone_image.p),
                           prefix.progress, seq, direction);
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
 * plane, page by page), followed on every plane by that plane's slice of
 * `tail_page` when it is not null -- a checkpoint's copy of the page its
 * opener ends inside, packed the same way. Consecutive pages go in one copy. */
void ignis_seq_pack_pages_to_host(const ignis_seq_pool &pool,
                                  const std::vector<std::int32_t> &pages, const void *tail_page,
                                  void *dst);

/* Bytes a materialized blob of `pages` KV pages occupies. */
std::uint64_t ignis_seq_materialized_blob_bytes(const ignis_seq_pool &pool, std::uint32_t pages);

/* Write the blob of retained state that stands on `chain` -- plus
 * `tail_page`, a partial page copied beside it, when not null -- with `image`
 * (laid out by ignis_seq_prefix_clone_layout) and `progress` as its state.
 * The layout ignis_seq_snapshot writes for a sequence standing at the same
 * point, so ignis_seq_restore takes it back. Throws on a short `dst`, an
 * extent that does not match the chain, or a failed device copy. */
void ignis_seq_write_materialized_blob(const ignis_seq_pool &pool, const ignis_seq_prefix *chain,
                                       const void *tail_page, const void *image,
                                       const ignis_seq_progress_image &progress,
                                       std::uint32_t pages, void *dst, std::uint64_t dst_bytes);

#endif /* IGNIS_SEQ_PREFIX_INTERNAL_H */
