/* ignis kernel leaf: the prompt checkpoint entry and the one transfer the
 * shared-prefix mechanism cannot do (GitHub #186, ADR 0029).
 *
 * A prompt checkpoint is a shared prefix **plus the rest of the way to the
 * generation opener**. A prefix stops at a whole page because a partial page
 * would be written by its owner and read by its claimants at once; an opener
 * lands wherever the rendered prompt puts it, which is almost never on a
 * page boundary. So a checkpoint is:
 *
 *   - one reference to the shared prefix holding the whole pages below the
 *     opener (shared in place, charged to the pool once, as always);
 *   - its own image of the mutable state **at the opener**, laid out by the
 *     same `ignis_seq_prefix_clone_layout` a prefix's image is -- the
 *     prefix's own image stands up to 63 tokens short of it;
 *   - a copy of the partial page the opener ends inside, which the capturing
 *     sequence is still writing and so can never share.
 *
 * Nothing here decides *what* a sequence is made of: that is the one
 * state-section table (`ignis_seq_sections.h`), read through
 * `ignis_seq_prefix_internal.h`. A CLONE section added there is carried by
 * this path for free, and refused loudly if it has no copy case -- the same
 * "carried by all three or by none" ADR 0024 asks of the snapshot and clone
 * paths, now for a fourth consumer.
 *
 * Not part of the public flat C ABI: `ignis_seq.h` keeps
 * `ignis_seq_checkpoint` opaque and exposes only capture / claim / release /
 * stats / image-bytes.
 */
#ifndef IGNIS_SEQ_CHECKPOINT_INTERNAL_H
#define IGNIS_SEQ_CHECKPOINT_INTERNAL_H

#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"

#include <cstdint>
#include <stdexcept>
#include <string>

/* A captured prompt checkpoint.
 *
 * It has no refcount of its own, unlike `ignis_seq_prefix`. A prefix is held
 * by every sequence standing on it, so it needs one; a checkpoint is held by
 * exactly one caller handle and hands out *copies*, so a claim adds no
 * holder and there is nothing to count. What a claim does add is a holder of
 * the **prefix** underneath, which `ignis_seq_alloc_shared`'s own refcount
 * already tracks. */
struct ignis_seq_checkpoint {
  /* The shared prefix holding the whole pages below the opener. One
   * reference, taken at capture and dropped at release: it is what keeps
   * those pages alive once every live request has gone. */
  ignis_seq_prefix *prefix = nullptr;
  /* The device image of every device-resident CLONE section, at the opener
   * rather than at the prefix's page boundary. */
  ignis_counted_device_buffer image{IGNIS_ALLOC_CHECKPOINT_IMAGE};
  /* A copy of the physical KV page the opener ends inside, packed plane by
   * plane the way `pack_paged_kv_allocation_to_host` packs one page -- so
   * this image and a snapshot blob's KV section lay a page out the same way. */
  ignis_counted_device_buffer tail_page{IGNIS_ALLOC_CHECKPOINT_TAIL_PAGE};
  /* The IGNIS_SEQ_SECTION_PROGRESS payload: host scalars, so they live here
   * rather than in a device image (the snapshot path does the same). */
  ignis_seq_progress_image progress{};
  /* The generation opener, in tokens. NOT a whole number of pages. */
  std::uint32_t tokens = 0;
  /* What a claim actually cost, rather than what it was assumed to cost (ADR
   * 0024). Reported through ignis_seq_checkpoint_stats. */
  std::uint64_t claim_count = 0;
  double last_claim_micros  = 0.0;
};

/* Bytes one physical KV page occupies across every plane of `pool`, packed.
 *
 * Read off the vendored pool rather than derived from the format, because
 * the pool is what knows its plane set: BF16 has two planes per GQA layer
 * and hq-e8-2b four, and a checkpoint must not have to know which. */
inline std::uint64_t ignis_seq_checkpoint_page_bytes(const ignis_seq_pool &pool) {
  return static_cast<std::uint64_t>(ninfer::paged_kv_host_image_bytes(pool.kv_pool, 1));
}

/* Device bytes one checkpoint of `pool` occupies: the mutable-state image
 * plus one page's copy. A pool property, not a per-checkpoint one -- every
 * checkpoint of one pool costs exactly this. */
inline std::uint64_t ignis_seq_checkpoint_image_bytes_of(const ignis_seq_pool &pool) {
  return ignis_seq_prefix_clone_bytes(ignis_seq_prefix_clone_layout(pool)) +
         ignis_seq_checkpoint_page_bytes(pool);
}

/* Copy one physical KV page between the pool's planes and a packed device
 * image, in `direction`.
 *
 * The pool's plane order is PageMajor (`kernel/src/seq.cu`'s
 * `ignis_seq_pool_create` fixes it), which is what makes a page one
 * contiguous run per plane and this a handful of copies rather than a
 * strided gather. A pool built any other way is refused rather than copied
 * wrongly -- the geometry is an invariant of this leaf, and the day it stops
 * being one this throws instead of silently shuffling bytes.
 *
 * The copies are issued on the default stream; the caller synchronizes. */
inline void ignis_seq_checkpoint_page_transfer(ignis_seq_pool &pool, std::int32_t page_id,
                                               unsigned char *image,
                                               ignis_seq_prefix_direction direction) {
  if (pool.kv_pool.plane_order() != ninfer::PagedKVPlaneOrder::PageMajor) {
    throw std::logic_error(
        "prompt checkpoints require a PageMajor paged-KV pool: a page is one contiguous run "
        "per plane there, and nothing else in this leaf builds one any other way");
  }
  const bool capture      = direction == IGNIS_SEQ_PREFIX_CAPTURE;
  unsigned char *packed   = image;
  const std::size_t count = pool.kv_pool.plane_count();
  // The bound is checked *before* each copy is enqueued, not after the loop:
  // an out-of-bounds `cudaMemcpyAsync` that has already been issued is not
  // something a later throw can take back. The sizing function and this loop
  // agree today by construction -- both read the PageMajor page stride off
  // each plane -- so this is here for the day one of them stops, and a
  // refused capture is a bet not taken where an overrun is someone else's
  // memory.
  const std::uint64_t budget = ignis_seq_checkpoint_page_bytes(pool);
  for (std::size_t index = 0; index < count; ++index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(index);
    const std::size_t bytes     = static_cast<std::size_t>(plane.nb[3]);
    if (static_cast<std::uint64_t>(packed - image) + bytes > budget) {
      throw std::logic_error("prompt checkpoint tail page: plane " + std::to_string(index) +
                             " would move past the " + std::to_string(budget) +
                             " bytes the pool prices a page at; the page layout and its "
                             "sizing have drifted apart");
    }
    unsigned char *page =
        static_cast<unsigned char *>(plane.data) + static_cast<std::int64_t>(page_id) * plane.nb[3];
    void *dst             = capture ? static_cast<void *>(packed) : static_cast<void *>(page);
    const void *src       = capture ? static_cast<const void *>(page)
                                    : static_cast<const void *>(packed);
    const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync(KV tail page, device to device) "
                                           "failed: ") +
                               cudaGetErrorString(err));
    }
    packed += bytes;
  }
  // And the whole page has to have been moved, not only part of one: a plane
  // set that shrank would otherwise leave the rest of the image stale.
  const std::uint64_t moved = static_cast<std::uint64_t>(packed - image);
  if (moved != budget) {
    throw std::logic_error("prompt checkpoint tail page: moved " + std::to_string(moved) +
                           " bytes for a page the pool prices at " + std::to_string(budget) +
                           "; the page layout and its sizing have drifted apart");
  }
}

/* Allocate a sequence against `prefix`, cloning the prefix's own mutable
 * image into it only when `clone_prefix_state` is set.
 *
 * `ignis_seq_alloc_shared` is this with the clone on. A prompt checkpoint's
 * claim is this with it off, because the checkpoint's image -- captured
 * further along, at the opener -- is written over the slot immediately
 * afterwards, and cloning the prefix's first would be ~0.25 ms of bytes
 * nothing ever reads.
 *
 * Defined in kernel/src/seq_prefix.cu beside the entry point it factors.
 * Throws nothing: it sets the leaf's last error and returns non-zero. */
int32_t ignis_seq_alloc_against_prefix(ignis_seq_pool *pool, std::uint32_t context_tokens,
                                       ignis_seq_prefix *prefix, bool clone_prefix_state,
                                       ignis_seq **out_seq);

#endif /* IGNIS_SEQ_CHECKPOINT_INTERNAL_H */
