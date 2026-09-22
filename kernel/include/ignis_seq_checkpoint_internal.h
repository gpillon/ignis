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
 *   - its own image of the mutable state **at the opener**, in a retained
 *     slot of its own (GitHub #215) -- the prefix's image stands up to 63
 *     tokens short of it;
 *   - a copy of the partial page the opener ends inside, which the capturing
 *     sequence is still writing and so can never share, in one KV page of
 *     the pool (GitHub #215).
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
 * stats / snapshot.
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
  /* The retained slot holding the image of every device-resident CLONE
   * section, at the opener rather than at the prefix's page boundary (GitHub
   * #215). Held from the capture until the release. */
  std::int32_t retained_slot = -1;
  /* What that image occupies: one slot's state and its hq residual window
   * (`ignis_seq_pool::retained_image_bytes`). */
  std::uint64_t image_bytes = 0;
  /* The page the opener ends inside, copied into one KV page of the pool this
   * entry owns (GitHub #215) -- never bound to a block-table row, and not
   * shared. Empty (`valid()` false) for an opener on a page boundary, which
   * ends inside no page. */
  ninfer::PagedKVAllocation tail;
  /* The IGNIS_SEQ_SECTION_PROGRESS payload: host scalars, so they live here
   * rather than in a retained slot (the snapshot path does the same). */
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

/* Copy physical KV page `src_page` over `dst_page`, every plane, device to
 * device.
 *
 * The pool's plane order is PageMajor (`kernel/src/seq.cu`'s
 * `ignis_seq_pool_create` fixes it), which is what makes a page one
 * contiguous run per plane and this a handful of copies rather than a
 * strided gather. A pool built any other way is refused rather than copied
 * wrongly -- the geometry is an invariant of this leaf, and the day it stops
 * being one this throws instead of silently shuffling bytes.
 *
 * The copies are issued on the default stream; the caller synchronizes. */
inline void ignis_seq_copy_kv_page(ignis_seq_pool &pool, std::int32_t src_page,
                                   std::int32_t dst_page) {
  if (pool.kv_pool.plane_order() != ninfer::PagedKVPlaneOrder::PageMajor) {
    throw std::logic_error(
        "prompt checkpoints require a PageMajor paged-KV pool: a page is one contiguous run "
        "per plane there, and nothing else in this leaf builds one any other way");
  }
  for (std::size_t index = 0; index < pool.kv_pool.plane_count(); ++index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(index);
    auto *base                  = static_cast<unsigned char *>(plane.data);
    const cudaError_t err       = cudaMemcpyAsync(
        base + static_cast<std::int64_t>(dst_page) * plane.nb[3],
        base + static_cast<std::int64_t>(src_page) * plane.nb[3],
        static_cast<std::size_t>(plane.nb[3]), cudaMemcpyDeviceToDevice, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync(KV tail page, device to device) "
                                           "failed: ") +
                               cudaGetErrorString(err));
    }
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
