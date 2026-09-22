/* ignis kernel leaf: the hq-e8-2b recent ring's validity words (GitHub #257,
 * spec runtime/06). OURS, not vendored: the reference keeps the same rule in
 * its host code (ninfer `core/kv_ring_bits.{h,cu}` and
 * `PagedKVCache::{re,in}validate_residual_ring` in
 * `targets/qwen3_6/impl/state/decoder_state.cpp`), not in a vendored kernel.
 *
 * A key `k >= 32` of a sequence lives in ring slot `k & 511` of its slot
 * row's residual window, and the vendored hq kernels read that slot as the
 * key's exact row iff the slot's bit is set. **The bit carries no
 * position**: it says "this slot holds the row of the last key appended
 * congruent to it", and an append sets it. So whenever the sequence's
 * frontier stops being the end of what was appended -- a verify round that
 * rejects its drafts, a trim back to an earlier point -- the bits that now
 * name a row the sequence did not keep must be cleared, or the kernels read
 * a rejected token's row as an older key's exact one with no error anywhere.
 *
 * The masks are computed here, on the host, and applied by a one-block
 * kernel that takes them by value: `words[i] = (words[i] & and.w[i]) |
 * or.w[i]`. No bit is ever recomputed on the device, so the rule has one
 * definition, testable on the CPU (kernel/tests/test_hq_ring.cpp), and the
 * apply launches outside every captured graph -- its masks differ from one
 * round to the next.
 *
 * Clearing a bit is always safe: the slot falls back to the codec, whose
 * rows are complete for every appended key. Setting one is never done here.
 */
#ifndef IGNIS_HQ_RING_H
#define IGNIS_HQ_RING_H

#include "ignis_seq_internal.h"

#include <cuda_runtime_api.h>

#include <cstdint>

/* One slot row's validity words, by value: what the host computes and the
 * apply kernel takes (the reference's `KvRingWords`). */
struct ignis_hq_ring_words {
  std::uint32_t w[kIgnisHqRingWords];
};

/* Every bit set: the AND mask that keeps everything. */
inline ignis_hq_ring_words ignis_hq_ring_all() {
  ignis_hq_ring_words words{};
  for (std::uint32_t &word : words.w) {
    word = ~0u;
  }
  return words;
}

/* The AND mask that clears the ring slots keys `[first_key, end_key)` were
 * appended to -- what a verify round applies over the columns it wrote and
 * did not commit. At most one lap of the ring: a range of 512 keys or more
 * clears every slot. The reference's `invalidate_residual_ring`. */
inline ignis_hq_ring_words ignis_hq_ring_invalidate_mask(std::uint64_t first_key,
                                                         std::uint64_t end_key) {
  ignis_hq_ring_words keep = ignis_hq_ring_all();
  for (std::uint64_t key = first_key;
       key < end_key && key < first_key + static_cast<std::uint64_t>(kIgnisHqRecentKeys); ++key) {
    const std::uint32_t r = static_cast<std::uint32_t>(key) & (kIgnisHqRecentKeys - 1);
    keep.w[r / 32] &= ~(1u << (r % 32));
  }
  return keep;
}

/* The AND mask for a sequence trimmed back from `retained_keys` appended
 * keys to a frontier at `base`: slot r stays valid only if its last writer
 * before the trim -- the largest key below `retained_keys` congruent to r --
 * lies inside the new recent window `[base - 512, base)`. The reference's
 * `revalidate_residual_ring`.
 *
 * No production path trims a live sequence in ignis: a prefix, a checkpoint
 * and a snapshot are each captured at exactly the point a claimant or a
 * restore resumes from, so the image's bits are already the ones this would
 * keep. It is here as the rule the lifecycle tests hold the device to. */
inline ignis_hq_ring_words ignis_hq_ring_revalidate_mask(std::uint64_t retained_keys,
                                                         std::uint64_t base) {
  const auto ring = static_cast<std::int64_t>(kIgnisHqRecentKeys);
  const auto window = static_cast<std::int64_t>(base);
  const auto retained = static_cast<std::int64_t>(retained_keys);
  ignis_hq_ring_words keep{};
  for (std::int64_t r = 0; r < ring; ++r) {
    if (retained == 0 || retained - 1 < r) {
      continue;
    }
    const std::int64_t key = r + (retained - 1 - r) / ring * ring;
    if (key >= window - ring && key < window) {
      keep.w[r / 32] |= 1u << (r % 32);
    }
  }
  return keep;
}

/* Apply `and_mask` then `or_mask` to the `kIgnisHqRingWords` words at
 * `words` (device memory), on `stream`. Asynchronous; the caller orders and
 * synchronizes it like any other launch. A null `words` is a no-op -- the
 * pool of a BF16 load has no ring. Returns the launch's error. Defined in
 * kernel/src/hq_ring.cu. */
cudaError_t ignis_hq_ring_apply(std::uint32_t *words, const ignis_hq_ring_words &and_mask,
                                const ignis_hq_ring_words &or_mask, cudaStream_t stream);

#endif /* IGNIS_HQ_RING_H */
