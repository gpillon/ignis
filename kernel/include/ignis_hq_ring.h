/* ignis kernel leaf: the masks the hq-e8-2b recent ring's validity words are
 * cleared with (GitHub #257, spec runtime/06). OURS; the kernel that applies
 * them is the vendored `ninfer::apply_kv_ring_valid_words`
 * (kernel/vendor/src/core/kv_ring_bits.{h,cu}), and the rule is the
 * reference's `PagedKVCache::invalidate_residual_ring`
 * (ninfer `targets/qwen3_6/impl/state/decoder_state.cpp`).
 *
 * A key `k >= 32` of a sequence lives in ring slot `k & 511` of its slot
 * row's residual window, and the vendored hq kernels read that slot as the
 * key's exact row iff the slot's bit is set. **The bit carries no
 * position**: it says "this slot holds the row of the last key appended
 * congruent to it", and an append sets it. So whenever columns are appended
 * past what the sequence keeps -- a verify round that rejects its drafts, or
 * one that fails after its pass -- the bits those columns set must be
 * cleared, or a later fetch reads a rejected token's row as an older key's
 * exact one with no error anywhere.
 *
 * The masks are computed here, on the host, and handed to the vendored apply
 * kernel by value: `words[i] = (words[i] & and.w[i]) | or.w[i]`. No bit is
 * ever recomputed on the device, so the rule has one definition, testable on
 * the CPU (kernel/tests/test_hq_ring.cpp), and the apply launches outside
 * every captured graph -- its masks differ from one round to the next.
 *
 * Clearing a bit is always safe: the slot falls back to the codec, whose
 * rows are complete for every appended key.
 */
#ifndef IGNIS_HQ_RING_H
#define IGNIS_HQ_RING_H

#include "ignis_seq_internal.h"

#include "core/kv_ring_bits.h"

#include <cstdint>

static_assert(sizeof(ninfer::KvRingWords::w) / sizeof(std::uint32_t) ==
                  static_cast<std::size_t>(kIgnisHqRingWords),
              "the vendored KvRingWords no longer holds one slot row's ring words");

/* Every bit set: the AND mask that keeps everything. */
inline ninfer::KvRingWords ignis_hq_ring_all() {
  ninfer::KvRingWords words{};
  for (std::uint32_t &word : words.w) {
    word = ~0u;
  }
  return words;
}

/* The AND mask that clears the ring slots keys `[first_key, end_key)` were
 * appended to -- what a verify round applies over the columns it wrote and
 * did not commit. At most one lap of the ring: a range of 512 keys or more
 * clears every slot. The reference's `invalidate_residual_ring`. */
inline ninfer::KvRingWords ignis_hq_ring_invalidate_mask(std::uint64_t first_key,
                                                         std::uint64_t end_key) {
  ninfer::KvRingWords keep = ignis_hq_ring_all();
  for (std::uint64_t key = first_key;
       key < end_key && key < first_key + static_cast<std::uint64_t>(kIgnisHqRecentKeys); ++key) {
    const std::uint32_t r = static_cast<std::uint32_t>(key) & (kIgnisHqRecentKeys - 1);
    keep.w[r / 32] &= ~(1u << (r % 32));
  }
  return keep;
}

#endif /* IGNIS_HQ_RING_H */
