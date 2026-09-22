// GitHub #257 (spec runtime/06): the hq-e8-2b recent ring's validity-word
// masks and the kernel that applies them -- OURS, not vendored.
//
// The masks are pure host arithmetic (kernel/include/ignis_hq_ring.h), and
// their rule is the reference's (`invalidate_residual_ring` /
// `revalidate_residual_ring`, ninfer decoder_state.cpp:200-246), so most of
// this file runs on the CPU. The last block applies a mask on the device and
// reads the words back: the kernel is 16 threads, but it is the only thing
// that ever clears a bit a verify round's rejected columns set.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test.

#include "ignis_hq_ring.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label);
    ++failures;
  }
}

bool bit(const ignis_hq_ring_words &words, std::uint32_t slot) {
  return ((words.w[slot / 32] >> (slot % 32)) & 1u) != 0;
}

std::uint32_t count(const ignis_hq_ring_words &words) {
  std::uint32_t n = 0;
  for (std::uint32_t slot = 0; slot < kIgnisHqRecentKeys; ++slot) {
    n += bit(words, slot) ? 1u : 0u;
  }
  return n;
}

#define CUDA_FATAL(expr)                                                                        \
  do {                                                                                          \
    const cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));          \
      std::exit(EXIT_FAILURE);                                                                  \
    }                                                                                           \
  } while (0)

} // namespace

int main() {
  // ---- invalidate: the slots a range of keys was appended to --------------
  {
    const ignis_hq_ring_words none = ignis_hq_ring_invalidate_mask(700, 700);
    expect(count(none) == kIgnisHqRecentKeys, "an empty range keeps every slot");

    // A verify round at frontier 1000 that wrote 8 columns and kept 3:
    // positions 1003..1007 are rejected, slots 491..495.
    const ignis_hq_ring_words rejected = ignis_hq_ring_invalidate_mask(1003, 1008);
    expect(count(rejected) == kIgnisHqRecentKeys - 5, "five rejected columns clear five slots");
    for (std::uint32_t key = 1003; key < 1008; ++key) {
      expect(!bit(rejected, key & 511), "a rejected column's slot is cleared");
    }
    expect(bit(rejected, 1002 & 511) && bit(rejected, 1008 & 511),
           "the committed column and the next position keep their slots");

    // The range wraps the ring: keys 1020..1030 are slots 508..511, 0..6.
    const ignis_hq_ring_words wrapped = ignis_hq_ring_invalidate_mask(1020, 1031);
    expect(count(wrapped) == kIgnisHqRecentKeys - 11, "a range across the wrap clears its 11 slots");
    expect(!bit(wrapped, 511) && !bit(wrapped, 0) && !bit(wrapped, 6) && bit(wrapped, 7) &&
               bit(wrapped, 507),
           "the wrapped range's ends");

    // More than one lap clears everything, once.
    expect(count(ignis_hq_ring_invalidate_mask(40, 40 + 2000)) == 0,
           "a range longer than the ring clears every slot");
    expect(count(ignis_hq_ring_invalidate_mask(40, 40 + 512)) == 0,
           "exactly one lap clears every slot");
  }

  // ---- revalidate: a trim back to an earlier frontier --------------------
  {
    // No trim: every slot's last writer below 2000 is in [1488, 2000).
    expect(count(ignis_hq_ring_revalidate_mask(2000, 2000)) == kIgnisHqRecentKeys,
           "a frontier where the appends ended keeps every slot");
    // Nothing retained: every bit goes (the reference's full reset).
    expect(count(ignis_hq_ring_revalidate_mask(0, 0)) == 0, "a full reset keeps no slot");
    // Trimmed from 2000 back to 1800: slot r's last writer is the largest key
    // below 2000 congruent to r. Keys [1800, 2000) now name positions past
    // the frontier -- 200 slots -- and every other last writer lies in
    // [1488, 1800), inside the new window [1288, 1800).
    const ignis_hq_ring_words trimmed = ignis_hq_ring_revalidate_mask(2000, 1800);
    expect(count(trimmed) == kIgnisHqRecentKeys - 200, "a 200-key trim drops the 200 slots past it");
    expect(!bit(trimmed, 1800 & 511) && !bit(trimmed, 1999 & 511) && bit(trimmed, 1799 & 511) &&
               bit(trimmed, 1488 & 511),
           "the trimmed slots are exactly the keys past the new frontier");
    // Trimmed further than a lap: no slot's last writer is inside the window.
    expect(count(ignis_hq_ring_revalidate_mask(4000, 1000)) == 0,
           "a trim of more than one lap keeps no slot");
    // A short history: 300 keys appended, nothing trimmed -- slots 300..511
    // have no writer at all.
    expect(count(ignis_hq_ring_revalidate_mask(300, 300)) == 300,
           "only the slots a key was appended to are kept");
  }

  // ---- the apply kernel ----------------------------------------------------
  {
    std::uint32_t *words = nullptr;
    CUDA_FATAL(cudaMalloc(&words, kIgnisHqRingWords * sizeof(std::uint32_t)));
    CUDA_FATAL(cudaMemset(words, 0xff, kIgnisHqRingWords * sizeof(std::uint32_t)));
    ignis_hq_ring_words set{};
    set.w[3] = 0x10u;
    CUDA_FATAL(ignis_hq_ring_apply(words, ignis_hq_ring_invalidate_mask(1003, 1008), set, nullptr));
    CUDA_FATAL(cudaDeviceSynchronize());
    ignis_hq_ring_words back{};
    CUDA_FATAL(cudaMemcpy(back.w, words, sizeof(back.w), cudaMemcpyDeviceToHost));
    ignis_hq_ring_words want = ignis_hq_ring_invalidate_mask(1003, 1008);
    want.w[3] |= 0x10u;
    bool same = true;
    for (int i = 0; i < kIgnisHqRingWords; ++i) {
      same = same && back.w[i] == want.w[i];
    }
    expect(same, "the device applies (words & and) | or, word for word");
    expect(ignis_hq_ring_apply(nullptr, ignis_hq_ring_all(), set, nullptr) == cudaSuccess,
           "a null row (a BF16 pool) is a no-op");
    CUDA_FATAL(cudaFree(words));
  }

  if (failures != 0) {
    std::fprintf(stderr, "hq ring test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("hq ring test: ok\n");
  return 0;
}
