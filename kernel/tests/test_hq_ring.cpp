// GitHub #257 (spec runtime/06): the hq-e8-2b recent ring's validity-word
// masks -- OURS -- and the vendored kernel that applies them.
//
// The masks are pure host arithmetic (kernel/include/ignis_hq_ring.h), and
// their rule is the reference's (`invalidate_residual_ring`, ninfer
// decoder_state.cpp:226-246), so most of this file runs on the CPU. The last
// block applies a mask through `ninfer::apply_kv_ring_valid_words` and reads
// the words back: the kernel is 16 threads, but it is the only thing that
// ever clears a bit a verify round's rejected columns set.
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

bool bit(const ninfer::KvRingWords &words, std::uint32_t slot) {
  return ((words.w[slot / 32] >> (slot % 32)) & 1u) != 0;
}

std::uint32_t count(const ninfer::KvRingWords &words) {
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
    const ninfer::KvRingWords none = ignis_hq_ring_invalidate_mask(700, 700);
    expect(count(none) == kIgnisHqRecentKeys, "an empty range keeps every slot");

    // A verify round at frontier 1000 that wrote 8 columns and kept 3:
    // positions 1003..1007 are rejected, slots 491..495.
    const ninfer::KvRingWords rejected = ignis_hq_ring_invalidate_mask(1003, 1008);
    expect(count(rejected) == kIgnisHqRecentKeys - 5, "five rejected columns clear five slots");
    for (std::uint32_t key = 1003; key < 1008; ++key) {
      expect(!bit(rejected, key & 511), "a rejected column's slot is cleared");
    }
    expect(bit(rejected, 1002 & 511) && bit(rejected, 1008 & 511),
           "the committed column and the next position keep their slots");

    // The range wraps the ring: keys 1020..1030 are slots 508..511, 0..6.
    const ninfer::KvRingWords wrapped = ignis_hq_ring_invalidate_mask(1020, 1031);
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

  // ---- the apply kernel ----------------------------------------------------
  {
    std::uint32_t *words = nullptr;
    CUDA_FATAL(cudaMalloc(&words, kIgnisHqRingWords * sizeof(std::uint32_t)));
    CUDA_FATAL(cudaMemset(words, 0xff, kIgnisHqRingWords * sizeof(std::uint32_t)));
    ninfer::KvRingWords set{};
    set.w[3] = 0x10u;
    ninfer::apply_kv_ring_valid_words(words, ignis_hq_ring_invalidate_mask(1003, 1008), set,
                                      kIgnisHqRingWords, nullptr);
    CUDA_FATAL(cudaGetLastError());
    CUDA_FATAL(cudaDeviceSynchronize());
    ninfer::KvRingWords back{};
    CUDA_FATAL(cudaMemcpy(back.w, words, sizeof(back.w), cudaMemcpyDeviceToHost));
    ninfer::KvRingWords want = ignis_hq_ring_invalidate_mask(1003, 1008);
    want.w[3] |= 0x10u;
    bool same = true;
    for (int i = 0; i < kIgnisHqRingWords; ++i) {
      same = same && back.w[i] == want.w[i];
    }
    expect(same, "the device applies (words & and) | or, word for word");
    CUDA_FATAL(cudaFree(words));
  }

  if (failures != 0) {
    std::fprintf(stderr, "hq ring test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("hq ring test: ok\n");
  return 0;
}
