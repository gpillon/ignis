// The windowed transfer's test helpers (spec vram-budget/03): a blob taken a
// window at a time, and a restore fed one -- shared by test_seq_snapshot.cpp
// (the 27B's pools, both formats) and test_seq_flash_next_sections.cpp
// (Flash-Next's), so both hold the same calls to the same rule: windows laid
// end to end are the whole call's bytes. OURS.
#ifndef IGNIS_SEQ_WINDOW_TEST_COMMON_H
#define IGNIS_SEQ_WINDOW_TEST_COMMON_H

#include "ignis_seq.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

// The window sizes every blob is cut at: one sector, the disk tier's own
// 32 MiB staging window, and a size that divides nothing.
inline const std::vector<std::uint64_t> &seq_window_sizes() {
  static const std::vector<std::uint64_t> sizes = {4096, 32ull << 20, 4096 * 3 + 1000};
  return sizes;
}

// A `total`-byte blob taken a `window` at a time by `take(offset, bytes, dst,
// transfer)`, the windows issued last to first (a snapshot's windows may come
// in any order), then fenced. Every byte starts as 0xCD, so a gap a window
// left unwritten shows. Returns an empty vector when a call or the fence
// failed.
inline std::vector<unsigned char> seq_take_windows(
    ignis_seq_pool *pool, std::uint64_t total, std::uint64_t window,
    const std::function<int32_t(unsigned char *, std::uint64_t, const ignis_seq_transfer &)> &take) {
  std::vector<unsigned char> blob(static_cast<std::size_t>(total), 0xCD);
  void *stream = ignis_seq_pool_transfer_stream(pool);
  std::vector<std::uint64_t> offsets;
  for (std::uint64_t offset = 0; offset < total; offset += window) {
    offsets.push_back(offset);
  }
  std::reverse(offsets.begin(), offsets.end());
  for (const std::uint64_t offset : offsets) {
    const std::uint64_t bytes = std::min(window, total - offset);
    const ignis_seq_transfer transfer{offset, bytes, total, stream};
    if (take(blob.data() + offset, bytes, transfer) != 0) {
      return {};
    }
  }
  std::uint64_t fence = 0;
  if (ignis_seq_pool_fence(pool, stream, &fence) != 0 || ignis_seq_pool_fence_wait(pool, fence) != 0) {
    return {};
  }
  return blob;
}

// `blob` fed into `seq` a `window` at a time, in order, then fenced. Returns
// the first non-zero code a window returned (and stops there), or 0.
inline int32_t seq_restore_windows(ignis_seq_pool *pool, ignis_seq *seq, const std::vector<unsigned char> &blob,
                                   std::uint64_t window) {
  void *stream = ignis_seq_pool_transfer_stream(pool);
  const std::uint64_t total = blob.size();
  for (std::uint64_t offset = 0; offset < total; offset += window) {
    const std::uint64_t bytes = std::min(window, total - offset);
    const ignis_seq_transfer transfer{offset, bytes, total, stream};
    const int32_t rc = ignis_seq_restore(pool, seq, blob.data() + offset, bytes, &transfer);
    if (rc != 0) {
      return rc;
    }
  }
  std::uint64_t fence = 0;
  if (ignis_seq_pool_fence(pool, stream, &fence) != 0 || ignis_seq_pool_fence_wait(pool, fence) != 0) {
    return -1;
  }
  return 0;
}

#endif /* IGNIS_SEQ_WINDOW_TEST_COMMON_H */
