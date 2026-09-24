// The vision encoder's workspace size -- OURS, not vendored.
//
// `ignis_vision_workspace_bytes` sizes the arena one media item is encoded
// in, and at the default envelope it is the largest single line of the VRAM
// plan beside the weights: the 2026-09-23 serving load reported
// `workspace_bytes` 2,219,837,184 for 32,768 merged tokens (131,072 patches,
// 384 segments). What is pinned here:
//
// 1. The BF16 patch plane is read once, by the patch embedding, and is dead
//    before the first block runs, so the blocks reuse its bytes: the same
//    item costs exactly one patch plane (131,072 x 1,536 x 2 = 402,653,184
//    bytes) less than that load reserved.
//
// Host-only: the size is layout arithmetic, no device call. It reaches into
// `kernel/src` for `model_internal.h`, a leaf-internal header, the way the
// RoPE scaling test does.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE.

#include "model_internal.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

constexpr std::int32_t kDefaultEnvelopeTokens = 32768;
constexpr std::uint64_t kServingLoadWorkspace = 2'219'837'184ULL;
constexpr std::uint64_t kPatchPlaneBytes =
    static_cast<std::uint64_t>(kDefaultEnvelopeTokens) * kVisionMergeUnit * kVisionPatchDim * 2;

} // namespace

int main() {
  // --- 1. the patch plane is not live across the blocks ----------------------
  {
    check(kPatchPlaneBytes == 402'653'184ULL, "the patch plane of a 32,768-token item");
    const std::uint64_t bytes = ignis_vision_workspace_bytes(kDefaultEnvelopeTokens, kVisionMaxSegments);
    check(bytes == kServingLoadWorkspace - kPatchPlaneBytes,
          "a 32,768-token item's workspace is the serving load's less one patch plane: got " +
              std::to_string(bytes));
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "vision workspace test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("vision workspace test: ok\n");
  return 0;
}
