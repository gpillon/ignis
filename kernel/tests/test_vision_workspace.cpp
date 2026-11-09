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

  // --- 2. an encode over Flash-Next's prefill scratch and expert staging ring --
  // GitHub #558 (ADR 0087): a Flash-Next load lends an item's encode two areas
  // idle while it runs, its prefill scratch and, on a card that cannot hold
  // every expert, its expert staging ring. The workspace and a host tower's
  // three staging slots go where they fit, both in the scratch first.
  {
    constexpr std::size_t kMiB = 1 << 20;
    constexpr std::size_t kSlots = 3 * 25 * kMiB;
    const auto placed = [](std::size_t scratch, std::size_t ring, std::size_t workspace, std::size_t slots,
                           bool workspace_in_ring, bool slots_in_ring) {
      const VisionEncodePlacement p = ignis_vision_encode_placement(scratch, ring, workspace, slots);
      return p.fits && p.workspace_in_ring == workspace_in_ring && p.slots_in_ring == slots_in_ring;
    };
    const auto refused = [](std::size_t scratch, std::size_t ring, std::size_t workspace, std::size_t slots) {
      return !ignis_vision_encode_placement(scratch, ring, workspace, slots).fits;
    };
    // The slots follow the workspace at the next 256-byte boundary.
    check(placed(1024 + kSlots, 0, 1000, kSlots, false, false), "both in a scratch of the aligned workspace and the slots");
    check(refused(1023 + kSlots, 0, 1000, kSlots), "a scratch one byte short, and no ring, holds no encode");
    check(placed(1000, kSlots, 1000, kSlots, false, true), "the slots in the ring when the scratch holds the workspace alone");
    check(placed(80 * kMiB, 1500 * kMiB, 100 * kMiB, kSlots, true, false),
          "the workspace in the ring when the scratch holds the slots alone");
    check(placed(10 * kMiB, 1500 * kMiB, 100 * kMiB, kSlots, true, true), "both in the ring when the scratch holds neither");
    check(refused(10 * kMiB, 100 * kMiB, 100 * kMiB, kSlots), "neither area holds the workspace and the slots");
    // A tower in VRAM stages nothing: the workspace alone, in the scratch, else the ring.
    check(placed(100, 0, 100, 0, false, false), "device: the scratch");
    check(placed(99, 100, 100, 0, true, false), "device: the ring when the scratch is short");
    check(refused(99, 99, 100, 0), "device: neither");
    // ADR 0087's figure: the RTX 5090's ring at the 2026-11-09 start holds a
    // 16,384-token item's workspace -- the processor's largest -- on its own.
    constexpr std::size_t kRtx5090Ring = 1'574'895'616;
    const std::size_t largest = ignis_vision_workspace_bytes(16384, kVisionMaxSegments);
    check(placed(kSlots, kRtx5090Ring, largest, kSlots, true, false),
          "a 16,384-token item's workspace fits the RTX 5090's ring, the slots a 75 MiB scratch: workspace " +
              std::to_string(largest));
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "vision workspace test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("vision workspace test: ok\n");
  return 0;
}
