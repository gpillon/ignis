// ignis kernel leaf - GitHub #714 (spec #711): the mixed graphs a 27B load captures and plans -- OURS.
// Host-side: the switch that asks for them, the rule that picks the mixes, their bytes.
//
// A DFlash2 load of the NVFP4 body at 15 drafts holds a verify graph and a drafter append graph for
// each mix of kMixTable (kernel/src/round_layout.h) it can run, when mixed rounds are asked: `w + n`
// lanes within the load's decode lanes, and the base group's fold within the pool's GDN slots (the
// decode lanes and the device retained slots), its `2w` leading rows that commit nothing on slots no
// base lane holds, beside its `n` lanes. Every other load captures none, and its plan holds none. The
// VRAM plan's `mixed_graphs` line is the planned mixes' bytes, at the figure one mix's graphs were
// measured to take; crates/core/src/lane_lines.rs `mixed_graphs` mirrors it.
#pragma once

#include <cstdint>

#include "process_environment.h"
#include "round_layout.h"

namespace ignis::mixed_graphs {

// The one window a mixed round runs at: at any other no base record layout shares the extended
// one's stride over the backing both shapes' records share (`ignis_mixed_round_refusal`).
inline constexpr std::uint32_t kWindow = 15;

// The device memory a load's mixed graphs take: 5 MiB a mix (its verify graph of 1,215 to 1,260
// nodes and its append graph of 37 to 43, on the 27B's NVFP4 body at 8 lanes, BF16 or FP8 exception
// weights) and 4 MiB beside them, in whole 2 MiB pages. Measured on the RTX 5090 (2026-11-22, driver
// 610.88) as the device's free memory before each mix's capture less after it, the twelve in
// kMixTable's order: 8, 4, 6, 4, 4, 6, 6, 4, 6, 4, 4 and 6 MiB, 62 MiB in all, which 2 MiB beside
// them holds at every count from one mix to twelve; and on the process's WDDM counter, where the
// twelve's load held up to 2,088,960 B more past its plan than the same load without them, and the
// driver held 67,112,960 B more (64 MiB and 4 KiB) once each was torn down: one page more, so that
// the line is never short of what the process holds.
inline constexpr std::uint64_t kBytesPerMix = std::uint64_t{5} << 20;
inline constexpr std::uint64_t kFirstMixBytes = std::uint64_t{4} << 20;
inline constexpr std::uint64_t kPageBytes = std::uint64_t{2} << 20;

// The measurement switch IGNIS_MIXED_ROUNDS (ADR 0093's kind), read at each load and at each capture
// of its decode graphs: `1` asks for the mixed graphs of every mix the load holds, `0` for none.
// GitHub #715: unset, the shape policy's: asked where it is `mixed` (`policy_mixed`), never under
// `aggregate` or `all`. Throws std::invalid_argument, naming the variable, on any other value.
inline bool asked(bool policy_mixed) {
  return named_switch("IGNIS_MIXED_ROUNDS", policy_mixed, {{"0", false}, {"1", true}});
}

// GitHub #715: whether a load runs mixed rounds at all: DFlash2 (`dflash2`) on the NVFP4 body (not
// `trellis`, which runs none yet) at kWindow drafts. The shape policy `mixed` runs as `aggregate`
// anywhere else, and no mixed graph is planned or captured there.
inline bool holds_mixed_rounds(bool dflash2, bool trellis, std::uint32_t window) {
  return dflash2 && !trellis && window == kWindow;
}

// GitHub #718: the measurement switch IGNIS_MIXED_GRAPH_BUILD (ADR 0093's kind), read at each capture
// of a load with an extended shape: unset or `1`, the mixed rounds take the mixed graph's build (one
// root, the target argmax zeroed by a kernel, the extended group at the greatest stream priority,
// `ignis_model::mixed_graph_build`); `0`, the mixed rounds without it, the comparison a measurement
// of it needs. Throws std::invalid_argument, naming the variable, on any other value.
inline bool built() { return named_switch("IGNIS_MIXED_GRAPH_BUILD", true, {{"0", false}, {"1", true}}); }

// The mixes a load captures and plans (bit m for kMixTable[m]): where mixed rounds are `asked`, on a
// DFlash2 load (`dflash2`) of the NVFP4 body (not `trellis`, which runs no mixed round yet) at
// kWindow drafts, each mix whose lanes fit the load's `lanes` and whose base group's fold fits the
// pool's `gdn_slots`; 0 anywhere else.
inline std::uint32_t planned(bool asked, bool dflash2, bool trellis, std::uint32_t window, std::uint32_t lanes,
                             std::int64_t gdn_slots) {
  if (!asked || !holds_mixed_rounds(dflash2, trellis, window)) {
    return 0;
  }
  std::uint32_t mask = 0;
  for (std::uint32_t m = 0; m < round_layout::kMixes; ++m) {
    const std::uint32_t extended = round_layout::kMixTable[m][0];
    const std::uint32_t base = round_layout::kMixTable[m][1];
    if (extended + base <= lanes && static_cast<std::int64_t>(2 * extended + base) <= gdn_slots) {
      mask |= 1u << m;
    }
  }
  return mask;
}

// The bytes of the mixes `mask` names: the plan's `mixed_graphs` line.
inline std::uint64_t bytes_of(std::uint32_t mask) {
  std::uint64_t mixes = 0;
  for (std::uint32_t m = 0; m < round_layout::kMixes; ++m) {
    mixes += (mask >> m) & 1u;
  }
  if (mixes == 0) {
    return 0;
  }
  return (kFirstMixBytes + mixes * kBytesPerMix + kPageBytes - 1) / kPageBytes * kPageBytes;
}

} // namespace ignis::mixed_graphs
