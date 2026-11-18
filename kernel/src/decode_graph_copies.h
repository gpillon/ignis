// ignis kernel leaf: which of the 27B's decode and verify rounds run copy-free -- OURS (spec #672,
// GitHub #677, #680), leaf-internal.
//
// A copy-free round's traversal (`ignis_decode_graph_run_batch` and `ignis_verify_graph_run_batch`,
// kernel/src/decode_graph.cu) records no device-to-device copy where the copying round of earlier
// releases copies a residual or a GDN layer's q, k and v: every layer adds its output into its input residual in place, as the
// prefill's GDN body does since GitHub #507, instead of copying the residual into a second buffer
// first; and in the decode round each GDN layer's conv writes q, k and v as planes of their own
// (kernel/src/gdn_conv_split.h `split_conv_slots`, GitHub #680) past one column, where its output
// plane holds them interleaved by lane, and at one column, where that plane holds them contiguous,
// the recurrence reads them as its views (GitHub #677). The verify round's GDN body writes q, k and v
// apart already, so only its residual copies go. The arithmetic is the same, so a round's tokens
// are the copying round's byte for byte (the round identity test,
// crates/core/tests/round_identity_gpu.rs); the
// decode graphs lose their 208 copy nodes and the verify graphs 64 of theirs, the drafter's keeping
// the rest.
//
// Decided at each capture of the decode graphs, from the card and the measurement switch
// IGNIS_DECODE_GRAPH_COPIES, for every round after it, replayed or eager:
//   - compute capability major 12 (the RTX 5090, the RTX PRO 6000), where the round identity test
//     ran, takes it at every width 1-8 and in the verify round, unless the switch names the
//     copying rounds: `1` all of them, `wide` those past width 1 and the verify round;
//   - every other card keeps the copying rounds whatever the switch says, until the same test has
//     run on it.
#pragma once

#include "ignis_step.h"
#include "process_environment.h"

#include <cstdint>

namespace ignis::decode_graph_copies {

// What the measurement switch IGNIS_DECODE_GRAPH_COPIES names at a capture: unset (or empty), the
// card decides; `1`, the copying rounds on any card; `wide`, the copying rounds past width 1
// and in the verify round, the width-1 round as unset (GitHub #680: the share of widths 2-8 and the
// verify round apart from width 1's).
enum class Switch { Unset, Copies, Wide };

// The switch in the process environment (kernel/src/process_environment.h), read at each capture.
// Throws std::invalid_argument, naming the variable, on any value but `1` or `wide`.
inline Switch switch_at_capture() {
  return ignis::named_switch("IGNIS_DECODE_GRAPH_COPIES", Switch::Unset,
                             {{"1", Switch::Copies}, {"wide", Switch::Wide}});
}

// The cards that take the copy-free round: compute capability major 12.
constexpr bool runs_on(int compute_capability_major) {
  return compute_capability_major == 12;
}

// The one rule for the decode round (true: copy-free): the round at `width`, on a card it runs on,
// the switch naming the copying round neither at every width nor, past width 1, under `wide`.
constexpr bool copy_free(Switch named, int compute_capability_major, std::uint32_t width) {
  return runs_on(compute_capability_major) &&
         (named == Switch::Unset || (named == Switch::Wide && width == 1));
}

// The same rule for the verify round, at every width: on a card it runs on, the switch unset.
constexpr bool verify_copy_free(Switch named, int compute_capability_major) {
  return runs_on(compute_capability_major) && named == Switch::Unset;
}

// The decode widths 1..IGNIS_DECODE_MAX_BATCH the rule makes copy-free, bit `width - 1`.
constexpr std::uint32_t copy_free_widths(Switch named, int compute_capability_major) {
  std::uint32_t widths = 0;
  for (std::uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    if (copy_free(named, compute_capability_major, width)) {
      widths |= 1u << (width - 1);
    }
  }
  return widths;
}

// Whether `width` is among `widths` (`copy_free_widths`).
constexpr bool takes(std::uint32_t widths, std::uint32_t width) {
  return width >= 1 && width <= IGNIS_DECODE_MAX_BATCH && ((widths >> (width - 1)) & 1u) != 0;
}

} // namespace ignis::decode_graph_copies
