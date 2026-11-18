// ignis kernel leaf: which of the 27B's decode rounds run copy-free -- OURS (spec #672, GitHub #677),
// leaf-internal.
//
// A copy-free round's traversal (`ignis_decode_graph_run_batch`, kernel/src/decode_graph.cu) records
// no device-to-device copy where main's copies: every layer adds its output into its input residual
// in place, as the prefill's GDN body does since GitHub #507, instead of copying the residual into
// a second buffer first; and at one column each GDN layer's recurrence reads q, k and v as views of
// the conv output plane, where they are contiguous already, instead of three copies out of it. The
// arithmetic is the same, so a round's tokens are main's byte for byte (the round identity test,
// crates/core/tests/round_identity_gpu.rs); the width-1 graph loses its 208 copy nodes.
//
// Decided at each capture of the decode graphs, from the card and the measurement switch
// IGNIS_DECODE_GRAPH_COPIES, for every round after it, replayed or eager:
//   - compute capability major 12 (the RTX 5090, the RTX PRO 6000), where the round identity test
//     ran, takes it at width 1, unless the switch names main's copying round;
//   - every other card keeps main's whatever the switch says, until the same test has run on it;
//   - widths 2-8 and the verify round keep main's on every card.
#pragma once

#include "ignis_step.h"
#include "process_environment.h"

#include <cstdint>

namespace ignis::decode_graph_copies {

// What the measurement switch IGNIS_DECODE_GRAPH_COPIES names at a capture: unset (or empty), the
// card decides; `1`, main's copying round on any card.
enum class Switch { Unset, Copies };

// The switch in the process environment (kernel/src/process_environment.h), read at each capture.
// Throws std::invalid_argument, naming the variable, on any value but `1`.
inline Switch switch_at_capture() {
  return ignis::named_switch("IGNIS_DECODE_GRAPH_COPIES", Switch::Unset, {{"1", Switch::Copies}});
}

// The cards that take the copy-free round: compute capability major 12.
constexpr bool runs_on(int compute_capability_major) {
  return compute_capability_major == 12;
}

// The one rule (true: copy-free): the decode round at `width`, on a card it runs on, the switch not
// naming main's round.
constexpr bool copy_free(Switch named, int compute_capability_major, std::uint32_t width) {
  return named == Switch::Unset && runs_on(compute_capability_major) && width == 1;
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
