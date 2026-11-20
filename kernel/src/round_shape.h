// ignis kernel leaf - GitHub #687, #689 (spec #684): a DFlash2 load's two round shapes, the shape
// rule and the shape policy that choose each round's, and the measurement switch that forces one --
// OURS. Pure functions of what the host holds before a round launches: no CUDA and no model, so a
// host-only test holds them from tables (kernel/tests/test_round_shape.cpp). An extended round's
// attention is kernel/src/extended_attention.h (GitHub #688).
//
// A DFlash2 load past IGNIS_BASE_SHAPE_DRAFT_TOKENS drafts holds two round shapes on one set of
// buffers (kernel/src/model_internal.h `ignis_model::extended_shape`): its base shape, 8 columns a
// lane at every lane count, and its extended shape, the load's window + 1 columns a lane at up to
// IGNIS_EXTENDED_SHAPE_MAX_LANES lanes. A round has one shape for all its lanes, chosen before it
// launches and handed to it (kernel/src/step.cu `ignis_program_decode`): the shape rule says what
// each lane asks from its own earlier rounds (`asks_extended`), which the host has before the next
// round launches (the deferred verify's ticket, GitHub #676), and the shape policy `all` runs the
// round extended when every lane in it asks (`runs_extended`).
#pragma once

#include "ignis_model.h"
#include "process_environment.h"

#include <cstdint>

namespace ignis::round_shape {

// Whether a load of `backend` at a draft window of `draft_tokens` holds an extended shape beside its
// base shape: DFlash2 alone, past the base shape's window.
inline bool has_extended_shape(int32_t backend, uint32_t draft_tokens) {
  return backend == IGNIS_SPECULATIVE_DFLASH2 && draft_tokens > IGNIS_BASE_SHAPE_DRAFT_TOKENS;
}

// GitHub #689: the shape rule's thresholds, a load's (`ignis_model_load_options`, the server's
// `spec.shape_enter_accepted` and `spec.shape_stay_accepted`).
struct ShapeRule {
  // The drafts a full base round must take for its lane to ask the extended shape: 7, every one.
  std::uint32_t enter_accepted = 7;
  // The drafts an extended round must take for its lane to keep asking it.
  std::uint32_t stay_accepted = 6;
};

// The shape rule: what a lane asks of its next round after one at the extended shape (`extended`)
// or the base shape, whose extent was `extent` drafts of which the target accepted `accepted`, the
// lane having asked `asked` of it. A lane with no history has not asked. After a base round it asks
// when the round drafted the base shape's whole window and the target took at least
// `enter_accepted` of it; a base round the lane's budget or context cut short says nothing, and the
// lane keeps what it asked. After an extended round it asks while the round took at least
// `stay_accepted`, whether or not it asked (a round forced extended counts as one). The target's
// accept, never what a stop or a host cut kept: the rule reads the drafts' quality.
inline bool asks_extended(const ShapeRule &rule, bool asked, bool extended, std::int32_t extent,
                          std::int32_t accepted) {
  if (extended) {
    return accepted >= static_cast<std::int32_t>(rule.stay_accepted);
  }
  if (extent < static_cast<std::int32_t>(IGNIS_BASE_SHAPE_DRAFT_TOKENS)) {
    return asked;
  }
  return accepted >= static_cast<std::int32_t>(rule.enter_accepted);
}

// The field of `rule` a load whose extended round drafts `window` tokens refuses, or nullptr: an
// entry threshold outside 1..IGNIS_BASE_SHAPE_DRAFT_TOKENS, which no full base round could meet (or
// every one would, at 0), or a stay threshold outside 1..`window`, which no extended round could
// meet (or every one would). Past the window the rule would run one extended round after each full
// base round and never a second, with no word said.
inline const char *refused_threshold(const ShapeRule &rule, std::int32_t window) {
  if (rule.enter_accepted < 1 || rule.enter_accepted > IGNIS_BASE_SHAPE_DRAFT_TOKENS) {
    return "shape_enter_accepted";
  }
  if (rule.stay_accepted < 1 || static_cast<std::int64_t>(rule.stay_accepted) > window) {
    return "shape_stay_accepted";
  }
  return nullptr;
}

// What IGNIS_ROUND_SHAPE names: unset, each round at the shape the rule and the policy give it
// (`kRule`); `base`, every round at the base shape; `extended`, every round the extended shape can
// take at it.
enum class Named : std::uint8_t { kRule, kBase, kExtended };

// IGNIS_ROUND_SHAPE, read at each verify round of a load with an extended shape (a measurement switch,
// ADR 0093's kind: one process flips it between rounds). Throws std::invalid_argument, naming the
// variable, on a value other than `base` or `extended`.
inline Named read_switch() {
  return ignis::named_switch("IGNIS_ROUND_SHAPE", Named::kRule,
                             {{"base", Named::kBase}, {"extended", Named::kExtended}});
}

// Whether a verify round of `lanes` lanes runs at the extended shape, `asks[i]` nonzero where lane i
// asks it: on a load that has one (`extended_lanes` its most lanes, 0 without one), at no more lanes
// than it runs, with no lane whose commit the host cuts (`host_cut`: such a round keeps the base
// shape), and then by `named`: unset, the shape policy `all`, the round extended when every lane in
// it asks, so that no lane pays for another; under `extended` every such round; under `base` none.
inline bool runs_extended(Named named, const std::uint8_t *asks, std::uint64_t lanes, std::uint32_t extended_lanes,
                          bool host_cut) {
  if (named == Named::kBase || extended_lanes == 0 || lanes > extended_lanes || host_cut) {
    return false;
  }
  if (named == Named::kExtended) {
    return true;
  }
  for (std::uint64_t i = 0; i < lanes; ++i) {
    if (asks[i] == 0) {
      return false;
    }
  }
  return true;
}

} // namespace ignis::round_shape
