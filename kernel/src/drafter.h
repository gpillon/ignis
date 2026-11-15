#pragma once
// GitHub #619 (spec #618): the drafter a Qwen 3.8-27B load names -- what
// fills its verify round's proposal seam (`GLOSSARY.md`, "Drafter") -- OURS.
//
// Every choice a 27B load makes for its drafter follows this, each in one
// place: the weights it binds (`validate_and_bind`), its plan lines and their
// buffers (`plan_load_sizes`, `ignis_model_load`), the state sections its
// pool carries (`plan_pool_layout`), what a prefill chunk does for it
// (`run_program_chunk`) and what fills a round's proposal
// (`ignis_verify_graph_run_batch`, `run_verify_round`). None of them asks
// whether some drafter buffer happens to exist.
//
// Flash-Next's MTP never reaches here: its program is its own
// (kernel/src/flash_next/), and the 27B's load and its pool spec refuse
// IGNIS_SPECULATIVE_MTP by name.

#include "ignis_model.h"

#include <cstdint>

enum class IgnisDrafter : std::uint8_t {
  // No speculation: no verify round, and nothing drafts.
  kNone,
  // The verify substrate alone (IGNIS_SPECULATIVE_VERIFY_ONLY, P5-04): no
  // drafter is bound, and each round's drafts come from its caller -- a
  // test's fake drafter.
  kHost,
  // DFlash2 (P5-05): its weights, a drafter window in every pool slot, its
  // feature taps in every prefill chunk, and its forward inside the verify
  // pass, proposing each lane's drafts from that window.
  kDflash2,
};

// The drafter a 27B load, or a pool built for one, names by
// `speculative_backend` (enum ignis_speculative_backend).
inline IgnisDrafter ignis_drafter_of(std::int32_t speculative_backend) {
  switch (speculative_backend) {
  case IGNIS_SPECULATIVE_DFLASH2:
    return IgnisDrafter::kDflash2;
  case IGNIS_SPECULATIVE_VERIFY_ONLY:
    return IgnisDrafter::kHost;
  default:
    return IgnisDrafter::kNone;
  }
}

// Whether `drafter` proposes inside the leaf, so that a verify round takes
// no drafts from its caller.
inline bool ignis_drafter_proposes_in_the_leaf(IgnisDrafter drafter) {
  switch (drafter) {
  case IgnisDrafter::kDflash2:
    return true;
  case IgnisDrafter::kHost:
  case IgnisDrafter::kNone:
    return false;
  }
  return false;
}

// `drafter` as a refusal names it.
inline const char *ignis_drafter_name(IgnisDrafter drafter) {
  switch (drafter) {
  case IgnisDrafter::kDflash2:
    return "the DFlash2 drafter";
  case IgnisDrafter::kHost:
    return "the verify substrate alone";
  case IgnisDrafter::kNone:
    return "no speculation";
  }
  return "no speculation";
}
