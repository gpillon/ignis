// ignis kernel leaf - GitHub #687 (spec #684): a DFlash2 load's two round shapes and the measurement
// switch that names the shape of its rounds -- OURS. Pure functions of what the host holds before a
// round launches: no CUDA and no model. An extended round's attention is
// kernel/src/extended_attention.h (GitHub #688).
//
// A DFlash2 load past IGNIS_BASE_SHAPE_DRAFT_TOKENS drafts holds two round shapes on one set of
// buffers (kernel/src/model_internal.h `ignis_model::extended_shape`): its base shape, 8 columns a
// lane at every lane count, and its extended shape, the load's window + 1 columns a lane at up to
// IGNIS_EXTENDED_SHAPE_MAX_LANES lanes. A round has one shape for all its lanes, chosen before it
// launches and handed to it (kernel/src/step.cu `ignis_program_decode`).
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

// What IGNIS_ROUND_SHAPE names: every round at the base shape (`base`, or unset), or every round the
// extended shape can take at it (`extended`).
enum class Named : std::uint8_t { kBase, kExtended };

// IGNIS_ROUND_SHAPE, read at each verify round of a load with an extended shape (a measurement switch,
// ADR 0093's kind: one process flips it between rounds). Throws std::invalid_argument, naming the
// variable, on a value other than `base` or `extended`.
inline Named read_switch() {
  return ignis::named_switch("IGNIS_ROUND_SHAPE", Named::kBase,
                             {{"base", Named::kBase}, {"extended", Named::kExtended}});
}

// Whether a verify round of `lanes` lanes runs at the extended shape: under `named` extended, on a load
// that has one (`extended_lanes` its most lanes, 0 without one), at no more lanes than it runs, and
// with no lane whose commit the host cuts (`host_cut`: such a round keeps the base shape).
inline bool runs_extended(Named named, std::uint64_t lanes, std::uint32_t extended_lanes, bool host_cut) {
  return named == Named::kExtended && extended_lanes != 0 && lanes <= extended_lanes && !host_cut;
}

} // namespace ignis::round_shape
