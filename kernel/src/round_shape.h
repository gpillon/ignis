// ignis kernel leaf - GitHub #687 (spec #684): a DFlash2 load's two round shapes, the measurement
// switch that names the shape of its rounds, and the one-lane attention of an extended round -- OURS.
// Pure functions of what the host holds before a round launches: no CUDA and no model, so a
// host-only test holds them from a table (kernel/tests/test_one_lane_chunks.cpp).
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

// Whether a batch round's attention runs its one lane in chunks of the small-T tile
// (`small_t_tile` columns: 8 over hq-e8-2b, 6 over BF16) instead of through the vendored wrapper: an
// extended round's one lane alone, whose 9..16 columns the wrapper's resolver would send to its
// prompt route -- a workspace sized for the whole context, which a decode load does not reserve. At
// two lanes or more the wrapper chunks by the tile itself; every round of the base shape, and every
// round of a load without an extended shape, is the wrapper's, on every KV format.
inline bool one_lane_in_chunks(bool extended_round, std::int32_t lanes, std::int32_t lane_columns,
                               std::int32_t small_t_tile) {
  return extended_round && lanes == 1 && lane_columns > small_t_tile;
}

} // namespace ignis::round_shape
