// ignis kernel leaf - GitHub #674, #689 (spec #684): the round timing's samples, attributed to the
// round shape of the round they time -- OURS. No CUDA here: kernel/src/step.cu reads the GPU clock's
// events around each replayed round and hands this their elapsed times, so a host-only test holds
// the attribution from a table (kernel/tests/test_round_shape.cpp).
#pragma once

#include "ignis_step.h"

#include <array>
#include <cstddef>
#include <vector>

namespace ignis::round_timing {

// Two rounds of a kind that begin this many ms apart or more have a request or a prefill between
// them: the earlier one's period and gap are left out.
inline constexpr float kPausedMs = 50.0f;

// One block of one round shape's samples, in us.
struct Block {
  std::vector<float> period;
  std::vector<float> span;
  std::vector<float> gap;
};

// The blocks of a kind of round, one per round shape (`IGNIS_ROUND_SHAPES_MAX`): the base shape's
// first, which is every plain round's and every verify round's of a load of one shape. GitHub #713:
// and one more after them, `kMixedBlock`, every mixed round's whatever its mix.
inline constexpr std::size_t kMixedBlock = IGNIS_ROUND_SHAPES_MAX;
using Blocks = std::array<Block, IGNIS_ROUND_SHAPES_MAX + 1>;

// One timed round at round shape `shape`, and the next timed round of its kind, launched
// `to_next_ms` after this one's launch and `gap_ms` after its end: the round's span (the event
// before its graph to the one after) goes in its shape's block, and so do its period (its launch to
// the next one's) and its gap (its end to the next one's launch), unless the two began kPausedMs
// or more apart. The period and the gap are the round's own whatever shape the next round is, so
// where shapes alternate a shape's periods are the time its rounds took and the periods of a run of
// rounds sum to its wall time; charging a round with the time since the round before it would give
// a shape its neighbour's span.
inline void add(Blocks &blocks, std::size_t shape, float span_ms, float to_next_ms, float gap_ms) {
  Block &block = blocks[shape < blocks.size() ? shape : 0];
  block.span.push_back(span_ms * 1000.0f);
  if (to_next_ms < kPausedMs) {
    block.period.push_back(to_next_ms * 1000.0f);
    block.gap.push_back(gap_ms * 1000.0f);
  }
}

} // namespace ignis::round_timing
