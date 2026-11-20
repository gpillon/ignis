// ignis kernel leaf - GitHub #687, #689, #690 (spec #684): a DFlash2 load's two round shapes, the
// shape rule and the shape policies that choose each round's, and the measurement switch that forces
// one -- OURS. Pure functions of what the host holds before a round launches: no CUDA and no model,
// so a host-only test holds them from tables (kernel/tests/test_round_shape.cpp). An extended
// round's attention is kernel/src/extended_attention.h (GitHub #688).
//
// A DFlash2 load past IGNIS_BASE_SHAPE_DRAFT_TOKENS drafts holds two round shapes on one set of
// buffers (kernel/src/model_internal.h `ignis_model::extended_shape`): its base shape, 8 columns a
// lane at every lane count, and its extended shape, the load's window + 1 columns a lane at up to
// IGNIS_EXTENDED_SHAPE_MAX_LANES lanes. A round has one shape for all its lanes, chosen before it
// launches and handed to it (kernel/src/step.cu `ignis_program_decode`): the shape rule says what
// each lane asks from its own earlier rounds (`asks_extended`), which the host has before the next
// round launches (the deferred verify's ticket, GitHub #676), and the shape policy turns the asks
// into the round's shape (`runs_extended`): `aggregate`, the default, runs it extended when what the
// asking lanes are expected to gain from their own recent rounds is more than what the wider round
// costs every lane in it (`policy_extends`); `all` only when every lane asks.
#pragma once

#include "ignis_model.h"
#include "process_environment.h"

#include <algorithm>
#include <array>
#include <cstddef>
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

// GitHub #690: a lane's recent verify rounds as the shape policy reads them, one byte a round
// (`record_round`): the drafts the target accepted, with kExtendedRound where the round ran at the
// extended shape. A ring of the widest window's rounds (`ignis_seq::shape_rounds`), of which
// `recorded` were ever written, the newest at `recorded - 1`.
using RecentRounds = std::array<std::uint8_t, IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS>;
inline constexpr std::uint8_t kExtendedRound = 0x80;

// What a lane's recent rounds say, summed over a window of them (`estimate_of`): its rounds, the
// tokens each would commit at the base shape (its accepts up to IGNIS_BASE_SHAPE_DRAFT_TOKENS, and
// the target's own token), its extended rounds among them and their accepts past
// IGNIS_BASE_SHAPE_DRAFT_TOKENS: the drafts the wider round gave the lane.
struct LaneEstimate {
  std::uint32_t rounds = 0;
  std::uint32_t tokens = 0;
  std::uint32_t extended_rounds = 0;
  std::uint32_t extra_tokens = 0;
};

// Records a verify round of a lane: whether it ran `extended`, its extent and the drafts the target
// accepted (the rule's reading, not what a stop or a host cut kept). A round the lane's budget or
// context cut under the base window says nothing of the drafts, as it says nothing to the rule, and
// is not recorded.
inline void record_round(RecentRounds &rounds, std::uint64_t &recorded, bool extended, std::int32_t extent,
                         std::int32_t accepted) {
  if (extent < static_cast<std::int32_t>(IGNIS_BASE_SHAPE_DRAFT_TOKENS)) {
    return;
  }
  const std::int32_t taken = std::clamp<std::int32_t>(accepted, 0, IGNIS_DFLASH2_MAX_DRAFT_TOKENS);
  rounds[recorded % rounds.size()] = static_cast<std::uint8_t>(taken) | (extended ? kExtendedRound : 0);
  ++recorded;
}

// A lane's estimate over its `estimate_rounds` most recent recorded rounds (all of them, when fewer).
inline LaneEstimate estimate_of(const RecentRounds &rounds, std::uint64_t recorded, std::uint32_t estimate_rounds) {
  LaneEstimate estimate{};
  const std::uint64_t window =
      std::min<std::uint64_t>({recorded, static_cast<std::uint64_t>(estimate_rounds), rounds.size()});
  for (std::uint64_t back = 1; back <= window; ++back) {
    const std::uint8_t round = rounds[(recorded - back) % rounds.size()];
    const std::uint32_t accepted = round & static_cast<std::uint8_t>(~kExtendedRound);
    estimate.rounds += 1;
    estimate.tokens += std::min<std::uint32_t>(accepted, IGNIS_BASE_SHAPE_DRAFT_TOKENS) + 1;
    if ((round & kExtendedRound) != 0) {
      estimate.extended_rounds += 1;
      estimate.extra_tokens += accepted > IGNIS_BASE_SHAPE_DRAFT_TOKENS ? accepted - IGNIS_BASE_SHAPE_DRAFT_TOKENS : 0;
    }
  }
  return estimate;
}

// GitHub #690: the shape policy a load runs (`ignis_model_load_options::shape_policy`,
// `shape_estimate_rounds`, `shape_period_ratios`; the server's `spec.` fields of the same names).
enum class Policy : std::uint8_t { kAggregate, kAll };

struct ShapePolicy {
  Policy policy = Policy::kAggregate;
  // The recorded rounds a lane's estimate reads, its most recent. 16: a lane whose extended rounds
  // do not pay is granted one again once its window no longer holds one, so a lane beside it that
  // does not ask pays one extended round in 17 for it -- about 1% of its rate at two lanes, where an
  // extended round costs it about 19% -- and a lane whose text turns predictable waits at most 16
  // rounds behind an estimate from before. Provisional, as the ratios are.
  std::uint32_t estimate_rounds = 16;
  // The extended round's period over the base round's, in thousandths, at 1..4 lanes: measured on
  // the RTX 5090 at 1, 2 and 4 lanes before the cost levers (1.15, 1.23, 1.31), 3 lanes the line
  // between 2 and 4 -- provisional until measured with the build they ship in. The one-lane entry
  // is not read: a lane alone runs at the shape it asks.
  std::array<std::uint32_t, IGNIS_EXTENDED_SHAPE_MAX_LANES> period_ratios = {1150, 1230, 1270, 1310};
};

// The shape policy a load's options give (`policy`, IGNIS_SHAPE_POLICY_*; `estimate_rounds`, 0 the
// default; `period_ratios`, IGNIS_EXTENDED_SHAPE_MAX_LANES thousandths, 0 the default of that lane
// count) in `out`, or the field a load refuses: a policy that is neither, a window past the rounds a
// lane keeps, a ratio under 1 (an extended round costing less than a base one) or past 2 (more than
// twice, for at most twice the columns).
inline const char *policy_of(std::uint32_t policy, std::uint32_t estimate_rounds, const std::uint32_t *period_ratios,
                             ShapePolicy &out) {
  ShapePolicy resolved{};
  if (policy != IGNIS_SHAPE_POLICY_AGGREGATE && policy != IGNIS_SHAPE_POLICY_ALL) {
    return "shape_policy";
  }
  resolved.policy = policy == IGNIS_SHAPE_POLICY_ALL ? Policy::kAll : Policy::kAggregate;
  if (estimate_rounds > IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS) {
    return "shape_estimate_rounds";
  }
  if (estimate_rounds != 0) {
    resolved.estimate_rounds = estimate_rounds;
  }
  for (std::size_t lanes = 0; lanes < resolved.period_ratios.size(); ++lanes) {
    const std::uint32_t ratio = period_ratios[lanes];
    if (ratio != 0 && (ratio < 1000 || ratio > 2000)) {
      return "shape_period_ratios";
    }
    if (ratio != 0) {
      resolved.period_ratios[lanes] = ratio;
    }
  }
  out = resolved;
  return nullptr;
}

// The shape policy over a round of `lanes` lanes (1..IGNIS_EXTENDED_SHAPE_MAX_LANES), `asks[i]`
// nonzero where lane i asks the extended shape, `estimates[i]` its recent rounds. `all`: extended
// when every lane asks. `aggregate`: at one lane, the lane's ask. At two or more, extended when a
// lane asks whose window holds no extended round -- granted, so that its estimate has one to read --
// and otherwise when the drafts the asking lanes are expected to take past the base window exceed
// what the wider round costs every lane in it:
//
//   sum over asking lanes of expected_extra_tokens
//       > (the period ratio at `lanes` - 1) * sum over all lanes of expected_tokens
//
// a lane's expected tokens being its rounds' mean at the base shape (a lane with no round yet the
// full base round's IGNIS_BASE_SHAPE_DRAFT_TOKENS + 1, the most the wider round can cost it), its
// expected extra tokens its extended rounds' mean past the base window. In integers, each mean in
// thousandths of a token, so a mirror of it elsewhere decides every round the same.
inline bool policy_extends(const ShapePolicy &policy, const std::uint8_t *asks, const LaneEstimate *estimates,
                           std::uint64_t lanes) {
  if (lanes == 0 || lanes > policy.period_ratios.size()) {
    return false;
  }
  if (policy.policy == Policy::kAll) {
    for (std::uint64_t i = 0; i < lanes; ++i) {
      if (asks[i] == 0) {
        return false;
      }
    }
    return true;
  }
  if (lanes == 1) {
    return asks[0] != 0;
  }
  bool any = false;
  for (std::uint64_t i = 0; i < lanes; ++i) {
    if (asks[i] != 0 && estimates[i].extended_rounds == 0) {
      return true;
    }
    any = any || asks[i] != 0;
  }
  if (!any) {
    return false;
  }
  std::uint64_t gain = 0;
  std::uint64_t tokens = 0;
  for (std::uint64_t i = 0; i < lanes; ++i) {
    const LaneEstimate &e = estimates[i];
    tokens += e.rounds == 0 ? (IGNIS_BASE_SHAPE_DRAFT_TOKENS + 1) * 1000ull
                            : static_cast<std::uint64_t>(e.tokens) * 1000 / e.rounds;
    if (asks[i] != 0) {
      gain += static_cast<std::uint64_t>(e.extra_tokens) * 1000 / e.extended_rounds;
    }
  }
  const std::uint64_t ratio = policy.period_ratios[lanes - 1];
  return gain * 1000 > (ratio - 1000) * tokens;
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
// than it runs, and then by `named`: unset, the shape policy (`policy_extends`, `estimates[i]` lane
// i's recent rounds); under `extended` every such round; under `base` none.
// GitHub #691: a lane whose commit the host cuts (a grammar's) asks like any other, and its round
// takes the shape this gives it: the cut reads the run at the round's own stride.
inline bool runs_extended(Named named, const std::uint8_t *asks, std::uint64_t lanes, std::uint32_t extended_lanes,
                          const ShapePolicy &policy, const LaneEstimate *estimates) {
  if (named == Named::kBase || extended_lanes == 0 || lanes > extended_lanes) {
    return false;
  }
  if (named == Named::kExtended) {
    return true;
  }
  return policy_extends(policy, asks, estimates, lanes);
}

} // namespace ignis::round_shape
