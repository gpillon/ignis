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
//
// GitHub #715 (spec #711): the shape policy `mixed` grants the extended shape lane by lane, so a
// round may carry both (a mixed round, kernel/src/round_layout.h): `grant` gives each lane its shape
// under every policy, from a cost table by lane count and extended lanes. Its twin is
// crates/core/src/shape_grant.rs `lane_shapes`, the definition the CPU tests hold to their tables;
// kernel/tests/test_round_shape.cpp holds this one to the same rows.
#pragma once

#include "ignis_model.h"
#include "mixed_graphs.h"
#include "process_environment.h"
#include "round_layout.h"

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

// The defaults a load's 0 stands for in the shape rule's and the shape policy's fields
// (`ignis_model_load_options`), and the range of a period ratio: the server's, declared in
// crates/server/src/config/schema.rs from ignis_core's (`ShapeRule`, `ShapeEstimate`), to which a
// CPU test holds these lines (crates/core/src/speculation.rs,
// `the_leafs_shape_defaults_are_the_cores`). One constant a line, `name = value;`, as it reads them.
inline constexpr std::uint32_t kDefaultEnterAccepted = 7;
inline constexpr std::uint32_t kDefaultStayAccepted = 6;
inline constexpr std::uint32_t kDefaultEstimateRounds = 16;
inline constexpr std::array<std::uint32_t, IGNIS_EXTENDED_SHAPE_MAX_LANES> kDefaultPeriodRatios = {1128, 1290, 1363, 1344};
// GitHub #715: the cost table's mixed entries, in kMixTable's order (crates/core/src/shape_grant.rs
// `PeriodTable::NVFP4`, where each is said to be measured or derived).
inline constexpr std::array<std::uint32_t, round_layout::kMixes> kDefaultMixedPeriodRatios = {1155, 1120, 1138, 1077, 1093, 1107, 1285, 1222, 1207, 1241, 1384, 1336};
inline constexpr std::uint32_t kPeriodRatioMinMilli = 1000;
inline constexpr std::uint32_t kPeriodRatioMaxMilli = 2000;

// GitHub #689: the shape rule's thresholds, a load's (`ignis_model_load_options`, the server's
// `spec.shape_enter_accepted` and `spec.shape_stay_accepted`).
struct ShapeRule {
  // The drafts a full base round must take for its lane to ask the extended shape: every one.
  std::uint32_t enter_accepted = kDefaultEnterAccepted;
  // The drafts an extended round must take for its lane to keep asking it.
  std::uint32_t stay_accepted = kDefaultStayAccepted;
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
enum class Policy : std::uint8_t { kAggregate, kAll, kMixed };

struct ShapePolicy {
  Policy policy = Policy::kAggregate;
  // The recorded rounds a lane's estimate reads, its most recent. 16: a lane whose extended rounds
  // do not pay is granted one again once its window no longer holds one, so a lane beside it that
  // does not ask pays one extended round in 17 for it -- under 2% of its rate at two lanes, where an
  // extended round costs it about 22% -- and a lane whose text turns predictable waits at most 16
  // rounds behind an estimate from before. Provisional.
  std::uint32_t estimate_rounds = kDefaultEstimateRounds;
  // The extended round's period over the base round's, in thousandths, at 1..4 lanes: measured on
  // the RTX 5090 (1.128, 1.29, 1.363, 1.344; GPU clock, FP8 exception weights, the default cost
  // levers and route, a matched context, two orders pooled). At 3 and 4 lanes the card's 575 W
  // power limit lowers the extended round's clock, so 3 lanes measured above 4 in either order: not
  // a slip. The ratio grows with context; these are its value at about 3.5-5K tokens. The one-lane
  // entry is not read: a lane alone runs at the shape it asks.
  std::array<std::uint32_t, IGNIS_EXTENDED_SHAPE_MAX_LANES> period_ratios = kDefaultPeriodRatios;
  // GitHub #715: the cost table's mixed entries, the round of w extended lanes beside n base lanes
  // over the base round of w + n lanes, in thousandths, in kMixTable's order: read by `mixed` alone.
  std::array<std::uint32_t, round_layout::kMixes> mixed_period_ratios = kDefaultMixedPeriodRatios;
};

// The shape policy a load's options give (`policy`, IGNIS_SHAPE_POLICY_*; `estimate_rounds`, 0 the
// default; `period_ratios`, IGNIS_EXTENDED_SHAPE_MAX_LANES thousandths, 0 the default of that lane
// count; GitHub #715: `mixed_period_ratios`, round_layout::kMixes thousandths, 0 the default of that
// mix) in `out`, or the field a load refuses: a policy it does not know, a window past the rounds a
// lane keeps, a ratio under 1 (a round with an extended lane costing less than the base one) or past
// 2 (more than twice, for at most twice the columns).
inline const char *policy_of(std::uint32_t policy, std::uint32_t estimate_rounds, const std::uint32_t *period_ratios,
                             const std::uint32_t *mixed_period_ratios, ShapePolicy &out) {
  ShapePolicy resolved{};
  if (policy != IGNIS_SHAPE_POLICY_AGGREGATE && policy != IGNIS_SHAPE_POLICY_ALL && policy != IGNIS_SHAPE_POLICY_MIXED) {
    return "shape_policy";
  }
  resolved.policy = policy == IGNIS_SHAPE_POLICY_ALL     ? Policy::kAll
                    : policy == IGNIS_SHAPE_POLICY_MIXED ? Policy::kMixed
                                                         : Policy::kAggregate;
  if (estimate_rounds > IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS) {
    return "shape_estimate_rounds";
  }
  if (estimate_rounds != 0) {
    resolved.estimate_rounds = estimate_rounds;
  }
  const auto take = [](const std::uint32_t *given, auto &table) {
    for (std::size_t at = 0; at < table.size(); ++at) {
      const std::uint32_t ratio = given[at];
      if (ratio != 0 && (ratio < kPeriodRatioMinMilli || ratio > kPeriodRatioMaxMilli)) {
        return false;
      }
      if (ratio != 0) {
        table[at] = ratio;
      }
    }
    return true;
  };
  if (!take(period_ratios, resolved.period_ratios)) {
    return "shape_period_ratios";
  }
  if (!take(mixed_period_ratios, resolved.mixed_period_ratios)) {
    return "shape_mixed_period_ratios";
  }
  out = resolved;
  return nullptr;
}

// GitHub #715: the cost table's entry at a round of `lanes` lanes with `extended` of them extended:
// 1000 for none, the uniform entry for all of them, a mix's otherwise (`extended` within the lanes
// 64 columns hold).
inline std::uint32_t period_ratio(const ShapePolicy &policy, std::uint32_t lanes, std::uint32_t extended) {
  if (extended == 0) {
    return 1000;
  }
  if (extended == lanes) {
    return policy.period_ratios[lanes - 1];
  }
  for (std::uint32_t m = 0; m < round_layout::kMixes; ++m) {
    if (round_layout::kMixTable[m][0] == extended && round_layout::kMixTable[m][1] == lanes - extended) {
      return policy.mixed_period_ratios[m];
    }
  }
  return kPeriodRatioMaxMilli;
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

// GitHub #715 (spec #711): one lane of a round as the grant reads it: what it asks, its recent
// rounds, and the verify rounds in a row it asked the extended shape and ran at the base shape
// (`ignis_seq::shape_refused_rounds`), which breaks a tie.
struct GrantLane {
  bool asks = false;
  LaneEstimate estimate{};
  std::uint32_t refused_rounds = 0;
};

// The most columns a verify round carries (8 lanes at the base shape), and a lane's at the base
// shape and, at the mixed round's window of 15, at the extended one.
inline constexpr std::uint32_t kRoundColumns = round_layout::kMaxColumns;
inline constexpr std::uint32_t kBaseColumns = IGNIS_BASE_SHAPE_DRAFT_TOKENS + 1;
inline constexpr std::uint32_t kMixedExtendedColumns = mixed_graphs::kWindow + 1;
inline constexpr std::uint32_t kMaxLanes = kRoundColumns / kBaseColumns;

// Each lane's shape in a round of `count` lanes (1..kMaxLanes) under `policy`, `extended[i]` 1 where
// lane i runs at the extended shape; the extended lanes' count returned. Every lane at one shape is a
// uniform round, both a mixed round, which only `mixed` gives.
//   `all`: every lane extended when every lane asks, at 1..IGNIS_EXTENDED_SHAPE_MAX_LANES lanes.
//   `aggregate`: every lane extended where `policy_extends` says, at those lanes.
//   `mixed`: at one lane, the lane's ask. At 2..7 the asking lanes are granted one by one, those
//   with an estimate (an extended round among their recent rounds) first, the highest expected
//   extra tokens first, at a tie the lane refused longest, then the caller's order. Each grant takes
//   the round from its w extended lanes to w + 1 while 16 (w + 1) + 8 (n - 1) columns stay within
//   64, and is made for a lane with no estimate whenever they do, and otherwise when
//
//     expected_extra_tokens(lane) * ratio(w) > (ratio(w + 1) - ratio(w)) * sum over lanes of expected_tokens
//
//   each mean in thousandths of a token (as `policy_extends` takes them), each ratio the cost
//   table's (`period_ratio`), signed: a step to a round that costs less is granted. A refused lane
//   does not end the grant; only the columns do, so 8 lanes are always at the base shape, and a step
//   to a mix outside `held` (bit m for kMixTable[m], the mixes the load holds a graph of), so the
//   grant stops at the last mix the load holds. A round at a lane count where the load holds no mix
//   (no 1+n) takes `aggregate`'s choice, as a round of a load that holds none.
// The columns are the mixed round's window of 15 drafts: a load at another window runs `mixed` as
// `aggregate` (resolved at load, kernel/src/model.cu).
inline std::uint32_t grant(const ShapePolicy &policy, const GrantLane *lanes, std::uint64_t count,
                           std::uint8_t *extended, std::uint32_t held) {
  std::fill_n(extended, count, std::uint8_t{0});
  if (count == 0 || count > kMaxLanes) {
    return 0;
  }
  const auto uniform = [&](bool at) {
    std::fill_n(extended, count, static_cast<std::uint8_t>(at ? 1 : 0));
    return at ? static_cast<std::uint32_t>(count) : 0u;
  };
  if (count == 1) {
    return uniform(lanes[0].asks);
  }
  const auto holds = [held](std::uint32_t w, std::uint32_t n) {
    const int mix = round_layout::mix_of(w, n);
    return mix >= 0 && ((held >> mix) & 1u) != 0;
  };
  if (policy.policy != Policy::kMixed || !holds(1, static_cast<std::uint32_t>(count) - 1)) {
    std::array<std::uint8_t, kMaxLanes> asks{};
    std::array<LaneEstimate, kMaxLanes> estimates{};
    for (std::uint64_t i = 0; i < count; ++i) {
      asks[i] = lanes[i].asks ? 1 : 0;
      estimates[i] = lanes[i].estimate;
    }
    return uniform(policy_extends(policy, asks.data(), estimates.data(), count));
  }
  const auto expected_tokens = [](const LaneEstimate &e) -> std::int64_t {
    return e.rounds == 0 ? kBaseColumns * 1000ll : static_cast<std::int64_t>(e.tokens) * 1000 / e.rounds;
  };
  const auto expected_extra = [](const LaneEstimate &e) -> std::int64_t {
    return e.extended_rounds == 0 ? -1 : static_cast<std::int64_t>(e.extra_tokens) * 1000 / e.extended_rounds;
  };
  std::int64_t tokens = 0;
  std::array<std::uint32_t, kMaxLanes> order{};
  std::uint32_t asking = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    tokens += expected_tokens(lanes[i].estimate);
    if (lanes[i].asks) {
      order[asking++] = i;
    }
  }
  // Lane a before lane b: an estimate before none, then the higher expected extra tokens, then the
  // lane refused longer, then the caller's order (an insertion sort keeps it).
  const auto before = [&](std::uint32_t a, std::uint32_t b) {
    const std::int64_t ea = expected_extra(lanes[a].estimate);
    const std::int64_t eb = expected_extra(lanes[b].estimate);
    if ((ea < 0) != (eb < 0)) {
      return eb < 0;
    }
    if (ea != eb) {
      return ea > eb;
    }
    return lanes[a].refused_rounds > lanes[b].refused_rounds;
  };
  for (std::uint32_t i = 1; i < asking; ++i) {
    for (std::uint32_t j = i; j > 0 && before(order[j], order[j - 1]); --j) {
      std::swap(order[j], order[j - 1]);
    }
  }
  const auto lanes32 = static_cast<std::uint32_t>(count);
  std::uint32_t granted = 0;
  for (std::uint32_t k = 0; k < asking; ++k) {
    if (kMixedExtendedColumns * (granted + 1) + kBaseColumns * (lanes32 - granted - 1) > kRoundColumns) {
      break;
    }
    if (granted + 1 < lanes32 && !holds(granted + 1, lanes32 - granted - 1)) {
      break;
    }
    const std::uint32_t lane = order[k];
    const std::int64_t current = period_ratio(policy, lanes32, granted);
    const std::int64_t next = period_ratio(policy, lanes32, granted + 1);
    const std::int64_t extra = expected_extra(lanes[lane].estimate);
    if (extra < 0 || extra * current > (next - current) * tokens) {
      extended[lane] = 1;
      ++granted;
    }
  }
  return granted;
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
