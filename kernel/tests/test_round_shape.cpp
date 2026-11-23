// GitHub #689, #690 (spec #684): the shape rule and the shape policies `aggregate` and `all` of a
// DFlash2 load of two round shapes, as pure functions from tables (kernel/src/round_shape.h), and
// the measurement switch IGNIS_ROUND_SHAPE that forces a shape over them -- OURS, host-only.
//
// A DFlash2 load past 7 drafts holds a base shape (8 columns a lane, at every lane count) and an
// extended shape (the window + 1 columns a lane, at up to 4 lanes). Each verify round has one shape
// for all its lanes, chosen before it launches from what the host holds:
//
//   1. The shape rule, what a lane asks for its next round, one row a case, each row carrying its
//      own thresholds (`spec.shape_enter_accepted`, `spec.shape_stay_accepted`): a lane with no
//      history, entry after a full base round taken whole, stay while its extended rounds take
//      enough drafts, exit after one that takes fewer, and a round the budget cut short, which
//      changes nothing after a base round. A threshold change is a row here, not a GPU run.
//   2. The thresholds a load takes: a stay threshold past the load's window could never keep a lane
//      at the extended shape, and an entry threshold past the base window could never send one
//      there, so a load refuses either, naming the field.
//   3. Under either shape policy a round of 5..8 lanes always runs at the base shape and a load
//      with no extended shape never extended; under `all` a round of 1..4 lanes is extended when
//      every lane in it asks. A lane whose commit the host cuts asks like any other (#691): the
//      policy reads no host cut.
//   4. What a lane's expectations read (#690, `spec.shape_estimate_rounds`): its verify rounds that
//      drafted the base window at least, the most recent of them, each round's tokens at the base
//      shape (its accepts up to 7, and the target's own token) and an extended round's accepts past
//      position 7.
//   5. The shape policy `aggregate` (#690), one row a round's lanes: at one lane the lane's ask; at
//      2..4 lanes extended when the drafts the asking lanes are expected to take past position 7
//      exceed (the period ratio at that lane count - 1) times the tokens every lane is expected to
//      take, the ratios a table (`spec.shape_period_ratios`). Rows: one, some and all lanes asking,
//      a lane granted the extended round because its window holds none, a lane whose extended rounds
//      stopped paying, a lane with no round yet, the boundary, a table of other ratios; the
//      policy's options a load refuses by name.
//   6. Two lanes' course through rounds under both policies, the rule, the record and the policy
//      together: a lane granted an extended round, kept at the base shape while that round says it
//      does not pay, and granted again once its window no longer holds it.
//   7. The switch as a round reads it: unset (or empty) the rule and the policy, `base` and
//      `extended` forcing a shape over them, any other value refused naming the variable.
//   8. The round timing where shapes alternate (kernel/src/round_timing.h): each round's period
//      and gap are its own, to the next round's launch whatever its shape, so each shape's periods
//      are its rounds' and a run's periods sum to its wall time.
//   9. A mixed round's layout and its GDN records (#713, kernel/src/round_layout.h): each group's
//      columns, drafts and record rows; at 15 drafts, over the vendored planner's own layouts of
//      the 27B's records at 2 and 8 lanes, every mix's extended and base rows in every layer and
//      plane disjoint and inside the backing the two shapes share; at 10 and 8 drafts no base
//      layout has the extended one's stride, the base shape's own collides with it, and the round
//      is refused; the base fold's leading rows on slots no base lane holds.
//  10. The grant (#715): each lane's shape under `all`, `aggregate` and `mixed` at the default cost
//      table, on the rows the core's twin holds (crates/core/src/shape_grant.rs): its order, the tie,
//      the lane without an estimate, the column limit, a step to a cheaper round, one lane, eight
//      lanes, and the uniform shapes `all` and `aggregate` give on the same lanes; the table's lookup.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "round_layout.h"
#include "round_shape.h"
#include "round_timing.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using ignis::round_shape::asks_extended;
using ignis::round_shape::estimate_of;
using ignis::round_shape::GrantLane;
using ignis::round_shape::kMaxLanes;
using ignis::round_shape::LaneEstimate;
using ignis::round_shape::period_ratio;
using ignis::round_shape::Named;
using ignis::round_shape::Policy;
using ignis::round_shape::RecentRounds;
using ignis::round_shape::record_round;
using ignis::round_shape::runs_extended;
using ignis::round_shape::ShapePolicy;
using ignis::round_shape::ShapeRule;

int g_failed = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failed;
}

// Sets (or, with null, removes) a variable in the operating system's environment, as a host
// process's own runtime does (Rust's std::env::set_var).
void set_environment(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  if (value != nullptr) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

ShapeRule rule_of(std::uint32_t enter_accepted, std::uint32_t stay_accepted) {
  ShapeRule rule{};
  rule.enter_accepted = enter_accepted;
  rule.stay_accepted = stay_accepted;
  return rule;
}

// The base shape's drafts a lane, and a load's widest window.
constexpr std::int32_t kBase = 7;
constexpr std::int32_t kWindow = 15;

// Table 1: one round of a lane, and what the lane asks of its next round.
struct RuleRow {
  const char *what;
  std::uint32_t enter_accepted;  // the drafts a full base round must take for the lane to ask
  std::uint32_t stay_accepted;   // the drafts an extended round must take for the lane to keep asking
  bool asked;                    // what the lane asked of the round
  bool extended;                 // the shape the round ran
  std::int32_t extent;           // the drafts the lane verified (a budget may cut it under the window)
  std::int32_t accepted;         // the drafts the target accepted
  bool asks;                     // what the lane asks of its next round
};

void check_the_rule() {
  const std::vector<RuleRow> rows = {
      // Entry: a base round that took every draft of a full base window.
      {"entry: a full base window, all 7 taken", 7, 6, false, false, kBase, 7, true},
      {"entry: whatever the stay threshold (1)", 7, 1, false, false, kBase, 7, true},
      {"entry: whatever the stay threshold (15)", 7, 15, false, false, kBase, 7, true},
      {"entry: a lane that asked and rode a base round, all 7 taken", 7, 6, true, false, kBase, 7, true},
      {"entry at a threshold of 5: 5 of a full window taken", 5, 6, false, false, kBase, 5, true},
      {"entry at a threshold of 1: 1 of a full window taken", 1, 6, false, false, kBase, 1, true},
      // No entry: under the threshold of a full window.
      {"no entry: a full base window, 6 of 7 taken", 7, 6, false, false, kBase, 6, false},
      {"no entry: a full base window, none taken", 7, 6, false, false, kBase, 0, false},
      {"no entry at a threshold of 5: 4 taken", 5, 6, false, false, kBase, 4, false},
      {"no entry at a threshold of 1: none taken", 1, 6, false, false, kBase, 0, false},
      {"a lane that asked and rode a full base round, 6 of 7 taken, asks no more", 7, 6, true, false, kBase, 6,
       false},
      // A base round the budget cut short says nothing: the lane keeps what it asked, taken whole or
      // not.
      {"short base round: 3 of 3 taken, the lane had not asked", 7, 6, false, false, 3, 3, false},
      {"short base round: 3 of 3 taken, the lane had asked", 7, 6, true, false, 3, 3, true},
      {"short base round: 0 of 3 taken, the lane had asked", 7, 6, true, false, 3, 0, true},
      {"short base round at a threshold of 1: 1 of 2 taken, the lane had not asked", 1, 6, false, false, 2, 1,
       false},
      {"short base round: an extent of 0, the lane had not asked", 7, 6, false, false, 0, 0, false},
      {"short base round: an extent of 0, the lane had asked", 7, 6, true, false, 0, 0, true},
      // Stay: an extended round that took at least the stay threshold.
      {"stay: an extended round took 6 at a threshold of 6", 7, 6, true, true, kWindow, 6, true},
      {"stay: an extended round took all 15 at a threshold of 6", 7, 6, true, true, kWindow, 15, true},
      {"stay: an extended round took 8 at a threshold of 8", 7, 8, true, true, kWindow, 8, true},
      {"stay: an extended round took 1 at a threshold of 1", 7, 1, true, true, kWindow, 1, true},
      {"stay: an extended round took all 15 at a threshold of 15", 7, 15, true, true, kWindow, 15, true},
      {"stay: an extended round cut to an extent of 9 took 6 at a threshold of 6", 7, 6, true, true, 9, 6, true},
      {"stay: a lane that had not asked rode an extended round (forced), took 6", 7, 6, false, true, kWindow, 6,
       true},
      // Exit: an extended round that took fewer than the stay threshold.
      {"exit: an extended round took 5 at a threshold of 6", 7, 6, true, true, kWindow, 5, false},
      {"exit: an extended round took 7 at a threshold of 8", 7, 8, true, true, kWindow, 7, false},
      {"exit: an extended round took 0 at a threshold of 1", 7, 1, true, true, kWindow, 0, false},
      {"exit: an extended round took 14 at a threshold of 15", 7, 15, true, true, kWindow, 14, false},
      {"exit: an extended round cut to an extent of 4 by its budget, all 4 taken", 7, 6, true, true, 4, 4, false},
      {"exit: an extended round cut to an extent of 0", 7, 6, true, true, 0, 0, false},
      // A narrower extended window (9, ten columns a lane) stays by the same threshold.
      {"stay: an extended round at a window of 9 took 6 at a threshold of 6", 7, 6, true, true, 9, 6, true},
      {"exit: an extended round at a window of 9 took 5 at a threshold of 6", 7, 6, true, true, 9, 5, false},
  };
  for (const RuleRow &row : rows) {
    const bool asks = asks_extended(rule_of(row.enter_accepted, row.stay_accepted), row.asked, row.extended,
                                    row.extent, row.accepted);
    check(asks == row.asks, std::string("rule, ") + row.what + ": the lane asks the " +
                                (row.asks ? "extended" : "base") + " shape next");
  }
  // Each threshold is a threshold: at every value an extended round keeps the lane asking exactly
  // from that many drafts taken, and a full base round sends it exactly from the entry threshold;
  // neither reads the other.
  for (std::uint32_t stay = 1; stay <= static_cast<std::uint32_t>(kWindow); ++stay) {
    for (std::uint32_t enter = 1; enter <= static_cast<std::uint32_t>(kBase); ++enter) {
      const ShapeRule rule = rule_of(enter, stay);
      bool stays = true;
      for (std::int32_t accepted = 0; accepted <= kWindow; ++accepted) {
        stays = stays && asks_extended(rule, true, true, kWindow, accepted) ==
                             (accepted >= static_cast<std::int32_t>(stay));
      }
      bool enters = true;
      for (std::int32_t accepted = 0; accepted <= kBase; ++accepted) {
        enters = enters && asks_extended(rule, false, false, kBase, accepted) ==
                               (accepted >= static_cast<std::int32_t>(enter));
      }
      check(stays && enters, "thresholds (enter " + std::to_string(enter) + ", stay " + std::to_string(stay) +
                                 "): a lane stays from " + std::to_string(stay) + " drafts taken and enters from " +
                                 std::to_string(enter) + " of a full base window");
    }
  }
}

// Table 2: the thresholds a load of a given window takes, and the field a refusal names.
void check_the_thresholds() {
  check(ShapeRule{}.enter_accepted == 7 && ShapeRule{}.stay_accepted == 6,
        "the default thresholds are 7 to enter and 6 to stay");
  for (std::int32_t window = kBase + 1; window <= kWindow; ++window) {
    const std::string at = " on a load whose extended round drafts " + std::to_string(window);
    for (std::uint32_t enter = 0; enter <= static_cast<std::uint32_t>(kBase) + 1; ++enter) {
      for (std::uint32_t stay = 0; stay <= static_cast<std::uint32_t>(kWindow) + 1; ++stay) {
        const char *refused = ignis::round_shape::refused_threshold(rule_of(enter, stay), window);
        const bool enter_ok = enter >= 1 && enter <= static_cast<std::uint32_t>(kBase);
        const bool stay_ok = stay >= 1 && stay <= static_cast<std::uint32_t>(window);
        const char *want = !enter_ok ? "shape_enter_accepted" : !stay_ok ? "shape_stay_accepted" : nullptr;
        // What the rule itself does with the thresholds: some accept of a full window keeps a lane
        // asking, and some accept of a full base window sends it -- the reason for the refusal.
        bool some_stay = false;
        for (std::int32_t accepted = 0; accepted <= window; ++accepted) {
          some_stay = some_stay || asks_extended(rule_of(enter, stay), true, true, window, accepted);
        }
        bool some_entry = false;
        for (std::int32_t accepted = 0; accepted <= kBase; ++accepted) {
          some_entry = some_entry || asks_extended(rule_of(enter, stay), false, false, kBase, accepted);
        }
        const std::string what = "enter " + std::to_string(enter) + ", stay " + std::to_string(stay) + at;
        check((refused == nullptr) == (want == nullptr) &&
                  (want == nullptr || std::string(refused) == want),
              what + (want == nullptr ? " is taken" : std::string(" is refused naming ") + want));
        // A threshold of 0 would keep or send every lane: taken by the rule, refused by the load.
        if (enter >= 1 && stay >= 1) {
          check((refused == nullptr) == (some_stay && some_entry),
                what + ": refused exactly where no accept could keep or send a lane");
        }
      }
    }
  }
}

// GitHub #721: the cost table the rows here are worked against, the tests' own (the core's tests
// hold the same, crates/core/src/shape_grant.rs `TABLE`), so that each row's arithmetic holds
// whatever the defaults are: the uniform entries 1.128, 1.29, 1.363 and 1.344 at 1 to 4 lanes, and
// the mixed entries in kMixTable's order, 1+1 1.155 ... 3+2 1.336, its 3+1 above the uniform entry
// at 4 lanes.
constexpr std::array<std::uint32_t, IGNIS_EXTENDED_SHAPE_MAX_LANES> kTestPeriodRatios = {1128, 1290, 1363, 1344};
constexpr std::array<std::uint32_t, ignis::round_layout::kMixes> kTestMixedPeriodRatios = {
    1155, 1120, 1138, 1077, 1093, 1107, 1285, 1222, 1207, 1241, 1384, 1336};

// The policies, `all` and `aggregate`, over the tests' cost table and the defaults of the rest of
// the policy.
ShapePolicy policy_of(Policy policy) {
  ShapePolicy shape_policy{};
  shape_policy.policy = policy;
  shape_policy.period_ratios = kTestPeriodRatios;
  shape_policy.mixed_period_ratios = kTestMixedPeriodRatios;
  return shape_policy;
}

std::string name_of(Policy policy) { return policy == Policy::kAll ? "`all`" : "`aggregate`"; }

// A lane's expectations over 16 rounds: its rounds, their tokens at the base shape, its extended
// rounds among them and their accepts past position 7. The figures of the spec's pairs, measured
// one lane at a time on the RTX 5090: a JSON answer (7.94 tokens a base round, 5.44 drafts past 7
// an extended round), a moderately predictable answer (5.69 and 1.75), a conversation (3.5 tokens,
// never extended).
constexpr LaneEstimate kJson{16, 127, 16, 87};
constexpr LaneEstimate kModerate{16, 91, 16, 28};
constexpr LaneEstimate kConversation{16, 56, 0, 0};
// A lane whose window holds no extended round: what it would take past 7 is unread.
constexpr LaneEstimate kUnread{16, 128, 0, 0};
// A lane whose extended rounds take 6 or 7 drafts, enough to keep asking, and nothing past 7.
constexpr LaneEstimate kStopped{16, 128, 16, 0};
// A lane with no round yet.
constexpr LaneEstimate kNewcomer{0, 0, 0, 0};

// Table 3: the round's shape from its lanes' asks and the forcing switch, under both policies: the
// lanes limit and a load with no extended shape; and `all`'s one quiet lane.
void check_the_policy() {
  constexpr std::uint32_t kExtendedLanes = 4;
  constexpr std::size_t kMostLanes = 8;
  for (const Policy policy : {Policy::kAll, Policy::kAggregate}) {
    const ShapePolicy shape_policy = policy_of(policy);
    for (const Named named : {Named::kRule, Named::kBase, Named::kExtended}) {
      const std::string under = name_of(policy) + (named == Named::kRule   ? ", under the rule (unset)"
                                                   : named == Named::kBase ? ", under `base`"
                                                                           : ", under `extended`");
      for (std::size_t lanes = 1; lanes <= kMostLanes; ++lanes) {
        const std::string at = under + ", " + std::to_string(lanes) + " lanes";
        const std::vector<std::uint8_t> all(lanes, 1);
        const std::vector<std::uint8_t> none(lanes, 0);
        // Every lane's extended rounds pay, or a granted lane's are unread: `aggregate` extends a
        // round of them wherever `all` does.
        const std::vector<LaneEstimate> paying(lanes, kJson);
        const std::vector<LaneEstimate> unread(lanes, kUnread);
        const auto runs = [&](const std::vector<std::uint8_t> &asks, const std::vector<LaneEstimate> &estimates,
                              std::uint32_t extended_lanes) {
          return runs_extended(named, asks.data(), lanes, extended_lanes, shape_policy, estimates.data());
        };
        // A load with no extended shape never runs one, whatever its lanes ask.
        check(!runs(all, paying, /*extended_lanes=*/0), at + ", no extended shape: base");
        const bool every = runs(all, paying, kExtendedLanes);
        const bool granted = runs(all, unread, kExtendedLanes);
        const bool nobody = runs(none, paying, kExtendedLanes);
        if (lanes > kExtendedLanes) {
          check(!every && !granted && !nobody, at + ": past the extended shape's 4 lanes, base whatever they ask");
          continue;
        }
        switch (named) {
        case Named::kBase:
          check(!every && !granted && !nobody, at + ": base, asked or not");
          break;
        case Named::kExtended:
          check(every && granted && nobody, at + ": extended, asked or not");
          break;
        case Named::kRule:
          check(every && granted, at + ", every lane asking: extended");
          check(!nobody, at + ", no lane asking: base");
          // Under `all` one lane that does not ask, wherever it rides, keeps the round at the base
          // shape: no lane pays for another.
          for (std::size_t quiet = 0; policy == Policy::kAll && quiet < lanes; ++quiet) {
            std::vector<std::uint8_t> asks(lanes, 1);
            asks[quiet] = 0;
            check(!runs(asks, paying, kExtendedLanes), at + ", lane " + std::to_string(quiet) + " not asking: base");
          }
          break;
        }
      }
    }
  }
}

bool same(const LaneEstimate &a, const LaneEstimate &b) {
  return a.rounds == b.rounds && a.tokens == b.tokens && a.extended_rounds == b.extended_rounds &&
         a.extra_tokens == b.extra_tokens;
}

std::string text_of(const LaneEstimate &e) {
  return "{" + std::to_string(e.rounds) + " rounds, " + std::to_string(e.tokens) + " tokens, " +
         std::to_string(e.extended_rounds) + " extended, " + std::to_string(e.extra_tokens) + " past 7}";
}

// Table 4: what a lane's rounds leave for its expectations, round by round, and the window they
// are read over.
void check_the_estimates() {
  RecentRounds rounds{};
  std::uint64_t recorded = 0;
  check(same(estimate_of(rounds, recorded, 16), LaneEstimate{}), "no round recorded: nothing to read");
  struct Step {
    const char *what;
    bool extended;
    std::int32_t extent;
    std::int32_t accepted;
    LaneEstimate after;  // over a window of 16
  };
  const std::vector<Step> steps = {
      {"a base round, all 7 taken: 8 tokens", false, kBase, 7, {1, 8, 0, 0}},
      {"an extended round, all 15 taken: 8 tokens at the base shape, 8 past 7", true, kWindow, 15, {2, 16, 1, 8}},
      {"an extended round, 9 taken: 2 past 7", true, kWindow, 9, {3, 24, 2, 10}},
      {"an extended round, 5 taken: 6 tokens, none past 7, an extended round still", true, kWindow, 5,
       {4, 30, 3, 10}},
      {"a base round, 3 taken: 4 tokens", false, kBase, 3, {5, 34, 3, 10}},
      {"a base round the budget cut to 3: says nothing", false, 3, 3, {5, 34, 3, 10}},
      {"an extended round the budget cut to 4: says nothing", true, 4, 4, {5, 34, 3, 10}},
      {"an extended round the budget cut to 9, all 9 taken: 2 past 7", true, 9, 9, {6, 42, 4, 12}},
      {"a base round, none taken: the target's own token", false, kBase, 0, {7, 43, 4, 12}},
  };
  for (const Step &step : steps) {
    record_round(rounds, recorded, step.extended, step.extent, step.accepted);
    const LaneEstimate got = estimate_of(rounds, recorded, 16);
    check(same(got, step.after), std::string("estimates, ") + step.what + ": " + text_of(got));
  }
  // A window reads the most recent rounds: the last is a base round of 1 token, the one before it
  // an extended round of 8 at the base shape and 2 past 7, the one before that a base round of 4.
  check(same(estimate_of(rounds, recorded, 1), LaneEstimate{1, 1, 0, 0}), "a window of 1: the last round alone");
  check(same(estimate_of(rounds, recorded, 2), LaneEstimate{2, 9, 1, 2}), "a window of 2: the last two rounds");
  check(same(estimate_of(rounds, recorded, 3), LaneEstimate{3, 13, 1, 2}), "a window of 3: the last three");
  // The record keeps the widest window's rounds: past it the oldest go, and a window reads only
  // what is left.
  for (int i = 0; i < 70; ++i) {
    record_round(rounds, recorded, false, kBase, 1);
  }
  check(same(estimate_of(rounds, recorded, IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS),
             LaneEstimate{IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS, 2 * IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS, 0, 0}),
        "past the widest window the oldest rounds go");
  record_round(rounds, recorded, true, kWindow, 12);
  check(same(estimate_of(rounds, recorded, 16), LaneEstimate{16, 38, 1, 5}), "a window of 16 after the wrap");
}

// Table 5: the shape policy `aggregate` beside `all`, one row a round.
void check_aggregate() {
  struct Row {
    const char *what;
    std::vector<std::uint8_t> asks;
    std::vector<LaneEstimate> lanes;
    bool aggregate;  // the round's shape under `aggregate`: extended or not
    bool all;        // and under `all`
  };
  const std::vector<Row> rows = {
      // One lane: its ask decides, under both.
      {"one lane asking, its extended rounds paying", {1}, {kJson}, true, true},
      {"one lane asking, its extended rounds no longer paying", {1}, {kStopped}, true, true},
      {"one lane asking, its window holding no extended round", {1}, {kUnread}, true, true},
      {"one lane not asking", {0}, {kJson}, false, false},
      // Two lanes. A JSON answer beside a conversation: 5.44 drafts past 7 against 0.29 x 11.44.
      {"two lanes: a JSON answer asking beside a conversation", {1, 0}, {kJson, kConversation}, true, false},
      {"two lanes: a conversation beside a JSON answer asking", {0, 1}, {kConversation, kJson}, true, false},
      // A moderately predictable answer beside a conversation: 1.75 against 0.29 x 9.19.
      {"two lanes: a moderately predictable answer asking beside a conversation", {1, 0},
       {kModerate, kConversation}, false, false},
      {"two lanes, both asking, both paying", {1, 1}, {kJson, kJson}, true, true},
      {"two lanes, both asking, neither paying past 7", {1, 1}, {kStopped, kStopped}, false, true},
      {"two lanes: a lane whose extended rounds stopped paying, asking beside a conversation", {1, 0},
       {kStopped, kConversation}, false, false},
      // A lane that asks with no extended round in its window is granted one, so the estimate has
      // something to read.
      {"two lanes: a lane granted its extended round beside a conversation", {1, 0}, {kUnread, kConversation},
       true, false},
      {"two lanes: a lane granted beside one whose extended rounds stopped paying", {1, 1}, {kStopped, kUnread},
       true, true},
      // A lane with no round yet is expected to take a full base round: 8 tokens, the most the wider
      // round can cost it (at 0 the moderately predictable answer would pass: 1.75 > 0.29 x 5.69).
      {"two lanes: a JSON answer asking beside a lane with no round yet", {1, 0}, {kJson, kNewcomer}, true, false},
      {"two lanes: a moderately predictable answer asking beside a lane with no round yet", {1, 0},
       {kModerate, kNewcomer}, false, false},
      {"two lanes, no lane asking", {0, 0}, {kJson, kJson}, false, false},
      {"two lanes, no lane asking, one with no extended round", {0, 0}, {kUnread, kConversation}, false, false},
      // The boundary: 10 rounds of 8 tokens and 29 drafts past 7 over 10 extended rounds, beside 10
      // rounds of 2 tokens: 2.9 against 0.29 x 10 is not more, 3 is.
      {"two lanes, the gain equal to the cost: base", {1, 0}, {{10, 80, 10, 29}, {10, 20, 0, 0}}, false, false},
      {"two lanes, the gain past the cost: extended", {1, 0}, {{10, 80, 10, 30}, {10, 20, 0, 0}}, true, false},
      // Three lanes, 0.363 the cost: a JSON answer still carries two conversations (5.44 against
      // 0.363 x 14.94).
      {"three lanes: a JSON answer asking beside two conversations", {1, 0, 0},
       {kJson, kConversation, kConversation}, true, false},
      {"three lanes: some asking", {1, 1, 0}, {kJson, kModerate, kConversation}, true, false},
      {"three lanes: a moderately predictable answer asking beside two conversations", {1, 0, 0},
       {kModerate, kConversation, kConversation}, false, false},
      {"three lanes, every lane asking, none paying past 7", {1, 1, 1}, {kStopped, kStopped, kStopped}, false,
       true},
      // Three moderately predictable answers, every lane asking: each one's 1.75 past 7 is under its
      // own 0.363 x 5.69, so `aggregate` keeps them at the base shape where `all` extends them.
      {"three lanes, every lane asking, each paying less than it costs", {1, 1, 1},
       {kModerate, kModerate, kModerate}, false, true},
      {"three lanes, every lane asking and paying", {1, 1, 1}, {kJson, kJson, kJson}, true, true},
      // Four lanes, 0.344: one JSON answer no longer carries three conversations (5.44 against 0.344 x
      // 18.44), two do.
      {"four lanes: a JSON answer asking beside three conversations", {1, 0, 0, 0},
       {kJson, kConversation, kConversation, kConversation}, false, false},
      {"four lanes: some asking", {1, 1, 0, 0}, {kJson, kJson, kConversation, kConversation}, true, false},
      {"four lanes, every lane asking", {1, 1, 1, 1}, {kJson, kJson, kJson, kJson}, true, true},
      {"four lanes: a lane granted beside three conversations", {1, 0, 0, 0},
       {kUnread, kConversation, kConversation, kConversation}, true, false},
      {"four lanes: a lane whose extended rounds stopped paying beside a granted one", {1, 1, 0, 0},
       {kStopped, kUnread, kConversation, kConversation}, true, false},
  };
  for (const Policy policy : {Policy::kAggregate, Policy::kAll}) {
    for (const Row &row : rows) {
      const bool want = policy == Policy::kAggregate ? row.aggregate : row.all;
      const bool got = runs_extended(Named::kRule, row.asks.data(), row.asks.size(), /*extended_lanes=*/4,
                                     policy_of(policy), row.lanes.data());
      check(got == want, name_of(policy) + ", " + row.what + ": " + (want ? "extended" : "base"));
    }
  }
  // 5..8 lanes: the base shape under both, every lane asking, paying or granted.
  for (const Policy policy : {Policy::kAggregate, Policy::kAll}) {
    for (std::size_t lanes = 5; lanes <= 8; ++lanes) {
      const std::vector<std::uint8_t> asks(lanes, 1);
      for (const LaneEstimate &estimate : {kJson, kUnread}) {
        const std::vector<LaneEstimate> estimates(lanes, estimate);
        check(!runs_extended(Named::kRule, asks.data(), lanes, /*extended_lanes=*/4, policy_of(policy),
                             estimates.data()),
              name_of(policy) + ", " + std::to_string(lanes) + " lanes, every lane asking: base");
      }
    }
  }
  // The ratios are a table, read at the round's lane count.
  ShapePolicy free = policy_of(Policy::kAggregate);
  free.period_ratios = {1000, 1000, 1000, 1000};
  ShapePolicy dear = policy_of(Policy::kAggregate);
  dear.period_ratios = {2000, 2000, 2000, 2000};
  const std::vector<std::uint8_t> one_of_two = {1, 0};
  const std::vector<LaneEstimate> moderate = {kModerate, kConversation};
  const std::vector<LaneEstimate> json = {kJson, kConversation};
  check(runs_extended(Named::kRule, one_of_two.data(), 2, 4, free, moderate.data()),
        "a ratio of 1 at two lanes: any draft past 7 extends the round");
  check(!runs_extended(Named::kRule, one_of_two.data(), 2, 4, dear, json.data()),
        "a ratio of 2 at two lanes: a JSON answer no longer carries a conversation");
  ShapePolicy two_lanes_only = policy_of(Policy::kAggregate);
  two_lanes_only.period_ratios = {1150, 1000, 2000, 2000};
  const std::vector<std::uint8_t> one_of_three = {1, 0, 0};
  const std::vector<LaneEstimate> moderate3 = {kModerate, kConversation, kConversation};
  check(runs_extended(Named::kRule, one_of_two.data(), 2, 4, two_lanes_only, moderate.data()) &&
            !runs_extended(Named::kRule, one_of_three.data(), 3, 4, two_lanes_only, moderate3.data()),
        "each lane count reads its own ratio");
}

// Table 5b: the policy's options as a load takes them, 0 the default of each, and the field a
// refusal names.
void check_the_policy_options() {
  const ShapePolicy defaults{};
  check(defaults.policy == Policy::kAggregate && defaults.estimate_rounds == 16 &&
            defaults.period_ratios == std::array<std::uint32_t, 4>{1108, 1263, 1321, 1304},
        "the defaults: `aggregate`, 16 rounds, ratios 1.108, 1.263, 1.321, 1.304");
  struct Row {
    const char *what;
    std::uint32_t policy;
    std::uint32_t estimate_rounds;
    std::array<std::uint32_t, 4> ratios;
    const char *refused;  // the field, or null
  };
  const std::vector<Row> rows = {
      {"every option 0: the defaults", 0, 0, {0, 0, 0, 0}, nullptr},
      {"`all`", IGNIS_SHAPE_POLICY_ALL, 0, {0, 0, 0, 0}, nullptr},
      {"`mixed` (GitHub #715)", IGNIS_SHAPE_POLICY_MIXED, 0, {0, 0, 0, 0}, nullptr},
      {"a policy it does not know", 3, 0, {0, 0, 0, 0}, "shape_policy"},
      {"a window of 1 round", 0, 1, {0, 0, 0, 0}, nullptr},
      {"the widest window", 0, IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS, {0, 0, 0, 0}, nullptr},
      {"a window past the widest", 0, IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS + 1, {0, 0, 0, 0}, "shape_estimate_rounds"},
      {"ratios at 1 and 2", 0, 0, {1000, 2000, 1000, 2000}, nullptr},
      {"a ratio under 1", 0, 0, {1150, 999, 0, 0}, "shape_period_ratios"},
      {"a ratio past 2", 0, 0, {0, 0, 0, 2001}, "shape_period_ratios"},
  };
  for (const Row &row : rows) {
    ShapePolicy got{};
    const std::array<std::uint32_t, ignis::round_layout::kMixes> mixed{};
    const char *refused =
        ignis::round_shape::policy_of(row.policy, row.estimate_rounds, row.ratios.data(), mixed.data(), got);
    check((refused == nullptr) == (row.refused == nullptr) &&
              (row.refused == nullptr || std::string(refused) == row.refused),
          std::string("policy options, ") + row.what + (row.refused == nullptr ? ": taken" : ": refused naming ") +
              (row.refused == nullptr ? "" : row.refused));
  }
  ShapePolicy got{};
  const std::array<std::uint32_t, 4> some = {0, 1100, 0, 1500};
  std::array<std::uint32_t, ignis::round_layout::kMixes> mixed{};
  check(ignis::round_shape::policy_of(IGNIS_SHAPE_POLICY_ALL, 8, some.data(), mixed.data(), got) == nullptr &&
            got.policy == Policy::kAll && got.estimate_rounds == 8 &&
            got.period_ratios == std::array<std::uint32_t, 4>{1108, 1100, 1321, 1500},
        "a ratio of 0 is that lane count's default, the others as given");
  // GitHub #715: the cost table's mixed entries, 0 each's default, refused by their own name.
  check(got.mixed_period_ratios == ignis::round_shape::kDefaultMixedPeriodRatios,
        "the mixed entries at their defaults: 1+1 1.15 ... 3+2 1.177");
  mixed[7] = 1500;
  check(ignis::round_shape::policy_of(IGNIS_SHAPE_POLICY_MIXED, 0, some.data(), mixed.data(), got) == nullptr &&
            got.policy == Policy::kMixed && got.mixed_period_ratios[7] == 1500 && got.mixed_period_ratios[0] == 1150,
        "`mixed` with the 2+2 entry given, the others their defaults");
  for (const std::uint32_t wrong : {999u, 2001u}) {
    mixed[11] = wrong;
    const char *refused = ignis::round_shape::policy_of(IGNIS_SHAPE_POLICY_MIXED, 0, some.data(), mixed.data(), got);
    check(refused != nullptr && std::string(refused) == "shape_mixed_period_ratios",
          "a mixed entry of " + std::to_string(wrong) + " thousandths is refused naming its field");
  }
}

// GitHub #715 (spec #711): the grant, each lane's shape under the three policies at the default cost
// table, on the rows crates/core/src/shape_grant.rs holds its twin to (`lane_shapes`), worked there
// from the spec's rule.
GrantLane lane(bool asks, std::uint32_t rounds, std::uint32_t tokens, std::uint32_t extended, std::uint32_t extra,
               std::uint32_t refused) {
  GrantLane at{};
  at.asks = asks;
  at.estimate = LaneEstimate{rounds, tokens, extended, extra};
  at.refused_rounds = refused;
  return at;
}

// A conversation (2 tokens a round, never asking), and a lane asking at 8 tokens a round with 4
// extended rounds taking `extra` drafts past position 7 each.
GrantLane conversation() { return lane(false, 16, 32, 0, 0, 0); }
GrantLane asking(std::uint32_t extra, std::uint32_t refused) { return lane(true, 16, 128, 4, 4 * extra, refused); }

void check_the_grant() {
  // The mixes a load holds a graph of, bit m for kMixTable[m]: every mix, and those of pools of 4
  // lanes beside one device retained slot (1+1, 1+2, 1+3, 2+1), of 4 lanes beside none (1+1, 1+2),
  // and of every mix but 1+2.
  constexpr std::uint32_t kEveryMix = (1u << ignis::round_layout::kMixes) - 1;
  constexpr std::uint32_t kFourAndOne = 0b1000111;
  constexpr std::uint32_t kFour = 0b11;
  constexpr std::uint32_t kButOnePlusTwo = kEveryMix & ~0b10u;
  struct Row {
    const char *what;
    Policy policy;
    std::vector<GrantLane> lanes;
    std::vector<int> want;
    std::uint32_t held = kEveryMix;
  };
  const auto seven = [](std::uint32_t at, GrantLane one, std::uint32_t other_at, GrantLane other) {
    std::vector<GrantLane> lanes(7, conversation());
    lanes[at] = one;
    lanes[other_at] = other;
    return lanes;
  };
  const GrantLane fresh = lane(true, 16, 128, 0, 0, 9);
  const std::vector<Row> rows = {
      {"the order: at 7 lanes the highest expected extra tokens take the one extended lane", Policy::kMixed,
       seven(0, lane(true, 16, 32, 4, 12, 0), 2, lane(true, 16, 32, 4, 20, 0)), {0, 0, 1, 0, 0, 0, 0}},
      {"4 lanes: both pay", Policy::kMixed, {asking(5, 0), conversation(), asking(6, 0), conversation()}, {1, 0, 1, 0}},
      {"4 lanes: the second step costs less than the first", Policy::kMixed,
       {asking(2, 0), conversation(), asking(6, 0), conversation()}, {1, 0, 1, 0}},
      {"4 lanes: 1 extra token does not pay the second step", Policy::kMixed,
       {asking(1, 0), conversation(), asking(6, 0), conversation()}, {0, 0, 1, 0}},
      {"the tie: the lane refused longest", Policy::kMixed,
       seven(1, lane(true, 16, 32, 4, 20, 2), 4, lane(true, 16, 32, 4, 20, 5)), {0, 0, 0, 0, 1, 0, 0}},
      {"the tie of that too: the caller's order", Policy::kMixed,
       seven(1, lane(true, 16, 32, 4, 20, 2), 4, lane(true, 16, 32, 4, 20, 2)), {0, 1, 0, 0, 0, 0, 0}},
      {"no estimate: after the estimated lane, granted when the columns allow", Policy::kMixed,
       {fresh, asking(7, 0), conversation()}, {1, 1, 0}},
      {"no round at all", Policy::kMixed, {lane(true, 0, 0, 0, 0, 0), conversation()}, {1, 0}},
      {"no estimate: after the estimated lane, out of columns at 7", Policy::kMixed,
       seven(0, fresh, 3, lane(true, 16, 32, 4, 20, 0)), {0, 0, 0, 1, 0, 0, 0}},
      {"no estimate: the estimated lane refused, granted", Policy::kMixed,
       seven(0, fresh, 3, lane(true, 16, 32, 4, 4, 0)), {1, 0, 0, 0, 0, 0, 0}},
      {"the column limit at 5 lanes: 3+2", Policy::kMixed, std::vector<GrantLane>(5, asking(8, 0)), {1, 1, 1, 0, 0}},
      {"the column limit at 6 lanes: 2+4", Policy::kMixed, std::vector<GrantLane>(6, asking(8, 0)),
       {1, 1, 0, 0, 0, 0}},
      {"4 lanes: the uniform extended round", Policy::kMixed, std::vector<GrantLane>(4, asking(8, 0)), {1, 1, 1, 1}},
      {"2 lanes: both", Policy::kMixed, std::vector<GrantLane>(2, asking(8, 0)), {1, 1}},
      {"a step to a cheaper round (3+1 to 4 extended)", Policy::kMixed,
       {asking(8, 0), asking(8, 0), asking(8, 0), asking(0, 0)}, {1, 1, 1, 1}},
      {"one lane asking, mixed", Policy::kMixed, {asking(0, 0)}, {1}},
      {"one lane asking, aggregate", Policy::kAggregate, {asking(0, 0)}, {1}},
      {"one lane asking, all", Policy::kAll, {asking(0, 0)}, {1}},
      {"one lane not asking, mixed", Policy::kMixed, {lane(false, 16, 128, 4, 32, 0)}, {0}},
      {"one lane not asking, aggregate", Policy::kAggregate, {lane(false, 16, 128, 4, 32, 0)}, {0}},
      {"one lane not asking, all", Policy::kAll, {lane(false, 16, 128, 4, 32, 0)}, {0}},
      {"eight lanes, mixed", Policy::kMixed, std::vector<GrantLane>(8, asking(8, 3)), std::vector<int>(8, 0)},
      {"eight lanes, aggregate", Policy::kAggregate, std::vector<GrantLane>(8, asking(8, 3)), std::vector<int>(8, 0)},
      {"eight lanes, all", Policy::kAll, std::vector<GrantLane>(8, asking(8, 3)), std::vector<int>(8, 0)},
      {"a pair, aggregate: uniform extended", Policy::kAggregate, {asking(6, 0), conversation()}, {1, 1}},
      {"a pair, all: base", Policy::kAll, {asking(6, 0), conversation()}, {0, 0}},
      {"a pair, mixed: the asking lane", Policy::kMixed, {asking(6, 0), conversation()}, {1, 0}},
      {"four, aggregate: 6 pays", Policy::kAggregate,
       {asking(6, 0), conversation(), conversation(), conversation()}, {1, 1, 1, 1}},
      {"four, aggregate: 4 does not", Policy::kAggregate,
       {asking(4, 0), conversation(), conversation(), conversation()}, {0, 0, 0, 0}},
      {"four, mixed: 4 pays its own lane", Policy::kMixed,
       {asking(4, 0), conversation(), conversation(), conversation()}, {1, 0, 0, 0}},
      {"aggregate: a lane without an estimate", Policy::kAggregate,
       {lane(true, 16, 32, 0, 0, 0), conversation(), conversation()}, {1, 1, 1}},
      {"all: four asking", Policy::kAll, std::vector<GrantLane>(4, asking(0, 0)), {1, 1, 1, 1}},
      {"all: five asking", Policy::kAll, std::vector<GrantLane>(5, asking(8, 0)), {0, 0, 0, 0, 0}},
      {"aggregate: five asking", Policy::kAggregate, std::vector<GrantLane>(5, asking(8, 0)), {0, 0, 0, 0, 0}},
      {"a partial pool: the grant stops at 1+3, the last mix it holds", Policy::kMixed,
       {asking(5, 0), conversation(), asking(6, 0), conversation()}, {0, 0, 1, 0}, kFourAndOne},
      {"a partial pool: every lane gaining, still 1+3", Policy::kMixed, std::vector<GrantLane>(4, asking(8, 0)),
       {1, 0, 0, 0}, kFourAndOne},
      {"1+2 failed: 1+1 and the uniform round at 2 lanes", Policy::kMixed, {asking(8, 0), asking(8, 0)}, {1, 1},
       kButOnePlusTwo},
      {"1+2 failed: no mix at 3 lanes, aggregate's choice", Policy::kMixed,
       {asking(8, 0), asking(8, 0), conversation()}, {1, 1, 1}, kButOnePlusTwo},
      {"no mix held at 2 lanes: aggregate's choice, 6 pays", Policy::kMixed, {asking(6, 0), conversation()}, {1, 1}, 0},
      {"no mix held at 2 lanes: aggregate's choice, 2 does not", Policy::kMixed, {asking(2, 0), conversation()},
       {0, 0}, 0},
      {"no mix held at 4 lanes: aggregate's choice", Policy::kMixed,
       {asking(6, 0), conversation(), conversation(), conversation()}, {1, 1, 1, 1}, kFour},
      {"no mix held at 5 lanes: aggregate's base round", Policy::kMixed, std::vector<GrantLane>(5, asking(8, 0)),
       {0, 0, 0, 0, 0}, kFourAndOne},
  };
  for (const Row &row : rows) {
    std::array<std::uint8_t, kMaxLanes> extended{};
    const std::uint32_t granted =
        ignis::round_shape::grant(policy_of(row.policy), row.lanes.data(), row.lanes.size(), extended.data(), row.held);
    bool same = granted == static_cast<std::uint32_t>(std::count(row.want.begin(), row.want.end(), 1));
    std::string got;
    for (std::size_t i = 0; i < row.lanes.size(); ++i) {
      same = same && extended[i] == row.want[i];
      got += extended[i] != 0 ? "E" : "b";
    }
    check(same, std::string("the grant, ") + row.what + ": got " + got);
  }
  // The table's lookup: the base round, a mix, the uniform entry.
  const ShapePolicy defaults{};
  check(period_ratio(defaults, 4, 0) == 1000 && period_ratio(defaults, 4, 2) == 1201 &&
            period_ratio(defaults, 4, 4) == 1304 && period_ratio(defaults, 7, 1) == 1000 &&
            period_ratio(defaults, 7, 2) == ignis::round_shape::kPeriodRatioMaxMilli,
        "the cost table at 4 lanes: 1, 1.201 at 2+2, 1.304 at 4 extended; 1 at 1+6; 2+5, past 64 columns, "
        "the widest ratio, as the core's");
}

// One lane through rounds under the default rule and policy, the rule, the record and the round's
// shape together: base until a full window is taken whole, extended while it takes 6, base again
// after a short take, and a round the budget cut short leaving the ask as it was. At one lane
// `aggregate` reads only the ask.
void check_a_lanes_course() {
  const ShapeRule rule{};
  const ShapePolicy policy{};
  struct Step {
    std::int32_t budget_extent;  // the extent a budget cuts the round to, or -1: the shape's whole window
    std::int32_t accepted;
    bool extended;  // the shape this round is expected to run
  };
  const std::vector<Step> course = {
      {-1, 3, false},  // no history: base; takes 3 of 7
      {-1, 7, false},  // still base; takes all 7: asks
      {-1, 15, true},  // extended; takes all 15: keeps asking
      {-1, 6, true},   // extended; takes 6, the threshold: keeps asking
      {-1, 7, true},   // extended; takes 7, nothing past 7: keeps asking, and alone still runs extended
      {4, 4, true},    // extended under a budget: 4 of 4, under the threshold: base next
      {-1, 7, false},  // base; takes all 7: asks
      {2, 2, true},    // extended, cut to 2: base next
      {3, 3, false},   // base, cut to 3: keeps not asking
      {-1, 7, false},  // base; all 7: asks
      {-1, 5, true},   // extended; 5: base next
      {-1, 0, false},  // base
  };
  // A lane with no history has not asked (a fresh sequence's ask, `ignis_seq::asks_extended_shape`).
  bool asks = false;
  RecentRounds rounds{};
  std::uint64_t recorded = 0;
  std::size_t at = 0;
  for (const Step &step : course) {
    const std::uint8_t lane = asks ? 1 : 0;
    const LaneEstimate estimate = estimate_of(rounds, recorded, policy.estimate_rounds);
    const bool extended = runs_extended(Named::kRule, &lane, 1, /*extended_lanes=*/4, policy, &estimate);
    check(extended == step.extended, "a lane's course, round " + std::to_string(at) + " runs " +
                                         (step.extended ? "extended" : "base"));
    const std::int32_t extent = step.budget_extent >= 0 ? step.budget_extent : extended ? kWindow : kBase;
    asks = asks_extended(rule, asks, extended, extent, step.accepted);
    record_round(rounds, recorded, extended, extent, step.accepted);
    ++at;
  }
}

// Table 6: two lanes through rounds, the rule, the record and the policy as the leaf runs them, at
// a window of 4 rounds, lane 0's accepts given a round and lane 1 taking 2 a round, so never
// asking. Lane 0 enters, is granted an extended round that takes nothing past 7, rides the base
// shape while its window holds that round, and is granted again the round its window no longer
// does; a granted round that takes all 15 then keeps the rounds extended. Under `all` every round
// is base.
void check_a_pairs_course() {
  const ShapeRule rule{};
  ShapePolicy aggregate = policy_of(Policy::kAggregate);
  aggregate.estimate_rounds = 4;
  ShapePolicy all = policy_of(Policy::kAll);
  all.estimate_rounds = 4;
  struct Step {
    std::int32_t accepted;  // lane 0's, of the round's whole window
    bool extended;          // the shape under `aggregate`
  };
  const std::vector<Step> course = {
      {7, false},   // round 0, no history: base; lane 0 takes all 7 and asks
      {7, true},    // 1: granted, its window holding no extended round; takes 7, nothing past 7: keeps asking
      {7, false},   // 2: rounds 0-1 in its window, round 1 nothing past 7 to gain: base; takes all 7, asks
      {7, false},   // 3: rounds 0-2
      {7, false},   // 4: rounds 0-3
      {7, false},   // 5: rounds 1-4, round 1 the last of them to go
      {7, true},    // 6: rounds 2-5 hold no extended round: granted again; takes 7
      {7, false},   // 7
      {7, false},   // 8
      {7, false},   // 9
      {7, false},   // 10: rounds 6-9
      {15, true},   // 11: granted again; takes all 15: 8 past 7
      {15, true},   // 12: 8 past 7 against 0.29 x 11 tokens: extended
      {15, true},   // 13
  };
  for (const ShapePolicy &policy : {aggregate, all}) {
    const std::string under = name_of(policy.policy);
    bool asks[2] = {false, false};
    RecentRounds rounds[2] = {};
    std::uint64_t recorded[2] = {0, 0};
    std::size_t at = 0;
    for (const Step &step : course) {
      const std::uint8_t lane_asks[2] = {static_cast<std::uint8_t>(asks[0] ? 1 : 0),
                                         static_cast<std::uint8_t>(asks[1] ? 1 : 0)};
      const LaneEstimate estimates[2] = {estimate_of(rounds[0], recorded[0], policy.estimate_rounds),
                                         estimate_of(rounds[1], recorded[1], policy.estimate_rounds)};
      const bool extended = runs_extended(Named::kRule, lane_asks, 2, /*extended_lanes=*/4, policy, estimates);
      const bool want = policy.policy == Policy::kAggregate && step.extended;
      check(extended == want, under + ", a pair's course, round " + std::to_string(at) + " runs " +
                                  (want ? "extended" : "base"));
      const std::int32_t extent = extended ? kWindow : kBase;
      const std::int32_t accepted[2] = {std::min(step.accepted, extent), 2};
      for (int lane = 0; lane < 2; ++lane) {
        asks[lane] = asks_extended(rule, asks[lane], extended, extent, accepted[lane]);
        record_round(rounds[lane], recorded[lane], extended, extent, accepted[lane]);
      }
      check(!asks[1], under + ", a pair's course, round " + std::to_string(at) + ": lane 1 never asks");
      ++at;
    }
  }
}

void check_the_switch() {
  constexpr const char *kVariable = "IGNIS_ROUND_SHAPE";
  struct Spelled {
    const char *value;  // null: unset
    Named named;
  };
  const std::vector<Spelled> spelled = {
      {nullptr, Named::kRule}, {"", Named::kRule}, {"base", Named::kBase}, {"extended", Named::kExtended}};
  for (const Spelled &s : spelled) {
    set_environment(kVariable, s.value);
    const std::string name = s.value == nullptr ? "unset" : std::string("`") + s.value + "`";
    try {
      check(ignis::round_shape::read_switch() == s.named, name + " names its shape choice");
    } catch (const std::exception &e) {
      check(false, name + " is taken, not refused: " + e.what());
    }
  }
  for (const char *bad : {"Base", "rule", "wide", "narrow", "escalate:6", "1"}) {
    set_environment(kVariable, bad);
    std::string refused;
    try {
      (void)ignis::round_shape::read_switch();
    } catch (const std::invalid_argument &e) {
      refused = e.what();
    }
    check(refused.find(kVariable) != std::string::npos && refused.find(bad) != std::string::npos,
          std::string("`") + bad + "` is refused naming the variable and the value: " + refused);
  }
  set_environment(kVariable, nullptr);
}

// Table 7: a run of verify rounds whose shapes alternate, each shape's span and the host's gap
// after each round fixed, so each shape's true period is known; then a pause. GitHub #719: and the
// mixed rounds (round shape 2 here, the mixed block's own in the table) among them.
void check_the_round_timing() {
  constexpr float kSpan[3] = {14.4f, 16.6f, 15.2f};  // ms: a base round's graph, an extended round's, a mixed one's
  constexpr float kGap = 0.5f;                // ms: the host between a round's end and the next launch
  const std::vector<std::uint8_t> shapes = {0, 0, 1, 2, 1, 2, 1, 0, 1, 2, 0, 1, 2, 1, 1, 0, 2, 0, 1, 2, 0, 1};
  std::vector<float> pre;
  std::vector<float> post;
  float clock = 3.0f;
  for (const std::uint8_t shape : shapes) {
    pre.push_back(clock);
    post.push_back(clock + kSpan[shape]);
    clock += kSpan[shape] + kGap;
  }
  // The round after the run begins 80 ms after the last one ended: a request between them.
  pre.push_back(post.back() + 80.0f);
  ignis::round_timing::Blocks blocks;
  for (std::size_t r = 0; r + 1 < pre.size(); ++r) {
    ignis::round_timing::add(blocks, shapes[r] == 2 ? ignis::round_timing::kMixedBlock : shapes[r], post[r] - pre[r], pre[r + 1] - pre[r], pre[r + 1] - post[r]);
  }
  double periods = 0.0;
  std::size_t rounds = 0;
  for (std::size_t shape = 0; shape < 3; ++shape) {
    const ignis::round_timing::Block &block = blocks[shape == 2 ? ignis::round_timing::kMixedBlock : shape];
    const std::size_t of_shape = static_cast<std::size_t>(std::count(shapes.begin(), shapes.end(), shape));
    const std::string name = shape == 0 ? "base" : shape == 1 ? "extended" : "mixed";
    check(block.span.size() == of_shape, name + ": a span for each of its " + std::to_string(of_shape) + " rounds");
    const bool spans = std::all_of(block.span.begin(), block.span.end(),
                                   [&](float us) { return std::fabs(us - kSpan[shape] * 1000.0f) < 1.0f; });
    const bool own = std::all_of(block.period.begin(), block.period.end(), [&](float us) {
      return std::fabs(us - (kSpan[shape] + kGap) * 1000.0f) < 1.0f;
    });
    const bool gaps = std::all_of(block.gap.begin(), block.gap.end(),
                                  [&](float us) { return std::fabs(us - kGap * 1000.0f) < 1.0f; });
    check(spans && own && gaps, name + ": every span, period and gap is its own rounds' (" +
                                    std::to_string(kSpan[shape] + kGap) + " ms a period), not its neighbours'");
    for (const float us : block.period) {
      periods += us / 1000.0;
    }
    rounds += block.period.size();
  }
  // The last round of the run is followed by a pause: its span counts, its period does not.
  check(rounds == shapes.size() - 1, "every round but the one before the pause has a period");
  const double wall = pre[shapes.size() - 1] - pre[0];
  check(std::fabs(periods - wall) < 0.01, "the periods sum to the run's wall time (" + std::to_string(periods) +
                                              " ms against " + std::to_string(wall) + ")");
  // A shape past the blocks' is the base shape's, never out of bounds; #713: the one past the round
  // shapes' is the mixed rounds' own.
  ignis::round_timing::Blocks spare;
  ignis::round_timing::add(spare, IGNIS_ROUND_SHAPES_MAX + 1, 1.0f, 2.0f, 1.0f);
  check(spare[0].span.size() == 1, "a shape past the blocks counts as the base shape");
  ignis::round_timing::add(spare, ignis::round_timing::kMixedBlock, 1.0f, 2.0f, 1.0f);
  check(spare[ignis::round_timing::kMixedBlock].span.size() == 1 && spare[0].span.size() == 1,
        "a mixed round counts in the mixed rounds' block");
}

// Table 9: a mixed round's layout, and its records over the vendored planner's layouts.
ninfer::GdnReplayRecordSpec records_27b(std::int32_t capacity, std::int32_t width) {
  ninfer::GdnReplayRecordSpec spec{};
  spec.layers = 48;
  spec.record_capacity = capacity;
  spec.width = width;
  spec.conv_channels = 10240;
  spec.qk_heads = 16;
  spec.value_heads = 48;
  spec.key_dim = 128;
  spec.value_dim = 128;
  return spec;
}

struct Span {
  std::size_t begin;
  std::size_t end;
};

// The bytes of rows [first, first + rows) of every layer in each of the four planes of `layout`.
void row_spans(const ninfer::GdnReplayRecordLayout &layout, std::uint32_t first, std::uint32_t rows,
               std::vector<Span> &out) {
  const std::int32_t capacity = layout.spec.record_capacity;
  const std::size_t outer = static_cast<std::size_t>(layout.spec.layers) * capacity;
  for (const ninfer::TensorRegion *plane : {&layout.conv, &layout.key, &layout.value, &layout.gate}) {
    const std::size_t row_bytes = plane->region.bytes / outer;
    for (std::int32_t layer = 0; layer < layout.spec.layers; ++layer) {
      for (std::uint32_t r = first; r < first + rows; ++r) {
        const std::size_t at = plane->region.offset + (static_cast<std::size_t>(layer) * capacity + r) * row_bytes;
        out.push_back(Span{at, at + row_bytes});
      }
    }
  }
}

bool disjoint_within(std::vector<Span> spans, std::size_t backing) {
  std::sort(spans.begin(), spans.end(), [](const Span &a, const Span &b) { return a.begin < b.begin; });
  for (std::size_t i = 0; i < spans.size(); ++i) {
    if (spans[i].end > backing || (i + 1 < spans.size() && spans[i].end > spans[i + 1].begin)) {
      return false;
    }
  }
  return true;
}

ninfer::GdnReplayRecordLayout plan(const ninfer::GdnReplayRecordSpec &spec, std::size_t &bytes) {
  ninfer::LayoutBuilder builder;
  const ninfer::GdnReplayRecordLayout layout = ninfer::plan_gdn_replay_records(builder, spec);
  bytes = builder.finish(256, "records");
  return layout;
}

void check_the_mixed_layout() {
  using ignis::round_layout::Layout;
  using ignis::round_layout::layout_of;
  const Layout one_one = layout_of(1, kWindow, 1, kBase);
  check(one_one.columns() == 24 && one_one.drafts() == 22 && !one_one.uniform(), "1+1 at 15: 24 columns, 22 drafts");
  check(one_one.column_of(0) == 0 && one_one.draft_of(0) == 0 && one_one.groups[0].first_record_row == 0 &&
            one_one.column_of(1) == 16 && one_one.draft_of(1) == 15 && one_one.groups[1].first_record_row == 2,
        "1+1 at 15: the extended lane at column 0, draft 0, record row 0; the base lane at 16, 15, row 2");
  const Layout at_ten = layout_of(1, 10, 1, kBase);
  check(at_ten.column_of(1) == 11 && at_ten.draft_of(1) == 10 && at_ten.groups[1].first_record_row == 2,
        "1+1 at 10: the base lane's columns from 11, its drafts from 10, its records from row 2");
  const Layout two_four = layout_of(2, kWindow, 4, kBase);
  check(two_four.column_of(5) == 56 && two_four.draft_of(5) == 51 && two_four.groups[1].first_record_row == 4,
        "2+4 at 15: the last base lane at column 56, draft 51; the base records from row 4");
  check(ignis::round_layout::mix_index(one_one) == 0 && ignis::round_layout::mix_index(layout_of(3, 15, 2, 7)) == 11 &&
            ignis::round_layout::mix_index(layout_of(4, 15, 1, 7)) < 0 &&
            ignis::round_layout::mix_index(ignis::round_layout::uniform(2, 15, true)) < 0,
        "the twelve mixes: 1+1 first, 3+2 last, none of 4+1 or a uniform round");
  for (const auto &mix : ignis::round_layout::kMixTable) {
    check(layout_of(mix[0], kWindow, mix[1], kBase).columns() <= ignis::round_layout::kMaxColumns,
          "mix " + std::to_string(mix[0]) + "+" + std::to_string(mix[1]) + " fits 64 columns");
  }

  // At 15 drafts: a load of 2 lanes and one of 8, the extended records at min(lanes, 4) rows of 16,
  // the base shape's at `lanes` rows of 8, one backing the larger; the mixed base layout at the
  // extended one's stride.
  for (const std::uint32_t lanes : {2u, 8u}) {
    const std::uint32_t capacity = std::min(lanes, 4u);
    const std::uint32_t base_capacity = ignis::round_layout::mixed_base_record_capacity(capacity, 16, 8, 8);
    const std::string at = std::to_string(lanes) + " lanes at 15";
    check(base_capacity == 2 * capacity, at + ": the base layout of " + std::to_string(base_capacity) + " rows");
    std::size_t extended_bytes = 0;
    std::size_t own_bytes = 0;
    std::size_t base_bytes = 0;
    const ninfer::GdnReplayRecordLayout extended =
        plan(records_27b(static_cast<std::int32_t>(capacity), 16), extended_bytes);
    (void)plan(records_27b(static_cast<std::int32_t>(lanes), 8), own_bytes);
    const ninfer::GdnReplayRecordLayout base = plan(records_27b(static_cast<std::int32_t>(base_capacity), 8), base_bytes);
    const std::size_t backing = std::max(extended_bytes, own_bytes);
    check(base_bytes <= backing && base.conv.region.offset == extended.conv.region.offset &&
              base.key.region.offset == extended.key.region.offset &&
              base.value.region.offset == extended.value.region.offset &&
              base.gate.region.offset == extended.gate.region.offset,
          at + ": the mixed base layout's planes start where the extended one's do, inside the backing");
    std::uint32_t mixes = 0;
    for (const auto &mix : ignis::round_layout::kMixTable) {
      if (mix[0] > capacity || mix[0] + mix[1] > lanes) {
        continue;
      }
      ++mixes;
      const Layout layout = layout_of(mix[0], kWindow, mix[1], kBase);
      std::vector<Span> spans;
      row_spans(extended, layout.groups[0].first_record_row, layout.groups[0].lanes, spans);
      row_spans(base, layout.groups[1].first_record_row, layout.groups[1].lanes, spans);
      const std::string name = at + ", mix " + std::to_string(mix[0]) + "+" + std::to_string(mix[1]);
      check(ignis::round_layout::records_hold(layout, capacity, base_capacity) && disjoint_within(spans, backing),
            name + ": every layer's extended and base rows, in all four planes, apart and inside the backing");
    }
    check(mixes == (lanes == 2 ? 1u : 12u), at + ": " + std::to_string(mixes) + " mixes fit the load");
  }

  // At 10 and 8 drafts no base layout has the extended stride (4 x 11, 2 x 11, 4 x 9, 2 x 9 columns
  // a layer); the base shape's own layout puts a 1+1 round's base rows over extended rows of another
  // layer, and its row 2 is not the base lane's column 11.
  for (const std::uint32_t window : {10u, 8u}) {
    for (const std::uint32_t capacity : {2u, 4u}) {
      check(ignis::round_layout::mixed_base_record_capacity(capacity, window + 1, 8, 8) == 0,
            std::to_string(capacity) + " extended rows at " + std::to_string(window) + " drafts: no base layout");
    }
  }
  std::size_t extended_bytes = 0;
  std::size_t own_bytes = 0;
  const ninfer::GdnReplayRecordLayout extended_ten = plan(records_27b(4, 11), extended_bytes);
  const ninfer::GdnReplayRecordLayout own = plan(records_27b(8, 8), own_bytes);
  std::vector<Span> spans;
  row_spans(extended_ten, 0, 1, spans);
  row_spans(own, 2, 1, spans);
  check(!disjoint_within(spans, std::max(extended_bytes, own_bytes)),
        "1+1 at 10 over the base shape's own layout: base rows land on extended rows");
  check(!ignis::round_layout::records_hold(at_ten, 4, 8), "1+1 at 10: refused, its row 2 is not its column 11");

  // The base fold's leading rows.
  std::array<std::int32_t, 4> out{};
  const std::array<std::int32_t, 4> held = {0, 1, 2, 3};
  const std::int32_t five = 5;
  const std::int32_t zero = 0;
  const std::int32_t one = 1;
  check(ignis::round_layout::leading_fold_slots(&five, 1, 2, 8, out.data()) && out[0] == 0 && out[1] == 1,
        "leading fold rows beside a base lane in slot 5: slots 0 and 1");
  check(ignis::round_layout::leading_fold_slots(&zero, 1, 2, 8, out.data()) && out[0] == 1 && out[1] == 2,
        "beside a base lane in slot 0: slots 1 and 2");
  check(ignis::round_layout::leading_fold_slots(held.data(), 4, 4, 8, out.data()) && out[0] == 4 && out[3] == 7,
        "2+4's four leading rows beside base lanes in 0..3: slots 4..7");
  check(!ignis::round_layout::leading_fold_slots(&one, 1, 2, 2, out.data()),
        "a pool of 2 slots has no 2 leading rows beside a base lane: refused");
}

} // namespace

int main() {
  check_the_rule();
  check_the_thresholds();
  check_the_policy();
  check_the_estimates();
  check_aggregate();
  check_the_policy_options();
  check_the_grant();
  check_a_lanes_course();
  check_a_pairs_course();
  check_the_switch();
  check_the_round_timing();
  check_the_mixed_layout();
  if (g_failed != 0) {
    std::fprintf(stderr, "round_shape: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("round_shape: all checks passed\n");
  return 0;
}
