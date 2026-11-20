// GitHub #689 (spec #684): the shape rule and the shape policy `all` of a DFlash2 load of two round
// shapes, as pure functions from tables (kernel/src/round_shape.h), and the measurement switch
// IGNIS_ROUND_SHAPE that forces a shape over them -- OURS, host-only.
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
//   3. The shape policy `all`: a round of 1..4 lanes is extended when every lane in it asks, a round
//      of 5..8 lanes always at the base shape, a load with no extended shape never extended, and a
//      round with a lane whose commit the host cuts at the base shape (#687).
//   4. The switch as a round reads it: unset (or empty) the rule and the policy, `base` and
//      `extended` forcing a shape over them, any other value refused naming the variable.
//   5. The round timing where shapes alternate (kernel/src/round_timing.h): each round's period
//      and gap are its own, to the next round's launch whatever its shape, so each shape's periods
//      are its rounds' and a run's periods sum to its wall time.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "round_shape.h"
#include "round_timing.h"

#include <algorithm>
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
using ignis::round_shape::Named;
using ignis::round_shape::runs_extended;
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

// Table 3: the round's shape from its lanes' asks, under the policy `all` and the forcing switch.
void check_the_policy() {
  constexpr std::uint32_t kExtendedLanes = 4;
  constexpr std::size_t kMostLanes = 8;
  for (const Named named : {Named::kRule, Named::kBase, Named::kExtended}) {
    const std::string under = named == Named::kRule   ? "under the rule (unset)"
                              : named == Named::kBase ? "under `base`"
                                                      : "under `extended`";
    for (std::size_t lanes = 1; lanes <= kMostLanes; ++lanes) {
      const std::string at = under + ", " + std::to_string(lanes) + " lanes";
      const std::vector<std::uint8_t> all(lanes, 1);
      const std::vector<std::uint8_t> none(lanes, 0);
      // A load with no extended shape never runs one, whatever its lanes ask.
      check(!runs_extended(named, all.data(), lanes, /*extended_lanes=*/0, false), at + ", no extended shape: base");
      // A round with a lane whose commit the host cuts keeps the base shape (#687; #691 changes it).
      check(!runs_extended(named, all.data(), lanes, kExtendedLanes, /*host_cut=*/true), at + ", a host cut: base");
      const bool every = runs_extended(named, all.data(), lanes, kExtendedLanes, false);
      const bool nobody = runs_extended(named, none.data(), lanes, kExtendedLanes, false);
      if (lanes > kExtendedLanes) {
        check(!every && !nobody, at + ": past the extended shape's 4 lanes, base whatever they ask");
        continue;
      }
      switch (named) {
      case Named::kBase:
        check(!every && !nobody, at + ": base, asked or not");
        break;
      case Named::kExtended:
        check(every && nobody, at + ": extended, asked or not");
        break;
      case Named::kRule:
        check(every, at + ", every lane asking: extended");
        check(!nobody, at + ", no lane asking: base");
        // One lane that does not ask, wherever it rides, keeps the round at the base shape: under
        // `all` no lane pays for another.
        for (std::size_t quiet = 0; quiet < lanes; ++quiet) {
          std::vector<std::uint8_t> asks(lanes, 1);
          asks[quiet] = 0;
          check(!runs_extended(named, asks.data(), lanes, kExtendedLanes, false),
                at + ", lane " + std::to_string(quiet) + " not asking: base");
        }
        break;
      }
    }
  }
}

// One lane through rounds under the default rule and the policy, the rule and the round's shape
// together: base until a full window is taken whole, extended while it takes 6, base again after a
// short take, and a round the budget cut short leaving the ask as it was.
void check_a_lanes_course() {
  const ShapeRule rule{};
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
  std::size_t at = 0;
  for (const Step &step : course) {
    const std::uint8_t lane = asks ? 1 : 0;
    const bool extended = runs_extended(Named::kRule, &lane, 1, /*extended_lanes=*/4, false);
    check(extended == step.extended, "a lane's course, round " + std::to_string(at) + " runs " +
                                         (step.extended ? "extended" : "base"));
    const std::int32_t extent = step.budget_extent >= 0 ? step.budget_extent : extended ? kWindow : kBase;
    asks = asks_extended(rule, asks, extended, extent, step.accepted);
    ++at;
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

// Table 5: a run of verify rounds whose shapes alternate, each shape's span and the host's gap
// after each round fixed, so each shape's true period is known; then a pause.
void check_the_round_timing() {
  constexpr float kSpan[2] = {14.4f, 16.6f};  // ms: a base round's graph, an extended round's
  constexpr float kGap = 0.5f;                // ms: the host between a round's end and the next launch
  const std::vector<std::uint8_t> shapes = {0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1};
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
    ignis::round_timing::add(blocks, shapes[r], post[r] - pre[r], pre[r + 1] - pre[r], pre[r + 1] - post[r]);
  }
  double periods = 0.0;
  std::size_t rounds = 0;
  for (std::size_t shape = 0; shape < 2; ++shape) {
    const ignis::round_timing::Block &block = blocks[shape];
    const std::size_t of_shape = static_cast<std::size_t>(std::count(shapes.begin(), shapes.end(), shape));
    const std::string name = shape == 0 ? "base" : "extended";
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
  // A shape past the blocks' is the base shape's, never out of bounds.
  ignis::round_timing::Blocks spare;
  ignis::round_timing::add(spare, IGNIS_ROUND_SHAPES_MAX, 1.0f, 2.0f, 1.0f);
  check(spare[0].span.size() == 1, "a shape past the blocks counts as the base shape");
}

} // namespace

int main() {
  check_the_rule();
  check_the_thresholds();
  check_the_policy();
  check_a_lanes_course();
  check_the_switch();
  check_the_round_timing();
  if (g_failed != 0) {
    std::fprintf(stderr, "round_shape: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("round_shape: all checks passed\n");
  return 0;
}
