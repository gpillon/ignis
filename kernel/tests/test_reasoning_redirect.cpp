// GitHub #315 (ADR 0048): the reasoning redirect's cut -- OURS. The one host
// function both leaves call wherever they assign a successor
// (kernel/include/ignis_reasoning_redirect.h): the prefill's draw, the plain
// round, the verify round's commit.
//
// Pure host arithmetic over token ids, and built without any CUDA library
// (kernel/tests/CMakeLists.txt links nothing), so this test can create no
// CUDA context: it runs on a card another process holds.

#include "ignis_reasoning_redirect.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label);
    ++failures;
  }
}

constexpr int32_t kEos = 2;
constexpr int32_t kImEnd = 3;
constexpr int32_t kThinkEnd = 9;
const std::vector<int32_t> kStops = {kEos};

IgnisRoundCut cut(const std::vector<int32_t> &run, int32_t draw, int32_t close_id,
                  const std::vector<int32_t> &stops = kStops) {
  return ignis_cut_and_redirect(run.data(), static_cast<int32_t>(run.size()), draw, stops.data(),
                                static_cast<uint32_t>(stops.size()), close_id);
}

bool is(const IgnisRoundCut &got, int32_t committed, int32_t next_pending, bool redirected) {
  return got.committed == committed && got.next_pending == next_pending &&
         got.redirected == redirected;
}

// The verify round's rule before the redirect existed (step.cu, speculative.cu):
// the run is cut at its first stop id inclusive, and the next pending token is
// the successor of the run's last committed token -- the draft after it, or the
// round's draw.
IgnisRoundCut today(const std::vector<int32_t> &run, int32_t draw) {
  const auto n = static_cast<int32_t>(run.size());
  for (int32_t j = 0; j < n; ++j) {
    if (run[static_cast<std::size_t>(j)] == kEos) {
      return {j + 1, j + 1 < n ? run[static_cast<std::size_t>(j) + 1] : draw, false};
    }
  }
  return {n, draw, false};
}

}  // namespace

int main() {
  // ---- no stop: nothing to cut, nothing to redirect -------------------------
  expect(is(cut({5, 6, 7}, 8, kThinkEnd), 3, 8, false), "a run with no stop id commits whole");
  expect(is(cut({5}, 8, kThinkEnd), 1, 8, false), "a plain round with no stop id draws freely");
  expect(is(cut({}, 8, kThinkEnd), 0, 8, false), "a prefill's free draw stands");

  // ---- the block closed: today's inclusive cut, unchanged -------------------
  expect(is(cut({5, kEos, 7}, 8, -1), 2, 7, false), "a stop with the block closed cuts inclusively");
  expect(is(cut({5}, kEos, -1), 1, kEos, false), "a drawn stop with the block closed stays a stop");
  expect(is(cut({}, kEos, -1), 0, kEos, false), "a prefill's drawn stop with the block closed stays");

  // ---- a stop as the next pending, the block open: redirected ---------------
  expect(is(cut({5}, kEos, kThinkEnd), 1, kThinkEnd, true), "a plain round's drawn stop becomes </think>");
  expect(is(cut({}, kEos, kThinkEnd), 0, kThinkEnd, true), "a prefill's drawn stop becomes </think>");
  expect(is(cut({5, 6, 7}, kEos, kThinkEnd), 3, kThinkEnd, true),
         "a verify round's correction or bonus stop becomes </think>");

  // ---- an accepted draft that is a stop, the block open: cut before it -------
  expect(is(cut({5, 6, kEos, 7}, 8, kThinkEnd), 2, kThinkEnd, true),
         "an accepted stop draft is cut before, and </think> is the next pending");
  expect(is(cut({5, kEos}, 8, kThinkEnd), 1, kThinkEnd, true),
         "the first accepted draft a stop: the anchor alone is committed");

  // ---- a </think> earlier in the run closes the block for the round ---------
  expect(is(cut({5, kThinkEnd, 6, kEos}, 8, kThinkEnd), 4, 8, false),
         "a stop after a committed </think> cuts inclusively, unredirected");
  expect(is(cut({5, kThinkEnd}, kEos, kThinkEnd), 2, kEos, false),
         "a draw after an accepted </think> is not redirected");
  expect(is(cut({kThinkEnd}, kEos, kThinkEnd), 1, kEos, false),
         "an anchor </think> (the last round's redirect) closes the block: no second </think>");

  // ---- the anchor is never a candidate --------------------------------------
  expect(is(cut({kEos, 6}, 8, kThinkEnd), 1, 6, false),
         "an anchor stop was decided when it was drawn: today's inclusive cut");
  expect(is(cut({kEos}, 8, kThinkEnd), 1, 8, false), "a plain round's anchor stop: today's cut");

  // ---- the lane's stop set, whole --------------------------------------------
  const std::vector<int32_t> both = {kEos, kImEnd};
  expect(is(cut({5}, kImEnd, kThinkEnd, both), 1, kThinkEnd, true), "any of the lane's stop ids redirects");
  expect(is(cut({5, kImEnd, 6}, 8, kThinkEnd, both), 1, kThinkEnd, true), "any stop id cuts before");
  expect(is(cut({5}, kEos, kThinkEnd, {}), 1, kEos, false), "a lane with no stop ids has nothing to redirect");

  // ---- no close id: every shape is today's ----------------------------------
  const std::vector<std::vector<int32_t>> runs = {
      {5}, {kEos}, {5, 6}, {5, kEos}, {kEos, 6}, {5, 6, kEos, 7}, {5, kThinkEnd, kEos}, {kThinkEnd}};
  for (const auto &run : runs) {
    for (const int32_t draw : {8, kEos, kThinkEnd}) {
      const IgnisRoundCut got = cut(run, draw, -1);
      const IgnisRoundCut want = today(run, draw);
      expect(is(got, want.committed, want.next_pending, false), "reasoning_close_id -1 is today's cut");
    }
  }

  if (failures != 0) {
    std::fprintf(stderr, "reasoning redirect test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("reasoning redirect test: ok\n");
  return 0;
}
