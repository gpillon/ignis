// GitHub #267 (test-only): a teacher-forced verify round's verdict -- OURS.
// The one host function the 27B verify round calls per lane when its options
// carry `forced_tokens` (kernel/src/teacher_forcing.h): the leading drafts the
// forced text agrees with, the forced token after them as the correction, and
// the refusal of a successor outside the vocabulary.
//
// Pure host arithmetic over token ids, built without any CUDA library
// (kernel/tests/CMakeLists.txt links nothing), so this test can create no
// CUDA context: it runs on a card another process holds.

#include "teacher_forcing.h"

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

constexpr int32_t kVocab = 100;
constexpr int32_t kImEnd = 3;
constexpr int32_t kAnchor = 50;
const std::vector<int32_t> kStops = {kImEnd, 2};

struct Got {
  IgnisForcedVerdict verdict;
  std::vector<int32_t> licensed;
};

// A round of window 4 (5 forced columns), `extent` drafts proposed.
Got verdict(const std::vector<int32_t> &drafts, const std::vector<int32_t> &forced, int32_t extent,
            int32_t anchor = kAnchor) {
  std::vector<int32_t> licensed(forced.size(), -7);
  const IgnisForcedVerdict v =
      ignis_teacher_forced_verdict(anchor, drafts.data(), forced.data(), extent, kVocab, kStops.data(),
                                   static_cast<uint32_t>(kStops.size()), licensed.data());
  return {v, licensed};
}

bool licensed_is(const Got &got, const std::vector<int32_t> &want) {
  for (std::size_t j = 0; j < want.size(); ++j) {
    if (got.licensed[j] != want[j]) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  // ---- the agreement rule ----------------------------------------------------
  {
    const Got got = verdict({10, 11, 12, 13}, {10, 11, 40, 41, 42}, 4);
    expect(!got.verdict.refused && got.verdict.accepted == 2, "the leading agreement is accepted");
    expect(licensed_is(got, {10, 11, 40}), "the agreed drafts, then the forced token after them");
  }
  {
    const Got got = verdict({10, 11, 12, 13}, {20, 11, 12, 13, 14}, 4);
    expect(!got.verdict.refused && got.verdict.accepted == 0, "a first-column disagreement accepts nothing");
    expect(licensed_is(got, {20}), "the forced token is the correction");
  }
  {
    const Got got = verdict({10, 11, 12, 13}, {10, 11, 12, 13, 14}, 4);
    expect(!got.verdict.refused && got.verdict.accepted == 4, "a whole agreement accepts every draft");
    expect(licensed_is(got, {10, 11, 12, 13, 14}), "the forced token after them is the bonus");
  }
  {
    const Got got = verdict({10, 11, 12, 13}, {10, 11, 12, 13, 14}, 2);
    expect(!got.verdict.refused && got.verdict.accepted == 2, "never more than the lane's extent");
    expect(licensed_is(got, {10, 11, 12}), "past the extent the forced token is the correction");
  }

  // ---- the successor's range -------------------------------------------------
  {
    const Got got = verdict({10, 11, 12, 13}, {10, 11, kVocab, 41, 42}, 4);
    expect(got.verdict.refused && got.verdict.accepted == 2, "a successor past the vocabulary is refused");
    expect(got.licensed[0] == -7, "a refused verdict writes nothing");
  }
  expect(verdict({10, 11, 12, 13}, {10, -2, 12, 13, 14}, 4).verdict.refused,
         "a negative successor other than the pad is refused");
  expect(verdict({10, 11, 12, 13}, {10, 11, -1, -1, -1}, 4).verdict.refused,
         "the pad after a recording with no stop would be the pending token: refused");
  expect(verdict({10, 11, 12, 13}, {-1, -1, -1, -1, -1}, 4).verdict.refused,
         "a recording that ends at its anchor without a stop: refused");
  expect(verdict({10, 11, 12, 13}, {10, 11, 12, 13, -1}, 4).verdict.refused,
         "the pad as a whole agreement's bonus with no stop: refused");
  {
    const Got got = verdict({10, kImEnd, 12, 13}, {10, kImEnd, -1, -1, -1}, 4);
    expect(!got.verdict.refused && got.verdict.accepted == 2, "the pad after an accepted stop is taken");
    expect(licensed_is(got, {10, kImEnd, -1}), "the stop ends the run, and nothing is pending after it");
  }
  {
    const Got got = verdict({10, 11, 12, 13}, {-1, -1, -1, -1, -1}, 4, kImEnd);
    expect(!got.verdict.refused && got.verdict.accepted == 0, "the pad after an anchor stop is taken");
    expect(licensed_is(got, {-1}), "a stop anchor ends the recording");
  }
  {
    // A stop mid-run cuts before the successor, which is never pending; it
    // must still be a vocabulary id or the pad.
    expect(!verdict({kImEnd, 11, 12, 13}, {kImEnd, 11, 40, 41, 42}, 4).verdict.refused,
           "a vocabulary successor after a stop mid-run is taken");
    expect(verdict({kImEnd, 11, 12, 13}, {kImEnd, 11, kVocab + 5, 41, 42}, 4).verdict.refused,
           "a successor past the vocabulary is refused after a stop too");
  }
  expect(!verdict({10, 11, 12, 13}, {10, 11, 0, 41, 42}, 4).verdict.refused, "id 0 is a vocabulary id");
  expect(!verdict({10, 11, 12, 13}, {10, 11, kVocab - 1, 41, 42}, 4).verdict.refused,
         "the last vocabulary id is one");

  // ---- GitHub #577: a forced sampling lane under the sampled draft ----------
  // Its run may stop at any column, so every successor up to its extent is
  // held to the rule above before the pass.
  const auto first_bad = [](std::vector<int32_t> forced, int32_t extent, int32_t anchor = kAnchor) {
    return ignis_teacher_forced_first_bad_successor(anchor, forced.data(), extent, kVocab, kStops.data(),
                                                    static_cast<uint32_t>(kStops.size()));
  };
  expect(first_bad({10, 11, 12, 13, 14}, 4) == -1, "a recording of vocabulary ids passes whole");
  expect(first_bad({10, kImEnd, -1, -1, -1}, 4) == -1, "pads after a recorded stop pass");
  expect(first_bad({-1, -1, -1, -1, -1}, 4, kImEnd) == -1, "pads after a stop anchor pass");
  expect(first_bad({10, 11, -1, -1, -1}, 4) == 2, "a pad with no stop before it is the first bad successor");
  expect(first_bad({10, 11, 12, kVocab, -1}, 4) == 3, "an id past the vocabulary is a bad successor");
  expect(first_bad({10, 11, -1, -1, -1}, 1) == -1, "successors past the extent are not the lane's");
  expect(first_bad({kImEnd, kVocab, -1, -1, -1}, 4) == 1, "an id past the vocabulary is bad after a stop too");

  // The device's licensed run is the recording's, or the round is refused.
  {
    const std::vector<int32_t> forced = {10, 11, 12, 13, 14};
    const auto recorded = [&](std::vector<int32_t> licensed, int32_t accepted, int32_t extent) {
      return ignis_teacher_forced_run_is_recorded(licensed.data(), forced.data(), accepted, extent);
    };
    expect(recorded({10, 11, 12, 0, 0}, 2, 4), "two recorded tokens taken, the third after them");
    expect(recorded({10, 11, 12, 13, 14}, 4, 4), "a whole run, the recorded token after it");
    expect(!recorded({10, 11, 40, 0, 0}, 2, 4), "a token after the run other than the recorded one is refused");
    expect(!recorded({10, 11, 12, 13, 14}, 4, 3), "more taken than the extent is refused");
  }

  if (failures != 0) {
    std::fprintf(stderr, "teacher forcing test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("teacher forcing test: ok\n");
  return 0;
}
