#pragma once
// GitHub #267 (test-only): a teacher-forced verify round's verdict for one
// lane (`ignis_decode_options::forced_tokens`, kernel/include/ignis_step.h)
// -- OURS. Host-only, no CUDA type: the CTest unit
// kernel/tests/test_teacher_forcing.cpp builds it with no CUDA library, so it
// runs on a card another process holds.
//
// The lane accepts the leading drafts its forced text agrees with, at most
// `extent` of them, and takes the forced token after them as its correction:
// `licensed[0..n)` are those drafts and `licensed[n]` is `forced[n]`, what
// the accept kernel writes for a target that sampled the text. The run is
// then `{anchor, drafts[0..n)}`, cut as any run is (ignis_cut_and_redirect),
// and `forced[n]` is its successor: the sequence's pending token.
//
// So `forced[n]` must be a vocabulary id. The one exception is the pad (-1)
// after a recording that ends at a stop id: the stop cuts the run, nothing
// recorded follows it, and the sequence is left with no pending token --
// refused by every round until a prefill gives it one, as the replay of
// recorded turns prefills the next turn's context. Any other id outside the
// vocabulary is refused: a pad past a recording that ends without a stop
// would leave a sequence that goes on with nothing to feed, and an id past
// the vocabulary would be the next round's embedding row. `forced[0..n)`
// need no check: they equal drafts, which are vocabulary ids.

#include "ignis_reasoning_redirect.h"

#include <cstdint>

struct IgnisForcedVerdict {
  int32_t accepted; /* n: the leading drafts the forced text agrees with, 0..extent */
  bool refused;     /* forced[n] cannot be the run's successor; licensed is unwritten */
};

inline IgnisForcedVerdict ignis_teacher_forced_verdict(int32_t anchor, const int32_t *drafts,
                                                       const int32_t *forced, int32_t extent,
                                                       int32_t vocab, const int32_t *stop_ids,
                                                       uint32_t stop_count, int32_t *licensed) {
  int32_t n = 0;
  while (n < extent && drafts[n] == forced[n]) {
    ++n;
  }
  bool stopped = ignis_is_stop_id(anchor, stop_ids, stop_count);
  for (int32_t j = 0; j < n && !stopped; ++j) {
    stopped = ignis_is_stop_id(drafts[j], stop_ids, stop_count);
  }
  const int32_t successor = forced[n];
  const bool in_vocabulary = successor >= 0 && successor < vocab;
  if (!in_vocabulary && !(successor == -1 && stopped)) {
    return {n, true};
  }
  for (int32_t j = 0; j < n; ++j) {
    licensed[j] = drafts[j];
  }
  licensed[n] = successor;
  return {n, false};
}

// GitHub #577 (test-only): under the sampled draft a forced sampling lane's
// verdict is the device's -- the teacher-forced coin of the accept kernel
// (kernel/src/sampled_draft.cuh, `ignis_forced_sampled_verdict`) -- so its
// run may stop at any column up to its extent, and the round checks every
// successor it could stop at before the pass: `forced[n]` for each n in
// 0..extent must be a vocabulary id, or the pad -1 after a stop id among the
// anchor and `forced[0..n)`, the rule above for every n. Returns the first n
// that breaks it, or -1. A well-formed recording passes whole: its pads
// follow its stop.
inline int32_t ignis_teacher_forced_first_bad_successor(int32_t anchor, const int32_t *forced,
                                                        int32_t extent, int32_t vocab,
                                                        const int32_t *stop_ids, uint32_t stop_count) {
  bool stopped = ignis_is_stop_id(anchor, stop_ids, stop_count);
  for (int32_t n = 0; n <= extent; ++n) {
    const int32_t successor = forced[n];
    const bool in_vocabulary = successor >= 0 && successor < vocab;
    if (!in_vocabulary && !(successor == -1 && stopped)) {
      return n;
    }
    stopped = stopped || ignis_is_stop_id(successor, stop_ids, stop_count);
  }
  return -1;
}

// GitHub #577 (test-only): whether a forced sampling lane's licensed run --
// what the device's forced verdict wrote, `accepted` recorded tokens then the
// one after them -- is the recording's. The round refuses one that is not:
// the device took something other than the recorded text.
inline bool ignis_teacher_forced_run_is_recorded(const int32_t *licensed, const int32_t *forced,
                                                 int32_t accepted, int32_t extent) {
  if (accepted < 0 || accepted > extent) {
    return false;
  }
  for (int32_t j = 0; j <= accepted; ++j) {
    if (licensed[j] != forced[j]) {
      return false;
    }
  }
  return true;
}
