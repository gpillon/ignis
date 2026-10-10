/* ignis kernel leaf: the reasoning redirect's cut (GitHub #315, ADR 0048).
 * OURS, host-only C++: no CUDA type crosses it, so the CTest unit
 * (kernel/tests/test_reasoning_redirect.cpp) builds it without a device.
 *
 * Every place a leaf assigns a sequence's successor -- the prefill's draw, the
 * plain round, the verify round's commit, on the 27B and on Flash-Next --
 * calls this one function with the round's committed run and its draw. A stop
 * id drawn while the lane's reasoning block is open becomes `close_id`
 * (`</think>`): a turn that ends inside its reasoning has no answer, and the
 * model was done thinking.
 *
 * The run is what the round feeds: `run[0]` is the anchor (the pending token
 * the previous call made ready), `run[1..]` the drafts the round accepted, in
 * order; `draw` is the successor of the run's last token -- the round's sample,
 * or a verify round's correction or bonus token. The successor of `run[j]` for
 * `j < run_count - 1` is `run[j + 1]` itself. A prefill is the empty run and
 * its draw; a plain round is `{anchor}` and its draw.
 *
 * The block is open while `close_id >= 0` -- the host's view, which lags by
 * the round it cannot see -- until a `close_id` in the run closes it for the
 * rest of the round, so a model that closed its own block and then ends an
 * empty answer is never given a second `</think>`.
 *
 *   - A stop id at `j >= 1` with the block open: the run is cut *before* it
 *     (`committed = j`) and `close_id` is the next pending token.
 *   - A stop id anywhere else: today's cut, at the first stop id inclusive,
 *     with the successor of the run's last committed token pending. The anchor
 *     is never a candidate: it was decided when it was drawn.
 *   - No stop id in the run: all of it is committed, and the draw is pending
 *     -- `close_id` in its place when it is a stop id and the block is open.
 *
 * With `close_id < 0` every shape is the cut the verify round made before
 * this existed. */
#ifndef IGNIS_REASONING_REDIRECT_H
#define IGNIS_REASONING_REDIRECT_H

#include <cstdint>

struct IgnisRoundCut {
  int32_t committed;    /* tokens of the run committed: 1..run_count, 0 for a prefill */
  int32_t next_pending; /* the successor the next round emits first */
  bool redirected;      /* next_pending is close_id in place of a drawn stop id */
};

inline bool ignis_is_stop_id(int32_t token, const int32_t *stop_ids, uint32_t stop_count) {
  for (uint32_t s = 0; s < stop_count; ++s) {
    if (stop_ids[s] == token) {
      return true;
    }
  }
  return false;
}

inline IgnisRoundCut ignis_cut_and_redirect(const int32_t *run, int32_t run_count, int32_t draw,
                                            const int32_t *stop_ids, uint32_t stop_count,
                                            int32_t close_id) {
  bool open = close_id >= 0;
  for (int32_t j = 0; j < run_count; ++j) {
    if (ignis_is_stop_id(run[j], stop_ids, stop_count)) {
      if (j >= 1 && open) {
        return {j, close_id, true};
      }
      return {j + 1, j + 1 < run_count ? run[j + 1] : draw, false};
    }
    if (run[j] == close_id) {
      open = false;
    }
  }
  if (open && ignis_is_stop_id(draw, stop_ids, stop_count)) {
    return {run_count, close_id, true};
  }
  return {run_count, draw, false};
}

#endif /* IGNIS_REASONING_REDIRECT_H */
