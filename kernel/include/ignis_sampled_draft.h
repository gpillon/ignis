/* ignis kernel leaf: the sampled draft's device arithmetic over explicit
 * uniforms -- a test-only seam (GitHub #576, spec #574, ADR 0090).
 *
 * NOT part of the serving path, and no serving code calls it. The verify
 * round draws its uniforms from the sampler's counter-based RNG, keyed by a
 * lane's seed and position; these entries run the same device functions
 * (kernel/src/sampled_draft.cuh) on uniforms the caller names, so a test can
 * sweep them over a grid and hold the kernels to the double-precision oracle
 * the host reference is held to (crates/core/tests/support/draft_oracle.rs).
 * Each allocates its own device memory, runs, copies its result back and
 * synchronizes: it touches no model and no sequence.
 *
 * Each returns 0, or -1 with `ignis_sampled_draft_last_error()` set: a null
 * argument, an empty distribution or grid, a drafter of more than
 * IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES candidates, or a CUDA error.
 */
#ifndef IGNIS_SAMPLED_DRAFT_H
#define IGNIS_SAMPLED_DRAFT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The candidates of a draft column: the selector's top-k. */
#define IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES 16

/* What `ignis_sampled_draft_verdicts` writes where the draft was taken. */
#define IGNIS_SAMPLED_DRAFT_ACCEPTED (-1)

/* The accept's verdict on `draft` at every point of a `cells` x `cells` grid:
 * `out_tokens[i * cells + j]` is the verdict with the accept's uniform
 * `grid[i]` and the refusal's `grid[j]` -- IGNIS_SAMPLED_DRAFT_ACCEPTED where
 * the draft was taken, else the token the refusal drew. The target's
 * distribution is `target_count` (id, probability) pairs in the sampler's
 * order, the drafter's `drafter_count` in the candidates' order. */
int32_t ignis_sampled_draft_verdicts(const int32_t *target_ids, const float *target_probs,
                                     uint32_t target_count, const int32_t *drafter_ids,
                                     const float *drafter_probs, uint32_t drafter_count,
                                     int32_t draft, const float *grid, uint32_t cells,
                                     int32_t *out_tokens);

/* The draft drawn from `q` over `count` candidates at each of `cells`
 * uniforms `grid`: `out_candidates[i]` is the candidate index drawn with
 * `grid[i]`. */
int32_t ignis_sampled_draft_draws(const float *q, uint32_t count, const float *grid,
                                  uint32_t cells, int32_t *out_candidates);

/* GitHub #577: the teacher-forced verdict (`ignis_forced_sampled_verdict`)
 * on the recorded token `recorded` at each of `cells` coin uniforms `grid`:
 * `out_taken[i]` is 1 where the recorded token is taken with the coin
 * `grid[i]`, else 0. `live_taken` (0 or 1) and `live_token` are the live
 * column's verdict on `live_draft`, as `ignis_sampled_draft_verdicts` writes
 * it (`live_token` is the draft where it was taken). The distributions as
 * there. */
int32_t ignis_sampled_draft_forced_verdicts(const int32_t *target_ids, const float *target_probs,
                                            uint32_t target_count, const int32_t *drafter_ids,
                                            const float *drafter_probs, uint32_t drafter_count,
                                            int32_t recorded, int32_t live_draft, int32_t live_taken,
                                            int32_t live_token, const float *grid, uint32_t cells,
                                            uint8_t *out_taken);

/* The drafter's distribution over `count` candidates from their transition
 * scores, at `temperature` and `top_p`, into `out_q[count]`. */
int32_t ignis_sampled_drafter_distribution(const float *scores, uint32_t count, float temperature,
                                           float top_p, float *out_q);

const char *ignis_sampled_draft_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_SAMPLED_DRAFT_H */
