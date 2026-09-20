// Per-lane permitted token sets for the decode round (GitHub #242, ADR 0034).
//
// A **constrained decode** restricts a lane's next token to a declared set of
// ids — the ten digits for a number read digit by digit, or a single id for a
// literal the caller forces. ADR 0034 puts that set in the sampling ABI rather
// than in the host: the alternative is to bring a 248,320-wide logits row back
// per step, which is six synchronizations and 607 KB per three-digit number,
// and it would also cost the seam its composition with temperature, seeds and
// speculation — the leaf still samples and still returns a token id.
//
// Two device steps, either side of the vendored sampler
// (`ninfer::ops::sample`, ADR 0010 — not patched):
//
//   * `ignis_permit_mask` drives every column that is not in a lane's set to
//     a value no softmax or argmax can pick, *before* the draw. Greedy then
//     takes the set's argmax and temperature draws from the set alone, so the
//     constraint composes with the sampler rather than replacing it.
//   * `ignis_permit_probability` reports, *after* the draw, how much of the
//     restricted distribution sat on the token that won. That is one float per
//     lane, not a logits row: the per-digit confidence a number's uncertainty
//     is summed from (spec 06) has to come from somewhere, and this is the
//     cheapest thing that is not the host reading logits.
#ifndef IGNIS_PERMITTED_TOKENS_H
#define IGNIS_PERMITTED_TOKENS_H

#include <cstdint>

#include <cuda_runtime.h>

// Drive every non-permitted column of each constrained lane's logits to
// `-1e30`. `logits` is the round's BF16 `[vocab, lanes]` staging buffer in the
// sampler's own column layout — lane `b`'s column is contiguous at `b * vocab`.
// `permitted` is `[lanes][max_permitted]` device I32 and `counts` is `[lanes]`;
// a lane whose count is 0 is left exactly as it was, which is what makes this
// safe to run on every round rather than only on constrained ones.
//
// `-1e30` rather than `-inf`: it is representable in BF16, it underflows every
// exponential the sampler takes, and it cannot produce the `inf - inf` NaN a
// max-subtraction would if a row were ever entirely masked.
int32_t ignis_permit_mask(void *logits, int32_t vocab, uint32_t lanes,
                          const int32_t *permitted, const int32_t *counts,
                          int32_t max_permitted, cudaStream_t stream);

// For each constrained lane, the softmax probability of `chosen[lane]` over
// that lane's permitted ids alone, written to `out[lane]`. An unconstrained
// lane gets 0. Reads the same logits the mask left, and only the permitted
// columns of them, which the mask does not touch.
//
// Temperature-free on purpose: this is the model's own confidence in the token
// it committed, which is what a per-digit trace is read as, not the sampler's
// probability of having drawn it.
int32_t ignis_permit_probability(const void *logits, int32_t vocab, uint32_t lanes,
                                 const int32_t *permitted, const int32_t *counts,
                                 int32_t max_permitted, const int32_t *chosen, float *out,
                                 cudaStream_t stream);

#endif  // IGNIS_PERMITTED_TOKENS_H
