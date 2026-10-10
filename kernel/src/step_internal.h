// ignis kernel leaf -- the step ABI's shared program pieces, for a second
// program behind the same entry points (GitHub #302: Flash-Next's,
// kernel/src/flash_next/program.cu). Defined in kernel/src/step.cu, where
// the 27B's program uses them.

#pragma once

#include "ignis_seq.h"
#include "ignis_step.h"
#include "model_internal.h"

#include "ninfer/ops/sampling.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ignis::step {

// The step ABI's error channel (ignis_step_last_error).
void set_error(std::string message);

// `sampling.size` is this leaf's (ADR 0016).
bool sampling_size_ok(const ignis_sampling_params &sampling);

// A lane that declared no permitted set (P6-06, GitHub #242).
bool unconstrained(const ignis_sampling_params &sampling);

// The ABI's sampling parameters as the vendored sampler's config, against
// the sequence's penalty-count row.
ninfer::ops::SamplingConfig to_sampling_config(const ignis_sampling_params &abi,
                                               std::int32_t *token_counts);

// One sequence's draw from its single-row BF16 logits, through the model's
// single-draw staging, at `position` (the logical position of the token the
// draw succeeds): its permitted set masked, its probability reported.
int32_t sample_single(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                      const ninfer::Tensor &logits, const ignis_sampling_params &sampling,
                      std::int32_t purpose, std::int32_t position, int32_t *out_token_id,
                      float *out_permitted_prob);

// GitHub #315 (ADR 0048): an armed reasoning redirect reads the lane's stop
// ids wherever its successor is assigned; false when they are missing.
bool redirect_inputs_ok(const ignis_sampling_params &sampling);

// Occurrences moved in penalty-count rows (a count's device address, what to
// add, clamped at 0), one read and one write per distinct count, both sides
// synchronized: a verify round's rollback past its cut, and a redirect's move.
cudaError_t adjust_penalty_counts(const std::vector<std::pair<std::int32_t *, std::int32_t>> &moves,
                                  cudaStream_t stream);

// GitHub #315: the occurrence a temperature draw counted for the stop id it
// drew, moved to `close_id`, appended to `moves` (nothing for a greedy draw).
void redirect_penalty_move(const ninfer::ops::SamplingConfig &cfg, std::int32_t drawn,
                           std::int32_t close_id,
                           std::vector<std::pair<std::int32_t *, std::int32_t>> *moves);

// GitHub #315: a span's draw, the sequence's pending token, redirected when it
// is a stop id drawn with the lane's reasoning block open; whether it was into
// `*out_redirected` (or nowhere). After the span's device work is confirmed.
int32_t redirect_span_draw(ignis_seq_pool *pool, ignis_seq *seq, const ignis_sampling_params &sampling,
                           cudaStream_t stream, int32_t *out_redirected);

}  // namespace ignis::step
