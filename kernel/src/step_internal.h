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

#include <cstdint>
#include <string>

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

}  // namespace ignis::step
