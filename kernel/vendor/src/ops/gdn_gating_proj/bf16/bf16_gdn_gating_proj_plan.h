#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16GdnGatingScheduleId {
    GemvPairedRows,
    SmallTSplit10,
    SimtWarpRowC4,
    SimtWarpRowC8,
    MmaCooperativeSplit40,
    MmaCooperativeSplit32,
    MmaCooperativeSplit16,
    MmaCooperativeSplit8,
    MmaCooperativeSplit4,
    MmaCooperativeSplit2,
    MmaUnsplit,
};

struct Bf16GdnGatingProblem {
    std::int32_t heads;
    std::int32_t input_rows;
    std::int32_t cols;
};

struct Bf16GdnGatingPlan {
    Bf16GdnGatingScheduleId schedule;
    Bf16GdnGatingTokenVariant token_variant;
    std::size_t workspace_bytes;
};

enum class Bf16GdnNormGatingScheduleId {
    Composed,
    MmaCooperativeSplit40,
    MmaCooperativeSplit32,
};

struct Bf16GdnNormGatingPlan {
    Bf16GdnNormGatingScheduleId schedule;
    Bf16GdnGatingPlan control;
    std::size_t workspace_bytes;
};

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept;

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept;
// Ignis portability patch (ADR 0077, GitHub #602): the schedule the route table names for an
// admitted problem, before the card's residency test. bf16_gdn_gating_resolve_plan runs it where
// the card holds its grid resident; where it does not, the plan is a fallback.
Bf16GdnGatingScheduleId bf16_gdn_gating_route_schedule(const Bf16GdnGatingProblem& problem);
Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem);
Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem);

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_cols, std::int32_t max_cols);
const char* bf16_gdn_norm_gating_schedule_name(Bf16GdnNormGatingScheduleId schedule) noexcept;
Bf16GdnNormGatingPlan bf16_gdn_norm_gating_resolve_plan(const Bf16GdnGatingProblem& problem);
std::size_t bf16_gdn_norm_gating_capacity_workspace_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_cols,
                                                          std::int32_t max_cols);

// Ignis portability patch (ADR 0077, GitHub #602), for tests only: caps the SM count the
// residency test multiplies each cooperative kernel's CTAs/SM by, so the gating test runs a
// smaller card's routes on the RTX 5090. The CTAs/SM still come from the occupancy query on the
// card. 0, or a cap at or above the card's SM count, changes nothing. It applies from the next
// plan on; nothing in the engine or the wrapper calls it.
void bf16_gdn_gating_set_test_multiprocessor_cap(int multiprocessors) noexcept;

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, cudaStream_t stream);
void bf16_gdn_gating_execute_candidate(Bf16GdnGatingScheduleId schedule, const Tensor& x,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias,
                                       WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                       cudaStream_t stream);
void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, cudaStream_t stream);
void bf16_gdn_norm_gating_dispatch(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                   const Weight& a_weight, const Weight& b_weight,
                                   const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                   Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops::detail
