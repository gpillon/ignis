#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

enum class Bf16GdnGatingTokenVariant {
    None,
    Full,
    Predicated,
};

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream);
void bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream);

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream);
void bf16_gdn_norm_gating_proj_27_mma_split40_launch(Bf16GdnGatingTokenVariant variant,
                                                     const Tensor& x, const Tensor& norm_weight,
                                                     float eps, Tensor& h, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, void* workspace,
                                                     Tensor& g, Tensor& beta, cudaStream_t stream);
void bf16_gdn_norm_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant,
                                                     const Tensor& x, const Tensor& norm_weight,
                                                     float eps, Tensor& h, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, void* workspace,
                                                     Tensor& g, Tensor& beta, cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split16_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream);

// Ignis portability patch (ADR 0077, GitHub #478): the cooperative MMA kernels the planner keeps
// resident, one per launched specialization family.
enum class Bf16GdnGatingCooperativeKernel {
    Mma27Split8,
    Mma27Split4,
    Mma27Split2,
    Norm27Split40,
    Mma35Split32,
    Norm35Split32,
    Mma35Split16,
    Mma35Split8,
    Mma35Split4,
    Mma35Split2,
};

// CTAs of `kernel` one SM of the current device holds at once: the occupancy query at the block
// size and dynamic shared memory its launch uses, the minimum over every token variant (and fused
// token capacity) a launch can pick.
int bf16_gdn_gating_cooperative_ctas_per_sm(Bf16GdnGatingCooperativeKernel kernel);

} // namespace ninfer::ops::detail
