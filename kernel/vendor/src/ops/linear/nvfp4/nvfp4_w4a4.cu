#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

// ignis (ADR 0031, GitHub #652): whether the W4A4 TMA products launched on a stream read their
// activations K-tile-major, as the load running on it names (kernel/src/w4a4_activation_layout.h).
namespace ignis::w4a4_activation_layout {
bool k_tile_major(cudaStream_t stream);
} // namespace ignis::w4a4_activation_layout

namespace ninfer::ops::detail {
namespace {

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_kernel<ActivationGeometry>
        <<<(tasks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

// ignis (GitHub #652): nvfp4_w4a4_quantize_k_tile_kernel, as launch_quantize_exact launches the
// stock one.
template <class ActivationGeometry>
void launch_quantize_k_tile(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                            cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_k_tile_kernel<ActivationGeometry>
        <<<(tasks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                    Nvfp4W4a4ActivationLayout activation_layout, std::int32_t tokens,
                    cudaStream_t stream) {
    constexpr bool kResidualGeometry = std::is_same_v<Geometry, Nvfp4Residual6144Geometry> ||
                                       std::is_same_v<Geometry, Nvfp4Residual17408Geometry>;
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear(
            resolve_nvfp4_problem(Geometry::kOutputRows, Geometry::kInputRows), workspace.codes,
            workspace.scales, activation_layout, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Geometry, M32N128>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 128) {
        if constexpr (kResidualGeometry) {
            launch_gemm<Geometry, M32N128>(weight, out, workspace, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, out, workspace, tokens, stream);
        }
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 512) {
        if constexpr (Geometry::kOutputRows == Nvfp4GdnInputGeometry::kOutputRows) {
            launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, out, workspace, tokens, stream);
        }
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, stream);
    }
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

// ignis (GitHub #652): the TMA route's widths K-tile-major where the load names it, through the
// same rule as the launchers' (a whole number of kTmaBlockM tiles from 1,024 tokens).
Nvfp4W4a4ActivationLayout launch_nvfp4_w4a4_quantize_for_products(const Tensor& x,
                                                                  const Weight& weight,
                                                                  Nvfp4W4a4Workspace workspace,
                                                                  cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens < 1024 || (tokens % kTmaBlockM) != 0 ||
        !ignis::w4a4_activation_layout::k_tile_major(stream)) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
        return Nvfp4W4a4ActivationLayout::TokenMajor;
    }
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_k_tile<Nvfp4Activation5120Geometry>(x, weight, workspace, stream);
        break;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_k_tile<Nvfp4Activation6144Geometry>(x, weight, workspace, stream);
        break;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_k_tile<Nvfp4Activation17408Geometry>(x, weight, workspace, stream);
        break;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
    return Nvfp4W4a4ActivationLayout::KTileMajor;
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const Nvfp4W4a4ActivationLayout layout =
        launch_nvfp4_w4a4_quantize_for_products(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launch_problem<Nvfp4AttnInputGeometry>(weight, out, workspace, layout, tokens, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_problem<Nvfp4GdnInputGeometry>(weight, out, workspace, layout, tokens, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_problem<Nvfp4MlpGateUpGeometry>(weight, out, workspace, layout, tokens, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Residual6144Geometry>(weight, out, workspace, layout, tokens, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Residual17408Geometry>(weight, out, workspace, layout, tokens, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
