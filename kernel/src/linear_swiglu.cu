// ignis kernel leaf: the public `ops::linear_swiglu` entry points (ADR 0009
// program layer, on top of the ADR 0010 vendored ops).
//
// Ours, not vendored (kernel/vendor/VENDOR.md): the reference's
// `src/ops/wrapper/linear_swiglu.cpp` dispatches across every registered
// weight qtype (Q4/W8/NVFP4/FP8), so vendoring it verbatim would pull in op
// families this ticket does not vendor (Q4, W8, FP8), breaking the build. So
// this file is leaf code that implements the reference's own public header,
// which *is* vendored byte-identical because it has no dependency on any op
// family (`kernel/vendor/include/ninfer/ops/linear_swiglu.h`): same
// namespace, same `ops::linear_swiglu` /
// `ops::linear_swiglu_workspace_capacity_bytes` signatures.
//
// P1-13 (GitHub #49) vendors only the NVFP4 arm (the fused MLP gate_up
// projection with the SiLU-mul epilogue, [34816,5120] -> [17408,T]; gate rows
// [0,17408) precede their matching up rows [17408,34816)). The large-T W4A4
// route (`Nvfp4LinearSwiGluRoute::LinearW4A4Post`, inside the vendored plan)
// falls back to the already-vendored `ops::linear` + `ops::silu_mul`, so it
// compiles here but its own reference test is deferred to G2, same as P1-09.

#include "ninfer/ops/linear_swiglu.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                    std::int32_t input_rows,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (qtype == QType::NVFP4) {
        if (gate_up_rows != detail::Nvfp4MlpGateUpGeometry::kOutputRows ||
            input_rows != detail::Nvfp4MlpGateUpGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear_swiglu workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument(
        "linear_swiglu workspace: unsupported weight format (only NVFP4 is vendored here)");
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = x.ne[0] == detail::Nvfp4MlpGateUpGeometry::kInputRows &&
                             out.ne[0] == detail::Nvfp4MlpGateUpGeometry::kOutputRows / 2 &&
                             gate_up_weight.n == detail::Nvfp4MlpGateUpGeometry::kOutputRows &&
                             gate_up_weight.k == detail::Nvfp4MlpGateUpGeometry::kInputRows &&
                             gate_up_weight.padded_shape[0] ==
                                 detail::Nvfp4MlpGateUpGeometry::kOutputRows &&
                             gate_up_weight.padded_shape[1] ==
                                 detail::Nvfp4MlpGateUpGeometry::kInputRows;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || !large_shape) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }
    if (gate_up_weight.qtype != QType::NVFP4) {
        throw std::invalid_argument(
            "linear_swiglu: unsupported weight format (only NVFP4 is vendored here)");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("NVFP4 linear_swiglu admits only A16 or A4");
    }
    detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
    detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

} // namespace ninfer::ops
