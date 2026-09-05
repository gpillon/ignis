// ignis kernel leaf: the public `ops::linear_add` entry points (ADR 0009
// program layer, on top of the ADR 0010 vendored ops).
//
// Ours, not vendored (kernel/vendor/VENDOR.md): the reference's
// `src/ops/wrapper/linear_add.cpp` dispatches across every registered weight
// qtype (BF16/Q5/W8/NVFP4/FP8), so vendoring it verbatim would pull in op
// families this ticket does not vendor (Q5, W8, FP8), breaking the build. So
// this file is leaf code that implements the reference's own public header,
// which *is* vendored byte-identical because it has no dependency on any op
// family (`kernel/vendor/include/ninfer/ops/linear_add.h`): same namespace,
// same `ops::linear_add` / `ops::linear_add_workspace_capacity_bytes`
// signatures.
//
// P1-11 (GitHub #47) vendors the NVFP4 and BF16 arms (the attention output
// projection's residual add, [5120,6144] and [5120,17408]).

#include "ninfer/ops/linear_add.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t columns,
                    const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != columns || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("linear_add: invalid ") + name);
    }
}

void require_bf16(const Weight& w) {
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous || w.qdata == nullptr) {
        throw std::invalid_argument("linear_add: weight must be contiguous BF16_CTRL");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_add: invalid compute policy");
}

} // namespace

std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                std::int32_t input_rows, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    return linear_add_workspace_capacity_bytes(qtype, output_rows, input_rows,
                                               LinearPolicy::A16Only, min_tokens, max_tokens);
}

std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                std::int32_t input_rows, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear_add workspace: invalid token interval");
    }
    if (qtype == QType::BF16_CTRL) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_add workspace: BF16 admits only A16");
        }
        (void)detail::bf16_linear_add_select(output_rows, input_rows, min_tokens);
        (void)detail::bf16_linear_add_select(output_rows, input_rows, max_tokens);
        return 0;
    }
    if (qtype == QType::NVFP4) {
        const bool supported = (output_rows == detail::Nvfp4Residual6144Geometry::kOutputRows &&
                                input_rows == detail::Nvfp4Residual6144Geometry::kInputRows) ||
                               (output_rows == detail::Nvfp4Residual17408Geometry::kOutputRows &&
                                input_rows == detail::Nvfp4Residual17408Geometry::kInputRows);
        if (!supported || (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear_add workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_add_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                                 min_tokens, max_tokens);
    }
    throw std::invalid_argument(
        "linear_add workspace: unsupported weight format (only NVFP4/BF16 are vendored here)");
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, WorkspaceArena& ws,
                cudaStream_t stream) {
    linear_add(x, w, residual_out, LinearPolicy::A16Only, ws, stream);
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t t = x.ne[1];
    if (t <= 0) { throw std::invalid_argument("linear_add: T must be positive"); }
    require_tensor(x, DType::BF16, w.k, t, "x");
    require_tensor(residual_out, DType::BF16, w.n, t, "residual_out");
    if (overlaps(x, residual_out)) {
        throw std::invalid_argument("linear_add: x and residual_out must not overlap");
    }

    if (w.qtype == QType::BF16_CTRL) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("BF16 linear_add admits only A16");
        }
        require_bf16(w);
        if (!detail::bf16_linear_add_admits(w.n, w.k, t)) {
            throw std::invalid_argument("linear_add: unsupported BF16 shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16)) {
            throw std::invalid_argument(
                "linear_add: BF16 requires 16-byte x/residual/weight alignment");
        }
        (void)ws;
        detail::bf16_linear_add_dispatch(x, w, residual_out, stream);
        return;
    }

    if (w.qtype == QType::NVFP4) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 linear_add admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(w, "nvfp4 linear_add");
        const bool supported_shape = (w.n == detail::Nvfp4Residual6144Geometry::kOutputRows &&
                                      w.k == detail::Nvfp4Residual6144Geometry::kInputRows) ||
                                     (w.n == detail::Nvfp4Residual17408Geometry::kOutputRows &&
                                      w.k == detail::Nvfp4Residual17408Geometry::kInputRows);
        if (!supported_shape) {
            throw std::invalid_argument("nvfp4 linear_add: unsupported weight shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16)) {
            throw std::invalid_argument("linear_add: NVFP4 requires 16-byte x/residual alignment");
        }
        detail::nvfp4_linear_add_dispatch(x, w, residual_out, policy, ws, stream);
        return;
    }

    throw std::invalid_argument(
        "linear_add: unsupported weight format (only NVFP4/BF16 are vendored here)");
}

} // namespace ninfer::ops
