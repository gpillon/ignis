// ignis kernel leaf: the public `ops::attn_input_proj` entry points (ADR 0009
// program layer, on top of the ADR 0010 vendored ops).
//
// Ours, not vendored (kernel/vendor/VENDOR.md): the reference's
// `src/ops/wrapper/attn_input_proj.cpp` dispatches across every registered
// parent weight qtype (BF16/NVFP4/FP8/W8) plus a separate Q4/Q5 dual-weight
// overload and a W8 companion overload, so vendoring it verbatim would pull
// in op families this ticket does not vendor (FP8, W8, Q4/Q5), breaking the
// build. So this file is leaf code that implements the reference's own public
// header, which *is* vendored byte-identical because it has no dependency on
// any op family (`kernel/vendor/include/ninfer/ops/attn_input_proj.h`): same
// namespace, same `ops::attn_input_proj` / `ops::attn_input_proj_workspace_capacity_bytes`
// signatures. Only the single-parent overloads (BF16_CTRL and NVFP4 target
// weights, the model's own fused q/k/gate/v projection) are implemented; the
// Q4/Q5 dual-weight overload and the W8 single/companion overloads are never
// defined, matching that no vendored code here calls them.
//
// P1-11 (GitHub #47) vendors the NVFP4 and BF16 arms (their dispatch headers
// need no workspace beyond NVFP4's private W4A4 route).

#include "ninfer/ops/attn_input_proj.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16_CTRL) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("BF16 attn_input_proj admits only A16");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 attn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    throw std::invalid_argument(
        "attn_input_proj: unsupported parent weight qtype (only NVFP4/BF16 are vendored here)");
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16_CTRL:
        if (parent_rows != 14336 || input_rows != 5120 || policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (parent_rows != detail::Nvfp4AttnInputGeometry::kOutputRows ||
            input_rows != detail::Nvfp4AttnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    default:
        break;
    }
    throw std::invalid_argument(
        "attn_input_proj workspace: unsupported parent qtype (only NVFP4/BF16 are vendored here)");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

} // namespace ninfer::ops
