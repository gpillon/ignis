// ignis kernel leaf: the public `ops::gdn_input_proj` entry points (ADR 0009
// program layer, on top of the ADR 0010 vendored ops).
//
// Ours, not vendored (kernel/vendor/VENDOR.md): the reference's
// `src/ops/wrapper/gdn_input_proj.cpp` dispatches across every registered
// parent weight qtype (NVFP4/FP8/W8) plus a separate Q4/Q5 dual-weight
// overload, so vendoring it verbatim would pull in op families this ticket
// does not vendor (FP8, W8, Q4/Q5), breaking the build. So this file is leaf
// code that implements the reference's own public header, which *is*
// vendored byte-identical because it has no dependency on any op family
// (`kernel/vendor/include/ninfer/ops/gdn_input_proj.h`): same namespace, same
// `ops::gdn_input_proj` / `ops::gdn_input_proj_workspace_capacity_bytes`
// signatures. Only the single-parent NVFP4 overloads are implemented; the
// Q4/Q5 dual-weight overload and the FP8/W8 single-parent branches are never
// defined, matching that no vendored code here calls them. The
// `gdn_input_proj_conv_snapshot` / `gdn_input_proj_conv_record` entry points
// are likewise not implemented here (G3/G5 work) — only their underlying
// vendored NVFP4 kernels are wired into the build so they compile now.
//
// P1-12 (GitHub #48) vendors the NVFP4 arm (qkv+z fused query/key/value/z
// projection, [16384,5120]). The artifact's `qwen3.8-27b-artifact.md` §14.1
// mixed-quant exception pattern gives every GDN layer's `gdn/query_key_value_z`
// as NVFP4 with no per-layer exception (unlike `gdn/output`, whose layer-4
// BF16 exception is already covered by the reference's registered BF16
// `linear_add` arm, kernel/src/linear_add.cu, P1-11/#47) — so gdn_input_proj
// needs no BF16 arm at all, matching that the reference registers none for it.

#include "ninfer/ops/gdn_input_proj.h"

#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden  = 5120;
constexpr std::int32_t kQkvRows = 10240;
constexpr std::int32_t kZRows   = 6144;
constexpr std::int32_t kRows    = kQkvRows + kZRows;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_single_parent_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);

    if (weight.qtype == QType::NVFP4) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 gdn_input_proj admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    throw std::invalid_argument(
        "gdn_input_proj: unsupported parent weight qtype (only NVFP4 is vendored here)");
}

} // namespace

std::size_t gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (parent_qtype == QType::NVFP4) {
        if (parent_rows != kRows || input_rows != kHidden ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument(
        "gdn_input_proj workspace: unsupported parent qtype (only NVFP4 is vendored here)");
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, policy, &workspace, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only, nullptr,
                           stream);
}

} // namespace ninfer::ops
