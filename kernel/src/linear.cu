// ignis kernel leaf: the public `ops::linear` entry point (ADR 0009 program
// layer, on top of the ADR 0010 vendored ops).
//
// Ours, not vendored (kernel/vendor/VENDOR.md): the reference's
// `src/ops/linear/linear.cpp` dispatches across every weight qtype it
// registers (Q4/Q5/Q6/W8/BF16/NVFP4/FP8), so vendoring it verbatim would pull
// in every op family's dispatch header at once — including families this
// ticket does not vendor, which would break the build. Editing it to drop
// those branches would be a structural change, not the kind of local patch
// VENDOR.md's patch policy is for. So this file is leaf code that implements
// the reference's own public header, which *is* vendored byte-identical
// because it has no dependency on any op family
// (`kernel/vendor/include/ninfer/ops/linear.h`): same namespace, same
// `ops::linear` / `ops::LinearPolicy` / `ops::linear_workspace_capacity_bytes`
// signatures, so the vendored reference test (`test_nvfp4_a16.cpp` and its
// harness) links against this and never notices the split.
//
// P1-09 (GitHub #45) vendors NVFP4 (codec/format/config/dispatch/GEMV/
// small-T; the large-T W4A4/TMA sources compile but are verified only at G2)
// and wires it in below. P1-10 (GitHub #46) adds the BF16 and W8G32 arms.
// Q4/Q5/Q6/FP8 are in the reference's registry but never used by this
// model's artifact (`.scratch/runtime/specs/01-device-resident-forward.md`),
// so they stay unsupported here rather than being vendored for nothing.

#include "ninfer/ops/linear.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    if (w.qtype == QType::NVFP4) {
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    }
    throw std::invalid_argument(
        "linear: unsupported weight qtype (only NVFP4 is vendored here; P1-10/#46 adds BF16/W8G32)");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }
    if (qtype == QType::NVFP4) {
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    }
    throw std::invalid_argument(
        "linear workspace: unsupported weight qtype (only NVFP4 is vendored here; "
        "P1-10/#46 adds BF16/W8G32)");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

} // namespace ninfer::ops
