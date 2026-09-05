// The top-level `ops::linear` dispatch across weight qtypes (GitHub #46,
// P1-10). Ours, not vendored (ADR 0010): it composes the vendored per-family
// dispatchers (`ops::detail::bf16_dispatch`, `ops::detail::w8_dispatch`)
// added by their own tickets. Extend the switch when a ticket vendors
// another family (NVFP4: P1-09/#45).

#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    if (x.ne[2] != 1 || x.ne[3] != 1) { throw std::invalid_argument("linear: x must have shape [K,T]"); }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) { throw std::invalid_argument("linear: weight n/k must be positive"); }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     cudaStream_t stream) {
    switch (w.qtype) {
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::NVFP4:
    case QType::FP8_E4M3FN_ROW_BF16S:
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }
    switch (qtype) {
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::NVFP4:
    case QType::FP8_E4M3FN_ROW_BF16S:
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& /*workspace*/, cudaStream_t stream) {
    validate_linear_semantics(x, w, out);
    dispatch_linear(x, w, out, policy, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, stream);
}

} // namespace ninfer::ops
