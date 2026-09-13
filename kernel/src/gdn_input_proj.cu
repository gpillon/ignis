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
// `gdn_input_proj_conv_snapshot` entry point is likewise not implemented here
// — only its underlying vendored NVFP4 kernels are wired into the build so
// they compile now.
//
// P5-04 (GitHub #153) implements the single-parent NVFP4
// `gdn_input_proj_conv_record` form and its capacity query, the
// reference's own dispatch (`src/ops/wrapper/gdn_input_proj.cpp`,
// `dispatch_single_parent_record`) restated for the one vendored arm: the
// verify round's GDN layers project, convolve from a lane's conv taps
// without touching them, and write the represented conv input to the
// ReplaySSM conv record the fold later consumes. B=1 takes the vendored
// fused small-T A16 route where the plan registers it and the materialized
// projection + `nvfp4_gdn_record_post_launch` otherwise; B=2..8 always
// materializes the projection straight into the conv record and runs the
// shared `gdn_projected_conv_record_launch` over it -- the reference's own
// schedule at every point, none of it ours.
//
// P1-12 (GitHub #48) vendors the NVFP4 arm (qkv+z fused query/key/value/z
// projection, [16384,5120]). The artifact's `qwen3.8-27b-artifact.md` §14.1
// mixed-quant exception pattern gives every GDN layer's `gdn/query_key_value_z`
// as NVFP4 with no per-layer exception (unlike `gdn/output`, whose layer-4
// BF16 exception is already covered by the reference's registered BF16
// `linear_add` arm, kernel/src/linear_add.cu, P1-11/#47) — so gdn_input_proj
// needs no BF16 arm at all, matching that the reference registers none for it.

#include "ninfer/ops/gdn_input_proj.h"

#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <algorithm>
#include <array>
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

// --- P5-04 (GitHub #153): the ReplaySSM record-producing form ---------------
//
// The reference's validation for the NVFP4 single-parent arm, verbatim in
// substance: the [C, T, B] operand shapes, the B=1..8 / T=2..16 record
// domain, the read-only conv-state view, the per-row selectors, and the
// pairwise non-overlap of every operand with each other and with the live
// workspace (the caller hands a workspace *span* of its own, not the arena
// its activations live in, exactly so this check can hold).

constexpr std::int32_t kQueryRows  = 2048;
constexpr std::int32_t kKeyRows    = 2048;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows; // == kQkvRows
constexpr std::int32_t kMaximumRecordBatch = 8;
constexpr std::int32_t kMinimumRecordWidth = 2;
constexpr std::int32_t kMaximumRecordWidth = 16;

struct ConvGeometry {
    std::int32_t width;
    std::int32_t batch;
    std::int32_t aggregate_columns;
};

void require_conv_tensor(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                         std::int32_t batch, const char* op, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != width ||
        tensor.ne[2] != batch || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

ConvGeometry require_record_input(const Tensor& x) {
    const std::int32_t width = x.ne[1];
    const std::int32_t batch = x.ne[2];
    if (width < kMinimumRecordWidth || width > kMaximumRecordWidth || batch <= 0 ||
        batch > kMaximumRecordBatch) {
        throw std::invalid_argument("gdn_input_proj_conv_record: unsupported B/T domain");
    }
    require_conv_tensor(x, kHidden, width, batch, "gdn_input_proj_conv_record", "x");
    return {width, batch, width * batch};
}

void require_record_operands(const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             ConvGeometry geometry) {
    require_matrix(conv_weight, kChannels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != kChannels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] <= 0 || conv_states.ne[3] != 1 ||
        !conv_states.is_contiguous() || !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid convolution state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid initial state selector");
    }
    if (valid_columns.data != nullptr && !valid_selector(valid_columns)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid valid columns");
    }
}

bool overlaps_range(const Tensor& tensor, const void* base, std::size_t bytes) {
    if (tensor.data == nullptr || base == nullptr || bytes == 0) { return false; }
    const auto tensor_begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto range_begin  = reinterpret_cast<std::uintptr_t>(base);
    return tensor_begin < range_begin + bytes && range_begin < tensor_begin + tensor.bytes();
}

void require_record_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& conv_record,
                               const Tensor& query, const Tensor& key, const Tensor& value,
                               const Tensor& z, const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{
        &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
        &conv_record, &query,       &key,         &value,         &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            if (tensors[rhs]->data != nullptr && overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_record: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_record: tensor operand overlaps live workspace");
        }
    }
}

void require_record_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                    std::int32_t max_width) {
    if (batch_size <= 0 || batch_size > kMaximumRecordBatch || min_width < kMinimumRecordWidth ||
        max_width < min_width || max_width > kMaximumRecordWidth) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: invalid B/T domain");
    }
}

Tensor flatten_columns(const Tensor& tensor, std::int32_t rows, ConvGeometry geometry) {
    return Tensor(tensor.data, tensor.dtype, {rows, geometry.aggregate_columns});
}

void dispatch_single_parent_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& conv_record,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   LinearPolicy policy, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype != QType::NVFP4) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_record: unsupported parent weight qtype (only NVFP4 is vendored here)");
    }
    const ConvGeometry geometry = require_record_input(x);
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("NVFP4 gdn_input_proj_conv_record admits only A16 or A4");
    }
    detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_record");
    if (weight.n != kRows || weight.k != kHidden) {
        throw std::invalid_argument("nvfp4 gdn_input_proj_conv_record: unsupported weight shape");
    }
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    const detail::Nvfp4GdnConvPlan plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
    if (plan.schedule == detail::Nvfp4GdnConvScheduleId::Materialized && geometry.batch > 1) {
        // The batched schedule: the projection lands in the conv record
        // itself (one flat [C, B*T] matrix), then the shared record kernel
        // convolves each row from its own initial window.
        auto scope         = workspace.scope();
        Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
        Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
        Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
        dispatch_single_parent(x_flat, weight, record_flat, z_flat, policy, &workspace, stream);
        detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states,
                                                 valid_columns, initial_state_slots, query, key,
                                                 value, stream);
        return;
    }
    if (plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) {
        detail::nvfp4_gdn_record_small_t_launch(x, weight, conv_weight, conv_states, valid_columns,
                                                initial_state_slots, conv_record, query, key, value,
                                                z, stream);
        return;
    }

    auto scope = workspace.scope();
    dispatch_single_parent(x, weight, conv_record, z, policy, &workspace, stream);
    detail::nvfp4_gdn_record_post_launch(conv_record, conv_weight, conv_states, valid_columns,
                                         initial_state_slots, query, key, value, stream);
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

// P5-04 (GitHub #153): the single-parent NVFP4 record-producing profile's
// capacity (the reference's own arithmetic for this arm): the batched
// schedule materializes the projection at `B*T` aggregate columns under the
// call's policy; B=1 needs storage only where its plan materializes, and
// then only the A4 projection's, at widths from 4 up.
std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype != QType::NVFP4 || parent_rows != kRows || input_rows != kHidden ||
        (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_record workspace: unsupported single-parent profile");
    }
    const detail::Nvfp4GdnConvPlan minimum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    const detail::Nvfp4GdnConvPlan maximum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);
    if (batch_size == 1) {
        if (minimum_plan.schedule == detail::Nvfp4GdnConvScheduleId::DecodeFusedA16) {
            throw std::logic_error("ReplaySSM record planner admitted NVFP4 decode");
        }
        if (maximum_plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) { return 0; }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy::AllowA4,
                                                                std::max(min_width, 4), max_width);
    }
    return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, batch_size * min_width,
                                                            batch_size * max_width);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, policy, workspace, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, LinearPolicy::A16Only, workspace, stream);
}

} // namespace ninfer::ops
