#include "ninfer/ops/gdn_gating_proj.h"

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Geometry {
    const char* label;
    std::int32_t hidden;
    std::int32_t heads;
    bool parent_weight;
};

constexpr Geometry kQwen27{"qwen3_6_27b", 5120, 48, false};
constexpr Geometry kQwen38Parent{"qwen3_8_27b_parent", 5120, 48, true};
constexpr Geometry kQwen35{"qwen3_6_35b_a3b", 2048, 32, true};

constexpr ReductionCriterion kGdnProjectionFp32{/*relative_l2=*/1.4e-6,
                                                /*gross_absolute=*/5.0e-7,
                                                /*gross_relative_to_max_reference=*/2.5e-6};
constexpr ReductionCriterion kGdnNormOutputBf16{/*relative_l2=*/1.75e-3,
                                                /*gross_absolute=*/1.0e-4,
                                                /*gross_relative_to_max_reference=*/4.0e-3};
constexpr ReductionCriterion kGdnNormControlFp32{/*relative_l2=*/8.0e-4,
                                                 /*gross_absolute=*/1.5e-4,
                                                 /*gross_relative_to_max_reference=*/1.05e-3};

// Ignis portability patch (ADR 0077, GitHub #478, #602): the cooperative schedules' grids are
// sized for the RTX 5090's 170 SMs. Where a card cannot hold a route's own schedule resident
// (fewer SMs, or fewer CTAs/SM in the build it runs), the planner runs a fallback: the largest
// smaller split of the model's family the card holds, the unsplit route when none fits, and for
// the 27B's fused norm/control route at T=1..8 (cooperative split40) the composed route. The test
// runs its whole case list at the card's own SM count, then under each test-only cap below it
// (bf16_gdn_gating_set_test_multiprocessor_cap): 84, the RTX 5080's, and 36, the RTX 5060 Ti's.
// The cap changes only the SM count the residency test multiplies by, so the RTX 5090 runs a
// smaller card's routes end to end: the planner's choice, the capacity, the launch and the
// workspace high-water against the query. Every line of a pass names its SM count. A case whose
// plan differs from the route's own schedule is a fallback, named on a FALLBACK line; on the
// RTX 5090 with no cap none is, and main checks that.
struct Pass {
    std::string label;
    int fallbacks;
};

// A case or a check that throws fails by name and the run goes on, so against a planner that
// cannot plan a case every such case is named.
template <typename Check>
int guarded(const std::string& label, const Check& check) {
    try {
        return check();
    } catch (const std::exception& error) {
        std::cerr << label << ": threw: " << error.what() << "\n";
        return 1;
    }
}

// The unsplit route, where the planner runs it in place of a cooperative route the card cannot
// hold resident, is checked against a criterion of its own. kGdnProjectionFp32 was tuned on the
// RTX 5090, where every T below 4097 runs split-K, and holds the unsplit route with no margin: on
// an sm_86 card on which the cooperative schedules are not resident, as on an RTX 3090, T=1024
// and T=2049 failed it.
//
// The unsplit route keeps one fp32 accumulator per output through 320 m16n8k16 MMAs in k order
// (K = 5120) and sums no partials. The tensor core normalizes each MMA's sum toward zero (the RTX
// 5090's unsplit outputs match a CPU model of 16 exact products and the accumulator, aligned to
// the largest exponent and truncated toward zero: 575 of 576 g values within 2.5e-7 relative),
// so step i loses a fraction below 2^-23 of the running sum s_i, always toward zero. The losses
// add as a bias, -sum(rho_i * s_i), not as a random walk: over n steps its rms grows as n^(3/2)
// against the sum's n^(1/2), a relative error that grows as n. A route of S splits runs S chains
// of n/S steps and carries 1/S of the unsplit route's error. On the RTX 5090, on this test's data,
// g's rel_l2 is 1.4e-7 on split8 (T=1024), 2.6e-7 on split4 (T=1025) and 5.8e-7 on split2
// (T=2049); the unsplit route forced at the same T gives 1.33e-6, 1.12e-6 and 1.30e-6, and
// 1.24e-6 at T=4097, where the planner runs it. At T=1024 that is 0.95 of kGdnProjectionFp32's
// rel_l2 and 1.12 of its gross limit. Over 300 seeds the CPU model gives a mean rel_l2 of 1.2e-6,
// at most 1.85e-6 and 1.6 times that gross limit; models of the same truncation that fit the sm_86
// card's failures reach 2.3e-6 and 2.2 times.
//
// This criterion is kGdnProjectionFp32 times 3 in each term: the unsplit chain carries twice the
// error of split2, the longest chain kGdnProjectionFp32 holds with margin (0.41 of it), and the
// spread of the bias across data takes the rest. Those models stay within 0.73 of it over 300
// seeds, the RTX 5090's unsplit route within 0.37 on this data; a wrong tile or K range errs by
// ~1e-2. The 35B's fallback (K = 2048, 128 MMAs, at most 3.3e-7 on the RTX 5090) is held to it. On
// the RTX 5090 no case is a fallback, so every check there is the reference's. A fallback to a
// smaller split (GitHub #602) runs chains no longer than split2's, which kGdnProjectionFp32 holds
// with margin, and is held to kGdnProjectionFp32.
constexpr ReductionCriterion kGdnProjectionUnsplitFp32{/*relative_l2=*/4.2e-6,
                                                       /*gross_absolute=*/1.5e-6,
                                                       /*gross_relative_to_max_reference=*/7.5e-6};

// Names a projection case's fallback: a plan other than the schedule the route table names
// (bf16_gdn_gating_route_schedule). Returns the criterion the schedule it runs is held to.
const ReductionCriterion& name_projection_fallback(Pass& pass, const Geometry& geometry,
                                                   std::int32_t tokens) {
    const ops::detail::Bf16GdnGatingProblem problem{geometry.heads, geometry.hidden, tokens};
    const ops::detail::Bf16GdnGatingScheduleId routed =
        ops::detail::bf16_gdn_gating_route_schedule(problem);
    const ops::detail::Bf16GdnGatingPlan plan = ops::detail::bf16_gdn_gating_resolve_plan(problem);
    if (plan.schedule == routed) { return kGdnProjectionFp32; }
    ++pass.fallbacks;
    std::cout << "FALLBACK " << pass.label << " gdn_gating_proj " << geometry.label
              << " T=" << tokens << ": the route's "
              << ops::detail::bf16_gdn_gating_schedule_name(routed)
              << " is not resident, the planner runs "
              << ops::detail::bf16_gdn_gating_schedule_name(plan.schedule) << "\n";
    return plan.schedule == ops::detail::Bf16GdnGatingScheduleId::MmaUnsplit
               ? kGdnProjectionUnsplitFp32
               : kGdnProjectionFp32;
}

// The fused route the norm/control op takes where the card holds it resident: split40 for the
// 27B at T=1..8, split32 for the 35B at T=1..16, the composed route elsewhere.
ops::detail::Bf16GdnNormGatingScheduleId fused_norm_route(const Geometry& geometry,
                                                          std::int32_t tokens) {
    if (geometry.heads == 48 && geometry.hidden == 5120 && tokens <= 8) {
        return ops::detail::Bf16GdnNormGatingScheduleId::MmaCooperativeSplit40;
    }
    if (geometry.heads == 32 && geometry.hidden == 2048 && tokens <= 16) {
        return ops::detail::Bf16GdnNormGatingScheduleId::MmaCooperativeSplit32;
    }
    return ops::detail::Bf16GdnNormGatingScheduleId::Composed;
}

// Names a norm/control case's fallback: a plan other than its fused route. A fallback runs the
// composed route, held to the norm cases' criteria as the composed cases are.
void name_norm_fallback(Pass& pass, const Geometry& geometry, std::int32_t tokens) {
    const ops::detail::Bf16GdnNormGatingScheduleId routed = fused_norm_route(geometry, tokens);
    const ops::detail::Bf16GdnNormGatingPlan plan =
        ops::detail::bf16_gdn_norm_gating_resolve_plan({geometry.heads, geometry.hidden, tokens});
    if (plan.schedule == routed) { return; }
    ++pass.fallbacks;
    std::cout << "FALLBACK " << pass.label << " gdn_norm_gating_proj " << geometry.label
              << " T=" << tokens << ": the route's "
              << ops::detail::bf16_gdn_norm_gating_schedule_name(routed)
              << " is not resident, the planner runs "
              << ops::detail::bf16_gdn_norm_gating_schedule_name(plan.schedule) << "\n";
}

// The planner takes each cooperative kernel's CTAs/SM from the occupancy query. On a compute
// capability 12.0 device the query must give the budget the planner hardcoded for the registered
// sm_120a build, so its routes there are the reference's; split40's 2 was counted against its 80
// workspace splits, the same grid limit as 1 against the 40 it launches. Elsewhere the lines only
// report the build's values.
struct CooperativeBudget {
    ops::detail::Bf16GdnGatingCooperativeKernel kernel;
    const char* name;
    int sm120a_ctas_per_sm;
};

constexpr CooperativeBudget kCooperativeBudgets[] = {
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma27Split8, "27b split8", 2},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma27Split4, "27b split4", 2},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma27Split2, "27b split2", 2},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Norm27Split40, "27b norm split40", 1},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma35Split32, "35b split32", 2},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Norm35Split32, "35b norm split32", 2},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma35Split16, "35b split16", 4},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma35Split8, "35b split8", 4},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma35Split4, "35b split4", 4},
    {ops::detail::Bf16GdnGatingCooperativeKernel::Mma35Split2, "35b split2", 4},
};

constexpr int kRtx5090Multiprocessors = 170;

struct DeviceShape {
    int major;
    int minor;
    int multiprocessors;
};

DeviceShape current_device_shape() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    DeviceShape shape{};
    cuda_check(cudaDeviceGetAttribute(&shape.major, cudaDevAttrComputeCapabilityMajor, device),
               "cudaDeviceGetAttribute(major)");
    cuda_check(cudaDeviceGetAttribute(&shape.minor, cudaDevAttrComputeCapabilityMinor, device),
               "cudaDeviceGetAttribute(minor)");
    cuda_check(cudaDeviceGetAttribute(&shape.multiprocessors, cudaDevAttrMultiProcessorCount,
                                      device),
               "cudaDeviceGetAttribute(multiprocessors)");
    return shape;
}

bool is_sm120(const DeviceShape& shape) { return shape.major == 12 && shape.minor == 0; }

int verify_cooperative_budgets(const DeviceShape& shape) {
    int failures = 0;
    for (const CooperativeBudget& budget : kCooperativeBudgets) {
        const int ctas = ops::detail::bf16_gdn_gating_cooperative_ctas_per_sm(budget.kernel);
        std::cout << "budget gdn_gating_proj " << budget.name << ": " << ctas
                  << " CTAs/SM (sm_120a " << budget.sm120a_ctas_per_sm << ")\n";
        if (is_sm120(shape) && ctas != budget.sm120a_ctas_per_sm) {
            std::cerr << "gdn_gating_proj " << budget.name << ": " << ctas
                      << " CTAs/SM on compute capability 12.0, the planner was tuned on "
                      << budget.sm120a_ctas_per_sm << "\n";
            ++failures;
        }
    }
    return failures;
}

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double e = std::exp(value);
    return e / (1.0 + e);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> read_fp32(const void* device, std::size_t elements) {
    const std::vector<float> values = from_device<float>(device, elements);
    return {values.begin(), values.end()};
}

int verify_normwise(const std::string& label, const std::vector<double>& actual,
                    const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, expected, criterion);
}

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

std::vector<std::int32_t> oracle_tokens(std::int32_t tokens) {
    std::vector<std::int32_t> selected;
    if (tokens <= 128) {
        selected.reserve(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) { selected.push_back(token); }
    } else {
        selected = {0, tokens / 2, tokens - 1};
    }
    return selected;
}

void projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                       const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                       const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                       const std::vector<std::int32_t>& selected_tokens, std::vector<double>& g,
                       std::vector<double>& beta) {
    const std::size_t output_elements =
        static_cast<std::size_t>(geometry.heads) * selected_tokens.size();
    g.resize(output_elements);
    beta.resize(output_elements);

    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t x_base =
            static_cast<std::size_t>(selected_tokens[sample]) * geometry.hidden;
        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = sample * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

void norm_projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                            const std::vector<float>& norm_weight,
                            const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                            const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                            std::int32_t tokens, double eps, std::vector<double>& h,
                            std::vector<double>& g, std::vector<double>& beta) {
    h.resize(static_cast<std::size_t>(geometry.hidden) * tokens);
    g.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    beta.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    std::vector<double> normalized(static_cast<std::size_t>(geometry.hidden));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t x_base = static_cast<std::size_t>(token) * geometry.hidden;
        double sum_squares       = 0.0;
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
            sum_squares += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(sum_squares / static_cast<double>(geometry.hidden) + eps);
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]) *
                                 inverse_rms * (1.0 + static_cast<double>(norm_weight[k]));
            normalized[static_cast<std::size_t>(k)] = value;
            h[x_base + static_cast<std::size_t>(k)] = value;
        }

        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = normalized[static_cast<std::size_t>(k)];
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = static_cast<std::size_t>(token) * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

std::vector<double> select_tokens(const std::vector<double>& full,
                                  const std::vector<std::int32_t>& selected_tokens,
                                  std::int32_t rows) {
    std::vector<double> selected(static_cast<std::size_t>(rows) * selected_tokens.size());
    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t source = static_cast<std::size_t>(selected_tokens[sample]) * rows;
        std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(source), rows,
                    selected.begin() + static_cast<std::ptrdiff_t>(sample * rows));
    }
    return selected;
}

int require_all_finite(const std::string& label, const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::cerr << label << ": output element " << i
                      << " was not written to a finite value\n";
            return 1;
        }
    }
    return 0;
}

int verify_inputs_unchanged(const std::string& label, const DeviceBuffer& device_x,
                            const std::vector<std::uint16_t>& x_bits,
                            const DeviceBuffer& device_weight,
                            const std::vector<std::uint16_t>& weight_bits,
                            const DeviceBuffer& device_a_log, const std::vector<float>& a_log,
                            const DeviceBuffer& device_dt_bias, const std::vector<float>& dt_bias) {
    int failures = 0;
    failures += verify_exact((label + " x immutable").c_str(),
                             from_device<std::uint16_t>(device_x, x_bits.size()), x_bits);
    failures +=
        verify_exact((label + " weight immutable").c_str(),
                     from_device<std::uint16_t>(device_weight, weight_bits.size()), weight_bits);
    failures += verify_exact((label + " A_log immutable").c_str(),
                             from_device<float>(device_a_log, a_log.size()), a_log);
    failures += verify_exact((label + " dt_bias immutable").c_str(),
                             from_device<float>(device_dt_bias, dt_bias.size()), dt_bias);
    return failures;
}

int run_projection_case(Pass& pass, const Geometry& geometry, std::int32_t tokens,
                        std::uint32_t seed) {
    const ReductionCriterion& criterion = name_projection_fallback(pass, geometry, tokens);
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(a_weight, seed + 1u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 3u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 4u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::int32_t> selected = oracle_tokens(tokens);
    std::vector<double> reference_g, reference_beta;
    projection_oracle(geometry, x, a_weight, b_weight, a_log, dt_bias, selected, reference_g,
                      reference_beta);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(x);
    std::vector<std::uint16_t> weight_bits         = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_weight_bits = bf16_bits(b_weight);
    if (geometry.parent_weight) {
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    }
    DeviceBuffer device_x      = to_device(x_bits);
    DeviceBuffer device_weight = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) { device_b_weight = to_device(b_weight_bits); }
    DeviceBuffer device_a_log         = to_device(a_log);
    DeviceBuffer device_dt_bias       = to_device(dt_bias);
    const std::size_t output_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer device_g(output_elements * sizeof(float));
    GuardedDeviceBuffer device_beta(output_elements * sizeof(float));
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    const std::size_t workspace_bytes = ops::gdn_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (geometry.parent_weight) {
        Weight parent = bf16_weight(device_weight.p, 2 * geometry.heads, geometry.hidden);
        ops::gdn_gating_proj(tensor_x, parent, tensor_a_log, tensor_dt_bias, workspace, tensor_g,
                             tensor_beta, nullptr);
    } else {
        Weight weight_a = bf16_weight(device_weight.p, geometry.heads, geometry.hidden);
        Weight weight_b = bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
        ops::gdn_gating_proj(tensor_x, weight_a, weight_b, tensor_a_log, tensor_dt_bias, workspace,
                             tensor_g, tensor_beta, nullptr);
    }
    cuda_synchronize();

    const std::vector<double> full_g    = read_fp32(device_g.data(), output_elements);
    const std::vector<double> full_beta = read_fp32(device_beta.data(), output_elements);
    const std::string label = pass.label + " gdn_gating_proj " + geometry.label +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += require_all_finite(label + " g", full_g);
    failures += require_all_finite(label + " beta", full_beta);
    failures += verify_normwise(label + " g", select_tokens(full_g, selected, geometry.heads),
                                reference_g, criterion);
    failures += verify_normwise(label + " beta", select_tokens(full_beta, selected, geometry.heads),
                                reference_beta, criterion);
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                        device_a_log, a_log, device_dt_bias, dt_bias);
    if (!geometry.parent_weight) {
        failures += verify_exact((label + " b_weight immutable").c_str(),
                                 from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()),
                                 b_weight_bits);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_norm_projection_case(Pass& pass, const Geometry& geometry, std::int32_t tokens,
                             std::uint32_t seed) {
    name_norm_fallback(pass, geometry, tokens);
    constexpr float kEps = 1.0e-6F;
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> norm_weight(static_cast<std::size_t>(geometry.hidden));
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(norm_weight, seed + 1u, -0.2F, 0.2F);
    fill_uniform(a_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 3u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 4u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 5u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    std::vector<double> reference_h, reference_g, reference_beta;
    norm_projection_oracle(geometry, x, norm_weight, a_weight, b_weight, a_log, dt_bias, tokens,
                           kEps, reference_h, reference_g, reference_beta);

    const std::vector<std::uint16_t> x_bits           = bf16_bits(x);
    const std::vector<std::uint16_t> norm_weight_bits = bf16_bits(norm_weight);
    std::vector<std::uint16_t> weight_bits            = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_weight_bits    = bf16_bits(b_weight);
    if (geometry.parent_weight) {
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    }
    DeviceBuffer device_x           = to_device(x_bits);
    DeviceBuffer device_norm_weight = to_device(norm_weight_bits);
    DeviceBuffer device_weight      = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) { device_b_weight = to_device(b_weight_bits); }
    DeviceBuffer device_a_log          = to_device(a_log);
    DeviceBuffer device_dt_bias        = to_device(dt_bias);
    const std::size_t h_elements       = static_cast<std::size_t>(geometry.hidden) * tokens;
    const std::size_t control_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer device_h(h_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_g(control_elements * sizeof(float));
    GuardedDeviceBuffer device_beta(control_elements * sizeof(float));
    device_h.fill(0xff);
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_norm_weight(device_norm_weight.p, DType::BF16, {geometry.hidden});
    Tensor tensor_h(device_h.data(), DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    const std::size_t workspace_bytes = ops::gdn_norm_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (geometry.parent_weight) {
        Weight parent = bf16_weight(device_weight.p, 2 * geometry.heads, geometry.hidden);
        ops::gdn_norm_gating_proj(tensor_x, tensor_norm_weight, kEps, parent, tensor_a_log,
                                  tensor_dt_bias, workspace, tensor_h, tensor_g, tensor_beta,
                                  nullptr);
    } else {
        Weight weight_a = bf16_weight(device_weight.p, geometry.heads, geometry.hidden);
        Weight weight_b = bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
        ops::gdn_norm_gating_proj(tensor_x, tensor_norm_weight, kEps, weight_a, weight_b,
                                  tensor_a_log, tensor_dt_bias, workspace, tensor_h, tensor_g,
                                  tensor_beta, nullptr);
    }
    cuda_synchronize();

    const std::string label = pass.label + " gdn_norm_gating_proj " + geometry.label +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_normwise(label + " h", from_device_bf16(device_h.data(), h_elements),
                                reference_h, kGdnNormOutputBf16);
    failures += verify_normwise(label + " g", read_fp32(device_g.data(), control_elements),
                                reference_g, kGdnNormControlFp32);
    failures += verify_normwise(label + " beta", read_fp32(device_beta.data(), control_elements),
                                reference_beta, kGdnNormControlFp32);
    failures += device_h.verify_guards((label + " h").c_str());
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                        device_a_log, a_log, device_dt_bias, dt_bias);
    failures += verify_exact(
        (label + " norm_weight immutable").c_str(),
        from_device<std::uint16_t>(device_norm_weight, norm_weight_bits.size()), norm_weight_bits);
    if (!geometry.parent_weight) {
        failures += verify_exact((label + " b_weight immutable").c_str(),
                                 from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()),
                                 b_weight_bits);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

using CapacityQuery = std::size_t (*)(std::int32_t, std::int32_t, std::int32_t, std::int32_t);

// Ignis portability patch (ADR 0077, GitHub #602): an interval's capacity is the peak over every
// width in it, so it must equal the largest single-width answer at every width from 1 to `last`.
// On the RTX 5090 with no cap (`reference_card`) it must also equal the largest answer at the
// route endpoints the reference checked, so the engine's reservation there is the reference's.
int verify_capacity_interval(const Pass& pass, const Geometry& geometry, const char* op,
                             CapacityQuery query, std::int32_t last,
                             std::initializer_list<std::int32_t> route_endpoints,
                             bool reference_card) {
    const std::string label = pass.label + " " + op + " " + geometry.label + " workspace T=1.." +
                              std::to_string(last);
    return guarded(label, [&] {
        const std::size_t interval = query(geometry.heads, geometry.hidden, 1, last);
        std::size_t witness        = 0;
        std::int32_t witness_cols  = 0;
        for (std::int32_t tokens = 1; tokens <= last; ++tokens) {
            const std::size_t bytes = query(geometry.heads, geometry.hidden, tokens, tokens);
            if (bytes > witness) {
                witness      = bytes;
                witness_cols = tokens;
            }
        }
        std::cout << "capacity " << label << ": " << interval << " bytes, the widest call "
                  << witness << " at T=" << witness_cols << "\n";
        int failures = 0;
        if (interval != witness) {
            std::cerr << label << ": the interval's capacity is " << interval << " bytes, a call of "
                      << witness_cols << " columns allocates " << witness << "\n";
            ++failures;
        }
        if (reference_card) {
            std::size_t endpoints = 0;
            for (const std::int32_t tokens : route_endpoints) {
                endpoints =
                    std::max(endpoints, query(geometry.heads, geometry.hidden, tokens, tokens));
            }
            if (interval != endpoints) {
                std::cerr << label << ": the interval's capacity is " << interval
                          << " bytes, its route endpoints' " << endpoints << "\n";
                ++failures;
            }
        }
        return failures;
    });
}

int verify_workspace_capacity_contract(const Pass& pass, const Geometry& geometry,
                                       std::initializer_list<std::int32_t> route_endpoints,
                                       bool reference_card) {
    const std::int32_t last = *std::max_element(route_endpoints.begin(), route_endpoints.end());
    int failures            = 0;
    failures += verify_capacity_interval(pass, geometry, "gdn_gating_proj",
                                         ops::gdn_gating_proj_workspace_capacity_bytes, last,
                                         route_endpoints, reference_card);
    failures += verify_capacity_interval(pass, geometry, "gdn_norm_gating_proj",
                                         ops::gdn_norm_gating_proj_workspace_capacity_bytes, 64,
                                         {8, 16, 64}, reference_card);
    return failures;
}

// The whole case list under the pass's SM count.
int run_pass(Pass& pass, bool reference_card) {
    int failures = 0;
    failures +=
        verify_workspace_capacity_contract(pass, kQwen27, {1, 8, 1024, 2048, 4096, 4097},
                                           reference_card);
    failures +=
        verify_workspace_capacity_contract(pass, kQwen35, {1, 127, 1024, 2048, 4096, 4097},
                                           reference_card);

    const auto projection = [&](const Geometry& geometry, std::int32_t tokens,
                                std::uint32_t seed) {
        return guarded(pass.label + " gdn_gating_proj " + geometry.label + " T=" +
                           std::to_string(tokens),
                       [&] { return run_projection_case(pass, geometry, tokens, seed); });
    };
    const auto norm = [&](const Geometry& geometry, std::int32_t tokens, std::uint32_t seed) {
        return guarded(pass.label + " gdn_norm_gating_proj " + geometry.label + " T=" +
                           std::to_string(tokens),
                       [&] { return run_norm_projection_case(pass, geometry, tokens, seed); });
    };

    // Every registered 27B projection route, including predicated and full token tiles, and the
    // full tiles at 2048 and 4096 a smaller card's fallbacks run (GitHub #602).
    for (const std::int32_t tokens : {1, 8, 9, 1024, 1025, 2048, 2049, 4096, 4097}) {
        failures += projection(kQwen27, tokens, 0x1000u + static_cast<std::uint32_t>(tokens));
    }
    // The Qwen3.8 parent changes only the public storage boundary. One direct oracle case proves
    // its [A,B] row partition; the split 27B cases above cover every unchanged execution route.
    failures += projection(kQwen38Parent, 1, 0x1801u);
    // Every registered 35B projection route and its contiguous-parent storage contract.
    for (const std::int32_t tokens : {1, 127, 128, 1024, 1025, 2048, 2049, 4096, 4097}) {
        failures += projection(kQwen35, tokens, 0x2000u + static_cast<std::uint32_t>(tokens));
    }

    // 27B qualifies the fused split-80 route (T=1..8) and its composed boundary; the 35B
    // qualifies both sides of its own fused boundary.
    failures += norm(kQwen27, 1, 0x3001u);
    failures += norm(kQwen27, 6, 0x3006u);
    failures += norm(kQwen27, 8, 0x3008u);
    failures += norm(kQwen27, 9, 0x3009u);
    failures += norm(kQwen27, 64, 0x3040u);
    failures += norm(kQwen38Parent, 1, 0x3801u);
    failures += norm(kQwen35, 1, 0x4001u);
    failures += norm(kQwen35, 16, 0x4010u);
    failures += norm(kQwen35, 17, 0x4011u);
    failures += norm(kQwen35, 64, 0x4040u);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    const DeviceShape device = current_device_shape();
    failures += verify_cooperative_budgets(device);

    // The card's own SM count, then each test-only cap below it: the RTX 5080's 84 and the
    // RTX 5060 Ti's 36 (GitHub #602).
    std::vector<int> sm_counts{device.multiprocessors};
    for (const int cap : {84, 36}) {
        if (cap < device.multiprocessors) { sm_counts.push_back(cap); }
    }
    for (const int sm_count : sm_counts) {
        const bool capped = sm_count != device.multiprocessors;
        ops::detail::bf16_gdn_gating_set_test_multiprocessor_cap(capped ? sm_count : 0);
        Pass pass{capped ? "[cap " + std::to_string(sm_count) + " SMs]"
                         : "[" + std::to_string(sm_count) + " SMs]",
                  0};
        std::cout << "pass " << pass.label << ": the planner's residency test reads "
                  << sm_count << " of the card's " << device.multiprocessors << " SMs\n";
        const bool reference_card =
            !capped && is_sm120(device) && device.multiprocessors >= kRtx5090Multiprocessors;
        failures += run_pass(pass, reference_card);
        if (reference_card && pass.fallbacks != 0) {
            std::cerr << pass.label << " gdn_gating_proj: " << pass.fallbacks
                      << " case(s) fell back on a card the schedules are sized for\n";
            ++failures;
        }
    }
    ops::detail::bf16_gdn_gating_set_test_multiprocessor_cap(0);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating_proj correctness\n";
    return failures == 0 ? 0 : 1;
}
