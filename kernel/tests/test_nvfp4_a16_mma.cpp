// The A16 NVFP4 linear's tensor-core route -- OURS, not vendored
// (kernel/include/ignis_nvfp4_a16_mma.h) -- held to the Linear contract with
// the vendored test harness's own oracle and criterion.
//
// The vendored `test_nvfp4_a16.cpp` stops at T=33, one column past the small-T
// family, because that family is all the reference has for A16. The MMA route
// takes every A16 call of 64 columns or more (128 on the narrow problems), so
// this test drives it where the engine does: prompt-width calls, and the
// drafter's 64-column round at eight lanes. The criterion is the vendored one for
// the A16 activation path (`tolerance_for(ActivationCompute::A16)`, one BF16
// unit roundoff), against the harness's FP64 oracle over the dequantized
// weight -- the route has to meet the same contract the GEMVs meet, not a
// looser one of its own.
//
// What the arms are for:
//
//   production   the two matrices the drafter's context append calls at
//                prefill width -- feature_projection [5120, 25600] and
//                query_key_value [6144, 5120] -- at a full 1,024-column
//                chunk and at a ragged tail, through the A16 convenience
//                overload the drafter uses.
//   full         every output element checked, on the two narrowest
//                registered problems ([256, 5120] and [1280, 5120]), so
//                every weight row -- all four row quartiles of a blockscale
//                tile, both halves a 64-row CTA can sit in -- and every
//                token of a ragged last CTA are compared, not a sample.
//   boundaries   the first width the route takes (64, or 128 below 4,096
//                rows), multiples of the 128-column CTA, and ragged tails.
//   geometries   every other registered NVFP4 problem, sampled, so no
//                instantiation ships untested.
//   dispatch     which kernel `ops::linear` actually ran, pinned bit for bit:
//                either side of each threshold its output must equal the MMA
//                route's own output or the vendored GEMVs'. Every arm above
//                would also pass on the GEMVs, which meet the same criterion,
//                so without this one a lost route would stay green.
//   generations  the widths per GPU generation (#521), for 8.x and 12.x
//                whatever card runs this, and the dispatch of a prefill's
//                feature taps (`ignis_linear_a16`) either side of the width
//                this card's generation gives them.
//   tail         the route at a prompt-reuse tail's widths, 1 to 63 columns,
//                on the taps' two problems, against the oracle and against
//                the GEMVs: what an 8.x card's taps take, checked on any card.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test rather than skipping it.

#include "ignis_nvfp4_a16_mma.h"

#include "ninfer/ops/linear.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr auto kPolicy = ops::LinearPolicy::A16Only;

int check_route_selection() {
    int failures = 0;
    auto expect = [&](bool actual, bool expected, const char* what) {
        if (actual != expected) {
            std::cerr << "route selection: " << what << '\n';
            ++failures;
        }
    };
    expect(ignis_nvfp4_a16_mma_applies(5120, 25600, 63), false, "T=63 stays on the GEMVs");
    expect(ignis_nvfp4_a16_mma_applies(5120, 25600, 64), true, "T=64 takes the MMA route");
    expect(ignis_nvfp4_a16_mma_applies(6144, 5120, 1024), true, "drafter qkv at prefill width");
    expect(ignis_nvfp4_a16_mma_applies(1280, 5120, 64), false, "a narrow problem at T=64");
    expect(ignis_nvfp4_a16_mma_applies(1280, 5120, 127), false, "a narrow problem at T=127");
    expect(ignis_nvfp4_a16_mma_applies(256, 5120, 128), true, "a narrow problem at T=128");
    expect(ignis_nvfp4_a16_mma_applies(4096, 4096, 1024), false, "an unregistered problem");
    return failures;
}

// #521: the route's widths per GPU generation, from the one place that chooses them
// (`ignis_nvfp4_a16_mma_min_tokens`), pinned for both generations whatever card runs this. On
// 12.x every call keeps 64, or 128 below 4,096 rows. On 8.x a prefill chunk's DFlash2 feature
// taps -- feature_projection [5120, 25600] and query_key_value [6144, 5120], both wide -- take
// the route from a width of their own, below 64, so a prompt-reuse tail (0-63 tokens) can reach
// it; every other call there, the decode round's included, keeps 64 and 128.
int check_widths_per_generation() {
    using Call   = IgnisA16Call;
    int failures = 0;
    auto expect  = [&](std::int32_t actual, std::int32_t expected, const char* what) {
        if (actual != expected) {
            std::cerr << "widths per generation: " << what << ": " << actual << ", not "
                      << expected << '\n';
            ++failures;
        }
    };
    expect(ignis_nvfp4_a16_mma_min_tokens(12, Call::General, 6144), 64, "12.x, a wide call");
    expect(ignis_nvfp4_a16_mma_min_tokens(12, Call::General, 1280), 128, "12.x, a narrow call");
    expect(ignis_nvfp4_a16_mma_min_tokens(12, Call::PrefillTaps, 5120), 64,
           "12.x, the taps' feature_projection");
    expect(ignis_nvfp4_a16_mma_min_tokens(12, Call::PrefillTaps, 6144), 64,
           "12.x, the taps' query_key_value");
    expect(ignis_nvfp4_a16_mma_min_tokens(12, Call::PrefillTaps, 1280), 128,
           "12.x, a narrow problem as taps");
    expect(ignis_nvfp4_a16_mma_min_tokens(8, Call::General, 5120), 64, "8.x, a wide call");
    expect(ignis_nvfp4_a16_mma_min_tokens(8, Call::General, 6144), 64,
           "8.x, the decode round's query_key_value");
    expect(ignis_nvfp4_a16_mma_min_tokens(8, Call::General, 1280), 128, "8.x, a narrow call");
    expect(ignis_nvfp4_a16_mma_min_tokens(8, Call::General, 256), 128, "8.x, the narrowest call");
    const std::int32_t taps = ignis_nvfp4_a16_mma_min_tokens(8, Call::PrefillTaps, 5120);
    if (taps < 1 || taps >= 64) {
        std::cerr << "widths per generation: 8.x, the prefill taps' width " << taps
                  << " is not a reuse tail's (1-63)\n";
        ++failures;
    }
    expect(ignis_nvfp4_a16_mma_min_tokens(8, Call::PrefillTaps, 6144), taps,
           "8.x, the taps' query_key_value takes the taps' width");
    expect(ignis_nvfp4_a16_mma_min_tokens(8, Call::PrefillTaps, 1280), 128,
           "8.x, a narrow problem as taps");
    expect(ignis_nvfp4_a16_mma_min_tokens(0, Call::PrefillTaps, 5120), 64,
           "a generation the device could not say");
    return failures;
}

void cuda_ok(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBytes {
    void* p = nullptr;
    explicit DeviceBytes(std::size_t bytes) { cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~DeviceBytes() { cudaFree(p); }
    DeviceBytes(const DeviceBytes&)            = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
};

// Runs one A16 call three ways on the same inputs -- through `ops::linear`
// (through `ignis_linear_a16` for a prefill's taps, #521), straight into the
// MMA route, straight into the vendored dispatch -- and requires the first to
// be bit-identical to the kernel the route rule names. Both kernels are
// deterministic (no atomics, fixed reduction order),
// and they accumulate in different orders, so above the threshold the MMA
// output must also differ from the GEMVs' somewhere: equality there would
// mean the comparison cannot tell the two routes apart.
int check_dispatch(std::int32_t n, std::int32_t k, std::int32_t t, bool expect_mma,
                   IgnisA16Call call = IgnisA16Call::General) {
    const std::string label = std::string(call == IgnisA16Call::PrefillTaps ? "prefill taps " : "") +
                              "dispatch [" + std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t);
    if (ignis_nvfp4_a16_mma_applies(n, k, t, call) != expect_mma) {
        std::cerr << label << ": route rule disagrees with the arm\n";
        return 1;
    }
    const auto host_weight = make_nvfp4_weight(n, k, 830U);
    DeviceBytes weight_bytes(host_weight.payload.size());
    cuda_ok(cudaMemcpy(weight_bytes.p, host_weight.payload.data(), host_weight.payload.size(),
                       cudaMemcpyHostToDevice),
            "copy weight");
    const Weight weight = host_weight.device_weight(weight_bytes.p);

    // Activations spread over 2^-12..2^12. With the harness's weights and a
    // unit-normal activation, every FP32 partial sum at K=5,120 is exact, so
    // any accumulation order gives the same bits and the two kernels cannot
    // be told apart; a wide exponent range makes the sums round.
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(831U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-12, 12);
    for (auto& bits : activation) {
        const __nv_bfloat16 value = __float2bfloat16_rn(std::ldexp(normal(rng), exponent(rng)));
        std::memcpy(&bits, &value, sizeof(bits));
    }
    const std::size_t out_elements = static_cast<std::size_t>(n) * t;
    DeviceBytes x_bytes(activation.size() * sizeof(std::uint16_t));
    DeviceBytes linear_bytes(out_elements * sizeof(std::uint16_t));
    DeviceBytes route_bytes(out_elements * sizeof(std::uint16_t));
    cuda_ok(cudaMemcpy(x_bytes.p, activation.data(), activation.size() * sizeof(std::uint16_t),
                       cudaMemcpyHostToDevice),
            "copy activation");

    Tensor x(x_bytes.p, DType::BF16, {k, t});
    Tensor through_linear(linear_bytes.p, DType::BF16, {n, t});
    Tensor direct(route_bytes.p, DType::BF16, {n, t});
    if (call == IgnisA16Call::General) {
        ops::linear(x, weight, through_linear, nullptr);
    } else {
        ignis_linear_a16(x, weight, through_linear, call, nullptr);
    }
    ops::detail::nvfp4_dispatch(x, weight, direct, kPolicy, nullptr, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "run linear and the GEMVs");
    std::vector<std::uint16_t> linear_out(out_elements);
    std::vector<std::uint16_t> gemv_out(out_elements);
    cuda_ok(cudaMemcpy(linear_out.data(), linear_bytes.p, out_elements * 2, cudaMemcpyDeviceToHost),
            "read linear");
    cuda_ok(cudaMemcpy(gemv_out.data(), route_bytes.p, out_elements * 2, cudaMemcpyDeviceToHost),
            "read GEMVs");

    if (!expect_mma) {
        if (linear_out != gemv_out) {
            std::cerr << label << ": ops::linear is not the vendored GEMVs below the threshold\n";
            return 1;
        }
        return 0;
    }
    ignis_nvfp4_a16_mma(x, weight, direct, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "run the MMA route");
    std::vector<std::uint16_t> mma_out(out_elements);
    cuda_ok(cudaMemcpy(mma_out.data(), route_bytes.p, out_elements * 2, cudaMemcpyDeviceToHost),
            "read MMA route");
    int failures = 0;
    if (linear_out != mma_out) {
        std::cerr << label << ": ops::linear is not the MMA route above the threshold\n";
        ++failures;
    }
    if (mma_out == gemv_out) {
        std::cerr << label << ": the MMA route and the GEMVs agree bit for bit, so this arm "
                              "cannot tell which one ops::linear ran\n";
        ++failures;
    }
    return failures;
}

// The compute capability major of the card this runs on.
int device_compute_major() {
    int device = 0;
    int major  = 0;
    cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    cuda_ok(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
            "the compute capability");
    return major;
}

// #521: the route at a prompt-reuse tail's widths, 1 to 63 columns, on a
// problem a prefill's feature taps call -- every width an 8.x card's taps may
// take it from, whatever the measured value, and on any card: the kernel
// takes a ragged CTA at any width (zero-filled token columns, guarded stores).
// Each width is held to the A16 criterion (`tolerance_for(A16)` in the
// harness: one BF16 unit roundoff of relative L2, gross error within 1/256 +
// 2/256 of the largest reference) against the harness's FP64 oracle over the
// dequantized weight, on a row sample that visits every residue of a 64-row
// CTA and of a 128-row scale tile, and against the vendored GEMVs, every
// output, on the same unit-normal input: the route agrees with the kernels a
// reuse tail ran on before.
int check_tail_widths(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    constexpr std::array<std::int32_t, 11> kWidths{1, 2, 7, 8, 16, 31, 32, 33, 40, 48, 63};
    constexpr std::int32_t kMaxT = 63;
    constexpr test::ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    const auto host_weight = make_nvfp4_weight(n, k, seed);
    DeviceBytes weight_bytes(host_weight.payload.size());
    cuda_ok(cudaMemcpy(weight_bytes.p, host_weight.payload.data(), host_weight.payload.size(),
                       cudaMemcpyHostToDevice),
            "copy weight");
    const Weight weight = host_weight.device_weight(weight_bytes.p);

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * kMaxT);
    std::vector<float> activation_f32(activation.size());
    std::mt19937 rng(seed + 1U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        const __nv_bfloat16 value = __float2bfloat16_rn(normal(rng));
        std::memcpy(&activation[i], &value, sizeof(std::uint16_t));
        activation_f32[i] = test::bf16_to_f32(activation[i]);
    }
    std::vector<std::int32_t> rows;
    for (std::int32_t row = 0; row < n; row += 13) { rows.push_back(row); }
    if (rows.back() != n - 1) { rows.push_back(n - 1); }
    const auto sampled = static_cast<std::int32_t>(rows.size());
    const std::vector<float> oracle_weight =
        test::quantized_weight::materialize_rows_fp32(host_weight, rows);
    // [kMaxT][sampled]: a width's oracle is its prefix, the columns being independent.
    std::vector<double> oracle(static_cast<std::size_t>(sampled) * kMaxT);
    cpu_linear_gemm_fp64(oracle_weight.data(), activation_f32.data(), oracle.data(), sampled, k,
                         kMaxT);

    const std::size_t out_bytes = static_cast<std::size_t>(n) * kMaxT * sizeof(std::uint16_t);
    DeviceBytes x_bytes(activation.size() * sizeof(std::uint16_t));
    DeviceBytes route_bytes(out_bytes);
    DeviceBytes gemv_bytes(out_bytes);
    cuda_ok(cudaMemcpy(x_bytes.p, activation.data(), activation.size() * sizeof(std::uint16_t),
                       cudaMemcpyHostToDevice),
            "copy activation");
    int failures = 0;
    for (const std::int32_t t : kWidths) {
        const std::string label = "tail [" + std::to_string(n) + "," + std::to_string(k) +
                                  "] T=" + std::to_string(t);
        const Tensor x(x_bytes.p, DType::BF16, {k, t});
        Tensor route(route_bytes.p, DType::BF16, {n, t});
        Tensor gemv(gemv_bytes.p, DType::BF16, {n, t});
        try {
            ignis_nvfp4_a16_mma(x, weight, route, nullptr);
        } catch (const std::exception& error) {
            std::cerr << label << ": the route refused the width: " << error.what() << '\n';
            ++failures;
            continue;
        }
        ops::detail::nvfp4_dispatch(x, weight, gemv, kPolicy, nullptr, nullptr);
        cuda_ok(cudaDeviceSynchronize(), "run the route and the GEMVs");
        const std::size_t elements = static_cast<std::size_t>(n) * t;
        std::vector<std::uint16_t> route_out(elements);
        std::vector<std::uint16_t> gemv_out(elements);
        cuda_ok(cudaMemcpy(route_out.data(), route_bytes.p, elements * 2, cudaMemcpyDeviceToHost),
                "read the route");
        cuda_ok(cudaMemcpy(gemv_out.data(), gemv_bytes.p, elements * 2, cudaMemcpyDeviceToHost),
                "read the GEMVs");

        std::vector<double> route_sampled;
        route_sampled.reserve(static_cast<std::size_t>(sampled) * t);
        for (std::int32_t column = 0; column < t; ++column) {
            for (const std::int32_t row : rows) {
                route_sampled.push_back(
                    test::bf16_to_f32(route_out[static_cast<std::size_t>(column) * n + row]));
            }
        }
        failures += test::verify_reduction(
            label + " route vs FP64 oracle", route_sampled,
            std::span<const double>(oracle.data(), route_sampled.size()), kA16);

        std::vector<double> route_all(elements);
        std::vector<double> gemv_all(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            route_all[i] = test::bf16_to_f32(route_out[i]);
            gemv_all[i]  = test::bf16_to_f32(gemv_out[i]);
        }
        failures += test::verify_reduction(label + " route vs GEMVs", route_all, gemv_all, kA16);
    }
    return failures;
}

int run_nvfp4_a16_mma() {
    int failures = check_route_selection();
    failures += check_widths_per_generation();

    failures += check_dispatch(6144, 5120, 63, false);
    failures += check_dispatch(6144, 5120, 64, true);
    failures += check_dispatch(1280, 5120, 127, false);
    failures += check_dispatch(1280, 5120, 128, true);
    failures += check_dispatch(5120, 25600, 394, true);

    // #521: a prefill's feature taps through `ignis_linear_a16`, either side of
    // the width this card's generation gives them (64 on 12.x, their own on
    // 8.x). The general arms above pin the decode widths unchanged on this
    // card: T=63 stays on the GEMVs whatever the generation.
    const std::int32_t taps = ignis_nvfp4_a16_mma_min_tokens(
        device_compute_major(), IgnisA16Call::PrefillTaps, 6144);
    std::cout << "prefill taps take the route from T=" << taps << " on this card\n";
    if (taps > 1) {
        failures += check_dispatch(6144, 5120, taps - 1, false, IgnisA16Call::PrefillTaps);
    }
    failures += check_dispatch(6144, 5120, taps, true, IgnisA16Call::PrefillTaps);
    failures += check_dispatch(5120, 25600, taps, true, IgnisA16Call::PrefillTaps);
    if (taps < 63) {
        failures += check_dispatch(6144, 5120, 63, true, IgnisA16Call::PrefillTaps);
    }
    failures += check_tail_widths(6144, 5120, 840U);
    failures += check_tail_widths(5120, 25600, 841U);

    constexpr std::array production{
        Invocation{1024, CallForm::A16Convenience, kPolicy},
        Invocation{1024, CallForm::Policy, kPolicy},
        Invocation{394, CallForm::A16Convenience, kPolicy},
        Invocation{64, CallForm::A16Convenience, kPolicy},
    };
    failures += run_shape("NVFP4_A16_MMA", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 25600, 811U, Comparison::Sampled, true, production});
    failures += run_shape("NVFP4_A16_MMA", ActivationCompute::A16, make_nvfp4_weight,
                          {6144, 5120, 812U, Comparison::Sampled, true, production});

    constexpr std::array full{
        Invocation{128, CallForm::Policy, kPolicy},
        Invocation{129, CallForm::Policy, kPolicy},
        Invocation{200, CallForm::Policy, kPolicy},
        Invocation{256, CallForm::Policy, kPolicy},
        Invocation{300, CallForm::Policy, kPolicy},
    };
    failures += run_shape("NVFP4_A16_MMA", ActivationCompute::A16, make_nvfp4_weight,
                          {256, 5120, 813U, Comparison::Full, true, full});
    failures += run_shape("NVFP4_A16_MMA", ActivationCompute::A16, make_nvfp4_weight,
                          {1280, 5120, 814U, Comparison::Full, true, full});

    constexpr std::array sampled{
        Invocation{64, CallForm::Policy, kPolicy},
        Invocation{256, CallForm::Policy, kPolicy},
        Invocation{1000, CallForm::Policy, kPolicy},
    };
    constexpr std::array<std::array<std::int32_t, 2>, 6> geometries{{
        {5120, 4096},
        {14336, 5120},
        {16384, 5120},
        {34816, 5120},
        {5120, 6144},
        {5120, 17408},
    }};
    std::uint32_t seed = 820U;
    for (const auto& [n, k] : geometries) {
        failures += run_shape("NVFP4_A16_MMA", ActivationCompute::A16, make_nvfp4_weight,
                              {n, k, seed++, Comparison::Sampled, true, sampled});
    }
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run_nvfp4_a16_mma();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 MMA route\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 MMA route: " << error.what() << '\n';
        return 1;
    }
}
