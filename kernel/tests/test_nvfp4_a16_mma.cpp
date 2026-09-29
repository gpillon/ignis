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
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test rather than skipping it.

#include "ignis_nvfp4_a16_mma.h"

#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

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

int run_nvfp4_a16_mma() {
    int failures = check_route_selection();

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
