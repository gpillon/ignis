// The A16 NVFP4 linear at prompt width, both routes timed on the same weight
// and activation -- OURS, and NOT a CTest test (a timing is a finding, not a
// pass/fail; docs/agents/testing.md).
//
//   gemv   the vendored A16 route (detail::nvfp4_dispatch, A16Only): one
//          small-T GEMV launch per 32-column slice
//   mma    ours (ignis_nvfp4_a16_mma): one tensor-core launch
//
// For each registered problem named on the command line (default: the two the
// DFlash2 context append calls) and each width, prints the median time per
// call over --iters launches, the achieved TFLOP/s, the speedup, and the
// largest |mma - gemv| over the output as a same-input sanity figure (both
// are held to the A16 criterion by the CTest; this only shows they agree).
//
//   ignis_nvfp4_a16_mma_bench [--iters 50] [--all | --drafter | --narrow]

#include "ignis_nvfp4_a16_mma.h"

#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace ninfer;

#define CUDA_OK(expr)                                                                              \
    do {                                                                                           \
        const cudaError_t status_ = (expr);                                                        \
        if (status_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(status_));                  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

template <class Launch>
double median_us(int iters, cudaStream_t stream, Launch&& launch) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    for (int i = 0; i < 3; ++i) { launch(); }
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        CUDA_OK(cudaEventRecord(start, stream));
        launch();
        CUDA_OK(cudaEventRecord(stop, stream));
        CUDA_OK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms * 1000.0);
    }
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

void bench_shape(std::int32_t n, std::int32_t k, const std::vector<std::int32_t>& widths,
                 int iters) {
    const auto host_weight = test::linear::make_nvfp4_weight(n, k, 901U);
    void* weight_payload   = nullptr;
    CUDA_OK(cudaMalloc(&weight_payload, host_weight.payload.size()));
    CUDA_OK(cudaMemcpy(weight_payload, host_weight.payload.data(), host_weight.payload.size(),
                       cudaMemcpyHostToDevice));
    const Weight weight = host_weight.device_weight(weight_payload);

    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_t);
    std::mt19937 rng(7U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& bits : activation) {
        const __nv_bfloat16 value = __float2bfloat16_rn(normal(rng));
        std::memcpy(&bits, &value, sizeof(bits));
    }
    void* x_data   = nullptr;
    void* gemv_out = nullptr;
    void* mma_out  = nullptr;
    const std::size_t out_bytes = static_cast<std::size_t>(n) * max_t * sizeof(std::uint16_t);
    CUDA_OK(cudaMalloc(&x_data, activation.size() * sizeof(std::uint16_t)));
    CUDA_OK(cudaMalloc(&gemv_out, out_bytes));
    CUDA_OK(cudaMalloc(&mma_out, out_bytes));
    CUDA_OK(cudaMemcpy(x_data, activation.data(), activation.size() * sizeof(std::uint16_t),
                       cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    for (const std::int32_t t : widths) {
        Tensor x(x_data, DType::BF16, {k, t});
        Tensor gemv(gemv_out, DType::BF16, {n, t});
        Tensor mma(mma_out, DType::BF16, {n, t});
        const double gemv_us = median_us(iters, stream, [&] {
            ops::detail::nvfp4_dispatch(x, weight, gemv, ops::LinearPolicy::A16Only, nullptr,
                                        stream);
        });
        const double mma_us =
            median_us(iters, stream, [&] { ignis_nvfp4_a16_mma(x, weight, mma, stream); });

        std::vector<std::uint16_t> a(static_cast<std::size_t>(n) * t);
        std::vector<std::uint16_t> b(a.size());
        CUDA_OK(cudaMemcpy(a.data(), gemv_out, a.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(b.data(), mma_out, b.size() * 2, cudaMemcpyDeviceToHost));
        double max_diff = 0.0;
        double max_abs  = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(bf16_to_float(a[i])) -
                                                    bf16_to_float(b[i])));
            max_abs  = std::max(max_abs, std::fabs(static_cast<double>(bf16_to_float(a[i]))));
        }
        const double flops = 2.0 * n * k * t;
        std::printf("[%5d,%5d] T=%5d  gemv %9.1f us %6.1f TF/s   mma %8.1f us %6.1f TF/s   "
                    "x%5.2f   max|diff| %.3g (max|y| %.3g)\n",
                    n, k, t, gemv_us, flops / gemv_us * 1e-6, mma_us, flops / mma_us * 1e-6,
                    gemv_us / mma_us, max_diff, max_abs);
    }
    CUDA_OK(cudaStreamDestroy(stream));
    CUDA_OK(cudaFree(x_data));
    CUDA_OK(cudaFree(gemv_out));
    CUDA_OK(cudaFree(mma_out));
    CUDA_OK(cudaFree(weight_payload));
}

} // namespace

int main(int argc, char** argv) {
    int iters = 50;
    bool all     = false;
    bool drafter = false;
    bool narrow  = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--all") == 0) {
            all = true;
        } else if (std::strcmp(argv[i], "--drafter") == 0) {
            drafter = true;
        } else if (std::strcmp(argv[i], "--narrow") == 0) {
            narrow = true;
        }
    }
    const std::vector<std::int32_t> widths{64, 128, 394, 1024};
    if (narrow) {
        // The narrow registered problems, where M / 64 CTAs leaves the card
        // idle until T spreads the grid over several token tiles.
        const std::vector<std::int32_t> spread{64, 128, 256, 394, 512, 1024};
        bench_shape(1280, 5120, spread, iters);
        bench_shape(256, 5120, spread, iters);
        return 0;
    }
    if (!drafter) {
        bench_shape(5120, 25600, widths, iters);
        bench_shape(6144, 5120, widths, iters);
    }
    if (all) {
        bench_shape(5120, 17408, widths, iters);
        bench_shape(34816, 5120, widths, iters);
    }
    if (drafter) {
        // The drafter's propose forward at IGNIS_DECODE_MAX_BATCH: a block of
        // draft 7 + bonus per lane, eight lanes, is exactly 64 columns.
        const std::vector<std::int32_t> round{64};
        bench_shape(1280, 5120, round, iters);
        bench_shape(5120, 4096, round, iters);
        bench_shape(6144, 5120, round, iters);
        bench_shape(5120, 17408, round, iters);
    }
    return 0;
}
