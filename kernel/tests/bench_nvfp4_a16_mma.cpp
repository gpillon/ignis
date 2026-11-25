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
//   ignis_nvfp4_a16_mma_bench [--iters 50] [--all | --drafter | --narrow | --taps | --body]
//
// --taps (GitHub #521) times a prompt-reuse tail: the two problems a
// prefill's DFlash2 feature taps call, feature_projection [5120, 25600] and
// query_key_value [6144, 5120], at every width from 1 to 64 columns. The
// first T from which the route's column (x, the GEMVs' time over the
// route's) stays above 1 on both is the crossover the taps' 8.x width
// (kPrefillTapsMinTokens8x, kernel/src/nvfp4_a16_mma.cu) is set from.
//
// --body (ADR 0107) times the 27B's NVFP4 body at decode and verify widths,
// T = 1, 8, 16, 32, 64 and 128: its five registered problems, each on both
// routes, cycling over enough copies of the weight that none is in the L2
// when it is read (as ignis_kernel_trellis_linear_bench does), then one
// token's body (48 GDN and 16 GQA layers: 48 GdnInput, 16 AttnInput and 64
// each of Residual6144, MlpGateUp and Residual17408) on the GEMVs, on the
// route, and on the faster of the two per problem. It is the NVFP4 side of
// the question which body a card wants; the trellis side is the trellis
// bench's `body` line.

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

        std::vector<std::uint16_t> gemv_host(static_cast<std::size_t>(n) * t);
        std::vector<std::uint16_t> mma_host(gemv_host.size());
        CUDA_OK(cudaMemcpy(gemv_host.data(), gemv_out, gemv_host.size() * 2,
                           cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(mma_host.data(), mma_out, mma_host.size() * 2,
                           cudaMemcpyDeviceToHost));
        double max_diff = 0.0;
        double max_abs  = 0.0;
        for (std::size_t i = 0; i < gemv_host.size(); ++i) {
            const double gemv = bf16_to_float(gemv_host[i]);
            max_diff          = std::max(max_diff, std::fabs(gemv - bf16_to_float(mma_host[i])));
            max_abs           = std::max(max_abs, std::fabs(gemv));
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

// --body: one weight problem at decode and verify widths, the weight cycled
// over `copies` device copies so that no call reads it from the L2. Returns
// the median microseconds per call of both routes at each width.
struct BodyTimes {
    std::vector<double> gemv;
    std::vector<double> mma;
};

BodyTimes bench_body_problem(std::int32_t n, std::int32_t k, const std::vector<std::int32_t>& widths,
                             int iters) {
    const auto host_weight = test::linear::make_nvfp4_weight(n, k, 901U);
    int device = 0;
    int l2_bytes = 0;
    CUDA_OK(cudaGetDevice(&device));
    CUDA_OK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
    const std::size_t bytes  = host_weight.payload.size();
    const std::size_t copies = std::max<std::size_t>(2, (4 * static_cast<std::size_t>(l2_bytes)) / bytes + 1);
    std::vector<void*> payloads(copies, nullptr);
    std::vector<Weight> weights;
    for (auto& payload : payloads) {
        CUDA_OK(cudaMalloc(&payload, bytes));
        CUDA_OK(cudaMemcpy(payload, host_weight.payload.data(), bytes, cudaMemcpyHostToDevice));
        weights.push_back(host_weight.device_weight(payload));
    }
    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_t);
    std::mt19937 rng(7U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& bits : activation) {
        const __nv_bfloat16 value = __float2bfloat16_rn(normal(rng));
        std::memcpy(&bits, &value, sizeof(bits));
    }
    void* x_data = nullptr;
    void* out    = nullptr;
    CUDA_OK(cudaMalloc(&x_data, activation.size() * sizeof(std::uint16_t)));
    CUDA_OK(cudaMalloc(&out, static_cast<std::size_t>(n) * max_t * sizeof(std::uint16_t)));
    CUDA_OK(cudaMemcpy(x_data, activation.data(), activation.size() * sizeof(std::uint16_t),
                       cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    BodyTimes times;
    for (const std::int32_t t : widths) {
        Tensor x(x_data, DType::BF16, {k, t});
        Tensor y(out, DType::BF16, {n, t});
        std::size_t next = 0;
        const double gemv_us = median_us(iters, stream, [&] {
            ops::detail::nvfp4_dispatch(x, weights[next++ % copies], y, ops::LinearPolicy::A16Only,
                                        nullptr, stream);
        });
        next = 0;
        const double mma_us = median_us(iters, stream, [&] {
            ignis_nvfp4_a16_mma(x, weights[next++ % copies], y, stream);
        });
        const double gb = static_cast<double>(bytes) * 1e-3;
        std::printf("[%5d,%5d] T=%4d  gemv %8.1f us %6.1f GB/s   mma %8.1f us %6.1f GB/s   x%5.2f\n", n, k,
                    t, gemv_us, gb / gemv_us, mma_us, gb / mma_us, gemv_us / mma_us);
        times.gemv.push_back(gemv_us);
        times.mma.push_back(mma_us);
    }
    CUDA_OK(cudaStreamDestroy(stream));
    CUDA_OK(cudaFree(x_data));
    CUDA_OK(cudaFree(out));
    for (void* payload : payloads) { CUDA_OK(cudaFree(payload)); }
    return times;
}

void bench_body(int iters) {
    const std::vector<std::int32_t> widths{1, 8, 16, 32, 64, 128};
    struct Problem {
        std::int32_t n, k, per_token;
    };
    // One token through the body: 48 GDN and 16 GQA layers, each with an
    // input projection, a 6,144-column out-projection and the MLP.
    const Problem problems[] = {{16384, 5120, 48}, {14336, 5120, 16}, {5120, 6144, 64},
                                {34816, 5120, 64}, {5120, 17408, 64}};
    std::vector<double> gemv(widths.size(), 0.0), mma(widths.size(), 0.0), best(widths.size(), 0.0);
    for (const Problem& p : problems) {
        const BodyTimes times = bench_body_problem(p.n, p.k, widths, iters);
        for (std::size_t i = 0; i < widths.size(); ++i) {
            gemv[i] += p.per_token * times.gemv[i];
            mma[i] += p.per_token * times.mma[i];
            best[i] += p.per_token * std::min(times.gemv[i], times.mma[i]);
        }
    }
    for (std::size_t i = 0; i < widths.size(); ++i) {
        std::printf("body T=%4d  gemv %8.2f ms   mma %8.2f ms   best per problem %8.2f ms\n", widths[i],
                    gemv[i] * 1e-3, mma[i] * 1e-3, best[i] * 1e-3);
    }
}

} // namespace

int main(int argc, char** argv) {
    int iters = 50;
    bool all     = false;
    bool drafter = false;
    bool narrow  = false;
    bool taps    = false;
    bool body    = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--all") == 0) {
            all = true;
        } else if (std::strcmp(argv[i], "--drafter") == 0) {
            drafter = true;
        } else if (std::strcmp(argv[i], "--narrow") == 0) {
            narrow = true;
        } else if (std::strcmp(argv[i], "--taps") == 0) {
            taps = true;
        } else if (std::strcmp(argv[i], "--body") == 0) {
            body = true;
        }
    }
    const std::vector<std::int32_t> widths{64, 128, 394, 1024};
    if (body) {
        bench_body(iters);
        return 0;
    }
    if (taps) {
        std::vector<std::int32_t> tail;
        for (std::int32_t t = 1; t <= 64; ++t) { tail.push_back(t); }
        bench_shape(5120, 25600, tail, iters);
        bench_shape(6144, 5120, tail, iters);
        return 0;
    }
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
