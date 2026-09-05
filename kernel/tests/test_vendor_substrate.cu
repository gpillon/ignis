// Vendored-substrate smoke test (GitHub #42, P1-06).
//
// Ours, not vendored: it drives the vendored reference core (ADR 0010) the way
// the op tests of P1-07..P1-16 will, so a broken vendor drop fails here first.
//
// Two things are checked:
//
//  1. the substrate *compiles* — every vendored header is included below, so a
//     file copied without its dependency breaks the build instead of being
//     discovered by the first op ticket that needs it;
//  2. the substrate *runs* — a device arena is allocated, a bf16 tensor is
//     suballocated from it, a kernel using the vendored `ops::silu` runs on the
//     vendored DeviceContext's stream, and the result is verified against a
//     double-precision CPU reference through the vendored op-test harness.
//
// ADR 0006 / docs/agents/testing.md: this is compute work, so a missing or
// busy GPU is a FAILURE, never a skip. There is no SKIP_RETURN_CODE on this
// test in CTest.

// The vendored substrate, in full — the compile check of (1).
#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/layout.h"
#include "core/nvtx.h"
#include "core/nvtx_range.h"
#include "core/pdl.cuh"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/sampling_workspace.h"
#include "ops/common/token_slices.h"
#include "ops/common/warp.cuh"

// The vendored op-test harness (tests/ops in the reference).
#include "ops/op_tester.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <exception>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::DeviceArena;
using ninfer::DeviceContext;
using ninfer::Tensor;

// bf16 in, bf16 out: the rounding of the store dominates, so the criterion is
// the bf16 grid (2^-8 relative) with a small absolute floor for values near 0.
constexpr ninfer::test::PointwiseCriterion kBf16Criterion{.absolute = 1.0 / 256.0,
                                                          .relative = 1.0 / 128.0};

__global__ void vendor_silu_kernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                                   std::int32_t count) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = __float2bfloat16(ninfer::ops::silu(__bfloat162float(input[index])));
}

// The arena hands out tensors, not raw pointers: check the view it produced is
// the one the kernel is about to read.
int check_tensor_view(const Tensor& tensor, std::int32_t count) {
    int failures = 0;
    if (tensor.dtype != DType::BF16) {
        std::fprintf(stderr, "arena tensor: dtype is not BF16\n");
        ++failures;
    }
    if (tensor.numel() != count) {
        std::fprintf(stderr, "arena tensor: numel %lld != %d\n",
                     static_cast<long long>(tensor.numel()), count);
        ++failures;
    }
    if (tensor.bytes() != static_cast<std::size_t>(count) * ninfer::dtype_size(DType::BF16)) {
        std::fprintf(stderr, "arena tensor: bytes do not match the bf16 element size\n");
        ++failures;
    }
    if (!tensor.is_contiguous()) {
        std::fprintf(stderr, "arena tensor: a fresh allocation must be contiguous\n");
        ++failures;
    }
    return failures;
}

// A scope must return the arena's high-water offset when it ends: that is what
// lets a per-op workspace be reused across the program's layers.
int check_arena_scope(DeviceArena& arena) {
    const std::size_t before = arena.used();
    {
        const auto scope = arena.scope();
        (void)arena.alloc(DType::FP32, {1024});
        if (arena.used() <= before) {
            std::fprintf(stderr, "arena scope: an allocation did not advance the offset\n");
            return 1;
        }
    }
    if (arena.used() != before) {
        std::fprintf(stderr, "arena scope: %zu bytes were not released (was %zu)\n", arena.used(),
                     before);
        return 1;
    }
    return 0;
}

int run() {
    constexpr std::int32_t kCount = 4096;
    constexpr std::size_t kArenaBytes = 1u << 20;

    DeviceContext context(0);
    DeviceArena arena(kArenaBytes);

    std::vector<float> host_input(kCount);
    ninfer::test::fill_uniform(host_input, /*seed=*/0x51715u, -8.0f, 8.0f);
    ninfer::test::round_to_bf16(host_input); // exactly what the kernel will read

    std::vector<double> reference(kCount);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double x = static_cast<double>(host_input[i]);
        reference[i] = x / (1.0 + std::exp(-x));
    }

    const Tensor input = arena.alloc(DType::BF16, {kCount});
    const Tensor output = arena.alloc(DType::BF16, {kCount});

    int failures = check_tensor_view(input, kCount);
    failures += check_tensor_view(output, kCount);

    std::vector<std::uint16_t> encoded(kCount);
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        encoded[i] = ninfer::test::f32_to_bf16(host_input[i]);
    }
    ninfer::test::cuda_check(cudaMemcpyAsync(input.data, encoded.data(), input.bytes(),
                                             cudaMemcpyHostToDevice, context.stream),
                             "cudaMemcpyAsync host-to-device");

    constexpr std::int32_t kBlock = 256;
    const std::int32_t grid = ninfer::ops::div_up(kCount, kBlock);
    vendor_silu_kernel<<<grid, kBlock, 0, context.stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(output.data),
        kCount);
    ninfer::test::cuda_check_last_launch("vendor_silu_kernel launch");
    context.synchronize();

    const std::vector<double> actual = ninfer::test::from_device_bf16(output.data, kCount);
    failures += ninfer::test::verify_pointwise("vendor substrate: silu over an arena bf16 tensor",
                                               actual, reference, kBf16Criterion);
    failures += check_arena_scope(arena);
    return failures;
}

} // namespace

int main() {
    try {
        // ADR 0006: compute work fails loudly when the GPU is not usable.
        if (ninfer::test::cuda_unavailable()) {
            std::fprintf(stderr,
                         "no CUDA device: the vendored substrate smoke test needs a free GPU "
                         "(stop the reference runner and re-run)\n");
            return 1;
        }
        const int failures = run();
        if (failures != 0) {
            std::fprintf(stderr, "vendor substrate: %d check(s) failed\n", failures);
            return 1;
        }
        std::printf("vendor substrate: ok\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vendor substrate: %s\n", error.what());
        return 1;
    }
}
