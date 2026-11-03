#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Ignis portability patch (ADR 0077, GitHub #456): programmatic dependent launch exists from
// compute capability 9.0. Whether a device below it ignores the programmatic-serialization
// attribute or rejects the launch is untested, so a launch there carries none. The current
// device's capability is read once per process; a device it cannot read launches as before.
[[nodiscard]] inline bool programmatic_launch_available() {
    static const bool available = [] {
        int device = 0;
        int major  = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
            return true;
        }
        return major >= 9;
    }();
    return available;
}

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = programmatic_launch_available() ? 1 : 0;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
// Ignis portability patch (ADR 0077): no instruction below sm_90, where nothing is launched early.
__device__ __forceinline__ void trigger_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
// Ignis portability patch (ADR 0077): no instruction below sm_90, where the stream already orders
// the producer before the consumer.
__device__ __forceinline__ void wait_for_dependencies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

// Entry wait for kernels whose inputs are entirely producer-dependent: block until the producer
// grid's writes are visible before any dependent read. No-op in launches without the
// programmatic-serialization attribute.
__device__ __forceinline__ void sync() { wait_for_dependencies(); }

// Exit publish for producer kernels: announces dependent launches once the calling thread has
// issued its global stores. Only writes issued before a CTA's publish are guaranteed visible to
// a dependent that waited, so publish belongs at kernel end; CTAs that return early without
// publishing still satisfy the launch gate through completion.
__device__ __forceinline__ void publish() { trigger_dependents(); }

} // namespace ninfer::pdl
