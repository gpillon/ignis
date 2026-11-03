// ignis kernel leaf: programmatic dependent launch in our own kernels (GitHub #456).
//
// The griddepcontrol instructions exist from sm_90. Below it they compile to nothing, and a
// launch carries no programmatic-serialization attribute, since whether a device below compute
// capability 9.0 ignores the attribute or rejects the launch is untested: the stream then orders
// the producer before its consumer, as it does any two launches. The vendored kernels get the
// same through the recorded patch of kernel/vendor/src/core/pdl.cuh (ADR 0077).
#ifndef IGNIS_PDL_LAUNCH_CUH
#define IGNIS_PDL_LAUNCH_CUH

#include <cuda_runtime.h>

namespace ignis::pdl {

// Wait for the producer grid's writes: a consumer's first step before it reads them.
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Let the dependent grid launch.
__device__ __forceinline__ void launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Whether a launch on the current device may carry the programmatic-serialization attribute:
// from compute capability 9.0. Read once per process; a device it cannot read launches as before.
inline bool available() {
  static const bool available = [] {
    int device = 0;
    int major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
      return true;
    }
    return major >= 9;
  }();
  return available;
}

}  // namespace ignis::pdl

#endif  // IGNIS_PDL_LAUNCH_CUH
