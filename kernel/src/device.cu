// ignis kernel leaf - Ticket 04: flat C ABI device-surface (ADR 0001).
//
// Implements the C ABI functions declared in include/ignis_device.h. One
// device handle owns a dedicated non-blocking load stream and a blocking-sync
// event (the reference `DeviceContext` pattern: the blocking-sync event makes
// `synchronize()` sleep instead of CPU-spinning on the stream). Style follows
// the ticket-01 leaf (hello.cu): explicit pointers + sizes, int32 return
// codes (0 = ok, -1 = CUDA error / bad argument), no C++ types across the
// boundary.
//
// NOTE: this module never runs in the default (CPU) build — it is linked only
// when the `cuda` cargo feature is enabled. ADR 0006: the RTX 5090 is held by
// the reference `ninfer-serve`, so real CUDA uploads are gated (the Rust
// `CudaDevice` skips unless `IGNIS_TEST_CUDA=1` and the GPU is free).

#include "ignis_device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

struct ignis_device {
  cudaStream_t load_stream;
  cudaEvent_t blocking_sync;
  int device_id;
};

namespace {

// Log a CUDA error (never aborts from a leaf: the caller decides the -1/NULL
// path). Returns the caller's own return value (kept separate from logging).
template <typename T>
T log_cuda_error(T value, const char *op, cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[ignis-device] CUDA error in %s: %s: %s\n", op,
                 cudaGetErrorName(err), cudaGetErrorString(err));
  }
  return value;
}

// Ensure the handle's device is current (the load stream + event are bound to
// it). Returns true on success, false (already logged) on a CUDA error.
bool ensure_device(struct ignis_device *d) {
  cudaError_t err = cudaSetDevice(d->device_id);
  if (err != cudaSuccess) {
    log_cuda_error(false, "cudaSetDevice", err);
    return false;
  }
  return true;
}

}  // namespace

extern "C" struct ignis_device *ignis_device_create(int device_id) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    log_cuda_error(nullptr, "cudaGetDeviceCount", err);
    return nullptr;
  }
  if (count <= 0) {
    std::fprintf(stderr, "[ignis-device] no CUDA devices available\n");
    return nullptr;
  }
  if (device_id < 0 || device_id >= count) {
    std::fprintf(stderr, "[ignis-device] invalid device id %d (have %d)\n", device_id, count);
    return nullptr;
  }

  err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    log_cuda_error(nullptr, "cudaSetDevice", err);
    return nullptr;
  }

  struct ignis_device *d =
      static_cast<struct ignis_device *>(std::calloc(1, sizeof(struct ignis_device)));
  if (d == nullptr) {
    std::fprintf(stderr, "[ignis-device] allocation failed\n");
    return nullptr;
  }
  d->device_id = device_id;

  // Dedicated non-blocking load stream: keeps the materialization H2D copies
  // off the default (blocking) stream.
  err = cudaStreamCreateWithFlags(&d->load_stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    log_cuda_error(nullptr, "cudaStreamCreateWithFlags(load)", err);
    std::free(d);
    return nullptr;
  }

  // A blocking-sync event makes `synchronize()` sleep instead of spinning a
  // host core (the reference DeviceContext pattern, the fix for the 100% CPU
  // observed during loads).
  err = cudaEventCreateWithFlags(&d->blocking_sync, cudaEventDisableTiming | cudaEventBlockingSync);
  if (err != cudaSuccess) {
    log_cuda_error(nullptr, "cudaEventCreateWithFlags(sync)", err);
    cudaStreamDestroy(d->load_stream);
    std::free(d);
    return nullptr;
  }

  return d;
}

extern "C" int32_t ignis_device_alloc(struct ignis_device *d, uint64_t bytes, void **out_ptr) {
  if (d == nullptr || out_ptr == nullptr) {
    return -1;
  }
  if (!ensure_device(d)) {
    return -1;
  }
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    return log_cuda_error(-1, "cudaMalloc", err);
  }
  *out_ptr = ptr;
  return 0;
}

extern "C" int32_t ignis_device_copy_h2d(struct ignis_device *d, void *dst, const void *src,
                                         uint64_t bytes) {
  if (d == nullptr) {
    return -1;
  }
  if (bytes == 0) {
    return 0;  // empty copy is a no-op (nothing to enqueue)
  }
  if (!ensure_device(d)) {
    return -1;
  }
  cudaError_t err =
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, d->load_stream);
  if (err != cudaSuccess) {
    return log_cuda_error(-1, "cudaMemcpyAsync(H2D)", err);
  }
  return 0;
}

extern "C" int32_t ignis_device_copy_d2h(struct ignis_device *d, void *dst, const void *src,
                                         uint64_t bytes) {
  if (d == nullptr) {
    return -1;
  }
  if (bytes == 0) {
    return 0;  // empty copy is a no-op (nothing to enqueue)
  }
  if (!ensure_device(d)) {
    return -1;
  }
  cudaError_t err =
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, d->load_stream);
  if (err != cudaSuccess) {
    return log_cuda_error(-1, "cudaMemcpyAsync(D2H)", err);
  }
  return 0;
}

extern "C" int32_t ignis_device_sync(struct ignis_device *d) {
  if (d == nullptr) {
    return -1;
  }
  if (!ensure_device(d)) {
    return -1;
  }
  cudaError_t err = cudaEventRecord(d->blocking_sync, d->load_stream);
  if (err != cudaSuccess) {
    return log_cuda_error(-1, "cudaEventRecord", err);
  }
  // The blocking-sync event sleeps (no CPU spin) until the stream drains.
  err = cudaEventSynchronize(d->blocking_sync);
  if (err != cudaSuccess) {
    return log_cuda_error(-1, "cudaEventSynchronize", err);
  }
  return 0;
}

extern "C" int32_t ignis_device_mem_info(struct ignis_device *d, uint64_t *free_bytes,
                                         uint64_t *total_bytes) {
  if (d == nullptr || free_bytes == nullptr || total_bytes == nullptr) {
    return -1;
  }
  if (!ensure_device(d)) {
    return -1;
  }
  size_t free_sz = 0;
  size_t total_sz = 0;
  cudaError_t err = cudaMemGetInfo(&free_sz, &total_sz);
  if (err != cudaSuccess) {
    return log_cuda_error(-1, "cudaMemGetInfo", err);
  }
  *free_bytes = static_cast<uint64_t>(free_sz);
  *total_bytes = static_cast<uint64_t>(total_sz);
  return 0;
}

extern "C" void ignis_device_destroy(struct ignis_device *d) {
  if (d == nullptr) {
    return;
  }
  if (ensure_device(d)) {
    // Drain the load stream before destroying it (frees any pending work).
    cudaStreamSynchronize(d->load_stream);
    cudaEventDestroy(d->blocking_sync);
    cudaStreamDestroy(d->load_stream);
  }
  std::free(d);
}