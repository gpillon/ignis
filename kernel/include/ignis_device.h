/* ignis kernel leaf: device-surface flat C ABI (ADR 0001).
 *
 * One device handle owns a dedicated non-blocking load stream and a
 * blocking-sync event (the reference `DeviceContext` pattern: a
 * blocking-sync event makes `synchronize()` sleep instead of CPU-spinning on
 * the stream). Flat C: explicit pointers + sizes, `int32_t` return codes
 * (0 = ok, -1 = CUDA error / bad argument), no C++ types across the boundary.
 *
 * Rust bindings: crates/artifact/src/ffi.rs (keep 1:1, ticket 04).
 */
#ifndef IGNIS_DEVICE_H
#define IGNIS_DEVICE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque device context (load stream + blocking-sync event). Allocated by
 * ignis_device_create, destroyed by ignis_device_destroy. Never dereferenced
 * across the boundary. */
struct ignis_device;

/* Create a device context: a non-blocking load stream + a blocking-sync
 * event. `device_id` selects the GPU (must be a valid id). Returns NULL on a
 * CUDA error (driver missing, no devices, or a bad device id). */
struct ignis_device *ignis_device_create(int device_id);

/* Allocate `bytes` of device memory; `*out_ptr` receives the device pointer.
 * Returns 0 on success, -1 on a CUDA error or a bad argument. */
int32_t ignis_device_alloc(struct ignis_device *d, uint64_t bytes, void **out_ptr);

/* Enqueue a host->device copy on the load stream (asynchronous: call
 * ignis_device_sync before reading the destination). Returns 0 on success,
 * -1 on a CUDA error or a bad argument. A zero-length copy is a no-op. */
int32_t ignis_device_copy_h2d(struct ignis_device *d, void *dst, const void *src,
                              uint64_t bytes);

/* Enqueue a device->host copy on the load stream (asynchronous). Returns 0 on
 * success, -1 on a CUDA error or a bad argument. A zero-length copy is a
 * no-op. */
int32_t ignis_device_copy_d2h(struct ignis_device *d, void *dst, const void *src,
                              uint64_t bytes);

/* Block until the load stream is idle. The blocking-sync event sleeps instead
 * of spinning the CPU. Returns 0 on success, -1 on a CUDA error. */
int32_t ignis_device_sync(struct ignis_device *d);

/* GitHub #341: enqueue ignis_fp8_quantize_rows (ignis_fp8_linear.h) on the load
 * stream -- a BF16 weight `w` [rows][cols] already on the device to the FP8
 * row-scale payload at `payload`, in stream order after the copies before it.
 * Returns 0 on success, -1 on a bad argument or a CUDA error. */
int32_t ignis_device_fp8_quantize_rows(struct ignis_device *d, const void *w, uint32_t rows,
                                       uint32_t cols, void *payload);

/* Free / total device memory in bytes (cudaMemGetInfo). Returns 0 on success,
 * -1 on a CUDA error or a bad argument. */
int32_t ignis_device_mem_info(struct ignis_device *d, uint64_t *free_bytes,
                              uint64_t *total_bytes);

/* The CUDA version the installed NVIDIA driver supports, as
 * cudaDriverGetVersion reports it (13010 for CUDA 13.1), into `*version`: 0
 * when the machine has no driver, no libcuda.so.1 / nvcuda.dll for the CUDA
 * runtime to load (GitHub #329). The leaf links no driver library, so the
 * binary starts without one; this is how a GPU path asks first, and names a
 * missing driver as such. Creates no context and needs no ignis_device.
 * Returns 0 on success, -1 on a CUDA error (logged to stderr) or a NULL
 * argument. */
int32_t ignis_device_driver_version(int32_t *version);

/* The CUDA version of the runtime this library was built with, as
 * cudaRuntimeGetVersion reports it (13010 for CUDA 13.1), into `*version`:
 * the figure a GPU path compares the driver's version against, to name a
 * driver too old for the build. Creates no context. Returns 0 on success,
 * -1 on a CUDA error (logged to stderr) or a NULL argument. */
int32_t ignis_device_runtime_version(int32_t *version);

/* Free / total memory of CUDA device `device_id` as NVML reports it -- the
 * figure nvidia-smi and Task Manager show -- read without creating a CUDA
 * context (GitHub #210). A VRAM budget is derived from this rather than from
 * ignis_device_mem_info: on Windows WDDM cudaMemGetInfo does not count the
 * calling process's own context and reads about 400 MiB more free memory
 * than the adapter has. NVML (libnvidia-ml.so.1, nvml.dll) is opened on the
 * first call, not linked (GitHub #329). Needs no ignis_device. Returns 0 on
 * success, -1 on a CUDA or NVML error (logged to stderr), on NVML that does
 * not load, or on a bad argument. */
int32_t ignis_device_nvml_mem_info(int device_id, uint64_t *free_bytes, uint64_t *total_bytes);

/* The name of CUDA device `device_id` as NVML reports it ("NVIDIA GeForce RTX
 * 5090"), NUL-terminated into `name` (`name_bytes` long, truncated to fit),
 * and its compute capability as the CUDA runtime reports it, into `*major`
 * and `*minor` (12 and 0 on an RTX 5090): what the start checks against
 * ignis_build_cuda_architectures (GitHub #457). Creates no context and needs
 * no ignis_device. Returns 0 on success, -1 on a CUDA or NVML error (logged
 * to stderr), on NVML that does not load, or on a bad argument. */
int32_t ignis_device_identity(int device_id, char *name, uint32_t name_bytes, int32_t *major, int32_t *minor);

/* The CUDA architectures this leaf's kernels were compiled for, as the build
 * named them (CMake's CUDA_ARCHITECTURES: "120a", "86;120a"): a static
 * NUL-terminated string. A card's kernels run on it when one entry's machine
 * code does (GitHub #457). */
const char *ignis_build_cuda_architectures(void);

/* Free a device allocation returned by ignis_device_alloc. NULL device or
 * pointer is a no-op. */
void ignis_device_free(struct ignis_device *d, void *ptr);

/* Allocate `bytes` of mapped pinned host memory (ADR 0062): one
 * cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable), never a
 * registration of pageable memory. `*out_host` receives the page-aligned host
 * pointer the loader writes the bytes through, `*out_device` the pointer a
 * kernel reads them through, in place over the bus. Not counted by the
 * allocation counter (ignis_seq.h): it is made once, at load. Returns 0 on
 * success, -1 on a CUDA error (the host cannot pin that much), a zero size or
 * a bad argument. */
int32_t ignis_device_host_alloc_mapped(struct ignis_device *d, uint64_t bytes, void **out_host,
                                       void **out_device);

/* Free a host allocation returned by ignis_device_host_alloc_mapped, by its
 * host pointer. NULL device or pointer is a no-op. */
void ignis_device_host_free(struct ignis_device *d, void *host);

/* Destroy the context: drain the load stream, then free the event and stream,
 * then release the handle. NULL is a no-op. */
void ignis_device_destroy(struct ignis_device *d);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_DEVICE_H */