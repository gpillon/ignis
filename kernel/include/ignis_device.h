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

/* Free / total device memory in bytes (cudaMemGetInfo). Returns 0 on success,
 * -1 on a CUDA error or a bad argument. */
int32_t ignis_device_mem_info(struct ignis_device *d, uint64_t *free_bytes,
                              uint64_t *total_bytes);

/* Destroy the context: drain the load stream, then free the event and stream,
 * then release the handle. NULL is a no-op. */
void ignis_device_destroy(struct ignis_device *d);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_DEVICE_H */