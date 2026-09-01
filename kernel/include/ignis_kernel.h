/* ignis kernel leaf: flat C ABI surface (ADR 0001).
 * No C++ types, no shared state across the boundary — explicit pointers and
 * sizes only. Rust bindings: crates/core/src/ffi.rs (keep 1:1, tickets 03+).
 */
#ifndef IGNIS_KERNEL_H
#define IGNIS_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ticket 01 smoke test: proves the FFI path end-to-end. */
uint32_t ignis_kernel_hello(void);

/* c[i] = a[i] + b[i] for i in [0, n). Returns 0 on success, -1 on CUDA error.
 * All three pointers must be host memory; the kernel does the device work. */
int ignis_kernel_vector_sum(const float *a, const float *b, float *c, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_KERNEL_H */