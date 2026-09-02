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

/* --------------------------------------------------------------------------
 * Ticket 03: decode step (NVFP4 GEMM + GQA attention).
 *
 * Flat C ABI (ADR 0001): explicit pointers + sizes, a stream handle (null =
 * stream 0), and an int return code (0 = ok, -1 = CUDA error / invalid
 * argument). No C++ types, no shared state across the boundary. All buffer
 * pointers are host memory; the leaf does the device work with internal H2D
 * / D2H copies (consistent with the ticket-01 leaf style).
 * ------------------------------------------------------------------------ */

/* NVFP4 decode GEMM (GEMV path, single token):
 *   out[m] = bias[m] + sum_k x[k] * W[m,k]
 * Weights are NVFP4-quantized: E2M1 codes (2 packed per byte) and a per-group-16
 * E4M3 scale. act is a bf16 activation vector of length k. wt_codes is
 * [m][k/2] bytes; wt_scales is [m][k/16] bytes (E4M3, one byte per group of 16).
 * bias (nullable) and out are bf16, length m. k must be a multiple of 16.
 * stream: null = stream 0. Returns 0 on success, -1 on error. */
int ignis_nvfp4_gemm_decode(const void *act, const void *wt_codes,
                           const void *wt_scales, const void *bias, void *out,
                           int64_t m, int64_t k, void *stream);

/* GQA attention decode step (single token, paged bf16 KV cache):
 *   out[h,d] = attention over seq_len keys for query head h,
 *   kv head = h / (num_q_heads / num_kv_heads).
 * q is bf16 [num_q_heads][head_dim]. kv_cache is bf16, TWO paged planes (K
 * first, V second), each laid out as [num_blocks][block_size][num_kv_heads]
 * [head_dim] (head_dim fastest). block_table is i32 [num_blocks], mapping a
 * logical block to its physical page id. out is bf16 [num_q_heads][head_dim].
 * seq_len must be <= num_blocks * block_size. softmax_scale is 1/sqrt(head_dim)
 * by convention (pass 0.0 to use the default). stream: null = stream 0.
 * Returns 0 on success, -1 on error. */
int ignis_gqa_attention_decode(const void *q, const void *kv_cache,
                               const void *block_table, void *out,
                               int64_t num_q_heads, int64_t num_kv_heads,
                               int64_t head_dim, int64_t seq_len,
                               int64_t block_size, int64_t num_blocks,
                               float softmax_scale, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_KERNEL_H */