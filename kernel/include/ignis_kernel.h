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
 * first, V second), each laid out as [num_blocks][num_kv_heads][block_size]
 * [head_dim] (kv_head-major within a page; head_dim fastest). block_table is i32 [num_blocks], mapping a
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

/* --------------------------------------------------------------------------
 * kernel-abi (tickets 05/06/10): prefill + GDN, pointwise / output path, and
 * eager CUDA-graph capture.
 *
 * Same flat-C-ABI conventions as the decode step above: explicit pointers +
 * sizes, a stream handle (null = stream 0), and an int return code
 * (0 = ok, -1 = CUDA error / invalid argument). No C++ types across the
 * boundary. All buffer pointers are host memory; the leaf does the H2D/D2H
 * copies internally.
 *
 * NOTE (ADR 0006 / 0007): the C-ABI surface + geometry are declared here (and
 * mirrored in crates/core/src/ffi.rs) so the contract is pinned and
 * CPU-verifiable. The ticket-05 kernels (GQA prefill + GDN step,
 * kernel/src/gqa_attention_prefill.cuh, gdn_step.cuh,
 * prefill_gdn_surface.cu) and the ticket-06 kernels (norms / embeddings /
 * greedy sampling, kernel/src/rmsnorm.cuh, embed_gather.cuh, argmax.cuh,
 * norms_sampling_surface.cu) are now implemented and GPU-verified (the
 * crates/core/tests/kernel_abi01_gpu + kernel_abi02_gpu launch them on the
 * GPU even with the model loaded, ADR 0006 nuance). The CUDA-graph (10) .cu
 * and the 99% performance gate (ADR 0007) driven by ignis-bench remain
 * pending.
 * ------------------------------------------------------------------------ */

/* Ticket 05 (kernel-abi-01): GQA prefill attention (batched, multi-token).
 * Attends a batch of queries over their sequences (the prefill path,
 * seq_len > 1). q is bf16 [batch][seq_len][num_q_heads][head_dim]. kv_cache
 * is bf16, two paged planes (K first, V second), each [batch][num_blocks]
 * [num_kv_heads][block_size][head_dim] (kv_head-major within a page; head_dim
 * fastest). block_table is i32
 * [batch][num_blocks] (logical block -> physical page). out is bf16
 * [batch][seq_len][num_q_heads][head_dim]. seq_len must be <=
 * num_blocks * block_size. softmax_scale <= 0 selects the default
 * 1/sqrt(head_dim). stream: null = stream 0. Returns 0 on success, -1 on
 * error. */
int ignis_gqa_attention_prefill(const void *q, const void *kv_cache,
                                const void *block_table, void *out,
                                int64_t batch, int64_t seq_len,
                                int64_t num_q_heads, int64_t num_kv_heads,
                                int64_t head_dim, int64_t block_size,
                                int64_t num_blocks, float softmax_scale,
                                void *stream);

/* Ticket 05 (kernel-abi-01): GDN (linear-attention) recurrent step, batched.
 * Updates the per-request recurrent state of the linear-attention (GDN)
 * layers. x is bf16 [batch][state_dim] (the current-step input feature).
 * state_in / state_out are bf16
 * [batch][num_gdn_layers][state_rows][state_cols] (the carried-forward
 * recurrent state; state_out receives the updated state, state_in may alias
 * state_out). Returns 0 on success, -1 on error. */
int ignis_gdn_step(const void *x, const void *state_in, void *state_out,
                   int64_t batch, int64_t num_gdn_layers, int64_t state_rows,
                   int64_t state_cols, int64_t state_dim, void *stream);

/* Ticket 06 (kernel-abi-02): RMSNorm (or LayerNorm when `center` is
 * non-null). out = x / rms(x) * weight, optionally centered first. x is bf16
 * [n]. weight (nullable): bf16 [n]. center (nullable): bf16 [n] (present =>
 * LayerNorm, absent => RMSNorm). out: bf16 [n]. eps: numerical epsilon
 * (<= 0 selects 1e-6). Returns 0 on success, -1 on error. */
int ignis_rmsnorm(const void *x, const void *weight, const void *center,
                  void *out, int64_t n, float eps, void *stream);

/* Ticket 06 (kernel-abi-02): embedding lookup. out[row] = table[id[row]].
 * table: bf16 [vocab][hidden]. id: i32 [batch]. out: bf16 [batch][hidden].
 * id values must be in [0, vocab). Returns 0 on success, -1 on error. */
int ignis_embedding(const void *table, const void *id, void *out,
                    int64_t batch, int64_t vocab, int64_t hidden,
                    void *stream);

/* Ticket 06 (kernel-abi-02): greedy sampling. out[i] = argmax over
 * logits[i]. logits: f32 [batch][vocab]. out: i32 [batch]. Ties resolve to
 * the lowest index (deterministic — the v1 correctness floor, ADR 0007:
 * greedy + fixed seed). Returns 0 on success, -1 on error. */
int ignis_greedy_sample(const void *logits, void *out, int64_t batch,
                        int64_t vocab, void *stream);

/* Ticket 10 (kernel-abi-03): eager CUDA-graph capture at startup. One opaque
 * graph handle, captured at startup and replayed on each step (v1 decision;
 * lazy capture is a later optimization). */
struct ignis_graph;

/* Begin a CUDA graph capture on `stream` (null = stream 0). The caller
 * issues the prefill/decode kernel launches (the entry points above) while
 * the capture is active, then calls ignis_graph_end_capture to materialize
 * the graph. Returns 0 on success, -1 on error. */
int ignis_graph_begin_capture(void *stream);

/* End the capture, materializing the graph into *out (a graph-executable).
 * Returns 0 on success, -1 on error. */
int ignis_graph_end_capture(void *stream, struct ignis_graph **out);

/* Launch a captured graph on `stream`. Returns 0 on success, -1 on error. */
int ignis_graph_launch(struct ignis_graph *g, void *stream);

/* Destroy a captured graph. NULL is a no-op. */
void ignis_graph_destroy(struct ignis_graph *g);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_KERNEL_H */