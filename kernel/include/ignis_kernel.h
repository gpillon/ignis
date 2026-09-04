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
 * GPU even with the model loaded, ADR 0006 nuance). The ticket-10 CUDA-graph
 * capture code (kernel/src/graph_capture.cu: the ignis_graph_* primitives +
 * the ignis_graph_startup_check startup verification) is now implemented —
 * the capture run is GPU-gated and self-skips (crates/core/tests/
 * kernel_abi03_gpu, ADR 0006); the 99% performance gate (ADR 0007) driven
 * by ignis-bench remains pending (ticket 20).
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

/* Ticket 22 (kernel-abi 05, GitHub #22): multi-token NVFP4 GEMM (the prefill
 * / FFN-projection path):
 *   out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]
 * `act` is bf16 [tokens][k]. Weights are NVFP4-quantized: E2M1 codes (2 packed
 * per byte) [m][k/2] and a per-group-16 E4M3 scale [m][k/16]. `bias`
 * (nullable) is bf16 [m]; `out` is bf16 [tokens][m]. `k` must be a multiple
 * of 16 (the NVFP4 group scale); `m` and `tokens` must be positive. The
 * rowsplit tiling (rows-of-W x tokens, fp32 FMA accumulation, no tensor cores
 * / no cuBLASLt) is a temporary starting point per ADR 0005; the tensor-core
 * W4A4 MMA is the later performance-gate material (ADR 0007). stream: null =
 * stream 0. Returns 0 on success, -1 on error. */
int ignis_nvfp4_gemm_prefill(const void *act, const void *wt_codes,
                            const void *wt_scales, const void *bias, void *out,
                            int64_t tokens, int64_t m, int64_t k, void *stream);

/* Ticket 26 (GitHub #26, the compute-adapter's production path): NVFP4 GEMM /
 * GEMV with DEVICE-RESIDENT weights. `wt_codes` / `wt_scales` are DEVICE
 * pointers (the artifact's materialized arena, ADR 0002); the leaf does NOT
 * H2D them (the #26 fix: the 19 GB of weights stay in VRAM, no per-call H2D).
 * `act` (host bf16) and `out` (host bf16) are H2D/D2H'd (small); `bias`
 * (nullable, host bf16) is H2D'd. The kernel runs on the current CUDA device
 * (device 0 in the single-GPU v1; the caller sets it via the artifact
 * CudaDevice). `k` must be a multiple of 16 (the NVFP4 group scale); `m`, `k`
 * (decode) / `tokens`, `m`, `k` (prefill) must be positive. stream: null =
 * stream 0. Returns 0 on success, -1 on error. */
int ignis_nvfp4_gemm_decode_device(const void *act, const void *wt_codes,
                                   const void *wt_scales, const void *bias,
                                   void *out, int64_t m, int64_t k, void *stream);

int ignis_nvfp4_gemm_prefill_device(const void *act, const void *wt_codes,
                                    const void *wt_scales, const void *bias,
                                    void *out, int64_t tokens, int64_t m,
                                    int64_t k, void *stream);

/* Ticket 29 (kernel-abi 10, GitHub #29): bf16 GEMM (the logits path for the
 * W8-dequantized lm_head — the A1 artifact dequant produces a bf16 lm_head
 * weight, which the NVFP4 GEMM surface (kernel-abi 01/05) cannot consume;
 * this is the third 27B-fidelity kernel):
 *   out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]
 * `act` is bf16 [tokens][k]; `wt` is bf16 [m][k] (the W8-dequantized
 * lm_head); `bias` (nullable) is bf16 [m]; `out` is bf16 [tokens][m]. A
 * rowsplit FMA GEMM (no tensor cores / no cuBLASLt, ADR 0001/0005); the
 * tensor-core MMA is the later performance-gate material (ADR 0005/0007).
 * `tokens == 1` is the GEMV special case (the decode logits path);
 * `tokens > 1` serves the batched-prefill logits path (B1, kernel-abi 08).
 * `m`, `k` and `tokens` must be positive (no alignment constraint — plain
 * bf16 planes, no NVFP4 group scales). stream: null = stream 0. Returns 0
 * on success, -1 on error. */
int ignis_bf16_gemm(const void *act, const void *wt, const void *bias,
                    void *out, int64_t tokens, int64_t m, int64_t k,
                    void *stream);

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

/* --------------------------------------------------------------------------
 * kernel-abi 06 (ticket 28, GitHub #28): GDN causal conv + GQA RoPE.
 *
 * The two kernel ops the full-correct 27B forward (A3) needs that the
 * kernel-abi 01-05 surface lacks, 1:1 ports of the proven reference
 * kernels (ADR 0005; provenance via kernel/NOTICE), following the
 * existing flat-C-ABI conventions (ADR 0001): explicit pointers + sizes,
 * a stream handle (null = stream 0), and an int return code (0 = ok,
 * -1 = CUDA error / invalid argument). All buffer pointers are host
 * memory; the leaf does the H2D/D2H copies internally.
 *
 * NOTE: the v1 does the norms + RoPE as two steps (the ignis_rmsnorm,
 * kernel-abi 02, then the ignis_rope_qk below) — the fused qk_norm_rope
 * kernel is a later performance item (ADR 0005); the bf16 logits GEMM
 * (the A2b ticket, kernel-abi 10) is a separate ABI surface.
 * ------------------------------------------------------------------------ */

/* (1) The GDN causal conv. `projected` is bf16 [tokens][channels] (the
 * GEMM output over the query+key+value rows); `conv_weight` is bf16
 * [4][channels] (the 4 taps w0..w3, tap-major — the artifact's
 * `gdn/convolution` tensor {4, channels}); `state_in` / `state_out` are
 * bf16 [channels][3] (the 3-tap rolling conv state s0,s1,s2 per channel,
 * channel-major; `state_out` receives the updated state — the last 3
 * consumed taps — and `state_in` may alias `state_out`); `out` is bf16
 * [tokens][channels] (the conv'd + SiLU'd q/k/v). The z-row contract:
 * `channels` covers q+k+v only (NVFP4 10240 = 2048q+2048k+6144v in the
 * full model); the `z` rows are a separate GEMM output assembled by the
 * caller and never pass through the conv (they bypass it entirely — no
 * z-size parameter exists in this ABI, and a caller feeding a
 * z-inclusive width would convolve those rows). One thread per channel:
 * the 3-tap rolling state s0,s1,s2 + the current tap w3·p, the SiLU
 * epilogue. stream: null = stream 0. Returns 0 on success, -1 on error. */
int ignis_gdn_causal_conv(const void *projected, const void *conv_weight,
                          const void *state_in, void *state_out, void *out,
                          int64_t tokens, int64_t channels, void *stream);

/* (2) The GQA RoPE (split-half NeoX). Rotates the first `rotary_dim` dims
 * (of `head_dim`) of each Q and K head, in-place: for a pair
 * (a = x[p], b = x[p + rotary_dim/2]), out[p] = a·cos − b·sin,
 * out[p + rotary_dim/2] = b·cos + a·sin (cos/sin = sincosf(pos ·
 * inv_freq[p]), the fp32 unscaled route — the reference's
 * attention_factor 1.0 bit-stable path; v1 is unscaled, factor 1.0).
 * `q` is bf16 [batch][seq][num_q_heads][head_dim]; `k` is bf16
 * [batch][seq][num_kv_heads][head_dim]; `inv_freq` is fp32 [rotary_dim/2]
 * (the per-pair frequencies — the reference's `rope_linear_frequencies`
 * table, θ^(-2p/rotary_dim); θ = 1e7, rotary_dim = 64 of head_dim = 256
 * (32 pairs) in the Qwen 3.8-27B GQA geometry; the table is computed
 * once at construction, host-side, a deterministic table — a non-goal is
 * the per-step table recompute). The un-rotated dims [rotary_dim,
 * head_dim) are never written. The pos contract: `pos` is a single
 * uniform position for the whole call (every (batch, seq) token rotates
 * at `pos`); a multi-token prefill must therefore invoke the kernel per
 * token (seq = 1) — a per-token `positions` array or a `pos_base + t`
 * mode would be an ABI extension, not current behaviour. stream: null =
 * stream 0. Returns 0 on success, -1 on error. */
int ignis_rope_qk(void *q, void *k, const void *inv_freq, int64_t batch,
                  int64_t seq, int64_t num_q_heads, int64_t num_kv_heads,
                  int64_t head_dim, int64_t rotary_dim, int32_t pos,
                  void *stream);

/* Ticket 10 (kernel-abi-03): eager CUDA-graph capture at startup. One
 * opaque graph handle, captured at startup and replayed on each step (v1
 * decision; lazy capture is a later optimization, design §1).
 *
 * Capture-stream note: a CUDA graph cannot be captured on the legacy
 * default stream. When a null stream is passed to ignis_graph_begin_capture,
 * the leaf creates a non-blocking capture stream owned by the graph handle
 * (destroyed by ignis_graph_destroy); a non-null stream is the caller's
 * (it must not be the legacy default stream, which cannot be captured, and
 * the leaf does not own it).
 * A null stream to ignis_graph_launch selects the graph's own capture
 * stream (the legacy default stream is avoided for graph launches). */
struct ignis_graph;

/* Begin a CUDA graph capture on `stream` (null = a leaf-owned non-blocking
 * stream — the legacy default stream cannot be captured). The caller
 * issues the prefill/decode kernel launches (the entry points above, or
 * raw kernel launches on this stream) while the capture is active, then
 * calls ignis_graph_end_capture to materialize the graph. One capture at a
 * time (v1 startup capture is single-shot; the launch happens on the
 * capturing thread, thread-local capture mode). Returns 0 on success,
 * -1 on error (a capture already in progress, a stream mismatch, or a
 * CUDA error — e.g. no GPU, the caller self-skips, ADR 0006). */
int ignis_graph_begin_capture(void *stream);

/* End the capture, materializing the graph into *out (a graph-executable).
 * `stream` must match the stream passed to ignis_graph_begin_capture
 * (null = the leaf-owned stream). Returns 0 on success, -1 on error (no
 * active capture, a stream mismatch, or a CUDA error). */
int ignis_graph_end_capture(void *stream, struct ignis_graph **out);

/* Launch a captured graph on `stream` (null = the graph's own capture
 * stream — the legacy default stream is avoided for graph launches).
 * Returns 0 on success, -1 on error (a null graph handle is a clean -1,
 * before any CUDA call). */
int ignis_graph_launch(struct ignis_graph *g, void *stream);

/* Destroy a captured graph (and, when the leaf created the capture stream,
 * the stream). NULL is a no-op (no CUDA calls). */
void ignis_graph_destroy(struct ignis_graph *g);

/* Ticket 10 (kernel-abi-03): the startup verification. Captures a
 * representative prefill + decode kernel sequence (GQA prefill attention +
 * GDN step + GQA decode attention, the per-step structure; a few KB of
 * VRAM — runs even with the model loaded, the ADR 0006 nuance) into a CUDA
 * graph, runs the same sequence eagerly and via graph replay, and confirms
 * the replayed outputs match the eager outputs bit-exactly. The canary-
 * suite 99% performance gate (ADR 0007) is driven by ignis-bench
 * (ticket 20), not here. stream: null = stream 0 for the eager phase (the
 * capture itself runs on the leaf-owned non-blocking stream). Returns 0 if
 * the capture verified and replay matches eager, -1 on a CUDA error (GPU
 * unavailable / busy — the caller self-skips, ADR 0006), -2 if the capture
 * succeeded but the replayed result diverged from the eager result (a real
 * failure — the graph path is broken; not a skip condition). */
int ignis_graph_startup_check(void *stream);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_KERNEL_H */