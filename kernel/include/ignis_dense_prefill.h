/* ignis kernel leaf: the dense prefill of the 27B's trellis body -- OURS (GitHub #509, spec #503
 * "The dense prefill route"), ported from the prototype on the branch
 * prototype/sm8x-dense-prefill (its bench, phases 2a and 2c, levers A and D), written from the
 * PTX ISA and our own kernels (kernel/src/trellis_linear_*, trellis_decode.cuh, moe_common.cuh).
 * No library GEMM.
 *
 * The fused prefill (ignis_trellis_linear.h) decodes the trellis tiles inside its GEMM's inner
 * loop, on fp16 tensor cores. The dense prefill runs a body matrix in three steps instead:
 *   decode    the matrix, once per prefill chunk, into a dense INT8 copy W8 (one buffer the size
 *             of the largest matrix serves every matrix, one at a time);
 *   quantize  the activations, rotated by the matrix's own suh and the 128-wide Hadamard, to INT8
 *             with one scale per token (A8); one launch serves every matrix that reads the same
 *             input, each with its own suh (the artifact never shares one);
 *   GEMM      mma.sync m16n8k32 s8 x s8 into int32 over the whole K, exact, then an fp32 epilogue
 *             on the CUDA cores: the scales and the offset's rank-1 term, the output Hadamard,
 *             svh, and the BF16 store, residual add or SwiGLU the fused prefill writes. The tile
 *             is 256 or 128 tokens x 128 outputs, the main loop 3 or 4 cp.async stages with the
 *             fragments double-buffered across each stage boundary; the tile height, the depth
 *             and the raster order are chosen per shape (ignis_dense_prefill_prepare's occupancy
 *             and SM count, the call's shape). Every kernel fits 99 KiB of shared memory per
 *             block, so the same code runs on compute capability 8.6, 8.9 and 12.0.
 *
 * The numerics are the prototype's, which the KLD gate measured (spec #503):
 *   W8   q_w = clamp(round_half_even((s - 510) / 4), -127, 127), s in [0, 1020] a tile step's
 *        codebook integer (the byte sum of state * 0x83DCD12D, ignis_trellis_linear.h's format).
 *        The trellis weight at s' = 4 q_w + 510 is exactly sw q_w + off, with sw = 4 c1 =
 *        887 / 2^15 and off = 1534 c1 + c0 = -119 / 2^16 (c1 = fp16 0x1EEE, c0 = fp16 0xC931),
 *        both exact in fp32: the dense copy carries one scale and one constant offset.
 *   A8   per token t: v = (x[t] o suh) H in fp32 -- each product rounded once, the Hadamard's
 *        butterfly in a fixed order, then times 1 / sqrt(128) -- amax[t] = max |v| over the whole
 *        K, q_a = clamp(rn(v * (127 / amax)), -127, 127) (ties to even), scale[t] = amax *
 *        (1 / 127), sum[t] = sum over K of q_a. A row whose amax is 0 or not finite quantizes to
 *        zeros with scale 0.
 *   acc  acc[t][n] = sum_k q_a[t][k] q_w[k][n], int32, exact (|acc| < 127^2 in < 2^31 for every
 *        in up to 2^17).
 *   y    y[t] = ((acc[t] scale[t] sw + off scale[t] sum[t]) H) o svh, the first term one fmaf
 *        per output in fp32, then the epilogue. The off term is the dense copy's zero point: an
 *        INT8 GEMM that left it out would not compute what the gate measured.
 *
 * Tolerance, against the fp64 value Y of `y` computed from the int32 accumulators (kernel/tests/
 * test_dense_prefill.cu holds every matrix to it): |y - Y| <= 2^-8 |Y| + 2^-9 s, s being |svh|
 * times the root mean square over the output's 128-column block of sqrt(sum_k (a_k w_kj)^2),
 * a_k = scale q_a and w_kj = sw q_w + off -- the BF16 output's rounding and the epilogue's fp32
 * steps; the accumulators themselves are exact.
 *
 * Layouts:
 * - The dense copy of a matrix [in][out]: blocks of 128 outputs x 64 inputs, 8 KiB each, [n][k]
 *   inside; block (nb, kb) at byte (nb * in / 64 + kb) * 8192. in * out bytes.
 * - The quantized activations of one matrix: q int8 [in / 64][tokens][64] (64-input slabs, so a
 *   GEMM stage's activations are one contiguous run), scale fp32 [tokens], sum int32 [tokens].
 * - x and y as ignis_trellis_linear.h: token-major BF16, x [tokens][in] 16-byte aligned, y
 *   [tokens][ldy] 8-byte aligned with ldy a multiple of 4.
 *
 * Every call enqueues on `stream`, allocates nothing, never synchronizes the host and is
 * capturable in a CUDA graph; results are deterministic (no atomics: one CTA per output tile).
 * `in` and `out` are positive multiples of 128. Return 0, or -1 with the reason in
 * ignis_trellis_last_error() (ignis_trellis_linear.h): the two ops share the error channel.
 */
#ifndef IGNIS_DENSE_PREFILL_H
#define IGNIS_DENSE_PREFILL_H

#include "ignis_trellis_linear.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The matrices one ignis_dense_prefill_quantize launch serves: the body's widest shared input is
 * the GQA layer's q, k and v. */
#define IGNIS_DENSE_PREFILL_MAX_GROUP 3

/* The GEMM's epilogues: IGNIS_TRELLIS_STORE and IGNIS_TRELLIS_ADD as ignis_trellis_linear, and
 * SwiGLU's up pass, y[t][c] = BF16(silu(y[t][c]) * v), y holding the gate pass's BF16 output:
 * the fused prefill's SwiGLU, which stores the gate in BF16 between its two passes. */
#define IGNIS_DENSE_PREFILL_SWIGLU_UP 3

/* One matrix's quantized activations (A8), device pointers; see the layouts above. */
typedef struct ignis_dense_prefill_activations {
  void *q;       /* int8 [in / 64][tokens][64], 16-byte aligned */
  float *scale;  /* [tokens] */
  int32_t *sum;  /* [tokens] */
} ignis_dense_prefill_activations;

/* Where a GEMM writes. `split` 0 writes y[t][c]. Nonzero, the output is heads of 2 split
 * columns, [first half | second half] per head (the GQA q_proj's [query | gate] rows): column c of
 * head h = c / (2 split) lands in `y` at h split + c mod (2 split) when that is below split, and
 * in `y2` at h split + c mod (2 split) - split otherwise, both [tokens][ldy]. split is then a
 * multiple of 128 and out a multiple of 2 split. */
typedef struct ignis_dense_prefill_output {
  void *y;
  uint32_t ldy;
  uint32_t epilogue; /* IGNIS_TRELLIS_STORE, IGNIS_TRELLIS_ADD or IGNIS_DENSE_PREFILL_SWIGLU_UP */
  uint32_t split;
  uint32_t reserved; /* 0 */
  void *y2;          /* NULL unless split */
} ignis_dense_prefill_output;

/* Prepares the current device for every op here: the GEMM's shared-memory attributes, its
 * resident CTAs per SM (the occupancy query) and the SM count the tile choice reads. Call once per
 * device at load, outside any stream capture; the ops refuse an unprepared device. Thread-safe; a
 * failure is not remembered. Independent of ignis_trellis_prepare. */
int32_t ignis_dense_prefill_prepare(void);

/* The dense copy's bytes for a matrix of (in, out): in * out, 0 for a shape the ops refuse. */
uint64_t ignis_dense_prefill_weight_bytes(uint32_t in, uint32_t out);

/* The ignis_trellis_workspace bytes an ignis_dense_prefill_linear call of `matrices` matrices of
 * `in` inputs needs at `tokens`: the fused op's counters (left untouched, so one workspace serves
 * both ops on one stream) followed by each matrix's quantized activations. 0 for a call the op
 * refuses. */
uint64_t ignis_dense_prefill_workspace_bytes(uint32_t in, uint32_t matrices, uint32_t tokens);

/* W8 of `w` into `dense` (at least ignis_dense_prefill_weight_bytes, 16-byte aligned). */
int32_t ignis_dense_prefill_decode(const ignis_trellis_weight *w, void *dense, uint64_t dense_bytes,
                                   void *stream);

/* A8 of x [tokens][in] for each of `count` (1..IGNIS_DENSE_PREFILL_MAX_GROUP) matrices of the same
 * `in`, with matrix i's suh into out[i]. Reads x once for every matrix of in <= 6144; a wider
 * input (the MLP's down) takes two passes over it per matrix. */
int32_t ignis_dense_prefill_quantize(const void *x, uint32_t tokens, const ignis_trellis_weight *const *w,
                                     uint32_t count, const ignis_dense_prefill_activations *out, void *stream);

/* y = the epilogue of `dense` (W8 of `w`, whose in, out and svh it reads) times `a`. */
int32_t ignis_dense_prefill_gemm(const ignis_trellis_weight *w, const void *dense,
                                 const ignis_dense_prefill_activations *a, uint32_t tokens,
                                 const ignis_dense_prefill_output *out, void *stream);

/* The GEMM's int32 accumulators alone, acc[t][n] at acc + t * ldacc + n (ldacc >= out), from
 * the same kernel and tile choice as ignis_dense_prefill_gemm. Not on the serving path: exposed
 * so the test can hold the integer MMA to an exact reference. */
int32_t ignis_dense_prefill_accumulate(const ignis_trellis_weight *w, const void *dense,
                                       const ignis_dense_prefill_activations *a, uint32_t tokens,
                                       int32_t *acc, uint32_t ldacc, void *stream);

/* The three steps for `count` (1..IGNIS_DENSE_PREFILL_MAX_GROUP) matrices that read the same x,
 * in order: one quantize for all of them into `workspace` (ignis_dense_prefill_workspace_bytes),
 * then for each matrix its decode into `dense` and its GEMM into out[i]. The MLP's SwiGLU is
 * {gate, up} with out[0] the gate's STORE into h and out[1] the up's IGNIS_DENSE_PREFILL_SWIGLU_UP into
 * the same h. */
int32_t ignis_dense_prefill_linear(const ignis_trellis_weight *const *w, uint32_t count, const void *x,
                                   uint32_t tokens, const ignis_dense_prefill_output *out,
                                   const ignis_trellis_workspace *workspace, void *dense, uint64_t dense_bytes,
                                   void *stream);

/* ---- A8 fused into its producers (GitHub #511, spec #503 "A8") ----------------------------------
 * Where A8's input has just been produced by an RMSnorm, A8 is no pass of its own: the RMSnorm
 * before a block's first matrices writes their activations as it normalizes the row. The
 * activations are the standalone A8's of the same BF16 row bit for bit
 * (ignis_dense_prefill_quantize: the same products, the same rounding); only the passes change.
 * Down's A8 stays ignis_dense_prefill_quantize's: fused into gate+up's SwiGLU epilogue (each
 * block's max kept there, then a light second pass) it measured slower than the standalone pass
 * on the RTX 5090 and on an sm_86 card. */

/* The RMSnorm with a unit offset, h = x rsqrt(mean(x^2) + eps) (1 + norm), of x [tokens][in] BF16
 * by norm [in] BF16, bit for bit the vendored rmsnorm's (ninfer::ops::rmsnorm, unit_offset) at the
 * widths whose wide-row form this restates: in a multiple of 1024 in (3072, 8192]. h [tokens][in]
 * is written when non-NULL, and for each of `count` (1..IGNIS_DENSE_PREFILL_MAX_GROUP) matrices of
 * that `in`, matrix i's activations into out[i], bit for bit ignis_dense_prefill_quantize's of h.
 * x, norm and h 4-byte aligned; eps positive and finite. */
int32_t ignis_dense_prefill_rmsnorm_quantize(const void *x, const void *norm, float eps, uint32_t tokens, void *h,
                                             const ignis_trellis_weight *const *w, uint32_t count,
                                             const ignis_dense_prefill_activations *out, void *stream);

/* ignis_dense_prefill_linear for matrices whose input is the RMSnorm of x: the RMSnorm with their
 * A8 (ignis_dense_prefill_rmsnorm_quantize, h written when non-NULL) into `workspace`
 * (ignis_dense_prefill_workspace_bytes of the same group), then each matrix's decode and GEMM. The
 * MLP's gate and up are such a group, with the outputs of ignis_dense_prefill_linear's SwiGLU. */
int32_t ignis_dense_prefill_norm_linear(const ignis_trellis_weight *const *w, uint32_t count, const void *x,
                                        const void *norm, float eps, uint32_t tokens, void *h,
                                        const ignis_dense_prefill_output *out, const ignis_trellis_workspace *workspace,
                                        void *dense, uint64_t dense_bytes, void *stream);

/* Test-only (GitHub #511): 0 sends the trellis body's dense prefill back to the standalone A8 of
 * #509 -- the RMSnorms write their BF16 rows and ignis_dense_prefill_quantize reads them -- so a
 * test can hold the fused producers' output to it; nonzero, the default, fuses them.
 * Process-wide, read at each body call: set it between prefills, never during one. The count of
 * fused RMSnorm launches this process has enqueued tells such a test which passes ran. */
void ignis_dense_prefill_set_fused_producers(uint32_t fused);
uint32_t ignis_dense_prefill_fused_producers(void);
uint64_t ignis_dense_prefill_fused_norm_launches(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_DENSE_PREFILL_H */
