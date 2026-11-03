/* ignis kernel leaf: the dense trellis linear -- OURS (spec 27b-16gb/02, GitHub #337; ADR 0060,
 * ADR 0051), built on our own Flash-Next trellis decode (kernel/src/trellis_decode.cuh,
 * spec flash-next/02). No exllamav3 or QTIP kernel source is read or copied: exllamav3's
 * `reconstruct` is the oracle the tile decode is held to (kernel/tests/fixtures/trellis_dense/),
 * nothing more.
 *
 * A trellis weight is one EXL3 tensor of the 27B's 16 GB artifact (ADR 0060): the mul1 code at K
 * bits per weight, K in {2, 2.5, 3, 4, 5, 6}. Its inner (rotated-basis) matrix `inner` [in][out]
 * is stored as 16 x 16 tiles, tile (tk, tn) the 16 K int16 words at word
 * (tk * tile_stride + tn) * 16 K of `trellis` (exllamav3's [in/16][out/16][16 K] layout when
 * tile_stride = out / 16), and the weight it encodes is
 *     W = diag(suh) H inner H diag(svh)          (in x out)
 * with H the block-diagonal 128-wide Sylvester Hadamard / sqrt(128) and suh [in], svh [out]
 * fp16. A call computes, for each of `tokens` tokens, y[t] = x[t] W.
 *
 * Two routes, chosen by the token count. Both first rotate the activations (x o suh, then H, in
 * fp32) into fp16 with a power-of-two scale per token and 128-input block, and both multiply on
 * fp16 tensor cores with fp16 accumulation folded into fp32 (times the block's inverse scale)
 * every 32 inputs: the scale keeps a fold's 32 products below 2^15 x 1.73, inside fp16's range
 * whatever the input. GeForce runs fp16-accumulate MMA at twice the fp32-accumulate rate.
 *   decode / verify, 1..IGNIS_TRELLIS_DECODE_MAX_TOKENS tokens: one persistent launch of work
 *            units (128 output columns x a split of the inputs) taken by ticket. Each weight
 *            fragment is decoded once from the packed words into registers and is the MMA's A
 *            operand, the tokens its N: one MMA serves eight tokens, so 1 and 8 tokens read,
 *            decode and multiply the same. The splits of an output block are summed in a fixed
 *            order by the last unit to finish it, which applies the output rotation, svh and
 *            the epilogue. The inputs are split so that the units fill the card's resident CTAs
 *            once (the debug SM cap's, when set).
 *   prefill, more tokens: 128 x 128 output tiles, the trellis tiles staged in shared memory and
 *            decoded from there into the MMAs' B fragments, a four-stage pipeline of 64 inputs.
 *
 * Tolerance, against the fp64 product of the decoded weights (kernel/tests/
 * test_trellis_linear.cu holds every route to it): for an output y with fp64 value Y,
 *     |y - Y| <= 2^-8 |Y| + 2^-9 s,
 * s being |svh| times the root mean square, over the output's 128-column block, of
 * sqrt(sum_k (x'_k inner_kj)^2) -- the size of the products the output sums, x' the rotated
 * activations. The first term is the BF16 output's rounding (at most 2^-8 |y| by itself), the
 * second the fp16 operands' and the accumulation's (the test prints the share of the bound the
 * worst output used). On a card below compute capability 9.0 the second term is 2^-8 s: those
 * tensor cores do not round the FP32 accumulation to nearest (Fasi, Higham, Mikaitis and
 * Pranesh, "Numerical behavior of NVIDIA tensor cores", 2021), and on an sm_86 card the head's
 * worst output reached 1.04 of the 2^-9 bound, against 0.92 on sm_120, at the same relative RMS
 * error (GitHub #456). The residual add
 * rounds Y + r once; SwiGLU's bound composes the two through silu (the test states it), and the
 * prefill SwiGLU stores the gate in BF16 between its two passes.
 *
 * Conventions shared by every entry point:
 * - Activations are device-resident, token-major BF16: x [tokens][in] contiguous, 16-byte
 *   aligned. Outputs are BF16 [tokens][ldy]: token t's `out` values at y + t * ldy, so a call can
 *   write its columns into a wider fused buffer; ldy is a multiple of 4, y 8-byte aligned.
 * - `in` and `out` are positive multiples of 128, `out` at most IGNIS_TRELLIS_MAX_OUT; trellis
 *   16-byte aligned, suh and svh 8-byte aligned.
 * - Every call enqueues on `stream` (a cudaStream_t, NULL for the legacy stream), allocates
 *   nothing, never synchronizes the host, and is capturable in a CUDA graph. Its workspace is a
 *   plan line (ADR 0030): sized by ignis_trellis_workspace_bytes from the load's maxima, reserved
 *   at load, zeroed once by ignis_trellis_workspace_init. One workspace serves the calls of one
 *   stream in sequence; two streams need two.
 * - Results are deterministic: a fixed input gives the same bits run to run, whatever order the
 *   CTAs finish in. No floating-point atomics. The decode route's split of the inputs is a
 *   function of the shape, the token count and the card's SM count, so two cards (or two SM caps)
 *   may round differently, within the tolerance.
 * - Return 0 on success, -1 on an argument the op does not accept or a launch error; the reason
 *   is in ignis_trellis_last_error().
 *
 * The debug SM cap (spec 02's small-card proxy, never a serving setting): IGNIS_TRELLIS_SM_CAP=N
 * in the environment when ignis_trellis_prepare runs limits every launch's persistent grid to
 * the first N SMs of the device (the others' CTAs exit at once), so the 5090 issues the kernels
 * as a card with N SMs would (36, 70, 84: a 5060 Ti, 5070 Ti, 5080). Unset or 0: no cap.
 */
#ifndef IGNIS_TRELLIS_LINEAR_H
#define IGNIS_TRELLIS_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The decode route serves 1..IGNIS_TRELLIS_DECODE_MAX_TOKENS tokens; more take the prefill
 * route. */
#define IGNIS_TRELLIS_DECODE_MAX_TOKENS 64

/* The widest `out` a call takes: the workspace keeps one arrival counter per 128 output columns
 * of this width for every call, so calls of every shape share one workspace (GitHub #339).
 * Workspace geometry, like the decode route's token count, not a serving threshold: the 27B's
 * widest matrix is its head's 248,320. */
#define IGNIS_TRELLIS_MAX_OUT 524288

typedef struct ignis_trellis_weight {
  const void *trellis;  /* int16 tiles, see above */
  const void *suh;      /* fp16 [in] */
  const void *svh;      /* fp16 [out] */
  uint32_t k2;          /* 2 K: 4, 5, 6, 8, 10 or 12 */
  uint32_t in;
  uint32_t out;
  /* Tiles per stored row: 0 for out / 16. A view of columns [c, c + out) of a wider fused plane
   * passes trellis advanced by c / 16 tiles, svh by c, and the plane's out / 16 here (c a
   * multiple of 128). */
  uint32_t tile_stride;
} ignis_trellis_weight;

typedef struct ignis_trellis_workspace {
  void *base;     /* device, 256-byte aligned */
  uint64_t bytes; /* at least what ignis_trellis_workspace_bytes asked for */
} ignis_trellis_workspace;

/* Prepares the current device for every op here: kernel attributes, launch geometry and the
 * debug SM cap. Call once per device at load, outside any stream capture; the ops refuse to run
 * on a device that was not prepared. Thread-safe; a failure is not remembered. */
int32_t ignis_trellis_prepare(void);

/* The SM cap the current device was prepared with, 0 for none. */
uint32_t ignis_trellis_sm_cap(void);

/* The workspace bytes of a call at (in, out) with `tokens` tokens on the current device (prepared
 * first: the decode route's split follows its SM count): `weights` 1 for ignis_trellis_linear,
 * 2 for ignis_trellis_linear_swiglu (gate and up each `out` wide). A load reserves the largest of
 * its calls' (one plan line); 0 for a shape the ops refuse or an unprepared device. */
uint64_t ignis_trellis_workspace_bytes(uint32_t in, uint32_t out, uint32_t weights, uint32_t tokens);

/* Zeroes the workspace's counters, once after it is reserved (every call leaves them zero). */
int32_t ignis_trellis_workspace_init(const ignis_trellis_workspace *workspace, void *stream);

/* The epilogues of ignis_trellis_linear. */
#define IGNIS_TRELLIS_STORE 0 /* y[t][c] = BF16(v) */
#define IGNIS_TRELLIS_ADD 1   /* y[t][c] = BF16(y[t][c] + v), the residual add, in place */

/* y = x W as above, through `epilogue`. */
int32_t ignis_trellis_linear(const ignis_trellis_weight *w, const void *x, uint32_t tokens, void *y,
                             uint32_t ldy, uint32_t epilogue, const ignis_trellis_workspace *workspace,
                             void *stream);

/* h[t][c] = BF16(silu(x[t] G)[c] * (x[t] U)[c]) for the gate and up weights G, U of the same
 * (in, out), each with its own scales and K: the MLP's gate/up pair. */
int32_t ignis_trellis_linear_swiglu(const ignis_trellis_weight *gate, const ignis_trellis_weight *up,
                                    const void *x, uint32_t tokens, void *h, uint32_t ldh,
                                    const ignis_trellis_workspace *workspace, void *stream);

/* The inner matrix of `w` decoded to fp16 [in][out] row-major, bit for bit exllamav3's
 * `reconstruct`. Not on the serving path: the decoder every op runs, exposed so its test can
 * hold it to the oracle exactly (and the fallback a failed fp16 route would take). */
int32_t ignis_trellis_reconstruct(const ignis_trellis_weight *w, void *w_f16, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_trellis_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_TRELLIS_LINEAR_H */
