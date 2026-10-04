/* ignis kernel leaf: Flash-Next's mixture-of-experts ops -- OURS (spec flash-next/02, GitHub
 * #300; ADR 0043, ADR 0044).
 *
 * Every op here is our own implementation. None is vendored, patched or translated from
 * another engine, and none carries a port claim (ADR 0010 / ADR 0043). ExLlamaV3's
 * `reconstruct` and the checkpoint's transformers modules are the oracles their tests are held
 * to (kernel/tests/fixtures/flash_next/), not sources.
 *
 * The block, per token: a router picks 10 of 512 experts from the BF16 router weight; each
 * selected expert is a SwiGLU MLP 2560 -> 640 -> 2560 stored as two trellis-coded *expert
 * projections* (ADR 0044, docs/specs/flash-next/layout.md §3) at its own bit width K; a shared
 * expert (FP8 row-scale, gated by a sigmoid of a BF16 2560 -> 1 projection) runs on every token;
 * the combine sums them.
 *
 * Conventions shared by every entry point:
 * - Activations are device-resident, token-major: token t's hidden vector is the `hidden`
 *   contiguous elements at `x + t * hidden` (the leaf's feature-major `[hidden, T]`).
 * - Every call enqueues on `stream` (a `cudaStream_t`, NULL for the legacy stream), allocates
 *   nothing, never synchronizes the host, and is capturable in a CUDA graph. Workspaces are
 *   sized by the `*_workspace_bytes` queries from the load's maxima, reserved at load (plan
 *   lines, ADR 0030), and initialized once by `ignis_moe_workspace_init`.
 * - Results are deterministic: a fixed input gives the same bits run to run, whatever the
 *   number of CTAs, whichever slot an expert occupies, and whatever order the hardware
 *   completes work in. Nothing depends on the order of floating-point atomics: the routed
 *   output is accumulated in exact fixed point (below).
 * - Return 0 on success, -1 on an argument the op does not accept or a launch error; the
 *   reason is in `ignis_moe_last_error()`.
 */
#ifndef IGNIS_MOE_H
#define IGNIS_MOE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Flash-Next's MoE geometry. The ops are written for it and refuse anything else. */
#define IGNIS_MOE_HIDDEN 2560
#define IGNIS_MOE_EXPERTS 512
#define IGNIS_MOE_TOP_K 10
#define IGNIS_MOE_INTERMEDIATE 640
/* The decode route serves 1..IGNIS_MOE_DECODE_MAX_TOKENS tokens in one launch. */
#define IGNIS_MOE_DECODE_MAX_TOKENS 8

/* Expert projections (layout.md §3): the fused gate/up plane, (in, out) = (2560, 1280) with
 * output columns [0, 640) gate and [640, 1280) up; and down, (640, 2560). */
#define IGNIS_MOE_PROJ_GATE_UP 0
#define IGNIS_MOE_PROJ_DOWN 1

/* One slot-table entry: where an expert projection's record lives on the device and its bit
 * width as k2 = 2 K (4, 5, 6 or 8). A record is layout.md §3's: the trellis tensor at offset 0,
 * then suh (fp16, `in`), then svh (fp16, `out`), each 16-byte aligned; the record itself must
 * be 16-byte aligned.
 *
 * The table of a layer is `IGNIS_MOE_EXPERTS * 2` entries indexed `expert * 2 + projection`.
 * Residency (spec 03) fills it and guarantees that every projection the router selected is
 * resident before an expert op runs; the kernels never wait, copy or read host memory. An
 * entry that is selected but not resident is a programming error: a NULL `record` or a `k2`
 * outside {4, 5, 6, 8} traps the kernel (the launch fails, the context is lost), it is never
 * read as weights. */
struct ignis_moe_slot {
  const void *record;
  uint32_t k2;
  uint32_t reserved;
};

/* The bytes of one expert-projection record of `projection` at `k2`, padding included
 * (layout.md's class table), into `*bytes`. */
int32_t ignis_moe_record_bytes(uint32_t projection, uint32_t k2, uint64_t *bytes);

/* ---- trellis decode ------------------------------------------------------------------------
 * The inner (rotated-basis) weight of a trellis tensor, decoded to an fp16 [in][out] row-major
 * matrix, bit for bit what exllamav3's `reconstruct` produces. Not on the serving path: this is
 * the decoder every expert op runs, exposed so its test can hold it to the oracle exactly.
 * `in` and `out` are multiples of 16. */
int32_t ignis_moe_trellis_reconstruct(const void *trellis, uint32_t k2, uint32_t in, uint32_t out,
                                      void *w_f16, void *stream);

/* ---- router ----------------------------------------------------------------------------------
 * For each of `tokens` tokens of `x` (BF16), the checkpoint's router: logits = x . W^T over the
 * BF16 weight `w_router` [512][2560], accumulated in fp32 and rounded once to BF16 as the
 * transformers module returns them; the 10 largest BF16 logits, ties to the lower expert id,
 * written to `ids` [tokens][10] in descending order; their weights, a softmax over those ten
 * logits in fp32 rounded to BF16 (the module's `.to(bfloat16)`), written as fp32 to `weights`
 * [tokens][10]. `logits` [tokens][512] receives the fp32 logits before rounding; it is required
 * (the op's scratch between its two launches, a plan line). Any `tokens` >= 1; `x` and
 * `w_router` 16-byte aligned. A token's results do not depend on how many tokens share the
 * call. */
int32_t ignis_moe_router(const void *x, uint32_t tokens, const void *w_router, int32_t *ids,
                         float *weights, float *logits, void *stream);

/* ---- routed experts ------------------------------------------------------------------------
 * The ten selected experts of every token, weighted by their routing weights and summed into
 * `acc`, an int64 [tokens][2560] fixed-point accumulator with 32 fractional bits: the value of
 * an element is acc * 2^-32. Each expert's contribution is converted exactly (rounded only
 * below 2^-32) and added with integer atomics, so the sum is independent of the order the
 * contributions arrive in. `acc` must be zero on entry (`ignis_moe_workspace_init` zeroes it,
 * `ignis_moe_combine` re-zeroes what it reads).
 *
 * `ids`/`weights` are the router's outputs; `slots` the layer's slot table.
 *
 * Decode route: 1..IGNIS_MOE_DECODE_MAX_TOKENS tokens in ONE launch for all experts and all
 * four K: gate/up, SwiGLU, down and the weighted accumulation. Prefill route: any number of
 * tokens up to the workspace's maximum chunk; the tokens are grouped by expert on the device
 * and each projection family runs as one launch over all experts; `max_tokens` is the
 * workspace's chunk maximum. */
int32_t ignis_moe_experts_decode(const void *x, uint32_t tokens, const int32_t *ids,
                                 const float *weights, const struct ignis_moe_slot *slots,
                                 void *workspace, int64_t *acc, void *stream);
int32_t ignis_moe_experts_prefill(const void *x, uint32_t tokens, const int32_t *ids,
                                  const float *weights, const struct ignis_moe_slot *slots,
                                  void *workspace, uint32_t max_tokens, int64_t *acc,
                                  void *stream);

/* ---- FP8 row-scale linear (spec 04's, used by the shared expert) -----------------------------
 * `weight` is an FP8_E4M3FN_ROW_BF16S / row-scale-v1 payload (layout.md §6.1): E4M3FN codes
 * [rows][cols], zero padding to a multiple of 256 bytes, then BF16 scales [rows]; 16-byte
 * aligned. y[t][r] = scale[r] * sum_c e4m3(code[r][c]) * x[t][c], BF16 activations, fp32
 * accumulation. `cols` is a multiple of 64 and `rows` a multiple of 16. `y_f32` selects fp32
 * output (else BF16). Decode-width calls take a GEMV route, wide ones the tensor cores. */
int32_t ignis_fp8_linear(const void *weight, uint32_t rows, uint32_t cols, const void *x,
                         uint32_t tokens, void *y, uint32_t y_f32, void *stream);

/* h[t][r] = silu(gate . x_t)[r] * (up . x_t)[r] in BF16, for two FP8 row-scale weights of the
 * same shape. */
int32_t ignis_fp8_linear_swiglu(const void *gate, const void *up, uint32_t rows, uint32_t cols,
                                const void *x, uint32_t tokens, void *h, void *stream);

/* ---- shared expert and combine ---------------------------------------------------------------
 * The shared expert's SwiGLU (FP8 gate_proj/up_proj [640][2560], down_proj [2560][640]) into
 * `shared` (fp32 [tokens][2560]); `h` is BF16 [tokens][640] scratch from the workspace plan. */
int32_t ignis_moe_shared_expert(const void *gate, const void *up, const void *down, const void *x,
                                uint32_t tokens, void *h, float *shared, void *stream);

/* out[t] = acc[t] * 2^-32 + sigmoid(x_t . w_gate) * shared[t], fp32, rounded once to BF16;
 * `w_gate` is the BF16 shared_expert_gate [2560]. Zeroes `acc` [tokens][2560] behind it. */
int32_t ignis_moe_combine(int64_t *acc, const float *shared, const void *x, const void *w_gate,
                          uint32_t tokens, void *out, void *stream);

/* ---- workspace -------------------------------------------------------------------------------
 * The device workspace the routed-expert ops need for chunks of up to `max_tokens` tokens
 * (decode and prefill share it; size it for the larger of the two), excluding `acc`. */
uint64_t ignis_moe_workspace_bytes(uint32_t max_tokens);

/* Prepares a fresh workspace (and zeroes `acc` [max_tokens][2560]); enqueued on `stream`. Call
 * once at load. The ops leave it ready for their next call. */
int32_t ignis_moe_workspace_init(void *workspace, uint32_t max_tokens, int64_t *acc, void *stream);

/* Blocks the host until `stream` (NULL: the legacy default stream) has finished its work; for
 * tests and tools that drive the ops directly. */
int32_t ignis_moe_stream_sync(void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_moe_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_MOE_H */
