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
 *   lines, ADR 0030), and initialized once by `ignis_moe_workspace_init`, which also prepares
 *   the device (`ignis_moe_prepare`).
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

#include "ignis_fp8_linear.h"

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

/* The router's first launch alone (GitHub #306): `logits` as ignis_moe_router writes them, for a
 * caller that ranks them itself (residency's lookahead) or selects from them in a later launch
 * (ignis_residency_step_demand_routed, the same selection bit for bit). */
int32_t ignis_moe_router_logits(const void *x, uint32_t tokens, const void *w_router, float *logits,
                                void *stream);

/* ---- preparation -----------------------------------------------------------------------------
 * Prepares the current device for every op in this header (and the FP8 linear's): kernel
 * attributes and launch geometry. Call once per device at load, outside any stream capture.
 * The ops never configure themselves lazily -- on a device that was not prepared they refuse to
 * run -- so a first call inside a CUDA graph capture behaves like any other. Thread-safe; a
 * failure is not remembered, the next call retries. */
int32_t ignis_moe_prepare(void);

/* ---- workspace -------------------------------------------------------------------------------
 * One MoE workspace serves both routes: its decode regions are sized for `decode_tokens` (the
 * load's lane count, 1..IGNIS_MOE_DECODE_MAX_TOKENS) and its prefill regions for
 * `prefill_tokens` (the load's maximum prefill chunk). The ops take the same description they
 * were sized with.
 *
 * `decode_route` picks the decode kernel of the model instance that owns the workspace -- one
 * contract, held to the same fp64 bounds -- and is chosen with it, at load:
 *   IGNIS_MOE_DECODE_TICKETS   (0) one persistent launch of work items taken by ticket: up to 4
 *                                  tokens, one CTA per SM whose producer warps stream each item's
 *                                  weights into shared memory ahead of its compute warps
 *                                  (GitHub #306); past 4 tokens, or on a device the staged kernel
 *                                  does not fit, the register kernel below;
 *   IGNIS_MOE_DECODE_CLUSTERS  (1) one thread-block cluster per selected expert, its reductions
 *                                  in distributed shared memory (needs no workspace regions);
 *   IGNIS_MOE_DECODE_REGISTERS (2) the tickets route before #306, at every width: work units that
 *                                  hold their weights in registers, two CTAs per SM.
 * The routes differ only in where partial sums are rounded (fp32 inside a work item, exact fixed
 * point across items), within the same fp64 bounds. A captured graph keeps the kernel it was
 * captured with. */
#define IGNIS_MOE_DECODE_TICKETS 0
#define IGNIS_MOE_DECODE_CLUSTERS 1
#define IGNIS_MOE_DECODE_REGISTERS 2
struct ignis_moe_workspace {
  void *base;
  uint32_t decode_tokens;
  uint32_t prefill_tokens;
  uint32_t decode_route;
};

/* The bytes `base` must hold (the routed accumulator `acc` is separate, see below). */
uint64_t ignis_moe_workspace_bytes(uint32_t decode_tokens, uint32_t prefill_tokens);

/* Makes a fresh workspace ready (and zeroes `acc`, int64 [max(decode, prefill)][2560]); enqueued
 * on `stream`, and prepares the device (ignis_moe_prepare). Call once at load. Every op leaves
 * the workspace ready for its next call. */
int32_t ignis_moe_workspace_init(const struct ignis_moe_workspace *workspace, int64_t *acc,
                                 void *stream);

/* Every device buffer one MoE block needs, as the plan lines the load reserves (ADR 0030); the
 * expert weights themselves are residency's (spec 03). `max` below is max(decode, prefill). */
struct ignis_moe_plan {
  uint64_t workspace;  /* ignis_moe_workspace_bytes(decode_tokens, prefill_tokens) */
  uint64_t acc;        /* the routed accumulator, int64 [max][2560] */
  uint64_t router;     /* ids int32 [max][10], weights fp32 [max][10], logits fp32 [max][512] */
  uint64_t shared;     /* the shared expert's h BF16 [max][640] and output fp32 [max][2560] */
  uint64_t total;      /* the four lines, each rounded up to 256 bytes */
};
int32_t ignis_moe_plan_bytes(uint32_t decode_tokens, uint32_t prefill_tokens,
                             struct ignis_moe_plan *plan);

/* ---- routed experts ------------------------------------------------------------------------
 * The ten selected experts of every token, weighted by their routing weights and summed into
 * `acc`, an int64 [tokens][2560] fixed-point accumulator with 32 fractional bits: the value of
 * an element is acc * 2^-32. Each expert's contribution is converted exactly (rounded only
 * below 2^-32) and added with integer atomics, so the sum is independent of the order the
 * contributions arrive in.
 *
 * `acc` is state the caller owns: the ops ADD into it. It must be zero on entry --
 * `ignis_moe_workspace_init` zeroes it and `ignis_moe_combine` re-zeroes the rows it reads -- and
 * an expert op called twice without a combine between leaves exactly the sum of both calls. A
 * contribution that is not finite, or too large for the format (|value| >= 2^31), traps the
 * kernel rather than turning into a finite wrong number.
 *
 * `ids`/`weights` are the router's outputs; an id outside [0, 512) traps. `slots` is the layer's
 * slot table.
 *
 * Decode route: 1..workspace->decode_tokens tokens in ONE launch for all experts and all four K:
 * gate/up, SwiGLU, down and the weighted accumulation. On IGNIS_MOE_DECODE_TICKETS at up to 4
 * tokens `x` must be 16-byte aligned (the staged kernel copies it 16 bytes at a time; an
 * unaligned `x` is refused). Prefill route: 1..prefill_tokens tokens;
 * they are grouped by expert on the device and each projection family runs as one launch over
 * all experts. */
int32_t ignis_moe_experts_decode(const void *x, uint32_t tokens, const int32_t *ids,
                                 const float *weights, const struct ignis_moe_slot *slots,
                                 const struct ignis_moe_workspace *workspace, int64_t *acc,
                                 void *stream);
int32_t ignis_moe_experts_prefill(const void *x, uint32_t tokens, const int32_t *ids,
                                  const float *weights, const struct ignis_moe_slot *slots,
                                  const struct ignis_moe_workspace *workspace, int64_t *acc,
                                  void *stream);

/* The CTAs per expert cluster the current device runs IGNIS_MOE_DECODE_CLUSTERS with (16, or the
 * portable 8), set by ignis_moe_prepare; 0 if it was not prepared or runs no such cluster (that
 * route then refuses to run). */
int32_t ignis_moe_decode_cluster_size(void);

/* ---- shared expert and combine ---------------------------------------------------------------
 * The shared expert's SwiGLU on the FP8 row-scale linear (ignis_fp8_linear.h; gate_proj and
 * up_proj [640][2560], down_proj [2560][640]) into `shared` (fp32 [tokens][2560]); `h` is BF16
 * [tokens][640] scratch from the plan. */
int32_t ignis_moe_shared_expert(const void *gate, const void *up, const void *down, const void *x,
                                uint32_t tokens, void *h, float *shared, void *stream);

/* out[t] = acc[t] * 2^-32 + sigmoid(x_t . w_gate) * shared[t], fp32, rounded once to BF16;
 * `w_gate` is the BF16 shared_expert_gate [2560]. Zeroes `acc` [tokens][2560] behind it. */
int32_t ignis_moe_combine(int64_t *acc, const float *shared, const void *x, const void *w_gate,
                          uint32_t tokens, void *out, void *stream);

/* Blocks the host until `stream` (NULL: the legacy default stream) has finished its work; for
 * tests and tools that drive the ops directly. */
int32_t ignis_moe_stream_sync(void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_moe_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_MOE_H */
