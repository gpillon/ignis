/* ignis kernel leaf: the FP8 row-scale linear -- OURS (spec flash-next/04's op, written for spec
 * 02's shared expert; ADR 0043, no port claim).
 *
 * `weight` is an FP8_E4M3FN_ROW_BF16S / row-scale-v1 payload (docs/formats/flash-next-layout.md
 * §6.1): E4M3FN codes [rows][cols] row-major, zero padding to a multiple of 256 bytes, then BF16
 * scales [rows]. y[t][r] = scale[r] * sum_c e4m3(code[r][c]) * x[t][c] with BF16 activations,
 * token-major (token t's row at x + t * cols), and fp32 accumulation: an E4M3 x BF16 product is
 * exact in fp32, so the only roundings are the accumulation's and the output's.
 *
 * Calls of up to 8 tokens take a GEMV route, wider ones BF16 tensor cores. Every call enqueues
 * on `stream` (NULL: the legacy stream), allocates nothing, is graph-capturable and
 * deterministic. Return 0, or -1 with the reason in ignis_fp8_linear_last_error().
 */
#ifndef IGNIS_FP8_LINEAR_H
#define IGNIS_FP8_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Prepares the current device for the FP8 linear (kernel attributes). Call once per device at
 * load, outside any stream capture; the ops refuse to run on a device that was not prepared.
 * Thread-safe; a failure is not remembered. */
int32_t ignis_fp8_linear_prepare(void);

/* y = W x as above. `cols` is a multiple of 64 and `rows` a multiple of 16; `weight` and `x`
 * are 16-byte aligned and `y` 8-byte aligned. `y_f32` selects fp32 output (else BF16). */
int32_t ignis_fp8_linear(const void *weight, uint32_t rows, uint32_t cols, const void *x,
                         uint32_t tokens, void *y, uint32_t y_f32, void *stream);

/* h[t][r] = silu(gate . x_t)[r] * (up . x_t)[r] in BF16, for two FP8 row-scale weights of the
 * same shape; `h` 8-byte aligned. */
int32_t ignis_fp8_linear_swiglu(const void *gate, const void *up, uint32_t rows, uint32_t cols,
                                const void *x, uint32_t tokens, void *h, void *stream);

/* GitHub #306: several FP8 row-scale linears of one input in one launch -- a decode round's
 * projections of the same x. Segment i is `weight` [rows][cols] written to its `y` exactly as
 * ignis_fp8_linear would write it (the same GEMV code for each output row, so the same bits). The
 * GEMV route only: 1..IGNIS_FP8_GEMV_MAX_TOKENS tokens, at most IGNIS_FP8_MAX_SEGMENTS segments,
 * every `rows` a positive multiple of 16, every weight and x 16-byte aligned, every y 8-byte
 * aligned. */
#define IGNIS_FP8_GEMV_MAX_TOKENS 8
#define IGNIS_FP8_MAX_SEGMENTS 4
typedef struct ignis_fp8_segment {
  const void *weight;
  uint32_t rows;
  void *y;
} ignis_fp8_segment;
int32_t ignis_fp8_linear_grouped(const ignis_fp8_segment *segments, uint32_t count, uint32_t cols,
                                 const void *x, uint32_t tokens, uint32_t y_f32, void *stream);

/* GitHub #678 (spec #672): ignis_fp8_linear_grouped's paired form, the 27B's grouped q | k | gate |
 * v exception projection at decode widths: two output rows a warp, every code load of a batch
 * issued before its first multiply-add, each output summed in the one-row form's order, so every
 * segment's y holds the bytes ignis_fp8_linear_grouped writes. That call's arguments and limits,
 * and `cols` at most 6144. ignis_fp8_linear_paired_launches counts its launches since the process
 * started: how a test tells which form a caller ran. */
int32_t ignis_fp8_linear_grouped_paired(const ignis_fp8_segment *segments, uint32_t count, uint32_t cols,
                                        const void *x, uint32_t tokens, uint32_t y_f32, void *stream);
uint64_t ignis_fp8_linear_paired_launches(void);

/* GitHub #341: the residual add, residual[t][r] = BF16(residual[t][r] + y[t][r]) in place, y the
 * fp32 value ignis_fp8_linear computes on the same route (one fp32 add, one rounding); `residual`
 * BF16 [tokens][rows], 8-byte aligned. The 27B's output projections with BF16 activations. */
int32_t ignis_fp8_linear_add(const void *weight, uint32_t rows, uint32_t cols, const void *x,
                             uint32_t tokens, void *residual, void *stream);

/* GitHub #341: the bytes of a row-scale-v1 payload [rows][cols] -- the codes, zero padding to a
 * multiple of 256, the BF16 scales. */
uint64_t ignis_fp8_payload_bytes(uint32_t rows, uint32_t cols);

/* GitHub #341: a BF16 weight w [rows][cols] (row-major, as a BF16_CTRL tensor is stored) to that
 * payload, on the device: per row, scale = the least BF16 >= fl32(amax / 448), where amax =
 * max |w[r][c]| (1 for an all-zero row), and codes[r][c] = E4M3(fl32(w[r][c] / scale)), rounded
 * to nearest even and saturated to +-448; the padding is zeroed. `cols` is a positive multiple of
 * 16, `rows` positive; `w` and `payload` 16-byte aligned, `payload` ignis_fp8_payload_bytes long.
 * Allocates nothing; deterministic. */
int32_t ignis_fp8_quantize_rows(const void *w, uint32_t rows, uint32_t cols, void *payload, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_fp8_linear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_FP8_LINEAR_H */
