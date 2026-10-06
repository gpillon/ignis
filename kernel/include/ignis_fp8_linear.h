/* ignis kernel leaf: the FP8 row-scale linear -- OURS (spec flash-next/04's op, written for spec
 * 02's shared expert; ADR 0043, no port claim).
 *
 * `weight` is an FP8_E4M3FN_ROW_BF16S / row-scale-v1 payload (docs/specs/flash-next/layout.md
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

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_fp8_linear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_FP8_LINEAR_H */
