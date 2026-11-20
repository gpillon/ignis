/* ignis kernel leaf: the image model's FP8 W8A8 GEMM and its per-token activation quantization
 * -- OURS (spec image-model/02, GitHub #319; ADR 0051 tier 1, no port claim. CUTLASS 2.x's sm80
 * multistage GEMM and vLLM's scaled_mm_sm120 were read, nothing was copied).
 *
 * Activations travel as an E4M3 pair, the contract a producer's epilogue (a norm or modulation,
 * SwiGLU, the attention output) emits when it fuses the quantization:
 *   codes  [tokens][cols] E4M3FN, row-major (token t's row at codes + t * cols);
 *   scales [tokens] fp32, x[t][c] ~ scales[t] * e4m3(codes[t][c]).
 * ignis_image_quantize_tokens is the standalone producer: per token, amax = max |x[t][c]|,
 * scales[t] = amax / 448 and codes[t][c] = E4M3(x[t][c] * (448 / amax)), rounded to nearest even
 * and saturated to +-448, each step one fp32 operation. A row whose amax is below 448 * 2^-126
 * (= 7 * 2^-120, the smallest amax whose scale is a normal fp32), an all-zero row among them, gets
 * scale 1 and zero codes. A non-finite x gives a non-finite scale or code: it is never clamped
 * into a finite one.
 *
 * Weights are the FP8_E4M3FN_ROW_BF16S / row-scale-v1 payload (docs/formats/flash-next-layout.md
 * §6.1): E4M3FN codes [rows][cols] row-major, zero padding to a multiple of 256 bytes, then BF16
 * scales [rows] -- what ignis_fp8_linear reads.
 *
 * The GEMM: y[t][r] = (scales[t] * acc) * wscale[r], acc = sum_c e4m3(codes[t][c]) *
 * e4m3(w[r][c]) accumulated in fp32 on block-scaled FP8 tensor cores (mma.sync kind::mxf8f6f4
 * with every block scale 2^0; spec image-model/02 "As built" for why). Output BF16 or fp32,
 * token-major [tokens][rows].
 *
 * Every call enqueues on `stream` (NULL: the legacy stream), allocates nothing, is
 * graph-capturable and deterministic (no atomics; a fixed summation order per output). Return 0,
 * or -1 with the reason in ignis_image_gemm_last_error().
 */
#ifndef IGNIS_IMAGE_GEMM_H
#define IGNIS_IMAGE_GEMM_H

#include <stdint.h>

#include "ignis_fp8_linear.h" /* ignis_fp8_segment */

#ifdef __cplusplus
extern "C" {
#endif

/* Prepares the current device for the image GEMM (kernel attributes). Call once per device at
 * load, outside any stream capture; the GEMM refuses to run on a device that was not prepared.
 * Thread-safe; a failure is not remembered. */
int32_t ignis_image_gemm_prepare(void);

/* The per-token E4M3 quantization above, BF16 x [tokens][cols] in. `cols` is a positive multiple
 * of 16; `x` and `codes` are 16-byte aligned, `scales` 4-byte aligned. */
int32_t ignis_image_quantize_tokens(const void *x, uint32_t tokens, uint32_t cols, void *codes, float *scales,
                                    void *stream);

/* y = the GEMM above for a row-scale weight [rows][cols]. `cols` is a positive multiple of 64 and
 * `rows` of 16; `weight` and `codes` are 16-byte aligned, `scales` 4-byte and `y` 8-byte aligned.
 * `y_f32` selects fp32 output (else BF16). */
int32_t ignis_image_gemm_w8a8(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                              const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream);

/* GitHub #341: several row-scale weights of one quantized input in one launch -- the 27B's q | k |
 * gate | v. Segment i is `weight` [rows][cols] written to its `y` exactly as ignis_image_gemm_w8a8
 * would write it (the same instructions per output, so the same bits). 1..IGNIS_FP8_MAX_SEGMENTS
 * segments, each held to ignis_image_gemm_w8a8's shape and alignment rules. */
int32_t ignis_image_gemm_w8a8_grouped(const ignis_fp8_segment *segments, uint32_t count, uint32_t cols,
                                      const void *codes, const float *scales, uint32_t tokens, uint32_t y_f32,
                                      void *stream);

/* GitHub #341: the residual add, residual[t][r] = BF16(residual[t][r] + y[t][r]) in place, y the fp32
 * value ignis_image_gemm_w8a8 computes (one fp32 add, one rounding); `residual` BF16 [tokens][rows],
 * 8-byte aligned. The output projections' form. */
int32_t ignis_image_gemm_w8a8_add(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                  const float *scales, uint32_t tokens, void *residual, void *stream);

/* GitHub #693 (spec #684): ignis_image_gemm_w8a8_grouped and ignis_image_gemm_w8a8_add on a tile of
 * 16 tokens and 64 output rows a CTA, for a call of 1..IGNIS_IMAGE_GEMM_TILE16_MAX_TOKENS tokens --
 * the 27B's exception matrices at 9..16 tokens. The default tile holds 128 tokens and 256 output
 * rows: such a call fills an eighth of it. Each output takes the same k32 steps in the same order
 * as on the default tile, so the same bits. The arguments and limits of the calls they stand in
 * for, and more tokens refused. ignis_image_gemm_tile16_launches counts their launches since the
 * process started: how a test tells which tile a caller ran. */
#define IGNIS_IMAGE_GEMM_TILE16_MAX_TOKENS 16
int32_t ignis_image_gemm_w8a8_grouped_tile16(const ignis_fp8_segment *segments, uint32_t count, uint32_t cols,
                                             const void *codes, const float *scales, uint32_t tokens,
                                             uint32_t y_f32, void *stream);
int32_t ignis_image_gemm_w8a8_add_tile16(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                         const float *scales, uint32_t tokens, void *residual, void *stream);
uint64_t ignis_image_gemm_tile16_launches(void);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_gemm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_GEMM_H */
