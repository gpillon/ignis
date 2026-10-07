/* ignis kernel leaf: Flash-Next residual-stack tap -- test-only diagnostic
 * seam (spec flash-next/07 phase A, GitHub #306).
 *
 * NOT part of the public flat C ABI. It copies the trunk's final residual
 * stack -- the output of the last decoder layer, before the final
 * hyper-connection mixer, BF16 [rows][streams * hidden], stream-major -- of
 * every prefill chunk to a host buffer, so a measurement can feed the
 * checkpoint's MTP head (whose hidden input is that pre-mixer stack) the
 * states the engine itself computes.
 *
 * It ships nowhere, for the same reason `ignis_kv_capture.h` does not:
 * `residual_tap.cu` compiles into the intermediate `ignis_kernel` archive, but
 * only `crates/core`'s non-default `residual-tap` feature declares these
 * symbols. The production trace is the disarmed flag load per prefill chunk
 * in the Flash-Next program's prefill.
 *
 * Process-wide: one arm at a time, and nothing else may prefill a Flash-Next
 * sequence while it is armed.
 */
#ifndef IGNIS_FN_RESIDUAL_TAP_H
#define IGNIS_FN_RESIDUAL_TAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Arm the tap: every prefill chunk's rows at positions
 * [first_position, first_position + max_rows) are copied to
 * `out_rows + (position - first_position) * row_elems`. `row_elems` is the
 * residual width the caller expects (streams * hidden); a chunk of another
 * width is refused by the record, which fails the prefill naming it. 0 on
 * success, -1 (see last_error) on a bad argument or when already armed. */
int32_t ignis_fn_residual_tap_arm(uint16_t *out_rows, int64_t first_position, int64_t max_rows,
                                  int32_t row_elems);

/* Disarm, and report how many rows were captured (rows_written may be
 * null). Disarming an unarmed tap is not an error. */
int32_t ignis_fn_residual_tap_disarm(int64_t *rows_written);

const char *ignis_fn_residual_tap_last_error(void);

#ifdef __cplusplus
}

#include <cuda_runtime.h>

/* The hook the prefill calls after the forward of a chunk of `rows` rows at
 * `position`: copies the armed overlap on `stream` (the chunk's own
 * synchronize completes it). 0 when disarmed, without touching the stream. */
int32_t ignis_fn_residual_tap_record(int64_t position, int32_t rows, int32_t row_elems,
                                     const void *device_rows, cudaStream_t stream);
#endif

#endif /* IGNIS_FN_RESIDUAL_TAP_H */
