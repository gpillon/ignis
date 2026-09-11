/* ignis kernel leaf: paged-KV row readback -- test-only diagnostic seam
 * (GitHub #119).
 *
 * NOT part of the public flat C ABI (ignis_seq.h, ADR 0009). It exists for
 * exactly one reason -- a one-shot capture of real BF16 K/V rows from a
 * real prefill of the 27B artifact, committed as
 * `kernel/tests/fixtures/hq_kv_rows_27b.bin` so the hq-e8-2b codec's
 * op-level oracle (P4-03, GitHub #119) can measure its round-trip error
 * against real activations instead of synthetic ones, without loading
 * 19 GB of weights on every CTest run. Once that fixture is captured,
 * nothing in the production engine calls this again; it stays only so the
 * fixture can be re-captured if the artifact or the prompt ever changes.
 *
 * A header comment alone cannot enforce that claim, so here is what
 * actually does: `kernel/CMakeLists.txt` globs every `.cu` under
 * `kernel/src` into `ignis_kernel`, so `kv_capture.cu` genuinely does
 * compile into that static library -- but a static library only pulls an
 * object file into a *final link* when something in that link references
 * one of its symbols. `ignis_kernel.lib` is an intermediate archive, not a
 * shipped artifact; nothing in the production Rust path (`ignis-server`,
 * `ignis-runtime`, or any crate built without the `kv-capture` feature)
 * ever references `ignis_kv_capture_rows`, so the linker never pulls this
 * object into a shipped binary. The real boundary is on the Rust side:
 * `crates/core/src/seq.rs` gates the `extern "C"` declarations and
 * `Seq::capture_kv_rows_for_test` behind the non-default `kv-capture`
 * cargo feature (see that crate's `Cargo.toml`), which is never enabled in
 * a production build. This header staying test-only is a consequence of
 * that gate, not of anything written here.
 *
 * Read-only, no state: it copies already-written K/V rows out of a live
 * `ignis_seq`'s paged KV pages (device to host), synchronously, on the
 * default stream. It never allocates, frees, or mutates device memory.
 *
 * Row addressing mirrors `ops/kernel/paged_kv_address.cuh`'s
 * `paged_kv_element_offset<head_dim, num_kv_heads>`: one (position, kv_head)
 * row is `head_dim` contiguous BF16 elements (raw bit patterns, carried here
 * as `uint16_t` -- the same host-side convention
 * `kernel/vendor/tests/ops/op_tester.h` uses for BF16). A position's
 * physical page comes from the sequence's own bound block table
 * (`ignis_seq::kv.page_ids()`), exactly as the GQA attention kernels
 * resolve it; nothing here re-derives the pool's own layout.
 *
 * Rust bindings: `crates/core/src/seq.rs`'s `ffi` block plus
 * `Seq::capture_kv_rows_for_test`, both `#[cfg(feature = "kv-capture")]` and
 * documented the same way.
 */
#ifndef IGNIS_KV_CAPTURE_H
#define IGNIS_KV_CAPTURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ignis_seq_pool;
struct ignis_seq;

/* Read back `row_count` consecutive (position, kv_head) rows of one GQA
 * layer's K or V plane, starting at `first_position`, into `out_rows`
 * (row-major, `row_count * head_dim` `uint16_t` BF16 bit patterns, caller-
 * allocated).
 *
 * `gqa_layer_ordinal` is the 0..kIgnisGqaLayerCount-1 index `seq`'s own
 * per-layer position tracking uses (kernel/include/ignis_seq_internal.h),
 * not an absolute backbone layer index. `role` is 0 for K, 1 for V. `head_dim`
 * must equal the pool's actual head_dim (the geometry `ignis_seq_pool_create`
 * was built with) -- it is not re-derived from `pool` so a mismatched caller
 * fails loudly instead of silently reading a differently-shaped row.
 *
 * Returns 0 on success. Returns -1 (see ignis_kv_capture_last_error) on a
 * null argument, an out-of-range `gqa_layer_ordinal`/`role`/`kv_head`, a
 * non-positive `row_count` or negative `first_position`, a `head_dim`
 * mismatch, or a `[first_position, first_position + row_count)` range that
 * reaches past what this sequence has actually written for this GQA layer
 * or past its currently mapped pages -- every one of those means the
 * fixture capture asked for something that either does not exist yet or
 * was never real activations, and this seam has exactly one caller who
 * would want to know that immediately, not read garbage. */
int32_t ignis_kv_capture_rows(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                               int32_t gqa_layer_ordinal, int32_t role, int32_t kv_head,
                               int32_t first_position, int32_t row_count, int32_t head_dim,
                               uint16_t *out_rows);

/* The message from the most recent failing call on this thread
 * (thread-local; overwritten by the next call; empty string if none failed
 * yet). Never NULL. Mirrors ignis_seq_last_error's convention
 * (kernel/include/ignis_seq.h) but is this header's own instance -- this
 * seam is not part of that ABI. */
const char *ignis_kv_capture_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_KV_CAPTURE_H */
