/* ignis kernel leaf: paged KV page-budget query, flat C ABI (ADR 0001).
 *
 * Host-only arithmetic over the vendored reference's paged KV pool geometry
 * (`core/paged_kv_cache.h`, ADR 0010): given the per-page plane layout of a
 * pool and a VRAM budget, reports how many physical page groups fit and the
 * payload bytes they consume. No device required — used at model-load time,
 * before any allocation, to size a pool's `page_group_count` from the VRAM
 * left after weights (consumed by the sequence handle ABI, GitHub #55).
 *
 * No Rust bindings yet: the sequence handle ABI that calls this (GitHub #55,
 * P1-19) adds them to crates/artifact/src/ffi.rs 1:1, alongside ignis_device.h's.
 */
#ifndef IGNIS_PAGED_KV_BUDGET_H
#define IGNIS_PAGED_KV_BUDGET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One storage plane's per-page geometry (mirrors ninfer::PagedKVPlaneSpec,
 * minus `alignment`: inter-plane padding is negligible next to a page
 * budget and is not part of this query).
 *
 * `dtype` mirrors the ordinals of `ninfer::DType` (core/dtype.h):
 * 0 BF16, 1 FP32, 2 I32, 3 U8, 4 I64, 5 I8, 6 FP16, 7 FP8_E4M3FN.
 */
struct ignis_paged_kv_plane {
    int32_t dtype;
    int32_t leading_extent;
    int32_t head_extent;
};

/* Reports the largest page_group_count whose payload fits `vram_budget_bytes`
 * for a pool built from `planes` (`plane_count` entries, `>= 1`), and the
 * payload bytes that count of pages consumes. Both outputs are 0 when no
 * plane fits even one page.
 *
 * Returns 0 on success, -1 on a bad argument (`plane_count <= 0`, a null
 * `planes`/output pointer, a non-positive extent, or an unrecognized dtype).
 */
int32_t ignis_paged_kv_page_budget(const struct ignis_paged_kv_plane *planes,
                                    int32_t plane_count, uint64_t vram_budget_bytes,
                                    uint32_t *out_page_count, uint64_t *out_page_bytes);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_PAGED_KV_BUDGET_H */
