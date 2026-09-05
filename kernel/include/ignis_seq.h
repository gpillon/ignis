/* ignis kernel leaf: sequence handle flat C ABI (ADR 0009, GitHub #55).
 *
 * A request owns real device state: KV pages from the vendored paged KV
 * pool (`core/paged_kv_cache.h`) and a GDN slot (recurrent state + conv
 * taps) from the vendored linear-attention state pool
 * (`core/linear_attention_state.h`, ADR 0010). One logical slot index
 * addresses both pools for a sequence: the KV pool's block-table row and
 * the GDN pool's state slot are the same number, so `ignis_seq_alloc`
 * hands out one slot that owns both.
 *
 * `ignis_seq_pool_create` builds the two device-resident pools once, sized
 * by `kv_page_group_count` (a physical KV page count the caller already
 * sized -- typically from `ignis_paged_kv_page_budget`, ignis_paged_kv_budget.h)
 * and `max_context_tokens` (the largest single-sequence KV reservation, via
 * the vendored `pages_for_tokens`). `ignis_seq_alloc` / `ignis_seq_release`
 * then draw from and return to that fixed pool -- no device allocation on
 * the request path.
 *
 * A freshly allocated sequence's KV pages and GDN slot (recurrent state +
 * conv taps) are zeroed before the handle is returned, so a released and
 * re-allocated sequence never observes another request's state.
 *
 * Snapshot / restore entry points are declared now (the KV-RAM host tier,
 * G4) but return `IGNIS_SEQ_ERR_NOT_IMPLEMENTED` until then.
 *
 * Rust bindings: crates/core/src/seq.rs (keep 1:1).
 */
#ifndef IGNIS_SEQ_H
#define IGNIS_SEQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A call that succeeds returns 0. A bad argument or pool exhaustion
 * returns -1 (see ignis_seq_last_error). A not-yet-implemented entry point
 * returns this instead. */
#define IGNIS_SEQ_ERR_NOT_IMPLEMENTED (-2)

/* Opaque device-resident pool of sequence state (never dereferenced across
 * the boundary). Allocated by ignis_seq_pool_create, destroyed by
 * ignis_seq_pool_free. */
struct ignis_seq_pool;

/* Opaque sequence handle: one slot's KV allocation + GDN state. Allocated
 * by ignis_seq_alloc, destroyed by ignis_seq_release. */
struct ignis_seq;

/* The geometry a sequence-state pool is built from. KV pages are two BF16
 * planes (K, V) of `[head_dim, kPagedKVPageSize, num_kv_heads]` each
 * (`core/paged_kv_cache.h`); the GDN pool is `gdn_num_layers` layers of
 * `[gdn_conv_channels]` conv taps (4-wide causal conv, the model's fixed
 * kernel width) and `gdn_value_heads` fp32 `[gdn_head_dim, gdn_head_dim]`
 * recurrent state matrices (mirrors ninfer::LinearAttentionStatePoolSpec;
 * the reference's GDN recurrence is square: value_head_dim == key_head_dim
 * == gdn_head_dim). */
struct ignis_seq_pool_spec {
  uint32_t num_kv_heads;
  uint32_t head_dim;
  /* Physical KV page count this pool holds -- the caller sizes this
   * (typically from ignis_paged_kv_page_budget against the VRAM left after
   * weights), not derived here. */
  uint32_t kv_page_group_count;
  /* The largest single sequence's KV reservation, in tokens; sets the
   * block-table's logical-page capacity (ninfer::pages_for_tokens). */
  uint32_t max_context_tokens;
  /* Max concurrent sequences: the KV pool's block-table row count and the
   * GDN pool's slot count (the same number addresses both). */
  uint32_t slot_count;
  uint32_t gdn_num_layers;
  uint32_t gdn_conv_channels;
  uint32_t gdn_value_heads;
  uint32_t gdn_head_dim;
};

struct ignis_seq_pool_stats {
  uint32_t kv_page_group_count;
  uint32_t kv_entitled_pages;
  uint32_t kv_free_pages;
  /* Bytes of one physical KV page across every plane (K + V). */
  uint64_t kv_page_bytes;
  uint32_t logical_page_capacity;
  uint32_t slot_count;
  uint32_t free_slot_count;
};

struct ignis_seq_stats {
  int32_t slot;
  uint32_t page_entitlement;
  uint32_t mapped_pages;
  uint64_t token_capacity;
};

/* Build the two device-resident pools from `spec`. Returns 0 and a handle
 * in `*out_pool` on success. Returns -1 (see ignis_seq_last_error) on a
 * null argument or a non-positive geometry field. */
int32_t ignis_seq_pool_create(const struct ignis_seq_pool_spec *spec,
                               struct ignis_seq_pool **out_pool);

/* Pool-wide geometry + live usage (the "runtime reports page geometry"
 * surface the scheduler's KV pool sizes from). Returns 0 on success, -1 on
 * a null argument. */
int32_t ignis_seq_pool_stats(const struct ignis_seq_pool *pool,
                              struct ignis_seq_pool_stats *out_stats);

/* Release a pool handle. Every sequence drawn from it must already be
 * released. NULL is a no-op. */
void ignis_seq_pool_free(struct ignis_seq_pool *pool);

/* Reserve a slot: KV pages for `context_tokens` (ninfer::pages_for_tokens)
 * plus the slot's GDN state, all zeroed before return. Returns 0 and a
 * handle in `*out_seq` on success. Returns -1 (no sequence produced; see
 * ignis_seq_last_error) on a null argument, `context_tokens == 0`, a
 * `context_tokens` beyond `max_context_tokens`, or pool exhaustion (no
 * free slot, or not enough free KV pages) -- the pool is left unchanged on
 * failure. */
int32_t ignis_seq_alloc(struct ignis_seq_pool *pool, uint32_t context_tokens,
                         struct ignis_seq **out_seq);

/* Release a sequence: returns its KV pages and slot to `pool`, which must
 * be the pool `seq` was allocated from. A NULL `seq` is a no-op. */
void ignis_seq_release(struct ignis_seq_pool *pool, struct ignis_seq *seq);

/* Statistics of a live sequence. Returns 0 on success, -1 on a null
 * argument. */
int32_t ignis_seq_stats(const struct ignis_seq *seq, struct ignis_seq_stats *out_stats);

/* Snapshot a sequence's device state to a caller-provided host region /
 * restore it from one (the KV-RAM host tier, G4). Not implemented yet:
 * always returns IGNIS_SEQ_ERR_NOT_IMPLEMENTED. */
int32_t ignis_seq_snapshot(const struct ignis_seq *seq, void *dst, uint64_t dst_bytes);
int32_t ignis_seq_restore(struct ignis_seq *seq, const void *src, uint64_t src_bytes);

/* The message from the most recent failing call on this thread
 * (thread-local; overwritten by the next call; empty string if none failed
 * yet). Never NULL. */
const char *ignis_seq_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_SEQ_H */
