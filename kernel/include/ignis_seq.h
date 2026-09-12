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
 * A freshly allocated sequence's KV pages, GDN slot (recurrent state + conv
 * taps) and penalty-count buffer (P3-03, GitHub #99: one int32 per vocab
 * entry, read and updated by device-side sampling's presence/frequency
 * penalties) are zeroed before the handle is returned, so a released and
 * re-allocated sequence never observes another request's state.
 *
 * A sequence can also leave the GPU and come back (P4-06, GitHub #124, ADR
 * 0024): `ignis_seq_snapshot_size` reports what a whole-sequence snapshot
 * costs as the sequence stands, `ignis_seq_snapshot` writes exactly that
 * many bytes of an opaque self-describing blob into a caller-provided host
 * region, and `ignis_seq_restore` validates the blob's own header before it
 * writes any of it into a sequence. What crosses this boundary is a size
 * and a format version -- the leaf's state-section table stays internal
 * (kernel/include/ignis_seq_sections.h).
 *
 * Sibling sequences that share a prompt head share its **pages** rather than
 * re-prefilling them (P4-10, GitHub #126, ADR 0024):
 * `ignis_seq_prefix_publish` hands a sequence's leading pages to a leaf-owned
 * refcount, and `ignis_seq_alloc_shared` gives a claimant those same physical
 * pages plus a device-to-device clone of the mutable state. Neither direction
 * crosses PCIe, which is what separates it from the snapshot path above -- and
 * a sequence that holds a prefix cannot take that path at all
 * (IGNIS_SEQ_ERR_SHARED_PREFIX).
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

/* ignis_seq_snapshot only: the sequence is mid-chunk, so its sections are
 * not mutually consistent and there is nothing coherent to capture (ADR
 * 0018 / 0024). Distinct from -1 because it is not a caller mistake about
 * arguments -- the same call succeeds once the in-flight chunk completes,
 * which is what an evicting scheduler waits for (GitHub #125). */
#define IGNIS_SEQ_ERR_NOT_AT_BOUNDARY (-3)

/* ignis_seq_restore only: the blob is not one this leaf can restore -- a
 * foreign buffer, a stale format version, a size that disagrees with the
 * blob's own header, or a geometry the target sequence does not match. The
 * target sequence is left untouched. Distinct from -1 so a host tier can
 * discard the blob and re-prefill rather than treat it as a bug in its own
 * call. */
#define IGNIS_SEQ_ERR_BAD_SNAPSHOT (-4)

/* The state-transfer calls (ignis_seq_snapshot_size, ignis_seq_snapshot,
 * ignis_seq_restore): the sequence holds a shared prefix (P4-10, GitHub
 * #126), so its KV history is not all its own and there is no
 * whole-sequence blob to size, to write, or to write back. The claim is
 * released with the sequence; a caller that needs this sequence off the GPU
 * releases it and re-prefills instead. Distinct from -1 because the call is
 * well formed -- it is the sequence, not the arguments, that cannot be
 * transferred. */
#define IGNIS_SEQ_ERR_SHARED_PREFIX (-5)

/* Opaque device-resident pool of sequence state (never dereferenced across
 * the boundary). Allocated by ignis_seq_pool_create, destroyed by
 * ignis_seq_pool_free. */
struct ignis_seq_pool;

/* Opaque sequence handle: one slot's KV allocation + GDN state. Allocated
 * by ignis_seq_alloc, destroyed by ignis_seq_release. */
struct ignis_seq;

/* Opaque shared prefix: physical KV pages a leaf-owned refcount keeps alive,
 * plus a device-resident image of the mutable state at the prefix's end
 * (P4-10, GitHub #126, ADR 0024). Published from a live sequence by
 * ignis_seq_prefix_publish, claimed by ignis_seq_alloc_shared, released by
 * ignis_seq_prefix_release. */
struct ignis_seq_prefix;

/* The KV cache storage format a pool stores its rows in (ADR 0022, GitHub
 * #122). Fixed for the life of a model load: it decides the pool's planes,
 * and so how many bytes one sequence-token costs and how many tokens a byte
 * budget holds.
 *
 * Rust binding: `ignis_core::kv_format::KvFormat::abi_code` (keep 1:1). */
enum ignis_kv_format {
  /* Unquantized BF16 K/V rows: one plane per role per GQA layer, of
   * `[head_dim, kPagedKVPageSize, num_kv_heads]`. 65,536 bytes per
   * sequence-token at the 27B geometry. */
  IGNIS_KV_FORMAT_BF16 = 0,
  /* hq-e8-2b (`ops/kernel/hq_codec.cuh`): two U8 planes per role per GQA
   * layer -- a `[kHqRowBudgetBytes=64, kPagedKVPageSize, num_kv_heads]` code
   * plane and a `[kHqMetaBytes=8, ...]` metadata plane. Every (token, KV
   * head) row occupies exactly those fixed budgets, so page addressing and
   * capacity math are unchanged; 9,216 bytes per sequence-token, 7.11x
   * denser than BF16. */
  IGNIS_KV_FORMAT_HQ_E8_2B = 1
};

/* The geometry a sequence-state pool is built from. KV pages carry
 * `kv_format`'s planes per GQA layer (see enum ignis_kv_format), addressed
 * by `core/paged_kv_cache.h`'s page-major layout; the GDN pool is
 * `gdn_num_layers` layers of `[gdn_conv_channels]` conv taps (4-wide causal
 * conv, the model's fixed kernel width) and `gdn_value_heads` fp32
 * `[gdn_head_dim, gdn_head_dim]` recurrent state matrices (mirrors
 * ninfer::LinearAttentionStatePoolSpec; the reference's GDN recurrence is
 * square: value_head_dim == key_head_dim == gdn_head_dim). */
struct ignis_seq_pool_spec {
  uint32_t num_kv_heads;
  uint32_t head_dim;
  /* One of enum ignis_kv_format. Rejected if it is neither. */
  int32_t kv_format;
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
  /* The model's vocabulary size (P3-03, GitHub #99): sizes each slot's
   * presence/frequency penalty count buffer (one int32 per vocab entry).
   * Must match the model the pool's sequences are stepped with. */
  uint32_t vocab;
};

struct ignis_seq_pool_stats {
  uint32_t kv_page_group_count;
  uint32_t kv_entitled_pages;
  uint32_t kv_free_pages;
  /* Bytes of one physical KV page across every plane of every GQA layer --
   * K and V under BF16, their code and metadata planes under hq-e8-2b. */
  uint64_t kv_page_bytes;
  uint32_t logical_page_capacity;
  uint32_t slot_count;
  uint32_t free_slot_count;
  /* The format the pool stores rows in (enum ignis_kv_format). */
  int32_t kv_format;
  /* Bytes one sequence-token costs across every GQA layer and both roles,
   * derived from the planes the pool actually planned (kv_page_bytes /
   * kPagedKVPageSize) -- not a per-format constant. */
  uint64_t kv_bytes_per_token;
  /* Resident sequence-tokens the whole pool holds: kv_page_group_count *
   * kPagedKVPageSize. Derived from the byte budget the caller sized the
   * pool with and the format in force, which is the number a load reports
   * (GitHub #122). */
  uint64_t kv_token_capacity;
};

struct ignis_seq_stats {
  int32_t slot;
  /* Pages, and tokens, of this sequence's whole history -- the shared
   * prefix's pages included, since its block-table row addresses those too
   * (P4-10, GitHub #126). */
  uint32_t page_entitlement;
  uint32_t mapped_pages;
  uint64_t token_capacity;
  /* How many of `mapped_pages` belong to a shared prefix rather than to this
   * sequence: 0 for a sequence that prefilled its own head. The pool is
   * charged for these once, not once per claimant. */
  uint32_t shared_pages;
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

/* --- state transfer (P4-06, GitHub #124, ADR 0024) -----------------------
 *
 * On ADR 0016: it rules that "later phases add fields, not parameters and
 * not entry points", and names G4's snapshot controls as an example. That
 * rule is about per-call *modulation* -- a prefill route, a compute policy,
 * sampling parameters -- which is what would otherwise multiply parameters
 * or `_ex` entry points. These two additions are neither. The `pool` handle
 * is the storage the call operates on, which every other sequence entry
 * point here already takes (ignis_seq_alloc, ignis_seq_release,
 * ignis_seq_pool_stats); and a size query and a version are queries with
 * nothing to modulate, so an options struct would have no field to carry.
 * Snapshot controls, when a phase needs one, still go in a struct. Nothing
 * called either entry point before this change, so per ADR 0016 the Rust
 * binding moves with it and no wrapper is kept.
 */

/* The snapshot blob format version this leaf writes and accepts. One of the
 * two things state transfer exposes across this boundary; the other is the
 * size below. A blob written under a different version is refused by
 * ignis_seq_restore, never reinterpreted, so a caller that persists blobs
 * across builds records this alongside them. */
uint32_t ignis_seq_snapshot_format_version(void);

/* Bytes ignis_seq_snapshot would write for `seq` as it stands now.
 *
 * A whole-sequence figure: its written KV history (not its whole
 * reservation), its GDN recurrent state and conv taps, its penalty-count
 * row and its progress scalars, plus the blob's own header. It therefore
 * grows with the sequence, and a caller sizes its host region per snapshot
 * rather than once per pool.
 *
 * Returns 0 and the size in `*out_bytes`. Returns -1 on a null argument or
 * a sequence that is not `pool`'s (see ignis_seq_last_error);
 * IGNIS_SEQ_ERR_NOT_AT_BOUNDARY if `seq` is mid-chunk, and
 * IGNIS_SEQ_ERR_SHARED_PREFIX if it claims a shared prefix -- for the same
 * reasons the snapshot itself is refused in each case. */
int32_t ignis_seq_snapshot_size(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                                 uint64_t *out_bytes);

/* Write `seq`'s whole device state into `dst` as an opaque, self-describing
 * blob of exactly the size ignis_seq_snapshot_size reports.
 *
 * The blob is opaque: its layout is the leaf's, it carries its own header,
 * and the only valid thing to do with it is hand it back to
 * ignis_seq_restore on a pool of the same geometry. `dst` is any host
 * region of at least that many bytes (pinned memory is faster, and is what
 * the host tier uses, GitHub #125); the call is synchronous -- it returns
 * with every byte already in `dst`.
 *
 * A snapshot is taken only at a **chunk boundary**: mid-chunk the
 * sequence's sections are not consistent with one another, so the call is
 * refused with IGNIS_SEQ_ERR_NOT_AT_BOUNDARY rather than capturing state
 * that would restore into a subtly wrong sequence. Returns -1 on a null
 * argument, a sequence that is not `pool`'s, a `dst_bytes` below the
 * reported size, or a failed device copy. `seq` is never modified.
 *
 * A sequence that claims a shared prefix is refused with
 * IGNIS_SEQ_ERR_SHARED_PREFIX: its leading pages belong to the prefix, so
 * there is no whole-sequence blob to write (P4-10, GitHub #126). */
int32_t ignis_seq_snapshot(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                            void *dst, uint64_t dst_bytes);

/* Restore a sequence from a blob ignis_seq_snapshot wrote.
 *
 * Whole-sequence, one call, no partial restore: GDN state cannot be
 * recomputed without re-running the prefix a restore exists to avoid (ADR
 * 0024). `src_bytes` must be exactly the blob's own recorded size.
 *
 * Every check runs before any byte is written, so a refused restore leaves
 * `seq` exactly as it was: the magic, the format version, the recorded size,
 * the section layout, and the geometry the blob was taken at, against
 * `pool`'s own. A mismatch returns IGNIS_SEQ_ERR_BAD_SNAPSHOT. `seq` must
 * also hold at least as many mapped KV pages as the blob carries, which is
 * the one requirement on the target beyond matching geometry. Returns -1 on
 * a null argument, a sequence that is not `pool`'s, or a failed device copy;
 * IGNIS_SEQ_ERR_SHARED_PREFIX for a target that claims a shared prefix, whose
 * leading pages are not its own to overwrite (P4-10, GitHub #126). */
int32_t ignis_seq_restore(struct ignis_seq_pool *pool, struct ignis_seq *seq, const void *src,
                           uint64_t src_bytes);

/* --- device prefix reuse (P4-10, GitHub #126, ADR 0024) ------------------
 *
 * Two mechanisms, one description. **KV pages are read-only history**, so a
 * shared prefix hands a second sequence the *same physical pages*: the leaf
 * owns them and their refcount, both sequences' block-table rows address
 * them, and they return to the pool when the last holder releases. **Mutable
 * sections are cloned device-to-device** -- the GDN recurrent state, the
 * conv taps and the penalty-count row -- through the same state-section
 * table the snapshot path walks (kernel/include/ignis_seq_sections.h). No
 * step of it crosses PCIe.
 *
 * Sharing pages *without* cloning the mutable sections would save nothing,
 * because prefill has to traverse every layer to produce them. That is
 * recorded here so it is not re-proposed as an optimization.
 *
 * These four are entry points rather than fields on an options struct, which
 * ADR 0016 would otherwise ask for. The carve-out and its reasoning are
 * recorded in ADR 0024 ("Lifetime calls are entry points"), not here.
 */

struct ignis_seq_prefix_stats {
  /* Tokens of history the prefix covers. Always a whole number of KV pages. */
  uint32_t tokens;
  /* Physical KV pages the prefix owns -- charged to the pool exactly once,
   * however many sequences hold it. */
  uint32_t pages;
  /* Live holders: the caller's own handle counts as one, and every sequence
   * allocated against it as one more. The pages return to the pool when this
   * reaches zero. */
  uint32_t refcount;
  /* Device-resident bytes of the cloned (mutable) state image: the GDN
   * recurrent state, the conv taps and the penalty-count row. Paid once per
   * prefix, not per claimant. */
  uint64_t clone_image_bytes;
  /* Claims served (ignis_seq_alloc_shared calls that cloned from this
   * prefix), and the wall time the most recent clone took, in microseconds
   * -- the device-to-device cost ADR 0024 asks to be measured rather than
   * assumed. 0 until the first claim. */
  uint64_t clone_count;
  double last_clone_micros;
};

/* Publish `seq`'s first `prefix_tokens` tokens of history as a shared prefix.
 *
 * `prefix_tokens` must be a whole number of KV pages and must be exactly
 * where `seq` stands: the mutable state cloned to a claimant is the state at
 * the prefix's end, and a sequence that has already run past it no longer
 * has that state to give. So this is called at the chunk boundary that lands
 * on the prefix, not at the end of a prompt.
 *
 * `seq` keeps serving: after the call its own row addresses the prefix's
 * pages for the head and its own pages for everything it writes from here
 * on, and it holds one reference to the prefix like any other claimant.
 *
 * Returns 0 and a handle in `*out_prefix`, which the caller releases with
 * ignis_seq_prefix_release (that handle is one reference of its own, so the
 * prefix outlives `seq`). Returns -1 (see ignis_seq_last_error) on a null
 * argument, a sequence that is not `pool`'s, a `prefix_tokens` of zero, not
 * page-aligned, or beyond what `seq` has written, a sequence that already
 * holds a shared prefix, or an exhausted pool;
 * IGNIS_SEQ_ERR_NOT_AT_BOUNDARY when `seq` is mid-chunk or its frontier is
 * not `prefix_tokens`. `seq` and the pool are unchanged on every failure. */
int32_t ignis_seq_prefix_publish(struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                  uint32_t prefix_tokens,
                                  struct ignis_seq_prefix **out_prefix);

/* Allocate a sequence that claims `prefix`: its first `prefix_tokens` pages
 * are the prefix's own physical pages (shared, not copied, not zeroed), the
 * rest is a fresh zeroed reservation of its own, and its mutable state is a
 * device-to-device clone of the prefix's.
 *
 * The returned sequence stands exactly where the publisher stood: same
 * frontier, same pending token, same GDN state, same penalty counts. It
 * prefills its own tail from `prefix_tokens` onwards and never writes a
 * shared page -- the prefix is whole pages, so its first write lands on the
 * first page it owns.
 *
 * `context_tokens` is the whole reservation, prefix included, and must leave
 * room for at least one page of its own. Returns 0 and a handle in
 * `*out_seq`; returns -1 (see ignis_seq_last_error, nothing allocated) on a
 * null argument, a prefix that is not `pool`'s, a `context_tokens` that
 * leaves no page of its own, or pool exhaustion (no free slot, or not enough
 * free KV pages). */
int32_t ignis_seq_alloc_shared(struct ignis_seq_pool *pool, uint32_t context_tokens,
                                struct ignis_seq_prefix *prefix, struct ignis_seq **out_seq);

/* Release the caller's handle on `prefix`. The prefix itself lives while any
 * sequence still holds it; its pages return to the pool when the last holder
 * releases. A NULL `prefix` is a no-op. */
void ignis_seq_prefix_release(struct ignis_seq_pool *pool, struct ignis_seq_prefix *prefix);

/* A live prefix's size, holders and measured clone cost. Returns 0 on
 * success, -1 on a null argument. */
int32_t ignis_seq_prefix_stats(const struct ignis_seq_prefix *prefix,
                                struct ignis_seq_prefix_stats *out_stats);

/* The message from the most recent failing call on this thread
 * (thread-local; overwritten by the next call; empty string if none failed
 * yet). Never NULL. */
const char *ignis_seq_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_SEQ_H */
