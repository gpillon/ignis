/* ignis kernel leaf: Flash-Next's expert residency, the device side -- OURS (spec
 * flash-next/03, GitHub #301; ADR 0030, ADR 0043, ADR 0044).
 *
 * Every expert projection lives in one pinned, mapped host pool for the life of the model; a
 * fixed set of them lives in eight VRAM slot pools, one per K class (two projection shapes x four
 * K), replaced by recency; prefill misses that find no free slot pass through a staging ring and
 * evict nothing. Residency fills each layer's slot table (`struct ignis_moe_slot`, the table the
 * MoE expert ops read) before the expert op runs, on the same stream, with no host round trip:
 * the miss path is an SM-driven copy from the mapped pool (measured at 88-112% of the copy
 * engine, docs/findings/2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md), so a
 * decode round stays one CUDA graph.
 *
 * The policy is crates/core/src/residency/policy.rs (`ResidencyModel::step`), exactly: a layer
 * step stamps every selected projection with the logical clock; a miss takes a free slot of its
 * class, else (decode) the unpinned slot with the smallest (stamp, key), else (prefill) the
 * staging ring; the next layer's lookahead candidates, rank-ordered, are prefetched the same way,
 * a decode prefetch held to a per-step byte budget that grows with the step's lookahead rows (a
 * candidate that would pass it is skipped) and dropped when no slot is unpinned, a prefill
 * step's taken unbudgeted to its own width. A key is (layer * experts + expert) * 2 + projection;
 * it is also the entry's index in the concatenated slot tables. kernel/tests/
 * test_residency_trace.cu holds this implementation to the policy model's outcomes.
 *
 * One departure in *when*, not *what*: the policy releases a layer's staged projections at the
 * end of its step; here the release (their table entries back to ABSENT) happens at the start of
 * the next step, because the expert op reads them after `ignis_residency_step` returns. When the
 * next step is another layer, the released keys are not its own; when it is the same layer (a
 * forward restarted at that layer), the step treats them as released before it classifies, as
 * the policy would. No outcome differs either way.
 *
 * Slot-table rule: a projection that is neither resident nor staged for its layer has the entry
 * `{NULL, 0}` (ignis_moe.h's ABSENT): at creation, after an eviction (written in the same step
 * that hands the slot on) and after its staging is released. The expert ops trap on a selected
 * ABSENT entry in every build, so a stale address is never read as another projection's bytes.
 *
 * Every structure belongs to one `struct ignis_residency`, created at load and freed with the
 * model (no process-wide state; phase 2's model switch is a reload). A step allocates nothing,
 * never synchronizes the host and is capturable in a CUDA graph: the resolve kernels and the
 * demand copy run on the caller's stream; prefetch copies run on a stream residency owns,
 * forked from the caller's stream after the demand copy and joined back at the start of the
 * next step (a split step forks after its demand resolve and runs its prefetch resolve there
 * too; its prefetch copy still waits for the demand copy). A step with a lookahead therefore
 * leaves a fork open until the next step, so call `ignis_residency_join` on the stream (a)
 * before beginning a capture, (b) before ending a capture that stops after such a step, and (c)
 * before launching a captured graph after eager steps. A full forward needs none of it: its last layer has no lookahead and joins the one
 * before.
 *
 * Return 0 on success, -1 on a refused argument or a CUDA error; the reason is in
 * `ignis_residency_last_error()`.
 */
#ifndef IGNIS_RESIDENCY_H
#define IGNIS_RESIDENCY_H

#include <stddef.h>
#include <stdint.h>

#include "ignis_moe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Classes in crates/core's KClass order: gate/up at K = 2, 2.5, 3, 4, then down at the same. */
#define IGNIS_RESIDENCY_CLASSES 8
#define IGNIS_RESIDENCY_DECODE 0
#define IGNIS_RESIDENCY_PREFILL 1
/* `prefetch_budget_one_row_bytes` meaning "no budget". */
#define IGNIS_RESIDENCY_NO_BUDGET UINT64_MAX

struct ignis_residency_desc {
  uint32_t layers;
  uint32_t experts;                                  /* <= IGNIS_MOE_EXPERTS */
  uint32_t capacity[IGNIS_RESIDENCY_CLASSES];        /* slots per class (< 2^20) */
  uint64_t record_bytes[IGNIS_RESIDENCY_CLASSES];    /* one record, a multiple of 16 */
  uint32_t max_tokens;          /* most tokens a step serves: decode lanes or the prefill chunk */
  uint32_t lookahead_width;     /* W: experts taken from each lookahead row; 0 = none */
  uint32_t prefill_lookahead_width; /* the same for a prefill step's rows (<= W); 0 = none */
  uint64_t prefetch_budget_one_row_bytes; /* a one-row decode step's prefetch budget, or
                                           * IGNIS_RESIDENCY_NO_BUDGET */
  uint64_t prefetch_budget_per_row_bytes; /* added per further row of a decode step's lookahead
                                           * (a lane, or a verify round's column); the sum
                                           * saturates to no budget (GitHub #306) */
  uint64_t staging_half_bytes;  /* one half of the prefill staging ring (>= the heaviest layer;
                                 * a multiple of 16) */
  uint64_t host_pool_bytes;     /* the pinned expert pool */
  uint32_t copy_blocks;         /* the copy kernel's grid; 0 = 16 */
  uint32_t report;              /* 1: keep each step's outcome for ignis_residency_last_report */
};

#ifdef __cplusplus
static_assert(sizeof(struct ignis_residency_desc) == 160 &&
                  offsetof(struct ignis_residency_desc, prefill_lookahead_width) == 112 &&
                  offsetof(struct ignis_residency_desc, prefetch_budget_one_row_bytes) == 120 &&
                  offsetof(struct ignis_residency_desc, prefetch_budget_per_row_bytes) == 128 &&
                  offsetof(struct ignis_residency_desc, copy_blocks) == 152,
              "ignis_residency_desc drifted from crates/core/src/residency/device.rs ResidencyDesc");
#endif

/* The device bytes a residency of `desc` reserves at load (ADR 0030 plan lines), each part
 * rounded up to 256 bytes: the class pools, the staging ring, and everything else (slot tables,
 * the residency map, the LRU stamps, the copy jobs, the lookahead scratch, counters, report). */
struct ignis_residency_plan {
  uint64_t pools;
  uint64_t staging;
  uint64_t tables;
  uint64_t total;
};
int32_t ignis_residency_plan_bytes(const struct ignis_residency_desc *desc,
                                   struct ignis_residency_plan *plan);

struct ignis_residency;

/* Creates a residency with every pool empty and every slot-table entry ABSENT, and allocates
 * the pinned, mapped host pool (`host_pool_bytes`), which the loader then fills. `k2` is the K
 * map, `[layers][experts][2]` (4, 5, 6 or 8), and `pool_offsets` each record's offset in the
 * host pool, same order (the binder's expert index); a record of class c is
 * `record_bytes[c]` bytes and 16-byte aligned. */
int32_t ignis_residency_create(const struct ignis_residency_desc *desc, const uint8_t *k2,
                               const uint64_t *pool_offsets, struct ignis_residency **out);

/* Frees everything `create` reserved; waits for its work first. NULL is a no-op. */
void ignis_residency_free(struct ignis_residency *r);

/* The pinned host pool (host address), `host_pool_bytes` long. */
void *ignis_residency_host_pool(struct ignis_residency *r);

/* Layer `layer`'s slot table: `experts * 2` device entries indexed `expert * 2 + projection`,
 * what ignis_moe_experts_{decode,prefill} take as `slots`. */
const struct ignis_moe_slot *ignis_residency_slot_table(struct ignis_residency *r,
                                                        uint32_t layer);

/* The optional warm start (spec 03): fills each pool from `keys`, hottest first, up to its
 * capacity, the hottest admitted the most recent. Synchronous; only before the first step.
 * Writes how many were admitted to `*admitted`. */
int32_t ignis_residency_warm_start(struct ignis_residency *r, const uint32_t *keys, uint32_t n,
                                   uint32_t *admitted);

/* One layer step. `ids` is the router's selection, device int32 `[tokens][IGNIS_MOE_TOP_K]`
 * (decode: one row per lane; prefill: one per token of the chunk). `lookahead_logits` is the
 * next layer's router applied to this layer's MoE input, device fp32 `[tokens][experts]` (the
 * `logits` scratch of ignis_moe_router), or NULL for none (the last layer, or no prefetch):
 * residency ranks each row's top `lookahead_width` experts itself (`prefill_lookahead_width` in
 * a prefill step; its width throughout below), as the router ranks (logits rounded to BF16,
 * ties to the lower id). A step whose width is 0 has no lookahead. `phase` is
 * IGNIS_RESIDENCY_DECODE or _PREFILL. */
int32_t ignis_residency_step(struct ignis_residency *r, uint32_t layer, uint32_t phase,
                             const int32_t *ids, uint32_t tokens,
                             const float *lookahead_logits, void *stream);

/* The same step with the lookahead already ranked: device int32 `[rows][stride]`, best first,
 * -1 for a hole; each row's first width (the step's) non-negative entries count. Candidates are
 * taken in rank order: every row's first, then every row's second, ..., a repeat kept where it
 * first appears. NULL for none. */
int32_t ignis_residency_step_ranked(struct ignis_residency *r, uint32_t layer, uint32_t phase,
                                    const int32_t *ids, uint32_t tokens, const int32_t *lookahead,
                                    uint32_t rows, uint32_t stride, void *stream);

/* The same step split in two, so its lookahead -- and the router that computes it -- runs beside
 * the expert op instead of before it. `ignis_residency_step_demand` resolves the selection and
 * queues the demand copy on `stream`: what the expert op needs. With `lookahead_stream` non-NULL
 * and a next layer to look at, it then forks residency's prefetch stream from `stream` and
 * writes it to `*lookahead_stream` (else NULL): the caller launches the lookahead's inputs there
 * (the next layer's router on this layer's MoE input) and then calls
 * `ignis_residency_step_prefetch` (logits, ranked as `ignis_residency_step` ranks them) or
 * `_prefetch_ranked` (as `ignis_residency_step_ranked` takes them), which place the prefetches
 * on that stream and copy them after the demand copy. The outcome is the whole step's. What the
 * caller's lookahead inputs read must stay unchanged until they ran: wait on an event recorded
 * after them before rewriting it. The next step, or `ignis_residency_join`, joins the branch
 * (also when the caller never called a prefetch half). Unlike the whole step, a ranked lookahead
 * with an id outside [0, experts) does not refuse the step: the demand half stands, nothing is
 * prefetched, and the report's status is IGNIS_RESIDENCY_STATUS_INVALID over the demand half's
 * lists and bytes (the only report whose status is not 0 and whose step changed something). */
int32_t ignis_residency_step_demand(struct ignis_residency *r, uint32_t layer, uint32_t phase,
                                    const int32_t *ids, uint32_t tokens, void *stream,
                                    void **lookahead_stream);
int32_t ignis_residency_step_prefetch(struct ignis_residency *r, const float *lookahead_logits,
                                      uint32_t tokens);
int32_t ignis_residency_step_prefetch_ranked(struct ignis_residency *r, const int32_t *lookahead,
                                             uint32_t rows, uint32_t stride);

/* GitHub #306: ignis_residency_step_demand with the router's selection made in the same launch.
 * `logits` is the router's device fp32 `[tokens][IGNIS_MOE_EXPERTS]` (ignis_moe_router_logits,
 * enqueued just before on `stream`); the step writes the selection ignis_moe_router would write
 * -- the same `ids` and `weights`, bit for bit -- and then resolves it as
 * ignis_residency_step_demand resolves `ids`. Everything else is that call's. */
int32_t ignis_residency_step_demand_routed(struct ignis_residency *r, uint32_t layer, uint32_t phase,
                                           const float *logits, int32_t *ids, float *weights,
                                           uint32_t tokens, void *stream, void **lookahead_stream);

/* Joins the prefetch copies of the last step into `stream` (see the capture rules above).
 * Steps do it themselves at their start. After a capture that failed with a fork open, the
 * wait on its dead event fails once; the fork is forgotten either way (its copies never ran),
 * so the next step runs normally. */
int32_t ignis_residency_join(struct ignis_residency *r, void *stream);

/* What residency counted since creation, 1:1 with crates/core's ResidencyCounters; `[class]
 * [phase]`, `[phase]`. Waits for residency's work. `stall_nanos` is the device time of the
 * demand copies -- what the expert op waits for -- from the first block's start to the last
 * block's end, read off %globaltimer by the copy itself; a step with no miss adds nothing. */
struct ignis_residency_counters {
  uint64_t hits[IGNIS_RESIDENCY_CLASSES][2];
  uint64_t misses[IGNIS_RESIDENCY_CLASSES][2];
  uint64_t prefetch_issued;
  uint64_t prefetch_used;
  uint64_t bytes_moved[2];
  uint64_t stall_nanos[2];
};
int32_t ignis_residency_read_counters(struct ignis_residency *r, struct ignis_residency_counters *out);

/* Slots holding a projection, per class (`out[class]`, the order of `capacity`), warm start
 * included; a filled slot stays filled, an eviction hands it on. Waits for residency's work,
 * like ignis_residency_read_counters. */
int32_t ignis_residency_read_occupancy(struct ignis_residency *r, uint32_t out[IGNIS_RESIDENCY_CLASSES]);

/* The counters and the slots in use, mirrored in host memory a host reader reads with no CUDA
 * call and no wait. */
struct ignis_residency_mirror {
  struct ignis_residency_counters counters;
  uint32_t in_use[IGNIS_RESIDENCY_CLASSES];
};

/* Mirrors into `host`, which the caller owns and keeps past ignis_residency_free: the call
 * page-locks and maps it (free unregisters it) and fills it with what residency holds now, the
 * warm start included; from then on the last layer of every step that runs writes the totals
 * there, and every demand copy its phase's stall, each 8-byte counter whole, so every value a
 * reader sees only grows. A refused step writes nothing. Only before the first step: a captured
 * graph keeps the step's arguments. */
int32_t ignis_residency_set_mirror(struct ignis_residency *r, struct ignis_residency_mirror *host);

/* The outcome of the last step of `layer` (needs `report` at creation; tests: a captured round
 * leaves one per layer). `status` is 0, 1 + the class a
 * decode step was refused for (its misses outnumber the class's free and unpinned slots; the
 * step changed nothing, so its missing projections stay ABSENT and an expert op run on it
 * traps), or
 * IGNIS_RESIDENCY_STATUS_INVALID for an expert id outside [0, experts) (the step changed nothing,
 * but for a split step's lookahead: see ignis_residency_step_demand). The six lists -- hits,
 * prefetch hits, misses, evictions, prefetches, dropped -- go to `entries`, list `l` at
 * `entries + l * capacity`, each entry a key with bit 31 set for a staging admission (misses,
 * prefetches); hits, prefetch hits and misses are in key order, the rest in processing order.
 * A list longer than `capacity` is refused. Waits for residency's work. */
#define IGNIS_RESIDENCY_LISTS 6
#define IGNIS_RESIDENCY_STATUS_INVALID 0x100u
#define IGNIS_RESIDENCY_STAGING_BIT 0x80000000u
struct ignis_residency_report {
  uint32_t status;
  uint32_t count[IGNIS_RESIDENCY_LISTS];
  uint32_t reserved;
  uint64_t bytes_moved;
};
int32_t ignis_residency_last_report(struct ignis_residency *r, uint32_t layer,
                                    struct ignis_residency_report *head, uint32_t *entries,
                                    uint32_t capacity);

/* Where the pools and the ring sit (tests: every non-ABSENT entry must point inside them). */
struct ignis_residency_layout {
  const void *pool[IGNIS_RESIDENCY_CLASSES];
  uint64_t pool_bytes[IGNIS_RESIDENCY_CLASSES];
  const void *ring;
  uint64_t ring_bytes;
};
int32_t ignis_residency_get_layout(struct ignis_residency *r, struct ignis_residency_layout *out);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_residency_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_RESIDENCY_H */
