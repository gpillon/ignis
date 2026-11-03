# ADR 0054 — a resident Flash-Next load keeps its experts only in VRAM

## Status

Accepted (2026-10-18, owner decision). Proposed the same day, after the measurement in
[`docs/findings/2026-10-18-flash-next-experts-device-only.md`](../findings/2026-10-18-flash-next-experts-device-only.md);
the owner asked for it ("se entrano tutti gli esperti in VRAM li togliamo dalla RAM e liberiamo
quello spazio"). Amends [ADR 0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md)'s
resident branch; its offloaded branch, and spec flash-next/03's residency on it, stand.

## Context

Expert residency (spec flash-next/03) keeps every expert projection in one pinned, mapped host
pool for the life of the load, and a VRAM cache of some of them: the 5090's shape, where the
cache holds about half the catalog and a miss is an SM-driven copy from the pool. ADR 0045 made
a card that holds every expert "resident": the cache gets the whole catalog's bytes, the KV pool
the rest. The host pool stayed whole: ~38 GB pinned, in one `cudaHostAlloc`.

On an RTX PRO 6000 the cache split, which models one LRU of the calibration traffic, gave no
slot to 244 projections at one lane and to 1,944 at eight. It holds no never-selected projection
and floors each class's share, so 1.2 GB of the cache stayed unassigned, and the warm start
admits selected experts only. The 38 GB pool served the misses on those projections alone:
0.04-0.05% of selections measured (the plan's model put it at 0.6% at eight lanes).

## Decision

**A cache that holds the whole catalog gives every projection its slot, and the load then keeps
no host copy of any expert.**
- `plan_expert_cache`: a cache of at least `ExpertCatalog::total_bytes` gives each class all its
  projections, expected hit rate 1.0. The LRU split runs only below it.
- The residency is device-only (`host_pool_bytes` 0 in its desc): no host pool. Every record is
  read from the artifact a layer at a time, the reads the pool fill made, into a buffer each, and
  admitted straight to its slot. A one-layer queue bounds the RAM held.
- The admission must be complete before the first step: a missing projection fails the load by
  name. A step that would still miss is refused (`IGNIS_RESIDENCY_STATUS_NO_HOST`), its
  projections left ABSENT so an expert op traps, and nothing is prefetched. A miss there is an
  invariant broken, never a copy from a pool that is not there.
- The host plan's `expert_pool` line is 0 on such a load. The layer buffers are a transient line
  of their own, `load_staging`, held only during the residency phase: the load frees them before
  it reads the n-gram hot rows, so the plan charges its peak, that phase or the lines kept while
  serving, whichever is more (#329 item 3).

## Considered options

- **Free the pool after the warm start.** Less code. But the load still pins 38 GB in one
  allocation, the shape a 32-34 GiB driver ceiling refuses on a 64 GiB host, and the plan would
  need a transient line for it. Rejected: it does not give the RAM to a smaller machine, which
  was the point.
- **Keep the pool and fix only the split.** The hit rate would reach 1.0 and the pool would serve
  nothing, 38 GB pinned for no reads. Rejected.
- **An exclusive pool for the offloaded path too** (RAM holds only what VRAM does not, a victim
  written back on eviction). A different question, with a cost on the critical path the first
  measurement on the 5090's Gen 3 link puts at 19-53% of a round. Not decided here.

## Consequences

- A machine with a 96 GB card serves Flash-Next with ~9 GB of RAM available instead of ~46 (the
  layer buffers are then the plan's peak), and the whole n-gram table fits beside it on 64 GB
  (from ~37 GB available). Both figures are arithmetic over the plan's lines, not measured.
- Decode, TTFT and greedy text are unchanged or better (no miss left), measured on the card.
- The load reads the artifact as before. Whether the residency phase is shorter (no second copy,
  pool to VRAM) or longer (one synchronous copy per record from pageable buffers) was not settled
  by a load of each with the page cache warm. The admission bench measures the copy alone: ~3.5-3.7
  GB/s on the 5090 (Gen 3 link, a lower bound for the card), above the network volume's ~1 GB/s. So,
  inferred: a cold load is disk-bound, and a warm one takes at least ~37.8 / 3.6 = ~10 s to admit.
  Admitting a layer in fewer, larger copies (4.9-12.6 GB/s measured) is the follow-up if the warm
  case matters.
- The offloaded path (cache below the catalog) is untouched.
- A device-only load reserves no prefill staging ring either (#329 item 2): nothing stages, so
  the residency is created with `staging_half_bytes` 0 (refused beside a host pool, as before)
  and the VRAM plan places the ring on the offloaded branch only. The resident branch's KV pool
  takes its 1,574,895,616 bytes (two of the trellis-a25 artifact's heaviest layer): ~5,825 pages,
  ~372,800 tokens at hq-e8-2b, +3.0% on an 8-lane rtxpro6000 plan the campaign logged (191,888
  pages, MTP off). Arithmetic from the plan, not measured on the card. The resident threshold
  drops by the same bytes.
