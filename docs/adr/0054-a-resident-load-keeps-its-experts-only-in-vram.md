---
status: accepted
---

# A resident Flash-Next load keeps its experts only in VRAM

An expert cache that holds the whole catalog gives every projection its slot, and the load then keeps no host copy of
any expert: on a card that holds every expert (ADR 0045's resident load) the pinned host pool, about 38 GB in one
allocation, served the misses of a few slotless projections only, 0.04-0.05% of selections measured
([the finding](../findings/2026-10-18-flash-next-experts-device-only.md)), and the owner asked for that RAM back.
The offloaded load, whose cache is below the catalog, is untouched.

- **Every projection has its slot.** A cache of at least the catalog's bytes gives each class all its projections;
  the split that models one LRU of the calibration traffic, which holds no never-selected projection and floors
  each class's share, runs only below that.
- **No host pool.** Every record is read from the artifact, a layer at a time, and admitted straight to its slot, a
  one-layer queue bounding the RAM held. The host plan's expert pool line is zero, and the layer buffers are a
  transient line of their own, held only while the experts load: the plan charges its peak, that phase or the lines
  kept while serving, whichever is more.
- **The admission is complete before the first step**: a missing projection fails the load by name. A step that
  would still miss is refused and nothing is prefetched: a miss there is an invariant broken, never a copy from a
  pool that is not there.
- **No prefill staging ring either**: nothing stages, so the VRAM plan places the ring on the offloaded branch only,
  the resident branch's KV pool takes its bytes, and the resident threshold drops by as much. A residency with no
  staging ring is refused beside a host pool.

## Considered Options

- **Free the pool after the warm start.** Less code, but the load still pins the whole pool in one allocation, the
  shape a 32-34 GiB driver ceiling refuses on a 64 GiB host, and the plan would need a transient line for it: it
  does not give the RAM to a smaller machine, which was the point.
- **Keep the pool and fix only the split.** The hit rate would reach 1.0 and the pool would serve nothing, pinned
  for no reads.
- **An exclusive pool for the offloaded path too**, RAM holding only what VRAM does not, a victim written back on
  eviction. A different question, with a cost on the critical path that a first measurement on the RTX 5090's Gen 3
  link put at 19-53% of a round. Not decided here.

## Consequences

- A machine with a 96 GB card serves Flash-Next with far less RAM, and the whole n-gram table fits beside it on
  64 GB; decode, TTFT and greedy text are unchanged or better, with no miss left.
- Whether the residency phase of a load is shorter (no second copy, pool to VRAM) or longer (one synchronous copy
  per record from pageable buffers) was not settled. Inferred from the admission bench, a cold load is disk-bound;
  admitting a layer in fewer, larger
  copies is the follow-up if the warm case matters.
