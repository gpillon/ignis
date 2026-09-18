# ADR 0030 — device memory reserved at load, within a VRAM budget

## Status

Accepted (2026-09-17). Spec: `.scratch/vram-budget/specs/01-vram-budget.md`.
Clarifies ADR 0029 (its device pool of retained images is physical: retained
slots) and absorbs GitHub #204. Measurement and code analysis:
`.scratch/vram-analysis/REPORT.md`. Amended 2026-09-18 (owner request):
§Observability names the memory series, their sources and their performance
classes. It is the future ADR that ADR 0017 required before KV usage and
capacity could be exported, and it preserves that ADR's zero-work invariant.
Built by GitHub #216 (the exposition) and #217 (the Playground's Monitor).

## Context

On the owner's box (RTX 5090, 31.84 GiB; the desktop holds 1.6–3 GiB), Task
Manager shows ninfer at a fixed dedicated and a fixed "Shared GPU memory"
figure. ignis instead starts at the same size and then grows into shared
memory.

A live run of `make start` defaults (262K, hq-e8-2b, DFlash2, vision, prompt
reuse) served a qwen-code trace replay alongside four parallel agent
conversations. It measured:

- 27.65 GiB right after load, against ninfer's ~27.5 GiB;
- the process commit rising by 3.16 GiB in 14 minutes;
- dedicated and shared then trading places at a constant commit. That is
  Windows paging device allocations to system RAM, not new memory.

Four causes, read at source:

- **Serving allocates on the device.** Every prefix publish and every
  checkpoint capture `cudaMalloc`s its own 228 MiB mutable-state image.
  - The retained pool of ADR 0029 is a ledger that charges checkpoint images
    only.
  - Prefix images (retained, chained, still claimed) are charged to nothing.
  - They are released only by a KV page shortfall, never by device memory
    running out.
- **Windows does not fail an oversubscribed `cudaMalloc`.** WDDM pages
  instead, and `cudaMemGetInfo` cannot see it. Every step slows, erratically
  (#204).
- **KV-RAM grows blob by blob.** It makes one `cudaHostAlloc` per spill. The
  pinned memory WDDM counts as shared therefore moves with load.
- **Vision's encoder workspace is summed with the prefill scratch** instead
  of sharing it. The two are never live together; the sum costs ~1.25 GiB.

ninfer reserves everything once:

- weights;
- a sequence block with fixed checkpoint slots;
- one workspace sized to the largest of prefill, round, drafter and vision
  encode;
- a request-transient block;
- one `cudaHostAlloc` for the whole KV-RAM capacity.

Its `request_memory.h` forbids request-time device allocation outright.

## Decision

**Serving allocates nothing on the device.** Every device reservation is
made at load, and a request only takes and returns places inside them.

**One VRAM budget, whole process.** The budget is what Task Manager shows
for the process, weights and CUDA context included. It comes from one of two
modes, which are mutually exclusive:

- **Derived (default):** the memory free at start minus a **VRAM headroom**
  (`--vram-headroom-bytes`, default 1 GiB) left to the desktop and other
  processes.
- **Explicit:** `--vram-budget-bytes`. The start is refused when that much is
  not free, unless `--allow-vram-oversubscription` is given; then it warns
  and proceeds. That flag is valid only with an explicit budget, and on a
  system without sysmem fallback the warning says the allocation will fail.

**The plan is printed, and a plan that does not fit refuses the start.**
The load lays out, in order:

1. the weights;
2. the workspaces (the prefill scratch and the vision encoder share one
   `max`, and the media embedding stays its own);
3. the round and sampling buffers;
4. every lane's state;
5. the **retained slots**;
6. the KV pool, which takes the rest.

The minimum is those fixed reservations plus the KV of one sequence at
`--max-context`, and one KV page per retained slot for a checkpoint's partial
tail page (see Consequences). Not fitting it refuses the start, naming the
shortfall and the knobs that shrink it.

**Retained state lives in retained slots.** A retained slot is a place for
one mutable-state image, the size of a lane's own state, reserved at load.
- The count is `--retained-slots`, default `N_DECODE_LANES`.
- It holds the images of prompt checkpoints and of shared prefixes, whether
  retained, chained or still claimed.
- When none is free, the lowest-ranked retained state gives its slot up, in
  ADR 0023's order and spilling to KV-RAM as today.
- When nothing can give one up, the publish or capture is skipped and the
  request runs without leaving reuse behind.
- `--retained-pool-bytes` is removed.

**KV-RAM is one pinned arena held from start.** It is the vendored
`HostPinnedArena`, which ignis carries and never used: one `cudaHostAlloc` of
the whole `--kv-host-pool-bytes`, with blobs placed first-fit inside it.
- A failed allocation refuses the start.
- A blob that finds no fitting hole is treated like a full budget.
- Windows counts the arena as the process's shared GPU memory, which is what
  makes that figure fixed.

## Observability

ADR 0017 left KV usage and capacity out of the Prometheus contract because
the only values available were the interval line's placeholder zeros, and it
required "a future ADR that preserves this ADR's zero-work invariant" before
they could be exported. This is that ADR. Reserving every line at load makes
the plan a set of load-time constants, and the scheduler's own host-side
accounting — not the leaf — is the authoritative source for what is occupied.

Every series below is a **gauge in bytes, pages or slots**. No percentage is
exported: a ratio hides which of its two terms moved, and both terms are
themselves series here.

### Load-time constants

Computed once during the load that already builds the plan, then never read
again. They add no serving work of any kind.

| Metric | Type | Labels | Source |
|---|---|---|---|
| `ignis_vram_reserved_bytes` | gauge | `line=weights\|cuda_context\|workspace\|media_embedding\|sampling\|decode_graph\|verify_round\|drafter_round\|lane_state\|retained_slots\|residual` | the plan's eleven lines, the same set and spelling the `ignis.runtime.vram_plan` event carries |
| `ignis_vram_budget_bytes` | gauge | none | the budget the plan was laid out inside, derived or explicit |
| `ignis_kv_pool_pages` | gauge | none | the pool's page count, leaf-verified at load |
| `ignis_kv_page_bytes` | gauge | none | one page's bytes |
| `ignis_kv_ram_arena_bytes` | gauge | `state="capacity"` | `--kv-host-pool-bytes`, pinned whole at start |
| `ignis_retained_slots` | gauge | `state="capacity"` | `--retained-slots` |

A load that refuses to start exports nothing: there is no process to scrape.

Three of these read differently from the source column, decided while building
them (#216):

- **`ignis_retained_slots{state="capacity"}` is the scheduler's effective
  count, not the flag.** `--prompt-reuse off` without an explicit
  `--retained-slots` hands out no slots at all (#215), so the flag's value
  would be a capacity nothing can ever fill. The gauge exports
  `ConcreteScheduler::retained_slot_count()`, which is what
  `{state="in_use"}` is measured against.
- **`ignis_kv_pool_used_pages` counts retained state as well as running
  requests.** The pool's charge is one counter for the whole load, and a
  retained checkpoint's tail page and a shared prefix's pages are in it. That
  is the pool's real occupancy, and pulling them out would be the per-class
  accounting this ADR declines below. So the gauge returns to zero after the
  last request only when nothing was retained; with reuse on it settles at
  what the retained images hold, which `ignis_retained_slots{state="in_use"}`
  names.
- **`ignis_kv_ram_arena_bytes{state="used"}` is the whole host tier**, live
  evicted snapshots included, not retained blobs alone. It is the figure
  admission itself runs against, and splitting it would be a second count of
  the same arena that could disagree with the first.

### Projections of facts that already cross

The model thread already emits these, unconditionally, with metrics off. Only
the projection is new, which is exactly the exemption #190 took for the
retained-state families.

| Metric | Type | Labels | Source |
|---|---|---|---|
| `ignis_retained_slots` | gauge | `state="in_use"` | `SchedEvent::RetainedSlots` |
| `ignis_retained_slot_skips_total` | counter | `reason=publish_skipped_no_slot\|capture_skipped_no_slot\|capture_skipped_no_page` | `SchedEvent::RetainedSlotSkipped`, the same bounded spellings the log uses |

The skip counter is the one that says a load has run out of room to leave
reuse behind. Consequences already names that state ("a long tool loop stops
leaving reuse once its chain holds every slot, and it runs on without it");
until now it was invisible.

### The retained-state families say which kind of state moved

ADR 0017's six retained-state families are split by `tier` alone. A
**prompt checkpoint** and a **shared prefix** are therefore counted together,
although they are retained for different reasons, are given up in a different
order, and cost differently to bring back. An operator reading ten spills into
KV-RAM cannot tell which of the two the load is shedding.

**The six families gain `kind="checkpoint"|"prefix"`**, beside the `tier`
they already carry:

`ignis_retained_reused_tokens_total`, `ignis_retained_state_hits_total`,
`ignis_retained_state_misses_total`, `ignis_retained_state_spills_total`,
`ignis_retained_state_discards_total`, `ignis_retained_state_restores_total`.

Six families across two tiers and two kinds is twenty-four series: bounded,
constant, and the same order of magnitude the contract already carries.

`SchedEvent::RetainedState` widens to name the kind. Every site that emits it
already knows which it is holding — the type is `RetainedBlob`'s two variants,
and the emitting call has a checkpoint entry or a prefix id in hand. Like the
tick, this is a wider fact and not a new one: it is sent unconditionally, with
metrics on or off, so flag-off and flag-on stay structurally identical.

Without this split the surface would be inconsistent with itself, because
`ignis_retained_slot_skips_total` already separates the two: a publish that
found no slot is a prefix, and a capture that found no slot or no page is a
checkpoint.

`SchedEvent::StateReused` widens the same way and for the same reason. It is
the fact `ignis_retained_reused_tokens_total` — one of the six — is projected
from, so leaving it alone would have meant the consumer supplying the kind
from what kind of event had arrived. That is exactly the downstream guess this
section exists to remove, and the emitting site holds a `CheckpointClaim`
either way.

`ignis_prefix_reused_tokens_total` is untouched and keeps the sibling-prefix
meaning #190 gave it: a live sibling's claim is not retained state and takes
no `kind`.

### The interval tick carries what is occupied

The model thread sends one `TelemetryFact::Tick` after every `advance()`.
**That tick widens** to carry the scheduler's live occupancy, read from
fields it already maintains for admission:

| Metric | Type | Labels | Source |
|---|---|---|---|
| `ignis_kv_pool_used_pages` | gauge | none | the main pool's pages reserved by running requests |
| `ignis_kv_ram_arena_bytes` | gauge | `state="used"` | the host tier's used bytes |

Both are plain field reads of state the admission machine keeps anyway. The
tick is already sent, already unconditional, and its payload is the same
whether metrics are on or off — so this adds no branch, atomic, clock read,
allocation, task wake or channel operation to the inference path, and
flag-off and flag-on remain structurally identical. That is the invariant
ADR 0017 asked a future ADR to preserve.

The same widening retires the interval line's `kv_used_pct` placeholder: the
line reports the real figure, and `prefilling` stays a placeholder and stays
unexported.

### Not taken

- **Polling the leaf while serving.** `ignis_seq_pool_stats` and
  `ignis_seq_stats` report page-exact occupancy, per pool and per sequence,
  and the load already reads the pool's once to verify the plan. Reading
  either again is an FFI call on the model thread, for a number the scheduler
  already has host-side. The leaf stays a load-time oracle.
- **A live free-VRAM gauge.** It is a driver call, and on Windows the reading
  cannot see paging — the fact this ADR's Context is built on. What the plan
  reserved is the honest figure; what the driver reports while serving is
  not.
- **A `lane` label.** ADR 0017 forbids sequence and lane IDs as labels by
  name, and this ADR does not reopen that.
- **KV pages per request class.** `class` is a bounded set of two and would
  be legal cardinality, but the pool's charge is one counter for the whole
  load. Splitting it means new per-class accounting on the admission path,
  which is not a free read and not in this ADR's scope.
- **A prefix or checkpoint entry-count gauge.** On the device every retained
  image holds a slot by construction, so `ignis_retained_slots{state="in_use"}`
  already is that count. A second series would restate it and could disagree.

## Considered options

- **Keep request-time allocation and charge every image to a hard byte
  budget.** It bounds growth, but it still asks the driver for memory while
  serving. On Windows that request never fails, so the bound is only as good
  as a free-memory reading that cannot see paging. Rejected.
- **Derive only, with no explicit mode.** The owner runs other GPU software
  beside ignis and wants to state a figure and have the start refuse when it
  is not there. Rejected.
- **Retained images in a dedicated slab** rather than extra sequence-pool
  slots. The slab would be freer, for example images without the drafter
  sections. But the pool already holds lane-sized slots and copies between
  them, and the reference sizes its checkpoint slots the same way. Chosen for
  now; the owner intends to revisit it (see Consequences).
- **Lazy pinned KV-RAM** (today). Cheaper when idle, but the shared figure
  moves and every spill pays a `cudaHostAlloc`. Rejected.

## Consequences

- **`--kv-host-pool-bytes` is locked in RAM from start**, even when idle:
  8 GiB of 63.8 with the Makefile default.
- **Retained slots are sequence-pool slots**, so slot count and lane count are
  both pool geometry. The owner wants retained storage to become its own
  structure later. The lane count stays the compile-time `N_DECODE_LANES` and
  must not be written as a literal.
- **The derived budget reads free memory once, at start.** Desktop growth
  after that eats the headroom. It can still page if the desktop outgrows the
  headroom, but it can no longer grow into it.
- **The KV pool now depends on the budget.** At the Makefile defaults it
  grows from 4 GiB to about 5 GiB. `--kv-pool-bytes` stays an explicit
  override that must fit the plan.
- **ADR 0029's "byte-budgeted device pool" is realized as retained slots.**
  Its semantics (first victim, a full pool skips the capture) are unchanged.
- **The KV pool's floor grows by one page per retained slot.** A retained
  checkpoint's partial tail page is a KV page, and a claimant cannot take back
  the page of the checkpoint it claimed. The plan therefore refuses a pool
  smaller than one `max_context` sequence plus `retained_slots` pages
  (GitHub #215).
- **`--prompt-reuse off` reserves no slot by default**, so live siblings share
  no head either. Naming `--retained-slots` with reuse off gives slots to those
  siblings alone, as before #215.
- **Retained slots bound reuse per conversation, not per byte.** Every link of
  a chained prefix and every checkpoint holds a slot of its own. A long tool
  loop stops leaving reuse once its chain holds every slot, and it runs on
  without it.
- **The metric surface grows by ten series and one label set.** Every one is
  bounded and constant in cardinality: eleven plan lines, three skip reasons,
  and two `state` values reused across two families.
- **The six retained-state families double**, from twelve series to
  twenty-four, and a dashboard that sums them without aggregating away `kind`
  now double-counts nothing but reads two lines where it read one.
- **`TelemetryFact::Tick` gains a payload.** It was a bare signal; it now
  carries occupancy. Anything that widens it again pays the same structural
  proof that the inference path is unchanged.
- **`kv_used_pct` stops being a placeholder**, so the interval line's
  documented caveat about it narrows to `prefilling` alone.
