---
status: accepted
amends: [ADR-0017]
amended-by: [ADR-0035, ADR-0045]
---

# ADR 0030 — device memory reserved at load, within a VRAM budget

## Current decision

**Serving allocates nothing on the device.** Every device reservation is made
at load, and a request only takes and returns places inside them. What a load
pins in host memory is reserved at load too, as lines of a host plan that
refuses a start it cannot hold (`crates/core/src/residency/plan.rs`): the
KV-RAM arena (`--reuse-kv-host-pool-bytes`, one pinned block held from start,
blobs placed inside it), the host retained slots, the 27B's token embedding
(ADR 0062) and, under the default `host` vision residency, its vision tower
and embedding pool (ADR 0075), and Flash-Next's expert pool, n-gram hot rows
and KV-disk staging.

**One VRAM budget for the whole process**, weights and CUDA context included,
as Task Manager counts it. It is derived by default, the memory free at start
less `--vram-headroom-bytes` (1,536 MiB since 2026-10-09; 1 GiB when this was
decided), or explicit, `--vram-budget-bytes`, which is refused above free
memory unless `--vram-allow-oversubscription` (explicit budgets only) makes
that a warning. A later load in the same process (a switch, the reload after
an idle unload) adds back what the process already holds, so the CUDA context
and the 27B's per-process state are not charged twice (#324).

**The plan is printed, and a plan that does not fit refuses the start.**
`ignis.runtime.vram_plan` names its lines in a fixed order (`VramLines` in
`crates/core/src/vram.rs`): weights, CUDA context, the workspace that prefill
and the vision encoder share, the 27B trellis body's workspace and dense
prefill buffer, the media embedding pool, sampling, the decode, verify and
drafter rounds, lane state, the device retained slots, the hq residual window
of every lane and device slot, and the measured residual. Flash-Next adds
residency's fixed bytes and the expert cache (`ignis.runtime.flash_next_plan`),
refused below 12 GiB unless `--vram-allow-expert-cache-below-floor`; the image
model's plan is its weights, one arena and its condition budget. The KV pool
follows ADR 0045's policy: the rest of the budget on a resident load, 524,288
tokens reserved first when experts stream. Its floor is one
`--model-max-context` sequence plus a KV page per retained slot of either
kind; below it the start is refused, naming the shortfall and the knobs that
shrink it, except that the 27B first lowers a `max_context` its profile or
the default gave (ADR 0063's adaptation).

**Retained state lives in retained slots** fixed at load (#281). Device slots
(`--reuse-retained-device`, 0 unless a profile names some) are the plan's
`retained_slots` line and are handed out first; host slots
(`--reuse-retained-host`, by default two per decode lane on the 27B and eight
on Flash-Next) live in one pinned host block. When none is free, the
lowest-ranked retained state gives its slot up in ADR 0023's order; when
nothing can, the publish or capture is skipped and the request runs without
leaving reuse behind. With prompt reuse off, no slot is reserved unless one
is named.

**Observability.** Every memory series is a gauge in bytes, pages or slots,
never a percentage. The plan's lines and capacities are load-time constants,
slot use and skips are projections of facts the model thread already sends,
and the KV pool's and the KV-RAM arena's use ride the tick, read from the
scheduler's own host-side accounting. No series polls the leaf or the driver
while serving, so ADR 0017's zero-work invariant holds: this is the ADR it
asked for before KV usage could be exported, and it splits ADR 0017's six
retained-state families by `kind`. The series are on
[Metrics](../user/reference/metrics.md), and what each plan line holds is on
[The VRAM plan](../user/concepts/vram-plan.md).

## History

### Status

Accepted (2026-09-17). Spec: `docs/specs/vram-budget/01-vram-budget.md`.
Clarifies ADR 0029 (its device pool of retained images is physical: retained
slots) and absorbs GitHub #204. Measurement and code analysis:
`.scratch/vram-analysis/REPORT.md`. Amended 2026-09-18 (owner request):
§Observability names the memory series, their sources and their performance
classes. It is the future ADR that ADR 0017 required before KV usage and
capacity could be exported, and it preserves that ADR's zero-work invariant.
Built by GitHub #216 (the exposition) and #217 (the Playground's Monitor).
**Amended 2026-09-20 by ADR 0035** (GitHub #243): the `media_embedding` line
reserved one item's encoder output and now reserves the embedding *pool*
(`--vision-embedding-pool-mib`, defaulting to one envelope-wide item, so the
number it carries at the default is unchanged). The line's name, its place in
the layout order and its refusal rule are untouched.
**Amended 2026-09-22 by spec runtime/06** (GitHub #257, whose acceptance asks
for the reservation as "its own named line"): a twelfth plan line,
`hq_residual_window` — the hq-e8-2b residual window of every lane and every
retained slot, 34 MiB a slot, 0 on a BF16 load — placed after the retained
slots. A retained image now carries the window too; its bytes are counted on
the new line, not on `retained_slots`.
**Amended 2026-10-23 by spec 27b-16gb/05a** (GitHub #339): a thirteenth plan
line, `trellis_workspace` -- the 27B trellis body's one workspace for the dense
trellis linear (spec 27b-16gb/02), the widest call its body and head make on
the device, reserved at load at the address the round graphs bake -- placed
after `workspace`. 0 on the served NVFP4 body and on Flash-Next, whose plans
are otherwise unchanged.
**Amended 2026-11-05 by spec #503** (GitHub #509): a fourteenth plan line,
`trellis_dense` -- the dense prefill's one buffer, the INT8 copy of the trellis
body's largest matrix that every matrix of a prefill chunk is decoded into in
turn -- placed after `trellis_workspace`. 0 on a load that runs the fused
prefill; under the dense prefill `trellis_workspace` holds its INT8
activations at the chunk in place of the fused prefill's.
Amended 2026-09-18 (#216, owner decision) where building it settled three
readings this section had left to the source column — the retained slots'
capacity, what the KV pool's occupancy counts, and what the KV-RAM arena's
use covers — and to record that `SchedEvent::StateReused` widens beside
`SchedEvent::RetainedState`. Each is marked below.
**Amended 2026-09-29 by spec vram-budget/02** (GitHub #281): retained slots
come in two kinds, and the default ones live in pinned host memory — see the
amendment at the end.
**Amended 2026-10-05 by spec flash-next/03** (GitHub #301): under Flash-Next
the **expert cache** takes the rest, not the KV pool. The load names its KV
pool (a fixed line, sized from the per-lane context), and three lines follow
it: `expert_cache` (budget − every line above − the two below; floor 12 GiB,
refused below it), `prefill_staging` (two of the heaviest layer's expert
projections, so a prefill's scan-resistant stream cannot overflow it) and
`residency_tables`. The expert cache is split into eight K-class pools at
load and nothing is allocated while serving. A 27B load's plan is unchanged.
Lines and arithmetic: `crates/core/src/residency/plan.rs`.
**Amended 2026-10-07 by ADR 0045** (spec `vram-budget/03`): the KV pool line
follows the **KV pool policy** — see the amendment at the end.
**Amended 2026-10-15 by GitHub #324**: a later load in the same process adds
back the device memory the process already holds, and the image model's load
reports its plan into the memory gauges — see the amendment at the end.

### Context

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

### Decision

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
   `max`, and the media embedding pool stays its own);
3. the round and sampling buffers;
4. every lane's state;
5. the **retained slots**;
6. under hq-e8-2b, the **residual window** of every lane and retained slot
   (#257);
7. the KV pool, which takes the rest.

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

### Observability

ADR 0017 left KV usage and capacity out of the Prometheus contract because
the only values available were the interval line's placeholder zeros, and it
required "a future ADR that preserves this ADR's zero-work invariant" before
they could be exported. This is that ADR. Reserving every line at load makes
the plan a set of load-time constants, and the scheduler's own host-side
accounting — not the leaf — is the authoritative source for what is occupied.

Every series below is a **gauge in bytes, pages or slots**. No percentage is
exported: a ratio hides which of its two terms moved, and both terms are
themselves series here.

#### Load-time constants

Computed once during the load that already builds the plan, then never read
again. They add no serving work of any kind.

The series are the plan's lines, as `ignis_vram_reserved_bytes{line}` in the
plan's own spellings (the fields of `ignis.runtime.vram_plan`, then
Flash-Next's `residency` and `expert_cache` and the image model's
`condition_budget`), the budget, the KV pool's pages and page bytes, and the
capacities of the KV-RAM arena and the retained slots, with the host slots and
their block since #281. The KV pool is no line, so the budget less every line
is the pool plus its page rounding. Each series, with its meaning, is on
[Metrics](../user/reference/metrics.md), and what each line holds is on
[The VRAM plan](../user/concepts/vram-plan.md).

A load that refuses to start exports nothing: there is no process to scrape.

**The slot capacity is the scheduler's effective count, not the flag** (#216).
`--prompt-reuse off` without an explicit `--retained-slots` hands out no slots
at all (#215), so the flag's value would be a capacity nothing can ever fill —
and `{state="in_use"}` would be measured against a bound that is not its own.
The gauge exports `ConcreteScheduler::retained_slot_count()`.

#### Projections of facts that already cross

The model thread already emits these, unconditionally, with metrics off. Only
the projection is new, which is exactly the exemption #190 took for the
retained-state families.

The two series are `ignis_retained_slots{state="in_use"}`, projected from
`SchedEvent::RetainedSlots`, and `ignis_retained_slot_skips_total{reason}`,
from `SchedEvent::RetainedSlotSkipped` with the log's own bounded spellings
([Metrics](../user/reference/metrics.md)).

The skip counter is the one that says a load has run out of room to leave
reuse behind. Consequences already names that state ("a long tool loop stops
leaving reuse once its chain holds every slot, and it runs on without it");
until now it was invisible.

#### The retained-state families say which kind of state moved

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

#### The interval tick carries what is occupied

The model thread sends one `TelemetryFact::Tick` after every `advance()`.
**That tick widens** to carry the scheduler's live occupancy, read from
fields it already maintains for admission: `ignis_kv_pool_used_pages`, the
main pool's pages in use (running requests, shared prefixes and retained tail
pages together, #216), and `ignis_kv_ram_arena_bytes{state="used"}`, the host
tier's used bytes (live evicted snapshots and retained blobs together, #216),
both on [Metrics](../user/reference/metrics.md).

**Both count everything the pool and the arena hold, not one kind of claimant**
(#216). The pool's charge is one counter for the whole load, and the host
tier's is one budget for the whole tier; a retained checkpoint's tail page is
as much a KV page as a running request's, and a retained blob is as much of
the arena as an evicted snapshot. Splitting either would be the per-class
accounting this ADR declines below, and a second count of the same memory that
could disagree with the first. What follows from it: `ignis_kv_pool_used_pages`
falls to zero after the last request only when nothing was retained. With
reuse on it settles at what the retained images hold, which
`ignis_retained_slots{state="in_use"}` names — a reading of the release, not a
stale figure.

Both are plain field reads of state the admission machine keeps anyway. The
tick is already sent, already unconditional, and its payload is the same
whether metrics are on or off — so this adds no branch, atomic, clock read,
allocation, task wake or channel operation to the inference path, and
flag-off and flag-on remain structurally identical. That is the invariant
ADR 0017 asked a future ADR to preserve.

The same widening retires the interval line's `kv_used_pct` placeholder: the
line reports the real figure, and `prefilling` stays a placeholder and stays
unexported.

#### Not taken

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

### Considered options

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

### Consequences

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
  bounded and constant in cardinality: eleven plan lines (twelve since #257,
  thirteen since #339),
  three skip reasons,
  and two `state` values reused across two families.
- **The six retained-state families double**, from twelve series to
  twenty-four, and a dashboard that sums them without aggregating away `kind`
  now double-counts nothing but reads two lines where it read one.
- **`TelemetryFact::Tick` gains a payload.** It was a bare signal; it now
  carries occupancy. Anything that widens it again pays the same structural
  proof that the inference path is unchanged.
- **`kv_used_pct` stops being a placeholder**, so the interval line's
  documented caveat about it narrows to `prefilling` alone.

### Amendment (2026-09-29) — retained slots on the host (GitHub #281)

Spec: `docs/specs/vram-budget/02-retained-host.md`.

Measured with an 8-agent swarm, the device slots did little of the reuse:
eight agents need about seventeen slots, so a conversation's image was
spilled to KV-RAM before its next turn and came back from there (51 of 52
checkpoint hits). Meanwhile the eight slots and their residual window held
1.74 GiB — about 200K tokens of hq KV.

- **Two kinds of retained slot.** `--retained-device <n>` (default 0) keeps
  images in the device state arenas, as this ADR built them;
  `--retained-host <n>` (default `2 x N_DECODE_LANES`) keeps them in one
  pinned host block of `n` packed clone images, reserved at load. The
  device slots take the low indices and the scheduler hands out the lowest
  free one, so a device slot is always used first. `--retained-slots` is
  removed and refuses the start, naming both.
- **"Serving allocates nothing" still holds.** The host block is one
  `cudaHostAlloc` at pool create; a failed one refuses the start. Like the
  KV-RAM arena, Windows counts it as the process's shared GPU memory, so
  that figure grows by the block at load and stays fixed.
- **The plan.** The `retained_slots` line is the device slots' alone, and
  `hq_residual_window` counts the lanes and the device slots. The host block
  is RAM, not a plan line: `ignis.runtime.vram_plan` carries it as
  `retained_host_bytes`, beside `retained_device_slots` and
  `retained_host_slots`. The KV floor keeps one tail page per slot of either
  kind.
- **The cost moves to PCIe.** A capture into a host slot and a claim from
  one copy ~222 MiB each way, synchronized: ~15–19 ms per request, never per
  token. On the shipped build, sixteen host slots against eight device slots
  were level on wall time and tok/s on both swarm loads, with 42% more KV
  (677K tokens against 475K) and a higher later-turn TTFT median on the long
  load. See the finding
  [Retained slots on the host](../findings/2026-09-28-retained-slots-on-the-host.md).
- **Two load-time gauges** join §Observability's constants, both without a
  label: `ignis_retained_host_slots` (of `ignis_retained_slots{state="capacity"}`,
  the host ones) and `ignis_retained_host_bytes` (their pinned block).
- **The metrics' `tier="device"` keeps meaning the slot tier**, whichever
  kind of slot holds the image; KV-RAM is the other tier. Splitting the label
  by slot kind is not done here.
- **Supersedes** this ADR's "The count is `--retained-slots`, default
  `N_DECODE_LANES`" and the consequence that retained slots are
  sequence-pool slots: the device ones still are, the host ones are not.

### Amendment (2026-10-07) — the KV pool policy (ADR 0045)

Spec: `docs/specs/vram-budget/03-kv-pool-policy-and-kv-disk.md`.

- **Layout item 7, "the KV pool, which takes the rest", holds for a resident
  load only**: the 27B, and Flash-Next on a card that holds every expert.
- **Flash-Next with streamed experts reserves its pool first.** The pool is
  524,288 tokens, at most every lane's whole context, and never below the
  floor. The expert cache then takes the rest. This replaces the 2026-10-05
  amendment's pool "sized from the per-lane context", and it is the same
  `plan_vram` call for both models.
- **The floor is unchanged and now binds both models**: one `--max-context`
  sequence plus one page per retained slot.
- **`--kv-pool-bytes` is honoured on both models** and accepts a token form
  (`512Ktok`). The pages are still derived from the format.
- **The expert cache's 12 GiB floor gains an opt-in,
  `--allow-expert-cache-below-floor`**, on this ADR's
  `--allow-vram-oversubscription` pattern: the start proceeds with a WARN. The
  class-minimum refusal is not overridable.
- **"Serving allocates nothing" holds for KV-disk too.** Its 64 MiB of pinned
  staging is reserved at load, as a host-plan line on Flash-Next. Its files are
  disk, not memory, and no plan line counts them.
- **§Observability gains the disk gauge**
  `ignis_kv_disk_bytes{state="capacity"|"used"}`, beside the KV-RAM arena's. It
  renders only on a load with the tier.

### Amendment (2026-10-15) — a later load in the same process (GitHub #324)

- **The derived budget still starts from what was free before this process
  held anything.** A first load reads NVML before its context exists, as #210
  built it.
- **A later load reads NVML with the context already held**: a model switch,
  or the reload after an idle unload. ignis never resets the primary context
  (`ignis_device_destroy` frees a device's stream and event only), and after a
  27B load the process also keeps state the 27B's kernels set up once per
  process. The plan's `cuda_context` line, and the 27B's `residual` line,
  charged those bytes a second time: the 27B came back from a switch with
  667,392 KV tokens against 726,848 cold (−8.2%), and a plain reload, what an
  idle unload's is, lost as much (12,457 → 11,516 pages in `vram_reload_gpu`
  before the fix).
- **The load adds back what the process holds** (`held_before_load` in
  `ignis-runtime`): nothing before the context exists, `CUDA_CONTEXT_BYTES`
  once it does, and on a 27B plan `QWEN27B_PROCESS_STATE_BYTES` too once a 27B
  load has run. `free_at_start_bytes` is NVML's reading plus that figure, and
  `ignis.runtime.vram_plan` and `ignis.runtime.flash_next_plan` carry it as
  `process_held_bytes`. The lines, their order and the metrics are unchanged.
- **Measured** with the WDDM counter `CUDA_CONTEXT_BYTES` came from, on an
  idle-unloaded server (RTX 5090, 2026-10-15): 536,485,888 B after any 27B
  load, the same at every unload with an image model's in between; 452,595,712
  B, the context alone, after only the image model. Nothing leaks, and the
  image leaf leaves nothing behind.
- **Flash-Next adds back the context only.** Its plan has no residual line,
  and its own per-process state is not measured: an under-count costs pool,
  never more memory than the card has.
- **The image plan is unchanged.** It reads NVML after its context and charges
  none, so a 27B's state, when one has run, stays outside its budget: about
  84 MB (0.7% of the condition budget) less than in a process that loaded only
  the image model.
- **Result** at the Makefile's 27B shape: cold 11,129 pages, then image, then
  27B 11,105 (−0.2%, the desktop's own drift between the two readings), and
  `vram_loaded`'s delta within 15 MB of `allocated_at_load` on both loads.
- **Re-measured later on 2026-10-15** with the same counter, at the Makefile
  defaults: `LOAD_RESIDUAL_BYTES` is 124,474,800 B (88,028,656 B since
  2026-09-17; a cold start and a switch round trip held 123.7 and 124.5 MB
  beyond the plan's other lines), and `QWEN27B_PROCESS_STATE_BYTES`
  83,894,272 B on that build. Of the 27B's state, 6 MiB + 8 KiB is module
  code CUDA loads lazily (gone under `CUDA_MODULE_LOADING=EAGER`, which loads
  every module with the context); 74 MiB is per-context state that survives
  eager loading and is not identified. It is not the stack reservation, the
  stream-ordered or graph memory pools, freed small allocations or pinned
  host mappings (the constants' docs say how each was ruled out).
- **The image model's load reports its plan** (#324, the image model's VRAM;
  spec image-model/05): until then it reported nothing, and with it loaded
  §Observability's gauges read no load. Its plan's three lines go into
  `ignis_vram_reserved_bytes` — the weights as `weights`, the one arena its
  calls share as `workspace`, the condition budget as `condition_budget`, a
  line no text plan has (0 there, as Flash-Next's two are on the 27B) — inside
  `ignis_vram_budget_bytes`. It has no KV pool (`ignis_kv_pool_pages` 0), and
  no `cuda_context` or `residual` line: the plan reads NVML with its context
  already held and charges neither, as above. The condition budget is
  set aside, not allocated: each job's prefix is allocated from it as the job
  encodes and freed as it ends, so `ignis_condition_budget_used_bytes`
  (ADR 0017) reads what the jobs hold against the line, the image model's
  counterpart of `ignis_kv_pool_used_pages`. The job sets it with the runner
  held, and every load's reservations reset it to 0, so a switch in either
  direction leaves nothing of the other model's. The meter reads below 100% at a full budget by
  `prefix_kv_bytes(n)`: the line includes the encode's transient chunked-pass
  BF16 K/V, which `condition_bytes` (the runs alone) does not count.
