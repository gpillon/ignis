# ADR 0030 — device memory reserved at load, within a VRAM budget

## Status

Accepted (2026-09-17). Spec: `.scratch/vram-budget/specs/01-vram-budget.md`.
Clarifies ADR 0029 (its device pool of retained images is physical: retained
slots) and absorbs GitHub #204. Measurement and code analysis:
`.scratch/vram-analysis/REPORT.md`.

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
- **Retained slots bound reuse per conversation, not per byte.** Every link of
  a chained prefix and every checkpoint holds a slot of its own. A long tool
  loop stops leaving reuse once its chain holds every slot, and it runs on
  without it.
