---
status: accepted
---

# Device memory reserved at load, within a VRAM budget

Serving allocates nothing on the device: every device reservation is made at load, inside one VRAM budget for the
whole process, and a request only takes and returns places inside them. Requests used to allocate as they ran, every
prefix publish and every checkpoint capture an image of mutable state of its own, and Windows does not fail an
oversubscribed device allocation: WDDM pages it to system RAM, where `cudaMemGetInfo` cannot see it, and every step
slows, erratically. The reference engine reserves everything once and forbids a device allocation at request time.

- **One budget for the whole process**, weights and CUDA context included, as Task Manager counts it: derived, by
  default, the memory free at start less a headroom left to the desktop and other processes
  (`--vram-headroom-bytes`), or explicit (`--vram-budget-bytes`), never both. An explicit budget above free memory
  refuses the start unless `--vram-allow-oversubscription`, valid only beside an explicit budget, makes that a
  warning, which says the allocation will fail where the system cannot page device memory. A later load in the same
  process, a model switch or the reload after an idle unload, adds back what the process already holds: the budget
  still starts from what was free before the process held anything, so the CUDA context and the 27B's per-process
  state are not charged twice (ADR 0092 says where both figures come from).
- **The plan is printed, and a plan that does not fit refuses the start.** The reservations are named lines in a
  fixed order (`VramLines` in `crates/core/src/vram.rs`), printed as `ignis.runtime.vram_plan`. Flash-Next's plan
  goes through the same call and adds residency's lines and its expert cache (`ignis.runtime.flash_next_plan`),
  refused below the cache's floor unless `--vram-allow-expert-cache-below-floor` (ADR 0045). The KV pool follows ADR
  0045's policy, the rest of the budget on a resident load and reserved first when experts stream, and a named
  `--vram-kv-pool-bytes` is an override that must fit the plan. The pool's floor, on both models, is one
  `--model-max-context` sequence plus one KV page per retained slot of either kind: a retained checkpoint's partial
  tail page is a KV page, and a claimant cannot take back the page of the checkpoint it claimed. Below the floor the
  start is refused, naming the shortfall and the knobs that shrink it, except that the 27B first adapts what its
  profile or the defaults gave, its `max_context` among them (ADR 0063).
- **What a load pins in host memory is reserved at load too**, as lines of a host plan that refuses a start it cannot
  hold (`crates/core/src/residency/plan.rs` names them, KV-disk's pinned staging among them on a load with the
  tier). KV-disk's files are disk, not memory, and no plan line counts them.
- **KV-RAM is one pinned arena held from start**: the whole `--reuse-kv-host-pool-bytes` in one pinned allocation,
  blobs placed inside it. A failed allocation refuses the start, and a blob that finds no fitting hole is treated
  like a full budget. The arena is locked in RAM even when idle, and Windows counts it as the process's shared GPU
  memory, which is what makes that figure fixed.
- **Retained state lives in retained slots** fixed at load, each a place for one mutable-state image the size of a
  lane's own state: the images of prompt checkpoints and of shared prefixes, whether retained, chained or still
  claimed. They are ADR 0029's device pool of retained images made physical, its semantics unchanged (first victim,
  a full pool skips the capture). Device slots (`--reuse-retained-device`) are the plan's `retained_slots` line and
  are handed out first; host slots (`--reuse-retained-host`) sit in one pinned host block reserved at load, whose
  failed allocation refuses the start and which Windows counts as shared GPU memory, as it does the arena. When no
  slot is free, the lowest-ranked retained state gives its slot up, in ADR 0023's order, spilling to KV-RAM; when
  nothing can, the publish or capture is skipped and the request runs without leaving reuse behind. With prompt
  reuse off no slot is reserved unless one is named, so live siblings share no head either, and named slots go to
  those siblings alone.
- **Every memory series is a gauge in bytes, pages or slots, never a percentage**, bounded and constant in
  cardinality: a ratio hides which of its two terms moved, and both terms are series. The plan's lines and
  capacities are load-time constants, slot use and skips are projections of facts the model thread already sends,
  and what is occupied of the KV pool, the KV-RAM arena and KV-disk rides the tick the model thread sends after every
  step, read from the scheduler's own host-side accounting, not from the leaf. No series polls the leaf or the driver
  while serving, so ADR 0017's zero-work invariant holds, and whatever widens the tick again owes the same proof
  that the inference path is unchanged.

## Considered Options

- **Request-time allocation charged to a hard byte budget**: it bounds growth but still asks the driver for memory
  while serving, and on Windows that request never fails, so the bound is only as good as a free-memory reading that
  cannot see paging.
- **Derive only, with no explicit mode**: the owner runs other GPU software beside ignis and wants to state a figure
  and have the start refuse when it is not there.
- **Lazy pinned KV-RAM**, a pinned allocation per spilled blob: cheaper when idle, but the shared figure moves with
  load and every spill pays the allocation.
- **The vision encoder's workspace summed with the prefill scratch**: the two are never live together, so they
  share one line sized to the larger, where the sum cost about 1.25 GiB; the media embedding pool stays a line of
  its own.
- **Retained images in a dedicated slab**, not in extra sequence-pool slots: freer (images without the drafter's
  sections), but the pool already holds lane-sized slots and copies between them, and the reference sizes its
  checkpoint slots the same way. Slots are chosen for now: the owner means retained storage to become a structure
  of its own.
- **Device retained slots alone**: eight agents need about seventeen, so with eight an image was spilled to KV-RAM
  before its conversation's next turn, while the slots and their residual window held the VRAM of about 200K
  tokens of KV.
  Sixteen host slots were level with them on wall time and tokens per second with 42% more KV, at a higher
  later-turn TTFT median on the long load and a PCIe copy per capture and per claim, paid per request and never per
  token ([finding](../findings/2026-09-28-retained-slots-on-the-host.md)).
- **Polling the leaf while serving** for its page-exact occupancy: an FFI call on the model thread for a number the
  scheduler already has host-side. The leaf is read once, at load, to verify the plan.
- **A live free-VRAM gauge**: a driver call, and on Windows the reading cannot see paging. What the plan reserved
  is the honest figure; what the driver reports while serving is not.

## Consequences

- **What a load holds on the device is a line of the plan, whoever adds it**, 0 on a load without it: the hq
  residual window of every lane and device slot, the trellis body's workspace and its dense prefill buffer, the
  mixed rounds' graphs (ADR 0106), which the driver keeps after a teardown, so a reload adds that line back too,
  and the vision embedding pool (ADR 0035), under the name, the place and the refusal rule of the one item's output
  the `media_embedding` line once held: a pool the budget cannot hold fails the load by name.
- **On Windows the W4A4 TMA products' descriptor slots are such a line, `tma_descriptors`.** There the NVFP4 body's
  products take their tensor maps through a pointer, and upstream's launchers allocated a device copy of them from
  the device's memory pool before every product: an allocation the plan could not state, whose first one cost the
  first request after a load about 110 ms on the RTX 5090. With the slots a first prefill allocates nothing a later
  one does not, and the bytes the products write are unchanged.
- **The image model's condition budget is set aside, not allocated**: each job's prefix is allocated from it as the
  job encodes and freed as it ends.
- **The derived budget reads free memory once, at start.** A desktop that grows afterwards eats the headroom and can
  still make the card page; ignis no longer grows into it.
- **Retained slots bound reuse per conversation, not per byte.** Every link of a chained prefix and every
  checkpoint holds a slot of its own, so a long tool loop stops leaving reuse once its chain holds every slot, and
  runs on without it; `ignis_retained_slot_skips_total` is the series that shows it.
- **Device retained slots are sequence-pool slots**, the host ones are not: device slot count and lane count are
  both pool geometry, and the lane count is never written as a literal (`N_DECODE_LANES`).
