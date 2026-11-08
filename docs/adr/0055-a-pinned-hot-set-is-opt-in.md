---
status: accepted
---

# ADR 0055 — a pinned hot set of experts: a field off by default, 2 GiB in the rtx5090 profile

## Status

Accepted (2026-10-18, owner decision), with the built-in `rtx5090` profile
pinning 2 GiB for Flash-Next (below). Proposed the same day for the owner, who
had asked for the feature as an opt-in (#329 item 12, 2026-10-18: "build it as
an opt-in feature; with the field at its default the behaviour, plans, logs
and outputs are byte-for-byte today's"); the measurement is
[`docs/findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md`](../findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md),
after the study [`2026-10-18-partial-exclusive-expert-pool.md`](../findings/2026-10-18-partial-exclusive-expert-pool.md).
Amends spec flash-next/03's "every expert in pinned RAM" for an offloaded load
that asks for a pin. ADR 0054's device-only load is untouched.

## Context

An offloaded Flash-Next load (the 5090: the VRAM expert cache holds about half
the experts) keeps every projection in the pinned host pool, ~38 GB, so that a
miss copies it in. A projection that never leaves VRAM needs no host copy. The
study found that today's LRU keeps no such set for long, but that pinning the
calibration-hottest 1-2 GB anyway, outside the LRU, costs under 1% of decode
in replay. On a 64 GB host the pool is most of the RAM; the KV-RAM arena and
the n-gram hot rows compete for the rest.

## Decision

**`vram.pinned_expert_bytes` (`--vram-pinned-expert-bytes`, Flash-Next only,
changed through a reload) pins the calibration-hottest whole experts until
their bytes reach it. 0, the field's default, pins none; the `rtx5090`
profile sets `2G`.**

- **The set.** Whole experts in the warm start's order (the sidecar's
  calibration selections), taken out of the pools the cache split already
  made (`pin_hot_experts`), as the study measured it. Not a new split of the
  rest. Each class keeps for its LRU the slots of the widest decode step the
  load runs: the class minimum of its decode rows, which under MTP is a
  verify round's lanes × (drafts + 1), up to the row budget, not the lanes
  (review 2026-10-18). The split gives every class that same minimum first,
  the plan's split and the load's alike: one rule, so no class is split
  below it, pinned or not (second review 2026-10-18; see Consequences). A
  budget the walk cannot reach while keeping them is refused at plan time,
  before anything loads, naming the most it can pin and where the budget was
  set: the flag, the environment variable, the file key or the profile.
- **One computation.** The server's plan (the host pool's line, the event)
  and the load build the pools and the pinned set from the same call with
  the same options (`flash_next::residency_pools`).
- **Host.** The pool is laid out without the pinned records, which have no copy
  in RAM; they stream from the artifact to their VRAM slots before the pool is
  filled. The host plan's `expert_pool` line drops their bytes; the buffers
  they stream through are its `load_staging` line, held only while the experts
  load.
- **Device.** A pinned key has a slot past its class's LRU slots. A selected
  pinned key is a hit and nothing else: never stamped, evicted, staged,
  prefetched or copied. The CPU policy model pins the same way
  (`ResidencyModel::pin`), and the device equals it step for step on random
  traces and on the committed recorded trace replayed with pins (spec
  flash-next/03 acceptance 4). With nothing pinned the resolve decides as
  before.
- **Device-only loads** (ADR 0054) ignore the field: they keep no expert in
  RAM already. The `rtxpro6000` profile sets nothing.
- **Observability.** A load that asked for a pin logs
  `ignis.runtime.flash_next_pinned_experts` (bytes, experts, per class, the
  LRU's slots left). `ignis_expert_cache_slots{state="capacity"}` counts the
  LRU's slots only; the pinned slots are in that event, not in a series of
  their own. The family's HELP text says so, and the Monitor's expert table
  heads its column "LRU slots in use". At a pin of 0 (`--profile none`, or
  the field set to 0) the HELP line is the one change to the `/metrics` text,
  whose series and values are unchanged. Under the `rtx5090` profile's `2G`
  the series are the same, but a Flash-Next load's `state="capacity"` is
  lower by the pinned slots (1,824 projections, mostly gate/up and down
  K = 4: the gate/up K4 LRU keeps 303 of its 1,075 slots).

## Considered options

- **On by default at 1-2 GiB.** Taken as a profile value, not as the field's
  default: the field's own default stays 0, and the built-in `rtx5090` profile
  sets `2G` for Flash-Next (the owner, 2026-10-18; the field is Flash-Next-only,
  so the 27B passes the profile's value over). Measured on the 5090: 2.149 GB
  of the host pool's RAM back for 0.1-0.2% of one-lane and 0.2-1.1% of
  three-lane decode (two sessions), greedy text unchanged; `1G` cost nothing
  measurable, and `4G` 4.296 GB for 4.3% of one-lane and 5.9% of three-lane
  decode, above the study's 3.8-5.3% (the owner chooses `2G` or `4G` from
  that). At a pin of 0 (the field's default: `--profile none`, or the field
  set to 0) the only visible difference from before the field existed is the
  field itself: `--help` lists `--vram-pinned-expert-bytes` with 0, and
  `config print` and `GET /v1/config` list it (ADR 0046: every limit is a
  field), at 0 under `--profile none` and at `2G` for Flash-Next under
  `rtx5090`. Under that profile a default Flash-Next load does differ: it
  logs the event, its `state="capacity"` is lower by the pinned slots, its
  host plan's `expert_pool` line is 2.149 GB smaller, and its experts take
  ~2 s longer to load (Consequences).
- **Pin what the LRU never evicts.** Small (0.4-0.5 GB) and still shrinking
  with the length of the trace, and not known at load.
- **A fully exclusive pool** (RAM holds only what VRAM does not, victims
  written back). ~18 GB back for 20-70% of a round on this Gen 3 link (the
  study's inference). Not decided here.

## Consequences

- The host pool and the process's RAM are smaller by exactly the pinned bytes
  (1.075 GB at `1G`, 2.149 GB at `2G`, 4.296 GB at `4G` on the served
  artifact); the plan's residency-phase peak adds 0.28-0.87 GB of stream
  buffers.
- Greedy output is unchanged; decode measured within noise at `1G`, 0.2-1.1%
  slower at three lanes at `2G` and 5.9% at `4G` (4.3% at one lane). The pin
  takes K = 4 slots mostly, so the cost climbs past 2 GiB: at `4G` the gate/up
  K4 LRU is down to the three-lane minimum of 78 slots.
- A default Flash-Next load under the `rtx5090` profile pins 2 GiB: it logs
  the event, its host plan's `expert_pool` line is 2.149 GB smaller and its
  `load_staging` line 0.50 GB while the experts load. At 0 (`--profile none`,
  or the field set to 0) no event and no plan line changes.
- A load with a pin takes ~2 s longer to load its experts; under the
  `rtx5090` profile every default Flash-Next load pays it.
- The cache's split gives every class the widest decode step's slots first
  (second review 2026-10-18), where it gave the lanes' before. That changes
  the split of an MTP load at any pin, 0 included: at three lanes and 3
  drafts a verify round is 8 rows, 208 slots a class where the lanes gave 78.
  A correctness fix, not a tuning: the old split could leave a class below
  what one verify round selects and looks ahead, which no pin refused. A
  load with MTP off, the `rtx5090` profile's, splits as before.
- No live resize of the expert cache may evict a pinned projection: it has no
  copy to come back from. Pools are fixed per load (ADR 0030), so today none
  does.
