# 03 - expert residency: every expert in pinned RAM, a VRAM expert cache, router lookahead

GitHub: #301 (master #298).

Flash-Next's experts weigh 37.7 GB at 2.5 bits. The 5090 has room for 14-21 GB
of them beside everything else. This spec decides where each expert projection
lives at each moment:
- **all of them** in pinned host RAM, for the life of the load;
- **a fixed-size cache of them** in VRAM, replaced by recency;
- **the ones the next layer will probably need**, copied ahead of time using the
  router of the next layer.

The design is the one the study's simulations ranked best: a dynamic global
LRU over the whole expert set, 102 tok/s at 1 lane and 186 at 3 lanes with a
21.5 GB cache (`.scratch/flash-next-compression-2026-10-03/review/PLACEMENT.md`).

ADRs: 0030 (every reservation is a plan line made at load), 0043, 0044
(Accepted 2026-10-04), 0017 (metrics).

## Decided for the autonomous run (2026-10-04)

The owner approved this spec on 2026-10-04 and vetoed none of the agent's
proposals: every *(proposed)* item below is decided, and ADRs 0043 and 0044 are
accepted. The prerequisites in Further Notes are the ticket's blockers on
GitHub, not open questions.

- The miss path is chosen by the stated 80% rule; the measurement is the first
  task.
- Host refusal margin 6 GB; warm start exists and is off by default.
- The prefill staging ring is sized for about two layers of one chunk's touched
  projections and is a plan line.
- The trace replay uses the routing traces spec 01 records.

## Departures (2026-10-05, measured)

Accepted by the coordinator under the owner's rule that the method the data
favours is the default. Evidence:
`docs/findings/2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md`
and `docs/findings/2026-10-05-expert-residency-replayed-on-the-study-s-routing.md`.

- **Miss path: device-resident.** An SM-driven copy from mapped pinned memory
  reaches 88-112% of the copy engine with 8-16 blocks; the 80% rule picks the
  device LRU and the copy kernel.
- **Pool split: as one LRU would hold the cache, not by raw traffic share.**
  Each class's pool is its expected occupancy under a single LRU of the
  cache's size, computed at load from the sidecar's per-expert traffic (the
  Che approximation). Pools proportional to traffic starved the K = 2 classes
  and cost 1.48x the simulated residency at three lanes (13.5 against
  9.1 ms per round); the occupancy split costs 1.23x (11.15 ms) and matches
  one lane (3.77 against 3.8 ms). The rest of the gap is the partition
  itself: a single byte-LRU on the same routing costs 9.07 ms. The known
  route to 1.0x, not needed for acceptance 5, is a byte-LRU across classes
  (a larger slot hosting a smaller projection, or periodic rebalancing).
- **Prefetch on a per-step byte budget (a load option).** Unbudgeted, W = 16
  moves 72.5 MB per token at one lane and 295 MB per round at three: 7.2 and
  29 ms of link per 6-7 ms round. A decode step's prefetches are taken in
  rank order (every lane's best expert first) and a candidate that would pass
  the budget is skipped. A prefill streams its lookahead whole. The first
  default was one layer's share of the round at the link's bandwidth (6 ms
  at one lane, +0.5 ms per lane, 12 GB/s), never below the largest class's
  projection: the study replay then hit 96.5% at one lane with 2.37 ms of
  demand misses where no prefetch costs 3.77 ms.
- **The decode budget follows the step's rows (2026-10-07, #306, owner
  decision).** At the 262K x 3-lane default no fixed budget served best at
  both ends (`docs/findings/2026-10-06-flash-next-decode-round.md`): 0.67x
  the 1.75 MB a step then got was faster at one lane, 1.5x at three. A
  decode step's budget is now `PrefetchBudget` (policy model) /
  `prefetch_budget_bytes` + `prefetch_budget_row_bytes` (leaf): 1,172,500 B
  at one row and 726,250 B more per further row, the line through those two
  points (`default_prefetch_budget`). Rows are the step's lookahead rows: the
  lanes decoding now, or a verify round's columns, never the configured
  lane count. There is no floor: at one row a gate/up projection at K >= 3
  (1.24-1.65 MB) is never prefetched. Served A-B-A-B against the old rule:
  one greedy lane 95.2-97.7 tok/s against 92.5-95.6 (+2-3%), three distinct
  lanes 97.8-100.3 against 98.7-99.4 on the undisturbed windows (neutral),
  with 4-7% less stall. The budget reserves no VRAM: prefetches land in the
  class pools' own slots.
  The study replay scores this rule at 95.9% at one lane (95.8% at three,
  where the old default replayed 95.3%), so acceptance's one-lane line in
  `the_lookahead_turns_most_remaining_misses_into_hits` is 95.5%, not 96%:
  the served gain is measured, and the replay's hit rate is a proxy that
  scores every prefetch as hidden, which served one-lane decode
  contradicts. Two bounds of the same test move with it: the one-lane
  demand cost holds at 0.8 of the simulation's 3.8 ms, not 0.75 (2.90 ms
  against 2.37 under the old rule), and the link-fit check takes the served
  round's compute, its ITL less its stall (7.9 ms at one lane, 10.9 at
  three), in place of the simulation's 6-7 ms step, which three rows'
  15.75 ms of link exceeded by 0.7 ms. Unbudgeted, three lanes (29 ms) still
  fail it.
- **A prefill's lookahead has its own width, the router's top-k (2026-10-07,
  #306).** Unbudgeted, W = 16 per token over-streams: 26-33% of a reused agent
  turn's prefetched projections were never read. At 10 (Flash-Next's
  `PREFILL_LOOKAHEAD_WIDTH`) the turn's ~1K-token tail moves 14% fewer bytes
  and starts 0.26 s sooner (2.12 → 1.86 s), and a cold 8.5K prompt 0.48 s
  sooner. At 0 the bytes fall further, but a long chunk loses its overlap and
  the cold prompt is 0.4 s slower than at 16. Decode keeps W = 16 under its
  budget (`docs/findings/2026-10-07-flash-next-agent-turn-tail.md`).
- **W counts experts**, each bringing both its projections: the router ranks
  experts, and the study's 62/77/81% recall is per expert.
- **The staging ring holds two of the heaviest layer's projections**, the
  most a chunk can touch in a layer, so it cannot overflow (about 1.6 GB at
  2.5 bits).
- **Units.** The 6 GB host margin and the 12 GB cache floor are GiB, as every
  plan line of the repo is (`--vram-headroom-bytes` defaults to 1 GiB).
- **The printed hit-rate expectation** (user story 3) is the per-class LRU's
  expected hit rate from calibration rates alone. It sees no locality: 73.8%
  at 21.5 GB where the replay measures 94.5%, so it compares plans, it does
  not predict tok/s.

## Problem Statement

The expert kernels (spec 02) need the selected experts in VRAM, and VRAM holds
only about half of them. A token whose experts are all resident decodes in under
a millisecond of expert time. Every projection a token has to fetch over PCIe 3
at 12 GB/s costs about 0.1 ms (one 2.5-bit projection is ~1-1.5 MB), and a token
selects 20 projections in each of 48 layers.

How often a token misses, and whether the copy overlaps compute, decides
whether Flash-Next runs at about 100 tok/s or at about 40. Fixed placement of
the hottest experts measured 39 tok/s in simulation, with hit rates collapsing
to 42-65% outside the calibration domains.

Prefill has a different shape. A chunk touches most experts of every layer: on
real routing, 71% of the expert bytes for 1K tokens, 80% for 2K, 86% for 4K and
91% for 8K. It must stream most of the 37.7 GB set over PCIe whatever the cache
does: a cold 4K prompt measured 2.65 s on the 5090, transfer-bound (13 GB/s),
with expert compute hidden under the copy
(`.scratch/flash-next-compression-2026-10-03/review/PREFILL_4K.md`). It can only overlap that stream with compute and
amortize it over a large chunk.

## Solution

**The expert pool.** At load, every expert projection is read from the artifact
into one pinned host allocation, 37.7 GB at the 2.5-bit mean. It stays there,
unchanged, until the model is unloaded.

**The expert cache.** Also at load, a VRAM expert cache is reserved:
- one slot pool per K class (two shapes × four K values);
- pool capacities proportional to each class's measured traffic share, from the
  artifact's sidecar;
- total size whatever the VRAM plan leaves after non-experts, KV, workspaces and
  graphs.

**Decode, per layer:**
1. The router selects ten experts per token.
2. Residency maps each selected projection to a slot.
3. A hit costs nothing. A miss evicts the least recently used unpinned slot of
   its class and copies the projection in from host memory.
4. Only then does the expert kernel run.
5. While layer L computes, the router of layer L+1 is evaluated on layer L's MoE
   input. Its top-16 projections not already resident are copied ahead, so most
   of layer L+1's misses are already in flight.

**Prefill.** Each chunk's selected projections are brought in layer by layer.
Layer L+1's copies overlap layer L's compute. Chunks are as large as the
workspace allows, because per-chunk transfer cost is nearly constant.

The cache exports its hit rate, misses, prefetch accuracy, bytes moved and
stall time as metrics.

## User Stories

1. As the owner, I want every Flash-Next expert held in pinned host RAM for the whole load, so that no expert is ever read from disk during serving.
2. As the owner, I want the VRAM expert cache to take all the VRAM the plan leaves, so that the hit rate, and with it the speed, is as high as the card allows.
3. As the owner, I want the cache size printed in the load plan beside its hit-rate expectation, so that I can see how much speed a smaller KV budget or a busy desktop costs.
4. As the owner, I want decode at ≥ 70 tok/s on one lane and ≥ 130 tok/s total on three lanes at short context (spec 04 measures it end to end), so that Flash-Next is usable interactively and for agents.
5. As the owner, I want long prompts to prefill at the speed PCIe allows, with transfer and compute overlapped, so that a 2048-token prompt costs about the time of one pass over the experts and not more.
6. As the owner, I want the load to refuse to start, with a clear message, when the machine lacks the RAM for the pinned experts plus the n-gram hot rows plus a safety margin, so that it never pushes Windows into paging.
7. As the owner, I want the residency metrics on the Monitor (hit rate, misses per token, bytes per second over PCIe, stall time), so that I can tell a slow turn caused by misses from one caused by compute.
8. As the owner, I want residency to cost no VRAM beyond the expert cache and its small tables, and nothing at all when the 27B is loaded, so that the switch in phase 2 is a clean reload.
9. As an engine developer, I want the unit residency moves to be one expert projection (fused gate/up or down) of one K class, so that every slot of a pool is interchangeable and a copy is one contiguous transfer.
10. As an engine developer, I want slot pools per K class sized from the sidecar's measured traffic shares, so that the cache's capacity matches where routing actually goes.
11. As an engine developer, I want replacement to be least-recently-used over all layers within a class, with projections selected in the current step pinned, so that an expert in use is never evicted under the kernel.
12. As the owner, I want a long prompt's prefill not to evict the experts my running conversations decode with, so that a big paste does not slow every other lane afterwards.
13. As an engine developer, I want residency to finish before the expert kernel reads the slot table, with the ordering enforced on the device, so that the kernel never sees a non-resident projection.
14. As an engine developer, I want the next layer's router evaluated on this layer's MoE input to prefetch its likely projections, with the width configurable (default 16), so that most misses are hidden behind compute (study: 62/77/81% recall at 10/16/20).
15. As an engine developer, I want prefetched projections to be ordinary cache entries, recency-tagged at arrival, so that a wrong prefetch costs one eviction and nothing more.
16. As an engine developer, I want decode residency to be graph-capturable, so that decode steps stay CUDA graphs (spec 04).
17. As an engine developer, I want the miss path chosen by a measurement on this machine (SM-driven copy from mapped pinned memory against copy-engine transfers), so that the graph-friendly design is adopted only if it reaches the bandwidth.
18. As an engine developer, I want the same policy available as a pure CPU model that, fed a routing trace, produces the exact hit/miss/evict sequence, so that the GPU implementation is tested against it.
19. As an engine developer, I want a trace-replay harness that drives the real cache, the real copies and the real expert kernels with recorded routing, so that residency's speed is measured before the whole model runs.
20. As an engine developer, I want every residency structure (the pinned pool, the slot pools, the tables, the copy workspace) owned by the model instance and freed when it is dropped, so that no process-wide singleton blocks the later model switch.
21. As an engine developer, I want the host and VRAM reservations expressed as plan lines computed at load (ADR 0030), so that residency obeys the same budget discipline as the KV pool.
22. As an engine developer, I want the cache to start cold and warm under traffic, with an optional warm-start that pre-fills it with the sidecar's hottest projections, so that the first request is not pathologically slow.
23. As a reviewer, I want the measured hit rates and per-token residency cost compared with the simulation's on the same routing traces, so that a gap is explained, not discovered by the owner.

## Implementation Decisions

**Owner-made decisions:** the fast version directly. The model switch is phase 2,
but nothing here may block it.

**Agent proposals**, which the owner may veto, are marked *(proposed)*.

**Unit and classes.**
- The residency unit is one expert projection: the fused gate/up plane or the
  down plane.
- Its byte size is fixed by its shape and K, which gives eight **K classes**.
- Each class has its own slot pool of equal-size slots.

**Host expert pool.**
- One pinned host allocation for all expert projections, filled at load from the
  artifact with buffered reads, layer by layer.
- The study verified a single 38 GB pinned allocation and ~12 GB/s
  host-to-device on this machine.
- It is mapped into the device address space if the measurement below selects
  the SM-driven miss path.
- It is owned by the model and released when the model drops.

**Host plan** *(proposed)*.
- At load the engine computes a host plan: pinned experts, n-gram hot rows
  (spec 04), staging buffers.
- It refuses to start when available physical memory minus the plan is below a
  margin of 6 GB.
- The plan is printed beside the VRAM plan.

**VRAM plan line.**
- The expert cache size = free VRAM at load − (non-experts + KV pool + MoE and
  attention workspaces + graphs + headroom).
- It is split across the K-class pools by the sidecar's traffic shares.
- Floor 12 GB: below it, the load fails with a message naming what to shrink.
- On this machine the plan expects about 14-21 GB, depending on the desktop's
  VRAM and the KV budget.

**Replacement.**
- Per-class LRU on a logical clock that advances per layer step.
- The projections selected by the current step are pinned until its expert
  kernel completes.
- No frequency term, no static core: the simulation found a pure dynamic LRU best,
  and the hybrid with a static core no better at 1 lane and 16% worse at 3.
- **Scan-resistant prefill admission (owner, 2026-10-04).** A prefill chunk
  touches 71-91% of the expert bytes, more than the cache holds. Under plain LRU
  it acts as a scan: it evicts the decode working set, and the chunks of a long
  prompt get no reuse from each other. On real routing, plain LRU doubles decode
  misses after a prefill (40 against 21 MB per token at 21.5 GB) and moves
  130-135 GB for a 31K prompt in 8K chunks. Therefore:
  - a prefill miss takes a **free slot** if its class has one and is then an
    ordinary cache entry;
  - otherwise it passes through a **prefill staging ring** (VRAM, a plan line
    sized for about two layers of one chunk's touched projections) and is never
    inserted into the LRU, so it evicts nothing;
  - prefill hits refresh recency as usual; decode misses evict LRU as before.

  Estimated on the same routing: a warm 4K prefill drops from ~2.0 s to ~1.1 s,
  a 31K prompt from ~10.4 s to ~6.5 s, and the decode set survives the prefill
  (`review/PREFILL_4K.md` in the study).

**Miss path** *(proposed, measurement-gated)*. The ticket's first task measures
on this machine, under WDDM:
- copy-engine host-to-device bandwidth for 1-3 MB transfers;
- SM-driven copies from mapped pinned memory into VRAM.

The outcome picks the design:
- **SM-driven reaches ≥ 80% of the copy engine's bandwidth: device-resident
  residency.**
  1. A small kernel after the router resolves hits and misses against a device
     LRU table, assigns slots, and writes the slot table.
  2. A copy kernel moves missing projections from mapped host memory.
  3. The expert kernel follows on the same stream.

  This is fully graph-capturable, with no host round trip per layer.
- **Otherwise: host-orchestrated residency.**
  1. The router's selection is read back per layer and the host LRU decides.
  2. Copies go on a copy stream with events.
  3. Decode graphs are split per layer segment around the readback.

  The finding records the measurement either way.

**Prefetch.**
- At layer L, the router weight of layer L+1 is applied to layer L's MoE-block
  input. This costs one 2560 × 512 GEMV, a negligible amount.
- The top-W projections not resident and not already in flight are copied ahead
  on the copy path, concurrently with layer L's expert compute.
- W defaults to 16 and is a load option.
- A prefetched projection enters the LRU at its arrival time.

**Prefill.**
- A chunk's selection covers most of each layer's experts, so residency for
  prefill brings in each layer's selected projections in order. Copies for layer
  L+1 overlap layer L's compute.
- Projections already resident are reused. Misses follow the scan-resistant
  admission above: free slots first, then the staging ring.
- The prefill chunk is the program's maximum workspace chunk. Per-chunk transfer
  is nearly constant, so larger chunks amortize it.
- Expected cost: the chunk's touched bytes / ~13 GB/s, about 2.3 s for a cold
  2K chunk and 2.65 s measured for a cold 4K one, less what is
  resident.

**Warm start** *(proposed)*. An optional load option pre-fills each pool with the
sidecar's hottest projections of its class, up to its capacity. This costs load
time; it is off by default.

**Metrics** (ADR 0017 naming, Monitor panel):
- expert-cache hits and misses, per class;
- prefetch issued, prefetch used;
- bytes moved host-to-device;
- time the expert kernel waited on residency;
- slot occupancy per class.

**Ownership.** Every residency structure belongs to the loaded Flash-Next model
instance and is freed in its drop path. No process-wide state is added.

## Testing Decisions

A good residency test feeds a routing trace and checks what residency did:
- which projections were resident when the kernel ran;
- how many bytes moved;
- how long the step waited.

It never checks slot indices or internal table layouts beyond what the CPU model
defines.

- **CPU, in `cargo test`:**
  - the policy model: given pool capacities, class sizes and a routing trace, it
    produces the hit/miss/evict/prefetch sequence;
  - properties: no eviction of a pinned projection; capacity never exceeded; a
    hit costs no bytes; prefetch-then-use counts as a hit;
  - the plan arithmetic: host plan, VRAM plan line, refusal below the floor.

  Prior art: the VRAM plan tests of the vram-budget work, and the KV pool
  accounting tests.
- **GPU, trace replay** (GPU profile, `--ignored`):
  - routing traces recorded by the converter on the study's test chunks, per
    domain (code, prose, chat, en, it, zh, math, py, mmlu), are replayed through
    the real slot pools, the real miss path and spec 02's real expert kernels on
    the real artifact's experts;
  - the GPU's hit/miss sequence must equal the CPU model's on the same trace;
  - it reports the hit rate per domain and the residency cost per token at 1 and
    3 lanes, compared with the simulation's (`PLACEMENT.md`).

  Prior art: the bench crate's recorded-trace instruments and the G4 trace
  replay.
- **Miss-path measurement:** a small GPU executable producing the bandwidth table
  that decides the design. Its output goes in the finding.
- End-to-end tok/s is measured in spec 04's serving acceptance, because it needs
  the whole forward.

## Acceptance

1. All expert projections are loaded into one pinned host pool at load and freed on model drop. The host plan is printed, and the load refuses when RAM is short by the 6 GB margin.
2. The VRAM expert cache is a plan line computed at load, split into eight K-class pools by the sidecar's traffic shares. Floor 12 GB, with a clear failure below it. Nothing is allocated while serving.
3. The miss-path measurement is recorded, and the chosen design (device-resident or host-orchestrated) follows the stated 80% rule. Decode residency is graph-capturable if device-resident.
4. Replacement is per-class LRU with current-step pinning. Router lookahead prefetch of width 16 (configurable) on the next layer's router. The GPU's hit/miss sequence on recorded traces equals the CPU policy model's.
5. Trace replay on the study's test-chunk traces, at a 21.5 GB cache, gives hit rates per domain within 3 points of the simulation (93-96%). The measured residency cost per token is within 1.3× of the simulated 3.8 ms (1 lane) and 9.1 ms (3 lanes), or the gap is explained in the finding.
6. Prefill: transfers for layer L+1 overlap compute of layer L. A cold 2048-token chunk's residency time is within 1.25× of (bytes moved / measured host-to-device bandwidth). Prefill admission is scan-resistant: on a recorded trace of decode, then a 4K prefill, then decode again, the post-prefill decode hit rate is within 2 points of the pre-prefill one, and the staging ring is a plan line.
7. The residency metrics are exported and on the Monitor: hits and misses per class, prefetch issued and used, bytes moved, stall time, occupancy.
8. No process-wide singleton is added. Every residency structure is owned by the model instance.
9. `cargo test` passes workspace-wide. `cargo check --workspace --features cuda --tests` is clean. The GPU replay is green under the GPU profile on a free 5090.

## Out of Scope

- The expert kernels themselves (spec 02) and the model forward (spec 04).
- Experts on NVMe or any tier below host RAM: the study's `ram_tier.txt` showed
  even 2.7 GB on NVMe costs up to ~6 ms per token.
- CPU compute of experts (ExLlamaV3's AVX-512 path; no AVX-512 here).
- Sharing pinned memory with the 27B, keeping experts across a model switch, or
  any fast switch: phase 2. The owner's rule is that a switch reloads everything,
  and VRAM is never spent for the switch.
- Learned or frequency-based replacement policies: only if the trace replay shows
  LRU falls short of acceptance 5.

## Further Notes

- Prerequisites to `ready-for-agent`: ADR 0043 and ADR 0044, spec 01's sidecar
  (traffic shares, routing traces) and spec 02's expert kernels and slot-table
  interface.
- Budget on this machine:
  - RAM: 63.8 GB visible; OS and apps ~11 GB; pinned experts 37.7 GB; n-gram hot
    rows 1-2 GB; the rest is page cache for the NVMe-resident n-gram table.
  - VRAM: 32.6 GB; desktop 1.7-5.4 GB; non-experts ~4.3 GB; KV for 3 lanes;
    workspaces and graphs ~2 GB; expert cache 14-21 GB.
- The simulation assumed copy-engine transfers at 12 GB/s and a one-layer
  prefetch window of about 125 µs. That window is shorter than one 2.5-bit
  projection's transfer, so prefetch hides latency only partly. Acceptance 5's
  tolerance reflects it.
- The KV budget trades 1:1 with the expert cache. Spec 04 sets the default
  context per lane with this in mind.
