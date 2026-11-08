# On a card that holds every expert, Flash-Next needs no host copy of them: 38 GB of pinned RAM back, the same decode and the same text

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: expert residency on ADR 0045's resident load (`kernel/src/residency.cu` device-only mode, `crates/core/src/residency/{plan,load,device}.rs`, `crates/core/src/flash_next.rs` `build_residency`, the host plan in `crates/server/src/runtime.rs`)
- Topic: flash-next
- Related: [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), [ADR 0054](../adr/0054-a-resident-load-keeps-its-experts-only-in-vram.md), the finding 2026-10-18-rtxpro6000-profile-measured (on its own branch), spec [flash-next/03](../specs/flash-next/03-expert-residency.md), https://github.com/gpillon/ignis/issues/329, `.scratch/rtxpro6000-profile/`
- Superseded by: none
- Summary: On a 96 GB card holding every expert, the 38 GB pinned host pool served 0.04-0.05% of selections; dropping it took pinned RAM from 41.1 to 3.3 GB and the host plan from 69.9 to 36.8 GB, with the same decode, TTFT and greedy text.

## Question

Expert residency was built for the 5090: every expert projection in one
pinned, mapped host pool for the life of the load (~37.8 GB, one
`cudaHostAlloc`), and a VRAM cache holding a copy of some of them. On a ~96 GB
card ADR 0045's resident branch gives the cache the whole catalog's bytes.
Then:
- Does every projection actually live in VRAM?
- Does the host pool still earn its 38 GB?
- What does removing it cost?

## Evidence

**The split left projections out.** On the trellis-a25 artifact,
`plan_expert_cache` was run with a cache of exactly `ExpertCatalog::total_bytes`
(37,795,446,784 bytes), the resident branch's line (CPU probe, the real
catalog and calibration traffic):

| lanes | projections without a slot | cache bytes left unassigned | expected hit rate |
|---:|---:|---:|---:|
| 1 | 244 | 152 MB | 0.99984 |
| 8 | 1,944 (all K = 2) | 1.21 GB | 0.99436 |

Both hit rates are the ones the served plans logged: 0.99983 in
anvil-serving's report, 0.99436 in the rtxpro6000 campaign. The split models
one LRU of the calibration traffic: it holds no never-selected projection and
floors each class's share. The warm start admits selected experts only
(`warm_start_order`). So those projections came from the host pool on a miss:
455-4,245 misses per measured cell, 0.04-0.05% of its selections (455 of
1.165M, 634 of 1.48M, 4,245 of 9.01M).

**The change.** A residency whose desc has `host_pool_bytes` 0 is
device-only:
- `create` refuses a class with fewer slots than projections and allocates no
  host pool;
- the loader admits every record from layer buffers it reads from the
  artifact (`ignis_residency_admit`, then `_admit_finish`, which refuses while
  any projection is missing). The reads are `stream_expert_layers`: four
  handles as the pool fill used, and a one-layer queue that holds the RAM to
  `stream_staging_bytes`, 4.7 GB on this artifact;
- a step that would still miss is refused with
  `IGNIS_RESIDENCY_STATUS_NO_HOST`, its projections left ABSENT so an expert
  op traps rather than reads, and a prefetch places nothing.

`plan_expert_cache` gives every class all its projections, and an expected
hit rate of 1.0, when the cache holds the catalog. `build_residency` takes
the device-only path exactly then. The host plan's `expert_pool` line is 0
there, and the layer buffers are a line of their own, `load_staging`,
charged only while the experts load (below, Limits).

**Tests.**
- `ignis_kernel_residency_device_only_test` (new CTest) passed on the RTX 5090
  and on the RTX PRO 6000:
  - creation and admission refusals;
  - every slot-table entry on its own class's slot with its K and its
    record's bytes;
  - random decode and prefill steps with either lookahead, and a captured
    decode round, all hits, no copy, no stall.

  After the review it also runs the split step (demand, or routed from the
  router's logits, then either prefetch half) at the router's 512 experts,
  every input uploaded on the step's stream, and a split step's out-of-range
  lookahead, still reported `STATUS_INVALID`. That version, with the kernel's
  matching fix, passed on the RTX 5090 only.
- The existing residency CTests (trace in its four modes, routed) pass
  unchanged on both cards.
- `expert_residency_plan.rs` gains the cache-equals-catalog case;
  `expert_residency_load.rs` checks the streamed layers against the pool fill
  byte for byte and a refused layer ending the stream; `vram.rs` checks that
  only a resident load's cache holds every expert.

**On the card** (RTX PRO 6000 Server Edition, a 16-thread host with 128 GiB
of RAM, Linux). The develop image (main 851100c7, host pool) was compared with
the branch built from the same commit, in the same pod shape: same flags
(`--profile rtxpro6000`, `IGNIS_NGRAM_HOT_BYTES=27G`, loop guard off), same
prompts, greedy, 512 tokens, 3 reps after a warm-up cell:

| | develop (host pool) | branch (device only) |
|---|---:|---:|
| `expected_hit_rate` | 0.994 | 1.0 |
| host plan `planned_bytes` | 69.9 GB | 36.8 GB |
| pinned memory (cgroup `shmem`) | 41.1 GB | 3.3 GB |
| 1 lane code / prose, tok/s | 262 / 197 | 263 / 198 |
| 8 lanes code, tok/s | 428 | 430 |
| expert misses in those cells | 455 / 634 / 4,245 | 0 / 0 / 0 |
| TTFT 32K | 7.37 s | 7.25 s |
| greedy text, 32 lanes | — | identical, every lane |

Every one of the 32 lanes (1-lane code and prose, 8-lane code, all reps)
produced the same text byte for byte. The two `shmem` figures were read in
the pods, with an exec into each pod (transcribed in
`.scratch/rtxpro6000-profile/observed-values-not-in-logs.md`).

**The residency load phase** is not compared: the logs saved hold develop's
51.3 s cold and 12.8 s with the page cache warm, but only one load of the
branch on the larger host, 38.1 s, whose page cache state was not recorded. On
the smaller host, whose 64 GiB cannot cache the artifact beside the whole n-gram
table, every load of the branch took 24-33 s (`ignis-2x-test-current.log`).

**Admission throughput** (`crates/core/examples/residency_admit_bench.rs`,
raw output `.scratch/admit-bench/`): a synthetic device-only residency of
Flash-Next's geometry on the RTX 5090 (12 layers, 12,288 records, 9.56 GB,
mean record 778 KB, the artifact's eight record sizes, a K mix with K = 2 the
most common), host RAM warm, 3 reps, median, two runs. The host is PCIe
Gen 3 x16 (~12 GB/s pinned peak), so these are lower bounds for the card's
Gen 5 link. Another agent was compiling on the same host during the bench;
CPU memory traffic slows such copies (6-15% for one copying thread, up to
~2x with many: finding 2026-10-18-flash-next-decode-cpu-dram-traffic), so
they are lower bounds for this host as well:

| copy, same bytes | run 1 | run 2 |
|---|---:|---:|
| (a) `DeviceResidency::admit` a layer at a time, one pageable `cudaMemcpy` per record (the loader's path) | 3.52 GB/s | 3.74 GB/s |
| (b) one pageable `cudaMemcpy` per layer | 4.91 GB/s (3.9-5.2) | 9.09 GB/s |
| (c) one copy per layer from a `cudaHostAlloc` buffer | 11.39 GB/s | 12.61 GB/s |

Over the six reps (a) ranged 3.15-4.70 GB/s and (b) 3.94-9.58; (c) stayed
within 10.7-12.6.

**On an 8-thread host with 64 GiB** (the same RTX PRO 6000): the branch
loaded Flash-Next with the whole n-gram table. Host plan: 61.0 GB available,
36.8 GB planned, 24.2 GB left. The develop image's plan, 69.9 GB, does not fit
there. A cold load (a freshly booted host, artifact off the network volume)
took 2 min: device weights 7.6 s, residency 67 s, n-gram 46 s. That first
boot's log was not saved; its plan and phases are transcribed in
`.scratch/rtxpro6000-profile/observed-values-not-in-logs.md`, and the later
loads of the same Deployment (62.2 and 61.5 GB available) are in
`ignis-2x-test-current.log`. The same cells as above:
- 1 lane code / prose: 263.6 / 197.8 tok/s;
- 8 lanes code: 430.2 tok/s;
- TTFT 32K: 7.25 s;
- 0 misses, and the same 32 texts as the larger host's.

Eight CPU threads instead of sixteen cost nothing measurable.

## Finding

- **On the resident load the host pool was 38 GB of RAM for 0.04-0.05% of
  selections** (measured; the plan's model put it at 0.6% at eight lanes). It
  served only the misses on the projections the cache split left slotless.
  Giving every projection its slot and dropping the pool costs nothing
  measurable:
  - decode equal or slightly faster, no miss left to wait on;
  - TTFT 1.6% lower;
  - the same text.
- **The host plan of such a load shrinks from ~74 GB to ~41 GB with the whole
  n-gram table** (36.8 GB of lines and the 4 GiB margin), and from ~46 GB to
  ~13 GB with the 1 GiB default. The pinned single allocation, the one a
  32-34 GiB driver ceiling would refuse, is gone. With the layer buffers
  charged only while the experts load (#329 item 3, arithmetic over the
  same lines, not measured): ~36 GB with the whole table (its ~32.1 GB held
  while serving and the margin), ~9 GB with the default (the 4.7 GB of
  buffers, then the peak, and the margin).
- **Both builds share one thing that is not about residency:** an MTP
  decode error. The ~73 tok/s first cells after a load were not slow decode
  (inferred from the warm-ups, whose logs show it; the server logs of those
  two cells were not kept): a request in those reps ended with
  `ignis.runtime.leaf_error`
  (`speculative_window 2 is not the window a round of 1 lanes runs on this
  load (3)`, `finish_reason: error`) a few tokens before its cap, and the
  client, given no usage on an error, counted it in chunks. Every warm-up's
  second rep hit it (255 of 256 tokens). The compared cells (`fn-base2-*`,
  `fn-donly2-*`) have none; their warm-ups do. Fixed on branch
  `fn-mtp-window` (4c0ce315; #329, items 4 and 6).

## Implications

- A machine around a 96 GB card no longer needs RAM for the experts. A 64 GB
  host holds Flash-Next with the whole n-gram table, which on a network block
  volume is worth 1.24x decode and 2.3x prefill. Measured on the smaller host
  above: it serves at the larger host's rates.
- The 5090's offloaded path is unchanged (cache < catalog): same pool, split,
  warm start and misses.
- The exclusive variant (a host pool of only what VRAM does not hold, with
  write-back on eviction) is a separate question for the offloaded path. A
  first link measurement on the 5090 (Gen 3) shows concurrent write-back
  costing the round 19-53%: `.scratch/residency-swap-5090/`.

## Limits and unknowns

- One card, one artifact (trellis-a25), Linux. Windows was compiled and the
  kernel tests run on the 5090; the full device-only load was not run there
  (a 32 GB card is never resident).
- The 4.7 GB of layer buffers were charged for the whole plan in the runs
  above, in the staging line beside the n-gram hot rows, though the load
  frees them before serving: the 36.8 GB `planned_bytes` measured includes
  them. Since #329 item 3 they are their own line, `load_staging`, charged
  only during the residency phase (`FlashNextLeaf::open` frees them before
  the MTP head, the n-gram hot rows and the KV-RAM arena; the retained host
  slots and KV-disk's staging come after the leaf), so the plan holds at its
  peak the larger of that phase and the lines kept while serving, and
  `--ngram-hot-bytes auto` gets the 4.7 GB back. Not re-run on the card.
- Whether the device-only residency phase loads faster or slower than the
  pool fill with the page cache warm was not measured end to end. Inference
  from the admission bench (above): admission makes one synchronous copy per
  record from pageable buffers and runs at ~3.5-3.7 GB/s on the 5090, so the
  37.8 GB catalog takes ~10 s to admit if the card's host sustains the same
  rate (its CPU and link were not measured), against the volume's 1,000 MiB/s (~1.05 GB/s,
  ~36 s). On a cold page cache the load is disk-bound and admission cannot be
  the bottleneck, which fits the one saved 38.1 s load; on a warm one admission would take ~10 s, and the develop image's 12.8 s warm residency
  phase leaves little room for the branch to be slower. Per-layer copies
  from a pageable buffer reached 3.9-9.6 GB/s and from a pinned one ~11-12.6
  (the Gen 3 link), so a layer in fewer, larger copies could cut the warm
  floor; only the warm case would see it.
- The runs above reserved the prefill staging ring (1.57 GB of VRAM) on the
  device-only build too, where no step can stage. Since #329 item 2 such a
  load reserves none (`staging_half_bytes` 0; the VRAM plan places the ring
  on the offloaded branch only) and the KV pool takes its bytes: ~5,825 more
  pages, ~372,800 tokens, +3.0% of a logged 8-lane plan's 191,888 pages
  (MTP off), by the plan's arithmetic (1,574,895,616 / 270,336 bytes a
  page). Not re-run on the card.

## Follow-ups

Noted in https://github.com/gpillon/ignis/issues/329:
- the prefill staging ring on a device-only load (done: none reserved, its
  bytes to the KV pool, above);
- the staging line charged beside the n-gram line (done: `load_staging`,
  above);
- the MTP decode error that ended requests a few tokens before their cap
  (fixed on branch `fn-mtp-window`);
- the residency phase's load time with the page cache warm, and admitting a
  layer in fewer, larger copies.
