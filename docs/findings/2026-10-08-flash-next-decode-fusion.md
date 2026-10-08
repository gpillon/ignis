# Flash-Next one-lane decode: the 4.3 ms above the bandwidth floor sits inside ~1,200 small launches; the plan's seven merges, all bit-exact, remove 512 of them and buy 0.62 ms (+6.2%) where they were estimated at 2 ms, because a merge pays only the serial work it removes, and 160 tok/s needs the stall and the kernels cut

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: kernel / Flash-Next decode round (graph anatomy, the hyper-connection mix's decode route and its folded inject and combine, the indexer's decode score, GDN's and QSA's grouped projections, the router's select in residency's resolve, the round's staging); core / the n-gram gather's wake; serving / one- and three-lane decode rate
- Related: https://github.com/gpillon/ignis/issues/306, [Flash-Next decode headroom](2026-10-08-flash-next-decode-headroom.md), [Flash-Next decode round](2026-10-06-flash-next-decode-round.md), [MoE decode is structure-bound](2026-10-05-moe-decode-is-structure-bound.md), [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md), [ADR 0043](../adr/0043-a-second-model-without-a-ninfer-reference.md)
- Superseded by: none

## Question

At `main` a one-lane Flash-Next token takes 9.85 ms (101.5 tok/s): 7.2-7.6 ms of
kernels and prefetch joins, a 1.84 ms demand-copy stall, 0.87 ms of device idle
between replays ([decode headroom](2026-10-08-flash-next-decode-headroom.md)),
against a DRAM floor near 3 ms. The owner's goal is 160-170 tok/s at one lane.

1. Where do the kernels' ~4.3 ms above the floor go, per op class, against each
   class's own bandwidth floor?
2. Which kernels can merge, what would each merge save, and at what cost to the
   bit-exactness ignis keeps?
3. Prototype the biggest bucket and measure it served (A-B-A, one and three
   lanes): does the measured gain say whether 160-170 tok/s is reachable, and by
   which steps?
4. (The owner's decision of the same evening.) Steps 2-7 of the plan below as one
   batch, each tested bit for bit against its unfused route at once and the
   whole measured only at the end: one served A-B-A against `main` and one
   node-level capture. What does each step buy, and what is left toward 160?

## Evidence

### 1. Anatomy of a one-lane round

**Setup.** The node-level Nsight Systems window of the decode-headroom study
(`n1lane`: release build of `4a84693`, `make config MODEL=flash-next`'s flags
plus `--kv-host-pool-bytes 0 --retained-host 0 --metrics`, one greedy lane,
`--cuda-graph-trace=node`, 3 s, 253 complete rounds of 1,531 nodes). No commit
between `4a84693` and `cddc29b` (this study's base) touches `kernel/`,
`crates/runtime`, `crates/core/src` or `crates/server/src`, so the round is
`main`'s. `.scratch/fusion/anatomy.py` labels every node once by its
`graphNodeId` (stable across replays) from its position in a reference round,
then charges every instant of every round to the main-chain node(s) running;
when only side-branch nodes run (the shared expert, the lookahead router and
rank/resolve, the prefetch copy) the instant is "exposed branch", and when
nothing runs it is a gap charged to the next main node.

**Floors** are each class's weight and state bytes at 1.79 TB/s, activations
ignored, FP8 one byte per weight. `anatomy.py` computes them for the linears,
the mixes, the router and the experts (15.3 MB per layer at the study's K mix);
three were added by hand: the GDN recurrent state, read and written (6.3 MB a
layer, 127 µs), the n-gram projections (18 µs) and the head's final mix (4 µs).
The side branch's own floors are off the critical path and not in the total:
the shared expert 4.9 MB a layer (132 µs a round), the lookahead router 69 µs.

Node tracing stretches the kernels: the replay span is 9.39 ms with 1.68 ms of
demand copies, so 7.71 ms of kernels, gaps and exposed branch work, against the
untraced 9.85 - 1.84 - 0.87 = 7.14 ms. The "scaled" column multiplies by
7.14 / 7.71 = 0.926 (inferred: the stretch is taken as uniform).

| class (per round) | nodes | traced µs | scaled µs | floor µs | above floor |
|---|---:|---:|---:|---:|---:|
| HC mix (96 mixes: norm, down, up + reduce) | 288 | 1,643 | 1,521 | 356 | **1,165** |
| HC inject (96) | 96 | 101 | 93 | 0 | 93 |
| routed experts (48) | 48 | 1,416 | 1,310 | 410 | **900** |
| MoE router on the main stream (logits + select) | 96 | 185 | 172 | 70 | 102 |
| residency `resolve_demand` (1 CTA) | 48 | 254 | 235 | 0 | 235 |
| MoE combine | 48 | 234 | 217 | 0 | 217 |
| GDN qkv / z / out GEMVs (FP8) | 108 | 1,463 | 1,354 | 1,160 | 194 |
| GDN a / b GEMVs (48 rows, 6 CTAs each) | 72 | 288 | 267 | 5 | 262 |
| GDN conv, gating, recurrence, gated norm | 144 | 290 | 269 | 127 | 142 |
| QSA q / o GEMVs | 24 | 394 | 364 | 316 | 48 |
| QSA k / v GEMVs | 24 | 99 | 92 | 18 | 74 |
| QSA indexer projection + 4 small kernels | 60 | 108 | 100 | 11 | 89 |
| QSA indexer `score_kernel` (1,024 CTAs) | 12 | 310 | 286 | 0 | **286** |
| QSA attention chain (prepare, hq rows, append, attend, combine, gate) | 72 | 293 | 271 | 0 | 271 |
| n-gram add (layer 1) | 5 | 37 | 34 | 18 | 16 |
| embed + head (final mix, GEMV, sampling) | 7 | 402 | 372 | 359 | 13 |
| side branch exposed (shared expert, lookahead, prefetch copy) | - | 139 | 129 | - | 129 |
| gaps between nodes | - | 58 | 54 | - | 54 |
| **kernels** | 1,200 main + 331 branch | 7,715 | **7,141** | **2,850** | **4,291** |
| demand copies (untraced stall timer) | 48 | 1,679 | 1,840 | - | - |
| device idle between replays (untraced) | - | - | 870 | - | - |
| **token** | | | **9,850** | | |

- **The gaps are not the cost.** A graph replay starts each node 0.1 µs after
  its dependency; gaps sum to 58 µs a round. The 4.3 ms above the floor is
  *inside* the ~1,200 main-chain launches.
- **Per-launch fixed cost, quantified.** The smallest nodes, with no traffic to
  speak of, take 0.95-1.15 µs (`hc_inject` 1.05, `gating` 1.08, `gated_norm`
  1.15, `append_tail` 0.95, `select` 0.96 mean per call;
  `.scratch/fusion/names-n1lane.txt`): about 1 µs is the bare cost of a launch in
  a replay, ~1.2 ms over 1,200 launches. The rest above the floor is serial
  latency inside small kernels: single-CTA reductions and dependent loads
  (`resolve_demand` 5.3 µs and `router_select` 3.8 µs on one CTA, the a/b
  GEMVs 4 µs each for 0.07 µs of bytes, `hc_norm` 6.3 µs on four CTAs).
- **The large GEMVs are near their floor** (qkv 17.7 µs for 26.2 MB, 83%; the
  head 378 µs for 636 MB, 94%): 0.26 ms above floor in all.
- **The QSA indexer's `score_kernel` launches 1,024 CTAs** (its grid is sized for
  the graph's 262,144-token bound) of which at this ~2K context ~8 have blocks
  to score; it still takes 25.8 µs a layer. Why the empty CTAs cost that much
  was not isolated.
- Grouped, the 4.29 ms above the floor: the HC mix and inject 1.26 ms (29%), the
  routed experts 0.90 (21%), the QSA small kernels 0.72 (17%), the MoE routing,
  residency and combine 0.55 (13%), the GDN small kernels 0.40 (9%), the large
  GEMVs and head 0.26 (6%), exposed branch work, gaps and the n-gram add 0.20
  (5%).
- `cudaGraphLaunch` submitted the 2026-10-06 round's 1,774 nodes in 0.42-0.46 ms
  untraced (~0.25 µs a node); today's 1,531 nodes were not timed untraced. The
  submission is part of the 0.87 ms between replays, with the host's sync
  return, n-gram gather and staging.

At three lanes (`n3lane`, same tool, 74 rounds; node tracing stretched the round
from 27.5 to 39.2 ms) the demand copies hold 21.1 of 34.6 ms traced. The HC mix
(1.66 -> 1.98 ms) and the experts (1.41 -> 2.47) grow far less than the lane
count, the GDN core (0.29 -> 1.09) and `resolve_demand` (0.25 -> 0.88) about
as much. Three lanes stay link-bound, as the headroom finding says.

### 2. The prototype: the HC norm folded into its down launch

**Why this bucket.** The HC mix is the largest class above its floor (1.17 ms),
and within it `hc_norm` is the one node that can go without a grid-wide
dependency: it normalizes each stream separately, and each down CTA owns one
stream. Folding `hc_up_reduce` too would need every down partial before any up
output, a grid-wide barrier that costs about the launch it removes and needs all
CTAs co-resident while the side branch holds SMs; the routed experts (0.90 ms)
are a kernel design problem with two failed restructurings
([2026-10-05](2026-10-05-moe-decode-is-structure-bound.md)), not a merge.
`hc_norm` held 6.27 µs a mix in situ, 609 µs traced per round over its 97 mixes
(0.56 ms scaled; `.scratch/fusion/names-n1lane.txt`).

**What changed** (`kernel/src/flash_next/hc.cu`, ours under ADR 0043; no
vendored file). A decode-route mix was three launches: `hc_norm` (one CTA per
stream and row: the grouped RMSNorm and the block-inject partials), then
`hc_down_split` (40 x 4 CTAs) and `hc_up_reduce` (160 CTAs). Now, at up to three
rows, `hc_norm_down` is one launch of 41 x 4 CTAs: every down CTA first
normalizes its own stream of every row into shared memory with `hc_norm`'s own
code (`norm_stream`, the same strided square sum and block reduce), then takes
`hc_down_split`'s outputs from there (`down_split<.., kShared>`, the same FMA
order); the 41st CTA of each stream writes the normed rows `hc_up_reduce` reads
and the inject partials (`inject_partials`, `hc_norm`'s code). Every value is
computed by the same code in the same order as before, so the mix's bits do not
change. The switch is on by default; `IGNIS_FN_HC_FUSED=0` at load restores the
three launches (`fn_hc_set_decode_fused` for tests), process-wide.

**The microbenchmark** (`ignis_kernel_flash_next_hc_bench --fused 0|1`, FP8
weights cycled over 24 sets so they stream from DRAM, 48 mixes per graph, median
replay per mix; two runs each agree to 0.1 µs, `.scratch/fusion/hcbench2.log`;
rows 4, 5 and 8 fused from a build with the row ceiling raised to 8,
`hcbench-ceiling8.log`):

| rows | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| three launches, µs per mix | 14.25 | 15.82 | 18.10 | 19.85 | 21.76 | 27.65 |
| fused, µs per mix | **10.65** | 12.50 | 16.65 | 20.13 | 23.79 | 35.12 |

Every down CTA normalizes each row in turn, so the fused launch loses past three
rows; the route takes it at up to three (the default decode lanes) and keeps
three launches past that (MTP verify rounds, wider loads).

**Correctness.**
- CTest (`test_flash_next_hc.cu`, new arm; `.scratch/fusion/hc-ctest.log`):
  fused against three launches on rows 1, 2, 3, 4, 8 and 9, BF16 and FP8
  projections and both mixed formats, with and without the inject, each run over
  a scratch arena poisoned with NaN: `x` and the injection weights equal bit for
  bit, and a captured mix is 2 kernel nodes fused and 3 unfused at up to three
  rows, the same count either way past them. `IGNIS_FN_HC_FUSED=0` at the first
  mix keeps three launches. The fp64-reference arms run the fused route at rows
  1 and 3 and pass as before.
- The test's power: red first with the switch stubbed (24 launch-count failures,
  at rows 1-3 and 8, the arm's row ceiling then; run interactively, not logged);
  a fused norm with `eps * 2` fails the bit checks (interactive); a fused launch
  whose writer CTA writes no normed rows, and one whose down CTAs write no
  partials, fail 33 checks each, 18 of them the new arm's
  (`hc-ctest-mutA.log`, `hc-ctest-mutB.log`).
- `flash_next_forward_gpu` under the GPU profile, fused route on
  (`.scratch/fusion/forward-gpu.log`): 3/3, G1 102/102, decode equals prefill on
  G1 real text (0 flips in 224 tokens), the arbitrary-id flips the test's header
  records and 2026-10-06 measured (3 wider at one lane, 3 + 1 near-tie at
  three), graph replay equal to eager.
- **`flash_next_serving_gpu` did not run** (`.scratch/fusion/serving-gpu.log`):
  its fixed load shapes need 38.9-42.1 GB of host RAM plus the plan's 6 GiB
  margin, and the host had 45.4 GB available; all three tests refused at the
  host plan, before any kernel. The served legs below run the same scheduler,
  prefill chunks and one- and three-lane rounds through the HTTP server, but not
  what that test adds (a second turn resumed from a checkpoint, retained slots).
- Served: every leg below generated the same text, request for request
  (SHA-256 of the streamed content, one lane and three).

**Served A-B-A.** `.scratch/fusion/aba/rate.sh`: the release build of this
branch at `cff4ee1` (the review fixes after it touch no kernel path),
`fn-study/rate`'s flags (`make config MODEL=flash-next` plus
`--kv-host-pool-bytes 0 --retained-host 0 --metrics`), one exe, the switch by
environment. One lane: the three distinct greedy prompts one after another
(1,800 tokens each: a story, a compilers essay, a travel diary; not one text
repeated, which flattered the cache in the earlier harness). Then the three
prompts as three concurrent lanes for 100 s (each lane runs its prompt twice,
alike in both arms). Six legs in two GPU lock holds, in the order A1 B1 A2 |
B2 A3 B3; A = three launches, B = fused.

| leg | 1 lane tok/s (story / essay / diary) | 1 lane ms per token, pooled | stall per token | expert cache | 3 lanes aggregate (req 1 / req 2) |
|---|---|---:|---:|---:|---|
| A1 (first of the session) | *92.6 / 78.3 / 87.4* | *11.67* | 2.95 ms | 15.55 GB | 104.0 / 108.2 |
| B1 | 105.5 / 87.6 / 98.5 | 10.35 | 2.78 ms | 15.50 GB | 107.7 / 107.6 |
| A2 | 100.5 / 84.2 / 94.2 | 10.81 | 2.76 ms | 15.55 GB | 107.7 / 108.4 |
| B2 | 105.5 / 87.5 / 97.7 | 10.38 | 2.78 ms | 15.54 GB | 104.7 / 106.5 |
| A3 | 100.6 / 84.3 / 93.1 | 10.85 | 2.78 ms | 15.55 GB | 105.0 / 102.5 |
| B3 | 100.9 / 86.1 / 97.2 | 10.61 | 2.92 ms | **15.32 GB** | 102.6 / 103.5 |

- **One lane: -0.47 ms per token, +4.5%.** B1 and B2 10.35-10.38 ms against A2
  and A3 10.81-10.85; per text +3.9 to +4.9% (B1/B2 against A2/A3). A1, the
  session's first leg, ran 0.84 ms slower than A2/A3 on the same traffic, 0.18
  ms of it a higher stall and the rest unexplained (the testing runbook's cold
  first leg); it is left out. B3 loaded with 0.22 GB less expert cache (the
  desktop held more VRAM at its load): 1.6 more misses and +0.14 ms of stall per
  token, about half of its smaller gain (-0.22 ms).
- **Three lanes: neutral.** Aggregates 102.6-108.4 tok/s in both arms, following
  each leg's stall (5.0-5.3 ms per token), as a link-bound round does; the
  microbenchmark's 1.5 µs per mix at three rows (~0.15 ms of a 27.5 ms round) is
  below this harness's resolution.
- **The removed node predicted it.** `hc_norm` held 0.56 ms (scaled) a round in
  situ; the fused launch gives back 0.47 ms of it. The microbenchmark's 3.6 µs a
  mix (0.35 ms) under-predicts: in situ the norm was the slower part, as
  2026-10-06 also found.
- These texts are harder than the headroom study's repeated one (hit rate 95.2
  against 96.6%, 65 against 52 MB a token), so the base is 92.5 tok/s, not 101.5.

### 3. Steps 2-7 as one batch (2026-10-08, evening)

**What changed** (commits `45b5365`, `3af35ea`, `54bd7c7`, `1abe6cb`, `1acd4ea`;
every file ours under ADR 0043, none vendored). Each step sits behind a switch in
`kernel/src/flash_next/fusion.h`, on by default: `IGNIS_FN_<STEP>_FUSED=0`
turns one off at its first read, `IGNIS_FN_FUSION=0` all of them (step 1's
`IGNIS_FN_HC_FUSED` moved there).

| step | what merges | departures from the plan |
|---|---|---|
| 2 Inject | a sublayer's inject is left pending and the next mix applies it (`fn_hc_mix_after`): on the fused decode route at one row (the first version took up to three: the bench below) each down CTA rebuilds its stream as `hc_inject_kernel` would leave it, after the MoE also recomputing the combine's gate and row with `ignis_moe_combine`'s own code (`moe_combine.cuh`, which both routes run); stream 0's writer CTA stores the combined `y`, `hc_up_reduce` (stream-ordered after every down CTA's read) stores the residual and zeroes the routed accumulator. The n-gram add and the forward's end flush the pending inject; wider calls run it as before | the residual is not double-buffered (the up launch stores it in place); the injection weights are |
| 3 Score | the decode score's own kernel: each block key staged in 16-byte loads behind one page lookup, then `score_kernel<1, 1>`'s per-block dot; grid capped at 1,024 CTAs a row, strided over the visible blocks read on the device | the microbenchmark below found the empty CTAs free: the fix is the staging, not the grid |
| 4 Gdn | qkv, z, a and b in one grouped FP8 GEMV launch (`ignis_fp8_linear_grouped`: a segment table, every row the GEMV's own code), the gating inside the convolution's launch | qkv rides the launch too (four projections, not three) |
| 5 Route | a decode round's router runs its logits launch only; residency's routed demand step (`ignis_residency_step_demand_routed`) selects in its own launch with the router's per-warp code (`moe_router_select.cuh`) and resolves; the lookahead router runs logits only (its selection was never read) | the select rides residency's resolve, launched as a programmatic dependent of the logits (its first instruction waits for them), not the logits' last CTA |
| 6 Qsa | q, k and v in one grouped launch; a split sparse call's combine applies the output gate | the append stays its own launch: under hq-e8-2b it must follow the listed-row decode (it rewrites the residual-window slot of position - 512, which that decode reads: #258), under BF16 precede the attention |
| 7 Staging | a round's inputs gathered into page-locked memory before their copies; the n-gram gather wakes its thread once, when the last read lands (host-gap fix 2; fix 3, every read in flight, was already in: 16 readers) | the copies stay separate (one per destination); the next graph is not launched ahead (Limits) |

**Correctness.** Every fusion was tested at once against its unfused route, bit
for bit, over poisoned scratch and outputs, with the launches it saves counted
from a captured graph (`.scratch/fusionb/ctest-*.log`):
- HC CTest: the folded inject, with and without the combine, against the
  combine, inject and mix one after another -- x, the injection weights, the
  residual, the combined y, the zeroed accumulator -- rows 1-4 and 8 (folded at
  one, the launch counts equal past it), BF16 and FP8; `IGNIS_FN_FUSION=0`
  turns every switch off.
- indexer CTest: three lanes of 1, 512 and 75,001 blocks under a 524,288-token
  bound (2,048 CTAs a row unfused, the stride fused), every score bit for bit.
- FP8 CTest: the grouped GEMV at 1, 2, 3, 5 and 8 tokens, BF16 and fp32 out,
  against one linear per segment; its refusals.
- GDN CTest: 1, 3 and 5 decode lanes and 6- and 70-token prefills: the output
  and both state planes of every slot, 4 kernels fewer (1 past 8 rows).
- QSA CTest: both KV formats, one lane and three: the output and every store
  plane (pages, metadata, residual window, ring), 3 kernels fewer.
- a new residency CTest: the routed step against the router plus the demand
  step over 12 steps that evict, a tied and a NaN token: ids, weights, logits,
  reports, slot tables, counters; 3 kernels for 4.
- **The tests bite** (`mut1-*.log`, `mut2-*.log`): eight mutants, each caught
  by its own test -- the residual not stored (12 checks), the score stride
  skipping tiles (1), the gating one element short (28), the routed select on
  the first token only (35), the combine's gate without its BF16 rounding (5),
  the folded gate dropping a warp's partial (16), the grouped GEMV writing at the
  launch's row stride (4 in the FP8 test, 29 in GDN's, 45 in QSA's); and on the
  fold as rewritten (`mut3-*.log`), the residual not stored (4), the combined y's
  last element not stored (4), every stream injected with stream 0's weight (8).
- Full kernel CTest 106/107; the one failure,
  `ignis_kernel_model_load_speculative_options_test` (backend 3 is now the MTP
  backend and refused with another message), predates the batch: neither the
  test nor `model.cu` is touched.
- `flash_next_forward_gpu` 3/3, on the first batch commit and again on the last
  (`forward-gpu.log`, `forward-gpu-final.log`): G1 102/102, decode equals
  prefill on real text, graph replay equals eager -- and its whole output, the
  arbitrary-id flips with their logits and margins included, is the step-1
  run's character for character, both times.
- `flash_next_serving_gpu` 3/3, both times (`serving-gpu.log`,
  `serving-gpu-final.log`): the host had the RAM (47.8-47.9 GB available), so
  step 1's open gap is closed too.
- The touched CTests again on the last commit (`final-*.log`, the MoE
  shared/combine, block and router tests with them: the combine's gate is now one
  helper both routes run), and `cargo test --workspace` 2,549 passed, 0 failed
  (`cargo-test-ws.log`); `cargo check --workspace --features ignis-server/cuda
  --tests` clean.
- The 27B: `git diff main` touches no file of its decode path (`model.cu`,
  `step.cu` and `decode_graph.cu` reach Flash-Next only through the program's
  entry points, unchanged); `cuda_leaf_gpu` (prefill and decode through the
  production leaf) 1/1.
- Served: all 24 requests of every leg below, A, B and C, generated the same
  text (SHA-256 of the streamed content).

**Step 3 first: what do the empty CTAs cost?** (`bench_flash_next_score`, one
lane, 48 calls a graph, median of 50 replays, us a call; `scorebench*.log`)

| context | blocks | the graph's grid (1,024 CTAs) | a grid fitted to the blocks | the new kernel |
|---:|---:|---:|---:|---:|
| 2,050 | 512 | 13.20 | 13.49 | **5.00** |
| 8,194 | 2,048 | 13.46 | 13.42 | 5.25 |
| 32,770 | 8,192 | 13.45 | 13.45 | 5.56 |
| 131,074 | 32,768 | 14.30 | 14.30 | 6.28 |
| 262,142 | 65,535 | 25.78 | 25.78 | 10.76 |

The empty CTAs cost nothing measurable: a fitted grid is as slow. The time is
each working CTA staging its 64 keys a 4-byte word at a time, each word behind
its own page lookup (64 dependent loads a thread). Staged in 16-byte loads behind
one lookup per key the same dot runs 2.4-2.6x faster at every context. (A first
version capped the grid at 128 CTAs a row: 1.2-1.3x slower than the full grid
from 32K blocks; the cap went to 1,024.)

**The fold, timed** (`bench_flash_next_hc --fold 1|0`, FP8 mix weights cycled
over 24 sets, 48 mixes a graph, median replay; us a mix: the mix with the
previous sublayer's inject pending, then with the MoE combine too, folded
against the combine, the inject and the mix launched apart;
`hcbench-fold.log`, `hcbench-fold2.log`):

| rows | inject: folded, first version | folded | apart | combine and inject: folded, first version | folded | apart |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.44 | **11.48** | 11.81 | 16.02 | **13.19** | 14.56 |
| 2 | 19.11 | 14.33 | 13.71 | 23.69 | 19.36 | 16.51 |
| 3 | 28.35 | 18.66 | 17.81 | 33.03 | 25.74 | 20.59 |

- **The first version was slower than not folding at every row count.** Its load
  lambda carried a runtime branch with the combined y's store inside, so the
  norm's loads, issued as one batch before, serialized behind it. With the
  pending kind a template parameter, the loads read-only and the store after
  the batch (`54bd7c7`), the fold wins at one row and still loses from two:
  every down CTA rebuilds each row in turn and recomputes each row's gate. The
  fold now takes one row only; the norm keeps its three-row ceiling.
- The first served round below ran the first version, at up to three rows.

**Served A-B-A-B, two rounds** (`.scratch/fusionb/aba/`: `rate.sh`, `table.py`).
A = a release build of `main` at `41907f5`; B = this branch's release build
(`main` merged in), every step on; C = B's binary with `IGNIS_FN_FUSION=0`. The
step-1 study's harness: the same flags (`make config MODEL=flash-next` plus
`--kv-host-pool-bytes 0 --retained-host 0 --metrics`), the same three texts at
one lane, then three lanes for 100 s. Each round one timing hold.

| leg | 1 lane tok/s (story / essay / diary) | 1 lane ms per token, pooled | stall per token | expert cache | 3 lanes aggregate (req 1 / req 2) |
|---|---|---:|---:|---:|---|
| *round 1: the first fold, 1-3 rows* | | | | | |
| W (cold, discarded) | *101.5 / 82.7 / 93.0* | *10.90* | 2.74 ms | 15.70 GB | 106.5 / 108.1 |
| A1 | 95.2 / 77.8 / 90.4 | 11.47 | 2.87 ms | 15.71 GB | 109.8 / 109.8 |
| B1 | 105.9 / 88.0 / 98.8 | 10.31 | 2.70 ms | 15.68 GB | 108.4 / 108.2 |
| A2 | 102.0 / 85.3 / 95.3 | 10.67 | 2.69 ms | 15.68 GB | 109.5 / 109.3 |
| B2 | 105.7 / 87.8 / 98.8 | 10.32 | 2.70 ms | 15.68 GB | 108.2 / 108.2 |
| A3 | 102.2 / 85.8 / 96.1 | 10.61 | 2.68 ms | 15.69 GB | 110.3 / 110.3 |
| B3 | 106.1 / 88.2 / 99.1 | 10.28 | 2.68 ms | 15.69 GB | 108.8 / 108.9 |
| C1 (B, all off) | 101.9 / 85.4 / 95.6 | 10.66 | 2.68 ms | 15.69 GB | 110.1 / 113.0 |
| *round 2: the fold fixed, 1 row* | | | | | |
| A4 | 103.7 / 83.5 / 95.7 | 10.68 | 2.53 ms | 16.33 GB | 114.3 / 114.9 |
| B4 | 110.6 / 91.3 / 103.5 | **9.88** | 2.53 ms | 16.18 GB | 111.5 / 113.8 |
| A5 | 103.2 / 86.9 / 97.0 | 10.50 | 2.51 ms | 16.19 GB | 113.3 / 113.0 |
| B5 | 109.4 / 93.2 / 104.7 | **9.81** | 2.52 ms | 16.17 GB | 114.3 / 114.3 |
| A6 | 100.9 / 87.2 / 96.2 | 10.59 | 2.55 ms | 16.09 GB | 108.4 / 113.2 |
| B6 | 103.9 / 87.6 / 104.2 | **10.21** | 2.63 ms | 16.12 GB | 110.7 / 113.3 |
| *round 3: the final commit (`254420a`: round 2's code plus the review fixes and the n-gram one-wake)* | | | | | |
| A7 | 102.6 / 85.8 / 96.9 | 10.57 | 2.55 ms | 16.09 GB | 113.5 / 113.3 |
| B7 | 110.4 / 93.4 / 104.4 | **9.78** | 2.52 ms | 16.13 GB | 112.8 / 109.8 |
| A8 | 104.1 / 87.7 / 98.0 | 10.40 | 2.49 ms | 16.16 GB | 114.7 / 113.9 |
| B8 | 111.9 / 90.8 / 103.5 | **9.87** | 2.55 ms | 16.15 GB | 112.8 / 113.6 |

- **One lane: -0.62 ms per token, +6.2% tok/s (round 2).** B4-B6 9.81-10.21 ms
  (mean 9.97) against A4-A6 10.50-10.68 (mean 10.59); pair by pair -0.80, -0.69
  and -0.38 ms, B6 with 0.08 ms more stall than A6. Per text, the means of B4-B6
  against A4-A6: story +5.2%, essay +5.6%, diary +8.1%.
- **Round 3, the final commit: -0.66 ms, +6.7% tok/s.** B7/B8 9.78-9.87 ms against
  A7/A8 10.40-10.57: the review fixes and the n-gram one-wake keep round 2's gain
  (the one-wake's own share is inside the noise).
- **Three lanes: within noise.** Round 2 B 110.7-114.3 tok/s against A
  108.4-114.9 (means 113.0 and 112.9), round 3 B 109.8-113.6 against A
  113.3-114.7 (B7 with 0.13 ms more stall a token); over both rounds -0.5%. Round 1's
  -1.2% (B 108.2-108.9 against A 109.3-110.3, every B leg below every A leg) was
  the first fold at three rows, which the bench shows 1.4x slower than not
  folding: limiting the fold to one row took it away.
- **Round 1, the first fold: -0.34 ms.** B1-B3 10.28-10.32 against A2/A3
  10.61-10.67. A1 ran 0.83 ms slower than A2/A3 on the same traffic (the third
  session in a row whose early base leg does) and is left out; round 2 opened
  with A4 and shows no such leg.
- **C equals A** (10.66 ms, A's texts): with the switches off the branch is
  `main`, so the switches isolate the steps as intended.
- Round 2's host had 0.4-0.6 GB more expert cache than round 1's (the desktop
  held less VRAM), hence the lower stalls (2.5 against 2.7 ms) in both arms.
- All requests of all 18 legs generated the same text.

**Per kernel** (`perkernel.py`, `dissect.py`): one node-level capture of the
new round (`n1new2`: round 2's B binary, `54bd7c7`, which differs from the final
commit by the review fixes (bit-identical) and the host's n-gram one-wake;
`fn-study/rate`'s `prompt_greedy` at one lane, 3 s from +90 s, `--cuda-graph-trace=node`, 304 complete rounds) and,
for a like-for-like base, the same capture of A (`n1base`, 254 rounds; `n1new`
is round 1's binary). Node durations summed per round, traced (node tracing
stretches small kernels: the step-1 study scaled its table by 0.93).

| step | base (`main`) | new | nodes | traced us a round |
|---|---|---|---:|---:|
| 1+2 HC mix, inject, MoE combine | `hc_norm` 609, `hc_down_split` 483, `hc_up_reduce` 560, `hc_inject` 99, combine 234 (435 nodes) | `hc_norm_down` 860 -- attention mixes, carrying the combine, 10.3 us each (11.3 + 4.9 base); MoE mixes, carrying the inject, 7.5 (11.3 + 1.1 base) -- `hc_up_reduce` 589, the two flushes 11 (196 nodes) | -239 | **-525** |
| 3 Score | `score_kernel` 308 | `score_decode_kernel` 106 | 0 | **-202** |
| 4 GDN | qkv/z/a/b GEMVs 1,329 (144), gating 39, conv 78 | grouped GEMV 1,294 (36), conv 89 | -144 | -63 |
| 5 Route | `router_select` 350 (48 main + 47 lookahead), `resolve_demand` 259 | `resolve_demand_routed` 309 | -95 | **-300** (-127 on the main chain, the lookahead's select -173 on its branch) |
| 6 QSA | q/k/v GEMVs 349 (36), gate 15, sparse combine 49 | grouped GEMV 361 (12), combine with the gate 57 | -36 | +6 |
| 7 launch | 1,531 nodes; `cudaGraphLaunch` 1,579 us; idle between replays 2,208 us | 1,019 nodes; 964 us; 1,512 us | -512 | (host side) |

- The replay span fell from 9,028 to 7,879 us (traced), 331 us of it the demand
  copies of a different window (B decodes faster, so its window sits later in
  the text). Kernels the batch does not touch moved by noise or overlap: the
  320-CTA GEMVs (GDN and QSA out, the shared expert's down on its branch) +73 us,
  `hq_rows` +27 us.
- **Merging launches paid only where it removed serial work.** Graph nodes start
  0.1 us after their dependency, so a removed node's "1 us" in the anatomy was
  the kernel's own minimum duration, not a launch toll: a merge pays what the
  absorbing kernel does not grow by. The grouped GEMVs removed 132 nodes for
  -23 us (the rows' work is unchanged). The fold, the score's staging and the
  select in the resolve removed real latency: -525, -202 and -300 us.
- Node tracing inflates the launch call and the replays' small kernels alike,
  so the traced savings (span -1.1 ms, `cudaGraphLaunch` -0.6 ms) overstate the
  served -0.62 ms.

## Finding

Observed:

1. **The one-lane round's kernels are 7.14 ms against a 2.85 ms bandwidth floor;
   the 4.29 ms above it is inside ~1,200 main-chain launches, not between
   them** (58 µs of gaps a round). About 1 µs a launch looked like bare launch
   cost (item 6 corrects this: it is each small kernel's own minimum); the rest
   is serial latency inside small kernels. The HC mix and inject hold 1.26
   ms of it, the routed experts 0.90, the QSA small kernels 0.72 (the indexer's
   1,024-CTA `score_kernel` alone 0.29), the MoE routing, residency and combine
   0.55, the GDN small kernels 0.40; the large GEMVs run at 83-94% of bandwidth.
2. **Folding the HC norm into the mix's down launch removes 97 launches a round
   and 0.47 ms per one-lane token (+4.5%), bit for bit** (the CTest, the forward
   test, identical served texts), neutral at three lanes. The serving GPU test
   could not run on that host's free RAM (it ran on the batch: item 5).
3. **The fused launch gave back 0.47 of the 0.56 ms the removed node held in
   situ**; the microbenchmark predicted 0.35.
4. **The fold has a row ceiling.** Serial per-row work in every CTA wins at 1-3
   rows and loses from 4 (8 rows: 27.7 -> 35.1 µs).
5. **Steps 1-7 together, all bit-exact, take a one-lane token from `main`'s
   10.59 ms to 9.97 (-0.62 ms, +6.2% tok/s; the final commit 10.49 to 9.83,
   -0.66 ms, +6.7%) and leave three lanes within noise** (rounds 2 and 3, every
   text identical to `main`'s, G1 102/102, the forward and serving GPU tests
   green). The batch removes 512 of 1,531 graph nodes. Per kernel (traced): the
   HC mix and its folded inject and
   combine -525 us a round, the select inside residency's resolve -300 (-127 on
   the main chain), the score's staging -202, GDN's grouped projections and
   gating -63, QSA's grouped projections and gate +6.
6. **A merge pays only what it takes off a serial path, not a launch toll.**
   Graph nodes start ~0.1 us after their dependency; the anatomy's "~1 us a
   launch" was each small kernel's own minimum duration. The grouped GEMVs
   removed 132 nodes for -23 us; the merges that removed serial latency (the
   select's own launch, the score's dependent loads, the fold's separate norm,
   inject and combine) carry the gain. The plan's estimates (-2.0 ms for steps
   1-7) were 2-3x too high for this reason.
7. **The indexer score's empty CTAs cost nothing; its staging did.** A grid
   fitted to the visible blocks is as slow as the graph's 1,024-CTA grid; staging
   each key in 16-byte loads behind one page lookup makes the same dot 2.4-2.6x
   faster at every context (13.2 -> 5.0 us at 2K tokens, 25.8 -> 10.8 at 262K).
8. **A fold must keep its loads in one batch, and it pays at one row only.**
   The first fold, a runtime branch with a store inside the load loop, ran
   slower than not folding (one row 14.4 against 11.8 us a mix) and cost three
   lanes 1.2%; with the pending kind a template parameter it wins at one row
   (11.5 / 13.2 against 11.8 / 14.6 us) and still loses from two, where every
   down CTA rebuilds each row in turn.

Inferred:

- From findings 3 and 6: the node-level anatomy predicts a merge where the
  removed node's time is serial work the absorbing kernel does not redo (the
  norm, the select, the score's staging); where the absorbing kernel redoes the
  work (the fold at several rows) or the rows' work stays (grouped GEMVs), it
  over-predicts. The step-1 study's 80% pay-back held for the first kind only.
- Step 1's -0.47 ms and this batch's -0.62 come from different sessions (the
  base differed by 0.2 ms between them), so steps 2-7's own served share is not
  separable; per kernel it is the traced -1.08 ms above minus the norm's part of
  the HC line.

**The batch stays on the branch** (`fusion`, `main` merged in at `5e9298a`),
ready for review: every gate this note names is green on its last commit, and
each step can be turned off by its environment variable. Merging can keep the
switches (process-wide, read at the first launch) or drop them, since every
fused route's bits equal the unfused one's.

## Implications

### The fusion plan, measured

In the plan's order; the plan's estimates were each merge's removed in-situ node
time at ~80% pay-back (inferred from step 1). "Traced" is the per-kernel change
of the capture above (us a round, node tracing); "served" is the A-B-A-B. Every
merge keeps the reduction order (exact). None touches a vendored file; `hc.cu`,
`fp8_linear.cu`, `moe_*.cu`, `residency.cu`, `gdn.cu`, `qsa*.cu` and
`indexer.cu` are ours (ADR 0043).

| step | what merged | plan's estimate | traced | nodes |
|---|---|---:|---:|---:|
| 1 | HC norm into the down launch | -0.56 in situ; **-0.47 served** (own session) | -525 with step 2 | -97 |
| 2 | the inject, after the MoE with its combine, into the next mix's down launch (one row) | -0.25 | (in the line above) | -142 |
| 3 | the decode score's keys in 16-byte loads (the grid stays) | -0.25 | -202 | 0 |
| 4 | GDN: qkv, z, a, b in one grouped launch; the gating in the convolution | -0.25 | -63 | -144 |
| 5 | the router's select inside residency's demand resolve; the lookahead's logits only | -0.2 | -300 (-127 main chain) | -95 |
| 6 | QSA: q, k, v in one grouped launch; the gate in the split combine (the append stays: #258) | -0.1 | +6 | -36 |
| 7 | page-locked staging; one wake per n-gram gather; fewer nodes (launch call -0.6 ms traced) | -0.4 to -0.6 | host side | (-512 in all) |
| 1-7 | | -2.05 | span -1.15 ms | **served -0.62 ms** |
| 8 | the routed-experts kernel from ~29 to ~18 us a layer | -0.5 | not done (out of scope) | |

A persistent per-layer or per-round kernel ("megakernel") remains the end state
of steps 1-7: the side branches, the residency copies and ~1,000 heterogeneous
nodes make it a rewrite, not a merge.

### What is left toward 160-170 tok/s (one lane; inferred, these three texts)

These texts are harder than the headroom study's repeated one: `main` takes
10.59 ms a token here (94 tok/s) against 9.85 there. The branch's 9.97 ms
(100 tok/s) is roughly 2.5 ms of demand stall, ~0.7 ms between replays (the
headroom study's 0.87 less the ~0.13 ms of submission 512 fewer nodes save; not
measured untraced) and ~6.8 ms of kernels.

| after | ms per token | tok/s |
|---|---:|---:|
| `main` (measured) | 10.59 | 94 |
| steps 1-7 (measured) | 9.97 | 100 |
| 8. an experts kernel at ~18 us (-0.5) | 9.5 | 105 |
| 7b. the next graph launched ahead (submission off the path, -0.25 to -0.4) | 9.1-9.25 | 108-110 |
| + prefetch precision at today's budget (perfect filter +2.4-2.9%, perfect predictor +5%) | 8.7-8.9 | 112-115 |
| **target** | **5.9-6.25** | **160-170** |

- **Merges and launch structure end near 105-110 tok/s on these texts.** What
  remains above the kernels' 2.85 ms floor sits in the HC mix (1.46 ms traced
  against 0.36), the routed experts (1.42 against 0.41), the QSA small kernels
  (~0.44 against ~0.01) and routing and residency (~0.45 on the main chain):
  each now a kernel-design question, not a launch to remove.
- **160-170 needs both the stall and the kernels cut, not one of them**: at a
  2.5 ms stall a 6.25 ms token leaves 3 ms for kernels and replay gaps, about
  the bandwidth floor. A stall near 0.5 ms (a predictor that is right *and*
  more bytes in flight, or more expert cache: the headroom finding's oracle at
  twice the budget +13.9%, at no budget +22%) plus kernels near 1.4x the floor
  (~4 ms, the persistent-kernel end state) gives 4 + 0.5 + ~0.5 = 5 ms.
- At three lanes the round is link-bound (the stall is 4.5-4.9 ms of every
  ~9 ms per token); steps 1-8 move it a few percent at most.

## Limits and unknowns

- One node-level capture for the anatomy (its window's demand copies 1.68 ms);
  the scaled column assumes node tracing stretches every class alike. The
  batch's attribution is traced too (two captures, A and B), and traced savings
  overstate served ones (span -1.15 ms traced against -0.62 served).
- Floors count weights and the GDN state only; at long contexts the QSA KV reads
  and the indexer's block keys add real traffic and the QSA rows move.
- Served: step 1's A-B-A had two clean legs an arm; the batch's round 2 three
  (B6 with a higher stall); contexts under 2K, greedy prose, one session a round.
  Three-lane noise hides anything under ~2%. Steps 2-7 were not measured one by
  one served (the owner's method: one A-B-A at the end), so their served shares
  are not separable; the switches allow it.
- The fold takes one row: past it the inject and combine keep their launches.
  A fold that pays at three rows would spread each row's rebuild over the CTAs
  instead of every CTA rebuilding every row.
- Step 6's append is not folded (the #258 ordering under hq-e8-2b; under BF16 it
  must precede the attention).
- Step 7: the copies from page-locked memory stay separate (one copy would need
  the round's device inputs contiguous: they are separate buffers prefill
  shares); the staging has no dedicated test, only the end-to-end ones (the
  forward test's decode against prefill and graph against eager, the serving
  test, the served texts). **The next graph is not launched ahead**: a graph
  enqueued before its round is known runs whatever happens, so a round that
  never comes (a lane ends, the width changes, a constrained round runs
  eagerly) needs the round's body inside a CUDA conditional node switched by a
  device flag the host sets after staging, the staging on another stream, and
  an error path that releases the wait -- a capture structure the leaf does not
  have; time-boxed out of this batch. Fix 3 of the host gap (every read in
  flight) was already in at one lane (16 readers); at three lanes a round can
  issue more.
- `ignis_kernel_model_load_speculative_options_test` fails on this branch and,
  untouched by it, on its base: backend 3 is now the MTP backend and refused
  with another message than the test expects.

Raw material, untracked, in the main checkout's `.scratch/fusion/` (step 1):
`anatomy.py` (`python anatomy.py <capture.sqlite> [lanes]`), `names.py`,
`anatomy-n1lane.txt`, `anatomy-n3lane.txt`, `names-n1lane.txt` (the captures are
the headroom study's `.scratch/fn-study/link/n{1,3}lane.sqlite`: they embed the
environment, never commit them), `hcbench.log`, `hcbench2.log`,
`hcbench-ceiling8.log`, `hc-ctest.log`, `hc-ctest-mutA.log`, `hc-ctest-mutB.log`,
`mutate.py`, `forward-gpu.log`, `serving-gpu.log`, `aba/` (`rate.sh`,
`seq_client.py`, the legs' server logs, `/metrics` snapshots and per-request
records with text hashes, `aba-summary.txt`, `aba2-summary.txt`). And in
`.scratch/fusionb/` (steps 2-7): `ctest-*.log`, `ctest-all.log` (the full kernel
CTest), `final-*.log` (the touched CTests on the last commit), `mut1-*`, `mut2-*`,
`mut3-*` (the mutants), `scorebench*.log`, `hcbench-fold*.log`,
`forward-gpu*.log`, `serving-gpu*.log`, `cuda-leaf-27b-gpu.log`,
`cargo-test-ws.log`, `perkernel.py`, `dissect.py`, `perkernel-*.txt`,
`dissect.txt`, `trace.sh` and the captures `n1base`, `n1new`, `n1new2`
(`.nsys-rep`/`.sqlite`: they embed the environment, never commit them), and
`aba/` (`rate.sh`, `aba.sh`, `aba2.sh`, `table.py`, the legs' logs, metrics and
records).

## Follow-ups

- Status in https://github.com/gpillon/ignis/issues/306: steps 1-7 on branch
  `fusion`, measured (-0.62 ms a one-lane token on these texts); step 8 (the
  experts kernel) and the graph launched ahead (step 7's open half) are the next
  launch- and kernel-side items, prefetch precision and the budget the stall's.
- The fold at several rows: spread each row's rebuild across the CTAs.
- The pre-existing `ignis_kernel_model_load_speculative_options_test` failure.
- The experts kernel (step 8) needs a design that keeps 340 CTAs busy at one
  token without the second wave; the 2026-10-05 finding's phase trace is still
  the next step there.
