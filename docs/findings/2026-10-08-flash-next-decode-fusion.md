# Flash-Next one-lane decode: the 4.3 ms above the bandwidth floor sits inside ~1,200 small launches, folding the HC norm into its down launch buys 0.47 ms, and 160 tok/s needs more than fusion

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: kernel / Flash-Next decode round (graph anatomy, the hyper-connection mix's decode route); serving / one- and three-lane decode rate
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
nothing runs it is a gap charged to the next main node. Floors are each class's
weight and state bytes at 1.79 TB/s (activations ignored; FP8 = 1 byte per
weight; the GDN recurrent state is read and written, 6.3 MB per layer).

Node tracing stretches the kernels: the replay span is 9.39 ms with 1.68 ms of
demand copies, so 7.71 ms of kernels, gaps and exposed branch work, against the
untraced 9.85 - 1.84 - 0.87 = 7.14 ms. The "scaled" column multiplies by
7.14 / 7.71 = 0.926 (inferred: the stretch is taken as uniform).

| class (per round) | nodes | traced µs | scaled µs | floor µs | above floor |
|---|---:|---:|---:|---:|---:|
| HC mix (96 mixes: norm, down, up + reduce) | 288 | 1,664 | 1,541 | 360 | **1,181** |
| HC inject (96) | 96 | 100 | 93 | 0 | 93 |
| routed experts (48) | 48 | 1,413 | 1,308 | 410 | **898** |
| MoE router on the main stream (logits + select) | 96 | 185 | 172 | 70 | 102 |
| residency `resolve_demand` (1 CTA) | 48 | 254 | 235 | 0 | 235 |
| MoE combine | 48 | 233 | 215 | 0 | 215 |
| GDN qkv / z / out GEMVs (FP8) | 108 | 1,452 | 1,344 | 1,160 | 184 |
| GDN a / b GEMVs (48 rows, 6 CTAs each) | 72 | 286 | 265 | 5 | 260 |
| GDN conv, gating, recurrence, gated norm | 144 | 289 | 267 | 127 | 140 |
| QSA q / o GEMVs | 24 | 394 | 364 | 316 | 48 |
| QSA k / v GEMVs | 24 | 99 | 92 | 18 | 74 |
| QSA indexer projection + 4 small kernels | 60 | 108 | 100 | 11 | 89 |
| QSA indexer `score_kernel` (1,024 CTAs) | 12 | 310 | 286 | 0 | **286** |
| QSA attention chain (prepare, hq rows, append, attend, combine, gate) | 72 | 293 | 271 | 0 | 271 |
| n-gram add (layer 1) | 5 | 37 | 34 | 18 | 16 |
| embed + head (final mix, GEMV, sampling) | 7 | 402 | 372 | 359 | 13 |
| side branch exposed (shared expert, lookahead, prefetch copy) | - | 138 | 128 | - | 128 |
| gaps between nodes | - | 58 | 53 | - | 53 |
| **kernels** | 1,200 main + 330 branch | 7,715 | **7,141** | **2,854** | **4,287** |
| demand copies (untraced stall timer) | 48 | 1,679 | 1,840 | - | - |
| device idle between replays (untraced) | - | - | 870 | - | - |
| **token** | | | **9,850** | | |

- **The gaps are not the cost.** A graph replay starts each node 0.1 µs after
  its dependency; gaps sum to 58 µs a round. The 4.3 ms above the floor is
  *inside* nodes: ~1,200 main-chain launches, most of them a few µs of fixed
  launch, ramp and tail around little or no traffic (`resolve_demand` 5.3 µs and
  `router_select` 3.8 µs on one CTA, the a/b GEMVs 4 µs each for 0.07 µs of
  bytes, `hc_inject` 1.0 µs, `gating` 1.1 µs).
- **The large GEMVs are near their floor** (qkv 17.5 µs for 26.2 MB, 84%; the
  head 378 µs for 636 MB, 94%): 0.23 ms above floor in all.
- **The QSA indexer's `score_kernel` launches 1,024 CTAs** (its grid is sized for
  the graph's 262,144-token bound) of which at this ~2K context ~8 have blocks
  to score; it still takes 25.8 µs a layer. Why the empty CTAs cost that much
  was not isolated.
- Grouped, the 4.29 ms above the floor: the HC mix and inject 1.27 ms (30%), the
  routed experts 0.90 (21%), the QSA small kernels 0.72 (17%), the MoE routing,
  residency and combine 0.55 (13%), the GDN small kernels 0.40 (9%), the large
  GEMVs 0.23 (6%), exposed branch work and gaps 0.18 (4%).
- `cudaGraphLaunch` submits the 1,531 nodes in 0.42-0.46 ms untraced (2026-10-06);
  it is part of the 0.87 ms between replays, with the host's sync return,
  n-gram gather and staging.

At three lanes (`n3lane`, same tool, 74 rounds, the trace stretched the round
from 27.5 to 39.2 ms) the demand copies hold 21.1 of 34.6 ms traced and every
kernel class grows less than the lane count (HC mix 1.98 ms, experts 2.47,
GDN core 1.09): three lanes stay link-bound, as the headroom finding says.

### 2. The prototype: the HC norm folded into its down launch

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
change. `IGNIS_FN_HC_FUSED=0` at load restores the three launches
(`fn_hc_set_decode_fused` for tests).

**The microbenchmark** (`ignis_kernel_flash_next_hc_bench --fused 0|1`, FP8
weights cycled over 24 sets so they stream from DRAM, 48 mixes per graph, median
replay per mix, two runs each, agreeing to 0.1 µs):

| rows | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| three launches, µs per mix | 14.19 | 15.87 | 18.07 | 19.88 | 21.57 | 27.74 |
| fused, µs per mix | **10.63** | 12.52 | 16.53 | 20.11 | 23.68 | 35.26 |

Every down CTA normalizes each row in turn, so the fused launch loses past three
rows; the route takes it at up to three (the default decode lanes) and keeps
three launches past that (MTP verify rounds, wider loads).

**Correctness.**
- CTest (`test_flash_next_hc.cu`, new arm): fused against three launches on
  rows 1, 2, 3, 4, 8 and 9, BF16 and FP8 projections and both mixed formats, with
  and without the inject: `x` and the injection weights equal bit for bit, and a
  captured mix is 2 kernel nodes fused and 3 unfused at up to three rows, the
  same count either way past them. Red first: with the switch stubbed, 24 checks
  failed (the launch counts). A deliberately wrong fused norm (`eps * 2` in the
  FP8 launch) fails the bit checks. The fp64-reference arms, which now run the
  fused route at rows 1 and 3, pass as before.
- `flash_next_forward_gpu` under the GPU profile, fused route on: 3/3, G1
  102/102, decode equals prefill on G1 real text (0 flips in 224 tokens), the
  arbitrary-id flips the test's header records and 2026-10-06 measured (3 wider
  at one lane, 3 + 1 near-tie at three), graph replay equal to eager.
- `flash_next_serving_gpu` did not run: its fixed load shape (prompt reuse, the
  2 GiB KV-RAM arena) needs ~48.5 GB of host RAM and the host had 45.4 GB
  available; it refused at the host plan, before any kernel. The served legs
  below run the same scheduler, prefill chunks and one- and three-lane rounds
  through the HTTP server.
- Served: every leg below generated the same text, request for request
  (SHA-256 of the streamed content, one lane and three).

**Served A-B-A.** `.scratch/fusion/aba/rate.sh`: the release build of this
branch, `fn-study/rate`'s flags (`make config MODEL=flash-next` plus
`--kv-host-pool-bytes 0 --retained-host 0 --metrics`), one exe, the switch by
environment. One lane: the three distinct greedy prompts one after another
(1,800 tokens each: a story, a compilers essay, a travel diary; not one text
repeated, which flattered the cache in the earlier harness). Then the three
prompts as three concurrent lanes for 100 s. Six legs in two GPU lock holds, in
the order A1 B1 A2 | B2 A3 B3; A = three launches, B = fused.

| leg | 1 lane tok/s (story / essay / diary) | 1 lane ms per token, pooled | stall per token | expert cache | 3 lanes aggregate (req 1 / req 2) |
|---|---|---:|---:|---:|---|
| A1 (first of the session) | *92.6 / 78.3 / 87.4* | *11.67* | 2.95 ms | 15.55 GB | 104.0 / 108.2 |
| B1 | 105.5 / 87.6 / 98.5 | 10.35 | 2.78 ms | 15.50 GB | 107.7 / 107.6 |
| A2 | 100.5 / 84.2 / 94.2 | 10.81 | 2.76 ms | 15.55 GB | 107.7 / 108.4 |
| B2 | 105.5 / 87.5 / 97.7 | 10.38 | 2.78 ms | 15.54 GB | 104.7 / 106.5 |
| A3 | 100.6 / 84.3 / 93.1 | 10.85 | 2.78 ms | 15.55 GB | 105.0 / 102.5 |
| B3 | 100.9 / 86.1 / 97.2 | 10.61 | 2.92 ms | **15.32 GB** | 102.6 / 103.5 |

- **One lane: -0.47 ms per token, +4.5%.** B1 and B2 10.35-10.38 ms against A2
  and A3 10.81-10.85, the same on each text (+3.9 to +5.2%). A1, the session's
  first leg, ran slow with a higher stall on the same traffic (the testing
  runbook's cold machine) and is left out. B3 loaded with 0.22 GB less expert
  cache (the desktop held more VRAM at its load): 1.6 more misses and +0.14 ms
  of stall per token, which accounts for half its smaller gain (-0.22 ms).
- **Three lanes: neutral.** Aggregates 102.6-108.4 tok/s in both arms, following
  each leg's stall (5.0-5.3 ms per token), as a link-bound round does; the
  microbenchmark's 1.6 µs per mix at three rows (~0.15 ms of a 27.5 ms round) is
  below this harness's resolution.
- **The traced node time predicted it.** `hc_norm` held 6.3 µs a mix in situ
  (0.61 ms traced, 0.57 scaled, over 97 mixes); the fused launch gives back 0.47
  ms of it. The microbenchmark's 3.6 µs a mix (0.35 ms) under-predicts: in situ
  the norm was the slower part, as 2026-10-06 also found.
- These texts are harder than the headroom study's repeated one (hit rate 95.2
  against 96.6%, 65 against 52 MB a token), so the base is 92.5 tok/s, not 101.5.

## Finding

Observed:

1. **The one-lane round's kernels are 7.14 ms against a 2.85 ms bandwidth floor;
   the 4.29 ms above it is inside ~1,200 main-chain launches, not between
   them** (58 µs of gaps a round). The HC mix and inject hold 1.27 ms of it, the
   routed experts 0.90, the QSA small kernels 0.72 (the indexer's 1,024-CTA
   `score_kernel` alone 0.29), the MoE routing, residency and combine 0.55, the
   GDN small kernels 0.40; the large GEMVs run at 84-94% of bandwidth.
2. **Folding the HC norm into the mix's down launch removes 97 launches a round
   and 0.47 ms per one-lane token (+4.5%), bit for bit** (the CTest, the forward
   test, and identical served texts), neutral at three lanes.
3. **A fused launch pays back about what the removed node held in situ** (0.47
   of 0.57 ms): the node-level anatomy is a usable predictor for the next
   merges; the microbenchmark is not (it under-predicted by a third).
4. **Fusion has a row ceiling.** Folding serial per-row work into every CTA wins
   at 1-3 rows and loses from 4 (8 rows: 27.7 -> 35.3 µs).

Inferred: see Implications; every figure there is an estimate from the anatomy
table and finding 3, not a measurement.

## Implications

### The fusion plan

Ordered by risk; one-lane gains estimated as each merge's removed in-situ node
time at finding 3's ~80% pay-back (inferred). "Bits" says whether the merge
keeps the reduction order (bit-exact) or not. None touches a vendored file
except where named; `hc.cu`, `fp8_linear.cu`, `moe_*.cu`, `residency.cu`,
`gdn.cu`, `qsa*.cu` and `indexer.cu` are ours (ADR 0043).

| step | what merges | est. one lane | bits | effort, risk |
|---|---|---:|---|---|
| 1 | HC norm into the down launch (**this prototype**) | **-0.47 ms, measured** | exact | done |
| 2 | the inject (and after MoE the combine) into the next mix's down launch: each down CTA rebuilds its stream as `residual + bf16(y * inj)`, with the residual and the injection weights double-buffered so no CTA reads what another writes; `hc_up_reduce` writes the new residual | -0.25 ms (inject 0.09 + combine 0.20) | exact (same expressions) | medium: crosses `ignis_moe_combine`'s zeroing contract and the n-gram add at layer 1 |
| 3 | `score_kernel` over the visible blocks only (a grid sized to the card, striding to the row's block count, read on the device) | -0.25 ms at short context, less at long | exact (same per-block dot) | small; first explain the 25.8 µs of mostly empty CTAs with a microbenchmark |
| 4 | GDN: a and b in the z GEMV's launch (one grouped FP8 GEMV, a row segment table), gating into conv | -0.25 ms | exact (row-wise) | small-medium; `fp8_gemv_kernel` is ours. The recurrence is vendored (`kernel/vendor/.../gated_delta_net/recurrent.cu`): merging the gated norm or the conv into it is an ADR 0031 patch, not counted here |
| 5 | MoE routing: `router_select` and `resolve_demand` in one single-CTA launch, the select taken by `router_logits`' last CTA | -0.2 ms | exact | medium: crosses the residency ABI (`ignis_residency_step_demand`) |
| 6 | QSA: k and v in one launch, `append_hq` into `hq_rows`, `gate` into the combine | -0.1 ms | exact | small |
| 7 | launch structure: ~470 fewer nodes after 1-6 (`cudaGraphLaunch` ~0.3 µs a node: -0.13 ms); the next round's graph launched before its token is known, its first node waiting on a device flag the host sets after staging (submission off the path: up to -0.4 ms); host-gap fixes 2, 3 and 5 of 2026-10-06 (-0.1) | -0.4 to -0.6 ms | exact | medium; the device-side wait is new to the leaf |
| 8 | the routed-experts kernel from ~29 to ~18 µs a layer | -0.5 ms | order changes (split-K) unless designed not to | high: two restructurings failed (2026-10-05/06); a design question, not a merge |

A persistent per-layer or per-round kernel ("megakernel") is the end state of
steps 1-7; it is not proposed as a first step (the side branches, the residency
copies and 1,531 heterogeneous nodes make it a rewrite).

### The roadmap toward 160-170 tok/s (one lane; inferred, from today's 9.85 ms)

| after | ms per token | tok/s |
|---|---:|---:|
| today (`main`) | 9.85 | 101.5 |
| 1. HC norm fold (measured -0.47) | 9.38 | 107 |
| 2-6. the small-kernel merges | 8.33 | 120 |
| 7. launch structure | 7.8 | 128 |
| 8. an experts kernel at ~18 µs | 7.3 | 137 |
| + prefetch precision at today's budget (headroom finding: perfect filter +2.4-2.9%, perfect predictor +5%) | 6.95-7.1 | 141-144 |
| **target** | **5.9-6.25** | **160-170** |

- **Fusion and launch work alone reach ~120-130 tok/s; 160 is not reachable by
  them.** After step 8 a token is ~5.1 ms of kernels, the 1.84 ms stall and
  ~0.35 ms between replays.
- **160-170 needs one of two further moves on top**:
  - the demand stall from 1.84 to below ~0.6 ms: the headroom finding's oracle at
    twice today's prefetch budget gives +13.9% and at no budget +22% -- a
    predictor that is right *and* more bytes in flight, or more expert cache;
  - or the kernels from ~5.1 to ~4 ms (1.4x the 2.85 ms floor; 4 + 1.84 + 0.35
    = 6.2 ms, 161 tok/s), which is the persistent-kernel end state, not a list
    of merges.
- At three lanes the round is link-bound (15 of 27.5 ms is the stall): steps 1-8
  move it a few percent at most; its lever stays prefetch precision and the
  budget.

## Limits and unknowns

- One node-level capture for the anatomy (its window's demand copies 1.68 ms);
  the scaled column assumes node tracing stretches every class alike.
- Floors count weights and the GDN state only; at long contexts the QSA KV reads
  and the indexer's block keys add real traffic and the QSA rows move.
- The served A-B-A: two clean legs per arm at one lane (B3 on a smaller cache),
  one session, contexts under 2K, greedy prose. Three-lane noise (~4%) hides
  anything under ~2%.
- Every step after 1 is an estimate; finding 3's pay-back ratio comes from one
  merge. Step 8 has no design.
- `score_kernel`'s cost with ~1,016 empty CTAs is not explained.

Raw material, untracked, in the main checkout's `.scratch/fusion/`:
`anatomy.py` (`python anatomy.py <capture.sqlite> [lanes]`), `anatomy-n1lane.txt`,
`anatomy-n3lane.txt` (the captures are the headroom study's
`.scratch/fn-study/link/n{1,3}lane.sqlite`: they embed the environment, never
commit them), `hcbench.log`, `aba/` (`rate.sh`, `seq_client.py`, the leg logs,
`/metrics` snapshots and per-request records with text hashes, `aba-summary.txt`,
`aba2-summary.txt`), `forward-gpu.log`, `serving-gpu.log`.

## Follow-ups

- Status in https://github.com/gpillon/ignis/issues/306: steps 2-7 are the
  decode-headroom kernel and launch items, in the order above.
- `score_kernel`'s empty-CTA cost: a microbenchmark before step 3.
- The experts kernel (step 8) needs a design that keeps 340 CTAs busy at one
  token without the second wave; the 2026-10-05 finding's phase trace is still
  the next step there.
