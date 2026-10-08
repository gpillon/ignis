# Flash-Next decode headroom: one lane is kernel-bound, three lanes are link-bound, and neither a perfect predictor at today's budget nor CPU experts changes that much

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: serving / Flash-Next decode round at the 262K x 3-lane default; core / expert residency prediction (the router lookahead, the prefetch budget); a CPU path for routed experts
- Related: https://github.com/gpillon/ignis/issues/306, [Flash-Next decode round](2026-10-06-flash-next-decode-round.md), [KV pool follows residency](2026-10-08-flash-next-kv-pool-follows-residency.md), [Expert residency replayed on the study's routing](2026-10-05-expert-residency-replayed-on-the-study-s-routing.md), [An SM-driven copy matches the copy engine](2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md), [MoE decode is structure-bound](2026-10-05-moe-decode-is-structure-bound.md), [spec flash-next/03](../specs/flash-next/03-expert-residency.md)
- Superseded by: none

## Question

Flash-Next decode leaves the GPU mostly idle (the 2026-10-06 capture: 1,774
graph nodes per one-lane round, 22% of DRAM bandwidth, 33% of SMs active).
Is the next investment better expert prediction and prefetch, a CPU path
that computes cold experts from the pinned host pool instead of copying them
(the Fiddler / ktransformers idea), or the round's kernels and launch
structure? Three numbers decide it:

1. where a decode round's time goes at today's `main`, at one lane and at three;
2. how much of the expert stall a perfect predictor would remove at the same
   prefetch budget and cache, and how good today's predictor is;
3. what one routed expert costs on the CPU against copying it over the link.

## Evidence

### 1. The decode round at `main` (`4a84693`)

**Setup.** The release build of `4a84693` (the main checkout's, no source newer
than the binary), `make config MODEL=flash-next`'s flags (hq-e8-2b, 262,144
tokens per lane, 3 lanes, `--prefill-chunk 8192`, `--vram-headroom-bytes 4G`,
MTP off, ADR 0045's offloaded pool) plus `--metrics`, and
`--kv-host-pool-bytes 0 --retained-host 0` for host RAM (46 GB available;
neither moves VRAM). The harness is the 2026-10-08 A-B-A's: one greedy lane
back to back for 70 s (four 1,800-token requests of one prompt), then three
distinct greedy prompts for 100 s; `/metrics` deltas per phase. The plan gave
the expert cache 15.70 GB. Then two graph-level Nsight Systems windows
(`nsys profile --trace=cuda --sample=none --delay --duration 6`), one at one
lane (586 rounds) and one at three (243 rounds), read with the 2026-10-06
`analyze.py graph`. One GPU lock hold, nothing else on the card.

| | 1 lane (per token = per round) | 3 lanes (per round of 3 tokens) |
|---|---:|---:|
| tok/s (rate run) | 101.1 / 101.5 / 101.7 / 101.6 | 109.1 aggregate (first requests), 105.4 (second) |
| mean round (from tok/s) | **9.85 ms** | **27.5 ms** |
| ITL p50 | 9.5 ms | 26.5-27.5 ms |
| demand-copy stall (`ignis_expert_residency_stall_seconds_total`) | **1.84 ms (19%)** | **15.0 ms (55%)** (5.01 per token) |
| device idle between replays (trace) | **0.87 ms (8.5%)** | 1.17 ms (4.7%) |
| kernels + prefetch joins (the rest of the replay) | **7.2-7.6 ms (73-77%)** | ~11.2 ms (41%) |
| decode hit rate | 96.59% | 91.05% |
| misses / prefetched / used per token | 32.7 / 46.5 / 22.4 | 81.7 / 56.5 / 38.3 |
| link bytes per token | 51.6 MB | 97.5 MB |
| link busy per round at 12.3 GB/s | 4.2 ms (43%) | **23.8 ms (87%)** |

- The trace windows: one lane, period 10.26 ms, replay 9.40 ms (tracing adds
  ~4% against the untraced 9.85 ms); three lanes, period 24.74 ms, replay
  23.59 ms. The three-lane window ran ~10% faster than the rate run's
  phase (a different place in the texts: misses follow the text), so the
  table takes tok/s from the rate run and only the proportions from the trace.
- The host synchronizes on the round's pageable 4-byte device-to-host
  `cudaMemcpyAsync`, not on `cudaStreamSynchronize` (13 µs): that copy's
  return to the next `cudaGraphLaunch` is 0.375 ms at one lane, 0.617 ms at
  three. The idle between replays is that host time plus the graph launch's
  submission.
- The stall counts demand copies only; a prefetch still copying at the next
  layer's join is in the "kernels" row (2026-10-07, lookahead41).

Against the earlier findings, at one lane: 2026-10-06 dec2 (131K, 16.14 GB of
cache) 99.0 tok/s greedy with 1.0-1.35 ms of demand copies; lookahead41
(262K x 3, 14.34 GB) 93.6-93.9 tok/s, stall 2.11 ms; the rows-scaled budget
97.7 tok/s, 2.24 ms, 58.6 MB; the 2026-10-08 B leg (15.62 GB) 101.7-102.1
tok/s, 1.853 ms, 52.1 MB. At three lanes: lookahead41 97.1 aggregate, 6.37
ms per token; the B leg 109.7, 4.92 ms, 98.2 MB. Today's `main` repeats the
B leg within run noise (P2/P3 do not touch the round).

### 2. The prediction ceiling (oracle replay, CPU)

**Setup.** The residency policy model (`crates/core/src/residency/policy.rs`)
replayed over the converter's own routing traces
(`IGNIS_FLASH_NEXT_TRACES=F:/ai/models/Qwen3.8-Flash-Next-ignis`: 66 test
chunks over eleven domains, teacher-forced quantized-stream routing, the
lookahead router's top-20), with the expert cache at the served 15,615,494,736
bytes split as the plan splits it, warm-started from calibration, 3 lanes'
minimum slots. Per chunk: 256 tokens to settle, then 512 measured; three
lanes run consecutive chunks together. The prefetch budget is today's
(1,172,500 B at one row, +726,250 B per further row). Predictors, each a
different lookahead list handed to the same model:

- **router W**: today's — layer L's MoE input through router L+1 (layout.md
  §12), its top W;
- **filtered W**: the router's top W, keeping only experts layer L+1
  actually selects (today's ranking with a perfect confidence filter);
- **oracle**: layer L+1's actual top-10, router order (a perfect predictor).

Demand and link times are bytes at 12.3 GB/s (the replay has no timing: a
prefetch inside the budget counts as hidden).

| predictor, budget | 1 lane: hit, demand + prefetch MB/token, used, link / demand ms per round | 3 lanes: same, per round |
|---|---|---|
| none | 93.23%, 47.4 + 0, -, 3.85 / 3.85 | 88.47%, 80.7 + 0, -, 19.68 / 19.68 |
| **router W = 16, rows (today)** | 94.18%, 40.3 + 33.5, 52%, 6.00 / **3.28** | 90.55%, 65.9 + 37.9, 65%, 25.33 / **16.08** |
| router W = 20, rows | 94.14%, 40.6 + 37.9, 49%, 6.38 / 3.30 | 90.54%, 66.0 + 38.1, 65%, 25.39 / 16.09 |
| router W = 16, no budget | 96.36%, 24.9 + 114.7, 46%, 11.35 / 2.02 | 94.10%, 40.6 + 189.9, 48%, 56.23 / 9.91 |
| filtered W = 16, rows | 94.81%, 35.3 + 12.1, 100%, 3.85 / 2.87 | 92.42%, 53.1 + 27.5, 100%, 19.68 / 12.96 |
| filtered W = 20, rows | 94.95%, 34.2 + 13.2, 100%, 3.85 / 2.78 | 92.61%, 51.9 + 28.8, 100%, 19.68 / 12.66 |
| **oracle, rows** | 95.51%, 30.1 + 17.3, 100%, 3.85 / **2.45** | 93.18%, 48.0 + 32.7, 100%, 19.68 / **11.70** |
| oracle, rows x 2 | 98.05%, 14.0 + 33.4, 100%, 3.85 / 1.14 | 96.56%, 24.3 + 56.5, 100%, 19.69 / 5.92 |
| oracle, no budget | 99.82%, 1.2 + 46.2, 100%, 3.85 / 0.10 | 99.66%, 2.4 + 78.4, 100%, 19.70 / 0.58 |

- **Today's predictor.** The router's top 10 / 16 / 20 hold 63.2% / 76.1% /
  80.6% of the next layer's selection. With no prefetch, 61% (one lane) to
  64% (three) of the experts that miss were in its top 16 and 68-71% in its
  top 20, so 29-39% of the misses are beyond its reach at any width.
  At the rows budget 52% (one lane) and 65% (three) of what it prefetches is
  used; served, 48% and 68%.
- **Replay against served.** Three lanes agree (hit 90.55 against 91.05%,
  103.8 against 97.5 MB per token). One lane does not (94.18 against 96.59%,
  73.8 against 51.6 MB): the served harness repeats one greedy text four
  times, so the cache holds its working set; the replay's real text is
  harder. The estimates below therefore use ratios of replayed demand times,
  applied to the served stall.

The decomposition, demand ms per round:

| step | 1 lane | 3 lanes |
|---|---|---|
| width (W 16 -> 20, rows budget) | 3.28 -> 3.30 | 16.08 -> 16.09 |
| precision (router 20 -> filtered 20) | 3.30 -> 2.78 (-16%) | 16.09 -> 12.66 (-21%) |
| recall (filtered 20 -> oracle) | 2.78 -> 2.45 | 12.66 -> 11.70 |
| budget (oracle x1 -> x2 -> none) | 2.45 -> 1.14 -> 0.10 | 11.70 -> 5.92 -> 0.58 |

### 3. One routed expert on the CPU

**Setup.** A standalone Rust bench (`.scratch/fn-study/cpu-expert/`, AVX2 +
FMA + F16C, `-C target-cpu=native`) reads expert records from the artifact
with plain file reads into heap memory. The host expert pool is
`cudaHostAlloc(Mapped | Portable)` (`kernel/src/residency.cu`), cacheable,
not write-combined, so a CPU reads it as it reads the heap: the same DRAM
path. One expert is the whole SwiGLU: `y = had128(had128(x o suh) . W_rot) o
svh` for gate/up (2560 -> 1280), SwiGLU, then down (640 -> 2560), weights
decoded from the trellis in registers (layout.md §3, `trellis_decode.cuh`),
fp32 accumulation, the decoded value kept as `bsum * kinv + (1024 kinv +
kbias)` without the fp16 rounding. A spinning thread pool (workers never
sleep), one 128-column block per work item; timed at layer 24 over 48
experts (73.4 MB, more than the 20 MB L3, so the weights stream from DRAM),
20 repetitions; i9-10900K (10 cores, 20 threads), no server loaded, the host
otherwise 7-11% busy.

**Correctness.**
- Decode: **48 of 48** projections bit-exact against
  `references/trellis_checksums.json` (exllamav3's `reconstruct`, every K
  class, both projections, layers 0, 24 and 47).
- The MoE block on recorded activations (`references/moe_block/L02, L24,
  L46`, four tokens each, the recorded selection and weights): the CPU's
  routed sum is within **3.1e-4** relative L2 of an f64 reference over the
  same decoded fp16 weights, and the block (CPU routed sum + recorded shared
  expert) is within **6.5e-3** of the recorded output, where the f64
  reference also sits at 6.5e-3 (the reference's own BF16 roundings). The GPU
  test `moe_artifact_gpu.rs` holds the kernels to 1.5e-2 against the same
  recording. The CPU result was not compared with the GPU kernel's output
  directly.

**Time per expert** (gate/up + down, one token unless noted; the link is the
expert's 1.53 MB mean at 12.3 GB/s = **124 µs**):

| CPU path | 1 thread | 10 threads |
|---|---:|---:|
| per-element `vpgatherdd` of each state's window | 6.3 ms | 690 µs (5.5x the link) |
| eight tiles per vector, one trellis step at a time, no gather | 1.47 ms | 238 µs |
| + vectorized tile transpose | 0.94 ms | 182 µs |
| + one pool thread per physical core (pinned) | **0.94-1.0 ms** (8x) | **112-132 µs p10-p50 (0.9-1.07x)**; four experts per dispatch 123-164 µs; one expert for three rows 194-216 µs |

- At 10 pinned threads a dispatch splits into gate/up 85 µs, down 45 µs and
  ~11 µs on the calling thread alone (input rotations, SwiGLU); an empty
  dispatch costs 0.4 µs. 20 threads (with SMT) are no faster (143 µs).
- Single-threaded the path runs ~0.9 cycles per weight. Inferred, not
  measured: the gather path is slow because AVX2 gathers are slow on this CPU
  under its microcode (the Downfall / GDS mitigation).

## Finding

Observed:

1. **One lane is kernel-bound.** Of a 9.85 ms token, 7.2-7.6 ms is kernels
   (and prefetch joins), 1.84 ms the demand-copy stall and 0.87 ms the device
   idle between replays; the link is busy 43% of the round.
2. **Three lanes are link-bound.** Of a 27.5 ms round, 15.0 ms is the stall
   and the link is busy 23.8 ms (87%).
3. **Today's predictor is the hidden state of layer L through router L+1**,
   already. Its width does not matter at the rows budget (W 20 = W 16), its
   precision does (48-52% of one-lane prefetches used, 65-68% at three), and
   29-39% of the misses lie outside its top 20 or 16 at any width.
4. **A perfect predictor at today's budget removes 25% of the one-lane demand
   time and 27% of the three-lane one** (replay). The budget binds even for
   it: only at twice the budget or none does the stall go (63-97%).
5. **One expert costs the CPU about what the link costs it**: 112-132 µs on all
   ten cores pinned and spinning, against 124 µs of copy; 0.94-1.0 ms on one
   core. The decode is exact and the block is within the GPU test's bound.

Inferred (tok/s from the 2026-10-08 A-B-A calibration, where the mean round
moved one for one with the stall: 0.40 ms per token against a 0.41 ms stall
drop at one lane, 2.76 against 2.78 ms per round at three; that calibration
came from a cache-size change, so applying it to precision at three lanes is
an extrapolation that plausibly understates the gain, since wasted prefetch
bytes queue demand copies on a link 87% busy):

| | 1 lane (101.5 tok/s today) | 3 lanes (109.1 aggregate today) |
|---|---|---|
| precision: perfect filter on today's ranking, same budget | +2.4-2.9% | +11.8-13.1% |
| perfect predictor, same budget | +5.0% | +17.5% |
| perfect predictor, twice the budget | +13.9% | +49% (link floor) |
| perfect predictor, no budget | +22% (its 3.85 ms of link per round hidden in ~8 ms) | +49% (link floor) |
| **link floor** (every needed byte, none wasted, all hidden) | - | 18.5 ms per round, 162 tok/s |

- The three-lane link floor is the replay's no-prefetch demand bytes (80.7
  MB per token) scaled to the served traffic (x 0.94): 227 MB per round, 18.5
  ms at 12.3 GB/s, above the round's ~12.5 ms of compute and idle. No
  predictor beats it without a larger cache or fewer bytes on the link.
- **At three lanes the perfect predictor at today's budget is only +17.5%**:
  better prediction alone does not fix three-lane decode; the budget and the
  wasted bytes do, and a predictor that knows its confidence is what lets
  the budget grow.

## Implications

The ranking, each with its estimated gain (inferred) at one and three lanes:

1. **Kernel and launch structure (one lane's lever).** 7.2-7.6 ms of
   kernels where the 2026-10-06 capture's DRAM traffic (~5.3 GB per round,
   21.7% of 1,792 GB/s over 13.6 ms) puts the bandwidth floor near 3 ms, plus
   0.87 ms of idle between replays. Realistic from the 2026-10-06 follow-ups
   (the one-CTA `router_select`, the 6-CTA GDN b/a GEMVs, the host-gap fixes
   2, 3 and 5, fewer graph nodes for the ~0.4 ms `cudaGraphLaunch`
   submission, taking the host off the path between replays): **-0.5 to
   -1.5 ms, +5-18% at one lane**; at three lanes **+3-8%**, capped at +15% by
   the link (27.5 -> 23.8 ms).
2. **Prefetch precision and a confidence-led budget (three lanes' lever).**
   Not width, not a new input: a filter that drops the router's candidates
   the next layer will not select, then a budget that grows with confidence.
   **+2-3% at one lane, +12-13% at three** for a perfect filter at today's
   budget; up to the link floor (+49%) at three lanes and +14-22% at one lane
   only for a predictor that is right and is given more bytes. Recall beyond
   the router (the 29-32% of misses outside its top 20) needs a different,
   learned predictor; worth +2% at either lane count at today's budget
   (filtered 20 -> oracle).
3. **CPU cold experts: park.** At one lane **~0%**: a token misses 0.68
   projections per layer and the CPU is no faster than the link for one
   expert. At three lanes, as a second channel beside the link at 1.0-1.3x
   its time per expert (the p50s above), the stall could fall by 43-50%:
   **+31-37%** (27.5 -> 20-21 ms per round) before a per-layer GPU-to-CPU
   handshake inside the graph (unmeasured; at the 2026-10-05
   host-orchestrated round trip of 57 µs per layer it costs 2.7 ms per
   round, leaving ~+16-21%), before the CPU competes with the n-gram
   gather's readers and the server's threads for the ten cores it needs
   spinning, and before the re-misses of experts it computed without caching.
   Large engineering cost, three-lane only.
4. **Nothing**: 101.5 tok/s at one lane, 109.1 aggregate at three.

## Limits and unknowns

- One served run per configuration, greedy prose prompts, contexts under 2K;
  the one-lane harness repeats one text and flatters the cache (the replay's
  real text is harder at one lane).
- The replay is teacher-forced routing with no timing: its prefetches are
  hidden by assumption, and the tok/s column applies served stall ratios.
- The one-lane ratios are measured on text the served harness does not
  resemble (73.8 against 51.6 MB per token): on the repeated greedy text,
  whose working set is already resident, the predictor's gains may be
  smaller, the recall step's most. Three lanes do not carry this (replay and
  served agree within 6%).
- The kernel row includes the prefetch joins the stall metric cannot see;
  the ~3 ms DRAM floor comes from one capture of an older build and is a
  bound, not a target.
- The CPU path was timed on a quiet host with ten pinned, spinning cores; in
  the server those cores also run the n-gram gather (on the path between
  rounds), tokio and detokenization. A CPU-computed expert is never admitted
  to the cache: whether to also copy it (bytes back on the link) or let it
  miss again is the first question a design would have to settle. The GPU
  side (a wait inside the graph, the hidden state to the host and the result
  back) was not built or timed.
- The CPU kernel is not tuned to its end (~0.9 cycles per weight on one
  core); a faster one moves item 3's three-lane bound, not the one-lane
  verdict, unless it beats the link by a wide margin.
- A learned predictor and a lookahead two layers ahead were not evaluated:
  the traces carry only layer L's ranking for layer L+1.

Raw material, untracked, in the main checkout's `.scratch/fn-study/`:
- `oracle-replay.patch` (the replay test's study variant, run as
  `FN_STUDY=1 FN_STUDY_SETTLE=256 FN_STUDY_STEPS=512
  IGNIS_FLASH_NEXT_TRACES=F:/ai/models/Qwen3.8-Flash-Next-ignis cargo test -p
  ignis-core --test expert_residency_study_replay fn_study_prediction_ceiling
  -- --nocapture`), `oracle-replay-settle256-512.log`, and
  `oracle-replay.log` (256 tokens from the warm start, no settle);
- `cpu-expert/` (the bench: `check`, and `PIN=1 THREADS=1,2,4,8,10,20 REPS=20
  bench`), `cpu-check*.log`, `cpu-bench-L24.log` (gather),
  `cpu-bench-L24-sliced.log`, `cpu-bench-L24-final.log`;
- `rate/` (`rate.sh`, `trace.sh`, `analyze.py`, `summary.py`, the `main1.*`
  logs and metrics, the `g1lane` / `g3lane` captures: the `.nsys-rep` and
  `.sqlite` embed the environment, never commit them).

## Follow-ups

- Status lives in https://github.com/gpillon/ignis/issues/306 (the decode
  headroom items) for the kernel and launch work of implication 1.
- A confidence filter for the router lookahead (implication 2) has no issue
  yet: what it would read (the router's logit margin, the candidate's rank,
  its recent use) is a design question for a spec.
