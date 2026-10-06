# A one-lane Flash-Next decode round is launch-bound, not bandwidth-bound: single-CTA kernels and host time between rounds own the gap to the simulation

- Kind: experiment
- Status: current
- Observed: 2026-10-06
- Last verified: 2026-10-06
- Scope: kernel / Flash-Next decode round, CUDA graph replay, expert residency on the critical path, host time between rounds
- Related: https://github.com/gpillon/ignis/issues/306, [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md), [27B decode round host idle](2026-09-18-decode-round-host-idle.md), [MoE decode is structure-bound](2026-10-05-moe-decode-is-structure-bound.md), [expert miss path](2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md)
- Superseded by: none

## Question

Served Flash-Next decodes at ~65 tok/s at one lane (15.3 ms per token), against
the simulation's 102 tok/s. Two parts of the step were already known:
- the routed MoE kernels, ~1.2 ms per token;
- ~31 MB per token of expert misses over the PCIe Gen 3 link.

The remaining ~11 ms was never decomposed. Where does a decode round's time go?

## Evidence

**Setup.**
- Build: `main` at `d83bb3e`, release build.
- Artifact: `qwen3_8_flash_next_trellis_a25-v2.ninfer`.
- Server flags: the ones `make config MODEL=flash-next` prints (hq-e8-2b, `--max-context 131072`, `--vram-headroom-bytes 4G`), plus `--prompt-reuse off --retained-host 0 --kv-host-pool-bytes 0`. Those three only shrink the host plan so it fits the RAM that was free; none of them is on the decode path.
- One client streams back-to-back 1,800-token generations (`ignore_eos`) of a ~150-token prompt, so the context stays under 2K.
- Each capture window (5-6 s) sat inside one generation's decode. In the two graph-level windows no eager kernel ran: pure decode.
- Exclusive card, GPU lock held.

**Captures.** Each one is an `nsys profile --trace=cuda --sample=none` run of the server:
- node-level (`--cuda-graph-trace=node`): 278 complete rounds;
- graph-level: 383 rounds;
- graph-level plus `--gpu-metrics-set=gb20x`: 369 rounds.

The server's `/metrics` was scraped every 5 s during the GPU-metrics run.

**Method.**
- A round is every node carrying its `cudaGraphLaunch`'s correlation id.
- The graph runs on several streams: residency forks its prefetch `copy_jobs` onto a side branch.
- Every instant of the round is charged to the node that owns it. A non-prefetch node beats a concurrent prefetch copy. An instant with only a prefetch copy running is "exposed prefetch". An empty instant is idle.
- Nodes are bucketed by position: `hc_norm` opens a sublayer's mix and `hc_inject_kernel` closes the sublayer. Sublayer *s* is layer *s*/2; even sublayers are attention (QSA when (layer+1) % 4 == 0, else GDN), odd sublayers are MoE.
- Node tracing inflates the round by 4.6% (span 14.22 ms against a graph-level replay of 13.60 ms). Totals and wall time come from the graph-level captures.
- Scripts: `analyze.py` (`node2`, `graph`), `kinds.py`, `extra.py`, `gm.py`.

### Wall time against device time (graph-level)

| per round | capture 1 | capture 2 (+GPU metrics) |
|---|---:|---:|
| period (launch to launch) | **15.70 ms** (63.7 tok/s) | 16.48 ms (60.7 tok/s) |
| graph replay on the device | **13.60 ms** | 13.83 ms |
| host work: sync returns → next CUDA call | 1.49 ms (p50 1.62, p90 2.24) | 1.98 ms (p50 1.87, p90 2.50) |
| `cudaGraphLaunch` call itself (1,774 nodes) | 0.42 ms | 0.46 ms |
| staging copies (6 `cudaMemcpyAsync`) | 0.08 ms | 0.07 ms |
| launch call ends → replay starts | 0.03 ms | 0.04 ms |
| replay ends → host sync returns | 0.09 ms | 0.10 ms |

- The device is idle **2.1-2.65 ms per round** (13-16% of wall), all of it between replays. Inside a replay it is idle 0.11 ms.
- For comparison, the 27B's round (graph capture `decode-B1lane` of the [27B finding](2026-09-18-decode-round-host-idle.md), re-read with the same script) makes its next CUDA call 44 µs after its sync returns. Its round also launches eager work, so the comparison is loose.
- No-capture reference, from the server's counters with nsys attached but no window open: 909 tokens in 15 s = 60.6 tok/s. The earlier finding's 65.3 tok/s ran with a 1G VRAM headroom, so a ~3 GB larger expert cache.

**GPU metrics during replays:**
- SMs active 33%, SM issue 6.9%;
- DRAM read bandwidth **21.7%** of peak;
- PCIe RX 4.9% of the metric's scale.

The replay is not bandwidth-bound.

### Inside a replay (node-level, per round, µs)

The round has **1,774 nodes**. 1,200 of them run under 10 µs, 4.44 ms in all.

By layer kind (node sums; the HC mix and inject are charged to the sublayer they serve; the prefetch copies are left out):

| kind | µs/round | of which |
|---|---:|---|
| MoE blocks (48) | **7,958** | HC mix 2,556; routed experts 1,283; demand copies 1,279; `resolve` 1,000; router + lookahead router 588; shared expert 496; `rank_lookahead` 428; combine 280; inject 47 |
| GDN attention halves (36) | **4,014** | GDN body 2,046 (FP8 projections 1,748); HC mix 1,935; inject 33 |
| QSA attention halves (12) | **1,855** | QSA body 1,196 (FP8 projections 538, indexer `score_kernel` 309, `hq_rows` 84, `attend` 75); HC mix 648; inject 12 |
| head + sampling | 429 | head GEMV 377 (636 MB FP8, ~94% of DRAM bandwidth); final HC mix 37 |
| n-gram add (layer 1) | 37 | |
| embed | 5 | |

**The HC mix is the largest single cost: 5,176 µs per round (36%).** It runs 97 times per round at ~53 µs each. A mix is six kernels:

| kernel | grid × block | µs | what it does |
|---|---|---:|---|
| `hc_norm` | **1 × 256** | 9.5 | RMS norm of the 4 × 2560 residual streams |
| `fp8_gemv_kernel` | 40 × 256 | 12.1 | `mix_down`, 10240 → 320 (3.3 MB) |
| `hc_down_activation` | 2 × 256 | 0.8 | |
| `fp8_gemv_kernel` | 1280 × 256 | 4.4 | `mix_up`, 320 → 10240 (3.3 MB) |
| `bf16_gemv` | **1 × 256** | 16.3 | `block_inject`, 10240 → 4 (82 KB) |
| `hc_reduce` | **1 × 256** | 10.5 | |

A mix reads ~6.6 MB of weights, ~3.7 µs at the card's bandwidth.

**Single-CTA kernels on the critical path, per round:**

| kernel | grid × block | µs each | per round |
|---|---|---:|---:|
| `bf16_gemv` (block inject) | 1 × 256 | 16.3 | 1.58 ms |
| `hc_reduce` | 1 × 256 | 10.5 | 1.02 ms |
| `resolve` | 1 × 1024 | 20.8 | 1.00 ms |
| `hc_norm` | 1 × 256 | 9.5 | 0.93 ms |
| `rank_lookahead` | 1 × 256 | 9.1 | 0.43 ms |
| `router_select_kernel` | 1 × 256 | 3.5 | 0.33 ms |
| MoE `combine_kernel` | 1 × 256 | 5.8 | 0.28 ms |

Together they take **5.57 ms of the 14.22 ms node-level round (39%)** on one SM of 170. The 6-CTA FP8 GEMVs (the GDN b/a projections, 4.0 µs each, 72 per round) add 0.29 ms.

**Expert copies.**
- Every MoE layer launches two `copy_jobs`:
  - the demand copy runs after `resolve` on the critical path;
  - the prefetch copy runs on the side branch, joined by the next layer's step.
- Demand copies: **1,270 µs per round, all exposed**. 9.6 layers per round have one, ~132 µs each (~1.6 MB at the link's ~12.4 GB/s); the other layers take 0.6 µs.
- Prefetch copies: ~3,580 µs per round, of which **43 µs is exposed**. The rest overlaps other work (3,533 µs).
- Contention from a concurrent prefetch copy is small: +3% on `hc_norm`/`hc_reduce`, +8% on `combine`, nothing on the routed experts.
- Counters over the 15 s before the window: **59.5 MB moved per decoded token**. That is 4.8 ms of link time per round, matching the 4.85 ms of `copy_jobs` time. The hit rate per projection is 97.35% (934.5 hits, 25.4 misses per token). The earlier finding measured 31 MB per token at 99.0-99.2% with the 1G-headroom cache.

**N-gram.**
- 12.2 file reads and 3.8 hot-row hits per token (51 KB read).
- The gather is synchronous in `FlashNextLeaf::decode`, before the round's staging.
- An emulation outside the engine (`ngram_reads.ps1`: 12 random 4 KiB unbuffered reads of the artifact over 4 threads, as `DEFAULT_READ_THREADS`) takes 0.31 ms p50 (0.36 mean) per round. One read takes 0.11 ms.

## Finding

Observed:

1. **A one-lane round is 13.6 ms of device replay plus 2.1-2.65 ms of device idle between replays.** The replay runs at 22% of DRAM bandwidth with 33% of SMs active. It is latency-bound: 1,774 nodes, 1,200 of them under 10 µs.
2. **Single-CTA kernels hold 5.57 ms of the 14.22 ms node-level round (39%).** The HC mix holds 5.18 ms (36%); three of its six kernels run on one CTA, and a fourth (`mix_down`) runs on 40 CTAs.
3. **Expert residency costs 2.7 ms per round on the critical path:**
   - `resolve` (1 CTA, 20.8 µs × 48) plus `rank_lookahead` (1 CTA, 9.1 µs × 47): 1.43 ms;
   - exposed demand copies: 1.27 ms.

   The prefetch copies are hidden: 43 µs exposed out of ~3.6 ms.
4. **The routed experts (1.28 ms) and the head (0.38 ms, ~94% of bandwidth) are not where the time goes.** Neither are the large GDN/QSA projections, the n-gram add (0.04 ms) or the prefetch copies.
5. **At the 4G-headroom default the decode hit rate is 97.35% and a token moves 59.5 MB**, about twice the 31 MB measured with the 1G headroom.
6. **The host spends 1.5-2.0 ms per round between `cudaStreamSynchronize` returning and the next round's first CUDA call**, where the 27B makes its next CUDA call after 44 µs. The `cudaGraphLaunch` call adds 0.42-0.46 ms.

Inferred (not measured):

- The n-gram gather plausibly accounts for ~0.3-0.4 ms of the host gap (emulated, not traced). The other ~1.1-1.6 ms is unattributed.
  - These captures have no CPU samples: `--sample=process-tree` needs administrative rights on Windows and silently records nothing without them (tested).
  - The candidates are the Flash-Next scheduler loop, token emission and detokenization, and the leaf's per-round allocations.
- The simulation's ~6 ms step assumed bandwidth-bound kernels. At one lane the step is dominated by per-kernel latency and serial bookkeeping instead.

## Implications

Targets ranked by estimated gain per token at one lane, from the 15.7 ms period. The gains overlap and do not simply add.

1. **The HC mix: −2.8 to −3.7 ms.** Spread `hc_norm`, `hc_reduce` and the 4-row `block_inject` GEMV over the SMs (split-K, or fuse into the neighbouring kernels), and give `mix_down` more than 40 CTAs. A mix moving 6.6 MB should cost ~15-20 µs, not 53.
2. **Host time between rounds: up to −1.5 to −2.0 ms.**
   - First name it: time spans in the leaf and scheduler, or nsys CPU sampling from an elevated shell.
   - Then take what does not depend on the drawn token off the path between sync and launch.
   - Fewer graph nodes also shrink the 0.45 ms `cudaGraphLaunch` call.
3. **Residency bookkeeping: −1.0 to −1.4 ms.**
   - Parallelize `resolve` (one 1,024-thread CTA today).
   - Move the lookahead router and `rank_lookahead` (~0.7 ms with the lookahead router) to the prefetch branch; only the prefetch consumes them.
4. **Demand copies: about −0.6 ms** by giving back ~3 GB of expert cache. This trades against spec 04 AC10's 29 GB bound. Better lookahead coverage is the other route.
5. **Serial small work: −0.8 to −1.0 ms.**
   - The shared expert (0.50 ms) depends only on `x`: it can run as a parallel graph branch beside router → resolve → demand copy.
   - The 1-CTA MoE combine (0.28 ms), the 1-CTA `router_select` (0.33 ms) and the GDN b/a GEMVs on 6 CTAs (0.29 ms) can be widened or fused.

## Limits and unknowns

- One lane, contexts of ~0.2-2K, one prompt family. No repeats beyond three captures of one build, but the per-round figures agree within ~5% across the two graph-level captures.
- The host gap's composition is inferred. Only its total and the `cudaGraphLaunch` share are measured.
- Per-kernel costs come from the node capture: +4.6% over the graph replay.
- The n-gram read time is an emulation with .NET unbuffered reads, not the engine's `ReadPool`.
- The decode rate drifts between runs (60.6-63.7 tok/s here, 65.3 in the earlier finding) with the expert cache's size and state.
- The gains above are estimates from these timings and the kernels' byte counts. None was implemented or measured.

Raw material in the `flash-next` worktree (untracked), `.scratch/fn-decode-prof-306/`:
- the `.nsys-rep` captures (they embed the process environment: never commit them) and their sqlite exports;
- `fn-node.report*.txt`, `fn-gm.metrics.*.txt`, `fn-gm.client.jsonl`;
- the scripts named above.
