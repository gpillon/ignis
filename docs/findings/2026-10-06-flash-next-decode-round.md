# A one-lane Flash-Next decode round is launch-bound, not bandwidth-bound: single-CTA kernels and host time between rounds own the gap to the simulation

- Kind: experiment
- Status: current
- Observed: 2026-10-06
- Last verified: 2026-10-07
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

- The n-gram gather plausibly accounts for ~0.3-0.4 ms of the host gap (emulated, not traced). The other ~1.1-1.6 ms is unattributed. (Since measured: the gather is nearly all of the gap. See "Host gap, named" at the end.)
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
- The host gap's composition is inferred. Only its total and the `cudaGraphLaunch` share are measured. (It was named later, in "Host gap, named" at the end.)
- Per-kernel costs come from the node capture: +4.6% over the graph replay.
- The n-gram read time is an emulation with .NET unbuffered reads, not the engine's `ReadPool`.
- The decode rate drifts between runs (60.6-63.7 tok/s here, 65.3 in the earlier finding) with the expert cache's size and state.
- The gains above are estimates from these timings and the kernels' byte counts. Item 1, item 2's fix 1, item 3's move of the lookahead off the critical path (not a parallel resolve) and item 5's shared expert and combine were since implemented and measured (attempts below); item 4 was not.

Raw material in the `flash-next` worktree (untracked), `.scratch/fn-decode-prof-306/`:
- the `.nsys-rep` captures (they embed the process environment: never commit them) and their sqlite exports;
- `fn-node.report*.txt`, `fn-gm.metrics.*.txt`, `fn-gm.client.jsonl`;
- the scripts named above.

## Attempt 2026-10-06 (hcmix)

Implication 1, the HC mix: `38119ea` and `e04d800` on `fn-perf-306`. At decode widths a mix is now three launches, each spread over the card:
- **`hc_norm`**: one CTA per (stream, row) instead of one per row. It also computes the 4-output block-inject matvec as per-stream partials, so the single-CTA `bf16_gemv` is gone on every route. Each thread issues its loads before the reduce and sums squares in the old order, so `normed` is bit-identical to before.
- **`hc_down_split`**: `mix_down` split by stream over 40 × 4 = 160 CTAs, into fp32 partials.
- **`hc_up_reduce`**: `mix_up` and the stream reduce in one 160-CTA kernel. Each CTA rebuilds the activation from the down partials; one CTA also writes the injection weights from the norm's partials.

This route takes up to 8 rows and 4 streams, with BF16 or FP8 projections. Wider calls (prefill) keep `fn_linear` for both projections, with the new norm and inject. Every BF16 rounding of the module stays where it was; only fp32 summation order changes. Partials are summed in a fixed order, so a replay gives the same bits.

**Setup.**
- "Before" is the release build this finding measured. It was built at 09:59, before `d83bb3e`'s merge commit; nothing on the decode path changed in between.
- Served rates come from `.scratch/hcmix-306/rate.sh`: the server flags above, no profiler, 1,800-token generations of the same prompt. One client runs for ~70 s, then three concurrent clients for ~100 s. A request's rate runs from its 50th token to its last.
- Node-level figures come from `trace.sh` + `analyze.py node2` (the mix now ends at `hc_up_reduce`), 278 rounds.
- The microbenchmark is `ignis_kernel_flash_next_hc_bench`: FP8 `mix_down`/`mix_up` cycled over 24 weight sets so they stream from DRAM, 48 calls in a graph, the median replay per call.

| | before | after (`e04d800`) |
|---|---:|---:|
| HC mix, node-sum per round | 5,176 µs | **1,660 µs** |
| one mix, in situ (node-level) | 53 µs, 6 kernels | 17.1 µs: norm 6.3, down 5.1, up + reduce 5.8 |
| node-level round span | 14.22 ms | 11.24 ms |
| served, 1 lane, tok/s (requests 1/2/3) | 63.9 / 63.7 / 64.1 | **78.6 / 78.5 / 80.5** |
| served, 1 lane, ITL p50 | 15.4-15.5 ms | 12.0-12.5 ms |
| served, 3 lanes, tok/s per lane (requests 1/2/3) | 38.8 / 36.7 / 33.6 | 45.3 / 43.4 / 43.9 |
| served, 3 lanes, aggregate (first requests) | 116.4 | **136.0** |

| microbenchmark, µs per mix | 1 row | 2 | 3 | 8 | 256 | 2048 |
|---|---:|---:|---:|---:|---:|---:|
| before | 42.7 | 47.9 | 59.5 | 105.6 | 544 | 1,266 |
| after | 14.0 | 15.7 | 17.9 | 27.4 | 346 | 839 |

Per op at one row in the microbenchmark (node-traced):
- before: norm 5.9, `mix_down` 12.0, activation 0.7, `mix_up` 4.2, `block_inject` 9.0, reduce 10.2 µs;
- after: norm 2.6, down 4.6, up + reduce 5.0 µs.

Correctness: the HC CTest now has FP8 arms beside the BF16 ones (and one projection of each format), rows 1/3/8/9/1027/1100, the final mixer, and a replay check. The arms of `38119ea` (BF16 and FP8, rows 1-1100) passed on the old kernels too, and a deliberately broken decode route fails them; the mixed-format arms, the 1027-row arm and the refusals came later and ran on the new code only. `flash_next_forward_gpu` passes 3/3 on `e04d800`:
- G1 agreement is 102/102;
- decode equals prefill on G1 real text, 0 flips in 224 tokens;
- decode is deterministic, and a graph replays what eager computes.

The reported arbitrary-id flips (3 wider at one lane, 3 + 1 near-tie at three) are in the range the test's header records.

- **The one-lane ITL fell by 3.2 ms per token**, inside the −2.8 to −3.7 ms estimated above.
- **In situ the norm costs more than in the microbenchmark.** The first version (`38119ea`) took 12.7 µs per `hc_norm` in the served round against 4.2 µs in the microbenchmark: its strided loop waited on one load per element. At that version the mix was 2,297 µs per round, one lane ran 78.5 / 75.4 / 79.4 tok/s, and three lanes 127.8 aggregate. `e04d800` issues every load before the reduce: 6.3 µs in situ, 2.6 in the microbenchmark. The remaining in-situ gap is not isolated. Candidates are the residual the inject wrote just before, and the concurrent prefetch copy.
- **Three-lane rates fall request over request inside one run, in both builds**, so compare request ordinals, not runs. Within a build, one-lane rates vary by up to 4 tok/s across requests.
- The mix is now 1.66 ms of an 11.24 ms node-level round. The next items by size are residency bookkeeping (implication 3) and the host gap (implication 2).

Raw material in `.scratch/hcmix-306/`: `rate.sh` with its logs (`before`, `after`, `after2`), the bench outputs and captures, `after-node` / `after2-node` (`.nsys-rep`: never commit them), and `kern.py`.

## Host gap, named (2026-10-06, admin capture)

Implication 2 asked what fills the time between `cudaStreamSynchronize` returning and the next round's first CUDA call. **It is the n-gram row gather.** The rows come from unbuffered reads of the artifact, and those reads run one at a time in the file system because the leaf keeps the whole artifact memory-mapped.

**Setup.**
- `trace_admin.sh`, run by the owner from an elevated shell: `--sample=process-tree --sampling-frequency=8000 --cpuctxsw=process-tree --cuda-graph-trace=node`.
- The build is the `fn-perf-306` release build of 17:02, with the HC mix fix.
- The window is 6 s of steady one-lane decode at +150 s: 416 rounds.
- `hostgap.py` cuts the decode thread (`ignis-model`) into gap windows (sync returns → next CUDA call). It times them from context switches, and times the `ngram-read-*` workers' waits the same way.
- `symres.py` resolves symbols offline with dbghelp, against a frozen copy of the exe and PDB.
  - The module base `0x7ff6ac340000` is inferred: 160 of 160 sampled return addresses follow a call instruction at that base.
  - Kernel frames stay unresolved.
- `dump1.py` and `dump2.py` print one round's raw events.
- Context switches set the times; the samples only name what runs. On Windows a switch-in carries the stack the thread waited in, and readying another thread also records a stack. So in windows under ~100 µs, sample counts are not time.
- Node tracing inflates the `cudaGraphLaunch` call to 1,669 µs (0.42-0.46 ms without it) and the period to 14.4 ms. The two completed requests ran at 13.5-13.6 ms ITL p50 with nsys attached; the window covers at most the tail of the second (the request open at its end was cut). Without a profiler the ITL is 12.0-12.5 ms. The gap itself is in line with the untraced captures above: 2.07 ms mean here, 1.49-1.98 ms there.

**Per round, decode thread, mean of 416 rounds (µs).**

| segment | µs | what runs |
|---|---:|---|
| sync returns → decode thread parks | 79 | `drain_tokens` hands the token to a tokio worker (a condvar notify). `NgramTable::begin_batch` hashes, runs `plan_gather` and sends one `mpsc` job per read, each send waking a reader. |
| parked in `PendingRows::finish` → `collect_and_gather` | **1,805** | 12.4 `Thread::park` → `ZwWaitForAlertByThreadId` waits, one per read result. This includes 172 µs of readied-to-running latency. |
| on-CPU between those waits | 163 | take one result, park again |
| last wake → first CUDA call | 21 | row gather into staging, `decode_flash_next` setup |
| **gap** | **2,068** | p10 1,245, p50 2,241, p90 2,816 |
| then 6 staging `cudaMemcpyAsync` (pageable, 4-1,440 B) | 112 | |
| then `cudaGraphLaunch` | 1,669 | node-traced; 420-460 untraced |

- **Without file reads there is no gap.** The 4 rounds whose rows were all hot have a 64 µs gap; the 27B's is 44 µs.
- In the gap, the decode thread waits nowhere outside `collect_and_gather`.
- Sampling readback, the residency mirror, the detokenizer/SSE and telemetry do not show on the decode thread in the gap. Token emission is the `drain_tokens` hand-off above.
- **The gap follows the read count.** A read is counted as one reader I/O wait.
  - A round makes 12.4 file reads on average, the same as the decode thread's wakes. The count runs 0-18, bimodal at 8 and 15-16.
  - A read is ~4.2 KB (`fn-gm` counters).
  - Rounds with 8 reads (136 of them) have a 1,404 µs gap. Rounds with 15 or 16 reads (217) have 2,450 and 2,629 µs.
  - Fit: gap ≈ 192 µs + 152 µs per read (R² 0.84).

**The reads run one at a time, below the pool.**
- **A read's own I/O is fast.** Its I/O wait is 105 µs mean, like the emulation's 0.11 ms per read.
- **Nearly every read first waits on an executive resource.** 11.4 of the 12.4 reads per round wait in nsys's `Resource` state inside `NtReadFile`, 410 µs mean, under `Ntfs.sys` and `FLTMGR.SYS`. Every read but about the first in a round waits for the one before it.
- The four workers issue 2-4 reads at once, yet the reads complete ~150 µs apart (the fit's slope).
- Reader time on the queue `Mutex<Receiver>` is idle time between rounds, as designed; it is not on the path.

**Cause: the artifact's memory map.**
- `FlashNextLeaf` holds the 71.8 GB artifact mapped for its whole lifetime (`Reader` → `Mmap::map`). After `open`, it uses the map only for `content_hash()`.
- An emulation reproduces the slowdown. Setup:
  - `ngram_reads_ab.ps1` is `ngram_reads.ps1` plus a flags parameter: 12 random 4 KiB reads over 4 threads, 400 rounds per arm.
  - `mmap_hold.py` maps the artifact read-only from a second process.
  - No server was running.

| arm, in run order | p50 µs per 12-read gather | mean |
|---|---:|---:|
| unmapped, unbuffered (as the engine) | 304 | 321 |
| mapped, unbuffered | 1,149 | 1,174 |
| mapped, buffered | 631 | 651 |
| mapped, unbuffered, after the buffered arm | 2,286 | 2,336 |
| unmapped again, unbuffered | 313 | 344 |

- **Mapped, the unbuffered gather is 3.7-7.5 times slower**: 96-190 µs per read, which brackets the engine's ~150. Unmapping restores 0.31 ms.
- The slowdown grows after the buffered arm. This suggests the cost scales with the file's cached pages, and the server's map is warm from the load. This is not isolated.
- What NTFS does on this path (which resource it takes, and why reads wait for it) is not traced: the kernel frames are unsymbolized.

**Fixes, ranked.** Gains are per token at one lane, from the ~12.3 ms ITL. They overlap and do not simply add.
1. **Unmap the artifact after load: about −1.4 to −1.6 ms (~+12-14% tok/s).**
   - Keep the content hash and drop the `Reader` at the end of `FlashNextLeaf::open`.
   - Check that no other handle keeps the file mapped or cached. The residency fill and the table's hot-row load run only at load.
   - Expected result: the gather runs at the emulation's ~0.3 ms for 12 reads.
   - Verify with the same admin capture: the reader waits leave `Resource`, and the gap falls to ~0.4 ms.
2. **One wake per gather: −0.1 to −0.2 ms once fix 1 lands.** Today the 12.4 park/wake cycles (~27 µs each, with the readied-to-running latency) overlap the reads. Once the reads are fast, the cycles are as long as the reads. The fix: the last reader to finish wakes the decode thread once, with a countdown.
3. **All reads in flight at once: up to −0.15 ms after fix 1.** One overlapped submit, or a worker per read, instead of 4 workers taking 2-4 waves.
4. **Fewer file reads.** In `fn-gm` (prefill plus 1,800 tokens), the 1 GiB hot cache served 24% of rows: 13,871 of 56,848. Each read saved is worth ~0.1 ms after fix 1 and ~0.15 ms before it. A larger hot set costs host RAM.
5. **Staging: −0.05 to −0.1 ms.** Replace the 6 pageable copies (112 µs here, 80 µs in the graph-level capture) with one pinned buffer and one copy, or with memcpy nodes in the graph.
6. **`cudaGraphLaunch`: 0.42-0.46 ms untraced.** Fewer graph nodes shrink it (implication 2).
7. The 79 µs before the gather (token hand-off, hash, plan, sends) is not worth taking on.

The rows depend on the drawn token, so the gather cannot move off the path as it stands. With drafted tokens (MTP), a draft's rows could be fetched during the replay.

**Limits.**
- One capture: one lane, one prompt.
- The ETW switch stacks and 8 kHz sampling add overhead to every switch: 12 per round on the decode thread, and several per read on the readers.
- The A/B ran outside the engine, through .NET `FileStream`. The engine gain from unmapping is a prediction until fix 1 is measured.
- The wait states are nsys's labels (`Resource`, `AlertByThreadId`, `NonBlocked`).

Raw material in `.scratch/fn-decode-prof-306/` (untracked):
- `fn-admin.nsys-rep` and its sqlite export: they embed the environment, never commit them.
- `hostgap.py`, `symres.py`, `dump1.py`, `dump2.py`, `ngram_reads_ab.ps1`, `mmap_hold.py`.

## Attempt 2026-10-06 (dec2)

Three targets, in order: the host gap's fix 1, residency bookkeeping (implication 3), serial small work (implication 5). Each is one commit on `fn-perf-306`, measured against the build before it.

**Setup.**
- Builds: release, one per step, each copied aside so the four ran back to back with the same flags. "base" is the 17:02 build the host-gap capture measured (the HC mix fix, nothing else).
- Served rates: `.scratch/dec2-306/rate.sh`, which is `hcmix-306/rate.sh` with the exe as a parameter. One lane for ~70 s, then three concurrent lanes for ~100 s.
  - "Sampled": the finding's prompt as it is, so the server's default sampling. A request's rate then varies by up to ~12 tok/s with its text (the expert misses follow it), so these compare only roughly.
  - "Greedy": the same prompt with `temperature` 0. Every request then decodes the same text, and the runs repeat to ±0.3 tok/s. The three greedy lanes decode the same text too, so they share their experts: their aggregate overstates three real agents.
- Node-level spans come from `trace.sh` + `analyze.py node2` (the `hcmix-306` copy), 478-503 rounds per window. The analyzer does not know the new kernel names: it files `resolve_demand` and `resolve_prefetch` under "moe.shared expert", so only the span and the named kernels are read here.

| build (commit) | 1 lane, greedy: tok/s, ITL p50 | 3 lanes, greedy: per lane, aggregate | 1 lane, sampled (requests 1-4) | 3 lanes, sampled (requests 1-3), aggregate of the first |
|---|---|---|---|---|
| base | 81.5, 12.0 ms | 64.1, 192.4 | 78.8 / 76.8 / 77.6 | 46.0 / 43.0 / 42.4, 137.9 |
| 1. artifact unmapped (`bcbe7a2`) | 88.4, 11.0 ms | 71.9, 215.6 | 88.6 / 87.2 / 86.1 / 84.7 | 45.3 / 53.9 / 48.0, 135.9 |
| 2. residency split (`c857585`) | 97.9, 10.0 ms | 80.5, 241.5 | 100.0 / 92.8 / 94.1 / 95.6 | 60.1 / 49.6 / 47.0, 180.2 |
| 3. shared expert branch, wide combine (`5490c26`) | **99.0, 9.7 ms** | **82.3, 246.9** | 98.3 / 103.3 / 100.6 / 88.7 | 54.0 / 59.9 / 52.6, 162.0 |

At one lane, greedy, the ITL falls by 2.3 ms per token: 1.0 + 1.0 + 0.3.

**1. The artifact is unmapped after load (`bcbe7a2`), host-gap fix 1.**
- `FlashNextLeaf` keeps the content hash and drops its `Reader` at the end of `open`. The n-gram table reads through its own handle; nothing else needed the map.
- A graph-level capture (`dec2-unmap-graph`, 484 rounds) puts the host's work between `cudaStreamSynchronize` returning and the next CUDA call at **470 µs** mean (p50 516, p90 673), against 1.49-1.98 ms in the captures above. Sync returns → next `cudaGraphLaunch`: 558 µs.
- The ITL fell 1.0 ms, under the −1.4 to −1.6 predicted. Inferred: the prediction started from the admin capture's 2.07 ms gap, which its tracing inflated.
- `flash_next_reuse_gpu` now checks the blob identity's artifact hash against the file's own (2/2 pass).
- Fixes 2, 3 and 5 of the host-gap list were not tried. At a predicted 0.05-0.2 ms each, they would sit inside the noise of one served run.

**2. The resolve split in two, its lookahead beside the experts (`c857585`), implication 3.**
- `resolve_demand` (hits, misses and their copy jobs) stays on the layer's stream. `resolve_prefetch` (the lookahead's candidates) runs on residency's prefetch stream. The stream forks right after the demand resolve, and carries the next layer's router and `rank_lookahead` too.
- The prefetch copy still waits for the demand copy, so the two never share the link.
- A prefetch cannot evict anything the expert op reads: every projection the step selected is stamped with the step's clock, which pins it. The layer's stream waits for the lookahead router before the next mix rewrites `x`.
- Run back to back on one stream, the two kernels are the old step. `ignis_residency_step` and `_ranked` still run them that way.
- In situ: `resolve_demand` takes **5.5 µs** per layer (264 µs per round), against `resolve`'s 20.8 µs (1,000 µs). `resolve_prefetch` takes 17.9 µs and `rank_lookahead` 10.3 µs, both on the branch.
- Node-level span: 10.22 ms (the hcmix build: 11.24 ms; different windows).
- The branch's copy now starts after ~34 µs of router, rank and resolve, not right after the demand copy. Exposed prefetch rose from 43 to 324 µs per round in this window (78 µs in the step 3 window).
- "Parallelize the resolve" beyond the split was not done: the half left on the critical path is 5.5 µs.
- `ignis_residency_resolve_bench`, synthetic and miss-heavy, at one lane: whole step 118.0 µs, demand half 32.8 µs.
- Correctness: the residency trace CTest replays the policy model's fixture as split steps too, eagerly and through 31 captured rounds (`ignis_kernel_residency_split_test`, `_graph_split_test`). The whole-step modes, `expert_residency_gpu` (617 steps), `flash_next_forward_gpu` 3/3 (G1 102/102) and `flash_next_reuse_gpu` 2/2 pass.

**3. The shared expert on a branch of its own, and a wide decode combine (`5490c26`), implication 5.**
- The forward forks a stream of its own after the MoE mix and runs the shared expert there. The combine joins it.
- At decode widths the combine spreads each row over ten CTAs. Each CTA computes the token's gate with the same reduction, so the bits do not change: the CTest checks three decode-width rows against a 4096-row call.
- Node-level span: 10.22 → 9.54 ms. Combine: 5.97 → 4.8 µs per layer.
- Served, the gain is 0.3 ms per token, under the −0.8 to −1.0 predicted for the whole of implication 5. Inferred, not measured: the shared expert's FP8 GEMVs now share bandwidth with the router and the routed experts.
- Tried and dropped: the GDN layer's small projections on the same side branch, beside its qkv projection.
  - With z, a and b on the branch, the span went 9.54 → 9.58 ms.
  - With a and b only, a greedy A/B gave 9.75 → 9.72 ms. The GDN kernels' node-sum grew by ~600 µs; inferred: GEMVs run concurrently stretch each other.
- Correctness: `flash_next_forward_gpu` 3/3 (G1 102/102; the same arbitrary-id flips as before: 3 wider at one lane, 3 + 1 near-tie at three). The `moe_shared_combine` and `moe_block` CTests pass.

**Limits.**
- One prompt, one lane or three identical ones, contexts under 2K. The greedy rows are repeatable but not representative of three agents.
- Per-target times are differences of served ITLs, not traces of each step.
- The node-level spans of steps 2 and 3 come from windows with different expert misses (demand copies 1.0-1.35 ms per round).

Raw material in `.scratch/dec2-306/` (rate logs, the builds' exes, `trace.sh`, `prompt_greedy.json`) and `.scratch/fn-decode-prof-306/` (`dec2-*`, `g-*` captures: never commit the `.nsys-rep`).

## KV format: BF16 against hq-e8-2b (2026-10-06, dec2)

hq-e8-2b stays the default. This section measures what BF16 KV would cost a decode at the same flags.

**Setup.**
- One exe for both formats: the `5490c26` build. Flags as `rate.sh`, plus `--metrics`.
- `.scratch/dec2-306/kv.sh` runs one greedy lane for ~70 s, then three lanes with three different greedy prompts for ~100 s. Three distinct texts, so the lanes do not share their experts as the identical greedy lanes above do.
- Hit rate and bytes per token are deltas of `/metrics` over each phase (`ignis_expert_cache_*_total`, `ignis_expert_bytes_moved_total`, `ignis_decoded_tokens_total`). The plan line is the load's `ignis.runtime.flash_next_plan` log event.
- The VRAM budget follows the VRAM free at load (the desktop's use varied by ~0.8 GB between runs), so the budget line differs by up to 0.5 GB.

**At `--max-context 131072` (the default), BF16 does not load.** The 3-lane pool would take ~10.8 GB and leave the expert cache 8.0 GB, below its 12 GB floor. hq-e8-2b: pool 2.05 GB, expert cache 16.14 GB.

**At `--max-context 32768`, both load:**

| | hq-e8-2b | BF16 |
|---|---:|---:|
| plan: KV pool / expert cache / budget (GB) | 0.81 / 16.58 / 26.25 | 2.84 / 15.03 / 26.74 |
| 1 lane: tok/s (requests 1-4) | 98.3 / 96.0 / 96.8 / 97.8 | 93.5 / 94.1 / 93.9 / 94.4 |
| 1 lane: ITL p50 | 10.0 ms | 10.15-10.47 ms |
| 1 lane: decode hit rate, MB moved per token | 97.45%, 59.6 | 96.40%, 73.6 |
| 3 distinct lanes: tok/s per lane (first requests), aggregate | 38.3, 114.8 | 32.0, 96.1 |
| 3 distinct lanes: ITL p50 | 25.2-25.4 ms | 30.5 ms |
| 3 distinct lanes: decode hit rate, MB moved per token | 91.63%, 80.6 | 88.63%, 101.7 |

- **BF16 costs ~3% at one lane and 16% at three.** The BF16 pool takes 2.0 GB more VRAM; the expert cache gets 1.55 GB less. The hit rate drops by 1 to 3 points, and every token moves 14-21 MB more over the link.
- Inferred, not separated: at contexts under 2K, most of the loss is the smaller expert cache, not attention's reads, which BF16 makes larger.
- **Prefill against decode picks** (mtpA's `flash_next_mtp_phase_a divergence`, recorded in [MTP phase A](2026-10-06-flash-next-mtp-phase-a.md), three prompts, 1,533 positions per format): hq-e8-2b differs at 201 (13.1%, max margin 3.6 logits), BF16 at 74 (4.8%, max 0.875). Not re-run here: the dec2 commits leave a decode's numbers bit for bit as they were.

Raw material in `.scratch/dec2-306/`: `kv.sh`, `kv_summary.py`, `kv*.server.log`, `kv*.m{0,1,2}.txt`, `kv*.lane*.jsonl`.

## Demand copies and the lookahead (2026-10-07, lookahead41)

How long does a decode wait on expert misses, how much of that could the lookahead have covered, and does a shorter prefetch branch or a wider lookahead cover more? VRAM held fixed: no KV pool bytes given to the expert cache.

**Setup.**
- Branch `decode-lookahead-306` from `c658fa7` (16 n-gram readers). Server: `make config MODEL=flash-next`'s flags (hq-e8-2b, 262,144 tokens per lane, 3 lanes, `--prefill-chunk 8192`, `--vram-headroom-bytes 4G`, `--kv-host-pool-bytes 2G`, no speculation), plus `--metrics`.
- The plan gives the expert cache 14.34-14.37 GB (KV pool 3.71 GB). The dec2 rows above ran at 131K with 16.14 GB, so their hit rates (97.45% at one lane) and copy times are not this config's.
- `.scratch/la41/rate.sh`: one greedy lane for ~70 s (four 1,800-token requests), then three lanes with three different greedy prompts for ~100 s. Hit rate, misses, prefetches and stall are `/metrics` deltas per phase, per decoded token.
- Session 5 ran with `--kv-host-pool-bytes 0 --retained-host 0`: other processes held host RAM, and the load needs ~48.5 GB available. That config moves the traffic (84.2 prefetched / 39.4 used per token against 79.2 / 35.7) and its base (92.3 against 93.8 tok/s), so rows compare within a session only.
- Three-lane aggregates (first requests) vary by ~±1 tok/s between sessions with the same build (96.1-97.1).

**The stall, measured (`63d4eb4`, item 9).** Each demand copy reads `%globaltimer` from its first block's start to its last block's end, and its last block adds the span to `stall_nanos[phase]` and to the mirror. Exported as `ignis_expert_residency_stall_seconds_total{phase}` (ADR 0017 amendment of 2026-10-07) and drawn on the Monitor's Expert residency card. It costs nothing measurable (93.6-93.9 against 93.2-94.2 tok/s, 97.1 against 96.6).
- **One greedy lane: 2.11 ms per token**, 21% of the 10.0 ms ITL.
- **Three distinct lanes: 6.4 ms per token**, ~19 ms of a ~30 ms round (64%). The link carries ~104 MB per token there: three-lane decode is bound by the misses.
- The stall counts the demand copies only. A prefetch still copying when the next layer's step starts holds that step at its join too; that wait is not in it, and it is what the budget rows below trade against.

| session: leg (build) | 1 lane tok/s | 1 lane: hit, misses / prefetched / used per token, stall per token | 3 lanes: aggregate | 3 lanes: hit, misses / prefetched / used per token, stall per token |
|---|---|---|---|---|
| 1: base (`c658fa7`) | 93.2-94.2 (first: 83.3, cold cache) | 96.24%, 36.1 / 79.0 / 35.6, — | 96.6 | 88.38%, 106.2 / 41.0 / 27.6, — |
| 2: stall metric (`63d4eb4`) | 93.6-93.9 | 96.22%, 36.2 / 79.1 / 35.7, 2.11 ms | 97.1 | 88.36%, 106.3 / 40.8 / 27.7, 6.37 ms |
| 3: (a) rank folded into the resolve | 93.6-93.9 | 96.22%, 36.3 / 79.2 / 35.7, 2.11 ms | 97.2 | 88.37%, 106.3 / 40.8 / 27.7, 6.39 ms |
| 3: (a) + (b) parallel residency test | 93.9-94.6 | 96.22%, 36.3 / 79.2 / 35.7, 2.12 ms | 96.0 | 88.31%, 107.1 / 41.2 / 27.8, 6.43 ms |
| 4: stall metric again | 93.6-93.9 | 96.22%, 36.3 / 79.2 / 35.7, 2.12 ms | 96.1 | 88.32%, 106.7 / 41.0 / 27.6, 6.43 ms |
| 4: (a) + (b), budget × 1.5 (2.63 MB) | 84.6-86.0 | 96.55%, 33.1 / 100.0 / 44.6, 1.94 ms | **98.4** | 89.15%, 99.1 / 56.6 / 38.9, 5.98 ms |
| 5: stall metric | 92.2-92.4 | 96.16%, 36.9 / 84.2 / 39.4, 2.16 ms | 93.6 | 88.19%, 107.8 / 42.7 / 29.1, 6.54 ms |
| 5: (b) alone (`96b3c51`) | 93.1-93.7 | 96.16%, 36.9 / 84.2 / 39.4, 2.14 ms | 93.8 | 88.20%, 107.8 / 42.7 / 29.1, 6.54 ms |
| 5: (a) + (b), budget × 0.67 (1.17 MB) | **95.2-95.4** | 95.68%, 41.5 / 52.7 / 25.7, 2.37 ms | 90.7 | 87.50%, 114.3 / 26.6 / 18.5, 6.92 ms |

**What the lookahead could have covered.** A diagnostic build (`.scratch/la41/diag.patch` on `63d4eb4`, not committed) marks each next-layer key the lookahead saw and classifies every decode miss at its step: dropped by the budget, ranked in the top 16 and still missed, ranked 16-31 (the ranking widened to 32 for this), or beyond. Its flag writes cost 2.9% at one lane (91.1 tok/s); its counts are of the same traffic as the stall build's.

| per decoded token | 1 lane | 3 lanes |
|---|---|---|
| demand misses | 36.3 (25.7 MB) | 106.3 (78.6 MB) |
| in the top 16, dropped by the budget | 13.0, 9.5 MB (**37%**) | 64.3, 48.1 MB (**61%**) |
| in the top 16, prefetched or resident, then evicted before use | 0.01 | 0.01 |
| ranked 16-31 | 9.9, 6.9 MB (27%) | 18.9, 13.7 MB (17%) |
| beyond rank 31 | 13.4, 9.3 MB (36%) | 23.1, 16.8 MB (21%) |
| candidates the budget dropped | 77.7, 56.2 MB | 311, 225.6 MB |
| prefetches used, of issued | 45% | 68% |

- **The budget, not the width, binds.** With no budget, W = 16 would have covered 37% of the one-lane demand-copy time (~0.8 of 2.11 ms) and 61% at three lanes (~3.9 of 6.4 ms). W = 32 adds 27% and 17%, but only behind the same budget: the walk takes candidates in rank order and runs out of budget first, so W changes how many are dropped, not which are copied.
- **Covering them is expensive.** Only 17% (one lane) and 21% (three) of the dropped candidates were selected at their layer: covering the one-lane 9.5 MB means copying 56 MB more per token.
- A prefetch is essentially never evicted before its use (0.01 per token).

**(a) Folding `rank_lookahead` into `resolve_prefetch`: neutral, not kept.** One warp per row ranks the logits inside the resolve, with the ranking shared with `rank_lookahead` so the bits cannot differ. The split CTests passed through it. A/B: 93.6-93.9 against 93.6-93.9 tok/s, 97.2 against 97.1. The branch's ~10 µs was not on the critical path: the prefetch copy waits for the demand copy anyway.

**(b) Every candidate's residency read at once (`96b3c51`): kept.** The walk read each candidate's slot and flags with a dependent global load, one at a time. They are now read in parallel into shared flags before the walk. A prefetch that evicts a later candidate clears that candidate's flag, so it is copied again at its turn, as the policy walks them. A CPU model test and a CTest scenario pin that case; the CTest fails without the flag patch. One lane: +0.6 tok/s on (a) (session 3) and +1.1 alone (92.3 → 93.4, session 5). Three lanes: within noise.

**(c) The budget: no single value is better at both ends; the default is unchanged.**
- × 0.67 (1.17 MB a step): +2% at one lane (95.3 against (b)'s 93.4), −3.3% at three (90.7 against 93.8).
- × 1.5 (2.63 MB): −9% at one lane (85.2 against 93.8), +2.4% at three (98.4 against 96.1).
- At one lane the prefetch copy follows the demand copy on the same link and must end before the next layer's join, inside ~165 µs of compute. A smaller budget raises the stall (2.14 → 2.37 ms) and still wins: less prefetch is exposed at the join. At three lanes the misses saturate the link and 68% of prefetches are used, so prefetched bytes displace demand bytes and a larger budget pays.
- A per-step budget keyed to the step's lookahead rows (~1.17 MB at one row, ~+0.72 MB per further row) would take both gains. It was not built here:
  - it changes the residency descriptor's ABI, the policy model's `PolicyConfig` and `plan.rs`'s default;
  - and spec 03's study replay rejects its one-row value: `the_lookahead_turns_most_remaining_misses_into_hits` asserts a one-lane hit rate of at least 96%, and those two budgets replay at 95.9%. That replay scores prefetches as hidden, which served decode contradicts above.
- Wider W was not tried: by the coverage table it moves nothing while the budget binds.

**Limits.**
- One prompt family, greedy, contexts under 2K. One lane or three distinct lanes.
- The budget's optimum is bracketed by two points per end; it may lie further out (× 0.5 at one lane, × 2 at three).
- The scaled-budget legs ran on (a) + (b) builds; (a) is neutral, so they read as budget effects.
- No node-level trace this time: the exposed prefetch is inferred from tok/s against the stall, not timed.

Raw material in the `flash-next` worktree, `.scratch/la41/`: `rate.sh`, `summary.py`, per leg `*.server.log`, `*.m{0,1,2}.txt` and `*.lane*.jsonl`, `diag.patch` (the coverage counters), `lever-a.patch`, `lever-ab.patch`. The harness is also in the main checkout's `.scratch/flash-next-306-307/harness/lookahead41/`.

## Follow-ups

Status lives in https://github.com/gpillon/ignis/issues/306.

- A per-step decode prefetch budget keyed to the lookahead rows (~1.17 MB at one row, ~+0.72 MB per further row; "Demand copies and the lookahead" above): +2% at one lane and +2.4% at three, each measured on its own scaled default. Bracket it first (× 0.5 at one lane, × 2 at three); it needs the descriptor and `PolicyConfig` to carry the per-row term, and spec 03's replay acceptance (one-lane hit rate ≥ 96%) revisited, since a smaller one-lane budget wins while hitting less.
- Prefetch precision is what is left once the budget is right: 45% of one-lane prefetches are used, and 36% of one-lane demand bytes rank beyond the lookahead's top 32.
- Host-gap fixes 2, 3 and 5 (one wake per gather, every read in flight at once, one pinned staging copy): ~0.05-0.2 ms each, worth a dedicated A/B on the greedy prompt.
- A wider `router_select` (3.5 µs per layer on one CTA).
- BF16 KV at long contexts: it needs a shorter `--max-context` (32K loads, 131K does not), or a smaller expert-cache floor.
