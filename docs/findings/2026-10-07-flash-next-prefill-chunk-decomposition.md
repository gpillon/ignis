# A Flash-Next prefill traversal streams ~21 GB of experts at the link's ceiling, as long as an 8192-token chunk's compute; fewer, wider traversals are the lever

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: serving / Flash-Next prefill at the make default: one chunk's wall time, the traversals of a cold prompt, expert copies and the VRAM expert cache, the n-gram gather
- Related: https://github.com/gpillon/ignis/issues/306, [the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), [the agent turn tail finding](2026-10-07-flash-next-agent-turn-tail.md), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md), [the expert miss path](2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md), [residency on the study's routing](2026-10-05-expert-residency-replayed-on-the-study-s-routing.md), spec [flash-next/03](../specs/flash-next/03-expert-residency.md), spec [flash-next/05](../specs/flash-next/05-prompt-reuse.md)
- Superseded by: none

## Question

Flash-Next prefills at ~2.2-2.8K tokens/s (a 34K prompt in ~15 s at 4 n-gram readers), against ~8K on the 27B. [The prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md) profiled a chunk at 131K context, a 16 GB cache and a width-16 lookahead. The default has since become 262K x 3 lanes, a 14.5 GB cache, a width-10 prefill lookahead and 16 readers. At that default:

1. Where does one cold 8192-token chunk's wall time go, and how is a whole cold prompt (~9K, ~30K) shaped?
2. Which experts does a chunk touch, what does the VRAM cache save, and what would perfect copy/compute overlap give?
3. Which levers are left, and what is each worth?

The owner's constraint on every lever: each lane keeps ≥ 262K tokens of context, and the KV pool is never shrunk for speed. A wider workspace must be paid from elsewhere.

## Evidence

**Setup.**
- Build: release, branch `fn-prefill-profile` at main c69d336, plus a diagnostic kept out of the tree (`diag.patch` in the raw data below). Per prefill span, the leaf appends one JSON line with:
  - the wall time of the n-gram gather (`rows_for`) and of the device call (`step::prefill_flash_next`, which ends on the chunk's synchronize);
  - the span's n-gram counters;
  - with `IGNIS_DIAG_FN_REPORT` set, residency's report of every layer: the projections hit, first used from a prefetch, missed (demand) and prefetched.
- Server: exactly the flags `make config MODEL=flash-next METRICS=1 UI=0` prints: hq-e8-2b, `--max-context 262144`, `--decode-lanes 3`, `--prefill-chunk 8192`, `--vram-headroom-bytes 4G`, `--kv-host-pool-bytes 2G`, prompt reuse on, speculation off. Ports 8017/9417.
- The load's plan:
  - VRAM: expert cache 14.53 GB (17,147 projection slots), KV pool 3.71 GB, program 1.85 GB, residency fixed 1.58 GB;
  - host: the pinned expert pool holds 37.80 GB in 49,152 projections.
- Link: PCIe Gen 3 x16. `nvidia-smi` reports `pcie.link.gen.current` 3, `gen.gpumax` 5, `gen.hostmax` 3. The CPU is an i9-10900K, whose lanes are Gen 3.
- Client (`client.py`):
  - a warm-up, then three prompts of 8,754-9,189 tokens and two of 29,754-29,929;
  - each a distinct slice of this repo's docs and Rust sources (real text, not word lists);
  - streamed, `max_tokens` 4, thinking off; `cached_tokens` was 0 on every request.
- Two loads, the same slices in the same order:
  - **A**, report mode, no profiler: TTFT, gather and device times, residency reports, metrics.
  - **B**, `nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node`, report off. The window covers every request but the warm-up.
- Method (`analyze.py`):
  - A traversal runs from an eager `embed_rows` launch to its last kernel. Its span matches the diagnostic's device time within 2.2 ms in every one of the 24 traversals.
  - Compute is the union of every non-copy kernel.
  - Copies are residency's copy kernels: `copy_jobs_timed` on the compute stream is the demand half (what the stall metric counts); `copy_jobs` on residency's stream is the next layer's prefetch.
  - "Hidden" is copy time under a compute kernel; "exposed" is copy time with no compute running, i.e. the compute waiting on the link.
  - The profiler's tax on TTFT is −0.1 to +0.6 s per request (B against A).
- Unless a line says otherwise, ranges cover the measured requests, not the warm-up.

**One cold 8192-token chunk** (the first chunk of the five measured prompts, load B; ms):

| bucket | mean | range | share | method |
|---|---:|---:|---:|---|
| n-gram gather, card idle | 460 | 296-617 | 17.5% | diagnostic: wall time of `rows_for` before the device call |
| compute alone (no copy in flight) | 410 | 383-434 | 15.6% | nsys: compute union minus copies |
| compute with copies under it | 1,318 | 1,284-1,355 | 50.1% | nsys: compute ∩ copies |
| copies alone (compute waits) | 438 | 414-459 | 16.6% | nsys: copy union minus compute |
| idle on the card | 4 | 4-6 | 0.2% | nsys: span minus busy |
| **chunk wall** | **2,632** | | 100% | |

- The device part is 2,138-2,191 ms under nsys; load A, without it, gives 2,118-2,173 ms.
- The demand half of the copies, on the compute stream, is 123-152 ms. `ignis_expert_residency_stall_seconds_total{phase="prefill"}` counts exactly that: its per-request deltas match nsys's demand copy time (0.417 s against 416 ms).
- Compute by kernel time (sums of durations; side streams overlap the main one by ~1%):
  - dense FP8 linears 629 ms (36%); 27 of them are the shared expert on its branch;
  - routed experts 595 (34%): gate/up 304, down 288;
  - QSA attention 160 (9%);
  - routers 111 (6%): the layer's own on the compute stream, 54, and the lookahead on residency's stream, 55;
  - hyper-connections 109 (6%), GDN 70 (4%), residency's resolve/rank kernels 26, QSA indexer 19, MoE combine 15, n-gram add 6.
- The later chunks of a 30K prompt are 2,243-2,327 ms on the card. QSA attention grows to 181-186 ms and the indexer to 27-36 ms, and 473-536 ms of copies are exposed.

**Bytes over the link** (load A's report and metrics; the same text in load B gives the time):
- A full 8192-token chunk moves **21.2-21.9 GB**, whatever its position in the prompt. The per-layer reports sum exactly to `ignis_expert_bytes_moved_total{phase="prefill"}`.
- While copies are in flight they run at **11.7-12.3 GB/s** in the full chunks, and 10.0-13.2 GB/s in the shorter traversals. That is the Gen 3 link's practical ceiling: the miss path finding measured the copy engine at 12-13 GB/s.
- The link is busy for 1.72-1.87 s of a chunk's 2.14-2.33 s on the card. Span minus copy union, the link's idle time, is 0.39-0.46 s per full chunk.

**What one chunk touches** (load A, the 9 measured full chunks):
- **456-476 of 512 experts per layer** (per layer 406-508): 43.8-45.6K projections, **34.1-35.3 GB of the 37.80 GB pool** (90-93%).
- Of those bytes:
  - **The cache serves 14.0-14.4 GB.** The 14.53 GB cache is almost entirely read by every chunk.
  - **The lookahead prefetched 18.2-19.4 GB** that the next layer used.
  - **Demand copies took 1.5-1.7 GB**: 2.1-2.4K projections the lookahead missed, all into the staging ring.
- The lookahead also prefetched 0.9-1.4 GB that nothing read.
- So a chunk moves its touched bytes, minus the cache, plus the unused prefetches.

**Per layer** (load B's first chunk, `analyze.py --layers`, cut at each layer's `resolve_demand`; the prefetch a layer issues is for the next one):
- **The 12 QSA layers** compute 43-46 ms against 29-41 ms of the next layer's prefetch, and expose 1-5 ms each.
- **The 36 GDN layers** compute 31-35 ms against 29-44 ms of prefetch, and expose 5-21 ms each (~10 ms typical).
- **Layer 0** has no lookahead before it: its 41 ms demand copy is all exposed. **Layer 47** prefetches nothing.
- **The link idles between copies.**
  - Inside the chunk it idles 406 ms in 94 gaps (median 6.1 ms).
  - Each layer's prefetch starts a median 8.8 ms (7.4-47.3) after the layer's `resolve_demand`. In front of it run the demand copy, the lookahead router and the prefetch resolve; a split step's prefetch copy waits for its demand copy (`ignis_residency.h`).

**Tokens per expert** (the compression study's BF16 routing traces, as replayed in [residency on the study's routing](2026-10-05-expert-residency-replayed-on-the-study-s-routing.md); `trace_touch.py` and `tiny_moved.py`; a proxy for this text):
- Data: 8,192 tokens per domain in eight domains, each four 2048-token windows of different texts, plus mmlu's 61,440.
- Distinct experts per layer: 452-485 per 8192 tokens, which matches the engine's 456-476.
- A touched expert gets a median of 46-88 tokens.
- Small experts carry few assignments but many bytes. The cache is rebuilt as the load's warm start fills it (14.53 GB), so the share below is of the moved bytes:

  | experts with | per layer | share of the token-expert assignments | GB of the moved bytes | share of the moved bytes |
  |---|---:|---:|---:|---:|
  | ≤ 16 tokens | 82-156 | 0.7-1.1% | 4.2-7.3 | 19-35% |
  | ≤ 4 tokens | — | 0.1-0.2% | 1.7-3.5 | 8-17% |

- **Consecutive 8192-token chunks of one run (mmlu) share 97.6-98.5% of their touched (layer, expert) pairs**: 35.1-35.5 GB of 35.7-36.1 GB.
- 32,768 tokens touch 501 of 512 experts per layer: 37.0 GB, 97.9% of the pool.

**The traversals of a cold prompt** (load A, no profiler; device ms per span):
- Prompt reuse cuts every prompt at its publish point and at the opener ([the agent turn tail finding](2026-10-07-flash-next-agent-turn-tail.md)). Below the publish point, the prompt runs in 8192-token chunks; then comes a ≤ 63-token piece up to the opener, then the 4-token opener.

| prompt | TTFT | spans (tokens) | device (ms) | gather | host rest |
|---|---:|---|---|---:|---:|
| 9,189 | 5.00 s | 8192, 960, 33, 4 | 2,173, 1,552, 528, 40 | 588 | 119 |
| 9,155 | 5.04 s | 8192, 896, 63, 4 | 2,170, 1,419, 736, 43 | 569 | 104 |
| 8,754 | 4.66 s | 8192, 512, 46, 4 | 2,147, 1,334, 652, 38 | 387 | 103 |
| 29,754 | 10.25 s | 3 x 8192, 5120, 54, 4 | 2,160, 2,189, 2,232, 1,929, 611, 36 | 875 | 217 |
| 29,929 | 10.87 s | 3 x 8192, 5312, 37, 4 | 2,118, 2,193, 2,225, 1,982, 495, 38 | 1,584 | 240 |

- "Host rest" is TTFT minus every span's gather and device time: HTTP, template, tokenization, the scheduler, sampling, SSE. The pieces' gathers are 16-24 ms each in load A and 1-5 ms in load B (not explained).
- **The tail of a prompt is copy-bound.** The non-resident pool is 23.3 GB (37.80 − 14.53). What each traversal touched and moved:

  | traversal | experts per layer | moved |
  |---|---:|---:|
  | 5120-5312-token chunk | 452-465 | 21.0-21.6 GB |
  | 512-960-token chunk | 337-389 | 15.9-18.5 GB |
  | 33-63-token piece | 133-194 | 5.9-8.8 GB |
  | 4-token opener | 18 | 0.3 GB |

  Load B times them:

  | traversal | wall | of it, copies exposed |
  |---|---:|---:|
  | 5120-5312-token chunk | 1.97-2.34 s | 0.74-1.10 s |
  | 512-960-token chunk | 1.37-1.55 s | 81-84% |
  | 33-63-token piece | 0.46-0.72 s | 77-83% |

- **The 30K prompt under nsys** (29,754 tokens, 9.75 s on the card):
  - compute 6.64 s: 1.53 alone, 5.12 under copies;
  - exposed copies 3.07 s: 1.42 in the three full chunks, 1.10 in the 5120-token chunk, 0.53 in the 54-token piece;
  - card idle 0.04 s;
  - the link busy 8.19 s for 93.5 GB, 4.0x the non-resident pool.

**A wider chunk** (zero code; one load each, the same client and slices, report mode):
- **16384 does not start at 262K x 3.** The plan refuses it: the expert cache would get 12.79 GB, under its 12 GiB floor.
- **12288 does.** The program grows from 1.85 to 2.70 GB, paid by the expert cache: 14.53 → 13.68 GB (16,103 slots). The KV pool stays 3.71 GB.
- TTFT against load A:

  | prompt | 8192 | 12288 | change | spans at 12288 |
  |---|---:|---:|---:|---|
  | 9,189 | 5.00 s | 3.49 s | −30% | 9152, 33, 4 |
  | 9,155 | 5.04 s | 3.69 s | −27% | 9088, 63, 4 |
  | 8,754 | 4.66 s | 3.40 s | −27% | 8704, 46, 4 |
  | 29,754 | 10.25 s | 9.11 s | −11% | 2 x 12288, 5120, 54, 4 |
  | 29,929 | 10.87 s | 9.81 s | −10% | 2 x 12288, 5312, 37, 4 |

- How the traversals change:
  - A 12288-token chunk: 2.70-2.80 s on the card, 22.6-23.1 GB moved, 0.22-0.23 ms/token. A full 8192-token chunk: 0.26-0.27 ms/token.
  - A ~9.1K prompt's first span is now one traversal: 2.26-2.30 s and 22.3-22.4 GB. At 8192 it was two: 2.15-2.17 s plus 1.33-1.55 s, and 37-40 GB.
  - The 30K prompts keep a copy-bound 5120-5312-token chunk: 1.98-2.04 s.
- **What 12288 costs one-lane decode** (`decode.py`: a short prompt, 800 tokens greedy, one warm-up and three reps per leg). The legs ran A-B-A-B. Three of them had a 1 GiB KV-RAM arena, because the host plan refused 2 GiB while other agents compiled; the arena is not on this path.

  | | 8192 | 12288 |
  |---|---:|---:|
  | first pair, other agents compiling | 86.9-88.3 tok/s | 93.3-95.2 tok/s |
  | second pair, quiet | 97.8-98.8 tok/s | 95.8-96.1 tok/s |
  | expert-cache hit rate | 97.0% | 96.5% |
  | experts moved per decode token | 66.5 MB | 73.7 MB |

  - Measured, and the same in every leg: the hit rate falls 0.5 points, and the bytes per decode token rise 11%.
  - The tok/s effect is not resolved. The two 8192 legs differ by 11%, and the four legs drift upward in run order; the quiet pair's −2.5% sits inside that spread.
  - The first pair also shows that one-lane decode on this box moves with other agents' CPU builds.

Raw data: `.scratch/fn-prefill-2026-10-07/` in the main checkout (untracked):
- `report/`, `nsys/`, `c12288/`, `c16384/`, `dec*/`: `prefill.jsonl`, `client.jsonl`, `decode.jsonl`, `server.log`, `residency.json`, metrics;
- `nsys/analyze*.txt`, `nsys/capture.traversals.json`, `report/touch.txt`, `c12288/touch.txt`, `summary.txt`, `trace_touch.txt`, `tiny_moved.txt`;
- the method: `diag.patch`, `run.sh`, `client.py`, `decode.py`, `analyze.py`, `touch.py`, `trace_touch.py`, `tiny_moved.py`.

The `.nsys-rep`/`.sqlite` capture is not copied there, and never committed: it carries the process environment. It stays in the `flash-next` worktree's `.scratch` while that worktree exists.

## Finding

Observed:

1. **A full chunk's compute and its expert stream are equal.**
   - An 8192-token chunk computes for 1.72-1.79 s and moves 21.2-21.9 GB at the link's ~12.3 GB/s, 1.72-1.87 s of copies.
   - 75% of the copy time is hidden under compute. The chunk still waits 0.41-0.54 s on copies alone, 19-23% of its time on the card.
   - The link idles for as long: 0.39-0.46 s per chunk, mostly in a ~6-9 ms gap at each layer before its prefetch starts.
   - Which exposure holds depends on the load:
     - 0.13-0.15 s at 131K context, a 16 GB cache and a width-16 lookahead ([the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md));
     - 0.52-0.56 s at this default with width 16 ([the agent turn tail finding](2026-10-07-flash-next-agent-turn-tail.md));
     - 0.41-0.54 s at this default with width 10, here.
2. **The stream is per traversal, not per token.**
   - A 512-960-token traversal touches 66-76% of the experts per layer and moves 15.9-18.5 GB. An 8192-token one touches 89-93% and moves 21.2-21.9 GB: 16x the tokens for 1.2-1.4x the bytes.
   - A 33-63-token piece still moves 6-9 GB.
   - A 9.2K prompt is four traversals: 2.17 + 1.55 + 0.53 + 0.04 s on the card. A 30K prompt is six, and moves 93 GB.
3. **The cache saves ~40% of each chunk's expert bytes**: 14.0-14.4 of 34-35 GB touched. Every chunk reads 96-99% of the cache.
4. **The compute splits** into dense FP8 linears 36%, routed experts 34%, QSA 9%, routers 6%, hyper-connections 6% and GDN 4%. Synchronization and host gaps inside a chunk are 4-6 ms.
5. **The n-gram gather is 8-15% of TTFT**: 0.18-0.62 s per 8192 tokens, serialized before each chunk with the card idle. It grows with the rows the hot cache misses: 12-55K file rows (11-51K reads, 48-216 MB) per chunk, depending on the text.
6. **The link is Gen 3 x16** because the host's lanes are Gen 3 (i9-10900K). The card supports Gen 5.
7. **A 12288-token chunk cuts TTFT 27-30% at ~9K and 10-11% at ~30K, at zero code.**
   - It folds a ~9K prompt's second chunk into the first, and a 30K prompt runs three chunks instead of four.
   - It takes 0.85 GB from the expert cache; the decode cost is in the 12288 evidence above.

Inferred:

- **A prompt's time is set by its traversal count, not its length.** That follows from items 1 and 2: every traversal pays most of a ~21 GB stream, and a full chunk's compute only matches it.
- **Speeding up one side alone moves a full chunk only to the other side's floor.** Compute and copies are equal (item 1), so cutting either leaves the chunk at the other.
- **For prefill, the cache's size matters, not its content.** Each chunk touches ~90% of the pool, so almost any content would be read; the warm start's hottest-first content is read 96-99%.
- **The perfect-overlap bound.** A full chunk cannot finish in less than max(compute, copies), 1.72-1.87 s, against 2.14-2.33 s measured: 0.39-0.46 s per chunk at most.
  - Over every traversal, the sum of max(compute, copies) leaves 0.48-0.52 s for a ~9K prompt and 1.54-1.56 s for a 30K one.
  - That gap is the link's idle time. Copy bytes match compute, so every millisecond the link waits for a layer's demand copy, lookahead router and resolve shows up as exposed copy.
  - It is not structural: a prefetch that starts earlier, or one more staging-ring half so the link runs a layer further ahead, would keep the link busy.
  - How much of the bound that recovers is not known. Another ring half (~0.8 GB) costs the cache and moves ~0.8 GB more per chunk, and continuous copies may slow the compute under them.
- **Faster kernels gain little at Gen 3 on 8192-token chunks.** A full chunk's copies already equal its compute, so cutting compute leaves the chunk at the copy floor, and copy-bound traversals do not move at all.
- **A 12288-token chunk is compute-bound.** It spends 2.70-2.80 s on the card for ~1.85 s of copies (22.6-23.1 GB at ~12.3 GB/s). It was not profiled.
- **Small experts are where the bytes go, not the work.** Experts with ≤ 16 tokens in a chunk hold ~1% of its assignments but 19-35% of its moved bytes (proxy routing).
  - Skipping them changes the model's output.
  - Merging their work inside a chunk gains nothing: the prefill op already runs an expert of ≤ 16 rows narrow, so a one-token expert costs its weight read and little else (`moe_prefill.cu`). Their cost is the bytes.
  - Only fewer traversals merge them: at 32K, 34 per layer have ≤ 16 tokens.

## Implications

The ranked levers: software first, by gain, then confidence, then cost; the hardware lever last. Gains are against the 8192 default at Gen 3, for the 30K prompt (10.2-10.9 s) and the ~9K prompt (4.7-5.0 s):

| # | lever | estimated gain | basis | cost |
|---:|---|---|---|---|
| 1 | **`--prefill-chunk 12288`** at 262K x 3 lanes. 16384 is refused; the plan grows ~0.21 GB per 1,024 tokens, so up to ~15360 should fit (inferred, not tried). | **9K −1.3-1.5 s (−27-30%), 30K −1.1 s (−10-11%)** | **measured** | Zero code. The expert cache gives 0.85 GB; the decode cost is in the evidence above (measured hit rate and bytes, tok/s unresolved). The decode lanes' gap during a prefill grows to one 12288-token chunk, ~2.7-2.8 s plus its gather (inferred). |
| 2 | **Fewer streams per prompt: layer-major chunk groups.** Run every chunk of a group through layer *l* before layer *l*+1, so each layer's experts stream once per group, not once per chunk. This is expert-major ordering across chunks; inside a chunk, the prefill op already groups the assignments by expert. | 30K: the four chunks move ~24 GB instead of 85.8 and take ~6.6 s on the card instead of 8.5 (their compute, 6.5 s, hides one stream), **TTFT −1.9 s (−19%)**. 9K: its 960-token chunk drops from 1.55 s to its ~0.3 s of compute, **−1.2 s (−25%)** | inferred from measured compute, bytes and the 98% chunk-to-chunk overlap | Large: the forward loop turns layers-outer. The KV, indexer and GDN state are already per layer and causal, so the order is legal. The group's residual stream (20 KB/token, 0.5 GB for 4 x 8192) is paid from the expert cache. The group is one block for the decode lanes unless rounds interleave. |
| 3 | **Keep the link busy**: start each layer's prefetch earlier, or one ring half deeper | Up to the link's idle time: **0.39-0.46 s per full chunk; 0.48-0.52 s per ~9K prompt, 1.54-1.56 s per 30K** (the bound); the realistic share is unknown | inferred from the measured link idle and per-layer prefetch start | Residency and the forward's step order; a ring half (~0.8 GB) from the cache if deeper |
| 4 | **Fold the publish-point piece** into the traversal before it: a pages-only chained prefix, ADR 0029 amendment, [the agent turn tail finding](2026-10-07-flash-next-agent-turn-tail.md) follow-up 1 | **−0.46-0.74 s on every prompt**, cold or reused: −5-6% at 30K, −10-15% at 9K | the piece's cost is measured; the saving assumes the fold adds its ~10 ms of compute and no stream (inferred) | A spec 05 / ADR 0029 change. The opener piece (37-43 ms) is not worth folding. |
| 5 | **Pipeline the n-gram gather**: gather chunk *k*+1's rows while chunk *k* is on the card, and take the first chunk's reads deeper than 16 synchronous threads | 30K: −0.5-1.1 s (the second and later chunks' gathers). One-chunk prompt: only what the deeper queue gives the first gather (≤ ~0.2-0.3 s) | measured gather, inferred overlap | The scheduler knows the next span ([the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md) follow-up) |
| 6 | **Balanced chunk split**: cut a span into equal chunks instead of full chunks plus a remainder | With 12288-token chunks, ≤ ~0.5 s at 30K. The 29,696-token span runs 12288 + 12288 + 5120 today; three ~9,900-token chunks would take 3 x ~2.14 s instead. Σ max(compute, copies) goes from ~7.0 s (2.6 + 2.6 + 1.78) to ~6.4 s. Compute is scaled from the 8192 chunk's 0.21 ms/token; copies are ~1.8-1.85 s per chunk. ~0 at 8192: the span stays four traversals of ~86 GB and 6.5 s of compute, each near the copy floor. | inferred | Scheduler only. Chunk ends may need page alignment. |
| 7 | Prefetch accuracy (unused prefetches, demand misses) | ≤ ~0.1 s per chunk: 0.9-1.4 GB unused + 1.5-1.7 GB demand | measured bytes, inferred time | Policy tuning |
| 8 | Kernel efficiency (dense FP8 36%, routed experts 34%) | ~0 at Gen 3 on 8192-token chunks; 1:1 on the chunks that items 1-3 make compute-bound | inferred | Kernel work |
| H | **A Gen 4/5 host** (hardware) | Gen 4 halves every copy: a full chunk → ~1.75-1.8 s, the copy-bound tail ÷2. 30K **−2.3-2.7 s**, 9K **−1.3 s** | inferred from the bytes and the link rate | A platform change (CPU and board); the card already supports Gen 5 |

- **Not levers.**
  - Skipping small experts changes the output, and merging their work inside a chunk gains nothing (see Inferred).
  - Copy/compute overlap as a whole is already 75% hidden; item 3 is what is left.
  - Narrower chunks multiply the streams: [the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md) measured 2048 at 1.86x the TTFT.
- **The owner's constraint holds for every item.** None shrinks the KV pool. Items 1, 2 and 3 take their VRAM from the expert cache, which costs decode (item 1's cost is measured above).
- **The items compose.**
  - Items 2, 4 and 6 remove or merge traversals, and stack on item 1.
  - At 12288 a full chunk is compute-bound (inferred), so kernel work (item 8) starts to pay there.

## Limits and unknowns

- **One load per mode** (load A, load B, the 12288 cell), five prompts each:
  - real text from this repo (docs and Rust), no agent trace;
  - a warm cache after a short warm-up;
  - the decode lanes idle, so the decode share never held a chunk;
  - prompts of 29,754-29,929 tokens where a 32K one was asked for: the client's token estimate came out low. A 32K prompt would run a fourth full chunk instead of the 5120-token one.
- **Not measured:** expert-major ordering, and merging small experts' work.
  - Item 2 is expert-major ordering's across-chunk form, and is inferred.
  - That merging inside a chunk gains nothing is argued from `moe_prefill.cu`'s narrow path, not timed.
- **The copies' cost to the compute they overlap is not separated.** Copy kernels hidden under compute share SMs and the memory bus with it. The compute here was measured with them running.
- **Tokens per expert come from the study's routing**, not from this engine's prompts. The engine's report gives the touched sets, not per-token counts. The proxy's 8192 tokens are four unrelated texts; the engine's touched counts agree with it (456-476 against 452-485 per layer).
- **What is measured and what is inferred, per item:**
  - item 1's TTFT is measured on five prompts; its decode cost at one lane only, not at 2-3 lanes, and its tok/s effect is unresolved;
  - items 2, 3, 5, 6 and H are inferred;
  - item 4's saving is inferred (the piece's cost is measured), and item 7's time is inferred (its bytes are measured);
  - item 2's group activations, its interaction with the decode share, and its gap for the decode lanes are not designed.
- **The gather varies with the text:** 12-55K file rows per 8192 tokens. Why the small spans' gathers took 16-24 ms in load A and 1-5 ms in load B is not known.
- **The owner's figures** (34K in ~15 s, an 8192 chunk in ~2.9 s) were word-list prompts at 4 n-gram readers. This finding's real text at 16 readers gives 10.2-10.9 s for ~30K; a 34K prompt adds one more traversal.

## Follow-ups

Ranked, not filed:
1. An owner decision on `--prefill-chunk 12288` as Flash-Next's make default (item 1). Before deciding, measure decode at 2-3 lanes and the lanes' gap with the decode share at 25.
2. Layer-major chunk groups (item 2): a design note first, covering the group's activations, the decode lanes' gap, and residency's ring per layer.
3. Keep the link busy (item 3): measure how much of the per-layer prefetch gap an earlier start recovers.
4. Pages-only chained prefix (item 4): already follow-up 1 of [the agent turn tail finding](2026-10-07-flash-next-agent-turn-tail.md); it also cuts every cold prompt.
5. Gather pipelining and a deeper read queue (item 5).
6. A balanced chunk split in the scheduler (item 6), once item 1 is decided.
7. For the owner: the link is Gen 3 because of the host's CPU (item H).
