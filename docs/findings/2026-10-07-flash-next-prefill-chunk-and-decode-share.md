# A Flash-Next prefill chunk hides its expert stream under compute; the host n-gram gather and one decode round per chunk own the rest

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: serving / Flash-Next prefill chunk, link/compute overlap, the n-gram gather, prefill/decode interleaving (decode share)
- Related: https://github.com/gpillon/ignis/issues/306 (items 3, 4), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md), [Flash-Next decode round](2026-10-06-flash-next-decode-round.md), [ADR 0018](../adr/0018-chunk-level-prefill-decode-interleaving.md), spec [flash-next/03](../specs/flash-next/03-expert-residency.md)
- Superseded by: none

## Question

The [2026-10-06 finding](2026-10-06-flash-next-on-the-5090.md) measured an 8K TTFT of 6.9-8.5 s and 32.7K in 25.7 s. It inferred ~17 GB of experts streamed per chunk with "compute ~3x the study's bound, or no link/compute overlap". It also measured the other lanes at 4.5-8 tok/s during a long prefill. Two questions follow:
- Where does one 8192-token chunk's time go: expert copies, compute, or idle? Do the copies overlap compute?
- How does the scheduler interleave chunks with decode rounds, and what bounds the other lanes' stall?

## Evidence

**Setup.**
- Build: release, branch `mtp-closeout-307` at 52a9990 (the decode share, off for the capture). Other agents' uncommitted MTP edits were also in the tree; MTP is off by default and not on the prefill path.
- Server: `make config MODEL=flash-next`'s flags (hq-e8-2b, `--prefill-chunk 8192`, `--vram-headroom-bytes 4G`), exclusive card, GPU lock held.
- Capture: `nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node`, one 25 s window, with `--prompt-reuse off --retained-host 0 --kv-host-pool-bytes 0`.
- Client: back-to-back distinct prompts with `max_tokens` 4. The prompt was 8,728 tokens served (the client's `/v1/tokenize` estimate was 7% low), so each request ran one 8192-token chunk and then a 536-token tail chunk.
- Method: `.scratch/prefill306/analyze_prefill.py`. A chunk is a run of eager device work with no hole over 50 ms. Busy time is the union of every activity on every stream. The residency copies are the `copy_jobs` kernels:
  - on the compute stream they are the demand half;
  - on residency's prefetch stream they are the next layer's lookahead staging.

  "Hidden" means copy time during which another kernel ran.

**One 8192-token chunk** (three samples, ms):

| | |
|---|---:|
| card span | 1,867 / 1,880 / 1,955 |
| idle inside the span | 4-5 |
| compute (any non-copy kernel) | 1,736-1,803 |
| expert copies | 1,224-1,284 (prefetch stream 1,104-1,167, demand 117-120) |
| of them hidden under compute | 1,091-1,137 |
| of them exposed | 126-147 |
| host time before the chunk, card idle | 1,244-1,349 |

- Compute: 0.21 ms per token.
- Kernel time inside one chunk (ms):
  - dense FP8 linears 635;
  - routed experts 594 (gate/up 304, down 290);
  - QSA attention 158 (12 layers);
  - routers 111 (57 of them the lookahead);
  - hyper-connections ~100;
  - GDN ~60.
- The host time before the chunk is the n-gram gather. Per request, the window's counters show ~94K file rows, ~51K reads and 217 MB, on the table's 4 reader threads. The 536-token tail waits ~130 ms for its own gather.

**The 536-token tail chunk** (two samples):
- span 988-1,049 ms;
- compute 208-216 ms;
- copies 897-925 ms, of which 723-744 ms exposed.

**Bytes over the link.** `ignis_expert_bytes_moved_total{phase="prefill"}` grew by 151.8 GB in the ~29 s metrics window, which held ~6.7 requests. That is ~22.6 GB per request.
- Split by copy time: ~13 GB for the 8192-token chunk and ~9.6 GB for the tail (inferred).
- The copies therefore ran at ~10.6 GB/s.

**TTFT now** (warm cache, no other lane decoding):
- 8,728 tokens: 4.21-4.43 s under nsys (23 requests).
- 34K tokens (33,966 served): 15.2-15.7 s. The 2026-10-06 finding measured 25.7 s at 32.7K.
- One-chunk prompt (~8,150 tokens), median of 13 requests:
  - 3.72 s at 4 n-gram reader threads;
  - **2.87 s at 16**, with the same reads per request (46.5K reads, 196 MB). Same source, one load each (`.scratch/prefill306/ttft-ng{4,16}.*`).
- Decode at one lane on the same two binaries (`fnperf.py decode`, ~205-token prompt, 3 x 1000 tokens each):
  - 87.6-88.4 tok/s at 4 reader threads;
  - **102.4-105.5 tok/s at 16**, with the same expert traffic (51-58 MB per token, hit rate 97.7-98.0%);
  - the short prompt's TTFT falls from 1.60-1.68 s to 1.27-1.33 s.

**How the scheduler interleaves.**
- Flash-Next runs the 27B's `ConcreteScheduler`. Each `advance()` runs at most one prefill chunk, then one decode round for every running lane (ADR 0018, K = 1).
- The chunk's whole `prefill_step` blocks the round: the leaf's n-gram gather, then the chunk on the card.
- One 8192-token chunk is ~3.2 s of wall time against the 27B's ~110 ms. The lanes therefore get one token per ~3.2 s while a long prompt prefills.

**The decode share** (52a9990, `--decode-share`):
- After a chunk that took `t`, the scheduler holds the next one until the decoding lanes have had `t * s / (1 - s)` of wall time.
- It holds nothing when no lane decodes.
- The defaults were 50 on Flash-Next and 0 on the 27B (the 27B went to 50 after #92). The default then became 25 on both models, by owner decision 2026-10-07.

**Stall A/B**:
- Setup: one load per cell, `make config`'s flags plus the cell's `--decode-share` / `--prefill-chunk`. Two lanes stream long generations of short prompts. Once both decode, a third lane sends a distinct 34K prompt (33,966 tokens served) with `max_tokens` 8. Two reps per cell, plus the prompt alone first.
- "During" is from the send to the prompt's first token. Script: `.scratch/prefill306/stall.py`.
- Cells s25 and s0-c2048 ran with a 1 GiB KV-RAM arena instead of 2 GiB, because the host plan was refused by ~0.3 GB of free RAM. The arena is not on this path.

| decode share / chunk | TTFT alone | TTFT, 2 lanes decoding | lanes' tok/s during (before) | tokens per lane during | longest gap |
|---|---:|---:|---:|---:|---:|
| 0 / 8192 (before) | 15.4 s | 15.2-15.3 s | **1.0-1.1** (62-69) | 15-17 | 3.2-3.4 s |
| 25 / 8192 | 15.2 s | 19.4-20.2 s | 15.6-18.3 (65-67) | 303-370 | 3.2 s |
| **50 / 8192** | 15.7 s | **30.1-31.6 s** | **30.8-32.6** (60-69) | 974-982 | 3.2 s |
| 0 / 2048 | 28.7 s | 28.3-28.6 s | 1.0-1.1 (69-70) | 29-30 | 1.7 s |

- Every cell had a 2,053,174,464-byte KV pool. The decode share takes no VRAM.
- The 2048-token chunk's smaller workspace gave the expert cache 17.22 GB, against 16.14 GB at 8192.

## Finding

Observed:

1. **An 8192-token chunk is compute-bound, and its expert stream already overlaps compute.**
   - 89% of the copy time runs under compute. Only 0.13-0.15 s of the 1.9 s chunk is copy alone, and the card is never idle inside the chunk.
   - Compute is 0.21 ms per token, within the study's bound (0.77-1.3 s per 4K).
   - The 2026-10-06 inference ("compute ~3x the bound, or no overlap") no longer holds.
2. **About 40% of a chunk's wall time was the host n-gram gather, with the card idle.**
   - At 4 reader threads the gather took ~1.2-1.3 s per 8192 tokens.
   - 16 threads take a one-chunk TTFT from 3.72 to 2.87 s (−23%). Committed in adec4d4.
   - The same change speeds up decode: one lane goes from 88 to 102-105 tok/s. A round's ~12 row reads now run one deep instead of three.
3. **A short chunk is copy-bound.**
   - A 536-token chunk costs ~1 s, of which ~0.9 s is copies with little compute to hide under. Per-chunk transfer is nearly constant (spec 03).
4. **With one round per chunk, the lanes fall to ~1 tok/s during a long prefill.** The longest gap is one chunk's wall time: 3.2 s at 8192 tokens.
5. **The decode share sets the lanes' rate during a prefill, and the TTFT pays exactly the hold.**
   - At 25%: 16-18 tok/s, TTFT ×1.3 (formula ×1.33).
   - At 50%: 31-33 tok/s (about half of 60-69), TTFT ×2.0 (formula ×2).
   - The prompt pays the hold only while lanes decode. Alone, its TTFT is unchanged (15.2-15.7 s in every 8192 cell).
6. **The chunk width moves only the gap, and at a high price.**
   - A 2048-token chunk halves the gap (1.7 s) but leaves the lanes at ~1 tok/s.
   - It costs 1.86× TTFT with nobody decoding (28.7 s), because every chunk pays a near-constant expert stream.

Inferred:

- **Why the 2026-10-06 8K TTFT was 6.9-8.5 s.** The drop to ~4.3 s is observed; its cause is not bisected. The candidates are the fixes in between: the hyper-connection mix, and the n-gram reads that the artifact's memory map serialized ([decode round finding](2026-10-06-flash-next-decode-round.md)).
- **Item 5's reused turn.** Its 1.1K-token tail (2.45 s) plausibly pays the same copy-bound cost as the 536-token tail here. Not measured on that turn.
- **Choosing the default.** (Written when the default was 50; it became 25 on both models by owner decision 2026-10-07: a reader reads ~5-8 tok/s, and 25 keeps the lanes at 15.6-18.3 tok/s for a TTFT of x1.33 instead of x2.) A long prefill and the decoding lanes share one card, so any share moves time from one to the other. 50% halves the lanes' rate rather than stopping them, and doubles the TTFT at worst, and only while lanes decode. With the agents' short tool-call generations, the decoding agents finish sooner. The prefilling agent finishes no later than the sum of the two jobs' work.
- **TTFT at 16 readers.** Each 8192 chunk should lose ~0.9 s of gather, so 34K should land near ~11.5 s. Not measured.

## Implications

- **Item 3 (TTFT).** The levers left are:
  - the gather, ~0.3 s per 8192 chunk at 16 threads;
  - compute: dense FP8 linears 35%, routed experts 34%;
  - copy-bound short chunks.

  Link/compute overlap is not a lever: ≤ 0.15 s per full chunk. (At the later 262K x 3-lane default this no longer holds: see [the prefill chunk decomposition](2026-10-07-flash-next-prefill-chunk-decomposition.md).)
- **Item 4 (stall).** The decode share bounds the lanes' rate during a prefill. Bounding the *gap* below one chunk's wall time needs decode rounds inside a chunk, which ADR 0018 deferred:
  - option 1: a leaf sub-step that runs a chunk's layers in slices and returns between them;
  - option 2: true overlap.

  With the 3 s gap, an interactive stream stutters once per chunk. An agent's tool-call generation keeps going at half rate.

## Limits and unknowns

- **One capture**, one prompt family (synthetic word lists), warm cache. The share and chunk cells have two reps each.
- **Hidden copies are not free.** Copy kernels hidden under compute still share SMs and PCIe with it. How much the copies slow the compute they overlap is not separated: that needs a fully cached chunk, which this card cannot hold.
- **Per-chunk bytes** come from a window delta split by copy time, not from a per-chunk counter.
- **Not measured at 16 reader threads:** the 34K TTFT, and decode at more than one lane.
- **The hold follows every chunk, a request's last one included.** That request is a decoding lane from then on, so a newcomer's first chunk queued behind a long prompt can wait up to `t * s / (1 - s)` while only that one lane decodes. This is the case an agent swarm hits at every turn boundary; it was not measured.
- **The gap at share > 0** is one chunk's wall time in every cell. The 25% and 50% candidate defaults were compared on one scenario: two lanes decoding throughout one long prompt.

## Follow-ups

- Rounds inside a chunk, to bound the gap: a leaf sub-step over layer slices. Prefill activations live in `fn.residual/x/y`, which the decode graph also uses, so the decode round would need buffers of its own. Residency's ring halves are tagged per layer.
- Pipeline the n-gram gather: gather chunk k+1's rows while chunk k is on the card (the scheduler knows the next span).
- Short chunks: stage only what a small chunk's own router selects (the lookahead stages each row's top 16, more than the top 10 used), or fold a short tail into the previous chunk when the width allows. (Done 2026-10-07 for the width: a chunk's lookahead is now the router's top 10, and at the 262K x 3-lane default an 8192-token chunk had 0.52-0.56 s of copies exposed, not 0.13-0.15: [the agent turn tail finding](2026-10-07-flash-next-agent-turn-tail.md).)
- Re-measure the 34K TTFT, and decode at 2-3 lanes, at 16 reader threads.
