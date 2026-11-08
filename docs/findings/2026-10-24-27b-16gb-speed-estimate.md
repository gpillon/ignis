# The 27B's 16 GB artifact: a speed estimate, not measured on a 16 GB card

- Kind: experiment
- Status: current
- Observed: 2026-10-24
- Last verified: 2026-10-24
- Scope: serving / the 27B's trellis body decode round on the RTX 5090 (PCIe Gen 3), uncapped and under the debug SM cap
  (84 / 70 / 36), its bytes and achieved bandwidth, and an ESTIMATE for the RTX 5060 Ti 16 GB, RTX 5070 Ti and RTX 5080
- Topic: artifacts-and-gates
- Related: [GitHub #340](https://github.com/gpillon/ignis/issues/340),
  [spec 27b-16gb/05](../specs/27b-16gb/05-gates-and-simulation.md) (Part B, Acceptance 6),
  [the dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md),
  [the 16 GB artifact: first light](2026-10-23-27b-16gb-first-light.md),
  [capacity on the 5090](2026-10-24-27b-16gb-capacity-on-the-5090.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md)
- Superseded by: none
- Summary: The 27B's trellis body decodes 416 tokens/s at one lane and 8K on the 5090, and a rule applied to that, not a measurement on any 16 GB card, estimates 243-307 on a 5080, 204-270 on a 5070 Ti and 103-153 on a 5060 Ti.

**This is an estimate. Nothing here was measured on a 16 GB card; the 5090 measurements are real, the per-card numbers are
a rule applied to them.**

## Question

How fast does the 16 GB artifact decode on the 5090 at one and two lanes and 8K / 64K / 253K of context, how much of that
is the trellis body (the part a smaller card slows), and what does that predict for the 5060 Ti 16 GB, 5070 Ti and 5080?

## Evidence

Build `340-gates` (release CUDA server). Every run is the B0 load: budget 17,179,869,184, `rtx50-16gb`, 262,144 context, 2
lanes, chunk 1,024, hq-e8-2b, DFlash2 7 / full head, loop guard at its default; one server start per SM cap
(`IGNIS_TRELLIS_SM_CAP` none, 84, 70, 36; cap 999 is refused, `cap999-server.log`). Files: `.scratch/340/speed/`
(untracked): `<cap>-{server.log,cells.jsonl,cells.txt,analysis.txt,cells.json,smi.csv,metrics.txt}`, `estimate.py` /
`estimate.txt`, `clocks.py` / `clocks.txt`, `cells.py`, `analyze.py`, `run-cap.sh`.

**Protocol** (`cells.py`; prompts cached in `prompts.json`, byte-identical across caps): the system message is a word soup
landed exactly through `/v1/tokenize`, the user message one coding ask; greedy, thinking off, 512 tokens, streamed. Per
length: A cold, B cold (the 1-lane cells), A and B at once (the 2-lane cell: both claim their prompt checkpoints, so both
decode from the start), A again (reuse check). Round time and tokens per round come from the request log (`analyze.py`). The
2-lane round time is the later-starting lane's decode span until the other lane is done, over its rounds in that span
(exact when it finishes first, else less its width-1 tail rounds); a lead correction on the first lane was rejected, its
first rounds interleave with the other lane's tail prefill. Every cell: `finish_reason` length, no error, no loop-guard,
eviction or restore event; both TTFTs before either done in every pair.

**256K.** The pool is 7,956 pages; 2 x (253,440 + 512) = 507,904 tokens = 7,936 pages fits (the most with 512 output is
254,080 a lane). So the 2-lane "256K" cell is **253,440 tokens a lane**, and the 1-lane one too; a 261,632-token 1-lane cell
(uncapped) reads the same round time (27.41 and 27.41 ms).

**What the cap reaches.** `IGNIS_TRELLIS_SM_CAP` admits only the trellis ops' CTAs (the body and the head). Attention, the
GDN recurrence, the drafter (NVFP4) and the sampler still run on 170 SMs. The capped columns are therefore: trellis ops
capped, the rest on all 170 SMs.

### Measured on the 5090: round ms / tokens per round per lane / tokens per second across the lanes

| cells | none | cap 84 | cap 70 | cap 36 | bytes per round | achieved, none |
|---|---|---|---|---|---:|---:|
| 1 x 8K | 14.47 / 6.02 / 416 | 19.60 / 6.06 / 309 | 22.30 / 5.99 / 269 | 39.34 / 6.02 / 153 | 12.15 GB | 840 GB/s (47%) |
| 1 x 64K | 17.44 / 6.17 / 354 | 23.12 / 5.82 / 252 | 25.24 / 5.79 / 229 | 42.39 / 6.02 / 142 | 12.68 GB | 727 GB/s (41%) |
| 1 x 253K | 27.41 / 5.92 / 216 | 32.04 / 5.85 / 183 | 34.36 / 5.92 / 172 | 51.40 / 6.06 / 118 | 14.41 GB | 526 GB/s (29%) |
| 2 x 8K | 17.62 / 6.02 / 684 | 23.36 / 6.02 / 516 | 26.29 / 6.06 / 461 | 47.10 / 5.99 / 254 | 12.76 GB | 724 GB/s (40%) |
| 2 x 64K | 24.11 / 6.02 / 500 | 30.26 / 5.79 / 382 | 32.23 / 5.99 / 372 | 53.86 / 6.02 / 224 | 13.82 GB | 573 GB/s (32%) |
| 2 x 253K | 43.00 / 5.92 / 275 | 48.11 / 6.10 / 253 | 50.55 / 5.95 / 236 | 71.33 / 6.06 / 170 | 17.28 GB | 402 GB/s (22%) |

The percentages are of the 5090's 1,792 GB/s. **Bytes a round reads** (`estimate.py`): the body 9.19 GB; the head 0.64 GB
twice (verify, and the draft proposals: the head's tile is decoded once a call for up to 8 columns, so 16 columns cost
compute, not bytes); the drafter 1.08 GB; plus per lane KV 9,216 B x context, GDN state 151 MB x 3 (the verify read, the fold
read and write), the drafter window 40 MiB and the hq window 36 MB.

Decode SM clock median (`clocks.txt`): 2,734 / 2,842 / 2,850 / 2,872 MHz for none / 84 / 70 / 36; power 537 / 513 / 490 / 349
W; at most 74 C. The capped decodes ran 4-5% faster than the uncapped, so the rest sped up too and the trellis share below
is slightly understated, inside its printed spread.

DFlash2's 2,048-token window: tokens per round held 5.8-6.2 from 8K to 253K on these cells, not the long-context drop
PIANO-VRAM section 10 feared. The soup carries nothing and the ask sits inside the window, so agent traffic at long context
may differ. Tokens per round varies 5.7-6.3 with the text: the greedy text differs across caps (the cap changes the decode
split) and between cold and reused prefill (the tail is its own chunk), so rounds are compared, not text. Reuse bias on
round time (A reused against A cold): +1.7 / +0.7 / +0.7% (none), +0.2 / +4.0 / -2.5% (84), +0.7 / -0.5 / -0.6% (70), +0.1
/ -0.1 / -0.2% (36): none systematic, and A cold against B cold differ by as much.

**Decomposition.** R(cap) = rest + T x r(cap), with r the op bench's body ratios (the 2026-10-23 finding: 1.53 / 1.81 /
3.68). The three caps give the same trellis time within the noise: **1 lane 9.40 ms (9 estimates, 8.59-10.72), 2 lanes 10.53
ms (9.33-11.62)**. The trellis's 10.41 GB at 1,108 / 988 GB/s is 62% / 55% of the roofline (the op bench's body: 65%). The
rest is 5.1 ms at 8K (drafter, GDN, sampler, launches), and +3.0 / +12.9 ms at 64K / 253K on one lane: the extra 2.26 GB of
KV at about 175 GB/s. Attention is far below DRAM speed.

### Estimated for the 16 GB cards: tokens per second across the lanes (the range is optimistic to conservative)

| card (GB/s, SMs) | 1 x 8K | 1 x 64K | 1 x 253K | 2 x 8K | 2 x 64K | 2 x 253K |
|---|---|---|---|---|---|---|
| RTX 5060 Ti 16 GB (448, 36) | 103-153 | 85-146 | 50-115 | 164-256 | 115-224 | 62-166 |
| RTX 5070 Ti (896, 70) | 204-270 | 168-244 | 98-172 | 331-458 | 233-374 | 122-234 |
| RTX 5080 (960, 84) | 243-307 | 197-267 | 117-185 | 393-516 | 273-398 | 146-246 |

**The rule** (additive): round = trellis + rest(8K) + context extra.

- Trellis: the capped round at the card's SM count less the rest (the cap's own trellis time, at the ~2.8 GHz the 5090
  decoded at under the cap; no clock correction), floored at 10.41 GB / (92% of the card's bandwidth).
- Optimistic end (the high figure): the capped 5090 measurement itself, with the rest and the context at the 5090's time,
  floored at their bytes over 90% of the card's bandwidth.
- Conservative end (the low figure): the rest and the context multiplied by max(1,792 / bandwidth, 170 / SMs), which is 4.72,
  2.43 and 2.02 for the three cards: compute-bound, as attention's ~175 GB/s on the 5090 suggests.
- The range is narrow at 8K and wide at 253K, where attention dominates and is the unknown.

**Assumptions.** The 5090's per-SM trellis throughput, L2 (96 MB) and DRAM behind the capped kernels carry over to the
card; a card's sustained clock moves the trellis part in proportion; tokens per round as measured (greedy coding, about
6.0); PCIe is off the decode path (the host embedding is a 10 KB row a token; Gen 3 here); the cards' bandwidths are
`budget.py`'s; SM counts (36 / 70 / 84) and boost clocks are NVIDIA's spec sheets (`budget.py` has no SM counts).

**Tokens per round caveat.** The table is at about 6.0 tokens per round, the greedy coding figure. Agent traffic is nearer
3.5 (the study's agent tau 3.46) and would scale every tokens-per-second figure by about 0.58.

**Against the study.** Its model (`budget.py`: tau x bandwidth x e / bytes, e 0.45-0.63, with our bytes and tau) is printed
beside each cell in `estimate.txt`: it agrees at 8K (5070 Ti 200-280, 5080 214-300) and is optimistic at 253K, which it
treats as bandwidth-bound. Against PIANO-VRAM.md's SC 3.0 H4 table, at its agent tau 3.46, 8K, 1 lane, this estimate gives
5060 Ti about 59-88, 5070 Ti about 117-155, 5080 about 140-176 tokens/s against its 59-82 / 117-164 / 125-176.

### By-product: cold TTFT (server) with the trellis GEMMs capped

| | 8K | 64K | 253K |
|---|---|---|---|
| none | 2.4 s (3,478 tok/s) | 19.5 s (3,363) | 96.9 s (2,616); 261,632: 101.3 s |
| cap 84 | 3.4 s (2,442) | 27.5 s (2,387) | 126.9 s (1,997) |
| cap 70 | 4.0 s (2,067) | 32.2 s (2,032) | 144.5 s (1,754) |
| cap 36 | 7.1 s (1,159) | 56.4 s (1,163) | 236.6 s (1,071) |

Attention ran on 170 SMs here, so these are optimistic for a real card: its long prompts are slower still.

## Finding

Observed:

1. **On the 5090 the 16 GB artifact decodes 416 tokens/s at one lane and 684 across two at 8K**, 216 and 275 at 253K, at
   about 6.0 tokens per round on greedy coding text; the round reads 12.15 GB at 8K and 17.28 GB for two lanes at 253K, at
   47% down to 22% of the card's bandwidth.
2. **The trellis part is 9.40 ms a round at one lane and 10.53 ms at two**, the same under all three caps within the noise,
   at 55-62% of the DRAM roofline.
3. **The rest grows with context and is not bandwidth-bound**: 5.1 ms at 8K, +3.0 ms at 64K and +12.9 ms at 253K on one lane,
   about 175 GB/s of KV read.
4. Tokens per round did not drop with context on these cells (5.8-6.2 from 8K to 253K).

Inferred (the estimate):

- A 5070 Ti or 5080 should decode in the low-to-mid hundreds of tokens/s at 8K, near the 5090's own capped numbers; a 5060 Ti
  about 100-150 at one lane. At 253K the range is wide (a factor 2-3) because attention's behaviour on a smaller card is
  unmeasured.

## Implications

- The speed gate is not decided here: the owner deferred it to phase 2 (Q8), and `ignis-bench small-card` is how a user
  contributes a real number.
- The long-context rows of the estimate carry the most uncertainty; quote the 8K rows with the range, and the 253K rows as
  bounds.
- A 36-SM cap on the 5090 is not a 5060 Ti: it leaves attention and the drafter on 170 SMs and the DRAM at 1,792 GB/s.

## Limits and unknowns

- Not measured on any 16 GB card; the per-card figures apply a rule to 5090 measurements and inherit every assumption above.
- The cap reaches only the trellis ops: attention, the GDN recurrence, the drafter and the sampler ran on 170 SMs. The
  conservative end is a guess at how those scale.
- Clock: the 5090 decoded at 2,734-2,872 MHz; a card's sustained clock is not modelled.
- Tokens per round is greedy coding (about 6.0) on a word-soup prompt; agent traffic is about 3.5, and its behaviour at long
  context is not measured here.
- The capped TTFT is optimistic for the same reason; no real card's prefill was run.
- Gen 3 PCIe on this host; a card on Gen 4 or 5 is not tested, and the decode path does not use the link.
- 1 x 253K and 2 x 253K are 253,440 tokens a lane, not 262,144.

## Follow-ups

- Real numbers: a `small-card` run on a 5060 Ti, 5070 Ti or 5080 (phase 2).
