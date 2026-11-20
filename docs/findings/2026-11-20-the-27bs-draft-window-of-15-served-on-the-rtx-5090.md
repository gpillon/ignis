# The 27B's DFlash2 at up to 15 drafts served on the RTX 5090 after spec #684: structured output 24.6-64.3% faster, no free-text cell more than 2% below main, plain decode, context and pages held, and the `rtx5090` profile's default at 15

- Kind: experiment
- Status: current
- Observed: 2026-11-20
- Last verified: 2026-11-20
- Scope: decode / the 27B NVFP4 artifact's DFlash2 rounds on the RTX 5090 (Windows, PCIe Gen 3), main at 6d671912
  at 7 drafts against spec #684's integration build at 7f7db1eb at 15 drafts with the default shape rule and shape
  policy; the round timing of the merged build at 889c1307; the prototype measurements behind the spec (2026-11-19)
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/696,
  https://github.com/gpillon/ignis/issues/684,
  https://github.com/gpillon/ignis/issues/666,
  https://github.com/gpillon/ignis/issues/690,
  https://github.com/gpillon/ignis/issues/694,
  [ADR 0104](../adr/0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md),
  [the 27B's decode served after spec #672](2026-11-18-the-27bs-decode-served-on-the-rtx-5090-after-spec-672.md)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, FP8, served vs main at 7 drafts: 15 drafts, two round shapes: structured output +24.6 to +64.3%, no free-text cell below -2%, plain decode and KV pages held, cold TTFT +0.7% (noise); rtx5090 default 15.

## Question

On main the 27B's DFlash2 drafts at most 7 tokens a round, a range taken from the reference and never measured past
(#666). On structured output every round is full: a greedy JSON answer commits 7.93 tokens a round against a ceiling
of 8. Spec #684 lets a DFlash2 load draft up to 15 and gives such a load two **round shapes**, the **base shape** (8
columns a lane, main's round) and the **extended shape** (16 columns a lane at 15 drafts, at 1 to 4 lanes), chosen per
round by the **shape rule** (what a lane asks: the extended shape after a base round that took all 7 drafts, kept
while its extended rounds take at least 6) and the **shape policy** (`aggregate` by default, `all` selectable; ADR
0104). #696 asks:

- served, against main at 7 drafts: does the 15-draft load with the default rule and policy gain on structured output,
  and lose nothing on free text, plain decode, the first token, the context or the KV pages;
- at two lanes, does `aggregate` raise the pair's committed tokens over `all`;
- what the merged build's round costs on the GPU clock: each shape's period, the ratios the policy reads, the
  extended round's attention route and the three cost levers, each kept or dropped;
- what the prototype measured behind the spec, which rule was served and refuted, and what the offline replays got
  wrong.

## Evidence

### Conditions

RTX 5090 under Windows (WDDM) on a PCIe Gen 3 x16 host, one card, one run on it at a time: each batch held the
machine's GPU lock for its whole length, and before every leg the card held no other ignis or ninfer process and no
build ran. The 27B NVFP4 artifact, `--model-exception-weights fp8` named, hq-e8-2b KV, a context of 262,144 and rope
none, an empty config file (no user config file read), host pools 8 GiB and 2 GiB, the Playground off, idle unload
off, `IGNIS_ROUND_TIMING=1`, no other `IGNIS_` variable (the environment dumped at every batch).

The executables were frozen and their sha256 checked at every batch:

- **M**: main 6d671912, `--spec-backend dflash2 --spec-draft-tokens 7`;
- **N**: the integration build 7f7db1eb, `--spec-draft-tokens 15`, the shape rule's defaults (enter 7, stay 6) and
  `aggregate` (16 recent rounds, period ratios 1.128, 1.29, 1.363, 1.344 at 1 to 4 lanes);
- **N-all**: N's executable with `--spec-shape-policy all`.

One server start a leg, legs mirrored (M N N M; the pairs N A M M A N), each leg opened by an untimed warm-up (one lane,
then two). The two M legs at a batch's ends are its **control**, main against main. Each leg was checked for foreign
GPU activity: the server log's request count against the client's, the largest gap between two streamed tokens of
every timed request, and the VRAM in use before the load.

Requests are streamed chat completions. A request's tok/s is (tokens - 1) over the time from its first streamed
chunk to its last; tokens a round and ms a round come from the server's `ignis.request.done` (`spec.rounds`,
`spec.accepted`); the shares of extended and of split rounds from `/metrics` deltas
(`ignis_speculative_shape_rounds_total`, `ignis_speculative_shape_split_rounds_total`). A request with no temperature
takes the model card's sampling (thinking on: temperature 1.0, top_p 0.95, top_k 20; thinking off: 0.7, top_p 0.8,
presence penalty 1.5), written "-d" below. A cell's **spread** is the larger of the arms' leg ranges and the
delta's 95% confidence interval (1.96 times the pooled standard error).

The prompts: a JSON answer (greedy, thinking off, 7,229 tokens on main), a conversation and prose (temperature 0.6,
thinking off, up to 1,200 tokens), fixed prompts shared with the prototype's measurements; JSON-long, a JSON array of
80 fictional employees (the #672 A/B's cell); a `response_format` json_schema answer of 200 items, strict; code, a
Rust LRU-cache module of about 300 lines with tests (thinking on, up to 6,144 tokens); an agent turn, a multi-turn
coding task with three tools ending in two `write_file` calls (thinking on, up to 4,096 tokens), its tools strict in
the strict-tools cell; the 8,192-token prompt of the #646 and #672 A/Bs for plain decode.

Eleven batches, 08:55 to 10:30, about 90 minutes under the locks, each under 14 minutes. The raw legs, client
records and summaries are in `.scratch/684/gate/` of the coordinator's checkout; the timing session's in
`.scratch/684/timing/`.

### Gain cells (one lane)

| cell | M tok/s | N tok/s | change | control M2-M1 | tokens a round M -> N | ms a round M -> N | N's extended share | text |
|---|---|---|---|---|---|---|---|---|
| json_schema, greedy (2 an arm) | 525.79 / 525.46 | 863.33 / 863.68 | **+64.3%** | -0.06% | 7.98 -> 14.83 | 15.18 -> 17.18 | 99.8% | main's (14,639 tokens), valid in 4 of 4 |
| JSON answer, greedy (2 an arm) | 529.15 / 529.02 | 799.97 / 800.02 | **+51.2%** | -0.03% | 7.93 -> 13.41 | 14.98 -> 16.76 | 98.5% | another, 7,229 tokens both |
| strict tools, greedy, thinking on (2 an arm) | 368.50 / 368.70 | 459.54 / 459.20 | **+24.6%** | +0.06% | 6.00 -> 8.04 | 16.43 -> 17.66 | 52.1% | another; calls parse 4 of 4 both |
| json_schema-d, thinking on (6 an arm) | 435.98 (sd 17.0) | 569.92 (sd 23.3) | **+30.7%** (CI +-5.3) | +6.17% | 6.78 -> 9.10 | 15.22 -> 16.43 | 69.5% | sampled; 6 of 6 valid both |
| JSON answer-d, thinking off (6 an arm) | 516.35 (sd 20.8) | 795.80 (sd 10.4) | **+54.1%** (CI +-3.6) | +3.71% | 7.92 -> 13.45 | 15.35 -> 16.90 | 97.9% | sampled |

- Every greedy text was the same in every request of an arm, in both legs.
- JSON answer-d: M's first request in one leg holds a 582 ms stall (473.96 tok/s; M's other five 522-527); without
  it M is 524.5 and the change +51.7%.
- json_schema-d: 5 of the 12 requests, in both arms, ran as two server requests inside one HTTP stream (the first
  cancelled by its consumer, the second continuing from a prompt checkpoint with the output so far in its prompt), a
  path of thinking under a grammar that is not the round's. Their first part's rounds reach neither `/metrics` nor a
  done line, so the tokens a round and the extended share cover the recorded part; tok/s is the client's whole
  request. The control's +6.2% follows which M leg had the two-part requests (two in one, none in the other).

### Guard cells

| cell | n an arm | M tok/s: mean (sd), legs | N tok/s: mean (sd), legs | change | 95% CI | control M2-M1 | tokens a round M -> N | ms a round M -> N | N's extended share |
|---|---|---|---|---|---|---|---|---|---|
| conversation, T 0.6 | 20 | 232.81 (10.5), 230.28 / 235.34 | 234.82 (16.9), 231.08 / 238.56 | +0.86% | +-3.75% | +2.20% | 3.446 -> 3.507 | 14.78 -> 14.91 | 11.8% |
| prose, T 0.6 | 20 | 181.59 (5.4), 182.33 / 180.85 | 181.26 (5.8), 181.43 / 181.09 | -0.18% | +-1.93% | -0.81% | 2.688 -> 2.677 | 14.78 -> 14.75 | 2.1% |
| conversation-d | 20 | 222.51 (11.5), 219.60 / 225.43 | 231.94 (17.1), 237.04 / 226.84 | +4.24% | +-4.06% | +2.66% | 3.293 -> 3.461 | 14.79 -> 14.91 | 10.9% |
| prose-d | 20 | 195.18 (5.7), 194.38 / 195.97 | 193.82 (5.4), 194.43 / 193.22 | -0.69% | +-1.77% | +0.82% | 2.889 -> 2.869 | 14.79 -> 14.79 | 4.0% |
| code-d (two batches) | 20 | 236.84 (9.6) | 243.29 (14.8) | +2.73% | +-3.27% | +3.19% / -0.86% | 3.636 -> 3.791 | 15.43 -> 15.65 | 14.4% |
| agent-d | 20 | 299.81 (27.0), 298.91 / 300.71 | 324.90 (50.3), 328.25 / 321.56 | +8.37% | +-8.35% | +0.60% | 4.802 -> 5.381 | 16.16 -> 16.60 | 31.1% |

- Every code-d request reached its 6,144-token cap. agent-d: the turns that wrote a file, the fast ones, were 13 of
  20 under M and 15 of 20 under N; on those alone, about 315 against 342 tok/s (+8.6%). Calls parsed in 19 of 20 in
  both arms, each failure a `write_file` cut by the cap.
- The two cells below main, prose and prose-d, are free text inside their spread.

**Plain decode** (`--spec-backend off` on both executables, one batch): the 8,192-token prompt's answer, greedy
(3,174 tokens, the same text in all four legs), 87.03 / 87.00 tok/s under M and 87.00 / 87.11 under N (+0.05%, CI
+-0.13%, control -0.03%); the conversation at T 0.6, 3 an arm a leg, +0.12% (control -0.04%).

**Context and pages.** Every speculative load held 262,144 tokens of context and 17,085 KV pages under both
executables, at the same budget (30,220,824,576 B, the same free memory at start); every plain load 19,997 pages. In
the pair batch the M, N and N-all loads at equal budgets held 17,048; two later legs started at other free memory:
17,164 pages with 2,120 MiB in use before the load, 16,598 with 2,445 (another program's VRAM on the desktop), against
2,164-2,185 MiB at every other leg.

**The first token** (a cold prompt of 16,676 tokens, cache-busted, one token; 3 a leg right after the warm-up in the
seven speculative batches, 4 in the depth batch below: 16 legs an arm): N later by 8.4 ms (+0.73%), leg-level 95% CI
+-7.4 ms (requests: +9.2 ms, CI +-5.5); the server's own `duration_ms`, +7.7 ms (CI +-4.5, 14 legs an arm). In the
five batches of the guard cells, +1.48, +0.13, +1.30, +0.15 and +0.57%. A batch of its own (cold prompts at three
depths, 4 a depth a leg, M N N M) did not reproduce it:

| depth | M (legs) | N (legs) | change | 95% CI |
|---|---|---|---|---|
| 4K | 0.3462 s (0.3463 / 0.3461) | 0.3377 s (0.3387 / 0.3368) | -8.5 ms (-2.4%) | +-13.1 ms |
| 16K | 1.1735 s (1.1656 / 1.1814) | 1.1760 s (1.1805 / 1.1714) | +2.5 ms (+0.2%) | +-15.3 ms |
| 64K | 5.7684 s (5.7456 / 5.7912) | 5.7930 s (5.7930 / 5.7929) | +24.5 ms (+0.4%) | +-26.1 ms |

Main's own two legs there differ by 16 ms at 16K and 46 ms at 64K. With speculation off the cold 16K prompt read
-0.8% (one batch, 2 legs an arm).

### Pair cells (two lanes)

One lane writes greedy JSON while a conversation at T 0.6 runs back to back on the other; 3 pairs of each kind a leg,
legs N A M M A N (A = N-all). Each request's rate as above, the conversation over the requests that ended inside the
JSON request's decode window; the **aggregate** is the tokens both lanes committed inside that window over the window
(a conversation that straddles it prorated by time); the split share is the share of the window's rounds in which one
lane asked the extended shape and the other did not.

| pair | arm | n | JSON lane tok/s | conversation tok/s (n) | aggregate (sd) [legs] | split share | extended share | JSON lane: tokens, ms a round | conversation: tokens, ms a round |
|---|---|---|---|---|---|---|---|---|---|
| JSON answer + conversation | M | 6 | 457.6 | 199.9 (12) | 653.4 (17.2) [659.4 / 647.4] | 0 | 0 | 7.93, 17.33 | 3.43, 17.19 |
| | N-all | 6 | 494.8 | 199.1 (12) | 690.1 (26.0) [699.7 / 680.5] | 86.6% | 10.6% | 8.52, 17.23 | 3.39, 17.02 |
| | **N** | 6 | **638.9** | **174.6** (6) | **812.8** (11.9) [823.1 / 802.5] | 84.4% | 98.1% | 13.36, 20.91 | 3.63, 20.77 |
| JSON-long + conversation | M | 6 | 333.9 | 200.8 (18) | 531.0 (7.6) [533.6 / 528.4] | 0 | 0 | 5.74, 17.19 | 3.43, 17.08 |
| | N-all | 6 | 358.1 | 202.8 (18) | 556.6 (11.2) [557.5 / 555.7] | 54.9% | 6.0% | 6.02, 16.79 | 3.39, 16.68 |
| | **N** | 6 | **375.7** | **189.9** (16) | **563.3** (23.4) [577.2 / 549.3] | 54.1% | 27.6% | 6.68, 17.75 | 3.35, 17.61 |

- JSON answer + conversation: under `aggregate` the pair committed 17.8% more than under `all` (24.4% more than main);
  the JSON lane +29.1% over `all` (+39.6% over main), the conversation -12.3% under `all` (-12.7% under main). #690's
  estimate from the measured periods, at the two-lane ratio 1.29, was about +15% together, +32% and -22%.
- JSON-long + conversation: `aggregate` +1.2% over `all`, inside the spread (N's legs, 577 and 549, straddle `all`'s
  557 and 556), +6.1% over main; the JSON lane +4.9%, the conversation -6.4%. `aggregate` ran 27.6% of these rounds
  extended where #690's table expected the base shape; `all` ran 6.0%, the rounds in which both lanes asked.
- `all` against main: +5.6% and +4.8% on mostly base rounds, which carry the cost levers' shorter two-lane base round
  (-0.6% and -2.3% ms a round) and each start's own multi-lane greedy texts, which vary between starts.
- Control M2-M1: -1.8% (one M leg's first pair held a 565 ms stall: its JSON lane 431.8, its conversation 159.6;
  the other pairs 661-662) and -1.0%.

### The round on the GPU clock (merged build 889c1307)

The timing session (2026-11-20, 07:26-08:21, seven batches, 47 minutes under the lock): the same pins with 8 lanes,
`--spec-draft-tokens 7` (main) or 15 (the build), one start a leg, legs mirrored; N lanes write N distinct greedy JSON
answers at once; a block of 256 periods is kept only when every one of its rounds ran at the cell's lane count. A
lever stays where the round's span (the append: its period) falls by more than the spread, the larger of the two
arms' leg ranges.

- **Calibration**: one lane, the JSON answer, main against the build at the base shape with the three levers off:
  period 14,961.8 against 14,953.0 us (spread 15.2), span 14,419.0 against 14,404.3, gap 528.7 against 529.9. The
  build's round timing reads main's round as main's did.
- **Period ratios** (`IGNIS_ROUND_SHAPE=base` against `=extended`, the levers on, two orders):

| lanes | base period (spread) | extended period (spread) | ratio | at a matched context | the prototype's |
|---|---|---|---|---|---|
| 1 | 14,915 us (7) | 16,878 (5) | 1.132 | 1.128 | 1.15 |
| 2 | 16,468 (83) | 21,202 (6) | 1.287 | 1.290 | 1.23 |
| 3 | 19,170 (1) / 19,250 (178) | 26,016 (16) / 26,349 (7) | 1.357 / 1.367 | 1.362 / 1.364 | 1.27 |
| 4 | 21,596 (110) / 21,595 (29) | 28,907 (13) / 28,834 (19) | 1.339 / 1.335 | 1.345 / 1.343 | 1.31 |

  The defaults became the matched-context values pooled over both orders, 1.128, 1.29, 1.363 and 1.344 (#690). The
  card's 575 W limit bites at the base shape's 4 lanes and the extended shape's 3 and 4 (SM clock medians 2,715-2,767
  MHz at the extended shape's 3 and 4 lanes, 2,801-2,816 at the base shape's 4, against 2,820-2,880 below the limit),
  so 3 lanes reads above 4 in either order; and the ratio grows with context (the
  extended period's slope 1.1 to 2.3 times the base one per 1K tokens), so each value is a figure at about 3.5-5K
  tokens. The base shape's 83-178 us spreads at 2-4 lanes are two multi-lane texts, not noise.
- **Route** (#688): the extended round's attention on ignis's hq-e8-2b route in 8-column chunks against the vendored
  chunks (`IGNIS_EXTENDED_ATTENTION=vendored`): period 16,878 / 21,177 / 28,865 us against 17,208 / 21,525 / 29,278 at
  1 / 2 / 4 lanes, ignis's faster by 331 / 348 / 414 us (span 348, spread 24; 339, 37; 408, 70).
- **Levers** (#693, #694; each switch off minus on, 2+2 legs):

| lever | where | off - on | spread | verdict |
|---|---|---|---|---|
| FP8 exception product on a 16-token tile | extended, 1 lane | span +337 us | 24 | stays |
| | base, 2 lanes | span +357 us paired, +324 against a text-matched leg | | stays |
| | extended 2 and 4 lanes, base 1 lane | +9, +72 paired, +5 | 20, 70, 10 | not reached by construction (32, 64, 8 tokens a call) |
| the drafter's append projects only its key and value rows | extended, 1 / 2 lanes | period +148 / +198 us | 16 / 18 | stays |
| | base, 1 / 2 lanes | period +91 / +176 us paired (+133 text-matched) | 26 / - | stays |
| the drafter's small products on 4 warps a block | extended, 1 lane | span +130 us | 10 | stays |
| | base, 2 lanes | span +118 us | 16 | stays |
| | base, 1 lane | span +10.8 then +8.8 us in a third batch; +9.6 pooled | 9.7, 6.6; pooled 14.3 | inside the pooled spread: the owner kept it |

  No lever was dropped. At the base shape's one lane the small products save about 10 us a round (0.07%), positive in
  each of the eight paired blocks, inside the spread once two batches are pooled; the lever is bit-identical, and the
  owner kept it there. Inferred from the lever arms, not timed against main on one workload: with the defaults a
  7-draft load's one-lane base round runs about 100 us under main's.

### What the prototype measured (2026-11-19)

Before the spec, prototype builds on main 6d671912 measured the following on the same card behind measurement
switches, one lane unless said, served over the streaming API with an empty config file and pinned flags, legs
alternated.

- **Acceptance by position.** With every round at 16 columns, the share of rounds that took the draft at each
  position among those that proposed it (thinking off unless said; the JSON answer, JSON-long and code greedy; the
  agent turn greedy with thinking on):

| text | tokens a round | 1 | 4 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JSON answer | 13.34 | 100 | 99 | 96 | 90 | 84 | 79 | 76 | 73 | 63 | 46 | 36 |
| JSON-long | 7.44 | 87 | 58 | 49 | 45 | 39 | 31 | 27 | 22 | 16 | 12 | 10 |
| code | 7.58 | 94 | 66 | 46 | 40 | 35 | 29 | 25 | 20 | 16 | 13 | 11 |
| agent turn | 8.13 | 90 | 64 | 51 | 44 | 40 | 36 | 33 | 31 | 24 | 20 | 17 |
| conversation, T 0.6 | 3.45 | 75 | 23 | 9 | 6 | 5 | 4 | 2 | 1 | 1 | 0 | 0 |
| prose, T 0.6 | 2.80 | 73 | 16 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

  The drafter, trained at a block of 8, keeps its acceptance past position 7 where the text is predictable; on
  sampled free text almost nothing past 7 is taken.
- **The round's cost.** A 16-column round at one lane against 8 columns, served on ignis's attention route: 17.17 ms
  against 14.93 (+2.24 ms, +15%) for 13.4 tokens a round against 7.9 on the JSON answer; on the vendored chunks 17.56
  ms. A host-wall sweep of 8 to 16 columns (FP8 exception weights, 4K context) read 15.04, 16.38, 16.78, 17.24 and 17.73
  ms at 8, 10, 12, 14 and 16 columns at one lane, 17.24 to 22.72 at two and 21.71 to 29.92 at four: the first
  columns past 8 are the expensive ones (+1.2-1.3 ms from 8 to 10 at one lane: the second attention chunk, the FP8
  product leaving its GEMV, the drafter's wider block), each further pair about 0.4-0.5 ms.
- **The cost by family** (one lane, 4K context, a 16-column round against an 8-column one, kernel time per round
  read under Nsight Systems, which inflates the round, with the extended attention on the vendored chunked route
  the merged build no longer takes by default; overlapping kernels make the sums exceed the span):

| family | 8 columns (us) | 16 columns (us) | added (us) | what changes |
|---|---|---|---|---|
| attention | 523 | 1,545 | +1,021 | ignis's one-pass verify kernel -> two vendored chunks |
| drafter | 1,998 | 2,882 | +884 | its forward over twice the columns, its head over 15 drafts |
| gated delta net (record, conv, gating) | 1,424 | 2,116 | +692 | the record 461 -> 773; the gating leaves its small-T route for a GEMM |
| FP8 exception products | 464 | 773 | +308 | one lane leaves the W8A16 GEMV for a per-token quantize and the W8A8 GEMM |
| NVFP4 products | 9,396 | 9,503 | +107 | weight-bound: twice the columns for 1% more |
| output head, norms, rope, embedding, accept | 1,124 | 1,150 | +26 | |
| sum (span) | 14,928 (14,271) | 17,974 (16,952) | +3,045 (+2,681) | |

  The attention row is why the extended round runs on ignis's route in 8-column chunks (#688, -0.39 ms served);
  the cost levers came from the FP8 and drafter rows (the GDN row has none): together they took the 16-column round from 17.28 to
  16.70 ms (span: the FP8 tile -278 to -301 us, the drafter's small products -82 to -105; period: the append -111),
  and the JSON answer to 798 tok/s.
- **Served, the prototype at 15 against main at 7** (one lane; 2 requests an arm on the greedy cells, 4 to 12 on the
  others):
  the JSON answer greedy 530 -> 774 tok/s (+46%), at default sampling 521.7 -> 756.2 (+45.0%); json_schema greedy
  516.9 -> 821.8 (+59.0%, main's text), at default sampling 429.5 -> 562.8 (+31.0%); strict tools greedy 358.0 -> 439.2
  (+22.7%); code-d +1.3% +- 2.2, agent-d +4.1% +- 8; conversation and prose at T 0.6 -1.9% +- 2.6 and -0.5% +- 1.7.
- **A window fixed at 15** would cost free text 7-8%: simulated from the fixed-shape round records (conversation
  -7.8%, prose -7.3% against 7 drafts), since it pays the wider round and takes no more drafts. Hence a shape per round.
- **Memory.** With the two shapes on one set of device buffers, the 15-draft load at 8 lanes planned the 7-draft
  load's bytes plus 528 and kept the same KV pages (17,995 against 17,995); the extended shape's captured graphs hold
  about 20 MiB outside the plan. A first prototype with a second set of buffers had lost 393 pages (2.2%).
- **Numerics.** Over 19,720 teacher-forced rows of three long answers the extended round's rows sat a median 18.6
  BF16 ulps from a plain round's (p99 97.8) where main's 8-column round sat 18.5 (p99 97.9), 213 moved picks against
  239: the measurement behind ADR 0104's class test.

### The rule that was refuted, and what the offline replays got wrong

Two offline replays of round records ranked rules, and both ranked them wrongly against served arms.

- **The first** walked the fixed-shape records position by position and simulated each rule's tok/s. It chose the
  rule the build ships (enter after one full base round, stay while an extended round takes at least 6) and could
  not rank its threshold among 4, 6 and 8: against a served A/B of that rule it missed by more than the gaps between
  them (the agent turn +40.0% simulated against +18.4% served, JSON-long +22.7% against +17.0%, the conversation +3.1%
  against -1.7%; the JSON answer +42.6% against +43.5%).
- **The second**, calibrated on that rule's own records, proposed another: enter after two full base rounds in a row,
  leave after two extended rounds in a row that each took fewer than 4. Replayed, it matched the shipped rule on
  predictable text and removed its small loss on free text. Served (one lane, the same day, against main and the
  shipped rule), it kept the JSON answer's gain (+47.4% against the shipped rule's +46.6%, main's text) and lost greedy
  code's (-2.6% against +19.6%) and the greedy agent turn's (+1.7% against +20.2%); on JSON-long at default sampling
  +2.2% +- 4.4 against +8.9% +- 2.3. The replay had it at +16.2% and +17.3% on code and the agent turn, and over the
  shipped rule by 0.8-2.2 points everywhere they differed; served, the shipped rule was ahead by 3-19 points on the
  four gain cells other than the JSON answer. None of its free-text cells fell below main by more than one standard
  error, at 6 to 10 requests an arm.
- **What the replays got wrong.** The extended share each rule's rounds would run, the replay had right (within 2
  points on every sampled cell). What an extended round then takes, it did not: a shape change is a text change (a
  greedy text differs from the round where its shape first differs), and acceptance depends on the shape history
  (under the second rule the rounds took fewer drafts at positions 1 to 7 than the records the replay carried over).
  So a rule is judged on served arms, never on a replay alone (ADR 0104).
- Handing the previous round's draft confidence to the rule was replayed at +0.2 points over the accept history
  alone, and is not built.

## Finding

**Observed.** Against main at 7 drafts, the 27B's DFlash2 at 15 drafts with the default shape rule and `aggregate`
decodes structured output faster on the RTX 5090 at one lane, greedy and at the model card's sampling: a
json_schema answer 525.6 -> 863.5 tok/s (+64.3%, main's text, 14.83 tokens a round against 7.98) and +30.7% at
default sampling; the JSON answer 529.1 -> 800.0 (+51.2%) and +54.1% at default sampling; an agent turn with strict
tools +24.6%. The JSON answer runs 98.5% of its rounds at the extended shape at 13.41 tokens a round; a conversation
at T 0.6, 11.8% at 3.51 tokens a round (main 3.45).

**Observed.** No guard cell is more than 2% below main: conversation and prose at T 0.6 +0.86% and -0.18%, at default
sampling +4.24% and -0.69%, code-d +2.73%, agent-d +8.37%, at 20 requests an arm; the two below main are free text
inside their spread. Plain decode +0.05% and +0.12%. The context (262,144) and the KV pages (17,085 speculative, 19,997
plain) are equal at equal budgets.

**Observed.** The first token of a cold 16K prompt came 8.4 ms (0.73%) later after the warm-up, at the edge of what
16 legs an arm resolve (CI +-7.4 ms; the server's own figure +7.7, CI +-4.5); a batch of its own did not reproduce it
(-2.4%, +0.2% and +0.4% at 4K, 16K and 64K, each inside its spread). **Decision (the owner, 2026-11-20):** the reading
is noise, and the gate passes.

**Observed.** At two lanes, `aggregate` raised a JSON answer beside a conversation by 17.8% in aggregate over `all`
(the JSON lane +29.1%, the conversation -12.3%), and a moderately predictable answer beside a conversation by 1.2%,
inside the spread, running 27.6% of its rounds extended. **Decision (the owner, 2026-11-20):** the policy's default
stays `aggregate`; `all` is the setting for a lane that must keep main's speed beside another.

**Observed.** On the GPU clock the merged build's extended round costs 1.13, 1.29, 1.36 and 1.34 times the base
round at 1 to 4 lanes (about 3.5-5K tokens of context), its attention is 0.33-0.41 ms faster on ignis's route than on
the vendored chunks, and each cost lever clears its spread where its calls reach it. **Decision (the owner,
2026-11-20):** the small-products lever stays at the base shape's one lane too, where its 10 us sits inside the
pooled spread. No lever was dropped.

**Inference.** The gain is the drafter's acceptance past position 7 on predictable text (90 to 36% at positions 8 to
15 on the JSON answer), paid by a wider round of 13% at one lane; free text rarely asks the extended shape, so it
decodes as at 7 drafts. Under `aggregate` a conversation pays for its neighbour's extended rounds by design.

The gate passed by #696's rule, so the `rtx5090` profile's default draft window is 15 from the commit that records
it; every other profile keeps 7.

**Not confirmed here.** The RTX PRO 6000 (Linux), the other card that runs the 27B's NVFP4 artifact on ignis's own
attention route, still has to confirm it: the gain and guard cells at 15 drafts against 7, the period ratios (its
power limit and clock are its own), the route and the three levers, which are on by default on compute capability
12 and were measured on this card alone. Its profile keeps 7 until then.

## Implications

- The `rtx5090` profile drafts the 27B up to 15 tokens a round ([profiles](../user/concepts/profiles.md#rtx5090));
  the guide's decode figures for the 27B on the RTX 5090 carry these cells ([One round for every
  lane](../user/why-ignis/decode-round.md)), and the speculative-decoding page the served pair figures in place of
  #690's estimate ([past seven drafts](../user/concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes)).
- `rtx-24gb`, `rtx-16gb` and `rtxpro6000` keep 7: a window past 7 is held correct on their artifacts and shapes, and
  unmeasured for speed off this card. `rtx-12gb` keeps the MTP head.
- A lane that must keep main's speed beside an agent writing JSON wants `--spec-shape-policy all`: under `aggregate`
  the conversation lost 12.3% against `all` in that pair.
- The period ratios are mid-context figures and the ratio grows with context: at long contexts `aggregate`
  underestimates the extended round's cost to every lane.
- A shape rule is judged on served arms: an offline replay predicts how often each shape runs, not what it takes.

## Limits and unknowns

- One machine: an RTX 5090 under Windows (WDDM) on a PCIe Gen 3 host; two legs an arm a cell, 2 requests an arm on
  the greedy gain cells, 6 on the default-sampling ones, 20 on the guard cells, 6 pairs an arm a pair.
- The gate pinned a 262,144-token context and rope none. The `rtx5090` profile's own load (917,504 tokens, YaRN at
  3.5, vision on, its headroom) was not served at 15 drafts; at its 8 lanes the plan is 528 bytes over a 7-draft
  load's, which the plan tests hold.
- Not served: windows 8 to 14, rounds of 3 and 4 lanes beside a conversation, a vision request at 15 drafts, the
  trellis artifacts (held correct by the round identity and class tests, not timed), BF16 exception weights, and any
  card off compute capability 12, where the extended round attends on the vendored chunks.
- Multi-lane greedy texts vary between starts of one configuration, so the pair cells compare texts that differ
  between legs; the JSON-long pair's +1.2% is inside that spread.
- Two stalls fell in M legs (582 ms in one JSON answer-d request and 366 ms in a json_schema-d one in one leg; 565 ms
  in one pair): they lower main's means there, and none touches a guard cell (the prose legs' largest gap 17-18 ms).
- json_schema-d's two-part requests are a path of thinking under a grammar, in both arms; it was not investigated.
- The first-token reading after the warm-up and the depth batch disagree; at most about 1%, which these legs do not
  resolve.
- The prototype's figures are from builds that are not the merged code (rewritten since), one lane, 2-12 requests an
  arm; the cost by family was read under a tracer on the attention route the merged build no longer takes by default;
  the 7-8% cost of a fixed window is simulated from round records, not served.
- The small-products lever's 10 us at the base shape's one lane is not resolved by the legs.

## Follow-ups

- The RTX PRO 6000 run named above, before its profile's default moves.
- #597: a drafted greedy decode against a plain one on the 27B, which the extended round does not widen.
- The spec's out-of-scope studies, each its own `idea` issue: mixed lane shapes in one round (the only way to an
  extended shape past 4 lanes), a rule that reads the same round's drafts, shapes below 8 columns for smaller cards.
