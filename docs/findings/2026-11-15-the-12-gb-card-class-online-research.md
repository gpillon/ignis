# The 12 GB card class: what the public evidence of 2026-11-15 says about the 27B below 3 bits, and what nobody has measured

- Kind: research
- Status: current
- Observed: 2026-11-15
- Last verified: 2026-11-17
- Scope: models / the 27B at 2.2 to 3.0 bits per weight in the trellis format, its public quants, the alternatives to it
  for a 12 GB card, and the public agent-task evidence
- Topic: artifacts-and-gates
- Related: [#630](https://github.com/gpillon/ignis/issues/630), [#632](https://github.com/gpillon/ignis/issues/632),
  [#633](https://github.com/gpillon/ignis/issues/633), [#634](https://github.com/gpillon/ignis/issues/634),
  [#635](https://github.com/gpillon/ignis/issues/635), [#636](https://github.com/gpillon/ignis/issues/636),
  [#638](https://github.com/gpillon/ignis/issues/638), [the 16 GB quality gates](2026-10-24-27b-16gb-quality-gates.md),
  [ADR 0098](../adr/0098-the-12-gb-card-class-serves-the-27b-from-a-recipe-of-our-own-under-a-byte-ceiling-on-one-lane-drafting-with-mtp.md)
- Superseded by: none
- Summary: No smaller Qwen3.8 exists, and the 27B's long agent tasks give way between 9.2 and 8.0 GB; on the RTX 5090 (KLD against BF16, set A, each pair at one shape) its 9.2 GB recipe ig28 reads 1.03x the 16 GB artifact, a 2.5-bit quant 1.66x.

## Question

A 12 GB card is to become a card class of its own, served from a 27B body of ignis's own recipe under a ceiling in
bytes (spec #630). Before that recipe is built: which public candidates exist, and which of them load with the K
classes the engine binds today; what do the public agent-task results say a body of about 9 GB, 8 GB and 7 GB costs a
coding agent; what is there to choose instead of the 27B; and what has nobody measured, so that the gate has to?

This finding opens with the online research of 2026-11-15 and the facts of the one public quant that binds today. It
has a section for the slope measured on ignis's own instrument by #632 (2026-11-16), one for the two candidate recipes
and the gate's numbers at the class's shape (#633 to #636, 2026-11-16), and says what those readings add.

## Evidence

Three research agents read the sources on 2026-11-15, one question each (quants below 3 bits, alternatives, task-level
evidence), with no GPU, no weights downloaded and nothing posted. Their reports, with every table's source and
harness, are the raw material:
[`.scratch/12gb-model-study-2026-11-15/`](../../.scratch/12gb-model-study-2026-11-15/) (`SINTESI.md`, the reports
`A-sub3bpw-quants.md`, `B-alternatives.md`, `C-task-level-evidence.md` and the common brief `BRIEF-COMMON.md`). The
folder is local to a clone; the tables below carry what must outlast it. The slope of #632 is measured, not read:
its records are in [`.scratch/630/632/`](../../.scratch/630/632/) (`comment-numbers.md`, the numbers as posted on #632;
`kld/compare-632.txt`, the KLD per kind and per window; `served/`, the loop set; `convert.log`), and the session's
decisions in `.scratch/630/orchestra/decisions.md`.

Conventions. GB is 10^9 bytes. "Body and head" is the device weights of the text scope (the token embedding lives in
host memory). Each table is one source and one harness: **numbers from different tables are never compared**. `chart
±0.5` marks a value read off a published chart image (no numeric table exists), good to about half a point on the
agent tasks and 0.2 on the others; a value printed on a chart is marked `printed`. Everything was fetched on
2026-11-15 unless a row says otherwise.

### Sources

| key | what | where |
|---|---|---|
| S1 | the integer-K and fractional-K ladder of the 27B in the trellis format, with its charts, recipes and per-branch `quantization_config.json` | `turboderp/Qwen3.8-27B-exl3` (refs, trees, README, discussions #3 and #12) |
| S2 | the quantizer's releases and its optimizer notes | `turboderp-org/exllamav3` (releases, `doc/optimize.md`) |
| S3 | a public quantization challenge: leaderboard of 122 entries, mean KLD and top-1 on one private corpus, updated 2026-11-03 | `StillDeadcode/qwen3.8-27b-quant-challenge` |
| S4 | the ladder above scored on that leaderboard, bit for bit, and a mixed-K re-quant | `Hero422/qwen3.8-27b-exl3-turboderp-ladder`, `Hero422/qwen3.8-27b-exl3-v2-rad` |
| S5 | an independent set with a multi-slice harness | `yeasah/Qwen3.8-27B-exl3` |
| S6 | the agent-task harness: the model page and the method page of a publisher of quants in another format, published 2026-11-11 | its Qwen3.8 model page and its "Dynamic 3.1" page (URLs in `A-sub3bpw-quants.md`, S6, and `C-task-level-evidence.md` §1) |
| S7 | third-party quants at 2.5 to 3.0 bits | `0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-2.5` and `-3.0`, `Derpyhue/Qwen3.8-27B-exl3-2.67bpw-SC-SH` (series 2.65 to 2.80), `nalimc/Qwen3.8-27B-exl3-SC_2.60bpw_H4`, `BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix`, `GestaltLabs/Qwen3.8-27B-EXL3-11.5GB` |
| S8 | fine-tunes of the 27B with low-bit builds | `JRamirez-UAB/Qwopus3.8-27B-Coder-Flash-EXL3-2.25bpw-24GB`, `jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2` |
| S9 | the same model in a second format, learned scalar, with its own card and per-tensor allocations | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` |
| S10 | the Qwen organisation's repositories and cards (27B, Flash-Next, predecessors) | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-Flash-Next`, `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-9B` |
| S11 | an independent blog run of the 27B in the second format against BF16 on Terminal-Bench 2.1 | a software company's engineering blog, 2026-08-26 (URL in `C-task-level-evidence.md` §2) |
| S12 | a sweep whose score includes tool-calling suites | the blog of a publisher of learned-allocation quants, 2026-09-14, updated 09-18 (URL in `C-task-level-evidence.md` §4) |
| S13 | the model's own MTP head, quantized | `HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3` |

The other cards and papers read are named where they are used and in the reports.

### The candidates

**Public quants of the 27B, by what the engine binds today** (integer K 2 to 5 in the body, a 4-bit head, codebook
`mul1`; a fractional K, a K of 1 or a 6-bit head is kernel work). Sizes are the sums of each branch's `n_bytes` in its
`quantization_config.json`, 2026-11-15 (S1, S7; `A-sub3bpw-quants.md` §2).

| quant (date) | quantizer | body and head GB | K classes of the 400 body linears | head | binds today |
|---|---|---:|---|---|---|
| `SC_3.00bpw_H4` of S1, the source of the 16 GB artifact | 1.4.2 | 9.820 | K2 x34, K3 x297, K4 x65, K5 x4 | 4 bit | yes |
| `0xBakeer/...-EXL3-2.5` (2026-10-08) | 1.6.0 | 8.299 | K2 x192, K3 x208 | 4 bit | yes |
| `0xBakeer/...-EXL3-3.0` (2026-10-08) | 1.6.0 | 9.821 | K2 x57, K3 x268, K4 x75 | 4 bit | yes |
| `BHCC2025/...-EXL3-3.0bpw-mix` (2026-11-09) | 1.6.1 | 9.980 | K2 x32, K3 x305, K4 x63 | 5 bit | body yes, head no |
| `2.50bpw` of S1 | 1.4.2 | 8.616 | K2 x137, K3 x262, K4 x1 | 6 bit | body yes, head no |
| `SC_2.20bpw_H3` of S1 | 1.4.2 | 7.229 | K1 x3, K2 x278, K3 x103, K4 x16 | 3 bit | no (K1) |
| `nalimc/...-SC_2.60bpw_H4` (2026-10-01) | 1.5.3 | 8.604 | K1 x4, K2 x134, K3 x225, K4 x31, K5 x6 | 4 bit | no (K1) |
| `Derpyhue/...-2.65bpw-SC` to `-2.80bpw-SC` (2026-10-05 to 08) | 1.5.4 | 8.757 to 9.212 | K1.5 x2 to x4, K2.5 and K3.5 in the tens to hundreds | 4 bit | no (fractional) |
| `yeasah/...` at `2.50bpw-H5` (2026-11-09) | 1.6.4 | 8.459 | K2 x90, K2.5 x191, K3 x83, K3.5 x16, K4 x20 | 5 bit | no (fractional) |
| `GestaltLabs/...-EXL3-11.5GB` (2026-09-05) | 1.4.6 | 9.419 | K1 x2, K6 x8 among K2 to K5 | 4 bit | no (K1, K6) |
| `SCP_2.00bpw`, `SCP_2.20bpw`, `SCP_3.00bpw` of S1 (2026-11-14) | 1.6.6, 1.6.7 | 6.621, 7.228, 9.821 | K1.5, K2.5 and K3.5 classes | 3 and 4 bit | no (fractional) |

- **Observed.** Between 2.2 and 3.0 bits, the only public quant that binds with the head at 4 bits is the third-party
  2.5 (`0xBakeer`, 8.299 GB; K2 and K3 in halves). The branches turboderp added on 2026-11-14 (`SCP_*`, "fractional-K +
  YAQA") need fractional K; the old `SC_*` branches below 3.0 need K1. The fractional classes are a different trellis
  width (24, 32, 40, 48 int16 words per 16x16 tile at K 1.5, 2, 2.5, 3), not a relabel.
- **Observed.** The public optimizer (`sc_optimize`, S2) writes "a YAML recipe mapping every quantizable tensor to an
  integer bitrate", and `-mink 2` forbids K1. Its recipe for the third-party 2.5 is on that repository
  (`calibration/recipe_2.5.yaml`: target 2.5, achieved 2.4996, predicted KLD 0.074612, head 4 bits, `k_proj`, `q_proj`
  and `v_proj` tied, `gate_proj` and `up_proj` tied). No public quant made this way exists at 2.4 to 2.9 bits.
- **Observed**, from the files of `0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-2.5` at revision `a7fef26e`, listed and read on
  2026-11-16 (copies of its metadata in `.scratch/630/632/hf-meta/`): two safetensors shards of 11.98 GB in all,
  configuration, tokenizer, card, licence (Apache 2.0), quality output and the recipe, no executable. Its
  configuration: quantizer 1.6.0, `out_scales` "auto" with `suh` and `svh` stored on all 401 matrices, codebook
  `mul1`, `head_bits` 4, `mtp_bits` 4, calibration 250 rows of 2,048 columns. K2 x192, K3 x208 in the body (attention
  at K3 in 12 of 16 layers, the GDN `out_proj` at K3 in 42 of 48, the GDN input projections at K2 in 38 of 48, the MLP
  about half and half), and a 4-bit head.
- **Observed**, the instrument-free reading of that quant: wikitext-2 KL 0.1112 and top-1 86.3% against BF16 (3.0 of
  the same author: 0.0621, 89.8%), by the author, BF16 perplexity 6.849 (S7). The card calls raw web text
  the publisher's "vestigial signal" for this model, so the ratio matters, not the level.
- **Inference.** Its allocation is a recipe of the same family as the 16 GB artifact's source, so its quality at 2.5 is
  what the ladder below predicts for a public integer-K allocation; whether ignis's own allocation does better is what
  the two candidate recipes of #630 are for.

**Measured quality of the same ladder, one table per harness** (S1, S3, S4, S5, S7; `A-sub3bpw-quants.md` §3). The
first column's steps are the only comparison inside a table.

| harness | what | 3.0 | 2.5 | 2.2 | 2.0 | 1.8 |
|---|---|---:|---:|---:|---:|---:|
| the publisher's suite on its new 347K-token trace (S1, chart dated 2026-11-14, `printed`) | mean KLD of the fractional ladder `SCP` | 0.0291 | not published | 0.0954 | 0.1355 | 0.2344 (1.75) |
| the challenge leaderboard (S3, S4, 2026-11-03) | mean KLD / top-1 of turboderp's tensors, whole-model bits | 0.0349 / 93.06% | 0.0778 / 89.96% | 0.1025 / 88.05% | 0.1476 to 0.1540 / 85.8% to 86.0% | 0.2584 / 80.99% |
| the same, a mixed integer-K re-quant (S4) | mean KLD / top-1 | 0.0306 / 93.58% | 0.0666 / 90.24% | | 0.1644 / 84.91% | |
| a multi-slice harness on 200 recorded agent sessions (S5, 2026-11-09, composite 0.7 swe + 0.3 wild) | excess KLD of turboderp's tensors | 0.0196 | 0.0519 | 0.0643 | 0.0835 to 0.0928 | 0.1755 |
| the same, S5's own ladder | excess KLD | 0.0136 | 0.0306 | | | |
| wikitext-2 (S7, the third-party 2.5 and 3.0) | KL / top-1 | 0.0621 / 89.8% | 0.1112 / 86.3% | | | |

- **Observed.** On five independent harnesses the cost of each quarter of a bit between 3.0 and 2.2 is a factor of
  about 1.45 to 1.5 in KLD (1.34 on wikitext-2) and 1.6 to 1.7 points of top-1; above 3.0 it is about 1.35 to 1.4 and
  0.8 points. Below 2.0 the step is 1.9 to 2.9 times. Two public 2.5s on the same multi-slice harness read 0.0519 and
  0.0306, 1.7 times apart: the allocation is worth about as much as a quarter of a bit.
- **Observed**, the instrument of this repository: the 16 GB artifact reads KLD 0.0403 against BF16 at a 1,024-token
  chunk on set A, its registered limit for a re-quantization is 0.0484, and the NVFP4 artifact the RTX 5090 serves
  reads 0.0492 with a limit of 0.0590 (the 16 GB quality gates). The publisher's chart reads the old 3.00 at 0.0257 and
  this repository's harness reads the same tensors at 0.0340: harnesses do not mix, only ratios do.
- **Observed.** The step is not a cliff in a distribution metric: the sharp bend is between 2.0 and 1.8 bits, where K1
  matrices go from 1.0% of the parameters (`SC_2.20`) to 4.8% (`SC_2.00`) and 28.6% (`SC_1.80`). The optimizer's notes
  give the mechanism: the KLD-vs-noise exponent was "measured ~2.35 for the K2->K1 octave on Qwen3.8-27B".

### The agent-task table

One public harness runs agent tasks on this model below 3 bits. It is the model page of a publisher of quants in
another format, published 2026-11-11 (S6), so the trellis rows are its competitor's: the harness ran them, through a
runtime its page does not name, beside its own files. **Every value is read off its charts (chart ±0.5); only BF16
is printed.** Terminal-Bench 2.1 with mini-swe-agent 2.4.6 and native tool calls, 82 tasks (7 excluded), 3 attempts
each, a limit of 150 steps, no internet for the agent; DeepSWE v1.1 on 113 tasks, at least 3 passes each, graded by
the tasks' own tests; LiveCodeBench v6 at 128K output tokens; temperature 1.0, top-p 0.95, top-k 20, reasoning at its
highest setting, 128K to 256K of context. The x axis is "GB without the token embedding", which is this
repository's body and head. The trellis points sit at 7.25 and 9.85 GB, which are the sizes of `SC_2.20bpw_H3` and
`SC_3.00bpw_H4` (the page does not name the branches).

| 27B build | GB without the token embedding | Terminal-Bench 2.1 | DeepSWE v1.1 | LiveCodeBench v6 |
|---|---:|---:|---:|---:|
| BF16 (printed) | | 59.15 | 40.40 | 88.62 |
| the publisher's 4-bit point | 16.6 | 60.2 | 38.4 | 89.8 |
| its 3-bit point | 11.15 | 59.7 | 34.5 | 89.0 |
| trellis, the 16 GB artifact's source recipe | 9.85 | 55.3 | 31.2 | 88.7 |
| the other format, dynamic 2-bit, 3.1 | 9.2 | 53.0 | 29.2 | 88.5 |
| a third publisher's learned-scalar 3.00 bpw (S9) | 9.7 | 51.2 | 25.6 | 87.1 |
| a third publisher's learned-scalar 2.75 bpw (S9) | 8.85 | 42.7 | 24.8 | 83.2 |
| the other format, dynamic 2-bit, 3.1 | 8.0 | 44.6 | 21.7 | 84.7 |
| a third publisher's learned-scalar 2.50 bpw (S9) | 8.15 | 37.8 | 16.8 | 84.0 |
| the other format, dynamic 2-bit, 3.1 | 7.3 | 41.0 | 12.8 | 84.5 |
| trellis, the public 2.2-bit branch | 7.25 | 43.1 | 9.8 | 83.3 |

The two reports read the same chart twice and agree within 0.1 point (DeepSWE 12.8 or 12.9 at 7.3 GB, 24.7 or 24.8 at
8.85 GB, 34.4 or 34.5 at 11.15 GB); the first value is kept. The chart prints a recovery figure that the readings
reproduce: trellis 2.20 mean(83.3/88.62, 43.1/59.15, 9.8/40.40) is 63.7%, trellis 3.00 is 90.3%.

- **Observed.** The loss is ordered by the length of the task. At 7.25 GB the trellis body keeps 94% of BF16 on
  single-turn code, 73% on terminal tasks and 24% on long repository tasks; at 9.85 GB, 100%, 93.5% and 77%. The
  dynamic 2-bit line keeps 90% and 72% of BF16 on terminal and long tasks at 9.2 GB, 75% and 54% at 8.0 GB. The step on
  the long tasks sits between 8.0 and 9.2 GB, and that publisher states the same break for its own files ("a sharp drop
  off from UD-Q2_K_XL to UD-IQ2_S ... tool calling and non thinking modes break down").
- **Observed.** Allocation alone moved the same quantizer at about 8 GB from the previous generation to the 3.1 one:
  Terminal-Bench 41.5 to 44.6, DeepSWE 12.8 to 21.7, LiveCodeBench 76.6 to 84.7 (recipe not disclosed).
- **Observed.** The harness's noise: above 11 GB its line is not monotone (Terminal-Bench 59.7, 56.7, 55.6, 57.8, 60.2
  from 11.15 to 16.6 GB; DeepSWE 34.5, 37.6, 36.3, 33.6, 38.4), so a single point carries about 2 to 3 points. On
  single-turn code, the 5.7-point step from 2.50 to 2.75 bits on a learned-scalar publisher's own card (76.57 to 82.29,
  one trial) is not reproduced in this harness (84.0 to 83.2): a quarter of a bit on LiveCodeBench is inside one trial's
  noise.
- **Observed.** At equal size the second format's 9.2 GB file and the trellis 9.85 GB body score the same within that
  noise here, although the trellis author's KLD chart puts the earlier file of the same name at more than twice the
  KLD (0.0587 against 0.0257).
- **Inference.** The agentic step between a 2.2-class and a 2.5-class body, that is between about 7.3 and 8.3 GB, is
  where the long tasks give way. A body of 9.2 GB sits on the right side of it on this evidence, and the ceiling that
  #630 takes is motivated by this table. It is motivation, not a measurement of the artifact ignis will build: the
  trellis curve has two points (7.25 and 9.85), so every value between them is an interpolation and the shape between
  is unknown (the second format's curve is concave in the band, which would make the trellis step smaller than the
  chord on DeepSWE).

For the byte budgets of the spec (at 65,536 tokens of context, a 12 GB card's 12.82 GB usable, about 1 GB for a
desktop and headroom), interpolating linearly inside the one harness, derived by the research agent: the two-lane
points (7.95 GB under DFlash2, 8.68 GB under MTP) cost about 4 to 6 Terminal-Bench points and 5 to 7 DeepSWE points
against the one-lane points (8.8 and 9.55 GB), out of 59 and 40. The chord between the two trellis points is 4.7
Terminal-Bench and 8.2 DeepSWE points a GB; the dynamic line between 8.0 and 9.2 GB is 7.0 and 6.3 (`inference`, one
harness).

Other agent-task and process evidence, each its own harness (`C-task-level-evidence.md`):

| source | build | result | what it shows |
|---|---|---|---|
| an independent blog run (S11, 2026-08-26): llama.cpp of 2026-08-16, Terminal-Bench 2.1 on 89 tasks, 3 h timeout, one run | the dynamic 2-bit file of 10.7 GB, KLD 0.079, top-1 87.3% | 57/89 (64.0%, CI 54 to 73) against 63/89 (70.8%) for BF16; 17 failed and 15 timed out against 18 and 8 | at KLD 0.08 the pass count falls inside its interval while the timeouts nearly double |
| the same, tasks both solved | the same | output tokens 1.27x BF16 (CI 1.06 to 1.81), 1.17x on GPQA, 1.40x on IFBench | a harder quant spends more tokens |
| the same | single-turn IFBench, strict | 240/300 against 245/300 | short instruction following does not move |
| a coder fine-tune of the 27B on 3090s (S8, 2026-11-02): own opencode tasks, 10 x 3 | trellis 2.25 bits (fractional) against 3.50 bits | agent tasks 25/30 against 30/30; tool calls 29/30 against 30/30; loops on long answers 8/30 against 0/30 at temperature 1.0 | at 2.25 bits tasks, loops and tool calls already cost; one model, n = 30 |
| a sweep including BFCL v4 multi-turn and ACEBench (S12, GGUF only, per-benchmark scores unpublished) | a quarter to a third of a bit starting near 2.5 | +3.5 points of BF16 for the best allocation, +7.2, +9.4 and +13.1 for the others | the method changes the cost of the same step about fourfold |
| MTP acceptance of the model's own head on a GGUF target (S9, llama.cpp speed bench, 3 drafts, 5 samples a bar; chart ±0.3) | 2.50, 2.75, 3.00, 3.50 bits | coding 58.5, 58.7, 55.9, 58.0%; reasoning 53.0, 51.8, 50.7, 53.3% | no fall in acceptance from 3.5 to 2.5 bits |

### The alternatives to the 27B (what was checked, 2026-11-15)

- **No smaller Qwen3.8.** The Qwen organisation's Hub repositories named Qwen3.8 are the 27B, its FP8, Flash-Next, its
  FP8, and the 2.4T-A95B with its FP8; nothing was created after 2026-10-20 and the newest repository is an image model
  of 2026-10-09 (S10, the Hub listing). Requests for a 9B stay open (27B repository discussions #223, #237, #238)
  without a Qwen reply. The only Qwen4 artifact found is a hosted preview that retires on 2026-11-25; a "Qwen4 27B"
  for local use is named by secondary sources only, with no date, licence or architecture.
- **Flash-Next is the one candidate that beats the 27B per byte of VRAM on agent suites, at a different cost.** In the
  harness of the agent-task table, its builds of about 66 to 68 GB score Terminal-Bench 57 to 70 and DeepSWE 35 to 54
  against 55.3 and 31.2 for the 27B at 9.85 GB (chart ±0.5; its quantizations differ a lot at one size: 35.4 and 54.0
  on DeepSWE at about 66 GB). It asks 34 GB of experts in host memory plus about 10 GB, with a 28.8 GB n-gram table on
  disk (48 GB of host memory for the 2.4-bit build, 64 GB above), a new artifact, and a licence that restricts
  commercial model-serving. Independent runs put it below the 27B on BFCL v4 multi-turn (0.523 against 0.570, one
  author, one box) and level on LiveCodeBench and GPQA. A public engine runs it on a 12 GB card (an RTX 5070 at 76 to
  93 tok/s decode up to 64K of context under its engine's own measurement), which is not ignis's. It is a later
  phase, not this class.
- **Fine-tunes of the 27B buy no quality per byte.** Three that cut reasoning tokens by 36 to 47% at the highest
  reasoning setting match the base within noise on one frozen suite (average 91.1 to 91.4 and 89.8 against 90.2); one
  claims a Terminal-Bench gain on its own harness (72.13 against 69.21) under a licence that caps free use at a
  revenue. None has low-bit task evidence.
- **Other architectures, reference only, not loadable.** The vendor cards of two 2026-08 releases claim Terminal-Bench
  2.1 46.2 for a 9B (7.6 to 9.8 GB at 6 to 8 bits) and 67.8 for a 35B-A3B with its experts in host memory, no
  independent reproduction found. Across harnesses, so only a pointer: a 27B at 2.4 to 2.65 bits keeps about 0.65 to
  0.8 of its Terminal-Bench, and its margin over a well-trained 9B is small on public data (`B-alternatives.md` §9,
  `inference`).

### The slope on our own instrument

Measured on 2026-11-16 in one session on the RTX 5090, integration tree `15d001e5` (engine code = `main` `0840b74c`),
at the 16 GB artifact's serving shape: 1,024-token chunk, hq-e8-2b, exception weights bf16. Records:
`.scratch/630/632/comment-numbers.md`, `kld/compare-632.txt`.

- **Observed, the 2.5-bit quant.** The third-party 2.5 converted with the named recipe `tandem25` (#631's converter,
  commit `a92871d1`): 401 trellis matrices (K2 x192, K3 x208, head K4), text scope 8,303,517,696 bytes, 2.4996 bits
  per weight; reconstruction errors measured against the BF16 checkpoint, VERIFY PASS; 11,858,786,564 bytes in all. It
  loads under `rtx-16gb` in a 16 GiB budget: 17,179,624,364 bytes allocated, context 262,144, nothing adapted,
  674,304 tokens of KV pool (the 16 GB artifact: 509,248).
- **Observed, KLD against BF16, set A** (16 windows of 4,096 tokens; the scorer is `span_logits` at the serving shape
  against `H:/kld-ref/base.pt`; window-bootstrap 95% interval, 20,000 resamples):

  | | KLD | 95% CI | top-1 | ppl | code | prose | chat |
  |---|---:|---|---:|---:|---:|---:|---:|
  | 16 GB artifact (3.00 bits, 9.82 GB) | 0.04049 | [0.0346, 0.0463] | 93.40% | 2.739 | 0.0385 | 0.0499 | 0.0362 |
  | third-party 2.5 (2.50 bits, 8.30 GB) | 0.06704 | [0.0564, 0.0780] | 91.00% | 2.790 | 0.0690 | 0.0949 | 0.0465 |

  BF16 ppl is 2.690. The ratio 2.5 / 3.0 is **1.656** (paired 95% interval [1.485, 1.810]) and the 2.5 reads above the
  3.0 in all 16 windows; by kind 1.79 times on code, 1.90 on prose, 1.28 on chat.
- **Observed, the instrument's reproduction.** The 16 GB artifact reads 0.04049 against its registered 0.04034 (+0.4%:
  today's engine against the reference of 2026-11-03; top-1 93.40% against 93.42%, ppl 2.7386 against 2.738); the
  limit of 1.2 times today's reading is 0.0486. No in-process reproduction was made: the difference is small and the
  two arms were read in the same session.
- **Observed, the loop set** (#340's 32 requests, 2 lanes, loop guard off, through `.scratch/630/gpu/served630.py`; a
  loop is a 6-gram repeated 25 times or a zlib window below 0.20): the 2.5 loops on **11 of 32**, the 16 GB artifact on
  **8 of 32** (one-sided Fisher exact p = 0.29, no difference at this size). Finish reasons, 2.5 against 16 GB: tool
  calls 15 against 24, stop 7 against 1, length 10 against 7; forced closes of the thinking 13 and 13.
- **Inference, the reading of the half bit.** The curve between 3.0 and 2.5 is a factor of 1.66 in KLD for half a bit,
  gentler than the public ladders' about 1.5 for every quarter of a bit (that is 2.25 for half a bit), but the 2.5
  does not hold close to the 3.0. Log-linear between the two points, a recipe at 2.79 bits (the 9.2 GB ceiling) would
  read about 0.049, at the limit of 0.0486; the allocation optimizer's own model predicts 1.17 times the 16 GB
  artifact's KLD for candidate A, about 0.047. Both are interpolations and a prediction, not readings of a recipe at
  9.2 GB. Neither of the spec's two cases (the 2.5 close to the 3.0, or 9.2 GB already low) is met, so the candidates
  are built at 9.2 GB (9,200,000,000 bytes of text scope as the converter counts it).
- **Not measured here:** agent tasks, tool-call validity and the drafter's acceptance on the 2.5; the loop rates rest
  on 32 requests each.

### Candidate B's recipe: a public GGUF allocation translated to K

#634 builds the second candidate of #630 from the per-tensor allocation of the dynamic 2-bit GGUF file at the 9.2 GB
row of the agent-task table, read from its header and translated to integer K by a stated rule. The tool is
`tools/27b-16gb-converter/gguf_recipe.py` (`header`, then `recipe`), the recipe `recipes/ig28b.json`. Read on
2026-11-16; the parsed table is in `.scratch/630/634/ud-q2_k_xl.table.json`, and `header` reads it again.

- **The source.** `unsloth/Qwen3.8-27B-GGUF`, file `Qwen3.8-27B-UD-Q2_K_XL.gguf`, at the repository's revision
  `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` (the file's last commit is `acb953b1`, "New UD3.1 quants", 2026-11-11),
  one file of 9,948,666,784 bytes, LFS oid `18f1187f38e7994b05c08cb45c674b90b7f17523687c83a4614fe493f7ea739d`. The
  repository's files were listed before anything was read. Its 400 body matrices and its output head hold 9,169,463,296
  bytes, which is the table's 9.2 GB; every tensor but the token embedding (the MTP layer, norms and GDN small set
  included) holds 9,520,492,544.
- **The read.** The GGUF v3 header by HTTP range requests: 50 metadata keys and 866 tensor infos in 10,996,621 bytes,
  read as 11 ranges of 1 MiB (11,534,336 bytes). The last range runs 537,715 bytes past the header, into the alignment
  and the first bytes of the head's data, which the parser does not read. Each tensor's size from the data offsets is
  its type's block size times its blocks, for all 866 (no mismatch): the type table below is the file's.
- **The bits of a type** are its bits per weight with its block scales: a block's bytes times 8 over its weights, from
  the `block_*` structs that llama.cpp's `ggml/src/ggml-common.h` pins with static asserts (tag v1.0.0, commit
  `b1b770d9`; the type ids from `ggml/include/ggml.h`).

  | type | bytes / weights a block | bits | | type | bytes / weights a block | bits |
  |---|---:|---:|---|---|---:|---:|
  | IQ1_S | 50 / 256 | 1.5625 | | IQ3_XXS | 98 / 256 | 3.0625 |
  | IQ2_XXS | 66 / 256 | 2.0625 | | IQ3_S, Q3_K | 110 / 256 | 3.4375 |
  | IQ2_XS | 74 / 256 | 2.3125 | | IQ4_XS | 136 / 256 | 4.25 |
  | IQ2_S | 82 / 256 | 2.5625 | | Q4_K | 144 / 256 | 4.5 |
  | Q2_K | 84 / 256 | 2.625 | | Q5_K | 176 / 256 | 5.5 |

- **The rule** (`gguf_recipe.fit`):
  1. Each matrix's K is the integer nearest its type's bits plus one shift for the whole body, held to 2 to 5. The head
     is at 4 whatever its type: the load binds it there, so it is outside the order.
  2. The shift is the largest at which the text scope's device weights, counted as the converter counts them (every
     trellis record, every other `text/` object but the token embedding and the draft head), are at or under the
     ceiling. K changes only where a class of bits crosses a half, so the shift is one of those points, and a matrix
     with more bits than another never gets a lower K.
  3. The class that would cross next does not fit whole: its matrices are promoted one by one in rank order, and the
     promotion stops at the first that does not fit. Its matrices tie on bits, so the rank is the source's allocation
     one grain coarser: the mean bits of the matrix's layer, weighted by its weights, higher first; then the key. Two
     classes whose bits differ by a whole number cross together and share that rank.
- **Where the rule chose.**
  - The 17 IQ1_S matrices (1.56 bits; MLP gate, up and down of layers 0, 1, 10, 12 to 15 and 17) are raised to K2, more
    than the source gives them. The one Q5_K (5.5 bits, layer 63's `v_proj`) is held to K5.
  - The shift landed at 0: every matrix at the integer nearest its bits. The 21 Q4_K matrices sit on a half (4.5) and go
    to K5 at any shift above 0, so they are K5 by the fit, not by a convention for rounding a half. The head is Q4_K
    too and stays at 4.
  - The next class is IQ3_S and Q3_K, 41 matrices at 3.4375 bits (shift 1/16), 285 MB whole against 103,496,704 bytes
    left under the ceiling (the recipe at the shift alone counts 9,096,503,296). Ten were promoted to K4: layer 60's
    `in_proj_qkv`, 59's `o_proj`, the gate and up of layers 56 and 57, the gate, up and down of 62 and 61's `out_proj`.
    The first left out, 61's `down_proj`, needed 11,141,120 bytes and 11,090,944 were left; they stay unspent.
  - The names map one to one. The GGUF file splits the body as exllamav3 does (`attn_qkv`/`attn_gate`/`ssm_out`,
    `attn_q`/`attn_k`/`attn_v`/`attn_output`, `ffn_gate`/`ffn_up`/`ffn_down`, `output` for the head): nothing is fused
    on one side and split on the other. The GDN `ssm_alpha` and `ssm_beta` (Q8_0 there) are copied objects here, not
    matrices of a recipe; the file's layer 64 is its MTP layer and is not read.
  - Nothing is tied. The source gives gate and up, q, k and v, and a GDN layer's `in_proj_qkv` and `in_proj_z` their
    own types, and the recipe keeps them: gate and up differ in 5 layers (4, 22, 29, 44, 48), k and v in 12 of the 16
    GQA layers, q, k and v are not all one K in any. exllamav3's optimizer ties q/k/v and gate/up, and its own runtime
    fuses gate and up only at equal K; its conversion from a recipe takes each matrix's K, and the trellis ops take each
    matrix at its own K (their op tests run a gate/up pair of different K, decode and dense prefill). `sc30` never has
    gate and up, or k and v, at different K: no artifact has loaded such a pair yet.
- **The recipe `ig28b`**: K2 x106, K3 x226, K4 x47 (the head among them), K5 x22 over 401 matrices; the body at 2.7907
  bits per weight (exllamav3's `bits` 2.79); the text scope 9,188,909,056 bytes of the 9,200,000,000 ceiling. Its
  calibration trace, quant configuration pins and exllamav3 job are `sc30`'s; `local_quant.py run --recipe ig28b
  --check` passes on the BF16 checkpoint and that trace (2026-11-16).

By family, the source's types, its mean bits weighted by weights, and the K of `ig28b` and of `sc30` (the 16 GB
artifact's) with their means:

| family | n | the source's types | source bits | `ig28b` K | mean | `sc30` K | mean |
|---|---:|---|---:|---|---:|---|---:|
| GDN `in_proj_qkv` | 48 | IQ3_XXS 20, IQ2_S 9, IQ2_XXS 7, IQ2_XS 5, IQ3_S 4, Q2_K 3 | 2.749 | K2 12, K3 35, K4 1 | 2.771 | K2 4, K3 42, K4 2 | 2.958 |
| GDN `in_proj_z` | 48 | IQ3_XXS 26, IQ2_XXS 7, IQ2_S 7, IQ2_XS 3, Q2_K 3, IQ4_XS 1, IQ3_S 1 | 2.802 | K2 10, K3 37, K4 1 | 2.812 | K2 3, K3 40, K4 5 | 3.042 |
| GDN `out_proj` | 48 | IQ3_XXS 22, IQ4_XS 8, IQ3_S 7, IQ2_S 5, IQ2_XS 4, Q4_K 1, Q3_K 1 | 3.238 | K2 4, K3 34, K4 9, K5 1 | 3.146 | K3 40, K4 8 | 3.167 |
| GQA `q_proj` | 16 | Q2_K 8, IQ3_XXS 4, IQ2_XXS 2, IQ2_XS 1, IQ2_S 1 | 2.641 | K2 3, K3 13 | 2.812 | K2 6, K3 10 | 2.625 |
| GQA `k_proj` | 16 | IQ4_XS 6, IQ3_S 5, Q4_K 4, IQ3_XXS 1 | 3.984 | K3 6, K4 6, K5 4 | 3.875 | K4 14, K5 2 | 4.125 |
| GQA `v_proj` | 16 | Q4_K 15, Q5_K 1 | 4.562 | K5 16 | 5.000 | K4 14, K5 2 | 4.125 |
| GQA `o_proj` | 16 | IQ4_XS 6, IQ3_S 5, IQ3_XXS 1, IQ2_XXS 1, IQ2_XS 1, Q4_K 1, Q2_K 1 | 3.578 | K2 2, K3 6, K4 7, K5 1 | 3.438 | K3 14, K4 2 | 3.125 |
| MLP `gate_proj` | 64 | IQ3_XXS 17, IQ2_XXS 13, IQ2_XS 10, IQ1_S 8, IQ2_S 8, IQ3_S 4, IQ4_XS 4 | 2.590 | K2 31, K3 26, K4 7 | 2.625 | K2 9, K3 49, K4 6 | 2.953 |
| MLP `up_proj` | 64 | IQ3_XXS 19, IQ2_XXS 13, IQ2_S 11, IQ2_XS 7, IQ1_S 6, IQ3_S 4, IQ4_XS 4 | 2.648 | K2 26, K3 31, K4 7 | 2.703 | K2 9, K3 49, K4 6 | 2.953 |
| MLP `down_proj` | 64 | IQ3_XXS 15, IQ2_S 13, IQ2_XXS 12, IQ3_S 9, IQ4_XS 7, IQ2_XS 3, IQ1_S 3, Q3_K 1, Q2_K 1 | 2.850 | K2 18, K3 38, K4 8 | 2.844 | K2 3, K3 53, K4 8 | 3.078 |
| output head | 1 | Q4_K | 4.5 | K4 | 4 | K4 | 4 |
| body | 400 | | 2.769 | | 2.791 | | 3.000 |

- **Observed.** Against `sc30`, `ig28b` keeps the K of 249 matrices, raises 49 and lowers 103. It moves bits from the
  MLP and the GDN input projections (K2 on 75 MLP matrices against 21) to the GQA layers' `v_proj` (all 16 at K5),
  `o_proj` and q, and from the first layers to the last: the mean K of layers 0 to 15 is 2.28 against 2.91, of 16 to
  31 2.64 against 2.95, of 32 to 47 2.86 against 2.91, of 48 to 63 3.39 against 3.23. The source's own allocation has
  that shape: 12 of its 17 IQ1_S matrices are the MLP of layers 12 to 17, and layers 56 to 63 are at IQ3 and IQ4
  types.
- **Inference.** At 0.21 bits apart the two differ less in their totals than in where the bits go, by family and by
  depth; which allocation holds on ignis's instrument is the gate's reading. Nothing here says the source's per-type
  quality carries over to a trellis K: the rule translates its bits, not its error.

### The gate at the class's shape (2026-11-16)

#633 and #634 built the two candidates at the 9.2 GB ceiling, #635 measured them against the 16 GB artifact
(`trellis-sc30`) and #636 measured the prefill chunk. All of it in one session on the RTX 5090, with the engine code of
`6a763649`, at the class's shape: `--profile rtx-12gb --vram-budget-bytes 12884901888` (one lane, MTP at window 7,
512-token chunk, hq-e8-2b, exception weights bf16, loop guard off, vision on host); the counterpart ran at the same
shape. Records: [`.scratch/630/635/`](../../.scratch/630/635/) (`kld/compare-635.txt`, `decode-compare.txt`,
`loops-compare.txt`, `plan-table.txt`, `chunk-table.txt`, `real-card-context.txt`, `replay/`, `served/`) and
`.scratch/630/orchestra/decisions.md`.

**The two recipes.**

- **A, `ig28a`** (`tools/27b-16gb-converter/recipes/ig28a.json`, #633): exllamav3's own allocation optimizer,
  `sc_optimize.py` (git blob `eeb1cbee`, the same file from 1.5.3 to 1.6.7), on the 16 GB artifact's source
  publisher's tables at `3f8a2d64` (`noise_attrib.json`, `rfn_2.00bpw.json`): `sc_optimize.py -m noise_attrib.json -rr
  rfn_2.00bpw.json -b 2.7942 -hb 4 -al 2.0 -mink 2 -maxk 5 -tie "k_proj+v_proj,gate_proj+up_proj"`. The optimizer
  targets bits per weight over the body, so the byte ceiling was met by iterating `-b` (2.7947 goes over). It does not
  reproduce the published 3.00 recipe (123 of 400 matrices differ; that one came from another script of the project).
  Text scope 9,199,394,816 bytes of the 9,200,000,000, body 2.7942 bits; K2 72, K3 287, K4 36 (the head among them),
  K5 6 over 401 matrices; against the 16 GB recipe 124 matrices differ (94 down, 30 up), 625,213,440 bytes smaller:
  the MLP and the GDN inputs lose bits, attention k/v and o gain. The optimizer's own model predicted 0.0505 against
  0.0430 for the 16 GB recipe (1.17 times). Converted with exllamav3 1.5.3 on the public calibration trace of the 16 GB
  artifact's revision `86b95530`: 2,602 s, 19.63 GiB VRAM allocated, 42.1 GiB host working set, `--no-checkpoints`.
- **B, `ig28b`** (#634): the GGUF allocation translated by the rule of the section above; the same conversion
  (2,543 s). Text scope 9,188,909,056 bytes, 2.7907 bits.
- Packed with the DFlash2 module and the MTP head: A 12,754,663,684 bytes, B 12,744,177,924, both VERIFY PASS. A's
  repack as `trellis-ig28` (a complete conversion, reconstruction errors against BF16 measured, VERIFY PASS) reads
  the same 0.04308: the same bytes. It packs to the same content hash twice from the same quant and recipe: a second
  complete conversion, under the public labels `Qwen/Qwen3.8-27B` and `exl3-ig28-local`, gave the same file,
  `qwen3_8_27b_trellis_ig28-v2.ninfer`, sha256 `fab06bea49ce114da99ca69588f00b10fb5081c564f9ce60ecbef8b3dbf6d6a5`,
  the bytes `gpillon/Qwen3.8-27B-trellis-2.8bpw-ignis` serves at revision `15333a0829f055e7ee25d6d5ef84fd8d521518c6`
  beside that conversion's record (309,831 bytes).

**Observed, KLD against BF16, set A** (16 windows x 4,096; `span_logits` at chunk 512; window-bootstrap 95% CI, 20,000
resamples; BF16 ppl 2.690):

| | text scope | KLD | 95% CI | ratio to 16 GB | top-1 | ppl | code | prose | chat |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| 16 GB artifact | 9.82 GB | 0.04177 | [0.0358, 0.0477] | 1 | 93.30% | 2.742 | 0.0400 | 0.0520 | 0.0367 |
| A, `ig28a` | 9.20 GB | 0.04308 | [0.0368, 0.0496] | 1.032 [0.993, 1.070], lower in 3/16 windows | 93.35% | 2.728 | 0.0418 | 0.0534 | 0.0375 |
| B, `ig28b` | 9.19 GB | 0.05084 | [0.0430, 0.0589] | 1.217 [1.179, 1.256], lower in 0/16 | 92.69% | 2.760 | 0.0473 | 0.0642 | 0.0455 |
| `trellis-ig28` on the RTX 30 and 40 routes (2026-11-17) | 9.20 GB | 0.04353 | [0.0374, 0.0495] | 1.010 [0.974, 1.035] to A's row, lower in 2/16 | 93.23% | 2.734 | 0.0429 | 0.0555 | 0.0362 |

The limit a re-quantization registers, 1.2 times the counterpart's reading in the same session, is 0.05012: A is under
it, B above. The last row is `trellis-ig28` on the RTX 5090 running the prefill routes an RTX 30 or 40 card runs unset
(`IGNIS_MODEL_PREFILL_LINEAR=int8`, `IGNIS_MODEL_PREFILL_ATTENTION=int8`), as #513 read Swift 1.5's build (1.0148):
its ratio is to A's row of 2026-11-16, under the 1.05 a route's numerics registers (ADR 0082), top-1 0.12 points
under A's (`kld/compare-ig28-8x.txt`). (The 16 GB artifact reads 0.04177 here at chunk 512 and 0.04049 at its own
1,024 in the slope above: the readings of one session at one shape are compared with each other.)

- **Observed, the estimate against the read.** The spec's estimate was about 0.056 and the slope's interpolation about
  0.049; the optimizer predicted 1.17 times the 16 GB artifact for A (about 0.047). A reads **0.0431**, 1.03 times.
  The estimate was high by about 0.013, and the optimizer's relative prediction over-called A's loss by 14 points.
- **Observed, the same-K comparison of the quantizer.** `ig28` was converted by exllamav3 1.5.3, the 16 GB artifact by
  the version its publisher used, so a 1.03 could be the recipe's or the quantizer's. On the 277 matrices that have the
  same K in both (reconstruction errors against the BF16 checkpoint, each artifact's own), the ratio of the errors has
  median 0.9993 (10th percentile 0.997, 90th 1.001): the quantizer version moves nothing, and the 1.03 is the
  recipe's.

**Observed, the drafter on against off** (A; greedy, 512 tokens, one lane; launches MTP-1, off-1, off-2, then
DFlash2-1, DFlash2-2, the counterpart and #636's, MTP-2 re-run last after a script error left its first launch with no
decode cell, a departure from ABBA; pooled tok/s; the verdict is on above off by more than the launch-to-launch spread;
`decode-compare.txt`):

| context | MTP on, cold / reuse | off (window 0), cold / reuse | ratio | DFlash2, cold / reuse | ratio |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 293.7 / 293.6 | 92.4 / 91.4 | 3.18x / 3.21x | 409.4 / 411.0 | 4.43x / 4.50x |
| 65,536 | 250.9 / 250.9 | 75.6 / 77.5 | 3.32x / 3.24x | 347.6 / 347.8 | 4.60x / 4.49x |
| 130,560 (the largest the load keeps) | 228.0 / 228.3 | 64.9 / 64.9 | 3.51x / 3.52x | 320.6 / 321.8 | 4.94x / 4.96x |

Every cell: on beats off. DFlash2 loads in the same 12 GiB with the same 131,072 context and nothing adapted; its KV
pool is 150,720 tokens against MTP's 214,144 (off: 280,320). The counterpart's own decode cells were not measured.

**Observed, the loop set** (#340's 32 requests, one lane): A **7/32**, the 16 GB artifact at the same shape **11/32**;
one-sided Fisher exact p (A loops more) = 0.92. Tool calls 27 against 26, length 5 against 6, forced closes 9 against
13. (The slope's loop sets ran two lanes and read 8/32 for the 16 GB artifact: not the same shape, not compared.)

**Observed, beside the gate and not binding.** The canary's four greedy prompts are sane and identical twice on both
candidates and on the counterpart. The replay of recorded agent turns under MTP at window 7 (#618's subset, jobs 0-21,
24 turns, 15,179 tokens; at the replay's fixed shape of the 16 GB artifact, two lanes and 1,024 chunk, since it
refuses one lane at 512 for a trellis body): **4.072 tokens a round** against the 16 GB artifact's 4.116 (-1.1%).

**Not run, and why.** The regression set (the contributed entailment cases, `decide_entailment_gpu`): its test file pins
the NVFP4 artifact by path. B's decode on/off, DFlash2 and loop legs: the KLD put B over the limit at equal bytes (1.22
times the counterpart against A's 1.03), so no leg could change the choice. B's 12 GiB load and canary did run
(allocated 12,884,646,144, pool 215,232 tokens, nothing adapted). The 16 GB counterpart's decode cells and any
agent-task check were not run.

**The 12 GiB plans** (`plan-table.txt`): A allocated 12,884,478,208 bytes, KV pool 214,144 tokens (3,346 pages),
context 131,072 with nothing adapted, weights 9,651,141,632; the 16 GB artifact 12,884,883,712 bytes and a pool of
150,336 tokens, also at 131,072.

**Observed, the prefill chunk** (#636; A in 12 GiB, one lane, MTP 7, vision on host; `ignis-bench ttft` cold, a nonce
per sample, three samples a cell, median; two launches a chunk, the 512 row's second being #635's `mtp-2`;
`chunk-table.txt`):

| chunk | first token, 8,192 prompt (ms) | first token, 65,536 prompt (ms) | decode at 8,192 (tok/s) | workspace | trellis workspace | KV pool tokens | context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 3,123 / 3,130 | 26,404 / 26,356 | 295.1 / 296.4 | 302,433,536 | 32,134,400 | 214,144 | 131,072 |
| 512 | 2,491 / 2,522 | 21,433 / 21,676 | 296.8 / 291.9 | 302,433,536 | 32,134,400 | 214,144 | 131,072 |
| 1,024 | 2,348 / 2,329 | 20,210 / 20,206 | 300.8 / 303.0 | 343,493,376 | 36,225,280 | 209,536 | 131,072 |

- 256 buys nothing with vision on: below 512 the workspace is set by the vision encoder's share (76,951,040 reserved),
  so the pool equals 512's, and the first token costs 25% more at 8,192 and 23% more at 65,536.
- 512 buys 4,608 tokens of pool over 1,024 (72 pages), not the spec's estimated 17,000, for 7% of the first token (7% at
  8,192; 1,350 ms or 6.7% at 65,536). In 12 GiB and in 12 GiB less 1 GiB the context is the same at both chunks (131,072
  and 98,304; at 11 GiB 1,024 leaves 1,561 pages against 98,304's 1,539). The chunk does not move decode.
- Inference: the 7% is below the roughly 10% the owner asks before trading pool for prefill speed.

**Inference, the context on a real card** (arithmetic on the plan's pinned lines, **not a measurement**: no 12 GB card
has run the load, and these are the RTX 5090's lines with its CUDA context and trellis workspace). A card's derived
budget is the memory free at the start less 512 MiB; the KV pool is 1,713 pages per GiB (about 0.627 MB a page), and
one 32,768-token step is 512 pages, 321 MB. With a card total of 11.9 GiB:

| case | free (GiB) | budget (GiB) | pages | context, 32,768 step | context, 16,384 step |
|---|---:|---:|---:|---:|---:|
| headless, nothing else on the card | 11.75 | 11.25 | 2,061 | 131,072 | 131,072 |
| light desktop, about 0.3 GB | 11.60 | 11.10 | 1,804 | 98,304 | 114,688 |
| desktop, about 0.5 GB | 11.43 | 10.93 | 1,513 | 65,536 | 81,920 |
| desktop, about 0.75 GB | 11.20 | 10.70 | 1,119 | 65,536 | 65,536 |
| desktop, about 1 GB | 10.97 | 10.47 | 725 | 32,768 | 32,768 |
| desktop, about 1.3 GB | 10.69 | 10.19 | 245 | refused | refused |

131,072 needs a budget of 11.245 GiB, 98,304 10.945 GiB and 65,536 10.646 GiB. The spec's "about 100,000 tokens with a
desktop of about 1 GB" does not hold on these lines: 1 GB of desktop leaves 32,768. The levers: a lower byte ceiling
(one step is about 0.1 bit; the two optimizer points suggest about +17% KLD a step, an estimate), the headroom from
512M to 256M (about 0.83 of a 32,768 step), a ladder step of 16,384 (the table's last column).

## Finding

Observed:

1. No Qwen3.8 smaller than the 27B exists (Hub listing, 2026-11-15). The model of the class stays the 27B.
2. No public quant at 2.4, 2.65 or 2.9 bits binds with today's K classes. The only one between 2.2 and 3.0 that does is
   the third-party 2.5 (K2 and K3 in halves, a 4-bit head, 8.299 GB, quantizer 1.6.0).
3. In KLD and top-1 the curve is smooth from 3.0 to 2.2 bits, about 1.5 times the KLD and 1.6 to 1.7 points of top-1
   for every quarter of a bit, on five harnesses, and bends sharply only below 2.0 bits, where K1 matrices appear. The
   allocation matters as much as a quarter of a bit (0.0519 against 0.0306 for two public 2.5s on one harness).
   On ignis's instrument (#632, 2026-11-16) the half bit from 3.0 to 2.5 costs a factor of 1.66 in KLD (0.0405 to
   0.0670, paired interval 1.49 to 1.81) and 2.4 points of top-1 (93.40% to 91.00%), above the 3.0 in all 16 windows.
4. In the one public agent-task harness the loss grows with the length of the task. A trellis body at 7.25 GB keeps
   94, 73 and 24% of BF16 on single-turn code, Terminal-Bench and DeepSWE; at 9.85 GB, 100, 93.5 and 77%. The dynamic
   2-bit line steps on the long tasks between 8.0 and 9.2 GB. A harder quant also spends more tokens and times out more
   (1.27 times the output tokens, 15 timeouts against 8, on one blog run). On the 32-request loop set the third-party
   2.5 ends on a tool call 15 times against 24 for the 16 GB artifact and loops on 11 against 8 (p = 0.29).
5. The model's own MTP head's acceptance is flat from 3.5 to 2.5 bits on one public card (chart, ±0.3, 5 samples a
   bar).
6. At the class's shape on the RTX 5090 (#635, 2026-11-16) candidate A, the optimizer's 9.2 GB recipe, reads 0.04308
   KLD against the 16 GB artifact's 0.04177 (ratio 1.032, interval 0.993 to 1.070), under the limit of 0.05012;
   candidate B, the GGUF allocation, reads 0.05084 (1.217), above it. The spec's estimate was 0.056. On the 277
   matrices at the same K the reconstruction errors of the two artifacts have a median ratio of 0.9993, so the 1.03 is
   the recipe's and not the quantizer version's.
7. On A, MTP decodes 3.2 to 3.5 times faster than no speculation (294, 251 and 228 tok/s at 8,192, 65,536 and 130,560
   tokens; off 92, 76 and 65), DFlash2 4.4 to 4.9 times (409, 348 and 321), the context 131,072 and nothing adapted in
   12 GiB under both; the KV pool 214,144 tokens under MTP, 150,720 under DFlash2. Loops: 7/32 against the 16 GB
   artifact's 11/32 (Fisher p = 0.92); replay 4.072 tokens a round against 4.116.
8. The prefill chunk (#636): 256 buys no pool with vision on and costs 25% more first token at 8,192; 512 buys 4,608
   tokens of pool over 1,024 (the spec estimated 17,000) for 7% of the first token; decode does not move.

Inferred (`inference` in the sections above):

- The step of the long tasks lies between about 7.3 and 8.3 GB of body and head, so a 9.2 GB body is on its upper side
  and a second lane at 8.0 GB or below is in its steepest stretch.
- Between a one-lane and a two-lane recipe (0.87 GB, about 0.28 bits of the body) a coding agent probably loses about
  one terminal task in ten and one long repository task in four or five of those the larger completes; short edits
  barely change. This is an interpolation between two trellis points 2.6 GB apart and one vendor's line, not a
  measurement.
- KLD per bit carries over to tasks only partly: the dynamic 2-bit line at 9.2 GB and the trellis at 9.85 GB score
  alike on tasks where their KLDs differ by more than two on an older file.
- A recipe at the 9.2 GB ceiling (2.79 bits) would read about 0.049 in KLD by log-linear interpolation between the
  3.0 and the 2.5 of #632, and about 0.047 by the optimizer's prediction for candidate A (1.17 times the 16 GB
  artifact's), both at the 0.0486 limit; interpolation and prediction, not measurements of a recipe.
- Staying with integer K costs a predicted 4 to 5% of KLD against half-steps, small against a factor of 1.5 a quarter
  of a bit.
- The optimizer's allocation at 9.2 GB loses much less than either interpolation said (1.03 against 1.17 predicted
  and about 1.2 interpolated); whether the tasks that give way below 8.3 GB (above) agree with the KLD is not
  measured.
- At 131,072 tokens in 12 GiB the class's context is a property of the RTX 5090's lines; on a real card with a desktop
  the arithmetic below gives 98,304 at about 0.3 GB taken and 32,768 at about 1 GB (a derived budget is the free memory
  less 512 MiB), not the spec's roughly 100,000 at 1 GB.

## Implications

- The spec's 9.2 GB ceiling is consistent with this evidence, which is its only support: at 9.2 GB the second format's
  files keep 90% (terminal) and 72% (repository) of BF16 in the one harness that has them, and the long tasks give way
  below about 8.3 GB. #632's slope on ignis's own instrument tests it and neither of the spec's two cases is met (the
  2.5 is not close to the 3.0, 1.66 times its KLD, and 9.2 GB is not already low: an interpolated 2.79 bits sits at
  the limit), so the candidates are built at 9.2 GB. The candidates built there read 1.03 (A) and 1.22 (B) times the
  16 GB artifact's KLD: A holds the registered limit with the ceiling where it is.
- The decisions taken on the gate's numbers (the artifact admitted, the ceiling, the chunk, the ladder's step) are
  [ADR 0098](../adr/0098-the-12-gb-card-class-serves-the-27b-from-a-recipe-of-our-own-under-a-byte-ceiling-on-one-lane-drafting-with-mtp.md)'s.
- The gate's numbers (KLD, the drafter's acceptance, the loop rate) are blind to what the public evidence says matters
  most: the length of a task, the tokens spent, the timeouts, the validity of a tool call. The agent-facing checks the
  repository has (the replay of recorded turns, the canary prompts, the regression set) are the instrument for those,
  beside the gate and not binding.
- A recipe built to a byte ceiling should be compared against the public integer-K 2.5 on KLD first: its allocation
  is the cheapest candidate there is, and the allocation is worth about a quarter of a bit.
- Flash-Next's per-byte lead is a later phase with its own cost in host memory and licence; a smaller sibling, if
  the Qwen4 27B of the secondary sources ships, would change the question and is worth a check at its date.

## Limits and unknowns

- **One agent-task harness, run by a competitor.** The trellis rows are a quantizer's competitor's runs, through a
  runtime its page does not name, with its own samplers and templates. Every value is a chart reading (±0.5), the
  harness's own noise is 2 to 3 points a point, and the trellis curve has two points: the shape between 7.25 and 9.85
  GB is not known, and the 2.2 point used a 3-bit head where the engine keeps 4.
- **No agent-task measurement of any trellis quant between 2.2 and 3.0 bits exists** in public, including the third-party
  2.5. The nearest proxy, the dynamic 2-bit line, is another format and another runtime.
- **Not measured by anyone, on any model of this size below 3 bits:** the validity of tool calls and of structured
  output (the papers read stop at 4 bits and find more of the errors a model already makes, hidden by retry budgets),
  retrieval or agent work past 100K tokens, drift over many turns, DFlash2's acceptance on a target below 4 bits
  (MTP's is flat on one card; a bound by argument for a drafter that misses every flipped token is a few percent
  relative, an argument and not a measurement), the effect of a half-bit on the tokens spent per solved task (its
  direction is measured, its size is not). The 30 to 50 agentic tasks, three times, on a 2.4 and a 2.65 build,
  logging pass rate, tokens a solved task, tool-call parse failures and accepted length, are what is missing and cheap.
- **Task evidence from other formats and models** (scalar GPTQ on a 30B mixture, AQLM on code models, a 30B-A3B at 2.0
  to 2.25 bits losing 4 to 5 points of HumanEval+ and 7 to 9 of BFCL) is at other widths, other formats or other
  architectures: it shows failure modes (a termination threshold that a mean KLD does not announce; function calling
  losing about twice what single-function code does), not the 27B's curve.
- **The third-party 2.5's quality, beyond the author's wikitext reading with the quantizer's own evaluator, is #632's
  KLD and loop set**; its agent-task behaviour and its tool calls are still unmeasured. Its calibration
  (250 rows of 2,048 columns) is not stated to be the 16 GB artifact's source's trace.
- **The published KLD figures come from five harnesses** with different corpora and units; ratios within a harness are
  the usable reading, never levels across them. Where a card gives a KLD with no stated set (the 2.65 to 2.80 series),
  its harness is unknown.
- **The card's own lines.** The research says nothing about the CUDA context or the workspace of a real 12 GB card.
  The gate's plans are the RTX 5090's lines held to a 12 GiB budget, and the context on a real card is arithmetic on
  them, not a measurement.
- **The gate's reading is one session, one card, one artifact each.** Each KLD rests on 16 windows (intervals above), the
  loop rates on 32 requests, the decode cells on two launches at most with a spread of 3 to 5 tok/s, the chunk table
  on two launches a chunk. The replay ran at the 16 GB shape (two lanes, 1,024 chunk) on 24 turns. The regression set
  was not run (its test pins the NVFP4 artifact by path), B's drafter and loop legs were not run (its KLD decided), the
  16 GB counterpart's own decode cells were not measured, and the optimizer's +17% KLD a step comes from two optimizer
  points, an estimate.
- Pages that could not be read: every thread of one discussion forum (blocked), two paywalled newsletters, the methods article of the
  quant publisher's 3.1 update, the Flash-Next technical report, a gated card. One page that printed tool-call validity
  per quant was not used: it names no primary source and contradicts itself on three figures.

## Follow-ups

- A set of 30 to 50 agentic tasks, three attempts each, on `trellis-ig28` and the 16 GB artifact, would measure what
  the public evidence cannot (tool-call validity, tokens a solved task).
- Readings on a real 12 GB card: [#639](https://github.com/gpillon/ignis/issues/639).
