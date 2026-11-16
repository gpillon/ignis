# The 27B's MTP head measured on the RTX 5090: 3.9 tokens a round on recorded agent turns, about 2.2 times window 0's committed tokens per second, and a default window of 7 set on the 16 GB artifact

- Kind: experiment
- Status: current
- Observed: 2026-11-15
- Last verified: 2026-11-16
- Scope: speculative decoding / the 27B's MTP head as a drafter (spec #618) on the RTX 5090, the integration branch at
  e6ecec69 with #628's instruments: the NVFP4 artifact under `rtx5090` and the 16 GB artifact in a 13 GiB budget under
  `rtx-16gb`; acceptance on #267's recorded agent turns, the cost of a round by draft window, lanes and context, the VRAM
  plan against DFlash2 and no speculation, the speed and quality gates, the default draft window
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/628,
  https://github.com/gpillon/ignis/issues/618,
  https://github.com/gpillon/ignis/issues/629,
  [ADR 0095](../adr/0095-the-27b-has-two-drafters-and-its-mtp-head-keeps-attention-state-over-the-whole-context-and-a-continuation-hidden.md),
  [the head drafts as the reference engine's](2026-11-15-the-27bs-mtp-head-drafts-as-the-reference-engines.md),
  [DFlash2 sampled acceptance](2026-09-24-dflash2-sampled-acceptance.md) (#267: the turns and the width-cost method),
  [the sampled draft served](2026-11-13-the-sampled-draft-served-on-the-27b.md),
  [the 16 GB artifact's capacity](2026-10-24-27b-16gb-capacity-on-the-5090.md)
- Superseded by: none
- Summary: RTX 5090, 27B MTP head at window 7: 3.90/3.88 tokens a round on 200 agent turns (NVFP4/16 GB), 1.3-2.9x window 0's decode, 1.01/0.80 GB less VRAM than DFlash2; windows 5-7 tie on 16 GB, default 7.

## Question

Spec #618 made the 27B's MTP head a second drafter (#619-#627) at a fixed draft window of 0 to 7, 7 until this
measurement. #625 showed the head is read right: 4.241 tokens a round against the reference engine's 4.207 on short
greedy prompts, 3.87 against 3.97 on four prompts of 2K to 15K tokens. What it does on traffic, what it costs and what
it saves were estimates: the spec's "fewer tokens a round than DFlash2", "about 0.7 to 0.8 GB less at one lane", a
chain step at "about 11% of a round by bytes read", and the 16 GB study's "~1.1 GB fixed saved" (spec 27b-16gb's
phase 2 notes), which did not count the head's own weights. On the RTX 5090, on two legs (the NVFP4 artifact under
`rtx5090`, the 16 GB artifact under `rtx-16gb`'s budget), #628 asks:

- how often the head's draft is accepted at each position 1 to 7 on the recorded agent turns, how many tokens a round
  commits at each window, and how the head's probability of a draft reads against its acceptance;
- what a round costs at each window 0 to 7, at one lane and at the load's lanes, at a short and a long context; what
  share of it the chain takes; what a prefill pays for the head's layer;
- what MTP holds in VRAM against DFlash2 and against no speculation, line by line;
- whether MTP at its default window commits more tokens per second than the same load at window 0 in every cell, and
  whether it keeps today's quality gates for a drafter;
- which window is the default: the most committed tokens per second on the 16 GB leg.

## Evidence

**The tree.** The integration branch at e6ecec69 (#619-#627) and this ticket's instruments: the example
`mtp_round_cost` (`crates/runtime/examples/`, the MTP sibling of `verify_width_cost`), the entailment regression set's
MTP test, `mtp_round_gpu`'s reference figure at 4.2 (it read 3.0, the shortlist-and-adaptive figure). One release
build of `ignis-server` (cuda), `ignis-bench` and the examples; nothing compiled while a timing leg ran. RTX 5090
(Windows 11, WDDM, PCIe Gen 3), the desktop holding about 2.6 GB; every GPU leg after `make gpu-status`, alone on the
card, the loop guard off. The protocol, the cells and the rule for the default were written before the first launch.
Every script, record and log is in the main checkout's `.scratch/618/628/` (untracked): `protocol.md`, `replay/`,
`cost/`, `served/`, the analysis scripts beside them.

**The legs.**
- NVFP4: `qwen3_8_27b_nvfp4full-v2`, served `--profile rtx5090`: 917,504 context, 8 lanes, 2,048-token chunks,
  hq-e8-2b, vision on (host residency), rope `yarn:3.5`.
- 16 GB: `qwen3_8_27b_trellis_sc30-v2`, served `--profile rtx-16gb --vram-budget-bytes 13958643712`: 13 GiB, the
  guide's simulated 16 GB card less 3 GiB of desktop; 2 lanes, 1,024-token chunks, hq-e8-2b, vision on (host).
- Every served start names an empty config file, so the host's user configuration file (FP8 exception weights, a
  1,500 MB headroom) is not read; beside it `--model-exception-weights bf16`, `--qwen38-vram-headroom-bytes 1200M`
  (NVFP4; the 16 GB leg names its budget) and `--model-idle-unload 0`.
- The replay and the cost example run the replay shape (`support/replay_shape.rs`): NVFP4 at 8 lanes on the card's
  derived budget, the trellis in `rtx-16gb`'s 16 GiB plan at 2 lanes; 262,144 context, hq-e8-2b, no vision, the head
  at the named window under the fixed draft, verify graphs captured.

### The recorded agent turns under the head

#267's 200 recorded agent turns (116 token streams, 6.47 M prefilled context tokens, 208,860 recorded output tokens,
sampled at temperature 1, top-p 0.95, top-k 20), replayed teacher-forced by `mtp_replay` (#626) at a window of 7 on
each leg: per round the head's chain as the readout reads it (each draft, the head's probability of it, its 16 leading
candidates), then a forced verify round that commits the recorded text. A greedy lane takes the fixed draft exactly
when the recording has it there, so the rounds are the ones a live run that sampled this text would have made. One
run a leg (51 and 64 minutes); 0 readout disagreements on either. Acceptance at a position is the rounds whose
accepted run reaches it over the rounds that tested it (`records.accepted_at` over `records.tested_at`). Tokens a
round are pooled with #267's traffic weights, each cell's share of the dataset's output tokens, as #267 and #574
pooled DFlash2's.

| leg | rounds | tokens | tokens a round (traffic-weighted) | unweighted |
|---|---:|---:|---:|---:|
| NVFP4 | 54,647 | 208,860 | **3.897** | 3.822 |
| 16 GB | 54,846 | 208,860 | **3.879** | 3.808 |

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVFP4, accepted / drafted | 0.741 | 0.564 | 0.441 | 0.352 | 0.287 | 0.239 | 0.203 |
| NVFP4, given the one before | 0.741 | 0.761 | 0.783 | 0.797 | 0.818 | 0.832 | 0.851 |
| 16 GB, accepted / drafted | 0.740 | 0.561 | 0.437 | 0.350 | 0.286 | 0.238 | 0.201 |
| 16 GB, given the one before | 0.740 | 0.758 | 0.778 | 0.802 | 0.819 | 0.834 | 0.846 |

- **Against DFlash2 on the same turns.** #574's replay, the same instrument at a window of 7, committed 3.450 tokens a
  round under the fixed draft and 3.825 under the sampled draft, DFlash2's default; #340's, on the 16 GB artifact at
  its build of 2026-10, 3.370 under the fixed draft.
- **By traffic cell** (NVFP4; the 16 GB leg within 1%): thinking 3.18 (A), 3.76 (B), 3.68 (C) tokens a round; content
  4.07, 4.66, 4.91.
- **By context**, the round's anchor position (NVFP4; 16 GB within 0.6%):

| context | rounds | tokens a round | accepted at position 1 / 4 / 7 |
|---|---:|---:|---|
| under 16K | 5,056 | 3.909 | 0.733 / 0.372 / 0.228 |
| 16K-48K | 17,583 | 3.620 | 0.720 / 0.319 / 0.175 |
| 48K-96K | 20,232 | 3.712 | 0.738 / 0.331 / 0.181 |
| 96K and past | 11,776 | 4.276 | 0.780 / 0.426 / 0.274 |

- **The head's probability beside acceptance** (every draft whose earlier drafts were accepted, all positions; the
  probability is the head's softmax at temperature 1 over the whole vocabulary):

| the head's probability | drafts (NVFP4) | accepted (NVFP4) | accepted (16 GB) |
|---|---:|---:|---:|
| 0.10-0.20 | 1,117 | 0.166 | 0.145 |
| 0.20-0.30 | 4,404 | 0.220 | 0.210 |
| 0.30-0.40 | 7,727 | 0.289 | 0.285 |
| 0.40-0.50 | 10,500 | 0.366 | 0.363 |
| 0.50-0.60 | 11,750 | 0.447 | 0.434 |
| 0.60-0.70 | 10,991 | 0.513 | 0.507 |
| 0.70-0.80 | 11,639 | 0.597 | 0.583 |
| 0.80-0.90 | 13,742 | 0.693 | 0.689 |
| 0.90-0.95 | 10,136 | 0.779 | 0.769 |
| 0.95-0.99 | 16,667 | 0.865 | 0.865 |
| 0.99-1.00 | 98,923 | 0.984 | 0.984 |

**Tokens a round at windows 1 to 6** are derived from the window-7 records: a round that accepted `a` drafts commits
`min(a, w) + 1` at window `w` (its committed run is already cut at a stop and at the turn's end). A walk at a smaller
window starts its rounds elsewhere, more of them in the middle of a predictable run, so the derivation reads it low.
The check, fixed in the protocol: jobs 0 to 21 (24 turns, 15,179 tokens) replayed directly at windows 4, 5 and 6 on
both legs, against the derivation from the window-7 records of the same jobs.

| window | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVFP4, derived | 1.752 | 2.331 | 2.786 | 3.150 | 3.446 | 3.690 | 3.897 |
| NVFP4, direct / derived on jobs 0-21 | | | | 1.035 | 1.017 | 1.017 | |
| NVFP4, corrected | | | | 3.260 | 3.503 | 3.754 | |
| 16 GB, derived | 1.750 | 2.325 | 2.774 | 3.135 | 3.430 | 3.675 | 3.879 |
| 16 GB, direct / derived on jobs 0-21 | | | | 1.034 | 1.028 | 1.018 | |
| 16 GB, corrected | | | | 3.241 | 3.526 | 3.740 | |

### What a round costs

`mtp_round_cost` loads, one after another in one process, the plain load (window 0), the head at each window 1 to 7,
and the verify substrate alone at each window 1 to 7 (`SpeculativeBackend::VerifyOnly`: the same verify round, its
graphs and its ReplaySSM records, fed each lane's next tokens of the text as drafts, no drafter bound). Each lane is
prefilled with its own window of a real agent stream (the five longest recorded streams, concatenated) to 4,096
tokens, measured, then to 65,536, measured again. Per context and lane count (one, and the load's: 8 and 2), 64
greedy rounds after 3 warm-up rounds, each timed on the host around the call, which returns after its device work.
A round costs what it costs whatever it accepts: the chain runs its steps and the verify round its columns whatever
the accept keeps. Two launches a leg, interleaved with the other leg's; a cell is the mean of the two launches'
medians, the spread their difference. What the head adds to a round is the MTP round less the substrate's at the same
window, lanes and context: its chain (a step per draft, each the head's layer and the target's full output head, the
head's attention over the whole context) and the commit of its rows. A chain step is the slope of the MTP round over
the window less the slope of the substrate's. The NVFP4 leg runs its served 2,048-token chunk.

Round milliseconds (spread between the launches in brackets):

| leg, lanes, context | plain | w1 | w2 | w3 | w4 | w5 | w6 | w7 | chain step | verify column |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4, 1, 4K | 11.78 (0.04) | 13.52 (0.08) | 15.03 (0.03) | 17.59 (1.49) | 17.99 (0.11) | 19.15 (0.07) | 20.36 (0.04) | 21.89 (0.28) | 1.182 | 0.151 |
| NVFP4, 8, 4K | 15.31 (0.10) | 18.61 (0.03) | 20.55 (0.13) | 23.73 (2.28) | 24.69 (0.15) | 26.57 (0.08) | 29.09 (0.18) | 30.96 (0.56) | 1.398 | 0.637 |
| NVFP4, 1, 64K | 14.12 (0.02) | 16.06 (0.08) | 18.59 (1.48) | 19.76 (0.12) | 20.94 (0.02) | 22.37 (0.15) | 23.93 (0.32) | 25.80 (1.57) | 1.342 | 0.177 |
| NVFP4, 8, 64K | 33.82 (0.30) | 40.05 (0.19) | 45.74 (3.83) | 47.84 (1.10) | 51.09 (0.38) | 54.51 (0.08) | 58.81 (1.18) | 62.71 (2.54) | 2.593 | 1.007 |
| 16 GB, 1, 4K | 10.22 (0.05) | 12.10 (0.12) | 12.97 (0.07) | 14.15 (0.27) | 15.03 (0.14) | 15.86 (0.05) | 17.46 (0.90) | 17.78 (0.14) | 0.795 | 0.195 |
| 16 GB, 2, 4K | 11.17 (0.01) | 12.92 (0.10) | 14.07 (0.01) | 15.51 (0.64) | 17.31 (0.18) | 18.28 (0.13) | 19.54 (1.15) | 19.99 (0.03) | 0.765 | 0.482 |
| 16 GB, 1, 64K | 12.65 (0.01) | 14.87 (0.04) | 15.98 (0.12) | 17.33 (0.37) | 18.27 (0.06) | 19.72 (0.60) | 20.89 (0.67) | 21.69 (0.14) | 0.917 | 0.250 |
| 16 GB, 2, 64K | 16.08 (0.03) | 18.51 (0.29) | 19.90 (0.00) | 21.42 (0.15) | 23.82 (0.05) | 25.92 (0.81) | 26.42 (0.01) | 27.86 (0.08) | 1.077 | 0.551 |

The head's share of the round (MTP less the substrate, over MTP), at windows 1, 3, 5 and 7:

| leg, lanes, context | w1 | w3 | w5 | w7 |
|---|---:|---:|---:|---:|
| NVFP4, 1, 4K | 11.1% | 25.8% | 32.0% | 39.4% |
| NVFP4, 8, 4K | 9.1% | 23.3% | 27.2% | 32.0% |
| NVFP4, 1, 64K | 10.9% | 22.4% | 30.8% | 38.9% |
| NVFP4, 8, 64K | 10.4% | 20.6% | 26.5% | 32.3% |
| 16 GB, 1, 4K | 8.8% | 19.3% | 22.7% | 32.1% |
| 16 GB, 2, 4K | 8.4% | 19.7% | 20.7% | 27.8% |
| 16 GB, 1, 64K | 9.3% | 19.5% | 23.9% | 31.7% |
| 16 GB, 2, 64K | 9.6% | 18.1% | 23.3% | 29.4% |

- **A launch can read one load slow**: NVFP4's window 3 at 4K and window 2 at 64K read 7-9% slower on the second
  launch at both lane counts, every other load within 1.5%. Each cell is the two launches' mean.
- **The trellis's verify width is not smooth**: on the 16 GB leg the substrate at window 5 (six columns) costs more
  than at windows 6 and 7 at every cell, and the MTP round at window 7 costs at most 0.5 ms more than at window 6.
- **The prefill pays for the head's layer** (each lane's prefill from 0 to 4,096 and from 4,096 to 65,536, the mean
  over the seven MTP loads against the plain load): NVFP4 14,181 against 12,212 tok/s (+16.1% time) and 10,078
  against 8,574 (+17.5%); 16 GB 3,797 against 3,565 (+6.5%) and 3,355 against 3,169 (+5.9%). The substrate's
  prefill is the plain load's within 1%.

### Committed tokens per second, and the default window

The rule, fixed before the cost launches: on the 16 GB leg, committed tokens per second at window `w` is T(w) / C(w),
T the replay's traffic-weighted tokens a round (T(0) = 1), C the cost table's round (C(0) the plain round), in each of
the four cells; one window best in every cell is the default, else the window with the highest geometric mean over
the cells of its rate against window 0's. With T from the corrected figures (windows 4 to 6):

| leg, lanes, context | w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 GB, 1, 4K | 97.9 | 144.6 | 179.2 | 196.0 | 215.7 | **222.3** | 214.2 | 218.2 |
| 16 GB, 2, 4K | 89.6 | 135.5 | 165.2 | 178.9 | 187.2 | 192.9 | 191.4 | **194.0** |
| 16 GB, 1, 64K | 79.1 | 117.7 | 145.5 | 160.1 | 177.4 | 178.8 | **179.1** | 178.8 |
| 16 GB, 2, 64K | 62.2 | 94.5 | 116.8 | 129.5 | 136.1 | 136.0 | **141.6** | 139.2 |
| 16 GB, geometric mean against w0 | 1 | 1.499 | 1.848 | 2.027 | 2.181 | 2.218 | 2.216 | **2.224** |
| NVFP4, 1, 4K | 84.9 | 129.5 | 155.1 | 158.4 | 181.2 | 182.9 | **184.4** | 178.1 |
| NVFP4, 8, 4K | 65.3 | 94.2 | 113.4 | 117.4 | **132.1** | 131.8 | 129.0 | 125.9 |
| NVFP4, 1, 64K | 70.8 | 109.1 | 125.4 | 141.0 | 155.7 | 156.6 | **156.9** | 151.0 |
| NVFP4, 8, 64K | 29.6 | 43.8 | 51.0 | 58.2 | 63.8 | **64.3** | 63.8 | 62.1 |
| NVFP4, geometric mean against w0 | 1 | 1.497 | 1.764 | 1.905 | 2.127 | **2.138** | 2.128 | 2.063 |

(tokens per second a lane; at the load's lanes a lane's rate times the lanes is the load's.) With the derived figures
uncorrected, window 7 is the 16 GB leg's best in all four cells (geometric mean 2.224 against window 6's 2.177) and
window 5 the NVFP4's (2.103).

### Served: the speed gate, the three plans, the quality gates

**Launches.** Per leg, ABBA: the head at its default window, window 0, window 0, the head (`--spec-backend mtp
--spec-draft-tokens 7`, then `0`: the same load as `--spec-backend off`), then one DFlash2 start (`--spec-backend dflash2
--spec-draft-tokens 7`) for its plan, `/v1/config` and the canary only (the owner's cut of 2026-11-16: DFlash2's speed is
no part of this gate). Each start runs the worktree's release build; the speculative counters are scraped before and
after each cell. Cells: `ignis-bench canary`; `g5-equivalence capture`; `ttft` at 8,192 and 32,768 tokens (three cold
samples); the agent trace `bench/traces/g4-load-trace.jsonl` as recorded (11 requests) at one lane and at the load's
lanes, greedy and sampled (temperature 1, top-p 0.95, top-k 20, seed 628); prompts of about 32K tokens of the
repository's code and a question (#574's 64K set cut in half), 1,024 tokens each, three at one lane and the load's
lanes' worth at once, greedy and sampled; on the first launch of each configuration, #340's loop set. Decode tokens per
second are a cell's decoded tokens over its decode time, each request's total less its time to first token, a
configuration's two launches pooled. The gate per cell, fixed in the protocol: the head's pooled rate above window 0's
by more than the larger of the two configurations' launch-to-launch differences.

| cell (lanes) | NVFP4: MTP | w0 | MTP / w0 | larger spread | 16 GB: MTP | w0 | MTP / w0 | larger spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agent trace, greedy, 1 | 171.1 | 75.6 | 2.26 | 0.8 | 245.8 | 86.1 | 2.86 | 3.9 |
| agent trace, greedy, N | 117.0 | 54.0 | 2.17 | 13.5 | 147.3 | 66.7 | 2.21 | 13.1 |
| agent trace, sampled, 1 | 137.1 | 75.6 | 1.81 | 2.9 | 155.0 | 85.1 | 1.82 | 2.5 |
| agent trace, sampled, N | 110.2 | 65.5 | 1.68 | 1.3 | 114.2 | 72.9 | 1.57 | 9.9 |
| 32K, greedy, 1 | 155.3 | 74.9 | 2.07 | 1.4 | 167.9 | 84.4 | 1.99 | 1.7 |
| 32K, greedy, N | 44.6 | 31.5 | 1.42 | 1.3 | 88.5 | 55.7 | 1.59 | 3.9 |
| 32K, sampled, 1 | 134.7 | 74.6 | 1.81 | 1.1 | 129.7 | 83.5 | 1.55 | 2.0 |
| 32K, sampled, N | 43.9 | 33.3 | 1.32 | 1.5 | 79.4 | 62.7 | 1.27 | 13.8 |

(N is 8 on NVFP4 and 2 on the 16 GB leg; decode tokens per second.) The narrowest margin is the 16 GB leg's sampled
32K cell at two lanes: 16.7 tokens per second against a 13.8 spread. The eight-lane cells overlap the other lanes'
prefills, and the eight-lane greedy trace changes its text between launches (window 0: 57.7 and 48.0). Time to first
token, the two launches' medians averaged: NVFP4 841.9 against 743.9 ms at 8K (+13.2%) and 3,365.9 against 2,977.8 ms
at 32K (+13.0%); 16 GB 2,382.9 against 2,338.4 ms (+1.9%) and 9,768.6 against 9,630.8 ms (+1.4%), inside window 0's
own launch spread at 32K (4%).

**The three plans** (the `ignis.runtime.vram_plan` line of each configuration's first start), bytes:

| line | NVFP4: MTP | DFlash2 | none | 16 GB: MTP | DFlash2 | none |
|---|---:|---:|---:|---:|---:|---:|
| weights | 16,307,401,216 | 16,939,004,164 | 15,856,133,632 | 10,276,355,072 | 10,907,958,020 | 9,825,087,488 |
| workspace | 983,876,864 | 983,876,864 | 983,876,864 | 343,493,376 | 395,924,480 | 303,645,184 |
| trellis workspace | 0 | 0 | 0 | 36,225,280 | 36,225,280 | 36,225,280 |
| decode graph | 102,049,376 | 82,125,920 | 13,743,200 | 25,602,584 | 20,621,336 | 3,447,832 |
| verify round | 152,139,488 | 152,139,488 | 0 | 38,034,872 | 38,034,872 | 0 |
| drafter round | 0 | 78,034,720 | 0 | 0 | 44,675,336 | 0 |
| lane state | 1,239,580,672 | 1,575,043,072 | 1,239,498,752 | 309,895,168 | 393,760,768 | 309,874,688 |
| hq residual window | 303,039,488 | 285,213,184 | 285,213,184 | 75,759,872 | 71,303,296 | 71,303,296 |
| sampling and residual | 129,098,144 | 129,098,144 | 129,098,144 | 125,631,568 | 125,631,568 | 125,631,568 |
| CUDA context (measured) | 452,161,536 | 452,161,536 | 452,161,536 | 452,161,536 | 434,401,280 | 452,161,536 |
| KV pool | 10,174,738,432 | 9,110,880,256 | 10,896,867,328 | 2,274,893,824 | 1,489,903,616 | 2,831,187,968 |
| KV pool tokens | 1,039,040 | 988,544 | 1,182,336 | 232,320 | 161,664 | 307,200 |
| context | 917,504 | 917,504 | 917,504 | 131,072 (adapted) | 65,536 (adapted) | 262,144 |

- **What MTP saves against DFlash2**, the drafter's lines: on NVFP4 631.6 MB of weights, 335.5 MB of drafter window in
  the lanes' state and 78.0 MB of drafter round, less 19.9 MB more decode graph and 17.8 MB more residual window:
  **1.007 GB**. On 16 GB 631.6 MB, 83.9 MB, 44.7 MB and 52.4 MB of workspace, less 5.0 MB and 4.5 MB: **0.803 GB** (the
  starts' measured CUDA contexts differ by 17.8 MB, which is no drafter's line).
- **Against no speculation** MTP holds 709.6 MB more on NVFP4 and 555.8 MB more on 16 GB outside the pool, and each of
  its pool's pages is 17/16 of a page.
- **The workspace line, settled**: on the NVFP4 load (2,048-token chunks, vision's 32,768-token envelope) the image
  encoder's share sizes the arena under all three; on the 16 GB load (1,024-token chunks, a 4,096-token envelope) the
  prefill's scratch does, and DFlash2's taps add 92.3 MB to it (the spec's "up to 92.3 MB"), the head's prefill 39.8 MB.
- **The residual, settled**: WDDM's counter for the server process 5 s after it answered its first `/v1/models`, before
  any cell (`GPU Process Memory`,
  `Dedicated Usage`: byte-exact, where NVML's free memory drifts with the desktop), less every other line of its plan:
  NVFP4 MTP 109.6 and 110.7 MB (two launches), DFlash2 117.0 MB, none 67.7 and 66.3 MB; 16 GB MTP 45.7 MB and, once,
  42.7 MB under its other lines, DFlash2 64.5 MB, none 32.6 MB twice. The plan's 124,474,800-byte residual line covers
  every one: no load held more than its plan.

**Quality, MTP on against off.**
- *Greedy text, up to the near-tie rule.* `mtp_round_gpu` (5 tests; the canaries at one, four and eight lanes against
  the same load's plain rounds, NVFP4 and 16 GB legs): pass, with the one divergence ADR 0095's exception names (the
  verify tile's pick, 3 BF16 ulps). `vision_dflash2_gpu`'s MTP test: the verify rounds' text equals the decode rounds'
  after each image, up to near-ties. Served, `g5-equivalence` on every launch: on the 16 GB leg the four canaries are
  the same tokens under MTP and window 0, both pairs of launches; on NVFP4 three part, at tokens 3, 4 and 23, the same
  on both pairs, each launch equal to its twin. Probed after the campaign by the near-tie rule (`support/near_tie.rs`'s
  two prefill routes, on a plain load at the served `hq-e8-2b` and `yarn:3.5`; a one-off test kept in
  `.scratch/618/628/judge/`): all three are near ties, the per-token route rating them 0 to 1 BF16 ulp apart. The
  probe stands on the served context: its prompts are the served ones' lengths (31, 40, 36, 35 tokens), its first
  greedy pick is the served first token on every canary, and at each divergence it picks one of the two served picks.
  A first probe under BF16 KV was set aside: there rust-sort's first token is already another (63 where both served
  launches wrote 5836), so it probed a different context, and it rated rust-sort's divergence 4 and 3 ulps to
  spec-off's side, past the rule.
- *The canary floor.* `vision_dflash2_gpu`'s MTP test, teacher-forced against the vision canary: 48 of 50, 96.0%
  (floor 95%). `ignis-bench canary` on every launch, DFlash2's included: sane and deterministic.
- *The regression set.* `decide_entailment_gpu` on the speculation-off load and on an MTP load: every gate case keeps
  its label, 16 of 18 agree on both (case 17 a watch case), and every case's probability is the same on both loads.
- *The loop measures.* #340's loop set (four bodies, seeds 101-108, 32 requests at the load's lanes), the loop guard
  off: a request loops if its reasoning holds a 6-gram at least 25 times or its worst 3,000-character zlib window
  compresses below 0.20. NVFP4: 7 of 32 under MTP, 6 under window 0 (one-sided Fisher exact p = 0.50 for "MTP loops
  more"); 16 GB: 11 and 11 (p = 0.60).

**The long-context row against the reference engine** (#625's follow-up leg, run by the coordinator on 2026-11-15 at
the integration branch's 41446d0c: NVFP4, BF16 KV, window 7, greedy, one lane): on four prompts of 2,305 to 15,179
tokens ignis commits 3.869 tokens a round to the reference engine's 3.973 (0.974), per prompt 3.75, 3.49, 3.66 and
4.74 against 3.18, 3.70, 3.92 and 5.09; by position 0.806 0.617 0.447 0.349 0.281 0.218 0.171 against 0.826 0.644
0.479 0.365 0.284 0.211 0.165. On #625's sixteen short prompts, served and cut at the stop: 4.204 against 4.207.

## Finding

Observed:
- **The head commits as many tokens a round as DFlash2 on agent traffic.** On 200 recorded agent turns sampled at
  temperature 1, at a window of 7, it commits 3.897 tokens a round on the NVFP4 artifact and 3.879 on the 16 GB one,
  traffic-weighted; DFlash2 commits 3.825 on the same turns under its default sampled draft and 3.450 under the fixed
  draft. The spec's "fewer tokens a round than DFlash2" does not hold on this traffic.
- **Acceptance by position** falls from 0.74 at the first draft to 0.20 at the seventh, while acceptance given the draft
  before it rises from 0.74 to 0.85: a run that survived its early drafts tends to go on. It does not fall with the
  context: 3.6 to 4.3 tokens a round from under 16K to past 96K.
- **The head's probability is overconfident under sampling**: a draft it rates 0.5 to 0.6 is accepted 0.43 to 0.45
  of the time, 0.8 to 0.9 about 0.69; at 0.99 and over, 0.984. Its order is right: acceptance rises in every bin.
- **A chain step** (the head's layer and the model's whole output head) costs 1.18 to 1.34 ms on the NVFP4 artifact at
  one lane, 1.40 at eight lanes 4K deep and 2.59 at eight lanes 64K deep; 0.77 to 1.08 ms on the 16 GB artifact. A
  verify column costs 0.15 to 0.18 ms at one lane, 0.64 to 1.01 at eight lanes (NVFP4), 0.20 to 0.55 (16 GB).
- **The chain's share of a round**, with the full proposal head: 32 to 39% at window 7 on NVFP4, 28 to 32% on 16 GB;
  27 to 32% and 21 to 24% at window 5.
- **What a prefill pays for the head's layer**: 16 to 18% more prefill time on the NVFP4 artifact (cost example, the
  same load shape either way), 13% more time to first token served at 8K and 32K tokens; 6% more prefill time on the
  16 GB artifact like for like, 1 to 2% more time to first token served, where the two loads also differ in context
  (131,072 against 262,144) and window 0's own launch spread at 32K is 4%.
- **Committed tokens per second against window 0** are 2.1 to 2.3 times at windows 5 to 7 in every cost cell, on both
  legs. On the 16 GB leg windows 5, 6 and 7 come within 0.4% of each other (geometric means 2.218, 2.216, 2.224);
  window 7 is the most, so the default stays 7, now measured. On NVFP4 window 5 is the most, 3.5% above window 7.
- **The speed gate passes on both legs**, in all eight cells of each: MTP at its default window decodes 1.32 to 2.26
  times window 0's rate on NVFP4 and 1.27 to 2.86 times on 16 GB, every margin past the larger launch spread, the
  narrowest the 16 GB sampled 32K cell at two lanes (16.7 against 13.8 tokens per second).
- **The VRAM saved**: outside the KV pool, the MTP load holds 1,007,350,308 bytes less than the same DFlash2 load on the
  NVFP4 artifact at 8 lanes (631.6 MB of weights, 335.5 MB of drafter window in the lanes' state, 78.0 MB of drafter
  round, less 19.9 MB of decode graph and 17.8 MB of the head's residual window), and 0.803 GB less on the 16 GB one at
  2 lanes (52.4 MB of it the workspace). Against no speculation it holds 709.6 MB more on NVFP4 and 555.8 MB more on
  16 GB. In the 13 GiB budget the
  16 GB artifact loads 131,072 tokens of context under MTP, 65,536 under DFlash2, 262,144 with no speculation.
- **The workspace line**: on the NVFP4 load the image encoder's share sizes the arena, and it is 983,876,864 bytes
  under all three; on the 16 GB load the prefill's scratch sizes it, DFlash2's taps adding 92.3 MB to it and the head's
  prefill 39.8 MB.
- **The residual**: the process held 109.6 and 110.7 MB past its other lines under MTP at 8 lanes (WDDM's counter right
  after load, two launches), 117.0 MB under DFlash2, 66-68 MB with no speculation; at 2 lanes on 16 GB at most 45.7 MB
  under MTP, 64.5 MB under DFlash2, 32.6 MB with none. The plan's 124,474,800-byte line covers every one.
- **Quality, MTP on against off, passes**: greedy text equal up to the near-tie rule (the leaf's canaries at 1, 4 and 8
  lanes, the vision canaries, and the served canaries, three of whose NVFP4 divergences are near ties by the rule and
  the 16 GB ones none); the canary floor (96.0% against 95%, and sane, deterministic canaries on every launch); the
  regression set (every gate case, every probability the same); the loop measures (7 against 6 and 11 against 11 of
  32, p = 0.50 and 0.60).

Inferred:
- On the 16 GB artifact the window-7 round costs at most 0.5 ms more than the window-6 one, and its substrate costs less
  at 7 than at 5: the trellis matrices' verify kernels take 8 columns about as cheaply as 7. That flattens the 16 GB
  curve past 5; on NVFP4 each window costs about a chain step more, and the optimum is 5.
- The head's prefill cost is large for one layer: about 11 µs a token on NVFP4 and 17 µs on the trellis, where a target
  layer takes about 1 µs. Its matrices run in calls of at most 128 rows on the W8 routes of a decode (ADR 0095), not on
  the prefill's GEMMs; that is the likely reason, not profiled here.
- The derivation of tokens a round at a smaller window from window-7 records reads it 2 to 3.5% low (windows 4 to 6),
  because a walk at a smaller window starts more rounds inside a predictable run.

## Implications

- **The default draft window stays 7**, measured (`QWEN38_MTP_DEFAULT_DRAFT_TOKENS`): the 16 GB leg's windows 5 to 7
  are a tie within the measurement, and 7 is its point estimate. On an NVFP4 load `--spec-draft-tokens 5` is estimated
  to commit about 3.5% more, from the replay and the cost table, not served: window 7's own round cost moved 4 to 6%
  between its two cost launches at 64K tokens.
- **When to choose MTP**: where DFlash2's memory is what the context lacks. On the NVFP4 artifact at 8 lanes it frees
  about 1.0 GB against DFlash2, about 50,000 tokens of pool more (1,039,040 against 988,544) after its own 6% a token;
  in a 13 GiB budget the 16 GB artifact keeps twice DFlash2's context (131,072 against 65,536). It commits as many
  tokens a round as DFlash2's sampled draft and decodes about twice as fast as no speculation, but pays 28 to 40% of a
  round for its chain and up to 13% more time to first token; its decode rate against DFlash2's was not measured
  here, so DFlash2 stays the default where it fits.
- **The shortlist head under MTP: not now.** The full output head is about three quarters of a chain step on NVFP4
  (1.35 of 1.80 GB read) and about two thirds on the 16 GB artifact (0.80 of 1.25 GB). The shortlist head (4-bit over
  the 131,072 most frequent tokens, 0.36 GB) would cut a step by about 0.66 ms on NVFP4 and 0.29 ms on the 16 GB
  artifact, by bytes: at the default window a round would be 11% shorter on the 16 GB artifact at one lane short (7%
  at two lanes 64K deep), 17-21% on NVFP4 at one lane, before any acceptance it loses (DFlash2's shortlist lost 7.4%
  of its tokens a round). On the 16 GB artifact, the one MTP is for, the net is near zero, and the window is not
  limited there (7 is the most); on NVFP4, which has the memory for DFlash2, it is the full head that pulls the
  optimum to 5. If the owner opens it, its first step is offline: the shortlist's token set against these records'
  drafts and candidates prices the acceptance before any kernel.
- **For #629**: the round records, one per round, in `.scratch/618/628/replay/{nvfp4,16gb}-w7/` (read by
  `tools/mtp-replay/records.py`, manifest `.scratch/dflash2-replay-267/replay/input/manifest.json`), the direct
  window 4-6 replays of jobs 0-21 in `replay/subset-{leg}-w{4,5,6}/`, and the cost table above (JSON in
  `cost/{nvfp4,16gb}-table.json`, raw launches `cost/{leg}-L{1,2}.jsonl`). A round runs its window's chain steps and
  verify columns whatever its lanes' extents (#621), so an extent shorter than the window saves nothing unless the
  window itself moves; a policy is priced at the cost of the window it runs, and the head's overconfidence is part of
  the confidence policy's signal.
- **The 1.1 GB estimate** (spec 27b-16gb's phase 2 notes, "~1.1 GB fixed saved") counted DFlash2's weights and not the
  head's: measured, MTP saves 1.01 GB on the NVFP4 load at 8 lanes and 0.80 GB on the 16 GB load at 2 lanes, about
  0.74 GB at one lane, which is the spec's "about 0.7 to 0.8 GB"; and the head's KV layer costs 6% of the pool's
  tokens beside it.
- **The head's prefill** is a lever of its own (16-18% of an NVFP4 prefill), for a later ticket.

## Limits and unknowns

- One card, one desktop (about 2.6 GB held), Windows; the 16 GB leg is a 13 GiB budget on the RTX 5090, not a 16 GB
  card: a smaller card's bandwidth moves every cost, and its compute moves the trellis's.
- In the 13 GiB budget the three 16 GB loads are not the same load: each adapts its context to what its plan leaves
  (MTP 131,072, DFlash2 65,536, none 262,144). The lines outside the KV pool do not depend on the context (DFlash2's
  92.3 MB of taps is the spec's figure at a 1,024-token chunk); the pool, and the served time to first token, do.
- The replay is #267's 200 turns of one agent workload, sampled at temperature 1, all recorded on DFlash2's engine;
  greedy traffic accepts more (#625: 4.24 tokens a round on short greedy prompts).
- Tokens a round at windows 1 to 3 are derived only; at 4 to 6 corrected by one 15,179-token subset.
- The cost example's rounds are greedy, all lanes at one context, prefilled one after another; a launch read one load
  7-9% slow twice on NVFP4 (window 3 at 4K, window 2 at 64K).
- Served cells: the agent trace's 11 requests and 32K-token prompts at 1,024 tokens; eight-lane greedy cells change
  their text between launches (window 0 at 8 lanes: 57.7 and 48.0 tokens per second). The long cells' decode overlaps
  the other lanes' prefills.
- The loop leg ran with the loop guard off, as every leg here did, where ADR 0082 runs it at its default; its loops
  are the 6-gram and compression calls alone.
- DFlash2 was not served for speed here (the owner's cut, 2026-11-16): its decode against MTP's on this build is
  not measured.

## Follow-ups

- The adaptive extent, go or no-go: https://github.com/gpillon/ignis/issues/629, replayed offline in
  [an adaptive extent for the head](2026-11-16-an-adaptive-extent-for-the-27bs-mtp-head-replayed-offline.md).
- The shortlist head under MTP: the owner's to open (above).
- The head's prefill on the prefill GEMMs: the owner's to open.
