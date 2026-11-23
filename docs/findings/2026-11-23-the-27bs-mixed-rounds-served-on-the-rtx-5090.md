# The 27B's mixed rounds served on the RTX 5090 after spec #711: a JSON answer beside three conversations 14.5% faster together, every multi-lane cell faster or inside its spread, one lane the same text in the same rounds, and a sampled conversation cell 2.05% below past the gate's 2% line, a miss by the letter after which the owner made `mixed` the `rtx5090` profile's default

- Kind: experiment
- Status: current
- Observed: 2026-11-23
- Last verified: 2026-11-23
- Scope: decode / the 27B NVFP4 artifact's DFlash2 rounds at 15 drafts on the RTX 5090 (Windows, PCIe Gen 3), the
  build of spec #684 (main at ecbd1c7a) under `aggregate` against spec #711's integration build at bfbb66f9 under
  `mixed`; the cost tables' timing of #721 (2026-11-23); the prototype measurements behind the spec (2026-11-20)
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/722,
  https://github.com/gpillon/ignis/issues/711,
  https://github.com/gpillon/ignis/issues/721,
  https://github.com/gpillon/ignis/issues/684,
  [ADR 0106](../adr/0106-the-27bs-dflash2-verify-round-may-carry-lanes-of-both-shapes-and-a-mixed-round-is-held-to-its-base-rounds-numerical-class.md),
  [ADR 0104](../adr/0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md),
  [the 27B's draft window of 15 served](2026-11-20-the-27bs-draft-window-of-15-served-on-the-rtx-5090.md)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, 15 drafts, served vs aggregate: mixed +14.5% on a JSON answer beside three conversations (worst lane -7.9%), one lane the same text and rounds; a conversation cell -2.05%, yet the owner made mixed rtx5090's default.

## Question

Since spec #684 a DFlash2 verify round of the 27B at 15 drafts takes one **round shape** for all its lanes: the base
shape (8 columns a lane) or the extended one (16 columns a lane, at 1 to 4 lanes), chosen by the shape policy
`aggregate`. Beside a JSON answer that asks the extended shape, the conversations either pay for 16 columns each or
the JSON answer is refused them. Spec #711 lets a round carry lanes of both shapes, a **mixed round**, under a third
shape policy, `mixed`, which grants the extended shape lane by lane (ADR 0106). #722 asks, served on the RTX 5090:

- the threshold: does a JSON answer beside three conversations at four lanes commit at least 10% more tokens a second
  together under `mixed` than the build of #684 under `aggregate`, with no conversation more than 15% slower;
- the reported cells: a JSON answer, a moderately predictable answer and strict tools each beside a conversation; two
  JSON answers beside two conversations; a `response_format` json_schema answer beside three conversations;
- #684's regression line: no one-lane or free-text cell more than 2% below that build (a free-text cell below it only
  inside its spread), plain decode, the first token and the served context equal, the KV pages lower by the planned
  mixed graphs only;
- whether this build under `aggregate` still runs that build's shapes, whether a BF16 KV cache loads and runs mixed
  rounds, and what the cost table's and the prototype's measurements behind the grant were.

## Evidence

### Conditions

RTX 5090 under Windows (WDDM) on a PCIe Gen 3 x16 host, one card, one run on it at a time: each batch held the
repository's GPU lock for its whole length, and before every leg the card held no other ignis, ninfer or GPU-test
process and less than 8 GiB in use, no compiler ran at a batch's start, and every build process seen during a batch
was logged. The 27B NVFP4 artifact, `--model-exception-weights fp8`, hq-e8-2b KV, a context of 262,144 and rope none,
an empty config file (no setting from a file: every origin a flag, the `rtx5090` profile or a default), host pools
8 GiB and 2 GiB, the Playground off, idle unload off, the loop guard at its default (`rewind`) in both arms, as every
leg's configuration in force shows, `IGNIS_ROUND_TIMING=1`, no other `IGNIS_` variable, the server on its own ports.

The executables were frozen and their sha256 checked at every batch:

- **M**: main ecbd1c7a (the commit spec #711's integration branch last merged; main moved after it by test and lock
  commits only), `--spec-backend dflash2 --spec-draft-tokens 15`: `aggregate`, the cost table 1.128, 1.29, 1.363,
  1.344 at 1 to 4 lanes;
- **N**: the integration build bfbb66f9 (#712 to #721 and the review's fixes), the same flags and
  `--spec-shape-policy mixed`, the NVFP4 body's cost table of #721
  (`1.128/1.159,1.29/1.112,1.249,1.363/1.132,1.217,1.344,1.344/1.083,1.196,1.196/1.109,1.109/1`, the load log's);
- **A**: N's executable with M's flags (`aggregate`).

One server start a leg, legs mirrored (M N N M; M N A A N M where A runs), each leg opened by an untimed warm-up (one
lane, then two). The two M legs at a batch's ends are its **control**, main against main. A request's tok/s is
(tokens - 1) over the time from its first streamed chunk to its last; tokens and ms a round come from the server's
`ignis.request.done`; the shares of lane-rounds by shape from `/metrics` deltas
(`ignis_speculative_shape_rounds_total`, each lane of a round counted once, a lane of a mixed round as `mixed`).

A multi-lane cell (a **window**) runs the JSON lane or lanes greedy while each conversation lane runs a conversation
at temperature 0.6 (up to 1,200 tokens, thinking off) back to back until the last JSON request ends. The window is the
JSON lanes' decode span; the **aggregate** is the tokens every lane committed inside it over its length (a
conversation that straddles it prorated by time); a conversation's rate is that of its requests wholly inside the
window (the prototype's measure), and a conversation lane's **prorated** rate its tokens inside the window over the
window. A window whose conversation lane failed or ended early would be void; none was. A cell's **spread** is the
larger of the arms' leg ranges and the delta's 95% confidence interval (1.96 times the pooled standard error). The
prompts are #684's gate's: the JSON answer (greedy, 7,229 tokens), the conversation and prose, JSON-long (80
fictional employees), the json_schema answer (200 items, strict), code, the agent turn and its strict-tools form.

Eighteen batches, 08:24 to 11:05, about 150 minutes under the lock, each under 18 minutes: two smokes (void: their
server stop read the wrong process, fixed), a BF16 KV smoke, the threshold cell twice, the pairs, four lanes, the
regression line (seven batches) and its repeat, speculation off, the first token by depth, and the profile's own
load. No build ran during any batch, and every timed request's largest gap between two streamed tokens was under
125 ms. The raw legs, client records and summaries are kept locally, not in the repository.

### The threshold cell: a JSON answer beside three conversations, four lanes

Two batches, 3 windows a leg, 12 windows an arm (A in the first batch).

| batch | arm | aggregate tok/s (sd) [legs] | JSON lane tok/s | conversations inside (n) | conversation lanes 0 / 1 / 2 | shares of lane-rounds | JSON lane: tokens, ms a round |
|---|---|---|---|---|---|---|---|
| 1 | M | 835.2 (10.8) [830.2 / 840.2] | 397.9 | 147.0 (36) | 143.8 / 144.2 / 149.4 | base 72.3%, extended 27.7% | 9.46, 23.76 |
| 1 | N | 955.3 (4.9) [953.6 / 956.9] | 536.7 | 139.5 (18) | 140.8 / 137.4 / 140.3 | mixed 97.6%, base 1.5%, extended 0.8% | 13.36, 24.88 |
| 1 | A | 830.1 (6.1) [830.9 / 829.4] | 396.1 | 146.0 (36) | 147.7 / 141.6 / 144.8 | base 72.8%, extended 27.2% | 9.42, 23.77 |
| 2 | M | 826.8 (8.4) [824.9 / 828.6] | 393.4 | 145.9 (36) | 141.1 / 146.1 / 146.2 | base 74.8%, extended 25.2% | 9.31, 23.65 |
| 2 | N | 947.4 (7.1) [949.8 / 944.9] | 536.5 | 137.6 (18) | 137.7 / 135.3 / 137.9 | mixed 97.6%, base 1.6%, extended 0.9% | 13.36, 24.90 |

| | batch 1 | batch 2 | pooled (12 windows an arm) |
|---|---|---|---|
| aggregate, N against M | **+14.37%** (CI +-1.13%) | **+14.59%** (CI +-1.06%) | **+14.48%** (CI +-0.85%, spread 1.85%) |
| control M2 - M1 | +1.20% | +0.46% | |
| JSON lane | +34.9% | +36.4% | +35.6% |
| conversations inside, pooled | -5.0% | -5.7% | **-5.4%** |
| conversations inside, by lane | -3.6 / -3.2 / -8.1% | -1.3 / -7.8 / -7.8% | -2.5 / -5.6 / **-7.9%** |
| conversation lanes prorated, pooled (by lane) | -4.3% | -5.2% | -4.7% (-2.3 / -6.0 / -5.9%) |

- The JSON lane commits 13.36 tokens a round beside three conversations under `mixed`, against 9.39 under `aggregate`
  and 13.41 alone: it drafts 15 at four lanes. The conversations commit 3.43 tokens a round against 3.47.
- On the GPU clock (`IGNIS_ROUND_TIMING`, medians of 256-round blocks): N's mixed rounds 24,521 to 24,591 us; M's base
  rounds at four lanes 21,565 to 21,634 us and its uniform extended ones 28,064 to 28,314 us. M's two legs' base
  periods agree within 0.3% in each batch.
- Every N answer parsed (12 of 12); one M answer of 12 stopped at 7,228 tokens with its last delimiter missing
  (`Expecting ',' delimiter` at its 18,477th and last character), a greedy multi-lane text of that start.

### The reported cells

| cell | lanes | arm | n | aggregate tok/s (sd) [legs] | JSON lane(s) tok/s | conversations inside (n) | shares of lane-rounds | N against M |
|---|---|---|---|---|---|---|---|---|
| JSON answer + conversation | 2 | M | 6 | 780.2 (6.8) [782.0 / 778.4] | 617.1 | 164.7 (6) | extended 98.1% | |
| | | N | 6 | 844.2 (5.1) [846.6 / 841.7] | 671.3 | 176.4 (6) | mixed 92.5%, extended 5.7% | **+8.20%** (CI +-0.87%); JSON +8.8%, conversation +7.1% |
| JSON-long + conversation (moderately predictable) | 2 | M | 6 | 537.0 (11.9) [534.3 / 539.6] | 348.1 | 191.6 (18) | base 76.3%, extended 23.7% | |
| | | N | 6 | 561.8 (12.5) [551.9 / 571.8] | 380.9 | 185.0 (15) | base 50.1%, mixed 46.6% | **+4.63%** (CI +-2.58%, spread 3.69%); JSON +9.4%, conversation -3.4% |
| strict tools (greedy, thinking on) + conversation | 2 | M | 6 | 508.5 (12.2) [509.5 / 507.5] | 323.4 | 192.5 (6) | base 78.6%, extended 21.4% | |
| | | N | 6 | 517.3 (32.3) [509.9 / 524.7] | 335.0 | 190.3 (6) | base 72.8%, mixed 25.3% | +1.73% (CI +-5.43%, inside); tool lane +3.6%, conversation -1.1% |
| two JSON answers + two conversations | 4 | M | 6 | 1,151.5 (6.7) [1,152.0 / 1,150.9] | 456.0 / 462.2 | 126.8 (12) | extended 91.7% | |
| | | N | 6 | 1,237.7 (6.2) [1,239.8 / 1,235.6] | 491.0 / 501.6 | 127.0 (12) | mixed 97.3% | **+7.49%** (CI +-0.63%); JSON +7.7% / +8.5%, conversations +0.2% |
| json_schema + three conversations | 4 | M | 6 | 861.5 (9.3) [863.6 / 859.4] | 482.4 | 126.4 (53) | extended 78.7% | |
| | | N | 6 | 976.3 (8.4) [980.2 / 972.4] | 577.0 | 134.7 (46) | mixed 98.8% | **+13.33%** (CI +-1.16%); json_schema +19.6%, conversations +6.6% |

- Controls (M2 - M1): -0.47%, +1.00%, -0.38% (the pairs), -0.09%, -0.48% (four lanes).
- At two lanes `aggregate` runs the JSON answer's pair at the uniform extended shape (32 columns) 98% of the time; the
  mixed round holds 24, so both lanes gain. On the GPU clock N's two-lane mixed rounds took 19,723 and 19,742 us, M's
  two-lane uniform extended rounds 21,605 and 21,625 us.
- Every answer was well formed in both arms: JSON 6 of 6 and 12 of 12, json_schema 6 of 6 valid (200 items), the
  strict tool calls' arguments parsed in 6 of 6 under N and M, 25.3% of that pair's lane-rounds mixed under N; no
  structured repair was counted in any of these legs (`ignis_structured_repairs_total` 0), and the grammar's cuts of
  a round's run are not counted served. A's strict-tools pair parsed 5 of 6.
- At four lanes beside the json_schema answer `aggregate` ran every lane extended in 78.7% of its lane-rounds; the
  mixed round keeps the conversations at 8 columns, so they too are faster.

### This build under `aggregate` (A against M)

| cell | A against M | shares of lane-rounds, A / M | split rounds, A / M |
|---|---|---|---|
| JSON answer + three conversations (6 windows an arm) | -0.61% (CI +-1.19%) | extended 27.2% / 27.7% | 97.4% / 97.7% |
| JSON answer + conversation | -0.72% (CI +-0.87%) | extended 98.1% / 98.1% | 86.2% / 84.9% |
| JSON-long + conversation | +0.32% (CI +-1.96%) | extended 25.5% / 23.7% | 53.1% / 52.5% |
| strict tools + conversation | -1.10% (CI +-4.62%) | extended 21.0% / 21.4% | 42.0% / 42.4% |

A runs no mixed round and its shares sit within 1.8 points of M's, the multi-lane greedy texts differing between
starts as they do between two starts of one build.

### The regression line (one lane)

Each cell in M N N M batches, 10 requests a leg (the greedy cells 3, json_schema-d 2); the conversation and prose at
temperature 0.6 ran twice, the second batch decided before it ran (below). Tokens and ms a round from the done lines,
the extended share from `/metrics`.

| cell | n an arm | M tok/s: mean (sd), legs | N tok/s: mean (sd), legs | change | 95% CI | spread | control M2-M1 | tokens a round M -> N | ms a round M -> N |
|---|---|---|---|---|---|---|---|---|---|
| conversation, T 0.6, first | 20 | 222.56 (14.1), 222.85 / 222.27 | 221.97 (14.0), 221.56 / 222.38 | -0.26% | +-3.90% | 3.90% | -0.26% | 3.460 -> 3.456 | 15.52 -> 15.54 |
| conversation, T 0.6, repeat | 20 | 228.48 (12.7), 226.21 / 230.75 | 219.81 (13.6), 219.09 / 220.53 | -3.79% | +-3.57% | 3.57% | +2.01% | 3.553 -> 3.416 | 15.53 -> 15.51 |
| **conversation, T 0.6, pooled** | 40 | 225.52 (13.6) | 220.89 (13.6) | **-2.05%** | +-2.64% | 3.76% | | 3.507 -> 3.436 | 15.52 -> 15.53 |
| prose, T 0.6, first | 20 | 175.38 (4.4), 174.88 / 175.89 | 171.84 (5.8), 172.43 / 171.24 | -2.02% | +-1.83% | 1.83% | +0.58% | 2.697 -> 2.648 | 15.36 -> 15.39 |
| prose, T 0.6, repeat | 20 | 176.83 (5.9), 178.94 / 174.72 | 175.42 (5.6), 172.85 / 178.00 | -0.80% | +-2.02% | 2.91% | -2.36% | 2.720 -> 2.699 | 15.36 -> 15.36 |
| **prose, T 0.6, pooled** | 40 | 176.11 (5.2) | 173.63 (5.9) | **-1.41%** | +-1.39% | 3.83% | | 2.709 -> 2.673 | 15.36 -> 15.38 |
| conversation-d | 20 | 222.04 (11.3), 221.68 / 222.41 | 220.96 (18.1), 218.52 / 223.41 | -0.49% | +-4.21% | 4.21% | +0.33% | 3.449 -> 3.432 | 15.52 -> 15.52 |
| prose-d | 20 | 186.53 (6.3), 191.08 / 181.97 | 186.01 (6.2), 186.31 / 185.71 | -0.28% | +-2.07% | 4.88% | -4.77% | 2.873 -> 2.867 | 15.39 -> 15.40 |
| code-d (two batches) | 20 | 233.93 (11.6), 236.90 / 237.93 / 224.81 / 236.09 | 228.21 (9.6), 225.46 / 221.58 / 228.42 / 237.38 | **-2.45%** | +-2.83% | 6.76% | +0.43%, +5.02% | 3.790 -> 3.688 | 16.27 -> 16.23 |
| agent-d | 20 | 318.73 (53.5), 337.79 / 299.66 | 314.07 (46.1), 320.80 / 307.34 | -1.46% | +-9.71% | 11.97% | -11.29% | 5.463 -> 5.417 | 17.20 -> 17.38 |
| JSON answer-d | 20 | 759.86 (10.2), 763.39 / 756.32 | 758.71 (9.6), 759.79 / 757.63 | -0.15% | +-0.81% | 0.93% | -0.93% | 13.372 -> 13.367 | 17.59 -> 17.61 |
| json_schema-d (reported, not judged) | 4 | 506.34 (55.0), 551.75 / 460.94 | 525.03 (64.0), 530.85 / 519.21 | +3.69% | +-16.34% | 17.94% | -16.46% | 8.368 -> 9.118 | 17.01 -> 17.17 |
| json_schema, greedy | 6 | 828.95 (0.8) | 828.17 (0.8) | -0.09% | +-0.11% | 0.11% | -0.04% | 14.832 -> 14.832 | 17.89 -> 17.91 |
| JSON answer, greedy | 6 | 767.69 (0.5) | 767.37 (0.3) | -0.04% | +-0.06% | 0.06% | -0.04% | 13.414 -> 13.414 | 17.47 -> 17.47 |
| strict tools, greedy | 6 | 443.69 (1.1) | 443.08 (1.0) | -0.14% | +-0.27% | 0.27% | +0.21% | 8.038 -> 8.038 | 18.28 -> 18.31 |

- **The same request, seeded** (seed 722, 2 a leg, after the repeat's 20): under M and N each wrote the same text in
  the same number of rounds, every request: prose 1,145 tokens in 423 rounds (176.0 tok/s against 176.0), the
  conversation 1,084 in 312 (223.4 against 223.2), code 6,144 in 1,457 (257.1 against 256.9), each N slower by 0.02
  to 0.11%. The greedy cells wrote one text an arm, the same under both (json_schema 14,639 tokens, valid in 6 of 6;
  the JSON answer 7,229; strict tools 3,221, its calls parsing in 6 of 6), the same tokens a round, N's rounds 0.04 to
  0.14% longer.
- Main against main (the control) moved more than 2% on five sampled cells: prose-d -4.77%, code-d +5.02% (its second
  batch), agent-d -11.29%, the repeat's conversation +2.01% and prose -2.36%. The M legs' GPU round periods agreed
  within 0.3% in every batch, so the card did not drift: those are the texts' sampling.
- The second batch of the conversation and prose was decided after the first read prose at -2.02%, before it ran:
  one repeat of that batch, its cells read pooled; code-d's own repeat was not run (below).
- JSON answer-d parsed 18 of 20 under M and 20 of 20 under N; agent-d's calls parsed 15 of 20 and 18 of 20, each
  failure a `write_file` cut by the 4,096-token cap.

**Plain decode** (`--spec-backend off` on both executables): the 8,192-token prompt's answer, greedy (the same text
in all four legs), 83.60 against 83.58 tok/s (-0.02%, CI +-0.03%, control +0.02%); the conversation at temperature
0.6, 3 a leg, +0.03%.

**The first token** (a cold prompt of 16,676 tokens, one token, 3 a leg after the warm-up in the eight speculative
batches of the regression line and its repeat, and the depth batch's 16K cell: 18 legs an arm): N later by 7.1 ms
(+0.60%), leg-level 95% CI +-4.9 ms; the server's own `duration_ms` over the eight batches +3.7 ms (CI +-3.0). The
depth batch alone (4 a depth a leg, M N N M):

| depth | M (legs) | N (legs) | change | 95% CI |
|---|---|---|---|---|
| 4K | 0.3408 s (0.3422 / 0.3394) | 0.3412 s (0.3465 / 0.3359) | +0.12% | +-3.52% |
| 16K | 1.1731 s (1.1703 / 1.1759) | 1.1830 s (1.1805 / 1.1855) | +0.84% | +-1.14% |
| 64K | 5.8616 s (5.8539 / 5.8693) | 5.8639 s (5.8579 / 5.8698) | +0.04% | +-0.27% |

With speculation off the cold 16K prompt read +1.33% (2 an arm).

### Context and pages

Every speculative load held 262,144 tokens of context. Each load planned its KV pool from the budget its start read
(29,126 to 29,192 MB, from the free memory at its start: 3,147 to 3,211 MiB in use on the desktop before each load).
Against M's pool at the same budget (a line fitted through M's 26 speculative loads, each within 0.9 page of it), every
one of N's 26 speculative loads held a pool 68.8 to 69.5 MB smaller, 116.7 to 117.8 pages of 589,833 bytes: the
plan's `mixed_graphs` line, 69,206,016 bytes (117.3 pages), within a page. A's four loads held M's pool within 0.4
page. With speculation off both executables planned the same pages at the same budget (18,155 against 18,152 and
18,156). Read by the WDDM per-process counter after the load, the process held 0.7 to 2.7 MB more than its plan under
M, 5.1 to 6.8 MB under A and 7.1 to 9.0 MB under N, out of the headroom.

### A BF16 KV cache under `mixed`

One N leg with `--model-kv-format bf16` and a 131,072-token context (2,213 pages, 65,536 bytes a token): the load ran
`shape_policy=mixed` and planned the `mixed_graphs` line (69,206,016 bytes); a JSON answer beside three
conversations ran 97.7% of its lane-rounds mixed (the JSON lane 13.31 tokens a round, the conversations 3.41) and
the json_schema answer beside a conversation 90.0%; the leaf logged mixed-round timing blocks and no capture error;
the JSON answer parsed and the json_schema answer was valid. The process held 9.9 MB more than its plan. No timing
verdict.

### The `rtx5090` profile's own load under `mixed`

One N leg with the profile's own values (no context, rope, exception-weights, KV-format or host-pool pin;
`--spec-backend dflash2 --spec-draft-tokens 15 --spec-shape-policy mixed`): 917,504 tokens held unadapted at YaRN
3.5 with vision on, 30,382,579,712 bytes free at start (3,145 MiB in use on the desktop), a budget of 29,124,288,512
bytes and 15,108 KV pages (966,912 tokens), the `mixed_graphs` line planned; a JSON answer beside three conversations
ran 97.0% of its lane-rounds mixed and parsed. No timing verdict.

### The cost table (#721)

The defaults of `spec.shape_period_ratios` were measured on the RTX 5090 through the round identity test's timing:
forced rounds of every cell, 600 rounds a cell, each timed lane prefilled to 3,584 tokens and ridden 1,536 more (a
served context), the model loaded at 262,144 tokens, four legs a body in the orders forward, reverse, reverse,
forward, each cell the median of its eight 256-round blocks on the GPU clock. On the NVFP4 body (FP8 exception
weights):

| lanes | base round (us) | entries, 1, 2, ... lanes extended (measured where held) |
|---|---|---|
| 1 | 15,471 | 1.128 (1.125) |
| 2 | 17,186 | 1.159, 1.29 (1.295) |
| 3 | 20,101 | 1.112, 1.249, 1.363 (1.352) |
| 4 | 22,422 | 1.132, 1.217, 1.344 (1.345), 1.344 (1.342) |
| 5 | 25,485 | 1.083, 1.196, 1.196 (1.188) |
| 6 | 27,592 | 1.109, 1.109 (1.106) |
| 7 | 30,647 | 1 (0.999) |

- The uniform entries are #690's served 1.128, 1.29, 1.363 and 1.344, which this ride read within 0.011 (1.125,
  1.295, 1.352, 1.342), so `aggregate` reads what it read before.
- Four mixed entries are held at a bound (ADR 0106): 3+1 at the uniform 1.344 (measured 1.345), 3+2 at 2+3's 1.196
  (1.188), 2+4 at 1+5's 1.109 (1.106), 1+6 at 1 (0.999). On this card the step from 56 columns to 64 costs nothing or
  less: the 56-column rounds ran at 2,771-2,781 MHz and the 64-column ones at 2,812-2,817 MHz, the card at its power
  limit (460-515 W mean, the power cap the clock reason in 31-80% of samples from two lanes up).
- Per leg every entry moved at most 0.007; dropping any one leg moves none more than 0.0017.
- Each mixed round costs within 2.5% of a base round of the same columns (0.978 to 1.012 on this body).
- The trellis body's table (1.364; 1.214, 1.407; 1.162, 1.502, 1.629; 1.291, 1.391, 1.55, 1.606; 1.082, 1.204, 1.243;
  1.117, 1.154; 1.036), the same method, holds no entry; its rounds run mixed only where a load names `mixed`.
- The mixed graphs' build (one root, the argmax zeroed by a kernel, the extended group at the greatest stream
  priority) against no build, 1,100 rounds a ride, legs on, off, off, on, on, off: 1+1 19,285 against 19,545 us, 1+3
  23,752 against 24,023, 2+2 25,314 against 25,603, 1+5 28,293 against 28,583 (260 to 290 us faster, every leg with
  the build below every leg without it; the uniform rides, which the build does not touch, 21 and 26 us apart), the
  same texts.

### The prototype (the spec's measurements, 2026-11-20)

Before the grant existed a prototype granted every ask in the caller's order, served against the build of #684 with
`aggregate` (legs mirrored, the #696 gate's flags plus a 4 GiB headroom): a JSON answer beside three conversations
+14.65% and +11.80% together in two clean batches (+13.1% pooled; the JSON lane +32 to +34%, the conversations -3.6%
and -8.0%); a JSON answer beside a conversation +6.5% (JSON +7.0%, conversation +8.7%); two JSON answers beside two
conversations +5.6%; a moderately predictable answer beside a conversation with every ask granted +0.6%, inside the
spread (JSON +3.4%, conversation -5.0%); one lane unchanged. Its round periods (GPU clock, us): at two lanes base
16,529, mixed 1+1 19,091 (with the graph build), uniform extended 21,349; at four lanes base 21,505, 1+3 24,467, 2+2
26,275, uniform extended 29,111; at six lanes base 26,574, 1+5 29,039.

## Finding

Observed:

- **The threshold passes**: a JSON answer beside three conversations at four lanes commits **14.48%** more tokens a
  second together under `mixed` than the build of #684 under `aggregate` (+14.37% and +14.59% in two batches, CI
  +-0.85% pooled), the JSON lane 35.6% more; the slowest conversation lane is 7.9% slower (the bound 15%), the
  conversations 5.4% slower pooled.
- Every reported cell is faster or inside its spread: a JSON answer beside a conversation +8.2% with both lanes
  faster, a moderately predictable answer +4.6%, two JSON answers beside two conversations +7.5%, the json_schema
  answer beside three conversations +13.3% with its neighbours faster too, strict tools beside a conversation +1.7%
  (inside its spread), every answer well formed.
- **The regression line misses by the letter of its 2% line**: the conversation at temperature 0.6 reads -2.05% over
  40 requests an arm (-0.26% and -3.79% in its two batches) and code at the model card's sampling -2.45% over 20; the
  other one-lane cells sit within 1.5% of the build of #684, the greedy ones within 0.14%. A cold 16K prompt's first
  token comes 7.1 ms (0.60%) later, outside its CI (+-4.9 ms), at 4K and 64K within 0.12%.
- The same request, seeded, writes the same text in the same rounds under both builds, sampled or greedy, at a rate
  within 0.02 to 0.14%: one lane runs the same rounds, each about 0.1% longer under this build.
- Plain decode and the served context are equal; the KV pages are lower by the planned mixed graphs alone (116.7 to
  117.8 pages against the line's 117.3); this build under `aggregate` plans and decodes as the build of #684 (its
  shares within 1.8 points, its pool within a page).
- A BF16 KV cache loads under `mixed`, runs mixed rounds and writes valid answers, and the `rtx5090` profile's own
  load holds its 917,504 tokens with the mixed graphs planned.

Inferred:

- The sampled cells' readings below the build of #684 are the sampling of different texts, not the rounds: seeded,
  the two builds write the same texts in the same rounds, and main against itself moved more than 2% on five of
  these cells. 20 to 40 unseeded requests an arm cannot resolve a 2% line on them; seeded pairs can.
- The first token's 7 ms looks like this build's, not `mixed`'s: with speculation off it read +1.3% (two requests an
  arm), and no mixed round runs in a prefill.
- The served round costs what the cost table says: N's mixed rounds at four lanes, most of them 1+3, ran 1.13 times
  M's base rounds of four lanes on the GPU clock (24,521 to 24,591 against 21,565 to 21,634 us), the table's 1.132
  for 1+3, while the JSON lane committed 13.4 tokens a round where `aggregate`'s mix of base and uniform extended
  rounds gave it 9.4.
- The grant's refusal pays on a moderately predictable answer: the prototype, granting every ask, gained 0.6% there;
  the grant, refusing some (46.6% of lane-rounds mixed), gained 4.6%.

## Implications

- By #722's rule a miss left the default to the owner, who made `mixed` the `rtx5090` profile's default after the
  miss by the letter (2026-11-23, #722): one lane forms no mix, and the seeded pairs wrote the same text in the same
  rounds under both builds, so the one-lane readings are the build's cost or the texts' sampling, not the policy's.
  Every other profile keeps `aggregate`, and under `rtx5090`
  `--spec-backend dflash2 --spec-draft-tokens 15 --spec-shape-policy aggregate` gives the uniform rounds back
  ([speculative decoding](../user/concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes)).
- A gate that holds one-lane sampled cells to a 2% line needs seeded pairs, or many more requests than 20 an arm: at
  20 to 40, main against main moves past 2%.
- A load under `mixed` gives up 117 KV pages (about 7,500 tokens) of the `rtx5090` pool for the mixed graphs, which
  the profile's 917,504 tokens hold with 66 MiB more free at start.
- The process holds a few MB its plan does not count, more under this build than under #684's (A 5-7 MB, N 7-9 MB,
  M 1-3 MB, N on BF16 KV 9.9 MB): out of the headroom. This build's first-token delay and its 0.1% longer one-lane
  round ship with it under either policy, if they are the build's.

## Limits and unknowns

- One machine: an RTX 5090 under Windows (WDDM) on a PCIe Gen 3 host; two legs an arm a cell (four on the threshold
  cell and the repeated cells), 12 windows an arm on the threshold cell, 6 on each reported cell, 20 to 40 requests an
  arm on the sampled one-lane cells, 6 on the greedy ones (each repeating one text), 4 seeded requests an arm.
- code-d's reading rests on 20 requests an arm; its repeat was not run, the seeded code request standing for it.
- The gate pinned a 262,144-token context and rope none; the profile's own load was one untimed leg.
- Not served: five to seven lanes (mixes 1+4 to 1+6, 2+3, 2+4, 3+2), a vision request in a mixed round, sampled JSON
  beside conversations, the trellis artifacts under `mixed`, BF16 exception weights, any other card. Their
  correctness is the round identity and numerical-class tests' (ADR 0106).
- Multi-lane greedy texts vary between starts of one configuration, so the windows compare texts that differ between
  legs.
- Nine of the ten unseeded sampled one-lane readings fell below the build of #684, in tokens a round as well, a
  pattern independent sampling noise gives about one time in a hundred; the seeded pairs, which wrote the same texts
  in the same rounds, are the evidence the one-lane rounds are the same, and the pattern is not explained.
- The first token's delay and the one-lane round's 0.1% were not timed under this build with `aggregate`, so whose
  they are (the build's or the policy's) is inferred.
- The cost table is a figure at about 3.5-5K tokens of context, and the ratio grows with context (ADR 0106).
- The served `/metrics` count lane-rounds by shape, not rounds by mix: the shares of mixes (1+3 against 2+2 at four
  lanes) were not read served.
- The BF16 KV smoke was one leg at 131,072 tokens and timed nothing.

## Follow-ups

- A regression line read from seeded pairs.
- The process VRAM over its plan named above.
- Not pursued, the owner's call on 2026-11-23, and no issue: whose the first token's 7 ms and the one-lane round's
  0.1% are, the build's or the policy's.
- The RTX PRO 6000 and the 16 GB and 24 GB cards, before their profiles' policies move.
