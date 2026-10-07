# Flash-Next's MTP speculation keeps the text but gains 1.02-1.30× at one lane and loses at two and three: decode is bound by expert copies, not by the round

- Kind: experiment
- Status: current
- Observed: 2026-10-06
- Last verified: 2026-10-07
- Scope: Flash-Next speculative decoding (spec flash-next/07 phases C-D): the verify round's and the MTP head's correctness on the card, tok/s at one to three lanes against spec-off, the expert traffic per round
- Related: https://github.com/gpillon/ignis/issues/307, [spec 07](../specs/flash-next/07-mtp-speculation.md), [MTP phase A](2026-10-06-flash-next-mtp-phase-a.md), [Flash-Next decode round](2026-10-06-flash-next-decode-round.md), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md)
- Superseded by: none

## Question

Phases C and D built the verify round on Flash-Next's state and the MTP head
drafting inside it. Does greedy speculation keep the spec-off text, and does
it reach spec 07's AC6: one-lane decode ≥ 1.25× spec-off, two and three lanes
no worse than spec-off minus 2%?

## Method

- **Correctness** (`crates/core/tests/flash_next_speculative_gpu.rs`, the real
  artifact, G1 prompts and a 2,600-token sparse prompt, BF16 and hq-e8-2b KV,
  one and three lanes, 40 tokens):
  - a verify-only load at 3 draft tokens runs four fake drafters -- none
    (extent 0), reject, random and oracle (the none run's own text) -- and
    holds the committed text and the next step's logits (one more token
    prefilled at each lane's frontier, through every state section) to each
    other bit for bit;
  - an MTP load (the 3.0-bit companion) runs its head's drafts against the
    same load's draft-free run, and both against spec-off one-token rounds
    under the near-tie rule.
- **Speed** (`crates/core/examples/flash_next_mtp_bench.rs`, release build,
  hq-e8-2b, captured graphs, a fixed 16 GB expert cache, 384 greedy tokens
  after 1,536-token reference windows -- two code, two prose -- at one lane,
  two pairs at two lanes, three windows at three lanes, and the 24,576-token
  code and prose documents at one lane). Each round's wall time includes the
  host's n-gram staging; the first 4 rounds are dropped. The residency's
  decode counters are read before and after each set.

## Results

**Correctness.** Every drafter commits the same text and leaves the same
state bit for bit, in both KV formats, dense and sparse, at one and three
lanes; the oracle's drafts are all accepted (30/30 at one lane, 60/60 at
three). The head's drafts commit the draft-free text bit for bit in BF16 and
the same text in hq-e8-2b. Against spec-off one-token rounds the only
divergence is one exact tie (gap 0).

**Speed**, MTP at 2 draft tokens (k = 2 at one and two lanes, 1 at three) and
the decode residency per round:

| set | spec-off tok/s | MTP tok/s | × | MTP acceptance (pos 1 / 2) | off: misses, MB copied / round | MTP: misses, MB / round |
|---|---:|---:|---:|---|---|---|
| 1 lane code | 75.1 | 87.7 | 1.17 | 0.72 / 0.72 | 76, 117 | 213, 237 |
| 1 lane code | 74.5 | 83.8 | 1.12 | 0.76 / 0.72 | 79, 117 | 251, 261 |
| 1 lane prose | 91.7 | 110.1 | 1.20 | 0.63 / 0.56 | 40, 76 | 84, 127 |
| 1 lane prose | 87.1 | 88.5 | 1.02 | 0.57 / 0.40 | 47, 86 | 126, 163 |
| 1 lane code 24K | 73.7 | 83.8 | 1.14 | 0.79 / 0.67 | 78, 118 | 251, 267 |
| 1 lane prose 24K | 80.0 | 103.8 | 1.30 | 0.78 / 0.74 | 64, 103 | 180, 206 |
| 2 lanes code+prose | 104.1 | 88.4 | 0.85 | 0.74 / 0.72 | 142, 180 | 494, 464 |
| 2 lanes code+code | 86.4 | 80.1 | 0.93 | 0.82 / 0.76 | 209, 233 | 697, 609 |
| 3 lanes | 89.4 | 76.1 | 0.85 | 0.75 / 0.84 | 348, 345 | 756, 666 |

With `--draft-rows 3` (k = 2 at one lane, no drafting from two lanes on) the
two- and three-lane sets run 101.9, 85.4 and 88.4 tok/s: −2.1%, −1.2% and
−1.1% against spec-off, the cost of the head's entries every one-token round
still writes. At one lane, k = 1 and k = 3 measured 1.08-1.09× on the first
code window (k = 3: 34.8 ms a round, of which the head's drafting is 2.2 ms).

## Interpretation

- **Speculation keeps the text, and the state machinery is exact.** A
  rejected column is a column never drafted, on every state component.
- **The round is bound by expert copies, not by its rows.** Spec-off copies
  ~76-118 MB of experts over the 12 GB/s link per token at one lane (most of
  a 11-13 ms round). A verify round's extra columns bring their own experts:
  the MTP rounds copy 64-107 MB per committed token against spec-off's
  76-118, so bytes per token drop only 10-15% and speculation mostly
  amortizes the compute between copies. At
  two and three lanes the link is already the bound, and the wider rounds
  lose. *(Inference from the counters: the stall time itself is not timed,
  `stall_nanos` stays 0.)*
- **Phase A's column cost was a warm-cache number.** Its ~2 ms per verify
  column came from rounds repeated over the same positions, whose experts
  stayed resident; with fresh tokens a column costs ~5 ms here (k = 1 with
  the oracle's drafts, BF16 KV: 19.4 ms a round, against hq-e8-2b spec-off's
  13.9). The projection built on it (1.5-1.7×) did not hold.
- AC6 is **not met**: one lane gains 1.02-1.30× (mean ~1.16×), two and three
  lanes lose unless drafting is off there.

## The head's chain read rejected columns (2026-10-07)

The head's chain steps ran before the commit's restore, so they read the
alignment's rejected columns from the head's own section: under hq-e8-2b as
ring rows of positions q − 512 + i (a ring slot carries no position), and in
both formats through the indexer tail a completed block pools from. Fixed by
`verify::restore_head` before the chain (spec 07, as built). Before the fix,
none and reject drafters committing the same text left the head drafting
differently on 1 of 64 rounds (hq, 1,536-token prompt; the test
`the_heads_drafts_never_read_a_rejected_column`). Bench, k = 3, one lane,
256 tokens after each 1,536-token window, tok/s and acceptance per position:

| window | hq before | hq after | BF16 after |
|---|---|---|---|
| code | 79.8, .761/.848/.786 | 83.1, .730/.846/.818 | 94.3, .872/.866/.879 |
| code | 72.7, .805/.826/.719 | 78.3, .857/.792/.719 | 83.0, .889/.778/.836 |
| prose | 77.9, .607/.541/.487 | 87.3, .621/.494/.432 | 83.2, .598/.589/.372 |
| prose | 68.2, .585/.421/.500 | 77.4, .636/.494/.474 | 82.5, .661/.538/.381 |

The four windows took 432 rounds before and 425 after (BF16: 405), and the
same acceptance measured 84.6/77.1/87.7/73.4 tok/s on 2026-10-06, so the
gain is within run-to-run noise: **nil**. The fix is correct but buys no
speed, and hq's lower code acceptance against BF16 is not this defect.

## Follow-ups

- The owner's call (spec 07: under 1.25× phase D stops and the owner
  decides), made 2026-10-07: MTP is off by default and `--spec mtp` turns it
  on. A card that holds every expert in VRAM (no PCIe misses, phase A's
  setting) is where it should pay; measure it there before changing the
  default.
- The lever is the expert traffic, not the verify round: a larger expert
  cache (the bench's 16 GB is the served default's size), a hit rate the
  verify rows' experts help (a prefetch budget that scales with the round's
  rows), or fewer bytes per expert.
