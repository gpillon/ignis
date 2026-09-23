# The sub-cell reading holds through `/v1/decide`: spec 15's acceptance

- Kind: experiment
- Status: current
- Observed: 2026-09-23
- Last verified: 2026-09-23
- Scope: serving / `/v1/decide` `point` and `box`, the anchored head set read
  inside the image cell (ADR 0040); hq-e8-2b KV
- Related: `docs/specs/decide/15-the-head-box-reads-inside-the-cell.md` (#264),
  ADR 0040, `2026-09-23-the-head-box-is-quantised-to-the-cell.md` (why),
  `2026-09-23-the-head-set-holds-through-decide.md` (spec 14's acceptance,
  the run this is measured against),
  `crates/server/tests/decide_point_acceptance_gpu.rs` (the run),
  `crates/server/tests/attention_readout_gpu.rs` (leaf against tap),
  `crates/server/tests/attention_set_cost_gpu.rs` (cost),
  `crates/core/tests/anchored_reading.rs` (the host rule against the
  reference), `.scratch/subcell-264/acceptance/` (raw, on disk)
- Superseded by: none

## Question

Spec 14's head box lost exactly one cell of its acceptance — 1024 px
buttons, 102 of 240 against the chain's 190 — and by the rule written before
that run, `box`'s default stayed on the chain. Spec 15 changes two things
about how the same prefill is read: each head sits at its **sub-cell peak**
instead of its cell's centre, and the flat half-cell growth becomes an
**allowance** that falls with the distinct cells the heads occupy. Does the
shipped path clear the floors spec 15 wrote before any run, and does `box`'s
default move?

## Pre-registration

In spec 15 (committed before this run), four sets through `/v1/decide`,
served artifact, hq-e8-2b with the residual window, `point` with no
`method`. The head box is floored on **every** set this time, because that is
what the spec changes:

| set | command | `point` inside | head `box` IoU >= 0.5 |
|---|---|---|---|
| F1 buttons, 1024 px | `scenes.py --varied --seed 20260950` (240) | >= 225 | >= 180 |
| F2 buttons, 4096 px | `scenes.py --varied --side 4096 --seed 20260951` (240) | >= 230 | >= 230 |
| F3 rectangles, 1024 px | `rectangles.py --seed 20260952 --n 120` | >= 108 | >= 108 |
| F4 rectangles, 4096 px | `rectangles.py --side 4096 --seed 20260953 --n 60` | >= 54 | >= 48 |

`box`'s default becomes `head` if and only if the head box's IoU >= 0.5 rate
is at least the chain box's on **every** set — spec 14's rule, unchanged.

## Result

**Every floor holds. `box`'s default does not move.**

| set | `point` inside: set / head alone / chain | head `box` / chain `box` IoU >= 0.5 | spec 14's head box, same regime |
|---|---|---|---|
| F1 | **236** / 222 / 211 of 240 | **183** / 190 | 102 of 240 |
| F2 | **240** / 236 / 194 of 240 | **237** / 126 | 238 of 240 |
| F3 | **120** / 77 / 118 of 120 | **118** / 83 | 111 of 120 |
| F4 | **60** / 55 / 57 of 60 | **60** / 41 | 53 of 60 |

Median distance from the target's centre, per diagonal — set / head alone /
chain: F1 0.039 / 0.218 / 0.014, F2 0.010 / 0.169 / 0.060, F3 0.020 / 0.464 /
0.003, F4 0.007 / 0.470 / 0.024.

- **The box improves everywhere, and most where spec 14 lost.** 1024 px
  buttons go from 102 of 240 to **183** (43% to 76%), 1024 px rectangles from
  111 to 118 of 120, 4096 px rectangles from 53 to **60 of 60**. The head box
  beats the chain on F2 (99% against 53%), F3 (98% against 69%) and F4 (100%
  against 68%).
- **The point improves too, on every set**, and now beats the chain on all
  four: 236/240 against 211, 240/240 against 194, 120/120 against 118, 60/60
  against 57. Spec 14's point was 231, 240, 119 and 59 on its own sets.
- **`box`'s default stays the chain**, by the rule written before the run:
  on F1 the head box is 76.2% against the chain's 79.2% — seven boxes short
  on one set of four. `"method": "head"` stays available and is the better
  answer on everything else measured.

**Wall time.** A default point costs 186 ms asked first and 135 ms asked
second at 1024 px, against the chain point's 318 and 269; at 4096 px 6.0 and
2.3 s against 6.2 and 2.5 s, the prefill dominating. Unchanged from spec 14.

## Where the last seven boxes are

The shortfall on F1 is **one bucket**, and it is a resolution limit rather
than a tuning miss. By the target's height in image cells (a cell is 32 px at
1024 px):

| target height | n | head box >= 0.5 | chain box >= 0.5 |
|---|---|---|---|
| under 1.25 cells | 92 | 40 (43.5%) | 70 (76.1%) |
| 1.25 - 1.50 | 34 | 32 (94.1%) | 28 (82.4%) |
| 1.50 - 2.00 | 66 | 65 (98.5%) | 54 (81.8%) |
| 2.00 and over | 48 | 46 (95.8%) | 38 (79.2%) |

**The head box wins every bucket at 1.25 cells and above, and loses only
below it** — where the object is smaller than one image cell and the box's
*size* is not observable at all. The sub-cell peak recovers a head's
*position* inside its cell; it does not tell you how big something is that
every head lands on the same one or two cells of. The same split reproduces
on the development sets (T1 47.4% and T1' 50.0% below 1.25 cells, 94-100%
above), so it is the reading's boundary and not F1's.

Measured on that bucket and rejected, so the next attempt does not pay for
them again:

- **A span-proportional allowance** (`clamp(a * span + b, lo, 0.05 + 4.5/n)`,
  swept and selected on the fit sets alone): buttons do not move at all
  (78.3% fit, 81.2% held, identical to the shipped rule). It only ever moves
  the real screenshots (70.6% to 79.4% at `a = 0.40, b = 0.40`) — worth
  carrying into a future spec, useless for this bucket.
- **A span-capped allowance** (`min(0.05 + 4.5/n, a * span + b)`): the cap
  never binds at the shipped constants.
- **Picking the method by the head's own extent height** — the only size a
  caller has before asking: the best threshold on F1 reaches exactly 190 of
  240, tying the chain and beating neither. The head's extent is biased *up*
  in this bucket (1.45 cells for a 0.95-cell target), so it does not separate
  the objects that need the chain from the ones that do not. An oracle
  picking the better of the two methods per scene would get 230 of 240, so
  the information exists — just not in the extent.

## The pieces below the endpoint

- **The leaf's peaks and neighbours are the attention's.** With the tap armed
  on all nine layers the set arms, over the 1024 px fixture (5 scenes) and the
  4096 px fixture (3), under BF16 and hq-e8-2b: every one of the 96 heads'
  peak scores and all four of its neighbours equal the tap's host-side
  `q . k / 16`, a peak on the image's border reports the missing neighbour
  absent, and spec 14's argmax and the pointing head's row are unchanged
  (0 argmax ties, worst row difference 2.2e-6 relative).
- **The kernel's own op test** scores the four keys around each argmax
  against a double-precision host restatement and pins NaN where the grid has
  no neighbour — the span is 1000 keys over 32 columns, so the short last row
  has nothing below it. 64 of 64 kernel op tests pass.
- **The host rule is the reference's.** `ignis_core::pointing::read_anchored`
  reproduces `ensemble_score.py read_anchored` on seven golden cases
  regenerated from the same real dumps, now carrying the peaks and
  neighbours (35 of their 2688 neighbours are absent, so the border path is
  covered by the golden cases and not only by the table).
- **Cost.** There is no build with the fused launches and not the gather, so
  the test measures both against the pointing head alone: **+0.157 ms** at
  1024 px and **+0.252 ms** at 4096 px on the reading chunk (paired, 25
  interleaved repetitions), against bounds of 0.8 and 1.6 ms. Spec 14
  measured 0.15 and 0.21 ms for the fused launches alone on the same path, so
  the gather itself is about **+0.007 ms** and **+0.042 ms** — nine extra
  launches whose work is nothing.

## Reproduce

```text
python tools/pointing-scenes/scenes.py --varied --seed 20260950 --out .scratch/subcell-264/F1
python tools/pointing-scenes/scenes.py --varied --side 4096 --seed 20260951 --out .scratch/subcell-264/F2
python tools/pointing-scenes/rectangles.py --seed 20260952 --n 120 --out .scratch/subcell-264/F3
python tools/pointing-scenes/rectangles.py --side 4096 --seed 20260953 --n 60 --out .scratch/subcell-264/F4
IGNIS_POINT_SCENES=<absolute path to a set> IGNIS_POINT_OUT=<absolute dir> \
  cargo test -p ignis-server --features cuda --test decide_point_acceptance_gpu \
  -- --ignored --test-threads=1 --nocapture
```

(under the GPU profile's preflight; see `docs/agents/testing.md`. The paths
must be absolute: the test runs with the crate as its working directory.)
The seeds are spent: a new acceptance needs new ones.
