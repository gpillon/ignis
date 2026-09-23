# The head box's one bad cell is the image cell itself

- Kind: experiment
- Status: current
- Observed: 2026-09-23
- Last verified: 2026-09-23
- Scope: serving / `/v1/decide` `box` with `"method": "head"` and `point`,
  the anchored head set (ADR 0039); hq-e8-2b and BF16 KV, 1024 px and 4096 px
- Related: `docs/specs/decide/14-point-and-box-from-the-head-set.md` (#263),
  `docs/specs/decide/15-the-head-box-reads-inside-the-cell.md`,
  `2026-09-23-the-head-set-holds-through-decide.md` (the acceptance this explains),
  `2026-09-23-an-anchored-head-set-points-and-boxes.md` (the study),
  `tools/pointing-scenes/ensemble_score.py`,
  `.scratch/head-box-study/` (the sweeps, raw, on disk)
- Superseded by: none

## Question

Spec 14's acceptance is four sets. The head box wins three of them and loses
one: on **E1, 1024 px buttons, it clears IoU >= 0.5 on 102 of 240 (43%)
against the chain's 190 (79%)**, and by the rule written before the run that
one cell keeps `box`'s default on the chain. Every other cell of the table
says the head box is the better answer — 4096 px buttons 99% against 54%,
1024 px rectangles 93% against 71%, 4096 px rectangles 88% against 62%.

Why that one cell, is it a measurement artefact, and can it be lifted?

## Result

It is not an artefact. It has a mechanical cause, and the cause is fixable.

### The cause: the box is quantised to the image cell, and a button is 1.4 cells tall

At 1024 px an image cell is 32 px. The E1 buttons are **5.4 cells wide and
1.44 cells tall** (median), and **199 of 240 are under two cells tall**. The
shipped reading places each head at its cell's **centre** and grows the
quantile span by **half a cell** on each side, so every box edge lands on a
cell centre plus or minus half a cell — that is, on cell geometry. The
smallest box the rule can express in y is one whole cell, and the one it
usually produces is two: measured on T1, the extent is **1.22x the target in
x and 1.78x in y**.

At that size, cell geometry alone caps the metric. On the same 240 targets:

| a box that can only sit on cell geometry | IoU >= 0.5 |
|---|---|
| the true box, edges snapped to the nearest cell boundary | 198 / 240 (82.5%) |
| every cell the true box touches | 126 / 240 (52.5%) |

So the best a cell-quantised box could do here is 82.5%, and the natural
"cover the cells it touches" reading is 52.5% — below the chain before any
attention is read. E2 (4096 px) does not have the problem because the same
button is four times the cells; E3 and E4 do not because a rectangle is 6-10
cells tall. **E1 is the only cell of the table where the target is about the
size of the quantum.**

### The fix: read inside the cell

The scores the leaf already computes are smooth around the peak, so the
peak's position inside its cell can be recovered from the peak and its two
in-line neighbours per axis — parabolic interpolation, the standard
sub-sample peak estimator, no new parameter. That is **four more scores per
head** at the seam (the peak's own score is already the high half of the
packed argmax), not a new row.

Reading inside the cell alone takes T1 buttons from 40% to 85% with no
growth at all, and costs the wider regimes nothing.

### What the growth was really for

Removing the half-cell growth outright would have cost the owner's
screenshots, and measuring why gave the second half of the rule. The head
span is a **lower bound** on the object: it reaches only the parts the heads
actually hit. How much it under-covers is read straight off how many
**distinct cells** the kept heads occupy:

| set | distinct cells (median) | target / span, x | y |
|---|---|---|---|
| T1, T1' buttons 1024 | 17-18 | 0.96 | 1.14-1.18 |
| shapes, round2, T2, T2' rectangles 1024 | 33-36 | 0.92-0.97 | 1.01-1.05 |
| T1c, T2c 4096 | 46-56 | 0.99-1.01 | 1.01-1.26 |
| the owner's screenshots (doom, s1-s7) | 6-13 | 1.30-2.48 | 1.17-1.93 |

On a synthetic scene the 96 heads tile the object and the span is the
object. On a photograph they pile onto one distinctive part — a monster's
face, a gun's barrel — and the span is half of it. The shipped half-cell
growth was, by accident, paying for that on screenshots while overpaying by
a whole cell on buttons.

So the allowance is **not a constant**: it is what one part leaves
unobserved, and it shrinks as more distinct parts are seen —
`(0.05 + 4.5 / n_distinct)` cells per side.

### Measured

Sub-cell reading, keep radius 3, quantile 0.15, that allowance. Chosen only
on T1, shapes, round2 and doom; **T1', T2, T2', T1c, T2c and s1-s7 were not
used to choose anything**.

| set | n | box IoU >= 0.5: shipped -> new | point inside: shipped -> new |
|---|---|---|---|
| T1 buttons 1024 | 240 | 40.0% -> **78.3%** | 98.8% -> 99.6% |
| T1' buttons 1024 * | 240 | 37.1% -> **81.2%** | 95.4% -> 99.2% |
| shapes rect 1024 | 49 | 63.3% -> 81.6% | 98.0% -> 98.0% |
| round2 rect 1024 | 64 | 87.5% -> 93.8% | 98.4% -> 100.0% |
| T2 rect 1024 * | 60 | 91.7% -> 100.0% | 100.0% -> 100.0% |
| T2' rect 1024 * | 60 | 90.0% -> 96.7% | 96.7% -> 100.0% |
| T1c buttons 4096 * | 10 | 90.0% -> 90.0% | 90.0% -> 90.0% |
| T2c rect 4096 * | 10 | 100.0% -> 100.0% | 100.0% -> 100.0% |
| doom screens | 16 | 56.2% -> 75.0% | 100.0% -> 100.0% |
| s1-s7 screens * | 34 | 64.7% -> 70.6% | 97.1% -> 97.1% |
| **all chosen on** | 369 | 52.0% -> **81.3%** | |
| **all held out** * | 414 | 57.7% -> **86.0%** | |

`*` never used to choose anything. **T1' at 81.2% is above the chain's 79.2%
on E1**, which is the cell spec 14's rule turned on. The point improves too,
everywhere it moves.

Small-n movement worth naming rather than hiding: s6 (4 questions) falls
from 4 boxes to 2, and s1 loses one point of 7. Both are inside the noise of
a four-question set; the screenshots are a report, not a floor.

## What was tried and did not work

Kept here so the next recalibration does not pay for them again.

- **Inverting the parts.** Each head has a calibrated part position, so
  `x_h = x0 + u_h * w` looks like a two-unknown least squares per scene. It
  is not: the per-head part positions correlate +0.98 between two button
  sets but only +0.21..+0.68 across regimes, and inverting an unstable
  design matrix amplifies the error — rectangles fall to 24%.
- **Edge heads.** Solving each edge from the heads calibrated onto that edge
  (a two-point solve) is worse than the blind quantile wherever it fires;
  the settings that scored well were the ones where it never fired and the
  quantile fallback did the work.
- **A union of the heads' footprints** above a threshold: over-covers
  everywhere (1.5-2.7x the target on buttons).
- **A marginal profile** of the summed head maps, thresholded per axis: 2 of
  240 on buttons.
- **A per-head attention spread** as a size feature: correlates +0.78 with
  the target's height on buttons and **-0.34** on rectangles. Not a feature.
- **A 3x3 neighbourhood** instead of the cross: no better than the cross
  (82-89% against 85-89% on buttons), so the seam carries four neighbours,
  not eight.

## Reproduce

The sweeps read the vision study's dumps (every GQA layer armed) and re-read
them on the host, so none of this needs the GPU:

```text
.scratch/head-box-study/study.py          the dumps, the shipped rule, the caches
.scratch/head-box-study/cross.py          parabolic sub-cell from the cross vs the 3x3
.scratch/head-box-study/final.py          the search, selecting on the fit sets only
.scratch/head-box-study/report_numbers.py every number above
```

They are raw material in the clone that ran them; the rule they chose is
`ignis_core::pointing::read_anchored` and spec 15.
