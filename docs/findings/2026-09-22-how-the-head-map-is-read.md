# How the head's map is read: where its error comes from

- Kind: experiment
- Status: current (pre-registration committed in da587d2, before any number below)
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: serving / decision primitives, one-pass pointing, `/v1/decide` `point` (spec 13)
- Related: `docs/specs/decide/13-point-by-attention-head.md` (#260),
  `2026-09-22-the-codec-costs-the-head-its-read.md`,
  `2026-09-21-one-attention-head-points.md`,
  `.scratch/latent-probe/map_reading.py`, `map_explore.py`, `results/engine*/`,
  `results/map-reading.{log,json}`, `results/map-explore.log` (raw, on disk)
- Superseded by: none

## Question

Spec 13 reads L39.h10's map with TAG's rule (min-max, cells >= 0.5, the
4-connected region with the highest mean, its score-weighted centre) and
does not re-tune it. Two things were never measured:

1. **Where the head's error comes from** — the map (it peaks on the wrong
   thing) or the rule (the map is right and the reading throws it away).
2. **Whether another reading of the same map is better** — inside more
   often, or closer to the target's centre.

No GPU: the engine dumps hold the pre-softmax scores of every head over the
image for every scene.

## Pre-registration (written 2026-09-22, before any candidate was scored)

**Head and scope.** L39.h10 only, as chosen; no new head search over the 384,
no change to `d`, no GPU. The only multi-head candidates use heads already
ranked by the earlier findings, or ranked within L39 on the development sets.

**Maps.** Per scene, `m = exp(s - max(s))` over the image span (proportional
to the softmax restricted to the image), `n = min-max(m)` — the maps every
number so far was scored on. Step 0 must reproduce the recorded head points
and totals (C with window 227, C4096 with window 236, A 235, B 232) before
anything else is trusted.

**Sets.** Development: A (`scenes-served-bf16`, blue target) and B
(`scenes-varied-served-bf16`), 1024 px, 480 scenes. Check, scored once:
C (`engine-window/scenes-c-served-hq`, 1024 px) and C4096
(`engine-window/scenes-c4096-served-hq`, 4096 px) — production's
configuration after #257/#258.

**Candidates**, all on L39.h10 unless stated:

| id | reading |
|---|---|
| T50 | TAG region, threshold 0.5, weighted centre (**baseline**, spec 13) |
| T30 | the same at threshold 0.3 |
| T70 | the same at threshold 0.7 |
| ARG | argmax cell centre |
| SMC | threshold-free softmax centroid: `m`-weighted centre of the whole span |
| BBC | TAG region (0.5), centre of its bounding box instead of weighted |
| X4 | mean of the min-max maps of L39.h10, L43.h9, L35.h6, L35.h16 (the earlier findings' ranking), then T50 — 1024 px only |
| L3 | mean of the min-max maps of L39's top 3 heads by argmax-inside rate on A+B, then T50 — both sizes |

**Metrics.** Inside the target box (the acceptance predicate); median
max-axis distance from the target's centre in 0-999 units, over all scenes
and over hits only; median signed x and y error.

**Decision rule.** Winner = the highest inside count on A+B; a tie goes to
the lower median max-axis distance over all A+B scenes. The winner replaces
T50 in spec 13 **only if** it beats T50 on A+B by that rule **and** is not
below T50's inside count on C nor on C4096 (a 1024-only candidate cannot
win, since spec 13 serves both sizes). Otherwise T50 stays, and the
alternatives are recorded as measured and lost.

**Decomposition (on T50's misses, every set).** From the map alone: is the
argmax cell inside the target; the share of the image softmax mass inside
the target; is the region the rule picked the one containing the target.
A miss whose argmax is inside is a **rule miss** (recoverable by some
reading); one whose argmax is outside is a **map miss**. Map misses are
further split by where the argmax lands: on another element (from the
generator's replayed boxes), adjacent to the target (within one cell), or
background. And: is the target shorter than one cell.

**Confidence (development sets only).** Does the region's weighted spread,
or its mass share, separate hits from misses (AUC) — i.e. can spec 13's
`uncertainty` be read as confidence, or only as size.

## Evidence

**Step 0 holds.** From the raw scores, T50 reproduces the recorded totals on
every set — A 235, B 232, C 227, C4096 236 — and the harness's own point on
239-240 of 240 per set.

### The candidates

Inside the target, of 240; median max-axis distance from the target's
centre (0-999, all scenes); median signed x and y error:

| | A | B | C | C4096 |
|---|---|---|---|---|
| **T50** (baseline) | **235** · 36.1 · x −36 | **232** · 37.6 · x −37 | **227** · 40.5 · x −39 | **236** · 28.8 · x −26 |
| T30 | 237 · 36.3 | 233 · 37.1 | 225 · 40.0 | 234 · 28.2 |
| T70 | 235 · 37.1 | 232 · 38.0 | 222 · 41.0 | 236 · 28.9 |
| ARG | 233 · 37.1 | 228 · 38.0 | 220 · 41.0 | 236 · 29.0 |
| SMC | 36 · 88.2 | 36 · 93.6 | 27 · 89.3 | 44 · 66.8 |
| BBC | 234 · 36.1 | 231 · 37.1 | 227 · 40.0 | 236 · 29.0 |
| X4 | **238** · **26.3** · x −23 | **233** · **24.5** · x −23 | **229** · **27.3** · x −23 | — (not in the dump) |
| L3 | 216 · 42.9 | 215 · 42.9 | 207 · 45.9 | 233 · 33.8 |

The y error is within ±4 for every reading of L39.h10 alone, on every set. L39's
top three heads on A+B are h10, h7 and h11 (461, 363 and 351 of 480 by
argmax).

**The pre-registered decision: T50 stays.** X4 wins the development sets
(471 of 480 against T50's 467) and holds on C (229 against 227), but it is
a 1024 px candidate — the 4096 px dumps hold only L39 — and the rule
written before scoring says a candidate that cannot be checked at both
sizes cannot win. T30, second on A+B (470), falls below T50 on C and C4096
(225, 234).

### Where the misses come from: the map

T50's misses, per set, split by where the map's own argmax lands:

| | misses | argmax inside (rule miss) | on another element | adjacent to the target | background | target shorter than a cell |
|---|---|---|---|---|---|---|
| A | 5 | 0 | 1 | 4 | 0 | 4 |
| B | 8 | 1 | 1 | 4 | 2 | 5 |
| C | 13 | 1 | 5 | 5 | 2 | 7 |
| C4096 | 4 | 0 | 4 | 0 | 0 | 0 |

**28 of the 30 misses are the map's**: its peak is outside the target, and
on those the target holds a median 0-21% of the image softmax mass at
1024 px (38% at 4096 px) against 44-64% on a hit. Only 2 are the rule's, so no reading of this map can
recover more than 2 of 30. At 1024 px the map misses are mostly **small
targets** — 16 of 26 are shorter than one token cell (32 px), and 13 of
26 have the peak on a cell adjacent to the target; at 4096 px, where the generator draws
every target many cells tall, the 4 misses are all **another element**.

### Where the imprecision comes from: the map again

*Exploratory from here to the end of Evidence — not pre-registered.*

- **The region is one cell.** `exp(s − max)` is so peaked that the cells at
  or above half its maximum are a single cell on 132-166 of 240 scenes, and
  at most three on 226-240. TAG's rule is, on this head, an argmax with a
  little smoothing; a threshold, a weighted centre or a bounding box have
  almost nothing to work with — which is why T30, T70, BBC and ARG land
  within a unit of each other in distance and a few scenes in count.
- **The peak sits on the start of the target's label.** Where the argmax is
  inside the target, it lies at **27-32% of the button's width** (medians) (quartiles
  0.22-0.36 across all four sets) and near its vertical middle, and — with
  the label's extent recomputed from the generator — a median **half a cell
  before the label's first letter**, at both sizes. That is the whole of
  the x offset (−26 to −39): the head finds the label's beginning, and the
  button's centre is half a label further right. The y error is nil because
  the label is vertically centred. Label start and "a fixed ~30% of the
  button" coincide on centred labels, so they were told apart
  (`label_start_check.py`): within one size the correlation with the label's
  start is weak (r 0.18-0.30, slopes +0.57 to +1.03), because labels fill a
  similar fraction of every button (start at 28-43%) and the cells quantize
  it; across sizes it is decisive — the peak sits **half a cell** before the
  label's start at both, which is 0.085 of a button at 1024 px and 0.026 at
  4096 px, where the same scene is drawn four times larger and a fixed
  fraction of the button would not have moved.
- **The pre-softmax scores are no better map**: TAG on min-max raw scores
  lands inside on 2-4 of 240 (the raw map is broad and its top half is
  elsewhere).
- **Other heads peak elsewhere**, which is what X4 buys: averaging four
  maps moves the point a third of the way toward the centre (x −23) and
  gains 1-3 scenes, without changing where L39.h10 looks.

### Confidence (development sets, pre-registered)

With a one-cell region its spread is zero on most scenes, and it does not
separate hits from misses (AUC 0.37 and 0.33 on A and B for "smaller spread
means a hit"; C and C4096 agree, 0.44 and 0.40). **The region's share of the
image softmax mass does**: AUC **0.83 and 0.69** on A and B (0.84 and 0.65
on C and C4096).

## Finding

- **The head's misses are the map's, not the rule's**: 28 of 30, with the
  peak on an adjacent cell (small targets, at 1024 px) or on another
  element. The rule can recover at most 2 of 30; TAG's rule stays, and the
  pre-registered comparison says so.
- **The head's imprecision is the map's too, and it is systematic**: L39.h10
  marks the **start of the target's label**, not its centre — 27-32% across
  the button, half a cell before the first letter. No reading of that one
  map can move a point it does not contain.
- **The region's mass share is a confidence, its spread is not.** Spec 13's
  `uncertainty` as "the region's weighted spread" would be zero on most
  answers and uninformative; the share (AUC 0.65-0.84) is the signal to
  expose.
- **Several heads read together are the only lever found**: X4 is better
  than T50 on every 1024 px set, inside and in precision, and is unmeasured
  at 4096 px.

## Implications

- **Spec 13 keeps T50**, and changes what it exposes: `uncertainty` is the
  map's resolution (one token cell per axis in pixels), not a spread, and
  the answer documents that the point sits on the start of the target's
  label; `region.share` is the confidence to report.
- **X4 is the next pre-registered check**, not a change now: it needs the
  4096 px maps of L35, L43 and L39 (one GPU run of the harness with those
  layers armed at 4096 px), then a decision on a fresh set. In the engine it
  is four GEMVs in three layers instead of one — still one pass.
- For a caller who needs the button's centre, the head is the wrong tool on
  its own: the chain (opt-in) or X4.

## Limits and unknowns

- **Synthetic buttons with a centred label.** "The peak is the label's
  start" is measured on one generator whose targets are labelled buttons;
  an icon with no text, a link, a text field — nothing says where the head
  peaks on those.
- **X4 at 4096 px is unmeasured**; the dumps there carry only L39.
- The confidence AUCs are for separating inside from outside on these
  sets; they are not calibrated probabilities.
