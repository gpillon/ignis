# How the head's map is read: where its error comes from

- Kind: experiment
- Status: draft — pre-registration written before any number below it
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: serving / decision primitives, one-pass pointing, `/v1/decide` `point` (spec 13)
- Related: `docs/specs/decide/13-point-by-attention-head.md` (#260),
  `2026-09-22-the-codec-costs-the-head-its-read.md`,
  `2026-09-21-one-attention-head-points.md`,
  `.scratch/latent-probe/map_reading.py` and `results/engine*/` (raw, on disk)
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
