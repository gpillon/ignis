# One attention head points, in the prefill, and guards the chain

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision primitives, one-pass pointing, `/v1/decide` `point` and `box`
- Related: `.scratch/latent-probe/c5*.py` (raw material, on disk),
  `docs/specs/decide/11-point-in-one-pass.md` (C5),
  `2026-09-21-qwen-vl-grounding-primary-sources.md` (the mechanism and its precedents),
  `2026-09-21-the-point-is-assembled-as-it-is-written.md`,
  `2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`,
  `2026-09-19-constrained-digit-readout-points.md`
- Superseded by: none

## Question

A one-pass `point` has been looked for three ways and not found: the residual
at the read position holds about one digit, a declared grid is read to one
part in ten, and a compositional label can name only a 9 x 9 grid. The
published one-pass grounding methods read something else — the model's
**attention from a text position to the image tokens**
(`2026-09-21-qwen-vl-grounding-primary-sources.md` has the precedents and
what each actually reads). On this model only 16 of the 64 layers have an
attention matrix; the other 48 are GDN. Does a quarter of the layers carry a
point, and does it survive a target that is not the only blue thing on the
screen?

## Evidence

**Method.** The PyTorch vehicle of the probe run (the artifact's own pinned
revision, NF4, 1024 px), the chain's own prompt with the forced `{"x":`. The
global `sdpa` attention entry is wrapped: for the text attention modules
only, and only for a few query rows, it computes `softmax(q k^T * scale)`
from the post-RoPE, post-norm q and k the module already produced, keeps the
image-token columns, and hands the call to the original SDPA. One prefill per
scene, nothing decoded. Three query rows: **`last`** — the position after
`{"x":`, the one a one-pass point would read — and two in the instruction.
Maps become points by argmax cell or by region centre (normalize, keep >= 0.5
of the peak, the 4-connected region with the highest mean, its weighted
centre). Head selection is cross-validated: heads ranked by their accuracy on
training folds, scored on the held-out one.

**Two scene sets, 240 each.** Set A is the probe run's: the target is always
the one blue button. Set B varies the target's colour, and on 124 of the 240
names it by its **label** ("click the Reset button") with distractors free to
share its colour — so a head that merely finds blue cannot score.

### Set A

Query `last`, inside the button, against the chain's 218/240:

| estimator | inside |
|---|---|
| all 384 heads averaged, no labels | 48 |
| best layer, all its heads (L35) | 158 |
| top-10 heads by attention mass, no labels | 89-97 |
| **CV-selected top-1 head, region centre** | **236** |
| CV-selected top-5 / top-10 / top-40 | 236 / 236 / 212 |

**Every fold chose the same head: L39.h10.** Per layer, all heads averaged:
nothing at L3-L27 (0-15), the signal at **L31-L55** (79, 158, 153, 129, 119,
104, 63), nothing at L59 and L63 (10, 3). The label-free selections fail, and
removing attention sinks by the spread of each head's argmax across scenes
does not rescue them (90-97).

**Orientation checked**, since a wrong token layout would transpose every
map: row-major 231/240, transposed 8.

### Set B — the head chosen on A, applied unchanged

| | named by colour | named by label | total |
|---|---|---|---|
| the chain | 101 / 116 | 112 / 124 | 213 |
| **L39.h10, chosen on A, region centre** | **113 / 116** | **120 / 124** | **233** |
| top-1 head chosen within B (CV) | 110 / 116 | 118 / 124 | 228 |

The head chosen on the blue set is **as good as one chosen on B itself**, and
holds on the label-named targets — reading the text on a button, among
distractors of the same colour. Its output gate at `last` has a median of
0.395 against 0.171 over all heads: the model uses this head more than most.

At the **instruction** positions the transfer collapses — 136 and 143 of 240
for the head A chose there — and the per-kind results swing between
transfer and within-B selection. The position that carries the point is
**inside the answer's scaffold**, after `{"x":`, not in the prompt.

### Precision, and the box

CV top-5 at `last`, 0-999 units, set A: argmax median x 30.2, y 8.8; region
centre **x 21.5, y 5.9**. One merged cell is 31 units at 1024 px, and the
buttons are 4-6 times wider than tall, so an argmax anywhere along the width
is inside and its x error is about half the width; the region centre pulls it
back. The chain's median on the same scenes is **0.8**.

A box read off the thresholded region has a median IoU of **0.28-0.38**, and
clears IoU 0.5 on at most 52 of 240. The attention lights part of the target,
not its extent.

### Against the chain, scene by scene

| | both right | only C5 | only chain | neither |
|---|---|---|---|---|
| set A | 214 | **22** | 4 | **0** |
| set B | 207 | 26 | 6 | 1 |

On A the head is right on **all 22** scenes the chain gets wrong; the 4 it
misses are the smallest buttons (median 9.9% x 2.4% of the side). The chain's
failures are not near misses: on B all 27 are more than 2% of the side off
on an axis, pointing at a different element.

**Combined** — keep the chain's point unless it is more than `d` from the
C5 region centre, else return the C5 centre:

| d (0-999) | set A | set B (d fixed on A) | chain kept on | median error of the hits |
|---|---|---|---|---|
| 30 | 238 | 235 | 65 / 54 | 37 / 37 (C5's precision) |
| **60** | **239** | **239** | 208 / 190 | **0.8-1.1 x, 0.6-0.7 y** |
| 100-150 | 239 | 238 | 217 / 214 | 0.8-1.0, 0.6 |

A plateau from 60 up, and the value chosen on A holds on B: **239/240 on
both**, at the chain's precision.

A tie-break on confidence instead of distance does worse (233 on A): ranking
two signals of different scales against each other is arbitrary. But one of
the signals is worth having on its own — **the chain's restricted probability
of x's and y's first digit** is a median 0.981 on its successes and 0.843 on
its failures, and the failures sit at ranks 0.00-0.05. C5's peak separates
far less (0.186 against 0.159).

## Finding

**A single attention head, in a quarter of the layers that have attention,
points.** Read at the position after `{"x":` in the same prefill the chain
already runs, L39.h10 lands inside the button on 236 of 240 scenes where the
target is blue and 233 of 240 where it is named by a colour or by its label —
**more often than the ten-round chain** (218 and 213), in one pass, with
nothing decoded. It is coarser: a median of about 20 units of 999 against
the chain's 0.8. So the owner's one-pass point exists, and it is not the
answer's precision that makes it; it is the answer's reliability.

**It is a grounding head, not a colour detector.** Chosen on a set where the
target was always the blue one, it transfers unchanged to targets named by
their label among same-coloured distractors (120/124), and matches a head
chosen on that set itself. That was the confound, and it is closed.

**The two instruments fail on different scenes, and that is the result worth
building on.** The chain is precise and has a catastrophic tail: when it is
wrong it is on a different element, not near the right one. The head is
coarse and has no such tail on these scenes; it fails on the smallest
targets. Together, with one distance test, they reach **239 of 240 on both
sets at the chain's precision** — and the threshold was chosen on one set and
held on the other.

**This reconciles with the literature rather than contradicting it.** The
only native attention readout measured on a Qwen (Trifuse, Qwen2.5-VL-3B)
beats coordinates on ScreenSpot and halves them on ScreenSpot-Pro — it is weak
on small targets (`2026-09-21-qwen-vl-grounding-primary-sources.md`). So is
this head: its own misses are the smallest buttons. What it repairs is a
different failure, **the wrong-element tail**, which the chain has and the
attention does not. Two phenomena, and they should be named apart: C5
inherits the small-target limit and removes the wrong-element one.

**Where the query sits decides whether it transfers.** At the instruction's
tokens the head choice does not survive a change of instruction; at the
position inside the answer's scaffold it does. The same holds in the
literature's best query placement.

**Label-free selection is not enough here; one calibrated head is.** The
average over all heads, the top heads by attention mass, and the same with
sinks removed all stay under 100 of 240. A head chosen once, with labels,
on synthetic scenes, holds out of sample. That makes the head a **calibrated
constant of the artifact**, in the same position as the answer alphabet:
computed against a specific load, refused against another.

**The chain already carries its own warning.** The restricted probability of
its first digits flags its wrong-element failures, ranks 0.00-0.05 on set A,
and `/v1/decide` returns that trace today. Note which signal this is: not
the **answer mass**, which certifies only that the model answered in the
alphabet (`2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`), but
the **restricted probability** of one digit — the same instrument that
flagged a contested alignment in #254, flagging a different failure.

## Implications

- **A one-pass `point` is available**, as a coarse primitive that lands
  inside more often than today's `point` and costs one prefill. What it does
  not deliver is today's precision.
- **Today's `point` can be made more reliable at no extra round**: the head's
  map comes out of the prefill the chain starts from, and the comparison is
  arithmetic on two points. 218 to 239 of 240, 213 to 239, precision kept.
- **Engine cost is one GEMV per decision**, not a new attention path: the
  query row of one head (256 values) at one position, captured during the
  prefill, against the image keys already in the paged cache — `q . K_img^T`
  over 1,024 image tokens at 1024 px, 16,384 at 4096. Normalizing per map
  makes the full-row softmax denominator irrelevant, so the logits over the
  image keys are enough. Under hq-e8-2b the keys need the attention kernel's
  own dequantization; under BF16 it is trivial. Two numbers cross the seam.
- **The head is keyed to the artifact.** A test that fails when the artifact
  changes and the head is not recalibrated, as the alphabet is recomputed
  from the loaded tokenizer.
- **The box does not come out of the map.** For `box`, the head is the guard
  — *where* — and the numbers still come from the chain.
- **An answer-centred zoom must be seeded by the head, not the chain**: a crop
  around the chain's answer contains the target on only 3 of 22 of its
  failures at half the side (`2026-09-21-qwen-vl-grounding-primary-sources.md`).
  And spec 11's two-pass certificate proves only that a run matches the chain
  — the model reproduces its own errors, so it cannot catch this tail; the
  head can.

## Limits and unknowns

- **1024 px, synthetic scenes, 480 of them from one generator family.** Flat
  chrome, one instruction shape ("click the ... button"), one target per
  screen. The head was chosen and tested on synthetic data; a real screenshot
  corpus is where it has to hold next, and where the small-target limit will
  matter most.
- **The PyTorch vehicle, not the engine.** NF4 weights in transformers, not
  ignis's NVFP4 kernels; BF16 attention, not hq-e8-2b keys. Whether the same
  head points under the engine's quantized KV is the first thing to measure
  there.
- **One head was chosen by 240 labelled scenes.** "Calibrated once" means
  someone has to do it, per artifact, with labelled data.
- **16 of 64 layers have attention.** L39.h10 is one head among 384; nothing
  here says the other 48 layers would not have carried a better signal if
  they had attention to read.
- **Precision is bounded by the vision grid**: one merged cell is 31 units at
  1024 px. A larger image has finer cells and more image keys; neither was
  measured.
- The combination's `d` is in 0-999 units at 1024 px; whether it transfers
  across image sizes is untested.
