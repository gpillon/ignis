# The point is assembled as it is written, not read out of the latent

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision readout primitives, one-pass pointing, `/v1/decide` `point`
- Related: `.scratch/latent-probe/` (raw material, on disk),
  `docs/specs/decide/12-the-coordinate-in-the-latent.md`,
  `docs/specs/decide/11-point-in-one-pass.md`, ADR 0034,
  `2026-09-19-constrained-digit-readout-points.md`
- Superseded by: none

## Question

`point` costs one prefill plus **10 decode rounds**, and `box` 25 (counted
through the loaded tokenizer, not estimated). The owner's proposal: the model
answers coordinates on a 0-999 scale of its own — a free probe with no scale
declared anywhere returns them — so take the coordinate out of the **hidden
state** at the first read position and skip the spelling entirely.

That is one pass and zero decode rounds, which is what makes a decision
compose into a fan-out (spec 04, ADR 0034). The question is whether the
coordinate is *there*.

Two readings of the measured per-digit trace, which falls from p ~ 0.99 on
the first digit to p ~ 0.15 on the last:

- **(a)** the latent holds the number and the chain merely serializes it;
- **(b)** the latent holds the digit the position is about to emit, and the
  rest is computed at its own position after the forced token is seen.

Reading (b) predicts a specific number. If the latent holds only the leading
digit, the best any probe can do is the centre of that digit's bucket and the
residual is the rest — **25 units on a 0-999 scale, 2.5% of the side**.

## Evidence

### The vehicle, and that it sits where the engine sits (E-P0)

`Qwen/Qwen3.8-27B` at revision `1d4bf0f2...`, **the artifact's own pin**,
quantized on load by bitsandbytes NF4 with `visual` and `lm_head` left alone
(the n-gram study's vehicle; the NVFP4 route is closed in transformers). The
same `point_system(3)` text byte for byte, the same forced `{"x":` prefix
appended to the rendered prompt.

At 4096 px this does not run: transformers' SDPA path asks for a **24.5 GiB
attention matrix** on a 32 GiB card at 16,538 tokens. Everything below is at
**1024 px** (1,178 prompt tokens), a regime the engine's own width walk
measures `point` inside the button in
(`2026-09-21-vision-tower-cost-at-width.md` §5).

On the three committed fixture scenes, against what the engine drew at 4096:

| scene | vehicle (0-999) | engine, direct | inside the button |
|---|---|---|---|
| large | 769, 815 | 767, 835 | yes |
| medium | 313, 237 | 311, 242 | yes |
| small | 892, 156 | 892, 165 | yes |

x agrees within 2 units on all three at a quarter of the resolution, and the
per-digit signature is the finding's: first digit p 0.958-0.984 with digit
mass **below 1** (0.714-0.835 — the position where a quote was still
plausible), mass **1.000 at every position after**, last digit falling to
0.431-0.603. Same shape, different vehicle, different image size.

### The probe (E-P1, E-P2)

240 generated 1024 px scenes: blue "Save" button at a random position, size
swept over the fixture's own 7.8%-26.9% range, 2-4 distractors, chrome
jittered. Ground truth by construction. The constrained chain run on each,
keeping the residual stream at **layers 19, 32, 47 and 64** at all six read
positions. Ridge in the dual, 5-fold cross-validated, out-of-fold
predictions scored.

The chain itself on these scenes: **218/240 inside the button (90.8%)**,
mean error 2.75% of the side.

**Controls.** A permuted-label fit lands at 20.5% of the side at every
position and layer, which is the predict-the-mean baseline (20.42%) — so
nothing leaks. Features are centred; a raw cosine in a residual stream is
~0.995 whatever is asked of it.

**At `x0`, the only position a one-pass point has** (after the forced
`{"x":`, before any digit is drawn):

| probe | mae x | mae y | % of side | inside |
|---|---|---|---|---|
| linear, L47 | 17.5 | 34.1 | 2.58% | **97 / 240** |
| linear, L64 (the final residual) | 20.8 | 109.8 | 6.54% | 45 / 240 |
| linear, L19+32+47+64 concatenated | 19.0 | 64.9 | 4.20% | 65 / 240 |
| RBF kernel, L32 | 34.8 | 25.9 | 3.04% | **111 / 240** |
| RBF kernel, L47 | 17.0 | 36.3 | 2.67% | 100 / 240 |
| *the chain* | 22.1 | 34.2 | 2.75% | **218 / 240** |

**Against the chain's own output** rather than the truth, the same position
recovers `x` to 20.9 units at L64 — against the leading-digit baseline of
25.0 — and `y` to 148 units against a chance of 204.

**The learning curve is flat.** Probe at `x0`/L47, against the truth:

| n | % of side | inside |
|---|---|---|
| 60 | 3.77% | 33% |
| 120 | 2.80% | 43% |
| 180 | 2.77% | 42% |
| 240 | 2.58% | 40% |

**Across the six read positions** (best layer each, against the truth):

| read | inside | |
|---|---|---|
| `x0` | 97 | before any digit |
| `x1` | 84 | |
| `x2` | 68 | |
| `y0` | 143 | x fully written |
| `y1` | **187** | |
| `y2` | 168 | |
| the chain | **218** | after all nine steps |

## Finding

**Reading (b), measured.** At the position where the model is about to write
x's first digit, the latent holds x to about one digit: 20.9 units against a
leading-digit baseline of 25.0. The number is not sitting there waiting to be
serialized.

Read carefully, because two different questions are being asked of the same
position and they have different answers. Predicting **what the chain will
say** for y from that position is near hopeless (148 units against a chance
of 204) — the chain's y has not been decided yet, and its eventual spelling
is partly its own noise. Predicting **where the button is** on y is not
hopeless at all (34.1), it is merely coarse. The first is what the
discriminator asks; the second is what a shipped primitive would want.

**The point is assembled as it is written.** The probe's accuracy climbs
monotonically with the chain's own progress — 97, then 143 once x is
written, then 187 — and never reaches the chain's 218. Each forced digit is
not a readout of something already decided; it is a step that decides it.

**Per axis, each one is most available at its own read position**, which is
the same statement from the other side: `x` is recovered to 17.5 at `x0` and
`y` to 24.0 at `y0`, against 34.1 for `y` at `x0` where the model has not
been asked about it yet. On the *means* each of those beats the chain's own
21.6 and 33.4 — but that comparison is the trap the distributions above
close, and on medians the chain is seventeen times sharper. The honest
reading is that an axis is **coarsely** present at its own position and the
pair is not present at one position at all.

**A fitted probe upper-bounds every readout of *this* residual.** A readout
is a *fixed* linear map into the vocabulary; a probe is the best *estimable*
linear map into the answer itself, fitted directly against it. The probe is
strictly the stronger instrument, and it reaches 97/240 where the chain
reaches 218. So no one-position reading of the `x0` residual **under the
digit prompt** yields a point.

**That bound does not cross prompts, and the finding's own next paragraph is
why.** If the residual at a position is about what it is *about to emit*,
then under the digit prompt it is about a hundreds digit — and the
measurement agrees, resolving about ten buckets (20.9 units against the
25.0 baseline). Under a *strip* prompt the same position is about to emit a
strip, and its residual is a different vector. Spec 11's C1b and C4 declare
strips, not digits, so they are **not** bounded by this row and have to be
measured on their own.

What this does give them is a **prior**, and a specific one: ten buckets of
resolution at one position is what was measured here, so C4's 11 and 13
strips sit right at it and C1b's 26 is a stretch beyond it.

**The mean absolute error is the wrong statistic for *both* instruments,
and by three orders of magnitude.** The best probe's 2.58% of the side
*beats* the chain's 2.75%, and lands inside the button less than half as
often. The distributions say why — measured, not inferred, over the same
out-of-fold predictions:

| | median | mean | p90 | max |
|---|---|---|---|---|
| chain, x | **0.8** | 21.6 | 1.8 | 868.4 |
| chain, y | **0.6** | 33.4 | 1.6 | 881.5 |
| probe `x0`/L47, x | 13.9 | 17.5 | 38.7 | 120.5 |
| probe `x0`/L47, y | 27.7 | 34.1 | 70.0 | 173.8 |

(0-999 units; the median button is 174 wide and 47 tall, so half-height is
23.4.)

**The chain is exact or catastrophic, and nothing in between.** Its median
error is **0.8 units of 999** and its p90 is 1.8 — on nine scenes in ten it
is on the button's centre to a tenth of a percent of the side. The mean of
21.6 is entirely the other tenth, where it points at a *different button*
(max 868). **The probe is neither**: it never reaches the chain's precision
and never has the chain's failure, with a median of 13.9 and a maximum of
120.

So the equal means compare a bimodal distribution against a smooth one and
say nothing. Acceptance 3 is a predicate and has to be scored as one — and
on the medians the chain is not marginally better than the probe, it is
seventeen times better.

**Neither more data nor a non-linear probe rescues it.** The learning curve
is flat from n=120 (2.80% to 2.58%, and *fewer* inside), so the ceiling is
the latent and not the sample. An RBF kernel buys 97 to 111 of 240, and
concatenating four layers makes it *worse* (65) — more capacity against the
same absent signal.

**The site is not the final residual, and that matters for what a fix would
cost.** For predicting the truth from `x0`, L47 is 2.58% against L64's
6.54% — two and a half times better at a layer that has no seam. The one
place a hidden state is available today (`final_residual`, where
`out_logits` is copied out) is the **worst** of the four sampled.

**The residual is about what to emit next, not a summary of what was
emitted.** At the `y1` position x has been fully written and is in the
context, yet the probe recovers it only to 38 units — worse than at `x0`,
where the model had not yet committed to it. What a position holds is its
own next token, not the run so far.

## Implications

- **The one-pass `point` via a latent probe is dead on this model**, and so
  is the one-pass point via any single-position readout. Spec 12's go/no-go
  fired negative, and it takes spec 11's C1, C1b and C4 with it — the
  arithmetic there about strip widths was about quantizing something that is
  not at that position.
- **What survives is a shorter chain, not no chain.** A probe at `y1` is
  187/240 after five rounds where the chain is 218 after nine. If a caller
  will trade 14% of the acceptance rate for 44% of the rounds, that is a
  real product; nobody has been asked.
- **A box in two passes is untouched** by this and is now the interesting
  half of spec 11: its second pass forces the first pass's digits, which is
  exactly the conditioning this finding shows the model needs.
- **If a probe is ever revisited, it belongs mid-stack**, which means a tap
  and not a reuse of `final_residual` — a much larger kernel change than the
  10 KB hidden-state crossing spec 12 costed. That crossing is no longer
  worth its ADR 0034 amendment.

## Limits and unknowns

- **1024 px, not 4096.** The digit traces of the original pointing finding
  are at 16,384 vision columns; these are at 1,024. E-P0 shows the chain
  behaves the same way at this size on the three fixture scenes, but a
  latent's contents at a quarter of the resolution are not proven to be the
  latent's contents at full resolution.
- **240 synthetic scenes from one generator**, flat-colour chrome, one
  obvious blue target, one instruction phrasing. The fixture's own limits,
  carried forward.
- **Whether the deepest tap is pre- or post-final-norm was not checked.**
  transformers 5.17 captures layer outputs through a hook, so index 64 is
  the last decoder layer's output and probably `h` rather than `n`; an
  RMSNorm is a per-sample rescale and not a linear map, so the two are not
  interchangeable for a probe. Nothing here turns on it — index 64 is the
  *worst* of the four sites either way — but a follow-up that leans on the
  final residual should establish which it has.
- **Four layers sampled of 64.** L47 is the best of {19, 32, 47, 64}; the
  true optimum could be elsewhere and a finer sweep was not run. It would
  not change the conclusion at `x0` unless some unsampled layer beats the
  chain outright, which nothing in the shape of the curve suggests.
- **Ridge and RBF, at n=240.** "The best estimable linear map" is not "the
  best linear map". A probe trained on thousands of scenes might do better;
  the flat learning curve is evidence against that, not proof.
- The probe was fitted and validated on the same generator. Generalization
  to real screenshots is unmeasured — as it is for every pointing number in
  this repo.
- **Nothing here is about the engine's own numbers.** The chain measured
  here is the PyTorch vehicle's, not `/v1/decide`'s. E-P0 ties the two
  together on three scenes and no more.
