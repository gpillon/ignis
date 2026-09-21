# A declared grid is read to one part in ten, however finely it is declared

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision readout primitives, one-pass pointing, answer mass
- Related: `.scratch/latent-probe/strips.py` (raw material, on disk),
  `docs/specs/decide/11-point-in-one-pass.md` (C1b, C4, E1, E2),
  `2026-09-21-the-point-is-assembled-as-it-is-written.md`,
  `2026-09-21-the-grid-a-bigram-can-name.md`, ADR 0034
- Superseded by: none

## Question

A `point` in **one pass** needs the whole coordinate out of a readout, and a
readout names one vocabulary entry. Spec 11's surviving shape for that is a
**strip**: "which of the N equal vertical strips contains the target", read
at one position, and the same horizontally at a second position in the same
prefill. Single uppercase letters are all admitted by this tokenizer, 26 of
26, so the labels are free (`2026-09-21-the-grid-a-bigram-can-name.md`).

The latent-probe result bounds what can be read out of the **digit** prompt's
residual, and explicitly not this: a residual at a position is about what it
is *about to emit*, so a strip prompt's residual is a different vector. This
had to be measured on its own, and the probe gave it a prior — about ten
buckets of resolution at one position, so 11 and 13 strips sit at it and 26
is a stretch.

A second question rides along. The two reads sit at two positions in one
prefill, so the second is conditioned on whatever letter was placed for the
first. Does a **placeholder** first answer damage it? That is spec 11's E2
for the cross-axis case, which the digit run could not answer because every
`y` read there had seen the true `x`.

## Evidence

The same 240 generated 1024 px scenes, the same vehicle, the same predicate
(inside the button) as the probe run, so every number is directly
comparable. Grid widths 26, 13 and 11; `x` read after a forced `{"x":"`, `y`
read after the drawn letter and `","y":"`. At 26 the `y` read is taken a
second time from a separate prefill with a fixed placeholder letter in
`x`'s place — separate because `Cache.crop` refuses to roll back a GDN
layer ("the current layer does not track past states"), which is spec 11's
branch-tree objection arriving in practice.

**Inside the button**, against the digit chain's 218/240 and the latent
probe's 97/240:

| strips | argmax | centroid |
|---|---|---|
| 26 | 11 / 240 | 3 / 240 |
| 13 | 18 / 240 | 34 / 240 |
| 11 | 19 / 240 | **39 / 240** |

**The readout is well formed.** The letters hold a median **0.981** of the
whole vocabulary's mass at 26 strips, 0.982 at 13, 0.934 at 11; and the
**unrestricted** winner — the model's own free next token, nothing masked —
is a declared letter on **240 of 240** scenes at 26 and 13 strips, 237 of
240 at 11. The model is answering the question asked, in the alphabet asked,
with confidence.

**It is answering it wrongly, and the error scales with the grid:**

| strips | median \|strip error\| x / y | exactly right x / y | as a fraction of the axis |
|---|---|---|---|
| 26 | 4.0 / 3.0 | 13 / 28 of 240 | ~15% |
| 13 | 2.0 / 1.0 | 49 / 65 | ~15% |
| 11 | 1.0 / 1.0 | 86 / 85 | ~9% |

**And the grid is not what is limiting it.** A *perfect* reader of these
grids — one that names the strip actually containing the button's centre —
would land inside on:

| strips | quantization ceiling | measured |
|---|---|---|
| 26 | **223 / 240** | 11 |
| 13 | 156 / 240 | 34 |
| 11 | 126 / 240 | 39 |

**E2, the placeholder.** At 26 strips, `y`'s argmax is the same under the
true `x` letter and under a fixed placeholder on **109 of 240** scenes; when
it differs it moves a mean of 2.58 strips.

## Finding

**The model reads a declared grid to about one part in ten, and declaring a
finer one does not help.** The median error is 9-15% of the axis at every
width tried: 1 strip of 11, 2 of 13, 4 of 26. Dividing the screen more
finely divides the error with it. Against the digit chain's median error of
**0.08%** of the axis, that is two orders of magnitude.

**The failure is the reading, not the resolution.** At 26 strips a perfect
reader would clear acceptance 3 on 223 of 240 scenes — better than the chain
itself — and the model clears it on 11. The grid was never the binding
constraint; spec 11's whole arithmetic about strip widths was pricing the
wrong thing.

**So C1b and C4 are closed, and the sidestep column with them.** C4's 11 and
13 strips were the candidates the probe's prior favoured, and they are the
ones measured here: 39 and 34 of 240. There is no width at which a strip
readout points, because the error is a fraction of the axis rather than a
number of strips.

**Answer mass certifies the shape of an answer, not its correctness — and
this is the case that shows it.** ADR 0034 names a silent collapse of answer
mass as *"the only failure of this endpoint that nothing else would show"*.
Here the mass does not collapse: it sits at 0.98, the unrestricted winner is
a declared label on every scene, and the answer is wrong by a third of the
screen. **Well-formed noise passes the mass check.** Every readout primitive
inherits this, `noul` and `choice` included — what the mass establishes is
that the model understood the alphabet, and nothing more.

**The centroid is not a free refinement and its sign flips.** It helps where
the grid is coarse (19 to 39 at 11 strips, 18 to 34 at 13) and *hurts* where
it is fine (11 to 3 at 26), because a distribution spread over many wrong
strips has a mean in the middle of the screen. Spec 11 rested C1b on the
centroid closing a factor of 1.7; it does not reliably close a factor of 1.

**E2 is not answered, and this run cannot answer it.** A placeholder changes
`y`'s argmax on 131 of 240 scenes and moves it 2.58 strips on average — but
the `y` read is 3 strips from the truth *with* the real letter, so this
measures the instrument's noise as much as the conditioning. E2 needs a read
that works.

## Implications

- **A one-pass `point` has no surviving candidate.** The latent side is
  closed for the digit prompt by the probe result; the vocabulary side is
  closed here for every grid width tried. What is left of spec 11 is the
  **two-pass box** and the **shorter chain**.
- **`point` and `box` should keep their chain**, and the chain's real
  character is now known from the probe run: exact on nine scenes in ten and
  pointing at a different button on the tenth. Effort is better spent on that
  tenth than on removing rounds from the nine.
- **A `scalar`-style readout over a *spatial* legend is a different thing
  from one over a value legend**, and it is worse. Spec 08 measured a value
  grid holding 4/5 at 20 and 32 cells; a spatial grid of 26 is exactly right
  on 13 of 240. The two should not be reasoned about together.
- **Any future readout primitive needs a correctness signal that is not
  answer mass.** None exists today.

## Limits and unknowns

- **Measured on the vehicle's render, not the served one**: an xhigh reasoning-effort instruction at the head of the system block, thinking open (`<think>\n` before the forced `{"x":`) and the instruction as plain text, where `/v1/decide` closes thinking and sends `{"instruction":…}`. The full note is in `2026-09-21-one-attention-head-points.md`, Limits.
- **Nothing was drawn on the image.** This is the model computing a strip
  index from a declared rule, which is the thing it is measured to be bad at
  elsewhere (`number_system`'s own note: asked for a total it had to
  compute, it answered 172 for 230.50). A **set-of-mark** overlay — labels
  rasterized onto the image, spec 11's C2 — is a different mechanism and is
  untouched by this. It is now the only unmeasured candidate in the
  sidestep column.
- One phrasing of the grid instruction, unmeasured as all new prompt text is.
  A better-worded rule might read better; nothing here says how much.
- 1024 px, 240 synthetic scenes from one generator, one instruction, one
  target colour. The fixture's limits carried forward.
- Letters A-Z as strip labels, left-to-right and top-to-bottom. Whether the
  model handles a numeric or a centre-out labelling differently is untested.
- The placeholder branch was run at 26 strips only.
