# 11 - a point in one pass, a box in two

> **STUDY, AND THREE OF ITS EXPERIMENTS HAVE RUN.** It lays out what "one
> forward pass" can and cannot mean on this engine and orders the
> experiments that decide between the candidates. E0, E1 and spec 12's E-P1
> then ran, on 2026-09-21, and between them they **close the one-pass
> point**:
>
> - E0 — the largest grid a compositional label can name is 9 x 9
>   (`2026-09-21-the-grid-a-bigram-can-name.md`);
> - E-P1 — the latent at that position holds about one digit
>   (`2026-09-21-the-point-is-assembled-as-it-is-written.md`);
> - E1 — a declared grid is read to one part in ten at every width
>   (`2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`).
>
> What survives is in *What survives* below: the **two-pass box**, **C2**
> (set-of-mark, the one candidate nothing measured), **C3** with E2 still to
> price it, and a **shorter chain**. The candidates are kept as argued, and
> the branch tree is still kept as killed by argument alone: a study whose
> predictions were checked is worth more than one edited to agree with the
> outcome. The repo's rule is that the ADR is a consequence of the finding
> and not a gate before it (`docs/findings/README.md`).

GitHub: (unfiled)

## The ask

`point` should cost **one forward pass** and `box` **two**. Today they cost
one prefill plus a run of decode rounds: **10 for a point and 25 for a box**
at three digits per axis (spec 06, `crates/server/src/numbers.rs`). Counted
through the loaded tokenizer in `crates/core/tests/grid_label_rectangle.rs`
rather than estimated: a point's schedule is 6 digit steps and 3 forced
literal ones, and a run costs one round more than its schedule because of
the leaf's one-round lag (`constrained.rs`).

## What "one forward pass" has to mean

A decode round *is* a forward pass. So the target is not "fewer rounds", it
is **zero rounds**: one prefill, and the answer read out of it. That is the
same shape `noul`, `choice` and `score` already have, and it is worth more
than the latency it saves:

- a readout holds **no residency and takes no decode lane**, so it composes
  into a fan-out the way `choice` does (spec 04, and spec 08's opening
  argument);
- nothing can be cut short, so `run_cut_short` and `too_many_digits` have no
  analogue;
- and `ignis_decoded_tokens_total` stops moving for a point, which is what
  ADR 0034 says a decision should look like.

A prefill of a scaffold is still **one** prefill even if the scaffold is a
hundred tokens long, so "one pass" admits more than "read the last position
and stop". That latitude is the whole design space below.

## The one constraint everything runs into

A readout reads **one position**, so it names **one vocabulary entry**
(`crates/core/src/decision.rs`). A number is not one symbol in this
tokenizer - `\p{N}` is isolated in the pre-tokenizer, so `10` is two tokens
(spec 08) - which is why `number` is K positions read **in order**, each
conditioned on the ones before it.

Spec 06 states the dependency as a rule: *"the y digit is read after x has
been forced, so the model knows where it put x. Two separate `number`
questions would be two prefills that can contradict each other."* Every
one-pass candidate below takes exactly one of three postures toward that
chain, and this study is mostly about which:

| posture | how | exact? |
|---|---|---|
| **satisfy** it | branch on every prefix inside one pass | would be - but it is dead, below |
| **approximate** it | condition each read on a fixed placeholder | measured, not argued (E2) |
| **sidestep** it | make the answer **one symbol per axis**, so there is no chain | exact by construction; the resolution is the question (E0, E1) |

## The enabling fact, and the enabling non-fact

**Every position's hidden state already exists when the head runs.**
`kernel/src/step.cu` computes the final norm and the output head on
`final_residual + (T - 1) * hidden` - the *last* row of a chunk whose other
`T - 1` rows are sitting right there. Reading k positions instead of one is a
gather of k rows and a GEMM with k columns against the same output head, so
its **weight traffic is the same** (hidden 5120 x vocab 248,320): k readouts
cost about what one costs. Nothing about a multi-position readout is
expensive.

What it is not is *free of work*. Today the head's output is copied whole to
the host (497 KB of BF16 per position) and gathered there, so k positions is
k copies across PCIe; and `PrefillOutcome` is `Copy` (spec 01), which k
readouts breaks. And `final_residual` is **per chunk**: at
`prefill_chunk_tokens` = 1024, a scaffold that straddles a chunk boundary has
its earlier positions already gone by the time the head runs. The ABI that
survives all three is *a list of absolute positions, each chunk gathering the
ones that fall in its own range* - not "the last k".

## Dead on arrival: the branch tree

The tempting exact construction is to put every continuation in the same
prefill behind a tree-shaped attention mask: ten branches give
`P(d2 | d1 = k)` for every k, a hundred give `P(d3 | d1 d2)`, and the greedy
digit string is read off host-side from a single pass.

**It cannot work on this model.** 48 of the 64 layers are GDN
(`ModelConfig::qwen38_27b`: a layer is GQA exactly when `(i + 1) % 4 == 0`),
and GDN is a **recurrent** layer with a per-sequence state - 6144 x 2048 per
layer, plus conv taps. An attention mask shapes the 16 GQA layers and nothing
else: the GDN recurrence consumes the scaffold strictly in order, so branch
k's tokens are in branch k+1's state whatever the mask says. Branching for
real means one sequence per branch - `seq_checkpoint.cu` clones into N slots
- which is a **batched round over N sequences**, not one forward pass, and N
times the GDN state in VRAM.

It is also exponential before it is anything else: exact recovery of six
chained digits needs 10^5 branches. Recorded so that the next reader does not
spend a day rediscovering it.

The *sequential* cousin survives the GDN objection - ten scaffolds one after
another in the same prefill do give `P(d2 | d1 = k)` for every k, because
later scaffolds simply see earlier ones - but it buys a **beam at depth
one**, not a chain, and it buys it into a context that now contains nine
wrong answers. Worth exactly one experiment (E5), and only if the first digit
turns out to be where the error is.

## A different question, asked next door

Every candidate below takes the coordinate out of the **vocabulary**. [Spec
12](12-the-coordinate-in-the-latent.md) asked whether it can be taken out of
the **hidden state** instead, and its first experiment was a go/no-go for
this document too.

**It fired, and it fired negative** (2026-09-21,
`docs/findings/2026-09-21-the-point-is-assembled-as-it-is-written.md`). At
the position a one-pass point would read, the latent holds the digit it is
about to emit and little more: a cross-validated probe lands inside the
button on 97 of 240 scenes, 111 with an RBF kernel, where the chain lands
218.

**That bound covers the digit prompt, and only it.** A fitted probe is
stronger than any fixed readout of *the same residual*, so C3's first read
is bounded. C1b and C4 declare **strips** rather than digits, so at that
position the model is about to emit something else and the residual is a
different vector — which is the finding's own rule about what a residual
holds. So E1 was still open, with a prior rather than a verdict: about ten
buckets of resolution at one position, so C4's 11 and 13 strips sat right at
it and C1b's 26 was a stretch beyond it.

**E1 then ran, and it closes the column** (2026-09-21,
`docs/findings/2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`).
Asked which of N strips holds the target, the model answers with a declared
letter on 240 of 240 scenes and is wrong by 9-15% of the axis at **every**
width: 1 strip of 11, 2 of 13, 4 of 26. Declaring a finer grid divides the
error with it. Inside the button: 39, 34 and 11 of 240 against the chain's
218. And the grid was never the constraint — at 26 strips a *perfect* reader
would clear acceptance 3 on 223 of 240, better than the chain, and the model
clears it on 11.

So the arithmetic below priced the wrong thing. It asked how finely to
quantize an axis; what fails is the reading, at any quantization.

The candidates are kept as written: they were argued before the measurement
and the measurement is what a study is for.

## What survives

**The sidestep column is closed.** Both halves of it were measured and both
failed: the latent at that position holds about one digit (E-P1), and a
declared grid is read to one part in ten however finely it is declared (E1).
C1, C1b and C4 are done.

**And then the literature was read**, which should have happened first
(`docs/findings/2026-09-21-what-the-literature-says-about-pointing.md`). It
renames the target — a one-pass point is *coordinate-free grounding* — and
names a mechanism none of the candidates below used. What is left:

1. **A box in two passes** — untouched, and now the interesting half of this
   document. Its second pass forces the first pass's digits, which is
   precisely the conditioning the probe result says the model needs; and its
   certificate is unaffected by anything measured.
2. **C5, attention as the answer** — new, and it outranks everything else
   here. The published one-pass grounding methods read the **attention from
   the instruction's tokens to the image tokens**, not the residual: TAG
   does it tuning-free, GUI-Actor trains a small head on it and answers in a
   single pass. Free to try in the PyTorch vehicle, where the 240 scenes and
   the harness already are. The engine-side blocker is specific and
   unmeasured by anyone: **16 of 64 layers have an attention matrix**, and a
   fused kernel never materializes it.
   (`docs/findings/2026-09-21-what-the-literature-says-about-pointing.md`.)
3. **C2, set-of-mark** — demoted. It was the last unmeasured candidate, and
   the literature says it is the one that *fails* to transfer: it lifts
   GPT-4V and generally **decreases** open-weight models, apparently because
   reading marks is an OCR task. Still untested here; no longer the obvious
   next thing.
4. **C3, the placeholder scaffold**, and E2 with it. E-P1 says the chain's
   later positions know progressively more, which is C3's own premise read
   from the other side. E2 is still the experiment that prices it, and E1
   could not answer it — a placeholder moved `y` by 2.58 strips, but `y` was
   3 strips wrong to begin with.
5. **A shorter chain.** Not in the original list, and it comes out of the
   measurement: a probe at `y1` is 187/240 after five rounds where the chain
   is 218 after nine. Fewer rounds at a known cost in acceptance is a
   product decision nobody has been offered.

Two things the pair of experiments settled beyond their own question. **The
latent is not a shortcut around the chain.** And **the bar below has to be
scored as a predicate**: the chain's median error is 0.8 units of 999 and
its mean is 21.6, so a mean comparison between it and anything else is
meaningless — which is exactly the mistake the bar's own table invites.

## The bar every candidate is measured against

Acceptance 3 of spec 06: **inside the button on all three scenes.** On the
committed fixture the hardest is `small`, 320 x 90 px on a 4096 px side -
7.8% wide and **2.2% tall**, so half its height is 1.1% of the side.

A readout over a grid of width W answers with a cell centre, which can sit
half a cell from the truth. So an **argmax alone clears the bar only when
W/2 < 1.1%**, i.e. W < 2.2%. That one line disposes of more of this list than
any argument in it, and it is applied to every candidate below rather than to
the ones that happen to fail it.

| candidate | W | W/2 | argmax clears the bar? |
|---|---|---|---|
| C1, contiguous bigram grid | 11.1% | 5.6% | no, by a factor of 5 |
| C1, free bigram grid | 5.6% | 2.8% | no, by a factor of 2.5 |
| C1b, 26 strips per axis | 3.8% | 1.9% | **no**, by a factor of 1.7 |
| C4, 11 x 13 coprime strips | 0.7% | 0.35% | **yes**, with room |
| C3 / today's digit chain | - | - | yes (measured, 2.1% worst case) |

So **C4 leads on arithmetic** and C1b does not. What C1b has instead is
plausibility - one declared strip index is the simplest thing the model could
be asked - so it is the **control** that says whether any strip readout works
at all, and the coarse pass for a two-pass box. If E1 shows the model cannot
read a single strip index, C4 dies with it and nothing in the sidestep column
survives.

The centroid is the one thing that could move C1b across the line, and it is
**unmeasured**. It is named as a bet with a falsifier (E1), not leaned on:
refusing to build C1 on a factor-of-five centroid and then building C1b on a
factor-of-1.7 one would be the same mistake at a discount.

## The candidates

### C1 - the compositional grid readout: one position, one symbol, no chain

**Measured, and it does not reach.** Kept because the measurement is the
reason the rest of the list is ordered the way it is.

The chain-free construction, and the only one exact by construction: make the
answer a single token naming a cell of a 2-D grid. The answer alphabet's
uppercase bigrams are *already compositional* — `AX` reads as row `A`, column
`X` — so the legend is one sentence rather than 576 lines, which is exactly
the difference spec 08's collapse is about (a legend the model has to
**read**: 4/5 at 20 and 32 cells, 0/6 at 100).

What E0 found (`docs/findings/2026-09-21-the-grid-a-bigram-can-name.md`,
`crates/core/tests/grid_label_rectangle.rs`): **114 of the 676 bigrams are
refused, and the holes cluster in the second half of the alphabet in both
directions.** Only four rows are clean across all 26 columns. So

| grid | size | per cell |
|---|---|---|
| largest **contiguous** square | 9 x 9 | **11.11%** of the side |
| largest free (non-contiguous) | 18 x 18 | **5.56%** of the side |

Against the bar above: half a cell is 5.6% (contiguous) or 2.8% (free) against
`small`'s 1.1% half-height, so an argmax cell clears acceptance 3 only by
luck. A centroid would have to close a factor of five, which is not a bet to
build a primitive on.

What survives:

- **as pass 1 of a two-pass box, it is more than enough.** 11% of the side is
  a fine coarse locate, and it costs one readout at one position with no
  kernel change at all.
- **its real lesson is about single letters.** 26 of 26 uppercase singles are
  admitted, with no holes — which is C1b.

### C1b - the strip readout: two positions, two self-contained questions

What E0 promoted — and the **control** the rest of the sidestep column rests
on, not the winner: the bar above puts C4 ahead of it on arithmetic. Instead
of one label naming a cell, **one label per axis naming a strip**: "which of
the 26 equal vertical strips contains the target", and the same horizontally. Single letters have no rectangle
problem, and the legend is a range.

Why this is not C3 in disguise: the two questions are **self-contained**.
"Which vertical strip" does not need the horizontal answer to be well posed,
where `x`'s tens digit is meaningless without its hundreds digit. So this is
still the *sidestep* posture — there is no chain, only two independent reads
that happen to sit at two positions.

- One prefill, zero decode rounds. The x read sits at the prompt's last
  position and costs nothing new - that is the position the chunk's head
  already computes. The y read sits at a second position, so **the design
  needs the multi-position head (E4) unconditionally**. E2 does not decide
  whether E4 is needed, only whether it is worth writing, by saying how far a
  second read conditioned on a placeholder first answer is damaged.
- An argmax strip is 3.8%, half of it 1.9%, against a 1.1% half-height: **it
  does not clear the bar either**, by a factor of 1.7. So C1b's honest claim
  is a *coarse* point - good enough to seed pass 2 of a box, not good enough
  to be the point. A point-in-one-pass at the fixture's accuracy rests
  entirely on the **centroid**: a 26-way distribution over ordered strips is
  not bounded by its cell width, and adjacent-strip mass is genuinely spatial
  information. That is a bet, and E1 is its falsifier. Spec 08 found the mean
  worse than the argmax, but that was an **ordinal legend** over values, not
  a spatial axis; the result does not transfer and has to be taken again.
- It also composes with C4 for free: a second strip count per axis is a third
  and fourth read, not a new mechanism.

### C2 - marks on the image (set-of-mark)

C1's weakness is that the model must compute the grid; drawing the labels
**onto the image** replaces the computation with a reading. The mechanism is
well established elsewhere, and it is still one prefill and one readout.

Two costs, one of them architectural:

- It needs a rasterizer in the media path (`crates/server/src/media.rs`
  decodes and resizes; it does not draw). A bitmap font and a compositing
  pass is a day of work, not a risk.
- **It breaks the two-pass reuse below.** An overlay changes the image bytes,
  so it changes the media-aware prefix key (#193, `MatchKey`), so a marked
  pass and an unmarked pass share no retained prefix at all. Both passes of a
  box would have to use the *same* marked image - which is fine, but it has
  to be chosen deliberately rather than discovered in a profile.

It also changes what the model grounds on: the free probe's box has to be
re-taken against the marked image before any accuracy number from it is
comparable with the finding's.

### C3 - the placeholder scaffold: k positions from one prefill

Prefill the prompt *plus* the answer's shape with fixed placeholder digits -
`{"x":PPP,"y":PPP}` - and read the head at the six positions that precede a
digit. One prefill, zero rounds, and the answer in the shape spec 06 already
returns.

It **approximates** the chain, and the approximation is not uniform:

- position `x1` is read after `{"x":` and is therefore **exactly** today's
  first digit: no approximation at all;
- `x2` is read after `{"x":P`, where P is a digit the model did not choose.
  If the truth is 767 and P is 0, the model is being asked for the tens digit
  of a number it has been told starts with 0. The finding says the digit mass
  is 1.000 at every position after the first, so it will certainly *write a
  digit* - which digit is the whole question, and the expectation is that
  within-axis chaining is fatal;
- `y1` is read after a whole withheld x. That is the conditioning spec 06's
  rule is actually about, and it is a much weaker dependency: "roughly where
  x is" is not "x's units digit".

So the honest shape of C3 is probably **not** six reads but a shortened one,
or a hybrid with C1 in which C1 supplies the coarse position and the scaffold
refines it. **E2 measures exactly this and nothing else**, because it decides
whether a line of C++ is worth writing.

### C4 - two coprime griddings, multiplied

A generalisation of C1 worth naming because it dodges C1's resolution ceiling
without introducing a chain: ask **two self-contained questions** about the
same axis over grids of coprime width - "which of 11 equal vertical strips",
"which of 13" - and multiply the two distributions as densities over x.

Each question is well-posed on its own, so there is no chain to break; the
product of an 11-strip and a 13-strip density resolves to 1/143 of the axis,
0.7% of the side, **better than the 2.1% the digit chain measures today**.
Decoding by the product rather than by CRT is deliberate: CRT on the two
argmaxes has no graceful failure - one strip of error moves the answer across
the screen - while the product degrades smoothly and reports its own
sharpness.

Its cost is two readout positions rather than one, so it needs E2's answer
about cross-question conditioning, and it needs the model to be reliable on a
strip count it cannot see. It ranks below C1 for exactly that reason, but it
is the one candidate that would make a one-pass point **more** accurate than
the chain rather than less.

## A box in two passes

Two is the interesting number, because the second pass can be **exact** in a
way no single pass can:

- **Pass 1** produces a coarse answer by whichever of C1-C4 survives.
- **Pass 2** prefills the same prompt plus a scaffold whose placeholder
  digits are *pass 1's answer*, and reads the same positions. Now every read
  is conditioned on digits that are already approximately right.
- **And the pass carries its own certificate.** If the argmax at each
  scaffold position equals the scaffold token that follows it, then the run
  is, position for position, what the autoregressive chain would have
  produced - **proved, inside the same pass, for free**. If it does not
  match, the mismatch names the first position that is wrong, and the answer
  is either re-scaffolded (a third pass) or returned with the certificate
  withheld. This is Jacobi iteration with a fixpoint test, and the fixpoint
  test is the part worth having: it turns "probably right" into "provably the
  same answer the chain would have given".

Two constraints on that shape, both sharp:

- **Pass 2 must reuse the image KV**, or "two passes" is two 1.85-second
  prefills at the default vision budget. That is the prompt checkpoint and
  the retained prefix (#183 / #191, ADR 0029).
- **Which means pass 1 and pass 2 must share their system block byte for
  byte.** A checkpoint match is all-or-nothing, and volatile text in the
  system block is zero reuse. If pass 1 asks a grid question and pass 2 asks
  a digit question under a different system text, the second pass re-prefills
  the image and the whole design is worthless. The difference between the two
  passes has to live **entirely in the forced tail**. That is a constraint on
  the prompts, and it is discovered here rather than in a profile.

A crop-and-zoom pass 2 - the pointing finding's own follow-up, *"whether a
second constrained pass over a crop refines the last digit"* - is the other
candidate, and it **cannot** reuse: a crop is a different image. It buys real
resolution (the crop's own 0-999 scale covers a fraction of the screen) at
the price of a second full vision encode. Worth measuring against the
scaffold pass, not assumed worse.

## What is not the limit

At the default 32,768-token vision budget a 4096 x 4096 image is not
downscaled: 16,384 merged tokens, a **128 x 128 grid, one token per 32
pixels** - 0.78% of the side. The measured worst error is 2.1% of the side.
So the model's own spatial grid is about three times finer than the reading
the digit chain gets out of it, and the resolution ceiling is in the
**readout code**, not in the vision tower. A 24 x 24 label grid (4.2%) throws
that away; C4's 1/143 (0.7%) roughly reaches it.

At a lower `--vision-max-tokens` - which the finding argues a real pointing
endpoint wants - the grid coarsens proportionally, and at 2048 tokens
(45 x 45) it is 2.2% of the side and becomes the binding constraint. Any
accuracy number taken below has to name the budget it was taken at.

## The experiments, in the order they should be run

Ordered so that each one can kill the work that follows it.

**E0 - which grids are nameable. DONE, 2026-09-21.**
`crates/core/tests/grid_label_rectangle.rs`, no GPU, 0.6 s:
9 x 9 contiguous, 18 x 18 free, 26 of 26 single letters.
Finding: `docs/findings/2026-09-21-the-grid-a-bigram-can-name.md`. It killed
C1 as a point answer and promoted C1b, which is what an experiment ordered
first is for.

**E1 - can the model read a declared grid at all? DONE, 2026-09-21: no.**
9-15% of the axis at every width, 39/240 at best against the chain's 218.
`docs/findings/2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`.
What it was: The existing harness
(`classify_pointing_gpu.rs`: three scenes, `out_logits` already wired), one
readout, no new code. Two questions in one experiment, because they share a
prefill: the 9 x 9 bigram cell (C1 as a coarse pass) and the 26-strip single
letter on **one** axis (C1b's x read, at the last position, where it needs no
multi-position head). Report the argmax, the centroid, the answer mass,
whether each lands inside the button - and, decisively, **whether
`Readout.full_argmax` is a declared strip letter at all**. That is the typed
option finding's own acceptance ("winner in the declared set at 100% of
rows") and it is one field already gathered: if the unrestricted winner is a
digit or a brace, the strip readout is renormalized noise and no centroid
rescues it. If the model cannot map a visual
position onto a declared strip index, C1b and C4 both die here and the study
collapses to C3.

**E2 - conditioning sensitivity.** The decisive one for everything with two
reads, and it needs **no multi-position head**: prefill up to a position,
read the last position, vary only what precedes it. Three probes per scene —
`x2` after the true `x1` against `x2` after a placeholder (within-axis),
`y1` after the true x against `y1` after a withheld x (cross-axis), and
C1b's `y` strip after the true x strip against after a placeholder. Use the
prompt checkpoint so the 16K image prefill happens once per scene rather than
once per probe. The expectation to be falsified: **within-axis conditioning
is fatal, cross-axis is tolerable.** Do not write a line of C++ until this
has an answer.

**E3 - two coprime strip counts.** Only if E1 shows the model reads a strip
index at all. 11 and 13 strips per axis, product density, same three scenes,
against E1's single 26-strip read as the control.

**E4 - the multi-position head.** C++, scoped by whatever E2 admitted: a list
of absolute positions, per-chunk gathering (`final_residual` is per chunk),
and `PrefillOutcome` carrying k readouts instead of one — which breaks its
`Copy`. Worth writing only once E2 says which positions are worth reading.

**E5 - the sequential beam.** Only if the error turns out to be in the
**first** digit, where the finding's own limit points: *"a target at the
exact centre of an axis puts the first digit on a coin flip between 4 and 5,
and the two digits that follow align to whichever won."*

**E6 - the box's second pass.** Scaffold-and-certificate against
crop-and-zoom, with the shared-system-block constraint honoured, measuring
the certificate's **hit rate** as well as the accuracy: a two-pass design
whose fixpoint test fails half the time is a three-pass design.

## What this study does not establish

- Nothing here is measured. Every accuracy figure quoted is from
  `docs/findings/2026-09-19-constrained-digit-readout-points.md` and is about
  **three synthetic scenes with one obvious target**, which is not a real
  screenshot corpus.
- Whether any of C1-C4 is more accurate than the chain is open in both
  directions. The finding is explicit that the chain returns the likeliest
  **digit string** and not the likeliest **number**, and that the units digit
  is p ~ 0.15 noise. A centroid estimator could be *more* accurate than the
  chain while being a different number, so **chain-equivalence is the wrong
  bar** - acceptance 3 (inside the button, on all three scenes) is the bar.
- The two-pass box assumes the retained prefix actually claims across two
  requests of a decision fan-out. #191 is the slice that would say so.
- `point` and `box` as they ship are untouched by any of this until a finding
  says otherwise. Spec 10 already sets the precedent: a new primitive ships
  **beside** the old one, and `number`, `point` and `box` keep their measured
  prompts byte for byte.

## References

- ADR 0034 (the leaf answers without generating) - the readout seam, the
  answer alphabet, and why the host is not in the digit loop.
- Spec 06 (`number`, `point`, `box`) - the chain this study is trying to
  collapse, and the rule about y being read after x.
- Spec 08 (the scalar readout, NOT IMPLEMENTED) - the legend-width collapse,
  and why it does **not** cover a compositional naming.
- Spec 10 (`scalar`) - the precedent for shipping a new primitive beside an
  old one rather than in place of it.
- ADR 0029 / #183 / #191 - the prompt checkpoint and retained prefix the
  two-pass box depends on.
- `docs/findings/2026-09-19-constrained-digit-readout-points.md` - every
  number quoted above, and three of the six experiments are its own
  follow-ups.
- `kernel/src/step.cu` (`ignis_program_prefill`) - where the head runs, and
  the one row it runs on.
