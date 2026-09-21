# 12 - the coordinate in the latent

> **STUDY, NOT A SPEC**, and an **addition to [spec 11](11-point-in-one-pass.md)**
> rather than a replacement for it. Spec 11 asks how to get a point out of the
> **vocabulary** in one pass and answers it with arithmetic that stands on its
> own. This asks a different question — whether the coordinate can be taken out
> of the **hidden state** instead, before the vocabulary projection throws most
> of it away — and it is the more ambitious of the two. Nothing here is
> measured. The go/no-go is one experiment and it is named below.

GitHub: (unfiled)

## The ask

The pointing finding establishes that this model answers coordinates on a
**0-999 normalized scale natively**: a free probe with no scale declared
anywhere returned `[634, 789, 901, 864]` on a 4096-pixel image and landed on
the right button once rescaled
(`docs/findings/2026-09-19-constrained-digit-readout-points.md`).

So the model *has* the coordinate before it writes a token. The owner's
question: **take it from there** — read the latent directly rather than
spelling it through ten digit tokens and nine forced literals.

## What "read the latent" is, exactly

It is worth being precise, because the phrase promises more than the
architecture can give.

`ignis_program_prefill` computes, for the chunk's last position:

```
h  = final_residual[T-1]        (5120 BF16)
n  = rmsnorm(h, final_norm)
logits = output_head . n        (248,320)
```

A readout gathers named entries of `logits`. **A probe is a different linear
map applied to the same `n`** — `probe . n`, a few numbers instead of
248,320. There is nothing "earlier" or "more direct" available at that
position: `h` is the most-processed state in the model, and the head is the
only thing after it. Anything genuinely earlier is an **intermediate-layer
tap**, which is a separate and larger change (and is where the ngram study
found its signal: L19 of 64).

So the whole proposition is: **is the coordinate a linear function of `n`
with more precision than the digit projection exposes?**

## The free version, and why it is dead

The obvious idea costs nothing and does not work, so it goes first.

At the position after `{"x":` the readout already gives ten digit logits.
Their **expectation** — `E[d1] = sum k * p(k)` — would be a continuous
estimate of `x / 100` at no cost, on the existing seam, today.

The measured trace kills it. `d1` comes back at **p = 0.993, 0.964, 0.970**
on the three scenes. A distribution that concentrated has an expectation
within 0.03 of its argmax, so `E[d1]` is the hundreds digit and nothing more:
`x = 767` reads as 700, an error of 6.7% of the side against the chain's
2.1%.

That is not a reason to drop the idea. It is the **precise statement of the
problem**: the model is certain about the hundreds place and the digit
projection at that position answers only "which hundreds digit". Everything
below the hundreds place is either still in `n` and unexposed, or not there
yet. Which of those it is has a name and a test.

## The discriminator

Two readings of the same measured trace — `d1` at p ~ 0.99 falling to `d3` at
p ~ 0.15:

**(a) The latent already holds the number.** `n` at the `x1` position encodes
x to about 1 part in 100, the chain is merely *serializing* it, and the digit
projection is a base-10 bottleneck that discards the sub-hundreds part at
each step.

**(b) The latent holds only the leading digit.** The tens and units are
*computed* at their own positions, after the model has seen its own forced
token, and the falling confidence is the model genuinely making up the rest.

**A linear probe on `n` at `x1` separates them, and that is the whole go/no-go.**
Fit `n -> (x, y)` and compare against the chain on held-out scenes.

The threshold is not a taste: **reading (b) predicts a specific number.** If
the latent holds only the digit that position is about to emit, the best any
probe can do is the centre of that digit's bucket, and the residual is the
rest of the number — about 25 units on a 0-999 scale, **2.5% of the side**.
That is the baseline to beat, and it is computed from the chain's own
outputs rather than assumed, because the chain's coordinates are not uniform
over the axis.

| probe error vs the chain | reading | consequence |
|---|---|---|
| well under the leading-digit baseline (~1% of the side) | **(a)** | a one-pass point is real; build it |
| **at** the leading-digit baseline (~2.5%) | **(b)** | the latent holds the digit it is about to emit and nothing more |
| at the predict-the-mean baseline | neither | the probe found nothing; check the pipeline before believing it |

An error *at* the baseline is the decisive negative, and it is a much
sharper test than "over 4%": it says the probe recovered exactly the
information the digit projection already exposes, and not one unit more.

Under (b), **no one-pass point exists at this position by any method** — not
the probe, not a strip readout, not a grid.

Note what (b) would mean for [spec 11](11-point-in-one-pass.md): it is not a
verdict on the probe, it is a verdict on the *position*. Every sidestep
candidate there reads one position and asks it for the whole coordinate. If
the position does not have it, the arithmetic about strip widths is arguing
about how finely to quantize something that is not there.

This is why E-P1 runs before anything else in either study.

## Fit it to the chain, not to the truth

The probe's target is **the chain's own output**, not the button's true
centre. Three reasons, and they are not a compromise:

- The chain is what `/v1/decide` claims today. A probe that reproduces it is
  a strictly cheaper way to serve the same answer, and needs no new accuracy
  argument at all.
- The failure becomes a **disagreement**, measurable per scene, rather than
  an error that could belong to the model or to the probe.
- The data is free. Ground truth needs an annotated screenshot corpus this
  repo does not have; the chain labels any screenshot, with any instruction,
  by running. `tests/fixtures/pointing/generate.py` already makes scenes with
  a known button, so N of them at random positions is a loop.

A ground-truth probe is a later question and a better one, and it is not
available yet.

## Keep the gate

A fitted probe has **no answer mass**. That matters more here than anywhere
else in the decision surface: ADR 0034 names a silent collapse of answer mass
as *"the only failure of this endpoint that nothing else would show"*, and a
probe would answer a confident coordinate to a prompt the model was about to
refuse.

So the probe does not replace the readout, it rides beside it: **the digit
readout at `x1` stays as the gate** (the ten digit logits, their mass against
the whole vocabulary, the unrestricted argmax) and the probe supplies the
*value*. Same prefill, same position, one 5120-vector more on the wire. If
the mass collapses, the decision fails exactly as it does today.

## What it would cost the engine

Less than spec 11's candidates, which is the pleasant surprise:

- **Shipping `n` is 50x cheaper than shipping the logits row.** 5120 BF16 =
  10 KB against 248,320 BF16 = 497 KB, on the *same* code path in
  `ignis_program_prefill` where `out_logits` is copied out today. The probe
  itself then runs on the host, which keeps fitted weights out of the kernel
  entirely.
- **Two positions still need spec 11's multi-position head** if x and y are
  read separately — but they may not be: one probe with four outputs answers
  a **box in one pass** as easily as a point, which is more than the ask.
- **It contradicts the letter of ADR 0034.** That ADR says the full-vocabulary
  buffer never crosses the `Compute` seam and only the answer logits do; a
  hidden state is neither. It needs an amendment, and the amendment is
  friendly — the reason for the rule was 970 KB per decision, and this is 10.
- **The probe's weights live outside the artifact**, so where they load from
  is a real design question with a precedent right beside it: the answer
  alphabet is *computed from the loaded tokenizer, never compiled in*, for
  exactly the reason a probe fitted against one artifact must be **keyed to
  that artifact** and refused against another.

## The second, better, harder probe: a heatmap over the image tokens

Rank it second, but it is the one that would clear spec 11's bar with room.

Score **every image token's hidden state** with the same kind of probe,
softmax over the positions, and the answer is a spatial distribution at the
vision grid's own resolution — at the default budget, 128 x 128 merged
tokens, one per 32 px, **0.78% of the side**, with a centroid finer than that
and a genuine 2-D covariance as the uncertainty. The cost is one GEMV of
16,384 x 5120, which is nothing beside the 27B forward it rides on.

Three things stand between it and a measurement, and they are why it is
second:

1. **Causal ordering.** Today the render is system -> user[Image, Text]
   (`classify_pointing_gpu.rs`). So the image tokens *do* see the system's
   pointing instruction, but **not the specific target** — "click the blue
   button" comes after them, and no image token can attend to it. A heatmap
   over those states can encode "what is here", not "am I the one". Moving
   the instruction before the image is possible (the parts are ordered by the
   caller) but it is a different prompt and therefore unmeasured text.
2. **Probe site.** Last-layer image-token states are frequently degenerate in
   VLMs. The live site is likely mid-stack — which is a **tap**, not a reuse
   of `final_residual`, and a much bigger kernel change. The ngram study's
   L19 is the local precedent for where to look.
3. **Supervision.** The chain produces a point, not a per-token label. The
   target would be a Gaussian around the chain's point, which is an
   assumption about the heatmap's shape rather than a measurement of it.

## The vehicle: PyTorch does all the discovery

**No engine change until a probe has a number**, and none is needed: the
ngram study already established a working vehicle on this machine and its
`visual` tower is **not quantized**, so the vision path is live there.

From `.scratch/ngram-study/results/env.json` on branch `ngram-study`:
`Qwen/Qwen3.8-27B` at revision `1d4bf0f2...`, which **matches the artifact's
pin**; loaded from `Y:/models/Qwen3.8-27B` (network share, 51.8 GiB) and
quantized on the way in by bitsandbytes 0.50.2 (NF4 + double quant, BF16
compute, cuda130/sm_120), with `lm_head`, `visual`, `in_proj_a` and
`in_proj_b` left alone. torch 2.13+cu130, transformers 5.17, python 3.12.6.

Do not re-derive that: the NVFP4 route is closed (transformers 5.17 hands
every non-FP8 compressed-tensors checkpoint to a pre-hook that decompresses
the whole model to ~54 GB of dense BF16) and the official FP8 checkpoint
leaves under 4 GiB for activations. Both are recorded in
`.scratch/ngram-study/03-risultati.md` §0.

## The experiments

**E-P0 - does the vehicle sit where the engine sits?** No fitting. On the
three committed scenes, render the same prompt, hook `hidden_states` at the
`x1` position, and check that `lm_head(n)`'s top-1 digit is the digit
`decide_point_gpu.rs` draws there. If it is not, the prompt, the position or
the rotation differs and every number fitted afterwards would be about a
different model state. This is the cheapest possible guard and it goes first.

**E-P1 - the discriminator.** Extend `tests/fixtures/pointing/generate.py` to
N >= 200 scenes with the button at random positions and varied distractors.
Run the chain in the vehicle, keeping `(n at x1, x, y)` per scene. Ridge
regression `n -> (x, y)`, 5-fold cross-validated, error reported **against
the chain** and, separately, against the button centre. Read the table in
*The discriminator* above. Report the permuted control too — a probe fitted
against shuffled labels must fail, or the pipeline is leaking
(`docs/findings/` records that trap from the ngram study: a raw cosine in the
residual sits at ~0.995 and means nothing without centring and a permuted
control).

**E-P2 - the layer sweep.** The same probe at L19, L32, L47 and the final
layer. Two questions at once: is the coordinate *more* linearly available
mid-stack, and does the answer move the probe site off `final_residual` and
into a tap.

**E-P3 - a box in one pass.** Same method, four outputs, fitted at the `x0`
position. If `n` there carries all four numbers, the box is one pass and not
two, and spec 11's second pass becomes a refinement rather than a
requirement.

**E-P4 - the heatmap.** Only if E-P2 finds a mid-stack site, and with the
instruction moved before the image so the image tokens can attend to it.
Gaussian target around the chain's point, reported as a heatmap and a
centroid.

**E-E1 - the engine.** Ship `n` at the readout position (10 KB, the
`out_logits` path), probe on the host, keyed to the artifact. Only after
E-P1 says (a).

## What this study does not establish

- Nothing is measured. Every number quoted is from the pointing finding and
  is about **three synthetic scenes with one obvious target**.
- A probe fitted on synthetic scenes says nothing about real screenshots, and
  a linear probe that fails says nothing about a non-linear one. Both are
  the same limitation the fixture already has, carried forward.
- The go/no-go is one-directional: E-P1 reading (a) says a one-pass point is
  *possible*, not that it is accurate enough to ship. It would still have to
  clear spec 11's bar — inside the button on all three scenes.
- Reading (b) would be the more valuable result, because it closes both
  studies at once, and it is the one this document exists to make cheap to
  reach.

## References

- [Spec 11](11-point-in-one-pass.md) - the vocabulary-side candidates, the
  bar every answer is measured against, and the multi-position head this
  would share.
- ADR 0034 - the readout seam, what is allowed to cross it, and why answer
  mass is the one production failure detector this endpoint has.
- `docs/findings/2026-09-19-constrained-digit-readout-points.md` - the 0-999
  native scale, the per-digit trace the discriminator is built on, and the
  three scenes.
- Branch `ngram-study` (**not merged**, worktree
  `../.inference-qwen-worktrees/ngram-study`): its
  `docs/findings/2026-09-19-codebase-as-ngram-memory.md` and
  `.scratch/ngram-study/` carry the vehicle, the closed NVFP4 route, the L19
  result and the centring/permutation traps. Nothing of it is on `main`, so
  a reader on `main` will not find those paths.
- `kernel/src/step.cu` (`ignis_program_prefill`) - where `final_residual`,
  the norm and the head are, and where `out_logits` is copied out.
