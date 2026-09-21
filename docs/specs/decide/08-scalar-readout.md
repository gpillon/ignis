# 08 - the scalar readout

GitHub: #252

> **NOT IMPLEMENTED.** This spec was written, built and measured, and then
> **superseded by [spec 10](10-scalar.md)** before it was merged. The name
> `scalar` now belongs to the constrained decode described there — a number
> that closes its own object — and not to the readout described below.
>
> It is kept, unimplemented, for two reasons. The number it establishes is
> durable and is cited by spec 10: a readout over a grid is accurate only
> while the legend stays narrow, 4/5 correct at 20 and 32 cells against 0/6
> at 100, which is *why* the primitive that shipped generates rather than
> reads (`docs/findings/2026-09-21-scalar-readout-legend-width.md`). And a
> reader who finds the gap between 07 and 09 should find out what was tried,
> rather than assume nobody thought of it.
>
> The code that implemented it is gone: it lived on the branch
> `scalar-readout`, which was deleted rather than merged. Only the study
> remains.

A **scalar**: one number, read from one position, over a grid the caller
declares. `number` (spec 06) generates its answer a digit at a time because a
readout names one vocabulary entry and this tokenizer has no multi-digit token
â€” `\p{N}` is isolated in the pre-tokenizer, so `10` is two tokens. A grid
sidesteps that by naming *cells* rather than digits: cell `i` is labelled with
a single-token answer label and described by its own value, and the readout
picks one.

This ships **beside** `number`, not in place of it. Measured, it is not more
accurate: `number` reads 58 and 25 exactly where a step-5 grid reads 60 and 20.
What it buys is one prefill and **zero decoded tokens**, no lane and no
residency â€” so it composes into a fan-out the way `choice` does â€” and it has no
width trap, because it never writes a digit.

- `{"type": "scalar", "min": 0, "max": 100, "step": 5}` â€” the grid is
  `min + i*step` for `i` in `0..=(max-min)/step`. `min`/`max`/`step` are
  refused on every other primitive, and `digits` and `criteria` are refused on
  this one: the same policy `digits` already has, for the same reason â€” a
  caller who wrote a field meant something by it.
- **The grid is computed in fixed point, never in `f64`.** The scale is the
  greatest number of decimals among the three literals, and a `step` that does
  not divide `max - min` exactly is **refused, never rounded or widened**.
  `0.1` three times is not `0.3`, and a grid whose last cell is not `max` is a
  grid the caller did not ask for. A bound is accepted by **enumerating the
  shape that works** â€” sign, digits, point, digits, at most four decimals â€”
  rather than by listing what to reject, so no spelling nobody thought of
  walks past it. Note what that does and does not catch: by the time a bound
  is read, `serde_json` and `f64::to_string` have already turned a caller's
  `1e2` into `100`, so what is left to refuse is a magnitude ryu itself
  re-emits in exponent form.
- **The cell count is capped at 32, and the cap is a measurement.** Beyond it
  the model stops reading the legend reliably: 4/5 correct at 20 and 32 cells,
  2/6 at 48, 0/6 at 100 (finding, `width.py`). The cap is refused rather than
  coarsened â€” silently widening a caller's step is a wrong answer that looks
  right.
- **The value is the argmax cell, not the probability-weighted mean.** A
  `score` returns the mean because a score is an ordinal judgement between
  named levels; a scalar has a true value, and on every set measured the mean
  is worse (0.00 against 1.24 median error on the main sweep, and low-biased on
  every stepped percentage). The mean is reported beside the value as `mean`,
  because a caller that wants the interval needs it and the distribution is
  ordered.
- **The prompt is `score`'s prompt, byte for byte.** `DIRECT_SYSTEM`, options
  as `{description, letter}` in that key order. That is what the experiment
  measured, and a reworded instruction is an unmeasured one (ADR 0034). The
  cell's description is its value formatted at the grid's decimals, which is
  the widest of the three bounds' and not the step's alone â€” `min: 0.5,
  step: 1` has to spell `0.5`, and `5`, not `5.0`, when all three are
  integers.
- `confidence` is the **spread** of the distribution, not its height â€” a
  score's measure, because a grid's cells are ordered and a split between two
  adjacent ones knows where the answer is. It is therefore not the quantity
  the option-readout finding's abstention threshold was measured on; that one
  is the top probability, and it is in `probabilities`.
- `probabilities` is keyed by the cell's **value**, not its index. `score`
  keys by index because a level's value *is* its index; a scalar's is not, and
  a caller reading `{"60": 0.34}` should not have to consult a legend to learn
  that cell 12 means 60.

## Acceptance

1. `{"min": 0, "max": 100, "step": 5}` yields 21 cells labelled `0`, `5` â€¦
   `100`, and the answer's `value` is one of them.
2. A `step` that does not divide `max - min` exactly is a 422 naming the
   question; so is `max <= min`, a `step` of zero or below, and a grid past
   32 cells. Two cells is the floor and is served.
3. `0.1`-scale grids are exact: `{"min": 0, "max": 1, "step": 0.1}` is 11
   cells ending at `1.0`, with no cell reading `0.30000000000000004`. Every
   cell of a grid is spelled to the same number of decimals.
4. `min`/`max`/`step` on a `noul`, `choice`, `score`, `number`, `point` or
   `box` is a 422; `digits` or `criteria` on a `scalar` is a 422.
5. `value` is the argmax cell and `mean` is the probability-weighted value;
   on a distribution whose mass sits on one cell they agree.
6. `usage.output_tokens` is 0 for a scalar question,
   `ignis_decisions_total{type="scalar"}` counts one per question, and
   `ignis_decision_answer_mass` observes it â€” a scalar is a readout.
7. The prompt a scalar builds is byte-identical to the prompt the same grid
   would build as a `score` whose levels are the cell values.
8. On the GPU, over the committed fixture, an **ordered** grid of at most 32
   cells reads the evidence-grounded truth on the direct-lookup items.

## References

- ADR 0034 (a readout receives logits and draws no sample; the answer alphabet
  is computed from the loaded tokenizer).
- Finding: `docs/findings/2026-09-21-scalar-readout-legend-width.md` â€” the
  width walk the cap comes from, the permuted-legend test, and the three
  things it does **not** establish.
- Spec 06 (`number`, `point`, `box`), whose `min`/`max` was left open for
  exactly this reason.
- Raw material: `.scratch/scalar-readout/` — on disk, not in the tree,
  like every other finding's.
