# The number prompt declared a width and not an alignment

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: server / `/v1/decide` constrained decode, prompt
- Related: [ADR 0034](../adr/0034-the-leaf-answers-without-generating.md),
  [a fixed-width number is wrong at exactly one width](2026-09-20-number-width-and-decide-e2e.md)
  (whose width rule this corrects),
  [constrained digit readout](2026-09-19-constrained-digit-readout-points.md),
  [GitHub #254](https://github.com/gpillon/ignis/issues/254)
- Superseded by: none

## Question

Asked *"for how many days have the payouts been failing?"* over evidence that
says **three**, `number` at `digits: 3` answers **300** — and the first digit
comes back at p = 0.998, so nothing in the trace flags it.

`numbers.rs::DIGITS` documented a rule for this: the model left-aligns, so a
width of exactly one more than the value's own multiplies by ten, while two or
more digits of headroom make it pad on the left and the answer is right again.
Three is one digit and the field was three wide — two of headroom — so by that
rule the answer should have been 3. Why was it 300, and what is the real rule?

## Evidence

A live server on the served 27B (`qwen3.8-27b`, hq-e8-2b), driven over HTTP.
Four truths stated verbatim in their own evidence, at every width in `DIGITS`,
with the whole per-digit trace kept: `.scratch/number-alignment/`.

### Before: the documented rule fails on all four truths

`*` marks the answer that equals the truth. Widths below the truth's own can
only truncate and are not counted.

| truth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 3 | 3\* | 30 | **300** | 3000 | 3\* | 3\* |
| 2 | 2\* | 22 | 200 | 2000 | 20000 | 2\* |
| 47 | 4 | 47\* | 470 | 47\* | 4700 | 47\* |
| 128 | 8 | 12 | 128\* | 128\* | 1280 | 128\* |

**11 of 21** widths that could hold the value read it. The documented rule is
refuted by three rows on its own terms: 3 at three digits (two of headroom) is
300, 47 at five (three of headroom) is 4700, 128 at five (two of headroom) is
1280.

There is no width that is safe for every truth, and the widths that work are
not the same set from row to row.

### The mechanism is a choice made at the first digit

Reading the traces rather than the answers: when the model pads on the left,
the first digit is a `0` at p = 0.48-0.91 and every later digit sits at ≈ 1.
When it left-aligns, the first digit is the value's own leading digit at
p ≥ 0.93 and the trailing zeros follow at 0.85-1.00.

So the model is not unsure of the *number*. It is deciding which end of the
field to pad, once, at the first position — and after that the rendering is
mechanical. The **schedule forces exactly K digits and carries no way to say a
number is finished**, so a value narrower than its field can only come out
left-aligned or zero-padded; `number_system` declared the width and named
neither.

### After: naming the alignment closes it

Appending *"right-aligned and padded on the left with zeros"* to the caller's
own `instructions` — no code, `number_system` untouched — took the same walk to
**21 of 21**. Shipping the clause in `number_system` itself, where it rides the
system block, reproduces that:

| truth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 3 | 3\* | 3\* | 3\* | 3\* | 3\* | 3\* |
| 2 | 2\* | 2\* | 2\* | 2\* | 2\* | 2\* |
| 47 | 4 | 47\* | 47\* | 47\* | 47\* | 47\* |
| 128 | 1 | 12 | 128\* | 128\* | 128\* | 128\* |

The padding zeros go from contested to certain with it: 0.48-0.91 before,
0.51-1.00 after with most above 0.95. The remaining wrong cells are widths
narrower than the truth, which can only truncate.

## Finding

**Observed.** `number`'s answer depended on an alignment the prompt never
named. The model chose one at the first digit — `0` at p = 0.48-0.91 for
right-alignment, the value's leading digit at p ≥ 0.93 for left — and 10 of the
21 widths that could hold the value came back multiplied by a power of ten.
Naming the alignment in the prompt makes every such width correct, on both
this fixture's one-, two- and three-digit truths.

**Inferred.** The failure was never about the width. It was about a constrained
decode of fixed length having no way to express "this number is finished", and
a prompt that left the only available workaround unspecified. Any scheme that
forces exactly K digits has this hole; naming the alignment fills it without
touching the schedule.

**Corrects.** `docs/findings/2026-09-20-number-width-and-decide-e2e.md` reports
that a width exactly one past the value's own is the broken case and that two
or more is safe. The first half holds on this fixture; the second does not, and
the earlier run did not walk widths far enough past the value to see it. The
tell it reports — a first digit at 0.65-0.71 — is real but is a symptom of the
alignment being contested, not of the width being wrong: the 300 that started
this had a first digit at 0.998.

## Implications

- `digits` becomes what a caller assumes it is: a field that holds the answer.
  Any width from the value's own width up is correct, and the advice to leave
  "at least two digits of headroom, never exactly one" is retired.
- `point` and `box` were never affected and are untouched. A coordinate on a
  0-999 scale fills its field, so the ambiguity had nothing to act on, and
  their prompts stay byte-identical to the ones the pointing finding measured.
- The clause is prompt-only: there is no padding code to regress, so it is
  pinned by a test at every width. A reword that dropped it would restore the
  bug silently.
- It weakens the case for a `scalar` readout ([#252](https://github.com/gpillon/ignis/issues/252)),
  whose pitch was partly that it has no width trap. Neither does `number` now,
  and `number` is the more accurate of the two.

## Limits and unknowns

- Four truths, three magnitudes, one model, one fixture. The alignment claim is
  strong (21/21 against 11/21) but the fixture is small and every truth is
  stated literally in its evidence — none of them asks the model to compute.
- Nothing here says the model's *arithmetic* improved. A value it has to derive
  is as wrong as it was; only the rendering is fixed.
- The clause was measured in two positions — appended to `instructions` and
  shipped in the system block — and both give 21/21. No other wording was
  tried, so nothing says this phrasing is the best one, only that it works.
- Whether a terminator in the schedule (letting a run end early and making
  `digits` mean "at most K") would be better is untested. It would also cut
  rounds on short answers, which this does not.

## Follow-ups

- A terminator in the schedule, which would make the width irrelevant rather
  than merely harmless, and would end a short run early.
- `Draw` carries only the winning digit's probability; the restricted softmax
  behind it would give `E[number]` and a real interval for ten f32 per step.
- Re-measure the 0.65-0.71 tell now that its cause is known: it may still be a
  usable abstention signal for a *contested* answer, which is a different claim
  from the one it was documented under.
