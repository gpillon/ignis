# 10 - scalar: a number that decides its own width

GitHub: #255

`number` (spec 06) forces **exactly** `digits` digits. The caller has to guess
the magnitude, the model has to fill a field it did not choose, and the whole
of #254 is the fallout of that field having two ends. `scalar` is the same
mechanism with one token added to the alphabet: the closing `}`. The run ends
when the number is complete, so `digits` is a **maximum** and a caller who has
no idea of the magnitude can leave it alone.

The same addition buys the decimal point, which is why this is a primitive and
not a flag on `number`: `scalar` answers **3**, **3.5** and **-0.25** with one
schedule and no grid.

It is `number`'s sibling, not its replacement. `number` still exists and still
answers a whole number in a declared field; `point` and `box` are untouched,
because there the width is the **scale** and not a field — a coordinate on a
0-999 axis is three digits by definition, and those numbers are measured.

## How wide a ceiling may be

A `number`'s `digits` and a `scalar`'s are not the same quantity and do not
share a bound. A width is a field the model must **fill**, so every digit of
it is a decode round spent whatever the answer is, and `number` is held to
1..=6 because that is what was measured. A ceiling is a bound the run may
close early out of, so raising it costs a caller who does not reach it
nothing at all — only the schedule, which is a list and not a spend.

| | range | when the caller says nothing |
|---|---|---|
| `number`, `point`, `box` | 1..=6 | 3, the measured width |
| `scalar` | 1..=15 | **8** |

Fifteen is where `f64` stops: every 15-significant-digit decimal round-trips
through an `f64` exactly and the sixteenth does not, so a wider ceiling would
permit a run whose spelling `value` could not carry — and `text` and `value`
disagreeing is the one thing those two fields exist to rule out.

Eight is **not** the maximum, deliberately. A caller who writes nothing is not
asking for the widest run served; they are saying they do not know the
magnitude, and eight digits covers the quantities that turn up in evidence — a
count, a duration, an amount of money — while keeping the schedule at eleven
steps. A caller who needs the other seven says so.

## The schedule

The prefix `{"value":` is prompt, as it is for `number`. Then:

| step | permitted |
|---|---|
| 0 | the ten digits, and `-` |
| 1 .. K+2 | the ten digits, `.`, and `}` |

The schedule is `K + 3` steps, not `K`, because it bounds **tokens** and the
answer spends three of them on structure: a sign, a point, and the brace. One
step fewer and `-12.3456` could never draw its terminator and would always
land at the cap.

That bound is not the digit count, and a step cannot tell a digit from a
point, so a `digits: 1` question could legally spell `123`. **The reader
enforces the count the prompt asked for** — the same division of labour the
decimal point needs, and the refusal is its own code (`too_many_digits`)
rather than `malformed_scalar`, because the run *is* a number and a caller
told otherwise would look for the fault in the wrong place.

`}` = 92, `.` = 13, `-` = 12 in the served 27B's tokenizer, each one token, and
no `.5`, `-3` or `0.` exists to compete with them — verified at load, never
compiled in, the rule [`AnswerAlphabet`] already applies to labels.

**A static schedule cannot say "at most one point".** `Schedule::step` is a
function of the index alone, and making it a function of what was drawn is a
different engine. So the alphabet permits what it can and **the reader
refuses what it must**: two points, a trailing point, a bare `-`, an empty
run. Refused and never repaired — a scalar read off a malformed run is a
wrong answer wearing the shape of a right one, which is the same reason
`run_cut_short` exists.

## Three ways a run ends, and only one of them answers

The signal is the **last token**, not the length. This table is the whole
contract:

| run | reading |
|---|---|
| ends with `}` | **terminated** — the number is what precedes it |
| no `}`, schedule unspent | **truncated** — the engine cut it off, an error |
| no `}`, schedule spent | **past the ceiling** — `too_many_digits`, an error |

The third row follows from the schedule leaving room for every structural
token: a well-formed answer can always close itself, so a run that spent the
whole schedule instead wrote at least one digit past its ceiling. "At the cap"
is a valid outcome for `number`, whose field has no terminator to miss, and
not for this.

Today's `run_cut_short` fires on length alone and would claim both of the
last two rows. It must name them apart.

## The prompt

`scalar_system` is new text and therefore unmeasured text (ADR 0034), so it is
measured here before it ships and it does not borrow `number`'s numbers. It
declares the maximum, says the value may have a decimal part, and asks the
model to close the object as soon as the number is complete.

It does **not** carry #254's alignment clause, and cannot: an instruction to
pad on the left tells the model to write `003}`, which fills the field the
terminator exists to avoid. The two are alternatives, and `number` keeps the
clause because `number` keeps the field.

## The answer

```json
{"type": "scalar", "value": 3.5, "uncertainty": 0.13, "text": "3.5",
 "digits": [{"digit": 3, "probability": 0.99}, ...]}
```

`value` is an `f64` and `text` is what the model actually wrote, because a
caller checking a reading against a trace needs the spelling that produced it.

**`uncertainty` cannot be computed the way `number` computes it.** `sigma` is
`sum((1 - p_k) * 10^place)`, and with a decimal point the place of a digit is
not known until the point has been seen — `3.5`'s digits are worth 1 and 0.1.
So the run is collected, parsed, and only then weighted by each digit's real
place. The `.`, the `-` and the `}` are steps of the run but contribute
nothing: the first two are structure and the third is the end.

## Acceptance

1. A schedule whose step permits a terminator ends the run when that token is
   drawn, with `FinishReason::Stop`, and the token is counted in
   `usage.output_tokens` — it was generated.
2. A run that spells more digits than the question allowed is
   `too_many_digits` — never a number, and never `run_cut_short`. A run that
   closed itself with steps to spare is the ordinary case, not a short one.
3. A run the engine cut short is still an error.
4. `{"value":3}` answers `3` in **two** rounds where `number` at six digits
   spends six.
4b. A question that names no `digits` is prepared at the default ceiling, not
   at the maximum, and every ceiling between `number`'s ceiling and this
   primitive's is legal — a width a field refuses is a bound a run may simply
   not reach.
5. `3.5` and `-0.25` round-trip: `value`, `text` and the per-digit trace
   agree, and `uncertainty` weights `5` in `3.5` at 0.1 and not at 1.
6. `..`, `3.`, `-` alone, and an empty run are each a 422-shaped
   `malformed_scalar` answer, never a number.
7. `number`, `point` and `box` are unchanged: `number_system` keeps its
   alignment clause, `point_system(3)` is byte-identical, and
   `decide_point_gpu.rs` passes.
8. On the GPU: the four whole-number truths of #254 at one `digits`, plus a
   declared decimal and a declared negative, each read exactly — with the
   caller never choosing a width.

## References

- ADR 0034 (a constrained decode restricts sampling per lane; the alphabet is
  computed from the loaded tokenizer).
- Spec 06 (`number`, `point`, `box`), whose fixed field this relaxes.
- Finding: `docs/findings/2026-09-21-the-number-prompt-declares-an-alignment.md`
  — why a fixed field has two ends, and the measurement that closed it.
- [Spec 08](08-scalar-readout.md) and
  `docs/findings/2026-09-21-scalar-readout-legend-width.md` are a *different*
  `scalar`: a readout over a legend of cells, built and measured and then
  left unimplemented. The name is reused deliberately. That measurement
  stands on its own — a grid's accuracy is bounded by its legend's width —
  and is the reason this primitive generates rather than reads.
