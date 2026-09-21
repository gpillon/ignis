# A number that closes its own object needs no width from its caller

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: server / `/v1/decide` constrained decode, engine / schedule stopping condition
- Related: [ADR 0034](../adr/0034-the-leaf-answers-without-generating.md),
  [the number prompt declares an alignment](2026-09-21-the-number-prompt-declares-an-alignment.md),
  [a scalar reads its legend](2026-09-21-scalar-readout-legend-width.md),
  [GitHub #255](https://github.com/gpillon/ignis/issues/255)
- Superseded by: none

## Question

`number` forces exactly `digits` digits. GitHub #254 established that the
resulting field has two ends and that the prompt has to say which one to pad;
naming it made every sufficient width correct. It did not make the width
unnecessary, and a caller still has to guess a magnitude, and still pays a
decode round per digit of a field the answer does not fill.

The schedule is a list of permitted sets, one per step. Can it simply permit
the token that *closes the answer*, so the run ends when the number is
complete — and if it can, does the model use it?

## Evidence

The served 27B (`qwen3.8-27b`, hq-e8-2b), over HTTP through the whole stack:
the chat template, the scheduler's schedule, the leaf masking its own logits.
`crates/server/tests/decide_scalar_gpu.rs` is the run below.

### The three tokens exist, singly

Checked in the artifact's own `tokenizer.json`, not assumed: `}` is 92, `.`
is 13 and `-` is 12, each one vocabulary entry. No `.5`, `-3` or `0.` exists
to compete with them — this tokenizer isolates `\p{N}`, so digits and
punctuation are always separate tokens. The check is done at load against the
loaded tokenizer, for the reason ADR 0034 gives for answer labels: a
two-token `.` would put a step's draw against a token belonging to something
else.

### The model closes the object on its own

Unconstrained, under a system text that only says *"reply with only a JSON
object of the form `{"value":N}`"*, the model writes exactly `{"value":3}` —
one digit, then the brace. So the token the run needs to end on is the one it
would write anyway.

### Six truths, no width, all read

No question names `digits`. Every truth is stated literally in its own
evidence; the last two are the half `number` cannot express at all.

| question | truth | read | wrote | tokens | uncertainty |
|---|---:|---:|---|---:|---:|
| days failing | 3 | 3 | `3` | 2 | 0.0005 |
| emails sent | 2 | 2 | `2` | 2 | 0.0038 |
| invoices | 47 | 47 | `47` | 3 | 0.0549 |
| rejected | 128 | 128 | `128` | 4 | 0.0441 |
| average hours | 2.5 | 2.5 | `2.5` | 4 | 0.0001 |
| balance moved | -0.75 | -0.75 | `-0.75` | 6 | 0.0001 |

Every run is the characters written plus the brace that closed it. `number`
at six digits spends six rounds on each of these, and at any fixed width
cannot produce the last two at all.

## Finding

**Observed.** Permitting the closing brace from the second step on lets a
constrained decode end when its answer is complete. On six evidence-grounded
truths the model closed every run itself, read every value exactly, and spent
between two and six rounds instead of a fixed six. A decimal point and a sign
in the same alphabet give `2.5` and `-0.75` off the same schedule.

**Inferred.** The fixed field was never load-bearing. It existed because a
schedule had no way to express "done", and every problem downstream of it —
the caller's guess at a magnitude, the alignment that GitHub #254 had to
name, the padding that then had to be excluded from the uncertainty — was a
consequence of that one gap rather than of anything about numbers.

**The cost is that a short run stops being self-evidently a fault.** Until
now, a run shorter than its schedule was an engine that cut it off, and the
refusal protecting against a place-weighted sum over half a number was simply
a length check. With a terminator there are three outcomes and only one of them
answers, told apart by the **last token** and never by the length: ended on
the terminator (complete), stopped without one with steps left (cut off), or
spent the whole schedule without one. A reader that kept the length check
would refuse every correct short answer.

The third outcome is worth stating because it is not the obvious one. The
schedule leaves room for the sign, the point *and* the brace, so a well-formed
answer can always close itself — which means a run that spent every step
instead wrote at least one digit past the ceiling the prompt declared. "At the
cap" is a valid answer for `number`, whose field has no terminator to miss,
and is always an error here.

**A static schedule cannot spell every rule it needs.** `Schedule::step` is a
function of the index alone, so "at most one decimal point" is not
expressible: the alphabet permits `.` at every step after the first and the
*reader* refuses `1..2`, `3.`, a bare `-` and an empty run. That division —
permit what the schedule can, refuse what it cannot — is the price of keeping
the permitted set independent of what was drawn.

## Implications

- `digits` becomes a ceiling a caller can leave out, which is what it should
  always have been for a quantity whose magnitude the caller does not know.
- The rounds a scalar spends are the rounds its answer needs. That is a
  latency property, not a throughput one: the saving is on a lane that would
  otherwise sit forced-padding a field.
- `number` keeps its fixed field and #254's alignment clause. The two are
  alternatives — an instruction to pad on the left tells the model to write
  `003}`, which fills the field the terminator exists to avoid — so a prompt
  cannot carry both.
- `point` and `box` keep theirs for a different reason: there `digits` is the
  **scale**, not a field. A coordinate on a 0-999 axis is three digits by
  definition and those numbers are measured.
- It supersedes the readout `scalar` of
  [the legend-width finding](2026-09-21-scalar-readout-legend-width.md) in
  practice. That measurement stands on its own — a grid's accuracy is bounded
  by its legend's width — and is the reason this primitive generates rather
  than reads.

## Limits and unknowns

- Six truths, one model, one fixture, and every truth stated literally in its
  evidence. Nothing here says the model's arithmetic improved: a value it has
  to derive is as wrong as it ever was.
- The system text is new, so it carries none of `number`'s figures and only
  this run's. No other wording was tried.
- The terminator was measured on `scalar` alone. Whether letting `number`
  end early would be better than its fixed field is untested, and would
  change a primitive whose numbers exist.
- Nothing has measured a scalar whose answer genuinely needs all six digits.
  The schedule can spell it and close it, and the CPU tests cover the shape,
  but no live row has written one.

## Follow-ups

- `Draw` carries only the winning token's probability. The restricted softmax
  behind it would give `E[value]` and a real interval for ten f32 per step,
  and a scalar is where that would pay: its uncertainty is already in the
  value's own units.
- A terminator would let `point` and `box` fold their separators into a
  cheaper shape, which is a different saving from this one and needs its own
  measurement.
