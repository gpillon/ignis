# 07 - The Decide tab

GitHub: #247

A third Playground view, between the chat and the Monitor, where a decision
is **built, sent and read**. The endpoint exists (spec 03, spec 06) and is
reachable only by `curl`; nothing in the Playground shows what a decision
costs or what an answer looks like. This is that surface.

It is not a chat. A decision is an *instrument reading*: point it at evidence,
declare what to measure, get a distribution and an uncertainty back. The tab
is shaped as a bench — the sample on the left, the readings on the right —
and not as a message thread.

## Scope

- A new `View` value, `decide`, third in the header switch after `Playground`
  and before `Monitor`.
- A **question builder**: add, reorder, edit and remove typed questions with
  no JSON typed at all — all six primitives (`noul`, `choice`, `score`,
  `number`, `point`, `box`).
- A **JSON editor** over the same request, two-way: the builder writes it,
  editing it re-enters the builder when it parses and validates.
- An **evidence editor** in three modes: text, JSON, and image + text (OpenAI
  content parts, the shape spec 03 accepts).
- A **gallery of examples**, selectable in one click, which doubles as the
  empty state.
- An **answer panel** per primitive: the distribution, the winner, the
  confidence, the score's expected value between its levels, a number's digit
  trace, a point or box drawn on the submitted image.
- The wire cost beside the answers — `usage.input_tokens`, `output_tokens`,
  wall-clock — and the raw response body, collapsed.

## Out of scope

- No new server code. This tab only speaks the endpoint spec 03 and spec 06
  already serve.
- No **answer mass**. It is deliberately not a response field (spec 03, ADR
  0034): the probabilities are renormalized after the restriction, so nothing
  in the body can reconstruct it. The tab says so once and points at the
  Monitor's histogram (spec 05) rather than computing a number that would be
  1.0 by construction.
- No editor dependency. The JSON editor is a `textarea` in the mono face.

## Decisions

- **One request per run.** Every question goes in one `POST /v1/decide`, which
  is what the endpoint is for: the evidence is prefilled once and the
  questions share it (spec 04). A per-question button would hide the only
  property worth showing.
- **Order is model state, not presentation.** `questions` and a `choice`'s
  `criteria` live in the TS model as **arrays** of `{id, ...}` / `{key,
  description}`, never as objects. The server reads both in order of
  appearance because a different order is a different prompt (spec 03), and a
  JS object cannot carry that: `JSON.stringify` hoists integer-like keys and
  sorts them ascending, so option keys `3, 1, 2` would leave the browser as
  `1, 2, 3`. The request body is therefore **serialized by hand**, and a test
  pins exactly that case.
- **The response's order is not the request's.** `answers`, `probabilities`
  and `legend` are `BTreeMap`s, so they arrive sorted alphabetically. Every
  panel renders in the *request's* declared order, and a `score`'s level keys
  are parsed to integers before use — `"10" < "2"` as strings.
- **The builder cannot produce a 422.** Client validation mirrors spec 03's
  refusals one for one, and `Decide` stays disabled while anything is
  invalid, naming the offending question. A real 422 is still rendered (a
  refusal this tab does not know about is a bug in this tab), through
  `apiErrorMessage`, and a 401 goes through `keyRequired()` like every other
  call.
- **Thinking is never sent.** Not a field the tab exposes, not a field it
  emits — spec 03 answers 422 for it, and there is nothing to gain by
  offering a control whose only outcome is a refusal.
- **A `point` or `box` needs an image.** The builder says so where the
  question is, before the send, rather than letting the per-question
  `state_carries_no_image` error come back.

## Client validation, mirroring spec 03

| Rule | Refusal it prevents |
| --- | --- |
| At least one question | `no_questions` |
| Question ids non-blank and unique | `duplicate_question` |
| `instructions` non-empty | `empty_instructions` |
| `noul`: `true`/`false` descriptions non-empty when given | `malformed_criteria` |
| `choice`: at least one option, keys non-blank and unique | `no_options`, `unclean_option`, `duplicate_option` |
| `choice`: at most 256 options | `too_many_options` |
| `score`: at least 2 levels, each non-empty | `too_few_levels`, `malformed_criteria` |
| `number`/`point`/`box`: no `criteria` | `criteria_unsupported` |
| `noul`/`choice`/`score`: no `digits` | `digits_unsupported` |
| `digits` in 1..=6 | `digits_out_of_range` |

`alphabet_exhausted` is a property of the load, not of the request, so it is
rendered, not predicted.

## Design

The look is the Playground's: kiln band, ember accent, the 45 degree `.cut`,
Chakra Petch for display and figures, Instrument Sans for prose, JetBrains
Mono for JSON. No new palette — the tokens in `styles.css` carry both
schemes already.

**Colour.** The distributions are one hue. Bar *length* encodes probability,
so shading a bar by the same number would double-encode it and burn the only
free channel; every bar is `--ember`, at one step, and the winner is marked
by its label going to ink-semibold with an ember chip — not by a second
colour. The only second mark in the panel is the `score`'s expected-value
pointer, which earns itself because the score genuinely lands *between* the
levels the bars sit on. `--good`/`--warn`/`--fault` stay reserved for state
(valid, unsent, refused).

**Type.** Three roles, no fourth: figures and headings in the display face
with `tabular-nums`, prose in the sans, JSON and digit traces in the mono.
Option labels are sentence case, as written by whoever declared them — the
tab never upper-cases a label it was handed.

**Layout.** Two columns on a wide screen, the bench and the readings, and one
stacked column below `lg` with the answers first once a run lands.

```
+-------------------------------------------------------------------------+
| ignis   Playground . Decide . Monitor                 qwen3-8b o ready  |
+----------------------------------+--------------------------------------+
| Evidence           text json img |  Answers        312 in . 0 out . 84ms|
| +------------------------------+ |                                      |
| | Help! My payouts have been   | |  is_urgent                      noul |
| | failing for 3 days.          | |  ################........  0.86      |
| +------------------------------+ |  Explicitly time-sensitive           |
|                                  |                                      |
| Questions              Build JSON|  department                   choice |
| +------------------------------+ |  billing    #################  0.91 *|
| | is_urgent   [noul   v]     x | |  technical  ##                0.07   |
| | Does this convey urgency?    | |  sales      #                 0.02   |
| | yes  Explicitly time-sensit. | |  confidence 0.91                     |
| | no   No urgency expressed    | |                                      |
| +------------------------------+ |  frustration                   score |
| +------------------------------+ |  Calm  Frustrated  Very angry        |
| | department  [choice v]     x | |  #     ######      ############      |
| | ...                          | |  |----------+-------v-----|  1.6     |
| +------------------------------+ |                                      |
| + noul  choice  score  number .. |  > Raw response                      |
|                                  |                                      |
|                      [ Decide ]  |                                      |
+----------------------------------+--------------------------------------+
```

**The one orchestrated moment, and it is information.** When a run lands,
every readout answer's bar grows from zero to its value *at the same time*,
over ~260 ms — because they genuinely arrived in the same instant: a readout
generates nothing and all of them come out of one prefill. A `number`,
`point` or `box` is a constrained decode, one forced token per digit, so its
digits land **left to right**, one per ~40 ms. The two reveals differ because
the two mechanisms differ; a reader learns which primitive costs a round per
digit by watching it. Nothing else on the page moves on its own, and
`prefers-reduced-motion` renders both at rest.

**Principles.**
1. The answers are the hero; the builder is a quiet form around them.
2. Every figure is also text, and every panel is readable with no pointer.
3. The tab never says something the body cannot support — no answer mass, no
   uncertainty presented as a bound.
4. Structure encodes the primitive: a `noul` is one bar, a `choice` is a list,
   a `score` is a list *plus an axis*, a `number` is places, a `point` is an
   image.

**What was reconsidered.** The first plan gave each option in a `choice` its
own series colour and a stacked mass bar. That is the generic dashboard
answer and it is wrong twice: 256 options cannot have 256 hues, and a stacked
bar of a renormalized distribution says "part of a whole" about a whole that
the response cannot show (the answer mass). One hue, one bar per option,
sorted in declared order.

## Acceptance

1. The header shows `Playground . Decide . Monitor` when metrics are on, and
   `Playground . Decide` when they are off — the Decide tab does not depend
   on the metrics listener, and turning metrics off while in Decide leaves
   the view where it is.
2. The chat stays mounted under the Decide tab: a reply streaming in the
   Playground is still streaming when the tab comes back.
3. An example loads into both editors: picking one fills the builder, and the
   JSON view shows the same request.
4. A `choice` whose option keys are `"3"`, `"1"`, `"2"` is sent in that order.
   (A test on the serialized body, not on an object.)
5. Editing the JSON to something invalid shows the error and leaves the
   builder's model untouched; fixing it re-enters the builder.
6. Every rule in the validation table above disables `Decide` and names the
   question, with a unit test per row.
7. A `score`'s levels render in index order with ten or more levels, and the
   expected value sits between the two levels it falls between.
8. A `number`'s answer shows one column per place with its probability, and
   the value with its uncertainty in the value's own units.
9. A `point` and a `box` draw on the submitted image at the answer's pixels,
   labelled as the model's own uncertainty and not as a bound.
10. A 422 renders the endpoint's own message; a 401 shows the key prompt.
11. A per-question `Error` answer renders beside its siblings' answers rather
    than replacing them.
12. `npm run typecheck` and `npm test` pass in `web/`, and `cargo test` passes
    workspace-wide.

## References

- Spec 03 (the wire shape, the refusals), spec 04 (the fan-out), spec 05
  (the answer-mass histogram), spec 06 (`number`, `point`, `box`).
- ADR 0026 (the Playground is embedded when built), ADR 0034.
- `crates/server/src/decide.rs` is the contract; `crates/server/src/numbers.rs`
  carries `DigitDraw` and the digit range.
