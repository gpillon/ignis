# 03 - POST /v1/decide

GitHub: #239

The endpoint, with the three one-position primitives. Wire shape copied from
TypeSafe's Jev (`POST /v1/systemone`) so an unmodified Jev client reaches it by
changing the URL.

- Route `/v1/decide`, with `/v1/systemone` as an alias.
- Request: `{state, model, questions: {<id>: {type, instructions, criteria}}}`.
  Response: `{model, answers: {<id>: ...}, usage}`.
- `noul` - two answer tokens; `criteria {true, false}` supply their
  descriptions. Answers `{type: "noul", noul: p}` where `p` is the probability
  of the true option.
- `choice` - `criteria` is a map of id to description; `null` means the id is
  its own description. Answers `{type, choice, probabilities, confidence}`.
- `score` - `criteria` is an ordered array of at least two levels. The score is
  the **expected value** of the distribution over level indices (Jev's own `1.6`
  for `{0: 0.05, 1: 0.3, 2: 0.65}`), with `legend` and `probabilities`.
- `confidence` is the **top probability**, documented as such. For `score` it is
  `1 - (standard deviation / half the level range)`: a distribution straddling
  two adjacent levels is confident, not uncertain.
- `usage.output_tokens` is `0`, honestly.
- Their names are taken verbatim; ours (`boolean`, `question`, `options`) are
  read as serde aliases.
- `state` accepts a string, a JSON object or array, **and** OpenAI content
  parts, so the evidence may be an image.
- **Option order is an input, not a presentation.** `criteria` as a JSON map is
  read in order of appearance (`preserve_order`), because a different order is a
  different prompt. Documented.
- Ceiling: 256 options per question, stated as "measured this far" rather than
  as a property of the model.
- Thinking: forced off. A request that asks for it is a 422, never a silent
  ignore.
- Validation is all-or-nothing **before** the GPU: malformed options, unclean
  labels, counts out of range, an empty `instructions` all refuse the whole
  request without spending a prefill.

## Acceptance

1. Jev's own documented request examples for `noul`, `choice` and `score`
   produce responses of Jev's documented shape.
2. A `score` over `{0: 0.05, 1: 0.3, 2: 0.65}` returns `1.6`.
3. `criteria` with a `null` value uses the id as the description.
4. The same question with its options in two different orders is two different
   prompts (and may be two different answers) - pinned by a rendered-prompt
   test, not a GPU run.
5. `enable_thinking: true` is a 422 naming the reason.
6. 257 options is a 422; 256 is served.
7. A malformed question refuses the whole request with no prefill performed.
8. `state` as content parts with an image renders the image into the prompt.
9. `usage.output_tokens` is 0 and `ignis_decoded_tokens_total` does not move.

## References

- ADR 0034; `docs.typesafe.ai/api` for the wire shape.
- Findings: `2026-09-19-typed-option-logit-readout.md` (ceiling, calibration,
  vision).
- Spec 02 (the request kind), spec 04 (fan-out is separate).
