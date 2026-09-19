# Typed option logits are readable from one prefill

- Kind: experiment
- Status: current
- Observed: 2026-09-19
- Last verified: 2026-09-19
- Scope: serving / classification readout, prefill logits, `Compute` seam
- Related: `crates/server/tests/classify_readout_gpu.rs`, GitHub #72 (`out_logits`), #191/#193 (prompt reuse), SemIf (github.com/TheoLeeCJ/SemIf), TypeSafe Jev `POST /v1/systemone`
- Superseded by: none

## Question

A "semantic if" endpoint — TypeSafe's Jev shape, one `state` plus a map of
typed `questions`, answered with a probability per declared option — does not
generate text. It builds a prompt that names each option by an uppercase
letter, prefills it, and reads the last position's logits restricted to those
letters' token ids. SemIf measured that technique on a 4B.

Before building any of the plumbing that would expose it (a readout through
the `Compute` seam, a request kind that terminates at prefill, the endpoint
itself), one thing had to be established on **our** model, the served Qwen 3.8
27B NVFP4:

**Does the restricted softmax carry signal, or is it renormalized noise?**

Accuracy does not answer that. A model that puts 95% of its mass on `\n` and
distributes the remaining 5% arbitrarily across `A`/`B`/`C` can still produce
a plausible-looking argmax. The question is where the mass actually sits.

## Evidence

`crates/server/tests/classify_readout_gpu.rs`, run under the GPU profile
(`--ignored --test-threads=1`) against
`qwen3_8_27b_nvfp4full-v2.ninfer`, BF16 KV, thinking disabled.

Fixture: SemIf's `benchmarks/data/authored144.jsonl` — 144 authored decisions
across three families (`candidate_selection`, `evidence_interpretation`,
`rule_application`), each with a `state`, a `question`, 2–3 options, and a
`label` naming the authored answer. Raw rows in
`.scratch/jev-classify/readout.jsonl`; the fixture itself is not committed.

The prompt is SemIf's `direct_messages` reproduced exactly: its `DIRECT_SYSTEM`
instruction verbatim, plus one user message carrying
`{"evidence", "criterion", "options":[{"letter","description"}]}` as JSON,
evidence first. Each letter's slot is verified per row the way SemIf verifies
it — one token, exact decode round-trip, and appending it to the rendered
prompt extends the tokenization by exactly that token rather than
re-tokenizing the tail.

| Measure | Value |
|---|---|
| Rows scored / excluded | 144 / 0 |
| `argmax_in_slots` (unrestricted winner is a declared letter) | 144 / 144 (100%) |
| `allowed_mass` median / p5 / min | 0.9983 / 0.9862 / 0.9288 |
| Accuracy | 0.938 |
| Balanced accuracy | 0.934 |
| Per family | 45/48 each, all three |
| Prefill p50 / p95 | 36.5 ms / 40.7 ms |
| Throughput | 28.0 decisions/s, serial, no prefix reuse, no batching |
| Prompt tokens median / max | 132 / 156 |

`allowed_mass` is `exp(logsumexp(slot logits) − logsumexp(all logits))`.

Calibration, reading the restricted softmax's top probability as a confidence:

| Top probability | Correct |
|---|---|
| ≥ 0.9 (118 rows) | 118 / 118 |
| < 0.9 (26 rows) | 17 / 26 |

Median top probability: 0.991 on correct rows, 0.634 on the nine wrong ones.
No wrong row has low `allowed_mass` (lowest is 0.9598).

SemIf's own reference points, for orientation only — a different model (4B),
a different runtime (HF transformers), a different card (3090): balanced
accuracy 0.813 on this fixture, 2.33 decisions/s fresh and 20.03 with prefix
reuse and batched suffixes. Published Jev scores 0.883 on a different
102-row subset.

## Finding

The readout is signal, not noise, and by a wide margin.

The model's own unrestricted winner is a declared letter on **every** row, and
the declared options hold a median 99.8% of the next-token distribution — the
restriction throws away almost nothing. Every letter slot is a single token in
this tokenizer and every answer boundary is clean, so no row was excluded for
a tokenization reason.

The accuracy that follows (0.938 / 0.934 balanced) is a consequence, not the
result: it sits above the 4B reference on the same fixture, which is what a
much larger model should do and says little on its own.

The calibration is the second finding and the more useful one. The restricted
softmax's top probability separates the decisions the model gets right from
the ones it does not: every one of the 118 rows above 0.9 is correct, and the
nine errors cluster at a median 0.634. A threshold turns this into an
abstention signal rather than a silent wrong answer — worth exposing on the
endpoint as a `confidence` field, as Jev's own `choice` and `score` answers do.

At 36.5 ms and zero decoded tokens per decision, the entire GPU cost of a
decision is one prefill of ~132 tokens.

## Implications

- **Nothing below the kernel leaf needs to change.** `cuda_leaf.rs`'s
  `prefill` already calls `step::prefill_program_sampled(..., None)`, where
  that `None` is `out_logits`. The work is the plumbing above it: the
  `StepLeaf::prefill` signature, a readout on the `Compute` seam, a request
  kind that terminates at `prefill_complete` without ever taking a decode
  lane, and the endpoint.
- **The gather belongs inside `RuntimeCompute::prefill_step`.** A full-vocab
  buffer is 151,936 × f32 ≈ 607 KB per decision; only the slot logits, the
  full-vocab logsumexp and the full-vocab argmax need to cross the seam.
- **Evidence-first is load-bearing.** Jev's shape is one `state` and many
  questions. With the evidence at the head of the payload, that is one shared
  token prefix and N short suffixes — exactly the reuse ignis already has
  (#191/#193), and the reason SemIf's own shared mode is 8.6x its fresh one.
- **28 decisions/s is the floor**, measured serially with no reuse and no
  batching, both of which the engine already does.

## Limits and unknowns

- 144 rows from one authored fixture whose own provenance records it as
  "all_source_groups_independently_model_reviewed_not_human_adjudicated". The
  accuracy figure carries that fixture's biases; the mass and calibration
  figures are properties of the model's distribution and are more robust to it.
- 118/118 above the 0.9 threshold is 118 samples, not a guarantee. The
  threshold needs re-measuring on any workload before it is trusted to abstain.
- Two and three options only. Jev allows up to 255, which no single-token
  uppercase-letter alphabet reaches (16 here, ~62 across `A-Za-z0-9`). Whether
  the mass stays on-slot with 16 or 60 declared letters is unmeasured.
- The prompt is SemIf's, tuned on a 4B. `serde_json`'s compact separators
  differ from Python's `json.dumps` defaults (`,`/`:` against `, `/`: `), so
  the token sequence is not byte-identical to theirs.
- Timing is a `prefill_program` call on an idle exclusive card with the model
  already resident, not a served request: no HTTP, no scheduler, no admission.
- BF16 logits promoted to f32. SemIf saw 5–6 of 777 argmaxes move between
  execution paths at BF16; near-ties here will behave the same way.

## Follow-ups

- Measure the mass with 16 and ~60 declared options, which decides whether an
  extended alphabet or TypeSafe's documented two-stage pattern is the answer
  above 16.
- Measure a Jev-shaped request end to end: one state, N questions, against the
  existing prefix reuse — the number that says what the endpoint is worth.
- Re-measure the 0.9 calibration threshold on a workload that is not this
  fixture.
