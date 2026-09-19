# Typed option logits are readable from one prefill

- Kind: experiment
- Status: current
- Observed: 2026-09-19
- Last verified: 2026-09-19
- Scope: serving / classification readout, prefill logits, `Compute` seam
- Related: `crates/server/tests/classify_readout_gpu.rs`, `classify_vision_readout_gpu.rs`, `slot_alphabet.rs`, GitHub #72 (`out_logits`), #178 (multimodal prefill), #191/#193 (prompt reuse), #235 (shared-suffix study), SemIf (github.com/TheoLeeCJ/SemIf), TypeSafe Jev `POST /v1/systemone`
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
`label` naming the authored answer. Committed verbatim at
`crates/server/tests/fixtures/semif/` (MIT; see its `NOTICE.md`), because a
missing fixture is a hard failure under the GPU profile. Raw per-row output in
`.scratch/jev-classify/readout.jsonl`.

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

### How many answer slots exist

`crates/server/tests/slot_alphabet.rs`, CPU-only. Labels that are exactly one
token and decode back to themselves, in this tokenizer:

| Alphabet | Clean single tokens |
|---|---|
| `A`–`Z` | 26 / 26 |
| `a`–`z` | 26 / 26 |
| `0`–`9` | 10 / 10 |
| `AA`–`ZZ` | 562 / 676 |
| `aa`–`zz` | 631 / 676 |
| `0`–`255` | **10 / 256** |
| Pooled, distinct | **1,255** (0 id collisions) |

### The same readout over an image

`crates/server/tests/classify_vision_readout_gpu.rs`, under the GPU profile:
the four vision canary images as evidence, the criterion and options as text,
three close distractors each (`42`/`47`/`74`, `Two`/`Three`/`Four`, a
near-miss of the same error string).

| Canary | Tokens (image columns) | Correct | `allowed_mass` | Prefill |
|---|---|---|---|---|
| number | 286 (196) | yes | 1.0000 | 76.9 ms |
| colour | 152 (64) | yes | 0.9992 | 39.8 ms |
| circles | 183 (96) | yes | 0.9998 | 39.2 ms |
| text | 200 (100) | yes | 0.9997 | 38.4 ms |

4/4 correct, `argmax_in_slots` 4/4, top probability ≥ 0.999 on every row.

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

**The label alphabet is not the ceiling.** 1,255 distinct single-token labels
exist here, five times Jev's 255-option limit, so an extended alphabet
(`A`–`Z`, `a`–`z`, `0`–`9`, then bigrams) removes the 16-option cap outright
and TypeSafe's documented two-stage pattern is not needed for the mechanism.
The one alphabet that does **not** work is the obvious one: numbering options
`1`..`255` would read the logit of `"1"` for options 1, 10 and 100 alike,
because only `0`–`9` are single tokens.

**The readout holds over an image.** `prefill_program_multimodal` takes the
same `out_logits` buffer, and on the four vision canaries the declared options
hold ≥ 0.999 of the distribution with the unrestricted winner in slot every
time. Four rows cannot support an accuracy claim and none is made; what they
establish is that the mechanism does not care whether the evidence is text or
pixels.

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
- **Evidence-first costs nothing and keeps reuse possible.** Jev's shape is one
  `state` and many questions; with the evidence at the head of the payload that
  is one shared token prefix and N short suffixes. It does **not** follow that
  prefix reuse pays here. A traversal on ignis is ~19 ms fixed
  (`2026-09-18-prompt-reuse-tax-on-short-ttft.md`), and splitting a 132-token
  prompt into prefix plus suffix is two traversals where one cost 36.5 ms —
  reuse only earns its keep once the `state` itself is hundreds of tokens. The
  likely win on short decisions is the **batched prefill** the scheduler
  already does: prompts this short do not fill the GEMM
  (`2026-09-11-prefill-chunk-wall-time.md`). SemIf's own 8.6x from shared state
  was measured on a different engine and does not transfer.
- **28 decisions/s is the floor**, measured serially with no reuse and no
  batching, both of which the engine already does.
- **An image-evidenced decision is the one thing Jev cannot do.** Its `state`
  is `string | object | array`. If ignis's takes content parts, the endpoint
  is a superset rather than a clone — at the cost of the media encode (the
  image columns dominate: 196 columns cost 76.9 ms against 64 columns'
  39.8 ms).
- **N questions over one state fan out as N internal requests** (option A):
  nothing new in the scheduler, and the existing batched prefill groups them
  because they arrive together. The single-request, N-suffix alternative is
  GitHub #235, and the measurement that decides it is how well A already
  groups.

## Limits and unknowns

- 144 rows from one authored fixture whose own provenance records it as
  "all_source_groups_independently_model_reviewed_not_human_adjudicated". The
  accuracy figure carries that fixture's biases; the mass and calibration
  figures are properties of the model's distribution and are more robust to it.
- 118/118 above the 0.9 threshold is 118 samples, not a guarantee. The
  threshold needs re-measuring on any workload before it is trusted to abstain.
- Two and three options only, on both fixtures. The slot count above is a
  property of the *tokenizer*; whether the model still puts its mass on the
  declared slots when there are 60 or 255 of them — and when the labels are
  `AA` and `JK` rather than the `A`/`B`/`C` of every multiple-choice question
  it was trained on — is the measurement that sets the real ceiling, and it
  has not been made.
- The vision rows are four, with a hand-written option set. They are a
  mechanism check. The text path's boundary verification has no counterpart
  there either: `prepare_prompt` returns scattered token ids rather than a
  string to append a letter to, so slot cleanliness is checked but the answer
  boundary is assumed from the shared `<|im_start|>assistant
` tail.
- The prompt is SemIf's, tuned on a 4B. `serde_json`'s compact separators
  differ from Python's `json.dumps` defaults (`,`/`:` against `, `/`: `), so
  the token sequence is not byte-identical to theirs.
- Timing is a `prefill_program` call on an idle exclusive card with the model
  already resident, not a served request: no HTTP, no scheduler, no admission.
- BF16 logits promoted to f32. SemIf saw 5–6 of 777 argmaxes move between
  execution paths at BF16; near-ties here will behave the same way.
- Measured with `KvFormat::Bf16`. The served default is **hq-e8-2b**
  (`make config`), whose near-ties can land differently (GitHub #160). The
  mass and in-slot figures are far from any margin that format could move,
  but the nine errors — whose top probability sits at a median 0.634 — are
  exactly the rows where it might. Re-measure under hq before the calibration
  threshold is trusted.

## Follow-ups

- Measure the mass with 16 and ~60 declared options, which decides whether an
  extended alphabet or TypeSafe's documented two-stage pattern is the answer
  above 16.
- Measure a Jev-shaped request end to end: one state, N questions, against the
  existing prefix reuse — the number that says what the endpoint is worth.
- Re-measure the 0.9 calibration threshold on a workload that is not this
  fixture.
