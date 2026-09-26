# Decision classes beyond the seven: answer shapes `/v1/decide` lacks, and what each costs

- Kind: research
- Status: current
- Observed: 2026-09-26
- Last verified: 2026-09-26
- Scope: `/v1/decide` primitives, readout / attention readout / constrained decode / fan-out, answer-space design
- Related: `docs/adr/0034-the-leaf-answers-without-generating.md`, `docs/adr/0038-the-seam-carries-one-attention-head.md`, `docs/specs/decide/` (03, 06, 08, 10, 12, 13, 17), [`2026-09-19-typed-option-logit-readout.md`](2026-09-19-typed-option-logit-readout.md), [`2026-09-21-the-point-is-assembled-as-it-is-written.md`](2026-09-21-the-point-is-assembled-as-it-is-written.md), [`2026-09-21-scalar-readout-legend-width.md`](2026-09-21-scalar-readout-legend-width.md), `docs/specs/decide/18-locate-by-attention.md`
- Superseded by: none

## Question

`/v1/decide` serves seven primitives: Jev's three readouts (`noul`, `choice`,
`score`), three constrained decodes (`number`, `scalar`, and `point`/`box` by
the chain), and the attention readout that answers `point`/`box` in one pass.
Which **answer shapes** does a System-1 endpoint still lack, which of them
can the engine's existing mechanisms serve, and what does the literature
measure about each?

## Evidence

### What makes a shape a System-1 class

Three properties hold for every primitive served today, and a candidate that
breaks one of them belongs to `/v1/chat/completions` rather than here:

1. **The answer space is closed and declared by the caller** (or, for the
   spatial types, is the submitted image).
2. **The answer is a distribution** (or a trace of per-step probabilities),
   not a single sample.
3. **It costs O(1) passes**: zero decode rounds (readout, attention
   readout), a few rounds (constrained decode), or N small suffix prefills
   over one shared state (fan-out).

A corollary shapes the whole list: **System 1 never writes, it points.**
Extracting a value from the evidence is pointing at where it is written;
open text that is not in the evidence is System 2.

### The engine's mechanisms

| Mechanism | What it reads | Cost | Serves today |
|---|---|---|---|
| R — readout | logits of answer tokens at one position | 1 prefill, 0 rounds | `noul`, `choice`, `score` |
| A — attention readout | one head's scores (+ a head set's argmax) over a key span | 1 prefill, 0 rounds | head `point`, head `box` |
| C — constrained decode | a schedule of ≤ 32-token permitted sets | 1 prefill + k rounds | `number`, `scalar`, chain `point`/`box` |
| F — fan-out | N questions over one shared state prefix | N suffix prefills | every multi-question request |

`AttentionQuery` (`crates/core/src/pointing.rs`) already names an arbitrary
key range (`key_begin`, `key_count`), and the C ABI's attention fields sit on
the prefill options struct shared by the text path
(`kernel/include/ignis_step.h`). Two host-side choices limit A to images
today:
- the runtime refuses a readout job with no multimodal span
  (`ignis.runtime.attention_without_image`);
- the score room is sized to one vision item (`kernel/src/step.cu`).

### The answer spaces, and which are missing

| Answer space | Class | Status |
|---|---|---|
| two points | `noul` | served (R) |
| unordered finite set | `choice` | served (R) |
| ordered finite set | `score` | served (R) |
| integers / reals | `number`, `scalar` | served (C) |
| the image plane | `point`, `box` | served (A, C) |
| **subsets of a set** | `multi` | missing |
| **orderings of a set** | `rank` | missing |
| **positions in the text evidence** | `locate` / `cite` | missing |
| **several points in the plane** | `points` / `count` | missing |
| **a product of the above** | `record` / cascade | missing |

### External evidence per candidate

Sources were fetched by a research pass on 2026-09-26. Numbers are the
sources' own, on their models, not this one.

**TypeSafe Jev** (`docs.typesafe.ai/api`, `docs.typesafe.ai/llms.txt`)
documents exactly three types:
- `noul`;
- `choice`, at most 255 options;
- `score`, 2 to 10 levels.

Everything else is composed in their cookbooks:
- rerank is one question per query-candidate pair;
- "line search" is one `choice` over 218 line ids;
- value extraction is a regex that finds candidates, then a `choice`;
- hierarchical classification is a beam search over `choice` probabilities.

**`multi` (select all that apply).** One readout cannot express it: the
softmax is exclusive, and a model suppresses all labels but one.
- Ma et al., arXiv 2505.17510 (EMNLP 2025), Llama-3-70B:
  - "compare-to-none" at one position: NLL 23.93, F1 0.27 on GoEmotions;
  - one yes/no pass per label: NLL 3.60, F1 0.43.
- SATA-Bench (arXiv 2506.00643), Qwen2.5-14B exact match:
  - first token 6.30;
  - per-option yes/no 25.64;
  - Choice Funnel (take the top option, remove it, repeat, stop on "none" or a threshold) 27.82, with 6,005 passes against yes/no's 15,517.
- PriDe debiasing *lowered* the single-readout variant (6.30 → 4.61).
- ⇒ **F over `noul`**, one per option. The option list can sit in the shared prefix, so each sibling's own suffix is a few tokens. No engine change.

**`rank`.** Ordering `choice` probabilities is reliable only for the top one.
- FIRST (arXiv 2406.15657, EMNLP 2024): sorting first-token ID logits needs fine-tuning. In a plain LLM, the logit ranking agrees much less with the generated ranking.
- ICR (arXiv 2410.02642, ICLR 2025): ranks by attention from query tokens to each document, with a content-free "N/A" query subtracted.
  - two forward passes;
  - no head calibration;
  - BEIR nDCG@10: 57.3 against RankGPT's 44.0 (Mistral-7B), 60.4 against 52.0 (Llama-3.1-8B);
  - weak on entity matching, and biased toward lexical overlap.
- QRHead (arXiv 2506.09944, EMNLP 2025): 16 heads (about 1-2%) picked on fewer than 100 labelled examples, BEIR 49.7 against ICR's 48.5 (Llama-3.1-8B). Heads found at 32K transfer to 128K.
- PRP pairwise (arXiv 2306.17563): all pairs is N(N−1) calls.
- ⇒ top-k by funnel (k readouts) today; at scale, the same A-over-text mechanism as `locate`.

**`locate` / `cite` (a position in the text evidence).** The textual twin of
`point`.
- Label route: Jev's "line search", a `choice` over labelled segments.
  - R, zero engine work.
  - Capped at the measured 256 options.
  - The labels change the state's bytes, so no reuse with the other kinds (spec 17).
- Attention route: ICR and QRHead above.
  - AT2 (arXiv 2504.13752): citation from a learned coefficient per head over attention-to-source, under one pass, transfers across datasets.
  - ContextCite (arXiv 2409.00729, NeurIPS 2024): ablation surrogate, 32-256 passes. It beats raw attention, but raw attention was fine on Llama-3-8B.
- ⇒ the one candidate that exercises ignis's own mechanism. Spec 18.

**`points` / `count` (every instance in the image).**
- Molmo (arXiv 2409.17146) counts by pointing (point-then-count 89.4 against 87.9 count-only). It generates one point at a time and was trained for it. MolmoPoint (arXiv 2603.28069) likewise.
- Localization heads (arXiv 2503.06287, CVPR 2025): training-free, k=3 heads, but one referent per map.
- **No primary source was found that localizes several instances from attention alone.** On this engine it is a host-side experiment on maps the pointing head already returns.

**`compare` / order debiasing (a modifier, not a class).**
- MT-Bench (arXiv 2306.05685): GPT-4 agrees with itself under swapped order on 65.0% of pairs, Claude-v1 on 23.8%. The standard fix, swap and average, doubles the cost.
- PriDe (arXiv 2309.03882, ICLR 2024): multiple-choice bias is mostly a prior on the **ID tokens**, not on positions. It is estimated on 5% of samples, for about 1.15× the cost.
- Contextual calibration (arXiv 2102.09690): a content-free input plus an affine correction.
- ⇒ measure the flip rate under reversed option order on the SemIf 144 fixture before building anything.

**`record` / cascade (a product space).** Fields where one conditions the next
(tool → argument).
- As a constrained decode it costs one round per field, plus one per forced literal.
- SGLang's compressed FSM (lmsys.org blog, 2024-02-05) appends forced literal runs in one extend: up to 2× lower latency.
- XGrammar (arXiv 2411.15100) computes full-vocabulary masks in under 40 µs per token, overlapped with the forward pass.
- Client-side, two calls already do this.

**Rejected: likelihood scoring of the answer text.**
- OLMES (arXiv 2406.08446): letter labels 93.7 against scoring the answer text 69.0 (Llama3-70B, ARC-C).
- SGLang's default token-length normalization favours long options (its issue #523).
- Labels are what `choice` already does.

**Numbers in one pass: no zero-shot method.**
- Linear value probes recover magnitude, not exact values (arXiv 2401.03735: under 50% within 1% error).
- Per-digit circular probes reach about 0.91 (arXiv 2410.11781), but only on numbers already in the input.
- Regression readouts (RAFT, arXiv 2403.04182, 2411.14708) are trained.
- This agrees with this repo's own negative (spec 12, the coordinate in the latent).
- A cheaper `number`/`scalar` is fewer rounds (drafter verify, jump-forward), not zero.

## Finding

The missing classes are **`multi`, `rank`, `locate`/`cite`, `points`/`count`
and `record`**. Order debiasing is a modifier across classes, not a class.

**Composition, no engine change:**
- `multi` = F over `noul`;
- top-k `rank` = a funnel of `choice` readouts.

**A generalization of an existing mechanism:**
- `locate`/`cite` = A over a text span;
- at scale, `rank` too;
- the engine gap is host-side (the runtime's image-only refusal and the score room), not the kernel.

**A new engine path:**
- `record` wants jump-forward, or a schedule that depends on the draws.

**Open research with no prior art to lean on:**
- `points`/`count` from attention maps.

## Implications

Suggested order:
1. `multi` (cheapest, pure composition, needs a multi-label fixture);
2. `locate` by attention (spec 18, the class that is most specific to this engine; `cite` and scalable `rank` follow from it);
3. `points`/`count` as an experiment on existing head maps;
4. the order-flip measurement, with debiasing only if the number asks for it;
5. `record` together with the `number`/`scalar` round reduction (both want forced-literal jump-forward).

## Limits and unknowns

- Every external number is on another model, often another size or family, and often fine-tuned. None of them is a claim about the served 27B.
- The research pass could not verify ICLR 2025 as the venue of Retrieval Heads (arXiv 2404.15574).
- The research pass did not re-fetch MTP heads (arXiv 2404.19737).
- ICR's lexical bias is the risk most specific to `locate`: a head that matches strings will look accurate on questions that share words with their answer.

## Follow-ups

- Spec 18 (`locate` by attention), written with this finding.
- A multi-label fixture for `multi`.
- The option-order flip rate on `crates/server/tests/fixtures/semif/`.
