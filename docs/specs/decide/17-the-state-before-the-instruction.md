# 17 - the state before the instruction: sharing one state across question kinds

GitHub: #271

Spec 16 (#270) keeps a fan-out's shared state for its siblings, but only for
questions of the **same kind**. Each kind's system text (`DIRECT_SYSTEM` for
`choice`/`noul`/`score`, `number_system`, `scalar_system`, `point_system`,
`box_system`) is the first thing in the prompt and the state comes after it,
so a `choice` and a `point` over one image share a few template tokens and
nothing else. On a hybrid model there is no exact way to reuse a segment that
is not a prefix (`docs/findings/2026-09-26-prefix-reuse-prior-art-for-decisions.md`
§4: every published composition method for recurrent state is lossy). The
only exact route is **layout**: the state first, the per-kind text after it.

This spec measures that layout and ships it only if the numbers say so.

## Problem Statement

The game agent's fast "eyes" ask a `choice` ("is there a monster?") and a
`point` ("where?") over `[one line][image]` about once a second, and its
second stage mixes `point` and `choice` questions over the same view. Each
question prefills the whole state on its own. So does any Jev-shaped caller
whose questions are not all one kind, which is the common case: Jev's own
API puts the state first and the questions after it.

External evidence leans towards that order being neutral or better for
accuracy, but none of it is this model, these readouts or these prompts:

- context before the question beat the reverse by over 14 points on 21 models
  up to 9B (arXiv 2601.14152); LongReason measured it on GPT-4o
  (arXiv 2501.15089);
- image first, question after is measured better on Qwen3-VL-8B
  (arXiv 2607.15565, 2607.20351), and Alibaba advises it for "multiple
  questions about the same image";
- OpenAI's GPT-4.1 guide says the opposite, with no numbers.

This repository's own measurement moved role and order together (evidence
into the system block, criterion after it: 0.963 against 0.934 balanced
accuracy, `docs/findings/2026-09-20-the-evidence-belongs-in-the-system-block.md`),
and the pointing head (L39.h10) and head set were calibrated on today's
prompts, so none of it can be assumed.

## Solution

Measure one candidate layout, **L1**, against today's, **L0**, and adopt it
per the rule below.

- **L0** (today): system = kind text [+ `{"evidence": …}` for a JSON state];
  user = [state parts] + ask.
- **L1**: nothing kind-specific before the state.
  - JSON state: system = `{"evidence": …}` alone; the kind text moves to the
    user turn, before the ask.
  - Parts state: no system message; user = [state parts] + kind text + ask.
  - The ask itself (`criterion`/`options`, `instruction`, the forced
    `{"x":` of a head `point`) is unchanged and stays last.

Under L1 every question over one state shares the state whatever its kind,
and spec 16's fan-out head reaches through it.

## Implementation Decisions

- **Pre-registered rule.** First run L0 twice on every set below to measure
  run-to-run noise per metric. L1 becomes the default layout **iff**, on
  every set, it is no worse than L0 by more than that noise. Otherwise L0
  stays, and the finding says which sets failed. The owner's standing rule:
  the layout the data favours is the default.
- **The sets**, each through `/v1/decide` on the served artifact, hq-e8-2b:
  - the 144-row typed-option sweep (balanced accuracy, answer mass);
  - the vision readout canaries;
  - spec 15's pointing sets F1-F4 with fresh seeds (head `point` inside,
    head `box` IoU >= 0.5, and the chain `point`/`box`);
  - the `number` and `scalar` truths of #254/#255.
- **Pointing calibration is a separate risk.** The pointing head and head set
  were chosen on L0 prompts. If L1 fails only on the head readouts, the
  finding reports it and L1 may still ship for the other kinds, with `point`
  and head `box` keeping L0 — decided by the same rule, per kind family.
- **Cost measurement**, reported: a `choice` + `point` fan-out over
  `[one line][image]` and over `[2K text][image]`, L0 against L1, with spec 16
  in place (prefill tokens per question and wall time).

## Acceptance

1. L0's noise and L1's numbers on every set are recorded as a finding with a
   README row, whichever way the rule goes.
2. If the rule adopts L1 (for all kinds or a family): it is the default
   rendering, the prompt-pinning tests pin the new bytes, and a `choice` +
   `point` fan-out over one image state prefills the state once (asserted on
   prefill token counts, as spec 16 does).
3. If it does not, nothing in the rendering changes.
4. ADR 0034 (and `CONTEXT.md` if a term moves) record the outcome.
5. `cargo test` passes workspace-wide.

## Out of Scope

- Re-calibrating the pointing head or the head set for L1.
- Any layout other than L1 (a "sandwich" with the instruction on both sides
  would put per-question text back in front of the state).
- Lossy position-independent reuse (HYPIC, LinearKV).

## References

- Spec 16 / #270 (the fan-out head this layout would extend).
- ADR 0034 (measured prompts are not changed without a measurement).
- `docs/findings/2026-09-26-prefix-reuse-prior-art-for-decisions.md` §6.
