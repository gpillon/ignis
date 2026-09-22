# 13 - point in one pass, read from an attention head

The study (`11-point-in-one-pass.md`) and its findings
(`docs/findings/2026-09-21-one-attention-head-points.md`,
`2026-09-21-the-head-points-in-the-engine.md`,
`2026-09-22-the-codec-costs-the-head-its-read.md`) found that the owner's
one-pass `point` exists: one attention head of the served model, **L39.h10**
(GQA ordinal 9, query head 10), read at the position after the forced
`{"x":`, lands inside the target more often than the ten-round digit chain.
This spec turns it into the default way `/v1/decide` answers `point`.

## Problem Statement

A caller asking `/v1/decide` for a `point` today gets it from a **constrained
decode**: one prefill, then ten decode rounds that write `x` and `y` digit by
digit. That costs ten rounds of card time per point, and it is not the best
answer the model has:

- On synthetic scenes at 1024 px the chain lands inside the target on 211-212
  of 240; its failures are all **wrong element** — a confident point on the
  wrong button.
- At 4096 px, the endpoint's default size, it lands inside on only 169-175 of
  240: the model's chain drifts ~11/999 **down** at its largest grid (the
  PyTorch reference drifts the same, scene for scene), so a third of the
  answers fall just below the button.

Meanwhile the prefill the chain already runs holds a better answer in one
place: the attention of one head, from the answer's own scaffold to the
image. The caller pays for ten rounds to get a worse point.

## Solution

`point` answers **in one pass** by default: the prefill that renders the
question and forces `{"x":` also reads, at its last position, the attention
of the artifact's calibrated **pointing head** over the image, and the server
turns that map into a point. No decode round runs. Measured in the engine
(served render, hq-e8-2b with the residual window, the keys attention
actually reads): **227 of 240 inside at 1024 px, 236 at 4096 px**, against the
chain's 212 and 169.

The head's point is **coarse** — its resolution is one image token (32 px of
the image), and a hit lands a median 28-39/999 from the target's centre where
the chain's lands 2-10. So the chain stays, **opt-in**, for what it is still
better at: `"method": "chain"` gives the digit-precise point, targets smaller
than a token, and the per-digit trace. A load whose artifact has no calibrated
head answers `point` with the chain, and every answer says which method
produced it.

## User Stories

1. As an agent calling `/v1/decide`, I want a `point` to cost one prefill and
   no decode round, so that locating an element is as cheap as a yes/no
   question.
2. As an agent clicking UI elements, I want the point to land inside the
   element I named more often than the digit chain does, so that fewer of my
   clicks miss.
3. As an agent working on 4096 px screenshots, I want a point that does not
   drift below the target at the model's largest grid, so that I do not have
   to downscale my screenshots to get a usable answer.
4. As a caller who sends no `method`, I want the best measured method to be
   used, so that I get the better answer without knowing the study.
5. As a caller who needs sub-token precision (a checkbox, a small icon, a
   text caret), I want to ask for `"method": "chain"`, so that I can still get
   the digit-precise point.
6. As a caller, I want every point answer to state the `method` that produced
   it, so that I can tell a coarse head point from a digit-precise one
   without inferring it.
7. As a caller, I want the head's answer in the same **pixels of the
   submitted image** as the chain's, with the model-scale reading beside it,
   so that switching method changes no code on my side.
8. As a caller, I want the head's `uncertainty` in pixels per axis, so that I
   know how coarse the point is and can decide whether to re-ask with the
   chain.
9. As a caller, I want to see how concentrated the head's attention was (the
   region's size and share), so that I can treat a diffuse map as a weaker
   answer.
10. As a caller who asks for `"method": "head"` on a load that has no
    calibrated head, I want the request refused before any GPU time is spent,
    so that I never get a chain answer I explicitly did not ask for.
11. As a caller who omits `method` on a load without a calibrated head, I
    want the chain's answer (labelled `chain`), so that `point` keeps working
    on any artifact.
12. As a caller sending an unknown `method`, I want a validation error naming
    the accepted values, so that a typo is not silently treated as the
    default.
13. As a caller asking for a `box`, I want a `method` field refused with a
    clear error, so that I do not believe a box came out of the head when it
    cannot (the map's region is not a box: IoU ~0.3).
14. As a caller asking several questions over one image (fan-out), I want
    head points to share the image's prefix like every other decision, so
    that ten questions about one screenshot do not re-prefill it ten times.
15. As a caller whose `state` carries no image, I want the same
    `state_carries_no_image` failure the chain gives, so that the error does
    not depend on the method.
16. As an operator, I want a head point to hold no decode lane and no
    residency, like a readout, so that pointing does not compete with
    chat traffic for decode slots.
17. As an operator, I want the answer the same under BF16 and hq-e8-2b KV to
    within the measured difference, so that the KV format stays a memory
    choice and not an accuracy one.
18. As an operator, I want the server to log at load whether the artifact has
    a calibrated pointing head, so that I know which method `point` will use
    before the first request.
19. As a maintainer, I want the pointing head to be a calibrated constant
    **keyed to the artifact's content hash**, so that loading another artifact
    can never read a head chosen for a different model.
20. As a maintainer, I want a GPU test that fails when the served artifact
    changes and its head has not been recalibrated, so that a new artifact
    cannot ship with a stale head silently.
21. As a maintainer recalibrating for a new artifact, I want the procedure
    (labelled scenes, cross-validated head choice, the region rule) written
    down with the harness that ran it, so that recalibration is a repeat and
    not a new study.
22. As a maintainer, I want the leaf's attention readout checked against the
    test-only attention tap on the same prompt, so that the production path
    is held to an independent oracle and not to itself.
23. As a maintainer, I want the region rule to be a pure host function, so
    that its edge cases (flat maps, ties, several regions, non-square grids)
    are tested on CPU in milliseconds.
24. As a maintainer, I want the mock backend to produce a deterministic
    attention map with a known peak, so that the whole `/v1/decide` head path
    is covered by CPU tests (ADR 0006).
25. As a maintainer, I want only the image span's scores to cross the
    `Compute` seam — never a full attention row, never a full logits row — so
    that the seam stays as narrow as ADR 0034 made it.
26. As a maintainer, I want a request that asks for no attention readout to
    pay nothing for the feature (no allocation, no device work), so that chat
    traffic is untouched.
27. As a maintainer, I want the readout to fail loudly — a failed question,
    not a wrong point — when it cannot read the keys it needs (an image
    outside what the layer's attention materialized), so that the edge is a
    visible error.
28. As a reviewer of the OpenAPI document, I want `method` and the head
    answer's fields described at `/v1`, so that the contract is readable
    without the source (ADR 0036).
29. As the owner, I want the acceptance measured once, on scenes never used
    to choose the head or the rule, with floors written before the run, so
    that the shipped default is not tuned to its own test.
30. As the owner, I want the latency of a head point and a chain point on the
    same prompt reported, so that the saving is a number and not a claim.

## Implementation Decisions

- **Default and opt-in.** The `point` question gains an optional `method`:
  `"head"` or `"chain"`. Omitted, it is `head` when the loaded artifact has a
  calibrated pointing head and `chain` otherwise. Explicit `head` on an
  uncalibrated load is refused at validation (422, before any prefill), with
  its own error code. `box` refuses `method`. Every point answer carries
  `method`. The chain path, when chosen, is unchanged.

- **The prompt is the chain's, byte for byte.** Same system text
  (`point_system`), same served render, same forced opening literal `{"x":`
  as prompt. The head's query must sit inside the answer's scaffold — at the
  instruction's own tokens the head does not transfer (136-143 of 240) — and
  a shared render keeps head and chain questions over one image sharing
  their prefix.

- **The attention readout is a third thing the `Compute` seam carries**,
  beside the answer-token readout (#237) and the permitted set (#242), and
  shaped like the first: the prefill job of a head point's **last chunk**
  names the GQA layer, the query head and the image's placeholder span; the
  prefill outcome returns **one score per key of that span**,
  `q · k / sqrt(head_dim)` for the query at the chunk's last position, before
  any softmax. At 1024 px that is 1,024 floats, at 4096 px 16,384 (64 KB).
  No full attention row and no logits row crosses. A new ADR records this
  as an extension of ADR 0034's rule, the way ADR 0034 recorded the readout.

- **The keys are the ones attention read, as attention read them.** Under
  hq-e8-2b that is the prompt route's own materialized key planes for that
  layer — codec decodes, with the residual window's sinks, ring and chunk
  exact (#257, #258); under BF16, the cache's pages. The accuracy above was
  measured on exactly those keys. Reading them where attention reads them is
  also what keeps **prefix reuse** working: in a fan-out the image's keys
  were written by an earlier request, and only the cache still has them.
  The query is the layer's own, after its norm and rotary embedding, in the
  same frame as the keys it is dotted with.

- **A head point is a readout-class decision**: one prefill, no decode round,
  no decode lane, no residency — the scheduler finishes it on its last
  prefill chunk as it finishes a `noul` or a `choice`.

- **The region rule runs on the host**, a pure function of the scores and
  the image's token grid (h × w merged tokens): min-max normalize, keep
  cells at or above 0.5, take the **4-connected region with the highest mean
  score**, return its score-weighted centre. That is the rule every number in
  the findings was measured with (TAG's), and it is not re-tuned here. The
  centre maps to the submitted image's pixels through the processor's own
  grid-to-pixel scale on each axis, and to the question's `digits` scale
  (0-999 at 3) for `normalized`.

- **The answer's evidence for a head point**: `pixels`, `normalized`,
  `uncertainty` in pixels per axis (the region's score-weighted spread, not
  a probability), and `region` — its size in cells and its share of the
  softmax mass over the image span. No `digits`. The chain answer is
  unchanged apart from `method`.

- **The pointing head is a calibrated constant keyed to the artifact.** A
  compiled-in table maps an artifact **content hash** to (GQA ordinal, query
  head); today one entry: the served NVFP4 27B artifact → ordinal 9, head 10.
  Unlike the answer alphabet, it cannot be computed at load — it was chosen
  with labelled scenes and cross-validation — so it is a constant, and like
  the answer alphabet it is refused against any other load. The load logs
  whether `point` will answer by head or by chain.

- **Failure is a failed question, never a wrong point.** If the leaf cannot
  read the span (for instance an image beyond the 262,144 keys the hq prompt
  route materializes in one band), the question fails at runtime with its
  own error, as a runtime failure does today.

- **Observability**: a head point is counted as a `point` decision like the
  chain's. A `method` label on the decision metrics is a contract change
  (ADR 0017) and is left out of this spec.

- **Documentation**: `CONTEXT.md` gains *Attention readout* and *Pointing
  head*; the OpenAPI document gains `method` and the head answer's fields.

## Testing Decisions

A good test here asserts what a caller or an operator can observe — the
answer, its method, its refusals, the rounds it did not run — and holds the
device path to an **independent** oracle rather than to itself.

- **Region rule (CPU, pure function).** Table-driven: a single peak; a flat
  map; two separated regions where the larger-mean one must win over the
  larger one; ties; a region touching the grid edge; a non-square grid and a
  non-square image, where each axis scales by its own side; the 0-999
  mapping at several `digits`. Prior art: the pure plan and reading tests
  beside the number layouts.

- **The decide endpoint over the mock (CPU).** The mock backend returns a
  deterministic attention map with a known peak for a job that asks for one
  (as it returns a deterministic readout today). Tests at `/v1/decide`:
  default `method` gives a head point at the peak, in pixels of the submitted
  image, with `method: "head"`; zero decode rounds ran; `"method": "chain"`
  gives today's answer plus `method: "chain"`; an uncalibrated artifact makes
  the default fall back to `chain` and an explicit `head` a 422 before any
  prefill; `box` with `method` is refused; an unknown `method` is refused; no
  image gives `state_carries_no_image`; a fan-out mixing head points and
  readouts over one image answers all of them. Prior art: the existing
  decide endpoint and fan-out tests over the mock.

- **The scheduler (CPU).** A head point finishes on its last prefill chunk
  and never takes a decode lane. Prior art: the readout request tests (#238).

- **Leaf equivalence (GPU profile, `attn-tap` feature).** On the committed
  4096 px pointing fixture and a handful of 1024 px scenes, under BF16 and
  under hq-e8-2b: the leaf's scores equal the tap's host-side `q · k / 16`
  for the same layer, head and position to within float accumulation, and
  the region rule gives the same point from both. The tap is test-only and
  independent of the new path, which is what makes it the oracle. Prior art:
  the attention-head harness and the consumed-key self-check test.

- **Artifact key (GPU profile).** The served artifact's content hash is in
  the calibration table; the test fails when the artifact changes, naming
  the recalibration procedure.

- **Acceptance, pre-registered here, measured once.** Two new sets from the
  scene generator never used to choose the head or the rule: 240 varied
  scenes at 1024 px (seed 20260925) and 240 at 4096 px (seed 20260926),
  asked **through `/v1/decide` with no `method`**, served artifact, hq-e8-2b
  with the residual window. Inside the target on **at least 221 of 240 at
  1024 px and at least 230 of 240 at 4096 px** — the engine's measured 227
  and 236 on set C and C4096 minus the same slack of six the earlier
  criteria used. Reported beside it, not asserted: the chain on the same
  scenes, both methods' median distance from the target's centre, and the
  wall time of a head point and a chain point on the same prompt.

- **Prefill cost.** A request that asks for no attention readout allocates
  nothing and launches nothing new; measured as the decode and prefill
  benchmarks already measure a change to the prefill path.

## Out of Scope

- **`box` in two passes** with the head as pass 1: the next spec, built on
  this readout.
- **The guard** (keep the chain unless it is far from the head): at 4096 px
  it keeps a drifted chain (201-206 against the head's 236), so a second
  guard needs its own design on sets A, B, C and C4096 and its own measure.
- **An exact copy of the head's KV head.** Worth 5 scenes of 240 at 1024 px
  and 2 at 4096 px over the consumed keys, and it breaks under prefix reuse
  (the copy exists only for keys this request wrote).
- **Recalibration tooling** beyond writing the procedure down: the harness
  that chose L39.h10 exists and is the tool.
- **The Playground's Decide tab** showing `method` and the region.
- **A `method` label on the decision metrics** (ADR 0017).
- **Reading the digit logits at the same position** as extra evidence (the
  chain's first-digit probability flags its wrong-element failures): free
  through the existing readout, and a separate question.
- **Multi-image states**: `point` answers about the submitted image exactly
  as the chain does today.

## Further Notes

- **Where the numbers come from.** Head 227 / 236 of 240 is L39.h10 on the
  keys hq attention consumes with the window wired (set C, set C4096);
  under BF16 it is 232 / 238. The chain is 212 / 169 on the same scenes.
  Everything is synthetic scenes from one generator, and at 4096 px that
  generator draws the 1024 px scene four times larger — so a 4K screen with
  1x UI elements is unmeasured for both methods, and the head's one-token
  resolution will matter more there.
- **Why the head, and why this position**: the literature's name for it is
  coordinate-free grounding, and the one training-free method (TAG) reads
  attention from text tokens to image tokens; here the query must be the
  answer's scaffold, after `{"x":`, not the instruction.
- **Recalibration** for a new artifact: run the attention-head harness over
  the labelled scene sets with every GQA layer armed, choose the head by
  cross-validation on the region rule's inside rate, add the artifact's
  content hash to the table, and re-run this spec's acceptance.
