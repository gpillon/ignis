# What the literature already knew about pointing, and what it changes here

- Kind: research
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision readout primitives, one-pass pointing, `/v1/decide` `point` and `box`
- Related: `docs/specs/decide/11-point-in-one-pass.md`,
  `docs/specs/decide/12-the-coordinate-in-the-latent.md`,
  `2026-09-21-the-point-is-assembled-as-it-is-written.md`,
  `2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`,
  `2026-09-19-constrained-digit-readout-points.md`
- Superseded by: none

## Question

Three experiments were run against this model to find a one-pass `point`
before anyone read what is published about pointing with this family of
models. This is that reading, and what it changes.

## Evidence

### 1. The model has a trained output format for points, and we are not using it

Qwen3-VL's grounding output is JSON with a **declared key name**, on a
**0-1000 relative scale**, and it has a *point* shape as well as a box one:

```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
[{"point_2d": [x, y],          "label": "..."}]
```

The coordinate system moved from Qwen2.5-VL's absolute pixels to 0-1000
relative in Qwen3-VL, and the cookbook ships `decode_json_points()` for the
second shape.

**We measured that scale ourselves and did not recognize the format.** The
free probe in `2026-09-19-constrained-digit-readout-points.md` returned
`[{"bbox_2d": [634, 789, 901, 864], "label": "blue Save button"}]` with no
scale declared anywhere — that is not the model improvising, it is the model
emitting **the shape it was trained on**. `point_system` asks instead for
`{"x":NNN,"y":NNN}`, which is off that distribution.

### 2. Text coordinates are a known-weak interface, and the reasons match our data

Two independent papers make the same argument, and their diagnosis is what
our probe measured from the inside.

`arXiv:2510.03230` calls it an **implicit regression problem**: the model
must map positional embeddings of visual features onto natural-language
number tokens with no explicit supervision on that mapping, which learns
unstably and does not transfer across resolutions.

GUI-Actor (`microsoft.github.io/GUI-Actor`) gives three reasons text
coordinates are a poor fit: weak spatial-semantic alignment, ambiguous
supervision (a single point penalizes equally valid answers inside the same
element), and a **granularity mismatch** between dense screen coordinates
and patch-level visual features.

That last one is our strip result restated. `2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`
found the model wrong by 9-15% of the axis at every grid width, and the grid
never the constraint — a granularity mismatch is exactly what that looks
like from the outside.

### 3. The one-pass mechanism that works is **attention**, not the residual

This is the important one, and it is the signal spec 12 did not try.

**GUI-Actor** adds an `<ACTOR>` token whose **attention over visual patch
tokens** is the answer: one forward pass produces the whole spatial
distribution and several candidate regions at no extra cost. It is trained —
supervision is every patch covered by the ground-truth box — but it has a
lightweight variant that **freezes the backbone** and trains only the action
head and special tokens, 19-103M parameters. ScreenSpot-Pro 40.7% on
Qwen2-VL and 44.6% on Qwen2.5-VL, against UI-TARS-72B at 38.1%.

**TAG** (`arXiv:2412.10840`) is the **tuning-free** cousin: no training at
all. It reads the self-attention from selected *text* tokens to the visual
tokens, keeps the top-K heads (K=10) by attention magnitude, propagates to
image patches, thresholds the relevance map and takes the centre of the
strongest connected region. On MiniCPMV2.5 it reports 84.5% against 48.1%
for direct coordinate generation on OCG, and 88.3% against 40.3% on
ScreenSpot mobile.

### 4. Set-of-mark does not transfer the way it reads

Set-of-mark (`arXiv:2310.11441`) lifts GPT-4V's grounding, and the
open-source result is the opposite: SoM **generally decreases** the
performance of LLaVA-based models, and the suggested reason is that reading
marks is an OCR task. Mark type also has to be chosen against the image's
own content.

### 5. The MRoPE explanation for our y-bias does not apply to this model

`arXiv:2510.03230`'s second diagnosis is a frequency imbalance in MRoPE: one
spatial axis gets only high-frequency components and another only
low-frequency ones, which would be a clean mechanism for the y-bias this
repo has recorded three times. **This model already has the fix.** Qwen3-VL
introduced Interleaved-MRoPE, which distributes t/h/w across the embedding
dimensions so each axis spans the whole spectrum, and this artifact's config
carries `mrope_interleaved: true` with `mrope_section: [11, 11, 10]`.

### 6. The model itself

Qwen3.8-27B, released August 2026: a dense hybrid, 64 layers of which **16
run full attention and 48 run gated-delta linear attention** with a constant
recurrent state, 262K context. The architecture the engine implements, and
the one number below turns on.

## Finding

**The thing we are trying to build has a name, a literature, and a shape we
did not consider.** A one-pass point is *coordinate-free grounding*, and the
mechanism that carries it is the model's **attention from a text token to the
image tokens** — not the residual stream, which is what spec 12 probed and
found holding about one digit.

**Spec 12's objection to a heatmap was pointed the wrong way, and it was
mine.** It said image tokens cannot encode "am I the target" because the
specific instruction is rendered *after* them and no image token can attend
forward to it. True, and irrelevant: TAG and GUI-Actor read attention in the
other direction — **from** the instruction's tokens, which are late in the
sequence, **to** the image tokens, which are early. Causality is satisfied by
construction and there is no ordering problem to design around. That
paragraph blocked the one mechanism the literature says works.

**The blocker on this engine is architectural and specific, and it is not the
one spec 12 named.** Only **16 of the 64 layers have an attention matrix at
all**; the other 48 are GDN and have a recurrent state instead. TAG
aggregates over heads *and* layers, and GUI-Actor trains a head over the
backbone's attention. Whether a quarter of the layers carries enough signal
is unmeasured by anyone — this model is newer than every paper cited here —
and it is the first thing to find out. The second blocker is ordinary: the
attention matrix is never materialized by a fused kernel, so reading one
query row over all keys needs a path that does not exist today.

**The literature's motivation is accuracy and ours is cost, and that changes
which results transfer.** TAG's gains are over base models that score 40-48%
generating coordinates; GUI-Actor's are over baselines at 31-44% on
ScreenSpot-Pro. Our chain is measured at 218/240 inside the button with a
**median error of 0.08% of the axis** — it is not a model that cannot point.
So the accuracy headline does not transfer, and what would transfer is the
*cost*: one pass instead of ten rounds. Any experiment here has to be scored
against our own chain, not against their baselines.

**The cheapest actionable item is the format.** The model was trained to emit
`point_2d` on a 0-1000 scale and volunteered `bbox_2d` when asked nothing;
`point_system` asks for a shape it was not trained on. The chain's failure is
**not** precision — median 0.08% — it is a **catastrophic tail**, 22 of 240
scenes where it points at a different button. An off-distribution prompt is a
plausible contributor to exactly that kind of failure, and it costs one
prompt string to test.

## Implications

- **Spec 11's C2 (set-of-mark) should be demoted, not promoted.** It was the
  last unmeasured candidate after E0/E1/E-P1, and the literature says it is
  the one that fails to transfer to open-weight models. It is still
  untested here, but it is no longer the obvious next thing.
- **A new candidate outranks everything left: attention-as-the-answer.** It
  needs an experiment in the PyTorch vehicle before any kernel work — the
  harness and the 240 scenes exist, and `output_attentions=True` on the GQA
  layers is free there.
- **Run the format A/B first.** It is one string, it uses the existing
  harness, and it targets the chain's actual failure mode.
- **The `<|box_start|>` / `<|object_ref_start|>` route stays untested**, as
  the pointing finding already noted. The trained JSON shape is now the more
  promising of the two.

## Limits and unknowns

- **Every paper cited here is about an older model.** Qwen3.8-27B postdates
  all of them, and none of them is about a hybrid whose attention lives in a
  quarter of its layers. Transfer is an assumption, and the 16/64 split is
  the specific reason it might not hold.
- The Qwen3-VL grounding format was read from documentation and the cookbook,
  not verified against this artifact's own training data. What *is* verified
  here is that the model emits `bbox_2d` at 0-999 unprompted.
- TAG's numbers are MiniCPMV2.5's; GUI-Actor's are Qwen2-VL's and
  Qwen2.5-VL's. Neither reports a hybrid-attention backbone.
- No benchmark used here is ours. ScreenSpot and ScreenSpot-Pro are real
  screenshot corpora; our three fixture scenes and 240 generated ones are
  not, in either direction.

## Sources

- [2D Object Grounding — Qwen3-VL](https://qwenlm-qwen3-vl.mintlify.app/capabilities/grounding-2d)
- [Qwen3-VL cookbook, `2d_grounding.ipynb`](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)
- [Qwen3-VL README (Interleaved-MRoPE)](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md)
- [Improving GUI Grounding with Explicit Position-to-Coordinate Mapping (arXiv:2510.03230)](https://arxiv.org/html/2510.03230v1)
- [Attention-driven GUI Grounding without Fine-Tuning (arXiv:2412.10840)](https://arxiv.org/html/2412.10840v1)
- [GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents](https://microsoft.github.io/GUI-Actor/) ([OpenReview](https://openreview.net/forum?id=5fSkinHw7w))
- [Set-of-Mark Prompting (arXiv:2310.11441)](https://arxiv.org/html/2310.11441v2)
- [Universal Visual Grounding for GUI Agents (arXiv:2410.05243)](https://arxiv.org/html/2410.05243v1) — the SoM-hurts-open-models result
- [Qwen3.8-27B architecture and benchmarks](https://www.mindstudio.ai/blog/qwen3-8-27b-architecture-benchmarks)
