# A constrained digit readout points at a button on a 4096px screenshot

- Kind: experiment
- Status: current
- Observed: 2026-09-19
- Last verified: 2026-09-20 (through the served endpoint)
- Scope: serving / constrained decode, multimodal grounding, decision endpoint primitives
- Related: `crates/server/tests/classify_pointing_gpu.rs`, `slot_alphabet.rs`, `crates/server/tests/decide_point_gpu.rs`, `2026-09-19-typed-option-logit-readout.md`, GitHub #178 (multimodal prefill), #235, #242
- Superseded by: none

## Question

A typed readout reads one position, so it produces one symbol — never a pair
of numbers. Jev's three primitives (`noul`, `choice`, `score`) are all one
position. Asking "where is the blue button?" needs something else.

The candidate is a **constrained decode**: one prefill, then K steps, each
restricting the next token to a declared alphabet and forcing the winner back
in. For a point that alphabet is the digits `0`–`9`, three per axis on a 0–999
scale, rescaled onto the image.

Three things had to be established before that shape could be designed:

1. Which **scale** does this model answer in? Qwen2-VL emitted coordinates
   normalized to 0–1000, Qwen3-VL moved to absolute pixels, and the grounding
   vocabulary alone does not say which. Constraining to three digits a model
   that thinks in pixels would look like a pointing failure but be a units
   mismatch.
2. Does the constraint **read** the model or **overrule** it?
3. Is it accurate enough to click with?

## Evidence

`crates/server/tests/classify_pointing_gpu.rs` under the GPU profile.

Fixture: three synthetic 4096×4096 app screenshots (`tests/fixtures/pointing/`,
`generate.py` beside them), each with a blue "Save" button at a known centre
beside distractors of other colours — so "the blue one" is a choice, not the
only rectangle. At the default 32,768-token vision budget a 4096×4096 image is
**not** downscaled: 16,384 image columns, 16,424 prompt tokens, ~1.85 s of
prefill.

Per scene: a free greedy probe asking for a bounding box with no constraint at
all, then the constrained readout with a forced `{"x":` prefix, three digits, a
forced `,"y":`, three more.

| Scene | Button (% of side) | Constrained | Rescaled px | Target px | Error | Inside button |
|---|---|---|---|---|---|---|
| large | 1100×280 (26.9%) | x=767 y=835 | (3145, 3424) | (3150, 3340) | 84 px (2.1%) | yes |
| medium | 560×150 (13.7%) | x=311 y=242 | (1275, 992) | (1280, 975) | 18 px (0.4%) | yes |
| small | 320×90 (7.8%) | x=892 y=165 | (3657, 677) | (3660, 645) | 32 px (0.8%) | yes |

The free probe, unconstrained, in the same declared units:

| Scene | Free probe output | Its box centre | Constrained | Delta (0–999) |
|---|---|---|---|---|
| large | ` ```json [{"bbox_2d": [634, 789, 901, 864], "label": "blue Save button"}] ``` ` | (767, 826) | (767, 835) | (0, 9) |
| medium | `[244,230,380,261]` | (312, 245) | (311, 242) | (1, 3) |
| small | `[853, 150, 931, 173]` | (892, 161) | (892, 165) | (0, 4) |

Per-digit restricted probability, and the mass the ten digits hold against the
whole vocabulary:

| Scene | x digits | y digits |
|---|---|---|
| large | 7 (p=0.993, mass=0.986) 6 (0.981, 1.000) 7 (0.574, 1.000) | 8 (0.980, 1.000) 3 (0.502, 1.000) 5 (0.226, 1.000) |
| medium | 3 (0.964, 0.871) 1 (0.971, 1.000) 1 (0.523, 1.000) | 2 (0.997, 1.000) 4 (0.615, 1.000) 2 (0.149, 1.000) |
| small | 8 (0.970, 0.963) 9 (0.981, 1.000) 2 (0.483, 1.000) | 1 (0.996, 1.000) 6 (0.665, 1.000) 5 (0.149, 1.000) |

The grounding vocabulary is present in the tokenizer (`<|box_start|>`,
`<|box_end|>`, `<|quad_start|>`, `<|object_ref_start|>` are single tokens;
`<|point_start|>` is not) but the model did not use it: it answered in JSON,
twice as a bare array and once fenced in markdown.

## Finding

**The model's native scale is 0–999 normalized**, not absolute pixels. The
free probe establishes this on its own: its system message was
`"You are a helpful assistant."` with **no scale declared anywhere**, and on a
4096-pixel image it answered `[634, 789, 901, 864]`, `[244,230,380,261]` and
`[853, 150, 931, 173]` — every coordinate inside 0–999, every one landing on
the right button once rescaled. Rescaling the constrained reading the same way
(`value / 999 × 4096`) lands within 84 px of the target every time.

**The constraint reads the model rather than overruling it.** Two independent
signs. First, the constrained reading reproduces the centre of the box the
model gives unprompted to within (0,9), (1,3) and (0,4) on a 0–999 scale —
sub-pixel to a few pixels once rescaled. Second, the digit mass is 1.000 at
every position after the first: with nothing restricting it the model was
already going to write a digit there. Only the first digit of each number sits
below 1.000 (0.871–0.986), which is the position where a space or a quote was
still plausible.

**It is accurate enough to click with.** All three centres land inside the
button, on a button as small as 7.8% of the side, with a worst error of 2.1%
of the side.

**The per-digit probability is a resolution readout.** It falls
monotonically across each number: hundreds ~0.97–0.99, tens ~0.50–0.98, units
~0.15–0.57. The model is certain about the coarse position and admits it is
guessing the last digit. That is not noise to be smoothed away — it is the
model reporting that its spatial resolution is about 1 part in 100, and a
caller can read it directly.

## Reproduced through the served endpoint (2026-09-20, GitHub #242)

The numbers above were taken by driving the kernel directly: the digits were
picked host-side out of a full logits row and forced back in. What ships does
none of that — the permitted set is masked into the logits on the device, the
schedule lives on the request, and the answer comes back over HTTP from
`POST /v1/decide` as `{"type": "point"}`. `decide_point_gpu.rs` runs the same
three scenes through that whole stack:

| Scene | Served (px) | Target (px) | Error | Inside button |
|---|---|---|---|---|
| large | (3149, 3424) | (3150, 3340) | 84 px (2.1%) | yes |
| medium | (1279, 988) | (1280, 975) | 13 px (0.3%) | yes |
| small | (3657, 611) | (3660, 645) | 34 px (0.8%) | yes |

Same worst case (84 px on `large`), and every scene still inside its button
— but the two runs are **not** the same reading, and one scene drifts more
than rounding:

| Scene | Direct (normalized) | Served (normalized) | Drift |
|---|---|---|---|
| large | x=767 y=835 | x=768 y=835 | 1 unit of x (4 px) |
| medium | x=311 y=242 | x=312 y=241 | 1 unit each (~4 px) |
| small | x=892 y=165 | x=892 y=149 | **16 units of y (66 px)** |

`small`'s y is a changed **tens digit**, 6 to 4, and the direct run read that
6 at p=0.665 — not a near-tie the two runs could be splitting. The point
moves from 32 px below the button's centre to 34 px above it; the button is
90 px tall, so acceptance 3 holds either way, and it holds by less than the
drift.

Two differences between the runs could produce it and this run does not
separate them:

- **KV format.** The served load is hq-e8-2b (ADR 0022's serving default);
  `classify_pointing_gpu.rs` asks for `KvFormat::Bf16` by name.
- **How the forced `{"x":` is rotated.** Here it is appended to the prompt
  and takes continued MRoPE positions (`Multimodal::append_text`); there it
  was forced through a *text* `prefill_program` call after the multimodal
  one. Whichever is the better rotation, they are not the same one.

The user turn is byte-identical between the two, so the prompt text is not a
candidate. Listed rather than attributed: a cause nobody measured is a guess.

The run also holds on a **drafter load** (`dflash2-7`), pixel for pixel. That
matters: a speculative load runs every round as a verify round and the leaf
refuses a constrained lane there, so `CudaLeaf::decode` routes a batch with
any constrained lane through the plain path. This is the only test that
exercises that claim on the card.

The per-digit trace survives the crossing. The `uncertainty` reported on each
axis is that axis's own place-weighted sum rescaled onto its side in pixels,
checked in the test rather than asserted in prose.

## Implications

- **Jev's three primitives are not enough**, and the missing one is not
  "point". It is **a number read digit by digit** over a declared range:
  `noul` and `choice` are one position with a categorical alphabet, `score` is
  one position with an ordered one, and a number is K positions with the digit
  alphabet. A point is then two numbers, and a box is four — sugar over the
  same primitive, not new machinery.
- **The cost is one prefill plus K steps.** The prefill is the whole expense
  here (~1.85 s for 16K image tokens); six constrained steps after it are
  ordinary decode rounds. That 16K is the *default* vision budget refusing to
  downscale a 4096² image — a pointing endpoint on real screenshots will want
  a lower `--vision-max-tokens` and a cheaper prefill, and the accuracy at
  that budget is unmeasured. It is a trade-off to be chosen, not a detail.
- **Nothing has to parse.** The free probe came back three different ways —
  bare array twice, markdown-fenced JSON once — and a shipped endpoint would
  have to handle all of them plus the ones it has not seen. A constrained
  readout cannot produce a malformed answer, because the answer's shape is the
  caller's.
- **Every digit carries its own confidence**, which composes into a usable
  interval: a units digit at p=0.149 says the reading is good to roughly the
  tens place, i.e. ±20 px on a 4096-pixel image.

## Limits and unknowns

- Three synthetic scenes with flat-colour chrome and one obvious target. Not a
  real screenshot, not a crowded UI, not small text, not overlapping controls.
  The error figures are about this fixture, not about pointing in general.
- One instruction phrasing ("click the blue button") and one target per image.
  Disambiguation was between colours, which is the easy axis; "the second
  Delete button" or "the checkbox next to Sync" is untested.
- Greedy only. No sampling, no temperature.
- The digits are picked by restricted argmax, so the reading is the model's
  most likely **digit string**, not its most likely **number**. This is not a
  theoretical corner: a target at the exact centre of an axis puts the first
  digit on a coin flip between 4 and 5, and the two digits that follow align
  to whichever won — landing somewhere neither 499 nor 500 would have. The
  first digit's probability is what exposes it, which is an argument for
  returning the per-digit trace and not only the number.
- The constrained prompt declared 0–999, which happens to be the model's
  native scale, so this run says nothing about whether it would **obey** a
  different declared range. That decides whether the endpoint can offer an
  arbitrary range or only rescale from the native one.
- BF16 KV; the served default is hq-e8-2b.
- The `<|box_start|>` path was never exercised — the model chose JSON on its
  own. Whether the special-token route is more accurate is untested.

## Follow-ups

- Declare a scale other than 0–999 and see whether it obeys: that separates
  the model's native units from prompt compliance, and decides whether the
  endpoint can offer an arbitrary range.
- Measure against a real screenshot corpus with several plausible targets.
- Compare restricted-argmax digits against the highest-probability number
  (beam over the digit positions), on cases where the first digit is
  uncertain.
- Whether a second constrained pass over a crop refines the last digit, which
  is where the resolution actually ends.
- Rerun `decide_point_gpu.rs` on a BF16 load, which splits the two candidates
  above for `small`'s 16-unit y drift: same rotation, different KV format.
