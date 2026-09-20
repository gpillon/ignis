# A fixed-width `number` is wrong at exactly one width, and an image `state` shares nothing

- Kind: experiment
- Status: current
- Observed: 2026-09-20
- Last verified: 2026-09-20
- Scope: serving / `POST /v1/decide`, constrained decode, fan-out cost
- Related: `2026-09-19-constrained-digit-readout-points.md`,
  `2026-09-19-typed-option-logit-readout.md`, `crates/server/src/numbers.rs`,
  GitHub #240 (fan-out), #242 (number/point/box)
- Superseded by: none

## Question

#242 shipped `number`, `point` and `box`, and said in the code that only
`point` at three digits was measured. Two things were therefore unknown on
the served endpoint: whether a bare `number` at an arbitrary declared width
answers correctly, and what a fan-out actually costs when the evidence is an
image rather than text.

## Evidence

A live `ignis-server` on the 5090: release build, `--vision`, `--metrics`,
hq-e8-2b KV, `--max-context 40960`, `--spec dflash2 --draft-tokens 7`, the
27B `qwen3_8_27b_nvfp4full-v2` artifact. Load to ready: 12.7 s. Every
request below is one `POST /v1/decide` over the real HTTP surface.

### 1. `number` against a known value, sweeping the declared width

Three states with an unambiguous integer in them, each asked at `digits`
1..6. `=` marks the correct answer; `p(d0)` is the reported probability of
the **first** digit.

| truth | w1 | w2 | w3 | w4 | w5 | w6 |
|---|---|---|---|---|---|---|
| 7 | **7** `=` | 70 | 700 | 7000 | **7** `=` | **7** `=` |
| 42 | 4 | **42** `=` | 420 | 4200 | **42** `=` | **42** `=` |
| 230 | 2 | 23 | **230** `=` | 2300 | **230** `=` | **230** `=` |

| p(d0) | w1 | w2 | w3 | w4 | w5 | w6 |
|---|---|---|---|---|---|---|
| 7 | 1.00 | 0.94 | 0.99 | 0.71 | 0.95 | 0.97 |
| 42 | 0.98 | 1.00 | 0.99 | **0.67** | 0.81 | 0.77 |
| 230 | 0.76 | 1.00 | 1.00 | **0.65** | 0.88 | 0.67 |

### 2. A `number` the model has to compute

`"3 boxes of A4 paper at 12.50 each, 2 toner cartridges at 89.00 each,
shipping 15.00"` — true total 230.50.

| width | answer | sigma | trace |
|---|---|---|---|
| 3 | 172 | 46.7 | 1(0.63) 7(0.15) 2(0.15) |
| 4 | 2300 | 446.9 | 2(0.65) 3(0.16) 0(0.20) 0(0.88) |

### 3. `point` and `box` over the committed pointing fixture, through the server

Three questions per scene in one request (`point`, `box`, one `noul`).

| Scene | point (px) | inside button | box (px) | target box |
|---|---|---|---|---|
| large | (3149, 3424) | yes | [2595, 3239, 3694, 3559] | [2600, 3200, 3700, 3480] |
| medium | (1279, 988) | yes | [1000, 931, 1562, 1078] | [1000, 900, 1560, 1050] |
| small | (3657, 611) | yes | [3506, 615, 3821, 705] | [3500, 600, 3820, 690] |

The `noul` in the same request ("is the button text legible?") reads 0.995 on
`large` and `medium` and **0.679** on `small`, whose button is 320x90 on a
4096-pixel image.

Asked the same screenshot in free-form chat, the model describes it
correctly and places the button as *"the top-left of the button group"* —
true, and not a coordinate.

### 4. What a fan-out costs

Same questions, same server, varying only how many and what the evidence is.

| evidence | 1q | 2q | 4q | 8q | 20q |
|---|---|---|---|---|---|
| text (~700-token state) | 0.13 s | 0.13 s | 0.23 s | 0.36 s | 1.14 s |
| image (4096², 16.5K tokens) | 6.84 s | 13.50 s | 27.92 s | — | — |

Input tokens, image: 16,506 / 32,968 / 65,977 — exactly 1x, 2x, 4x.

For comparison on the same load: a 33-token chat prompt generating 88 tokens
takes 0.672 s (~131 tok/s with dflash2-7), and the same 4096² image in a
free-form chat completion is 16,413 prompt tokens in 7.14 s.

### 5. The metric contract, live

After the session above:

```
ignis_decisions_total{type="noul"}   17      ignis_decision_answer_mass_count 27
ignis_decisions_total{type="choice"}  6      ignis_decision_answer_mass_sum   26.799
ignis_decisions_total{type="score"}   4      ignis_decoded_tokens_total      178
ignis_decisions_total{type="number"}  9      ignis_requests_completed_total   46
ignis_decisions_total{type="point"}   6
ignis_decisions_total{type="box"}     4
```

## Finding

**A fixed-width `number` is wrong at exactly one width: one more than the
value's own.** The model writes the number left-aligned and fills the
remaining step with a zero, multiplying the answer by ten — 7 becomes 70, 42
becomes 420, 230 becomes 2300, on every value tried. At two or more digits
past the natural width it pads *left* with zeros and the answer is correct
again; below the natural width it truncates from the right, which is the only
behaviour a narrower field could have.

**The first digit's probability exposes it.** At the broken width it falls to
0.65–0.71 — the model is genuinely undecided between a leading zero and the
leading digit — against 0.94–1.00 when the width fits. That is the same
signal the pointing finding described, working as an error detector rather
than as a resolution readout.

**So `digits` is not a formatting knob, it is part of the question.** A
caller who knows the magnitude should set `digits` to it exactly; a caller
who does not should set it two or more above the largest plausible value, and
never exactly one above. This is a property of the model's own writing
habits, not of the constraint: the schedule forces K digits and the model
decides where to put them.

**Arithmetic is not the primitive's strength, and it says so.** On a total
the model has to compute, the three-digit answer (172, true 230.50) comes
back with a first digit at p=0.63 and a sigma of 46.7 — a wrong answer,
loudly flagged. That is the intended behaviour of the trace and the argument
for returning it.

**`box` works, at least on this fixture.** #242 shipped it as an explicitly
unmeasured extension of `point`'s prompt. Over the three committed scenes its
edges land within 6 px on x and 30–80 px on y of the ground truth — the same
y-bias `point` shows. Still three synthetic scenes; still not a claim about
bounding boxes in general.

**An image `state` shares nothing across a fan-out, and the cost is exactly
linear.** 6.9 s and 16.5 K tokens per question, every question. A text state
rides the system block and is prefilled once: eight questions cost 0.36 s
against 0.13 s for one, a marginal 33 ms each. This is GitHub #240's unmet
acceptance 2 measured rather than reasoned about — the ratio between the two
rows is 200x per question.

**The readout-only histogram reads correctly in the wild.** 46 decisions
counted, 27 masses observed, difference 19 = the `number` + `point` + `box`
questions served. The mean observed mass is 0.993, in line with the
readout baseline.

## Implications

- `numbers.rs`'s doc for `number` should carry the width rule, not only the
  "unmeasured" caveat. A caller reading only the API would choose `digits: 4`
  for a value around 500 and be wrong by a factor of ten, with nothing in
  the answer's shape to show it.
- A `min`/`max` range — deliberately not offered in spec 06 — would fix this
  properly by making the width a consequence of the declared range rather
  than a caller's guess. The follow-up in the pointing finding (does the
  model obey a declared range other than 0-999?) is the prerequisite.
- An image fan-out is worth serialising differently or not at all: at 6.9 s
  per question a caller is better served asking one question with a richer
  `criterion` than four cheap ones. Whatever #235 concludes about shared
  prefixes applies here with a 200x lever.

## Limits and unknowns

- Three values in the width sweep, one model, greedy (`/v1/decide` uses
  `DecodeParams::default()`). The pad-right-at-natural+1 rule is consistent
  across all three but is three points, not a law.
- The values were small (7, 42, 230). Whether the same rule holds at five and
  six natural digits is untested.
- `box` on three synthetic flat-colour scenes with one obvious target, as in
  the pointing finding. The y-bias is reproduced but not explained.
- Timings are one run each, on an otherwise idle card; they are ratios worth
  trusting and absolute numbers worth re-measuring.
