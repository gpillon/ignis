# The codec costs the head its read; at 4096 px the model's chain drifts down

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: serving / decision primitives, one-pass pointing, `/v1/decide` `point`; hq-e8-2b KV; kernel test seams
- Related: `crates/server/tests/attention_head_point_gpu.rs`,
  `crates/server/tests/attn_tap_hq_consumed_gpu.rs`,
  `kernel/include/ignis_attn_tap.h`, `crates/core/src/attn_tap.rs`,
  `.scratch/latent-probe/results/engine/` and `engine-runs4.log`,
  `engine-runs5.log`, `vehicle4096.py` and `results/vehicle4096.json`,
  `results/engine-window/` (raw, on disk),
  `2026-09-21-the-head-points-in-the-engine.md` (the arms this completes),
  `2026-09-12-hq-attention-route-agreement.md`, ADR 0030 (VRAM plan)
- Superseded by: none

## Question

`2026-09-21-the-head-points-in-the-engine.md` measured L39.h10 in the engine
on every arm but production's: its hq rows scored the head against the keys
*given to* the codec, not the keys attention reads back. Production is the
served render under hq-e8-2b at up to 4096 px. Does the head still point
there, and does the guard still hold?

## Evidence

**What hq attention reads, in this engine.** A second capture on the tap
(`with_attn_tap_hq`) copies the rotated-frame scratch planes the hq prompt
route actually attends over, after the op returns. The vendored route can
fill them from three sources — current chunk, 32 sinks and a 512-key ring
exact, the rest codec-decoded — but the exact sources need residual side
planes on the cache view, and ignis never supplies them. Measured
(`attn_tap_hq_consumed_gpu.rs`, 4096 px fixture): **every** row sits at the
codec's own error — min 0.333, median 0.370, max 0.784 relative L2 against
the rotated pre-codec key, zero rows under 0.1 — including the 666 the
three-source rule would have kept exact. Under hq-e8-2b every key this
engine's attention reads is a codec decode, at every image size.

**Pre-registered for set C** (2026-09-22, before any run on it, agreed
between the two sessions on this branch): 240 new varied scenes at 1024 px
(`scenes.py --varied --seed 20260923`), served render, consumed hq keys —
the production configuration at this size. L39.h10 inside on at least 224,
the guard at d = 60/999 (unchanged) on at least 233. "Not broken" floors,
not "as good as before".

### Set C — the criterion fails

Inside the button, of 240, served render, 1024 px:

| KV | head (L39.h10) | chain | guard |
|---|---|---|---|
| **hq-e8-2b, consumed keys (pre-registered)** | **212** | 211 | **222** |
| hq-e8-2b, keys before the codec | 232 | 211 | 237 |
| BF16 | 232 | 211 | 236 |

**The pre-registered criterion fails on both counts: 212 < 224, 222 < 233.**

The two controls were run after the failure, to attribute it:

- **The scenes are not the cause.** Under BF16 the same 240 pass both floors
  by a wide margin (232, 236).
- **hq upstream of L39 is not the cause.** With hq in every layer but the
  head scored against the keys before the codec, the head is 232 — BF16's
  number. The chain is the same 211 in all three arms (inside predicate
  agrees BF16 vs hq on 238 of 240, points within 3/999 at the median).
- **The codec on L39's own keys is the whole loss**: 21 scenes the head had
  with exact keys are lost with decoded ones, 1 gained. 11 of the 21 are
  targets under one token tall (< 32 px at 1024 px) — 11 of the 51 such
  targets in the set, against 10 of the other 189.

The guard falls with the head: 17 of its 18 failures are the head's region,
moved by the codec, taken over the chain — 12 of them over a chain that was
inside.

### Set C4096 — reported, not evaluated

240 scenes at 4096 px (`--side 4096 --seed 20260924`), served render:

| KV | head (L39.h10) | chain | guard |
|---|---|---|---|
| hq-e8-2b, consumed keys | **235** | 175 | 206 |
| BF16 | **238** | 170 | 203 |

**At 4096 px the codec barely touches the head** (235 against 238): a button
covers 16 times as many image keys, and the codec's per-key noise averages
out over the region. Its damage is worst where the target is small *in
tokens*: 1024 px, and thin buttons.

**The chain collapses at 4096 px, and not because of hq.** 170-175 against
211 at 1024 px, the same under BF16 and hq (inside agreement 217 of 240). Its
failures are of a kind it never had at 1024 px:

- At 1024 px all 29 chain failures are **wrong-element** (more than 150/999
  off), and the near misses do not exist: the error of the 211 inside is
  x −1.8, y −0.9 at the median.
- At 4096 px, of 70 failures (BF16) 31 are wrong-element and **39 are near
  misses, every one y-dominant**. Over all 209 scenes within 150/999 the
  chain's error is **x −1.4, y +11.4** at the median: a systematic drift
  **down**, x untouched.
- The drift grows with the target's height in the image — median y error by
  the target's y band (BF16): 0-199 +8.0, 200-399 +6.4, 400-599 +15.2,
  600-799 +15.3, 800-999 +14.6. hq gives the same table to within 2.
- The committed 4096 px fixture shows it too, under BF16 and the served
  render: +9.7, +4.2, +20.4 on its three scenes.

**The guard does not catch it, by construction.** The guard was designed on
a wrong-element tail: keep the chain unless it is far from the head. A chain
that drifts 10-30/999 below a button whose head region is centred stays
under d = 60, so the guard keeps it: 35 of the guard's 37 failures (BF16)
are the head inside and the guard trusting the drifted chain. **At 4096 px
the head alone (238) beats the guard (203) and the chain (170).** `d` was
not re-tuned on this set, and must not be: this set is now development data.

### Where the drift is not

Excluded, each by a measurement or a line-by-line check:

- **The codec**: the drift is the same under BF16.
- **Prefill chunk boundaries.** At 4096 px a 1024-token chunk is 8 image
  rows, so the image crosses ~16 chunks and later rows cross more of them.
  The harness now takes `IGNIS_POINT_CHUNK`; with the whole 16.5K-token
  prompt in **one** chunk (20480), on the first 80 scenes of C4096 BF16,
  the chain is 60 of 80 inside against 58 at 1024-token chunks, its
  near-miss y error +10.5 against +12.0 at the median, the band table the
  same to within 1.5, and its inside predicate agrees on 78 of 80 (points
  within 5/999 on 63). No boundary, the same drift.
- **Where the image's content lands in the cache** — the head's region is
  centred on the target in y at 4096 px (median −4/999), and a misplaced
  embedding would move the head with it. The code agrees on reading
  (`span_positions`, `chunk_media`).
- **The prompt's MRoPE positions** — checked exact against the reference
  on a 4800×3600 fixture (`crates/artifact/tests/fixtures/vision/expected/downscale.json`),
  axis by axis.
- **The interleaved MRoPE axis assignment** — `rope.cuh` (`axis = pair % 3`)
  and the reference's `apply_interleaved_mrope` give the same pair-to-axis
  map on all 32 rotary pairs (H 11, W 10, T 11).
- **The vision tower's position-table interpolation** — `vision_item_control`
  and the reference's `get_vision_interpolation_indices_and_weights`
  (transformers 5.17, `bilinear`, `align_corners=True` for this model) are
  the same taps and weights in the same arithmetic order.

### The reference drifts too: it is the model

What those checks leave is the model itself at its largest grid, or a
numeric difference in the engine's 4096 px path (the vision tower at 65,536
patches, the LLM at 16.5K positions). The PyTorch vehicle tells them apart:
BF16 source weights quantized NF4 with the vision tower left alone, the
reference implementation, and at 1024 px a chain that agreed with the
engine's inside predicate on 240 of 240. It had run out of memory at
4096 px because the LLM's SDPA got a mask and materialized
[heads, 16.5K, 16.5K]; `vehicle4096.py` computes that attention in blocks
of 1024 queries with an explicit causal mask — the same arithmetic, not all
rows at once — and leaves the tower's unmasked attention whole. Served
render, checked byte for byte against the engine dump's first render; the
processor's grid asserted at 256 x 256 patches and the prompt at the
engine's 16,506 tokens.

On the first 20 scenes of C4096 (BF16 engine arm):

- inside: **vehicle 11, engine 12**, the predicate agreeing on 19 of 20; the
  same 4 scenes are wrong-element in both.
- on the 16 near misses, the median y error is **+9.3 in the vehicle** and
  +14.6 in the engine, x −0.8 and −1.8; the vehicle lands below the target
  on 15 of 16. **The paired engine − vehicle difference in y has median 0**
  (mean +1.5, range −9 to +16): the two drift by the same amount, scene by
  scene, and the medians differ only because medians do not subtract.

The drift is the model's, at its largest grid. The engine's chain is
faithful to it.

### Re-measured with the residual window (#257, #258)

Everything above describes the engine before the residual window was wired.
After #257 (the window: 32 sinks, the 512 keys before the query's chunk and
the chunk itself read exact) and #258 (the chunk attended before it is
appended), the same harness on merged `main` (`a6b3791`), served render,
consumed hq keys, same scenes. The self-check now holds the capture to the
three-source rule: 666 exact rows at a median relative L2 of 0.0016, the rest
at the codec's 0.369, none clobbered, at both sizes.

| set, served, hq consumed | image keys from the codec | head | chain | guard |
|---|---|---|---|---|
| C, before the window | 100% | 212 | 211 | 222 |
| **C, with the window** | **40%** | **227** | 212 | **233** |
| C, BF16 | 0% | 232 | 211 | 236 |
| C4096, before the window | 100% | 235 | 175 | 206 |
| C4096, with the window | 96% | 236 | 169 | 201 |
| C4096, BF16 | 0% | 238 | 170 | 203 |

- **At 1024 px the window gives back 15 of the 20** — the head regains 21
  scenes and loses 6 — and set C's floors are met on the fixed engine:
  head 227 against 224, guard 233 against 233, the guard exactly on its
  floor. The pre-registered verdict stays what it was: it was taken once, on
  the engine of the time. This is the same criterion on a changed engine,
  with the head and `d` unchanged and set C used for no tuning.
- **At 4096 px nothing moves**: 96% of the image is still older than the
  window, and the head was already within 3 of BF16.
- **What an exact copy of L39's KV head would still buy** is the rest of
  the gap to BF16: 5 scenes at 1024 px (227 → 232, the guard 233 → 236-237)
  and 2 at 4096 px.
- The chain moves by a few units with the window (inside agreement 237 of
  240 at 1024 px, 220 at 4096 px) and keeps its 4096 px drift.

## Finding

- **Set C fails its pre-registered criterion, and the failure is the codec
  on the head's own read.** Everything else in production — the served
  render, hq in the other 63 layers, new scenes — leaves the head at 232,
  its BF16 number.
- **The residual window gives most of it back at 1024 px, and nothing at
  4096 px.** This bullet first said the window would not give the head its
  keys; at 1024 px that was wrong — it keeps 60% of the image exact, and
  the head went from 212 to 227 once #257 wired it. At 4096 px the image is
  96% older than the window, and there it holds.
- **What gives the head all its keys is an exact copy of one KV head.** The
  head reads one KV head of one layer (L39, KV head 1 = query head 10 / 6)
  over the image span. The exact rows exist in `run_gqa_layer` right after
  `qk_norm_rope`, before the append — the point the tap already reads —
  and a device-to-device copy into a per-sequence buffer costs 16,384 x
  256 x 2 bytes = **8 MB at 4096 px** (512 KB at 1024 px), nothing for a
  request without an image, no host, no codec. The "keys before the codec"
  arm measures exactly that configuration: 232 and 237 on set C. In the
  VRAM plan (ADR 0030) it is a reserve per slot, like the others, not an
  allocation per request.
- **At 4096 px the chain, not the head, is the problem** — and it is the
  shipped `point` at the endpoint's default size. It drifts down by ~11/999
  at the median, more in the lower half of the image; the head does not.
  **The drift is the model's, not the engine's**: the reference drifts the
  same, scene for scene. The guard as designed does not correct a drift,
  only a wrong element.

## Implications

- **With the window wired, the 1024 px floors are met without the exact
  key copy** — the guard only just. The copy is now worth 5 scenes of 240
  at 1024 px and 2 at 4096 px: a margin, not a precondition.
- **At 4096 px the head alone is the better point** on this set (238 against
  the chain's 170). Whether the product answer at 4096 is the head, the
  chain with a drift-aware guard, or a chain run at a smaller grid is a
  product decision, and it no longer waits on anything: there is no engine
  bug to fix first.
- **The chain's 4096 px drift is a property of the model at its largest
  grid**, and `/v1/decide` `point` inherits it at the default size: 170-175
  of 240 inside at 4096 px against 211 at 1024 px on scenes from the same
  generator (different seeds, so not paired). The levers are the head, or a
  smaller grid for `point`; neither is measured here as a product.
- **d = 60/999 was chosen on a 32 x 32 grid** and on a failure mode (wrong
  element) that is not the one at 4096 px. A second guard is designed on
  sets A, B, C and C4096 and measured once, on a new set.

## Limits and unknowns

- **The reference is 20 scenes in NF4**, not the served NVFP4: its points
  sit a median 6/999 from the engine's at 4096 px (3 at 1024 px). Enough to
  tell "the model drifts" from "the engine drifts" — the paired difference
  is centred on 0 — not to price the drift to the unit. The reference
  server (ninfer) was not run.
- **Synthetic scenes from one generator family**, two seeds, two sizes.
  The generator scales the content with the side: a 4096 px scene is a
  1024 px scene drawn four times larger (buttons 4.5% of the side tall at
  both sizes), not a 4K screen with 1x UI elements. So "the model at its
  largest grid" is here confounded with "the model on scaled-up content";
  whether the chain drifts on a real 4K screenshot is unmeasured.
- **The exact-copy design is measured by proxy**: "keys before the codec" is
  those keys scored on the host, not a copy in the engine.
- **The consumed-key capture runs one layer's scratch per query chunk**, and
  its self-check holds it to "no exact row"; the day the residual window is
  wired, that test fails on purpose and these hq numbers need re-running.
  That day was 2026-09-22 ([the residual window was the tool-call
  gap](2026-09-22-the-residual-window-was-the-tool-call-gap.md), GitHub #257):
  every hq number here describes the engine without it. At the query's chunk
  the window keeps 59.9% of a 1,024 px prompt's image exact and 3.7% of the
  4,096 px fixture's, so the 1,024 px arms are the ones to re-measure first.
  Re-measured the same day on set C and C4096: see *Re-measured with the
  residual window* above.
