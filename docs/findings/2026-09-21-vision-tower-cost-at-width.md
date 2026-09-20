# The vision tower is quadratic because the checkpoint is, and it already runs at the card's rate

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / vision encode (`kernel/src/vision_encode.cu`,
  `crates/core/src/vision.rs`, `crates/server/src/media.rs`)
- Related: [GitHub #244](https://github.com/gpillon/ignis/issues/244),
  `.scratch/vision-fanout/specs/02-the-towers-cost-at-width.md`,
  [`2026-09-20-number-width-and-decide-e2e.md`](2026-09-20-number-width-and-decide-e2e.md),
  [`2026-09-20-shared-vision-embedding-fan-out.md`](2026-09-20-shared-vision-embedding-fan-out.md),
  [GitHub #243](https://github.com/gpillon/ignis/issues/243),
  [GitHub #248](https://github.com/gpillon/ignis/issues/248),
  [GitHub #249](https://github.com/gpillon/ignis/issues/249),
  [ADR 0010](../adr/0010-vendored-reference-kernels.md)
- Superseded by: none

## Question

`media.encode_seconds` gets steadily worse per merged column as an image gets
bigger — 6.6x worse at 16,384 columns than at 576, with a local exponent of
1.83 near the top. `vision_item_control` builds one attention segment per
temporal frame, so a still image is full attention over every patch, which
would produce exactly that curve.

Three things were unknown, and they lead opposite ways: whether full
attention is the served checkpoint's intended scheme or an unexamined path,
where the time actually goes at 16,384 columns, and whether there is an
operator or caller lever worth reaching for.

## Evidence

Raw material, scripts and the full tables: `.scratch/vision-fanout/244/state.md`.
Every run on the 5090, release build, `qwen3_8_27b_nvfp4full-v2`, `--vision`,
`hq-e8-2b`, `--max-context 40960`, `--spec dflash2 --draft-tokens 7`.

### 1. The scheme, from primary sources

- ignis pushes one `cu_seqlens` segment per temporal frame, of length `h*w`
  (`crates/core/src/vision.rs:409-410`).
- ninfer does the same: `segment_length = h*w`, `segment_count = t`
  (`src/targets/qwen3_6/impl/vision/control.cpp`).
- ninfer's independent Python reference builds `cu_seqlens` from
  `repeat_interleave(h*w, t)` and attends with
  `F.scaled_dot_product_attention(..., is_causal=False)`
  (`tools/reference/qwen3_6/common/vision_ops.py:56-58,133-146`).
- Its `VisionConfig` (`tools/reference/qwen3_6_27b/config.py:73-85`) declares
  depth 27, hidden 1152, intermediate 4304, heads 16, patch 16,
  temporal_patch 2, spatial_merge 2, position_embeddings 2304 — and **no**
  `window_size`, `fullatt_block_indexes` or `deepstack_visual_indexes`. That
  config is what `tools/parity/qwen3_6_27b/vision.py` checks against
  transformers' own `AutoModel.from_config(config.vision_config)` at blocks
  0, 13, 26 and the merger.
- The kernel is byte-identical to the reference's:
  `kernel/vendor/src/ops/kernel/vision_attention.cuh` and
  `.../launcher/vision_attention.cu` `diff` clean against ninfer's
  `src/ops/{kernel,launcher}/vision_attention.*`.

### 2. Where the time goes (Nsight Systems, two widths)

One server; a 768px warm-up outside the capture; then one `noul` decision
over a 1536px image (2,304 merged columns) and one over a 4096px image
(16,384 columns). Device time inside the encode, by stage:

| stage | 2,304 cols | % | 16,384 cols | % |
|---|---|---|---|---|
| attention (27x `vision_attention_flash_kernel<64,64>`) | 0.0640 s | 55.0 | 3.2440 s | 88.2 |
| block GEMMs (108x Q4/Q5 row-split: qkv, proj, fc1, fc2) | 0.0409 s | 35.2 | 0.3040 s | 8.3 |
| everything else (bias, gelu, residual, rope, merger, norms, patch embed) | 0.0114 s | 9.8 | 0.1311 s | 3.5 |
| **wall / device busy** | 0.1175 / 0.1163 s | | 3.6810 / 3.6791 s | |

Device idle inside the encode: 1.0% at 2,304 columns, **0.1%** at 16,384.

From 9,216 to 65,536 patches (x7.11): attention **x50.69**, where exact
quadratic is x50.57; block GEMMs x7.43, where linear is x7.11.

Achieved rate, counting QK^T + PV only (no softmax), BF16-equivalent:

| | 2,304 cols | 16,384 cols |
|---|---|---|
| attention | 165.1 TFLOP/s | **164.7 TFLOP/s** |
| block GEMMs (Q4/Q5 weights) | 185.3 TFLOP/s | 177.2 TFLOP/s |

The QK mainloop runs `QKKs = 5` k-steps of 16 for a head dim of 72
(`vision_attention.cuh:117,217`), so it pads 72 to 80; PV's `PVNt = D/8 = 9`
tiles are exact. Issued MMA work is `(80+72)/(72+72) = 1.0556x` the useful
FLOPs, which puts the attention kernel at 173.8 issued TFLOP/s.

The command that produced the capture (the `.nsys-rep` stays on disk — it
carries the process environment):

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --delay 40 --duration 100 -o .scratch/vision-fanout/244/nsys-vision \
  ignis-server.exe --vision ...
```

### 3. The width curve, un-profiled, and the fit

One server, default budget, three 4096px pointing scenes downscaled to six
sizes. `media.encode_seconds`, median of the three scenes:

| px | merged cols | encode | `a*P + b*P^2` | linear part | quadratic part |
|---|---|---|---|---|---|
| 768 | 576 | 0.0201 s | 0.0194 | 0.0154 | 0.0040 |
| 1024 | 1,024 | 0.0400 s | 0.0401 | 0.0275 | 0.0126 |
| 1536 | 2,304 | 0.1190 s | 0.1256 | 0.0618 | 0.0638 |
| 2048 | 4,096 | 0.3078 s | 0.3115 | 0.1098 | 0.2017 |
| 3072 | 9,216 | 1.2741 s | 1.2681 | 0.2471 | 1.0210 |
| 4096 | 16,384 | 3.6646 s | 3.6661 | 0.4394 | 3.2268 |

`a = 6.704e-6 s/patch`, `b = 7.513e-10 s/patch^2`, `P` = patches = 4x merged
columns. Crossover (quadratic = linear) at 2,231 merged columns.

The fit was made without the profiler, and its two terms land on the
profiler's two groups: quadratic 3.2268 s against attention 3.2440 s (0.5%)
at 16,384 columns, 0.0638 against 0.0640 (0.3%) at 2,304; linear 0.4394 s
against everything-but-attention 0.4351 s.

### 4. The processor's own ceiling

A 5120x5120 PNG came back with `media.vision_tokens = 16,384` and
`media.preprocess_seconds = 3.877` — downscaled to the same grid as the
4096x4096 one. The artifact's `max_pixels` is therefore 16,777,216 = 4096^2.

### 5. Pointing against image size

Same runs as §3; `point` rescaled to 4096-space, `box` edge error as the
larger of the two edges on each axis, `noul` = "the button text is legible".

| scene | px | cols | inside button | box err x / y | noul |
|---|---|---|---|---|---|
| large | 768 | 576 | yes | 33 / 11 | 0.980 |
| large | 2048 | 4,096 | yes | 2 / 8 | 0.993 |
| large | 4096 | 16,384 | yes | 6 / **67** | 0.967 |
| medium | 768 | 576 | yes | 35 / 17 | 0.893 |
| medium | 2048 | 4,096 | yes | 6 / 6 | 0.998 |
| medium | 4096 | 16,384 | yes | 2 / **49** | 0.974 |
| small | 768 | 576 | yes | 25 / 8 | **0.469** |
| small | 2048 | 4,096 | yes | 6 / 10 | 0.988 |
| small | 4096 | 16,384 | yes | 6 / **28** | 0.989 |

All eighteen rows are in `state.md`. Every point lands inside its button at
every size tried, on all three scenes.

### 6. The reference, same card, same artifact

One `POST /v1/chat/completions`, 4096^2 PNG, `max_tokens: 1`. Both sides
report `prompt_tokens = 16,404`, so the grid is the same.

| run | ignis | ninfer-serve |
|---|---|---|
| 1 (cold) | 6,387 ms | 6,205 ms |
| 2 | 2,300 ms | 60 ms |
| 3 | 2,257 ms | 67 ms |

ignis's line for run 1: preprocess 0.194 s, encode 3.856 s, prefill 17 chunks
over 16,404 tokens. On the repeats neither side re-encodes; ignis re-prefills
all 16,404 tokens while ninfer serves the identical prompt from its prefix
cache.

## Finding

**Full attention over every patch of a still image is the served
checkpoint's scheme, and ignis matches it.** Observed: three independent
implementations build the same one-segment-per-frame `cu_seqlens`, the config
carries no windowing fields, and the kernel is byte-identical to the
reference's. Inferred: since ninfer's Python reference is the thing its parity
tool compares against transformers' own vision tower, a windowed checkpoint
would have failed that comparison. Branch 2 of the spec — "a
correctness-shaped performance bug" — is closed.

**The quadratic term is the attention and nothing else.** Observed: from
9,216 to 65,536 patches the attention kernel scales x50.69 where exact
quadratic is x50.57, while every other stage scales x7.43 against a linear
x7.11. A two-term model fitted to six un-profiled widths reproduces the
profiler's split to within 0.5%.

**The tower is compute-bound at the kernel's asymptotic rate; there is no
unexamined path.** Observed: 0.1% device idle across the whole 16,384-column
encode, and the attention kernel achieves 164.7 TFLOP/s at 16,384 columns
against 165.1 at 2,304 — the same rate, so width costs it nothing in
efficiency. Inferred: the 1.0556x padding of head dim 72 to 80 in the QK
k-loop puts its issued rate at 173.8 TFLOP/s, inside the 177-185 TFLOP/s band
the tower's own Q4/Q5 GEMMs reach on the same card in the same encode; that
padding is the checkpoint's head dim, not a path. (As a footnote only, the
5090's spec dense BF16 peak is ~209 TFLOP/s.)

**3.67 s is the worst case for one still image, by construction.** Observed:
`max_pixels` = 16,777,216 = 4096^2, so 16,384 merged tokens is the processor's
own ceiling and a 5120^2 image is downscaled to the same grid. Cold, on the
same card and artifact, the reference takes 6,205 ms against ignis's 6,387 ms
for the same 16,404-token request — within the spread of the encode samples
themselves (3.66-3.93 s), and both run the byte-identical kernel, so the
comparison could only ever have shown an orchestration difference.

**A caller who wants a cheap image should send a smaller one, and loses
nothing by it.** Observed: `point` lands inside the button on all three scenes
at every size from 768 px up, where 576 columns encode in 0.020 s against
3.665 s — **180x cheaper**. `box` edges are *worst* at 4096 px, with a y error
of 28-67 px against 6-10 px at 2048 px; 2048 px / 4,096 columns (0.31 s) is
the best accuracy in the sweep as well as 12x cheaper than the full-size
image. The `noul` collapse to 0.469 on `small` at 768 px is the model being
right: that button's text is 60x17 px there and genuinely is not legible.

### Two things found on the way

**`--vision-max-tokens` refuses images; it never shrinks them.** Observed: at
every budget below an image's natural token count, all three scenes came back
`400` in ~100 ms with no `ignis.request.admitted` line. The flag reaches
`ProcessorOptions::max_vision_tokens` (`crates/server/src/media.rs:169`) and
is spent as `check_budget(Budget::VisionTokens, ...)`
(`crates/artifact/src/vision/mod.rs:390`) **after** `smart_resize`
(`mod.rs:378`), which sizes the image from `min_pixels`/`max_pixels` read from
the artifact's own `preprocessor_config.json`. So the spec's assumption that
the budget is the operator's cost lever is wrong: the only lever today is the
caller resizing the image before sending it. ninfer has no corresponding flag.

**An image fan-out already pays for the tower once, and it still does.**
Re-running the exact bodies of the 6.84 / 13.50 / 27.92 s fan-out — measured
at `f50e52c`, before [#243](https://github.com/gpillon/ignis/issues/243)
landed — at `43636c5` gives 6.71 / 4.78 / 9.61 s, with `media.encode_seconds`
across the four admitted decisions reading 3.931, 0.000, 0.000, 0.000. That
corroborates
[`2026-09-20-shared-vision-embedding-fan-out.md`](2026-09-20-shared-vision-embedding-fan-out.md)
on a later build: its 4q figure for a picture already seen is 9.49 s against
9.61 s here, and the 2q figure is lower here only because the preceding 1q
had already paid the encode.

## Implications

- `.scratch/vision-fanout/specs/02` closes on branch 1: the price of a
  16,384-column image is the price of the scheme. Nothing in the encode is
  worth optimising; the whole of it is at the card's rate with 0.1% idle.
- The engineering lever that remains is not making the tower faster but
  running it less often — the embedding cache
  ([#243](https://github.com/gpillon/ignis/issues/243), spec 01) and warming
  it ahead (spec 04) — and the product lever is telling callers that a 2048 px
  screenshot points as well as a 4096 px one for a twelfth of the cost.
- [`2026-09-20-number-width-and-decide-e2e.md`](2026-09-20-number-width-and-decide-e2e.md)'s
  fan-out row and its "the vision embedding wants a cache of its own"
  implication were both true when written and are both answered by #243 and
  its own finding. Its width table (576-16,384 columns, 0.022-4.111 s) reads
  10-14% above this one's on the same widths; the two were taken on different
  builds and neither has been re-run against the other.

## Limits and unknowns

- No HF `config.json` for this checkpoint is on this machine. The chain from
  ignis to the checkpoint's declared attention scheme runs through ninfer's
  reference and its parity tool, not through the config file itself.
- The pointing sweep is the same three synthetic 4096x4096 scenes the pointing
  finding used, downscaled. It says nothing about photographs, dense text, or
  a scene whose target is small in the *original*.
- `box` getting worse at the largest size is three scenes and one axis. It is
  an observation, not a rule about bounding boxes.
- The reference comparison is one cold request per side. It establishes that
  ignis is not slow against ninfer at this width; it does not resolve a 3%
  difference.
- Cross-request reuse of the embedding held here because the requests carried
  identical bytes and the default pool holds one envelope-wide embedding. The
  width sweep shows every *different* image paying a fresh encode.
- The first encode on a fresh server reads 3.86-3.93 s against 3.66-3.69 s
  warm. Not decomposed.

## Follow-ups

- `--vision-max-tokens` semantics: either wire the budget into the resize's
  `max_pixels` so it shrinks, or keep it a refusal cap and say so in `--help`.
  [GitHub #248](https://github.com/gpillon/ignis/issues/248).
- An identical repeat request re-prefills all 16,404 tokens (2.26 s) where the
  reference serves it from its prefix cache in 60 ms. Not diagnosed; it may be
  what the turn-opening capture rule is supposed to do for a single-turn
  request. [GitHub #249](https://github.com/gpillon/ignis/issues/249).
