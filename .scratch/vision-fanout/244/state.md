# #244 — what the vision tower costs at width

## Established without the GPU (primary sources)

1. **Segment rule.** ignis `crates/core/src/vision.rs:409-410` pushes one
   `cu_seqlens` segment per temporal frame, length `h*w`.
2. **The reference does the same.** ninfer `src/targets/qwen3_6/impl/vision/
   control.cpp` sets `segment_length = h*w`, `segment_count = t`, and pushes
   the same cumulative bounds. Its Python reference
   `tools/reference/qwen3_6/common/vision_ops.py:56-58` builds `cu_seqlens`
   from `repeat_interleave(h*w, t)` and attends with
   `F.scaled_dot_product_attention(..., is_causal=False)`.
3. **The checkpoint declares no windowing.** ninfer's `VisionConfig`
   (`tools/reference/qwen3_6_27b/config.py:73-85`) carries depth 27, hidden
   1152, intermediate 4304, heads 16, patch 16, temporal_patch 2,
   spatial_merge 2, position_embeddings 2304 — and **no** `window_size`,
   `fullatt_block_indexes` or `deepstack_visual_indexes`. That config is the
   one `tools/parity/qwen3_6_27b/vision.py` checks against transformers'
   own `AutoModel.from_config(config.vision_config)` at blocks 0/13/26.
   Caveat: no HF `config.json` is on this machine; the chain to the
   checkpoint runs through ninfer's reference + parity tool.

=> Branch 1 of the spec. Still-image full attention is the scheme, not a bug.

## The free fit (no GPU)

`encode = a*P + b*P^2`, P = patches = 4 x merged columns, least squares over
the six measured points:

  a = 7.639e-6 s/patch, b = 8.409e-10 s/patch^2

| cols | P | measured | predicted | linear | quadratic |
|---|---|---|---|---|---|
| 576 | 2,304 | 0.022 | 0.022 | 79.8% | 20.2% |
| 1,296 | 5,184 | 0.063 | 0.062 | 63.7% | 36.3% |
| 2,304 | 9,216 | 0.137 | 0.142 | 49.6% | 50.4% |
| 5,184 | 20,736 | 0.516 | 0.520 | 30.5% | 69.5% |
| 9,216 | 36,864 | 1.430 | 1.424 | 19.8% | 80.2% |
| 16,384 | 65,536 | 4.111 | 4.112 | 12.2% | 87.8% |

Crossover (quadratic = linear) at P = 9,085 patches = 2,271 merged columns.
Two terms fit all six points to within 4%.

Analytic check at P = 65,536 (pending the profile):
- attention 4*P^2*D*H*27 = 5.34e14 FLOP; against the fitted 3.61 s that is
  148 TFLOP/s.
- the per-patch GEMMs (qkv/proj/fc1/fc2 x27 + patch embed) 0.83 GFLOP/patch
  = 5.4e13 FLOP; against the fitted 0.50 s that is 109 TFLOP/s.
- RTX 5090 dense BF16 peak ~209 TFLOP/s.

## GPU runs planned

- A: nsys over one server, encodes at 2,304 and 16,384 columns, attribute
  device time by kernel name. Separates the quadratic term empirically.
- B: `--vision-max-tokens` sweep over the three committed pointing scenes,
  `point`+`box`+`noul`, recording `media.encode_seconds` and inside-button.
- C: ninfer-serve on the same artifact, same 4096^2 image, whole-request
  time against ignis's 7.14 s.

## Run A — nsys, two widths (done)

`.scratch/vision-fanout/244/nsys-vision.nsys-rep` (not committed: it embeds
the process env). One server, `--vision`, default budget; a 768px warm-up
outside the capture, then one `noul` decision over `s1536.png` (2,304
columns, 2,381 prompt tokens) and one over `s4096.png` (16,384 columns,
16,461 prompt tokens).

Device time inside the encode, by stage:

| stage | 2,304 cols | % | 16,384 cols | % |
|---|---|---|---|---|
| attention (`vision_attention_flash_kernel<64,64>`, 27x) | 0.0640 s | 55.0 | 3.2440 s | 88.2 |
| block GEMMs (qkv/proj/fc1/fc2, 108x Q4/Q5 row-split) | 0.0409 s | 35.2 | 0.3040 s | 8.3 |
| add bias | 0.0039 s | 3.3 | 0.0485 s | 1.3 |
| gelu | 0.0025 s | 2.1 | 0.0218 s | 0.6 |
| residual add | 0.0008 s | 0.7 | 0.0170 s | 0.5 |
| rope | 0.0006 s | 0.5 | 0.0155 s | 0.4 |
| merger GEMMs | 0.0022 s | 1.9 | 0.0152 s | 0.4 |
| layer norm | 0.0013 s | 1.1 | 0.0119 s | 0.3 |
| patch embedding GEMM | 0.0002 s | 0.2 | 0.0012 s | 0.0 |
| position embedding | 0.0000 s | 0.0 | 0.0002 s | 0.0 |
| **encode wall / device busy** | 0.1175 / 0.1163 s | | 3.6810 / 3.6791 s | |

Device idle inside the encode: **1.0%** at 2,304 columns, **0.1%** at 16,384.

Scaling, 9,216 -> 65,536 patches (x7.11):
- attention x50.69 — quadratic would be x50.57.
- block GEMMs x7.43 — linear.

Achieved rate (BF16-equivalent FLOP; the block GEMMs carry Q4/Q5 weights):
- attention 165.1 TFLOP/s at 2,304 columns, **164.7 TFLOP/s** at 16,384 —
  the same rate at both widths.
- block GEMMs 185.3 and 177.2 TFLOP/s.
- So attention runs at 89-93% of the rate the tower's own GEMMs reach on
  the same card in the same encode. (RTX 5090 dense BF16 peak ~209 TFLOP/s.)

## Run B — `--vision-max-tokens` sweep (done, and it answered a different question)

`.scratch/vision-fanout/244/budget/`. Six servers, budgets 16,384 / 8,192 /
4,096 / 2,048 / 1,024 / 512, three 4096^2 pointing scenes each.

Only 16,384 answered. Every lower budget returned **400 in ~100 ms with no
`ignis.request.admitted` line**: `"question \"where\": its prompt was refused
(400 Bad Request)"`. The flag does not downscale — it refuses.

The site: `crates/server/src/media.rs:169` puts the flag into
`ProcessorOptions::max_vision_tokens`, and
`crates/artifact/src/vision/mod.rs:390` spends it as
`check_budget(Budget::VisionTokens, ...)` **after** `smart_resize`
(`mod.rs:378`), which sizes the image from `min_pixels`/`max_pixels` read
from the artifact's own `preprocessor_config.json`
(`size.shortest_edge` / `size.longest_edge`). The budget can refuse an image;
it can never shrink one.

ninfer has no equivalent flag at all (`--vision` is the only vision switch).

## Run B2 — the caller's lever: image size (done)

`.scratch/vision-fanout/244/size/`. One server, default budget, three scenes
at 768/1024/1536/2048/3072/4096 px, `point` + `box` + `noul` per request.

`media.encode_seconds`, median of the three scenes:

| px | merged cols | encode | fit | linear part | quadratic part |
|---|---|---|---|---|---|
| 768 | 576 | 0.0201 | 0.0194 | 0.0154 | 0.0040 |
| 1024 | 1,024 | 0.0400 | 0.0401 | 0.0275 | 0.0126 |
| 1536 | 2,304 | 0.1190 | 0.1256 | 0.0618 | 0.0638 |
| 2048 | 4,096 | 0.3078 | 0.3115 | 0.1098 | 0.2017 |
| 3072 | 9,216 | 1.2741 | 1.2681 | 0.2471 | 1.0210 |
| 4096 | 16,384 | 3.6646 | 3.6661 | 0.4394 | 3.2268 |

Refitted `a = 6.704e-6 s/patch`, `b = 7.513e-10 s/patch^2`; crossover at
2,231 merged columns. The fit's **quadratic term lands on the profiler's
attention time**: 3.2268 s fitted vs 3.2440 s measured at 16,384 columns
(0.5%), 0.0638 vs 0.0640 at 2,304 (0.3%). Its linear term lands on the rest:
0.4394 vs 0.4351 s.

Accuracy, point rescaled to 4096-space, box edge error as max over the two
edges on each axis:

| scene | px | cols | point | inside | box err x / y | noul |
|---|---|---|---|---|---|---|
| large | 768 | 576 | (3152, 3349) | yes | 33 / 11 | 0.980 |
| large | 1024 | 1,024 | (3168, 3332) | yes | 12 / 24 | 0.994 |
| large | 1536 | 2,304 | (3149, 3339) | yes | 7 / 11 | 0.893 |
| large | 2048 | 4,096 | (3148, 3334) | yes | 2 / 8 | 0.993 |
| large | 3072 | 9,216 | (3145, 3337) | yes | 4 / 1 | 0.997 |
| large | 4096 | 16,384 | (3149, 3424) | yes | 6 / 67 | 0.967 |
| medium | 768 | 576 | (1301, 971) | yes | 35 / 17 | 0.893 |
| medium | 1024 | 1,024 | (1300, 972) | yes | 20 / 10 | 0.953 |
| medium | 1536 | 2,304 | (1280, 971) | yes | 11 / 7 | 0.982 |
| medium | 2048 | 4,096 | (1280, 980) | yes | 6 / 6 | 0.998 |
| medium | 3072 | 9,216 | (1283, 980) | yes | 4 / 3 | 0.958 |
| medium | 4096 | 16,384 | (1279, 988) | yes | 2 / 49 | 0.974 |
| small | 768 | 576 | (3653, 667) | yes | 25 / 8 | **0.469** |
| small | 1024 | 1,024 | (3668, 636) | yes | 16 / 32 | 0.924 |
| small | 1536 | 2,304 | (3656, 643) | yes | 7 / 8 | 0.984 |
| small | 2048 | 4,096 | (3658, 644) | yes | 6 / 10 | 0.988 |
| small | 3072 | 9,216 | (3665, 652) | yes | 3 / 7 | 0.977 |
| small | 4096 | 16,384 | (3657, 611) | yes | 6 / 28 | 0.989 |

Every point is inside its button at every size. Box edges are **worst at
4096** on the y axis (67 / 49 / 28 px) and best at 2048-3072 (2-10 px).

The oversize probe: a 5120x5120 PNG came back with
`media.vision_tokens = 16,384` and `media.preprocess_seconds = 3.877` —
the processor downscaled it to exactly the same grid. So the artifact's
`max_pixels` is 16,777,216 = 4096^2, **16,384 merged tokens is the
processor's own ceiling**, and 3.67 s is the worst case for a still image
by construction.

## Run C — the reference, same card, same artifact (done)

`.scratch/vision-fanout/244/ref/`. One `POST /v1/chat/completions`,
4096^2 PNG + one question, `max_tokens: 1`. Both sides report
`prompt_tokens = 16,404`, so the grid is the same.

| run | ignis | ninfer-serve |
|---|---|---|
| 1 (cold) | 6,387 ms | 6,205 ms |
| 2 | 2,300 ms | 60 ms |
| 3 | 2,257 ms | 67 ms |

Cold, ignis is **2.9% slower** than the reference on the whole request.
ignis's own line for run 1: preprocess 0.194 s, encode 3.856 s, prefill
17 chunks / 16,404 tokens.

On the repeats the two diverge for a reason that is not the tower: ignis
re-prefills all 16,404 tokens (`prefill_chunks_consumed = 17`) while
ninfer serves the identical prompt from its prefix cache. Neither re-encodes.

The vendored kernel is byte-identical to ninfer's:
`kernel/vendor/src/ops/kernel/vision_attention.cuh` and
`.../launcher/vision_attention.cu` `diff` clean against
`/f/ai/q38/ninfer/src/ops/kernel|launcher/`.

## Run D — the image fan-out, re-measured after #243 (done)

`.scratch/vision-fanout/244/fan/`, the exact bodies the earlier finding used
(`.scratch/decide-live/fan_{1,2,4}q.json`).

| | 1q | 2q | 4q |
|---|---|---|---|
| 2026-09-20 finding (pre-#243) | 6.84 s | 13.50 s | 27.92 s |
| this run (HEAD 43636c5) | 6.71 s | 4.78 s | 9.61 s |

`media.encode_seconds` across the four admitted decisions: 3.931, 0.000,
0.000, 0.000 — the tower ran **once**. #243's merge commit ef851e6 says so
in its subject ("the image fan-out that pays once").

## Padding note

`kernel/vendor/src/ops/kernel/vision_attention.cuh:117` fixes the QK MMA at
`QKKs = 5` k-steps of 16 = 80, for a head dim of 72; PV uses `PVNt = D/8 = 9`
tiles, exactly 72. Issued MMA work is therefore `(80+72)/(72+72) = 1.0556x`
the useful FLOPs, so 164.7 useful TFLOP/s is **173.8 issued TFLOP/s** —
inside the 177-185 TFLOP/s band the tower's own Q4/Q5 GEMMs reach.
