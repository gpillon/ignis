# A public 3-bit EXL3 quant of the 27B is closer to BF16 than the served NVFP4 on our text, and DFlash2 keeps its acceptance on it

- Kind: experiment
- Status: current
- Observed: 2026-10-21
- Last verified: 2026-10-21
- Scope: model / weight quantization of Qwen3.8-27B: the public EXL3 quant `turboderp/Qwen3.8-27B-exl3` branch `SC_3.00bpw_H4` (trellis, `mul1`, K per matrix 2-5, head at 4 bits), against BF16 and against ignis's served NVFP4; DFlash2 speculation on that target
- Topic: artifacts-and-gates
- Related: [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md); spec [27b-16gb](../specs/27b-16gb/spec.md); [KLD against BF16](2026-09-24-kld-against-bf16.md); [DFlash2 sampled acceptance](2026-09-24-dflash2-sampled-acceptance.md)
- Superseded by: none
- Summary: On the 2026-09-24 windows the public 3.0 bpw EXL3 quant of the 27B is KLD 0.0340 against BF16 versus the served NVFP4's 0.0458, lower in 16 of 16 windows, and DFlash2's tokens per round on it equal a 5 bpw control at 8K-128K.

## Question

Can a ~3 bpw body make the 27B fit a 16 GB card without losing what the 5090 serves today? Two questions:
- **(M1) Quality.** How close to BF16, on this repo's own text, is the best public ~3 bpw quant of the model, against
  the served NVFP4?
- **(M2) Speculation.** Does DFlash2, trained against a BF16 target, keep its acceptance when the target is that
  3-bit quant?

## Evidence

**M1, weights only.**

The harness (`.scratch/27b-16gb-study-2026-10-20/m1/m1_exl3.py`, untracked) is the method of `ref_kld.py`:
- transformers' Qwen3_5 modules run layer by layer;
- two hidden-state streams on one RTX 5090;
- the BF16 checkpoint (`Y:/models/Qwen3.8-27B`), and the same layers with every quantized linear replaced by
  exllamav3 1.5.3's reconstruction (`LinearEXL3.get_weight_tensor()`, fp16 cast to BF16);
- activations and attention in BF16, no KV quantization;
- the tensors the quant leaves unquantized (GDN a/b, conv1d, A_log, dt_bias, norms, embedding) taken from the
  checkpoint.

Two sets:
- **A:** the 2026-09-24 KLD windows, 16 × 4,096;
- **B:** the Flash-Next study's held-out chunks, 66 × 2,048, with 281 MMLU-Pro questions scored by argmax over the
  option letters.

Set A, against BF16:

| | KLD mean | median | p99 | top-1 | ppl (BF16 2.690) |
|---|---:|---:|---:|---:|---:|
| EXL3 `SC_3.00bpw_H4`, weights only | **0.0340** | 0.0054 | 0.282 | **93.88%** | 2.733 |
| served NVFP4, weights only (2026-09-24 finding) | 0.0458 | 0.0078 | 0.377 | 92.92% | 2.744 |
| served NVFP4, serving (hq-e8-2b) | 0.0513 | 0.0087 | 0.438 | 92.45% | 2.752 |

- Per kind, EXL3 against NVFP4 weights only: code 0.0321 / 0.0441, prose 0.0415 / 0.0597, chat 0.0310 / 0.0382.
- Window by window, EXL3 is lower in 16 of 16. The paired difference is −0.0118, 95% CI [−0.0144, −0.0091]; top-1
  +0.95 points.
- The body is 3.000 bits per weight against the NVFP4's ~4.5.

Set B, against BF16:

| set | KLD | top-1 |
|---|---:|---:|
| all, MMLU excluded | 0.0548 | 91.73% |
| all, Python also excluded | 0.0408 | 91.56% |

| domain | ppl change |
|---|---:|
| **Python** | **+8.6%** |
| de | +3.7% |
| prose | +3.1% |
| it | +2.8% |
| ja | +2.7% |
| zh | +2.6% |
| code | +2.4% |
| en | +1.9% |
| chat | +1.4% |

MMLU-Pro: 68.33% → 64.77% (−3.56 points; McNemar 16 lost, 6 gained, p = 0.052).

The reference reproduces the earlier runs:
- set A's ppl 2.690 against 2.689;
- set B's ppl exact on all 11 kinds;
- MMLU 281 of 281 answers identical to 2026-10-03.

**M2, DFlash2 acceptance.** exllamav3 1.5.3's own DFlash2 path:
- drafter `z-lab/Qwen3.8-27B-DFlash2` (BF16 loaded as fp16), 7 drafts per round;
- 8-bit KV on every target;
- free generation of up to 768 tokens;
- 52 real agent turns (8K / 64K / 128K buckets) and 8 coding canaries;
- greedy, and sampled at T=1, top-p 0.95, top-k 20.

The control is the same family's EXL3 at 5.00 bpw. The 6.00 bpw does not fit at 128K; it is checked at 8K and on the
canaries.

| bucket | mode | τ SC 3.0 | τ control 5.0 | ratio [95% CI] |
|---|---|---:|---:|---:|
| canary coding | greedy | 5.313 | 5.306 | 1.001 [0.949, 1.057] |
| 8K | greedy / sampled | 4.247 / 3.841 | 4.242 / 3.781 | 1.001 [0.958, 1.039] / 1.016 [0.975, 1.050] |
| 64K | greedy / sampled | 4.022 / 3.645 | 3.998 / 3.438 | 1.006 [0.952, 1.065] / 1.060 [1.022, 1.103] |
| 128K | greedy / sampled | 4.482 / 3.904 | 4.374 / 3.818 | 1.025 [0.933, 1.128] / 1.023 [0.967, 1.087] |
| all agent turns | greedy / sampled | 4.219 / 3.777 | 4.173 / 3.650 | 1.011 [0.977, 1.047] / 1.035 [1.010, 1.059] |

Against the 6.00 bpw: 0.959-0.987, lower bounds ≥ 0.92.

The control matches the live reference engine on the same prompts: sampled τ 3.78 / 3.44 / 3.82 against ninfer's
3.34 / 3.45 / 3.89. On the canaries greedy it gives 5.31, against ignis's measured 5.43.

**The MMLU anomaly**, a check inside M2. On the MMLU-format chunks the BF16 reference makes confident absurd
predictions at 5.9% of positions, and a non-letter argmax after `Answer:` in 210 of 281 questions:
- exllamav3's native forward reproduces them on the same quant: ppl 16.49 against the harness's 16.60; MMLU 65.5%
  against 64.8%;
- FP32 does not cure them on the worst chunk: 312 of 347 anomalous positions are the same;
- the packing is right: after `Answer:` the true token is the letter;
- on that text the model is ~450× more sensitive to numerics than on English, and worse with a longer few-shot context.

Raw material, untracked:
- `.scratch/27b-16gb-study-2026-10-20/RISULTATI-M1.md`, `m1/`;
- `.scratch/27b-16gb-study-2026-10-20/m2/RISULTATI-M2.md`, `m2/`.

## Finding

Observed:
- **Quality.** On this repo's code, prose and chat, the public 3.0 bpw EXL3 quant is closer to BF16 than the served
  NVFP4: KLD 0.0340 against 0.0458, lower in every window, top-1 +0.95 points. Its body is 1.5 bits per weight
  smaller.
- **The weak spot.** Memorized Python is the one domain with a clearly larger loss (ppl +8.6%).
- **MMLU-Pro** drops 3.56 points, at the edge of significance (p = 0.052).
- **Speculation.** DFlash2's tokens per round on the 3-bit target equal those on a 5-bit control at 8K, 64K and 128K,
  greedy and sampled (ratio 1.00-1.06, every lower bound ≥ 0.92).
- **The MMLU anomaly is the model's,** not the harness's: two independent forward implementations and FP32 reproduce
  it. The MMLU proxy is a relative measure, ref against quant on one harness, not an absolute score.

Inferred:
- The comparison with NVFP4 is not like for like. The NVFP4 number is the engine's W4A4: FP4 activations, engine
  numerics, chunk 1024. M1 is weights only, BF16 activations, PyTorch.
- FP4 activations account for half or more of an NVFP4 KLD in two external sources. Part of the gap is that term,
  which a trellis body served with 16-bit activations does not pay.
- An ignis kernel adds its own numerics, and hq-e8-2b KV adds ~0.005 at 4K (more at 32-64K).

## Implications

- A trellis body at ~3 bpw is the 16 GB card class's body. The owner chose to repack this quant (ADR 0060 §4).
- The 16 GB artifact can keep DFlash2 with no drafter re-training.
- The gates on the ignis artifact (spec 27b-16gb/05) compare against the served NVFP4 on the same suite and path. This
  finding's numbers are the weights-only reference point, not the gate.
- Python-heavy work is where a 3-bit body would be noticed first. If spec 05's gates or users show it, a conversion of
  our own with code-heavy calibration is the measured alternative.

## Limits and unknowns

- Weights only, in PyTorch. The quant has not been run in ignis: not its kernels' numerics, not hq-e8-2b KV, not the
  host embedding.
- One quant, `SC_3.00bpw_H4`. 3.25 bpw and an allocation of our own are not measured.
- M2:
  - is exllamav3's engine: fp16 drafter, 8-bit KV, free generation, outputs ≤ 768 tokens, prompts ≤ 136K, one lane;
  - is not measured in ignis, not on a 16 GB card, not at more lanes;
  - its +2-6% on long buckets is inside the noise of free generation, not a gain.
- MMLU-Pro n = 281 has a ±5-point binomial half-width. The model's own fragility on that text limits what the proxy
  says.
- No speed is measured here on any 16 GB card. The project has none.

## Follow-ups

- Spec 27b-16gb/05's gates on the ignis artifact: KLD on the same path, MMLU on this harness, τ on the in-house
  replay, loop rate.
- The worst MMLU chunk in the ignis engine, as a third independent forward, once the 16 GB artifact loads.
