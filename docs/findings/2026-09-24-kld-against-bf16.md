# The all-FP4 artifact is KLD 0.046 from BF16, and hq-e8-2b adds half as much again at 32K-64K context

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: artifact quality (`qwen3_8_27b_nvfp4full-v2`, spec `artifact/04`), KV format (hq-e8-2b vs BF16), prefill chunk width
- Related: [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md), [the residual window was the tool-call gap](2026-09-22-the-residual-window-was-the-tool-call-gap.md),
  `.scratch/sota-research-2026-09-24/04-quant-model.md` (third-party KLD tables),
  raw material in `.scratch/kld-2026-09-24/`
- Superseded by: none

## Question

How far are ignis's next-token distributions from the BF16 checkpoint's? The
artifact quantizes every projection to NVFP4 W4A4 (GDN, attention and MLP), and
third-party measurements put such an "all-FP4" profile at twice the KLD of a mixed
one. How much of the distance is the weights, how much is the hq-e8-2b KV codec, and
does either depend on context length or on the prefill chunk width?

## Evidence

**Reference.**
- `Y:/models/Qwen3.8-27B`, the artifact's own revision pin, in BF16 under
  transformers 5.17's Qwen3_5 modules.
- Run layer by layer: each decoder layer is read once and run over every window, and
  the hidden states stay on the GPU. Host RAM (64 GB) does not hold the model beside
  the desktop.
- Teacher-forced, one sequence per window.
- The 4K set uses transformers' torch GDN path. The long set uses
  `flash-linear-attention` 0.5.2, which agrees with it to 0.45% relative, and a
  512-row blocked causal SDPA, which agrees with SDPA `is_causal` to 1e-3.
  Windows torch has no SDPA flash kernel.
- Scripts: `ref_kld.py`, `make_windows.py`, `make_long_windows.py`.

**Engine side.**
- A new measurement readout, `ignis_prefill_options::out_span_logits` /
  `step::prefill_program_span_logits`, returns the BF16 logits of every position of a
  prefilled span.
- The example `crates/core/examples/span_logits.rs` dumps them per window.
- GPU test `crates/core/tests/span_logits_gpu.rs`: rows match the prefix prompt's
  last-position logits bit for bit where the chunks coincide.
- Serving route: chunked prefill, engine default policy.

**Metric.** Per position, KL(ref ‖ engine) in nats over the full 248,320 vocabulary,
top-1 agreement, and the NLL of the actual next token.

**Set 1: 16 windows × 4,096 tokens (65,536 positions).**
- 6 windows of this repo's Rust/CUDA/TSX.
- 4 windows of findings and ADR prose.
- 6 windows of the model's own sampled chat turns: reasoning + answers from the
  effort sweep, templated.

| engine configuration | mean KLD | median | p99 | top-1 | ppl (ref 2.689) |
|---|---:|---:|---:|---:|---:|
| weights only: BF16 KV, 1024-token chunks | **0.0458** | 0.0078 | 0.377 | 92.92% | 2.744 |
| BF16 KV, 128-token chunks | 0.0453 | 0.0077 | 0.389 | 92.94% | 2.740 |
| BF16 KV, 1152-token chunks (unfused SwiGLU) | 0.0452 | 0.0077 | 0.377 | 92.93% | 2.740 |
| **serving: hq-e8-2b, 1024-token chunks** | **0.0513** | 0.0087 | 0.438 | 92.45% | 2.752 |

By kind, weights only:

| kind | mean KLD | top-1 |
|---|---:|---:|
| code | 0.044 | 93.3% |
| prose | 0.060 | 88.7% |
| own chat | 0.038 | 95.4% |

- The worst 1% of positions carry 24% of the KL mass.
- Paired over windows, bootstrap 95% CI:
  - hq − BF16 = +0.0055 [+0.0032, +0.0076], 14/16 windows;
  - 128-token − 1024-token chunks = −0.0005 [−0.0017, +0.0006];
  - 1152 − 1024 = −0.0006 [−0.0022, +0.0007].
- hq − BF16 by position:

  | positions | hq − BF16 |
  |---|---:|
  | 0–512 | −0.002 |
  | 512–1024 | −0.003 |
  | 1024–2048 | +0.006 |
  | 2048–3072 | +0.008 |
  | 3072–4096 | +0.011 |

**Set 2: long context, the last 2,048 positions of 6 windows.** Four windows are
32,768 tokens (code, code, prose, code) and two are 65,536 tokens (code, prose). The
head of each window is prefilled without logits, through the same 1024-token chunks.

| context | hq-e8-2b KLD | BF16 KLD | hq − BF16 | hq / BF16 | top-1 hq / BF16 |
|---|---:|---:|---:|---:|---:|
| 32K | 0.0541 | 0.0359 | +0.0182 | 1.57× | 92.85% / 94.01% |
| 64K | 0.0693 | 0.0477 | +0.0217 | 1.51× | 91.72% / 92.80% |

- All six windows: hq − BF16 = +0.0194, bootstrap 95% CI [+0.0141, +0.0240], 6/6
  windows.
- Pooled p99: 0.574 against 0.357.

## Finding

**Observed: the weights.**
- The all-FP4 artifact's distribution is KLD 0.046 from BF16, with 92.9% top-1, over
  code, prose and the model's own chat.
- On its own sampled text the gap is smallest: 0.038 KLD and a perplexity equal to
  BF16's.
- This sits between the third-party figures for this model: Unsloth's mixed NVFP4 at
  0.031, and "all-FP4" profiles at 0.064–0.065. Their corpora and engines differ, so
  the comparison is an order of magnitude, not a ranking.

**Observed: the chunk width.** The prefill chunk width does not move the mean. The
fused SwiGLU route at 128 and 1024 columns and the unfused route at 1152 are equally
far from BF16. The two routes still disagree with each other at single positions (KL
up to 0.08; `span_logits_gpu.rs`), and those per-position differences are noise of the
same size as the quantization itself.

**Observed: the KV codec.** hq-e8-2b's cost grows with the distance a query attends
over:
- it is nothing inside the first 1024-token chunk;
- +0.006 to +0.011 across 1K–4K;
- **+0.018 at 32K and +0.022 at 64K**: 50–57% more KLD than the weights alone, and a
  point of top-1.

**Inference.**
- At the contexts coding agents run in (20K–100K), the KV format is a
  quality lever of the same order as the weight format.
- hq-e8-2b is the default because it makes 8 lanes × long context fit (ADR 0022). This
  measurement puts a price on that choice.

## Implications

- A KV format between hq-e8-2b (9,216 B/token) and BF16 (65,536 B/token) is worth
  measuring for long-context quality. FP8 E4M3 is 32,768 B/token; ninfer's "hq-8b" is
  the other candidate. The hq long-context attention cost (compute-bound, SINTESI §2.2)
  points the same way.
- A weight upgrade (a mixed profile keeping sensitive tensors at FP8/MXFP6, as the
  third-party 0.03-and-below profiles do) buys at most the 0.046. The per-tensor
  sensitivity sweep (arXiv 2607.12266, L+1 runs) can now run against this reference:
  `ref_kld.py` already runs the model layer by layer.
- Measured-better is the default, but the KV format trades quality for lanes and
  context. The decision is the owner's.

## Limits and unknowns

- 65,536 positions at 4K, and 12,288 at 32K/64K from 6 windows. Enough to see the
  hq − BF16 difference (CIs exclude 0), not to rank the three chunk widths.
- Teacher-forced prefill logits only. The verify/decode route's numerics (T = 8 per
  lane) were not dumped; the 128-token chunk is the nearest measured shape.
- One reference implementation (transformers). Its BF16 arithmetic differs from the
  checkpoint's training-time numerics by an unmeasured, presumably small, amount.
- Corpus: this repository's code and docs, plus the model's own samples. No external
  benchmark text.
