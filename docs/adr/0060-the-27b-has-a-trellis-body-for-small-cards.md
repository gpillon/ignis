---
status: accepted
---

# ADR 0060 — the 27B gets a second body format, trellis, for the 16 GB card class

## Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. The owner's decisions are from the study and grill of 2026-10-20/21:
- the 27B on 16 GB Blackwell cards;
- a body around 3 bits per weight;
- the target context at least 256K;
- DFlash2 kept;
- two phases.

The body's source was decided by the owner on 2026-10-21, after M1 and M2: a **repack** of the public
`SC_3.00bpw_H4` (point 4).

Still open:
- the token embedding's format, W8 or BF16 (point 1).

Sources (untracked, local to the owner's clone): `.scratch/27b-16gb-study-2026-10-20/`, in particular
`SINTESI.md`, `PIANO-VRAM.md` §8-10, `PIANO-LAVORO.md`, `internal-facts.md`, `external-ecosystem.md`, and the
literature files there.

Related:
- [ADR 0044](0044-experts-in-trellis-with-a-k-per-expert.md): the trellis format, on Flash-Next's experts;
- [ADR 0051](0051-kernel-code-is-ours-adapted-or-vendored.md): kernel code is ours, adapted or vendored;
- [ADR 0061](0061-prefill-attention-reads-the-hq-codec.md), [ADR 0062](0062-the-token-embedding-lives-in-host-memory.md),
  [ADR 0063](0063-the-profile-follows-the-card-class.md) and [ADR 0064](0064-a-catalog-entry-states-its-card-class.md):
  the rest of the 16 GB plan.

## Context

The 27B's served artifact cannot be loaded on a 16 GiB card at any context:
- its text scope alone is 17.21 GB (16.03 GiB), measured from the artifact's directory;
- today's minimum plan (no speculation, no vision, 32K, 8 lanes) is 20.1 GB.

The body, 24.35 G linear parameters, is 14.51 GB at 4.76 bits per weight:
- NVFP4 at 4.5 bpw;
- plus 9 BF16 exception matrices.

Every lever that keeps NVFP4 was costed and fails:
- **NVFP4, squeezed** (embedding in host memory, the exceptions in NVFP4, an NVFP4 head, 1 lane, 32K, no DFlash2, no
  vision):
  - it only fits a card with nothing else on it;
  - it decodes 2.5-3× slower than the option below.
- **Weight offload over PCIe:** streaming the bits that do not fit costs 2-5× decode at DFlash2's acceptance. The
  offload wins in the literature need τ of 10-30 or batches of 100+.
- **A smaller model:** Qwen3.8 has no dense model below 27B. The nearest is two generations older and far weaker on
  agentic coding: Qwen3.5-9B scores 65.6 on LiveCodeBench, the 27B 90.3.
- **GGUF k/i-quants:** at equal VRAM they show ~2.5× the KLD of EXL3 on this exact model (turboderp's chart), and they
  would need a kernel family of their own.
- **Pruning:** every positive result paid 50-480B tokens of distillation.

A trellis code at ~3 bpw fits, with room for 256K and DFlash2. Measured on others' suites:
- turboderp's self-calibrated EXL3 at 3.0 bpw: KLD 0.0257;
- Unsloth's NVFP4: 0.0092;
- EXL3 at 4.0 bpw: 0.0082;
- a 12 GiB GGUF Q3: 0.0209.

Our served NVFP4 runs W4A4 (FP4 activations too), and FP4 activations account for half or more of an NVFP4 KLD in two
sources. A trellis body runs with 16-bit activations.

## Decision

1. **The 27B gets a second artifact of the same family, for the 16 GB card class, whose body format is trellis.**
   - The code is EXL3's: `mul1` codebook, Hadamard rotation with `suh`/`svh` scales, out scales on. It is bit-compatible
     with exllamav3's records, and the public EXL3 quants of this model use the same codebook, multiplier `0x83DCD12D`.
   - A **K** per matrix, from 2 to 6, the output head included.
   - The GDN small set (a/b, conv1d, A_log, dt_bias) and the norms stay BF16.
   - The token embedding lives in host memory (ADR 0062). Its format is the artifact's: W8 as today, or BF16, is open.
   - The 32 GB NVFP4 artifact, its kernels and its numbers do not change.
2. **Target body: ~3.0 bpw on average**; the first candidate is turboderp's `SC_3.00bpw_H4`. If it fails the gates, 3.25
   bpw, which needs ADR 0061 to reach 256K. The gates are the owner's; the suites are this repo's KLD set (2026-09-24
   windows) and the Flash-Next study's held-out set. Each is answered by a different measurement:

   | gate | answered by |
   |---|---|
   | **KLD ≤ 1.2×** the served NVFP4's on the same suite | M1 (weights only, BF16 activations) |
   | **MMLU-Pro** (n=281) not significantly below BF16 (McNemar p > 0.05) | M1 |
   | **DFlash2 tokens per round ≥ 90%** of those on the NVFP4 target | M2 (exllamav3's DFlash2 path), then again on the ignis artifact |
   | **loop rate** on the loop set no worse than today's | only on an ignis artifact |
3. **The DFlash2 module is grafted as it is.** The 66 NVFP4 objects are copied byte for byte from the served artifact,
   their provenance in the sidecar, by our own converter. The reference engine's graft tool refuses any base that is
   not its own NVFP4 recipe.
4. **Source of the body: a repack of `turboderp/Qwen3.8-27B-exl3` branch `SC_3.00bpw_H4`** (Apache-2.0, like the
   checkpoint). Decided by the owner on 2026-10-21. It rests on two measurements, both on this repo's text:

   **M1, weights only, BF16 activations** (`.scratch/27b-16gb-study-2026-10-20/RISULTATI-M1.md`):
   - KLD 0.0340 / top-1 93.88% on the 2026-09-24 windows;
   - the served NVFP4 on the same windows: 0.0458 / 92.92%;
   - lower in 16 of 16 windows;
   - MMLU-Pro 68.33% → 64.77%, McNemar p = 0.052;
   - weak spot: memorized Python, ppl +8.6%.

   **M2, DFlash2 tokens per round** (`m2/RISULTATI-M2.md`):
   - against an EXL3 5.0 bpw control, 1.00-1.06 in every cell (8K, 64K and 128K; greedy and sampled);
   - the 95% interval's lower bound is ≥ 0.92.
   - An MMLU anomaly in the BF16 reference proved to be the model's own numerical fragility on that text, not the
     harness.

   exllamav3 quantizes `in_proj_qkv` / `in_proj_z` and `q/k/v` as separate matrices with their own scales, so the
   27B's **fused planes** cannot be rebuilt by concatenation. The repack serves them split (spec 01, spec 02).

   **Our own conversion is the measured alternative, not the plan.** It would run exllamav3's quantizer on the fused
   planes, with our Hessians and a broad, self-generated calibration set. It is worth it only if the gates fail in
   ignis, or if the Python weak spot matters.
5. **The oracles.**
   - exllamav3's reconstruction is the bit-exact oracle of the tile decode, as for Flash-Next.
   - The BF16 checkpoint, run layer by layer, is the oracle of quality.
   - No reference engine runs this artifact: ADR 0002/0010 parity does not apply to it.
6. **The kernels are ours.**
   - Dense trellis GEMV/GEMM for 1 to lanes × 8 rows, decoding each fragment once for every verify row.
   - The prefill GEMM.
   - The head.
   - Adapted from our Flash-Next trellis decode under ADR 0051. The layer code picks the linear op by the tensor's format.
7. **Two phases.**
   - Phase 1: this artifact, on the 16 GB card class, with capacity and quality measured on the 5090. Decode speed is
     stated as an estimate until a real 16 GB card measures it; there is none in the project.
   - Phase 2: Flash-Next on 16 GB, the abliterated variant, the speed gate on real hardware, 12 GB.
   - After that, images.

## Considered options

- **2.5 bpw, as Flash-Next's experts.** Rejected for 16 GB. A dense model degrades faster than routed experts at equal
  bits: EXL3-SC at 2.2 bpw has KLD 0.088 on this model, past the point where AIME collapses on its predecessor. 2.5 is
  the 12 GB answer (phase 2).
- **3.25 bpw from the start.** Kept as the fallback. With prompt planes it did not reach 256K; without them it reaches
  256K, with less room for the second lane.
- **ExLlamaV3 as the serving engine.** Rejected, for the reasons ADR 0043/0044 recorded: it is an oracle here, not the
  engine.

## Consequences

- Two artifacts of one family to convert, publish and keep in step: two Hugging Face repos, two catalog entries
  (ADR 0064).
- Prefill on GeForce runs on FP16 tensor cores, without an FP4 copy of the weights. At ~65% of the FP16-accumulate
  peak that is ~1.3K tok/s on a 5060 Ti and ~2.3K on a 5070 Ti, so prefix reuse carries the agentic load on these
  cards.
- The body's quality is measured on the 5090. Its speed on 16 GB cards is a bandwidth estimate (5070 Ti ~100-160
  tok/s on agent traffic) until phase 2.
