# ADR 0044 — Flash-Next's experts are trellis-coded with a bit width per expert, on our own kernels

## Status

Accepted (2026-10-04). The owner decided per-expert bit widths, our own MoE
kernels, and ExLlamaV3 as a reference only, and approved the specs without
vetoing the agent's proposals (format details, converter dependency). Spec
`flash-next/01`'s acceptance (per-layer distortion) and spec `flash-next/02`'s
(bit-exact decode) verify it.
**Builds on ADR 0043.**

Sources: the compression study, `.scratch/flash-next-compression-2026-10-03/`
(`RISULTATI_3.md` runs 6-8, `exl3/run8_phaseA.md`; untracked, local to the owner's
clone).

## Context

Flash-Next's 120.8B expert parameters must fit host RAM beside the OS: ~53 GB is
available while it runs. At 2.5 bits per weight they take 37.7 GB pinned, and
that budget fixes the average rate.

The study measured, at that rate:
- **Entropy-coded scalar quantization with bit allocation across experts**
  (run 6): MMLU-Pro proxy 73.0% against BF16 73.7% (p = 0.82). This is the
  quality to keep.
- **A plain integer format** (K ∈ {2, 3, 4} per expert, group-128 scales, GPTQ;
  run 7): KLD 1.4-1.9× run 6's on every domain, MMLU 70.1%. Even 3.1 bits does
  not catch up, and more bits do not fit RAM. A simple format is not enough.
- **ExLlamaV3's trellis** (mul1 codebook) fed with our per-expert Hessians
  (run 8, layers 0-5): MoE error −16.8 dB with allocation, −16.5 dB uniform,
  against run 6's −15.7 dB, better on every layer. The trellis is the format.

Allocation against one rate for all experts, end to end in run 6: allocation
had 1.15-1.35× lower KLD on calibrated domains and +2.9 MMLU points (not
significant). The uniform rate was more robust on languages absent from
calibration (de/ja KLD 0.13-0.15 against 0.21-0.22). Per layer the two differ by
only 0.2-0.3 dB. The owner chose allocation.

ExLlamaV3 stores K per tensor, but its MoE kernel and loader take one K per
projection per layer, so it cannot run an allocated model.

## Decision

**Experts are stored in a trellis format with a K per expert (owner).**
- K ∈ {2, 2.5, 3, 4} bits. 2.5 is the mul1 codebook's half step.
- The average is 2.5 bits per weight including scale overhead. Gate/up and down
  are allocated separately.
- A per-expert K is set by a Lagrangian allocation on calibration distortion
  curves.
- The calibration corpus is broad and balanced: code and agent traces, Italian
  and English prose, e-mail and technical documents, mathematics, exam-format
  questions, and the other languages the owner needs. It uses public data or
  the owner's own only.

**The encoding is ExLlamaV3's trellis tensor, bit for bit (agent proposal).**
Each expert projection carries:
- mul1 codebook, 16×16 tiles, a 128-wide Hadamard rotation on both sides;
- fp16 input and output channel scales (`suh`, `svh`);
- K packed in ExLlamaV3's tile layout;
- one added field: the expert's K.

Bit-compatible storage makes ExLlamaV3's `reconstruct` an exact oracle for our
decode kernel, and lets the converter use the quantizer that produced run 8's
numbers.

**The MoE kernels are ours (owner).** The router, the per-K-class decode GEMV
and prefill grouped GEMM, the shared expert and the combine are written for
ignis (spec `flash-next/02`). ExLlamaV3's kernels are read as a reference, never
copied. Its GEMV declares a QTIP-derived structure (GPL-3.0) and is not read
for implementation at all.

**The converter depends on the exllamav3 Python package as an offline tool
(agent proposal).** Its quantizer (MIT) takes our Hessians and K and returns the
encoded tiles. It runs only at conversion time, its CUDA extension JIT-builds
there, and no engine code links it. Writing our own trellis quantizer (Viterbi
over the bitshift trellis, LDLQ feedback, scale refit) is months of work for a
result we can already measure.

## Consequences

- The kernel dispatches four K values. Residency has eight slot classes, two
  projection shapes × four K. At 2.5 bits on average, run 8's
  allocation over {2, 2.5, 3, 4} put, in layer 1, about 43% of the experts at 2,
  22% at 2.5, 32% at 3 and 2-3% at 4. In run 7's integer sweep the deepest
  layers asked more of the top class.
- Allocation trades robustness on unseen languages for quality on calibrated
  ones. The calibration corpus is therefore part of the artifact's contract: a
  language that matters must be in it. The artifact records its corpus manifest
  and its K map.
- Run 8 covered layers 0-5. Whole-model KLD with the trellis is first measured
  on the converter's output (spec `flash-next/01`). If it misses run 6's
  quality, the fallback is a 3.0-bit average, which still fits RAM (45.3 GB
  pinned).
- Updating the exllamav3 pin is a converter change, re-verified by the
  bit-exact decode test.
- Uniform K remains expressible: one K for every expert is a degenerate map.
  The question is not reopened by these specs.
