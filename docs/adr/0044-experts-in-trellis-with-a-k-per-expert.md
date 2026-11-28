---
status: accepted
---

# Flash-Next's experts are trellis-coded with a bit width per expert, on our own kernels

Flash-Next's 120.8B expert parameters must fit host RAM beside the OS, about 53 GB while it runs: at 2.5 bits per
weight they take 37.7 GB pinned, and that budget fixes the average rate. The experts are stored in ExLlamaV3's
trellis format with a bit width K per expert, because at that rate the trellis measured better than the alternatives
on every layer and the owner chose a rate allocated per expert
([the compression study](../findings/2026-10-04-flash-next-compression-study.md)).

- **The rate.** K is one of 2, 2.5, 3 and 4 bits, 2.5 being the mul1 codebook's half step. The average is 2.5 bits
  per weight including scale overhead, with gate/up and down allocated separately. An expert's K is set by an
  allocation on calibration distortion curves.
- **The calibration corpus** is broad and balanced: code and agent traces, Italian and English prose, e-mail and
  technical documents, mathematics, exam-format questions, and the other languages the owner needs. It uses public
  data or the owner's own only.
- **The encoding is ExLlamaV3's trellis tensor, bit for bit**, with one added field, the expert's K
  (`docs/formats/flash-next-layout.md`). Bit-compatible storage makes ExLlamaV3's `reconstruct` an exact oracle for
  our decode kernel, and lets the converter use the quantizer that produced the study's numbers.
- **The MoE kernels are ours** (ADR 0043): ExLlamaV3 stores K per tensor, but its MoE kernel and loader take one K
  per projection per layer, so it cannot run an allocated model. The router, the decode GEMV and the prefill grouped
  GEMM per K class, the shared expert and the combine are written for ignis. ExLlamaV3's GEMV declares a
  QTIP-derived structure (GPL-3.0) and is not read for implementation at all (ADR 0051).
- **The converter depends on the exllamav3 Python package as an offline tool.** Its quantizer (MIT) takes our
  Hessians and K and returns the encoded tiles. It runs only at conversion time, its CUDA extension JIT-builds
  there, and no engine code links it.

## Considered Options

- **Entropy-coded scalar quantization with bit allocation across experts** set the quality to keep, an MMLU-Pro
  proxy of 73.0% against BF16's 73.7% (p = 0.82); the trellis, fed our per-expert Hessians, beat its MoE error on
  every layer measured.
- **A plain integer format** (K of 2, 3 or 4 per expert, group-128 scales, GPTQ): rejected. Its KLD is 1.4 to 1.9
  times higher on every domain and its MMLU-Pro 70.1%; even 3.1 bits does not catch up, and more bits do not fit
  RAM.
- **One rate for every expert**: the owner chose allocation, which had a KLD 1.15 to 1.35 times lower on calibrated
  domains and 2.9 MMLU points more (not significant). The uniform rate was more robust on languages absent from
  calibration.
- **Our own trellis quantizer** (Viterbi over the bitshift trellis, LDLQ feedback, scale refit): months of work for
  a result we can already measure.

## Consequences

- The kernel dispatches four K values, and residency has eight slot classes, two projection shapes by four K.
- Allocation trades robustness on unseen languages for quality on calibrated ones, so the calibration corpus is part
  of the artifact's contract: a language that matters must be in it. The artifact records its corpus manifest and
  its K map.
- A conversion is held to the MMLU-Pro proxy floor of 71% (the study measured 73.0%); one below it is judged case
  by case and the owner decides: to serve it,
  or to convert again, with the 3.0-bit average as the fallback that still fits RAM (45.3 GB pinned). The served
  artifact is 0.89 points under the floor (70.11% against 71%, equal to its quantized reference) and is served at
  2.5 bits by the owner's decision ([finding](../findings/2026-10-06-flash-next-on-the-5090.md)).
- Updating the exllamav3 pin is a converter change, re-verified by the bit-exact decode test.
- A uniform K remains expressible: one K for every expert is a degenerate map.
