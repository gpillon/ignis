# Flash-Next's experts at 2.5 bits a weight: allocation keeps MMLU-Pro at 73.0% against BF16's 73.7%, plain integers lose it, and ExLlamaV3's trellis beats both per layer

- Kind: experiment
- Status: current
- Observed: 2026-10-04
- Last verified: 2026-10-04
- Scope: Flash-Next's expert format and its converter (`tools/flash-next-converter/`, `docs/formats/flash-next-layout.md`)
- Topic: flash-next
- Related: [ADR 0044](../adr/0044-experts-in-trellis-with-a-k-per-expert.md),
  [ADR 0043](../adr/0043-a-second-model-without-a-ninfer-reference.md),
  [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md),
  [KLD against BF16](2026-09-24-kld-against-bf16.md)
- Superseded by: none
- Summary: Teacher-forced on 281 MMLU-Pro questions, Flash-Next's experts at 2.5 bits with a rate per expert score 73.0% (BF16 73.7%, the 27B 68.3%); plain integers reach 70.1%; ExLlamaV3's trellis is ~1 dB better a layer on layers 0-5.

Written on 2026-11-27 from the compression study of 2026-10-03 and 2026-10-04,
when ADRs 0043 and 0044, which had carried these numbers, were brought to the
short format. Nothing was measured again. The study's reports and scripts are
in the main checkout's `.scratch/flash-next-compression-2026-10-03/`
(`RISULTATI_3.md`, runs 6 to 8; `exl3/run8_phaseA.md`; untracked).

## Question

Qwen3.8-Flash-Next has 120.8B expert parameters. On the development machine
(63.8 GB of RAM, about 53 GB available beside the OS while the model runs)
they must sit in pinned host RAM. Which storage format keeps the model's
quality at a rate that fits, and should every expert get the same rate?

## Evidence

All runs are teacher-forced through the checkpoint's own modeling code, layer
by layer: a KLD per domain against the BF16 weights with a bootstrap per
chunk, and an MMLU-Pro proxy on 281 questions with a paired test (McNemar).
The calibration set is the same in every run; German and Japanese are never
in it. The non-expert tensors are FP8 with a scale per row and the n-gram
table is INT4 in groups of 32, unless a row says otherwise.

**Memory.** At 2.5 bits a weight the experts take 37.7 GB pinned, which leaves
about 14 GB for the table's hot rows and the page cache; at 3.0 bits they take
45.3 GB and leave about 6.5 GB.

**Run 6: entropy-coded scalar quantization** (the bits are an ideal entropy,
not a format a kernel reads).

| variant | expert bits | KLD code / prose | KLD en / it / zh / math / py | KLD de / ja | MMLU-Pro | against BF16 | against the 27B (68.3%) |
|---|---|---|---|---|---|---|---|
| BF16 | 16 | - | - | - | 73.7% | - | p = 0.08 |
| non-experts in FP8 only | 16 | 0.058 / 0.084 | 0.029 / 0.015 / 0.025 / 0.015 / 0.009 | 0.024 / 0.022 | 74.4% | p = 0.79 | p = 0.04 |
| 2.5 bits allocated per expert, floor 1.5 | 2.50 | 0.119 / 0.177 | 0.114 / 0.056 / 0.089 / 0.044 / 0.028 | 0.224 / 0.211 | 73.0% | p = 0.82 | p = 0.15 |
| the same, n-gram table INT2 | 2.50 | 0.143 / 0.207 | 0.280 / 0.061 / 0.102 / 0.145 / 0.107 | 0.245 / 0.225 | 71.9% | p = 0.36 | p = 0.25 |
| 3.0 bits allocated per expert, floor 1.5 | 3.00 | 0.093 / 0.133 | 0.079 / 0.036 / 0.058 / 0.038 / 0.024 | 0.176 / 0.175 | 72.6% | p = 0.65 | p = 0.16 |
| 2.5 bits uniform (rotated GPTQ) | 2.51 | 0.144 / 0.195 | 0.128 / 0.074 / 0.105 / 0.059 / 0.042 | 0.147 / 0.129 | 70.1% | p = 0.06 | p = 0.63 |

Allocation against the uniform rate, both at 2.5 bits: on the calibrated
domains the allocated variant's KLD is 1.15 to 1.35 times lower and its
MMLU-Pro 2.9 points higher (p = 0.13); on German and Japanese the uniform
variant is better, 0.13-0.15 against 0.21-0.22.

**Run 7: a plain integer format.** K of 2, 3 or 4 whole bits per expert, a
uniform grid, one fp16 scale per 128 weights with a clipping search, GPTQ with
the same Hessians. The BF16 stream answers identically in runs 6 and 7, so the
two are paired.

| variant | bits | KLD code / prose / en / it | KLD de / ja | MMLU-Pro | against BF16 | against run 6 |
|---|---|---|---|---|---|---|
| run 6, allocated | 2.50 | 0.119 / 0.177 / 0.114 / 0.056 | 0.224 / 0.211 | 73.0% | p = 0.82 | - |
| integers, K allocated per expert | 2.50 | 0.171 / 0.255 / 0.194 / 0.107 | 0.341 / 0.328 | 70.1% | p = 0.08 | KLD 1.4 to 1.9 times on every domain; MMLU 17 lost, 9 gained, p = 0.17 |
| integers, K = 3 for all | 3.12 | 0.151 / 0.196 / 0.147 / 0.078 | 0.153 / 0.142 | 71.2% | p = 0.17 | KLD 1.4 to 2.2 times run 6's 3.0-bit variant, 0.8 to 0.9 times on de / ja |

In the allocated integer variant, over all layers, 66.2% of the gate/up
projections are at 2 bits, 30.2% at 3 and 3.7% at 4 (down: 67.3%, 28.0%,
4.8%); the last layers ask for more of the top class, up to 10%.

**Run 8, phase A: ExLlamaV3's trellis on our Hessians.** ExLlamaV3 1.5.3, the
mul1 codebook, per-expert Hessians; K of 2, 2.5, 3 or 4 per expert by
allocation, against K = 2.5 for all, its native form. Layers 0 to 5 only. The
MoE error of a layer in dB, lower is better; the pass rule, set before the
run, was within 0.5 dB of run 6.

| layer | trellis, allocated | trellis, uniform | run 6, allocated | run 6, uniform | run 7, integers |
|---|---|---|---|---|---|
| 0 | -20.97 | -20.92 | -19.96 | -20.17 | -15.23 |
| 1 | -18.38 | -18.15 | -17.28 | -17.23 | -13.43 |
| 2 | -16.86 | -17.76 | -16.26 | -16.72 | -10.75 |
| 3 | -17.37 | -15.59 | -16.61 | -14.60 | -12.73 |
| 4 | -16.68 | -15.79 | -15.67 | -14.83 | -11.95 |
| 5 | -14.53 | -15.11 | -12.71 | -14.10 | -10.53 |
| mean of 1-5 | -16.76 | -16.48 | -15.71 | -15.50 | -11.88 |

Rates with the scales included: 2.4992 (gate/up) and 2.5000 (down) allocated,
2.5187 and 2.5312 uniform. In layer 1 the allocation put 221, 114, 165 and 12
of the 512 gate/up projections at K = 2, 2.5, 3 and 4 (43%, 22%, 32%, 2%), and
233, 106, 159 and 14 of the down projections. The sweep of four K values took
about 650 s a layer.

## Finding

Observed: at 2.5 bits a weight with a rate per expert, Flash-Next stays at its
BF16 level on the MMLU-Pro proxy (73.0% against 73.7%, p = 0.82) and about 5
points above the 27B on the same 281 questions (68.3%), a lead the sample does
not make significant (p = 0.15).

Observed: a plain integer format at the same rate does not keep that quality.
Its KLD is 1.4 to 1.9 times run 6's on every domain and its MMLU-Pro 70.1%,
and at 3.12 bits it is still behind run 6's 2.5 bits on the calibrated
domains.

Observed: ExLlamaV3's trellis, fed the same Hessians, has about 1 dB less MoE
error than run 6 on each of layers 0 to 5, and 2 to 6 dB less than the
integers.

Observed: allocation and one rate for all differ by 0.2 to 0.3 dB a layer, and
uniform wins on two layers of five; end to end, in run 6, allocation was
better on calibrated domains and worse on languages the calibration never saw.

Inferred: the error per layer does not decide between allocation and a uniform
rate; the end-to-end figures do, and they trade quality on calibrated domains
against robustness on unseen languages.

## Implications

- ADR 0044 takes the trellis as the experts' format, with a K per expert and
  a mean of 2.5 bits, and makes the calibration corpus part of the artifact's
  contract.
- Matching run 6 with integers would need more than 3 bits, about 47 GB
  pinned, which the memory plan does not hold.
- ADR 0043's reason for a second model rests on the 2.5-bit row: compressed,
  Flash-Next keeps its lead over the 27B.

## Limits and unknowns

- Every figure is teacher-forced: KLD, perplexity, MMLU-Pro on the answer
  letter. The samples are small: 2 to 4 chunks a domain, 281 questions.
- Run 6's bits are an ideal entropy. Run 8 measured the trellis on layers 0 to
  5 only; the whole model's KLD and MMLU-Pro with the trellis were first
  measured on the converter's output, where every KLD is under its limit and
  the proxy is 70.1%
  ([Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md)).
- The speeds and switch times the study estimated were simulated, and are not
  repeated here.

## Follow-ups

None.
