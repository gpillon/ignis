# The 27B's dense trellis kernels on the 5090

- Kind: experiment
- Status: current
- Observed: 2026-10-23
- Last verified: 2026-10-23
- Scope: kernel / the dense trellis linear (`kernel/include/ignis_trellis_linear.h`): decode, verify and prefill
  routes at every matrix shape of the 27B's 16 GB artifact, with and without the debug SM cap
- Topic: artifacts-and-gates
- Related: [GitHub #337](https://github.com/gpillon/ignis/issues/337),
  [spec 27b-16gb/02](../specs/27b-16gb/02-the-dense-trellis-kernels.md) (Acceptance 4 and 5),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md),
  [Flash-Next's MoE decode](2026-10-05-moe-decode-is-structure-bound.md)
- Superseded by: none
- Summary: The 27B's dense trellis body reads at 65.0-65.4% of the 5090's DRAM roofline at one token, eight tokens costing 1.03-1.04x one, but its prefill reaches 212-238 TFLOPS at 1,024 tokens, 42-47% of the card's measured MMA rate.

## Question

Spec 02 asks the decode route to read the body at one token at 65% or more of the 5090's DRAM roofline, with eight
tokens costing at most 1.35 times one, and the prefill route to reach 60% of the FP16-accumulate tensor peak at 1,024
tokens. Where do the kernels land, what limits them, and what do the SM caps (36, 70, 84: a 5060 Ti, 5070 Ti, 5080)
say about small cards?

## Evidence

`kernel/build/tests/ignis_kernel_trellis_linear_bench.exe` (source `kernel/tests/bench_trellis_linear.cu`), RTX 5090,
2026-10-23, one run on a cool card: the decode leg first (`--caps 0,36,70,84 --no-prefill`), then the prefill leg (all
caps). Device time per call from a CUDA graph of calls cycling over enough copies of each weight (> 512 MB) that none
is in the 96 MB L2; every time includes the call's activation rotation launch. Bytes are the trellis plus `suh`/`svh`.
Roofline 1,792 GB/s; a streaming read measured 1,662-1,713 GB/s. `nvidia-smi` sampled every 500 ms beside the
prefill leg: SM clock median 2,887 MHz (1,965-2,917), power median 430 W, peak 584 W, at most 76 °C. The bench's
register-only fp16-input, fp16-accumulate `mma.sync` loop measured **502 TFLOPS at 2,890 MHz**, against the
whitepaper's 419 TFLOPS (rated at the 2.41 GHz boost). The raw report is the appendix below.

**Decode, uncapped** (each matrix at the dominant K the public `SC_3.00bpw_H4` quant gives it):

| matrix | in x out | K | 1 tok µs | % roofline | 8 tok | 16 tok | 64 tok | 8/1 | 64/1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| GDN qkv | 5120 x 10240 | 3 | 17.7 | 62.0% | 18.4 | 20.3 | 45.8 | 1.04 | 2.58 |
| GDN z | 5120 x 6144 | 3 | 13.1 | 50.5% | 13.4 | 15.2 | 33.7 | 1.02 | 2.58 |
| GDN out | 6144 x 5120 | 3 | 12.8 | 51.7% | 13.4 | 14.8 | 31.3 | 1.05 | 2.45 |
| GQA q | 5120 x 12288 | 3 | 21.1 | 62.6% | 21.6 | 24.7 | 73.2 | 1.03 | 3.48 |
| GQA k, v | 5120 x 1024 | 4 | 5.9 | 24.8% | 7.0 | 11.6 | 24.5 | 1.18 | 4.1 |
| GQA o | 6144 x 5120 | 3 | 12.7 | 51.9% | 13.4 | 14.9 | 31.3 | 1.06 | 2.47 |
| MLP gate + up (one SwiGLU call) | 5120 x 2 x 17408 | 3 | 50.7 | 73.7% | 53.1 | 62.2 | 168.7 | 1.05 | 3.33 |
| MLP down | 17408 x 5120 | 3 | 27.0 | 69.3% | 27.8 | 31.4 | 73.6 | 1.03 | 2.73 |
| fused gate_up | 5120 x 34816 | 3 | 49.6 | 75.3% | 51.6 | 60.7 | 153.5 | 1.04 | 3.10 |
| head | 5120 x 248320 | 4 | 385.7 | 92.0% | 403.4 | 471.8 | 1074 | 1.05 | 2.79 |
| a call's floor | 128 x 128 | 3 | 3.3 | — | 3.4 | 3.8 | 6.1 | | |

**The body at one token**, every (matrix, K) of the public quant at its layer count (`quantization_config.json`:
e.g. GDN qkv 4 x K2, 42 x K3, 2 x K4; gate and up, k and v, share a K in every layer; 9,136 MB): **7,845 µs =
1,165 GB/s, 65.0% of the roofline**, and 65.4% in a second run; eight tokens **1.03-1.04x**. With every matrix at its
dominant K instead (9,157 MB) the same kernels read 64.1-66.4% over the day's runs, 65.6% on a cool card: lower when
the card is hot from a prefill leg (64.7% in the prefill leg's own decode pass).

**Decode under the SM cap** (the body at the quant's K mix, one token; eight tokens):

| cap | body µs | GB/s | per SM | % of the 5090's roofline | 8/1 |
|---|---:|---:|---:|---:|---:|
| none (170) | 7,845 | 1,165 | 6.9 | 65.0% | 1.03 |
| 84 | 11,999 | 761 | 9.1 | 42.5% | 1.02 |
| 70 | 14,194 | 644 | 9.2 | 35.9% | 1.01 |
| 36 | 28,897 | 316 | 8.8 | 17.6% | 1.01 |

**Prefill**, TFLOP/s (2 x tokens x in x out per call), against the whitepaper's 419 and the measured 502:

| matrix | 1,024 tok | 8,192 tok | cap 84, 1K | cap 70, 1K | cap 36, 1K |
|---|---:|---:|---:|---:|---:|
| GDN qkv | 231.7 | 205.0 | 136.6 | 111.1 | 62.0 |
| GDN z | 196.7 | 197.2 | 130.4 | 109.8 | 60.7 |
| GDN out | 226.9 | 199.2 | 137.0 | 111.1 | 62.3 |
| GQA q | 225.9 | 198.5 | 132.5 | 120.6 | 61.0 |
| GQA k, v | 101-102 | 160 | 84-91 | 76-83 | 54 |
| GQA o | 233.9 | 191.9 | 136.6 | 110.6 | 62.3 |
| MLP gate + up | 212.4 | 208.3 | 137.0 | 113.6 | 59.2 |
| MLP down | 238.1 | 207.6 | 116.1 | 103.2 | 55.8 |
| head | 201.7 | 200.9 | 137.6 | 116.0 | 61.5 |

At 1,024 tokens the body's large matrices reach **212-238 TFLOPS: 51-57% of 419, 42-47% of the measured 502**.

**What limits the prefill**, Nsight Compute on `prefill_kernel<6>` at GDN qkv, 1,024 tokens (ncu's base clock,
1.89 GHz): tensor pipe active 57%, issue slots busy 39%, 2 warps per scheduler (1 CTA of 8 warps per SM, 213-235
registers), top stalls `wait` (fixed-latency dependencies) and `math_pipe_throttle`. The main loop's SASS, per 64
inputs (64 HMMA): 128 HADD2 + 128 FFMA (the fp32 fold every 32 inputs: two converts and two FMAs per MMA), ~190 for
the four tile decodes, 32 LDSM, ~100 addressing and control: 10 instructions per MMA. SM120 has no packed f32 FMA:
`fma.rn.f32x2` compiles to two FFMAs.

Variants measured on the way (each against the same day's baseline, uncapped):

| change | effect |
|---|---|
| decode: split for the card's resident CTAs (one unit each) instead of ~1,024 units | body 53.5% → 60.2% |
| decode: programmatic dependent launch after the rotation | +2.2 points (off: 57.6%) |
| decode: weights as the MMA's A, tokens as N (1 MMA per tile up to 8 tokens), fp16 accumulation folded every 32 | body 60.2% → 65.6%; fold every 128 inputs exceeded the tolerance (worst 1.12 of the bound) |
| decode: activation rows padded to 272 B in shared memory | 16 and 64 tokens 1.4-2x faster, one token unchanged; the same layout with a runtime stride lost 4 points at one token |
| prefill: 4-stage, 64-input pipeline, next tile row decoded under this row's MMAs | 185-226 → 200-236 TFLOPS |
| prefill: 64-token tiles, 2 CTAs per SM | ~195, slower |
| prefill: A fragments prefetched / decode plan hoisted | within run-to-run noise (±5%) |

## Finding

- **Decode meets Acceptance 4, at the threshold.** The body at one token, at the quant's own K per matrix, reads at
  65.0-65.4% of the roofline (64-66% across the day's runs at the dominant K), and eight tokens cost 1.03-1.04x one — the weight tile is the MMA's A operand and the tokens
  its N, so up to eight tokens are the same instructions. The large matrices stream at 69-75% and the head at 92%;
  the body is held back by the small ones (GDN z/out, GQA o at ~51%, k/v at 25%), which pay a call's fixed ~3.3 µs
  (two launches, the split reduction) on 4-12 µs of bytes.
- **Prefill misses Acceptance 5.** 212-238 TFLOPS at 1,024 tokens is 51-57% of the whitepaper's 419 and 42-47% of
  the 502 this card's MMA loop reaches at its real ~2.9 GHz clock. The decided fold every 32 inputs costs four
  instructions per MMA with no packed alternative on SM120, and one 8-warp CTA per SM cannot hide the decode's and the
  fold's fixed latencies. Four restructurings landed on the same plateau.
- **Small cards are near compute balance.** Capped, the decode streams ~9 GB/s per SM whatever the cap (the uncapped
  5090 is DRAM-bound at 6.9): 316 / 644 / 761 GB/s at 36 / 70 / 84 SMs, 71% of a 5060 Ti's 448 GB/s, 72% of a
  5070 Ti's 896 and 79% of a 5080's 960 — so on those cards decode speed is set by the tile decode's instruction
  count, as spec 02's Further Notes expected. These are the 5090's clocks and L2; a real card's own clock moves them.
- **Numerics.** Both routes accumulate in fp16 and fold into fp32 every 32 inputs, with a power-of-two activation
  scale per token and 128-input block that bounds every fold below 56,576 (< 65,504). Against the fp64 product of the
  decoded weights every route and shape stays inside |y − Y| ≤ 2^-8 |Y| + 2^-9 s (worst 0.92 of the bound, of which
  the BF16 output's rounding alone can take 2^-8 |Y|), including an adversarial input at the scale's bound at
  in = 17,408 on both routes (worst 0.67). The spec's fallback (reconstruct and BF16 GEMM) is not needed.

## Implications

- The prefill's speed is what Acceptance 5 cannot certify: spec 05's end-to-end prefill on the 16 GB artifact is the
  measurement that matters for a user, and the fold interval is the lever that moves it most.
- The decode's margin over 65% is inside run-to-run spread, so a regression of a point shows as a miss.
- The plan line at 16 decode and 2,048 prefill tokens is 72.4 MB, almost all the down projection's rotated
  activations (2,048 x 17,408 fp16); a 16 GB card's prefill chunk sets it.

## Limits and unknowns

One card, one day; the cool/hot spread of the body figure is 2 points. Synthetic weights (counter-hash trellis
words; the quant's K mix, not its bytes). The caps model a small card's SM count only — its DRAM, L2 and clocks are
the 5090's. ncu profiles at base clock.

## Follow-ups

- Acceptance 5 is open on GitHub #337 for the owner: accept the prefill as built, allow a fold every 64 inputs
  (halves the fold's cost; moves a decided spec item and needs the numerics re-checked at K = 17,408), or fund a
  deeper prefill restructure (stream-K for the wave tail, a decode-to-shared-memory B tile with a 2 x 4 warp layout).
- Decode levers left: fewer calls (q/k/v and qkv/z as one launch: ~3 µs each, ~0.25 ms a token), and a split rule
  that counts waves rather than units (at a 70-SM cap GDN qkv runs 80 units on 140 CTAs).
- Full-shape checks on real weights wait for spec 01's artifact.

## Appendix: the raw report

The bench's output for the run above. Decode leg (`--caps 0,36,70,84 --no-prefill`, its own run on a cool card, the
body at the quant's K mix):

```
== SM cap none (all SMs) ==
matrix              in     out   K |  1 tok us  %roof |  8 tok us 16 tok us 64 tok us |    8/1   64/1 
gdn qkv           5120   10240 3.0 |     17.61  62.4% |     18.27     20.10     45.40 |  1.037  2.578 
gdn z             5120    6144 3.0 |     13.02  50.7% |     13.28     15.10     33.58 |  1.020  2.580 
gdn out           6144    5120 3.0 |     12.67  52.1% |     13.29     14.75     31.02 |  1.049  2.449 
gqa q             5120   12288 3.0 |     21.02  62.7% |     21.45     24.54     72.80 |  1.020  3.463 
gqa k             5120    1024 4.0 |      5.95  24.7% |      8.28     11.66     24.91 |  1.392  4.189 
gqa v             5120    1024 4.0 |      6.56  22.4% |      8.37     11.46     24.93 |  1.276  3.802 
gqa o             6144    5120 3.0 |     12.61  52.3% |     13.28     14.78     31.02 |  1.053  2.461 
mlp gate+up       5120   17408 3.0 |     50.36  74.2% |     52.62     61.31    165.84 |  1.045  3.293 
mlp down         17408    5120 3.0 |     26.59  70.2% |     27.50     31.07     72.75 |  1.034  2.736 
fused gate_up     5120   34816 3.0 |     49.17  76.0% |     51.27     60.24    153.24 |  1.043  3.117 
fused qkvz        5120   16384 3.0 |     27.19  64.7% |     27.87     32.56     74.93 |  1.025  2.756 
fused qkgv        5120   14336 3.0 |     22.87  67.2% |     23.70     26.38     73.07 |  1.036  3.195 
head              5120  248320 4.0 |    386.97  91.7% |    417.92    472.73   1036.28 |  1.080  2.678 
floor 128x128      128     128 3.0 |      3.23   0.1% |      3.29      3.68      5.88 |  1.020  1.823 
body at the quant's K mix, one token: 9135.6 MB in 7844.8 us = 1165 GB/s, 65.0% of the 1792 GB/s roofline; 8 tokens 8108.7 us (1.034x)
== SM cap 36 ==
matrix              in     out   K |  1 tok us  %roof |  8 tok us 16 tok us 64 tok us |    8/1   64/1 
gdn qkv           5120   10240 3.0 |     65.07  16.9% |     65.46     77.46    209.42 |  1.006  3.218 
gdn z             5120    6144 3.0 |     42.01  15.7% |     42.46     50.27    140.11 |  1.011  3.335 
gdn out           6144    5120 3.0 |     49.70  13.3% |     50.11     59.56    170.09 |  1.008  3.423 
gqa q             5120   12288 3.0 |     72.42  18.2% |     65.75     97.30    212.49 |  0.908  2.934 
gqa k             5120    1024 4.0 |      8.89  16.5% |      9.37     11.90     25.44 |  1.054  2.861 
gqa v             5120    1024 4.0 |      8.96  16.4% |      9.53     11.90     25.46 |  1.064  2.842 
gqa o             6144    5120 3.0 |     49.68  13.3% |     50.14     59.52    167.13 |  1.009  3.364 
mlp gate+up       5120   17408 3.0 |    169.28  22.1% |    171.33    201.05    606.82 |  1.012  3.585 
mlp down         17408    5120 3.0 |    136.00  13.7% |    136.84    161.99    484.15 |  1.006  3.560 
fused gate_up     5120   34816 3.0 |    161.79  23.1% |    164.57    194.41    572.57 |  1.017  3.539 
fused qkvz        5120   16384 3.0 |     82.08  21.4% |     83.33     98.26    278.13 |  1.015  3.389 
fused qkgv        5120   14336 3.0 |     81.87  18.8% |     82.66     98.20    277.52 |  1.010  3.390 
head              5120  248320 4.0 |   1022.10  34.7% |   1037.95   1262.07   3787.38 |  1.016  3.705 
floor 128x128      128     128 3.0 |      3.16   0.1% |      3.20      3.48      5.60 |  1.010  1.768 
body at the quant's K mix, one token: 9135.6 MB in 28896.8 us = 316 GB/s, 17.6% of the 1792 GB/s roofline; 8 tokens 29156.0 us (1.009x)
== SM cap 70 ==
matrix              in     out   K |  1 tok us  %roof |  8 tok us 16 tok us 64 tok us |    8/1   64/1 
gdn qkv           5120   10240 3.0 |     42.27  26.0% |     42.91     50.62    140.59 |  1.015  3.326 
gdn z             5120    6144 3.0 |     24.02  27.5% |     24.34     28.97     71.58 |  1.013  2.981 
gdn out           6144    5120 3.0 |     20.38  32.4% |     20.80     24.43     85.17 |  1.021  4.178 
gqa q             5120   12288 3.0 |     42.54  31.0% |     43.29     50.72    140.83 |  1.018  3.310 
gqa k             5120    1024 4.0 |      6.94  21.2% |      7.48     10.25     17.92 |  1.078  2.582 
gqa v             5120    1024 4.0 |      7.49  19.6% |      8.38     10.97     18.26 |  1.119  2.438 
gqa o             6144    5120 3.0 |     20.34  32.4% |     20.62     24.56     85.17 |  1.014  4.188 
mlp gate+up       5120   17408 3.0 |     86.24  43.3% |     87.41    103.58    306.41 |  1.014  3.553 
mlp down         17408    5120 3.0 |     49.81  37.5% |     50.62     59.49    233.64 |  1.016  4.691 
fused gate_up     5120   34816 3.0 |     83.68  44.6% |     84.85     99.82    278.90 |  1.014  3.333 
fused qkvz        5120   16384 3.0 |     42.85  41.0% |     43.86     51.07    141.52 |  1.024  3.303 
fused qkgv        5120   14336 3.0 |     42.82  35.9% |     43.84     51.01    144.39 |  1.024  3.372 
head              5120  248320 4.0 |    570.05  62.3% |    564.98    681.92   1975.52 |  0.991  3.466 
floor 128x128      128     128 3.0 |      3.10   0.1% |      3.14      3.42      5.56 |  1.012  1.795 
body at the quant's K mix, one token: 9135.6 MB in 14193.9 us = 644 GB/s, 35.9% of the 1792 GB/s roofline; 8 tokens 14295.5 us (1.007x)
== SM cap 84 ==
matrix              in     out   K |  1 tok us  %roof |  8 tok us 16 tok us 64 tok us |    8/1   64/1 
gdn qkv           5120   10240 3.0 |     24.69  44.5% |     25.07     29.92     72.83 |  1.016  2.950 
gdn z             5120    6144 3.0 |     18.53  35.6% |     18.93     22.29     72.00 |  1.022  3.886 
gdn out           6144    5120 3.0 |     16.67  39.6% |     17.12     20.00     49.65 |  1.027  2.978 
gqa q             5120   12288 3.0 |     42.64  30.9% |     43.22     51.72    141.18 |  1.014  3.311 
gqa k             5120    1024 4.0 |      6.40  23.0% |      8.15     10.17     19.81 |  1.275  3.098 
gqa v             5120    1024 4.0 |      7.49  19.6% |      8.00     10.36     19.86 |  1.068  2.652 
gqa o             6144    5120 3.0 |     16.67  39.6% |     17.06     20.10     49.73 |  1.023  2.983 
mlp gate+up       5120   17408 3.0 |     86.35  43.3% |     87.12    104.08    298.95 |  1.009  3.462 
mlp down         17408    5120 3.0 |     39.30  47.5% |     39.57     46.40    125.32 |  1.007  3.189 
fused gate_up     5120   34816 3.0 |     83.97  44.5% |     85.13    100.12    288.72 |  1.014  3.438 
fused qkvz        5120   16384 3.0 |     43.08  40.8% |     43.82     52.10    142.27 |  1.017  3.302 
fused qkgv        5120   14336 3.0 |     42.81  35.9% |     43.47     52.07    142.22 |  1.016  3.322 
head              5120  248320 4.0 |    491.62  72.2% |    507.38    613.93   1673.56 |  1.032  3.404 
floor 128x128      128     128 3.0 |      3.10   0.1% |      3.13      3.42      5.56 |  1.010  1.795 
body at the quant's K mix, one token: 9135.6 MB in 11999.4 us = 761 GB/s, 42.5% of the 1792 GB/s roofline; 8 tokens 12240.4 us (1.020x)
```

Prefill leg, TF/s at 1,024 and 8,192 tokens per cap (its decode columns, run on a hot card, are omitted):

```
== SM cap none (all SMs) ==
matrix              in     out   K |  1K TF/s  8K TF/s
gdn qkv           5120   10240 3.0 |    231.7    205.0
gdn z             5120    6144 3.0 |    196.7    197.2
gdn out           6144    5120 3.0 |    226.9    199.2
gqa q             5120   12288 3.0 |    225.9    198.5
gqa k             5120    1024 4.0 |    101.2    160.5
gqa v             5120    1024 4.0 |    102.1    160.3
gqa o             6144    5120 3.0 |    233.9    191.9
mlp gate+up       5120   17408 3.0 |    212.4    208.3
mlp down         17408    5120 3.0 |    238.1    207.6
fused gate_up     5120   34816 3.0 |    230.4    216.7
fused qkvz        5120   16384 3.0 |    210.2    214.2
fused qkgv        5120   14336 3.0 |    206.0    214.2
head              5120  248320 4.0 |    201.7    200.9
floor 128x128      128     128 3.0 |      3.4     23.8
prefill TF/s against the whitepaper's 419 TFLOPS fp16-accumulate peak; this card's register loop: 501.9 TFLOPS at 2890 MHz
== SM cap 36 ==
matrix              in     out   K |  1K TF/s  8K TF/s
gdn qkv           5120   10240 3.0 |     62.0     51.6
gdn z             5120    6144 3.0 |     60.7     55.0
gdn out           6144    5120 3.0 |     62.3     52.3
gqa q             5120   12288 3.0 |     61.0     57.8
gqa k             5120    1024 4.0 |     53.8     51.8
gqa v             5120    1024 4.0 |     53.8     51.7
gqa o             6144    5120 3.0 |     62.3     53.8
mlp gate+up       5120   17408 3.0 |     59.2     51.7
mlp down         17408    5120 3.0 |     55.8     54.3
fused gate_up     5120   34816 3.0 |     56.5     54.6
fused qkvz        5120   16384 3.0 |     62.1     54.3
fused qkgv        5120   14336 3.0 |     62.9     57.5
head              5120  248320 4.0 |     61.5     50.7
floor 128x128      128     128 3.0 |      3.4     14.8
prefill TF/s against the whitepaper's 419 TFLOPS fp16-accumulate peak; this card's register loop: 505.3 TFLOPS at 2904 MHz
== SM cap 70 ==
matrix              in     out   K |  1K TF/s  8K TF/s
gdn qkv           5120   10240 3.0 |    111.1    102.7
gdn z             5120    6144 3.0 |    109.8    106.6
gdn out           6144    5120 3.0 |    111.1    102.9
gqa q             5120   12288 3.0 |    120.6    108.4
gqa k             5120    1024 4.0 |     83.3     91.0
gqa v             5120    1024 4.0 |     76.3     94.4
gqa o             6144    5120 3.0 |    110.6    105.4
mlp gate+up       5120   17408 3.0 |    113.6    103.0
mlp down         17408    5120 3.0 |    103.2    103.7
fused gate_up     5120   34816 3.0 |    109.9    105.8
fused qkvz        5120   16384 3.0 |    118.7    110.1
fused qkgv        5120   14336 3.0 |    119.4    109.5
head              5120  248320 4.0 |    116.0    108.0
floor 128x128      128     128 3.0 |      3.5     23.0
prefill TF/s against the whitepaper's 419 TFLOPS fp16-accumulate peak; this card's register loop: 502.1 TFLOPS at 2889 MHz
== SM cap 84 ==
matrix              in     out   K |  1K TF/s  8K TF/s
gdn qkv           5120   10240 3.0 |    136.6    124.1
gdn z             5120    6144 3.0 |    130.4    118.6
gdn out           6144    5120 3.0 |    137.0    121.6
gqa q             5120   12288 3.0 |    132.5    126.1
gqa k             5120    1024 4.0 |     90.8    106.0
gqa v             5120    1024 4.0 |     83.6    106.0
gqa o             6144    5120 3.0 |    136.6    122.5
mlp gate+up       5120   17408 3.0 |    137.0    120.3
mlp down         17408    5120 3.0 |    116.1    123.0
fused gate_up     5120   34816 3.0 |    134.8    126.3
fused qkvz        5120   16384 3.0 |    136.0    128.0
fused qkgv        5120   14336 3.0 |    140.3    124.7
head              5120  248320 4.0 |    137.6    127.9
floor 128x128      128     128 3.0 |      3.6     23.8
prefill TF/s against the whitepaper's 419 TFLOPS fp16-accumulate peak; this card's register loop: 501.1 TFLOPS at 2889 MHz
```
