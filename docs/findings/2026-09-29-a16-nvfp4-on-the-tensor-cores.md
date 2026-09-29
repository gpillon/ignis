# The drafter's context append on the tensor cores: an A16 NVFP4 MMA route, 6-7x per call, TTFT −12.8%

- Kind: experiment
- Status: current
- Observed: 2026-09-29
- Last verified: 2026-09-29
- Scope: kernel / `ops::linear` NVFP4 A16 route; dflash2 drafter context append; prefill TTFT
- Related: [GPU resources: prefill vs decode](2026-09-29-gpu-resources-prefill-vs-decode.md),
  [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [Drafter top-k](2026-09-18-dflash2-topk-one-warp-per-column.md)
- Superseded by: none

**Hardware:** RTX 5090, exclusive card (ADR 0006).
**Code:** branch `nvfp4-a16-mma`, which adds `kernel/src/nvfp4_a16_mma.cu`,
`kernel/include/ignis_nvfp4_a16_mma.h` and the route in `kernel/src/linear.cu`.
**Harness:** `kernel/tests/bench_nvfp4_a16_mma.cpp` (tool, not CTest), plus
`ab_run.sh` and `trace_phase.sh` in `.scratch/gpu-resources-2026-09-29/`.

## Question

The dflash2 context append was 21% of prefill device time. It runs A16
small-T GEMVs, 32 columns per launch, with the tensor pipe at 0%. Can a
tensor-core route that keeps the A16 numerics remove that cost, and what does
it do to TTFT, output and decode?

## Evidence

**The route.** `ops::linear` (our `kernel/src/linear.cu`) sends an NVFP4
`A16Only` call of **64 columns or more** to `ignis_nvfp4_a16_mma`. Narrower
calls, and every `AllowA4` call, keep the vendored dispatch.

The kernel is a BF16 m16n8k16 MMA GEMM: 64x128x64 CTA tile, 8 warps, a
2-stage cp.async ring. Each ring slot holds the weight exactly as it is stored:
the E2M1 code rows and the 512-byte blockscale tile. Once per K step, one
thread per (row, 16-element group) decodes the tile into a swizzled BF16 tile
in shared memory. An E2M1 code times its E4M3 scale has at most 5 significant
bits, so each decoded element is exact in BF16. The per-tensor
`1 / weight_scale_divisor` is applied in FP32 in the epilogue.

**Correctness.** `ignis_kernel_nvfp4_a16_mma_test` runs through the vendored
linear harness: an FP64 oracle over the dequantised weight, and the vendored
A16 criterion (one BF16 unit roundoff, relative L2). It covers:

- the two drafter matrices at T = 64, 394 and 1,024;
- a full comparison of every output element on [256, 5120] and [1280, 5120] at
  T = 64, 127, 128, 129 and 300;
- the six other registered geometries, sampled.

A mutation that scales the epilogue by 1.01 fails it, and fails the vendored
`test_nvfp4_a16`. The full kernel CTest suite passes: 67/67.

**Per call** (`ignis_nvfp4_a16_mma_bench`, median of 50, same weight and
input):

| problem | T | vendored GEMV slices | MMA route | speedup |
|---|---|---|---|---|
| fc [5120, 25600] | 1,024 | 10,196 us (26 TF/s) | 1,378 us (195 TF/s) | **7.40x** |
| fc [5120, 25600] | 394 | 3,855 us | 718 us | 5.37x |
| fc [5120, 25600] | 64 | 583 us | 421 us | 1.38x |
| qkv [6144, 5120] | 1,024 | 2,243 us | 354 us (182 TF/s) | **6.34x** |
| qkv [6144, 5120] | 64 | 136 us | 94 us | 1.44x |
| [5120, 17408] | 1,024 | 6,869 us | 941 us | 7.30x |
| [34816, 5120] | 1,024 | 13,260 us | 1,812 us (202 TF/s) | 7.32x |

195-202 TF/s is about 93-96% of the card's BF16 peak with FP32 accumulation
(about 209 TF/s on the RTX 5090, per NVIDIA's spec), so at prompt width the A16 route has no
headroom left.

**Where the threshold sits.** At T = 33 the route was 0.76-0.92x the GEMVs.
With a 5,120-row matrix the grid is only 80 CTAs on 170 SMs, and each CTA
walks the whole K axis. At T = 64 it wins on every shape measured, so the
route starts at 64.

**Prefill, as served** (nsys, `trace_phase.sh`, ~5,500-token cold prompts, one
lane): the CUDA-core class of prefill device time fell from **21.1% to 1.5%**,
and the tensor-core class rose from 64.4% to 76.8%. In one 8 s window the
drafter projections went from 1,197 ms of `nvfp4_small_t` to 201 ms of
`nvfp4_a16_mma` plus 60 ms of remaining GEMV tails. The new kernel runs at 67%
Tensor Active in the samples.

**End to end** (`ab_run.sh`, production flags, the same prompts and flags on
`main` fb144ac and on the branch; 12 cold prompts of ~5,130 tokens after 2
warm-ups, then two greedy 512-token generations):

| | main | A16 MMA route |
|---|---|---|
| prefill wall time, median | 737 ms | **643 ms (−12.8%)** |
| prefill throughput | 6,959 tok/s | 7,983 tok/s |
| 512-token generation | 3.82 s, 3.62 s | 3.79 s, 3.65 s |
| generated text | — | **byte-identical** to main (both runs) |

## Finding

**Observed.** A16 NVFP4 at prompt width now runs on the tensor cores at
93-96% of the BF16 peak. The drafter's context append no longer shows up as a
cost class in prefill. Single-lane TTFT on a ~5K-token prompt dropped 12.8%.
Greedy output and decode speed are unchanged.

**Inferred.** The TTFT gain is smaller than the ~18% of device time removed:
end-to-end time also includes host time between requests and the gaps between
kernels inside a chunk, which the route does not touch. Output is identical
because greedy verification makes the target model alone decide each token. A
drafter context that differs in accumulation order can change which drafts are
accepted, but not what is emitted. Equal generation times say acceptance did
not move measurably. Acceptance itself was not counted.

## Implications

- The remaining prefill cost is the backbone on the tensor cores. The next
  prefill lever is the backbone GEMMs (already 71-87% tensor), attention, or
  the ~15% "other" class, not the drafter.
- Any other A16 NVFP4 caller at 64 or more columns now gets the MMA route
  automatically, because it is taken inside `ops::linear`.

## Limits and unknowns

- 33-63 columns stay on the GEMVs. The drafter's decode-time append (width x
  batch columns, up to 64 at eight lanes) mostly lives there. A split-K or a
  smaller CTA tile could win that range too. Not built.
- One lane, one prompt length (~5.1K tokens). Longer prompts spend more of
  their time in attention, so the relative gain shrinks with length.
- dflash2 acceptance was not measured directly, only through generation time
  on one prompt.

## Follow-ups

- Split-K (or a narrower tile) for 33-63 columns, then lower the threshold.
