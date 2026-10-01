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
`ab_run.sh`, `lanes8.sh` and `trace_phase.sh` in `.scratch/gpu-resources-2026-09-29/`.

## Question

The dflash2 context append was 21% of prefill device time. It runs A16
small-T GEMVs, 32 columns per launch, with the tensor pipe at 0%. Can a
tensor-core route that keeps the A16 numerics remove that cost, and what does
it do to TTFT, output and decode?

## Evidence

**The route.** `ops::linear` (our `kernel/src/linear.cu`) sends an NVFP4
`A16Only` call to `ignis_nvfp4_a16_mma` from **64 columns** (128 for problems
under 4,096 output rows). Narrower calls, and every `AllowA4` call, keep the
vendored dispatch.

Only the dflash2 drafter makes A16 NVFP4 calls, so only its calls change
route (`grep "ops::linear(" kernel/src`: the output heads are W8, and the
vision tower is Q4/Q5/Q6). They are:

- the context append (`feature_projection`, and the five layers'
  `query_key_value`), at prompt width in prefill, and at width x batch columns
  after each decode round;
- the propose forward (`attention_conv_proj`, `query_key_value`, `output`,
  `mlp_conv_proj`, `mlp_down`), at (draft + 1) x batch columns: 64 at eight
  lanes;
- not the selector (56 columns at eight lanes).

The kernel is a BF16 m16n8k16 MMA GEMM: 64x128x64 CTA tile, 8 warps, a
2-stage cp.async ring. Each ring slot holds the weight exactly as it is stored:
the E2M1 code rows and the 512-byte blockscale tile. Once per K step, one
thread per (row, 16-element group) decodes the tile into a swizzled BF16 tile
in shared memory. An E2M1 code times its E4M3 scale has at most 6 significant
bits (2 x 4), so each decoded element is exact in BF16. The per-tensor
`1 / weight_scale_divisor` is applied in FP32 in the epilogue. The scale
addressing restates the codec's `nvfp4_scale_offset` in 32-bit arithmetic.
Calling the 64-bit helper instead, hoisted out of the K loop or not, measured
3.5-4% slower on the latency-bound K loop.

**Correctness.** `ignis_kernel_nvfp4_a16_mma_test` runs through the vendored
linear harness: an FP64 oracle over the dequantised weight, and the vendored
A16 criterion (one BF16 unit roundoff, relative L2). It covers:

- the two drafter matrices at T = 64, 394 and 1,024;
- a full comparison of every output element on [256, 5120] and [1280, 5120] at
  T = 128, 129, 200, 256 and 300;
- the six other registered geometries, sampled;
- the route selection at its thresholds;
- which kernel `ops::linear` actually ran, bit for bit. Either side of each
  threshold its output must equal the MMA route's own output, or the vendored
  GEMVs'. The route must also differ from the GEMVs somewhere, or the arm
  could not tell them apart.

Building that last arm showed something worth knowing. With the harness's
weights and unit-normal activations, every FP32 partial sum at K = 5,120 is
exact, so the MMA route and the GEMVs agree **bit for bit** on
[6144, 5120] and [1280, 5120]. The arm therefore uses activations spread over
2^-12..2^12, where 1,280-22,400 outputs per case differ. The earlier arms
alone would still have passed with the route deleted, because the GEMVs meet
the same criterion.

A mutation that scales the epilogue by 1.01 fails it. With the route's first
threshold (33 columns), the same mutation also failed the vendored
`test_nvfp4_a16`. Disabling the route in `ops::linear` fails the dispatch
arm on all three cases above the threshold. The full kernel CTest suite passes: 67/67, as does
`cargo test --workspace` (1,916 passed).

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

**Where the threshold sits.** The route has a latency floor: one CTA walks
the whole K axis at ~1.1 us per 64-wide step, so ~90 us at K = 5,120 and
~420 us at K = 25,600, however few tokens there are.

- At T = 33 it ran 0.76-0.92x the GEMVs on every shape.
- At T = 64 it wins on every problem with 4,096+ rows: 1.38-1.50x, including
  the drafter's [5120, 4096] and [5120, 17408].
- On the narrow problems it loses at T = 64 ([1280, 5120] 0.70x,
  [256, 5120] 0.59x) and wins from 128 (1.37x, 1.05x; 10.2x and 8.2x at
  1,024).

So the threshold is 64, or 128 below 4,096 rows (`--narrow`, `--drafter` in
the bench).

**Prefill, as served** (nsys, `trace_phase.sh`, ~5,500-token cold prompts, one
lane): the CUDA-core class of prefill device time fell from **21.1% to 1.5%**,
and the tensor-core class rose from 64.4% to 80.6% (76.8% as first
reported: the class script had missed `nvfp4_a16_mma` itself; see
[the prefill "other" class](2026-09-30-prefill-other-class.md)). In one 8 s window the
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

**Eight lanes** (`lanes8.sh`: 8 concurrent greedy 384-token generations with
distinct prompts, two rounds, a fresh server per arm). This is the shape where
the drafter's propose forward runs at 64 columns and takes the route every
decode round.

| | main | main again | A16 MMA route |
|---|---|---|---|
| round 1 wall time | 6.24 s | 6.27 s | **5.93 s** |
| round 2 wall time | 5.59 s | 5.58 s | **5.42 s** |
| outputs identical to main | — | 16/16 | 15/16 |

The one different output diverges at character 1,771 of 1,828 (~token 370
of 384), inside a list of SQL terms: a near-tie resolved the other way.

On one server, main itself changes 3 of 8 outputs between its two rounds. The
server log shows no prompt reuse in either round (`prefilled_tokens` equals
`prompt_tokens`; the prompts are ~67 tokens). What differs is admission:

- in round 1 the lanes join one at a time (first token at 140 ms for the
  first lane, 1,185 ms for the last), each prompt in 2-3 chunks interleaved
  with the running lanes' decode rounds;
- in round 2 the prompts are packed into shared chunks and the lanes start
  almost together.

Different co-batching means different GEMM widths in prefill and different
lane sets in decode. So a near-tie can flip without any change to the code.

**dflash2 acceptance**, from the `spec.*` counters of the `request.done` log
events of the same runs:

| | main | A16 MMA route |
|---|---|---|
| 8 lanes, 16 requests: accepted / drafted | 3,674 / 17,148 = 0.214 | 3,679 / 17,118 = 0.215 |
| 8 lanes: tokens per round | 2.487 | 2.492 |
| 1 lane, 2 requests: accepted / drafted | 606 / 2,906 = 0.209 | 602 / 2,934 = 0.205 |

## Finding

**Observed.** A16 NVFP4 at prompt width now runs on the tensor cores at
93-96% of the BF16 peak. The drafter's context append no longer shows up as a
cost class in prefill. Single-lane TTFT on a ~5K-token prompt dropped 12.8%.
At one lane, greedy output is byte-identical and decode speed unchanged. At
eight lanes, a round of 8 generations is 3-5% faster, and 15 of 16 outputs
match main. dflash2 acceptance is unchanged: 0.214 against 0.215 at eight
lanes, and 0.209 against 0.205 at one lane (two requests).

The route keeps the A16 contract, not the GEMVs' bits: where a sum rounds,
the different accumulation order can round it to a different BF16.

**Inferred.** The TTFT gain is smaller than the ~18% of device time removed:
end-to-end time also includes host time between requests and the gaps between
kernels inside a chunk, which the route does not touch. Output is identical
because greedy verification makes the target model alone decide each token. A
drafter context that differs in accumulation order can change which drafts are
accepted, but at one lane not what is emitted. At eight lanes, a different
acceptance changes the width of the target's batched rounds, and batched
decode is not width-invariant
([batched decode width drift](2026-09-14-batched-decode-width-drift.md)). A
late near-tie can therefore flip, which fits the single divergence. Main is
16/16 against itself only because a fresh server replays the same admission
schedule. Its 3/8 change between two rounds on one server says the same
near-ties already flip in production whenever requests arrive differently.
That co-batching, not reuse, is the cause is read from the logs. No
experiment isolated it.

## Implications

- The remaining prefill cost is the backbone on the tensor cores. The next
  prefill lever is the backbone GEMMs (already 71-87% tensor), attention, or
  the ~11% "other" class, not the drafter. Past the kernels, the retained
  host slot capture stalls ~12% of a cold TTFT
  ([the prefill "other" class](2026-09-30-prefill-other-class.md)).
- A future A16 NVFP4 caller gets the route automatically, because it is taken
  inside `ops::linear`. Today the only such caller is the drafter.

## Limits and unknowns

- Below the thresholds the GEMVs stay. That covers the drafter at one to
  seven lanes (8-56 columns) and the narrow problems at 64-127. The cause is
  the route's latency floor: the K loop is serial within a CTA. A split-K, or
  a deeper pipeline with the decode overlapped (the ring runs three
  `__syncthreads` per K step), would lower it. Not built.
- One prompt length (~5.1K tokens). Longer prompts spend more of their time in
  attention, so the relative gain shrinks with length.
- Acceptance was counted on 16 eight-lane and 2 one-lane generations of short
  essay prompts, not on agent traces
  ([sampled acceptance](2026-09-24-dflash2-sampled-acceptance.md) measures
  those).

## Follow-ups

- Lower the latency floor (split-K, or a pipelined decode), then lower the
  thresholds. No ticket yet: the owner decides whether to open one.
