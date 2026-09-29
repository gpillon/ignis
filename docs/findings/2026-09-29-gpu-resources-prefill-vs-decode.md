# Which GPU units prefill and decode use: tensor cores in prefill, DRAM in decode, and a fifth of prefill on CUDA cores for the drafter

- Kind: experiment
- Status: current
- Observed: 2026-09-29
- Last verified: 2026-09-29
- Scope: kernel / prefill and decode rounds, GPU unit utilisation; dflash2 drafter context append
- Related: [Decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [Decode round host idle](2026-09-18-decode-round-host-idle.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)
- Superseded by: none

**Hardware:** RTX 5090 (GB202, 170 SMs), exclusive card (ADR 0006), clocks not locked.
**Engine:** `main` at fb144ac, release build, the `make config` production flags
(`hq-e8-2b`, `--prefill-chunk 1024`, `--spec dflash2 --draft-tokens 7`).
**Harness, tables and raw reports:** `.scratch/gpu-resources-2026-09-29/`.

## Question

Which GPU units (tensor cores, the FP32/INT pipes of the CUDA cores, the
memory system) do prefill and decode keep busy, how busy, and which kernels
account for it?

## Evidence

Two instruments. Each one covers something the other cannot.

**1. Phase level: Nsight Systems GPU Metrics sampling.** `trace_phase.sh`
runs the server under `nsys profile --trace=cuda --cuda-graph-trace=node
--gpu-metrics-devices=0 --gpu-metrics-set=gb20x --gpu-metrics-frequency=20000`,
takes an 8 s window, and drives one lane:

- **prefill:** ~5,500-token prompts, `max_tokens: 1`, with a marker unique to
  each request so that prompt reuse cannot remove the prefill.
- **decode:** one 3,000-token generation, back to back.

`phase_report.py` attributes each 50 us sample to the kernel running at its
midpoint. `classes.py` splits device time by the unit each kernel saturates.

Mean over the samples where a kernel is running ("busy"), as % of each unit's
peak:

| phase | device busy | SMs active | SM issue | **Tensor active** | warps in flight | **DRAM read** | DRAM write |
|---|---|---|---|---|---|---|---|
| prefill | 72.7% | 81.2 | 22.1 | **37.3** | 21.0 | 17.5 | 8.6 |
| decode (1 lane) | 87.0% | 67.3 | 12.5 | **4.6** | 20.5 | **66.7** | 2.1 |

A second capture of each phase, taken earlier while other GPU work may have
been running, agreed to within 1 point. Prefill's idle 27% is mostly host
time between single-lane requests (HTTP, tokenisation), with the gaps between
kernels inside a chunk making up the rest.

Device time by the unit that bounds each kernel:

| class | prefill | decode |
|---|---|---|
| tensor cores (GEMM / attention at M >= 1024) | **64.4%** | 1.3% |
| CUDA-core FMA/ALU (A16 GEMV, tensor 0%) | **21.1%** | 14.4% |
| DRAM (weight streaming, tensor < 20%) | 5.3% | **67.8%** |
| other (norms, quantize, recurrent, small attention) | 9.2% | 16.6% |

**2. Kernel level: Nsight Compute.** `ncu_phase.sh` profiles each distinct
launch shape of the main kernels twice, after skipping 10 warm-up launches
(`--filter-mode per-launch-config`), with 15 metrics. Selected rows, as % of
peak (full tables: `ncu-prefill-table.txt`, `ncu-decode-table.txt`):

| phase | kernel (N x K, grid) | us | GHz | tensor | FMA | ALU | issue | DRAM |
|---|---|---|---|---|---|---|---|---|
| prefill | `bf16_gemm_mma` 14336x5120 | 781 | 2.72 | **86.9** | 1.7 | 7.2 | 18.3 | 13.0 |
| prefill | `nvfp4_w4a4_tma` 16384x5120 | 160 | 2.43 | **78.8** | 1.1 | 3.3 | 14.6 | 18.9 |
| prefill | `nvfp4_linear_swiglu_w4a4_tma` 34816x5120 | 332 | 2.44 | **71.1** | 1.7 | 3.4 | 14.9 | 22.3 |
| prefill | `gqa_attention_prefill_bf16` (grid 312) | 705 | 2.85 | **77.5** | 4.1 | 3.6 | 13.0 | 2.9 |
| prefill | `nvfp4_small_t` 5120x25600 (drafter fc) | 319 | 2.57 | **0.0** | **36.1** | 25.1 | **60.7** | 13.6 |
| prefill | `nvfp4_small_t` 6144x5120 (drafter qkv) | 68 | 2.77 | **0.0** | **33.7** | 23.4 | **58.5** | 14.9 |
| prefill/decode | `w8_small_t_mma` 248320x5120 (output head) | 806 | 2.89 | 10.8 | 5.0 | 7.0 | 13.7 | **95.5** |
| decode | `nvfp4_w4a4_mma` 34816x5120 (grid 136) | 68 | 2.90 | 5.7 | 1.1 | 5.0 | 9.3 | **89.4** |
| decode | `nvfp4_w4a4_mma` 16384x5120 (grid 128) | 32 | 2.90 | 6.2 | 0.7 | 2.9 | 7.6 | **83.4** |
| decode | `nvfp4_w4a4_mma` 5120x17408 (grid 80) | 42 | 2.91 | 17.0 | 1.0 | 5.2 | 17.0 | 71.0 |
| decode | `nvfp4_small_t` 5120x17408 (drafter) | 83 | 2.61 | 0.0 | 42.8 | 30.1 | 73.3 | 35.8 |

In the prefill capture, `nvfp4_small_t` launches arrive as runs of up to 32
back to back after the last layer of each 1,024-token chunk: 301 runs, 1,034 ms
of an 8 s window. The fc template is
`Nvfp4GemvGeometry<5120, 25600>` with 32 active tokens, i.e. the drafter's
`dflash2/feature_projection` (`SH_DF_FEATURES`, `crates/artifact/src/inventory.rs:228`).

## Finding

**Observed.**

- **Prefill runs on the tensor cores.** The GEMMs and prefill attention keep
  the tensor pipe at 71-87% of peak, with DRAM at 13-22% and issue slots at
  13-18%. Tensor-bound kernels are 64% of prefill device time. They also run
  at 2.41-2.51 GHz, against 2.9 GHz for the memory-bound kernels: under full
  tensor load the card clocks down.
- **Decode runs on DRAM bandwidth.** The NVFP4 backbone GEMMs of a verify
  round move weights at 83-89% of DRAM peak on the large shapes, with the
  tensor pipe at 6-17% and issue at 7-17%. The INT8 output head sits at 95.5%
  DRAM. This matches the [2026-09-18 anatomy](2026-09-18-decode-round-anatomy.md):
  there is no compute headroom to take in decode.
- **About a fifth of prefill device time runs on CUDA cores alone.** The dflash2
  context append (`ignis_dflash2_append_context`,
  `kernel/src/dflash2_drafter.cu:88` and `:97`) projects every prompt token
  through `feature_projection` and each drafter layer's `query_key_value`. It
  calls the A16-only `ninfer::ops::linear(x, w, out, stream)` overload, which
  sends an NVFP4 weight to the small-T GEMV (`nvfp4_small_t_kernel`) in
  32-column slices whatever T is. These kernels issue at 58-61% on the FMA/ALU
  pipes with the tensor pipe at 0%. Together they are 21% of prefill device
  time: the fc alone is ~10 ms per 1,024-token chunk.

**Inferred.** A tensor-core route for those two projections at T = 1,024
should cost what the backbone's equivalent shapes cost. The `w4a4_tma` kernel
takes 159 us for 5120x17408 and 63 us for 5120x6144. Today the fc costs ~10 ms
per chunk, and the qkv ~2.2 ms per chunk *per drafter layer*
(32 slices x 68 us). So roughly a fifth of prefill device time, and of
single-lane TTFT on long prompts, is recoverable. Not measured.

## Implications

- Prefill optimisation should look at the drafter context append before it
  looks at any backbone GEMM. The backbone already sits at 70-87% of the tensor
  pipe.
- A policy change will not do it. The vendored NVFP4 registry declares the five
  DFlash2 geometries "A16 weight-only; no activation-quant sites"
  (`kernel/vendor/src/ops/linear/nvfp4/nvfp4_config.h:113`), and
  `nvfp4_dispatch.cpp:40-46` throws on any policy other than A16 for them. So
  there is no tensor-core route for these shapes today. Three ways to get one,
  cheapest first:
  - Dequantise `feature_projection` and the drafter `query_key_value` weights
    to BF16 once, at load, and run the prefill-width append through the
    existing `bf16_gemm_mma` kernel. FP4 to BF16 is exact, so the numerics stay
    A16 (only the accumulation order changes). It costs VRAM: ~262 MB for the
    fc, plus 63 MB per drafter layer for the qkv.
  - A new A16 big-T NVFP4 kernel: dequantise weights into shared memory, then a
    BF16 MMA. The same A16 contract, no VRAM cost, but it is a kernel of our
    own beside a vendored route (ADR 0031).
  - Register W4A4 instances for the two geometries. That changes the numerics
    the drafter sees, so dflash2 acceptance has to be re-measured.
- Because the vendored ops copy the reference's design, ninfer probably pays
  the same cost in its prefill. That should be checked against the reference
  before claiming it; if it holds, this is a place ignis can pull ahead.
- Decode stays where the 2026-09-18 anatomy left it: bound by memory, not by
  compute.

## Limits and unknowns

- One lane only. Batched decode (8 lanes) widens the GEMMs and should move
  decode toward the tensor cores. Not measured here.
- ncu covered the dominant launch shapes, not every kernel. In decode the
  output head, `nvfp4_linear_swiglu_small_t`, `bf16_small_t_inner` and
  `recurrent_fold` never passed the 10-launch skip inside the window. Their
  profile comes from the GPU Metrics samples only; the head's ncu row is from
  the prefill run, same grid.
- The ncu runs used a smaller server footprint (`--max-context 131072`,
  `--vram-budget-bytes 24G`, no host KV pool, no retained host slots), because
  kernel replay must back up device-mapped host memory. With the production
  pools it filled C: through a backup file. Kernel shapes are unchanged by this.
- GPU Metrics samples are 50 us. Kernels shorter than a sample are folded into
  a "boundary" group, so per-kernel sampled rows are biased toward the long
  launches. The ncu rows are the per-kernel authority.
- The prefill capture with `max_tokens: 1` still shows short decode-shaped
  clusters (~12 ms, small-T attention and the head) around each request.
  Whether a verify round runs after a one-token prefill was not investigated.
- The GPC clock metric exported as an unusable integer. The clocks quoted come
  from ncu's `gpc__cycles_elapsed.avg.per_second`.

## Follow-ups

- Done the same day, second route (our A16 big-T kernel):
  [The drafter's context append on the tensor cores](2026-09-29-a16-nvfp4-on-the-tensor-cores.md).
  The CUDA-core share of prefill fell from 21% to 1.5%, and TTFT from 737 to
  643 ms.
- Optional: the same study with 8 decode lanes, the production shape.
