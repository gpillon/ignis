# #92 criterion 1 — where the per-chunk prefill wall time goes

Measured 2026-09-11 on a free RTX 5090 (ADR 0006 profile, preflight on
record), warm, live/live. Model `qwen3_8_27b_nvfp4full-v2.ninfer`, 8,192-token
span, default chunked route (`run_program_chunk`, `kernel/src/step.cu`),
64 layers.

This file answers **acceptance criterion 1 only**. The survey of directions
(criterion 2), the CUDA-graph spike and any fix are deliberately not here.

> Consolidated into `docs/findings/2026-09-11-prefill-chunk-wall-time.md`,
> which is the durable record. This file is the working writeup the #92
> comments link to, kept as taken.

## The answer, first

At the production 1,024-token chunk width, a chunk costs **98.6 ms of device
timeline**, and it splits:

| bucket | ms/chunk | share |
| --- | ---: | ---: |
| (c) per-layer compute — kernels actually executing | 88.9 | **90.2 %** |
| (a) kernel launch / dispatch — device idle between kernels | 9.2 | 9.3 % |
| (b) the forced `cudaStreamSynchronize` — device idle at the chunk boundary | 0.47 | **0.5 %** |

The "8 semaphores" cost **3.8 ms of a ~790 ms 8K prefill**. The premise this
ticket was opened on — that the synchronous stops dominate per-chunk wall
time — does not survive measurement. The route is compute-bound.

Why #86 changed nothing is therefore not "the cost is in the semaphores". The
tensor-core entry points landed, and the per-chunk number did not move,
because whatever they saved was inside the 90 % that is already compute and
was not large relative to it. The remaining 10 % is launch latency spread
across ~1,160 kernels per chunk, not the 1 synchronization per chunk.

## How it was measured

Three independent measurements, deliberately overlapping, so no single
instrument has to be believed on its own.

### 1. Chunk-width sweep, no instrumentation at all

`crates/core/tests/chunk_decomposition_gpu.rs`, pass A. The same 8,192-token
span, prefilled at six chunk widths. Wider chunks mean fewer forced
synchronizations for identical total token work: at 8,192 the span is one
chunk and one synchronization, against the default's eight.

| width | chunks | wall mean (ms) | wall min (ms) | ms/chunk | ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 1316.4 | 1304.9 | 41.14 | 0.1607 |
| 512 | 16 | 942.2 | 938.8 | 58.89 | 0.1150 |
| **1024** | **8** | **797.8** | **785.2** | **99.72** | **0.0974** |
| 2048 | 4 | 775.7 | 768.7 | 193.93 | 0.0947 |
| 4096 | 2 | 767.0 | 764.8 | 383.50 | 0.0936 |
| 8192 | 1 | 790.2 | 768.9 | 790.20 | 0.0965 |

Collapsing eight synchronizations into one buys **16.3 ms on a 785.2 ms span
(2.1 %)**, best-of-three against best-of-three.

A naive least-squares fit over all six widths reports a 17.7 ms fixed cost per
chunk boundary, which would be 141 ms of the default route's 790 ms. **That fit
is wrong, and the reason matters**: at 256 and 512 tokens the chunk is too
narrow for the GEMM shapes, so per-token *device compute itself* rises (0.158
against 0.093 ms/token, measured directly in section 2), and the regression
charges that lost compute efficiency to the chunk boundary. Fit only the
widths whose per-token cost has plateaued (>= 1,024) and the slope is
**2.66 ms per boundary**, of which sections 2 and 3 show 0.47 ms is the
synchronization and the rest is dispatch.

This is the trap the #110 comment on this ticket walked into from the other
side: chunk 512 having a worse p95 tail than chunk 1024 is consistent with a
fixed per-boundary cost, but it is equally consistent with narrow chunks
simply computing less efficiently — which is what is actually happening.

### 2. The leaf's own CUDA-event decomposition

Pass B, same test, with `IGNIS_CHUNK_PROFILE` set. `run_program_chunk` records
events on the model's own stream around the chunk, around each of the 64 layer
bodies, and around the head, and emits one JSONL record per chunk after the
existing synchronize (`chunks.jsonl`). Per-chunk means, 3 timed spans per
width, warm-up span dropped:

| width | chunk_ms | enqueue | sync_stall | entry_gap | embed | layers | head | layer_gap | gpu_span | layers/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 40.66 | 22.73 | 17.93 | 0.22 | 0.05 | 40.40 | 0.04 | 0.11 | 40.60 | 0.1578 |
| 512 | 59.69 | 32.32 | 27.36 | 0.43 | 0.06 | 59.21 | 0.08 | 0.11 | 59.45 | 0.1156 |
| **1024** | **97.99** | **57.00** | **40.99** | **0.47** | **0.05** | **97.29** | **0.14** | **0.10** | **97.58** | **0.0950** |
| 2048 | 191.60 | 122.84 | 68.76 | 0.46 | 0.09 | 190.68 | 0.28 | 0.10 | 191.15 | 0.0931 |
| 4096 | 381.23 | 292.88 | 88.35 | 0.25 | 0.11 | 379.95 | 0.63 | 0.10 | 380.79 | 0.0928 |
| 8192 | 763.79 | 763.45 | 0.33 | 0.00 | 0.20 | 761.71 | 1.35 | 0.10 | 763.36 | 0.0930 |

- `entry_gap` is the measurement the ticket actually asked for: device idle
  between the **previous** chunk's last operation and this chunk's first. That
  is the bubble the forced synchronization opens, and it is **0.47 ms**.
- `sync_stall` (41.0 ms) is *not* that bubble. It is host wall spent blocked
  inside `cudaStreamSynchronize` while the device is busy computing. Reading
  it as overhead is the mistake this ticket was built on: the host being
  blocked and the device being idle are different things, and only the second
  costs anything.
- `enqueue` (57.0 ms) is host wall issuing the chunk's launches, overlapping
  device work. At width 8,192 it absorbs the whole span (763.5 ms) with a
  0.33 ms stall — the host throttled by a full launch queue, which is what a
  saturated device looks like from the host side, not host-bound dispatch.
- `layer_gap` (0.10 ms) is device idle *between* layer bodies: essentially
  nothing. The host is never late with the next layer.
- `layers/token` is flat from 1,024 up and rises 66 % at 256. That column, not
  the synchronization count, is what makes narrow chunks slow.

Instrumentation cost is inside run-to-run noise: the same sweep read 785.3 ms
at width 1,024 with the events on against 797.8 ms with them off.

### 3. Nsight Systems, as the independent check

The event decomposition cannot see idle *inside* a layer body — gaps between a
layer's ~18 kernels land in `layers`. Nsight Systems can, so a width-1,024-only
run was captured (`--trace=cuda`) and the kernel timeline read directly
(`nsys_gap_report.py`):

```
prefill region: 37133 kernels, 3154.2 ms wall      (4 spans x 8 chunks)
  device busy :    2844.0 ms  (90.17%)
  device idle :     310.1 ms  ( 9.83%)
  overlapping kernel pairs: 0 (one stream, as expected)
  mean gap between consecutive kernels: 8.4 us over 37132 gaps
  idle in gaps <= 100 us:  185.4 ms over 36790 gaps (mean 5.0 us)  -- launch latency
  idle in gaps >  100 us:  124.7 ms over   342 gaps               -- scheduling hiccups
```

**90.2 % of the prefill's device timeline is a kernel executing.** That is the
(c) bucket, measured end to end with no reliance on the leaf's own events. The
9.8 % idle divides into ~5 us of launch latency between consecutive kernels
(Windows WDDM submission, ~1,160 kernels per chunk) and 342 wider gaps
scattered across many different kernel types — no single structural host
round-trip, and only ~11 per chunk.

The capture's own wall (98.6 ms/chunk) matches the uninstrumented sweep
(98.2-99.7 ms/chunk), so tracing is not inflating the idle it reports.

Host-side, the same capture: `cudaStreamSynchronize` 1384.8 ms over 33 calls,
44 % of prefill wall. Again — host blocked, device busy. `cudaLaunchKernel`
plus `cudaLaunchKernelExC` total 1158.5 ms of host time against 3154.2 ms of
wall, comfortably hidden.

## What this settles, and what it does not

Settled: the per-chunk cost is **compute-bound**. Dispatch is 9 %, the forced
synchronization is 0.5 %, per-layer math is 90 %.

Not touched here, on purpose: whether the 90 % itself can be made smaller
(kernel work, not this ticket), and the criterion-2 survey.

One constraint to carry forward for whoever writes criterion 2: that
synchronization is today the **only** failure-detection point for the P2-02 /
#84 retry contract — the caller may not advance `seq`'s position state until
it confirms the chunk's device work completed. Removing or deferring it is not
a one-line deletion; the detection has to land somewhere else (a recorded
event queried before the advance, or the advance itself deferred to the span's
end). At 0.5 % of wall, that is a large contract change for a small number.
The #110 probe already measured the serving-level version of this: skipping
the non-final-chunk sync moved the ITL p95 ratio 1.130 to 1.110.

## Addendum — "busy" is not "saturated", and packing is a separate question

Raised after the first reading: 90.2 % of the device timeline being inside a
kernel says the device is not *idle*. It says nothing about whether that
kernel is using the SMs and tensor cores well. A GEMM that occupies a quarter
of the machine still counts as 100 % busy on the timeline. So "compute-bound"
above must be read narrowly: **the wall time is spent in kernels, not in
synchronization or dispatch.** It is not a claim that the math is at peak.

Nsight Compute would answer the saturation question directly, and it refuses
here: `ERR_NVGPUCTRPERM`, hardware performance counters are not readable
without the driver's "Manage GPU Performance Counters" permission opened to
all users (or an elevated session). That measurement is still outstanding.

What can be measured without counters is the thing the question is really
about: **does a traversal carrying more tokens cost less per token?** Packing
N requests into one traversal gives every projection and FFN GEMM the shapes
of a single N*L-token traversal, and those GEMMs are ~75 % of kernel time. So
sweep the token count of an isolated traversal.

This is *not* the width sweep in section 1. There, a 256-token chunk sits
inside an 8,192-token span and still attends to up to 8K of KV prefix, so its
per-token cost carries attention work a standalone 256-token request would
never do. Here each span is prefilled on its own, from position zero, in
exactly one chunk of its own width
(the second sweep of `prefill_chunk_and_traversal_sweeps`):

| tokens in the traversal | wall min (ms) | ms/token | vs 256 |
| ---: | ---: | ---: | ---: |
| 256 | 39.3 | 0.1534 | 1.00x |
| 512 | 55.1 | 0.1077 | 1.42x |
| **1024** | **88.6** | **0.0865** | **1.77x** |
| 2048 | 181.3 | 0.0885 | 1.73x |
| 4096 | 370.0 | 0.0903 | 1.70x |
| 8192 | 784.8 | 0.0958 | 1.60x |

Per-token cost bottoms out at **~1,024 tokens per traversal**. Below it the
traversal is shape-starved and packing pays: four 256-token prompts in one
traversal is **1.77x** cheaper per token than four separate ones. Above it the
curve turns back up, but that rise is attention, not saturation — one
8,192-token span attends to a longer average prefix than eight independent
1,024-token spans would, so this table is a conservative floor for packing at
the wide end, not a ceiling.

Splitting that 1.77x by mechanism, using the ~9.2 ms of per-traversal launch
idle measured in section 3 (roughly constant, since a traversal is ~1,160
kernels regardless of its token count):

| | 256-token traversal | 1,024-token traversal |
| --- | ---: | ---: |
| wall | 39.3 ms | 88.6 ms |
| launch idle (approximately fixed per traversal) | ~9.2 ms (23 %) | ~9.2 ms (10 %) |
| kernels executing, per token | ~0.1176 ms | ~0.0775 ms |

So roughly **1.5x of the win is GEMM shape** and **1.17x is dispatch
amortization**. The dispatch part is the one the ticket's premise predicted,
and it is the smaller half — and note it is amortizing the *launch* overhead
(9 %), not the synchronization (0.5 %).

### What this means for packed prefill (`DEFERRED-DECISIONS.md` item 1)

The conclusion there needs restating, because the reason given for it is wrong
and the answer still comes out "worth building", for a different reason and
under a condition:

- **Wrong reason:** "per-chunk cost is dominated by synchronization and
  dispatch, so packing amortizes it across N". Synchronization is 0.5 % and
  dispatch 9 %. There is no dominant overhead to amortize.
- **Right reason, conditional on prompt length:** below ~1,024 tokens per
  traversal the model is shape-starved, and packing is what fills the
  traversal. At 256-token prompts that is a 1.77x prefill win; at 1,024-token
  prompts and above it is nothing.

So the phase question turns on the workload, not on this ticket's overhead
numbers. The gates measure 8K prompts, where one request already fills a
traversal and packing is worthless. G4's subagent-burst trace is many short
prompts, which is exactly the regime where the table above pays — which is
where that item already proposed to measure it.

## Second addendum — the hardware counters, and a VRAM audit

Two things the first addendum left open: the saturation question it could not
answer (`ERR_NVGPUCTRPERM`), and whether the sweeps were spilling the card
into system memory and inflating their own numbers.

### The counters (permission opened, so these are real)

`ncu` over the dominant NVFP4 projections, one traversal width per run,
`launch__grid_size` read against this card's 170 SMs (`ncu_summary.py`):

**T = 256 tokens** — the engine takes the `mma` route here:

| kernel | grid | waves | SM % | tensor pipe % |
| --- | ---: | ---: | ---: | ---: |
| `nvfp4_w4a4_mma<5120,17408>` (MLP down) | 80 | **0.47** | 21.0 | 45.7 |
| `nvfp4_w4a4_mma<5120,6144>` (GDN out) | 80 | **0.47** | 19.0 | 42.1 |
| `nvfp4_w4a4_mma<16384,5120>` (GDN qkvz) | 256 | 1.51 | 42.6 | 52.9 |
| `nvfp4_w4a4_mma<34816,5120>` (MLP gate/up) | 544 | 3.20 | 45.3 | 54.7 |

**T = 1024 tokens** — a *different* kernel runs; the `mma` variant is not
launched at all at this width:

| kernel | grid | waves | SM % | tensor pipe % |
| --- | ---: | ---: | ---: | ---: |
| `nvfp4_linear_swiglu_w4a4_tma` | 1088 | 6.40 | 50.4 | 61.7 |
| `nvfp4_w4a4_tma<5120,17408>` | 160 | **0.94** | 65.0 | 75.1 |
| `nvfp4_w4a4_tma<5120,6144>` | 160 | **0.94** | 57.7 | 66.0 |
| `nvfp4_w4a4_tma<16384,5120>` | 512 | 3.01 | 61.6 | 73.5 |

So the objection was right, and now it has a number. At 256 tokens two of the
four dominant GEMMs launch **80 blocks onto a 170-SM card** — under half the
machine has any work at all — and run at ~20 % SM throughput. Filling the
traversal fixes both halves at once: the grids double, *and* the engine
crosses into the TMA route, taking SM throughput to 58-65 % and the tensor
pipe to 66-75 %.

That the route switches with token count is worth recording on its own. The
1.77x in the first addendum is not one effect but two, and neither of them is
the "8 semaphores".

Nothing is saturated even at 1,024 tokens: `<5120,17408>` is still at 0.94
waves, so ~10 SMs sit idle through it, and the tensor pipe tops out at 75 %.
But the wall clock says pushing past 1,024 tokens does not convert that
headroom into speed — the limiter above that width is inside the kernels, not
in how many tokens the traversal carries.

### VRAM audit

The sweeps now print the leaf's own reservation (`ignis_program_stats`:
weights + prefill scratch arena + KV and GDN pools + sampling and
decode-graph buffers) per row:

| tokens per traversal | leaf VRAM (GiB) |
| ---: | ---: |
| 256 | 17.42 |
| 512 | 17.50 |
| 1024 | 17.66 |
| 2048 | 17.98 |
| 4096 | 18.61 |
| 8192 | 19.88 |

The baseline is flat and the growth is the prefill scratch arena, which
`ignis_model_load` sizes for the traversal width (P2-01, GitHub #83): about
2.5 GiB from the narrowest row to the widest.

The KV pool is not the driver. This budget is 258 page groups of 64 tokens
over 16 GQA layers at 4 KV heads x 256 head dim, K and V, BF16:
258 * 64 * 16 * 4 * 256 * 2 * 2 = **1.01 GiB**, plus 2 slots of GDN state
(48 layers x 48 heads x 128x128) at 151 MiB. About 1.2 GiB of a 17.4 GiB
baseline, and identical in every row — it cannot explain a difference between
rows, and it is not close to the card's limit.

Sampled during an `ncu` run at T=1024 (`\GPU Process Memory(*)` counters):
**dedicated peak 21,367 MiB, shared peak 195 MiB** of a 32,607 MiB card. No
spill into system memory was reproduced, so the wide rows stand as measured.

Two cautions that came out of the audit:

- **The preflight cannot tell a quiet card from a slow one, and this bit.**
  Re-running hours later on the same machine, the repo's own pre-existing
  diagnostic (`chunk_timing_diagnostic_gpu`, historically 94.9 ms/chunk)
  reported **122.8 ms/chunk**, and every row of both sweeps moved by the same
  ~25-30 %. Nothing about the engine had changed: profiling on against off
  measured 1008.0 against 991.7 ms on the same shape, and the control test is
  one nobody here touched. `nvidia-smi` showed the card idle at 5 % with
  2,969 MiB resident, so `scripts/gpu-preflight.ps1` passed it -- it checks
  free memory and the absence of ninfer, not whether the card will clock up.
  Under load the SM clock sat at 1,980 MHz against a 3,090 MHz maximum, at
  360 W of a 575 W limit, with `SW Power Capping` showing 219 ms accumulated.
  Every number published here comes from the earlier session, where the same
  control reproduced its historical value. **Re-take the control diagnostic
  before trusting any new reading against these tables.**
- `cudaMemGetInfo` is what the load-time guard checks before reserving the
  arena (`kernel/src/model.cu`), and on WDDM a reservation that exceeds free
  device memory is served from system memory rather than refused. The guard
  cannot catch an overcommit on this platform. Nothing here overcommitted,
  but a wider chunk on a busier card would, silently.
- **Watching GPU memory perturbs the measurement.** Any concurrent sampler
  (an `nvidia-smi` poll loop, `Get-Counter`, or Task Manager's GPU page)
  inflated every row by roughly 20 %: 48.4 / 110.7 / 922.4 ms at 256 / 1024 /
  8192 tokens while sampling, against 39.3 / 88.6 / 784.8 ms clean. Only the
  unsampled runs are quoted as timings anywhere in this file.

## Reproducing

```
powershell -NoProfile -ExecutionPolicy Bypass -File .scratch/issue-92/run.ps1
python .scratch/issue-92/analyze.py

# the addendum's packing proxy (free card, preflight on record, IGNIS_GPU_PROFILE=1):
cargo test -p ignis-core --features cuda --test chunk_decomposition_gpu \
  -- --ignored --nocapture --test-threads=1
```

The default sweeps stop at 4,096 tokens so a standing run does not sit near
the card's limit; the 8,192 rows above need `IGNIS_DECOMP_WIDTHS=8192` and
`IGNIS_DECOMP_SPANS=8192`.

For the Nsight Systems pass (a free card, preflight on record,
`IGNIS_GPU_PROFILE=1`, `IGNIS_DECOMP_WIDTHS=1024`):

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o .scratch/issue-92/nsys-w1024 \
  target/x86_64-pc-windows-msvc/debug/deps/chunk_decomposition_gpu-*.exe \
  --ignored --nocapture --test-threads=1
nsys stats --report cuda_api_sum --format csv -o .scratch/issue-92/stats \
  .scratch/issue-92/nsys-w1024.nsys-rep
python .scratch/issue-92/nsys_gap_report.py .scratch/issue-92/nsys-w1024.sqlite
```

The `.nsys-rep` / `.sqlite` captures are not committed (14 MB); the two
commands above regenerate them. `chunks.jsonl` here keeps the per-chunk
records; re-run with `IGNIS_CHUNK_PROFILE_LAYERS` set for the per-layer rows.

The kernel-side instrumentation (`ChunkProfiler` in `kernel/src/step.cu`) is
inert unless `IGNIS_CHUNK_PROFILE` names a file: with it unset the cost is one
cached boolean test per chunk and no CUDA event is created.
