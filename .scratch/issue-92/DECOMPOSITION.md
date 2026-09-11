# #92 criterion 1 — where the per-chunk prefill wall time goes

Measured 2026-09-11 on a free RTX 5090 (ADR 0006 profile, preflight on
record), warm, live/live. Model `qwen3_8_27b_nvfp4full-v2.ninfer`, 8,192-token
span, default chunked route (`run_program_chunk`, `kernel/src/step.cu`),
64 layers.

This file answers **acceptance criterion 1 only**. The survey of directions
(criterion 2), the CUDA-graph spike and any fix are deliberately not here.

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

## Reproducing

```
powershell -NoProfile -ExecutionPolicy Bypass -File .scratch/issue-92/run.ps1
python .scratch/issue-92/analyze.py
```

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
