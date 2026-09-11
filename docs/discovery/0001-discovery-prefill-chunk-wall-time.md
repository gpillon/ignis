# 0001 — Discovery: where prefill chunk wall time goes, and what packing would buy

**Date:** 2026-09-11
**Issue:** [#92](https://github.com/gpillon/ignis/issues/92), acceptance criterion 1
**Hardware:** RTX 5090 (GB202, 170 SMs, 32,607 MiB), free card, ADR 0006 profile
**Model:** `qwen3_8_27b_nvfp4full-v2.ninfer`, 64 layers (16 GQA / 48 GDN)
**Branch:** `issue-92-chunk-decomp`
**Raw data and scripts:** `.scratch/issue-92/`

A *discovery* record, not an ADR: it decides nothing, it reports what was
measured and how, so a later decision does not have to re-take the readings.
ADR 0005 (performance first) and ADR 0006 (exclusive GPU testing) govern how
the numbers were taken.

---

## 1. The question, and why it mattered

The default prefill route (`run_program_prefill_chunked`, `kernel/src/step.cu`)
cuts a span into `prefill_chunk_tokens`-wide chunks and issues exactly one
`cudaStreamSynchronize` per chunk before any position state may advance. For
an 8K span at the default 1,024-wide chunk that is eight forced device stops,
the "8 semaphores" #92 was opened about.

The standing belief was that those stops, plus per-chunk launch overhead,
dominated the ~94.9 ms/chunk reading — supported by #86 landing tensor-core
GDN/GQA entry points without moving that number at all.

Two decisions were waiting on the answer:

- **P4 / #65**, via `.scratch/DEFERRED-DECISIONS.md` item 1: packed prefill's
  phase was left explicitly dependent on what #92 reported.
- **#110**, where a chunk=512 run showed a *worse* ITL p95 tail than
  chunk=1024, read at the time as evidence of a fixed per-boundary cost.

## 2. What was built

| artifact | what it is |
| --- | --- |
| `ChunkProfiler` in `kernel/src/step.cu` | CUDA events on the model's own stream around the chunk, each of the 64 layer bodies and the head. Emits one JSONL record per chunk after the chunk's existing synchronize. Inert unless `IGNIS_CHUNK_PROFILE` names a file: one cached boolean test per chunk, no event created. |
| `crates/core/tests/chunk_decomposition_gpu.rs` | One GPU test, `prefill_chunk_and_traversal_sweeps`, running two sweeps off a single materialization: chunk width over a fixed 8,192-token span, then isolated spans one chunk each from position zero. Both print the leaf's own VRAM reservation per row. It is one test rather than two because two `materialize` calls do not fit on a 32 GiB card at this model's size -- the second leaves `ignis_model_load` with zero free bytes. Integration tests in separate files get separate processes and never meet this. |
| `.scratch/issue-92/run.ps1` | Drives two passes, clean and instrumented, each with its own preflight. |
| `.scratch/issue-92/analyze.py` | Regresses the sweep and aggregates the JSONL records. |
| `.scratch/issue-92/nsys_gap_report.py` | Reads an Nsight Systems capture and splits the device timeline into busy and idle. |
| `.scratch/issue-92/ncu_summary.py` | Condenses an `ncu --csv` dump into one row per kernel geometry, with grid size read against the card's SM count. |

Environment overrides, all optional and all diagnostic: `IGNIS_CHUNK_PROFILE`,
`IGNIS_CHUNK_PROFILE_LAYERS`, `IGNIS_DECOMP_WIDTHS`, `IGNIS_DECOMP_SPANS`,
`IGNIS_DECOMP_REPS`.

## 3. Method

Five measurements, deliberately overlapping, so no single instrument has to be
believed on its own. Each one covers a blind spot of the one before it.

1. **Chunk-width sweep, no instrumentation.** Same span, six widths. Wider
   chunks mean fewer synchronizations for identical token work.
2. **The leaf's own CUDA events.** Separates host-blocked time from
   device-idle time, which is the distinction the whole question turns on.
3. **Nsight Systems.** Sees the idle *inside* a layer body, which per-layer
   events cannot — those gaps land in the layer's own span.
4. **Isolated-span sweep (the packing proxy).** Removes the KV-prefix confound
   that makes measurement 1 useless for the packing question.
5. **Nsight Compute.** Answers what a timeline cannot: whether a kernel that
   is executing is actually filling the machine.

## 4. Findings

### 4.1 The per-chunk decomposition — the route is compute-bound

At the production 1,024-token chunk width, a chunk is 98.6 ms of device
timeline:

| bucket | ms/chunk | share |
| --- | ---: | ---: |
| (c) per-layer compute, kernels executing | 88.9 | **90.2 %** |
| (a) kernel launch / dispatch, device idle between kernels | 9.2 | 9.3 % |
| (b) the forced `cudaStreamSynchronize`, device idle at the boundary | 0.47 | **0.5 %** |

**The eight semaphores cost 3.8 ms of a ~790 ms 8K prefill.** The premise #92
was opened on does not survive measurement.

Why #86 changed nothing follows: the tensor-core routes landed inside the
90 % that was already compute, and what they saved was not large relative to
it. The residual 10 % is launch latency spread over ~1,160 kernels per chunk,
not the one synchronization per chunk.

### 4.2 Chunk-width sweep (measurement 1)

8,192-token span, 3 timed reps after a warm-up, per width:

| width | chunks | wall mean (ms) | wall min (ms) | ms/chunk | ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 1316.4 | 1304.9 | 41.14 | 0.1607 |
| 512 | 16 | 942.2 | 938.8 | 58.89 | 0.1150 |
| **1024** | **8** | **797.8** | **785.2** | **99.72** | **0.0974** |
| 2048 | 4 | 775.7 | 768.7 | 193.93 | 0.0947 |
| 4096 | 2 | 767.0 | 764.8 | 383.50 | 0.0936 |
| 8192 | 1 | 790.2 | 768.9 | 790.20 | 0.0965 |

Collapsing eight synchronizations into one buys 16.3 ms on a 785.2 ms span,
**2.1 %**, best-of-three against best-of-three.

**A trap worth recording.** Regress wall on chunk count over all six widths
and the slope reads **17.7 ms per boundary**, which would be 141 ms of the
default route's 790 ms and would have confirmed the ticket. It is wrong: at
256 and 512 tokens the chunk is too narrow for the GEMM shapes, so per-token
device compute itself rises, and the regression charges that lost compute to
the boundary. Fit only the widths whose per-token cost has plateaued (>= 1024)
and the slope is **2.66 ms**, of which 0.47 ms is the synchronization.

This is also the answer to #110's chunk=512 observation from the other side: a
worse tail at the narrower width is consistent with a fixed per-boundary cost,
but equally consistent with narrow chunks computing less efficiently, and the
per-token compute column says it is the second.

### 4.3 Leaf event decomposition (measurement 2)

Per-chunk means, 3 timed spans per width, warm-up span dropped:

| width | chunk_ms | enqueue | sync_stall | entry_gap | embed | layers | head | layer_gap | gpu_span | layers/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 40.66 | 22.73 | 17.93 | 0.22 | 0.05 | 40.40 | 0.04 | 0.11 | 40.60 | 0.1578 |
| 512 | 59.69 | 32.32 | 27.36 | 0.43 | 0.06 | 59.21 | 0.08 | 0.11 | 59.45 | 0.1156 |
| **1024** | **97.99** | **57.00** | **40.99** | **0.47** | **0.05** | **97.29** | **0.14** | **0.10** | **97.58** | **0.0950** |
| 2048 | 191.60 | 122.84 | 68.76 | 0.46 | 0.09 | 190.68 | 0.28 | 0.10 | 191.15 | 0.0931 |
| 4096 | 381.23 | 292.88 | 88.35 | 0.25 | 0.11 | 379.95 | 0.63 | 0.10 | 380.79 | 0.0928 |
| 8192 | 763.79 | 763.45 | 0.33 | 0.00 | 0.20 | 761.71 | 1.35 | 0.10 | 763.36 | 0.0930 |

- `entry_gap` is the measurement #92 actually asked for: device idle between
  the previous chunk's last operation and this chunk's first, which is the
  bubble the forced synchronization opens. **0.47 ms.**
- `sync_stall` is **not** that bubble. It is host wall blocked inside
  `cudaStreamSynchronize` while the device computes. Reading it as overhead is
  the mistake the ticket rested on: the host being blocked and the device
  being idle are different quantities, and only the second costs anything.
- `enqueue` overlaps device work. At width 8,192 it absorbs the whole span
  with a 0.33 ms stall, which is a host throttled by a full launch queue, not
  host-bound dispatch.
- `layer_gap` is device idle *between* layer bodies: 0.10 ms, essentially
  nothing. The host is never late with the next layer.

Instrumentation cost is inside run-to-run noise: 785.3 ms at width 1,024 with
the events on against 797.8 ms with them off.

### 4.4 Nsight Systems (measurement 3)

Width-1,024-only capture, `--trace=cuda`:

```
prefill region: 37133 kernels, 3154.2 ms wall      (4 spans x 8 chunks)
  device busy :    2844.0 ms  (90.17%)
  device idle :     310.1 ms  ( 9.83%)
  overlapping kernel pairs: 0 (one stream, as expected)
  mean gap between consecutive kernels: 8.4 us over 37132 gaps
  idle in gaps <= 100 us:  185.4 ms over 36790 gaps (mean 5.0 us) -- launch latency
  idle in gaps >  100 us:  124.7 ms over   342 gaps              -- scheduling hiccups
```

90.2 % of the device timeline is a kernel executing, with no reliance on the
leaf's own events. The wider gaps are scattered across many kernel types, ~11
per chunk, with no single structural host round-trip behind them.

The capture's own wall (98.6 ms/chunk) matches the uninstrumented sweep
(98.2-99.7 ms/chunk), so tracing is not inflating the idle it reports.

Host-side, same capture: `cudaStreamSynchronize` 1384.8 ms over 33 calls, 44 %
of prefill wall. Host blocked, device busy. `cudaLaunchKernel` plus
`cudaLaunchKernelExC` total 1158.5 ms against 3154.2 ms of wall, hidden.

### 4.5 The packing proxy (measurement 4)

"Busy" is not "saturated". A GEMM occupying a quarter of the machine still
counts as 100 % busy on a timeline, so 4.1 must be read narrowly: **the wall
time is in kernels, not in synchronization or dispatch.** It is not a claim
that the math is near peak.

Whether a traversal carrying more tokens costs less per token is a separate
question, and it is the one packed prefill turns on. Packing N requests gives
every projection and FFN GEMM the shapes of one N*L-token traversal, and those
GEMMs are ~75 % of kernel time.

Measurement 1 cannot answer it: a 256-token chunk there sits inside an
8,192-token span and still attends to up to 8K of KV prefix, attention work a
standalone 256-token request would never do. Here each span is prefilled on
its own, from position zero, in exactly one chunk of its own width:

| tokens in the traversal | wall min (ms) | ms/token | vs 256 | leaf VRAM (GiB) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 39.3 | 0.1534 | 1.00x | 17.42 |
| 512 | 55.1 | 0.1077 | 1.42x | 17.50 |
| **1024** | **88.6** | **0.0865** | **1.77x** | **17.66** |
| 2048 | 181.3 | 0.0885 | 1.73x | 17.98 |
| 4096 | 370.0 | 0.0903 | 1.70x | 18.61 |
| 8192 | 784.8 | 0.0958 | 1.60x | 19.88 |

Per-token cost bottoms out at **~1,024 tokens per traversal**. Four 256-token
prompts in one traversal are **1.77x cheaper per token** than four separate
ones. Above 1,024 the curve turns back up, but that rise is attention over a
longer average prefix, so the wide end is a conservative floor for packing
rather than a ceiling.

Splitting the 1.77x by mechanism, using the ~9.2 ms of per-traversal launch
idle from 4.4 (roughly fixed, since a traversal is ~1,160 kernels regardless
of token count):

| | 256-token traversal | 1,024-token traversal |
| --- | ---: | ---: |
| wall | 39.3 ms | 88.6 ms |
| launch idle, approximately fixed per traversal | ~9.2 ms (23 %) | ~9.2 ms (10 %) |
| kernels executing, per token | ~0.1176 ms | ~0.0775 ms |

Roughly **1.5x is GEMM shape** and **1.17x is dispatch amortization**, and the
amortized part is the 9 % launch overhead, not the 0.5 % synchronization.

### 4.6 Hardware counters (measurement 5)

`ncu` over the dominant NVFP4 projections, `launch__grid_size` read against
this card's 170 SMs.

**T = 256 tokens** — the engine takes the `mma` route:

| kernel | grid | waves | SM % | tensor pipe % |
| --- | ---: | ---: | ---: | ---: |
| `nvfp4_w4a4_mma<5120,17408>` (MLP down) | 80 | **0.47** | 21.0 | 45.7 |
| `nvfp4_w4a4_mma<5120,6144>` (GDN out) | 80 | **0.47** | 19.0 | 42.1 |
| `nvfp4_w4a4_mma<16384,5120>` (GDN qkvz) | 256 | 1.51 | 42.6 | 52.9 |
| `nvfp4_w4a4_mma<34816,5120>` (MLP gate/up) | 544 | 3.20 | 45.3 | 54.7 |

**T = 1024 tokens** — a *different* kernel runs; the `mma` variant is not
launched at this width at all:

| kernel | grid | waves | SM % | tensor pipe % |
| --- | ---: | ---: | ---: | ---: |
| `nvfp4_linear_swiglu_w4a4_tma` | 1088 | 6.40 | 50.4 | 61.7 |
| `nvfp4_w4a4_tma<5120,17408>` | 160 | **0.94** | 65.0 | 75.1 |
| `nvfp4_w4a4_tma<5120,6144>` | 160 | **0.94** | 57.7 | 66.0 |
| `nvfp4_w4a4_tma<16384,5120>` | 512 | 3.01 | 61.6 | 73.5 |

At 256 tokens two of the four dominant GEMMs launch **80 blocks onto a 170-SM
card** — under half the machine has any work — at ~20 % SM throughput. Filling
the traversal fixes both halves at once: the grids double, *and* the engine
crosses into the TMA route, taking SM throughput to 58-65 % and the tensor
pipe to 66-75 %.

**The route switching with token count is a finding in its own right.** The
1.77x of 4.5 is two effects stacked, a wider grid and a kernel swap, and
neither is the synchronization this ticket set out to chase.

Nothing is saturated even at 1,024 tokens: `<5120,17408>` is still at 0.94
waves, so ~10 SMs sit idle through it, and the tensor pipe tops out at 75 %.
But the wall clock says pushing past 1,024 tokens does not convert that
headroom into speed, so the limiter above that width is inside the kernels,
not in how many tokens the traversal carries.

### 4.7 VRAM audit

Taken because the card looked close to saturated during a profiling run. The
leaf's own reservation (`ignis_program_stats`: weights, prefill scratch arena,
KV and GDN pools, sampling and decode-graph buffers) is the `leaf VRAM` column
of 4.5: flat at 17.4 GiB and rising to 19.88 GiB at the widest traversal. All
the growth is the prefill scratch arena, which `ignis_model_load` sizes for
the chunk width (P2-01, #83), about 2.5 GiB across the sweep.

The KV pool is not the driver. For this budget — 258 page groups of 64 tokens,
16 GQA layers, 4 KV heads x 256 head dim, K and V, BF16:

```
258 * 64 * 16 * 4 * 256 * 2 * 2 = 1.01 GiB
```

plus 2 slots of GDN state (48 layers x 48 heads x 128x128) at 151 MiB. About
1.2 GiB of a 17.4 GiB baseline, identical in every row, so it cannot explain a
difference between rows.

Sampled with the Windows `\GPU Process Memory(*)` counters during an `ncu` run
at T=1024:

| | MiB |
| --- | ---: |
| peak dedicated | 21,367 |
| peak shared | 195 |
| card | 32,607 |

No spill into system memory was reproduced, so every row above stands as
measured.

## 5. Caveats

- **`cudaMemGetInfo` cannot guard an overcommit on WDDM.** The load-time check
  in `kernel/src/model.cu` compares the scratch reservation against free
  device memory, but on Windows a reservation that exceeds it is served from
  system memory rather than refused. Nothing here overcommitted; a wider chunk
  on a busier card would, silently.
- **Watching GPU memory perturbs the measurement.** Any concurrent sampler
  (an `nvidia-smi` poll loop, `Get-Counter`, or Task Manager's GPU page)
  inflated every row by roughly 20 %: 48.4 / 110.7 / 922.4 ms at 256 / 1024 /
  8192 tokens while sampling, against 39.3 / 88.6 / 784.8 ms clean. Only
  unsampled runs are quoted as timings in this document.
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
- **`ncu` needs the performance-counter permission.** Without it every run
  fails with `ERR_NVGPUCTRPERM`. It is a driver-level setting, not something
  the harness can set.
- **Debug profile.** All Rust-side timings are `cargo test`'s default debug
  build, matching every prior reading on this ticket (#86, #88). The host
  enqueue loop being profiled lives in the release-compiled kernel leaf, so
  the profile does not affect what is measured.
- **The 8,192-token row does more attention work per token** than eight
  independent 1,024-token spans would, so it understates packing at the wide
  end.

## 6. What this settles

- Criterion 1 of #92: per-chunk wall time is **90 % per-layer compute, 9 %
  launch/dispatch, 0.5 % the forced synchronization**. The route is
  compute-bound in the sense that matters for that question.
- The P4 / #65 input: the cost is **not** dominated by sync/dispatch.
- `.scratch/DEFERRED-DECISIONS.md` item 1 needs restating rather than
  confirming. Its stated reason (a dominant sync/dispatch cost to amortize
  across N) is not there. Its conclusion survives on a different and
  conditional footing: below ~1,024 tokens per traversal the model is
  shape-starved and packing is what fills it, worth up to 1.77x at 256-token
  prompts and nothing at 1,024 and above. So the phase turns on the workload,
  not on this ticket's overhead numbers. The gates measure 8K prompts, where
  one request already fills a traversal; G4's subagent-burst trace is many
  short prompts, which is the regime that pays.

## 7. What it does not settle

- Whether the 90 % itself can be made smaller. The counters say no kernel is
  at peak (tensor pipe 66-75 % at the best width) but the wall clock says more
  tokens do not convert that headroom into speed, so the remaining limiter is
  inside the kernels. Kernel work, not this ticket.
- Criterion 2 of #92, the survey of directions, and the CUDA-graph spike.
- **The retry-contract constraint stands.** That synchronization is today the
  only failure-detection point for the P2-02 / #84 contract: the caller may
  not advance `seq`'s position state until it confirms the chunk's device work
  completed. Moving it means relocating detection (an event queried before the
  advance, or deferring the advance to the span's end), not deleting a line.
  At 0.5 % of wall that is a large contract change for a small number. #110's
  probe already measured the serving-level version: skipping the
  non-final-chunk sync moved the ITL p95 ratio from 1.130 to 1.110.

## 8. Reproducing

All of it needs a free card and a preflight pass on record
(`docs/agents/testing.md`).

```
# measurements 1 and 2 -- the width sweep, clean then instrumented
powershell -NoProfile -ExecutionPolicy Bypass -File .scratch/issue-92/run.ps1
python .scratch/issue-92/analyze.py

# measurements 1 and 4 -- both sweeps, one process
cargo test -p ignis-core --features cuda --test chunk_decomposition_gpu \
  -- --ignored --nocapture --test-threads=1
```

The default sweeps stop at 4,096 tokens so a standing run does not sit near
the card's limit; the 8,192 rows above need `IGNIS_DECOMP_WIDTHS=8192` and
`IGNIS_DECOMP_SPANS=8192`.

Measurement 3, Nsight Systems, with `IGNIS_GPU_PROFILE=1` and
`IGNIS_DECOMP_WIDTHS=1024`:

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o .scratch/issue-92/nsys-w1024 \
  target/x86_64-pc-windows-msvc/debug/deps/chunk_decomposition_gpu-*.exe \
  --ignored --nocapture --test-threads=1
python .scratch/issue-92/nsys_gap_report.py .scratch/issue-92/nsys-w1024.sqlite
```

Measurement 5, Nsight Compute, with `IGNIS_GPU_PROFILE=1`,
`IGNIS_DECOMP_SPANS=<width>` and `IGNIS_DECOMP_REPS=1`:

```
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,gpu__time_duration.sum \
  -k "regex:nvfp4" -s 40 -c 30 --csv \
  target/x86_64-pc-windows-msvc/debug/deps/chunk_decomposition_gpu-*.exe \
  --ignored --test-threads=1 > .scratch/issue-92/ncu-tN.csv
python .scratch/issue-92/ncu_summary.py .scratch/issue-92/ncu-tN.csv
```

The `.nsys-rep` and `.sqlite` captures are not committed (14 MB); the commands
above regenerate them. `.scratch/issue-92/chunks.jsonl` keeps the per-chunk
records from the instrumented pass; re-run with `IGNIS_CHUNK_PROFILE_LAYERS`
set to get the per-layer rows as well.

## 9. References

- `.scratch/issue-92/DECOMPOSITION.md` — the working writeup this record
  consolidates, linked from the #92 comments.
- `kernel/src/step.cu` — `run_program_chunk`, `run_program_prefill_chunked`,
  `ChunkProfiler`.
- `kernel/src/model.cu` — `compute_program_scratch_bytes`, the load-time
  scratch reservation (P2-01, #83).
- ADR 0016 — the chunked-span prefill options struct.
- ADR 0006 — exclusive GPU testing, the profile every reading here was taken
  under.
- GitHub #84 (P2-02, the route and its retry contract), #86 (P2-04,
  tensor-core routes), #88 (G2 gate), #110 (ITL p95 tail), #65 (P4), #13 (the
  original "we may be compute-bound" caveat).
