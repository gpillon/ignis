# What the prefill "other" class is

- Kind: experiment
- Scope: kernel / prefill TTFT; retained host slots (#281), per-layer copies,
  the vendored TMA GEMM's descriptor upload on Windows
- Observed: 2026-09-30
- Status: current
- Related: [GPU resources: prefill vs decode](2026-09-29-gpu-resources-prefill-vs-decode.md),
  [A16 NVFP4 on the tensor cores](2026-09-29-a16-nvfp4-on-the-tensor-cores.md),
  [Retained slots on the host](2026-09-28-retained-slots-on-the-host.md),
  [Sequence snapshot transfer cost](2026-09-12-sequence-snapshot-transfer-cost.md),
  [Prompt-reuse tax on short TTFT](2026-09-18-prompt-reuse-tax-on-short-ttft.md)

## Question

After the A16 route, the class split of prefill device time left an "other"
class of ~15%. What is in it, and is there a lever?

## Method

No new GPU run. The input is the nsys capture taken for the A16 finding
(`prefill-a16`, `.scratch/gpu-resources-2026-09-29/trace_phase.sh`, production
flags, one lane, distinct ~5,500-token prompts with `max_tokens` 1, 8 s
window, 10 whole requests). The capture already holds every kernel, memcpy
and memset, so it can be read for the time *between* kernels too. The scripts
are in `.scratch/gpu-resources-2026-09-29/`: `other.py` (per-kernel split of the class),
`mc.py` (copies and their overlap with kernels), `req.py` (budget per
request), `gaps.py` (idle gaps by what surrounds them), `burst.py` (D→H
bursts and the stall after each), `h2d.py` (who the small uploads sit
between).

## Result

**The class was mislabelled.** `classes.py` did not list our own
`nvfp4_a16_mma_kernel`, a tensor-core kernel (200 ms of the window). Fixed,
the split is tensor 80.6%, other 11.3%, DRAM 6.6%, CUDA core 1.5%.

**The real "other" kernels are ~30 small kernels with no single lever.** The
largest are `nvfp4_w4a4_quantize` (1.6% of kernel time, 21,546 launches of
4 µs), `causal_conv1d_prefill_pairs` (1.6%), `gqa_attention_prefill_fill_hq`
(1.6%; ncu: 10% SM, 4% DRAM, latency bound), then the two rmsnorm kernels
(1.0% + 1.0%, 16,800 launches), `gqa_attention_prefill_reduce` (0.9%) and
the GDN gating projection (0.8%).

**Kernel time is not TTFT.** Per request, from the first kernel of the
prefill to the first-token sampler (`argmax`), the span is ~600 ms:

| per request (~600 ms span) | ms | share |
|---|---|---|
| tensor-core kernels | ~383 | 64% |
| DRAM-bound + CUDA-core kernels | ~38 | 6% |
| "other" kernels | ~54 | 9% |
| **retained slots**: 2 captures into host slots (D→H) | ~45 | 8% |
| **retained slots**: 1 eviction into KV-RAM (D→H pages + host memcpy) | ~30 | 5% |
| launch gaps under 5 µs (~12,000 of them, eager prefill) | ~26 | 4% |
| gaps around the TMA GEMM's descriptor uploads | ~12 | 2% |
| D→D copies and their gaps | ~16 | 3% |

The table is rounded, so the rows sum to about 600 (the retained rows come
from `burst.py`, the idle rows from `gaps.py`; they overlap by a few ms). The rest of the idle time
is gaps between 20 µs and 1 ms that no single cause explains.

### 1. Retained slots on the host stall the GPU (~75 ms, ~12%)

Every request moves ~513 MB device → pinned host during its own prefill, at
10.2 GB/s. That is the PCIe Gen 3 ceiling of this host, as in the snapshot
finding. None of the 1,184 copies overlaps a kernel. The source is #281: with
`--retained-device 0 --retained-host 16` (the serving default), retained state
lives in host slots. Two paths run in every request, and they stall the GPU in
different ways:

**Capture (2 per request, ~20-25 ms each).** A prompt checkpoint captured at
a reuse boundary goes into a host retained slot.

- `ignis_seq_copy_slot_state` (`kernel/src/seq.cu`) copies the lane's clone
  image of 232.5 MB in 15 copies: GDN conv + recurrent (151 MB), the hq
  residual window (K and V, 17.8 MB each), the drafter window, and the
  penalty counts.
- The copies run on the legacy default stream, then the function calls
  `cudaStreamSynchronize(nullptr)`.
- The legacy stream orders against the compute stream, so no kernel runs
  during the transfer. Afterwards the gap is only ~1 ms.

A prompt of this length is cut at two reuse boundaries (the A/B log shows 8
chunks consumed where 6 would do), so it captures twice.

**Eviction (1 per request, ~30 ms).** From the ninth request on, the 16 host
slots are full, and a new capture first evicts the lowest-ranked checkpoint
into KV-RAM:

- `spill_checkpoint` calls `ignis_seq_write_materialized_blob`
  (`kernel/src/seq.cu`).
- That call packs the victim's KV pages D→H: 44-51 MB in 64 copies, ~5.5 ms.
- It then `std::memcpy`s the victim's 232 MB host image into the arena on the
  scheduler thread, with the checkpoint's `cudaStreamSynchronize`
  (`seq_checkpoint.cu:64`).
- The GPU sits idle for **15-31 ms** after the page burst, every request.

In this benchmark every prompt is distinct, so no capture is ever claimed and
every eviction is churn. That is the worst case. In agent traffic the
captures are the point of retention: the cost per capture is the same, and it
pays off only on a hit.

#281 measured KV capacity (+42%) and decode (unchanged). It noted "TTFT p50
on the long load higher" and accepted "a few ms per request". Here it is
~75 ms per ~5.5K-token cold prompt.

### 2. The Windows descriptor upload of the TMA GEMM (~12 ms, ~2%)

12,298 pageable 512-byte H→D copies sit between `nvfp4_w4a4_quantize` and
each `nvfp4_w4a4_tma_kernel` / `nvfp4_linear_swiglu_w4a4_tma_kernel`. The
vendored launcher (`nvfp4_w4a4_tma.cu:103`, `_WIN32` branch) copies the four
`CUtensorMap`s to a device block for every call. MSVC cannot lay out an
`alignas(128)` by-value kernel parameter, so Linux passes them as
`__grid_constant__` and pays nothing. Each upload leaves ~11 µs of idle GPU
around it.

### 3. Per-layer D→D copies (~16 ms, ~3%)

These are ~980 copies per request, on the compute stream:

- every layer copies the residual in → out (`hidden × T` BF16, 10.5 MB per
  chunk) before its in-place residual add;
- every GDN layer splits q / k / v out of the conv output with three
  `cudaMemcpy2DAsync` calls, because the per-head views need contiguous
  storage (`gdn_layer.cu`).

They run at ~960 GB/s, and the gaps between them cost as much as the copies.

### 4. Launch gaps (~26 ms, ~4%)

Prefill is launched eagerly, ~8,700 kernels plus ~2,100 copies per request.
Most boundaries cost 1-3 µs. The elementwise kernels are the densest:
`nvfp4_w4a4_quantize` and `rmsnorm_cta` alone are 34,000 launches of ~4 µs
in the window.

## Implications

In order of size:

1. **Take retained slots off the critical path** (~75 ms, ~12% of a cold
   5.5K TTFT). The two paths need different fixes, and neither is a kernel
   change:
   - *Capture* (~45 ms): snapshot D→D into one device staging image
     (~0.3 ms for 232 MB). Then drain it to the host slot on a non-blocking
     copy stream, behind an event, without a host sync. The price is 232 MB
     of VRAM (1/16 of what #281 moved off the device), and the next capture
     waits on the previous drain.
   - *Eviction* (~30 ms): the host memcpy of the victim's image does not need
     the GPU at all. It can hand the pinned image over to the arena, run off
     the scheduler thread, or be avoided by less churn (a capture that is
     never claimed need not evict a checkpoint that might be).
2. **Cache the TMA descriptors on Windows** (~12 ms, ~2%). Keep a small
   device-side cache keyed by (pointers, tokens). Workspace activations and
   weights have stable addresses, so the upload happens once, not per call.
   This is a vendored-op patch under ADR 0031.
3. **Fuse and remove small work** (~16 + part of 26 + part of 54 ms). Two
   candidates:
   - drop the residual copy (write the residual add out of place) and the
     q/k/v split copies (strided views);
   - fuse rmsnorm with the NVFP4 activation quantize that follows it.
   Each is a few ms. Together they are comparable to item 2.

The "other" kernels themselves (9%) have no dominant member. Making any one of
them twice as fast saves under 1% of TTFT.

## Limits and unknowns

- The numbers are read from one capture, not an A/B. The retained-slot
  cost (~75 ms) is read from the trace, not measured end to end. The direct
  check is not run yet: `ab_run.sh` at `--retained-device 8 --retained-host 0`
  (the #281 baseline) against the default. It should show a ~75 ms median
  drop. A ~20 ms drop would mean this finding overstates the cost.
- One lane, cold distinct prompts: the worst case for capture volume. At 8
  lanes, other lanes' decode rounds can fill part of the capture stall, but the
  legacy-stream ordering stalls every stream, so they probably do not.
- The ~100 ms between one request's first token and the next request's first
  kernel is the harness (curl restart) plus host tokenization. It is not in
  the budget above.
