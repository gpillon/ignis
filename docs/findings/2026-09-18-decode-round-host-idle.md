# A decode round loses ~1 ms to host-side stalls, fixed per round, and the memcpy nodes are not the cause

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / decode round, CUDA graph replay, device timeline
- Related: [ADR 0019](../adr/0019-decode-cuda-graph-slot-indirection.md),
  [ADR 0020](../adr/0020-batch-wide-decode-round.md),
  [ADR 0021](../adr/0021-live-live-launch-pooling.md),
  [hq vs BF16 live/live](2026-09-13-hq-vs-bf16-live-live.md),
  [GQA workspace memset](2026-09-18-gqa-workspace-memset.md)
- Superseded by: none

**Hardware:** RTX 5090, exclusive card (ADR 0006).
**Engine:** production defaults — `hq-e8-2b`, `--max-context 262144`,
`--spec dflash2 --draft-tokens 7`, so every round is a width-8 verify round.
**Captures, harness and report script:** `.scratch/decode-idle-2026-09-18/`.

## Question

The GQA and GDN layer bodies enqueue 224 `cudaMemcpy`/`cudaMemset` nodes per
decode round — 16 GQA layers × (1 memset + 1 residual copy) and 48 GDN layers ×
(3 strided QKV copies + 1 residual copy). The vendored ops around them are
almost all PDL-chained (`pdl::launch_dependent` in linear, linear_add,
linear_swiglu, attn_input_proj, gdn_input_proj, gated_delta_net, rmsnorm, rope,
qk_norm_rope, gqa decode). A memcpy node carries no programmatic trigger, so
the standing hypothesis was that each one degrades the next kernel's PDL edge
to a full stream dependency — 224 forfeited prologue overlaps per round, which
would have connected those "small" items to the memory idleness in the
[live/live finding](2026-09-13-hq-vs-bf16-live-live.md).

## Evidence

Two Nsight Systems captures of 8 s of steady-state decode against the running
server, one lane and eight lanes, plus a third with `--cuda-graph-trace=node`
to see inside a replay. The device timeline is one stream, so idle is the sum
of the holes between activities, and every hole is attributed to the pair it
sits between (`gap_report.py`).

### The hypothesis is wrong

From the node-level capture, gaps by transition:

| transition | count | total | mean | p50 |
|---|---|---|---|---|
| kernel → kernel | 289,067 | 71.1 ms | 0.25 µs | 0.10 µs |
| memcpy → kernel, under 50 µs | 31,428 | 8.4 ms | 0.27 µs | 0.10 µs |
| **after a residual copy (81,920 B)** | **15,842** | **1.4 ms** | **0.09 µs** | **0.10 µs** |

The gap after the per-layer residual copy — the memcpy node most suspected —
is 0.10 µs at p50 and 0.13 µs at p99, indistinguishable from a kernel-to-kernel
edge. All in-round idle following any memcpy is 8.4 ms of an 8,000 ms window
(0.10%), against kernel-to-kernel's 61.2 ms (0.76%). Every memcpy in the
window executes in 0.73 µs on average, 29.7 ms in total.

The `memcpy → kernel` mean of 17.3 µs that first suggested the hypothesis is
an artefact of mixing scales: 757 gaps of 50 µs or more carry 548 ms of it,
and those are round boundaries, not layer-body edges.

### Where the idle actually is

From the graph-level captures (lower overhead; node tracing roughly doubles
measured idle):

| | 1 lane | 8 lanes |
|---|---|---|
| rounds in 8 s | 388 | 215 |
| graph replay, mean | 19.18 ms | 33.54 ms |
| device idle | 393.5 ms (**4.91%**) | 182.2 ms (**2.27%**) |
| idle per round | 1,014 µs | 847 µs |

The same five stalls appear in both, in the same order, with the staged buffers
scaled by lane count (40 B → 320 B, 4 B → 32 B, 32 B → 256 B):

| after | before | 1 lane | 8 lanes |
|---|---|---|---|
| the sampling-config staging copy | **the graph launch** | 389 µs | 406 µs |
| a per-lane staging copy | `recurrent_fold_kernel` | 109 µs | 105 µs |
| `kv_cache_append_prefix_cyclic_kernel` | a per-lane copy | 64 µs | 81 µs |
| the graph launch | a per-lane copy | 51 µs | 43 µs |
| a per-lane copy | a per-lane copy | 50 µs | 52 µs |

## Finding

**Observed.** The 224 memcpy/memset nodes per round cost their own execution
and nothing more: the device resumes 0.10 µs after a residual copy retires.
Removing them would return ~0.16% of decode, not the several percent the PDL
argument predicted.

**Observed.** A decode round leaves the device idle for ~1 ms, and that cost is
**fixed per round**: 847-1,014 µs whether the round serves one lane or eight.
It is not one bubble but five recurring host-side stalls, the largest of them
389-406 µs sitting between the round's last staging copy and `cudaGraphLaunch`.

**Inference.** Because the cost is per round rather than per lane, it is a
throughput tax that batching already dilutes — 4.91% of the timeline at one
lane against 2.27% at eight — and a latency tax that batching does not: single-
stream ITL pays the full ~1 ms on a 19.18 ms round. That is the same order as
the 1.03x ITL p95 gap against the reference that both live/live measurements
report, in both KV formats.

**Observed, incidental.** A round at eight lanes costs 33.54 ms against 19.18 ms
at one — 1.75x the device time for 8x the lanes.

## Implications

- The residual ping-pong copy and the GDN QKV split copies are not performance
  work. If they are removed it should be for the code, not the clock.
- The remaining lever in decode is host-side: what runs between a round's
  staging copies and its launch, and between the graph replay and the fold.
  None of it is kernel work.
- The measurement to make next is a host-side one. These captures ran with
  `--sample=none`, so they locate the stalls on the device timeline but say
  nothing about which host code occupies them.

## Limits and unknowns

- 8 s windows, one capture per width, no repeats: the per-round stall figures
  are stable across two widths and five stall sites, but no variance is
  established.
- The client drove back-to-back generations of one fixed prompt, so the window
  contains that workload's prefills too; the per-round figures are keyed on
  graph replays and are unaffected, but the window percentages include prefill.
- Nothing here attributes the stalls to specific host code, and nothing
  measures what removing them would be worth.
- The node-level capture inflates idle (768 ms against 393 ms in the window),
  so only its *relative* comparisons are used above.

## Follow-ups

- Re-capture with CPU sampling or NVTX ranges around `ignis_program_decode` to
  name the host code in the 389 µs pre-launch stall.
- ADR 0021 (live/live launch pooling) is the decision this bears on.
