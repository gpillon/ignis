# A decode round is at its bandwidth bound: 1,166 nodes, 4.8% idle, and the compute is weight streaming

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / decode round, CUDA graph submission cost, roofline accounting
- Related: [ADR 0019](../adr/0019-decode-cuda-graph-slot-indirection.md),
  [ADR 0020](../adr/0020-batch-wide-decode-round.md),
  [ADR 0021](../adr/0021-live-live-launch-pooling.md),
  [Decode round host idle](2026-09-18-decode-round-host-idle.md),
  [Drafter top-k](2026-09-18-dflash2-topk-one-warp-per-column.md)
- Superseded by: none

**Hardware:** RTX 5090, exclusive card (ADR 0006).
**Engine:** production defaults, with the drafter top-k already replaced.
**Captures, harness and the launch benchmark:** `.scratch/decode-idle-2026-09-18/`.

## Question

With the drafter's top-k fixed, a decode round at one lane is 15.81 ms of
device time and 0.80 ms of idle. Two things were still unknown: whether the
~370 us stall before the graph launch is proportional to the graph's node
count (which would make node count a lever, and would put the per-layer memcpy
nodes back in play), and whether the kernels that remain have any headroom at
all. Both decide where the next decode work should go — or whether there
should be any.

## Evidence

### The graph launch is linear in node count

`cudaGraphLaunch`'s host duration, measured on graphs of N chained trivial
kernels (`graphlaunch_bench.cu`, standalone, no server):

| nodes | launch | per node |
|---|---|---|
| 1 | 5.40 us | 5.40 us |
| 64 | 23.06 us | 0.36 us |
| 256 | 106.84 us | 0.42 us |
| 512 | 244.39 us | 0.48 us |
| 1024 | 643.84 us | 0.63 us |
| 2048 | 1,490.38 us | 0.73 us |

It is linear, at 0.36-0.73 us per node, with `cudaGraphUpload` already paid and
the exec relaunched warm. So the ~370 us the server spends there is
submission, not our code and not a one-time upload — and node count is the
only lever on it.

### What the 1,166 nodes are

Per round, from the node-level capture:

| | per round | device time |
|---|---|---|
| `nvfp4_w4a4_mma_kernel` | 247.4 | 9.37 ms |
| `nvfp4_w4a4_quantize_kernel` | 247.4 | 0.35 ms |
| `rmsnorm_cta_bf16x2_kernel` | 141.2 | 0.43 ms |
| memcpy nodes, all sizes | 107.1 | 0.08 ms |
| `recurrent_record_kernel` | 48.1 | 0.48 ms |
| `rmsnorm_warp_bf16x2_kernel` | 48.1 | 0.09 ms |
| GDN gating proj, partial + reduce | 96.2 | 0.24 ms |
| `nvfp4_gdn_conv_post_kernel` | 48.1 | 0.28 ms |
| `nvfp4_small_t_kernel` | 32.0 | 0.80 ms |
| `w8_small_t_mma_kernel` | 2.0 | 1.66 ms |
| everything else | ~100 | ~1.5 ms |
| **total** | **1,166** | **15.81 ms** |

At 0.32 us per node (the server's own 370 us over 1,166) the whole submission
budget is **2.2% of the round**. The memcpy nodes are 9% of the count and
0.5% of the device time, so removing every one of them — the residual
ping-pong copies and the GDN QKV splits both — is worth about **0.5%**. The
largest single fusion available, folding `nvfp4_w4a4_quantize_kernel` into the
MMA it feeds, removes 21% of the nodes and is worth about **0.5%** too.

### The compute is at the bandwidth bound

`nvfp4_w4a4_mma_kernel` is the NVFP4 backbone. This load holds 18.29 GB of
weights, of which the INT8 output head is 1.27 GB and the drafter its own
share; the backbone is therefore ~16.8 GB, which at this card's ~1.79 TB/s is
**9.4 ms**. Measured: **9.37 ms**. It is at its bound.

`w8_small_t_mma_kernel` runs twice — the drafter's proposal head and the
round's own — streaming 2 x 1.27 GB, a 1.42 ms bound against 1.66 ms measured
(86%). The two cannot be folded into one call: the drafter's head must finish
before the verify columns it proposes exist.

## Finding

**Observed.** A decode round's device time is weight streaming at this card's
bandwidth. The backbone GEMM is at 100% of its roofline and the output head at
86%. There is no kernel left in decode with meaningful headroom.

**Observed.** The graph launch's cost is 0.36-0.73 us per node on this
platform, so the round's 1,166 nodes cost ~370 us of host submission that the
device waits through — 2.2% of the round. No single available change removes
more than about a fifth of it.

**Inference.** Decode's remaining headroom is the 4.8% it spends idle, and
it is structural rather than per-kernel: ~370 us of graph submission plus
~430 us spread over four host-side stalls, which are the two
`cudaStreamSynchronize` a verify round takes — one to read the accept results
back, one to confirm the fold — and the host logic between them. Every one of
those is a host round trip on the critical path, and batching dilutes the
whole of it (2.27% at eight lanes) while single-stream latency pays all of it.

**Inference.** The only change that would take a large bite is architectural:
making the accept and fold decisions device-resident so a round can be
submitted without waiting for the previous round's results to reach the host.
That is an ADR-level decision (ADR 0021 is where it belongs), not a kernel
change.

## Implications

- **Stop optimising decode kernels.** The profile that found the top-k a 71x
  defect now clears everything else it ranks.
- The per-layer residual copies and the GDN QKV splits are worth ~0.5%
  together. If they are removed it should be for the code, as the
  [host idle finding](2026-09-18-decode-round-host-idle.md) already concluded
  for a different reason.
- Prefill was not re-examined here and is a separate question: it is not
  submission-bound (it enqueues eagerly) and its own layer bodies were only
  ever measured for the workspace memset.

## Limits and unknowns

- The weight accounting behind the roofline is derived from the load's
  `vram_plan` total minus the head, not from a per-tensor sum, so "100% of
  the bound" is good to perhaps ±10%. The conclusion — no meaningful headroom
  — survives that band; a claim of an exact percentage would not.
- 0.32 us per node is the server's own ratio, inferred by dividing its
  measured 370 us by a node count taken from a different capture of the same
  configuration. The benchmark's own range is 0.36-0.73 us.
- One capture per configuration. The stall figures agree across two lane
  counts, which is the only repetition here.
- Nothing here measures the host code inside the four non-submission stalls;
  it identifies which syncs bracket them, not what runs between.

## Follow-ups

- If the 4.8% is wanted: spec device-resident accept/fold against ADR 0021,
  and size it before building it.
- The drafter's proposal head is 820 us of every round to produce 7 columns of
  logits that are immediately reduced to 16 candidates each. Whether a
  narrower head — or a head over a candidate subset — is admissible is a model
  question nobody has asked.
