# An SM-driven copy from mapped pinned memory matches the copy engine on this host

- Kind: experiment
- Status: current
- Observed: 2026-10-05
- Last verified: 2026-10-05
- Scope: kernel / Flash-Next expert residency, the miss path from the pinned host expert pool to the VRAM expert cache
- Related: [GitHub #301](https://github.com/gpillon/ignis/issues/301),
  [spec flash-next/03](../specs/flash-next/03-expert-residency.md),
  [ADR 0044](../adr/0044-experts-in-trellis-with-a-k-per-expert.md),
  [Sequence snapshot transfer cost](2026-09-12-sequence-snapshot-transfer-cost.md)
- Superseded by: none

## Question

Spec flash-next/03 keeps every Flash-Next expert projection in one pinned
host pool and a cache of them in VRAM. A projection the router selects that
is not resident has to be copied in before the expert kernel runs, and there
are two ways to do it:

- **copy engine:** `cudaMemcpyAsync` (or `cudaMemcpyBatchAsync`). The host
  must issue it, so the host must first know the miss: the router's selection
  is read back every layer and the decode graph is split around it
  (host-orchestrated residency);
- **SM-driven:** a kernel whose threads load from the pool mapped into the
  device address space and store into the slot. A device-side LRU can launch
  it with no host in the loop, so decode stays one CUDA graph
  (device-resident residency).

The spec's rule: device-resident only if the SM-driven copy reaches at least
80% of the copy engine's bandwidth on the transfers residency makes — one
projection is 0.42-1.65 MB (spec 01's eight K classes), the spec's range is
1-3 MB — measured on this machine under WDDM.

## Evidence

`kernel/tests/bench_expert_miss_path.cu` (a tool, not a CTest test; CUDA
runtime only). Each cell copies `n` projections of `S` bytes from random
4 KiB-aligned places in a 4 GiB pinned, mapped pool into `n` device slots;
every rep uses a fresh list, so no rep rereads what L2 kept from the last.
Times are device time between events, median of 15 reps. The SM-driven copy
is one launch per list, a grid-stride walk over 16-byte vectors, swept over
8-680 blocks × 256/512 threads × 1/4/8 loads in flight per thread; the
correctness of its copy was checked byte for byte.

RTX 5090 (170 SMs, one async copy engine reported), WDDM, PCIe Gen 3 x16 (the
host's cap, see the snapshot-transfer finding). Two consecutive full runs,
2026-10-05, nothing else on the card. Raw output:
`.scratch/residency/miss-path-run{1,2}.md` in the flash-next worktree.

Best of each path, GB/s, run 1 / run 2 (`ce` = best of 1, 2, 4 streams and
`cudaMemcpyBatchAsync`; `sm ≤16` = the best grid of at most 16 blocks):

| S | n | ce | sm (any grid) | sm ≤16 | sm ≤16 / ce |
|---:|---:|---:|---:|---:|---:|
| 0.5 MiB | 1 | 9.80 / 10.74 | 10.11 / 9.97 | 9.62 / 9.65 | 98% / 90% |
| 0.5 MiB | 20 | 11.28 / 11.12 | 12.16 / 11.86 | 12.16 / 11.36 | 108% / 102% |
| 1 MiB | 1 | 11.68 / 12.05 | 11.20 / 11.19 | 11.09 / 11.07 | 95% / 92% |
| 1 MiB | 20 | 11.45 / 11.62 | 12.38 / 12.23 | 12.38 / 12.23 | 108% / 105% |
| 1 MiB | 32 | 11.04 / 12.08 | 12.09 / 12.29 | 12.09 / 12.29 | 110% / 102% |
| 1.5 MiB | 20 | 12.17 / 13.05 | 12.35 / 12.01 | 12.20 / 12.01 | 100% / 92% |
| 2 MiB | 20 | 12.44 / 12.73 | 12.49 / 11.15 | 12.40 / 11.15 | 100% / 88% |
| 3 MiB | 32 | 12.39 / 13.11 | 12.28 / 11.70 | 12.28 / 11.70 | 99% / 89% |
| 3 MiB | 64 | 12.32 / 12.99 | 11.88 / 11.69 | 11.88 / 11.69 | 96% / 90% |

Over all twenty cells of both runs (0.5, 1, 1.5, 2, 3 MiB × n = 1, 20, 32,
64) the SM-driven copy is **88-112%** of the copy engine, and so is the best
grid of at most 16 blocks.

The copy engine itself: 9.8-13.1 GB/s for each cell's best configuration,
12-13 GB/s from 1.5 MiB up, whatever the stream count; `cudaMemcpyBatchAsync`
is no faster than a loop of `cudaMemcpyAsync` on one stream.

How the SM-driven copy scales with the grid (n = 20 × 1 MiB, run 1):

| blocks × threads, 1 load in flight | GB/s |
|---|---:|
| 8 × 256 / 8 × 512 | 11.06 / 11.65 |
| 64 × 512 | 11.94 |
| 170 × 256 / 170 × 512 | 10.73 / 7.21 |
| 340-680 × 256-512 (any unroll) | 5.1-7.9 |

More loads in flight per thread (unroll 4, 8) never helped and mostly hurt.

Write-combined host memory makes no difference (12.41 against 12.40 GB/s).

What a host-orchestrated layer pays before its first copy — a trivial kernel,
a 120-byte readback (three lanes' selections) and a stream sync, host clock:
**57.0 / 57.4 us**, which is **2.7 ms per token** over 48 MoE layers.

## Finding

**Observed.** On this host an SM-driven copy from mapped pinned memory moves
expert-sized transfers at 88-112% of the copy engine's bandwidth, with 8-16
blocks. Both are capped by the Gen 3 x16 link at ~12-13 GB/s, so neither path
has bandwidth the other lacks.

**Observed.** The SM-driven copy is fastest with *few* blocks: 8-64 blocks
saturate the link, while grids of 340-680 blocks are 34-57% slower, and so
are most configurations with 4-8 loads in flight per thread. The link, not the SMs, is the resource,
and flooding it with outstanding reads costs bandwidth.

**Decision (spec 03's 80% rule).** **Device-resident residency.** A small
kernel after the router resolves hits and misses against a device LRU table
and writes the slot table; a copy kernel of about 8-16 blocks moves the
missing projections from the mapped host pool; the expert kernel follows on
the same stream. Decode residency is graph-capturable.

**Inference.** The rule would have chosen the same even at a margin: the
alternative pays a 57 us host round trip per layer, 2.7 ms per token, which
is about as much as the whole simulated residency cost at one lane (3.8 ms,
`PLACEMENT.md` of the compression study) and would come on top of it.

## Implications

- The host expert pool is allocated mapped (`cudaHostAllocMapped`) so the
  copy kernel can read it. The expert kernels still read device slots only
  (spec 02).
- The copy kernel should launch with a small grid (start at 8-16 blocks ×
  256 threads, one 16-byte load in flight per thread). That also leaves
  ~150 SMs to the expert kernel a prefetch runs beside.
- Prefill's scan-resistant stream (spec 03) goes through the same copy
  kernel into the staging ring; at ~12 GB/s a cold 2048-token chunk's ~30 GB
  is ~2.5 s, the bound acceptance 6 measures against.
- The copy engine stays free for everything else that crosses PCIe (KV-RAM
  spills, retained host slots).

## Limits and unknowns

- Each cell ran its copy alone. How much an SM-driven prefetch slows the
  expert kernel it runs beside (and the reverse) is not measured here; it is
  part of spec 03's trace replay with the real kernels.
- 4 GiB of the pool, not 37.7 GB: a pinned allocation of the full size under
  WDDM is still to be confirmed at load (the study reports one 38 GB
  allocation working).
- One machine, one driver; the ratio is only meaningful while both paths are
  link-bound. On a Gen 4/5 host the copy engine may pull ahead and the rule
  would need re-measuring.

## Follow-ups

- Spec 03's GPU side: the device LRU table, the copy kernel and the trace
  replay against the CPU policy model (`crates/core/src/residency/`).
