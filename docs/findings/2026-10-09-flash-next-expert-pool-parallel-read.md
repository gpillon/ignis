# Parallel reads cut Flash-Next's expert pool fill in half

- Kind: experiment
- Status: current
- Observed: 2026-10-09
- Last verified: 2026-10-09
- Scope: core / Flash-Next host expert pool fill at load (`residency::load::fill_expert_pool`, spec flash-next/03 acceptance 1, GitHub #301); serving / Flash-Next load time
- Related: [early unmap and n-gram cache](2026-10-07-flash-next-startup-early-unmap.md)
- Superseded by: none

## Question

After the early-unmap and n-gram persistence fix (112 s -> 42-46 s cache-hit
load), the dominant remaining phase was the host expert pool fill: one file
handle, synchronous `read_exact` calls, layer by layer. Does splitting that
read across a few handles in parallel, on the real artifact, actually buy
anything on this drive, and where is the ceiling?

## Evidence

Branch `experiment/flash-next-expert-pool-parallel-read` (no worktree,
owner's call), local to this checkout. Probe archived at
`.scratch/expert-pool-fill-bench/expert_pool_fill_bench.rs` (never a
tracked Cargo example — the prior startup finding set that convention).

Real artifact `F:/ai/models/Qwen3.8-Flash-Next-ignis/qwen3_8_flash_next_trellis_a25-v2.ninfer`
(71.76 GB file). CPU/disk only: `flash_next::bind` with
`FlashNextGeometry::qwen38_flash_next()` gives the real `ExpertIndex`, no
CUDA touched. The probe fills a fresh `vec![0u8; pool_bytes]` from a
sequential baseline (one handle, the original algorithm) and from a
parallel variant (N handles, layers round-robined, each worker reading its
own disjoint byte range of the pool). Host: i9-10900K, 20 logical CPUs,
model on an NVMe drive (`F:`), PCIe Gen 3 (see
[host PCIe Gen 3](ignis-host-pcie-gen3.md) in memory — unrelated to this
read path, same drive).

Pool size: 48 layers x 512 experts/layer, **35.20 GiB** (37,795,446,784
bytes) — the host pool holds every expert for the life of the load, not a
budget-limited subset (`residency::mod` doc: "all of them in one pinned
host pool").

| Workers | Time (s) | Throughput (MiB/s) |
|---:|---:|---:|
| 1 | 29.327 | 1229.1 |
| 2 | 16.530 | 2180.5 |
| 4 | 13.526 | 2664.8 |
| 8 | 15.842 | 2275.2 |
| 16 | 16.542 | 2179.0 |

Second pass, narrower range, workers in run order 3, 5, 6, 4, 1 (checks the
first pass's shape isn't an artifact of ascending order / page-cache warm-up):

| Workers | Time (s) | Throughput (MiB/s) |
|---:|---:|---:|
| 3 | 13.918 | 2589.8 |
| 5 | 12.420 | 2902.1 |
| 6 | 12.983 | 2776.3 |
| 4 | 14.965 | 2408.6 |
| 1 (re-run last) | 27.032 | 1333.4 |

The re-run single-worker baseline (27.0 s) matches the first pass's (29.3 s)
even though it ran last and should have benefited most from any page-cache
warm-up: the gain is from concurrent I/O, not caching.

## Finding

One handle holds this drive to ~1.2-1.3 GiB/s. From 3 to 6 concurrent
handles throughput plateaus at ~2.6-2.9 GiB/s (2.1-2.4x); 8 and 16 regress
slightly from contention. The decision is flat across 3-6, so pick a small
fixed worker count rather than keep measuring it (owner: stop measuring
once the decision is clear).

## Implications

`fill_expert_pool` now takes the artifact `&Path` instead of one open
handle: it computes each layer's disjoint `[lo, hi)` byte range in the pool
(returning `Err` if the index ever produced overlapping ranges — external
artifact data), carves the pool into per-layer `&mut [u8]` slices with
`split_at_mut` (no unsafe aliasing), round-robins the layers across
`READ_WORKERS = 4` worker threads, and opens one `File` handle per worker.
The single call site (`flash_next::build_residency`, reached by the
server's `FlashNextLeaf::load`) no longer opens its own handle. Chosen
default per the tables above: 4 is mid-plateau and simple; the repo adds no
CLI flag or `EngineOptions` plumbing for it (the ceiling is the drive, not
a workload a user would tune).

## Limits and unknowns

Two passes, one machine, one drive, no reboot or forced page-cache purge
(the baseline's stability across passes argues this didn't matter here).
Only this artifact's K-class/layer layout was exercised; a differently
shaped index (very few very large layers) would see less benefit from
round-robining by layer. Windows/NTFS only.

## Follow-ups

Owner's live model-swap idea wants progressive load next (serve before the
host pool is full, stream the rest in the background) — this parallel fill
shortens that background fill too, independent of when serving starts.
Separately, the n-gram cache-hit phase (6.8 s for a 1.03 GB file, ~150
MiB/s) is single-threaded and looks like a whole-payload SHA-256 on one
core against a drive doing 1.3 GiB/s elsewhere; not measured here.

## Real server verification

Built `ignis-server` release with `--features cuda` on this branch.
`MODEL=flash-next make start` (never piped — daemon holds the handle),
default Make settings (context 262144, prefill 8192, lanes 3), n-gram cache
path `model` (default), a prior run had already written the cache file
beside the artifact so this was a cache **hit**. GPU checked free
(`make gpu-status`) immediately before.

Log timestamps, `.scratch/serve/ignis-server.log`: first line (artifact
verified) `14:51:48.884`; `ignis.runtime.flash_next_host_plan` (just before
the expert pool fill starts) `14:51:49.531`; n-gram cache hit logged
`14:52:20.007` with `duration_ms: 5524` (so the n-gram phase itself started
`14:52:14.483`); `ignis.model.loaded` `14:52:20.660`; `ignis.process.ready`
`14:52:20.787` (`warm_up_ms: 106`).

**Total process start to API-ready: ~31.9 s.** The segment between the
host plan log and the n-gram phase's start (device weights + expert
allocation + the parallel expert pool fill + warm start) is ~25.0 s,
against the pre-fix finding's recorded 0.896 + 2.430 + 3.599 + 28.814 +
1.656 = 37.4 s for the same four sub-phases — consistent with the
isolated benchmark's ~2x fill speedup plus the other three subphases'
original cost. The prior finding's equivalent real-server check (same
n-gram-cache-hit condition, different Make defaults: context 131072,
prefill 8192) measured 46.4 s to `/v1/models` readiness; this run's ~31.9 s
is about 31% faster end to end.

The server was stopped (owner's call, mid-session: it was already serving
live chat traffic through the Playground when this check ran) with
`make stop` before any further work. Workspace tests: `cargo test
--workspace --release -j 4`, all green (log:
`.scratch/expert-pool-fill-bench/workspace-test.log`).
