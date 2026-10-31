# NVML counts a fresh allocation late: the free reading right after cudaMalloc of 1 GiB missed 64 to 448 MiB in 6 of 20 allocations on the 5090, and a reading 27 to 63 ms later had all of it

- Kind: experiment
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: serving / `CudaDevice::nvml_memory`, the `ignis.runtime.vram_loaded` log line, `crates/artifact/tests/nvml_memory_gpu.rs`; on the RTX 5090 (Windows/WDDM)
- Related: https://github.com/gpillon/ignis/issues/424, https://github.com/gpillon/ignis/issues/210, https://github.com/gpillon/ignis/issues/324
- Superseded by: none

## Question

`nvml_free_memory_counts_the_context_and_every_allocation` failed about one run in three on a quiet 5090. The issue guessed the desktop's drift between two NVML readings. Is it drift, or does NVML's free reading right after an allocation not yet hold the whole allocation?

## Evidence

**Setup.** The 5090 as the only GPU process, `ignis-artifact` built with `--features cuda`, `CudaDevice::create(0)` then 1 GiB allocations and frees on one device, and `CudaDevice::nvml_memory(0)` (the free reading #210 pins) beside the per-process WDDM counter (`own`). Raw data: the run's `nvml-probe-1`, `nvml-lag-1`, `nvml-before` and `nvml-after` logs (kept in `.scratch/424/` of the work, untracked).

**The failing test, 20 runs each.** Before the change, 6 of 20 runs failed at the allocation's delta assertion. After the test waits for the count (up to 2 s), 20 of 20 passed.

**Probe 1: ten 1 GiB allocations, a reading right after each, a second reading after it.**
- The first reading missed part of the allocation in 4 of 10: by 128 MiB, 64 MiB, 448 MiB and 256 MiB (each short by a multiple of 64 MiB, plus at most 0.6 MB of unrelated drift).
- The second reading counted the whole 1 GiB in all 10, to within 1 MB, and `own` (the WDDM counter) read the whole allocation every time.
- One NVML read took 8 to 45 ms, the allocation call 1.5 to 17 ms.
- Between the two readings taken before the allocation the free memory moved by at most 602,112 bytes in the ten, so under 1 MB: the desktop did not move memory in a way that explains a missing 64 to 448 MiB.

**Probe 2: ten allocations, NVML polled with a 1 ms pause until it settled.**
- The first reading was short in 2 of 10: by 64 MiB (the rest arrived in one step 27.2 ms after the first reading) and by 128 MiB (in one step, 62.9 ms after).
- Never more than one step later; the allocation call took 1.5 to 3.2 ms.
- The ten frees were counted in full by the first reading (10 of 10). Two frees later moved by 128 KiB and 256 KiB, 154 ms and 1,164 ms after: desktop-sized.

**The context.** After `CudaDevice::create(0)` NVML's free memory fell by 452,161,536 bytes and `own` rose by 452,325,376, within 0.2 MB. After dropping the device NVML gave back 450,543,616 bytes and `own` 452,595,712.

## Finding

Observed:
- Under WDDM on the 5090, NVML's free reading right after `cudaMalloc` of 1 GiB sometimes lacks the last 64 MiB pieces of it (6 of 20 allocations in the two runs above, 64 to 448 MiB); a later reading has all of it, and the late part arrived in one step 27 and 63 ms after, where timed.
- Frees were counted at once (10 of 10).
- The per-process WDDM counter counted every allocation at once.
- NVML and that counter agree on the context.
- The desktop moved under 1 MB in the probes.

Inferred:
- NVML counts WDDM allocations in 64 MiB pieces that can land after the allocation call returns; the evidence shows the lag, not its mechanism.

## Implications

- The test that reads NVML after an allocation must wait for the count. `nvml_memory_gpu.rs` does, for up to 2 s (about 30 times the longest lag seen), and the wait hands back NVML's own reading on a timeout, so a count that never comes still fails.
- One production read touches this: the `ignis.runtime.vram_loaded` line's `free_memory_delta_bytes` (`free_at_start_bytes` less the free reading after load). If that reading lands within tens of ms of the load's last allocation, the delta can be short of what the process holds. The line is a log beside the plan, not an input to it.
- The VRAM plan reads free memory before the context exists, so it is not affected.

## Limits and unknowns

- Two runs of ten 1 GiB allocations on one machine and driver; sizes other than 1 GiB, other cards and the driver's version dependence are untested.
- Whether a load's own last allocation shows the lag, and how often, was not measured.
- The 2 s bound is not a measured maximum: the longest lag seen is 63 ms.

## Follow-ups

None filed. If `free_memory_delta_bytes` is ever used to check the plan, take it from the per-process counter or wait for the count first.
