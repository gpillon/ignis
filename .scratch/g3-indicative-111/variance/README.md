# #116 investigation: per-process-launch throughput variance

Controlled follow-up to the variance first noticed while checking #114's
raised cap (`.scratch/g3-indicative-111/README.md`'s own "instrument varies
between server processes" section, which this refines). Taken 2026-09-10 on
a free RTX 5090, HEAD at the time was `636a5e6` (post-#111/#114/#115).

## Setup

`monitor.ps1` samples `nvidia-smi` (SM/memory clock, pstate, temperature,
power, GPU/memory utilization, throttle-reason bitmask) and the server
process's own CPU%/thread count/working set every ~0.5-1s (actual interval
governed by how long `nvidia-smi` + `Get-Process` take) for the server's
whole lifetime. `start-server.ps1` launches `ignis-server` release with
defaults. Two process launches were monitored:

- **A**: started, sampled once (`run-A.json`), sampled twice more back to
  back (`run-A-r2/r3.json`), then left idle 200s and sampled a fourth time
  (`run-A-r4.json`), then stopped. ~21 minutes total lifetime.
- **B**: started immediately after A was stopped, sampled three times back
  to back (`run-B-r1/r2/r3.json`), then stopped.

## Results

| launch | sample | C=1 tok/s | C=4 tok/s | ITL p95 ms |
|---|---|---:|---:|---:|
| A | r1 | 72.9 | 47.6 | 184.08 |
| A | r2 | 68.7 | 45.4 | 175.35 |
| A | r3 | 72.6 | 47.8 | 178.10 |
| A | r4 (+200s idle) | 73.4 | 48.3 | 182.73 |
| B | r1 | 68.8 | 46.1 | 180.72 |
| B | r2 | 69.5 | 46.1 | 178.58 |
| B | r3 | 69.8 | 46.2 | 178.67 |

Launch A: 68.7-73.4 (6.4% spread), no trend across 21 minutes including the
idle gap -- r4 (after idling) lands with r1/r3, not further from them. Launch
B: 68.8-69.8 (1.4% spread), reached on its *first* sample, immediately after
A stopped.

## What this rules out

- **The code.** Not tested again here (already ruled out in the original
  #116 report: both the pre-#111 and post-#111 trees produced both a fast
  and a slow band). Not re-litigated.
- **Slow drift over a launch's lifetime.** A's own samples span 21 minutes
  including an idle gap and stay together; if the band moved with time this
  would show it moving, and it does not.
- **Heat or clock state carried over from the previous process.** B started
  the instant A stopped and its first sample already sits in B's own band,
  not in A's. If B had inherited A's recent thermal/boost state its first
  reading would resemble A's, not immediately diverge from it.
- **The GPU's own sustained clock.** Restricting each launch's samples to
  moments `nvidia-smi` reported `utilization.gpu > 30%` (real compute
  activity, not idle-between-requests polling):

  | launch | busy-sample SM clock avg | busy-sample SM clock max |
  |---|---:|---:|
  | A | 2,824 MHz | 2,910 MHz |
  | B | 2,823 MHz | 2,910 MHz |

  Identical, within noise. Whatever separates a 73 tok/s launch from a 69
  tok/s launch, it is not the GPU running any faster while it works.
  `clocks_event_reasons.active` was `0x0` throughout both launches -- no
  power, thermal or reliability throttle was ever asserted.

## What is left

Given busy-time GPU clock is the same, the difference has to be in the part
of the per-token critical path that is not raw SM throughput: host-side
kernel-launch dispatch, `cudaStreamSynchronize` wait behavior, OS thread
scheduling for the server's own worker threads, or the async runtime's
wakeup latency. At these rates a decode token is only ~13-15 ms wide, so a
fraction of a millisecond of consistently added host overhead is a few
percent of throughput -- enough to explain the whole spread. This project
has no host/device-correlated profiler (Nsight Systems is the right tool and
is not part of this repo's toolchain) to take the investigation further than
"plausible and consistent with every observation," which is where it stops.

## What this changed

ADR 0021 and `.scratch/runtime/specs/03-serving-loop.md`'s G3 section: a
live/live gate now pools at least two independent process launches per
engine, because a launch, not a request, is the unit this variance attaches
to, and ADR 0015's within-launch repetition cannot cancel a bias that is
constant for a whole launch.
