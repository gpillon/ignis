# Flash-Next's one-launch MoE decode is bound by its structure, not by DRAM

- Kind: experiment
- Status: current
- Observed: 2026-10-05
- Last verified: 2026-10-06
- Scope: kernel / Flash-Next routed-expert decode (`kernel/src/moe_decode.cu`)
- Related: [GitHub #300](https://github.com/gpillon/ignis/issues/300),
  [spec flash-next/02](../specs/flash-next/02-the-moe-kernels.md) (Acceptance 6),
  [ADR 0044](../adr/0044-experts-in-trellis-with-a-k-per-expert.md)
- Superseded by: none

## Question

Spec flash-next/02 asks the routed-expert decode launch for one token to read its bytes (the
ten selected experts' trellis records, ~15.4 MB at the study's K mix) at 50% or more of the
DRAM roofline. What limits it?

## Evidence

`kernel/build/tests/ignis_kernel_moe_bench.exe` (source `kernel/tests/bench_moe.cu`) on the
RTX 5090: device time per call from a CUDA graph of 256-320 calls, every call routed to experts
the previous calls did not touch (a 320-expert pool, ~480 MB, five times the 96 MB L2), study K
mix (run 8, layer 1). Roofline 1,792 GB/s from the device attributes; a plain streaming read
measured 1,531-1,677 GB/s in the same runs.

| decode kernel | 1 token | 2 tokens | 3 tokens | L2-resident, 1 token | all K=2 / all K=4, 1 token |
|---|---|---|---|---|---|
| first (6f69558): 255 regs, 1 CTA/SM, loads in 5 dependent groups | 31.5 us, 27.2% | 34.3% | 35.6% | — | — |
| every load of a unit in flight, 128 regs, 2 CTAs/SM (5fae17c) | 26.2 us, 32.7% | 41.6% | 41.6% | 21.7 us | 24.1 / 33.0 us |
| + next ticket taken during the unit (76cff4c) | 25.6 us, 33.5% | 42.2% | 48.3% | 21.1 us | 22.5 / 28.3 us |
| variant: 80 regs, 3 CTAs/SM, rotation scale from a norm bound | 35.6 us | — | — | 29.9 us | 33.6 / 34.2 us |
| variant: 128 regs, 2 CTAs/SM, norm-bound scale | 29.5 us | — | — | 23.3 us | 26.7 / 32.6 us |
| tickets again, session 2 (2026-10-06; streaming read 1,390 GB/s that run) | 26.4 us, 32.5% | 38.9% | 45.0% | 24.7 us | 23.9 / 31.3 us |
| clusters: one 8-CTA thread-block cluster per expert, DSMEM reductions (d6bb40c), same run | 38.3 us, 22.4% | 21.3% | 26.7% | 32.8 us | 36.8 / 40.1 us |

`cuobjdump --dump-resource-usage` / `-sass` on the object: the inner loop is the designed
~5 instructions per weight (SHF, LOP3, IMAD, IDP4A per state, PRMT + HFMA2 per pair), no spills
in the measured versions.

## Finding

- With the ten experts' weights resident in the L2 the launch still takes ~21 us of its 25.6 us,
  and doubling the bytes at the same weight count (all K=2 to all K=4) costs 1.26x the time,
  not 2x. The launch is bound by its structure and issue rate, not by bandwidth.
- The structure: gate/up units (400 at one token) exceed what one wave of resident CTAs holds,
  so ~1.5 experts' gate/up work lands in a second wave and their down units wait a whole unit
  latency more; each unit pays round trips (ticket, activation loads, fence, arrival) that
  dominate when there are ~2 units per CTA.
- Raising occupancy by cutting registers (3 CTAs/SM) made it slower: the compiler sinks the
  weight loads towards their uses under register pressure, and the per-unit fixed latency grows.
- More tokens amortize the structure: three tokens reach 48.3% of the roofline.
- One cluster per expert (plan item 1) is slower, 38.3 us. The card does not co-schedule ten
  16-CTA clusters (`ignis_moe_decode_cluster_size` falls back to the portable 8), so a token's
  ten experts run on 80 CTAs -- about half the 170 SMs -- and each CTA streams an eighth of an
  expert serially; with the weights in the L2 it still takes 32.8 us. Removing the reduction
  stage and the tickets does not pay for losing half the card: at one token the expert count
  (ten) times the cluster size that fits a GPC caps the SMs a cluster-per-expert launch uses.

## Implications

The decode floor needs a different launch shape rather than more bandwidth: fewer dependent
stages per expert (the gate/up reduction folded into the down units, or gate/up and down of one
expert in one CTA pipeline), first-wave placement for every gate/up unit, and less per-unit
synchronization. Per-weight decode cost (~5 integer instructions, half on the half-rate pipe)
puts a compute floor near 5-8 us per token-layer at this card's issue rate, below the 17 us
the 50% floor allows.

## Limits and unknowns

Device time from graph replay; no Nsight profile (nsys launch mode hangs on this Windows host).
The routing in the bench has no expert shared between tokens; real decode with three lanes
shares some, which helps the multi-token figures.

## Follow-ups

Acceptance 6 of spec flash-next/02 stays open; the bench's `--check-floor` is the check. The
ticket route stays the default (`ignis_moe_workspace::decode_route`); the cluster route is kept,
tested on both routes, for the shapes where experts outnumber what one wave of tickets covers.
Plan item 2 (the gate/up reduction folded into the down units, cp.async-staged unit weights)
is the next candidate.

## Attempt 2026-10-06 (opt)

Plan item 2 was v2 (`086d1c5`), which measured 29.8 us and was reverted in `1dbfc2b`. This was a
limited attempt at the other levers: two kernel experiments on the ticket kernel, both measured
in one GPU session with the first ticket kernel (v1, `1dbfc2b`) re-measured alongside them. Logs:
`.scratch/opt/{v1,e1,e2,e2n}-bench.log`. The sources are in `.scratch/opt/{e1f,e2,e2n}/`.

- **E1, fewer round trips per unit.** The call's slots are read into shared memory once. The
  first ticket is taken before routing setup. svh is loaded alongside the other loads, and one
  fence before `h_ready` is dropped. Each warp L2-prefetches its tiles of the CTA's next unit
  while it multiplies.
- **E2, all gate/up work in one wave.** E1 plus, up to four tokens, three gate/up k-splits of
  896/896/768 inputs instead of four of 640, so one token has 300 gate/up units for the 340
  resident CTAs. The extra 8-16 tiles of a wide unit are L2-prefetched and loaded into the
  registers the first tiles free.
- **E2n** is E2 without the next-unit prefetch.

| kernel | 1 token | 2 tokens | 3 tokens | L2-resident, 1 token | streaming read |
|---|---|---|---|---|---|
| v1 (unchanged) | 26.1 us, 32.8% | 38.0 us, 45.1% | 52.5 us, 49.0% | 22.4 us | 1,655 GB/s |
| E1 | 28.5 us, 30.0% | 38.0 us, 45.1% | 54.5 us, 47.2% | 23.0 us | 1,560 GB/s |
| E2 | 30.1 us, 28.5% | 43.3 us, 39.7% | 56.6 us, 45.5% | 21.9 us | 1,468 GB/s |
| E2n | 29.8 us, 28.7% | 40.6 us, 42.2% | 53.9 us, 47.7% | 22.3 us | 1,655 GB/s |

Every MoE/trellis/fp8 CTest passed on E1 (17/17). Neither experiment is faster than v1, so
both were reverted and v1 stays.

- **The round trips are not where the L2-resident time goes.** Taking them out (E1) leaves the
  L2-resident time where it was: 23.0 us against 22.4 us.
- **One gate/up wave is about even in the L2 and slower from DRAM.** E2 cuts at most 0.5 us with
  the weights in the L2 and is 3.7 us slower with them in DRAM; the L2-resident diagnostic did
  not predict that. The next-unit prefetch is not the cause: E2n (29.8 us) is as slow as E2. The
  cause is not isolated. Candidates are the order in which DRAM delivers a 10 MB first wave
  (v1's second wave streams in while its first wave multiplies), the first wave's units being
  40% longer, and the wider A rows. E1's 2.4 us DRAM-side loss is from one run and is not
  isolated either.
- **The next unit's weights cannot be held in registers.** Refilling each consumed word with the
  next unit's word (40 registers carried across units) compiles badly: ptxas moves the refills
  to the end of the loop and spills 748 B. This variant was not measured.
- **Compare only within one run.** v1's 2- and 3-token figures here are 38.0 / 52.5 us, against
  44.1 / 57.1 us in session 2.

Acceptance 6 stays open. v1's L2-resident 22 us is still unexplained, and per-unit round trips
are not it. The next step is a phase trace of v1 (port 086d1c5's `moe_trace.h`), from DRAM and
from the L2, before another restructuring.
