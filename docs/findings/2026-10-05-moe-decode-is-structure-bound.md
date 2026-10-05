# Flash-Next's one-launch MoE decode is bound by its structure, not by DRAM

- Kind: experiment
- Status: current
- Observed: 2026-10-05
- Last verified: 2026-10-05
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

Acceptance 6 of spec flash-next/02 stays open; the bench's `--check-floor` is the check.
