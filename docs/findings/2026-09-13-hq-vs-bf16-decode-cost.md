# hq-e8-2b costs decode speed against BF16, and the card is neither throttled nor bandwidth-bound

- Kind: experiment
- Status: current
- Observed: 2026-09-13
- Last verified: 2026-09-13
- Scope: runtime / KV format, decode throughput, GPU utilization
- Related: [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md), [ADR 0020](../adr/0020-batch-wide-decode-round.md), [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md), [hq attention route agreement](2026-09-12-hq-attention-route-agreement.md), [ITL lane terminator race](2026-09-13-itl-lane-terminator-race.md)
- Superseded by: none

## Question

During a G3 smoke run the RTX 5090 sat at high reported utilization but low
temperature, which the project owner read as poor utilization. Is the card
actually limited by anything, and does the hq-e8-2b KV format cost decode speed
against BF16?

## Evidence

Four `ignis-bench g3` runs against `ignis-server` on the same machine within one
hour, same artifact (`qwen3_8_27b_nvfp4full-v2.ninfer`), same corpus
(`bench_corpus.ids`), release build with `--features cuda`, `ninfer-serve` not
running. GPU telemetry sampled at 1 Hz throughout
(`nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu`).

| run | KV format | max-context | pool pages | C=1 tok/s | C=4 tok/s | ITL p50 | ITL p95 | ITL max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `g3-power-hq.json` | hq-e8-2b | 262,144 | 7,281 | 51.0 | 25.4 | 183.41 | 248.04 | 582.53 |
| `g3-power-hq1024.json` | hq-e8-2b | 40,960 | 1,024 | 54.2 | 27.0 | 183.49 | 248.16 | 264.22 |
| `g3-power-bf16.json` | BF16 | 40,960 | 1,024 | 61.2 | 40.1 | 164.70 | 214.73 | 230.10 |

The first two differ only in KV pool geometry; the last two differ only in KV
format. Every run covered 10 of 10 prefill windows with all four lanes ending at
the measurement boundary.

GPU telemetry over each run's ITL cell (~75-85 s of the ~170 s run):

| run | util | mem-util | power (mean / max) | SM clock | temp |
|---|---:|---:|---:|---:|---:|
| hq-e8-2b, 7,281 pages | 87% | 22% | 284.9 W / 355.3 W | 1,996 MHz | 49 °C |
| hq-e8-2b, 1,024 pages | 96% | 25% | 306.1 W / 354.7 W | 1,996 MHz | 49 °C |
| BF16, 1,024 pages | 87% | 27% | 300.5 W / 361.3 W | 1,995 MHz | 49 °C |
| idle, no model loaded | 14% | 1% | 67.8 W | 2,010 MHz | 37 °C |

The card's nominal boost is 2,010 MHz and its board limit ~575 W.

## Finding

**Observed — the KV pool's page count does not move the ITL cell.** Holding the
format at hq-e8-2b and shrinking the pool from 7,281 pages to 1,024 changes p50
by 0.04% (183.41 to 183.49 ms) and p95 by 0.05% (248.04 to 248.16 ms). The
larger pool's one visible cost is the `max` interval, 582.53 ms against 264.22 —
a single stall, not a shift in the distribution.

**Inference.** The page-table size is therefore not a confound in the format
comparison, which was the reason this run was made: an earlier BF16 comparison
had differed in both format and pool geometry at once.

**Observed — BF16 is faster than hq-e8-2b at identical geometry** (1,024 pages,
`--max-context 40960`): C=1 +12.9%, C=4 +48.5%, ITL p50 -10.2%, p95 -13.5%. Per
lane, BF16 decoded ~924 tokens in a 68.2 s window against hq's ~873 in 76.0 s,
about 18% more tokens per second.

**Observed — nothing is limiting the card.** SM clocks hold ~1,996 MHz against a
2,010 MHz nominal boost under every load, and temperature peaks at 49 °C. Mean
power during the ITL cell is 285-306 W against a ~575 W board limit.

**Observed — the memory subsystem is idle roughly three quarters of the time**
(`utilization.memory` 22-27%) while `utilization.gpu` reports 87-96%.

**Inference.** A decode round rereads the model's weights, so a decode loop that
was bandwidth-bound would keep the memory subsystem near saturation. It does
not. High `utilization.gpu` with low `utilization.memory` means kernels are
resident but not moving data — time spent in small launches, latency, and
synchronization rather than in either compute or bandwidth. The low temperature
is a consequence of that, not an independent symptom.

**Inference — hq trades SM cycles for footprint, and on this card the trade is
losing.** At matched geometry hq keeps the SMs busier than BF16 (96% against
87%) while moving *less* memory (25% against 27%) and producing *fewer* tokens.
Occupancy that does not become output is the signature of the dequantization and
attention-route work hq adds.

**Observed — the ITL cell is the most reproducible of the three.** Across the two
hq runs its p50 and p95 agree to 0.05%, where C=1 and C=4 move 6% between the
same pair.

## Implications

- The ~14% gap between this session's first smoke run and the records committed
  on 2026-09-12 closes as the machine settles: C=1 went 46.6 → 51.3 → 51.0 →
  54.2 against a committed 55.6, and ITL p95 276.2 → 264.0 → 248.0 → 248.2
  against a committed 241.8. There is no evidence here of a regression in the
  tree; a first run after an idle period should not be treated as a measurement.
- hq-e8-2b is not replaceable by BF16: at a 4 GiB budget BF16 holds 65,536
  tokens in total, which does not fit one G4 needle cell at 128K, let alone N=8
  concurrency (see [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md)).
  The finding is that hq's decode path is worth optimizing, not that the format
  is wrong.
- Roughly half the board's power envelope is unused at every configuration
  measured, so the headroom for that optimization is real rather than notional.
- A G3 gate leg should not be run on a cold machine, and a leg whose C=1 sits
  well under its own recent history should be repeated before it is pooled.

## Limits and unknowns

- `utilization.memory` is a duty-cycle sample — the fraction of time any memory
  access is active — not achieved bandwidth. It supports "not bandwidth-bound"
  as a strong hint; proving it needs DRAM-throughput counters from Nsight
  Compute or CUPTI, which were not collected.
- Where the non-memory time actually goes was not measured. Launch overhead,
  synchronization and dequantization are hypotheses consistent with the
  occupancy/bandwidth split, not observations.
- One run per configuration. C=4 in particular is known to swing 26% between
  launches of one binary (`ignis-g3-launch1/2`), so its +48.5% carries much less
  weight than the ITL percentiles, which repeated to 0.05%.
- Nothing here is live/live: no reference-engine run shares these sessions, so
  none of these numbers can decide a gate (ADR 0015).
- Only the 27B NVFP4 artifact on one RTX 5090 was measured.

## Follow-ups

- Profile one decode round under Nsight Compute to convert the occupancy /
  bandwidth split into a named cost.
- Repeat the matched A/B once per configuration to put an error bar on C=4.
- Sample the same telemetry during a `ninfer-serve` run on the same cells: the
  reference is the only available answer to "is 300 W what this model costs on
  this card, or is it what our decode loop costs".
