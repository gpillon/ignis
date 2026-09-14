# The hq-e8-2b decode penalty measured live/live: both engines pay 14% of ITL p95 for the format, and ignis is 3.1x the reference at C=4 under it

- Kind: experiment
- Status: current
- Observed: 2026-09-13
- Last verified: 2026-09-13
- Scope: runtime / KV format, decode throughput, GPU utilization
- Related: [ADR 0015](../adr/0015-g2-live-live-cold-prefix-gate.md), [ADR 0021](../adr/0021-live-live-launch-pooling.md), [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md), [hq vs BF16 decode cost](2026-09-13-hq-vs-bf16-decode-cost.md) (the same question, measured in separate sessions), [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md), [ITL lane terminator race](2026-09-13-itl-lane-terminator-race.md), [GitHub #65](https://github.com/gpillon/ignis/issues/65)
- Superseded by: none

## Question

[hq vs BF16 decode cost](2026-09-13-hq-vs-bf16-decode-cost.md) answered what the
hq-e8-2b KV format costs and whose cost it is, but every leg of it was measured
in a session of its own: its own "Limits" section records that none of its
engine-to-engine numbers may decide anything, because ADR 0015 requires both
engines measured in one session and ADR 0021 requires two process launches per
side. Does the result survive being measured the way a gate is measured?

## Evidence

Eight `ignis-bench g3` legs in **one session** (`kvab-20260913T194359Z`), on a
free RTX 5090 with the other engine stopped for each leg: **two independent
process launches for each of the four (engine, format) cells**. Same artifact
(`qwen3_8_27b_nvfp4full-v2.ninfer`), same corpus (`bench_corpus.ids`), same
release `ignis-bench` build, same flags but the format: `--max-context 40960`,
`--prefill-chunk 1024`, CUDA graphs and prefix reuse on.

**Matched KV capacity, verified per launch.** Every leg resolved 65,536 tokens /
1,024 pages — ignis hq via `--kv-pool-bytes 576M`, ignis BF16 via its 4 GiB
default, the reference via `--kv-capacity 65536` — so the format comparison is
not confounded with page count. GPU telemetry sampled at 1 Hz per leg, started
with the cell and stopped the moment it exited.

Round 1 ran the cells forward (ignis hq, ninfer hq, ignis BF16, ninfer BF16) and
round 2 reversed them, so each format's two launches sit at slots 1-2 and 7-8
(hq) or 3-4 and 5-6 (BF16) of the session — mean slot 4.5 for both.

Records and captures: `.scratch/g4-run2/kvab/`.

| engine / format | C=1 tok/s | C=4 tok/s | ITL p50 | ITL p95 | the two launches' own C=1 |
|---|---:|---:|---:|---:|---|
| ignis hq-e8-2b | 54.2 | 29.4 | 194.38 | 262.81 | 55.8 / 52.5 |
| ignis BF16 | 58.6 | 38.2 | 169.08 | 224.78 | 59.8 / 57.5 |
| ninfer hq-e8-2b | 51.3 | 9.6 | 167.56 | 254.80 | 54.7 / 48.0 |
| ninfer BF16 | 57.8 | 19.7 | 152.18 | 218.33 | 57.0 / 58.6 |

Cell values are the mean of that cell's two launches; the last column is the
per-launch spread ADR 0021 exists for.

GPU telemetry over each leg's own ITL window:

| leg | util | mem-util | power (mean / max) | SM clock | temp |
|---|---:|---:|---:|---:|---:|
| ignis hq L1 / L2 | 98% / 99% | **26% / 25%** | 321.7 / 293.5 W | 1,996 / 1,992 MHz | 53 / 51 °C |
| ignis BF16 L1 / L2 | 98% / 98% | **30% / 31%** | 331.0 / 321.4 W | 1,994 / 1,993 MHz | 54 / 52 °C |
| ninfer hq L1 / L2 | 99% / 98% | **36% / 31%** | 295.4 / 291.7 W | 1,996 / 1,996 MHz | 55 / 51 °C |
| ninfer BF16 L1 / L2 | 98% / 98% | **40% / 44%** | 309.8 / 301.1 W | 1,995 / 1,996 MHz | 54 / 53 °C |

## Finding

**Observed — BF16 beats hq-e8-2b on both engines, and the ITL p95 cost agrees to
0.2 points.** Going hq to BF16 at identical geometry:

| | C=1 | C=4 | ITL p50 | ITL p95 |
|---|---:|---:|---:|---:|
| ignis | +8.3% | +30.0% | -13.0% | **-14.5%** |
| ninfer | +12.6% | +105.9% | -9.2% | **-14.3%** |

**Inference.** Two independently written engines paying 14.5% and 14.3% of ITL
p95 for the same format change is the format's cost, not either
implementation's. This reproduces the separate-session result (13.5% and 13.0%)
under the measurement discipline a gate uses, so it is no longer a number that
may not decide anything. Optimizing ignis's hq route can at best reach the
reference's hq route; it cannot recover that 14%.

**Observed — ignis is ahead of the reference at every cell but ITL, in both
formats.** At matched format and capacity, ignis over ninfer is C=1 **1.056**
(hq) and **1.015** (BF16); C=4 **3.074** and **1.941**; ITL p95 **1.031** and
**1.030**.

**Inference.** The C=4 advantage is not an hq artifact — it is 1.9x under BF16
too — and the ITL p95 deficit is not one either: 1.031 and 1.030 are the same
number in both formats. Whatever ignis pays on the decode tail, it pays
independently of the KV format.

**Observed — ignis moves less memory than the reference in both formats.**
`utilization.memory` is 25-26% for ignis against 31-36% for the reference under
hq, and 30-31% against 40-44% under BF16, at comparable power.

**Inference.** This reproduces the earlier session's gap under live/live
conditions. Where ignis's non-memory time goes is still unmeasured, but the gap
is a property of the two engines rather than of two sessions.

**Observed — nothing is limiting the card.** SM clocks hold 1,992-1,996 MHz
against a 2,010 MHz nominal boost in all eight legs, temperature peaks at 55 °C,
and mean power is 292-331 W against a ~575 W board limit.

**Observed — the hq cells' launch-to-launch spread is the larger one.** Across a
cell's two launches, ignis hq moved 11.7% on ITL p95 and 13.6% on C=4, and
ninfer hq 9.2% on C=1, while both BF16 cells held within 1.2-3.6% on everything
except ninfer's C=4 (14.9%). Both engines' *second* hq launch was the slower one.

**Inference.** ADR 0021's launch band is real here and is wider under hq than
under BF16 in this session. Two launches per cell is the floor, not a
comfortable margin.

## Implications

- The KV-format inequality that has sat beside every gate verdict since G2 is
  now measured rather than noted, under the discipline gates use: hq costs ~14%
  of ITL p95 on both engines, and ignis is not the reason.
- hq-e8-2b stays the serving default regardless: at a 4 GiB budget BF16 holds
  65,536 tokens in total, which does not fit one 128K needle cell, let alone
  N=8 (see [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md)).
- ITL p95 ratio 1.03 in **both** formats is the one cell where ignis is
  consistently behind, and it is now format-independent — which points the next
  optimization at the decode round's own structure rather than at the codec.
- The memory-idleness gap survives live/live measurement, so the follow-up it
  implies (profile ignis's decode round against the reference's for launch count
  and synchronization points) stands on firmer evidence than before.

## Limits and unknowns

- Two launches per cell. That meets ADR 0021's floor and no more; the hq cells'
  11-14% across-launch spread is larger than several of the differences reported
  above, and only the ITL p95 format cost (-14.5% / -14.3%) sits comfortably
  outside it.
- `utilization.memory` is a duty-cycle sample, not achieved bandwidth. It
  supports "not bandwidth-bound" as a strong hint; proving it needs DRAM
  counters from Nsight Compute or CUPTI, which were not collected.
- The reference's ITL cell covered 5-10 of 10 prefill windows depending on the
  leg, against ignis's 10 of 10 in all four, because its lane 0 can stop on its
  own EOS ([ITL lane terminator race](2026-09-13-itl-lane-terminator-race.md)).
  The percentiles on those legs are pooled over less of the series.
- One artifact, one card, one session. Nothing here says anything about other
  geometries or other hardware.

## Follow-ups

- Profile one decode round under Nsight Compute to convert the occupancy /
  bandwidth split into a named cost.
- Profile ignis's decode round against the reference's for launch count and
  synchronization points — the format-independent ITL p95 1.03 makes that the
  remaining unexplained cell.
