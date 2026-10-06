# Expert residency replayed on the study's routing: one-LRU pools, a budgeted lookahead

- Kind: experiment
- Status: current
- Observed: 2026-10-05
- Last verified: 2026-10-06 (on the converter's traces)
- Scope: core / Flash-Next expert residency, the CPU policy model and its pool split, prefetch and prefill admission
- Related: [GitHub #301](https://github.com/gpillon/ignis/issues/301),
  [spec flash-next/03](../specs/flash-next/03-expert-residency.md) (its
  "Departures" section),
  [An SM-driven copy matches the copy engine](2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md)
- Superseded by: none

## Question

Spec flash-next/03 sizes eight K-class slot pools "proportional to each
class's measured traffic share", prefetches the next layer's top-16 by the
router lookahead, and admits prefill misses scan-resistantly. Its acceptance
5 holds the hit rates within 3 points of the compression study's simulation
(`PLACEMENT.md`: 93-96% per domain) and the residency cost within 1.3x of
3.8 ms (one lane) and 9.1 ms (three lanes) per round; acceptance 6 holds the
post-prefill decode hit rate within 2 points. Does the policy as specified
meet that, before any GPU code exists?

## Evidence

`crates/core/tests/expert_residency_study_replay.rs` replays the CPU policy
model (`crates/core/src/residency/policy.rs`) over the study's run-3 routing
(BF16 path, `real/e2e3/routing.pt`), which
`crates/core/tests/expert_residency_study_proxy.py` rewrites in the
artifact's trace layout (layout.md §9, §12) as a machine-local proxy: 58 test
chunks of 2,048 tokens over nine domains, 48 layers, the lookahead router's
top-20. The proxy's K map is a stand-in for the converter's: a Lagrangian over
K in {2, 2.5, 3, 4} at 2.5 bits per weight (scales included) on the run-3
distortion curves, giving 37.79 GB of 4 KiB-aligned records. The cache is the
simulation's 21.5 GB, warm-started from calibration traffic; decode is 256
tokens per chunk; the cost per round is the simulation's model: demand bytes
at 12 GB/s plus 20 us per expert with a missed projection.

What the simulation was (the research session, 2026-10-05): **one** global
byte-LRU over whole experts (gate/up and down together), variable sizes from
the entropy-coded rates, seeded with the hottest experts by routing mass per
byte, no prefetch. Earlier simulations capped a prefetch at one layer's
window, about 1.5 MB; the 62/77/81% at W = 10/16/20 is prediction recall,
never "copy all W".

**Pool split.** Decode without prefetch, as the simulation ran:

| pools split by | 1 lane, ms/round | 3 lanes, ms/round | 3-lane hit |
|---|---:|---:|---:|
| raw byte traffic (selections × slot bytes) | 3.82 | 13.50 (1.48x) | 92.6% |
| expected occupancy under one LRU (Che), per-class pools | 3.77 | 11.15 (1.23x) | 94.1% |
| measured three-lane occupancy of a global LRU (oracle split; Python, no pinning) | 3.90 | 10.69 | 95.0% |
| no split: one byte-LRU over projections (Python, same data) | 3.77 | 9.07 | 95.6% |
| no split: one byte-LRU over whole experts | 3.54 | 8.85 | 95.4% |
| simulation (`PLACEMENT.md`) | 3.8 | 9.1 | — |

Where a global LRU puts the cache, against the two static splits (share of
the 21.5 GB, gate/up then down, K = 2 / 2.5 / 3 / 4):

| | gate/up | down |
|---|---|---|
| global LRU, 1 lane (measured) | 14.5 / 10.8 / 21.6 / 12.6 | 10.4 / 8.3 / 14.3 / 7.5 |
| global LRU, 3 lanes (measured) | 20.0 / 13.5 / 22.3 / 9.9 | 10.9 / 6.8 / 11.4 / 5.3 |
| Che occupancy from calibration | 14.5 / 12.9 / 24.8 / 14.4 | 7.9 / 6.3 / 11.9 / 7.2 |
| raw byte traffic | 8.9 / 9.8 / 24.1 / 24.1 | 5.0 / 4.9 / 11.7 / 11.7 |

With the Che split, the one-lane hit rate by domain against the simulation:
code 96.3 (96), prose 95.6 (95), chat 95.9 (95), en 94.7 (94), it 94.4
(94), zh 94.8 (95), math 92.7 (93), py 95.5 (95), mmlu 94.1 (94); all
94.5%, 38.5 MB of demand per token.

**The lookahead.** W = 16 experts (both projections each), rank order:

| prefetch | 1 lane: hit, demand + prefetch MB/token, link ms, demand cost ms | 3 lanes: same, per round |
|---|---|---|
| none | 94.5%, 38.5 + 0, 3.21, 3.77 | 94.1%, 113.2 + 0, 9.44, 11.15 |
| unbudgeted | 97.9%, 14.0 + 72.5, **7.22**, 1.38 | 97.2%, 53.0 + 295.1, **29.01**, 5.26 |
| budget, stop at the first that does not fit | 96.3%, 25.1 + 36.1, 5.10, 2.51 | 95.2%, 92.0 + 60.6, 12.71, 9.12 |
| budget, skip what does not fit | 96.4%, 24.5 + 38.1, 5.22, 2.46 | 95.3%, 89.7 + 73.2, 13.58, 9.00 |
| budget, skip, floored at the largest projection (default) | 96.5%, 23.8 + 39.2, 5.25, 2.37 | 95.3%, 89.8 + 73.2, 13.58, 9.01 |
| budget, the last one may overshoot | 97.0%, 20.4 + 49.5, 5.83, 2.04 | 95.6%, 82.6 + 97.1, 14.97, 8.31 |

The budget is one layer's share of the round at 12 GB/s: 1.5 MB at one lane
(6 ms), 1.75 MB at three (7 ms). Sparing the lookahead's resident
candidates from the step's own misses changed nothing measurable (2.46
against 2.46 ms, 24.5 against 24.6 MB/token) and was dropped.

**Scan resistance.** Per chunk: decode 256 tokens to settle, 256 measured,
a 4,096-token prefill of the next two chunks, 256 measured:

| | pre | post | drop | a warm 4K prefill moved |
|---|---:|---:|---:|---:|
| W = 16 at the default budget, scan-resistant (58 runs) | 97.3% | 97.0% | 0.3 points | 15.1 GB |
| no prefetch, scan-resistant (29 runs) | 96.2% (24.7 MB/token) | 96.2% (24.5) | 0.0 | 13.7 GB |
| no prefetch, plain LRU admission (29 runs) | 96.2% (24.7) | 94.1% (39.5) | 2.1 | 22.5 GB |

**The plan's expectation.** The per-class LRU's hit rate from calibration
rates alone (Che per class, no locality) is 73.8% at 21.5 GB, where the
replay measures 94.5%.

**The converter's traces (2026-10-06).** The same replay on the converted
artifact's own traces (`IGNIS_FLASH_NEXT_TRACES` at the artifact directory):
routing of the quantized stream, the converter's K map, class record sizes and
calibration traffic (`work/converter.json`), 66 test chunks over eleven
domains (de and ja beyond the study's nine). The experts total 37.80 GB, so
the 21.5 GB cache holds 57% of them; the Che split gives slots gate/up
3,494 / 3,505 / 4,786 / 1,155 and down 3,919 / 3,321 / 4,458 / 1,243 (K = 2 /
2.5 / 3 / 4).

| | 1 lane: hit, demand + prefetch MB/token, link ms, demand cost ms | 3 lanes: same, per round |
|---|---|---|
| no prefetch | 94.7%, 38.1 + 0, 3.18, **3.74** (sim 3.8) | 94.8%, 102.7 + 0, 8.56, **10.07** (sim 9.1, 1.11x) |
| W = 16, unbudgeted | 98.1%, 13.5 + 66.3, 6.65, 1.33 | 97.5%, 48.1 + 253.3, 25.12, 4.76 |
| W = 16, default budget | 96.7%, 23.0 + 37.7, 5.06, 2.29 | 96.0%, 80.0 + 71.3, 12.61, 8.00 |

One-lane hit rate by domain without prefetch, against the simulation: code
95.5 (96), prose 94.7 (95), chat 95.3 (95), en 95.2 (94), it 94.9 (94), zh
95.5 (95), math 93.8 (93), py 95.1 (95), mmlu 94.3 (94); de 94.5 and ja 95.1
(not simulated). At the default budget: 95.8 (math) to 97.4 (code, zh).

Scan resistance: at W = 16 and the default budget, 97.8% before a 4K prefill
and 97.5% after (0.2 points, 66 runs). Without prefetch, scan-resistant 96.8%
then 96.8% (21.7 then 21.6 MB/token); plain LRU admission 96.8% then 94.3%
(39.3 MB/token), and the warm prefill moves 23.1 GB against 13.4 GB.

The calibration-rate expectation is 68.8%, against 94.7% replayed.

## Finding

**Observed.** Pools proportional to traffic starve the K = 2 classes: their
many cold experts are a small share of the traffic but a large share of what
an LRU keeps (gate/up K2: 8.9% of byte traffic, 14.5-20% of a global LRU).
Splitting by the occupancy a single LRU of the cache's size is expected to
give each class, computed from the same calibration traffic, matches the
simulation at one lane (3.77 against 3.8 ms, every domain within 0.7
points) and stays inside acceptance 5 at three (1.23x).

**Observed.** What remains at three lanes is the partition itself: even the
measured oracle split costs 10.69 ms, against 9.07 for one byte-LRU on the
same data. Per-class pools give up some flexibility the simulation had.

**Observed.** An unbudgeted W = 16 lookahead moves more bytes than the link
carries in a round: 7.2 ms of transfers per 6 ms round at one lane, 29 per
7 ms at three. Held to one layer's window it fits (5.25 and 13.58 ms) and
still cuts the demand stall from 3.77 to 2.37 ms at one lane.

**Observed.** Scan-resistant admission keeps the decode set through a 4K
prefill (0.0-0.3 points), where plain LRU loses 2.1 points and moves 1.6x
the bytes; decode demand after the prefill is 24.5 against 39.5 MB/token,
the study's 21 against 40.

**Observed (2026-10-06).** On the converter's own traces the policy holds
and the three-lane gap narrows: every domain within 1.2 points of the
simulation, 3.74 ms at one lane and 10.07 ms (1.11x) at three without
prefetch; the default budget cuts the one-lane stall to 2.29 ms at 5.06 ms of
link; a 4K prefill costs decode 0.0-0.2 points scan-resistant, 2.5 under plain
LRU. The quantized stream's routing keeps the BF16 path's locality.

**Inference.** The calibration-rate expectation is a floor, not a forecast:
real routing's locality adds about 20 points. It compares two plans; it does
not predict throughput.

## Implications

- Spec flash-next/03 records these as its departures (pool split, prefetch
  budget, W in experts, ring sizing, units).
- The GPU residency must reproduce the policy exactly: per-class LRU by
  `(stamp, key)`, rank-ordered lookahead with the skip rule, scan-resistant
  prefill, the budget as a load option.
- If three lanes must reach the simulation's 9.1 ms, the lever is a byte-LRU
  across classes (a larger slot hosting a smaller projection, or periodic
  rebalancing toward the highest marginal miss cost per byte), not a better
  static split.
- The "overshoot" budget variant (the last prefetch may pass the window)
  cut one lane's stall to 2.04 ms at 5.83 ms of link; whether the link
  really has that slack is a timing question for the GPU replay, which
  should tune the budget.

## Limits and unknowns

- The 2026-10-05 tables use BF16-path routing (run 3) and a stand-in K map;
  the converter's traces and K map (2026-10-06) moved the per-domain hit
  rates by at most 1.1 points and brought the three-lane cost from 1.23x to
  1.11x.
- Teacher-forced routing: decode is replayed over the traced text's tokens,
  not over the model's own samples.
- No timing: costs use the simulation's model, and prefetches are taken as
  hidden behind compute up to the budget. The GPU replay measures stall.
- The round times (6 / 7 ms) behind the budget are the study's 27B-anchored
  estimates.

## Follow-ups

- The GPU trace replay of spec 03 acceptance 4-6 on these traces, through
  the expert op, with measured stall; re-tune the budget's round times on it.
