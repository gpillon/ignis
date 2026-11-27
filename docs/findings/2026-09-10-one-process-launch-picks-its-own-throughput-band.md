# One process launch picks its own throughput band: two launches of one build read up to 17% apart, and repetition inside a launch cannot cancel it

- Kind: experiment
- Status: current
- Observed: 2026-09-10
- Last verified: 2026-09-10
- Scope: bench / `ignis-bench g3` against `ignis-server`, 27B decode on the RTX 5090
- Topic: harness-and-build
- Related: [GitHub #116](https://github.com/gpillon/ignis/issues/116),
  [ADR 0082](../adr/0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md),
  [hq vs BF16 live/live](2026-09-13-hq-vs-bf16-live-live.md),
  [decode round host idle](2026-09-18-decode-round-host-idle.md)
- Superseded by: none
- Summary: On the RTX 5090, `ignis-bench g3` runs against one `ignis-server` launch agree within 1.5%, and two launches of the same build sit up to 17% apart on C=1 and ITL p95; the GPU's busy clock is the same in both.

Written on 2026-11-27 from GitHub #116 and its closing comment, when the ADR
that held this measurement left the tree (it is `0021-live-live-launch-pooling.md`
in `docs/adr/REMOVED.md`). Nothing was measured again. The records were in the
main checkout's `.scratch/g3-indicative-111/`.

## Question

`ignis-bench g3` gave C=1 and ITL numbers that differed by far more than its
own samples' spread from one run to the next on the same tree. Is that the
code, the run order, the card, or the server process?

## Evidence

**The first report** (#116): three `g3` runs against one `ignis-server`
process agree to better than 1%, and three runs against another process, on
the same tree and the same fixture, agree with each other as well and sit 17%
away. Every cell reported all its samples cold, so no run was served from a
warm prefix.

| server process | tree | C=1 tok/s | ITL p95 ms |
|---|---|---|---|
| A | `01f96fe` | 63.9, 64.5, 64.2 | 191.7, 194.4, 194.4 |
| C | `01f96fe` + #114's cap | 74.6, 74.7, 74.6 | 167.6, 167.8, 167.2 |
| B | `0e1a275` | 73.9, then 63.4, 63.6, 63.8, 63.5 | 247.9, then 234.6-237.9 |

Both trees produced both bands. `nvidia-smi` showed no throttle flag, and the
card idled at 40 C between runs.

**The controlled follow-up**, on a free RTX 5090: launch A sampled four times
over 21 minutes, a 3-minute idle gap included, and launch B sampled three
times, started the moment A was stopped.

| launch | sample | C=1 tok/s | ITL p95 ms |
|---|---|---:|---:|
| A | r1 | 72.9 | 184.08 |
| A | r2 | 68.7 | 175.35 |
| A | r3 | 72.6 | 178.10 |
| A | r4 (after 200 s idle) | 73.4 | 182.73 |
| B | r1 | 68.8 | 180.72 |
| B | r2 | 69.5 | 178.58 |
| B | r3 | 69.8 | 178.67 |

Over the `nvidia-smi` samples with `utilization.gpu` above 30% (compute, not
the idle between requests) the two launches drove the SM clock to 2,823 and
2,824 MHz on average and to a 2,910 MHz peak both, and
`clocks_event_reasons.active` stayed `0x0` in both.

Within-launch spread is not one number: launch B's three samples span 1.4%
and launch A's four, on the same tree, 6.4%.

## Finding

Observed:

- The unit that picks a throughput band is one process launch. Runs against
  one launch agree within about 1.5% in the first report, and within 1.4% (B)
  and 6.4% (A, one low outlier) in the follow-up; two launches of the same
  binary, tree and fixture, moments apart on the same card, sit 5% (the
  follow-up) to 17% (the first report) apart on C=1 and on ITL p95.
- It is not the code: the trees before and after #111 each produced a fast
  and a slow launch.
- It is not slow drift over a launch's life: A holds its band across 21
  minutes and an idle gap, with one low outlier and no trend.
- It is not heat or clock state left by the process before: B's first sample
  is already in B's band, not in A's.
- It is not the GPU's sustained clock: during equal work it is the same in
  both launches, and no throttle or thermal event was set in either.
- B changed band once, after its first run in the first report; the launches
  of the follow-up did not.

Inferred, not shown: with the GPU equally fast in both bands, what sets a
launch's band is on the host side of each token, kernel-launch dispatch, the
wait in `cudaStreamSynchronize`, the OS's thread scheduling or the async
runtime's wake-up latency. Each is a fraction of a millisecond against a
decode token of 13 to 15 ms at these rates, so a few percent of throughput,
and it has only to hold for one launch's life to give a stable band.

## Implications

- Repeating a sample inside one launch cancels the noise between requests and
  cannot cancel a bias that is constant for the whole launch. A comparison of
  one launch a side compares, some of the time, two launches and not two
  engines or two builds: the 17% is larger than the 1% tolerance the G3 check
  holds on C=1 and C=4 and the 10% it holds on ITL p95.
- So a speed comparison pools at least two independent launches a side, the
  server stopped and started again between them, and reads the statistic over
  all of them, neither the best launch nor one chosen at will. That does not
  depend on finding the cause. It is the practice ADR 0082 keeps for a speed
  claim, and what `ignis-bench g4-gate` and `g5-gate` hold their records to
  (`crates/bench/src/lib.rs`).
- A cell whose spread across launches is larger than the spread inside its
  own launches is worth recording, not averaging away.

## Limits and unknowns

- The mechanism is a hypothesis. On 2026-09-10 the repo had no profiler that
  correlates the host and the device timelines, and no later record traces
  this band with one.
- Measured on the 27B's decode on 2026-09-10, at one lane and on the ITL
  cell: the size of the band on today's engine was not measured here.
  [hq vs BF16 live/live](2026-09-13-hq-vs-bf16-live-live.md) met the band
  again three days later, wider under hq-e8-2b.
- Two launches a side is a floor that catches one unlucky launch; it does not
  bound the band.

## Follow-ups

None open: #116 closed on the pooling rule.
