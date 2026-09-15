# GitHub #90 — Prometheus metrics, real-GPU A/B

The ADR 0017 check that metrics add no inference-path cost: metrics disabled
(`off`), enabled but never scraped (`on`), enabled and scraped every 15 s
(`s15`), plus a scrape every second (`s1`) as a stress diagnostic. Each leg is
one fresh `make start` launch (serving config, only `METRICS` toggled) running
`ignis-bench g3` (C=1 / C=4 throughput, ITL) and `ignis-bench g4` (the G4 load
trace replay, per-class TTFT and delivered tok/s), all legs of a session under
one `--session`. `ab.sh` runs the legs, `verdict.py` reduces them.

**Owner decision, 2026-09-15: accepted as "no evident regression".** The
repeatability proof the ticket describes was not completed — see below — and
the owner closed the measurement there, consistent with benchmarks being
error detectors rather than per-ticket proofs.

## Session 1 — `session1-spec-on/` (`metrics90-20260915T123315Z`)

Serving config with DFlash2 speculation (7 drafts), legs warm, off, on, s15,
off, on, s15, s1 — all completed, 28 requests per leg, 0 of 387 scrapes failed
(s15: 24 and 25, s1: 338), GPU power flat at ~300 W across legs.

`verdict.py` reports FAIL: `g4` delivered tok/s 2-3% lower with metrics on,
TTFT median ratio 1.013-1.016 at s15. **Not attributable to metrics:**

- the order was always off → on → s15, while C=1 fell steadily over the
  session (96.4 → 89.6 tok/s), so drift lands on the enabled legs;
- the unscraped `on` legs lose 2.5% too, though they only add atomics on the
  asynchronous telemetry consumer;
- under speculation tokens arrive in bursts: ITL p50 reads 0.01 ms (no step
  timing) and C=4 swings 38.9-61.9 tok/s between two `off` launches;
- with two launches a side, "every enabled launch worse than every disabled
  one" happens by chance about one time in six.

## Session 2 — `session2-spec-off/` (`metrics90-20260915T132424Z`), interrupted

Speculation off, counterbalanced order (off, on, s15, s15, on, off, off, on,
s15, s1). Stopped at the owner's request after `off-1`, `on-1` and `s15-1`'s
g3 (`s15-1`'s g4 was killed mid-run and has no record).

| leg | C=1 tok/s | C=4 tok/s | ITL p50 / p95 ms | g4 main / sub tok/s | TTFT median main / sub |
|---|---|---|---|---|---|
| off-1 | 54.59 | 26.76 | 181.72 / 246.66 | 31.27 / 27.07 | 3697 / 4899 ms |
| on-1 | 55.43 | 32.42 | 180.39 / 243.80 | 31.57 / 27.73 | 3681 / 4613 ms |
| s15-1 | 53.08 | 33.82 | 184.64 / 249.13 | — | — |

Enabled-unscraped is level with or better than disabled everywhere; s15's C=1
(-2.8%) and ITL (+1-1.6%) sit inside the cell's launch-to-launch noise (#143)
while its C=4 is +26%. One launch a side proves nothing either way.

## Not done

- Three counterbalanced launches per configuration (repeatability), and the
  1-second stress leg without speculation.
- The G4 gate against the reference (not re-run; its `sub` gap was already
  accepted).
- A GPU step-timing source: none exists server-side, and adding one would be
  inference-path work, so g3's ITL stands in for it.
