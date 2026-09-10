# Indicative G3 measurement of #111 (not a gate verdict)

An A/B of the batch-wide decode round (#111, ADR 0020) against its own
parent commit, taken on 2026-09-10 on a free RTX 5090.

**This decides nothing.** ADR 0015 requires the gate to be measured
live/live against the reference in one session, and no reference leg was
run here. What this answers is narrower and was blocking the ordering: did
#111 move the ITL tail, and did it cost C=1 anything.

## Why it is shaped this way

The first read compared this tree's ignis leg against `ignis-final.json`
from #110, recorded the same day. That showed C=1 falling from 68.3 to 63.9
tok/s, which read as a possible regression — the exact risk #111's remaining
acceptance criterion names. Two things made that comparison untrustworthy:
it crosses sessions, which ADR 0015 forbids for a reason, and nine commits
separate the two trees, two of which touch the server and the measurement
instrument.

So the comparison here holds everything constant except the leaf: same
machine, same session, same server and bench binaries, same fixture and
corpus. Only `kernel/` moves, between `0e1a275` (the parent) and `01f96fe`.
Each leg was sampled repeatedly rather than once, because nobody had
measured this instrument's own repeatability before.

## Raw samples

Server: `ignis-server` release, defaults (BF16 KV, 1024 prefill chunk, CUDA
graphs all eight widths captured, K=1, 65,536-token KV pool, max-context
40,960). Fixture: `ignis-bench g3` defaults — C=1 and C=4 at 8,192-token
prompts and 256 tokens, ITL at ten sequential 32,768-token prefillers over
four 4,096-token decode lanes. Prompts cut from
`F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids`.

| run | kernel | C=1 tok/s | C=4 tok/s | ITL p50 ms | ITL p95 ms |
|---|---|---:|---:|---:|---:|
| `ignis.json` | `01f96fe` | 63.9 | 43.1 | 149.62 | 191.73 |
| `ignis-r2.json` | `01f96fe` | 64.5 | 43.2 | 150.71 | 194.35 |
| `ignis-r3.json` | `01f96fe` | 64.2 | 43.1 | 150.09 | 194.40 |
| `baseline-r1.json` | `0e1a275` | 73.9 | 20.9 | 198.55 | 247.86 |
| `baseline-r2.json` | `0e1a275` | 63.4 | 18.5 | 196.80 | 234.59 |
| `baseline-r3.json` | `0e1a275` | 63.6 | 18.4 | 195.92 | 237.91 |
| `baseline-r4.json` | `0e1a275` | 63.8 | 18.4 | 194.78 | 237.90 |
| `baseline-r5.json` | `0e1a275` | 63.5 | 17.9 | 195.91 | 236.75 |

`baseline-r1` is an outlier on C=1 and C=4 and is kept in the record rather
than dropped; the medians below are medians over every sample, so it does
not carry the comparison either way. It was the first run against a
freshly started server. The post-#111 leg's own first run shows no such
spike, so what produced it is unexplained. #110's recorded ignis leg (68.3
tok/s C=1) sits between that outlier and the stable cluster, which is the
likeliest reason the cross-session read looked like a regression.

## Medians

| cell | pre-#111 (n=5) | post-#111 (n=3) | change |
|---|---:|---:|---:|
| C=1 | 63.6 tok/s | 64.2 tok/s | +1% |
| C=4 aggregate | 18.4 tok/s | 43.1 tok/s | +134% |
| ITL p50 | 195.92 ms | 150.09 ms | -23% |
| ITL p95 | 237.90 ms | 194.35 ms | **-18%** |

Within-session repeatability, post-#111: C=1 spread 0.9%, C=4 0.2%, ITL p95
1.4% across three runs. The instrument is tight enough that a 6% shift is a
signal, not noise — which is why the cross-session read had to be chased
rather than waved off.

## What it says

- **No C=1 regression.** The cell is marginally faster, and well inside the
  spread. The remaining acceptance criterion on #111 is about this, and the
  answer here is not the gate's answer, but it is not a red flag either.
- **C=4 aggregate more than doubles.** 18.4 to 43.1 tok/s.
- **The ITL tail moves by the predicted amount.** #110 decomposed a blocked
  inter-token interval as a prefill chunk plus a decode round and predicted
  that fixing the round would bring the interval from 186.5 ms to near
  136 ms. The measured p50 moved 195.9 to 150.1 ms.
- For scale only, and across sessions, so not a verdict: the reference leg
  recorded in `.scratch/g3-gate-110/reference-final.json` had ITL p95
  201.11 ms and C=1 62.0 tok/s. This tree's 194.35 ms would put the p95
  ratio near 0.97, inside the envelope that #110's 1.106 failed.

## What still has to happen

The closing run is live/live in one session against `ninfer-serve` in the
owner's production profile, and it should come after #114 (finish reason per
lane, and a cap that is spare rather than load-bearing). On the reference
leg of #110's run the 3,072-token cap was already load-bearing; this tree's
lanes generate more tokens in the same window, so the cap binds harder, and
without a recorded finish reason a reader cannot tell a valid run from a
truncated one.
