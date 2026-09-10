# G3 re-measurement after #111 (GitHub #110, #113)

ADR 0021 launch-pooled live/live re-run to check whether #111 (ADR 0020,
B-wide decode traversal, merged 2026-09-10) closed #110's ITL p95 gap and
changed the picture for #113.

Everything here was produced by `ignis-bench g3` against
`F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer`, prompts cut from
`F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids`, on a free RTX 5090, one
session (`g3-110-113-20260910`). Per ADR 0021, each engine was launched
**twice** (process stopped and restarted between launches), and the verdict
is read from the pooled behavior across launches, not from one arbitrarily
chosen pair.

## Profile change from prior runs

The reference's `--kv-capacity` was set **explicitly to 65,536** tokens this
session, matching ignis's fixed 65,536-token pool (`ignis_runtime::kv_pool_tokens_for`
at the default 40,960-token `--max-context`). Prior runs (`.scratch/g3-gate-110/`)
left it on `auto`, which resolved to 327,680 tokens (`max-context x
max-concurrency`, 8x more headroom than ignis ever gets). Server log
confirms: `KV capacity explicit resolved=65536 tokens pages=1024/5120`,
same page count ignis reports. Everything else matches the owner's
production profile: hq-e8-2b KV, 1,024-token prefill chunk, CUDA graphs,
prefix reuse — none of `--no-cuda-graph` / `--no-prefix-reuse` passed.

```
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe `
  F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --model-id qwen3.8-27b-nvfp4full-v2 --host 127.0.0.1 --port 8080 `
  --kv-dtype hq-e8-2b --prefill-chunk 1024 --max-context 40960 --max-concurrency 8 --kv-capacity 65536
```

ignis ran at every flag's default: `ignis-server.exe --artifact <artifact>
--model qwen3.8-27b-nvfp4full-v2 --bind 127.0.0.1:8000 --prefill-chunk 1024
--max-context 40960` (BF16 KV, pool 65,536 tokens, K=1..8 decode graph
captured at load).

## Records

| file | engine | launch | C=1 | C=4 aggregate | ITL p50/p95/p99/max (ms) | intervals |
|---|---|---:|---:|---:|---|---:|
| `reference-1.json` | reference | 1 | 63.0 tok/s | 12.1 tok/s | 130.02 / 193.69 / 210.56 / 227.74 | 1,880 |
| `reference-2.json` | reference | 2 | 64.9 tok/s | 10.0 tok/s | 140.99 / 212.91 / 235.61 / 261.49 | 1,860 |
| `ignis-1.json` | ignis | 1 | 65.3 tok/s | 43.7 tok/s | 151.17 / 194.40 / 200.04 / 208.16 | 1,471 |
| `ignis-2.json` | ignis | 2 | 61.6 tok/s | 43.0 tok/s | 153.10 / 196.05 / 202.44 / 211.21 | 1,461 |

**The ITL cell's cold/complete guard rejected the reference twice per launch
before landing a valid record** (2 of 4 decode lanes' worth of retries per
launch): a lane hitting its own EOS (or the 4,032-token safety cap) before
the final prefill window closed, exactly the known-but-not-guaranteed
termination path spec 03 documents. This is launch-to-launch/run-to-run
non-determinism in which lane stops early — not the same content twice — so
it is scheduling jitter among the four concurrent decode lanes and
prefillers, not a corpus or fixture defect. **ignis's ITL cell was cold and
complete on both launches, first attempt, no retries.** Discarded attempts
were not kept (only the final valid record per launch is on disk); each
retry used the same `--session` and produced a differently-shaped rejection
(different lane, different token count), which is itself evidence for
jitter rather than a deterministic corpus issue.

## Launch-to-launch spread (ADR 0021)

| cell | ignis L1 | ignis L2 | spread | reference L1 | reference L2 | spread |
|---|---:|---:|---:|---:|---:|---:|
| C=1 | 65.3 | 61.6 | 5.9% | 63.0 | 64.9 | 2.9% |
| C=4 | 43.7 | 43.0 | 1.6% | 12.1 | 10.0 | **18.8%** |
| ITL p95 | 194.40 | 196.05 | 0.8% | 193.69 | 212.91 | 9.4% |

ignis is tight across launches on every cell. The reference is not — its C=4
spread (18.8%) is the same order of magnitude as #116's original 17%
finding, on this exact tree, tonight. This is exactly the failure mode ADR
0021 exists to catch: a verdict read off a single reference launch would
have been one of two visibly different numbers.

## The four pooled gate verdicts

`ignis-bench g3-gate` run over every launch pairing (`verdict-ref{1,2}-ignis{1,2}.json`):

| pairing | C=1 ratio | C=4 ratio | ITL p95 ratio | verdict |
|---|---:|---:|---:|---|
| ref L1 x ignis L1 | 1.037 PASS | 3.614 PASS | 1.004 PASS | **PASS** |
| ref L1 x ignis L2 | 0.978 FAIL | 3.561 PASS | 1.012 PASS | FAIL (C=1 only) |
| ref L2 x ignis L1 | 1.007 PASS | 4.385 PASS | 0.913 PASS | **PASS** |
| ref L2 x ignis L2 | 0.950 FAIL | 4.321 PASS | 0.921 PASS | FAIL (C=1 only) |

Mean-of-launches ratio per cell (ignis mean / reference mean):

| cell | ignis mean | reference mean | ratio |
|---|---:|---:|---:|
| C=1 | 63.45 tok/s | 63.95 tok/s | **0.992** |
| C=4 | 43.35 tok/s | 11.05 tok/s | **3.924** |
| ITL p95 | 195.23 ms | 203.30 ms | **0.960** |

**ITL p95 clears the gate in all four pairings, most with margin** (0.913 to
1.012, ceiling 1.10). This is the cell #110 tracks.

**C=1 splits exactly on which reference launch it's paired against**, both
misses landing at 0.95-0.98 — a hair under the 0.99 floor, not a collapse.
The pooled mean (0.992) sits right at the floor. Given the reference's own
launch-to-launch band is visibly wider than this margin on this same
session (18.8% on C=4), this reads as launch noise dominating a
near-parity result, not a regression — but it is reported as measured, not
rounded up. C=1 is not #110's or #113's cell and this run does not
propose closing anything on it; it is flagged here for whoever looks at C=1
next.

## Why ITL p95 moved: the decode round, isolated

Splitting every pooled interval by whether a prefill chunk was in flight
(the same method `.scratch/g3-gate-110/README.md` used), taking the
intervals that intersect **no** prefiller window — a decode round with
nothing else going on:

| | ignis L1 | ignis L2 | reference L1 | reference L2 |
|---|---:|---:|---:|---:|
| decode round B=4, median | 18.56 ms | 18.62 ms | 16.68 ms | 18.16 ms |
| decode round B=4, mean | 19.83 ms | 19.91 ms | 17.78 ms | 18.80 ms |

Pooled: ignis's median decode round is **18.59 ms** against the reference's
**17.42 ms** — ratio **1.067**. Before #111, this exact measurement (#110's
last comment, `.scratch/g3-gate-110/README.md`) was **70.2 ms against 17.3
ms — ratio 4.06**. `kernel/src/decode_graph.cu` and `kernel/src/step.cu`'s
B-wide traversal (ADR 0020) took the one term that was 4x the reference and
put it within 7% of it. That 7% is well inside the launch-to-launch noise
this session independently measured (up to 18.8% on the reference's own
C=4), so it is not obviously a further gap to chase.

## #110 — recommend closing

ITL p95 passes at every pooled launch pairing (0.913-1.012, ceiling 1.10),
and the mechanism #110's last comment named — the decode round costing 4.06x
the reference — is now measured at 1.067x, consistent with #111 landing.
Requirement 17 (spec `03-serving-loop.md`) is met at this phase.

## #113 — recommend closing as not needed for now

C=4 aggregate: ignis 43.35 tok/s pooled mean against the reference's 11.05 —
**ignis is 3.9x the reference**, not the 99%-of-reference bar the issue's
own closure criterion asked for. The reasoning in the issue's last comment
("adding more rounds per tick would cost ignis more than it gains, because a
round costs 70ms against the reference's 17ms... better asked once a round
is cheap") — the round is now cheap (18.6 ms against 17.4 ms, measured
above). The premise that motivated deferring the multi-round-per-tick
question no longer holds, but neither does the original justification for
pursuing it: `ConcreteScheduler::advance`'s one-round-per-chunk policy is no
longer leaving throughput on the table at this shape. Reopen if a future
gate shape (more lanes, a narrower chunk) shows a live gap C=4 doesn't
already cover.

## Reproduction

```powershell
# reference, either launch
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3 `
  --endpoint http://127.0.0.1:8080 --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --label reference --profile "hq-e8-2b KV, 1024 chunk, graphs, kv-capacity 65536 explicit (comparable to ignis)" `
  --session g3-110-113-20260910 --corpus F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids `
  --out .scratch\g3-gate-110-113-rerun\reference-N.json

# ignis, either launch
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3 `
  --endpoint http://127.0.0.1:8000 --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --label ignis --profile "BF16 KV, 1024 chunk, graphs, post-#111 B-wide decode" `
  --session g3-110-113-20260910 --corpus F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids `
  --out .scratch\g3-gate-110-113-rerun\ignis-N.json

# any pairing
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3-gate `
  --ours .scratch\g3-gate-110-113-rerun\ignis-1.json `
  --ref .scratch\g3-gate-110-113-rerun\reference-1.json `
  --out .scratch\g3-gate-110-113-rerun\verdict-ref1-ignis1.json

# blocked/free interval decomposition (this README's table)
python .scratch\g3-gate-110-113-rerun\decompose.py
```

Never compare these records against `.scratch/g3-gate-110/`'s: the
reference's KV capacity differs (65,536 explicit here vs 327,680 auto
there), so the two sessions measure different reference configurations even
though both are "the production profile."
