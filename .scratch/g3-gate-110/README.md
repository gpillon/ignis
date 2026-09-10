# G3 ITL re-measurement records (GitHub #110)

The records behind #110's second G3 run, after the ITL fixture's own two
measurement defects were fixed. `.scratch/g3-logs/` holds #104's original
run, whose 1.130 ITL p95 ratio these records supersede.

Everything here was produced by `ignis-bench g3` against
`F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer`, prompts cut from
`F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids`, on a free RTX 5090, one
session (`g3-110-window-cancel-v3`), reference and ignis measured back to
back with no other process touching the GPU between legs (ADR 0015).

## The verdict

| file | what it is |
|---|---|
| `reference-final.json` | The reference leg: `ninfer-serve` in the owner's production profile (hq-e8-2b KV, 1024 chunk, CUDA graphs, prefix reuse). C=1 62.0 tok/s, C=4 aggregate 13.0 tok/s, ITL p50/p95 133.99/201.11 ms over 1,845 intervals, ten cold prefillers, four complete decode lanes. |
| `ignis-final.json` | The ignis leg (BF16 KV, 1024 chunk, CUDA graphs, K=1). C=1 68.3 tok/s, C=4 aggregate 19.7 tok/s, ITL p50/p95 184.32/222.46 ms over 1,352 intervals, same fixture shape. |
| `verdict-final.json` | `ignis-bench g3-gate` over the two above: C=1 ratio 1.101 PASS, C=4 ratio 1.514 PASS, ITL p95 ratio 1.106 **FAIL** (tolerance is <= 1.1). |
| `reference.json` | First 3,072-cap attempt, one EOS id suppressed. No ITL verdict: the reference lanes still stopped early. |
| `reference-2.json` | All artifact EOS ids sent as `logit_bias`. Still no ITL verdict — `finish=stop_token` persisted, which is what motivated ending the lanes by cancellation rather than by exhausting a cap. |

## What the fixture now guarantees

Both legs record a shared monotonic timeline, wait for a first token on
every decode lane before the first prefiller starts, pool only the
intervals intersecting a prefiller's request-to-first-token window, and
refuse a lane that ends before the last window. Records without the
timeline are refused as incomparable, so #104's records cannot be compared
against these.

## What the numbers say

Splitting the pooled intervals into those blocked behind a prefill chunk
(>= 40 ms) and those that are a decode round alone (< 40 ms):

| | ignis | reference |
|---|---:|---:|
| blocked intervals, mean | 180.8 ms | 155.4 ms |
| intervals under 40 ms | 0 of 1,352 | 530 of 1,845 |
| decode rounds per lane per window | 33.80 | 46.12 |
| prefill throughput per prefiller | 5,375 tok/s | 6,164 tok/s |
| time per 1,024-token chunk | 190.5 ms | 166.1 ms |

During a prefill window the ITL floor is one chunk plus one decode round,
so p95 is effectively a second measurement of prefill throughput. The
blocked-interval ratio is 1.163; restricting the reference to its blocked
intervals alone moves its p95 only from 201.1 ms to 204.0 ms, so its cheap
decode rounds are not what wins it the p95.

The two engines are not running the same KV precision — the profiles say so
(`BF16 KV` against `hq-e8-2b KV`). ADR 0015 chose that deliberately and
records the difference next to the verdict rather than correcting for it;
the v1 design doc schedules hq-e8-2b for phase 4. At 32,768 tokens every
chunk's attention rereads the whole prior KV, so ignis moves about twice
the bytes on a bandwidth-bound operation.

The p50 gap is separate and is ours: `ConcreteScheduler::advance` runs
exactly one prefill chunk and then exactly one batched decode round, so a
lane can never emit two tokens between chunk boundaries.

## Profile caveat

The first reference restart failed during CUDA Graph preparation
(114,163,712 bytes consumed against a 100,663,296-byte allowance). An
identical retry started successfully, reporting `graphs=0.00 MiB/96.00
MiB` at startup. Coldness and completeness are unaffected, but the caveat
should be carried if the narrow 1.106 miss is ever reproduced.
