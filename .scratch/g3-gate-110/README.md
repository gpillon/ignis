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

## How each lane actually ended

The two legs were **not** terminated by the same mechanism, which is worth
knowing before trusting a re-run:

| leg | lane | tokens | ended by |
|---|---|---:|---|
| ignis | all four | 897-909 | cancellation, all four at 102,325 ms |
| reference | itl-decode-0 | 2,856 | its own EOS |
| reference | itl-decode-1..3 | 3,072 | the safety cap |

Only ignis honors `ignore_eos`; the reference was sent every artifact EOS id
as a `logit_bias: -100` exclusion, which `reference-2.json` already recorded
as insufficient (`finish=stop_token` persisted). So on the reference leg the
3,072-token cap is load-bearing rather than spare: the lanes outlived the
final window by 3.4 to 7.0 s, and a reference roughly 7% faster would have
exhausted the cap before the window closed and had its lane refused.

This does not invalidate these records. Every lane on both legs outlived the
final prefill window, which is the property the pooling guard checks and the
only one the metric depends on. It does mean a re-run should raise the cap
and record each lane's finish reason, which is filed as #114.

## What the fixture now guarantees

Both legs record a shared monotonic timeline, wait for a first token on
every decode lane before the first prefiller starts, pool only the
intervals intersecting a prefiller's request-to-first-token window, and
refuse a lane that ends before the last window. Records without the
timeline are refused as incomparable, so #104's records cannot be compared
against these.

## What the numbers say

A decode lane's interval falls in one of two states: blocked behind a prefill
chunk, or a decode round on its own between windows. The second kind is what
separates the two engines, and it is measurable straight off these records by
taking the intervals that intersect *no* prefiller window.

| | ignis | reference | ratio |
|---|---:|---:|---|
| decode round, 4 lanes (median, outside every window) | 70.2 ms | 17.3 ms | 4.06 |
| decode round, 1 lane (from the C=1 cell) | 14.6 ms | 16.1 ms | 0.91 |

**At one lane ignis is ahead. At four it costs 4.06x the reference, and 4.8x
its own single-lane round.** That is GitHub #111: the width-W decode graph
replays W sequential per-lane model traversals instead of one B-wide one, and
`kernel/src/decode_graph.cu:105` and `kernel/src/step.cu:717` both say so in
their own comments. Only the sampling is batched.

With the decode round known, the prefill chunk follows from the window
arithmetic (`32 chunks + rounds x round_cost = window`):

| term | ignis | reference | ratio |
|---|---:|---:|---|
| prefill chunk, 1,024 tokens at 32K context | 116.3 ms | 141.2 ms | 0.82 |
| decode round, B=4 | 70.2 ms | 17.3 ms | 4.06 |
| blocked ITL interval, the sum | 186.5 ms | 158.5 ms | 1.18 |

The reconstructed sum matches the measured blocked-interval means of 180.8 ms
and 155.4 ms, so the split is sound.

**ignis's prefill is 21% faster, not 15% slower, and the whole ITL p95 gap is
the decode round.** An earlier reading of these same records attributed the
blocked interval to prefill throughput; that assumed the decode round was
small on both sides, which is true of the reference and false of ignis.

The KV precision difference the profiles record (`BF16 KV` against
`hq-e8-2b KV`) would act on prefill, and ignis already wins prefill while
carrying it. So these records do **not** support KV precision as the
explanation for the p95 gap. Bringing the decode round to the reference's
order would put the blocked interval near 136 ms against 158 ms, a ratio of
about 0.86.

The lockstep of `ConcreteScheduler::advance` (#113) is real but secondary:
ignis runs 1.056 decode rounds per prefill chunk against the reference's
1.44. Note that ignis's "no interval under 40 ms" is *not* evidence of it —
that is just a 70 ms decode round never fitting under the threshold.

## How to reproduce the comparison

Each live record used `ignis-bench g3` with `--session g3-110-window-cancel-v3`
against its own endpoint: `:8000` for ignis, `:8080` for the reference. Max
context 40,960; the reference's KV capacity resolved to 327,680 tokens at max
concurrency 8; greedy, thinking disabled on both. Never compare records
carrying different session ids — the gate check refuses to.

```powershell
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3-gate `
  --ours .scratch\g3-gate-110\ignis-final.json `
  --ref .scratch\g3-gate-110\reference-final.json `
  --out .scratch\g3-gate-110\verdict-final.json
```

## Profile caveat

The first reference restart failed during CUDA Graph preparation
(114,163,712 bytes consumed against a 100,663,296-byte allowance). An
identical retry started successfully, reporting `graphs=0.00 MiB/96.00
MiB` at startup. Coldness and completeness are unaffected, but the caveat
should be carried if the narrow 1.106 miss is ever reproduced.
