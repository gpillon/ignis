# GitHub #138 — verification records

The 131,072-token needle cell, after the transport's deadline was made
explicit. Both engines, both gate lengths, no transport-level abort.

## Root cause

`reqwest::blocking::Client::new()` carries an undeclared **30-second total
request timeout** that covers reading the response body
(`reqwest-0.12.28/src/blocking/client.rs:1503`, documented at `:385` as
"Default is 30 seconds"). A streamed measurement request lives entirely
inside that deadline, so the needle cell died mid-prefill while the engine
was still computing normally.

The two gate lengths sit either side of the line. From the failing
session's own reference request log
(`.scratch/g4-logs/ninfer-launch{1,2}-requests.jsonl`):

| prompt | prefill | rate | outcome |
|--------|---------|------|---------|
| 65,536 | 14.46 s | 4,531 tok/s | `stop_token`, total 15.0 s — **passed** |
| 131,072 | never completed | — | `cancelled` at **30.85 s**, `gen=0` — failed |

Both launches cancelled at 30.8 s, which is the deadline and not a
property of either engine.

> Note on the rates quoted in the issue body: `prefill=1146.9tok/s` and
> `1618.9tok/s` come from `event: "throughput"` records, which are
> `computed_prefill / interval_seconds` over intervals that also contain
> decode work — an interval average, not the prefill rate. The same log's
> `request_done` records give the per-request prefill rate: 8,827 tok/s at
> 4,096 tokens, 6,235 tok/s at 32,768, 4,531 tok/s at 65,536.

## ignis

`crates/server/tests/needle_128k_gpu.rs`, run under the preflight-gated GPU
profile (`scripts/gpu-preflight.ps1` + `IGNIS_GPU_PROFILE=1`), 2026-09-13:

```
needle@65536: retrieved=true in 20.4s
needle@131072: retrieved=true in 60.1s
```

An earlier run of the same test the same day read 20.6 s / 63.1 s.

## Reference (ninfer)

`ignis-bench g4` against `ninfer-serve` on the same artifact at
`--max-context 262144`, needle cells only (a one-line trace, so the replay
leg is trivial and the run is just the two cells):

```
needle@65536 RETRIEVED
needle@131072 RETRIEVED
```

Full record: `reference-needle-verification.json` (session
`issue-138-verify`, 2026-09-13T00:22:59Z).

This is a verification run, **not** a gate run: it is one launch per engine
and the two engines were measured minutes apart, so it does not meet spec
04's live/live pooling rule (ADR 0021, two launches per engine in one
session). What it establishes is only what #138 asked for — that the cell
reaches a verdict at all at 131,072 tokens. The real G4 gate run is #128's.
