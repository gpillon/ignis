# The KV-format A/B, live/live (the 2x2 over engine and format)

Runbook: `docs/agents/testing.md`, "The KV-format A/B". This is the second
execution of it, and it differs from the first
([`docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md`](../../../docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md))
in the one way that decides what the numbers may be used for: **every leg ran
inside one session**, alternating on the card, rather than each engine being
measured in a session of its own. ADR 0015 asks for exactly that, and ADR 0021
asks for two launches per cell — so this run does both: **8 legs, 2 independent
process launches per (engine, format) cell, session `kvab-20260913T194359Z`**.

Round 1 ran the four cells forward (ignis hq, ninfer hq, ignis BF16, ninfer
BF16) and round 2 ran them reversed, so machine drift over the hour spreads
across the cells instead of landing on whichever ran last.

## Matched capacity

The runbook's one rule: every leg gets the same **65,536-token** KV capacity —
1,024 pages — so the format comparison is not confounded with the page count a
byte budget happens to buy. Each engine's own log is the evidence:

- ignis hq: `--kv-pool-bytes 576M` -> `ignis.runtime.kv_pool` `page_count: 1024`
- ignis BF16: the 4 GiB default -> `page_count: 1024`
- ninfer, both formats: `--kv-capacity 65536` -> `KV capacity explicit resolved=65536 tokens pages=1024/5120`

All legs: `--max-context 40960`, `--prefill-chunk 1024`, CUDA graphs and prefix
reuse on, `ninfer-serve` stopped for every ignis leg and vice versa.

## Files

| file | what it is |
|---|---|
| `g3-<engine>-<format>-launch<N>.json` | the `ignis-bench g3` record for one leg (C=1, C=4, ITL) |
| `gpu-<engine>-<format>-launch<N>.csv` | that leg's 1 Hz `nvidia-smi` capture, started with the cell and stopped the moment it exited |
| `SESSION.txt` | the session id every leg carries |
| `probe-bf16-startup-*.txt` | the reference's startup logs from the graph-allowance incident below |

Reduce a capture over the ITL cell with
`python scripts/gpu-telemetry-summary.py <csv> --label "<leg> ITL" --last <the
record's own window length in seconds>`.

## The reference's CUDA-graph allowance is not deterministic

`ninfer-bf16-launch1` died at startup on its first attempt:

```
[error] ninfer-serve: CUDA Graph preparation consumed 235962368 bytes,
        exceeding the planned allowance of 100663296 bytes
```

Two probes with the same argv immediately afterwards — one with
`--model-id`, one without, ruling that flag out — both started cleanly, at the
identical memory plan (`runtime=6.58 GiB ... slack=7.66 GiB ... /96.00 MiB`),
and the retry of the leg itself succeeded on its first attempt. So the plan is
fixed at 96 MiB and what varies is what graph capture actually consumes in a
given launch. It is a reference-side startup flake, it fails loudly and before
any measurement, and no leg in this matrix was measured through it. Recorded
here rather than filed: it is not ignis's, and the reference is pinned
(`kernel/vendor/manifest.json`), not developed here.
