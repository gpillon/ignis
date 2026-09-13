# The G4 run's zero-token `main` request was the pre-#137 reader, not the engine

- Kind: experiment
- Status: current
- Observed: 2026-09-13
- Last verified: 2026-09-13
- Scope: bench / G4 per-class cell, trace replay, SSE reader
- Related: [#147](https://github.com/gpillon/ignis/issues/147), [#137](https://github.com/gpillon/ignis/issues/137), [#128](https://github.com/gpillon/ignis/issues/128), [spec 04](../../.scratch/runtime/specs/04-reference-feature-floor.md)
- Superseded by: none

## Question

In the G4 gate run of 2026-09-12 (session `g4-20260912T162610Z`) the trace's
single `main` request, `req-001`, came back with `n_tokens: 0` on three of four
launches — on ninfer as well as ignis — and `ttft_ms` equal to `total_ms`. The
`main` cell read 0.000 tok/s and failed. Did the request produce nothing, and if
so, why?

## Evidence

**The records were written by a reader that ignored the thinking channel.** The
`ignis-bench.exe` that produced them was built at 2026-09-12 18:24 +0200
(`.scratch/g4-logs-build.log` in the #128 worktree, and the binary's own mtime).
#137's fix, which made the SSE reader count `delta.reasoning_content` as
generated tokens, is `e7b216a`, committed at 23:44 +0200 the same day.

**The engine did generate.** ignis's own request log for launch 1
(`ignis-launch1.log`, #128 worktree) against the same requests:

| harness record (`ignis-launch1-g4.json`) | server `ignis.request.done` |
|---|---|
| `req-001` main: 0 tokens, 451,654 ms | `request_id` 0, `interactive`: **16,000 tokens**, 451,567 ms, `finish_reason: length`, plus a WARN "generation produced reasoning but no content or tool call" |
| `req-007` sub: 0 tokens, 357,507 ms | `request_id` 6: 12,000 tokens, `length`, same WARN |
| `req-010` sub: 0 tokens, 347,897 ms | `request_id` 9: 12,000 tokens, `length`, same WARN |
| `req-003` sub: 22 tokens, 135,870 ms | `request_id` 2: 3,822 tokens, `stop` |

A turn that spends its whole budget thinking streams nothing but
`reasoning_content`; a reader that counts only `content` sees no token at all,
which is exactly `n_tokens: 0` with `ttft_ms == total_ms`. A turn that thinks
and then answers is counted only from its answer (22 of 3,822).

**Replayed with the current reader** (`ignis-bench` at main `7637893`,
ignis-server in the G4 profile: hq-e8-2b, max-context 262,144, auto 4 GiB pool,
1,024 prefill chunk):

| run | `req-001` ttft | n_tokens | total | ok |
|---|---:|---:|---:|---|
| alone (`bench/traces/req-001-alone.jsonl`, one line of the trace) | 3,167 ms | 2,229 | 57,567 ms | true |
| the whole 11-request trace, `--conc 8` | 3,135 ms | 15,613 | 467,511 ms | true |

Every sub request of the whole-trace replay also reported tokens (202 to
11,156). Raw output: `.scratch/g4-147-logs/` in the #147 worktree.

## Finding

Observed: under the G4 load `req-001` generates ~16,000 tokens on ignis, and the
current reader counts them, so the `main` cell is a measurement.

Observed: the 0.000 in the G4 verdict came from records written by an
`ignis-bench` built before #137; the zero-token rows on both engines are
thinking-only (or mostly-thinking) turns that reader could not see.

Inferred: the difference between 2,229 tokens alone and 15,613 under load is a
different greedy trajectory, not an engine defect — batched decode perturbs the
logits enough to change where a long thinking turn ends. Both are real
generations.

Separately, and independent of the reader: the harness recorded a 200 response
that delivered no token as `ok: true`, and the gate ranked a cell computed from
it. That is fixed with #147: `replay` records such a response as `ok: false`,
and `g4-gate` refuses a cell holding any request that failed or generated
nothing — including records written before the fix, which it reads by token
count rather than trusting `ok`. A cell of ours that decodes nothing at all
(single-token requests) is refused too, as the reference's already was. On the
#128 records `g4-gate` now refuses with: the `main` cell holds requests that
failed or generated no tokens — ignis's records: req-001 (launch 1), req-001
(launch 2); ninfer's records: req-001 (launch 2).

## Implications

- The #128 G4 records cannot decide the per-class cells; the gate has to be
  re-run with a post-#137 `ignis-bench`.
- A trace's `main` request exhausting its 16,000-token budget in thinking is a
  property of the workload. It is still decode work and still a throughput
  measurement.

- The refusal also covers a request that *failed* (`ok: false`), which used to
  be pooled silently as zero tokens and zero time. One transient failure in any
  launch now refuses the G4 check outright rather than skewing a cell.

## Limits and unknowns

- The whole-trace replay is one launch per engine, not the pooled two-launch
  gate of ADR 0021; it shows the cell is a measurement, not what it measures.
- The harness counts text-bearing chunks, not engine tokens (2,229 against the
  server's 2,546 for the isolated run). Both engines are read the same way.
