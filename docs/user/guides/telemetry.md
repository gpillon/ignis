---
title: "Telemetry and metrics"
description: "Logs, OpenTelemetry traces and the Prometheus metrics listener."
sidebar:
  order: 10
---

Everything goes through one structured log on stdout, in the format
`IGNIS_LOG_FORMAT` picks (pretty on a terminal, JSON otherwise);
`IGNIS_LOG_LEVEL` sets the level.

- The request lifecycle is the `ignis.request.*` events (`admitted` / `ttft` /
  `done`), with the request id doubling as the OTel trace id
  ([ADR 0012](../../adr/0012-request-id-as-trace-id.md)). A cancelled request
  ends with `cancelled` instead of `done`: its `cause` is `timeout` (past its
  deadline, `--server-request-timeout` or `--server-stream-timeout`, whatever
  it was answered), `consumer` (its client hung up, its queued
  response was cancelled, or a loop-guard rewind replaced it) or `shutdown`
  (a model switch stopped it), beside how long it ran, whether it had a lane
  and the tokens it generated.
- The `ttft` line's `duration_ms` runs from the submission to the first
  token, and `after_prefill_ms` from the prefill chunk that completed the
  prompt to that token: what the first token waited for once the prompt
  was read. The first token leaves at the end of the prefill, before the
  request is dealt a decode lane, so `ttft` comes before `admitted` and
  names a `lane` only when one was dealt first. A request that ends on its
  first token (`max_tokens: 1`, an end-of-sequence token, a stop sequence
  the token completes) never takes a lane and logs no `admitted` line: its
  `done` line names no `lane` and carries the prefill fields instead, `prompt_tokens`,
  `prefill_chunks_consumed`, `prefill_chunks_solo` and `prefilled_tokens`,
  as does a decision's.
- From 0.11.0, a request held to a grammar (`response_format`, see
  [Structured output](../api/structured-output.md)) says so on the line it
  ends with, `done` or `cancelled`: `structured.type` (`json_object` or
  `json_schema`), `structured.misses` (the sampled tokens the grammar
  rejected) and `structured.repairs` (the rejected tokens sampled again under
  the grammar). A request its caller's stop sequence ends says them on its
  `done` line, and one cut by its deadline or let go by its client on its
  `cancelled` line, with what its grammar did up to the cut. Any other
  request carries none of them, and a miss the engine could not repair is a
  miss with no repair, on a request that ends `error`. The server's totals
  are the `ignis_structured_*` series in the
  [metrics reference](../reference/metrics.md), which count the requests that
  completed (a `done` line, a stop sequence's included), as the decision
  series count only answered questions: a cancelled request's misses are on
  its line alone.
- A request with images says what they cost on the line it ends with,
  `done` or `cancelled`: `media.items`, `media.vision_tokens` (after any
  downscale to `--vision-max-tokens`) and `media.natural_vision_tokens`,
  `media.bytes`, `media.preprocess_seconds`, `media.encode_seconds` (the
  vision tower), `media.cache_hits` and `media.cache_misses`. A question
  past its deadline, or let go when its client left, reports there the
  encode it paid for before the cancel. A request that takes a decode lane
  reports the same numbers on `admitted` too, where they explain a slow
  first token, so a sum over requests reads `done` and `cancelled` only:
  each request's media are on exactly one of them, and the `done` a
  request logs after its `cancelled`, when it finished as the cancel
  landed, carries none. A decision answered by its prefill (`noul`,
  `choice`, `score`, and a `point` or `box` read from the heads) never
  takes a lane, so it has no `admitted` line. A text request carries none
  of them, and neither does a decide question the engine never ran, which
  logs no `ignis.request.*` line at all: one answered `engine_full` when
  its fan-out found no room, or one answered without the model, such as a
  head question whose image is not a grid the heads can read. It never
  reached the vision tower, and what preparing its image cost is on no
  line.
- The scheduler counters are the DEBUG event `ignis.scheduler.interval`, emitted
  whenever `waiting`, `running` or `kv_evictions` change
  ([ADR 0025](../../adr/0025-scheduler-interval-as-a-log-event.md)):

```text
2026-09-13T10:00:00.000Z DEBUG ignis.scheduler.interval - scheduler counters changed {tick=123, waiting=2, running=3, kv_evictions=1}
```

- The VRAM layout decided at load is the `ignis.runtime.vram_plan` event.

A request id names one request for the life of the process: a switch, a reload
and the load after an idle unload go on counting from the same number.

Writing a line never waits on the disk or the pipe behind stdout. A call site
hands its event to a queue, and a background thread is the only thing that ever
performs the real write: INFO, WARN and ERROR go through a bounded priority
channel that blocks briefly rather than drop, so an ERROR never silently
vanishes, while DEBUG and TRACE go through a bounded ring buffer that drops its
oldest entry on overflow rather than ever blocking. At shutdown the flush waits,
up to a fixed budget, only on the priority channel, since a pending DEBUG or
TRACE line is what that channel already permits losing. An attribute whose key
contains a sensitive whole word (`password`, `secret`, `token`,
`authorization`, `bearer`, `cookie`, `credential`) or word pair (`api_key`,
`private_key`) is logged as `[REDACTED]` by construction, not by call-site
discretion; a key that counts tokens, such as `token_capacity`, is not one.

With metrics on (the default; `--server-metrics false` turns them off), the Prometheus exposition covers the request lifecycle and
rejections, KV pool pages and evictions, the KV-RAM arena, retained slots and
what reuse held or gave up, the VRAM plan, decoded tokens, forced thinking
closes, and decision counters with the answer-mass histogram ([ADR 0017](../../adr/0017-prometheus-metrics.md)).
The Playground's **Monitor** tab scrapes the same exposition. It scrapes every 2-5 s while the tab is visible and stops while the tab is hidden (so a tab left open does not keep a scale-to-zero deployment awake), scraping again when the tab shows.
