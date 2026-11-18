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
- From 0.11.0, a request held to a grammar (`response_format`, or the calls
  of a held tool, see [Structured output](../api/structured-output.md))
  says so on the line it ends with, `done` or `cancelled`: `structured.type`
  (`json_object`, `json_schema`, or `tool` for held tool calls; a request
  with both counts under its `response_format`'s, its calls' misses included),
  `structured.misses` (the sampled tokens the grammar
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
  vision tower), `media.cache_hits` and `media.cache_misses`. From 0.11.0, a
  request whose images together pass `--vision-max-tokens` or the media byte
  budget loses its oldest ([Vision](../configuration/vision.md)), and says so
  on the same lines: `media.dropped_items`, `media.dropped_vision_tokens` and
  `media.dropped_bytes`, 0 when none was dropped; `media.items`,
  `media.vision_tokens` and `media.bytes` then count the images the model
  saw. A question
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
- From 0.11.0, a request refused for its images leaves the WARN event
  `ignis.media.refused`: its media are acquired before admission, so the
  refusal has no request id and no `ignis.request.*` line. It carries the
  answer's `status` and `code` (`invalid_media`, `media_budget_exceeded`,
  `vision_disabled`, `video_unsupported`, `media_fetch_failed`,
  `media_fetch_timeout`, or `request_timeout` when the deadline passed while
  the images were prepared), the `part` the acquisition refused
  (`message 3 content part 1`; none for a limit on the whole request, nor for
  a refusal the request's parts meet before their images are acquired, such
  as an image in a system message), never the message the client read, which
  for a fetch names its URL, and, for a budget, the `budget`
  (`encoded_bytes`, `decoded_pixels`, `media_items`, `raw_patches`,
  `vision_tokens` or `item_vision_tokens`), its `limit` and the amount
  `requested` ([Troubleshooting](../troubleshooting.md#vision_disabled-or-media_budget_exceeded)):

```text
2026-11-18T10:00:00.000Z  WARN ignis.media.refused - request media refused {budget="encoded_bytes", code="media_budget_exceeded", limit=268435456, part="message 3 content part 1", requested=268500000, status=400}
```

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
From 0.11.0, every tool call block the model writes is checked against its
tool's `parameters` as it closes, on chat completions and on the Responses API
(HTTP and WebSocket), and counted, the check changing nothing the response
carries: `ignis_tool_calls_total` by outcome (`valid`; `invalid`, arguments that
break the schema or a call to a tool the request did not declare; `dropped`, a
closed block that did not parse, which the client never sees),
`ignis_tool_call_violations_total` by kind (`missing`, `type`, `enum`,
`unknown_parameter` where the schema closes its object with
`additionalProperties: false`, `constraint` for a pattern, a format, a bound, a
length or a count, and `undeclared_tool`; a call counts once under each kind it
has), and `ignis_tool_strict_outside_subset_total`, the strict tools
(`strict: true`) whose schema lies outside OpenAI's strict subset, once per
request that sends one, counted as it arrives, before the 27B refuses it with
a `400` on chat completions and on the Responses API
([Held tools](../api/structured-output.md#held-tools-strict-true); read on a
model that serves structured output, so not on Flash-Next). A call still open when
generation ends is not counted: its `finish_reason` says it was cut.
`ignis_tool_call_undeclared_parameters_total` counts, once per call, valid or
not, the calls holding a parameter that an object with no
`additionalProperties` does not declare, which JSON Schema admits, in a tool
the operator's switch at `held` would hold: what `held`, which closes such
objects, would change that `unknown_parameter` does not show
([Holding every tool](../api/structured-output.md#holding-every-tool)); never
on a model that holds no tool (Flash-Next). Beside
them, `ignis_tool_call_values_split_total` counts, in the calls to tools that
are not held, the string values (of a
parameter the tool declares a string) that the parser ended at a tag inside
them and whose text after the tag it dropped: text after a `</parameter>` that
no next parameter follows, or after a `</function>` not followed by
`</tool_call>`, on a line of its own or inside one. The rule held tools' calls
are read by, the confirmed close
([Held tools](../api/structured-output.md#held-tools-strict-true)), would have
kept that text in the value: under it a value
ends only at a line that opens a declared parameter not yet written, or that
closes the call once every required parameter is written. It measures what
that rule would save if every tool were read by it. Not counted: a value whose text the parser
moved into another parameter, dropping nothing (an opener of a parameter
already written or not declared, which may be the model's own), a value whose
call closes with a required parameter missing, which that rule would not close
at all, and one whose end under that rule lies past the call's first
`</tool_call>`. The
family appears with the first call block or such a strict tool, and the check
runs only with metrics on.

The Playground's **Monitor** tab scrapes the same exposition. It scrapes every 2-5 s while the tab is visible and stops while the tab is hidden (so a tab left open does not keep a scale-to-zero deployment awake), scraping again when the tab shows.
