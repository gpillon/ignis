# 01 - the Responses API as OpenAI serves it: streaming, function tools and WebSocket mode

GitHub: #282

`POST /v1/responses` today is a thin v1: non-streaming (`stream: true` is a
400), no `tools`, no `instructions`, no reasoning in the answer, no
`previous_response_id`. The clients that speak the Responses API cannot use
ignis through it: Codex CLI has removed `wire_api = "chat"` and speaks only
Responses, and the OpenAI SDKs and the Agents SDK drive agent loops over
Responses, with a WebSocket mode for tool-heavy runs. This spec makes
`/v1/responses` the OpenAI Responses API, faithfully, over both of its
transports: HTTP with server-sent events, and the WebSocket mode OpenAI
launched on 2026-02-23.

The guiding rule, the owner's: **ignis speaks the interface the community
speaks.** Field names, event names, error codes and semantics are OpenAI's.
ignis extensions are additive fields (as `class`, `thinking_budget` and
`enable_thinking` already are) and never change what a standard field means.
A transport of ignis's own could be faster; it would also be one no client
speaks.

Research: `docs/findings/2026-09-29-openai-websocket-inference-interfaces.md`
(every event, field and error code below is sourced there).
ADRs: 0029 (cross-request reuse, unchanged by this spec), 0036 (the `/v1`
surface documents itself), 0012 (request id as trace id), 0017 (metrics),
0034 (a request that ends where its prefill ends).

## Problem Statement

A developer who points Codex CLI at ignis gets nothing: Codex requires the
Responses API with streaming and function tools, and ignis answers
`stream: true` with a 400 and has no `tools` field on that endpoint. The same
is true of an application written against `client.responses.create(stream=True)`
or `client.responses.connect()` in the OpenAI SDKs, or against the Agents SDK.
The only way to drive an agent loop on ignis is chat completions, which those
clients no longer offer or never offered.

An agent loop over HTTP also re-sends its whole history on every turn. ignis's
content-matched reuse (ADR 0029) already spares the re-prefill of that history,
but the client still has to transmit it, the server still has to parse it, and
there is no way to warm the prompt state before the first token is needed.
OpenAI's WebSocket mode is the community's answer to that: one persistent
connection, `previous_response_id` plus only the new items per turn, a
`generate: false` prewarm, and several independent streams multiplexed on one
connection. ignis offers none of it.

Finally, an agent run that fans out (subagents) meets the engine's admission
limit: over HTTP a request that finds the engine full gets a 503 and the
client must retry. A persistent connection can do better than that and queue.

## Solution

`/v1/responses` becomes the OpenAI Responses API, on the engine path chat
completions already uses:

- **HTTP:** `POST /v1/responses` serves `stream: true` as the standard
  Responses server-sent events (`response.created`, `response.output_item.added`,
  `response.output_text.delta`, ..., `response.completed`), and without it the
  standard response object. Requests carry `instructions`, function `tools`,
  `tool_choice`, and input items including `function_call` and
  `function_call_output`; the answer carries `reasoning`, `message` and
  `function_call` output items.
- **WebSocket mode:** `GET /v1/responses` upgrades to a WebSocket. The client
  sends `response.create` events whose body is the HTTP body; the server
  answers with the same events, byte for byte in payload, as the HTTP stream.
  On the socket a request can name `previous_response_id` and send only new
  items, can prewarm with `generate: false` (ignis really prefills and keeps
  the prompt state), and can run several **streams** (`stream_id`) in parallel
  on one connection. When the engine is full, a socket request **waits in a
  queue** instead of failing.

Everything a client of OpenAI's own endpoint does on a text model works
unchanged against ignis, with the base URL swapped. What ignis cannot do
(hosted tools, background responses, stored responses, steering) is refused
with OpenAI's own error for it, never silently ignored when ignoring it would
change the answer.

## User Stories

1. As a Codex CLI user, I want to point a custom `model_providers` entry with `wire_api = "responses"` at ignis, so that Codex runs on my local model.
2. As a Codex CLI user, I want to set `supports_websockets = true` on that provider, so that Codex keeps one connection open and sends only new items each turn.
3. As a Codex CLI user, I want Codex's `generate: false` prewarm to actually prefill my system prompt and tools, so that the first real turn starts with the prompt state already built.
4. As a Codex CLI user, I want Codex's tool calls (shell, apply_patch as a function tool, MCP tools) to come back as `function_call` output items, so that Codex executes them.
5. As a Codex CLI user, I want the model's thinking streamed as reasoning events, so that Codex shows it while the model thinks and never mixes it into the answer.
6. As an OpenAI SDK user, I want `client.responses.create(..., stream=True)` against ignis to yield the same typed events it yields against OpenAI, so that my code needs no ignis branch.
7. As an OpenAI SDK user, I want `client.responses.connect()` / `ResponsesWS` to work against ignis, so that I get WebSocket mode through the official client.
8. As an Agents SDK user, I want `OpenAIResponsesWSModel` with a custom base URL to run a multi-tool agent against ignis, so that my agents run locally.
9. As an agent developer, I want to continue a turn with `previous_response_id` and only the tool outputs, so that I do not re-send a 100K-token history on every tool call.
10. As an agent developer, I want an unknown or expired `previous_response_id` to fail with `previous_response_not_found`, so that my client knows to resend the full input.
11. As an agent developer, I want several named streams on one socket to run concurrently, so that my subagents run in parallel over one connection.
12. As an agent developer, I want requests on one stream to run strictly in order, so that a stream behaves like one conversation.
13. As an agent developer, I want to fork a conversation by continuing another stream's latest response on a new `stream_id`, so that I can branch an agent from a shared point.
14. As an agent developer, I want events of a named stream to carry its `stream_id`, so that I can demultiplex interleaved events.
15. As an agent developer, I want a request that finds the engine full to wait and be told it is queued (`response.queued`), so that I do not write retry loops.
16. As an agent developer, I want queued requests served in arrival order across all connections, so that one busy client cannot starve another.
17. As an agent developer, I want closing the socket to cancel everything it had running or queued, so that an abandoned agent stops using the GPU.
18. As an agent developer, I want a request-scoped error to leave the socket and the other streams running, so that one bad request does not kill the session.
19. As an agent developer, I want `tool_choice` (`auto`, `none`, `required`, a named function) honoured as chat completions honours it, so that I can force or forbid a tool call.
20. As an agent developer, I want a non-function tool type (web search, file search, code interpreter, custom) refused with a 400 naming the tool, so that I learn at once that ignis does not host tools.
21. As an agent developer, I want `instructions` treated as the system prompt, under the server's instruction policies, and not carried over from the previous response, so that the semantics match OpenAI's.
22. As an agent developer, I want reasoning items I pass back in `input` accepted, so that a client that round-trips them (Codex does) is not refused.
23. As an agent developer, I want a response that hit `max_output_tokens` to end as `response.incomplete` with `incomplete_details.reason: "max_output_tokens"`, so that I can tell a cut answer from a finished one.
24. As an agent developer, I want `usage.input_tokens_details.cached_tokens` to report the prompt tokens ignis resumed from retained state, so that I can see reuse working.
25. As an application developer using HTTP only, I want the non-streaming response to have the same `output` items as the streamed one, so that switching `stream` does not change what I parse.
26. As a browser application developer, I want to authenticate the socket with the `openai-insecure-api-key.<key>` subprotocol, so that a page, which cannot set headers on a WebSocket, can connect to a keyed server.
27. As an operator, I want the API key to guard the socket exactly as it guards HTTP, so that exposing the server does not open an unauthenticated door.
28. As an operator, I want the key never to appear in logs or traces when it arrives as a subprotocol, so that logs stay shareable.
29. As an operator, I want gauges for open sockets and queued socket requests, so that I can see load that is waiting rather than refused.
30. As an operator, I want each socket request to be one request in the request log and traces, with its own request id, so that WebSocket traffic is as observable as HTTP.
31. As an API reference reader, I want the WebSocket upgrade and the streaming variant documented at `/v1/docs/`, so that the reference covers the whole endpoint.
32. As an ignis user, I want ignis extensions (`class`, `thinking_budget`, `enable_thinking`, `reasoning_effort`, ...) accepted on `response.create` as on the HTTP body, so that the socket loses nothing HTTP has.
33. As a qwen-code user, I want chat completions unchanged, so that nothing I use today moves.
34. As a client on a shared socket, I want to cancel one response (`response.cancel`, the Realtime API's event) without closing the connection, so that stopping one stream does not kill the others.

## Implementation Decisions

**One module produces the Responses event sequence.** A deep module turns a
submitted request's scheduler event stream (through the existing output
decoder, reasoning split and tool-call scanner) into the ordered list of
Responses events for one response: `response.created`, `response.in_progress`,
then per output item `response.output_item.added`, its content events
(`response.reasoning_text.delta`/`.done`, `response.content_part.added`,
`response.output_text.delta`/`.done`, `response.content_part.done`,
`response.function_call_arguments.delta`/`.done`), `response.output_item.done`,
and one terminal event: `response.completed`, `response.incomplete` or
`response.failed`. Every event carries a `sequence_number` increasing from 0
within the response. It has three consumers and no other producer: the SSE
writer (`event: <type>` + `data: <json>` per event), the WebSocket writer (one
text frame per event, plus `stream_id` for a named stream), and the
non-streaming handler, whose body is the terminal event's `response` object.
So the three transports cannot disagree on a payload.

**Output items.** In generation order: a `reasoning` item when the thinking
channel produced text (`content: [{type: "reasoning_text", text}]`,
`summary: []`, no `encrypted_content`), a `message` item with one
`output_text` part (`annotations: []`) when the content channel produced text,
and one `function_call` item per scanned tool call (`id`, `call_id`, `name`,
`arguments` as a JSON string, `status`). Ids are `resp_<request id>`,
`msg_...`, `rs_...`, `fc_...`, derived from the request id (ADR 0012).
`thinking_budget_forced_at` stays as today's top-level extension.

**Input items.** `input` is a string (one user message) or a list of items:
`message` (with or without `type`; roles `user`, `system`, `developer`,
`assistant`; `content` a string or parts `input_text`, `input_image` under
`--vision`, whose flat `image_url` string is translated to the nested
`image_url` object chat completions takes, and `output_text` on assistant
messages), `function_call`, `function_call_output`, and `reasoning`. Items map
onto the existing chat message path: consecutive `function_call` items attach
as `tool_calls` to the preceding assistant message (or a new one),
`function_call_output` becomes a tool message, and a `reasoning` item becomes
the reasoning content of the assistant turn that follows it, which the chat
template keeps or drops by its own rules (and `preserve_thinking`). The prompt
therefore renders exactly as the same conversation would through chat
completions, and ADR 0029's content-matched reuse hits across both endpoints.
Any other item type (`item_reference`, `computer_call`, ...) is a 400 with
`param` naming the item.

**Request fields.** `instructions` is rendered as a system message placed
first, through the #209 instruction policies like any system message, and is
never inherited from a previous response (OpenAI's rule). `tools` accepts
`type: "function"` in the Responses flat shape (`name`, `description`,
`parameters`, `strict`), converted to the shape the chat template already
renders (spec server/07); any other tool type is a 400 with
`param: "tools[i].type"`. `tool_choice` and `parallel_tool_calls` behave as on
chat completions. `max_output_tokens`, `temperature`, `top_p` and the ignis
sampling and thinking extensions behave as today. `reasoning.effort` resolves
through the existing reasoning-effort path; `reasoning.summary` is accepted
and produces nothing (no summaries). `text.format` of type `text` or absent is
served; any other format (`json_schema`, `json_object`) is a 400 (structured
output is not served; ignoring it would change the answer). `background: true`
is a 400. Fields that do not change the answer are accepted and ignored:
`store`, `include` (including `reasoning.encrypted_content`), `metadata`,
`user`, `safety_identifier`, `prompt_cache_key`, `prompt_cache_retention`,
`service_tier`, `truncation` (`disabled` behaviour), `stream_options`,
`client_metadata`, `text.verbosity`, and unknown top-level fields (tolerant
reader, as today).

**Usage.** `input_tokens`, `output_tokens`, `total_tokens`,
`input_tokens_details.cached_tokens` (the prompt tokens this request resumed
from a retained prefix or prompt checkpoint, 0 when none; the engine reports
reuse today only as telemetry facts feeding the metrics, so it gains a
per-request reused-token count delivered on the request's own event stream),
`output_tokens_details.reasoning_tokens` (tokens on the thinking channel).

**Terminal states.** `completed` on a stop; `incomplete` with
`incomplete_details.reason: "max_output_tokens"` on a length stop; `failed`
with an `error` object on an engine error. A request timeout mid-stream ends
the stream with `response.failed` (code `request_timeout`); on non-streaming
HTTP it stays today's 504. Errors before the response exists (validation,
unknown model, context exceeded, oversized) keep today's HTTP status and body
on HTTP, and are an `error` event on the socket.

**HTTP.** `stream: true` is served. `previous_response_id` on HTTP is a 400
`previous_response_not_found` with `param: "previous_response_id"`: ignis
stores no responses, so every id is unknown there (the same answer OpenAI
gives an id it does not have). `generate` on HTTP is a 400 (it is a
WebSocket-mode field; ignoring `false` would generate). An engine that is full
stays a 503, as today.

**WebSocket upgrade.** `GET /v1/responses` with an upgrade request switches
protocols (101); any other `GET` is a 400. It is registered through the same
`routes!` seam as the other operations and documented as a `get` operation on
`/v1/responses` whose description names the event model and links the
finding, so ADR 0036's path set grows by `get /v1/responses` deliberately.
Authentication is the existing key check, which now accepts either
`Authorization: Bearer <key>` on the upgrade request or a
`Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<key>` (the OpenAI
Realtime browser convention); a missing or wrong key is the existing 401
before the upgrade. The server selects as the connection's subprotocol the
first offered entry that is not a credential, and never echoes the credential
entry. The credential entry is redacted from every log line and span that
records request headers. The socket is exposed exactly as HTTP is: same
listener, same `--expose` behaviour, same open-by-default when no key is set.
A WebSocket has no CORS, and no `Origin` check is added: HTTP already answers
every origin (`Access-Control-Allow-Origin: *`), so an unkeyed server is
already reachable from any page, and an exposed one is always keyed
(ADR 0028).
The dependency is axum's `ws` feature (`tokio-tungstenite` 0.29); a probe on
2026-09-29 found it adds no host `windows-sys` chain.

**Client events.** Text frames carrying one JSON event each. `response.create`
takes the HTTP body's fields flat beside `type`, plus `stream_id` and
`generate`; `stream` is accepted and ignored (Codex always sends it). A
`response.steer` is answered with an `error` event, code
`steering_not_supported`. **`response.cancel`** is the one ignis extension
event, borrowed with its name and shape from OpenAI's Realtime API because
Responses WebSocket mode has no cancel: `{type: "response.cancel",
response_id?}` cancels that response (or, without `response_id`, the default
stream's active response) through the existing engine cancel, or removes it
from a queue if it has not started; it ends with a terminal
`response.incomplete` event whose response has `status: "cancelled"` (a
standard response status). An id that is not running or queued on this
connection is an `error` event, code `response_not_found`. Standard clients
never send it and never see it. Any other `type`, a binary frame or a frame that is
not a JSON object is an `error` event with type `invalid_request_error` and
`param: "type"` (or none for a malformed frame); none of these close the
socket.

**Error events.** `{type: "error", status, error: {type, code, message, param},
stream_id?}`: `status` is the HTTP status the same failure has on HTTP, the
`error` object is today's error body, and `stream_id` is present when the
error belongs to a named stream. Documented codes used verbatim:
`previous_response_not_found`, `invalid_stream_id`,
`websocket_stream_limit_reached`, `steering_not_supported`,
`response_not_found`.

**Streams.** A `response.create` without `stream_id` runs on the connection's
default stream; one with `stream_id` runs on that named stream. A stream id is
1-256 characters of `[A-Za-z0-9_.-]`, else `invalid_stream_id`. A connection
accepts 32 distinct named stream ids (the default stream does not count), the
33rd is `websocket_stream_limit_reached`. Requests on one stream run in order
and never overlap; requests on different streams run concurrently. A
connection has at most 16 active responses across its streams, active meaning
handed to the engine (running, or waiting in the socket admission queue);
further `response.create` events wait on the connection, in order, until one
finishes.
The standard's word is "stream"; this spec never calls it a lane, which in
ignis is a decode slot (CONTEXT.md).

**Connection-local continuation.** Each connection keeps, per stream, its
latest finished response: the full item history that response saw (its
context plus its input) followed by its output items. A `response.create`
with `previous_response_id` resolves the id among the connection's entries
**when the event is received**, copying that history, so a request that then
waits (behind its stream, the connection's limit or the admission queue)
cannot lose its parent to the source stream advancing; OpenAI resolves at
`response.in_progress` because it has no admission queue. The request
prepends that history to its own `input` and renders the whole conversation: the same prompt the client would
have produced by re-sending everything. A fork is a `previous_response_id`
from another stream on a new `stream_id`. An id that is not a stream's latest,
belongs to another connection, or was never seen is `previous_response_not_found`.
A same-stream continuation that fails with a 4xx or 5xx evicts the id it
referenced; a failing fork does not evict its source. The cache never chooses
retained state: reuse stays ADR 0029's content match, so the connection is not
a session identity and ADR 0029 is not amended. The cache lives and dies with
the connection; `store` does not change any of this.

**Prewarm (`generate: false`).** The request renders its prompt as a
generating request would and is submitted as a **warm-up request**: a new
request kind in the core that, like a decision (ADR 0034), ends where its
prefill ends and generates nothing, and, unlike a decision, publishes the
retained state an ordinary request publishes: the retained prefix at the
system-block boundary and the prompt checkpoint at the generation opener.
Its response has no output items, `output_tokens: 0` and status `completed`;
its id is cached like any response (history = its input), so the next turn's
`previous_response_id` chains from it, and that turn resumes from whichever
of the two its content matches. A prewarm carrying only `instructions` and
`tools` is served by the retained prefix (the real turn adds a user message
before the opener, so the checkpoint cannot match); one carrying the user
message too is served by the checkpoint. Both are best effort: when no
retained slot is free nothing is kept (ADR 0029) and the warm-up still
completes.

**Socket admission queue.** When the engine answers a socket request with
"full", the request enters one server-wide FIFO queue shared by every
connection, and the client receives `response.created` (status `queued`) and
`response.queued`. When any request finishes or is cancelled, the queue's head
retries admission; an admitted request emits `response.in_progress` and
proceeds. Only "full" queues: context exceeded, oversized and unknown model
are immediate `error` events. `--request-timeout` counts from admission, not
from queueing. Closing a connection removes its queued requests. HTTP never
queues (503, as today).

**Connection lifetime.** No 60-minute cap and no idle timeout: a local server
has no reason to impose OpenAI's. Closing the socket, by either side or by a
dropped connection, cancels every response the connection has running
(through the existing engine cancel), drops its queued requests and discards
its cache. Pings are answered.

**Observability.** Each `response.create` is one request to the engine, with
its own request id and root span (ADR 0012), in the request log and the
existing metrics like an HTTP request. New gauges: open Responses sockets and
requests waiting in the socket admission queue (ADR 0017 naming).

**Chat completions** is untouched.

## Testing Decisions

A good test here drives the server from outside, as a client would, over a
deterministic mock engine, and asserts on what a client sees: the event
sequence, the payloads, the error events, the metrics. It never reaches into
the event module's internals or the cache's layout.

- **Seam 1: HTTP `/v1/responses` over the in-process router** (`tower`
  `oneshot` over `MockCompute` / `GatedCompute`). Prior art:
  `crates/server/tests/openai_http.rs`, `openai_http_toolcalls.rs`,
  `openai_http_thinking.rs`. Covers the whole event model once: event order
  and `sequence_number`, reasoning / message / function_call items, input
  item mapping (a Responses conversation renders the same prompt as the
  equivalent chat completions request), `tool_choice`, the 400s (non-function
  tools, `text.format`, `background`, `previous_response_id`, `generate`),
  `incomplete` on length, `failed` on engine error, usage including
  `cached_tokens`, non-streaming body = terminal event's response.
- **Seam 2: the WebSocket on a live listener** (a real socket, a
  `tokio-tungstenite` 0.29 test client, `MockCompute` / `GatedCompute`). Prior
  art: `crates/server/tests/serve_on_listener.rs`, `support/live_server.rs`,
  `checkpoint_lineage.rs` for reuse assertions. Covers only what is socket
  specific: upgrade and 401 by header and by subprotocol, selected
  subprotocol never the credential, events identical in payload to seam 1's,
  default stream and named streams (order within, concurrency across,
  `stream_id` on events, the 32 and 16 limits), continuation hit, miss,
  eviction and fork, `generate: false` followed by a continuation that adds a user
  message and resumes from the warm-up's state (retained-state counters), the admission queue under a gated
  full engine (`response.queued`, FIFO across two connections, timeout from
  admission), close cancelling running and queued requests (cancel counter),
  error events not closing the socket, `response.steer` refused,
  `response.cancel` of a running and of a queued response leaving the other
  streams running.
- The warm-up request kind is exercised through seam 2; a core scheduler test
  is added only if the mock engine cannot show the checkpoint being taken.
- `openapi_http.rs`'s path set gains `get /v1/responses`.
- No new GPU test: the engine path is chat completions'. The owner's live
  acceptance is a real Codex CLI session (acceptance 13).

## Acceptance

1. `POST /v1/responses` with `stream: true` answers `text/event-stream` with the standard event sequence and increasing `sequence_number`; without it, the body equals the terminal event's `response`.
2. Thinking streams as `response.reasoning_text.*` in a `reasoning` item, content as `response.output_text.*` in a `message` item, tool calls as `response.function_call_arguments.*` in `function_call` items; the non-streaming body carries the same items.
3. `instructions`, function `tools`, `tool_choice`, and input items `message` / `function_call` / `function_call_output` / `reasoning` are accepted, and a Responses conversation renders the same prompt tokens as the equivalent chat completions conversation.
4. Non-function tools, `text.format` other than `text`, `background: true`, `generate` and `previous_response_id` on HTTP, and unknown input item types are 400s with OpenAI's error shape and a `param`; the accepted-and-ignored fields listed above do not fail.
5. Length stops end `incomplete` (`max_output_tokens`), engine errors `failed`; usage reports `cached_tokens` and `reasoning_tokens`.
6. `GET /v1/responses` upgrades to a WebSocket; the key is checked by bearer header or `openai-insecure-api-key.<key>` subprotocol, the credential is never echoed nor logged, and `get /v1/responses` is in the OpenAPI document.
7. `response.create` over the socket produces the same event payloads as HTTP streaming; `stream` is ignored; `response.steer` gets `steering_not_supported`; a bad frame gets an `error` event and the socket stays open.
8. `response.cancel` (ignis extension, Realtime's shape) cancels one running or queued response with a terminal `response.incomplete` whose status is `cancelled`, leaves the connection's other responses running, and answers an unknown id with `response_not_found`.
9. Default and named streams: FIFO within a stream, concurrent across streams, `stream_id` on named-stream events, `invalid_stream_id`, `websocket_stream_limit_reached` at 33 named ids, at most 16 active responses per connection with the rest waiting in order.
10. `previous_response_id` continues from the connection's cache with only new items; unknown, superseded or foreign ids get `previous_response_not_found`; a failed same-stream continuation evicts its parent, a failed fork does not.
11. `generate: false` prefills without generating, publishes the retained prefix and prompt checkpoint an ordinary request would when a retained slot is free, completes with no output, and a continuation from its id (with a user message added) resumes from the state the warm-up left.
12. A socket request that finds the engine full is queued server-wide in FIFO order with `response.queued`, is admitted when capacity frees, times out from admission; closing the socket cancels its running and queued requests; the open-socket and queued gauges are exported.
13. Owner's live check: Codex CLI with a custom provider (`wire_api = "responses"`, `supports_websockets = true`) completes a multi-tool task against ignis, prewarm included.
14. `cargo test` passes workspace-wide; chat completions' tests are unchanged.

## Out of Scope

- **Continuation from the exact generated tokens** (keeping the KV state after
  generation and appending only new items). It would spare the re-prefill of
  the last assistant output but changes what the model sees (the template
  drops thinking); a future opt-in ignis extension with its own ADR and
  measurement.
- `response.steer` (refused with its standard code), `response.inject` (beta,
  for OpenAI-hosted agents), `background` responses, stored responses
  (`GET /v1/responses/{id}`, `POST .../cancel`, `previous_response_id` over
  HTTP), the Conversations API, compaction (`context_management`,
  `/responses/compact`).
- Hosted tools (web search, file search, code interpreter, image generation,
  MCP run by the server, custom/freeform tools), structured outputs
  (`text.format` `json_schema`), reasoning summaries and
  `encrypted_content`, logprobs.
- The Realtime API (`/v1/realtime`): a different, audio-first protocol.
- The HTTP 426 fallback that makes Codex drop to HTTP: not needed once the
  socket is served.
- Any WebSocket for chat completions: no standard exists.
- The Playground's move onto the socket: spec 02.

## Further Notes

- Codex's provider defaults: `supports_websockets` is `false` for custom
  providers, so a Codex user opts in; Codex over plain HTTP works from
  acceptances 1-5 alone and still gets ADR 0029's prefill reuse.
- The WebSocket's gain on ignis is smaller than OpenAI's quoted ~40%: ignis
  already skips the re-prefill of a re-sent history by content. The socket
  saves re-sending and re-parsing the history, adds the prewarm, the queue
  and multiplexed streams (which is what lifts the browser's six-connection
  cap for the Playground, spec 02).
- Not established by the research: whether Codex's `generate: false`
  prewarm carries the first user message or only `instructions` and `tools`;
  the prewarm decision above holds either way.
- Codex sends extra handshake headers (`OpenAI-Beta: responses_websockets=2026-02-06`,
  `x-codex-*`); they are ignored.
