---
title: "The Responses API over a WebSocket"
description: "One socket per agent, only new items per turn, a real prewarm and a queue instead of a 503."
sidebar:
  order: 9
---

An agent loop of twenty tool calls sends its whole history twenty times over
plain HTTP, and the server renders it twenty times. `/v1/responses` in ignis
is OpenAI's Responses API, with its **WebSocket mode**: the client keeps one
socket open, sends only the new items each turn, and the server continues from
the last response. It is the wire Codex CLI and the OpenAI SDKs speak, and a
full engine queues the request instead of answering `503`.

## How it works

`POST /v1/responses` renders the same prompt tokens as chat completions for
the same conversation, so prompt reuse hits across both endpoints. `GET
/v1/responses` upgrades to a WebSocket that carries `response.create` events
with the HTTP body's fields, and the server answers with the same events as the
server-sent stream, one per text frame. Authentication is `Authorization:
Bearer` on the upgrade, or a `Sec-WebSocket-Protocol` entry
`openai-insecure-api-key.<key>`, which is never echoed or logged.

On the socket, `previous_response_id` continues from the connection's latest
response with only the new items. `generate: false` is a real prewarm: it
prefills the prompt, generates nothing and keeps the retained state an ordinary
request would publish, so the first real turn starts from it. A `stream_id`
runs requests in order within a stream and concurrently across streams, so one
socket can carry a main agent and its subagents; a `previous_response_id` from
another stream on a new `stream_id` forks a conversation.

When the engine is full the request gets `response.queued` and waits in one
server-wide first-in first-out queue shared by every connection. Its deadline
counts from admission, not from the wait ([deadlines](../api/deadlines.md)).
A client carries no retry logic. `response.cancel` cancels one response, and
closing the socket cancels everything it had running or queued.

This is also what lets the Playground run several agents at once. A browser
holds six HTTP/1.1 connections per origin, and with six agents streaming every
same-origin request waited, while the engine answered each in about a
millisecond. A socket is one connection however many streams it carries.

Every field, event and refusal: [The Responses API](../api/responses.md).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| 6 connections per origin over HTTP/1.1 cap the Playground at 6 parallel streams; the engine answered each request in about 1 ms | the Playground, six agents | [Browser connection limit](../../findings/2026-09-18-browser-connection-limit-caps-parallel-streams.md) |
| 16 active streams and 32 stream ids per connection; idle close after 300 s by default | the code, as the API page states it | [The Responses API](../api/responses.md) |
| As of 2026-09-29, none of the open inference servers surveyed (SGLang and llama.cpp HTTP only; Ollama refuses the upgrade with a 426) shipped the Responses WebSocket mode | a survey of public sources, dated | [Inference interfaces survey](../../findings/2026-09-29-openai-websocket-inference-interfaces.md) |

:::caution[No ignis latency figure]
No finding compares the socket with HTTP on ignis. OpenAI's own guide says a
socket makes a 20-call agent loop "up to roughly 40% faster"; that is OpenAI's
figure for its own service, not an ignis measurement. The survey row is a
snapshot of one date, so check it before you repeat it.
:::

## See it, control it

- **Routes**: `POST /v1/responses` and `GET /v1/responses` (the WebSocket).
  Codex CLI and the OpenAI and Agents SDKs connect as they do to OpenAI
  ([connect a client](../start/connect-a-client.md)).
- **Config**: [`server.socket_idle_close`](../reference/configuration.md#serversocket_idle_close)
  (300 s, `0` never, patchable live),
  [`server.stream_timeout`](../reference/configuration.md#serverstream_timeout)
  and [`server.request_timeout`](../reference/configuration.md#serverrequest_timeout).
- **Response fields**: `usage.input_tokens_details.cached_tokens`,
  `output_tokens_details.reasoning_tokens` and
  `output_tokens_details.rewound_tokens`.
- **Metrics**: `ignis_responses_sockets` and `ignis_responses_queued_requests`
  ([metrics](../reference/metrics.md)).
- **Playground**: [Chat](../playground/chat.md) talks to the server over this
  socket; it opens one for the turn that needs it, and closes an unused one
  after a minute while its tab is hidden.

## Limits

- Function tools only. `background: true`,
  non-function tools and other input item types are refused with OpenAI's
  error and a `param`. `store`, `metadata` and `prompt_cache_key` are accepted
  and change nothing.
- ignis stores no response. `previous_response_id` over HTTP is `400
  previous_response_not_found`; on a socket an id the connection does not hold
  is the same error, and the client resends the full input.
- `response.steer` is `steering_not_supported`.
- OpenAI's Agents SDK page says one response at a time per connection; ignis
  follows the API guide's `stream_id` multiplexing.

## Read more

- [The Responses API](../api/responses.md): the requests, the refusals, the events.
- [Deadlines](../api/deadlines.md): what a passed deadline answers.
- [Connect a client](../start/connect-a-client.md): Codex CLI and the SDKs.
- [Lanes that know who is asking](lanes.md): the queue's order.
