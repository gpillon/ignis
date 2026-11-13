---
title: "The Responses API"
description: "POST /v1/responses and its WebSocket mode."
sidebar:
  order: 3
---

`/v1/responses` is OpenAI's Responses API on the same engine path as chat
completions (GitHub #282): the wire Codex CLI, the OpenAI SDKs'
`responses.create` / `responses.connect` and the Agents SDK speak.

- **Over HTTP**, `POST /v1/responses` answers the response object, or with
  `stream: true` the standard server-sent events (`response.created`,
  `response.output_item.added`, `response.output_text.delta`, ...,
  `response.completed`), each with a `sequence_number`. The thinking comes
  back as a `reasoning` item (`response.reasoning_text.*` when streaming),
  the answer as a `message` item, each tool call as a `function_call` item.
  The first token is sent alone, in the first `response.reasoning_text.delta`
  or `response.output_text.delta`, as soon as the prompt's prefill ends and
  before any decode round, and each round's tokens follow it
  ([how ignis serves a request](../concepts/how-ignis-serves.md)).
- **Requests** carry `input` (a string, or `message`, `function_call`,
  `function_call_output` and `reasoning` items), `instructions`, function
  `tools`, `tool_choice`, `parallel_tool_calls`, `max_output_tokens`,
  `temperature`, `top_p`, `reasoning.effort` and the ignis sampling and
  thinking extensions. A conversation renders the same prompt tokens as the
  same conversation through chat completions, so prompt reuse hits across
  both endpoints.
- **What it cannot do is refused**, with OpenAI's error and a `param`, never
  ignored where ignoring it would change the answer: tools that are not
  functions, `text.format` other than `text` (no structured output on this
  route yet; chat completions serve `json_object` and `json_schema` from 0.11.0,
  [Structured output](structured-output.md)),
  `background: true`, and an input item of another type. `store`,
  `metadata`, `prompt_cache_key`, `service_tier`, a function tool's `strict`
  (its call's arguments are not held to its `parameters`, GitHub #582) and
  the like are accepted and change nothing; `include: ["reasoning.encrypted_content"]` adds the
  reasoning as the model saw it ([Chat completions](chat-completions.md)). ignis stores no response, so `previous_response_id` on
  HTTP is a `400 previous_response_not_found`.
- A length stop ends `incomplete` (`max_output_tokens`), an engine error
  `failed`; `usage` carries `input_tokens_details.cached_tokens` and
  `output_tokens_details.reasoning_tokens`.

**WebSocket mode.** `GET /v1/responses` upgrades to a WebSocket, keyed like
HTTP: `Authorization: Bearer <key>` on the upgrade, or a
`Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<key>` (never echoed,
never logged). The client sends `response.create` events carrying the HTTP
body's fields; the server answers with the same events as the HTTP stream,
one per text frame. On the socket:

- **`previous_response_id`** continues from the connection's latest response
  on that stream with only the new items, so an agent loop does not re-send
  its whole history on every tool call. An id the connection does not hold is
  `previous_response_not_found`, and the client resends the full input.
- **`generate: false`** is a real prewarm: the prompt is prefilled, nothing
  is generated, and the retained state an ordinary request would publish is
  kept, so the first real turn starts from it.
- **Streams**: a `stream_id` (1–256 characters of `[A-Za-z0-9_.-]`, up to 32
  per connection) runs requests in order within a stream and concurrently
  across streams, up to 16 active per connection; events of a named stream
  carry its `stream_id`. A `previous_response_id` from another stream on a
  new `stream_id` forks a conversation.
- **A full engine queues instead of refusing**: the request gets
  `response.queued` and waits in one server-wide first-in first-out queue
  shared by every connection, where HTTP would answer `503`.
  A queued request's deadline, `--server-stream-timeout`, counts from its
  admission, not from its wait ([deadlines](deadlines.md)).
- `response.cancel` (`{type, response_id?}`, an ignis extension in the shape
  of OpenAI's Realtime API) cancels one running or queued response; closing
  the socket cancels everything it had running or queued. `response.steer` is
  `steering_not_supported`. A bad frame is an `error` event and the socket
  stays open; a connection has no time limit.
- **An idle socket is closed.** A socket with nothing running or queued on it
  and no text or binary frame from its client (a ping does not count) for
  `--server-socket-idle-close` seconds (file key `server.socket_idle_close`,
  default `300`) is closed by the server with a normal close frame (`1000`)
  whose reason says so. The client's next request opens a new socket; a
  `previous_response_id` from the old one is `previous_response_not_found`
  there, so it sends the full input. `0` never closes. The value is patchable
  live and reaches the sockets already open within a second.

The Playground's chat talks to the server over this socket, so its parallel
agents are not held to the browser's six connections per host. It opens the
socket for the turn that needs it and opens another after any close; while
its tab is hidden it closes a socket no turn is using after a minute. The open
sockets and the queued socket requests are gauges on `/metrics`.
