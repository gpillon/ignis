---
title: "Deadlines"
description: "How long a request may take, what happens when it runs over, and how a client states its own deadline."
sidebar:
  order: 7
---

Every route that runs the model holds a request to a deadline
([ADR 0058](../../adr/0058-every-route-has-a-deadline-and-a-passed-one-answers-what-was-generated.md)):
`--server-request-timeout` for one answered whole, `--server-stream-timeout`
for one that streams, both `2400` seconds by default and `0` for none, counted
from the request's admission (a WebSocket response's never from its queue, an
image job's from its start). Past it the request is cancelled, its lane freed,
and the request log's `ignis.request.cancelled` line says `cause: timeout`.
Before any ending the client gets everything the loop guard held back. What it
is answered is `--server-timeout-reply`:

| Route | `partial` (the default) | `error` |
|---|---|---|
| `/v1/chat/completions`, whole | `200`, the answer so far with `finish_reason: "length"` and its `usage`, and the header `ignis-stopped-by: request_timeout` | `504 request_timeout` |
| `/v1/chat/completions`, streamed | the `length` finish chunk, the usage chunk if `include_usage`, then `[DONE]` | one `{"error": {...}}` chunk, type and code `request_timeout`, then `[DONE]`, as an engine error ends a stream |
| `/v1/responses`, whole | `200`, `status: "incomplete"`, `incomplete_details.reason: "max_output_tokens"`, the output so far, and the same header | `504 request_timeout` |
| `/v1/responses`, streamed or WebSocket | `response.incomplete`, as above | `response.failed` with code `request_timeout` |
| `/v1/decide` | the question `not_completed` | the same |
| `/v1/images/*` | `504 request_timeout`; streamed, the images already made, then an `error` event with `request_timeout` | the same |

`length` and `max_output_tokens` are the only values the two APIs define for
an answer a limit cut, so every client already reads them; the header, on a
whole answer, and the log say it was the deadline. A browser on another origin
can read the header: the server exposes it, with `ignis-model-status` and
`Retry-After` (`Access-Control-Expose-Headers`). A tool call the deadline cut
mid-way comes back as one the cap cuts does: its name and the arguments
written so far, last, with `length` (an
`incomplete` `function_call` item on `/v1/responses`;
[Chat completions](chat-completions.md)). Under `error` no call is sent. A
decision has no partial answer, and an image cut
short is no image, so those two are errors either way. A chat request's media
acquisition keeps its own deadline, `--server-request-timeout` from its start;
past it the request is answered `504 request_timeout`, whatever
`--server-timeout-reply` says; at `0` only each fetch's timeout bounds it.

Upgrading from 0.7.1: the default deadline is 40 minutes, not 120 seconds; a
request past it is answered with what it generated, not a `504`, unless
`--server-timeout-reply error`; a chat stream and an image job now have a
deadline; `0` is accepted and means none. A chat completion the engine let go
of without a finish, answered whole, is a `500 engine_error`, no longer a
`504 request_timeout` logged as a timeout.

**Readiness.** A loaded model is not served the moment it is on the GPU: the
server first runs one two-token request through the scheduler so the decode
graphs are captured, and until that is done every `/v1` route answers `503`
with code `server_not_ready` and `Retry-After: 5` (a preflight and `/v1/docs`
still answer). The `ignis.process.ready` log event, with `warm_up_ms`, marks
the moment it admits requests; `make start` waits for it, since it waits for
`/v1/models` to answer. The warm-up is one request in the counters, and the
CPU mock, which has nothing to capture, is ready at once.

**Probes.** Two, both outside `/v1`: neither asks for an API key, and neither
counts as a use of the model (idle unload does not see them).
- `GET /health` is **liveness**: `200 {"status": "ok"}` whenever the process
  answers, and it answers from its start: the API listener is bound before
  the start-up download and load (`ignis.server.listening` names its
  address), then through the warm-up, every switch and reload, and an idle
  unload's load. Point a restarting probe here (Kubernetes `livenessProbe`),
  never at `/ready`: a loading server is alive, and restarting it would
  restart the load in a loop. It needs no `startupProbe` or
  `initialDelaySeconds` for the load.
- `GET /ready` is **readiness**: whether to send it requests. It answers
  `503 server_not_ready` with `Retry-After: 5` through the start-up load and
  until that first warm-up has run; while the model loads at start, every
  path but `/health` answers that way, whatever the key. After that it
  answers `200 {"status": "ok"}` while a model serves or is on its way:
  through a switch, an idle unload and the load after it
  (failed too, since the next request tries it again), and a failed switch
  that is reloading the old model. The one exception is a failed switch
  that could reload nothing, which leaves no model serving and none loading
  until another switch succeeds. There it answers `503 server_not_ready`
  with no `Retry-After`, since asking again does not help and a switch does.
  Point a load balancer, or a Kubernetes `readinessProbe` or `startupProbe`,
  here.

What the model itself is doing is `GET /v1/models`'s `status`.
