---
status: accepted
amends: [ADR-0017]
---

# ADR 0058 — every route has a deadline, and a passed one answers what was generated

## Status

Accepted (2026-10-18, owner decisions). Supersedes the 120 s default of
`--server-request-timeout` (GitHub #327) and its being the only deadline, on
non-streaming text alone. Amends ADR 0017 (a sixth `outcome` of
`ignis_image_jobs_total`, `timed_out`) and extends ADR 0052's "everything held
back flushes on a cancel and a timeout" to every surface with a deadline.

## Context

Before this decision a deadline existed on some routes and not others, all of
them `server.request_timeout` (120 s, `1..=3600`):

- `/v1/chat/completions` answered whole: a `504 request_timeout` at 120 s
  from admission, the request cancelled (#327) and everything it had
  generated lost. Streamed: **no deadline at all**; only a disconnect ended it.
- `/v1/responses` over HTTP: a 504 answered whole, `response.failed` with
  `request_timeout` streamed; over its WebSocket, `failed` from admission.
- `/v1/decide`: each question's readout bounded by the same 120 s.
- Image jobs: **no deadline**, streamed or not. Only the prompt enhancer's
  rewrite was bounded (its job then ran on the prompt as sent).
- A chat request's media acquisition: its own deadline of the same value.

At 120 s an agent's long reasoning step was cut and, on a 504, generated again
from nothing by the retry. An inference engine leaves a request to its own end
by default; a deadline is there to end what a wedged engine holds, and a whole
answer thrown away for being late helps no client.

## Decision

**Every route that runs the model has a deadline; `0` is none; a passed one
answers what was generated, as a length limit's cut does, unless the operator
asks for an error.** Three server fields, all live-patchable (`PATCH
/v1/config`, read per request):

| Field | Flag / env | Default | Range |
|---|---|---|---|
| `server.request_timeout` | `--server-request-timeout` / `IGNIS_SERVER_REQUEST_TIMEOUT` | `2400` (was `120`) | `0..=3600` (was `1..=3600`) |
| `server.stream_timeout` | `--server-stream-timeout` / `IGNIS_SERVER_STREAM_TIMEOUT` | `2400` | `0..=3600` |
| `server.timeout_reply` | `--server-timeout-reply` / `IGNIS_SERVER_TIMEOUT_REPLY` | `partial` | `partial`, `error` |

The ceiling stays 3600: the forty-minute default is under it, an hour still
catches a fat-fingered value, and whoever wants no deadline says `0`. A zero
deadline is an instant a decade away (`deadline::after`), so every timer keeps
one shape.

The contract per route:

| Route | Deadline, counted from | `partial` (default) | `error` |
|---|---|---|---|
| chat completions, whole | `request_timeout`, admission | `200`, the answer so far, `finish_reason: "length"`, `usage` filled, header `ignis-stopped-by: request_timeout` | `504 request_timeout` |
| chat completions, streamed | `stream_timeout`, admission | its deltas, the `length` finish chunk, the usage chunk if `include_usage`, `[DONE]` | its deltas, one `{"error": {type, code: "request_timeout"}}` chunk, `[DONE]` |
| `/v1/responses`, whole | `request_timeout`, admission | `200`, `status: "incomplete"`, `incomplete_details.reason: "max_output_tokens"`, the output so far, the same header | `504 request_timeout` |
| `/v1/responses`, streamed | `stream_timeout`, admission | `response.incomplete`, as above | `response.failed`, `request_timeout` |
| `/v1/responses` WebSocket | `stream_timeout`, admission (never the socket's queue) | `response.incomplete`, as above | `response.failed`, `request_timeout` |
| `/v1/decide`, each question | `request_timeout`, its submission | `not_completed` | `not_completed` |
| images, whole | `request_timeout`, the job's turn (never its queue) | `504 request_timeout` | `504 request_timeout` |
| images, streamed | `stream_timeout`, the job's turn | the images completed, then `error` with `request_timeout` | the same |
| media acquisition (chat) | `request_timeout`, the acquisition's start; `0`: each fetch's own timeout only | `504 request_timeout` | `504 request_timeout` |

On every route a passed deadline **cancels the engine request**, which frees
its lane (#327); the `ignis.request.cancelled` line says `cause: timeout`
whatever answered it, and an image job's `ignis.image.job` line says `outcome:
timed_out`, counted on `ignis_image_jobs_total` (no new family; the Playground
Monitor reads the label). Before any ending, the client gets **what the loop
guard held back** (ADR 0052) and the tokens already waiting on the request's
stream; nothing waits on the engine itself, which may be what is wedged. A tool
call cut mid-way is **dropped whole**, as at any other end (#121): no
half-written call reaches a client. A stop sequence the owed tokens complete
ends the answer `stop`, as it would have before the deadline.

**Why `length` and `incomplete` / `max_output_tokens`.** They are the only
values the two APIs define for "cut by a limit": chat's `finish_reason` is
`stop | length | tool_calls | content_filter | function_call`, and Responses'
`incomplete_details.reason` is `max_output_tokens | content_filter`. vLLM
already reports its own model-length limit as `length`. A client handles them
today; a new value (`timeout`) is one a strict client reads as an unknown finish
and retries, the failure #296 removed for engine errors. The deadline stays
distinguishable where it matters: the header on a whole answer, the request
log on every one.

**Why the chat stream's error ending is an error chunk.** A stream has
answered `200` before its deadline passes. The `504`'s body as the last
`data:` line, then `[DONE]`, with no finish chunk, is how a stream already ends
for an engine error (#296): a client that reads one reads the other.

**Why a decision and an image are always errors.** A decision's answer is one
readout of one position; there is no partial readout to return. An image cut
between two denoising steps is noise, not a smaller image (owner: "meglio
errore"); a streamed job's images already completed have gone out, and the
error event follows them.

## Considered options

- **One deadline for every route.** Rejected (owner): a stream and an answer
  awaited whole are read differently; one field would force the same bound on
  both.
- **A new `finish_reason: "timeout"` and `incomplete_details.reason:
  "timeout"`.** Rejected: neither API defines it, and the clients ignis serves
  (agents, the Playground, OpenAI SDKs) would meet an unknown value.
- **Keep `error` the default.** Rejected (owner): a long answer thrown away at
  its deadline is regenerated by the retry; what was generated is worth more
  to the client than a 504.
- **A partial image.** Rejected (owner): an unfinished image is no image.
- **Count from the request's arrival.** Rejected: a request waiting for a lane
  or in a socket's queue has used none of the engine; #327 already counted the
  socket's from admission, and an image job's from its turn is the same rule.

## Consequences

- **Upgrade notes.** The default is 2400 s, not 120: a request that used to
  end at two minutes runs to its end or to forty. Past its deadline a request
  is answered `200` with what it generated by default, not `504`; a client that
  relied on the 504 sets `--server-timeout-reply error`. A chat stream now has
  a deadline (`--server-stream-timeout`), and so does an image job. `0` is
  accepted and means none (it was refused).
  A chat completion the engine drops (its stream closes without a finish) is
  now `500 engine_error`; before, it was a `504 request_timeout`, logged as a
  timeout.
- A `length` from a deadline and one from `max_tokens` read the same in the
  body. A whole answer carries `ignis-stopped-by: request_timeout`, spelled as
  the server's other custom response header is (`ignis-model-status`, on
  `/v1/config`), with no `x-` prefix, and exposed to a browser on another
  origin (`Access-Control-Expose-Headers`, with `ignis-model-status` and
  `Retry-After`, which no browser there could read before either). A stream
  sent its headers before the deadline, so only the request log tells them
  apart there.
- A Responses WebSocket keeps an `incomplete` response for `previous_response_id`,
  as it keeps one a length limit ended: the next turn continues from it.
- A wedged engine is still bounded: a request it holds ends at its deadline,
  answered with what reached the client, and the cancel frees the lane once the
  model thread runs again.
- The decision's and the enhancer's collectors take `0` as none too; a media
  acquisition with no deadline is bounded by each fetch's timeout alone.
