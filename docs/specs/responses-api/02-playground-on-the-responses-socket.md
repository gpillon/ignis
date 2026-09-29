# 02 - the Playground talks to ignis over the Responses WebSocket

GitHub: #283

The Playground sends every chat turn, tool round and `agent()` subagent as its
own `POST /v1/chat/completions` stream. Over HTTP/1.1, which is what a browser
speaks to `http://localhost`, Chrome opens at most six connections per origin,
so #220 made the page ration them (`connections.ts`: five streams, the sixth
connection kept for the rest of the page), and parallel agents past five wait
their turn in the browser, not in the engine. Spec 01 makes `/v1/responses` a
standard Responses API with a WebSocket mode that multiplexes named streams on
one connection. This spec moves the Playground's conversation traffic onto
that socket: one connection, one stream per piece of concurrent work, no
browser cap.

Depends on spec 01 (`docs/specs/responses-api/01-responses-streaming-and-websocket-mode.md`).
Finding: `docs/findings/2026-09-18-browser-connection-limit-caps-parallel-streams.md` (#220).
Research: `docs/findings/2026-09-29-openai-websocket-inference-interfaces.md`.

## Problem Statement

An owner who runs several sessions at once, or a turn whose `agent()` tool
fans out into subagents, sees the sixth and later streams stall with nothing
happening on the GPU: the browser, not ignis, is holding them. The engine has
room; the page cannot reach it. Every turn also re-sends the session's whole
history, images included, and the Playground has no way to warm a session's
prompt state before the owner presses send.

## Solution

The Playground opens one Responses WebSocket to the ignis it is served by and
sends each turn as a `response.create` on a named stream: one stream per
session, one per running subagent. Replies stream back as Responses events
and are shown exactly as today (thinking, content, tool calls, figures). A
turn that continues the previous response on the same open socket sends
`previous_response_id` and only the new items; anything else sends the full
history. The socket is the default transport; a setting switches the page
back to HTTP chat completions, and the page falls back to it by itself when
the socket cannot be opened. The Decide tab keeps `/v1/decide`.

## User Stories

1. As the owner, I want ten sessions streaming at once to all make progress, so that the Playground shows the engine's real concurrency.
2. As the owner, I want an `agent()` tool call that starts eight subagents to run them all at once, so that parallel agents are actually parallel.
3. As the owner, I want replies to look exactly as they do today (thinking, content, tool calls, errors), so that the transport change is invisible in the transcript.
4. As the owner, I want the per-request figures (TTFT, tokens/s, totals) to stay correct, so that I can keep comparing runs.
5. As the owner, I want the next turn of a session to send only what is new, so that long sessions with images do not re-upload everything.
6. As the owner, I want edit, regenerate and branching to keep working, so that the history the model sees is always the one on screen.
7. As the owner, I want a keyed server to accept the socket with the key I already entered, so that I do not configure anything twice.
8. As the owner, I want a wrong or missing key on the socket to show the key prompt, as a 401 does today, so that the flow is the same.
9. As the owner, I want the page to fall back to HTTP by itself when the socket cannot be opened (an older ignis, a proxy that refuses upgrades), so that the Playground keeps working.
10. As the owner, I want a switch in General settings to force HTTP, so that I can compare transports or work around a network.
11. As the owner, I want the stop button to stop one reply on the socket as it does over HTTP, without touching the other sessions and subagents sharing it, so that I can interrupt the model.
12. As the owner, I want a dropped socket to fail the turns it carried with a visible error and reconnect for the next turn, so that nothing hangs silently.
13. As the owner, I want a turn queued by a full engine to say it is queued, so that I can tell engine back-pressure from a stuck page.
14. As the owner, I want the lane tag (`interactive` / `agent`) and thinking controls sent on the socket as over HTTP, so that scheduling and thinking behave the same.
15. As the owner, I want the Decide tab untouched, so that decisions keep working as they do.

## Implementation Decisions

**One transport seam.** The page already sends every conversation request
through one function (`streamChat`: a request body in, events out through a
callback, a timeline back for the figures). A socket transport implements the
same contract: it takes the Playground's own request (messages, settings,
tools), sends it as a `response.create`, and maps Responses events onto the
events the turn loop already consumes (content delta, reasoning delta, tool
calls, usage, done, error). The turn loop, sessions, tools and transcript do
not learn which transport ran. The transport is chosen once per page from the
setting and the fallback state.

**One socket per page.** Opened lazily on the first conversation request to
`GET /v1/responses` on the page's own origin, kept open, reopened on the next
request after a drop. Authentication: the stored key travels as the
`openai-insecure-api-key.<key>` subprotocol, next to a non-credential
subprotocol the server echoes (spec 01); no key, no credential entry. A 401 on
the upgrade shows the key prompt, as `auth.ts` does for HTTP.

**Streams.** A session's turns run on a stream named after the session; each
`agent()` subagent runs on its own stream for its lifetime. Stream ids are
sanitised to spec 01's pattern. The connection's 32 named ids and 16 active
responses (spec 01) are respected: past 32 distinct ids the page opens a new
socket for further streams, and responses past 16 are left to the server's
per-connection wait, which keeps them in order.

**Continuation.** The page keeps each stream's latest response id and the
exact items that response was built from. A request sends
`previous_response_id` plus only the new items when the socket that produced
that response is still open and the request's history is exactly that
response's history plus its output plus new items; otherwise (edit,
regenerate, branch, reconnect, a `previous_response_not_found` answer) it
sends the full history with no `previous_response_id`, retrying once in full
after `previous_response_not_found`. The prompt the model reads is identical
either way.

**Request mapping.** Messages become input items (`message` with
`input_text` / `input_image` parts, assistant turns with `output_text`, tool
calls as `function_call`, tool results as `function_call_output`, thinking
round-tripped as `reasoning` items when the page keeps it). The system prompt
(the tools' prompt followed by the owner's, as today) becomes `instructions`.
Tools are declared as function tools in the Responses flat shape. Settings
map field for field: `max_output_tokens`, `temperature`, `top_p`,
`reasoning_effort` / `thinking_budget` / `enable_thinking` as the ignis
extensions, `class` from the lane tag.

**Stop, queue, errors.** Stop sends `response.cancel` with the reply's
response id (spec 01's one extension event, the Realtime API's shape), which
ends that reply, running or queued, and leaves every other stream on the
socket running. `response.queued` shows the turn as queued; `error` events surface through the existing error display, a 401 as
the key prompt.

**Fallback and setting.** A setting in General, "Transport: WebSocket / HTTP",
default WebSocket. If the upgrade fails (not a 401), the page switches to HTTP
chat completions for the rest of the page's life and says so once. Over HTTP
the #220 stream budget still applies; over the socket it does not.

**Figures.** The timeline is taken from the socket: sent when the
`response.create` frame is written, first token at the first
`output_text.delta` or `reasoning_text.delta`, end at the terminal event,
token counts from the terminal usage. Time spent in the server's admission
queue is shown as queue time, not as TTFT.

## Testing Decisions

Tests drive the transport through its public contract with a fake WebSocket
(and a fake clock), as `stream.test.ts` drives `streamChat` with an injected
`fetch`, and assert on the events and timeline the turn loop receives. They do
not inspect frames beyond what the server contract (spec 01) fixes.

- The socket transport: request mapping (a Playground request becomes the
  expected `response.create`), event mapping (a recorded Responses event
  sequence yields the same callback events as the equivalent chat SSE
  sequence), continuation choice (new items only vs full history, retry after
  `previous_response_not_found`), streams per session and per subagent, the
  32-id rollover, stop, drop and reconnect, 401 to key prompt, fallback to
  HTTP on a failed upgrade, figures and queue time. Prior art:
  `web/src/api/stream.test.ts`, `sse.test.ts`, `connections.test.ts`,
  `auth.test.ts`.
- The settings switch and the fallback notice: component tests as in
  `web/src/settings`.
- The owner watches the result live on `:5173` (no screenshot verification).

## Acceptance

1. With the socket transport, twelve concurrent streams (sessions and subagents mixed) all stream at once; none waits in the browser.
2. Transcript, thinking, tool calls, errors and figures are the same as over HTTP for the same conversation.
3. A follow-up turn on an open socket sends `previous_response_id` and only new items; edit, regenerate, branch and reconnect send the full history; `previous_response_not_found` is retried once in full.
4. The key is sent as the credential subprotocol; a 401 shows the key prompt.
5. A failed upgrade falls back to HTTP chat completions with a one-time notice; the General setting forces either transport, default WebSocket.
6. Stop ends the reply with `response.cancel` and the socket's other streams keep running; a dropped socket fails its turns visibly and the next turn reconnects.
7. A queued turn shows as queued, and queue time is not counted as TTFT.
8. The Decide tab is unchanged.
9. `npm test` in `web/` and `cargo test` pass.

## Out of Scope

- The Decide tab and `/v1/decide`: they stay on HTTP.
- `generate: false` prewarm from the Playground (a possible follow-up: warm a
  session's prompt while the owner types).
- Monitor panels for the new socket gauges.
- Any server change: spec 01 owns the endpoint.

## Further Notes

- HTTP/2 would also lift the six-connection cap, but browsers speak it only
  over TLS, so it does nothing for `http://localhost`; the socket is the
  standard way this page gets past it.
- The chat completions path stays in the page as the fallback, so it keeps
  its tests.
