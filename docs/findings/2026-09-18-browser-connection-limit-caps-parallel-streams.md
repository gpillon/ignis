# The browser's connection limit caps the Playground's parallel streams

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: Playground transport / parallel agents, Monitor scraping, anything the page requests while streams run
- Related: https://github.com/gpillon/ignis/issues/220, [ADR 0017](../adr/0017-prometheus-metrics.md), [ADR 0026](../adr/0026-playground-embedded-when-built.md), [ADR 0028](../adr/0028-expose-modes-always-keyed.md)
- Superseded by: none

## Question

The Playground's Monitor stops updating as soon as six agents stream in
parallel, and only recovers much later. Is the server failing to answer
`/ui/metrics` under load, or is something between the browser and the server?

## Evidence

Repro: Playground on the Vite dev server (`:5173`, proxying to
`ignis-server --ui --metrics` on `:8000`), the `agent` tool on, prompt "use 6
agents to write ~2000 tokens in parallel". Raw material: [`.scratch/220/`](../../.scratch/220/).

A curl poller hit the endpoints once a second from outside the browser while
the six agents decoded (`ignis.request.admitted` for request_id 1..6 at
00:32:49.6–00:32:50.5 UTC in [`server-during-repro.log`](../../.scratch/220/server-during-repro.log)):

| clock | `/metrics` (:9464) | `/ui/metrics` (:8000) | `/v1/models` (:8000) |
|---|---|---|---|
| 02:32:49 | 200 in 1.0 ms | 200 in 1.1 ms | 200 in 0.9 ms |
| 02:32:50 | 200 in 1.0 ms | 200 in 1.1 ms | 200 in 1.0 ms |

The server answered every poll in about a millisecond for the whole repro, and
the Vite proxy passed them through in about two. Meanwhile the owner saw a long
**Stalled** bar on the browser's own `/ui/metrics` request in DevTools —
Chrome's word for "queued, waiting for a connection".

`netstat` counted the page's sockets next to each poll. Loopback shows both
ends of a connection, so the count halves; the second repro ran from 02:36:03
([`poll-3-endpoints-and-sockets.csv`](../../.scratch/220/poll-3-endpoints-and-sockets.csv), every sample, unedited):

| clock | established to `:5173` | connections |
|---|---|---|
| 02:36:03 … 02:36:12 | 12 | 6 |
| 02:36:13 … 02:36:16 | 10 | 5 |
| 02:36:17 … 02:36:37 | 8 | 4 |
| 02:36:38 | 6 | 3 |
| 02:36:39 | 4 | 2 |
| 02:36:41 … 02:36:55 | 6 | 3 |

Six streams, six connections: exactly the six a browser opens per origin over
HTTP/1.1. The scrape had no socket left and waited for an agent to finish. The
bounce back to three at 02:36:41 is the page reconnecting after the agents
ended, not an agent restarting.

## Finding

Observed: while six agents stream, the page holds six connections to its
origin, the server answers every request in about a millisecond, and the
browser reports the page's own scrape as Stalled.

Inferred, from the three together: over HTTP/1.1 the Playground's parallelism
is bounded by the browser, not by the engine. A stream holds one of the six
connections per origin for its whole life, so with six agents the page has none
left and **every** same-origin request waits — the Monitor scrape first, but
equally a new chat or an uncached asset. `MAX_PARALLEL_AGENTS = 8` is
unreachable from a browser on HTTP/1.1: agents seven and eight cannot start
until earlier ones end, so two of the engine's eight lanes stay unusable
however many agents the model asks for.

The limit is per origin and specific to HTTP/1.1. A multiplexed connection
(h2, h3) carries all the streams at once, which is what a TLS reverse proxy or
`--expose` provides; plain localhost does not, and serving it over TLS is not
wanted.

## Implications

The budget belongs around every stream, not around one caller: agents, an
agent's own tool round and the JavaScript safety check all open one, and a
reservation each counts separately reserves nothing. `web/src/api/connections.ts`
holds it, reading the protocol from `PerformanceResourceTiming.nextHopProtocol`
for the page's own origin — five streams when it cannot be confirmed to
multiplex, unbounded on h2 and h3.

Any future Playground feature that holds a connection open — a second live
panel, a server-sent log tail, per-agent progress channels — spends from the
same six. Measure before assuming the server is slow: poll the endpoint with
curl from outside the browser and count sockets with `netstat`. If both are
healthy while the page is not, the queue is in the browser.

## Limits and unknowns

- One browser (Chrome on Windows 11), one session, through the Vite dev proxy.
  Six per origin is the documented HTTP/1.1 behaviour of every major browser,
  but only Chrome was measured, and only the dev server: the embedded build,
  served by `ignis-server --ui` on one origin, was not.
- The live check was the owner's own run of the repro, read from the page, not
  a second round of socket counts.
- The socket counts are the process-wide count for that port, not attributed
  per connection. Which agent owned which socket was not established, and the
  Vite HMR WebSocket was open throughout — the counts suggest Chrome keeps it
  out of the HTTP pool, but the evidence does not establish that.
- Whether an h2 page really lifts the limit was not observed; it follows from
  the protocol, not from a measurement taken here.

## Follow-ups

- https://github.com/gpillon/ignis/issues/220 — closed: the owner ran the
  six-agent repro in the browser on 2026-09-18 and confirmed the page keeps
  answering.
- https://github.com/gpillon/ignis/issues/221 — how to reach eight streams on
  plain localhost, where the browser allows six and TLS is not wanted.
