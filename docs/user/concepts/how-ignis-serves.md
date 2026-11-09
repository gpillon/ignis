---
title: "How ignis serves a request"
description: "The path of a request through the server: admission, prefill, decode lanes, speculation and the stream back, and how the Rust server and the CUDA kernel leaf divide the work."
sidebar:
  order: 1
---

ignis is one process that holds one model at a time and serves it to many
requests at once. This page follows a single request from the HTTP socket to
its last streamed token, and says why each stage is shaped the way it is. The
flags that tune each stage are on the [configuration pages](../configuration/index.md);
the other pages in this section take one stage each.

## The shape of the process

Two layers divide the work
([ADR 0001](../../adr/0001-cpp-kernel-leaf-c-abi.md),
[ADR 0009](../../adr/0009-step-level-device-resident-abi.md)):

- **Rust owns everything above the step**: the HTTP API, the chat template,
  admission, the scheduler, KV accounting, streaming, configuration and the
  Playground.
- **The kernel leaf owns the forward pass.** It is a C++/CUDA static library
  linked into the server behind a flat C ABI. The ABI is named after its
  granularity, the *step*: one call per model load, per sequence lifecycle
  event, per prefill span or per decode round, never one per operator.
  Activations stay on the device; nothing the leaf computes crosses the ABI as
  a host pointer.

The split matters to a user in one way: scheduling decisions (which request
gets a lane, what moves to host RAM, when a loop is taken back) are made in
Rust, between steps, and can change without touching a kernel. What a step
costs is the leaf's, and is mostly the time the card needs to stream the
model's weights.

One model is loaded at a time. `GET /v1/models` reports its `status`:
`serving`, `switching`, `loading`, `unloaded` or `failed`
([switching and idle unload](switching-and-idle-unload.md)).

## The path of a request

```text
client
  |  POST /v1/chat/completions   (or /v1/responses, /v1/decide ...)
  v
HTTP server (Rust)
  |  API key, body limit, deadline clock
  |  chat template -> prompt tokens   (images: acquired and prepared here)
  v
admission
  |  lane tag: interactive | agent
  |  KV bound = prompt + generation cap
  |  retained state? claim the longest matching prefix
  v
prefill lane (one prompt at a time)
  |  chunks of --model-prefill-chunk tokens (the solo width while no lane decodes)
  |  a decode round between chunks (decode share)
  v
decode lanes (8 on the 27B; 1..8 on Flash-Next)
  |  one batch-wide round for every decoding lane
  |  with speculation: a draft + verify round
  v
stream back
  |  tokens -> reasoning / content / tool calls -> SSE or WebSocket
  |  loop guard watches the reasoning on its way
  v
client           the finished prompt's state is retained for the next turn
```

### 1. The server turns the request into tokens

Before the scheduler sees it, the server checks the API key and the body
limits, renders the messages through the model's chat template (tools,
thinking switches and the system message policies included), and tokenizes the
result. A request with images has them acquired, decoded and prepared first; a
refused image refuses the request before it reaches the scheduler. From there a
multimodal request is an ordinary request, chunked and interleaved with decode
lanes, speculated by DFlash2, sharing prefixes by content with its images in
the key, evicted to and restored from host RAM, whose prefill also encodes each
image once and whose rounds rotate at its rope delta. The same
rendering is what `POST /v1/tokenize` counts, without serving it
([counting tokens](../api/count.md)).

A deadline clock starts at admission. A request still running when it passes
is cut and answered as [the deadlines page](../api/deadlines.md) describes.

### 2. Admission decides whether and where it runs

A request states its own class, `interactive` or `agent`, through the `class`
extension field or an `@<lane>` suffix on the model name. The class, together
with arrival order, ranks the request: it decides who is protected, who is
backfilled first and who loses residency when memory is short.

Admission also checks the KV pool. A request's *bound* is its prompt plus its
generation cap (`max_tokens`, or `--model-default-max-tokens`), never more than
`--model-max-context`. A request alone always fits the pool; when it does not
fit beside the others, a lower-ranked sequence is moved to host RAM or disk and
resumes later, or the request waits ([the KV cache](kv-cache-and-prompt-reuse.md)).

When every decode lane is in use, an HTTP request is answered
`503 engine_full` with a `Retry-After` estimate and the client's retry is the
admission. A request on the Responses API's WebSocket is queued instead
([the API](../api/index.md), [the Responses API](../api/responses.md)).

Last, admission looks for retained state: the longest earlier prompt or shared
prefix whose tokens are a prefix of this one. A hit means the request skips
that part of prefill.

### 3. Prefill reads the prompt

Prefill runs the prompt through the model once to fill its KV pages and
recurrent state and to draw the first token. Two rules shape it:

- **One prompt prefills at a time.** There is a single prefill lane; other
  prompts wait their turn behind it.
- **A prompt is prefilled in chunks, and decode takes turns with it.** The
  scheduler hands the leaf one chunk (`--model-prefill-chunk`, 1,024 tokens by
  default, 2,048 for the 27B under the built-in profiles) and runs a decode
  round for the lanes already generating before the next chunk. While no lane
  is generating, a load that planned a wider
  [solo width](../configuration/context-and-speculation.md#the-solo-width)
  takes it instead, decided again at every chunk boundary; under the built-in
  profiles the two are the same width. Nothing runs
  concurrently on the card: prefill and decode alternate on one stream
  ([ADR 0018](../../adr/0018-chunk-level-prefill-decode-interleaving.md)).

The reason for chunking is that a 32K-token prompt prefilled in one call stopped
every decoding lane for the whole span, about 3.8 seconds at the time of the
decision. With chunks the lanes pause for one chunk instead. How long the lanes
keep the card while a prompt prefills is `--model-decode-share` (25 percent by
default): after a chunk that took `t`, the prompt's next chunk waits `t * s /
(1 - s)` while the lanes decode. The cost lands on the prefilling request's own
first token (about 1.33 times slower at 25), and the gain on the lanes, which on
Flash-Next keep 15.6-18.3 tokens per second during a prefill instead of about 1
([configuration](../configuration/context-and-speculation.md)).

A prompt that arrives alone, or with no lane decoding, is unaffected: nothing
waits.

The first token leaves as soon as the prompt's last chunk is done: the server
streams it before any decode round runs, even while every decode lane is taken.
A request that ends on it (`max_tokens: 1`, an end-of-sequence token, or a stop
sequence the token completes) never takes a lane at all, and its answer is cut
at the stop as always.

### 4. Decode generates, one round for every lane

Once prefilled, a request holds a decode lane. Its first round starts from the
first token, already sent, and does not send it again. A **decode round** is one
traversal of the model for all decoding lanes together: the lanes are rows of
one batch, so eight lanes stream the weights once instead of eight times.
The round is replayed from a CUDA graph captured for each exact batch width
(1 to 8), because padding a narrow round to a wider graph would move state the
round does not need.

Sampling happens on the device, per lane: temperature, seed and penalties
belong to the request, so what a request generates depends on its own seed
alone, not on which lanes shared its round.

With [speculative decoding](speculative-decoding.md) on, the round becomes a
draft-and-verify round: each lane carries a few proposed tokens, the model
checks them all in the same traversal, and the lane keeps the longest run the
model agrees with.

### 5. The stream goes back, split into channels

The leaf returns token ids; the server turns them into text and splits the text
into channels:

- **Reasoning** (between `<think>` and `</think>`) goes to `reasoning_content`.
- **Content** is the answer.
- **Tool calls** are parsed out of the content into the API's tool-call shape.

Chat completions stream as server-sent events; the Responses API streams the
same way over HTTP or over a WebSocket. The first token arrives
alone, at the end of the prefill; then each round's tokens follow. A speculative round can
commit several tokens at once, and the server still streams them one at a time.

Between the engine and the client, a small task watches the reasoning for a
loop and can take it back ([reasoning and the loop guard](reasoning.md)). At
the end of the stream `usage` reports the tokens, including
`prompt_tokens_details.cached_tokens`, the part of the prompt the request
resumed instead of prefilling.

### 6. The finished prompt is kept

When a request ends, the state at the point where its prompt handed over to the
model is retained, and the system block of its prompt may be kept as a shared
prefix. The next turn of the same conversation, or the next subagent of a
burst, finds it by content and starts from there
([the KV cache and prompt reuse](kv-cache-and-prompt-reuse.md)).

## What runs beside the request path

- **The Playground** at `/ui/` is built into the binary and talks to the same
  API; it is not a separate service. Its Monitor view reads the same metrics
  Prometheus does ([the Playground](../playground/index.md)).
- **Metrics** are exposed on a separate loopback listener (`--server-metrics-bind`),
  on by default: request lifecycle, KV pool pages, retained state, the VRAM
  plan, loop-guard counters ([telemetry](../guides/telemetry.md)).
- **The log** is one structured stream. A request's life is the
  `ignis.request.admitted`, `ttft` and `done` events, and the plan the load
  made is `ignis.runtime.vram_plan` ([the VRAM plan](vram-plan.md)).
- **The configuration API** (`GET` and `PATCH /v1/config`) reads and changes
  the running server, reloading the model only for a field that needs it
  ([configuration](../configuration/index.md)).

## The same path for other requests

- `POST /v1/decide` answers a classification by prefilling the evidence once
  and reading the answer from the logits at one position, with no token
  decoded. It takes the single prefill lane, never a decode lane, and so it
  defaults to the `agent` class ([decisions](../api/decide.md)).
- The image routes go to the image model, which is a different model with its
  own load; reaching it is a switch ([images](../api/images.md),
  [models and artifacts](models-and-artifacts.md)).
