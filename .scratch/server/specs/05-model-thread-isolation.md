# 05 — isolate the inference driver on its own thread (never blocked, never blocking)

GitHub: #69

## Problem Statement

Ignis's HTTP surface and its GPU-bound inference loop currently share one lock
and one async task, and that coupling defeats both streaming and
concurrency — the two things the north-star (ADR 0005) actually asks for.

Measured directly (curl `--trace-time`, not shell timing, so the evidence is
at the TCP layer): a streaming chat completion's SSE response delivers **zero
bytes** for the entire generation, then **all** chunks — every token delta,
the `finish_reason` chunk, the usage chunk, `[DONE]` — arrive in a single TCP
read, right at the end. Headers return instantly; the body does not stream at
all, despite `Content-Type: text/event-stream` and per-token `Event`s being
constructed correctly in code. A second, independent test made the scope of
the problem clear: while one request was decoding, a concurrent `GET
/v1/models` — a trivial, GPU-free handler — took **1726 ms** to answer, on a
request that normally answers in single-digit milliseconds. The entire server
stalls for the duration of any in-flight generation, not just that
generation's own response.

The root cause is structural, in `crates/server`'s `Engine`. Its driver loop
(`Engine::run`) calls `Scheduler::advance()` — a synchronous call that blocks
on real GPU work (~85 ms/token observed against the real model) — inside a
loop whose only yield point (`tokio::time::sleep(tick).await`) is reached
**only when the engine is fully idle**. While any request is in flight, the
loop runs start-to-finish with zero `.await` points: it locks `EngineInner`'s
`std::sync::Mutex` for the GPU call itself, unlocks, relocks per routed event
to hand tokens to their request's stream, and relocks again to record
telemetry — a tight, back-to-back cycle with no meaningful gap. Every other
consumer of that same `Engine` — `model_id()`, `submit()`, and telemetry sink
writes (themselves synchronous I/O, executed **while the lock from event
routing is still held**) — contends for that identical lock and, empirically,
loses the race for as long as the generation runs.

This is not a cosmetic issue. It is the opposite of the project's stated
priority. ADR 0005 makes performance — "maximum throughput **and** agent
parallelism that saturates the GPU in prefill *and* decode" — the #1
objective, and the dogfood target is explicitly "a 1 main agent + N subagents"
concurrent load (README). Every piece of upcoming work makes the coupling
worse, not better: N=8 concurrent decode lanes (ADR 0004) and batched decode
rounds (roadmap phase 3, G3) mean *more* simultaneous GPU-bound time per
driver step, which under the current design means longer and more frequent
full-server stalls for anything that is not GPU work — including, absurdly,
the very SSE bytes of the request that is generating.

## Solution

Give the GPU-bound inference loop exclusive, uncontended ownership of its own
execution resource, and make every other concern in the server — request
submission, response streaming, telemetry, and any future metrics or
introspection surface — talk to it only through channels and lock-free reads,
never through a lock shared with the compute path.

Concretely: a single, dedicated OS thread ("the model thread") owns the
`Scheduler` — and, through it, the CUDA leaf — for the entire life of the
server. Nothing else ever touches it directly. The async/HTTP world (SSE
response bodies, `GET /v1/models`, request submission, telemetry sink I/O)
runs entirely on the tokio runtime and never blocks on, or is blocked by, GPU
work. A request's own token stream stays coupled to its own generation — that
coupling is inherent, you cannot stream tokens faster than they are produced —
but nothing *unrelated* to a given request's own generation ever waits on it
again, and the model thread's own progress never waits on anything the async
side is doing.

## User Stories

1. As a streaming API client, I want each token chunk delivered to my
   connection as soon as it is generated, so that the API behaves like real
   streaming rather than a delayed dump at the end.
2. As a streaming API client, I want a long generation to feel responsive from
   the first token, so that a coding-agent UI built on this API can render
   output incrementally instead of appearing frozen.
3. As an API client, I want `GET /v1/models` to answer immediately regardless
   of how many requests are currently generating, so a health check or model
   discovery call never times out under load.
4. As an API client, I want to submit a new chat completion while other
   requests are actively decoding, so concurrent agents (the "1 main + N
   subagents" north-star load) are not serialized by an implementation detail
   unrelated to GPU capacity.
5. As an API client, I want my request's admission outcome (accepted, queued,
   or rejected) to arrive quickly even while the GPU is fully busy decoding
   other requests, so I can react — retry, back off — without an artificial
   multi-second delay that has nothing to do with actual queueing.
6. As an API client, I want two concurrent streaming requests to each receive
   their own tokens incrementally, not have both withhold all output until
   both finish, so that concurrency does not degrade into the worst case of
   the slowest request.
7. As an operator, I want telemetry JSONL lines to keep flowing at their
   designed cadence (one interval line per driver tick) even if the
   configured sink is momentarily slow — a loaded disk, a full pipe — so a
   telemetry hiccup never costs decode throughput.
8. As an operator, I want the interval line's live counters
   (waiting/running/kv_used_pct/kv_evictions) readable at any time with zero
   chance of contending the inference path, so a metrics scrape is safe to run
   continuously in production without a throughput cost.
9. As an operator, I want the telemetry wire format (design §5's JSONL shapes)
   unchanged by this refactor, so existing log tooling and `ignis-bench` keep
   working without modification.
10. As a maintainer, I want the `Scheduler` owned by exactly one thread for
    the server's whole lifetime, so CUDA's thread-affinity expectations are
    satisfied by construction, not by convention that the next contributor has
    to remember.
11. As a maintainer, I want no code path in the async/HTTP layer able to
    acquire a lock that the model thread also holds during compute, so a
    future contributor cannot silently reintroduce this bug by adding one more
    `.lock()` call inside a handler.
12. As a maintainer, I want the model thread's loop to service pending
    submissions and queries between decode steps at bounded latency (at most
    one step's worth, not the whole generation), so "the model thread is
    exclusive" never degrades into "the model thread ignores everything else
    until it goes idle."
13. As a maintainer, I want the existing per-request token/event delivery
    mechanism (the routed `SchedEvent` stream feeding `ChunkStream`) preserved
    exactly as observed at the API boundary, so this refactor does not ripple
    into `crates/core` or change the OpenAI wire contract.
14. As a maintainer, I want the `Scheduler` trait itself
    (`submit`/`advance`/`is_idle`/`model_id`) untouched, so the seam for this
    fix stays entirely inside `crates/server`'s `Engine` layer.
15. As a maintainer, I want the OpenAI HTTP handlers
    (`crates/server/src/api.rs`) to need no structural changes beyond
    `.await`ing what is newly async, so the blast radius of this fix is
    the `Engine`/driver layer, not the request-handling code.
16. As a maintainer, I want `model_id()` to require zero cross-thread
    coordination, since it is immutable for the server's lifetime, so the one
    piece of truly static state does not pay a channel round-trip for nothing.
17. As a maintainer, I want telemetry sink writes (stdout/file I/O) to happen
    entirely off the model thread, so a slow sink cannot, even in principle,
    add latency to a decode step.
18. As a maintainer, I want a regression test proving a concurrent, unrelated
    request completes quickly while a slow generation is in flight, so this
    class of bug has an automated check, not just a manual `curl` session.
19. As a maintainer, I want a regression test proving true incremental
    delivery — a per-request event observable before the request's generation
    fully completes — so the original streaming complaint is pinned, not just
    the concurrency symptom.
20. As a benchmark author, I want this change to not regress the recorded
    canary/performance numbers (ADR 0007), so the isolation itself costs
    nothing in aggregate decode throughput.
21. As a developer running the CPU gate, I want the new concurrency behavior
    covered by `MockCompute`-backed tests with a deterministic, injectable
    per-step delay, so `cargo test` stays fast, free of real sleeps, and green
    on a busy machine (ADR 0006: no wall-clock dependence in tests).
22. As a maintainer, I want the model thread to shut down cleanly when the
    server exits — no hung thread, no in-flight request silently dropped — so
    the server can be stopped and restarted without leaking resources, in
    keeping with the README's "model lifecycle decoupled from server
    lifecycle" hot-reload direction.

## Implementation Decisions

### The model thread

A single, dedicated `std::thread` — not a tokio task, not a `spawn_blocking`
pool thread — is spawned once, at server startup, and owns the `Scheduler`
for the process's entire life. Choosing a fixed dedicated thread over tokio's
blocking pool is deliberate: `spawn_blocking` draws from a pool of
interchangeable threads, and CUDA context/device association is thread-affine
in this codebase's usage — a single, permanent thread satisfies that by
construction rather than requiring every call site to re-establish the
device context.

The `Scheduler` and the per-request route table (today's `EngineInner.streams`
map) both become plain, unshared, thread-owned state on the model thread — no
`Arc<Mutex<..>>` around either. Nothing outside this thread ever reads or
writes them.

### The model thread's loop

Replaces today's `Engine::run`. Each iteration:

1. Drain every currently-queued command (non-blocking) — submissions and any
   other admin query — and handle each one against the thread-owned
   `Scheduler`/route table.
2. If the scheduler has anything in flight, perform exactly one
   `Scheduler::advance()` call (the GPU-blocking step) and route its emitted
   events: each routed `SchedEvent` still goes straight to its request's
   existing per-request channel (unchanged), and a copy is also pushed onto
   the telemetry facts channel (below).
3. If the scheduler is idle, block on the command channel (with a bounded
   wait, replacing today's fixed 1 ms poll) rather than busy-spinning, so an
   idle server still costs ~no CPU.

This bounds command latency to "at most one decode step" while busy (tens of
milliseconds, matching real per-token latency) instead of "the whole
generation" — the concrete fix for the `/v1/models`-during-generation
observation.

### The command channel (submission and queries)

An unbounded channel from the async side to the model thread, matching the
existing per-request channel's pattern. Each command carries a one-shot reply
channel for its specific answer. `submit()` becomes this: build the
per-request route pair exactly as today, send a submit command carrying it,
await the one-shot reply carrying `(RequestId, SubmitError)`. This is the one
call site that changes shape at the `Engine` API boundary — `submit` becomes
`async fn` — and it is the only change `crates/server/src/api.rs` needs to
make (adding `.await`); no other request-handling code changes.

`model_id()` does **not** go through this channel. The model id is immutable
for the server's lifetime, so it is captured once at `Engine` construction
(before the model thread starts) and stored as plain, lock-free, cloneable
state on the `Engine` handle. Reading it never touches the model thread.

`is_idle()` stays entirely internal to the model thread's own loop; it was
never part of the public surface `api.rs` calls and does not need to cross
the thread boundary at all.

### Telemetry: computed off the model thread, published wait-free

Today, `Telemetry`'s per-event bookkeeping (`note_submit` / `on_admitted` /
`on_token` / `on_evicted` / `on_done` / `emit_interval`) runs inline inside
the driver loop, under the same lock event-routing just took — meaning sink
I/O (a disk write in `FileSink`, a stdout write in `StdoutSink`) currently
executes *while a lock shared with the compute path is held*. This is exactly
the class of thing the model thread must never do.

The fix: the model thread only ever pushes lightweight *facts* onto an
unbounded channel — a routed `SchedEvent`, a submission notice, or a per-step
tick marker, mirroring the three call sites `Telemetry` has today — and does
no formatting, no sink I/O, and no counter computation itself. A new async
task, spawned alongside the model thread at startup, owns the existing
`Telemetry` value unchanged and drains this facts channel, calling the exact
same `note_submit` / `on_admitted` / `on_token` / `on_evicted` / `on_done` /
`emit_interval` methods it does today. The §5 JSONL shapes, the event-derived
interval-counter estimator, the injectable `TelemetrySink` /
`TelemetryClock` / `IntervalStatsProvider` seams — all unchanged in behavior;
only the location that drives them moves off the model thread. The facts
channel is unbounded and the model thread's send is fire-and-forget: telemetry
volume is inherently bounded by real generation throughput, never by sink
speed, so the model thread can never be made to wait on it, no matter how slow
the configured sink is.

After computing each interval line, the async telemetry consumer also
publishes the resulting `IntervalCounters` into an `ArcSwap<IntervalCounters>`
(new small dependency, `arc-swap`). Any reader — a future metrics/introspection
endpoint, or a test — gets the latest snapshot with a wait-free `load()`, with
no channel round-trip and no possibility of blocking on, or being blocked by,
either the model thread or the telemetry consumer. This is additive: it does
not replace the JSONL emission path, it gives the same computed counters a
second, always-available reading surface.

### What does not change

- The `Scheduler` trait (`crates/core`) — untouched. The seam for this fix is
  entirely `crates/server`'s `Engine`/driver layer, the highest point at which
  the problem can be fixed without touching core.
- The per-request `SchedEvent` → `ChunkStream` delivery mechanism — untouched;
  it was never the bottleneck (it does not touch the contended lock today),
  and isolating the model thread is what finally lets it behave as designed.
- The OpenAI wire contract and `crates/server/src/api.rs`'s handlers — no
  changes beyond `.await`ing the now-async `submit()`.
- The §5 telemetry JSONL format and its existing sink/clock/stats-provider
  seams.

## Testing Decisions

A good test here asserts observable behavior — response timing and content —
not internal scheduling details (thread count, lock acquisition order). Two
properties matter and neither existed as an automated check before this spec:
that an unrelated request is never held up by someone else's generation, and
that a single request's own tokens are delivered incrementally rather than in
one lump at the end.

Per ADR 0006, tests must not depend on wall-clock timing to prove
concurrency. A test-only `Compute` decorator wraps `MockCompute` (the existing
`Compute` trait is already the seam production and tests both go through) and
blocks its `decode_step` on an explicit synchronization primitive (a
channel/barrier under test control) rather than a real delay — this lets a
test deterministically hold one request "mid-decode" for as long as it needs
to, and then unblock it, without a single `sleep()` anywhere in the suite.

### Primary seam — the isolated model thread

A new test module exercises the model-thread wrapper directly (not through
HTTP): submit request A, use the synchronization primitive to hold its decode
step, and while held, submit request B / query the model id / send another
command — all of which must complete promptly, proving the command channel is
serviced between A's steps rather than after A finishes. This is the direct,
deterministic proof for stories 4, 5, 10-12.

### The HTTP seam — `crates/server/tests/openai_http.rs`

Prior art and still the seam of record for wire-level behavior. A new case
drives two concurrent streaming requests over the mock engine (one held
mid-decode via the same synchronization primitive) and asserts that the
*other* request's chunks arrive and are readable before the held request
completes — this is the automated version of the `/v1/models`-during-generation
finding, expressed as "two concurrent streams don't serialize," and covers
stories 1, 2, 6, 19.

### Telemetry

Existing `crates/server/tests/telemetry.rs` (from spec 02) is the prior art
for the JSONL shapes and sink/clock seams; it is re-run unchanged against the
relocated consumer to confirm the wire format is unaffected. A new test
covers the `ArcSwap` snapshot: after driving a few events through the model
thread, the snapshot readable via `load()` reflects the same counters the
JSONL interval line reported, read without going through the facts channel at
all.

### GPU end-to-end

`crates/server/tests/openai_http_gpu.rs` (existing, GPU profile, ADR 0006)
gains one case re-running this spec's curl-based finding as an assertion:
a streaming request's SSE frames are observed arriving before the request's
own generation completes (a time-bounded read of the first chunk, well under
the full expected generation time) against the real model. This anchors the
CPU-mock coverage to the real GPU path, matching the profile's "fail, never
skip" rule.

### The gate

`cargo test` stays green, CPU-only, and deterministic workspace-wide; the GPU
case runs under the explicit profile only.

## Out of Scope

- **Batched decode rounds and per-width CUDA graphs (roadmap G3).** This spec
  fixes the isolation problem that batching would otherwise make worse; it
  does not implement batching itself.
- **`Scheduler::stats()` on the core trait.** The interval line's
  `prefilling` / `kv_used_pct` fields stay estimator-derived (0 for
  `prefilling`/`kv_used_pct` as today) exactly as documented in
  `telemetry.rs`'s existing module doc — that blocker is tracked separately
  and is not this spec's job to close.
- **A new `/metrics` or `/v1/status` HTTP endpoint.** The `ArcSwap` snapshot
  makes wait-free live counters *available*; wiring an endpoint that serves
  them is a separate, later decision.
- **Hot-reloading the model** (swapping the artifact without a server
  restart). The model thread's clean-shutdown property is a prerequisite for
  that direction, not an implementation of it.
- **Changing admission, KV accounting, or any other `crates/core` scheduling
  behavior.** This is purely a `crates/server`-side ownership/threading
  refactor.
- **Reworking the per-request `SchedEvent` channel or `ChunkStream`.** Neither
  is the bottleneck; both are reused unchanged.

## Further Notes

The two symptoms this spec fixes were established with completely different
tools, and both point at the same root cause: `curl --trace-time` proved the
SSE body is not incrementally written to the socket (a networking-layer
observation, immune to any of this process's own buffering), and a concurrent
`GET /v1/models` ping proved the *entire* server — not just the SSE body path
— stalls for the duration of an in-flight generation. The second observation
is the more direct one: `model_id()`'s lock acquisition, contended against the
driver loop's tight re-lock cycle during a real GPU call, is sufficient on its
own to explain a multi-second stall on a trivial endpoint, with no need to
invoke any exotic tokio-scheduling theory. Whoever implements this should keep
both tests (the direct model-thread test and the HTTP-level concurrent-stream
test) — they pin the mechanism and the symptom separately, and either one
regressing without the other narrows down which layer broke.

The telemetry module's own doc comment currently claims sink writes "never
invert a lock with the engine's scheduler mutex." That claim is true about
lock *ordering* but not about lock *duration*: today, sink I/O executes while
still holding the per-event lock taken for routing, which is precisely the
kind of coupling this spec removes. Worth a follow-up doc correction once this
lands, so the module doc describes the new (correct) guarantee rather than the
old, weaker one.
