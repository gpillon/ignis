# Request-lifecycle tracing spans (GitHub #81, ADR 0012)

The shape of the span tree ignis instruments per request, and where the
`trace_id`/`span_id` on every log record inside it come from. ADR 0012 is
the decision record (why `request.id` *is* `trace_id`, not a fresh id);
this document is the implementation map: which span opens where, in which
crate, and why it is (or, in two places, deliberately is not) a real
`tracing` parent/child of its neighbor.

## The shape

```
ignis.http.request                 (crates/server/src/api.rs, tower-http TraceLayer)
  │  request_id: Empty at creation, recorded once Engine::submit resolves
  │
  ├── ignis.admission               (crates/core/src/concrete.rs, try_admit)
  ├── ignis.prefill                 (concrete.rs, per prefill_step call)
  ├── ignis.decode.round  (× N)     (concrete.rs, per decode_step call)
  │     └── ignis.decode.mtp_verify  ← attach point for a future MTP/
  │                                    speculative-decoding verify span
  │                                    (not built yet — see "Future stages")
  ├── ignis.completion              (concrete.rs, mark_done)
  └── ignis.telemetry.emit          (server/src/telemetry.rs, emit_request —
                                      backs the `ignis.request.admitted` /
                                      `ttft` / `done` events, GitHub #79)
```

Every span above carries a `request_id` field (`u64`, `ignis_core::RequestId`
verbatim). `ignis_logging::trace_context` reads that field off whichever
span is in an event's active scope (walking from the innermost span
outward) and turns it into the record's `trace_id` — zero-extended into the
32-hex-character OTel shape (`format!("{id:032x}")`), never a separately
generated identifier (spec §19, ADR 0012). `span_id` is the innermost
active span's own id (`tracing`'s own span id, formatted as 16 hex
characters). An event with no active span, or an active span whose scope
never records `request_id` anywhere in it, gets neither field — genuinely
absent from the serialized record, not a placeholder.

## Why the tree is not one literal `tracing` parent/child graph

Two edges in the diagram above are **not** real `tracing` spans-of-spans,
by construction, and that is deliberate rather than an oversight:

1. **`ignis.http.request` → the four `concrete.rs` spans.** The HTTP
   handler runs on the async task `axum`/`tokio` schedule it on;
   `ignis-core`'s admission/prefill/decode-round/completion spans are
   opened on the engine's one dedicated model thread (GitHub #69's
   threading model — the `Scheduler` and its request table are never
   touched from any other thread). `tracing`'s span-stack propagation is
   thread-local: a span entered on one OS thread is not "current" on
   another. Passing the HTTP span's id across the command channel so core
   could declare it as an explicit `parent:` would work mechanically, but
   would mean threading a tracing handle through `ignis-core`'s public API
   for a crate that otherwise has zero opinions about how spans render —
   deliberately out of scope for this issue. The two sides are correlated
   by **sharing `request_id`** (and therefore `trace_id`), which is the
   property every acceptance test and the ADR actually care about — not by
   `tracing`'s own parent-pointer graph.
2. **`ignis.decode.round` → many requests in one batched round.** A single
   `Compute::decode_step` (or `prefill_step`) call may cover several
   *different* requests at once (batched decode across resident lanes,
   `docs/adr/0018-chunk-level-prefill-decode-interleaving.md`). That call
   itself has no single `request_id`, so it is never wrapped in a span.
   Instead, one `ignis.decode.round`/`ignis.prefill` span is opened **per
   request, per round/chunk**, inside the per-request outcome-processing
   loop that already exists right after the call returns — never inside
   the call itself, and never subdividing further. For a single request
   this reduces to exactly one span per round it participates in; for a
   batch of N concurrent requests, one round yields N round-spans (one per
   request), never one shared span with N trace ids and never spans nested
   inside the compute call.

## Granularity: round, not token

`ignis.decode.round` is opened once per request per `Compute::decode_step`
call — the same unit a decode CUDA graph is captured over (`CONTEXT.md`,
**Decode round**). A round can only ever advance a sequence by one token
today (the leaf has no host-visible sub-round hook — "device-resident, no
host activation pointer crosses it," ADR 0009), so for the current
codebase "one span per round" and "one span per token" coincide for any
single request. The distinction is intentionally forward-looking: DFlash2 /
MTP verify (not built yet) may resolve *more than one* token per round.
When it lands, its span (`ignis.decode.mtp_verify`, attaching as a child of
the same round's outcome-processing scope) must stay **one span per
request per round**, exactly like `ignis.decode.round` itself — never one
per verified token. That is the failure mode ADR 0012 and spec §26 call
out explicitly, and why a reviewer should treat any finer subdivision as a
defect, not a style choice.

## Pretty output and trace ids

Spec §19: "pretty output MAY omit trace IDs by default for readability;
debug/verbose modes MAY display them." Rather than adding a second config
surface, `ignis-logging`'s `PrettyLayer` reuses the existing
`IGNIS_LOG_LEVEL`: `Debug`/`Trace` show `trace_id`/`span_id` inline,
`Info`/`Warn`/`Error` (the default) omit them. JSON output always includes
both fields when they exist, regardless of level — nothing about the JSON
event model changes with verbosity.

## Where to attach the next stage

A future stage's span should:

1. Carry `request_id` directly at creation (the id is always known by the
   time any per-request work happens in `concrete.rs` — nothing here needs
   the `Empty` + later `record` pattern the HTTP root span uses).
2. Open and close within the same per-request accounting scope an
   existing loop already provides (mirroring `ignis.prefill` /
   `ignis.decode.round`) rather than around a batched compute call.
3. Stay at the coarsest unit that loop naturally provides — never finer
   than "once per request per existing iteration."

`ignis.decode.mtp_verify` (MTP verify, spec's example) is one span per
request per round, a child of the same scope `ignis.decode.round` opens in,
right where a verify step's result would be applied to the outcome.
