## Problem Statement

Once structured logging exists and is proven safe on the hot path (Phases 1-3), individual log records still don't tell the owner which internal stages a given request actually traversed and how long each took — admission, prefill, one or more decode rounds, MTP verify, completion. Correlating that today means manually cross-referencing timestamps and request IDs across unrelated log lines.

## Solution

Instrument request handling with a hierarchical `tracing` span tree: a root span opened at HTTP ingress, with child spans for admission, prefill, each decode round (not per-token), MTP verify (when active), and completion. The OTel `trace_id` carried on every log record within that tree is the existing `request.id` (ADR 0012) — no separate, fabricated trace identifier. This is internal, single-process observability, not distributed tracing across services.

## User Stories

1. As the owner, I want to see, for a given `request.id`, the full sequence of internal stages it went through (admitted → prefill → decode round(s) → done), so that I can understand where time went in a specific request without guessing from timestamps alone.
2. As the owner, I want the `trace_id` on every log record inside a request's lifecycle to equal that request's existing `request.id`, so that I only ever need to grep one identifier to reconstruct the whole story, instead of correlating two separate IDs.
3. As the owner, I want span granularity to stop at the decode round (the same unit a decode CUDA graph is captured over), never per-token, so that tracing itself doesn't become hot-path logging by another name.
4. As the owner, I want the HTTP root span to come from `tower-http`'s `TraceLayer` on the existing `axum` server rather than a hand-rolled equivalent, so that this reuses well-tested infrastructure instead of a bespoke HTTP-layer span implementation.
5. As the owner, I want events emitted outside an active request (startup, GPU discovery, shutdown) to remain valid with no `trace_id`/`span_id` populated, so that the logging system never fabricates a trace context that doesn't exist just to fill a field.
6. As the owner, I want any span added inside the true per-token decode loop (as opposed to per-round) to require the same G4 performance gate re-validation as any other hot-path logging change, so that tracing doesn't quietly reintroduce the overhead Phase 3 worked to prevent.
7. As a future maintainer, I want the span hierarchy's shape (which stages nest under which) documented once, so that adding a new stage (e.g. a future speculative-decoding verify step) has an obvious place to attach its span.

## Implementation Decisions

- `tower-http`'s `TraceLayer` added to the `axum` router in `crates/server`, providing the root span per HTTP request.
- Child spans opened by whichever code already owns each stage's boundary: admission (entry/exit of the admission state machine for a request), prefill (the existing chunked-prefill call boundary), each decode round (the existing unit a decode CUDA graph is captured over — one span per round, not per token), MTP verify (when speculative decoding is active), and completion.
- `trace_id` = `request.id` (ADR 0012): the root span's trace identifier is derived from the request's existing `RequestId` (already used by `crates/core`'s admission machinery and `crates/server/src/telemetry.rs`), not a separately generated ID. No new ID-generation code path is introduced for this purpose.
- Events emitted with no active span (process startup/shutdown, GPU discovery, CUDA init, background tasks) simply have no `trace_id`/`span_id` — the logging system (Phase 1's JSON/pretty layers) already treats these fields as optional; this issue does not change that.
- Pretty output may omit trace/span IDs by default for readability; a debug/verbose pretty mode may show them (small, layer-level formatting decision, not a new config surface unless one is already needed).

## Testing Decisions

- Tests asserting: a request's root span and all its child spans/log records within that request's lifecycle carry the same `trace_id`, and that `trace_id` equals the request's `RequestId`; an event emitted outside any active span has no `trace_id`/`span_id` present (not a zero/placeholder value — genuinely absent); span boundaries align with the documented stages (one span per decode round, confirmed by asserting span count equals round count for a multi-round test request, not per-token count).
- These correspond to spec §39 tests #10 and #11 (trace IDs included when context exists / not fabricated when it doesn't).
- **GPU test: re-run `ignis-bench gate` (G4) once span instrumentation is wired into the live prefill/decode path, per Phase 3's standing rule that any hot-path-adjacent change needs the performance gate, not just a design argument. If any span is found to land inside the per-token loop rather than per-round, that is a defect to fix before this issue is considered done, not an accepted tradeoff.**

## Out of Scope

- Distributed tracing across processes/services (incoming `traceparent` propagation from external callers) — ignis is single-process; this is explicitly deferred until (if ever) a multi-process/multi-node deployment exists, at which point `request.id`-as-`trace_id` may need revisiting (noted in ADR 0012).
- Direct OTLP export of spans/logs — spec §21 already settles that stdout + an external collector is the supported shape; this issue does not add an exporter.
- Any change to the C/C++/CUDA/FFI logging bridge (§33/34) — still N/A.
- New CLI/env configuration specific to tracing — this issue reuses whatever `IGNIS_LOG_FORMAT`/`IGNIS_LOG_LEVEL` (and, by then, CLI flag) surface already exists from Phase 1.

## Further Notes

- Depends on Phase 1, 2, and 3 landing first. Filed last deliberately, per the owner's explicit sequencing (tracing is the piece most likely to need the other phases' groundwork — event model, migrated call sites, and proven hot-path safety — settled first).
- Full spec: `.scratch/Ignis Structured Logging Specification.md` §19-20. ADR 0012 records the `request.id`-as-`trace_id` decision and its rationale in full. Roadmap: `.scratch/logging-roadmap.md`.
