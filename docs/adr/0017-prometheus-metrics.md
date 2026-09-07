# ADR 0017 — Opt-in Prometheus metrics with zero inference-path work

## Status

Accepted (2026-09-08, owner decision).

## Context

ADR 0011 chooses `tracing` and `tracing-subscriber` as Ignis's structured
logging foundation. It also draws an intentional boundary around the existing
server telemetry stream: request lifecycle records migrate to canonical log
events, while scheduler interval counters are metrics-shaped and remain outside
the `logging` crate.

The model-thread isolation work (#69) already provides a safe observability
seam. The model thread sends lightweight submission, routed-event, and tick
facts to one asynchronous telemetry consumer. That consumer performs request
bookkeeping and sink I/O off the model thread and publishes its latest interval
counters through a wait-free snapshot.

## Problem Statement

Operators need Prometheus-compatible visibility into request load, queue
pressure, latency, generated work, KV evictions, and sibling-prefix reuse.
Ignis currently has no Prometheus endpoint, and building one from rendered logs
would create a second, fragile interpretation of operational facts.

Metrics must be optional and must not tax inference when enabled. In
particular, their acceptable regression on the scheduler, prefill, decode,
runtime, kernel, model-thread, and GPU critical path is exactly zero. A
percentage allowance is appropriate only for asynchronous HTTP and telemetry
work. The current telemetry interval also exposes placeholder values for
prefilling and KV usage, so those values cannot become authoritative metrics.

## Decision

Ignis will expose a bounded Prometheus text endpoint when the operator supplies
`--metrics`. The endpoint will project facts already consumed asynchronously;
it will not introduce metric work or new fact traffic on the inference path.

## Solution

When metrics are enabled, the existing asynchronous server telemetry consumer
maintains a fixed-cardinality aggregate projection from facts that the model
thread already emits. `GET /metrics` renders the latest projection on the
existing server listener. When metrics are disabled, neither the projection nor
the route exists.

Prometheus remains distinct from structured logging: it does not parse rendered
logs, and the logging subsystem does not own metric aggregation or exposition.
Only authoritative values enter the initial metric contract. The implementation
must demonstrate structural equality of inference-side work and no repeatable
critical-path performance regression. The only numerical allowance is at most
1% on the HTTP/telemetry plane under the defined scrape workload.

## User Stories

1. As an operator, I want to enable metrics explicitly with `--metrics`, so that a default Ignis deployment has no metrics surface or aggregation work.
2. As an operator, I want `GET /metrics` on the existing listener, so that I do not need to provision a second management endpoint.
3. As an operator, I want the metrics route to be absent when disabled, so that opt-in behavior is unambiguous.
4. As a Prometheus administrator, I want a standards-compatible text response, so that Prometheus can scrape Ignis without a custom adapter.
5. As an operator, I want build identity, so that I can associate observations with the running Ignis version.
6. As an operator, I want current waiting and running request counts, so that I can see scheduler pressure.
7. As an operator, I want accepted, completed, and rejected request totals, so that I can understand admission and completion behavior.
8. As an operator, I want rejection totals split by a fixed reason set, so that I can distinguish capacity and request-shape failures without unbounded labels.
9. As an operator, I want generated-token totals, so that I can observe delivered work.
10. As an operator, I want KV eviction totals, so that I can identify cache pressure.
11. As an operator, I want sibling-prefix reuse totals, so that I can quantify avoided prompt work.
12. As an operator, I want TTFT and request-duration histograms, so that I can monitor latency distributions rather than averages alone.
13. As a performance engineer, I want enabled metrics to add zero work to inference, so that observability cannot reduce model-thread or GPU throughput.
14. As a performance engineer, I want any repeatable inference critical-path regression to fail acceptance, even below 1%, so that the zero-work invariant is not weakened into a budget.
15. As a server operator, I want slow or disconnected scrapers isolated from inference, so that monitoring cannot apply backpressure to request execution.
16. As a server operator, I want bounded labels and fixed histogram buckets, so that metric cardinality and storage cost remain predictable.
17. As a developer, I want placeholder telemetry values omitted, so that every exported series has an authoritative meaning.
18. As a developer, I want functional coverage without requiring a GPU, so that endpoint behavior remains fast to verify in the workspace test suite.
19. As a release engineer, I want a real-GPU comparison in addition to structural tests, so that the zero-regression claim is supported by measured evidence.
20. As an observability maintainer, I want logs and metrics derived independently from canonical facts, so that neither signal depends on parsing the other's rendered output.

## Implementation Decisions

- Prometheus metrics extend, but do not supersede, ADR 0011. A
  `tracing::Event` remains the canonical log event; Prometheus is not a logging
  layer, formatter, or consumer of rendered log output.
- Metrics are disabled by default and enabled only by the boolean CLI flag
  `--metrics`. There is no short alias, environment variable, or config-file
  key in this scope.
- When enabled, `GET /metrics` is installed on the existing `ignis-server`
  listener. When disabled, normal router not-found behavior applies. No second
  management listener is introduced.
- The response uses Prometheus text exposition format 0.0.4 and content type
  `text/plain; version=0.0.4; charset=utf-8`, with stable `HELP` and `TYPE`
  declarations.
- The existing asynchronous telemetry consumer is the single projection owner.
  It updates aggregate state from facts already emitted by the model thread;
  scrape-time encoding happens only in the HTTP task.
- The scheduler interface, core, runtime, kernel leaf, and model-thread loop are
  unchanged. Flag-off and flag-on produce identical inference-side fact
  traffic for the same successful workload.
- Enabling metrics adds no scheduler query, snapshot, event, branch, atomic
  operation, clock read, allocation, task wake, event clone, or channel
  operation to the inference path.
- Rejected submissions may be recorded after the existing asynchronous
  submission call returns its error. This is HTTP/control-plane work and does
  not add a model-thread telemetry fact.
- Aggregation runs only when metrics are enabled. A disabled server installs
  neither the Prometheus projection nor the route.
- Aggregate state may use fixed atomics or immutable snapshots owned by the
  asynchronous telemetry side. No lock is shared with inference, and a scrape
  never sends a command to or waits for the model thread.
- A scrape allocates only its response buffer and formats the latest published
  projection. A slow or disconnected scraper cannot apply backpressure to the
  telemetry facts channel or inference.
- Only authoritative values are exported. Prefilling and KV usage/capacity are
  omitted while only placeholder zeros are available. Adding them requires a
  future ADR that preserves this ADR's zero-work invariant.

The initial stable metric contract is:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `ignis_build_info` | gauge | `version` | Constant build identity with value 1 |
| `ignis_scheduler_requests` | gauge | `state=waiting\|running` | Current requests by observable scheduler state |
| `ignis_kv_cache_evictions_total` | counter | none | Cumulative host-tier evictions |
| `ignis_prefix_reused_tokens_total` | counter | none | Cumulative tokens skipped through sibling-prefix reuse |
| `ignis_requests_accepted_total` | counter | none | Accepted submissions |
| `ignis_requests_completed_total` | counter | none | Completed requests |
| `ignis_requests_rejected_total` | counter | `reason=full\|unknown_model\|oversized` | Rejected submissions by fixed reason |
| `ignis_generated_tokens_total` | counter | none | Generated tokens on completed requests |
| `ignis_request_ttft_seconds` | histogram | none | Submission-to-first-token latency |
| `ignis_request_duration_seconds` | histogram | none | Submission-to-completion latency |

Histogram buckets are fixed. TTFT uses 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5,
10, 30, 60, 120, and 300 seconds. Request duration uses 0.1, 0.25, 0.5, 1,
2.5, 5, 10, 30, 60, 120, 300, and 600 seconds. Both include Prometheus's
implicit positive-infinity bucket.

Labels are limited to the finite sets above. Request IDs, trace/span IDs,
prompts, completions, artifact paths, client-provided model strings, arbitrary
error text, sequence IDs, lane IDs, and token IDs are forbidden as labels.
ADR 0012's request identity is for log/span correlation, not a metric
dimension.

## Testing Decisions

- Functional tests assert external behavior rather than private aggregation
  structure. The primary seam is the public server router backed by a concrete
  scheduler over deterministic `MockCompute`; the secondary seam is pure CLI
  configuration resolution.
- CLI coverage proves that metrics default to disabled, that `--metrics`
  enables them, and that no alias, environment variable, or configuration-file
  path is introduced.
- Router coverage proves route absence when disabled, valid exposition and
  content type when enabled, lifecycle-driven value changes, bounded labels,
  concurrent scrapes, and slow-scraper isolation.
- Structural review and tests prove there is no metrics dependency or
  metrics-aware code in core, runtime, the kernel leaf, or the model-thread
  loop, and that flag-off and flag-on produce identical inference-side fact
  traffic.
- A real-GPU trace replay compares metrics disabled, enabled but unscraped, and
  enabled with a 15-second scrape cadence. There must be no repeatable
  regression in GPU step timings or model-thread throughput. Any repeatable
  critical-path regression is a failure, even below 1%.
- The only permitted 1% budget applies to asynchronous HTTP/telemetry behavior:
  under a 15-second scrape cadence, per-class HTTP-observed TTFT is at most
  1.01 times the disabled baseline and delivered generation throughput is at
  least 0.99 times the disabled baseline.
- A one-second scrape cadence is recorded as a stress diagnostic. Slow and
  disconnected scraper scenarios must not perturb inference-side fact traffic
  or block telemetry consumption.
- The existing G4 performance gate remains independently mandatory.
- `cargo test` must pass workspace-wide. Functional Prometheus tests require no
  GPU and follow the existing CLI resolution, telemetry, and OpenAI HTTP test
  patterns.

## Out of Scope

- Pushgateway, remote write, OTLP metrics, and StatsD.
- Dashboards, alerting rules, and deployment manifests.
- A separate management listener or metrics-specific authentication/TLS.
- Prefilling and KV usage/capacity metrics without authoritative zero-work
  sources.
- Request-, trace-, token-, sequence-, lane-, prompt-, or error-derived labels.
- Replacing structured logging or reconstructing either observability signal
  from the other's rendered output.

## Further Notes

- Work is delivered through two tracer-bullet implementation tickets.
- #89 delivers a minimal end-to-end Prometheus surface: opt-in CLI behavior,
  route presence, valid exposition, and core request/scheduler metrics backed
  by the asynchronous projection.
- #90 is blocked by #89 and completes the operational metric contract,
  slow/concurrent scraper isolation, and all structural and measured
  performance acceptance.
- Performance wording is intentionally asymmetric: zero repeatable regression
  is required on the inference critical path; the 1% budget exists only for
  asynchronous HTTP/telemetry behavior.

## Consequences

- Operators gain a standard pull-based monitoring surface without introducing
  a second observability path through inference.
- The metric surface is intentionally small. Missing but desirable values are
  omitted rather than guessed, and every future series needs an authoritative
  source, a type/unit contract, a bounded-cardinality review, and a performance
  classification.
- Existing JSONL interval telemetry remains compatible. Canonical request logs
  and aggregate metrics may originate from the same lifecycle facts, but
  neither is reconstructed by parsing the other's rendered output.
- Prometheus encoding and aggregation can consume asynchronous CPU time and
  affect HTTP-plane latency; that is why the explicit 1% HTTP budget exists.
  They cannot consume inference-path time by design.
