# ADR 0017 — Opt-in Prometheus metrics with zero inference-path work

## Status

Accepted (2026-09-08, owner decision).

## Context

ADR 0011 chooses `tracing` and `tracing-subscriber` as Ignis's structured
logging foundation. It also draws an intentional line around the pre-existing
server telemetry stream: request lifecycle records migrate to canonical log
events, while scheduler interval counters are metrics-shaped and remain outside
the `logging` crate.

The model-thread isolation work (#69) already provides the safe seam needed for
metrics. The model thread sends lightweight submission, routed-event, and tick
facts to one asynchronous telemetry consumer. That consumer performs request
bookkeeping and sink I/O off the model thread and publishes its latest interval
counters through a wait-free snapshot.

Prometheus support is useful for monitoring request load, queue pressure,
latency, generated work, KV evictions, and sibling-prefix reuse. It must not be
implemented by parsing rendered logs, by turning metrics into log records, or
by placing metric macros, atomics, locks, clock reads, allocations,
serialization, registry operations, or new channel sends in scheduling,
prefill, decode, the runtime, or the kernel leaf.

The current telemetry interval contains two placeholder values:
`prefilling = 0` and `kv_used_pct = 0`. The scheduler does not expose
authoritative values for them. Adding a per-step scheduler snapshot solely for
Prometheus would violate the stronger requirement established here: metrics
add zero work to the inference critical path.

## Decision

- Prometheus metrics are a distinct observability signal that extends, but does
  not supersede, ADR 0011. `tracing::Event` remains the canonical log event;
  Prometheus is not a logging layer, formatter, or consumer of rendered log
  output. The `logging` crate does not own metrics.
- Metrics are disabled by default and enabled only by the boolean CLI flag
  `--metrics`. There is no short alias, environment variable, or config-file
  key in this scope.
- When enabled, `GET /metrics` is installed on the existing `ignis-server`
  listener. When disabled, the route is absent and follows the router's normal
  not-found behavior. No second management listener is introduced.
- The response uses Prometheus text exposition format 0.0.4 and content type
  `text/plain; version=0.0.4; charset=utf-8`, with stable `HELP` and `TYPE`
  declarations.
- The existing asynchronous server telemetry consumer is deepened into the
  single projection owner for operational facts. It updates aggregate metric
  state from facts the model thread already emits. Scrape-time encoding happens
  only in the HTTP task.
- The scheduler interface, core, runtime, kernel leaf, and model-thread loop are
  unchanged. Flag-off and flag-on use the same inference path and produce the
  same successful-workload telemetry messages. Enabling metrics adds no
  scheduler query, snapshot, event, branch, atomic operation, clock read,
  allocation, task wake, event clone, or channel operation to that path.
- Rejected submissions may be recorded after the existing async submission
  call receives its error. This is HTTP/control-plane work and does not add a
  model-thread telemetry fact.
- Metric aggregation runs only when metrics are enabled. A disabled server
  installs neither the Prometheus projection nor the route.
- Aggregate state may use fixed atomics or immutable snapshots owned by the
  async telemetry side. No lock is shared with inference, and a scrape never
  sends a command to or waits for the model thread.
- A scrape allocates only its response buffer and formats the latest published
  projection. A slow or disconnected scraper cannot apply backpressure to the
  telemetry facts channel or inference.
- Only authoritative values are exported. `prefilling` and KV usage/capacity
  are omitted while the implementation can provide only placeholder zeros.
  They may be added by a future ADR only if their source preserves this ADR's
  zero-work inference-path guarantee.

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

Zero critical-path regression is an architectural invariant, not a percentage
budget. Acceptance requires both structural and measured evidence:

- review/tests prove there is no metrics dependency or metrics-aware code in
  core, runtime, the kernel leaf, or the model-thread loop, and that flag-off
  and flag-on produce identical inference-side fact traffic;
- a real-GPU trace replay shows no repeatable regression in GPU step timings or
  model-thread throughput; any repeatable critical-path regression is a failure,
  even below 1%;
- the only permitted 1% budget is on the asynchronous HTTP/telemetry plane:
  under a 15-second scrape cadence, per-class HTTP-observed TTFT is at most
  1.01 times the disabled baseline and delivered generation throughput is at
  least 0.99 times the disabled baseline;
- an enabled-but-unscraped run must also pass, and a one-second scrape cadence
  is recorded as a stress diagnostic;
- the existing G4 performance gate remains independently mandatory.

Functional acceptance uses the highest existing seams: pure CLI resolution for
the flag, and the public server router backed by a concrete scheduler over
deterministic `MockCompute` for route presence, exposition, lifecycle changes,
concurrency, and slow-scraper isolation. `cargo test` must pass workspace-wide;
functional Prometheus tests require no GPU.

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
- Prometheus encoding and aggregation can consume async CPU time and affect
  HTTP-plane latency; that is why the explicit 1% HTTP budget exists. They
  cannot consume inference-path time by design.
- Work is split into two implementation tickets: #89 owns the opt-in
  asynchronous metrics projection; #90 owns `--metrics`, the HTTP exporter,
  and performance acceptance. #90 is blocked by #89.
- Pushgateway, remote write, OTLP metrics, StatsD, dashboards, alerting rules,
  deployment manifests, a separate management listener, and metrics-specific
  authentication/TLS are out of scope.
