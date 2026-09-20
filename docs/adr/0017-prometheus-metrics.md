# ADR 0017 — Opt-in Prometheus metrics with zero inference-path work

## Status

Accepted (2026-09-08, owner decision). The "Existing JSONL interval telemetry
remains compatible" consequence is superseded by ADR 0025; the projection
still reads the telemetry consumer, never rendered log output. Amended
2026-09-15 (#89, owner request): `ignis_requests_cancelled_total` joins the
contract, recorded from the control plane's cancel rather than a model-thread
fact. Amended again 2026-09-15 (#89, owner decision): `GET /metrics` moves to
its own listener (`--metrics-bind`, no API key, never exposed), and the
Playground reads the same exposition at `/ui/metrics` on the API listener,
under the API key when one is set. This replaces "on the existing listener".
Clarified 2026-09-15 (#90): a submission longer than the per-sequence context
(`ContextExceeded`, #166, which postdates the table) counts under
`reason="oversized"` — like a request larger than the KV pool, it can never
fit — so the reason set stays the three below. The latency histograms' "submission"
is the scheduler's acceptance as the telemetry consumer observes it, the same
anchor as the `ignis.request.ttft`/`done` log events: measuring from HTTP
ingress would take a fact the consumer does not already receive. A request
cancelled after its first token has a TTFT observation but no duration one.
Amended 2026-09-16 (#190, accepted by the owner the same day): six
fixed-cardinality counter families for retained state, each split by
`tier="device|kv_ram"`, and the core emits the retained-state lifecycle facts
they are projected from (see Decision). The proposed widening of
`ignis_prefix_reused_tokens_total` (#188), at the end of this document, was
**not** taken: the counter keeps its sibling-prefix meaning, and retained
reuse is counted in its own family instead. Amended 2026-09-18 (owner
request): the KV usage and capacity values this ADR leaves out, pending "a
future ADR that preserves this ADR's zero-work invariant", are taken by
ADR 0030 §Observability, which also adds the VRAM plan, the KV-RAM arena and
the retained slots as gauges (#216, #217). The same amendment splits the six
retained-state families by `kind="checkpoint|prefix"` as well as by `tier`,
so the six rows below read "by tier and kind"; a prompt checkpoint and a
shared prefix are no longer counted as one thing. Lane and sequence labels
stay forbidden.

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
thread already emits. `GET /metrics` renders the latest projection on a
metrics listener of its own, and the Playground reads the same rendering at
`/ui/metrics` on the API listener. When metrics are disabled, neither the
projection nor any route to it exists.

Prometheus remains distinct from structured logging: it does not parse rendered
logs, and the logging subsystem does not own metric aggregation or exposition.
Only authoritative values enter the initial metric contract. The implementation
must demonstrate structural equality of inference-side work and no repeatable
critical-path performance regression. The only numerical allowance is at most
1% on the HTTP/telemetry plane under the defined scrape workload.

## User Stories

1. As an operator, I want to enable metrics explicitly with `--metrics`, so that a default Ignis deployment has no metrics surface or aggregation work.
2. As an operator, I want `GET /metrics` on its own local listener, so that Prometheus can scrape it without the API key while an exposed API never publishes it.
3. As an operator, I want the metrics route to be absent when disabled, so that opt-in behavior is unambiguous.
4. As a Prometheus administrator, I want a standards-compatible text response, so that Prometheus can scrape Ignis without a custom adapter.
5. As an operator, I want build identity, so that I can associate observations with the running Ignis version.
6. As an operator, I want current waiting and running request counts, so that I can see scheduler pressure.
7. As an operator, I want accepted, completed, cancelled, and rejected request totals, so that I can understand admission and completion behavior.
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
- When enabled, `GET /metrics` is served by a second `ignis-server` listener
  of its own, at `--metrics-bind <addr>` (flag only, default
  `127.0.0.1:9464`; refused without `--metrics` or on the API's `--bind`). It
  serves nothing else and asks for no API key: it is kept private by its bind
  address, and `--expose` (ADR 0028) only ever tunnels the API listener. One
  shutdown stops both listeners.
- The API listener has no `/metrics`. With the Playground on (`--ui`, ADR
  0026), it serves the same exposition at `GET /ui/metrics` for the browser,
  behind the same API key as `/v1` when one is set and open when none is.
- When disabled, there is no metrics listener and normal router not-found
  behavior applies to both paths.
- The response uses Prometheus text exposition format 0.0.4 and content type
  `text/plain; version=0.0.4; charset=utf-8`, with stable `HELP` and `TYPE`
  declarations.
- The existing asynchronous telemetry consumer is the single projection owner.
  It updates aggregate state from facts already emitted by the model thread;
  scrape-time encoding happens only in the HTTP task.
- Retained prompt-checkpoint lookup and lifecycle operations emit bounded
  domain facts for hit, miss, spill, discard and restore (#190). Those facts
  are emitted identically with metrics off and on; only the asynchronous
  telemetry consumer conditionally projects them. Runtime and kernel code do
  no metrics work.
- Enabling metrics adds no scheduler query, snapshot, event, branch, atomic
  operation, clock read, allocation, task wake, event clone, or channel
  operation to the inference path.
- Rejected submissions may be recorded after the existing asynchronous
  submission call returns its error. This is HTTP/control-plane work and does
  not add a model-thread telemetry fact.
- Cancelled requests are recorded the same way, from the control plane. A
  client that goes away cancels its request, and the scheduler releases a
  cancelled request without routing any event, so the engine's cancel call
  tells the telemetry consumer directly; the model-thread loop and its facts
  stay unchanged. Each accepted request is counted once, as completed or as
  cancelled; when both happen in the same scheduler step, whichever the
  consumer observes first wins. Once the consumer has drained, accepted equals
  completed plus cancelled plus the requests still in flight.
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
| `ignis_kv_ram_evictions_total` | counter | none | Live host-tier snapshots dropped **out of** KV-RAM to make room; the owning request re-prefills from the start (#224) |
| `ignis_prefix_reused_tokens_total` | counter | none | Cumulative tokens skipped through sibling-prefix reuse — a live sibling's prefix only; a retained prefix's claim is counted below (#190) |
| `ignis_retained_reused_tokens_total` | counter | `tier=device\|kv_ram`, `kind=checkpoint\|prefix` | Cumulative tokens skipped through retained state: a retained prefix (always `device`) or a prompt checkpoint, by the tier it came from and the kind it was |
| `ignis_retained_state_hits_total` | counter | `tier=device\|kv_ram`, `kind=checkpoint\|prefix` | Retained state a request chose to resume from: a prompt checkpoint (not yet restored), or a retained prefix brought back from KV-RAM |
| `ignis_retained_state_misses_total` | counter | `tier=device\|kv_ram`, `kind=checkpoint\|prefix` | Requests whose first prefill chunk landed with no checkpoint matching in a tier this load carries. **`kind=prefix` is zero by construction** — see the note below the table (#216, #222) |
| `ignis_retained_state_spills_total` | counter | `tier=device\|kv_ram`, `kind=checkpoint\|prefix` | Retained checkpoints and prefixes written into the tier (today only `kv_ram`) |
| `ignis_retained_state_discards_total` | counter | `tier=device\|kv_ram`, `kind=checkpoint\|prefix` | Retained checkpoints and prefixes that left the tier for nowhere |
| `ignis_retained_state_restores_total` | counter | `tier=device\|kv_ram`, `kind=checkpoint\|prefix` | Prefills that landed on a checkpoint from the tier, and prefixes brought back from it; a hit whose prefill never lands has no restore |
| `ignis_requests_accepted_total` | counter | none | Accepted submissions |
| `ignis_requests_completed_total` | counter | none | Completed requests |
| `ignis_requests_cancelled_total` | counter | none | Accepted requests cancelled before completion |
| `ignis_requests_rejected_total` | counter | `reason=full\|unknown_model\|oversized` | Rejected submissions by fixed reason |
| `ignis_generated_tokens_total` | counter | none | Generated tokens on completed requests |
| `ignis_decoded_tokens_total` | counter | none | Tokens generated so far, counted as each one is emitted |
| `ignis_request_ttft_seconds` | histogram | none | Submission-to-first-token latency |
| `ignis_request_duration_seconds` | histogram | none | Submission-to-completion latency |
| `ignis_decisions_total` | counter | `type=noul\|choice\|score` | Questions answered by a readout, by typed primitive (#241, ADR 0034). **Absent until the first one** — see below |
| `ignis_decision_answer_mass` | histogram | none | Share of the next-token distribution held by a decision's declared options. **Absent until the first one** |

ADR 0030 §Observability adds the memory gauges to this contract: the plan's
eleven reserved lines, the budget, the KV pool's pages and page bytes, the
pages occupied of it, the KV-RAM arena's capacity and use, the retained slots'
capacity and use, and the retained-slot skips. Every one is bytes, pages or
slots; no percentage is exported.

**The decision family is the one thing here that is absent when it is zero
(#241).** Every other series is exported from the first scrape, zeros
included, because a zero is a reading. A decision's are not exported until a
decision has been served, and that is this ADR's other rule — *only
authoritative values are exported* — applied to a route most loads never
call: four permanently-zero series and an eleven-bucket histogram on every
scrape of every server would be clutter that says nothing about the server
it is scraped from. Prometheus handles a series that appears mid-window the
way it handles a new target. Once the family exists, **all three `type`
values are exported**, including the zeros: a label value that vanishes with
its count is a series that breaks `sum by (type)` the moment traffic shifts.

The pair is recorded by the HTTP handler, like `ignis_requests_rejected_total`
and for the same reason: there is no fact for it on the model thread's
stream, and inventing one would put a decision's arithmetic on the inference
path to observe something the handler already holds. It is counted **per
question**, not per request — twenty questions over one `state` are twenty
readouts, and one of them collapsing while its siblings are fine is exactly
what the histogram exists to show.

`ignis_decision_answer_mass` is a ratio in `[0, 1]`, which is not the
percentage this ADR forbids: the forbidden thing is a ratio *standing in for*
two terms a reader needs separately, and answer mass has no second term — it
is the quantity itself. Its buckets are 0.5, 0.9, 0.95, 0.98, 0.99, 0.995,
0.998, 0.999, 0.9995 and 1, plus the implicit `+Inf`. The measured baseline
is a median of 0.996 and above from 8 to 256 options, so an evenly spaced
scale would put every healthy reading in one bucket and show a flat line
whatever happened; the resolution is where the signal is, and the two coarse
buckets below exist to make a collapse unmissable rather than to resolve it.
`le="1"` equals `_count` on a correct readout, and that redundancy is the
assertion the exposition carries.

**`confidence` is not exported.** The endpoint reports a per-answer
confidence, and aggregating it would produce a histogram over callers who
each mean something different by it — a threshold is a property of a domain,
not of a server. Answer mass is the server's own reading of the same prompt
and is comparable across every caller.

**Eviction is a departure from a tier, and there are five of them (#224).**
The contract names each one separately rather than summing them, because they
cost different things:

| Departure | Series |
|---|---|
| A live sequence leaves the device for KV-RAM | `ignis_kv_cache_evictions_total` |
| Retained state leaves the device for nowhere | `ignis_retained_state_discards_total{tier="device"}` |
| Retained state is demoted device → KV-RAM | `ignis_retained_state_spills_total{tier="kv_ram"}` |
| Retained state leaves KV-RAM for nowhere | `ignis_retained_state_discards_total{tier="kv_ram"}` |
| A **live snapshot** is dropped out of KV-RAM | `ignis_kv_ram_evictions_total` |

The last is the most expensive event in the system — the request loses every
prefilled token — and until #224 it was counted nowhere. It is projected from
`SchedEvent::SnapshotDropped` and never from the `SchedEvent::Requeued` beside
it, because requeue also follows a *failed restore*, where KV-RAM evicted
nothing. `ignis_kv_cache_evictions_total` keeps its unlabelled identity: no
`tier` label was retro-fitted onto a stable row.

**A disk tier is reserved, not exported.** `ignis_*` carries no
`tier="disk"` label value and `ReuseSource` has no `Disk` variant. Widening
five per-tier families and the request log's tier spelling for a tier that
does not exist would put a permanently-zero label on the contract. The name
`tier="disk"` is reserved here for whoever builds one; the Monitor shows a
disk row as explicitly not implemented, fed by no metric.

**The miss family is checkpoint-only, and stays that way until someone decides
otherwise.** `ignis_retained_state_misses_total` is projected from a fact the
*checkpoint pool's* lookup raises. The prefix walk beside it — the longest
device match and the longest spilled match — records no miss at all, so
`kind="prefix"` is zero on every load, forever. The `kind` split (#216) made
this visible rather than creating it: the row above has always counted
checkpoint misses alone, which is why its wording names a checkpoint where its
five siblings name both. No prefix miss is synthesised to make the families
look symmetric. Whether the prefix walk should raise one is a change to the
fact stream, not to this projection, and is GitHub #222.

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
  enables them on the default metrics address, that `--metrics-bind` moves
  that address and is refused without `--metrics` or on the API's `--bind`,
  and that no alias, environment variable, or configuration-file path is
  introduced for either flag.
- Router coverage proves route absence when disabled, valid exposition and
  content type when enabled, lifecycle-driven value changes, bounded labels,
  concurrent scrapes, and slow-scraper isolation; which listener serves which
  path; and `/ui/metrics` following the API key.
- Structural review and tests prove there is no metrics dependency or
  flag-dependent code in core, runtime, the kernel leaf, or the model-thread
  loop. Core emits canonical retained-state lifecycle facts independent of
  Prometheus, and flag-off and flag-on produce identical inference-side fact
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
- Metrics-specific authentication or TLS (the metrics listener has none; the
  Playground's copy reuses the API key).
- Prefilling and KV usage/capacity metrics without authoritative zero-work
  sources.
- Request-, trace-, token-, sequence-, lane-, prompt-, or error-derived labels.
- Replacing structured logging or reconstructing either observability signal
  from the other's rendered output.

## Further Notes

- Work is delivered through two tracer-bullet implementation tickets.
- #89 delivers a minimal end-to-end Prometheus surface: opt-in CLI behavior,
  route presence, valid exposition, and core request/scheduler metrics backed
  by the asynchronous projection, cancellations included.
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

## Proposed amendment (2026-09-16) — the prefix-reuse counter widens (#188) — superseded

**Superseded by #190 (2026-09-16).** The owner chose the decline path below:
`SchedEvent::PrefixReused` carries whether the claimed entry's publisher had
already finished, the projection counts only a live sibling's claim in
`ignis_prefix_reused_tokens_total`, and a retained prefix's claim lands in
`ignis_retained_reused_tokens_total{tier="device"}`. The text below is kept as
the record of what was proposed.

**Not decided.** This ADR is an owner-decision ADR whose clarifications are
reserved for the owner's sign-off, so #188 records this rather than taking it.
Until it is signed off, the table row above stands as written and this section
describes what the code already emits.

**What changed underneath the counter.** `ignis_prefix_reused_tokens_total`
counted sibling-prefix reuse (core-07). ADR 0029's **retained prefix**, built
in #188, is the same object claimed by the same call and reported by the same
lifecycle fact (`SchedEvent::PrefixReused`), so reuse by a request whose
publisher has already finished now lands in this series too. The two cannot be
told apart at this seam.

**The proposal.** Read the series as *all* prefix reuse, sibling and retained
together — "Cumulative tokens skipped through prefix reuse, sibling and
retained together (separated per tier by #190)".

**What accepting it also touches**, so the blast radius is on one page: the
Problem Statement's list of what operators need visibility into, and user
story 11, both say "sibling-prefix reuse". Accepting this widens their reading
too. Nothing else in this ADR names the series.

Safe to accept as it stands:

- the series' **name, type and empty label set are unchanged**, so no
  consumer breaks and no dashboard query is rewritten;
- **prompt-checkpoint reuse stays out of it.** That is the other kind of
  cross-request reuse, and it remains a per-request field on the request log
  (`reuse_source` / `reused_prompt_tokens` / `restore_ms`, #186) — the
  distinction this row was kept narrow to preserve;
- the widening is already declared in the code, at the three places a reader
  meets the counter: its help text (`crates/server/src/metrics.rs`),
  `TelemetrySink::on_prefix_reused` and `SchedEvent::PrefixReused`. Those
  declarations exist so the code does not silently contradict this table while
  the question is open.

**If the owner declines**, the counter stays narrow and the *routing* changes
instead, which is #190's work rather than a revert of #188: either
`SchedEvent::PrefixReused` carries which kind of entry was claimed and the
projection counts only the sibling kind here, or the scheduler emits a
distinct fact for a retained-prefix claim. #190 already owns the per-tier
hit / miss / spill / restore counters and the amendment that declares them,
so the separation has a home either way. The three in-code declarations would
then be updated to say the counter is narrow again.
