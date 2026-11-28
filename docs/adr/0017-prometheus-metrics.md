---
status: accepted
---

# Prometheus metrics with zero inference-path work, on by default

ignis-server serves a Prometheus text exposition, so that operators have a standard pull-based monitoring surface
without a second observability path through inference. The model thread already sends lightweight submission,
routed-event and tick facts to one asynchronous telemetry consumer, which does its bookkeeping off that thread: the
consumer owns one fixed-cardinality projection of those facts, kept in fixed atomics, and a scrape formats the atomics
in the HTTP task. Metrics are on by default (`server.metrics`, a field declared as ADR 0046 has it): they were opt-in
only until it was shown that they cost the inference path nothing, and every server the owner runs carried them.

**Metrics add no work to the inference path.**

- The facts are sent whether metrics are on or off, the retained-state lifecycle facts and the occupancy ADR 0030 put
  on the tick included. Enabling metrics adds no query, branch, atomic, clock read, allocation, wake or channel
  operation, and no new fact traffic, to the scheduler, the model thread, the runtime or the kernel leaf.
  `server.metrics: false` builds neither the projection nor a route to it.
- A scrape never waits on the model thread, and a slow scraper cannot hold back the facts channel.
- Core keeps a count as a fact and the server names and renders it, so nothing on the inference path names an
  exposition.
- A count the kernel leaf or a table takes identically with metrics off and on is such a fact, not metrics work.
  Flash-Next's residency and n-gram counts never travel the fact channel: they are written into host memory handed
  over at load, and the consumer reads them on every tick (`FlashNextCounterSource`) with no CUDA call and no wait. It
  stays the projection's single owner, and a scrape still formats atomics only.
- A value no telemetry fact carries is written into the same atomics, off the model thread, by whatever already holds
  it: the HTTP handlers (rejections, decisions and `locate`s, the counting routes, the tool calls), the Responses
  socket queue, the image and speech jobs, the loop guard's own task. No fact is invented for it: that would put work
  on the inference path to observe what its writer already holds.
- A request held to a grammar reads the clock in one place: the scheduler times each repair where the repair
  happens, and reports each miss with its repair time as a fact of its own, so a round with no grammar adds nothing
  (ADR 0089).
- Encoding and aggregation may consume asynchronous CPU time and affect HTTP-plane latency, never inference-path
  time. Their allowance is at most 1% on the HTTP and telemetry plane under a 15-second scrape; it was the acceptance
  of the first implementation and has not been verified again since.
- Logs and metrics come from the same facts, and neither is parsed from the other's rendered output: the projection
  reads the telemetry consumer's state, the logs are ADR 0011's canonical events, and the scheduler's interval
  counters are a DEBUG log event (ADR 0025), with no line or sink of their own.

**Only authoritative values, under bounded labels.**

- A missing but desirable value is omitted rather than guessed, and a placeholder (the interval's `prefilling`) is not
  exported. The surface is kept small on purpose: every new series needs an authoritative source, a type and unit
  contract, a bounded-cardinality review and a performance classification.
- No percentage is exported: a ratio hides which of its two terms moved, and both terms are series themselves.
- The memory gauges are ADR 0030's. KV usage and capacity stayed out of the exposition until that ADR gave them a
  source that keeps the invariant above: the load's own plan, and the scheduler's host-side accounting on the tick.
- Labels take fixed, finite sets. Request, trace, span, sequence, lane and token ids, prompts, completions, paths,
  client model strings and error text are never labels. Histogram buckets are fixed.
- The retained-state families are split by `tier` and by `kind`, so a dashboard that sums one without aggregating
  `kind` away double-counts nothing and reads one line a kind.

**When a family appears.** A family is exported from the first scrape, zeros included, unless it belongs to a route or
a feature most servers never use. Such a family is absent until its first event: a server that never uses the feature
is the normal one, and its permanent zeros on every scrape would be clutter that says nothing. The families of one
model or of an optional tier (Flash-Next's, KV-disk's) render only on a load that has them, and so does the label
value that is the tier's on families every load has, `tier="disk"` on the retained-state families. The plan's lines
(`ignis_vram_reserved_bytes`) and the image model's condition-budget gauge are exported on every load, 0 for a line or
a budget the load does not have: a memory gauge's zero is a reading. Once a family is there, every series of it
renders, zeros included, so a label value never vanishes with its count.

**The listener.** Prometheus scrapes `GET /metrics` on a listener of its own, `server.metrics_bind`: it asks for no
API key, is never the API's `bind`, is never tunnelled by `--server-expose` (the tunnel of ADR 0028), and is refused
beside an explicit `server.metrics: false`. Its default address is a fixed port on loopback, so a listener that asks
for no key is reachable from the host alone unless the operator names another; a second server on the host finds the
port taken, so a default address that cannot be bound is a warning and the server serves without that listener:
refusing would make a default feature a reason not to start. An address other than the default that the operator
named (a flag, its environment variable, a file) is theirs, and refuses the start when it cannot be bound. The API
listener has no `/metrics`: with the Playground on, it serves the same exposition at `GET /ui/metrics`, behind the API
key when one is set (the key that guards `/v1` there), and still does when the metrics listener could not bind, since
the projection stays.

**The catalogue is the guide's.** Every family, its type, its meaning and when it appears is on
[Metrics](../user/reference/metrics.md), generated from `crates/server/src/metrics.rs`, whose help text and doc
comments hold each family's labels and finer rules; `cargo test` fails while the page and the code disagree.

## Considered Options

- **`/metrics` on the API listener**, the first design: replaced by a listener that asks for no key and is never
  exposed, with the Playground's copy left on the API listener behind its key.
- **Opt-in metrics**: held only while their cost to the inference path was unproven.
