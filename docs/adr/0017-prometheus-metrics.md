---
status: accepted
amended-by: [ADR-0025, ADR-0030, ADR-0045, ADR-0046, ADR-0058, ADR-0089]
---

# ADR 0017 — Opt-in Prometheus metrics with zero inference-path work

## Current decision

Metrics are **on by default** (owner decision, 2026-10-15). `server.metrics`
(`--server-metrics`, `IGNIS_SERVER_METRICS`: ADR 0046's spellings of the
original `--metrics`) defaults to `true`, and `false` builds neither the
projection nor a route to it. Prometheus scrapes `GET /metrics`, text format
0.0.4, on a listener of its own, `server.metrics_bind`
(`--server-metrics-bind`, default `127.0.0.1:9464`): it asks for no API key,
is never the API's `bind`, is never tunnelled by `--server-expose` (ADR 0028),
and is refused beside an explicit `false`. A default address that cannot be
bound (a second server on the host) is a warning,
`ignis.server.metrics_unavailable`, and the server serves without that
listener; an address the operator named that cannot be bound refuses the
start. `ignis.process.started` names the address bound, `unavailable` or
`off`. The API listener has no `/metrics`: with the Playground on, it serves
the same exposition at `GET /ui/metrics`, behind the API key when one is set.

**Metrics add no work to the inference path.** The asynchronous telemetry
consumer owns one fixed-cardinality projection, kept in fixed atomics and
updated from facts the model thread sends whether metrics are on or off (the
retained-state lifecycle facts of #190, and the occupancy ADR 0030 put on the
tick, included). Enabling metrics adds no query, branch, atomic, clock read,
allocation, wake or channel operation to the scheduler, the model thread, the
runtime or the kernel leaf. A scrape formats the atomics in the HTTP task and
never waits on the model thread, and a slow scraper cannot hold back the
facts channel. The one allowance, at most 1% on the HTTP and telemetry plane
under a 15-second scrape, was #90's acceptance and is not re-verified here.

A family that no telemetry fact carries is written into the same atomics, off
the model thread, by whatever already holds its value: the HTTP handlers
(rejections, decisions and `locate`s, the counting routes), the Responses
socket queue, the image and speech jobs, and the loop guard's own task.
Flash-Next's residency and n-gram counts are read on every tick from host
memory the leaf and the table write anyway (`FlashNextCounterSource`), with no
CUDA call. Logs and metrics come from the same facts and neither is parsed
from the other's output (ADR 0011); the scheduler's interval counters are a
DEBUG log event (ADR 0025).

**Only authoritative values, under bounded labels.** A placeholder (the
interval's `prefilling`) is not exported, and neither is a percentage. Labels
take fixed, finite sets: request, trace, span, sequence, lane and token ids,
prompts, completions, paths, client model strings and error text are never
labels. Histogram buckets are fixed. A family is exported from the first
scrape, zeros included, unless it belongs to a route or a feature most
servers never use: those are absent until their first event (the decision,
`locate`, counting-route, forced-close, speculative, loop-guard, image-job
and speech families), and the Flash-Next and KV-disk families render only on
a load that has them. Once a family is there, every series of it renders,
zeros included, so a label value never vanishes with its count.

**The catalogue is the guide's.** Every family, its type, its meaning and
when it appears: [Metrics](../user/reference/metrics.md), generated from
`crates/server/src/metrics.rs`, whose help text and doc comments also hold
each family's labels and finer rules; `cargo test` fails while the page and
the code disagree. The memory gauges are ADR 0030's (§Observability).

## History

### Status

Accepted (2026-09-08, owner decision). The "Existing JSONL interval telemetry
remains compatible" consequence is superseded by ADR 0025; the projection
still reads the telemetry consumer, never rendered log output. Amended
2026-09-15 (#89, owner request): `ignis_requests_cancelled_total` joins the
contract, recorded from the control plane's cancel rather than a model-thread
fact. Amended again 2026-09-15 (#89, owner decision): `GET /metrics` moves to
its own listener (`--metrics-bind`, no API key, never exposed), and the
Playground reads the same exposition at `/ui/metrics` on the API listener,
under the API key when one is set. This replaces "on the existing listener".
Amended 2026-09-21 (#255): `type="scalar"` joins `ignis_decisions_total`'s
label set. It is a **generated** answer, so like `number`, `point` and `box`
it observes no answer mass. Clarified 2026-09-15 (#90): a submission longer than the per-sequence context
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
stay forbidden. Amended 2026-09-20 (#241): two rows for the decision
endpoint, `ignis_decisions_total` and `ignis_decision_answer_mass`, both
recorded by the HTTP handler the way the rejection counter is, and both
carrying **the first exception to this ADR's "zeros are exported too"** —
they are absent until a decision has been served. The reasoning is below
the table; the exception is named here because an operator reading only
this section should not meet it as a surprise. Amended 2026-09-20 (#242):
three more `type` values — `number`, `point` and `box` — which are counted
like any other question and observe **no** answer mass, so the histogram is
readout-only and its `_count` sits below the counter's sum by exactly the
number of programs served. That asymmetry is stated below the table too, for
the same reason. Amended 2026-09-24 (#265, spec server/08): one counter,
`ignis_thinking_forced_closes_total`, the completed requests whose reasoning
block the thinking budget closed. It is projected by the telemetry consumer
from the finish event the scheduler already emits, so the model thread does
no new work, and it takes the decision family's exception: absent until the
first forced close. Amended 2026-09-27 (#275): `type="locate"` joins
`ignis_decisions_total`'s label set. A `locate` is read off attention, not an
answer position, so like `point` and `box` it observes **no** answer mass; it
is counted once per question, and its content-free baseline prefill is not a
decision and is not counted. Amended 2026-09-28 (#278, spec 22):
`ignis_locates_total{kind, method, compression, found}` counts each answered
`locate` by the kind it resolved to (`log|prose|records`), its method
(`shortlist|vote`), its compression (`template_fold|none`) and whether it
found an answer (`true|false`, or `unmeasured` on a route with no `found`
rule). Only the eleven combinations the endpoint serves have a series, and the
family takes the decision family's exception: absent until the first `locate`.
A shortlist's labelled `choice`s and yes/no are steps of its one `locate`,
not decisions: they move neither `ignis_decisions_total{type="choice"}` nor
the answer-mass histogram. No answer mass, as for every `locate`.
Amended 2026-09-29 (#282, spec responses-api/01): two unlabelled gauges for
the Responses API's WebSocket mode, `ignis_responses_sockets` (open sockets)
and `ignis_responses_queued_requests` (socket requests waiting in the
server-wide admission queue), set by the server as sockets open and close and
as requests join and leave the queue. A socket request the engine finds full
is queued, not rejected, so it does not move `ignis_requests_rejected_total`;
one refused for good (unknown model, context, size) does, as on HTTP. Both
are exported at zero like every other gauge.
Amended 2026-09-30 (#285, spec server/10): one counter,
`ignis_tokenize_requests_total{route}`, the calls to the two counting routes.
Recorded by the handler, like the rejection counter, and it takes the decision
family's exception: absent until the first call. A tokenize call is not a
request of the lifecycle — it has no lane, no TTFT and no completion — so it
moves none of `ignis_requests_*`.
Amended 2026-10-05 (#301, spec flash-next/03): seven families for Flash-Next's
**expert residency**, the last seven rows of the table. Core keeps the facts
(`ignis_core::residency::ResidencyCounters`, counted by the residency step);
the server names and renders them (`render_expert_residency`), so nothing on
the inference path names an exposition. Fixed labels: `class` the eight K-class
spellings (`gate_up_k2` … `down_k4`), `phase=decode|prefill`,
`state=capacity|in_use`; zeros exported. Only a Flash-Next load has an expert
cache: a 27B load never renders them.
Amended 2026-10-06 (#301, #302, spec flash-next/04): three families for
Flash-Next's **n-gram rows**, `ignis_ngram_rows_total{source=hot|file}`,
`ignis_ngram_reads_total` and `ignis_ngram_read_bytes_total`, the table's last
three rows; and `ignis_expert_residency_stall_seconds_total` is withdrawn
until the device times the wait (it stays in the residency ABI), since only
authoritative values are exported. How the Flash-Next families reach the
projection makes two exceptions, named here:
- *The consumer "updates aggregate state from facts already emitted by the
  model thread".* These counts never travel the fact channel. Residency's
  last-layer step writes its totals into host memory it was handed at load
  (`ignis_residency_set_mirror`), and the n-gram table counts in host atomics;
  `ignis_core::flash_next_counters::FlashNextCounterSource` reads both with
  no CUDA call and no wait. The server hands that source to the consumer once,
  as an async-side `SetCounterSource` message like `SetMetrics`, and the
  consumer reads it on every tick into the projection's fixed atomics. It
  stays the single projection owner, and a scrape still formats atomics only.
- *"Runtime and kernel code do no metrics work."* The kernel's mirror write
  and the table's counts are facts taken identically with metrics off and
  on, like #190's lifecycle facts; the leaf registers the mirror at open and
  does nothing for it after, so enabling metrics adds nothing to a step.
Every Flash-Next family renders after all the others: a 27B load's
exposition is the same bytes it was, and a load the consumer is told has no
source renders none of them.
Amended 2026-10-07 (#306, spec flash-next/03 acceptance 7):
`ignis_expert_residency_stall_seconds_total{phase=decode|prefill}` returns,
now that the device times the wait. Each step's demand copy reads
`%globaltimer` from its first block's start to its last block's end, and its
last block adds the span to the phase's counter and stores the total in the
mirror: no host sync, no extra launch, the same path to the projection as the
other residency counts. Only the demand copies count, so a step with no miss
adds nothing; the prefetch copies on residency's own stream, and a step's wait
for the previous one's, are not in it. Seconds, rendered exactly from the
device's nanoseconds.
Amended 2026-10-07 by ADR 0045 (its rows land with spec `vram-budget/03`): the
reserved `tier="disk"` becomes an exported value of the six retained-state
families on a load with KV-disk. Three disk families join them:
`ignis_kv_disk_bytes{state}`, `ignis_kv_disk_spills_total{from}` and
`ignis_kv_disk_failures_total{op}`. The `disk` series and the three families
render only on such a load.
`ignis_kv_cache_evictions_total` and `ignis_kv_ram_evictions_total` keep their
meanings.
Amended 2026-10-15 (#324 item 4, spec image-model/05): five families for the
**image jobs**, rendered after the counting routes' counter and before
Flash-Next's. The image model has no engine, so no fact of it reaches the
telemetry consumer: the job itself records them (`images::job::spawn`), once,
as it ends, into the same atomics; a scrape still formats atomics only. They
take the decision family's exception: **absent until the image model has
admitted a job**, then every series renders, zeros included. Fixed labels:
`outcome=completed|cancelled|stopped|refused|failed`, `phase=queued|encoding|
denoising|decoding`, `state=waiting|running`. A completed job's duration runs
from its turn to its last image (the PNG encodes and the sends included: the log's
`run_ms`), so the wait is the `queued` phase's alone. The queue gauge is set by the
queue itself, under its lock, at each admission, turn and departure, so the last
change to land is the last one shown.
Each ended job also writes one `ignis.image.job` log line (id, kind, model,
`n`, size, condition images, enhancement, outcome, images made, and the
milliseconds of enhancing, switching, queueing, encoding, denoising,
decoding, the run and the total). The Playground's Monitor reads the families
(an *Image jobs* card, shown once they exist).
Amended 2026-10-15 (owner decision): **metrics are on by default.** The reason
they were opt-in -- #90 had to show they cost the inference path nothing -- is
closed, and every server the owner runs carried them. `server.metrics`
defaults to `true` (ADR 0046 spells the flag `--server-metrics`, its env
var `IGNIS_SERVER_METRICS`, its file key `server.metrics`);
`--server-metrics false`, `IGNIS_SERVER_METRICS=false` or
`server.metrics: false` turns them off, and then neither the projection nor a
route to it exists, as below. `--server-metrics-bind` no longer needs
`--server-metrics` beside it; it is refused only beside an explicit `false`.
Everything else stands: the listener's default is still `127.0.0.1:9464`, it
asks for no key and is never exposed, and the Playground's `/ui/metrics` is
behind the API key when one is set -- the same key that guards `/v1` on the
same listener. Since the default address is a fixed port, a second server on
the host finds it taken, and refusing that start would make a default feature
a reason not to start: **a default address that cannot be bound is a warning**
(`ignis.server.metrics_unavailable`, with the address and the error), and the
server serves without the listener; the projection stays, so `/ui/metrics`
still answers. An address other than the default that the operator named --
`server.metrics_bind` from a flag, its env var or a file -- is theirs, and a
bind failure on it still refuses the start (`ignis.server.failed`).
`ignis.process.started` names the address the listener bound (the port a `:0`
bind was given), or `off`. The user stories below and "disabled by default" in
the decisions read with this amendment.
Amended again 2026-10-15 (#324, the image model's VRAM, spec image-model/05):
the image model's load reports what its plan reserved, into ADR 0030's
memory gauges rather than a family of its own: its weights as
`ignis_vram_reserved_bytes{line="weights"}`, its one arena as `workspace`, and
its condition budget as a new `line="condition_budget"` (always rendered, 0 on
a text load, as Flash-Next's two lines are on the 27B), inside
`ignis_vram_budget_bytes`; it has no KV pool, and no `cuda_context` or
`residual` line. One family joins them, `ignis_condition_budget_used_bytes`:
the bytes of the condition prefixes the image model's jobs hold, read against
the `condition_budget` line as `ignis_kv_pool_used_pages` is against
`ignis_kv_pool_pages`. Like the image job families it is recorded by the job,
not projected from a telemetry fact: the job sets it, with the runner held, as
it encodes its prefix and releases it, and every load's reservations reset it.
Unlike them it is **rendered from the first scrape, 0 on a text load** --
a memory gauge's zero is a reading -- which is why its name stays outside the
`ignis_image_` families that are absent until a job. The Monitor's *VRAM plan*
lays an image load out by its condition budget, and *In use now* reads the
prefixes against it.
Amended 2026-10-18 (#330, spec server/14): five families for the **loop
guard** (ADR 0052, 0053), rendered after the forced-close counter. No fact of
the guard reaches the telemetry consumer: the guard's own task records them,
off the model thread, into the same atomics. `ignis_loop_guard_loops_total`'s
`outcome` is the guard's log line it counts, one increment beside each
(`rewound`, `closed` for a `rewound` line that closed the reasoning,
`detected`, `too_little_kept`, `no_room`, `no_tokenizer`, `submit_refused`,
this last counted by the guard when its continuation's submit gives nothing,
which the ops log). They take the decision family's exception: **absent
until a request the guard watched ended or it saw a loop** (a server running
it `off` exports none of them), then every series renders, zeros included. A
loop counts as the guard sees it; its request counts once, as it ends and
before its `Done` reaches the client: watched, looped when it did, and its
lowest score, together, so a window never reads more looped requests than
watched ones. A rewind is
two engine requests: its cancel counts once in
`ignis_requests_cancelled_total` and its continuation once in
`ignis_requests_accepted_total`, which keep counting engine requests; the
Monitor's *Request flow* names the cancels that were rewinds, and a
*Reasoning* card reads the families with the forced-close counter. A guard's
close is prefilled, not forced, so it is never a forced close.
Amended 2026-10-18 (ADR 0058): `outcome="timed_out"` joins
`ignis_image_jobs_total`'s label set, a job its deadline ended between two
calls (`server.request_timeout`, or `server.stream_timeout` when it streams,
counted from its turn). Image jobs had no deadline before that ADR. No new
family: the job records it as it records the other five outcomes, its
`ignis.image.job` line says `outcome: timed_out`, and the Monitor's *Image
jobs* card reads it with them.
Amended 2026-10-31 (#222, owner decision): the prefix walk raises a miss
too, so `ignis_retained_state_misses_total{kind="prefix"}` counts on the
device and in KV-RAM instead of sitting at zero, and each kind misses only
where it can be hit on the load: no checkpoint miss with `--reuse-prompt off`,
no prefix miss without a retained slot. No new family; the rule is the note
below the table, and the Monitor's *Retained state* card shows the count where
it read "not measured".

### Context

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

### Decision

Ignis will expose a bounded Prometheus text endpoint when the operator supplies
`--metrics`. The endpoint will project facts already consumed asynchronously;
it will not introduce metric work or new fact traffic on the inference path.

The PRD sections around this Decision (Problem Statement, Solution, User
Stories, Implementation Decisions with the metric contract table and its
notes, Testing Decisions, Out of Scope, Further Notes) were cut on
2026-11-08; commit `107eb757` is the last that has them.

### Consequences

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

### Proposed amendment (2026-09-16) — the prefix-reuse counter widens (#188) — superseded

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
