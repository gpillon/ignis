//! v1 telemetry (server-02, GitHub #15, design §5): a **JSONL** sink — one
//! compact JSON object per line — carrying the scheduler's live counters:
//!
//! - the **interval** line — the live scheduler counters, one per scheduler
//!   step / driver tick:
//!   `{"kind":"interval","t":3,"waiting":2,"prefilling":1,"running":3,"kv_used_pct":62,"kv_evictions":0}`
//!
//! The per-request lifecycle events (`admitted` / `ttft` / `done`) used to be
//! a second JSONL record kind (`kind:"request"`) on this same sink; GitHub
//! #79 migrated them onto the canonical structured-logging system
//! (`ignis-logging`) as `ignis.request.admitted`/`ttft`/`done` tracing
//! events instead — request lifecycle observability lives in the one
//! canonical system now. This sink only carries the interval line (metrics,
//! not logs, GitHub #77 — untouched by that migration).
//!
//! The sink is injectable (tests capture lines in memory; production targets
//! stdout or a file) — since GitHub #108, through `ignis_logging::LineSink`
//! and its `StdoutSink`/`FileSink`/`MemorySink`/`NullSink` impls, rather
//! than a duplicate trait of this module's own (the two had carried
//! identical `fn write_line(&str)` shapes since #78; only the sink
//! plumbing is shared — the interval line's facts/counters/shape, and the
//! logging crate's own event model, stay independent, per ADR 0011/0017).
//! Since GitHub #69, all of this module's work (sink I/O, counter math)
//! runs on an async task off the model thread — the thread that owns the
//! `Scheduler` never calls into `Telemetry` at all, so a slow sink can
//! never add latency to a decode step, no matter how long a write takes or
//! how it is implemented.
//!
//! **Live counters (blocker for the coordinator).** The core [`Scheduler`]
//! trait — the public API the server drives (`Box<dyn Scheduler>`) — does not
//! expose the live counters (`waiting` / `prefilling` / `running` /
//! `kv_used_pct` / `kv_evictions`). `ConcreteScheduler` only exposes raw
//! pieces (`kv_used_pages`, `host_tier`, …), none of which are on the
//! `Scheduler` trait, so a trait object cannot reach them. This module
//! therefore fills the interval line from an injectable
//! [`IntervalStatsProvider`]; the default is an **event-derived** estimator
//! (it counts `running` / `waiting` / `kv_evictions` from the routed
//! [`SchedEvent`]s) and reports `prefilling` / `kv_used_pct` as 0 until core
//! exposes a `Scheduler::stats(&self)` accessor. That accessor is the missing
//! seam this module is built to close.

use std::collections::HashMap;
use std::sync::Arc;

use ignis_core::{FinishReason, LaneId, RequestId};
use ignis_logging::LineSink;
use serde::Serialize;
use serde_json;

use crate::api::finish_reason_str;

// ── the clock (the determinism seam) ────────────────────────────────────────

/// The time source for request-line `ms` / `tok_s`. Injectable so tests are
/// deterministic (ADR 0006: no wall-clock dependence in tests).
pub trait TelemetryClock: Send + Sync {
    /// The current time in milliseconds (an arbitrary monotonic epoch).
    fn now_ms(&self) -> u64;
}

/// The wall-clock source (production).
pub struct SystemClock;

impl TelemetryClock for SystemClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// A fixed (deterministic) clock: every read returns the same value.
pub struct FixedClock {
    ms: u64,
}

impl FixedClock {
    /// A clock that always reads `ms`.
    pub fn new(ms: u64) -> Self {
        Self { ms }
    }
}

impl TelemetryClock for FixedClock {
    fn now_ms(&self) -> u64 {
        self.ms
    }
}

// ── the interval-line counters + the live-source seam ──────────────────────

/// The live scheduler counters an interval line reports (design §5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct IntervalCounters {
    /// Queued (submitted, not yet dealt a lane).
    pub waiting: u32,
    /// Mid-prefill (KV warming) — 0 until core exposes it.
    pub prefilling: u32,
    /// On a decode lane.
    pub running: u32,
    /// Main-pool KV occupancy, percent — 0 until core exposes it.
    pub kv_used_pct: u32,
    /// Cumulative evictions to the host KV-RAM tier.
    pub kv_evictions: u64,
}

/// Supplies the live scheduler counters for the interval line.
///
/// **v1 blocker:** the core `Scheduler` trait does not expose these counters,
/// so the default is an event-derived estimator (built into [`Telemetry`]).
/// A concrete provider — wired once core adds a `Scheduler::stats(&self)`
/// accessor (or the server downcasts to `ConcreteScheduler`) — returns the
/// scheduler's live state instead of the estimator.
pub trait IntervalStatsProvider: Send + Sync {
    /// The counters to emit on the next interval line.
    fn counters(&self) -> IntervalCounters;
}

// ── the records (one compact JSON object per line) ──────────────────────────

/// The interval line (design §5): the live counters, one per scheduler step.
#[derive(Debug, Serialize)]
struct IntervalLine {
    /// The record kind (the JSONL discriminator; always `"interval"`).
    kind: &'static str,
    /// The tick number (a per-step counter; the §5 `t` field).
    t: u64,
    /// Queued requests (submitted, not yet dealt a lane).
    waiting: u32,
    /// Mid-prefill requests — 0 until core exposes it.
    prefilling: u32,
    /// Requests on a decode lane.
    running: u32,
    /// Main-pool KV occupancy, percent — 0 until core exposes it.
    kv_used_pct: u32,
    /// Cumulative evictions to the host tier.
    kv_evictions: u64,
}


// ── the telemetry state ─────────────────────────────────────────────────────

/// Per-request telemetry state (just enough for the request lines + the
/// event-derived interval counters).
#[derive(Debug, Default)]
struct RequestTelemetry {
    /// When the request was submitted (the `ms` timeline anchor).
    submitted_ms: u64,
    /// A lane has been dealt (the `admitted` line was emitted).
    admitted: bool,
    /// The first-token (`ttft`) line has been emitted.
    ttft: bool,
    /// The submitted prompt's token count (P3-06; from `RequestInput` at
    /// submit time, since a `Request`'s own history is not reachable from
    /// the telemetry consumer).
    prompt_tokens: u32,
    /// Completed `SchedEvent::PrefillChunk`s seen for this request (P3-06).
    prefill_chunks: u32,
    /// The most recent `prefilled_tokens` cumulative count (P3-06); equals
    /// `prompt_tokens` once prefill completes.
    prefilled_tokens: u32,
    /// The decode lane this request holds (P3-06; set at `admitted`, valid
    /// for `ttft`/`done` — a request never reaches those before admission).
    lane: LaneId,
    /// The clock reading at the last generated token (P3-06; the inter-
    /// token-latency anchor for the *next* token).
    last_token_ms: Option<u64>,
    /// Inter-token gaps recorded so far (P3-06): count, sum and max, enough
    /// for a mean without keeping every sample.
    itl_count: u64,
    itl_sum_ms: u64,
    itl_max_ms: u64,
}

/// The server's telemetry: tracks per-request state and emits the interval +
/// request JSONL lines through the injected sink.
pub struct Telemetry {
    sink: Arc<dyn LineSink>,
    clock: Arc<dyn TelemetryClock>,
    /// A live counter source (the §5 blocker seam); `None` → event-derived.
    stats: Option<Arc<dyn IntervalStatsProvider>>,
    /// The tick number (per-step counter; the §5 `t` field).
    tick: u64,
    /// Cumulative evictions to the host KV-RAM tier.
    kv_evictions: u64,
    /// In-flight request telemetry (id → state); removed on completion.
    requests: HashMap<RequestId, RequestTelemetry>,
}

impl Telemetry {
    /// Telemetry over `sink` (lines go here) and `clock` (`ms` / `tok_s`).
    /// No live counter source is set, so the interval line is event-derived.
    pub fn new(sink: Arc<dyn LineSink>, clock: Arc<dyn TelemetryClock>) -> Self {
        Self {
            sink,
            clock,
            stats: None,
            tick: 0,
            kv_evictions: 0,
            requests: HashMap::new(),
        }
    }

    /// Use `stats` as the live counter source (overrides the estimator).
    pub fn with_stats(&mut self, stats: std::sync::Arc<dyn IntervalStatsProvider>) {
        self.stats = Some(stats);
    }

    /// Point the sink at `sink` (the clock and any stats source are kept).
    pub fn set_sink(&mut self, sink: Arc<dyn LineSink>) {
        self.sink = sink;
    }

    /// A request was submitted: anchor its `ms` timeline and record its
    /// prompt length (P3-06's `prompt_tokens` field). A re-submit of an
    /// in-flight id keeps the original anchor (and prompt length), so `ms`
    /// is not reset.
    pub fn note_submit(&mut self, id: RequestId, prompt_tokens: u32) {
        self.requests
            .entry(id)
            .or_insert_with(|| RequestTelemetry {
                submitted_ms: self.clock.now_ms(),
                prompt_tokens,
                ..Default::default()
            });
    }

    /// A request was admitted (dealt `lane`): emit the `admitted` line,
    /// carrying the prefill phase's summary fields (P3-06: `prompt_tokens`,
    /// `prefill_chunks_consumed`, `prefilled_tokens`) accumulated from the
    /// `PrefillChunk` events already seen for this request.
    pub fn on_admitted(&mut self, id: RequestId, lane: LaneId) {
        let (submitted_ms, prompt_tokens, prefill_chunks, prefilled_tokens) = {
            let rt = self
                .requests
                .entry(id)
                .or_insert_with(|| RequestTelemetry {
                    submitted_ms: self.clock.now_ms(),
                    ..Default::default()
                });
            rt.admitted = true;
            rt.lane = lane;
            (rt.submitted_ms, rt.prompt_tokens, rt.prefill_chunks, rt.prefilled_tokens)
        };
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        self.emit_admitted(id, ms, lane, prompt_tokens, prefill_chunks, prefilled_tokens);
    }

    /// A chunked-prefill step landed for a request still queued or mid-
    /// prefill (P3-06, ADR 0018): accumulate the phase fields the
    /// `admitted` line reports once the request finishes prefill. Not
    /// itself logged — a request may sit `Prefilling` across many chunks,
    /// and per-chunk log lines would be per-round logging by another name.
    pub fn on_prefill_chunk(&mut self, id: RequestId, prefilled_tokens: u32) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.prefill_chunks = rt.prefill_chunks.saturating_add(1);
            rt.prefilled_tokens = prefilled_tokens;
        }
    }

    /// A token was generated: on the first one, emit the `ttft` line;
    /// subsequent tokens accumulate the inter-token-latency stats (P3-06)
    /// the `done` line reports.
    pub fn on_token(&mut self, id: RequestId) {
        let now = self.clock.now_ms();
        let first = {
            let rt = match self.requests.get_mut(&id) {
                Some(rt) => rt,
                None => return,
            };
            if !rt.ttft {
                rt.ttft = true;
                rt.last_token_ms = Some(now);
                Some((rt.submitted_ms, rt.lane))
            } else {
                if let Some(last) = rt.last_token_ms {
                    let gap = now.saturating_sub(last);
                    rt.itl_count = rt.itl_count.saturating_add(1);
                    rt.itl_sum_ms = rt.itl_sum_ms.saturating_add(gap);
                    rt.itl_max_ms = rt.itl_max_ms.max(gap);
                }
                rt.last_token_ms = Some(now);
                None
            }
        };
        if let Some((submitted_ms, lane)) = first {
            let ms = now.saturating_sub(submitted_ms);
            self.emit_ttft(id, ms, lane);
        }
    }

    /// A request completed (`n` = its total tokens, `reason` why it
    /// stopped): emit the `done` line — carrying `n`, `reason`, and the
    /// decode phase's per-lane inter-token-latency summary (P3-06) — and
    /// drop the request from the in-flight set.
    pub fn on_done(&mut self, id: RequestId, n: u32, reason: FinishReason) {
        let rt = self.requests.remove(&id);
        let (submitted_ms, lane, itl_count, itl_sum_ms, itl_max_ms) = match rt {
            Some(rt) => (
                rt.submitted_ms,
                rt.lane,
                rt.itl_count,
                rt.itl_sum_ms,
                rt.itl_max_ms,
            ),
            None => (self.clock.now_ms(), 0, 0, 0, 0),
        };
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        let itl_mean_ms = if itl_count == 0 {
            0.0
        } else {
            itl_sum_ms as f64 / itl_count as f64
        };
        self.emit_done(
            id,
            ms,
            n,
            throughput(n, ms),
            lane,
            reason,
            itl_mean_ms,
            itl_max_ms,
            itl_count,
        );
    }

    /// A request was evicted to the host tier: bump the eviction counter.
    pub fn on_evicted(&mut self, _id: RequestId) {
        self.kv_evictions = self.kv_evictions.saturating_add(1);
    }

    /// A request's host-tier snapshot was discarded (core-06): it goes back
    /// to `Admitted` and re-prefills from the start (`SchedEvent::Requeued`'s
    /// own doc comment). The prefill phase's summary fields (P3-06) reset
    /// with it — otherwise the eventual `admitted` line would sum chunks
    /// and tokens across two unrelated prefill attempts, which is exactly
    /// the misattribution this ticket's request log exists to prevent.
    pub fn on_requeued(&mut self, id: RequestId) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.prefill_chunks = 0;
            rt.prefilled_tokens = 0;
        }
    }

    /// The interval-line counters: the live provider's when one is set,
    /// otherwise the event-derived estimator (from the routed events).
    pub fn counters(&self) -> IntervalCounters {
        if let Some(stats) = &self.stats {
            return stats.counters();
        }
        let (mut waiting, mut running) = (0u32, 0u32);
        for rt in self.requests.values() {
            if rt.admitted {
                running = running.saturating_add(1);
            } else {
                waiting = waiting.saturating_add(1);
            }
        }
        IntervalCounters {
            waiting,
            // `prefilling` has no event (the prefill→decode transition is
            // not a `SchedEvent`) — 0 until core exposes it.
            prefilling: 0,
            running,
            // `kv_used_pct` needs the scheduler's KV pool state, which the
            // `Scheduler` trait does not expose — 0 until core exposes it.
            kv_used_pct: 0,
            kv_evictions: self.kv_evictions,
        }
    }

    /// Emit the interval line (called once per scheduler step / driver
    /// tick), returning the counters it computed (GitHub #69: the async
    /// telemetry consumer republishes this into the wait-free `ArcSwap`
    /// snapshot without recomputing it).
    pub fn emit_interval(&mut self) -> IntervalCounters {
        self.tick = self.tick.saturating_add(1);
        let counters = self.counters();
        let line = IntervalLine {
            kind: "interval",
            t: self.tick,
            waiting: counters.waiting,
            prefilling: counters.prefilling,
            running: counters.running,
            kv_used_pct: counters.kv_used_pct,
            kv_evictions: counters.kv_evictions,
        };
        self.sink.write_line(&to_line(&line));
        counters
    }

    /// Emit the `admitted` line: the prefill phase's summary (P3-06) —
    /// `prompt_tokens`, `prefill_chunks_consumed`, `prefilled_tokens` —
    /// alongside the lane dealt and how long the request queued. See
    /// [`Telemetry::emit_done`] for why this runs on the telemetry consumer
    /// task rather than inline.
    fn emit_admitted(
        &self,
        id: RequestId,
        ms: u64,
        lane: LaneId,
        prompt_tokens: u32,
        prefill_chunks_consumed: u32,
        prefilled_tokens: u32,
    ) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.admitted",
            request_id = id,
            duration_ms = ms,
            lane = lane as u64,
            prompt_tokens,
            prefill_chunks_consumed,
            prefilled_tokens,
            "request admitted"
        );
    }

    /// Emit the `ttft` line.
    fn emit_ttft(&self, id: RequestId, ms: u64, lane: LaneId) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.ttft",
            request_id = id,
            duration_ms = ms,
            lane = lane as u64,
            tokens = 1,
            tok_s = throughput(1, ms),
            "first token"
        );
    }

    /// Emit the `done` line: `tokens`/`tok_s` as before, plus the decode
    /// phase's per-phase fields this ticket adds (P3-06) — `finish_reason`
    /// (the same mapping `api.rs` uses for the HTTP response, so the two
    /// never disagree) and the per-lane inter-token-latency summary
    /// (`itl_ms_mean`/`itl_ms_max`/`itl_samples`) a failing G3 ITL cell is
    /// attributed against, without a second run, by reading this stream
    /// alongside the `admitted` lines of whatever else was prefilling in
    /// the same window (ADR 0011: the request log is diagnosis, never the
    /// gate's own oracle — that stays HTTP/SSE-side, P3-07).
    ///
    /// GitHub #81 / ADR 0012: this runs on the async telemetry consumer
    /// task (`engine.rs`'s `telemetry_task`), not inside the HTTP root span
    /// or any of `ignis-core`'s model-thread spans — a genuinely separate
    /// execution context, so it opens its own short span carrying
    /// `request_id` rather than relying on inherited span context that
    /// does not reach here. That gives these events the same `trace_id` as
    /// the rest of the request's lifecycle.
    #[allow(clippy::too_many_arguments)]
    fn emit_done(
        &self,
        id: RequestId,
        ms: u64,
        n: u32,
        tok_s: f64,
        lane: LaneId,
        reason: FinishReason,
        itl_ms_mean: f64,
        itl_ms_max: u64,
        itl_samples: u64,
    ) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.done",
            request_id = id,
            duration_ms = ms,
            lane = lane as u64,
            tokens = n,
            tok_s = tok_s,
            finish_reason = finish_reason_str(reason),
            itl_ms_mean,
            itl_ms_max,
            itl_samples,
            "request done"
        );
    }
}

/// Tokens per second for `n` tokens over `ms` milliseconds (0.0 for a
/// zero-span, so a deterministic fixed-clock test reports a stable 0.0).
fn throughput(n: u32, ms: u64) -> f64 {
    if ms == 0 {
        0.0
    } else {
        n as f64 * 1000.0 / ms as f64
    }
}

/// Serialize a record to one compact JSON object (a single JSONL line).
fn to_line(record: &impl Serialize) -> String {
    serde_json::to_string(record).expect("a telemetry record always serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ignis_logging::{FileSink, MemorySink};

    /// A provider that reports a fixed, known set of counters.
    struct FixedStats(IntervalCounters);
    impl IntervalStatsProvider for FixedStats {
        fn counters(&self) -> IntervalCounters {
            self.0
        }
    }

    fn telemetry() -> (Telemetry, Arc<MemorySink>) {
        let sink = Arc::new(MemorySink::new());
        let telemetry = Telemetry::new(sink.clone(), Arc::new(FixedClock::new(0)));
        (telemetry, sink)
    }

    #[test]
    fn an_interval_line_has_the_section5_shape() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(1, 3);
        telemetry.on_admitted(1, 0); // now an `ignis.request.admitted` tracing event, not a sink line
        telemetry.emit_interval(); // emits the interval line
        let lines = sink.lines();
        assert_eq!(lines.len(), 1, "only the interval line goes through `LineSink` now");
        let last: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(last["kind"], "interval");
        assert_eq!(last["t"], 1, "the first interval is tick 1");
        assert_eq!(last["running"], 1, "the admitted request is on a lane");
        assert_eq!(last["waiting"], 0, "no queued request");
        assert_eq!(last["prefilling"], 0);
        assert_eq!(last["kv_used_pct"], 0);
        assert_eq!(last["kv_evictions"], 0);
    }

    /// Captures the `ignis.request.*` events a block of code emits through
    /// `tracing`, via `ignis-logging`'s own `JsonLayer`/`MemorySink` (GitHub
    /// #79's canonical replacement for the old `RequestLine` JSONL shape).
    fn capture_request_events(f: impl FnOnce()) -> Vec<serde_json::Value> {
        use tracing_subscriber::layer::SubscriberExt;

        let log_sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber =
            tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(log_sink.clone()));
        tracing::subscriber::with_default(subscriber, f);
        log_sink
            .lines()
            .iter()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect()
    }

    #[test]
    fn request_events_carry_the_lifecycle_names_and_attributes() {
        let (mut telemetry, _sink) = telemetry();
        let events = capture_request_events(|| {
            telemetry.note_submit(7, 10);
            telemetry.on_admitted(7, 2);
            telemetry.on_token(7); // first token → ttft
            telemetry.on_token(7); // subsequent tokens are not re-emitted
            telemetry.on_done(7, 4, FinishReason::Stop);
        });

        let event_names: Vec<&str> = events.iter().map(|e| e["event_name"].as_str().unwrap()).collect();
        assert_eq!(
            event_names,
            vec!["ignis.request.admitted", "ignis.request.ttft", "ignis.request.done"]
        );
        for event in &events {
            assert_eq!(event["attributes"]["request_id"], 7);
            assert_eq!(event["attributes"]["lane"], 2, "every lifecycle line names the held lane");
        }
        let admitted = &events[0];
        assert_eq!(admitted["attributes"]["prompt_tokens"], 10);
        let done = events.last().unwrap();
        assert_eq!(done["attributes"]["tokens"], 4, "the done event carries the total tokens");
        assert_eq!(done["attributes"]["finish_reason"], "stop");
    }

    #[test]
    fn a_fixed_clock_keeps_request_events_deterministic() {
        let (mut telemetry, _sink) = telemetry();
        let events = capture_request_events(|| {
            telemetry.note_submit(1, 3);
            telemetry.on_admitted(1, 0);
            telemetry.on_done(1, 3, FinishReason::Stop);
        });
        let done = events.last().unwrap();
        // FixedClock(0): a zero elapsed span → `duration_ms` and `tok_s` are 0.
        assert_eq!(done["attributes"]["duration_ms"], 0);
        assert_eq!(done["attributes"]["tok_s"], 0.0);
    }

    #[test]
    fn a_step_clock_reports_the_elapsed_span_on_request_events() {
        let sink = Arc::new(MemorySink::new());
        // A clock that advances 100 ms per read: submit at 100, done at 200.
        // An `AtomicU32` (not a `Cell`) so the clock stays `Sync` for the
        // `TelemetryClock` bound.
        struct StepClock {
            reads: std::sync::atomic::AtomicU32,
        }
        impl TelemetryClock for StepClock {
            fn now_ms(&self) -> u64 {
                let n = self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                n as u64 * 100
            }
        }
        let mut telemetry =
            Telemetry::new(sink.clone(), Arc::new(StepClock {
                reads: std::sync::atomic::AtomicU32::new(0),
            }));
        let events = capture_request_events(|| {
            telemetry.note_submit(1, 5); // read #1 → 100 ms
            telemetry.on_done(1, 5, FinishReason::Length); // read #2 → 200 ms, so ms = 100
        });
        let done = events.last().unwrap();
        assert_eq!(done["attributes"]["duration_ms"], 100, "200 - 100 = 100 ms elapsed");
        assert_eq!(done["attributes"]["tok_s"], 50.0, "5 tokens / 0.1 s");
        assert_eq!(done["attributes"]["finish_reason"], "length");
    }

    #[test]
    fn event_derived_counters_split_waiting_and_running() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(1, 3); // queued (not yet admitted)
        telemetry.note_submit(2, 3); // queued
        telemetry.on_admitted(2, 0); // dealt a lane
        telemetry.emit_interval();
        let v: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        assert_eq!(v["waiting"], 1, "request 1 is still queued");
        assert_eq!(v["running"], 1, "request 2 is on a lane");
    }

    #[test]
    fn evictions_bump_the_counter() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(1, 3);
        telemetry.on_admitted(1, 0);
        telemetry.on_evicted(1);
        telemetry.on_evicted(1);
        telemetry.emit_interval();
        let v: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        assert_eq!(v["kv_evictions"], 2, "each eviction bumps the counter");
    }

    #[test]
    fn admitted_reports_the_prefill_phase_summary() {
        let (mut telemetry, _sink) = telemetry();
        let events = capture_request_events(|| {
            telemetry.note_submit(1, 300); // a 300-token prompt
            telemetry.on_prefill_chunk(1, 128); // chunk 1: 128/300
            telemetry.on_prefill_chunk(1, 256); // chunk 2: 256/300
            telemetry.on_prefill_chunk(1, 300); // chunk 3: prefill complete
            telemetry.on_admitted(1, 4);
        });
        let admitted = events.last().unwrap();
        assert_eq!(admitted["attributes"]["prompt_tokens"], 300);
        assert_eq!(admitted["attributes"]["prefill_chunks_consumed"], 3);
        assert_eq!(admitted["attributes"]["prefilled_tokens"], 300);
        assert_eq!(admitted["attributes"]["lane"], 4);
    }

    #[test]
    fn a_requeue_resets_the_prefill_phase_summary_for_the_re_prefill() {
        // core-06: a requeued request's host-tier snapshot was discarded, so
        // it re-prefills from the start (`SchedEvent::Requeued`'s own doc
        // comment). Without a reset, the eventual `admitted` line would sum
        // the discarded first attempt's chunks onto the second attempt's —
        // exactly the misattribution P3-06's request log exists to prevent.
        let (mut telemetry, _sink) = telemetry();
        let events = capture_request_events(|| {
            telemetry.note_submit(1, 300);
            telemetry.on_prefill_chunk(1, 128); // 1st attempt: 1 chunk, then evicted/discarded
            telemetry.on_requeued(1); // re-queued: the summary resets
            telemetry.on_prefill_chunk(1, 150); // 2nd attempt, chunk 1
            telemetry.on_prefill_chunk(1, 300); // 2nd attempt, chunk 2: complete
            telemetry.on_admitted(1, 4);
        });
        let admitted = events.last().unwrap();
        assert_eq!(
            admitted["attributes"]["prefill_chunks_consumed"], 2,
            "only the surviving (2nd) prefill attempt's chunks are counted"
        );
        assert_eq!(admitted["attributes"]["prefilled_tokens"], 300);
        // `prompt_tokens` is unaffected — it is the request's own submitted
        // length, not something a prefill attempt accumulates.
        assert_eq!(admitted["attributes"]["prompt_tokens"], 300);
    }

    #[test]
    fn done_reports_the_per_lane_inter_token_latency_summary() {
        let sink = Arc::new(MemorySink::new());
        // A clock that advances 10 ms per read, so every generated token is
        // exactly 10 ms apart — a deterministic, known ITL distribution.
        struct StepClock(std::sync::atomic::AtomicU32);
        impl TelemetryClock for StepClock {
            fn now_ms(&self) -> u64 {
                (self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as u64 + 1) * 10
            }
        }
        let mut telemetry = Telemetry::new(
            sink,
            Arc::new(StepClock(std::sync::atomic::AtomicU32::new(0))),
        );
        let events = capture_request_events(|| {
            telemetry.note_submit(1, 3);
            telemetry.on_admitted(1, 0);
            telemetry.on_token(1); // ttft — no ITL sample yet
            telemetry.on_token(1); // 1st gap: 10ms
            telemetry.on_token(1); // 2nd gap: 10ms
            telemetry.on_done(1, 3, FinishReason::Stop);
        });
        let done = events.last().unwrap();
        assert_eq!(done["attributes"]["itl_samples"], 2);
        assert_eq!(done["attributes"]["itl_ms_mean"], 10.0);
        assert_eq!(done["attributes"]["itl_ms_max"], 10);
    }

    /// P3-06's attribution acceptance criterion: a failing G3 ITL cell must
    /// be attributable from this stream alone, without another run. Here a
    /// long cold prefiller (request 2) lands a wide chunk while request 1's
    /// decode lane is waiting on its next token — request 1's `done` line
    /// shows the resulting ITL spike, and request 2's `admitted` line shows
    /// the prefill activity that caused it, correlated by wall-clock
    /// position in the one recorded stream (no re-run needed).
    #[test]
    fn a_failing_itl_cell_is_attributable_from_one_recorded_stream() {
        let sink = Arc::new(MemorySink::new());
        struct ScriptClock(std::sync::atomic::AtomicUsize, Vec<u64>);
        impl TelemetryClock for ScriptClock {
            fn now_ms(&self) -> u64 {
                let i = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.1[i]
            }
        }
        // Request 1 (decode lane): admitted at 0, tokens at 10/20/220 —
        // a 200ms gap where request 2's prefill chunk lands at 25.
        // Request 2 (cold prefiller): submitted at 5, one wide chunk
        // completing at 25 (the interleaved chunk that stalls lane 0),
        // admitted (starts decoding) at 230.
        let clock = ScriptClock(
            std::sync::atomic::AtomicUsize::new(0),
            vec![0, 5, 10, 20, 25, 220, 230, 220],
        );
        let mut telemetry = Telemetry::new(sink, Arc::new(clock));
        let events = capture_request_events(|| {
            telemetry.note_submit(1, 4); // t=0
            telemetry.on_admitted(1, 0); // t=5, lane 0
            telemetry.on_token(1); // t=10, ttft
            telemetry.on_token(1); // t=20, 1st gap 10ms
            telemetry.note_submit(2, 32_768); // t=25 (cold prefiller submitted)
            telemetry.on_prefill_chunk(2, 32_768); // the wide chunk lands
            telemetry.on_token(1); // t=220, 2nd gap: 200ms — the stall
            telemetry.on_admitted(2, 1); // t=230, prefiller starts decoding
            telemetry.on_done(1, 3, FinishReason::Stop); // t=220 (clock re-read), lane 0 finishes
        });

        // Request 2's `admitted` line shows the wide cold-prefill chunk
        // that ran while lane 0 was waiting on its next token.
        let prefiller_admitted = events
            .iter()
            .find(|e| {
                e["event_name"] == "ignis.request.admitted" && e["attributes"]["request_id"] == 2
            })
            .expect("request 2's admitted line is in the stream");
        assert_eq!(prefiller_admitted["attributes"]["prompt_tokens"], 32_768);
        assert_eq!(
            prefiller_admitted["attributes"]["prefill_chunks_consumed"],
            1
        );

        // Request 1's `done` line shows the ITL spike, attributable to that
        // same window purely by reading this one stream — no re-run needed.
        let lane0_done = events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.done" && e["attributes"]["request_id"] == 1)
            .expect("request 1's done line is in the stream");
        assert_eq!(
            lane0_done["attributes"]["itl_ms_max"], 200,
            "the stall shows up as the max gap"
        );
        assert_eq!(lane0_done["attributes"]["itl_samples"], 2);
    }

    #[test]
    fn a_live_provider_overrides_the_estimator() {
        let sink = Arc::new(MemorySink::new());
        let mut telemetry = Telemetry::new(
            sink.clone(),
            Arc::new(FixedClock::new(0)),
        );
        telemetry.with_stats(Arc::new(FixedStats(IntervalCounters {
            waiting: 3,
            prefilling: 2,
            running: 5,
            kv_used_pct: 62,
            kv_evictions: 9,
        })));
        telemetry.emit_interval();
        let v: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        // The provider's counters win over the (empty) event-derived set.
        assert_eq!(v["waiting"], 3);
        assert_eq!(v["prefilling"], 2);
        assert_eq!(v["running"], 5);
        assert_eq!(v["kv_used_pct"], 62);
        assert_eq!(v["kv_evictions"], 9);
    }

    #[test]
    fn the_memory_sink_records_lines_in_order() {
        let sink = Arc::new(MemorySink::new());
        sink.write_line("a");
        sink.write_line("b");
        assert_eq!(sink.lines(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn a_file_sink_appends_jsonl_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ignis-telemetry-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let sink = Arc::new(FileSink::open(&path).unwrap());
            let mut telemetry = Telemetry::new(sink.clone(), Arc::new(FixedClock::new(0)));
            telemetry.note_submit(1, 3);
            telemetry.on_admitted(1, 0);
            telemetry.emit_interval();
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // `on_admitted` now emits through `tracing`, not `LineSink` — only
        // the interval line lands in the file (GitHub #79).
        assert_eq!(lines.len(), 1, "the interval line");
        assert!(lines.iter().all(|l| l.starts_with('{') && l.ends_with('}')));
        let _ = std::fs::remove_file(&path);
    }
}