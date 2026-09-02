//! v1 telemetry (server-02, GitHub #15, design §5): a **JSONL** sink — one
//! compact JSON object per line — carrying two record kinds:
//!
//! - the **interval** line — the live scheduler counters, one per scheduler
//!   step / driver tick:
//!   `{"kind":"interval","t":3,"waiting":2,"prefilling":1,"running":3,"kv_used_pct":62,"kv_evictions":0}`
//! - the **request** line — one per request lifecycle event
//!   (`admitted` / `ttft` / `done`):
//!   `{"kind":"request","id":7,"event":"done","ms":210,"n":512,"tok_s":41.2}`
//!
//! The sink is injectable (tests capture lines in memory; production targets
//! stdout or a file) and stays cheap on the request path: a write is a
//! single buffered line into a lock-protected buffer, so it never blocks a
//! request and never inverts a lock with the engine's scheduler mutex.
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
use std::io::Write;
use std::sync::Arc;

use ignis_core::RequestId;
use serde::Serialize;
use serde_json;

// ── the JSONL sink ──────────────────────────────────────────────────────────

/// A telemetry sink: receives one compact JSON line per event.
///
/// Implementations must be cheap and non-blocking on the request path — a
/// lock-protected buffer (or a single buffered write) is acceptable; no long
/// holds and no I/O that can stall a request.
pub trait TelemetrySink: Send + Sync {
    /// Record one JSONL line (no trailing newline; the sink owns framing).
    fn write_line(&self, line: &str);
}

/// A no-op sink (the default when telemetry is not configured: the engine
/// still tracks request state, nothing is written).
pub struct NullSink;

impl TelemetrySink for NullSink {
    fn write_line(&self, _line: &str) {}
}

/// An in-memory sink: captures emitted lines so tests can assert on them.
#[derive(Default)]
pub struct MemorySink {
    lines: std::sync::Mutex<Vec<String>>,
}

impl MemorySink {
    /// An empty in-memory sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// A snapshot of the lines emitted so far, in order.
    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().unwrap().clone()
    }
}

impl TelemetrySink for MemorySink {
    fn write_line(&self, line: &str) {
        self.lines.lock().unwrap().push(line.to_string());
    }
}

/// A stdout sink (the §5 default target: one JSONL line per event).
///
/// `stdout` is line-buffered and a single write per event never meaningfully
/// blocks the request path.
pub struct StdoutSink;

impl TelemetrySink for StdoutSink {
    fn write_line(&self, line: &str) {
        let _ = writeln!(std::io::stdout(), "{line}");
    }
}

/// A file sink: appends one JSONL line per event to an opened (buffered)
/// file. The file is held in a `Mutex` so concurrent emits are serialized
/// into a single, short, non-inverting lock.
pub struct FileSink {
    file: std::sync::Mutex<std::fs::File>,
}

impl FileSink {
    /// Open (creating or truncating is the caller's choice) a sink at `path`.
    /// An existing file is appended to, matching a long-running server's
    /// telemetry log.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: std::sync::Mutex::new(file),
        })
    }
}

impl TelemetrySink for FileSink {
    fn write_line(&self, line: &str) {
        let mut file = self.file.lock().unwrap();
        let _ = writeln!(file, "{line}");
    }
}

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

/// The request line (design §5): one per lifecycle event.
#[derive(Debug, Serialize)]
struct RequestLine {
    /// The record kind (the JSONL discriminator; always `"request"`).
    kind: &'static str,
    /// The request's id (the scheduler's `RequestId`).
    id: RequestId,
    /// The lifecycle event (`"admitted"`, `"ttft"`, or `"done"`).
    event: &'static str,
    /// Milliseconds elapsed since the request was submitted.
    ms: u64,
    /// Tokens generated so far (0 at admitted, 1 at ttft, the total at done).
    n: u32,
    /// Throughput (tokens/s) at this event (0 until a done has a span).
    tok_s: f64,
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
}

/// The server's telemetry: tracks per-request state and emits the interval +
/// request JSONL lines through the injected sink.
pub struct Telemetry {
    sink: Arc<dyn TelemetrySink>,
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
    pub fn new(sink: Arc<dyn TelemetrySink>, clock: Arc<dyn TelemetryClock>) -> Self {
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
    pub fn set_sink(&mut self, sink: Arc<dyn TelemetrySink>) {
        self.sink = sink;
    }

    /// A request was submitted: anchor its `ms` timeline (a re-submit of an
    /// in-flight id keeps the original anchor, so `ms` is not reset).
    pub fn note_submit(&mut self, id: RequestId) {
        self.requests
            .entry(id)
            .or_insert_with(|| RequestTelemetry {
                submitted_ms: self.clock.now_ms(),
                ..Default::default()
            });
    }

    /// A request was admitted (dealt a lane): emit the `admitted` line.
    pub fn on_admitted(&mut self, id: RequestId) {
        let submitted_ms = {
            let rt = self
                .requests
                .entry(id)
                .or_insert_with(|| RequestTelemetry {
                    submitted_ms: self.clock.now_ms(),
                    ..Default::default()
                });
            rt.admitted = true;
            rt.submitted_ms
        };
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        self.emit_request(id, "admitted", ms, 0, 0.0);
    }

    /// A token was generated: on the first one, emit the `ttft` line.
    pub fn on_token(&mut self, id: RequestId) {
        let first = {
            let rt = self.requests.get_mut(&id);
            match rt {
                Some(rt) if !rt.ttft => {
                    rt.ttft = true;
                    Some(rt.submitted_ms)
                }
                _ => None,
            }
        };
        if let Some(submitted_ms) = first {
            let ms = self.clock.now_ms().saturating_sub(submitted_ms);
            self.emit_request(id, "ttft", ms, 1, throughput(1, ms));
        }
    }

    /// A request completed (`n` = its total tokens): emit the `done` line
    /// and drop the request from the in-flight set.
    pub fn on_done(&mut self, id: RequestId, n: u32) {
        let submitted_ms = self.requests.get(&id).map(|rt| rt.submitted_ms);
        self.requests.remove(&id);
        let submitted_ms = submitted_ms.unwrap_or_else(|| self.clock.now_ms());
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        self.emit_request(id, "done", ms, n, throughput(n, ms));
    }

    /// A request was evicted to the host tier: bump the eviction counter.
    pub fn on_evicted(&mut self, _id: RequestId) {
        self.kv_evictions = self.kv_evictions.saturating_add(1);
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

    /// Emit the interval line (called once per scheduler step / driver tick).
    pub fn emit_interval(&mut self) {
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
    }

    /// Emit a request line (one compact JSON object).
    fn emit_request(&mut self, id: RequestId, event: &'static str, ms: u64, n: u32, tok_s: f64) {
        let line = RequestLine {
            kind: "request",
            id,
            event,
            ms,
            n,
            tok_s,
        };
        self.sink.write_line(&to_line(&line));
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
        telemetry.note_submit(1);
        telemetry.on_admitted(1); // emits an `admitted` request line
        telemetry.emit_interval(); // emits the interval line
        let lines = sink.lines();
        assert_eq!(lines.len(), 2, "one request (admitted) + one interval line");
        let last: serde_json::Value =
            serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["kind"], "interval");
        assert_eq!(last["t"], 1, "the first interval is tick 1");
        assert_eq!(last["running"], 1, "the admitted request is on a lane");
        assert_eq!(last["waiting"], 0, "no queued request");
        assert_eq!(last["prefilling"], 0);
        assert_eq!(last["kv_used_pct"], 0);
        assert_eq!(last["kv_evictions"], 0);
    }

    #[test]
    fn request_lines_carry_the_lifecycle_events() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(7);
        telemetry.on_admitted(7);
        telemetry.on_token(7); // first token → ttft
        telemetry.on_token(7); // subsequent tokens are not re-emitted
        telemetry.on_done(7, 4);

        let lines = sink.lines();
        let kinds: Vec<String> = lines
            .iter()
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .unwrap()
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .unwrap()
                    .to_string()
            })
            .collect();
        // Every line is a `request` line (no interval line was emitted).
        assert!(kinds.iter().all(|k| *k == "request"));
        let events: Vec<String> = lines
            .iter()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event"]
                .as_str()
                .unwrap()
                .to_string())
            .collect();
        assert_eq!(events, vec!["admitted", "ttft", "done"]);
        let done: serde_json::Value =
            serde_json::from_str(&lines[lines.len() - 1]).unwrap();
        assert_eq!(done["id"], 7);
        assert_eq!(done["n"], 4, "the done line carries the total tokens");
    }

    #[test]
    fn a_fixed_clock_keeps_the_request_line_deterministic() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(1);
        telemetry.on_admitted(1);
        telemetry.on_done(1, 3);
        let done: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        // FixedClock(0): a zero elapsed span → `ms` and `tok_s` are 0.
        assert_eq!(done["ms"], 0);
        assert_eq!(done["tok_s"], 0.0);
    }

    #[test]
    fn a_step_clock_reports_the_elapsed_span() {
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
        telemetry.note_submit(1); // read #1 → 100 ms
        telemetry.on_done(1, 5); // read #2 → 200 ms, so ms = 100
        let done: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        assert_eq!(done["ms"], 100, "200 - 100 = 100 ms elapsed");
        assert_eq!(done["tok_s"], 50.0, "5 tokens / 0.1 s");
    }

    #[test]
    fn event_derived_counters_split_waiting_and_running() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(1); // queued (not yet admitted)
        telemetry.note_submit(2); // queued
        telemetry.on_admitted(2); // dealt a lane
        telemetry.emit_interval();
        let v: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        assert_eq!(v["waiting"], 1, "request 1 is still queued");
        assert_eq!(v["running"], 1, "request 2 is on a lane");
    }

    #[test]
    fn evictions_bump_the_counter() {
        let (mut telemetry, sink) = telemetry();
        telemetry.note_submit(1);
        telemetry.on_admitted(1);
        telemetry.on_evicted(1);
        telemetry.on_evicted(1);
        telemetry.emit_interval();
        let v: serde_json::Value = serde_json::from_str(sink.lines().last().unwrap()).unwrap();
        assert_eq!(v["kv_evictions"], 2, "each eviction bumps the counter");
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
            telemetry.note_submit(1);
            telemetry.on_admitted(1);
            telemetry.emit_interval();
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 2, "at least one request + one interval line");
        assert!(lines.iter().all(|l| l.starts_with('{') && l.ends_with('}')));
        let _ = std::fs::remove_file(&path);
    }
}