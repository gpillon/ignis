//! v1 telemetry (server-02, GitHub #15, design §5): the server's
//! request-lifecycle and scheduler-counter observability, emitted as
//! canonical `ignis-logging` events — it has no output format of its own:
//!
//! - the request lifecycle — `ignis.request.admitted`/`ttft`/`done`/
//!   `evicted`/`restored` (GitHub #79, #125);
//! - the scheduler counters — `ignis.scheduler.interval` at DEBUG, emitted
//!   when `waiting` / `running` / `kv_evictions` change (ADR 0025). Until
//!   ADR 0025 these were a `{"kind":"interval",...}` JSONL line written
//!   straight to stdout on every step, past the operator's log format.
//!
//! Since GitHub #69, all of this module's work (counter math, event
//! emission) runs on an async task off the model thread — the thread that
//! owns the `Scheduler` never calls into `Telemetry` at all.
//!
//! **Live counters (blocker for the coordinator).** The core [`Scheduler`]
//! trait — the public API the server drives (`Box<dyn Scheduler>`) — does not
//! expose the live counters (`waiting` / `prefilling` / `running` /
//! `kv_used_pct` / `kv_evictions`). `ConcreteScheduler` only exposes raw
//! pieces (`kv_used_pages`, `host_tier`, …), none of which are on the
//! `Scheduler` trait, so a trait object cannot reach them. This module
//! therefore reads the counters from an injectable
//! [`IntervalStatsProvider`]; the default is an **event-derived** estimator
//! (it counts `running` / `waiting` / `kv_evictions` from the routed
//! [`SchedEvent`]s) and reports `prefilling` as 0 until core exposes it —
//! which is why that one never reaches the interval event.
//!
//! `kv_used_pct` is no longer among them (GitHub #216, ADR 0030). The seam
//! this module was built to close is closed from the other side: rather than
//! an accessor a consumer on another thread calls into the live scheduler,
//! the model thread reads its own occupancy after each step and carries it on
//! the tick it already sends. So the counter is authoritative without anyone
//! reaching across a thread for it, and `Scheduler::occupancy` is a plain
//! field read on the side that owns the scheduler.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use ignis_core::checkpoint::{RetainedKind, RetainedStateOperation, ReuseSource};
use ignis_core::scheduler::{DiskOp, DiskSource};
use ignis_core::flash_next_counters::FlashNextCounterSource;
use ignis_core::thinking_budget::BudgetOutcome;
use ignis_core::{
    FinishReason, LaneId, Occupancy, RequestClass, RequestId, RetainedSkip, SpecCounters,
};
use serde::Serialize;

use crate::api::finish_reason_str;
use crate::media::MediaStats;
use crate::metrics::Metrics;

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

/// A clock its owner sets (deterministic tests): every read returns what
/// was last set.
pub struct ManualClock(std::sync::atomic::AtomicU64);

impl ManualClock {
    /// A clock reading `ms` until [`ManualClock::set`] moves it.
    pub fn at(ms: u64) -> Arc<Self> {
        Arc::new(Self(std::sync::atomic::AtomicU64::new(ms)))
    }

    /// Read `ms` from now on.
    pub fn set(&self, ms: u64) {
        self.0.store(ms, std::sync::atomic::Ordering::SeqCst);
    }
}

impl TelemetryClock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
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
    /// Main-pool KV occupancy, percent of the pool's pages (GitHub #216):
    /// the scheduler's own reading, carried on the tick. A convenience for
    /// the one log line — the two terms it divides are exported separately,
    /// as pages, so no reader has to take a ratio's word for which moved.
    pub kv_used_pct: u32,
    /// Cumulative evictions to the host KV-RAM tier.
    pub kv_evictions: u64,
}

/// The main pool's occupancy as a whole percent of its pages (GitHub #216).
///
/// A pool of no pages is 0% rather than undefined: the mock load has none, and
/// a division there would be the only way this projection could panic.
fn kv_used_pct(occupancy: Occupancy) -> u32 {
    if occupancy.kv_pool_pages == 0 {
        return 0;
    }
    let used = u64::from(occupancy.kv_used_pages) * 100;
    (used / u64::from(occupancy.kv_pool_pages)).min(100) as u32
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

// ── the telemetry state ─────────────────────────────────────────────────────

/// What a request's media cost, as the `admitted` and `done` lines report it
/// (GitHub #245): what acquiring them cost (GitHub #179) and what encoding
/// them cost during prefill (GitHub #192), which the two halves of the
/// pipeline measure in different places and only meet here.
#[derive(Debug, Clone, Copy)]
struct MediaSummary {
    stats: MediaStats,
    encode_micros: u64,
}

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
    /// The request's Lane tag (GitHub #120, `GLOSSARY.md`), read off
    /// `submit`'s own argument at submit time — same reason `prompt_tokens`
    /// is captured here rather than read back off the scheduler's `Request`.
    class: RequestClass,
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
    /// What acquiring the request's media cost (GitHub #179); `None` for a
    /// request without media.
    media: Option<MediaStats>,
    /// Microseconds this request's prefill chunks spent encoding media
    /// (GitHub #192), summed as the chunks land.
    encode_micros: u64,
    /// The retained state this request's prefill resumed from (GitHub #186,
    /// ADR 0029): the residency tier, the prompt tokens it skipped, and what
    /// the restore itself cost. `None` for a request that reused nothing,
    /// which is how the request log says `reuse_source: none` — by saying
    /// nothing at all, exactly as it does for `spec.*` on a load with no
    /// drafter.
    reuse: Option<RequestReuse>,
    /// `reasoning_effort: "max"` dropped a thinking budget this request
    /// would otherwise have run under (spec server/08), read off the
    /// submission since the scheduler never sees it.
    thinking_budget_dropped: bool,
}

/// The retained state a request's prefill resumed from, as its `done` line
/// reports it (GitHub #186, #325).
#[derive(Debug, Clone, Copy)]
struct RequestReuse {
    source: ReuseSource,
    tokens: u32,
    /// What the restore cost; `None` for a prefix, which is claimed by
    /// sharing pages and restores nothing.
    restore_micros: Option<u64>,
    kind: RetainedKind,
}

/// Why a request was cancelled: the `cause` of its `ignis.request.cancelled`
/// line (GitHub #327).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelCause {
    /// What waited for it let it go: a client that hung up, a queued
    /// response cancelled, a loop-guard rewind (whose
    /// `ignis.loop_guard.rewound` line names it `cancelled`).
    Consumer,
    /// Past its deadline (`--server-request-timeout`,
    /// `--server-stream-timeout`, ADR 0058), whatever answered it: the answer
    /// so far cut `length` or `incomplete`, a 504, a response `failed` with
    /// `request_timeout`, a stream's error chunk, a decision question
    /// `not_completed`, an image prompt rewrite given up.
    Timeout,
    /// The model thread stopped with it in flight: a model switch past its
    /// drain window.
    Shutdown,
}

impl CancelCause {
    /// The `cause` field's value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Consumer => "consumer",
            Self::Timeout => "timeout",
            Self::Shutdown => "shutdown",
        }
    }
}

/// The server's telemetry: tracks per-request state and emits the interval +
/// request events.
pub struct Telemetry {
    clock: Arc<dyn TelemetryClock>,
    /// A live counter source (the §5 blocker seam); `None` → event-derived.
    stats: Option<Arc<dyn IntervalStatsProvider>>,
    /// The tick number (per-step counter; the interval event's `tick`).
    tick: u64,
    /// The `(waiting, running, kv_used_pct, kv_evictions)` the last interval
    /// event carried; `None` before the first tick (ADR 0025: unchanged
    /// counters are not logged again). `kv_used_pct` is in the tuple because
    /// it is in the line: left out, the line would report a percentage frozen
    /// at the last time some *other* counter moved (GitHub #216).
    last_logged: Option<(u32, u32, u32, u64)>,
    /// What the scheduler held at the last tick (GitHub #216, ADR 0030).
    occupancy: Occupancy,
    /// Cumulative evictions to the host KV-RAM tier.
    kv_evictions: u64,
    /// In-flight request telemetry (id → state); removed on completion.
    requests: HashMap<RequestId, RequestTelemetry>,
    /// The Prometheus projection this consumer keeps up to date, when
    /// `--server-metrics` installed one (GitHub #89, ADR 0017).
    metrics: Option<Arc<Metrics>>,
    /// Where the load's own counters are read, a Flash-Next load's (GitHub
    /// #301, #302): host memory, read at every tick into `metrics`.
    counter_source: Option<Arc<FlashNextCounterSource>>,
    /// The latest cancelled (or stopped, [`Telemetry::on_stopped`])
    /// requests, kept only while `metrics` is set: a request can finish in
    /// the same step its cancel is sent, and then its `Done` reaches this
    /// consumer after the cancel. It was counted once, as cancelled or as
    /// completed, and must not be counted again as completed.
    recently_cancelled: VecDeque<RequestId>,
    /// How long a request stays in flight, submit to release (GitHub #327):
    /// what [`Telemetry::lane_free_at_ms`] adds to the oldest request in
    /// flight, since a lane's place in `max_in_flight` frees only when its
    /// request leaves. `None` until one has left. A loop-guard rewind cancels
    /// a request and submits its continuation (ADR 0052), which this sees as
    /// two shorter stays: the estimate then comes early, which costs a client
    /// one more refusal at most. Past the estimate a lane is expected one
    /// hold from the refusal, at most 15 s, so it comes late by at most that.
    lane_held_ms: Option<u64>,
}

/// How many cancelled ids [`Telemetry`] remembers for that race. A late
/// `Done` was emitted before the model thread handled the cancel — within one
/// scheduler step — so this bound is far beyond any real window.
const RECENTLY_CANCELLED: usize = 1024;

/// The expected lane hold moves 1/8 of the way toward each request that
/// leaves (GitHub #327): the weight TCP smooths its round-trip time with
/// (RFC 6298). A constant: it only shapes an estimate.
const LANE_HOLD_SMOOTHING: u64 = 8;

impl Telemetry {
    /// Telemetry over `clock` (`ms` / `tok_s`). No live counter source is
    /// set, so the interval counters are event-derived.
    pub fn new(clock: Arc<dyn TelemetryClock>) -> Self {
        Self {
            clock,
            stats: None,
            tick: 0,
            last_logged: None,
            occupancy: Occupancy::default(),
            kv_evictions: 0,
            requests: HashMap::new(),
            metrics: None,
            counter_source: None,
            recently_cancelled: VecDeque::new(),
            lane_held_ms: None,
        }
    }

    /// Use `stats` as the live counter source (overrides the estimator).
    pub fn with_stats(&mut self, stats: std::sync::Arc<dyn IntervalStatsProvider>) {
        self.stats = Some(stats);
    }

    /// Keep `metrics` up to date from the same calls that emit the request
    /// and interval events — never from their rendered output.
    pub fn with_metrics(&mut self, metrics: Arc<Metrics>) {
        self.metrics = Some(metrics);
        self.project_counter_source();
    }

    /// A request was submitted: anchor its `ms` timeline and record its
    /// prompt length (P3-06's `prompt_tokens` field) and admission class
    /// (GitHub #120). A re-submit of an in-flight id keeps the original
    /// anchor (and prompt length / class), so `ms` is not reset.
    pub fn note_submit(&mut self, id: RequestId, prompt_tokens: u32, class: RequestClass) {
        if let Some(metrics) = self.metrics.as_ref().filter(|_| !self.requests.contains_key(&id)) {
            metrics.record_accepted();
        }
        self.requests
            .entry(id)
            .or_insert_with(|| RequestTelemetry {
                submitted_ms: self.clock.now_ms(),
                prompt_tokens,
                class,
                ..Default::default()
            });
    }

    /// When a lane is expected to free, on this telemetry's clock (GitHub
    /// #327): what a `503 engine_full` turns into its `Retry-After`. `None`
    /// with nothing to go on.
    pub fn lane_free_at_ms(&self) -> Option<u64> {
        let held = self.lane_held_ms?;
        let oldest = self.requests.values().map(|rt| rt.submitted_ms).min()?;
        Some(oldest + held)
    }

    /// How long requests have held a lane (GitHub #327): once the oldest
    /// request in flight is past [`Telemetry::lane_free_at_ms`], a lane is
    /// expected this long after a refusal asks. `None` until one has left.
    pub fn lane_held_ms(&self) -> Option<u64> {
        self.lane_held_ms
    }

    /// A request held its lane `ms` before it left: the expected hold moves
    /// [`LANE_HOLD_SMOOTHING`]th of the way toward it.
    fn note_lane_held(&mut self, ms: u64) {
        self.lane_held_ms = Some(match self.lane_held_ms {
            None => ms,
            Some(held) => (held * (LANE_HOLD_SMOOTHING - 1) + ms) / LANE_HOLD_SMOOTHING,
        });
    }

    /// A submitted request carried media (GitHub #179): its acquisition
    /// summary rides on the `admitted` line and on the `done` line (GitHub
    /// #245), which a request that never takes a lane reaches too.
    pub fn note_media(&mut self, id: RequestId, media: MediaStats) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.media = Some(media);
        }
    }

    /// `max` dropped a submitted request's thinking budget (spec server/08):
    /// its `done` line says so, or a request that set a budget and was never
    /// forced would carry no explanation.
    pub fn note_thinking_budget_dropped(&mut self, id: RequestId) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.thinking_budget_dropped = true;
        }
    }

    /// A request was admitted (dealt `lane`): emit the `admitted` line,
    /// carrying the prefill phase's summary fields (P3-06: `prompt_tokens`,
    /// `prefill_chunks_consumed`, `prefilled_tokens`) accumulated from the
    /// `PrefillChunk` events already seen for this request.
    pub fn on_admitted(&mut self, id: RequestId, lane: LaneId) {
        let (submitted_ms, prompt_tokens, prefill_chunks, prefilled_tokens, class, media) =
            match self.requests.get_mut(&id) {
                Some(rt) => {
                    rt.admitted = true;
                    rt.lane = lane;
                    (
                        rt.submitted_ms,
                        rt.prompt_tokens,
                        rt.prefill_chunks,
                        rt.prefilled_tokens,
                        rt.class,
                        rt.media.map(|stats| MediaSummary { stats, encode_micros: rt.encode_micros }),
                    )
                }
                // Not in flight — typically cancelled before this admission
                // reached the consumer (GitHub #89). Still logged, but not
                // re-added: no `Done` would ever remove it again, and it
                // would count as running forever.
                None => (self.clock.now_ms(), 0, 0, 0, RequestClass::default(), None),
            };
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        self.emit_admitted(id, ms, lane, prompt_tokens, prefill_chunks, prefilled_tokens, class, media);
    }

    /// A request was cancelled, for `cause` (GitHub #89): emit its
    /// `cancelled` line (GitHub #327). The scheduler releases a cancelled
    /// request without a `Done`, so it leaves the in-flight set here, or it
    /// would count as waiting or running forever. Events the model thread
    /// emitted for it before the cancel may still arrive afterwards; none of
    /// them re-adds it, and a late `Done` does not count it a second time, as
    /// completed. A second cancel, or one after its `Done`, finds nothing.
    pub fn on_cancelled(&mut self, id: RequestId, cause: CancelCause) {
        let Some(rt) = self.requests.remove(&id) else {
            return;
        };
        let ms = self.clock.now_ms().saturating_sub(rt.submitted_ms);
        self.emit_cancelled(id, cause, ms, &rt);
        self.note_lane_held(ms);
        let Some(metrics) = self.metrics.clone() else {
            return;
        };
        metrics.record_cancelled();
        self.remember_released(id);
        let counters = self.counters();
        metrics.set_scheduler_requests(counters.waiting, counters.running);
    }

    /// A request ended at its caller's stop sequence after `n` generated
    /// tokens (spec server/09, [`crate::engine::Engine::stop`]). The
    /// scheduler released it as it releases a cancel, with no `Done`, but it
    /// finished: its `done` line is emitted here, with `stop`, and it counts
    /// as completed. A `Done` that raced the release is not counted again.
    pub fn on_stopped(&mut self, id: RequestId, n: u32) {
        if !self.requests.contains_key(&id) {
            // Its own `Done` came first: already reported.
            return;
        }
        self.on_done(id, n, FinishReason::Stop, None, None, None);
        if self.metrics.is_some() {
            self.remember_released(id);
        }
    }

    /// Remember `id` for the cancel/`Done` race (`recently_cancelled`).
    fn remember_released(&mut self, id: RequestId) {
        if self.recently_cancelled.len() == RECENTLY_CANCELLED {
            self.recently_cancelled.pop_front();
        }
        self.recently_cancelled.push_back(id);
    }

    /// A chunked-prefill step landed for a request still queued or mid-
    /// prefill (P3-06, ADR 0018): accumulate the phase fields the
    /// `admitted` line reports once the request finishes prefill. Not
    /// itself logged — a request may sit `Prefilling` across many chunks,
    /// and per-chunk log lines would be per-round logging by another name.
    pub fn on_prefill_chunk(&mut self, id: RequestId, prefilled_tokens: u32, encode_micros: u64) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.prefill_chunks = rt.prefill_chunks.saturating_add(1);
            rt.prefilled_tokens = prefilled_tokens;
            rt.encode_micros = rt.encode_micros.saturating_add(encode_micros);
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
            if let Some(metrics) = &self.metrics {
                metrics.record_decoded_token();
            }
            if !rt.ttft {
                rt.ttft = true;
                rt.last_token_ms = Some(now);
                Some((rt.submitted_ms, rt.lane, rt.class))
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
        if let Some((submitted_ms, lane, class)) = first {
            let ms = now.saturating_sub(submitted_ms);
            if let Some(metrics) = &self.metrics {
                metrics.observe_ttft_ms(ms);
            }
            self.emit_ttft(id, ms, lane, class);
        }
    }

    /// A request completed (`n` = its total tokens, `reason` why it
    /// stopped): emit the `done` line — carrying `n`, `reason`, the decode
    /// phase's per-lane inter-token-latency summary (P3-06), its
    /// speculative counters when it ran any rounds (P5-06, GitHub #154),
    /// what its thinking budget did when it had one (spec server/08), where
    /// its EOS became `</think>` when it did (GitHub #315) and what its media
    /// cost when it had any (GitHub #245) — and drop the request from the
    /// in-flight set.
    pub fn on_done(
        &mut self,
        id: RequestId,
        n: u32,
        reason: FinishReason,
        spec: Option<SpecCounters>,
        thinking: Option<BudgetOutcome>,
        reasoning_redirected_at: Option<u32>,
    ) {
        if let Some(metrics) = &self.metrics {
            if let Some(spec) = spec.as_ref() {
                metrics.record_speculation(spec);
            }
            match self.recently_cancelled.iter().position(|&c| c == id) {
                // Already counted as cancelled (see `recently_cancelled`).
                Some(i) => {
                    self.recently_cancelled.remove(i);
                }
                None => {
                    metrics.record_completed(n);
                    if thinking.is_some_and(|t| t.forced_at.is_some()) {
                        metrics.record_thinking_forced_close();
                    }
                }
            }
        }
        let rt = self.requests.remove(&id);
        let known = rt.is_some();
        let (submitted_ms, lane, itl_count, itl_sum_ms, itl_max_ms, class, reuse, budget_dropped, media) = match rt {
            Some(rt) => (
                rt.submitted_ms,
                rt.lane,
                rt.itl_count,
                rt.itl_sum_ms,
                rt.itl_max_ms,
                rt.class,
                rt.reuse,
                rt.thinking_budget_dropped,
                rt.media.map(|stats| MediaSummary { stats, encode_micros: rt.encode_micros }),
            ),
            None => (
                self.clock.now_ms(),
                0,
                0,
                0,
                0,
                RequestClass::default(),
                None,
                false,
                None,
            ),
        };
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        // Only a request this consumer saw submitted has a real span; one
        // already counted as cancelled is not observed as completed.
        if known {
            self.note_lane_held(ms);
            if let Some(metrics) = &self.metrics {
                metrics.observe_duration_ms(ms);
            }
        }
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
            class,
            spec,
            reuse,
            ThinkingReport {
                outcome: thinking,
                dropped_by_max: budget_dropped,
                redirected_at: reasoning_redirected_at,
            },
            media,
        );
    }

    /// A request was evicted to the host tier: bump the eviction counter
    /// and emit the `evicted` line (GitHub #125) — the snapshot's wall time
    /// (`snapshot_micros`; a windowed move's span, GitHub #309), so the
    /// tier's cost is attributable from the request log alone, without a
    /// second run.
    pub fn on_evicted(&mut self, id: RequestId, snapshot_micros: u64) {
        self.kv_evictions = self.kv_evictions.saturating_add(1);
        if let Some(metrics) = &self.metrics {
            metrics.record_eviction();
        }
        self.emit_evicted(id, snapshot_micros);
    }

    /// A request's prefill skipped `tokens` prompt tokens through a cached
    /// prefix (core-07).
    ///
    /// Two kinds arrive here since GitHub #188: a concurrent sibling's prefix,
    /// and a **retained prefix** left by a request that has already finished.
    /// `retained` tells them apart (#190), so `ignis_prefix_reused_tokens_total`
    /// keeps the *sibling*-prefix meaning ADR 0017's row gives it, and
    /// cross-request reuse is counted as reuse of retained state, on the tier
    /// the prefix came from: the same tier its hit and its `done` line name
    /// (#325).
    pub fn on_prefix_reused(&mut self, id: RequestId, tokens: u32, retained: bool, source: ReuseSource) {
        if let Some(metrics) = &self.metrics {
            if retained {
                // A `PrefixReused` fact is a prefix by its own name, so the
                // kind is read off the event and not off the tier (#216).
                metrics.record_retained_reused(source, RetainedKind::Prefix, tokens);
            } else {
                metrics.record_prefix_reused(tokens);
            }
        }
        // GitHub #325: a retained prefix is per-request reuse too, so the
        // request's `done` line says so. A live sibling's prefix is not
        // retained state and reports nothing, as before.
        if retained {
            if let Some(rt) = self.requests.get_mut(&id) {
                rt.reuse = Some(RequestReuse {
                    source,
                    tokens,
                    restore_micros: None,
                    kind: RetainedKind::Prefix,
                });
            }
        }
    }

    /// A request's prefill resumed from retained state left by an earlier,
    /// already-finished request (GitHub #186, ADR 0029). Unlike a sibling
    /// prefix this is a *per-request* fact — which tier served it, how much
    /// prefill it skipped, what the restore cost — so it is stashed and
    /// reported on the request's own `done` line rather than only summed
    /// into a server-wide counter. The skipped tokens are also summed per
    /// tier into `ignis_retained_reused_tokens_total` (#190) — never into
    /// `ignis_prefix_reused_tokens_total`, which counts sibling-prefix reuse.
    pub fn on_state_reused(
        &mut self,
        id: RequestId,
        source: ReuseSource,
        kind: RetainedKind,
        tokens: u32,
        restore_micros: u64,
    ) {
        if let Some(metrics) = &self.metrics {
            metrics.record_retained_reused(source, kind, tokens);
        }
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.reuse = Some(RequestReuse {
                source,
                tokens,
                restore_micros: Some(restore_micros),
                kind,
            });
        }
    }

    /// Something happened to retained state in one tier (GitHub #190). Not
    /// logged: none of it belongs to one request's line, so only the
    /// installed projection observes it.
    pub fn on_retained_state(
        &mut self,
        operation: RetainedStateOperation,
        source: ReuseSource,
        kind: RetainedKind,
    ) {
        if let Some(metrics) = &self.metrics {
            metrics.record_retained_state(operation, source, kind);
        }
    }

    /// The scheduler held a different number of retained slots after a step
    /// (GitHub #215): emit the change-driven `ignis.scheduler.retained_slots`
    /// DEBUG event, as the interval event reports its own counters (ADR 0025).
    pub fn on_retained_slots(&mut self, in_use: u32, capacity: u32) {
        if let Some(metrics) = &self.metrics {
            metrics.set_retained_slots_in_use(in_use);
        }
        tracing::debug!(
            name: "ignis.scheduler.retained_slots",
            retained_slots_in_use = in_use,
            retained_slots = capacity,
            "retained slots held"
        );
    }

    /// A publish or a capture `id` would have left was skipped (GitHub #215):
    /// emit the `retained_slot_skipped` line, naming why and how many of the
    /// `capacity` retained slots were held then. The request itself runs
    /// regardless.
    pub fn on_retained_slot_skipped(&mut self, id: RequestId, skip: RetainedSkip, in_use: u32, capacity: u32) {
        if let Some(metrics) = &self.metrics {
            metrics.record_retained_slot_skip(skip);
        }
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.retained_slot_skipped",
            request_id = id,
            reason = skip.as_str(),
            retained_slots_in_use = in_use,
            retained_slots = capacity,
            "retained state skipped: no retained slot or page to hold it"
        );
    }

    /// A request was restored from the host tier onto a decode lane: emit
    /// the `restored` line (GitHub #125) — the restore's wall time
    /// (`restore_micros`; a windowed move's span, GitHub #309), the other
    /// half of the tier's own cost attribution alongside
    /// [`Telemetry::on_evicted`].
    pub fn on_restored(&mut self, id: RequestId, restore_micros: u64) {
        self.emit_restored(id, restore_micros);
    }

    /// A request's host-tier snapshot was discarded (core-06): it goes back
    /// to `Admitted` and re-prefills from the start (`SchedEvent::Requeued`'s
    /// own doc comment). The prefill phase's summary fields (P3-06) reset
    /// with it — otherwise the eventual `admitted` line would sum chunks
    /// and tokens across two unrelated prefill attempts, which is exactly
    /// the misattribution this ticket's request log exists to prevent.
    /// A **live** snapshot was dropped out of the KV-RAM tier to make room
    /// (GitHub #224): the tier's own eviction, projected into
    /// `ignis_kv_ram_evictions_total` and emitted as the `dropped` line.
    ///
    /// The line exists for the same reason `evicted` and `restored` do
    /// (GitHub #125): the tier's cost has to be attributable from the request
    /// log alone. `Requeued` carries no line of its own, so without this one a
    /// dropped request shows up as an `evicted` followed by a second
    /// `admitted` with nothing in between saying why it started over.
    ///
    /// Deliberately *not* driven off [`Telemetry::on_requeued`]: a requeue
    /// also follows a failed restore, where KV-RAM evicted nothing.
    ///
    /// The scheduler's interval line (ADR 0025) is deliberately left alone:
    /// it carries `kv_evictions` and gains no second field here, so this
    /// ticket changes the Prometheus contract and the request log, not the
    /// interval contract.
    pub fn on_snapshot_dropped(&mut self, id: RequestId) {
        if let Some(metrics) = &self.metrics {
            metrics.record_kv_ram_eviction();
        }
        self.emit_dropped(id);
    }

    /// A live snapshot was written to KV-disk (spec vram-budget/03): straight
    /// from the device, or demoted from KV-RAM. Counted in
    /// `ignis_kv_disk_spills_total{from}` -- never in the two KV-RAM eviction
    /// counters, whose meanings stay theirs -- and logged as the `spilled`
    /// line, so a request's trip through the tiers reads from the request log
    /// alone, as its `evicted` and `restored` lines already make it.
    pub fn on_disk_spilled(&mut self, id: RequestId, from: DiskSource) {
        if let Some(metrics) = &self.metrics {
            metrics.record_kv_disk_spill(from);
        }
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.spilled",
            request_id = id,
            from = from.as_str(),
            "written to the KV-disk tier"
        );
    }

    /// A KV-disk transfer failed (spec vram-budget/03). Not a request-log
    /// line: a refused write belongs to no request, and a refused read is
    /// already an ERROR on the model thread naming its request.
    pub fn on_disk_failure(&mut self, op: DiskOp) {
        if let Some(metrics) = &self.metrics {
            metrics.record_kv_disk_failure(op);
        }
    }

    pub fn on_requeued(&mut self, id: RequestId) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.prefill_chunks = 0;
            rt.prefilled_tokens = 0;
            // GitHub #192: a requeued multimodal request re-prefills from
            // the start, so it encodes its items again —
            // the encode seconds of the discarded attempt are not this
            // prefill's, for the same reason its chunks are not.
            rt.encode_micros = 0;
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
            kv_used_pct: kv_used_pct(self.occupancy),
            kv_evictions: self.kv_evictions,
        }
    }

    /// Called once per scheduler step / driver tick: returns the counters it
    /// computed (GitHub #69: the async telemetry consumer republishes them
    /// into the wait-free `ArcSwap` snapshot every tick, without
    /// recomputing), and emits the `ignis.scheduler.interval` DEBUG event
    /// only when the authoritative counters differ from the last one logged
    /// (ADR 0025). A decode run holds them constant for thousands of steps;
    /// one event per step would crowd every other DEBUG event out of the
    /// logging queue's drop-oldest buffer. `prefilling` is a placeholder zero
    /// (see the module doc), so it is not an attribute.
    ///
    /// `occupancy` is what the scheduler held when the step that sent this
    /// tick ended (GitHub #216, ADR 0030). It joins the change detection
    /// rather than riding a line whose cadence some other counter sets: an
    /// integer percentage takes at most 101 values, so a pool filling
    /// steadily logs a line per point and not a line per step.
    /// Read the load's own counters from `source` from now on (GitHub #301,
    /// #302), at once and then at every tick; `None` for a load without
    /// them.
    pub fn with_counter_source(&mut self, source: Option<Arc<FlashNextCounterSource>>) {
        self.counter_source = source;
        self.project_counter_source();
    }

    /// The counter source's latest reading into the projection, when there
    /// is one to keep up to date.
    fn project_counter_source(&self) {
        if let Some(metrics) = &self.metrics {
            metrics.set_flash_next(self.counter_source.as_ref().map(|source| source.read()).as_ref());
        }
    }

    pub fn emit_interval(&mut self, occupancy: Occupancy) -> IntervalCounters {
        self.tick = self.tick.saturating_add(1);
        self.occupancy = occupancy;
        let counters = self.counters();
        if let Some(metrics) = &self.metrics {
            metrics.set_scheduler_requests(counters.waiting, counters.running);
            metrics.set_occupancy(occupancy);
        }
        self.project_counter_source();
        let logged =
            (counters.waiting, counters.running, counters.kv_used_pct, counters.kv_evictions);
        if self.last_logged != Some(logged) {
            self.last_logged = Some(logged);
            tracing::debug!(
                name: "ignis.scheduler.interval",
                tick = self.tick,
                waiting = counters.waiting,
                running = counters.running,
                kv_used_pct = counters.kv_used_pct,
                kv_evictions = counters.kv_evictions,
                "scheduler counters changed"
            );
        }
        counters
    }

    /// Emit the `admitted` line: the prefill phase's summary (P3-06) —
    /// `prompt_tokens`, `prefill_chunks_consumed`, `prefilled_tokens` —
    /// alongside the lane dealt, how long the request queued, and its
    /// admission `class` (GitHub #120: the request log's per-class
    /// attribution). See [`Telemetry::emit_done`] for why this runs on the
    /// telemetry consumer task rather than inline. `media.*` is the
    /// request's media: what acquiring them cost (GitHub #179 — items,
    /// vision tokens, acquired bytes, preprocessing seconds, cache hits and
    /// misses) and what encoding them cost in prefill (GitHub #192 —
    /// `media.encode_seconds`). Every one of them is absent on a request
    /// without media, so a slow multimodal TTFT is attributable from this
    /// one line, and a text request's line is untouched.
    #[allow(clippy::too_many_arguments)]
    fn emit_admitted(
        &self,
        id: RequestId,
        ms: u64,
        lane: LaneId,
        prompt_tokens: u32,
        prefill_chunks_consumed: u32,
        prefilled_tokens: u32,
        class: RequestClass,
        summary: Option<MediaSummary>,
    ) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        let encode_seconds = summary.map(|s| s.encode_micros as f64 / 1e6);
        let media = summary.map(|s| s.stats);
        tracing::info!(
            name: "ignis.request.admitted",
            request_id = id,
            duration_ms = ms,
            lane = lane as u64,
            prompt_tokens,
            prefill_chunks_consumed,
            prefilled_tokens,
            class = class.as_extension_str(),
            media.items = media.map(|m| m.items),
            media.vision_tokens = media.map(|m| m.vision_tokens),
            media.natural_vision_tokens = media.map(|m| m.natural_vision_tokens),
            media.bytes = media.map(|m| m.media_bytes),
            media.preprocess_seconds = media.map(|m| m.preprocess_seconds),
            media.cache_hits = media.map(|m| m.cache_hits),
            media.cache_misses = media.map(|m| m.cache_misses),
            media.encode_seconds = encode_seconds,
            "request admitted"
        );
    }

    /// Emit the `ttft` line.
    fn emit_ttft(&self, id: RequestId, ms: u64, lane: LaneId, class: RequestClass) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.ttft",
            request_id = id,
            duration_ms = ms,
            lane = lane as u64,
            tokens = 1,
            tok_s = throughput(1, ms),
            class = class.as_extension_str(),
            "first token"
        );
    }

    /// Emit the `cancelled` line (GitHub #327): why, after how long, whether
    /// a lane had been dealt (`lane` only then) and how many tokens it had
    /// generated. A cancel ends a request with no `done` line, so without
    /// this one a request that hit its 504, or whose client went away, left
    /// nothing in the request log past its `admitted`.
    fn emit_cancelled(&self, id: RequestId, cause: CancelCause, ms: u64, rt: &RequestTelemetry) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.cancelled",
            request_id = id,
            cause = cause.as_str(),
            duration_ms = ms,
            admitted = rt.admitted,
            lane = rt.admitted.then_some(rt.lane as u64),
            tokens = if rt.ttft { 1 + rt.itl_count } else { 0 },
            class = rt.class.as_extension_str(),
            "request cancelled"
        );
    }

    /// Emit the `evicted` line (P4-07, GitHub #125): the KV-RAM host tier's
    /// snapshot cost, attributable per request from the request log alone.
    fn emit_evicted(&self, id: RequestId, snapshot_micros: u64) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.evicted",
            request_id = id,
            snapshot_micros,
            "evicted to the host KV-RAM tier"
        );
    }

    /// The `dropped` line (GitHub #224): the host tier gave this request's
    /// snapshot up to make room for another, so every prefilled token is
    /// gone and the request starts over. The costly counterpart to
    /// [`Telemetry::emit_evicted`], whose snapshot survives.
    fn emit_dropped(&self, id: RequestId) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.dropped",
            request_id = id,
            "snapshot dropped from the host KV-RAM tier: re-prefills from the start"
        );
    }

    /// Emit the `restored` line (P4-07, GitHub #125): the host tier's
    /// restore cost — the other half of [`Telemetry::emit_evicted`]'s
    /// attribution.
    fn emit_restored(&self, id: RequestId, restore_micros: u64) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        tracing::info!(
            name: "ignis.request.restored",
            request_id = id,
            restore_micros,
            "restored from the host KV-RAM tier"
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
    /// gate's own oracle — that stays HTTP/SSE-side, P3-07). `class`
    /// (GitHub #120) makes that same attribution per-class, not just
    /// per-window. `spec.rounds`/`spec.drafted`/`spec.accepted` (P5-06,
    /// GitHub #154) are the request's speculative rounds, summed once here
    /// rather than logged per round, and absent on a request that ran none;
    /// `spec.pos` (GitHub #160) is their per-position acceptance profile,
    /// the reference's `pos=[...]`, so the two logs read side by side.
    /// `media.*` (GitHub #245) are the `admitted` line's, from the same
    /// [`MediaSummary`]: a readout-only decision is never admitted onto a
    /// lane (#238), so this is the one line it reports its media on, and a
    /// request that was admitted reports the same numbers on both. Absent on
    /// a request without media, as there.
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
        class: RequestClass,
        spec: Option<SpecCounters>,
        reuse: Option<RequestReuse>,
        thinking: ThinkingReport,
        summary: Option<MediaSummary>,
    ) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        // A `None` field records nothing, so a request without speculative
        // rounds carries no `spec.*` attributes at all.
        let spec_pos = spec.map(|s| s.acceptance_profile());
        let encode_seconds = summary.map(|s| s.encode_micros as f64 / 1e6);
        let media = summary.map(|s| s.stats);
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
            class = class.as_extension_str(),
            spec.rounds = spec.map(|s| s.rounds),
            spec.drafted = spec.map(|s| s.drafted),
            spec.accepted = spec.map(|s| s.accepted),
            spec.pos = spec_pos.as_deref(),
            // GitHub #186 (ADR 0029). Absent on a request that reused
            // nothing: `reuse_source: none` *is* the absence of the field,
            // the way `spec.*` is absent on a load with no drafter, so the
            // log never reports a placeholder for something that never
            // happened.
            reuse_source = reuse.map(|r| r.source.as_str()),
            reused_prompt_tokens = reuse.map(|r| r.tokens),
            // GitHub #325: which kind of retained state the tokens came back
            // through. A prefix is page-floored and a checkpoint is not, so
            // the number alone does not say which one it was.
            reuse_kind = reuse.map(|r| r.kind.as_str()),
            // Absent for a prefix: it is claimed by sharing pages, so there
            // is no restore to time (0.0 would claim a free one).
            restore_ms = reuse.and_then(|r| r.restore_micros).map(|micros| micros as f64 / 1000.0),
            // Spec server/08. Absent on a request with no budget, the same
            // way: `thinking_forced: false` is said only about a budget that
            // could have been forced.
            thinking_budget = thinking.outcome.map(|t| t.budget),
            thinking_forced = thinking.outcome.map(|t| t.forced_at.is_some()),
            thinking_forced_at = thinking.outcome.and_then(|t| t.forced_at),
            // Where `</think>` came, or `thinking_closed: false` for a turn
            // that ended with its block open -- a client then shows the whole
            // output as reasoning. That tells a model that never closed from
            // a split that missed the close (#306).
            thinking_closed = thinking.outcome.map(|t| t.closed_at.is_some()),
            thinking_closed_at = thinking.outcome.and_then(|t| t.closed_at),
            thinking_budget_dropped = thinking.dropped_by_max.then_some("max"),
            // GitHub #315 (ADR 0048): the output index of the `</think>` the
            // leaf drew in place of an EOS inside the reasoning block --
            // absent on every request it did not happen to, budget or not.
            reasoning_redirected_at = thinking.redirected_at,
            // GitHub #245: the `admitted` line's `media.*`, again, on the
            // line every request reaches. A readout-only decision ends where
            // its prefill ends and is never admitted onto a lane (#238), so
            // without these its encode was paid for and logged nowhere.
            media.items = media.map(|m| m.items),
            media.vision_tokens = media.map(|m| m.vision_tokens),
            media.natural_vision_tokens = media.map(|m| m.natural_vision_tokens),
            media.bytes = media.map(|m| m.media_bytes),
            media.preprocess_seconds = media.map(|m| m.preprocess_seconds),
            media.cache_hits = media.map(|m| m.cache_hits),
            media.cache_misses = media.map(|m| m.cache_misses),
            media.encode_seconds = encode_seconds,
            "request done"
        );
    }
}

/// What a request's `done` line says about its thinking budget (spec
/// server/08).
#[derive(Debug, Clone, Copy)]
struct ThinkingReport {
    /// What the scheduler reported the budget did; `None` with no budget.
    outcome: Option<BudgetOutcome>,
    /// `max` dropped a budget at the handler.
    dropped_by_max: bool,
    /// Where an EOS inside the reasoning block became `</think>` (GitHub
    /// #315); `None` where none did.
    redirected_at: Option<u32>,
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

    fn telemetry() -> Telemetry {
        Telemetry::new(Arc::new(FixedClock::new(0)))
    }

    /// The `ignis.scheduler.interval` events (ADR 0025) among `events`.
    fn intervals(events: &[serde_json::Value]) -> Vec<&serde_json::Value> {
        events
            .iter()
            .filter(|e| e["event_name"] == "ignis.scheduler.interval")
            .collect()
    }

    /// An occupancy of `used` pages out of `pool`, with no KV-RAM in it.
    fn occupied(used: u32, pool: u32) -> Occupancy {
        Occupancy { kv_used_pages: used, kv_pool_pages: pool, ..Occupancy::default() }
    }

    #[test]
    fn an_interval_event_carries_the_authoritative_counters_at_debug() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.emit_interval(occupied(31, 100));
        });
        let intervals = intervals(&events);
        assert_eq!(intervals.len(), 1, "one tick, one interval event: {events:?}");
        let interval = intervals[0];
        assert_eq!(interval["severity_text"], "DEBUG");
        let attributes = &interval["attributes"];
        assert_eq!(attributes["tick"], 1, "the first interval is tick 1");
        assert_eq!(attributes["running"], 1, "the admitted request is on a lane");
        assert_eq!(attributes["waiting"], 0, "no queued request");
        assert_eq!(attributes["kv_evictions"], 0);
        // GitHub #216, ADR 0030: the scheduler carries its own occupancy on
        // the tick, so this one is a fact and is reported. `prefilling` is
        // still a placeholder zero and is still absent (ADR 0025).
        assert_eq!(attributes["kv_used_pct"], 31, "{interval}");
        assert!(attributes.get("prefilling").is_none(), "{interval}");
    }

    #[test]
    fn a_pool_filling_up_moves_the_interval_line_on_its_own() {
        // Before GitHub #216 the line changed only when `waiting`, `running`
        // or `kv_evictions` did, so a decode run that filled the pool logged
        // one line at the start and nothing after. The percentage is in the
        // change detection, so it is never a figure frozen at the last time
        // some other counter moved.
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            for used in [10u32, 10, 20, 20, 40] {
                telemetry.emit_interval(occupied(used, 100));
            }
        });
        let pcts: Vec<u64> = intervals(&events)
            .iter()
            .map(|e| e["attributes"]["kv_used_pct"].as_u64().unwrap())
            .collect();
        assert_eq!(pcts, vec![10, 20, 40], "a line per point, not a line per step");
    }

    #[test]
    fn an_empty_pool_reports_no_occupancy_rather_than_dividing_by_it() {
        // The placeholder load builds no KV pool; a percentage of nothing is
        // reported as 0 rather than panicking the telemetry consumer.
        let mut telemetry = telemetry();
        let counters = telemetry.emit_interval(occupied(0, 0));
        assert_eq!(counters.kv_used_pct, 0);
    }

    #[test]
    fn a_counter_source_reaches_the_projection_at_once_at_every_tick_and_leaves_with_none() {
        // GitHub #301, #302: the consumer reads host memory the device and
        // the n-gram table write; nothing is asked of the model thread.
        use ignis_core::flash_next_counters::FlashNextCounterSource;
        use ignis_core::ngram_table::NgramCounts;
        use ignis_core::residency::{ResidencyCounters, ResidencyMirror};

        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        let without = metrics.render();
        let mirror = Arc::new(ResidencyMirror::new());
        let ngram = Arc::new(NgramCounts::default());
        let capacity = [8, 0, 0, 0, 0, 0, 0, 0];
        let source = FlashNextCounterSource::new(Arc::clone(&mirror), capacity, Arc::clone(&ngram));
        telemetry.with_counter_source(Some(Arc::new(source)));
        // The capacity before the first step.
        let installed = metrics.render();
        assert!(installed.contains("\nignis_expert_cache_slots{class=\"gate_up_k2\",state=\"capacity\"} 8\n"), "{installed}");

        let mut residency = ResidencyCounters::default();
        residency.misses[0][0] = 6;
        residency.stall_nanos = [2_500_000, 0];
        mirror.store(&residency, &[6, 0, 0, 0, 0, 0, 0, 0]);
        ngram.record(5, 1, 1, 4096);
        telemetry.emit_interval(Occupancy::default());
        let ticked = metrics.render();
        assert!(ticked.contains("\nignis_expert_cache_misses_total{class=\"gate_up_k2\",phase=\"decode\"} 6\n"), "{ticked}");
        assert!(ticked.contains("\nignis_expert_cache_slots{class=\"gate_up_k2\",state=\"in_use\"} 6\n"), "{ticked}");
        assert!(ticked.contains("\nignis_expert_residency_stall_seconds_total{phase=\"decode\"} 0.0025\n"), "{ticked}");
        assert!(ticked.contains("\nignis_ngram_rows_total{source=\"file\"} 1\n"), "{ticked}");

        telemetry.with_counter_source(None);
        assert_eq!(metrics.render(), without, "a load without counters renders none of a dead model's");
    }

    #[test]
    fn the_release_the_last_step_reports_reaches_the_projection() {
        // GitHub #216: no tick is sent while the scheduler is idle, so the
        // step that releases the last request's pages is the last chance to
        // report the release. A projection that read a stale figure would
        // hold it for as long as the load stayed idle.
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.on_admitted(1, 0);
        telemetry.emit_interval(Occupancy {
            kv_used_pages: 64,
            kv_pool_pages: 256,
            kv_ram_used_bytes: 1 << 30,
                kv_disk_used_bytes: 0,
});
        let busy = metrics.render();
        assert!(busy.contains("\nignis_kv_pool_used_pages 64\n"), "{busy}");
        assert!(busy.contains("\nignis_kv_ram_arena_bytes{state=\"used\"} 1073741824\n"), "{busy}");

        telemetry.on_done(1, 3, FinishReason::Stop, None, None, None);
        telemetry.emit_interval(Occupancy {
            kv_used_pages: 0,
            kv_pool_pages: 256,
            kv_ram_used_bytes: 0,
                kv_disk_used_bytes: 0,
});
        let idle = metrics.render();
        assert!(idle.contains("\nignis_kv_pool_used_pages 0\n"), "{idle}");
        assert!(idle.contains("\nignis_kv_ram_arena_bytes{state=\"used\"} 0\n"), "{idle}");
    }

    #[test]
    fn the_retained_slot_facts_reach_the_projection_and_not_only_a_debug_line() {
        // GitHub #216: `on_retained_slots` logged at DEBUG and never touched
        // the projection, so slot occupancy was invisible to a scrape.
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.on_retained_slots(3, 8);
        telemetry.on_retained_slot_skipped(7, RetainedSkip::CaptureNoSlot, 8, 8);
        telemetry.on_retained_slot_skipped(9, RetainedSkip::CaptureNoSlot, 8, 8);

        let text = metrics.render();
        assert!(text.contains("\nignis_retained_slots{state=\"in_use\"} 3\n"), "{text}");
        assert!(
            text.contains("\nignis_retained_slot_skips_total{reason=\"capture_skipped_no_slot\"} 2\n"),
            "{text}"
        );
        // Capacity is the load's own record, not something a skip teaches.
        assert!(text.contains("\nignis_retained_slots{state=\"capacity\"} 0\n"), "{text}");
    }

    #[test]
    fn unchanged_counters_are_not_logged_again_but_are_still_returned() {
        let mut telemetry = telemetry();
        let mut returned = Vec::new();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            // Three steps of a steady decode: one change, then nothing new.
            for _ in 0..3 {
                returned.push(telemetry.emit_interval(Occupancy::default()));
            }
            telemetry.on_done(1, 3, FinishReason::Stop, None, None, None);
            returned.push(telemetry.emit_interval(Occupancy::default()));
        });
        let intervals = intervals(&events);
        let ticks: Vec<u64> = intervals
            .iter()
            .map(|e| e["attributes"]["tick"].as_u64().unwrap())
            .collect();
        assert_eq!(ticks, vec![1, 4], "only ticks whose counters changed are logged");
        assert_eq!(intervals[1]["attributes"]["running"], 0, "the request finished");
        // The snapshot the consumer republishes every tick is unaffected.
        let running: Vec<u32> = returned.iter().map(|c| c.running).collect();
        assert_eq!(running, vec![1, 1, 1, 0]);
    }

    /// Captures every event a block of code emits through `tracing` — the
    /// `ignis.request.*` lifecycle (GitHub #79) and `ignis.scheduler.interval`
    /// (ADR 0025) — via `ignis-logging`'s own `JsonLayer`/`MemorySink`, with
    /// no level filter.
    fn capture_events(f: impl FnOnce()) -> Vec<serde_json::Value> {
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
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(7, 10, RequestClass::Interactive);
            telemetry.on_admitted(7, 2);
            telemetry.on_token(7); // first token → ttft
            telemetry.on_token(7); // subsequent tokens are not re-emitted
            telemetry.on_done(7, 4, FinishReason::Stop, None, None, None);
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
        for field in ["spec.rounds", "spec.drafted", "spec.accepted", "spec.pos"] {
            assert!(
                done["attributes"].get(field).is_none(),
                "no speculative rounds, no placeholder `{field}`: {done}"
            );
        }
    }

    #[test]
    fn the_done_event_carries_the_speculative_counters_once() {
        // P5-06 (GitHub #154): three verify rounds that proposed 21 drafts and
        // committed 9 of them, reported on the one `done` line -- with their
        // per-position profile (GitHub #160): 7, 2 and 0 committed.
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(3, 10, RequestClass::Agent);
            telemetry.on_admitted(3, 1);
            for _ in 0..12 {
                telemetry.on_token(3);
            }
            telemetry.on_done(
                3,
                12,
                FinishReason::Stop,
                Some(SpecCounters::round(7, 7) + SpecCounters::round(7, 2) + SpecCounters::round(7, 0)),
                None,
                None,
            );
        });

        let done: Vec<_> = events
            .iter()
            .filter(|e| e["event_name"] == "ignis.request.done")
            .collect();
        assert_eq!(done.len(), 1, "{events:?}");
        assert_eq!(done[0]["attributes"]["spec.rounds"], 3);
        assert_eq!(done[0]["attributes"]["spec.drafted"], 21);
        assert_eq!(done[0]["attributes"]["spec.accepted"], 9);
        assert_eq!(done[0]["attributes"]["spec.pos"], "67,67,33,33,33,33,33");
        assert_eq!(events.len(), 3, "admitted, ttft, done -- nothing per token or per round");
    }

    #[test]
    fn admitted_carries_the_media_acquisition_only_for_a_request_with_media() {
        let mut telemetry = telemetry();
        let media = MediaStats {
            items: 2,
            vision_tokens: 8,
            natural_vision_tokens: 20,
            media_bytes: 4096,
            preprocess_seconds: 0.25,
            cache_hits: 1,
            cache_misses: 1,
            source_pixels: Some((64, 64)),
        };
        let events = capture_events(|| {
            telemetry.note_submit(1, 20, RequestClass::Agent);
            telemetry.note_media(1, media);
            telemetry.on_admitted(1, 0);
            telemetry.note_submit(2, 3, RequestClass::Agent);
            telemetry.on_admitted(2, 1);
        });
        let admitted = |id: u64| {
            events
                .iter()
                .find(|e| e["event_name"] == "ignis.request.admitted" && e["attributes"]["request_id"] == id)
                .unwrap_or_else(|| panic!("no admitted event for {id}: {events:?}"))
        };
        let attributes = &admitted(1)["attributes"];
        assert_eq!(attributes["media.items"], 2, "{attributes}");
        assert_eq!(attributes["media.vision_tokens"], 8);
        assert_eq!(attributes["media.natural_vision_tokens"], 20);
        assert_eq!(attributes["media.bytes"], 4096);
        assert_eq!(attributes["media.preprocess_seconds"], 0.25);
        assert_eq!(attributes["media.cache_hits"], 1);
        assert_eq!(attributes["media.cache_misses"], 1);
        let text = admitted(2)["attributes"].as_object().unwrap();
        assert!(text.keys().all(|k| !k.starts_with("media.")), "{text:?}");
    }

    /// The `media.encode_seconds` of the one `admitted` event `f` produced.
    fn admitted_encode_seconds(events: &[serde_json::Value]) -> serde_json::Value {
        events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.admitted")
            .expect("an admitted event")["attributes"]["media.encode_seconds"]
            .clone()
    }

    #[test]
    fn encode_seconds_sum_over_the_requests_media_items() {
        let mut telemetry = telemetry();
        let media = MediaStats { items: 2, vision_tokens: 8, ..MediaStats::default() };
        let events = capture_events(|| {
            telemetry.note_submit(1, 40, RequestClass::Agent);
            telemetry.note_media(1, media);
            // Two images, each encoded on the chunk that first covers it,
            // with a chunk that reuses a live embedding between them.
            telemetry.on_prefill_chunk(1, 8, 300_000);
            telemetry.on_prefill_chunk(1, 20, 0);
            telemetry.on_prefill_chunk(1, 32, 200_000);
            telemetry.on_prefill_chunk(1, 40, 0);
            telemetry.on_admitted(1, 0);
        });
        assert_eq!(admitted_encode_seconds(&events), 0.5);
    }

    #[test]
    fn a_requeue_resets_the_encode_seconds_of_the_discarded_attempt() {
        let mut telemetry = telemetry();
        let media = MediaStats { items: 1, vision_tokens: 4, ..MediaStats::default() };
        let events = capture_events(|| {
            telemetry.note_submit(1, 20, RequestClass::Agent);
            telemetry.note_media(1, media);
            telemetry.on_prefill_chunk(1, 8, 250_000);
            // The attempt is discarded: the request re-prefills from the
            // start, so it encodes its item again.
            telemetry.on_requeued(1);
            telemetry.on_prefill_chunk(1, 8, 125_000);
            telemetry.on_prefill_chunk(1, 20, 0);
            telemetry.on_admitted(1, 0);
        });
        assert_eq!(admitted_encode_seconds(&events), 0.125);
    }

    #[test]
    fn a_text_request_carries_no_encode_seconds_even_when_a_chunk_reports_some() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(2, 5, RequestClass::Interactive);
            telemetry.on_prefill_chunk(2, 5, 9_000);
            telemetry.on_admitted(2, 0);
        });
        let attributes = events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.admitted")
            .expect("an admitted event")["attributes"]
            .as_object()
            .expect("attributes are an object")
            .clone();
        assert!(attributes.keys().all(|k| !k.starts_with("media.")), "{attributes:?}");
    }

    #[test]
    fn a_request_that_never_takes_a_lane_reports_its_media_on_done() {
        // GitHub #245: a readout-only decision ends where its prefill ends
        // (#238), so it never reaches `admitted`. Its media were paid for
        // all the same, and its `done` line is where they are reported.
        let mut telemetry = telemetry();
        let media = MediaStats {
            items: 1,
            vision_tokens: 4,
            natural_vision_tokens: 4,
            media_bytes: 2048,
            preprocess_seconds: 0.125,
            cache_hits: 0,
            cache_misses: 1,
            source_pixels: Some((64, 64)),
        };
        let events = capture_events(|| {
            telemetry.note_submit(1, 20, RequestClass::Agent);
            telemetry.note_media(1, media);
            telemetry.on_prefill_chunk(1, 20, 250_000);
            telemetry.on_done(1, 0, FinishReason::Stop, None, None, None);
            // A text request that never takes a lane either, whose chunk
            // reports encode time anyway: no media, so no `media.*`.
            telemetry.note_submit(2, 5, RequestClass::Agent);
            telemetry.on_prefill_chunk(2, 5, 9_000);
            telemetry.on_done(2, 0, FinishReason::Stop, None, None, None);
        });
        assert!(
            events.iter().all(|e| e["event_name"] != "ignis.request.admitted"),
            "neither request was dealt a lane: {events:?}"
        );
        let done = |id: u64| {
            events
                .iter()
                .find(|e| e["event_name"] == "ignis.request.done" && e["attributes"]["request_id"] == id)
                .unwrap_or_else(|| panic!("no done event for {id}: {events:?}"))["attributes"]
                .as_object()
                .expect("attributes are an object")
                .clone()
        };
        let image = done(1);
        assert_eq!(image["media.encode_seconds"], 0.25, "{image:?}");
        assert_eq!(image["media.vision_tokens"], 4, "{image:?}");
        assert_eq!(image["media.natural_vision_tokens"], 4);
        assert_eq!(image["media.items"], 1);
        assert_eq!(image["media.bytes"], 2048);
        assert_eq!(image["media.preprocess_seconds"], 0.125);
        assert_eq!(image["media.cache_hits"], 0);
        assert_eq!(image["media.cache_misses"], 1);
        let text = done(2);
        assert!(text.keys().all(|k| !k.starts_with("media.")), "{text:?}");
    }

    #[test]
    fn a_request_on_a_lane_reports_the_same_media_on_admitted_and_done() {
        // GitHub #245: `admitted` keeps its fields, and `done` repeats them,
        // so a reader finds them on the line every request reaches whatever
        // the primitive.
        let mut telemetry = telemetry();
        let media = MediaStats { items: 1, vision_tokens: 4, media_bytes: 2048, ..MediaStats::default() };
        let events = capture_events(|| {
            telemetry.note_submit(1, 20, RequestClass::Agent);
            telemetry.note_media(1, media);
            telemetry.on_prefill_chunk(1, 20, 250_000);
            telemetry.on_admitted(1, 0);
            telemetry.on_token(1);
            telemetry.on_done(1, 1, FinishReason::Stop, None, None, None);
        });
        let media_of = |name: &str| {
            let attributes = events
                .iter()
                .find(|e| e["event_name"] == name)
                .unwrap_or_else(|| panic!("no {name}: {events:?}"))["attributes"]
                .as_object()
                .expect("attributes are an object")
                .clone();
            attributes.into_iter().filter(|(k, _)| k.starts_with("media.")).collect::<Vec<_>>()
        };
        let admitted = media_of("ignis.request.admitted");
        assert_eq!(admitted.len(), 8, "every media field: {admitted:?}");
        assert_eq!(media_of("ignis.request.done"), admitted);
    }

    #[test]
    fn a_fixed_clock_keeps_request_events_deterministic() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.on_done(1, 3, FinishReason::Stop, None, None, None);
        });
        let done = events.last().unwrap();
        // FixedClock(0): a zero elapsed span → `duration_ms` and `tok_s` are 0.
        assert_eq!(done["attributes"]["duration_ms"], 0);
        assert_eq!(done["attributes"]["tok_s"], 0.0);
    }

    #[test]
    fn a_step_clock_reports_the_elapsed_span_on_request_events() {
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
            Telemetry::new(Arc::new(StepClock {
                reads: std::sync::atomic::AtomicU32::new(0),
            }));
        let events = capture_events(|| {
            telemetry.note_submit(1, 5, RequestClass::Interactive); // read #1 → 100 ms
            telemetry.on_done(1, 5, FinishReason::Length, None, None, None); // read #2 → 200 ms, so ms = 100
        });
        let done = events.last().unwrap();
        assert_eq!(done["attributes"]["duration_ms"], 100, "200 - 100 = 100 ms elapsed");
        assert_eq!(done["attributes"]["tok_s"], 50.0, "5 tokens / 0.1 s");
        assert_eq!(done["attributes"]["finish_reason"], "length");
    }

    #[test]
    fn event_derived_counters_split_waiting_and_running() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive); // queued (not yet admitted)
            telemetry.note_submit(2, 3, RequestClass::Interactive); // queued
            telemetry.on_admitted(2, 0); // dealt a lane
            telemetry.emit_interval(Occupancy::default());
        });
        let v = intervals(&events).last().copied().expect("an interval event");
        assert_eq!(v["attributes"]["waiting"], 1, "request 1 is still queued");
        assert_eq!(v["attributes"]["running"], 1, "request 2 is on a lane");
    }

    #[test]
    fn evictions_bump_the_counter() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.on_evicted(1, 450);
            telemetry.on_evicted(1, 450);
            telemetry.emit_interval(Occupancy::default());
        });
        let v = intervals(&events).last().copied().expect("an interval event");
        assert_eq!(v["attributes"]["kv_evictions"], 2, "each eviction bumps the counter");
    }

    /// P4-07 (GitHub #125): the host tier's snapshot/restore wall time is
    /// recorded in the request log, so the tier's cost is attributable
    /// without a second run.
    #[test]
    fn evicted_and_restored_report_the_tier_s_wall_time() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Agent);
            telemetry.on_admitted(1, 0);
            telemetry.on_evicted(1, 45_000);
            telemetry.on_restored(1, 44_500);
        });
        let event_names: Vec<&str> =
            events.iter().map(|e| e["event_name"].as_str().unwrap()).collect();
        assert!(event_names.contains(&"ignis.request.evicted"));
        assert!(event_names.contains(&"ignis.request.restored"));
        let evicted = events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.evicted")
            .unwrap();
        assert_eq!(evicted["attributes"]["snapshot_micros"], 45_000);
        let restored = events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.restored")
            .unwrap();
        assert_eq!(restored["attributes"]["restore_micros"], 44_500);
    }

    /// GitHub #215: how many retained slots are held is logged when it
    /// changes, and a skipped publish or capture is logged on the request it
    /// happened to, with why and what was held.
    #[test]
    fn retained_slots_in_use_and_skips_are_logged() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(4, 40, RequestClass::Agent);
            telemetry.on_retained_slots(3, 3);
            telemetry.on_retained_slot_skipped(4, RetainedSkip::CaptureNoSlot, 3, 3);
            telemetry.on_retained_slot_skipped(4, RetainedSkip::PublishNoSlot, 2, 3);
        });
        let slots = events
            .iter()
            .find(|e| e["event_name"] == "ignis.scheduler.retained_slots")
            .expect("the held count is logged");
        assert_eq!(slots["severity_text"], "DEBUG");
        assert_eq!(slots["attributes"]["retained_slots_in_use"], 3);
        assert_eq!(slots["attributes"]["retained_slots"], 3);
        let skips: Vec<&serde_json::Value> = events
            .iter()
            .filter(|e| e["event_name"] == "ignis.request.retained_slot_skipped")
            .collect();
        assert_eq!(skips.len(), 2, "{events:?}");
        assert_eq!(skips[0]["attributes"]["reason"], "capture_skipped_no_slot");
        assert_eq!(skips[1]["attributes"]["reason"], "publish_skipped_no_slot");
        assert_eq!(skips[0]["attributes"]["request_id"], 4);
        assert_eq!(skips[0]["attributes"]["retained_slots_in_use"], 3);
        assert_eq!(skips[1]["attributes"]["retained_slots_in_use"], 2);
        assert_eq!(skips[1]["attributes"]["retained_slots"], 3);
    }

    #[test]
    fn admitted_reports_the_prefill_phase_summary() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 300, RequestClass::Interactive); // a 300-token prompt
            telemetry.on_prefill_chunk(1, 128, 0); // chunk 1: 128/300
            telemetry.on_prefill_chunk(1, 256, 0); // chunk 2: 256/300
            telemetry.on_prefill_chunk(1, 300, 0); // chunk 3: prefill complete
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
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 300, RequestClass::Interactive);
            telemetry.on_prefill_chunk(1, 128, 0); // 1st attempt: 1 chunk, then evicted/discarded
            telemetry.on_requeued(1); // re-queued: the summary resets
            telemetry.on_prefill_chunk(1, 150, 0); // 2nd attempt, chunk 1
            telemetry.on_prefill_chunk(1, 300, 0); // 2nd attempt, chunk 2: complete
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
        // A clock that advances 10 ms per read, so every generated token is
        // exactly 10 ms apart — a deterministic, known ITL distribution.
        struct StepClock(std::sync::atomic::AtomicU32);
        impl TelemetryClock for StepClock {
            fn now_ms(&self) -> u64 {
                (self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as u64 + 1) * 10
            }
        }
        let mut telemetry =
            Telemetry::new(Arc::new(StepClock(std::sync::atomic::AtomicU32::new(0))));
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.on_token(1); // ttft — no ITL sample yet
            telemetry.on_token(1); // 1st gap: 10ms
            telemetry.on_token(1); // 2nd gap: 10ms
            telemetry.on_done(1, 3, FinishReason::Stop, None, None, None);
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
        let mut telemetry = Telemetry::new(Arc::new(clock));
        let events = capture_events(|| {
            telemetry.note_submit(1, 4, RequestClass::Interactive); // t=0
            telemetry.on_admitted(1, 0); // t=5, lane 0
            telemetry.on_token(1); // t=10, ttft
            telemetry.on_token(1); // t=20, 1st gap 10ms
            telemetry.note_submit(2, 32_768, RequestClass::Interactive); // t=25 (cold prefiller submitted)
            telemetry.on_prefill_chunk(2, 32_768, 0); // the wide chunk lands
            telemetry.on_token(1); // t=220, 2nd gap: 200ms — the stall
            telemetry.on_admitted(2, 1); // t=230, prefiller starts decoding
            telemetry.on_done(1, 3, FinishReason::Stop, None, None, None); // t=220 (clock re-read), lane 0 finishes
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

    /// GitHub #89 / ADR 0017: an installed projection follows the same
    /// lifecycle calls that emit the request and interval events.
    #[test]
    fn an_installed_projection_follows_the_request_lifecycle() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        let has = |line: &str| {
            let text = metrics.render();
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        };

        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.note_submit(2, 3, RequestClass::Agent);
        telemetry.on_admitted(2, 0);
        telemetry.emit_interval(Occupancy::default());
        has("ignis_requests_accepted_total 2");
        has("ignis_scheduler_requests{state=\"waiting\"} 1");
        has("ignis_scheduler_requests{state=\"running\"} 1");
        has("ignis_requests_completed_total 0");

        telemetry.on_token(2);
        telemetry.on_done(2, 5, FinishReason::Stop, None, None, None);
        telemetry.emit_interval(Occupancy::default());
        has("ignis_requests_completed_total 1");
        has("ignis_generated_tokens_total 5");
        has("ignis_scheduler_requests{state=\"waiting\"} 1");
        has("ignis_scheduler_requests{state=\"running\"} 0");
    }

    /// GitHub #165: decoded tokens count as each one arrives, so a long
    /// request's work shows while it runs — not only when it completes, as
    /// `ignis_generated_tokens_total` does. A token for a request no longer in
    /// flight (cancelled, routed late) is not counted.
    #[test]
    fn decoded_tokens_count_as_they_arrive_not_when_the_request_completes() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        let has = |line: &str| {
            let text = metrics.render();
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        };

        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.on_admitted(1, 0);
        telemetry.on_token(1);
        telemetry.on_token(1);
        telemetry.on_token(1);
        has("ignis_decoded_tokens_total 3");
        has("ignis_generated_tokens_total 0");

        telemetry.on_done(1, 3, FinishReason::Stop, None, None, None);
        has("ignis_decoded_tokens_total 3");
        has("ignis_generated_tokens_total 3");

        telemetry.on_token(1); // late, for a request already done
        telemetry.on_token(9); // never submitted
        has("ignis_decoded_tokens_total 3");
    }

    /// GitHub #89: a cancelled request leaves the in-flight set, and an
    /// admission the model thread emitted before the cancel — arriving after
    /// it — does not bring it back.
    #[test]
    fn a_cancelled_request_leaves_the_counters_even_if_its_admission_arrives_late() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.note_submit(2, 3, RequestClass::Interactive);
        telemetry.on_admitted(2, 0);

        telemetry.on_cancelled(1, CancelCause::Consumer);
        let text = metrics.render();
        assert!(text.contains("\nignis_scheduler_requests{state=\"waiting\"} 0\n"), "{text}");
        assert!(text.contains("\nignis_scheduler_requests{state=\"running\"} 1\n"), "{text}");

        telemetry.on_admitted(1, 1); // emitted before the cancel, routed after
        telemetry.on_cancelled(2, CancelCause::Consumer);
        telemetry.on_cancelled(2, CancelCause::Consumer); // a second cancel is a no-op
        let counters = telemetry.emit_interval(Occupancy::default());
        assert_eq!((counters.waiting, counters.running), (0, 0));
        let text = metrics.render();
        assert!(text.contains("\nignis_scheduler_requests{state=\"running\"} 0\n"), "{text}");
        assert!(text.contains("\nignis_requests_completed_total 0\n"), "{text}");
        assert!(text.contains("\nignis_requests_cancelled_total 2\n"), "{text}");
    }

    /// A request that finished in the same step its cancel was sent: the
    /// cancel reaches the consumer first, its `Done` after. Counted once.
    #[test]
    fn a_done_arriving_after_the_cancel_is_not_counted_again() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.on_admitted(1, 0);
        telemetry.on_cancelled(1, CancelCause::Consumer);
        telemetry.on_done(1, 4, FinishReason::Stop, None, None, None);
        // A `Done` for a request that was never cancelled still counts.
        telemetry.note_submit(2, 3, RequestClass::Interactive);
        telemetry.on_done(2, 3, FinishReason::Length, None, None, None);

        let text = metrics.render();
        for line in [
            "ignis_requests_accepted_total 2",
            "ignis_requests_cancelled_total 1",
            "ignis_requests_completed_total 1",
            "ignis_generated_tokens_total 3",
        ] {
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        }
    }

    /// GitHub #90 / ADR 0017: evictions, prefix reuse and both latency
    /// histograms follow the same lifecycle calls, on the telemetry clock.
    #[test]
    fn an_installed_projection_observes_evictions_prefix_reuse_and_latency() {
        // Reads, in order: submit 0, first token 120, second token 130,
        // done 3000.
        struct ScriptClock(std::sync::atomic::AtomicUsize, Vec<u64>);
        impl TelemetryClock for ScriptClock {
            fn now_ms(&self) -> u64 {
                self.1[self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst)]
            }
        }
        let metrics = Arc::new(Metrics::new());
        let mut telemetry =
            Telemetry::new(Arc::new(ScriptClock(Default::default(), vec![0, 120, 130, 3000])));
        telemetry.with_metrics(Arc::clone(&metrics));

        telemetry.note_submit(1, 3, RequestClass::Agent);
        telemetry.on_prefix_reused(7, 32, false, ReuseSource::Device);
        telemetry.on_evicted(1, 450);
        telemetry.on_token(1);
        telemetry.on_token(1);
        telemetry.on_done(1, 2, FinishReason::Stop, None, None, None);

        let text = metrics.render();
        for line in [
            "ignis_kv_cache_evictions_total 1",
            "ignis_prefix_reused_tokens_total 32",
            "ignis_request_ttft_seconds_bucket{le=\"0.1\"} 0",
            "ignis_request_ttft_seconds_bucket{le=\"0.25\"} 1",
            "ignis_request_ttft_seconds_sum 0.12",
            "ignis_request_ttft_seconds_count 1",
            "ignis_request_duration_seconds_bucket{le=\"2.5\"} 0",
            "ignis_request_duration_seconds_bucket{le=\"5\"} 1",
            "ignis_request_duration_seconds_sum 3",
            "ignis_request_duration_seconds_count 1",
        ] {
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        }
    }

    /// GitHub #224 — the two ends of the KV-RAM tier are separate series,
    /// and a requeue on its own is not one of them. A failed restore raises
    /// `Requeued` with no `SnapshotDropped` beside it: the request pays the
    /// same re-prefill, but KV-RAM evicted nothing, so the tier's eviction
    /// counter must not move.
    #[test]
    fn a_requeue_without_a_drop_is_not_a_kv_ram_eviction() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = Telemetry::new(Arc::new(FixedClock::new(0)));
        telemetry.with_metrics(Arc::clone(&metrics));

        // Arrival at the tier, then a departure from it.
        telemetry.on_evicted(1, 450);
        telemetry.on_snapshot_dropped(1);
        // A restore that failed: requeue only.
        telemetry.on_requeued(2);

        let text = metrics.render();
        for line in ["ignis_kv_cache_evictions_total 1", "ignis_kv_ram_evictions_total 1"] {
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        }
    }

    /// Spec vram-budget/03 AC 38: a live snapshot written to KV-disk is
    /// counted where it lands, by where it came from -- and in neither of the
    /// two KV-RAM eviction counters, whose meanings stay theirs. A failure is
    /// counted by its direction.
    #[test]
    fn a_disk_spill_counts_where_it_lands_and_in_neither_kv_ram_counter() {
        let metrics = Arc::new(Metrics::new());
        metrics.set_load_reservations(crate::metrics::LoadReservations {
            lines: ignis_core::VramLines::default(),
            budget_bytes: 0,
            kv_pool_pages: 0,
            kv_page_bytes: 0,
            residency_bytes: 0,
            expert_cache_bytes: 0,
            condition_budget_bytes: 0,
            kv_ram_arena_bytes: 0,
            retained_slots: 0,
            retained_host_slots: 0,
            retained_host_bytes: 0,
            kv_disk_bytes: Some(16 << 30),
            flash_next: None,
        });
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Agent);
            telemetry.on_admitted(1, 0);
            telemetry.on_disk_spilled(1, DiskSource::Device);
            telemetry.on_disk_spilled(2, DiskSource::KvRam);
            telemetry.on_disk_spilled(3, DiskSource::KvRam);
            telemetry.on_disk_failure(DiskOp::Write);
            telemetry.on_disk_failure(DiskOp::Read);
            telemetry.on_disk_failure(DiskOp::Read);
        });
        let text = metrics.render();
        for line in [
            "ignis_kv_disk_spills_total{from=\"device\"} 1",
            "ignis_kv_disk_spills_total{from=\"kv_ram\"} 2",
            "ignis_kv_disk_failures_total{op=\"write\"} 1",
            "ignis_kv_disk_failures_total{op=\"read\"} 2",
            "ignis_kv_cache_evictions_total 0",
            "ignis_kv_ram_evictions_total 0",
        ] {
            assert!(text.contains(&format!("
{line}
")), "no `{line}` in:
{text}");
        }
        let spilled = events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.spilled")
            .expect("a `spilled` line");
        assert_eq!(spilled["attributes"]["request_id"], 1);
        assert_eq!(spilled["attributes"]["from"], "device");
    }

    /// GitHub #224 — and the drop is in the request log too, for the reason
    /// `evicted` and `restored` are (#125): `Requeued` carries no line, so
    /// without `dropped` the log would show an eviction and then a second
    /// admission with nothing saying the work in between was thrown away.
    #[test]
    fn a_dropped_snapshot_is_a_line_in_the_request_log() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Agent);
            telemetry.on_admitted(1, 0);
            telemetry.on_evicted(1, 45_000);
            telemetry.on_snapshot_dropped(1);
            telemetry.on_requeued(1);
        });
        let dropped = events
            .iter()
            .find(|e| e["event_name"] == "ignis.request.dropped")
            .expect("a `dropped` line");
        assert_eq!(dropped["attributes"]["request_id"], 1);
        // The requeue that follows still writes no line of its own, which is
        // exactly why this one has to exist.
        assert!(
            !events.iter().any(|e| e["event_name"] == "ignis.request.requeued"),
            "a requeue has no line of its own: {events:?}"
        );
    }

    /// Spec server/09: a request ended at its caller's stop sequence is
    /// released like a cancel, but it completed — one `done` line with
    /// `stop`, counted once as completed, even when its own `Done` raced the
    /// release; and one whose `Done` came first is not reported twice.
    #[test]
    fn a_stopped_request_is_done_with_stop_and_counted_once_as_completed() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.on_stopped(1, 5);
            telemetry.on_done(1, 6, FinishReason::Stop, None, None, None); // raced the release
            telemetry.note_submit(2, 3, RequestClass::Interactive);
            telemetry.on_done(2, 2, FinishReason::Stop, None, None, None);
            telemetry.on_stopped(2, 2); // its `Done` came first
        });
        let done: Vec<_> = events.iter().filter(|e| e["event_name"] == "ignis.request.done").collect();
        assert_eq!(done[0]["attributes"]["request_id"], 1);
        assert_eq!(done[0]["attributes"]["finish_reason"], "stop");
        assert_eq!(done[0]["attributes"]["tokens"], 5);
        assert!(!done.iter().any(|e| e["attributes"]["request_id"] == 2 && e["attributes"]["tokens"] != 2));

        let text = metrics.render();
        for line in [
            "ignis_requests_completed_total 2",
            "ignis_requests_cancelled_total 0",
            "ignis_generated_tokens_total 7",
        ] {
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        }
    }

    /// A cancelled request never completed: its late `Done` is not a
    /// request-duration observation, and one never seen submitted has no span.
    #[test]
    fn cancelled_and_unknown_requests_are_not_observed_as_durations() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.on_cancelled(1, CancelCause::Consumer);
        telemetry.on_done(1, 4, FinishReason::Stop, None, None, None);
        telemetry.on_done(99, 4, FinishReason::Stop, None, None, None);

        let text = metrics.render();
        assert!(text.contains("\nignis_request_duration_seconds_count 0\n"), "{text}");
    }

    /// GitHub #325: the `done` line says which retained state a request came
    /// back through, and says nothing for one that reused nothing.
    #[test]
    fn the_done_line_reports_a_prefix_restore_a_checkpoint_restore_and_a_cold_prefill() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            for id in 1..=4 {
                telemetry.note_submit(id, 3, RequestClass::Interactive);
                telemetry.on_admitted(id, 0);
            }
            telemetry.on_prefix_reused(1, 78_784, true, ReuseSource::KvRam); // a retained prefix, brought back
            telemetry.on_state_reused(2, ReuseSource::KvRam, RetainedKind::Checkpoint, 1536, 42);
            telemetry.on_prefix_reused(4, 32, false, ReuseSource::Device); // a live sibling's
            for id in 1..=4 {
                telemetry.on_done(id, 1, FinishReason::Stop, None, None, None);
            }
        });
        let done = |id: u64| {
            events
                .iter()
                .find(|e| e["event_name"] == "ignis.request.done" && e["attributes"]["request_id"] == id)
                .unwrap_or_else(|| panic!("no done line for {id}: {events:?}"))["attributes"]
                .clone()
        };

        let prefix = done(1);
        assert_eq!(prefix["reuse_source"], "kv_ram");
        assert_eq!(prefix["reused_prompt_tokens"], 78_784);
        assert_eq!(prefix["reuse_kind"], "prefix");
        assert!(prefix.get("restore_ms").is_none(), "a prefix restores nothing to time: {prefix}");

        let checkpoint = done(2);
        assert_eq!(checkpoint["reuse_source"], "kv_ram");
        assert_eq!(checkpoint["reused_prompt_tokens"], 1536);
        assert_eq!(checkpoint["reuse_kind"], "checkpoint");
        assert_eq!(checkpoint["restore_ms"], 0.042);

        for cold in [done(3), done(4)] {
            for field in ["reuse_source", "reused_prompt_tokens", "reuse_kind", "restore_ms"] {
                assert!(cold.get(field).is_none(), "`{field}` on a request that reused nothing: {cold}");
            }
        }
    }

    #[test]
    fn reuse_is_counted_by_kind_and_tier_never_summed_into_the_sibling_counter() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(7, 2048, RequestClass::Interactive);
        telemetry.on_state_reused(7, ReuseSource::KvRam, RetainedKind::Checkpoint, 1536, 42);
        telemetry.on_prefix_reused(7, 64, true, ReuseSource::Device);
        telemetry.on_prefix_reused(7, 128, true, ReuseSource::KvRam);
        telemetry.on_prefix_reused(7, 32, false, ReuseSource::Device);
        telemetry.on_retained_state(
            RetainedStateOperation::Spill,
            ReuseSource::KvRam,
            RetainedKind::Prefix,
        );

        let text = metrics.render();
        for line in [
            // A checkpoint's restore and a retained prefix's claim are both
            // retained reuse, and the two are told apart (GitHub #216).
            "ignis_retained_reused_tokens_total{tier=\"kv_ram\",kind=\"checkpoint\"} 1536",
            // A prefix its claim brought back from KV-RAM is counted on the
            // tier its hit and its `done` line name (#325).
            "ignis_retained_reused_tokens_total{tier=\"kv_ram\",kind=\"prefix\"} 128",
            "ignis_retained_reused_tokens_total{tier=\"device\",kind=\"prefix\"} 64",
            "ignis_retained_reused_tokens_total{tier=\"device\",kind=\"checkpoint\"} 0",
            "ignis_prefix_reused_tokens_total 32",
            "ignis_retained_state_spills_total{tier=\"kv_ram\",kind=\"prefix\"} 1",
            "ignis_retained_state_spills_total{tier=\"kv_ram\",kind=\"checkpoint\"} 0",
            // A restore is the scheduler's fact, not this call's: nothing here
            // may count one a second time.
            "ignis_retained_state_restores_total{tier=\"kv_ram\",kind=\"checkpoint\"} 0",
        ] {
            assert!(text.contains(&format!("
{line}
")), "{line} in:
{text}");
        }
    }

    #[test]
    fn a_live_provider_overrides_the_estimator() {
        let mut telemetry = telemetry();
        telemetry.with_stats(Arc::new(FixedStats(IntervalCounters {
            waiting: 3,
            prefilling: 2,
            running: 5,
            kv_used_pct: 62,
            kv_evictions: 9,
        })));
        let mut returned = IntervalCounters::default();
        let events = capture_events(|| returned = telemetry.emit_interval(Occupancy::default()));
        let v = intervals(&events).last().copied().expect("an interval event");
        // The provider's counters win over the (empty) event-derived set.
        assert_eq!(v["attributes"]["waiting"], 3);
        assert_eq!(v["attributes"]["running"], 5);
        assert_eq!(v["attributes"]["kv_evictions"], 9);
        // The snapshot carries the provider's whole set, placeholders
        // included; only the log event leaves them out.
        assert_eq!(returned.prefilling, 2);
        assert_eq!(returned.kv_used_pct, 62);
    }

    // ── when a lane frees (GitHub #327, `Retry-After`) ──────────────────────────

    #[test]
    fn no_lane_free_estimate_before_a_request_has_finished() {
        let clock = ManualClock::at(1_000);
        let mut telemetry = Telemetry::new(clock.clone());
        telemetry.note_submit(1, 5, RequestClass::Interactive);
        clock.set(4_000);
        assert_eq!(telemetry.lane_free_at_ms(), None, "nothing has finished: nothing to go on");
    }

    #[test]
    fn a_lane_is_expected_free_when_the_oldest_request_has_held_it_as_long_as_the_last_did() {
        let clock = ManualClock::at(0);
        let mut telemetry = Telemetry::new(clock.clone());
        telemetry.note_submit(1, 5, RequestClass::Interactive);
        clock.set(3_000);
        telemetry.on_done(1, 5, FinishReason::Stop, None, None, None);
        clock.set(10_000);
        telemetry.note_submit(2, 5, RequestClass::Interactive);
        clock.set(11_000);
        telemetry.note_submit(3, 5, RequestClass::Interactive);
        assert_eq!(telemetry.lane_free_at_ms(), Some(13_000), "request 2, in since 10 s, held as long as 1 did (3 s)");
    }

    /// The hold is smoothed the way TCP smooths its round-trip time (RFC
    /// 6298, alpha 1/8): each finished request moves it an eighth of the way.
    #[test]
    fn the_expected_hold_moves_an_eighth_of_the_way_toward_each_finished_request() {
        let clock = ManualClock::at(0);
        let mut telemetry = Telemetry::new(clock.clone());
        telemetry.note_submit(1, 5, RequestClass::Interactive);
        clock.set(1_000);
        telemetry.on_done(1, 5, FinishReason::Stop, None, None, None); // held 1 s: the first sample
        telemetry.note_submit(2, 5, RequestClass::Interactive);
        clock.set(10_000);
        telemetry.on_done(2, 5, FinishReason::Stop, None, None, None); // held 9 s: 1 + (9 - 1) / 8 = 2 s
        clock.set(20_000);
        telemetry.note_submit(3, 5, RequestClass::Interactive);
        assert_eq!(telemetry.lane_free_at_ms(), Some(22_000));
    }

    /// A cancelled request (its client gone, or its 504) held its lane until
    /// the cancel: that hold counts as one that finished does.
    #[test]
    fn a_cancelled_request_counts_for_how_long_a_lane_is_held() {
        let clock = ManualClock::at(0);
        let mut telemetry = Telemetry::new(clock.clone());
        telemetry.note_submit(1, 5, RequestClass::Interactive);
        clock.set(5_000);
        telemetry.on_cancelled(1, CancelCause::Timeout);
        clock.set(6_000);
        telemetry.note_submit(2, 5, RequestClass::Interactive);
        assert_eq!(telemetry.lane_free_at_ms(), Some(11_000));
    }

    // ── a cancelled request's line (GitHub #327) ────────────────────────────────

    /// A cancel used to leave no `ignis.request.*` line: an orphaned 504's
    /// lane was invisible in the log. Now it says why, how long the request
    /// was in, whether it had a lane and what it generated, under the
    /// request's own `trace_id`, which the 504's HTTP line carries too.
    #[test]
    fn a_cancelled_request_logs_why_how_long_and_what_it_generated() {
        let clock = ManualClock::at(0);
        let mut telemetry = Telemetry::new(clock.clone());
        let events = capture_events(|| {
            telemetry.note_submit(4, 40, RequestClass::Interactive);
            clock.set(300);
            telemetry.on_admitted(4, 7);
            for _ in 0..3 {
                telemetry.on_token(4);
            }
            telemetry.note_submit(5, 10, RequestClass::Agent);
            clock.set(20_000);
            telemetry.on_cancelled(4, CancelCause::Timeout);
            clock.set(25_000);
            telemetry.on_cancelled(5, CancelCause::Consumer); // still queued
        });

        let cancelled: Vec<_> = events.iter().filter(|e| e["event_name"] == "ignis.request.cancelled").collect();
        assert_eq!(cancelled.len(), 2, "{events:?}");
        let running = &cancelled[0]["attributes"];
        assert_eq!(cancelled[0]["trace_id"], "00000000000000000000000000000004");
        assert_eq!(running["request_id"], 4);
        assert_eq!(running["cause"], "timeout");
        assert_eq!(running["duration_ms"], 20_000);
        assert_eq!(running["admitted"], true);
        assert_eq!(running["lane"], 7);
        assert_eq!(running["tokens"], 3);
        assert_eq!(running["class"], "interactive");
        let queued = &cancelled[1]["attributes"];
        assert_eq!(queued["request_id"], 5);
        assert_eq!(queued["cause"], "consumer");
        assert_eq!(queued["duration_ms"], 24_700, "in since 300 ms");
        assert_eq!(queued["admitted"], false);
        assert!(queued.get("lane").is_none(), "no lane was dealt: {queued}");
        assert_eq!(queued["tokens"], 0);
        assert_eq!(queued["class"], "agent");
    }

    /// One line per request: a second cancel (a loop guard and a response
    /// both letting go) and a cancel after the request's own end log
    /// nothing. A `Done` that raced the cancel still logs its `done` line,
    /// after the `cancelled` one: the request finished as the cancel landed.
    #[test]
    fn a_request_logs_one_cancelled_line_and_a_racing_done_still_logs_done() {
        let clock = ManualClock::at(0);
        let mut telemetry = Telemetry::new(clock.clone());
        let events = capture_events(|| {
            telemetry.note_submit(1, 5, RequestClass::Interactive);
            telemetry.on_cancelled(1, CancelCause::Timeout);
            telemetry.on_cancelled(1, CancelCause::Consumer);
            telemetry.on_done(1, 2, FinishReason::Stop, None, None, None);
            telemetry.note_submit(2, 5, RequestClass::Interactive);
            telemetry.on_done(2, 2, FinishReason::Stop, None, None, None);
            telemetry.on_cancelled(2, CancelCause::Consumer);
        });

        let lines: Vec<(&str, u64)> = events
            .iter()
            .filter(|e| e["event_name"] == "ignis.request.cancelled" || e["event_name"] == "ignis.request.done")
            .map(|e| (e["event_name"].as_str().unwrap(), e["attributes"]["request_id"].as_u64().unwrap()))
            .collect();
        assert_eq!(
            lines,
            [("ignis.request.cancelled", 1), ("ignis.request.done", 1), ("ignis.request.done", 2)],
            "{events:?}"
        );
        let cancelled = events.iter().find(|e| e["event_name"] == "ignis.request.cancelled").unwrap();
        assert_eq!(cancelled["attributes"]["cause"], "timeout", "the first cancel's cause");
    }
}
