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
//! [`SchedEvent`]s) and reports `prefilling` / `kv_used_pct` as 0 until core
//! exposes a `Scheduler::stats(&self)` accessor — which is why those two
//! never reach the interval event. That accessor is the missing seam this
//! module is built to close.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use ignis_core::checkpoint::{RetainedStateOperation, ReuseSource};
use ignis_core::{FinishReason, LaneId, RequestClass, RequestId, SpecCounters};
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

// ── the telemetry state ─────────────────────────────────────────────────────

/// What a request's media cost, as the `admitted` line reports it: what
/// acquiring them cost (GitHub #179) and what encoding them cost during
/// prefill (GitHub #192), which the two halves of the pipeline measure in
/// different places and only meet here.
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
    /// The request's Lane tag (GitHub #120, `CONTEXT.md`), read off
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
    reuse: Option<(ReuseSource, u32, u64)>,
}

/// The server's telemetry: tracks per-request state and emits the interval +
/// request events.
pub struct Telemetry {
    clock: Arc<dyn TelemetryClock>,
    /// A live counter source (the §5 blocker seam); `None` → event-derived.
    stats: Option<Arc<dyn IntervalStatsProvider>>,
    /// The tick number (per-step counter; the interval event's `tick`).
    tick: u64,
    /// The `(waiting, running, kv_evictions)` the last interval event
    /// carried; `None` before the first tick (ADR 0025: unchanged counters
    /// are not logged again).
    last_logged: Option<(u32, u32, u64)>,
    /// Cumulative evictions to the host KV-RAM tier.
    kv_evictions: u64,
    /// In-flight request telemetry (id → state); removed on completion.
    requests: HashMap<RequestId, RequestTelemetry>,
    /// The Prometheus projection this consumer keeps up to date, when
    /// `--metrics` installed one (GitHub #89, ADR 0017).
    metrics: Option<Arc<Metrics>>,
    /// The latest cancelled requests, kept only while `metrics` is set: a
    /// request can finish in the same step its cancel is sent, and then its
    /// `Done` reaches this consumer after the cancel. It was counted once,
    /// as cancelled, and must not be counted again as completed.
    recently_cancelled: VecDeque<RequestId>,
}

/// How many cancelled ids [`Telemetry`] remembers for that race. A late
/// `Done` was emitted before the model thread handled the cancel — within one
/// scheduler step — so this bound is far beyond any real window.
const RECENTLY_CANCELLED: usize = 1024;

impl Telemetry {
    /// Telemetry over `clock` (`ms` / `tok_s`). No live counter source is
    /// set, so the interval counters are event-derived.
    pub fn new(clock: Arc<dyn TelemetryClock>) -> Self {
        Self {
            clock,
            stats: None,
            tick: 0,
            last_logged: None,
            kv_evictions: 0,
            requests: HashMap::new(),
            metrics: None,
            recently_cancelled: VecDeque::new(),
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

    /// A submitted request carried media (GitHub #179): its acquisition
    /// summary rides on the `admitted` line.
    pub fn note_media(&mut self, id: RequestId, media: MediaStats) {
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.media = Some(media);
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

    /// A request was cancelled — its client went away (GitHub #89). The
    /// scheduler releases a cancelled request without a `Done`, so it leaves
    /// the in-flight set here, or it would count as waiting or running
    /// forever. Events the model thread emitted for it before the cancel may
    /// still arrive afterwards; none of them re-adds it, and a late `Done`
    /// does not count it a second time, as completed.
    pub fn on_cancelled(&mut self, id: RequestId) {
        if self.requests.remove(&id).is_none() {
            return;
        }
        let Some(metrics) = &self.metrics else {
            return;
        };
        metrics.record_cancelled();
        if self.recently_cancelled.len() == RECENTLY_CANCELLED {
            self.recently_cancelled.pop_front();
        }
        self.recently_cancelled.push_back(id);
        let counters = self.counters();
        metrics.set_scheduler_requests(counters.waiting, counters.running);
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
    /// phase's per-lane inter-token-latency summary (P3-06) and its
    /// speculative counters when it ran any rounds (P5-06, GitHub #154) —
    /// and drop the request from the in-flight set.
    pub fn on_done(
        &mut self,
        id: RequestId,
        n: u32,
        reason: FinishReason,
        spec: Option<SpecCounters>,
    ) {
        if let Some(metrics) = &self.metrics {
            match self.recently_cancelled.iter().position(|&c| c == id) {
                // Already counted as cancelled (see `recently_cancelled`).
                Some(i) => {
                    self.recently_cancelled.remove(i);
                }
                None => metrics.record_completed(n),
            }
        }
        let rt = self.requests.remove(&id);
        let known = rt.is_some();
        let (submitted_ms, lane, itl_count, itl_sum_ms, itl_max_ms, class, reuse) = match rt {
            Some(rt) => (
                rt.submitted_ms,
                rt.lane,
                rt.itl_count,
                rt.itl_sum_ms,
                rt.itl_max_ms,
                rt.class,
                rt.reuse,
            ),
            None => (
                self.clock.now_ms(),
                0,
                0,
                0,
                0,
                RequestClass::default(),
                None,
            ),
        };
        let ms = self.clock.now_ms().saturating_sub(submitted_ms);
        // Only a request this consumer saw submitted has a real span; one
        // already counted as cancelled is not observed as completed.
        if known {
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
        );
    }

    /// A request was evicted to the host tier: bump the eviction counter
    /// and emit the `evicted` line (GitHub #125) — the snapshot's wall time
    /// (`snapshot_micros`), so the tier's cost is attributable from the
    /// request log alone, without a second run.
    pub fn on_evicted(&mut self, id: RequestId, snapshot_micros: u64) {
        self.kv_evictions = self.kv_evictions.saturating_add(1);
        if let Some(metrics) = &self.metrics {
            metrics.record_eviction();
        }
        self.emit_evicted(id, snapshot_micros);
    }

    /// A request's prefill skipped `tokens` prompt tokens through a cached
    /// prefix (core-07). Not logged — the request log has no line for it — so
    /// only the installed projection observes it (GitHub #90).
    ///
    /// Two kinds arrive here since GitHub #188: a concurrent sibling's prefix,
    /// and a **retained prefix** left by a request that has already finished.
    /// `retained` tells them apart (#190), so `ignis_prefix_reused_tokens_total`
    /// keeps the *sibling*-prefix meaning ADR 0017's row gives it, and
    /// cross-request reuse is counted as reuse of retained state on the device.
    pub fn on_prefix_reused(&mut self, tokens: u32, retained: bool) {
        if let Some(metrics) = &self.metrics {
            if retained {
                metrics.record_retained_reused(ReuseSource::Device, tokens);
            } else {
                metrics.record_prefix_reused(tokens);
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
        tokens: u32,
        restore_micros: u64,
    ) {
        if let Some(metrics) = &self.metrics {
            metrics.record_retained_reused(source, tokens);
        }
        if let Some(rt) = self.requests.get_mut(&id) {
            rt.reuse = Some((source, tokens, restore_micros));
        }
    }

    /// Something happened to retained state in one tier (GitHub #190). Not
    /// logged: none of it belongs to one request's line, so only the
    /// installed projection observes it.
    pub fn on_retained_state(&mut self, operation: RetainedStateOperation, source: ReuseSource) {
        if let Some(metrics) = &self.metrics {
            metrics.record_retained_state(operation, source);
        }
    }

    /// A request was restored from the host tier onto a decode lane: emit
    /// the `restored` line (GitHub #125) — the restore's wall time
    /// (`restore_micros`), the other half of the tier's own cost
    /// attribution alongside [`Telemetry::on_evicted`].
    pub fn on_restored(&mut self, id: RequestId, restore_micros: u64) {
        self.emit_restored(id, restore_micros);
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
            // `kv_used_pct` needs the scheduler's KV pool state, which the
            // `Scheduler` trait does not expose — 0 until core exposes it.
            kv_used_pct: 0,
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
    /// logging queue's drop-oldest buffer. `prefilling` / `kv_used_pct` are
    /// placeholder zeros (see the module doc), so they are not attributes.
    pub fn emit_interval(&mut self) -> IntervalCounters {
        self.tick = self.tick.saturating_add(1);
        let counters = self.counters();
        if let Some(metrics) = &self.metrics {
            metrics.set_scheduler_requests(counters.waiting, counters.running);
        }
        let logged = (counters.waiting, counters.running, counters.kv_evictions);
        if self.last_logged != Some(logged) {
            self.last_logged = Some(logged);
            tracing::debug!(
                name: "ignis.scheduler.interval",
                tick = self.tick,
                waiting = counters.waiting,
                running = counters.running,
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
        reuse: Option<(ReuseSource, u32, u64)>,
    ) {
        let _span = tracing::info_span!("ignis.telemetry.emit", request_id = id).entered();
        // A `None` field records nothing, so a request without speculative
        // rounds carries no `spec.*` attributes at all.
        let spec_pos = spec.map(|s| s.acceptance_profile());
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
            reuse_source = reuse.map(|(source, _, _)| source.as_str()),
            reused_prompt_tokens = reuse.map(|(_, tokens, _)| tokens),
            restore_ms = reuse.map(|(_, _, micros)| micros as f64 / 1000.0),
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

    #[test]
    fn an_interval_event_carries_the_authoritative_counters_at_debug() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.emit_interval();
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
        // Placeholder zeros are not facts (ADR 0025, after ADR 0017).
        assert!(attributes.get("prefilling").is_none(), "{interval}");
        assert!(attributes.get("kv_used_pct").is_none(), "{interval}");
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
                returned.push(telemetry.emit_interval());
            }
            telemetry.on_done(1, 3, FinishReason::Stop, None);
            returned.push(telemetry.emit_interval());
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
            telemetry.on_done(7, 4, FinishReason::Stop, None);
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
            media_bytes: 4096,
            preprocess_seconds: 0.25,
            cache_hits: 1,
            cache_misses: 1,
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
    fn a_fixed_clock_keeps_request_events_deterministic() {
        let mut telemetry = telemetry();
        let events = capture_events(|| {
            telemetry.note_submit(1, 3, RequestClass::Interactive);
            telemetry.on_admitted(1, 0);
            telemetry.on_done(1, 3, FinishReason::Stop, None);
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
            telemetry.on_done(1, 5, FinishReason::Length, None); // read #2 → 200 ms, so ms = 100
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
            telemetry.emit_interval();
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
            telemetry.emit_interval();
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
            telemetry.on_done(1, 3, FinishReason::Stop, None);
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
            telemetry.on_done(1, 3, FinishReason::Stop, None); // t=220 (clock re-read), lane 0 finishes
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
        telemetry.emit_interval();
        has("ignis_requests_accepted_total 2");
        has("ignis_scheduler_requests{state=\"waiting\"} 1");
        has("ignis_scheduler_requests{state=\"running\"} 1");
        has("ignis_requests_completed_total 0");

        telemetry.on_token(2);
        telemetry.on_done(2, 5, FinishReason::Stop, None);
        telemetry.emit_interval();
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

        telemetry.on_done(1, 3, FinishReason::Stop, None);
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

        telemetry.on_cancelled(1);
        let text = metrics.render();
        assert!(text.contains("\nignis_scheduler_requests{state=\"waiting\"} 0\n"), "{text}");
        assert!(text.contains("\nignis_scheduler_requests{state=\"running\"} 1\n"), "{text}");

        telemetry.on_admitted(1, 1); // emitted before the cancel, routed after
        telemetry.on_cancelled(2);
        telemetry.on_cancelled(2); // a second cancel is a no-op
        let counters = telemetry.emit_interval();
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
        telemetry.on_cancelled(1);
        telemetry.on_done(1, 4, FinishReason::Stop, None);
        // A `Done` for a request that was never cancelled still counts.
        telemetry.note_submit(2, 3, RequestClass::Interactive);
        telemetry.on_done(2, 3, FinishReason::Length, None);

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
        telemetry.on_prefix_reused(32, false);
        telemetry.on_evicted(1, 450);
        telemetry.on_token(1);
        telemetry.on_token(1);
        telemetry.on_done(1, 2, FinishReason::Stop, None);

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

    /// A cancelled request never completed: its late `Done` is not a
    /// request-duration observation, and one never seen submitted has no span.
    #[test]
    fn cancelled_and_unknown_requests_are_not_observed_as_durations() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(1, 3, RequestClass::Interactive);
        telemetry.on_cancelled(1);
        telemetry.on_done(1, 4, FinishReason::Stop, None);
        telemetry.on_done(99, 4, FinishReason::Stop, None);

        let text = metrics.render();
        assert!(text.contains("\nignis_request_duration_seconds_count 0\n"), "{text}");
    }

    #[test]
    fn reuse_is_counted_by_kind_and_tier_never_summed_into_the_sibling_counter() {
        let metrics = Arc::new(Metrics::new());
        let mut telemetry = telemetry();
        telemetry.with_metrics(Arc::clone(&metrics));
        telemetry.note_submit(7, 2048, RequestClass::Interactive);
        telemetry.on_state_reused(7, ReuseSource::KvRam, 1536, 42);
        telemetry.on_prefix_reused(64, true);
        telemetry.on_prefix_reused(32, false);
        telemetry.on_retained_state(RetainedStateOperation::Spill, ReuseSource::KvRam);

        let text = metrics.render();
        for line in [
            "ignis_retained_reused_tokens_total{tier=\"kv_ram\"} 1536",
            "ignis_retained_reused_tokens_total{tier=\"device\"} 64",
            "ignis_prefix_reused_tokens_total 32",
            "ignis_retained_state_spills_total{tier=\"kv_ram\"} 1",
            // A restore is the scheduler's fact, not this call's: nothing here
            // may count one a second time.
            "ignis_retained_state_restores_total{tier=\"kv_ram\"} 0",
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
        let events = capture_events(|| returned = telemetry.emit_interval());
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
}
