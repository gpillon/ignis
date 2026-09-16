//! Prometheus metrics (GitHub #89, ADR 0017): the opt-in exposition
//! `ignis-server` serves with `--metrics` — at `GET /metrics` on its own
//! listener, and at `GET /ui/metrics` on the API listener for the Playground.
//!
//! [`Metrics`] is an aggregate projection of facts the model thread already
//! sends to the asynchronous telemetry consumer (`engine.rs`'s
//! `telemetry_task`). That consumer is its only writer; the HTTP task only
//! reads it, and all text encoding happens there, at scrape time. Nothing in
//! the scheduler, the model thread, the runtime or the kernel leaf knows it
//! exists, and without `--metrics` it is never built.
//!
//! Fixed atomics rather than a lock: a scrape can never make the consumer
//! wait, and the consumer can never make a scrape wait. A scrape therefore
//! reads each series on its own, not one consistent cut across all of them —
//! which Prometheus does not assume either.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::http::header;
use axum::routing::get;
use ignis_core::checkpoint::{RetainedStateOperation, ReuseSource};
use ignis_core::SubmitError;

/// The exposition's content type: Prometheus text format 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// `ignis_request_ttft_seconds`' bucket boundaries (ADR 0017), in
/// milliseconds — the telemetry clock's unit.
const TTFT_BOUNDS_MS: [u64; 12] =
    [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000];

/// `ignis_request_duration_seconds`' bucket boundaries (ADR 0017), in
/// milliseconds.
const DURATION_BOUNDS_MS: [u64; 12] =
    [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000];

/// Why a submission was rejected: `ignis_requests_rejected_total`'s fixed
/// `reason` set (ADR 0017).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The engine could not admit it right now.
    Full,
    /// It named a model the engine does not load.
    UnknownModel,
    /// It can never fit, however empty the engine is.
    Oversized,
}

impl Rejection {
    /// The reason a submit error counts under. A request longer than the
    /// per-sequence context (GitHub #166, after ADR 0017's table) is a
    /// request that can never fit, like one larger than the KV pool.
    pub fn of(err: &SubmitError) -> Self {
        match err {
            SubmitError::Full => Self::Full,
            SubmitError::UnknownModel(_) => Self::UnknownModel,
            SubmitError::Oversized | SubmitError::ContextExceeded { .. } => Self::Oversized,
        }
    }

    const ALL: [Rejection; 3] = [Self::Full, Self::UnknownModel, Self::Oversized];

    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::UnknownModel => "unknown_model",
            Self::Oversized => "oversized",
        }
    }
}

/// A fixed-bucket histogram over millisecond observations.
#[derive(Debug)]
struct Histogram {
    bounds_ms: &'static [u64; 12],
    /// Observations per bucket, not cumulative; the last is `+Inf`'s own.
    buckets: [AtomicU64; 13],
    sum_ms: AtomicU64,
}

impl Histogram {
    fn new(bounds_ms: &'static [u64; 12]) -> Self {
        Self { bounds_ms, buckets: Default::default(), sum_ms: AtomicU64::new(0) }
    }

    fn observe(&self, ms: u64) {
        let bucket = self.bounds_ms.iter().position(|&bound| ms <= bound).unwrap_or(self.bounds_ms.len());
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Cumulative `_bucket` lines, then `_sum` and `_count`. The count is the
    /// `+Inf` bucket as read here, so the two always agree.
    fn render(&self, out: &mut String, name: &str, help: &str) {
        declare(out, name, "histogram", help);
        let mut cumulative = 0;
        for (bucket, count) in self.buckets.iter().enumerate() {
            cumulative += count.load(Ordering::Relaxed);
            let le = match self.bounds_ms.get(bucket) {
                Some(&bound) => seconds(bound),
                None => "+Inf".to_owned(),
            };
            let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {cumulative}");
        }
        let _ = writeln!(out, "{name}_sum {}", seconds(self.sum_ms.load(Ordering::Relaxed)));
        let _ = writeln!(out, "{name}_count {cumulative}");
    }
}

/// `ms` milliseconds as a decimal number of seconds, without trailing zeros.
fn seconds(ms: u64) -> String {
    let (whole, frac) = (ms / 1000, ms % 1000);
    if frac == 0 {
        return whole.to_string();
    }
    let frac = format!("{frac:03}");
    format!("{whole}.{}", frac.trim_end_matches('0'))
}

/// The aggregate the telemetry consumer maintains while metrics are on —
/// except the rejections, which the HTTP handler records after the submit
/// call returns its error (ADR 0017).
#[derive(Debug)]
pub struct Metrics {
    waiting: AtomicU64,
    running: AtomicU64,
    accepted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    rejected: [AtomicU64; 3],
    generated_tokens: AtomicU64,
    decoded_tokens: AtomicU64,
    kv_evictions: AtomicU64,
    prefix_reused_tokens: AtomicU64,
    /// Per [`ReuseSource::index`], for each of the families below (#190).
    retained_reused_tokens: [AtomicU64; ReuseSource::ALL.len()],
    retained_state_hits: [AtomicU64; ReuseSource::ALL.len()],
    retained_state_misses: [AtomicU64; ReuseSource::ALL.len()],
    retained_state_spills: [AtomicU64; ReuseSource::ALL.len()],
    retained_state_discards: [AtomicU64; ReuseSource::ALL.len()],
    retained_state_restores: [AtomicU64; ReuseSource::ALL.len()],
    ttft: Histogram,
    duration: Histogram,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// An all-zero projection.
    pub fn new() -> Self {
        Self {
            waiting: AtomicU64::new(0),
            running: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            rejected: Default::default(),
            generated_tokens: AtomicU64::new(0),
            decoded_tokens: AtomicU64::new(0),
            kv_evictions: AtomicU64::new(0),
            prefix_reused_tokens: AtomicU64::new(0),
            retained_reused_tokens: Default::default(),
            retained_state_hits: Default::default(),
            retained_state_misses: Default::default(),
            retained_state_spills: Default::default(),
            retained_state_discards: Default::default(),
            retained_state_restores: Default::default(),
            ttft: Histogram::new(&TTFT_BOUNDS_MS),
            duration: Histogram::new(&DURATION_BOUNDS_MS),
        }
    }

    /// A submission was rejected, for `reason`.
    pub fn record_rejected(&self, reason: Rejection) {
        // `ALL` lists the reasons in declaration order, so a reason's
        // discriminant is its slot.
        self.rejected[reason as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A request was evicted to the host KV-RAM tier.
    pub(crate) fn record_eviction(&self) {
        self.kv_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// A request's prefill skipped `tokens` through a sibling's prefix.
    pub(crate) fn record_prefix_reused(&self, tokens: u32) {
        self.prefix_reused_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A request's prefill skipped `tokens` through retained state in
    /// `source` — a retained prefix, or a prompt checkpoint (GitHub #190).
    pub(crate) fn record_retained_reused(&self, source: ReuseSource, tokens: u32) {
        self.retained_reused_tokens[source.index()].fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// One retained-state lifecycle operation in its residency tier (GitHub
    /// #190). The telemetry consumer is the only writer.
    pub(crate) fn record_retained_state(&self, operation: RetainedStateOperation, source: ReuseSource) {
        let series = match operation {
            RetainedStateOperation::Hit => &self.retained_state_hits,
            RetainedStateOperation::Miss => &self.retained_state_misses,
            RetainedStateOperation::Spill => &self.retained_state_spills,
            RetainedStateOperation::Discard => &self.retained_state_discards,
            RetainedStateOperation::Restore => &self.retained_state_restores,
        };
        series[source.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// A request's first token came `ms` after its submission.
    pub(crate) fn observe_ttft_ms(&self, ms: u64) {
        self.ttft.observe(ms);
    }

    /// A request completed `ms` after its submission.
    pub(crate) fn observe_duration_ms(&self, ms: u64) {
        self.duration.observe(ms);
    }

    /// A submission was accepted by the scheduler.
    pub(crate) fn record_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    /// A request completed, having generated `tokens`.
    pub(crate) fn record_completed(&self, tokens: u32) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.generated_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A request in flight was dealt one generated token (GitHub #165): the
    /// live counterpart of the tokens `record_completed` adds only at the end.
    pub(crate) fn record_decoded_token(&self) {
        self.decoded_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// An accepted request was cancelled before it completed (its client
    /// went away).
    pub(crate) fn record_cancelled(&self) {
        self.cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// The scheduler's current request counts by observable state.
    pub(crate) fn set_scheduler_requests(&self, waiting: u32, running: u32) {
        self.waiting.store(u64::from(waiting), Ordering::Relaxed);
        self.running.store(u64::from(running), Ordering::Relaxed);
    }

    /// The Prometheus text exposition of the latest projection.
    pub fn render(&self) -> String {
        let read = |series: &AtomicU64| series.load(Ordering::Relaxed);
        let mut out = String::with_capacity(1024);
        declare(&mut out, "ignis_build_info", "gauge", "Constant build identity with value 1.");
        let _ = writeln!(
            out,
            "ignis_build_info{{version=\"{}\"}} 1",
            escape_label_value(env!("CARGO_PKG_VERSION"))
        );
        declare(
            &mut out,
            "ignis_scheduler_requests",
            "gauge",
            "Current requests by observable scheduler state.",
        );
        let _ = writeln!(out, "ignis_scheduler_requests{{state=\"waiting\"}} {}", read(&self.waiting));
        let _ = writeln!(out, "ignis_scheduler_requests{{state=\"running\"}} {}", read(&self.running));
        let counters = [
            ("ignis_requests_accepted_total", "Accepted submissions.", &self.accepted),
            ("ignis_requests_completed_total", "Completed requests.", &self.completed),
            (
                "ignis_requests_cancelled_total",
                "Accepted requests cancelled before completion.",
                &self.cancelled,
            ),
            (
                "ignis_generated_tokens_total",
                "Generated tokens on completed requests.",
                &self.generated_tokens,
            ),
            (
                "ignis_decoded_tokens_total",
                "Tokens generated so far, counted as each one is emitted.",
                &self.decoded_tokens,
            ),
            ("ignis_kv_cache_evictions_total", "Cumulative host-tier evictions.", &self.kv_evictions),
            // Sibling-prefix reuse only, as ADR 0017's table row says. A
            // retained prefix is claimed through the same path (#188), and
            // `SchedEvent::PrefixReused::retained` is what keeps its tokens
            // out of here and in `ignis_retained_reused_tokens_total` (#190).
            (
                "ignis_prefix_reused_tokens_total",
                "Cumulative tokens skipped through sibling-prefix reuse.",
                &self.prefix_reused_tokens,
            ),
        ];
        for (name, help, series) in counters {
            declare(&mut out, name, "counter", help);
            let _ = writeln!(out, "{name} {}", read(series));
        }
        for (name, help, series) in [
            (
                "ignis_retained_reused_tokens_total",
                "Cumulative tokens skipped through retained state, by residency tier.",
                &self.retained_reused_tokens,
            ),
            (
                "ignis_retained_state_hits_total",
                "Retained state chosen to resume from or brought back, by residency tier.",
                &self.retained_state_hits,
            ),
            (
                "ignis_retained_state_misses_total",
                "First prefill chunks with no retained checkpoint matching in the tier.",
                &self.retained_state_misses,
            ),
            (
                "ignis_retained_state_spills_total",
                "Retained checkpoints and prefixes spilled into the tier.",
                &self.retained_state_spills,
            ),
            (
                "ignis_retained_state_discards_total",
                "Retained checkpoints and prefixes discarded from the tier.",
                &self.retained_state_discards,
            ),
            (
                "ignis_retained_state_restores_total",
                "Retained state restored from the tier.",
                &self.retained_state_restores,
            ),
        ] {
            declare(&mut out, name, "counter", help);
            for (source, slot) in ReuseSource::ALL.iter().zip(series) {
                let _ = writeln!(out, "{name}{{tier=\"{}\"}} {}", source.as_str(), read(slot));
            }
        }
        declare(
            &mut out,
            "ignis_requests_rejected_total",
            "counter",
            "Rejected submissions by fixed reason.",
        );
        for (reason, series) in Rejection::ALL.iter().zip(&self.rejected) {
            let _ = writeln!(out, "ignis_requests_rejected_total{{reason=\"{}\"}} {}", reason.label(), read(series));
        }
        self.ttft.render(&mut out, "ignis_request_ttft_seconds", "Submission-to-first-token latency.");
        self.duration.render(&mut out, "ignis_request_duration_seconds", "Submission-to-completion latency.");
        out
    }
}

/// A metric's `HELP` and `TYPE` lines.
fn declare(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

/// A label value escaped for the text format: backslash, double quote and
/// line feed.
fn escape_label_value(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', "\\\"").replace('\n', r"\n")
}

/// `GET path` over `metrics`: `/metrics` for the metrics listener,
/// `/ui/metrics` for the Playground's copy on the API listener.
pub fn router<S>(path: &str, metrics: Arc<Metrics>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route(
        path,
        get(move || async move { ([(header::CONTENT_TYPE, CONTENT_TYPE)], metrics.render()) }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sample lines of `text`: `(name, labels, value)`, comments skipped.
    fn samples(text: &str) -> Vec<(String, String, String)> {
        text.lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let (series, value) = line.rsplit_once(' ').expect("`series value`");
                let (name, labels) = match series.split_once('{') {
                    Some((name, labels)) => (name, labels.trim_end_matches('}')),
                    None => (series, ""),
                };
                (name.to_owned(), labels.to_owned(), value.to_owned())
            })
            .collect()
    }

    fn value(text: &str, name: &str, labels: &str) -> String {
        samples(text)
            .into_iter()
            .find(|(n, l, _)| n == name && l == labels)
            .unwrap_or_else(|| panic!("no `{name}{{{labels}}}` in:\n{text}"))
            .2
    }

    #[test]
    fn every_metric_is_declared_once_with_help_and_type_before_its_samples() {
        let text = Metrics::new().render();
        let expected = [
            ("ignis_build_info", "gauge"),
            ("ignis_scheduler_requests", "gauge"),
            ("ignis_requests_accepted_total", "counter"),
            ("ignis_requests_completed_total", "counter"),
            ("ignis_requests_cancelled_total", "counter"),
            ("ignis_generated_tokens_total", "counter"),
            ("ignis_decoded_tokens_total", "counter"),
            ("ignis_kv_cache_evictions_total", "counter"),
            ("ignis_prefix_reused_tokens_total", "counter"),
            ("ignis_retained_reused_tokens_total", "counter"),
            ("ignis_retained_state_hits_total", "counter"),
            ("ignis_retained_state_misses_total", "counter"),
            ("ignis_retained_state_spills_total", "counter"),
            ("ignis_retained_state_discards_total", "counter"),
            ("ignis_retained_state_restores_total", "counter"),
            ("ignis_requests_rejected_total", "counter"),
            ("ignis_request_ttft_seconds", "histogram"),
            ("ignis_request_duration_seconds", "histogram"),
        ];
        let lines: Vec<&str> = text.lines().collect();
        for (name, kind) in expected {
            let help = lines
                .iter()
                .position(|l| l.starts_with(&format!("# HELP {name} ")))
                .unwrap_or_else(|| panic!("no HELP for {name}:\n{text}"));
            assert_eq!(lines[help + 1], format!("# TYPE {name} {kind}"), "{text}");
            let first_sample = lines
                .iter()
                .position(|l| {
                    l.starts_with(&format!("{name} "))
                        || l.starts_with(&format!("{name}{{"))
                        || (kind == "histogram" && l.starts_with(&format!("{name}_bucket{{")))
                })
                .unwrap_or_else(|| panic!("no sample for {name}:\n{text}"));
            assert!(first_sample > help + 1, "{name}'s samples follow its TYPE:\n{text}");
            assert_eq!(
                lines.iter().filter(|l| l.starts_with(&format!("# TYPE {name} "))).count(),
                1,
                "{text}"
            );
        }
        assert!(text.ends_with('\n'), "the exposition ends with a line feed");
    }

    #[test]
    fn a_fresh_projection_reports_build_identity_and_zeros() {
        let text = Metrics::new().render();
        let version = format!("version=\"{}\"", env!("CARGO_PKG_VERSION"));
        assert_eq!(value(&text, "ignis_build_info", &version), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "0");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "0");
        for name in [
            "ignis_requests_accepted_total",
            "ignis_requests_completed_total",
            "ignis_requests_cancelled_total",
            "ignis_generated_tokens_total",
            "ignis_decoded_tokens_total",
            "ignis_kv_cache_evictions_total",
            "ignis_prefix_reused_tokens_total",
        ] {
            assert_eq!(value(&text, name, ""), "0", "{name}");
        }
        for reason in ["full", "unknown_model", "oversized"] {
            assert_eq!(value(&text, "ignis_requests_rejected_total", &format!("reason=\"{reason}\"")), "0");
        }
    }

    #[test]
    fn a_submit_error_counts_under_its_fixed_reason() {
        assert_eq!(Rejection::of(&SubmitError::Full), Rejection::Full);
        assert_eq!(Rejection::of(&SubmitError::UnknownModel("x".into())), Rejection::UnknownModel);
        assert_eq!(Rejection::of(&SubmitError::Oversized), Rejection::Oversized);
        assert_eq!(
            Rejection::of(&SubmitError::ContextExceeded { requested: 9000, limit: 8192 }),
            Rejection::Oversized
        );
    }

    #[test]
    fn seconds_are_rendered_without_trailing_zeros() {
        assert_eq!(seconds(0), "0");
        assert_eq!(seconds(50), "0.05");
        assert_eq!(seconds(2_500), "2.5");
        assert_eq!(seconds(600_000), "600");
        assert_eq!(seconds(1_001), "1.001");
    }

    #[test]
    fn recorded_facts_move_their_series() {
        let metrics = Metrics::new();
        metrics.record_accepted();
        metrics.record_accepted();
        metrics.record_completed(7);
        metrics.record_completed(5);
        metrics.record_cancelled();
        metrics.record_decoded_token();
        metrics.record_decoded_token();
        metrics.set_scheduler_requests(3, 4);
        metrics.set_scheduler_requests(1, 2);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_requests_accepted_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_completed_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_cancelled_total", ""), "1");
        assert_eq!(value(&text, "ignis_generated_tokens_total", ""), "12");
        // Decoded tokens are their own series, not derived from completions.
        assert_eq!(value(&text, "ignis_decoded_tokens_total", ""), "2");
        // Gauges are the latest state, not a sum.
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "2");
    }

    #[test]
    fn the_operational_counters_move_with_their_facts() {
        let metrics = Metrics::new();
        metrics.record_eviction();
        metrics.record_eviction();
        metrics.record_prefix_reused(32);
        metrics.record_prefix_reused(64);
        metrics.record_rejected(Rejection::Full);
        metrics.record_rejected(Rejection::Oversized);
        metrics.record_rejected(Rejection::Oversized);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_cache_evictions_total", ""), "2");
        assert_eq!(value(&text, "ignis_prefix_reused_tokens_total", ""), "96");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"full\""), "1");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"unknown_model\""), "0");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"oversized\""), "2");
    }

    #[test]
    fn retained_state_operations_are_counted_per_residency_tier() {
        let metrics = Metrics::new();
        metrics.record_retained_state(RetainedStateOperation::Hit, ReuseSource::Device);
        metrics.record_retained_state(RetainedStateOperation::Miss, ReuseSource::Device);
        metrics.record_retained_state(RetainedStateOperation::Spill, ReuseSource::KvRam);
        metrics.record_retained_state(RetainedStateOperation::Discard, ReuseSource::KvRam);
        metrics.record_retained_state(RetainedStateOperation::Restore, ReuseSource::KvRam);

        let text = metrics.render();
        for (name, tier) in [
            ("ignis_retained_state_hits_total", "device"),
            ("ignis_retained_state_misses_total", "device"),
            ("ignis_retained_state_spills_total", "kv_ram"),
            ("ignis_retained_state_discards_total", "kv_ram"),
            ("ignis_retained_state_restores_total", "kv_ram"),
        ] {
            assert_eq!(value(&text, name, &format!("tier=\"{tier}\"")), "1", "{name}");
        }
        assert_eq!(
            value(&text, "ignis_retained_state_hits_total", "tier=\"kv_ram\""),
            "0"
        );
    }

    /// ADR 0017's fixed boundaries, in seconds, `+Inf` implied.
    const TTFT_BOUNDS: [&str; 12] =
        ["0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300"];
    const DURATION_BOUNDS: [&str; 12] =
        ["0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300", "600"];

    /// The `le` values of `name`'s buckets, in exposition order.
    fn bucket_bounds(text: &str, name: &str) -> Vec<String> {
        samples(text)
            .into_iter()
            .filter(|(n, _, _)| *n == format!("{name}_bucket"))
            .map(|(_, labels, _)| {
                labels.strip_prefix("le=\"").and_then(|l| l.strip_suffix('"')).expect("only `le`").to_owned()
            })
            .collect()
    }

    #[test]
    fn the_histograms_use_exactly_the_adr_s_buckets_and_positive_infinity() {
        let text = Metrics::new().render();
        for (name, bounds) in [
            ("ignis_request_ttft_seconds", TTFT_BOUNDS),
            ("ignis_request_duration_seconds", DURATION_BOUNDS),
        ] {
            let mut expected: Vec<String> = bounds.iter().map(|b| (*b).to_owned()).collect();
            expected.push("+Inf".to_owned());
            assert_eq!(bucket_bounds(&text, name), expected, "{text}");
            assert_eq!(value(&text, &format!("{name}_sum"), ""), "0");
            assert_eq!(value(&text, &format!("{name}_count"), ""), "0");
        }
    }

    #[test]
    fn an_observation_lands_in_every_bucket_at_or_above_it() {
        let metrics = Metrics::new();
        // 50 ms sits on the first TTFT boundary (`le` is inclusive); 250 ms
        // on the third; 400 s is past the last one, so only `+Inf` has it.
        metrics.observe_ttft_ms(50);
        metrics.observe_ttft_ms(250);
        metrics.observe_ttft_ms(400_000);
        metrics.observe_duration_ms(700);

        let text = metrics.render();
        let ttft = |le: &str| value(&text, "ignis_request_ttft_seconds_bucket", &format!("le=\"{le}\""));
        assert_eq!(ttft("0.05"), "1");
        assert_eq!(ttft("0.1"), "1");
        assert_eq!(ttft("0.25"), "2");
        assert_eq!(ttft("300"), "2");
        assert_eq!(ttft("+Inf"), "3");
        assert_eq!(value(&text, "ignis_request_ttft_seconds_count", ""), "3");
        assert_eq!(value(&text, "ignis_request_ttft_seconds_sum", ""), "400.3");

        let duration = |le: &str| value(&text, "ignis_request_duration_seconds_bucket", &format!("le=\"{le}\""));
        assert_eq!(duration("0.5"), "0");
        assert_eq!(duration("1"), "1");
        assert_eq!(duration("+Inf"), "1");
        assert_eq!(value(&text, "ignis_request_duration_seconds_sum", ""), "0.7");
    }

    #[test]
    fn a_label_value_is_escaped_per_the_text_format() {
        assert_eq!(escape_label_value(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape_label_value("a\nb"), r"a\nb");
    }
}
