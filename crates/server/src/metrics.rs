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

/// The exposition's content type: Prometheus text format 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The aggregate the telemetry consumer maintains while metrics are on.
#[derive(Debug, Default)]
pub struct Metrics {
    waiting: AtomicU64,
    running: AtomicU64,
    accepted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    generated_tokens: AtomicU64,
}

impl Metrics {
    /// An all-zero projection.
    pub fn new() -> Self {
        Self::default()
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
        ];
        for (name, help, series) in counters {
            declare(&mut out, name, "counter", help);
            let _ = writeln!(out, "{name} {}", read(series));
        }
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
                .position(|l| l.starts_with(&format!("{name} ")) || l.starts_with(&format!("{name}{{")))
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
        ] {
            assert_eq!(value(&text, name, ""), "0", "{name}");
        }
    }

    #[test]
    fn recorded_facts_move_their_series() {
        let metrics = Metrics::new();
        metrics.record_accepted();
        metrics.record_accepted();
        metrics.record_completed(7);
        metrics.record_completed(5);
        metrics.record_cancelled();
        metrics.set_scheduler_requests(3, 4);
        metrics.set_scheduler_requests(1, 2);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_requests_accepted_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_completed_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_cancelled_total", ""), "1");
        assert_eq!(value(&text, "ignis_generated_tokens_total", ""), "12");
        // Gauges are the latest state, not a sum.
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "2");
    }

    #[test]
    fn a_label_value_is_escaped_per_the_text_format() {
        assert_eq!(escape_label_value(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape_label_value("a\nb"), r"a\nb");
    }
}
