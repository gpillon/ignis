//! Per-request metrics + aggregation for the "1 main + N subagents" load.
//!
//! A `Run` is the set of per-request metrics from one harness run (the ignis
//! run or the reference run). The gate (ADR 0007) compares our run against
//! the reference run *per class* (main vs subagent), so the aggregation keeps
//! classes separate — a single pooled number would hide a fast main and a slow
//! subagent (or the reverse).
//!
//! Timing model (streaming request):
//!   - `ttft_ms` : request start -> first token (the prefill phase).
//!   - `total_ms`: request start -> last token.
//!   - decode    : (n_tokens - 1) tokens produced over (total_ms - ttft_ms).
//! The aggregate decode speed is throughput-weighted (total decoded tokens
//! over total decode time), so a single fast outlier cannot inflate it.

use serde::{Deserialize, Serialize};

use crate::trace::RequestClass;

/// Per-request metrics for a single request in a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Stable request id (matches the trace line and the reference run).
    pub id: String,
    /// Class: main agent vs subagent (per-class gate, ADR 0007).
    pub class: RequestClass,
    /// Time to first token (ms). For a streaming request this is the prefill
    /// latency; for a non-streaming request it equals `total_ms`.
    pub ttft_ms: f64,
    /// Total number of tokens generated.
    pub n_tokens: u32,
    /// Wall-clock duration of the whole request (ms).
    pub total_ms: f64,
    /// Whether the request completed normally (false on a mid-stream error).
    pub ok: bool,
}

impl RequestMetrics {
    /// Decode-phase throughput in tokens/second:
    /// `(n_tokens - 1) / ((total_ms - ttft_ms) / 1000)`.
    ///
    /// The first token is produced by the prefill (captured by `ttft_ms`);
    /// every token after that is a decode token. Returns 0.0 when there is no
    /// decode phase (a single token, or a non-streaming request where
    /// `total_ms == ttft_ms`).
    pub fn tok_s(&self) -> f64 {
        let decode_ms = self.total_ms - self.ttft_ms;
        if decode_ms <= 0.0 || self.n_tokens <= 1 {
            return 0.0;
        }
        (self.n_tokens as f64 - 1.0) / (decode_ms / 1000.0)
    }
}

/// Aggregated metrics for one class (or, with `class == Any`, the whole run).
///
/// `tok_s` here is the *throughput-weighted* decode speed for the class:
/// total decoded tokens over total decode time, which is the honest aggregate
/// (it matches what the GPU actually produced per second).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassStats {
    pub class: RequestClass,
    pub n_requests: u32,
    pub total_tokens: u64,
    /// Throughput-weighted decode speed (tokens/s), 0.0 if no decode phase.
    pub tok_s: f64,
    /// p50 / p95 / p99 of per-request ttft_ms (the latency percentiles the
    /// report and the gate check).
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
}

/// One harness run: a label plus the per-request metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub label: String,
    pub metrics: Vec<RequestMetrics>,
}

impl Run {
    /// Build a run from a label and the collected per-request metrics.
    pub fn new(label: impl Into<String>, metrics: Vec<RequestMetrics>) -> Self {
        Self {
            label: label.into(),
            metrics,
        }
    }

    /// The per-class stats for a class present in the run.
    pub fn stats_for(&self, class: RequestClass) -> Option<ClassStats> {
        let items: Vec<RequestMetrics> = self
            .metrics
            .iter()
            .filter(|m| m.class == class)
            .cloned()
            .collect();
        if items.is_empty() {
            return None;
        }
        Some(class_stats(class, &items))
    }

    /// The classes present in the run, in a stable order (main, sub).
    pub fn classes(&self) -> Vec<RequestClass> {
        let mut out: Vec<RequestClass> = self
            .metrics
            .iter()
            .map(|m| m.class)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // BTreeSet on the enum (by discriminant): Main (0), Sub (1).
        out.sort_by_key(|c| *c as u8);
        out
    }
}

/// Compute the `ClassStats` for a set of same-class request metrics.
pub fn class_stats(class: RequestClass, items: &[RequestMetrics]) -> ClassStats {
    let n_requests = items.len() as u32;
    let total_tokens: u64 = items.iter().map(|m| m.n_tokens as u64).sum();

    let decoded: f64 = items
        .iter()
        .map(|m| m.n_tokens.saturating_sub(1) as f64)
        .sum();
    let decode_s: f64 = items
        .iter()
        .map(|m| (m.total_ms - m.ttft_ms).max(0.0) / 1000.0)
        .sum();
    let tok_s = if decode_s > 0.0 && decoded > 0.0 {
        decoded / decode_s
    } else {
        0.0
    };

    let mut ttfts: Vec<f64> = items.iter().map(|m| m.ttft_ms).collect();
    ttfts.sort_by(f64::total_cmp);

    ClassStats {
        class,
        n_requests,
        total_tokens,
        tok_s,
        ttft_p50_ms: percentile(&ttfts, 50.0),
        ttft_p95_ms: percentile(&ttfts, 95.0),
        ttft_p99_ms: percentile(&ttfts, 99.0),
    }
}

/// Linear-interpolated percentile of a *sorted* slice (p in [0, 100]).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(id: &str, class: RequestClass, ttft: f64, n: u32, total: f64) -> RequestMetrics {
        RequestMetrics {
            id: id.into(),
            class,
            ttft_ms: ttft,
            n_tokens: n,
            total_ms: total,
            ok: true,
        }
    }

    #[test]
    fn tok_s_counts_only_decode_tokens() {
        // 10 tokens, 200 ms prefill, 1000 ms total -> 9 decode tokens over
        // 0.8 s = 11.25 tok/s.
        let req = m("r", RequestClass::Sub, 200.0, 10, 1000.0);
        assert!((req.tok_s() - 11.25).abs() < 1e-9);
    }

    #[test]
    fn single_token_has_no_decode() {
        let req = m("r", RequestClass::Sub, 200.0, 1, 200.0);
        assert_eq!(req.tok_s(), 0.0);
    }

    #[test]
    fn non_streaming_reports_zero_decode_speed() {
        // ttft == total (no measurable decode phase).
        let req = m("r", RequestClass::Main, 500.0, 64, 500.0);
        assert_eq!(req.tok_s(), 0.0);
    }

    #[test]
    fn class_stats_are_throughput_weighted() {
        // Two requests: 10 tok (0.8 s decode) and 20 tok (1.0 s decode).
        // Weighted: (9 + 19) / (0.8 + 1.0) = 28 / 1.8 = 15.555...
        let items = vec![
            m("a", RequestClass::Sub, 200.0, 10, 1000.0),
            m("b", RequestClass::Sub, 300.0, 20, 1300.0),
        ];
        let s = class_stats(RequestClass::Sub, &items);
        assert_eq!(s.n_requests, 2);
        assert_eq!(s.total_tokens, 30);
        assert!((s.tok_s - 28.0 / 1.8).abs() < 1e-6, "got {}", s.tok_s);
    }

    #[test]
    fn ttft_percentiles_are_computed() {
        let items = (0..100)
            .map(|i| m(&format!("r{i}"), RequestClass::Sub, (i + 1) as f64, 10, 100.0))
            .collect::<Vec<_>>();
        let s = class_stats(RequestClass::Sub, &items);
        // Linear interpolation (numpy "linear") over values 1..=100:
        // p50 = 50.5, p95 = 95.05, p99 = 99.01.
        assert!((s.ttft_p50_ms - 50.5).abs() < 1e-6);
        assert!((s.ttft_p95_ms - 95.05).abs() < 1e-6);
        assert!((s.ttft_p99_ms - 99.01).abs() < 1e-6);
    }

    #[test]
    fn run_groups_metrics_by_class() {
        let run = Run::new(
            "ours",
            vec![
                m("main", RequestClass::Main, 100.0, 512, 6000.0),
                m("s1", RequestClass::Sub, 80.0, 128, 4000.0),
                m("s2", RequestClass::Sub, 90.0, 64, 2000.0),
            ],
        );
        assert!(run.stats_for(RequestClass::Main).is_some());
        let sub = run.stats_for(RequestClass::Sub).expect("sub stats");
        assert_eq!(sub.n_requests, 2);
        assert_eq!(run.classes(), vec![RequestClass::Main, RequestClass::Sub]);
    }
}