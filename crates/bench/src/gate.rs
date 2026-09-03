//! The v1 acceptance artifact (spec 02, ADR 0007): the **performance
//! report** (tok-s / ttft vs the reference, per class, the 99% performance
//! gate) + the **divergence report** (the canary self-consistency check),
//! shipped as a single JSON file.
//!
//! ADR 0007: the v1 acceptance is a *performance* gate (≥ 99% of the
//! reference's speed on the trace-replay load, with a per-class ttft /
//! tok-s check) **plus** the self-consistency check (the canary suite:
//! sane output, greedy, fixed seed). It is *not* a token-parity gate. This
//! module composes the two halves into the shipped artifact: the
//! `PerformanceReport` (the per-class 99% verdict) and the canary results
//! (the divergence report), with a single overall verdict.
//!
//! The module is pure (no I/O) so it is testable without a running engine;
//! the endpoint plumbing lives in `client.rs` / `canary.rs`.

use serde::{Deserialize, Serialize};

use crate::canary::{suite_consistent, CanaryResult};
use crate::report::PerformanceReport;

/// The v1 acceptance artifact (spec 02, ADR 0007): the performance report
/// (the 99% gate, per class) + the canary divergence report, shipped as a
/// single JSON file (`ignis-bench gate --out`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateReport {
    /// The performance report + the per-class 99% gate verdict.
    pub performance: PerformanceReport,
    /// The canary (divergence) results, from the suite run against the
    /// engine under test: the self-consistency half of the v1 verdict
    /// (ADR 0007: the gate is the 99% performance gate **and** the
    /// self-consistency check — both are shipped in the artifact).
    pub canary: Vec<CanaryResult>,
}

impl GateReport {
    /// Build the shipped artifact from the performance report and the
    /// canary results (the v1 acceptance is their conjunction — spec 02).
    pub fn new(performance: PerformanceReport, canary: Vec<CanaryResult>) -> Self {
        Self {
            performance,
            canary,
        }
    }

    /// The overall v1 verdict (ADR 0007): the performance gate (≥ 99% of
    /// the reference's speed, per class) **and** the self-consistency
    /// check (sane, greedy-deterministic canary output).
    pub fn passed(&self) -> bool {
        self.performance.gate_passed() && suite_consistent(&self.canary)
    }

    /// A human-readable rendering of the shipped artifact: the performance
    /// report, the divergence report, and the overall v1 verdict.
    pub fn render(&self) -> String {
        let mut out = format!(
            "v1 gate (ADR 0007): {} vs {}\n",
            self.performance.ours, self.performance.reference
        );
        out.push_str("  performance (99% gate, per class):\n");
        // The performance report's own header line is redundant here (the
        // artifact's header already names both runs) — keep the per-class
        // lines and its gate line.
        for line in self.performance.render().lines().skip(1) {
            out.push_str(&format!("    {line}\n"));
        }
        out.push_str("  canary (self-consistency):\n");
        for c in &self.canary {
            out.push_str(&format!(
                "    {:<14} sane={} deterministic={}{}\n",
                c.id,
                c.sane,
                c.deterministic,
                if c.consistent() { "" } else { "  <-- DIVERGENT" },
            ));
        }
        out.push_str(&format!(
            "    self-consistency: {}\n",
            if suite_consistent(&self.canary) { "PASS" } else { "FAIL" }
        ));
        out.push_str(&format!(
            "  v1 gate: {}\n",
            if self.passed() { "PASS" } else { "FAIL" }
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canary::evaluate;
    use crate::metrics::{RequestMetrics, Run};
    use crate::trace::RequestClass;

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

    /// A run pair with both classes. When `slow`, our sub class decodes
    /// far below the reference (the performance gate fails).
    fn run_pair(slow: bool) -> (Run, Run) {
        let reference = Run::new(
            "ninfer",
            vec![
                m("main", RequestClass::Main, 200.0, 512, 6200.0),
                // 63 decode tokens over 1.9 s ~= 33.2 tok/s.
                m("s1", RequestClass::Sub, 100.0, 64, 2000.0),
            ],
        );
        let sub = if slow {
            // 63 decode tokens over 2.9 s ~= 21.7 tok/s — below 99% of the
            // reference's 33.2 tok/s (the performance gate fails).
            m("s1", RequestClass::Sub, 100.0, 64, 3000.0)
        } else {
            m("s1", RequestClass::Sub, 100.0, 64, 2000.0)
        };
        let ours = Run::new(
            "ignis",
            vec![m("main", RequestClass::Main, 200.0, 512, 6200.0), sub],
        );
        (ours, reference)
    }

    fn consistent_canaries() -> Vec<CanaryResult> {
        vec![
            evaluate("rust-hello", "it prints `hi`", "it prints `hi`"),
            evaluate("math-greedy", "the answer is 10", "the answer is 10"),
        ]
    }

    fn divergent_canaries() -> Vec<CanaryResult> {
        vec![
            evaluate("rust-hello", "it prints `hi`", "it prints `hi`"),
            // The two greedy runs disagree (the divergence the suite
            // exists to detect).
            evaluate("math-greedy", "the answer is 10", "the answer is 11"),
        ]
    }

    fn build(slow: bool, canary: Vec<CanaryResult>) -> GateReport {
        let (ours, reference) = run_pair(slow);
        GateReport::new(
            PerformanceReport::new(&ours, &reference),
            canary,
        )
    }

    #[test]
    fn the_v1_gate_passes_when_performance_and_self_consistency_hold() {
        let artifact = build(false, consistent_canaries());
        assert!(artifact.performance.gate_passed(), "the 99% performance gate holds");
        assert!(artifact.passed(), "the v1 gate must pass");
    }

    #[test]
    fn a_divergent_canary_fails_the_v1_gate() {
        // The performance gate holds (identical runs), but one canary is
        // not deterministic -> the self-consistency check fails and the
        // overall v1 verdict is FAIL (ADR 0007: correctness is
        // self-checked, not reference-matched).
        let artifact = build(false, divergent_canaries());
        assert!(
            artifact.performance.gate_passed(),
            "the performance gate still holds"
        );
        assert!(
            !artifact.passed(),
            "a divergent canary must fail the v1 gate"
        );
    }

    #[test]
    fn a_slow_run_fails_the_v1_gate_even_when_canaries_are_consistent() {
        // ADR 0007: the performance gate is the binding constraint here —
        // a sub class below 99% of the reference fails the gate even when
        // the canaries are perfectly consistent.
        let artifact = build(true, consistent_canaries());
        assert!(
            !artifact.performance.gate_passed(),
            "the 99% performance gate must fail"
        );
        assert!(
            !artifact.passed(),
            "the v1 gate fails when the performance gate fails"
        );
    }

    #[test]
    fn the_gate_report_round_trips_through_json() {
        // The shipped artifact is a JSON file on disk: it must survive a
        // round-trip (the `gate --out` artifact is read back on re-runs).
        let artifact = build(false, consistent_canaries());
        let shipped = serde_json::to_string_pretty(&artifact).expect("ship the artifact");
        let back: GateReport = serde_json::from_str(&shipped).expect("parse the artifact");
        assert!(back.passed(), "the verdict survives the round-trip");
        assert!(back.performance.gate_passed(), "the performance gate survives");
        assert_eq!(
            back.canary.len(),
            2,
            "the divergence report survives the round-trip"
        );
        // A FAIL verdict survives too (the shipped artifact records the
        // failure, not just the success).
        let failing = build(true, consistent_canaries());
        let shipped = serde_json::to_string_pretty(&failing).expect("ship the artifact");
        let back: GateReport = serde_json::from_str(&shipped).expect("parse the artifact");
        assert!(!back.passed(), "a failing verdict survives the round-trip");
    }

    #[test]
    fn render_carries_the_overall_verdict() {
        let passing = build(false, consistent_canaries());
        assert!(passing.render().contains("v1 gate: PASS"), "the PASS verdict renders");
        assert!(
            passing.render().contains("canary (self-consistency):"),
            "the divergence report renders"
        );

        let failing = build(true, divergent_canaries());
        let text = failing.render();
        assert!(
            text.contains("v1 gate: FAIL"),
            "the FAIL verdict renders: {text}"
        );
        assert!(
            text.contains("DIVERGENT"),
            "the divergent canary is flagged: {text}"
        );
    }
}