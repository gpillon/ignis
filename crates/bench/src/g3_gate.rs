//! The **G3 gate check** (P3-07, GitHub #100): two G3 records in, one
//! verdict out.
//!
//! Spec: `.scratch/runtime/specs/03-serving-loop.md`; ADR 0015. Mirrors
//! [`crate::g2`]'s shape and its live/live discipline exactly: a verdict is
//! only computed from two records sharing a session, with every cold-prefix
//! obligation satisfied on both sides — otherwise [`check`] returns a
//! [`Refusal`], never a failing verdict, because the tool enforces
//! live/live rather than trusting the operator's discipline.
//!
//! Three per-cell checks, all thresholds a *floor*:
//!
//! - **C=1 / C=4**: `ours.aggregate_tok_s / reference.aggregate_tok_s >=`
//!   [`THROUGHPUT_RATIO_THRESHOLD`] (spec 03: "at least 99% of the live
//!   reference's tok/s", "aggregate at least 99% of the live reference's
//!   aggregate").
//! - **ITL**: `ours.p95_ms / reference.p95_ms`, judged in three bands —
//!   spec 03 states the verdict as "p95 within the live reference's
//!   envelope" with no numeric tolerance, so the bands are the owner's
//!   explicit reading of that envelope (grilling session, 2026-09-09):
//!   - at or under [`ITL_RATIO_PASS_THRESHOLD`] (1.0): a clean pass — ours
//!     is no worse than the reference's own p95;
//!   - above that but at or under [`ITL_RATIO_FAIL_THRESHOLD`] (1.1): still
//!     a pass, but [`ItlVerdict::warning`] is set — tolerated, flagged for
//!     review rather than silently accepted;
//!   - above [`ITL_RATIO_FAIL_THRESHOLD`]: fails the gate outright.
//!
//!   p50/p95/p99/max are all carried on the verdict regardless, so a later
//!   reader sees the shape and not only the number that decided it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::g3::Record;

/// The C=1 / C=4 threshold: ignis's aggregate tok/s over the reference's,
/// per cell. G3 tightens G2's 1.5x floor to 99% (spec 03).
pub const THROUGHPUT_RATIO_THRESHOLD: f64 = 0.99;

/// The ITL clean-pass ceiling: at or under this ratio, ours is no worse
/// than the reference's own p95 — no warning.
pub const ITL_RATIO_PASS_THRESHOLD: f64 = 1.0;

/// The ITL hard-fail ceiling: above this ratio the gate fails outright.
/// Between [`ITL_RATIO_PASS_THRESHOLD`] and this one, the gate still
/// passes but [`ItlVerdict::warning`] is set (see the module docs).
pub const ITL_RATIO_FAIL_THRESHOLD: f64 = 1.1;

/// Why the gate check will not produce a verdict at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A throughput cell's comparison (C=1 or C=4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThroughputVerdict {
    pub ours_tok_s: f64,
    pub reference_tok_s: f64,
    /// `ours / reference`.
    pub ratio: f64,
    /// `ratio >= threshold`.
    pub passed: bool,
}

/// The ITL cell's comparison: every percentile carried, p95 decides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItlVerdict {
    pub ours_p50_ms: f64,
    pub ours_p95_ms: f64,
    pub ours_p99_ms: f64,
    pub ours_max_ms: f64,
    pub reference_p50_ms: f64,
    pub reference_p95_ms: f64,
    pub reference_p99_ms: f64,
    pub reference_max_ms: f64,
    /// `ours_p95 / reference_p95`.
    pub ratio: f64,
    /// `ratio <= `[`ITL_RATIO_FAIL_THRESHOLD`].
    pub passed: bool,
    /// Set when `ratio` is in the tolerated-but-flagged band
    /// (`ITL_RATIO_PASS_THRESHOLD`, `ITL_RATIO_FAIL_THRESHOLD`]: the gate
    /// still passes, but this is not silently a clean result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// The G3 verdict: C=1, C=4 and ITL, and their conjunction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// The session both records were measured in.
    pub session: String,
    pub ours_label: String,
    pub reference_label: String,
    pub ours_engine: String,
    pub reference_engine: String,
    pub ours_profile: String,
    pub reference_profile: String,
    pub ours_date: String,
    pub reference_date: String,
    pub throughput_threshold: f64,
    pub itl_pass_threshold: f64,
    pub itl_fail_threshold: f64,
    pub c1: ThroughputVerdict,
    pub c4: ThroughputVerdict,
    pub itl: ItlVerdict,
    /// Known inequalities between the two engines as measured — recorded
    /// next to the verdict rather than corrected for (e.g. spec 03's
    /// recorded inequality that N=8 at long context is unreachable with
    /// BF16 KV; that inequality lives on the roadmap, not a G3 record, but
    /// the field exists for whatever a given run needs to note).
    #[serde(default)]
    pub notes: Vec<String>,
    /// Every cell passed.
    pub passed: bool,
}

impl Verdict {
    /// Serialize to pretty JSON (the on-disk verdict format).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialize the verdict: {e}"))
    }

    /// Write the verdict to `path` as pretty JSON.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_json()?).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// The terminal rendering: one line per cell, then the verdict.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("G3 gate check  session={}\n", self.session));
        out.push_str(&format!(
            "  {} ({}) profile={} date={}\n",
            self.ours_label, self.ours_engine, self.ours_profile, self.ours_date
        ));
        out.push_str(&format!(
            "  {} ({}) profile={} date={}\n",
            self.reference_label, self.reference_engine, self.reference_profile, self.reference_date
        ));
        for (name, cell) in [("C=1", &self.c1), ("C=4", &self.c4)] {
            out.push_str(&format!(
                "\n  {name}  {:>9.1} tok/s  vs  {:>9.1} tok/s  ratio {:>7.3}  {}\n",
                cell.ours_tok_s,
                cell.reference_tok_s,
                cell.ratio,
                if cell.passed { "PASS" } else { "FAIL" },
            ));
        }
        out.push_str(&format!(
            "\n  ITL  p50 {:>7.2}/{:>7.2}  p95 {:>7.2}/{:>7.2}  p99 {:>7.2}/{:>7.2}  max {:>7.2}/{:>7.2} ms  \
             ratio(p95) {:>7.3}  {}\n",
            self.itl.ours_p50_ms,
            self.itl.reference_p50_ms,
            self.itl.ours_p95_ms,
            self.itl.reference_p95_ms,
            self.itl.ours_p99_ms,
            self.itl.reference_p99_ms,
            self.itl.ours_max_ms,
            self.itl.reference_max_ms,
            self.itl.ratio,
            if self.itl.passed { "PASS" } else { "FAIL" },
        ));
        if let Some(warning) = &self.itl.warning {
            out.push_str(&format!("  ITL WARNING: {warning}\n"));
        }
        for note in &self.notes {
            out.push_str(&format!("\n  note: {note}\n"));
        }
        out.push_str(&format!(
            "\nG3 verdict: {} (tok/s ratio >= {} on C=1/C=4, ITL p95 ratio <= {} clean / <= {} \
             tolerated-with-warning)\n",
            if self.passed { "PASS" } else { "FAIL" },
            self.throughput_threshold,
            self.itl_pass_threshold,
            self.itl_fail_threshold,
        ));
        out
    }
}

/// Compare two G3 records and report the verdict, or refuse.
///
/// `ours` is the numerator (ignis), `reference` the denominator. See the
/// module docs for the live/live contract (ADR 0015) this enforces.
pub fn check(ours: &Record, reference: &Record) -> Result<Verdict, Refusal> {
    if ours.session != reference.session {
        return Err(Refusal(format!(
            "the records are not from the same session ({} vs {}): G3 is judged live/live \
             (ADR 0015), so both engines must be measured in one session on one machine",
            ours.session, reference.session
        )));
    }

    for (name, ours_cell, reference_cell) in
        [("C=1", &ours.c1, &reference.c1), ("C=4", &ours.c4, &reference.c4)]
    {
        for (label, cell) in [(ours.label.as_str(), ours_cell), (reference.label.as_str(), reference_cell)] {
            if let Some(error) = &cell.error {
                return Err(Refusal(format!(
                    "the {label} record's {name} cell failed and was never measured: {error}"
                )));
            }
            if let Some(sample) = cell.bad_samples().first() {
                return Err(Refusal(format!(
                    "the {label} record's {name} cell has a bad sample ({}): {} — a \
                     contaminated run cannot decide the gate",
                    sample.id,
                    sample.void_reason.as_deref().unwrap_or("(no reason recorded)"),
                )));
            }
            if cell.samples.is_empty() {
                return Err(Refusal(format!("the {label} record's {name} cell has no samples")));
            }
        }
    }

    for (label, itl) in [(ours.label.as_str(), &ours.itl), (reference.label.as_str(), &reference.itl)] {
        if let Some(error) = &itl.error {
            return Err(Refusal(format!(
                "the {label} record's ITL cell failed and was never measured: {error}"
            )));
        }
        if let Some(prefiller) = itl.void_prefillers().first() {
            return Err(Refusal(format!(
                "the {label} record's ITL cell has a void prefiller (#{}): {} — a contaminated \
                 run cannot decide the gate",
                prefiller.index,
                prefiller.void_reason.as_deref().unwrap_or("(no reason recorded)"),
            )));
        }
        if itl.prefillers.iter().any(|p| p.first_token_ms <= p.started_ms) {
            return Err(Refusal(format!(
                "the {label} record's ITL cell has no valid shared timeline; re-record it with the current harness"
            )));
        }
        if itl.lanes.iter().any(|l| l.error.is_some()) {
            return Err(Refusal(format!(
                "the {label} record's ITL cell has a decode lane that failed mid-series"
            )));
        }
        if itl.p95_ms.is_none() {
            return Err(Refusal(format!(
                "the {label} record's ITL cell has no p95 (no intervals were sampled)"
            )));
        }
    }

    let c1 = throughput_verdict(&ours.c1, &reference.c1)?;
    let c4 = throughput_verdict(&ours.c4, &reference.c4)?;
    let itl = itl_verdict(&ours.itl, &reference.itl)?;
    let passed = c1.passed && c4.passed && itl.passed;

    Ok(Verdict {
        session: ours.session.clone(),
        ours_label: ours.label.clone(),
        reference_label: reference.label.clone(),
        ours_engine: ours.engine.clone(),
        reference_engine: reference.engine.clone(),
        ours_profile: ours.profile.clone(),
        reference_profile: reference.profile.clone(),
        ours_date: ours.date.clone(),
        reference_date: reference.date.clone(),
        throughput_threshold: THROUGHPUT_RATIO_THRESHOLD,
        itl_pass_threshold: ITL_RATIO_PASS_THRESHOLD,
        itl_fail_threshold: ITL_RATIO_FAIL_THRESHOLD,
        c1,
        c4,
        itl,
        notes: Vec::new(),
        passed,
    })
}

fn throughput_verdict(
    ours: &crate::g3::ThroughputCell,
    reference: &crate::g3::ThroughputCell,
) -> Result<ThroughputVerdict, Refusal> {
    if reference.aggregate_tok_s <= 0.0 {
        return Err(Refusal(format!(
            "the reference's cell has an aggregate of {} tok/s: a ratio against it would be \
             meaningless",
            reference.aggregate_tok_s
        )));
    }
    let ratio = ours.aggregate_tok_s / reference.aggregate_tok_s;
    Ok(ThroughputVerdict {
        ours_tok_s: ours.aggregate_tok_s,
        reference_tok_s: reference.aggregate_tok_s,
        ratio,
        passed: ratio >= THROUGHPUT_RATIO_THRESHOLD,
    })
}

fn itl_verdict(ours: &crate::g3::ItlCell, reference: &crate::g3::ItlCell) -> Result<ItlVerdict, Refusal> {
    let (Some(ours_p95), Some(ref_p95)) = (ours.p95_ms, reference.p95_ms) else {
        return Err(Refusal("the ITL cell has no p95 on one of the two sides".to_string()));
    };
    if ref_p95 <= 0.0 {
        return Err(Refusal(format!(
            "the reference's ITL p95 is {ref_p95} ms: a ratio against it would be meaningless"
        )));
    }
    let ratio = ours_p95 / ref_p95;
    let warning = (ratio > ITL_RATIO_PASS_THRESHOLD && ratio <= ITL_RATIO_FAIL_THRESHOLD).then(|| {
        format!(
            "ITL p95 ratio {ratio:.3} exceeds the reference's own p95 (tolerated up to \
             {ITL_RATIO_FAIL_THRESHOLD}, but not a clean pass — flagged for review)"
        )
    });
    Ok(ItlVerdict {
        ours_p50_ms: ours.p50_ms.unwrap_or(0.0),
        ours_p95_ms: ours_p95,
        ours_p99_ms: ours.p99_ms.unwrap_or(0.0),
        ours_max_ms: ours.max_ms.unwrap_or(0.0),
        reference_p50_ms: reference.p50_ms.unwrap_or(0.0),
        reference_p95_ms: ref_p95,
        reference_p99_ms: reference.p99_ms.unwrap_or(0.0),
        reference_max_ms: reference.max_ms.unwrap_or(0.0),
        ratio,
        passed: ratio <= ITL_RATIO_FAIL_THRESHOLD,
        warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g3::{DecodeLaneTrace, ItlCell, PrefillerSample, ThroughputCell, ThroughputSample};

    fn throughput_sample(id: &str, prompt_tokens: u32) -> ThroughputSample {
        ThroughputSample {
            id: id.into(),
            ttft_ms: 50.0,
            total_ms: 550.0,
            n_tokens: 100,
            ok: true,
            computed_prefill_tokens: Some(prompt_tokens),
            void: false,
            void_reason: None,
        }
    }

    fn throughput_cell(prompt_tokens: u32, concurrency: usize, aggregate_tok_s: f64) -> ThroughputCell {
        let samples = (0..concurrency)
            .map(|i| throughput_sample(&format!("s{i}"), prompt_tokens))
            .collect();
        ThroughputCell {
            prompt_tokens,
            max_tokens: 256,
            concurrency,
            samples,
            aggregate_tok_s,
            error: None,
        }
    }

    fn itl_cell(p50: f64, p95: f64, p99: f64, max: f64) -> ItlCell {
        ItlCell {
            prefill_prompt_tokens: 32_768,
            prefill_max_tokens: 64,
            prefill_count: 10,
            decode_prompt_tokens: 4_096,
            decode_max_tokens: 512,
            decode_lanes: 4,
            prefillers: (0..10)
                .map(|i| PrefillerSample {
                    index: i,
                    started_ms: i as f64 * 1_000.0,
                    first_token_ms: i as f64 * 1_000.0 + 900.0,
                    ttft_ms: 900.0,
                    computed_prefill_tokens: Some(32_768),
                    void: false,
                    void_reason: None,
                })
                .collect(),
            lanes: vec![DecodeLaneTrace {
                id: "lane-0".into(),
                started_ms: 0.0,
                n_tokens: 300,
                token_times_ms: vec![0.0, 10.0, 20.0],
                finish: crate::g3::LaneFinish::Window,
                error: None,
            }],
            intervals_ms: vec![10.0, 10.0],
            p50_ms: Some(p50),
            p95_ms: Some(p95),
            p99_ms: Some(p99),
            max_ms: Some(max),
            error: None,
        }
    }

    fn record(label: &str, session: &str, c1: f64, c4: f64, itl_p95: f64) -> Record {
        Record {
            session: session.into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-09T12:00:00Z".into(),
            c1: throughput_cell(8_192, 1, c1),
            c4: throughput_cell(8_192, 4, c4),
            itl: itl_cell(6.0, itl_p95, 12.0, 20.0),
        }
    }

    #[test]
    fn matched_engines_pass_every_cell() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.c1.passed && verdict.c4.passed && verdict.itl.passed);
        assert!(verdict.passed, "{}", verdict.render());
    }

    #[test]
    fn a_slow_c1_fails_only_that_cell() {
        let ours = record("ignis", "S1", 90.0, 380.0, 8.0);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(!verdict.c1.passed, "0.9 ratio is under the 0.99 floor");
        assert!(verdict.c4.passed);
        assert!(!verdict.passed);
    }

    #[test]
    fn an_itl_p95_worse_than_the_reference_fails_the_gate() {
        let ours = record("ignis", "S1", 100.0, 380.0, 12.0);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!((verdict.itl.ratio - 1.5).abs() < 1e-9);
        assert!(!verdict.itl.passed);
        assert!(!verdict.passed);
    }

    #[test]
    fn an_itl_p95_at_or_under_the_reference_passes_cleanly() {
        let ours = record("ignis", "S1", 100.0, 380.0, 7.5);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.itl.passed);
        assert!(verdict.itl.warning.is_none(), "a ratio <= 1.0 is a clean pass, no warning");
    }

    #[test]
    fn an_itl_p95_between_the_pass_and_fail_thresholds_passes_with_a_warning() {
        // 8.4 / 8.0 = 1.05: over the clean-pass ceiling (1.0) but under the
        // hard-fail ceiling (1.1) — tolerated, but flagged (owner
        // direction, grilling session 2026-09-09).
        let ours = record("ignis", "S1", 100.0, 380.0, 8.4);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.itl.passed, "1.05 is tolerated");
        assert!(verdict.passed);
        let warning = verdict.itl.warning.as_deref().expect("a warning must be set");
        assert!(warning.contains("1.1"), "{warning}");
        assert!(verdict.render().contains("WARNING"), "the render must surface the warning");
    }

    #[test]
    fn an_itl_p95_over_the_fail_threshold_fails_even_though_it_would_be_a_warning_below_it() {
        // 9.0 / 8.0 = 1.125: over the hard-fail ceiling (1.1).
        let ours = record("ignis", "S1", 100.0, 380.0, 9.0);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(!verdict.itl.passed);
        assert!(!verdict.passed);
        // A hard fail is not also flagged as a "tolerated" warning — it's
        // just a failure.
        assert!(verdict.itl.warning.is_none());
    }

    #[test]
    fn records_from_different_sessions_are_refused() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let reference = record("reference", "S2-committed-fixture", 100.0, 380.0, 8.0);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("same session"), "{refusal}");
    }

    #[test]
    fn a_bad_sample_on_c4_refuses_a_verdict_rather_than_failing_one() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let mut reference = record("reference", "S1", 100.0, 380.0, 8.0);
        reference.c4.samples[1].void = true;
        reference.c4.samples[1].void_reason = Some("computed prefill 100 is short of the 8192-token prompt".into());
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("bad sample"), "{refusal}");
        assert!(refusal.0.contains("C=4"), "{refusal}");
    }

    #[test]
    fn a_void_prefiller_refuses_a_verdict() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let mut reference = record("reference", "S1", 100.0, 380.0, 8.0);
        reference.itl.prefillers[3].void = true;
        reference.itl.prefillers[3].void_reason = Some("computed prefill short of the prompt".into());
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("void prefiller"), "{refusal}");
    }

    #[test]
    fn a_legacy_itl_record_without_the_shared_timeline_is_refused() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let mut reference = record("reference", "S1", 100.0, 380.0, 8.0);
        for prefiller in &mut reference.itl.prefillers {
            prefiller.started_ms = 0.0;
            prefiller.first_token_ms = 0.0;
        }

        let refusal = check(&ours, &reference).expect_err("must refuse an incomparable record");
        assert!(refusal.0.contains("shared timeline"), "{refusal}");
    }

    #[test]
    fn a_failed_decode_lane_refuses_a_verdict() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let mut reference = record("reference", "S1", 100.0, 380.0, 8.0);
        reference.itl.lanes[0].error = Some("connection reset".into());
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("decode lane"), "{refusal}");
    }

    #[test]
    fn a_failed_cell_is_refused() {
        let mut ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        ours.c1.error = Some("the request failed: connection refused".into());
        ours.c1.samples.clear();
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("never measured"), "{refusal}");
    }

    #[test]
    fn the_verdict_renders_every_cell() {
        let ours = record("ignis", "S1", 100.0, 380.0, 8.0);
        let reference = record("reference", "S1", 100.0, 380.0, 8.0);
        let mut verdict = check(&ours, &reference).expect("a verdict");
        verdict.notes.push("N=8 at long context is unreachable with BF16 KV".into());
        let text = verdict.render();
        for expected in ["S1", "ignis-engine", "reference-engine", "C=1", "C=4", "ITL", "PASS", "N=8"] {
            assert!(text.contains(expected), "render must mention {expected}:\n{text}");
        }
    }
}
