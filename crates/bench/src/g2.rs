//! The **G2 gate check** (P2-05, GitHub #87): two TTFT records in, one
//! verdict out.
//!
//! Spec: `.scratch/runtime/specs/02-real-prefill.md`; ADR 0015. G2 asks one
//! question per cell — is ignis's median time to first token within
//! [`RATIO_THRESHOLD`] of the reference's? — and the answer is only
//! meaningful if the two numbers were produced **live/live**: the same
//! session, the same cells, and every sample provably cold.
//!
//! Those preconditions are enforced *here*, in the tool, rather than left
//! to the operator's discipline (ADR 0015). [`check`] returns a
//! [`Refusal`], not a failing verdict, when:
//!
//! - the two records do not share a session identifier — one of them is a
//!   memory, and a committed fixture can therefore never be the live side;
//! - a cell is present on one side and missing on the other — there is
//!   nothing to compare;
//! - any sample on either side is void — the record contains a measurement
//!   of something other than prefill.
//!
//! A refusal is not a failed gate. It is the tool saying the evidence does
//! not support *any* verdict, which is the distinction that keeps a
//! contaminated run from quietly passing.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::ttft::Record;

/// The G2 threshold: ignis's median TTFT over the reference's, per cell.
///
/// One number, applied per cell, so the verdict is unambiguous. It is a
/// *floor* ("not meaningfully worse than the reference"), not a target —
/// the project's north star is being at least as fast as the reference,
/// and G3/G4 tighten this to 99%.
pub const RATIO_THRESHOLD: f64 = 1.5;

/// Why the gate check will not produce a verdict at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One cell's comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellVerdict {
    /// The cell's prompt length, in post-template tokens.
    pub prompt_tokens: u32,
    /// ignis's median TTFT for this cell, in ms.
    pub ours_median_ms: f64,
    /// The reference's median TTFT for this cell, in ms.
    pub reference_median_ms: f64,
    /// `ours / reference`.
    pub ratio: f64,
    /// `ratio <= threshold`.
    pub passed: bool,
}

/// The G2 verdict: every cell's ratio, and their conjunction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// The session both records were measured in (they must agree, or
    /// there is no verdict).
    pub session: String,
    /// The label of the record on the numerator side (ignis).
    pub ours_label: String,
    /// The label of the record on the denominator side (the reference).
    pub reference_label: String,
    /// The engine identity each side reported, for the audit trail.
    pub ours_engine: String,
    /// The engine identity the reference reported.
    pub reference_engine: String,
    /// The profile each side was measured in — the gate compares engines
    /// as run, so the profiles are part of the verdict.
    pub ours_profile: String,
    /// The reference's profile.
    pub reference_profile: String,
    /// When the verdict was computed (from the records' own dates).
    pub ours_date: String,
    /// The reference record's date.
    pub reference_date: String,
    /// The threshold applied per cell.
    pub threshold: f64,
    /// One entry per cell, in the numerator record's order.
    pub cells: Vec<CellVerdict>,
    /// Known inequalities between the two engines as measured (the KV
    /// format difference at G2, for one) — recorded next to the verdict
    /// rather than corrected for.
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
        out.push_str(&format!("G2 gate check  session={}\n", self.session));
        out.push_str(&format!(
            "  {} ({}) profile={} date={}\n",
            self.ours_label, self.ours_engine, self.ours_profile, self.ours_date
        ));
        out.push_str(&format!(
            "  {} ({}) profile={} date={}\n",
            self.reference_label,
            self.reference_engine,
            self.reference_profile,
            self.reference_date
        ));
        out.push_str(&format!(
            "\n  {:>9}  {:>12}  {:>12}  {:>7}  {}\n",
            "tokens", self.ours_label, self.reference_label, "ratio", "verdict"
        ));
        for cell in &self.cells {
            out.push_str(&format!(
                "  {:>9}  {:>9.1} ms  {:>9.1} ms  {:>7.3}  {}\n",
                cell.prompt_tokens,
                cell.ours_median_ms,
                cell.reference_median_ms,
                cell.ratio,
                if cell.passed { "PASS" } else { "FAIL" },
            ));
        }
        for note in &self.notes {
            out.push_str(&format!("\n  note: {note}\n"));
        }
        out.push_str(&format!(
            "\nG2 verdict: {} (ratio <= {} on every cell)\n",
            if self.passed { "PASS" } else { "FAIL" },
            self.threshold
        ));
        out
    }
}

/// Compare two TTFT records and report the G2 verdict, or refuse.
///
/// `ours` is the numerator (ignis), `reference` the denominator. See the
/// module docs for the three refusals — they are the live/live contract
/// (ADR 0015) expressed as code.
pub fn check(ours: &Record, reference: &Record) -> Result<Verdict, Refusal> {
    // Live/live: the two records must come from the same measurement
    // session. This is also what stops the committed reference fixture
    // from ever being the live side of a gate.
    if ours.session != reference.session {
        return Err(Refusal(format!(
            "the records are not from the same session ({} vs {}): G2 is judged live/live \
             (ADR 0015), so both engines must be measured in one session on one machine",
            ours.session, reference.session
        )));
    }

    // Both sides must offer the same cells: a cell present on one side
    // only has nothing to be compared against.
    for cell in &ours.cells {
        if reference.cell(cell.prompt_tokens).is_none() {
            return Err(Refusal(format!(
                "cell {} is missing from the {} record",
                cell.prompt_tokens, reference.label
            )));
        }
    }
    for cell in &reference.cells {
        if ours.cell(cell.prompt_tokens).is_none() {
            return Err(Refusal(format!(
                "cell {} is missing from the {} record",
                cell.prompt_tokens, ours.label
            )));
        }
    }
    if ours.cells.is_empty() {
        return Err(Refusal("the records carry no cells".to_string()));
    }

    // Every sample on both sides must be provably cold.
    for record in [ours, reference] {
        for cell in &record.cells {
            if let Some(error) = &cell.error {
                return Err(Refusal(format!(
                    "the {} record's cell {} failed and was never measured: {error}",
                    record.label, cell.prompt_tokens
                )));
            }
            if let Some(sample) = cell.samples.iter().find(|s| s.void) {
                return Err(Refusal(format!(
                    "the {} record's cell {} has a void sample (#{}): {} — a contaminated run \
                     cannot decide the gate",
                    record.label,
                    cell.prompt_tokens,
                    sample.index,
                    sample.void_reason.as_deref().unwrap_or("(no reason recorded)"),
                )));
            }
            if cell.samples.is_empty() {
                return Err(Refusal(format!(
                    "the {} record's cell {} has no samples",
                    record.label, cell.prompt_tokens
                )));
            }
        }
    }

    let mut cells = Vec::with_capacity(ours.cells.len());
    for cell in &ours.cells {
        let theirs = reference.cell(cell.prompt_tokens).expect("checked above");
        let (Some(ours_median), Some(reference_median)) =
            (cell.median_ttft_ms, theirs.median_ttft_ms)
        else {
            return Err(Refusal(format!(
                "cell {} has no median on one of the two sides",
                cell.prompt_tokens
            )));
        };
        if reference_median <= 0.0 {
            return Err(Refusal(format!(
                "the {} record's cell {} has a median of {reference_median} ms: a ratio against \
                 it would be meaningless",
                reference.label, cell.prompt_tokens
            )));
        }
        let ratio = ours_median / reference_median;
        cells.push(CellVerdict {
            prompt_tokens: cell.prompt_tokens,
            ours_median_ms: ours_median,
            reference_median_ms: reference_median,
            ratio,
            passed: ratio <= RATIO_THRESHOLD,
        });
    }

    let passed = cells.iter().all(|c| c.passed);
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
        threshold: RATIO_THRESHOLD,
        cells,
        notes: Vec::new(),
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttft::{Cell, Sample};

    fn sample(index: usize, ttft_ms: f64, prompt_tokens: u32) -> Sample {
        Sample {
            index,
            ttft_ms,
            computed_prefill_tokens: Some(prompt_tokens),
            void: false,
            void_reason: None,
        }
    }

    fn cell(prompt_tokens: u32, ttfts: &[f64]) -> Cell {
        let samples: Vec<Sample> = ttfts
            .iter()
            .enumerate()
            .map(|(i, t)| sample(i, *t, prompt_tokens))
            .collect();
        Cell {
            prompt_tokens,
            warmup_ttft_ms: Some(999.0),
            median_ttft_ms: crate::ttft::median(ttfts),
            samples,
            error: None,
        }
    }

    fn record(label: &str, session: &str, cells: Vec<Cell>) -> Record {
        Record {
            session: session.into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-08T12:00:00Z".into(),
            max_tokens: 8,
            cells,
        }
    }

    #[test]
    fn the_ratio_is_our_median_over_the_references() {
        let ours = record("ignis", "S1", vec![cell(8192, &[120.0, 100.0, 110.0])]);
        let reference = record("reference", "S1", vec![cell(8192, &[100.0, 90.0, 80.0])]);
        let verdict = check(&ours, &reference).expect("a verdict");
        // Medians: ours 110, reference 90 -> 1.222...
        let c = &verdict.cells[0];
        assert!((c.ours_median_ms - 110.0).abs() < 1e-9);
        assert!((c.reference_median_ms - 90.0).abs() < 1e-9);
        assert!((c.ratio - 110.0 / 90.0).abs() < 1e-9, "ratio {}", c.ratio);
        assert!(c.passed && verdict.passed);
    }

    #[test]
    fn a_cell_above_the_threshold_fails_the_gate() {
        let ours = record("ignis", "S1", vec![cell(8192, &[100.0]), cell(32768, &[400.0])]);
        let reference =
            record("reference", "S1", vec![cell(8192, &[100.0]), cell(32768, &[200.0])]);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.cells[0].passed, "1.0 is within the threshold");
        assert!(!verdict.cells[1].passed, "2.0 is over the threshold");
        assert!(!verdict.passed, "one failing cell fails the gate");
    }

    #[test]
    fn exactly_the_threshold_passes() {
        let ours = record("ignis", "S1", vec![cell(8192, &[150.0])]);
        let reference = record("reference", "S1", vec![cell(8192, &[100.0])]);
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!((verdict.cells[0].ratio - RATIO_THRESHOLD).abs() < 1e-9);
        assert!(verdict.passed, "the threshold is inclusive");
    }

    #[test]
    fn records_from_different_sessions_are_refused() {
        // The live/live rule (ADR 0015) — and what stops a committed
        // fixture from deciding a gate.
        let ours = record("ignis", "S1", vec![cell(8192, &[100.0])]);
        let reference = record("reference", "S2-committed-fixture", vec![cell(8192, &[100.0])]);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("same session"), "{refusal}");
    }

    #[test]
    fn a_cell_missing_on_the_reference_side_is_refused() {
        let ours = record("ignis", "S1", vec![cell(8192, &[100.0]), cell(32768, &[300.0])]);
        let reference = record("reference", "S1", vec![cell(8192, &[100.0])]);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("32768") && refusal.0.contains("missing"), "{refusal}");
    }

    #[test]
    fn a_cell_missing_on_our_side_is_refused() {
        let ours = record("ignis", "S1", vec![cell(8192, &[100.0])]);
        let reference =
            record("reference", "S1", vec![cell(8192, &[100.0]), cell(32768, &[300.0])]);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("32768") && refusal.0.contains("missing"), "{refusal}");
    }

    #[test]
    fn a_void_sample_refuses_a_verdict_rather_than_failing_the_gate() {
        let mut reference = record("reference", "S1", vec![cell(8192, &[100.0, 100.0, 100.0])]);
        reference.cells[0].samples[1].void = true;
        reference.cells[0].samples[1].void_reason =
            Some("computed prefill 0 is short of the 8192-token prompt".into());
        let ours = record("ignis", "S1", vec![cell(8192, &[100.0, 100.0, 100.0])]);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("void sample"), "{refusal}");
        assert!(refusal.0.contains("short of the 8192-token prompt"), "{refusal}");
    }

    #[test]
    fn a_void_sample_on_our_own_side_is_refused_too() {
        let mut ours = record("ignis", "S1", vec![cell(8192, &[100.0, 100.0])]);
        ours.cells[0].samples[0].void = true;
        ours.cells[0].samples[0].void_reason = Some("the request failed: connection reset".into());
        let reference = record("reference", "S1", vec![cell(8192, &[100.0, 100.0])]);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("void sample"), "{refusal}");
    }

    #[test]
    fn a_failed_cell_is_refused() {
        let mut ours = record("ignis", "S1", vec![cell(8192, &[100.0])]);
        ours.cells[0].error = Some("the warmup request failed: connection refused".into());
        ours.cells[0].samples.clear();
        ours.cells[0].median_ttft_ms = None;
        let reference = record("reference", "S1", vec![cell(8192, &[100.0])]);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("never measured"), "{refusal}");
    }

    #[test]
    fn records_with_no_cells_are_refused() {
        let ours = record("ignis", "S1", Vec::new());
        let reference = record("reference", "S1", Vec::new());
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("no cells"), "{refusal}");
    }

    #[test]
    fn the_verdict_renders_both_engines_and_every_cell() {
        let ours = record("ignis", "S1", vec![cell(8192, &[100.0]), cell(32768, &[300.0])]);
        let reference =
            record("reference", "S1", vec![cell(8192, &[100.0]), cell(32768, &[300.0])]);
        let mut verdict = check(&ours, &reference).expect("a verdict");
        verdict.notes.push("KV format differs: ignis BF16, reference hq-e8-2b".into());
        let text = verdict.render();
        for expected in ["S1", "ignis-engine", "reference-engine", "8192", "32768", "PASS", "KV format"]
        {
            assert!(text.contains(expected), "render must mention {expected}:\n{text}");
        }
    }
}
