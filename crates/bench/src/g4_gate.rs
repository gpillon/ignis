//! The **G4 gate check** (P4-01, GitHub #117): at least two launches per
//! engine in, one pooled verdict out.
//!
//! Spec: `.scratch/runtime/specs/04-reference-feature-floor.md` ("Gate G4");
//! ADR 0015 (live/live), ADR 0021 (launch pooling). Mirrors
//! [`crate::g2`] / [`crate::g3_gate`]'s refusal discipline exactly — a
//! verdict is only computed when every launch of both engines agrees on the
//! session id and the trace hash, and every needle cell is present on both
//! sides — but adds what ADR 0021 asks and G2/G3 have not needed yet: the
//! *pooled* cell statistic across two or more independent process launches
//! per engine, with each launch's own value carried as a diagnostic rather
//! than decisive.
//!
//! ## Pooling, kept to the smallest thing that works
//!
//! ADR 0021: "report every launch's cells, and require the cell statistic
//! taken across all launches ... to meet the threshold." Concretely: the
//! per-class (and aggregate) tok/s cell pools an engine's launches by
//! concatenating every launch's per-request metrics and reducing them with
//! the same throughput-weighted aggregation [`crate::metrics::class_stats`]
//! already uses for one run — no new aggregation framework, the existing
//! one applied across launches instead of within one. Each launch's own
//! aggregate is reported too (diagnostic only), and a cell whose
//! across-launch spread (the coefficient of variation of the launches' own
//! aggregates) exceeds the largest within-launch spread (the coefficient of
//! variation of that launch's own per-request tok/s) is flagged — ADR
//! 0021's own finding, not silently averaged away.
//!
//! The needle-retrieval cell is never pooled as a ratio (spec 04: "reported
//! as a pass/fail floor and never folded into a ratio") — every launch must
//! retrieve the planted fact for [`NeedleVerdict::passed`] to hold.

use serde::{Deserialize, Serialize};

use crate::g4::Record;
use crate::metrics::{class_stats, ClassStats, RequestMetrics};
use crate::trace::RequestClass;

/// The G4 per-class / aggregate throughput threshold (spec 04's Gate G4
/// table: "pooled ratio >= 0.99 against the live reference" — the same 99%
/// floor G3 tightened G2's 1.5x to).
pub const RATIO_THRESHOLD: f64 = 0.99;

/// Why the gate check will not produce a verdict at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One cell's (a class, or the aggregate) pooled comparison across every
/// launch of both engines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellVerdict {
    /// `"main"`, `"sub"`, or `"aggregate"`.
    pub label: String,
    /// The pooled aggregate (concatenated across every launch, then
    /// throughput-weighted — [`crate::metrics::class_stats`]'s reduction).
    pub ours_pooled_tok_s: f64,
    pub reference_pooled_tok_s: f64,
    /// `ours_pooled / reference_pooled` — the decisive number.
    pub ratio: f64,
    /// `ratio >= `[`RATIO_THRESHOLD`].
    pub passed: bool,
    /// Diagnostic only (ADR 0021): each of ours' launches' own aggregate.
    pub ours_launch_tok_s: Vec<f64>,
    /// Diagnostic only: each of the reference's launches' own aggregate.
    pub reference_launch_tok_s: Vec<f64>,
    /// Diagnostic: the coefficient of variation (%) of ours' launches'
    /// aggregates.
    pub ours_across_launch_spread_pct: f64,
    /// Diagnostic: the same, for the reference.
    pub reference_across_launch_spread_pct: f64,
    /// Diagnostic: the largest per-launch coefficient of variation (%) of
    /// per-request tok/s, over ours' own launches — the baseline ADR
    /// 0021's flag compares *ours'* across-launch spread against ("its own
    /// launches' internal spread").
    pub ours_within_launch_spread_pct: f64,
    /// Diagnostic: the same, over the reference's own launches.
    pub reference_within_launch_spread_pct: f64,
    /// Set when either engine's across-launch spread exceeds *that same
    /// engine's own* within-launch spread: "a cell whose across-launch
    /// spread looks larger than its own launches' internal spread is
    /// itself a finding worth recording, not silently averaged away" (ADR
    /// 0021) — compared per engine, not against the other engine's own
    /// spread, so a quiet reference launch cannot mask a real flag on the
    /// engine under test, or vice versa.
    pub spread_flagged: bool,
}

/// The needle-retrieval verdict at one context length: a correctness
/// floor, never a ratio (spec 04).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeedleVerdict {
    pub context_tokens: u32,
    /// Whether *every* one of ours' launches retrieved the planted fact at
    /// this length.
    pub ours_retrieved: bool,
    /// The same, for the reference (recorded for the audit trail; the
    /// floor this ticket gates is on the engine under test).
    pub reference_retrieved: bool,
    /// `== ours_retrieved`.
    pub passed: bool,
}

/// The G4 verdict: the pooled per-class / aggregate throughput cells, the
/// needle-retrieval floor, and their conjunction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// The session every launch of both engines shared.
    pub session: String,
    /// The SHA-256 every launch of both engines replayed.
    pub trace_sha256: String,
    pub ours_label: String,
    pub reference_label: String,
    /// How many independent launches were pooled per side (>= 2, ADR 0021).
    pub ours_launches: usize,
    pub reference_launches: usize,
    pub threshold: f64,
    /// One entry per class present on either side (main, sub).
    pub per_class: Vec<CellVerdict>,
    /// The whole-run cell: every request, regardless of class — so a
    /// record whose requests carry no recognizable tag still produces a
    /// verdict (spec 04's third acceptance criterion).
    pub aggregate: CellVerdict,
    /// One entry per needle-retrieval context length measured.
    pub needles: Vec<NeedleVerdict>,
    /// Known inequalities between the two engines as measured — recorded
    /// next to the verdict rather than corrected for (the G2/G3 pattern).
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
    pub fn write(&self, path: &std::path::Path) -> Result<(), String> {
        std::fs::write(path, self.to_json()?).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// The terminal rendering: one line per cell, the needle floor, then
    /// the verdict.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "G4 gate check  session={}  trace={}\n",
            self.session, self.trace_sha256
        ));
        out.push_str(&format!(
            "  {} ({} launches)  vs  {} ({} launches)\n",
            self.ours_label, self.ours_launches, self.reference_label, self.reference_launches
        ));
        for cell in self.per_class.iter().chain(std::iter::once(&self.aggregate)) {
            out.push_str(&format!(
                "\n  {:<9}  {:>9.1} tok/s  vs  {:>9.1} tok/s  ratio {:>7.3}  {}\n",
                cell.label,
                cell.ours_pooled_tok_s,
                cell.reference_pooled_tok_s,
                cell.ratio,
                if cell.passed { "PASS" } else { "FAIL" },
            ));
            out.push_str(&format!(
                "    per-launch  ours={:?}  reference={:?}\n",
                cell.ours_launch_tok_s, cell.reference_launch_tok_s
            ));
            if cell.spread_flagged {
                out.push_str(&format!(
                    "    WARNING: across-launch spread exceeds that engine's own within-launch \
                     spread (ours: {:.1}% across vs {:.1}% within; reference: {:.1}% across vs \
                     {:.1}% within) — ADR 0021\n",
                    cell.ours_across_launch_spread_pct,
                    cell.ours_within_launch_spread_pct,
                    cell.reference_across_launch_spread_pct,
                    cell.reference_within_launch_spread_pct,
                ));
            }
        }
        out.push_str("\n  needle retrieval (correctness floor, not a ratio):\n");
        for needle in &self.needles {
            out.push_str(&format!(
                "    {:>7}K  ours={}  reference={}  {}\n",
                needle.context_tokens / 1024,
                needle.ours_retrieved,
                needle.reference_retrieved,
                if needle.passed { "PASS" } else { "FAIL" },
            ));
        }
        for note in &self.notes {
            out.push_str(&format!("\n  note: {note}\n"));
        }
        out.push_str(&format!(
            "\nG4 verdict: {} (pooled tok/s ratio >= {} per class and aggregate, every needle \
             cell retrieved)\n",
            if self.passed { "PASS" } else { "FAIL" },
            self.threshold,
        ));
        out
    }
}

fn class_name(class: RequestClass) -> &'static str {
    match class {
        RequestClass::Main => "main",
        RequestClass::Sub => "sub",
    }
}

/// This record's metrics for `class`, or every metric when `class` is
/// `None` (the aggregate cell: every request, regardless of tag).
fn class_metrics(record: &Record, class: Option<RequestClass>) -> Vec<RequestMetrics> {
    match class {
        Some(c) => record.run.metrics.iter().filter(|m| m.class == c).cloned().collect(),
        None => record.run.metrics.clone(),
    }
}

/// The pooled `ClassStats` over every launch's metrics for `class`
/// concatenated together — the "cell statistic taken across all launches"
/// ADR 0021 asks for, built from the aggregation `class_stats` already
/// performs within one run.
fn pooled_stats(records: &[Record], class: Option<RequestClass>) -> Option<ClassStats> {
    let items: Vec<RequestMetrics> = records.iter().flat_map(|r| class_metrics(r, class)).collect();
    if items.is_empty() {
        None
    } else {
        Some(class_stats(class.unwrap_or(RequestClass::Main), &items))
    }
}

/// Each launch's own (un-pooled) aggregate tok/s for `class` — diagnostic
/// only (`0.0` for a launch with no matching requests).
fn per_launch_tok_s(records: &[Record], class: Option<RequestClass>) -> Vec<f64> {
    records
        .iter()
        .map(|r| {
            let items = class_metrics(r, class);
            if items.is_empty() {
                0.0
            } else {
                class_stats(class.unwrap_or(RequestClass::Main), &items).tok_s
            }
        })
        .collect()
}

/// The coefficient of variation (%) of `values` — `0.0` for fewer than two
/// values or a non-positive mean (nothing to disperse).
fn coefficient_of_variation(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    (variance.sqrt() / mean) * 100.0
}

/// One record's within-launch spread (%) for `class`: the coefficient of
/// variation of its own requests' per-request tok/s.
fn within_launch_spread_pct(record: &Record, class: Option<RequestClass>) -> f64 {
    let values: Vec<f64> = class_metrics(record, class).iter().map(|m| m.tok_s()).collect();
    coefficient_of_variation(&values)
}

/// Build one cell's pooled verdict (a class, or the aggregate).
fn pooled_cell(
    label: &str,
    ours: &[Record],
    reference: &[Record],
    class: Option<RequestClass>,
) -> Result<CellVerdict, Refusal> {
    let ours_pooled = pooled_stats(ours, class).ok_or_else(|| {
        Refusal(format!(
            "the `{label}` cell has no requests in {}'s records — nothing to compare",
            ours[0].label
        ))
    })?;
    let reference_pooled = pooled_stats(reference, class).ok_or_else(|| {
        Refusal(format!(
            "the `{label}` cell has no requests in {}'s records — nothing to compare",
            reference[0].label
        ))
    })?;
    if reference_pooled.tok_s <= 0.0 {
        return Err(Refusal(format!(
            "the reference's `{label}` cell has a pooled aggregate of {} tok/s: a ratio against \
             it would be meaningless",
            reference_pooled.tok_s
        )));
    }
    let ratio = ours_pooled.tok_s / reference_pooled.tok_s;

    let ours_launch_tok_s = per_launch_tok_s(ours, class);
    let reference_launch_tok_s = per_launch_tok_s(reference, class);
    let ours_across = coefficient_of_variation(&ours_launch_tok_s);
    let reference_across = coefficient_of_variation(&reference_launch_tok_s);
    // ADR 0021: "a cell whose across-launch spread looks larger than its
    // own launches' internal spread" — each engine's across-launch number
    // is compared against *that engine's own* within-launch baseline, not
    // the other engine's, so a quiet reference launch cannot mask a real
    // flag on the engine under test (or vice versa).
    let ours_within = ours.iter().map(|r| within_launch_spread_pct(r, class)).fold(0.0_f64, f64::max);
    let reference_within = reference
        .iter()
        .map(|r| within_launch_spread_pct(r, class))
        .fold(0.0_f64, f64::max);
    let spread_flagged = ours_across > ours_within || reference_across > reference_within;

    Ok(CellVerdict {
        label: label.to_string(),
        ours_pooled_tok_s: ours_pooled.tok_s,
        reference_pooled_tok_s: reference_pooled.tok_s,
        ratio,
        passed: ratio >= RATIO_THRESHOLD,
        ours_launch_tok_s,
        reference_launch_tok_s,
        ours_across_launch_spread_pct: ours_across,
        reference_across_launch_spread_pct: reference_across,
        ours_within_launch_spread_pct: ours_within,
        reference_within_launch_spread_pct: reference_within,
        spread_flagged,
    })
}

/// Compare pooled launches of two engines and report the G4 verdict, or
/// refuse. `ours` is the engine under test (>= 2 launches), `reference`
/// the live reference (>= 2 launches) — ADR 0021's launch-pooling floor.
pub fn check(ours: &[Record], reference: &[Record]) -> Result<Verdict, Refusal> {
    if ours.is_empty() {
        return Err(Refusal("no launches were recorded for the engine under test".to_string()));
    }
    if reference.is_empty() {
        return Err(Refusal("no launches were recorded for the reference".to_string()));
    }
    if ours.len() < 2 {
        return Err(Refusal(format!(
            "{} has {} launch(es) recorded; G4 is a live/live gate and pools at least two \
             independent process launches per engine within the session (ADR 0021)",
            ours[0].label,
            ours.len()
        )));
    }
    if reference.len() < 2 {
        return Err(Refusal(format!(
            "{} has {} launch(es) recorded; G4 is a live/live gate and pools at least two \
             independent process launches per engine within the session (ADR 0021)",
            reference[0].label,
            reference.len()
        )));
    }

    let all: Vec<&Record> = ours.iter().chain(reference.iter()).collect();
    let session = &all[0].session;
    if let Some(bad) = all.iter().find(|r| &r.session != session) {
        return Err(Refusal(format!(
            "{} (session {}) and {} (session {}) are not from the same session: G4 is judged \
             live/live (ADR 0015), so every launch of both engines must share one session",
            all[0].label, session, bad.label, bad.session
        )));
    }
    let trace_sha256 = &all[0].trace_sha256;
    if let Some(bad) = all.iter().find(|r| &r.trace_sha256 != trace_sha256) {
        return Err(Refusal(format!(
            "{} (trace {}) and {} (trace {}) replayed different load traces: a G4 verdict can \
             only be computed when every launch replayed the identical recorded load",
            all[0].label, trace_sha256, bad.label, bad.trace_sha256
        )));
    }

    let mut classes: Vec<RequestClass> = ours
        .iter()
        .chain(reference.iter())
        .flat_map(|r| r.run.classes())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    classes.sort_by_key(|c| *c as u8);

    let mut per_class = Vec::with_capacity(classes.len());
    for class in classes {
        per_class.push(pooled_cell(class_name(class), ours, reference, Some(class))?);
    }
    let aggregate = pooled_cell("aggregate", ours, reference, None)?;

    // Every launch of both engines must carry the same set of needle
    // context lengths — a length present on one launch only is a missing
    // cell (spec 04's second acceptance criterion: "a cell is missing —
    // refusal names which").
    let mut lengths: Vec<u32> = all
        .iter()
        .flat_map(|r| r.needles.iter().map(|n| n.context_tokens))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    lengths.sort_unstable();
    if lengths.is_empty() {
        return Err(Refusal("no needle-retrieval cells were recorded on either side".to_string()));
    }
    for record in &all {
        for &len in &lengths {
            if !record.needles.iter().any(|n| n.context_tokens == len) {
                return Err(Refusal(format!(
                    "{} (session {}) has no needle-retrieval cell at {len} context tokens",
                    record.label, record.session
                )));
            }
        }
    }

    let needles: Vec<NeedleVerdict> = lengths
        .into_iter()
        .map(|len| {
            let ours_retrieved = ours
                .iter()
                .flat_map(|r| r.needles.iter().filter(|n| n.context_tokens == len))
                .all(|n| n.passed());
            let reference_retrieved = reference
                .iter()
                .flat_map(|r| r.needles.iter().filter(|n| n.context_tokens == len))
                .all(|n| n.passed());
            NeedleVerdict {
                context_tokens: len,
                ours_retrieved,
                reference_retrieved,
                passed: ours_retrieved,
            }
        })
        .collect();

    let passed =
        per_class.iter().all(|c| c.passed) && aggregate.passed && needles.iter().all(|n| n.passed);

    Ok(Verdict {
        session: session.clone(),
        trace_sha256: trace_sha256.clone(),
        ours_label: ours[0].label.clone(),
        reference_label: reference[0].label.clone(),
        ours_launches: ours.len(),
        reference_launches: reference.len(),
        threshold: RATIO_THRESHOLD,
        per_class,
        aggregate,
        needles,
        notes: Vec::new(),
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g4::NeedleResult;
    use crate::metrics::{RequestMetrics, Run};

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

    fn needle(context_tokens: u32, retrieved: bool) -> NeedleResult {
        NeedleResult {
            context_tokens,
            secret: format!("secret-{context_tokens}"),
            retrieved,
            error: None,
        }
    }

    /// A launch with `sub_tok_s` decode tok/s on a single sub-class
    /// request (100 tokens: 99 decode tokens over `99.0 / sub_tok_s`
    /// seconds), plus one main request.
    fn record(label: &str, session: &str, trace: &str, sub_tok_s: f64, needles_ok: bool) -> Record {
        let decode_s = 99.0 / sub_tok_s;
        Record {
            session: session.into(),
            trace_sha256: trace.into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-11T12:00:00Z".into(),
            run: Run::new(
                label,
                vec![
                    m("main", RequestClass::Main, 200.0, 100, 6000.0),
                    m("sub", RequestClass::Sub, 100.0, 100, 100.0 + decode_s * 1000.0),
                ],
            ),
            needles: vec![needle(65_536, needles_ok), needle(131_072, needles_ok)],
        }
    }

    #[test]
    fn two_matched_launches_per_engine_pass_every_cell() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.passed, "{}", verdict.render());
        assert_eq!(verdict.ours_launches, 2);
        assert_eq!(verdict.reference_launches, 2);
        assert!(verdict.per_class.iter().any(|c| c.label == "main"));
        assert!(verdict.per_class.iter().any(|c| c.label == "sub"));
        assert_eq!(verdict.aggregate.label, "aggregate");
        assert!(verdict.needles.iter().all(|n| n.passed));
    }

    #[test]
    fn a_single_launch_per_engine_is_refused() {
        let ours = vec![record("ignis", "S1", "abc", 100.0, true)];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("ignis"), "{refusal}");
        assert!(refusal.0.contains("ADR 0021"), "{refusal}");
    }

    #[test]
    fn a_single_reference_launch_is_refused_too() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference = vec![record("reference", "S1", "abc", 100.0, true)];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("reference"), "{refusal}");
        assert!(refusal.0.contains("ADR 0021"), "{refusal}");
    }

    #[test]
    fn no_launches_recorded_for_ours_is_refused() {
        let ours: Vec<Record> = Vec::new();
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("engine under test"), "{refusal}");
    }

    #[test]
    fn no_launches_recorded_for_the_reference_is_refused() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference: Vec<Record> = Vec::new();
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("reference"), "{refusal}");
    }

    #[test]
    fn a_zero_tok_s_reference_cell_is_refused_rather_than_a_meaningless_ratio() {
        // A reference whose sub-class requests never decoded (ttft ==
        // total, tok_s() == 0.0 for every request): a ratio against a
        // pooled aggregate of 0.0 tok/s would be meaningless.
        let mk = |label: &str| Record {
            session: "S1".into(),
            trace_sha256: "abc".into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-11T12:00:00Z".into(),
            run: Run::new(label, vec![m("only", RequestClass::Sub, 100.0, 1, 100.0)]),
            needles: vec![needle(65_536, true), needle(131_072, true)],
        };
        // `ours` must also carry only `Sub`, or the `main` cell (present in
        // `ours` but absent from this all-`Sub` reference) refuses first.
        let ours = vec![mk("ignis"), mk("ignis")];
        let reference = vec![mk("reference"), mk("reference")];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("meaningless"), "{refusal}");
    }

    #[test]
    fn no_needle_cells_recorded_anywhere_is_refused() {
        let mk = |label: &str| Record {
            session: "S1".into(),
            trace_sha256: "abc".into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-11T12:00:00Z".into(),
            run: Run::new(
                label,
                vec![m("main", RequestClass::Main, 200.0, 100, 6000.0)],
            ),
            needles: Vec::new(),
        };
        let ours = vec![mk("ignis"), mk("ignis")];
        let reference = vec![mk("reference"), mk("reference")];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("no needle-retrieval cells"), "{refusal}");
    }

    #[test]
    fn mismatched_sessions_are_refused_and_named() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S2", "abc", 100.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("S1") && refusal.0.contains("S2"), "{refusal}");
        assert!(refusal.0.contains("same session"), "{refusal}");
    }

    #[test]
    fn mismatched_trace_hashes_are_refused_and_named() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "def", 100.0, true),
        ];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("abc") && refusal.0.contains("def"), "{refusal}");
        assert!(refusal.0.contains("different load traces"), "{refusal}");
    }

    #[test]
    fn a_missing_needle_cell_on_one_launch_is_refused_and_named() {
        let mut ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        ours[1].needles.retain(|n| n.context_tokens != 131_072);
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("ignis"), "{refusal}");
        assert!(refusal.0.contains("131072"), "{refusal}");
    }

    #[test]
    fn a_missing_needle_cell_on_a_reference_launch_is_refused_and_named() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let mut reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        reference[0].needles.retain(|n| n.context_tokens != 65_536);
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("reference"), "{refusal}");
        assert!(refusal.0.contains("65536"), "{refusal}");
    }

    #[test]
    fn a_class_present_only_on_one_side_is_refused_and_named() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let mut reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        for r in &mut reference {
            r.run.metrics.retain(|m| m.class != RequestClass::Sub);
        }
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("sub"), "{refusal}");
        assert!(refusal.0.contains("nothing to compare"), "{refusal}");
    }

    #[test]
    fn a_class_present_only_on_the_reference_side_is_refused_and_named() {
        let mut ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        for r in &mut ours {
            r.run.metrics.retain(|m| m.class != RequestClass::Sub);
        }
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("sub"), "{refusal}");
        assert!(refusal.0.contains("ignis"), "{refusal}");
        assert!(refusal.0.contains("nothing to compare"), "{refusal}");
    }

    #[test]
    fn a_slow_sub_class_fails_only_that_cell() {
        let ours = vec![
            record("ignis", "S1", "abc", 90.0, true), // 0.9 ratio, under 0.99
            record("ignis", "S1", "abc", 90.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let verdict = check(&ours, &reference).expect("a verdict");
        let sub = verdict.per_class.iter().find(|c| c.label == "sub").expect("sub cell");
        assert!(!sub.passed);
        let main = verdict.per_class.iter().find(|c| c.label == "main").expect("main cell");
        assert!(main.passed, "main is unaffected: identical on both sides");
        assert!(!verdict.passed);
    }

    #[test]
    fn a_missed_needle_fails_the_gate_without_touching_the_ratio_cells() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, false),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.per_class.iter().all(|c| c.passed), "throughput is untouched");
        assert!(verdict.aggregate.passed);
        assert!(
            !verdict.needles.iter().all(|n| n.passed),
            "one launch missed the needle -> the floor fails for that length"
        );
        assert!(!verdict.passed, "a missed needle fails the whole gate");
    }

    #[test]
    fn an_untagged_run_still_produces_an_aggregate_verdict() {
        // Every request maps to `Sub` (the trace format's own default for
        // an absent/unrecognized tag) — there is no `Main` request on
        // either side, yet the aggregate cell (spec 04's third acceptance
        // criterion) must still produce a verdict.
        let mk = |label: &str, tok_s: f64| Record {
            session: "S1".into(),
            trace_sha256: "abc".into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-11T12:00:00Z".into(),
            run: Run::new(
                label,
                vec![m("only", RequestClass::Sub, 100.0, 100, 100.0 + (99.0 / tok_s) * 1000.0)],
            ),
            needles: vec![needle(65_536, true), needle(131_072, true)],
        };
        let ours = vec![mk("ignis", 100.0), mk("ignis", 100.0)];
        let reference = vec![mk("reference", 100.0), mk("reference", 100.0)];
        let verdict = check(&ours, &reference).expect("a verdict");
        assert_eq!(verdict.per_class.len(), 1, "only `sub` is present");
        assert_eq!(verdict.aggregate.label, "aggregate");
        assert!(verdict.aggregate.passed);
        assert!(verdict.passed);
    }

    #[test]
    fn an_across_launch_spread_larger_than_any_within_launch_spread_is_flagged() {
        // Two `ignis` launches with wildly different sub-class tok/s (a
        // large across-launch spread) but each launch's own single request
        // has no internal spread to compare against (0%) — the flag must
        // fire (ADR 0021's own finding: not silently averaged away).
        let ours = vec![
            record("ignis", "S1", "abc", 80.0, true),
            record("ignis", "S1", "abc", 120.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let verdict = check(&ours, &reference).expect("a verdict");
        let sub = verdict.per_class.iter().find(|c| c.label == "sub").expect("sub cell");
        assert!(sub.spread_flagged, "{sub:?}");
        assert!(sub.ours_across_launch_spread_pct > 0.0);
    }

    #[test]
    fn matched_launches_are_not_flagged() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.per_class.iter().all(|c| !c.spread_flagged));
        assert!(!verdict.aggregate.spread_flagged);
    }

    #[test]
    fn the_verdict_round_trips_through_json_and_renders() {
        let ours = vec![
            record("ignis", "S1", "abc", 100.0, true),
            record("ignis", "S1", "abc", 100.0, true),
        ];
        let reference = vec![
            record("reference", "S1", "abc", 100.0, true),
            record("reference", "S1", "abc", 100.0, true),
        ];
        let mut verdict = check(&ours, &reference).expect("a verdict");
        verdict.notes.push("hq KV format inequality retired by this run".into());
        let json = verdict.to_json().expect("serialize");
        let back: Verdict = serde_json::from_str(&json).expect("parse");
        assert_eq!(back, verdict);
        let text = verdict.render();
        for expected in ["S1", "abc", "ignis", "reference", "main", "sub", "aggregate", "PASS", "64K"] {
            assert!(text.contains(expected), "render must mention {expected}:\n{text}");
        }
    }
}
