//! The **G5 gate check** (P5-07, GitHub #151): at least two launches per
//! engine in, one pooled ratio out — per depth.
//!
//! Spec: `.scratch/runtime/specs/05-speculative-decoding.md` (not written
//! yet; see [`crate::g5`]'s doc comment). ADR 0015 (live/live), ADR 0021
//! (launch pooling). #151: "a `g5-gate` reuses `g4-gate`'s per-cell pooling
//! rule across launches and reports one ratio per depth against 0.99" — this
//! module is [`crate::g4_gate`]'s pooling rule (concatenate every launch's
//! per-request metrics, reduce with [`crate::metrics::class_stats`]'s
//! throughput-weighted aggregation, flag a cell whose across-launch spread
//! exceeds its own within-launch spread) applied per depth instead of per
//! request class, with no needle-retrieval floor (G5 has none) and no trace
//! hash (G5 is not a trace replay).

use serde::{Deserialize, Serialize};

use crate::g5::Record;
use crate::metrics::{class_stats, ClassStats, RequestMetrics};
use crate::trace::RequestClass;

/// The G5 per-depth throughput threshold (#151: "reports one ratio per
/// depth against 0.99" — the same 99% floor G4's per-class cells use).
pub const RATIO_THRESHOLD: f64 = 0.99;

/// Why the gate check will not produce a verdict at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One depth's pooled comparison across every launch of both engines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellVerdict {
    pub depth_tokens: u32,
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
    /// per-request tok/s, over ours' own launches.
    pub ours_within_launch_spread_pct: f64,
    /// Diagnostic: the same, over the reference's own launches.
    pub reference_within_launch_spread_pct: f64,
    /// Set when either engine's across-launch spread exceeds *that same
    /// engine's own* within-launch spread (ADR 0021's own finding — see
    /// `crate::g4_gate::CellVerdict::spread_flagged`'s doc comment, which
    /// this mirrors exactly).
    pub spread_flagged: bool,
}

/// The G5 verdict: the pooled per-depth throughput cells and their
/// conjunction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// The session every launch of both engines shared.
    pub session: String,
    pub ours_label: String,
    pub reference_label: String,
    /// How many independent launches were pooled per side (>= 2, ADR 0021).
    pub ours_launches: usize,
    pub reference_launches: usize,
    pub threshold: f64,
    /// One entry per depth measured ([`crate::g5::DEPTHS`] order).
    pub per_depth: Vec<CellVerdict>,
    /// Known inequalities between the two engines as measured — recorded
    /// next to the verdict rather than corrected for (the G2/G3/G4
    /// pattern).
    #[serde(default)]
    pub notes: Vec<String>,
    /// Every depth cell passed.
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

    /// The terminal rendering: one line per depth, then the verdict.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("G5 gate check  session={}\n", self.session));
        out.push_str(&format!(
            "  {} ({} launches)  vs  {} ({} launches)\n",
            self.ours_label, self.ours_launches, self.reference_label, self.reference_launches
        ));
        for cell in &self.per_depth {
            out.push_str(&format!(
                "\n  {:>4}K  {:>9.1} tok/s  vs  {:>9.1} tok/s  ratio {:>7.3}  {}\n",
                cell.depth_tokens / 1024,
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
        for note in &self.notes {
            out.push_str(&format!("\n  note: {note}\n"));
        }
        out.push_str(&format!(
            "\nG5 verdict: {} (pooled committed tok/s ratio >= {} at every depth)\n",
            if self.passed { "PASS" } else { "FAIL" },
            self.threshold,
        ));
        out
    }
}

/// This record's samples at `depth_tokens`, as [`RequestMetrics`] — the
/// seam onto [`crate::metrics::class_stats`]'s existing throughput-weighted
/// reduction. `RequestClass::Main` is a placeholder tag: G5's cells carry no
/// class distinction of their own, and `class_stats`'s math never reads it.
fn depth_metrics(record: &Record, depth_tokens: u32) -> Option<Vec<RequestMetrics>> {
    record.cell_at(depth_tokens).map(|cell| {
        cell.samples
            .iter()
            .map(|s| RequestMetrics {
                id: s.id.clone(),
                class: RequestClass::Main,
                ttft_ms: s.ttft_ms,
                n_tokens: s.n_tokens,
                total_ms: s.total_ms,
                ok: s.ok,
            })
            .collect()
    })
}

/// The pooled `ClassStats` over every launch's samples at `depth_tokens`
/// concatenated together.
fn pooled_stats(records: &[Record], depth_tokens: u32) -> Option<ClassStats> {
    let items: Vec<RequestMetrics> =
        records.iter().filter_map(|r| depth_metrics(r, depth_tokens)).flatten().collect();
    if items.is_empty() {
        None
    } else {
        Some(class_stats(RequestClass::Main, &items))
    }
}

/// Each launch's own (un-pooled) aggregate tok/s at `depth_tokens` —
/// diagnostic only (`0.0` for a launch with no matching cell).
fn per_launch_tok_s(records: &[Record], depth_tokens: u32) -> Vec<f64> {
    records
        .iter()
        .map(|r| match depth_metrics(r, depth_tokens) {
            Some(items) if !items.is_empty() => class_stats(RequestClass::Main, &items).tok_s,
            _ => 0.0,
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

/// One record's within-launch spread (%) at `depth_tokens`: the coefficient
/// of variation of its own samples' per-request tok/s (always `0.0` at
/// `concurrency = 1`, since a C=1 cell has exactly one sample per launch —
/// recorded rather than assumed, so a future concurrency change is not
/// silently wrong here).
fn within_launch_spread_pct(record: &Record, depth_tokens: u32) -> f64 {
    let values: Vec<f64> =
        depth_metrics(record, depth_tokens).unwrap_or_default().iter().map(|m| m.tok_s()).collect();
    coefficient_of_variation(&values)
}

/// Every sample at `depth_tokens` across `records` that failed or generated
/// nothing, as `"<id> (launch <n>)"` — the same #147 refusal `g4_gate` makes
/// for its own cells.
fn unmeasured_samples(records: &[Record], depth_tokens: u32) -> Vec<String> {
    records
        .iter()
        .enumerate()
        .flat_map(|(i, r)| {
            depth_metrics(r, depth_tokens)
                .unwrap_or_default()
                .into_iter()
                .filter(|m| !m.is_measurement())
                .map(move |m| format!("{} (launch {})", m.id, i + 1))
        })
        .collect()
}

/// Build one depth's pooled verdict.
fn pooled_cell(depth_tokens: u32, ours: &[Record], reference: &[Record]) -> Result<CellVerdict, Refusal> {
    let unmeasured: Vec<String> = [ours, reference]
        .into_iter()
        .filter_map(|side| {
            let ids = unmeasured_samples(side, depth_tokens);
            (!ids.is_empty()).then(|| format!("{}'s records: {}", side[0].label, ids.join(", ")))
        })
        .collect();
    if !unmeasured.is_empty() {
        return Err(Refusal(format!(
            "the {}K depth cell holds samples that failed or generated no tokens — {} — a cell \
             computed from them is not a measurement",
            depth_tokens / 1024,
            unmeasured.join("; ")
        )));
    }
    let ours_pooled = pooled_stats(ours, depth_tokens).ok_or_else(|| {
        Refusal(format!(
            "the {}K depth cell is missing from {}'s records — nothing to compare",
            depth_tokens / 1024,
            ours[0].label
        ))
    })?;
    let reference_pooled = pooled_stats(reference, depth_tokens).ok_or_else(|| {
        Refusal(format!(
            "the {}K depth cell is missing from {}'s records — nothing to compare",
            depth_tokens / 1024,
            reference[0].label
        ))
    })?;
    if reference_pooled.tok_s <= 0.0 {
        return Err(Refusal(format!(
            "the reference's {}K depth cell has a pooled aggregate of {} tok/s: a ratio against \
             it would be meaningless",
            depth_tokens / 1024,
            reference_pooled.tok_s
        )));
    }
    if ours_pooled.tok_s <= 0.0 {
        return Err(Refusal(format!(
            "{}'s {}K depth cell has a pooled aggregate of {} tok/s (no decode phase): ranking it \
             would be meaningless",
            ours[0].label,
            depth_tokens / 1024,
            ours_pooled.tok_s
        )));
    }
    let ratio = ours_pooled.tok_s / reference_pooled.tok_s;

    let ours_launch_tok_s = per_launch_tok_s(ours, depth_tokens);
    let reference_launch_tok_s = per_launch_tok_s(reference, depth_tokens);
    let ours_across = coefficient_of_variation(&ours_launch_tok_s);
    let reference_across = coefficient_of_variation(&reference_launch_tok_s);
    let ours_within =
        ours.iter().map(|r| within_launch_spread_pct(r, depth_tokens)).fold(0.0_f64, f64::max);
    let reference_within = reference
        .iter()
        .map(|r| within_launch_spread_pct(r, depth_tokens))
        .fold(0.0_f64, f64::max);
    let spread_flagged = ours_across > ours_within || reference_across > reference_within;

    Ok(CellVerdict {
        depth_tokens,
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

/// Compare pooled launches of two engines and report the G5 verdict, or
/// refuse. `ours` is the engine under test (>= 2 launches), `reference` the
/// live reference (>= 2 launches) — ADR 0021's launch-pooling floor.
pub fn check(ours: &[Record], reference: &[Record]) -> Result<Verdict, Refusal> {
    if ours.is_empty() {
        return Err(Refusal("no launches were recorded for the engine under test".to_string()));
    }
    if reference.is_empty() {
        return Err(Refusal("no launches were recorded for the reference".to_string()));
    }
    if ours.len() < 2 {
        return Err(Refusal(format!(
            "{} has {} launch(es) recorded; G5 is a live/live gate and pools at least two \
             independent process launches per engine within the session (ADR 0021)",
            ours[0].label,
            ours.len()
        )));
    }
    if reference.len() < 2 {
        return Err(Refusal(format!(
            "{} has {} launch(es) recorded; G5 is a live/live gate and pools at least two \
             independent process launches per engine within the session (ADR 0021)",
            reference[0].label,
            reference.len()
        )));
    }

    let all: Vec<&Record> = ours.iter().chain(reference.iter()).collect();
    let session = &all[0].session;
    if let Some(bad) = all.iter().find(|r| &r.session != session) {
        return Err(Refusal(format!(
            "{} (session {}) and {} (session {}) are not from the same session: G5 is judged \
             live/live (ADR 0015), so every launch of both engines must share one session",
            all[0].label, session, bad.label, bad.session
        )));
    }

    // Every launch of both engines must carry the same set of depths — a
    // depth present on one launch only is a missing cell.
    let mut depths: Vec<u32> = all
        .iter()
        .flat_map(|r| r.cells.iter().map(|c| c.prompt_tokens))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    depths.sort_unstable();
    if depths.is_empty() {
        return Err(Refusal("no depth cells were recorded on either side".to_string()));
    }
    for record in &all {
        for &depth in &depths {
            if record.cell_at(depth).is_none() {
                return Err(Refusal(format!(
                    "{} (session {}) has no cell at {}K",
                    record.label,
                    record.session,
                    depth / 1024
                )));
            }
        }
    }

    let mut per_depth = Vec::with_capacity(depths.len());
    for depth in depths {
        per_depth.push(pooled_cell(depth, ours, reference)?);
    }

    let passed = per_depth.iter().all(|c| c.passed);

    Ok(Verdict {
        session: session.clone(),
        ours_label: ours[0].label.clone(),
        reference_label: reference[0].label.clone(),
        ours_launches: ours.len(),
        reference_launches: reference.len(),
        threshold: RATIO_THRESHOLD,
        per_depth,
        notes: Vec::new(),
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g3::{ThroughputCell, ThroughputSample};
    use crate::g5::{DEPTHS, DEPTH_196K, DEPTH_24K, DEPTH_98K};

    fn sample(id: &str, ttft: f64, n: u32, total: f64) -> ThroughputSample {
        ThroughputSample {
            id: id.into(),
            ttft_ms: ttft,
            total_ms: total,
            n_tokens: n,
            ok: true,
            computed_prefill_tokens: Some(24_576),
            void: false,
            void_reason: None,
        }
    }

    /// A launch with one C=1 sample at `depth` running `tok_s` decode tok/s
    /// over 511 decode tokens (512 committed).
    fn cell(depth: u32, tok_s: f64) -> ThroughputCell {
        let decode_s = 511.0 / tok_s;
        ThroughputCell {
            prompt_tokens: depth,
            max_tokens: 512,
            concurrency: 1,
            samples: vec![sample("s0", 40.0, 512, 40.0 + decode_s * 1000.0)],
            aggregate_tok_s: tok_s,
            error: None,
        }
    }

    fn record(label: &str, session: &str, tok_s_by_depth: &[(u32, f64)]) -> Record {
        Record {
            session: session.into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-13T12:00:00Z".into(),
            cells: tok_s_by_depth.iter().map(|&(depth, tok_s)| cell(depth, tok_s)).collect(),
        }
    }

    fn full(label: &str, session: &str, tok_s: f64) -> Record {
        record(label, session, &DEPTHS.map(|d| (d, tok_s)))
    }

    #[test]
    fn two_matched_launches_per_engine_pass_every_depth() {
        let ours = vec![full("ignis", "S1", 140.0), full("ignis", "S1", 140.0)];
        let reference = vec![full("reference", "S1", 140.0), full("reference", "S1", 140.0)];
        let verdict = check(&ours, &reference).expect("a verdict");
        assert!(verdict.passed, "{}", verdict.render());
        assert_eq!(verdict.per_depth.len(), 3);
        assert!(verdict.per_depth.iter().all(|c| c.passed));
    }

    #[test]
    fn a_single_launch_per_engine_is_refused() {
        let ours = vec![full("ignis", "S1", 140.0)];
        let reference = vec![full("reference", "S1", 140.0), full("reference", "S1", 140.0)];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("ignis"), "{refusal}");
        assert!(refusal.0.contains("ADR 0021"), "{refusal}");
    }

    #[test]
    fn mismatched_sessions_are_refused_and_named() {
        let ours = vec![full("ignis", "S1", 140.0), full("ignis", "S2", 140.0)];
        let reference = vec![full("reference", "S1", 140.0), full("reference", "S1", 140.0)];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("S1") && refusal.0.contains("S2"), "{refusal}");
        assert!(refusal.0.contains("same session"), "{refusal}");
    }

    #[test]
    fn a_missing_depth_cell_is_refused_and_named() {
        let ours = vec![
            full("ignis", "S1", 140.0),
            record("ignis", "S1", &[(DEPTH_24K, 140.0), (DEPTH_98K, 140.0)]),
        ];
        let reference = vec![full("reference", "S1", 140.0), full("reference", "S1", 140.0)];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("ignis"), "{refusal}");
        assert!(refusal.0.contains("196K"), "{refusal}");
    }

    #[test]
    fn a_slow_depth_fails_only_that_cell() {
        let ours = vec![
            record("ignis", "S1", &[(DEPTH_24K, 90.0), (DEPTH_98K, 140.0), (DEPTH_196K, 140.0)]),
            record("ignis", "S1", &[(DEPTH_24K, 90.0), (DEPTH_98K, 140.0), (DEPTH_196K, 140.0)]),
        ];
        let reference = vec![full("reference", "S1", 100.0), full("reference", "S1", 100.0)];
        let verdict = check(&ours, &reference).expect("a verdict");
        let d24 = verdict.per_depth.iter().find(|c| c.depth_tokens == DEPTH_24K).unwrap();
        assert!(!d24.passed, "0.9 ratio, under 0.99");
        let d98 = verdict.per_depth.iter().find(|c| c.depth_tokens == DEPTH_98K).unwrap();
        assert!(d98.passed);
        assert!(!verdict.passed);
    }

    #[test]
    fn a_cell_holding_a_sample_that_generated_nothing_is_refused_not_ranked() {
        let mut ours = vec![full("ignis", "S1", 140.0), full("ignis", "S1", 140.0)];
        let reference = vec![full("reference", "S1", 140.0), full("reference", "S1", 140.0)];
        let bad = ours[1].cells.iter_mut().find(|c| c.prompt_tokens == DEPTH_98K).unwrap();
        bad.samples[0].n_tokens = 0;
        bad.samples[0].ok = true;
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("98K"), "{refusal}");
        assert!(refusal.0.contains("launch 2"), "{refusal}");
        assert!(refusal.0.contains("not a measurement"), "{refusal}");
    }

    #[test]
    fn a_zero_tok_s_reference_cell_is_refused_rather_than_a_meaningless_ratio() {
        let mk = |label: &str| {
            let mut r = full(label, "S1", 100.0);
            for cell in &mut r.cells {
                cell.samples[0].total_ms = cell.samples[0].ttft_ms; // no decode phase
                cell.aggregate_tok_s = 0.0;
            }
            r
        };
        let ours = vec![mk("ignis"), mk("ignis")];
        let reference = vec![mk("reference"), mk("reference")];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("meaningless"), "{refusal}");
    }

    #[test]
    fn no_depth_cells_recorded_anywhere_is_refused() {
        let mk = |label: &str| Record {
            session: "S1".into(),
            label: label.into(),
            endpoint: format!("http://127.0.0.1:8000/{label}"),
            engine: format!("{label}-engine"),
            artifact: "qwen3.8-27b.ninfer".into(),
            profile: format!("{label}-profile"),
            date: "2026-09-13T12:00:00Z".into(),
            cells: Vec::new(),
        };
        let ours = vec![mk("ignis"), mk("ignis")];
        let reference = vec![mk("reference"), mk("reference")];
        let refusal = check(&ours, &reference).expect_err("must refuse");
        assert!(refusal.0.contains("no depth cells"), "{refusal}");
    }

    #[test]
    fn an_across_launch_spread_larger_than_any_within_launch_spread_is_flagged() {
        let ours = vec![
            record("ignis", "S1", &DEPTHS.map(|d| (d, 80.0))),
            record("ignis", "S1", &DEPTHS.map(|d| (d, 120.0))),
        ];
        let reference = vec![full("reference", "S1", 100.0), full("reference", "S1", 100.0)];
        let verdict = check(&ours, &reference).expect("a verdict");
        let d24 = verdict.per_depth.iter().find(|c| c.depth_tokens == DEPTH_24K).unwrap();
        assert!(d24.spread_flagged, "{d24:?}");
        assert!(d24.ours_across_launch_spread_pct > 0.0);
    }

    #[test]
    fn the_verdict_round_trips_through_json_and_renders() {
        let ours = vec![full("ignis", "S1", 140.0), full("ignis", "S1", 140.0)];
        let reference = vec![full("reference", "S1", 140.0), full("reference", "S1", 140.0)];
        let mut verdict = check(&ours, &reference).expect("a verdict");
        verdict.notes.push("MTP deferred, not measured".into());
        let json = verdict.to_json().expect("serialize");
        let back: Verdict = serde_json::from_str(&json).expect("parse");
        assert_eq!(back, verdict);
        let text = verdict.render();
        for expected in ["S1", "ignis", "reference", "24K", "98K", "196K", "PASS"] {
            assert!(text.contains(expected), "render must mention {expected}:\n{text}");
        }
    }
}
