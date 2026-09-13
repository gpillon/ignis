//! The G5 measurement instrument (P5-07, GitHub #151): committed tok/s at
//! three prompt depths, reusing [`crate::g3`]'s C=1 throughput cell
//! (concurrency 1) instead of a new instrument. The verdict half lives in
//! [`crate::g5_gate`].
//!
//! Spec: `.scratch/runtime/specs/05-speculative-decoding.md` — not written
//! yet (#66: "to write when G4 lands"); this module is built from #66's and
//! #151's own acceptance criteria plus `.scratch/REVIEW-2026-09-05.md` §6
//! Phase 5's depth table. ADR 0007 (performance gate, not parity), ADR 0015
//! (live/live, cold samples), ADR 0021 (pooled launches).
//!
//! ## The counter
//!
//! #151: "the counter the reference uses — `(completion_tokens − 1) /
//! decode_seconds`, first token excluded, decode phase only." That is
//! exactly [`crate::metrics::RequestMetrics::tok_s`]'s own formula
//! (`decode window = ttft -> last token`), which [`crate::g3::ThroughputCell`]'s
//! `aggregate_tok_s` already reduces a cell's samples with
//! ([`crate::metrics::class_stats`]'s throughput-weighted aggregation). G5
//! needs no new counter — only three new depths and a 512-token budget, cut
//! from g3's C=1 cell at `concurrency = 1`.
//!
//! ## Three depths, no new cell shape
//!
//! Spec 05's Gate G5 table: 24K / 98K / 196K post-template prompt tokens
//! ([`DEPTH_24K`] / [`DEPTH_98K`] / [`DEPTH_196K`], each `X * 1024` — the
//! same `XK` convention [`crate::g4::NEEDLE_CONTEXT_64K`] uses), 512
//! committed tokens ([`COMMITTED_TOKENS`]), cold, greedy (this crate's
//! `HttpEndpoint` fixes `temperature: 0` / `seed: 0` on every request — the
//! same contract every other gate in this crate measures under). At
//! 24K–196K a corpus is required in practice ([`G5Config::corpus`]) for the
//! same reason G4's 64K/128K needle cells need one: the filler word-growth
//! path's O(n²) re-encoding is impractical at this scale
//! ([`crate::g4::build_haystack_from_corpus`]'s doc comment).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::client::Endpoint;
use crate::g3::{measure_throughput_cell, measure_throughput_cell_from_corpus, ThroughputCell, C1_CONCURRENCY};
use crate::time::{unix_now, utc_timestamp};
use crate::ttft::{load_corpus, PromptTemplate};

/// The three prompt depths spec 05's Gate G5 table measures
/// (`.scratch/REVIEW-2026-09-05.md` §6 Phase 5: "24K / 98K / 196K"), each
/// `X * 1024` post-template tokens.
pub const DEPTH_24K: u32 = 24 * 1024;
pub const DEPTH_98K: u32 = 98 * 1024;
pub const DEPTH_196K: u32 = 196 * 1024;

/// The depths a `g5` run measures, in the order they are reported.
pub const DEPTHS: [u32; 3] = [DEPTH_24K, DEPTH_98K, DEPTH_196K];

/// The committed-token budget every depth cell asks for (#151: "512
/// committed tokens").
pub const COMMITTED_TOKENS: u32 = 512;

/// One depth's C=1 throughput cell — g3's own cell type and counter, just
/// measured at this phase's depth instead of G3's fixed 8,192.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthCell {
    pub depth_tokens: u32,
    pub cell: ThroughputCell,
}

/// A failed cell (corpus load error, or an unreachable fixture) — built
/// directly from [`ThroughputCell`]'s public fields rather than a new
/// constructor, since every field it needs is already `pub`.
fn failed_cell(depth_tokens: u32, max_tokens: u32, error: String) -> ThroughputCell {
    ThroughputCell {
        prompt_tokens: depth_tokens,
        max_tokens,
        concurrency: C1_CONCURRENCY,
        samples: Vec::new(),
        aggregate_tok_s: 0.0,
        error: Some(error),
    }
}

/// A G5 record: what one engine measured at the three depths, in which
/// session. Mirrors [`crate::g3::Record`] / [`crate::g4::Record`]'s
/// identity fields, so [`crate::g5_gate::check`] can enforce the same
/// live/live, pooled-launch discipline as `g4_gate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The measurement session every launch of both engines must share.
    pub session: String,
    /// Which engine this is ("ignis", "reference", ...).
    pub label: String,
    /// The endpoint measured.
    pub endpoint: String,
    /// The engine's own identity: the model id it reports at
    /// `GET /v1/models`.
    pub engine: String,
    /// The artifact this engine was serving, as the operator named it.
    pub artifact: String,
    /// The profile the engine was running in.
    pub profile: String,
    /// When the record was made (UTC, RFC 3339 seconds).
    pub date: String,
    /// One cell per depth measured, in [`DEPTHS`] order.
    pub cells: Vec<DepthCell>,
}

impl Record {
    /// This record's cell at `depth_tokens`, if measured.
    pub fn cell_at(&self, depth_tokens: u32) -> Option<&ThroughputCell> {
        self.cells.iter().find(|c| c.depth_tokens == depth_tokens).map(|c| &c.cell)
    }

    /// True when every depth cell produced a full set of cold, complete
    /// samples — the property a gate verdict may be computed over.
    pub fn all_cold(&self) -> bool {
        !self.cells.is_empty() && self.cells.iter().all(|c| c.cell.all_cold())
    }

    /// Serialize to pretty JSON (the on-disk record format).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialize the record: {e}"))
    }

    /// Parse a record from JSON.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("parse the record: {e}"))
    }

    /// Write the record to `path` as pretty JSON.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_json()?).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Read a record previously written by [`Record::write`].
    pub fn read(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    /// A human-readable rendering for the terminal.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "g5 record  session={}  label={}  engine={}  profile={}\n",
            self.session, self.label, self.engine, self.profile
        ));
        out.push_str(&format!("  endpoint={}  artifact={}  date={}\n", self.endpoint, self.artifact, self.date));
        for depth_cell in &self.cells {
            let cell = &depth_cell.cell;
            match &cell.error {
                Some(err) => out.push_str(&format!(
                    "  {:>4}K FAILED: {err}\n",
                    depth_cell.depth_tokens / 1024,
                )),
                None => {
                    let bad = cell.bad_samples().len();
                    out.push_str(&format!(
                        "  {:>4}K  {:>9.1} tok/s  over {} sequence(s){}\n",
                        depth_cell.depth_tokens / 1024,
                        cell.aggregate_tok_s,
                        cell.samples.len(),
                        if bad == 0 { "  (all cold)".to_string() } else { format!("  ({bad} BAD)") },
                    ));
                }
            }
        }
        out
    }
}

/// What a `g5` run measures and how the record identifies it.
#[derive(Debug, Clone)]
pub struct G5Config {
    /// Which engine this run is measuring ("ignis", "reference", ...).
    pub label: String,
    /// The profile the engine is running in.
    pub profile: String,
    /// The artifact the engine is serving, as the operator names it.
    pub artifact: String,
    /// The measurement session every launch of both engines must share.
    pub session: String,
    /// The committed-token budget every depth cell asks for (defaults to
    /// [`COMMITTED_TOKENS`]).
    pub committed_tokens: u32,
    /// A pre-tokenized prompt bank (whitespace-separated ids) to cut every
    /// depth cell's prompt from, when set (the `--corpus` flag): bounded
    /// generation instead of the filler path's O(n²) growth, needed at this
    /// cell's 24K–196K scale ([`crate::g4::build_haystack_from_corpus`]'s
    /// doc comment makes the same tradeoff at 64K/128K). Absent, the filler
    /// generator is used (fine at the small depths a unit test measures).
    pub corpus: Option<PathBuf>,
}

impl Default for G5Config {
    fn default() -> Self {
        Self {
            label: "ignis".into(),
            profile: "unrecorded".into(),
            artifact: String::new(),
            session: String::new(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: None,
        }
    }
}

/// Measure the three G5 depth cells against one endpoint and return the
/// record.
///
/// A configured corpus (`cfg.corpus`) is loaded once and every depth cell is
/// cut from it; a corpus that cannot be read fails every depth cell with
/// that error rather than falling back to the (impractically slow, at this
/// scale) filler generator.
pub fn measure(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    engine: String,
    endpoint: String,
    cfg: &G5Config,
) -> Record {
    let corpus = cfg.corpus.as_ref().map(|path| load_corpus(path));
    let cells = DEPTHS
        .iter()
        .map(|&depth| {
            let cell = match &corpus {
                Some(Ok(ids)) => measure_throughput_cell_from_corpus(
                    ep,
                    template,
                    depth,
                    cfg.committed_tokens,
                    C1_CONCURRENCY,
                    ids,
                ),
                Some(Err(error)) => failed_cell(depth, cfg.committed_tokens, error.clone()),
                None => measure_throughput_cell(ep, template, depth, cfg.committed_tokens, C1_CONCURRENCY),
            };
            DepthCell { depth_tokens: depth, cell }
        })
        .collect();
    Record {
        session: cfg.session.clone(),
        label: cfg.label.clone(),
        endpoint,
        engine,
        artifact: cfg.artifact.clone(),
        profile: cfg.profile.clone(),
        date: utc_timestamp(unix_now()),
        cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{Endpoint, FinishReason, Outcome, Request};

    /// A mock template: one token per whitespace-separated word — matches
    /// `g3.rs`'s / `g4.rs`'s own `MockTemplate` (a mock engine with no
    /// artifact, ADR 0006).
    struct MockTemplate;

    impl PromptTemplate for MockTemplate {
        fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
            Ok(content
                .split_whitespace()
                .map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32)))
                .collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String, String> {
            Ok(ids.iter().map(|id| format!("t{id}")).collect::<Vec<_>>().join(" "))
        }
    }

    /// A deterministic mock endpoint: fixed ttft, fixed decode interval, and
    /// the cold-prefix evidence the void rule reads — matches `g3.rs`'s own
    /// `MockEndpoint` (kept local rather than shared: this crate's mock
    /// endpoints are each small and test-specific).
    struct DeterministicEndpoint {
        ttft_ms: f64,
        interval_ms: f64,
    }

    impl Endpoint for DeterministicEndpoint {
        fn complete(&self, req: &Request) -> Result<Outcome, String> {
            let prompt_tokens = req.prompt.split_whitespace().count() as u32;
            let n = req.max_tokens;
            let token_times_ms: Vec<f64> =
                (0..n).map(|i| self.ttft_ms + i as f64 * self.interval_ms).collect();
            let total_ms = token_times_ms.last().copied().unwrap_or(self.ttft_ms);
            Ok(Outcome {
                ttft_ms: self.ttft_ms,
                total_ms,
                n_tokens: n,
                output: String::new(),
                reasoning_output: String::new(),
                reasoning_tokens: Some(0),
                prompt_tokens: Some(prompt_tokens),
                cached_prompt_tokens: None,
                token_times_ms,
                finish_reason: Some(FinishReason::Engine("length".into())),
            })
        }
    }

    #[test]
    fn depths_are_the_spec_05_table_in_1024_multiples() {
        assert_eq!(DEPTH_24K, 24_576);
        assert_eq!(DEPTH_98K, 100_352);
        assert_eq!(DEPTH_196K, 200_704);
        assert_eq!(DEPTHS, [DEPTH_24K, DEPTH_98K, DEPTH_196K]);
    }

    /// #151's acceptance criterion: "the counter is unit-tested on a
    /// synthetic Outcome (first token excluded, decode window = first-token
    /// to last-token)." A `DeterministicEndpoint` stands in for a synthetic
    /// `Outcome`: 512 tokens at a known ttft + interval, so the expected
    /// committed tok/s is computable by hand and independent of
    /// `class_stats`'s own implementation.
    #[test]
    fn each_depth_cell_uses_the_committed_token_counter() {
        let ep = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 };
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: None,
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        assert_eq!(record.cells.len(), 3);
        for depth_cell in &record.cells {
            let cell = &depth_cell.cell;
            assert!(cell.all_cold(), "{depth_cell:?}");
            assert_eq!(cell.samples.len(), 1, "C=1: one sequence per depth");
            let sample = &cell.samples[0];
            assert_eq!(sample.n_tokens, COMMITTED_TOKENS);
            // First token excluded, decode window = ttft -> last token.
            let decode_ms = sample.total_ms - sample.ttft_ms;
            let expected_tok_s = (sample.n_tokens as f64 - 1.0) / (decode_ms / 1000.0);
            assert!(
                (cell.aggregate_tok_s - expected_tok_s).abs() < 1e-6,
                "depth {}: got {}, expected {}",
                depth_cell.depth_tokens,
                cell.aggregate_tok_s,
                expected_tok_s,
            );
        }
    }

    #[test]
    fn a_corpus_that_cannot_be_read_fails_every_depth_cell_with_that_error() {
        let ep = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 };
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: Some(PathBuf::from("/does/not/exist/bank.ids")),
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        assert_eq!(record.cells.len(), 3);
        assert!(record.cells.iter().all(|c| c.cell.error.is_some()));
        assert!(!record.all_cold());
    }

    #[test]
    fn cell_at_finds_the_matching_depth() {
        let ep = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 };
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: None,
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        assert!(record.cell_at(DEPTH_98K).is_some());
        assert!(record.cell_at(999).is_none());
    }

    #[test]
    fn a_record_round_trips_through_json_and_renders() {
        let ep = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 };
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: None,
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        let json = record.to_json().expect("serialize");
        assert_eq!(Record::from_json(&json).expect("parse"), record);
        let text = record.render();
        for expected in ["session=S1", "24K", "98K", "196K", "tok/s"] {
            assert!(text.contains(expected), "render must mention {expected}:\n{text}");
        }
    }
}
