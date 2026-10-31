//! The G5 measurement instrument (P5-07, GitHub #151): committed tok/s at
//! three prompt depths, reusing [`crate::g3`]'s C=1 throughput cell
//! (concurrency 1) instead of a new instrument. The verdict half lives in
//! [`crate::g5_gate`].
//!
//! Spec: `docs/specs/runtime/05-speculative-decoding.md` — not written
//! yet (#66: "to write when G4 lands"); this module is built from #66's and
//! #151's own acceptance criteria plus `docs/REVIEW-2026-09-05.md` §6
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
use crate::g3::{measure_throughput_cell_with, ThroughputCell, C1_CONCURRENCY};
use crate::time::{unix_now, utc_timestamp};
use crate::ttft::{self, load_corpus, PromptSet, PromptSetSpec, PromptTemplate};

/// The three prompt depths spec 05's Gate G5 table measures
/// (`docs/REVIEW-2026-09-05.md` §6 Phase 5: "24K / 98K / 196K"), each
/// `X * 1024` post-template tokens.
pub const DEPTH_24K: u32 = 24 * 1024;
pub const DEPTH_98K: u32 = 98 * 1024;
pub const DEPTH_196K: u32 = 196 * 1024;

/// The depths a `g5` run measures, in the order they are reported.
pub const DEPTHS: [u32; 3] = [DEPTH_24K, DEPTH_98K, DEPTH_196K];

/// The committed-token budget every depth cell asks for (#151: "512
/// committed tokens").
pub const COMMITTED_TOKENS: u32 = 512;

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

/// The instruction every depth prompt ends with (#159). The reference
/// ignores `ignore_eos` and a `logit_bias` exclusion has been recorded as
/// insufficient against it, so the prompt is the one lever both engines see
/// identically — the same one `g3`'s ITL lanes use.
pub fn decode_instruction(committed_tokens: u32) -> String {
    format!("Produce at least {committed_tokens} tokens. Do not stop, conclude, or emit EOS earlier.")
}

/// Void every completed sample that committed fewer than `committed_tokens`
/// (#159): a run that stopped at EOS after a few verify rounds measures
/// those rounds, not decode, and must not enter a ratio against an engine
/// that ran the whole budget.
///
/// Only a cell whose prompt carried the [`decode_instruction`] has anything to
/// hold the engine to (`instructed`): the filler path sends none, so a short
/// run there is the engine's own answer, not a stop the prompt forbade.
fn require_committed_budget(mut cell: ThroughputCell, committed_tokens: u32, instructed: bool) -> ThroughputCell {
    if !instructed {
        return cell;
    }
    for sample in &mut cell.samples {
        if sample.ok && !sample.void && sample.n_tokens < committed_tokens {
            sample.void = true;
            sample.void_reason = Some(format!(
                "the engine stopped after {} of the {committed_tokens} committed tokens: a short \
                 run measures a few verify rounds, not decode",
                sample.n_tokens
            ));
        }
    }
    cell
}

/// The prompts one `g5` run sends, one per depth in [`DEPTHS`] order,
/// generated together before the first is sent (GitHub #443) and numbered
/// across the run ([`ttft::generate_run_prompts`]): each depth's prompt
/// starts with its own nonce, or its window in its own third of `corpus`,
/// so an engine with prompt reuse has nothing a shallower depth left behind
/// to serve a deeper one from, and a run whose prompts would share more is
/// refused before it starts. Only the corpus path carries `instruction`
/// (#162). A prompt that cannot be generated fails its own depth.
fn depth_prompts(
    template: &dyn PromptTemplate,
    corpus: Option<&[u32]>,
    instruction: &str,
) -> Result<[Result<PromptSet, String>; DEPTHS.len()], String> {
    let labels = DEPTHS.map(|depth| format!("{depth}-token depth"));
    let suffix = if corpus.is_some() { instruction } else { "" };
    ttft::generate_run_prompts(
        template,
        corpus,
        std::array::from_fn(|index| PromptSetSpec {
            label: &labels[index],
            count: C1_CONCURRENCY,
            target_tokens: DEPTHS[index] as usize,
            suffix,
        }),
    )
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
    /// One cell per depth measured, in [`DEPTHS`] order. Each cell's own
    /// [`ThroughputCell::prompt_tokens`] *is* the depth it was measured at
    /// — no separate depth field to keep in sync with it.
    pub cells: Vec<ThroughputCell>,
}

impl Record {
    /// This record's cell at `depth_tokens`, if measured.
    pub fn cell_at(&self, depth_tokens: u32) -> Option<&ThroughputCell> {
        self.cells.iter().find(|c| c.prompt_tokens == depth_tokens)
    }

    /// True when every depth cell produced a full set of cold, complete
    /// samples — the property a gate verdict may be computed over.
    pub fn all_cold(&self) -> bool {
        !self.cells.is_empty() && self.cells.iter().all(|c| c.all_cold())
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
        for cell in &self.cells {
            match &cell.error {
                Some(err) => out.push_str(&format!("  {:>4}K FAILED: {err}\n", cell.prompt_tokens / 1024)),
                None => {
                    let bad = cell.bad_samples().len();
                    out.push_str(&format!(
                        "  {:>4}K  {:>9.1} tok/s  over {} sequence(s){}\n",
                        cell.prompt_tokens / 1024,
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
/// scale) filler generator. The depths' prompts are the run's
/// (`depth_prompts`, GitHub #443), generated before the first request is
/// sent; a run they refuse fails every depth cell with that refusal.
pub fn measure(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    engine: String,
    endpoint: String,
    cfg: &G5Config,
) -> Record {
    let corpus = cfg.corpus.as_ref().map(|path| load_corpus(path)).transpose();
    let instruction = decode_instruction(cfg.committed_tokens);
    let instructed = cfg.corpus.is_some();
    let cells = match corpus.and_then(|bank| depth_prompts(template, bank.as_deref(), &instruction)) {
        Err(error) => DEPTHS.iter().map(|&depth| failed_cell(depth, cfg.committed_tokens, error.clone())).collect(),
        Ok(sets) => DEPTHS
            .iter()
            .zip(sets)
            .map(|(&depth, set)| {
                let cell = measure_throughput_cell_with(ep, depth, cfg.committed_tokens, C1_CONCURRENCY, || set);
                require_committed_budget(cell, cfg.committed_tokens, instructed)
            })
            .collect(),
    };
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
    use crate::g3::measure_throughput_cell;
    use crate::testutil::{PromptReuse, Steady, CHAT_FOOTER, CHAT_HEADER};

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

    /// The bank the corpus tests cut from — mirrors `g3.rs`'s / `g4.rs`'s
    /// own `corpus_bank()`, sized past the largest depth ([`DEPTH_196K`]).
    /// The real `DEPTHS` (up to 200,704) go through the corpus path in
    /// every test below, never the filler word-growth generator: this
    /// module's own doc comment already names that path as "impractical at
    /// this scale" (O(n²) re-encoding). A bank shorter than a depth tiles
    /// (#159); this one is longer so the plain path stays covered too.
    fn corpus_bank() -> Vec<u32> {
        (0..(DEPTH_196K as usize + 1_000)).map(|i| ((i * 37) % 900) as u32 + 100).collect()
    }

    /// Write `bank` as a `--corpus` file in a process- and call-unique temp
    /// directory (mirrors `ttft.rs`'s own corpus-file test). Returns the
    /// directory (for cleanup) and the file path.
    fn write_corpus_file(bank: &[u32]) -> (PathBuf, PathBuf) {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("ignis-bench-g5-corpus-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("corpus.ids");
        let text = bank.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
        std::fs::write(&path, text).expect("write corpus");
        (dir, path)
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
        let (dir, corpus) = write_corpus_file(&corpus_bank());
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: Some(corpus),
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(record.cells.len(), 3);
        for cell in &record.cells {
            assert!(cell.all_cold(), "{cell:?}");
            assert_eq!(cell.samples.len(), 1, "C=1: one sequence per depth");
            let sample = &cell.samples[0];
            assert_eq!(sample.n_tokens, COMMITTED_TOKENS);
            // First token excluded, decode window = ttft -> last token.
            let decode_ms = sample.total_ms - sample.ttft_ms;
            let expected_tok_s = (sample.n_tokens as f64 - 1.0) / (decode_ms / 1000.0);
            assert!(
                (cell.aggregate_tok_s - expected_tok_s).abs() < 1e-6,
                "depth {}: got {}, expected {}",
                cell.prompt_tokens,
                cell.aggregate_tok_s,
                expected_tok_s,
            );
        }
    }

    fn cfg_with(corpus: PathBuf) -> G5Config {
        G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: Some(corpus),
        }
    }

    /// An endpoint that stops every request after `stop_after` tokens (the
    /// reference at its own EOS) and remembers the prompts it was sent.
    struct RecordingEndpoint {
        stop_after: Option<u32>,
        prompts: std::sync::Mutex<Vec<String>>,
    }

    impl Endpoint for RecordingEndpoint {
        fn complete(&self, req: &Request) -> Result<Outcome, String> {
            self.prompts.lock().unwrap().push(req.prompt.clone());
            let mut short = req.clone();
            if let Some(n) = self.stop_after {
                short.max_tokens = short.max_tokens.min(n);
            }
            let mut out = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 }.complete(&short)?;
            if self.stop_after.is_some() {
                out.finish_reason = Some(FinishReason::Engine("stop".into()));
            }
            Ok(out)
        }
    }

    /// #159 A: the reference's bank is 65,536 ids; every depth still lands,
    /// and each depth's window starts elsewhere in the bank so prefix reuse
    /// cannot warm a deeper cell from a shallower one.
    #[test]
    fn a_bank_shorter_than_the_deepest_cell_is_tiled_from_a_distinct_offset_per_depth() {
        let bank: Vec<u32> = (0..65_536).map(|i| ((i * 37) % 900) as u32 + 100).collect();
        let (dir, corpus) = write_corpus_file(&bank);
        let ep = RecordingEndpoint { stop_after: None, prompts: Default::default() };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg_with(corpus));
        std::fs::remove_dir_all(&dir).ok();
        assert!(record.all_cold(), "{}", record.render());
        let prompts = ep.prompts.into_inner().unwrap();
        assert_eq!(prompts.len(), 3);
        let heads: Vec<String> =
            prompts.iter().map(|p| p.split_whitespace().take(8).collect::<Vec<_>>().join(" ")).collect();
        assert_ne!(heads[0], heads[1], "24K and 98K must not share a prefix");
        assert_ne!(heads[1], heads[2], "98K and 196K must not share a prefix");
        assert_ne!(heads[0], heads[2], "24K and 196K must not share a prefix");
    }

    /// #159 B: the reference cannot be told to ignore EOS, so the prompt
    /// asks for a long answer, the same text to both engines.
    #[test]
    fn every_depth_prompt_asks_for_at_least_the_committed_budget() {
        let (dir, corpus) = write_corpus_file(&corpus_bank());
        let ep = RecordingEndpoint { stop_after: None, prompts: Default::default() };
        measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg_with(corpus));
        std::fs::remove_dir_all(&dir).ok();
        for prompt in ep.prompts.into_inner().unwrap() {
            assert!(prompt.ends_with(&decode_instruction(COMMITTED_TOKENS)), "{}", &prompt[prompt.len() - 200..]);
        }
    }

    /// #159 B: a sample that stopped short of the committed budget measured
    /// a few rounds, not decode: void, with the reason in the record.
    #[test]
    fn a_sample_that_stops_short_of_the_committed_budget_is_void() {
        let (dir, corpus) = write_corpus_file(&corpus_bank());
        let ep = RecordingEndpoint { stop_after: Some(10), prompts: Default::default() };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg_with(corpus));
        std::fs::remove_dir_all(&dir).ok();
        assert!(!record.all_cold());
        for cell in &record.cells {
            let sample = &cell.samples[0];
            assert!(sample.void, "{sample:?}");
            let reason = sample.void_reason.as_deref().unwrap_or("");
            assert!(reason.contains("10") && reason.contains(&COMMITTED_TOKENS.to_string()), "{reason}");
        }
    }

    /// #162: the filler path sends no instruction, so a short run there is
    /// not held to the committed budget.
    #[test]
    fn a_cell_without_the_instruction_is_not_held_to_the_committed_budget() {
        let ep = RecordingEndpoint { stop_after: Some(10), prompts: Default::default() };
        let cell = measure_throughput_cell(&ep, &MockTemplate, 64, 512, C1_CONCURRENCY);
        let held = require_committed_budget(cell.clone(), 512, true);
        assert!(held.samples.iter().all(|s| s.void), "the instructed path voids a short run");
        let free = require_committed_budget(cell, 512, false);
        assert!(free.samples.iter().all(|s| !s.void), "{:?}", free.samples);
    }

    /// #162, GitHub #443: a bank no rotation frees, one id throughout, opens
    /// every depth's window on the same ids; the run refuses to start,
    /// naming two depths, and sends nothing. A bank whose period only aligns
    /// the depths' thirds is cut a rotation of its own per depth
    /// (`a_whole_run_cut_from_a_tiled_corpus_stays_cold_under_prompt_reuse`).
    #[test]
    fn a_bank_no_rotation_frees_refuses_the_run_naming_two_depths() {
        let (dir, corpus) = write_corpus_file(&[7; 12]);
        let ep = RecordingEndpoint { stop_after: None, prompts: Default::default() };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg_with(corpus));
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(record.cells.len(), 3);
        for cell in &record.cells {
            let error = cell.error.as_deref().expect("every depth refused");
            assert!(error.contains("24576-token depth prompt 0") && error.contains("100352-token depth prompt 0"), "{error}");
        }
        assert!(ep.prompts.into_inner().unwrap().is_empty(), "nothing was sent");
    }

    /// #162: the instruction the ITL lanes carry and the one G5 builds are
    /// one text.
    #[test]
    fn the_g5_instruction_is_the_itl_instruction_at_its_budget() {
        assert_eq!(decode_instruction(crate::g3::ITL_DECODE_MAX_TOKENS), crate::g3::ITL_DECODE_INSTRUCTION);
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
        assert!(record.cells.iter().all(|c| c.error.is_some()));
        assert!(!record.all_cold());
    }

    #[test]
    fn cell_at_finds_the_matching_depth() {
        let ep = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 };
        let (dir, corpus) = write_corpus_file(&corpus_bank());
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: Some(corpus),
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        std::fs::remove_dir_all(&dir).ok();
        assert!(record.cell_at(DEPTH_98K).is_some());
        assert!(record.cell_at(999).is_none());
    }

    #[test]
    fn a_record_round_trips_through_json_and_renders() {
        let ep = DeterministicEndpoint { ttft_ms: 40.0, interval_ms: 8.0 };
        let (dir, corpus) = write_corpus_file(&corpus_bank());
        let cfg = G5Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            committed_tokens: COMMITTED_TOKENS,
            corpus: Some(corpus),
        };
        let record = measure(&ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &cfg);
        std::fs::remove_dir_all(&dir).ok();
        let json = record.to_json().expect("serialize");
        assert_eq!(Record::from_json(&json).expect("parse"), record);
        let text = record.render();
        for expected in ["session=S1", "24K", "98K", "196K", "tok/s"] {
            assert!(text.contains(expected), "render must mention {expected}:\n{text}");
        }
    }

    // ── a whole run under prompt reuse (GitHub #443) ───────────────────────

    /// [`MockTemplate`] inside the artifact's chat template in shape, for the
    /// runs measured under prompt reuse.
    struct ChatTemplate;

    impl PromptTemplate for ChatTemplate {
        fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
            let mut ids: Vec<u32> = (0..CHAT_HEADER as u32).map(|i| 1_000 + i).collect();
            ids.extend(MockTemplate.encode_user_message(content)?);
            ids.extend((0..CHAT_FOOTER as u32).map(|i| 2_000 + i));
            Ok(ids)
        }
        fn decode(&self, ids: &[u32]) -> Result<String, String> {
            MockTemplate.decode(ids)
        }
    }

    fn reuse_run(corpus: Option<PathBuf>) -> G5Config {
        G5Config { profile: "prompt reuse on".into(), session: "S-reuse".into(), corpus, ..G5Config::default() }
    }

    fn depth_lengths() -> Vec<usize> {
        DEPTHS.iter().map(|&depth| depth as usize).collect()
    }

    /// GitHub #443: the filler path's three depths under prompt reuse come
    /// back cold, each prompt sharing no more than the chat template's header
    /// with the others, where every depth started with nonce 0 and the deeper
    /// two were served the 24K prompt's pages.
    #[test]
    fn a_whole_run_stays_cold_under_prompt_reuse() {
        let ep = PromptReuse::new(Steady, &ChatTemplate);
        let record = measure(&ep, &ChatTemplate, "mock-engine".into(), "http://mock".into(), &reuse_run(None));

        assert!(record.all_cold(), "{}", record.render());
        assert_eq!(ep.sharing_more_than(CHAT_HEADER), Vec::new(), "no two prompts share content");
        let sent = ep.admitted();
        assert_eq!(sent.iter().map(Vec::len).collect::<Vec<_>>(), depth_lengths());
        // The 24K prompt is the one it always was.
        let alone = crate::ttft::generate_cell_prompts(&ChatTemplate, DEPTH_24K as usize, 1).expect("prompt");
        assert_eq!(sent[0], alone.tokens[0]);
        // Not a vacuous mock: the run again sends the same prompts, served
        // from what the first run left.
        let again = measure(&ep, &ChatTemplate, "mock-engine".into(), "http://mock".into(), &reuse_run(None));
        assert!(again.cells.iter().all(|cell| !cell.all_cold()), "{}", again.render());
        assert_eq!(ep.admitted()[DEPTHS.len()..], sent[..], "the same prompts, byte for byte");
    }

    /// GitHub #443: a bank as long as the reference's and, like it, tiled
    /// from a shorter one, whose period is the depths' stride, so all three
    /// depths' windows open on the same ids: the run cuts the deeper two a
    /// rotation of their own and comes back cold, where it was refused.
    #[test]
    fn a_whole_run_cut_from_a_tiled_corpus_stays_cold_under_prompt_reuse() {
        const PERIOD: usize = 65_536 / 3;
        let bank: Vec<u32> = (0..65_536).map(|i| ((i % PERIOD) as u64 * 35_761 % 700_000) as u32 + 100).collect();
        let (dir, corpus) = write_corpus_file(&bank);
        let ep = PromptReuse::new(Steady, &ChatTemplate);
        let record = measure(&ep, &ChatTemplate, "mock-engine".into(), "http://mock".into(), &reuse_run(Some(corpus)));
        std::fs::remove_dir_all(&dir).ok();

        assert!(record.all_cold(), "{}", record.render());
        assert_eq!(ep.sharing_more_than(CHAT_HEADER), Vec::new(), "no two prompts share content");
        assert_eq!(ep.admitted().iter().map(Vec::len).collect::<Vec<_>>(), depth_lengths());
    }

    /// GitHub #443: on a bank whose depths' windows do not collide, the run
    /// cuts each depth the window it always did, the bank rotated by a third
    /// per depth.
    #[test]
    fn each_depth_is_cut_from_its_own_third_of_the_bank_as_before() {
        let bank: Vec<u32> = (0..65_536).map(|i| ((i * 37) % 900) as u32 + 100).collect();
        let (dir, corpus) = write_corpus_file(&bank);
        let ep = PromptReuse::new(Steady, &ChatTemplate);
        measure(&ep, &ChatTemplate, "mock-engine".into(), "http://mock".into(), &reuse_run(Some(corpus)));
        std::fs::remove_dir_all(&dir).ok();

        let instruction = decode_instruction(COMMITTED_TOKENS);
        for (index, (&depth, sent)) in DEPTHS.iter().zip(ep.admitted()).enumerate() {
            let mut rotated = bank.clone();
            rotated.rotate_left(index * bank.len() / DEPTHS.len());
            let alone = crate::ttft::generate_cell_prompts_from_corpus_with_suffix(
                &ChatTemplate,
                &rotated,
                depth as usize,
                1,
                &instruction,
            )
            .expect("cut");
            assert_eq!(sent, alone.tokens[0], "the {depth}-token depth");
        }
    }
}
