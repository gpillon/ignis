//! Canary oracle tooling (P1-04, GitHub #40): a recorder that captures the
//! reference engine's greedy completions on the canary suite as a fixture,
//! and a comparer that reports how well a candidate stream agrees with it.
//!
//! This is the *tooling* half of the oracle (spec `01-device-resident-forward`
//! §"Oracle (two levels)"): the fixture format, the recorder, and the
//! agreement math for both scorings (ADR 0014) — teacher-forced next-token
//! agreement, which is the G1 correctness floor, and the free-running
//! comparison, which is a diagnostic. Recording the fixture against the real
//! reference engine is P1-05 (a human/GPU step); this module is fully
//! CPU-testable against a mock [`Endpoint`] and a mock [`Tokenize`]
//! implementation — no artifact, no GPU (prior art: `record.rs`'s capture
//! proxy tests).
//!
//! The canary prompts themselves are [`crate::canary::CANARIES`] (already
//! committed and referenced by `CONTEXT.md`'s "Canary suite" entry) — the
//! oracle fixture and the self-consistency check (ADR 0007) run the *same*
//! fixed prompt set, just checked two different ways.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::canary::CANARIES;
use crate::client::{Endpoint, Request};
use crate::trace::RequestClass;

/// Encodes text to token ids. A seam over the artifact's real tokenizer
/// ([`ignis_artifact::Tokenizer`], implemented below) so the recorder and
/// comparer are unit-testable with a trivial mock — no `.ninfer` artifact
/// needed.
pub trait Tokenize: Send + Sync {
    /// Encode `text` to token ids. Returns a human-readable error on failure
    /// (a malformed tokenizer input is a recorded failure, not a panic).
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
}

impl Tokenize for ignis_artifact::Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        ignis_artifact::Tokenizer::encode(self, text).map_err(|e| e.to_string())
    }
}

/// One canary's recorded (or candidate) output: the prompt, the generated
/// text, and its token ids under the artifact's tokenizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixturePrompt {
    /// The canary's id (matches [`crate::canary::Canary::id`]).
    pub id: String,
    /// The canary prompt sent (kept so the fixture is self-describing).
    pub prompt: String,
    /// The engine's greedy completion text.
    pub text: String,
    /// `text` tokenized with the artifact's tokenizer.
    pub token_ids: Vec<u32>,
}

/// The canary oracle fixture: `oracle record`'s output, `oracle compare`'s
/// input. Round-trips through JSON (`to_json` / `from_json`, `write` /
/// `read`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fixture {
    /// The recording engine's model id (`GET /v1/models`), so a fixture is
    /// traceable to the artifact it was recorded against.
    pub model: String,
    /// The `max_tokens` budget every canary was recorded with.
    pub max_tokens: u32,
    /// One entry per canary, in [`CANARIES`] order.
    pub prompts: Vec<FixturePrompt>,
}

impl Fixture {
    /// Serialize to pretty JSON (the on-disk fixture format).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialize the fixture: {e}"))
    }

    /// Parse a fixture from JSON.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("parse the fixture: {e}"))
    }

    /// Write the fixture to `path` as pretty JSON.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_json()?)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Read a fixture previously written by [`Fixture::write`].
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    /// The fixture's prompt entry for canary `id`, if recorded.
    pub fn prompt(&self, id: &str) -> Option<&FixturePrompt> {
        self.prompts.iter().find(|p| p.id == id)
    }
}

/// Record the canary oracle fixture: for each canary prompt ([`CANARIES`]),
/// send it to `ep` (greedy — `HttpEndpoint` fixes `temperature: 0` /
/// `seed: 0`, the exact-argmax contract), and tokenize the returned text
/// with `tokenizer`. `model` and `max_tokens` are stamped into the fixture
/// as recorded (the caller probes `model` via `Endpoint::list_models`-style
/// preflight, matching `canary`/`replay`'s existing CLI pattern).
///
/// `enable_thinking: false` is sent on every request (GitHub #68): the
/// oracle fixture was recorded from the reference with thinking off, so a
/// candidate fixture recorded any other way compares a template difference
/// instead of engine agreement.
pub fn record(
    ep: &dyn Endpoint,
    tokenizer: &dyn Tokenize,
    model: String,
    max_tokens: u32,
) -> Result<Fixture, String> {
    let mut prompts = Vec::with_capacity(CANARIES.len());
    for c in CANARIES {
        let req = Request {
            id: format!("oracle-{}", c.id),
            class: RequestClass::Sub,
            prompt: c.prompt.to_string(),
            max_tokens,
            stream: false,
            include_usage: false,
            enable_thinking: Some(false),
        };
        let outcome = ep
            .complete(&req)
            .map_err(|e| format!("canary {}: {e}", c.id))?;
        let token_ids = tokenizer
            .encode(&outcome.output)
            .map_err(|e| format!("canary {}: tokenize: {e}", c.id))?;
        prompts.push(FixturePrompt {
            id: c.id.to_string(),
            prompt: c.prompt.to_string(),
            text: outcome.output,
            token_ids,
        });
    }
    Ok(Fixture { model, max_tokens, prompts })
}

/// One canary's agreement result: how far a candidate token stream tracks
/// the oracle's, over the first `compared` positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgreementResult {
    /// The canary's id.
    pub id: String,
    /// Positions actually compared: `min(first_n, oracle.len())` — a
    /// candidate shorter than the oracle is *not* excused past its own
    /// length; a missing position counts as a mismatch (see
    /// [`compare_tokens`]).
    pub compared: usize,
    /// Matching positions among `compared`.
    pub agree: usize,
    /// `agree / compared` (`1.0` when `compared == 0` — nothing to
    /// disagree on).
    pub agreement: f64,
    /// The first mismatching position (0-indexed), or `None` when every
    /// compared position agreed.
    pub first_divergence: Option<usize>,
}

impl AgreementResult {
    /// `true` when every compared position agreed.
    pub fn is_full_agreement(&self) -> bool {
        self.first_divergence.is_none()
    }
}

/// Compare an oracle token stream against a candidate over the first
/// `first_n` oracle positions (fewer when the oracle itself is shorter). A
/// candidate position beyond the candidate's own length is a mismatch (a
/// truncated/early-stopped candidate is a divergence, not an excused gap).
/// Pure and CPU-only — the core of the comparer.
///
/// **Diagnostic only** (ADR 0014). This scores a *free-running* candidate:
/// the engine fed its own emitted tokens back in, so the two streams stop
/// sharing a prefix after the first divergence and every later position is
/// decorrelated. That measures continuation similarity, not whether the
/// forward pass is grossly broken, so it is no longer the G1 floor — see
/// [`score_teacher_forced`].
pub fn compare_tokens(oracle: &[u32], candidate: &[u32], first_n: usize) -> AgreementResult {
    let compared = first_n.min(oracle.len());
    let mut agree = 0usize;
    let mut first_divergence = None;
    for i in 0..compared {
        let matches = candidate.get(i) == Some(&oracle[i]);
        if matches {
            agree += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(i);
        }
    }
    let agreement = if compared == 0 { 1.0 } else { agree as f64 / compared as f64 };
    AgreementResult {
        id: String::new(),
        compared,
        agree,
        agreement,
        first_divergence,
    }
}

/// Compare every canary in `oracle` against its counterpart in `candidate`
/// (matched by id), over the first `first_n` oracle tokens. Returns one
/// [`AgreementResult`] per oracle prompt, in the oracle's order. A canary
/// missing from `candidate` is an error (a partial candidate is a recorder
/// bug, not a silent skip).
pub fn compare_fixtures(
    oracle: &Fixture,
    candidate: &Fixture,
    first_n: usize,
) -> Result<Vec<AgreementResult>, String> {
    oracle
        .prompts
        .iter()
        .map(|o| {
            let c = candidate
                .prompt(&o.id)
                .ok_or_else(|| format!("candidate is missing canary {}", o.id))?;
            let mut result = compare_tokens(&o.token_ids, &c.token_ids, first_n);
            result.id = o.id.clone();
            Ok(result)
        })
        .collect()
}

/// The overall free-running agreement across every compared canary: total
/// matches over total compared positions (not a per-canary average — a
/// canary with more compared tokens weighs proportionally more).
///
/// **Diagnostic only** (ADR 0014): this is the free-running figure, and it
/// is no longer the G1 correctness floor. Use
/// [`overall_teacher_forced_agreement`] for the gate.
pub fn overall_agreement(results: &[AgreementResult]) -> f64 {
    let (agree, compared) = results
        .iter()
        .fold((0usize, 0usize), |(a, c), r| (a + r.agree, c + r.compared));
    if compared == 0 {
        1.0
    } else {
        agree as f64 / compared as f64
    }
}

// ── the G1 gate: teacher-forced next-token agreement (ADR 0014) ───────────

/// The G1 correctness floor (ADR 0014): overall teacher-forced next-token
/// agreement across the canary suite. Deliberately a *sanity* floor — it
/// exists to catch gross implementation errors (wrong layouts, missing ops,
/// broken state wiring, wrong positions, corrupted activations), not to
/// require numerical or token-level parity with the reference (ADR 0007).
pub const G1_AGREEMENT_FLOOR: f64 = 0.95;

/// One position where the engine's teacher-forced next token differed from
/// the oracle's. Recorded for reporting only — a mismatch is a mismatch, and
/// no class of mismatch (exact BF16 logit ties included) is waived or
/// excluded from the score (ADR 0014).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherForcedMismatch {
    /// The 0-indexed oracle position.
    pub position: usize,
    /// The oracle's recorded token at this position.
    pub expected: u32,
    /// The engine's greedy next-token argmax given the oracle's prefix, or
    /// `None` when the engine produced no prediction for this position.
    pub predicted: Option<u32>,
}

/// One canary's teacher-forced result: how often the engine's greedy
/// next-token argmax matched the oracle's token, each position scored
/// against the *oracle's own* prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherForcedResult {
    /// The canary's id.
    pub id: String,
    /// Positions actually scored: `min(first_n, oracle.len())`.
    pub compared: usize,
    /// Matching positions among `compared`.
    pub agree: usize,
    /// `agree / compared` (`1.0` when `compared == 0`).
    pub agreement: f64,
    /// Every mismatching position, in order.
    pub mismatches: Vec<TeacherForcedMismatch>,
}

/// Score one canary teacher-forced (ADR 0014, the G1 metric).
///
/// `predictions[i]` is the engine's greedy next-token argmax **after being
/// fed the oracle's tokens `[0..i)`** on top of the canary prompt — so
/// position `i` is judged on a prefix both engines share, and a divergence
/// cannot cascade into the positions after it. This is the whole point of
/// the metric: free-running comparison ([`compare_tokens`]) lets one flip
/// decorrelate the rest of the stream.
///
/// A position the engine produced no prediction for counts as a mismatch,
/// never an excused gap (same rule as [`compare_tokens`]). Pure and
/// CPU-only, so the gate's arithmetic is unit-tested without a GPU; the
/// GPU-side driver is `crates/server/tests/oracle_teacher_forced_gpu.rs`.
pub fn score_teacher_forced(
    id: &str,
    oracle: &[u32],
    predictions: &[u32],
    first_n: usize,
) -> TeacherForcedResult {
    let compared = first_n.min(oracle.len());
    let mut agree = 0usize;
    let mut mismatches = Vec::new();
    for i in 0..compared {
        let predicted = predictions.get(i).copied();
        if predicted == Some(oracle[i]) {
            agree += 1;
        } else {
            mismatches.push(TeacherForcedMismatch {
                position: i,
                expected: oracle[i],
                predicted,
            });
        }
    }
    let agreement = if compared == 0 { 1.0 } else { agree as f64 / compared as f64 };
    TeacherForcedResult {
        id: id.to_string(),
        compared,
        agree,
        agreement,
        mismatches,
    }
}

/// The suite's overall teacher-forced agreement: total matches over total
/// scored positions (not a per-canary average — a canary with more scored
/// positions weighs proportionally more). This is the figure
/// [`meets_g1_floor`] judges.
pub fn overall_teacher_forced_agreement(results: &[TeacherForcedResult]) -> f64 {
    let (agree, compared) = results
        .iter()
        .fold((0usize, 0usize), |(a, c), r| (a + r.agree, c + r.compared));
    if compared == 0 {
        1.0
    } else {
        agree as f64 / compared as f64
    }
}

/// Whether an overall teacher-forced agreement clears the G1 floor
/// ([`G1_AGREEMENT_FLOOR`]). No tolerance, no tie waiver (ADR 0014).
pub fn meets_g1_floor(overall: f64) -> bool {
    overall >= G1_AGREEMENT_FLOOR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{MockEndpoint, Outcome};

    /// A trivial deterministic tokenizer for tests: one id per whitespace
    /// word (a stable checksum of the word, so the same word always maps to
    /// the same id, and different words almost never collide) — no real
    /// artifact needed.
    struct WordIdTokenizer;

    impl Tokenize for WordIdTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
            Ok(text
                .split_whitespace()
                .map(|w| w.bytes().fold(0u32, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u32)))
                .collect())
        }
    }

    fn outcome(text: &str) -> Outcome {
        Outcome {
            ttft_ms: 1.0,
            total_ms: 1.0,
            n_tokens: text.split_whitespace().count() as u32,
            output: text.to_string(),
            prompt_tokens: None,
            cached_prompt_tokens: None,
            token_times_ms: Vec::new(),
        }
    }

    // ── fixture round-trip ────────────────────────────────────────────────

    #[test]
    fn a_fixture_round_trips_through_json() {
        let fixture = Fixture {
            model: "qwen-3.8-27b".into(),
            max_tokens: 32,
            prompts: vec![FixturePrompt {
                id: "rust-hello".into(),
                prompt: "what does main do?".into(),
                text: "it prints hi".into(),
                token_ids: vec![1, 2, 3],
            }],
        };
        let json = fixture.to_json().expect("serialize");
        let back = Fixture::from_json(&json).expect("parse");
        assert_eq!(back, fixture);
    }

    #[test]
    fn a_fixture_round_trips_through_a_file() {
        let path = std::env::temp_dir().join(format!(
            "ignis-bench-oracle-fixture-{}.json",
            std::process::id()
        ));
        let fixture = Fixture {
            model: "qwen-3.8-27b".into(),
            max_tokens: 32,
            prompts: vec![FixturePrompt {
                id: "math-greedy".into(),
                prompt: "2*3+4".into(),
                text: "10".into(),
                token_ids: vec![7],
            }],
        };
        fixture.write(&path).expect("write the fixture");
        let back = Fixture::read(&path).expect("read the fixture");
        assert_eq!(back, fixture);
        let _ = std::fs::remove_file(&path);
    }

    // ── the comparer's agreement math ─────────────────────────────────────

    #[test]
    fn identical_streams_agree_fully() {
        let r = compare_tokens(&[1, 2, 3, 4], &[1, 2, 3, 4], 4);
        assert_eq!(r.compared, 4);
        assert_eq!(r.agree, 4);
        assert_eq!(r.agreement, 1.0);
        assert_eq!(r.first_divergence, None);
        assert!(r.is_full_agreement());
    }

    #[test]
    fn a_partial_stream_diverges_at_the_first_mismatch() {
        let r = compare_tokens(&[1, 2, 3, 4], &[1, 2, 9, 4], 4);
        assert_eq!(r.compared, 4);
        assert_eq!(r.agree, 3);
        assert_eq!(r.agreement, 0.75);
        assert_eq!(r.first_divergence, Some(2));
        assert!(!r.is_full_agreement());
    }

    #[test]
    fn a_fully_divergent_stream_disagrees_from_position_zero() {
        let r = compare_tokens(&[1, 2, 3], &[9, 9, 9], 3);
        assert_eq!(r.agree, 0);
        assert_eq!(r.agreement, 0.0);
        assert_eq!(r.first_divergence, Some(0));
    }

    #[test]
    fn first_n_caps_the_compared_window() {
        // Only the first 2 positions are compared even though both streams
        // are longer, and even though they diverge at position 3.
        let r = compare_tokens(&[1, 2, 3, 4], &[1, 2, 9, 9], 2);
        assert_eq!(r.compared, 2);
        assert_eq!(r.agree, 2);
        assert_eq!(r.first_divergence, None);
    }

    #[test]
    fn a_short_oracle_shrinks_the_compared_window() {
        // The oracle has only 2 tokens; first_n = 10 does not manufacture
        // positions past it.
        let r = compare_tokens(&[1, 2], &[1, 2, 3, 4, 5], 10);
        assert_eq!(r.compared, 2);
        assert_eq!(r.agree, 2);
    }

    #[test]
    fn a_truncated_candidate_counts_missing_positions_as_mismatches() {
        // The candidate stopped after 2 tokens (e.g. early EOS); the oracle
        // has 4. The missing positions are divergences, not excused.
        let r = compare_tokens(&[1, 2, 3, 4], &[1, 2], 4);
        assert_eq!(r.compared, 4);
        assert_eq!(r.agree, 2);
        assert_eq!(r.first_divergence, Some(2));
    }

    #[test]
    fn an_empty_comparison_window_is_full_agreement_by_convention() {
        let r = compare_tokens(&[], &[1, 2, 3], 10);
        assert_eq!(r.compared, 0);
        assert_eq!(r.agreement, 1.0);
        assert_eq!(r.first_divergence, None);
    }

    #[test]
    fn compare_fixtures_matches_by_id_and_rejects_a_missing_canary() {
        let oracle = Fixture {
            model: "ref".into(),
            max_tokens: 8,
            prompts: vec![
                FixturePrompt { id: "a".into(), prompt: "pa".into(), text: "ta".into(), token_ids: vec![1, 2] },
                FixturePrompt { id: "b".into(), prompt: "pb".into(), text: "tb".into(), token_ids: vec![3, 4] },
            ],
        };
        let candidate = Fixture {
            model: "ours".into(),
            max_tokens: 8,
            prompts: vec![FixturePrompt {
                id: "a".into(),
                prompt: "pa".into(),
                text: "ta".into(),
                token_ids: vec![1, 2],
            }],
        };
        let err = compare_fixtures(&oracle, &candidate, 8)
            .expect_err("candidate is missing canary b");
        assert!(err.contains('b'));

        let candidate_complete = Fixture {
            prompts: vec![
                oracle.prompts[0].clone(),
                FixturePrompt { id: "b".into(), prompt: "pb".into(), text: "tb".into(), token_ids: vec![3, 9] },
            ],
            ..candidate
        };
        let results = compare_fixtures(&oracle, &candidate_complete, 8).expect("both canaries present");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!(results[0].is_full_agreement());
        assert_eq!(results[1].id, "b");
        assert_eq!(results[1].first_divergence, Some(1));
    }

    #[test]
    fn overall_agreement_weighs_by_compared_positions_not_per_canary_average() {
        let results = vec![
            AgreementResult { id: "a".into(), compared: 10, agree: 10, agreement: 1.0, first_divergence: None },
            AgreementResult { id: "b".into(), compared: 2, agree: 0, agreement: 0.0, first_divergence: Some(0) },
        ];
        // 10/12, not (1.0 + 0.0) / 2 = 0.5.
        assert!((overall_agreement(&results) - 10.0 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn overall_agreement_of_no_results_is_full_by_convention() {
        assert_eq!(overall_agreement(&[]), 1.0);
    }

    // ── the recorder ──────────────────────────────────────────────────────

    #[test]
    fn record_builds_one_fixture_prompt_per_canary() {
        let ep = MockEndpoint::new(
            CANARIES.iter().map(|c| outcome(&format!("answer for {}", c.id))).collect(),
        );
        let fixture = record(&ep, &WordIdTokenizer, "qwen-3.8-27b".into(), 32).expect("record");
        assert_eq!(fixture.model, "qwen-3.8-27b");
        assert_eq!(fixture.max_tokens, 32);
        assert_eq!(fixture.prompts.len(), CANARIES.len());
        for (p, c) in fixture.prompts.iter().zip(CANARIES) {
            assert_eq!(p.id, c.id);
            assert_eq!(p.prompt, c.prompt);
            assert_eq!(p.text, format!("answer for {}", c.id));
            assert!(!p.token_ids.is_empty(), "the mock tokenizer produced ids");
        }
    }

    #[test]
    fn record_sends_max_tokens_and_greedy_shaped_requests() {
        let ep = MockEndpoint::new(CANARIES.iter().map(|_| outcome("x")).collect());
        record(&ep, &WordIdTokenizer, "m".into(), 7).expect("record");
        let received = ep.received.lock().unwrap();
        assert_eq!(received.len(), CANARIES.len());
        assert!(received.iter().all(|r| r.max_tokens == 7 && !r.stream));
        // GitHub #68: the recorder drives thinking disabled — the recorded
        // candidate must describe the same prompt as the thinking-off
        // reference fixture, or the G1 comparison measures a template
        // difference instead of engine agreement.
        assert!(
            received.iter().all(|r| r.enable_thinking == Some(false)),
            "oracle record must send enable_thinking: false"
        );
    }

    #[test]
    fn a_failing_endpoint_fails_the_recording_not_a_panic() {
        struct Failing;
        impl Endpoint for Failing {
            fn complete(&self, _req: &Request) -> Result<Outcome, String> {
                Err("engine down".into())
            }
        }
        let err = record(&Failing, &WordIdTokenizer, "m".into(), 8).expect_err("the engine is down");
        assert!(err.contains("engine down"));
    }

    // ── the G1 gate: teacher-forced scoring (ADR 0014) ────────────────────

    #[test]
    fn every_matching_prediction_is_full_teacher_forced_agreement() {
        let r = score_teacher_forced("c", &[1, 2, 3, 4], &[1, 2, 3, 4], 32);
        assert_eq!(r.compared, 4);
        assert_eq!(r.agree, 4);
        assert_eq!(r.agreement, 1.0);
        assert!(r.mismatches.is_empty());
    }

    #[test]
    fn a_teacher_forced_mismatch_does_not_cascade_into_later_positions() {
        // The whole point of the metric (ADR 0014): position 0 is wrong, but
        // every later position is still scored against the oracle's own
        // prefix, so it can still agree. Free-running comparison would have
        // decorrelated everything after the flip.
        let r = score_teacher_forced("c", &[1, 2, 3, 4], &[9, 2, 3, 4], 32);
        assert_eq!(r.agree, 3);
        assert_eq!(r.compared, 4);
        assert_eq!(r.mismatches.len(), 1);
        assert_eq!(r.mismatches[0].position, 0);
        assert_eq!(r.mismatches[0].expected, 1);
        assert_eq!(r.mismatches[0].predicted, Some(9));
    }

    #[test]
    fn a_missing_teacher_forced_prediction_counts_as_a_mismatch() {
        let r = score_teacher_forced("c", &[1, 2, 3, 4], &[1, 2], 4);
        assert_eq!(r.compared, 4);
        assert_eq!(r.agree, 2);
        assert_eq!(r.mismatches.len(), 2);
        assert_eq!(r.mismatches[0].predicted, None);
    }

    #[test]
    fn a_short_oracle_shrinks_the_teacher_forced_window() {
        let r = score_teacher_forced("c", &[1, 2], &[1, 2, 3, 4], 32);
        assert_eq!(r.compared, 2);
        assert_eq!(r.agree, 2);
    }

    #[test]
    fn an_empty_teacher_forced_window_is_full_agreement_by_convention() {
        let r = score_teacher_forced("c", &[], &[1, 2], 32);
        assert_eq!(r.compared, 0);
        assert_eq!(r.agreement, 1.0);
    }

    #[test]
    fn overall_teacher_forced_agreement_weighs_by_position_not_by_canary() {
        // 1/1 on a tiny canary and 1/3 on a longer one is 2/4 = 50%, not the
        // per-canary average of 66.7%.
        let results = vec![
            score_teacher_forced("small", &[1], &[1], 32),
            score_teacher_forced("big", &[1, 2, 3], &[1, 9, 9], 32),
        ];
        assert_eq!(overall_teacher_forced_agreement(&results), 0.5);
    }

    #[test]
    fn the_g1_floor_is_ninety_five_percent_with_no_tolerance() {
        assert_eq!(G1_AGREEMENT_FLOOR, 0.95);
        assert!(meets_g1_floor(0.95));
        assert!(meets_g1_floor(0.971));
        assert!(!meets_g1_floor(0.9499));
        // The free-running figure the old gate scored (GitHub #72): 53/102.
        assert!(!meets_g1_floor(53.0 / 102.0));
    }

    #[test]
    fn the_recorded_g1_measurement_clears_the_floor_with_ties_counted_as_mismatches() {
        // The 2026-09-07 measurement (ADR 0014), reproduced as arithmetic:
        // rust-hello 21/21, rust-sort 30/32, math-greedy 32/32,
        // explain-reverse 16/17 = 99/102. The two BF16 exact-tie positions
        // are counted as mismatches, not waived, and it still passes.
        let results = vec![
            TeacherForcedResult { id: "rust-hello".into(), compared: 21, agree: 21, agreement: 1.0, mismatches: vec![] },
            TeacherForcedResult { id: "rust-sort".into(), compared: 32, agree: 30, agreement: 30.0 / 32.0, mismatches: vec![] },
            TeacherForcedResult { id: "math-greedy".into(), compared: 32, agree: 32, agreement: 1.0, mismatches: vec![] },
            TeacherForcedResult { id: "explain-reverse".into(), compared: 17, agree: 16, agreement: 16.0 / 17.0, mismatches: vec![] },
        ];
        let overall = overall_teacher_forced_agreement(&results);
        assert_eq!(overall, 99.0 / 102.0);
        assert!(meets_g1_floor(overall), "99/102 = 97.1% clears the 95% floor");
    }
}
