//! The phase 5 correctness oracle (P5-07, GitHub #151): greedy
//! spec-on/spec-off equivalence.
//!
//! DFlash2 (spec 05) verifies every drafted token against the target model
//! before it commits, so a correct implementation's greedy output is
//! byte-for-byte identical whether speculation is on or off — never merely
//! close. This runs [`crate::canary::CANARIES`] against two endpoints (one
//! with speculation on, one with it off — or the same endpoint restarted
//! between runs, #151) and compares the token sequences for **exact**
//! equality, naming the first divergence rather than scoring an agreement
//! percentage.
//!
//! Contrast [`crate::oracle`], which is a *tolerance* floor
//! ([`crate::oracle::G1_AGREEMENT_FLOOR`]) built for catching a grossly
//! broken forward pass. This check has no floor to tune: any divergence at
//! all means speculation changed what greedy decoding would otherwise have
//! produced, which is exactly the bug class it exists to catch.
//!
//! Pure comparison logic only (no I/O) beyond the two `Endpoint::complete`
//! calls and one `Tokenize::encode` call per side — the same seams
//! `crate::oracle` uses, so this module is fully CPU-testable against a
//! mock [`Endpoint`] and a mock [`crate::oracle::Tokenize`], no artifact, no
//! GPU (ADR 0006).

use serde::{Deserialize, Serialize};

use crate::canary::CANARIES;
use crate::client::{Endpoint, Request};
use crate::oracle::Tokenize;
use crate::trace::RequestClass;

/// The output budget of every equivalence request. Matches
/// [`crate::canary::CANARY_MAX_TOKENS`]: long enough to catch a real
/// divergence, short enough to run at every gate.
pub const EQUIVALENCE_MAX_TOKENS: u32 = crate::canary::CANARY_MAX_TOKENS;

/// One canary's spec-on vs spec-off comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanaryEquivalence {
    /// The canary's id (matches [`crate::canary::Canary::id`]).
    pub id: String,
    pub spec_on_text: String,
    pub spec_off_text: String,
    pub spec_on_tokens: Vec<u32>,
    pub spec_off_tokens: Vec<u32>,
    /// The first position (0-indexed) where the two token streams differ —
    /// a position past the shorter stream's end counts as a divergence
    /// there, not an excused gap ([`first_divergence`]'s doc comment).
    /// `None` when every position agreed and both streams were the same
    /// length.
    pub first_divergence: Option<usize>,
}

impl CanaryEquivalence {
    /// `true` when the two streams matched exactly, position for position,
    /// same length.
    pub fn equivalent(&self) -> bool {
        self.first_divergence.is_none()
    }
}

/// The first position where `a` and `b` differ, treating "one stream ran
/// out before the other" as a divergence at the shorter stream's length
/// rather than excusing it — an early stop is exactly the kind of
/// speculative-decoding bug this check exists to catch (a rejected draft
/// that truncated the commit instead of falling back correctly).
fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().max(b.len());
    (0..n).find(|&i| a.get(i) != b.get(i))
}

/// Run every canary against both endpoints (greedy — this crate's
/// `HttpEndpoint` fixes `temperature: 0` / `seed: 0` on every request) and
/// compare the tokenized outputs exactly. One request per canary per
/// endpoint; a request or tokenize failure on either side fails the whole
/// check with that canary named (a partial equivalence report is not a
/// report).
pub fn compare_endpoints(
    spec_on: &dyn Endpoint,
    spec_off: &dyn Endpoint,
    tokenizer: &dyn Tokenize,
    max_tokens: u32,
) -> Result<Vec<CanaryEquivalence>, String> {
    CANARIES
        .iter()
        .map(|c| {
            let req = |suffix: &str| Request {
                id: format!("equiv-{}-{suffix}", c.id),
                class: RequestClass::Sub,
                prompt: c.prompt.to_string(),
                max_tokens,
                stream: false,
                include_usage: false,
                enable_thinking: Some(false),
            };
            let on = spec_on
                .complete(&req("on"))
                .map_err(|e| format!("canary {} (spec-on): {e}", c.id))?;
            let off = spec_off
                .complete(&req("off"))
                .map_err(|e| format!("canary {} (spec-off): {e}", c.id))?;
            let spec_on_tokens = tokenizer
                .encode(&on.output)
                .map_err(|e| format!("canary {} (spec-on): tokenize: {e}", c.id))?;
            let spec_off_tokens = tokenizer
                .encode(&off.output)
                .map_err(|e| format!("canary {} (spec-off): tokenize: {e}", c.id))?;
            let divergence = first_divergence(&spec_on_tokens, &spec_off_tokens);
            Ok(CanaryEquivalence {
                id: c.id.to_string(),
                spec_on_text: on.output,
                spec_off_text: off.output,
                spec_on_tokens,
                spec_off_tokens,
                first_divergence: divergence,
            })
        })
        .collect()
}

/// Whether every canary in the suite matched exactly.
pub fn all_equivalent(results: &[CanaryEquivalence]) -> bool {
    results.iter().all(CanaryEquivalence::equivalent)
}

/// The first divergence across the suite, as `(canary id, position)`, in
/// [`CANARIES`] order — what a caller (the CLI) names and exits non-zero
/// on. `None` when every canary was equivalent.
pub fn first_failure(results: &[CanaryEquivalence]) -> Option<(&str, usize)> {
    results.iter().find_map(|r| r.first_divergence.map(|pos| (r.id.as_str(), pos)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::Outcome;

    struct MockTokenizer;

    impl Tokenize for MockTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
            Ok(text.split_whitespace().map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32))).collect())
        }
    }

    /// An endpoint that always returns the same fixed text, regardless of
    /// the request — a stand-in for one side of a spec-on/spec-off pair.
    struct FixedEndpoint(&'static str);

    impl Endpoint for FixedEndpoint {
        fn complete(&self, _req: &Request) -> Result<Outcome, String> {
            Ok(Outcome {
                ttft_ms: 5.0,
                total_ms: 5.0,
                n_tokens: 1,
                output: self.0.to_string(),
                reasoning_output: String::new(),
                reasoning_tokens: Some(0),
                prompt_tokens: Some(10),
                cached_prompt_tokens: None,
                token_times_ms: Vec::new(),
                finish_reason: Some(crate::client::FinishReason::Engine("stop".into())),
            })
        }
    }

    #[test]
    fn identical_outputs_are_fully_equivalent() {
        let a = FixedEndpoint("the quick brown fox");
        let b = FixedEndpoint("the quick brown fox");
        let results =
            compare_endpoints(&a, &b, &MockTokenizer, EQUIVALENCE_MAX_TOKENS).expect("results");
        assert_eq!(results.len(), CANARIES.len());
        assert!(all_equivalent(&results), "{results:?}");
        assert!(first_failure(&results).is_none());
    }

    #[test]
    fn a_divergent_word_is_reported_at_its_position() {
        let a = FixedEndpoint("the quick brown fox");
        let b = FixedEndpoint("the quick red fox");
        let results =
            compare_endpoints(&a, &b, &MockTokenizer, EQUIVALENCE_MAX_TOKENS).expect("results");
        assert!(!all_equivalent(&results));
        for r in &results {
            assert_eq!(r.first_divergence, Some(2), "{r:?}");
            assert!(!r.equivalent());
        }
        let (id, pos) = first_failure(&results).expect("a failure");
        assert_eq!(id, CANARIES[0].id);
        assert_eq!(pos, 2);
    }

    #[test]
    fn a_shorter_stream_diverges_at_its_own_end_not_excused() {
        let a = FixedEndpoint("the quick brown fox");
        let b = FixedEndpoint("the quick brown");
        let results =
            compare_endpoints(&a, &b, &MockTokenizer, EQUIVALENCE_MAX_TOKENS).expect("results");
        for r in &results {
            assert_eq!(r.first_divergence, Some(3), "{r:?}");
        }
    }

    #[test]
    fn a_failed_request_fails_the_whole_check_naming_the_canary() {
        struct Failing;
        impl Endpoint for Failing {
            fn complete(&self, _req: &Request) -> Result<Outcome, String> {
                Err("endpoint down".into())
            }
        }
        let a = Failing;
        let b = FixedEndpoint("anything");
        let err = compare_endpoints(&a, &b, &MockTokenizer, EQUIVALENCE_MAX_TOKENS)
            .expect_err("must fail");
        assert!(err.contains(CANARIES[0].id), "{err}");
        assert!(err.contains("spec-on"), "{err}");
    }

    #[test]
    fn every_canary_is_checked() {
        let a = FixedEndpoint("same output every time");
        let b = FixedEndpoint("same output every time");
        let results =
            compare_endpoints(&a, &b, &MockTokenizer, EQUIVALENCE_MAX_TOKENS).expect("results");
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        let expected: Vec<&str> = CANARIES.iter().map(|c| c.id).collect();
        assert_eq!(ids, expected);
    }
}
