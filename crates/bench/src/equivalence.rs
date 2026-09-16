//! The phase 5 correctness oracle (P5-07, GitHub #151): greedy
//! spec-on/spec-off equivalence.
//!
//! DFlash2 (spec 05) verifies every drafted token against the target model
//! before it commits, so a correct implementation's greedy output is
//! byte-for-byte identical whether speculation is on or off — never merely
//! close. This runs [`crate::canary::CANARIES`] against two endpoints (one
//! with speculation on, one with it off — or, since two engines do not fit
//! on the card, a [`Capture`] of each taken against its own launch, #156)
//! and compares the token sequences for **exact**
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
    let on = capture(spec_on, tokenizer, max_tokens, Side::SpecOn, LIVE_SESSION)
        .map_err(|e| format!("spec-on: {e}"))?;
    let off = capture(spec_off, tokenizer, max_tokens, Side::SpecOff, LIVE_SESSION)
        .map_err(|e| format!("spec-off: {e}"))?;
    compare_captures(&on, &off)
}

/// The session [`compare_endpoints`] stamps on both of its in-memory
/// captures: they are taken in one call, so they share it by construction.
const LIVE_SESSION: &str = "live";

/// Which side of the check a capture was taken on. The endpoint cannot tell
/// the harness whether speculation is on, so the operator names it and the
/// capture carries it: a spec-off capture compared with itself is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Side {
    SpecOn,
    SpecOff,
}

impl Side {
    pub fn as_str(self) -> &'static str {
        match self {
            Side::SpecOn => "spec-on",
            Side::SpecOff => "spec-off",
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "spec-on" => Ok(Side::SpecOn),
            "spec-off" => Ok(Side::SpecOff),
            other => Err(format!("side must be spec-on or spec-off, not {other:?}")),
        }
    }
}

/// One side of the check, taken against one launch and kept on disk (#156).
///
/// Two 27B engines do not fit on the one card (ADR 0006), so the gate cannot
/// hold a spec-on and a spec-off endpoint live at once: it captures the
/// spec-off launch, restarts the engine with speculation on, captures again,
/// and compares the two files with [`compare_captures`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capture {
    pub side: Side,
    /// The gate session the capture belongs to; two captures compare only
    /// within one session (ADR 0015).
    pub session: String,
    /// The output budget every request was sent with; two captures compare
    /// only at the same budget.
    pub max_tokens: u32,
    pub canaries: Vec<CanaryCapture>,
}

/// One canary's greedy output from one endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanaryCapture {
    pub id: String,
    pub text: String,
    pub tokens: Vec<u32>,
}

impl Capture {
    pub fn write(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("serialize the capture: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))
    }

    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    }
}

/// Run every canary against one endpoint (greedy — this crate's
/// `HttpEndpoint` fixes `temperature: 0` / `seed: 0` on every request) and
/// tokenize its output. A request or tokenize failure fails the whole
/// capture with that canary named (a partial capture is not a capture).
pub fn capture(
    endpoint: &dyn Endpoint,
    tokenizer: &dyn Tokenize,
    max_tokens: u32,
    side: Side,
    session: &str,
) -> Result<Capture, String> {
    let canaries = CANARIES
        .iter()
        .map(|c| {
            let req = Request {
                id: format!("equiv-{}-{}", c.id, side.as_str()),
                // `Sub` only because `Request` needs some class and nothing
                // here ever feeds `class_stats` — matches `oracle::record`'s
                // own canary requests.
                class: RequestClass::Sub,
                prompt: c.prompt.to_string(),
                max_tokens,
                stream: false,
                include_usage: false,
                enable_thinking: Some(false),
                images: Vec::new(),
            };
            let out = endpoint.complete(&req).map_err(|e| format!("canary {}: {e}", c.id))?;
            let tokens = tokenizer
                .encode(&out.output)
                .map_err(|e| format!("canary {}: tokenize: {e}", c.id))?;
            Ok(CanaryCapture { id: c.id.to_string(), text: out.output, tokens })
        })
        .collect::<Result<_, String>>()?;
    Ok(Capture { side, session: session.to_string(), max_tokens, canaries })
}

/// Compare a spec-on and a spec-off capture canary by canary, in
/// [`CANARIES`] order. Refuses a capture on the wrong side, captures from
/// different sessions or at different budgets (a shorter budget would read
/// as a divergence at its end), and a capture whose canaries are not
/// exactly the suite — one missing, repeated or unknown.
pub fn compare_captures(
    spec_on: &Capture,
    spec_off: &Capture,
) -> Result<Vec<CanaryEquivalence>, String> {
    for (capture, expected) in [(spec_on, Side::SpecOn), (spec_off, Side::SpecOff)] {
        if capture.side != expected {
            return Err(format!(
                "the {} capture was taken on the {} side",
                expected.as_str(),
                capture.side.as_str()
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for c in &capture.canaries {
            if !CANARIES.iter().any(|k| k.id == c.id) {
                return Err(format!(
                    "canary {} in the {} capture is not in the suite",
                    c.id,
                    expected.as_str()
                ));
            }
            if !seen.insert(c.id.as_str()) {
                return Err(format!(
                    "canary {} appears more than once in the {} capture",
                    c.id,
                    expected.as_str()
                ));
            }
        }
    }
    if spec_on.session != spec_off.session {
        return Err(format!(
            "the spec-on capture is from session {} and the spec-off capture from {}: \
             captures compare only within one session",
            spec_on.session, spec_off.session
        ));
    }
    if spec_on.max_tokens != spec_off.max_tokens {
        return Err(format!(
            "the spec-on capture ran at max_tokens {} and the spec-off capture at {}: \
             captures compare only at the same budget",
            spec_on.max_tokens, spec_off.max_tokens
        ));
    }
    let find = |capture: &Capture, side: &str, id: &str| {
        capture
            .canaries
            .iter()
            .find(|c| c.id == id)
            .cloned()
            .ok_or_else(|| format!("canary {id} is missing from the {side} capture"))
    };
    CANARIES
        .iter()
        .map(|c| {
            let on = find(spec_on, "spec-on", c.id)?;
            let off = find(spec_off, "spec-off", c.id)?;
            let divergence = first_divergence(&on.tokens, &off.tokens);
            Ok(CanaryEquivalence {
                id: c.id.to_string(),
                spec_on_text: on.text,
                spec_off_text: off.text,
                spec_on_tokens: on.tokens,
                spec_off_tokens: off.tokens,
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

    /// #156: two 27B engines do not fit on the one card, so the gate
    /// captures each side against its own launch and compares the files.
    #[test]
    fn captures_taken_one_launch_at_a_time_compare_like_two_live_endpoints() {
        let on = FixedEndpoint("the quick brown fox");
        let off = FixedEndpoint("the quick red fox");
        let live = compare_endpoints(&on, &off, &MockTokenizer, EQUIVALENCE_MAX_TOKENS)
            .expect("live results");

        let on_capture = capture(&on, &MockTokenizer, EQUIVALENCE_MAX_TOKENS, Side::SpecOn, "s1")
            .expect("capture on");
        let off_capture =
            capture(&off, &MockTokenizer, EQUIVALENCE_MAX_TOKENS, Side::SpecOff, "s1")
                .expect("capture off");
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let reload = |c: &Capture, name: &str| {
            let path = dir.join(format!("ignis-bench-equivalence-{name}-{pid}.json"));
            c.write(&path).expect("write the capture");
            let back = Capture::read(&path).expect("read the capture back");
            let _ = std::fs::remove_file(&path);
            back
        };
        let on_back = reload(&on_capture, "on");
        let off_back = reload(&off_capture, "off");
        assert_eq!(on_back, on_capture);
        assert_eq!(off_back, off_capture);

        let from_files = compare_captures(&on_back, &off_back).expect("file results");
        assert_eq!(from_files, live);
        let (id, pos) = first_failure(&from_files).expect("the divergence survives the files");
        assert_eq!((id, pos), (CANARIES[0].id, 2));
    }

    /// Every request names its side, so an engine log tells the two apart.
    #[test]
    fn a_capture_sends_requests_named_by_side() {
        use std::sync::Mutex;
        struct Recording(Mutex<Vec<String>>);
        impl Endpoint for Recording {
            fn complete(&self, req: &Request) -> Result<Outcome, String> {
                self.0.lock().unwrap().push(req.id.clone());
                FixedEndpoint("x").complete(req)
            }
        }
        let ep = Recording(Mutex::new(Vec::new()));
        capture(&ep, &MockTokenizer, 8, Side::SpecOff, "s1").expect("capture");
        let ids = ep.0.into_inner().unwrap();
        assert_eq!(ids[0], format!("equiv-{}-spec-off", CANARIES[0].id));
    }

    #[test]
    fn captures_at_different_budgets_are_refused() {
        let ep = FixedEndpoint("same output every time");
        let on = capture(&ep, &MockTokenizer, 64, Side::SpecOn, "s1").expect("capture on");
        let off = capture(&ep, &MockTokenizer, 32, Side::SpecOff, "s1").expect("capture off");
        let err = compare_captures(&on, &off).expect_err("budgets differ");
        assert!(err.contains("64") && err.contains("32"), "{err}");
    }

    /// A spec-off capture compared with itself must not pass as equivalence.
    #[test]
    fn captures_on_the_wrong_side_are_refused() {
        let ep = FixedEndpoint("same output every time");
        let off = capture(&ep, &MockTokenizer, 8, Side::SpecOff, "s1").expect("capture off");
        let err = compare_captures(&off, &off).expect_err("both sides are spec-off");
        assert!(err.contains("spec-on"), "{err}");
    }

    #[test]
    fn captures_from_different_sessions_are_refused() {
        let ep = FixedEndpoint("same output every time");
        let on = capture(&ep, &MockTokenizer, 8, Side::SpecOn, "g5-a").expect("capture on");
        let off = capture(&ep, &MockTokenizer, 8, Side::SpecOff, "g5-b").expect("capture off");
        let err = compare_captures(&on, &off).expect_err("sessions differ");
        assert!(err.contains("g5-a") && err.contains("g5-b"), "{err}");
    }

    #[test]
    fn a_capture_missing_a_canary_is_refused_naming_it() {
        let ep = FixedEndpoint("same output every time");
        let on = capture(&ep, &MockTokenizer, 8, Side::SpecOn, "s1").expect("capture on");
        let mut off = capture(&ep, &MockTokenizer, 8, Side::SpecOff, "s1").expect("capture off");
        let dropped = off.canaries.pop().expect("a canary").id;
        let err = compare_captures(&on, &off).expect_err("a canary is missing");
        assert!(err.contains(&dropped), "{err}");
        assert!(err.contains("spec-off"), "{err}");
    }

    #[test]
    fn a_capture_with_an_extra_or_repeated_canary_is_refused() {
        let ep = FixedEndpoint("same output every time");
        let on = capture(&ep, &MockTokenizer, 8, Side::SpecOn, "s1").expect("capture on");
        let off = capture(&ep, &MockTokenizer, 8, Side::SpecOff, "s1").expect("capture off");

        let mut repeated = off.clone();
        repeated.canaries.push(repeated.canaries[0].clone());
        let err = compare_captures(&on, &repeated).expect_err("a canary repeats");
        assert!(err.contains(CANARIES[0].id), "{err}");

        let mut extra = off.clone();
        extra.canaries.push(CanaryCapture { id: "stale".into(), text: String::new(), tokens: vec![] });
        let err = compare_captures(&on, &extra).expect_err("an unknown canary");
        assert!(err.contains("stale"), "{err}");
    }

    #[test]
    fn a_failed_capture_request_names_the_canary() {
        struct Failing;
        impl Endpoint for Failing {
            fn complete(&self, _req: &Request) -> Result<Outcome, String> {
                Err("endpoint down".into())
            }
        }
        let err = capture(&Failing, &MockTokenizer, EQUIVALENCE_MAX_TOKENS, Side::SpecOn, "s1")
            .expect_err("must fail");
        assert!(err.contains(CANARIES[0].id), "{err}");
    }

    #[test]
    fn a_side_parses_from_its_flag_spelling() {
        assert_eq!(Side::parse("spec-on"), Ok(Side::SpecOn));
        assert_eq!(Side::parse("spec-off"), Ok(Side::SpecOff));
        assert!(Side::parse("on").is_err());
        assert_eq!(Side::SpecOff.as_str(), "spec-off");
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
