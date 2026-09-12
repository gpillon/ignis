//! Canary suite + self-consistency check (ADR 0007).
//!
//! Correctness is *self-checked*, not reference-matched: the engine must
//! produce *sane* output for the same model (greedy, fixed seed). We do **not**
//! require it to match the reference's tokens. The canary suite is a fixed,
//! high-signal set of prompts. For each canary the harness (1) checks the
//! output is *sane*, and (2) re-runs the prompt and checks the output is
//! *deterministic* (greedy + fixed seed ⇒ identical output).
//!
//! A canary is judged on everything the turn generated, thinking channel
//! included (GitHub #137): the question is whether the *engine* is healthy,
//! and a turn whose token budget ran out mid-thought has generated sane
//! tokens for every one of them.
//!
//! This module is pure (no I/O) so it is unit-testable without a running
//! engine; the endpoint plumbing lives in `client.rs`.

use serde::{Deserialize, Serialize};

use crate::client::{Endpoint, Outcome, Request};
use crate::trace::RequestClass;

/// A single canary: a fixed, high-signal prompt.
#[derive(Debug, Clone)]
pub struct Canary {
    pub id: &'static str,
    pub prompt: &'static str,
}

/// A small fixed set of high-signal prompts. They are short and deterministic
/// so a *sane* greedy output is easy to check.
pub const CANARIES: &[Canary] = &[
    Canary {
        id: "rust-hello",
        prompt: "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    },
    Canary {
        id: "rust-sort",
        prompt: "What does `let v = vec![3,1,2]; v.sort();` set `v` to, after the call?",
    },
    Canary {
        id: "math-greedy",
        prompt: "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    },
    Canary {
        id: "explain-reverse",
        prompt: "Explain in one sentence what `x.reverse()` does to a `Vec<i32>` named `x`.",
    },
];

/// The result of running one canary (sent twice for the determinism check).
///
/// Serializable: the canary results *are* the divergence report, which is
/// shipped as JSON alongside the performance report (spec 02, ADR 0007 —
/// `ignis-bench canary --out`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryResult {
    pub id: String,
    /// `true` when the first output passed the sanity check.
    pub sane: bool,
    /// Why the output is not sane (the `is_sane` error), when `sane` is
    /// `false`; kept so the shipped divergence report is actionable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sane_reason: Option<String>,
    /// `true` when the two greedy runs produced identical output.
    pub deterministic: bool,
    /// The two greedy outputs (kept for the divergence report) — the
    /// **content** channel of each run.
    pub first: String,
    pub second: String,
    /// The thinking channel of each run, when the turn thought (GitHub
    /// #137). Kept apart from `first` / `second` so the shipped divergence
    /// report still shows which channel the text came back on; absent from
    /// the JSON when there was none, so a thinking-disabled record keeps the
    /// shape it always had.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub first_reasoning: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub second_reasoning: String,
}

impl CanaryResult {
    /// Overall self-consistency verdict: sane **and** deterministic.
    pub fn consistent(&self) -> bool {
        self.sane && self.deterministic
    }
}

/// A self-consistency check for a single generated output: is it *sane*?
///
/// "Sane" here means: non-empty, no NUL bytes, and not a runaway single
/// character (a stuck generation). This is the "produces *sane* output"
/// self-check of ADR 0007 — it does **not** compare against the reference.
pub fn is_sane(output: &str) -> Result<(), String> {
    if output.trim().is_empty() {
        return Err("empty output".into());
    }
    if output.contains('\0') {
        return Err("output contains NUL bytes".into());
    }
    // Reject runaway repetition: a single byte repeated more than 30 times in
    // a row is almost certainly a degenerate / stuck generation, not a sane
    // answer.
    let bytes = output.as_bytes();
    let mut run = 1usize;
    for i in 1..bytes.len() {
        if bytes[i] == bytes[i - 1] {
            run += 1;
            if run > 30 {
                return Err(format!(
                    "degenerate repetition ({} x '{}')",
                    run,
                    bytes[i] as char
                ));
            }
        } else {
            run = 1;
        }
    }
    Ok(())
}

/// Determinism check: two greedy runs of the same prompt (greedy + fixed
/// seed) must produce identical output.
pub fn is_deterministic(first: &str, second: &str) -> bool {
    first == second
}

/// What one greedy run generated, kept per channel the way the engine sent
/// it (GitHub #137).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Generation<'a> {
    /// The answer channel (`delta.content`).
    pub content: &'a str,
    /// The thinking channel (`delta.reasoning_content`), empty when the turn
    /// did not think or thinking was disabled.
    pub reasoning: &'a str,
}

impl<'a> Generation<'a> {
    /// A run that produced an answer and nothing else.
    pub fn answer_only(content: &'a str) -> Self {
        Self {
            content,
            reasoning: "",
        }
    }

    /// Everything the run generated, in generation order: it thought, then
    /// it answered.
    ///
    /// This is what sanity is judged on. ADR 0007's self-check asks whether
    /// the *engine* produced sane output, and a turn that spends its whole
    /// 64-token budget thinking has produced 64 perfectly sane tokens — it
    /// simply has not reached the answer yet. Judging on `delta.content`
    /// alone called three of the four canaries "empty output" against a
    /// server whose streams were well formed, on nothing but the default
    /// `enable_thinking`.
    fn text(&self) -> String {
        format!("{}{}", self.reasoning, self.content)
    }
}

/// Evaluate one canary from its two greedy outputs (the harness sends each
/// prompt twice and passes both outputs here). Thinking-free shorthand for
/// [`evaluate_channels`].
pub fn evaluate(id: &str, first: &str, second: &str) -> CanaryResult {
    evaluate_channels(
        id,
        Generation::answer_only(first),
        Generation::answer_only(second),
    )
}

/// Evaluate one canary from its two greedy runs.
pub fn evaluate_channels(id: &str, first: Generation, second: Generation) -> CanaryResult {
    let sane_reason = match is_sane(&first.text()) {
        Ok(()) => None,
        Err(reason) => Some(reason),
    };
    CanaryResult {
        id: id.to_string(),
        sane: sane_reason.is_none(),
        sane_reason,
        // Each channel is compared to its own: greedy plus a fixed seed
        // reproduces the whole generation, so text that merely *moved*
        // between the channels is a divergence, not a match.
        deterministic: is_deterministic(first.content, second.content)
            && is_deterministic(first.reasoning, second.reasoning),
        first: first.content.to_string(),
        second: second.content.to_string(),
        first_reasoning: first.reasoning.to_string(),
        second_reasoning: second.reasoning.to_string(),
    }
}

/// Run the whole canary suite against an endpoint: each canary prompt is sent
/// **twice** (greedy determinism) and both outputs are checked. Returns one
/// `CanaryResult` per canary.
pub fn run_canaries(ep: &dyn Endpoint) -> Vec<CanaryResult> {
    CANARIES
        .iter()
        .map(|c| {
            let req = Request {
                id: format!("canary-{}", c.id),
                class: RequestClass::Sub,
                prompt: c.prompt.to_string(),
                max_tokens: 64,
                stream: false,
                include_usage: false,
                enable_thinking: None,
            };
            let run = |ep: &dyn Endpoint| {
                ep.complete(&req)
                    .map(|o: Outcome| (o.output, o.reasoning_output))
                    .unwrap_or_default()
            };
            let (first, first_reasoning) = run(ep);
            let (second, second_reasoning) = run(ep);
            evaluate_channels(
                c.id,
                Generation {
                    content: &first,
                    reasoning: &first_reasoning,
                },
                Generation {
                    content: &second,
                    reasoning: &second_reasoning,
                },
            )
        })
        .collect()
}

/// The overall self-consistency verdict for a suite result set: `true` when
/// every canary is sane **and** deterministic.
pub fn suite_consistent(results: &[CanaryResult]) -> bool {
    results.iter().all(|r| r.consistent())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_normal_answer_is_sane() {
        assert!(is_sane("It prints `hi` to standard output.").is_ok());
    }

    #[test]
    fn empty_output_is_not_sane() {
        assert!(is_sane("").is_err());
        assert!(is_sane("   \n  ").is_err());
    }

    #[test]
    fn nul_bytes_are_not_sane() {
        assert!(is_sane("ok\0").is_err());
    }

    #[test]
    fn runaway_repetition_is_not_sane() {
        let stuck = "a".repeat(60);
        assert!(is_sane(&stuck).is_err());
        // A short run of repeats is fine.
        assert!(is_sane(&"a".repeat(10)).is_ok());
    }

    #[test]
    fn determinism_is_an_exact_match() {
        assert!(is_deterministic("x", "x"));
        assert!(!is_deterministic("x", "y"));
    }

    #[test]
    fn evaluate_combines_sane_and_determinism() {
        let r = evaluate("c1", "fine answer", "fine answer");
        assert!(r.consistent());

        let r2 = evaluate("c2", "", "");
        assert!(!r2.sane);
        assert!(r2.deterministic); // both empty -> "equal"

        let r3 = evaluate("c3", "a", "b");
        assert!(!r3.deterministic);
    }

    #[test]
    fn the_canary_suite_has_entries() {
        assert!(!CANARIES.is_empty());
        // Every canary has a non-empty prompt and a unique id.
        let ids: std::collections::BTreeSet<_> = CANARIES.iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), CANARIES.len());
        for c in CANARIES {
            assert!(!c.prompt.is_empty());
        }
    }

    #[test]
    fn a_canary_result_round_trips_through_json() {
        // The divergence report is shipped as JSON (`canary --out`), so a
        // `CanaryResult` must survive a JSON round-trip (spec 02: the
        // divergence report is *shipped*, ADR 0007).
        let r = evaluate("math-greedy", "the answer is 10", "the answer is 10");
        let json = serde_json::to_string(&r).expect("serialize the canary result");
        let back: CanaryResult = serde_json::from_str(&json).expect("parse the canary result");
        assert_eq!(back.id, "math-greedy");
        assert!(back.sane);
        assert!(back.deterministic);
        assert_eq!(back.first, "the answer is 10");
        assert_eq!(back.second, "the answer is 10");
    }

    #[test]
    fn an_unsane_canary_keeps_the_sanity_reason() {
        // The shipped divergence report must say *why* a canary is unsane —
        // a bare "sane: false" is not actionable.
        let stuck = "x".repeat(40);
        let divergent = evaluate("stuck", &stuck, &stuck);
        assert!(!divergent.sane);
        assert!(
            divergent.sane_reason.is_some(),
            "an unsane canary must carry the sanity reason"
        );
        let healthy = evaluate("ok", "a sane answer", "a sane answer");
        assert!(healthy.sane);
        assert!(
            healthy.sane_reason.is_none(),
            "a sane canary has no sanity reason"
        );
    }

    #[test]
    fn a_turn_that_spent_its_budget_thinking_is_sane() {
        // GitHub #137 / #144: against a thinking-enabled server the 64-token
        // budget can run out inside the reasoning channel, leaving the
        // content channel empty. The engine generated 64 sane tokens; it did
        // not produce "empty output".
        let thought = "the vector is sorted ascending, so ";
        let run = Generation { content: "", reasoning: thought };
        let r = evaluate_channels("rust-sort", run, run);
        assert!(r.sane, "thinking is generated output: {:?}", r.sane_reason);
        assert!(r.deterministic);
        assert!(r.consistent());
        assert_eq!(r.first, "", "the content channel is reported as it was");
        assert_eq!(r.first_reasoning, thought, "and the thinking channel beside it");
    }

    #[test]
    fn a_turn_that_generated_nothing_at_all_is_still_unsane() {
        let r = evaluate_channels("dead", Generation::answer_only(""), Generation::answer_only(""));
        assert!(!r.sane);
        assert_eq!(r.sane_reason.as_deref(), Some("empty output"));
    }

    #[test]
    fn determinism_covers_the_thinking_channel_too() {
        // Greedy + fixed seed reproduces the whole generation, scratch work
        // included: two runs that reason differently are not deterministic
        // even when they land on the same answer.
        let r = evaluate_channels(
            "math-greedy",
            Generation { content: "10", reasoning: "2*3=6, +4 " },
            Generation { content: "10", reasoning: "6+4 " },
        );
        assert!(r.sane);
        assert!(!r.deterministic, "the reasoning diverged");
    }

    #[test]
    fn text_that_moved_between_the_channels_is_a_divergence() {
        // Two runs whose concatenation is identical but whose channel split
        // is not: one answered "b" after thinking "a", the other thought
        // "ab" and never answered. Judging the concatenation alone would
        // call that deterministic.
        let r = evaluate_channels(
            "explain-reverse",
            Generation { content: "b", reasoning: "a" },
            Generation { content: "", reasoning: "ab" },
        );
        assert!(!r.deterministic, "the answer channel is not reproducible");
    }

    #[test]
    fn a_thinking_free_canary_keeps_the_json_shape_it_always_had() {
        let r = evaluate("math-greedy", "the answer is 10", "the answer is 10");
        let json = serde_json::to_value(&r).expect("serialize the canary result");
        assert!(
            json.get("first_reasoning").is_none(),
            "an empty thinking channel is omitted, not written as \"\": {json}"
        );
        let run = Generation { content: "10", reasoning: "2*3=6 " };
        let thinking = evaluate_channels("math-greedy", run, run);
        let json = serde_json::to_value(&thinking).expect("serialize the canary result");
        assert_eq!(json["first_reasoning"], "2*3=6 ");
        let back: CanaryResult = serde_json::from_value(json).expect("parse it back");
        assert_eq!(back.second_reasoning, "2*3=6 ");
    }
}
