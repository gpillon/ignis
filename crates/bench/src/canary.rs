//! Canary suite + self-consistency check (ADR 0007).
//!
//! Correctness is *self-checked*, not reference-matched: the engine must
//! produce *sane* output for the same model (greedy, fixed seed). We do **not**
//! require it to match the reference's tokens. The canary suite is a fixed,
//! high-signal set of prompts. For each canary the harness (1) checks the
//! output is *sane*, and (2) re-runs the prompt and checks the output is
//! *deterministic* (greedy + fixed seed ⇒ identical output).
//!
//! This module is pure (no I/O) so it is unit-testable without a running
//! engine; the endpoint plumbing lives in `client.rs`.

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
#[derive(Debug, Clone)]
pub struct CanaryResult {
    pub id: &'static str,
    /// `true` when the first output passed the sanity check.
    pub sane: bool,
    /// `true` when the two greedy runs produced identical output.
    pub deterministic: bool,
    /// The two greedy outputs (kept for the divergence report).
    pub first: String,
    pub second: String,
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

/// Evaluate one canary from its two greedy outputs (the harness sends each
/// prompt twice and passes both outputs here).
pub fn evaluate(id: &'static str, first: &str, second: &str) -> CanaryResult {
    let sane = is_sane(first).is_ok();
    CanaryResult {
        id,
        sane,
        deterministic: is_deterministic(first, second),
        first: first.to_string(),
        second: second.to_string(),
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
            };
            let first = ep.complete(&req).map(|o: Outcome| o.output).unwrap_or_default();
            let second = ep.complete(&req).map(|o: Outcome| o.output).unwrap_or_default();
            evaluate(c.id, &first, &second)
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
}