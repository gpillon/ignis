//! A **program**: a decision whose answer is *generated*, one constrained
//! token at a time (GitHub #242, ADR 0034, spec 06).
//!
//! The readout primitives ([`crate::decision`]) read one position and name
//! their answer with one vocabulary entry, so they can only ever produce one
//! symbol. A **number** is the primitive they cannot express: K positions,
//! each restricted to the digit alphabet, read in order. A point is two of
//! them and a box is four.
//!
//! What crosses the `Compute` seam for this is a **permitted token set per
//! step** — never logits. The leaf masks everything outside the set and
//! samples normally, so the constraint composes with temperature, top-k,
//! seeds and penalties, and the answer comes back as a token id exactly as
//! an unconstrained one does. ADR 0034 rejects the alternative (a
//! full-vocabulary row back per step: 248,320 f32 on the 27B, six
//! synchronisations for a three-digit number) and it must not creep back.
//!
//! # The one-round lag, which is a property of the leaf and not of this type
//!
//! A decode round **returns the token the previous call drew** and draws the
//! next one. The first token of a run is therefore drawn by the *prefill*,
//! which is why [`crate::scheduler::PrefillJob::permitted`] exists at all:
//! a run that constrained only its rounds would commit the prompt's own free
//! successor as the first token of its forced text. That cost a failed GPU
//! test to learn (`crates/core/tests/permitted_decode_gpu.rs`), and the seam
//! keeps it visible rather than hiding it — a backend that hid it would let
//! a scheduler with exactly that bug pass on CPU.
//!
//! So a K-step program costs one constrained prefill and K rounds: round `i`
//! returns step `i-1`'s token and draws step `i`'s. The final round draws
//! nothing anybody wants, and its token is discarded with the sequence.

use std::sync::Arc;

use crate::types::TokenId;

/// The most tokens one step may permit.
///
/// Restated here rather than imported from [`crate::step`], which is
/// `#![cfg(feature = "cuda")]` — the same reason
/// [`crate::scheduler::NO_HOST_ROOM`] restates a leaf ABI constant. The
/// scheduler, the mock and the wire types are all CPU-only and all need to
/// refuse an over-wide set *before* a GPU is involved; the leaf checks it
/// again on its own side, because a cap enforced in one place only is a cap
/// that a second caller walks past.
pub const MAX_PERMITTED_TOKENS: usize = 32;

/// The tokens one step of a program may draw from.
///
/// An `Arc` because a program is cloned onto every job of the request that
/// carries it, and a step's set is the same handful of digit ids each time.
pub type PermittedSet = Arc<[TokenId]>;

/// The schedule of permitted sets a **program request** generates under:
/// one set per token it will emit, in order.
///
/// Its length is the request's whole generation budget. There is no other
/// stopping condition: a program ends when its schedule is exhausted and
/// **never on EOS**, because a token drawn from a set of digits cannot be
/// the EOS token, and a program that could stop early would return a number
/// with fewer digits than the caller asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    steps: Vec<PermittedSet>,
}

impl Program {
    /// Build a program from its steps, or refuse it.
    ///
    /// A step is refused rather than truncated or widened, for the reason
    /// the leaf refuses one: a constraint silently dropped is a wrong answer
    /// that looks like a right one — a digit read off an unconstrained
    /// position is still a digit.
    ///
    /// A **step of one** is legal and load-bearing: that is how a literal is
    /// forced mid-generation (spec 06's `,"y":`), and its probability is 1
    /// by construction, so it contributes nothing to the answer's
    /// uncertainty.
    pub fn new(steps: Vec<Vec<TokenId>>) -> Result<Self, String> {
        if steps.is_empty() {
            return Err("a program must carry at least one step".to_owned());
        }
        for (index, step) in steps.iter().enumerate() {
            if step.is_empty() {
                return Err(format!("program step {index} permits no token at all"));
            }
            if step.len() > MAX_PERMITTED_TOKENS {
                return Err(format!(
                    "program step {index} permits {} tokens; {MAX_PERMITTED_TOKENS} is the leaf's cap, \
                     and a set past it is refused, never truncated",
                    step.len()
                ));
            }
        }
        Ok(Self {
            steps: steps.into_iter().map(PermittedSet::from).collect(),
        })
    }

    /// How many tokens this program emits — its generation budget.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Never true: [`Program::new`] refuses an empty schedule. Present
    /// because `len` without it is a clippy lint and a reader's question.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The set step `index` draws from, or `None` past the end — which is
    /// what the last round is handed, and is the whole stopping condition.
    pub fn step(&self, index: usize) -> Option<PermittedSet> {
        self.steps.get(index).cloned()
    }

    /// The steps, in order.
    pub fn steps(&self) -> &[PermittedSet] {
        &self.steps
    }
}

/// One token a program drew, and how much of its own permitted set stood
/// behind it.
///
/// The probability is the **restricted softmax** over that step's set alone,
/// temperature-free on purpose: it is the model's own confidence in the
/// digit, not the chance the sampler happened to draw it. That distinction
/// is what makes the trace readable — a units digit at 0.15 says the reading
/// is good to the tens place, whatever temperature the caller asked for
/// (`docs/findings/2026-09-19-constrained-digit-readout-points.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw {
    /// The token emitted, always a member of its step's permitted set.
    pub token: TokenId,
    /// Its probability within that set, in `(0, 1]`. Exactly 1 for a step
    /// of one.
    pub probability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_refuses_what_the_leaf_would_refuse() {
        assert!(Program::new(Vec::new()).is_err(), "an empty schedule");
        assert!(
            Program::new(vec![vec![1, 2], Vec::new()]).is_err(),
            "a step that permits nothing"
        );
        let too_wide: Vec<TokenId> = (0..=MAX_PERMITTED_TOKENS as TokenId).collect();
        let refusal = Program::new(vec![too_wide]).expect_err("a step past the cap");
        assert!(
            refusal.contains("refused, never truncated"),
            "and says so rather than quietly cutting it: {refusal}"
        );
    }

    #[test]
    fn a_program_hands_out_its_steps_and_then_stops() {
        let program = Program::new(vec![vec![1, 2], vec![3]]).expect("two steps");
        assert_eq!(program.len(), 2);
        assert_eq!(program.step(0).as_deref(), Some(&[1, 2][..]));
        assert_eq!(program.step(1).as_deref(), Some(&[3][..]));
        assert_eq!(program.step(2), None, "the last round draws nothing wanted");
    }
}
