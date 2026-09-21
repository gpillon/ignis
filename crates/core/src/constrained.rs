//! A **constrained decode**: a decision whose answer is *generated*, one constrained
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
//! So a K-step run costs one constrained prefill and K rounds: round `i`
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

/// The tokens one step of a constrained decode may draw from.
///
/// An `Arc` because a constrained decode is cloned onto every job of the request that
/// carries it, and a step's set is the same handful of digit ids each time.
pub type PermittedSet = Arc<[TokenId]>;

/// The schedule of permitted sets a **constrained request** generates under:
/// one set per token it will emit, in order.
///
/// Its length is the request's whole generation **budget**, and for a
/// schedule with no [`Schedule::terminator`] it is also the whole stopping
/// condition: the run ends when the schedule is spent and **never on EOS**,
/// because a token drawn from a set of digits cannot be the EOS token.
///
/// A schedule **with** a terminator ends on whichever comes first. That is
/// not EOS by another name and the distinction is the point: EOS is the
/// model deciding it has nothing more to say, while a terminator is a token
/// the *caller* put in the alphabet because it closes the shape the caller
/// asked for. `scalar` (GitHub #255) puts `}` there, so a number that is
/// complete closes its own object instead of filling a field, and the
/// length becomes a maximum.
///
/// **A short run is then not automatically a fault**, which is the one
/// thing a reader must get right. Three outcomes, distinguished by the last
/// token and not by the length: ended on the terminator (complete), spent
/// its schedule without one (at the cap, complete), or stopped without one
/// and short (the engine cut it off, an error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    steps: Vec<PermittedSet>,
    terminator: Option<TokenId>,
}

impl Schedule {
    /// Build a constrained decode from its steps, or refuse it.
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
            return Err("a constrained decode must carry at least one step".to_owned());
        }
        for (index, step) in steps.iter().enumerate() {
            if step.is_empty() {
                return Err(format!("schedule step {index} permits no token at all"));
            }
            if step.len() > MAX_PERMITTED_TOKENS {
                return Err(format!(
                    "schedule step {index} permits {} tokens; {MAX_PERMITTED_TOKENS} is the leaf's cap, \
                     and a set past it is refused, never truncated",
                    step.len()
                ));
            }
        }
        Ok(Self {
            steps: steps.into_iter().map(PermittedSet::from).collect(),
            terminator: None,
        })
    }

    /// The same schedule, ending early when `terminator` is drawn.
    ///
    /// The token must be permitted by some step, or it could never be drawn
    /// and the schedule would silently be the one without it — a terminator
    /// that cannot fire is a caller who believes their run can stop early
    /// and is wrong about it.
    pub fn ending_on(mut self, terminator: TokenId) -> Result<Self, String> {
        if !self.steps.iter().any(|step| step.contains(&terminator)) {
            return Err(format!(
                "token {terminator} would end this run, but no step permits it, so it can never be drawn"
            ));
        }
        self.terminator = Some(terminator);
        Ok(self)
    }

    /// The token that ends this run early, if it has one.
    pub fn terminator(&self) -> Option<TokenId> {
        self.terminator
    }

    /// Whether `token` ends this run.
    pub fn ends_on(&self, token: TokenId) -> bool {
        self.terminator == Some(token)
    }

    /// How many tokens this constrained decode may emit — its generation
    /// budget, and the exact count only when it has no terminator.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Never true: [`Schedule::new`] refuses an empty schedule. Present
    /// because `len` without it is a clippy lint and a reader's question.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The set step `index` draws from, or `None` past the end — which is
    /// what the last round is handed, and is the stopping condition for a
    /// schedule that has no [`Schedule::terminator`].
    pub fn step(&self, index: usize) -> Option<PermittedSet> {
        self.steps.get(index).cloned()
    }

    /// The steps, in order.
    pub fn steps(&self) -> &[PermittedSet] {
        &self.steps
    }
}

/// One token a constrained decode drew, and how much of its own permitted set stood
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
    fn a_schedule_refuses_what_the_leaf_would_refuse() {
        assert!(Schedule::new(Vec::new()).is_err(), "an empty schedule");
        assert!(
            Schedule::new(vec![vec![1, 2], Vec::new()]).is_err(),
            "a step that permits nothing"
        );
        let too_wide: Vec<TokenId> = (0..=MAX_PERMITTED_TOKENS as TokenId).collect();
        let refusal = Schedule::new(vec![too_wide]).expect_err("a step past the cap");
        assert!(
            refusal.contains("refused, never truncated"),
            "and says so rather than quietly cutting it: {refusal}"
        );
    }

    /// The cap is written down three times — here, in `step.rs`'s FFI
    /// block, and in `ignis_step.h` — and a restatement nobody checks is a
    /// restatement that drifts. This ties the two Rust copies together; the
    /// C copy is tied to `step.rs`'s by the size-prefixed struct the leaf
    /// validates (ADR 0016).
    ///
    /// Drift here does not fail loudly: the wide side would build a set the
    /// narrow side refuses, and the refusal would name a cap the caller
    /// cannot see.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_cap_is_the_leafs_cap() {
        assert_eq!(MAX_PERMITTED_TOKENS, crate::step::ffi::MAX_PERMITTED_TOKENS);
    }

    #[test]
    fn a_schedule_hands_out_its_steps_and_then_stops() {
        let schedule = Schedule::new(vec![vec![1, 2], vec![3]]).expect("two steps");
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule.step(0).as_deref(), Some(&[1, 2][..]));
        assert_eq!(schedule.step(1).as_deref(), Some(&[3][..]));
        assert_eq!(schedule.step(2), None, "the last round draws nothing wanted");
    }
}
