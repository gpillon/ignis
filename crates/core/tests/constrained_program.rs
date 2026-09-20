//! The `Compute` seam under a **program** (GitHub #242, spec 06): the
//! permitted set per step, and the one-round lag that carries it.
//!
//! What is pinned here is the contract a backend has to honour, stated on
//! [`DecodeJob::permitted`] and modelled by [`MockCompute`]:
//!
//! - a prefill **draws**, and a program's first token is the one it draws;
//! - a decode round **returns the token the previous call drew** and draws
//!   the next one, so round `i` carries step `i`'s set and emits step
//!   `i-1`'s token;
//! - the probability reported beside a token is that **token's** own, not
//!   the one the same round drew — the backend holds the lag, because a
//!   caller holding it would be off by one place value on every number.
//!
//! The steps below are deliberately **disjoint**, which is what makes the
//! lag testable at all: with one shared alphabet every off-by-one still
//! produces a digit, and nothing fails.

use std::sync::Arc;

use ignis_core::mock::MockCompute;
use ignis_core::program::{Draw, MAX_PERMITTED_TOKENS, Program};
use ignis_core::scheduler::{Compute, DecodeJob, PrefillJob};
use ignis_core::types::{DecodeParams, RequestId, TokenId};

const REQUEST: RequestId = 77;

/// Three disjoint steps, then one of exactly one token — the shape a forced
/// literal takes (spec 06's `,"y":` is a step per token, each permitting
/// one).
fn program() -> Program {
    Program::new(vec![
        vec![10, 11, 12],
        vec![20, 21, 22],
        vec![30, 31, 32],
        vec![40],
    ])
    .expect("four legal steps")
}

fn prefill_job(program: &Program) -> PrefillJob {
    PrefillJob {
        request: REQUEST,
        tokens: vec![1, 2, 3],
        context_tokens: 64,
        start_position: 0,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        permitted: program.step(0),
    }
}

fn decode_job(permitted: Option<ignis_core::program::PermittedSet>) -> DecodeJob {
    DecodeJob {
        request: REQUEST,
        lane: 0,
        params: DecodeParams::default(),
        remaining_tokens: 8,
        permitted,
    }
}

/// Run `program` through the mock the way a scheduler has to: one
/// constrained prefill, then one round per step, each carrying the *next*
/// step's set.
fn run(compute: &MockCompute, program: &Program) -> Vec<Draw> {
    compute
        .prefill_step(&[prefill_job(program)])
        .expect("the constrained prefill");
    let mut drawn = Vec::new();
    for round in 0..program.len() {
        let outcome = compute
            .decode_step(&[decode_job(program.step(round + 1))])
            .expect("a constrained round")
            .remove(0);
        assert_eq!(
            outcome.tokens.len(),
            1,
            "a constrained round commits exactly one token"
        );
        assert_eq!(
            outcome.probabilities.len(),
            outcome.tokens.len(),
            "and one probability per token"
        );
        drawn.push(Draw {
            token: outcome.tokens[0],
            probability: outcome.probabilities[0],
        });
    }
    drawn
}

#[test]
fn every_token_comes_from_its_own_step() {
    let program = program();
    let drawn = run(&MockCompute::new(), &program);
    assert_eq!(drawn.len(), program.len(), "one token per step, in order");
    for (index, (draw, step)) in drawn.iter().zip(program.steps()).enumerate() {
        assert!(
            step.contains(&draw.token),
            "step {index} permits {step:?} and the run emitted {} — the sets are \
             disjoint, so this is the lag being off, not a near miss",
            draw.token
        );
    }
}

#[test]
fn the_lag_is_the_backends_and_a_round_that_kept_it_would_fail_here() {
    // The failure this guards against, spelled out: emit the token the round
    // *drew* rather than the one it returns, and every token lands one step
    // late. The first round would then emit a member of step 1, not step 0.
    let program = program();
    let drawn = run(&MockCompute::new(), &program);
    assert!(
        program.steps()[0].contains(&drawn[0].token),
        "the first token emitted is the one the PREFILL drew, from step 0"
    );
    assert!(
        !program.steps()[1].contains(&drawn[0].token),
        "and not the one the first round drew, which belongs to step 1"
    );
}

#[test]
fn a_probability_is_the_tokens_own_and_a_forced_literal_is_certain() {
    let program = program();
    let drawn = run(&MockCompute::new(), &program);
    for (index, draw) in drawn.iter().enumerate().take(3) {
        assert!(
            draw.probability > 0.0 && draw.probability < 1.0,
            "step {index} reports a real confidence, neither absent nor perfect: {}",
            draw.probability
        );
    }
    assert_eq!(
        drawn[3].probability, 1.0,
        "a step permitting one token is drawn with probability 1 — a softmax over \
         one logit — so a forced literal adds nothing to an answer's uncertainty"
    );
}

#[test]
fn an_unconstrained_lane_is_untouched_and_reports_no_probability() {
    // The mock's ordinary path, unchanged: no program, no pending draw, and
    // the round behaves exactly as every other test in this crate expects.
    let compute = MockCompute::new();
    compute
        .prefill_step(&[PrefillJob {
            permitted: None,
            ..prefill_job(&program())
        }])
        .expect("an ordinary prefill");
    let outcome = compute
        .decode_step(&[decode_job(None)])
        .expect("an ordinary round")
        .remove(0);
    assert_eq!(outcome.tokens.len(), 1, "one token, as before");
    assert!(
        outcome.probabilities.is_empty(),
        "and no probability: this lane drew from the whole vocabulary, and a \
         number computed over a set it does not have would be a lie"
    );
}

#[test]
fn a_lane_beside_a_program_is_not_dragged_into_it() {
    let program = program();
    let compute = MockCompute::new();
    let free = 91;
    compute
        .prefill_step(&[
            prefill_job(&program),
            PrefillJob {
                request: free,
                permitted: None,
                ..prefill_job(&program)
            },
        ])
        .expect("one constrained prefill beside a free one");
    let outcomes = compute
        .decode_step(&[
            decode_job(program.step(1)),
            DecodeJob {
                request: free,
                ..decode_job(None)
            },
        ])
        .expect("a mixed round");
    assert!(
        program.steps()[0].contains(&outcomes[0].tokens[0]),
        "the program lane kept its schedule"
    );
    assert!(
        outcomes[1].probabilities.is_empty(),
        "and the lane beside it drew freely"
    );
}

#[test]
fn a_program_refuses_a_step_the_leaf_could_not_honour() {
    let too_wide: Vec<TokenId> = (0..=MAX_PERMITTED_TOKENS as TokenId).collect();
    assert!(Program::new(vec![too_wide]).is_err());
    assert!(Program::new(vec![Vec::new()]).is_err());
    // And a set at exactly the cap is legal: the refusal is `>`, not `>=`.
    let at_cap: Vec<TokenId> = (0..MAX_PERMITTED_TOKENS as TokenId).collect();
    let program = Program::new(vec![at_cap]).expect("a set at the cap");
    assert_eq!(program.step(0).map(|set| set.len()), Some(MAX_PERMITTED_TOKENS));
    let _: Arc<[TokenId]> = program.step(0).expect("the set");
}
