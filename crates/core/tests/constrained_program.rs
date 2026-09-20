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

// ---------------------------------------------------------------------------
// The scheduler's program (GitHub #242, layer 4)
// ---------------------------------------------------------------------------
//
// One `ConcreteScheduler` over the mock above, which is the whole point of
// modelling the lag there: the scheduler's cursor, its budget and its
// stopping condition are testable without a GPU (ADR 0006).

mod scheduler {
    use std::sync::Arc;

    use ignis_core::mock::MockCompute;
    use ignis_core::program::Program;
    use ignis_core::types::{
        DecodeParams, RequestClass, RequestInput, SchedEvent, TokenId,
    };
    use ignis_core::{ConcreteScheduler, FinishReason, Scheduler, SchedulerConfig};

    const MODEL: &str = "qwen3.8-27b";

    /// A program over four disjoint steps, the last of which permits one
    /// token — a forced literal.
    fn steps() -> Vec<Vec<TokenId>> {
        vec![
            vec![10, 11, 12],
            vec![20, 21, 22],
            vec![30, 31, 32],
            vec![40],
        ]
    }

    fn program_request(steps: Vec<Vec<TokenId>>, max_tokens: Option<u32>) -> RequestInput {
        RequestInput {
            model: MODEL.into(),
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            params: DecodeParams {
                max_tokens,
                ..DecodeParams::default()
            },
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            decision: None,
            program: Some(Arc::new(Program::new(steps).expect("a legal program"))),
        }
    }

    fn run(input: RequestInput) -> (Vec<TokenId>, Option<Vec<ignis_core::program::Draw>>, FinishReason) {
        let mut sched =
            ConcreteScheduler::with_config(
                SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
                Arc::new(MockCompute::new()),
            );
        sched.submit(input, RequestClass::Agent).expect("admitted");
        let mut tokens = Vec::new();
        let mut finished = None;
        let mut ticks = 0;
        while !sched.is_idle() {
            for event in sched.advance() {
                match event {
                    SchedEvent::Token { token, .. } => tokens.push(token),
                    SchedEvent::Done { reason, drawn, .. } => finished = Some((drawn, reason)),
                    _ => {}
                }
            }
            ticks += 1;
            assert!(ticks < 200, "the engine never went idle");
        }
        let (drawn, reason) = finished.expect("the request completed");
        (tokens, drawn, reason)
    }

    /// The cursor: one token per step, each from its own step's set, and the
    /// trace beside it.
    #[test]
    fn a_program_emits_its_schedule_and_nothing_else() {
        let steps = steps();
        let (tokens, drawn, reason) = run(program_request(steps.clone(), None));
        assert_eq!(tokens.len(), steps.len(), "one token per step, no more");
        for (index, (token, step)) in tokens.iter().zip(&steps).enumerate() {
            assert!(
                step.contains(token),
                "token {index} is {token}, which step {index} ({step:?}) does not permit — \
                 the steps are disjoint, so this is the cursor being off by one"
            );
        }
        assert_eq!(
            reason,
            FinishReason::Stop,
            "a number that has read its last digit is finished, not truncated: \
             `length` would tell every caller their answer may be incomplete"
        );
        let drawn = drawn.expect("a program's trace rides its completion");
        assert_eq!(
            drawn.iter().map(|draw| draw.token).collect::<Vec<_>>(),
            tokens,
            "the trace is the run, in order"
        );
        assert_eq!(
            drawn.last().expect("four draws").probability,
            1.0,
            "and the forced literal at the end is certain by arithmetic"
        );
        for draw in &drawn[..3] {
            assert!(
                draw.probability > 0.0 && draw.probability < 1.0,
                "while a real step reports a real confidence: {}",
                draw.probability
            );
        }
    }

    /// `max_tokens` never reaches the backend at all.
    ///
    /// Not the same claim as the one below, and the one that matters on the
    /// card: `RuntimeCompute` finishes a lane whose `generated` has reached
    /// `max_tokens` *before* the leaf runs, and it cannot tell a program's
    /// last round (which carries no set) from an ordinary one. A program
    /// that carried a `max_tokens` of 1 would therefore return one digit
    /// with `length` on a GPU and four with `stop` here, and only the mock
    /// would agree with the doc comment.
    #[test]
    fn a_programs_jobs_carry_no_max_tokens_for_a_backend_to_enforce() {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::with_config(
            SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
            compute.clone(),
        );
        sched
            .submit(program_request(steps(), Some(1)), RequestClass::Agent)
            .expect("admitted");
        let mut ticks = 0;
        while !sched.is_idle() {
            sched.advance();
            ticks += 1;
            assert!(ticks < 200, "the engine never went idle");
        }
        let prefill_caps: Vec<_> = compute
            .prefill_calls()
            .iter()
            .flatten()
            .map(|job| job.params.max_tokens)
            .collect();
        let decode_caps: Vec<_> = compute
            .decode_calls()
            .iter()
            .flatten()
            .map(|job| job.params.max_tokens)
            .collect();
        assert!(!prefill_caps.is_empty() && !decode_caps.is_empty(), "the run happened");
        assert!(
            prefill_caps.iter().chain(&decode_caps).all(Option::is_none),
            "a program's jobs carry no cap for any backend to cut it short with:              prefill {prefill_caps:?}, decode {decode_caps:?}"
        );
    }

    /// `max_tokens` is not a program's budget, in either direction.
    #[test]
    fn max_tokens_neither_shortens_nor_lengthens_a_program() {
        let steps = steps();
        for max_tokens in [Some(1), Some(64), None] {
            let (tokens, drawn, reason) = run(program_request(steps.clone(), max_tokens));
            assert_eq!(
                tokens.len(),
                steps.len(),
                "max_tokens = {max_tokens:?} must not change how many digits a \
                 number has: a short number is not a short answer, it is a wrong one"
            );
            assert_eq!(reason, FinishReason::Stop);
            assert_eq!(drawn.expect("a trace").len(), steps.len());
        }
    }

    /// One step: the degenerate program, and the one the off-by-one is
    /// easiest to get wrong on — the prefill draws the only token and the
    /// single round carries no set at all.
    #[test]
    fn a_one_step_program_is_one_token() {
        let (tokens, drawn, reason) = run(program_request(vec![vec![7, 8, 9]], None));
        assert_eq!(tokens.len(), 1);
        assert!([7, 8, 9].contains(&tokens[0]), "drawn by the prefill, from its set");
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(drawn.expect("a trace").len(), 1);
    }

    /// An ordinary request is untouched: no trace, and `Length` still means
    /// `length`.
    #[test]
    fn an_ordinary_request_carries_no_trace_and_still_stops_on_length() {
        let mut input = program_request(steps(), Some(3));
        input.program = None;
        let (tokens, drawn, reason) = run(input);
        assert_eq!(tokens.len(), 3, "its max_tokens, as before");
        assert_eq!(reason, FinishReason::Length);
        assert!(
            drawn.is_none(),
            "and `None` rather than an empty trace: a reader can tell a request \
             that was never a program from a program that emitted nothing"
        );
    }

    /// The empty-last-chunk trap, for a program: its first token is drawn by
    /// its prefill, so it must always be left something to prefill.
    #[test]
    fn a_program_is_always_left_a_token_to_prefill() {
        let input = program_request(steps(), None);
        assert_eq!(
            input.reuse_reach(),
            input.tokens.len() - 1,
            "a program may match retained state over all but one of its prompt \
             tokens — a chunk with nothing to prefill draws nothing, and the run \
             would begin with whatever the claimed state left pending"
        );
        assert_eq!(
            input.publish_reach(),
            input.tokens.len() - 1,
            "and publishes only what it could itself claim"
        );
    }
}
