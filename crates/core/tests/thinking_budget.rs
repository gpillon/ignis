//! The thinking budget (2026-09-24, `ignis_core::thinking_budget`): once a
//! request has emitted its budget with the reasoning block still open, the
//! scheduler forces the model's own close, one single-token permitted set per
//! round, through the same seam a constrained decode uses — and then lets the
//! request generate freely again.
//!
//! The mock runs speculative runs of four tokens, so the switch from verify
//! rounds to the forced plain rounds and back is exercised, and it models the
//! leaf's one-round lag for a set handed to a free lane mid-run.
//!
//! The budget always leaves [`ANSWER_RESERVE`] tokens of the request's
//! generation for the answer (spec server/08), so every request here asks for
//! that much more than the budget it is testing.

use std::sync::Arc;

use ignis_core::thinking_budget::{BudgetOutcome, ThinkingClose, ANSWER_RESERVE};
use ignis_core::{
    Compute, ConcreteScheduler, DecodeParams, MockCompute, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId,
};

const THINK_END: TokenId = 999;
const CLOSE: [TokenId; 4] = [900, 901, THINK_END, 902];
/// The answer room plus forty tokens: a budget of 10 fits well inside it.
const MAX_TOKENS: u32 = ANSWER_RESERVE + 40;

fn scheduler(compute: Arc<dyn Compute>, close: bool) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "m".into(),
            thinking_close: close
                .then(|| Arc::new(ThinkingClose::new(CLOSE.to_vec(), THINK_END).expect("close"))),
            ..SchedulerConfig::default()
        },
        compute,
    )
}

fn submit(sched: &mut ConcreteScheduler, budget: Option<u32>, max_tokens: u32) -> RequestId {
    sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "m".into(),
                tokens: vec![1, 2, 3],
                params: DecodeParams {
                    max_tokens: Some(max_tokens),
                    thinking_budget: budget,
                    ..DecodeParams::default()
                },
                constrained: None,
            },
            RequestClass::Interactive,
        )
        .expect("submit")
}

/// Run the scheduler idle: the request's emitted tokens, and what its finish
/// event said the budget did.
fn run(sched: &mut ConcreteScheduler, id: RequestId) -> (Vec<TokenId>, Option<BudgetOutcome>) {
    let mut events = Vec::new();
    for _ in 0..10_000 {
        if sched.is_idle() {
            let tokens = events
                .iter()
                .filter_map(|e| match e {
                    SchedEvent::Token { request, token } if *request == id => Some(*token),
                    _ => None,
                })
                .collect();
            let thinking = events
                .iter()
                .find_map(|e| match e {
                    SchedEvent::Done { request, thinking, .. } if *request == id => Some(*thinking),
                    _ => None,
                })
                .expect("the request finished");
            return (tokens, thinking);
        }
        events.extend(sched.advance());
    }
    panic!("the scheduler never went idle");
}

fn close_at(tokens: &[TokenId]) -> Vec<usize> {
    tokens
        .windows(CLOSE.len())
        .enumerate()
        .filter(|(_, w)| *w == CLOSE)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn a_spent_budget_forces_the_close_once_and_then_generates_freely() {
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, Some(10), MAX_TOKENS);
    let (out, thinking) = run(&mut sched, id);
    assert_eq!(out.len(), MAX_TOKENS as usize, "the whole max_tokens is generated");
    let at = close_at(&out);
    assert_eq!(at.len(), 1, "the close appears exactly once: {:?}", &out[..20]);
    // Verify rounds of four reach 12 tokens before a round finds the budget
    // spent; that round emits the one token it had already drawn freely, and
    // the close follows.
    assert_eq!(at[0], 13, "{:?}", &out[..20]);
    assert!(out[at[0] + CLOSE.len()..].iter().all(|t| !CLOSE.contains(t)), "free after the close");
    // The forced rounds were plain rounds with a one-token set each.
    let forced: Vec<usize> = mock
        .decode_calls()
        .iter()
        .flatten()
        .filter_map(|job| job.permitted.as_ref().map(|set| set.len()))
        .collect();
    assert_eq!(forced, vec![1; CLOSE.len()]);
    // The finish event says so: the budget it ran under, and the tokens it
    // had emitted when the close began.
    assert_eq!(thinking, Some(BudgetOutcome { budget: 10, forced_at: Some(13) }));
}

#[test]
fn no_budget_or_no_close_never_forces() {
    for (budget, close) in [(None, true), (Some(10), false)] {
        let mock = Arc::new(MockCompute::with_runs(&[4]));
        let mut sched = scheduler(mock.clone(), close);
        let id = submit(&mut sched, budget, MAX_TOKENS);
        let (out, thinking) = run(&mut sched, id);
        assert!(close_at(&out).is_empty(), "budget {budget:?}, close {close}");
        assert!(mock.decode_calls().iter().flatten().all(|job| job.permitted.is_none()));
        // No budget in effect, so nothing to report — not a `false`.
        assert_eq!(thinking, None, "budget {budget:?}, close {close}");
    }
}

#[test]
fn a_budget_that_would_crowd_out_the_answer_is_clamped_below_the_request() {
    // Six tokens of room above the answer reserve: the budget of 10 the
    // request asked for would leave the answer less than the reserve, so the
    // close is forced after 6 (plain rounds: the round that finds 6 emitted
    // lets the token it already drew through, and the close follows).
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, Some(10), ANSWER_RESERVE + 6);
    let (out, thinking) = run(&mut sched, id);
    assert_eq!(close_at(&out), vec![7]);
    assert_eq!(thinking, Some(BudgetOutcome { budget: 6, forced_at: Some(7) }));
}

#[test]
fn a_generation_no_longer_than_the_answer_reserve_runs_without_a_budget() {
    // No room above the reserve: no budget at all, and never a close forced
    // at token 0.
    for max_tokens in [ANSWER_RESERVE, 40] {
        let mock = Arc::new(MockCompute::new());
        let mut sched = scheduler(mock.clone(), true);
        let id = submit(&mut sched, Some(10), max_tokens);
        let (out, thinking) = run(&mut sched, id);
        assert!(close_at(&out).is_empty(), "max_tokens {max_tokens}");
        assert!(mock.decode_calls().iter().flatten().all(|job| job.permitted.is_none()));
        assert_eq!(thinking, None, "max_tokens {max_tokens}");
    }
}

#[test]
fn a_natural_close_before_the_budget_reports_the_budget_but_no_forced_close() {
    // The mock's own token at step 2 is the reasoning block's end marker: the
    // model closed the block itself, long before its budget.
    let mock = Arc::new(MockCompute::new());
    let think_end = mock.token_for(0, 2);
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "m".into(),
            thinking_close: Some(Arc::new(
                ThinkingClose::new(vec![900, 901, think_end, 902], think_end).expect("close"),
            )),
            ..SchedulerConfig::default()
        },
        mock.clone(),
    );
    let id = submit(&mut sched, Some(10), MAX_TOKENS);
    assert_eq!(id, 0, "the mock's stream is keyed by the request id");
    let (out, thinking) = run(&mut sched, id);
    assert_eq!(out[2], think_end);
    assert!(mock.decode_calls().iter().flatten().all(|job| job.permitted.is_none()));
    assert_eq!(thinking, Some(BudgetOutcome { budget: 10, forced_at: None }));
}
