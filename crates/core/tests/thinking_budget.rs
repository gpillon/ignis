//! The thinking budget (2026-09-24, `ignis_core::thinking_budget`): once a
//! request has emitted its budget with the reasoning block still open, the
//! scheduler forces the model's own close, one single-token permitted set per
//! round, through the same seam a constrained decode uses — and then lets the
//! request generate freely again.
//!
//! The mock runs speculative runs of four tokens, so the switch from verify
//! rounds to the forced plain rounds and back is exercised, and it models the
//! leaf's one-round lag for a set handed to a free lane mid-run.

use std::sync::Arc;

use ignis_core::thinking_budget::ThinkingClose;
use ignis_core::{
    Compute, ConcreteScheduler, DecodeParams, MockCompute, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId,
};

const THINK_END: TokenId = 999;
const CLOSE: [TokenId; 4] = [900, 901, THINK_END, 902];

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

fn submit(sched: &mut ConcreteScheduler, budget: Option<u32>) -> RequestId {
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
                    max_tokens: Some(40),
                    thinking_budget: budget,
                    ..DecodeParams::default()
                },
                constrained: None,
            },
            RequestClass::Interactive,
        )
        .expect("submit")
}

fn emitted(sched: &mut ConcreteScheduler, id: RequestId) -> Vec<TokenId> {
    let mut events = Vec::new();
    for _ in 0..1000 {
        if sched.is_idle() {
            return events
                .iter()
                .filter_map(|e| match e {
                    SchedEvent::Token { request, token } if *request == id => Some(*token),
                    _ => None,
                })
                .collect();
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
    let id = submit(&mut sched, Some(10));
    let out = emitted(&mut sched, id);
    assert_eq!(out.len(), 40, "the whole max_tokens is generated: {out:?}");
    let at = close_at(&out);
    assert_eq!(at.len(), 1, "the close appears exactly once: {out:?}");
    // Verify rounds of four reach 12 tokens before a round finds the budget
    // spent; that round emits the one token it had already drawn freely, and
    // the close follows.
    assert_eq!(at[0], 13, "{out:?}");
    assert!(out[at[0] + CLOSE.len()..].iter().all(|t| !CLOSE.contains(t)), "free after the close");
    // The forced rounds were plain rounds with a one-token set each.
    let forced: Vec<usize> = mock
        .decode_calls()
        .iter()
        .flatten()
        .filter_map(|job| job.permitted.as_ref().map(|set| set.len()))
        .collect();
    assert_eq!(forced, vec![1; CLOSE.len()]);
}

#[test]
fn no_budget_or_no_close_never_forces() {
    for (budget, close) in [(None, true), (Some(10), false)] {
        let mock = Arc::new(MockCompute::with_runs(&[4]));
        let mut sched = scheduler(mock.clone(), close);
        let id = submit(&mut sched, budget);
        let out = emitted(&mut sched, id);
        assert!(close_at(&out).is_empty(), "budget {budget:?}, close {close}: {out:?}");
        assert!(mock.decode_calls().iter().flatten().all(|job| job.permitted.is_none()));
    }
}
