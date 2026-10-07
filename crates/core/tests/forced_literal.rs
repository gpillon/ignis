//! A **forced literal** (GitHub #286, spec server/11) through the scheduler:
//! the tokens `tool_choice` forces, one single-token permitted set per round,
//! after which the request generates freely — at the generation's first
//! token with thinking off, right after the reasoning block with it on.
//!
//! The mock models the leaf's one-round lag (the prefill draws the first
//! token of a run; a set handed to a free lane is drawn for the round after)
//! and, with `with_runs`, speculative runs that commit past a `</think>` in
//! one round.

use std::sync::Arc;

use ignis_core::constrained::Schedule;
use ignis_core::forced_literal::ForcedLiteral;
use ignis_core::thinking_budget::{ThinkingClose, ANSWER_RESERVE};
use ignis_core::{
    Compute, ConcreteScheduler, DecodeParams, MockCompute, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId,
};

const OPEN: TokenId = 900;
const NL: TokenId = 901;
/// `<tool_call>`, `\n`, `<`, `function`.
const OPENER: [TokenId; 4] = [OPEN, NL, 902, 903];
const THINK_END: TokenId = 999;

fn scheduler(compute: Arc<dyn Compute>, close: Option<ThinkingClose>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig { model: "m".into(), thinking_close: close.map(Arc::new), ..SchedulerConfig::default() },
        compute,
    )
}

fn input(max_tokens: u32, budget: Option<u32>, forced: Option<ForcedLiteral>) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: "m".into(),
        tokens: vec![1, 2, 3],
        params: DecodeParams { max_tokens: Some(max_tokens), thinking_budget: budget, ..DecodeParams::default() },
        constrained: None,
        forced_literal: forced.map(Arc::new),
        warm_up: false,
    }
}

fn submit(sched: &mut ConcreteScheduler, input: RequestInput) -> RequestId {
    let id = sched.submit(input, RequestClass::Interactive).expect("submit");
    assert_eq!(id, 0, "the mock's stream is keyed by the request id");
    id
}

/// Run the scheduler idle and return the request's emitted tokens.
fn run(sched: &mut ConcreteScheduler, id: RequestId) -> Vec<TokenId> {
    let mut tokens = Vec::new();
    for _ in 0..10_000 {
        if sched.is_idle() {
            return tokens;
        }
        for event in sched.advance() {
            if let SchedEvent::Token { request, token } = event {
                if request == id {
                    tokens.push(token);
                }
            }
        }
    }
    panic!("the scheduler never went idle");
}

/// The width of every permitted set the decode rounds were handed, in order.
fn forced_rounds(mock: &MockCompute) -> Vec<Vec<TokenId>> {
    mock.decode_calls()
        .iter()
        .flatten()
        .filter_map(|job| job.permitted.as_ref().map(|set| set.to_vec()))
        .collect()
}

#[test]
fn at_the_generation_the_prefill_draws_the_first_token_and_a_spent_literal_leaves_the_run_generating() {
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(mock.clone(), None);
    let literal = ForcedLiteral::at_generation(OPENER.to_vec()).unwrap();
    let id = submit(&mut sched, input(20, None, Some(literal)));
    let out = run(&mut sched, id);
    // The first token is the prefill's draw, under the literal's first set.
    let prefill = mock.prefill_calls().concat();
    assert_eq!(prefill.last().and_then(|job| job.permitted.as_deref()), Some(&[OPEN][..]));
    assert_eq!(&out[..4], &OPENER[..], "{out:?}");
    // Released: the model's own tokens follow, to the whole max_tokens.
    assert_eq!(out.len(), 20);
    assert_eq!(out[4], mock.token_for(id, 4));
    assert!(out[4..].iter().all(|t| !OPENER.contains(t)));
    assert_eq!(forced_rounds(&mock), vec![vec![NL], vec![902], vec![903]]);
}

/// The contrast the spec's release rests on: a spent `Schedule` ends its
/// run, and every constrained decode depends on that.
#[test]
fn a_spent_schedule_still_ends_its_run() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone(), None);
    let mut constrained = input(20, None, None);
    constrained.constrained = Some(Arc::new(Schedule::new(OPENER.iter().map(|&t| vec![t]).collect()).unwrap()));
    let id = submit(&mut sched, constrained);
    assert_eq!(run(&mut sched, id), OPENER.to_vec());
}

#[test]
fn after_reasoning_the_models_own_close_is_followed_by_the_joiner_and_then_the_literal() {
    // Speculative runs of four: the round that commits `</think>` (step 2)
    // commits step 3 with it, and the round handed the joiner lets through
    // the step-4 token it already drew — unseen when the joiner was chosen.
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let think_end = mock.token_for(0, 2);
    let mut sched = scheduler(mock.clone(), None);
    let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), think_end, 4).unwrap();
    let id = submit(&mut sched, input(20, None, Some(literal)));
    let out = run(&mut sched, id);
    let model = |step| mock.token_for(id, step);
    assert_eq!(
        &out[..10],
        &[model(0), model(1), think_end, model(3), model(4), NL, OPEN, NL, 902, 903][..],
        "{out:?}"
    );
    assert_eq!(out.len(), 20, "released, the request generates to its cap");
    assert_eq!(out[10], model(10));
    assert_eq!(forced_rounds(&mock), vec![vec![NL], vec![OPEN], vec![NL], vec![902], vec![903]]);
}

#[test]
fn an_unseen_token_that_opened_the_call_is_not_opened_twice() {
    // The close ends the first run (step 3); the token the joiner's round
    // lets through (step 4) is the model's own `<tool_call>`.
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let think_end = mock.token_for(0, 3);
    let open = mock.token_for(0, 4);
    let mut sched = scheduler(mock.clone(), None);
    let literal = ForcedLiteral::after_reasoning(vec![open, NL, 902, 903], think_end, 4).unwrap();
    let id = submit(&mut sched, input(20, None, Some(literal)));
    let out = run(&mut sched, id);
    assert_eq!(&out[3..8], &[think_end, open, NL, 902, 903][..], "{out:?}");
    assert_eq!(out.iter().filter(|&&t| t == open).count(), 1);
}

/// A speculative run commits `</think>` and the model's own `<tool_call>`
/// together, ending there. The name is not the model's yet: the unseen
/// token after it is the `\n` every call writes next, and the literal
/// resumes after that — `<`, `function`, and here the name, which is what
/// keeps a named call on the tool asked for.
#[test]
fn the_models_own_opener_committed_with_its_close_is_continued_to_the_name() {
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let think_end = mock.token_for(0, 2);
    let open = mock.token_for(0, 3);
    let mut sched = scheduler(mock.clone(), None);
    // `<tool_call>`, `\n`, `<`, `function` are common; 904 is the name.
    let literal = ForcedLiteral::after_reasoning(vec![open, NL, 902, 903, 904], think_end, 4).unwrap();
    let id = submit(&mut sched, input(20, None, Some(literal)));
    let out = run(&mut sched, id);
    assert_eq!(&out[2..8], &[think_end, open, mock.token_for(id, 4), 902, 903, 904][..], "{out:?}");
    assert_eq!(out.len(), 20, "released after the name");
    assert_eq!(forced_rounds(&mock), vec![vec![902], vec![903], vec![904]]);
}

/// A run that committed the common tokens and went on into a name of its
/// own is left alone: forcing another name into it would write neither.
#[test]
fn the_models_own_call_past_the_common_tokens_is_never_forced() {
    let mock = Arc::new(MockCompute::with_runs(&[8]));
    let think_end = mock.token_for(0, 2);
    let common: Vec<TokenId> = (3..7).map(|step| mock.token_for(0, step)).collect();
    let mut sched = scheduler(mock.clone(), None);
    let literal = ForcedLiteral::after_reasoning([&common[..], &[904]].concat(), think_end, 4).unwrap();
    let id = submit(&mut sched, input(20, None, Some(literal)));
    assert_eq!(run(&mut sched, id).len(), 20);
    assert!(forced_rounds(&mock).is_empty(), "nothing forced");
}

#[test]
fn a_close_the_budget_forced_is_followed_by_the_literal_with_no_joiner_between() {
    // The budget's close, `\n\n`, `</think>`, `\n\n`, forced after 6 plain
    // tokens and the one drawn freely in the lag; the literal follows its
    // last token directly. The two never share a round.
    const CLOSE: [TokenId; 3] = [800, THINK_END, 800];
    let mock = Arc::new(MockCompute::new());
    let close = ThinkingClose::new(CLOSE.to_vec(), THINK_END).unwrap();
    let mut sched = scheduler(mock.clone(), Some(close));
    let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), THINK_END, 4).unwrap();
    let id = submit(&mut sched, input(ANSWER_RESERVE + 6, Some(6), Some(literal)));
    let out = run(&mut sched, id);
    assert_eq!(&out[7..14], &[800, THINK_END, 800, OPEN, NL, 902, 903][..], "{:?}", &out[..16]);
    assert_eq!(out[14], mock.token_for(id, 14), "then the model's own");
    let rounds = forced_rounds(&mock);
    assert_eq!(rounds, vec![vec![800], vec![THINK_END], vec![800], vec![OPEN], vec![NL], vec![902], vec![903]]);
}

/// A literal that starts at the generation has its first token drawn by
/// the prefill, so — like a constrained decode — the request must be left a
/// token to prefill: a claim over its whole prompt would start the call on
/// whatever the claimed state left pending, and a prefix published over the
/// whole prompt would hand its forced `<tool_call>` to the next request that
/// claims it. A literal forced after the reasoning block holds nothing back.
#[test]
fn a_literal_drawn_by_the_prefill_leaves_the_prompt_a_token_to_prefill() {
    let at_generation = input(20, None, Some(ForcedLiteral::at_generation(OPENER.to_vec()).unwrap()));
    assert_eq!(at_generation.prefill_tail(), 1);
    assert_eq!((at_generation.reuse_reach(), at_generation.publish_reach()), (2, 2));
    let after = input(20, None, Some(ForcedLiteral::after_reasoning(OPENER.to_vec(), THINK_END, 4).unwrap()));
    assert_eq!(after.prefill_tail(), 0);
    assert_eq!((after.reuse_reach(), after.publish_reach()), (3, 3));
}

#[test]
fn a_request_with_no_literal_is_never_handed_a_set() {
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(mock.clone(), None);
    let id = submit(&mut sched, input(20, None, None));
    let out = run(&mut sched, id);
    assert_eq!(out, (0..20).map(|step| mock.token_for(id, step)).collect::<Vec<_>>());
    assert!(forced_rounds(&mock).is_empty());
    assert!(mock.prefill_calls().concat().iter().all(|job| job.permitted.is_none()));
}
