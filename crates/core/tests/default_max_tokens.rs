//! ADR 0045, spec vram-budget/03 AC 11 (GitHub #309): a request that names
//! no generation cap runs under the server's **default `max_tokens`**
//! (`SchedulerConfig::default_max_tokens`, the operator's
//! `--model-default-max-tokens`), clamped to what its prompt leaves of the context.
//!
//! The scheduler resolves the cap once, in `generation_budget`, and `submit`
//! writes it into the request's `params.max_tokens`, so every reader -- the
//! scheduler's hard cap, the backend's own check, a speculative round's --
//! reads the one number an explicit `max_tokens` would have put there. The
//! mock learns its limit from that field, which is how these tests see it.
//!
//! Seams (ADR 0006): the `Scheduler` trait over `MockCompute`, no GPU.

use std::sync::Arc;

use ignis_core::constrained::Schedule;
use ignis_core::thinking_budget::{BudgetOutcome, ThinkingClose, ANSWER_RESERVE};
use ignis_core::types::SubmitError;
use ignis_core::{
    ConcreteScheduler, DecisionRead, DecodeParams, FinishReason, MockCompute, RequestClass, RequestId,
    RequestInput, SchedEvent, Scheduler, SchedulerConfig, TokenId,
};

const MODEL: &str = "m";
/// The server's default cap (ADR 0045: Qwen's recommended output length for
/// complex tasks, reasoning included).
const DEFAULT_CAP: u32 = 38_912;
/// The non-default value AC 10 and 11 take from every spelling.
const OTHER_CAP: u32 = 8_192;
/// Flash-Next's and the make default's context, so the default cap is the
/// binding limit, not the context.
const CONTEXT: u32 = 262_144;
const THINK_END: TokenId = 999;
const CLOSE: [TokenId; 4] = [900, 901, THINK_END, 902];

fn scheduler(mock: Arc<MockCompute>, default_max_tokens: u32, max_sequence_tokens: u32) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_page_tokens: 64,
            max_sequence_tokens,
            kv_capacity_pages: 8 * max_sequence_tokens.div_ceil(64),
            default_max_tokens,
            ..SchedulerConfig::default()
        },
        mock,
    )
}

fn request(prompt: u32, max_tokens: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: MODEL.into(),
        tokens: (0..prompt).map(|t| 1 + t % 500).collect(),
        params: DecodeParams { max_tokens, ..DecodeParams::default() },
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

/// Run until idle: the tokens `id` generated and why it stopped.
fn run(sched: &mut ConcreteScheduler, id: RequestId) -> (u32, FinishReason, Option<BudgetOutcome>) {
    let mut generated = 0;
    for _ in 0..200_000 {
        for event in sched.advance() {
            match event {
                SchedEvent::Token { request, .. } if request == id => generated += 1,
                SchedEvent::Done { request, reason, thinking, .. } if request == id => {
                    return (generated, reason, thinking);
                }
                _ => {}
            }
        }
    }
    panic!("request {id} never finished");
}

/// What the backend was told the request may generate: the cap its first
/// prefill job carried.
fn backend_cap(mock: &MockCompute, id: RequestId) -> Option<u32> {
    mock.prefill_calls()
        .iter()
        .flatten()
        .find(|job| job.request == id)
        .expect("the request was prefilled")
        .params
        .max_tokens
}

#[test]
fn a_request_without_a_cap_stops_at_the_default_with_length() {
    for cap in [DEFAULT_CAP, OTHER_CAP] {
        // Speculative runs of eight: the cap binds a verify round as it
        // binds an explicit `max_tokens`.
        let mock = Arc::new(MockCompute::with_runs(&[8]));
        let mut sched = scheduler(mock.clone(), cap, CONTEXT);
        let id = sched.submit(request(1_000, None), RequestClass::Agent).expect("admitted");
        let (generated, reason, _) = run(&mut sched, id);
        assert_eq!(generated, cap, "default {cap}");
        assert_eq!(reason, FinishReason::Length, "default {cap}");
        assert_eq!(backend_cap(&mock, id), Some(cap), "the resolved cap is in params.max_tokens");
    }
}

#[test]
fn zero_is_no_default_and_runs_to_the_context() {
    let mock = Arc::new(MockCompute::with_runs(&[8]));
    let mut sched = scheduler(mock.clone(), 0, 4_096);
    let id = sched.submit(request(1_000, None), RequestClass::Agent).expect("admitted");
    let (generated, reason, _) = run(&mut sched, id);
    assert_eq!(generated, 4_096 - 1_000, "up to the context, as before the default existed");
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(backend_cap(&mock, id), None, "nothing is written without a default");
}

#[test]
fn a_prompt_that_leaves_less_than_the_default_gets_what_is_left() {
    let mock = Arc::new(MockCompute::with_runs(&[8]));
    let mut sched = scheduler(mock.clone(), OTHER_CAP, 4_096);
    let id = sched.submit(request(1_000, None), RequestClass::Agent).expect("not refused for the default");
    let (generated, reason, _) = run(&mut sched, id);
    assert_eq!(generated, 3_096);
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(backend_cap(&mock, id), Some(3_096));

    // A default past the context acts as the context.
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone(), CONTEXT * 4, 4_096);
    let id = sched.submit(request(4_000, None), RequestClass::Agent).expect("admitted");
    assert_eq!(run(&mut sched, id).0, 96);

    // Only a prompt that fills the context alone is refused, as before.
    let mut sched = scheduler(Arc::new(MockCompute::new()), OTHER_CAP, 4_096);
    assert_eq!(
        sched.submit(request(4_096, None), RequestClass::Agent),
        Err(SubmitError::ContextExceeded { requested: 4_096, limit: 4_096 })
    );
}

#[test]
fn an_explicit_cap_wins_larger_or_smaller() {
    for max_tokens in [100, OTHER_CAP + 1_000] {
        let mock = Arc::new(MockCompute::with_runs(&[8]));
        let mut sched = scheduler(mock.clone(), OTHER_CAP, CONTEXT);
        let id = sched.submit(request(1_000, Some(max_tokens)), RequestClass::Agent).expect("admitted");
        assert_eq!(run(&mut sched, id).0, max_tokens);
        assert_eq!(backend_cap(&mock, id), Some(max_tokens));
    }
    // One past the context is refused as before, the default notwithstanding.
    let mut sched = scheduler(Arc::new(MockCompute::new()), OTHER_CAP, 4_096);
    assert_eq!(
        sched.submit(request(1_000, Some(3_097)), RequestClass::Agent),
        Err(SubmitError::ContextExceeded { requested: 4_097, limit: 4_096 })
    );
}

#[test]
fn the_reservation_follows_the_default() {
    // The pool holds 200 pages (12,800 tokens), well short of one 262,144-token
    // context. With no default a request without a cap reserves the whole
    // context and can never fit; with 8,192 it reserves its prompt plus 8,192.
    let config = |default_max_tokens| SchedulerConfig {
        model: MODEL.into(),
        kv_page_tokens: 64,
        max_sequence_tokens: CONTEXT,
        kv_capacity_pages: 200,
        default_max_tokens,
        ..SchedulerConfig::default()
    };
    let mut none = ConcreteScheduler::with_config(config(0), Arc::new(MockCompute::new()));
    assert_eq!(none.submit(request(1_000, None), RequestClass::Agent), Err(SubmitError::Oversized));
    let mut capped = ConcreteScheduler::with_config(config(OTHER_CAP), Arc::new(MockCompute::new()));
    assert_eq!(capped.refusal(&request(1_000, None)), None, "1,000 + 8,192 tokens are 144 pages");
    assert!(capped.submit(request(1_000, None), RequestClass::Agent).is_ok());
}

#[test]
fn reasoning_counts_inside_the_default_and_the_budget_keeps_its_answer_reserve() {
    // A budget well inside the default cap (6,144, the server default until
    // 2026-10-09): its close is still forced at 6,144, and the reasoning
    // before it is part of the 38,912 the request generates in all.
    let mock = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_page_tokens: 64,
            max_sequence_tokens: CONTEXT,
            kv_capacity_pages: CONTEXT / 64,
            default_max_tokens: DEFAULT_CAP,
            thinking_close: Some(Arc::new(ThinkingClose::new(CLOSE.to_vec(), THINK_END).expect("close"))),
            ..SchedulerConfig::default()
        },
        mock.clone(),
    );
    let mut input = request(100, None);
    input.params.thinking_budget = Some(6_144);
    let id = sched.submit(input, RequestClass::Interactive).expect("admitted");
    let (generated, reason, thinking) = run(&mut sched, id);
    assert_eq!(generated, DEFAULT_CAP, "reasoning and answer share the one cap");
    assert_eq!(reason, FinishReason::Length);
    // Plain rounds: the round that finds the budget spent lets the token it
    // already drew through, and the close follows (`thinking_budget.rs`).
    assert_eq!(thinking, Some(BudgetOutcome { budget: 6_144, forced_at: Some(6_145), closed_at: Some(6_147) }));

    // At a default just above the reserve, the budget is clamped to leave
    // the answer its 2,048 tokens, as under an explicit `max_tokens`.
    let mock = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_page_tokens: 64,
            max_sequence_tokens: CONTEXT,
            kv_capacity_pages: CONTEXT / 64,
            default_max_tokens: ANSWER_RESERVE + 6,
            thinking_close: Some(Arc::new(ThinkingClose::new(CLOSE.to_vec(), THINK_END).expect("close"))),
            ..SchedulerConfig::default()
        },
        mock,
    );
    let mut input = request(100, None);
    input.params.thinking_budget = Some(6_144);
    let id = sched.submit(input, RequestClass::Interactive).expect("admitted");
    let (_, _, thinking) = run(&mut sched, id);
    assert_eq!(thinking.map(|t| t.budget), Some(6));

    // The server default since 2026-10-09, 32,768, sits inside the default
    // cap less the reserve (36,864): a turn that never closes on its own is
    // forced closed at 32,768 and answers in the rest of its 38,912.
    let mock = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_page_tokens: 64,
            max_sequence_tokens: CONTEXT,
            kv_capacity_pages: CONTEXT / 64,
            default_max_tokens: DEFAULT_CAP,
            thinking_close: Some(Arc::new(ThinkingClose::new(CLOSE.to_vec(), THINK_END).expect("close"))),
            ..SchedulerConfig::default()
        },
        mock,
    );
    let mut input = request(100, None);
    input.params.thinking_budget = Some(32_768);
    let id = sched.submit(input, RequestClass::Interactive).expect("admitted");
    let (generated, reason, thinking) = run(&mut sched, id);
    assert_eq!(generated, DEFAULT_CAP);
    assert_eq!(reason, FinishReason::Length);
    let budget = 32_768;
    assert!(budget <= DEFAULT_CAP - ANSWER_RESERVE, "inside the cap less the reserve: not clamped");
    assert_eq!(
        thinking,
        Some(BudgetOutcome { budget, forced_at: Some(budget + 1), closed_at: Some(budget + 3) })
    );
}

#[test]
fn a_decision_a_warm_up_and_a_constrained_decode_keep_their_own_budget() {
    // A decision generates nothing: no cap is written, and it reserves its
    // prompt alone -- here a pool of one page holds it.
    let mock = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_page_tokens: 64,
            max_sequence_tokens: CONTEXT,
            kv_capacity_pages: 1,
            default_max_tokens: DEFAULT_CAP,
            ..SchedulerConfig::default()
        },
        mock.clone(),
    );
    let mut decision = request(10, None);
    decision.opener_tokens = Some(10);
    decision.decision = Some(DecisionRead::Answers(Arc::from(vec![32, 33])));
    let id = sched.submit(decision, RequestClass::Agent).expect("a decision fits its prompt's page");
    let (generated, _, _) = run(&mut sched, id);
    assert_eq!(generated, 0);
    assert_eq!(backend_cap(&mock, id), None);

    // A warm-up, the other prefill-only request (GitHub #282): no cap
    // written, nothing generated, its prompt the whole reservation.
    let mock = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_page_tokens: 64,
            max_sequence_tokens: CONTEXT,
            kv_capacity_pages: 1,
            default_max_tokens: DEFAULT_CAP,
            ..SchedulerConfig::default()
        },
        mock.clone(),
    );
    let mut warm_up = request(10, None);
    warm_up.warm_up = true;
    let id = sched.submit(warm_up, RequestClass::Interactive).expect("a warm-up fits its prompt's page");
    let (generated, _, _) = run(&mut sched, id);
    assert_eq!(generated, 0);
    assert_eq!(backend_cap(&mock, id), None);

    // A constrained decode runs its schedule to the end, whatever the default.
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone(), 2, CONTEXT);
    let mut constrained = request(8, None);
    constrained.constrained =
        Some(Arc::new(Schedule::new(vec![vec![10, 11], vec![20, 21], vec![30, 31], vec![40]]).expect("a program")));
    let id = sched.submit(constrained, RequestClass::Agent).expect("admitted");
    let (generated, reason, _) = run(&mut sched, id);
    assert_eq!((generated, reason), (4, FinishReason::Stop), "the whole schedule, past a default of 2");
    assert_eq!(backend_cap(&mock, id), None);
}
