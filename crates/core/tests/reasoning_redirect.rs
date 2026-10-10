//! The reasoning redirect (GitHub #315, ADR 0048) on the scheduler's side of
//! the seam: an EOS drawn while a request's reasoning block is open becomes
//! the block's `</think>`, and the request goes on to answer.
//!
//! The scheduler hands the leaf the close id on every draw it makes while a
//! request that starts inside its reasoning has not been seen closing it; the
//! mock redirects the way the leaf does, one round ahead of the token it
//! emits (`MockCompute::eos_after`). The request log's field rides the finish
//! event.

use std::sync::Arc;

use ignis_core::scheduler::{DecodeJob, PrefillJob};
use ignis_core::thinking_budget::{BudgetOutcome, ThinkingClose, ANSWER_RESERVE};
use ignis_core::{
    Compute, ConcreteScheduler, DecodeParams, FinishReason, MockCompute, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId,
};

const THINK_END: TokenId = 999;
const CLOSE: [TokenId; 4] = [900, 901, THINK_END, 902];
const MAX_TOKENS: u32 = 24;

fn scheduler(compute: Arc<dyn Compute>, close: bool) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "m".into(),
            // Three prompt chunks for a ten-token prompt.
            serving_chunk_tokens: 4,
            thinking_close: close.then(|| Arc::new(ThinkingClose::new(CLOSE.to_vec(), THINK_END).expect("close"))),
            ..SchedulerConfig::default()
        },
        compute,
    )
}

fn submit(sched: &mut ConcreteScheduler, params: DecodeParams) -> RequestId {
    sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                reuse_boundaries: Vec::new(),
                model: "m".into(),
                tokens: (1..=10).collect(),
                params,
                constrained: None,
                forced_literal: None,
                warm_up: false,
            },
            RequestClass::Interactive,
        )
        .expect("submit")
}

/// A thinking request: it starts inside its reasoning block.
fn thinking(max_tokens: u32) -> DecodeParams {
    DecodeParams {
        max_tokens: Some(max_tokens),
        starts_in_reasoning: true,
        ..DecodeParams::default()
    }
}

/// What a request's finish event said, beside its emitted tokens.
struct Finished {
    tokens: Vec<TokenId>,
    reason: FinishReason,
    thinking: Option<BudgetOutcome>,
    redirected_at: Option<u32>,
}

fn run(sched: &mut ConcreteScheduler, id: RequestId) -> Finished {
    let mut tokens = Vec::new();
    for _ in 0..10_000 {
        for event in sched.advance() {
            match event {
                SchedEvent::Token { request, token } if request == id => tokens.push(token),
                SchedEvent::Done {
                    request,
                    reason,
                    thinking,
                    reasoning_redirected_at,
                    ..
                } if request == id => {
                    return Finished {
                        tokens,
                        reason,
                        thinking,
                        redirected_at: reasoning_redirected_at,
                    }
                }
                _ => {}
            }
        }
    }
    panic!("the request never finished");
}

fn prefill_closes(mock: &MockCompute) -> Vec<Option<TokenId>> {
    mock.prefill_calls().iter().flatten().map(|job: &PrefillJob| job.reasoning_close).collect()
}

fn decode_closes(mock: &MockCompute) -> Vec<Option<TokenId>> {
    mock.decode_calls().iter().flatten().map(|job: &DecodeJob| job.reasoning_close).collect()
}

// ── the flag (AC 4) ─────────────────────────────────────────────────────

/// The prompt's last chunk draws the first token, and every round draws the
/// next: each carries the close id while the block is open, and none does
/// once the scheduler has seen the `</think>`.
#[test]
fn the_close_id_rides_every_draw_until_the_block_closes() {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 5);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, thinking(MAX_TOKENS));
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens[5], THINK_END, "{:?}", finished.tokens);

    // The ten-token prompt in chunks of four: only the last one draws.
    assert_eq!(prefill_closes(&mock), vec![None, None, Some(THINK_END)]);
    // The rounds that emit tokens 0..=5 draw with the block open as far as
    // the scheduler knows -- the one that emits the `</think>` included, since
    // it learns of it from that round -- and every round after draws without.
    let rounds = decode_closes(&mock);
    assert_eq!(rounds[..6], [Some(THINK_END); 6], "{rounds:?}");
    assert!(rounds[6..].iter().all(Option::is_none), "{rounds:?}");
}

/// Thinking off, `ignore_eos` and a scheduler with no close configured never
/// hand the leaf a close id.
#[test]
fn no_close_id_without_an_open_block_a_stop_id_or_a_close() {
    let cases = [
        (DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() }, true),
        (DecodeParams { ignore_eos: true, ..thinking(MAX_TOKENS) }, true),
        (thinking(MAX_TOKENS), false),
    ];
    for (params, close) in cases {
        let mock = Arc::new(MockCompute::new());
        let mut sched = scheduler(mock.clone(), close);
        let id = submit(&mut sched, params);
        run(&mut sched, id);
        let label = format!("{params:?}, close {close}");
        assert!(prefill_closes(&mock).iter().all(Option::is_none), "{label}");
        assert!(decode_closes(&mock).iter().all(Option::is_none), "{label}");
    }
}

/// A block the model closes itself is never redirected: the rounds after it
/// carry no close id, and an EOS there ends the turn.
#[test]
fn a_block_the_model_closed_ends_on_its_eos() {
    let mock = Arc::new(MockCompute::new());
    // Whatever the mock writes at step 2 is this close's end marker.
    let natural_end = mock.token_for(0, 2);
    let close = ThinkingClose::new(vec![900, natural_end, 902], natural_end).expect("close");
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "m".into(),
            thinking_close: Some(Arc::new(close)),
            ..SchedulerConfig::default()
        },
        mock.clone(),
    );
    mock.eos_after(0, 6);
    let id = submit(&mut sched, thinking(MAX_TOKENS));
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens.len(), 6, "{:?}", finished.tokens);
    assert_eq!(finished.reason, FinishReason::Stop);
    assert_eq!(finished.redirected_at, None);
}

// ── the mock's emulation (AC 5) ─────────────────────────────────────────

/// An EOS inside the open block is emitted as `</think>`, at the same index,
/// and the request goes on generating -- to its own EOS once the block is
/// closed, which then ends the turn as an EOS always has.
#[test]
fn an_eos_inside_the_open_block_becomes_the_close_and_the_answer_follows() {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 5);
    mock.eos_after(0, 9);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, thinking(MAX_TOKENS));
    let finished = run(&mut sched, id);
    let expected: Vec<TokenId> = (0..9).map(|step| if step == 5 { THINK_END } else { mock.token_for(0, step) }).collect();
    assert_eq!(finished.tokens, expected);
    assert_eq!(finished.reason, FinishReason::Stop, "the answer's own EOS ends the turn");
    assert_eq!(finished.redirected_at, Some(5));
}

/// The same EOS on a request that does not start inside its reasoning ends
/// the turn, as before.
#[test]
fn a_thinking_off_request_ends_on_the_same_eos() {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 5);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() });
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens.len(), 5);
    assert_eq!(finished.reason, FinishReason::Stop);
    assert_eq!(finished.redirected_at, None);
}

/// The prefill's own draw is redirected too: the first token is the close.
#[test]
fn an_eos_as_the_first_token_is_redirected_at_the_prefill() {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 0);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, thinking(MAX_TOKENS));
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens[0], THINK_END);
    assert_eq!(finished.tokens.len(), MAX_TOKENS as usize, "generation goes on");
    assert_eq!(finished.redirected_at, Some(0));
}

/// A speculative run that accepts the EOS is cut before it: the run ends
/// short, and the next round begins with the close.
#[test]
fn an_accepted_eos_is_cut_before_and_the_close_comes_next() {
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    // The second run covers steps 4..=7.
    mock.eos_after(0, 6);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, thinking(MAX_TOKENS));
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens[6], THINK_END, "{:?}", finished.tokens);
    assert!(finished.tokens[..6].iter().chain(&finished.tokens[7..]).all(|&t| t != THINK_END));
    assert_eq!(finished.tokens.len(), MAX_TOKENS as usize);
    assert_eq!(finished.redirected_at, Some(6));
}

/// The mock reports the redirect on the round that drew it, the leaf's lag:
/// the round before the one that emits the close.
#[test]
fn the_redirect_is_reported_by_the_round_that_drew_it() {
    let mock = MockCompute::new();
    mock.eos_after(7, 2);
    let job = DecodeJob {
        request: 7,
        lane: 0,
        params: thinking(MAX_TOKENS),
        remaining_tokens: 1,
        permitted: None,
        reasoning_close: Some(THINK_END),
    };
    let round = |mock: &MockCompute| mock.decode_step(std::slice::from_ref(&job)).expect("round").remove(0);
    let first = round(&mock);
    assert!(!first.reasoning_redirected, "the draw for step 1 is free");
    let second = round(&mock);
    assert!(second.reasoning_redirected, "the draw for step 2 was the EOS");
    assert_eq!(second.tokens, vec![mock.token_for(7, 1)]);
    let third = round(&mock);
    assert_eq!(third.tokens, vec![THINK_END]);
    assert!(!third.reasoning_redirected && third.finish.is_none());
}

/// A redirect the request never got to emit -- it reached its token cap
/// first -- is not reported.
#[test]
fn a_redirect_past_the_last_token_is_not_reported() {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 5);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, thinking(5));
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens.len(), 5);
    assert!(!finished.tokens.contains(&THINK_END));
    assert_eq!(finished.redirected_at, None);
}

// ── the budget's lag (AC 7) ─────────────────────────────────────────────

/// Budget 4: the round that emits token 4 starts the forced close, and token
/// 4 itself was drawn freely the round before. When that free draw is the
/// EOS it is redirected, and the redirected `</think>` is a natural close
/// inside the lag: the forcing stops, the one forced token already drawn
/// follows, and the request answers.
#[test]
fn an_eos_drawn_in_the_budgets_lag_is_redirected_and_stops_the_forcing() {
    let max_tokens = ANSWER_RESERVE + 40;
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 4);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(&mut sched, DecodeParams { thinking_budget: Some(4), ..thinking(max_tokens) });
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens[4], THINK_END, "{:?}", &finished.tokens[..8]);
    assert_eq!(finished.tokens[5], CLOSE[0], "the one forced draw already made");
    assert!(finished.tokens[6..].iter().all(|t| !CLOSE.contains(t)), "free after it");
    assert_eq!(finished.tokens.len(), max_tokens as usize);
    assert_eq!(finished.thinking, Some(BudgetOutcome { budget: 4, forced_at: None, closed_at: Some(4) }));
    assert_eq!(finished.redirected_at, Some(4));

    // Without the redirect the same draw ended the turn inside its reasoning.
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 4);
    let mut sched = scheduler(mock.clone(), true);
    let id = submit(
        &mut sched,
        DecodeParams { thinking_budget: Some(4), starts_in_reasoning: false, ..thinking(max_tokens) },
    );
    let finished = run(&mut sched, id);
    assert_eq!(finished.tokens.len(), 4);
    assert_eq!(finished.reason, FinishReason::Stop);
    assert_eq!(finished.thinking.map(|t| t.closed_at), Some(None));
}
