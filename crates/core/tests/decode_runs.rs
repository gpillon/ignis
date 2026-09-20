//! P5-06 (GitHub #154): a decode round commits a *run* of tokens per lane.
//!
//! `Compute::decode_step` returns the committed tokens of the round in
//! order, 1..=k+1 of them under speculation, and the scheduler emits each as
//! its own `SchedEvent::Token`, stops the request on the right reason, and
//! never emits a token past the request's budget. The default mock still
//! commits runs of one, so every other scheduler test is today's round.

use std::sync::Arc;

use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, DecodeParams,
    FinishReason, MockCompute, PrefillJob, PrefillOutcome, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, SpecCounters, TokenId,
};

fn scheduler(compute: Arc<dyn Compute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "m".into(),
            ..SchedulerConfig::default()
        },
        compute,
    )
}

fn submit(sched: &mut ConcreteScheduler, max_tokens: u32) -> RequestId {
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
                    ..DecodeParams::default()
                },
            },
            RequestClass::Interactive,
        )
        .expect("submit")
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    for _ in 0..1000 {
        if sched.is_idle() {
            return events;
        }
        events.extend(sched.advance());
    }
    panic!("the scheduler never went idle: {events:?}");
}

fn emitted(events: &[SchedEvent], id: RequestId) -> Vec<TokenId> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Token { request, token } if *request == id => Some(*token),
            _ => None,
        })
        .collect()
}

fn done(events: &[SchedEvent], id: RequestId) -> (u32, FinishReason, Option<SpecCounters>) {
    events
        .iter()
        .find_map(|e| match e {
            SchedEvent::Done {
                request,
                tokens,
                reason,
                spec,
                ..
            } if *request == id => Some((*tokens, *reason, *spec)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("request {id} never finished: {events:?}"))
}

#[test]
fn runs_of_every_length_up_to_k_plus_one_emit_the_committed_tokens_in_order() {
    // k = 7: runs of 1..=8, cycling. 1+2+..+7 = 28, so the eighth round's run
    // of 8 is cut at the 30-token cap after two tokens.
    let mock = Arc::new(MockCompute::with_runs(&[1, 2, 3, 4, 5, 6, 7, 8]));
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 30);

    let events = run_to_idle(&mut sched);

    let expected: Vec<TokenId> = (0..30).map(|step| mock.token_for(id, step)).collect();
    assert_eq!(emitted(&events, id), expected, "every committed token, once, in order");
    assert_eq!(done(&events, id).0, 30);
    assert_eq!(done(&events, id).1, FinishReason::Length);
    assert_eq!(mock.decode_calls().len(), 8, "eight rounds, not thirty");
}

#[test]
fn a_run_ending_in_eos_finishes_with_stop_and_emits_the_tokens_before_it() {
    // Runs of 4; the seventh token (step 6) is EOS. Round one commits
    // steps 0..4, round two commits 4, 5 and the EOS -- which is never
    // emitted, as today.
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 100);
    mock.eos_after(id, 6);

    let events = run_to_idle(&mut sched);

    let expected: Vec<TokenId> = (0..6).map(|step| mock.token_for(id, step)).collect();
    assert_eq!(emitted(&events, id), expected);
    let (tokens, reason, _) = done(&events, id);
    assert_eq!((tokens, reason), (6, FinishReason::Stop));
    assert_eq!(mock.decode_calls().len(), 2, "the EOS ends the request in its own round");
}

#[test]
fn a_run_that_reaches_max_tokens_mid_run_finishes_with_length() {
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 6);

    let events = run_to_idle(&mut sched);

    let expected: Vec<TokenId> = (0..6).map(|step| mock.token_for(id, step)).collect();
    assert_eq!(emitted(&events, id), expected);
    let (tokens, reason, _) = done(&events, id);
    assert_eq!((tokens, reason), (6, FinishReason::Length));
    assert_eq!(mock.decode_calls().len(), 2, "the capped run ends the request at once");
}

/// A backend that commits eight tokens every round whatever budget it is
/// handed -- the case the scheduler's own reservation cap exists for.
struct OvershootingCompute;

impl Compute for OvershootingCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        Ok(PrefillOutcome::nothing_encoded(jobs.len()))
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        Ok(jobs
            .iter()
            .map(|_| DecodeOutcome::run((100..108).collect()))
            .collect())
    }
}

#[test]
fn a_run_past_the_budget_is_cut_at_the_budget_and_finishes_with_length() {
    let mut sched = scheduler(Arc::new(OvershootingCompute));
    let id = submit(&mut sched, 5);

    let events = run_to_idle(&mut sched);

    assert_eq!(emitted(&events, id), vec![100, 101, 102, 103, 104]);
    let (tokens, reason, _) = done(&events, id);
    assert_eq!((tokens, reason), (5, FinishReason::Length));
}

#[test]
fn every_decode_job_carries_the_requests_remaining_budget() {
    // The leaf clamps a lane's extent to this budget, so a sequence never
    // commits past the text its request may emit.
    let mock = Arc::new(MockCompute::with_runs(&[3]));
    let mut sched = scheduler(mock.clone());
    submit(&mut sched, 7);

    run_to_idle(&mut sched);

    let budgets: Vec<u32> = mock
        .decode_calls()
        .iter()
        .map(|batch| batch[0].remaining_tokens)
        .collect();
    assert_eq!(budgets, vec![7, 4, 1]);
}

#[test]
fn speculative_rounds_add_up_on_the_done_event() {
    // Rounds of 3, 1, 3: the mock drafts len - 1 and accepts committed - 1.
    // The third round is cut by the 5-token cap to one committed token.
    let mock = Arc::new(MockCompute::with_runs(&[3, 1]));
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 5);

    let events = run_to_idle(&mut sched);

    assert_eq!(
        done(&events, id).2,
        Some(SpecCounters {
            rounds: 3,
            drafted: 4,
            accepted: 2,
            // GitHub #160: the first and third rounds proposed two drafts
            // each; only the first committed them.
            drafted_at: [2, 2, 0, 0, 0, 0, 0],
            accepted_at: [1, 1, 0, 0, 0, 0, 0],
        })
    );
}

#[test]
fn a_round_without_speculation_reports_no_counters() {
    let mut sched = scheduler(Arc::new(MockCompute::new()));
    let id = submit(&mut sched, 3);

    let events = run_to_idle(&mut sched);

    assert_eq!(done(&events, id), (3, FinishReason::Length, None));
}
