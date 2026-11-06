//! #518: a prompt's first token leaves with its prefill.
//!
//! The prefill's draw is the sequence's **pending token**. The scheduler
//! emits it at the end of the prefill phase of the step that ran the prompt's
//! last job, and the step hands what that phase emitted to its caller before
//! the decode round runs ([`Scheduler::advance_with`]). The round then takes
//! the token as its anchor and does not emit it again.

use std::sync::{Arc, Mutex};

use ignis_core::{
    Compute, ConcreteScheduler, DecodeParams, FinishReason, MockCompute, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId, N_DECODE_LANES,
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

fn input(max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: "m".into(),
        tokens: vec![1, 2, 3],
        params: DecodeParams {
            max_tokens: Some(max_tokens),
            ..DecodeParams::default()
        },
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

fn submit(sched: &mut ConcreteScheduler, max_tokens: u32) -> RequestId {
    sched.submit(input(max_tokens), RequestClass::Interactive).expect("submit")
}

fn chunks_of(events: &[SchedEvent], id: RequestId) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == id))
        .count()
}

fn tokens_of(events: &[SchedEvent], id: RequestId) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, SchedEvent::Token { request, .. } if *request == id))
        .count()
}

/// The prefill phase's events reach the caller while the step's decode round
/// has not run yet, and the round's own tokens come back with the rest.
#[test]
fn the_prefill_phase_events_leave_before_the_decode_round_runs() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone());
    let running = submit(&mut sched, 100);
    // Its prefill, then rounds until it holds a lane and decodes.
    while tokens_of(&sched.advance(), running) == 0 {}
    let fresh = submit(&mut sched, 100);
    let rounds_before = mock.decode_calls().len();

    let early: Mutex<Vec<(Vec<SchedEvent>, usize)>> = Mutex::new(Vec::new());
    let rest = sched.advance_with(&mut |events| early.lock().unwrap().push((events, mock.decode_calls().len())));

    let early = early.into_inner().unwrap();
    assert_eq!(early.len(), 1, "one hand-off, at the end of the prefill phase: {early:?}");
    let (prefilled, rounds_then) = &early[0];
    assert_eq!(chunks_of(prefilled, fresh), 1, "the fresh prompt's chunk is in it: {prefilled:?}");
    assert_eq!(*rounds_then, rounds_before, "handed over before the step's decode round ran");
    assert_eq!(mock.decode_calls().len(), rounds_before + 1, "the round ran after the hand-off");
    assert_eq!(chunks_of(&rest, fresh), 0, "nothing is handed over twice: {rest:?}");
    assert!(tokens_of(&rest, running) >= 1, "the round's tokens come back with the rest: {rest:?}");
}

/// [`Scheduler::advance`] is the same step with the hand-off folded in: the
/// prefill phase's events first, in the order they were emitted.
#[test]
fn advance_returns_the_prefill_phase_events_first() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone());
    let running = submit(&mut sched, 100);
    while tokens_of(&sched.advance(), running) == 0 {}
    let fresh = submit(&mut sched, 100);

    let events = sched.advance();

    let chunk = events
        .iter()
        .position(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == fresh))
        .expect("the fresh prompt's chunk");
    let token = events
        .iter()
        .position(|e| matches!(e, SchedEvent::Token { request, .. } if *request == running))
        .expect("the running request's token");
    assert!(chunk < token, "{events:?}");
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

fn done(events: &[SchedEvent], id: RequestId) -> Option<(u32, FinishReason)> {
    events.iter().find_map(|e| match e {
        SchedEvent::Done { request, tokens, reason, .. } if *request == id => Some((*tokens, *reason)),
        _ => None,
    })
}

fn admitted(events: &[SchedEvent], id: RequestId) -> bool {
    events.iter().any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == id))
}

/// One step, its prefill phase's hand-off kept apart from the rest.
fn step(sched: &mut ConcreteScheduler) -> (Vec<SchedEvent>, Vec<SchedEvent>) {
    let mut early = Vec::new();
    let rest = sched.advance_with(&mut |events| early.extend(events));
    (early, rest)
}

/// Every step until idle, each one's hand-off and rest in order.
fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<(Vec<SchedEvent>, Vec<SchedEvent>)> {
    let mut steps = Vec::new();
    for _ in 0..1000 {
        if sched.is_idle() {
            return steps;
        }
        steps.push(step(sched));
    }
    panic!("the scheduler never went idle");
}

/// The prompt's pending token is in the prefill phase's hand-off, alone,
/// before any decode round has run for the request.
#[test]
fn the_first_token_leaves_with_the_prefill() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 10);

    let (early, _) = step(&mut sched);

    assert_eq!(emitted(&early, id), vec![mock.token_for(id, 0)], "{early:?}");
    let chunk = early.iter().position(|e| matches!(e, SchedEvent::PrefillChunk { .. })).unwrap();
    let token = early.iter().position(|e| matches!(e, SchedEvent::Token { .. })).unwrap();
    assert!(chunk < token, "the chunk, then the token it drew: {early:?}");
}

/// The streaming order, on a plain load and on a speculative one: the first
/// hand-off holds one token, each round its run after it, every token once
/// and in order. The first round takes the first token as its anchor and
/// does not emit it again, so its run is one token short.
#[test]
fn the_stream_is_the_first_token_then_each_rounds_run() {
    for runs in [&[][..], &[3][..]] {
        let mock = Arc::new(MockCompute::with_runs(runs));
        let mut sched = scheduler(mock.clone());
        let id = submit(&mut sched, 20);

        let steps = run_to_idle(&mut sched);

        let all: Vec<SchedEvent> = steps.iter().flat_map(|(e, r)| e.iter().chain(r).cloned()).collect();
        let expected: Vec<TokenId> = (0..20).map(|step| mock.token_for(id, step)).collect();
        assert_eq!(emitted(&all, id), expected, "runs {runs:?}: every token once, in order");
        assert_eq!(done(&all, id), Some((20, FinishReason::Length)), "runs {runs:?}");
        let (first_early, first_rest) = &steps[0];
        assert_eq!(emitted(first_early, id).len(), 1, "runs {runs:?}: the first token alone");
        let first_round = match runs {
            [] => 0,
            [k_plus_one] => *k_plus_one as usize - 1,
            _ => unreachable!(),
        };
        assert_eq!(emitted(first_rest, id).len(), first_round, "runs {runs:?}: the first round's run, its anchor left out");
    }
}

/// `max_tokens` 1 ends with the prefill: no decode round, no lane.
#[test]
fn max_tokens_one_ends_with_no_decode_round() {
    let mock = Arc::new(MockCompute::with_runs(&[3]));
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 1);

    let (early, rest) = step(&mut sched);

    assert_eq!(emitted(&early, id), vec![mock.token_for(id, 0)]);
    assert_eq!(done(&early, id), Some((1, FinishReason::Length)), "{early:?}");
    assert!(!admitted(&rest, id), "no lane dealt: {rest:?}");
    assert!(sched.is_idle());
    assert!(mock.decode_calls().is_empty(), "no decode round, no drafter proposal");
}

/// An EOS drawn by the prefill ends the request there, emitted as no token.
#[test]
fn an_eos_as_the_first_token_ends_the_request() {
    let mock = Arc::new(MockCompute::with_runs(&[3]));
    let mut sched = scheduler(mock.clone());
    let id = submit(&mut sched, 10);
    mock.eos_after(id, 0);

    let (early, rest) = step(&mut sched);

    assert!(emitted(&early, id).is_empty(), "{early:?}");
    assert_eq!(done(&early, id), Some((0, FinishReason::Stop)), "{early:?}");
    assert!(!admitted(&rest, id));
    assert!(sched.is_idle());
    assert!(mock.decode_calls().is_empty());
}

/// A first token leaves even when every lane is taken, and the request's
/// second token comes once a lane is dealt, with nothing emitted twice.
#[test]
fn a_first_token_leaves_with_no_free_lane_and_the_second_follows_the_lane() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "m".into(),
            max_in_flight: N_DECODE_LANES + 1,
            resident_slot_capacity: N_DECODE_LANES as u32 + 1,
            kv_capacity_pages: ((N_DECODE_LANES + 1) * (8192 / 16)) as u32,
            // No overflow to the host: a waiting request waits for a lane
            // to free rather than taking one through an eviction.
            host_capacity_bytes: 0,
            ..SchedulerConfig::default()
        },
        mock.clone(),
    );
    let holders: Vec<RequestId> = (0..N_DECODE_LANES).map(|_| submit(&mut sched, 30)).collect();
    let mut events = Vec::new();
    while holders.iter().any(|&h| !admitted(&events, h)) {
        events.extend(sched.advance());
    }
    let late = submit(&mut sched, 5);

    let (early, rest) = step(&mut sched);

    assert_eq!(emitted(&early, late), vec![mock.token_for(late, 0)], "{early:?}");
    assert!(!admitted(&rest, late), "every lane is taken: {rest:?}");
    let mut all = early;
    all.extend(rest);
    for (e, r) in run_to_idle(&mut sched) {
        all.extend(e);
        all.extend(r);
    }
    let expected: Vec<TokenId> = (0..5).map(|step| mock.token_for(late, step)).collect();
    assert_eq!(emitted(&all, late), expected, "once each, in order: {all:?}");
    let lane_dealt = all.iter().position(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == late));
    let second = all.iter().position(|e| matches!(e, SchedEvent::Token { request, .. } if *request == late)).map(|first| {
        all[first + 1..]
            .iter()
            .position(|e| matches!(e, SchedEvent::Token { request, .. } if *request == late))
            .map(|n| first + 1 + n)
    });
    assert!(lane_dealt.unwrap() < second.flatten().unwrap(), "the second token follows the lane deal");
}
