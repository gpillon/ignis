//! #518: a prompt's first token leaves with its prefill.
//!
//! The prefill's draw is the sequence's **pending token**. The scheduler
//! emits it at the end of the prefill phase of the step that ran the prompt's
//! last job, and the step hands what that phase emitted to its caller before
//! the decode round runs ([`Scheduler::advance_with`]). The round then takes
//! the token as its anchor and does not emit it again.

use std::sync::{Arc, Mutex};

use ignis_core::{
    Compute, ConcreteScheduler, DecodeParams, MockCompute, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig,
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
