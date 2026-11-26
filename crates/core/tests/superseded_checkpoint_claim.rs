//! GitHub #598 — identical requests that reach the scheduler in one pass are
//! all served, when the checkpoint they claimed together is superseded before
//! the later ones are built from it.
//!
//! The shape the issue's burst makes: a conversation's turn left a prompt
//! checkpoint; its next turn arrives twice (or three times) at once. Both
//! claim that checkpoint in the same pass, but the leader's tail is wider than
//! a chunk, so it prefills alone and its twin waits with its claim. The
//! leader's capture opens a turn of the same conversation and supersedes the
//! checkpoint (GitHub #187), whose device image the backend then lets go: the
//! twin's first job, built from that image, failed three times and ended with
//! `FinishReason::Error`. A waiting claim on a checkpoint that goes is given
//! up instead, and its request looks again: it finds the leader's capture.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` that,
//! as the real adapter does, fails a job claiming a device image it no longer
//! holds.

use std::sync::Arc;

use ignis_core::checkpoint::ReuseSource;
use ignis_core::types::{DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A turn whose generation opener ends `opener` tokens in, and whose last
/// real user query ends `user_turn` tokens in.
fn turn(prompt: Vec<u32>, opener: u32, user_turn: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(4),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: Some(opener),
        user_turn_tokens: user_turn,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

/// The first turn: 40 tokens, its opener at 37. It leaves a checkpoint there.
fn first_turn() -> RequestInput {
    turn(tokens(1, 40), 37, None)
}

/// The next turn: the first up to its opener, then a reply and a new user
/// message (the user's query ends at 60, past the checkpoint, so its own
/// capture opens a turn and supersedes the first turn's), its opener at 97.
fn next_turn() -> RequestInput {
    turn([tokens(1, 37), tokens(500, 63)].concat(), 97, Some(60))
}

/// A scheduler whose serving chunk is `chunk` tokens.
fn scheduler(compute: Arc<MockCompute>, chunk: u32) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            serving_chunk_tokens: chunk,
            ..SchedulerConfig::default()
        },
        compute,
    )
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// The prompt tokens `request` resumed from a device checkpoint, if it did.
fn resumed_from(events: &[SchedEvent], request: RequestId) -> Option<u32> {
    events.iter().find_map(|e| match e {
        SchedEvent::StateReused { request: r, source: ReuseSource::Device, tokens, .. } if *r == request => {
            Some(*tokens)
        }
        _ => None,
    })
}

/// Every request ended without an error, and with a token.
fn all_served(events: &[SchedEvent], requests: &[RequestId]) {
    for &request in requests {
        let done = events.iter().find_map(|e| match e {
            SchedEvent::Done { request: r, reason, tokens, .. } if *r == request => Some((*reason, *tokens)),
            _ => None,
        });
        let (reason, tokens) = done.unwrap_or_else(|| panic!("request {request} never finished: {events:?}"));
        assert_ne!(reason, FinishReason::Error, "request {request} ended with an engine error");
        assert!(tokens > 0, "request {request} was answered with nothing");
    }
}

/// `twins` identical next turns submitted at once, after the first turn
/// finished, with a chunk of `chunk` tokens.
fn burst(twins: usize, chunk: u32) -> (Vec<RequestId>, Vec<SchedEvent>) {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute, chunk);
    sched.submit(first_turn(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let ids: Vec<RequestId> =
        (0..twins).map(|_| sched.submit(next_turn(), RequestClass::Interactive).unwrap()).collect();
    let events = run_to_idle(&mut sched);
    (ids, events)
}

#[test]
fn a_twin_waiting_on_a_superseded_checkpoint_resumes_from_the_leaders_capture() {
    // A 32-token chunk: the leader's 63-token tail takes two, and its twin is
    // cut from the batch the two claimed in.
    let (ids, events) = burst(2, 32);
    all_served(&events, &ids);
    assert_eq!(resumed_from(&events, ids[0]), Some(37), "the leader resumes from the first turn");
    assert_eq!(resumed_from(&events, ids[1]), Some(97), "its twin resumes from the leader's capture");
}

#[test]
fn two_twins_waiting_on_one_superseded_checkpoint_both_resume_from_the_leaders_capture() {
    let (ids, events) = burst(3, 32);
    all_served(&events, &ids);
    assert_eq!(resumed_from(&events, ids[0]), Some(37));
    for &twin in &ids[1..] {
        assert_eq!(resumed_from(&events, twin), Some(97), "twin {twin} resumes from the leader's capture");
    }
}

/// The control: a chunk wide enough for both tails, so the twins prefill in
/// one batch and both are built from the first turn's checkpoint before the
/// leader's capture supersedes it. The twin's claim has landed, so nothing is
/// given up.
#[test]
fn twins_built_in_one_batch_both_resume_from_the_first_turn() {
    let (ids, events) = burst(2, 64);
    all_served(&events, &ids);
    for &id in &ids {
        assert_eq!(resumed_from(&events, id), Some(37), "request {id} resumes from the first turn");
    }
}
