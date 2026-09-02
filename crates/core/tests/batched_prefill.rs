//! core-04 — batched (concurrent) prefill: queued requests are grouped
//! into ONE compute call per step (not one call per request), and decode is
//! batched across all resident lanes.
//!
//! Seams (ADR 0006): the `Scheduler` trait surface driven with a
//! `MockCompute` that records the call shape behind the `Compute` seam.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, N_DECODE_LANES, RequestClass, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler};

fn input(model: &str, tokens: &[u32], max_tokens: Option<u32>) -> RequestInput {
    RequestInput {
        model: model.into(),
        tokens: tokens.to_vec(),
        params: DecodeParams {
            max_tokens,
            ..DecodeParams::default()
        },
    }
}

#[test]
fn prefill_is_grouped_into_one_batch_per_step() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
    for _ in 0..4 {
        sched
            .submit(
                input("qwen3.8-27b", &[1, 2, 3], Some(2)),
                RequestClass::Agent,
            )
            .unwrap();
    }
    sched.advance();

    // The 4 queued prefills went to the backend as ONE batched call with 4
    // jobs — not 4 separate calls (the whole point of batched prefill).
    let calls = compute.prefill_calls();
    assert_eq!(calls.len(), 1, "expected a single batched prefill call");
    assert_eq!(calls[0].len(), 4);

    // Decode is batched the same way: one call spanning every running lane.
    let dec = compute.decode_calls();
    assert_eq!(dec.len(), 1);
    assert_eq!(dec[0].len(), 4);
}

#[test]
fn decode_steps_are_batched_across_all_lanes() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
    for _ in 0..N_DECODE_LANES {
        sched
            .submit(input("qwen3.8-27b", &[1], Some(2)), RequestClass::Agent)
            .unwrap();
    }
    for _ in 0..2 {
        sched.advance();
    }
    // Each advance issued exactly one decode call spanning all 8 lanes.
    let calls = compute.decode_calls();
    assert_eq!(calls.len(), 2);
    assert!(
        calls.iter().all(|c| c.len() == N_DECODE_LANES),
        "every decode batch must span all 8 lanes"
    );
}

#[test]
fn token_stream_is_deterministic_and_seed_sensitive() {
    // The token stream of a request is a pure function of (mock seed,
    // request seed): same inputs → same stream (determinism, the
    // self-consistency property the 99% gate relies on, ADR 0007), and a
    // different request seed → a different stream (the mock is not a
    // constant).
    fn stream_for(request_seed: u64) -> Vec<u32> {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
        let mut params = DecodeParams {
            max_tokens: Some(3),
            ..DecodeParams::default()
        };
        params.seed = request_seed;
        let id = sched
            .submit(
                RequestInput {
                    model: "qwen3.8-27b".into(),
                    tokens: vec![1, 2],
                    params,
                },
                RequestClass::Agent,
            )
            .unwrap();
        let mut out = Vec::new();
        while !sched.is_idle() {
            for e in sched.advance() {
                if let SchedEvent::Token { request, token } = e {
                    assert_eq!(request, id);
                    out.push(token);
                }
            }
        }
        out
    }
    assert_eq!(stream_for(7), stream_for(7), "same seed → same stream");
    assert_ne!(
        stream_for(7),
        stream_for(8),
        "streams must differ across seeds"
    );
}

#[test]
fn token_stream_depends_on_the_request_seed() {
    // The mock must not be a constant: a different request seed generates a
    // different token stream.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
    let mk = |seed: u64| DecodeParams {
        max_tokens: Some(3),
        seed,
        ..DecodeParams::default()
    };
    let r7 = sched
        .submit(
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2, 3],
                params: mk(7),
            },
            RequestClass::Agent,
        )
        .unwrap();
    let r8 = sched
        .submit(
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2, 3],
                params: mk(8),
            },
            RequestClass::Agent,
        )
        .unwrap();
    let mut evs = Vec::new();
    while !sched.is_idle() {
        evs.extend(sched.advance());
    }
    let stream = |id: u64| {
        evs.iter()
            .filter_map(|e| match e {
                SchedEvent::Token { request, token } if *request == id => Some(*token),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(stream(r7).len(), 3);
    assert_eq!(stream(r8).len(), 3);
    assert_ne!(stream(r7), stream(r8), "streams must differ across seeds");
}
