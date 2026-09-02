//! core-04 — the concrete scheduler holds N=8 resident decode lanes.
//!
//! Seams (ADR 0006): the `Scheduler` trait surface (submit → advance →
//! events), driven with a `MockCompute` behind the `Compute` seam so the
//! whole test runs on a CPU.

use std::collections::BTreeSet;
use std::sync::Arc;

use ignis_core::types::{
    DecodeParams, EngineMode, LaneId, N_DECODE_LANES, RequestClass, RequestInput, SchedEvent,
    SubmitError,
};
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
fn holds_eight_resident_lanes_and_rejects_the_ninth() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());

    for _ in 0..N_DECODE_LANES {
        assert!(
            sched
                .submit(
                    input("qwen3.8-27b", &[1, 2, 3], Some(1)),
                    RequestClass::Agent
                )
                .is_ok()
        );
    }
    // The 9th in-flight request is rejected (no host-tier overflow yet —
    // that lands in core-06).
    assert!(matches!(
        sched.submit(input("qwen3.8-27b", &[9], Some(1)), RequestClass::Agent),
        Err(SubmitError::Full)
    ));

    // One step: the 8 queued prefills are dealt onto 8 distinct resident
    // lanes.
    let ev = sched.advance();
    let lanes: BTreeSet<LaneId> = ev
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Admitted { lane, .. } => Some(*lane),
            _ => None,
        })
        .collect();
    assert_eq!(lanes, (0..N_DECODE_LANES).collect::<BTreeSet<_>>());
    assert!(!sched.is_idle());

    // Next step: every request reached max_tokens → all Done, lanes released.
    let ev = sched.advance();
    assert_eq!(
        ev.iter()
            .filter(|e| matches!(e, SchedEvent::Done { .. }))
            .count(),
        N_DECODE_LANES
    );
    assert!(sched.is_idle());
}

#[test]
fn lanes_are_reused_after_requests_finish() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
    for _ in 0..N_DECODE_LANES {
        sched
            .submit(input("qwen3.8-27b", &[1], Some(1)), RequestClass::Agent)
            .unwrap();
    }
    sched.advance();
    sched.advance(); // all Done
    assert!(sched.is_idle());
    // The freed lanes accept new requests (capacity released, not stuck).
    for _ in 0..N_DECODE_LANES {
        assert!(
            sched
                .submit(input("qwen3.8-27b", &[2], Some(1)), RequestClass::Agent)
                .is_ok()
        );
    }
}

#[test]
fn idle_and_mode_track_in_flight_work() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
    assert!(sched.is_idle());
    assert_eq!(sched.mode(), EngineMode::Idle);

    sched
        .submit(input("qwen3.8-27b", &[1], Some(1)), RequestClass::Agent)
        .unwrap();
    assert!(!sched.is_idle());
    assert_eq!(sched.mode(), EngineMode::Serving);

    sched.advance();
    sched.advance();
    assert!(sched.is_idle());
}
