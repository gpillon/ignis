//! core-04/05 boundary — lane assignment respects class priority
//! (Interactive before Agent) then FIFO within a class; unknown models are
//! rejected at submit.
//!
//! The full admission state machine (protection, backfill, temporal credit,
//! frontier distance) lands in core-05; this pins the *basic* policy the
//! concrete scheduler must honor: class priority + FIFO.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent, SubmitError};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

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
fn lanes_are_dealt_by_class_priority_then_fifo() {
    // 8 filler lanes: 6 long (stay Running), 2 short (free 2 lanes early).
    // 6 queued behind them: 2 Agents first, then 4 Interactives. The first
    // two freed lanes must go to Interactives (class priority beats the
    // FIFO order in which the Agents were queued); only once all
    // Interactives are seated do the Agents take the last lanes (FIFO
    // within a class).
    let compute = Arc::new(MockCompute::new());
    let cfg = SchedulerConfig {
        model: "qwen3.8-27b".into(),
        max_in_flight: 14, // 8 filler lanes + 6 queued (pre-host-tier knob)
        max_prefill_batch: 8,
    };
    let mut sched = ConcreteScheduler::with_config(cfg, compute.clone());

    for _ in 0..6 {
        sched
            .submit(input("qwen3.8-27b", &[1], Some(20)), RequestClass::Agent)
            .unwrap();
    }
    for _ in 0..2 {
        sched
            .submit(input("qwen3.8-27b", &[1], Some(2)), RequestClass::Agent)
            .unwrap();
    }
    sched.advance(); // prefills the 8 fillers; they take all 8 lanes

    let a1 = sched
        .submit(input("qwen3.8-27b", &[2], Some(1)), RequestClass::Agent)
        .unwrap();
    let a2 = sched
        .submit(input("qwen3.8-27b", &[2], Some(1)), RequestClass::Agent)
        .unwrap();
    let i1 = sched
        .submit(
            input("qwen3.8-27b", &[3], Some(1)),
            RequestClass::Interactive,
        )
        .unwrap();
    let i2 = sched
        .submit(
            input("qwen3.8-27b", &[3], Some(1)),
            RequestClass::Interactive,
        )
        .unwrap();
    let i3 = sched
        .submit(
            input("qwen3.8-27b", &[3], Some(1)),
            RequestClass::Interactive,
        )
        .unwrap();
    let i4 = sched
        .submit(
            input("qwen3.8-27b", &[3], Some(1)),
            RequestClass::Interactive,
        )
        .unwrap();

    let mut evs = Vec::new();
    while !sched.is_idle() {
        evs.extend(sched.advance());
    }

    let admitted: Vec<u64> = evs
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Admitted { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert_eq!(
        admitted,
        vec![i1, i2, i3, i4, a1, a2],
        "Interactives are seated before the earlier-queued Agents; FIFO within a class"
    );

    // The 6 queued prefills went to the backend in ONE batch.
    let calls = compute.prefill_calls();
    assert_eq!(calls.len(), 2, "fillers then queued");
    assert_eq!(calls[1].len(), 6, "the 6 queued prefills are one batch");
}

#[test]
fn submit_rejects_unknown_model() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
    let err = sched
        .submit(input("some-other-model", &[1], None), RequestClass::Agent)
        .unwrap_err();
    assert_eq!(err, SubmitError::UnknownModel("some-other-model".into()));
}
