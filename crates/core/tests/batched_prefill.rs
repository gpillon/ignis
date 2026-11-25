//! core-04 — batched (concurrent) prefill: queued requests are grouped
//! into ONE compute call per step (not one call per request), and decode is
//! batched across all resident lanes.
//!
//! Seams (ADR 0006): the `Scheduler` trait surface driven with a
//! `MockCompute` that records the call shape behind the `Compute` seam.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, N_DECODE_LANES, RequestClass, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

fn input(model: &str, tokens: &[u32], max_tokens: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: model.into(),
        tokens: tokens.to_vec(),
        params: DecodeParams {
            max_tokens,
            ..DecodeParams::default()
        },
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

#[test]
fn warm_short_prefill_uses_retained_device_history_and_preserves_long_offset() {
    let model = "qwen3.8-27b";
    for (enabled, matching, tail, borrows) in [
        (false, true, 14, false),
        (true, false, 14, false),
        (true, true, 96, false),
        (true, true, 14, true),
    ] {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::with_config(
            SchedulerConfig {
                model: model.into(), serving_chunk_tokens: 32, kv_page_tokens: 16,
                resident_slot_capacity: 4, prompt_reuse: true,
                read_continuation_priority: enabled,
                ..SchedulerConfig::default()
            },
            compute.clone(),
        );
        let head: Vec<u32> = (100..228).collect();
        let mut first = input(model, &[head.as_slice(), &[5; 5]].concat(), Some(1));
        first.system_block_tokens = Some(128);
        first.opener_tokens = Some(133);
        sched.submit(first, RequestClass::Interactive).unwrap();
        for _ in 0..30 {
            if sched.is_idle() { break; }
            sched.advance();
        }
        assert!(sched.is_idle(), "warm fixture did not finish");
        let long = sched.submit(input(model, &[9; 256], Some(1)), RequestClass::Agent).unwrap();
        sched.advance();
        let mut next = [head.as_slice(), &vec![6; tail]].concat();
        if !matching { next[0] = 37; }
        let short = sched.submit(input(model, &next, Some(1)), RequestClass::Interactive).unwrap();
        sched.advance();
        sched.advance();
        let jobs: Vec<_> = compute.prefill_calls().iter().flatten()
            .filter(|job| job.request == long || job.request == short)
            .map(|job| (job.request, job.start_position, job.tokens.len())).collect();
        if borrows {
            assert_eq!(jobs, [(long, 0, 32), (short, 128, 14), (long, 32, 32)]);
        } else {
            assert_eq!(jobs, [(long, 0, 32), (long, 32, 32), (long, 64, 32)]);
        }
    }
}

fn short_prefill_case(
    enabled: bool,
    slots: u32,
    fail_short: bool,
) -> (Vec<(u64, u32, usize)>, u64, u64, u64) {
    let model = "qwen3.8-27b";
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: model.into(),
            serving_chunk_tokens: 32,
            resident_slot_capacity: slots,
            read_continuation_priority: enabled,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let long = sched.submit(input(model, &[9; 256], Some(2)), RequestClass::Agent).unwrap();
    sched.advance();
    let short = sched.submit(input(model, &[5; 18], Some(2)), RequestClass::Interactive).unwrap();
    let next = sched.submit(input(model, &[6; 18], Some(2)), RequestClass::Interactive).unwrap();
    if fail_short {
        compute.fail_prefill(short);
    }
    sched.advance();
    if fail_short {
        assert!(sched.last_error().is_some(), "the borrowed prefill must reach the fault");
    }
    for _ in 0..2 {
        sched.advance();
    }
    let jobs = compute.prefill_calls().iter().flatten()
        .map(|job| (job.request, job.start_position, job.tokens.len())).collect();
    (jobs, long, short, next)
}

#[test]
fn cold_short_prefill_borrows_one_tick_and_long_resumes_at_its_exact_offset() {
    let (jobs, long, short, next) = short_prefill_case(true, 4, false);
    assert_eq!(jobs, [(long, 0, 32), (short, 0, 18), (long, 32, 32), (long, 64, 32)]);
    assert!(!jobs.iter().any(|job| job.0 == next));
}

#[test]
fn short_prefill_is_opt_in_and_never_evicts_to_make_room() {
    for (enabled, slots) in [(false, 4), (true, 1)] {
        let (jobs, long, _, _) = short_prefill_case(enabled, slots, false);
        assert_eq!(jobs, [(long, 0, 32), (long, 32, 32), (long, 64, 32), (long, 96, 32)]);
    }
}

#[test]
fn a_short_prefill_fault_does_not_rewind_the_parked_sequence() {
    let (jobs, long, _, _) = short_prefill_case(true, 4, true);
    // MockCompute reports the failed call through last_error, not its
    // successful-prefill trace.
    assert_eq!(jobs, [(long, 0, 32), (long, 32, 32), (long, 64, 32)]);
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
                    decision: None,
                    multimodal: None,
                    opener_tokens: None,
                    user_turn_tokens: None,
                    system_block_tokens: None,
                    reuse_boundaries: Vec::new(),
                    model: "qwen3.8-27b".into(),
                    tokens: vec![1, 2],
                    params,
                    constrained: None,
                    forced_literal: None,
                    grammar: None,
                    warm_up: false,
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
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                reuse_boundaries: Vec::new(),
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2, 3],
                params: mk(7),
                constrained: None,
                forced_literal: None,
                grammar: None,
                warm_up: false,
            },
            RequestClass::Agent,
        )
        .unwrap();
    let r8 = sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                reuse_boundaries: Vec::new(),
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2, 3],
                params: mk(8),
                constrained: None,
                forced_literal: None,
                grammar: None,
                warm_up: false,
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
