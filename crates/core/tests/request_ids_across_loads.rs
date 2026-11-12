//! Request ids across loads: every load builds a scheduler of its own, and a
//! scheduler counts its ids from 0 — so the log's `request_id` (ADR 0012)
//! named one request per load, not per process. Schedulers handed one
//! [`RequestIds`] continue one count; one handed none keeps its own, from 0,
//! as the tests' and the mocks' scripts expect.

use std::sync::Arc;

use ignis_core::{ConcreteScheduler, DecodeParams, MockCompute, RequestClass, RequestId, RequestIds, RequestInput, Scheduler, SchedulerConfig};

fn scheduler() -> ConcreteScheduler {
    ConcreteScheduler::with_config(SchedulerConfig { model: "m".into(), ..SchedulerConfig::default() }, Arc::new(MockCompute::new()))
}

fn input() -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: "m".into(),
        tokens: vec![1, 2, 3],
        params: DecodeParams { max_tokens: Some(1), ..Default::default() },
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

fn submit(scheduler: &mut ConcreteScheduler) -> RequestId {
    scheduler.submit(input(), RequestClass::Interactive).expect("admitted")
}

#[test]
fn schedulers_sharing_request_ids_continue_one_count() {
    let ids = RequestIds::default();
    let mut first = scheduler();
    first.share_request_ids(ids.clone());
    let earlier = [submit(&mut first), submit(&mut first)];
    drop(first);
    let mut second = scheduler();
    second.share_request_ids(ids);
    assert_eq!((earlier, submit(&mut second)), ([0, 1], 2));
    // An earlier load's id is none of this scheduler's: unknown, not reaped.
    assert_eq!(second.request_state(earlier[0]), None);
}

#[test]
fn a_scheduler_sharing_none_counts_its_own_from_zero() {
    let (mut first, mut second) = (scheduler(), scheduler());
    assert_eq!([submit(&mut first), submit(&mut first), submit(&mut second)], [0, 1, 0]);
}
