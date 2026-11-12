//! GitHub #506 (spec #503, "Drafter taps") — the **drafter floor** each
//! prefill job carries: the earliest position a later drafter window reads,
//! from the job's own retained-image points.
//!
//! A prefill job taps the DFlash2 drafter's window from `max(the span rule,
//! floor)`, where the span rule is the job's last 2048 positions and the floor
//! is the earliest retained-image point at or past the job's end, minus the
//! drafter window (2048). The points are the ones the request's own prefill
//! can leave an image at — the prefix it publishes at its opener's page floor
//! (or at its head), any reuse boundary, its prompt checkpoint at the
//! generation opener — and the prompt's end, where decode reads the window. A
//! job whose span lies wholly below its floor taps nothing.
//!
//! What these tests can prove: which floor the scheduler hands each job, in
//! both shapes a load cuts a prompt in — the 27B's two cuts (a prefix at the
//! opener's page floor, the checkpoint at the opener) and the single cut where
//! the opener's page rides the capture (`opener_page_rides_capture`, as
//! Flash-Next does and as #508 makes the 27B do on Ampere and Ada). That the
//! window then holds the right rows needs the model and lives in
//! `drafter_floor_gpu.rs`.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` that
//! records every prefill job behind the `Compute` seam, as
//! `prompt_checkpoint.rs` does.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request of `prompt`, its generation opener `opener` tokens in and its
/// system block (a reuse boundary) `block` tokens in.
fn input(prompt: Vec<u32>, block: Option<u32>, opener: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(4),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: opener,
        user_turn_tokens: None,
        system_block_tokens: block,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

/// The default scheduler (16-token pages, 1024-token chunks, 8192-token
/// sequences), with or without the opener's page riding the capture.
fn config(opener_page_rides_capture: bool) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        opener_page_rides_capture,
        ..SchedulerConfig::default()
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) {
    while !sched.is_idle() {
        sched.advance();
    }
}

/// Every prefill job dealt to `request`, as (start, tokens, drafter floor).
fn jobs(compute: &MockCompute, request: RequestId) -> Vec<(u32, usize, u32)> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .map(|j| (j.start_position, j.tokens.len(), j.drafter_floor))
        .collect()
}

/// The 27B's two cuts: a 6000-token prompt whose opener ends at 5000 publishes
/// its head at the opener's page floor (4992) and captures its checkpoint at
/// the opener. Every job before the publish reads its floor from it, 4992 -
/// 2048; the opener's own short chunk from the checkpoint, 5000 - 2048; and the
/// last from the prompt's end, 6000 - 2048. The first two chunks lie wholly
/// below their floor.
#[test]
fn two_cuts_give_each_job_the_floor_of_the_next_image_ahead_of_it() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(false), compute.clone());
    let request = sched.submit(input(tokens(1, 6000), None, Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![
            (0, 1024, 2944),
            (1024, 1024, 2944),
            (2048, 1024, 2944),
            (3072, 1024, 2944),
            (4096, 896, 2944),
            (4992, 8, 2952),
            (5000, 1000, 3952),
        ]
    );
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "the checkpoint was captured at the opener");
}

/// The single cut: with the opener's page riding the capture the request
/// publishes nothing at its floor, so every job before the opener reads its
/// floor from the checkpoint, 5000 - 2048 — never from a cut it does not make.
#[test]
fn a_single_cut_gives_each_job_the_floor_of_the_checkpoint() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(true), compute.clone());
    let request = sched.submit(input(tokens(1, 6000), None, Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![
            (0, 1024, 2952),
            (1024, 1024, 2952),
            (2048, 1024, 2952),
            (3072, 1024, 2952),
            (4096, 904, 2952),
            (5000, 1000, 3952),
        ]
    );
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "the checkpoint was captured at the opener");
}

/// A reuse boundary is an image point too: a system block ending at 3000
/// publishes a retained prefix at its page floor, 2992, so the jobs before it
/// read their floor from there, 2992 - 2048 = 944, and the first chunk taps
/// its last 80 positions.
#[test]
fn a_reuse_boundary_ahead_sets_the_floor_of_the_jobs_before_it() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(false), compute.clone());
    let request =
        sched.submit(input(tokens(1, 6000), Some(3000), Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![
            (0, 1024, 944),
            (1024, 1024, 944),
            (2048, 944, 944),
            (2992, 1024, 2944),
            (4016, 976, 2944),
            (4992, 8, 2952),
            (5000, 1000, 3952),
        ]
    );
}

/// A prompt with no opener still publishes its head as a sibling's prefix, at
/// its page floor (6000 of 6010), and decode reads the window at its end: the
/// jobs before the head read their floor from it, 6000 - 2048, and the last
/// from the prompt's end, 6010 - 2048.
#[test]
fn without_an_opener_the_floor_comes_from_the_head_and_the_prompt_end() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(false), compute.clone());
    let request = sched.submit(input(tokens(1, 6010), None, None), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![
            (0, 1024, 3952),
            (1024, 1024, 3952),
            (2048, 1024, 3952),
            (3072, 1024, 3952),
            (4096, 1024, 3952),
            (5120, 880, 3952),
            (6000, 10, 3962),
        ]
    );
}

/// A prompt shorter than the drafter window taps all of it: the floor
/// saturates at 0, today's rule.
#[test]
fn a_prompt_under_the_window_taps_from_its_start() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(false), compute.clone());
    let request = sched.submit(input(tokens(1, 1500), None, Some(1400)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert!(
        jobs(&compute, request).iter().all(|&(_, _, floor)| floor == 0),
        "{:?}",
        jobs(&compute, request)
    );
}

/// A later turn resumes from the first turn's checkpoint and keeps its own
/// points: its opener's page floor (7088 for an opener at 7100) and its
/// checkpoint there, then its end. Its first chunk taps from 7088 - 2048 =
/// 5040: the 40 positions after the checkpoint it resumed from are read by no
/// later window.
#[test]
fn a_claimant_reads_its_floor_from_its_own_points() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(false), compute.clone());
    let first = tokens(1, 6000);
    sched.submit(input(first.clone(), None, Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let next = [first[..5000].to_vec(), tokens(20_000, 2200)].concat();
    let request = sched.submit(input(next, None, Some(7100)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![(5000, 1024, 5040), (6024, 1024, 5040), (7048, 40, 5040), (7088, 12, 5052), (7100, 100, 5152)]
    );
}
