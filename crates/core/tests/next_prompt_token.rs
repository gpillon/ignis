//! GitHub #519 (spec #502, part 4) — the **next prompt token** each prefill
//! job carries: the prompt's token right after the job's span, or `None` on
//! the job that ends at the prompt's last position, whose draw is the
//! sequence's pending token.
//!
//! The leaf runs the output head and samples only where that is `None`
//! (`RuntimeCompute::prefill_step` also hands it `None` on a job whose state
//! a prefix or a prompt checkpoint retains); every other job's draw was
//! discarded by the next one. So what the scheduler deals here decides where
//! the head runs.
//!
//! What these tests can prove: which token the scheduler hands each job, on a
//! first turn cut at its retained-image points, on a later turn resuming from
//! the first's checkpoint, and on a full-prompt match. That the leaf then
//! skips the head and leaves the sequence as today needs the model and lives
//! in `crates/core/tests/next_prompt_token_gpu.rs`.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` that
//! records every prefill job behind the `Compute` seam, as `drafter_floor.rs`
//! does.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, TokenId};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";

/// `n` distinct tokens starting at `start`: the token at position `p` of
/// `tokens(1, n)` is `p + 1`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request of `prompt`, its generation opener `opener` tokens in.
fn input(prompt: Vec<u32>, opener: Option<u32>) -> RequestInput {
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
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

fn scheduler(compute: Arc<MockCompute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    )
}

fn run_to_idle(sched: &mut ConcreteScheduler) {
    while !sched.is_idle() {
        sched.advance();
    }
}

/// Every prefill job dealt to `request`, as (start, tokens, next prompt token).
fn jobs(compute: &MockCompute, request: RequestId) -> Vec<(u32, usize, Option<TokenId>)> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .map(|j| (j.start_position, j.tokens.len(), j.next_prompt_token))
        .collect()
}

/// A 6000-token prompt whose opener ends at 5000, cut at the opener's page
/// floor (4992) and at the opener: each job carries the token at its end --
/// position `p` holds `p + 1` -- and the last carries none.
#[test]
fn each_job_carries_the_token_after_its_span_and_the_last_none() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let request = sched.submit(input(tokens(1, 6000), Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![
            (0, 1024, Some(1025)),
            (1024, 1024, Some(2049)),
            (2048, 1024, Some(3073)),
            (3072, 1024, Some(4097)),
            (4096, 896, Some(4993)),
            (4992, 8, Some(5001)),
            (5000, 1000, None),
        ]
    );
}

/// A later turn resumes from the first turn's checkpoint at 5000 and prefills
/// its own tail, whose tokens start at 20,000: each of its jobs carries its
/// own prompt's next token, and its last none.
#[test]
fn a_claimant_carries_its_own_prompts_tokens() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = tokens(1, 6000);
    sched.submit(input(first.clone(), Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let next = [first[..5000].to_vec(), tokens(20_000, 2200)].concat();
    let request = sched.submit(input(next, Some(7100)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        jobs(&compute, request),
        vec![
            (5000, 1024, Some(21_024)),
            (6024, 1024, Some(22_048)),
            (7048, 40, Some(22_088)),
            (7088, 12, Some(22_100)),
            (7100, 100, None),
        ]
    );
}

/// A prompt that ends exactly at the first turn's checkpoint claims all of it:
/// its one job warms nothing and ends the prompt, so it carries no token.
#[test]
fn a_full_prompt_match_carries_none() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = tokens(1, 6000);
    sched.submit(input(first.clone(), Some(5000)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let request = sched.submit(input(first[..5000].to_vec(), None), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(jobs(&compute, request), vec![(5000, 0, None)]);
}
