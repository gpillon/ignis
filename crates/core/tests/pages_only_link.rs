//! ADR 0029 as amended 2026-10-07 (GitHub #306): on a load whose generation
//! opener's page **rides the capture** -- Flash-Next's -- the whole pages below
//! a request's opener are handed over by its prompt-checkpoint capture as a
//! **pages-only link**, so its prefill is cut once, at the opener, instead of
//! at the opener's page floor too.
//!
//! What these tests pin is what the next layer up observes without a card:
//! where each prefill is cut, what a later request reuses and from where, that
//! a link is never claimed on its own, and that every page comes back. That a
//! claimant *computes* what a cold prefill split at the same boundaries
//! computes needs the model, and lives in
//! `crates/runtime/tests/flash_next_reuse_gpu.rs`; what the leaf does with the
//! pages lives in `kernel/tests/test_seq_checkpoint.cpp`.
//!
//! Seams (ADR 0006): the `Scheduler` trait over a `MockCompute`, as
//! `prompt_checkpoint.rs` drives the imaged chain the 27B keeps.

use std::sync::Arc;

use ignis_core::checkpoint::ReuseSource;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, ReuseBoundary, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-flash-next";
/// The default scheduler's KV page, in tokens.
const PAGE: u32 = 16;

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request whose rendered prompt's generation opener ends `opener` tokens
/// in.
fn input(prompt: Vec<u32>, opener: Option<u32>, max: u32) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(max),
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

/// **Turn N**: 40 prompt tokens whose opener ends at 37, two pages and five
/// tokens in -- the opener's page floor is 32.
fn turn_n() -> RequestInput {
    input(tokens(1, 40), Some(37), 4)
}

/// A scheduler whose opener's page rides the capture.
fn config() -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        opener_page_rides_capture: true,
        ..SchedulerConfig::default()
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// Every prefill chunk width dealt to `request`, in order.
fn chunk_widths(compute: &MockCompute, request: RequestId) -> Vec<usize> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .map(|j| j.tokens.len())
        .collect()
}

/// The `StateReused` events for `request`, as (source, skipped tokens).
fn reuses(events: &[SchedEvent], request: RequestId) -> Vec<(ReuseSource, u32)> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::StateReused {
                request: r,
                source,
                tokens,
                ..
            } if *r == request => Some((*source, *tokens)),
            _ => None,
        })
        .collect()
}

// ── Where the prefill is cut ────────────────────────────────────────────

#[test]
fn the_prefill_is_cut_once_at_the_opener() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, n),
        vec![37, 3],
        "one cut, at the 37-token opener: none at its 32-token page floor"
    );
    let jobs: Vec<(Option<u32>, Option<u32>)> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == n)
        .map(|j| (j.publish_prefix.map(|p| p.tokens), j.capture_checkpoint.map(|c| c.tokens)))
        .collect();
    assert_eq!(
        jobs,
        vec![(None, Some(37)), (None, None)],
        "nothing is published at the floor; the chunk on the opener captures"
    );
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "and the checkpoint is kept");
}

// ── The reuse a link carries ────────────────────────────────────────────

#[test]
fn each_turn_resumes_from_the_last_and_is_cut_once_at_its_own_opener() {
    // Three turns of one conversation, each extending the last up to its
    // opener. Every turn after the first stands on the checkpoint before it,
    // which stands on a pages-only link -- and hands over a link of its own,
    // chained over that one, so the turn after it can resume in turn.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);

    let n1_prompt = [tokens(1, 37), tokens(500, 23)].concat();
    let n1 = sched
        .submit(input(n1_prompt.clone(), Some(57), 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::Device, 37)], "turn N+1 resumed at turn N's opener");
    assert_eq!(
        chunk_widths(&compute, n1),
        vec![20, 3],
        "from turn N's opener to its own in one chunk: no cut at its 48-token page floor"
    );

    let n2 = sched
        .submit(
            input([&n1_prompt[..57], &tokens(900, 30)[..]].concat(), Some(84), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n2), vec![(ReuseSource::Device, 57)], "turn N+2 resumed at turn N+1's opener");
    assert_eq!(chunk_widths(&compute, n2), vec![27, 3]);
    let _ = n;
}

#[test]
fn a_link_is_never_claimed_on_its_own() {
    // Turn N's link covers its first two pages. A prompt that shares them and
    // parts from turn N before its opener matches neither the checkpoint nor
    // -- since no request stands at the link's end, there is no state there
    // to hand over -- the link. It prefills its whole prompt.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.prefix_pinned_pages(), 2, "the link holds the two pages below the opener");
    assert_eq!(
        sched.kv_used_pages(),
        2 + sched.retained_tail_pages(),
        "and the pool is charged for them once, beside the checkpoint's own tail page"
    );
    assert_eq!(sched.retained_slots_in_use(), 1, "the link takes no retained slot: the checkpoint's image is the only one");

    let other = sched
        .submit(input([tokens(1, 34), tokens(800, 26)].concat(), Some(57), 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert!(reuses(&events, other).is_empty(), "no checkpoint matched");
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::PrefixReused { request, .. } if *request == other)),
        "and no prefix was claimed"
    );
    assert_eq!(chunk_widths(&compute, other).iter().sum::<usize>(), 60, "it prefilled its whole prompt");
    assert!(
        compute
            .prefill_calls()
            .iter()
            .flatten()
            .all(|j| j.shared_prefix.is_none()),
        "no job ever asked the backend for a prefix it holds no handle on"
    );
}

#[test]
fn a_declined_capture_publishes_nothing_at_the_floor() {
    // With no retained slot the capture is skipped -- and with it the link:
    // the prefill is not cut at all, and nothing is left behind.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            retained_slots: 0,
            ..config()
        },
        compute.clone(),
    );
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(chunk_widths(&compute, n), vec![40], "no cut for a capture that cannot happen");
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
    assert_eq!(sched.prefix_pinned_pages(), 0, "no link");
    assert_eq!(sched.kv_used_pages(), 0, "every page came back");
}

// ── Beside the reuse boundaries ─────────────────────────────────────────

#[test]
fn the_system_block_keeps_its_own_cut_and_its_image() {
    // A system block ending in the first page past one: its retained prefix is
    // a reuse boundary, published with its image at its own cut as before.
    // The opener's floor is two pages further on, and is not cut.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched
        .submit(
            RequestInput {
                system_block_tokens: Some(20),
                ..input(tokens(1, 56), Some(53), 4)
            },
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);
    let jobs: Vec<(usize, Option<u32>, Option<u32>)> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == n)
        .map(|j| (j.tokens.len(), j.publish_prefix.map(|p| p.tokens), j.capture_checkpoint.map(|c| c.tokens)))
        .collect();
    assert_eq!(
        jobs,
        vec![(PAGE as usize, Some(PAGE), None), (37, None, Some(53)), (3, None, None)],
        "the block's cut and publish, then one chunk to the opener"
    );

    // The next subagent of the burst shares the block and nothing more: it
    // claims the retained prefix, image and all.
    let sibling = sched
        .submit(
            RequestInput {
                system_block_tokens: Some(20),
                ..input([tokens(1, 20), tokens(700, 30)].concat(), Some(47), 4)
            },
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert!(events.iter().any(|e| matches!(
        e,
        SchedEvent::PrefixReused { request, tokens, .. } if *request == sibling && *tokens == PAGE
    )));
}

#[test]
fn a_block_on_the_opener_s_floor_is_what_the_capture_stands_on() {
    // The block's page floor is the opener's: the block's publish is the cut
    // there, and the capture chains nothing over it.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched
        .submit(
            RequestInput {
                system_block_tokens: Some(34),
                ..turn_n()
            },
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(chunk_widths(&compute, n), vec![32, 5, 3], "the block's cut, then the opener's");
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
    assert_eq!(sched.prefix_pinned_pages(), 2, "one retained prefix and no link");
    assert_eq!(sched.retained_slots_in_use(), 2, "the block's image and the checkpoint's");
}

#[test]
fn a_lender_publishes_nothing_past_its_opener() {
    // A reuse boundary past the opener: after the capture the sequence is the
    // link's lender, and the leaf refuses it a publish (`seq_prefix.cu`, a
    // second owner for the lent pages) -- which the backend would turn into a
    // failed batch. So the boundary is not cut, and nothing is published there.
    // (An 8-token serving chunk keeps the boundary out of the opener's chunk.)
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            serving_chunk_tokens: 8,
            ..config()
        },
        compute.clone(),
    );
    let n = sched
        .submit(
            RequestInput {
                reuse_boundaries: vec![ReuseBoundary::retained(52)],
                ..input(tokens(1, 60), Some(37), 4)
            },
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);
    let jobs: Vec<(u32, Option<u32>, Option<u32>)> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == n)
        .map(|j| {
            let end = j.start_position + j.tokens.len() as u32;
            (end, j.publish_prefix.map(|p| p.tokens), j.capture_checkpoint.map(|c| c.tokens))
        })
        .collect();
    assert_eq!(
        jobs,
        vec![
            (8, None, None),
            (16, None, None),
            (24, None, None),
            (32, None, None),
            (37, None, Some(37)),
            (45, None, None),
            (53, None, None),
            (60, None, None),
        ],
        "the capture lends, and the boundary at 48 past it is neither cut nor published"
    );
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
}

// ── A leaf that lent more than this scheduler moves ─────────────────────
//
// The leaf decides on its own view of the sequence whether a capture lends
// (`seq_checkpoint.cu`), the scheduler on its own whether to move the charge
// (`register_link`). Pages the leaf lent and the scheduler did not move would
// outlive the request in the leaf and go back to the pool in the ledger, and
// admission would over-commit. So a capture whose backend reports such a loan
// is released, not retained: once the lender goes, the link goes with it and
// every page is back where the ledger says it is.

#[test]
fn a_27b_capture_that_lent_is_released() {
    // The 27B moves nothing at a capture: it published the opener's floor.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            opener_page_rides_capture: false,
            ..config()
        },
        compute.clone(),
    );
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    compute.lend_at_capture(n, 1);
    let events = run_to_idle(&mut sched);
    assert_eq!(chunk_widths(&compute, n), vec![32, 5, 3], "the 27B's cuts: the floor, then the opener");
    assert_eq!(sched.checkpoint_pool().entry_count(), 0, "nothing retained");
    assert_eq!(compute.released_checkpoints(), vec![n], "the backend's checkpoint released");
    assert_eq!(sched.retained_slots_in_use(), 0, "its slot given back");
    assert_eq!(sched.kv_used_pages(), 0, "and every page back in the ledger");
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == n)),
        "the request completed normally"
    );
}

#[test]
fn a_link_the_leaf_made_longer_than_the_scheduler_s_is_released() {
    // Standing on the block's one-page prefix, the scheduler moves the one
    // page between it and the opener's floor; a leaf that lent two stood on
    // less than the scheduler thinks.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched
        .submit(
            RequestInput {
                system_block_tokens: Some(20),
                ..turn_n()
            },
            RequestClass::Interactive,
        )
        .unwrap();
    compute.lend_at_capture(n, 2);
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0, "nothing retained");
    assert_eq!(compute.released_checkpoints(), vec![n]);
    assert_eq!(sched.prefix_pinned_pages(), 1, "the block's retained prefix and no link");
    assert_eq!(sched.kv_used_pages(), 1, "its one page, and nothing of the capture's");
    assert_eq!(sched.retained_slots_in_use(), 1, "the block's image alone");
}

#[test]
fn a_loan_the_scheduler_moved_is_retained() {
    // The control: a loan of exactly the pages the scheduler moves.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    compute.lend_at_capture(n, 2);
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
    assert!(compute.released_checkpoints().is_empty());
    assert_eq!(sched.prefix_pinned_pages(), 2, "the link");
}

#[test]
fn two_identical_prompts_in_one_batch_leave_one_checkpoint() {
    // P4-10 kept one publisher per head in a batch, and with it one capture:
    // the loser stood on no prefix and could capture nothing. With no publish
    // at the floor the captures themselves are deduplicated by content.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let a = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    let b = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let captures: Vec<_> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter_map(|j| j.capture_checkpoint.map(|c| (j.request, c.tokens)))
        .collect();
    assert_eq!(captures, vec![(a, 37)], "the first of the two captures, the second does not");
    assert_eq!(chunk_widths(&compute, b), vec![40], "and the second is not cut for it");
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
    assert_eq!(sched.kv_used_pages(), 2 + 1, "one link and one tail page, nothing of the second's");
}

// ── KV-RAM ──────────────────────────────────────────────────────────────

#[test]
fn a_checkpoint_on_a_link_goes_to_kv_ram_and_its_restorer_leaves_one_of_its_own() {
    // A request needing the whole device pool spills turn N's checkpoint,
    // link pages included, to KV-RAM. The next turn restores it into pages of
    // its own -- standing on no prefix at all -- and its capture hands every
    // page below its opener over as a root link, so the turn after it resumes
    // on the device.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            max_sequence_tokens: 1280,
            kv_capacity_pages: 80,
            ..config()
        },
        compute.clone(),
    );
    let history = tokens(1, 1100);
    let n = sched
        .submit(input(history.clone(), Some(1050), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(chunk_widths(&compute, n), vec![1024, 26, 50], "the serving chunk, then the opener");

    sched
        .submit(input(tokens(5000, 1272), None, 8), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entries()[0].tier, ReuseSource::KvRam, "spilled");
    assert_eq!(sched.kv_used_pages(), 0, "the link's pages came back with the device image");
    assert!(
        !compute.released_prefixes().contains(&n),
        "and no backend handle was dropped for a link that never had one"
    );

    let n1_prompt = [&history[..1050], &tokens(9000, 100)[..]].concat();
    let n1 = sched
        .submit(input(n1_prompt.clone(), Some(1140), 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::KvRam, 1050)], "restored from KV-RAM");
    assert_eq!(chunk_widths(&compute, n1), vec![90, 10], "from the restored opener to its own, uncut");
    assert_eq!(sched.prefix_pinned_pages(), 71, "a root link: every whole page below 1140");

    let n2 = sched
        .submit(
            input([&n1_prompt[..1140], &tokens(12000, 20)[..]].concat(), Some(1157), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n2), vec![(ReuseSource::Device, 1140)], "on the device, over the root link");
}

#[test]
fn a_page_aligned_opener_lends_its_whole_pages_and_copies_none() {
    // An opener on a page boundary ends inside no page: the one cut is the
    // opener's, the link holds every page below it, and the checkpoint keeps
    // no tail page of its own.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let n = sched
        .submit(input(tokens(1, 40), Some(2 * PAGE), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(chunk_widths(&compute, n), vec![32, 8], "cut once, at the opener on the page boundary");
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
    assert_eq!(sched.prefix_pinned_pages(), 2, "the link holds both pages below the opener");
    assert_eq!(sched.retained_tail_pages(), 0, "and the checkpoint copies no page");

    let later = sched
        .submit(input([tokens(1, 32), tokens(600, 28)].concat(), Some(57), 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, later), vec![(ReuseSource::Device, 32)]);
}
