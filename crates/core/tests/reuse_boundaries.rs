//! GitHub #270 (spec `docs/specs/decide/16-reuse-boundaries.md`, ADR 0029 as
//! amended 2026-09-26) — a request carries an ordered **list of reuse
//! boundaries**, not one.
//!
//! #188 published a retained prefix at one place, the end of the system block.
//! A decision whose `state` is content parts keeps it in the user turn, so
//! nothing of the state was ever inside that place. The server now predicts
//! more of them (a fan-out's head, an observed fork, a caller's marker) and
//! hands each to the scheduler as a boundary with a **lifetime**: *retained*,
//! kept until the device needs the room, or *fan-out*, kept until the fan-out
//! that owns it ends.
//!
//! What the scheduler owns of that, and what this file pins: every boundary a
//! request carries is published, in prompt order, one retained slot each (the
//! first plainly, the rest chained, #187); a later request claims the longest
//! one its prompt matches; and a fan-out's head goes when its owner says the
//! fan-out ended — not before, and not at all when a longer-lived boundary sits
//! at the same place.
//!
//! Seams (ADR 0006): the `Scheduler` trait over `MockCompute`, as
//! `retained_prefix.rs` does.

use std::sync::Arc;

use ignis_core::types::{
    DecodeParams, RequestClass, RequestId, RequestInput, ReuseBoundary, SchedEvent,
};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
/// The default scheduler's KV page, in tokens.
const PAGE: u32 = 16;
/// The system block: 20 tokens, which floor to one whole page.
const BLOCK: u32 = 20;

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request over `prompt` whose render reported the system block at
/// [`BLOCK`] and no generation opener — so the only heads it publishes are
/// the ones in its boundary list, and no prompt checkpoint muddies the count.
fn input(prompt: Vec<u32>, boundaries: Vec<ReuseBoundary>) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(2),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: Some(BLOCK),
        reuse_boundaries: boundaries,
        constrained: None,
    }
}

/// The block, then a 30-token static part ending at 50, then 30 tokens that
/// change from request to request, numbered from `tail_start`.
fn prompt(tail_start: u32) -> Vec<u32> {
    [tokens(1, BLOCK), tokens(100, 30), tokens(tail_start, 30)].concat()
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

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// The leading prompt tokens `request` skipped through a shared prefix.
fn prefix_reuses(events: &[SchedEvent], request: RequestId) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::PrefixReused {
                request: r, tokens, ..
            } if *r == request => Some(*tokens),
            _ => None,
        })
        .collect()
}

/// The head each of `request`'s prefill chunks published, in order.
fn published(compute: &MockCompute, request: RequestId) -> Vec<Option<u32>> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .map(|j| j.publish_prefix.map(|p| p.tokens))
        .collect()
}

#[test]
fn every_boundary_a_request_carries_is_published_in_order_as_a_chain() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = sched
        .submit(input(prompt(500), vec![ReuseBoundary::retained(50)]), RequestClass::Agent)
        .unwrap();
    run_to_idle(&mut sched);

    // The block floors to one page, the static part's end to three: two cuts,
    // two heads, the second chained over the first.
    assert_eq!(
        published(&compute, first),
        vec![Some(16), Some(48), None],
        "the block, then the boundary past it"
    );
    assert_eq!(sched.retained_slots_in_use(), 2, "one retained slot each");
    assert_eq!(
        sched.prefix_pinned_pages(),
        48 / PAGE,
        "a chain: the second head owns only the pages past the first"
    );

    // A later request repeating the static part claims the longer head; one
    // that shares only the block claims the block.
    let repeat = sched
        .submit(input(prompt(900), vec![ReuseBoundary::retained(50)]), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, repeat), vec![48], "through the static part");

    let other = sched
        .submit(
            input([tokens(1, BLOCK), tokens(700, 60)].concat(), Vec::new()),
            RequestClass::Agent,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, other), vec![16], "only the block matches");
}

#[test]
fn a_request_that_resumed_still_publishes_what_the_server_predicted() {
    // With no generation opener, a request standing on a claim publishes no
    // head of its own (#187's gate): nobody extends a burst sibling's prompt.
    // A predicted boundary is exactly the claim that somebody will, so the
    // gate does not hold it back.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    sched
        .submit(input(prompt(500), vec![ReuseBoundary::retained(50)]), RequestClass::Agent)
        .unwrap();
    run_to_idle(&mut sched);
    let longer = [tokens(1, BLOCK), tokens(100, 30), tokens(200, 30)].concat();
    let second = sched
        .submit(input(longer, vec![ReuseBoundary::retained(70)]), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, second), vec![48]);
    assert_eq!(published(&compute, second), vec![Some(64), None], "chained over its claim");
}

#[test]
fn a_fan_out_head_outlives_its_publisher_and_goes_when_the_fan_out_ends() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = sched
        .submit(input(prompt(500), vec![ReuseBoundary::fan_out(50, 7)]), RequestClass::Agent)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(published(&compute, first), vec![Some(16), Some(48), None]);
    assert_eq!(sched.retained_slots_in_use(), 2);

    // The fan-out's followers arrive after its first question finished, and
    // the head is still there for them.
    let follower = sched
        .submit(input(prompt(900), Vec::new()), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, follower), vec![48], "the follower claims the head");
    assert_eq!(sched.retained_slots_in_use(), 2, "nothing ended yet");

    // Another fan-out ending gives up nothing of this one.
    sched.end_fan_out(8);
    assert_eq!(sched.retained_slots_in_use(), 2);

    let events = sched.end_fan_out(7);
    assert_eq!(sched.retained_slots_in_use(), 1, "the head's slot is back; the block's is not");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::RetainedSlots { in_use: 1, .. })),
        "and the change is reported without waiting for another step: {events:?}"
    );
    assert_eq!(sched.prefix_pinned_pages(), BLOCK / PAGE, "its own pages came back");

    let later = sched
        .submit(input(prompt(1300), Vec::new()), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, later), vec![16], "only the retained block is left");
}

#[test]
fn a_fan_out_head_at_a_retained_boundarys_place_stays_retained() {
    // 49 and 50 floor to the same page: one boundary, the longer lifetime.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = sched
        .submit(
            input(
                prompt(500),
                vec![ReuseBoundary::fan_out(49, 7), ReuseBoundary::retained(50)],
            ),
            RequestClass::Agent,
        )
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(published(&compute, first), vec![Some(16), Some(48), None], "one cut there, not two");

    sched.end_fan_out(7);
    assert_eq!(sched.retained_slots_in_use(), 2, "the fan-out's end gives up nothing");
    let later = sched
        .submit(input(prompt(900), Vec::new()), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, later), vec![48]);
}

#[test]
fn a_fan_out_that_ended_before_its_head_was_published_publishes_none() {
    // The client went away while the first question was still queued: its
    // head is a bet on followers that will never be asked.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = sched
        .submit(input(prompt(500), vec![ReuseBoundary::fan_out(50, 7)]), RequestClass::Agent)
        .unwrap();
    sched.end_fan_out(7);
    run_to_idle(&mut sched);
    assert_eq!(published(&compute, first), vec![Some(16), None], "only the block");
    assert_eq!(sched.retained_slots_in_use(), 1);
}

#[test]
fn a_boundary_that_floors_to_nothing_or_onto_the_block_adds_nothing() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());
    let first = sched
        .submit(
            input(
                prompt(500),
                vec![ReuseBoundary::retained(15), ReuseBoundary::retained(BLOCK + 3)],
            ),
            RequestClass::Agent,
        )
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(published(&compute, first), vec![Some(16), None]);
    assert_eq!(sched.retained_slots_in_use(), 1);
}

#[test]
fn prompt_reuse_off_publishes_no_boundary_it_was_handed() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            prompt_reuse: false,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let first = sched
        .submit(
            input(prompt(500), vec![ReuseBoundary::retained(50), ReuseBoundary::fan_out(60, 7)]),
            RequestClass::Agent,
        )
        .unwrap();
    run_to_idle(&mut sched);
    assert!(
        !published(&compute, first).iter().any(|p| *p == Some(48)),
        "no boundary is cut: {:?}",
        published(&compute, first)
    );
    assert_eq!(sched.retained_slots_in_use(), 0);
}
