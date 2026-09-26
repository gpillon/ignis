//! GitHub #215 (ADR 0030) — retained state lives in **retained slots**.
//!
//! A prefix publish and a checkpoint capture each take one of the load's
//! retained slots instead of allocating device memory, and a checkpoint's
//! partial tail page is one KV page of the pool. When no slot is free, retained
//! state gives one up in ADR 0023's order — checkpoints before retained
//! prefixes, `Agent` before `Interactive`, least recently used — spilling to
//! KV-RAM as the page path does; a claimed prefix and the chain under it are
//! never given up; and when nothing can give a slot up the publish or capture
//! is skipped, never waited for.
//!
//! What these tests can prove, and what they cannot. That the state in a slot
//! is the *right* state needs the model and lives in the GPU tests. What lives
//! here is what the next layer up observes without a card: which slot a job
//! names, how many are held, what is given up for one, and that a request
//! finding none still runs.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` that
//! records the prefill jobs and every spill behind the `Compute` seam.

use std::sync::Arc;

use ignis_core::checkpoint::ReuseSource;
use ignis_core::retained_slot::RetainedSkip;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
/// The default scheduler's KV page, in tokens.
const PAGE: u32 = 16;

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request reporting `block` as its system-and-tools block and `opener` as
/// its generation opener.
fn input(prompt: Vec<u32>, block: Option<u32>, opener: Option<u32>, max: u32) -> RequestInput {
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
        system_block_tokens: block,
        reuse_boundaries: Vec::new(),
        constrained: None,
    }
}

/// A 40-token turn starting at `start` whose opener ends at 37: one 32-token
/// prefix published under it, one checkpoint captured at it.
fn turn(start: u32) -> RequestInput {
    input(tokens(start, 40), None, Some(37), 4)
}

/// One subagent of a burst: a 35-token block starting at `block`, then its
/// own 25-token question. It publishes the retained block (32), chains the
/// opener's page over it (48) and captures at the opener (57): three slots.
fn subagent(block: u32, query: u32) -> RequestInput {
    input(
        [tokens(block, 35), tokens(query, 25)].concat(),
        Some(35),
        Some(57),
        4,
    )
}

/// The subagent's next iteration: its prompt up to its opener, then a tool
/// result and a new opener at 84, so it claims the checkpoint at 57 and would
/// chain a link at 80.
fn next_iteration(block: u32, query: u32, max: u32) -> RequestInput {
    input(
        [tokens(block, 35), tokens(query, 22), tokens(900, 30)].concat(),
        Some(35),
        Some(84),
        max,
    )
}

fn scheduler(compute: Arc<MockCompute>, retained_slots: u32) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            retained_slots,
            ..SchedulerConfig::default()
        },
        compute,
    )
}

/// Every event until the scheduler is idle. Capped, so a request that is never
/// admitted fails the test rather than hanging it.
fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    for _ in 0..10_000 {
        if sched.is_idle() {
            return events;
        }
        events.extend(sched.advance());
    }
    panic!("the scheduler never went idle: {events:?}");
}

/// The skips `request` was told about.
fn skips(events: &[SchedEvent], request: RequestId) -> Vec<RetainedSkip> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::RetainedSlotSkipped { request: r, skip, .. } if *r == request => Some(*skip),
            _ => None,
        })
        .collect()
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

fn device_checkpoints(sched: &ConcreteScheduler) -> Vec<RequestId> {
    sched
        .checkpoint_pool()
        .entries()
        .iter()
        .filter(|e| e.tier == ReuseSource::Device)
        .map(|e| e.publisher)
        .collect()
}

fn done(events: &[SchedEvent], request: RequestId) -> bool {
    events
        .iter()
        .any(|e| matches!(e, SchedEvent::Done { request: r, .. } if *r == request))
}

// ── What a slot holds ───────────────────────────────────────────────────

#[test]
fn a_checkpoint_and_the_prefix_under_it_hold_two_slots() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 8);
    assert_eq!(sched.retained_slot_count(), 8);
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(sched.retained_slots_in_use(), 2, "the prefix and the checkpoint");
    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().filter(|j| j.request == a).collect();
    let publish = jobs.iter().find_map(|j| j.publish_prefix).expect("a publish");
    let capture = jobs.iter().find_map(|j| j.capture_checkpoint).expect("a capture");
    assert_eq!((publish.tokens, capture.tokens), (32, 37));
    assert_ne!(publish.slot, capture.slot, "each image has a slot of its own");
    assert!(skips(&events, a).is_empty());
    let reported = events.iter().rev().find_map(|e| match e {
        SchedEvent::RetainedSlots { in_use, capacity } => Some((*in_use, *capacity)),
        _ => None,
    });
    assert_eq!(reported, Some((2, 8)), "the last report is what is held");
}

#[test]
fn a_chained_link_is_one_more_slot() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 8);
    sched.submit(turn(1), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    // Turn N+1: turn N up to its opener, then more, and an opener at 57. It
    // claims the checkpoint, chains a link at 48 and captures at 57.
    let next = input([tokens(1, 37), tokens(500, 23)].concat(), None, Some(57), 4);
    let n1 = sched.submit(next, RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(device_checkpoints(&sched).len(), 2, "turn N's and turn N+1's");
    assert_eq!(
        sched.retained_slots_in_use(),
        4,
        "two links of one chain and two checkpoints: {:?}",
        compute.prefill_calls().iter().flatten().filter(|j| j.request == n1).collect::<Vec<_>>()
    );
}

#[test]
fn a_checkpoint_holds_its_partial_tail_page_as_one_kv_page() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute, 8);
    sched.submit(turn(1), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        sched.kv_used_pages(),
        2 + 1,
        "the prefix's two pages, and the page the opener ends inside"
    );
    assert_eq!(sched.retained_tail_pages(), 1);

    // An opener on a page boundary ends inside no page: nothing to copy.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute, 8);
    sched
        .submit(input(tokens(1, 40), None, Some(2 * PAGE), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(device_checkpoints(&sched).len(), 1);
    assert_eq!(sched.kv_used_pages(), 2, "the prefix's pages alone");
    assert_eq!(sched.retained_tail_pages(), 0);
}

// ── When no slot is free ────────────────────────────────────────────────

#[test]
fn a_full_set_of_slots_gives_up_an_agent_checkpoint_before_an_interactive_one() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 4);
    let interactive = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let agent = sched.submit(turn(1000), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.retained_slots_in_use(), 4, "full");

    // The Interactive checkpoint is the least recently used, and the Agent one
    // still goes first: class before age.
    let c = sched.submit(turn(2000), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.spilled_checkpoints(), vec![agent], "spilled to KV-RAM");
    assert!(skips(&events, c).is_empty(), "the prefix under it went with it: two slots");
    assert_eq!(device_checkpoints(&sched), vec![interactive, c]);
    assert_eq!(sched.retained_slots_in_use(), 4);
}

#[test]
fn checkpoints_are_given_up_before_retained_prefixes() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 3);
    let a = sched.submit(subagent(1, 500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.retained_slots_in_use(), 3, "block, chained link, checkpoint");

    // A burst from another parent. Its first chunk lands on its own block and
    // needs one slot: the checkpoint goes, and the link only it held goes with
    // it — the retained block stays.
    let b = sched.submit(subagent(5000, 700), RequestClass::Agent).unwrap();
    sched.advance();
    assert_eq!(compute.spilled_checkpoints(), vec![a]);
    assert!(compute.spilled_prefixes().is_empty(), "the wide bet is still on the device");

    // Its capture finds no checkpoint left to give up: then the block goes.
    let events = run_to_idle(&mut sched);
    assert!(skips(&events, b).is_empty(), "{events:?}");
    assert_eq!(compute.spilled_prefixes(), vec![(a, 32)]);
    assert_eq!(device_checkpoints(&sched), vec![b]);
    assert_eq!(sched.retained_slots_in_use(), 3);
}

#[test]
fn a_claimed_checkpoint_and_the_chain_under_a_live_request_are_never_given_up() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 3);
    let a = sched.submit(subagent(1, 500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);

    // The next iteration claims A's checkpoint. The link it would chain at 80
    // finds every slot held: the checkpoint it is being built from, and the
    // chain under it. Nothing is given up, and the publish is skipped.
    let c = sched
        .submit(next_iteration(1, 500, 40), RequestClass::Agent)
        .unwrap();
    let mut events = sched.advance();
    assert_eq!(skips(&events, c), vec![RetainedSkip::PublishNoSlot]);
    assert!(compute.spilled_checkpoints().is_empty());
    assert!(
        events.iter().any(|e| matches!(
            e,
            SchedEvent::StateReused { request, source: ReuseSource::Device, tokens: 57, .. } if *request == c
        )),
        "the claim itself landed: {events:?}"
    );

    // While it decodes, another request arrives. The checkpoint is no longer
    // being claimed, so its slot is given up; the chain the live request
    // stands on is not, so the newcomer's capture finds nothing.
    while !events.iter().any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == c)) {
        events = sched.advance();
    }
    let d = sched.submit(turn(3000), RequestClass::Interactive).unwrap();
    let first = sched.advance();
    assert!(
        first.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == d)),
        "admitted on the next tick, never made to wait: {first:?}"
    );
    let mut rest = run_to_idle(&mut sched);
    rest.splice(0..0, first);
    assert_eq!(compute.spilled_checkpoints(), vec![a]);
    assert!(compute.spilled_prefixes().is_empty(), "a claimed chain is never a victim");
    assert_eq!(skips(&rest, d), vec![RetainedSkip::CaptureNoSlot]);
    assert!(done(&rest, c) && done(&rest, d));
    assert!(
        !rest.iter().any(|e| matches!(e, SchedEvent::Evicted { .. } | SchedEvent::Requeued { .. })),
        "and nobody live was disturbed for it"
    );
}

#[test]
fn a_skipped_capture_does_not_cut_the_prefill_at_the_opener() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 1);
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(skips(&events, a), vec![RetainedSkip::CaptureNoSlot]);
    assert_eq!(
        chunk_widths(&compute, a),
        vec![32, 8],
        "cut where the prefix was published, and not again at 37"
    );
    assert!(device_checkpoints(&sched).is_empty());
    assert!(done(&events, a));
}

#[test]
fn a_capture_with_no_kv_page_for_its_tail_is_skipped() {
    // 40 prompt tokens and 4 to generate reserve three pages, and the pool has
    // exactly three: the page the opener ends inside has nowhere to go.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            kv_capacity_pages: 3,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(skips(&events, a), vec![RetainedSkip::CaptureNoPage]);
    assert_eq!(chunk_widths(&compute, a), vec![32, 8], "and not cut at 37 for it");
    assert!(device_checkpoints(&sched).is_empty());
    assert_eq!(sched.retained_tail_pages(), 0);
    assert!(done(&events, a));
}

#[test]
fn a_failed_prefill_batch_gives_its_slots_back() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 2);
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    compute.fail_prefill(a);
    for _ in 0..10 {
        sched.advance();
        if sched.last_error().is_some() {
            break;
        }
    }
    assert!(sched.last_error().is_some(), "the publish chunk failed");
    assert_eq!(sched.retained_slots_in_use(), 0, "its publish's slot came back with it");
    let events = run_to_idle(&mut sched);
    assert!(done(&events, a));
    assert_eq!(sched.retained_slots_in_use(), 2, "and the retry took it again");
}

#[test]
fn a_tail_page_beside_a_live_request_comes_back_before_anyone_live_waits() {
    // A turn decoding on the prefix it published keeps its checkpoint's tail
    // page beside it. That page is retained state, so a request that needs it
    // gets it at once (ADR 0029): the checkpoint is given up, and nobody waits
    // for the turn to finish or is evicted for it.
    let long = || input(tokens(1, 40), None, Some(37), 200);
    let other = || input(tokens(5000, 40), None, None, 4);
    let with_pages = |compute: Arc<MockCompute>, kv_capacity_pages: u32| {
        ConcreteScheduler::with_config(
            SchedulerConfig {
                model: MODEL.into(),
                kv_capacity_pages,
                ..SchedulerConfig::default()
            },
            compute,
        )
    };
    let until_captured = |sched: &mut ConcreteScheduler| {
        for _ in 0..100 {
            if !device_checkpoints(sched).is_empty() {
                return;
            }
            sched.advance();
        }
        panic!("the long turn never captured");
    };
    // Each reserves its prompt and its generation, in whole pages: the long
    // turn 240 tokens, the other 44.
    let (long_pages, other_pages) = ((40 + 200u32).div_ceil(PAGE), (40 + 4u32).div_ceil(PAGE));

    // A pool with room for both only without the tail page.
    let compute = Arc::new(MockCompute::new());
    let mut sched = with_pages(compute.clone(), long_pages + other_pages);
    let a = sched.submit(long(), RequestClass::Interactive).unwrap();
    until_captured(&mut sched);
    assert_eq!(sched.kv_used_pages(), long_pages + 1, "the long turn's pages and its tail page");
    let b = sched.submit(other(), RequestClass::Interactive).unwrap();
    let mut events = Vec::new();
    for _ in 0..10 {
        events.extend(sched.advance());
    }
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == b)),
        "the other request is admitted while the long turn decodes: {events:?}"
    );
    assert!(!done(&events, a), "the long turn is still decoding");
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::Evicted { .. } | SchedEvent::Requeued { .. })),
        "and nobody live was disturbed for it"
    );
    assert_eq!(sched.retained_tail_pages(), 0, "the checkpoint gave its page up");
}

#[test]
fn a_skip_names_the_slots_held_when_it_happened() {
    // Two turns prefilled in one batch, one slot: the first publish takes it,
    // and the second is skipped in the same tick — before any report of the
    // slot the first one took.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), 1);
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    let b = sched.submit(turn(100), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    let first = compute.prefill_calls().into_iter().next().expect("a prefill batch");
    assert!(
        [a, b].iter().all(|r| first.iter().any(|j| j.request == *r)),
        "both turns share the first batch: {first:?}"
    );
    let held = events.iter().find_map(|e| match e {
        SchedEvent::RetainedSlotSkipped {
            skip: RetainedSkip::PublishNoSlot,
            in_use,
            capacity,
            ..
        } => Some((*in_use, *capacity)),
        _ => None,
    });
    assert_eq!(held, Some((1, 1)), "one of the two publishes finds no slot: {events:?}");
}

#[test]
fn a_load_with_no_retained_slots_and_prompt_reuse_on_says_what_it_skipped() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute, 0);
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(skips(&events, a), vec![RetainedSkip::PublishNoSlot]);
    assert!(done(&events, a));
}

#[test]
fn prompt_reuse_off_with_slots_shares_live_siblings_and_retains_nothing() {
    // The slots a load gives with prompt reuse off are for live siblings
    // alone: a head is published and claimed while its publisher runs, no
    // checkpoint is captured, and the slot comes back when the head goes.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            prompt_reuse: false,
            retained_slots: 2,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let a = sched.submit(input(tokens(1, 40), None, Some(37), 50), RequestClass::Interactive).unwrap();
    let mut events = Vec::new();
    for _ in 0..3 {
        events.extend(sched.advance());
    }
    let b = sched.submit(input(tokens(1, 40), None, Some(37), 4), RequestClass::Interactive).unwrap();
    events.extend(run_to_idle(&mut sched));
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::PrefixReused { request, .. } if *request == b)),
        "the sibling claims the live head: {events:?}"
    );
    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().collect();
    assert!(jobs.iter().any(|j| j.request == a && j.publish_prefix.is_some()));
    assert!(jobs.iter().all(|j| j.capture_checkpoint.is_none()), "no checkpoint with reuse off");
    assert!(device_checkpoints(&sched).is_empty());
    assert_eq!(sched.retained_slots_in_use(), 0, "nothing retained once both are done");
}

#[test]
fn prompt_reuse_off_reserves_no_slots() {
    // What the server resolves `--prompt-reuse off` to by default.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            prompt_reuse: false,
            retained_slots: 0,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    assert_eq!(sched.retained_slot_count(), 0);
    let a = sched.submit(turn(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().collect();
    assert!(jobs.iter().all(|j| j.publish_prefix.is_none() && j.capture_checkpoint.is_none()));
    assert!(skips(&events, a).is_empty(), "nothing asked for, nothing skipped");
    assert_eq!(chunk_widths(&compute, a), vec![40], "and nothing cut for");
}
