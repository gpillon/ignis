//! Spec vram-budget/03 P3, the fixed branch (ADR 0045, GitHub #309): live
//! moves under pool pressure, through the concrete scheduler on a CPU
//! (`MockCompute`, ADR 0006).
//!
//! P0 chose the fixed branch (`docs/findings/2026-10-08-kv-page-growth-gate.md`):
//! a request reserves its whole bound -- prompt plus its cap -- when it is
//! admitted, and only an admission ever moves a sequence. When an admission
//! finds too few pages, retained state goes first, then the lowest-ranked live
//! sequence *below the requester* moves down a tier, mid-generation if need
//! be, and resumes later where it stopped. Rank is class, then submission
//! order; a moved sequence keeps it, and entries onto the device -- restores
//! and admissions alike -- are taken in that order.
//!
//! The mock's token streams are pure functions of the request, so "its output
//! continues unbroken" is checked against the stream it would have emitted had
//! it never moved. That the bytes that come back are the right bytes needs the
//! card: `crates/server/tests/live_moves_gpu.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use ignis_core::disk::disk_file_bytes;
use ignis_core::scheduler::{DiskBlob, DiskSource};
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, RequestState, SchedEvent};
use ignis_core::{ConcreteScheduler, FakeDisk, MockCompute, MockSections, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
const PAGE: u32 = 16;

fn input(prompt: Vec<u32>, max_tokens: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams { max_tokens, ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

/// A prompt of `n` tokens from `start`, so prompts differ.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A four-token prompt and `max` tokens to generate: `ceil((4 + max) / 16)`
/// pages.
fn live(start: u32, max: u32) -> RequestInput {
    input(tokens(start, 4), Some(max))
}

/// A pool of 64 pages -- one 1,024-token sequence -- with KV-RAM for `host`
/// nominal blobs and no disk. Prompt reuse off: these scenarios are about live
/// work, and retained state would only move pages around them.
fn pool(host: u64) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_in_flight: 8,
        max_sequence_tokens: 1024,
        kv_capacity_pages: 64,
        kv_page_tokens: PAGE,
        host_capacity_bytes: host,
        prompt_reuse: false,
        retained_slots: 0,
        ..SchedulerConfig::default()
    }
}

/// Every event of a run, the advance each one came in, and a check run on
/// every advance: no sequence on the device is ever held out of a round for
/// pages (AC 35, fixed bullet). A request `Running` before an advance and
/// still `Running` after it, that did not leave the device in it and is not
/// mid-transfer to or from the disk, emitted a token in it.
struct Run {
    sched: ConcreteScheduler,
    tracked: Vec<RequestId>,
    events: Vec<(usize, SchedEvent)>,
    advances: usize,
}

impl Run {
    fn new(sched: ConcreteScheduler) -> Self {
        Self { sched, tracked: Vec::new(), events: Vec::new(), advances: 0 }
    }

    fn submit(&mut self, input: RequestInput, class: RequestClass) -> RequestId {
        let id = self.sched.submit(input, class).expect("admitted to the queue");
        self.tracked.push(id);
        id
    }

    fn step(&mut self) -> Vec<SchedEvent> {
        let running: Vec<RequestId> = self
            .tracked
            .iter()
            .copied()
            .filter(|&r| self.sched.request_state(r) == Some(RequestState::Running) && !self.sched.in_transfer(r))
            .collect();
        let out = self.sched.advance();
        for r in running {
            let left = out.iter().any(|e| {
                matches!(e, SchedEvent::Evicted { request, .. } | SchedEvent::DiskSpilled { request, .. } if *request == r)
            });
            if self.sched.request_state(r) == Some(RequestState::Running) && !self.sched.in_transfer(r) && !left {
                assert!(
                    out.iter().any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == r)),
                    "request {r} stayed on its lane through advance {} and decoded nothing",
                    self.advances
                );
            }
        }
        for e in &out {
            self.events.push((self.advances, e.clone()));
        }
        self.advances += 1;
        out
    }

    fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    fn to_idle(&mut self) {
        while !self.sched.is_idle() {
            self.step();
            assert!(self.advances < 100_000, "the scheduler never went idle");
        }
    }

    fn until(&mut self, mut done: impl FnMut(&Self) -> bool) {
        while !done(self) {
            assert!(!self.sched.is_idle(), "the run went idle first");
            assert!(self.advances < 100_000, "the run never got there");
            self.step();
        }
    }

    fn state(&self, r: RequestId) -> Option<RequestState> {
        self.sched.request_state(r)
    }

    fn tokens(&self, r: RequestId) -> Vec<u32> {
        self.events
            .iter()
            .filter_map(|(_, e)| match e {
                SchedEvent::Token { request, token } if *request == r => Some(*token),
                _ => None,
            })
            .collect()
    }

    fn first(&self, f: impl Fn(&SchedEvent) -> bool) -> Option<usize> {
        self.events.iter().find(|(_, e)| f(e)).map(|(at, _)| *at)
    }

    fn has(&self, f: impl Fn(&SchedEvent) -> bool) -> bool {
        self.first(f).is_some()
    }

    /// Any sequence leaving the device, to KV-RAM or to the disk.
    fn moved_any(&self) -> bool {
        self.has(|e| {
            matches!(e, SchedEvent::Evicted { .. } | SchedEvent::DiskSpilled { from: DiskSource::Device, .. })
        })
    }

    fn evicted(&self, r: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == r))
    }

    fn done(&self, r: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Done { request, .. } if *request == r))
    }

    fn assert_no_work_lost(&self) {
        assert!(!self.has(|e| matches!(e, SchedEvent::Requeued { .. })), "no request lost its work");
        assert!(!self.has(|e| matches!(e, SchedEvent::SnapshotDropped { .. })), "no snapshot was dropped");
    }
}

/// `request`'s stream as the mock emits it had it never moved.
fn expected(compute: &MockCompute, request: RequestId, n: u32) -> Vec<u32> {
    (0..n).map(|step| compute.token_for(request, step)).collect()
}

// ── AC 28 (fixed): the reservation is the bound, from admission ─────────────

#[test]
fn a_request_reserves_its_whole_bound_at_admission_and_never_grows() {
    // Flash-Next's page and context: the default cap, not the context, binds.
    const CONTEXT: u32 = 262_144;
    const PROMPT: u32 = 1_000;
    let pages = |tokens: u32| tokens.div_ceil(64);
    for (default_max_tokens, max_tokens, bound) in [
        // Without a cap: prompt plus the flag's value, at its default and at
        // another value.
        (38_912, None, PROMPT + 38_912),
        (8_192, None, PROMPT + 8_192),
        // `0` is no default: the whole context, today's rule.
        (0, None, CONTEXT),
        // An explicit cap is today's reservation, whatever the flag says.
        (8_192, Some(500), PROMPT + 500),
        (38_912, Some(100_000), PROMPT + 100_000),
    ] {
        let mut sched = ConcreteScheduler::with_config(
            SchedulerConfig {
                model: MODEL.into(),
                kv_page_tokens: 64,
                max_sequence_tokens: CONTEXT,
                kv_capacity_pages: 8_192,
                default_max_tokens,
                ..SchedulerConfig::default()
            },
            Arc::new(MockCompute::new()),
        );
        let id = sched
            .submit(input(tokens(1, PROMPT), max_tokens), RequestClass::Agent)
            .expect("admitted");
        sched.advance();
        assert_ne!(sched.request_state(id), Some(RequestState::Admitted), "materialized at its first chunk");
        let label = format!("default {default_max_tokens}, max_tokens {max_tokens:?}");
        assert_eq!(sched.kv_used_pages(), pages(bound), "{label}: the whole bound at admission");
        for _ in 0..64 {
            sched.advance();
            assert_eq!(sched.kv_used_pages(), pages(bound), "{label}: and it never grows");
        }
    }
}

// ── AC 33: an admission is the trigger ──────────────────────────────────────

#[test]
fn an_interactive_admission_moves_a_decoding_agent_which_resumes_on_a_lane_unbroken() {
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(pool(4), compute.clone()));
    let a = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    run.steps(3);
    assert!(!run.tokens(a).is_empty(), "A decodes alone first");

    // B needs 38 pages beside A's 38 of 64: A moves, mid-generation.
    let b = run.submit(live(100, 600), RequestClass::Interactive);
    run.step();
    assert!(run.evicted(a), "A left the device for KV-RAM, at the round boundary");
    assert_eq!(run.state(a), Some(RequestState::Evicted));
    assert_eq!(run.state(b), Some(RequestState::Running), "B took its pages in the same advance");

    run.until(|run| run.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a)));
    assert!(
        run.has(|e| matches!(e, SchedEvent::Restored { request, lane: Some(_), .. } if *request == a)),
        "a victim that held a lane comes back on one"
    );
    assert_eq!(run.state(a), Some(RequestState::Running));
    run.to_idle();
    assert!(run.done(a) && run.done(b));
    assert_eq!(run.tokens(a), expected(&compute, a, 600), "A's output continues unbroken across the move");
    run.assert_no_work_lost();
    assert_eq!(run.sched.kv_used_pages(), 0);
}

#[test]
fn an_interactive_admission_moves_a_lane_less_agent_which_resumes_prefilling_from_its_progress() {
    // A lane-less victim is one waiting for a lane: in this loop a fresh
    // admission never runs beside a half-prefilled prompt (the prefill lane
    // serves that one alone), so a sequence at a chunk boundary is never an
    // admission's victim. Eight Interactive lanes hold every lane, each with a
    // blob larger than the disk -- so the head's lane deal cannot move one --
    // and an Agent finishes its prompt and waits for a lane. No KV-RAM: there
    // a lane holder's refused move would give up what the tier holds.
    let compute = Arc::new(
        MockCompute::with_sections(MockSections { image_bytes: 0, bytes_per_token: 100 })
            .with_disk(FakeDisk::with_room(1 << 30)),
    );
    let mut run = Run::new(ConcreteScheduler::with_config(
        SchedulerConfig {
            max_in_flight: 16,
            resident_slot_capacity: 16,
            kv_capacity_pages: 80,
            // One file of up to 40 tokens: P's, never a lane holder's.
            kv_disk_capacity_bytes: disk_file_bytes(4_000),
            ..pool(0)
        },
        compute.clone(),
    ));
    let lanes: Vec<RequestId> =
        (0..8).map(|i| run.submit(input(tokens(1000 + 100 * i, 40), Some(100)), RequestClass::Interactive)).collect(); // 9 pages each
    run.step();
    assert!(lanes.iter().all(|&l| run.state(l) == Some(RequestState::Running)), "every lane is held");
    let p = run.submit(live(5000, 12), RequestClass::Agent); // 1 page
    run.steps(2);
    assert_eq!(run.state(p), Some(RequestState::Prefilling), "P finished its prompt and waits for a lane");
    assert_eq!(run.sched.prefill_progress(p), Some(4));
    assert!(!run.moved_any(), "no lane holder could be moved for P's lane");

    // I needs 8 pages, and 73 of 80 are held: P, the Agent not decoding, goes.
    let i = run.submit(live(6000, 120), RequestClass::Interactive);
    run.until(|run| run.has(|e| matches!(e, SchedEvent::DiskSpilled { request, .. } if *request == p)));
    assert_eq!(compute.disk_spills(), vec![(DiskBlob::Live(p), DiskSource::Device)], "P, and only P, left the device");

    run.until(|run| run.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == p)));
    assert!(
        run.has(|e| matches!(e, SchedEvent::Restored { request, lane: None, .. } if *request == p)),
        "a lane-less victim comes back without one"
    );
    // It resumed as Prefilling, from its progress: a lane deal is all it
    // waited for, and none of its prompt was sent again.
    assert_eq!(run.sched.prefill_progress(p), Some(4), "from its progress, not from zero");
    let chunks = |run: &Run| {
        run.events
            .iter()
            .filter(|(_, e)| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == p))
            .count()
    };
    assert_eq!(chunks(&run), 1);
    run.to_idle();
    assert_eq!(chunks(&run), 1, "P never prefilled again");
    assert!(run.done(p) && run.done(i));
    assert_eq!(run.tokens(p), expected(&compute, p, 12), "P's output continues unbroken");
    run.assert_no_work_lost();
}

#[test]
fn an_agent_admission_behind_a_resident_agent_moves_nothing_and_waits() {
    // KV-RAM and the disk both have room: the wait is the rank, not the tiers.
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut run = Run::new(ConcreteScheduler::with_config(
        SchedulerConfig { kv_disk_capacity_bytes: 1 << 20, ..pool(4) },
        compute.clone(),
    ));
    let a = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    run.steps(3);
    let n = run.submit(live(100, 600), RequestClass::Agent); // 38 pages: never both
    for _ in 0..50 {
        run.step();
        assert_eq!(run.state(n), Some(RequestState::Admitted), "N waits, held in Admitted");
    }
    assert!(!run.moved_any(), "an Agent admission moves no older Agent");
    assert!(run.tokens(a).len() >= 50, "A keeps decoding meanwhile");

    // A completes: N is admitted at the next advance.
    run.until(|run| run.done(a));
    let finished = run.advances - 1;
    run.to_idle();
    let admitted = run.first(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == n));
    assert_eq!(admitted, Some(finished + 1), "N enters when room returns");
    assert!(!run.moved_any());
    assert_eq!(run.tokens(n), expected(&compute, n, 600));
    run.assert_no_work_lost();
}

// ── AC 32: the rank gate ────────────────────────────────────────────────────

#[test]
fn an_agent_need_never_moves_an_interactive_sequence_nor_a_newer_interactive_an_older_one() {
    for newcomer in [RequestClass::Agent, RequestClass::Interactive] {
        let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
        let mut run = Run::new(ConcreteScheduler::with_config(
            SchedulerConfig { kv_disk_capacity_bytes: 1 << 20, ..pool(4) },
            compute.clone(),
        ));
        let resident = run.submit(live(1, 600), RequestClass::Interactive);
        run.steps(3);
        let n = run.submit(live(100, 600), newcomer);
        for _ in 0..50 {
            run.step();
            assert_eq!(run.state(n), Some(RequestState::Admitted), "a {newcomer:?} newcomer waits");
        }
        assert!(!run.moved_any(), "the older Interactive is never moved for a {newcomer:?}");
        run.to_idle();
        assert_eq!(run.tokens(resident), expected(&compute, resident, 600));
        assert_eq!(run.tokens(n), expected(&compute, n, 600));
        run.assert_no_work_lost();
    }
}

#[test]
fn an_interactive_newcomer_moves_the_youngest_agent_first() {
    // Two Agents decode; an Interactive needs one of them out: the
    // latest-submitted goes, the older stays and keeps decoding.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(pool(4), compute.clone()));
    let older = run.submit(live(1, 380), RequestClass::Agent); // 24 pages
    let younger = run.submit(live(100, 380), RequestClass::Agent); // 24 pages
    run.steps(3);
    let i = run.submit(live(200, 300), RequestClass::Interactive); // 19 pages: one Agent out
    run.step();
    assert!(run.evicted(younger) && !run.evicted(older));
    run.to_idle();
    for (r, n) in [(older, 380), (younger, 380), (i, 300)] {
        assert_eq!(run.tokens(r), expected(&compute, r, n), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

// ── AC 34 (fixed): the entry rule ───────────────────────────────────────────

#[test]
fn a_moved_agent_comes_back_before_a_later_agent_newcomer_that_would_fit_sooner() {
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(pool(4), compute.clone()));
    let a = run.submit(live(1, 380), RequestClass::Agent); // 24 pages
    run.steps(3);
    let i = run.submit(live(100, 700), RequestClass::Interactive); // 44 pages: A out
    run.step();
    assert!(run.evicted(a));

    // N would fit in the 20 pages I leaves, now: it waits for A all the same.
    let n = run.submit(live(200, 300), RequestClass::Agent); // 19 pages
    while run.state(a) == Some(RequestState::Evicted) {
        assert_eq!(run.state(n), Some(RequestState::Admitted), "N waits while A, which outranks it, is off the device");
        run.step();
    }
    let restored = run.first(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a));
    let entered = run.first(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == n));
    assert!(restored.is_some());
    assert!(entered.is_none() || entered >= restored, "A came back before N entered");
    run.to_idle();
    for (r, max) in [(a, 380), (i, 700), (n, 300)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_restore_never_moves_anything_and_waits_for_its_whole_reservation() {
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(pool(4), compute.clone()));
    let a = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    run.steps(3);
    let i1 = run.submit(live(100, 420), RequestClass::Interactive); // 27 pages: A out
    run.step();
    assert!(run.evicted(a));
    // I2 outranks A: it enters into the room I1 left, before A is back.
    let i2 = run.submit(live(200, 500), RequestClass::Interactive); // 32 pages
    run.steps(2);
    assert_eq!(run.state(i2), Some(RequestState::Running));

    // I1 ends: its 27 pages and the 5 free are short of A's 38. A waits for
    // free room and moves nothing to make it.
    run.until(|run| run.done(i1));
    let moves = |run: &Run| run.events.iter().filter(|(_, e)| matches!(e, SchedEvent::Evicted { .. })).count();
    let before = moves(&run);
    while !run.done(i2) {
        assert_eq!(run.state(a), Some(RequestState::Evicted), "A is not back while its reservation does not fit");
        run.step();
    }
    assert_eq!(moves(&run), before, "nothing was moved for A's restore");
    run.to_idle();
    for (r, max) in [(a, 600), (i1, 420), (i2, 500)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn restores_are_taken_in_rank_order() {
    // Two Agents moved out by one Interactive need. While it runs, the room
    // it leaves would take the younger and not the older: the younger waits
    // all the same, and comes back with the older once there is room for
    // both.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(pool(4), compute.clone()));
    let older = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    let younger = run.submit(live(100, 380), RequestClass::Agent); // 24 pages
    run.steps(3);
    let i = run.submit(live(200, 600), RequestClass::Interactive); // 38 pages: both out
    run.step();
    assert!(run.evicted(older) && run.evicted(younger));
    while !run.done(i) {
        assert_eq!(
            run.state(younger),
            Some(RequestState::Evicted),
            "the 26 pages beside I would take the younger, which waits for the older"
        );
        run.step();
    }
    run.to_idle();
    let at = |r: RequestId| run.first(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == r));
    assert!(at(older).is_some() && at(older) <= at(younger), "the older Agent came back first");
    for (r, max) in [(older, 600), (younger, 380), (i, 600)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    run.assert_no_work_lost();
}

#[test]
fn rank_order_holds_across_kv_ram_and_the_disk() {
    // The older Agent's blob is too large for KV-RAM and goes to the disk;
    // the younger's lands in KV-RAM, a synchronous restore away. While the
    // Interactive need runs, the younger would fit and the older would not:
    // the younger waits for the older, whichever tier each is in.
    let compute = Arc::new(
        MockCompute::with_sections(MockSections { image_bytes: 0, bytes_per_token: 1 })
            .with_disk(FakeDisk::with_room(1 << 30)),
    );
    let mut run = Run::new(ConcreteScheduler::with_config(
        SchedulerConfig {
            host_capacity_bytes: 100,
            kv_disk_capacity_bytes: 4 * disk_file_bytes(1_000),
            ..pool(0)
        },
        compute.clone(),
    ));
    let older = run.submit(input(tokens(1, 300), Some(300)), RequestClass::Agent); // 38 pages, a ~300-byte blob
    let younger = run.submit(live(1_000, 380), RequestClass::Agent); // 24 pages, a ~10-byte blob
    run.steps(3);
    let i = run.submit(live(2_000, 600), RequestClass::Interactive); // 38 pages: both out
    run.until(|run| run.has(|e| matches!(e, SchedEvent::DiskSpilled { request, .. } if *request == older)));
    assert!(run.evicted(younger), "the younger went to KV-RAM");
    while !run.done(i) {
        assert_eq!(run.state(younger), Some(RequestState::Evicted), "the younger waits for the older");
        run.step();
    }
    run.to_idle();
    for (r, max) in [(older, 300), (younger, 380), (i, 600)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    run.assert_no_work_lost();
}

#[test]
fn a_victim_already_on_the_disk_does_not_come_back_into_the_room_it_left() {
    // An Interactive need two Agents have to leave, one spill at a time. The
    // first is on the disk while the second's spill runs: its pages are free,
    // and it would fit back into them, but the need it left for outranks it.
    // Without that, the two would trade places on the disk forever.
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut run = Run::new(ConcreteScheduler::with_config(
        SchedulerConfig { kv_disk_capacity_bytes: 4 * disk_file_bytes(1), ..pool(0) },
        compute.clone(),
    ));
    let a1 = run.submit(live(1, 380), RequestClass::Agent); // 24 pages
    let a2 = run.submit(live(100, 380), RequestClass::Agent); // 24 pages
    run.steps(3);
    let i = run.submit(live(200, 700), RequestClass::Interactive); // 44 pages: both out
    run.until(|run| run.has(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == i)));
    let entered = run.first(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == i)).unwrap();
    for a in [a1, a2] {
        assert!(
            run.has(|e| matches!(e, SchedEvent::DiskSpilled { request, .. } if *request == a)),
            "{a} went to the disk"
        );
        assert!(
            !run.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a)),
            "{a} did not come back before I entered (advance {entered})"
        );
    }
    run.to_idle();
    let spills = run.events.iter().filter(|(_, e)| matches!(e, SchedEvent::DiskSpilled { .. })).count();
    assert_eq!(spills, 2, "each went to the disk once");
    for (r, max) in [(a1, 380), (a2, 380), (i, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    run.assert_no_work_lost();
}

#[test]
fn a_need_the_eligible_victims_cannot_cover_moves_nothing() {
    // An older Interactive and an Agent hold 57 of 64 pages. A newer
    // Interactive needs 38: the Agent's 19 and the 7 free fall short, and the
    // older Interactive ranks above it. Nothing moves -- the Agent would only
    // come back into its own room -- and the newcomer waits for the older one
    // to finish.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(pool(4), compute.clone()));
    let j = run.submit(live(1, 600), RequestClass::Interactive); // 38 pages
    let a = run.submit(live(100, 300), RequestClass::Agent); // 19 pages
    run.steps(3);
    let i = run.submit(live(200, 600), RequestClass::Interactive); // 38 pages
    run.until(|run| run.done(j));
    assert!(!run.moved_any(), "nothing moved for a need it could not meet");
    run.to_idle();
    for (r, max) in [(j, 600), (a, 300), (i, 600)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    assert!(!run.moved_any());
    run.assert_no_work_lost();
}

// ── AC 33 (fixed) and AC 35 (fixed) ─────────────────────────────────────────

#[test]
fn nothing_but_an_admission_moves_a_sequence() {
    // Two Agents fill 62 of 64 pages and decode to their ends; nothing else
    // arrives, so nothing moves, and every round takes both lanes (the
    // per-advance check in `Run::step`).
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut run = Run::new(ConcreteScheduler::with_config(
        SchedulerConfig { kv_disk_capacity_bytes: 1 << 20, ..pool(4) },
        compute.clone(),
    ));
    let a = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    let b = run.submit(live(100, 380), RequestClass::Agent); // 24 pages
    run.to_idle();
    assert!(!run.moved_any(), "no admission, no move");
    let lengths: HashMap<RequestId, usize> = [a, b].into_iter().map(|r| (r, run.tokens(r).len())).collect();
    assert_eq!(lengths[&a], 600);
    assert_eq!(lengths[&b], 380);
}
