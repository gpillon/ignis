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

use ignis_core::checkpoint::ReuseSource;
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

    /// [`Run::to_idle`] within `bound` advances of the run's start, so that a
    /// run that would never end fails at once, naming what still waits.
    fn to_idle_within(&mut self, bound: usize) {
        while !self.sched.is_idle() {
            let waiting: Vec<(RequestId, Option<RequestState>)> = self
                .tracked
                .iter()
                .map(|&r| (r, self.state(r)))
                .filter(|(_, state)| *state != Some(RequestState::Done))
                .collect();
            assert!(
                self.advances < bound,
                "the run is not idle after {bound} advances: {waiting:?} still wait, {} pages in use",
                self.sched.kv_used_pages()
            );
            self.step();
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

// ── GitHub #444: a move counts the pages it leaves to retained state alone ──

/// [`pool`] with prompt reuse on, and retained slots for the checkpoints and
/// prefixes the turns below leave.
fn reuse_pool() -> SchedulerConfig {
    SchedulerConfig { prompt_reuse: true, retained_slots: 4, ..pool(4) }
}

/// An Agent conversation's turn: 70 prompt tokens whose generation opener ends
/// at 66, so its checkpoint stands on the four whole pages below the opener,
/// and 300 tokens to generate: 24 pages in all.
fn agent_turn() -> RequestInput {
    let mut turn = input(tokens(1, 70), Some(300));
    turn.opener_tokens = Some(66);
    turn
}

/// [`agent_turn`] with a picture whose placeholders run from token 4 to 64,
/// across its opener's page floor at 64. No claimable prefix may end inside
/// it (#193), so the turn's capture lends the pages below the opener to a
/// pages-only link (ADR 0029, #306; on the 27B since #249) instead of
/// publishing them.
fn agent_image_turn() -> RequestInput {
    use ignis_core::vision::{Grid, MediaItem, Multimodal, TokenSpan};
    let mut turn = agent_turn();
    turn.multimodal = Some(Arc::new(Multimodal {
        positions: (0..3).flat_map(|_| 0..70).collect(),
        rope_delta: 0,
        media: vec![MediaItem {
            grid: Grid { t: 1, h: 2, w: 2 * 61 },
            token_span: TokenSpan { begin: 4, count: 61 },
            patches: Vec::new(),
            content_digest: [0xAA; 32],
        }],
    }));
    turn
}

/// The turn's prompt up to its opener's page floor, then a question of its
/// own: a sibling that claims the head the turn published.
fn sibling_of_agent_turn(max: u32) -> RequestInput {
    input(tokens(1, 64).into_iter().chain(900..906).collect(), Some(max))
}

#[test]
fn an_interactive_need_moves_an_agent_that_published_its_prefix_in_the_same_advance() {
    // A's system block, two pages, is a retained prefix; the two pages above
    // it, up to its opener's page floor, are a prefix chained over it, which
    // A's checkpoint stands on. A's own reservation is the 20 pages left. I
    // needs 63 of 64: A's 20 fall short, and A's 20 with its 4 published
    // pages do not. Moving A leaves the chained prefix to nothing and the
    // system block to its retention alone, so both come back -- the retention
    // given up for I, as retained state always is -- and I takes its pages in
    // the advance A moves.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let mut turn = agent_turn();
    turn.system_block_tokens = Some(2 * PAGE);
    let a = run.submit(turn, RequestClass::Agent);
    run.steps(4);
    assert!(!run.tokens(a).is_empty(), "A decodes alone first");
    assert_eq!(run.sched.prefix_pinned_pages(), 4, "A published its system block and the pages up to its floor");

    let i = run.submit(live(500, 1000), RequestClass::Interactive); // 63 pages
    run.step();
    assert!(run.evicted(a), "A left the device for KV-RAM, its published pages with it");
    assert_eq!(run.state(i), Some(RequestState::Running), "I took its pages in the same advance");
    run.to_idle();
    for (r, max) in [(a, 300), (i, 1000)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn an_interactive_need_moves_an_agent_that_lent_its_pages_to_a_link() {
    // A's picture spans its opener's page floor, so its capture lends the four
    // pages below its opener to a pages-only link, and its reservation shrinks
    // to 20 pages. I needs 63 of 64: only A's 20 and its 4 lent pages cover it.
    //
    // On a load whose opener's page rides the capture (Flash-Next, #306) the
    // turn lends on `main`. The 27B leg lends nothing before #249: the turn
    // publishes nothing and is moved for its own 24 pages, as on `main`, so
    // it starts to bite only once #249 lands and the 27B turn lends too.
    for rides_capture in [true, false] {
        let compute = Arc::new(MockCompute::new());
        let mut run = Run::new(ConcreteScheduler::with_config(
            SchedulerConfig { opener_page_rides_capture: rides_capture, ..reuse_pool() },
            compute.clone(),
        ));
        let a = run.submit(agent_image_turn(), RequestClass::Agent);
        run.steps(4);
        assert!(!run.tokens(a).is_empty(), "A decodes alone first");
        if rides_capture {
            assert_eq!(run.sched.prefix_pinned_pages(), 4, "A lent the four pages below its opener to a link");
        }

        let i = run.submit(live(500, 1000), RequestClass::Interactive); // 63 pages
        run.step();
        assert!(run.evicted(a), "rides the capture: {rides_capture}: A left the device, its lent pages with it");
        assert_eq!(run.state(i), Some(RequestState::Running), "I took its pages in the same advance");
        run.to_idle();
        for (r, max) in [(a, 300), (i, 1000)] {
            assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
        }
        run.assert_no_work_lost();
    }
}

#[test]
fn a_move_counts_no_page_a_live_claimant_stands_on() {
    // S, an Interactive request older than I, claimed the four pages A
    // published and decodes on them, with three pages of its own above them;
    // it ends long before A does. I needs 61 pages beside A's 20, S's 3 and
    // the 4 they share: moving A would free its 20 and leave the 4 to S,
    // short of I's need. Nothing moves until S is done; then A's 20 and the 4
    // it alone holds cover I, its 20 alone do not, and A moves.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let a = run.submit(agent_turn(), RequestClass::Agent);
    run.steps(4);
    let s = run.submit(sibling_of_agent_turn(40), RequestClass::Interactive); // 7 pages, 4 of them A's
    run.steps(2);
    assert_eq!(run.sched.shared_prefix_of(s).map(|(_, tokens)| tokens), Some(64), "S stands on A's head");

    let i = run.submit(live(500, 960), RequestClass::Interactive); // 61 pages
    run.until(|run| run.done(s));
    assert!(!run.moved_any(), "A did not move while S stood on its pages");
    assert_eq!(run.state(i), Some(RequestState::Admitted), "I waited");
    run.step();
    assert!(run.evicted(a), "once S was done, A moved for I");
    assert_eq!(run.state(i), Some(RequestState::Running), "and I took its pages in the same advance");
    run.to_idle();
    for (r, max) in [(a, 300), (s, 40), (i, 960)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_move_counts_the_pages_a_claim_waiting_below_the_need_holds() {
    // R continues A's conversation: it claimed A's checkpoint, which holds the
    // four pages under it and its tail page for R's first chunk, and R waits
    // for room behind A, which outranks it. I needs 63 of 64: A's 20 fall
    // short, and so do A's 20 and the four while R's claim keeps them. That
    // claim is a bet R has not cashed yet, and R ranks below I, so it goes as
    // retained state goes (GitHub #445): I moves A, and the four and the tail
    // page come back with R's claim and the checkpoint, in the advance I
    // arrives. R prefills its whole prompt later. Until #445 this test
    // cancelled R to get I in: without the cancel, nothing ever ran again.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let a = run.submit(agent_turn(), RequestClass::Agent);
    run.steps(4);
    // 45 pages past the 4 it shares with A's checkpoint: they do not fit beside
    // A's 25.
    let r = run.submit(next_turn(), RequestClass::Agent);
    run.step();
    assert_eq!(run.state(r), Some(RequestState::Admitted), "R waits for room");
    assert!(run.sched.shared_prefix_of(r).is_some(), "R holds the pages under A's checkpoint");

    let i = run.submit(live(500, 1000), RequestClass::Interactive); // 63 pages
    run.step();
    assert!(run.evicted(a), "A moved for I");
    assert_eq!(run.state(i), Some(RequestState::Running), "and I took its pages in the same advance");
    assert!(run.sched.shared_prefix_of(r).is_none(), "R's claim went for them");
    run.to_idle_within(3_000);
    for (r, max) in [(a, 300), (i, 1000), (r, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

// ── GitHub #445: a claim a waiting request holds yields to a need above it ──

/// The next turn of [`agent_turn`]'s conversation: its prompt up to the
/// opener, then ten tokens of its own, so it claims the turn's checkpoint.
/// 776 tokens in all: 49 pages, 45 past the four it shares with the turn.
fn next_turn() -> RequestInput {
    input(tokens(1, 66).into_iter().chain(700..710).collect(), Some(700))
}

#[test]
fn a_waiting_claim_yields_once_it_is_what_keeps_a_need_out() {
    // A is an Interactive turn; R continues its conversation as an Agent: it
    // claimed A's checkpoint, which holds the four pages under it and its tail
    // page for R's first chunk, and it waits for room. I, Interactive and
    // after A, needs 63 of 64 and can move nothing, since A outranks it.
    // While A decodes, giving R's claim up would free the tail page alone,
    // short of I's need, so it stays. Once A is done, R's claim is all that
    // keeps I out: it ranks below I and goes as retained state goes, and I
    // enters in the next advance. R prefills its whole prompt later.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let a = run.submit(agent_turn(), RequestClass::Interactive);
    run.steps(4);
    let r = run.submit(next_turn(), RequestClass::Agent);
    run.step();
    assert_eq!(run.state(r), Some(RequestState::Admitted), "R waits for room");
    assert!(run.sched.shared_prefix_of(r).is_some(), "R holds the pages under A's checkpoint");

    let i = run.submit(live(500, 1000), RequestClass::Interactive); // 63 pages
    run.until(|run| run.done(a));
    assert_eq!(run.state(i), Some(RequestState::Admitted), "I waited for A, which it cannot move");
    assert!(run.sched.shared_prefix_of(r).is_some(), "R's claim stayed while it could not make I's room");
    assert_eq!(run.sched.retained_tail_pages(), 1, "and so did the checkpoint's tail page");
    run.step();
    assert_eq!(run.state(i), Some(RequestState::Running), "I took its pages in the next advance");
    assert!(run.sched.shared_prefix_of(r).is_none(), "R's claim went for them");
    assert!(!run.moved_any(), "nothing moved");
    run.to_idle_within(3_000);
    for (r, max) in [(a, 300), (i, 1000), (r, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_waiting_claim_yields_to_a_moved_sequence_it_keeps_out() {
    // The entry rule's side. I's need moved M, an Agent of 60 pages, to
    // KV-RAM. P, an Interactive turn, then left a checkpoint, and R, an Agent
    // continuing P's conversation and ranked below M, claimed it: R waits
    // behind M, as entries are taken in rank order. Once I and P are done, M
    // needs 60 pages and R's claim keeps five of the 64. It ranks below M and
    // goes as retained state goes, M comes back in the advance I ends, and R
    // prefills its whole prompt once M is done.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let m = run.submit(live(10, 956), RequestClass::Agent); // 60 pages
    run.steps(3);
    let i = run.submit(live(500, 476), RequestClass::Interactive); // 30 pages
    run.step();
    assert!(run.evicted(m), "I moved M");
    let p = run.submit(agent_turn(), RequestClass::Interactive); // 24 pages and a tail page
    run.steps(4);
    let r = run.submit(next_turn(), RequestClass::Agent);
    run.step();
    assert!(run.sched.shared_prefix_of(r).is_some(), "R holds the pages under P's checkpoint");

    run.until(|run| run.done(p));
    assert!(run.sched.shared_prefix_of(r).is_some(), "R's claim stays while I holds the room M needs");
    run.until(|run| run.done(i));
    let restored = run.first(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == m));
    assert_eq!(restored, run.first(|e| matches!(e, SchedEvent::Done { request, .. } if *request == i)), "M came back in the advance I ended");
    assert!(run.sched.shared_prefix_of(r).is_none(), "R's claim went for its room");
    run.to_idle_within(3_000);
    for (r, max) in [(m, 956), (i, 476), (p, 300), (r, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_waiting_claim_yields_to_a_moved_sequence_coming_back_from_the_disk() {
    // As above with no KV-RAM: I's need moved M, an Agent of 60 pages, to the
    // disk. P left a checkpoint and R, ranked below M, claimed it. Once I and
    // P are done, M needs 60 pages and R's claim keeps five of the 64: it goes
    // as retained state goes, and M's restore from the disk starts in the
    // advance I ends. A disk restore lands a window at a time.
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut run = Run::new(ConcreteScheduler::with_config(
        SchedulerConfig { kv_disk_capacity_bytes: 1 << 20, ..SchedulerConfig { prompt_reuse: true, retained_slots: 4, ..pool(0) } },
        compute.clone(),
    ));
    let m = run.submit(live(10, 956), RequestClass::Agent); // 60 pages
    run.steps(3);
    let i = run.submit(live(500, 476), RequestClass::Interactive); // 30 pages
    run.until(|run| run.state(i) == Some(RequestState::Running));
    assert_eq!(compute.disk_spills(), vec![(DiskBlob::Live(m), DiskSource::Device)], "I moved M to the disk");
    let p = run.submit(agent_turn(), RequestClass::Interactive); // 24 pages and a tail page
    run.steps(4);
    let r = run.submit(next_turn(), RequestClass::Agent);
    run.step();
    assert!(run.sched.shared_prefix_of(r).is_some(), "R holds the pages under P's checkpoint, on the device");

    run.until(|run| run.done(p));
    assert!(run.sched.shared_prefix_of(r).is_some(), "R's claim stays while I holds the room M needs");
    assert!(compute.disk_restores().is_empty(), "and M stays on the disk");
    run.until(|run| run.done(i));
    assert_eq!(compute.disk_restores(), vec![DiskBlob::Live(m)], "M's restore started in the advance I ended");
    assert!(run.sched.shared_prefix_of(r).is_none(), "R's claim went for its room");
    run.to_idle_within(3_000);
    for (r, max) in [(m, 956), (i, 476), (p, 300), (r, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_waiting_claim_never_yields_to_an_entry_below_its_claimant() {
    // H, Interactive, decodes on 38 pages to the end. P, an Interactive turn,
    // moved M, an Agent of 3 pages, for its room and left a checkpoint. C, an
    // Interactive request continuing P's conversation, claimed it and waits
    // for H: its 23 pages past the four do not fit beside H once P is done.
    // M's room comes after C's; giving C's claim up would make it, but C
    // outranks M, so the claim stays, and C resumes from P's checkpoint once
    // H is done.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let h = run.submit(live(600, 604), RequestClass::Interactive); // 38 pages
    let m = run.submit(live(10, 40), RequestClass::Agent); // 3 pages
    run.steps(2);
    let p = run.submit(agent_turn(), RequestClass::Interactive); // 24 pages and a tail page
    run.step();
    assert!(run.evicted(m), "P moved M");
    run.steps(4);
    let c = run.submit(input(tokens(1, 66).into_iter().chain(700..710).collect(), Some(356)), RequestClass::Interactive); // 27 pages
    run.step();
    assert_eq!(run.state(c), Some(RequestState::Admitted), "C waits for room");
    assert!(run.sched.shared_prefix_of(c).is_some(), "C holds the pages under P's checkpoint");

    run.until(|run| run.done(p));
    run.steps(10);
    assert!(run.sched.shared_prefix_of(c).is_some(), "C's claim stayed for a sequence below it");
    assert_eq!(run.sched.retained_tail_pages(), 1, "and so did the checkpoint's tail page");
    assert!(!run.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == m)), "M waits behind C");
    run.to_idle_within(3_000);
    assert!(
        run.has(|e| matches!(
            e,
            SchedEvent::StateReused { request, source: ReuseSource::Device, tokens: 66, .. } if *request == c
        )),
        "C resumed from P's checkpoint"
    );
    for (r, max) in [(h, 604), (m, 40), (p, 300), (c, 356)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_need_that_fits_takes_no_claim_and_is_not_overtaken() {
    // A, an Interactive turn, decodes beside its checkpoint, and R, an Agent
    // continuing its conversation, claimed it and waits: its 45 pages past
    // the four do not fit beside A's 25. I, Interactive, needs 30 of the 39
    // free: it enters at once, ahead of R, and R keeps its claim. R enters
    // once A and I are done, resuming from A's checkpoint.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let a = run.submit(agent_turn(), RequestClass::Interactive);
    run.steps(4);
    let r = run.submit(next_turn(), RequestClass::Agent);
    run.step();
    let i = run.submit(live(500, 476), RequestClass::Interactive); // 30 pages
    run.step();
    assert_eq!(run.state(i), Some(RequestState::Running), "I entered at once");
    assert_eq!(run.state(r), Some(RequestState::Admitted), "R still waits");
    assert!(run.sched.shared_prefix_of(r).is_some(), "with its claim");
    assert_eq!(run.sched.retained_tail_pages(), 1, "and the checkpoint's tail page");
    run.to_idle_within(3_000);
    let entered = |x: RequestId| run.first(|e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == x));
    assert!(entered(i) < entered(r), "R did not enter before I");
    assert!(
        run.has(|e| matches!(
            e,
            SchedEvent::StateReused { request, source: ReuseSource::Device, tokens: 66, .. } if *request == r
        )),
        "R resumed from A's checkpoint"
    );
    assert!(!run.moved_any(), "nothing moved");
    for (r, max) in [(a, 300), (i, 476), (r, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_waiting_claim_on_a_retained_prefix_yields_too() {
    // A, an Interactive turn, published its two-page system block as a
    // retained prefix, and the two pages above it, to its opener's floor, as a
    // prefix chained over the block, which its checkpoint stands on. R, an
    // Agent opening with the same system block and nothing more of A's,
    // claimed the block and waits. I, Interactive and after A, needs 63 of
    // the 64 pages and cannot move A. Once A is done the checkpoint and the
    // link above the block go for I, which is then one page short: the
    // block's two pages, which R's claim keeps beside its retention. R's
    // claim goes, the retention with it, and I enters.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let mut turn = agent_turn();
    turn.system_block_tokens = Some(2 * PAGE);
    let a = run.submit(turn, RequestClass::Interactive);
    run.steps(4);
    // 782 tokens: 49 pages, 47 past the block.
    let r = run.submit(input(tokens(1, 32).into_iter().chain(800..850).collect(), Some(700)), RequestClass::Agent);
    run.step();
    assert_eq!(run.state(r), Some(RequestState::Admitted), "R waits for room");
    assert_eq!(run.sched.shared_prefix_of(r).map(|(_, tokens)| tokens), Some(32), "R stands on A's system block");

    let i = run.submit(live(500, 1000), RequestClass::Interactive); // 63 pages
    run.until(|run| run.done(a));
    assert_eq!(run.state(i), Some(RequestState::Admitted), "I waited for A");
    run.step();
    assert_eq!(run.state(i), Some(RequestState::Running), "I took its pages in the next advance");
    assert!(run.sched.shared_prefix_of(r).is_none(), "R's claim on the block went for them");
    assert_eq!(run.sched.prefix_pinned_pages(), 0, "and so did the block");
    run.to_idle_within(3_000);
    assert!(
        !run.has(|e| matches!(e, SchedEvent::PrefixReused { request, .. } if *request == r)),
        "R prefilled from its first token and reports no reuse of the block it let go of"
    );
    for (r, max) in [(a, 300), (i, 1000), (r, 700)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

/// [`agent_turn`], opening with a two-page system block it publishes as a
/// retained prefix; the two pages above it, to its opener's floor, become a
/// prefix chained over the block, which its checkpoint stands on.
fn agent_turn_with_block() -> RequestInput {
    let mut turn = agent_turn();
    turn.system_block_tokens = Some(2 * PAGE);
    turn
}

/// An Agent's prompt opening with [`agent_turn_with_block`]'s block and
/// nothing more of it, then 50 tokens of its own from `own`: 782 tokens, 49
/// pages, 47 past the block.
fn block_then(own: u32) -> RequestInput {
    input(tokens(1, 32).into_iter().chain(own..own + 50).collect(), Some(700))
}

#[test]
fn a_waiting_claim_whose_going_alone_frees_nothing_stays() {
    // A, an Interactive turn, left its block, the link over it and its
    // checkpoint. K, an Agent continuing A's conversation, claimed the
    // checkpoint; J, an Agent after K, claimed the block. Both wait. I,
    // Interactive, needs 60 of the 64 pages and cannot move A. Once A is done
    // I is one page short. K's claim, let go, gives back the checkpoint's tail
    // page and the link under it; J's gives back nothing while K's checkpoint
    // chains over the block. J ranks lowest, but only K's claim goes, and I
    // enters. J resumes from the block once I is done.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let a = run.submit(agent_turn_with_block(), RequestClass::Interactive);
    run.steps(4);
    let k = run.submit(next_turn(), RequestClass::Agent);
    run.step();
    let j = run.submit(block_then(800), RequestClass::Agent);
    run.step();
    let (k_link, j_block) = (run.sched.shared_prefix_of(k), run.sched.shared_prefix_of(j));
    assert!(k_link.is_some(), "K stands on the link under A's checkpoint");
    assert_eq!(j_block.map(|(_, tokens)| tokens), Some(32), "J stands on A's block");
    assert_ne!(k_link.map(|(link, _)| link), j_block.map(|(block, _)| block), "two links, one over the other");

    let i = run.submit(live(500, 956), RequestClass::Interactive); // 60 pages
    run.until(|run| run.done(a));
    assert_eq!(run.state(i), Some(RequestState::Admitted), "I waited for A");
    run.step();
    assert_eq!(run.state(i), Some(RequestState::Running), "I took its pages in the next advance");
    assert!(run.sched.shared_prefix_of(k).is_none(), "K's claim went for them");
    assert_eq!(
        run.sched.shared_prefix_of(j).map(|(_, tokens)| tokens),
        Some(32),
        "J's claim, which would have freed nothing, stayed"
    );
    run.to_idle_within(3_000);
    assert!(
        run.has(|e| matches!(e, SchedEvent::PrefixReused { request, tokens: 32, .. } if *request == j)),
        "J resumed from the block it kept"
    );
    for (r, max) in [(a, 300), (k, 700), (j, 700), (i, 956)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn two_waiting_claims_that_free_a_block_only_together_both_go() {
    // J1 and J2, Agents opening with A's block and nothing more of it, each
    // claimed the block and wait. I, Interactive, needs 63 of the 64 pages and
    // cannot move A. Once A is done the checkpoint and the link over the block
    // go for I, which is then the block's two pages short. Neither claim alone
    // gives the block back, since the other still stands on it; together they
    // do, so both go, J2's first, and I enters.
    let compute = Arc::new(MockCompute::new());
    let mut run = Run::new(ConcreteScheduler::with_config(reuse_pool(), compute.clone()));
    let a = run.submit(agent_turn_with_block(), RequestClass::Interactive);
    run.steps(4);
    let j1 = run.submit(block_then(800), RequestClass::Agent);
    let j2 = run.submit(block_then(900), RequestClass::Agent);
    run.step();
    for j in [j1, j2] {
        assert_eq!(run.sched.shared_prefix_of(j).map(|(_, tokens)| tokens), Some(32), "{j} stands on A's block");
    }

    let i = run.submit(live(500, 1000), RequestClass::Interactive); // 63 pages
    run.until(|run| run.done(a));
    assert_eq!(run.state(i), Some(RequestState::Admitted), "I waited for A");
    run.step();
    assert_eq!(run.state(i), Some(RequestState::Running), "I took its pages in the next advance");
    assert!(run.sched.shared_prefix_of(j1).is_none() && run.sched.shared_prefix_of(j2).is_none(), "both claims went");
    assert_eq!(run.sched.prefix_pinned_pages(), 0, "and the block with them");
    run.to_idle_within(3_000);
    for (r, max) in [(a, 300), (j1, 700), (j2, 700), (i, 1000)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}
