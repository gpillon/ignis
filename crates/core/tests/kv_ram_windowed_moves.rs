//! GitHub #309 (spec vram-budget/03 AC 37): live moves through KV-RAM a window
//! at a time, through the concrete scheduler on a CPU (`MockCompute`, ADR
//! 0006).
//!
//! A synchronous move through KV-RAM held every decoding lane for the whole
//! copy, ~0.32 s for a 1.13 GB Flash-Next blob. The adapter now moves the blob
//! a window at a time between steps, the way KV-disk does, and the scheduler
//! treats such a move as a transfer: the victim keeps its pages, its resident
//! slot and its lane, in no round, until the last window lands, and the
//! request that wanted the room waits for it; a sequence coming back is
//! charged at the start and runs once it has landed. The other lanes decode
//! every advance meanwhile.
//!
//! The mock's windowed moves end a fixed number of advances after they start
//! (`MockCompute::with_windowed_kv_ram`). Its token streams are pure functions
//! of the request, so "its output continues unbroken" is checked against the
//! stream it would have emitted had it never moved; the bytes are the card's
//! to check (`crates/server/tests/live_moves_gpu.rs`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ignis_core::disk::disk_file_bytes;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, RequestState, SchedEvent};
use ignis_core::{ConcreteScheduler, FakeDisk, MockCompute, MockSections, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
const PAGE: u32 = 16;
/// Advances a windowed move takes, start to landing.
const MOVE: u32 = 3;

fn input(prompt: Vec<u32>, max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams { max_tokens: Some(max_tokens), ..DecodeParams::default() },
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

/// A four-token prompt from `start` and `max` tokens to generate:
/// `ceil((4 + max) / 16)` pages.
fn live(start: u32, max: u32) -> RequestInput {
    input((start..start + 4).collect(), max)
}

/// A pool of 64 pages -- one 1,024-token sequence -- with KV-RAM for `host`
/// nominal blobs and no disk, prompt reuse off.
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

/// A mock whose KV-RAM is an arena of `host` nominal blobs and whose moves
/// through it take [`MOVE`] advances.
fn windowed(host: u64) -> Arc<MockCompute> {
    Arc::new(MockCompute::with_host_arena(host).with_windowed_kv_ram(MOVE))
}

/// A clock the test moves by hand, in milliseconds.
fn manual_clock() -> (ignis_core::Clock, Arc<AtomicU64>) {
    let base = Instant::now();
    let offset = Arc::new(AtomicU64::new(0));
    let read = offset.clone();
    (Arc::new(move || base + Duration::from_millis(read.load(Ordering::SeqCst))), offset)
}

/// Every event of a run, the advance each came in, and two checks on every
/// advance: KV-RAM's ledger agrees with the arena's spans (when the mock
/// models one), and a request `Running` before and after an advance that is
/// not mid-move decoded in it (AC 35, fixed bullet).
struct Run {
    sched: ConcreteScheduler,
    compute: Arc<MockCompute>,
    arena: bool,
    tracked: Vec<RequestId>,
    events: Vec<(usize, SchedEvent)>,
    advances: usize,
}

impl Run {
    fn new(config: SchedulerConfig, compute: Arc<MockCompute>) -> Self {
        let sched = ConcreteScheduler::with_config(config, compute.clone());
        Self { sched, compute, arena: true, tracked: Vec::new(), events: Vec::new(), advances: 0 }
    }

    /// A run over a mock that models no arena.
    fn without_arena(config: SchedulerConfig, compute: Arc<MockCompute>) -> Self {
        Self { arena: false, ..Self::new(config, compute) }
    }

    fn submit(&mut self, input: RequestInput, class: RequestClass) -> RequestId {
        let id = self.sched.submit(input, class).expect("admitted to the queue");
        self.tracked.push(id);
        id
    }

    fn running(&self) -> Vec<RequestId> {
        self.tracked
            .iter()
            .copied()
            .filter(|&r| self.state(r) == Some(RequestState::Running) && !self.sched.in_transfer(r))
            .collect()
    }

    fn step(&mut self) -> Vec<SchedEvent> {
        let before = self.running();
        let out = self.sched.advance();
        for r in before {
            let left = out.iter().any(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == r));
            if self.running().contains(&r) && !left {
                assert!(
                    out.iter().any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == r)),
                    "request {r} stayed on its lane through advance {} and decoded nothing",
                    self.advances
                );
            }
        }
        if self.arena {
            assert_eq!(
                self.sched.host_tier().used_bytes(),
                self.compute.host_arena_used(),
                "KV-RAM's ledger holds what the arena's spans hold, after advance {}",
                self.advances
            );
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

    fn evicted(&self, r: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == r))
    }

    fn restored(&self, r: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == r))
    }

    fn done(&self, r: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Done { request, .. } if *request == r))
    }

    fn decoded_in(&self, r: RequestId, advance: usize) -> bool {
        self.events
            .iter()
            .any(|(at, e)| *at == advance && matches!(e, SchedEvent::Token { request, .. } if *request == r))
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

/// An Agent C (30 pages) and an Interactive B (24 pages) decode; an
/// Interactive E (13 pages) needs C out, and C fits back only once E has
/// ended, while B still decodes. Returns them after the advance that started
/// C's move.
fn c_moves_out_for_e(run: &mut Run) -> (RequestId, RequestId, RequestId) {
    let c = run.submit(live(1, 476), RequestClass::Agent);
    let b = run.submit(live(100, 380), RequestClass::Interactive);
    run.steps(3);
    assert!(run.decoded_in(c, 2) && run.decoded_in(b, 2), "C and B decode together first");
    let e = run.submit(live(200, 204), RequestClass::Interactive);
    run.step();
    assert!(run.sched.in_transfer(c), "E's admission started C's move");
    (c, b, e)
}

const C_TOKENS: u32 = 476;
const B_TOKENS: u32 = 380;
const E_TOKENS: u32 = 204;

// ── a move out ──────────────────────────────────────────────────────────────

#[test]
fn a_windowed_move_out_keeps_the_victim_charged_until_its_last_window_lands() {
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let c = run.submit(live(1, 600), RequestClass::Agent);
    let b = run.submit(live(100, 100), RequestClass::Interactive);
    run.steps(3);
    assert_eq!(run.sched.kv_used_pages(), 38 + 7);
    let e = run.submit(live(200, 400), RequestClass::Interactive);
    run.step();
    let started = run.advances - 1;
    assert_eq!(compute.kv_ram_moves_started(), vec![(c, true)], "E's admission moved C, and only C");
    assert!(run.sched.transfer_busy() && !run.sched.disk_busy());

    // The move takes MOVE advances: C keeps its 38 pages and its lane, and E
    // waits for the room, while B decodes in every one of them.
    while !run.evicted(c) {
        assert_eq!(run.sched.kv_used_pages(), 38 + 7, "C's pages stay charged until it has landed");
        assert_eq!(run.state(c), Some(RequestState::Running), "C keeps its lane");
        assert_eq!(run.state(e), Some(RequestState::Admitted), "E waits for the room");
        assert_eq!(run.sched.host_tier().used_bytes(), 1, "C's blob is KV-RAM's from the start");
        run.step();
        assert!(run.decoded_in(b, run.advances - 1), "B decodes through the move");
        assert!(!run.decoded_in(c, run.advances - 1), "C, mid-move, is in no round");
    }
    let landed = run.first(|ev| matches!(ev, SchedEvent::Evicted { request, .. } if *request == c)).unwrap();
    assert_eq!(landed - started, MOVE as usize, "C left the device when its last window landed");
    assert_eq!(run.state(c), Some(RequestState::Evicted));
    assert!(!run.sched.in_transfer(c));
    assert_eq!(
        run.first(|ev| matches!(ev, SchedEvent::PrefillChunk { request, .. } if *request == e)),
        Some(landed),
        "E took the room in the advance C's move landed in"
    );
    run.to_idle();
    for (r, max) in [(c, 600), (b, 100), (e, 400)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
    assert_eq!(run.sched.kv_used_pages(), 0);
    assert_eq!(compute.kv_ram_moves(), 0);
}

#[test]
fn one_move_off_the_device_at_a_time() {
    // Two Agents have to leave for one Interactive need: the second starts
    // only once the first has landed.
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let older = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    let younger = run.submit(live(100, 380), RequestClass::Agent); // 24 pages
    run.steps(3);
    let i = run.submit(live(200, 600), RequestClass::Interactive); // 38 pages: both out
    run.until(|run| run.evicted(older) && run.evicted(younger));
    let first = run.first(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == younger)).unwrap();
    let second = run.first(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == older)).unwrap();
    assert_eq!(second - first, MOVE as usize, "the older one's move started when the younger's landed");
    assert_eq!(compute.kv_ram_moves_started(), vec![(younger, true), (older, true)], "the youngest Agent first");
    run.to_idle();
    for (r, max) in [(older, 600), (younger, 380), (i, 600)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_lane_less_victim_moves_out_and_comes_back_prefilling_from_its_progress() {
    // Eight Interactive lanes hold every lane; an Agent P finishes its prompt
    // and waits for one; an Interactive need moves P, the Agent not
    // decoding, a window at a time. KV-RAM takes P's blob and no lane
    // holder's, nor does the disk, so the head's lane deal cannot move one
    // for P (`live_moves.rs`, its synchronous twin).
    let compute = Arc::new(
        MockCompute::with_sections(MockSections { image_bytes: 0, bytes_per_token: 100 })
            .with_disk(FakeDisk::with_room(1 << 30))
            .with_windowed_kv_ram(MOVE),
    );
    let mut run = Run::without_arena(
        SchedulerConfig {
            max_in_flight: 16,
            resident_slot_capacity: 16,
            kv_capacity_pages: 80,
            kv_disk_capacity_bytes: disk_file_bytes(4_000),
            ..pool(1_000)
        },
        compute.clone(),
    );
    let lanes: Vec<RequestId> =
        (0..8).map(|n| run.submit(input((1000 + 100 * n..1040 + 100 * n).collect(), 100), RequestClass::Interactive)).collect();
    run.step();
    let p = run.submit(live(5000, 12), RequestClass::Agent);
    run.steps(2);
    assert_eq!(run.state(p), Some(RequestState::Prefilling));
    let i = run.submit(live(6000, 120), RequestClass::Interactive);
    run.until(|run| run.evicted(p));
    assert_eq!(compute.kv_ram_moves_started(), vec![(p, true)], "P, and only P, left the device, for KV-RAM");
    run.until(|run| run.restored(p));
    assert!(
        run.has(|e| matches!(e, SchedEvent::Restored { request, lane: None, .. } if *request == p)),
        "a lane-less victim comes back without one"
    );
    assert_eq!(run.sched.prefill_progress(p), Some(4), "from its progress, not from zero");
    run.to_idle();
    assert_eq!(run.tokens(p), expected(&compute, p, 12), "P's output continues unbroken");
    for &l in &lanes {
        assert!(run.done(l));
    }
    assert!(run.done(i));
    run.assert_no_work_lost();
}

// ── a move in ───────────────────────────────────────────────────────────────

#[test]
fn a_windowed_move_in_is_charged_at_its_start_and_runs_once_it_has_landed() {
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let (c, b, e) = c_moves_out_for_e(&mut run);
    run.until(|run| run.done(e));
    // E ended: its room is C's again, and C comes back a window at a time.
    let ended = run.advances - 1;
    assert!(run.sched.in_transfer(c), "C's move in started in the advance E ended in");
    assert_eq!(compute.kv_ram_moves_started().last(), Some(&(c, false)));
    assert_eq!(run.sched.kv_used_pages(), 30 + 24, "C is charged from the start of its move in");
    while !run.restored(c) {
        assert_eq!(run.state(c), Some(RequestState::Evicted), "not schedulable before its last window lands");
        run.step();
        assert!(run.decoded_in(b, run.advances - 1), "B decodes through the move in");
        if !run.restored(c) {
            assert!(!run.decoded_in(c, run.advances - 1), "C is in no round while it comes back");
        }
    }
    let landed = run.first(|ev| matches!(ev, SchedEvent::Restored { request, .. } if *request == c)).unwrap();
    assert_eq!(landed - ended, MOVE as usize);
    assert!(
        run.has(|ev| matches!(ev, SchedEvent::Restored { request, lane: Some(_), .. } if *request == c)),
        "a victim that held a lane comes back on one"
    );
    assert!(run.decoded_in(c, landed), "C landed at the top of an advance, and decoded in its round");
    run.to_idle();
    for (r, max) in [(c, C_TOKENS), (b, B_TOKENS), (e, E_TOKENS)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
    assert_eq!(run.sched.host_tier().used_bytes(), 0);
}

#[test]
fn restores_come_back_one_at_a_time_in_rank_order() {
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let older = run.submit(live(1, 600), RequestClass::Agent); // 38 pages
    let younger = run.submit(live(100, 380), RequestClass::Agent); // 24 pages
    run.steps(3);
    let i = run.submit(live(200, 600), RequestClass::Interactive); // 38 pages: both out
    run.until(|run| run.done(i));
    while !(run.restored(older) && run.restored(younger)) {
        run.step();
        assert!(compute.kv_ram_moves() <= 1, "one move onto the device at a time");
    }
    let at = |r: RequestId| run.first(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == r)).unwrap();
    assert!(at(older) < at(younger), "the older Agent came back first");
    run.to_idle();
    for (r, max) in [(older, 600), (younger, 380), (i, 600)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    run.assert_no_work_lost();
}

#[test]
fn a_move_in_from_kv_ram_waits_while_one_from_the_disk_is_under_way() {
    // The older Agent's blob is too large for KV-RAM and goes to the disk;
    // the younger's lands in KV-RAM. When room returns, the older comes back
    // first, and the younger's move in starts only once it has landed: two
    // moves never share the link beside the expert stream.
    let compute = Arc::new(
        MockCompute::with_sections(MockSections { image_bytes: 0, bytes_per_token: 1 })
            .with_disk(FakeDisk {
                room_bytes: 1 << 30,
                window_bytes: 64,
                advances_per_window: 1,
            })
            .with_windowed_kv_ram(MOVE),
    );
    let mut run = Run::without_arena(
        SchedulerConfig {
            host_capacity_bytes: 100,
            kv_disk_capacity_bytes: 4 * disk_file_bytes(1_000),
            ..pool(0)
        },
        compute.clone(),
    );
    let older = run.submit(input((1..301).collect(), 300), RequestClass::Agent); // 38 pages, a ~300-byte blob
    let younger = run.submit(live(1_000, 380), RequestClass::Agent); // 24 pages, a ~10-byte blob
    run.steps(3);
    let i = run.submit(live(2_000, 600), RequestClass::Interactive); // 38 pages: both out
    run.until(|run| run.done(i));
    while !(run.restored(older) && run.restored(younger)) {
        run.step();
        assert!(
            !(run.sched.disk_busy() && compute.kv_ram_moves() > 0),
            "a move in from KV-RAM never runs beside one from the disk (advance {})",
            run.advances - 1
        );
    }
    let at = |r: RequestId| run.first(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == r)).unwrap();
    assert!(at(older) < at(younger), "the older Agent, on the disk, came back first");
    run.to_idle();
    for (r, max) in [(older, 300), (younger, 380), (i, 600)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    run.assert_no_work_lost();
}

// ── a cancel mid-move ───────────────────────────────────────────────────────

#[test]
fn a_request_cancelled_mid_move_out_frees_its_span_and_its_pages() {
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let (c, b, e) = c_moves_out_for_e(&mut run);
    assert_eq!(run.sched.host_tier().used_bytes(), 1);
    assert!(run.sched.cancel(c));
    run.step();
    assert_eq!(compute.kv_ram_moves(), 0, "the move was abandoned");
    assert_eq!(run.sched.host_tier().used_bytes(), 0, "KV-RAM gave its bytes back");
    assert_eq!(run.sched.host_tier().entry_count(), 0, "and holds no entry for C");
    assert!(!run.evicted(c));
    assert!(
        run.first(|ev| matches!(ev, SchedEvent::PrefillChunk { request, .. } if *request == e)) == Some(run.advances - 1),
        "E took the room C's cancel gave back, in the same advance"
    );
    assert!(!run.sched.transfer_busy());
    run.to_idle();
    for (r, max) in [(b, B_TOKENS), (e, E_TOKENS)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max));
    }
    assert_eq!(run.sched.kv_used_pages(), 0);
}

#[test]
fn a_request_cancelled_mid_move_in_gives_back_its_lane_its_pages_and_its_snapshot() {
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let (c, b, e) = c_moves_out_for_e(&mut run);
    run.until(|run| run.done(e));
    assert!(run.sched.in_transfer(c), "C is on its way back");
    let charged = run.sched.kv_used_pages();
    assert!(run.sched.cancel(c));
    run.step();
    assert_eq!(compute.kv_ram_moves(), 0, "the move was abandoned");
    assert!(!run.sched.transfer_busy());
    assert_eq!(run.sched.host_tier().entry_count(), 0, "C's snapshot went with it");
    assert_eq!(run.sched.host_tier().used_bytes(), 0);
    assert!(run.sched.kv_used_pages() < charged, "C's pages came back");
    assert!(!run.restored(c));
    run.to_idle();
    assert_eq!(run.tokens(b), expected(&compute, b, B_TOKENS));
    assert_eq!(run.sched.kv_used_pages(), 0);
    // Every lane came back: eight newcomers decode together.
    let all: Vec<RequestId> = (0..8).map(|n| run.submit(live(9_000 + 10 * n, 4), RequestClass::Interactive)).collect();
    run.steps(2);
    assert!(all.iter().all(|&r| run.state(r) == Some(RequestState::Running)), "no lane was lost");
}

// ── a copy that failed ──────────────────────────────────────────────────────

#[test]
fn a_failed_move_out_leaves_the_victim_decoding_and_nothing_lost() {
    let compute = windowed(4);
    compute.fail_kv_ram_moves(true);
    let (clock, millis) = manual_clock();
    let mut run = Run::new(pool(4), compute.clone());
    run.sched = ConcreteScheduler::with_config(pool(4), compute.clone()).with_clock(clock);
    let (c, b, e) = c_moves_out_for_e(&mut run);
    run.steps(MOVE as usize);
    assert!(!run.evicted(c), "the move failed");
    assert!(!run.sched.in_transfer(c), "and did not start again in the advance it failed in");
    assert_eq!(run.sched.host_tier().used_bytes(), 0, "KV-RAM gave its bytes back");
    // For the backoff, nothing moves: C decodes on its lane, E waits.
    for _ in 0..5 {
        run.step();
        assert!(run.decoded_in(c, run.advances - 1), "C decodes while nothing moves");
        assert_eq!(run.state(e), Some(RequestState::Admitted), "E still waits for its room");
    }
    assert_eq!(compute.kv_ram_moves_started(), vec![(c, true)], "no second attempt yet");
    // The backoff ends and the copies work again: the next attempt moves C,
    // and nothing was lost.
    millis.store(2_000, Ordering::SeqCst);
    compute.fail_kv_ram_moves(false);
    run.until(|run| run.evicted(c));
    run.to_idle();
    for (r, max) in [(c, C_TOKENS), (b, B_TOKENS), (e, E_TOKENS)] {
        assert_eq!(run.tokens(r), expected(&compute, r, max), "{r} generated all of it, unbroken");
    }
    run.assert_no_work_lost();
}

#[test]
fn a_failed_move_in_prefills_the_request_again() {
    let compute = windowed(4);
    let mut run = Run::new(pool(4), compute.clone());
    let (c, _b, e) = c_moves_out_for_e(&mut run);
    run.until(|run| run.done(e));
    assert!(run.sched.in_transfer(c));
    compute.fail_kv_ram_moves(true);
    run.steps(MOVE as usize);
    assert!(
        run.has(|ev| matches!(ev, SchedEvent::Requeued { request, .. } if *request == c)),
        "C's snapshot could not come back: it prefills again"
    );
    assert_eq!(run.sched.host_tier().entry_count(), 0, "its snapshot is gone");
    compute.fail_kv_ram_moves(false);
    run.to_idle();
    assert!(run.done(c), "C finished all the same");
    assert_eq!(run.sched.kv_used_pages(), 0);
}
