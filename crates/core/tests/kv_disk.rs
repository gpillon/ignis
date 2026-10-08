//! Spec vram-budget/03 (ADR 0045) — KV-disk, Tier 2, through the concrete
//! scheduler on a CPU: `MockCompute` with a fake disk whose transfers take a
//! set number of advances per window, and fail on demand (ADR 0006).
//!
//! What these hold the scheduler to is the chain and its timing: a device
//! victim goes to KV-RAM when it can, KV-RAM gives a victim to the disk
//! instead of discarding it, a victim KV-RAM cannot take goes straight to the
//! disk, nothing live is ever discarded for room, a request waits when no
//! tier has room, and a transfer is a few advances during which nothing else
//! stops. That the bytes that come back are the right bytes needs the card:
//! `crates/server/tests/kv_disk_gpu.rs`.
//!
//! The mock's token streams are pure functions of the request, so a request
//! whose state went to the disk and came back emits exactly the tokens it
//! would have emitted had it never moved: that is what "its output continues
//! unbroken" is checked against here.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ignis_core::checkpoint::{RetainedKind, RetainedStateOperation, ReuseSource};
use ignis_core::disk::disk_file_bytes;
use ignis_core::scheduler::{DiskBlob, DiskOp, DiskSource};
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, RequestState, SchedEvent, SubmitError};
use ignis_core::{ConcreteScheduler, FakeDisk, MockCompute, MockSections, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
const PAGE: u32 = 16;

fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

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

/// A four-token prompt (from `start`, so prompts differ) and `max` tokens to
/// generate: `ceil((4 + max) / 16)` pages.
fn live(start: u32, max: u32) -> RequestInput {
    input(tokens(start, 4), None, max)
}

/// A pool of 64 pages -- one 1,024-token sequence -- with `host` nominal
/// KV-RAM blobs and room on the disk for `disk_files` one-blob files.
/// Prompt reuse off: these scenarios are about live work, and retained state
/// would only move pages around them.
fn pool(host: u64, disk_files: u64) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_in_flight: 8,
        max_sequence_tokens: 1024,
        kv_capacity_pages: 64,
        kv_page_tokens: PAGE,
        host_capacity_bytes: host,
        kv_disk_capacity_bytes: disk_files * disk_file_bytes(1),
        kv_disk_restore_floor_tokens: 1024,
        prompt_reuse: false,
        retained_slots: 0,
        ..SchedulerConfig::default()
    }
}

/// A clock the test moves by hand.
fn manual_clock() -> (ignis_core::Clock, Arc<AtomicU64>) {
    let base = Instant::now();
    let offset = Arc::new(AtomicU64::new(0));
    let read = offset.clone();
    (
        Arc::new(move || base + Duration::from_millis(read.load(Ordering::SeqCst))),
        offset,
    )
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    let mut steps = 0;
    while !sched.is_idle() {
        events.extend(sched.advance());
        steps += 1;
        assert!(steps < 100_000, "the scheduler never went idle");
    }
    events
}

fn tokens_of(events: &[SchedEvent], request: RequestId) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Token { request: r, token } if *r == request => Some(*token),
            _ => None,
        })
        .collect()
}

fn has(events: &[SchedEvent], f: impl Fn(&SchedEvent) -> bool) -> bool {
    events.iter().any(f)
}

fn count(events: &[SchedEvent], f: impl Fn(&SchedEvent) -> bool) -> usize {
    events.iter().filter(|e| f(e)).count()
}

fn spilled(events: &[SchedEvent], request: RequestId, from: DiskSource) -> bool {
    has(events, |e| matches!(e, SchedEvent::DiskSpilled { request: r, from: f } if *r == request && *f == from))
}

fn restored(events: &[SchedEvent], request: RequestId) -> bool {
    has(events, |e| matches!(e, SchedEvent::Restored { request: r, .. } if *r == request))
}

fn done(events: &[SchedEvent], request: RequestId) -> bool {
    has(events, |e| matches!(e, SchedEvent::Done { request: r, .. } if *r == request))
}

/// What every scenario here holds: with the tier on, no live work is lost --
/// no snapshot dropped, no request re-queued for room (spec AC 17, ADR 0045).
fn assert_no_work_lost(events: &[SchedEvent]) {
    assert!(
        !has(events, |e| matches!(e, SchedEvent::SnapshotDropped { .. })),
        "with the tier on no live snapshot is ever dropped"
    );
    assert!(
        !has(events, |e| matches!(e, SchedEvent::Requeued { .. })),
        "and no request loses its work"
    );
}

/// `request`'s stream, as the mock would emit it had it never moved.
fn expected_stream(compute: &MockCompute, request: RequestId, n: u32) -> Vec<u32> {
    (0..n).map(|step| compute.token_for(request, step)).collect()
}

// ── AC 17: the chain ────────────────────────────────────────────────────────

#[test]
fn a_device_victim_kv_ram_cannot_take_goes_straight_to_the_disk_and_comes_back_unbroken() {
    // No KV-RAM arena at all: the only way off the device is the disk.
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(pool(0, 4), compute.clone());

    let a = sched.submit(live(1, 600), RequestClass::Agent).unwrap(); // 38 pages
    let mut events = Vec::new();
    for _ in 0..3 {
        events.extend(sched.advance());
    }
    assert!(!tokens_of(&events, a).is_empty(), "A decodes alone first");

    // B needs 38 pages beside A's 38 of 64: A has to leave the device.
    let b = sched.submit(live(100, 600), RequestClass::Interactive).unwrap();
    let step = sched.advance();
    assert!(sched.disk_busy(), "A's spill is under way");
    assert_eq!(sched.request_state(b), Some(RequestState::Admitted), "B waits for the room");
    assert_eq!(compute.disk_spills(), vec![(DiskBlob::Live(a), DiskSource::Device)]);
    assert!(!spilled(&step, a, DiskSource::Device), "nothing is given up before the file commits");
    events.extend(step);

    events.extend(run_to_idle(&mut sched));
    assert!(spilled(&events, a, DiskSource::Device), "A went straight to the disk");
    assert!(
        !has(&events, |e| matches!(e, SchedEvent::Evicted { request, .. } if *request == a)),
        "which is not a device-to-KV-RAM eviction"
    );
    assert!(restored(&events, a), "A came back from the disk");
    assert_eq!(compute.disk_restores(), vec![DiskBlob::Live(a)]);
    assert!(done(&events, a) && done(&events, b));
    assert_eq!(
        tokens_of(&events, a),
        expected_stream(&compute, a, 600),
        "A's output continues unbroken across the move"
    );
    assert_no_work_lost(&events);
    assert!(compute.disk_files().is_empty(), "a live file goes once it has landed");
    assert_eq!(sched.disk_tier().unwrap().used_bytes(), 0);
    assert_eq!(sched.kv_used_pages(), 0);
}

#[test]
fn kv_ram_gives_its_victim_to_the_disk_and_takes_the_next_one() {
    // A KV-RAM of one blob (spec AC 23's second leg, on a CPU): the first
    // victim lands there, and is demoted to the disk to make room for the
    // second -- never discarded.
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(pool(1, 4), compute.clone());

    let a = sched.submit(live(1, 380), RequestClass::Agent).unwrap(); // 24 pages
    let b = sched.submit(live(100, 380), RequestClass::Agent).unwrap(); // 24 pages
    let mut events = Vec::new();
    for _ in 0..3 {
        events.extend(sched.advance());
    }
    // C needs 44 pages: both A and B have to go.
    let c = sched.submit(live(200, 700), RequestClass::Interactive).unwrap();
    events.extend(run_to_idle(&mut sched));

    let to_kv_ram: Vec<RequestId> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Evicted { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert_eq!(to_kv_ram.len(), 2, "both left the device for KV-RAM: {to_kv_ram:?}");
    let first = to_kv_ram[0];
    assert!(
        spilled(&events, first, DiskSource::KvRam),
        "the first was demoted to the disk to make room for the second"
    );
    assert_eq!(
        count(&events, |e| matches!(e, SchedEvent::DiskSpilled { from: DiskSource::KvRam, .. })),
        1
    );
    for r in [a, b, c] {
        assert!(done(&events, r), "request {r} finished");
    }
    for r in [a, b] {
        assert_eq!(tokens_of(&events, r), expected_stream(&compute, r, 380), "{r} generated all of it, unbroken");
        assert!(restored(&events, r));
    }
    assert_no_work_lost(&events);
}

#[test]
fn without_the_tier_kv_ram_still_drops_a_live_snapshot_for_a_newer_one() {
    // The same pressure on a load without KV-disk: today's guarantee, no
    // stronger (ADR 0045) -- which is what the scenario above is not.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(pool(1, 0), compute.clone());
    sched.submit(live(1, 380), RequestClass::Agent).unwrap();
    sched.submit(live(100, 380), RequestClass::Agent).unwrap();
    for _ in 0..3 {
        sched.advance();
    }
    sched.submit(live(200, 700), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(has(&events, |e| matches!(e, SchedEvent::SnapshotDropped { .. })));
    assert!(sched.disk_tier().is_none(), "and no ledger exists to say otherwise");
}

// ── AC 18: no room ──────────────────────────────────────────────────────────

#[test]
fn with_every_tier_full_of_live_work_a_new_request_waits_and_nothing_is_discarded() {
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            max_in_flight: 3,
            ..pool(0, 1) // no KV-RAM, a disk of one file
        },
        compute.clone(),
    );
    let b = sched.submit(live(1, 600), RequestClass::Agent).unwrap(); // 38 pages
    let mut events = Vec::new();
    for _ in 0..2 {
        events.extend(sched.advance());
    }
    let a = sched.submit(live(100, 760), RequestClass::Interactive).unwrap(); // 48 pages
    for _ in 0..4 {
        events.extend(sched.advance());
    }
    assert!(spilled(&events, b, DiskSource::Device), "B went to the disk for A");
    assert_eq!(sched.request_state(a), Some(RequestState::Running));

    // Every tier is full of live work now: A on the device, B on the disk.
    let c = sched.submit(live(200, 300), RequestClass::Agent).unwrap(); // 19 pages
    for _ in 0..20 {
        events.extend(sched.advance());
        assert_eq!(sched.request_state(c), Some(RequestState::Admitted), "C waits, held in Admitted");
    }
    assert!(!sched.disk_busy(), "no move was started: none could make the room");
    assert!(
        !has(&events, |e| matches!(e, SchedEvent::DiskFailure { .. })),
        "a full ledger is no failure: nothing was refused, the request waits"
    );
    assert!(tokens_of(&events, a).len() >= 20, "A keeps decoding meanwhile");
    // The in-flight cap is unchanged: a fourth request is refused `Full`.
    assert_eq!(sched.submit(live(300, 4), RequestClass::Agent), Err(SubmitError::Full));

    // A completes: B comes back first (a moved sequence before a newcomer),
    // and C fits beside it at the next advance.
    let mut finished_at = None;
    let mut admitted_at = None;
    for step in 0..2_000 {
        let out = sched.advance();
        if finished_at.is_none() && done(&out, a) {
            finished_at = Some(step);
        }
        if admitted_at.is_none()
            && has(&out, |e| matches!(e, SchedEvent::Admitted { request, .. } if *request == c))
        {
            admitted_at = Some(step);
        }
        events.extend(out);
        if sched.is_idle() {
            break;
        }
    }
    assert_eq!(
        admitted_at,
        finished_at.map(|s| s + 1),
        "C is admitted at the first advance after a lane completes"
    );
    for r in [a, b, c] {
        assert!(done(&events, r), "request {r} finished");
    }
    assert_no_work_lost(&events);
}

// ── AC 20: the model thread never waits on the disk ─────────────────────────

/// A disk of four-byte blobs moved a byte a window, a window every second
/// advance: eight advances a transfer.
fn slow_disk() -> (Arc<MockCompute>, FakeDisk) {
    let disk = FakeDisk {
        room_bytes: 1 << 30,
        window_bytes: 1,
        advances_per_window: 2,
    };
    let compute = MockCompute::with_sections(MockSections {
        image_bytes: 4,
        bytes_per_token: 0,
    })
    .with_disk(disk);
    (Arc::new(compute), disk)
}

/// Two Interactive lanes decoding beside an Agent victim, and an Interactive
/// request that needs the victim's pages.
struct Moving {
    sched: ConcreteScheduler,
    compute: Arc<MockCompute>,
    lanes: [RequestId; 2],
    victim: RequestId,
    requester: RequestId,
    events: Vec<SchedEvent>,
}

fn moving_victim(clock: Option<ignis_core::Clock>) -> Moving {
    let (compute, _) = slow_disk();
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            kv_disk_capacity_bytes: 8 * disk_file_bytes(4),
            ..pool(0, 0)
        },
        compute.clone(),
    );
    if let Some(clock) = clock {
        sched = sched.with_clock(clock);
    }
    let l1 = sched.submit(live(1, 400), RequestClass::Interactive).unwrap(); // 26 pages
    let l2 = sched.submit(live(100, 40), RequestClass::Interactive).unwrap(); // 3 pages
    let victim = sched.submit(live(200, 500), RequestClass::Agent).unwrap(); // 32 pages
    let mut events = Vec::new();
    for _ in 0..3 {
        events.extend(sched.advance());
    }
    // 61 of 64 pages are held: the requester's 19 need the Agent out.
    let requester = sched.submit(live(300, 290), RequestClass::Interactive).unwrap();
    Moving {
        sched,
        compute,
        lanes: [l1, l2],
        victim,
        requester,
        events,
    }
}

#[test]
fn the_other_lanes_keep_decoding_while_a_victim_spills_and_while_it_comes_back() {
    let Moving {
        mut sched,
        compute,
        lanes,
        victim,
        requester,
        mut events,
    } = moving_victim(None);
    let before = sched.kv_used_pages();
    let victim_tokens = tokens_of(&events, victim).len();

    // The spill: eight advances, a window every second one.
    let mut spill_steps = 0;
    loop {
        let out = sched.advance();
        spill_steps += 1;
        let committed = spilled(&out, victim, DiskSource::Device);
        if !committed {
            for lane in lanes {
                assert_eq!(tokens_of(&out, lane).len(), 1, "lane {lane} decodes on every advance of the spill");
            }
            assert!(tokens_of(&out, victim).is_empty(), "the victim is out of every round while it moves");
            assert_eq!(sched.kv_used_pages(), before, "its pages stay charged until its file commits");
            assert_eq!(sched.request_state(requester), Some(RequestState::Admitted));
        }
        events.extend(out);
        if committed {
            break;
        }
        assert!(spill_steps < 50, "the spill never committed");
    }
    assert!(spill_steps >= 8, "four windows at two advances each: {spill_steps}");
    assert!(
        compute.disk_windows().iter().all(|moved| moved.len() <= 1),
        "each advance copies at most one window per transfer"
    );
    assert_eq!(tokens_of(&events, victim).len(), victim_tokens, "the victim stood still");

    // The restore: charged at its start, not schedulable until its last
    // window lands.
    let mut restore_started = false;
    let mut landed = false;
    for _ in 0..5_000 {
        let out = sched.advance();
        landed |= restored(&out, victim);
        if !restore_started && compute.disk_restores().contains(&DiskBlob::Live(victim)) {
            restore_started = true;
        }
        if restore_started && !landed {
            assert!(
                tokens_of(&out, victim).is_empty(),
                "a restoring request is not scheduled before its last window lands"
            );
        }
        events.extend(out);
        if sched.is_idle() {
            break;
        }
    }
    assert!(
        restored(&events, victim),
        "victim {:?}, idle {}, restores {:?}, done {:?}",
        sched.request_state(victim),
        sched.is_idle(),
        compute.disk_restores(),
        [lanes[0], lanes[1], victim, requester].map(|r| done(&events, r))
    );
    assert_eq!(tokens_of(&events, victim), expected_stream(&compute, victim, 500));
    for r in [lanes[0], lanes[1], victim, requester] {
        assert!(done(&events, r), "request {r} finished");
    }
    assert_no_work_lost(&events);
}

#[test]
fn a_write_that_fails_leaves_the_victim_on_the_device_and_the_requester_waiting() {
    let (clock, millis) = manual_clock();
    let Moving {
        mut sched,
        compute,
        victim,
        requester,
        mut events,
        ..
    } = moving_victim(Some(clock));
    compute.fail_disk_writes(true);
    let mut failed = false;
    for _ in 0..10 {
        let out = sched.advance();
        failed |= has(&out, |e| matches!(e, SchedEvent::DiskFailure { op: DiskOp::Write }));
        events.extend(out);
    }
    assert!(failed, "the failed write is reported");
    assert!(!spilled(&events, victim, DiskSource::Device));
    assert_eq!(sched.request_state(victim), Some(RequestState::Running), "the victim is resumable on the device");
    assert_eq!(sched.request_state(requester), Some(RequestState::Admitted), "the requester still waits");
    let after_failure = tokens_of(&events, victim).len();
    let out: Vec<SchedEvent> = (0..3).flat_map(|_| sched.advance()).collect();
    assert_eq!(tokens_of(&out, victim).len(), 3, "and it decodes again");
    events.extend(out);
    assert!(after_failure > 0);
    assert_eq!(
        compute.disk_spills().len(),
        1,
        "a volume that just failed is not asked again within the backoff"
    );

    // Past the backoff, a volume that writes again takes the victim.
    compute.fail_disk_writes(false);
    millis.fetch_add(2_000, Ordering::SeqCst);
    events.extend(run_to_idle(&mut sched));
    assert!(spilled(&events, victim, DiskSource::Device));
    assert!(done(&events, victim) && done(&events, requester));
    assert_eq!(tokens_of(&events, victim), expected_stream(&compute, victim, 500));
    assert_no_work_lost(&events);
}

// ── AC 21: cancel mid-transfer ──────────────────────────────────────────────

#[test]
fn a_request_cancelled_mid_spill_releases_its_pages_and_its_file_goes() {
    let Moving {
        mut sched,
        compute,
        victim,
        requester,
        ..
    } = moving_victim(None);
    let pages_before = sched.kv_used_pages();
    let disk_before = sched.disk_tier().unwrap().used_bytes();
    sched.advance(); // the spill starts
    sched.advance();
    assert!(sched.disk_busy());
    assert!(sched.disk_tier().unwrap().used_bytes() > disk_before, "the file is charged while it lands");

    assert!(sched.cancel(victim));
    let out = sched.advance();
    assert_eq!(sched.request_state(victim), Some(RequestState::Done));
    assert!(compute.disk_discards().contains(&DiskBlob::Live(victim)), "its file is deleted");
    assert_eq!(compute.disk_transfers(), 0, "and its transfer abandoned");
    assert_eq!(
        sched.disk_tier().unwrap().used_bytes(),
        disk_before,
        "the disk ledger returns to its value before the transfer"
    );
    assert!(sched.kv_used_pages() < pages_before, "its pages are released");
    assert!(!has(&out, |e| matches!(e, SchedEvent::DiskSpilled { .. })));
    let events = run_to_idle(&mut sched);
    assert!(done(&events, requester), "the requester gets the room");
}

#[test]
fn a_request_cancelled_mid_restore_releases_its_pages_and_its_file_goes() {
    let Moving {
        mut sched,
        compute,
        lanes,
        victim,
        ..
    } = moving_victim(None);
    // Run until the victim is on its way back.
    let mut guard = 0;
    while !compute.disk_restores().contains(&DiskBlob::Live(victim)) {
        sched.advance();
        guard += 1;
        assert!(guard < 5_000, "the victim never started back");
    }
    assert!(sched.disk_busy());
    let charged = sched.kv_used_pages();
    assert!(sched.cancel(victim));
    sched.advance();
    assert_eq!(sched.request_state(victim), Some(RequestState::Done));
    assert!(compute.disk_discards().contains(&DiskBlob::Live(victim)), "its file is deleted");
    assert_eq!(compute.disk_transfers(), 0);
    assert_eq!(sched.disk_tier().unwrap().used_bytes(), 0, "the ledger holds nothing of it");
    assert!(sched.kv_used_pages() < charged, "the pages charged for the restore come back");
    let events = run_to_idle(&mut sched);
    let _ = (events, lanes);
    assert_eq!(sched.kv_used_pages(), 0);
}

// ── AC 15 (the scheduler's half): a refused file is never restored ──────────

#[test]
fn a_live_file_that_fails_its_check_is_never_restored_and_its_request_prefills_again() {
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(pool(0, 4), compute.clone());
    let a = sched.submit(live(1, 600), RequestClass::Agent).unwrap();
    for _ in 0..3 {
        sched.advance();
    }
    let b = sched.submit(live(100, 600), RequestClass::Interactive).unwrap();
    let mut events = Vec::new();
    while !spilled(&events, a, DiskSource::Device) {
        events.extend(sched.advance());
    }
    compute.corrupt_disk_file(DiskBlob::Live(a));
    events.extend(run_to_idle(&mut sched));
    assert!(has(&events, |e| matches!(e, SchedEvent::DiskFailure { op: DiskOp::Read })));
    assert!(
        has(&events, |e| matches!(e, SchedEvent::Requeued { request } if *request == a)),
        "its request prefills again"
    );
    assert!(!restored(&events, a), "the file was never restored");
    assert!(done(&events, a) && done(&events, b));
    assert!(
        !has(&events, |e| matches!(e, SchedEvent::SnapshotDropped { .. })),
        "a refused file is a read failure, not KV-RAM dropping a snapshot"
    );
}

// ── AC 17 and AC 19: retained state on the disk ─────────────────────────────

/// A device pool of one 1,280-token sequence plus a checkpoint's tail page,
/// KV-RAM of `host` nominal blobs, a disk of eight files, and the disk's
/// restore floor at `floor`.
fn tight(host: u64, floor: u32) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_in_flight: 16,
        max_sequence_tokens: 1280,
        kv_capacity_pages: 80 + 1,
        kv_page_tokens: PAGE,
        host_capacity_bytes: host,
        kv_disk_capacity_bytes: 8 * disk_file_bytes(1),
        kv_disk_restore_floor_tokens: floor,
        ..SchedulerConfig::default()
    }
}

/// A conversation's turn: 1,200 prompt tokens from `start`, opener at 1,150.
fn turn_at(start: u32) -> RequestInput {
    input(tokens(start, 1200), Some(1150), 4)
}

/// Its next turn: the first up to the opener, then 100 new tokens.
fn next_turn_at(start: u32) -> RequestInput {
    input([tokens(start, 1150), tokens(start + 50_000, 100)].concat(), Some(1240), 4)
}

/// A request needing the whole device pool, so every retained page comes
/// back for it.
fn whole_pool(start: u32) -> RequestInput {
    input(tokens(start, 1272), None, 8)
}

fn retained(events: &[SchedEvent], operation: RetainedStateOperation, source: ReuseSource) -> usize {
    count(events, |e| {
        matches!(e, SchedEvent::RetainedState { operation: o, source: s, kind: RetainedKind::Checkpoint }
            if *o == operation && *s == source)
    })
}

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

/// Turn N's checkpoint pushed off the device into a KV-RAM of one blob, then
/// pushed on to the disk by a second conversation's: returns its events.
fn checkpoint_on_disk(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    for (turn, pressure) in [(1, 100_000), (200_000, 300_000)] {
        sched.submit(turn_at(turn), RequestClass::Interactive).unwrap();
        events.extend(run_to_idle(sched));
        sched.submit(whole_pool(pressure), RequestClass::Interactive).unwrap();
        events.extend(run_to_idle(sched));
    }
    events
}

#[test]
fn a_kv_ram_checkpoint_given_up_goes_to_the_disk_and_a_later_turn_resumes_from_it() {
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(tight(1, 1024), compute.clone());
    let events = checkpoint_on_disk(&mut sched);
    assert_eq!(
        retained(&events, RetainedStateOperation::Spill, ReuseSource::Disk),
        1,
        "turn N's checkpoint left KV-RAM for the disk, not for nowhere"
    );
    let on_disk: Vec<_> = sched
        .checkpoint_pool()
        .entries()
        .iter()
        .filter(|e| e.tier == ReuseSource::Disk)
        .map(|e| e.tokens)
        .collect();
    assert_eq!(on_disk, vec![1150]);
    assert_eq!(compute.disk_files().len(), 1);

    // Turn N+1 matches nothing above the disk: its checkpoint comes back
    // into its own sequence, and its prefill starts at the opener.
    let n1 = sched.submit(next_turn_at(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::Disk, 1150)]);
    assert_eq!(retained(&events, RetainedStateOperation::Restore, ReuseSource::Disk), 1);
    assert!(compute.disk_restores().iter().any(|b| matches!(b, DiskBlob::Checkpoint(_))));
    assert_eq!(compute.disk_files().len(), 1, "a claim never consumes the file");
    let first_chunk = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .find(|job| job.request == n1)
        .expect("n1 prefilled");
    assert_eq!(first_chunk.start_position, 1150, "its prefill starts at the opener");
    assert!(first_chunk.checkpoint.is_none(), "on the sequence the restore built");
    assert!(done(&events, n1));
}

#[test]
fn a_disk_match_short_of_the_family_floor_is_not_taken() {
    // The same checkpoint behind Flash-Next's 8,192-token floor: 1,150
    // tokens of reuse do not pay for the crossing.
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(tight(1, 8192), compute.clone());
    checkpoint_on_disk(&mut sched);
    let n1 = sched.submit(next_turn_at(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(reuses(&events, n1).is_empty(), "it prefills instead");
    assert!(compute.disk_restores().is_empty());
}

#[test]
fn a_retained_file_that_fails_its_check_is_discarded_and_its_claimant_prefills() {
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(tight(1, 1024), compute.clone());
    checkpoint_on_disk(&mut sched);
    let file = *compute.disk_files().keys().next().unwrap();
    compute.corrupt_disk_file(file);
    let n1 = sched.submit(next_turn_at(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(has(&events, |e| matches!(e, SchedEvent::DiskFailure { op: DiskOp::Read })));
    assert!(reuses(&events, n1).is_empty(), "nothing was restored");
    assert!(
        sched.checkpoint_pool().entries().iter().all(|e| e.tier != ReuseSource::Disk),
        "the refused checkpoint is discarded"
    );
    assert!(done(&events, n1), "and its claimant prefilled the whole prompt");
    let prefilled: u32 = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .filter(|job| job.request == n1)
        .map(|job| job.tokens.len() as u32)
        .sum();
    assert_eq!(prefilled, 1250);
}

#[test]
fn occupancy_carries_the_disk_s_used_bytes() {
    let compute = Arc::new(MockCompute::new().with_disk(FakeDisk::with_room(1 << 30)));
    let mut sched = ConcreteScheduler::with_config(tight(1, 1024), compute.clone());
    assert_eq!(sched.occupancy().kv_disk_used_bytes, 0);
    checkpoint_on_disk(&mut sched);
    assert_eq!(sched.occupancy().kv_disk_used_bytes, disk_file_bytes(1));
    assert_eq!(sched.occupancy().kv_disk_used_bytes, sched.disk_tier().unwrap().used_bytes());
}
