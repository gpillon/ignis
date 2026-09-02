//! core-05 — the full admission state machine (ADR 0004) driven end-to-end
//! through the concrete scheduler: protection freeze, backfill
//! classification (persistent vs temporal), temporal-credit decay, the drain
//! phase, and oversized rejection — each pinned as a dedicated invariant
//! test, per ADR 0004.
//!
//! Every scenario runs on `MockCompute` (ADR 0006: the `Compute` seam is
//! mocked, so the whole machine runs on a CPU). The KV pool is deliberately
//! small so the resource arithmetic is exact: a request reserves
//! `ceil((prompt + max_tokens) / kv_page_tokens)` pages at submit, and the
//! machine admits a backfill only while it fits the protection's invariants.

use std::sync::Arc;

use ignis_core::types::{
    BackfillClass, DecodeParams, RequestClass, RequestInput, SchedEvent, SubmitError,
};
use ignis_core::{
    ConcreteScheduler, MockCompute, ProtectionPhase, Scheduler, SchedulerConfig,
};

/// A request with the test model's prompt (`tokens`) and an explicit
/// generation cap of `max` tokens.
fn input(tokens: &[u32], max: u32) -> RequestInput {
    RequestInput {
        model: "qwen3.8-27b".into(),
        tokens: tokens.to_vec(),
        params: DecodeParams {
            max_tokens: Some(max),
            ..Default::default()
        },
    }
}

/// A small KV pool: 16-token pages, 16-page capacity, plenty of in-flight
/// budget and a large prefill batch so *page* arithmetic (not queueing) is
/// the constraint.
fn small_pool() -> SchedulerConfig {
    SchedulerConfig {
        model: "qwen3.8-27b".into(),
        max_in_flight: 16,
        max_prefill_batch: 8,
        kv_page_tokens: 16,
        max_sequence_tokens: 1024,
        kv_capacity_pages: 16,
    }
}

/// Drive the scheduler until it is idle, collecting every event.
fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// The backfill class a request was admitted under (from its `Admitted`
/// event), or `None` if it was never admitted.
fn backfill_of(events: &[SchedEvent], request: u64) -> Option<BackfillClass> {
    events.iter().find_map(|e| match e {
        SchedEvent::Admitted {
            request: r,
            backfill,
            ..
        } if *r == request => Some(*backfill),
        _ => None,
    })
}

/// The `Protected` events, in emission order.
fn protections(events: &[SchedEvent]) -> Vec<&SchedEvent> {
    events
        .iter()
        .filter(|e| matches!(e, SchedEvent::Protected { .. }))
        .collect()
}

/// Scenario 1 — a protected head freezes a protection and classifies its
/// backfills: a candidate that fits the pool *now* but overflows the head's
/// *future* capacity is admitted as **temporal** (spending the protection's
/// temporal credit), a second candidate that no longer fits the pool is
/// held, and the head is dealt normally only once its donors release.
#[test]
fn blocked_head_freezes_protection_and_classifies_backfills() {
    // Pool 12, 16-token pages. Two 4-page incumbents (8 pages) + a 5-page
    // head = 13 > 12, so the head is blocked while the donors run.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            kv_capacity_pages: 12,
            ..small_pool()
        },
        compute,
    );

    // Phase A: two incumbents (4 pages each, work 63) fill 8 of the 12
    // pool pages.
    let n1 = sched
        .submit(input(&[1], 63), RequestClass::Agent)
        .unwrap(); // ceil(64/16) = 4 pages
    let n2 = sched
        .submit(input(&[1], 63), RequestClass::Agent)
        .unwrap(); // 4 pages
    sched.advance(); // step 1: both prefilled + dealt (8 pages in use)

    // Phase B: the Interactive head (5 pages) and two 4-page temporal
    // candidates are queued behind the incumbents.
    let head = sched
        .submit(input(&[1], 79), RequestClass::Interactive)
        .unwrap(); // ceil(80/16) = 5 pages
    let t1 = sched
        .submit(input(&[1], 61), RequestClass::Agent)
        .unwrap(); // ceil(62/16) = 4 pages, work 61
    let t2 = sched
        .submit(input(&[1], 61), RequestClass::Agent)
        .unwrap(); // 4 pages, work 61
    let ev2 = sched.advance(); // step 2: prefill all three; admission below

    // Step 2: the head (8 + 5 = 13 > 12 pages) is blocked → a protection
    // freezes. t1 fits *now* (8 + 4 = 12 ≤ 12) but overflows the head's
    // *future* (head 5 + n2 4 + t1 4 = 13 > 12) → admitted as **temporal**
    // (work 61 ≤ frontier 62 ∧ credit 62), decaying the credit to 1. t2
    // (12 + 4 = 16 > 12) no longer fits the pool → held.
    let protected = protections(&ev2);
    assert_eq!(protected.len(), 1, "one protection for the blocked head");
    let SchedEvent::Protected {
        epoch,
        head: phead,
        donors,
    } = protected[0]
    else {
        unreachable!()
    };
    assert_eq!(*phead, head, "the blocked head is protected");
    assert_eq!(*epoch, 1, "first protection epoch");
    assert!(
        donors.iter().any(|d| *d == n1),
        "the earliest-completion incumbent (n1) is a donor"
    );
    assert!(
        !donors.iter().any(|d| *d == n2),
        "n1 alone releases enough: the donor prefix stops at n1"
    );

    assert_eq!(
        backfill_of(&ev2, t1),
        Some(BackfillClass::Temporal),
        "t1 overflows the head's future → admitted as a temporal backfill"
    );
    assert_eq!(
        backfill_of(&ev2, t2),
        None,
        "t2 no longer fits the pool (8 + 4 + 4 > 12) → held"
    );
    let prot = sched.protection().expect("a protection is open");
    assert_eq!(prot.epoch_id, 1);
    assert_eq!(prot.head_request_id, head);
    assert!(prot.donor_ids.iter().any(|d| *d == n1));
    assert_eq!(
        prot.temporal_credit, 1,
        "the temporal credit decays by t1's own work (62 − 61)"
    );
    assert_eq!(
        sched.kv_used_pages(),
        12,
        "the pool is full: 8 (n1 + n2) + 4 (t1)"
    );

    // Let the machine run to idle: t1 finishes (work 61), the donors finish
    // (work 63), the head is dealt normally, and t2 — still queued — takes a
    // normal (non-backfill) deal once the pool has room.
    let rest = run_to_idle(&mut sched);
    let events = ev2.into_iter().chain(rest).collect::<Vec<_>>();

    assert_eq!(protections(&events).len(), 1, "exactly one protection");
    assert_eq!(
        backfill_of(&events, t2),
        Some(BackfillClass::None),
        "t2 is dealt normally once the pool has room"
    );
    assert_eq!(
        backfill_of(&events, head),
        Some(BackfillClass::None),
        "the protected head is dealt normally after its donors release"
    );
    // The temporal backfill is dealt *before* the head (that is the point of
    // backfilling a blocked head).
    let t1_admitted = events
        .iter()
        .position(|e| matches!(e, SchedEvent::Admitted { request: r, .. } if *r == t1))
        .unwrap();
    let head_admitted = events
        .iter()
        .position(|e| matches!(e, SchedEvent::Admitted { request: r, .. } if *r == head))
        .unwrap();
    assert!(
        t1_admitted < head_admitted,
        "the temporal backfill precedes the protected head"
    );
    assert!(
        sched.protection().is_none(),
        "the protection is cleared once its head is dealt"
    );
    assert_eq!(sched.kv_used_pages(), 0, "all reservations released");
    assert!(sched.is_idle());
}

/// Scenario 2 — **lane pressure**: with all 8 resident lanes occupied, a
/// blocked head is held by the *lane* dimension, not the page pool (the
/// pool is huge). The opened protection stays **Open** — the
/// "safe without temporals" check (head + active set ≤ capacity) fails on
/// lanes (1 + 8 = 9 > 8) — and backfill candidates are held too, because
/// none of them can `fits` into a full lane set. Once the earliest donor's
/// lane frees, the head and the held backfill are both dealt normally
/// (no backfill class — the protection is cleared the moment its head is
/// dealt).
///
/// Note on the *drain* phase: it is unreachable in v1's resource model and
/// is kept only for reference fidelity (ADR 0004). A temporal backfill's
/// work is bounded by its temporal credit (≤ the last donor's work), so a
/// temporal borrower always finishes before the last donor; by the time
/// "safe without temporals" could become true, the head fits and is dealt
/// through the plain deal branch instead.
#[test]
fn lane_pressure_holds_head_and_backfills_until_a_lane_frees() {
    let compute = Arc::new(MockCompute::new());
    let cfg = SchedulerConfig {
        model: "qwen3.8-27b".into(),
        max_in_flight: 16,
        max_prefill_batch: 8,
        ..SchedulerConfig::default() // 4096-page pool: pages are never tight
    };
    let mut sched = ConcreteScheduler::with_config(cfg, compute);

    // Eight fillers occupy all eight resident lanes (2–3 pages each — the
    // pool is huge, so pages are not the constraint; lanes are).
    for i in 0..8u32 {
        sched
            .submit(input(&[1], 30 + i), RequestClass::Agent)
            .unwrap();
    }
    let ev1 = sched.advance(); // step 1: the eight fillers are dealt onto all 8 lanes

    // The blocked head and the backfill candidate are submitted *after* the
    // lanes fill up, so they queue behind the incumbents (an Interactive
    // head submitted earlier would prefill first and take a lane at step 1).
    let head = sched
        .submit(input(&[1], 30), RequestClass::Interactive)
        .unwrap();
    let t1 = sched
        .submit(input(&[1], 15), RequestClass::Agent)
        .unwrap();

    // Step 2: the head (9th request, 8 lanes) is blocked on lanes. The
    // backfill candidate t1 is held too: no free lane for it to fit into.
    let ev2 = sched.advance();
    let prot = sched.protection().expect("a protection is open");
    assert_eq!(
        prot.phase,
        ProtectionPhase::Open,
        "head + 8 incumbents = 9 > 8 lanes → not safe without temporals"
    );
    assert_eq!(prot.head_request_id, head);
    assert_eq!(
        backfill_of(&ev2, t1),
        None,
        "no free lane → the backfill candidate is held (it cannot fit)"
    );
    assert!(!sched.is_idle());

    // Run to idle: the shortest fillers finish first, their lanes free, and
    // the head (then t1) are dealt *normally* — a head dealt while a
    // protection is open clears it (the next blocked head opens a fresh
    // epoch).
    let events = ev1
        .into_iter()
        .chain(ev2)
        .chain(run_to_idle(&mut sched))
        .collect::<Vec<_>>();
    assert_eq!(
        backfill_of(&events, head),
        Some(BackfillClass::None),
        "the head is dealt normally once a donor lane frees"
    );
    assert_eq!(
        backfill_of(&events, t1),
        Some(BackfillClass::None),
        "t1 is dealt normally after the head (not as a backfill)"
    );
    assert_eq!(
        protections(&events).len(),
        1,
        "one protection for the single blocked head"
    );
    assert!(
        sched.protection().is_none(),
        "the protection is cleared when its head is dealt"
    );
    assert!(sched.is_idle());
}

/// Scenario 3 — a request whose KV reservation alone exceeds the whole pool
/// is rejected at submit with `Oversized` (it can never be admitted).
#[test]
fn oversized_requests_are_rejected_at_submit() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            kv_capacity_pages: 8,
            ..small_pool()
        },
        compute,
    );

    // ceil((1 + 127) / 16) = 8 pages == the whole pool: allowed.
    assert!(sched.submit(input(&[1], 127), RequestClass::Agent).is_ok());
    // ceil((1 + 128) / 16) = 9 pages > the pool: oversized, rejected.
    assert_eq!(
        sched.submit(input(&[1], 128), RequestClass::Agent),
        Err(SubmitError::Oversized)
    );
    // A second oversized request is also rejected (a rejected submit
    // consumes no capacity).
    assert_eq!(
        sched.submit(input(&[1], 200), RequestClass::Agent),
        Err(SubmitError::Oversized)
    );
}

/// Scenario 4 — a **persistent** backfill: a candidate that fits the head's
/// *future* capacity (head + non-donors + this backfill ≤ pool) is admitted
/// as `Persistent` — it never borrows the donor's reserved pages, so it is
/// safe even though a protection is open.
#[test]
fn persistent_backfill_fits_the_protected_future() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            kv_capacity_pages: 12,
            ..small_pool()
        },
        compute,
    );

    // Two 4-page incumbents (8 pages) + a 5-page head = 13 > 12 → the head is
    // blocked. A 1-page backfill fits the pool now (8 + 1 = 9 ≤ 12) and the
    // head's future (head 5 + the non-donor 4 + 1 = 10 ≤ 12) → it is
    // admitted as **persistent**.
    let a = sched
        .submit(input(&[1], 63), RequestClass::Agent)
        .unwrap(); // 4 pages
    let b = sched
        .submit(input(&[1], 63), RequestClass::Agent)
        .unwrap(); // 4 pages
    let ev1 = sched.advance(); // step 1: the two incumbents are dealt (8 pages)

    // The head and the backfill are submitted *after* the pool is loaded,
    // so the head queues behind the incumbents and is the one that gets
    // blocked (an earlier submission would let it take a lane at step 1).
    let head = sched
        .submit(input(&[1], 79), RequestClass::Interactive)
        .unwrap(); // ceil(80/16) = 5 pages — blocked
    let p = sched
        .submit(input(&[1], 8), RequestClass::Agent)
        .unwrap(); // ceil(9/16) = 1 page
    let ev2 = sched.advance(); // step 2: head blocked → protection; p dealt

    assert_eq!(
        backfill_of(&ev2, p),
        Some(BackfillClass::Persistent),
        "a 1-page backfill fits the head's future (5 + 4 + 1 = 10 ≤ 12)"
    );
    assert_eq!(
        backfill_of(&ev2, head),
        None,
        "the head is still blocked while the donors run"
    );

    // Let it run: the 1-page backfill finishes early (work 8), the donors
    // finish (work 63), and the head is dealt normally.
    let events = ev1
        .into_iter()
        .chain(ev2)
        .chain(run_to_idle(&mut sched))
        .collect::<Vec<_>>();
    assert_eq!(
        backfill_of(&events, head),
        Some(BackfillClass::None),
        "the protected head is dealt normally after its donors release"
    );
    // The persistent backfill is dealt *before* the head (it rides the
    // protection, the head waits for the donors).
    let p_admitted = events
        .iter()
        .position(|e| matches!(e, SchedEvent::Admitted { request: r, .. } if *r == p))
        .unwrap();
    let head_admitted = events
        .iter()
        .position(|e| matches!(e, SchedEvent::Admitted { request: r, .. } if *r == head))
        .unwrap();
    assert!(
        p_admitted < head_admitted,
        "the persistent backfill precedes the protected head"
    );
    assert!(sched.is_idle());
    let _ = (a, b);
}