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
//!
//! **P4-07 (GitHub #125) moved KV-page charging off the decode-lane deal
//! and onto first materialization** (a request's first `Prefilling` chunk —
//! mirroring `ignis_seq_alloc`'s own real reservation timing). One
//! consequence for *this* file specifically: a page-tight, lane-plentiful
//! pool can no longer reach Phase 2 (the protection/backfill machinery)
//! at all — page pressure is now resolved earlier, at admission. See
//! `kv_page_pressure_holds_a_fresh_candidate_until_room_frees`'s own doc
//! comment for the full reasoning, and the note left where scenario 4
//! (a persistent-backfill integration scenario built on that now-
//! unreachable shape) used to live.

use std::sync::Arc;

use ignis_core::types::{
    BackfillClass, DecodeParams, RequestClass, RequestInput, RequestState, SchedEvent,
    SubmitError,
};
use ignis_core::{ConcreteScheduler, MockCompute, ProtectionPhase, Scheduler, SchedulerConfig};

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
        // P4-07, GitHub #125: generous — these scenarios exercise the
        // core-05 admission machine's lane/kv_pages dimensions in
        // isolation; a tight resident-slot budget is a different
        // scheduler.rs test's concern (host_tier.rs).
        resident_slot_capacity: 16,
        // The core-05 scenarios exercise the admission machine in isolation:
        // the host tier is disabled (no overflow), so a blocked head waits
        // for its donors instead of being admitted via a lane eviction.
        host_capacity_bytes: 0,
        serving_chunk_tokens: ignis_core::DEFAULT_SERVING_CHUNK_TOKENS,
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

/// Scenario 1 — KV-page pressure holds a fresh candidate at admission,
/// *before* it ever reaches the protection machinery below (P4-07, GitHub
/// #125): a request's KV pages are now charged the moment it first
/// materializes on the leaf (its first `Prefilling` chunk,
/// `Self::fits_for_materialization`), not at its later decode-lane deal —
/// mirroring `ignis_seq_alloc`'s own real reservation timing, so admission
/// and the leaf never disagree about how full the pool is. A candidate
/// that cannot yet fit is never dispatched to the compute backend at all;
/// it waits `Admitted`, exactly like any other admission refusal with
/// nowhere to evict to (the host tier is disabled here).
///
/// This is also why the *old* shape of this test — a page-tight pool
/// leaving decode lanes plentiful, so a backfill could be classified
/// Persistent/Temporal purely on page pressure while a lane sat free for
/// it — no longer reaches Phase 2 at all: page pressure is now resolved
/// here, before a request is ever `Prefilling`-complete. That policy
/// (donor selection, temporal credit decay, persistent-vs-temporal
/// classification) is still fully exercised at the pure-function level in
/// `admission.rs`'s own tests; scenario 2 below covers the one dimension
/// that *can* still block an already-materialized head at Phase 2: decode
/// lanes.
#[test]
fn kv_page_pressure_holds_a_fresh_candidate_until_room_frees() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            kv_capacity_pages: 8,
            ..small_pool()
        },
        compute.clone(),
    );

    // One incumbent (4 pages) leaves only 4 of the 8-page pool free.
    sched.submit(input(&[1], 63), RequestClass::Agent).unwrap(); // ceil(64/16) = 4 pages
    let ev1 = sched.advance(); // step 1: n1 materializes and is dealt a lane

    // A 5-page candidate cannot materialize (4 + 5 = 9 > 8), and the host
    // tier is disabled (nothing to evict into) — it must wait `Admitted`,
    // never reaching the compute backend at all.
    let waiting = sched.submit(input(&[1], 79), RequestClass::Agent).unwrap(); // ceil(80/16) = 5 pages
    let ev2 = sched.advance(); // step 2: n1 keeps decoding; `waiting` is held

    assert_eq!(
        sched.request_state(waiting),
        Some(RequestState::Admitted),
        "KV-page pressure holds the candidate before it ever materializes"
    );
    assert!(
        compute
            .prefill_calls()
            .iter()
            .flatten()
            .all(|job| job.request != waiting),
        "a candidate that cannot fit the pool is never dispatched to the leaf"
    );
    assert!(
        !ev2.iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == waiting)),
        "no Admitted event for a request that never materialized"
    );
    assert_eq!(sched.kv_used_pages(), 4, "only n1's pages are charged so far");

    // Once n1 completes and its 4 pages return, `waiting` finally
    // materializes and runs to completion.
    let rest = run_to_idle(&mut sched);
    let events = ev1.into_iter().chain(ev2).chain(rest).collect::<Vec<_>>();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == waiting)),
        "the held candidate is dealt once room frees"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == waiting)),
        "the held candidate completes normally"
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
        // P4-07, GitHub #125: generous — this scenario means *lanes* to be
        // the sole constraint. Left at the default (== N_DECODE_LANES) a
        // 9th request could never even reach `Prefilling`-complete while
        // 8 fillers hold every resident slot, which would block it before
        // Phase 2 (the admission machine this test exercises) ever saw it.
        resident_slot_capacity: 16,
        host_capacity_bytes: 0, // host tier disabled: lane pressure is the
        // constraint (the head waits for a lane to free, core-05).
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
    let t1 = sched.submit(input(&[1], 15), RequestClass::Agent).unwrap();

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

/// GitHub #98 (P3-02) — the scheduler's capacity is built from
/// `ignis_core::kv::verified_kv_pool`'s output, i.e. the *leaf-verified*
/// page geometry (`ignis_seq_pool_stats`), not an arbitrary constant: once
/// the scheduler is configured from that verified pool's own page count, a
/// request it cannot cover is refused at `submit` and never handed to the
/// compute seam — the leaf never sees a job it would have to fail on.
#[test]
fn admission_capacity_is_built_from_the_leaf_verified_kv_pool_and_never_dispatches_a_refusal() {
    // 64-token pages (the leaf's fixed `kPagedKVPageSize`), 8 physical
    // pages of 64 KiB each — exactly what `ignis_seq_pool_stats` would
    // report for a small pool. The scheduler's own page count (here, a
    // stand-in for `CudaLeafConfig::kv_pool_plan`'s) agrees with it, so the
    // verification succeeds and hands back the real pool.
    let leaf_geometry = ignis_core::kv::LeafPoolGeometry {
        page_count: 8,
        page_bytes: 64 * 1024,
    };
    let pool = ignis_core::kv::verified_kv_pool(8, leaf_geometry)
        .expect("the scheduler's formula agrees with the leaf's reported pool");
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            max_in_flight: 16,
            max_prefill_batch: 8,
            kv_page_tokens: 64,
            max_sequence_tokens: 1024,
            kv_capacity_pages: pool.block_count() as u32,
            resident_slot_capacity: 16,
            host_capacity_bytes: 0,
            serving_chunk_tokens: ignis_core::DEFAULT_SERVING_CHUNK_TOKENS,
        },
        compute.clone(),
    );

    // ceil((1 + 511) / 64) = 8 pages == the whole KvPool: allowed.
    let admitted = sched.submit(input(&[1], 511), RequestClass::Agent).unwrap();
    // ceil((1 + 512) / 64) = 9 pages > the pool KvPool actually reports:
    // refused at submit, before it can ever reach the compute seam.
    let refused = sched.submit(input(&[1], 512), RequestClass::Agent);
    assert_eq!(refused, Err(SubmitError::Oversized));

    sched.advance();
    let dispatched: Vec<u64> = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .map(|job| job.request)
        .collect();
    assert!(
        dispatched.contains(&admitted),
        "the request the pool covers is prefilled normally"
    );
    // The refused request was never even assigned a request id (`submit`
    // returned `Err`), so by construction it cannot appear in a dispatched
    // job — the assertion above is the positive half of that same guarantee.
}

// Scenario 4 (persistent backfill fitting the protected future) used to
// live here, blocked on page pressure while lanes stayed plentiful — the
// same shape scenario 1's page-tight pool used to take, before GitHub #125
// moved page-pressure resolution off the lane-deal path entirely (see
// scenario 1's doc comment). It is not replaced 1:1: a page-tight,
// lane-plentiful integration scenario can no longer reach Phase 2 at all,
// and a lane-tight one (the only dimension that still can) never reaches a
// Persistent/Temporal classification either — blocking a head on lanes
// specifically means every lane is already taken, leaving none for a
// backfill to ride (`ConcreteScheduler::try_admit` pops a real lane before
// recording anything). The persistent-backfill *policy* itself — the
// future-capacity fit this scenario existed to check — is still fully
// exercised at the pure-function level:
// `admission::tests::persistent_backfill_respects_the_protected_future_capacity`
// and `admission::tests::a_second_backfill_sees_the_first_persistent_occupant`.
