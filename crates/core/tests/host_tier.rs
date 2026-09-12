//! core-06 — end-to-end host-tier scenarios, driven through the concrete
//! scheduler on a CPU (`MockCompute`, ADR 0006): the KV-RAM host tier lets
//! the scheduler admit **beyond** the N=8 resident lanes (the overflow
//! path) by snapshotting a lower-value lane into host RAM. The evicted
//! (suspended) request is later **restored** — it resumes from where it was
//! evicted, not re-prefilled — and the tier itself stays **bounded** (when
//! it fills, it discards its lowest-value snapshot, and that request
//! re-prefills later).
//!
//! Each scenario keeps the KV page pool generous (the default 4096-page
//! pool) so the *lane* dimension (8 resident lanes), not the page pool, is
//! the constraint — that is what forces the overflow (evict-to-tier) path.
//! `host_capacity_bytes` is the knob that drives the tier's bounded
//! behavior: large (no discards, everything restores) vs small (the tier
//! evicts to stay within budget, discarding the oldest snapshot).
//! `MockCompute::evict` (GitHub #125) reports a nominal 1 byte per
//! snapshot, so a byte budget here reads exactly like the old page-count
//! one — an N-byte tier holds N snapshots.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

/// A request with a 4-token prompt and a `max` generation cap. With the
/// default 16-token page, `4 + max` tokens reserves `ceil((4 + max) / 16)`
/// pages — 1 page for `max ≤ 12`.
fn input(max: u32) -> RequestInput {
    RequestInput {
        model: "qwen3.8-27b".into(),
        tokens: vec![1, 2, 3, 4],
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
    }
}

/// A scheduler with a generous page pool (the lane dimension is the
/// constraint) and a host tier of `host_bytes` bytes. `max_in_flight` is
/// raised above the 8 resident lanes so the overflow (beyond-N) requests
/// can be submitted (the host-tier overflow path, core-06).
/// Drive the scheduler to idle, collecting every event.
fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

fn sched_with(host_bytes: u64) -> ConcreteScheduler {
    let compute = Arc::new(MockCompute::new());
    let cfg = SchedulerConfig {
        model: "qwen3.8-27b".into(),
        max_in_flight: 16, // 8 resident lanes + the overflow beyond them
        max_prefill_batch: 8,
        host_capacity_bytes: host_bytes,
        ..SchedulerConfig::default() // 4096-page pool: pages are never tight
    };
    ConcreteScheduler::with_config(cfg, compute)
}

/// The request ids of the `events` for which `kind` holds (only the evict /
/// restore / requeue event variants carry request ids here).
fn ids_of<F>(events: &[SchedEvent], kind: F) -> Vec<u64>
where
    F: Fn(&SchedEvent) -> bool,
{
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Evicted { request, .. } if kind(e) => Some(*request),
            SchedEvent::Restored { request, .. } if kind(e) => Some(*request),
            SchedEvent::Requeued { request } if kind(e) => Some(*request),
            _ => None,
        })
        .collect()
}

/// Scenario 1 — a blocked head (the overflow request) is admitted by
/// **evicting** a lower-value lane into the host tier; once the evicted
/// request's warmed KV is no longer needed, it is **restored** (it resumes
/// from where it was evicted — no re-prefill) and completes.
#[test]
fn evict_frees_a_blocked_head_and_restore_skips_reprefill() {
    // A large tier: every evicted snapshot fits (no discards), so the
    // evicted request is always restored (never re-prefilled).
    let mut sched = sched_with(64);

    // Eight Agent fillers occupy all eight resident lanes.
    for _ in 0..8 {
        sched.submit(input(8), RequestClass::Agent).unwrap();
    }
    let ev1 = sched.advance(); // step 1: the 8 fillers are prefilled + dealt

    // The ninth request (the overflow "head") is blocked: all eight lanes
    // are occupied, so it cannot be dealt without freeing one.
    let head = sched.submit(input(8), RequestClass::Agent).unwrap();
    let ev2 = sched.advance(); // step 2: the head is blocked -> a lane is evicted

    // A lower-value (non-donor) lane was evicted into the host tier, and
    // the blocked head was admitted onto the freed lane.
    let evicted: Vec<u64> = ids_of(&ev2, |e| matches!(e, SchedEvent::Evicted { .. }));
    assert!(
        !evicted.is_empty(),
        "a lane is evicted into the host tier to free room for the head"
    );
    assert!(
        ev2.iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == head)),
        "the blocked head is admitted once the eviction frees a lane"
    );
    // The tier holds the evicted snapshot and stays within its budget.
    assert!(
        sched.host_tier().used_bytes() <= sched.host_tier().capacity_bytes(),
        "the host tier never exceeds its budget"
    );

    // Run to idle: the head finishes (freeing its lane), the evicted request
    // is restored (no re-prefill) and completes, and the fillers finish.
    let mut events = ev1.into_iter().chain(ev2).collect::<Vec<_>>();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }

    // The evicted request was **restored**, not re-prefilled: it was dealt
    // exactly once (its initial deal — no second `Admitted`, which would
    // signal a re-prefill) and it got a `Restored` event (it resumed from
    // where it was evicted).
    for &d in &evicted {
        let admitted = events
            .iter()
            .filter(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == d))
            .count();
        assert_eq!(
            admitted, 1,
            "the evicted request {d} was dealt once (no re-prefill deal)"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == d)),
            "the evicted request {d} was restored (it resumes, not re-prefills)"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == d)),
            "the restored request {d} completes"
        );
    }
    // All nine requests (the eight fillers + the overflow head) completed.
    let done = events
        .iter()
        .filter(|e| matches!(e, SchedEvent::Done { .. }))
        .count();
    assert_eq!(done, 9, "all eight fillers + the overflow head complete");
    assert!(sched.is_idle());
}

/// Scenario 2 — under N=8 + overflow load the evictions are **bounded**:
/// with a small host-RAM budget the tier evicts its lowest-value (probation
/// LRU) snapshot to make room, so it never exceeds `capacity_bytes`; the
/// discarded snapshot's request is re-queued (it re-prefills later), while
/// the retained snapshots are still restored.
#[test]
fn evictions_are_bounded_under_overflow_load() {
    // A 2-byte tier holding 1-byte snapshots: at most two fit, so a third
    // capture must evict (discard) the oldest snapshot to stay bounded.
    let mut sched = sched_with(2);

    // Eight Agent fillers occupy all eight resident lanes.
    for _ in 0..8 {
        sched.submit(input(8), RequestClass::Agent).unwrap();
    }
    sched.advance(); // step 1: the 8 fillers are dealt onto all 8 lanes

    // Three overflow requests: each is blocked (no free lane) and is
    // admitted by evicting a lane into the (small) host tier.
    let o1 = sched.submit(input(8), RequestClass::Agent).unwrap();
    let o2 = sched.submit(input(8), RequestClass::Agent).unwrap();
    let o3 = sched.submit(input(8), RequestClass::Agent).unwrap();

    // Run to idle, checking the tier stays bounded at every step.
    let mut events = Vec::new();
    while !sched.is_idle() {
        for e in sched.advance() {
            events.push(e);
        }
        // Boundedness invariant: the tier never holds more than its budget.
        assert!(
            sched.host_tier().used_bytes() <= sched.host_tier().capacity_bytes(),
            "the host tier must stay within its {}-byte budget (holds {})",
            sched.host_tier().capacity_bytes(),
            sched.host_tier().used_bytes()
        );
    }

    // The tier filled (three 1-byte captures into a 2-byte tier), so at
    // least one snapshot was discarded and its request re-queued (it will
    // re-prefill) — that is the bounded behavior (the tier does not grow
    // without bound; it drops its lowest-value entry instead).
    let requeued: Vec<u64> = ids_of(&events, |e| matches!(e, SchedEvent::Requeued { .. }));
    assert!(
        !requeued.is_empty(),
        "a snapshot was discarded (re-queued) to keep the tier bounded"
    );
    // The retained (non-discarded) snapshots were still restored (not
    // re-prefilled): some request got a `Restored` event.
    let restored: Vec<u64> = ids_of(&events, |e| matches!(e, SchedEvent::Restored { .. }));
    assert!(
        !restored.is_empty(),
        "the retained snapshots are restored (sibling requests resume, not re-prefill)"
    );
    // Every request (the eight fillers + the three overflow heads)
    // completed: the re-queued one re-prefilled, the retained ones
    // restored, and all finished.
    let done = events
        .iter()
        .filter(|e| matches!(e, SchedEvent::Done { .. }))
        .count();
    assert_eq!(
        done, 11,
        "all eight fillers + the three overflow heads complete"
    );
    for &o in [&o1, &o2, &o3] {
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == o)),
            "overflow head {o} completes"
        );
    }
    assert!(sched.is_idle());
}

// ── P4-07 (GitHub #125): half-prefilled eviction ────────────────────────

/// Scenario 3 — a **half-prefilled** (`Prefilling`) request is
/// device-resident (real KV pages, a GDN slot, conv taps) long before it
/// ever holds a decode lane, and is therefore evictable — restoring it
/// resumes without re-prefilling, and it does not oscillate host↔GPU once
/// restored.
///
/// This drives the trigger that is actually reachable end-to-end: a
/// `Prefilling`, prefill-*complete* request queued for a lane (blocked on
/// lanes, not resident slots — `eight` fillers hold every lane) is a valid
/// [`Self::prefilling_eviction_candidate`] once nothing else needs its
/// slot; the *chunk-boundary* / mid-multi-chunk case (`prefill_progress`
/// short of the whole prompt) is exercised directly at the data-model
/// level instead — `host::tests::a_half_prefilled_entry_carries_its_resume_phase_and_progress`
/// (the tier round-trips an arbitrary boundary unchanged) and
/// `request::tests::a_prefilling_request_can_be_evicted_and_restored_to_prefilling`
/// (the state machine) — because P3-01's own durable-prefill rule ("exactly
/// one request may hold multi-tick prefill progress at a time") means the
/// active continuer is *never* competing with anything else for Phase 1's
/// admission-refusal path while it is still mid-chunk: nothing else is ever
/// considered until it finishes or is cut by the chunk-width rule. Once
/// `prefill_complete()` is true, though, it stops being "the" active
/// continuer and becomes exactly like any other resident, lane-less
/// candidate — which is the shape this test drives through the real
/// `Compute` seam, host tier, and restore path together.
#[test]
fn a_half_prefilled_request_is_evicted_and_resumes_without_reprefilling() {
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            max_in_flight: 10, // 8 fillers + a + b
            max_prefill_batch: 8,
            // Exactly enough resident slots for the 8 lane-holding fillers
            // plus `a` — none left over for `b`.
            resident_slot_capacity: 9,
            host_capacity_bytes: 64,
            ..SchedulerConfig::default()
        },
        Arc::new(MockCompute::new()),
    );

    // Eight fillers share one 16-token prefix (a whole page): the first
    // publishes it, the rest claim it — every one of them ends up holding
    // `prefix_entry`, which is what excludes *all eight* from
    // `Self::retained_lane_candidates` despite occupying every lane.
    for _ in 0..8 {
        sched
            .submit(
                RequestInput {
                    model: "qwen3.8-27b".into(),
                    tokens: (1..=16).collect(),
                    params: DecodeParams {
                        max_tokens: Some(20),
                        ..DecodeParams::default()
                    },
                },
                RequestClass::Agent,
            )
            .unwrap();
        sched.advance();
    }

    // `a`: a short, unique (unshared) prompt. It materializes fine (the
    // 9th resident slot is free) and completes its own one-chunk prefill,
    // but every lane is already taken — it queues, `Prefilling` and
    // complete, holding no lane and no prefix.
    let a = sched
        .submit(
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: (1000..1004).collect(),
                params: DecodeParams {
                    max_tokens: Some(8),
                    ..DecodeParams::default()
                },
            },
            RequestClass::Agent,
        )
        .unwrap();
    let ev_a = sched.advance();
    assert_eq!(
        sched.prefill_progress(a),
        Some(4),
        "a's whole (unshared) 4-token prompt prefilled in one chunk"
    );
    assert_eq!(
        sched.request_state(a),
        Some(ignis_core::types::RequestState::Prefilling),
        "complete, but still queued — every lane is taken"
    );

    // `b`: another short, unique prompt. The resident-slot budget (9) is
    // now fully spent (8 fillers + `a`) — the only eligible victim for
    // `b`'s own materialization is `a`: the fillers are excluded (shared
    // prefix), and nothing else is resident.
    let b = sched
        .submit(
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: (2000..2004).collect(),
                params: DecodeParams {
                    max_tokens: Some(4),
                    ..DecodeParams::default()
                },
            },
            RequestClass::Agent,
        )
        .unwrap();
    let ev_b = sched.advance();

    let a_evicted_now = ev_b
        .iter()
        .filter(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == a))
        .count();
    assert_eq!(
        a_evicted_now, 1,
        "the fully-prefilled, lane-less `a` is evicted to make room for `b`'s materialization"
    );
    assert_eq!(
        sched.request_state(a),
        Some(ignis_core::types::RequestState::Evicted),
        "`a` is suspended, not discarded"
    );

    // Run to idle: as fillers complete and free lanes (and, eventually,
    // resident slots), `a` is restored — never re-queued, so its prefill
    // is never redone — and every request finishes.
    let mut events = ev_a.into_iter().chain(ev_b).collect::<Vec<_>>();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a)),
        "`a` is restored (not re-queued) once room exists again"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::Requeued { request } if *request == a)),
        "`a` is never re-queued — its completed prefill is never redone"
    );
    let a_evictions = events
        .iter()
        .filter(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == a))
        .count();
    assert_eq!(
        a_evictions, 1,
        "anti-thrashing: `a` is evicted exactly once, never oscillates once restored"
    );
    assert!(events.iter().any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == a)));
    assert!(events.iter().any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == b)));
    assert!(sched.is_idle());
}

/// Scenario 4 — anti-thrashing: with capacity to spare, the scheduler
/// performs **no** evictions at all (P4-07, GitHub #125's acceptance
/// criterion: eviction runs only on the admission-refusal path — a run
/// with sufficient capacity performs none).
#[test]
fn sufficient_capacity_performs_no_evictions() {
    let mut sched = sched_with(64);
    for _ in 0..4 {
        sched.submit(input(8), RequestClass::Agent).unwrap();
    }
    let events = run_to_idle(&mut sched);
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::Evicted { .. })),
        "four requests well within every budget trigger no eviction at all"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, SchedEvent::Done { .. }))
            .count(),
        4
    );
    assert!(sched.is_idle());
}

/// Scenario 5 — a request holding (or holding open) a shared prefix is
/// never an eviction victim (P4-10, GitHub #126's `IGNIS_SEQ_ERR_SHARED_PREFIX`:
/// its leading pages are not its own, so there is no whole-sequence blob to
/// snapshot). With the resident-slot budget pinned to exactly the two
/// prefix-sharing requests, a third candidate can never materialize — there
/// is capacity pressure, but no eligible victim — so it stays queued
/// indefinitely rather than the scheduler evicting a prefix holder anyway.
#[test]
fn a_request_holding_a_shared_prefix_is_never_an_eviction_victim() {
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            max_in_flight: 3,
            max_prefill_batch: 8,
            resident_slot_capacity: 2, // exactly enough for the two prefix-sharing requests
            host_capacity_bytes: 64,
            ..SchedulerConfig::default()
        },
        Arc::new(MockCompute::new()),
    );

    // `main`: a 16-token prompt (exactly one page — the whole prompt is
    // shareable, so its own prefill is a single chunk with no publish-
    // boundary cut) and a long generation budget, so it stays `Running`
    // for the whole test.
    let main = sched
        .submit(
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: (1..=16).collect(),
                params: DecodeParams {
                    max_tokens: Some(50),
                    ..DecodeParams::default()
                },
            },
            RequestClass::Agent,
        )
        .unwrap();
    sched.advance(); // main: one chunk, publishes the prefix, admitted

    // `sub`: claims `main`'s shared 16-token head, prefills only its own
    // 4-token tail.
    let sub = sched
        .submit(
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: (1..=16).chain(100..104).collect(),
                params: DecodeParams {
                    max_tokens: Some(50),
                    ..DecodeParams::default()
                },
            },
            RequestClass::Agent,
        )
        .unwrap();
    sched.advance(); // sub: claims the prefix, prefills the tail, admitted

    assert_eq!(sched.prefix_pinned_pages(), 1, "the shared page is pinned once");
    assert_eq!(
        sched.request_state(main),
        Some(ignis_core::types::RequestState::Running)
    );
    assert_eq!(
        sched.request_state(sub),
        Some(ignis_core::types::RequestState::Running)
    );

    // A third, unrelated candidate cannot materialize: the resident-slot
    // budget (2) is spent by `main` and `sub`, and neither is an eligible
    // victim (both hold the shared prefix) — it stays `Admitted`, queued,
    // over several ticks, and neither `main` nor `sub` is ever evicted.
    let third = sched.submit(input(4), RequestClass::Agent).unwrap();
    let mut events = Vec::new();
    for _ in 0..5 {
        events.extend(sched.advance());
    }

    assert_eq!(
        sched.request_state(third),
        Some(ignis_core::types::RequestState::Admitted),
        "the third candidate never materializes: no eligible victim exists"
    );
    assert!(
        !events.iter().any(|e| matches!(
            e,
            SchedEvent::Evicted { request, .. } if *request == main || *request == sub
        )),
        "a prefix-holding request (publisher or claimant) is never evicted"
    );
}
