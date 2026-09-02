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
//! `host_capacity_pages` is the knob that drives the tier's bounded
//! behavior: large (no discards, everything restores) vs small (the tier
//! evicts to stay within budget, discarding the oldest snapshot).

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
/// constraint) and a host tier of `host_pages` pages. `max_in_flight` is
/// raised above the 8 resident lanes so the overflow (beyond-N) requests
/// can be submitted (the host-tier overflow path, core-06).
fn sched_with(host_pages: u32) -> ConcreteScheduler {
    let compute = Arc::new(MockCompute::new());
    let cfg = SchedulerConfig {
        model: "qwen3.8-27b".into(),
        max_in_flight: 16, // 8 resident lanes + the overflow beyond them
        max_prefill_batch: 8,
        host_capacity_pages: host_pages,
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
            SchedEvent::Evicted { request } if kind(e) => Some(*request),
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
        sched.host_tier().used_pages() <= sched.host_tier().capacity_pages(),
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
/// LRU) snapshot to make room, so it never exceeds `capacity_pages`; the
/// discarded snapshot's request is re-queued (it re-prefills later), while
/// the retained snapshots are still restored.
#[test]
fn evictions_are_bounded_under_overflow_load() {
    // A 2-page tier holding 1-page snapshots: at most two fit, so a third
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
            sched.host_tier().used_pages() <= sched.host_tier().capacity_pages(),
            "the host tier must stay within its {}-page budget (holds {})",
            sched.host_tier().capacity_pages(),
            sched.host_tier().used_pages()
        );
    }

    // The tier filled (three 1-page captures into a 2-page tier), so at
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
