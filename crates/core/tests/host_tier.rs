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
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        model: "qwen3.8-27b".into(),
        tokens: vec![1, 2, 3, 4],
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
        program: None,
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

/// Scenario 3 — overflow through a burst standing on one prefix: nobody is
/// ever re-prefilled, and nothing oscillates host↔GPU once restored.
///
/// Before GitHub #190 this scenario drove a **half-prefilled** (`Prefilling`,
/// lane-less) request into the host tier, and it could, because the eight
/// lane holders all stood on one shared prefix and a prefix holder was never a
/// victim. It is one now — its snapshot materializes the shared pages — and a
/// lane holder is always preferred over a lane-less candidate
/// (`ConcreteScheduler::evict_one_victim`), so a lane-less `Prefilling`
/// request is only chosen when every lane is protected. That path keeps its
/// coverage where the case it needs can be built directly:
/// `host::tests::a_half_prefilled_entry_carries_its_resume_phase_and_progress`
/// (the tier round-trips an arbitrary boundary) and
/// `request::tests::a_prefilling_request_can_be_evicted_and_restored_to_prefilling`
/// (the state machine), with the lane-less ranking in `admission::tests`.
/// What this one drives end to end is what the burst does now.
#[test]
fn a_burst_on_one_prefix_overflows_through_materialized_snapshots_without_reprefilling() {
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
    let prompt = |tokens: Vec<u32>, max: u32| RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        model: "qwen3.8-27b".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
        program: None,
};

    // Eight fillers share one 16-token prefix (a whole page): the first
    // publishes it, the rest claim it, and together they hold every lane.
    for _ in 0..8 {
        sched
            .submit(prompt((1..=16).collect(), 20), RequestClass::Agent)
            .unwrap();
        sched.advance();
    }
    // Two unshared requests need a lane and a resident slot each.
    let a = sched
        .submit(prompt((1000..1004).collect(), 8), RequestClass::Agent)
        .unwrap();
    let b = sched
        .submit(prompt((2000..2004).collect(), 4), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);

    let evicted: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Evicted { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert!(!evicted.is_empty(), "the overflow went through the host tier");
    for victim in &evicted {
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SchedEvent::Restored { request, .. } if request == victim)),
            "request {victim} is restored from its blob"
        );
        assert_eq!(
            evicted.iter().filter(|r| *r == victim).count(),
            1,
            "anti-thrashing: request {victim} is evicted exactly once"
        );
    }
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::Requeued { .. })),
        "nobody's completed prefill is ever redone"
    );
    for request in [a, b] {
        assert!(events
            .iter()
            .any(|e| matches!(e, SchedEvent::Done { request: r, .. } if *r == request)));
    }
    assert!(sched.is_idle());
    assert_eq!(sched.prefix_pinned_pages(), 0);
    assert_eq!(sched.kv_used_pages(), 0);
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

/// Scenario 5 — a request holding a shared prefix is an eviction victim like
/// any other (GitHub #190). Until then it never was: its leading pages were
/// the prefix's, and the leaf refused to snapshot a sequence that did not own
/// its history (`IGNIS_SEQ_ERR_SHARED_PREFIX`), so a pool full of prefix
/// holders left a new request queued with no victim at all — the shape every
/// subagent burst standing on one system block has. The snapshot now
/// materializes the shared pages, so the victim resumes as a sequence that
/// owns every page: suspended, restored, never re-prefilled.
#[test]
fn a_request_holding_a_shared_prefix_is_evicted_with_its_prefix_materialized() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            max_in_flight: 3,
            max_prefill_batch: 8,
            resident_slot_capacity: 2, // exactly enough for the two prefix-sharing requests
            host_capacity_bytes: 64,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let prompt = |tokens: Vec<u32>, max: u32| RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        model: "qwen3.8-27b".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
        program: None,
};

    // `main` publishes a one-page head; `sub` claims it and prefills only its
    // own 4-token tail. Both hold the prefix, and both keep running.
    let main = sched
        .submit(prompt((1..=16).collect(), 50), RequestClass::Agent)
        .unwrap();
    sched.advance();
    let sub = sched
        .submit(prompt((1..=16).chain(100..104).collect(), 50), RequestClass::Agent)
        .unwrap();
    sched.advance();
    assert_eq!(sched.prefix_pinned_pages(), 1, "the shared page is pinned once");

    // A third candidate needs a resident slot, and both are held by prefix
    // holders — one of them is snapshotted to make room.
    let third = sched.submit(input(4), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);

    let evicted: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Evicted { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert!(
        !evicted.is_empty() && evicted.iter().all(|r| *r == main || *r == sub),
        "a prefix holder was the victim: {evicted:?}"
    );
    for victim in &evicted {
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SchedEvent::Restored { request, .. } if request == victim)),
            "the victim comes back from its blob"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, SchedEvent::Requeued { request } if request == victim)),
            "and is never re-prefilled"
        );
    }
    for request in [main, sub, third] {
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SchedEvent::Done { request: r, .. } if *r == request)),
            "request {request} finishes"
        );
    }
    assert_eq!(sched.prefix_pinned_pages(), 0, "the prefix's pages came back");
    assert_eq!(sched.kv_used_pages(), 0, "and nothing is charged twice or left behind");
}

// ── ADR 0023 / GitHub #127: class-aware GPU-residency eviction ──────────

/// Scenario 6 — the `Agent` request is evicted ahead of an older
/// `Interactive` one (ADR 0023: class outranks LRU on the GPU-residency side,
/// once eligibility and protection are settled). Before GitHub #127 this was
/// plain oldest-submitted, which would have picked the *older*
/// (`Interactive`) candidate here instead.
///
/// Until GitHub #190 both candidates reached this point lane-less, because
/// the fillers' shared prefix kept them from being victims and so from giving
/// up their lanes. Each now takes a filler's lane on arrival, so the choice is
/// made between two lane holders; the lane-less ranking itself is
/// `admission::tests`' `choose_resident_candidate_victim` cases.
#[test]
fn eviction_prefers_agent_over_an_older_interactive_request() {
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            max_in_flight: 11, // 8 fillers + interactive_a + agent_a2 + b
            max_prefill_batch: 8,
            // Exactly enough resident slots for the 8 lane-holding fillers
            // plus both lane-less candidates — none left over for `b`.
            resident_slot_capacity: 10,
            host_capacity_bytes: 64,
            ..SchedulerConfig::default()
        },
        Arc::new(MockCompute::new()),
    );

    // Eight fillers share one 16-token prefix and occupy every lane.
    for _ in 0..8 {
        sched
            .submit(
                RequestInput {
                    decision: None,
                    multimodal: None,
                    opener_tokens: None,
                    user_turn_tokens: None,
                    system_block_tokens: None,
                    model: "qwen3.8-27b".into(),
                    tokens: (1..=16).collect(),
                    params: DecodeParams {
                        max_tokens: Some(20),
                        ..DecodeParams::default()
                    },
                    program: None,
                },
                RequestClass::Interactive,
            )
            .unwrap();
        sched.advance();
    }

    // `interactive_a`: submitted first (older), a unique short prompt —
    // completes its one-chunk prefill, holds no lane, no shared prefix.
    let interactive_a = sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "qwen3.8-27b".into(),
                tokens: (1000..1004).collect(),
                params: DecodeParams {
                    max_tokens: Some(8),
                    ..DecodeParams::default()
                },
                program: None,
            },
            RequestClass::Interactive,
        )
        .unwrap();
    sched.advance();

    // `agent_a2`: submitted second (younger), same shape, `Agent` class.
    let agent_a2 = sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "qwen3.8-27b".into(),
                tokens: (2000..2004).collect(),
                params: DecodeParams {
                    max_tokens: Some(8),
                    ..DecodeParams::default()
                },
                program: None,
            },
            RequestClass::Agent,
        )
        .unwrap();
    sched.advance();

    // `b`: the resident-slot budget (10) is now fully spent (8 fillers +
    // `interactive_a` + `agent_a2`) — materializing it forces an eviction
    // between the two lane-less candidates.
    sched.submit(input(4), RequestClass::Agent).unwrap();
    let ev_b = sched.advance();

    assert!(
        ev_b.iter().any(
            |e| matches!(e, SchedEvent::Evicted { request, .. } if *request == agent_a2)
        ),
        "the Agent candidate is evicted despite being the younger one: class outranks LRU"
    );
    assert!(
        !ev_b.iter().any(
            |e| matches!(e, SchedEvent::Evicted { request, .. } if *request == interactive_a)
        ),
        "the older Interactive candidate is not evicted while a lower-class candidate remains"
    );
}

/// Scenario 7 — the GPU side of ADR 0023's asymmetry, from a `Running`
/// lane: a frozen donor is never a victim, even though its class (`Agent`)
/// would otherwise make it the *preferred* victim by class alone
/// (`retained_lane_is_better_victim`) — protection outranks class. AC:
/// "a protected lane mid-stream is not evicted ahead of an unprotected one
/// of a lower class." ADR 0023's own words: "a protected lane mid-stream
/// is not a victim because its owner is an agent."
#[test]
fn a_protected_donor_lane_is_not_evicted_ahead_of_an_unprotected_lower_class_lane() {
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            max_in_flight: 16,
            max_prefill_batch: 8,
            resident_slot_capacity: 16, // lanes, not resident slots, are the constraint
            host_capacity_bytes: 64, // host tier enabled: eviction can satisfy the head immediately
            ..SchedulerConfig::default() // 4096-page pool: pages are never tight
        },
        Arc::new(MockCompute::new()),
    );

    // Seven Interactive fillers with a generous generation budget.
    for i in 0..7u32 {
        sched.submit(input(30 + i), RequestClass::Interactive).unwrap();
    }
    // One Agent filler with the *shortest* remaining work: by ADR 0004's
    // donor selection (earliest-completion prefix), it is the protection's
    // sole donor once a head blocks — despite `Agent` otherwise being the
    // class-preferred victim.
    let agent_donor = sched.submit(input(5), RequestClass::Agent).unwrap();
    let ev1 = sched.advance(); // step 1: all eight fillers dealt onto the eight lanes

    // The ninth request (the blocked head) opens a protection; the
    // shortest-work incumbent (the Agent filler) freezes as the sole donor.
    // With the host tier enabled, the blocked head is satisfied by eviction
    // within this same `advance()` call, which also clears the protection
    // — so it is read off the `Protected` event, not `sched.protection()`.
    let head = sched.submit(input(8), RequestClass::Interactive).unwrap();
    let ev2 = sched.advance();

    let donors: Vec<u64> = ev2
        .iter()
        .find_map(|e| match e {
            SchedEvent::Protected { donors, .. } => Some(donors.clone()),
            _ => None,
        })
        .expect("a protection was opened for the blocked head");
    assert_eq!(
        donors,
        vec![agent_donor],
        "the Agent filler (shortest remaining work) is the sole frozen donor"
    );

    // The host tier is enabled and has room: the scheduler evicts a lane to
    // admit the head immediately rather than waiting for a natural
    // completion. The evicted lane must never be the protected donor —
    // even though `Agent` alone would normally be the preferred victim.
    let evicted: Vec<u64> = ev2
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Evicted { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert!(
        !evicted.is_empty(),
        "an eviction frees the blocked head's lane"
    );
    assert!(
        !evicted.contains(&agent_donor),
        "the frozen Agent donor is never a victim, protection outranks class"
    );

    // Run to idle: everything (the fillers, the donor, the head, and the
    // evicted request once restored) completes.
    let mut events = ev1.into_iter().chain(ev2).collect::<Vec<_>>();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    assert!(events.iter().any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == head)));
    assert!(events.iter().any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == agent_donor)));
    assert!(sched.is_idle());
}
