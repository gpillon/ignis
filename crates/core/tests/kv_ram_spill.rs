//! GitHub #190 (ADR 0029, spec `.scratch/kv-reuse/specs/01-cross-request-reuse.md`
//! §"Retained state in KV-RAM") — retained state the device gives up is
//! **spilled** to KV-RAM instead of lost, and comes back from there.
//!
//! What these tests can prove without a card is what the next layer up
//! observes: which checkpoints cross to KV-RAM and when, which are discarded
//! instead, which reuse a request lands (`reuse_source`), the retained-state
//! lifecycle facts, and that the pool and the tier are charged exactly what
//! they hold. That the bytes that come back are the *right* bytes
//! needs the model, and lives in `prompt_checkpoint_gpu.rs` and
//! `crates/runtime/tests/cuda_leaf_kv_ram_gpu.rs`.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute`, whose
//! checkpoint image, materialized blob and live snapshot are one nominal byte
//! each — so `host_capacity_bytes` counts entries.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ignis_core::checkpoint::{RetainedKind, RetainedStateOperation, ReuseSource};
use ignis_core::host::{RetainedBlob, Tier};
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
const PAGE: u32 = 16;

fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

fn input(prompt: Vec<u32>, opener: Option<u32>, max: u32) -> RequestInput {
    RequestInput {
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
    }
}

/// A device pool of exactly one full sequence (80 pages = 1280 tokens), so a
/// request needing all of it takes the retained pages back — and `host`
/// KV-RAM entries below it. Plus one page: the one a checkpoint's opener ends
/// inside, which a request claiming that checkpoint cannot take back from it
/// (GitHub #215; a real load's plan reserves one per retained slot).
fn tight(host: u64) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_in_flight: 16,
        max_sequence_tokens: 1280,
        kv_capacity_pages: 80 + 1,
        kv_page_tokens: PAGE,
        host_capacity_bytes: host,
        ..SchedulerConfig::default()
    }
}

/// A clock the test moves by hand.
fn manual_clock() -> (ignis_core::Clock, Arc<AtomicU64>) {
    let base = Instant::now();
    let offset = Arc::new(AtomicU64::new(0));
    let read = offset.clone();
    (
        Arc::new(move || base + Duration::from_secs(read.load(Ordering::SeqCst))),
        offset,
    )
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// A system-and-tools block of four pages, published and retained as a prefix
/// (GitHub #188) — the shape every qwen-code request has.
const BLOCK: u32 = 64;

/// A conversation's turn: 1200 prompt tokens starting at `start`, opener at
/// 1150 — more than KV-RAM's 1024-token restore floor past a block, so its
/// checkpoint is worth restoring from there even while the block is on the
/// device. No block of its own: see [`with_block`].
fn turn_at(start: u32) -> RequestInput {
    input(tokens(start, 1200), Some(1150), 4)
}

/// Its next turn: the first up to the opener, then 100 new tokens and an
/// opener of its own.
fn next_turn_at(start: u32) -> RequestInput {
    input([tokens(start, 1150), tokens(start + 50_000, 100)].concat(), Some(1240), 4)
}

/// `request`, opening with a [`BLOCK`]-token system block.
fn with_block(request: RequestInput) -> RequestInput {
    RequestInput {
        system_block_tokens: Some(BLOCK),
        ..request
    }
}

/// A request that needs the whole device pool, so every retained page has to
/// come back for it.
fn whole_pool(start: u32) -> RequestInput {
    input(tokens(start, 1272), None, 8)
}

fn retained_state(
    events: &[SchedEvent],
) -> Vec<(RetainedStateOperation, ReuseSource, RetainedKind)> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::RetainedState { operation, source, kind } => {
                Some((*operation, *source, *kind))
            }
            _ => None,
        })
        .collect()
}

/// Both kinds together — what the fact said before it named one (GitHub #216).
fn count(events: &[SchedEvent], operation: RetainedStateOperation, source: ReuseSource) -> usize {
    retained_state(events)
        .into_iter()
        .filter(|&(o, s, _)| (o, s) == (operation, source))
        .count()
}

/// One kind alone.
fn count_kind(
    events: &[SchedEvent],
    operation: RetainedStateOperation,
    source: ReuseSource,
    kind: RetainedKind,
) -> usize {
    retained_state(events)
        .into_iter()
        .filter(|&fact| fact == (operation, source, kind))
        .count()
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

/// Leave `start`'s conversation turn on the device, then push it to KV-RAM.
/// Returns the turn's request id.
fn turn_spilled_to_kv_ram(
    sched: &mut ConcreteScheduler,
    turn: RequestInput,
    class: RequestClass,
) -> RequestId {
    let pressure = turn.tokens[0] + 100_000;
    let turn = sched.submit(turn, class).unwrap();
    run_to_idle(sched);
    sched
        .submit(whole_pool(pressure), RequestClass::Interactive)
        .unwrap();
    run_to_idle(sched);
    turn
}

// ── What the scheduler reports it is holding (GitHub #216) ───────────────

#[test]
fn occupancy_reports_both_pools_and_empties_when_nothing_is_retained() {
    // The plain field reads ADR 0030 routes through the tick. With prompt
    // reuse off nothing is left behind, so the pool must come back to zero on
    // the very step that releases the last request — there is no later one:
    // the server sends no tick while the scheduler is idle.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig { prompt_reuse: false, retained_slots: 0, ..tight(4) },
        compute.clone(),
    );
    let empty = sched.occupancy();
    assert_eq!(empty.kv_used_pages, 0);
    assert_eq!(empty.kv_pool_pages, 81, "the pool it is empty of is named too");
    assert_eq!(empty.kv_ram_used_bytes, 0);

    sched.submit(turn_at(1), RequestClass::Interactive).unwrap();
    sched.advance();
    let busy = sched.occupancy();
    assert!(busy.kv_used_pages > 0, "the request holds pages while it runs");
    assert_eq!(busy.kv_pool_pages, empty.kv_pool_pages, "capacity is a constant");

    run_to_idle(&mut sched);
    assert_eq!(sched.occupancy().kv_used_pages, 0, "and gives them all back");
}

#[test]
fn occupancy_reports_the_kv_ram_arena_a_spill_filled() {
    // The arena's used bytes are the host tier's own accounting: live
    // snapshots and retained blobs together. Nothing else in the process
    // knows the figure, which is why ADR 0030 takes it off the scheduler
    // rather than by asking the leaf.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(2), compute.clone());
    assert_eq!(sched.occupancy().kv_ram_used_bytes, 0, "nothing spilled yet");

    turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Agent);
    assert_eq!(sched.occupancy().kv_ram_used_bytes, 1, "the blob is charged to the arena");
    assert_eq!(
        sched.occupancy().kv_ram_used_bytes,
        sched.host().used_bytes(),
        "and it is the tier's own figure, not a second count of it"
    );

    // Lane pressure that makes room by giving the retained blob up: a
    // retained bet goes before an evicted live sequence
    // (`a_retained_agent_checkpoint_is_discarded_before_an_evicted_interactive_sequence`).
    for i in 0..8 {
        sched
            .submit(input(tokens(300_000 + i * 10, 4), None, 60), RequestClass::Interactive)
            .unwrap();
    }
    sched.advance();
    for i in 0..2 {
        sched
            .submit(input(tokens(400_000 + i * 10, 4), None, 4), RequestClass::Interactive)
            .unwrap();
        sched.advance();
    }
    assert_eq!(sched.host().retained_count(), 0, "the pressure took the blob");
    assert_eq!(
        sched.occupancy().kv_ram_used_bytes,
        sched.host().used_bytes(),
        "and the arena still reports the tier, live snapshots included"
    );
}

// ── AC1: an idle conversation pushed off the device resumes from KV-RAM ──

#[test]
fn an_idle_conversation_pushed_off_the_device_resumes_from_kv_ram_and_keeps_going() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(4), compute.clone());

    let n = sched.submit(with_block(turn_at(1)), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let pressure = sched.submit(whole_pool(100_000), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(events
        .iter()
        .any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == pressure)));
    assert_eq!(compute.spilled_checkpoints(), vec![n], "the device gave turn N up to KV-RAM");
    assert!(compute.released_checkpoints().is_empty(), "rather than discarding it");
    assert_eq!(
        count(&events, RetainedStateOperation::Spill, ReuseSource::KvRam),
        2,
        "the checkpoint, and the block under it (GitHub #190)"
    );
    // And the two are told apart, not summed (GitHub #216).
    for kind in RetainedKind::ALL {
        assert_eq!(
            count_kind(&events, RetainedStateOperation::Spill, ReuseSource::KvRam, kind),
            1,
            "one spill of each kind, not two of one: {kind:?}"
        );
    }
    assert_eq!(compute.spilled_prefixes(), vec![(n, BLOCK)]);
    let spilled = RetainedBlob::Checkpoint(sched.checkpoint_pool().entries()[0].id);
    assert_eq!(sched.host().retained(spilled).unwrap().tier, Tier::Probation);

    // Turn N+1 resumes from the blob.
    let n1 = sched
        .submit(with_block(next_turn_at(1)), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::KvRam, 1150)]);
    assert!(
        compute.returned_prefixes().is_empty(),
        "the checkpoint reaches further than the block: nothing else comes back"
    );
    assert_eq!(count(&events, RetainedStateOperation::Hit, ReuseSource::KvRam), 1);
    assert_eq!(count(&events, RetainedStateOperation::Restore, ReuseSource::KvRam), 1);
    assert_eq!(
        sched.checkpoint_pool().reused_tok(),
        1150,
        "chosen once, however many ticks it took to land"
    );
    assert_eq!(
        sched.host().retained(spilled).unwrap().tier,
        Tier::Protected,
        "a landed restore is the proof two-tier eviction waits for"
    );
    assert_eq!(
        sched.host().retained_count(),
        2,
        "and the restore consumed no blob: the checkpoint's and the block's are both there"
    );

    // And the conversation keeps its reuse: turn N+1 — a sequence that owns
    // every page it restored, the block's included — cannot publish at the
    // block any more, which is behind it, and publishes and captures at its
    // own opener instead, so turn N+2 resumes on the device.
    assert!(
        sched
            .checkpoint_pool()
            .entries()
            .iter()
            .any(|e| e.tier == ReuseSource::Device && e.tokens == 1240),
        "turn N+1 left a device checkpoint at its opener: {:?}",
        sched.checkpoint_pool().entries()
    );
    let n2 = sched
        .submit(
            with_block(input(
                [next_turn_at(1).tokens[..1240].to_vec(), tokens(900_000, 30)].concat(),
                Some(1265),
                4,
            )),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n2), vec![(ReuseSource::Device, 1240)]);
    assert_eq!(
        sched.kv_used_pages(),
        sched.prefix_pinned_pages() + sched.retained_tail_pages(),
        "nothing is charged but the pages the retained chain still holds"
    );
}

#[test]
fn a_kv_ram_claimant_with_no_opener_to_capture_at_pays_no_chunk_split() {
    // What it would publish is a head over history it restored, which only
    // earns a split when a checkpoint is captured there — the rule a device
    // claimant's chained publish already follows.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(4), compute.clone());
    turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Interactive);

    let n1 = sched
        .submit(
            RequestInput {
                opener_tokens: None,
                ..next_turn_at(1)
            },
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::KvRam, 1150)]);
    let widths: Vec<usize> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|job| job.request == n1)
        .map(|job| job.tokens.len())
        .collect();
    assert_eq!(widths, vec![100], "the whole tail in one chunk");
}

// ── AC4: no device-to-host copy while capacity is sufficient ─────────────

#[test]
fn nothing_crosses_to_kv_ram_while_the_device_has_room() {
    let compute = Arc::new(MockCompute::new());
    // Room means retained slots too (GitHub #215): a conversation of two turns
    // holds four — two links of its chain and its two checkpoints — so three
    // of them need twelve.
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            retained_slots: 12,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let mut events = Vec::new();
    for start in [1, 200_000, 400_000] {
        sched.submit(turn_at(start), RequestClass::Interactive).unwrap();
        events.extend(run_to_idle(&mut sched));
        sched.submit(next_turn_at(start), RequestClass::Agent).unwrap();
        events.extend(run_to_idle(&mut sched));
    }

    assert!(compute.spilled_checkpoints().is_empty(), "no blob was written");
    assert_eq!(count(&events, RetainedStateOperation::Spill, ReuseSource::KvRam), 0);
    assert_eq!(sched.host().retained_count(), 0);
    assert_eq!(sched.host().used_bytes(), 0);
    assert_eq!(
        count(&events, RetainedStateOperation::Hit, ReuseSource::Device),
        3,
        "and every next turn still reused its checkpoint on the device"
    );
}

// ── AC3: retained state before evicted live sequences, class aside ───────

#[test]
fn a_retained_agent_checkpoint_is_discarded_before_an_evicted_interactive_sequence() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(2), compute.clone());

    let agent_turn = turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Agent);
    assert_eq!(compute.spilled_checkpoints(), vec![agent_turn]);
    assert_eq!(sched.host().used_bytes(), 1, "KV-RAM: the Agent blob");

    // Eight Interactive requests hold every lane; two more Interactive heads
    // each take one by snapshotting a lane holder into KV-RAM. The second
    // snapshot does not fit next to the first *and* the blob.
    for i in 0..8 {
        sched
            .submit(input(tokens(300_000 + i * 10, 4), None, 60), RequestClass::Interactive)
            .unwrap();
    }
    sched.advance();
    let mut events = Vec::new();
    for i in 0..2 {
        sched
            .submit(input(tokens(400_000 + i * 10, 4), None, 4), RequestClass::Interactive)
            .unwrap();
        events.extend(sched.advance());
    }

    let evicted = events
        .iter()
        .filter(|e| matches!(e, SchedEvent::Evicted { .. }))
        .count();
    assert_eq!(evicted, 2, "both heads went through the host tier");
    assert_eq!(
        count(&events, RetainedStateOperation::Discard, ReuseSource::KvRam),
        1,
        "the retained Agent blob made the room"
    );
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::Requeued { .. })),
        "no evicted Interactive sequence was discarded for it"
    );
    assert_eq!(compute.released_checkpoints(), vec![agent_turn], "the blob was released");
    assert_eq!(sched.checkpoint_pool().entry_count(), 0, "and left the pool");
    assert_eq!(sched.host().retained_count(), 0);
}

// ── Spill admission: rank, TTL, budget, failure ───────────────────────────

#[test]
fn a_spill_never_displaces_a_higher_ranked_entry_until_it_has_sat_idle_past_the_ttl() {
    let compute = Arc::new(MockCompute::new());
    let (clock, elapsed) = manual_clock();
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            retained_interactive_ttl: Duration::from_secs(300),
            ..tight(1)
        },
        compute.clone(),
    )
    .with_clock(clock);

    // KV-RAM holds one blob: the main conversation's.
    let main = turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Interactive);
    assert_eq!(compute.spilled_checkpoints(), vec![main]);

    // A subagent's checkpoint ranks below it: its spill is refused and the
    // device discards it instead, leaving the main conversation's in place.
    elapsed.store(299, Ordering::SeqCst);
    let sub = turn_at(200_000);
    let sub = sched.submit(sub, RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    sched.submit(whole_pool(300_000), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.spilled_checkpoints(), vec![main], "no second blob was written");
    assert_eq!(compute.released_checkpoints(), vec![sub], "the subagent's checkpoint went");
    assert_eq!(count(&events, RetainedStateOperation::Discard, ReuseSource::Device), 1);
    assert_eq!(count(&events, RetainedStateOperation::Discard, ReuseSource::KvRam), 0);

    // Past the TTL the main conversation's blob, idle since before either
    // turn, ranks as an Agent's probation entry — older than the newcomer.
    elapsed.store(600, Ordering::SeqCst);
    let later = sched.submit(turn_at(500_000), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    sched.submit(whole_pool(600_000), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.spilled_checkpoints(), vec![main, later]);
    assert_eq!(count(&events, RetainedStateOperation::Discard, ReuseSource::KvRam), 1);
    assert_eq!(count(&events, RetainedStateOperation::Spill, ReuseSource::KvRam), 1);
    let kv_ram: Vec<RequestId> = sched
        .checkpoint_pool()
        .entries()
        .iter()
        .filter(|e| e.tier == ReuseSource::KvRam)
        .map(|e| e.publisher)
        .collect();
    assert_eq!(kv_ram, vec![later], "the idle entry made room for the fresh one");
}

#[test]
fn a_spill_the_leaf_fails_discards_only_the_victim() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(4), compute.clone());
    let kept = turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Agent);

    let doomed = sched.submit(turn_at(200_000), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    compute.fail_spill(doomed);
    sched.submit(whole_pool(300_000), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(compute.spilled_checkpoints(), vec![kept]);
    assert_eq!(compute.released_checkpoints(), vec![doomed], "discarded, not leaked");
    assert_eq!(count(&events, RetainedStateOperation::Discard, ReuseSource::Device), 1);
    assert_eq!(sched.host().retained_count(), 1, "KV-RAM is as it was");
    assert_eq!(sched.kv_used_pages(), 0, "and the device got its pages back either way");
}

#[test]
fn prompt_reuse_without_a_kv_ram_budget_discards_what_the_device_gives_up() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(0), compute.clone());
    let turn = turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Interactive);

    assert!(compute.spilled_checkpoints().is_empty());
    assert_eq!(compute.released_checkpoints(), vec![turn]);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
}

// ── A restore that has not landed ────────────────────────────────────────

#[test]
fn a_kv_ram_restore_is_promoted_only_when_it_lands() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(4), compute.clone());
    turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Interactive);
    let blob = RetainedBlob::Checkpoint(sched.checkpoint_pool().entries()[0].id);

    let n1 = sched.submit(next_turn_at(1), RequestClass::Interactive).unwrap();
    compute.fail_prefill(n1);
    let events = sched.advance();
    assert_eq!(count(&events, RetainedStateOperation::Hit, ReuseSource::KvRam), 1, "chosen");
    assert_eq!(count(&events, RetainedStateOperation::Restore, ReuseSource::KvRam), 0);
    assert_eq!(
        sched.host().retained(blob).unwrap().tier,
        Tier::Probation,
        "a restore that failed proved nothing"
    );

    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::KvRam, 1150)], "the retry landed it");
    assert_eq!(
        count(&events, RetainedStateOperation::Hit, ReuseSource::KvRam),
        0,
        "the retry keeps the claim it had rather than choosing again"
    );
    assert_eq!(count(&events, RetainedStateOperation::Restore, ReuseSource::KvRam), 1);
    assert_eq!(sched.host().retained(blob).unwrap().tier, Tier::Protected);
}

#[test]
fn a_request_cancelled_before_its_kv_ram_restore_lands_lets_the_blob_go() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(4), compute.clone());
    turn_spilled_to_kv_ram(&mut sched, turn_at(1), RequestClass::Interactive);
    let blob = RetainedBlob::Checkpoint(sched.checkpoint_pool().entries()[0].id);

    let n1 = sched.submit(next_turn_at(1), RequestClass::Interactive).unwrap();
    compute.fail_prefill(n1);
    sched.advance();
    sched.cancel(n1);
    run_to_idle(&mut sched);

    // Unpromoted, and discardable again: a live snapshot's room takes it.
    let entry = sched.host().retained(blob).expect("still retained");
    assert_eq!(entry.tier, Tier::Probation);
    assert!(sched.host().plan_retained_room(4, RequestClass::Interactive, Instant::now(), Instant::now()).is_some());
}

// ── Miss facts ───────────────────────────────────────────────────────────

#[test]
fn a_first_turn_misses_every_configured_tier_once() {
    let compute = Arc::new(MockCompute::new());
    let mut with_kv_ram = ConcreteScheduler::with_config(tight(4), compute.clone());
    with_kv_ram.submit(turn_at(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut with_kv_ram);
    assert_eq!(count(&events, RetainedStateOperation::Miss, ReuseSource::Device), 1);
    assert_eq!(count(&events, RetainedStateOperation::Miss, ReuseSource::KvRam), 1);
    // Every miss is a checkpoint miss (GitHub #216): the checkpoint pool's
    // lookup is the only one that reports one, and the prefix walk beside it
    // records none — no prefix miss is invented to make the two symmetric.
    for source in [ReuseSource::Device, ReuseSource::KvRam] {
        assert_eq!(
            count_kind(&events, RetainedStateOperation::Miss, source, RetainedKind::Checkpoint),
            1
        );
        assert_eq!(
            count_kind(&events, RetainedStateOperation::Miss, source, RetainedKind::Prefix),
            0,
            "no prefix miss is synthesised"
        );
    }

    let mut device_only = ConcreteScheduler::with_config(tight(0), Arc::new(MockCompute::new()));
    device_only.submit(turn_at(1), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut device_only);
    assert_eq!(count(&events, RetainedStateOperation::Miss, ReuseSource::Device), 1);
    assert_eq!(
        count(&events, RetainedStateOperation::Miss, ReuseSource::KvRam),
        0,
        "a tier this load does not carry is never missed"
    );
}

#[test]
fn a_superseded_checkpoint_is_reported_as_a_discard_from_its_tier() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    // A tool loop: three iterations of one turn keep two checkpoints, so the
    // first is superseded by the third.
    let mut prompt = tokens(1, 40);
    let mut opener = 37;
    let mut events = Vec::new();
    for step in 0..3 {
        let request = RequestInput {
            user_turn_tokens: Some(0),
            ..input(prompt.clone(), Some(opener), 4)
        };
        sched.submit(request, RequestClass::Agent).unwrap();
        events.extend(run_to_idle(&mut sched));
        prompt = [prompt[..opener as usize].to_vec(), tokens(10_000 * (step + 1), 40)].concat();
        opener += 37;
    }
    assert!(!compute.released_checkpoints().is_empty(), "the loop superseded an entry");
    assert_eq!(
        compute.released_checkpoints().len(),
        count(&events, RetainedStateOperation::Discard, ReuseSource::Device),
        "every image released is a discard reported"
    );
}

// ── Retained prefixes in KV-RAM ──────────────────────────────────────────

/// A system block past the restore floor: 68 pages.
const BIG_BLOCK: u32 = 1088;

/// A burst member: the shared block, then a question of its own. No opener,
/// so it leaves no checkpoint and only the block is retained.
fn subagent(question: u32) -> RequestInput {
    RequestInput {
        system_block_tokens: Some(BIG_BLOCK),
        ..input([tokens(1, BIG_BLOCK), tokens(question, 40)].concat(), None, 4)
    }
}

fn prefix_reuses(events: &[SchedEvent], request: RequestId) -> Vec<(u32, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::PrefixReused {
                request: r,
                tokens,
                retained,
            } if *r == request => Some((*tokens, *retained)),
            _ => None,
        })
        .collect()
}

fn widths(compute: &MockCompute, request: RequestId) -> Vec<usize> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|job| job.request == request)
        .map(|job| job.tokens.len())
        .collect()
}

#[test]
fn a_burst_block_the_device_gave_up_comes_back_once_and_the_burst_shares_it() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(8), compute.clone());

    let first = sched.submit(subagent(10_000), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.prefix_pinned_pages(), BIG_BLOCK / PAGE, "the block is retained");

    sched.submit(whole_pool(900_000), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.spilled_prefixes(), vec![(first, BIG_BLOCK)], "the device gave it to KV-RAM");
    assert_eq!(count(&events, RetainedStateOperation::Spill, ReuseSource::KvRam), 1);
    assert_eq!(sched.prefix_pinned_pages(), 0, "and its pages came back");
    assert_eq!(sched.host().retained_count(), 1);

    // The next member brings it back — one restore — and claims it in place.
    let second = sched.submit(subagent(20_000), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.returned_prefixes(), vec![(first, BIG_BLOCK)]);
    assert_eq!(count(&events, RetainedStateOperation::Hit, ReuseSource::KvRam), 1);
    assert_eq!(count(&events, RetainedStateOperation::Restore, ReuseSource::KvRam), 1);
    assert_eq!(prefix_reuses(&events, second), vec![(BIG_BLOCK, true)]);
    assert_eq!(widths(&compute, second), vec![40], "it prefilled its own question only");
    assert_eq!(sched.prefix_pinned_pages(), BIG_BLOCK / PAGE, "the block is on the device again");

    // Every later member shares it there: no second restore.
    let third = sched.submit(subagent(30_000), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.returned_prefixes().len(), 1);
    assert_eq!(prefix_reuses(&events, third), vec![(BIG_BLOCK, true)]);

    // Given up again, it copies nothing: the blob is still in KV-RAM...
    sched.submit(whole_pool(700_000), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(compute.spilled_prefixes().len(), 1, "no second device-to-host copy");
    assert_eq!(sched.prefix_pinned_pages(), 0);
    assert_eq!(sched.host().retained_count(), 1);

    // ...and brings the block back again.
    let fourth = sched.submit(subagent(40_000), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(compute.returned_prefixes().len(), 2);
    assert_eq!(prefix_reuses(&events, fourth), vec![(BIG_BLOCK, true)]);
    assert_eq!(
        sched.kv_used_pages(),
        sched.prefix_pinned_pages() + sched.retained_tail_pages(),
        "nothing is charged but the block on the device"
    );
}

#[test]
fn a_block_short_of_the_restore_floor_stays_in_kv_ram() {
    // Four pages cost less to prefill than to bring across the bus.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(8), compute.clone());
    let small = |question| RequestInput {
        system_block_tokens: Some(BLOCK),
        ..input([tokens(1, BLOCK), tokens(question, 40)].concat(), None, 4)
    };
    let first = sched.submit(small(10_000), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    sched.submit(whole_pool(900_000), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(compute.spilled_prefixes(), vec![(first, BLOCK)]);

    let second = sched.submit(small(20_000), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(compute.returned_prefixes().is_empty());
    assert!(prefix_reuses(&events, second).is_empty(), "prefilled cold");
    assert_eq!(widths(&compute, second).iter().sum::<usize>(), (BLOCK + 40) as usize);
}

#[test]
fn a_subagent_block_in_kv_ram_gives_way_to_the_main_conversation() {
    // One blob fits. The burst's block is an Agent's bet; the main
    // conversation's checkpoint outranks it.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(tight(1), compute.clone());
    let first = sched.submit(subagent(10_000), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    sched.submit(whole_pool(900_000), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(compute.spilled_prefixes(), vec![(first, BIG_BLOCK)]);

    let main = turn_spilled_to_kv_ram(&mut sched, turn_at(300_000), RequestClass::Interactive);
    assert_eq!(compute.spilled_checkpoints(), vec![main]);
    assert_eq!(compute.discarded_spilled_prefixes(), vec![(first, BIG_BLOCK)], "the block's blob went");

    let second = sched.submit(subagent(20_000), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(compute.returned_prefixes().is_empty(), "there is nothing left to bring back");
    assert!(prefix_reuses(&events, second).is_empty());
}
