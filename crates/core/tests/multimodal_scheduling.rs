//! GitHub #178 — how the scheduler carries a multimodal request, on a CPU
//! (`MockCompute`, ADR 0006): its prefill chunks hold at most one media
//! item's placeholders, it neither publishes nor claims a shared prefix, and
//! an eviction releases and re-prefills it instead of snapshotting it to
//! KV-RAM.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent};
use ignis_core::vision::{Grid, MediaItem, Multimodal, TokenSpan};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";

/// An image of `count` merged tokens at prompt tokens `begin..begin+count`.
fn image(begin: usize, count: usize) -> MediaItem {
    MediaItem {
        grid: Grid { t: 1, h: 2, w: 2 * count as u32 },
        token_span: TokenSpan { begin, count },
        patches: vec![0; 4 * count * 1536],
        content_digest: [begin as u8; 32],
    }
}

/// A `tokens`-long prompt carrying `media`, generating up to `max` tokens.
fn multimodal_input(tokens: usize, media: Vec<MediaItem>, max: u32) -> RequestInput {
    let positions = (0..3).flat_map(|_| 0..tokens as i32).collect();
    RequestInput {
        model: MODEL.into(),
        tokens: (0..tokens as u32).collect(),
        params: DecodeParams { max_tokens: Some(max), ..DecodeParams::default() },
        multimodal: Some(Arc::new(Multimodal { positions, rope_delta: -2, media })),
        opener_tokens: None,
        user_turn_tokens: None,
    }
}

fn text_input(tokens: usize, max: u32) -> RequestInput {
    RequestInput {
        multimodal: None,
        ..multimodal_input(tokens, Vec::new(), max)
    }
}

fn scheduler(compute: Arc<MockCompute>, config: SchedulerConfig) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..config },
        compute,
    )
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

#[test]
fn a_prefill_chunk_carries_at_most_one_media_item() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig { serving_chunk_tokens: 64, ..SchedulerConfig::default() },
    );
    // Two images inside what would otherwise be one 40-token chunk.
    sched
        .submit(multimodal_input(40, vec![image(5, 10), image(20, 10)], 2), RequestClass::Agent)
        .unwrap();
    run_to_idle(&mut sched);

    let chunks: Vec<(u32, usize)> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .map(|job| {
            assert!(job.multimodal.is_some(), "every chunk carries the multimodal part");
            (job.start_position, job.tokens.len())
        })
        .collect();
    assert_eq!(chunks, [(0, 20), (20, 20)]);
}

#[test]
fn a_capped_multimodal_request_ends_a_fresh_batch() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig { serving_chunk_tokens: 64, ..SchedulerConfig::default() },
    );
    // Fits one serving chunk by length, but its second image makes it two:
    // the text request queued behind it waits, so only one request is ever
    // mid-prefill.
    sched
        .submit(multimodal_input(40, vec![image(5, 10), image(20, 10)], 2), RequestClass::Agent)
        .unwrap();
    sched.submit(text_input(8, 2), RequestClass::Agent).unwrap();
    sched.advance();
    let first = &compute.prefill_calls()[0];
    assert_eq!(first.len(), 1, "{first:?}");
    assert_eq!(first[0].tokens.len(), 20);
}

#[test]
fn a_multimodal_request_neither_publishes_nor_claims_a_shared_prefix() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), SchedulerConfig::default());
    // Several whole pages long, identical, one after the other: text prompts
    // like these would publish and then claim.
    let prompt = || multimodal_input(80, vec![image(10, 20)], 2);
    sched.submit(prompt(), RequestClass::Agent).unwrap();
    let mut events = run_to_idle(&mut sched);
    sched.submit(prompt(), RequestClass::Agent).unwrap();
    events.extend(run_to_idle(&mut sched));

    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::PrefixReused { .. })),
        "{events:?}"
    );
    for job in compute.prefill_calls().iter().flatten() {
        assert_eq!(job.publish_prefix_tokens, None, "{job:?}");
        assert_eq!(job.shared_prefix, None, "{job:?}");
        assert_eq!(job.start_position, 0, "the second request prefills its whole prompt");
    }
}

#[test]
fn an_evicted_multimodal_request_is_released_and_reprefilled_not_snapshotted() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            max_in_flight: 16,
            max_prefill_batch: 8,
            host_capacity_bytes: 64,
            ..SchedulerConfig::default()
        },
    );
    // Eight multimodal requests fill every resident lane; a ninth has to
    // evict one of them.
    let fillers: Vec<u64> = (0..8)
        .map(|_| sched.submit(multimodal_input(4, vec![image(1, 2)], 8), RequestClass::Agent).unwrap())
        .collect();
    sched.advance();
    let head = sched.submit(text_input(4, 8), RequestClass::Agent).unwrap();
    let events = sched.advance();

    let evicted: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Evicted { request, .. } => Some(*request),
            _ => None,
        })
        .collect();
    assert_eq!(evicted.len(), 1, "{events:?}");
    let victim = evicted[0];
    assert!(fillers.contains(&victim));
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Requeued { request } if *request == victim)),
        "the victim goes back to the queue: {events:?}"
    );
    assert_eq!(sched.host_tier().used_bytes(), 0, "nothing was snapshotted");
    assert!(compute.released_requests().contains(&victim), "its device state was released");
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == head)),
        "{events:?}"
    );

    // It re-prefills from the start and finishes like everyone else.
    let before = compute.prefill_calls().len();
    let rest = run_to_idle(&mut sched);
    assert!(
        compute.prefill_calls()[before..]
            .iter()
            .flatten()
            .any(|job| job.request == victim && job.start_position == 0),
        "the victim re-prefills its whole prompt"
    );
    let done = rest.iter().filter(|e| matches!(e, SchedEvent::Done { .. })).count();
    assert_eq!(done, 9, "{rest:?}");
}
