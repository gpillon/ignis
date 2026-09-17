//! GitHub #178 — how the scheduler carries a multimodal request, on a CPU
//! (`MockCompute`, ADR 0006): its prefill chunks hold at most one media
//! item's placeholders. An eviction snapshots it to KV-RAM like a text
//! request (GitHub #194: the blob carries its `rope_delta`).
//!
//! GitHub #193 — it publishes and claims shared prefixes like a text request,
//! under an identity that is its token ids **and** its images: a sibling
//! sending another picture of the same size never matches past it, and no
//! prefix ever ends inside an image.

use std::sync::Arc;

use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent};
use ignis_core::vision::{Grid, MediaItem, Multimodal, TokenSpan};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";

/// An image of `count` merged tokens at prompt tokens `begin..begin+count`.
fn image(begin: usize, count: usize) -> MediaItem {
    image_with_digest(begin, count, begin as u8)
}

/// [`image`] with content `digest`: two of these differing only in `digest`
/// are two images of the same size, expanding to the same placeholder ids.
fn image_with_digest(begin: usize, count: usize, digest: u8) -> MediaItem {
    MediaItem {
        grid: Grid { t: 1, h: 2, w: 2 * count as u32 },
        token_span: TokenSpan { begin, count },
        patches: vec![0; 4 * count * 1536],
        content_digest: [digest; 32],
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
        system_block_tokens: None,
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
    // The second chunk is cut once more at the 32-token head the request
    // publishes (GitHub #193) — past the second image, not inside it.
    assert_eq!(chunks, [(0, 20), (20, 12), (32, 8)]);
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

/// The leading prompt tokens `request` skipped through a shared prefix.
fn prefix_reuses(events: &[SchedEvent], request: u64) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::PrefixReused { request: r, tokens, .. } if *r == request => Some(*tokens),
            _ => None,
        })
        .collect()
}

/// `request`'s prefill jobs as `(start, width)`.
fn chunks(compute: &MockCompute, request: u64) -> Vec<(u32, usize)> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|job| job.request == request)
        .map(|job| (job.start_position, job.tokens.len()))
        .collect()
}

#[test]
fn a_sibling_sending_the_same_image_claims_the_prefix_and_another_image_never_does() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), SchedulerConfig::default());
    // Four whole pages with an image inside them, then a sibling's own tail.
    let prompt = |digest: u8, tail: usize| {
        let mut input = multimodal_input(64 + tail, vec![image_with_digest(20, 16, digest)], 4);
        input.tokens.truncate(64);
        input.tokens.extend((0..tail as u32).map(|t| 1000 + t));
        input
    };
    let main = sched.submit(prompt(0xAA, 0), RequestClass::Agent).unwrap();
    sched.advance(); // the main prefills and publishes its 64-token head
    let same = sched.submit(prompt(0xAA, 8), RequestClass::Agent).unwrap();
    let other = sched.submit(prompt(0xBB, 8), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);

    assert!(prefix_reuses(&events, main).is_empty());
    assert_eq!(prefix_reuses(&events, same), [64], "the same picture shares the head: {events:?}");
    assert_eq!(chunks(&compute, same), [(64, 8)], "and prefills only its tail");
    assert!(
        prefix_reuses(&events, other).is_empty(),
        "equal token ids, another picture: nothing is shared: {events:?}"
    );
    assert_eq!(chunks(&compute, other)[0].0, 0, "it prefills its whole prompt");
    assert_eq!(sched.sibling_prefix_reused_tok(), 64);
}

#[test]
fn siblings_in_one_batch_with_different_images_each_publish_their_own_head() {
    // Two requests arriving together see an empty cache, and only one of two
    // *identical* heads is published. Same-size images make the token ids
    // identical, but not the heads: each publishes, and a later sibling
    // sending either image claims that one.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), SchedulerConfig::default());
    let prompt = |digest: u8| multimodal_input(72, vec![image_with_digest(20, 16, digest)], 4);
    let a = sched.submit(prompt(0xAA), RequestClass::Agent).unwrap();
    let b = sched.submit(prompt(0xBB), RequestClass::Agent).unwrap();
    let twin = sched.submit(prompt(0xAA), RequestClass::Agent).unwrap();
    sched.advance();
    let published = |request: u64| -> Vec<u32> {
        compute
            .prefill_calls()
            .iter()
            .flatten()
            .filter(|job| job.request == request)
            .filter_map(|job| job.publish_prefix.map(|p| p.tokens))
            .collect()
    };
    assert_eq!(published(a), [64]);
    assert_eq!(published(b), [64], "another image is another head");
    assert!(published(twin).is_empty(), "the same image is the same head");

    let later = sched.submit(prompt(0xBB), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, later), [64], "{events:?}");
}

#[test]
fn a_multimodal_claim_always_leaves_a_tail_to_prefill() {
    // A claim reaching the prompt's end would prefill nothing, and nothing
    // else hands the leaf the request's rope_delta: the identical prompt is
    // not given the whole-prompt head.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), SchedulerConfig::default());
    let prompt = || multimodal_input(64, vec![image_with_digest(20, 16, 0xAA)], 4);
    sched.submit(prompt(), RequestClass::Agent).unwrap();
    sched.advance();
    let twin = sched.submit(prompt(), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(prefix_reuses(&events, twin).is_empty(), "{events:?}");
    assert!(
        compute
            .prefill_calls()
            .iter()
            .flatten()
            .filter(|job| job.request == twin)
            .all(|job| !job.tokens.is_empty() && job.multimodal.is_some()),
        "the twin prefills, and so sets its rope delta"
    );
}

#[test]
fn a_multimodal_prefix_is_never_published_inside_an_image() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), SchedulerConfig::default());
    // 72 tokens floor to a 64-token head, which is inside the image at
    // 40..70: the head walks back to the page holding its first placeholder.
    let prompt = |digest: u8| multimodal_input(72, vec![image_with_digest(40, 30, digest)], 4);
    let main = sched.submit(prompt(0xAA), RequestClass::Agent).unwrap();
    sched.advance();
    let published: Vec<u32> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|job| job.request == main)
        .filter_map(|job| job.publish_prefix.map(|p| p.tokens))
        .collect();
    assert_eq!(published, [32]);
    assert_eq!(chunks(&compute, main)[0], (0, 32), "the chunk is cut there too");

    // The head before the image is the same history whatever the picture.
    let other = sched.submit(prompt(0xBB), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(prefix_reuses(&events, other), [32], "{events:?}");
    for job in compute.prefill_calls().iter().flatten() {
        if let Some(at) = job.publish_prefix.map(|p| p.tokens) {
            assert!(!(40 < at && at < 70), "published inside the image: {job:?}");
        }
    }
}

#[test]
fn an_evicted_multimodal_request_is_snapshotted_and_restored_not_reprefilled() {
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
    assert!(sched.host_tier().used_bytes() > 0, "the victim's blob went to KV-RAM");
    assert!(!compute.released_requests().contains(&victim), "its state was snapshotted, not dropped");
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == head)),
        "{events:?}"
    );

    // It comes back from its blob and never re-prefills.
    let before = compute.prefill_calls().len();
    let rest = run_to_idle(&mut sched);
    assert!(
        rest.iter().any(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == victim)),
        "{rest:?}"
    );
    assert!(
        !events.iter().chain(&rest).any(|e| matches!(e, SchedEvent::Requeued { .. })),
        "nothing is requeued"
    );
    assert!(
        !compute.prefill_calls()[before..].iter().flatten().any(|job| job.request == victim),
        "the victim is not prefilled again"
    );
    let done = rest.iter().filter(|e| matches!(e, SchedEvent::Done { .. })).count();
    assert_eq!(done, 9, "{rest:?}");
}
