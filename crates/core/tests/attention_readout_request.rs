//! GitHub #260 (spec 13, ADR 0038) — a `point` answered in one pass is a
//! **readout-class decision**: one prefill, no decode round, no decode lane,
//! no residency. The scheduler finishes it on its last prefill chunk with
//! the **attention readout**'s scores, as it finishes a readout with its
//! logits.
//!
//! What these tests hold down beyond #238's: the scores ride only the last
//! chunk; a head point whose leaf could not read the keys fails, it is never
//! answered empty; and the last chunk is kept wide enough for the leaf to
//! read the keys at all — by the chunk cut and by the reuse trim alike.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute`
//! behind the `Compute` seam, no GPU anywhere.

use std::sync::Arc;

use ignis_core::pointing::{ATTENTION_MIN_CHUNK_TOKENS, AttentionQuery, AttentionScores, PointingHead, SetQuery};
use ignis_core::scheduler::PrefillJob;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent, TokenId};
use ignis_core::{
    ConcreteScheduler, DecisionRead, FinishReason, MockCompute, Scheduler, SchedulerConfig,
};

const MODEL: &str = "qwen3.8-27b";

const HEAD: PointingHead = PointingHead {
    gqa_ordinal: 9,
    query_head: 10,
};

/// A head point over `prompt`, reading the keys of `[begin, begin + count)`.
fn head_point(prompt: Vec<TokenId>, begin: u32, count: u32) -> RequestInput {
    let opener = prompt.len() as u32;
    RequestInput {
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams::default(),
        multimodal: None,
        opener_tokens: Some(opener),
        user_turn_tokens: None,
        system_block_tokens: None,
        decision: Some(DecisionRead::Attention(AttentionQuery {
            head: HEAD,
            key_begin: begin,
            key_count: count,
            set: None,
        })),
        constrained: None,
    }
}

fn tokens(start: u32, n: u32) -> Vec<TokenId> {
    (start..start + n).collect()
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        ..SchedulerConfig::default()
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    let mut ticks = 0;
    while !sched.is_idle() {
        events.extend(sched.advance());
        ticks += 1;
        assert!(ticks < 200, "the engine never went idle: {events:?}");
    }
    events
}

/// The `Done` event for `request`, as (tokens generated, reason, attention).
fn finish(events: &[SchedEvent], request: RequestId) -> (u32, FinishReason, Option<AttentionScores>) {
    events
        .iter()
        .find_map(|e| match e {
            SchedEvent::Done {
                request: r,
                tokens,
                reason,
                attention,
                readout,
                drawn,
                ..
            } if *r == request => {
                assert!(readout.is_none() && drawn.is_none(), "a head point reads attention only");
                Some((*tokens, *reason, attention.clone()))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("request {request} never finished: {events:?}"))
}

fn jobs_of(compute: &MockCompute, request: RequestId) -> Vec<PrefillJob> {
    compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .filter(|job| job.request == request)
        .collect()
}

#[test]
fn a_head_point_is_answered_at_the_end_of_prefill_and_never_decoded() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let id = sched
        .submit(head_point(tokens(1, 40), 4, 16), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);

    let (generated, reason, attention) = finish(&events, id);
    assert_eq!(generated, 0, "a head point generates nothing");
    assert_eq!(reason, FinishReason::Stop, "it is answered, not cut short");
    let attention = attention.expect("its answer is the attention readout");
    assert_eq!(attention.set_argmax, None, "it named no head set");
    let scores = attention.scores;
    assert_eq!(scores.len(), 16, "one score per key of the span it named");
    assert_eq!(
        scores.iter().position(|&s| s == MockCompute::ATTENTION_PEAK_SCORE),
        Some(MockCompute::attention_peak(16))
    );
    assert!(
        compute.decode_calls().iter().flatten().all(|job| job.request != id),
        "no decode round ever carried it"
    );
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == id)),
        "and it emitted no token"
    );
}

/// Spec 14: a head point naming a head set carries the set — heads, excluded
/// keys and the grid's columns (spec 15) — on its reading chunk, and
/// finishes with one key index per head beside the pointing head's scores,
/// in the set's order, each with the four scores around it.
#[test]
fn a_head_set_rides_the_reading_chunk_and_comes_back_one_key_per_head() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let set = SetQuery {
        heads: Arc::from(vec![HEAD, PointingHead { gqa_ordinal: 7, query_head: 1 }, PointingHead { gqa_ordinal: 15, query_head: 18 }]),
        excluded: Arc::from(vec![0, 6, 15]),
        grid_cols: 4,
    };
    let mut input = head_point(tokens(1, 40), 4, 16);
    input.decision = Some(DecisionRead::Attention(AttentionQuery {
        head: HEAD,
        key_begin: 4,
        key_count: 16,
        set: Some(set.clone()),
    }));
    let id = sched.submit(input, RequestClass::Agent).expect("admitted");
    let events = run_to_idle(&mut sched);

    let asked = jobs_of(&compute, id).last().and_then(|job| job.attention.clone()).expect("the last chunk reads");
    assert_eq!(asked.set.as_ref(), Some(&set), "the job names the set it was given");
    let (generated, reason, attention) = finish(&events, id);
    assert_eq!((generated, reason), (0, FinishReason::Stop));
    let attention = attention.expect("answered");
    assert_eq!(attention.scores.len(), 16);
    let argmax = attention.set_argmax.expect("one key per head of the set");
    assert_eq!(&*argmax, MockCompute::attention_set_argmax(16, 3, &set.excluded).as_slice());
    // The mock's peak is key 5 and the set reads 5, 6 and 7 — 6 is excluded,
    // so the second head moves on to 7.
    assert_eq!(&*argmax, &[5, 7, 7]);
    // Spec 15: the four scores around each peak, and the grid says which of
    // them exist. On a 4x4 grid key 7 is the last column, so it has no
    // neighbour to its right.
    let peak = attention.set_peak.expect("a peak score per head");
    let around = attention.set_neighbours.expect("four neighbours per head");
    assert_eq!(peak.len(), 3);
    assert_eq!(around.len(), 12);
    assert_eq!(
        &*around,
        MockCompute::attention_set_neighbours(&argmax, 16, 4).as_slice(),
    );
    assert!(around[4 * 1 + 1].is_none(), "key 7 is in the last column: no right neighbour");
    assert!(around[4 * 0 + 1].is_some(), "key 5 is not");
}

#[test]
fn a_head_point_never_takes_a_decode_lane() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute);
    let id = sched
        .submit(head_point(tokens(1, 40), 4, 16), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == id)),
        "a head point is never dealt a lane: {events:?}"
    );
}

#[test]
fn the_attention_readout_rides_the_last_chunk_and_no_other() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            serving_chunk_tokens: 16,
            ..config()
        },
        compute.clone(),
    );
    let id = sched
        .submit(head_point(tokens(1, 48), 2, 30), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);

    let asked: Vec<bool> = jobs_of(&compute, id).iter().map(|job| job.attention.is_some()).collect();
    assert!(asked.len() > 1, "the prompt was chunked: {asked:?}");
    assert_eq!(asked.iter().filter(|&&a| a).count(), 1, "exactly one chunk reads: {asked:?}");
    assert_eq!(asked.last(), Some(&true), "and it is the last one");
    assert!(
        jobs_of(&compute, id).iter().all(|job| job.readout.is_none() && job.permitted.is_none()),
        "a head point asks for no logits and restricts no draw"
    );
}

/// A leaf that could not read the keys — a route that materialized none, a
/// span outside the band it materialized — returns no scores, and the
/// question fails. Reporting `Stop` with nothing would be a point answered by
/// nothing, dressed as a point answered.
#[test]
fn a_head_point_the_leaf_could_not_read_ends_in_error() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let id = sched
        .submit(head_point(tokens(1, 40), 4, 16), RequestClass::Agent)
        .expect("admitted");
    compute.refuse_attention(id);
    let events = run_to_idle(&mut sched);
    let (generated, reason, attention) = finish(&events, id);
    assert_eq!((generated, reason), (0, FinishReason::Error));
    assert!(attention.is_none());
}

/// The chunk cut: 20 tokens at a 16-token chunk would leave a last chunk of
/// 4, which under hq-e8-2b takes the small-T route and materializes no keys
/// to read. The cut moves so the last chunk keeps the minimum.
#[test]
fn a_head_points_last_chunk_is_never_narrower_than_the_leaf_can_read() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            serving_chunk_tokens: 16,
            ..config()
        },
        compute.clone(),
    );
    let id = sched
        .submit(head_point(tokens(1, 20), 2, 12), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);

    let widths: Vec<usize> = jobs_of(&compute, id).iter().map(|job| job.tokens.len()).collect();
    assert_eq!(widths.iter().sum::<usize>(), 20, "every prompt token is prefilled once: {widths:?}");
    assert_eq!(
        widths.last().copied(),
        Some(ATTENTION_MIN_CHUNK_TOKENS as usize),
        "the last chunk keeps the minimum: {widths:?}"
    );
    assert_eq!(finish(&events, id).1, FinishReason::Stop);
}

/// An ordinary request over the same prompt is cut where it always was: the
/// rule is the attention readout's, not a new chunking policy for everyone.
#[test]
fn only_an_attention_readout_moves_the_chunk_cut() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            serving_chunk_tokens: 16,
            ..config()
        },
        compute.clone(),
    );
    let input = RequestInput {
        decision: None,
        params: DecodeParams {
            max_tokens: Some(1),
            ..DecodeParams::default()
        },
        ..head_point(tokens(1, 20), 2, 12)
    };
    let id = sched.submit(input, RequestClass::Agent).expect("admitted");
    run_to_idle(&mut sched);
    let widths: Vec<usize> = jobs_of(&compute, id).iter().map(|job| job.tokens.len()).collect();
    assert_eq!(widths, vec![16, 4]);
}

/// The reuse trim: a sibling head point over retained evidence claims no
/// further than its prompt minus the minimum, so its own last chunk — the
/// one that reads — is still wide enough. At a 16-token page and a 72-token
/// prompt that is the difference between claiming 64 (a last chunk of 8) and
/// claiming 48.
#[test]
fn a_head_point_claims_no_prefix_that_would_leave_it_a_narrow_last_chunk() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let over_evidence = |prompt| RequestInput {
        system_block_tokens: Some(72),
        ..head_point(prompt, 2, 30)
    };
    sched
        .submit(over_evidence(tokens(1, 72)), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);
    let second = sched
        .submit(over_evidence(tokens(1, 72)), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);

    let jobs = jobs_of(&compute, second);
    let first = jobs.first().expect("the sibling prefilled");
    assert!(
        first.shared_prefix.is_some() || first.checkpoint.is_some(),
        "the sibling reused the retained evidence: {jobs:?}"
    );
    let last = jobs.last().unwrap();
    assert!(
        last.tokens.len() >= ATTENTION_MIN_CHUNK_TOKENS as usize,
        "its reading chunk carries {} tokens",
        last.tokens.len()
    );
    assert_eq!(finish(&events, second).1, FinishReason::Stop);
}

#[test]
fn an_attention_readout_holds_back_its_tail_from_reuse_and_publish() {
    let input = head_point(tokens(1, 72), 2, 30);
    assert_eq!(input.prefill_tail(), ATTENTION_MIN_CHUNK_TOKENS as usize);
    assert_eq!(input.reuse_reach(), 72 - ATTENTION_MIN_CHUNK_TOKENS as usize);
    assert_eq!(input.publish_reach(), 72 - ATTENTION_MIN_CHUNK_TOKENS as usize);
}
