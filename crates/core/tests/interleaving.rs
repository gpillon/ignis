//! P3-01 (GitHub #97, ADR 0018) — chunk-level prefill/decode interleaving:
//! `advance()` sends at most one prefill chunk per tick, `Prefilling` is
//! durable (it can carry partial progress across many ticks), exactly one
//! request holds device-resident prefill progress at a time, a failed
//! chunk's retry never re-sends an already-applied span, and cancel is
//! abort (not suspend).
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` (or a
//! thin wrapper over it) that records the call shape behind the `Compute`
//! seam.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ignis_core::scheduler::{Compute, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome};
use ignis_core::types::{
    ComputeError, DecodeParams, FinishReason, RequestClass, RequestInput, RequestState,
    SchedEvent,
};
use ignis_core::{
    Clock, ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig, resolve_serving_chunk_tokens,
};

fn input(tokens: &[u32], max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: "qwen3.8-27b".into(),
        tokens: tokens.to_vec(),
        params: DecodeParams {
            max_tokens: Some(max_tokens),
            ..DecodeParams::default()
        },
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

/// A scheduler whose serving prefill chunk width is `chunk` tokens (small,
/// so a short test prompt still needs several chunks).
fn sched_with_chunk(chunk: u32, compute: Arc<dyn Compute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            serving_chunk_tokens: chunk,
            ..SchedulerConfig::default()
        },
        compute,
    )
}

#[test]
fn a_long_prompt_is_split_into_chunk_wide_jobs() {
    // A 10-token prompt with a 4-token chunk width needs 3 chunks
    // (4 + 4 + 2): no single call may ever carry more than the configured
    // width.
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute.clone());
    let id = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();

    while !sched.is_idle() {
        sched.advance();
    }

    let calls = compute.prefill_calls();
    let jobs_for_id: Vec<&PrefillJob> = calls
        .iter()
        .flat_map(|batch| batch.iter())
        .filter(|j| j.request == id)
        .collect();
    assert_eq!(
        jobs_for_id.len(),
        3,
        "a 10-token prompt at a 4-token chunk width takes 3 chunks, not 1 big call"
    );
    assert!(
        jobs_for_id.iter().all(|j| j.tokens.len() <= 4),
        "no job may exceed the configured serving chunk width"
    );
    assert_eq!(jobs_for_id[0].start_position, 0);
    assert_eq!(jobs_for_id[1].start_position, 4);
    assert_eq!(jobs_for_id[2].start_position, 8);
    assert_eq!(
        jobs_for_id[2].tokens.len(),
        2,
        "the last chunk carries the remainder"
    );
}

#[test]
fn advance_emits_a_prefill_chunk_event_per_chunk_with_cumulative_progress() {
    // P3-06: the request log's per-phase fields (chunks consumed, prefilled
    // tokens) are counted from `SchedEvent::PrefillChunk`, since `Request`
    // itself carries no chunk history, only its current progress. A
    // 10-token prompt at a 4-token chunk width must emit exactly 3 such
    // events, `prefilled_tokens` climbing 4, 8, 10 — never restated as each
    // chunk's own (non-cumulative) width.
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute);
    let id = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();

    let mut chunk_events: Vec<(u32, u32)> = Vec::new();
    while !sched.is_idle() {
        for event in sched.advance() {
            if let SchedEvent::PrefillChunk {
                request,
                chunk_tokens,
                prefilled_tokens,
                ..
            } = event
                && request == id
            {
                chunk_events.push((chunk_tokens, prefilled_tokens));
            }
        }
    }
    assert_eq!(
        chunk_events,
        vec![(4, 4), (4, 8), (2, 10)],
        "one PrefillChunk event per chunk, prefilled_tokens cumulative"
    );
}

#[test]
fn only_the_final_prefill_chunk_receives_stochastic_sampling_params() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute.clone());
    let sampling = DecodeParams {
        max_tokens: Some(1),
        temperature: 0.8,
        top_p: 0.7,
        top_k: 7,
        presence_penalty: 0.4,
        frequency_penalty: -0.4,
        seed: 9,
        ignore_eos: false,
        thinking_budget: None,
        starts_in_reasoning: false,
    };
    sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                reuse_boundaries: Vec::new(),
                model: "qwen3.8-27b".into(),
                tokens: (1..=10).collect(),
                params: sampling,
                constrained: None,
                forced_literal: None,
                warm_up: false,
            },
            RequestClass::Agent,
        )
        .unwrap();

    while !sched.is_idle() {
        sched.advance();
    }

    let calls = compute.prefill_calls();
    let params: Vec<DecodeParams> = calls
        .iter()
        .flat_map(|batch| batch.iter().map(|job| job.params))
        .collect();
    assert_eq!(params.len(), 3);
    assert_eq!(
        params[0],
        DecodeParams {
            max_tokens: sampling.max_tokens,
            ..DecodeParams::default()
        }
    );
    assert_eq!(params[1], params[0]);
    assert_eq!(params[2], sampling);
}

#[test]
fn a_decode_round_accompanies_every_chunk_while_a_lane_is_decode_ready() {
    // The anti-serialization property (P3-01's non-negotiable, stated
    // independently of K): while a prefill is active and a decode-ready
    // lane exists, the recorded call sequence never places two prefill
    // chunks between two decode rounds. A short filler occupies a running
    // lane for the whole test; a long prompt needs several chunks to
    // finish. Every tick that advances the long prompt's prefill must also
    // decode the filler in the same tick.
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute.clone());

    let filler = sched.submit(input(&[1], 20), RequestClass::Agent).unwrap();
    let long = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();

    let mut prev_prefill = 0usize;
    let mut prev_decode = 0usize;
    let mut saw_a_chunk = false;
    while sched.request_state(long) != Some(RequestState::Done) {
        sched.advance();
        let prefill_calls = compute.prefill_calls().len();
        let decode_calls = compute.decode_calls().len();
        if prefill_calls > prev_prefill
            && sched.request_state(filler) == Some(RequestState::Running)
        {
            saw_a_chunk = true;
            assert_eq!(
                decode_calls,
                prev_decode + 1,
                "a prefill chunk was sent this tick while the filler lane was decode-ready, \
                 but no decode round accompanied it"
            );
        }
        prev_prefill = prefill_calls;
        prev_decode = decode_calls;
    }
    assert!(
        saw_a_chunk,
        "the long prompt must have needed at least one chunk"
    );
}

#[test]
fn exactly_one_request_holds_prefill_progress_at_a_time() {
    // Two long prompts submitted together: only the first may hold
    // multi-tick (device-resident) progress. The second stays untouched
    // (`Admitted`, zero progress) until the first finishes prefilling
    // entirely — "the rest queue".
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute.clone());

    let first = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 3), RequestClass::Agent)
        .unwrap();
    let second = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 3), RequestClass::Agent)
        .unwrap();

    // Tick 1: only `first` gets a chunk (it sorts first, FIFO).
    sched.advance();
    assert_eq!(sched.prefill_progress(first), Some(4));
    assert_eq!(
        sched.request_state(second),
        Some(RequestState::Admitted),
        "the second long prompt must not start until the first finishes"
    );
    assert_eq!(sched.prefill_progress(second), Some(0));

    // Tick 2: `first` continues; `second` still untouched.
    sched.advance();
    assert_eq!(sched.prefill_progress(first), Some(8));
    assert_eq!(sched.request_state(second), Some(RequestState::Admitted));
    assert_eq!(sched.prefill_progress(second), Some(0));

    // Tick 3: `first` finishes its last (2-token) chunk and is dealt a lane
    // (10 tokens complete, `Prefilling` -> `Running` in the same tick).
    // `second` is chosen as this tick's active holder only when none
    // exists at the *start* of the tick — `first` still held it here — so
    // `second` is untouched for one more tick.
    sched.advance();
    assert_eq!(sched.prefill_progress(first), Some(10));
    assert_eq!(sched.request_state(first), Some(RequestState::Running));
    assert_eq!(sched.request_state(second), Some(RequestState::Admitted));
    assert_eq!(sched.prefill_progress(second), Some(0));

    // Tick 4: no active holder remains — `second` is now free to start.
    sched.advance();
    assert_eq!(sched.request_state(second), Some(RequestState::Prefilling));
    assert_eq!(sched.prefill_progress(second), Some(4));

    // Both eventually complete.
    while !sched.is_idle() {
        sched.advance();
    }
    assert_eq!(sched.request_state(first), Some(RequestState::Done));
    assert_eq!(sched.request_state(second), Some(RequestState::Done));
}

/// A `Compute` that fails the `fail_on`-th (0-indexed) `prefill_step` call
/// once, then delegates to a real [`MockCompute`].
struct PrefillFailsOnce {
    inner: MockCompute,
    call: Mutex<u32>,
    fail_on: u32,
}

impl Compute for PrefillFailsOnce {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut n = self.call.lock().unwrap();
        let this_call = *n;
        *n += 1;
        if this_call == self.fail_on {
            return Err(ComputeError::Kernel(-3));
        }
        self.inner.prefill_step(jobs)
    }
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.inner.decode_step(jobs)
    }
}

#[test]
fn a_failed_chunk_never_resends_an_already_applied_span() {
    // Chunk 1 (call #0) succeeds; chunk 2 (call #1) fails. A failed batch
    // releases the request's sequence, and a chunk partway through the
    // prompt cannot rebuild it (GitHub #296), so the request ends there:
    // neither chunk 1's span nor chunk 2's ever reaches the backend again.
    let compute = Arc::new(PrefillFailsOnce {
        inner: MockCompute::new(),
        call: Mutex::new(0),
        fail_on: 1,
    });
    let mut sched = sched_with_chunk(4, compute.clone());
    let id = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();

    sched.advance(); // call #0: chunk 1 (0..4) succeeds
    assert_eq!(sched.prefill_progress(id), Some(4));
    let events = sched.advance(); // call #1: chunk 2 (4..8) fails
    assert!(
        events.iter().any(|e| matches!(
            e,
            SchedEvent::Done { request, reason: FinishReason::Error, .. } if *request == id
        )),
        "the request ends on the failed chunk: {events:?}"
    );
    assert!(sched.is_idle(), "nothing is left to retry");

    let jobs_for_id: Vec<PrefillJob> = compute
        .inner
        .prefill_calls()
        .iter()
        .flat_map(|b| b.iter().cloned())
        .filter(|j| j.request == id)
        .collect();
    // Only chunk 1 ever landed (the failed attempt never got past the
    // wrapper, so `MockCompute` itself only saw the succeeding call).
    assert_eq!(jobs_for_id.len(), 1);
    assert_eq!(jobs_for_id[0].start_position, 0);
    assert_eq!(*compute.call.lock().unwrap(), 2, "no retry was attempted");
}

#[test]
fn a_failed_first_chunk_is_retried_with_exactly_its_span() {
    // Chunk 1 (call #0) fails. A first job builds its sequence, so it is
    // retried (call #1) with the same 0..4 span, and the prompt goes on.
    let compute = Arc::new(PrefillFailsOnce {
        inner: MockCompute::new(),
        call: Mutex::new(0),
        fail_on: 0,
    });
    let mut sched = sched_with_chunk(4, compute.clone());
    let id = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();

    sched.advance(); // call #0: chunk 1 (0..4) fails
    assert_eq!(sched.prefill_progress(id), Some(0), "a failed chunk must not advance progress");
    sched.advance(); // call #1: the retry
    assert_eq!(sched.prefill_progress(id), Some(4), "the retry applied chunk 1");

    let jobs_for_id: Vec<PrefillJob> = compute
        .inner
        .prefill_calls()
        .iter()
        .flat_map(|b| b.iter().cloned())
        .filter(|j| j.request == id)
        .collect();
    assert_eq!(jobs_for_id[0].start_position, 0);
    assert_eq!(jobs_for_id[0].tokens, vec![1, 2, 3, 4], "the retry carries exactly chunk 1's span");
}

#[test]
fn cancel_mid_prefill_aborts_and_releases_without_finishing() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute.clone());
    let id = sched
        .submit(input(&(1..=10).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();

    sched.advance(); // chunk 1 of 3 lands; the request is mid-prefill
    assert_eq!(sched.request_state(id), Some(RequestState::Prefilling));
    assert_eq!(sched.prefill_progress(id), Some(4));

    assert!(
        sched.cancel(id),
        "cancel must succeed on an in-flight request"
    );
    let calls_before = compute.prefill_calls().len();

    sched.advance(); // the abort takes effect: no further chunk is sent
    assert_eq!(
        sched.request_state(id),
        Some(RequestState::Done),
        "cancel is abort, not suspend: the request is done, not parked"
    );
    assert_eq!(
        compute.prefill_calls().len(),
        calls_before,
        "a cancelled request is never sent another chunk"
    );
    assert!(sched.is_idle());
    assert_eq!(
        sched.kv_used_pages(),
        0,
        "the cancelled request's reservation is released"
    );

    // Cancelling again (already Done) reports nothing to cancel.
    assert!(!sched.cancel(id));
}

#[test]
fn gdn_position_keeps_advancing_through_decode_after_a_chunked_prefill() {
    // Regression: prefill's chunk checkpoints now land at real, large
    // absolute prompt positions (P3-01), and `GdnState::checkpoint` only
    // ever moves forward (`position >= self.position`). Decode's own
    // checkpoint call must therefore keep counting from where prefill left
    // off — not restart from 0 — or every decode-time checkpoint after a
    // completed prefill is silently dropped and the GDN position freezes.
    let compute = Arc::new(MockCompute::new());
    let mut sched = sched_with_chunk(4, compute);
    let prompt: Vec<u32> = (1..=10).collect();
    let prompt_len = prompt.len();
    let id = sched
        .submit(input(&prompt, 5), RequestClass::Agent)
        .unwrap();

    // Drive prefill to completion (3 chunks at width 4) and the lane deal.
    // The tick that deals the lane also runs this tick's decode round in
    // the same `advance()`, so by the time the request is `Running` its
    // GDN position already reflects the prompt plus one generated token.
    while sched.request_state(id) != Some(RequestState::Running) {
        sched.advance();
    }
    let after_first_decode = sched
        .gdn_position(id)
        .expect("the request is known to the scheduler");
    assert!(
        after_first_decode > prompt_len,
        "the position must have moved past the prefill boundary ({prompt_len}) after decoding, \
         got {after_first_decode}"
    );

    // Each further decode tick must still move the GDN position forward —
    // not freeze because the checkpoint call restarted counting from 0
    // (which `GdnState::checkpoint`'s `position >= self.position` guard
    // would then silently reject forever, since prefill left it far ahead
    // of any small decode-token count).
    let mut previous = after_first_decode;
    for _ in 0..3 {
        sched.advance();
        let now = sched
            .gdn_position(id)
            .expect("still known to the scheduler");
        assert_eq!(
            now,
            previous + 1,
            "each decode tick must advance the position by one"
        );
        previous = now;
    }
}

#[test]
fn resolve_serving_chunk_tokens_allows_narrowing_and_refuses_widening() {
    assert_eq!(resolve_serving_chunk_tokens(512, 1024), Ok(512));
    assert_eq!(resolve_serving_chunk_tokens(1024, 1024), Ok(1024));
    let err = resolve_serving_chunk_tokens(2048, 1024).unwrap_err();
    assert!(
        err.contains("1024"),
        "the refusal must name the load width: {err}"
    );
}

/// A `Compute` whose calls take time on a hand-driven clock -- a chunk
/// `prefill_ms`, a round `decode_ms` -- and that records the call order: `P`
/// for a prefill call, `D` for a decode round.
struct TimedCompute {
    inner: MockCompute,
    elapsed_ms: Arc<AtomicU64>,
    prefill_ms: u64,
    decode_ms: u64,
    calls: Mutex<String>,
}

impl Compute for TimedCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.elapsed_ms.fetch_add(self.prefill_ms, Ordering::SeqCst);
        self.calls.lock().unwrap().push('P');
        self.inner.prefill_step(jobs)
    }
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.elapsed_ms.fetch_add(self.decode_ms, Ordering::SeqCst);
        self.calls.lock().unwrap().push('D');
        self.inner.decode_step(jobs)
    }
}

/// A scheduler at a 4-token chunk width and `decode_share`, over a backend
/// whose chunk takes 1000 ms and whose round takes 10 ms of its own clock.
fn timed_sched(decode_share: f64) -> (ConcreteScheduler, Arc<TimedCompute>) {
    let elapsed_ms = Arc::new(AtomicU64::new(0));
    let compute = Arc::new(TimedCompute {
        inner: MockCompute::new(),
        elapsed_ms: elapsed_ms.clone(),
        prefill_ms: 1000,
        decode_ms: 10,
        calls: Mutex::new(String::new()),
    });
    let base = Instant::now();
    let clock: Clock = Arc::new(move || base + Duration::from_millis(elapsed_ms.load(Ordering::SeqCst)));
    let sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-27b".into(),
            serving_chunk_tokens: 4,
            decode_share,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    )
    .with_clock(clock);
    (sched, compute)
}

#[test]
fn a_decode_share_holds_the_next_chunk_until_the_decoding_lanes_have_had_their_share() {
    // GitHub #306: a Flash-Next chunk runs for seconds, so one decode round
    // per chunk starves the other lanes. At a share of one half, every
    // 1000 ms chunk is followed by 1000 ms of 10 ms rounds before the next
    // chunk: 100 rounds, the one in the chunk's own advance included.
    let (mut sched, compute) = timed_sched(0.5);
    let filler = sched.submit(input(&[1], 1000), RequestClass::Agent).unwrap();
    sched.advance();
    assert_eq!(sched.request_state(filler), Some(RequestState::Running));
    let long = sched
        .submit(input(&(1..=16).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();
    while sched.request_state(long) != Some(RequestState::Done) {
        sched.advance();
    }

    let calls = compute.calls.lock().unwrap().clone();
    let between: Vec<usize> = calls.split('P').map(str::len).collect();
    // The filler's chunk, then the long prompt's four. The filler's prompt
    // ends in that one chunk, so nothing holds after it; the long prompt's
    // first three chunks are each followed by a hold.
    assert_eq!(between.len(), 6, "{calls}");
    assert_eq!(&between[1..5], &[1, 100, 100, 100], "rounds between chunks: {calls}");
}

#[test]
fn a_decode_share_does_not_hold_a_newcomer_behind_a_prompts_last_chunk() {
    // GitHub #306: the hold exists to give the decoding lanes time between
    // chunks of one prompt. After a prompt's LAST chunk the next request is
    // a newcomer (an agent swarm's turn boundary), and making it wait for
    // lanes that already had a round gains nothing.
    let (mut sched, compute) = timed_sched(0.5);
    sched.submit(input(&[1], 1000), RequestClass::Agent).unwrap();
    sched.advance();
    let long = sched
        .submit(input(&(1..=8).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();
    let newcomer = sched.submit(input(&[1, 2, 3, 4], 1), RequestClass::Agent).unwrap();
    while sched.request_state(newcomer) != Some(RequestState::Done) {
        sched.advance();
    }
    assert_eq!(sched.request_state(long), Some(RequestState::Done));

    let calls = compute.calls.lock().unwrap().clone();
    let between: Vec<usize> = calls.split('P').map(str::len).collect();
    // filler | long 1 | long 2 (its last) | newcomer: the only hold is the
    // one between the long prompt's two chunks.
    assert_eq!(&between[1..4], &[1, 100, 1], "rounds between chunks: {calls}");
}

#[test]
fn a_decode_share_holds_nothing_while_no_lane_decodes() {
    // With nobody decoding there is no one to give time to: the chunks run
    // back to back, as they do without a share.
    let (mut sched, compute) = timed_sched(0.5);
    let long = sched
        .submit(input(&(1..=16).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();
    let mut advances = 0;
    while sched.request_state(long) != Some(RequestState::Done) {
        sched.advance();
        advances += 1;
    }
    assert_eq!(advances, 4, "one chunk per advance");
    assert!(compute.calls.lock().unwrap().starts_with("PPPP"));
}

#[test]
fn without_a_decode_share_a_chunk_still_gets_exactly_one_round() {
    // The default (ADR 0018's K = 1) is unchanged: one round per chunk.
    let (mut sched, compute) = timed_sched(0.0);
    sched.submit(input(&[1], 1000), RequestClass::Agent).unwrap();
    sched.advance();
    let long = sched
        .submit(input(&(1..=16).collect::<Vec<_>>(), 1), RequestClass::Agent)
        .unwrap();
    while sched.request_state(long) != Some(RequestState::Done) {
        sched.advance();
    }
    let calls = compute.calls.lock().unwrap().clone();
    assert!(calls.starts_with("PDPDPDPDPD"), "{calls}");
}

#[test]
#[should_panic(expected = "decode share")]
fn a_decode_share_of_one_is_refused() {
    // A share of 1 would hold every chunk for as long as any lane decodes.
    timed_sched(1.0);
}
