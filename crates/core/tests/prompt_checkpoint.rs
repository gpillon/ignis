//! GitHub #186 (ADR 0029) — the prompt checkpoint on the device: the tracer
//! bullet of cross-request state reuse.
//!
//! Every request's prefill is cut at its **generation opener**, the point
//! where the rendered prompt hands over to the model, and the whole-sequence
//! state there is retained after the request finishes. A later request whose
//! prompt *extends* it — the next turn of a chat, the next iteration of an
//! agent's tool loop — resumes there instead of re-prefilling the whole
//! conversation.
//!
//! What these tests can prove, and what they cannot. The claim "reused state
//! is the *right* state" needs the model to run, and lives in
//! `prompt_checkpoint_gpu.rs`. What lives here is everything the next layer
//! up observes without a card: which prompt tokens are prefilled at all,
//! which are skipped and from where, what the retained pool costs, and — the
//! load-bearing one — that none of it ever makes a live request wait.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` that
//! records the prefill call shape behind the `Compute` seam, exactly as
//! `prefix_reuse.rs` drives sibling reuse.

use std::sync::Arc;

use ignis_core::checkpoint::ReuseSource;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
/// The default scheduler's KV page, in tokens.
const PAGE: u32 = 16;

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request whose rendered prompt's generation opener ends `opener` tokens
/// in (`None` for a frontend that reported none).
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
    }
}

/// **Turn N** of a conversation: 40 prompt tokens whose opener ends at 37 —
/// the last two or three tokens are the `<think>\n` the opener is deliberately
/// placed *before* (ADR 0029), so the prompt really does continue past it.
fn turn_n() -> RequestInput {
    input(tokens(1, 40), Some(37), 4)
}

/// **Turn N+1**: turn N's prompt up to its opener, then the assistant's reply,
/// the user's next message and a new opener of its own — the shape every
/// later turn of a chat (and every iteration of a tool loop) actually has.
fn turn_n_plus_1() -> RequestInput {
    input([tokens(1, 37), tokens(500, 23)].concat(), Some(57), 4)
}

fn scheduler(compute: Arc<MockCompute>, config: SchedulerConfig) -> ConcreteScheduler {
    ConcreteScheduler::with_config(config, compute)
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        ..SchedulerConfig::default()
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// The `StateReused` events for `request`, as (source, skipped tokens,
/// restore micros).
fn reuses(events: &[SchedEvent], request: RequestId) -> Vec<(ReuseSource, u32, u64)> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::StateReused {
                request: r,
                source,
                tokens,
                restore_micros,
            } if *r == request => Some((*source, *tokens, *restore_micros)),
            _ => None,
        })
        .collect()
}

/// Every prefill chunk width dealt to `request`, in order.
fn chunk_widths(compute: &MockCompute, request: RequestId) -> Vec<usize> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .map(|j| j.tokens.len())
        .collect()
}

// ── The reuse itself ────────────────────────────────────────────────────

#[test]
fn turn_n_plus_1_reuses_turn_n_prompt_checkpoint() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());

    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "turn N left a checkpoint at its opener"
    );

    // Turn N+1 arrives after turn N has finished: no sibling is alive to
    // share anything with, which is exactly the case sibling prefix reuse
    // cannot serve.
    let n1 = sched.submit(turn_n_plus_1(), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(
        reuses(&events, n1),
        vec![(ReuseSource::Device, 37, 1)],
        "turn N+1 resumed from turn N's checkpoint, on the device"
    );
    // It prefilled its own 23 new tokens and nothing else: no chunk of it
    // ever carried a token turn N had already warmed.
    let widths = chunk_widths(&compute, n1);
    assert_eq!(widths.iter().sum::<usize>(), 23, "only the new tokens");
    let first = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .find(|j| j.request == n1)
        .expect("turn N+1 was prefilled");
    assert_eq!(first.start_position, 37, "it starts at turn N's opener");
    assert_eq!(
        first.checkpoint.map(|c| (c.publisher, c.tokens)),
        Some((n, 37)),
        "its first job claims turn N's checkpoint by name"
    );
    assert!(
        first.shared_prefix.is_none(),
        "a checkpoint claim subsumes a sibling-prefix claim, never doubles it"
    );
    assert_eq!(sched.checkpoint_pool().reused_tok(), 37);
}

#[test]
fn a_claim_does_not_consume_the_checkpoint() {
    // Regenerate, retry and two forks from one history all hit (ADR 0029).
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);

    let mut claimants = Vec::new();
    for tail in [500u32, 700, 900] {
        claimants.push(
            sched
                .submit(
                    input([tokens(1, 37), tokens(tail, 23)].concat(), Some(57), 4),
                    RequestClass::Interactive,
                )
                .unwrap(),
        );
        let events = run_to_idle(&mut sched);
        assert_eq!(
            reuses(&events, *claimants.last().unwrap()),
            vec![(ReuseSource::Device, 37, 1)],
            "fork from {tail} hits the same checkpoint"
        );
    }
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the entry survived every claimant"
    );
    assert_eq!(sched.checkpoint_pool().reused_tok(), 3 * 37);
}

#[test]
fn a_prompt_that_diverges_before_the_opener_reuses_nothing() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);

    // A different conversation that happens to open the same way. Its prompt
    // parts from turn N's at token 20, well before the opener.
    let other = sched
        .submit(
            input([tokens(1, 20), tokens(800, 40)].concat(), Some(57), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert!(reuses(&events, other).is_empty(), "no retained state matched");
    assert_eq!(
        chunk_widths(&compute, other).iter().sum::<usize>(),
        60,
        "it prefilled its whole prompt"
    );
}

// ── Where the prefill is cut ────────────────────────────────────────────

#[test]
fn the_prefill_is_cut_at_the_publish_point_and_again_at_the_opener() {
    // The state a claimant receives is the state *at the opener*, so the
    // chunk that lands on it has to stop there — the same reason a shared
    // prefix is published at a chunk boundary (ADR 0024). The opener is at
    // most one page past the publish point, so the second cut costs one short
    // chunk and only on the tick that takes the checkpoint.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        chunk_widths(&compute, n),
        vec![32, 5, 3],
        "cut at the 2-page publish point, then at the 37-token opener"
    );
    let capture: Vec<Option<u32>> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == n)
        .map(|j| j.capture_checkpoint_tokens)
        .collect();
    assert_eq!(
        capture,
        vec![None, Some(37), None],
        "exactly the chunk that ends on the opener asks for the capture"
    );
}

#[test]
fn a_page_aligned_opener_is_captured_on_the_publish_chunk() {
    // One prompt in sixty-four has its opener land exactly on a page
    // boundary. There is no second chunk to cut, so the capture rides the
    // publish chunk: the backend publishes the prefix and captures against
    // it in one call. Without this the aligned prompts would be the only
    // ones that silently never retain anything.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched
        .submit(input(tokens(1, 40), Some(2 * PAGE), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, n),
        vec![32, 8],
        "no extra cut: the opener already is the publish point"
    );
    let jobs: Vec<(Option<u32>, Option<u32>)> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == n)
        .map(|j| (j.publish_prefix_tokens, j.capture_checkpoint_tokens))
        .collect();
    assert_eq!(
        jobs,
        vec![(Some(32), Some(32)), (None, None)],
        "one chunk publishes and captures"
    );
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);

    // And it is claimable like any other.
    let later = sched
        .submit(
            input([tokens(1, 32), tokens(600, 28)].concat(), Some(57), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, later), vec![(ReuseSource::Device, 32, 1)]);
}

#[test]
fn the_chunk_that_captures_is_always_greedy() {
    // Why "the penalty-count row at the opener is zero" is a design property
    // and not a hope. A prefill chunk that ends the prompt is dealt the
    // request's *real* sampling parameters — it samples, and sampling updates
    // the sequence's penalty-count row. Every other chunk is dealt greedy
    // parameters, whose leaf branch has no sampling side effect at all. So
    // the capture is only ever asked for from an intermediate chunk, and the
    // row it captures has never been written.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let stochastic = RequestInput {
        params: DecodeParams {
            max_tokens: Some(4),
            temperature: 0.9,
            presence_penalty: 0.7,
            frequency_penalty: 0.5,
            ..DecodeParams::default()
        },
        ..turn_n()
    };
    let n = sched.submit(stochastic, RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "it did capture");

    let jobs: Vec<(Option<u32>, f32, f32)> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == n)
        .map(|j| {
            (
                j.capture_checkpoint_tokens,
                j.params.temperature,
                j.params.presence_penalty,
            )
        })
        .collect();
    let capturing = jobs
        .iter()
        .find(|(at, _, _)| at.is_some())
        .expect("one chunk captures");
    assert_eq!(
        (capturing.1, capturing.2),
        (0.0, 0.0),
        "the capturing chunk is greedy, so nothing has been sampled at the opener"
    );
    // And the request really was stochastic: the chunk that ends its prompt
    // carries what it asked for, which is the chunk a capture must never be.
    let final_chunk = jobs.last().expect("a last chunk");
    assert_eq!(final_chunk.0, None, "the last chunk never captures");
    assert_eq!(
        (final_chunk.1, final_chunk.2),
        (0.9, 0.7),
        "and it is the one that samples"
    );
}

#[test]
fn an_opener_at_the_very_end_of_the_prompt_captures_nothing() {
    // Such a prompt hands over to the model with nothing after the opener, so
    // the chunk that would capture is also the chunk that samples. Refused
    // rather than captured from — and the next turn loses nothing it could
    // have had, since a checkpoint at the prompt's end is the point ADR 0029
    // rejected in the first place.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched
        .submit(input(tokens(1, 40), Some(40), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
    assert_eq!(
        chunk_widths(&compute, n),
        vec![32, 8],
        "and its prefill is not cut for a capture that cannot happen"
    );
}

#[test]
fn a_prompt_with_no_reported_opener_captures_nothing() {
    // The frontend could not place the opener — no `<|im_start|>assistant\n`,
    // or a byte offset that does not tokenize to an exact prefix. No
    // checkpoint is taken, rather than one taken at a point the tokenizer
    // disagrees about.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched
        .submit(input(tokens(1, 40), None, 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
    assert_eq!(
        chunk_widths(&compute, n),
        vec![32, 8],
        "only the shared-prefix publish cut remains"
    );
}

#[test]
fn a_prompt_shorter_than_one_page_captures_nothing() {
    // There is no whole page under the opener to hang the checkpoint's
    // history on, so nothing is retained — and the request pays nothing.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched
        .submit(input(tokens(1, 12), Some(9), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
    assert_eq!(chunk_widths(&compute, n), vec![12], "one uncut chunk");
}

#[test]
fn a_checkpoint_claimant_does_not_capture_its_own() {
    // The slice boundary, pinned so it is a decision rather than a surprise:
    // turn N+1 resumed at turn N's opener, so its *own* first KV page is not
    // the one its own opener falls in, and the leaf could not capture there.
    // Superseding turn N's checkpoint with turn N+1's is #187's lineage work;
    // here the chain stops at turn 2.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let n1 = sched.submit(turn_n_plus_1(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "turn N+1 left no checkpoint of its own"
    );
    assert!(
        compute
            .prefill_calls()
            .iter()
            .flatten()
            .filter(|j| j.request == n1)
            .all(|j| j.capture_checkpoint_tokens.is_none()),
        "and never asked the backend for one"
    );
}

#[test]
fn the_cached_prefix_covers_exactly_the_head_the_backend_published() {
    // The scheduler's cache entry and the leaf's prefix are two ledgers over
    // one set of pages, and they were the same number until the publish
    // point was floored to the *opener's* page rather than the prompt's. If
    // the cache registered the prompt's head instead, a sibling would skip
    // prefill for tokens nothing ever warmed — and would answer from a hole,
    // silently.
    //
    // A 40-token prompt whose opener ends at 30: two whole pages of prompt,
    // but only one whole page below the opener.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let first = sched
        .submit(input(tokens(1, 40), Some(30), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        chunk_widths(&compute, first),
        vec![16, 14, 10],
        "cut at the opener's page, then at the opener"
    );

    // The cache entry under the checkpoint covers one page — the head the
    // backend published — and not the prompt's own two. Its pages are what
    // the pool is charged and what a claimant's reservation shrinks by, so an
    // entry claiming a page the leaf never published would hand a claimant
    // warm history that does not exist.
    assert_eq!(
        sched.checkpoint_pool().retained_pages(),
        1,
        "the retained entry holds exactly the published page"
    );
    assert_eq!(
        sched.kv_used_pages(),
        1,
        "and the pool is charged for exactly that page once every live request is gone"
    );

    // A later request sharing the whole 40-token head resumes at the
    // checkpoint's 30-token opener, which reaches further than the 16-token
    // prefix under it: longest reuse wins (ADR 0029).
    let later = sched
        .submit(
            input([tokens(1, 30), tokens(700, 30)].concat(), None, 4),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, later), vec![(ReuseSource::Device, 30, 1)]);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::PrefixReused { request, .. } if *request == later)),
        "and it took the checkpoint rather than the shorter prefix under it"
    );
    assert_eq!(
        chunk_widths(&compute, later).iter().sum::<usize>(),
        30,
        "so it prefills only what the checkpoint does not cover"
    );
}

#[test]
fn turn_3_still_matches_turn_1_when_turn_2_left_no_checkpoint() {
    // Deferring lineage to #187 is only honest if the fallback works. Turn 2
    // resumed from turn 1's opener and prefilled past it, so it leaves no
    // checkpoint of its own — and the same is true of every concurrent
    // sibling in a subagent burst. What must not happen is the conversation
    // falling off the cache: turn 3 still starts with turn 1's head, so it
    // still reuses turn 1's entry. The saving degrades to "everything up to
    // turn 1's opener" rather than vanishing.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let n1 = sched.submit(turn_n_plus_1(), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, n1), vec![(ReuseSource::Device, 37, 1)]);
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "turn 2 left none of its own"
    );

    // Turn 3: turn 2's prompt, plus turn 2's answer and a third question.
    let turn_3 = input(
        [tokens(1, 37), tokens(500, 23), tokens(900, 20)].concat(),
        Some(77),
        4,
    );
    let n2 = sched.submit(turn_3, RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(
        reuses(&events, n2),
        vec![(ReuseSource::Device, 37, 1)],
        "turn 3 still resumes from turn 1's checkpoint"
    );
    assert_eq!(
        chunk_widths(&compute, n2).iter().sum::<usize>(),
        80 - 37,
        "so it prefills only what that entry does not cover"
    );
}

#[test]
fn a_prompt_that_ends_at_the_checkpoint_prefills_nothing_at_all() {
    // The retry/regenerate shape (spec story 3) taken to its limit: a prompt
    // that *is* the retained head. The claim alone puts the sequence where
    // its prompt ends — position, pending token and all — so there is no span
    // left to warm, and the job that builds the sequence carries no tokens.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);

    let exact = sched
        .submit(input(tokens(1, 37), None, 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, exact), vec![(ReuseSource::Device, 37, 1)]);
    let jobs: Vec<usize> = chunk_widths(&compute, exact);
    assert_eq!(jobs, vec![0], "one job, carrying no prompt tokens at all");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == exact)),
        "and the request still completes"
    );
}

// ── The byte budget ─────────────────────────────────────────────────────

#[test]
fn a_full_pool_skips_the_capture_and_evicts_nothing() {
    // `MockCompute` prices a checkpoint image at one nominal byte, so a
    // one-byte budget holds exactly one.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            retained_pool_bytes: 1,
            ..config()
        },
    );
    let first = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
    assert_eq!(sched.checkpoint_pool().used_bytes(), 1);

    // A second, unrelated conversation reaches its own opener with the pool
    // already full.
    let second = sched
        .submit(
            input([tokens(900, 37), tokens(950, 3)].concat(), Some(37), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the capture was skipped, not made room for"
    );
    assert_eq!(
        sched.checkpoint_pool().counters().discards,
        0,
        "nothing was evicted to take a checkpoint"
    );
    assert!(
        compute.released_checkpoints().is_empty(),
        "and no retained image was released"
    );
    assert!(
        compute
            .prefill_calls()
            .iter()
            .flatten()
            .filter(|j| j.request == second)
            .all(|j| j.capture_checkpoint_tokens.is_none()),
        "a capture the budget cannot hold is never even asked for"
    );
    assert_eq!(
        chunk_widths(&compute, second),
        vec![32, 8],
        "so its prefill is not cut at its opener either — a full pool is free"
    );
    // The entry that is there is still the first conversation's.
    let later = sched
        .submit(turn_n_plus_1(), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, later), vec![(ReuseSource::Device, 37, 1)]);
    let _ = first;
}

#[test]
fn a_backend_that_declines_the_capture_retains_nothing() {
    // A capture is a bet the backend may refuse — no room in its own image
    // pool, a sequence the leaf will not capture. The chunk lands normally,
    // the request is none the wiser, and the ledger records no entry, so the
    // scheduler and the device never disagree about what exists.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    compute.refuse_capture(n);
    let events = run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Done { request, .. } if *request == n)),
        "the request completed normally"
    );
}

#[test]
fn prompt_reuse_off_captures_and_reuses_nothing() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            prompt_reuse: false,
            ..config()
        },
    );
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0, "nothing retained");
    assert_eq!(
        chunk_widths(&compute, n),
        vec![32, 8],
        "and the prefill is not cut at the opener: off costs nothing"
    );

    let n1 = sched.submit(turn_n_plus_1(), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(reuses(&events, n1).is_empty(), "nothing reused");
    assert_eq!(
        chunk_widths(&compute, n1).iter().sum::<usize>(),
        60,
        "a cold bench measures a cold engine"
    );
}

// ── Cancellation ────────────────────────────────────────────────────────

#[test]
fn cancel_after_the_capture_keeps_the_checkpoint() {
    // "As a client whose request was cancelled mid-decode, I want the prompt
    // checkpoint kept, so that the client's retry hits" (spec, story 8).
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    sched.advance(); // the publish chunk
    sched.advance(); // the chunk that ends on the opener: captured
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "captured");

    assert!(sched.cancel(n));
    run_to_idle(&mut sched);
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the cancelled request's checkpoint stays"
    );
    // And the client's retry hits it.
    let retry = sched.submit(turn_n_plus_1(), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuses(&events, retry), vec![(ReuseSource::Device, 37, 1)]);
}

#[test]
fn cancel_before_the_capture_leaves_none() {
    // No partial checkpoints: a request cancelled before it reached its
    // opener leaves nothing behind, however far its prefill had got.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    sched.advance(); // the publish chunk only — the opener is not reached yet
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);

    assert!(sched.cancel(n));
    run_to_idle(&mut sched);
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        0,
        "nothing was retained"
    );
    let later = sched.submit(turn_n_plus_1(), RequestClass::Interactive).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(reuses(&events, later).is_empty(), "and nothing to reuse");
}

// ── Retained state never costs a live request anything ──────────────────

/// A pool tight enough that one large request needs all of it: 8 pages of 16
/// tokens, and a 128-token sequence envelope.
fn tight_pool(prompt_reuse: bool) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_sequence_tokens: 128,
        kv_capacity_pages: 8,
        kv_page_tokens: PAGE,
        prompt_reuse,
        ..SchedulerConfig::default()
    }
}

/// Run turn N to completion, then the `hungry` request that needs the whole
/// KV pool, and report every event the second one's admission produced.
fn retained_then_hungry(prompt_reuse: bool) -> (Vec<SchedEvent>, Arc<MockCompute>, u32) {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), tight_pool(prompt_reuse));
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    let retained = sched.checkpoint_pool().retained_pages();

    // 120 prompt + 8 generated = 128 tokens = the whole 8-page pool. It
    // reports no opener of its own, so its chunk decomposition is identical
    // in both runs and the only thing that differs is whether the pool it has
    // to fit into is holding something retained. (What *taking* a checkpoint
    // costs — one short extra chunk — is a different claim, pinned by
    // `the_prefill_is_cut_at_the_publish_point_and_again_at_the_opener`.)
    let hungry = sched
        .submit(input(tokens(2000, 120), None, 8), RequestClass::Interactive)
        .unwrap();
    let mut events = Vec::new();
    let mut ticks_to_admission = None;
    let mut tick = 0;
    while !sched.is_idle() {
        let step = sched.advance();
        if ticks_to_admission.is_none()
            && step
                .iter()
                .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == hungry))
        {
            ticks_to_admission = Some(tick);
        }
        events.extend(step);
        tick += 1;
    }
    (
        events,
        compute,
        ticks_to_admission.expect("the hungry request was admitted") * 10 + retained,
    )
}

#[test]
fn a_retained_entry_never_causes_an_admission_refusal_or_wait() {
    // Retained state is the *first victim* on the device (ADR 0023 as amended
    // by 0029): a live request that needs the pages takes them back before
    // admission considers freezing a protection or evicting anybody.
    //
    // The proof is a comparison, not an absolute: the same request, against
    // the same pool, with and without something retained in it. It must be
    // admitted on the same tick either way, and the run that had retained
    // state must not have had to protect, evict or re-queue anything to get
    // there.
    let (with_retained, compute, packed) = retained_then_hungry(true);
    let (without, _, packed_cold) = retained_then_hungry(false);

    assert!(packed % 10 > 0, "turn N really did retain pages");
    assert_eq!(packed_cold % 10, 0, "the control retained nothing");
    assert_eq!(
        packed / 10,
        packed_cold / 10,
        "the hungry request is admitted on the same tick either way"
    );
    for event in &with_retained {
        assert!(
            !matches!(
                event,
                SchedEvent::Protected { .. }
                    | SchedEvent::Evicted { .. }
                    | SchedEvent::Requeued { .. }
            ),
            "retained state made a live request wait: {event:?}"
        );
    }
    assert!(
        !without.iter().any(|e| matches!(
            e,
            SchedEvent::Protected { .. } | SchedEvent::Evicted { .. }
        )),
        "the control itself must be an uncontended baseline"
    );
    assert_eq!(
        compute.released_checkpoints().len(),
        1,
        "the retained entry was given up — and it is the backend's image that went"
    );
}

#[test]
fn a_prefix_a_live_request_stands_on_is_not_given_up_for_nothing() {
    // The steady state this whole feature exists for: turn N's prefix is held
    // by its retained entry *and* by the live turn N+1 standing on it. Giving
    // the entry up there returns not one page — the prefix still has a holder
    // — so a first-victim loop that did not know it would discard every
    // checkpoint in the pool and still not fit. The pool must come through
    // the pressure intact, and the request must go to the eviction machinery
    // instead, which is what that machinery is for.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), tight_pool(true));
    sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);

    // Turn N+1 claims it and is still generating when the pressure arrives.
    let n1 = sched
        .submit(
            input([tokens(1, 37), tokens(500, 23)].concat(), None, 20),
            RequestClass::Interactive,
        )
        .unwrap();
    while sched.request_state(n1) != Some(ignis_core::types::RequestState::Running) {
        sched.advance();
    }
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the claim did not consume the entry"
    );

    // A request that needs the whole pool arrives.
    sched
        .submit(input(tokens(2000, 120), None, 8), RequestClass::Interactive)
        .unwrap();
    for _ in 0..4 {
        sched.advance();
    }
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the pool was not emptied for pages that were never going to come back"
    );
    assert!(
        compute.released_checkpoints().is_empty(),
        "and no image was released"
    );

    // Once turn N+1 is gone the entry *is* reclaimable, and the first-victim
    // path takes it — the deferral above was about liveness, not a refusal.
    run_to_idle(&mut sched);
    assert_eq!(compute.released_checkpoints(), vec![0]);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0);
}

#[test]
fn a_reclaimed_checkpoint_returns_its_pages_to_the_pool() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), tight_pool(true));
    let n = sched.submit(turn_n(), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        sched.kv_used_pages(),
        sched.checkpoint_pool().retained_pages(),
        "once every live request is gone, the pool holds exactly the retained pages"
    );
    assert!(sched.kv_used_pages() > 0);

    // A request needing the whole pool, reporting no opener of its own so
    // that what is left retained afterwards is unambiguous.
    sched
        .submit(input(tokens(2000, 120), None, 8), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 0, "reclaimed");
    assert_eq!(sched.kv_used_pages(), 0, "and its pages came back");
    assert_eq!(compute.released_checkpoints(), vec![n]);
    assert_eq!(
        compute
            .released_prefixes()
            .iter()
            .filter(|&&p| p == n)
            .count(),
        1,
        "the shared pages under it were let go exactly once — the retained \
         entry was the last holder, not a second, uncounted one"
    );
}
