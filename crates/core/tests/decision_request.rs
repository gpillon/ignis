//! GitHub #238 (ADR 0034) — a request kind that **ends where prefill ends**.
//!
//! A decision never takes a decode lane, never enters `Running`, generates
//! nothing, and finishes with the **readout** of its answer tokens' logits
//! at its prompt's last position. What these tests hold down is the whole
//! shape of that: that it is answered, that it is answered *once*, that the
//! decode path never sees it, and — the one that would silently produce
//! well-formed nonsense — that an exact repeat still runs a forward pass.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute`
//! behind the `Compute` seam, no GPU anywhere.

use std::sync::{Arc, Mutex};

use ignis_core::decision::Readout;
use ignis_core::scheduler::{Compute, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome};
use ignis_core::types::{
    ComputeError, DecodeParams, RequestClass, RequestId, RequestInput, RequestState, SchedEvent,
    SubmitError, TokenId,
};
use ignis_core::{ConcreteScheduler, FinishReason, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
/// The answer tokens of a three-option decision.
const ANSWERS: [TokenId; 3] = [32, 33, 34];

/// A decision: the answer tokens it reads out, over a prompt whose opener is
/// reported at its very end.
///
/// That opener is the **worst case**, not the usual one: the 27B's template
/// appends a closed think block after the opener with thinking off, so a
/// real decision prompt runs a few tokens past it
/// (`crates/server/tests/decide_prompt_tail.rs`). Pinning the extreme here
/// is deliberate — an opener at the prompt's end is the shape that walks
/// into the empty-last-chunk trap, and a template that produced one must not
/// break the engine.
fn decision(prompt: Vec<TokenId>, max_tokens: Option<u32>) -> RequestInput {
    let opener = prompt.len() as u32;
    RequestInput {
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens,
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: Some(opener),
        user_turn_tokens: None,
        system_block_tokens: None,
        decision: Some(Arc::from(ANSWERS.to_vec())),
        program: None,
    }
}

/// [`decision`], whose evidence is its system-and-tools block: the head a
/// fan-out of questions shares, and the only head a decision can share at
/// all — a decision finishes the tick its prefill does, so it is never a
/// *live* publisher for a sibling to claim from. A **retained** prefix
/// outlives it; a plain one does not.
fn decision_over_evidence(prompt: Vec<TokenId>, evidence: u32) -> RequestInput {
    RequestInput {
        system_block_tokens: Some(evidence),
        ..decision(prompt, None)
    }
}

/// An ordinary generating request over the same prompt, for the comparisons
/// that only mean something side by side.
fn generating(prompt: Vec<TokenId>, max_tokens: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        ..decision(prompt, max_tokens)
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

/// The `Done` event for `request`, as (tokens generated, reason, readout).
fn finish(
    events: &[SchedEvent],
    request: RequestId,
) -> (u32, FinishReason, Option<Readout>) {
    events
        .iter()
        .find_map(|e| match e {
            SchedEvent::Done {
                request: r,
                tokens,
                reason,
                readout,
                ..
            } if *r == request => Some((*tokens, *reason, readout.clone())),
            _ => None,
        })
        .unwrap_or_else(|| panic!("request {request} never finished: {events:?}"))
}

// ── acceptance 5: the lane tag a decision defaults to ────────────────────

#[test]
fn a_decision_that_states_no_lane_tag_is_an_agent() {
    assert_eq!(RequestClass::for_decision(None), RequestClass::Agent);
}

#[test]
fn a_decision_that_states_a_lane_tag_is_believed() {
    assert_eq!(
        RequestClass::for_decision(Some(RequestClass::Interactive)),
        RequestClass::Interactive,
        "a caller whose one decision really is interactive says so"
    );
    assert_eq!(
        RequestClass::for_decision(Some(RequestClass::Agent)),
        RequestClass::Agent
    );
}

#[test]
fn a_lane_tag_nobody_has_defined_still_means_interactive_on_a_decision() {
    // The unrecognized-value rule is the *tag's*, not the route's: a tag
    // that was stated but not understood is already `Interactive` by the
    // time it reaches here. Only a tag that is absent altogether takes the
    // decision's own default, which is what the two cases below separate.
    assert_eq!(
        RequestClass::for_decision(Some(RequestClass::from_extension("urgent"))),
        RequestClass::Interactive,
        "stated but not understood keeps the tag's own safe default"
    );
    assert_eq!(
        RequestClass::for_decision(None),
        RequestClass::Agent,
        "not stated at all takes the decision's"
    );
}

// ── acceptance 4: the reservation is the prompt alone ────────────────────

#[test]
fn a_decision_is_admitted_however_large_its_max_tokens() {
    let mut sched = ConcreteScheduler::with_config(config(), Arc::new(MockCompute::new()));
    // Prompt + `max_tokens` is far past `max_sequence_tokens` (8192), and
    // an ordinary request carrying these numbers is refused.
    let huge = Some(1_000_000);
    sched
        .submit(decision(tokens(1, 132), huge), RequestClass::Agent)
        .expect("a decision reserves its prompt and nothing else");
    assert!(
        matches!(
            sched.submit(generating(tokens(1, 132), huge), RequestClass::Agent),
            Err(SubmitError::ContextExceeded { .. })
        ),
        "the same numbers on a generating request are refused, which is what \
         makes the decision's admission mean something"
    );
}

#[test]
fn a_decision_whose_prompt_alone_overruns_the_context_is_refused() {
    let mut sched = ConcreteScheduler::with_config(config(), Arc::new(MockCompute::new()));
    let over = sched.submit(decision(tokens(1, 9000), None), RequestClass::Agent);
    assert!(
        matches!(over, Err(SubmitError::ContextExceeded { .. })),
        "a prompt past the per-sequence limit can never be allocated: {over:?}"
    );
}

#[test]
fn a_decision_reserves_its_prompt_not_its_generation_budget() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let id = sched
        .submit(decision(tokens(1, 40), Some(4096)), RequestClass::Agent)
        .expect("admitted");
    sched.advance();
    let job = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .find(|job| job.request == id)
        .expect("a prefill job");
    assert_eq!(
        job.context_tokens, 40,
        "the whole-sequence reservation handed to the leaf is the prompt"
    );
}

// ── acceptance 1 + 2: answered once, and never decoded ───────────────────

/// A `Compute` that fails the test the moment a decision reaches the decode
/// path. Everything else is the mock's.
struct NeverDecodes {
    inner: Arc<MockCompute>,
    forbidden: Mutex<Vec<RequestId>>,
}

impl NeverDecodes {
    fn new(inner: Arc<MockCompute>) -> Self {
        Self {
            inner,
            forbidden: Mutex::new(Vec::new()),
        }
    }

    fn forbid(&self, request: RequestId) {
        self.forbidden.lock().unwrap().push(request);
    }
}

impl Compute for NeverDecodes {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let forbidden = self.forbidden.lock().unwrap();
        for job in jobs {
            assert!(
                !forbidden.contains(&job.request),
                "request {} reached a decode round; a decision must never take a lane",
                job.request
            );
        }
        drop(forbidden);
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.inner.release(request);
    }

    fn release_prefix(&self, publisher: RequestId, tokens: u32) {
        self.inner.release_prefix(publisher, tokens);
    }
}

#[test]
fn a_decision_is_answered_at_the_end_of_prefill_and_never_decoded() {
    let mock = Arc::new(MockCompute::new());
    let compute = Arc::new(NeverDecodes::new(mock.clone()));
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());
    let id = sched
        .submit(decision(tokens(1, 40), None), RequestClass::Agent)
        .expect("admitted");
    compute.forbid(id);
    // A generating sibling, so decode rounds really do happen this run and
    // the assertion above is not vacuous.
    let sibling = sched
        .submit(generating(tokens(500, 40), Some(3)), RequestClass::Interactive)
        .expect("admitted");

    let events = run_to_idle(&mut sched);

    let (generated, reason, readout) = finish(&events, id);
    assert_eq!(generated, 0, "a decision generates nothing, and says so");
    assert_eq!(reason, FinishReason::Stop, "it was answered, not cut short");
    let readout = readout.expect("the finish event carries the readout");
    assert_eq!(
        readout.logits.len(),
        ANSWERS.len(),
        "one logit per answer token"
    );
    let mass = readout.answer_mass();
    assert!((0.0..=1.0).contains(&mass), "answer mass is a probability: {mass}");

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == id)),
        "a decision emits no token"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == id)),
        "and is never dealt a decode lane"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == sibling)),
        "its generating sibling did decode, so the decode path really ran"
    );

    let released: Vec<RequestId> = mock
        .released_requests()
        .into_iter()
        .filter(|&r| r == id)
        .collect();
    assert_eq!(
        released,
        vec![id],
        "`Compute::release` is called exactly once for it, like any completed request"
    );
}

#[test]
fn a_decision_never_enters_running() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute);
    let id = sched
        .submit(decision(tokens(1, 40), None), RequestClass::Agent)
        .expect("admitted");
    let mut seen = Vec::new();
    while !sched.is_idle() {
        sched.advance();
        if let Some(state) = sched.request_state(id) {
            seen.push(state);
        }
    }
    assert!(
        !seen.contains(&RequestState::Running),
        "a decision passes Admitted -> Prefilling -> Done, never Running: {seen:?}"
    );
    assert_eq!(seen.last(), Some(&RequestState::Done));
}

#[test]
fn a_decision_reads_out_on_its_last_chunk_and_on_no_other() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            serving_chunk_tokens: 16,
            ..config()
        },
        compute.clone(),
    );
    let id = sched
        .submit(decision(tokens(1, 40), None), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);

    let asked: Vec<bool> = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .filter(|job| job.request == id)
        .map(|job| job.readout.is_some())
        .collect();
    assert!(asked.len() > 1, "the prompt was chunked: {asked:?}");
    assert_eq!(
        asked.iter().filter(|&&asked| asked).count(),
        1,
        "exactly one chunk reads out: {asked:?}"
    );
    assert_eq!(
        asked.last(),
        Some(&true),
        "and it is the last one — the prompt's final position holds the decision"
    );
}

// ── acceptance 3: an exact repeat still runs a forward pass ──────────────

#[test]
fn an_exact_repeat_of_a_decision_still_prefills_a_token() {
    // The empty-last-chunk trap. A decision's prompt ends at its generation
    // opener, so without the reuse trim the repeat would claim state
    // covering every token it has, prefill nothing, run no forward pass and
    // produce no logits — a decision answered by nothing at all.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());

    sched
        .submit(decision(tokens(1, 40), None), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);

    let second = sched
        .submit(decision(tokens(1, 40), None), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);

    let jobs: Vec<PrefillJob> = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .filter(|job| job.request == second)
        .collect();
    let last = jobs.last().expect("the repeat was prefilled at all");
    assert!(
        !last.tokens.is_empty(),
        "the repeat's final chunk carries at least one token to run the model over"
    );
    assert!(
        last.readout.is_some(),
        "and that chunk is the one that reads out"
    );

    let (_, _, readout) = finish(&events, second);
    assert!(
        readout.is_some(),
        "so the repeat is answered, exactly as the first was"
    );
}

#[test]
fn a_decision_retains_nothing_for_a_later_request_to_claim() {
    // Its prompt ends at the opener, so a checkpoint captured there would
    // cover the whole prompt — and no decision may ever claim that far.
    // Retaining it would be state nothing can use.
    //
    // Nothing in the decision path enforces this: `Request::checkpoint_point`
    // already refuses an opener that leaves no prompt token after it, which
    // a decision's opener by definition does not. The test is here because
    // that is an invariant two features hold by accident of arithmetic, and
    // this is the side of it that would break silently.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute);
    sched
        .submit(decision(tokens(1, 64), None), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        0,
        "a decision captures no prompt checkpoint"
    );
}

#[test]
fn decisions_over_one_evidence_share_its_prefix() {
    // The trim removes the *last* token from a decision's reuse, not its
    // reuse. This is the shape the endpoint is built around — one evidence,
    // many questions — and it only works if a decision publishes a prefix a
    // sibling can claim. The shared head is skipped; only the tail carrying
    // each question, and the one token holding the answer, is re-run.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());

    let prompt = tokens(1, 96);
    let first = sched
        .submit(decision_over_evidence(prompt.clone(), 80), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);
    let sibling = sched
        .submit(decision_over_evidence(prompt, 80), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);

    let prefilled = |request: RequestId| -> u32 {
        compute
            .prefill_calls()
            .into_iter()
            .flatten()
            .filter(|job| job.request == request)
            .map(|job| job.tokens.len() as u32)
            .sum()
    };
    let (head, tail) = (prefilled(first), prefilled(sibling));
    assert_eq!(head, 96, "the first decision prefilled the whole evidence");
    assert!(
        tail >= 1,
        "its sibling always runs the model at least once: {tail}"
    );
    assert!(
        tail < head,
        "and skips the shared head it is entitled to: {tail} of {head}"
    );
}

#[test]
fn an_exact_repeat_over_a_retained_evidence_prefix_still_prefills_a_token() {
    // The trap with teeth. The test above cannot catch it: a 40-token
    // decision with no system block retires its prefix the moment it
    // finishes, so its repeat claims nothing whatever the code does, and
    // every trim in the engine could be deleted with it still green.
    //
    // This is the shape that discriminates — a **retained** evidence prefix
    // (so it outlives its publisher) over a prompt whose length is an exact
    // multiple of the KV page (so the shareable head lands *on* the prompt
    // rather than below it). Without the trim the repeat claims all 64
    // tokens, prefills none, runs no forward pass and is answered by
    // nothing; with it the head is published a page lower and the repeat
    // re-runs the page that carries its answer.
    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(config(), compute.clone());

    let prompt = tokens(1, 64);
    sched
        .submit(decision_over_evidence(prompt.clone(), 64), RequestClass::Agent)
        .expect("admitted");
    run_to_idle(&mut sched);
    let repeat = sched
        .submit(decision_over_evidence(prompt, 64), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);

    let widths: Vec<usize> = compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .filter(|job| job.request == repeat)
        .map(|job| job.tokens.len())
        .collect();
    assert!(
        widths.last().is_some_and(|&last| last > 0),
        "the repeat's final chunk runs the model: {widths:?}"
    );
    let (_, reason, readout) = finish(&events, repeat);
    assert!(readout.is_some(), "so it is answered");
    assert_eq!(reason, FinishReason::Stop);
}

#[test]
fn a_decision_that_cannot_be_read_out_ends_in_error_not_in_silence() {
    // The one thing the termination must never do is report `Stop` with an
    // empty answer: that is a decision answered by nothing, dressed as a
    // decision answered. Unreachable through the real seam — the backend
    // refuses such a job outright (`READOUT_WITHOUT_TOKENS`) — so the only
    // way to see the guard is to build a backend that lies.
    struct SwallowsTheReadout(Arc<MockCompute>);

    impl Compute for SwallowsTheReadout {
        fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
            let mut outcomes = self.0.prefill_step(jobs)?;
            for outcome in &mut outcomes {
                outcome.readout = None;
            }
            Ok(outcomes)
        }

        fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
            self.0.decode_step(jobs)
        }

        fn release(&self, request: RequestId) {
            self.0.release(request);
        }
    }

    let compute = Arc::new(SwallowsTheReadout(Arc::new(MockCompute::new())));
    let mut sched = ConcreteScheduler::with_config(config(), compute);
    let id = sched
        .submit(decision(tokens(1, 40), None), RequestClass::Agent)
        .expect("admitted");
    let events = run_to_idle(&mut sched);

    let (generated, reason, readout) = finish(&events, id);
    assert_eq!(
        reason,
        FinishReason::Error,
        "a decision with no answer could not be served, and ends saying so"
    );
    assert!(readout.is_none());
    assert_eq!(generated, 0);
}
