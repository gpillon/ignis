//! core-07 — sibling prefix caching: concurrent requests sharing a prompt
//! prefix skip the redundant prefill. A "1 main + N subagents" load shares
//! one big system-prompt prefix; the main's prefill registers it, and each
//! subagent **claims** the cached prefix — its prefill carries only the
//! tail (the shared head is already warm in the pool), and the
//! `sibling_prefix_reused_tok` counter accumulates every skipped token.
//!
//! The shared entry's pages are charged to the pool **once** (for every
//! claimant), not `N` times, and the pool is fully freed once the last
//! claimant completes (no leak).
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute`
//! that records the prefill call shape behind the `Compute` seam.

use std::sync::{Arc, Mutex};

use ignis_core::scheduler::{Compute, DecodeJob, PrefillJob};
use ignis_core::types::{ComputeError, DecodeParams, RequestClass, RequestInput, SchedEvent, TokenId};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler};

/// A prompt of `n` distinct tokens starting at `start` (a deterministic,
/// distinct token stream per request).
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request with the given prompt and a `max` generation cap.
fn input(model: &str, prompt: Vec<u32>, max: u32) -> RequestInput {
    RequestInput {
        model: model.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
    }
}

fn scheduler(compute: Arc<MockCompute>) -> ConcreteScheduler {
    ConcreteScheduler::new("qwen3.8-27b", compute)
}

/// Drive the scheduler to idle, collecting every event.
fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

#[test]
fn a_sibling_skips_the_shared_prefix() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());

    // The main agent: a 32-token prompt (2 KV pages at the default 16-token
    // page). Its prefill registers the shared prefix.
    let main_id = sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    sched.advance(); // prefill + register the main's prompt, admit it

    // A subagent that shares the main's 32-token head + an 8-token tail
    // claims the cached prefix: its prefill carries only the 8-token tail
    // (the shared head is already warm in the pool).
    let sub_id = sched
        .submit(input("qwen3.8-27b", [tokens(1, 32), tokens(50, 8)].concat(), 8), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);

    // The main's prefill warmed the full 32-token prompt; the subagent's
    // prefill warmed only its 8-token tail (the shared head was skipped).
    let calls = compute.prefill_calls();
    assert_eq!(calls[0].len(), 1, "the main prefills alone in the first batch");
    assert_eq!(calls[0][0].tokens.len(), 32, "the main warms its full prompt");
    assert_eq!(calls[1].len(), 1, "the subagent prefills alone in the second batch");
    assert_eq!(
        calls[1][0].tokens.len(),
        8,
        "the subagent warms only its tail (the shared head is skipped)"
    );

    // One `PrefixReused` event (the subagent skipped 32 shared tokens), and
    // the counter accumulates exactly those 32 tokens.
    let reused: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::PrefixReused { request, tokens } if *request == sub_id => Some(*tokens),
            _ => None,
        })
        .collect();
    assert_eq!(reused, vec![32], "the subagent reuses the shared 32-token prefix");
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        32,
        "the counter counts the skipped shared tokens"
    );
    // The main (the registrant) did a full prefill — it is not a reuse.
    assert!(!events
        .iter()
        .any(|e| matches!(e, SchedEvent::PrefixReused { request, .. } if *request == main_id)));
}

#[test]
fn a_1_main_n_subagents_load_increments_the_counter() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());

    // The main agent (a 32-token system-prompt prefix).
    sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    sched.advance(); // prefill + register + admit the main

    // Three subagents, all sharing the main's 32-token head but each with a
    // distinct 8-token tail (a "1 main + 3 subagents" concurrent load).
    for tail in [100u32, 200, 300] {
        sched
            .submit(
                input(
                    "qwen3.8-27b",
                    [tokens(1, 32), tokens(tail, 8)].concat(),
                    8,
                ),
                RequestClass::Agent,
            )
            .unwrap();
    }
    let events = run_to_idle(&mut sched);

    // Each of the 3 subagents skipped the shared 32-token prefix: the
    // counter accumulates 3 × 32 = 96 (the main's full prefill is not a
    // reuse).
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        96,
        "the counter accumulates every skipped shared token (3 × 32)"
    );
    // Three `PrefixReused` events (one per subagent), each for 32 tokens.
    let reused: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::PrefixReused { tokens, .. } => Some(*tokens),
            _ => None,
        })
        .collect();
    assert_eq!(reused, vec![32, 32, 32], "one reuse event per subagent");

    // All three subagents' prefills carried only their 8-token tails (batched
    // into ONE call — batched prefill, core-04).
    let calls = compute.prefill_calls();
    assert_eq!(calls[1].len(), 3, "the 3 subagents prefill in one batched call");
    assert!(
        calls[1].iter().all(|job| job.tokens.len() == 8),
        "every subagent warms only its tail"
    );
}

#[test]
fn the_shared_prefix_is_charged_once_not_per_claimant() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());

    // The main registers a 32-token prefix (2 KV pages).
    sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    sched.advance(); // prefill + register + admit the main

    // Three subagents claim the same shared prefix (each skips 32 tokens).
    for tail in [100u32, 200, 300] {
        sched
            .submit(
                input(
                    "qwen3.8-27b",
                    [tokens(1, 32), tokens(tail, 8)].concat(),
                    8,
                ),
                RequestClass::Agent,
            )
            .unwrap();
        sched.advance(); // each subagent claims + prefills + is admitted
    }

    // While the main + 3 subagents are all in flight, the pool pins the
    // shared prefix **once** (2 pages), not 4 times (2 × 4 claimants).
    assert_eq!(
        sched.prefix_pinned_pages(),
        2,
        "the shared prefix's 2 pages are pinned once, for every claimant"
    );

    // Once the last claimant completes, the entry drops and its pages return
    // to the pool — no leak.
    run_to_idle(&mut sched);
    assert_eq!(
        sched.kv_used_pages(),
        0,
        "the pool is fully freed once the last claimant completes"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        0,
        "the entry is dropped when the last claimant releases"
    );
}

#[test]
fn no_reuse_when_there_is_no_shared_prefix() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());

    // Two requests with entirely different prompts: no shared prefix, so no
    // reuse (each does a full prefill) and the counter stays 0.
    let a = sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    sched.advance(); // prefill + register + admit the first
    let b = sched
        .submit(input("qwen3.8-27b", tokens(100, 32), 8), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        0,
        "no shared prefix → no skipped tokens"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::PrefixReused { .. })),
        "no `PrefixReused` event for unrelated prompts"
    );
    let calls = compute.prefill_calls();
    assert_eq!(
        calls[1][0].tokens.len(),
        32,
        "the second request warms its full (unshared) prompt"
    );
    let _ = (a, b);
}

#[test]
fn a_full_prompt_match_skips_the_entire_prefill() {
    // A request whose prompt is *exactly* a cached prefix (no tail) skips
    // the whole prefill — its prefill job carries an empty tail (it only
    // sets up the decode state).
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone());

    // The main registers a 32-token prefix.
    let main = sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    sched.advance(); // prefill + register + admit the main

    // A duplicate of the main's exact prompt claims the full prefix (32
    // tokens, an empty tail).
    let dup = sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);

    let calls = compute.prefill_calls();
    assert_eq!(
        calls[1][0].tokens.len(),
        0,
        "a full-prompt match warms nothing (empty tail)"
    );
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        32,
        "the duplicate reuses all 32 shared tokens"
    );
    let _ = (main, dup);
    let _ = events;
}

/// A compute that fails exactly the `fail_on`-th (0-indexed) prefill call
/// with a kernel fault, then behaves like the deterministic mock (so a
/// single scheduler can be driven through: the registrant's prefill
/// succeeds, a claimant's prefill faults, and the retry recovers).
struct PrefillFailsOn {
    inner: MockCompute,
    call: Mutex<u32>,
    fail_on: u32,
}

impl Compute for PrefillFailsOn {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        let n = *self.call.lock().unwrap();
        *self.call.lock().unwrap() += 1;
        if n == self.fail_on {
            return Err(ComputeError::Kernel(-3));
        }
        self.inner.prefill_step(jobs)
    }
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError> {
        self.inner.decode_step(jobs)
    }
}

#[test]
fn a_failed_prefill_retry_does_not_double_claim() {
    // The main registers a 32-token prefix; a subagent claims it but its
    // prefill (the 2nd prefill call) faults. The subagent stays Admitted,
    // still holding its claim. On the retry, the claim loop must NOT
    // re-claim it (a re-claim would double-count the entry's refcount and
    // the `sibling_prefix_reused_tok` counter, and pin the entry forever
    // — the release happens once, at completion).
    let compute = Arc::new(PrefillFailsOn {
        inner: MockCompute::new(),
        call: Mutex::new(0),
        fail_on: 1, // the claimant's prefill (the 2nd prefill call) faults
    });
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);

    // The main (a 32-token prefix) prefills (call #0, succeeds) and
    // registers the shared prefix.
    let main = sched
        .submit(input("qwen3.8-27b", tokens(1, 32), 8), RequestClass::Agent)
        .unwrap();
    sched.advance();

    // A subagent that shares the main's 32-token head + an 8-token tail
    // claims the cached prefix; its prefill (call #1) faults (it stays
    // Admitted, holding the claim).
    let sub = sched
        .submit(
            input("qwen3.8-27b", [tokens(1, 32), tokens(50, 8)].concat(), 8),
            RequestClass::Agent,
        )
        .unwrap();
    sched.advance(); // prefill call #1 faults: the claimant claims, then faults
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        32,
        "the claimant's single claim counts its 32 skipped tokens"
    );

    // The retry (call #2, succeeds): the claimant keeps its claim (no
    // double-claim), prefills its tail, and completes.
    run_to_idle(&mut sched);
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        32,
        "a failed-prefill retry does not double-claim (the counter stays 32)"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        0,
        "the entry drops when the last claimant releases (no pinned leak)"
    );
    let _ = (main, sub);
}