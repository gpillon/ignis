//! core-04 — a failed compute step is not swallowed: the request stays
//! retryable (never dealt a lane with an unwarmed KV) and the fault is
//! surfaced through `ConcreteScheduler::last_error` (the "surface, don't
//! swallow" house rule).

use std::sync::{Arc, Mutex};

use ignis_core::scheduler::{Compute, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome};
use ignis_core::types::{ComputeError, FinishReason, RequestId, SchedEvent};
use ignis_core::{ConcreteScheduler, MAX_PREFILL_ATTEMPTS, MockCompute, Scheduler};

/// A compute that fails its first `prefill_step` with a kernel fault, then
/// behaves like the deterministic mock (so a single scheduler can be driven
/// through failure → recovery).
struct FailingCompute {
    inner: MockCompute,
    failed_once: Mutex<bool>,
    faults: Mutex<u32>,
}

impl FailingCompute {
    fn new() -> Self {
        Self {
            inner: MockCompute::new(),
            failed_once: Mutex::new(false),
            faults: Mutex::new(0),
        }
    }

    fn faults(&self) -> u32 {
        *self.faults.lock().unwrap()
    }
}

impl Compute for FailingCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut first = self.failed_once.lock().unwrap();
        if *first {
            drop(first);
            return self.inner.prefill_step(jobs);
        }
        *first = true;
        *self.faults.lock().unwrap() += 1;
        Err(ComputeError::Kernel(-7))
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.inner.decode_step(jobs)
    }
}

#[test]
fn failed_prefill_leaves_the_request_retryable() {
    let compute = Arc::new(FailingCompute::new());
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
    let id = sched
        .submit(
            ignis_core::types::RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2],
                params: Default::default(),
                program: None,
            },
            ignis_core::types::RequestClass::Agent,
        )
        .unwrap();

    // Advance #1: prefill faults. No events are emitted (the request must
    // not be dealt a lane with an unwarmed KV), and the fault is surfaced.
    let ev = sched.advance();
    assert!(ev.is_empty(), "a failed step emits no events");
    assert_eq!(
        sched.last_error(),
        Some(&ComputeError::Kernel(-7)),
        "the fault must be surfaced, not swallowed"
    );
    assert!(!sched.is_idle());

    // Advance #2: prefill succeeds; the same request is prefilled, dealt a
    // lane, and decoded — the fault did not strand it.
    let ev = sched.advance();
    let admitted = ev.iter().any(|e| {
        matches!(
            e,
            SchedEvent::Admitted { request, .. } if *request == id
        )
    });
    assert!(
        admitted,
        "the retried request must be seated after recovery"
    );
    assert!(ev.iter().any(|e| matches!(
        e,
        SchedEvent::Token { request, .. } if *request == id
    )));
    // The fault is cleared once a step succeeds.
    assert!(sched.last_error().is_none());
    // Exactly one prefill call faulted (the retry went to the mock).
    assert_eq!(compute.faults(), 1);
}

/// A compute whose `prefill_step` always faults — the leaf refusing a
/// sequence alloc it will never grant (GitHub #166).
struct AlwaysFailingPrefill {
    faults: Mutex<u32>,
}

impl Compute for AlwaysFailingPrefill {
    fn prefill_step(&self, _jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        *self.faults.lock().unwrap() += 1;
        Err(ComputeError::Kernel(-1))
    }

    fn decode_step(&self, _jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        Ok(Vec::new())
    }
}

/// GitHub #166 — a prefill that keeps failing is retried a bounded number
/// of times, then its request ends with `FinishReason::Error` and releases
/// everything, instead of being retried on every advance forever.
#[test]
fn a_prefill_that_keeps_failing_ends_its_request_with_an_error() {
    let compute = Arc::new(AlwaysFailingPrefill {
        faults: Mutex::new(0),
    });
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
    let id = sched
        .submit(
            ignis_core::types::RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2],
                params: Default::default(),
                program: None,
            },
            ignis_core::types::RequestClass::Agent,
        )
        .unwrap();

    // Every attempt but the last is the ordinary retryable fault.
    for attempt in 1..MAX_PREFILL_ATTEMPTS {
        let ev = sched.advance();
        assert!(ev.is_empty(), "attempt {attempt} is still retryable");
        assert!(!sched.is_idle());
    }
    // The last attempt ends the request.
    let ev = sched.advance();
    assert!(
        ev.iter().any(|e| matches!(
            e,
            SchedEvent::Done { request, tokens: 0, reason: FinishReason::Error, .. } if *request == id
        )),
        "the request ends with an error: {ev:?}"
    );
    assert_eq!(sched.last_error(), Some(&ComputeError::Kernel(-1)));
    assert!(sched.is_idle(), "nothing is left to retry");
    assert_eq!(sched.kv_used_pages(), 0, "its reservation is released");

    // No further attempt reaches the backend.
    sched.advance();
    assert_eq!(*compute.faults.lock().unwrap(), MAX_PREFILL_ATTEMPTS);
}

/// A compute that always faults on `decode_step`.
struct DecodeFaultCompute {
    inner: MockCompute,
    faults: Mutex<u32>,
}

impl Compute for DecodeFaultCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, _jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        *self.faults.lock().unwrap() += 1;
        Err(ComputeError::Kernel(-9))
    }
}

#[test]
fn failed_decode_ends_the_request_and_releases_its_lane() {
    let compute = Arc::new(DecodeFaultCompute {
        inner: MockCompute::new(),
        faults: Mutex::new(0),
    });
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
    let id: RequestId = sched
        .submit(
            ignis_core::types::RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2],
                params: Default::default(),
                program: None,
            },
            ignis_core::types::RequestClass::Agent,
        )
        .unwrap();
    let sibling: RequestId = sched
        .submit(
            ignis_core::types::RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "qwen3.8-27b".into(),
                tokens: vec![3, 4],
                params: Default::default(),
                program: None,
            },
            ignis_core::types::RequestClass::Agent,
        )
        .unwrap();

    // Advance #1: prefill + lane deal succeed.
    let ev = sched.advance();
    assert!(
        ev
            .iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == id))
    );
    assert_eq!(sched.last_error(), Some(&ComputeError::Kernel(-9)));
    // The request ends with Error; it is neither retried nor left holding a
    // decode lane.
    assert!(
        ev.iter().any(|e| matches!(
            e,
            SchedEvent::Done { request, reason: FinishReason::Error, .. } if *request == id
        )),
        "the failed decode ends its request: {ev:?}"
    );
    assert!(
        ev.iter().any(|e| matches!(
            e,
            SchedEvent::Done { request, reason: FinishReason::Error, .. } if *request == sibling
        )),
        "the failed decode ends every request in its batch: {ev:?}"
    );
    assert!(sched.is_idle(), "the failed decode releases its lane");
    assert_eq!(sched.kv_used_pages(), 0, "its reservation is released");

    sched.advance();
    assert_eq!(*compute.faults.lock().unwrap(), 1);
}
