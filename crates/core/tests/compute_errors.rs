//! core-04 — a failed compute step is not swallowed: the request stays
//! retryable (never dealt a lane with an unwarmed KV) and the fault is
//! surfaced through `ConcreteScheduler::last_error` (the "surface, don't
//! swallow" house rule).

use std::sync::{Arc, Mutex};

use ignis_core::scheduler::{Compute, DecodeJob, PrefillJob};
use ignis_core::types::{ComputeError, RequestId, SchedEvent, TokenId};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler};

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
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        let mut first = self.failed_once.lock().unwrap();
        if *first {
            drop(first);
            return self.inner.prefill_step(jobs);
        }
        *first = true;
        *self.faults.lock().unwrap() += 1;
        Err(ComputeError::Kernel(-7))
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError> {
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
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2],
                params: Default::default(),
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

/// A compute that always faults on `decode_step`.
struct DecodeFaultCompute {
    inner: MockCompute,
}

impl Compute for DecodeFaultCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, _jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError> {
        Err(ComputeError::Kernel(-9))
    }
}

#[test]
fn failed_decode_keeps_the_request_running() {
    let compute = Arc::new(DecodeFaultCompute {
        inner: MockCompute::new(),
    });
    let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
    let id: RequestId = sched
        .submit(
            ignis_core::types::RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2],
                params: Default::default(),
            },
            ignis_core::types::RequestClass::Agent,
        )
        .unwrap();

    // Advance #1: prefill + lane deal succeed; decode faults.
    let ev = sched.advance();
    assert!(
        ev.iter()
            .any(|e| matches!(e, SchedEvent::Admitted { request, .. } if *request == id))
    );
    assert_eq!(
        sched.last_error(),
        Some(&ComputeError::Kernel(-9)),
        "the decode fault must be surfaced"
    );
    // The request still holds its lane (Running) — nothing is dropped or
    // re-queued, the step is simply retried.
    assert!(!sched.is_idle());
}
