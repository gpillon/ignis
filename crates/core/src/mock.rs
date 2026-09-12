//! The deterministic [`Compute`] mock — the kernel-leaf stand-in for CPU
//! tests (ADR 0006: a mock stands in for the kernel leaf, which keeps the
//! whole scheduler CPU-testable without a GPU).
//!
//! The mock is *deterministic by construction*: the token a request
//! generates at step `i` is a pure function of (mock seed, request id,
//! request seed, `i`) — no RNG, no clocks. That mirrors the production
//! floor (greedy + fixed seed, ADR 0007) and lets tests pin exact event
//! streams. It also *records* the batch shape of every call, so tests can
//! assert the scheduler's batching behavior (batched prefill groups N
//! requests into one call, not N calls) without a GPU.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use crate::scheduler::{Compute, DecodeJob, DecodeOutcome, PrefillJob};
use crate::types::{ComputeError, FinishReason, RequestId, TokenId};

/// Recording handle onto the mock's call history (shared through the
/// `Arc<dyn Compute>` the scheduler holds, so tests can assert the batch
/// shape after driving the scheduler).
#[derive(Default)]
struct Inner {
    /// Max tokens per request, learned from the prefill jobs' params.
    limits: HashMap<RequestId, Option<u32>>,
    /// Per-request generation seeds, learned from the prefill jobs' params.
    seeds: HashMap<RequestId, u64>,
    /// Explicit stop points (`stop_after`), overriding the learned limit.
    stops: HashMap<RequestId, u32>,
    /// Tokens generated so far, per request.
    generated: HashMap<RequestId, u32>,
    /// Every prefill batch the mock received (batch shape for assertions).
    prefill_batches: Vec<Vec<PrefillJob>>,
    /// Every decode batch the mock received (batch shape for assertions).
    decode_batches: Vec<Vec<DecodeJob>>,
    /// Shared prefixes the scheduler told the backend to let go of (P4-10,
    /// GitHub #126), in order: the publishing request of each. The real
    /// adapter drops its leaf handle here, so a test that never sees the
    /// call is looking at a prefix the engine would have pinned forever.
    prefixes_released: Vec<RequestId>,
}

/// A deterministic, recording [`Compute`] implementation for tests.
///
/// Token generation: the token a request emits at decode step `i` mixes the
/// mock's seed, the request id, the request's (learned) seed, and `i` — a
/// fixed, side-effect-free function, so identical runs produce identical
/// streams and different request seeds produce different streams.
pub struct MockCompute {
    seed: u64,
    inner: Mutex<Inner>,
}

impl MockCompute {
    /// A mock with the default (zero) seed.
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// A mock whose token streams are mixed with `seed` (test variation
    /// knob; the request's own seed always takes part in the mix too).
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// The prefill batches the mock received, in order (each entry is one
    /// `prefill_step` call; its jobs are the batched prefill group).
    pub fn prefill_calls(&self) -> Vec<Vec<PrefillJob>> {
        self.inner.lock().unwrap().prefill_batches.clone()
    }

    /// The decode batches the mock received, in order.
    pub fn decode_calls(&self) -> Vec<Vec<DecodeJob>> {
        self.inner.lock().unwrap().decode_batches.clone()
    }

    /// The publishers whose shared prefix the scheduler released (P4-10,
    /// GitHub #126), in order.
    pub fn released_prefixes(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().prefixes_released.clone()
    }

    /// Force `request` to stop after `n` generated tokens, regardless of
    /// its learned `max_tokens` (for driving streams of requests submitted
    /// without a token cap).
    pub fn stop_after(&self, request: RequestId, n: u32) {
        self.inner.lock().unwrap().stops.insert(request, n);
    }

    /// The token this mock would emit for `request` at decode step `step`
    /// (pure function of the seeds — exposed so tests can pin a stream
    /// without running the scheduler).
    pub fn token_for(&self, request: RequestId, step: u32) -> TokenId {
        Self::mix(self.seed, request, 0, step)
    }
}

impl Default for MockCompute {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for MockCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        let mut g = self.inner.lock().unwrap();
        for job in jobs {
            // Learn the request's limits / seed from its params.
            g.limits.insert(job.request, job.params.max_tokens);
            g.seeds.insert(job.request, job.params.seed);
        }
        g.prefill_batches.push(jobs.to_vec());
        Ok(())
    }

    fn release_prefix(&self, publisher: RequestId) {
        self.inner.lock().unwrap().prefixes_released.push(publisher);
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let mut g = self.inner.lock().unwrap();
        g.decode_batches.push(jobs.to_vec());
        Ok(jobs
            .iter()
            .map(|job| {
                let step = *g.generated.entry(job.request).or_insert(0);
                // An explicit stop_after overrides the learned limit; an
                // unlearned request (no prefill seen) is treated as
                // unbounded.
                let limit = g
                    .stops
                    .get(&job.request)
                    .copied()
                    .or_else(|| g.limits.get(&job.request).copied().flatten());
                match limit {
                    // The mock has no real EOS token — its stop condition
                    // is always a token-count cap, so it always finishes
                    // with `Length` (never `Stop`).
                    Some(n) if step >= n => DecodeOutcome::Finished(FinishReason::Length),
                    _ => {
                        let seed = g.seeds.get(&job.request).copied().unwrap_or(0);
                        let token = Self::mix(self.seed, job.request, seed, step);
                        *g.generated.get_mut(&job.request).unwrap() += 1;
                        DecodeOutcome::Token(token)
                    }
                }
            })
            .collect())
    }

    // core-06 (P4-07, GitHub #125): the mock holds no real GPU sequence, so
    // it has no real snapshot to move — but a host-tier eviction scenario
    // still needs *some* nonzero, deterministic byte cost per request for
    // the byte-budget bookkeeping to be exercisable at all (a `Compute`
    // that always reports 0, the trait's own default, would make a tier of
    // any size always "fit," and no eviction test could ever force a
    // discard). One nominal byte per snapshot, uniform across every
    // request, is exactly what the existing page-based scenarios already
    // assumed before this ticket's byte-budget rewrite: a fixed per-entry
    // cost that scales purely with entry *count*.
    fn snapshot_size(&self, _request: RequestId) -> Result<u64, ComputeError> {
        Ok(1)
    }

    fn evict(&self, _request: RequestId) -> Result<u64, ComputeError> {
        Ok(1)
    }
}

impl MockCompute {
    /// The deterministic token mix: a pure function of (mock seed, request
    /// id, request seed, step).
    fn mix(seed: u64, request: RequestId, request_seed: u64, step: u32) -> TokenId {
        let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        h ^= request.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= request_seed.wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h % (u32::MAX as u64)) as u32
    }
}

/// A test-only `Compute` decorator (GitHub #69) that lets a test hold one
/// `decode_step` call open deterministically — a stand-in for the real
/// backend's per-step GPU latency, without a single `sleep()` anywhere
/// (ADR 0006). Wraps any `Compute` (in practice, always a [`MockCompute`]).
pub struct GatedCompute {
    inner: Arc<dyn Compute>,
    armed: AtomicBool,
    entered_tx: SyncSender<()>,
    release_rx: Mutex<Receiver<()>>,
}

/// The test's handle onto a [`GatedCompute`]'s gate.
pub struct GateController {
    entered_rx: Receiver<()>,
    release_tx: SyncSender<()>,
}

impl GateController {
    /// Blocks the calling thread until the gated `decode_step` call has
    /// entered the gate — proof that whatever is driving `Compute` (in
    /// production, the model thread) is now stuck inside this call and
    /// cannot do anything else until [`GateController::release`] is called.
    pub fn wait_entered(&self) {
        self.entered_rx
            .recv()
            .expect("the armed decode_step must enter the gate before the compute is dropped");
    }

    /// Releases the held `decode_step` call, letting it complete.
    pub fn release(&self) {
        self.release_tx
            .send(())
            .expect("the armed decode_step must still be waiting to be released");
    }
}

impl GatedCompute {
    /// Wrap `inner`; the gate starts disarmed (`decode_step` passes through
    /// untouched until [`GatedCompute::arm`] is called).
    pub fn new(inner: Arc<dyn Compute>) -> (Arc<Self>, GateController) {
        let (entered_tx, entered_rx) = sync_channel(0);
        let (release_tx, release_rx) = sync_channel(0);
        (
            Arc::new(Self {
                inner,
                armed: AtomicBool::new(false),
                entered_tx,
                release_rx: Mutex::new(release_rx),
            }),
            GateController {
                entered_rx,
                release_tx,
            },
        )
    }

    /// Arms the gate: the next `decode_step` call blocks until
    /// [`GateController::release`] is called. Consumed on entry (one arm =
    /// one held call) — a test re-arms explicitly for each hold it wants.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl Compute for GatedCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        if self.armed.swap(false, Ordering::SeqCst) {
            // A rendezvous pair: `send` blocks until `wait_entered`'s `recv`
            // is there to receive it, then this call blocks again until
            // `release` sends the go-ahead.
            let _ = self.entered_tx.send(());
            let _ = self.release_rx.lock().unwrap().recv();
        }
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.inner.release(request);
    }

    fn snapshot_size(&self, request: RequestId) -> Result<u64, ComputeError> {
        self.inner.snapshot_size(request)
    }

    fn evict(&self, request: RequestId) -> Result<u64, ComputeError> {
        self.inner.evict(request)
    }

    fn restore(&self, request: RequestId, context_tokens: u32) -> Result<(), ComputeError> {
        self.inner.restore(request, context_tokens)
    }

    fn discard_snapshot(&self, request: RequestId) {
        self.inner.discard_snapshot(request);
    }
}
