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
use std::sync::Mutex;

use crate::scheduler::{Compute, DecodeJob, PrefillJob};
use crate::types::{ComputeError, RequestId, TokenId};

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

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError> {
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
                    Some(n) if step >= n => None, // finished: max_tokens / EOS reached
                    _ => {
                        let seed = g.seeds.get(&job.request).copied().unwrap_or(0);
                        let token = Self::mix(self.seed, job.request, seed, step);
                        *g.generated.get_mut(&job.request).unwrap() += 1;
                        Some(token)
                    }
                }
            })
            .collect())
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
