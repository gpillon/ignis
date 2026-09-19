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

use crate::decision::{Readout, log_sum_exp};
use crate::scheduler::{Compute, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome, NO_HOST_ROOM};
use crate::types::{ComputeError, FinishReason, RequestId, SpecCounters, TokenId};

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
    /// Run lengths to commit per decode round (P5-06, GitHub #154), cycled
    /// per request; empty is today's round of one.
    run_lengths: Vec<u32>,
    /// Decode rounds served so far, per request (indexes `run_lengths`).
    rounds: HashMap<RequestId, usize>,
    /// The generation step whose token is EOS (`eos_after`), per request.
    eos: HashMap<RequestId, u32>,
    /// Every prefill batch the mock received (batch shape for assertions).
    prefill_batches: Vec<Vec<PrefillJob>>,
    /// Every decode batch the mock received (batch shape for assertions).
    decode_batches: Vec<Vec<DecodeJob>>,
    /// Shared prefixes the scheduler told the backend to let go of (P4-10,
    /// GitHub #126), in order: the publishing request of each. The real
    /// adapter drops its leaf handle here, so a test that never sees the
    /// call is looking at a prefix the engine would have pinned forever.
    prefixes_released: Vec<RequestId>,
    /// Requests whose device state the scheduler released, in order.
    released: Vec<RequestId>,
    /// Retained prompt checkpoints the scheduler told the backend to let go
    /// of (GitHub #186), in order: the capturing request of each. A retained
    /// entry the scheduler drops without this call is a device image nothing
    /// will ever free.
    checkpoints_released: Vec<RequestId>,
    /// Retained checkpoints the scheduler moved to KV-RAM (GitHub #190), in
    /// order — each one a device-to-host copy. Kept apart from
    /// `checkpoints_released` so a test can tell a spill from a discard.
    checkpoints_spilled: Vec<RequestId>,
    /// Requests whose next checkpoint spill the backend will fail.
    spill_failures: std::collections::HashSet<RequestId>,
    /// Retained prefixes written to KV-RAM, brought back, and freed there
    /// (GitHub #190), as (publisher, tokens).
    prefixes_spilled: Vec<(RequestId, u32)>,
    prefixes_returned: Vec<(RequestId, u32)>,
    spilled_prefixes_discarded: Vec<(RequestId, u32)>,
    /// Requests whose next prefill batch the backend will fail (GitHub #190).
    prefill_failures: std::collections::HashSet<RequestId>,
    /// Requests whose next asked-for checkpoint capture the backend will
    /// decline (`refuse_capture`) — a leaf with no room in its own image
    /// pool, or a sequence it will not capture.
    capture_refusals: std::collections::HashSet<RequestId>,
    /// The leaf's KV-RAM arena, modelled (GitHub #213), or `None` for a
    /// backend whose blobs are ordinary allocations — today's default, and
    /// what every scenario that is not about placement wants.
    arena: Option<MockHostArena>,
}

/// Which blob a span of [`MockHostArena`] holds: the three kinds the host
/// tier places there (`CONTEXT.md`: "Snapshot blob").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MockBlob {
    /// An evicted live sequence, named by the request it suspends.
    Live(RequestId),
    /// A spilled prompt checkpoint, named by its publisher.
    Checkpoint(RequestId),
    /// A spilled retained prefix, named by publisher and length.
    Prefix(RequestId, u32),
}

/// A first-fit model of the leaf's one pinned KV-RAM arena (GitHub #213,
/// ADR 0030), so the scheduler's behaviour under *fragmentation* is testable
/// on a CPU.
///
/// It is not a second implementation of the vendored `HostPinnedArena` — it
/// is a way to make holes appear on demand. What it shares with the real one
/// is the only thing the scheduler can observe: first fit, bytes counted as
/// asked for, and a blob that finds no long-enough span refused even while
/// the tier's byte ledger says the bytes are free. The real arena's own
/// placement is pinned in `kernel/tests/test_seq_snapshot.cpp`.
///
/// It packs with no alignment, where the real one rounds each blob's start
/// up and gives the padding back to the free list. That is where the two
/// first-fits can diverge, and it is deliberate: a scenario here says how
/// many blobs of what size, never where the bytes land.
#[derive(Debug)]
struct MockHostArena {
    capacity: u64,
    /// The live blobs as (blob, offset, length), sorted by offset, so the
    /// gaps between them are the free spans.
    live: Vec<(MockBlob, u64, u64)>,
}

impl MockHostArena {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            live: Vec::new(),
        }
    }

    /// The lowest offset a blob of `bytes` fits at, or `None` when no span
    /// is long enough.
    fn first_fit(&self, bytes: u64) -> Option<u64> {
        let mut at = 0;
        for &(_, offset, length) in &self.live {
            if offset - at >= bytes {
                return Some(at);
            }
            at = offset + length;
        }
        (self.capacity - at >= bytes).then_some(at)
    }

    fn place(&mut self, blob: MockBlob, bytes: u64) -> bool {
        let Some(offset) = self.first_fit(bytes) else {
            return false;
        };
        let at = self.live.partition_point(|&(_, o, _)| o < offset);
        self.live.insert(at, (blob, offset, bytes));
        true
    }

    /// Give `blob`'s span back. A blob the arena never placed is a no-op:
    /// the scheduler frees what it believes it spilled, and a scenario that
    /// spilled nothing has nothing to return.
    fn free(&mut self, blob: MockBlob) {
        self.live.retain(|&(held, _, _)| held != blob);
    }

    fn used(&self) -> u64 {
        self.live.iter().map(|&(_, _, length)| length).sum()
    }
}

/// A deterministic, recording [`Compute`] implementation for tests.
///
/// Token generation: the token a request emits at decode step `i` mixes the
/// mock's seed, the request id, the request's (learned) seed, and `i` — a
/// fixed, side-effect-free function, so identical runs produce identical
/// streams and different request seeds produce different streams.
pub struct MockCompute {
    seed: u64,
    identity: crate::identity::BlobIdentity,
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
            identity: crate::identity::BlobIdentity::UNSET,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// A mock whose state is produced under `identity` (GitHub #189): what a
    /// test uses to stand in for a real load, or for a *different* load than
    /// the one some blob came from.
    pub fn with_blob_identity(identity: crate::identity::BlobIdentity) -> Self {
        Self {
            identity,
            ..Self::new()
        }
    }

    /// A mock that commits runs like a speculative round (P5-06, GitHub
    /// #154): round `r` of a request commits up to `lengths[r % len]` tokens,
    /// cut short by the request's limit or its EOS, and reports the round's
    /// counters — `length - 1` drafted, `committed - 1` accepted.
    pub fn with_runs(lengths: &[u32]) -> Self {
        assert!(lengths.iter().all(|&n| n >= 1), "a run commits at least its anchor");
        let mock = Self::new();
        mock.inner.lock().unwrap().run_lengths = lengths.to_vec();
        mock
    }

    /// A mock whose KV-RAM blobs are placed first-fit in one arena of
    /// `capacity_bytes` (GitHub #213), the way the leaf's pinned arena
    /// places them.
    ///
    /// Give it the same figure as `SchedulerConfig::host_capacity_bytes` and
    /// the two start out saying the same thing. Every blob here is the
    /// nominal byte, so what fragments the arena is asking it for a blob
    /// longer than any one hole — two free bytes in two holes are not two
    /// bytes of room.
    pub fn with_host_arena(capacity_bytes: u64) -> Self {
        let mock = Self::new();
        mock.inner.lock().unwrap().arena = Some(MockHostArena::new(capacity_bytes));
        mock
    }

    /// The bytes the modelled arena's live blobs hold — the figure
    /// `HostTier::used_bytes` must agree with after every step.
    ///
    /// Panics when the mock has no arena: a test that asks this without
    /// [`Self::with_host_arena`] is asserting against nothing.
    pub fn host_arena_used(&self) -> u64 {
        self.inner
            .lock()
            .unwrap()
            .arena
            .as_ref()
            .expect("host_arena_used needs MockCompute::with_host_arena")
            .used()
    }

    /// Make `request`'s token at generation step `step` its EOS: the round
    /// that commits it finishes the request with `Stop`, emitting only the
    /// tokens before it.
    pub fn eos_after(&self, request: RequestId, step: u32) {
        self.inner.lock().unwrap().eos.insert(request, step);
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

    /// The requests the scheduler released through [`Compute::release`], in
    /// order.
    pub fn released_requests(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().released.clone()
    }

    /// The capturing requests whose retained prompt checkpoint the scheduler
    /// released (GitHub #186), in order.
    pub fn released_checkpoints(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().checkpoints_released.clone()
    }

    /// The capturing requests whose retained checkpoint the scheduler spilled
    /// to KV-RAM (GitHub #190), in order.
    pub fn spilled_checkpoints(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().checkpoints_spilled.clone()
    }

    /// Make the backend fail the next spill of `publisher`'s checkpoint, the
    /// way a leaf that could not write the blob does (GitHub #190).
    pub fn fail_spill(&self, publisher: RequestId) {
        self.inner.lock().unwrap().spill_failures.insert(publisher);
    }

    /// The retained prefixes spilled to KV-RAM (GitHub #190), in order.
    pub fn spilled_prefixes(&self) -> Vec<(RequestId, u32)> {
        self.inner.lock().unwrap().prefixes_spilled.clone()
    }

    /// The spilled prefixes brought back onto the device, in order.
    pub fn returned_prefixes(&self) -> Vec<(RequestId, u32)> {
        self.inner.lock().unwrap().prefixes_returned.clone()
    }

    /// The spilled prefixes whose blob was freed, in order.
    pub fn discarded_spilled_prefixes(&self) -> Vec<(RequestId, u32)> {
        self.inner.lock().unwrap().spilled_prefixes_discarded.clone()
    }

    /// Fail the next prefill batch carrying a job for `request`, the way a
    /// leaf error fails the whole call (GitHub #190).
    pub fn fail_prefill(&self, request: RequestId) {
        self.inner.lock().unwrap().prefill_failures.insert(request);
    }

    /// Make the backend decline `request`'s next checkpoint capture (GitHub
    /// #186): the chunk lands normally and reports that nothing was captured,
    /// which is how a real leaf refuses a bet it cannot afford.
    pub fn refuse_capture(&self, request: RequestId) {
        self.inner.lock().unwrap().capture_refusals.insert(request);
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
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut g = self.inner.lock().unwrap();
        if jobs.iter().any(|job| g.prefill_failures.remove(&job.request)) {
            return Err(ComputeError::Kernel(-1));
        }
        for job in jobs {
            // Learn the request's limits / seed from its params.
            g.limits.insert(job.request, job.params.max_tokens);
            g.seeds.insert(job.request, job.params.seed);
        }
        g.prefill_batches.push(jobs.to_vec());
        Ok(jobs
            .iter()
            .map(|job| PrefillOutcome {
                encode_micros: 0,
                // GitHub #186: a nominal, deterministic restore cost on the
                // job that claimed a checkpoint, so the request log's
                // `restore_ms` is observable on CPU without inventing a
                // clock. 0 on every other job, as a real backend reports.
                restore_micros: if job.checkpoint.is_some() { 1 } else { 0 },
                // The mock holds no device image, but it *does* answer the
                // question the scheduler is really asking — "did the capture
                // you asked for happen?" — so the retained ledger is
                // exercisable. `capture_failures` makes a backend that
                // declines the bet testable too.
                checkpoint_captured: job.capture_checkpoint.is_some()
                    && !g.capture_refusals.remove(&job.request),
                // GitHub #237 / ADR 0034: the `Compute` seam now carries a
                // second kind of answer, and every CPU-only implementation
                // of it has to produce one or the scheduler's tests stop
                // covering the path (ADR 0006). A job that asked for no
                // readout gets none, exactly as a real backend reports.
                readout: job
                    .readout
                    .as_deref()
                    .map(|answers| Self::readout(self.seed, job.request, answers)),
            })
            .collect())
    }

    fn release_prefix(&self, publisher: RequestId, _tokens: u32) {
        self.inner.lock().unwrap().prefixes_released.push(publisher);
    }

    fn blob_identity(&self) -> crate::identity::BlobIdentity {
        self.identity
    }

    fn release_checkpoint(&self, publisher: RequestId) {
        // The device image and, if the checkpoint had been spilled, its
        // KV-RAM blob: the real adapter drops both here.
        self.free_blob(MockBlob::Checkpoint(publisher));
        self.inner
            .lock()
            .unwrap()
            .checkpoints_released
            .push(publisher);
    }

    // GitHub #190: a materialized checkpoint blob is one nominal byte, so
    // `host_capacity_bytes` counts how many spilled checkpoints KV-RAM holds.
    fn checkpoint_snapshot_size(&self, _publisher: RequestId) -> Result<u64, ComputeError> {
        Ok(1)
    }

    fn spill_checkpoint(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        if self.inner.lock().unwrap().spill_failures.remove(&publisher) {
            return Err(ComputeError::Kernel(-1));
        }
        // Placed before it is recorded: a spill the arena turns away is one
        // that did not happen, and a test reading `spilled_checkpoints()`
        // must not see it.
        if !self.place_blob(MockBlob::Checkpoint(publisher), 1) {
            return Err(ComputeError::Kernel(NO_HOST_ROOM));
        }
        self.inner.lock().unwrap().checkpoints_spilled.push(publisher);
        Ok(1)
    }

    // GitHub #190: a retained prefix's blob is one nominal byte too.
    fn prefix_snapshot_size(&self, _publisher: RequestId, _tokens: u32) -> Result<u64, ComputeError> {
        Ok(1)
    }

    fn spill_prefix(&self, publisher: RequestId, tokens: u32) -> Result<u64, ComputeError> {
        // Placed before it is recorded, as in `spill_checkpoint`.
        if !self.place_blob(MockBlob::Prefix(publisher, tokens), 1) {
            return Err(ComputeError::Kernel(NO_HOST_ROOM));
        }
        self.inner.lock().unwrap().prefixes_spilled.push((publisher, tokens));
        Ok(1)
    }

    fn restore_prefix(&self, publisher: RequestId, tokens: u32, _slot: u32) -> Result<u64, ComputeError> {
        self.inner.lock().unwrap().prefixes_returned.push((publisher, tokens));
        Ok(1)
    }

    fn discard_spilled_prefix(&self, publisher: RequestId, tokens: u32) {
        self.free_blob(MockBlob::Prefix(publisher, tokens));
        self.inner
            .lock()
            .unwrap()
            .spilled_prefixes_discarded
            .push((publisher, tokens));
    }

    fn release(&self, request: RequestId) {
        self.inner.lock().unwrap().released.push(request);
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let mut g = self.inner.lock().unwrap();
        g.decode_batches.push(jobs.to_vec());
        Ok(jobs
            .iter()
            .map(|job| {
                let round = {
                    let rounds = g.rounds.entry(job.request).or_insert(0);
                    *rounds += 1;
                    *rounds - 1
                };
                let length = match g.run_lengths.len() {
                    0 => 1,
                    n => g.run_lengths[round % n],
                };
                // An explicit stop_after overrides the learned limit; an
                // unlearned request (no prefill seen) is treated as
                // unbounded.
                let limit = g
                    .stops
                    .get(&job.request)
                    .copied()
                    .or_else(|| g.limits.get(&job.request).copied().flatten());
                let eos = g.eos.get(&job.request).copied();
                let seed = g.seeds.get(&job.request).copied().unwrap_or(0);
                let mut run = Vec::new();
                let mut committed = 0;
                let mut finish = None;
                for _ in 0..length {
                    let step = *g.generated.entry(job.request).or_insert(0);
                    if limit.is_some_and(|n| step >= n) {
                        break;
                    }
                    *g.generated.get_mut(&job.request).unwrap() += 1;
                    committed += 1;
                    if eos == Some(step) {
                        finish = Some(FinishReason::Stop);
                        break;
                    }
                    run.push(Self::mix(self.seed, job.request, seed, step));
                }
                if committed == 0 {
                    // Without `eos_after` the mock has no real EOS token —
                    // its stop condition is a token-count cap, reached
                    // before this round committed anything: `Length`.
                    return DecodeOutcome::finished(FinishReason::Length);
                }
                DecodeOutcome {
                    tokens: run,
                    finish,
                    spec: (!g.run_lengths.is_empty())
                        .then(|| SpecCounters::round(length - 1, committed - 1)),
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

    fn host_blob_fits(&self, bytes: u64) -> bool {
        let g = self.inner.lock().unwrap();
        match g.arena.as_ref() {
            Some(arena) => arena.first_fit(bytes).is_some(),
            None => true,
        }
    }

    fn evict(&self, request: RequestId) -> Result<u64, ComputeError> {
        let mut g = self.inner.lock().unwrap();
        if let Some(arena) = g.arena.as_mut() {
            if !arena.place(MockBlob::Live(request), 1) {
                // What the leaf reports when no span is long enough
                // (`crate::seq::NO_HOST_ROOM`). The scheduler probes with
                // `host_blob_fits` first, so reaching this means a test
                // drove the call directly.
                return Err(ComputeError::Kernel(NO_HOST_ROOM));
            }
        }
        Ok(1)
    }

    fn restore(&self, request: RequestId, _context_tokens: u32) -> Result<(), ComputeError> {
        self.free_blob(MockBlob::Live(request));
        Ok(())
    }

    fn discard_snapshot(&self, request: RequestId) {
        self.free_blob(MockBlob::Live(request));
    }
}

impl MockCompute {
    /// Place `blob` of `bytes` in the modelled arena, if there is one.
    /// `false` when it is there and has no span long enough.
    fn place_blob(&self, blob: MockBlob, bytes: u64) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.arena.as_mut() {
            Some(arena) => arena.place(blob, bytes),
            None => true,
        }
    }

    fn free_blob(&self, blob: MockBlob) {
        if let Some(arena) = self.inner.lock().unwrap().arena.as_mut() {
            arena.free(blob);
        }
    }

    /// How much of the mock's modelled distribution sits *outside* the
    /// answer tokens, in nats. Small and nonzero on purpose: a readout
    /// whose answer mass were exactly 1 would let a caller that forgot to
    /// check the mass pass every CPU test and fail on the card, and the
    /// real measurement is a median 99.8% held by the declared options
    /// (`docs/findings/2026-09-19-typed-option-logit-readout.md`).
    const OUTSIDE_THE_ANSWERS: f64 = 0.002;

    /// The deterministic readout (GitHub #237): a pure function of (mock
    /// seed, request id, answer token id, slot), shaped like a real one
    /// rather than uniform — separated logits with one clear winner, an
    /// answer mass just under 1, and an unrestricted argmax that *is* a
    /// declared answer, which is what the served model does on every row
    /// the finding scored.
    fn readout(seed: u64, request: RequestId, answers: &[TokenId]) -> Readout {
        let logits: Vec<f32> = answers
            .iter()
            .enumerate()
            .map(|(slot, &id)| {
                let mixed = Self::mix(seed, request, u64::from(id), slot as u32);
                // -4 ..= +4 in thousandths: wide enough to separate slots,
                // fine enough that two of one batch practically never tie.
                (f64::from(mixed % 8_000) / 1000.0 - 4.0) as f32
            })
            .collect();
        let answer_lse = log_sum_exp(&logits);
        let winner = logits
            .iter()
            .enumerate()
            .fold(None::<(usize, f32)>, |best, (index, &value)| match best {
                Some((_, high)) if !(value > high) => best,
                _ => Some((index, value)),
            });
        Readout {
            full_log_sum_exp: if answer_lse.is_finite() {
                answer_lse + Self::OUTSIDE_THE_ANSWERS
            } else {
                0.0
            },
            full_argmax: winner.map_or(0, |(index, _)| answers[index]),
            logits,
        }
    }

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
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
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

    fn release_checkpoint(&self, publisher: RequestId) {
        self.inner.release_checkpoint(publisher);
    }

    fn checkpoint_snapshot_size(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        self.inner.checkpoint_snapshot_size(publisher)
    }

    fn spill_checkpoint(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        self.inner.spill_checkpoint(publisher)
    }
}
