//! The engine's scheduling contract.
//!
//! Two traits define the public surface:
//!
//! - [`Scheduler`] — what the *server* drives: submit requests, advance the
//!   engine, stream events. This is the stable contract the `ignis-server`
//!   crate codes against.
//! - [`Compute`] — the *compute seam*: the only GPU-coupled step (prefill /
//!   decode). Production is the kernel leaf (C ABI, see `ffi.rs`); tests use
//!   a deterministic mock. Keeping this seam narrow is what makes the whole
//!   scheduler (admission, lanes, batched prefill, eviction) CPU-testable
//!   without a GPU (ADR 0006).

use crate::decision::Readout;
use crate::types::{
    ComputeError, DecodeParams, FinishReason, LaneId, RequestClass, RequestId, RequestInput,
    SchedEvent, SpecCounters, SubmitError, TokenId,
};

/// The shared prefix a prefill job claims (P4-10, GitHub #126, ADR 0024):
/// which request published it, and how many leading prompt tokens it covers.
///
/// The backend needs the publisher's identity rather than the scheduler's own
/// entry id because the leaf's prefix is what actually owns the pages, and
/// the publisher is the request whose prefill produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedPrefixClaim {
    /// The request whose prefill published the prefix.
    pub publisher: RequestId,
    /// The leading prompt tokens the prefix covers — always a whole number
    /// of KV pages, and always equal to this job's `start_position`.
    pub tokens: u32,
}

/// The retained **prompt checkpoint** a prefill job claims (GitHub #186, ADR
/// 0029): which request captured it, and how many leading prompt tokens it
/// covers.
///
/// Shaped like [`SharedPrefixClaim`] and for the same reason — the backend
/// needs a name for the device-resident thing, and the request that captured
/// it is that name. The difference is what the number means: a shared
/// prefix's `tokens` is always whole KV pages, a checkpoint's is the
/// generation opener **wherever it falls**, so the claimant copies the
/// partial tail page rather than sharing it (CONTEXT.md, "publish point").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointClaim {
    /// The request whose prefill captured the checkpoint. It is long
    /// finished; its id is never reused, so it stays a valid name for the
    /// bytes it left behind.
    pub publisher: RequestId,
    /// The leading prompt tokens the checkpoint covers — always equal to this
    /// job's `start_position`.
    pub tokens: u32,
    /// Where the retained bytes live. Device claims clone an image and share
    /// pages; KV-RAM claims restore one materialized blob into a fresh slot.
    pub source: crate::checkpoint::ReuseSource,
}

/// Where a prefill job leaves retained state (GitHub #215, ADR 0030): after
/// `tokens` of the prompt, into retained slot `slot`.
///
/// The scheduler takes the slot before it builds the job, so the backend
/// never allocates device memory for a publish or a capture: it copies the
/// sequence's mutable state into the slot it is named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedAt {
    /// The prompt position the state is taken at.
    pub tokens: u32,
    /// The retained slot the image goes into (`0..retained_slots`).
    pub slot: u32,
}

/// One prefill job handed to the compute backend (batched prefill groups
/// several of these into one GPU batch to saturate the GPU and cut burst TTFT).
#[derive(Debug, Clone)]
pub struct PrefillJob {
    /// The request being prefilled.
    pub request: RequestId,
    /// The prompt tokens to warm the KV for. A request that reuses a
    /// cached sibling prefix (core-07) carries only its *tail* — the
    /// leading shared tokens are already warm in the pool (the shared
    /// prefix's blocks are bound read-only by the kernel leaf). An empty
    /// tail (a full-prompt match) warms nothing: the job only sets up the
    /// decode state.
    pub tokens: Vec<TokenId>,
    /// Total sequence reservation, including the prompt and effective
    /// generation cap. The runtime uses it when it first allocates a leaf
    /// sequence for this request.
    pub context_tokens: u32,
    /// Position of `tokens[0]` in the sequence (nonzero after shared-prefix
    /// reuse).
    pub start_position: u32,
    /// The request's generation parameters (carried so the backend can set
    /// up the decode state; prefill only warms the KV).
    pub params: DecodeParams,
    /// The shared prefix this request claims (P4-10, GitHub #126), if any.
    /// Set on the request's **first** job: the backend allocates its
    /// sequence against the prefix, which shares the leading KV pages in
    /// place and clones the mutable state device-to-device, so the sequence
    /// begins at `start_position` with the publisher's state.
    pub shared_prefix: Option<SharedPrefixClaim>,
    /// Publish this request's first `tokens` tokens as a shared prefix once
    /// this chunk lands (P4-10, GitHub #126), its image in the retained slot
    /// named (GitHub #215). Set only on the chunk that ends exactly there: the
    /// mutable state a claimant clones is the state at the prefix's end, so
    /// the publish happens at that boundary and nowhere else.
    pub publish_prefix: Option<RetainedAt>,
    /// The retained prompt checkpoint this request claims (GitHub #186), if
    /// any. Set on the request's **first** job, exactly as `shared_prefix` is:
    /// the backend allocates its sequence against the checkpoint, sharing the
    /// whole KV pages under it, cloning its mutable state and copying its
    /// partial tail page, so the sequence begins at `start_position` — the
    /// generation opener — with the captured state.
    ///
    /// Never set together with `shared_prefix`: a checkpoint claim subsumes
    /// one (the checkpoint sits on a shared prefix of its own), and longest
    /// reuse wins between the two.
    pub checkpoint: Option<CheckpointClaim>,
    /// Capture this request's state as a prompt checkpoint once this chunk
    /// lands (GitHub #186), its image in the retained slot named (GitHub
    /// #215). Set only on the chunk that ends exactly at the generation
    /// opener: the state a claimant receives is the state *there*, and a chunk
    /// that overshot it would have moved that state on — the same reason
    /// `publish_prefix` exists.
    ///
    /// The capture is a pure read of the live sequence: it perturbs nothing,
    /// and the request goes on prefilling its last few prompt tokens and
    /// decoding as if it had not been asked.
    pub capture_checkpoint: Option<RetainedAt>,
    /// The request's multimodal part (GitHub #178), whole-prompt: the
    /// backend reads this job's span of it at `start_position`. The chunk
    /// holds at most one media item's placeholders
    /// ([`crate::vision::Multimodal::cap_chunk`]).
    pub multimodal: Option<std::sync::Arc<crate::vision::Multimodal>>,
    /// The **answer tokens** this job reads out (GitHub #237, ADR 0034):
    /// the vocabulary entries naming the options of a decision, whose
    /// logits at this chunk's last position are the answer. `None` on every
    /// job that is not a decision's last chunk — which is every job the
    /// engine has today — and a job that asks for no readout pays for none:
    /// the backend allocates no logits buffer and gathers nothing.
    ///
    /// Set only on the chunk that ends at the prompt's **last position**,
    /// because that is the position whose next-token distribution holds the
    /// decision. The chunk must also carry at least one token: a chunk with
    /// nothing to prefill runs no forward pass, so there would be no logits
    /// to read (GitHub #238 trims a reuse claim for exactly this reason).
    ///
    /// The order is the answer's order — [`PrefillOutcome::readout`]'s
    /// `logits[i]` is the logit of `readout[i]`.
    pub readout: Option<std::sync::Arc<[TokenId]>>,
}

/// One decode job: a single lane step for a running request.
#[derive(Debug, Clone)]
pub struct DecodeJob {
    /// The request decoding.
    pub request: RequestId,
    /// The resident lane it holds (used for KV block mapping).
    pub lane: LaneId,
    /// The request's generation parameters (sampler setup, `max_tokens` /
    /// EOS handling, fixed seed — ADR 0007).
    pub params: DecodeParams,
    /// Tokens the request may still emit, `>= 1` (P5-06, GitHub #154): the
    /// scheduler's remaining reservation, which a speculative round clamps
    /// its lane's extent to so the sequence never commits past the text the
    /// request can emit.
    pub remaining_tokens: u32,
}

/// One job's result from a prefill step (GitHub #192): what the chunk cost
/// beyond warming its KV. Today that is the media encode alone — everything
/// else a prefill does is already attributable from the chunk event itself.
///
/// `encode_micros` is the wall time the leaf's media encode took for this
/// chunk, and is 0 on every chunk that encoded nothing: a text chunk, a
/// chunk that reuses the embedding an earlier chunk of the same item
/// encoded, and a full-prefix match that warms nothing at all.
///
/// A failed prefill batch reports nothing at all — it emits no chunk event,
/// so an encode the failure discarded is never counted. Its retry
/// (`MAX_PREFILL_ATTEMPTS`) re-encodes whatever the batch dropped, and
/// *that* encode is counted. So a request whose second chunk failed carries
/// its item's encode twice: once from the chunk that reported before the
/// failure, once from the retry. What a request's total answers is "how
/// much encode work did this request cause", not "what did this image cost
/// to encode".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefillOutcome {
    /// Wall time this chunk spent encoding a media item, in microseconds.
    pub encode_micros: u64,
    /// Wall time this job spent restoring retained state, in microseconds
    /// (GitHub #186): the device-to-device clone of a claimed prompt
    /// checkpoint into this request's slot. 0 on every job that claimed
    /// nothing. It is measured rather than assumed (ADR 0024) and it is the
    /// `restore_ms` the request log reports, so the TTFT a reuse bought is
    /// attributable against what the reuse itself cost.
    pub restore_micros: u64,
    /// Whether this job's `capture_checkpoint` actually produced a retained
    /// image (GitHub #186).
    ///
    /// A capture is a **bet**, never certain work: a backend that cannot take
    /// one — a sequence the leaf refuses to capture — says so here and
    /// completes the chunk normally. The scheduler records the retained entry
    /// only on `true`, and gives the slot back on `false`, so its ledger and
    /// the device never disagree about what exists.
    pub checkpoint_captured: bool,
    /// The **readout** this job asked for (GitHub #237, ADR 0034), if it
    /// asked for one: the logits of its [`PrefillJob::readout`] answer
    /// tokens at the chunk's last position, the full-vocabulary log-sum-exp
    /// behind them, and the unrestricted argmax.
    ///
    /// This is the only thing a decision ever produces — no token is
    /// sampled for it and none is emitted. What does *not* cross this seam
    /// is the buffer it was read from: one f32 per vocabulary column, near
    /// a megabyte per decision, gathered on the backend's side and dropped
    /// there.
    pub readout: Option<Readout>,
}

impl PrefillOutcome {
    /// One outcome per job, none of which encoded anything — what every
    /// text-only backend returns.
    pub fn nothing_encoded(jobs: usize) -> Vec<Self> {
        vec![Self::default(); jobs]
    }
}

/// One job's result from a decode step (GitHub #61 / P1-25; a run since
/// P5-06, GitHub #154): the tokens the round committed for the lane, in
/// order, and whether the request finished.
///
/// Today's round commits one token ([`DecodeOutcome::token`]); a speculative
/// round commits 1..=k+1 ([`DecodeOutcome::run`]). A request that finishes
/// carries the reason — the scheduler forwards it straight into the
/// [`SchedEvent::Done`] it emits, which the server maps to `finish_reason` —
/// after whatever tokens preceded it in the round: a run cut at EOS is the
/// tokens before the EOS plus [`FinishReason::Stop`]. `tokens` is empty only
/// on a finished outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    /// The committed tokens to emit, in order.
    pub tokens: Vec<TokenId>,
    /// Why the request finished this round, after `tokens`; `None` while it
    /// keeps running.
    pub finish: Option<FinishReason>,
    /// The round's speculative counters, when it was a verify round.
    pub spec: Option<SpecCounters>,
}

impl DecodeOutcome {
    /// One committed token; the request keeps running.
    pub fn token(token: TokenId) -> Self {
        Self::run(vec![token])
    }

    /// A run of committed tokens (`tokens.len() >= 1`); the request keeps
    /// running.
    pub fn run(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            finish: None,
            spec: None,
        }
    }

    /// The request finished without committing a token to emit.
    pub fn finished(reason: FinishReason) -> Self {
        Self::run_then_finished(Vec::new(), reason)
    }

    /// `tokens`, then the request finished.
    pub fn run_then_finished(tokens: Vec<TokenId>, reason: FinishReason) -> Self {
        Self {
            tokens,
            finish: Some(reason),
            spec: None,
        }
    }

    /// The same outcome, from a verify round with these counters.
    pub fn with_spec(self, spec: SpecCounters) -> Self {
        Self {
            spec: Some(spec),
            ..self
        }
    }
}

/// The [`ComputeError::Kernel`] code a backend reports when KV-RAM has no
/// free span long enough for a blob (GitHub #213): the leaf's
/// `IGNIS_SEQ_ERR_NO_HOST_ROOM`, restated here because the scheduler and its
/// mock are CPU-only and the leaf's ABI constants are not.
///
/// The scheduler asks [`Compute::host_blob_fits`] before it spills or
/// evicts, so it does not branch on this — it is what a backend returns when
/// the arena changed hands between the probe and the call, and the refusal
/// path is the same one every other failed spill takes.
pub const NO_HOST_ROOM: i32 = -6;

/// The compute seam the scheduler drives for actual token generation.
///
/// This is the *only* GPU-coupled step in the engine. The scheduler's logic —
/// admission, lane assignment, batched prefill grouping, eviction, state
/// machine — never touches the GPU directly; it only calls this trait. That
/// is the seam that keeps the engine testable on a CPU-only machine (ADR
/// 0006: GPU testing is exclusive; a mock stands in for the kernel leaf).
pub trait Compute: Send + Sync {
    /// Prefill a batch of prompts, warming their KV (fills the block tables).
    /// No tokens are emitted; this only sets the request up for decode.
    ///
    /// Returns one [`PrefillOutcome`] per job, in order (GitHub #192) — the
    /// same shape [`Compute::decode_step`] returns its outcomes in. A failed
    /// batch returns none of them: the scheduler retries the whole batch, so
    /// a partial answer would be attributed to a chunk that never landed.
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError>;

    /// Run one decode round over every running lane. Returns, per job in
    /// order, the [`DecodeOutcome`]: the tokens the round committed for that
    /// lane (one, or a run under speculation) and, when the request finished
    /// (EOS or `max_tokens`), why.
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError>;

    /// Release leaf-owned state for a request that completed or was
    /// cancelled (never a request being evicted to the host tier — that
    /// goes through [`Compute::evict`], which snapshots before releasing).
    /// CPU-only compute implementations need no lifecycle bookkeeping.
    fn release(&self, _request: RequestId) {}

    /// Release the backend's own handle on the `tokens`-long shared prefix
    /// `publisher` published (P4-10, GitHub #126), once the scheduler's last
    /// claimant has gone. The leaf's pages return to the pool when every
    /// sequence holding them has been released too, so this is a handle drop
    /// and not a free. A head is named by both: one request may publish its
    /// system block and then a chained head over it (#187 x #188).
    /// CPU-only compute implementations hold no such handle.
    fn release_prefix(&self, _publisher: RequestId, _tokens: u32) {}

    // ── core-06: the KV-RAM host tier (P4-07, GitHub #125, ADR 0024) ────
    //
    // The host tier itself (`crate::host::HostTier`) is pure CPU bookkeeping
    // — tier membership, byte-budget accounting, LRU order — so it stays
    // testable without a GPU (ADR 0006). The four methods below are where
    // that bookkeeping meets real device state: production (`RuntimeCompute`)
    // moves bytes through the kernel leaf's snapshot/restore ABI over pinned
    // host memory; every CPU-only `Compute` (the mock, test stubs) keeps the
    // safe zero-byte default, which preserves today's behavior exactly —
    // those backends hold no real per-sequence device state for eviction to
    // move in the first place.

    /// Bytes a snapshot of `request` would need right now (ADR 0024): a
    /// cheap query, no bytes moved and no leaf-owned state released — the
    /// scheduler calls this *before* deciding whether the host tier's byte
    /// budget can hold the snapshot, so a tier that cannot make room never
    /// pays for (or discards) a snapshot it cannot use. `Err` while
    /// `request` is mid-chunk (the leaf's `NOT_AT_BOUNDARY`) — unreachable
    /// in production by construction (only a `Running`, chunk-complete
    /// request is ever an eviction candidate), but still a `Result` rather
    /// than an infallible query since the leaf's ABI is.
    fn snapshot_size(&self, _request: RequestId) -> Result<u64, ComputeError> {
        Ok(0)
    }

    /// Whether KV-RAM has anywhere to put a blob of `bytes` right now
    /// (GitHub #213, ADR 0030). See [`NO_HOST_ROOM`], the code a backend
    /// that was asked anyway reports.
    ///
    /// [`HostTier`](crate::host::HostTier)'s byte ledger says whether the
    /// tier may *hold* the blob; this says whether the one pinned arena the
    /// blobs live in has a free span long enough to *place* it, which a
    /// ledger's worth of free bytes scattered across holes does not. The
    /// scheduler asks before it spills or evicts, so a refusal costs no
    /// device work — and when the answer is no, the tier's victim order runs
    /// exactly as it does for a full budget.
    ///
    /// A backend whose blobs are ordinary allocations has no arena to
    /// fragment and always fits, which is the default.
    fn host_blob_fits(&self, _bytes: u64) -> bool {
        true
    }

    /// Snapshot `request`'s device state into pinned host memory and
    /// release its GPU-resident sequence — its KV pages, GDN slot and conv
    /// taps (ADR 0024): the host tier's evict-to-tier transport. Returns
    /// the snapshot's byte size (normally identical to
    /// [`Compute::snapshot_size`]'s answer moments earlier — nothing else
    /// runs between the two calls on this single-threaded scheduler). The
    /// caller (the scheduler) times the call itself for the request log.
    fn evict(&self, _request: RequestId) -> Result<u64, ComputeError> {
        Ok(0)
    }

    /// Restore `request` from the snapshot [`Compute::evict`] took:
    /// re-acquire a GPU sequence reserving `context_tokens` and write the
    /// blob back into it. The request resumes decoding from exactly where
    /// it was evicted — no re-prefill. `Err` (`BAD_SNAPSHOT` at the leaf, or
    /// allocation failure) leaves the snapshot in place; the caller falls
    /// back to discarding it and re-prefilling, the same fallback the
    /// tier's own byte-budget discard uses.
    fn restore(&self, _request: RequestId, _context_tokens: u32) -> Result<(), ComputeError> {
        Ok(())
    }

    /// Discard `request`'s pending snapshot without restoring it (the host
    /// tier's byte budget could not hold it, or a restore attempt failed):
    /// frees the pinned buffer. The request re-prefills from scratch later.
    /// A request with no pending snapshot is a no-op.
    fn discard_snapshot(&self, _request: RequestId) {}

    // ── Prompt checkpoints on the device (GitHub #186, ADR 0029) ─────────
    //
    // Retained state's *policy* — which checkpoints exist, which one a
    // prompt matches, and which one a live request takes back — is CPU
    // bookkeeping in `crate::checkpoint`, testable without a GPU exactly as
    // the host tier's is, and what bounds it is the scheduler's retained slots
    // (GitHub #215). The methods below are where that bookkeeping meets device
    // memory. Capture and claim are not here at all: they ride on
    // [`PrefillJob`], because a claim has to happen at the moment the backend
    // builds the sequence and a capture at the moment the chunk ending on the
    // opener lands.

    /// The compatibility identity of the state this backend produces
    /// (GitHub #189, ADR 0029): the artifact it loaded, the KV format its
    /// pages are written in, the blob layout version its sequence pool
    /// writes, and the drafter bound at load.
    ///
    /// Read from the backend rather than assembled here, because only the
    /// backend knows two of the four. The blob layout version in particular
    /// belongs to the leaf — its state-section table is internal by design
    /// (ADR 0024), so a version this crate held its own copy of would not
    /// move when the table did, and a stale blob would restore garbage
    /// instead of being refused.
    ///
    /// The default is [`BlobIdentity::UNSET`]: a backend that retains nothing
    /// has no blobs to hand anybody, and an identity that matches no real
    /// load is the right answer for one.
    fn blob_identity(&self) -> crate::identity::BlobIdentity {
        crate::identity::BlobIdentity::UNSET
    }

    /// Release the device image `publisher`'s prompt checkpoint left behind,
    /// and the backend's own hold on the KV pages under it.
    ///
    /// Called when the scheduler discards a retained entry — a live request
    /// taking the pages back (retained state is the first victim, ADR 0023 as
    /// amended), or, from #187 on, a successor superseding it. The pages come
    /// back to the pool when every other holder has released too, so this is a
    /// handle drop and not a free, exactly like
    /// [`Compute::release_prefix`].
    fn release_checkpoint(&self, _publisher: RequestId) {}

    /// Bytes a materialized KV-RAM blob of a retained checkpoint needs.
    fn checkpoint_snapshot_size(&self, _publisher: RequestId) -> Result<u64, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }

    /// Lazily materialize a retained checkpoint into pinned host memory and
    /// release its device image. Returns the host allocation's byte size.
    fn spill_checkpoint(&self, _publisher: RequestId) -> Result<u64, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }

    // ── retained prefixes in KV-RAM (GitHub #190) ───────────────────────
    //
    // A retained prefix is named as a published head is: its publisher and
    // its length (one request may publish two heads, #187 x #188).

    /// Bytes a materialized KV-RAM blob of the prefix needs.
    fn prefix_snapshot_size(&self, _publisher: RequestId, _tokens: u32) -> Result<u64, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }

    /// Write the prefix's blob into pinned host memory. The device prefix is
    /// left as it is: the scheduler releases it the usual way, through
    /// [`Compute::release_prefix`], once the blob exists. Returns its size.
    fn spill_prefix(&self, _publisher: RequestId, _tokens: u32) -> Result<u64, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }

    /// Bring a spilled prefix back onto the device as a published prefix
    /// under the same name, its image in retained slot `slot` (GitHub #215),
    /// from its blob, which stays in KV-RAM. Returns the restore's wall time
    /// in microseconds.
    fn restore_prefix(
        &self,
        _publisher: RequestId,
        _tokens: u32,
        _slot: u32,
    ) -> Result<u64, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }

    /// Free a spilled prefix's blob.
    fn discard_spilled_prefix(&self, _publisher: RequestId, _tokens: u32) {}
}

/// The engine's scheduling interface — what the server drives.
///
/// The server submits requests and calls [`Scheduler::advance`] in a loop,
/// streaming the emitted [`SchedEvent`]s back to clients (SSE) and into the
/// telemetry writer. Production is the real engine (kernel leaf via FFI);
/// tests use a mock.
pub trait Scheduler: Send {
    /// Enqueue a new request. `class` is the admission / backfill class the
    /// server assigns at submission time (a foreground interactive request vs
    /// a background agent subtask); it drives admission priority + eviction
    /// order (ADR 0004). Returns the request's id. Fails with
    /// [`SubmitError::Full`] when the engine cannot admit it right now — the
    /// caller should retry or queue.
    fn submit(
        &mut self,
        input: RequestInput,
        class: RequestClass,
    ) -> Result<RequestId, SubmitError>;

    /// Abort an in-flight request. The scheduler must stop dealing it work
    /// and release its resources no later than the next [`Scheduler::advance`].
    /// Returns `false` when the request is unknown or already complete.
    fn cancel(&mut self, request: RequestId) -> bool;

    /// Advance the engine by one scheduling step:
    /// - run **batched prefill** for queued requests (grouped into one GPU
    ///   batch to saturate the GPU and cut burst TTFT),
    /// - decode one token on each resident decode lane.
    ///
    /// Returns the events emitted this step (new tokens, completions,
    /// evictions, admissions).
    fn advance(&mut self) -> Vec<SchedEvent>;

    /// True when no request is in flight (nothing to schedule).
    fn is_idle(&self) -> bool;

    /// The loaded model id (for `GET /v1/models`).
    fn model_id(&self) -> &str;

    /// The model's context for one sequence, prompt plus generation, in
    /// tokens (for `GET /v1/models`; `submit` refuses anything over it).
    fn max_sequence_tokens(&self) -> u32;

    /// The operating mode (reported by telemetry interval lines).
    fn mode(&self) -> crate::types::EngineMode;

    /// What this scheduler has occupied right now (GitHub #216, ADR 0030).
    ///
    /// Plain reads of fields the admission machine already maintains for its
    /// own decisions — no allocation, no clock, no call into the leaf. The
    /// server reads it once per step, on the thread that owns the scheduler,
    /// and carries the answer along the fact channel it already sends after
    /// every step, so nothing downstream ever reaches across a thread for it.
    fn occupancy(&self) -> Occupancy;
}

/// What a scheduler has occupied, at one instant (GitHub #216, ADR 0030).
///
/// `kv_used_pages` and `kv_pool_pages` are the two terms of the same pool, so
/// a reader has both on the same reading and never a ratio that hides which of
/// them moved. The pool's capacity rides along for exactly that reason;
/// nothing else constant does, because a constant read on every step is a
/// constant the load already knew before the first one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Occupancy {
    /// Main-pool KV pages reserved by running requests, shared prefixes and
    /// retained tail pages.
    pub kv_used_pages: u32,
    /// The main pool's page capacity — constant for the process's life.
    pub kv_pool_pages: u32,
    /// The host KV-RAM tier's bytes in use: live snapshots and retained
    /// blobs together.
    pub kv_ram_used_bytes: u64,
}
