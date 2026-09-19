//! Safe ownership of the step ABI and its [`ignis_core::Compute`] adapter.
//!
//! The runtime owns a loaded model, one opaque sequence per scheduler
//! request, integer error-code mapping, and the sequence-release lifecycle.
//! The C ABI adapter lands with P1-23; the small [`StepLeaf`] seam lets this
//! ownership logic be tested today against a CPU stub.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ignis_core::vision::{MediaItem, Multimodal};
use ignis_core::{
    BlobIdentity, Compute, ComputeError, DecodeJob, DecodeOutcome, DecodeParams, FinishReason,
    N_DECODE_LANES, PrefillJob, PrefillOutcome, Readout, RequestId, RetainedAt, SpecCounters,
    TokenId,
};

#[cfg(feature = "cuda")]
mod cuda_leaf;
#[cfg(feature = "cuda")]
pub use cuda_leaf::{CudaLeaf, CudaLeafConfig, CudaModel, PlannedReservations};

/// Tokens held by one physical KV page, in either format
/// (`kPagedKVPageSize`). Re-exported from `ignis-core` so the server's
/// scheduler accounting and the leaf name the same constant.
pub use ignis_core::KV_PAGE_TOKENS;

/// The default prefill chunk width, in tokens (spec
/// `.scratch/runtime/specs/02-real-prefill.md`): the reference's own
/// default, left alone — the chunk width is a knob this phase exposes,
/// not a number it tunes. Unconditional on the `cuda` feature: it is a
/// plain number, and both `ignis_server::config` (always compiled) and
/// [`CudaLeafConfig::default`] (`cuda` only) fall back to it, so it has to
/// live somewhere both can reach without one depending on the other.
pub const DEFAULT_PREFILL_CHUNK: u32 = 1024;

/// The prefill chunk width's alignment rule, in tokens: the reference's
/// own alignment, and a multiple of the 64-token chunk the vendored GDN
/// chunked kernels work in.
pub const PREFILL_CHUNK_ALIGNMENT: u32 = 128;

/// The default maximum per-sequence context, in tokens: a 32,768-token
/// prompt plus an 8,192-token generation budget. G2's largest cell is a
/// 32K prompt, so the default must admit one without editing code (spec
/// `02-real-prefill.md`, user story 22).
pub const DEFAULT_MAX_CONTEXT: u32 = 32_768 + 8_192;

/// The paged-KV pool's auto byte budget for a configured `max_context`
/// under `format` (P4-04, GitHub #122): [`ignis_core::DEFAULT_KV_POOL_BYTES`]
/// (4 GiB), raised if one configured context would not fit inside it.
///
/// Deliberately not `slot_count * max_context`: reserving a full
/// 40,960-token context for each of the eight decode lanes is ~20 GiB of
/// BF16 paged KV at this model's geometry, which does not fit next to
/// ~19 GB of weights. The pool is sized so one sequence can take the whole
/// 32K cell and the other lanes still have a working budget; a request the
/// free pool cannot cover is a scheduler admission decision, not a load
/// failure.
///
/// The budget is in bytes, and what it *buys* is derived from the format:
/// 4 GiB is 65,536 resident BF16 tokens and 465,984 hq-e8-2b ones. That is
/// why this replaced the former `kv_pool_tokens_for` — a token target is
/// exactly the thing that cannot be format-independent.
pub fn auto_kv_pool_bytes(format: ignis_core::KvFormat, max_context: u32) -> u64 {
    ignis_core::auto_kv_pool_bytes(format, ignis_core::KvGeometry::qwen38_27b(), max_context)
}

/// The CUDA context's device bytes (GitHub #210): the dedicated usage of a
/// process that has only run `ignis_device_create`, read from the WDDM
/// counter `\GPU Process Memory(pid_*)\Dedicated Usage` (Task Manager's
/// figure) on the RTX 5090, 2026-09-17.
///
/// A plan line: the budget starts from the free memory NVML reports before
/// the context exists (`CudaDevice::nvml_memory`), so the context is one more
/// thing the process holds inside it.
pub const CUDA_CONTEXT_BYTES: u64 = 452_595_712;

/// What a load holds beyond every reservation the VRAM plan names (GitHub
/// #210): allocator rounding, the decode graph captures, handles the kernel
/// creates lazily. Measured on the RTX 5090 on 2026-09-17 at the Makefile
/// defaults with `--vision` (262K hq-e8-2b, DFlash2/7): the server's WDDM
/// dedicated usage right after load minus the context and every line the load
/// allocates (the plan's total less the retained checkpoint ledger line, which
/// that load did not allocate) -- 30,047,830,016 B held against a 4,653,252,608 B KV pool, in a
/// run planned with no context or residual line. The confirming run, with
/// both lines, held 30,704,246,784 B against 30,704,304,128 B planned.
pub const LOAD_RESIDUAL_BYTES: u64 = 88_028_656;

/// Device bytes a load holds beside its weights, line by line as the VRAM
/// plan lays them out (GitHub #210): planned before the load, read back off
/// the loaded model and pool after it, and compared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReservedBytes {
    /// The scratch arena prefill chunks and media encode share (GitHub #212).
    pub workspace: u64,
    pub media_embedding: u64,
    pub sampling: u64,
    pub decode_graph: u64,
    pub verify_round: u64,
    pub drafter_round: u64,
    /// Every lane's state in the sequence pool.
    pub lane_state: u64,
    /// The sequence pool's retained slots (GitHub #211).
    pub retained_slots: u64,
    /// The KV pool's arena, planes and block tables.
    pub kv_pool: u64,
}

/// A failure returned by the step ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// The leaf's integer return code.
    Leaf(i32),
}

/// Counters and geometry reported by the step runtime.
///
/// The scheduler consumes the page geometry for admission accounting; the
/// timing and dispatch counters feed the server/bench telemetry once P1-23's
/// FFI leaf exposes them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeStats {
    /// Bytes retained by the leaf for the loaded model and live sequences.
    pub vram_bytes: u64,
    /// Tokens held by one physical KV page.
    pub kv_page_tokens: u32,
    /// Bytes in one physical KV page.
    pub kv_page_bytes: u64,
    /// Physical KV pages the pool actually holds (the leaf's own build, not
    /// a requested budget; `ignis_core::kv::verified_kv_pool` cross-checks
    /// this against the scheduler's capacity).
    pub kv_page_count: u32,
    /// Duration of the most recent leaf step.
    pub last_step_micros: u64,
    /// Kernels dispatched by the most recent leaf step.
    pub kernel_count: u64,
    /// CUDA graph launches by the most recent leaf step.
    pub graph_launches: u64,
    /// Device bytes still free (GitHub #186): what `cudaMemGetInfo` reports
    /// once the model, its KV pool and every other reservation have landed.
    /// `0` when the backend cannot say — a CPU-only leaf, or a device query
    /// that failed.
    pub free_vram_bytes: u64,
    /// What the loaded model and its pool hold beside the weights (GitHub
    /// #210); all zero for a leaf with no device.
    pub reserved: ReservedBytes,
}

impl From<RuntimeError> for ComputeError {
    fn from(value: RuntimeError) -> Self {
        match value {
            RuntimeError::Leaf(code) => Self::Kernel(code),
        }
    }
}

/// One lane's inputs to a decode round (P5-06, GitHub #154).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeLane<'a> {
    /// The lane's sampling parameters.
    pub params: DecodeParams,
    /// Tokens the lane may commit this round, `>= 1`: the anchor plus the
    /// drafts a verify round may accept.
    pub remaining_tokens: u32,
    /// The committed run is cut at the first of these, inclusive: the
    /// model's EOS, or none for a lane that decodes past it.
    pub stop_ids: &'a [TokenId],
}

/// One lane's committed run from a decode round (P5-06, GitHub #154).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneRun {
    /// The committed tokens, in order.
    pub tokens: Vec<TokenId>,
    /// The round's speculative counters, when it was a verify round.
    pub spec: Option<SpecCounters>,
}

impl LaneRun {
    /// Today's round: one committed token.
    pub fn token(token: TokenId) -> Self {
        Self {
            tokens: vec![token],
            spec: None,
        }
    }
}

/// One prefill span's multimodal inputs (GitHub #178).
#[derive(Debug, Clone, Copy)]
pub struct MultimodalSpan<'a, M> {
    /// Axis-major `[3, tokens]` rope positions of the span.
    pub positions: &'a [i32],
    /// The sequence's rope delta: every decode round after the prompt
    /// rotates at `position + rope_delta`.
    pub rope_delta: i32,
    /// The media embedding the span's placeholder columns take, if the span
    /// covers any.
    pub media: Option<SpanMedia<'a, M>>,
}

/// The placeholder columns of one media item a prefill span covers.
#[derive(Debug, Clone, Copy)]
pub struct SpanMedia<'a, M> {
    /// The item's device-resident encoder output.
    pub embedding: &'a M,
    /// The embedding column the first covered placeholder takes.
    pub first_column: u32,
    /// Span-relative positions of the covered placeholders, ascending.
    pub scatter_indices: &'a [i32],
}

/// The replaceable step-ABI leaf seam.
///
/// The FFI implementation will map these calls to ADR 0009. Its opaque
/// handles stay inside the runtime; callers can only use the safe model and
/// compute adapter.
pub trait StepLeaf: Send + Sync + 'static {
    /// Opaque loaded-model handle.
    type Model: Send + Sync + 'static;
    /// Opaque device-resident sequence handle.
    type Sequence: Send + 'static;
    /// Opaque shared-prefix handle (P4-10, GitHub #126): leaf-owned KV pages
    /// several sequences address, plus the mutable state each claimant
    /// clones. Released when the scheduler's last claimant is gone.
    type Prefix: Send + 'static;
    /// Host-memory buffer a snapshot is captured into / restored from
    /// (P4-07, GitHub #125): pinned host memory in the production leaf
    /// (`CudaLeaf` — `ignis_core::seq::PinnedBuffer`, pinned being what
    /// makes the D2H/H2D crossing fast) and a plain `Vec<u8>` in a CPU-only
    /// stub, which never touches a real PCIe bus.
    type SnapshotBuf: AsRef<[u8]> + AsMut<[u8]> + Send + 'static;
    /// Opaque leaf-owned media embedding (GitHub #178): one media item's
    /// device-resident encoder output, live from its encode until the item's
    /// last placeholder is prefilled.
    type Media: Send + 'static;
    /// Opaque retained **prompt checkpoint** (GitHub #186, ADR 0029): one
    /// finished request's whole state at its generation opener, which a later
    /// request whose prompt extends it stands up on. Outlives the request
    /// that captured it — that is the point.
    type Checkpoint: Send + 'static;

    /// Load a model handle.
    fn load_model(&self) -> Result<Self::Model, i32>;
    /// Release a model handle.
    fn release_model(&self, model: Self::Model);
    /// Read the leaf's current geometry and step counters.
    fn stats(&self, model: &Self::Model) -> Result<RuntimeStats, i32>;
    /// Columns the loaded model's output head writes — the length a
    /// [`StepLeaf::prefill`] logits buffer must have (GitHub #237).
    ///
    /// The leaf's own number, and it is **not** the 151,936 ADR 0034
    /// quotes: that is Qwen2/Qwen3's vocabulary, and this artifact's runs to
    /// 248,320 (its `<|image_pad|>` alone sits at id 248,056). So a readout
    /// buffer is nearer 970 KB than the ADR's 607, and one sized from the
    /// wrong number would be a short write into the caller's memory.
    /// Required rather than defaulted for the same reason — every leaf
    /// knows this, and a default would be a wrong answer waiting to be
    /// believed.
    ///
    /// `model` is taken for symmetry with [`StepLeaf::stats`], not because
    /// today's leaf reads it: ignis is specialized for one topology
    /// (`CONTEXT.md`), so `CudaLeaf` answers from its `ModelConfig` and
    /// ignores the handle.
    fn vocab(&self, model: &Self::Model) -> u32;
    /// Allocate one sequence with its full context reservation.
    fn allocate_sequence(
        &self,
        model: &Self::Model,
        context_tokens: u32,
    ) -> Result<Self::Sequence, i32>;
    /// Release a sequence allocation.
    fn release_sequence(&self, model: &Self::Model, sequence: Self::Sequence);
    /// Allocate one sequence that **claims `prefix`** (P4-10, GitHub #126):
    /// the prefix's KV pages are shared in place and its mutable state is
    /// cloned device-to-device, so the sequence starts where the publisher
    /// stood and prefills only its own tail. `context_tokens` is the whole
    /// reservation, the prefix included.
    fn allocate_sequence_shared(
        &self,
        model: &Self::Model,
        context_tokens: u32,
        prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32>;
    /// Publish `sequence`'s first `prefix_tokens` tokens as a shared prefix,
    /// its mutable state in retained slot `retained_slot` (GitHub #215).
    /// Called at the chunk boundary that lands on `prefix_tokens`: the state
    /// a claimant clones is the state at the prefix's end.
    fn publish_prefix(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        prefix_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Prefix, i32>;
    /// Release the adapter's own handle on a prefix. Its pages return to the
    /// pool once every sequence holding it has gone too.
    fn release_prefix(&self, model: &Self::Model, prefix: Self::Prefix);

    // ── prompt checkpoints (GitHub #186, ADR 0029) ───────────────────────

    /// Capture `sequence`'s state at `opener_tokens` as a prompt checkpoint,
    /// its mutable state in retained slot `retained_slot` (GitHub #215).
    /// Called on the chunk boundary that lands on the generation opener, for
    /// the same reason [`StepLeaf::publish_prefix`] is called on its own: what
    /// a claimant receives is the state *there*. `sequence` is read and left
    /// exactly as it was, including on failure — a capture is a bet, and a
    /// lost bet costs the request nothing.
    fn capture_checkpoint(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        _opener_tokens: u32,
        _retained_slot: u32,
    ) -> Result<Self::Checkpoint, i32> {
        Err(-1)
    }
    /// Allocate one sequence that **claims `checkpoint`**: the whole pages
    /// below the opener are shared in place, the mutable state and the
    /// partial tail page are copied device-to-device, and the sequence stands
    /// at the opener. Never consumes the checkpoint. Returns the wall time the
    /// copies took, in microseconds — the `restore_ms` of the request log.
    fn allocate_sequence_from_checkpoint(
        &self,
        _model: &Self::Model,
        _context_tokens: u32,
        _checkpoint: &Self::Checkpoint,
    ) -> Result<(Self::Sequence, u64), i32> {
        Err(-1)
    }
    /// Release the adapter's handle on a checkpoint: its images go, and the
    /// pages under it return to the pool once nothing else holds them.
    fn release_checkpoint(&self, _model: &Self::Model, _checkpoint: Self::Checkpoint) {}
    /// Bytes a checkpoint occupies after its shared pages are materialized
    /// into a whole-sequence host blob.
    fn checkpoint_snapshot_bytes(
        &self,
        _model: &Self::Model,
        _checkpoint: &Self::Checkpoint,
    ) -> Result<u64, i32> {
        Err(-1)
    }
    /// Materialize a checkpoint into a host snapshot buffer.
    fn checkpoint_snapshot_into(
        &self,
        _model: &Self::Model,
        _checkpoint: &Self::Checkpoint,
        _dst: &mut [u8],
    ) -> Result<(), i32> {
        Err(-1)
    }
    /// Bytes a prefix occupies as a materialized whole-sequence host blob
    /// (GitHub #190).
    fn prefix_snapshot_bytes(&self, _model: &Self::Model, _prefix: &Self::Prefix) -> Result<u64, i32> {
        Err(-1)
    }
    /// Materialize a prefix into a host snapshot buffer, leaving it claimable.
    fn prefix_snapshot_into(
        &self,
        _model: &Self::Model,
        _prefix: &Self::Prefix,
        _dst: &mut [u8],
    ) -> Result<(), i32> {
        Err(-1)
    }
    /// The compatibility identity of the state this leaf produces (GitHub
    /// #189, ADR 0029): the artifact it loaded, its KV format, the blob
    /// layout version its sequence pool writes, and the drafter bound at
    /// load. Two of the four are the leaf's alone to know — which artifact it
    /// opened, and what version its own state-section table is at — which is
    /// why the identity is read from here and never assembled above it.
    ///
    /// [`BlobIdentity::UNSET`] from a leaf that retains nothing: it has no
    /// blobs to hand anybody, and an identity that matches no real load is
    /// the right answer for one.
    fn blob_identity(&self) -> BlobIdentity {
        BlobIdentity::UNSET
    }
    /// Warm one sequence with a prefill span.
    ///
    /// `out_logits`, when `Some`, is filled with the span's **last**
    /// position's full logits — the readout path (GitHub #237, ADR 0034).
    /// It must be [`StepLeaf::vocab`] entries long. No token is sampled for
    /// it and the sequence is left exactly as a `None` call would leave it:
    /// a readout observes the prefill, it does not change it.
    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        out_logits: Option<&mut [f32]>,
    ) -> Result<(), i32>;
    /// Encode one media item's patch rows into a device-resident embedding
    /// (the media encode step, GitHub #178). A leaf without vision refuses.
    fn encode_media(&self, _model: &Self::Model, _item: &MediaItem) -> Result<Self::Media, i32> {
        Err(-1)
    }
    /// Release a media embedding.
    fn release_media(&self, _model: &Self::Model, _media: Self::Media) {}
    /// [`StepLeaf::prefill`] over a span of a multimodal prompt: rotated at
    /// the span's three-axis positions, its placeholder columns taking the
    /// embedding's columns. A leaf without vision refuses.
    ///
    /// `out_logits` is [`StepLeaf::prefill`]'s, and is here for the same
    /// reason the text path has it: the evidence a decision is put to may
    /// be an image, and a readout wired only to the text path would return
    /// nothing at all for one rather than failing (GitHub #237).
    fn prefill_multimodal(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        _tokens: &[TokenId],
        _start_position: u32,
        _params: DecodeParams,
        _span: MultimodalSpan<'_, Self::Media>,
        _out_logits: Option<&mut [f32]>,
    ) -> Result<(), i32> {
        Err(-1)
    }
    /// Decode one round over a batch of warmed sequences, `lanes` parallel
    /// to `sequences`. Returns each lane's committed run (P5-06, GitHub
    /// #154): at least one token, never more than its `remaining_tokens`,
    /// cut at its first stop id inclusive — one token on a load without
    /// speculation. On an error, the leaf must leave every input sequence
    /// unchanged so the scheduler can retry the round without corrupting
    /// token order.
    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        lanes: &[DecodeLane<'_>],
    ) -> Result<Vec<LaneRun>, i32>;

    // ── state transfer (P4-07, GitHub #125, ADR 0024) ────────────────────

    /// Allocate a snapshot buffer of at least `bytes` (a span of the pinned
    /// KV-RAM arena in production, GitHub #213). `Err` with
    /// `ignis_core::seq::NO_HOST_ROOM` when the arena has no free span long
    /// enough, which [`StepLeaf::host_blob_fits`] is the way to ask first.
    fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32>;
    /// Whether [`StepLeaf::alloc_snapshot_buf`] would find room for `bytes`
    /// right now (GitHub #213).
    ///
    /// The host tier's byte ledger says whether the tier may hold the blob;
    /// this says whether the arena has anywhere to put it, which a full
    /// ledger's worth of free bytes scattered across holes does not. A
    /// backend whose buffers are ordinary allocations always fits.
    fn host_blob_fits(&self, bytes: u64) -> bool {
        let _ = bytes;
        true
    }
    /// Bytes a snapshot of `sequence` would need right now. `Err` with
    /// `ignis_core::seq::NOT_AT_BOUNDARY` while mid-chunk.
    fn snapshot_bytes(&self, model: &Self::Model, sequence: &Self::Sequence) -> Result<u64, i32>;
    /// Write `sequence`'s whole device state into `dst`, which must be at
    /// least [`StepLeaf::snapshot_bytes`] long. `sequence` is never
    /// modified.
    fn snapshot_into(
        &self,
        model: &Self::Model,
        sequence: &Self::Sequence,
        dst: &mut [u8],
    ) -> Result<(), i32>;
    /// Restore `sequence` (freshly drawn from [`StepLeaf::allocate_sequence`])
    /// from a blob [`StepLeaf::snapshot_into`] wrote. `Err` (e.g.
    /// `ignis_core::seq::BAD_SNAPSHOT`) leaves `sequence` untouched.
    fn restore_sequence(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        src: &[u8],
    ) -> Result<(), i32>;
}

/// A loaded model whose leaf handle is released exactly once on drop.
pub struct Model<L: StepLeaf> {
    leaf: Arc<L>,
    handle: Option<L::Model>,
}

impl<L: StepLeaf> Model<L> {
    /// Load the leaf model behind a safe, owning handle.
    pub fn load(leaf: Arc<L>) -> Result<Self, RuntimeError> {
        let handle = leaf.load_model().map_err(RuntimeError::Leaf)?;
        Ok(Self {
            leaf,
            handle: Some(handle),
        })
    }

    fn handle(&self) -> &L::Model {
        self.handle
            .as_ref()
            .expect("a live model always owns its leaf handle")
    }

    /// Read the loaded model's leaf statistics through the safe wrapper.
    pub fn stats(&self) -> Result<RuntimeStats, RuntimeError> {
        self.leaf.stats(self.handle()).map_err(RuntimeError::Leaf)
    }
}

impl<L: StepLeaf> Drop for Model<L> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.leaf.release_model(handle);
        }
    }
}

struct LiveSequence<S> {
    handle: S,
    generated: u32,
}

/// A request's live media embedding (GitHub #178): the item it encodes.
struct LiveMedia<M> {
    item: usize,
    handle: M,
}

/// A request evicted to the host tier (P4-07, GitHub #125): its snapshot
/// blob and the decode progress it resumes from, held here (not in
/// `ignis_core::host::HostTier`, which is pure CPU bookkeeping) because
/// only this side ever touches real device state.
struct EvictedSequence<B> {
    buf: B,
    generated: u32,
}

/// Scheduler adapter over a loaded step-ABI model.
///
/// A request gains a sequence on its first prefill. The map is private, so a
/// caller cannot decode without a sequence or forget its leaf release.
pub struct RuntimeCompute<L: StepLeaf> {
    model: Arc<Model<L>>,
    eos: TokenId,
    sequences: Mutex<HashMap<RequestId, LiveSequence<L::Sequence>>>,
    /// Shared prefixes (P4-10, GitHub #126), keyed by the request whose
    /// prefill published each. Separate from `sequences` on purpose: a
    /// prefix outlives its publisher, so its handle cannot hang off the
    /// publisher's sequence.
    ///
    /// Keyed by the head's length too: one request may publish its system
    /// block and then a chained head over it (#187 x #188), and a claimant of
    /// the block must be handed the block.
    prefixes: Mutex<HashMap<(RequestId, u32), L::Prefix>>,
    /// Retained prompt checkpoints (GitHub #186, ADR 0029), keyed by the
    /// request that captured each. That request is long finished; its id is
    /// never reused, so it stays a valid name for the bytes it left behind —
    /// the same identity `prefixes` above is keyed by, and for the same
    /// reason: the scheduler has to be able to name a device-resident thing
    /// in a job.
    checkpoints: Mutex<HashMap<RequestId, L::Checkpoint>>,
    /// Requests currently suspended in the host tier (P4-07, GitHub #125):
    /// their snapshot blob, held here until [`Compute::restore`] or
    /// [`Compute::discard_snapshot`] consumes it.
    evicted: Mutex<HashMap<RequestId, EvictedSequence<L::SnapshotBuf>>>,
    /// Materialized retained checkpoints in pinned KV-RAM. Unlike live
    /// evictions these are non-consuming: every matching request restores
    /// from the same immutable blob.
    retained: Mutex<HashMap<RequestId, L::SnapshotBuf>>,
    /// Materialized retained prefixes in pinned KV-RAM (GitHub #190), named
    /// as `prefixes` is. Non-consuming too: a prefix brought back onto the
    /// device keeps its blob, so giving it up again copies nothing.
    spilled_prefixes: Mutex<HashMap<(RequestId, u32), L::SnapshotBuf>>,
    /// Media embeddings still needed by a request's next chunk (GitHub
    /// #178): at most one per request, held from its item's first covered
    /// chunk until its last placeholder is prefilled.
    media: Mutex<HashMap<RequestId, LiveMedia<L::Media>>>,
}

impl<L: StepLeaf> RuntimeCompute<L> {
    /// Build an adapter for `model`; the server obtains `eos` from artifact
    /// generation defaults when it wires the real leaf.
    pub fn new(model: Arc<Model<L>>, eos: TokenId) -> Self {
        Self {
            model,
            eos,
            sequences: Mutex::new(HashMap::new()),
            prefixes: Mutex::new(HashMap::new()),
            checkpoints: Mutex::new(HashMap::new()),
            evicted: Mutex::new(HashMap::new()),
            retained: Mutex::new(HashMap::new()),
            spilled_prefixes: Mutex::new(HashMap::new()),
            media: Mutex::new(HashMap::new()),
        }
    }

    /// Number of live media embeddings (the CPU-stub observation point for
    /// their lifetime, GitHub #178).
    pub fn live_media(&self) -> usize {
        self.media.lock().unwrap().len()
    }

    fn release_media_handle(&self, media: L::Media) {
        self.model.leaf.release_media(self.model.handle(), media);
    }

    /// Release `request`'s live media embedding, if it holds one.
    fn release_media_of(&self, request: RequestId) {
        let media = self.media.lock().unwrap().remove(&request);
        if let Some(media) = media {
            self.release_media_handle(media.handle);
        }
    }

    /// One chunk of a multimodal prompt (GitHub #178): encode the media item
    /// the chunk covers unless its embedding is already live, prefill the
    /// span at its three-axis positions with the item's columns, and release
    /// the embedding once the chunk covered its last placeholder.
    ///
    /// Returns the microseconds the encode took (GitHub #192), 0 when this
    /// chunk encoded nothing — the same clock read `ConcreteScheduler` takes
    /// around `evict`, and taken unconditionally, so this path never varies
    /// with what an operator turned on.
    fn prefill_multimodal_job(
        &self,
        sequence: &mut L::Sequence,
        media: &mut HashMap<RequestId, LiveMedia<L::Media>>,
        job: &PrefillJob,
        multimodal: &Multimodal,
        out_logits: Option<&mut [f32]>,
    ) -> Result<u64, i32> {
        let (start, len) = (job.start_position, job.tokens.len() as u32);
        let chunk = multimodal.chunk_media(start, len);
        let mut encode_micros = 0;
        if let Some(chunk) = &chunk {
            if media.get(&job.request).is_some_and(|live| live.item != chunk.item) {
                let stale = media.remove(&job.request).expect("checked above");
                self.release_media_handle(stale.handle);
            }
            if !media.contains_key(&job.request) {
                let item = &multimodal.media[chunk.item];
                let _span = tracing::debug_span!(
                    "ignis.media.encode",
                    request_id = job.request,
                    item = chunk.item,
                    vision_tokens = item.grid.vision_tokens(),
                )
                .entered();
                let started = std::time::Instant::now();
                let handle = self.model.leaf.encode_media(self.model.handle(), item)?;
                encode_micros = started.elapsed().as_micros() as u64;
                media.insert(job.request, LiveMedia { item: chunk.item, handle });
            }
        }
        let positions = multimodal.span_positions(start as usize, len as usize);
        let span_media = chunk.as_ref().map(|chunk| SpanMedia {
            embedding: &media[&job.request].handle,
            first_column: chunk.first_column,
            scatter_indices: &chunk.scatter_indices,
        });
        self.model.leaf.prefill_multimodal(
            self.model.handle(),
            sequence,
            &job.tokens,
            start,
            job.params,
            MultimodalSpan {
                positions: &positions,
                rope_delta: multimodal.rope_delta,
                media: span_media,
            },
            out_logits,
        )?;
        if chunk.is_some_and(|chunk| chunk.completes_item)
            && let Some(done) = media.remove(&job.request)
        {
            self.release_media_handle(done.handle);
        }
        Ok(encode_micros)
    }

    /// Number of live leaf sequences (the CPU-stub observation point).
    pub fn live_sequences(&self) -> usize {
        self.sequences.lock().unwrap().len()
    }

    /// Number of shared prefixes this adapter holds a handle on (P4-10,
    /// GitHub #126) — the CPU-stub observation point for prefix lifetime.
    pub fn live_prefixes(&self) -> usize {
        self.prefixes.lock().unwrap().len()
    }

    /// Number of retained prompt checkpoints this adapter holds a handle on
    /// (GitHub #186) — the CPU-stub observation point for checkpoint
    /// lifetime, and the count a leaked device image would show up in.
    pub fn live_checkpoints(&self) -> usize {
        self.checkpoints.lock().unwrap().len()
    }

    /// Number of non-consuming materialized checkpoint blobs in KV-RAM.
    pub fn retained_checkpoints(&self) -> usize {
        self.retained.lock().unwrap().len()
    }

    /// Number of retained prefix blobs in KV-RAM (GitHub #190).
    pub fn spilled_prefixes(&self) -> usize {
        self.spilled_prefixes.lock().unwrap().len()
    }

    /// Number of requests currently suspended in the host tier (the
    /// CPU-stub observation point for eviction round trips).
    pub fn evicted_sequences(&self) -> usize {
        self.evicted.lock().unwrap().len()
    }

    fn release_sequence(&self, sequence: L::Sequence) {
        self.model
            .leaf
            .release_sequence(self.model.handle(), sequence);
    }

    fn release_prefix_handle(&self, prefix: L::Prefix) {
        self.model.leaf.release_prefix(self.model.handle(), prefix);
    }
}

impl<L: StepLeaf> Compute for RuntimeCompute<L> {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut outcomes = PrefillOutcome::nothing_encoded(jobs.len());
        let mut sequences = self.sequences.lock().unwrap();
        let mut prefixes = self.prefixes.lock().unwrap();
        let mut checkpoints = self.checkpoints.lock().unwrap();
        let retained = self.retained.lock().unwrap();
        let mut media = self.media.lock().unwrap();
        // Requests whose chunk lands on the prefix they publish. Collected
        // here and published after every job has warmed, so a later job's
        // failure cannot leave a published prefix behind with no sequence
        // and no scheduler entry to release it. Deferring is safe because a
        // job only ever touches its own sequence: nothing else in this batch
        // can move the publisher's state off the boundary.
        let mut to_publish: Vec<(RequestId, RetainedAt)> = Vec::new();
        // GitHub #186: and the ones whose chunk lands on their generation
        // opener. Deferred for the same reason, and taken *after* the
        // publishes: a checkpoint stands on the shared prefix below it, which
        // for a page-aligned opener is published in this very call.
        let mut to_capture: Vec<(RequestId, RetainedAt, usize)> = Vec::new();
        // Core retries a failed *batch*, not only the job that failed, so any
        // failure below returns every sequence in the batch to zero state --
        // otherwise the retry would prefill an already-warmed span twice.
        macro_rules! unwind {
            ($err:expr) => {{
                let released: Vec<_> = jobs
                    .iter()
                    .filter_map(|job| sequences.remove(&job.request))
                    .collect();
                let unpublished: Vec<_> = jobs
                    .iter()
                    .filter_map(|job| {
                        job.publish_prefix
                            .and_then(|publish| prefixes.remove(&(job.request, publish.tokens)))
                    })
                    .collect();
                // GitHub #178: the retry re-encodes whatever it needs.
                let unencoded: Vec<_> = jobs
                    .iter()
                    .filter_map(|job| media.remove(&job.request))
                    .collect();
                drop(media);
                drop(prefixes);
                drop(sequences);
                for live in unencoded {
                    self.release_media_handle(live.handle);
                }
                for sequence in released {
                    self.release_sequence(sequence.handle);
                }
                for prefix in unpublished {
                    self.release_prefix_handle(prefix);
                }
                return Err($err);
            }};
        }
        for (index, job) in jobs.iter().enumerate() {
            if !sequences.contains_key(&job.request) {
                // A claimant is allocated *against* the prefix: its leading
                // KV pages are the publisher's own, shared in place, and its
                // mutable state is cloned device-to-device (P4-10, GitHub
                // #126). Allocating it normally and then prefilling from
                // `start_position` would leave those leading pages zeroed —
                // the sequence would attend over history it does not have.
                // GitHub #186: a checkpoint claimant is the same idea one
                // step further — the whole pages below the generation opener
                // shared in place, the mutable state *and* the partial page
                // the opener ends inside copied, so the sequence stands at
                // the opener rather than at a page boundary.
                let allocated = match (&job.checkpoint, &job.shared_prefix) {
                    (Some(claim), _) if claim.source == ignis_core::checkpoint::ReuseSource::Device => {
                        match checkpoints.get(&claim.publisher) {
                            Some(checkpoint) => self
                                .model
                                .leaf
                                .allocate_sequence_from_checkpoint(
                                    self.model.handle(),
                                    job.context_tokens,
                                    checkpoint,
                                )
                                .map(|(handle, micros)| {
                                    outcomes[index].restore_micros = micros;
                                    handle
                                }),
                            None => Err(-1),
                        }
                    }
                    // GitHub #190: a KV-RAM claim restores the materialized
                    // blob into a fresh sequence, and leaves the blob in place.
                    (Some(claim), _) => match retained.get(&claim.publisher) {
                        Some(blob) => {
                            let started = std::time::Instant::now();
                            self.model
                                .leaf
                                .allocate_sequence(self.model.handle(), job.context_tokens)
                                .and_then(|mut handle| {
                                    match self.model.leaf.restore_sequence(
                                        self.model.handle(),
                                        &mut handle,
                                        blob.as_ref(),
                                    ) {
                                        Ok(()) => {
                                            outcomes[index].restore_micros =
                                                started.elapsed().as_micros() as u64;
                                            Ok(handle)
                                        }
                                        Err(code) => {
                                            self.model
                                                .leaf
                                                .release_sequence(self.model.handle(), handle);
                                            Err(code)
                                        }
                                    }
                                })
                        }
                        None => Err(-1),
                    },
                    (None, Some(claim)) => match prefixes.get(&(claim.publisher, claim.tokens)) {
                        Some(prefix) => self.model.leaf.allocate_sequence_shared(
                            self.model.handle(),
                            job.context_tokens,
                            prefix,
                        ),
                        // The scheduler holds a claim on an entry this
                        // adapter has no handle for. Nothing correct can be
                        // built from that, and prefilling the tail alone
                        // would answer from a hole, so it fails loudly.
                        None => Err(-1),
                    },
                    (None, None) => self
                        .model
                        .leaf
                        .allocate_sequence(self.model.handle(), job.context_tokens),
                };
                let handle = match allocated {
                    Ok(handle) => handle,
                    Err(code) => unwind!(RuntimeError::Leaf(code).into()),
                };
                sequences.insert(
                    job.request,
                    LiveSequence {
                        handle,
                        generated: 0,
                    },
                );
            }
            let sequence = sequences
                .get_mut(&job.request)
                .expect("sequence was inserted or already existed");
            // A full-prompt match carries no tail: the claim already put the
            // sequence where its prompt ends, with the pending token the
            // entry carried, so there is nothing left to warm. True of a
            // shared prefix, whose image is the publisher's state at the
            // prefix's end, and of a prompt checkpoint (GitHub #186), whose
            // progress section carries the pending token the capturing
            // sequence had at its opener.
            if !job.tokens.is_empty() {
                // GitHub #237: the full-vocabulary buffer, allocated only
                // for a job that asked to read one out — every other job
                // pays nothing, not an allocation and not a gather. It is
                // one f32 per output-head column (248,320 of them on the
                // 27B) and it dies at the end of this iteration: what
                // crosses the `Compute` seam is the gather below.
                let mut logits = job
                    .readout
                    .as_ref()
                    .map(|_| vec![0f32; self.model.leaf.vocab(self.model.handle()) as usize]);
                let warmed = match &job.multimodal {
                    None => self
                        .model
                        .leaf
                        .prefill(
                            self.model.handle(),
                            &mut sequence.handle,
                            &job.tokens,
                            job.start_position,
                            job.params,
                            logits.as_deref_mut(),
                        )
                        .map(|()| 0),
                    Some(multimodal) => self.prefill_multimodal_job(
                        &mut sequence.handle,
                        &mut media,
                        job,
                        multimodal,
                        logits.as_deref_mut(),
                    ),
                };
                match warmed {
                    Ok(encode_micros) => outcomes[index].encode_micros = encode_micros,
                    Err(code) => unwind!(RuntimeError::Leaf(code).into()),
                }
                if let (Some(answers), Some(logits)) = (&job.readout, &logits) {
                    outcomes[index].readout = Some(Readout::gather(logits, answers));
                }
            } else if job.readout.is_some() {
                // A chunk with nothing to prefill runs no forward pass, so
                // there are no logits at this position to read — an exact
                // repeat of a decision, whose reuse claim covered its whole
                // prompt, is the way to get here (GitHub #238 trims such a
                // claim by one token for exactly this reason). Failing
                // loudly rather than returning `None`: a readout that
                // silently did not happen is a decision answered by
                // whatever the caller does with an absent answer.
                // hotpath-lint-allow: failure-only path (the batch returns `Err` on the next line), reviewed exception (GitHub #237).
                tracing::error!(
                    name: "ignis.runtime.readout_without_tokens",
                    request_id = job.request,
                    start_position = job.start_position,
                    "a readout job whose chunk carries no tokens runs no forward pass"
                );
                unwind!(RuntimeError::Leaf(ignis_core::scheduler::READOUT_WITHOUT_TOKENS).into());
            }
            if let Some(publish) = job.publish_prefix {
                to_publish.push((job.request, publish));
            }
            if let Some(capture) = job.capture_checkpoint {
                to_capture.push((job.request, capture, index));
            }
        }
        // The publisher stands exactly on its prefix now, and the next chunk
        // it is dealt would move it off -- which is why the boundary is the
        // scheduler's decision and the publish is the last thing this call
        // does with the sequence.
        for (request, publish) in to_publish {
            let sequence = sequences
                .get_mut(&request)
                .expect("the publishing request's sequence was built above");
            match self.model.leaf.publish_prefix(
                self.model.handle(),
                &mut sequence.handle,
                publish.tokens,
                publish.slot,
            ) {
                Ok(prefix) => {
                    prefixes.insert((request, publish.tokens), prefix);
                }
                Err(code) => unwind!(RuntimeError::Leaf(code).into()),
            }
        }
        // GitHub #186 — the prompt checkpoints, last of all, and **never a
        // reason to fail the batch**. A capture is a bet on a request that
        // may never come; the request that paid for the chunk gets its chunk
        // either way, and the scheduler is told what actually happened so its
        // ledger and the device cannot disagree about what exists. The leaf
        // leaves the sequence untouched on a refusal, so there is nothing to
        // unwind.
        for (request, capture, index) in to_capture {
            let sequence = sequences
                .get_mut(&request)
                .expect("the capturing request's sequence was built above");
            match self.model.leaf.capture_checkpoint(
                self.model.handle(),
                &mut sequence.handle,
                capture.tokens,
                capture.slot,
            ) {
                Ok(checkpoint) => {
                    outcomes[index].checkpoint_captured = true;
                    if let Some(stale) = checkpoints.insert(request, checkpoint) {
                        // Unreachable: the scheduler captures at most one
                        // checkpoint per request. Released rather than
                        // dropped silently if that ever stops being true.
                        drop(stale);
                    }
                }
                Err(_) => outcomes[index].checkpoint_captured = false,
            }
        }
        Ok(outcomes)
    }

    fn blob_identity(&self) -> BlobIdentity {
        self.model.leaf.blob_identity()
    }

    fn release_checkpoint(&self, publisher: RequestId) {
        let checkpoint = self.checkpoints.lock().unwrap().remove(&publisher);
        if let Some(checkpoint) = checkpoint {
            self.model
                .leaf
                .release_checkpoint(self.model.handle(), checkpoint);
        }
        self.retained.lock().unwrap().remove(&publisher);
    }

    fn checkpoint_snapshot_size(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        let checkpoints = self.checkpoints.lock().unwrap();
        let checkpoint = checkpoints.get(&publisher).ok_or(ComputeError::Kernel(-1))?;
        self.model
            .leaf
            .checkpoint_snapshot_bytes(self.model.handle(), checkpoint)
            .map_err(|code| RuntimeError::Leaf(code).into())
    }

    fn spill_checkpoint(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        let mut checkpoints = self.checkpoints.lock().unwrap();
        let checkpoint = checkpoints.get(&publisher).ok_or(ComputeError::Kernel(-1))?;
        // Every step before the release leaves the device image where it was,
        // so a failed spill is one the caller can still discard normally.
        let leaf = &self.model.leaf;
        let buf = leaf
            .checkpoint_snapshot_bytes(self.model.handle(), checkpoint)
            .and_then(|bytes| Ok((bytes, leaf.alloc_snapshot_buf(bytes)?)))
            .and_then(|(bytes, mut buf)| {
                leaf.checkpoint_snapshot_into(self.model.handle(), checkpoint, buf.as_mut())?;
                Ok((bytes, buf))
            });
        let (bytes, buf) = buf.map_err(RuntimeError::Leaf)?;
        let checkpoint = checkpoints.remove(&publisher).expect("looked up above");
        leaf.release_checkpoint(self.model.handle(), checkpoint);
        drop(checkpoints);
        self.retained.lock().unwrap().insert(publisher, buf);
        Ok(bytes)
    }

    fn prefix_snapshot_size(&self, publisher: RequestId, tokens: u32) -> Result<u64, ComputeError> {
        let prefixes = self.prefixes.lock().unwrap();
        let prefix = prefixes.get(&(publisher, tokens)).ok_or(ComputeError::Kernel(-1))?;
        self.model
            .leaf
            .prefix_snapshot_bytes(self.model.handle(), prefix)
            .map_err(|code| RuntimeError::Leaf(code).into())
    }

    fn spill_prefix(&self, publisher: RequestId, tokens: u32) -> Result<u64, ComputeError> {
        let prefixes = self.prefixes.lock().unwrap();
        let prefix = prefixes.get(&(publisher, tokens)).ok_or(ComputeError::Kernel(-1))?;
        let leaf = &self.model.leaf;
        let (bytes, buf) = leaf
            .prefix_snapshot_bytes(self.model.handle(), prefix)
            .and_then(|bytes| Ok((bytes, leaf.alloc_snapshot_buf(bytes)?)))
            .and_then(|(bytes, mut buf)| {
                leaf.prefix_snapshot_into(self.model.handle(), prefix, buf.as_mut())?;
                Ok((bytes, buf))
            })
            .map_err(RuntimeError::Leaf)?;
        drop(prefixes);
        self.spilled_prefixes.lock().unwrap().insert((publisher, tokens), buf);
        Ok(bytes)
    }

    fn restore_prefix(&self, publisher: RequestId, tokens: u32, slot: u32) -> Result<u64, ComputeError> {
        let key = (publisher, tokens);
        let spilled = self.spilled_prefixes.lock().unwrap();
        let blob = spilled.get(&key).ok_or(ComputeError::Kernel(-1))?;
        let mut prefixes = self.prefixes.lock().unwrap();
        if prefixes.contains_key(&key) {
            return Err(ComputeError::Kernel(-1));
        }
        let leaf = &self.model.leaf;
        let started = std::time::Instant::now();
        // A carrier sequence one page longer than the head, so publishing
        // leaves it a page of its own; it goes as soon as the prefix exists,
        // and the prefix keeps the pages.
        let mut carrier = leaf
            .allocate_sequence(self.model.handle(), tokens + KV_PAGE_TOKENS)
            .map_err(RuntimeError::Leaf)?;
        let published = leaf
            .restore_sequence(self.model.handle(), &mut carrier, blob.as_ref())
            .and_then(|()| leaf.publish_prefix(self.model.handle(), &mut carrier, tokens, slot));
        leaf.release_sequence(self.model.handle(), carrier);
        let prefix = published.map_err(RuntimeError::Leaf)?;
        prefixes.insert(key, prefix);
        Ok(started.elapsed().as_micros() as u64)
    }

    fn discard_spilled_prefix(&self, publisher: RequestId, tokens: u32) {
        self.spilled_prefixes.lock().unwrap().remove(&(publisher, tokens));
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        if jobs.len() > N_DECODE_LANES {
            return Err(ComputeError::Kernel(-1));
        }
        let mut sequences = self.sequences.lock().unwrap();
        let mut batch: Vec<(DecodeJob, LiveSequence<L::Sequence>)> = Vec::with_capacity(jobs.len());
        for job in jobs {
            let Some(sequence) = sequences.remove(&job.request) else {
                for (job, sequence) in batch {
                    sequences.insert(job.request, sequence);
                }
                return Err(ComputeError::Kernel(-1));
            };
            batch.push((job.clone(), sequence));
        }

        // `None` until filled below; every index is either already-finished
        // (a prior round's `max_tokens`, `Length`), gets a fresh run, or
        // finishes this round (EOS, `Stop`) — so every slot is set once.
        let mut outcomes: Vec<Option<DecodeOutcome>> = vec![None; jobs.len()];
        let mut active = vec![false; jobs.len()];
        for (index, (job, sequence)) in batch.iter().enumerate() {
            if job
                .params
                .max_tokens
                .is_some_and(|max| sequence.generated >= max)
            {
                outcomes[index] = Some(DecodeOutcome::finished(FinishReason::Length));
            } else {
                active[index] = true;
            }
        }
        // P5-06 (GitHub #154): the leaf cuts each lane's run at its budget
        // and at the EOS, so the sequence never commits past the text its
        // request emits.
        let eos = [self.eos];
        let decoded = {
            let lanes: Vec<DecodeLane<'_>> = batch
                .iter()
                .enumerate()
                .filter(|(index, _)| active[*index])
                .map(|(_, (job, sequence))| DecodeLane {
                    params: job.params,
                    remaining_tokens: job
                        .params
                        .max_tokens
                        .map_or(job.remaining_tokens, |max| {
                            job.remaining_tokens.min(max.saturating_sub(sequence.generated))
                        })
                        .max(1),
                    stop_ids: if job.params.ignore_eos { &[] } else { &eos },
                })
                .collect();
            let mut handles: Vec<&mut L::Sequence> = batch
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| active[*index])
                .map(|(_, (_, sequence))| &mut sequence.handle)
                .collect();
            // Every job was already capped: nothing for the leaf to run, and
            // it refuses an empty batch.
            if lanes.is_empty() {
                Ok(Vec::new())
            } else {
                self.model
                    .leaf
                    .decode(self.model.handle(), &mut handles, &lanes)
            }
        };
        let decoded = match decoded {
            Ok(runs)
                if runs.len() == active.iter().filter(|&&active| active).count()
                    && runs.iter().all(|run| !run.tokens.is_empty()) =>
            {
                runs
            }
            Ok(_) => {
                for (job, sequence) in batch {
                    sequences.insert(job.request, sequence);
                }
                return Err(ComputeError::Kernel(-1));
            }
            Err(code) => {
                for (job, sequence) in batch {
                    sequences.insert(job.request, sequence);
                }
                return Err(RuntimeError::Leaf(code).into());
            }
        };

        let mut decoded = decoded.into_iter();
        let mut released = Vec::new();
        for (index, (job, mut sequence)) in batch.into_iter().enumerate() {
            if !active[index] {
                released.push(sequence);
                continue;
            }
            let LaneRun { mut tokens, spec } = decoded.next().expect("decoded result length was checked");
            let eos_at = (!job.params.ignore_eos)
                .then(|| tokens.iter().position(|&token| token == self.eos))
                .flatten();
            let outcome = match eos_at {
                // The EOS itself is never emitted.
                Some(at) => {
                    tokens.truncate(at);
                    released.push(sequence);
                    DecodeOutcome::run_then_finished(tokens, FinishReason::Stop)
                }
                None => {
                    sequence.generated = sequence.generated.saturating_add(tokens.len() as u32);
                    sequences.insert(job.request, sequence);
                    DecodeOutcome::run(tokens)
                }
            };
            outcomes[index] = Some(DecodeOutcome { spec, ..outcome });
        }
        drop(sequences);
        for sequence in released {
            self.release_sequence(sequence.handle);
        }
        Ok(outcomes
            .into_iter()
            .map(|o| o.expect("every job index is filled by one of the branches above"))
            .collect())
    }

    fn release(&self, request: RequestId) {
        // GitHub #178: a request completed or cancelled mid-item releases
        // the item's embedding with its sequence.
        self.release_media_of(request);
        let sequence = self.sequences.lock().unwrap().remove(&request);
        if let Some(sequence) = sequence {
            self.release_sequence(sequence.handle);
        }
    }

    fn release_prefix(&self, publisher: RequestId, tokens: u32) {
        // Only this adapter's handle. The leaf's pages come back when every
        // sequence still holding the prefix has been released too, which is
        // what lets a publisher finish while its claimants keep serving.
        let prefix = self.prefixes.lock().unwrap().remove(&(publisher, tokens));
        if let Some(prefix) = prefix {
            self.release_prefix_handle(prefix);
        }
    }

    fn snapshot_size(&self, request: RequestId) -> Result<u64, ComputeError> {
        let sequences = self.sequences.lock().unwrap();
        let Some(live) = sequences.get(&request) else {
            return Err(ComputeError::Kernel(-1));
        };
        self.model
            .leaf
            .snapshot_bytes(self.model.handle(), &live.handle)
            .map_err(|code| RuntimeError::Leaf(code).into())
    }

    fn host_blob_fits(&self, bytes: u64) -> bool {
        self.model.leaf.host_blob_fits(bytes)
    }

    fn evict(&self, request: RequestId) -> Result<u64, ComputeError> {
        let mut sequences = self.sequences.lock().unwrap();
        let Some(live) = sequences.remove(&request) else {
            return Err(ComputeError::Kernel(-1));
        };
        let bytes = match self.model.leaf.snapshot_bytes(self.model.handle(), &live.handle) {
            Ok(bytes) => bytes,
            Err(code) => {
                sequences.insert(request, live);
                return Err(RuntimeError::Leaf(code).into());
            }
        };
        let mut buf = match self.model.leaf.alloc_snapshot_buf(bytes) {
            Ok(buf) => buf,
            Err(code) => {
                sequences.insert(request, live);
                return Err(RuntimeError::Leaf(code).into());
            }
        };
        if let Err(code) =
            self.model
                .leaf
                .snapshot_into(self.model.handle(), &live.handle, buf.as_mut())
        {
            sequences.insert(request, live);
            return Err(RuntimeError::Leaf(code).into());
        }
        drop(sequences);
        self.release_sequence(live.handle);
        // GitHub #194: vision state is not part of the blob. An item evicted
        // part-way gives its embedding back — the load reserves room for
        // one — and the chunk that continues it after restore encodes it
        // again.
        self.release_media_of(request);
        self.evicted.lock().unwrap().insert(
            request,
            EvictedSequence {
                buf,
                generated: live.generated,
            },
        );
        Ok(bytes)
    }

    fn restore(&self, request: RequestId, context_tokens: u32) -> Result<(), ComputeError> {
        let Some(evicted) = self.evicted.lock().unwrap().remove(&request) else {
            return Err(ComputeError::Kernel(-1));
        };
        let mut handle = match self
            .model
            .leaf
            .allocate_sequence(self.model.handle(), context_tokens)
        {
            Ok(handle) => handle,
            Err(code) => {
                self.evicted.lock().unwrap().insert(request, evicted);
                return Err(RuntimeError::Leaf(code).into());
            }
        };
        if let Err(code) =
            self.model
                .leaf
                .restore_sequence(self.model.handle(), &mut handle, evicted.buf.as_ref())
        {
            // A refused restore (e.g. a stale/foreign blob) leaves `handle`
            // untouched but unusable for this request — nothing to resume
            // into. Release it and surface the failure; the scheduler falls
            // back to discarding the (already-consumed) snapshot and
            // re-prefilling.
            self.model.leaf.release_sequence(self.model.handle(), handle);
            return Err(RuntimeError::Leaf(code).into());
        }
        self.sequences.lock().unwrap().insert(
            request,
            LiveSequence {
                handle,
                generated: evicted.generated,
            },
        );
        Ok(())
    }

    fn discard_snapshot(&self, request: RequestId) {
        // Dropping the entry frees its buffer (`PinnedBuffer::drop` calls
        // `ignis_host_pinned_free` in production).
        self.evicted.lock().unwrap().remove(&request);
    }
}

impl<L: StepLeaf> Drop for RuntimeCompute<L> {
    fn drop(&mut self) {
        let media = std::mem::take(
            self.media
                .get_mut()
                .expect("RuntimeCompute is not dropped while its media lock is held"),
        );
        for (_, media) in media {
            self.release_media_handle(media.handle);
        }
        let sequences = std::mem::take(
            self.sequences
                .get_mut()
                .expect("RuntimeCompute is not dropped while its sequence lock is held"),
        );
        for (_, sequence) in sequences {
            self.release_sequence(sequence.handle);
        }
        // Checkpoints before prefixes, for the same reason prefixes come
        // after sequences: a checkpoint holds one reference to the prefix
        // under it (GitHub #186), so releasing it first leaves the prefix
        // loop below to drop the last holder rather than a middle one.
        let checkpoints = std::mem::take(
            self.checkpoints
                .get_mut()
                .expect("RuntimeCompute is not dropped while its checkpoint lock is held"),
        );
        for (_, checkpoint) in checkpoints {
            self.model
                .leaf
                .release_checkpoint(self.model.handle(), checkpoint);
        }
        // Every snapshot blob, freed here rather than left to the fields'
        // own drop. They own only host memory — there is no leaf device
        // handle left after a spill — but since GitHub #213 that memory is a
        // span of the leaf's KV-RAM arena, and `model` is this struct's
        // *first* field, so it would let go of the leaf (and the arena with
        // it) before any of these dropped. Freeing them while the leaf is
        // still here is what keeps that a return rather than a dangling one.
        self.evicted
            .get_mut()
            .expect("RuntimeCompute is not dropped while its evicted lock is held")
            .clear();
        self.retained
            .get_mut()
            .expect("RuntimeCompute is not dropped while its retained lock is held")
            .clear();
        self.spilled_prefixes
            .get_mut()
            .expect("RuntimeCompute is not dropped while its spilled-prefix lock is held")
            .clear();
        // Prefixes after sequences: a prefix's pages are released by its last
        // holder, and a live sequence is one.
        let prefixes = std::mem::take(
            self.prefixes
                .get_mut()
                .expect("RuntimeCompute is not dropped while its prefix lock is held"),
        );
        for (_, prefix) in prefixes {
            self.release_prefix_handle(prefix);
        }
    }
}
