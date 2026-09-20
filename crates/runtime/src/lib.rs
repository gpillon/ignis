//! Safe ownership of the step ABI and its [`ignis_core::Compute`] adapter.
//!
//! The runtime owns a loaded model, one opaque sequence per scheduler
//! request, integer error-code mapping, and the sequence-release lifecycle.
//! The C ABI adapter lands with P1-23; the small [`StepLeaf`] seam lets this
//! ownership logic be tested today against a CPU stub.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ignis_core::vision::{MEDIA_ENCODE_POOL_FULL, MediaItem, Multimodal};
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
    /// The ids this lane's **draw** is restricted to (GitHub #242), or
    /// empty for an ordinary round.
    ///
    /// It constrains the token the lane draws *this* round, which the leaf
    /// returns on the *next* one
    /// ([`ignis_core::scheduler::DecodeJob::permitted`]). A constrained lane
    /// commits exactly one token: the drafts a verify round would accept are
    /// proposed by a second model that knows nothing of a set, so a round
    /// with any constrained lane in it is run as a plain round.
    pub permitted: &'a [TokenId],
}

/// One lane's committed run from a decode round (P5-06, GitHub #154).
///
/// `Eq` is deliberately absent since GitHub #242: `drawn_probability` is an
/// `f32`, and a run is compared for equality in tests and nowhere else.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneRun {
    /// The committed tokens, in order.
    pub tokens: Vec<TokenId>,
    /// The round's speculative counters, when it was a verify round.
    pub spec: Option<SpecCounters>,
    /// The probability of the token this round **drew** within its
    /// [`DecodeLane::permitted`] set (GitHub #242), or `None` for an
    /// unconstrained lane.
    ///
    /// It belongs to the token the *next* round returns, not to anything in
    /// `tokens`. The leaf reports it here because by the time that token is
    /// emitted the sequence has moved past the position it was drawn at, and
    /// there is nothing left to recompute it from;
    /// [`RuntimeCompute`] is what holds it for the one round in between.
    pub drawn_probability: Option<f32>,
}

impl LaneRun {
    /// Today's round: one committed token.
    pub fn token(token: TokenId) -> Self {
        Self {
            tokens: vec![token],
            spec: None,
            drawn_probability: None,
        }
    }

    /// One committed token, and the probability of the one this round drew
    /// for the next (GitHub #242).
    pub fn drawn(token: TokenId, probability: Option<f32>) -> Self {
        Self {
            drawn_probability: probability,
            ..Self::token(token)
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
    ///
    /// `permitted`, when non-empty, restricts the draw this span makes
    /// (GitHub #242) — the successor the first decode round returns — and
    /// the call reports its probability within that set. Empty is an
    /// ordinary prefill and reports 0.
    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        out_logits: Option<&mut [f32]>,
    ) -> Result<f32, i32>;
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
        _permitted: &[TokenId],
        _span: MultimodalSpan<'_, Self::Media>,
        _out_logits: Option<&mut [f32]>,
    ) -> Result<f32, i32> {
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

/// What decides an encoder run's output, and so what two requests have to
/// share to share its result (GitHub #243).
///
/// Deliberately *not* [`ignis_core::identity::MediaKey`], which also carries
/// where the item sits in the prompt. Two siblings of a fan-out ask different
/// questions about one picture; a question that comes before the image moves
/// its placeholders. The encoder sees neither — it sees these bytes at this
/// grid — so these two fields are the identity and the offset is not.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct MediaEmbeddingKey {
    digest: [u8; 32],
    grid: [u32; 3],
}

impl From<&MediaItem> for MediaEmbeddingKey {
    fn from(item: &MediaItem) -> Self {
        Self {
            digest: item.content_digest,
            grid: [item.grid.t, item.grid.h, item.grid.w],
        }
    }
}

/// One encoded item the cache holds (GitHub #243).
struct CachedMedia<M> {
    handle: M,
    /// Requests whose current prefill chunk covers this item. Zero is an
    /// entry nobody is using *right now* — which is not an entry to drop,
    /// it is the whole point: the next question about the same picture is
    /// the one that finds it.
    holders: usize,
    /// The cache clock when the last holder let go, for LRU among unheld
    /// entries. Meaningless while `holders > 0`.
    released_at: u64,
    /// The item's merged columns, for the resident-bytes fact (GitHub #216).
    columns: u64,
}

/// The media embedding cache (GitHub #243): encoded items, kept past the
/// request that encoded them.
///
/// The miss this closes is a fan-out — N questions over one image. It cannot
/// be closed by reference counting alone, because the siblings never overlap:
/// exactly one request holds multi-tick prefill progress at a time
/// (`ignis_core::concrete`), and the embedding is given up at its item's last
/// placeholder, before the sibling's first chunk runs. So an entry outliving
/// its last holder *is* the mechanism, not a tuning knob on top of one.
///
/// Bounded by the leaf's pool and nothing here: this side never counts bytes
/// or pages. It asks the leaf to encode, and a leaf that answers
/// [`MEDIA_ENCODE_POOL_FULL`] gets an unheld entry released and the same
/// question again ([`RuntimeCompute::encode_into`]). Policy here, bytes
/// there.
struct MediaCache<M> {
    entries: HashMap<MediaEmbeddingKey, CachedMedia<M>>,
    /// What each request is holding. At most one entry per request: a prefill
    /// chunk covers at most one media item.
    held: HashMap<RequestId, MediaEmbeddingKey>,
    clock: u64,
}

impl<M> MediaCache<M> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            held: HashMap::new(),
            clock: 0,
        }
    }

    /// Entries a request is prefilling against right now.
    fn live(&self) -> usize {
        self.entries.values().filter(|e| e.holders > 0).count()
    }

    /// Take a hold on `key` for `request`, which must be in `entries`.
    fn acquire(&mut self, request: RequestId, key: MediaEmbeddingKey) {
        if self.held.get(&request) == Some(&key) {
            return;
        }
        self.release(request);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.holders += 1;
            self.held.insert(request, key);
        }
    }

    /// Drop `request`'s hold, if it has one. The entry stays.
    fn release(&mut self, request: RequestId) {
        let Some(key) = self.held.remove(&request) else {
            return;
        };
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.holders = entry.holders.saturating_sub(1);
            if entry.holders == 0 {
                entry.released_at = self.clock;
            }
        }
    }

    /// Give up the least recently released unheld entry, or `None` when
    /// every entry is held.
    fn evict_lru(&mut self) -> Option<M> {
        let victim = self
            .entries
            .iter()
            .filter(|(_, e)| e.holders == 0)
            .min_by_key(|(_, e)| e.released_at)
            .map(|(key, _)| *key)?;
        self.entries.remove(&victim).map(|e| e.handle)
    }
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
    /// Encoded media items (GitHub #178, cached across requests by GitHub
    /// #243): held from an item's first covered chunk until the leaf's pool
    /// needs the room, so the next question about the same picture does not
    /// run the tower again.
    media: Mutex<MediaCache<L::Media>>,
    /// The probability of the token a **constrained decode** request has drawn and not
    /// yet emitted (GitHub #242), per request.
    ///
    /// This is the one-round lag, held on this side of the `Compute` seam
    /// because this is the only side that can hold it. The leaf draws a
    /// token at the end of one call and returns it at the start of the
    /// next, and its probability within the permitted set is computed from
    /// at most 32 logits that exist only at the moment of the draw. By the
    /// round that emits the token those logits are gone, so the number is
    /// carried across here — one `f32` per program in flight — rather than
    /// recomputed or reported a round early
    /// ([`ignis_core::scheduler::DecodeJob::permitted`]).
    drawn: Mutex<HashMap<RequestId, f32>>,
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
            media: Mutex::new(MediaCache::new()),
            drawn: Mutex::new(HashMap::new()),
        }
    }

    /// Media embeddings a request is prefilling against right now (the
    /// CPU-stub observation point for their lifetime, GitHub #178).
    ///
    /// Not the cache's size: since GitHub #243 an embedding outlives the
    /// request that encoded it, and this counts holders, not residents. A
    /// fan-out still brings it back to zero — see [`Self::cached_media`] for
    /// what is left behind.
    pub fn live_media(&self) -> usize {
        self.media.lock().unwrap().live()
    }

    /// Encoded items the cache holds, held or not (GitHub #243).
    pub fn cached_media(&self) -> usize {
        self.media.lock().unwrap().entries.len()
    }

    /// The merged columns the cache's entries occupy (GitHub #243) — the
    /// shape of what it costs, in the one unit both sides of the seam agree
    /// on. The bytes are the leaf's: a column is `hidden x 2`.
    pub fn cached_media_columns(&self) -> u64 {
        self.media.lock().unwrap().entries.values().map(|e| e.columns).sum()
    }

    fn release_media_handle(&self, media: L::Media) {
        self.model.leaf.release_media(self.model.handle(), media);
    }

    /// Drop `request`'s hold on its embedding, if it has one.
    ///
    /// The embedding stays cached (GitHub #243). A request finishing,
    /// failing or being evicted is exactly the moment a sibling asking the
    /// next question about the same picture is about to want it.
    fn release_media_of(&self, request: RequestId) {
        self.media.lock().unwrap().release(request);
    }

    /// Encode `item` under `key`, making room by releasing unheld entries
    /// until the leaf's pool takes it (GitHub #243).
    ///
    /// The eviction policy is here and the bytes are the leaf's: it answers
    /// [`MEDIA_ENCODE_POOL_FULL`] and this decides what to give up. The loop
    /// terminates because the load floors the pool at one envelope-wide item
    /// and a chunk covers one item, so releasing every unheld entry leaves
    /// room for the one item being prefilled right now. It cannot spin: each
    /// turn removes an entry, and a turn with nothing left to remove returns
    /// the refusal.
    fn encode_into(
        &self,
        media: &mut MediaCache<L::Media>,
        item: &MediaItem,
        key: MediaEmbeddingKey,
    ) -> Result<(), i32> {
        loop {
            match self.model.leaf.encode_media(self.model.handle(), item) {
                Ok(handle) => {
                    media.entries.insert(
                        key,
                        CachedMedia {
                            handle,
                            holders: 0,
                            released_at: media.clock,
                            columns: item.grid.vision_tokens(),
                        },
                    );
                    return Ok(());
                }
                Err(MEDIA_ENCODE_POOL_FULL) => match media.evict_lru() {
                    Some(handle) => self.release_media_handle(handle),
                    None => return Err(MEDIA_ENCODE_POOL_FULL),
                },
                Err(code) => return Err(code),
            }
        }
    }

    /// One chunk of a multimodal prompt (GitHub #178): encode the media item
    /// the chunk covers unless the cache already holds it, prefill the span
    /// at its three-axis positions with the item's columns, and let the
    /// embedding go once the chunk covered its last placeholder.
    ///
    /// "Let go" is a hold, not a free, since GitHub #243: the entry stays in
    /// the cache for the next request that names the same picture, and only
    /// the leaf's pool running out takes it away.
    ///
    /// Returns the microseconds the encode took (GitHub #192), 0 when this
    /// chunk encoded nothing — which is now the common case in a fan-out, and
    /// exactly what acceptance 1 reads. The same clock read
    /// `ConcreteScheduler` takes around `evict`, and taken unconditionally,
    /// so this path never varies with what an operator turned on.
    fn prefill_multimodal_job(
        &self,
        sequence: &mut L::Sequence,
        media: &mut MediaCache<L::Media>,
        job: &PrefillJob,
        multimodal: &Multimodal,
        out_logits: Option<&mut [f32]>,
    ) -> Result<(u64, f32), i32> {
        let (start, len) = (job.start_position, job.tokens.len() as u32);
        let chunk = multimodal.chunk_media(start, len);
        let mut encode_micros = 0;
        let mut held = None;
        if let Some(chunk) = &chunk {
            let item = &multimodal.media[chunk.item];
            let key = MediaEmbeddingKey::from(item);
            if !media.entries.contains_key(&key) {
                let _span = tracing::debug_span!(
                    "ignis.media.encode",
                    request_id = job.request,
                    item = chunk.item,
                    vision_tokens = item.grid.vision_tokens(),
                )
                .entered();
                let started = std::time::Instant::now();
                // A hold this request already has is not in the way: the
                // cache takes the new one only after dropping the old, and
                // an eviction inside the encode may need the old one's room.
                media.release(job.request);
                self.encode_into(media, item, key)?;
                encode_micros = started.elapsed().as_micros() as u64;
            }
            media.acquire(job.request, key);
            held = Some(key);
        }
        let positions = multimodal.span_positions(start as usize, len as usize);
        let span_media = chunk.as_ref().map(|chunk| SpanMedia {
            embedding: &media.entries[&held.expect("a covered chunk took a hold")].handle,
            first_column: chunk.first_column,
            scatter_indices: &chunk.scatter_indices,
        });
        let permitted = job.permitted.clone().unwrap_or_else(|| Vec::new().into());
        let probability = self.model.leaf.prefill_multimodal(
            self.model.handle(),
            sequence,
            &job.tokens,
            start,
            job.params,
            &permitted,
            MultimodalSpan {
                positions: &positions,
                rope_delta: multimodal.rope_delta,
                media: span_media,
            },
            out_logits,
        )?;
        if chunk.is_some_and(|chunk| chunk.completes_item) {
            media.release(job.request);
        }
        Ok((encode_micros, probability))
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
                // GitHub #178: every job in the batch gives its embedding
                // back. GitHub #243: gives back its *hold* — the encode
                // succeeded, the item has not changed, and the retry that
                // follows finds it in the cache instead of running the tower
                // a second time.
                for job in jobs {
                    media.release(job.request);
                }
                drop(media);
                drop(prefixes);
                drop(sequences);
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
                            // GitHub #242: the set this chunk's own draw is
                            // restricted to, borrowed and not cloned — every
                            // chunk that is not a constrained run's last
                            // passes an empty slice and allocates nothing.
                            job.permitted.as_deref().unwrap_or(&[]),
                            logits.as_deref_mut(),
                        )
                        .map(|probability| (0, probability)),
                    Some(multimodal) => self.prefill_multimodal_job(
                        &mut sequence.handle,
                        &mut media,
                        job,
                        multimodal,
                        logits.as_deref_mut(),
                    ),
                };
                match warmed {
                    Ok((encode_micros, probability)) => {
                        outcomes[index].encode_micros = encode_micros;
                        // The first token of a run's run was just drawn
                        // here; the first decode round returns it, and this
                        // is the only place its probability exists.
                        if job.permitted.is_some() {
                            self.drawn.lock().unwrap().insert(job.request, probability);
                        }
                    }
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
        // GitHub #242: cloned out of `batch` before the lanes borrow it,
        // because the handles below take it mutably. An `Option<Arc>` clone
        // is a refcount bump for a constrained lane and nothing at all for
        // any other — an ordinary round allocates no more than it did before
        // this ticket, which is the bar for anything on this path.
        let permitted: Vec<Option<ignis_core::constrained::PermittedSet>> = batch
            .iter()
            .map(|(job, _)| job.permitted.clone())
            .collect();
        let decoded = {
            let lanes: Vec<DecodeLane<'_>> = batch
                .iter()
                .enumerate()
                .filter(|(index, _)| active[*index])
                .map(|(index, (job, sequence))| DecodeLane {
                    params: job.params,
                    remaining_tokens: job
                        .params
                        .max_tokens
                        .map_or(job.remaining_tokens, |max| {
                            job.remaining_tokens.min(max.saturating_sub(sequence.generated))
                        })
                        .max(1),
                    stop_ids: if job.params.ignore_eos { &[] } else { &eos },
                    permitted: permitted[index].as_deref().unwrap_or(&[]),
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
        // GitHub #242: taken once for the whole round rather than per lane.
        // The map is empty unless a constrained run is in flight, and this
        // is the decode path — one uncontended lock per round is a cost an
        // ordinary round can carry; eight of them would be a cost this
        // ticket added to every request the engine serves.
        let mut drawn_probabilities = self.drawn.lock().unwrap();
        for (index, (job, mut sequence)) in batch.into_iter().enumerate() {
            if !active[index] {
                released.push(sequence);
                continue;
            }
            let LaneRun {
                mut tokens,
                spec,
                drawn_probability,
            } = decoded.next().expect("decoded result length was checked");
            // GitHub #242 — the one-round lag, closed here. The round
            // returns the token drawn *last* time, so its probability is the
            // one held from then; what this round drew is held for the next.
            // A lane that drew unconstrained leaves nothing behind, so the
            // entry disappears on the round after a run's last set and
            // never outlives the request.
            let probabilities = match drawn_probabilities.is_empty() {
                true => Vec::new(),
                false => match drawn_probabilities.remove(&job.request) {
                    Some(probability) => vec![probability],
                    None => Vec::new(),
                },
            };
            if let Some(probability) = drawn_probability {
                drawn_probabilities.insert(job.request, probability);
            }
            debug_assert!(
                probabilities.is_empty() || tokens.len() == 1,
                "a constrained lane commits exactly one token, so there is one probability for it"
            );
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
            outcomes[index] = Some(DecodeOutcome {
                spec,
                probabilities,
                ..outcome
            });
        }
        drop(drawn_probabilities);
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
        // GitHub #178: a request completed or cancelled mid-item lets its
        // item's embedding go with its sequence. GitHub #243: lets its
        // *hold* go — the picture it was asked about is very often the
        // picture the next request is about.
        self.release_media_of(request);
        // GitHub #242: and a constrained decode cancelled mid-run leaves a draw nobody
        // will ever emit. One `f32`, but the id is never reused, so an entry
        // left here would be leaked for the life of the process.
        self.drawn.lock().unwrap().remove(&request);
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
            &mut self
                .media
                .get_mut()
                .expect("RuntimeCompute is not dropped while its media lock is held")
                .entries,
        );
        for (_, entry) in media {
            self.release_media_handle(entry.handle);
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
