//! The production step-ABI leaf (GitHub #61 / P1-25): a real GPU-backed
//! [`StepLeaf`] over the vendored full program (P1-23, `ignis_core::step`)
//! — the first backend `ignis-server` drives beyond `MockCompute`.
//!
//! [`CudaLeaf`] owns everything `ignis_model_load` needs to (re)build a
//! model handle on demand — the artifact reader, its device-materialized
//! weights, the bound-tensor handles — plus the CUDA device context they
//! were materialized on (kept alive: the device memory is not valid once
//! the context is destroyed). [`StepLeaf::load_model`] sizes the
//! sequence-state pool deterministically from [`CudaLeafConfig`]
//! (`slot_count` sequences of up to `max_context_tokens` each) rather than
//! from the VRAM left over after the weights land: an earlier attempt at
//! the latter (`Device::free_bytes` minus a fixed reserve, handed straight
//! to `ignis_paged_kv_page_budget`) OOM'd on the real 27B artifact — the
//! model's real post-load footprint (weights + the program's own
//! workspace) leaves far less headroom than a naive `total - weights`
//! guess, and a wrong guess fails as a hard `cudaMalloc` error, not a
//! graceful degradation. Auto-sizing from real free VRAM (the target
//! envelope README describes) needs the leaf to report its own workspace
//! footprint first — later work, not needed for a correct G1 server.

#![cfg(feature = "cuda")]

use ignis_artifact::{
    CudaDevice, Device, MaterializationPlan, MaterializedArtifact, ObjectHandle, Reader,
};
use ignis_core::model_load::{self, Model as CoreModel};
use ignis_core::seq::{
    HostPinnedPool, PinnedAllocError, PinnedBuffer, Seq, SeqCheckpoint, SeqPool, SeqPoolBudget,
    SeqPrefix,
    snapshot_format_version,
};
use ignis_core::step;
use ignis_core::{
    ArtifactHash, BlobIdentity, DecodeParams, KvFormat, KvGeometry, KvPoolPlan, ModelConfig,
    N_DECODE_LANES, SpecCounters, TokenId, plan_kv_pool_for_context,
};

use ignis_core::vision::MediaItem;

use crate::{
    AttentionRead, DecodeLane, LaneRun, MultimodalSpan, ReservedBytes, RuntimeStats, StepLeaf,
};

/// Sizing knobs for the leaf's sequence-state pool and program scratch.
#[derive(Debug, Clone, Copy)]
pub struct CudaLeafConfig {
    /// The largest single sequence's KV reservation, in tokens (mirrors
    /// `ignis_core::SchedulerConfig::max_sequence_tokens`). Also the bound
    /// `ignis_model_load` sizes the GQA attention workspace reservation
    /// for (P2-01, GitHub #83) — must not be raised without also rebuilding
    /// the model handle.
    pub max_context_tokens: u32,
    /// The KV storage format this load runs on (ADR 0022, GitHub #122),
    /// fixed for the life of the model handle. It decides the pool's
    /// planes, so the same `kv_pool_bytes` buys 7.11x the tokens under
    /// hq-e8-2b as under BF16.
    pub kv_format: KvFormat,
    /// The paged-KV pool budget, in **bytes**: the pool every live sequence
    /// draws its pages from. Never in tokens — what this budget is worth in
    /// tokens is derived from `kv_format` and reported at load. Sized
    /// independently of `slot_count * max_context_tokens` — see
    /// [`CudaLeafConfig::default`].
    pub kv_pool_bytes: u64,
    /// Max concurrent sequences (mirrors [`N_DECODE_LANES`]).
    pub slot_count: u32,
    /// Retained slots the sequence pool holds past the lanes (GitHub #211,
    /// #215): a lane's mutable state each, reserved at load, where every
    /// published prefix's and captured checkpoint's image lives. The server's
    /// `--retained-slots`; 0 reserves none, and then nothing can be published
    /// or captured.
    pub retained_slots: u32,
    /// The prefill chunk width, in tokens: how wide a span the program's
    /// prefill scratch must serve (`--prefill-chunk`, GitHub #87). A
    /// nonzero multiple of 128, validated by the server's config module
    /// before any loader work starts. `ignis_model_load` reserves the
    /// program scratch for a chunk of this width at load time (P2-01,
    /// GitHub #83), and the chunk loop that spends it landed with GitHub
    /// #84: prefill traverses a span one chunk at a time, synchronizing
    /// once per chunk, not once per token.
    pub prefill_chunk_tokens: u32,
    /// Speculative decoding, fixed for the life of the model handle (P5-02,
    /// GitHub #150). With it, the leaf binds the drafter's weights — which
    /// `handles` must then carry (`ignis_artifact::bind_model_scope_27b`) —
    /// and the sequence pool carries its per-slot window; `None` is today's
    /// load.
    pub speculation: Option<ignis_core::Speculation>,
    /// Vision, fixed for the life of the model handle (GitHub #177). With it,
    /// `handles` must carry the `vision/*` objects
    /// (`ignis_artifact::bind_model_scope_27b_with`) and the load reserves the
    /// encoder workspace (inside the prefill scratch, GitHub #212) and output
    /// transient before the pool is built; `None` is today's load.
    pub vision: Option<ignis_core::Vision>,
    /// The text rotary table this load runs on (`--rope-scaling`, GitHub
    /// #227): [`ignis_core::RopeScaling::NONE`] is the linear table the
    /// engine has always used, and a YaRN factor rescales the trained
    /// 262,144-position envelope so a longer context means something.
    /// Frozen for the life of the load, like the two above: a sequence's
    /// cached keys are rotated with it.
    pub rope_scaling: ignis_core::RopeScaling,
}

impl Default for CudaLeafConfig {
    fn default() -> Self {
        Self {
            // The same defaults `ignis_server::config` falls back to
            // (`crate::{DEFAULT_MAX_CONTEXT, auto_kv_pool_bytes,
            // DEFAULT_PREFILL_CHUNK}`, defined once alongside this module
            // rather than restated here): `cuda_scheduler` always
            // overrides these fields from the operator's resolved
            // `EngineShape`, so this default only matters to a caller that
            // builds a leaf directly (the GPU layer/program tests) rather
            // than through the server.
            max_context_tokens: crate::DEFAULT_MAX_CONTEXT,
            // hq-e8-2b since GitHub #123 wired its attention routes: the
            // serving default ADR 0022 named, now that it can serve. A test
            // that wants the oracle format says `KvFormat::Bf16` rather than
            // inheriting it from here.
            kv_format: KvFormat::default(),
            kv_pool_bytes: crate::auto_kv_pool_bytes(
                KvFormat::default(),
                crate::DEFAULT_MAX_CONTEXT,
            ),
            slot_count: N_DECODE_LANES as u32,
            // `--retained-slots`' own default: a slot per lane.
            retained_slots: N_DECODE_LANES as u32,
            prefill_chunk_tokens: crate::DEFAULT_PREFILL_CHUNK,
            speculation: None,
            vision: None,
            rope_scaling: ignis_core::RopeScaling::NONE,
        }
    }
}

/// The production leaf: owns the CUDA device context, the device-
/// materialized artifact, and the bound-tensor handles `ignis_model_load`
/// reads on every (re)load.
pub struct CudaLeaf {
    // Read only for `RuntimeStats::free_vram_bytes` (GitHub #186), never to
    // size the KV pool — see the module doc; otherwise held purely so the
    // device context outlives `artifact` and every loaded model, since
    // dropping it would invalidate their device memory.
    device: CudaDevice,
    reader: Reader,
    artifact: MaterializedArtifact,
    handles: Vec<ObjectHandle>,
    config: CudaLeafConfig,
    /// The pinned KV-RAM arena every snapshot blob is placed in (GitHub
    /// #213), or `None` before [`CudaLeaf::with_kv_ram_arena`] pins one —
    /// which is what a leaf built for a test that never spills wants.
    ///
    /// Declared last on purpose: fields drop in declaration order, so this
    /// one goes after everything above it. What it must outlive is not in
    /// this struct at all but in `RuntimeCompute`'s snapshot maps, whose
    /// `Drop` frees them before it lets go of the model.
    host_pool: Option<HostPinnedPool>,
}

impl CudaLeaf {
    /// Build a leaf over an already device-materialized artifact
    /// (`ignis_artifact::materialize` against `bind_text_scope_27b`'s
    /// plan). The caller does the one-time device setup + weight upload;
    /// the leaf owns all of it from here so it can (re)load the model and
    /// size the sequence pool.
    pub fn new(
        device: CudaDevice,
        reader: Reader,
        artifact: MaterializedArtifact,
        handles: Vec<ObjectHandle>,
        config: CudaLeafConfig,
    ) -> Self {
        Self {
            device,
            reader,
            artifact,
            handles,
            config,
            host_pool: None,
        }
    }

    /// Pin `bytes` of KV-RAM as the one arena every snapshot blob is placed
    /// in (GitHub #213, ADR 0030), and hold it for as long as this leaf
    /// lives. `bytes` of 0 pins nothing and leaves the tier disabled.
    ///
    /// A load calls this before it serves anything: the pinning happens once
    /// here rather than per blob, which is what keeps `cudaHostAlloc` off the
    /// serving path and the process's shared GPU memory fixed on Windows.
    /// `Err` when the region cannot be pinned — the start is refused, since a
    /// leaf that cannot hold KV-RAM is not the engine the operator asked for.
    pub fn with_kv_ram_arena(mut self, bytes: u64) -> Result<Self, String> {
        self.host_pool = Some(HostPinnedPool::create(bytes)?);
        Ok(self)
    }
}

impl Drop for CudaLeaf {
    fn drop(&mut self) {
        // `artifact`'s weight arena has no `Drop` of its own — releasing it
        // needs the `Device` that produced it, so it must happen explicitly
        // here, before field auto-drop runs `device`'s own `Drop` (which
        // only tears down its load stream/event, never this allocation).
        // Without this, every load leaks its ~19 GB arena for the rest of
        // the process (the fault this fixes: multiple GPU tests in one
        // process accumulate one arena per `harness()` call).
        let _ = self.artifact.release_arena(&mut self.device);
    }
}

impl CudaLeafConfig {
    /// What this config's byte budget buys under its format: the pool the
    /// leaf will build and the token capacity it holds.
    ///
    /// `Err` (a **load** failure, GitHub #122) when the budget cannot hold
    /// one `max_context_tokens` sequence, naming the budget, the format and
    /// the capacity it bought. The scheduler's admission accounting
    /// (`ignis_server::runtime::cuda_scheduler`) reads the page count from
    /// this same call, so the two can never drift.
    pub fn kv_pool_plan(&self) -> Result<KvPoolPlan, ignis_core::KvBudgetTooSmall> {
        plan_kv_pool_for_context(
            self.kv_format,
            KvGeometry::qwen38_27b(),
            self.kv_pool_bytes,
            self.max_context_tokens,
        )
    }

    /// The compatibility identity of state a load with these options
    /// produces, given the `artifact` it runs on and the blob
    /// `layout_version` its leaf writes (GitHub #189, ADR 0029).
    ///
    /// **This is where a load option becomes part of the identity or does
    /// not.** Two of this struct's eight fields do: `kv_format`, which
    /// decides what a KV page *is*, and `speculation`, whose presence and
    /// draft window decide whether the sequence pool carries the drafter's
    /// per-slot sections. The other six — `max_context_tokens`,
    /// `kv_pool_bytes`, `slot_count`, `retained_slots`,
    /// `prefill_chunk_tokens` and `vision` —
    /// decide how much work fits and how fast it goes, never what the bytes
    /// of a sequence mean, so state produced under one value must still be
    /// usable under another. `kv_pool_bytes` is the sharpest of them: it is
    /// the rest of a VRAM budget derived from the memory free at start, a
    /// number that differs from one start of the same server to the next, and
    /// an identity that moved with any of this would refuse every blob after
    /// a reboot for no reason at all.
    ///
    /// The operator's other flags — the bind address, the API key, the
    /// request timeout, `--prompt-reuse` — are not fields here at all: `ignis_server::runtime::cuda_scheduler` builds
    /// this struct out of an `EngineShape`, which never carried them.
    pub fn blob_identity(&self, artifact: ArtifactHash, layout_version: u32) -> BlobIdentity {
        BlobIdentity::of_load(artifact, self.kv_format, self.speculation, layout_version)
    }

    /// Every reservation a load with these options makes beside the weights
    /// and the KV pool, asked of the leaf before anything is on the device
    /// (GitHub #210). `plan` and
    /// `handles` are the binder's, not yet materialized.
    pub fn plan_reservations(
        &self,
        reader: &Reader,
        plan: &MaterializationPlan,
        handles: &[ObjectHandle],
    ) -> Result<PlannedReservations, String> {
        let model = model_load::plan_qwen38_27b_reservations(
            reader,
            plan,
            handles,
            self.prefill_chunk_tokens,
            self.max_context_tokens,
            self.kv_format,
            self.speculation,
            self.vision,
            self.rope_scaling,
        )?;
        let pool = self.pool_plan(1)?;
        Ok(PlannedReservations {
            reserved: reserved_bytes(
                model,
                pool.lane_state_bytes,
                pool.retained_state_bytes,
                pool.hq_residual_bytes,
                0,
            ),
        })
    }

    /// The device bytes a KV pool of `pages` pages occupies under these
    /// options, block tables included (GitHub #210).
    pub fn kv_pool_arena_bytes(&self, pages: u32) -> Result<u64, String> {
        self.pool_plan(pages).map(|plan| plan.kv_bytes)
    }

    fn pool_plan(&self, pages: u32) -> Result<ignis_core::seq::IgnisSeqPoolPlan, String> {
        SeqPool::plan(
            &ModelConfig::qwen38_27b(),
            &self.pool_budget(pages),
            self.speculation.map(|s| s.backend()),
        )
    }

    fn pool_budget(&self, pages: u32) -> SeqPoolBudget {
        SeqPoolBudget {
            kv_format: self.kv_format,
            kv_page_group_count: pages,
            max_context_tokens: self.max_context_tokens,
            slot_count: self.slot_count,
            retained_slot_count: self.retained_slots,
        }
    }
}

/// The plan lines of a model's reservations and its pool's (GitHub #210,
/// #211).
fn reserved_bytes(
    model: model_load::IgnisModelReservations,
    lane_state: u64,
    retained_slots: u64,
    hq_residual_window: u64,
    kv_pool: u64,
) -> ReservedBytes {
    ReservedBytes {
        workspace: model.workspace_bytes,
        media_embedding: model.media_embedding_bytes,
        sampling: model.sampling_bytes,
        decode_graph: model.decode_graph_bytes,
        verify_round: model.verify_round_bytes,
        drafter_round: model.drafter_round_bytes,
        lane_state,
        retained_slots,
        hq_residual_window,
        kv_pool,
    }
}

/// What [`CudaLeafConfig::plan_reservations`] found (GitHub #210).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedReservations {
    /// Every line but the KV pool's, which the VRAM plan sizes from the rest.
    pub reserved: ReservedBytes,
}

/// The leaf's model handle: the loaded weights plus the sequence-state
/// pool sized for them. Both travel together — the step ABI takes the
/// pool and the model as separate parameters on every prefill/decode call.
pub struct CudaModel {
    model: CoreModel,
    pool: SeqPool,
}

// These wrap raw FFI pointers with no synchronization of their own. That is
// sound here because every `Compute` call the scheduler makes runs under
// `ignis-server::engine::Engine`'s single `Mutex` — never concurrently —
// matching `ignis_core::seq::SeqPool`'s own documented single-thread-driver
// contract.
unsafe impl Send for CudaModel {}
unsafe impl Sync for CudaModel {}
unsafe impl Send for CudaLeaf {}
unsafe impl Sync for CudaLeaf {}

/// Log `message` (with `context`) and collapse it to the generic leaf
/// error code `StepLeaf` expects — `ignis_core`'s step/seq/model_load
/// wrappers already discarded the raw C return code in favor of a
/// descriptive string (`ignis_*_last_error`), so there is no real code
/// left to preserve here.
///
/// Goes through `tracing::error!` (GitHub #80), not a raw `eprintln!`: this
/// fires from `prefill`/`decode` on the leaf's own error path, and a raw
/// `eprintln!` is exactly the uncontrolled blocking console I/O spec §27
/// forbids on that path — `ignis-logging`'s bounded priority queue decouples
/// it from physical I/O the same as every other ERROR site. This is a
/// failure path only (never once per token/layer/kernel in the success
/// case, i.e. not the frequency spec §26 constrains), so it stays an ERROR
/// rather than needing further demotion.
fn leaf_error(context: &str, message: String) -> i32 {
    // hotpath-lint-allow: failure-only path (prefill/decode error return, see the doc comment above), reviewed exception (GitHub #80).
    tracing::error!(name: "ignis.runtime.leaf_error", context, error = %message, "step leaf error");
    -1
}

impl StepLeaf for CudaLeaf {
    type Model = CudaModel;
    type Sequence = Seq<'static>;
    type Prefix = SeqPrefix<'static>;
    type Checkpoint = SeqCheckpoint<'static>;
    type SnapshotBuf = PinnedBuffer;
    type Media = step::MediaEmbedding<'static>;

    fn encode_media(&self, model: &Self::Model, item: &MediaItem) -> Result<Self::Media, i32> {
        let control = ignis_core::vision::vision_item_control(item.grid);
        let embedding = step::encode_media(&model.model, item.grid, &item.patches, &control)
            // GitHub #243: a full pool is not a leaf error — it is the leaf
            // telling the caller to release an embedding and call again, and
            // `RuntimeCompute` does exactly that. Logging it as an error
            // would put a line in the log for every cache miss under
            // pressure.
            .map_err(|(rc, message)| {
                if rc == step::MEDIA_ENCODE_POOL_FULL {
                    rc
                } else {
                    leaf_error("media encode", message)
                }
            })?;
        // Safety: as for sequences -- `RuntimeCompute` releases every live
        // embedding before its `Arc<Model<L>>` (and so this model) can drop.
        Ok(unsafe { embedding.into_static() })
    }

    fn release_media(&self, _model: &Self::Model, _media: Self::Media) {
        // Drops here: `MediaEmbedding::drop` calls `ignis_media_embedding_release`.
    }

    fn prefill_multimodal(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        span: MultimodalSpan<'_, Self::Media>,
        out_logits: Option<&mut [f32]>,
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let permitted_ids: Vec<i32> = permitted.iter().map(|&t| t as i32).collect();
        let (readout, read) = match attention {
            Some(AttentionRead { query, scores, set_argmax, set_peak, set_neighbours, read }) => (
                Some(step::AttentionReadout {
                    query,
                    scores: scores.as_mut_slice(),
                    set_argmax: set_argmax.as_mut_slice(),
                    set_peak: set_peak.as_mut_slice(),
                    set_neighbours: set_neighbours.as_mut_slice(),
                }),
                Some(read),
            ),
            None => (None, None),
        };
        let (probability, was_read) = step::prefill_program_multimodal(
            &model.model,
            &model.pool,
            sequence,
            &token_ids,
            u64::from(start_position),
            sampling_params(params),
            &permitted_ids,
            step::MultimodalPrefill {
                positions: span.positions,
                rope_delta: span.rope_delta,
                media: span.media.map(|media| step::SpanMediaColumns {
                    embedding: media.embedding,
                    first_column: media.first_column,
                    scatter_indices: media.scatter_indices,
                }),
            },
            out_logits,
            readout,
        )
        .map_err(|e| leaf_error("prefill", e))?;
        if let Some(read) = read {
            *read = was_read;
        }
        Ok(probability)
    }

    fn load_model(&self) -> Result<Self::Model, i32> {
        // P4-04 (GitHub #122): plan the pool before the weights go up. A
        // byte budget that cannot hold one configured context fails the
        // load here — naming the budget, the format and the capacity it
        // bought — rather than after a ~19 GB upload, or later still as an
        // admission promise the leaf can never honour.
        let plan = self
            .config
            .kv_pool_plan()
            .map_err(|e| leaf_error("kv pool plan", e.to_string()))?;
        // The format reaches the load itself (P4-05, GitHub #123), not just
        // the pool: the leaf sizes its attention workspace from it, and the
        // GQA layers refuse a pool built in the other one.
        // GitHub #177: the vision reservation is taken inside this load, so it
        // is on the device before the pool below is built.
        let model = model_load::load_qwen38_27b_with_options(
            &self.reader,
            &self.artifact,
            &self.handles,
            self.config.prefill_chunk_tokens,
            self.config.max_context_tokens,
            self.config.kv_format,
            self.config.speculation,
            self.config.vision,
            self.config.rope_scaling,
        )
        .map_err(|e| leaf_error("model load", e))?;
        let cfg = ModelConfig::qwen38_27b();
        // The pool holds the pages the byte budget bought under this load's
        // format, shared across the slots; `max_context_tokens` is the
        // per-sequence cap drawn against it.
        // `ignis_server::runtime::cuda_scheduler` reads the same plan, so
        // admission can never promise more pages than this pool holds.
        // P5-03 (GitHub #152): the drafter's window is per-sequence state,
        // so a speculative load's pool carries it for every slot.
        let pool = SeqPool::create_with_speculation(
            &cfg,
            &self.config.pool_budget(plan.page_count),
            self.config.speculation.map(|s| s.backend()),
        )
        .map_err(|e| leaf_error("seq pool create", e))?;
        // GitHub #122: report what the budget actually bought, read back
        // from the pool the leaf built rather than from the plan that asked
        // for it — a format change has to show up here as a number, not as
        // a surprise under load. Fires once per model load, the same
        // frequency class as `ignis.process.started`.
        let pool_stats = pool.stats();
        // hotpath-lint-allow: model-load-time only (`load_model`, runs once per process start), not per-token/decode-round (GitHub #80).
        tracing::info!(
            name: "ignis.runtime.kv_pool",
            kv_format = self.config.kv_format.as_str(),
            budget_bytes = self.config.kv_pool_bytes,
            pool_bytes = plan.pool_bytes,
            bytes_per_token = pool_stats.kv_bytes_per_token,
            page_count = pool_stats.kv_page_group_count,
            token_capacity = pool_stats.kv_token_capacity,
            max_context_tokens = self.config.max_context_tokens,
            // GitHub #177: what vision took before the pool (0 without it),
            // beside the capacity the pool holds after it.
            vision_max_tokens = self.config.vision.map_or(0, |v| v.max_tokens()),
            vision_reserved_bytes = model.stats().vision_reserved_bytes,
            // GitHub #227: `none`, or the YaRN spec the table was built
            // from -- the one place a long-context run can be checked
            // against what the operator meant to ask for.
            rope_scaling = %self.config.rope_scaling,
            "kv pool"
        );
        // P3-05 (GitHub #102, ADR 0019): capture the decode graphs once,
        // right after the pool exists and before any sequence is ever
        // allocated -- a per-width capture failure degrades that width to
        // the eager per-lane loop, never model load. `capture_decode_graphs`
        // itself never returns Err for that reason; only a null model/pool
        // (impossible here) would.
        //
        // Format-independent since P4-05 (GitHub #123): a capture set is one
        // per process either way, and the hq codec keeps fixed bytes per row
        // with a bounded, host-free escalation path, so the addresses a
        // width's graph bakes in are as stable under hq as under BF16 (ADR
        // 0022). #122's outright skip under hq is gone with the refusal it
        // existed to avoid eight copies of.
        let capture = step::capture_decode_graphs(&model, &pool)
            .map_err(|e| leaf_error("decode graph capture", e))?;
        // Reported unconditionally (GitHub #102's acceptance: "startup cost
        // of capturing eight graphs is measured and reported, not assumed")
        // -- not only when a width failed, so the common all-8-ready case
        // still surfaces the number rather than computing and discarding it.
        // GitHub #80: structured, not a raw `eprintln!` -- this still fires
        // exactly once per model load (never per-token/decode-round), same
        // frequency class as `ignis.process.started`.
        // hotpath-lint-allow: model-load-time only (`load_model`, runs once per process start), not per-token/decode-round (GitHub #80/#102).
        tracing::info!(
            name: "ignis.runtime.decode_graph_capture",
            ready = capture.ready_count(),
            capture_micros = capture.capture_micros,
            "decode graph capture"
        );
        if capture.ready_count() < 8 {
            // hotpath-lint-allow: same model-load-time call as above, one line down.
            tracing::warn!(
                name: "ignis.runtime.decode_graph_capture_incomplete",
                error = %step::last_decode_graph_error(),
                "decode graph capture: not all widths ready"
            );
        }
        Ok(CudaModel { model, pool })
    }

    fn release_model(&self, _model: Self::Model) {
        // `CudaModel`'s fields release themselves on drop (`ignis_model_free`,
        // `ignis_seq_pool_free`).
    }

    fn stats(&self, model: &Self::Model) -> Result<RuntimeStats, i32> {
        let program = step::program_stats(&model.model, &model.pool)
            .map_err(|e| leaf_error("program stats", e))?;
        let pool_stats = model.pool.stats();
        let reserved = model.model.stats().reserved;
        Ok(RuntimeStats {
            vram_bytes: program.vram_bytes,
            // The leaf's paged KV page size is fixed at 64 tokens
            // (`kernel/vendor/src/core/paged_kv_cache.h`'s
            // `kPagedKVPageSize`) — the ABI does not report it per call.
            kv_page_tokens: 64,
            kv_page_bytes: pool_stats.kv_page_bytes,
            kv_page_count: pool_stats.kv_page_group_count,
            last_step_micros: program.last_step_micros,
            kernel_count: program.kernel_count,
            // P3-05 (GitHub #102, ADR 0019): 1 when the most recent decode
            // round replayed a captured graph, 0 for every prefill step and
            // for a decode round whose exact width has no ready graph.
            graph_launches: program.graph_launches,
            // GitHub #186: what the device says is free right now. Read
            // rather than derived — the module doc above records what a
            // `total - weights` guess cost the last time one was made — and
            // 0 when the query fails rather than a number nobody measured.
            free_vram_bytes: self.device.free_bytes().unwrap_or(0),
            // GitHub #210: read off the model's and the pool's own buffers,
            // for the load to check against its VRAM plan.
            reserved: reserved_bytes(
                reserved,
                pool_stats.lane_state_bytes,
                pool_stats.retained_state_bytes,
                pool_stats.hq_residual_bytes,
                pool_stats.kv_arena_bytes,
            ),
        })
    }

    fn allocate_sequence(
        &self,
        model: &Self::Model,
        context_tokens: u32,
    ) -> Result<Self::Sequence, i32> {
        let seq = model
            .pool
            .alloc(context_tokens)
            .map_err(|e| leaf_error("sequence alloc", e))?;
        // Safety: `model.pool` outlives every sequence drawn from it —
        // `RuntimeCompute::drop` (`ignis-runtime/src/lib.rs`) releases all
        // live sequences before its `Arc<Model<L>>` (and so this pool) can
        // drop.
        Ok(unsafe { seq.into_static() })
    }

    fn release_sequence(&self, _model: &Self::Model, _sequence: Self::Sequence) {
        // Drops here: `Seq::drop` calls `ignis_seq_release`.
    }

    fn allocate_sequence_shared(
        &self,
        model: &Self::Model,
        context_tokens: u32,
        prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32> {
        let seq = model
            .pool
            .alloc_shared(context_tokens, prefix)
            .map_err(|e| leaf_error("shared sequence alloc", e))?;
        // Safety: as in `allocate_sequence` — `model.pool` outlives every
        // sequence drawn from it.
        Ok(unsafe { seq.into_static() })
    }

    fn publish_prefix(
        &self,
        _model: &Self::Model,
        sequence: &mut Self::Sequence,
        prefix_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Prefix, i32> {
        // A prefix borrows the pool, not the sequence — so publishing from a
        // `Seq<'static>` yields a `SeqPrefix<'static>` with no detaching
        // needed, and the publisher stays usable for its own tail.
        sequence
            .publish_prefix(prefix_tokens, retained_slot)
            .map_err(|e| leaf_error("prefix publish", e.to_string()))
    }

    fn capture_checkpoint(
        &self,
        _model: &Self::Model,
        sequence: &mut Self::Sequence,
        opener_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Checkpoint, i32> {
        // A checkpoint borrows the pool, not the sequence — capturing from a
        // `Seq<'static>` yields a `SeqCheckpoint<'static>` with no detaching
        // needed, exactly as `publish_prefix` above does, and the capturing
        // sequence stays usable for the rest of its prompt and its decode.
        sequence
            .capture_checkpoint(opener_tokens, retained_slot)
            .map_err(|e| leaf_error("checkpoint capture", e.to_string()))
    }

    fn allocate_sequence_from_checkpoint(
        &self,
        model: &Self::Model,
        context_tokens: u32,
        checkpoint: &Self::Checkpoint,
    ) -> Result<(Self::Sequence, u64), i32> {
        let sequence = model
            .pool
            .alloc_from_checkpoint(context_tokens, checkpoint)
            .map_err(|e| leaf_error("checkpoint claim", e))?;
        // Measured by the leaf during the claim, not timed around the call:
        // what a claimant pays is the point at which it can be stepped.
        let micros = checkpoint.stats().last_claim_micros;
        // Safety: as above — the pool outlives the sequences drawn from it.
        Ok((unsafe { sequence.into_static() }, micros.round().max(0.0) as u64))
    }

    fn release_checkpoint(&self, _model: &Self::Model, _checkpoint: Self::Checkpoint) {
        // `SeqCheckpoint`'s own `Drop` gives back its retained slot and its
        // tail page and lets go of the prefix under it
        // (`ignis_seq_checkpoint_release`).
    }

    fn checkpoint_snapshot_bytes(
        &self,
        _model: &Self::Model,
        checkpoint: &Self::Checkpoint,
    ) -> Result<u64, i32> {
        checkpoint
            .snapshot_bytes()
            .map_err(|e| leaf_error("checkpoint snapshot size", e.to_string()))
    }

    fn checkpoint_snapshot_into(
        &self,
        _model: &Self::Model,
        checkpoint: &Self::Checkpoint,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        checkpoint
            .snapshot_into(dst)
            .map_err(|e| leaf_error("checkpoint snapshot", e.to_string()))
    }

    fn prefix_snapshot_bytes(&self, _model: &Self::Model, prefix: &Self::Prefix) -> Result<u64, i32> {
        prefix
            .snapshot_bytes()
            .map_err(|e| leaf_error("prefix snapshot size", e.to_string()))
    }

    fn prefix_snapshot_into(
        &self,
        _model: &Self::Model,
        prefix: &Self::Prefix,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        prefix
            .snapshot_into(dst)
            .map_err(|e| leaf_error("prefix snapshot", e.to_string()))
    }

    /// GitHub #189: the four facts that decide whether retained state may be
    /// written into a sequence of this load, read from the load itself.
    ///
    /// Two of them exist nowhere else: the artifact's content hash comes from
    /// the container this leaf opened, and the blob layout version comes from
    /// the leaf's own state-section table (ADR 0024 keeps that table
    /// internal, so a version restated in Rust would not move when the table
    /// did). Which of the load's *options* join them is
    /// [`CudaLeafConfig::blob_identity`]'s decision, so that it can be put
    /// under test without a card.
    fn blob_identity(&self) -> BlobIdentity {
        self.config.blob_identity(
            ArtifactHash::from_bytes(self.reader.content_hash()),
            snapshot_format_version(),
        )
    }

    fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {
        // Drops here: `SeqPrefix::drop` calls `ignis_seq_prefix_release`,
        // which is one holder fewer — not a free.
    }

    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        out_logits: Option<&mut [f32]>,
    ) -> Result<f32, i32> {
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        // The unconstrained path is left exactly as it was — every request
        // this engine serves takes it, and a constrained prefill is a
        // different options struct, not a flag on this one.
        if permitted.is_empty() {
            return step::prefill_program_sampled(
                &model.model,
                &model.pool,
                sequence,
                &token_ids,
                u64::from(start_position),
                sampling_params(params),
                out_logits,
            )
            .map(|()| 0.0)
            .map_err(|e| leaf_error("prefill", e));
        }
        let permitted_ids: Vec<i32> = permitted.iter().map(|&t| t as i32).collect();
        step::prefill_program_permitted(
            &model.model,
            &model.pool,
            sequence,
            &token_ids,
            u64::from(start_position),
            sampling_params(params),
            &permitted_ids,
            out_logits,
        )
        .map_err(|e| leaf_error("prefill", e))
    }

    /// The output head's width — 248,320 columns, which is what the kernel
    /// writes into a readout buffer (GitHub #237). ignis is specialized for
    /// this one topology (`CONTEXT.md`), so the number comes from the same
    /// `ModelConfig` the load itself is built from rather than being asked
    /// of the leaf.
    fn vocab(&self, _model: &Self::Model) -> u32 {
        ModelConfig::qwen38_27b().vocab as u32
    }

    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        lanes: &[DecodeLane<'_>],
    ) -> Result<Vec<LaneRun>, i32> {
        if lanes.len() > N_DECODE_LANES {
            return Err(leaf_error(
                "decode",
                format!(
                    "batch has {} sampling parameter sets; maximum is {N_DECODE_LANES}",
                    lanes.len()
                ),
            ));
        }
        // GitHub #242: a **constrained** round, which is a different round
        // in two ways. It commits one token per lane — no drafts, because a
        // draft is proposed by a second model that knows nothing of a set,
        // and the leaf refuses a constrained lane in a verify round outright
        // rather than accepting one it cannot honour. And it takes the plain
        // path for the *whole batch*: a speculative load runs every round as
        // a verify round, so one `number` in flight would otherwise fail
        // every lane's round. The siblings lose that round's drafts, which
        // is a number's six rounds' worth and not a mode the engine stays
        // in.
        if lanes.iter().any(|lane| !lane.permitted.is_empty()) {
            let permitted: Vec<Vec<i32>> = lanes
                .iter()
                .map(|lane| lane.permitted.iter().map(|&id| id as i32).collect())
                .collect();
            let constrained: Vec<step::PermittedLane<'_>> = lanes
                .iter()
                .zip(&permitted)
                .map(|(lane, ids)| step::PermittedLane {
                    sampling: sampling_params(lane.params),
                    permitted: ids,
                })
                .collect();
            let (ids, probabilities) = step::decode_program_batch_permitted(
                &model.model,
                &model.pool,
                sequences,
                &constrained,
            )
            .map_err(|e| leaf_error("decode", e))?;
            return Ok(ids
                .into_iter()
                .zip(probabilities)
                .zip(lanes)
                .map(|((id, probability), lane)| {
                    LaneRun::drawn(
                        id as TokenId,
                        (!lane.permitted.is_empty()).then_some(probability),
                    )
                })
                .collect());
        }
        let Some(speculation) = self.config.speculation else {
            let mut sampling = [step::SamplingParams::greedy(); N_DECODE_LANES];
            for (target, lane) in sampling.iter_mut().zip(lanes) {
                *target = sampling_params(lane.params);
            }
            let ids = step::decode_program_batch_sampled(
                &model.model,
                &model.pool,
                sequences,
                &sampling[..lanes.len()],
            )
            .map_err(|e| leaf_error("decode", e))?;
            return Ok(ids.into_iter().map(|id| LaneRun::token(id as TokenId)).collect());
        };
        // P5-06 (GitHub #154, spec 05): a speculative load runs every round
        // as a verify round at the window it was loaded with. The DFlash2
        // drafter proposes inside the leaf (P5-05, GitHub #155), so no lane
        // passes drafts here; the leaf reports what each lane verified.
        let stop_ids: Vec<Vec<i32>> = lanes
            .iter()
            .map(|lane| lane.stop_ids.iter().map(|&id| id as i32).collect())
            .collect();
        let verify: Vec<step::VerifyLane<'_>> = lanes
            .iter()
            .zip(&stop_ids)
            .map(|(lane, stops)| step::VerifyLane {
                sampling: sampling_params(lane.params),
                remaining_tokens: lane.remaining_tokens,
                stop_ids: stops,
                drafts: &[],
            })
            .collect();
        let runs = step::decode_program_verify_runs(
            &model.model,
            &model.pool,
            sequences,
            &verify,
            speculation.draft_tokens(),
        )
        .map_err(|e| leaf_error("decode", e))?;
        Ok(runs
            .into_iter()
            .map(|run| {
                // Committed drafts: the run past its anchor, which a stop cut
                // may leave shorter than what the target accepted.
                let accepted = (run.tokens.len() as u32).saturating_sub(1).min(run.extent);
                LaneRun {
                    tokens: run.tokens.into_iter().map(|id| id as TokenId).collect(),
                    spec: Some(SpecCounters::round(run.extent, accepted)),
                    drawn_probability: None,
                }
            })
            .collect())
    }

    fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32> {
        PinnedBuffer::new(bytes).map_err(|e| match e {
            // GitHub #213: a fragmented arena is a refusal, not a failure.
            // The scheduler probed with `host_blob_fits` before asking, so
            // reaching this means blobs changed hands in between — the caller
            // turns the spill away, and there is no error to report.
            PinnedAllocError::NoRoom => ignis_core::seq::NO_HOST_ROOM,
            PinnedAllocError::Failed(message) => leaf_error("snapshot alloc", message),
        })
    }

    fn host_blob_fits(&self, bytes: u64) -> bool {
        ignis_core::seq::host_blob_fits(bytes)
    }

    fn snapshot_bytes(&self, _model: &Self::Model, sequence: &Self::Sequence) -> Result<u64, i32> {
        sequence
            .snapshot_bytes()
            .map_err(|e| leaf_error("snapshot size", e.to_string()))
    }

    fn snapshot_into(
        &self,
        _model: &Self::Model,
        sequence: &Self::Sequence,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        sequence
            .snapshot_into(dst)
            .map_err(|e| leaf_error("snapshot", e.to_string()))
    }

    fn restore_sequence(
        &self,
        _model: &Self::Model,
        sequence: &mut Self::Sequence,
        src: &[u8],
    ) -> Result<(), i32> {
        sequence
            .restore(src)
            .map_err(|e| leaf_error("restore", e.to_string()))
    }
}

fn sampling_params(params: DecodeParams) -> step::SamplingParams {
    step::SamplingParams {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        presence_penalty: params.presence_penalty,
        frequency_penalty: params.frequency_penalty,
        seed: params.seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure arithmetic, no device — compiled (and run) whenever the `cuda`
    // feature is built, without needing the GPU profile (ADR 0006).

    /// A stand-in artifact hash and blob layout version — held fixed so the
    /// only thing varying below is the operator's load options.
    fn artifact() -> ArtifactHash {
        ArtifactHash::from_bytes([7; 32])
    }
    const LAYOUT: u32 = 2;

    #[test]
    fn the_load_options_that_do_not_change_state_do_not_change_the_identity() {
        // GitHub #189, ADR 0029, on the struct production actually assembles
        // the identity from (`cuda_scheduler` builds one of these out of the
        // operator's `EngineShape`, and `CudaLeaf::blob_identity` reads it).
        // Every field varied here moves how much work fits or how fast it
        // goes; none of them changes what a sequence's bytes mean, so a blob
        // taken under one must still be usable under the other.
        let lean = CudaLeafConfig::default();
        let generous = CudaLeafConfig {
            max_context_tokens: lean.max_context_tokens / 2,
            kv_pool_bytes: lean.kv_pool_bytes * 2,
            slot_count: lean.slot_count * 2,
            prefill_chunk_tokens: 512,
            vision: Some(ignis_core::Vision::default()),
            ..lean
        };
        assert_ne!(
            (generous.max_context_tokens, generous.slot_count),
            (lean.max_context_tokens, lean.slot_count),
            "the two configs really do differ"
        );
        assert_eq!(
            lean.blob_identity(artifact(), LAYOUT),
            generous.blob_identity(artifact(), LAYOUT),
            "context, budget, concurrency, chunk width and vision stay out of it"
        );
    }

    #[test]
    fn the_load_options_that_change_state_do_change_the_identity() {
        let shape = CudaLeafConfig::default();
        let baseline = shape.blob_identity(artifact(), LAYOUT);

        let other_format = CudaLeafConfig {
            kv_format: match shape.kv_format {
                KvFormat::Bf16 => KvFormat::HqE8_2b,
                KvFormat::HqE8_2b => KvFormat::Bf16,
            },
            ..shape
        };
        assert_eq!(
            baseline
                .accepts(&other_format.blob_identity(artifact(), LAYOUT))
                .expect_err("another KV format")
                .field,
            ignis_core::IdentityField::KvFormat
        );

        let drafter = CudaLeafConfig {
            speculation: Some(
                ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Dflash2, 4).unwrap(),
            ),
            ..shape
        };
        assert_eq!(
            baseline
                .accepts(&drafter.blob_identity(artifact(), LAYOUT))
                .expect_err("a drafter this load has not bound")
                .field,
            ignis_core::IdentityField::Drafter
        );
        // And the window at equal presence: a wider draft window is a
        // different per-slot section, not a different speed.
        let wider = CudaLeafConfig {
            speculation: Some(
                ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Dflash2, 6).unwrap(),
            ),
            ..shape
        };
        assert_eq!(
            drafter
                .blob_identity(artifact(), LAYOUT)
                .accepts(&wider.blob_identity(artifact(), LAYOUT))
                .expect_err("another draft window")
                .field,
            ignis_core::IdentityField::Drafter
        );

        // GitHub #195 landed `--vision` and `--spec dflash2` as independently
        // resolvable load options, so both can be present at once — and the
        // drafter half of the identity has to stay the drafter's, not a mode.
        let both = CudaLeafConfig {
            vision: Some(ignis_core::Vision::default()),
            ..drafter
        };
        assert_eq!(
            drafter.blob_identity(artifact(), LAYOUT),
            both.blob_identity(artifact(), LAYOUT),
            "vision rides beside the drafter without disturbing it"
        );

        // The artifact and the leaf's blob layout are the load's too.
        let elsewhere = ArtifactHash::from_bytes([9; 32]);
        assert_eq!(
            baseline
                .accepts(&shape.blob_identity(elsewhere, LAYOUT))
                .expect_err("another artifact")
                .field,
            ignis_core::IdentityField::Artifact
        );
        assert_eq!(
            baseline
                .accepts(&shape.blob_identity(artifact(), LAYOUT + 1))
                .expect_err("another blob layout")
                .field,
            ignis_core::IdentityField::LayoutVersion
        );
    }

    #[test]
    fn the_pool_plan_reports_whole_pages_of_the_configured_format() {
        let config = CudaLeafConfig::default();
        let plan = config.kv_pool_plan().expect("the auto default always fits");
        // hq-e8-2b since GitHub #123: the serving default ADR 0022 named.
        assert_eq!(plan.format, KvFormat::HqE8_2b);
        assert_eq!(
            u64::from(plan.page_count) * u64::from(ignis_core::KV_PAGE_TOKENS),
            plan.token_capacity
        );
        assert!(plan.pool_bytes <= config.kv_pool_bytes);
    }

    #[test]
    fn the_default_leaf_config_pool_can_hold_the_default_context() {
        // `cuda_scheduler` always overrides these fields from the operator's
        // `EngineShape` in production; this default is what a GPU test gets
        // when it builds a leaf directly, and it must not promise a context
        // the pool it also defaults to cannot serve.
        let config = CudaLeafConfig::default();
        let plan = config.kv_pool_plan().expect("the auto default always fits");
        assert!(plan.token_capacity >= u64::from(config.max_context_tokens));
        assert_eq!(config.max_context_tokens, crate::DEFAULT_MAX_CONTEXT);
        assert_eq!(config.prefill_chunk_tokens, crate::DEFAULT_PREFILL_CHUNK);
    }

    #[test]
    fn vision_leaves_the_derived_kv_capacity_as_it_is_today() {
        // GitHub #177: the vision reservation is taken beside the operator's
        // KV byte budget, not out of it -- the pool a load plans is the same
        // with or without vision, for every format.
        for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
            let plain = CudaLeafConfig {
                kv_format: format,
                ..CudaLeafConfig::default()
            };
            let with_vision = CudaLeafConfig {
                vision: Some(ignis_core::Vision::default()),
                ..plain
            };
            assert_eq!(
                with_vision.kv_pool_plan().expect("plan"),
                plain.kv_pool_plan().expect("plan"),
                "{format:?}"
            );
        }
    }

    #[test]
    fn a_byte_budget_too_small_for_the_context_fails_the_load_naming_the_numbers() {
        // GitHub #122: the refusal happens in `load_model`, before the
        // weights go up — and the message has to name the budget, the
        // format and the capacity it bought. Both formats are asked, because
        // "the format" is whichever one is in force and the message is only
        // useful if it names that one.
        for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
            let config = CudaLeafConfig {
                kv_format: format,
                kv_pool_bytes: 1024 * 1024,
                ..CudaLeafConfig::default()
            };
            let err = config.kv_pool_plan().expect_err("1 MiB holds no full context");
            let message = err.to_string();
            assert!(message.contains("1048576"), "{message}");
            assert!(message.contains(format.as_str()), "{message}");
            assert!(
                message.contains(&crate::DEFAULT_MAX_CONTEXT.to_string()),
                "{message}"
            );
        }
    }

    #[test]
    fn the_same_budget_buys_more_tokens_under_hq_than_under_bf16() {
        // The load option is a real option: nothing but the format changes
        // between these two plans. Both are named, so neither arm depends on
        // which one `Default` currently is.
        let bf16 = CudaLeafConfig {
            kv_format: KvFormat::Bf16,
            ..CudaLeafConfig::default()
        };
        let hq = CudaLeafConfig {
            kv_format: KvFormat::HqE8_2b,
            ..CudaLeafConfig::default()
        };
        assert_eq!(bf16.kv_pool_bytes, hq.kv_pool_bytes);
        let bf16_capacity = bf16.kv_pool_plan().expect("bf16 plan").token_capacity;
        let hq_capacity = hq.kv_pool_plan().expect("hq plan").token_capacity;
        assert!(hq_capacity > 7 * bf16_capacity, "{hq_capacity} vs {bf16_capacity}");
    }

    #[test]
    fn leaf_error_emits_a_structured_event_and_returns_the_generic_code() {
        use std::sync::Arc;
        use tracing_subscriber::layer::SubscriberExt;

        let sink = Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        let code = tracing::subscriber::with_default(subscriber, || {
            leaf_error("prefill", "device out of memory".to_owned())
        });

        assert_eq!(code, -1, "the generic leaf error code, regardless of context");
        let lines = sink.lines();
        assert_eq!(lines.len(), 1);
        let record: serde_json::Value = serde_json::from_str(&lines[0]).expect("valid json");
        assert_eq!(record["event_name"], "ignis.runtime.leaf_error");
        assert_eq!(record["severity_text"], "ERROR");
        assert_eq!(record["attributes"]["context"], "prefill");
        assert_eq!(record["attributes"]["error"], "device out of memory");
    }

    #[test]
    fn decode_params_map_every_sampling_field_without_changing_seed_bits() {
        let mapped = sampling_params(DecodeParams {
            max_tokens: Some(17),
            temperature: 1.25,
            top_p: 0.75,
            top_k: 13,
            presence_penalty: -0.5,
            frequency_penalty: 0.625,
            seed: u64::MAX,
            // Not a sampling field the leaf sees: `ignore_eos` is the
            // scheduler's own stop-condition switch, so `sampling_params`
            // below must not carry it into `step::SamplingParams`.
            ignore_eos: false,
        });

        assert_eq!(
            mapped,
            step::SamplingParams {
                temperature: 1.25,
                top_p: 0.75,
                top_k: 13,
                presence_penalty: -0.5,
                frequency_penalty: 0.625,
                seed: u64::MAX,
            }
        );
    }
}
