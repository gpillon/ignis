//! The Flash-Next step leaf (spec flash-next/04, GitHub #302): the
//! [`StepLeaf`] a server serving Qwen3.8-Flash-Next runs on, beside
//! [`crate::CudaLeaf`] for the 27B.
//!
//! What it adds to a step is the n-gram embedding's host half: each
//! sequence carries its hashing context ([`NgramContext`]), and every
//! prefill span and every decode round stages its tokens' table rows
//! (`NgramTable`, the hot rows in RAM and the rest read from the file)
//! before the leaf runs. The model borrows the leaf's expert residency, so
//! the leaf outlives every model it loads -- [`crate::Model`] holds the leaf
//! by `Arc` and releases its model first.
//!
//! What it does not serve (spec flash-next/04 and 05): prompt reuse --
//! shared prefixes, checkpoints, KV-RAM snapshots -- until its pool's
//! Flash-Next sections enter the clone table (the leaf refuses each by
//! name), vision and attention readouts. The server runs it with prompt
//! reuse off and the KV-RAM tier disabled.

use std::path::Path;

use ignis_artifact::flash_next::{self, FlashNextGeometry, FlashNextPlan};
use ignis_artifact::{materialize, CudaDevice, Device, MaterializedArtifact, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next::{build_residency, EngineOptions};
use ignis_core::model_load::{self, Model as CoreModel};
use ignis_core::ngram::NgramContext;
use ignis_core::ngram_table::NgramTable;
use ignis_core::residency::device::DeviceResidency;
use ignis_core::seq::{Seq, SeqCheckpoint, SeqPool, SeqPrefix};
use ignis_core::step;
use ignis_core::types::{DecodeParams, TokenId};

use crate::{AttentionRead, DecodeLane, LaneRun, ReservedBytes, RuntimeStats, StepLeaf};

fn leaf_error(context: &str, message: String) -> i32 {
    // hotpath-lint-allow: failure-only path, as CudaLeaf's own (GitHub #80).
    tracing::error!(name: "ignis.runtime.leaf_error", context, error = %message, "flash-next leaf error");
    -1
}

fn refused(what: &str) -> i32 {
    leaf_error(
        what,
        "Qwen3.8-Flash-Next serves no prompt reuse yet: its pool's sections are not in the clone table \
         (spec flash-next/05)"
            .to_string(),
    )
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

/// A loaded Flash-Next model and its sequence pool.
pub struct FlashNextModel {
    model: CoreModel,
    pool: SeqPool,
}

/// One sequence: its pool state and its n-gram hashing context, which
/// moves with it.
pub struct FlashNextSequence {
    seq: Seq<'static>,
    context: NgramContext,
}

/// The leaf: the bound artifact, its device weights, the expert residency
/// and the n-gram table. Field order is drop order (the device last).
pub struct FlashNextLeaf {
    residency: DeviceResidency,
    table: NgramTable,
    plan: FlashNextPlan,
    geometry: FlashNextGeometry,
    config: ModelConfig,
    options: EngineOptions,
    artifact: MaterializedArtifact,
    _reader: Reader,
    device: CudaDevice,
}

// Raw FFI handles with no synchronization of their own: sound for the same
// reason as CudaLeaf's -- every `Compute` call runs under the engine's single
// `Mutex`, one step at a time.
unsafe impl Send for FlashNextModel {}
unsafe impl Sync for FlashNextModel {}
unsafe impl Send for FlashNextLeaf {}
unsafe impl Sync for FlashNextLeaf {}

impl FlashNextLeaf {
    /// Bind and place the artifact at `path`, build the expert residency
    /// (the expert cache split into its eight K-class pools by resid's plan,
    /// the pinned pool filled from the file) and open the n-gram table.
    pub fn open(path: &Path, options: EngineOptions) -> Result<Self, String> {
        let options = options.normalized();
        let reader = Reader::open(path).map_err(|e| format!("open {}: {e:?}", path.display()))?;
        let geometry = FlashNextGeometry::qwen38_flash_next();
        let plan = flash_next::bind(&reader, &geometry).map_err(|e| format!("bind the Flash-Next artifact: {e:?}"))?;
        let config = ModelConfig::flash_next_from(&geometry);
        let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
        let artifact = materialize(&reader, &plan.plan, &mut device, None).map_err(|e| format!("materialize: {e}"))?;
        let residency = build_residency(path, &plan, &options)?;
        let ngram = config.ngram.ok_or("the Flash-Next topology has no n-gram embedding")?;
        let table = NgramTable::from_artifact(path, &reader, &plan, ngram, options.ngram)?;
        Ok(Self {
            residency,
            table,
            plan,
            geometry,
            config,
            options,
            artifact,
            _reader: reader,
            device,
        })
    }

    /// The decode lanes the load serves: the scheduler's resident lanes.
    pub fn decode_lanes(&self) -> u32 {
        self.options.decode_lanes
    }

    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    pub fn ngram_table(&self) -> &NgramTable {
        &self.table
    }

    fn rows_for(&self, context: &mut NgramContext, tokens: &[u32]) -> Result<Vec<u8>, String> {
        let mut rows = vec![0u8; tokens.len() * self.table.token_bytes()];
        self.table.stage(context, tokens, &mut rows)?;
        Ok(rows)
    }
}

impl StepLeaf for FlashNextLeaf {
    type Model = FlashNextModel;
    type Sequence = FlashNextSequence;
    type Prefix = SeqPrefix<'static>;
    type Checkpoint = SeqCheckpoint<'static>;
    type SnapshotBuf = Vec<u8>;
    type Media = ();

    fn load_model(&self) -> Result<Self::Model, i32> {
        let o = &self.options;
        let model = model_load::load_flash_next(
            &self.plan,
            &self.geometry,
            &self.artifact,
            o.prefill_chunk_tokens,
            o.max_context_tokens,
            o.kv_format,
            o.decode_lanes,
            &self.residency,
        )
        .map_err(|e| leaf_error("model load", e))?;
        let pool =
            SeqPool::create(&self.config, &o.pool_budget()).map_err(|e| leaf_error("seq pool create", e))?;
        if o.capture_graphs {
            let capture = step::capture_decode_graphs(&model, &pool).map_err(|e| leaf_error("graph capture", e))?;
            // hotpath-lint-allow: model-load-time only, once per process start (GitHub #80).
            tracing::info!(
                name: "ignis.runtime.flash_next_graphs",
                ready = capture.ready_count(),
                lanes = o.decode_lanes,
                "flash-next round graphs"
            );
        }
        Ok(FlashNextModel { model, pool })
    }

    fn release_model(&self, _model: Self::Model) {}

    fn stats(&self, model: &Self::Model) -> Result<RuntimeStats, i32> {
        let program =
            step::program_stats(&model.model, &model.pool).map_err(|e| leaf_error("program stats", e))?;
        let pool_stats = model.pool.stats();
        let reserved = model.model.stats().reserved;
        Ok(RuntimeStats {
            vram_bytes: program.vram_bytes,
            kv_page_tokens: 64,
            kv_page_bytes: pool_stats.kv_page_bytes,
            kv_page_count: pool_stats.kv_page_group_count,
            last_step_micros: program.last_step_micros,
            kernel_count: program.kernel_count,
            graph_launches: program.graph_launches,
            free_vram_bytes: self.device.free_bytes().unwrap_or(0),
            reserved: ReservedBytes {
                workspace: reserved.workspace_bytes + reserved.activation_bytes,
                sampling: reserved.sampling_bytes,
                decode_graph: reserved.decode_graph_bytes,
                lane_state: pool_stats.lane_state_bytes + pool_stats.indexer_bytes + pool_stats.ngram_conv_bytes,
                hq_residual_window: pool_stats.hq_residual_bytes,
                kv_pool: pool_stats.kv_arena_bytes,
                ..ReservedBytes::default()
            },
        })
    }

    fn vocab(&self, _model: &Self::Model) -> u32 {
        self.config.vocab as u32
    }

    fn allocate_sequence(&self, model: &Self::Model, context_tokens: u32) -> Result<Self::Sequence, i32> {
        let seq = model.pool.alloc(context_tokens).map_err(|e| leaf_error("seq alloc", e))?;
        // Safety: as CudaLeaf's -- `RuntimeCompute` releases every live
        // sequence before its `Arc<Model<L>>` (and so the pool) can drop.
        Ok(FlashNextSequence { seq: unsafe { seq.into_static() }, context: self.table.new_context() })
    }

    fn release_sequence(&self, _model: &Self::Model, _sequence: Self::Sequence) {}

    fn allocate_sequence_shared(
        &self,
        _model: &Self::Model,
        _context_tokens: u32,
        _prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32> {
        Err(refused("shared prefix claim"))
    }

    fn publish_prefix(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        _prefix_tokens: u32,
        _retained_slot: u32,
    ) -> Result<Self::Prefix, i32> {
        Err(refused("prefix publish"))
    }

    fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {}

    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        out_logits: Option<&mut [f32]>,
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        if attention.is_some() {
            return Err(leaf_error("prefill", "Qwen3.8-Flash-Next serves no attention readouts".to_string()));
        }
        let rows = self.rows_for(&mut sequence.context, tokens).map_err(|e| leaf_error("n-gram rows", e))?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let permitted: Vec<i32> = permitted.iter().map(|&t| t as i32).collect();
        step::prefill_flash_next(
            &model.model,
            &model.pool,
            &mut sequence.seq,
            &ids,
            u64::from(start_position),
            sampling_params(params),
            &permitted,
            &rows,
            None,
            out_logits,
        )
        .map_err(|e| leaf_error("prefill", e))
    }

    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        lanes: &[DecodeLane<'_>],
    ) -> Result<Vec<LaneRun>, i32> {
        // Each lane's pending token, hashed on its own context: the rows the
        // round's n-gram embedding reads for the token it consumes.
        let pending: Vec<[u32; 1]> = sequences
            .iter()
            .map(|s| s.seq.pending_token().map(|t| [t as u32]).ok_or(()))
            .collect::<Result<_, ()>>()
            .map_err(|()| leaf_error("decode", "a lane was not prefilled".to_string()))?;
        let mut rows = vec![0u8; sequences.len() * self.table.token_bytes()];
        {
            let mut batch: Vec<(&mut NgramContext, &[u32])> =
                sequences.iter_mut().zip(&pending).map(|(s, t)| (&mut s.context, &t[..])).collect();
            self.table
                .begin_batch(&mut batch)
                .and_then(|pending| pending.finish(&mut rows))
                .map_err(|e| leaf_error("n-gram rows", e))?;
        }
        let permitted: Vec<Vec<i32>> =
            lanes.iter().map(|lane| lane.permitted.iter().map(|&t| t as i32).collect()).collect();
        let sampling: Vec<(step::SamplingParams, &[i32])> =
            lanes.iter().zip(&permitted).map(|(lane, ids)| (sampling_params(lane.params), &ids[..])).collect();
        let mut seqs: Vec<&mut Seq<'static>> = sequences.iter_mut().map(|s| &mut s.seq).collect();
        let drawn = step::decode_flash_next(&model.model, &model.pool, &mut seqs, &sampling, &rows)
            .map_err(|e| leaf_error("decode", e))?;
        Ok(drawn
            .into_iter()
            .zip(lanes)
            .map(|((id, probability), lane)| {
                LaneRun::drawn(id as TokenId, (!lane.permitted.is_empty()).then_some(probability))
            })
            .collect())
    }

    fn alloc_snapshot_buf(&self, _bytes: u64) -> Result<Self::SnapshotBuf, i32> {
        Err(refused("snapshot buffer"))
    }

    fn host_blob_fits(&self, _bytes: u64) -> bool {
        false
    }

    fn snapshot_bytes(&self, _model: &Self::Model, _sequence: &Self::Sequence) -> Result<u64, i32> {
        Err(refused("snapshot size"))
    }

    fn snapshot_into(&self, _model: &Self::Model, _sequence: &Self::Sequence, _dst: &mut [u8]) -> Result<(), i32> {
        Err(refused("snapshot"))
    }

    fn restore_sequence(&self, _model: &Self::Model, _sequence: &mut Self::Sequence, _src: &[u8]) -> Result<(), i32> {
        Err(refused("restore"))
    }
}
