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
//! Prompt reuse (spec flash-next/05, GitHub #303) is ADR 0029's, on the
//! pool's Flash-Next sections: shared prefixes and checkpoints in retained
//! slots, and snapshot blobs in a KV-RAM arena this leaf owns (no
//! process-wide arena: the model instance frees it). What the pool's state
//! does not hold is the n-gram hashing context, so it travels beside it: a
//! prefix and a checkpoint keep the context at their end, and every blob
//! carries it in a block before the pool's own bytes.
//!
//! What it does not serve (spec flash-next/04): vision and attention
//! readouts.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use ignis_artifact::flash_next::{self, FlashNextGeometry, FlashNextPlan};
use ignis_artifact::{Device, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next::{build_residency, DeviceWeights, EngineOptions};
use ignis_core::flash_next_counters::FlashNextCounterSource;
use ignis_core::model_load::{self, Model as CoreModel};
use ignis_core::ngram::NgramContext;
use ignis_core::ngram_table::NgramTable;
use ignis_core::residency::device::DeviceResidency;
use ignis_core::residency::ResidencyMirror;
use ignis_core::seq::{ArenaBuffer, HostArena, PinnedAllocError, Seq, SeqCheckpoint, SeqPool, SeqPrefix};
use ignis_core::step;
use ignis_core::types::{DecodeParams, TokenId};
use ignis_core::{ArtifactHash, BlobIdentity};

use crate::{AttentionRead, DecodeLane, LaneRun, ReservedBytes, RuntimeStats, StepLeaf};

/// The block before the pool's bytes in a Flash-Next blob: the n-gram
/// context's token count and tokens, `u32` little-endian, zero-padded to
/// the pool's own 256-byte section alignment so its blob stays aligned in
/// the arena. Part of the blob layout the identity's version names
/// (`kIgnisSeqSnapshotFormatVersionFlashNext`): changing it bumps that.
const CONTEXT_BLOCK_BYTES: usize = 256;

fn leaf_error(context: &str, message: String) -> i32 {
    // hotpath-lint-allow: failure-only path, as CudaLeaf's own (GitHub #80).
    tracing::error!(name: "ignis.runtime.leaf_error", context, error = %message, "flash-next leaf error");
    -1
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

/// A shared prefix and the n-gram context at its end, which a claimant
/// hashes its first tokens from.
pub struct FlashNextPrefix {
    prefix: SeqPrefix<'static>,
    context: NgramContext,
}

/// A prompt checkpoint and the n-gram context at its opener.
pub struct FlashNextCheckpoint {
    checkpoint: SeqCheckpoint<'static>,
    context: NgramContext,
}

/// The leaf: the bound artifact, its device weights, the expert residency,
/// the n-gram table and the KV-RAM arena. Field order is drop order (the
/// device last).
pub struct FlashNextLeaf {
    /// Where residency's mirror and the n-gram table's counts are read
    /// (GitHub #301, #302): host memory only, so it may outlive the leaf.
    counters: Arc<FlashNextCounterSource>,
    arena: Option<Arc<HostArena>>,
    /// The blob layout version of the model's pool, read once it exists.
    layout_version: OnceLock<u32>,
    residency: DeviceResidency,
    table: NgramTable,
    plan: FlashNextPlan,
    geometry: FlashNextGeometry,
    config: ModelConfig,
    options: EngineOptions,
    /// The artifact's content hash, read at open. The leaf keeps no
    /// `Reader`: its map of the whole 71.8 GB file made every n-gram row
    /// read wait in NTFS for the one before it (~0.15 ms a read).
    artifact_hash: ArtifactHash,
    /// The weight arena and the device, freed together (`DeviceWeights`).
    weights: DeviceWeights,
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
    /// the pinned pool filled from the file), open the n-gram table and pin
    /// the KV-RAM arena (`options.kv_ram_arena_bytes`, none for 0).
    pub fn open(path: &Path, options: EngineOptions) -> Result<Self, String> {
        let options = options.normalized();
        let reader = Reader::open(path).map_err(|e| format!("open {}: {e:?}", path.display()))?;
        let geometry = FlashNextGeometry::qwen38_flash_next();
        let plan = flash_next::bind(&reader, &geometry).map_err(|e| format!("bind the Flash-Next artifact: {e:?}"))?;
        let config = ModelConfig::flash_next_from(&geometry);
        let weights = DeviceWeights::place(&reader, &plan)?;
        let mut residency = build_residency(path, &plan, &options)?;
        // Before the first step, and so before any graph captures one: the
        // last layer of every step writes the totals into host memory.
        let mirror = Arc::new(ResidencyMirror::new());
        residency.mirror(Arc::clone(&mirror))?;
        let ngram = config.ngram.ok_or("the Flash-Next topology has no n-gram embedding")?;
        let table = NgramTable::from_artifact(path, &reader, &plan, ngram, options.ngram)?;
        let counters = Arc::new(FlashNextCounterSource::new(mirror, residency.desc().capacity, table.counts()));
        let arena = match options.kv_ram_arena_bytes {
            0 => None,
            bytes => Some(HostArena::create(bytes)?),
        };
        let artifact_hash = ArtifactHash::from_bytes(reader.content_hash());
        // Unmapped here: everything after the load reads the file through
        // its own handles (the n-gram table, nothing else).
        drop(reader);
        Ok(Self {
            counters,
            arena,
            layout_version: OnceLock::new(),
            residency,
            table,
            plan,
            geometry,
            config,
            options,
            artifact_hash,
            weights,
        })
    }

    /// Where this load's counters are read, on any thread, with no call into
    /// the leaf: the steps do nothing for it.
    pub fn counter_source(&self) -> Arc<FlashNextCounterSource> {
        Arc::clone(&self.counters)
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

    /// The KV-RAM arena's capacity and the bytes its blobs hold; (0, 0)
    /// without one.
    pub fn kv_ram_arena_stats(&self) -> (u64, u64) {
        self.arena.as_ref().map_or((0, 0), |arena| arena.stats())
    }

    fn rows_for(&self, context: &mut NgramContext, tokens: &[u32]) -> Result<Vec<u8>, String> {
        let mut rows = vec![0u8; tokens.len() * self.table.token_bytes()];
        self.table.stage(context, tokens, &mut rows)?;
        Ok(rows)
    }

    /// A blob's bytes: the context block, then the pool's `pool_bytes`.
    fn blob_bytes(pool_bytes: u64) -> u64 {
        CONTEXT_BLOCK_BYTES as u64 + pool_bytes
    }

    /// Write `context` into the block at the head of `dst` and hand back
    /// the rest, where the pool's blob goes.
    fn write_context<'a>(context: &NgramContext, dst: &'a mut [u8]) -> Result<&'a mut [u8], i32> {
        let recent = context.recent();
        if dst.len() < CONTEXT_BLOCK_BYTES || 4 * (recent.len() + 1) > CONTEXT_BLOCK_BYTES {
            return Err(leaf_error("snapshot", format!("a {}-byte buffer holds no context block", dst.len())));
        }
        let (block, rest) = dst.split_at_mut(CONTEXT_BLOCK_BYTES);
        block.fill(0);
        block[..4].copy_from_slice(&(recent.len() as u32).to_le_bytes());
        for (word, token) in block[4..].chunks_exact_mut(4).zip(recent) {
            word.copy_from_slice(&token.to_le_bytes());
        }
        Ok(rest)
    }

    /// The context in the block at the head of `src`, and the pool's blob
    /// after it.
    fn read_context<'a>(&self, src: &'a [u8]) -> Result<(NgramContext, &'a [u8]), String> {
        if src.len() < CONTEXT_BLOCK_BYTES {
            return Err(format!("a {}-byte blob has no context block", src.len()));
        }
        let (block, rest) = src.split_at(CONTEXT_BLOCK_BYTES);
        let word = |i: usize| u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().expect("four bytes"));
        let count = word(0) as usize;
        if 4 * (count + 1) > CONTEXT_BLOCK_BYTES {
            return Err(format!("a context block of {count} tokens"));
        }
        let recent: Vec<u32> = (1..=count).map(word).collect();
        Ok((NgramContext::from_recent(self.table.hasher(), &recent)?, rest))
    }
}

impl StepLeaf for FlashNextLeaf {
    type Model = FlashNextModel;
    type Sequence = FlashNextSequence;
    type Prefix = FlashNextPrefix;
    type Checkpoint = FlashNextCheckpoint;
    type SnapshotBuf = ArenaBuffer;
    type Media = ();

    fn load_model(&self) -> Result<Self::Model, i32> {
        let o = &self.options;
        let model = model_load::load_flash_next(
            &self.plan,
            &self.geometry,
            self.weights.artifact(),
            o.prefill_chunk_tokens,
            o.max_context_tokens,
            o.kv_format,
            o.decode_lanes,
            &self.residency,
            o.speculation,
            None,
        )
        .map_err(|e| leaf_error("model load", e))?;
        let pool =
            SeqPool::create_with_speculation(&self.config, &o.pool_budget(), o.pool_backend())
                .map_err(|e| leaf_error("seq pool create", e))?;
        let _ = self.layout_version.set(pool.snapshot_format_version());
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
            free_vram_bytes: self.weights.device().free_bytes().unwrap_or(0),
            reserved: ReservedBytes {
                workspace: reserved.workspace_bytes + reserved.activation_bytes,
                sampling: reserved.sampling_bytes,
                decode_graph: reserved.decode_graph_bytes,
                lane_state: pool_stats.lane_state_bytes + pool_stats.indexer_bytes + pool_stats.ngram_conv_bytes,
                retained_slots: pool_stats.retained_state_bytes,
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
        model: &Self::Model,
        context_tokens: u32,
        prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32> {
        let seq = model
            .pool
            .alloc_shared(context_tokens, &prefix.prefix)
            .map_err(|e| leaf_error("shared sequence alloc", e))?;
        // Safety: as `allocate_sequence`'s.
        Ok(FlashNextSequence { seq: unsafe { seq.into_static() }, context: prefix.context.clone() })
    }

    fn publish_prefix(
        &self,
        _model: &Self::Model,
        sequence: &mut Self::Sequence,
        prefix_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Prefix, i32> {
        // The publisher stands exactly at `prefix_tokens` (the pool refuses
        // anywhere else), so its context is the one at the prefix's end.
        let prefix = sequence
            .seq
            .publish_prefix(prefix_tokens, retained_slot)
            .map_err(|e| leaf_error("prefix publish", e.to_string()))?;
        Ok(FlashNextPrefix { prefix, context: sequence.context.clone() })
    }

    fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {}

    fn capture_checkpoint(
        &self,
        _model: &Self::Model,
        sequence: &mut Self::Sequence,
        opener_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Checkpoint, i32> {
        let checkpoint = sequence
            .seq
            .capture_checkpoint(opener_tokens, retained_slot)
            .map_err(|e| leaf_error("checkpoint capture", e.to_string()))?;
        Ok(FlashNextCheckpoint { checkpoint, context: sequence.context.clone() })
    }

    fn allocate_sequence_from_checkpoint(
        &self,
        model: &Self::Model,
        context_tokens: u32,
        checkpoint: &Self::Checkpoint,
    ) -> Result<(Self::Sequence, u64), i32> {
        let seq = model
            .pool
            .alloc_from_checkpoint(context_tokens, &checkpoint.checkpoint)
            .map_err(|e| leaf_error("checkpoint claim", e))?;
        // The leaf's own measure of the claim, as CudaLeaf reports it.
        let micros = checkpoint.checkpoint.stats().last_claim_micros;
        // Safety: as `allocate_sequence`'s.
        let sequence = FlashNextSequence { seq: unsafe { seq.into_static() }, context: checkpoint.context.clone() };
        Ok((sequence, micros.round().max(0.0) as u64))
    }

    fn checkpoint_snapshot_bytes(&self, _model: &Self::Model, checkpoint: &Self::Checkpoint) -> Result<u64, i32> {
        checkpoint
            .checkpoint
            .snapshot_bytes()
            .map(Self::blob_bytes)
            .map_err(|e| leaf_error("checkpoint snapshot size", e.to_string()))
    }

    fn checkpoint_snapshot_into(
        &self,
        _model: &Self::Model,
        checkpoint: &Self::Checkpoint,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        let rest = Self::write_context(&checkpoint.context, dst)?;
        checkpoint.checkpoint.snapshot_into(rest).map_err(|e| leaf_error("checkpoint snapshot", e.to_string()))
    }

    fn prefix_snapshot_bytes(&self, _model: &Self::Model, prefix: &Self::Prefix) -> Result<u64, i32> {
        prefix
            .prefix
            .snapshot_bytes()
            .map(Self::blob_bytes)
            .map_err(|e| leaf_error("prefix snapshot size", e.to_string()))
    }

    fn prefix_snapshot_into(&self, _model: &Self::Model, prefix: &Self::Prefix, dst: &mut [u8]) -> Result<(), i32> {
        let rest = Self::write_context(&prefix.context, dst)?;
        prefix.prefix.snapshot_into(rest).map_err(|e| leaf_error("prefix snapshot", e.to_string()))
    }

    /// ADR 0029's identity (spec flash-next/05): this artifact's content
    /// hash, the KV format, the Flash-Next pool's own layout version, and no
    /// drafter. A 27B blob differs in the artifact and the version both.
    fn blob_identity(&self) -> BlobIdentity {
        BlobIdentity::of_load(
            self.artifact_hash,
            self.options.kv_format,
            None,
            self.layout_version.get().copied().unwrap_or(0),
        )
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
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        if attention.is_some() {
            return Err(leaf_error("prefill", "Qwen3.8-Flash-Next serves no attention readouts".to_string()));
        }
        // The context moves only with a span that ran: a failed one leaves
        // the sequence as it was.
        let mut context = sequence.context.clone();
        let rows = self.rows_for(&mut context, tokens).map_err(|e| leaf_error("n-gram rows", e))?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let permitted: Vec<i32> = permitted.iter().map(|&t| t as i32).collect();
        let drawn = step::prefill_flash_next(
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
        .map_err(|e| leaf_error("prefill", e))?;
        sequence.context = context;
        Ok(drawn)
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
        // Each lane's context moves only once the round has run: on an error
        // every sequence is left as it was.
        let mut contexts: Vec<NgramContext> = sequences.iter().map(|s| s.context.clone()).collect();
        let mut rows = vec![0u8; sequences.len() * self.table.token_bytes()];
        {
            let mut batch: Vec<(&mut NgramContext, &[u32])> =
                contexts.iter_mut().zip(&pending).map(|(c, t)| (c, &t[..])).collect();
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
        for (sequence, context) in sequences.iter_mut().zip(contexts) {
            sequence.context = context;
        }
        Ok(drawn
            .into_iter()
            .zip(lanes)
            .map(|((id, probability), lane)| {
                LaneRun::drawn(id as TokenId, (!lane.permitted.is_empty()).then_some(probability))
            })
            .collect())
    }

    fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32> {
        let arena = self
            .arena
            .as_ref()
            .ok_or_else(|| leaf_error("snapshot alloc", "no KV-RAM arena (--kv-host-pool-bytes 0)".to_string()))?;
        arena.alloc(bytes).map_err(|e| match e {
            // A fragmented arena is a refusal, not a failure (GitHub #213).
            PinnedAllocError::NoRoom => ignis_core::seq::NO_HOST_ROOM,
            PinnedAllocError::Failed(message) => leaf_error("snapshot alloc", message),
        })
    }

    fn host_blob_fits(&self, bytes: u64) -> bool {
        self.arena.as_ref().is_some_and(|arena| arena.fits(bytes))
    }

    fn snapshot_bytes(&self, _model: &Self::Model, sequence: &Self::Sequence) -> Result<u64, i32> {
        sequence.seq.snapshot_bytes().map(Self::blob_bytes).map_err(|e| leaf_error("snapshot size", e.to_string()))
    }

    fn snapshot_into(&self, _model: &Self::Model, sequence: &Self::Sequence, dst: &mut [u8]) -> Result<(), i32> {
        let rest = Self::write_context(&sequence.context, dst)?;
        sequence.seq.snapshot_into(rest).map_err(|e| leaf_error("snapshot", e.to_string()))
    }

    fn restore_sequence(&self, _model: &Self::Model, sequence: &mut Self::Sequence, src: &[u8]) -> Result<(), i32> {
        let (context, blob) = self.read_context(src).map_err(|e| leaf_error("restore", e))?;
        sequence.seq.restore(blob).map_err(|e| leaf_error("restore", e.to_string()))?;
        sequence.context = context;
        Ok(())
    }
}
