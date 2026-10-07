//! A Flash-Next load end to end, on the host side of the program (spec
//! flash-next/04, GitHub #302): the artifact bound and its non-expert
//! tensors on the device, the expert residency created and its pinned pool
//! filled, the n-gram table open, the model loaded over both, a sequence
//! pool with Flash-Next's sections, and the round graphs captured.
//!
//! [`FlashNextEngine`] is what the acceptance scorers and the GPU tests
//! drive: a span's logits teacher-forced from position 0, and greedy
//! generation through decode rounds. The server's runtime builds the same
//! pieces in the same order; ownership is the point of keeping them in one
//! struct -- the model borrows the residency and the device weights, so its
//! fields drop first (no process-wide singleton, spec flash-next/05).

use std::path::{Path, PathBuf};

use ignis_artifact::flash_next::{self, FlashNextGeometry, FlashNextPlan};
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{materialize, CudaDevice, Device, MaterializedArtifact, Reader};

use crate::compute::ModelConfig;
use crate::kv_format::KvFormat;
use crate::model_load::{load_flash_next, plan_flash_next_reservations, IgnisModelReservations, Model};
use crate::ngram::NgramContext;
use crate::ngram_table::{NgramTable, NgramTableOptions};
use crate::residency::device::{DeviceResidency, ResidencyDesc};
use crate::residency::load::{catalog, fill_expert_pool, pool_layout};
use crate::residency::{
    default_prefetch_budget_bytes, min_slots_per_class, plan_expert_cache, prefill_staging_ring_bytes,
    residency_table_bytes, warm_start_order, ExpertCacheRequest, ExpertTraffic, KClass,
};
use crate::seq::{SeqPool, SeqPoolBudget};
use crate::speculation::{FlashNextSpeculation, SpeculativeBackend};
use crate::step::{
    capture_decode_graphs, decode_flash_next, decode_flash_next_verify, prefill_flash_next, SamplingParams, VerifyLane,
};

/// The prefetch lookahead residency ranks per lane: the next layer's top 16.
pub const LOOKAHEAD_WIDTH: u32 = 16;

/// The lookahead residency ranks per token of a prefill chunk: the next
/// router's own top-k, since a chunk streams it unbudgeted (GitHub #306,
/// `docs/findings/2026-10-07-flash-next-agent-turn-tail.md`).
pub const PREFILL_LOOKAHEAD_WIDTH: u32 = crate::moe::TOP_K;

/// The decode lanes a Flash-Next load serves by default (spec flash-next/04:
/// three agents); the leaf's own default for a load option of 0.
pub const DEFAULT_DECODE_LANES: u32 = 3;

/// What a Flash-Next engine is loaded with.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub prefill_chunk_tokens: u32,
    pub max_context_tokens: u32,
    pub kv_format: KvFormat,
    pub decode_lanes: u32,
    /// The VRAM expert cache, split into the eight K-class pools.
    pub expert_cache_bytes: u64,
    pub ngram: NgramTableOptions,
    /// Persist the selected n-gram hot rows between process loads.
    pub ngram_cache: crate::ngram_cache::PersistenceOptions,
    /// Capture the decode rounds' graphs after the pool exists.
    pub capture_graphs: bool,
    /// Prompt reuse's retained slots (spec flash-next/05): on the device
    /// (`--retained-device`, each one an image's worth of expert cache) and
    /// in the pool's pinned host block (`--retained-host`). Both 0 for an
    /// engine that retains nothing.
    pub retained_device_slots: u32,
    pub retained_host_slots: u32,
    /// The KV-RAM arena this load pins for its host tier
    /// (`--kv-host-pool-bytes`); 0 creates none.
    pub kv_ram_arena_bytes: u64,
    /// Speculative decoding (spec flash-next/07): the verify round at the
    /// load's windows; `None` runs one-token rounds only.
    pub speculation: Option<FlashNextSpeculation>,
}

impl EngineOptions {
    /// These options with the decode lanes normalized once: 0 is the
    /// default, as the leaf reads it -- so the pool, the residency and the
    /// load agree on one count.
    pub fn normalized(mut self) -> Self {
        if self.decode_lanes == 0 {
            self.decode_lanes = DEFAULT_DECODE_LANES;
        }
        self
    }

    /// The speculative backend the pool is built for: an MTP load's head is
    /// one more attention section of it; nothing else changes the pool.
    pub fn pool_backend(&self) -> Option<SpeculativeBackend> {
        self.speculation.map(|s| s.backend()).filter(|&b| b == SpeculativeBackend::Mtp)
    }

    /// Every retained slot, device and host: the scheduler's slot indices,
    /// the device ones first.
    pub fn retained_slots(&self) -> u32 {
        self.retained_device_slots + self.retained_host_slots
    }

    /// The sequence pool these options build: every lane's whole context,
    /// plus one page per retained slot -- a checkpoint keeps the page its
    /// opener ends inside, as on the 27B (`VramRequest::retained_slots`).
    pub fn pool_budget(&self) -> SeqPoolBudget {
        SeqPoolBudget {
            kv_format: self.kv_format,
            kv_page_group_count: self.max_context_tokens.div_ceil(64) * self.decode_lanes + self.retained_slots(),
            max_context_tokens: self.max_context_tokens,
            slot_count: self.decode_lanes,
            retained_slot_count: self.retained_device_slots,
            retained_host_slot_count: self.retained_host_slots,
        }
    }
}

/// The device bytes a pool of `budget` holds for `config`: its KV arena,
/// the lanes' state, the device retained slots' images, the residual window
/// and the indexer and n-gram sections (the indexer's keys, 768 bytes per
/// token of context, are part of the context's price, not the KV line
/// alone). The host retained slots are host memory, in none of these.
pub fn pool_device_bytes(
    config: &ModelConfig,
    budget: &SeqPoolBudget,
    backend: Option<SpeculativeBackend>,
) -> Result<u64, String> {
    let plan = SeqPool::plan(config, budget, backend)?;
    Ok(plan.kv_bytes
        + plan.lane_state_bytes
        + plan.retained_state_bytes
        + plan.hq_residual_bytes
        + plan.indexer_bytes
        + plan.ngram_conv_bytes)
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            prefill_chunk_tokens: 2048,
            max_context_tokens: 32 * 1024,
            kv_format: KvFormat::HqE8_2b,
            decode_lanes: DEFAULT_DECODE_LANES,
            expert_cache_bytes: 12 << 30,
            ngram: NgramTableOptions::default(),
            ngram_cache: crate::ngram_cache::PersistenceOptions::default(),
            capture_graphs: true,
            retained_device_slots: 0,
            retained_host_slots: 0,
            kv_ram_arena_bytes: 0,
            speculation: None,
        }
    }
}

/// The calibration selections per (layer, expert) the converter recorded
/// in the artifact's sidecar (`<artifact>.conversion.json`, its
/// `expert_traffic`), or `None` when the sidecar has none.
pub fn sidecar_traffic(artifact: &Path) -> Option<Vec<u64>> {
    let text = std::fs::read_to_string(ignis_artifact::packer::sidecar_path(artifact)).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json.get("expert_traffic")?
        .get("layers")?
        .as_array()?
        .iter()
        .map(|layer| layer.get("counts")?.as_array()?.iter().map(|c| c.as_u64()).collect::<Option<Vec<u64>>>())
        .collect::<Option<Vec<Vec<u64>>>>()
        .map(|layers| layers.concat())
}

/// The expert residency of the artifact at `path` bound as `plan`, for a
/// load of `options` (spec flash-next/03): the eight K-class pools split from
/// `expert_cache_bytes` the way one LRU over the calibration traffic would
/// hold them (the sidecar's; uniform when it has none), the pinned pool
/// filled from the file, and the warm start -- the hottest projections,
/// hottest first, up to each pool -- before the first step.
pub fn build_residency(path: &Path, plan: &FlashNextPlan, options: &EngineOptions) -> Result<DeviceResidency, String> {
    let index = &plan.experts;
    let cat = catalog(index)?;
    let layout = pool_layout(index);
    let (layers, experts) = (u64::from(cat.layers()), u64::from(cat.experts()));
    let counts = sidecar_traffic(path)
        .filter(|counts| counts.len() == (layers * experts) as usize)
        .unwrap_or_else(|| vec![1; (layers * experts) as usize]);
    let traffic = ExpertTraffic::new(&cat, counts).map_err(|e| e.to_string())?;
    let max_tokens = options.prefill_chunk_tokens.max(options.decode_lanes);
    let staging_ring = prefill_staging_ring_bytes(&cat);
    let tables = residency_table_bytes(layers, experts, u64::from(max_tokens), u64::from(LOOKAHEAD_WIDTH));
    let cache = plan_expert_cache(&ExpertCacheRequest {
        budget_bytes: options.expert_cache_bytes + staging_ring + tables,
        planned_bytes: 0,
        staging_ring_bytes: staging_ring,
        table_bytes: tables,
        floor_bytes: 0,
        catalog: &cat,
        traffic: &traffic,
        min_slots: min_slots_per_class(options.decode_lanes, 10, LOOKAHEAD_WIDTH),
    })
    .map_err(|e| e.to_string())?;
    let desc = ResidencyDesc {
        layers: layers as u32,
        experts: experts as u32,
        capacity: cache.capacity(),
        record_bytes: std::array::from_fn(|i| cat.slot_bytes(KClass::ALL[i])),
        max_tokens,
        lookahead_width: LOOKAHEAD_WIDTH,
        prefill_lookahead_width: PREFILL_LOOKAHEAD_WIDTH,
        prefetch_budget_bytes: default_prefetch_budget_bytes(options.decode_lanes, &cat),
        staging_half_bytes: staging_ring / 2,
        host_pool_bytes: layout.bytes,
        copy_blocks: 16,
        report: 0,
    };
    let mut residency = DeviceResidency::new(&desc, &layout.k2, &layout.offsets)?;
    let mut file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    fill_expert_pool(&mut file, index, residency.host_pool_mut()).map_err(|e| format!("fill the expert pool: {e}"))?;
    residency.warm_start(&warm_start_order(&traffic))?;
    Ok(residency)
}

/// The artifact's non-expert tensors placed on the device, and the device
/// they live on. Dropping it frees the weight arena: `MaterializedArtifact`
/// has no `Drop` of its own, since its release needs the device that placed
/// it, and without this every load in a process left its ~5 GB of weights
/// behind (the fourth load of `flash_next_forward_gpu` ran out of VRAM).
pub struct DeviceWeights {
    artifact: MaterializedArtifact,
    device: CudaDevice,
}

impl DeviceWeights {
    /// Place the device tensors of `plan` on device 0.
    pub fn place(reader: &Reader, plan: &FlashNextPlan) -> Result<Self, String> {
        let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
        let artifact = materialize(reader, &plan.plan, &mut device, None).map_err(|e| format!("materialize: {e}"))?;
        Ok(Self { artifact, device })
    }

    pub fn artifact(&self) -> &MaterializedArtifact {
        &self.artifact
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

impl Drop for DeviceWeights {
    fn drop(&mut self) {
        let _ = self.artifact.release_arena(&mut self.device);
    }
}

/// A loaded Flash-Next model and everything it runs on. Field order is drop
/// order: the model goes before the pool it captured graphs against, the
/// residency and device weights it borrows, and the device they live on.
pub struct FlashNextEngine {
    model: Model,
    pool: SeqPool,
    // An MTP load's head (spec flash-next/07): the model points into it.
    _mtp: Option<crate::flash_next_mtp::MtpHead>,
    residency: DeviceResidency,
    table: NgramTable,
    _weights: DeviceWeights,
    config: ModelConfig,
    options: EngineOptions,
    graphs_ready: u32,
    graph_error: Option<String>,
    planned: IgnisModelReservations,
}

impl FlashNextEngine {
    /// Load the artifact in `dir` (`Qwen3.8-Flash-Next-ignis`, spec 01).
    pub fn load(dir: &Path, options: EngineOptions) -> Result<Self, String> {
        let options = options.normalized();
        let path: PathBuf = dir.join(ARTIFACT_FILE_NAME);
        let reader = Reader::open(&path).map_err(|e| format!("open {}: {e:?}", path.display()))?;
        let geometry = FlashNextGeometry::qwen38_flash_next();
        let plan = flash_next::bind(&reader, &geometry).map_err(|e| format!("bind the Flash-Next artifact: {e:?}"))?;
        let config = ModelConfig::flash_next_from(&geometry);
        // Spec flash-next/07: the head's companion, pinned to this container before any byte of it
        // is read.
        let mtp = match options.pool_backend() {
            Some(_) => Some(crate::flash_next_mtp::MtpHead::load(
                &crate::flash_next_mtp::companion_path(dir),
                &reader,
                &geometry,
            )?),
            None => None,
        };
        let planned = plan_flash_next_reservations(
            &plan,
            &geometry,
            options.prefill_chunk_tokens,
            options.max_context_tokens,
            options.kv_format,
            options.decode_lanes,
            options.speculation,
            mtp.as_ref().map(|head| head.plan()),
        )?;

        let weights = DeviceWeights::place(&reader, &plan)?;
        let residency = build_residency(&path, &plan, &options)?;
        let ngram = config.ngram.ok_or("the Flash-Next topology has no n-gram embedding")?;
        let table = NgramTable::from_cached_artifact(
            &path, reader, &plan, ngram, options.ngram, &options.ngram_cache,
        )?;
        let model = load_flash_next(
            &plan,
            &geometry,
            weights.artifact(),
            options.prefill_chunk_tokens,
            options.max_context_tokens,
            options.kv_format,
            options.decode_lanes,
            &residency,
            options.speculation,
            mtp.as_ref(),
        )?;
        // The pool is planned against what is free before it is created, so a
        // context the card cannot hold is a refusal naming it, not an
        // allocation failure.
        let budget = options.pool_budget();
        let pool_bytes = pool_device_bytes(&config, &budget, options.pool_backend())?;
        let free = weights.device().free_bytes().unwrap_or(u64::MAX);
        if pool_bytes > free {
            return Err(format!(
                "the sequence pool of {} lanes of {} tokens needs {pool_bytes} bytes (KV, lane state, the \
                 indexer's keys and the n-gram state), and {free} are free after the weights, the program and \
                 the expert cache: shorten max_context_tokens or the expert cache",
                options.decode_lanes, options.max_context_tokens
            ));
        }
        let pool = SeqPool::create_with_speculation(&config, &budget, options.pool_backend())?;
        let (graphs_ready, graph_error) = if options.capture_graphs {
            let capture = capture_decode_graphs(&model, &pool)?;
            let mask = (1..=options.decode_lanes).filter(|&w| capture.is_ready(w)).fold(0, |mask, w| mask | 1 << (w - 1));
            let complete = mask == (1u32 << options.decode_lanes) - 1;
            (mask, (!complete).then(crate::step::last_decode_graph_error))
        } else {
            (0, None)
        };
        Ok(Self {
            model,
            pool,
            _mtp: mtp,
            residency,
            table,
            _weights: weights,
            config,
            options,
            graphs_ready,
            graph_error,
            planned,
        })
    }

    pub fn vocab(&self) -> usize {
        self.config.vocab as usize
    }

    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    /// Bit `w - 1` set when the round of `w` lanes replays a graph.
    pub fn graphs_ready(&self) -> u32 {
        self.graphs_ready
    }

    /// Bit `w - 1` set when the verify round of `w` lanes replays its pass
    /// and commit graphs (spec flash-next/07); 0 without speculation.
    pub fn verify_graphs_ready(&self) -> Result<u32, String> {
        crate::step::program_stats(&self.model, &self.pool).map(|stats| stats.verify_graph_ready_mask)
    }

    /// Why a width's graph did not capture (the leaf's last capture error),
    /// when one did not.
    pub fn graph_error(&self) -> Option<&str> {
        self.graph_error.as_deref()
    }

    /// The reservations the load planned before anything was on the device,
    /// and what the loaded model holds (`ignis_model_stats`): equal, or the
    /// plan no longer describes the load.
    pub fn planned(&self) -> IgnisModelReservations {
        self.planned
    }

    pub fn reserved(&self) -> IgnisModelReservations {
        self.model.stats().reserved
    }

    pub fn residency(&self) -> &DeviceResidency {
        &self.residency
    }

    pub fn ngram_table(&self) -> &NgramTable {
        &self.table
    }

    /// Prefill `tokens` from position 0 on a fresh sequence and hand every
    /// position's BF16 logits to `sink(first_row, rows)`, a chunk at a time.
    pub fn span_logits(
        &mut self,
        tokens: &[u32],
        sink: &mut dyn FnMut(usize, &[u16]) -> Result<(), String>,
    ) -> Result<(), String> {
        if tokens.len() > self.options.max_context_tokens as usize {
            return Err(format!("{} tokens past the load's {} context", tokens.len(), self.options.max_context_tokens));
        }
        let vocab = self.vocab();
        let mut seq = self.pool.alloc(self.options.max_context_tokens)?;
        let mut context = self.table.new_context();
        let chunk = self.options.prefill_chunk_tokens as usize;
        let mut rows = Vec::new();
        let mut logits = Vec::new();
        for (index, piece) in tokens.chunks(chunk).enumerate() {
            rows.resize(piece.len() * self.table.token_bytes(), 0);
            self.table.stage(&mut context, piece, &mut rows)?;
            logits.resize(piece.len() * vocab, 0);
            let ids: Vec<i32> = piece.iter().map(|&t| t as i32).collect();
            let start = (index * chunk) as u64;
            prefill_flash_next(
                &self.model,
                &self.pool,
                &mut seq,
                &ids,
                start,
                SamplingParams::greedy(),
                &[],
                &rows,
                Some(&mut logits),
                None,
            )?;
            sink(index * chunk, &logits)?;
        }
        Ok(())
    }

    /// The last position's logits of `tokens` prefilled from position 0 on a
    /// fresh sequence, as floats: the draw's own row, which a span's row at
    /// the same position equals bit for bit where the two cut the same chunks.
    pub fn last_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut seq = self.pool.alloc(self.options.max_context_tokens)?;
        let mut context = self.table.new_context();
        let mut rows = vec![0u8; tokens.len() * self.table.token_bytes()];
        self.table.stage(&mut context, tokens, &mut rows)?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut logits = vec![0f32; self.vocab()];
        prefill_flash_next(
            &self.model,
            &self.pool,
            &mut seq,
            &ids,
            0,
            SamplingParams::greedy(),
            &[],
            &rows,
            None,
            Some(&mut logits),
        )?;
        Ok(logits)
    }

    /// `tokens` prefilled from position 0 on a fresh sequence, returned so a
    /// test can read its K/V rows back (`Seq::capture_kv_rows_for_test`):
    /// the real-row fixture of spec flash-next/04 acceptance 7. One prefill
    /// call, so at most one chunk.
    #[cfg(feature = "kv-capture")]
    pub fn prefill_for_kv_capture(&mut self, tokens: &[u32]) -> Result<crate::seq::Seq<'_>, String> {
        if tokens.len() > self.options.prefill_chunk_tokens as usize {
            return Err(format!("{} tokens past the load's {}-token chunk", tokens.len(), self.options.prefill_chunk_tokens));
        }
        let mut seq = self.pool.alloc(self.options.max_context_tokens)?;
        let mut context = self.table.new_context();
        let mut rows = vec![0u8; tokens.len() * self.table.token_bytes()];
        self.table.stage(&mut context, tokens, &mut rows)?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        prefill_flash_next(&self.model, &self.pool, &mut seq, &ids, 0, SamplingParams::greedy(), &[], &rows, None, None)?;
        Ok(seq)
    }

    /// `tokens` prefilled from position 0 on a fresh sequence with the
    /// residual-stack tap armed (spec flash-next/07 phase A): every
    /// position's final pre-mixer stack, and the engine's own greedy pick
    /// after it with that row's margin -- what the MTP head is fed and what a
    /// verify would accept, from one prefill.
    #[cfg(feature = "residual-tap")]
    pub fn tapped_span(&mut self, tokens: &[u32]) -> Result<TappedSpan, String> {
        let geometry = FlashNextGeometry::qwen38_flash_next();
        let width = (geometry.hc_streams * geometry.hidden) as usize;
        let vocab = self.vocab();
        let mut argmax = Vec::with_capacity(tokens.len());
        let mut margin = Vec::with_capacity(tokens.len());
        let (result, stacks, written) = crate::residual_tap::with_residual_tap(0, tokens.len(), width, || {
            self.span_logits(tokens, &mut |_, rows| {
                for row in rows.chunks_exact(vocab) {
                    let (best, gap) = top_with_margin(row);
                    argmax.push(best);
                    margin.push(gap);
                }
                Ok(())
            })
        })?;
        result?;
        if written != tokens.len() || argmax.len() != tokens.len() {
            return Err(format!(
                "the tap wrote {written} rows and the span {} picks for {} tokens",
                argmax.len(),
                tokens.len()
            ));
        }
        Ok(TappedSpan { width, stacks, argmax, margin })
    }

    /// Greedy generation of `count` tokens after `prompt`, on lanes of one
    /// prompt each (all of them at once: one round of `prompts.len()` lanes
    /// per token).
    pub fn generate(&mut self, prompts: &[Vec<u32>], count: usize) -> Result<Vec<Vec<u32>>, String> {
        self.generate_timed(prompts, count).map(|(tokens, _)| tokens)
    }

    /// [`generate`](Self::generate), with each round's wall time: the host's
    /// n-gram staging of the round's tokens plus the round itself, as a
    /// serving loop pays it (spec flash-next/07 phase A's cost per row).
    pub fn generate_timed(
        &mut self,
        prompts: &[Vec<u32>],
        count: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<std::time::Duration>), String> {
        if prompts.is_empty() || prompts.len() > self.options.decode_lanes as usize {
            return Err(format!("{} prompts on {} lanes", prompts.len(), self.options.decode_lanes));
        }
        let mut seqs = Vec::with_capacity(prompts.len());
        let mut contexts = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let mut seq = self.pool.alloc(self.options.max_context_tokens)?;
            let mut context = self.table.new_context();
            let mut rows = vec![0u8; prompt.len() * self.table.token_bytes()];
            self.table.stage(&mut context, prompt, &mut rows)?;
            let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            prefill_flash_next(&self.model, &self.pool, &mut seq, &ids, 0, SamplingParams::greedy(), &[], &rows, None, None)?;
            seqs.push(seq);
            contexts.push(context);
        }
        let mut out = vec![Vec::with_capacity(count); prompts.len()];
        let mut times = Vec::with_capacity(count);
        let mut rows = vec![0u8; prompts.len() * self.table.token_bytes()];
        for _ in 0..count {
            let began = std::time::Instant::now();
            let pending: Vec<[u32; 1]> = seqs
                .iter()
                .map(|seq| seq.pending_token().map(|t| [t as u32]).ok_or("a lane has no pending token"))
                .collect::<Result<_, _>>()?;
            let mut lanes: Vec<(&mut crate::ngram::NgramContext, &[u32])> =
                contexts.iter_mut().zip(pending.iter()).map(|(c, t)| (c, &t[..])).collect();
            self.table.begin_batch(&mut lanes)?.finish(&mut rows)?;
            let mut handles: Vec<_> = seqs.iter_mut().collect();
            let greedy = vec![(SamplingParams::greedy(), &[][..]); handles.len()];
            let emitted = decode_flash_next(&self.model, &self.pool, &mut handles, &greedy, &rows)?;
            times.push(began.elapsed());
            for (lane, (token, _)) in emitted.into_iter().enumerate() {
                out[lane].push(token as u32);
            }
        }
        Ok((out, times))
    }

    /// Greedy generation of `count` tokens after each prompt with the
    /// load's speculation (spec flash-next/07), on lanes of one prompt each.
    /// Every round, `drafter(lane, emitted, window)` proposes the lane's
    /// drafts (at most `window`, the width's; empty: extent 0); with no
    /// drafter an MTP load verifies its head's own, made at the end of the
    /// lane's last round (a verify-only load then verifies none). A round of
    /// window 0, or whose lanes cannot all fit the positions it writes, runs
    /// as a one-token round.
    pub fn generate_speculative(
        &self,
        prompts: &[Vec<u32>],
        count: usize,
        mut drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>,
    ) -> Result<SpeculativeRun<'_>, String> {
        let speculation = self.options.speculation.ok_or("this engine was loaded without speculation")?;
        if prompts.is_empty() || prompts.len() > self.options.decode_lanes as usize {
            return Err(format!("{} prompts on {} lanes", prompts.len(), self.options.decode_lanes));
        }
        let mut seqs = Vec::with_capacity(prompts.len());
        let mut contexts = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let mut seq = self.pool.alloc(self.options.max_context_tokens)?;
            let mut context = self.table.new_context();
            let mut rows = vec![0u8; prompt.len() * self.table.token_bytes()];
            self.table.stage(&mut context, prompt, &mut rows)?;
            let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            prefill_flash_next(&self.model, &self.pool, &mut seq, &ids, 0, SamplingParams::greedy(), &[], &rows, None, None)?;
            seqs.push(seq);
            contexts.push(context);
        }
        let mut out = vec![Vec::with_capacity(count); prompts.len()];
        let mut rounds = Vec::new();
        let mut proposals = Vec::new();
        let mut times = Vec::new();
        let context_tokens = self.table.new_context().recent().len();
        let mtp = speculation.backend() == SpeculativeBackend::Mtp;
        // An MTP load's drafts for each lane's next round: none after a prefill or a one-token round.
        let mut head_drafts: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        loop {
            let active: Vec<usize> = (0..prompts.len()).filter(|&i| out[i].len() < count).collect();
            if active.is_empty() {
                break;
            }
            let began = std::time::Instant::now();
            let width = active.len() as u32;
            let window = speculation.window(width);
            let anchors: Vec<u32> = active
                .iter()
                .map(|&i| seqs[i].pending_token().map(|t| t as u32).ok_or("a lane has no pending token"))
                .collect::<Result<_, _>>()?;
            let written = if mtp { 2 * window } else { window + 1 };
            let fits = active
                .iter()
                .all(|&i| seqs[i].stats().position + u64::from(written) <= u64::from(self.options.max_context_tokens));
            let mut round = Vec::with_capacity(active.len());
            if window == 0 || !fits {
                let mut rows = vec![0u8; active.len() * self.table.token_bytes()];
                let mut lane_contexts: Vec<NgramContext> = active.iter().map(|&i| contexts[i].clone()).collect();
                {
                    let mut batch: Vec<(&mut NgramContext, &[u32])> = lane_contexts
                        .iter_mut()
                        .zip(&anchors)
                        .map(|(c, t)| (c, std::slice::from_ref(t)))
                        .collect();
                    self.table.begin_batch(&mut batch)?.finish(&mut rows)?;
                }
                let mut handles: Vec<_> =
                    seqs.iter_mut().enumerate().filter(|(i, _)| active.contains(i)).map(|(_, s)| s).collect();
                let greedy = vec![(SamplingParams::greedy(), &[][..]); handles.len()];
                let emitted = decode_flash_next(&self.model, &self.pool, &mut handles, &greedy, &rows)?;
                for ((&lane, (token, _)), context) in active.iter().zip(emitted).zip(lane_contexts) {
                    out[lane].push(token as u32);
                    contexts[lane] = context;
                    head_drafts[lane].clear();
                    round.push(LaneRound { extent: 0, committed: 1 });
                }
            } else {
                let columns = window as usize + 1;
                let drafts: Vec<Vec<i32>> = active
                    .iter()
                    .map(|&i| {
                        let head = std::mem::take(&mut head_drafts[i]);
                        let mut proposed = match drafter.as_mut() {
                            Some(drafter) => drafter(i, &out[i], window),
                            None => head,
                        };
                        proposed.truncate(window as usize);
                        proposed.into_iter().map(|t| t as i32).collect()
                    })
                    .collect();
                // Each lane's columns hashed as if committed in order: the
                // anchor, its drafts, then the anchor again up to the window.
                let column_tokens: Vec<Vec<u32>> = (0..active.len())
                    .map(|a| {
                        let mut tokens = vec![anchors[a]];
                        tokens.extend(drafts[a].iter().map(|&t| t as u32));
                        tokens.resize(columns, anchors[a]);
                        tokens
                    })
                    .collect();
                let mut rows = vec![0u8; active.len() * columns * self.table.token_bytes()];
                {
                    let mut scratch: Vec<NgramContext> = active.iter().map(|&i| contexts[i].clone()).collect();
                    let mut batch: Vec<(&mut NgramContext, &[u32])> =
                        scratch.iter_mut().zip(&column_tokens).map(|(c, t)| (c, &t[..])).collect();
                    self.table.begin_batch(&mut batch)?.finish(&mut rows)?;
                }
                let lanes: Vec<VerifyLane<'_>> = active
                    .iter()
                    .zip(&drafts)
                    .map(|(&i, d)| VerifyLane {
                        sampling: SamplingParams::greedy(),
                        remaining_tokens: (count - out[i].len()) as u32,
                        stop_ids: &[],
                        drafts: d,
                    })
                    .collect();
                let mut handles: Vec<_> =
                    seqs.iter_mut().enumerate().filter(|(i, _)| active.contains(i)).map(|(_, s)| s).collect();
                let runs = decode_flash_next_verify(&self.model, &self.pool, &mut handles, &lanes, window, &rows)?;
                for (&lane, run) in active.iter().zip(runs) {
                    let committed: Vec<u32> = run.tokens.iter().map(|&t| t as u32).collect();
                    // The context after the run: its last tokens.
                    let mut history = contexts[lane].recent().to_vec();
                    history.extend_from_slice(&committed);
                    contexts[lane] =
                        NgramContext::from_recent(self.table.hasher(), &history[history.len() - context_tokens..])?;
                    round.push(LaneRound { extent: run.extent, committed: committed.len() as u32 });
                    out[lane].extend_from_slice(&committed);
                    head_drafts[lane] = run.next_drafts.iter().map(|&t| t as u32).collect();
                }
            }
            times.push(began.elapsed());
            rounds.push(round);
            proposals.push(active.iter().map(|&i| head_drafts[i].clone()).collect());
        }
        Ok(SpeculativeRun { tokens: out, rounds, head_drafts: proposals, times, sequences: seqs, contexts })
    }

    /// The logits after one more token on a sequence `generate_speculative`
    /// left standing: its pending token prefilled at its frontier, through
    /// every state section the next step reads (spec flash-next/07's probe
    /// of "the state a later step reads"). Moves the sequence one token on.
    pub fn probe_logits(&self, seq: &mut crate::seq::Seq<'_>, context: &mut NgramContext) -> Result<Vec<f32>, String> {
        let pending = seq.pending_token().ok_or("the sequence has no pending token")?;
        let mut rows = vec![0u8; self.table.token_bytes()];
        self.table.stage(context, &[pending as u32], &mut rows)?;
        let mut logits = vec![0f32; self.vocab()];
        let position = seq.stats().position;
        prefill_flash_next(
            &self.model,
            &self.pool,
            seq,
            &[pending],
            position,
            SamplingParams::greedy(),
            &[],
            &rows,
            None,
            Some(&mut logits),
        )?;
        Ok(logits)
    }
}

/// One lane's share of a speculative round ([`FlashNextEngine::generate_speculative`]):
/// the drafts it verified (its extent; 0 in a one-token round) and the tokens
/// it committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneRound {
    pub extent: u32,
    pub committed: u32,
}

/// What [`FlashNextEngine::generate_speculative`] produced: each lane's
/// tokens, every round's lanes (the lanes still generating, in prompt order)
/// and wall time, and the lanes' sequences, standing after their last
/// committed token with their next one pending.
pub struct SpeculativeRun<'p> {
    pub tokens: Vec<Vec<u32>>,
    pub rounds: Vec<Vec<LaneRound>>,
    /// Every round's lanes' drafts the MTP head made for their next round,
    /// whether or not a drafter overrides them (empty after a one-token round
    /// and on a verify-only load).
    pub head_drafts: Vec<Vec<Vec<u32>>>,
    pub times: Vec<std::time::Duration>,
    pub sequences: Vec<crate::seq::Seq<'p>>,
    pub contexts: Vec<NgramContext>,
}

/// What [`FlashNextEngine::tapped_span`] captured: `stacks` is BF16 bits
/// `[tokens][width]`, stream-major (`width` = streams x hidden); `argmax[p]`
/// is the greedy pick after position p and `margin[p]` its lead over the
/// runner-up, in logits.
#[cfg(feature = "residual-tap")]
pub struct TappedSpan {
    pub width: usize,
    pub stacks: Vec<u16>,
    pub argmax: Vec<u32>,
    pub margin: Vec<f32>,
}

/// One BF16 row's argmax (the lowest id among equals) and its lead over the
/// runner-up.
#[cfg_attr(not(feature = "residual-tap"), allow(dead_code))]
fn top_with_margin(row: &[u16]) -> (u32, f32) {
    let (mut best, mut first, mut second) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (id, &bits) in row.iter().enumerate() {
        let v = f32::from_bits(u32::from(bits) << 16);
        if v > first {
            second = first;
            first = v;
            best = id as u32;
        } else if v > second {
            second = v;
        }
    }
    (best, first - second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// The pick is the first of equal maxima, and the margin is the lead over
    /// the runner-up (zero on a tie).
    #[test]
    fn the_span_pick_is_the_first_maximum_with_its_lead() {
        let row: Vec<u16> = [1.0, 3.0, -2.0, 2.5].iter().map(|&v| bf16(v)).collect();
        assert_eq!(top_with_margin(&row), (1, 0.5));
        let tie: Vec<u16> = [3.0, 1.0, 3.0].iter().map(|&v| bf16(v)).collect();
        assert_eq!(top_with_margin(&tie), (0, 0.0));
    }

    /// The sidecar's `expert_traffic`, layer after layer, as the residency's
    /// split reads it; none without a sidecar or the section.
    #[test]
    fn the_sidecar_traffic_is_read_layer_major() {
        let dir = std::env::temp_dir().join(format!("ignis-sidecar-traffic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("model.ninfer");
        assert_eq!(sidecar_traffic(&artifact), None, "no sidecar");
        let sidecar = ignis_artifact::packer::sidecar_path(&artifact);
        std::fs::write(
            &sidecar,
            r#"{"expert_traffic": {"layers": [{"layer": 0, "counts": [1, 2]}, {"layer": 1, "counts": [3, 4]}]}}"#,
        )
        .unwrap();
        assert_eq!(sidecar_traffic(&artifact), Some(vec![1, 2, 3, 4]));
        std::fs::write(&sidecar, r#"{"schema": "flash-next-converter-v1"}"#).unwrap();
        assert_eq!(sidecar_traffic(&artifact), None, "a sidecar without the section");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
