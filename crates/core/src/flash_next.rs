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
use crate::model_load::{load_flash_next, plan_flash_next_reservations, FlashNextLoadShape, IgnisModelReservations, Model};
use crate::ngram::NgramContext;
use crate::ngram_table::{NgramTable, NgramTableOptions};
use crate::residency::device::{budget_words, DeviceResidency, ResidencyDesc};
use crate::residency::load::{
    catalog, fill_expert_pool, fill_expert_pool_except, pool_layout, pool_layout_without, stream_expert_layers,
    stream_pinned_records,
};
use crate::residency::{
    default_prefetch_budget, holds_every_expert, min_slots_per_class, pin_hot_experts, plan_expert_cache,
    residency_table_bytes, staging_ring_bytes_for, warm_start_order, ExpertCachePlan, ExpertCachePlanError,
    ExpertCacheRequest, ExpertCatalog, ExpertTraffic, KClass, PinRefused, PinnedExperts, EXPERT_CACHE_FLOOR_BYTES,
};
use crate::rope_scaling::RopeScaling;
use crate::seq::{SeqPool, SeqPoolBudget};
use crate::speculation::{FlashNextSpeculation, SpeculativeBackend};
use crate::step::{
    capture_decode_graphs, decode_flash_next, decode_flash_next_verify, prefill_flash_next, MediaEmbedding,
    MultimodalPrefill, Redirect, SamplingParams, VerifyLane,
};

/// The prefetch lookahead residency ranks per lane: the next layer's top 16.
pub const LOOKAHEAD_WIDTH: u32 = 16;

/// The lookahead residency ranks per token of a prefill chunk: the next
/// router's own top-k, since a chunk streams it unbudgeted (GitHub #306,
/// `docs/findings/2026-10-07-flash-next-agent-turn-tail.md`).
pub const PREFILL_LOOKAHEAD_WIDTH: u32 = crate::moe::TOP_K;

/// The decode lanes a Flash-Next load serves by default (spec flash-next/04:
/// three agents); the leaf's own default for a load option of 0. The
/// family's own figure (GitHub #336), where the 27B's 8 sits beside it.
pub const DEFAULT_DECODE_LANES: u32 = crate::compute::ModelFamily::FlashNext.default_decode_lanes();

/// What a Flash-Next engine is loaded with.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub prefill_chunk_tokens: u32,
    pub max_context_tokens: u32,
    pub kv_format: KvFormat,
    pub decode_lanes: u32,
    /// The VRAM expert cache, split into the eight K-class pools.
    pub expert_cache_bytes: u64,
    /// The pinned hot set's budget (`--vram-pinned-expert-bytes`, GitHub
    /// #329 item 12): the calibration-hottest experts held in the cache's
    /// pools outside the LRU, with no copy in the host pool. 0 pins none; a
    /// device-only load ignores it ([`pinned_experts`]).
    pub pinned_expert_bytes: u64,
    /// The KV pool's pages, shared by the lanes (ADR 0045): what the server's
    /// VRAM plan gave the pool (`ignis_core::plan_vram`, 524,288 tokens when
    /// experts stream). `None` is every lane's whole context and a page per
    /// retained slot -- not a policy, but what an engine that allocates a
    /// whole context per lane and never plans needs ([`FlashNextEngine`] and
    /// the leaf's GPU tests).
    pub kv_pool_pages: Option<u32>,
    pub ngram: NgramTableOptions,
    /// Persist the selected n-gram hot rows between process loads.
    pub ngram_cache: crate::ngram_cache::PersistenceOptions,
    /// Capture the decode rounds' graphs after the pool exists.
    pub capture_graphs: bool,
    /// Prompt reuse's retained slots (spec flash-next/05): on the device
    /// (`--reuse-retained-device`, each one an image's worth of expert cache) and
    /// in the pool's pinned host block (`--reuse-retained-host`). Both 0 for an
    /// engine that retains nothing.
    pub retained_device_slots: u32,
    pub retained_host_slots: u32,
    /// The KV-RAM arena this load pins for its host tier
    /// (`--reuse-kv-host-pool-bytes`); 0 creates none.
    pub kv_ram_arena_bytes: u64,
    /// Speculative decoding (spec flash-next/07): the verify round at the
    /// load's windows; `None` runs one-token rounds only.
    pub speculation: Option<FlashNextSpeculation>,
    /// The text rotary table (ADR 0059): [`RopeScaling::NONE`] is the
    /// checkpoint's linear one, a YaRN factor rescales its trained 262,144
    /// positions -- on attention, the indexer and the MTP head alike. The
    /// plan and the load are handed this one value.
    pub rope_scaling: RopeScaling,
    /// Vision (GitHub #557, ADR 0087): the media embedding pool the forward
    /// reads image rows from; `None`, a text load as before.
    pub vision: Option<crate::vision::Vision>,
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

    /// What the reservation plan and the leaf's load are both asked under
    /// (ADR 0059): one value of each, read off these options.
    pub fn load_shape(&self) -> FlashNextLoadShape {
        FlashNextLoadShape {
            prefill_chunk_tokens: self.prefill_chunk_tokens,
            max_context_tokens: self.max_context_tokens,
            kv_format: self.kv_format,
            decode_lanes: self.decode_lanes,
            speculation: self.speculation,
            rope_scaling: self.rope_scaling,
            vision: self.vision,
        }
    }

    /// The rows one decode step carries at most: the lanes, or a verify
    /// round's `lanes * (k + 1)` under speculation (the leaf's
    /// `verify::decode_rows`).
    pub fn decode_rows(&self) -> u32 {
        self.speculation.map_or(self.decode_lanes, |s| s.decode_rows(self.decode_lanes))
    }

    /// The slots every K class gets first and keeps for its LRU: those of
    /// the widest decode step ([`class_minimum_slots`] of
    /// [`EngineOptions::decode_rows`]). The one minimum the cache's split,
    /// the plan's and the load's alike, and the pinned hot set keep.
    pub fn class_minimum_slots(&self) -> u32 {
        class_minimum_slots(self.decode_rows())
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

    /// The sequence pool these options build: [`EngineOptions::kv_pool_pages`]
    /// shared by the lanes, or with none every lane's whole context, plus one
    /// page per retained slot -- a checkpoint keeps the page its opener ends
    /// inside, as on the 27B (`VramRequest::retained_slots`).
    pub fn pool_budget(&self) -> SeqPoolBudget {
        let whole_contexts = self.max_context_tokens.div_ceil(64) * self.decode_lanes + self.retained_slots();
        self.pool_budget_of(self.kv_pool_pages.unwrap_or(whole_contexts))
    }

    /// [`EngineOptions::pool_budget`] at `pages` KV pages: what the VRAM plan
    /// asks the pool's arena of at any size.
    pub fn pool_budget_of(&self, pages: u32) -> SeqPoolBudget {
        SeqPoolBudget {
            kv_format: self.kv_format,
            kv_page_group_count: pages,
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
            pinned_expert_bytes: 0,
            kv_pool_pages: None,
            ngram: NgramTableOptions::default(),
            ngram_cache: crate::ngram_cache::PersistenceOptions::default(),
            capture_graphs: true,
            retained_device_slots: 0,
            retained_host_slots: 0,
            kv_ram_arena_bytes: 0,
            speculation: None,
            rope_scaling: RopeScaling::NONE,
            vision: None,
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

/// The calibration traffic the expert cache is split by: the artifact's
/// sidecar's over `catalog`, or uniform when it has none (or one for other
/// experts).
pub fn expert_traffic(path: &Path, catalog: &ExpertCatalog) -> Result<ExpertTraffic, String> {
    let entries = usize::from(catalog.layers()) * usize::from(catalog.experts());
    let counts = sidecar_traffic(path).filter(|counts| counts.len() == entries).unwrap_or_else(|| vec![1; entries]);
    ExpertTraffic::new(catalog, counts).map_err(|e| e.to_string())
}

/// The slots a K class needs for a decode step of `rows` rows: its
/// selection and lookahead, as if all of them fell in that class. A load
/// asks it of its widest step ([`EngineOptions::class_minimum_slots`]).
pub fn class_minimum_slots(rows: u32) -> u32 {
    min_slots_per_class(rows, crate::moe::TOP_K, LOOKAHEAD_WIDTH)
}

/// The expert residency of the artifact at `path` bound as `plan`, for a
/// load of `options` (spec flash-next/03): the eight K-class pools split from
/// `expert_cache_bytes` the way one LRU over the calibration traffic would
/// hold them (the sidecar's; uniform when it has none), the pinned pool
/// filled from the file, and the warm start -- the hottest projections,
/// hottest first, up to each pool -- before the first step.
///
/// A cache that holds every projection (ADR 0045's resident load, a ~96 GB
/// card) gets no pinned pool at all: every record goes from the file straight
/// to its slot, a layer at a time ([`stream_expert_layers`]), and no step
/// ever misses. The ~38 GB the pool pinned for the life of the load stays
/// free for the rest of the host plan, the n-gram table first; and with no
/// miss nothing stages, so it reserves no prefill staging ring (~1.6 GB of
/// VRAM on the trellis-a25 artifact, the KV pool's on that branch).
///
/// With a pinned hot set ([`pinned_experts`], GitHub #329 item 12) the pool
/// is laid out without the pinned records, which go from the file straight
/// to their pinned slots ([`stream_pinned_records`]) before the pool is
/// filled and the LRU warm-started around them.
pub fn build_residency(path: &Path, plan: &FlashNextPlan, options: &EngineOptions) -> Result<DeviceResidency, String> {
    let index = &plan.experts;
    let cat = catalog(index)?;
    let layout = pool_layout(index);
    let (layers, experts) = (u64::from(cat.layers()), u64::from(cat.experts()));
    let traffic = expert_traffic(path, &cat)?;
    // A device-only residency never misses, so it never stages: no ring, as
    // the VRAM plan's resident branch placed none (GitHub #329 item 2).
    let no_host_pool = device_only(options, &cat);
    let ResidencyPools { cache, pinned } = residency_pools(options, &cat, &traffic).map_err(|e| e.to_string())?;
    let (one_row, per_row) = budget_words(Some(default_prefetch_budget()));
    let desc = ResidencyDesc {
        layers: layers as u32,
        experts: experts as u32,
        capacity: cache.capacity(),
        record_bytes: std::array::from_fn(|i| cat.slot_bytes(KClass::ALL[i])),
        max_tokens: residency_max_tokens(options),
        lookahead_width: LOOKAHEAD_WIDTH,
        prefill_lookahead_width: PREFILL_LOOKAHEAD_WIDTH,
        prefetch_budget_one_row_bytes: one_row,
        prefetch_budget_per_row_bytes: per_row,
        staging_half_bytes: cache.staging_ring_bytes / 2,
        host_pool_bytes: if no_host_pool { 0 } else { layout.bytes },
        copy_blocks: 16,
        report: 0,
        pinned: [0; KClass::COUNT],
    };
    if no_host_pool {
        let mut residency = DeviceResidency::new(&desc, &layout.k2, &[])?;
        stream_expert_layers(path, index, |layer| residency.admit(&layer.keys, &layer.bytes, &layer.offsets))
            .map_err(|e| format!("admit the experts: {e}"))?;
        residency.admit_finish()?;
        return Ok(residency);
    }
    if !pinned.is_empty() {
        let by_key = pinned.by_key(&cat);
        let layout = pool_layout_without(index, &by_key);
        let desc = ResidencyDesc {
            capacity: pinned.lru_capacity(cache.capacity()),
            host_pool_bytes: layout.bytes,
            pinned: pinned.per_class,
            ..desc
        };
        let mut residency = DeviceResidency::new(&desc, &layout.k2, &layout.offsets)?;
        stream_pinned_records(path, index, &by_key, |layer| residency.admit(&layer.keys, &layer.bytes, &layer.offsets))
            .map_err(|e| format!("admit the pinned experts: {e}"))?;
        residency.admit_finish()?;
        let pool = residency.host_pool_mut().ok_or("residency: no host pool")?;
        fill_expert_pool_except(path, index, &layout, pool).map_err(|e| format!("fill the expert pool: {e}"))?;
        residency.warm_start(&warm_start_order(&traffic))?;
        return Ok(residency);
    }
    let mut residency = DeviceResidency::new(&desc, &layout.k2, &layout.offsets)?;
    let pool = residency.host_pool_mut().ok_or("residency: no host pool")?;
    fill_expert_pool(path, index, pool).map_err(|e| format!("fill the expert pool: {e}"))?;
    residency.warm_start(&warm_start_order(&traffic))?;
    Ok(residency)
}

/// Whether a load of `options` keeps every expert of `catalog` in VRAM and
/// none in RAM: its cache holds the whole catalog (the VRAM plan's resident
/// branch hands it exactly that).
pub fn device_only(options: &EngineOptions, catalog: &ExpertCatalog) -> bool {
    holds_every_expert(catalog, options.expert_cache_bytes)
}

/// The tokens one residency step classifies at most for a load of
/// `options`: a prefill chunk, or a decode round's lanes.
fn residency_max_tokens(options: &EngineOptions) -> u32 {
    options.prefill_chunk_tokens.max(options.decode_lanes)
}

/// The expert cache a load of `options` builds ([`residency_pools`]): its
/// eight K-class pools and the pinned hot set it keeps in them.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyPools {
    pub cache: ExpertCachePlan,
    pub pinned: PinnedExperts,
}

impl ResidencyPools {
    /// The LRU's slots per class: the pools less the pinned slots.
    pub fn lru_capacity(&self) -> [u32; KClass::COUNT] {
        self.pinned.lru_capacity(self.cache.capacity())
    }
}

/// Why [`residency_pools`] refused a load: the cache's split, or a pinned
/// hot set it cannot hold, which the server names by where its budget was
/// set ([`PinRefused::naming`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolsRefused {
    Split(ExpertCachePlanError),
    Pin(PinRefused),
}

impl std::fmt::Display for PoolsRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Split(refused) => refused.fmt(f),
            Self::Pin(refused) => refused.fmt(f),
        }
    }
}

/// The expert cache a load of `options` builds: the eight K-class pools
/// split from `expert_cache_bytes` the way one LRU over the calibration
/// traffic would hold them, each class given the widest decode step's slots
/// first ([`EngineOptions::class_minimum_slots`]), and the pinned hot set
/// ([`pinned_experts`]). [`build_residency`] builds them, and the server's
/// plan charges the host pool and names the pinned set from the same call
/// with the same options (GitHub #329 item 12), so the plan and the load pin
/// the same experts.
pub fn residency_pools(
    options: &EngineOptions,
    catalog: &ExpertCatalog,
    traffic: &ExpertTraffic,
) -> Result<ResidencyPools, PoolsRefused> {
    let staging_ring = staging_ring_bytes_for(catalog, options.expert_cache_bytes);
    let tables = residency_table_bytes(
        u64::from(catalog.layers()),
        u64::from(catalog.experts()),
        u64::from(residency_max_tokens(options)),
        u64::from(LOOKAHEAD_WIDTH),
    );
    let cache = plan_expert_cache(&ExpertCacheRequest {
        budget_bytes: options.expert_cache_bytes + staging_ring + tables,
        planned_bytes: 0,
        staging_ring_bytes: staging_ring,
        table_bytes: tables,
        floor_bytes: 0,
        allow_below_floor: false,
        catalog,
        traffic,
        min_slots: options.class_minimum_slots(),
    })
    .map_err(PoolsRefused::Split)?;
    let pinned = pinned_experts(options, catalog, traffic, &cache).map_err(PoolsRefused::Pin)?;
    Ok(ResidencyPools { cache, pinned })
}

/// The expert cache's split the server's plan checks at start (ADR 0045):
/// the cache the VRAM plan `vram` left beside its other lines, against the
/// floor (`--vram-allow-expert-cache-below-floor` lets it below) and the
/// class minimum of a load of `options` ([`EngineOptions::class_minimum_slots`]).
/// The same bytes and minimum as the load's split ([`residency_pools`]),
/// since the plan's residency line is the ring and `table_bytes`.
pub fn plan_cache_request<'a>(
    vram: &crate::VramPlan,
    table_bytes: u64,
    allow_below_floor: bool,
    options: &EngineOptions,
    catalog: &'a ExpertCatalog,
    traffic: &'a ExpertTraffic,
) -> ExpertCacheRequest<'a> {
    ExpertCacheRequest {
        budget_bytes: vram.total_bytes,
        planned_bytes: vram.total_bytes - vram.residency_bytes - vram.expert_cache_bytes,
        staging_ring_bytes: vram.staging_ring_bytes,
        table_bytes,
        floor_bytes: EXPERT_CACHE_FLOOR_BYTES,
        allow_below_floor,
        catalog,
        traffic,
        min_slots: options.class_minimum_slots(),
    }
}

/// The pinned hot set a load of `options` keeps beside the expert cache
/// `cache` split for it (GitHub #329 item 12): the calibration-hottest whole
/// experts until they reach `pinned_expert_bytes` ([`pin_hot_experts`]).
/// Each class keeps for its LRU the slots of the widest decode step the load
/// runs ([`EngineOptions::class_minimum_slots`]: a verify round's
/// `lanes * (k + 1)` rows under speculation, not the lanes), the minimum the
/// split gave it first, so no step finds its class without room. A budget
/// the walk cannot reach while keeping them is refused ([`PinRefused`]),
/// naming the most it can. None at 0, and none on a device-only load, which
/// keeps no expert in RAM to begin with.
pub fn pinned_experts(
    options: &EngineOptions,
    catalog: &ExpertCatalog,
    traffic: &ExpertTraffic,
    cache: &ExpertCachePlan,
) -> Result<PinnedExperts, PinRefused> {
    let asked = options.pinned_expert_bytes;
    if asked == 0 || device_only(options, catalog) {
        return Ok(PinnedExperts::default());
    }
    let keep = options.class_minimum_slots();
    let pinned = pin_hot_experts(catalog, traffic, cache.capacity(), keep, asked);
    if pinned.bytes >= asked {
        return Ok(pinned);
    }
    let most = pin_hot_experts(catalog, traffic, cache.capacity(), keep, u64::MAX).bytes;
    Err(PinRefused { asked, most, keep, rows: options.decode_rows(), lanes: options.decode_lanes })
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
    // A vision load's tower (GitHub #557): the model points into it too.
    _tower: Option<crate::flash_next_vision::Tower>,
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
        // GitHub #557: a vision load's tower, from the companion held to this
        // container before any byte of it is read.
        let tower = match options.vision {
            Some(vision) => {
                let companion = crate::flash_next_vision::Companion::open(&path, &reader, config.hidden)
                    .map_err(|e| e.to_string())?;
                Some(crate::flash_next_vision::Tower::load(&companion, vision.residency())?)
            }
            None => None,
        };
        let planned = plan_flash_next_reservations(
            &plan,
            &geometry,
            options.load_shape(),
            mtp.as_ref().map(|head| head.plan()),
            tower.as_ref().map(|tower| tower.plan()),
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
            options.load_shape(),
            &residency,
            mtp.as_ref(),
            tower.as_ref(),
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
            _tower: tower,
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

    /// The item bound this load's encode holds (GitHub #557, #558): the
    /// largest item its prefill scratch and its expert staging ring take, at
    /// most the bound it was loaded with; 0 without vision.
    pub fn vision_item_max_tokens(&self) -> u32 {
        self.model.vision_item_max_tokens()
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
                None,
                false,
                SamplingParams::greedy(),
                &[],
                Redirect::NONE,
                &rows,
                None,
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
            None,
            false,
            SamplingParams::greedy(),
            &[],
            Redirect::NONE,
            &rows,
            None,
            None,
            Some(&mut logits),
        )?;
        Ok(logits)
    }

    /// GitHub #557: `columns` (BF16 bits, `[count][hidden]`) placed in a
    /// vision load's embedding pool as an embedding
    /// ([`crate::step::MediaEmbedding::place`]): rows a test knows, for
    /// [`FlashNextEngine::span`] to put over its placeholders.
    pub fn place_media(&self, columns: &[u16]) -> Result<MediaEmbedding<'_>, String> {
        MediaEmbedding::place(&self.model, columns)
            .map_err(|(rc, message)| format!("place the media columns ({rc}): {message}"))
    }

    /// GitHub #557: `tokens` prefilled from position 0 on a fresh sequence as
    /// one span -- of a multimodal prompt with `multimodal`, a text one
    /// without -- its n-gram rows hashing `ngram_tokens` (the span's own ids
    /// when `None`: a test's reference may embed other ids than it hashes).
    /// Returns every position's BF16 logits, `[tokens][vocab]`, with the
    /// standing sequence and its context, for rounds after it.
    pub fn span(
        &self,
        tokens: &[u32],
        ngram_tokens: Option<&[u32]>,
        multimodal: Option<MultimodalPrefill<'_>>,
    ) -> Result<(crate::seq::Seq<'_>, NgramContext, Vec<u16>), String> {
        let hashed = ngram_tokens.unwrap_or(tokens);
        if hashed.len() != tokens.len() {
            return Err(format!("{} n-gram ids for {} tokens", hashed.len(), tokens.len()));
        }
        let mut seq = self.pool.alloc(self.options.max_context_tokens)?;
        let mut context = self.table.new_context();
        let mut rows = vec![0u8; tokens.len() * self.table.token_bytes()];
        self.table.stage(&mut context, hashed, &mut rows)?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut logits = vec![0u16; tokens.len() * self.vocab()];
        prefill_flash_next(
            &self.model,
            &self.pool,
            &mut seq,
            &ids,
            0,
            None,
            false,
            SamplingParams::greedy(),
            &[],
            Redirect::NONE,
            &rows,
            multimodal,
            Some(&mut logits),
            None,
        )?;
        Ok((seq, context, logits))
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
        prefill_flash_next(&self.model, &self.pool, &mut seq, &ids, 0, None, false, SamplingParams::greedy(), &[], Redirect::NONE, &rows, None, None, None)?;
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

    /// [`generate`](Self::generate), handing every round's logits to
    /// `on_round` (GitHub #329 item 5): one row a lane, as its draw read it.
    pub fn generate_with_logits(
        &mut self,
        prompts: &[Vec<u32>],
        count: usize,
        on_round: &mut dyn FnMut(RoundLogits<'_>),
    ) -> Result<Vec<Vec<u32>>, String> {
        self.generate_rounds(prompts, count, Some(on_round)).map(|(tokens, _)| tokens)
    }

    /// [`generate`](Self::generate), with each round's wall time: the host's
    /// n-gram staging of the round's tokens plus the round itself, as a
    /// serving loop pays it (spec flash-next/07 phase A's cost per row).
    pub fn generate_timed(
        &mut self,
        prompts: &[Vec<u32>],
        count: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<std::time::Duration>), String> {
        self.generate_rounds(prompts, count, None)
    }

    fn generate_rounds(
        &mut self,
        prompts: &[Vec<u32>],
        count: usize,
        on_round: Option<&mut dyn FnMut(RoundLogits<'_>)>,
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
            prefill_flash_next(&self.model, &self.pool, &mut seq, &ids, 0, None, false, SamplingParams::greedy(), &[], Redirect::NONE, &rows, None, None, None)?;
            seqs.push(seq);
            contexts.push(context);
        }
        let mut out = vec![Vec::with_capacity(count); prompts.len()];
        let mut times = Vec::with_capacity(count);
        let mut rows = vec![0u8; prompts.len() * self.table.token_bytes()];
        let all: Vec<usize> = (0..prompts.len()).collect();
        let mut logits = LogitsSink::new(on_round, self.vocab());
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
            let read = logits.round(all.len());
            let emitted = decode_flash_next(&self.model, &self.pool, &mut handles, &greedy, &rows, read)?;
            times.push(began.elapsed());
            for (lane, (token, _)) in emitted.into_iter().enumerate() {
                out[lane].push(token as u32);
            }
            logits.hand(&all, 1);
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
        drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>,
    ) -> Result<SpeculativeRun<'_>, String> {
        self.generate_speculative_with_logits(prompts, count, drafter, None)
    }

    /// [`generate_speculative`](Self::generate_speculative), handing every
    /// round's logits to `on_round` (GitHub #329 item 5): a one-token round's
    /// row a lane, a verify round's every column of every lane -- drafted or
    /// padding -- as the accept read them.
    pub fn generate_speculative_with_logits(
        &self,
        prompts: &[Vec<u32>],
        count: usize,
        mut drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>,
        on_round: Option<&mut dyn FnMut(RoundLogits<'_>)>,
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
            prefill_flash_next(&self.model, &self.pool, &mut seq, &ids, 0, None, false, SamplingParams::greedy(), &[], Redirect::NONE, &rows, None, None, None)?;
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
        let mut logits = LogitsSink::new(on_round, self.vocab());
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
                let read = logits.round(active.len());
                let emitted = decode_flash_next(&self.model, &self.pool, &mut handles, &greedy, &rows, read)?;
                logits.hand(&active, 1);
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
                        reasoning_close: -1,
                    })
                    .collect();
                let mut handles: Vec<_> =
                    seqs.iter_mut().enumerate().filter(|(i, _)| active.contains(i)).map(|(_, s)| s).collect();
                let read = logits.round(active.len() * columns);
                let runs = decode_flash_next_verify(&self.model, &self.pool, &mut handles, &lanes, window, &rows, read)?;
                logits.hand(&active, columns);
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
            None,
            false,
            SamplingParams::greedy(),
            &[],
            Redirect::NONE,
            &rows,
            None,
            None,
            Some(&mut logits),
        )?;
        Ok(logits)
    }

    /// One plain round over sequences [`generate_speculative`](Self::generate_speculative)
    /// (or [`FlashNextEngine::span`]) left standing (`contexts` theirs), each
    /// lane drawn under its own sampling and permitted set, its logits copied
    /// to `out_logits` if `Some` ([`decode_flash_next`]): a round no generate
    /// path runs, for a test of what the leaf refuses (GitHub #329 item 5) or
    /// of the round after an image (GitHub #557). Moves each sequence and its
    /// context one token on, or neither when the round is refused.
    ///
    /// Test-only: public because its callers,
    /// `a_permitted_set_refuses_the_logits_copy` in
    /// `tests/flash_next_speculative_gpu.rs` and
    /// `the_axes_and_the_rope_delta_reach_the_forward` in
    /// `tests/flash_next_vision_forward_gpu.rs`, are integration tests outside
    /// the crate, and it reads the engine's private model, pool and n-gram
    /// table. No serving path calls it.
    #[doc(hidden)]
    pub fn decode_standing(
        &self,
        sequences: &mut [crate::seq::Seq<'_>],
        contexts: &mut [NgramContext],
        lanes: &[(SamplingParams, &[i32])],
        out_logits: Option<&mut [u16]>,
    ) -> Result<Vec<(i32, f32)>, String> {
        let pending: Vec<u32> = sequences
            .iter()
            .map(|seq| seq.pending_token().map(|t| t as u32).ok_or("a lane has no pending token"))
            .collect::<Result<_, _>>()?;
        let mut rows = vec![0u8; sequences.len() * self.table.token_bytes()];
        let mut staged = contexts.to_vec();
        {
            let mut batch: Vec<(&mut NgramContext, &[u32])> =
                staged.iter_mut().zip(&pending).map(|(c, t)| (c, std::slice::from_ref(t))).collect();
            self.table.begin_batch(&mut batch)?.finish(&mut rows)?;
        }
        let mut handles: Vec<_> = sequences.iter_mut().collect();
        let drawn = decode_flash_next(&self.model, &self.pool, &mut handles, lanes, &rows, out_logits)?;
        contexts.clone_from_slice(&staged);
        Ok(drawn)
    }
}

/// A run's `on_round` callback and the buffer its rounds copy their logits
/// into: a round asks for the buffer only when someone listens.
struct LogitsSink<'r> {
    on_round: Option<&'r mut dyn FnMut(RoundLogits<'_>)>,
    vocab: usize,
    rows: Vec<u16>,
}

impl<'r> LogitsSink<'r> {
    fn new(on_round: Option<&'r mut dyn FnMut(RoundLogits<'_>)>, vocab: usize) -> Self {
        Self { on_round, vocab, rows: Vec::new() }
    }

    /// The buffer a round of `rows` rows copies its logits into, or None.
    fn round(&mut self, rows: usize) -> Option<&mut [u16]> {
        self.on_round.as_ref()?;
        self.rows.resize(rows * self.vocab, 0);
        Some(&mut self.rows[..])
    }

    /// Hands the round just copied -- `lanes` (prompt indices), `columns` a
    /// lane -- to the callback.
    fn hand(&mut self, lanes: &[usize], columns: usize) {
        if let Some(on_round) = self.on_round.as_mut() {
            on_round(RoundLogits { lanes, columns, rows: &self.rows });
        }
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

/// One round's logits, as a run hands them to its `on_round`
/// ([`FlashNextEngine::generate_with_logits`],
/// [`FlashNextEngine::generate_speculative_with_logits`]): the prompt indices
/// of the lanes that ran it, in order, its columns a lane (1 in a one-token
/// round, the window + 1 in a verify round) and its BF16 rows,
/// `[lanes][columns][vocab]`. Every row the round's head produced, copied
/// while the round still samples -- not a **readout**, which reads named
/// tokens and draws nothing (GLOSSARY.md).
pub struct RoundLogits<'a> {
    pub lanes: &'a [usize],
    pub columns: usize,
    pub rows: &'a [u16],
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

    /// ADR 0045: the pool is the plan's pages, shared by the lanes, whenever
    /// the load planned them; every lane's whole context and a page per
    /// retained slot only for an engine that never plans.
    #[test]
    fn the_pool_takes_the_plan_s_pages_when_the_load_planned_them() {
        let options = EngineOptions {
            max_context_tokens: 262_144,
            decode_lanes: 3,
            retained_host_slots: 8,
            ..EngineOptions::default()
        };
        assert_eq!(options.pool_budget().kv_page_group_count, 3 * 4_096 + 8);
        let planned = EngineOptions { kv_pool_pages: Some(8_192), ..options.clone() };
        let budget = planned.pool_budget();
        assert_eq!(budget.kv_page_group_count, 8_192);
        assert_eq!((budget.max_context_tokens, budget.slot_count, budget.retained_host_slot_count), (262_144, 3, 8));
        assert_eq!(planned.pool_budget_of(100).kv_page_group_count, 100);
    }

    /// 128 experts over `layers` layers, every projection K = 2 (gate/up 200
    /// bytes, down 100), every expert selected; and a split of `slots` slots
    /// in each of their two classes.
    fn two_class_catalog(layers: u16, slots: u32) -> (ExpertCatalog, ExpertTraffic, ExpertCachePlan) {
        use crate::residency::{ClassPool, KBits};
        let slot_bytes = [200, 250, 300, 400, 100, 125, 150, 200];
        let catalog = ExpertCatalog::new(layers, 128 / layers, vec![(KBits::K2, KBits::K2); 128], slot_bytes).expect("catalog");
        let traffic = ExpertTraffic::new(&catalog, (0..128).map(|i| 1 + i % 7).collect()).expect("traffic");
        let mut pools = [ClassPool::default(); KClass::COUNT];
        let n = u64::from(slots);
        (pools[0], pools[4]) = (ClassPool { slots, bytes: n * 200 }, ClassPool { slots, bytes: n * 100 });
        let cache = ExpertCachePlan {
            cache_bytes: n * 300,
            pools,
            staging_ring_bytes: 0,
            table_bytes: 0,
            expected_hit_rate: 0.0,
            below_floor: false,
        };
        (catalog, traffic, cache)
    }

    /// GitHub #329 item 12: a load pins the plan's hottest experts at its
    /// budget, each class keeping its decode minimum for the LRU, and none at
    /// 0 or on a device-only load (a cache that holds every expert).
    #[test]
    fn a_load_pins_its_hottest_experts_only_when_asked_and_only_beside_a_host_pool() {
        let (catalog, traffic, cache) = two_class_catalog(2, 40);
        let options = EngineOptions { decode_lanes: 1, expert_cache_bytes: cache.cache_bytes, ..EngineOptions::default() };
        let pin = |options: &EngineOptions| pinned_experts(options, &catalog, &traffic, &cache);
        assert!(pin(&options).expect("0").is_empty(), "0 pins none");
        let asked = EngineOptions { pinned_expert_bytes: 1_000, ..options.clone() };
        let pinned = pin(&asked).expect("1,000 bytes fit");
        assert_eq!(pinned, pin_hot_experts(&catalog, &traffic, cache.capacity(), class_minimum_slots(1), 1_000));
        assert_eq!((pinned.experts(), pinned.bytes), (4, 1_200), "whole experts until the budget is reached");
        // Each class keeps 26 LRU slots of 40: at most 14 experts pinned, 4,200 bytes.
        let most = EngineOptions { pinned_expert_bytes: 4_200, ..options.clone() };
        assert_eq!(pin(&most).expect("the most it can pin").per_class, [14, 0, 0, 0, 14, 0, 0, 0]);
        let device_only = EngineOptions { expert_cache_bytes: catalog.total_bytes(), ..asked };
        assert!(pin(&device_only).expect("ignored").is_empty(), "nothing in RAM to save");
    }

    /// GitHub #329 item 12, review 2026-10-18: the pin leaves each class the
    /// LRU slots of the widest decode step the load can run -- under MTP a
    /// verify round's `lanes * (k + 1)` rows, more than the lanes -- and a
    /// pin that cannot leave them is refused at plan time, naming what can.
    #[test]
    fn a_pin_leaves_every_class_the_widest_decode_step_or_is_refused() {
        let (catalog, traffic, cache) = two_class_catalog(2, 100);
        let mtp = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 2, 0).expect("mtp");
        let plain = EngineOptions { decode_lanes: 1, expert_cache_bytes: cache.cache_bytes, ..EngineOptions::default() };
        let speculative = EngineOptions { speculation: Some(mtp), ..plain.clone() };
        assert_eq!((plain.decode_rows(), speculative.decode_rows()), (1, 3), "one lane verifies 2 drafts: 3 rows");
        let pin = |options: &EngineOptions, bytes| {
            pinned_experts(&EngineOptions { pinned_expert_bytes: bytes, ..options.clone() }, &catalog, &traffic, &cache)
        };
        // 24 experts (7,200 bytes) leave each class 76 LRU slots: one lane's
        // step needs 26, a 3-row verify round 78.
        let starving = pin(&plain, 7_000).expect("one lane keeps 26");
        assert_eq!(starving.lru_capacity(cache.capacity())[0], 76);
        let err = pin(&speculative, 7_000).expect_err("a verify round would find too few slots");
        assert_eq!(err, PinRefused { asked: 7_000, most: 6_600, keep: 78, rows: 3, lanes: 1 });
        let named = err.naming("vram.pinned_expert_bytes (profile p in c.yaml)");
        for part in ["`vram.pinned_expert_bytes (profile p in c.yaml)` asks for 7000", "at most 6600", "78 LRU slots", "3 rows: a verify round"] {
            assert!(named.contains(part), "{part}: {named}");
        }
        let kept = pin(&speculative, 6_600).expect("22 experts leave 78");
        assert!(kept.lru_capacity(cache.capacity()).iter().zip(cache.capacity()).all(|(&lru, pool)| pool == 0 || lru >= 78));
        // A class too small for one verify round's step pins nothing.
        let (catalog, traffic, cache) = two_class_catalog(2, 40);
        let options = EngineOptions { pinned_expert_bytes: 1, ..speculative };
        let err = pinned_experts(&options, &catalog, &traffic, &cache).expect_err("40 slots, 78 kept");
        assert_eq!(err.most, 0, "{err}");
    }

    /// 256 experts over eight layers (so that the plan's staging ring, two
    /// of them, offloads): the first four layers' hot (100 selections each)
    /// at K = 2, 300 bytes an expert, the last four layers' cold (1 each) at
    /// K = 3, 450 bytes -- four classes of 128 projections, the cold ones
    /// (gate/up and down K3) those a split by traffic gives least.
    fn hot_and_cold_catalog() -> (ExpertCatalog, ExpertTraffic) {
        use crate::residency::KBits;
        let ks = (0..256).map(|i| if i < 128 { (KBits::K2, KBits::K2) } else { (KBits::K3, KBits::K3) }).collect();
        let catalog = ExpertCatalog::new(8, 32, ks, [200, 250, 300, 400, 100, 125, 150, 200]).expect("catalog");
        let traffic = ExpertTraffic::new(&catalog, (0..256).map(|i| if i < 128 { 100 } else { 1 }).collect()).expect("traffic");
        (catalog, traffic)
    }

    /// GitHub #329 item 12, second review 2026-10-18: the split gives every
    /// class the slots of the widest decode step first -- under MTP a verify
    /// round's `lanes * (k + 1)` rows, not the lanes -- the minimum the pin
    /// keeps too, so no class is split below what one round selects and
    /// looks ahead, pinned or not.
    /// Without speculation the widest step is the lanes', so a load with MTP
    /// off (the rtx5090 profile's) splits as before.
    #[test]
    fn the_split_gives_every_class_the_widest_decode_step() {
        let (catalog, traffic) = hot_and_cold_catalog();
        let mtp = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 2, 0).expect("mtp");
        let plain = EngineOptions { decode_lanes: 1, prefill_chunk_tokens: 64, expert_cache_bytes: 60_000, ..EngineOptions::default() };
        let speculative = EngineOptions { speculation: Some(mtp), ..plain.clone() };
        assert_eq!((plain.class_minimum_slots(), speculative.class_minimum_slots()), (26, 78));
        assert_eq!(EngineOptions::default().class_minimum_slots(), class_minimum_slots(DEFAULT_DECODE_LANES), "MTP off: the lanes");
        // GitHub #329 item 5: a row budget past 8 widens the step -- 8 lanes
        // x 4 columns at 32 rows -- and the minimum with it.
        let wide = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 3, 32).expect("a 32-row budget");
        let wide = EngineOptions { decode_lanes: 8, speculation: Some(wide), ..plain.clone() };
        assert_eq!((wide.decode_rows(), wide.class_minimum_slots()), (32, class_minimum_slots(32)));
        let split = |options: &EngineOptions| residency_pools(options, &catalog, &traffic).expect("split").cache.capacity();
        // One lane's step: 26 slots a class first, and the cold classes get
        // little more.
        let lanes = split(&plain);
        assert!(lanes[2] < 78 && lanes[6] < 78, "{lanes:?}");
        // A 3-row verify round's: 78 a class first.
        let rows = split(&speculative);
        assert!(rows.iter().all(|&slots| slots == 0 || slots >= 78), "{rows:?}");
        // The plan's split takes the same minimum from the same options.
        let tables = residency_table_bytes(8, 32, 64, u64::from(LOOKAHEAD_WIDTH));
        let vram = offloaded_vram(&catalog, tables, 60_000);
        for (options, split) in [(&plain, lanes), (&speculative, rows)] {
            let request = plan_cache_request(&vram, tables, true, options, &catalog, &traffic);
            assert_eq!(plan_expert_cache(&request).expect("plan's split").capacity(), split);
        }
    }

    /// An offloaded VRAM plan at one lane of `catalog`'s experts: the pool's
    /// one page of one byte (a lane's whole 64-token context), the staging
    /// ring, `tables`, and `cache_bytes` left to the cache.
    fn offloaded_vram(catalog: &ExpertCatalog, tables: u64, cache_bytes: u64) -> crate::VramPlan {
        use crate::vram::{ExpertResidency, Residency, VramLines, VramMode, VramRequest};
        let ring = crate::residency::prefill_staging_ring_bytes(catalog);
        let budget = 1 + ring + tables + cache_bytes;
        let vram = crate::plan_vram(&VramRequest {
            mode: VramMode::Explicit { budget_bytes: budget, allow_oversubscription: false },
            free_at_start_bytes: budget,
            lines: VramLines::default(),
            kv_page_bytes: 1,
            max_context_tokens: 64,
            retained_slots: 0,
            kv_pool: None,
            residency: Residency::Experts(ExpertResidency {
                expert_pool_bytes: catalog.total_bytes(),
                staging_ring_bytes: ring,
                table_bytes: tables,
                decode_lanes: 1,
            }),
            embedding_pool_named: false,
            kv_arena_bytes: &|pages: u32| u64::from(pages),
            can_page: true,
        })
        .expect("plan");
        assert_eq!(vram.expert_cache_bytes, cache_bytes, "offloaded");
        vram
    }

    /// GitHub #329 item 12, review 2026-10-18: the server's plan and the load
    /// split one cache alike. The plan checks the floor on what the VRAM
    /// plan left the cache ([`plan_cache_request`]); the load splits the
    /// options' `expert_cache_bytes`, and the plan charges the host pool and
    /// names the pinned set from the same [`residency_pools`] the load
    /// builds from -- so the pinned set the plan logs is the one the load
    /// pins.
    #[test]
    fn the_plan_and_the_load_split_the_cache_alike_and_pin_the_same_experts() {
        // Four layers, so that the staging ring (two of them) is less than
        // every expert and the plan offloads.
        let (catalog, traffic, _) = two_class_catalog(4, 0);
        let options = EngineOptions { decode_lanes: 1, prefill_chunk_tokens: 64, ..EngineOptions::default() };
        let tables = residency_table_bytes(4, 32, 64, u64::from(LOOKAHEAD_WIDTH));
        for cache_bytes in [9_000u64, 12_000, 15_000] {
            let vram = offloaded_vram(&catalog, tables, cache_bytes);
            let planned =
                plan_expert_cache(&plan_cache_request(&vram, tables, true, &options, &catalog, &traffic)).expect("plan's split");
            let options = EngineOptions { expert_cache_bytes: vram.expert_cache_bytes, pinned_expert_bytes: 600, ..options.clone() };
            let pools = residency_pools(&options, &catalog, &traffic).expect("load's split");
            assert_eq!(pools.cache.capacity(), planned.capacity(), "{cache_bytes}: one split");
            assert_eq!(pools.pinned, pinned_experts(&options, &catalog, &traffic, &planned).expect("pin"), "{cache_bytes}");
            assert_eq!(pools.lru_capacity(), pools.pinned.lru_capacity(planned.capacity()));
            assert!(!pools.pinned.is_empty(), "{cache_bytes}: something pinned to compare");
        }
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
