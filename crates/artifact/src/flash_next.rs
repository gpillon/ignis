//! Qwen3.8-Flash-Next's artifact family (spec flash-next/01, ADRs 0002,
//! 0043, 0044): its object inventory, the per-layer expert index and the
//! binder plan that consumes every object.
//!
//! The container is an ordinary `.ninfer` v2 file; what makes it Flash-Next
//! is its identity (`model_id` [`MODEL_ID`], never a 27B one) and its
//! inventory, which `docs/specs/flash-next/layout.md` fixes:
//! - every non-expert tensor under its checkpoint name (without the
//!   `model.language_model.` prefix): linears FP8 with a per-row scale, the
//!   router, norms and small vectors BF16;
//! - one tensor per **expert projection**
//!   (`layers.{L}.mlp.experts.{E}.gate_up_proj` / `.down_proj`), trellis-coded
//!   at its own K (the K is the format code), 4096-aligned, its size a pure
//!   function of its shape and K: eight K classes;
//! - the n-gram table as one INT4 tensor of fixed-stride rows, its hash
//!   buffers (I64) and its hot-row list (I32);
//! - the frontend resources (`frontend/<file>`).
//!
//! [`bind`] gives each object its role: non-experts go to the device arena,
//! expert projections to the **expert pool** in pinned host RAM, which
//! residency moves them from (spec 03), the table is **host-streamed** (its
//! file range is handed over, never read at load: spec 04's n-gram reader
//! owns it), the small n-gram
//! tensors and the frontend stay in host memory. An object the inventory
//! does not name fails the bind (ADR 0002).

use std::collections::BTreeMap;

use crate::binder::{Binder, MaterializationPlan, ObjectHandle};
use crate::{fail, NumericFormat, Object, Reader, ResourceEncoding, Result, StorageLayout};

pub mod fixture;

/// The container identity's `model_id` for every Flash-Next artifact.
pub const MODEL_ID: &str = "qwen3.8-flash-next";

/// The MTP companion container's `model_id` (layout.md §13.2): never the
/// main container's, so neither binds as the other.
pub const MTP_MODEL_ID: &str = "qwen3.8-flash-next-mtp";

/// The MTP head's names keep the checkpoint's own prefix (layout.md §13.3).
pub const MTP_PREFIX: &str = "mtp.";

/// The frontend resources the converter copies from the checkpoint
/// (layout.md §8): the 27B's six plus the model config.
pub const FRONTEND_FILES: [&str; 7] = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
];

/// A routed expert's two projections: the unit residency moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Projection {
    /// The fused gate/up plane (gate rows first).
    GateUp,
    Down,
}

impl Projection {
    pub const ALL: [Projection; 2] = [Projection::GateUp, Projection::Down];

    /// The code `experts.idx` stores (0 = gate/up, 1 = down).
    pub fn code(self) -> u8 {
        match self {
            Projection::GateUp => 0,
            Projection::Down => 1,
        }
    }

    /// The object-name suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            Projection::GateUp => "gate_up_proj",
            Projection::Down => "down_proj",
        }
    }
}

/// An expert projection's trellis bit width, K ∈ {2, 2.5, 3, 4} (ADR 0044).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TrellisK {
    K2,
    K2p5,
    K3,
    K4,
}

impl TrellisK {
    pub const ALL: [TrellisK; 4] = [TrellisK::K2, TrellisK::K2p5, TrellisK::K3, TrellisK::K4];

    /// `2·K`, the integer every file stores (4, 5, 6, 8).
    pub fn k2(self) -> u8 {
        match self {
            TrellisK::K2 => 4,
            TrellisK::K2p5 => 5,
            TrellisK::K3 => 6,
            TrellisK::K4 => 8,
        }
    }

    pub fn from_k2(k2: u8) -> Result<Self> {
        TrellisK::ALL
            .into_iter()
            .find(|k| k.k2() == k2)
            .ok_or_else(|| fail(format!("k2 {k2} is not a trellis bit width (4, 5, 6 or 8)")))
    }

    /// The container format code carrying this K (the inverse of
    /// [`NumericFormat::trellis_k2`]).
    pub fn format(self) -> NumericFormat {
        match self {
            TrellisK::K2 => NumericFormat::TrellisMul1K2,
            TrellisK::K2p5 => NumericFormat::TrellisMul1K2p5,
            TrellisK::K3 => NumericFormat::TrellisMul1K3,
            TrellisK::K4 => NumericFormat::TrellisMul1K4,
        }
    }

    pub fn from_format(format: NumericFormat) -> Option<Self> {
        let k2 = format.trellis_k2()?;
        TrellisK::from_k2(u8::try_from(k2).ok()?).ok()
    }
}

/// Every count and width the inventory is generated from: the checkpoint's
/// ([`FlashNextGeometry::qwen38_flash_next`]) or a reduced fixture's
/// ([`FlashNextGeometry::fixture`]).
///
/// The checkpoint's numbers are the ones `ignis_core::compute::ModelConfig::
/// qwen38_flash_next` holds for the forward; this crate sits below core and
/// cannot read them from there, so the two are kept equal by hand until one
/// is derived from the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashNextGeometry {
    pub layers: usize,
    /// Layer `l` is an attention layer iff `(l + 1) % interval == 0`.
    pub full_attention_interval: usize,
    pub hidden: u64,
    pub vocab: u64,
    pub hc_streams: u64,
    pub hc_rank: u64,
    pub gdn_key_heads: u64,
    pub gdn_value_heads: u64,
    pub gdn_head_dim: u64,
    pub conv_kernel: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub indexer_heads: u64,
    pub indexer_kv_heads: u64,
    pub indexer_head_dim: u64,
    pub experts: u64,
    pub expert_intermediate: u64,
    pub shared_intermediate: u64,
    /// The layer holding the n-gram (PLE) block.
    pub ple_layer: usize,
    pub ple_conv_kernel: u64,
    pub ngram_size: u64,
    pub heads_per_ngram: u64,
    /// One n-gram head's width: the table's row width.
    pub ngram_head_dim: u64,
    pub ngram_rows: u64,
}

impl FlashNextGeometry {
    /// `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d`, from its text
    /// config and its tensor headers.
    pub fn qwen38_flash_next() -> Self {
        Self {
            layers: 48,
            full_attention_interval: 4,
            hidden: 2560,
            vocab: 248_320,
            hc_streams: 4,
            hc_rank: 320,
            gdn_key_heads: 16,
            gdn_value_heads: 48,
            gdn_head_dim: 128,
            conv_kernel: 4,
            attention_heads: 24,
            kv_heads: 2,
            head_dim: 256,
            indexer_heads: 4,
            indexer_kv_heads: 1,
            indexer_head_dim: 128,
            experts: 512,
            expert_intermediate: 640,
            shared_intermediate: 640,
            ple_layer: 1,
            ple_conv_kernel: 4,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_head_dim: 160,
            // 128 checkpoint shards of 2,500,012 rows (layout.md §7.1).
            ngram_rows: 320_001_536,
        }
    }

    /// A reduced Flash-Next for CPU tests: two layers (a GDN layer, then an
    /// attention layer carrying the n-gram block), eight experts, a
    /// 1,000-row table. Every projection dimension is a multiple of 128, as
    /// the trellis's Hadamard rotation requires.
    pub fn fixture() -> Self {
        Self {
            layers: 2,
            full_attention_interval: 2,
            hidden: 256,
            vocab: 512,
            hc_streams: 4,
            hc_rank: 32,
            gdn_key_heads: 2,
            gdn_value_heads: 4,
            gdn_head_dim: 32,
            conv_kernel: 4,
            attention_heads: 2,
            kv_heads: 1,
            head_dim: 64,
            indexer_heads: 2,
            indexer_kv_heads: 1,
            indexer_head_dim: 32,
            experts: 8,
            expert_intermediate: 128,
            shared_intermediate: 128,
            ple_layer: 1,
            ple_conv_kernel: 4,
            ngram_size: 3,
            heads_per_ngram: 1,
            ngram_head_dim: 160,
            ngram_rows: 1_000,
        }
    }

    pub fn is_attention_layer(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attention_interval == 0
    }

    /// An expert projection's stored shape, `[out, in]`.
    pub fn projection_shape(&self, projection: Projection) -> [u64; 2] {
        match projection {
            Projection::GateUp => [2 * self.expert_intermediate, self.hidden],
            Projection::Down => [self.hidden, self.expert_intermediate],
        }
    }

    /// The n-gram heads (orders 2..=ngram_size, `heads_per_ngram` each).
    pub fn ngram_heads(&self) -> u64 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The concatenated n-gram embedding width (`ple_embed_dim`).
    pub fn ple_embed_dim(&self) -> u64 {
        self.ngram_heads() * self.ngram_head_dim
    }

    /// The hyper-connection width: every stream side by side.
    pub fn hc_width(&self) -> u64 {
        self.hc_streams * self.hidden
    }
}

/// An expert projection's object name.
pub fn expert_name(layer: usize, expert: u64, projection: Projection) -> String {
    format!("layers.{layer}.mlp.experts.{expert}.{}", projection.suffix())
}

/// An MTP expert projection's object name (layout.md §13.3).
pub fn mtp_expert_name(expert: u64, projection: Projection) -> String {
    format!("{MTP_PREFIX}layers.0.mlp.experts.{expert}.{}", projection.suffix())
}

/// The bytes one expert projection of `projection`'s shape stores at `k`:
/// its trellis, its channel scales, the padding to 4096.
pub fn record_bytes(geometry: &FlashNextGeometry, projection: Projection, k: TrellisK) -> u64 {
    crate::tensor_encoded_size(
        StorageLayout::TrellisTile16V1,
        k.format(),
        &geometry.projection_shape(projection),
    )
    .expect("a Flash-Next projection shape is a valid trellis shape")
}

/// The eight K classes (two projection shapes × four K) and their bytes.
pub fn k_classes(geometry: &FlashNextGeometry) -> Vec<(Projection, TrellisK, u64)> {
    Projection::ALL
        .into_iter()
        .flat_map(|p| TrellisK::ALL.into_iter().map(move |k| (p, k, record_bytes(geometry, p, k))))
        .collect()
}

// ---------------------------------------------------------------------------
// The inventory
// ---------------------------------------------------------------------------

/// Where a bound object goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The device arena (the materializer uploads it).
    Device,
    /// Host memory (the n-gram hash buffers and hot rows).
    Host,
    /// Read from the file on demand; only its range is handed over.
    HostStreamed,
}

/// The shape a tensor entry requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeRule {
    Exact(Vec<u64>),
    /// Rank 1, any positive length (the hot-row list: its length is data).
    AnyVector,
}

/// One non-expert tensor of the inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub format: NumericFormat,
    pub layout: StorageLayout,
    pub shape: ShapeRule,
    pub role: Role,
}

/// Linears narrower than this stay BF16 (layout.md 6.2, the converter's
/// `FP8_MIN_ROWS`).
const FP8_MIN_DIM: u64 = 16;

/// A linear weight `[rows, columns]`: FP8 with a per-row scale when both
/// dimensions reach [`FP8_MIN_DIM`], BF16 otherwise (layout.md 6.2: the
/// converter's own rule, so a reduced geometry picks the same formats).
fn linear(name: String, rows: u64, columns: u64) -> Entry {
    if rows.min(columns) < FP8_MIN_DIM {
        return bf16(name, &[rows, columns]);
    }
    Entry {
        name,
        format: NumericFormat::Fp8E4M3FnRowBf16S,
        layout: StorageLayout::RowScaleV1,
        shape: ShapeRule::Exact(vec![rows, columns]),
        role: Role::Device,
    }
}

fn bf16(name: String, shape: &[u64]) -> Entry {
    Entry {
        name,
        format: NumericFormat::Bf16,
        layout: StorageLayout::ContiguousLeV1,
        shape: ShapeRule::Exact(shape.to_vec()),
        role: Role::Device,
    }
}

fn host_vector(name: String, format: NumericFormat, shape: ShapeRule) -> Entry {
    Entry {
        name,
        format,
        layout: StorageLayout::ContiguousLeV1,
        shape,
        role: Role::Host,
    }
}

/// The table's object name.
pub fn ngram_table_name(geometry: &FlashNextGeometry) -> String {
    format!("layers.{}.ple.ple_embedding.ngram_embedding.weight", geometry.ple_layer)
}

/// The global (non-layer) tensors, in container order.
pub fn global_entries(g: &FlashNextGeometry) -> Vec<Entry> {
    let hc = g.hc_width();
    vec![
        linear("embed_tokens.weight".into(), g.vocab, g.hidden),
        bf16("hyper_connection_mixer.hc_norm.weight".into(), &[hc]),
        linear("hyper_connection_mixer.input_mix_weight_down.weight".into(), g.hc_rank, hc),
        linear("hyper_connection_mixer.input_mix_weight_up.weight".into(), hc, g.hc_rank),
        linear("lm_head.weight".into(), g.vocab, g.hidden),
    ]
}

/// The n-gram unit's objects: the table and its small tensors.
pub fn ngram_entries(g: &FlashNextGeometry) -> Vec<Entry> {
    let prefix = format!("layers.{}.ple.ple_embedding", g.ple_layer);
    vec![
        Entry {
            name: ngram_table_name(g),
            format: NumericFormat::Q4G32F16S,
            layout: StorageLayout::RowInterleavedV1,
            shape: ShapeRule::Exact(vec![g.ngram_rows, g.ngram_head_dim]),
            role: Role::HostStreamed,
        },
        host_vector(format!("{prefix}.layer_multipliers"), NumericFormat::I64, ShapeRule::Exact(vec![g.ngram_size])),
        host_vector(format!("{prefix}.ngram_heads_vocab_sizes"), NumericFormat::I64, ShapeRule::Exact(vec![g.ngram_heads()])),
        host_vector(format!("{prefix}.ngram_heads_offsets"), NumericFormat::I64, ShapeRule::Exact(vec![g.ngram_heads()])),
        host_vector(format!("{prefix}.ngram_embedding.hot_rows"), NumericFormat::I32, ShapeRule::AnyVector),
    ]
}

/// The MTP head's non-expert tensors (layout.md §13.3): the combine, the
/// head's own mixer, and its one layer under `mtp.layers.0.`, which is a
/// trunk attention layer without the n-gram block. Each is the trunk's
/// entry for the same tensor, so the formats are the trunk's rule.
pub fn mtp_entries(g: &FlashNextGeometry) -> Vec<Entry> {
    let mut entries = vec![
        linear(format!("{MTP_PREFIX}fc_embedding.weight"), g.hidden, g.hidden),
        linear(format!("{MTP_PREFIX}fc_hidden.weight"), g.hidden, g.hidden),
        bf16(format!("{MTP_PREFIX}pre_fc_norm_embedding.weight"), &[g.hidden]),
        bf16(format!("{MTP_PREFIX}pre_fc_norm_hidden.weight"), &[g.hc_width()]),
    ];
    let renamed = |mut entry: Entry, from: &str, to: &str| {
        entry.name = format!("{to}{}", &entry.name[from.len()..]);
        entry
    };
    entries.extend(
        global_entries(g)
            .into_iter()
            .filter(|e| e.name.starts_with("hyper_connection_mixer."))
            .map(|e| renamed(e, "", MTP_PREFIX)),
    );
    let attention_layer = g.full_attention_interval - 1;
    let without_ngram = FlashNextGeometry {
        ple_layer: usize::MAX,
        ..g.clone()
    };
    let from = format!("layers.{attention_layer}.");
    let to = format!("{MTP_PREFIX}layers.0.");
    entries.extend(
        layer_entries(&without_ngram, attention_layer)
            .into_iter()
            .map(|e| renamed(e, &from, &to)),
    );
    entries
}

/// A layer's non-expert tensors, in the order the inventory lists them
/// (the container's own order is the converter's; binding is by name).
pub fn layer_entries(g: &FlashNextGeometry, layer: usize) -> Vec<Entry> {
    let n = |suffix: &str| format!("layers.{layer}.{suffix}");
    let h = g.hidden;
    let hc = g.hc_width();
    let mut entries = Vec::new();
    for block in ["attn_hyper_connection", "mlp_hyper_connection"] {
        entries.push(linear(n(&format!("{block}.block_inject_weight.weight")), g.hc_streams, hc));
        entries.push(bf16(n(&format!("{block}.hc_norm.weight")), &[hc]));
        entries.push(linear(n(&format!("{block}.input_mix_weight_down.weight")), g.hc_rank, hc));
        entries.push(linear(n(&format!("{block}.input_mix_weight_up.weight")), hc, g.hc_rank));
    }
    if g.is_attention_layer(layer) {
        let q_rows = 2 * g.attention_heads * g.head_dim;
        let kv_rows = g.kv_heads * g.head_dim;
        entries.push(linear(n("self_attn.q_proj.weight"), q_rows, h));
        entries.push(linear(n("self_attn.k_proj.weight"), kv_rows, h));
        entries.push(linear(n("self_attn.v_proj.weight"), kv_rows, h));
        entries.push(linear(n("self_attn.o_proj.weight"), h, g.attention_heads * g.head_dim));
        entries.push(bf16(n("self_attn.q_norm.weight"), &[g.head_dim]));
        entries.push(bf16(n("self_attn.k_norm.weight"), &[g.head_dim]));
        let index_rows = (g.indexer_heads + g.indexer_kv_heads) * g.indexer_head_dim;
        entries.push(linear(n("self_attn.indexer.index_qk_proj.weight"), index_rows, h));
        entries.push(bf16(n("self_attn.indexer.q_layernorm.weight"), &[g.indexer_head_dim]));
        entries.push(bf16(n("self_attn.indexer.k_layernorm.weight"), &[g.indexer_head_dim]));
    } else {
        let key_dim = g.gdn_key_heads * g.gdn_head_dim;
        let value_dim = g.gdn_value_heads * g.gdn_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        entries.push(linear(n("linear_attn.in_proj_qkv.weight"), conv_dim, h));
        entries.push(linear(n("linear_attn.in_proj_z.weight"), value_dim, h));
        entries.push(linear(n("linear_attn.in_proj_a.weight"), g.gdn_value_heads, h));
        entries.push(linear(n("linear_attn.in_proj_b.weight"), g.gdn_value_heads, h));
        entries.push(linear(n("linear_attn.out_proj.weight"), h, value_dim));
        entries.push(bf16(n("linear_attn.conv1d.weight"), &[conv_dim, 1, g.conv_kernel]));
        entries.push(bf16(n("linear_attn.A_log"), &[g.gdn_value_heads]));
        entries.push(bf16(n("linear_attn.dt_bias"), &[g.gdn_value_heads]));
        entries.push(bf16(n("linear_attn.norm.weight"), &[g.gdn_head_dim]));
    }
    entries.push(bf16(n("mlp.gate.weight"), &[g.experts, h]));
    entries.push(linear(n("mlp.shared_expert.gate_proj.weight"), g.shared_intermediate, h));
    entries.push(linear(n("mlp.shared_expert.up_proj.weight"), g.shared_intermediate, h));
    entries.push(linear(n("mlp.shared_expert.down_proj.weight"), h, g.shared_intermediate));
    entries.push(linear(n("mlp.shared_expert_gate.weight"), 1, h));
    if layer == g.ple_layer {
        let e = g.ple_embed_dim();
        entries.push(bf16(n("ple.conv1d.weight"), &[hc, 1, g.ple_conv_kernel]));
        entries.push(linear(n("ple.key_proj.weight"), hc, e));
        entries.push(bf16(n("ple.norm_conv.weight"), &[hc]));
        entries.push(bf16(n("ple.norm_key.weight"), &[hc]));
        entries.push(bf16(n("ple.norm_query.weight"), &[hc]));
        entries.push(linear(n("ple.value_proj.weight"), h, e));
    }
    entries
}

// ---------------------------------------------------------------------------
// The expert index
// ---------------------------------------------------------------------------

/// Where one expert projection lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRecord {
    pub handle: ObjectHandle,
    /// Absolute file offset (4096-aligned: one direct read moves it).
    pub file_offset: u64,
    /// Stored bytes: its K class's size.
    pub bytes: u64,
    pub k: TrellisK,
    /// Offset in the expert pool (pinned host RAM) the plan lays out.
    pub pool_offset: u64,
}

/// (layer, expert id, projection) → record, for every expert projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertIndex {
    experts_per_layer: usize,
    records: Vec<ExpertRecord>,
}

impl ExpertIndex {
    pub fn layers(&self) -> usize {
        self.records.len() / (2 * self.experts_per_layer)
    }

    pub fn experts_per_layer(&self) -> usize {
        self.experts_per_layer
    }

    pub fn get(&self, layer: usize, expert: usize, projection: Projection) -> Option<&ExpertRecord> {
        if expert >= self.experts_per_layer {
            return None;
        }
        let at = (layer * self.experts_per_layer + expert) * 2 + projection.code() as usize;
        self.records.get(at)
    }

    /// A layer's expert projections as one file range (offset, bytes): the
    /// records sit back to back in index order, exactly the converter's
    /// `experts.bin`. `None` for a layer out of range or records that are
    /// not contiguous.
    pub fn layer_range(&self, layer: usize) -> Option<(u64, u64)> {
        let per_layer = 2 * self.experts_per_layer;
        let records = self.records.get(layer * per_layer..(layer + 1) * per_layer)?;
        let start = records.first()?.file_offset;
        let mut end = start;
        for record in records {
            if record.file_offset != end {
                return None;
            }
            end += record.bytes;
        }
        Some((start, end - start))
    }

    /// How many projections each K class holds.
    pub fn class_counts(&self) -> BTreeMap<(Projection, TrellisK), usize> {
        let mut counts = BTreeMap::new();
        for (at, record) in self.records.iter().enumerate() {
            let projection = Projection::ALL[at % 2];
            *counts.entry((projection, record.k)).or_insert(0) += 1;
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// The binder plan
// ---------------------------------------------------------------------------

/// A Flash-Next artifact bound: the plan (every object placed) plus the
/// handles the forward and residency look objects up by.
#[derive(Debug, Clone)]
pub struct FlashNextPlan {
    pub plan: MaterializationPlan,
    pub experts: ExpertIndex,
    /// Every non-expert tensor and frontend resource, by name.
    pub handles: BTreeMap<String, ObjectHandle>,
}

/// Bind a Flash-Next artifact: every object of the inventory is required
/// with its exact format, layout and shape, placed by its role, and any
/// object the inventory does not name fails the bind (ADR 0002).
pub fn bind(reader: &Reader, geometry: &FlashNextGeometry) -> Result<FlashNextPlan> {
    if reader.identity().model_id != MODEL_ID {
        return Err(fail(format!(
            "artifact model_id {} is not a Flash-Next artifact ({MODEL_ID})",
            reader.identity().model_id
        )));
    }
    let mut binder = Binder::new(reader);
    let mut handles = BTreeMap::new();

    for file in FRONTEND_FILES {
        let name = format!("frontend/{file}");
        let handle = binder.require_resource(&name, ResourceEncoding::RawBytesV1)?;
        binder.retain_on_host(handle)?;
        handles.insert(name, handle);
    }

    let mut entries = global_entries(geometry);
    entries.extend(ngram_entries(geometry));
    for layer in 0..geometry.layers {
        entries.extend(layer_entries(geometry, layer));
    }
    for entry in &entries {
        let shape = match &entry.shape {
            ShapeRule::Exact(shape) => shape.clone(),
            ShapeRule::AnyVector => match reader.find(&entry.name) {
                Some(Object::Tensor(t)) if t.shape.len() == 1 => t.shape.clone(),
                Some(Object::Tensor(_)) => {
                    return Err(fail(format!("tensor {} must be a vector", entry.name)))
                }
                _ => return Err(fail(format!("required artifact object is missing: {}", entry.name))),
            },
        };
        let handle = binder.require_tensor(&entry.name, entry.format, entry.layout, &shape)?;
        match entry.role {
            Role::Device => binder.materialize_on_device(handle)?,
            Role::Host => binder.retain_tensor_on_host(handle)?,
            Role::HostStreamed => binder.stream_from_host(handle)?,
        }
        handles.insert(entry.name.clone(), handle);
    }

    let mut records = Vec::with_capacity(geometry.layers * geometry.experts as usize * 2);
    for layer in 0..geometry.layers {
        for expert in 0..geometry.experts {
            for projection in Projection::ALL {
                let name = expert_name(layer, expert, projection);
                let (handle, format) = binder.require_tensor_of(
                    &name,
                    &TrellisK::ALL.map(TrellisK::format),
                    StorageLayout::TrellisTile16V1,
                    &geometry.projection_shape(projection),
                )?;
                let pool_offset = binder.place_in_expert_pool(handle)?;
                let span = binder.payload(handle)?;
                records.push(ExpertRecord {
                    handle,
                    file_offset: span.absolute_offset,
                    bytes: span.data.len() as u64,
                    k: TrellisK::from_format(format).expect("a trellis format"),
                    pool_offset,
                });
            }
        }
    }

    let plan = binder.finish()?;
    Ok(FlashNextPlan {
        plan,
        experts: ExpertIndex {
            experts_per_layer: geometry.experts as usize,
            records,
        },
        handles,
    })
}

/// Check every layer's expert range in the container against the
/// `experts.bin` size and SHA-256 the converter recorded (`experts_bin` in
/// its `converter.json`, merged into the sidecar): the bytes the converter's
/// self-check decoded are the bytes the artifact holds. Returns the layers
/// checked.
pub fn check_experts_sha256(reader: &Reader, index: &ExpertIndex, sidecar: &serde_json::Value) -> Result<usize> {
    use sha2::{Digest, Sha256};
    let records = sidecar
        .get("experts_bin")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| fail("the sidecar records no experts_bin"))?;
    if records.len() != index.layers() {
        return Err(fail(format!(
            "the sidecar records experts_bin for {} layers, the artifact has {}",
            records.len(),
            index.layers()
        )));
    }
    for record in records {
        let field = |key: &str| record.get(key).ok_or_else(|| fail(format!("experts_bin entry has no {key}")));
        let layer = field("layer")?.as_u64().ok_or_else(|| fail("experts_bin layer must be an integer"))? as usize;
        let bytes = field("bytes")?.as_u64().ok_or_else(|| fail("experts_bin bytes must be an integer"))?;
        let sha256 = field("sha256")?.as_str().ok_or_else(|| fail("experts_bin sha256 must be a string"))?;
        let (offset, length) = index
            .layer_range(layer)
            .ok_or_else(|| fail(format!("layer {layer}'s expert records are not one range")))?;
        if length != bytes {
            return Err(fail(format!("layer {layer}: the artifact holds {length} expert bytes, experts.bin had {bytes}")));
        }
        let range = &reader.mapped_bytes()[offset as usize..(offset + length) as usize];
        let digest: String = Sha256::digest(range).iter().map(|b| format!("{b:02x}")).collect();
        if !digest.eq_ignore_ascii_case(sha256) {
            return Err(fail(format!("layer {layer}: expert bytes hash to {digest}, experts.bin to {sha256}")));
        }
    }
    Ok(records.len())
}

#[cfg(test)]
mod tests {
    use super::fixture::{self, fixture_k, pattern, WorkTree};
    use super::*;
    use crate::{materialize, CpuDevice};

    fn open(artifact: &fixture::FixtureArtifact) -> Reader {
        Reader::open(&artifact.path).expect("the packed fixture opens")
    }

    #[test]
    fn the_fixture_artifact_reads_back_with_every_object_in_place() {
        let artifact = fixture::build("read-back").unwrap();
        let reader = open(&artifact);
        assert_eq!(reader.identity().model_id, MODEL_ID);
        // 7 frontend + 5 global + 5 n-gram + layer 0 (GDN: 8 hyper-connection,
        // 9 GDN, 5 MoE) + layer 1 (attention: 8 + 9 + 5, and 6 PLE) + 2
        // layers x 8 experts x 2 projections.
        assert_eq!(reader.objects().len(), 7 + 5 + 5 + 22 + 28 + 32);
        for object in reader.objects() {
            let data = reader.payload_at(object).unwrap().data;
            if let Object::Tensor(t) = object {
                if t.layout == StorageLayout::ContiguousLeV1 && t.format != NumericFormat::Bf16 {
                    continue; // the n-gram vectors hold real values
                }
                assert_eq!(data, pattern(&t.name, t.bytes).as_slice(), "{}", t.name);
            }
        }
        let table = reader.find(&ngram_table_name(&FlashNextGeometry::fixture())).unwrap();
        assert_eq!(table.bytes(), 1_000 * 90, "1,000 rows of 90 bytes");
    }

    #[test]
    fn every_expert_projection_resolves_to_its_offset_size_and_k() {
        let artifact = fixture::build("index").unwrap();
        let reader = open(&artifact);
        let geometry = FlashNextGeometry::fixture();
        let bound = bind(&reader, &geometry).unwrap();
        // The fixture's classes: gate/up [256, 256], down [256, 128], each
        // record trellis + 2·(in + out) bytes of scales, padded to 4096.
        let class_bytes = |projection: Projection, k: TrellisK| match (projection, k) {
            (Projection::GateUp, TrellisK::K2) => 20_480,
            (Projection::GateUp, TrellisK::K2p5) => 24_576,
            (Projection::GateUp, TrellisK::K3) => 28_672,
            (Projection::GateUp, TrellisK::K4) => 36_864,
            (Projection::Down, TrellisK::K2) => 12_288,
            (Projection::Down, TrellisK::K2p5) => 12_288,
            (Projection::Down, TrellisK::K3) => 16_384,
            (Projection::Down, TrellisK::K4) => 20_480,
        };
        let index = &bound.experts;
        assert_eq!((index.layers(), index.experts_per_layer()), (2, 8));
        for layer in 0..2 {
            for expert in 0..8u64 {
                for projection in Projection::ALL {
                    let record = index.get(layer, expert as usize, projection).unwrap();
                    let k = fixture_k(layer, expert, projection);
                    assert_eq!(record.k, k);
                    assert_eq!(record.bytes, class_bytes(projection, k));
                    assert_eq!(record.file_offset % 4096, 0, "one direct read moves it");
                    let start = record.file_offset as usize;
                    let stored = &reader.mapped_bytes()[start..start + record.bytes as usize];
                    let name = expert_name(layer, expert, projection);
                    assert_eq!(stored, pattern(&name, record.bytes).as_slice(), "{name}");
                }
            }
        }
        assert!(index.get(0, 8, Projection::GateUp).is_none());
        assert!(index.get(2, 0, Projection::GateUp).is_none());
        let classes = index.class_counts();
        assert_eq!(classes.len(), 8, "all eight K classes occur");
        assert_eq!(classes.values().sum::<usize>(), 32);
    }

    #[test]
    fn each_layers_expert_bytes_are_the_converters_experts_bin() {
        let artifact = fixture::build("experts-sha").unwrap();
        let reader = open(&artifact);
        let bound = bind(&reader, &FlashNextGeometry::fixture()).unwrap();
        let sidecar_path = crate::packer::sidecar_path(&artifact.path);
        let sidecar: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&sidecar_path).unwrap()).unwrap();
        assert_eq!(check_experts_sha256(&reader, &bound.experts, &sidecar).unwrap(), 2);

        let mut tampered = sidecar.clone();
        tampered["experts_bin"][1]["sha256"] = serde_json::json!("00".repeat(32));
        let err = check_experts_sha256(&reader, &bound.experts, &tampered).unwrap_err().to_string();
        assert!(err.contains("layer 1: expert bytes hash to"), "{err}");
    }

    #[test]
    fn the_trellis_k_and_its_format_code_are_one_mapping() {
        for k in TrellisK::ALL {
            assert_eq!(TrellisK::from_format(k.format()), Some(k));
            assert_eq!(k.format().trellis_k2(), Some(u64::from(k.k2())));
        }
        assert_eq!(TrellisK::from_format(NumericFormat::Bf16), None);
    }

    #[test]
    fn the_full_models_eight_classes_are_the_converters() {
        // layout.md §3's class table.
        let classes: Vec<u64> = k_classes(&FlashNextGeometry::qwen38_flash_next())
            .into_iter()
            .map(|(_, _, bytes)| bytes)
            .collect();
        assert_eq!(
            classes,
            [827_392, 1_032_192, 1_236_992, 1_646_592, 417_792, 520_192, 622_592, 827_392]
        );
    }

    #[test]
    fn the_binder_consumes_every_object_and_places_each_by_its_role() {
        let artifact = fixture::build("roles").unwrap();
        let reader = open(&artifact);
        let bound = bind(&reader, &FlashNextGeometry::fixture()).unwrap();
        let plan = &bound.plan;
        assert_eq!(plan.object_count, reader.objects().len());
        assert_eq!(plan.expert_pool_objects.len(), 32, "every expert projection");
        assert_eq!(plan.streamed_objects.len(), 1, "the table");
        assert_eq!(plan.host_objects.len(), 7, "the frontend resources");
        assert_eq!(plan.host_tensor_objects.len(), 4, "3 hash buffers and the hot rows");
        assert_eq!(plan.device_objects.len(), 99 - 32 - 1 - 7 - 4);
        let pool: u64 = plan.expert_pool_objects.iter().map(|p| p.bytes).sum();
        assert_eq!(plan.expert_pool_capacity_bytes, pool, "4096-sized records pack with no gap");
    }

    #[test]
    fn the_host_streamed_table_and_the_expert_pool_are_not_materialized() {
        let artifact = fixture::build("streamed").unwrap();
        let reader = open(&artifact);
        let geometry = FlashNextGeometry::fixture();
        let bound = bind(&reader, &geometry).unwrap();
        let table = bound.handles[&ngram_table_name(&geometry)];
        let streamed = bound.plan.streamed_objects[0];
        assert_eq!(streamed.handle, table);
        let span = reader.payload(&ngram_table_name(&geometry)).unwrap();
        assert_eq!((streamed.file_offset, streamed.bytes), (span.absolute_offset, 90_000));
        assert_eq!(streamed.file_offset % 4096, 0, "rows are read in place");

        let mut device = CpuDevice::new();
        let loaded = materialize(&reader, &bound.plan, &mut device, None).unwrap();
        assert!(loaded.device_view(table).is_err());
        assert!(loaded.resource_bytes(table).is_err());
        let expert = bound.experts.get(0, 0, Projection::GateUp).unwrap().handle;
        assert!(loaded.device_view(expert).is_err(), "residency fills the pool, not the load");
        assert!(loaded.resource_bytes(expert).is_err());
        let device_bytes: u64 = bound.plan.device_objects.iter().map(|p| p.bytes).sum();
        assert_eq!(loaded.stats().h2d_bytes, device_bytes);
        // The host keeps the hash buffers it hashes with.
        let multipliers = bound.handles[&format!("layers.{}.ple.ple_embedding.layer_multipliers", geometry.ple_layer)];
        let bytes = loaded.host_tensor_bytes(multipliers).unwrap();
        assert_eq!(i64::from_le_bytes(bytes[..8].try_into().unwrap()), 23_703_573_157_769);
        assert!(loaded.resource_bytes(multipliers).is_err(), "a tensor is not a resource");
        assert_eq!(loaded.stats().resource_count, 7, "the frontend only");
    }

    #[test]
    fn an_object_the_inventory_does_not_name_fails_the_bind() {
        let tree = WorkTree::new("stray").unwrap();
        for unit in crate::packer::unit_names(tree.geometry.layers) {
            tree.write_unit_files(&unit).unwrap();
            if unit == "layers/L00" {
                tree.add_stray_tensor(&unit, "layers.0.stray.weight").unwrap();
            }
            tree.mark_done(&unit).unwrap();
        }
        tree.write_converter_json().unwrap();
        let artifact = fixture::build_from(tree).unwrap();
        let reader = open(&artifact);
        let err = bind(&reader, &FlashNextGeometry::fixture()).unwrap_err().to_string();
        assert!(err.contains("not consumed by the selected target: layers.0.stray.weight"), "{err}");
    }

    #[test]
    fn an_artifact_missing_an_object_fails_the_bind() {
        let artifact = fixture::build("missing").unwrap();
        let reader = open(&artifact);
        let mut three_layers = FlashNextGeometry::fixture();
        three_layers.layers = 3;
        let err = bind(&reader, &three_layers).unwrap_err().to_string();
        assert!(err.contains("required artifact object is missing: layers.2."), "{err}");
    }

    /// The MTP head's 29 non-expert tensors are the checkpoint's own `mtp.*`
    /// names (its 31 tensors less the two fused expert tensors), 16 FP8 and
    /// 13 BF16 by the trunk's rule (layout.md §13.3).
    #[test]
    fn the_mtp_inventory_is_the_checkpoints_head_in_the_trunks_formats() {
        let g = FlashNextGeometry::qwen38_flash_next();
        let entries = mtp_entries(&g);
        let fp8 = [
            ("mtp.fc_embedding.weight", [2560, 2560]),
            ("mtp.fc_hidden.weight", [2560, 2560]),
            ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", [320, 10240]),
            ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", [10240, 320]),
            ("mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", [320, 10240]),
            ("mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", [10240, 320]),
            ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", [320, 10240]),
            ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", [10240, 320]),
            ("mtp.layers.0.self_attn.q_proj.weight", [12288, 2560]),
            ("mtp.layers.0.self_attn.k_proj.weight", [512, 2560]),
            ("mtp.layers.0.self_attn.v_proj.weight", [512, 2560]),
            ("mtp.layers.0.self_attn.o_proj.weight", [2560, 6144]),
            ("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", [640, 2560]),
            ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", [640, 2560]),
            ("mtp.layers.0.mlp.shared_expert.up_proj.weight", [640, 2560]),
            ("mtp.layers.0.mlp.shared_expert.down_proj.weight", [2560, 640]),
        ];
        let bf16: [(&str, &[u64]); 13] = [
            ("mtp.pre_fc_norm_embedding.weight", &[2560]),
            ("mtp.pre_fc_norm_hidden.weight", &[10240]),
            ("mtp.hyper_connection_mixer.hc_norm.weight", &[10240]),
            ("mtp.layers.0.attn_hyper_connection.hc_norm.weight", &[10240]),
            ("mtp.layers.0.mlp_hyper_connection.hc_norm.weight", &[10240]),
            ("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", &[4, 10240]),
            ("mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", &[4, 10240]),
            ("mtp.layers.0.self_attn.q_norm.weight", &[256]),
            ("mtp.layers.0.self_attn.k_norm.weight", &[256]),
            ("mtp.layers.0.self_attn.indexer.q_layernorm.weight", &[128]),
            ("mtp.layers.0.self_attn.indexer.k_layernorm.weight", &[128]),
            ("mtp.layers.0.mlp.gate.weight", &[512, 2560]),
            ("mtp.layers.0.mlp.shared_expert_gate.weight", &[1, 2560]),
        ];
        assert_eq!(entries.len(), fp8.len() + bf16.len());
        let find = |name: &str| entries.iter().find(|e| e.name == name).unwrap_or_else(|| panic!("{name} missing"));
        for (name, shape) in fp8 {
            let e = find(name);
            assert_eq!((e.format, &e.shape), (NumericFormat::Fp8E4M3FnRowBf16S, &ShapeRule::Exact(shape.to_vec())), "{name}");
        }
        for (name, shape) in bf16 {
            let e = find(name);
            assert_eq!((e.format, &e.shape), (NumericFormat::Bf16, &ShapeRule::Exact(shape.to_vec())), "{name}");
        }
        assert!(entries.iter().all(|e| e.role == Role::Device));
        assert_eq!(mtp_expert_name(511, Projection::Down), "mtp.layers.0.mlp.experts.511.down_proj");
    }

    #[test]
    fn a_27b_artifact_is_not_bound_as_flash_next() {
        let objects = crate::fixture::all_layout_objects();
        let payload = crate::fixture::all_layout_payload();
        let artifact = crate::fixture::write_fixture(&objects, &payload, "not-flash-next").unwrap();
        let reader = Reader::open(&artifact.path).unwrap();
        let err = bind(&reader, &FlashNextGeometry::fixture()).unwrap_err().to_string();
        assert!(err.contains("is not a Flash-Next artifact"), "{err}");
    }
}
