//! The 27B's NVFP4 artifact family (GitHub #467, spec #449): the whole
//! inventory of an NVFP4 container of the 27B, the served
//! `qwen3_8_27b_nvfp4full-v2.ninfer`'s or a fine-tune's, which the packer's
//! [`crate::packer::Family::Qwen27bNvfp4`] holds a work tree to.
//!
//! The 1,325 objects are the served artifact's, in its directory order:
//! - `frontend/`: the six resources of [`FRONTEND_FILES`];
//! - `text/`: ignis's 27B text scope (the positional BF16 exceptions on the
//!   early attention and GDN layers included), then the draft head pair;
//! - `mtp/`: the checkpoint's MTP layer, W8 and BF16 (ignis binds none of
//!   it; the container carries it as the served one does);
//! - `vision/`: the tower;
//! - `dflash2/`: the drafter module.
//!
//! [`inventory()`] does not call `crate::inventory`'s templates, which are
//! fixed at the 27B's widths: it restates them with a [`Geometry`] argument
//! (`text/` as [`crate::inventory::text_scope_27b`] and
//! [`crate::inventory::draft_head_scope_27b`], `vision/` as
//! [`crate::inventory::vision_scope_27b`], `dflash2/` as
//! [`crate::inventory::dflash2_scope_27b`]), and `tests/nvfp4_27b.rs` holds
//! the two equal, object for object, at the served geometry.
//!
//! The inventory is generated from a [`Geometry`]: the 27B's
//! ([`Geometry::qwen3_8_27b`]), or the converter's reduced synthetic
//! checkpoint ([`Geometry::fixture`]), which has every object, name, format
//! and layout of the served one, only narrower. The layer counts and the
//! exception layers are the model's at both. The work tree's contract is
//! `docs/formats/27b-nvfp4-work-tree.md`.

use crate::inventory::{ATTENTION_OUT_BF16_LAYERS, DFLASH2_LAYERS, GDN_OUT_BF16_LAYERS, QKGV_BF16_LAYERS, VISION_LAYERS};
use crate::{ArtifactIdentity, NumericFormat, StorageLayout};

/// The family's `model_id`: it decides the forward program and the drafter.
/// The weights id tells one build of the layout from another.
pub const MODEL_ID: &str = "qwen3.8-27b";

/// The work tree's units in container order; every object's name starts
/// with its unit and a `/`.
pub const UNITS: [&str; 5] = ["frontend", "text", "mtp", "vision", "dflash2"];

/// The frontend resources (`frontend/<file>`), the checkpoint's files byte
/// for byte, in container order.
pub const FRONTEND_FILES: [&str; 6] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
];

/// The text decoder's layer count; layer `i` is a GQA layer iff
/// `(i + 1) % 4 == 0`, a GDN layer otherwise.
pub const TEXT_LAYERS: usize = 64;

/// The container identity of a build: the family's `model_id` and its own
/// weights id.
pub fn identity(weights_id: &str) -> ArtifactIdentity {
    ArtifactIdentity { model_id: MODEL_ID.into(), weights_id: weights_id.into() }
}

/// Every width the inventory is generated from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    /// The residual stream (and the drafter's) width.
    pub hidden: u64,
    pub vocab: u64,
    /// The MLP's intermediate width (the target's, the MTP layer's and the
    /// drafter's).
    pub intermediate: u64,
    pub gdn_key_heads: u64,
    pub gdn_value_heads: u64,
    pub gdn_head_dim: u64,
    /// The GDN's short causal convolution's taps.
    pub gdn_conv_kernel: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    /// The shortlist draft head's rows.
    pub draft_head_rows: u64,
    pub vision_hidden: u64,
    pub vision_intermediate: u64,
    /// The patch embedding's columns: channels x temporal patch x patch².
    pub vision_patch: u64,
    /// The position table's rows.
    pub vision_positions: u64,
    pub dflash2_heads: u64,
    pub dflash2_kv_heads: u64,
    pub dflash2_head_dim: u64,
    /// The target layers the drafter reads (its feature projection takes
    /// `taps x hidden`).
    pub dflash2_taps: u64,
    /// The rows of each `conv_proj` (the 27B's 320 groups x 2 taps x 2).
    pub dflash2_conv_rows: u64,
    pub dflash2_selector_rank: u64,
}

impl Geometry {
    /// `Qwen/Qwen3.8-27B` and its DFlash2 drafter: the served artifact's.
    pub fn qwen3_8_27b() -> Self {
        Self {
            hidden: 5120,
            vocab: 248_320,
            intermediate: 17_408,
            gdn_key_heads: 16,
            gdn_value_heads: 48,
            gdn_head_dim: 128,
            gdn_conv_kernel: 4,
            attention_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            draft_head_rows: 131_072,
            vision_hidden: 1152,
            vision_intermediate: 4304,
            vision_patch: 3 * 2 * 16 * 16,
            vision_positions: 48 * 48,
            dflash2_heads: 32,
            dflash2_kv_heads: 8,
            dflash2_head_dim: 128,
            dflash2_taps: 5,
            dflash2_conv_rows: 1280,
            dflash2_selector_rank: 256,
        }
    }

    /// The NVFP4 converter's reduced synthetic checkpoint: every object of
    /// the served inventory, each NVFP4 matrix at least one 128 x 64 block
    /// (`blockscale-k16-m128x4-v1`'s unit), the tree a few MB.
    pub fn fixture() -> Self {
        Self {
            hidden: 128,
            vocab: 256,
            intermediate: 64,
            gdn_key_heads: 1,
            gdn_value_heads: 3,
            gdn_head_dim: 64,
            gdn_conv_kernel: 4,
            attention_heads: 4,
            kv_heads: 2,
            head_dim: 32,
            draft_head_rows: 128,
            vision_hidden: 32,
            vision_intermediate: 64,
            vision_patch: 3 * 2 * 4 * 4,
            vision_positions: 4 * 4,
            dflash2_heads: 2,
            dflash2_kv_heads: 1,
            dflash2_head_dim: 32,
            dflash2_taps: 5,
            dflash2_conv_rows: 128,
            dflash2_selector_rank: 128,
        }
    }
}

/// One object of the inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// A `raw-bytes-v1` resource.
    Resource { name: String },
    Tensor { name: String, format: NumericFormat, layout: StorageLayout, shape: Vec<u64> },
}

impl Entry {
    pub fn name(&self) -> &str {
        match self {
            Entry::Resource { name } | Entry::Tensor { name, .. } => name,
        }
    }

    /// The work-tree unit that holds it: its name up to the first `/`.
    pub fn unit(&self) -> &str {
        self.name().split('/').next().unwrap_or_default()
    }
}

const BF16: (NumericFormat, StorageLayout) = (NumericFormat::Bf16, StorageLayout::ContiguousLeV1);
const FP32: (NumericFormat, StorageLayout) = (NumericFormat::Fp32, StorageLayout::ContiguousLeV1);
const I32: (NumericFormat, StorageLayout) = (NumericFormat::I32, StorageLayout::ContiguousLeV1);
const NVFP4: (NumericFormat, StorageLayout) = (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1);
const W8: (NumericFormat, StorageLayout) = (NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1);
const Q4: (NumericFormat, StorageLayout) = (NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1);
const Q5: (NumericFormat, StorageLayout) = (NumericFormat::Q5G64F16S, StorageLayout::RowSplitK128V1);
const Q6: (NumericFormat, StorageLayout) = (NumericFormat::Q6G64F16S, StorageLayout::RowSplitK128V1);

/// The objects in container order, collected by [`inventory`].
struct Objects(Vec<Entry>);

impl Objects {
    fn tensor(&mut self, name: impl Into<String>, (format, layout): (NumericFormat, StorageLayout), shape: &[u64]) {
        self.0.push(Entry::Tensor { name: name.into(), format, layout, shape: shape.to_vec() });
    }

    /// A text-scope projection: NVFP4 with its activation divisor right
    /// after it (a rank-0 FP32 `<site>/input_scale_divisor`), or one of the
    /// positional BF16 exceptions, which have none.
    fn projection(&mut self, name: String, bf16: bool, site: String, shape: &[u64]) {
        if bf16 {
            self.tensor(name, BF16, shape);
        } else {
            self.tensor(name, NVFP4, shape);
            self.tensor(format!("{site}/input_scale_divisor"), FP32, &[]);
        }
    }
}

/// The inventory at `g`: every object of the container, in its order.
pub fn inventory(g: &Geometry) -> Vec<Entry> {
    let h = g.hidden;
    let mut o = Objects(Vec::with_capacity(1325));
    for file in FRONTEND_FILES {
        o.0.push(Entry::Resource { name: format!("frontend/{file}") });
    }

    // text/: the layers between the two W8 endpoints, then the draft head.
    let gdn_qk = g.gdn_key_heads * g.gdn_head_dim;
    let gdn_v = g.gdn_value_heads * g.gdn_head_dim;
    let (q, kv) = (g.attention_heads * g.head_dim, g.kv_heads * g.head_dim);
    o.tensor("text/token_embedding", W8, &[g.vocab, h]);
    for i in 0..TEXT_LAYERS {
        let n = |suffix: &str| format!("text/layers/{i}/{suffix}");
        o.tensor(n("input_norm"), BF16, &[h]);
        if (i + 1) % 4 == 0 {
            o.projection(n("attention/query_key_gate_value"), QKGV_BF16_LAYERS.contains(&i), n("attention/input_projection"), &[2 * q + 2 * kv, h]);
            o.tensor(n("attention/query_norm"), BF16, &[g.head_dim]);
            o.tensor(n("attention/key_norm"), BF16, &[g.head_dim]);
            o.projection(n("attention/output"), ATTENTION_OUT_BF16_LAYERS.contains(&i), n("attention/output_projection"), &[h, q]);
        } else {
            o.tensor(n("gdn/a_log"), FP32, &[g.gdn_value_heads]);
            o.tensor(n("gdn/dt_bias"), FP32, &[g.gdn_value_heads]);
            o.tensor(n("gdn/convolution"), BF16, &[g.gdn_conv_kernel, 2 * gdn_qk + gdn_v]);
            o.tensor(n("gdn/a_b_projection"), BF16, &[2 * g.gdn_value_heads, h]);
            o.projection(n("gdn/query_key_value_z"), false, n("gdn/input_projection"), &[2 * gdn_qk + 2 * gdn_v, h]);
            o.tensor(n("gdn/norm"), BF16, &[g.gdn_head_dim]);
            o.projection(n("gdn/output"), GDN_OUT_BF16_LAYERS.contains(&i), n("gdn/output_projection"), &[h, gdn_v]);
        }
        o.tensor(n("post_attention_norm"), BF16, &[h]);
        o.projection(n("mlp/gate_up"), false, n("mlp/gate_up_projection"), &[2 * g.intermediate, h]);
        o.projection(n("mlp/down"), false, n("mlp/down_projection"), &[h, g.intermediate]);
    }
    o.tensor("text/final_norm", BF16, &[h]);
    o.tensor("text/output_head", W8, &[g.vocab, h]);
    o.tensor("text/draft_head", Q4, &[g.draft_head_rows, h]);
    o.tensor("text/draft_head_token_ids", I32, &[g.draft_head_rows]);

    // mtp/: the checkpoint's one MTP layer, its matrices W8.
    o.tensor("mtp/input_projection", W8, &[h, 2 * h]);
    o.tensor("mtp/embedding_norm", BF16, &[h]);
    o.tensor("mtp/hidden_norm", BF16, &[h]);
    o.tensor("mtp/layer/input_norm", BF16, &[h]);
    o.tensor("mtp/layer/attention/query_key_gate_value", W8, &[2 * q + 2 * kv, h]);
    o.tensor("mtp/layer/attention/query_norm", BF16, &[g.head_dim]);
    o.tensor("mtp/layer/attention/key_norm", BF16, &[g.head_dim]);
    o.tensor("mtp/layer/attention/output", W8, &[h, q]);
    o.tensor("mtp/layer/post_attention_norm", BF16, &[h]);
    o.tensor("mtp/layer/mlp/gate_up", W8, &[2 * g.intermediate, h]);
    o.tensor("mtp/layer/mlp/down", W8, &[h, g.intermediate]);
    o.tensor("mtp/final_norm", BF16, &[h]);

    // vision/: the tower, its merger writing the text model's width.
    o.0.extend(vision_tower(&g.vision_tower()));

    // dflash2/: weight-only NVFP4 matrices (no divisor sites), BF16 norms
    // and conv base kernels.
    let dq = g.dflash2_heads * g.dflash2_head_dim;
    let dkv = g.dflash2_kv_heads * g.dflash2_head_dim;
    o.tensor("dflash2/feature_projection", NVFP4, &[h, g.dflash2_taps * h]);
    o.tensor("dflash2/context_norm", BF16, &[h]);
    for l in 0..DFLASH2_LAYERS {
        let n = |suffix: &str| format!("dflash2/layers/{l}/{suffix}");
        o.tensor(n("input_norm"), BF16, &[h]);
        o.tensor(n("attention/query_key_value"), NVFP4, &[dq + 2 * dkv, h]);
        o.tensor(n("attention/query_norm"), BF16, &[g.dflash2_head_dim]);
        o.tensor(n("attention/key_norm"), BF16, &[g.dflash2_head_dim]);
        o.tensor(n("attention/output"), NVFP4, &[h, dq]);
        o.tensor(n("attention/conv_base"), BF16, &[2, 2, h]);
        o.tensor(n("attention/conv_proj"), NVFP4, &[g.dflash2_conv_rows, h]);
        o.tensor(n("post_attention_norm"), BF16, &[h]);
        o.tensor(n("mlp/gate_up"), NVFP4, &[2 * g.intermediate, h]);
        o.tensor(n("mlp/down"), NVFP4, &[h, g.intermediate]);
        o.tensor(n("mlp/conv_base"), BF16, &[2, 2, h]);
        o.tensor(n("mlp/conv_proj"), NVFP4, &[g.dflash2_conv_rows, h]);
    }
    o.tensor("dflash2/final_norm", BF16, &[h]);
    o.tensor("dflash2/selector/hidden", NVFP4, &[g.dflash2_selector_rank, h]);
    o.tensor("dflash2/selector/predecessor", NVFP4, &[g.vocab, g.dflash2_selector_rank]);
    o.tensor("dflash2/selector/successor", NVFP4, &[g.vocab, g.dflash2_selector_rank]);
    o.0
}

/// A vision tower's widths: the 27B's ([`Geometry::vision_tower`]), or
/// another model's tower encoded with the 27B's recipe (Flash-Next's,
/// [`crate::flash_next::vision::geometry`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTower {
    pub hidden: u64,
    pub intermediate: u64,
    /// The patch embedding's columns: channels x temporal patch x patch².
    pub patch: u64,
    /// The position table's rows.
    pub positions: u64,
    /// The merger's output: the text model's hidden width.
    pub out_hidden: u64,
}

impl Geometry {
    /// The 27B's tower, its merger writing the 27B's hidden width.
    pub fn vision_tower(&self) -> VisionTower {
        VisionTower {
            hidden: self.vision_hidden,
            intermediate: self.vision_intermediate,
            patch: self.vision_patch,
            positions: self.vision_positions,
            out_hidden: self.hidden,
        }
    }
}

/// The tower's objects in container order, each object kind in the 27B
/// recipe's format: the patch embedding Q6, the blocks' attention input and
/// MLP first linear Q4, their output linears Q5, the 2 x 2 merger's linears
/// W8; the biases, norms and position table BF16.
pub fn vision_tower(t: &VisionTower) -> Vec<Entry> {
    let (vh, vi) = (t.hidden, t.intermediate);
    let merged = 4 * vh;
    let mut o = Objects(Vec::with_capacity(3 + VISION_LAYERS * 12 + 6));
    o.tensor("vision/patch_embedding", Q6, &[vh, t.patch]);
    o.tensor("vision/patch_embedding_bias", BF16, &[vh]);
    o.tensor("vision/position_embedding", BF16, &[t.positions, vh]);
    for b in 0..VISION_LAYERS {
        let n = |suffix: &str| format!("vision/layers/{b}/{suffix}");
        o.tensor(n("attention/qkv"), Q4, &[3 * vh, vh]);
        o.tensor(n("attention/qkv_bias"), BF16, &[3 * vh]);
        o.tensor(n("attention/output"), Q5, &[vh, vh]);
        o.tensor(n("attention/output_bias"), BF16, &[vh]);
        o.tensor(n("mlp/fc1"), Q4, &[vi, vh]);
        o.tensor(n("mlp/fc1_bias"), BF16, &[vi]);
        o.tensor(n("mlp/fc2"), Q5, &[vh, vi]);
        o.tensor(n("mlp/fc2_bias"), BF16, &[vh]);
        for norm in ["norm1", "norm2"] {
            o.tensor(n(&format!("{norm}/weight")), BF16, &[vh]);
            o.tensor(n(&format!("{norm}/bias")), BF16, &[vh]);
        }
    }
    o.tensor("vision/merger/fc1", W8, &[merged, merged]);
    o.tensor("vision/merger/fc1_bias", BF16, &[merged]);
    o.tensor("vision/merger/fc2", W8, &[t.out_hidden, merged]);
    o.tensor("vision/merger/fc2_bias", BF16, &[t.out_hidden]);
    o.tensor("vision/merger/norm/weight", BF16, &[vh]);
    o.tensor("vision/merger/norm/bias", BF16, &[vh]);
    o.0
}
