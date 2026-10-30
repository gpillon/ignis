//! The Qwen 3.8-27B **text-scope** tensor inventory (A1, spec 04).
//!
//! The 27B artifact (`qwen3_8_27b_nvfp4full-v2.ninfer`, ADR 0002: the
//! container is the authority for the per-tensor `format` + `layout`)
//! carries 1,319 tensors; the v1 *text* inference scope is the `text/*`
//! objects minus the draft-head pair (`text/draft_head` +
//! `text/draft_head_token_ids` — spec 04 non-goal: noted, not normalized
//! for v1 text). The 906 in-scope tensors are generated parametrically
//! from the model constants (the 64 layers: layer `i` is GQA iff
//! `(i + 1) % 4 == 0`) plus the artifact's directory facts (the early-
//! attention BF16 projections, the layer-4 `gdn/output` quirk) — the
//! [`text_scope_27b_is_complete`] test pins the generated table against
//! the format rule + the geometry checks, and the `tests/real_artifact`
//! cross-check pins it against the container's actual directory (the
//! authority, ADR 0002).
//!
//! **The format distribution (the A1 rule's input space, spec 04):** 2
//! W8G32 (`text/token_embedding` + `text/output_head`), 314 BF16, 343
//! FP32, 247 NVFP4 (the 27B model's GEMM weights + the norms / GDN
//! params / `*_input_scale_divisor` scalars).

use crate::binder::{Binder, DeviceEncoding, MaterializationPlan, ObjectHandle, Residence};
use crate::{NumericFormat, Object, Reader, Result, StorageLayout};

// ---------------------------------------------------------------------------
// The entry
// ---------------------------------------------------------------------------

/// One text-scope tensor entry (the container's directory projection:
/// `name` + `format` + `layout` + `shape`), and where the model keeps it.
///
/// The `name` is a `'static` string (the layer-templated names are leaked
/// once by the generator — bounded by the table size, a one-time cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InventoryEntry {
    /// The object's directory name (e.g. `text/layers/3/gdn/norm`).
    pub name: &'static str,
    pub format: NumericFormat,
    pub layout: StorageLayout,
    /// The stored shape (rank-0 for the `*_input_scale_divisor` scalars).
    pub shape: &'static [u64],
    /// Where the load places it: the device, but for the token embedding,
    /// which lives in mapped host memory (ADR 0062).
    pub residence: Residence,
    /// How a device tensor is held: as stored, or (GitHub #341) a BF16
    /// exception tensor converted to FP8 at load ([`ModelScope::fp8_exceptions`]).
    pub encoding: DeviceEncoding,
}

// ---------------------------------------------------------------------------
// The shape table (the model constants; every in-scope tensor's shape)
// ---------------------------------------------------------------------------

const SHAPE_NONE: [u64; 0] = []; // the `*_input_scale_divisor` scalars
const SH_HIDDEN: [u64; 1] = [5120]; // the norm / residual-stream width
const SH_GDN_HEADS: [u64; 1] = [48]; // the GDN recurrence params (48 heads)
const SH_GDN_NORM: [u64; 1] = [128]; // the GDN layer's short-norm width
const SH_ATTN_NORM: [u64; 1] = [256]; // the GQA layer's q/k-norm width
const SH_EMBED: [u64; 2] = [248320, 5120]; // vocab x hidden (the W8 endpoints)
const SH_GDN_QKVZ: [u64; 2] = [16384, 5120]; // the GDN input projection (k + v + z)
const SH_GDN_OUT: [u64; 2] = [5120, 6144]; // the GDN state readout
const SH_QKGV: [u64; 2] = [14336, 5120]; // the fused q + k + v + gate
const SH_ATTN_OUT: [u64; 2] = [5120, 6144]; // the GQA output projection
const SH_MLP_GATE_UP: [u64; 2] = [34816, 5120]; // the fused gate + up
const SH_MLP_DOWN: [u64; 2] = [5120, 17408];
const SH_GDN_CONV: [u64; 2] = [4, 10240]; // the short causal-conv kernel
const SH_GDN_AB: [u64; 2] = [96, 5120]; // the GDN a/b projection

/// The GQA layers whose `attention/query_key_gate_value` projection is
/// BF16 (the "early-attention exception" — the artifact's directory: the
/// first six GQA layers, 3 through 23).
const QKGV_BF16_LAYERS: &[usize] = &[3, 7, 11, 15, 19, 23];

/// The GQA layers whose `attention/output` projection is BF16 (the
/// artifact's directory: the first two GQA layers, 3 and 7).
const ATTENTION_OUT_BF16_LAYERS: &[usize] = &[3, 7];

/// The GDN layers whose `gdn/output` projection is BF16 (the artifact's
/// directory quirk: layer 4 — its `gdn/output` is BF16, so the
/// `gdn/output_projection/input_scale_divisor` object is absent).
const GDN_OUT_BF16_LAYERS: &[usize] = &[4];

/// GitHub #341: the row blocks a BF16 exception tensor converts to FP8 in --
/// the fused projection's q | k | gate | v (its stored row order), and an
/// output projection whole. The leaf reads the same blocks back
/// (`kQkgvSegmentRows`, kernel/src/fp8_exception.h).
const QKGV_FP8_BLOCKS: &[u32] = &[6144, 1024, 6144, 1024];
const OUTPUT_FP8_BLOCKS: &[u32] = &[5120];

/// The out-of-scope text objects (spec 04 non-goals: the draft-head
/// endpoints — noted, not normalized for v1 text).
pub const OUT_OF_SCOPE_TEXT_NAMES: &[&str] = &["text/draft_head", "text/draft_head_token_ids"];

// ---------------------------------------------------------------------------
// The layer templates (the artifact's directory order)
// ---------------------------------------------------------------------------

/// The GDN-layer template (15 entries; the layer-4 variant is 14 — its
/// `gdn/output` is BF16, and a BF16 projection carries no
/// `input_scale_divisor` object).
fn gdn_layer(i: usize) -> Vec<InventoryEntry> {
    let prefix = format!("text/layers/{i}/");
    let name = |suffix: &str| format!("{prefix}{suffix}");
    let mut entries: Vec<InventoryEntry> = Vec::with_capacity(15);
    entries.push(mk(&name("input_norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_HIDDEN));
    entries.push(mk(&name("gdn/a_log"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SH_GDN_HEADS));
    entries.push(mk(&name("gdn/dt_bias"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SH_GDN_HEADS));
    entries.push(mk(&name("gdn/convolution"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_GDN_CONV));
    entries.push(mk(&name("gdn/a_b_projection"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_GDN_AB));
    entries.push(mk(&name("gdn/query_key_value_z"), NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &SH_GDN_QKVZ));
    entries.push(mk(&name("gdn/input_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    entries.push(mk(&name("gdn/norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_GDN_NORM));
    // The state readout (NVFP4 in every layer except the layer-4 BF16 quirk).
    let gdn_out = if GDN_OUT_BF16_LAYERS.contains(&i) {
        (NumericFormat::Bf16, StorageLayout::ContiguousLeV1)
    } else {
        (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1)
    };
    entries.push(mk(&name("gdn/output"), gdn_out.0, gdn_out.1, &SH_GDN_OUT));
    // The readout's divisor exists exactly when the projection is NVFP4.
    if gdn_out.0 == NumericFormat::Nvfp4 {
        entries.push(mk(&name("gdn/output_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    }
    entries.push(mk(&name("post_attention_norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_HIDDEN));
    entries.push(mk(&name("mlp/gate_up"), NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &SH_MLP_GATE_UP));
    entries.push(mk(&name("mlp/gate_up_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    entries.push(mk(&name("mlp/down"), NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &SH_MLP_DOWN));
    entries.push(mk(&name("mlp/down_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    entries
}

/// The GQA-layer template (10 / 11 / 12 entries, by the attention
/// projections' directory formats: the early layers are BF16 — no
/// `*_input_scale_divisor` objects — the later ones NVFP4, with a
/// divisor per NVFP4 projection).
fn gqa_layer(i: usize) -> Vec<InventoryEntry> {
    let prefix = format!("text/layers/{i}/");
    let name = |suffix: &str| format!("{prefix}{suffix}");
    let qkgv = if QKGV_BF16_LAYERS.contains(&i) {
        (NumericFormat::Bf16, StorageLayout::ContiguousLeV1)
    } else {
        (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1)
    };
    let out = if ATTENTION_OUT_BF16_LAYERS.contains(&i) {
        (NumericFormat::Bf16, StorageLayout::ContiguousLeV1)
    } else {
        (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1)
    };
    let mut entries: Vec<InventoryEntry> = Vec::with_capacity(12);
    entries.push(mk(&name("input_norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_HIDDEN));
    entries.push(mk(&name("attention/query_key_gate_value"), qkgv.0, qkgv.1, &SH_QKGV));
    // The fused-projection divisor (NVFP4 `query_key_gate_value` only).
    if qkgv.0 == NumericFormat::Nvfp4 {
        entries.push(mk(&name("attention/input_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    }
    entries.push(mk(&name("attention/query_norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_ATTN_NORM));
    entries.push(mk(&name("attention/key_norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_ATTN_NORM));
    entries.push(mk(&name("attention/output"), out.0, out.1, &SH_ATTN_OUT));
    // The output-projection divisor (NVFP4 `attention/output` only).
    if out.0 == NumericFormat::Nvfp4 {
        entries.push(mk(&name("attention/output_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    }
    entries.push(mk(&name("post_attention_norm"), NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_HIDDEN));
    entries.push(mk(&name("mlp/gate_up"), NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &SH_MLP_GATE_UP));
    entries.push(mk(&name("mlp/gate_up_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    entries.push(mk(&name("mlp/down"), NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &SH_MLP_DOWN));
    entries.push(mk(&name("mlp/down_projection/input_scale_divisor"), NumericFormat::Fp32, StorageLayout::ContiguousLeV1, &SHAPE_NONE));
    entries
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/// The 27B text-scope tensor inventory (the 906 in-scope tensors: the
/// top-level `text/*` objects + the 64 `text/layers/{i}` tensors; the
/// draft-head pair is out of scope, spec 04).
///
/// The layer pattern is the model constant (`i` is a GQA layer iff
/// `(i + 1) % 4 == 0`); the attention-projection formats + the layer-4
/// quirk are the artifact's directory facts (pinned by the `tests/
/// real_artifact` cross-check against the container, ADR 0002).
pub fn text_scope_27b() -> Vec<InventoryEntry> {
    let mut entries = Vec::with_capacity(906);
    // The directory order: `token_embedding` first, then the 64 layers,
    // then the top-level tails (the container's actual order).
    // ADR 0062: a token reads one 5,440-byte row of the 1.35 GB table, so the
    // table lives in mapped host memory and the gather reads it over the bus.
    entries.push(InventoryEntry {
        residence: Residence::HostMapped,
        ..mk("text/token_embedding", NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1, &SH_EMBED)
    });
    for i in 0..64usize {
        // Layer `i` is GQA iff `(i + 1) % 4 == 0` (the model constant).
        entries.extend(if (i + 1) % 4 == 0 { gqa_layer(i) } else { gdn_layer(i) });
    }
    entries.push(mk("text/final_norm", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_HIDDEN));
    entries.push(mk("text/output_head", NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1, &SH_EMBED));
    entries
}

/// Bind every text-scope tensor ([`text_scope_27b`]) against `reader`,
/// materializing each on the device (P1-17, GitHub #53) but the token
/// embedding, placed in mapped host memory (ADR 0062).
///
/// Loudly rejects a missing or mis-shaped text-scope object (a
/// [`Binder::require_tensor`] mismatch): the exact (format, layout, shape)
/// this function requires is the model's contract, not the container's, so
/// a container that stores a different shape under the same name is a load
/// failure, never a silent reinterpretation.
///
/// Scoped to the text objects: the artifact's non-text objects (frontend
/// resources, vision / MTP tensors) are left unconsumed — [`Binder::plan`]
/// (not [`Binder::finish`]) builds the plan, since this binder does not own
/// the whole directory (ADR 0002's "every object consumed" contract is
/// [`Binder::finish`]'s, for a binder that does).
///
/// Returns the plan (in [`text_scope_27b`] order) alongside the matching
/// handles, so a caller can zip them with the inventory's names / formats
/// when building the model-load descriptors.
pub fn bind_text_scope_27b(reader: &Reader) -> Result<(MaterializationPlan, Vec<ObjectHandle>)> {
    bind_model_scope_27b(reader, None)
}

// ---------------------------------------------------------------------------
// The DFlash2 drafter module (P5-02, GitHub #150)
// ---------------------------------------------------------------------------

/// A drafter module the v2 artifact carries beside the text scope, bound
/// only when a load selects speculation (spec 05: engine residency, chosen at
/// load). Without one the binder leaves the module's objects unconsumed,
/// exactly as before it existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftModule {
    /// The 66 `dflash2/*` objects (the reference's
    /// `docs/maintainer/qwen3.8-27b-artifact.md` §15).
    Dflash2,
    /// [`DraftModule::Dflash2`] plus the shortlist proposal head its drafter
    /// scores with instead of the target's output head: `text/draft_head` and
    /// `text/draft_head_token_ids` ([`draft_head_scope_27b`]).
    Dflash2ShortlistHead,
}

/// The drafter's layer count (§15.1: five sliding-attention layers).
pub const DFLASH2_LAYERS: usize = 5;

const SH_DF_FEATURES: [u64; 2] = [5120, 25600]; // hidden x (5 target taps x hidden)
const SH_DF_QKV: [u64; 2] = [6144, 5120]; // concat(q 32x128, k 8x128, v 8x128)
const SH_DF_HEAD_NORM: [u64; 1] = [128]; // the drafter's q/k-norm width
const SH_DF_ATTN_OUT: [u64; 2] = [5120, 4096];
const SH_DF_CONV_BASE: [u64; 3] = [2, 2, 5120]; // the two-tap dynamic conv's base kernel
const SH_DF_CONV_PROJ: [u64; 2] = [1280, 5120]; // 320 groups x 2 taps x 2
const SH_DF_SELECTOR_HIDDEN: [u64; 2] = [256, 5120]; // the selector's rank-256 projection
const SH_DF_CODEBOOK: [u64; 2] = [248320, 256]; // the per-token predecessor/successor codebooks

/// The 66 DFlash2 module objects, in the artifact's directory order (two
/// globals, five layer groups of twelve, four closing globals). Matrices are
/// weight-only NVFP4 — the drafter runs A16, so unlike the text scope no
/// `*_input_scale_divisor` object pairs with them — and norms and conv base
/// kernels are BF16 (34 NVFP4 + 32 BF16). The `tests/real_artifact`
/// cross-check pins this table against the container (ADR 0002).
pub fn dflash2_scope_27b() -> Vec<InventoryEntry> {
    let nvfp4 = (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1);
    let bf16 = (NumericFormat::Bf16, StorageLayout::ContiguousLeV1);
    let mut entries = Vec::with_capacity(66);
    let mut push = |name: &str, (format, layout): (NumericFormat, StorageLayout), shape: &'static [u64]| {
        entries.push(mk(name, format, layout, shape));
    };
    push("dflash2/feature_projection", nvfp4, &SH_DF_FEATURES);
    push("dflash2/context_norm", bf16, &SH_HIDDEN);
    for l in 0..DFLASH2_LAYERS {
        let name = |suffix: &str| format!("dflash2/layers/{l}/{suffix}");
        push(&name("input_norm"), bf16, &SH_HIDDEN);
        push(&name("attention/query_key_value"), nvfp4, &SH_DF_QKV);
        push(&name("attention/query_norm"), bf16, &SH_DF_HEAD_NORM);
        push(&name("attention/key_norm"), bf16, &SH_DF_HEAD_NORM);
        push(&name("attention/output"), nvfp4, &SH_DF_ATTN_OUT);
        push(&name("attention/conv_base"), bf16, &SH_DF_CONV_BASE);
        push(&name("attention/conv_proj"), nvfp4, &SH_DF_CONV_PROJ);
        push(&name("post_attention_norm"), bf16, &SH_HIDDEN);
        push(&name("mlp/gate_up"), nvfp4, &SH_MLP_GATE_UP);
        push(&name("mlp/down"), nvfp4, &SH_MLP_DOWN);
        push(&name("mlp/conv_base"), bf16, &SH_DF_CONV_BASE);
        push(&name("mlp/conv_proj"), nvfp4, &SH_DF_CONV_PROJ);
    }
    push("dflash2/final_norm", bf16, &SH_HIDDEN);
    push("dflash2/selector/hidden", nvfp4, &SH_DF_SELECTOR_HIDDEN);
    push("dflash2/selector/predecessor", nvfp4, &SH_DF_CODEBOOK);
    push("dflash2/selector/successor", nvfp4, &SH_DF_CODEBOOK);
    entries
}

const SH_DRAFT_HEAD: [u64; 2] = [131072, 5120]; // the shortlist rows x hidden
const SH_DRAFT_HEAD_IDS: [u64; 1] = [131072]; // one token id per shortlist row

/// The shortlist proposal head, in the artifact's directory order: the Q4
/// row-split head over the 131,072 most frequent tokens and the token id each
/// of its rows stands for. They are the [`OUT_OF_SCOPE_TEXT_NAMES`] pair: out
/// of the text scope, which never reads them, and bound only with
/// [`DraftModule::Dflash2ShortlistHead`].
pub fn draft_head_scope_27b() -> Vec<InventoryEntry> {
    vec![
        mk("text/draft_head", NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1, &SH_DRAFT_HEAD),
        mk("text/draft_head_token_ids", NumericFormat::I32, StorageLayout::ContiguousLeV1, &SH_DRAFT_HEAD_IDS),
    ]
}

/// Every object a load binds: the text scope, then the drafter module when
/// one is selected. The text entries are [`text_scope_27b`] unchanged and
/// first, so a drafter-bearing plan places every text tensor at the offset a
/// text-only plan does.
pub fn model_scope_27b(draft: Option<DraftModule>) -> Vec<InventoryEntry> {
    with_modules(text_scope_27b(), ModelScope { draft, ..ModelScope::default() })
}

/// [`bind_text_scope_27b`], plus the drafter module's objects when `draft`
/// names one (P5-02, GitHub #150). A missing or mis-shaped drafter object is
/// a load failure naming it, the same contract as the text scope's. Returns
/// the handles in [`model_scope_27b`] order.
pub fn bind_model_scope_27b(
    reader: &Reader,
    draft: Option<DraftModule>,
) -> Result<(MaterializationPlan, Vec<ObjectHandle>)> {
    bind_model_scope_27b_with(reader, ModelScope { draft, vision: false, vision_residency: VisionResidency::Device, fp8_exceptions: false })
}

// ---------------------------------------------------------------------------
// The vision tower (GitHub #177)
// ---------------------------------------------------------------------------

/// The vision encoder's block count.
pub const VISION_LAYERS: usize = 27;

const SH_V_HIDDEN: [u64; 1] = [1152]; // the encoder width
const SH_V_PATCH: [u64; 2] = [1152, 1536]; // hidden x (3 x 2 x 16 x 16)
const SH_V_POSITION: [u64; 2] = [2304, 1152]; // the 48 x 48 position table
const SH_V_QKV: [u64; 2] = [3456, 1152];
const SH_V_QKV_BIAS: [u64; 1] = [3456];
const SH_V_SQUARE: [u64; 2] = [1152, 1152];
const SH_V_FC1: [u64; 2] = [4304, 1152];
const SH_V_FC1_BIAS: [u64; 1] = [4304];
const SH_V_FC2: [u64; 2] = [1152, 4304];
const SH_V_MERGER: [u64; 2] = [4608, 4608]; // the 2 x 2 merge block
const SH_V_MERGER_BIAS: [u64; 1] = [4608];
const SH_V_MERGER_OUT: [u64; 2] = [5120, 4608]; // into the target's hidden width

/// The 333 `vision/*` objects in the artifact's directory order (the
/// reference's `qwen3.8-27b-artifact.md` §vision): patch embedding, bias and
/// position table, 27 blocks of twelve, then the merger. Stored natively —
/// Q6 patch embedding, Q4 qkv/fc1, Q5 output/fc2, W8 merger, BF16 biases and
/// norms, every quantized one row-split. The `tests/real_artifact` cross-check
/// pins this table against the container (ADR 0002).
pub fn vision_scope_27b() -> Vec<InventoryEntry> {
    let row_split = StorageLayout::RowSplitK128V1;
    let q4 = (NumericFormat::Q4G64F16S, row_split);
    let q5 = (NumericFormat::Q5G64F16S, row_split);
    let q6 = (NumericFormat::Q6G64F16S, row_split);
    let w8 = (NumericFormat::W8G32F16S, row_split);
    let bf16 = (NumericFormat::Bf16, StorageLayout::ContiguousLeV1);
    let mut entries = Vec::with_capacity(3 + VISION_LAYERS * 12 + 6);
    let mut push = |name: &str, (format, layout): (NumericFormat, StorageLayout), shape: &'static [u64]| {
        entries.push(mk(name, format, layout, shape));
    };
    push("vision/patch_embedding", q6, &SH_V_PATCH);
    push("vision/patch_embedding_bias", bf16, &SH_V_HIDDEN);
    push("vision/position_embedding", bf16, &SH_V_POSITION);
    for b in 0..VISION_LAYERS {
        let name = |suffix: &str| format!("vision/layers/{b}/{suffix}");
        push(&name("attention/qkv"), q4, &SH_V_QKV);
        push(&name("attention/qkv_bias"), bf16, &SH_V_QKV_BIAS);
        push(&name("attention/output"), q5, &SH_V_SQUARE);
        push(&name("attention/output_bias"), bf16, &SH_V_HIDDEN);
        push(&name("mlp/fc1"), q4, &SH_V_FC1);
        push(&name("mlp/fc1_bias"), bf16, &SH_V_FC1_BIAS);
        push(&name("mlp/fc2"), q5, &SH_V_FC2);
        push(&name("mlp/fc2_bias"), bf16, &SH_V_HIDDEN);
        push(&name("norm1/weight"), bf16, &SH_V_HIDDEN);
        push(&name("norm1/bias"), bf16, &SH_V_HIDDEN);
        push(&name("norm2/weight"), bf16, &SH_V_HIDDEN);
        push(&name("norm2/bias"), bf16, &SH_V_HIDDEN);
    }
    push("vision/merger/fc1", w8, &SH_V_MERGER);
    push("vision/merger/fc1_bias", bf16, &SH_V_MERGER_BIAS);
    push("vision/merger/fc2", w8, &SH_V_MERGER_OUT);
    push("vision/merger/fc2_bias", bf16, &SH_HIDDEN);
    push("vision/merger/norm/weight", bf16, &SH_V_HIDDEN);
    push("vision/merger/norm/bias", bf16, &SH_V_HIDDEN);
    entries
}

/// Where a vision load keeps the vision tower's weights (ADR 0075, GitHub
/// #414): `device`, in VRAM for the life of the load, or `host`, in the
/// mapped pinned host allocation the token embedding lives in (ADR 0062),
/// staged into the prefill scratch for each media encode. A load option,
/// fixed for the life of the load. `host` is the default (#416); the
/// `rtxpro6000` profile takes `device`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VisionResidency {
    /// The tower staged from pinned host memory for each encode.
    #[default]
    Host,
    /// The tower in VRAM, as every vision load before ADR 0075.
    Device,
}

impl VisionResidency {
    /// The spellings [`Self::parse`] accepts.
    pub const NAMES: [&'static str; 2] = ["host", "device"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
        }
    }

    /// `struct ignis_model_load_options::vision_residency`: 0 = device, so a
    /// zeroed options struct is today's load; 1 = host.
    pub const fn abi_code(self) -> u32 {
        match self {
            Self::Device => 0,
            Self::Host => 1,
        }
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "host" => Ok(Self::Host),
            "device" => Ok(Self::Device),
            other => Err(format!("unknown vision residency `{other}` (expected `host` or `device`)")),
        }
    }
}

impl std::fmt::Display for VisionResidency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every module a load binds beside the text scope: the drafter, the vision
/// tower, both or neither (GitHub #177). The default is the text scope alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelScope {
    pub draft: Option<DraftModule>,
    pub vision: bool,
    /// ADR 0075 (GitHub #414): where the vision tower's tensors live when
    /// `vision` binds them -- on the device, or host-mapped beside the token
    /// embedding. Read only with `vision`.
    pub vision_residency: VisionResidency,
    /// GitHub #341: the nine BF16 exception tensors -- the fused attention
    /// projection of GQA layers 3-23, the attention output of GQA layers 3
    /// and 7, the GDN output of layer 4 -- held on the device in FP8,
    /// converted at load. Unset, they are held as stored: this struct's
    /// default is the artifact's bytes, for a caller binding the 27B
    /// directly, and the server's default (`ignis_core::ExceptionWeights`).
    pub fp8_exceptions: bool,
}

/// [`model_scope_27b`] for a full [`ModelScope`]: the text scope, then the
/// drafter's objects, then the vision tower's — so neither module moves a
/// text placement, and the vision tower does not move the drafter's.
pub fn model_scope_27b_with(scope: ModelScope) -> Vec<InventoryEntry> {
    with_modules(text_scope_27b(), scope)
}

/// `text`, then the drafter's objects, then the vision tower's where its
/// residency puts it, with the exception tensors held as `scope` says.
fn with_modules(text: Vec<InventoryEntry>, scope: ModelScope) -> Vec<InventoryEntry> {
    let mut entries = text;
    match scope.draft {
        None => {}
        Some(DraftModule::Dflash2) => entries.extend(dflash2_scope_27b()),
        Some(DraftModule::Dflash2ShortlistHead) => {
            entries.extend(dflash2_scope_27b());
            entries.extend(draft_head_scope_27b());
        }
    }
    if scope.vision {
        let residence = match scope.vision_residency {
            VisionResidency::Host => Residence::HostMapped,
            VisionResidency::Device => Residence::Device,
        };
        entries.extend(vision_scope_27b().into_iter().map(|entry| InventoryEntry { residence, ..entry }));
    }
    if scope.fp8_exceptions {
        for entry in &mut entries {
            // Only a BF16 tensor converts (GitHub #339): the trellis body's
            // `attention/output` of layers 3 and 7 and `gdn/output` of layer 4
            // share the exception tensors' names, and are already 3 bits.
            if entry.format != NumericFormat::Bf16 {
                continue;
            }
            if let Some(blocks) = fp8_blocks(entry.name) {
                entry.encoding = DeviceEncoding::Fp8Rows(blocks);
            }
        }
    }
    entries
}

/// The body a 27B artifact carries (spec 27b-16gb/05a, GitHub #339): the
/// served NVFP4 planes, or the 16 GB card class's trellis matrices (spec
/// 27b-16gb/01). A [`ModelScope`] is the modules a load binds; the body is
/// the artifact's, read off its directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Body27b {
    Nvfp4,
    Trellis,
}

/// `reader`'s body, by its output head's format: the trellis artifact's head
/// is trellis like its body, the served one's W8. The leaf decides by the
/// same tensor (`kernel/src/model.cu`), and a body that mixes the two fails
/// there naming the tensor.
pub fn body_27b(reader: &Reader) -> Body27b {
    match reader.find("text/output_head") {
        Some(Object::Tensor(head)) if head.format.trellis_k2().is_some() => Body27b::Trellis,
        _ => Body27b::Nvfp4,
    }
}

/// [`model_scope_27b_with`] over `reader`'s body ([`body_27b`]): the served
/// text scope, or the trellis one with each matrix's K from the directory
/// ([`text_scope_27b_trellis`]), then the modules `scope` names, which both
/// artifacts carry as the served bytes. The binder and the load's
/// descriptors both read it, so the two cannot name different entries.
pub fn model_scope_27b_of(reader: &Reader, scope: ModelScope) -> Result<Vec<InventoryEntry>> {
    let text = match body_27b(reader) {
        Body27b::Nvfp4 => text_scope_27b(),
        Body27b::Trellis => text_scope_27b_trellis(|name| match reader.find(name) {
            Some(Object::Tensor(t)) => Some(t.format),
            _ => None,
        })?,
    };
    Ok(with_modules(text, scope))
}

/// The FP8 row blocks of a BF16 exception tensor's name, or None.
fn fp8_blocks(name: &str) -> Option<&'static [u32]> {
    let rest = name.strip_prefix("text/layers/")?;
    let (layer, tail) = rest.split_once('/')?;
    let layer: usize = layer.parse().ok()?;
    match tail {
        "attention/query_key_gate_value" if QKGV_BF16_LAYERS.contains(&layer) => Some(QKGV_FP8_BLOCKS),
        "attention/output" if ATTENTION_OUT_BF16_LAYERS.contains(&layer) => Some(OUTPUT_FP8_BLOCKS),
        "gdn/output" if GDN_OUT_BF16_LAYERS.contains(&layer) => Some(OUTPUT_FP8_BLOCKS),
        _ => None,
    }
}

/// [`bind_model_scope_27b`] for a full [`ModelScope`]. A missing or
/// mis-shaped vision object is a load failure naming it. Returns the handles
/// in [`model_scope_27b_of`] order: the scope over `reader`'s own body
/// (GitHub #339), the served one's [`model_scope_27b_with`].
pub fn bind_model_scope_27b_with(
    reader: &Reader,
    scope: ModelScope,
) -> Result<(MaterializationPlan, Vec<ObjectHandle>)> {
    let entries = model_scope_27b_of(reader, scope)?;
    let mut binder = Binder::new(reader);
    let mut handles = Vec::with_capacity(entries.len());
    bind_entries(&mut binder, &entries, &mut handles)?;
    Ok((binder.plan(), handles))
}

/// What `plan`'s mapped host allocation holds of the vision tower (ADR 0075,
/// GitHub #414): its spans, with their 4 KiB padding, which the host plan
/// charges on a line of its own -- the rest of the allocation is the token
/// embedding's (ADR 0062). 0 when the tower is on the device or not bound.
pub fn vision_host_mapped_bytes(reader: &Reader, plan: &MaterializationPlan) -> u64 {
    plan.host_mapped_objects
        .iter()
        .filter(|placement| reader.objects()[placement.handle.index].name().starts_with("vision/"))
        .map(|placement| placement.span_bytes)
        .sum()
}

/// [`bind_model_scope_27b_with`] with every tensor on the device, the token
/// embedding included: the placement before ADR 0062, kept as the oracle a
/// host-mapped load's output is checked against. The handles are in the
/// same order.
pub fn bind_model_scope_27b_on_device(
    reader: &Reader,
    scope: ModelScope,
) -> Result<(MaterializationPlan, Vec<ObjectHandle>)> {
    let entries = all_on_device(&model_scope_27b_of(reader, scope)?);
    let mut binder = Binder::new(reader);
    let mut handles = Vec::with_capacity(entries.len());
    bind_entries(&mut binder, &entries, &mut handles)?;
    Ok((binder.plan(), handles))
}

/// `entries`, every one on the device.
fn all_on_device(entries: &[InventoryEntry]) -> Vec<InventoryEntry> {
    entries.iter().map(|e| InventoryEntry { residence: Residence::Device, ..*e }).collect()
}

/// Require and place each entry where it lives, in order.
fn bind_entries(
    binder: &mut Binder,
    entries: &[InventoryEntry],
    handles: &mut Vec<ObjectHandle>,
) -> Result<()> {
    for entry in entries {
        let handle = binder.require_tensor(entry.name, entry.format, entry.layout, entry.shape)?;
        match (entry.residence, entry.encoding) {
            (Residence::Device, DeviceEncoding::Stored) => binder.materialize_on_device(handle)?,
            (Residence::Device, DeviceEncoding::Fp8Rows(blocks)) => binder.materialize_on_device_as_fp8(handle, blocks)?,
            (Residence::HostMapped, _) => binder.place_in_host_mapped(handle)?,
        }
        handles.push(handle);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The trellis body (spec 27b-16gb/01, GitHub #338)
// ---------------------------------------------------------------------------

const SH_GDN_QKV: [u64; 2] = [10240, 5120]; // the GDN q + k + v (exllamav3's `in_proj_qkv`)
const SH_GDN_Z: [u64; 2] = [6144, 5120]; // the GDN output gate (`in_proj_z`)
const SH_Q_GATE: [u64; 2] = [12288, 5120]; // q and its gate, interleaved per head (`q_proj`)
const SH_KV: [u64; 2] = [1024, 5120]; // `k_proj`, `v_proj`
const SH_MLP_HALF: [u64; 2] = [17408, 5120]; // `gate_proj`, `up_proj`

/// The trellis objects standing for an NVFP4 text-scope plane (its suffix
/// under `text/layers/{i}/`): a fused plane's split parts under their
/// exllamav3 names (exllamav3 quantized them separately, each with its own
/// scales: ADR 0060 §4), a whole one under its own name.
fn trellis_parts(suffix: &str) -> Option<&'static [(&'static str, &'static [u64])]> {
    Some(match suffix {
        "gdn/query_key_value_z" => &[("gdn/in_proj_qkv", &SH_GDN_QKV), ("gdn/in_proj_z", &SH_GDN_Z)],
        "gdn/output" => &[("gdn/output", &SH_GDN_OUT)],
        "attention/query_key_gate_value" => {
            &[("attention/q_proj", &SH_Q_GATE), ("attention/k_proj", &SH_KV), ("attention/v_proj", &SH_KV)]
        }
        "attention/output" => &[("attention/output", &SH_ATTN_OUT)],
        "mlp/gate_up" => &[("mlp/gate_proj", &SH_MLP_HALF), ("mlp/up_proj", &SH_MLP_HALF)],
        "mlp/down" => &[("mlp/down", &SH_MLP_DOWN)],
        _ => return None,
    })
}

/// The text scope of the 27B's trellis artifact (the 16 GB card class, spec
/// 27b-16gb/01): [`text_scope_27b`] with every body linear and the output
/// head in trellis (layout `trellis-tile16-v1`), the NVFP4 planes'
/// `*_input_scale_divisor` scalars gone with their W4A4 activations, and
/// every other entry unchanged -- the token embedding in mapped host memory
/// included. 803 entries, 401 of them trellis.
///
/// K is per matrix (the public quant's), so a trellis entry's format is
/// `format_of(name)`'s: the artifact's directory, which is the authority
/// for the format (ADR 0002). A name it does not know, or a format that is
/// not a trellis one, is an error naming the object.
pub fn text_scope_27b_trellis(format_of: impl Fn(&str) -> Option<NumericFormat>) -> Result<Vec<InventoryEntry>> {
    let trellis = |name: String, shape: &'static [u64]| -> Result<InventoryEntry> {
        let format = format_of(&name)
            .ok_or_else(|| crate::fail(format!("the trellis artifact has no {name}")))?;
        if format.trellis_k2().is_none() {
            return Err(crate::fail(format!("{name} is {}, not a trellis format", format.name())));
        }
        Ok(mk(&name, format, StorageLayout::TrellisTile16V1, shape))
    };
    let mut entries = Vec::with_capacity(803);
    for entry in text_scope_27b() {
        if entry.name == "text/output_head" {
            entries.push(trellis(entry.name.to_string(), &SH_EMBED)?);
            continue;
        }
        let Some((prefix, suffix)) = entry
            .name
            .strip_prefix("text/layers/")
            .and_then(|rest| rest.split_once('/'))
            .map(|(layer, suffix)| (format!("text/layers/{layer}/"), suffix))
        else {
            entries.push(entry);
            continue;
        };
        if suffix.ends_with("input_scale_divisor") {
            continue;
        }
        match trellis_parts(suffix) {
            Some(parts) => {
                for (part, shape) in parts {
                    entries.push(trellis(format!("{prefix}{part}"), shape)?);
                }
            }
            None => entries.push(entry),
        }
    }
    Ok(entries)
}

/// Build a device entry, leaking the (templated) name into a `'static`
/// string (the generator runs once; the leak is bounded by the table size).
fn mk(name: &str, format: NumericFormat, layout: StorageLayout, shape: &'static [u64]) -> InventoryEntry {
    InventoryEntry {
        name: Box::leak(name.to_string().into_boxed_str()),
        format,
        layout,
        shape,
        residence: Residence::Device,
        encoding: DeviceEncoding::Stored,
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU-only — ADR 0006: no GPU, no artifact I/O)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_encoded_size;
    use std::collections::BTreeMap;

    /// The 27B text-scope inventory is complete and rule-mapped (spec 04's
    /// acceptance: "the 27B model's weights are complete, no
    /// `placeholder` left for the text scope" — every in-scope tensor's
    /// `(format, layout, shape)` combination is geometrically valid
    /// (normalizable) and maps to the A1 rule's kernel-expected form).
    #[test]
    fn text_scope_27b_is_complete() {
        let entries = text_scope_27b();
        // The inventory's exact size + format distribution (the artifact's
        // directory projection, spec 04).
        assert_eq!(entries.len(), 906, "the 906 in-scope text tensors");
        let mut counts = BTreeMap::new();
        for e in &entries {
            *counts.entry(e.format).or_insert(0) += 1;
        }
        assert_eq!(counts.get(&NumericFormat::W8G32F16S), Some(&2), "the two W8 endpoints");
        assert_eq!(counts.get(&NumericFormat::Bf16), Some(&314));
        assert_eq!(counts.get(&NumericFormat::Fp32), Some(&343));
        assert_eq!(counts.get(&NumericFormat::Nvfp4), Some(&247));

        // Every entry is geometrically valid (the layout's encoded size
        // exists — the normalize step's geometry check accepts it).
        for e in &entries {
            let size = tensor_encoded_size(e.layout, e.format, e.shape)
                .unwrap_or_else(|err| panic!("entry {} does not pass the geometry check: {err}", e.name));
            assert!(size > 0);
        }

        // The two W8 tensors are exactly the text-scope endpoints (the
        // embedding table + the lm_head logits GEMM, spec 04).
        let w8: Vec<&str> = entries
            .iter()
            .filter(|e| e.format == NumericFormat::W8G32F16S)
            .map(|e| e.name)
            .collect();
        assert_eq!(w8, vec!["text/token_embedding", "text/output_head"]);

        // Every NVFP4 entry is a blockscale GEMM weight (the rule's
        // preserve path — the kernel consumes the codes + scales directly).
        for e in entries.iter().filter(|e| e.format == NumericFormat::Nvfp4) {
            assert_eq!(
                e.layout,
                StorageLayout::BlockScaleK16M128x4V1,
                "{} must be the NVFP4 (GEMM) layout",
                e.name
            );
        }
        // The per-format layout mapping is exact (each format has its
        // layout in the rule: W8 -> row-split, BF16 / FP32 -> contiguous,
        // NVFP4 -> blockscale).
        for e in &entries {
            let want = match e.format {
                NumericFormat::W8G32F16S => StorageLayout::RowSplitK128V1,
                NumericFormat::Bf16 | NumericFormat::Fp32 => StorageLayout::ContiguousLeV1,
                NumericFormat::Nvfp4 => StorageLayout::BlockScaleK16M128x4V1,
                other => panic!("unexpected format {other:?} in the text scope"),
            };
            assert_eq!(e.layout, want, "{} carries an unexpected layout", e.name);
        }
    }

    /// ADR 0062: the token embedding is the one tensor of a 27B scope that
    /// lives in mapped host memory under the `device` vision residency;
    /// everything else, the output head included, is on the device.
    #[test]
    fn the_token_embedding_is_the_one_host_mapped_tensor_of_every_scope() {
        for scope in [
            ModelScope::default(),
            ModelScope {
                draft: Some(DraftModule::Dflash2ShortlistHead),
                vision: true,
                vision_residency: VisionResidency::Device,
                fp8_exceptions: false,
            },
        ] {
            let host: Vec<&str> = model_scope_27b_with(scope)
                .iter()
                .filter(|e| e.residence == Residence::HostMapped)
                .map(|e| e.name)
                .collect();
            assert_eq!(host, ["text/token_embedding"], "{scope:?}");
        }
    }

    /// ADR 0075 (GitHub #414): under the `host` vision residency the tower's
    /// 333 tensors take the token embedding's mapped allocation, so the
    /// binder places them in host memory and the encode stages them; the
    /// token embedding is host-mapped whatever the residency, and nothing
    /// else moves -- not the drafter, not the shortlist head, not a text
    /// tensor.
    #[test]
    fn the_vision_scope_is_host_mapped_only_under_the_host_residency() {
        for draft in [None, Some(DraftModule::Dflash2ShortlistHead)] {
            let at = |vision_residency| ModelScope { draft, vision: true, vision_residency, fp8_exceptions: false };
            let host: Vec<&str> = model_scope_27b_with(at(VisionResidency::Host))
                .iter()
                .filter(|e| e.residence == Residence::HostMapped)
                .map(|e| e.name)
                .collect();
            assert_eq!(host.len(), 1 + 333, "{draft:?}");
            assert_eq!(host[0], "text/token_embedding");
            assert!(host[1..].iter().all(|name| name.starts_with("vision/")), "{host:?}");
            // The same entries in the same order: only where the tower lives moves.
            let names = |scope| model_scope_27b_with(scope).iter().map(|e| e.name).collect::<Vec<_>>();
            assert_eq!(names(at(VisionResidency::Host)), names(at(VisionResidency::Device)));
        }
        // A residency without the tower places nothing new.
        let text = ModelScope { vision_residency: VisionResidency::Host, ..ModelScope::default() };
        let host: Vec<&str> =
            model_scope_27b_with(text).iter().filter(|e| e.residence == Residence::HostMapped).map(|e| e.name).collect();
        assert_eq!(host, ["text/token_embedding"]);
    }

    #[test]
    fn the_vision_residency_parses_its_two_values_and_defaults_to_the_host() {
        assert_eq!(VisionResidency::default(), VisionResidency::Host);
        assert_eq!(VisionResidency::NAMES, ["host", "device"]);
        for residency in [VisionResidency::Host, VisionResidency::Device] {
            assert_eq!(VisionResidency::parse(residency.as_str()), Ok(residency));
        }
        assert_eq!(VisionResidency::parse(" Host "), Ok(VisionResidency::Host));
        let err = VisionResidency::parse("cpu").unwrap_err();
        assert!(err.contains("cpu") && err.contains("host") && err.contains("device"), "{err}");
        // `ignis_model_load_options::vision_residency` (kernel/include/ignis_model.h).
        assert_eq!((VisionResidency::Device.abi_code(), VisionResidency::Host.abi_code()), (0, 1));
    }

    /// `bind_entries` places each entry where the inventory says it lives,
    /// and the all-on-device binding (the placement before ADR 0062, the
    /// oracle a host-mapped load is checked against) moves it to the device.
    #[test]
    fn a_host_mapped_entry_binds_into_the_mapped_allocation_unless_all_go_on_the_device() {
        let artifact = crate::fixture::write_fixture(
            &crate::fixture::all_layout_objects(),
            &crate::fixture::all_layout_payload(),
            "inventory-host-mapped",
        )
        .expect("write fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");
        let entries = [
            mk("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8]),
            InventoryEntry {
                residence: Residence::HostMapped,
                ..mk("w/q4", NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1, &[128, 128])
            },
        ];
        let mut binder = Binder::new(&reader);
        let mut handles = Vec::new();
        bind_entries(&mut binder, &entries, &mut handles).expect("bind");
        let plan = binder.plan();
        assert_eq!(plan.device_objects.len(), 1);
        assert_eq!(plan.device_objects[0].handle, handles[0]);
        assert_eq!(plan.host_mapped_objects.len(), 1);
        assert_eq!(plan.host_mapped_objects[0].handle, handles[1]);

        let mut binder = Binder::new(&reader);
        let mut handles = Vec::new();
        bind_entries(&mut binder, &all_on_device(&entries), &mut handles).expect("bind");
        let plan = binder.plan();
        assert_eq!(plan.device_objects.len(), 2);
        assert!(plan.host_mapped_objects.is_empty());
        assert_eq!(plan.host_mapped_capacity_bytes, 0);
    }

    /// GitHub #341: under `fp8_exceptions` exactly the nine BF16 exception
    /// tensors convert, each in blocks that cover its rows; without it none.
    #[test]
    fn fp8_exceptions_convert_exactly_the_nine_bf16_exception_tensors() {
        for draft in [None, Some(DraftModule::Dflash2)] {
            let plain = model_scope_27b_with(ModelScope { draft, vision: true, vision_residency: VisionResidency::Device, fp8_exceptions: false });
            assert!(plain.iter().all(|e| e.encoding == DeviceEncoding::Stored));
            let fp8 = model_scope_27b_with(ModelScope { draft, vision: true, vision_residency: VisionResidency::Device, fp8_exceptions: true });
            assert_eq!(fp8.len(), plain.len(), "the same tensors");
            let converted: Vec<&InventoryEntry> = fp8.iter().filter(|e| e.encoding != DeviceEncoding::Stored).collect();
            let names: Vec<&str> = converted.iter().map(|e| e.name).collect();
            assert_eq!(names.len(), 9, "{names:?}");
            for layer in [3, 7, 11, 15, 19, 23] {
                assert!(names.contains(&format!("text/layers/{layer}/attention/query_key_gate_value").as_str()));
            }
            for name in ["text/layers/3/attention/output", "text/layers/7/attention/output", "text/layers/4/gdn/output"] {
                assert!(names.contains(&name), "{name}");
            }
            for e in converted {
                let DeviceEncoding::Fp8Rows(blocks) = e.encoding else { unreachable!() };
                assert_eq!(e.format, NumericFormat::Bf16, "{}", e.name);
                assert_eq!(blocks.iter().map(|&b| u64::from(b)).sum::<u64>(), e.shape[0], "{}", e.name);
            }
        }
    }

    /// The generated table has no duplicate names (the reader's
    /// directory invariant: exact names, no duplicates).
    #[test]
    fn text_scope_27b_has_unique_names() {
        let entries = text_scope_27b();
        let mut names: BTreeMap<&str, u32> = BTreeMap::new();
        for e in &entries {
            *names.entry(e.name).or_insert(0) += 1;
        }
        let dups: Vec<_> = names.iter().filter(|(_, c)| **c > 1).collect();
        assert!(dups.is_empty(), "duplicate names: {dups:?}");
    }

    // -- bind_text_scope_27b: the model-load binder's loud failures (P1-17,
    // GitHub #53). `text_scope_27b()` is bound in directory order, so a
    // fixture missing (or mis-shaping) only the first entry
    // (`text/token_embedding`) fails immediately — no need to fabricate all
    // 906 production-scale tensors just to exercise the failure path.

    /// A minimal one-resource fixture: satisfies the reader's "objects must
    /// be nonempty" rule without carrying any text-scope tensor.
    fn empty_of_text_tensors(tag: &str) -> (crate::fixture::TempArtifact, Reader) {
        let objects = vec![crate::fixture::FixtureObject::Resource {
            name: "frontend/tokenizer.json",
            encoding: "raw-bytes-v1",
            offset: 0,
            bytes: 4,
        }];
        let artifact =
            crate::fixture::write_fixture(&objects, &[0u8; 4], tag).expect("write fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");
        (artifact, reader)
    }

    #[test]
    fn bind_text_scope_27b_rejects_a_missing_object() {
        let (_artifact, reader) = empty_of_text_tensors("missing-object");
        let err = bind_text_scope_27b(&reader)
            .expect_err("no text-scope tensor is present")
            .to_string();
        assert!(err.contains("is missing"), "{err}");
        assert!(err.contains("text/token_embedding"), "{err}");
    }

    #[test]
    fn bind_text_scope_27b_rejects_a_mis_shaped_object() {
        // The first entry (`text/token_embedding`) is present, but with a
        // tiny stand-in shape instead of the model's `[248320, 5120]`.
        let objects = vec![crate::fixture::FixtureObject::Tensor {
            name: "text/token_embedding",
            shape: vec![2, 32],
            format: "W8G32_F16S",
            layout: "row-split-k128-v1",
            offset: 0,
            bytes: crate::row_split_geometry(NumericFormat::W8G32F16S, &[2, 32])
                .unwrap()
                .encoded_bytes,
        }];
        let payload = vec![
            0u8;
            crate::row_split_geometry(NumericFormat::W8G32F16S, &[2, 32])
                .unwrap()
                .encoded_bytes as usize
        ];
        let artifact = crate::fixture::write_fixture(&objects, &payload, "mis-shaped")
            .expect("write fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");

        let err = bind_text_scope_27b(&reader)
            .expect_err("the stored shape does not match the model's contract")
            .to_string();
        assert!(err.contains("does not match target contract"), "{err}");
        assert!(err.contains("text/token_embedding"), "{err}");
    }

    // -- the DFlash2 drafter module (P5-02, GitHub #150) ---------------------

    #[test]
    fn dflash2_scope_is_the_66_module_objects() {
        let entries = dflash2_scope_27b();
        assert_eq!(entries.len(), 66);
        let nvfp4 = entries.iter().filter(|e| e.format == NumericFormat::Nvfp4).count();
        let bf16 = entries.iter().filter(|e| e.format == NumericFormat::Bf16).count();
        assert_eq!((nvfp4, bf16), (34, 32), "the reference's 34 NVFP4 + 32 BF16");
        let mut names = std::collections::BTreeSet::new();
        for e in &entries {
            assert!(e.name.starts_with("dflash2/"), "{}", e.name);
            assert!(names.insert(e.name), "duplicate {}", e.name);
            let size = tensor_encoded_size(e.layout, e.format, e.shape)
                .unwrap_or_else(|err| panic!("{} fails the geometry check: {err}", e.name));
            assert!(size > 0);
            // Weight-only: the drafter runs A16, no divisor object pairs up.
            assert!(!e.name.ends_with("input_scale_divisor"), "{}", e.name);
        }
    }

    #[test]
    fn without_a_draft_module_the_scope_is_todays_text_scope() {
        assert_eq!(model_scope_27b(None), text_scope_27b());
    }

    #[test]
    fn with_dflash2_the_scope_appends_the_module_and_leaves_the_text_scope_alone() {
        let text = text_scope_27b();
        let scope = model_scope_27b(Some(DraftModule::Dflash2));
        assert_eq!(scope.len(), text.len() + 66);
        assert_eq!(&scope[..text.len()], text.as_slice(), "text entries unchanged and first");
        assert_eq!(&scope[text.len()..], dflash2_scope_27b().as_slice());
    }

    #[test]
    fn the_shortlist_head_module_is_the_drafter_then_the_out_of_scope_head_pair() {
        let with_drafter = model_scope_27b(Some(DraftModule::Dflash2));
        let scope = model_scope_27b(Some(DraftModule::Dflash2ShortlistHead));
        assert_eq!(scope.len(), with_drafter.len() + 2);
        assert_eq!(&scope[..with_drafter.len()], with_drafter.as_slice(), "no placement moves");
        let head = &scope[with_drafter.len()..];
        let names: Vec<&str> = head.iter().map(|e| e.name).collect();
        assert_eq!(names, OUT_OF_SCOPE_TEXT_NAMES, "exactly the pair the text scope leaves out");
        for e in head {
            tensor_encoded_size(e.layout, e.format, e.shape)
                .unwrap_or_else(|err| panic!("{} fails the geometry check: {err}", e.name));
        }
        assert_eq!(head[0].shape, &[131072, 5120]);
        assert_eq!(head[1].shape, &[131072]);
    }

    #[test]
    fn the_vision_scope_is_the_artifacts_333_objects_in_their_stored_formats() {
        let vision = vision_scope_27b();
        assert_eq!(vision.len(), 333);
        let mut by_format = BTreeMap::new();
        for entry in &vision {
            assert!(entry.name.starts_with("vision/"), "{}", entry.name);
            *by_format.entry(entry.format.name()).or_insert(0) += 1;
            // Every entry's (layout, format, shape) is a valid encoding.
            tensor_encoded_size(entry.layout, entry.format, entry.shape)
                .unwrap_or_else(|e| panic!("{}: {e}", entry.name));
        }
        assert_eq!(by_format["Q6G64_F16S"], 1);
        assert_eq!(by_format["Q4G64_F16S"], 54);
        assert_eq!(by_format["Q5G64_F16S"], 54);
        assert_eq!(by_format["W8G32_F16S"], 2);
        assert_eq!(by_format["BF16"], 222);
        let names: std::collections::BTreeSet<_> = vision.iter().map(|e| e.name).collect();
        assert_eq!(names.len(), 333, "no duplicate name");
    }

    #[test]
    fn the_model_scope_appends_vision_after_the_text_and_the_drafter() {
        assert_eq!(model_scope_27b_with(ModelScope::default()), text_scope_27b());
        let text = text_scope_27b();
        let vision = vision_scope_27b();

        let only_vision = model_scope_27b_with(ModelScope { draft: None, vision: true, vision_residency: VisionResidency::Device, fp8_exceptions: false });
        assert_eq!(&only_vision[..text.len()], text.as_slice());
        assert_eq!(&only_vision[text.len()..], vision.as_slice());

        let both = model_scope_27b_with(ModelScope { draft: Some(DraftModule::Dflash2), vision: true, vision_residency: VisionResidency::Device, fp8_exceptions: false });
        let with_drafter = model_scope_27b(Some(DraftModule::Dflash2));
        assert_eq!(&both[..with_drafter.len()], with_drafter.as_slice());
        assert_eq!(&both[with_drafter.len()..], vision.as_slice());
    }

    /// `bind_entries` over the drafter table, on a fixture carrying the
    /// module's first object (or not): the drafter's refusals are the text
    /// scope's, naming the object.
    fn bind_dflash2_error(objects: Vec<crate::fixture::FixtureObject>, payload: &[u8], tag: &str) -> String {
        let artifact = crate::fixture::write_fixture(&objects, payload, tag).expect("write fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");
        let mut binder = Binder::new(&reader);
        let mut handles = Vec::new();
        bind_entries(&mut binder, &dflash2_scope_27b(), &mut handles)
            .expect_err("the drafter module is not in this fixture as the model needs it")
            .to_string()
    }

    #[test]
    fn a_missing_dflash2_object_is_a_load_failure_naming_it() {
        let objects = vec![crate::fixture::FixtureObject::Resource {
            name: "frontend/tokenizer.json",
            encoding: "raw-bytes-v1",
            offset: 0,
            bytes: 4,
        }];
        let err = bind_dflash2_error(objects, &[0u8; 4], "dflash2-missing");
        assert!(err.contains("is missing"), "{err}");
        assert!(err.contains("dflash2/feature_projection"), "{err}");
    }

    #[test]
    fn a_mis_shaped_dflash2_object_is_a_load_failure_naming_it() {
        // Present under the right name, stored BF16 [4, 8] instead of the
        // model's NVFP4 [5120, 25600].
        let objects = vec![crate::fixture::FixtureObject::Tensor {
            name: "dflash2/feature_projection",
            shape: vec![4, 8],
            format: "BF16",
            layout: "contiguous-le-v1",
            offset: 0,
            bytes: 64,
        }];
        let err = bind_dflash2_error(objects, &[0u8; 64], "dflash2-mis-shaped");
        assert!(err.contains("does not match target contract"), "{err}");
        assert!(err.contains("dflash2/feature_projection"), "{err}");
    }

    /// Spec 27b-16gb/01: the trellis artifact's text scope is the NVFP4
    /// one with the body linears and the head in trellis (fused planes
    /// split, divisors gone) and everything else unchanged.
    #[test]
    fn the_trellis_text_scope_replaces_the_body_and_the_head_and_keeps_the_rest() {
        let k3 = |_: &str| Some(NumericFormat::TrellisMul1K3);
        let entries = text_scope_27b_trellis(k3).unwrap();
        assert_eq!(entries.len(), 803);
        let trellis: Vec<&InventoryEntry> =
            entries.iter().filter(|e| e.layout == StorageLayout::TrellisTile16V1).collect();
        assert_eq!(trellis.len(), 48 * 6 + 16 * 7 + 1, "the public quant's 400 body matrices and its head");
        for e in &trellis {
            tensor_encoded_size(e.layout, e.format, e.shape).unwrap_or_else(|err| panic!("{}: {err}", e.name));
            assert_eq!(e.residence, Residence::Device, "{}", e.name);
        }
        assert!(entries.iter().all(|e| !e.name.ends_with("input_scale_divisor")));

        let nvfp4: BTreeMap<&str, InventoryEntry> = text_scope_27b().into_iter().map(|e| (e.name, e)).collect();
        let kept: Vec<&InventoryEntry> =
            entries.iter().filter(|e| e.layout != StorageLayout::TrellisTile16V1).collect();
        assert_eq!(kept.len(), 402);
        for e in kept {
            assert_eq!(nvfp4.get(e.name), Some(e), "{} is the NVFP4 scope's entry", e.name);
        }
        let names: std::collections::BTreeSet<&str> = entries.iter().map(|e| e.name).collect();
        assert_eq!(names.len(), entries.len(), "unique names");

        let layer = |i: usize| -> Vec<(&str, &[u64])> {
            let prefix = format!("text/layers/{i}/");
            trellis
                .iter()
                .filter_map(|e| e.name.strip_prefix(prefix.as_str()).map(|s| (s, e.shape)))
                .collect()
        };
        assert_eq!(
            layer(0),
            vec![
                ("gdn/in_proj_qkv", &[10240u64, 5120][..]),
                ("gdn/in_proj_z", &[6144, 5120][..]),
                ("gdn/output", &[5120, 6144][..]),
                ("mlp/gate_proj", &[17408, 5120][..]),
                ("mlp/up_proj", &[17408, 5120][..]),
                ("mlp/down", &[5120, 17408][..]),
            ]
        );
        assert_eq!(
            layer(3),
            vec![
                ("attention/q_proj", &[12288u64, 5120][..]),
                ("attention/k_proj", &[1024, 5120][..]),
                ("attention/v_proj", &[1024, 5120][..]),
                ("attention/output", &[5120, 6144][..]),
                ("mlp/gate_proj", &[17408, 5120][..]),
                ("mlp/up_proj", &[17408, 5120][..]),
                ("mlp/down", &[5120, 17408][..]),
            ]
        );
        let head = entries.iter().find(|e| e.name == "text/output_head").unwrap();
        assert_eq!((head.layout, head.shape), (StorageLayout::TrellisTile16V1, &SH_EMBED[..]));
        let embedding = entries.iter().find(|e| e.name == "text/token_embedding").unwrap();
        assert_eq!(embedding.residence, Residence::HostMapped);
    }

    #[test]
    fn a_trellis_entry_takes_its_k_from_the_directory() {
        let entries = text_scope_27b_trellis(|name: &str| {
            Some(if name.ends_with("mlp/down") { NumericFormat::TrellisMul1K5 } else { NumericFormat::TrellisMul1K2 })
        })
        .unwrap();
        let format = |name: &str| entries.iter().find(|e| e.name == name).unwrap().format;
        assert_eq!(format("text/layers/7/mlp/down"), NumericFormat::TrellisMul1K5);
        assert_eq!(format("text/layers/7/mlp/up_proj"), NumericFormat::TrellisMul1K2);
        assert_eq!(format("text/layers/7/input_norm"), NumericFormat::Bf16);
    }

    #[test]
    fn a_missing_or_non_trellis_body_object_is_an_error_naming_it() {
        let err = text_scope_27b_trellis(|name: &str| {
            (name != "text/layers/5/gdn/in_proj_z").then_some(NumericFormat::TrellisMul1K4)
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("text/layers/5/gdn/in_proj_z"), "{err}");
        let err = text_scope_27b_trellis(|name: &str| {
            Some(if name == "text/output_head" { NumericFormat::W8G32F16S } else { NumericFormat::TrellisMul1K4 })
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("text/output_head is W8G32_F16S, not a trellis format"), "{err}");
    }

    /// GitHub #339: `fp8_exceptions` names the exception tensors by name, and
    /// on the trellis body three of those names are trellis matrices
    /// (`attention/output` of layers 3 and 7, `gdn/output` of layer 4). Only
    /// a BF16 tensor converts, so the option is a no-op there -- the body has
    /// no BF16 exception tensor -- and the served body's nine still convert.
    #[test]
    fn fp8_exceptions_never_reencode_a_trellis_matrix() {
        let k3 = |_: &str| Some(NumericFormat::TrellisMul1K3);
        for draft in [None, Some(DraftModule::Dflash2ShortlistHead)] {
            let scope = ModelScope { draft, vision: true, vision_residency: VisionResidency::Device, fp8_exceptions: true };
            let trellis = with_modules(text_scope_27b_trellis(k3).unwrap(), scope);
            assert!(trellis.iter().all(|e| e.encoding == DeviceEncoding::Stored), "{draft:?}");
            assert_eq!(trellis.len(), with_modules(text_scope_27b_trellis(k3).unwrap(), ModelScope { fp8_exceptions: false, ..scope }).len());
            for name in ["text/layers/3/attention/output", "text/layers/7/attention/output", "text/layers/4/gdn/output"] {
                let entry = trellis.iter().find(|e| e.name == name).unwrap();
                assert_eq!((entry.layout, entry.encoding), (StorageLayout::TrellisTile16V1, DeviceEncoding::Stored), "{name}");
            }
            let served = model_scope_27b_with(scope);
            assert_eq!(served.iter().filter(|e| e.encoding != DeviceEncoding::Stored).count(), 9, "{draft:?}");
        }
    }

    /// A directory of one tensor, `text/output_head` in `format`.
    fn head_only_reader(format: NumericFormat, layout: StorageLayout, tag: &str) -> (crate::fixture::TempArtifact, Reader) {
        let shape = vec![256u64, 128];
        let bytes = crate::tensor_encoded_size(layout, format, &shape).unwrap();
        let objects = vec![crate::fixture::FixtureObject::Tensor {
            name: "text/output_head",
            shape,
            format: format.name(),
            layout: layout.name(),
            offset: 0,
            bytes,
        }];
        let artifact = crate::fixture::write_fixture(&objects, &vec![0u8; bytes as usize], tag).unwrap();
        let reader = Reader::open(&artifact.path).unwrap();
        (artifact, reader)
    }

    /// GitHub #339: the body is the artifact's, read off its head's format;
    /// a trellis head makes the scope the trellis one, whose first missing
    /// matrix is then named.
    #[test]
    fn the_body_is_read_off_the_output_heads_format() {
        let (_served, served) = head_only_reader(NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1, "body-served");
        assert_eq!(body_27b(&served), Body27b::Nvfp4);
        assert_eq!(model_scope_27b_of(&served, ModelScope::default()).unwrap(), model_scope_27b_with(ModelScope::default()));
        let (_trellis, trellis) = head_only_reader(NumericFormat::TrellisMul1K4, StorageLayout::TrellisTile16V1, "body-trellis");
        assert_eq!(body_27b(&trellis), Body27b::Trellis);
        let err = model_scope_27b_of(&trellis, ModelScope::default()).unwrap_err().to_string();
        assert!(err.contains("the trellis artifact has no text/layers/0/gdn/in_proj_qkv"), "{err}");
        let err = bind_model_scope_27b_with(&trellis, ModelScope::default()).unwrap_err().to_string();
        assert!(err.contains("text/layers/0/gdn/in_proj_qkv"), "{err}");
    }
}