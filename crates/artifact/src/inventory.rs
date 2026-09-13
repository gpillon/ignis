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

use crate::binder::{Binder, MaterializationPlan, ObjectHandle};
use crate::{NumericFormat, Reader, Result, StorageLayout};

// ---------------------------------------------------------------------------
// The entry
// ---------------------------------------------------------------------------

/// One text-scope tensor entry (the container's directory projection:
/// `name` + `format` + `layout` + `shape`).
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
    entries.push(mk("text/token_embedding", NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1, &SH_EMBED));
    for i in 0..64usize {
        // Layer `i` is GQA iff `(i + 1) % 4 == 0` (the model constant).
        entries.extend(if (i + 1) % 4 == 0 { gqa_layer(i) } else { gdn_layer(i) });
    }
    entries.push(mk("text/final_norm", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &SH_HIDDEN));
    entries.push(mk("text/output_head", NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1, &SH_EMBED));
    entries
}

/// Bind every text-scope tensor ([`text_scope_27b`]) against `reader`,
/// materializing each on the device (P1-17, GitHub #53).
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

/// Every object a load binds: the text scope, then the drafter module when
/// one is selected. The text entries are [`text_scope_27b`] unchanged and
/// first, so a drafter-bearing plan places every text tensor at the offset a
/// text-only plan does.
pub fn model_scope_27b(draft: Option<DraftModule>) -> Vec<InventoryEntry> {
    let mut entries = text_scope_27b();
    match draft {
        None => {}
        Some(DraftModule::Dflash2) => entries.extend(dflash2_scope_27b()),
    }
    entries
}

/// [`bind_text_scope_27b`], plus the drafter module's objects when `draft`
/// names one (P5-02, GitHub #150). A missing or mis-shaped drafter object is
/// a load failure naming it, the same contract as the text scope's. Returns
/// the handles in [`model_scope_27b`] order.
pub fn bind_model_scope_27b(
    reader: &Reader,
    draft: Option<DraftModule>,
) -> Result<(MaterializationPlan, Vec<ObjectHandle>)> {
    let entries = model_scope_27b(draft);
    let mut binder = Binder::new(reader);
    let mut handles = Vec::with_capacity(entries.len());
    bind_entries(&mut binder, &entries, &mut handles)?;
    Ok((binder.plan(), handles))
}

/// Require and place each entry on the device, in order.
fn bind_entries(
    binder: &mut Binder,
    entries: &[InventoryEntry],
    handles: &mut Vec<ObjectHandle>,
) -> Result<()> {
    for entry in entries {
        let handle = binder.require_tensor(entry.name, entry.format, entry.layout, entry.shape)?;
        binder.materialize_on_device(handle)?;
        handles.push(handle);
    }
    Ok(())
}

/// Build an entry, leaking the (templated) name into a `'static` string
/// (the generator runs once; the leak is bounded by the table size).
fn mk(name: &str, format: NumericFormat, layout: StorageLayout, shape: &'static [u64]) -> InventoryEntry {
    InventoryEntry {
        name: Box::leak(name.to_string().into_boxed_str()),
        format,
        layout,
        shape,
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
}