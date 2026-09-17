//! The model-load step ABI call (ADR 0009, GitHub #53, P1-17).
//!
//! Builds the flat bound-tensor + topology descriptors the kernel leaf's
//! `ignis_model_load` (`kernel/include/ignis_model.h`) consumes from a
//! [`MaterializedArtifact`] the [`ignis_artifact::CudaDevice`] path placed on
//! the device, and wraps the loaded handle so it releases on [`Drop`].
//!
//! Scope (P1-17): the text-scope tensors ([`ignis_artifact::bind_text_scope_27b`]) become
//! bound-tensor descriptors; the [`ModelConfig::qwen38_27b`] topology
//! crosses once. The `*_input_scale_divisor` objects are bound and
//! shape-checked against the artifact (ADR 0002 — a missing or mis-shaped
//! one is still a load failure) and never get their own bound-tensor
//! descriptor, but each one's value is read and carried on its paired
//! NVFP4 weight's descriptor (GitHub #58): the reference's NVFP4 `Weight`
//! validation requires a finite, positive divisor regardless of compute
//! policy, even though the W4A4 path that multiplies by it is still G2
//! (`.scratch/runtime/specs/01-device-resident-forward.md`).

#![cfg(feature = "cuda")]

use std::ffi::{CStr, CString};
use std::os::raw::c_void;

use ignis_artifact::{
    model_scope_27b_with, DraftModule, InventoryEntry, MaterializationPlan, MaterializedArtifact,
    ModelScope, NumericFormat, ObjectHandle, Reader, StorageLayout,
};

use crate::compute::{LayerKind, ModelConfig};
use crate::kv_format::KvFormat;
use crate::speculation::{SpeculativeBackend, Speculation};
use crate::vision::Vision;

pub(crate) mod ffi {
    use std::os::raw::{c_char, c_void};

    /// Opaque loaded-model handle (`kernel/include/ignis_model.h`).
    ///
    /// FFI-safe: `#[repr(C)]` + non-zero-sized so `*mut IgnisModel` is a
    /// valid C pointer across the boundary (mirrors
    /// `ignis_artifact::ffi::IgnisDevice`). `pub(crate)`: the step ABI
    /// (`crate::step`, GitHub #54) passes the same handle to
    /// `ignis_prefill` / `ignis_decode`.
    #[repr(C)]
    pub struct IgnisModel([u8; 1]);

    /// 1:1 with `struct ignis_bound_tensor`.
    #[repr(C)]
    pub struct IgnisBoundTensor {
        pub name: *const c_char,
        pub qtype: i32,
        pub layout: i32,
        pub qdata: *const c_void,
        pub qhigh: *const c_void,
        pub scales: *const c_void,
        pub bytes: u64,
        pub shape: [i32; 4],
        pub padded_shape: [i32; 4],
        pub ndim: u32,
        pub weight_scale_divisor: f32,
        pub input_scale_divisor: f32,
    }

    /// 1:1 with `struct ignis_topology`.
    #[repr(C)]
    pub struct IgnisTopology {
        pub num_layers: u32,
        pub layer_kinds: *const i32,
        pub hidden: u64,
        pub vocab: u64,
        pub num_q_heads: u64,
        pub num_kv_heads: u64,
        pub head_dim: u64,
        pub rotary_dim: u64,
        pub rope_theta: f64,
        pub gdn_state_rows: u64,
        pub gdn_state_cols: u64,
        pub gdn_num_layers: u64,
        pub gdn_q_width: u64,
        pub gdn_z_width: u64,
        pub gdn_ab_width: u64,
        pub ffn_intermediate: u64,
        pub rms_norm_eps: f32,
    }

    /// 1:1 with `struct ignis_model_load_options` (ADR 0016: `size` first).
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IgnisModelLoadOptions {
        pub size: u32,
        pub speculative_backend: i32,
        pub draft_tokens: u32,
        /// GitHub #177: the vision envelope, 0 = no vision.
        pub vision_max_tokens: u32,
    }

    /// 1:1 with `struct ignis_model_reservations` (GitHub #210): every
    /// device reservation a load makes beside the weights, one field per
    /// VRAM plan line.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct IgnisModelReservations {
        pub workspace_bytes: u64,
        pub media_embedding_bytes: u64,
        pub sampling_bytes: u64,
        pub decode_graph_bytes: u64,
        pub verify_round_bytes: u64,
        pub drafter_round_bytes: u64,
    }

    /// 1:1 with `struct ignis_model_stats`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct IgnisModelStats {
        pub vram_bytes: u64,
        pub bound_tensor_count: u64,
        /// GitHub #177, #212: what vision adds beside a text load -- the
        /// output transient, and whatever the encoder's workspace grows the
        /// shared scratch past the prefill scratch.
        pub vision_reserved_bytes: u64,
        /// GitHub #210: what the load holds beside the weights, read off its
        /// own buffers.
        pub reserved: IgnisModelReservations,
    }

    unsafe extern "C" {
        pub fn ignis_model_load(
            tensors: *const IgnisBoundTensor,
            count: u64,
            topology: *const IgnisTopology,
            prefill_chunk_tokens: u32,
            max_context_tokens: u32,
            kv_format: i32,
            options: *const IgnisModelLoadOptions,
            out_model: *mut *mut IgnisModel,
        ) -> i32;

        pub fn ignis_model_plan_reservations(
            tensors: *const IgnisBoundTensor,
            count: u64,
            topology: *const IgnisTopology,
            prefill_chunk_tokens: u32,
            max_context_tokens: u32,
            kv_format: i32,
            options: *const IgnisModelLoadOptions,
            out: *mut IgnisModelReservations,
        ) -> i32;

        pub fn ignis_model_stats(model: *const IgnisModel, out_stats: *mut IgnisModelStats) -> i32;

        pub fn ignis_model_free(model: *mut IgnisModel);

        pub fn ignis_model_last_error() -> *const c_char;
    }
}

pub use ffi::{IgnisModelReservations, IgnisModelStats};

/// A loaded model handle (releases via `ignis_model_free` on [`Drop`]).
pub struct Model {
    handle: *mut ffi::IgnisModel,
}

impl Model {
    /// VRAM / bound-tensor statistics the leaf reports for this load.
    pub fn stats(&self) -> IgnisModelStats {
        let mut stats = IgnisModelStats::default();
        let rc = unsafe { ffi::ignis_model_stats(self.handle, &mut stats) };
        assert_eq!(rc, 0, "ignis_model_stats: null handle (unreachable — Model always holds one)");
        stats
    }

    /// The raw handle the step ABI (`crate::step`, GitHub #54) passes to
    /// `ignis_prefill` / `ignis_decode`. `pub(crate)`: never exposed outside
    /// this crate (mirrors the handle's own C-ABI opacity).
    pub(crate) fn handle(&self) -> *mut ffi::IgnisModel {
        self.handle
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::ignis_model_free(self.handle) };
    }
}

/// Mirrors `ninfer::QType` (`kernel/vendor/src/core/tensor.h`) numeric
/// values -- 1:1 with `enum ignis_qtype`.
fn qtype_code(format: NumericFormat) -> i32 {
    match format {
        NumericFormat::Q4G64F16S => 0,
        NumericFormat::Q5G64F16S => 1,
        NumericFormat::Q6G64F16S => 2,
        NumericFormat::W8G32F16S => 3,
        NumericFormat::Bf16 => 4,
        NumericFormat::Fp32 => 5,
        NumericFormat::I32 => 6,
        NumericFormat::Nvfp4 => 7,
        NumericFormat::Fp8E4M3FnRowBf16S => 8,
    }
}

/// Mirrors `ninfer::QuantLayout` -- 1:1 with `enum ignis_quant_layout`.
fn layout_code(layout: StorageLayout) -> i32 {
    match layout {
        StorageLayout::RowSplitK128V1 => 0,
        StorageLayout::ContiguousLeV1 => 1,
        StorageLayout::BlockScaleK16M128x4V1 => 2,
        StorageLayout::RowScaleV1 => 3,
    }
}

/// Validates the P2-01 (GitHub #83) load options before any device or FFI
/// work: `prefill_chunk_tokens` must be a nonzero multiple of 128 (the
/// reference's own alignment rule, and the alignment the vendored GDN
/// chunked kernels' 64-token chunk divides evenly), `max_context_tokens`
/// must be positive. The leaf validates the same rule again
/// (`kernel/src/model.cu`, defense in depth for a caller that bypasses this
/// wrapper); checking here first fails fast without touching the reader,
/// the artifact, or the device.
fn validate_prefill_config(prefill_chunk_tokens: u32, max_context_tokens: u32) -> Result<(), String> {
    if prefill_chunk_tokens == 0 || prefill_chunk_tokens % 128 != 0 {
        return Err(format!(
            "load_qwen38_27b: prefill_chunk_tokens ({prefill_chunk_tokens}) must be a nonzero multiple of 128"
        ));
    }
    if max_context_tokens == 0 {
        return Err("load_qwen38_27b: max_context_tokens must be positive".to_string());
    }
    // A chunk wider than the sequence pool's own context bound can never be
    // prefilled anyway, and the GQA attention workspace query needs
    // max_visible_keys >= the query width it is sized for.
    if prefill_chunk_tokens > max_context_tokens {
        return Err(format!(
            "load_qwen38_27b: prefill_chunk_tokens ({prefill_chunk_tokens}) must not exceed max_context_tokens ({max_context_tokens})"
        ));
    }
    Ok(())
}

/// The input divisor a weight-only NVFP4 descriptor carries. Nothing reads it
/// as a divisor (the drafter runs A16), but the vendored NVFP4 weight
/// validation requires every weight's input divisor to be finite and
/// positive whatever the route, so 0 fails the first dispatch ("invalid NVFP4
/// weight", P5-03, GitHub #152). 1.0 is the reference's own value for these
/// weights (`targets/qwen3_6_27b/impl/load/bindings.cpp`, `dflash2_matrix`).
pub(crate) const WEIGHT_ONLY_NVFP4_INPUT_DIVISOR: f32 = 1.0;

/// A bound tensor does not cross the ABI if it is a
/// `*_input_scale_divisor` scalar (the W4A4 path that reads them is
/// P2-03, GitHub #85) -- its presence and shape are already validated by
/// [`ignis_artifact::bind_text_scope_27b`].
fn crosses_the_abi(name: &str) -> bool {
    !name.ends_with("/input_scale_divisor")
}

/// An NVFP4 weight with no paired `*_input_scale_divisor` object: the DFlash2
/// drafter's matrices are weight-only (it runs A16; the reference's
/// `qwen3.8-27b-artifact.md` §15.1), so its descriptors carry
/// [`WEIGHT_ONLY_NVFP4_INPUT_DIVISOR`] — no W4A4 path reads it.
fn is_weight_only_nvfp4(name: &str) -> bool {
    name.starts_with("dflash2/")
}

/// The artifact module a speculation option binds beside the text scope
/// (P5-02, GitHub #150). One mapping, shared by the server's binder call and
/// this load, so the handles and the descriptors cannot name different
/// scopes.
pub fn draft_module(speculation: Option<Speculation>) -> Option<DraftModule> {
    speculation.and_then(|s| match s.backend() {
        SpeculativeBackend::Dflash2 => Some(DraftModule::Dflash2),
        // The verify substrate binds no drafter (P5-04, GitHub #153): the
        // text scope alone.
        SpeculativeBackend::VerifyOnly => None,
    })
}

/// Every module a load with these options binds (GitHub #177): one mapping,
/// shared by the server's binder call and this load, so the handles and the
/// descriptors cannot name different scopes.
pub fn model_scope(speculation: Option<Speculation>, vision: Option<Vision>) -> ModelScope {
    ModelScope {
        draft: draft_module(speculation),
        vision: vision.is_some(),
    }
}

/// The options struct a load crosses the ABI with: `None` (a NULL pointer,
/// ADR 0016's production defaults) when the load selects neither speculation
/// nor vision, so such a load is exactly what it was before either existed.
fn load_options(
    speculation: Option<Speculation>,
    vision: Option<Vision>,
) -> Option<ffi::IgnisModelLoadOptions> {
    (speculation.is_some() || vision.is_some()).then(|| ffi::IgnisModelLoadOptions {
        size: std::mem::size_of::<ffi::IgnisModelLoadOptions>() as u32,
        // IGNIS_SPECULATIVE_NONE
        speculative_backend: speculation.map_or(0, |s| s.backend().abi_code()),
        draft_tokens: speculation.map_or(0, |s| s.draft_tokens()),
        vision_max_tokens: vision.map_or(0, |v| v.max_tokens()),
    })
}

/// Read the NVFP4 blockscale layout's trailing FP32 weight divisor
/// directly from the container (host-side, via the mapping -- the same
/// bytes the device upload copied, ADR 0002).
fn read_weight_divisor(reader: &Reader, name: &str, shape: &[u64]) -> Result<f32, String> {
    let geometry = ignis_artifact::block_scale_geometry(NumericFormat::Nvfp4, shape)
        .map_err(|e| e.to_string())?;
    let span = reader.payload(name).map_err(|e| e.to_string())?;
    let offset = geometry.weight_divisor_offset as usize;
    let bytes: [u8; 4] = span.data[offset..offset + 4]
        .try_into()
        .map_err(|_| format!("{name}: weight divisor span is truncated"))?;
    Ok(f32::from_le_bytes(bytes))
}

/// Read a `*_input_scale_divisor` object's FP32 scalar directly from the
/// container (a standalone rank-0 object, not a trailing region of its
/// paired weight -- [`read_weight_divisor`] reads that one).
///
/// Every NVFP4 projection's reference `Weight` carries a finite, positive
/// `input_scale_divisor` regardless of which compute policy consumes it
/// (`kernel/vendor/src/ops/linear/nvfp4/nvfp4_format.cpp`'s
/// `validate_nvfp4_weight` requires it unconditionally). The W4A4 path that
/// multiplies by it is live (P2-03, GitHub #85), so a wrong or missing
/// divisor now fails a test rather than degrading quietly (acceptance #85.4).
fn read_input_scale_divisor(reader: &Reader, name: &str) -> Result<f32, String> {
    let span = reader.payload(name).map_err(|e| e.to_string())?;
    let bytes: [u8; 4] = span
        .data
        .get(0..4)
        .ok_or_else(|| format!("{name}: input divisor span is truncated"))?
        .try_into()
        .map_err(|_| format!("{name}: input divisor span is truncated"))?;
    Ok(f32::from_le_bytes(bytes))
}

/// Where one bound tensor's bytes are, as a descriptor carries them.
struct TensorPlacement {
    shape: Vec<u64>,
    bytes: u64,
    qdata: *const c_void,
    qhigh: *const c_void,
    scales: *const c_void,
}

/// The placements of a materialized artifact: its device views.
fn device_placement(
    artifact: &MaterializedArtifact,
) -> impl Fn(ObjectHandle, &InventoryEntry) -> Result<TensorPlacement, String> + '_ {
    move |handle, _entry| {
        let view = artifact.device_view(handle).map_err(|e| e.to_string())?;
        Ok(TensorPlacement {
            shape: view.shape.clone(),
            bytes: view.bytes,
            qdata: view.base as *const c_void,
            qhigh: view
                .high_plane()
                .map_or(std::ptr::null(), |p| p as *const c_void),
            scales: view
                .scale_plane()
                .map_or(std::ptr::null(), |p| p as *const c_void),
        })
    }
}

/// The placements a materialization plan will make, before any device
/// memory exists (GitHub #210): each tensor's inventory shape and planned
/// bytes, with no data pointers -- enough for the leaf to size a load, which
/// is all `ignis_model_plan_reservations` reads.
fn planned_placement(
    plan: &MaterializationPlan,
) -> impl Fn(ObjectHandle, &InventoryEntry) -> Result<TensorPlacement, String> + '_ {
    move |handle, entry| {
        let placed = plan
            .device_objects
            .iter()
            .find(|placed| placed.handle == handle)
            .ok_or_else(|| format!("{}: not a device object of the materialization plan", entry.name))?;
        Ok(TensorPlacement {
            shape: entry.shape.to_vec(),
            bytes: placed.bytes,
            qdata: std::ptr::null(),
            qhigh: std::ptr::null(),
            scales: std::ptr::null(),
        })
    }
}

/// Build the bound-tensor descriptors for every tensor
/// [`ignis_artifact::bind_model_scope_27b_with`] placed on the device for
/// `scope`, in [`model_scope_27b_with`] order, at the addresses `placement`
/// gives.
///
/// Returns the descriptors alongside the [`CString`] names they point
/// into: the caller must keep both alive across the `ignis_model_load`
/// call (the leaf only reads `name` for the duration of that call).
fn build_bound_tensors(
    reader: &Reader,
    handles: &[ObjectHandle],
    scope: ModelScope,
    placement: impl Fn(ObjectHandle, &InventoryEntry) -> Result<TensorPlacement, String>,
) -> Result<(Vec<CString>, Vec<ffi::IgnisBoundTensor>), String> {
    let entries = model_scope_27b_with(scope);
    if entries.len() != handles.len() {
        return Err(format!(
            "model-scope handle count ({}) does not match the inventory ({}, scope: {scope:?})",
            handles.len(),
            entries.len()
        ));
    }

    let mut names = Vec::with_capacity(entries.len());
    let mut tensors = Vec::with_capacity(entries.len());
    for (i, (entry, &handle)) in entries.iter().zip(handles.iter()).enumerate() {
        if !crosses_the_abi(entry.name) {
            continue;
        }
        let view = placement(handle, entry)?;

        let (weight_scale_divisor, input_scale_divisor) = if entry.format == NumericFormat::Nvfp4
            && is_weight_only_nvfp4(entry.name)
        {
            (
                read_weight_divisor(reader, entry.name, entry.shape)?,
                WEIGHT_ONLY_NVFP4_INPUT_DIVISOR,
            )
        } else if entry.format == NumericFormat::Nvfp4 {
            // The paired `<name>/..._projection/input_scale_divisor` object
            // (present for every NVFP4 projection) is generated immediately
            // after its weight in `text_scope_27b`'s per-layer templates
            // (`crates/artifact/src/inventory.rs`); the `real_artifact`
            // cross-check pins that adjacency against the container.
            let divisor_name = entries
                .get(i + 1)
                .map(|next| next.name)
                .filter(|name| name.ends_with("/input_scale_divisor"))
                .ok_or_else(|| {
                    format!("{}: expected a paired input_scale_divisor object", entry.name)
                })?;
            (
                read_weight_divisor(reader, entry.name, entry.shape)?,
                read_input_scale_divisor(reader, divisor_name)?,
            )
        } else {
            (0.0, 0.0)
        };

        if view.shape.len() > 4 {
            return Err(format!("{}: rank {} exceeds the ABI's rank-4 shape", entry.name, view.shape.len()));
        }
        let mut shape = [1i32; 4];
        for (dst, &dim) in shape.iter_mut().zip(view.shape.iter()) {
            *dst = i32::try_from(dim)
                .map_err(|_| format!("{}: dimension {dim} overflows i32", entry.name))?;
        }
        // No distinct padded shape is tracked by `TensorView` today (every
        // text-scope tensor's stored shape is already the padded one, per
        // the container's own geometry); carry the same value until a
        // padded-vs-logical distinction is needed.
        let padded_shape = shape;

        let name = CString::new(entry.name).map_err(|e| e.to_string())?;
        tensors.push(ffi::IgnisBoundTensor {
            name: name.as_ptr(),
            qtype: qtype_code(entry.format),
            layout: layout_code(entry.layout),
            qdata: view.qdata,
            qhigh: view.qhigh,
            scales: view.scales,
            bytes: view.bytes,
            shape,
            padded_shape,
            ndim: view.shape.len() as u32,
            weight_scale_divisor,
            input_scale_divisor,
        });
        names.push(name);
    }
    Ok((names, tensors))
}

/// The Qwen 3.8-27B topology descriptor (ADR 0009: one source for the
/// leaf's per-layer schema, not guesses). `layer_kinds_buf` is the backing
/// storage for the returned descriptor's pointer -- keep it alive for as
/// long as the descriptor is used.
fn qwen38_27b_topology(layer_kinds_buf: &mut Vec<i32>) -> ffi::IgnisTopology {
    let cfg = ModelConfig::qwen38_27b();
    layer_kinds_buf.clear();
    layer_kinds_buf.extend(cfg.layer_kinds.iter().map(|kind| match kind {
        LayerKind::Gdn => 0,
        LayerKind::Gqa => 1,
    }));
    ffi::IgnisTopology {
        num_layers: cfg.num_layers as u32,
        layer_kinds: layer_kinds_buf.as_ptr(),
        hidden: cfg.hidden,
        vocab: cfg.vocab,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        rotary_dim: cfg.rotary_dim,
        rope_theta: cfg.rope_theta,
        gdn_state_rows: cfg.gdn_state_rows,
        gdn_state_cols: cfg.gdn_state_cols,
        gdn_num_layers: cfg.gdn_num_layers,
        gdn_q_width: cfg.gdn_q_width,
        gdn_z_width: cfg.gdn_z_width,
        gdn_ab_width: cfg.gdn_ab_width,
        ffn_intermediate: cfg.ffn_intermediate,
        // The Qwen 3.8-27B text config's RMSNorm epsilon (the reference's
        // `TextConfig::rms_epsilon`, `qwen3_6_27b/impl/config.h`).
        rms_norm_eps: 1.0e-6,
    }
}

/// Load the Qwen 3.8-27B text model from a device-materialized artifact
/// (P1-17): build the bound-tensor + topology descriptors and call
/// `ignis_model_load`. `handles` must be the handles [`ignis_artifact::bind_text_scope_27b`]
/// returned for the same `reader` that produced `artifact`.
///
/// `prefill_chunk_tokens` is the widest prefill chunk this model handle will
/// ever be asked to run (P2-01, GitHub #83): the load reserves the program
/// scratch once for a chunk of that width. Must be a nonzero multiple of
/// 128. `max_context_tokens` must match (or bound) the largest
/// `max_context_tokens` the caller's [`crate::seq::SeqPool`] will be built
/// with — it sizes the GQA attention workspace for the worst-case visible-key
/// count.
///
/// `kv_format` must be the format of every [`crate::seq::SeqPool`] used with
/// the returned handle (P4-05, GitHub #123). It is a load argument, not
/// something the leaf reads off the pool, because the attention workspace is
/// part of the scratch reservation and its size depends on the format: the
/// hq-e8-2b prompt route materializes the envelope's visible history into two
/// rotated-frame BF16 scratch planes that BF16's own prompt route has no
/// counterpart for. A pool built in the other format is refused by the
/// layer entry points rather than run against an arena sized for this one —
/// the format is fixed for the life of a load (ADR 0022).
pub fn load_qwen38_27b(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    prefill_chunk_tokens: u32,
    max_context_tokens: u32,
    kv_format: KvFormat,
) -> Result<Model, String> {
    load_qwen38_27b_with_speculation(
        reader,
        artifact,
        handles,
        prefill_chunk_tokens,
        max_context_tokens,
        kv_format,
        None,
    )
}

/// [`load_qwen38_27b`] with speculation chosen at load (P5-02, GitHub #150).
///
/// With `Some`, `handles` must be the handles
/// [`ignis_artifact::bind_model_scope_27b`] returned for [`draft_module`] of
/// the same option; the leaf binds the drafter's weights from them and sizes
/// the prefill scratch for its context append. The drafter's per-sequence
/// window lives in the pool, which must be built with
/// [`crate::seq::SeqPool::create_with_speculation`] for the same backend
/// (P5-03, GitHub #152); both are reported by [`crate::step::program_stats`].
/// With `None` the options pointer crosses as
/// NULL (ADR 0016: production defaults) and the load is exactly today's.
pub fn load_qwen38_27b_with_speculation(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    prefill_chunk_tokens: u32,
    max_context_tokens: u32,
    kv_format: KvFormat,
    speculation: Option<Speculation>,
) -> Result<Model, String> {
    load_qwen38_27b_with_options(
        reader,
        artifact,
        handles,
        prefill_chunk_tokens,
        max_context_tokens,
        kv_format,
        speculation,
        None,
    )
}

/// [`load_qwen38_27b_with_speculation`] with vision chosen at load too
/// (GitHub #177). With `Some`, `handles` must be the handles
/// [`ignis_artifact::bind_model_scope_27b_with`] returned for [`model_scope`]
/// of the same options: the leaf binds every `vision/*` weight from them,
/// sizes its prefill scratch to fit the encoder workspace too (GitHub #212)
/// and reserves the output transient for the envelope, reported as
/// [`IgnisModelStats::vision_reserved_bytes`]. With neither
/// option the options pointer crosses as NULL and the load is exactly today's.
#[allow(clippy::too_many_arguments)]
pub fn load_qwen38_27b_with_options(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    prefill_chunk_tokens: u32,
    max_context_tokens: u32,
    kv_format: KvFormat,
    speculation: Option<Speculation>,
    vision: Option<Vision>,
) -> Result<Model, String> {
    validate_prefill_config(prefill_chunk_tokens, max_context_tokens)?;
    let (_names, tensors) = build_bound_tensors(
        reader,
        handles,
        model_scope(speculation, vision),
        device_placement(artifact),
    )?;
    let mut layer_kinds_buf = Vec::new();
    let topology = qwen38_27b_topology(&mut layer_kinds_buf);
    let options = load_options(speculation, vision);

    let mut handle: *mut ffi::IgnisModel = std::ptr::null_mut();
    let rc = unsafe {
        ffi::ignis_model_load(
            tensors.as_ptr(),
            tensors.len() as u64,
            &topology,
            prefill_chunk_tokens,
            max_context_tokens,
            kv_format.abi_code(),
            options
                .as_ref()
                .map_or(std::ptr::null(), |o| o as *const ffi::IgnisModelLoadOptions),
            &mut handle,
        )
    };
    if rc != 0 || handle.is_null() {
        let message = unsafe { CStr::from_ptr(ffi::ignis_model_last_error()) };
        return Err(message.to_string_lossy().into_owned());
    }
    Ok(Model { handle })
}

/// What [`load_qwen38_27b_with_options`] would reserve beside the weights for
/// the same options, asked before the weights are on the device (GitHub
/// #210): `plan` and `handles` are what
/// [`ignis_artifact::bind_model_scope_27b_with`] returned, not yet
/// materialized. The leaf binds the descriptors and sizes every reservation
/// exactly as the load would, and allocates nothing.
#[allow(clippy::too_many_arguments)]
pub fn plan_qwen38_27b_reservations(
    reader: &Reader,
    plan: &MaterializationPlan,
    handles: &[ObjectHandle],
    prefill_chunk_tokens: u32,
    max_context_tokens: u32,
    kv_format: KvFormat,
    speculation: Option<Speculation>,
    vision: Option<Vision>,
) -> Result<IgnisModelReservations, String> {
    validate_prefill_config(prefill_chunk_tokens, max_context_tokens)?;
    let (_names, tensors) = build_bound_tensors(
        reader,
        handles,
        model_scope(speculation, vision),
        planned_placement(plan),
    )?;
    let mut layer_kinds_buf = Vec::new();
    let topology = qwen38_27b_topology(&mut layer_kinds_buf);
    let options = load_options(speculation, vision);

    let mut reservations = IgnisModelReservations::default();
    let rc = unsafe {
        ffi::ignis_model_plan_reservations(
            tensors.as_ptr(),
            tensors.len() as u64,
            &topology,
            prefill_chunk_tokens,
            max_context_tokens,
            kv_format.abi_code(),
            options
                .as_ref()
                .map_or(std::ptr::null(), |o| o as *const ffi::IgnisModelLoadOptions),
            &mut reservations,
        )
    };
    if rc != 0 {
        let message = unsafe { CStr::from_ptr(ffi::ignis_model_last_error()) };
        return Err(message.to_string_lossy().into_owned());
    }
    Ok(reservations)
}

// ---------------------------------------------------------------------------
// Tests (CPU-only: no CUDA call in this file's mapping / host-read helpers,
// so these run without a GPU -- only `--features cuda` gates compiling them
// at all, the same as the rest of this module).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_prefill_config_accepts_the_default_chunk() {
        assert!(validate_prefill_config(1024, 4096).is_ok());
    }

    #[test]
    fn validate_prefill_config_rejects_a_zero_chunk() {
        let err = validate_prefill_config(0, 4096).unwrap_err();
        assert!(err.contains("nonzero multiple of 128"), "{err}");
    }

    #[test]
    fn validate_prefill_config_rejects_a_chunk_not_a_multiple_of_128() {
        let err = validate_prefill_config(100, 4096).unwrap_err();
        assert!(err.contains("nonzero multiple of 128"), "{err}");
    }

    #[test]
    fn validate_prefill_config_rejects_a_zero_max_context() {
        let err = validate_prefill_config(1024, 0).unwrap_err();
        assert!(err.contains("max_context_tokens"), "{err}");
    }

    #[test]
    fn validate_prefill_config_rejects_a_chunk_wider_than_the_context_bound() {
        let err = validate_prefill_config(1024, 128).unwrap_err();
        assert!(err.contains("must not exceed max_context_tokens"), "{err}");
    }

    #[test]
    fn qtype_code_matches_the_leaf_enum() {
        // 1:1 with `enum ignis_qtype` (kernel/include/ignis_model.h).
        assert_eq!(qtype_code(NumericFormat::Q4G64F16S), 0);
        assert_eq!(qtype_code(NumericFormat::Q5G64F16S), 1);
        assert_eq!(qtype_code(NumericFormat::Q6G64F16S), 2);
        assert_eq!(qtype_code(NumericFormat::W8G32F16S), 3);
        assert_eq!(qtype_code(NumericFormat::Bf16), 4);
        assert_eq!(qtype_code(NumericFormat::Fp32), 5);
        assert_eq!(qtype_code(NumericFormat::I32), 6);
        assert_eq!(qtype_code(NumericFormat::Nvfp4), 7);
        assert_eq!(qtype_code(NumericFormat::Fp8E4M3FnRowBf16S), 8);
    }

    #[test]
    fn layout_code_matches_the_leaf_enum() {
        // 1:1 with `enum ignis_quant_layout` (kernel/include/ignis_model.h).
        assert_eq!(layout_code(StorageLayout::RowSplitK128V1), 0);
        assert_eq!(layout_code(StorageLayout::ContiguousLeV1), 1);
        assert_eq!(layout_code(StorageLayout::BlockScaleK16M128x4V1), 2);
        assert_eq!(layout_code(StorageLayout::RowScaleV1), 3);
    }

    #[test]
    fn crosses_the_abi_excludes_only_input_scale_divisor_objects() {
        assert!(!crosses_the_abi(
            "text/layers/3/attention/input_projection/input_scale_divisor"
        ));
        assert!(crosses_the_abi("text/layers/3/attention/query_key_gate_value"));
        assert!(crosses_the_abi("text/token_embedding"));
    }

    #[test]
    fn a_weight_only_nvfp4_input_divisor_passes_the_vendored_weight_validation() {
        // nvfp4_format.cpp refuses a non-finite or non-positive input divisor
        // on every route, A16 included.
        assert!(WEIGHT_ONLY_NVFP4_INPUT_DIVISOR.is_finite());
        assert!(WEIGHT_ONLY_NVFP4_INPUT_DIVISOR > 0.0);
        assert_eq!(WEIGHT_ONLY_NVFP4_INPUT_DIVISOR, 1.0, "the reference's value");
    }

    #[test]
    fn only_the_drafter_carries_weight_only_nvfp4() {
        assert!(is_weight_only_nvfp4("dflash2/layers/0/mlp/gate_up"));
        assert!(is_weight_only_nvfp4("dflash2/selector/successor"));
        assert!(!is_weight_only_nvfp4("text/layers/3/mlp/gate_up"));
    }

    #[test]
    fn a_speculation_option_selects_its_draft_module_and_none_selects_nothing() {
        assert_eq!(draft_module(None), None);
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        assert_eq!(draft_module(Some(spec)), Some(DraftModule::Dflash2));
    }

    #[test]
    fn the_model_scope_names_vision_only_when_the_load_asks_for_it() {
        assert_eq!(model_scope(None, None), ModelScope::default());
        let vision = Vision::default();
        assert_eq!(model_scope(None, Some(vision)), ModelScope { draft: None, vision: true });
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 3).unwrap();
        assert_eq!(
            model_scope(Some(spec), Some(vision)),
            ModelScope { draft: Some(DraftModule::Dflash2), vision: true }
        );
    }

    #[test]
    fn a_load_with_neither_option_crosses_a_null_options_pointer() {
        assert_eq!(load_options(None, None), None, "today's load, byte for byte");
    }

    #[test]
    fn a_vision_only_load_crosses_the_envelope_with_no_speculation() {
        let options = load_options(None, Some(Vision::new(8192).unwrap())).expect("options");
        assert_eq!(options.size as usize, std::mem::size_of::<ffi::IgnisModelLoadOptions>());
        assert_eq!(options.speculative_backend, 0);
        assert_eq!(options.draft_tokens, 0);
        assert_eq!(options.vision_max_tokens, 8192);

        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        let without_vision = load_options(Some(spec), None).expect("options");
        assert_eq!(without_vision.vision_max_tokens, 0);
        assert_eq!(without_vision.draft_tokens, 7);
    }

    #[test]
    fn the_options_mirror_is_the_leaf_structs_size() {
        // uint32 size, int32 backend, uint32 draft_tokens, uint32 vision_max_tokens.
        assert_eq!(std::mem::size_of::<ffi::IgnisModelLoadOptions>(), 16);
        // uint64 x 3.
        assert_eq!(std::mem::size_of::<IgnisModelStats>(), 24);
    }

    #[test]
    fn read_weight_divisor_decodes_the_trailing_fp32() {
        let shape = vec![128u64, 64];
        let geometry = ignis_artifact::block_scale_geometry(NumericFormat::Nvfp4, &shape)
            .expect("blockscale geometry");
        let mut payload = vec![0u8; geometry.encoded_bytes as usize];
        let divisor: f32 = 2.5;
        let offset = geometry.weight_divisor_offset as usize;
        payload[offset..offset + 4].copy_from_slice(&divisor.to_le_bytes());

        let objects = vec![ignis_artifact::fixture::FixtureObject::Tensor {
            name: "w/nvfp4",
            shape: shape.clone(),
            format: "NVFP4",
            layout: "blockscale-k16-m128x4-v1",
            offset: 0,
            bytes: geometry.encoded_bytes,
        }];
        let artifact =
            ignis_artifact::fixture::write_fixture(&objects, &payload, "weight-divisor")
                .expect("write fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");

        let got = read_weight_divisor(&reader, "w/nvfp4", &shape).expect("read divisor");
        assert_eq!(got, divisor);
    }
}
