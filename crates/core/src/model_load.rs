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
//! one is still a load failure) but do not cross this ABI yet: the W4A4
//! activation-quant path that reads them is G2
//! (`.scratch/runtime/specs/01-device-resident-forward.md`).

#![cfg(feature = "cuda")]

use std::ffi::{CStr, CString};
use std::os::raw::c_void;

use ignis_artifact::{
    text_scope_27b, MaterializedArtifact, NumericFormat, ObjectHandle, Reader, StorageLayout,
};

use crate::compute::{LayerKind, ModelConfig};

mod ffi {
    use std::os::raw::{c_char, c_void};

    /// Opaque loaded-model handle (`kernel/include/ignis_model.h`).
    ///
    /// FFI-safe: `#[repr(C)]` + non-zero-sized so `*mut IgnisModel` is a
    /// valid C pointer across the boundary (mirrors
    /// `ignis_artifact::ffi::IgnisDevice`).
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

    /// 1:1 with `struct ignis_model_stats`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct IgnisModelStats {
        pub vram_bytes: u64,
        pub bound_tensor_count: u64,
    }

    unsafe extern "C" {
        pub fn ignis_model_load(
            tensors: *const IgnisBoundTensor,
            count: u64,
            topology: *const IgnisTopology,
            out_model: *mut *mut IgnisModel,
        ) -> i32;

        pub fn ignis_model_stats(model: *const IgnisModel, out_stats: *mut IgnisModelStats) -> i32;

        pub fn ignis_model_free(model: *mut IgnisModel);

        pub fn ignis_model_last_error() -> *const c_char;
    }
}

pub use ffi::IgnisModelStats;

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

/// A bound tensor does not cross the ABI if it is a
/// `*_input_scale_divisor` scalar (the W4A4 path that reads them is G2) --
/// its presence and shape are already validated by
/// [`ignis_artifact::bind_text_scope_27b`].
fn crosses_the_abi(name: &str) -> bool {
    !name.ends_with("/input_scale_divisor")
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

/// Build the bound-tensor descriptors for every text-scope tensor
/// [`ignis_artifact::bind_text_scope_27b`] placed on the device, in [`text_scope_27b`]
/// order.
///
/// Returns the descriptors alongside the [`CString`] names they point
/// into: the caller must keep both alive across the `ignis_model_load`
/// call (the leaf only reads `name` for the duration of that call).
fn build_bound_tensors(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
) -> Result<(Vec<CString>, Vec<ffi::IgnisBoundTensor>), String> {
    let entries = text_scope_27b();
    if entries.len() != handles.len() {
        return Err(format!(
            "text-scope handle count ({}) does not match the inventory ({})",
            handles.len(),
            entries.len()
        ));
    }

    let mut names = Vec::with_capacity(entries.len());
    let mut tensors = Vec::with_capacity(entries.len());
    for (entry, &handle) in entries.iter().zip(handles.iter()) {
        if !crosses_the_abi(entry.name) {
            continue;
        }
        let view = artifact.device_view(handle).map_err(|e| e.to_string())?;

        let weight_scale_divisor = if entry.format == NumericFormat::Nvfp4 {
            read_weight_divisor(reader, entry.name, entry.shape)?
        } else {
            0.0
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
            qdata: view.base as *const c_void,
            qhigh: view
                .high_plane()
                .map_or(std::ptr::null(), |p| p as *const c_void),
            scales: view
                .scale_plane()
                .map_or(std::ptr::null(), |p| p as *const c_void),
            bytes: view.bytes,
            shape,
            padded_shape,
            ndim: view.shape.len() as u32,
            weight_scale_divisor,
            input_scale_divisor: 0.0,
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
pub fn load_qwen38_27b(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
) -> Result<Model, String> {
    let (_names, tensors) = build_bound_tensors(reader, artifact, handles)?;
    let mut layer_kinds_buf = Vec::new();
    let topology = qwen38_27b_topology(&mut layer_kinds_buf);

    let mut handle: *mut ffi::IgnisModel = std::ptr::null_mut();
    let rc = unsafe {
        ffi::ignis_model_load(tensors.as_ptr(), tensors.len() as u64, &topology, &mut handle)
    };
    if rc != 0 || handle.is_null() {
        let message = unsafe { CStr::from_ptr(ffi::ignis_model_last_error()) };
        return Err(message.to_string_lossy().into_owned());
    }
    Ok(Model { handle })
}
