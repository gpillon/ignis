//! One GDN layer in the program (ADR 0009, GitHub #58, P1-22): the layer's
//! full attention + MLP tail, composed in the kernel leaf from the ADR 0010
//! vendored ops on top of the sequence's GDN state (the conv taps + fp32
//! recurrent slot, GitHub #55). Verified against the P1-20 f64 layer
//! reference (crates/artifact f64_reference.rs, evaluate_layer on the GDN
//! layer).
//!
//! Device-resident: `in_residual` / `out_residual` are device BF16
//! `[hidden, num_tokens]` feature-major buffers (no host activation pointer
//! crosses the boundary); `out_residual` is written in place and receives the
//! final residual.

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

use ignis_artifact::DeviceBuffer;

use crate::model_load::Model;
use crate::seq::{Seq, SeqPool};

mod gdn_ffi {
    use std::os::raw::{c_char, c_void};

    use crate::model_load::ffi::IgnisModel;
    use crate::seq::ffi::{IgnisSeq, IgnisSeqPool};

    // 1:1 with `kernel/include/ignis_gdn_layer.h`.
    unsafe extern "C" {
        pub fn ignis_gdn_layer_step(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            seq: *mut IgnisSeq,
            layer: u32,
            in_residual: *const c_void,
            out_residual: *mut c_void,
            num_tokens: u64,
        ) -> i32;

        pub fn ignis_gdn_layer_last_error() -> *const c_char;
    }
}

fn last_error() -> String {
    let message = unsafe { CStr::from_ptr(gdn_ffi::ignis_gdn_layer_last_error()) };
    message.to_string_lossy().into_owned()
}

/// Runs one GDN layer (GitHub #58) for `num_tokens` sequential tokens of one
/// sequence: input RMSNorm -> fused GDN input projection + causal conv -> GDN
/// gating -> per-head fp32 recurrence on the sequence's GDN slot -> gated
/// RMSNorm with z -> output projection + residual -> MLP tail.
///
/// `in_residual` / `out_residual` are device BF16 `[hidden, num_tokens]`
/// feature-major buffers (allocate them with [`ignis_artifact::CudaDevice`]);
/// `out_residual` is written in place and receives the final residual. The
/// layer's GDN state (conv taps + fp32 recurrent slot) is drawn from `seq`'s
/// slot in `pool` and carries across the `num_tokens` tokens; releasing and
/// re-allocating the sequence resets it (a fresh slot reads zero, ignis_seq.h).
pub fn run_gdn_layer(
    model: &Model,
    pool: &SeqPool,
    seq: &Seq,
    layer: u32,
    in_residual: &DeviceBuffer,
    out_residual: &DeviceBuffer,
    num_tokens: u64,
) -> Result<(), String> {
    let in_ptr = in_residual.base_ptr() as *const c_void;
    let out_ptr = out_residual.base_ptr() as *mut c_void;
    let rc = unsafe {
        gdn_ffi::ignis_gdn_layer_step(
            model.handle(),
            pool.handle(),
            seq.handle(),
            layer,
            in_ptr,
            out_ptr,
            num_tokens,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(())
}