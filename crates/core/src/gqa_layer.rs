//! One full-attention (GQA) layer in the device-resident program (ADR 0009,
//! GitHub #57, P1-21). The leaf keeps all intermediate activations, positions,
//! and paged K/V storage on the device; Rust passes opaque model/sequence
//! handles and device buffers only.

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

use ignis_artifact::DeviceBuffer;

use crate::model_load::Model;
use crate::seq::{Seq, SeqPool};

mod gqa_ffi {
    use std::os::raw::{c_char, c_void};

    use crate::model_load::ffi::IgnisModel;
    use crate::seq::ffi::{IgnisSeq, IgnisSeqPool};

    unsafe extern "C" {
        pub fn ignis_gqa_layer_step(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            seq: *mut IgnisSeq,
            layer: u32,
            in_residual: *const c_void,
            out_residual: *mut c_void,
            num_tokens: u64,
        ) -> i32;

        pub fn ignis_gqa_layer_last_error() -> *const c_char;
    }
}

fn last_error() -> String {
    let message = unsafe { CStr::from_ptr(gqa_ffi::ignis_gqa_layer_last_error()) };
    message.to_string_lossy().into_owned()
}

/// Runs one GQA layer for `num_tokens` sequential device-resident BF16 tokens.
/// The buffers have feature-major `[hidden, num_tokens]` layout; `out_residual`
/// receives the layer's final residual, while `seq` owns the appended paged K/V
/// and advances this GQA layer's position frontier.
pub fn run_gqa_layer(
    model: &Model,
    pool: &SeqPool,
    seq: &Seq,
    layer: u32,
    in_residual: &DeviceBuffer,
    out_residual: &DeviceBuffer,
    num_tokens: u64,
) -> Result<(), String> {
    let rc = unsafe {
        gqa_ffi::ignis_gqa_layer_step(
            model.handle(),
            pool.handle(),
            seq.handle(),
            layer,
            in_residual.base_ptr() as *const c_void,
            out_residual.base_ptr() as *mut c_void,
            num_tokens,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(())
}
