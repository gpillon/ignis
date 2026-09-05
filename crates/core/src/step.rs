//! The degenerate step ABI call (ADR 0009, GitHub #54, P1-18): embedding ->
//! final RMSNorm -> W8G32 output head -> argmax, with every decoder layer
//! skipped.
//!
//! Scope (P1-18): `skip_layers` is a test-only switch on the leaf's
//! `ignis_prefill` / `ignis_decode` (`kernel/include/ignis_step.h`) -- no
//! production caller sets it, since the GQA/GDN layer bodies do not exist
//! yet (P1-21 / #57, P1-22 / #58). The functions here always set it, so
//! their names say what they do: [`prefill_degenerate`] and
//! [`decode_degenerate_batch`] are for the acceptance test only.

#![cfg(feature = "cuda")]

use std::ffi::CStr;

use crate::model_load::Model;

mod ffi {
    use std::os::raw::c_char;

    use crate::model_load::ffi::IgnisModel;

    /// 1:1 with `struct ignis_sampling_params`.
    #[repr(C)]
    pub struct IgnisSamplingParams {
        pub greedy: i32,
    }

    unsafe extern "C" {
        pub fn ignis_prefill(
            model: *mut IgnisModel,
            token_ids: *const i32,
            num_tokens: u64,
            start_position: u64,
            skip_layers: i32,
            sampling: *const IgnisSamplingParams,
            out_token_id: *mut i32,
            out_logits: *mut f32,
        ) -> i32;

        pub fn ignis_decode(
            model: *mut IgnisModel,
            token_ids: *const i32,
            batch_size: u64,
            skip_layers: i32,
            sampling: *const IgnisSamplingParams,
            out_token_ids: *mut i32,
            out_logits: *mut f32,
        ) -> i32;

        pub fn ignis_step_last_error() -> *const c_char;
    }
}

/// Greedy sampling (G1) -- the only supported mode until G3 adds
/// temperature / top-p / top-k / penalties / seed.
const GREEDY: ffi::IgnisSamplingParams = ffi::IgnisSamplingParams { greedy: 1 };

fn last_error() -> String {
    let message = unsafe { CStr::from_ptr(ffi::ignis_step_last_error()) };
    message.to_string_lossy().into_owned()
}

/// Runs the degenerate program (GitHub #54) over a token span for one
/// sequence: embedding -> final RMSNorm -> output head -> argmax, with
/// every decoder layer skipped. Returns the last position's argmax token
/// id. `out_logits`, if `Some`, is filled with that position's full
/// vocab-length logits (promoted from the leaf's BF16 storage) -- the
/// caller sizes it to the model's vocab (`ModelConfig::qwen38_27b().vocab`).
///
/// Test-only (`skip_layers` is hardcoded here): no production caller exists
/// until P1-21 / P1-22 (#57 / #58) land the GQA/GDN layer bodies.
pub fn prefill_degenerate(
    model: &Model,
    token_ids: &[i32],
    start_position: u64,
    out_logits: Option<&mut [f32]>,
) -> Result<i32, String> {
    let mut out_token_id: i32 = -1;
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let rc = unsafe {
        ffi::ignis_prefill(
            model.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            start_position,
            1, // skip_layers: test-only (GitHub #54)
            &GREEDY,
            &mut out_token_id,
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(out_token_id)
}

/// Runs the degenerate program (GitHub #54) once per token id in
/// `token_ids` (batch decode: one id per sequence, no cross-token state --
/// the leaf's `ignis_decode`). Returns one argmax token id per input.
/// `out_logits`, if `Some`, must be exactly `token_ids.len() * vocab`
/// entries long; sequence `i`'s logits land at `out_logits[i * vocab ..
/// (i + 1) * vocab]`.
///
/// Test-only (`skip_layers` is hardcoded here): see [`prefill_degenerate`].
pub fn decode_degenerate_batch(
    model: &Model,
    token_ids: &[i32],
    out_logits: Option<&mut [f32]>,
) -> Result<Vec<i32>, String> {
    let mut out_token_ids = vec![-1i32; token_ids.len()];
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let rc = unsafe {
        ffi::ignis_decode(
            model.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            1, // skip_layers: test-only (GitHub #54)
            &GREEDY,
            out_token_ids.as_mut_ptr(),
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(out_token_ids)
}
