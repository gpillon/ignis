//! The 27B's dense trellis linear in the kernel leaf (spec 27b-16gb/02, GitHub #337): the flat C
//! ABI of `kernel/include/ignis_trellis_linear.h`, 1:1, thin wrappers that turn its return codes
//! into `Result`s carrying the leaf's own message, and the workspace's plan line (ADR 0030).
//!
//! The ops are program-side ops (ADR 0009): the 27B's trellis body calls them inside the leaf
//! (spec 27b-16gb/05a, `kernel/src/trellis_body.h`), whose load sizes and reserves their workspace
//! itself (the `trellis_workspace` plan line); these bindings exist for Rust-side tests and tools.
//! Every pointer argument is a device pointer the caller vouches for,
//! so the wrappers are `unsafe`; the shapes and counts they take are checked by the leaf.
//!
//! All of it is our own implementation, built on our own Flash-Next trellis decode (ADR 0051).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// The decode route serves 1..=`DECODE_MAX_TOKENS` tokens; more take the prefill route.
pub const DECODE_MAX_TOKENS: u32 = 64;

/// `ignis_trellis_linear`'s epilogues: store, or add into the residual in place.
pub const STORE: u32 = 0;
pub const ADD: u32 = 1;

/// One trellis weight, 1:1 with `struct ignis_trellis_weight`: exllamav3's tiles at `k2 = 2 K`,
/// `suh` [in] and `svh` [out] fp16; `tile_stride` 0 for `out / 16`, or a fused plane's row width
/// in tiles for a view of its columns.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisWeight {
    pub trellis: *const c_void,
    pub suh: *const c_void,
    pub svh: *const c_void,
    pub k2: u32,
    pub in_features: u32,
    pub out_features: u32,
    pub tile_stride: u32,
}

/// A workspace, 1:1 with `struct ignis_trellis_workspace`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisWorkspace {
    pub base: *mut c_void,
    pub bytes: u64,
}

/// 1:1 with `kernel/include/ignis_trellis_linear.h`.
pub mod ffi {
    use super::{TrellisWeight, TrellisWorkspace};
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_trellis_prepare() -> i32;
        pub fn ignis_trellis_sm_cap() -> u32;
        pub fn ignis_trellis_workspace_bytes(in_features: u32, out_features: u32, weights: u32, tokens: u32) -> u64;
        pub fn ignis_trellis_workspace_init(workspace: *const TrellisWorkspace, stream: *mut c_void) -> i32;
        pub fn ignis_trellis_linear(
            w: *const TrellisWeight,
            x: *const c_void,
            tokens: u32,
            y: *mut c_void,
            ldy: u32,
            epilogue: u32,
            workspace: *const TrellisWorkspace,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_trellis_linear_swiglu(
            gate: *const TrellisWeight,
            up: *const TrellisWeight,
            x: *const c_void,
            tokens: u32,
            h: *mut c_void,
            ldh: u32,
            workspace: *const TrellisWorkspace,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_trellis_reconstruct(w: *const TrellisWeight, w_f16: *mut c_void, stream: *mut c_void) -> i32;
        pub fn ignis_trellis_last_error() -> *const c_char;
        // `kernel/include/ignis_dense_prefill.h`'s test-only switch (GitHub #511).
        pub fn ignis_dense_prefill_set_fused_producers(fused: u32);
        pub fn ignis_dense_prefill_fused_producers() -> u32;
        pub fn ignis_dense_prefill_fused_norm_launches() -> u64;
    }
}

/// The leaf's message for the most recent failed trellis call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_trellis_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Prepares the current device for every op (kernel attributes, launch geometry, the debug SM
/// cap `IGNIS_TRELLIS_SM_CAP`); once per device at load, outside any stream capture.
pub fn prepare() -> Result<(), String> {
    check(unsafe { ffi::ignis_trellis_prepare() })
}

/// The SM cap the current device was prepared with, 0 for none.
pub fn sm_cap() -> u32 {
    unsafe { ffi::ignis_trellis_sm_cap() }
}

/// Test-only (GitHub #511): `false` sends the trellis body's dense prefill (`PrefillLinear::Int8`)
/// back to its standalone A8 passes, the RMSnorms and the SwiGLU writing their BF16 rows for the
/// quantize to read; `true`, the default, fuses A8 into those producers. The activations are the
/// same bits either way, which is what a test of the switch holds. Process-wide and read at each
/// body call: set it between prefills, never during one. Public for the GPU test
/// `tests/prefill_linear_gpu.rs`; no serving path calls it.
#[doc(hidden)]
pub fn set_dense_prefill_fused_producers(fused: bool) {
    unsafe { ffi::ignis_dense_prefill_set_fused_producers(u32::from(fused)) }
}

/// Whether the dense prefill fuses A8 into its producers ([`set_dense_prefill_fused_producers`]).
#[doc(hidden)]
pub fn dense_prefill_fused_producers() -> bool {
    unsafe { ffi::ignis_dense_prefill_fused_producers() != 0 }
}

/// The fused RMSnorm launches this process has enqueued: which passes a run under
/// [`set_dense_prefill_fused_producers`] took.
#[doc(hidden)]
pub fn dense_prefill_fused_norm_launches() -> u64 {
    unsafe { ffi::ignis_dense_prefill_fused_norm_launches() }
}

/// One call's shape: a linear (`weights` 1) or a gate/up pair (`weights` 2, each `out` wide).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub in_features: u32,
    pub out_features: u32,
    pub weights: u32,
}

/// The workspace one call needs at `tokens` tokens on the current device, prepared first: the
/// decode route splits its inputs for the device's resident CTAs (or the SM cap's). `Err` for a
/// shape the ops refuse or an unprepared device.
pub fn workspace_bytes(call: Call, tokens: u32) -> Result<u64, String> {
    let bytes = unsafe { ffi::ignis_trellis_workspace_bytes(call.in_features, call.out_features, call.weights, tokens) };
    if bytes == 0 {
        return Err(format!(
            "no trellis workspace for {call:?} at {tokens} tokens (unprepared device, or a shape the ops refuse): {}",
            last_error()
        ));
    }
    Ok(bytes)
}

/// The plan line (ADR 0030): the largest workspace any of a load's `calls` needs, on the decode
/// route at up to `decode_tokens` and on the prefill route at up to `prefill_tokens` (the load's
/// chunk). One workspace serves every call of one stream in sequence. The decode route's split
/// grows its partial sums with the device's resident CTAs, so the line is the device's own.
pub fn plan_bytes(calls: &[Call], decode_tokens: u32, prefill_tokens: u32) -> Result<u64, String> {
    if decode_tokens == 0 || decode_tokens > DECODE_MAX_TOKENS {
        return Err(format!("decode tokens must be 1..={DECODE_MAX_TOKENS}, got {decode_tokens}"));
    }
    let mut line = 0u64;
    for &call in calls {
        // Every decode width up to the load's: a narrower call may split further.
        for tokens in 1..=decode_tokens {
            line = line.max(workspace_bytes(call, tokens)?);
        }
        if prefill_tokens > DECODE_MAX_TOKENS {
            line = line.max(workspace_bytes(call, prefill_tokens)?);
        }
    }
    Ok(line)
}

/// Zeroes a fresh workspace (its counters); once after it is reserved.
///
/// # Safety
/// `workspace.base` is a device buffer of `workspace.bytes`.
pub unsafe fn workspace_init(workspace: &TrellisWorkspace, stream: *mut c_void) -> Result<(), String> {
    check(unsafe { ffi::ignis_trellis_workspace_init(workspace, stream) })
}

/// `y = x W` through `epilogue` (`STORE` or `ADD`), BF16, token-major.
///
/// # Safety
/// Every pointer is a device pointer of the size the shapes imply.
pub unsafe fn linear(
    w: &TrellisWeight,
    x: *const c_void,
    tokens: u32,
    y: *mut c_void,
    ldy: u32,
    epilogue: u32,
    workspace: &TrellisWorkspace,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_trellis_linear(w, x, tokens, y, ldy, epilogue, workspace, stream) })
}

/// `h = silu(x G) * (x U)`, BF16, token-major.
///
/// # Safety
/// Every pointer is a device pointer of the size the shapes imply.
pub unsafe fn linear_swiglu(
    gate: &TrellisWeight,
    up: &TrellisWeight,
    x: *const c_void,
    tokens: u32,
    h: *mut c_void,
    ldh: u32,
    workspace: &TrellisWorkspace,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_trellis_linear_swiglu(gate, up, x, tokens, h, ldh, workspace, stream) })
}

/// The inner matrix of `w` decoded to fp16 `[in][out]`, bit for bit exllamav3's `reconstruct`.
///
/// # Safety
/// `w`'s pointers and `w_f16` are device pointers of the sizes the shape implies.
pub unsafe fn reconstruct(w: &TrellisWeight, w_f16: *mut c_void, stream: *mut c_void) -> Result<(), String> {
    check(unsafe { ffi::ignis_trellis_reconstruct(w, w_f16, stream) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_abi_structs_match_the_header() {
        // ignis_trellis_weight: three pointers then four u32; ignis_trellis_workspace: a pointer
        // and a u64.
        assert_eq!(std::mem::size_of::<TrellisWeight>(), 3 * 8 + 4 * 4);
        assert_eq!(std::mem::size_of::<TrellisWorkspace>(), 16);
    }

    #[test]
    fn a_plan_refuses_a_decode_width_the_route_does_not_serve() {
        let call = Call { in_features: 5120, out_features: 1024, weights: 1 };
        assert!(plan_bytes(&[call], 0, 2048).unwrap_err().contains("decode tokens"));
        assert!(plan_bytes(&[call], 65, 2048).unwrap_err().contains("decode tokens"));
    }
}
