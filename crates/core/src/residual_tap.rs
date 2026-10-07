//! Test-only Flash-Next residual-stack tap (`kernel/include/ignis_fn_residual_tap.h`).
//!
//! Spec flash-next/07 phase A feeds the checkpoint's MTP head the trunk's
//! final pre-mixer stack (`[streams][hidden]` per position, the last decoder
//! layer's output before `hyper_connection_mixer`), as the engine computes
//! it. This arms a process-wide copy of that stack from every prefill chunk
//! to a host buffer.
//!
//! Gated behind the non-default `residual-tap` feature, like `kv-capture`
//! and `attn-tap`: no production build declares these symbols, so none links
//! the capture; the only production-side trace is the disarmed flag load in
//! the Flash-Next prefill.

use std::ffi::CStr;
use std::os::raw::c_char;

mod ffi {
    use super::*;
    unsafe extern "C" {
        pub fn ignis_fn_residual_tap_arm(out_rows: *mut u16, first_position: i64, max_rows: i64, row_elems: i32) -> i32;
        pub fn ignis_fn_residual_tap_disarm(rows_written: *mut i64) -> i32;
        pub fn ignis_fn_residual_tap_last_error() -> *const c_char;
    }
}

fn last_error() -> String {
    let msg = unsafe { CStr::from_ptr(ffi::ignis_fn_residual_tap_last_error()) };
    msg.to_string_lossy().into_owned()
}

/// Arm the tap for positions `[first_position, first_position + rows)` of
/// `row_elems` BF16 elements each, run `f` (one sequence's prefill), and
/// disarm -- always, even when `f` panics. Returns `f`'s value, the captured
/// rows (BF16 bits, row-major) and how many rows the prefill wrote.
///
/// The arm is process-wide: nothing else may prefill while `f` runs.
pub fn with_residual_tap<T>(
    first_position: i64,
    rows: usize,
    row_elems: usize,
    f: impl FnOnce() -> T,
) -> Result<(T, Vec<u16>, usize), String> {
    let mut out = vec![0u16; rows * row_elems];
    let rc = unsafe {
        ffi::ignis_fn_residual_tap_arm(out.as_mut_ptr(), first_position, rows as i64, row_elems as i32)
    };
    if rc != 0 {
        return Err(last_error());
    }
    struct Disarm(bool);
    impl Drop for Disarm {
        fn drop(&mut self) {
            if !self.0 {
                unsafe { ffi::ignis_fn_residual_tap_disarm(std::ptr::null_mut()) };
            }
        }
    }
    let mut guard = Disarm(false);
    let value = f();
    let mut written = 0i64;
    unsafe { ffi::ignis_fn_residual_tap_disarm(&mut written) };
    guard.0 = true;
    Ok((value, out, written as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arm refuses what it cannot copy into, and one arm at a time; an
    /// arm with nothing recorded disarms with zero rows, and a panicking
    /// prefill still disarms. One test, because the arm is process-wide and
    /// tests run on parallel threads. No GPU: arming touches none.
    #[test]
    fn the_arm_refuses_bad_shapes_and_a_second_arm_and_always_disarms() {
        let err = with_residual_tap(-1, 4, 8, || ()).unwrap_err();
        assert!(err.contains("first_position -1"), "{err}");
        let err = with_residual_tap(0, 0, 8, || ()).unwrap_err();
        assert!(err.contains("max_rows 0"), "{err}");
        let err = with_residual_tap(0, 4, 0, || ()).unwrap_err();
        assert!(err.contains("row_elems 0"), "{err}");

        let (inner, out, written) = with_residual_tap(0, 4, 8, || with_residual_tap(0, 4, 8, || ())).unwrap();
        assert!(inner.unwrap_err().contains("already armed"));
        assert_eq!((out.len(), written), (32, 0));

        let caught = std::panic::catch_unwind(|| with_residual_tap(0, 2, 2, || panic!("prefill failed")));
        assert!(caught.is_err());
        with_residual_tap(0, 1, 1, || ()).expect("the panic disarmed the tap");
    }
}
