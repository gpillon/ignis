//! The explicit GPU test profile (GitHub #38, ADR 0006).
//!
//! Outside the profile, a GPU test that hits a busy GPU, a kernel error, or
//! a missing fixture (the real artifact) **skips** — the default
//! `cargo test` stays green without a free GPU. Under the profile
//! (`IGNIS_GPU_PROFILE=1`, set after the preflight check passes,
//! `scripts/gpu-preflight.ps1`) the same condition is a **hard failure**:
//! a busy GPU, a missing artifact, or a kernel error must never hide behind
//! "skip" — that pattern is what let a broken forward pass stay green for
//! two tickets (`.scratch/REVIEW-2026-09-05.md` §4.1).

use std::env;

use crate::types::ComputeError;

/// Whether the explicit GPU test profile is active.
pub fn active() -> bool {
    active_impl(env::var("IGNIS_GPU_PROFILE").ok())
}

fn active_impl(var: Option<String>) -> bool {
    matches!(var.as_deref(), Some("1"))
}

/// A condition that is a skip outside the profile and a hard failure under
/// it: a busy/absent GPU, a missing fixture, or (via [`check_rc`] /
/// [`check_compute_err`]) a kernel error. Returns `true` when the caller
/// should skip (the message is already printed); panics with `reason` under
/// the profile.
pub fn skip_or_fail(reason: &str) -> bool {
    skip_or_fail_impl(reason, active())
}

fn skip_or_fail_impl(reason: &str, profile_active: bool) -> bool {
    if profile_active {
        panic!(
            "{reason} -- GPU profile active (IGNIS_GPU_PROFILE=1): a busy \
             GPU, a missing fixture, or a kernel error is a hard failure, \
             never a skip (ADR 0006, GitHub #38)"
        );
    }
    eprintln!(
        "SKIP: {reason} (ADR 0006 -- run with IGNIS_GPU_PROFILE=1, after \
         scripts/gpu-preflight.ps1, to make this a hard failure)"
    );
    true
}

/// A kernel return code under ADR 0006 / the GPU profile: `false` (no skip)
/// when `rc == 0`; otherwise [`skip_or_fail`].
pub fn check_rc(rc: i32, what: &str) -> bool {
    check_rc_impl(rc, what, active())
}

fn check_rc_impl(rc: i32, what: &str, profile_active: bool) -> bool {
    if rc == 0 {
        return false;
    }
    skip_or_fail_impl(&format!("{what} returned {rc} (GPU busy/unavailable)"), profile_active)
}

/// A [`ComputeError`] under ADR 0006 / the GPU profile: `false` (no skip)
/// for a non-kernel error (e.g. [`ComputeError::Stopped`], a soft stop, not
/// a fault); otherwise the wrapped kernel rc via [`check_rc`].
pub fn check_compute_err(e: &ComputeError, what: &str) -> bool {
    check_compute_err_impl(e, what, active())
}

fn check_compute_err_impl(e: &ComputeError, what: &str, profile_active: bool) -> bool {
    match e {
        ComputeError::Kernel(rc) => check_rc_impl(*rc, what, profile_active),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_impl_reads_the_flag_value() {
        assert!(!active_impl(None));
        assert!(!active_impl(Some("0".to_string())));
        assert!(!active_impl(Some("true".to_string())));
        assert!(active_impl(Some("1".to_string())));
    }

    #[test]
    fn check_rc_ok_never_skips_or_fails() {
        assert!(!check_rc_impl(0, "thing", false));
        assert!(!check_rc_impl(0, "thing", true));
    }

    #[test]
    fn check_rc_skips_when_profile_inactive() {
        assert!(check_rc_impl(-1, "thing", false));
    }

    #[test]
    #[should_panic(expected = "thing returned -1")]
    fn check_rc_fails_when_profile_active() {
        check_rc_impl(-1, "thing", true);
    }

    #[test]
    fn check_compute_err_stopped_is_never_a_skip_or_failure() {
        assert!(!check_compute_err_impl(&ComputeError::Stopped, "thing", false));
        assert!(!check_compute_err_impl(&ComputeError::Stopped, "thing", true));
    }

    #[test]
    fn check_compute_err_kernel_skips_when_profile_inactive() {
        assert!(check_compute_err_impl(&ComputeError::Kernel(-1), "thing", false));
    }

    #[test]
    #[should_panic(expected = "thing returned -1")]
    fn check_compute_err_kernel_fails_when_profile_active() {
        check_compute_err_impl(&ComputeError::Kernel(-1), "thing", true);
    }
}
