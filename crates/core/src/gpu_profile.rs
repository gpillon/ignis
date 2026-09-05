//! The explicit GPU test profile (GitHub #38, ADR 0006).
//!
//! Outside the profile, a GPU test that hits a busy GPU, a kernel error, or
//! a missing fixture (the real artifact) **skips** — the default
//! `cargo test` stays green without a free GPU. Under the profile the same
//! condition is a **hard failure**: a busy GPU, a missing artifact, or a
//! kernel error must never hide behind "skip" — that pattern is what let a
//! broken forward pass stay green for two tickets
//! (`.scratch/REVIEW-2026-09-05.md` §4.1).
//!
//! Turning the profile on takes **two** things, not one: `IGNIS_GPU_PROFILE=1`
//! *and* a recent preflight pass. ADR 0006 asks for a preflight check in the
//! harness, not a script one has to remember; since `crates/core` has no GPU
//! access of its own (GitHub #39 removed the C ABI), the preflight
//! (`scripts/gpu-preflight.ps1`) leaves a marker file behind when it passes
//! and this module refuses the profile without one. Setting the env var by
//! hand is therefore not enough — run `scripts/gpu-profile.ps1`, which does
//! both and clears up after itself.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use crate::types::ComputeError;

/// Where `scripts/gpu-preflight.ps1` records a pass (and
/// `scripts/gpu-profile.ps1` deletes it once the run is over). The scripts
/// resolve the same path as `$env:TEMP\ignis-gpu-preflight.ok`.
fn marker_path() -> PathBuf {
    env::temp_dir().join("ignis-gpu-preflight.ok")
}

/// How long a passed preflight authorizes the profile for. The marker is
/// removed when a `gpu-profile.ps1` run ends, so this bound only matters
/// when a run was killed: it stops a stale pass from authorizing a later
/// run that never checked whether the reference engine had come back.
const PREFLIGHT_MAX_AGE: Duration = Duration::from_secs(30 * 60);

/// Whether the preflight marker at `path` records a pass recent enough to
/// authorize a profile run at `now`.
fn preflight_passed(path: &Path, now: SystemTime) -> bool {
    let Ok(stamped) = path.metadata().and_then(|meta| meta.modified()) else {
        return false; // no marker: the preflight never passed here
    };
    match now.duration_since(stamped) {
        Ok(age) => age <= PREFLIGHT_MAX_AGE,
        // Stamped in the future — a clock adjustment, not a stale pass.
        Err(_) => true,
    }
}

/// Whether the explicit GPU test profile is active.
///
/// Decided once per process: a GPU run must not start under the profile and
/// then slide out of it when the marker ages out mid-suite.
pub fn active() -> bool {
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| {
        active_impl(
            env::var("IGNIS_GPU_PROFILE").ok(),
            preflight_passed(&marker_path(), SystemTime::now()),
        )
    })
}

fn active_impl(var: Option<String>, preflight_passed: bool) -> bool {
    if !matches!(var.as_deref(), Some("1")) {
        return false;
    }
    if !preflight_passed {
        panic!(
            "IGNIS_GPU_PROFILE=1, but no preflight pass is on record (no recent \
             marker at the path scripts/gpu-preflight.ps1 writes). The profile \
             must not run un-preflighted while the reference engine may hold the \
             GPU (ADR 0006): run `scripts/gpu-profile.ps1`, which preflights, \
             runs the GPU work, and cleans up -- do not set the variable by hand."
        );
    }
    true
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
        // Matched by name, not by `_`: a variant added later must be a
        // deliberate decision here, not silently treated as "not a fault".
        ComputeError::Stopped => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A marker file at a path unique to `name`, stamped now. Returned so
    /// the caller can delete it; the temp dir is the same one `active`
    /// looks in, so the name must not collide with the real marker.
    fn temp_marker(name: &str) -> PathBuf {
        let path = env::temp_dir().join(format!("ignis-gpu-profile-test-{name}.ok"));
        std::fs::write(&path, b"stamped by a unit test").expect("write the test marker");
        path
    }

    #[test]
    fn active_impl_stays_off_without_the_switch_however_the_preflight_went() {
        for preflight_passed in [false, true] {
            assert!(!active_impl(None, preflight_passed));
            assert!(!active_impl(Some("0".to_string()), preflight_passed));
            assert!(!active_impl(Some("true".to_string()), preflight_passed));
        }
    }

    #[test]
    fn active_impl_is_on_when_the_switch_and_a_preflight_pass_agree() {
        assert!(active_impl(Some("1".to_string()), true));
    }

    #[test]
    #[should_panic(expected = "no preflight pass is on record")]
    fn active_impl_refuses_the_switch_without_a_preflight_pass() {
        active_impl(Some("1".to_string()), false);
    }

    #[test]
    fn a_missing_marker_is_not_a_preflight_pass() {
        let absent = env::temp_dir().join("ignis-gpu-profile-test-absent.ok");
        let _ = std::fs::remove_file(&absent);
        assert!(!preflight_passed(&absent, SystemTime::now()));
    }

    #[test]
    fn a_fresh_marker_is_a_preflight_pass() {
        let marker = temp_marker("fresh");
        let passed = preflight_passed(&marker, SystemTime::now());
        let _ = std::fs::remove_file(&marker);
        assert!(passed, "a marker stamped just now must authorize the profile");
    }

    #[test]
    fn a_marker_older_than_the_max_age_is_not_a_preflight_pass() {
        let marker = temp_marker("stale");
        // Ask as of well past the bound instead of back-dating the file:
        // the decision is "how old is the pass", whichever end moves.
        let later = SystemTime::now() + PREFLIGHT_MAX_AGE + Duration::from_secs(60);
        let passed = preflight_passed(&marker, later);
        let _ = std::fs::remove_file(&marker);
        assert!(!passed, "a pass older than PREFLIGHT_MAX_AGE must not authorize a run");
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
