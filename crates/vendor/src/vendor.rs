//! Copy, verify and patch-record the vendored reference subtree (ADR 0010).
//!
//! Two independent checks, because they catch different mistakes:
//!
//! - **the vendor tree** — every file the manifest lists exists under the
//!   vendor root and hashes to what the manifest expects (the reference's
//!   hash, or the patched hash when a patch is recorded). This runs without
//!   a reference checkout, so it works from a bare clone.
//! - **the reference** — every listed file in the reference checkout still
//!   hashes to the manifest's pinned hash. This is what catches a manifest
//!   pinned to one commit being synced from another.
//!
//! A sync verifies the reference *first* and copies nothing on a mismatch, so
//! a wrong reference cannot half-overwrite the leaf.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::manifest::Manifest;
use crate::sha256::sha256_hex;

/// What is wrong with one vendored file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    /// The manifest lists it; the tree does not have it.
    Missing,
    /// The file exists but its content is not what the manifest records.
    HashMismatch { expected: String, actual: String },
    /// The entry records a patch whose diff is not committed.
    MissingPatchDiff { diff: String },
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(f, "missing"),
            Self::HashMismatch { expected, actual } => {
                write!(f, "hash mismatch (expected {expected}, found {actual})")
            }
            Self::MissingPatchDiff { diff } => write!(f, "recorded patch {diff} is not committed"),
        }
    }
}

/// One file's problem, named by its manifest path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub path: String,
    pub problem: Problem,
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.problem)
    }
}

/// The verdict of one verification pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub checked: usize,
    pub findings: Vec<Finding>,
}

impl Report {
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }
}

/// What a sync did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncReport {
    pub copied: Vec<String>,
    /// Patched files left alone (a sync would discard the recorded patch).
    pub kept_patched: Vec<String>,
}

/// Hash a file; `None` when it does not exist.
fn hash_file(path: &Path) -> std::io::Result<Option<String>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(sha256_hex(&bytes))),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

/// Check the vendor tree against the manifest. No reference checkout needed.
pub fn verify_vendor_tree(vendor_root: &Path, manifest: &Manifest) -> std::io::Result<Report> {
    let mut report = Report::default();
    for entry in &manifest.files {
        report.checked += 1;
        let local = vendor_root.join(&entry.path);
        match hash_file(&local)? {
            None => report.findings.push(Finding {
                path: entry.path.clone(),
                problem: Problem::Missing,
            }),
            Some(actual) if actual != entry.expected_local_sha256() => {
                report.findings.push(Finding {
                    path: entry.path.clone(),
                    problem: Problem::HashMismatch {
                        expected: entry.expected_local_sha256().to_string(),
                        actual,
                    },
                });
            }
            Some(_) => {}
        }
        if let Some(patch) = &entry.patch {
            if !vendor_root.join(&patch.diff).exists() {
                report.findings.push(Finding {
                    path: entry.path.clone(),
                    problem: Problem::MissingPatchDiff {
                        diff: patch.diff.clone(),
                    },
                });
            }
        }
    }
    Ok(report)
}

/// Check the reference checkout still carries the pinned content.
pub fn verify_reference(reference_root: &Path, manifest: &Manifest) -> std::io::Result<Report> {
    let mut report = Report::default();
    for entry in &manifest.files {
        report.checked += 1;
        let source = reference_root.join(&entry.path);
        match hash_file(&source)? {
            None => report.findings.push(Finding {
                path: entry.path.clone(),
                problem: Problem::Missing,
            }),
            Some(actual) if actual != entry.sha256 => report.findings.push(Finding {
                path: entry.path.clone(),
                problem: Problem::HashMismatch {
                    expected: entry.sha256.clone(),
                    actual,
                },
            }),
            Some(_) => {}
        }
    }
    Ok(report)
}

/// Anything that stops a sync.
#[derive(Debug)]
pub enum SyncError {
    /// The reference does not match the manifest's pin — nothing was copied.
    ReferenceMismatch(Report),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReferenceMismatch(report) => {
                writeln!(
                    f,
                    "the reference checkout does not match the pinned manifest; nothing was copied:"
                )?;
                for finding in &report.findings {
                    writeln!(f, "  {finding}")?;
                }
                Ok(())
            }
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for SyncError {}

/// Copy every manifest file from the reference into the vendor tree.
///
/// Refuses to run at all unless the reference matches the manifest's pinned
/// hashes. Files with a recorded patch are kept (copying would silently drop
/// the patch) unless `overwrite_patched` is set.
pub fn sync(
    reference_root: &Path,
    vendor_root: &Path,
    manifest: &Manifest,
    overwrite_patched: bool,
) -> Result<SyncReport, SyncError> {
    let report = verify_reference(reference_root, manifest).map_err(|source| SyncError::Io {
        path: reference_root.to_path_buf(),
        source,
    })?;
    if !report.is_clean() {
        return Err(SyncError::ReferenceMismatch(report));
    }

    let mut sync_report = SyncReport::default();
    for entry in &manifest.files {
        if entry.patch.is_some() && !overwrite_patched {
            sync_report.kept_patched.push(entry.path.clone());
            continue;
        }
        let source = reference_root.join(&entry.path);
        let destination = vendor_root.join(&entry.path);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).map_err(|err| SyncError::Io {
                path: parent.to_path_buf(),
                source: err,
            })?;
        }
        std::fs::copy(&source, &destination).map_err(|err| SyncError::Io {
            path: destination.clone(),
            source: err,
        })?;
        sync_report.copied.push(entry.path.clone());
    }
    Ok(sync_report)
}

/// The recorded diff between the reference's file and the vendored one.
///
/// `git diff --no-index` prints the paths it was given, and an absolute
/// reference path would put this machine's layout into a committed diff. So
/// both sides are staged into a scratch directory under stable
/// `reference/<path>` and `vendor/<path>` names and diffed from there, so the
/// header reads `a/reference/<path>` -> `b/vendor/<path>` on every machine.
pub fn unified_diff(
    reference_file: &Path,
    local_file: &Path,
    manifest_path: &str,
) -> Result<String, String> {
    let scratch = std::env::temp_dir().join(format!(
        "ignis-vendor-diff-{}-{}",
        std::process::id(),
        manifest_path.replace(['/', '\\', ':'], "_")
    ));
    let staged_reference = scratch.join("reference").join(manifest_path);
    let staged_local = scratch.join("vendor").join(manifest_path);
    let result = (|| -> Result<String, String> {
        for (source, destination) in [
            (reference_file, &staged_reference),
            (local_file, &staged_local),
        ] {
            let parent = destination.parent().expect("staged path has a parent");
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            std::fs::copy(source, destination)
                .map_err(|e| format!("{}: {e}", source.display()))?;
        }
        // git exits 1 when the files differ, which is the expected case here.
        let output = std::process::Command::new("git")
            .current_dir(&scratch)
            .args(["diff", "--no-index", "--"])
            .arg(format!("reference/{manifest_path}"))
            .arg(format!("vendor/{manifest_path}"))
            .output()
            .map_err(|error| format!("git diff --no-index: {error}"))?;
        match output.status.code() {
            Some(0) | Some(1) => Ok(String::from_utf8_lossy(&output.stdout).into_owned()),
            _ => Err(format!(
                "git diff --no-index failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )),
        }
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

/// Recompute every entry's `sha256` from the reference checkout. Used when the
/// manifest is re-pinned to a new reference commit; it does not touch patched
/// hashes (a re-pin must re-apply and re-record each patch by hand).
pub fn refresh_reference_hashes(
    reference_root: &Path,
    manifest: &mut Manifest,
) -> std::io::Result<Vec<String>> {
    let mut changed = Vec::new();
    for entry in &mut manifest.files {
        let source = reference_root.join(&entry.path);
        let Some(actual) = hash_file(&source)? else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} is not in the reference checkout", entry.path),
            ));
        };
        if actual != entry.sha256 {
            entry.sha256 = actual;
            changed.push(entry.path.clone());
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Patch, Reference, VendoredFile};
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// A throwaway directory under the system temp dir (no dev-dependency).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ignis-vendor-{tag}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
            std::fs::write(&path, contents).expect("write");
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn manifest_of(files: Vec<VendoredFile>) -> Manifest {
        Manifest {
            reference: Reference {
                repo: "gpillon/ninfer".into(),
                branch: "feat/dflash2-local".into(),
                commit: "a00648cb828457986cf5b4b4f712b4cbcd7af0d1".into(),
                default_path: "F:/ai/q38/ninfer".into(),
            },
            vendor_root: "kernel/vendor".into(),
            files,
        }
    }

    fn entry(path: &str, contents: &str) -> VendoredFile {
        VendoredFile {
            path: path.into(),
            sha256: sha256_hex(contents.as_bytes()),
            patch: None,
        }
    }

    fn patch_of(contents: &str) -> Patch {
        Patch {
            diff: "patches/src/core/dtype.h.diff".into(),
            sha256: sha256_hex(contents.as_bytes()),
            reason: "test".into(),
        }
    }

    #[test]
    fn a_byte_identical_vendor_tree_verifies_clean() {
        let vendor = TempDir::new("clean");
        vendor.write("src/core/dtype.h", "#pragma once\n");
        let manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);

        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert!(report.is_clean(), "{:?}", report.findings);
        assert_eq!(report.checked, 1);
    }

    #[test]
    fn an_edited_vendored_file_without_a_patch_fails_verification() {
        let vendor = TempDir::new("edited");
        vendor.write("src/core/dtype.h", "#pragma once\n// local edit\n");
        let manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);

        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].path, "src/core/dtype.h");
        assert!(matches!(
            report.findings[0].problem,
            Problem::HashMismatch { .. }
        ));
    }

    #[test]
    fn a_missing_vendored_file_fails_verification() {
        let vendor = TempDir::new("missing");
        let manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);

        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert_eq!(
            report.findings,
            vec![Finding {
                path: "src/core/dtype.h".into(),
                problem: Problem::Missing,
            }]
        );
    }

    #[test]
    fn a_recorded_patch_makes_the_edited_content_the_expected_content() {
        let vendor = TempDir::new("patched");
        vendor.write("src/core/dtype.h", "#pragma once\n// local edit\n");
        vendor.write("patches/src/core/dtype.h.diff", "--- a\n+++ b\n");
        let mut manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);
        manifest.files[0].patch = Some(patch_of("#pragma once\n// local edit\n"));

        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert!(report.is_clean(), "{:?}", report.findings);
    }

    #[test]
    fn a_patch_whose_diff_is_not_committed_fails_verification() {
        let vendor = TempDir::new("patch-no-diff");
        vendor.write("src/core/dtype.h", "#pragma once\n// local edit\n");
        let mut manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);
        manifest.files[0].patch = Some(patch_of("#pragma once\n// local edit\n"));

        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert_eq!(report.findings.len(), 1);
        assert!(matches!(
            report.findings[0].problem,
            Problem::MissingPatchDiff { .. }
        ));
    }

    #[test]
    fn a_reference_at_a_different_revision_fails_before_anything_is_copied() {
        let reference = TempDir::new("ref-drift");
        reference.write("src/core/dtype.h", "#pragma once\n// upstream moved on\n");
        let vendor = TempDir::new("ref-drift-vendor");
        let manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);

        let error = sync(reference.path(), vendor.path(), &manifest, false)
            .expect_err("a drifted reference must not be copied");
        assert!(matches!(error, SyncError::ReferenceMismatch(_)));
        assert!(
            !vendor.path().join("src/core/dtype.h").exists(),
            "nothing may be copied when the reference does not match the pin"
        );
    }

    #[test]
    fn a_sync_from_the_pinned_reference_reproduces_the_vendor_tree() {
        let reference = TempDir::new("sync-ref");
        reference.write("src/core/dtype.h", "#pragma once\n");
        reference.write("src/ops/common/warp.cuh", "#pragma once\n// warp\n");
        let vendor = TempDir::new("sync-vendor");
        let manifest = manifest_of(vec![
            entry("src/core/dtype.h", "#pragma once\n"),
            entry("src/ops/common/warp.cuh", "#pragma once\n// warp\n"),
        ]);

        let sync_report = sync(reference.path(), vendor.path(), &manifest, false).expect("sync");
        assert_eq!(sync_report.copied.len(), 2);
        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert!(report.is_clean(), "{:?}", report.findings);
    }

    #[test]
    fn a_sync_keeps_a_patched_file_unless_it_is_told_to_overwrite() {
        let reference = TempDir::new("keep-ref");
        reference.write("src/core/dtype.h", "#pragma once\n");
        let vendor = TempDir::new("keep-vendor");
        vendor.write("src/core/dtype.h", "#pragma once\n// local edit\n");
        vendor.write("patches/src/core/dtype.h.diff", "--- a\n+++ b\n");
        let mut manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);
        manifest.files[0].patch = Some(patch_of("#pragma once\n// local edit\n"));

        let kept = sync(reference.path(), vendor.path(), &manifest, false).expect("sync");
        assert_eq!(kept.kept_patched, vec!["src/core/dtype.h".to_string()]);
        assert!(kept.copied.is_empty());
        assert!(
            verify_vendor_tree(vendor.path(), &manifest)
                .expect("verify")
                .is_clean()
        );

        let overwritten = sync(reference.path(), vendor.path(), &manifest, true).expect("sync");
        assert_eq!(overwritten.copied, vec!["src/core/dtype.h".to_string()]);
        let report = verify_vendor_tree(vendor.path(), &manifest).expect("verify");
        assert_eq!(
            report.findings.len(),
            1,
            "the patch is gone, so the hash no longer matches"
        );
    }

    #[test]
    fn a_recorded_diff_names_the_files_by_their_manifest_path_not_by_this_machine() {
        let reference = TempDir::new("diff-ref");
        reference.write("src/core/dtype.h", "one\ntwo\n");
        let vendor = TempDir::new("diff-vendor");
        vendor.write("src/core/dtype.h", "one\ntwo patched\n");

        let diff = unified_diff(
            &reference.path().join("src/core/dtype.h"),
            &vendor.path().join("src/core/dtype.h"),
            "src/core/dtype.h",
        )
        .expect("diff");

        assert!(
            diff.contains("--- a/reference/src/core/dtype.h"),
            "the diff must not carry an absolute path: {diff}"
        );
        assert!(diff.contains("+++ b/vendor/src/core/dtype.h"), "{diff}");
        assert!(diff.contains("-two"), "{diff}");
        assert!(diff.contains("+two patched"), "{diff}");
        assert!(
            !diff.contains(reference.path().to_string_lossy().as_ref()),
            "the diff leaked this machine's reference path: {diff}"
        );
    }

    #[test]
    fn refreshing_hashes_repins_the_manifest_to_the_reference_on_disk() {
        let reference = TempDir::new("refresh");
        reference.write("src/core/dtype.h", "#pragma once\n// upstream moved on\n");
        let mut manifest = manifest_of(vec![entry("src/core/dtype.h", "#pragma once\n")]);

        let changed = refresh_reference_hashes(reference.path(), &mut manifest).expect("refresh");
        assert_eq!(changed, vec!["src/core/dtype.h".to_string()]);
        assert_eq!(
            manifest.files[0].sha256,
            sha256_hex(b"#pragma once\n// upstream moved on\n")
        );
        assert!(
            verify_reference(reference.path(), &manifest)
                .expect("verify")
                .is_clean()
        );
    }
}
