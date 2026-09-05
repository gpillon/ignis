//! The repository's own vendored subtree, checked by the workspace gate.
//!
//! `cargo test` does not build the kernel leaf (`docs/agents/testing.md`), so
//! without this the byte-identical guarantee of ADR 0010 would only be checked
//! when somebody remembered to run `scripts/vendor-ninfer.ps1 verify`. Here it
//! is checked on every run: an edit to a vendored file that is not recorded as
//! a patch turns the workspace red.
//!
//! The reference checkout is a machine-local fixture (it lives outside this
//! repository), so the source-side check runs only where it exists.

use std::path::{Path, PathBuf};

use ignis_vendor::manifest::Manifest;
use ignis_vendor::vendor;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/vendor sits two levels below the repository root")
        .to_path_buf()
}

fn load() -> (Manifest, PathBuf) {
    let root = repository_root();
    let manifest_path = root.join("kernel/vendor/manifest.json");
    let manifest = Manifest::load(&manifest_path)
        .unwrap_or_else(|error| panic!("load {}: {error}", manifest_path.display()));
    (manifest, root.join("kernel/vendor"))
}

#[test]
fn every_vendored_file_is_what_the_manifest_records() {
    let (manifest, vendor_root) = load();
    assert!(
        !manifest.files.is_empty(),
        "the manifest lists no files — the vendored substrate cannot be empty"
    );

    let report = vendor::verify_vendor_tree(&vendor_root, &manifest).expect("verify");
    assert!(
        report.is_clean(),
        "kernel/vendor no longer matches kernel/vendor/manifest.json. A vendored \
         file may not be edited: restore it with `scripts/vendor-ninfer.ps1 sync`, \
         or record the change with `record-patch` (kernel/vendor/VENDOR.md).\n{}",
        report
            .findings
            .iter()
            .map(|finding| format!("  {finding}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_vendored_files_still_match_the_pinned_reference_commit() {
    let (manifest, _) = load();
    let reference = PathBuf::from(&manifest.reference.default_path);
    if !reference.exists() {
        eprintln!(
            "skip: the reference checkout {} is not on this machine",
            reference.display()
        );
        return;
    }

    let report = vendor::verify_reference(&reference, &manifest).expect("verify");
    assert!(
        report.is_clean(),
        "the reference checkout at {} no longer carries the content pinned by \
         reference.commit ({}). Move it back to that commit, or re-pin with \
         `scripts/vendor-ninfer.ps1 repin` (kernel/vendor/VENDOR.md).\n{}",
        reference.display(),
        manifest.reference.commit,
        report
            .findings
            .iter()
            .map(|finding| format!("  {finding}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
