//! The vendoring manifest: what was copied from the reference, from which
//! commit, and at which content hash (ADR 0010).
//!
//! One entry per vendored file. `sha256` is the file's hash **in the
//! reference at the pinned commit**; a vendored file must be byte-identical
//! to it unless the entry records a patch, in which case the local file must
//! hash to `patch.sha256` and the diff must be committed next to the manifest.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The reference checkout a manifest is pinned to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reference {
    /// The reference fork, e.g. `gpillon/ninfer`.
    pub repo: String,
    /// The branch the pinned commit lives on.
    pub branch: String,
    /// The pinned commit — the only revision these hashes describe.
    pub commit: String,
    /// Where that checkout usually sits on this machine. Every command takes
    /// `--reference` to override it; nothing fails if the path is absent.
    pub default_path: String,
}

/// A local change to a vendored file, recorded as a committed diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    /// The diff's path, relative to the vendor root.
    pub diff: String,
    /// The hash of the vendored file *after* the patch — what the leaf builds.
    pub sha256: String,
    /// Why the patch exists. A vendored file may not be edited without one.
    pub reason: String,
}

/// One vendored file. The path is shared by the reference and the leaf so a
/// `diff` against the source stays trivial (ADR 0010).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VendoredFile {
    /// Path relative to the reference root *and* to the vendor root.
    pub path: String,
    /// The file's hash in the reference at the pinned commit.
    pub sha256: String,
    /// Present only when the leaf's copy deliberately differs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch: Option<Patch>,
}

impl VendoredFile {
    /// The hash the file in the vendor tree must have: the reference's hash,
    /// or the patched hash when a patch is recorded.
    pub fn expected_local_sha256(&self) -> &str {
        match &self.patch {
            Some(patch) => &patch.sha256,
            None => &self.sha256,
        }
    }
}

/// The manifest as it lives on disk (`kernel/vendor/manifest.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub reference: Reference,
    /// The vendor root, relative to the repository root.
    pub vendor_root: String,
    pub files: Vec<VendoredFile>,
}

/// Anything that stops a manifest from being read or written.
#[derive(Debug)]
pub enum ManifestError {
    Io { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, source: serde_json::Error },
    DuplicatePath(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Parse { path, source } => write!(f, "{}: invalid manifest: {source}", path.display()),
            Self::DuplicatePath(path) => write!(f, "manifest lists {path} more than once"),
        }
    }
}

impl std::error::Error for ManifestError {}

impl Manifest {
    /// Read a manifest, rejecting a duplicated path (two entries for one file
    /// would make verification order-dependent).
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let manifest: Self = serde_json::from_str(&text).map_err(|source| ManifestError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        manifest.check_unique_paths()?;
        Ok(manifest)
    }

    /// Write the manifest back, sorted by path so a re-run produces a diff
    /// that shows only what actually changed.
    pub fn store(&self, path: &Path) -> Result<(), ManifestError> {
        self.check_unique_paths()?;
        let mut sorted = self.clone();
        sorted.files.sort_by(|a, b| a.path.cmp(&b.path));
        let mut text = serde_json::to_string_pretty(&sorted).map_err(|source| ManifestError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        text.push('\n');
        std::fs::write(path, text).map_err(|source| ManifestError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn check_unique_paths(&self) -> Result<(), ManifestError> {
        let mut seen: Vec<&str> = self.files.iter().map(|f| f.path.as_str()).collect();
        seen.sort_unstable();
        for pair in seen.windows(2) {
            if pair[0] == pair[1] {
                return Err(ManifestError::DuplicatePath(pair[0].to_string()));
            }
        }
        Ok(())
    }

    pub fn file(&self, path: &str) -> Option<&VendoredFile> {
        self.files.iter().find(|f| f.path == path)
    }

    pub fn file_mut(&mut self, path: &str) -> Option<&mut VendoredFile> {
        self.files.iter_mut().find(|f| f.path == path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_json() -> &'static str {
        r#"{
  "reference": {
    "repo": "gpillon/ninfer",
    "branch": "feat/dflash2-local",
    "commit": "a00648cb828457986cf5b4b4f712b4cbcd7af0d1",
    "default_path": "F:/ai/q38/ninfer"
  },
  "vendor_root": "kernel/vendor",
  "files": [
    { "path": "src/core/dtype.h", "sha256": "aa" },
    {
      "path": "src/core/tensor.h",
      "sha256": "bb",
      "patch": { "diff": "patches/src/core/tensor.h.diff", "sha256": "cc", "reason": "why" }
    }
  ]
}"#
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let parsed: Manifest = serde_json::from_str(manifest_json()).expect("parse");
        assert_eq!(parsed.reference.commit, "a00648cb828457986cf5b4b4f712b4cbcd7af0d1");
        assert_eq!(parsed.vendor_root, "kernel/vendor");
        assert_eq!(parsed.files.len(), 2);

        let text = serde_json::to_string(&parsed).expect("serialize");
        let again: Manifest = serde_json::from_str(&text).expect("re-parse");
        assert_eq!(parsed, again);
    }

    #[test]
    fn an_unpatched_file_is_expected_to_be_byte_identical_to_the_reference() {
        let parsed: Manifest = serde_json::from_str(manifest_json()).expect("parse");
        let entry = parsed.file("src/core/dtype.h").expect("entry");
        assert_eq!(entry.expected_local_sha256(), "aa");
    }

    #[test]
    fn a_patched_file_is_expected_to_match_the_recorded_patched_hash() {
        let parsed: Manifest = serde_json::from_str(manifest_json()).expect("parse");
        let entry = parsed.file("src/core/tensor.h").expect("entry");
        assert_eq!(entry.sha256, "bb");
        assert_eq!(entry.expected_local_sha256(), "cc");
    }

    #[test]
    fn a_duplicated_path_is_rejected() {
        let text = r#"{
          "reference": { "repo": "r", "branch": "b", "commit": "c", "default_path": "p" },
          "vendor_root": "kernel/vendor",
          "files": [
            { "path": "src/core/dtype.h", "sha256": "aa" },
            { "path": "src/core/dtype.h", "sha256": "bb" }
          ]
        }"#;
        let parsed: Manifest = serde_json::from_str(text).expect("parse");
        let err = parsed.check_unique_paths().expect_err("duplicate must be rejected");
        assert!(matches!(err, ManifestError::DuplicatePath(ref p) if p == "src/core/dtype.h"));
    }
}
