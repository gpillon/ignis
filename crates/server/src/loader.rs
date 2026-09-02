//! The server's artifact loader path (server-03, GitHub #21): the one
//! entry point that loads a `.ninfer` container — open the reader, load
//! the sidecar (ADR 0002's provenance record), run the checksum
//! verification (`checksum::verify`), and only then extract the
//! [`FrontendSet`].
//!
//! A report that is not clean is a **load failure**, not a warning:
//! [`load_artifact_with_report`] refuses the load with a descriptive
//! [`ArtifactError`] that names every flagged object and failed global
//! invariant (the report's own `flagged` / `global_flags` lists — a
//! mismatch is never a panic, per the `checksum` module's contract).
//!
//! The sidecar is discovered next to the artifact
//! (`<artifact>.graft.json`, then `<artifact>.conversion.json` — the
//! reference tooling's naming, both accepted by [`Sidecar::load`]).
//! Without a sidecar there is nothing to verify against, so the load
//! fails rather than silently skipping the check.

use std::path::{Path, PathBuf};

use ignis_artifact::{ArtifactError, ChecksumReport, FrontendSet, Reader, Sidecar, verify};

/// A load/verify failure (the module's error surface).
pub type Result<T> = std::result::Result<T, ArtifactError>;

/// The sidecar record suffixes, in precedence order (the reference
/// tooling's naming — the sidecar's file name is the full artifact file
/// name plus the suffix): the graft record first (the newest provenance
/// record — the graft that produced the artifact), then the conversion
/// record (the original base conversion). Both shapes are accepted by
/// [`Sidecar::load`].
pub const SIDECAR_SUFFIXES: [&str; 2] = [".graft.json", ".conversion.json"];

/// Open `artifact`, load its sidecar `sidecar`, run the checksum
/// verification, and extract the artifact's frontend set.
///
/// Fails (descriptively, never panicking) when:
/// - the container cannot be opened or validated ([`Reader::open`]),
/// - the sidecar cannot be read or parsed ([`Sidecar::load`]),
/// - a recorded object cannot be resolved (`verify`), or
/// - the report is not clean — the load is refused with the flagged
///   objects / failed invariants in the error message.
pub fn load_artifact(artifact: &Path, sidecar: &Path) -> Result<FrontendSet> {
    let reader = Reader::open(artifact)?;
    let sidecar = Sidecar::load(sidecar)?;
    let report = verify(&reader, &sidecar)?;
    load_artifact_with_report(&reader, &report)
}

/// The verification gate (the injectable seam — tests drive it with a
/// mock [`ChecksumReport`]): extract the frontend set when `report` is
/// clean, refuse the load with a descriptive error otherwise.
pub fn load_artifact_with_report(reader: &Reader, report: &ChecksumReport) -> Result<FrontendSet> {
    if !report.is_clean() {
        return Err(ArtifactError::new(describe_report(report)));
    }
    FrontendSet::from_reader(reader)
}

/// The sidecar next to `artifact` (the full artifact file name plus one
/// of [`SIDECAR_SUFFIXES` — the reference tooling's naming, the graft
/// record winning over the conversion record when both exist), when one
/// exists. A descriptive error otherwise (no silent skip — a missing
/// sidecar means there is nothing to verify against).
pub fn find_sidecar(artifact: &Path) -> Result<PathBuf> {
    let name = artifact
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ArtifactError::new("artifact path has no file name"))?;
    let mut looked: Vec<PathBuf> = Vec::new();
    for suffix in SIDECAR_SUFFIXES {
        let candidate = artifact.with_file_name(format!("{name}{suffix}"));
        if candidate.exists() {
            return Ok(candidate);
        }
        looked.push(candidate);
    }
    Err(ArtifactError::new(format!(
        "no sidecar next to {} (looked for {})",
        name,
        looked
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" and ")
    )))
}

/// The human-readable refusal reason for a non-clean report (the failed
/// global invariants first, then each flagged object).
fn describe_report(report: &ChecksumReport) -> String {
    let mut parts: Vec<String> = Vec::new();
    for flag in &report.global_flags {
        parts.push(format!("global: {flag}"));
    }
    for check in report.flagged() {
        parts.push(format!("object '{}': {}", check.name, check.detail));
    }
    format!(
        "checksum verification failed — load refused: {}",
        parts.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use ignis_artifact::fixture::{self, FixtureObject, TempArtifact};
    use ignis_artifact::{ObjectCheck, Outcome};

    /// A minimal word-level tokenizer (the `tokenizers` 0.21 schema, the
    /// same shape the artifact-02 fixture pins).
    const TOKENIZER_JSON: &str = r#"{"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":{"hello":0,"world":1},"unk_token":"hello"}}"#;

    /// A minimal chat template (minijinja — one line per message).
    const TEMPLATE: &str = "{%- for m in messages -%}{{ m.role }}={{ m.content }};{%- endfor -%}";

    /// A fixture artifact carrying the six frontend resources (the same
    /// shape the `artifact_template` tests build).
    fn frontend_fixture(tag: &str) -> (TempArtifact, Vec<u8>) {
        let resources: [(&'static str, &'static [u8]); 6] = [
            ("frontend/tokenizer.json", TOKENIZER_JSON.as_bytes()),
            ("frontend/tokenizer_config.json", b"{}"),
            ("frontend/chat_template.jinja", TEMPLATE.as_bytes()),
            ("frontend/generation_config.json", b"{}"),
            ("frontend/preprocessor_config.json", b"{}"),
            ("frontend/video_preprocessor_config.json", b"{}"),
        ];
        let mut objects = Vec::new();
        let mut payload = Vec::new();
        for (name, bytes) in resources {
            objects.push(FixtureObject::Resource {
                name,
                encoding: "raw-bytes-v1",
                offset: payload.len() as u64,
                bytes: bytes.len() as u64,
            });
            payload.extend_from_slice(bytes);
        }
        let fixture = fixture::write_fixture(&objects, &payload, tag).expect("fixture");
        (fixture, payload)
    }

    /// A sidecar JSON (the shared `graft.json` / `conversion.json`
    /// fields) written next to `path` as `name`.
    fn sidecar_file(path: &Path, name: &str, artifact_bytes: u64, object_count: u64) -> PathBuf {
        let json = format!(
            r#"{{"recipe_id":"test-recipe","artifact":{{"bytes":{artifact_bytes}}},
              "objects":{{"count":{object_count}}}}}"#
        );
        let file = path.with_file_name(format!(
            "{}{name}",
            path.file_name().unwrap().to_str().unwrap()
        ));
        std::fs::write(&file, json).expect("write sidecar");
        file
    }

    /// A report with one mismatched object + one failed global invariant
    /// (the load-failure surface — a mock, not a real `verify` run).
    fn mismatched_report() -> ChecksumReport {
        ChecksumReport {
            global_ok: false,
            global_flags: vec!["file size is 1, sidecar records 2".to_owned()],
            objects: vec![
                ObjectCheck {
                    name: "dflash2/foo".to_owned(),
                    outcome: Outcome::Mismatched,
                    detail: "container stores weight divisor 1.0, sidecar records 2.0".to_owned(),
                },
                ObjectCheck {
                    name: "dflash2/absent".to_owned(),
                    outcome: Outcome::Missing,
                    detail: "recorded in the sidecar, absent from the container (6 objects)"
                        .to_owned(),
                },
            ],
            ..Default::default()
        }
    }

    /// A clean report (no global flags, no flagged objects — matched
    /// records carry no detail).
    fn clean_report() -> ChecksumReport {
        ChecksumReport {
            global_ok: true,
            objects: vec![ObjectCheck {
                name: "dflash2/foo".to_owned(),
                outcome: Outcome::Matched,
                detail: String::new(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn a_mismatched_report_refuses_the_load() {
        // Spec acceptance: inject a mock `ChecksumReport` with a
        // mismatch → the load fails with a descriptive error (no panic).
        let (fixture, _payload) = frontend_fixture("chk-refuse");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let report = mismatched_report();
        assert!(!report.is_clean(), "the mock report must be non-clean");
        let err = load_artifact_with_report(&reader, &report)
            .expect_err("a mismatched report must refuse the load");
        let message = err.to_string();
        // The error names what was flagged (both the mismatch and the
        // missing object) and the failed global invariant.
        assert!(message.contains("load refused"), "{message}");
        assert!(message.contains("dflash2/foo"), "{message}");
        assert!(message.contains("dflash2/absent"), "{message}");
        assert!(message.contains("file size"), "{message}");
    }

    #[test]
    fn a_clean_report_loads_the_frontend_set() {
        // Spec acceptance: inject a clean report → the load proceeds.
        let (fixture, _payload) = frontend_fixture("chk-clean");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let frontend =
            load_artifact_with_report(&reader, &clean_report()).expect("a clean report must load");
        // The set was really built from the fixture's resources (the
        // configs ride along as raw bytes).
        assert_eq!(frontend.tokenizer_config(), b"{}");
        assert_eq!(frontend.generation_config(), b"{}");
    }

    #[test]
    fn the_full_path_verifies_against_the_sidecar_and_loads() {
        // The real path: a sidecar whose `artifact.bytes` /
        // `objects.count` match the container verifies clean, and the
        // frontend set is extracted.
        let (fixture, _payload) = frontend_fixture("chk-full-clean");
        let file_bytes = std::fs::metadata(&fixture.path)
            .expect("fixture metadata")
            .len();
        let sidecar = sidecar_file(&fixture.path, ".graft.json", file_bytes, 6);
        let found = find_sidecar(&fixture.path).expect("the graft sidecar is found");
        assert_eq!(found, sidecar);
        let frontend = load_artifact(&fixture.path, &sidecar).expect("clean load");
        assert_eq!(frontend.tokenizer_config(), b"{}");
    }

    #[test]
    fn the_full_path_refuses_a_sidecar_that_does_not_match() {
        // The real `verify` path: a sidecar with a wrong `artifact.bytes`
        // is not clean (a global invariant fails) and the load is
        // refused with the failed invariant in the error.
        let (fixture, _payload) = frontend_fixture("chk-full-mismatch");
        let file_bytes = std::fs::metadata(&fixture.path)
            .expect("fixture metadata")
            .len();
        let sidecar = sidecar_file(&fixture.path, ".graft.json", file_bytes + 1, 6);
        let err = load_artifact(&fixture.path, &sidecar)
            .expect_err("a mismatched sidecar must refuse the load");
        let message = err.to_string();
        assert!(message.contains("load refused"), "{message}");
        assert!(message.contains("file size"), "{message}");
    }

    #[test]
    fn a_missing_sidecar_is_a_load_failure() {
        // No sidecar on disk: `Sidecar::load` cannot parse what is not
        // there — the error is descriptive (the sidecar path), not a panic.
        let (fixture, _payload) = frontend_fixture("chk-no-sidecar");
        let sidecar = fixture.path.with_file_name(format!(
            "{}.graft.json",
            fixture.path.file_name().unwrap().to_str().unwrap()
        ));
        assert!(!sidecar.exists(), "the test fixture has no sidecar");
        let err = load_artifact(&fixture.path, &sidecar)
            .expect_err("a missing sidecar must fail the load");
        assert!(
            err.to_string().contains(sidecar.to_string_lossy().as_ref()),
            "{err}"
        );
    }

    #[test]
    fn sidecar_discovery_prefers_graft_over_conversion() {
        // `<artifact>.graft.json` wins when both records exist (the
        // graft is the newest provenance record); `<artifact>.conversion.json`
        // is the fallback; neither → a descriptive error (the looked-for
        // candidates are named).
        let (fixture, _payload) = frontend_fixture("chk-discovery");
        let err = find_sidecar(&fixture.path)
            .expect_err("no sidecar has been written yet");
        let message = err.to_string();
        assert!(message.contains(".graft.json"), "{message}");
        assert!(message.contains(".conversion.json"), "{message}");
        let conversion = sidecar_file(&fixture.path, ".conversion.json", 1, 6);
        let found = find_sidecar(&fixture.path)
            .expect("the conversion sidecar alone is found");
        assert_eq!(found.as_path(), conversion.as_path());
        let graft = sidecar_file(&fixture.path, ".graft.json", 1, 6);
        let found = find_sidecar(&fixture.path).expect("the graft sidecar exists");
        assert_eq!(
            found.as_path(),
            graft.as_path(),
            "the graft sidecar takes precedence"
        );
        let _ = std::fs::remove_file(conversion);
        let _ = std::fs::remove_file(graft);
    }
}
