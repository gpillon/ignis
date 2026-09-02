//! Sidecar checksum verification (ticket 03 / GitHub #8).
//!
//! The v2 container deliberately carries no per-tensor digests (the
//! reference container contract: "version 2 requires no checksum, digest,
//! signature, publisher identity, or sidecar"). The
//! `<artifact>.graft.json` / `<artifact>.conversion.json` sidecars are the
//! provenance records ADR 0002 mandates the loader consume; the per-object
//! checksum datum they record is the `local_nvfp4.parents` table — a
//! `weight_scale_divisor` (the FP32 NVFP4 weight divisor the container
//! stores at the tail of each NVFP4 payload, the layout's trailing
//! `weight_divisor` word) plus a `relative_frobenius_error` quality note —
//! alongside the whole-file invariants `artifact.bytes` and
//! `objects.count`.
//!
//! [`verify`] checks every datum the sidecar records: the global
//! invariants, then each `local_nvfp4.parents` entry (the name resolves to
//! an NVFP4 tensor, and — when the sidecar records a `weight_scale_divisor`
//! — the container's trailing FP32 `weight_divisor` value-matches the
//! recorded number).
//! Results are *reported*, never panicking: a [`ChecksumReport`] with a
//! `flagged` list is the load-failure / flagged-tensor surface (ticket 03
//! acceptance).
//!
//! Objects the sidecar does not record (e.g. the 1,259 base tensors of the
//! v2 artifact, whose provenance lives in the v1 conversion sidecar) are
//! *uncovered*, not flagged — coverage is reported, a gap in the record is
//! not a load failure.

use std::path::Path;

use serde_json::Value;

use crate::{block_scale_geometry, fail, Object, NumericFormat, Reader, Result};

// ---------------------------------------------------------------------------
// Sidecar (the graft.json / conversion.json records)
// ---------------------------------------------------------------------------

/// A `local_nvfp4.parents` entry: the sidecar's recorded NVFP4 datum for
/// one tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Record {
    /// The tensor's directory name.
    pub name: String,
    /// The recorded NVFP4 weight divisor as the sidecar's JSON number
    /// (`f64`, kept verbatim; `None` when the sidecar records none — the
    /// weight-only DFlash2 grafts of the v2 artifact, which carry no
    /// activation-quant site). The container stores the divisor as a
    /// 4-byte FP32 word; the verifier casts the stored value to `f64`
    /// exactly and compares values, so a record that is not exactly
    /// FP32-representable can never match.
    pub weight_scale_divisor: Option<f64>,
    /// The sidecar's quantization-quality note (informational: it is not
    /// stored in the container, so it is not verified here).
    pub relative_frobenius_error: f64,
}

/// The sidecar's `grafted_from` block (graft sidecars only): the pre-
/// graft artifact the object set was grafted on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraftedSource {
    /// The grafted-from artifact path (the sidecar's record, verbatim).
    pub path: String,
    /// The grafted-from artifact size the sidecar records.
    pub bytes: u64,
}

/// The parsed sidecar (the shared fields of `graft.json` and
/// `conversion.json`).
#[derive(Debug, Clone)]
pub struct Sidecar {
    /// The sidecar's `recipe_id`.
    pub recipe_id: String,
    /// The sidecar's `grafted_from` block (graft sidecars; absent from
    /// conversion sidecars).
    pub grafted: Option<GraftedSource>,
    /// `artifact.bytes` — the artifact file size the sidecar records.
    pub artifact_bytes: u64,
    /// `objects.count` — the object inventory the sidecar records.
    pub object_count: u64,
    /// `local_nvfp4.parents` (the per-object checksum records, in the
    /// sidecar's key order).
    pub nvfp4_parents: Vec<Nvfp4Record>,
}

impl Sidecar {
    /// Parse a sidecar file (`<artifact>.graft.json` or
    /// `<artifact>.conversion.json`).
    ///
    /// Both records share the `recipe_id`, `artifact.bytes`,
    /// `objects.count`, and `local_nvfp4.parents` fields; `grafted_from`
    /// is graft-only. Missing *required* fields (`recipe_id`,
    /// `artifact.bytes`, `objects.count`) are an error — the verifier
    /// never verifies against a half-parsed record. *Optional* fields fall
    /// back to defaults: a `grafted_from` block is read verbatim (empty
    /// path, zero bytes when absent), and a `local_nvfp4.parents` entry's
    /// `relative_frobenius_error` defaults to 0.0 when absent (the
    /// quality note is informational, never verified).
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            fail(format!("read sidecar {}: {e}", path.display()))
        })?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| fail(format!("invalid sidecar JSON in {}: {e}", path.display())))?;

        let recipe_id = value
            .get("recipe_id")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("sidecar has no string recipe_id"))?
            .to_owned();

        let artifact_bytes = value
            .get("artifact")
            .and_then(|v| v.get("bytes"))
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("sidecar has no artifact.bytes"))?;
        let object_count = value
            .get("objects")
            .and_then(|v| v.get("count"))
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("sidecar has no objects.count"))?;

        let grafted = value.get("grafted_from").map(|g| GraftedSource {
            path: g
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            bytes: g
                .get("bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        });

        let mut nvfp4_parents = Vec::new();
        if let Some(parents) = value.get("local_nvfp4").and_then(|v| v.get("parents")) {
            let parents = parents
                .as_object()
                .ok_or_else(|| fail("sidecar local_nvfp4.parents must be an object"))?;
            for (name, entry) in parents {
                let record = Nvfp4Record {
                    name: name.clone(),
                    weight_scale_divisor: entry
                        .get("weight_scale_divisor")
                        .and_then(Value::as_f64),
                    relative_frobenius_error: entry
                        .get("relative_frobenius_error")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                };
                nvfp4_parents.push(record);
            }
        }

        Ok(Self {
            recipe_id,
            grafted,
            artifact_bytes,
            object_count,
            nvfp4_parents,
        })
    }
}

// ---------------------------------------------------------------------------
// Verification report
// ---------------------------------------------------------------------------

/// The per-object verification outcome (ticket 03: matched / mismatched /
/// missing from the sidecar's record).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Every datum the sidecar records for this object matches the
    /// container (or the sidecar records no per-object datum for it — a
    /// `null` divisor check is presence + NVFP4 format).
    Matched,
    /// A recorded datum conflicts with the container (a stored value
    /// differs, or the object is not the recorded kind).
    Mismatched,
    /// The sidecar records an object the container does not contain.
    Missing,
}

/// The sidecar verification report: the global invariants, the per-object
/// outcomes, and the NVFP4 inventory coverage.
///
/// [`ChecksumReport::is_clean`] is the load-failure surface (ticket 03
/// acceptance: any mismatch is reported — a flagged list, never a panic).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChecksumReport {
    /// The global invariants hold (file size + object inventory).
    pub global_ok: bool,
    /// Global invariant failures (e.g. `file size is 1, sidecar records 2`),
    /// when any.
    pub global_flags: Vec<String>,
    /// The per-object outcomes, one per sidecar `local_nvfp4.parents`
    /// entry (the sidecar's key order); `detail` is empty for matched.
    pub objects: Vec<ObjectCheck>,
    /// NVFP4 tensors in the container.
    pub nvfp4_objects: usize,
    /// The sidecar's `local_nvfp4.parents` records (one per checked
    /// object, matched / mismatched / missing).
    pub nvfp4_records: usize,
}

/// One object's verification outcome (the per-object result of
/// [`verify`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectCheck {
    /// The sidecar's recorded object name.
    pub name: String,
    /// The outcome (matched / mismatched / missing).
    pub outcome: Outcome,
    /// The human-readable reason (empty for a matched object).
    pub detail: String,
}

impl ChecksumReport {
    /// The flagged objects (mismatched + missing) — the reported load-
    /// failure surface.
    pub fn flagged(&self) -> Vec<&ObjectCheck> {
        self.objects
            .iter()
            .filter(|c| c.outcome != Outcome::Matched)
            .collect()
    }

    /// The cleanly verified objects.
    pub fn matched(&self) -> Vec<&ObjectCheck> {
        self.objects
            .iter()
            .filter(|c| c.outcome == Outcome::Matched)
            .collect()
    }

    /// The load-failure check: any flagged object or failed global
    /// invariant means the load must be reported as a failure (not
    /// silently continued).
    pub fn is_clean(&self) -> bool {
        self.global_ok && self.flagged().is_empty()
    }

    /// The NVFP4 tensors the sidecar does not record (uncovered, not
    /// flagged — the record's coverage gap, reported for the record).
    ///
    /// Coverage heuristic: the container's NVFP4 tensor count minus the
    /// sidecar's `local_nvfp4.parents` record count (records that resolve
    /// to a `Missing` object are still counted, so this is a lower bound
    /// on the true uncovered count).
    pub fn nvfp4_uncovered(&self) -> usize {
        self.nvfp4_objects.saturating_sub(self.nvfp4_records)
    }
}

// ---------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------

/// Verify a reader's container against its sidecar.
///
/// Never panics on a mismatch — every conflict lands in the report's
/// `flagged` list (mismatched / missing objects) or `global_flags`
/// (whole-file invariants). Errors only when the sidecar's recorded object
/// cannot be resolved (an unknown span or an invalid NVFP4 geometry).
pub fn verify(reader: &Reader, sidecar: &Sidecar) -> Result<ChecksumReport> {
    let mut report = ChecksumReport::default();

    // --- global invariants (the sidecar's whole-file record) --------------
    if reader.file_bytes() != sidecar.artifact_bytes {
        report.global_flags.push(format!(
            "file size is {}, sidecar records {}",
            reader.file_bytes(),
            sidecar.artifact_bytes
        ));
    }
    let object_count = reader.objects().len() as u64;
    if object_count != sidecar.object_count {
        report.global_flags.push(format!(
            "object count is {}, sidecar records {}",
            object_count, sidecar.object_count
        ));
    }
    report.global_ok = report.global_flags.is_empty();

    // --- per-object: every recorded local_nvfp4 parent --------------------
    for record in &sidecar.nvfp4_parents {
        let (outcome, detail) = check_nvfp4_record(reader, record)?;
        report.objects.push(ObjectCheck {
            name: record.name.clone(),
            outcome,
            detail,
        });
    }

    // --- NVFP4 inventory coverage ------------------------------------------
    report.nvfp4_objects = reader
        .objects()
        .iter()
        .filter(|o| matches!(o, Object::Tensor(t) if t.format == NumericFormat::Nvfp4))
        .count();
    report.nvfp4_records = report.objects.len();

    Ok(report)
}

/// Check one `local_nvfp4.parents` record: the name resolves to an NVFP4
/// tensor, and (when the sidecar records a `weight_scale_divisor`) the
/// container's trailing FP32 `weight_divisor` bit-matches it.
///
/// Never panics: a missing object or a conflicting datum is a returned
/// outcome, not an error. Errors only when a resolved object's payload span
/// or NVFP4 geometry is invalid (a container-integrity fault, not a
/// sidecar mismatch).
fn check_nvfp4_record(
    reader: &Reader,
    record: &Nvfp4Record,
) -> Result<(Outcome, String)> {
    let object = match reader.find(&record.name) {
        Some(o) => o,
        None => {
            return Ok((
                Outcome::Missing,
                format!(
                    "recorded in the sidecar, absent from the container ({} objects)",
                    reader.objects().len()
                ),
            ))
        }
    };

    // A `local_nvfp4.parents` record must name an NVFP4 tensor.
    match object {
        Object::Tensor(tensor) => {
            if tensor.format != NumericFormat::Nvfp4 {
                return Ok((
                    Outcome::Mismatched,
                    format!(
                        "sidecar records an NVFP4 parent, container stores {format}",
                        format = tensor.format.name()
                    ),
                ));
            }
            // No recorded divisor: the object's presence + NVFP4 format are
            // the check (the v2 DFlash2 grafts record `weight_scale_divisor:
            // null` — weight-only quantization, no activation-quant site).
            let Some(expected) = record.weight_scale_divisor else {
                return Ok((Outcome::Matched, String::new()));
            };
            // The container's trailing FP32 weight divisor (the blockscale
            // layout's `weight_divisor` word). The stored FP32 is promoted
            // to `f64` exactly and value-compared against the sidecar's
            // JSON number (kept verbatim, so a record that is not exactly
            // FP32-representable can never match).
            let geometry = block_scale_geometry(tensor.format, &tensor.shape)?;
            let span = reader.payload_at(object)?;
            let start = geometry.weight_divisor_offset as usize;
            let stored = f32::from_le_bytes(span.data[start..start + 4].try_into().unwrap());
            if f64::from(stored) != expected {
                return Ok((
                    Outcome::Mismatched,
                    format!(
                        "container stores weight divisor {stored}, sidecar records {expected}"
                    ),
                ));
            }
            Ok((Outcome::Matched, String::new()))
        }
        Object::Resource(_) => Ok((
            Outcome::Mismatched,
            "sidecar records an NVFP4 parent, container stores a resource".to_owned(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU-only; synthetic fixture sidecar)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{write_fixture, FixtureObject};
    use std::path::PathBuf;

    /// A one-NVFP4-tensor fixture (128x64 blockscale: 4612 encoded bytes —
    /// 4096 code plane + 512 scale plane + 4-byte weight divisor).
    ///
    /// Returns the fixture, the reader, and the weight divisor the fixture
    /// stores (independent of the sidecar, so the matching test is not
    /// tautological).
    fn nvfp4_fixture(tag: &str) -> (crate::fixture::TempArtifact, Reader, f32) {
        let objects = vec![FixtureObject::Tensor {
            name: "w/nvfp4",
            shape: vec![128, 64],
            format: "NVFP4",
            layout: "blockscale-k16-m128x4-v1",
            offset: 0,
            bytes: 4612,
        }];
        let geometry = crate::block_scale_geometry(NumericFormat::Nvfp4, &[128, 64]).unwrap();
        let mut payload = vec![0u8; geometry.encoded_bytes as usize];
        payload[..geometry.code_plane_bytes as usize].fill(0x42);
        payload[geometry.scale_plane_offset as usize
            ..(geometry.scale_plane_offset as usize + geometry.scale_plane_bytes as usize)]
            .fill(0x77);
        let divisor = 2688.0f32;
        payload[geometry.weight_divisor_offset as usize..]
            .copy_from_slice(&divisor.to_le_bytes());
        let fixture = write_fixture(&objects, &payload, tag).expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        (fixture, reader, divisor)
    }

    /// A sidecar JSON matching the real sidecar schema (the shared fields of
    /// `graft.json` / `conversion.json`), written to `<base>.sidecar.json`.
    fn sidecar_file(
        base: &Path,
        object_count: u64,
        artifact_bytes: u64,
        parents: &[(&str, Option<f32>)],
    ) -> PathBuf {
        let mut parents_json = String::from("{");
        for (i, (name, divisor)) in parents.iter().enumerate() {
            if i > 0 {
                parents_json.push(',');
            }
            let divisor = match divisor {
                Some(d) => format!("\"weight_scale_divisor\":{d}"),
                None => "\"weight_scale_divisor\":null".to_owned(),
            };
            parents_json.push_str(&format!(
                "\"{name}\":{{{divisor},\"relative_frobenius_error\":0.095}}"
            ));
        }
        parents_json.push('}');
        let json = format!(
            r#"{{"recipe_id":"test-recipe","grafted_from":{{"path":"x.ninfer","bytes":123}},
              "artifact":{{"bytes":{artifact_bytes}}},
              "objects":{{"count":{object_count},"grafted":{count}}},
              "local_nvfp4":{{"encoder_profile":"NVFP4_MAXABS_DIVISOR_RNE_V1","parents":{parents_json}}}}}"#,
            count = parents.len(),
        );
        let path = match base.to_str() {
            Some(s) => base
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("{s}.sidecar.json")),
            None => base.to_path_buf(),
        };
        std::fs::write(&path, json).expect("write sidecar");
        path
    }

    /// A `conversion.json`-shaped sidecar (no `grafted_from` block — the
    /// converter sidecar records its grafted-from artifact as `artifact`,
    /// which the shared `artifact.bytes` field already carries). Written to
    /// `<base>.conversion-sidecar.json`.
    fn conversion_sidecar_file(
        base: &Path,
        object_count: u64,
        artifact_bytes: u64,
        parents: &[(&str, Option<f32>)],
    ) -> PathBuf {
        let mut parents_json = String::from("{");
        for (i, (name, divisor)) in parents.iter().enumerate() {
            if i > 0 {
                parents_json.push(',');
            }
            let divisor = match divisor {
                Some(d) => format!("\"weight_scale_divisor\":{d}"),
                None => "\"weight_scale_divisor\":null".to_owned(),
            };
            parents_json.push_str(&format!(
                "\"{name}\":{{{divisor},\"relative_frobenius_error\":0.095}}"
            ));
        }
        parents_json.push('}');
        // No `grafted_from` block (the conversion sidecar's shape).
        let json = format!(
            r#"{{"recipe_id":"test-recipe",
              "artifact":{{"bytes":{artifact_bytes}}},
              "objects":{{"count":{object_count}}},
              "local_nvfp4":{{"encoder_profile":"NVFP4_MAXABS_DIVISOR_RNE_V1","parents":{parents_json}}}}}"#
        );
        let path = match base.to_str() {
            Some(s) => base
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("{s}.conversion-sidecar.json")),
            None => base.to_path_buf(),
        };
        std::fs::write(&path, json).expect("write sidecar");
        path
    }

    #[test]
    fn sidecar_matching_records_verify_clean() {
        let (fixture, reader, divisor) = nvfp4_fixture("chk-match");
        let file_bytes = std::fs::metadata(&fixture.path).expect("meta").len() as u64;
        let sidecar_path = sidecar_file(&fixture.path, 1, file_bytes, &[("w/nvfp4", Some(divisor))]);
        let sidecar = Sidecar::load(&sidecar_path).expect("load sidecar");
        assert_eq!(sidecar.nvfp4_parents.len(), 1);
        assert!(
            sidecar.grafted.is_some(),
            "the graft sidecar's grafted_from block parsed"
        );
        let report = verify(&reader, &sidecar).expect("verify");
        assert!(report.global_ok, "{report:?}");
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.objects.len(), 1);
        assert_eq!(report.objects[0].outcome, Outcome::Matched);
        assert_eq!(report.matched().len(), 1);
        assert_eq!(report.flagged(), Vec::<&ObjectCheck>::new());
        assert_eq!(report.nvfp4_objects, 1);
        assert_eq!(report.nvfp4_records, 1);
        assert_eq!(report.nvfp4_uncovered(), 0);
        let _ = std::fs::remove_file(sidecar_path);
    }

    #[test]
    fn conversion_sidecar_without_grafted_from_parses_and_verifies() {
        // The conversion sidecar has no `grafted_from` block (unlike the
        // graft sidecar); `Sidecar::load` must parse it, leave `grafted`
        // as `None`, and `verify` must run the same checks.
        let (fixture, reader, divisor) = nvfp4_fixture("chk-conversion");
        let file_bytes = std::fs::metadata(&fixture.path).expect("meta").len() as u64;
        let sidecar_path =
            conversion_sidecar_file(&fixture.path, 1, file_bytes, &[("w/nvfp4", Some(divisor))]);
        let sidecar = Sidecar::load(&sidecar_path).expect("load conversion sidecar");
        assert!(
            sidecar.grafted.is_none(),
            "a conversion sidecar has no grafted_from block"
        );
        let report = verify(&reader, &sidecar).expect("verify");
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.matched().len(), 1);
        let _ = std::fs::remove_file(sidecar_path);
    }

    #[test]
    fn sidecar_mismatched_divisor_is_flagged() {
        let (fixture, reader, divisor) = nvfp4_fixture("chk-mismatch");
        let file_bytes = std::fs::metadata(&fixture.path).expect("meta").len() as u64;
        // The sidecar records a different weight divisor than the
        // container stores (an independent value, so the test is not
        // tautological).
        let sidecar_path =
            sidecar_file(&fixture.path, 1, file_bytes, &[("w/nvfp4", Some(divisor + 1.0))]);
        let sidecar = Sidecar::load(&sidecar_path).expect("load sidecar");
        let report = verify(&reader, &sidecar).expect("verify");
        assert!(report.global_ok, "{report:?}");
        assert!(!report.is_clean(), "{report:?}");
        assert_eq!(report.matched(), Vec::<&ObjectCheck>::new());
        let flagged = report.flagged();
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].name, "w/nvfp4");
        assert_eq!(flagged[0].outcome, Outcome::Mismatched);
        assert!(
            flagged[0].detail.contains("weight divisor"),
            "{:?}",
            flagged[0].detail
        );
        let _ = std::fs::remove_file(sidecar_path);
    }

    #[test]
    fn sidecar_missing_object_is_flagged() {
        let (fixture, reader, _) = nvfp4_fixture("chk-missing");
        let file_bytes = std::fs::metadata(&fixture.path).expect("meta").len() as u64;
        // The sidecar records a second NVFP4 parent the container does not
        // hold (and records the object count it implies: 2, not 1).
        let sidecar_path = sidecar_file(
            &fixture.path,
            2,
            file_bytes,
            &[("w/nvfp4", None), ("w/absent", None)],
        );
        let sidecar = Sidecar::load(&sidecar_path).expect("load sidecar");
        let report = verify(&reader, &sidecar).expect("verify");
        // The missing object is flagged (not a panic).
        let flagged = report.flagged();
        assert!(
            flagged.iter().any(|c| c.name == "w/absent"),
            "{report:?}"
        );
        assert_eq!(
            flagged.iter().find(|c| c.name == "w/absent").unwrap().outcome,
            Outcome::Missing
        );
        // The recorded-but-present object still verifies (its `null`
        // divisor check is presence + NVFP4 format).
        assert!(
            report.matched().iter().any(|c| c.name == "w/nvfp4"),
            "{report:?}"
        );
        // The object-count invariant also fails (sidecar records 2, the
        // container holds 1).
        assert!(!report.global_ok, "{report:?}");
        assert!(
            report.global_flags.iter().any(|f| f.contains("object count")),
            "{report:?}"
        );
        assert!(!report.is_clean(), "{report:?}");
        let _ = std::fs::remove_file(sidecar_path);
    }

    #[test]
    fn sidecar_file_size_mismatch_is_flagged() {
        let (fixture, reader, divisor) = nvfp4_fixture("chk-size");
        let sidecar_path = sidecar_file(&fixture.path, 1, 999_999, &[("w/nvfp4", Some(divisor))]);
        let sidecar = Sidecar::load(&sidecar_path).expect("load sidecar");
        let report = verify(&reader, &sidecar).expect("verify");
        assert!(!report.global_ok, "{report:?}");
        assert!(
            report.global_flags.iter().any(|f| f.contains("file size")),
            "{report:?}"
        );
        // The per-object checks are independent of the whole-file
        // invariants (the recorded object still verifies).
        assert!(
            report.matched().iter().any(|c| c.name == "w/nvfp4"),
            "{report:?}"
        );
        assert!(!report.is_clean(), "{report:?}");
        let _ = std::fs::remove_file(sidecar_path);
    }

    #[test]
    fn sidecar_null_divisor_is_a_presence_check() {
        let (fixture, reader, _) = nvfp4_fixture("chk-null-divisor");
        let file_bytes = std::fs::metadata(&fixture.path).expect("meta").len() as u64;
        // The v2 graft sidecar records the DFlash2 parents with
        // `weight_scale_divisor: null` — the check is presence + NVFP4
        // format, not a divisor comparison.
        let sidecar_path = sidecar_file(&fixture.path, 1, file_bytes, &[("w/nvfp4", None)]);
        let sidecar = Sidecar::load(&sidecar_path).expect("load sidecar");
        let report = verify(&reader, &sidecar).expect("verify");
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.matched().len(), 1);
        assert_eq!(report.flagged(), Vec::<&ObjectCheck>::new());
        let _ = std::fs::remove_file(sidecar_path);
    }
}