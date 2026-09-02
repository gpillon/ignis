//! Integration test against the real Qwen 3.8-27B `nvfp4full` v2 artifact
//! (the 1,325-object ~19.4 GB container) and its `graft.json` sidecar
//! (ticket 03 / GitHub #8). Skips gracefully when the files are not at
//! their machine-local paths.
//!
//! The v2 container carries no per-tensor digests (the v2 contract:
//! "requires no checksum, digest, signature, or sidecar"); the sidecar's
//! checksum record is its `local_nvfp4.parents` table (the FP32 NVFP4
//! weight divisors — `null` for the weight-only DFlash2 grafts) plus the
//! whole-file invariants `artifact.bytes` and `objects.count`. This test
//! verifies every datum the sidecar records: the global invariants hold,
//! and each of the 34 recorded parents resolves to an NVFP4 tensor in the
//! container (CPU-only — payload reads page-fault through the mapping).

use std::path::Path;

use ignis_artifact::{verify, Reader, Sidecar};

/// The fork-local model cache (the artifact the running `ninfer-serve`
/// loads).
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The v2 graft sidecar (the v1 artifact's `conversion.json` records the
/// base-tensor provenance; the graft sidecar records the v2 graft).
const SIDECAR: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer.graft.json";

/// Open the real artifact + sidecar, or `None` (with a skip note) when
/// either file is absent (the machine-local skip convention of
/// `real_artifact.rs`).
fn open_or_skip() -> Option<(Reader, Sidecar)> {
    for path in [ARTIFACT, SIDECAR] {
        if !Path::new(path).exists() {
            eprintln!("skip: {path} does not exist");
            return None;
        }
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let sidecar = Sidecar::load(Path::new(SIDECAR)).expect("load sidecar");
    Some((reader, sidecar))
}

#[test]
fn real_nvfp4full_sidecar_verification() {
    let (reader, sidecar) = match open_or_skip() {
        Some(pair) => pair,
        None => return,
    };

    // The sidecar's own records (independent of the container): the v2
    // recipe, the 1,325-object inventory, the whole-file size, and the 34
    // locally-quantized NVFP4 parents (weight-only: no recorded divisors).
    assert_eq!(sidecar.recipe_id, "qwen3_8_27b_nvfp4full-v2");
    assert_eq!(sidecar.object_count, 1325);
    assert_eq!(sidecar.artifact_bytes, 19_406_942_468);
    assert_eq!(sidecar.nvfp4_parents.len(), 34);
    assert!(
        sidecar.nvfp4_parents.iter().all(|r| r.weight_scale_divisor.is_none()),
        "the v2 DFlash2 grafts are weight-only (no recorded divisors)"
    );
    // The graft was made from the v1 artifact (the 18.3 GB pre-graft file).
    assert!(
        sidecar
            .grafted
            .as_ref()
            .is_some_and(|g| g.bytes == 18_324_059_648),
        "grafted_from records the v1 artifact size"
    );

    // Verify: the global invariants hold (file size + object inventory) and
    // every recorded parent resolves to an NVFP4 tensor in the container.
    let report = verify(&reader, &sidecar).expect("verify");
    assert!(report.global_ok, "{report:?}");
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.matched().len(), 34);
    assert!(
        report.flagged().is_empty(),
        "no flagged objects: {:?}",
        report.flagged()
    );

    // NVFP4 inventory coverage: 281 NVFP4 tensors in the container (247
    // base + 34 DFlash2); the sidecar records the 34 grafts — the base
    // tensors' provenance is inherited from the v1 conversion sidecar
    // (uncovered here, not flagged).
    assert_eq!(report.nvfp4_objects, 281);
    assert_eq!(report.nvfp4_records, 34);
    assert_eq!(report.nvfp4_uncovered(), 247);
}