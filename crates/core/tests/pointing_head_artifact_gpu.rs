//! The **pointing head** is keyed to the artifact (spec 13 acceptance 3,
//! GitHub #260): the served artifact's content hash must be in the
//! calibration table, and this test fails the day it is not.
//!
//! The head is a calibrated constant — L39.h10 was chosen on labelled scenes
//! by cross-validation on *this* artifact — so a new artifact that shipped
//! without a recalibration would have `/v1/decide` fall back to the chain
//! silently, or, worse, read a head chosen for a different model if the
//! table were keyed loosely. The table is keyed by the full hash; this test
//! is what notices the served artifact moving under it.
//!
//! It touches no GPU — the hash is of the container's declared structure —
//! but it lives in the explicit GPU profile with the artifact it reads
//! (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing artifact
//! is a **skip**; under the profile it is a **hard failure**.

use std::path::Path;

use ignis_artifact::Reader;
use ignis_core::gpu_profile;
use ignis_core::pointing::{PointingHead, calibrated_artifacts, calibrated_head};
use ignis_core::ArtifactHash;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

#[test]
#[ignore = "needs the served artifact on disk; run via scripts/gpu-profile.ps1 (ADR 0006)"]
fn the_served_artifact_has_a_calibrated_pointing_head() {
    if !Path::new(ARTIFACT).exists() {
        gpu_profile::skip_or_fail(&format!("artifact not found at {ARTIFACT}"));
        return;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("the served artifact opens");
    let hash = ArtifactHash::from_bytes(reader.content_hash());
    let known: Vec<String> = calibrated_artifacts().map(|h| hex(h.as_bytes())).collect();
    let head = calibrated_head(hash).unwrap_or_else(|| {
        panic!(
            "the served artifact {ARTIFACT} (content hash {}) has no calibrated pointing head, \
             so `/v1/decide` would answer every `point` with the digit chain. The table knows \
             {known:?}. Recalibrate before shipping this artifact: the procedure is \
             tools/pointing-scenes/README.md and docs/specs/decide/13-point-by-attention-head.md \
             § Further Notes (run \
             crates/server/tests/attention_head_point_gpu.rs over the labelled scene sets with \
             every GQA layer armed, choose the head by cross-validation on the region rule's \
             inside rate, add the hash to crates/core/src/pointing.rs, re-run spec 13's \
             acceptance).",
            hex(hash.as_bytes())
        )
    });
    assert_eq!(
        head,
        PointingHead {
            gqa_ordinal: 9,
            query_head: 10
        },
        "the served artifact's head is the one the findings chose, L39.h10"
    );
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
