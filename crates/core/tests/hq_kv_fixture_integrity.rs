//! Cheap, non-GPU integrity check for the committed hq-e8-2b KV row fixture
//! (P4-03, GitHub #119): `kernel/tests/fixtures/hq_kv_rows_27b.bin` must
//! still hash to the SHA-256 recorded in its own
//! `hq_kv_rows_27b.provenance.json` sidecar. This is what `cargo test`
//! (default suite, no GPU, no `#[ignore]`) can catch on its own between
//! recordings of the GPU-only capture test
//! (`crates/core/tests/hq_kv_fixture_capture_gpu.rs`) -- a corrupted fixture
//! or a hand-edit that drifted from its provenance turns the workspace red
//! the same way an unpatched vendored-file edit does (ADR 0010).

use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

#[test]
fn hq_kv_fixture_bin_matches_its_provenance_sha256() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernel/tests/fixtures");
    let bin_path = dir.join("hq_kv_rows_27b.bin");
    let provenance_path = dir.join("hq_kv_rows_27b.provenance.json");

    let bin_bytes =
        fs::read(&bin_path).unwrap_or_else(|e| panic!("read {}: {e}", bin_path.display()));
    let provenance_text = fs::read_to_string(&provenance_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", provenance_path.display()));
    let provenance: serde_json::Value = serde_json::from_str(&provenance_text)
        .unwrap_or_else(|e| panic!("parse {}: {e}", provenance_path.display()));
    let recorded_sha256 = provenance["bin_sha256"].as_str().unwrap_or_else(|| {
        panic!("{} has no string \"bin_sha256\" field", provenance_path.display())
    });

    let mut hasher = Sha256::new();
    hasher.update(&bin_bytes);
    let actual_sha256 = format!("{:x}", hasher.finalize());

    assert_eq!(
        actual_sha256, recorded_sha256,
        "kernel/tests/fixtures/hq_kv_rows_27b.bin does not match the SHA-256 recorded in \
         hq_kv_rows_27b.provenance.json -- the fixture was corrupted or hand-edited; re-run the \
         GPU capture test (hq_kv_fixture_capture_gpu.rs) to regenerate both files together, \
         never one alone"
    );
}
