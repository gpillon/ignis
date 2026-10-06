//! The converter's own trees pack and bind (spec flash-next/01: the reader
//! parses what the converter's writer produces).
//!
//! - `tests/fixtures/flash_next_work/` is a reduced work tree the converter
//!   wrote from transformers' Qwen4Exp modules at
//!   [`FlashNextGeometry::fixture`] through its real writers
//!   (`convert.py fixture --reduced`); committed, it runs everywhere.
//! - Machine-local: `convert.py fixture` itself is run with the study's
//!   Python environment (CPU only; skipped when that is absent). That tree
//!   carries real-size expert records but only three non-expert tensors per
//!   layer, so it is packed and read, not bound.

use std::path::{Path, PathBuf};
use std::process::Command;

use ignis_artifact::flash_next::{self, expert_name, FlashNextGeometry, Projection, TrellisK};
use ignis_artifact::packer::{pack, PackOptions, PackOutcome};
use ignis_artifact::{Object, Reader, StorageLayout};

const PYTHON: &str = "F:/ai/ngram-venv/Scripts/python.exe";

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_converters_reduced_tree_packs_and_binds() {
    let work = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flash_next_work");
    let scratch = Scratch(std::env::temp_dir().join(format!("ignis-converter-reduced-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&scratch.0);
    std::fs::create_dir_all(&scratch.0).unwrap();
    let geometry = FlashNextGeometry::fixture();
    let artifact = scratch.0.join("reduced-v2.ninfer");
    let mut options = PackOptions::new(work.clone(), artifact.clone(), geometry.clone());
    // The committed tree is read, never consumed.
    options.keep_work = true;
    options.accept_status = vec!["fixture".into()];
    options.header_bytes = 1 << 20;
    let outcome = pack(&options, &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { converter_merged: true, .. }), "{outcome:?}");
    assert!(work.join("layers/L00/experts.bin").exists());

    let reader = Reader::open(&artifact).unwrap();
    let bound = flash_next::bind(&reader, &geometry).expect("the converter's tree binds");
    assert_eq!(bound.plan.object_count, reader.objects().len());
    assert_eq!(bound.experts.class_counts().len(), 8, "all eight K classes");

    // The K map the converter recorded is the one the container carries.
    let record = read_json(&work.join("converter.json"));
    for entry in record["k_map"]["layers"].as_array().unwrap() {
        let layer = entry["layer"].as_u64().unwrap() as usize;
        for (key, projection) in [("gu", Projection::GateUp), ("dn", Projection::Down)] {
            for (expert, k2) in entry[key].as_array().unwrap().iter().enumerate() {
                let k = bound.experts.get(layer, expert, projection).unwrap().k;
                assert_eq!(u64::from(k.k2()), k2.as_u64().unwrap(), "layer {layer} expert {expert} {key}");
            }
        }
    }
    // Each layer's expert bytes are its experts.bin, as the converter's DONE
    // hashed it.
    let experts_bin: Vec<serde_json::Value> = (0..geometry.layers)
        .map(|layer| {
            let done = read_json(&work.join(format!("layers/L{layer:02}/DONE")));
            let file = &done["files"]["experts.bin"];
            serde_json::json!({"layer": layer, "bytes": file["bytes"], "sha256": file["sha256"]})
        })
        .collect();
    let checked = flash_next::check_experts_sha256(
        &reader,
        &bound.experts,
        &serde_json::json!({"experts_bin": experts_bin}),
    )
    .unwrap();
    assert_eq!(checked, 2);
}

#[test]
fn the_converters_python_fixture_tree_packs_and_reads() {
    if !Path::new(PYTHON).exists() {
        eprintln!("skip: {PYTHON} does not exist");
        return;
    }
    let tool = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/flash-next-converter");
    let scratch = Scratch(std::env::temp_dir().join(format!("ignis-converter-fixture-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&scratch.0);
    let output = Command::new(PYTHON)
        .current_dir(&tool)
        .env("CUDA_VISIBLE_DEVICES", "")
        .arg("convert.py")
        .arg("fixture")
        .arg("--out")
        .arg(&scratch.0)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "convert.py fixture failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // The fixture's shapes are the checkpoint's, with 2 layers of 8 experts.
    let mut geometry = FlashNextGeometry::qwen38_flash_next();
    geometry.layers = 2;
    geometry.experts = 8;
    let artifact = scratch.0.join("converter-fixture-v2.ninfer");
    let mut options = PackOptions::new(scratch.0.join("work"), artifact.clone(), geometry.clone());
    options.header_bytes = 1 << 20;
    options.accept_status = vec!["fixture".into()];
    let outcome = pack(&options, &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { converter_merged: true, .. }), "{outcome:?}");

    let reader = Reader::open(&artifact).unwrap();
    let mut widths = std::collections::BTreeSet::new();
    for layer in 0..geometry.layers {
        let mut end = None;
        for expert in 0..geometry.experts {
            for projection in Projection::ALL {
                let name = expert_name(layer, expert, projection);
                let Some(Object::Tensor(t)) = reader.find(&name) else {
                    panic!("{name} is not in the container");
                };
                assert_eq!(t.layout, StorageLayout::TrellisTile16V1);
                assert_eq!(t.shape, geometry.projection_shape(projection));
                assert_eq!(t.offset % 4096, 0);
                if let Some(end) = end {
                    assert_eq!(t.offset, end, "{name}: a layer's records are back to back");
                }
                end = Some(t.offset + t.bytes);
                widths.insert(TrellisK::from_format(t.format).expect("a trellis K"));
            }
        }
    }
    assert!(widths.len() >= 3, "the fixture spreads its experts over K classes: {widths:?}");
    let table = reader.find("layers.1.ple.ple_embedding.ngram_embedding.weight").unwrap();
    assert_eq!(table.bytes(), 1_000 * 90);
}
