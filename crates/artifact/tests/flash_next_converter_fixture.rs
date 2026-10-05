//! The converter's own fixture tree packs (spec flash-next/01: the reader
//! parses what the converter's writer produces). Machine-local: it runs
//! `tools/flash-next-converter/convert.py fixture` with the study's Python
//! environment, CPU only, and skips when that environment is absent.
//!
//! That tree carries real-size expert records but only three non-expert
//! tensors per layer, so it is packed and read here, not bound: the bind of
//! a converter-written tree is checked on the real dry run and the real
//! artifact (`flash_next_artifact.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

use ignis_artifact::flash_next::{expert_name, FlashNextGeometry, Projection, TrellisK};
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
