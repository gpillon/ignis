//! The Flash-Next container end to end (spec flash-next/01, acceptance 6):
//! the shipped binaries on the fixture artifact, and the real artifact when
//! this machine has it.
//!
//! The real artifact is machine-local (`F:/ai/models/Qwen3.8-Flash-Next-ignis/`,
//! written by the converter and the packer). Its test skips when the model
//! directory is absent, as `real_artifact.rs` does for the 27B, or while the
//! conversion is still running; once the converter has recorded a complete
//! run, a missing container is a failure, not a skip.

use std::path::Path;
use std::process::Command;

use ignis_artifact::flash_next::{self, fixture, FlashNextGeometry};
use ignis_artifact::packer::{ARTIFACT_FILE_NAME, MTP_ARTIFACT_FILE_NAME};
use ignis_artifact::{verify, Reader, Sidecar};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";

#[test]
fn inspect_reads_the_flash_next_fixture() {
    let artifact = fixture::build("inspect").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ignis-artifact-inspect"))
        .arg(&artifact.path)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("identity: qwen3.8-flash-next/"), "{stdout}");
    assert!(stdout.contains("objects: 99 (92 tensors, 7 resources)"), "{stdout}");
    for format in ["TRELLIS_MUL1_K2", "TRELLIS_MUL1_K2P5", "TRELLIS_MUL1_K3", "TRELLIS_MUL1_K4", "Q4G32_F16S", "I64"] {
        assert!(stdout.contains(&format!("format {format}:")), "{format} in {stdout}");
    }
}

#[test]
fn the_pack_binary_waits_then_finishes_a_work_tree_beside_it() {
    let tree = fixture::WorkTree::new("pack-binary").unwrap();
    for unit in ["frontend", "global", "ngram"] {
        tree.write_unit(unit).unwrap();
    }
    tree.write_converter_json().unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_ignis-artifact-pack"))
            .arg("--work")
            .arg(tree.work_dir())
            .args(["--geometry", "fixture", "--header-mib", "1"])
            .output()
            .unwrap()
    };
    let waiting = run();
    let stdout = String::from_utf8_lossy(&waiting.stdout);
    assert_eq!(waiting.status.code(), Some(3), "{stdout}");
    assert!(stdout.contains("waiting: 3/5 units appended, layers/L00 is not complete yet"), "{stdout}");

    tree.write_unit("layers/L00").unwrap();
    tree.write_unit("layers/L01").unwrap();
    let done = run();
    let stdout = String::from_utf8_lossy(&done.stdout);
    assert!(done.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&done.stderr));
    assert!(stdout.contains("99 objects"), "{stdout}");
    // --out defaults to the artifact's name beside work/.
    let artifact = tree.work_dir().parent().unwrap().join(ARTIFACT_FILE_NAME);
    let reader = Reader::open(&artifact).unwrap();
    flash_next::bind(&reader, &FlashNextGeometry::fixture()).expect("the packed tree binds");
}

#[test]
fn the_pack_binary_packs_an_mtp_companion_beside_its_work_tree() {
    let main = fixture::build("pack-binary-mtp").unwrap();
    main.tree.write_mtp(&main.path).unwrap();
    let run = |pair: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ignis-artifact-pack"));
        command.arg("--work").arg(main.tree.mtp_work_dir()).args(["--geometry", "fixture", "--family", "mtp"]);
        if pair {
            command.arg("--pair-main").arg(&main.path);
        }
        command.output().unwrap()
    };
    let unpaired = run(false);
    assert_eq!(unpaired.status.code(), Some(2), "{}", String::from_utf8_lossy(&unpaired.stderr));
    assert!(String::from_utf8_lossy(&unpaired.stderr).contains("--family mtp needs --pair-main"));

    let done = run(true);
    let stdout = String::from_utf8_lossy(&done.stdout);
    assert!(done.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&done.stderr));
    // --out defaults to the companion's name beside work-mtp/, with its identity.
    let companion = main.tree.mtp_work_dir().parent().unwrap().join(MTP_ARTIFACT_FILE_NAME);
    let reader = Reader::open(&companion).unwrap();
    assert_eq!(reader.identity(), &ignis_artifact::packer::mtp_identity());
    assert!(reader.find("mtp.fc_hidden.weight").is_some());
}

/// The real artifact's path, or `None` (with the reason printed) when this
/// machine has none -- a CI runner, or a conversion still in progress. A
/// conversion recorded complete with no artifact is a failure, not a skip.
fn real_artifact_path() -> Option<std::path::PathBuf> {
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("skip: {MODEL_DIR} does not exist");
        return None;
    }
    let path = model_dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        let record = model_dir.join("work").join("converter.json");
        let complete = std::fs::read(&record)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .is_some_and(|value| value["status"] == "complete");
        assert!(
            !complete,
            "{} records a complete conversion but {} is missing: run ignis-artifact-pack --work {}",
            record.display(),
            path.display(),
            model_dir.join("work").display()
        );
        eprintln!("skip: the conversion in {MODEL_DIR} has not completed");
        return None;
    }
    Some(path)
}

/// The real artifact, when present: the reader parses it, the binder
/// consumes every object and the sidecar invariants hold. Seconds: nothing
/// here reads the 71.8 GB of payload.
#[test]
fn real_flash_next_artifact_binds_every_object() {
    let Some(path) = real_artifact_path() else { return };
    let reader = Reader::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let geometry = FlashNextGeometry::qwen38_flash_next();
    let bound = flash_next::bind(&reader, &geometry)
        .unwrap_or_else(|e| panic!("bind {}: {e}", path.display()));
    assert_eq!(bound.plan.object_count, reader.objects().len(), "every object consumed");
    assert_eq!(bound.experts.layers(), 48);
    assert_eq!(bound.experts.class_counts().values().sum::<usize>(), 48 * 512 * 2);
    assert_eq!(bound.plan.streamed_objects[0].bytes, 320_001_536 * 90, "the host-streamed n-gram table");

    let sidecar_path = format!("{}.conversion.json", path.display());
    let sidecar = Sidecar::load(Path::new(&sidecar_path))
        .unwrap_or_else(|e| panic!("load {sidecar_path}: {e}"));
    assert!(verify(&reader, &sidecar).unwrap().is_clean(), "file size and object count hold");
}

/// Every layer's expert bytes in the real artifact are the converter's
/// `experts.bin`. Ignored: it hashes tens of GB, minutes on every
/// `cargo test`. Run it after a conversion or a repack:
/// `cargo test -p ignis-artifact --test flash_next_artifact -- --ignored`
/// (the GPU profile's `--ignored` leg runs it too).
#[test]
#[ignore = "hashes every expert byte of the real 71.8 GB artifact: run after a conversion or repack"]
fn real_flash_next_expert_bytes_are_the_converters() {
    let Some(path) = real_artifact_path() else { return };
    let reader = Reader::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let bound = flash_next::bind(&reader, &FlashNextGeometry::qwen38_flash_next())
        .unwrap_or_else(|e| panic!("bind {}: {e}", path.display()));
    let sidecar_path = format!("{}.conversion.json", path.display());
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&sidecar_path).unwrap()).unwrap();
    assert_eq!(
        flash_next::check_experts_sha256(&reader, &bound.experts, &record).unwrap(),
        48,
        "every layer's expert bytes are the converter's experts.bin"
    );
}
