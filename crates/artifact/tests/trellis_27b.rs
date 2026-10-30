//! The 27B's 16 GB artifact (spec 27b-16gb/01, GitHub #338).
//!
//! - The packer's `qwen3.8-27b-trellis` family packs a converter work tree
//!   (`tools/27b-16gb-converter`) in the served artifact's unit order, the
//!   same tree to the same bytes, and refuses an object outside its unit.
//! - Machine-local (a directory walk, CPU only; skips when the file is
//!   absent): the real artifact is consumed whole by the binder (ADR 0002),
//!   its text scope binds with the token embedding in host memory at
//!   <= 10.5 GB of device bytes (Acceptance 4), and the drafter, the draft
//!   head and the vision tower bind as in the served artifact.

use std::path::{Path, PathBuf};

use ignis_artifact::binder::{Binder, Residence};
use ignis_artifact::packer::{pack, sidecar_path, PackOptions, PackOutcome, TRELLIS_27B_ARTIFACT_FILE_NAME};
use ignis_artifact::{
    bind_model_scope_27b_with, body_27b, dflash2_scope_27b, draft_head_scope_27b, model_scope_27b_of,
    tensor_encoded_size, text_scope_27b_trellis, vision_scope_27b, Body27b, DraftModule, InventoryEntry, ModelScope,
    NumericFormat, Object, Reader, Sidecar, StorageLayout, VisionResidency,
};
use serde_json::json;
use sha2::{Digest, Sha256};

const REAL_ARTIFACT: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
/// The served 32 GB artifact the 16 GB one copies from.
const SERVED_ARTIFACT: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer";

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ignis-trellis-27b-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

type Tensor = (&'static str, NumericFormat, StorageLayout, &'static [u64]);

const BF16: (NumericFormat, StorageLayout) = (NumericFormat::Bf16, StorageLayout::ContiguousLeV1);

/// The reduced tree's tensors per unit: a norm, two trellis matrices of
/// different K, a W8 row-split table, and one object in each other unit.
fn units() -> Vec<(&'static str, Vec<Tensor>)> {
    vec![
        (
            "text",
            vec![
                ("text/token_embedding", NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1, &[256, 128]),
                ("text/layers/0/input_norm", BF16.0, BF16.1, &[128]),
                ("text/layers/0/mlp/gate_proj", NumericFormat::TrellisMul1K3, StorageLayout::TrellisTile16V1, &[256, 128]),
                ("text/output_head", NumericFormat::TrellisMul1K4, StorageLayout::TrellisTile16V1, &[256, 128]),
            ],
        ),
        ("mtp", vec![("mtp/final_norm", BF16.0, BF16.1, &[128])]),
        ("vision", vec![("vision/patch_embedding_bias", BF16.0, BF16.1, &[64])]),
        ("dflash2", vec![("dflash2/context_norm", BF16.0, BF16.1, &[128])]),
    ]
}

/// Deterministic payload bytes of `len` for `name`.
fn payload(name: &str, len: u64) -> Vec<u8> {
    let seed = Sha256::digest(name.as_bytes());
    (0..len).map(|i| seed[(i % 32) as usize] ^ (i as u8)).collect()
}

/// Write `files` (name -> bytes) into `dir` and its `DONE`.
fn write_files(dir: &Path, files: &[(String, Vec<u8>)]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut done = serde_json::Map::new();
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).unwrap();
        done.insert(name.clone(), json!({"bytes": bytes.len(), "sha256": hex(&Sha256::digest(bytes))}));
    }
    std::fs::write(dir.join("DONE"), serde_json::to_vec(&json!({"files": done})).unwrap()).unwrap();
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The converter's work tree: `frontend/`, the four tensor units (each a
/// `tensors.json` and one file per tensor) and `converter.json`.
fn work_tree(root: &Path, units: &[(&'static str, Vec<Tensor>)]) -> PathBuf {
    let work = root.join("work");
    write_files(
        &work.join("frontend"),
        &[("tokenizer.json".into(), b"{\"t\":1}".to_vec()), ("chat_template.jinja".into(), b"{{ x }}".to_vec())],
    );
    for (unit, tensors) in units {
        let mut files = Vec::new();
        let mut listed = Vec::new();
        for (name, format, layout, shape) in tensors {
            let bytes = tensor_encoded_size(*layout, *format, shape).unwrap();
            let file = format!("{}.bin", name[unit.len() + 1..].replace('/', "."));
            listed.push(json!({"name": name, "file": file, "format": format.name(), "layout": layout.name(),
                               "shape": shape, "bytes": bytes}));
            files.push((file, payload(name, bytes)));
        }
        files.push(("tensors.json".into(), serde_json::to_vec(&json!({"tensors": listed})).unwrap()));
        write_files(&work.join(unit), &files);
    }
    std::fs::write(
        work.join("converter.json"),
        serde_json::to_vec(&json!({"schema": "qwen3.8-27b-trellis-converter-v1", "status": "complete"})).unwrap(),
    )
    .unwrap();
    work
}

/// Pack `work` into `artifact` without consuming the work files.
fn pack_tree(work: &Path, artifact: &Path) -> Reader {
    let mut options = PackOptions::trellis_27b(work.to_path_buf(), artifact.to_path_buf());
    options.keep_work = true;
    let outcome = pack(&options, &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { converter_merged: true, .. }), "{outcome:?}");
    Reader::open(artifact).unwrap()
}

/// ADR 0002: bind every object against its own descriptor and finish.
fn bind_all(reader: &Reader) -> ignis_artifact::MaterializationPlan {
    let mut binder = Binder::new(reader);
    for object in reader.objects() {
        match object {
            Object::Tensor(t) => {
                let handle = binder.require_tensor(&t.name, t.format, t.layout, &t.shape).unwrap();
                binder.materialize_on_device(handle).unwrap();
            }
            Object::Resource(r) => {
                let handle = binder.require_resource(&r.name, r.encoding).unwrap();
                binder.retain_on_host(handle).unwrap();
            }
        }
    }
    binder.finish().expect("every object consumed (ADR 0002)")
}

#[test]
fn the_converters_tree_packs_in_the_served_unit_order() {
    let scratch = Scratch::new("order");
    let work = work_tree(&scratch.0, &units());
    let artifact = scratch.0.join(TRELLIS_27B_ARTIFACT_FILE_NAME);
    let reader = pack_tree(&work, &artifact);

    assert_eq!(reader.identity().model_id, "qwen3.8-27b");
    assert_eq!(reader.identity().weights_id, "trellis-sc30");
    let names: Vec<&str> = reader.objects().iter().map(|o| o.name()).collect();
    assert_eq!(
        names,
        [
            "frontend/chat_template.jinja",
            "frontend/tokenizer.json",
            "text/token_embedding",
            "text/layers/0/input_norm",
            "text/layers/0/mlp/gate_proj",
            "text/output_head",
            "mtp/final_norm",
            "vision/patch_embedding_bias",
            "dflash2/context_norm",
        ]
    );
    for (_, tensors) in units() {
        for (name, format, layout, shape) in tensors {
            let Some(Object::Tensor(t)) = reader.find(name) else { panic!("{name} is not a tensor") };
            assert_eq!((t.format, t.layout, t.shape.as_slice()), (format, layout, shape), "{name}");
            assert_eq!(reader.payload(name).unwrap().data, payload(name, t.bytes).as_slice(), "{name}");
            if layout == StorageLayout::TrellisTile16V1 {
                assert_eq!(reader.payload(name).unwrap().absolute_offset % 4096, 0, "{name} is 4096-aligned");
            }
        }
    }
    let plan = bind_all(&reader);
    assert_eq!(plan.object_count, names.len());

    let sidecar: serde_json::Value = serde_json::from_slice(&std::fs::read(sidecar_path(&artifact)).unwrap()).unwrap();
    assert_eq!(sidecar["recipe_id"], "qwen3_8_27b_trellis_sc30-v2");
    assert_eq!(sidecar["artifact"]["bytes"], reader.file_bytes());
    assert_eq!(sidecar["objects"]["count"], names.len());
    assert_eq!(sidecar["schema"], "qwen3.8-27b-trellis-converter-v1", "converter.json is merged");
}

#[test]
fn the_same_tree_packs_to_the_same_bytes() {
    let scratch = Scratch::new("same");
    let work = work_tree(&scratch.0, &units());
    let a = scratch.0.join("a").join(TRELLIS_27B_ARTIFACT_FILE_NAME);
    let b = scratch.0.join("b").join(TRELLIS_27B_ARTIFACT_FILE_NAME);
    std::fs::create_dir_all(a.parent().unwrap()).unwrap();
    std::fs::create_dir_all(b.parent().unwrap()).unwrap();
    pack_tree(&work, &a);
    pack_tree(&work, &b);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

#[test]
fn an_object_outside_its_unit_is_refused() {
    let scratch = Scratch::new("outside");
    let mut tree = units();
    // `mtp` is a plain unit here, not Flash-Next's: its objects are `mtp/...`
    tree[1].1.push(("text/final_norm", BF16.0, BF16.1, &[128]));
    let work = work_tree(&scratch.0, &tree);
    let mut options = PackOptions::trellis_27b(work, scratch.0.join(TRELLIS_27B_ARTIFACT_FILE_NAME));
    options.keep_work = true;
    let err = pack(&options, &mut |_| {}).unwrap_err().to_string();
    assert!(err.contains("text/final_norm, outside mtp/"), "{err}");
}

/// Bind `entries` with each one's residence, on top of whatever `binder`
/// holds.
fn bind_entries(binder: &mut Binder, entries: &[InventoryEntry]) {
    for e in entries {
        let handle = binder
            .require_tensor(e.name, e.format, e.layout, e.shape)
            .unwrap_or_else(|err| panic!("{}: {err}", e.name));
        match e.residence {
            Residence::Device => binder.materialize_on_device(handle).unwrap(),
            Residence::HostMapped => binder.place_in_host_mapped(handle).unwrap(),
        }
    }
}

/// Spec 27b-16gb/05a (GitHub #339): the 27B's own scope binds the real
/// artifact by its body, read off the directory, with no caller naming it --
/// the scope a load asks for, over whichever body the artifact carries -- at
/// the device bytes spec 01's As built states.
#[test]
fn real_trellis_artifact_binds_through_the_27bs_model_scope() {
    let path = Path::new(REAL_ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {REAL_ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap();
    assert_eq!(body_27b(&reader), Body27b::Trellis);
    let text = ModelScope::default();
    let drafted = ModelScope { draft: Some(DraftModule::Dflash2ShortlistHead), ..text };
    let with_vision = ModelScope { vision: true, vision_residency: VisionResidency::Device, ..drafted };
    for (scope, entries, device_bytes) in
        [(text, 803, 9_825_087_488u64), (drafted, 803 + 66 + 2, 11_264_998_400), (with_vision, 803 + 68 + 333, 11_560_717_824)]
    {
        assert_eq!(model_scope_27b_of(&reader, scope).unwrap().len(), entries, "{scope:?}");
        let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap();
        assert_eq!(handles.len(), entries, "{scope:?}");
        assert_eq!(plan.device_capacity_bytes, device_bytes, "{scope:?}");
    }
    // The served artifact's head is W8: its body is the NVFP4 one.
    let served_path = Path::new(SERVED_ARTIFACT);
    if served_path.exists() {
        assert_eq!(body_27b(&Reader::open(served_path).unwrap()), Body27b::Nvfp4);
    }
}

#[test]
fn real_trellis_artifact_is_consumed_whole_and_its_text_scope_fits() {
    let path = Path::new(REAL_ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {REAL_ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap();
    assert_eq!(reader.identity().model_id, "qwen3.8-27b");
    assert_eq!(reader.identity().weights_id, "trellis-sc30");

    // Acceptance 3: every object consumed (ADR 0002).
    let all = bind_all(&reader);
    assert_eq!(all.object_count, reader.objects().len());

    // The sidecar's whole-file record holds, and its NVFP4 records cover
    // exactly the copied drafter's NVFP4 objects, every one clean.
    let sidecar = Sidecar::load(&sidecar_path(path)).unwrap();
    let report = ignis_artifact::verify(&reader, &sidecar).unwrap();
    assert!(report.is_clean(), "{:?} {:?}", report.global_flags, report.flagged());
    assert_eq!((report.nvfp4_objects, report.nvfp4_records, report.nvfp4_uncovered()), (34, 34, 0));

    // Acceptance 4: the text scope, the embedding in host memory.
    let format_of = |name: &str| match reader.find(name) {
        Some(Object::Tensor(t)) => Some(t.format),
        _ => None,
    };
    let text = text_scope_27b_trellis(format_of).expect("the trellis text scope");
    let mut binder = Binder::new(&reader);
    bind_entries(&mut binder, &text);
    let text_plan = binder.plan();
    let text_bytes = text_plan.device_capacity_bytes;
    eprintln!("text scope: {} objects, device arena {text_bytes} bytes", text.len());
    assert!(text_bytes <= 10_500_000_000, "text scope {text_bytes} bytes > 10.5 GB");

    // The drafter, its draft head and the vision tower are the served
    // artifact's objects: their inventories bind unchanged.
    bind_entries(&mut binder, &dflash2_scope_27b());
    bind_entries(&mut binder, &draft_head_scope_27b());
    let drafted = binder.plan().device_capacity_bytes;
    bind_entries(&mut binder, &vision_scope_27b());
    let with_vision = binder.plan().device_capacity_bytes;
    eprintln!("with DFlash2 + draft head: {drafted} bytes; with vision too: {with_vision} bytes");

    // Every object is the text scope's, a module's, the MTP layer's or the
    // frontend's: nothing else rides along.
    let bound = binder.plan().object_count;
    let mtp = reader.objects().iter().filter(|o| o.name().starts_with("mtp/")).count();
    let frontend = reader.objects().iter().filter(|o| o.name().starts_with("frontend/")).count();
    assert_eq!(bound + mtp + frontend, reader.objects().len());

    // Copied objects of every kind are the served artifact's bytes, read
    // through this crate's reader on both sides (the converter's own check
    // reads both through its Python reader).
    let served_path = Path::new(SERVED_ARTIFACT);
    if !served_path.exists() {
        eprintln!("skip the copy cross-check: {SERVED_ARTIFACT} does not exist");
        return;
    }
    let served = Reader::open(served_path).unwrap();
    for name in [
        "frontend/tokenizer.json",
        "text/token_embedding",
        "text/layers/4/gdn/a_b_projection",
        "text/layers/63/post_attention_norm",
        "text/draft_head",
        "text/draft_head_token_ids",
        "mtp/input_projection",
        "vision/layers/26/mlp/fc2",
        "dflash2/selector/successor",
    ] {
        let ours = Sha256::digest(reader.payload(name).unwrap().data);
        let theirs = Sha256::digest(served.payload(name).unwrap().data);
        assert_eq!(ours, theirs, "{name} is the served object's bytes");
    }
}
