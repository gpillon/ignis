//! The 27B's NVFP4 artifact family (GitHub #467): its inventory, and the
//! packer's `qwen3.8-27b-nvfp4` family on a reduced synthetic work tree.
//!
//! - The inventory at the served geometry is ignis's 27B inventory (the text
//!   scope, the draft head, the vision tower, the DFlash2 module) with the
//!   MTP layer and the frontend, in the served artifact's order; at the
//!   fixture geometry it is the same objects, narrower.
//! - The packer packs the fixture's work tree (`docs/formats/
//!   27b-nvfp4-work-tree.md`) to the inventory under the weights id it is
//!   given, with a sidecar the loader's checksum verification accepts; it
//!   refuses, by name and before writing anything, a tree that is not the
//!   inventory's or a `converter.json` short of the sidecar's fields.
//! - Machine-local (a directory walk, CPU only; skips when the file is
//!   absent): the inventory is the served artifact's directory.

use std::path::{Path, PathBuf};

use ignis_artifact::binder::Binder;
use ignis_artifact::nvfp4_27b::{self, Entry, Geometry, FRONTEND_FILES, UNITS};
use ignis_artifact::packer::{pack, sidecar_path, state_path, PackOptions, PackOutcome};
use ignis_artifact::{
    dflash2_scope_27b, draft_head_scope_27b, tensor_encoded_size, text_scope_27b, verify, vision_scope_27b,
    NumericFormat, Object, Reader, Sidecar, StorageLayout,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// The served artifact (the one `real_artifact.rs` and `sidecar_verify.rs` read).
const SERVED_ARTIFACT: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer";

type Described = (String, Option<(NumericFormat, StorageLayout, Vec<u64>)>);

fn described(entries: &[Entry]) -> Vec<Described> {
    entries
        .iter()
        .map(|e| match e {
            Entry::Resource { name } => (name.clone(), None),
            Entry::Tensor { name, format, layout, shape } => (name.clone(), Some((*format, *layout, shape.clone()))),
        })
        .collect()
}

/// The served artifact's 12 MTP objects, as its directory lists them.
fn served_mtp() -> Vec<Described> {
    let w8 = (NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1);
    let bf16 = (NumericFormat::Bf16, StorageLayout::ContiguousLeV1);
    [
        ("mtp/input_projection", w8, vec![5120, 10240]),
        ("mtp/embedding_norm", bf16, vec![5120]),
        ("mtp/hidden_norm", bf16, vec![5120]),
        ("mtp/layer/input_norm", bf16, vec![5120]),
        ("mtp/layer/attention/query_key_gate_value", w8, vec![14336, 5120]),
        ("mtp/layer/attention/query_norm", bf16, vec![256]),
        ("mtp/layer/attention/key_norm", bf16, vec![256]),
        ("mtp/layer/attention/output", w8, vec![5120, 6144]),
        ("mtp/layer/post_attention_norm", bf16, vec![5120]),
        ("mtp/layer/mlp/gate_up", w8, vec![34816, 5120]),
        ("mtp/layer/mlp/down", w8, vec![5120, 17408]),
        ("mtp/final_norm", bf16, vec![5120]),
    ]
    .into_iter()
    .map(|(name, (format, layout), shape)| (name.to_string(), Some((format, layout, shape))))
    .collect()
}

#[test]
fn the_served_inventory_is_ignis_27b_inventory_with_the_mtp_layer_and_the_frontend() {
    let inventory = nvfp4_27b::inventory(&Geometry::qwen3_8_27b());
    let scope = |entries: Vec<ignis_artifact::InventoryEntry>| -> Vec<Described> {
        entries.into_iter().map(|e| (e.name.to_string(), Some((e.format, e.layout, e.shape.to_vec())))).collect()
    };
    let mut expected: Vec<Described> = FRONTEND_FILES.iter().map(|f| (format!("frontend/{f}"), None)).collect();
    expected.extend(scope(text_scope_27b()));
    expected.extend(scope(draft_head_scope_27b()));
    expected.extend(served_mtp());
    expected.extend(scope(vision_scope_27b()));
    expected.extend(scope(dflash2_scope_27b()));
    assert_eq!(described(&inventory), expected);

    // The served recipe's counts (ninfer's `validate_inventory`).
    assert_eq!(inventory.len(), 1325);
    let mut formats = std::collections::BTreeMap::new();
    let mut layouts = std::collections::BTreeMap::new();
    for e in &inventory {
        if let Entry::Tensor { format, layout, .. } = e {
            *formats.entry(format.name()).or_insert(0) += 1;
            *layouts.entry(layout.name()).or_insert(0) += 1;
        }
    }
    let formats: Vec<(&str, i32)> = formats.into_iter().collect();
    assert_eq!(
        formats,
        [("BF16", 575), ("FP32", 343), ("I32", 1), ("NVFP4", 281), ("Q4G64_F16S", 55), ("Q5G64_F16S", 54),
         ("Q6G64_F16S", 1), ("W8G32_F16S", 9)]
    );
    let layouts: Vec<(&str, i32)> = layouts.into_iter().collect();
    assert_eq!(layouts, [("blockscale-k16-m128x4-v1", 281), ("contiguous-le-v1", 919), ("row-split-k128-v1", 119)]);
    // Every object belongs to its unit, the units in container order.
    let units: Vec<&str> = inventory.iter().map(Entry::unit).collect();
    let mut order: Vec<&str> = units.clone();
    order.dedup();
    assert_eq!(order, UNITS);
}

#[test]
fn the_fixture_inventory_is_the_served_one_narrower() {
    let served = nvfp4_27b::inventory(&Geometry::qwen3_8_27b());
    let fixture = nvfp4_27b::inventory(&Geometry::fixture());
    assert_eq!(fixture.len(), served.len());
    let mut bytes = 0;
    for (f, s) in fixture.iter().zip(&served) {
        match (f, s) {
            (Entry::Resource { name: a }, Entry::Resource { name: b }) => assert_eq!(a, b),
            (
                Entry::Tensor { name: a, format: fa, layout: la, shape: sa },
                Entry::Tensor { name: b, format: fb, layout: lb, shape: sb },
            ) => {
                assert_eq!((a, fa, la, sa.len()), (b, fb, lb, sb.len()), "{a}");
                bytes += tensor_encoded_size(*la, *fa, sa).unwrap_or_else(|e| panic!("{a}: {e}"));
            }
            _ => panic!("{} and {} are not the same kind", f.name(), s.name()),
        }
    }
    assert!(bytes < 16 << 20, "the fixture's tensors hold {bytes} bytes");
}

/// Machine-local: the inventory is the served artifact's directory, object
/// for object and in its order.
#[test]
fn the_served_inventory_is_the_served_artifacts_directory() {
    let path = Path::new(SERVED_ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {SERVED_ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap();
    assert_eq!(described(&nvfp4_27b::inventory(&Geometry::qwen3_8_27b())), directory(&reader));
}

// ---------------------------------------------------------------------------
// The packer's `qwen3.8-27b-nvfp4` family on the fixture's work tree
// ---------------------------------------------------------------------------

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ignis-nvfp4-27b-{tag}-{}", std::process::id()));
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

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

/// Rewrite `dir`'s DONE over the files it holds now (the converter's
/// `work.mark_done`).
fn mark_done(dir: &Path) {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n != "DONE")
        .collect();
    names.sort();
    let mut files = serde_json::Map::new();
    for name in names {
        let bytes = std::fs::read(dir.join(&name)).unwrap();
        files.insert(name, json!({"bytes": bytes.len(), "sha256": hex(&Sha256::digest(&bytes))}));
    }
    write_json(&dir.join("DONE"), &json!({"files": files}));
}

/// Deterministic payload bytes of `len` for `name`.
fn payload(name: &str, len: u64) -> Vec<u8> {
    let seed = Sha256::digest(name.as_bytes());
    (0..len).map(|i| seed[(i % 32) as usize] ^ (i as u8)).collect()
}

/// The weight divisor the tree stores at the tail of the `index`-th text
/// NVFP4 parent.
fn divisor(index: usize) -> f32 {
    2688.0 / (index as f32 + 1.5)
}

/// The work file of an object: its name past the unit, `/` -> `.`, `.bin`
/// for a tensor.
fn file_name(name: &str) -> String {
    let rest = &name[name.find('/').unwrap() + 1..];
    if name.starts_with("frontend/") { rest.to_string() } else { format!("{}.bin", rest.replace('/', ".")) }
}

const WEIGHTS_ID: &str = "nvfp4full-fixture";
const ARTIFACT: &str = "qwen3_8_27b_nvfp4full-v2-fixture.ninfer";

/// The converter's work tree at the fixture geometry: `frontend/`, the four
/// tensor units (a `tensors.json` and one file per tensor) and
/// `converter.json`, whose `local_nvfp4.parents` records every NVFP4
/// parent: the text ones with the divisor their file ends with, the
/// drafter's with none.
fn work_tree(root: &Path) -> PathBuf {
    let work = root.join("work");
    let mut listed: std::collections::BTreeMap<String, Vec<Value>> = Default::default();
    let mut parents = serde_json::Map::new();
    let mut text_parents = 0;
    for unit in UNITS {
        std::fs::create_dir_all(work.join(unit)).unwrap();
    }
    for entry in nvfp4_27b::inventory(&Geometry::fixture()) {
        let dir = work.join(entry.unit());
        let file = file_name(entry.name());
        match &entry {
            Entry::Resource { name } => std::fs::write(dir.join(&file), format!("{{\"resource\":\"{name}\"}}")).unwrap(),
            Entry::Tensor { name, format, layout, shape } => {
                let bytes = tensor_encoded_size(*layout, *format, shape).unwrap();
                let mut data = payload(name, bytes);
                if *format == NumericFormat::Nvfp4 {
                    if name.starts_with("dflash2/") {
                        parents.insert(name.clone(), json!({"weight_scale_divisor": null, "relative_frobenius_error": 0.095}));
                    } else {
                        let d = divisor(text_parents);
                        text_parents += 1;
                        let tail = data.len() - 4;
                        data[tail..].copy_from_slice(&d.to_le_bytes());
                        parents.insert(name.clone(), json!({"weight_scale_divisor": f64::from(d), "relative_frobenius_error": 0.09}));
                    }
                }
                std::fs::write(dir.join(&file), &data).unwrap();
                listed.entry(entry.unit().to_string()).or_default().push(json!({
                    "name": name, "file": file, "format": format.name(), "layout": layout.name(), "shape": shape,
                    "bytes": bytes,
                }));
            }
        }
    }
    for unit in UNITS {
        if let Some(tensors) = listed.remove(unit) {
            write_json(&work.join(unit).join("tensors.json"), &json!({"tensors": tensors}));
        }
        mark_done(&work.join(unit));
    }
    write_json(
        &work.join("converter.json"),
        &json!({
            "schema": "qwen3.8-27b-nvfp4-converter-v1",
            "status": "complete",
            "recipe_id": "qwen3_8_27b_nvfp4full-v2-fixture",
            "identity": {"model_id": "qwen3.8-27b", "weights_id": WEIGHTS_ID},
            "source": {"repo": "example/Qwen3.8-27B-fine-tune", "revision": "0123456789abcdef0123456789abcdef01234567",
                       "license": "apache-2.0"},
            "local_nvfp4": {"encoder_profile": "NVFP4_MAXABS_DIVISOR_RNE_V1", "parents": parents},
        }),
    );
    work
}

/// The options that pack `work` into `artifact` under `weights_id`, keeping
/// the work files.
fn options(work: &Path, artifact: &Path, weights_id: &str) -> PackOptions {
    let mut options = PackOptions::nvfp4_27b(work.to_path_buf(), artifact.to_path_buf(), weights_id, Geometry::fixture());
    options.keep_work = true;
    options
}

fn pack_tree(work: &Path, artifact: &Path, weights_id: &str) -> Reader {
    let outcome = pack(&options(work, artifact, weights_id), &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { converter_merged: true, .. }), "{outcome:?}");
    Reader::open(artifact).unwrap()
}

/// ADR 0002: bind every object against its own descriptor and finish.
fn bind_all(reader: &Reader) -> usize {
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
    binder.finish().expect("every object consumed (ADR 0002)").object_count
}

fn directory(reader: &Reader) -> Vec<Described> {
    reader
        .objects()
        .iter()
        .map(|o| match o {
            Object::Resource(r) => (r.name.clone(), None),
            Object::Tensor(t) => (t.name.clone(), Some((t.format, t.layout, t.shape.clone()))),
        })
        .collect()
}

#[test]
fn the_work_tree_packs_to_the_inventory_under_the_given_weights_id() {
    let scratch = Scratch::new("pack");
    let work = work_tree(&scratch.0);
    let artifact = scratch.0.join(ARTIFACT);
    let reader = pack_tree(&work, &artifact, WEIGHTS_ID);

    assert_eq!(reader.identity().model_id, "qwen3.8-27b");
    assert_eq!(reader.identity().weights_id, WEIGHTS_ID);
    // Every object with its format, layout and shape, in the served order
    // (`tokenizer.json` first: not DONE's sorted order).
    assert_eq!(directory(&reader), described(&nvfp4_27b::inventory(&Geometry::fixture())));
    for object in reader.objects() {
        let unit = object.name().split('/').next().unwrap();
        let file = std::fs::read(work.join(unit).join(file_name(object.name()))).unwrap();
        assert_eq!(reader.payload(object.name()).unwrap().data, file.as_slice(), "{}", object.name());
    }
    assert_eq!(bind_all(&reader), 1325);

    // The sidecar: the converter's record carried through, and the whole-file
    // invariants.
    let sidecar = read_json(&sidecar_path(&artifact));
    assert_eq!(sidecar["recipe_id"], "qwen3_8_27b_nvfp4full-v2-fixture");
    assert_eq!(sidecar["artifact"]["bytes"], reader.file_bytes());
    assert_eq!(sidecar["objects"]["count"], 1325);
    assert_eq!(sidecar["source"]["license"], "apache-2.0");
    assert_eq!(sidecar["source"]["repo"], "example/Qwen3.8-27B-fine-tune");
    assert_eq!(sidecar["source"]["revision"], "0123456789abcdef0123456789abcdef01234567");
}

#[test]
fn the_loaders_checksum_verification_accepts_the_sidecar_and_refuses_a_divisor_that_differs() {
    let scratch = Scratch::new("verify");
    let work = work_tree(&scratch.0);
    let artifact = scratch.0.join(ARTIFACT);
    let reader = pack_tree(&work, &artifact, WEIGHTS_ID);

    let sidecar = Sidecar::load(&sidecar_path(&artifact)).unwrap();
    let report = verify(&reader, &sidecar).unwrap();
    assert!(report.is_clean(), "{:?} {:?}", report.global_flags, report.flagged());
    // Every NVFP4 object is recorded: the 247 text parents with the divisor
    // the container stores, the drafter's 34 with none.
    assert_eq!((report.nvfp4_objects, report.nvfp4_records, report.nvfp4_uncovered()), (281, 281, 0));
    let with_divisor: Vec<&str> =
        sidecar.nvfp4_parents.iter().filter(|p| p.weight_scale_divisor.is_some()).map(|p| p.name.as_str()).collect();
    assert_eq!(with_divisor.len(), 247);
    assert!(with_divisor.iter().all(|name| name.starts_with("text/layers/")), "{with_divisor:?}");
    let first = sidecar.nvfp4_parents.iter().find(|p| p.name == "text/layers/0/gdn/query_key_value_z").unwrap();
    assert_eq!(first.weight_scale_divisor, Some(f64::from(divisor(0))));

    // A recorded divisor that is not the container's word is flagged, by name.
    let mut record = read_json(&sidecar_path(&artifact));
    record["local_nvfp4"]["parents"]["text/layers/27/attention/query_key_gate_value"]["weight_scale_divisor"] =
        json!(1.25);
    write_json(&sidecar_path(&artifact), &record);
    let report = verify(&reader, &Sidecar::load(&sidecar_path(&artifact)).unwrap()).unwrap();
    assert!(!report.is_clean());
    let flagged = report.flagged();
    assert_eq!(flagged.len(), 1, "{flagged:?}");
    assert_eq!(flagged[0].name, "text/layers/27/attention/query_key_gate_value");
    assert!(flagged[0].detail.contains("sidecar records 1.25"), "{}", flagged[0].detail);
}

#[test]
fn the_same_tree_under_two_weights_ids_gives_two_content_hashes() {
    let scratch = Scratch::new("hashes");
    let work = work_tree(&scratch.0);
    let mut hashes = Vec::new();
    for (dir, weights_id) in [("a", WEIGHTS_ID), ("b", WEIGHTS_ID), ("c", "nvfp4full-other")] {
        std::fs::create_dir_all(scratch.0.join(dir)).unwrap();
        // converter.json records the identity the run packs.
        let converter = work.join("converter.json");
        let mut record = read_json(&converter);
        record["identity"]["weights_id"] = json!(weights_id);
        write_json(&converter, &record);
        let artifact = scratch.0.join(dir).join(ARTIFACT);
        let reader = pack_tree(&work, &artifact, weights_id);
        assert_eq!(reader.identity().weights_id, weights_id);
        hashes.push((reader.content_hash(), std::fs::read(&artifact).unwrap()));
    }
    assert_eq!(hashes[0], hashes[1], "the same tree and weights id: the same bytes");
    assert_ne!(hashes[0].0, hashes[2].0, "another weights id: another content hash");
}

/// Change `unit`'s `tensors.json` entries with `change`, and its DONE with it.
fn edit_listing(work: &Path, unit: &str, change: impl FnOnce(&mut Vec<Value>)) {
    let path = work.join(unit).join("tensors.json");
    let mut listing = read_json(&path);
    change(listing["tensors"].as_array_mut().unwrap());
    write_json(&path, &listing);
    mark_done(&work.join(unit));
}

/// The listing entry of `name` in `tensors`.
fn entry_of<'a>(tensors: &'a mut [Value], name: &str) -> &'a mut Value {
    tensors.iter_mut().find(|t| t["name"] == name).unwrap()
}

/// Change `converter.json` with `change`.
fn edit_record(work: &Path, change: impl FnOnce(&mut Value)) {
    let path = work.join("converter.json");
    let mut record = read_json(&path);
    change(&mut record);
    write_json(&path, &record);
}

/// The packer's error on the tree `change` makes (by a run that deletes its
/// work files), having written nothing and consumed nothing.
fn refusal(tag: &str, weights_id: &str, change: impl FnOnce(&Path)) -> String {
    let scratch = Scratch::new(tag);
    let work = work_tree(&scratch.0);
    change(&work);
    let artifact = scratch.0.join(ARTIFACT);
    let mut options = options(&work, &artifact, weights_id);
    options.keep_work = false;
    let err = pack(&options, &mut |_| {}).unwrap_err().to_string();
    for written in [artifact.clone(), state_path(&artifact), sidecar_path(&artifact)] {
        assert!(!written.exists(), "{tag}: {} was written", written.display());
    }
    for unit in UNITS {
        assert!(work.join(unit).join("DONE").exists(), "{tag}: {unit} was consumed");
    }
    err
}

#[test]
fn a_tree_that_is_not_the_inventorys_is_refused_by_name_and_nothing_is_written() {
    let cases: Vec<(&str, Box<dyn FnOnce(&Path)>, &str)> = vec![
        (
            "missing",
            Box::new(|work: &Path| {
                edit_listing(work, "text", |t| t.retain(|e| e["name"] != "text/layers/5/gdn/norm"));
                std::fs::remove_file(work.join("text/layers.5.gdn.norm.bin")).unwrap();
                mark_done(&work.join("text"));
            }),
            "the work tree has no text/layers/5/gdn/norm: text/tensors.json does not list it",
        ),
        (
            "format",
            Box::new(|work: &Path| {
                edit_listing(work, "text", |t| entry_of(t, "text/layers/4/gdn/output")["format"] = json!("NVFP4"));
            }),
            "text/layers/4/gdn/output is NVFP4 contiguous-le-v1 [128, 192] in text/tensors.json; \
             the 27B NVFP4 inventory has BF16 contiguous-le-v1 [128, 192]",
        ),
        (
            "layout",
            Box::new(|work: &Path| {
                edit_listing(work, "vision", |t| entry_of(t, "vision/merger/fc1")["layout"] = json!("contiguous-le-v1"));
            }),
            "vision/merger/fc1 is W8G32_F16S contiguous-le-v1 [128, 128] in vision/tensors.json; \
             the 27B NVFP4 inventory has W8G32_F16S row-split-k128-v1 [128, 128]",
        ),
        (
            "shape",
            Box::new(|work: &Path| {
                edit_listing(work, "text", |t| {
                    entry_of(t, "text/layers/3/attention/query_key_gate_value")["shape"] = json!([384, 64])
                });
            }),
            "text/layers/3/attention/query_key_gate_value is BF16 contiguous-le-v1 [384, 64]",
        ),
        (
            "unknown",
            Box::new(|work: &Path| {
                std::fs::write(work.join("mtp/layer.extra.bin"), [0u8; 256]).unwrap();
                edit_listing(work, "mtp", |t| {
                    t.push(json!({"name": "mtp/layer/extra", "file": "layer.extra.bin", "format": "BF16",
                                  "layout": "contiguous-le-v1", "shape": [128], "bytes": 256}))
                });
            }),
            "mtp/tensors.json lists mtp/layer/extra, which is not a mtp/ object of the 27B NVFP4 inventory",
        ),
        (
            "frontend-missing",
            Box::new(|work: &Path| {
                std::fs::remove_file(work.join("frontend/chat_template.jinja")).unwrap();
                mark_done(&work.join("frontend"));
            }),
            "the work tree has no frontend/chat_template.jinja",
        ),
        (
            "frontend-extra",
            Box::new(|work: &Path| {
                std::fs::write(work.join("frontend/config.json"), b"{}").unwrap();
                mark_done(&work.join("frontend"));
            }),
            "frontend/config.json is not an object of the 27B NVFP4 inventory",
        ),
    ];
    for (tag, change, expected) in cases {
        let err = refusal(tag, WEIGHTS_ID, change);
        assert!(err.contains(expected), "{tag}: {err}");
    }
}

#[test]
fn a_converter_record_short_of_the_sidecar_or_off_the_work_files_is_refused_by_name() {
    let cases: Vec<(&str, &str, Box<dyn FnOnce(&Path)>, &str)> = vec![
        (
            "divisor",
            WEIGHTS_ID,
            Box::new(|work: &Path| {
                edit_record(work, |r| r["local_nvfp4"]["parents"]["text/layers/63/mlp/down"]["weight_scale_divisor"] = json!(1.0))
            }),
            "text/layers/63/mlp/down stores weight divisor",
        ),
        (
            "unrecorded",
            WEIGHTS_ID,
            Box::new(|work: &Path| {
                edit_record(work, |r| {
                    r["local_nvfp4"]["parents"].as_object_mut().unwrap().remove("dflash2/selector/successor");
                })
            }),
            "converter.json's local_nvfp4.parents does not record dflash2/selector/successor",
        ),
        (
            "drafter-divisor",
            WEIGHTS_ID,
            Box::new(|work: &Path| {
                edit_record(work, |r| r["local_nvfp4"]["parents"]["dflash2/feature_projection"]["weight_scale_divisor"] = json!(2.0))
            }),
            "converter.json records a weight_scale_divisor for dflash2/feature_projection, which is weight-only",
        ),
        (
            "stray",
            WEIGHTS_ID,
            Box::new(|work: &Path| {
                edit_record(work, |r| r["local_nvfp4"]["parents"]["text/layers/4/gdn/output"] = json!({"weight_scale_divisor": 1.0}))
            }),
            "records text/layers/4/gdn/output, which is not an NVFP4 object of the inventory",
        ),
        (
            "recipe",
            WEIGHTS_ID,
            Box::new(|work: &Path| edit_record(work, |r| { r.as_object_mut().unwrap().remove("recipe_id"); })),
            "converter.json has no recipe_id",
        ),
        (
            "license",
            WEIGHTS_ID,
            Box::new(|work: &Path| edit_record(work, |r| { r["source"].as_object_mut().unwrap().remove("license"); })),
            "converter.json has no source.license",
        ),
        (
            "identity",
            "nvfp4full-other",
            Box::new(|_: &Path| {}),
            "converter.json records identity qwen3.8-27b/nvfp4full-fixture; this run packs qwen3.8-27b/nvfp4full-other",
        ),
    ];
    for (tag, weights_id, change, expected) in cases {
        let err = refusal(tag, weights_id, change);
        assert!(err.contains(expected), "{tag}: {err}");
    }
}

#[test]
fn the_packer_waits_for_the_whole_tree_and_writes_nothing_meanwhile() {
    let scratch = Scratch::new("waiting");
    let work = work_tree(&scratch.0);
    let artifact = scratch.0.join(ARTIFACT);
    let converter = std::fs::read(work.join("converter.json")).unwrap();
    std::fs::remove_file(work.join("converter.json")).unwrap();
    std::fs::remove_file(work.join("vision/DONE")).unwrap();
    // Even a run that keeps its work files waits for converter.json.
    for keep_work in [false, true] {
        let options = PackOptions { keep_work, ..options(&work, &artifact, WEIGHTS_ID) };
        let outcome = pack(&options, &mut |_| {}).unwrap();
        assert_eq!(outcome, PackOutcome::Waiting { appended: 0, total: 5, next: "converter.json".into() });
    }
    std::fs::write(work.join("converter.json"), converter).unwrap();
    let outcome = pack(&options(&work, &artifact, WEIGHTS_ID), &mut |_| {}).unwrap();
    assert_eq!(outcome, PackOutcome::Waiting { appended: 0, total: 5, next: "vision".into() });
    for written in [artifact.clone(), state_path(&artifact), sidecar_path(&artifact)] {
        assert!(!written.exists(), "{} was written", written.display());
    }
    mark_done(&work.join("vision"));
    pack_tree(&work, &artifact, WEIGHTS_ID);
}

#[test]
fn the_pack_binary_names_the_family_and_needs_a_weights_id_and_an_out() {
    let scratch = Scratch::new("binary");
    let work = work_tree(&scratch.0);
    let artifact = scratch.0.join(ARTIFACT);
    let run = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_ignis-artifact-pack"))
            .arg("--work")
            .arg(&work)
            .args(["--family", "qwen3.8-27b-nvfp4", "--geometry", "fixture", "--keep-work"])
            .args(args)
            .output()
            .unwrap()
    };
    let out = artifact.to_str().unwrap();
    for (args, missing) in [(vec!["--out", out], "needs --weights-id"), (vec!["--weights-id", WEIGHTS_ID], "needs --out")] {
        let refused = run(&args);
        let stderr = String::from_utf8_lossy(&refused.stderr);
        assert_eq!(refused.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains(&format!("--family qwen3.8-27b-nvfp4 {missing}")), "{stderr}");
        // The usage names the family and its work tree.
        assert!(stderr.contains("--family qwen3.8-27b-nvfp4: --work <NVFP4 work tree: frontend/ text/ mtp/ vision/ dflash2/"), "{stderr}");
        assert!(stderr.contains("docs/formats/27b-nvfp4-work-tree.md"), "{stderr}");
    }
    assert!(!artifact.exists());

    let done = run(&["--out", out, "--weights-id", WEIGHTS_ID]);
    let stdout = String::from_utf8_lossy(&done.stdout);
    assert!(done.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&done.stderr));
    assert!(stdout.contains("1325 objects"), "{stdout}");
    let reader = Reader::open(&artifact).unwrap();
    assert_eq!((reader.identity().model_id.as_str(), reader.identity().weights_id.as_str()), ("qwen3.8-27b", WEIGHTS_ID));
}

/// The 27B families have no Flash-Next geometry: the NVFP4 one takes its own
/// `--geometry` and refuses `--layers` and `--experts`, the trellis one all three.
#[test]
fn the_pack_binary_refuses_flash_next_flags_for_the_27b_families() {
    let scratch = Scratch::new("binary-flags");
    let artifact = scratch.0.join(ARTIFACT);
    for (family, flag, refused) in [
        ("qwen3.8-27b-nvfp4", ["--layers", "2"], "--layers and --experts are Flash-Next's, not this --family's"),
        ("qwen3.8-27b-nvfp4", ["--experts", "2"], "--layers and --experts are Flash-Next's, not this --family's"),
        ("qwen3.8-27b-trellis", ["--geometry", "fixture"], "--layers, --experts and --geometry are Flash-Next's, not this --family's"),
        ("qwen3.8-27b-trellis", ["--layers", "2"], "--layers, --experts and --geometry are Flash-Next's, not this --family's"),
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_ignis-artifact-pack"))
            .arg("--work")
            .arg(scratch.0.join("work"))
            .args(["--family", family, "--out"])
            .arg(&artifact)
            .args(["--weights-id", WEIGHTS_ID])
            .args(flag)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{family} {flag:?}: {stderr}");
        assert!(stderr.starts_with(&format!("{refused}\n")), "{family} {flag:?}: {stderr}");
    }
    assert!(!artifact.exists());
}
