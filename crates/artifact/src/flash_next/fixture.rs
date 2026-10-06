//! A small Flash-Next-shaped artifact for CPU tests (spec flash-next/01's
//! testing decisions): two layers, eight experts over every K class, a
//! 1,000-row n-gram table.
//!
//! It is made the way the real one is: a [`WorkTree`] writes what the
//! converter writes (layout.md: one directory per unit, `tensors.json`,
//! `experts.bin` + `experts.idx`, table shards, `DONE` files with SHA-256s)
//! and the packer assembles the container through the crate's writer. The
//! payload bytes are a pattern of the object's name ([`pattern`]), so a test
//! can tell every object's bytes apart; nothing in them decodes.
//!
//! The same tree can hold the MTP head's work unit (`mtp/`, layout.md §13),
//! packed into a companion container of its own.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    expert_name, global_entries, layer_entries, mtp_entries, mtp_expert_name, record_bytes, Entry,
    FlashNextGeometry, Projection, ShapeRule, TrellisK, FRONTEND_FILES,
};
use crate::packer::{pack, PackOptions, PackOutcome};
use crate::{fail, tensor_encoded_size, Result};

/// The fixture's stand-in payload for `name`: its bytes, repeated to `len`.
pub fn pattern(name: &str, len: u64) -> Vec<u8> {
    name.bytes().cycle().take(len as usize).collect()
}

/// The fixture's K map: experts cycle through the four widths, the two
/// projections of one expert start apart, so all eight classes occur.
pub fn fixture_k(layer: usize, expert: u64, projection: Projection) -> TrellisK {
    let at = layer as u64 + expert + 2 * u64::from(projection.code());
    TrellisK::ALL[(at % 4) as usize]
}

/// The hot-row list the fixture stores (row ids, most frequent first).
pub const HOT_ROWS: [u32; 6] = [17, 3, 999, 0, 512, 64];

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A converter work tree in a temporary directory (removed on drop), and
/// the artifact path beside it.
pub struct WorkTree {
    root: PathBuf,
    pub geometry: FlashNextGeometry,
}

impl WorkTree {
    pub fn new(tag: &str) -> Result<Self> {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "ignis-flash-next-{tag}-{}-{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("work")).map_err(io(&root))?;
        Ok(Self {
            root,
            geometry: FlashNextGeometry::fixture(),
        })
    }

    pub fn work_dir(&self) -> PathBuf {
        self.root.join("work")
    }

    pub fn artifact_path(&self) -> PathBuf {
        self.root.join("qwen3_8_flash_next_fixture-v2.ninfer")
    }

    /// The packer options for this tree (small delete batches, so a test
    /// sees files go one by one).
    pub fn pack_options(&self) -> PackOptions {
        let mut options = PackOptions::new(self.work_dir(), self.artifact_path(), self.geometry.clone());
        options.header_bytes = 1 << 20;
        options.delete_batch_bytes = 1;
        options
    }

    /// The MTP head converter's work tree (layout.md §13.1).
    pub fn mtp_work_dir(&self) -> PathBuf {
        self.root.join("work-mtp")
    }

    pub fn mtp_artifact_path(&self) -> PathBuf {
        self.root.join("qwen3_8_flash_next_mtp_fixture-v2.ninfer")
    }

    /// The packer options for the MTP companion, paired with `main`.
    pub fn mtp_pack_options(&self, main: PathBuf) -> PackOptions {
        let mut options =
            PackOptions::mtp(self.mtp_work_dir(), self.mtp_artifact_path(), self.geometry.clone(), main);
        options.delete_batch_bytes = 1;
        options
    }

    /// Write the MTP unit (its non-expert tensors, its expert files, `DONE`)
    /// and its `converter.json`, naming `main_file` as the main container.
    pub fn write_mtp(&self, main_file: &str) -> Result<()> {
        let dir = self.mtp_work_dir().join("mtp");
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        write_tensors(&dir, &mtp_entries(&self.geometry))?;
        let (experts, index) = experts_files(&self.geometry, 0, mtp_expert_name);
        write(&dir.join("experts.bin"), &experts)?;
        write(&dir.join("experts.idx"), &index)?;
        mark_done(&dir)?;
        let record = json!({
            "schema": "flash-next-mtp-converter-v1",
            "status": "complete",
            "source": {"repo": "fixture", "revision": "fixture"},
            "pair": {"main": {"file": main_file, "file_sha256": "fixture"}},
            "experts_bin": {"bytes": experts.len(), "sha256": hex_digest(&experts)},
        });
        write(&self.mtp_work_dir().join("converter.json"), record.to_string().as_bytes())
    }

    /// Mark the MTP unit complete again (after a test edited it).
    pub fn mark_mtp_done(&self) -> Result<()> {
        mark_done(&self.mtp_work_dir().join("mtp"))
    }

    /// Write every unit and `converter.json`.
    pub fn write_all(&self) -> Result<()> {
        for unit in crate::packer::unit_names(self.geometry.layers) {
            self.write_unit(&unit)?;
        }
        self.write_converter_json()
    }

    /// Write one unit's files, then its `DONE`.
    pub fn write_unit(&self, unit: &str) -> Result<()> {
        self.write_unit_files(unit)?;
        self.mark_done(unit)
    }

    /// Write one unit's files without `DONE` (a unit still being converted).
    pub fn write_unit_files(&self, unit: &str) -> Result<()> {
        let dir = self.work_dir().join(unit);
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        let g = &self.geometry;
        match unit {
            "frontend" => {
                for file in FRONTEND_FILES {
                    write(&dir.join(file), json!({"fixture": file}).to_string().as_bytes())?;
                }
            }
            "global" => write_tensors(&dir, &global_entries(g))?,
            "ngram" => {
                let table = dir.join("table");
                std::fs::create_dir_all(&table).map_err(io(&table))?;
                let name = super::ngram_table_name(g);
                let row_bytes = crate::row_interleaved_geometry(
                    crate::NumericFormat::Q4G32F16S,
                    &[1, g.ngram_head_dim],
                )?
                .row_bytes;
                let bytes = pattern(&name, g.ngram_rows * row_bytes);
                // Two shards, split on a row boundary.
                let split = (g.ngram_rows / 2 * row_bytes) as usize;
                write(&table.join("shard_000.int4"), &bytes[..split])?;
                write(&table.join("shard_001.int4"), &bytes[split..])?;
                mark_done(&table)?;
                let words = |values: Vec<i64>| -> Vec<u8> {
                    values.iter().flat_map(|v| v.to_le_bytes()).collect()
                };
                let heads = g.ngram_heads() as i64;
                write(&dir.join("layer_multipliers.i64"), &words(vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]))?;
                // Prime head ranges that fit the table (the checkpoint's are primes
                // above 20M; the hashing only needs each range inside the table).
                let sizes: Vec<i64> = (0..heads).map(|h| [491, 499][h as usize % 2]).collect();
                let offsets: Vec<i64> = sizes.iter().scan(0, |at, &size| { let first = *at; *at += size; Some(first) }).collect();
                write(&dir.join("ngram_heads_vocab_sizes.i64"), &words(sizes))?;
                write(&dir.join("ngram_heads_offsets.i64"), &words(offsets))?;
                let hot: Vec<u8> = HOT_ROWS.iter().flat_map(|r| r.to_le_bytes()).collect();
                write(&dir.join("hot_rows.u32"), &hot)?;
                let manifest = json!({
                    "rows": HOT_ROWS.len(),
                    "table_rows": g.ngram_rows,
                    "shards": 2,
                    "complete": true,
                });
                write(&dir.join("hot_rows.json"), manifest.to_string().as_bytes())?;
            }
            _ => {
                let layer: usize = unit
                    .strip_prefix("layers/L")
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| fail(format!("unknown unit {unit}")))?;
                write_tensors(&dir, &layer_entries(g, layer))?;
                let (experts, index) =
                    experts_files(g, layer, |expert, projection| expert_name(layer, expert, projection));
                write(&dir.join("experts.bin"), &experts)?;
                write(&dir.join("experts.idx"), &index)?;
            }
        }
        Ok(())
    }

    /// Mark a unit complete, the converter's way: `DONE` last.
    pub fn mark_done(&self, unit: &str) -> Result<()> {
        mark_done(&self.work_dir().join(unit))
    }

    /// Add a tensor the inventory does not know to a unit's `tensors.json`
    /// (before its `DONE`): the ADR 0002 drift a bind must refuse.
    pub fn add_stray_tensor(&self, unit: &str, name: &str) -> Result<()> {
        let dir = self.work_dir().join(unit);
        let path = dir.join("tensors.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(&path).map_err(io(&path))?)
            .map_err(|e| fail(format!("{}: {e}", path.display())))?;
        let entry = super::bf16(name.to_owned(), &[4]);
        let tensors = value["tensors"].as_array_mut().ok_or_else(|| fail("tensors.json has no tensors"))?;
        tensors.push(tensor_record(&dir, &entry)?);
        write(&path, value.to_string().as_bytes())
    }

    /// The converter's run record (a minimal one: the packer merges it):
    /// its status and each layer's `experts.bin` size and SHA-256.
    pub fn write_converter_json(&self) -> Result<()> {
        let experts_bin: Vec<Value> = (0..self.geometry.layers)
            .map(|layer| {
                let (experts, _) =
                    experts_files(&self.geometry, layer, |expert, projection| expert_name(layer, expert, projection));
                json!({"layer": layer, "bytes": experts.len(), "sha256": hex_digest(&experts)})
            })
            .collect();
        let record = json!({
            "schema": "flash-next-converter-v1",
            "status": "complete",
            "source": {"repo": "fixture", "revision": "fixture"},
            "experts_bin": experts_bin,
        });
        write(&self.work_dir().join("converter.json"), record.to_string().as_bytes())
    }
}

impl Drop for WorkTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A packed fixture artifact (its work tree, emptied by the packer, owns
/// the temporary directory).
pub struct FixtureArtifact {
    pub tree: WorkTree,
    pub path: PathBuf,
}

/// Write the fixture's whole work tree and pack it.
pub fn build(tag: &str) -> Result<FixtureArtifact> {
    let tree = WorkTree::new(tag)?;
    tree.write_all()?;
    build_from(tree)
}

/// Pack an already written work tree.
pub fn build_from(tree: WorkTree) -> Result<FixtureArtifact> {
    match pack(&tree.pack_options(), &mut |_| {})? {
        PackOutcome::Finished { .. } => {}
        other => return Err(fail(format!("the fixture work tree did not pack: {other:?}"))),
    }
    let path = tree.artifact_path();
    Ok(FixtureArtifact { tree, path })
}

/// A layer's `experts.bin` and `experts.idx` (layout.md §4): every
/// record in index order (K from [`fixture_k`] at `layer`), each the
/// pattern of its object name.
fn experts_files(
    g: &FlashNextGeometry,
    layer: usize,
    name: impl Fn(u64, Projection) -> String,
) -> (Vec<u8>, Vec<u8>) {
    let mut experts = Vec::new();
    let mut index = Vec::new();
    for expert in 0..g.experts {
        for projection in Projection::ALL {
            let k = fixture_k(layer, expert, projection);
            let bytes = record_bytes(g, projection, k);
            index.extend_from_slice(&(expert as u16).to_le_bytes());
            index.push(projection.code());
            index.push(k.k2());
            index.extend_from_slice(&(bytes as u32).to_le_bytes());
            index.extend_from_slice(&(experts.len() as u64).to_le_bytes());
            experts.extend(pattern(&name(expert, projection), bytes));
        }
    }
    (experts, index)
}

/// Write each entry's payload file and the unit's `tensors.json`.
fn write_tensors(dir: &Path, entries: &[Entry]) -> Result<()> {
    let records = entries
        .iter()
        .map(|entry| tensor_record(dir, entry))
        .collect::<Result<Vec<_>>>()?;
    write(&dir.join("tensors.json"), json!({"tensors": records}).to_string().as_bytes())
}

/// Write one tensor's payload file and return its `tensors.json` record.
fn tensor_record(dir: &Path, entry: &Entry) -> Result<Value> {
    let ShapeRule::Exact(shape) = &entry.shape else {
        return Err(fail(format!("{} has no fixed shape", entry.name)));
    };
    let bytes = tensor_encoded_size(entry.layout, entry.format, shape)?;
    let payload = pattern(&entry.name, bytes);
    let file = format!("{}.bin", entry.name);
    write(&dir.join(&file), &payload)?;
    Ok(json!({
        "name": entry.name,
        "file": file,
        "format": entry.format.name(),
        "layout": entry.layout.name(),
        "shape": shape,
        "bytes": bytes,
        "sha256": hex_digest(&payload),
    }))
}

/// `DONE`: every other file of `dir` with its size and SHA-256.
fn mark_done(dir: &Path) -> Result<()> {
    let mut files = serde_json::Map::new();
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .map_err(io(dir))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "DONE" && n != "DONE.tmp")
        .collect();
    names.sort();
    for name in names {
        let path = dir.join(&name);
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        files.insert(name, json!({"bytes": bytes.len(), "sha256": hex_digest(&bytes)}));
    }
    write(&dir.join("DONE"), json!({"files": files}).to_string().as_bytes())
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).map_err(io(path))
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> crate::ArtifactError + '_ {
    move |e| fail(format!("{}: {e}", path.display()))
}
