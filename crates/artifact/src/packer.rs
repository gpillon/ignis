//! The packer: assembles the Flash-Next `.ninfer` v2 container from the
//! converter's work tree (spec flash-next/01, "Who writes the container";
//! the work-file contract is `docs/formats/flash-next-layout.md`).
//!
//! The converter never writes the container format. Its work tree holds one
//! directory per **unit** — `frontend/`, `global/`, `ngram/` (with the table
//! shards in `ngram/table/`), `layers/L00` .. — each complete when its
//! `DONE` file exists (written last; it lists every other file with its size
//! and SHA-256), plus `converter.json`, the converter's half of the sidecar.
//!
//! [`pack`] appends the units in one fixed order (frontend, global, ngram,
//! the layers) through the crate's [`ContainerWriter`] and stops at the first
//! unit that is not complete yet. Every file is checked against `DONE` (size
//! before a byte is appended, SHA-256 as it streams in) and every tensor
//! against the reader's own size rule before it is accepted. Consumed files
//! are deleted once the container is synced and the packer's state
//! (`<artifact>.pack-state.json`) records them — in batches inside a unit
//! (the 29 GB table goes shard by shard), and the whole unit directory at
//! its end — so the disk peak is the container plus the work files not yet
//! appended. An interrupted run resumes from the state, cutting the
//! container back to what the state records.
//!
//! The converter reads its own work files until it ends (a resumed
//! conversion replays finished layers from them, its self-check decodes
//! `experts.bin`), so the packer deletes nothing before `converter.json`
//! exists: until then it waits. It packs only a `converter.json` whose
//! `status` it accepts (`complete`; a dry run or a fixture is accepted only
//! when asked for). A `keep_work` run deletes nothing and needs no
//! `converter.json`, but it keeps two copies on disk, so it refuses to start
//! unless the volume has room for both: it is for test packs, not the
//! documented flow.
//!
//! After the last layer the directory is written, the container is reopened
//! with the reader, and `<artifact>.conversion.json` gets the whole-file
//! invariants `Sidecar::load` requires (`recipe_id`, `artifact.bytes`,
//! `objects.count`) merged with `converter.json`'s fields; a later run
//! merges a `converter.json` written after the container was finished.
//!
//! The same mechanics pack the MTP head's companion container
//! ([`Family::Mtp`], flash-next-layout.md §13): one unit, `mtp/`, and a sidecar that
//! pins the main container it belongs to (`pair.main`, read from the main
//! container itself, so its hash is the one the binder computes).
//!
//! And the image model's container ([`Family::QwenImage21`], spec
//! image-model/01): `frontend/`, then four units of tensors.json tensors
//! ([`crate::qwen_image::WORK_UNITS`]), with no Flash-Next geometry.
//!
//! And the 27B's 16 GB artifact ([`Family::Qwen27bTrellis`], spec
//! 27b-16gb/01): `frontend/`, then the tensors.json units of
//! [`TRELLIS_27B_UNITS`], each object under its unit's prefix, written by
//! `tools/27b-16gb-converter`.
//!
//! And an NVFP4 container of the 27B ([`Family::Qwen27bNvfp4`], GitHub
//! #467): the same five units, planned from the inventory
//! ([`crate::nvfp4_27b::inventory`]) rather than from the converter's lists,
//! so the container holds exactly the 1,325 objects in the served order
//! under the weights id the run is given. It packs only a whole tree: until
//! every unit and `converter.json` are complete it waits, and a missing
//! object, one of the wrong format, layout or shape, an object the
//! inventory does not name, or a `converter.json` whose record is short of
//! the sidecar's fields or whose NVFP4 divisors are not the work files' is
//! refused by name before anything is created. The work tree's contract is
//! `docs/formats/27b-nvfp4-work-tree.md`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::flash_next::{
    expert_name, mtp_expert_name, FlashNextGeometry, Projection, TrellisK, MTP_MODEL_ID, MTP_PREFIX,
};
use crate::writer::{entry_json_bytes, ContainerWriter, WriterState};
use crate::{
    fail, require_string, tensor_encoded_size, ArtifactIdentity, NumericFormat, Object, Reader,
    ResourceDescriptor, ResourceEncoding, Result, StorageLayout, TensorDescriptor, PREFIX_BYTES,
};

/// The default header reservation: the Flash-Next directory (~49,000
/// objects) is ~10 MB of JSON; 64 MiB is 0.1% of the ~71 GB artifact.
pub const DEFAULT_HEADER_BYTES: u64 = 64 << 20;

/// Consumed work files are deleted (after a sync and a state write) once
/// this many bytes of them are pending.
pub const DEFAULT_DELETE_BATCH_BYTES: u64 = 1 << 30;

/// The container's file name, beside the converter's `work/` directory
/// (flash-next-layout.md §1; the packer's default `--out`).
pub const ARTIFACT_FILE_NAME: &str = "qwen3_8_flash_next_trellis_a25-v2.ninfer";

/// The MTP companion container's file name, beside the main one
/// (flash-next-layout.md §13.1).
pub const MTP_ARTIFACT_FILE_NAME: &str = "qwen3_8_flash_next_mtp_3p0-v2.ninfer";

/// The companion's header reservation: its directory is ~0.2 MB.
pub const MTP_HEADER_BYTES: u64 = 1 << 20;

/// Room a `keep_work` run leaves free beyond its second copy.
const KEEP_WORK_MARGIN_BYTES: u64 = 2 << 30;

/// The default artifact identity (flash-next-layout.md: never mistakable for a 27B).
pub fn default_identity() -> ArtifactIdentity {
    ArtifactIdentity {
        model_id: crate::flash_next::MODEL_ID.into(),
        weights_id: "trellis-a25-fp8rows-q4g32-de4b8e4".into(),
    }
}

/// The image model's container file name, beside the converter's `work/`
/// (`tools/image-converter`; "Qwen" is never its primary name, ADR 0050).
pub const QWEN_IMAGE_ARTIFACT_FILE_NAME: &str = "ignis_image_2_1_turbo-v2.ninfer";

/// The image model's header reservation: its directory is ~0.4 MB.
pub const QWEN_IMAGE_HEADER_BYTES: u64 = 4 << 20;

/// The image model's identity: FP8 row-scaled linears, BF16 elsewhere, the
/// vision tower's matrices W8, from revision `fc13ba19`.
pub fn qwen_image_identity() -> ArtifactIdentity {
    ArtifactIdentity {
        model_id: crate::qwen_image::MODEL_ID.into(),
        weights_id: "fp8rows-bf16-fc13ba1".into(),
    }
}

/// The 27B's 16 GB artifact's file name, beside the converter's `work/`
/// (spec 27b-16gb/01).
pub const TRELLIS_27B_ARTIFACT_FILE_NAME: &str = "qwen3_8_27b_trellis_sc30-v2.ninfer";

/// The 16 GB artifact's header reservation: its directory is ~0.25 MB.
pub const TRELLIS_27B_HEADER_BYTES: u64 = 4 << 20;

/// The 16 GB artifact's tensor units after `frontend/`, in container order
/// (the served artifact's): each object's name starts with its unit.
pub const TRELLIS_27B_UNITS: &[&str] = &["text", "mtp", "vision", "dflash2"];

/// The 16 GB artifact's identity: the 27B family, its body the public
/// `SC_3.00bpw_H4` quant's trellis (ADR 0060 §4).
pub fn trellis_27b_identity() -> ArtifactIdentity {
    ArtifactIdentity {
        model_id: "qwen3.8-27b".into(),
        weights_id: "trellis-sc30".into(),
    }
}

/// An NVFP4 27B container's header reservation: its directory is ~0.3 MB.
pub const NVFP4_27B_HEADER_BYTES: u64 = 4 << 20;

/// The MTP companion container's identity (flash-next-layout.md §13.2).
pub fn mtp_identity() -> ArtifactIdentity {
    ArtifactIdentity {
        model_id: MTP_MODEL_ID.into(),
        weights_id: "mtp-trellis-a30-fp8rows-de4b8e4".into(),
    }
}

/// Which container a work tree packs into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Family {
    /// The model: frontend, globals, the n-gram unit, the layers.
    Main,
    /// The MTP head's companion (flash-next-layout.md §13): the one unit `mtp/`,
    /// pinned to the main container `pair_main`.
    Mtp { pair_main: PathBuf },
    /// The image model (spec image-model/01): `frontend/` and the four
    /// tensor units of [`crate::qwen_image::WORK_UNITS`].
    QwenImage21,
    /// The 27B's 16 GB artifact (spec 27b-16gb/01): `frontend/` and the
    /// tensor units of [`TRELLIS_27B_UNITS`].
    Qwen27bTrellis,
    /// An NVFP4 container of the 27B (GitHub #467): the units of
    /// [`crate::nvfp4_27b::UNITS`], holding the inventory at `geometry`.
    Qwen27bNvfp4 { geometry: crate::nvfp4_27b::Geometry },
}

/// What one [`pack`] call is asked to do.
#[derive(Debug, Clone)]
pub struct PackOptions {
    /// The converter's work directory (`.../work`).
    pub work_dir: PathBuf,
    /// The container to write.
    pub artifact: PathBuf,
    /// Flash-Next's shapes and layer count (a dry run packs fewer layers);
    /// `None` for the image model, whose units are fixed.
    pub geometry: Option<FlashNextGeometry>,
    pub identity: ArtifactIdentity,
    /// Header reservation for a new container (a resume keeps its own).
    pub header_bytes: u64,
    /// Keep every work file (test packs; refused without room for two
    /// copies).
    pub keep_work: bool,
    pub delete_batch_bytes: u64,
    /// The `converter.json` statuses to pack (`complete`; a dry run's
    /// `dry-run` or a fixture's `fixture` only when asked for).
    pub accept_status: Vec<String>,
    pub family: Family,
}

impl PackOptions {
    /// The defaults for `work_dir` → `artifact` at `geometry`.
    pub fn new(work_dir: PathBuf, artifact: PathBuf, geometry: FlashNextGeometry) -> Self {
        Self {
            work_dir,
            artifact,
            geometry: Some(geometry),
            identity: default_identity(),
            header_bytes: DEFAULT_HEADER_BYTES,
            keep_work: false,
            delete_batch_bytes: DEFAULT_DELETE_BATCH_BYTES,
            accept_status: vec!["complete".into()],
            family: Family::Main,
        }
    }

    /// The defaults for an MTP companion: `work_dir` → `artifact`, pinned to
    /// the main container `pair_main`.
    pub fn mtp(work_dir: PathBuf, artifact: PathBuf, geometry: FlashNextGeometry, pair_main: PathBuf) -> Self {
        Self {
            identity: mtp_identity(),
            header_bytes: MTP_HEADER_BYTES,
            family: Family::Mtp { pair_main },
            ..Self::new(work_dir, artifact, geometry)
        }
    }

    /// The defaults for the image model: `work_dir` → `artifact`.
    pub fn qwen_image(work_dir: PathBuf, artifact: PathBuf) -> Self {
        Self {
            work_dir,
            artifact,
            geometry: None,
            identity: qwen_image_identity(),
            header_bytes: QWEN_IMAGE_HEADER_BYTES,
            keep_work: false,
            delete_batch_bytes: DEFAULT_DELETE_BATCH_BYTES,
            accept_status: vec!["complete".into()],
            family: Family::QwenImage21,
        }
    }

    /// The defaults for the 27B's 16 GB artifact: `work_dir` → `artifact`.
    pub fn trellis_27b(work_dir: PathBuf, artifact: PathBuf) -> Self {
        Self {
            work_dir,
            artifact,
            geometry: None,
            identity: trellis_27b_identity(),
            header_bytes: TRELLIS_27B_HEADER_BYTES,
            keep_work: false,
            delete_batch_bytes: DEFAULT_DELETE_BATCH_BYTES,
            accept_status: vec!["complete".into()],
            family: Family::Qwen27bTrellis,
        }
    }

    /// The defaults for an NVFP4 container of the 27B: `work_dir` →
    /// `artifact` under `weights_id`, which has no default (the content hash
    /// covers the identity and the directory, not the payload, so the
    /// weights id is what tells one build of this layout from another).
    pub fn nvfp4_27b(work_dir: PathBuf, artifact: PathBuf, weights_id: &str, geometry: crate::nvfp4_27b::Geometry) -> Self {
        Self {
            work_dir,
            artifact,
            geometry: None,
            identity: crate::nvfp4_27b::identity(weights_id),
            header_bytes: NVFP4_27B_HEADER_BYTES,
            keep_work: false,
            delete_batch_bytes: DEFAULT_DELETE_BATCH_BYTES,
            accept_status: vec!["complete".into()],
            family: Family::Qwen27bNvfp4 { geometry },
        }
    }

    /// The layer count the state records: Flash-Next's, 0 for the image model.
    fn layers(&self) -> usize {
        self.geometry.as_ref().map_or(0, |g| g.layers)
    }
}

/// Where a [`pack`] call left the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackOutcome {
    /// Some units are in; `next` is not complete yet. Run again later.
    Waiting {
        appended: usize,
        total: usize,
        next: String,
    },
    /// Every unit is in, the directory is written and verified.
    Finished {
        file_bytes: u64,
        object_count: usize,
        /// Whether `converter.json` was there to merge into the sidecar.
        converter_merged: bool,
    },
}

/// The packer state file beside `artifact`.
pub fn state_path(artifact: &Path) -> PathBuf {
    with_suffix(artifact, ".pack-state.json")
}

/// The sidecar the loader reads beside `artifact`.
pub fn sidecar_path(artifact: &Path) -> PathBuf {
    with_suffix(artifact, ".conversion.json")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// The units of `family` in container order.
pub fn family_units(family: &Family, layers: usize) -> Vec<String> {
    match family {
        Family::Main => unit_names(layers),
        Family::Mtp { .. } => vec!["mtp".into()],
        Family::QwenImage21 => {
            std::iter::once("frontend").chain(crate::qwen_image::WORK_UNITS).map(String::from).collect()
        }
        Family::Qwen27bTrellis => {
            std::iter::once("frontend").chain(TRELLIS_27B_UNITS.iter().copied()).map(String::from).collect()
        }
        Family::Qwen27bNvfp4 { .. } => crate::nvfp4_27b::UNITS.map(String::from).to_vec(),
    }
}

/// The main container's units in container order.
pub fn unit_names(layers: usize) -> Vec<String> {
    let mut units: Vec<String> = ["frontend", "global", "ngram"].map(String::from).to_vec();
    units.extend((0..layers).map(|layer| format!("layers/L{layer:02}")));
    units
}

/// The packer's persisted progress.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PackState {
    layers: usize,
    units_done: Vec<String>,
    ngram_source_digest: Option<String>,
    finished: bool,
    writer: WriterState,
}

/// Append every complete unit of `options.work_dir` to the container, and
/// finish it after the last layer. Safe to call again at any point.
///
/// `progress` receives one line per appended unit.
pub fn pack(options: &PackOptions, progress: &mut dyn FnMut(&str)) -> Result<PackOutcome> {
    let _lock = lock(&options.artifact)?;
    let state_file = state_path(&options.artifact);
    let units = family_units(&options.family, options.layers());
    let pair = match &options.family {
        Family::Mtp { pair_main } => Some(pair_record(pair_main)?),
        Family::Main | Family::QwenImage21 | Family::Qwen27bTrellis | Family::Qwen27bNvfp4 { .. } => None,
    };

    let mut state = if state_file.exists() {
        let state = read_state(&state_file)?;
        if state.writer.identity != options.identity {
            return Err(fail(format!(
                "{} records identity {}/{}, this run asks for {}/{}",
                state_file.display(),
                state.writer.identity.model_id,
                state.writer.identity.weights_id,
                options.identity.model_id,
                options.identity.weights_id
            )));
        }
        if state.layers != options.layers() {
            return Err(fail(format!(
                "{} packs {} layers, this run asks for {}",
                state_file.display(),
                state.layers,
                options.layers()
            )));
        }
        if !units.starts_with(&state.units_done) {
            return Err(fail(format!(
                "{} records units {:?} out of order",
                state_file.display(),
                state.units_done
            )));
        }
        state
    } else {
        if options.artifact.exists() {
            return Err(fail(format!(
                "{} exists without {}: refusing to overwrite it (remove it to pack from scratch)",
                options.artifact.display(),
                state_file.display()
            )));
        }
        if let Family::Qwen27bNvfp4 { .. } = options.family {
            if let Some(waiting) = preflight_nvfp4_27b(options, &units)? {
                return Ok(waiting);
            }
        }
        let writer =
            ContainerWriter::create(&options.artifact, options.identity.clone(), options.header_bytes)?;
        let state = PackState {
            layers: options.layers(),
            units_done: Vec::new(),
            ngram_source_digest: None,
            finished: false,
            writer: writer.state().clone(),
        };
        write_state(&state_file, &state)?;
        state
    };

    // The converter reads its own work files until it ends: a resumed run
    // replays finished layers from them, and the end-of-run self-check
    // decodes records from `experts.bin`. Nothing may be deleted before
    // `converter.json` says it is done, so a run that deletes waits for it
    // before any deletion path below (a `keep_work` run deletes nothing).
    let converter = options.work_dir.join("converter.json");
    if !options.keep_work && !converter.exists() {
        return Ok(PackOutcome::Waiting {
            appended: state.units_done.len(),
            total: units.len(),
            next: "converter.json".into(),
        });
    }
    if converter.exists() {
        check_status(&converter, &read_json(&converter)?, &options.accept_status)?;
        if let Some(pair) = &pair {
            check_pair(&read_json(&converter)?, pair)?;
        }
    }
    if options.keep_work && !state.finished {
        let mut needed = 0u64;
        for unit in &units[state.units_done.len()..] {
            needed += unit_bytes(&options.work_dir.join(unit))?;
        }
        let volume = options.artifact.parent().unwrap_or(Path::new("."));
        check_keep_work_room(needed, free_bytes(volume)?)?;
    }

    // A unit recorded as appended whose work files are still there: a
    // `keep_work` run, or a run stopped between the state write and the
    // deletion.
    if !options.keep_work {
        for unit in &state.units_done {
            remove_dir(&options.work_dir.join(unit))?;
        }
    }

    if state.finished {
        let reader = Reader::open(&options.artifact)?;
        let converter_merged =
            write_sidecar(&options.artifact, &options.work_dir, reader.file_bytes(), reader.objects().len(), pair.as_ref(), state.ngram_source_digest.as_deref())?;
        return Ok(PackOutcome::Finished {
            file_bytes: reader.file_bytes(),
            object_count: reader.objects().len(),
            converter_merged,
        });
    }

    let mut writer = ContainerWriter::resume(&options.artifact, state.writer.clone())?;
    let mut largest_unit_directory = 0u64;
    for (position, unit) in units.iter().enumerate().skip(state.units_done.len()) {
        let Some(plan) = plan_unit(&options.work_dir, unit, &options.family, options.geometry.as_ref())? else {
            return Ok(PackOutcome::Waiting {
                appended: position,
                total: units.len(),
                next: unit.clone(),
            });
        };
        // The directory must fit the reservation with this unit in and the
        // remaining units at the largest unit's size so far: checked before
        // a byte of the unit is appended, so a failure deletes nothing.
        let unit_directory = plan.directory_bound();
        largest_unit_directory = largest_unit_directory.max(unit_directory);
        let remaining = (units.len() - position - 1) as u64;
        let projected =
            writer.state().directory_bytes() + unit_directory + remaining * largest_unit_directory;
        let room = writer.state().header_bytes - PREFIX_BYTES;
        if projected > room {
            return Err(fail(format!(
                "the directory would reach ~{projected} bytes with {unit} and {remaining} more units; \
                 the header reserves {room} (pack from scratch with a larger header)"
            )));
        }
        let started = std::time::Instant::now();
        let cursor_before = writer.state().cursor;
        append_unit(&mut writer, &mut state, &state_file, &plan, options)?;

        writer.sync()?;
        if unit == "ngram" { state.ngram_source_digest = Some(ngram_source_digest(&plan)); }
        state.units_done.push(unit.clone());
        state.writer = writer.state().clone();
        write_state(&state_file, &state)?;
        if !options.keep_work {
            remove_dir(&plan.dir)?;
        }
        progress(&format!(
            "appended {unit} ({}/{}): {} objects, {} bytes, {:.1} s",
            position + 1,
            units.len(),
            plan.objects.len(),
            writer.state().cursor - cursor_before,
            started.elapsed().as_secs_f64()
        ));
    }

    let object_count = writer.state().objects.len();
    let file_bytes = writer.finish()?;
    let reader = Reader::open(&options.artifact)?;
    if reader.file_bytes() != file_bytes || reader.objects().len() != object_count {
        return Err(fail(format!(
            "the written container reads back as {} objects / {} bytes, expected {object_count} / {file_bytes}",
            reader.objects().len(),
            reader.file_bytes()
        )));
    }
    let converter_merged =
        write_sidecar(&options.artifact, &options.work_dir, file_bytes, object_count, pair.as_ref(), state.ngram_source_digest.as_deref())?;
    state.finished = true;
    write_state(&state_file, &state)?;
    progress(&format!("finished: {object_count} objects, {file_bytes} bytes"));
    Ok(PackOutcome::Finished {
        file_bytes,
        object_count,
        converter_merged,
    })
}

// ---------------------------------------------------------------------------
// Unit plans
// ---------------------------------------------------------------------------

/// A work file a unit consumes, as its `DONE` records it.
#[derive(Debug, Clone)]
struct UnitFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

/// A byte range of one unit file.
#[derive(Debug, Clone, Copy)]
struct Slice {
    file: usize,
    offset: u64,
    len: u64,
}

#[derive(Debug, Clone)]
enum Spec {
    Tensor {
        format: NumericFormat,
        layout: StorageLayout,
        shape: Vec<u64>,
    },
    Resource {
        bytes: u64,
    },
}

#[derive(Debug, Clone)]
struct Planned {
    name: String,
    spec: Spec,
    slices: Vec<Slice>,
}

/// One complete unit, checked against its `DONE` and ready to append.
#[derive(Debug, Clone)]
struct UnitPlan {
    dir: PathBuf,
    files: Vec<UnitFile>,
    objects: Vec<Planned>,
}

/// Composite content identity: ordered object descriptions and verified DONE
/// digests with slice boundaries. No second pass over the 29 GB table.
fn ngram_source_digest(plan: &UnitPlan) -> String {
    let mut hash = Sha256::new();
    hash.update(b"ignis-ngram-source-v1");
    for object in &plan.objects {
        hash.update((object.name.len() as u64).to_le_bytes());
        hash.update(object.name.as_bytes());
        hash.update(format!("{:?}", object.spec).as_bytes());
        for slice in &object.slices {
            hash.update(plan.files[slice.file].sha256.as_bytes());
            hash.update(slice.offset.to_le_bytes());
            hash.update(slice.len.to_le_bytes());
        }
    }
    hex(&hash.finalize())
}

impl UnitPlan {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            files: Vec::new(),
            objects: Vec::new(),
        }
    }

    /// An upper bound of what this unit's objects add to the directory.
    fn directory_bound(&self) -> u64 {
        self.objects
            .iter()
            .map(|object| {
                let placed = match &object.spec {
                    Spec::Tensor { format, layout, shape } => Object::Tensor(TensorDescriptor {
                        name: object.name.clone(),
                        shape: shape.clone(),
                        format: *format,
                        layout: *layout,
                        offset: u64::MAX,
                        bytes: u64::MAX,
                    }),
                    Spec::Resource { .. } => Object::Resource(ResourceDescriptor {
                        name: object.name.clone(),
                        encoding: ResourceEncoding::RawBytesV1,
                        offset: u64::MAX,
                        bytes: u64::MAX,
                    }),
                };
                entry_json_bytes(&placed)
            })
            .sum()
    }

    /// Register `relative` (as `DONE` lists it) and return its index.
    fn file(&mut self, done: &Done, relative: &str) -> Result<usize> {
        let (bytes, sha256) = done.get(relative).ok_or_else(|| {
            fail(format!("{} is not listed in {}'s DONE", relative, self.dir.display()))
        })?;
        let path = self.dir.join(relative);
        // A file an interrupted run consumed and deleted is gone; one that
        // is there must have DONE's size (checked again when it is opened).
        if let Ok(meta) = std::fs::metadata(&path) {
            check_size(&path, meta.len(), *bytes)?;
        }
        self.files.push(UnitFile {
            path,
            bytes: *bytes,
            sha256: sha256.clone(),
        });
        Ok(self.files.len() - 1)
    }

    /// A tensor stored whole in one file.
    fn tensor(
        &mut self,
        done: &Done,
        name: String,
        relative: &str,
        format: NumericFormat,
        layout: StorageLayout,
        shape: Vec<u64>,
    ) -> Result<()> {
        let file = self.file(done, relative)?;
        let len = self.files[file].bytes;
        let encoded = tensor_encoded_size(layout, format, &shape)
            .map_err(|e| fail(format!("tensor {name}: {e}")))?;
        if len != encoded {
            return Err(fail(format!(
                "tensor {name}: {relative} holds {len} bytes, its layout stores {encoded}"
            )));
        }
        self.objects.push(Planned {
            name,
            spec: Spec::Tensor { format, layout, shape },
            slices: vec![Slice { file, offset: 0, len }],
        });
        Ok(())
    }

    /// A `tensors.json` entry, whose `bytes` must be its file's.
    fn listed(&mut self, done: &Done, tensor: Listed) -> Result<()> {
        self.tensor(done, tensor.name.clone(), &tensor.file, tensor.format, tensor.layout, tensor.shape)?;
        let stored = self.files.last().unwrap().bytes;
        if tensor.bytes != stored {
            return Err(fail(format!(
                "tensor {}: tensors.json says {} bytes, the file holds {stored}",
                tensor.name, tensor.bytes
            )));
        }
        Ok(())
    }
}

/// `DONE`'s file list: relative name → (bytes, sha256).
type Done = BTreeMap<String, (u64, String)>;

/// Read a directory's `DONE`, or `None` when the directory is not complete.
fn read_done(dir: &Path) -> Result<Option<Done>> {
    let path = dir.join("DONE");
    if !path.exists() {
        return Ok(None);
    }
    let value = read_json(&path)?;
    let files = value
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| fail(format!("{} has no files object", path.display())))?;
    let mut done = Done::new();
    for (name, entry) in files {
        let bytes = entry
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail(format!("{}: {name} has no bytes", path.display())))?;
        let sha256 = entry
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| fail(format!("{}: {name} has no sha256", path.display())))?;
        done.insert(name.clone(), (bytes, sha256.to_ascii_lowercase()));
    }
    Ok(Some(done))
}

/// Plan one unit of `family`, or `None` when it is not complete yet.
/// `geometry` is Flash-Next's, which its expert and n-gram units need.
fn plan_unit(
    work_dir: &Path,
    unit: &str,
    family: &Family,
    geometry: Option<&FlashNextGeometry>,
) -> Result<Option<UnitPlan>> {
    let dir = work_dir.join(unit);
    let Some(done) = read_done(&dir)? else {
        return Ok(None);
    };
    let flash_next = || geometry.ok_or_else(|| fail(format!("unit {unit} needs Flash-Next's geometry")));
    let mut plan = UnitPlan::new(dir);
    // Every unit of this family, the frontend included, is the inventory's.
    if let Family::Qwen27bNvfp4 { geometry } = family {
        plan_nvfp4_27b(&mut plan, &done, unit, geometry)?;
        return Ok(Some(plan));
    }
    // Only the files flash-next-layout.md names are read; anything else in a unit
    // directory (a layer's `layer.json`, `route_*.npy`) is the converter's
    // own record. A container object the converter forgot to list is caught
    // by the bind, which requires every object of the inventory.
    match unit {
        "frontend" => {
            for name in done.keys() {
                let file = plan.file(&done, name)?;
                let bytes = plan.files[file].bytes;
                plan.objects.push(Planned {
                    name: format!("frontend/{name}"),
                    spec: Spec::Resource { bytes },
                    slices: vec![Slice { file, offset: 0, len: bytes }],
                });
            }
        }
        // the image model's and the 16 GB 27B's units: tensors.json only,
        // every name under the unit's own prefix (`vision/...`, `vae/...`,
        // `text/...`; the 27B's `mtp/...` is not Flash-Next's `mtp` unit)
        _ if *family == Family::Qwen27bTrellis || crate::qwen_image::WORK_UNITS.contains(&unit) => {
            plan_tensors_json(&mut plan, &done, Some(&format!("{unit}/")))?;
        }
        "global" => plan_tensors_json(&mut plan, &done, None)?,
        "mtp" => {
            plan_tensors_json(&mut plan, &done, Some(MTP_PREFIX))?;
            plan_experts(&mut plan, &done, flash_next()?, mtp_expert_name)?;
        }
        "ngram" => {
            if !plan_ngram(&mut plan, &done, flash_next()?)? {
                return Ok(None);
            }
        }
        _ => {
            let layer: usize = unit
                .strip_prefix("layers/L")
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| fail(format!("unknown unit {unit}")))?;
            plan_tensors_json(&mut plan, &done, Some(&format!("layers.{layer}.")))?;
            plan_experts(&mut plan, &done, flash_next()?, |expert, projection| expert_name(layer, expert, projection))?;
        }
    }
    Ok(Some(plan))
}

/// One entry of a `tensors.json` (flash-next-layout.md §6.3).
struct Listed {
    name: String,
    file: String,
    format: NumericFormat,
    layout: StorageLayout,
    shape: Vec<u64>,
    bytes: u64,
}

/// The entries of the unit's `tensors.json`, in its order.
fn read_tensors_json(plan: &UnitPlan, done: &Done) -> Result<Vec<Listed>> {
    if !done.contains_key("tensors.json") {
        return Err(fail(format!("{}'s DONE does not list tensors.json", plan.dir.display())));
    }
    let path = plan.dir.join("tensors.json");
    let value = read_json(&path)?;
    let tensors = value
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| fail(format!("{} has no tensors array", path.display())))?;
    let mut listed = Vec::with_capacity(tensors.len());
    for entry in tensors {
        let name = require_string(&entry["name"], "tensor name")?.to_owned();
        let file = require_string(&entry["file"], "tensor file")?.to_owned();
        let format = NumericFormat::parse(require_string(&entry["format"], "tensor format")?)?;
        let layout = StorageLayout::parse(require_string(&entry["layout"], "tensor layout")?)?;
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| fail(format!("tensor {name} has no shape")))?
            .iter()
            .map(|d| d.as_u64().ok_or_else(|| fail(format!("tensor {name} has a bad shape"))))
            .collect::<Result<Vec<_>>>()?;
        let bytes = entry
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail(format!("tensor {name} has no bytes")))?;
        listed.push(Listed { name, file, format, layout, shape, bytes });
    }
    Ok(listed)
}

/// The tensors a `tensors.json` lists (flash-next-layout.md §6.3), each whole in its
/// file, under its own name, which must start with `prefix` when one is
/// given (a layer's `layers.{L}.`, the MTP head's `mtp.`).
fn plan_tensors_json(plan: &mut UnitPlan, done: &Done, prefix: Option<&str>) -> Result<()> {
    let path = plan.dir.join("tensors.json");
    for listed in read_tensors_json(plan, done)? {
        if let Some(prefix) = prefix {
            if !listed.name.starts_with(prefix) {
                return Err(fail(format!("{} lists {}, outside {prefix}", path.display(), listed.name)));
            }
        }
        plan.listed(done, listed)?;
    }
    Ok(())
}

/// A unit of [`Family::Qwen27bNvfp4`], planned from the inventory: the
/// objects it gives `unit`, in its order, each with the inventory's exact
/// format, layout and shape. A missing object, one described otherwise, and
/// one the inventory does not give the unit are refused by name.
fn plan_nvfp4_27b(plan: &mut UnitPlan, done: &Done, unit: &str, geometry: &crate::nvfp4_27b::Geometry) -> Result<()> {
    use crate::nvfp4_27b::{inventory, Entry};
    let expected: Vec<Entry> = inventory(geometry).into_iter().filter(|e| e.unit() == unit).collect();
    let names: BTreeSet<&str> = expected.iter().map(Entry::name).collect();
    if unit == "frontend" {
        if let Some(file) = done.keys().find(|file| !names.contains(format!("frontend/{file}").as_str())) {
            return Err(fail(format!("frontend/{file} is not an object of the 27B NVFP4 inventory")));
        }
        for entry in &expected {
            let file = &entry.name()["frontend/".len()..];
            if !done.contains_key(file) {
                return Err(fail(format!("the work tree has no {}: frontend/DONE does not list {file}", entry.name())));
            }
            let index = plan.file(done, file)?;
            let bytes = plan.files[index].bytes;
            plan.objects.push(Planned {
                name: entry.name().to_owned(),
                spec: Spec::Resource { bytes },
                slices: vec![Slice { file: index, offset: 0, len: bytes }],
            });
        }
        return Ok(());
    }
    let mut listed: BTreeMap<String, Listed> = BTreeMap::new();
    for tensor in read_tensors_json(plan, done)? {
        if !names.contains(tensor.name.as_str()) {
            return Err(fail(format!(
                "{unit}/tensors.json lists {}, which is not a {unit}/ object of the 27B NVFP4 inventory",
                tensor.name
            )));
        }
        if let Some(twice) = listed.insert(tensor.name.clone(), tensor) {
            return Err(fail(format!("{unit}/tensors.json lists {} twice", twice.name)));
        }
    }
    for entry in expected {
        let Entry::Tensor { name, format, layout, shape } = entry else {
            return Err(fail(format!("{} is a resource outside frontend/", entry.name())));
        };
        let Some(tensor) = listed.remove(&name) else {
            return Err(fail(format!("the work tree has no {name}: {unit}/tensors.json does not list it")));
        };
        if (tensor.format, tensor.layout, &tensor.shape) != (format, layout, &shape) {
            return Err(fail(format!(
                "{name} is {} {} {:?} in {unit}/tensors.json; the 27B NVFP4 inventory has {} {} {shape:?}",
                tensor.format.name(),
                tensor.layout.name(),
                tensor.shape,
                format.name(),
                layout.name()
            )));
        }
        plan.listed(done, tensor)?;
    }
    Ok(())
}

/// A layer's expert projections: `experts.bin` sliced by `experts.idx`
/// (flash-next-layout.md §4), one tensor per projection, named by `name`.
fn plan_experts(
    plan: &mut UnitPlan,
    done: &Done,
    geometry: &FlashNextGeometry,
    name: impl Fn(u64, Projection) -> String,
) -> Result<()> {
    if !done.contains_key("experts.idx") {
        return Err(fail(format!("{}'s DONE does not list experts.idx", plan.dir.display())));
    }
    let index_path = plan.dir.join("experts.idx");
    let index = std::fs::read(&index_path)
        .map_err(|e| fail(format!("read {}: {e}", index_path.display())))?;
    let entries = 2 * geometry.experts as usize;
    if index.len() != 16 * entries {
        return Err(fail(format!(
            "{} is {} bytes, {entries} entries of 16 expected",
            index_path.display(),
            index.len()
        )));
    }
    let file = plan.file(done, "experts.bin")?;
    let mut offset = 0u64;
    for (at, entry) in index.chunks_exact(16).enumerate() {
        let expert = u16::from_le_bytes([entry[0], entry[1]]);
        let projection = entry[2];
        let k = TrellisK::from_k2(entry[3])?;
        let bytes = u64::from(u32::from_le_bytes(entry[4..8].try_into().unwrap()));
        let record_offset = u64::from_le_bytes(entry[8..16].try_into().unwrap());
        let expected = Projection::ALL[at % 2];
        if usize::from(expert) != at / 2 || projection != expected.code() {
            return Err(fail(format!(
                "{} entry {at} is expert {expert} projection {projection}; expert {} projection {} expected",
                index_path.display(),
                at / 2,
                expected.code()
            )));
        }
        let shape = geometry.projection_shape(expected).to_vec();
        let encoded = tensor_encoded_size(StorageLayout::TrellisTile16V1, k.format(), &shape)?;
        if bytes != encoded || record_offset != offset {
            return Err(fail(format!(
                "{} entry {at}: {bytes} bytes at {record_offset}; its class stores {encoded} at {offset}",
                index_path.display()
            )));
        }
        plan.objects.push(Planned {
            name: name(u64::from(expert), expected),
            spec: Spec::Tensor {
                format: k.format(),
                layout: StorageLayout::TrellisTile16V1,
                shape,
            },
            slices: vec![Slice { file, offset, len: bytes }],
        });
        offset += bytes;
    }
    if offset != plan.files[file].bytes {
        return Err(fail(format!(
            "experts.bin is {} bytes, its index covers {offset}",
            plan.files[file].bytes
        )));
    }
    Ok(())
}

/// The n-gram unit (flash-next-layout.md §7): the table from its shards, the hash
/// buffers, the hot rows. `false` while `table/` is not complete yet.
fn plan_ngram(plan: &mut UnitPlan, done: &Done, geometry: &FlashNextGeometry) -> Result<bool> {
    // The shards live in `table/`, with their own DONE (or, equally, listed
    // in the unit's DONE as `table/<shard>`).
    let mut shards: Done = done
        .iter()
        .filter_map(|(name, entry)| name.strip_prefix("table/").map(|s| (s.to_owned(), entry.clone())))
        .collect();
    let table_dir = plan.dir.join("table");
    if let Some(table_done) = read_done(&table_dir)? {
        shards.extend(table_done);
    } else if shards.is_empty() && table_dir.exists() {
        return Ok(false);
    }
    if shards.is_empty() {
        return Err(fail(format!("{} has no table shards", plan.dir.display())));
    }
    for (at, name) in shards.keys().enumerate() {
        if *name != format!("shard_{at:03}.int4") {
            return Err(fail(format!(
                "table file {name} is not shard_{at:03}.int4: the shards must run 000.. without a gap"
            )));
        }
    }
    // hot_rows.json records how many shards the sweep converted and whether
    // that is the whole table (flash-next-layout.md §7.3).
    let manifest = if done.contains_key("hot_rows.json") {
        read_json(&plan.dir.join("hot_rows.json"))?
    } else {
        return Err(fail(format!("{}'s DONE does not list hot_rows.json", plan.dir.display())));
    };
    if let Some(expected) = manifest.get("shards").and_then(Value::as_u64) {
        if expected != shards.len() as u64 {
            return Err(fail(format!(
                "hot_rows.json records {expected} table shards, table/ holds {}",
                shards.len()
            )));
        }
    }
    let mut table_done = Done::new();
    for (name, entry) in &shards {
        table_done.insert(format!("table/{name}"), entry.clone());
    }
    let columns = geometry.ngram_head_dim;
    let row_bytes = crate::row_interleaved_geometry(NumericFormat::Q4G32F16S, &[1, columns])?.row_bytes;
    let mut slices = Vec::with_capacity(shards.len());
    let mut rows = 0u64;
    for name in shards.keys() {
        let file = plan.file(&table_done, &format!("table/{name}"))?;
        let len = plan.files[file].bytes;
        if len == 0 || !len.is_multiple_of(row_bytes) {
            return Err(fail(format!("table shard {name} is {len} bytes, not whole {row_bytes}-byte rows")));
        }
        rows += len / row_bytes;
        slices.push(Slice { file, offset: 0, len });
    }
    let whole_table = manifest.get("complete").and_then(Value::as_bool).unwrap_or(true);
    if let Some(table_rows) = manifest.get("table_rows").and_then(Value::as_u64) {
        if whole_table && table_rows != rows {
            return Err(fail(format!(
                "hot_rows.json records a table of {table_rows} rows, the shards hold {rows}"
            )));
        }
    }
    plan.objects.push(Planned {
        name: crate::flash_next::ngram_table_name(geometry),
        spec: Spec::Tensor {
            format: NumericFormat::Q4G32F16S,
            layout: StorageLayout::RowInterleavedV1,
            shape: vec![rows, columns],
        },
        slices,
    });

    let prefix = format!("layers.{}.ple.ple_embedding", geometry.ple_layer);
    for (file, name, format, word) in [
        ("layer_multipliers.i64", format!("{prefix}.layer_multipliers"), NumericFormat::I64, 8),
        ("ngram_heads_vocab_sizes.i64", format!("{prefix}.ngram_heads_vocab_sizes"), NumericFormat::I64, 8),
        ("ngram_heads_offsets.i64", format!("{prefix}.ngram_heads_offsets"), NumericFormat::I64, 8),
        // u32 row ids below 2^31 are the same bytes as I32.
        ("hot_rows.u32", format!("{prefix}.ngram_embedding.hot_rows"), NumericFormat::I32, 4),
    ] {
        let (bytes, _) = done
            .get(file)
            .ok_or_else(|| fail(format!("{}'s DONE does not list {file}", plan.dir.display())))?;
        if *bytes == 0 || bytes % word != 0 {
            return Err(fail(format!("{file} is {bytes} bytes, not whole {word}-byte words")));
        }
        plan.tensor(done, name, file, format, StorageLayout::ContiguousLeV1, vec![bytes / word])?;
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Appending a unit
// ---------------------------------------------------------------------------

/// Append `plan`'s objects after whatever of them the state already holds,
/// verifying each file's SHA-256 as its last byte goes in, and deleting
/// consumed files in batches behind a synced state.
fn append_unit(
    writer: &mut ContainerWriter,
    state: &mut PackState,
    state_file: &Path,
    plan: &UnitPlan,
    options: &PackOptions,
) -> Result<()> {
    // Where a previous run stopped inside this unit: a prefix of its objects
    // finished, maybe the next one open.
    let finished: HashSet<&str> = writer.state().objects.iter().map(Object::name).collect();
    let resume_at = plan.objects.iter().take_while(|o| finished.contains(o.name.as_str())).count();
    if let Some(stray) = plan.objects[resume_at..].iter().find(|o| finished.contains(o.name.as_str())) {
        return Err(fail(format!("{} is already in the container out of order", stray.name)));
    }
    let mut skip_bytes = 0u64;
    if let Some(open) = &writer.state().open {
        let next = plan.objects.get(resume_at);
        if next.map(|o| o.name.as_str()) != Some(open.object.name()) {
            return Err(fail(format!(
                "the container has {} open, the unit continues with {}",
                open.object.name(),
                next.map_or("nothing", |o| o.name.as_str())
            )));
        }
        skip_bytes = open.written;
    }

    // Files whose every slice is already in were verified before the state
    // that records them was written.
    let mut last_slice_of: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); plan.files.len()];
    for (o, object) in plan.objects.iter().enumerate() {
        for (s, slice) in object.slices.iter().enumerate() {
            last_slice_of[slice.file] = (o, s);
        }
    }
    let mut pending: Vec<usize> = Vec::new();
    let mut pending_bytes = 0u64;
    let mut hashers: BTreeMap<usize, Sha256> = BTreeMap::new();
    let mut source: Option<(usize, BufReader<File>)> = None;

    for (o, object) in plan.objects.iter().enumerate().skip(resume_at) {
        let first = o == resume_at;
        if !(first && writer.state().open.is_some()) {
            match &object.spec {
                Spec::Tensor { format, layout, shape } => {
                    writer.begin_tensor(&object.name, *format, *layout, shape)?
                }
                Spec::Resource { bytes } => {
                    writer.begin_resource(&object.name, ResourceEncoding::RawBytesV1, *bytes)?
                }
            }
        }
        let mut covered = 0u64;
        for (s, slice) in object.slices.iter().enumerate() {
            covered += slice.len;
            if first && covered <= skip_bytes {
                continue;
            }
            if first && covered - slice.len < skip_bytes {
                return Err(fail(format!(
                    "{} was left open inside a work file; the state is not a packer's",
                    object.name
                )));
            }
            // Slices of one file are consecutive and in order, so one
            // sequential reader per file serves them all.
            if source.as_ref().map(|(f, _)| *f) != Some(slice.file) {
                let file = &plan.files[slice.file];
                if slice.offset != 0 {
                    return Err(fail(format!(
                        "{} would resume at byte {}; the state is not a packer's",
                        file.path.display(),
                        slice.offset
                    )));
                }
                let handle = File::open(&file.path)
                    .map_err(|e| fail(format!("open {}: {e}", file.path.display())))?;
                let len = handle
                    .metadata()
                    .map_err(|e| fail(format!("stat {}: {e}", file.path.display())))?
                    .len();
                check_size(&file.path, len, file.bytes)?;
                source = Some((slice.file, BufReader::with_capacity(8 << 20, handle)));
            }
            let (_, reader) = source.as_mut().unwrap();
            let hasher = hashers.entry(slice.file).or_default();
            let mut hashing = HashingReader {
                inner: Read::take(&mut *reader, slice.len),
                hasher,
            };
            let written = writer.write(&mut hashing)?;
            if written != slice.len {
                return Err(fail(format!(
                    "{} ended {} bytes early",
                    plan.files[slice.file].path.display(),
                    slice.len - written
                )));
            }
            if last_slice_of[slice.file] == (o, s) {
                let file = &plan.files[slice.file];
                let digest = hex(&hashers.remove(&slice.file).unwrap().finalize());
                if digest != file.sha256 {
                    return Err(fail(format!(
                        "{} has SHA-256 {digest}, DONE records {}",
                        file.path.display(),
                        file.sha256
                    )));
                }
                source = None;
                pending.push(slice.file);
                pending_bytes += file.bytes;
                if pending_bytes >= options.delete_batch_bytes {
                    writer.sync()?;
                    state.writer = writer.state().clone();
                    write_state(state_file, state)?;
                    if !options.keep_work {
                        for file in pending.drain(..) {
                            remove_file(&plan.files[file].path)?;
                        }
                    }
                    pending_bytes = 0;
                }
            }
        }
        writer.end_object()?;
    }
    Ok(())
}

fn check_size(path: &Path, on_disk: u64, done: u64) -> Result<()> {
    if on_disk != done {
        return Err(fail(format!("{} is {on_disk} bytes, DONE records {done}", path.display())));
    }
    Ok(())
}

/// A reader that feeds a SHA-256 with what passes through it.
struct HashingReader<'a, R: Read> {
    inner: R,
    hasher: &'a mut Sha256,
}

impl<R: Read> Read for HashingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// State, lock, sidecar
// ---------------------------------------------------------------------------

/// An exclusive OS lock on `<artifact>.pack.lock`, held for the whole call
/// (two packers on one container would interleave appends). The OS releases
/// it when the process ends, so an interrupted run leaves nothing to clean.
/// The empty lock file stays: deleting it would let a second packer lock a
/// fresh file while the first still holds the old one.
fn lock(artifact: &Path) -> Result<File> {
    let path = with_suffix(artifact, ".pack.lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| fail(format!("open {}: {e}", path.display())))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => {
            Err(fail(format!("another packer holds {}", path.display())))
        }
        Err(std::fs::TryLockError::Error(e)) => {
            Err(fail(format!("lock {}: {e}", path.display())))
        }
    }
}

/// The bytes a complete unit's `DONE` files (and its table's) record; 0 for
/// a unit not complete yet.
fn unit_bytes(dir: &Path) -> Result<u64> {
    let mut bytes = 0u64;
    for done in [read_done(dir)?, read_done(&dir.join("table"))?].into_iter().flatten() {
        bytes += done.values().map(|(b, _)| b).sum::<u64>();
    }
    Ok(bytes)
}

/// A `keep_work` run writes a second copy of what it appends.
fn check_keep_work_room(needed: u64, free: u64) -> Result<()> {
    if needed + KEEP_WORK_MARGIN_BYTES > free {
        return Err(fail(format!(
            "a keep-work pack keeps two copies: it needs {needed} bytes plus a {KEEP_WORK_MARGIN_BYTES}-byte \
             margin, the volume has {free} free (pack without keep-work, after the converter ends)"
        )));
    }
    Ok(())
}

/// Free bytes on the volume holding `dir`.
#[cfg(windows)]
fn free_bytes(dir: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut available = 0u64;
    let ok = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(fail(format!(
            "free space of {}: {}",
            dir.display(),
            std::io::Error::last_os_error()
        )));
    }
    Ok(available)
}

/// Free bytes on the volume holding `dir`.
#[cfg(unix)]
fn free_bytes(dir: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes())
        .map_err(|_| fail(format!("{} contains a NUL", dir.display())))?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return Err(fail(format!(
            "free space of {}: {}",
            dir.display(),
            std::io::Error::last_os_error()
        )));
    }
    Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| fail(format!("read {}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| fail(format!("invalid JSON in {}: {e}", path.display())))
}

fn read_state(path: &Path) -> Result<PackState> {
    let value = read_json(path)?;
    let layers = value
        .get("layers")
        .and_then(Value::as_u64)
        .ok_or_else(|| fail(format!("{} has no layers", path.display())))? as usize;
    let units_done = value
        .get("units_done")
        .and_then(Value::as_array)
        .ok_or_else(|| fail(format!("{} has no units_done", path.display())))?
        .iter()
        .map(|unit| require_string(unit, "unit name").map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    let finished = value
        .get("finished")
        .and_then(Value::as_bool)
        .ok_or_else(|| fail(format!("{} has no finished flag", path.display())))?;
    let writer = WriterState::from_value(
        value
            .get("writer")
            .ok_or_else(|| fail(format!("{} has no writer state", path.display())))?,
    )?;
    Ok(PackState {
        layers,
        units_done,
        ngram_source_digest: value.get("ngram_source_digest").and_then(Value::as_str).map(str::to_owned),
        finished,
        writer,
    })
}

fn write_state(path: &Path, state: &PackState) -> Result<()> {
    let value = json!({
        "layers": state.layers,
        "units_done": state.units_done,
        "ngram_source_digest": state.ngram_source_digest,
        "finished": state.finished,
        "writer": state.writer.to_value(),
    });
    write_atomically(path, value.to_string().as_bytes())
}

/// Replace `path` with `bytes` so that a crash leaves either the old or the
/// new content, never a torn file.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = with_suffix(path, ".tmp");
    {
        let mut file = File::create(&temporary)
            .map_err(|e| fail(format!("create {}: {e}", temporary.display())))?;
        std::io::Write::write_all(&mut file, bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| fail(format!("write {}: {e}", temporary.display())))?;
    }
    std::fs::rename(&temporary, path)
        .map_err(|e| fail(format!("replace {}: {e}", path.display())))
}

fn remove_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).map_err(|e| fail(format!("delete {}: {e}", dir.display())))?;
    }
    Ok(())
}

fn remove_file(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(fail(format!("delete {}: {e}", path.display()))),
    }
}

/// `converter.json`'s `status` must be one this run packs.
fn check_status(converter: &Path, record: &Value, accept: &[String]) -> Result<()> {
    let status = record.get("status").and_then(Value::as_str).unwrap_or_default();
    if !accept.iter().any(|a| a == status) {
        return Err(fail(format!(
            "{} has status {status:?}; this run packs only {accept:?}",
            converter.display()
        )));
    }
    Ok(())
}

/// [`Family::Qwen27bNvfp4`] before its container exists: the whole tree is
/// checked, so a refusal leaves nothing behind (but the lock file). Waits
/// (`Some`) until `converter.json` and every unit are complete; refuses
/// `converter.json` short of the sidecar's fields, a unit that is not the
/// inventory's ([`plan_nvfp4_27b`]), and NVFP4 records that are not the
/// work files'.
fn preflight_nvfp4_27b(options: &PackOptions, units: &[String]) -> Result<Option<PackOutcome>> {
    let waiting = |next: &str| Some(PackOutcome::Waiting { appended: 0, total: units.len(), next: next.into() });
    let converter = options.work_dir.join("converter.json");
    if !converter.exists() {
        return Ok(waiting("converter.json"));
    }
    let record = read_json(&converter)?;
    check_status(&converter, &record, &options.accept_status)?;
    check_nvfp4_27b_record(&record, &options.identity)?;
    let mut plans = Vec::with_capacity(units.len());
    for unit in units {
        match plan_unit(&options.work_dir, unit, &options.family, None)? {
            Some(plan) => plans.push(plan),
            None => return Ok(waiting(unit)),
        }
    }
    check_nvfp4_27b_divisors(&record, &plans)?;
    Ok(None)
}

/// The fields of `converter.json` the 27B NVFP4 sidecar must carry: the
/// recipe id (never the file name's), the identity this run packs, the
/// source's repo, revision and license, and the NVFP4 parents' records.
fn check_nvfp4_27b_record(record: &Value, identity: &ArtifactIdentity) -> Result<()> {
    for pointer in ["/recipe_id", "/source/repo", "/source/revision", "/source/license"] {
        if record.pointer(pointer).and_then(Value::as_str).is_none_or(str::is_empty) {
            return Err(fail(format!("converter.json has no {}", pointer[1..].replace('/', "."))));
        }
    }
    let recorded = (
        record.pointer("/identity/model_id").and_then(Value::as_str),
        record.pointer("/identity/weights_id").and_then(Value::as_str),
    );
    if recorded != (Some(identity.model_id.as_str()), Some(identity.weights_id.as_str())) {
        return Err(fail(format!(
            "converter.json records identity {}/{}; this run packs {}/{}",
            recorded.0.unwrap_or("-"),
            recorded.1.unwrap_or("-"),
            identity.model_id,
            identity.weights_id
        )));
    }
    if !record.pointer("/local_nvfp4/parents").is_some_and(Value::is_object) {
        return Err(fail("converter.json has no local_nvfp4.parents"));
    }
    Ok(())
}

/// `local_nvfp4.parents` records exactly the NVFP4 objects: a text parent
/// with the weight divisor its work file ends with (the container's
/// trailing FP32 word, which `checksum::verify` compares), a drafter parent
/// (`dflash2/`, weight-only) with none.
fn check_nvfp4_27b_divisors(record: &Value, plans: &[UnitPlan]) -> Result<()> {
    let parents = record.pointer("/local_nvfp4/parents").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut seen = BTreeSet::new();
    for plan in plans {
        for object in &plan.objects {
            let Spec::Tensor { format: NumericFormat::Nvfp4, shape, .. } = &object.spec else { continue };
            let name = object.name.as_str();
            seen.insert(name);
            let recorded = parents
                .get(name)
                .ok_or_else(|| fail(format!("converter.json's local_nvfp4.parents does not record {name}")))?
                .get("weight_scale_divisor")
                .unwrap_or(&Value::Null);
            if name.starts_with("dflash2/") {
                if !recorded.is_null() {
                    return Err(fail(format!(
                        "converter.json records a weight_scale_divisor for {name}, which is weight-only"
                    )));
                }
                continue;
            }
            let recorded = recorded
                .as_f64()
                .ok_or_else(|| fail(format!("converter.json records no weight_scale_divisor for {name}")))?;
            let file = &plan.files[object.slices[0].file];
            let at = crate::block_scale_geometry(NumericFormat::Nvfp4, shape)?.weight_divisor_offset;
            let mut word = [0u8; 4];
            File::open(&file.path)
                .and_then(|mut f| f.seek(SeekFrom::Start(at)).and_then(|_| f.read_exact(&mut word)))
                .map_err(|e| fail(format!("read {}'s weight divisor: {e}", file.path.display())))?;
            let stored = f32::from_le_bytes(word);
            if f64::from(stored) != recorded {
                return Err(fail(format!(
                    "{name} stores weight divisor {stored}; converter.json records {recorded}"
                )));
            }
        }
    }
    if let Some(stray) = parents.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(fail(format!(
            "converter.json's local_nvfp4.parents records {stray}, which is not an NVFP4 object of the inventory"
        )));
    }
    Ok(())
}

/// What an MTP companion records of its main container (flash-next-layout.md §13.5
/// `pair.main`), read from the container itself: `content_hash` is
/// [`Reader::content_hash`], the hash the binder compares at load, and
/// `file_sha256` the whole file's (one pass over it, at pack time only).
fn pair_record(main: &Path) -> Result<Value> {
    let reader = Reader::open(main)?;
    let file = main
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| fail(format!("{} has no file name", main.display())))?;
    let mut hasher = Sha256::new();
    let handle = File::open(main).map_err(|e| fail(format!("open {}: {e}", main.display())))?;
    std::io::copy(&mut BufReader::with_capacity(8 << 20, handle), &mut hasher)
        .map_err(|e| fail(format!("read {}: {e}", main.display())))?;
    Ok(json!({
        "file": file,
        "model_id": reader.identity().model_id,
        "weights_id": reader.identity().weights_id,
        "bytes": reader.file_bytes(),
        "objects": reader.objects().len(),
        "content_hash": hex(&reader.content_hash()),
        "file_sha256": hex(&hasher.finalize()),
    }))
}

/// The companion's `converter.json` records the main container the head was
/// calibrated on (its whole-file SHA-256), and it must be the one this run
/// pins: the hashes are compared, not the names.
fn check_pair(converter: &Value, pair: &Value) -> Result<()> {
    let ours = pair["file_sha256"].as_str().unwrap_or_default();
    match converter.pointer("/pair/main/file_sha256").and_then(Value::as_str) {
        Some(theirs) if theirs == ours => Ok(()),
        Some(theirs) => Err(fail(format!(
            "converter.json calibrated the head on a main container with SHA-256 {theirs}; {} has {ours}",
            pair["file"].as_str().unwrap_or_default()
        ))),
        None => Err(fail("converter.json does not record the head's main container (pair.main.file_sha256)")),
    }
}

/// Write the sidecar: its existing keys, then `converter.json`'s (when the
/// converter has written it), then the whole-file invariants and, for an
/// MTP companion, `pair.main`'s fields read from the main container.
/// Returns whether `converter.json` was merged.
fn write_sidecar(
    artifact: &Path,
    work_dir: &Path,
    file_bytes: u64,
    object_count: usize,
    pair: Option<&Value>,
    ngram_source_digest: Option<&str>,
) -> Result<bool> {
    let path = sidecar_path(artifact);
    let mut root = if path.exists() {
        match read_json(&path)? {
            Value::Object(map) => map,
            _ => return Err(fail(format!("{} is not a JSON object", path.display()))),
        }
    } else {
        Map::new()
    };
    let converter = work_dir.join("converter.json");
    let converter_merged = converter.exists();
    if converter_merged {
        match read_json(&converter)? {
            Value::Object(map) => root.extend(map),
            _ => return Err(fail(format!("{} is not a JSON object", converter.display()))),
        }
    }
    if !root.contains_key("recipe_id") {
        let stem = artifact
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| fail(format!("{} has no file stem", artifact.display())))?;
        root.insert("recipe_id".into(), json!(stem));
    }
    for (key, member, value) in [
        ("artifact", "bytes", file_bytes),
        ("objects", "count", object_count as u64),
    ] {
        let entry = root.entry(key).or_insert_with(|| Value::Object(Map::new()));
        let map = entry
            .as_object_mut()
            .ok_or_else(|| fail(format!("sidecar member {key} is not an object")))?;
        map.insert(member.into(), json!(value));
    }
    if let Some(Value::Object(fields)) = pair {
        let pair = root
            .entry("pair")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| fail("sidecar member pair is not an object"))?;
        pair.entry("main")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| fail("sidecar member pair.main is not an object"))?
            .extend(fields.clone());
    }
    if let Some(digest) = ngram_source_digest {
        root.insert("ngram_cache_source".into(), json!({"schema": 1, "digest": digest}));
    }
    let text = serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| fail(format!("serialize sidecar: {e}")))?;
    write_atomically(&path, text.as_bytes())?;
    Ok(converter_merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_next::fixture::{self, WorkTree};
    use crate::Sidecar;

    #[test]
    fn ngram_digest_tracks_content_order_and_slice_boundaries() {
        let mut plan = UnitPlan::new(PathBuf::from("unused"));
        plan.files.push(UnitFile { path: PathBuf::from("unused"), bytes: 10, sha256: "01".repeat(32) });
        plan.objects.push(Planned { name: "table".into(), spec: Spec::Resource { bytes: 10 }, slices: vec![Slice { file: 0, offset: 0, len: 10 }] });
        let first=ngram_source_digest(&plan);
        assert_eq!(first.len(),64); assert_eq!(first,ngram_source_digest(&plan));
        plan.files[0].sha256="02".repeat(32); assert_ne!(first,ngram_source_digest(&plan));
        plan.files[0].sha256="01".repeat(32); plan.objects[0].slices[0].offset=1;
        assert_ne!(first,ngram_source_digest(&plan));
    }
    fn quiet() -> impl FnMut(&str) {
        |_: &str| {}
    }

    /// The container one uninterrupted run makes of the fixture tree.
    fn reference_container() -> Vec<u8> {
        let artifact = fixture::build("reference").unwrap();
        std::fs::read(&artifact.path).unwrap()
    }

    fn finished(outcome: PackOutcome) -> (u64, usize, bool) {
        match outcome {
            PackOutcome::Finished { file_bytes, object_count, converter_merged } => {
                (file_bytes, object_count, converter_merged)
            }
            other => panic!("not finished: {other:?}"),
        }
    }

    #[test]
    fn the_packer_waits_for_an_incomplete_unit_and_resumes_to_the_same_bytes() {
        let tree = WorkTree::new("waiting").unwrap();
        for unit in ["frontend", "global", "ngram", "layers/L00"] {
            tree.write_unit(unit).unwrap();
        }
        // Layer 1 is still being converted: files there, no DONE.
        tree.write_unit_files("layers/L01").unwrap();
        tree.write_converter_json().unwrap();

        let outcome = pack(&tree.pack_options(), &mut quiet()).unwrap();
        assert_eq!(
            outcome,
            PackOutcome::Waiting { appended: 4, total: 5, next: "layers/L01".into() }
        );
        assert!(Reader::open(&tree.artifact_path()).is_err(), "no directory before the last unit");
        for unit in ["frontend", "global", "ngram", "layers/L00"] {
            assert!(!tree.work_dir().join(unit).exists(), "{unit} is deleted once appended");
        }
        assert!(tree.work_dir().join("layers/L01").exists(), "an incomplete unit is left alone");

        // A crash halfway through appending layer 1: bytes past what the
        // state records.
        {
            let mut file = OpenOptions::new().append(true).open(tree.artifact_path()).unwrap();
            std::io::Write::write_all(&mut file, &[0xEEu8; 5000]).unwrap();
        }
        tree.mark_done("layers/L01").unwrap();
        let (_, object_count, _) = finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        assert_eq!(object_count, 99);
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), reference_container());

        let record = read_json(&sidecar_path(&tree.artifact_path())).unwrap();
        let digest = record["ngram_cache_source"]["digest"].as_str().unwrap().to_owned();
        assert_eq!(digest.len(), 64);
        let reference = fixture::build("digest-reference").unwrap();
        assert_eq!(read_json(&sidecar_path(&reference.path)).unwrap()["ngram_cache_source"]["digest"], digest);
        // Running again after the end preserves both bytes and the source digest.
        finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), reference_container());
        assert_eq!(read_json(&sidecar_path(&tree.artifact_path())).unwrap()["ngram_cache_source"]["digest"], digest);
    }

    #[test]
    fn a_run_stopped_inside_the_table_resumes_after_its_last_synced_shard() {
        let tree = WorkTree::new("mid-table").unwrap();
        tree.write_all().unwrap();
        // The second shard does not match its DONE: the run stops there,
        // after the first shard was appended, recorded and deleted.
        let shard = tree.work_dir().join("ngram/table/shard_001.int4");
        let good = std::fs::read(&shard).unwrap();
        let mut bad = good.clone();
        bad[0] ^= 0xFF;
        std::fs::write(&shard, &bad).unwrap();

        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("shard_001.int4 has SHA-256"), "{err}");
        assert!(!tree.work_dir().join("ngram/table/shard_000.int4").exists(), "shard 0 is in and deleted");
        assert!(shard.exists(), "the refused shard is kept");

        std::fs::write(&shard, &good).unwrap();
        finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), reference_container());
    }

    #[test]
    fn a_work_file_that_does_not_match_its_done_is_never_deleted() {
        let tree = WorkTree::new("corrupt").unwrap();
        tree.write_all().unwrap();
        let experts = tree.work_dir().join("layers/L00/experts.bin");
        let mut bytes = std::fs::read(&experts).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&experts, &bytes).unwrap();

        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("experts.bin has SHA-256"), "{err}");
        assert!(experts.exists());
    }

    #[test]
    fn the_converters_own_records_stay_out_of_the_container() {
        let tree = WorkTree::new("records").unwrap();
        tree.write_all().unwrap();
        let layer = tree.work_dir().join("layers/L00");
        std::fs::write(layer.join("layer.json"), b"{}").unwrap();
        std::fs::write(layer.join("route_test_0.npy"), b"npy").unwrap();
        tree.mark_done("layers/L00").unwrap();
        let (_, object_count, _) = finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        assert_eq!(object_count, 99);
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), reference_container());
    }

    #[test]
    fn nothing_is_deleted_while_the_converter_may_still_read_its_work_files() {
        let tree = WorkTree::new("converter-running").unwrap();
        for unit in unit_names(tree.geometry.layers) {
            tree.write_unit(&unit).unwrap();
        }
        // Every unit is complete, but the converter has not ended: its
        // self-check still decodes experts.bin.
        let outcome = pack(&tree.pack_options(), &mut quiet()).unwrap();
        assert_eq!(
            outcome,
            PackOutcome::Waiting { appended: 0, total: 5, next: "converter.json".into() }
        );
        for unit in unit_names(tree.geometry.layers) {
            assert!(tree.work_dir().join(&unit).join("DONE").exists(), "{unit} untouched");
        }
        // A run that deletes nothing may go ahead.
        let keep = PackOptions { keep_work: true, ..tree.pack_options() };
        finished(pack(&keep, &mut quiet()).unwrap());
        assert!(tree.work_dir().join("layers/L00/experts.bin").exists());
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), reference_container());
    }

    #[test]
    fn a_dry_run_followed_by_a_plain_run_deletes_nothing_before_the_converter_ends() {
        let tree = WorkTree::new("dry-then-plain").unwrap();
        for unit in unit_names(tree.geometry.layers) {
            tree.write_unit(&unit).unwrap();
        }
        // A dry-run check while the converter still runs records every unit
        // as appended and keeps the files ...
        let keep = PackOptions { keep_work: true, ..tree.pack_options() };
        finished(pack(&keep, &mut quiet()).unwrap());
        // ... and a plain run before converter.json must not clean them up.
        let outcome = pack(&tree.pack_options(), &mut quiet()).unwrap();
        assert_eq!(
            outcome,
            PackOutcome::Waiting { appended: 5, total: 5, next: "converter.json".into() }
        );
        for unit in unit_names(tree.geometry.layers) {
            assert!(tree.work_dir().join(&unit).join("DONE").exists(), "{unit} untouched");
        }
        assert!(tree.work_dir().join("layers/L01/experts.bin").exists());

        tree.write_converter_json().unwrap();
        let (_, _, merged) = finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        assert!(merged);
        assert!(!tree.work_dir().join("layers/L01").exists(), "deleted once the converter is done");
    }

    #[test]
    fn only_a_complete_conversion_is_packed_unless_asked() {
        let tree = WorkTree::new("status").unwrap();
        tree.write_all().unwrap();
        std::fs::write(
            tree.work_dir().join("converter.json"),
            json!({"schema": "flash-next-converter-v1", "status": "dry-run"}).to_string(),
        )
        .unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("has status \"dry-run\""), "{err}");
        assert!(tree.work_dir().join("layers/L00/experts.bin").exists(), "nothing deleted");

        let dry = PackOptions { accept_status: vec!["dry-run".into()], ..tree.pack_options() };
        finished(pack(&dry, &mut quiet()).unwrap());
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), reference_container());
    }

    #[test]
    fn a_table_short_of_its_manifest_is_refused() {
        let tree = WorkTree::new("short-table").unwrap();
        tree.write_all().unwrap();
        let ngram = tree.work_dir().join("ngram");
        std::fs::write(
            ngram.join("hot_rows.json"),
            json!({"rows": 6, "table_rows": 1000, "shards": 3, "complete": true}).to_string(),
        )
        .unwrap();
        tree.mark_done("ngram").unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("records 3 table shards, table/ holds 2"), "{err}");

        std::fs::write(
            ngram.join("hot_rows.json"),
            json!({"rows": 6, "table_rows": 2000, "shards": 2, "complete": true}).to_string(),
        )
        .unwrap();
        tree.mark_done("ngram").unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("a table of 2000 rows, the shards hold 1000"), "{err}");

        // A dry run's partial sweep says so, and packs.
        std::fs::write(
            ngram.join("hot_rows.json"),
            json!({"rows": 6, "table_rows": 2000, "shards": 2, "complete": false}).to_string(),
        )
        .unwrap();
        tree.mark_done("ngram").unwrap();
        finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
    }

    #[test]
    fn a_keep_work_pack_needs_room_for_two_copies() {
        assert!(check_keep_work_room(10 << 30, 20 << 30).is_ok());
        let err = check_keep_work_room(19 << 30, 20 << 30).unwrap_err().to_string();
        assert!(err.contains("keeps two copies"), "{err}");
        // The real volume answers.
        assert!(free_bytes(&std::env::temp_dir()).unwrap() > 0);
    }

    #[test]
    fn a_unit_the_header_cannot_hold_is_refused_before_its_files_are_touched() {
        let tree = WorkTree::new("small-header").unwrap();
        tree.write_all().unwrap();
        // 8 KiB holds the first three units' entries, not a layer's 38.
        let small = PackOptions { header_bytes: 8192, ..tree.pack_options() };
        let err = pack(&small, &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("with layers/L00") && err.contains("the header reserves 8176"), "{err}");
        assert!(tree.work_dir().join("layers/L00/experts.bin").exists());
        assert!(tree.work_dir().join("layers/L00/DONE").exists());
    }

    #[test]
    fn table_shards_must_run_without_a_gap() {
        let tree = WorkTree::new("shard-gap").unwrap();
        tree.write_all().unwrap();
        let table = tree.work_dir().join("ngram/table");
        std::fs::rename(table.join("shard_001.int4"), table.join("shard_002.int4")).unwrap();
        tree.mark_done("ngram/table").unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("shard_002.int4 is not shard_001.int4"), "{err}");
    }

    #[test]
    fn an_index_out_of_its_class_sizes_is_refused() {
        let tree = WorkTree::new("bad-index").unwrap();
        tree.write_all().unwrap();
        let idx = tree.work_dir().join("layers/L01/experts.idx");
        let mut bytes = std::fs::read(&idx).unwrap();
        // Entry 3 (expert 1, down) claims one byte more than its class.
        let size = u32::from_le_bytes(bytes[52..56].try_into().unwrap());
        bytes[52..56].copy_from_slice(&(size + 1).to_le_bytes());
        std::fs::write(&idx, &bytes).unwrap();
        tree.mark_done("layers/L01").unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("experts.idx entry 3"), "{err}");
    }

    #[test]
    fn the_sidecar_carries_the_invariants_and_the_converters_record_whenever_it_comes() {
        let tree = WorkTree::new("sidecar").unwrap();
        for unit in unit_names(tree.geometry.layers) {
            tree.write_unit(&unit).unwrap();
        }
        // A dry run finishes the container before the converter's
        // end-of-run record exists.
        let keep = PackOptions { keep_work: true, ..tree.pack_options() };
        let (file_bytes, object_count, merged) = finished(pack(&keep, &mut quiet()).unwrap());
        assert!(!merged);
        let sidecar = Sidecar::load(&sidecar_path(&tree.artifact_path())).unwrap();
        assert_eq!(sidecar.recipe_id, "qwen3_8_flash_next_fixture-v2");
        assert_eq!((sidecar.artifact_bytes, sidecar.object_count), (file_bytes, object_count as u64));
        let reader = Reader::open(&tree.artifact_path()).unwrap();
        assert!(crate::verify(&reader, &sidecar).unwrap().is_clean());

        tree.write_converter_json().unwrap();
        let (_, _, merged) = finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        assert!(merged);
        let record = read_json(&sidecar_path(&tree.artifact_path())).unwrap();
        assert_eq!(record["schema"], "flash-next-converter-v1");
        assert_eq!(record["objects"]["count"], object_count);
        assert_eq!(record["artifact"]["bytes"], file_bytes);
    }

    #[test]
    fn an_artifact_without_pack_state_is_never_overwritten() {
        let tree = WorkTree::new("existing").unwrap();
        tree.write_all().unwrap();
        std::fs::write(tree.artifact_path(), b"another file").unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("refusing to overwrite"), "{err}");
        assert_eq!(std::fs::read(tree.artifact_path()).unwrap(), b"another file");
    }

    #[test]
    fn a_second_packer_on_the_same_artifact_is_refused() {
        let tree = WorkTree::new("locked").unwrap();
        tree.write_all().unwrap();
        let held = lock(&tree.artifact_path()).unwrap();
        let err = pack(&tree.pack_options(), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("another packer holds"), "{err}");
        assert!(!tree.artifact_path().exists(), "the refused packer wrote nothing");
        drop(held);
        finished(pack(&tree.pack_options(), &mut quiet()).unwrap());
        // The lock file stays for the next packer to lock.
        assert!(with_suffix(&tree.artifact_path(), ".pack.lock").exists());
    }

    /// The main fixture container and, in its tree, the MTP head's work unit
    /// naming it.
    fn mtp_tree(tag: &str) -> fixture::FixtureArtifact {
        let main = fixture::build(tag).unwrap();
        main.tree.write_mtp(&main.path).unwrap();
        main
    }

    #[test]
    fn an_mtp_companion_holds_the_heads_objects_and_pins_its_main_container() {
        let main = mtp_tree("mtp-pack");
        let tree = &main.tree;
        let (file_bytes, object_count, merged) =
            finished(pack(&tree.mtp_pack_options(main.path.clone()), &mut quiet()).unwrap());
        assert!(merged);
        assert!(!tree.mtp_work_dir().join("mtp").exists(), "the unit is deleted once appended");

        let g = &tree.geometry;
        let companion = Reader::open(&tree.mtp_artifact_path()).unwrap();
        assert_eq!(companion.identity(), &mtp_identity());
        let entries = crate::flash_next::mtp_entries(g);
        assert_eq!(object_count, entries.len() + 2 * g.experts as usize);
        for entry in &entries {
            let Some(Object::Tensor(t)) = companion.find(&entry.name) else {
                panic!("{} is missing", entry.name)
            };
            assert_eq!((t.format, t.layout), (entry.format, entry.layout), "{}", entry.name);
        }
        for expert in 0..g.experts {
            for projection in Projection::ALL {
                let name = mtp_expert_name(expert, projection);
                let Some(Object::Tensor(t)) = companion.find(&name) else { panic!("{name} is missing") };
                assert_eq!(t.shape, g.projection_shape(projection).to_vec(), "{name}");
                assert_eq!(t.layout, StorageLayout::TrellisTile16V1, "{name}");
            }
        }

        // pair.main is the main container as the reader sees it, beside what
        // the converter recorded of it.
        let reader = Reader::open(&main.path).unwrap();
        let record = read_json(&sidecar_path(&tree.mtp_artifact_path())).unwrap();
        let pinned = &record["pair"]["main"];
        assert_eq!(pinned["content_hash"], hex(&reader.content_hash()));
        assert_eq!(pinned["model_id"], reader.identity().model_id);
        assert_eq!(pinned["weights_id"], reader.identity().weights_id);
        assert_eq!(pinned["bytes"], reader.file_bytes());
        assert_eq!(pinned["objects"], reader.objects().len());
        assert_eq!(pinned["file_sha256"], hex(&Sha256::digest(std::fs::read(&main.path).unwrap())));
        assert_eq!(pinned["file"], main.path.file_name().unwrap().to_str().unwrap());
        assert_eq!(record["schema"], "flash-next-mtp-converter-v1");
        assert_eq!((record["artifact"]["bytes"].as_u64(), record["objects"]["count"].as_u64()),
                   (Some(file_bytes), Some(object_count as u64)));
    }

    #[test]
    fn an_mtp_companion_is_refused_when_its_converter_records_another_main_container_or_none() {
        let main = mtp_tree("mtp-refused");
        let tree = &main.tree;
        let converter = tree.mtp_work_dir().join("converter.json");
        let mut record = read_json(&converter).unwrap();
        record["pair"]["main"]["file_sha256"] = json!("00".repeat(32));
        std::fs::write(&converter, record.to_string()).unwrap();
        let err = pack(&tree.mtp_pack_options(main.path.clone()), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains(&format!("a main container with SHA-256 {}", "00".repeat(32))), "{err}");

        record["pair"]["main"].as_object_mut().unwrap().remove("file_sha256");
        std::fs::write(&converter, record.to_string()).unwrap();
        let err = pack(&tree.mtp_pack_options(main.path.clone()), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("does not record the head's main container"), "{err}");
        assert!(tree.mtp_work_dir().join("mtp").exists(), "nothing is consumed");
    }

    #[test]
    fn an_mtp_unit_naming_a_tensor_outside_the_head_is_refused() {
        let main = mtp_tree("mtp-stray");
        let tree = &main.tree;
        let unit = tree.mtp_work_dir().join("mtp");
        let listing = unit.join("tensors.json");
        let text = std::fs::read_to_string(&listing).unwrap();
        std::fs::write(&listing, text.replacen("mtp.fc_hidden.weight\"", "fc_hidden.weight\"", 1)).unwrap();
        std::fs::remove_file(unit.join("DONE")).unwrap();
        tree.mark_mtp_done().unwrap();
        let err = pack(&tree.mtp_pack_options(main.path.clone()), &mut quiet()).unwrap_err().to_string();
        assert!(err.contains("lists fc_hidden.weight, outside mtp."), "{err}");
    }
}
