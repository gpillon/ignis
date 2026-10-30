//! `ignis-server model download` and `model list` (spec model-download/02
//! §`ignis-server model download` / `model list`).
//!
//! Which entries, where to, from where and with which token are decided by
//! resolution (`config::cli`), from the configuration a start reads — config
//! discovery, `--config`, the environment and the field flags all apply —
//! and handed to `main` as a [`ModelCommand`], which [`run`] carries out.
//! Nothing here asks: a typed command is the operator's yes, so it ignores
//! `download.enabled`, which gates only the implicit start-time fetch. And
//! nothing here needs `cuda`: any machine can fetch the files an air-gapped
//! one is carried.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_core::compute::ModelFamily;
use serde_json::{json, Value};

use super::catalog::{Catalog, CatalogEntry};
use super::transfer::{part_path, Downloader, Report};
use crate::config::profile::CardClass;
use crate::config::ApiKey;

/// What a `model` command line asked for, resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelCommand {
    /// Fetch every file of each of `entries` into `dir`, from `endpoint`.
    Download {
        entries: Vec<CatalogEntry>,
        dir: PathBuf,
        endpoint: String,
        /// The token [`super::bearer_token`] chose; `Debug` never shows it.
        token: Option<ApiKey>,
        /// `download.free_margin`: what each entry must leave free beyond
        /// the bytes it still misses ([`room_for`]).
        free_margin: u64,
    },
    /// One row per entry of `catalog`, with its state under `dir`.
    List { catalog: Catalog, dir: PathBuf, format: ListFormat },
}

/// How `model list` prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListFormat {
    /// Aligned columns, for a terminal.
    Text,
    /// One object per entry, for a tool.
    Json,
}

/// How much of an entry is under a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnDisk {
    /// Every file at its final name with its pinned byte count.
    Complete,
    /// Some of it: a file, or a `.part` a fetch left to resume.
    Partial,
    /// Nothing of it.
    Absent,
}

impl OnDisk {
    /// `complete`, `partial` or `absent`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Absent => "absent",
        }
    }
}

/// How much of `entry` is under `dir`; `size_of` is a file's size, `None`
/// when there is no file there. Sizes only, never a hash: hashing what is on
/// disk is `model verify`'s, which is not this command.
pub fn on_disk(entry: &CatalogEntry, dir: &Path, size_of: &dyn Fn(&Path) -> Option<u64>) -> OnDisk {
    if entry.files.iter().all(|file| size_of(&dir.join(&file.name)) == Some(file.bytes)) {
        return OnDisk::Complete;
    }
    let any = entry.files.iter().any(|file| {
        let path = dir.join(&file.name);
        size_of(&path).is_some() || size_of(&part_path(&path)).is_some()
    });
    if any { OnDisk::Partial } else { OnDisk::Absent }
}

/// How many bytes of `entry` are under `dir`: each file's at its final name
/// when it is there, else its `.part`'s, at most its pinned byte count — what
/// a download running, cancelled or interrupted has fetched so far.
/// `size_of` as for [`on_disk`]: sizes only.
pub fn bytes_on_disk(entry: &CatalogEntry, dir: &Path, size_of: &dyn Fn(&Path) -> Option<u64>) -> u64 {
    entry
        .files
        .iter()
        .map(|file| {
            let path = dir.join(&file.name);
            size_of(&path).or_else(|| size_of(&part_path(&path))).map_or(0, |bytes| bytes.min(file.bytes))
        })
        .sum()
}

/// The size of the file at `path`, read from the filesystem now; `None` when
/// nothing is there, or no file: the probe [`on_disk`] and [`bytes_on_disk`]
/// take everywhere but in their tests.
pub fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().filter(|meta| meta.is_file()).map(|meta| meta.len())
}

/// Whether the volume `dir` is on has room for what of `entry` is still
/// missing there plus `margin` (`download.free_margin`, GitHub #378), read
/// from the filesystem now: what the Server view's Download, `model
/// download` and the start's own fetch each check before they write a byte.
/// The reason, naming both numbers, when it has not. An entry with nothing
/// missing writes nothing, so it always has room.
pub fn room_for(entry: &CatalogEntry, dir: &Path, margin: u64) -> Result<(), String> {
    let missing = entry.total_bytes().saturating_sub(bytes_on_disk(entry, dir, &file_size));
    if missing == 0 {
        return Ok(());
    }
    fits(&entry.id, dir, missing, free_under(dir)?, margin)
}

/// [`room_for`]'s rule: `free` bytes cover `missing` plus `margin`, or the
/// refusal says how many bytes are needed and how many are free.
fn fits(id: &str, dir: &Path, missing: u64, free: u64, margin: u64) -> Result<(), String> {
    let needed = missing.saturating_add(margin);
    if free >= needed {
        return Ok(());
    }
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    Err(format!(
        "not enough room for {id} under {}: {needed} bytes needed ({missing} still missing plus download.free_margin {margin}), {free} bytes free ({:.2} GiB needed, {:.2} GiB free); free some space there or lower download.free_margin",
        dir.display(),
        gib(needed),
        gib(free)
    ))
}

/// The bytes free to this process on the volume `dir` is on: Windows'
/// `GetDiskFreeSpaceExW`, `statvfs` elsewhere
/// ([`ignis_artifact::volume_free_bytes`]). Read at `dir`, or at the nearest
/// folder above it that exists, since the first download into a models
/// folder creates it.
fn free_under(dir: &Path) -> Result<u64, String> {
    let dir = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    let existing = dir.ancestors().find(|folder| folder.is_dir()).unwrap_or(&dir);
    ignis_artifact::volume_free_bytes(existing).map_err(|err| format!("{err}, which a download checks before it begins"))
}

/// Whether `path` is one of `entry`'s files under `dir`: the same file,
/// however each was spelled (GitHub #380) — the file a loaded, start or
/// known model is loaded from, which removing the entry takes away.
pub fn lists_file(entry: &CatalogEntry, dir: &Path, path: &Path) -> bool {
    let same = |a: &Path, b: &Path| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => std::path::absolute(a).ok() == std::path::absolute(b).ok(),
    };
    entry.files.iter().any(|file| same(&dir.join(&file.name), path))
}

/// What a **Removal** of an entry did (GitHub #380): the files it deleted,
/// by name, with their bytes, and the files it left because another entry
/// lists them too.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Removed {
    /// The names deleted under the models folder, a `.part` under its own.
    pub files: Vec<String>,
    /// The bytes they held.
    pub bytes: u64,
    /// Each file left in place, with the entry that also lists it.
    pub kept: Vec<FileKept>,
}

/// A file a removal left, and the entry that also lists it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct FileKept {
    /// Its name under `download.path`.
    pub file: String,
    /// The other entry's id.
    pub by: String,
}

/// Another entry listing a file of the one a removal deletes: whether its
/// shared files stay, and its lock, held while they do not.
struct Sharer<'a> {
    id: &'a str,
    keeps: bool,
    _lock: Option<super::EntryLock>,
}

/// Delete `entry`'s files under `dir`, finished or partial — each file at
/// its name and its `.part` — as a removal from the Server view does (GitHub
/// #380). The caller holds `entry`'s lock ([`super::EntryLock`]). A file
/// another entry of `catalog` also lists stays, with its `.part`, while that
/// entry is on disk — every file of it at its name — or being downloaded —
/// its lock held by a fetch, which may be writing that file or have verified
/// it already — so removing one entry never breaks another. Whether an
/// entry only partly there holds a shared file does not count: it holds
/// nothing a removal breaks, and two entries sharing a file would otherwise
/// each keep it for the other. The other entries' locks are held while the
/// files go, so none of them begins a fetch meanwhile. Nothing but the
/// entry's own files is touched, its `.lock` file included. A file that
/// cannot be deleted stops the removal, naming it and what was deleted
/// before it.
pub fn remove(entry: &CatalogEntry, catalog: &Catalog, dir: &Path) -> Result<Removed, String> {
    let mut removed = Removed::default();
    // Each entry sharing a file, once.
    let mut others: Vec<Sharer<'_>> = Vec::new();
    for file in &entry.files {
        let mut kept_by = None;
        for other in catalog.entries().iter().filter(|other| other.id != entry.id && other.files.iter().any(|theirs| theirs.name == file.name)) {
            let keeps = match others.iter().find(|sharer| sharer.id == other.id) {
                Some(sharer) => sharer.keeps,
                None => {
                    let (keeps, lock) = if on_disk(other, dir, &file_size) == OnDisk::Complete {
                        (true, None)
                    } else {
                        // Held, or not to be opened: that entry may be being
                        // downloaded, and its file stays.
                        match super::EntryLock::take(other, dir, super::Holder::Server) {
                            Ok(lock) => (false, Some(lock)),
                            Err(_) => (true, None),
                        }
                    };
                    others.push(Sharer { id: other.id.as_str(), keeps, _lock: lock });
                    keeps
                }
            };
            if keeps && kept_by.is_none() {
                kept_by = Some(other.id.clone());
            }
        }
        if let Some(by) = kept_by {
            removed.kept.push(FileKept { file: file.name.clone(), by });
            continue;
        }
        let path = dir.join(&file.name);
        for path in [part_path(&path), path] {
            let Some(bytes) = file_size(&path) else { continue };
            if let Err(err) = std::fs::remove_file(&path) {
                let before = if removed.files.is_empty() { "nothing".to_owned() } else { removed.files.join(", ") };
                return Err(format!("{} could not be deleted: {err}; deleted before it: {before}", path.display()));
            }
            removed.files.push(path.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned()));
            removed.bytes += bytes;
        }
    }
    Ok(removed)
}

/// One row of `model list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListRow {
    pub entry: CatalogEntry,
    pub on_disk: OnDisk,
    /// What of it is on disk, partial files included ([`bytes_on_disk`]):
    /// read with `on_disk`, from the same sizes.
    pub bytes_on_disk: u64,
    /// The family the artifact's own header names, when the artifact is on
    /// disk and names one — never declared in the catalog (ADR 0047).
    pub family: Option<ModelFamily>,
}

/// Every entry of `catalog` as a row, probed under `dir` through `size_of`
/// and, for an artifact that is there, `family_of` (its header).
///
/// Each file's size is read once per listing and every row is taken from
/// that one reading (GitHub #376): a row's `on_disk` and its `bytes_on_disk`
/// never disagree, as they did when a download landed files between two
/// reads of them (`absent` with bytes).
pub fn list_rows(
    catalog: &Catalog,
    dir: &Path,
    size_of: &dyn Fn(&Path) -> Option<u64>,
    family_of: &dyn Fn(&Path) -> Option<ModelFamily>,
) -> Vec<ListRow> {
    let sizes = std::cell::RefCell::new(std::collections::HashMap::<PathBuf, Option<u64>>::new());
    let read = |path: &Path| *sizes.borrow_mut().entry(path.to_path_buf()).or_insert_with(|| size_of(path));
    catalog
        .entries()
        .iter()
        .map(|entry| {
            let artifact = entry.artifact_path(dir);
            ListRow {
                entry: entry.clone(),
                on_disk: on_disk(entry, dir, &read),
                bytes_on_disk: bytes_on_disk(entry, dir, &read),
                family: read(&artifact).and_then(|_| family_of(&artifact)),
            }
        })
        .collect()
}

/// The entries `model download` fetches (spec model-download/02, ADR 0064
/// §3): each of `ids`, matched case-insensitively as a command line's ids
/// are; every entry with `all`; the configured model, `configured`, with
/// neither. Each is fitted to the card's class `card` first
/// ([`Catalog::answering`]): an id another entry answers to on this card
/// fetches that one (`qwen3.8-27b` on a 16 GB card, the 16 GB 27B), and one
/// this card cannot load refuses the whole command — every such entry named,
/// with its class and the card's — before anything is fetched. So `--all` on
/// a card below one of the entries is refused too: name the ids it loads. A
/// card of no known class refuses nothing. Each entry once, in the order
/// named.
pub fn entries_to_fetch(catalog: &Catalog, ids: &[String], all: bool, configured: &str, card: Option<CardClass>) -> Result<Vec<CatalogEntry>, String> {
    let named: Vec<&CatalogEntry> = match (all, ids) {
        (true, []) => catalog.entries().iter().collect(),
        (true, _) => return Err("`model download --all` fetches every entry: name ids or `--all`, not both".to_owned()),
        (false, []) => vec![catalog_entry(catalog, configured)?],
        (false, ids) => ids.iter().map(|id| catalog_entry(catalog, id)).collect::<Result<_, _>>()?,
    };
    let mut entries: Vec<CatalogEntry> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    for entry in named {
        match catalog.answering(&entry.id, card) {
            Ok(fetched) => {
                let fetched = fetched.unwrap_or(entry);
                if !entries.iter().any(|earlier| earlier.id == fetched.id) {
                    entries.push(fetched.clone());
                }
            }
            Err(refusal) => refused.push(refusal.to_string()),
        }
    }
    if !refused.is_empty() {
        return Err(format!("nothing fetched: {}; `model download` fetches only what this card can load", refused.join("; ")));
    }
    Ok(entries)
}

/// The entry `id` names, or why it names none.
fn catalog_entry<'a>(catalog: &'a Catalog, id: &str) -> Result<&'a CatalogEntry, String> {
    catalog.entry(id).ok_or_else(|| format!("`{id}` is not in the catalog (the ids are {})", catalog.ids().join(", ")))
}

/// The rows as `model list` prints them.
pub fn render_list(rows: &[ListRow], format: ListFormat) -> String {
    match format {
        ListFormat::Json => {
            let rows: Vec<Value> = rows
                .iter()
                .map(|row| {
                    json!({
                        "id": row.entry.id,
                        "layer": row.entry.layer.as_str(),
                        "repo": row.entry.repo,
                        "revision": row.entry.revision,
                        "bytes": row.entry.total_bytes(),
                        "on_disk": row.on_disk.as_str(),
                        "family": row.family.map(|family| family.name()),
                    })
                })
                .collect();
            let mut text = serde_json::to_string_pretty(&rows).expect("plain JSON");
            text.push('\n');
            text
        }
        ListFormat::Text => {
            let mut table = vec![["ID".to_owned(), "LAYER".to_owned(), "REPO@REVISION".to_owned(), "SIZE".to_owned(), "ON DISK".to_owned(), "FAMILY".to_owned()]];
            for row in rows {
                table.push([
                    row.entry.id.clone(),
                    row.entry.layer.as_str().to_owned(),
                    format!("{}@{}", row.entry.repo, row.entry.revision),
                    format!("{:.1} GiB", row.entry.gib()),
                    row.on_disk.as_str().to_owned(),
                    row.family.map_or("—", |family| family.name()).to_owned(),
                ]);
            }
            let widths: Vec<usize> = (0..6).map(|column| table.iter().map(|line| line[column].chars().count()).max().unwrap_or(0)).collect();
            let mut text = String::new();
            for line in &table {
                let cells: Vec<String> = line.iter().zip(&widths).map(|(cell, width)| format!("{cell:<width$}")).collect();
                text.push_str(cells.join("  ").trim_end());
                text.push('\n');
            }
            text
        }
    }
}

/// Carry out `command`: what it produced to `out` (stdout — the artifact
/// paths fetched, or the listing), what it is doing to `progress` (stderr,
/// a line at a time). Returns the exit code: 0 only when every entry named
/// ended verified.
pub async fn run(command: ModelCommand, out: &mut dyn Write, progress: Arc<dyn Fn(&str) + Send + Sync>) -> i32 {
    match command {
        ModelCommand::List { catalog, dir, format } => {
            let family_of = |path: &Path| crate::loader::artifact_family(path).ok().flatten();
            let rows = list_rows(&catalog, &dir, &file_size, &family_of);
            let _ = out.write_all(render_list(&rows, format).as_bytes());
            0
        }
        ModelCommand::Download { entries, dir, endpoint, token, free_margin } => {
            let downloader = match Downloader::new(&endpoint, token) {
                Ok(downloader) => downloader.reporting(Report::Lines(Arc::clone(&progress))).holding_as(super::Holder::Command),
                Err(err) => {
                    progress(&format!("ignis-server model download: {err}"));
                    return 1;
                }
            };
            let mut fetched = Vec::new();
            let mut failed = Vec::new();
            for entry in &entries {
                progress(&format!(
                    "{}: {} files, {:.1} GiB, from {}/{} @ {} into {}",
                    entry.id,
                    entry.files.len(),
                    entry.gib(),
                    endpoint.trim_end_matches('/'),
                    entry.repo,
                    entry.revision,
                    dir.display()
                ));
                // GitHub #378: the room is read before each entry, as the
                // server reads it: the entry before may have used it.
                if let Err(err) = room_for(entry, &dir, free_margin) {
                    progress(&format!("{}: failed: {err}", entry.id));
                    failed.push(entry.id.as_str());
                    continue;
                }
                match downloader.fetch(entry, &dir).await {
                    Ok(path) => {
                        progress(&format!("{}: every file verified", entry.id));
                        fetched.push(path);
                    }
                    // The next entry is still fetched: one that cannot be
                    // served says nothing about the others.
                    Err(err) => {
                        progress(&format!("{}: failed: {err}", entry.id));
                        failed.push(entry.id.as_str());
                    }
                }
            }
            for path in &fetched {
                let _ = writeln!(out, "{}", path.display());
            }
            if failed.is_empty() {
                return 0;
            }
            progress(&format!("ignis-server model download: not verified: {}", failed.join(", ")));
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// A directory listing as a size probe.
    fn sizes(files: &[(&str, u64)]) -> impl Fn(&Path) -> Option<u64> {
        let files: BTreeMap<PathBuf, u64> = files.iter().map(|(name, bytes)| (Path::new("m").join(name), *bytes)).collect();
        move |path: &Path| files.get(path).copied()
    }

    /// Spec model-download/02 AC 13, the on-disk column: complete only when
    /// every file has its pinned size; a file or a `.part` of it is partial.
    #[test]
    fn an_entry_is_complete_partial_or_absent_by_its_files_sizes() {
        let entry = Catalog::built_in().entry("qwen3.8-27b").unwrap();
        let (sidecar, artifact) = (&entry.files[0], &entry.files[1]);
        let dir = Path::new("m");
        for (files, want) in [
            (vec![], OnDisk::Absent),
            (vec![(sidecar.name.clone(), sidecar.bytes), (artifact.name.clone(), artifact.bytes)], OnDisk::Complete),
            (vec![(sidecar.name.clone(), sidecar.bytes)], OnDisk::Partial),
            (vec![(sidecar.name.clone(), sidecar.bytes), (format!("{}.part", artifact.name), 4096)], OnDisk::Partial),
            (vec![(format!("{}.part", sidecar.name), 1)], OnDisk::Partial),
            (vec![(sidecar.name.clone(), sidecar.bytes), (artifact.name.clone(), 7)], OnDisk::Partial),
        ] {
            let listed: Vec<(&str, u64)> = files.iter().map(|(name, bytes)| (name.as_str(), *bytes)).collect();
            assert_eq!(on_disk(entry, dir, &sizes(&listed)), want, "{files:?}");
        }
    }

    /// GitHub #375: the bytes of an entry on disk count each file once — at
    /// its final name when it is there, else its `.part` — so a running or
    /// cancelled download reads as what it has fetched so far; never more
    /// than a file's pin, whatever a file of another size holds.
    #[test]
    fn the_bytes_on_disk_count_each_file_at_its_final_name_or_its_part() {
        let entry = Catalog::built_in().entry("qwen3.8-27b").unwrap();
        let (sidecar, artifact) = (&entry.files[0], &entry.files[1]);
        let dir = Path::new("m");
        let part = format!("{}.part", artifact.name);
        for (files, want) in [
            (vec![], 0),
            (vec![(format!("{}.part", sidecar.name), 3)], 3),
            (vec![(sidecar.name.clone(), sidecar.bytes), (part.clone(), 4096)], sidecar.bytes + 4096),
            (vec![(sidecar.name.clone(), sidecar.bytes), (artifact.name.clone(), artifact.bytes)], entry.total_bytes()),
            // A stale `.part` beside a complete file is not counted twice.
            (vec![(sidecar.name.clone(), sidecar.bytes), (artifact.name.clone(), artifact.bytes), (part.clone(), 4096)], entry.total_bytes()),
            (vec![(sidecar.name.clone(), sidecar.bytes + 99)], sidecar.bytes),
        ] {
            let listed: Vec<(&str, u64)> = files.iter().map(|(name, bytes)| (name.as_str(), *bytes)).collect();
            assert_eq!(bytes_on_disk(entry, dir, &sizes(&listed)), want, "{files:?}");
        }
    }

    /// GitHub #376 (13a45bcf): a listing reads each file's size once and
    /// takes a row's state on disk and its bytes from that one reading, so a
    /// download landing files while the catalog is listed never gives a row
    /// that is `absent` with bytes, or `partial` by one reading and `absent`
    /// by another. The probe is such a download: each file is missing at its
    /// first reading and whole at any later one.
    #[test]
    fn a_listing_reads_each_file_once_and_each_row_agrees_with_itself() {
        let catalog = Catalog::built_in();
        let entry = catalog.entry("qwen3.8-27b").unwrap();
        let dir = Path::new("m");
        let reads = std::cell::RefCell::new(BTreeMap::<PathBuf, usize>::new());
        let landing = |path: &Path| {
            let mut reads = reads.borrow_mut();
            let count = reads.entry(path.to_path_buf()).or_insert(0);
            *count += 1;
            let whole = entry.files.iter().find(|file| dir.join(&file.name) == path).map(|file| file.bytes);
            if *count == 1 { None } else { whole }
        };
        let rows = list_rows(&catalog, dir, &landing, &|_| None);
        let row = rows.iter().find(|row| row.entry.id == entry.id).unwrap();
        assert_eq!((row.on_disk, row.bytes_on_disk), (OnDisk::Absent, 0), "one reading: nothing there yet");
        let reads = reads.into_inner();
        assert!(reads.contains_key(&dir.join(&entry.files[1].name)), "{reads:?}");
        assert!(reads.values().all(|&count| count == 1), "each file read once: {reads:?}");
    }

    /// GitHub #378: the free space must cover the bytes still missing plus
    /// the margin, exactly at the boundary too; a refusal names the bytes
    /// needed and the bytes free, and a margin past any disk never wraps.
    #[test]
    fn a_download_fits_when_the_free_space_covers_what_is_missing_plus_the_margin() {
        let dir = Path::new("m");
        assert_eq!(fits("acme", dir, 3_000, 4_024, 1_024), Ok(()));
        let short = fits("acme", dir, 3_000, 4_023, 1_024).unwrap_err();
        assert!(short.starts_with("not enough room for acme under m: 4024 bytes needed (3000 still missing plus download.free_margin 1024), 4023 bytes free"), "{short}");
        let past_any_disk = fits("acme", dir, 3_000, u64::MAX - 1, u64::MAX).unwrap_err();
        assert!(past_any_disk.contains(&format!("{} bytes needed", u64::MAX)), "{past_any_disk}");
    }

    /// AC 13: one row per entry — layer, repo@revision, size, on-disk state,
    /// and the family read from the artifact only when it is there.
    #[test]
    fn the_listing_shows_every_entry_with_its_layer_state_and_family() {
        let catalog = super::super::catalog::parse_operator(
            "models:\n  - id: acme-ft\n    repo: acme/ft\n    revision: v1\n    artifact: a.ninfer\n    card_class: 32\n    files:\n      - { name: a.ninfer.graft.json, bytes: 1, sha256: 0000000000000000000000000000000000000000000000000000000000000001 }\n      - { name: a.ninfer, bytes: 2, sha256: 0000000000000000000000000000000000000000000000000000000000000002 }\n",
            crate::config::file::Format::Yaml,
            "acme.yaml",
        )
        .unwrap();
        let entry = catalog.entry("qwen3.8-27b").unwrap();
        let on = [(entry.files[0].name.as_str(), entry.files[0].bytes), (entry.files[1].name.as_str(), entry.files[1].bytes), ("a.ninfer", 2)];
        let probed = std::cell::RefCell::new(Vec::new());
        let family_of = |path: &Path| {
            probed.borrow_mut().push(path.to_path_buf());
            path.ends_with(&entry.artifact).then_some(ModelFamily::Qwen38_27b)
        };
        let rows = list_rows(&catalog, Path::new("m"), &sizes(&on), &family_of);
        assert_eq!(rows.len(), 11);
        assert_eq!(
            rows.iter().map(|row| (row.entry.id.as_str(), row.on_disk, row.family)).collect::<Vec<_>>(),
            [
                ("qwen3.8-27b", OnDisk::Complete, Some(ModelFamily::Qwen38_27b)),
                ("qwen3.8-27b-abliterated", OnDisk::Absent, None),
                ("qwen3.8-flash-next", OnDisk::Absent, None),
                ("ignis-image-2.1-turbo", OnDisk::Absent, None),
                ("qwen3.8-27b-16g", OnDisk::Absent, None),
                ("qwen3.8-27b-abliterated-16g", OnDisk::Absent, None),
                ("qwen3-asr-1.7b", OnDisk::Absent, None),
                ("qwen3-asr-1.7b-q8", OnDisk::Absent, None),
                ("qwen3-asr-0.6b", OnDisk::Absent, None),
                ("nemotron-3-diarization", OnDisk::Absent, None),
                // On disk, but its header names neither family: `—`.
                ("acme-ft", OnDisk::Partial, None),
            ]
        );
        assert_eq!(probed.into_inner(), [Path::new("m").join(&entry.artifact), Path::new("m").join("a.ninfer")], "only an artifact on disk is opened");

        let text = render_list(&rows, ListFormat::Text);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 12, "{text}");
        assert!(lines[0].starts_with("ID") && lines[0].contains("REPO@REVISION") && lines[0].contains("FAMILY"), "{text}");
        assert!(lines[1].contains("built-in") && lines[1].contains("gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer@e961b419b672e183aa55df8c4b975abc82006e8a"), "{text}");
        assert!(lines[1].contains("18.1 GiB") && lines[1].contains("complete") && lines[1].ends_with("Qwen3.8-27B"), "{text}");
        assert!(lines[3].contains("67.8 GiB") && lines[3].contains("absent") && lines[3].ends_with('—'), "{text}");
        assert!(lines[5].contains("built-in") && lines[5].contains("gpillon/Qwen3.8-27B-trellis-3bpw-ignis@f5bbd2cc") && lines[5].contains("12.5 GiB"), "{text}");
        assert!(lines[6].contains("built-in") && lines[6].contains("gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis@94a8bfc0") && lines[6].contains("12.5 GiB"), "{text}");
        assert!(lines[10].contains("gpillon/Nemotron-3-Diarization-GGUF-ignis@d46d59d0") && lines[10].contains("absent"), "{text}");
        assert!(lines[11].contains("operator") && lines[11].contains("acme/ft@v1") && lines[11].contains("partial"), "{text}");
        assert!(lines[7].contains("gpillon/Qwen3-ASR-1.7B-GGUF-ignis@f8a5263e") && lines[7].contains("1.5 GiB") && lines[7].contains("absent"), "{text}");

        let json: Vec<Value> = serde_json::from_str(&render_list(&rows, ListFormat::Json)).unwrap();
        assert_eq!(json[0]["id"], "qwen3.8-27b");
        assert_eq!(json[0]["layer"], "built-in");
        assert_eq!(json[0]["revision"], "e961b419b672e183aa55df8c4b975abc82006e8a");
        assert_eq!(json[0]["bytes"], entry.total_bytes());
        assert_eq!(json[0]["on_disk"], "complete");
        assert_eq!(json[0]["family"], "Qwen3.8-27B");
        assert_eq!(json[10]["layer"], "operator");
        assert_eq!(json[10]["family"], Value::Null);
    }

    fn fetched(catalog: &Catalog, ids: &[&str], all: bool, card: Option<CardClass>) -> Result<Vec<String>, String> {
        let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        entries_to_fetch(catalog, &ids, all, crate::config::DEFAULT_MODEL, card).map(|entries| entries.into_iter().map(|entry| entry.id).collect())
    }

    /// ADR 0064 §2-3: on a 16 GB card the default id — named, or the
    /// configured model with none — fetches the entry that answers to it,
    /// each entry once.
    #[test]
    fn a_download_fetches_the_entry_that_serves_an_id_on_the_card() {
        let catalog = super::super::catalog::testing::built_in();
        let sixteen = Some(CardClass::Gb16);
        assert_eq!(fetched(&catalog, &[], false, sixteen), Ok(vec!["qwen3.8-27b-16g".to_owned()]), "the configured model");
        assert_eq!(fetched(&catalog, &["QWEN3.8-27B", "qwen3.8-27b-16g"], false, sixteen), Ok(vec!["qwen3.8-27b-16g".to_owned()]));
        for card in [Some(CardClass::Gb32), None] {
            assert_eq!(fetched(&catalog, &["qwen3.8-27b"], false, card), Ok(vec!["qwen3.8-27b".to_owned()]), "{card:?}: nothing changes");
            assert_eq!(fetched(&catalog, &[], true, card).unwrap().len(), 10, "{card:?}");
        }
    }

    /// ADR 0074: an alias that is another entry's id resolves for a download
    /// as for a request — `model download qwen3.8-27b-abliterated` on a 16 GB
    /// card fetches the built-in abliterated 16 GB build (GitHub #423), and
    /// on a larger card the id's own entry.
    #[test]
    fn a_download_of_the_uncensored_variant_fetches_its_16gb_build_on_a_16gb_card() {
        let catalog = super::super::catalog::testing::built_in();
        let abliterated = ["qwen3.8-27b-abliterated"];
        assert_eq!(fetched(&catalog, &abliterated, false, Some(CardClass::Gb16)), Ok(vec!["qwen3.8-27b-abliterated-16g".to_owned()]));
        for card in [Some(CardClass::Gb32), Some(CardClass::Gb96), None] {
            assert_eq!(fetched(&catalog, &abliterated, false, card), Ok(vec!["qwen3.8-27b-abliterated".to_owned()]), "{card:?}");
        }
    }

    /// ADR 0064 §3: a download of an entry above the card's class is
    /// refused whole, before anything is fetched — each such entry named
    /// with both classes, `--all` too — and the default model on a 16 GB
    /// card is, in a catalog without the 16 GB 27B.
    #[test]
    fn a_download_of_an_entry_above_the_card_class_is_refused_before_any_fetch() {
        let catalog = super::super::catalog::testing::built_in();
        let sixteen = Some(CardClass::Gb16);
        let err = fetched(&catalog, &["qwen3.8-27b-16g", "qwen3.8-flash-next", "ignis-image-2.1-turbo"], false, sixteen).unwrap_err();
        assert!(err.starts_with("nothing fetched"), "{err}");
        for said in ["qwen3.8-flash-next needs a 32 GB card", "ignis-image-2.1-turbo needs a 32 GB card", "16 GB class"] {
            assert!(err.contains(said), "{said}: {err}");
        }
        // `--all` names every entry, each through the same resolve step: the
        // 27B's own and the uncensored variant's resolve to their 16 GB
        // entries, the others are refused.
        let err = fetched(&catalog, &[], true, sixteen).unwrap_err();
        assert!(err.contains("qwen3.8-flash-next needs") && !err.contains("qwen3.8-27b needs") && !err.contains("qwen3.8-27b-abliterated needs"), "{err}");
        let err = fetched(&super::super::catalog::testing::thirty_two(), &[], false, sixteen).unwrap_err();
        assert!(err.contains("qwen3.8-27b needs a 32 GB card"), "{err}");
        assert!(fetched(&catalog, &["acme-nothing"], false, sixteen).unwrap_err().contains("not in the catalog"));
    }
}
