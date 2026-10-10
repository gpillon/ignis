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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// One row of `model list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListRow {
    pub entry: CatalogEntry,
    pub on_disk: OnDisk,
    /// The family the artifact's own header names, when the artifact is on
    /// disk and names one — never declared in the catalog (ADR 0047).
    pub family: Option<ModelFamily>,
}

/// Every entry of `catalog` as a row, probed under `dir` through `size_of`
/// and, for an artifact that is there, `family_of` (its header).
pub fn list_rows(
    catalog: &Catalog,
    dir: &Path,
    size_of: &dyn Fn(&Path) -> Option<u64>,
    family_of: &dyn Fn(&Path) -> Option<ModelFamily>,
) -> Vec<ListRow> {
    catalog
        .entries()
        .iter()
        .map(|entry| {
            let artifact = entry.artifact_path(dir);
            ListRow {
                entry: entry.clone(),
                on_disk: on_disk(entry, dir, size_of),
                family: size_of(&artifact).and_then(|_| family_of(&artifact)),
            }
        })
        .collect()
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
            let size_of = |path: &Path| std::fs::metadata(path).ok().filter(|meta| meta.is_file()).map(|meta| meta.len());
            let family_of = |path: &Path| crate::loader::artifact_family(path).ok().flatten();
            let rows = list_rows(&catalog, &dir, &size_of, &family_of);
            let _ = out.write_all(render_list(&rows, format).as_bytes());
            0
        }
        ModelCommand::Download { entries, dir, endpoint, token } => {
            let downloader = match Downloader::new(&endpoint, token) {
                Ok(downloader) => downloader.reporting(Report::Lines(Arc::clone(&progress))),
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

    /// AC 13: one row per entry — layer, repo@revision, size, on-disk state,
    /// and the family read from the artifact only when it is there.
    #[test]
    fn the_listing_shows_every_entry_with_its_layer_state_and_family() {
        let catalog = super::super::catalog::parse_operator(
            "models:\n  - id: acme-ft\n    repo: acme/ft\n    revision: v1\n    artifact: a.ninfer\n    files:\n      - { name: a.ninfer.graft.json, bytes: 1, sha256: 0000000000000000000000000000000000000000000000000000000000000001 }\n      - { name: a.ninfer, bytes: 2, sha256: 0000000000000000000000000000000000000000000000000000000000000002 }\n",
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
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.iter().map(|row| (row.entry.id.as_str(), row.on_disk, row.family)).collect::<Vec<_>>(),
            [
                ("qwen3.8-27b", OnDisk::Complete, Some(ModelFamily::Qwen38_27b)),
                ("qwen3.8-27b-abliterated", OnDisk::Absent, None),
                ("qwen3.8-flash-next", OnDisk::Absent, None),
                // On disk, but its header names neither family: `—`.
                ("acme-ft", OnDisk::Partial, None),
            ]
        );
        assert_eq!(probed.into_inner(), [Path::new("m").join(&entry.artifact), Path::new("m").join("a.ninfer")], "only an artifact on disk is opened");

        let text = render_list(&rows, ListFormat::Text);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 5, "{text}");
        assert!(lines[0].starts_with("ID") && lines[0].contains("REPO@REVISION") && lines[0].contains("FAMILY"), "{text}");
        assert!(lines[1].contains("built-in") && lines[1].contains("gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer@e961b419b672e183aa55df8c4b975abc82006e8a"), "{text}");
        assert!(lines[1].contains("18.1 GiB") && lines[1].contains("complete") && lines[1].ends_with("Qwen3.8-27B"), "{text}");
        assert!(lines[3].contains("67.8 GiB") && lines[3].contains("absent") && lines[3].ends_with('—'), "{text}");
        assert!(lines[4].contains("operator") && lines[4].contains("acme/ft@v1") && lines[4].contains("partial"), "{text}");

        let json: Vec<Value> = serde_json::from_str(&render_list(&rows, ListFormat::Json)).unwrap();
        assert_eq!(json[0]["id"], "qwen3.8-27b");
        assert_eq!(json[0]["layer"], "built-in");
        assert_eq!(json[0]["revision"], "e961b419b672e183aa55df8c4b975abc82006e8a");
        assert_eq!(json[0]["bytes"], entry.total_bytes());
        assert_eq!(json[0]["on_disk"], "complete");
        assert_eq!(json[0]["family"], "Qwen3.8-27B");
        assert_eq!(json[3]["layer"], "operator");
        assert_eq!(json[3]["family"], Value::Null);
    }
}
