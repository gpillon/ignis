//! The catalog (ADR 0047, spec
//! `docs/specs/model-download/02-catalog-and-model-command.md`): every model
//! the server knows how to fetch, each pinned file by file.
//!
//! Two layers, with two trust standings. The **built-in catalog**
//! (`catalog.yaml`, compiled in with `include_str!`) holds the owner's
//! releases, and its pins are the trust anchor ADR 0033 made them. The
//! **operator catalog** is the file `download.catalog` names, in the same
//! format: the operator's own word about the operator's own files, never a
//! built-in entry's — an operator id equal to a built-in one is refused, so
//! no editable file can silently re-pin an official id.
//!
//! Both are read by [`parse`], which refuses a catalog whole: every refusal
//! names the file, the entry and the key, so the operator knows which line
//! to fix, and nothing downstream ever sees an entry that could write outside
//! its destination directory or fetch a file it cannot verify.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde_json::{Map, Value};

use crate::config::file::{Files, Format};

/// The built-in catalog's text, as shipped.
const BUILT_IN_TEXT: &str = include_str!("catalog.yaml");

/// The built-in catalog, parsed once. A binary whose own catalog does not
/// parse is a build that must not ship: the test below parses it on every
/// `cargo test`.
pub static BUILT_IN_CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    let document = Format::Yaml.read(BUILT_IN_TEXT).expect("the built-in catalog is YAML");
    let entries = parse(&document, "built into this binary", CatalogLayer::BuiltIn).expect("the built-in catalog is valid");
    Catalog { entries }
});

/// The keys an entry takes.
const ENTRY_KEYS: [&str; 5] = ["id", "repo", "revision", "artifact", "files"];

/// The keys one of an entry's files takes.
const FILE_KEYS: [&str; 3] = ["name", "bytes", "sha256"];

/// Which catalog an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogLayer {
    /// Compiled into the binary: the owner's releases.
    BuiltIn,
    /// The file `download.catalog` names.
    Operator,
}

impl CatalogLayer {
    /// `built-in` or `operator`, as `model list` shows it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BuiltIn => "built-in",
            Self::Operator => "operator",
        }
    }
}

/// One file an entry needs: the artifact, a sidecar, a companion container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogFile {
    /// Its name in the repo and on disk — a plain name, never a path, so the
    /// destination stays flat and nothing lands outside it.
    pub name: String,
    /// Its exact size.
    pub bytes: u64,
    /// Its SHA-256, lowercase hex.
    pub sha256: String,
}

/// One published model (`CONTEXT.md`, **Catalog entry**): its served id,
/// where it is published, and every file it needs with its byte count and
/// digest pinned.
///
/// The pins are never read from the repo: the published `.sha256` sits in
/// the same trust domain as the artifact it describes, so it can attest
/// nothing about it (ADR 0033).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogEntry {
    /// The served id: what `--model-id` names and `GET /v1/models` reports.
    pub id: String,
    /// The repo, `<owner>/<name>`.
    pub repo: String,
    /// The commit, tag or branch the pins are of. The built-in entries name
    /// a commit, so a republish never breaks a binary already shipped.
    pub revision: String,
    /// The file the server loads, one of [`CatalogEntry::files`].
    pub artifact: String,
    /// Every file, as the catalog lists them.
    pub files: Vec<CatalogFile>,
    /// Which catalog it came from.
    pub layer: CatalogLayer,
}

impl CatalogEntry {
    /// Where the artifact lives under `dir` (flat: every file of an entry
    /// sits beside the others under the name the repo publishes it under).
    pub fn artifact_path(&self, dir: &Path) -> PathBuf {
        dir.join(&self.artifact)
    }

    /// Every file's bytes together: what a fetch of the whole entry spends.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.bytes).sum()
    }

    /// [`CatalogEntry::total_bytes`] in GiB — for the question and the
    /// listing, never for a decision.
    pub fn gib(&self) -> f64 {
        (self.total_bytes() as f64) / (1024.0 * 1024.0 * 1024.0)
    }

    /// The files in the order they are fetched: ascending byte count, so a
    /// repo that cannot serve its small files fails in a second instead of
    /// after the large one.
    pub fn files_by_size(&self) -> Vec<&CatalogFile> {
        let mut files: Vec<&CatalogFile> = self.files.iter().collect();
        files.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.name.cmp(&b.name)));
        files
    }
}

/// The models a server knows how to fetch: the built-in entries, then the
/// operator's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catalog {
    entries: Vec<CatalogEntry>,
}

impl Catalog {
    /// The built-in catalog alone.
    pub fn built_in() -> &'static Catalog {
        &BUILT_IN_CATALOG
    }

    /// Every entry, the built-in ones first.
    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// The entry for `id`, matched case-insensitively: a served id is a
    /// name, not a checksum.
    pub fn entry(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.id.eq_ignore_ascii_case(id.trim()))
    }

    /// Every entry's id, in order — what an unknown id is answered with.
    pub fn ids(&self) -> Vec<&str> {
        self.entries.iter().map(|entry| entry.id.as_str()).collect()
    }
}

/// A catalog that cannot be used: unreadable, malformed, or naming an entry
/// it may not. Refuses the start, and every `model` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogError(pub String);

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CatalogError {}

/// The catalog a server runs with: the built-in one, and the operator's file
/// over it when `path` (`download.catalog`, already resolved against the
/// config file that named it) names one. A path that cannot be read is
/// refused, not skipped: an operator who named a catalog meant it.
pub fn load(files: &dyn Files, path: Option<&Path>) -> Result<Catalog, CatalogError> {
    let Some(path) = path else {
        return Ok(Catalog::built_in().clone());
    };
    let text = files
        .read(path)
        .map_err(|err| CatalogError(format!("the catalog {} (`download.catalog`) cannot be read: {err}", path.display())))?;
    parse_operator(&text, Format::of_path(path).unwrap_or(Format::Yaml), &path.display().to_string())
}

/// An operator catalog's text, read in `format` and put after the built-in
/// entries. `origin` (the file's path) is named in every refusal.
pub fn parse_operator(text: &str, format: Format, origin: &str) -> Result<Catalog, CatalogError> {
    let document = format
        .read(text)
        .map_err(|err| CatalogError(format!("the catalog {origin} is not valid: {err}")))?;
    let operator = parse(&document, origin, CatalogLayer::Operator)?;
    let mut entries = Catalog::built_in().entries.clone();
    for entry in operator {
        if let Some(official) = Catalog::built_in().entry(&entry.id) {
            return Err(CatalogError(format!(
                "the catalog {origin}: entry `{}` reuses the built-in id `{}` — an operator entry never re-pins an official model (a mirror of it is `download.endpoint`, a variant of it a new id)",
                entry.id, official.id
            )));
        }
        entries.push(entry);
    }
    Ok(Catalog { entries })
}

/// A catalog document's entries, every one validated: the one reader both
/// layers go through, so the built-in catalog is held to the rules an
/// operator's is.
pub fn parse(document: &Value, origin: &str, layer: CatalogLayer) -> Result<Vec<CatalogEntry>, CatalogError> {
    let refuse = |reason: String| CatalogError(format!("the catalog {origin}: {reason}"));
    let Value::Object(top) = document else {
        return Err(refuse("must be a map holding a `models` list".to_owned()));
    };
    if let Some(key) = top.keys().find(|key| *key != "models") {
        return Err(refuse(format!("`{key}` is not a catalog key (the one key is `models`)")));
    }
    let Some(Value::Array(models)) = top.get("models") else {
        return Err(refuse("`models` must be a list of entries".to_owned()));
    };
    let mut entries: Vec<CatalogEntry> = Vec::new();
    for (index, value) in models.iter().enumerate() {
        let named = match value.get("id").and_then(Value::as_str).map(str::trim).filter(|id| !id.is_empty()) {
            Some(id) => format!("entry {} (`{id}`)", index + 1),
            None => format!("entry {}", index + 1),
        };
        let entry = read_entry(value, layer).map_err(|reason| refuse(format!("{named}: {reason}")))?;
        if let Some(earlier) = entries.iter().find(|earlier| earlier.id.eq_ignore_ascii_case(&entry.id)) {
            return Err(refuse(format!(
                "{named}: the id is already `{}`'s, an earlier entry (ids are compared case-insensitively)",
                earlier.id
            )));
        }
        entries.push(entry);
    }
    Ok(entries)
}

/// One entry, or why it is refused (the caller names the entry).
fn read_entry(value: &Value, layer: CatalogLayer) -> Result<CatalogEntry, String> {
    let Value::Object(map) = value else {
        return Err(format!("must be a map of {}", ENTRY_KEYS.join(", ")));
    };
    known_keys(map, &ENTRY_KEYS, "an entry")?;
    let id = text(map, "id")?;
    let repo = text(map, "repo")?;
    let plain_part = |part: &str| plain_name(part) && !part.contains(char::is_whitespace);
    if !repo.split_once('/').is_some_and(|(owner, name)| plain_part(owner) && plain_part(name)) {
        return Err(format!("`repo` must be `<owner>/<name>`, got `{repo}`"));
    }
    let revision = text(map, "revision")?;
    let artifact = text(map, "artifact")?;
    let Some(Value::Array(listed)) = map.get("files") else {
        return Err("`files` must be a list of `{ name, bytes, sha256 }`".to_owned());
    };
    let mut files: Vec<CatalogFile> = Vec::new();
    for (index, value) in listed.iter().enumerate() {
        let named = match value.get("name").and_then(Value::as_str) {
            Some(name) => format!("file {} (`{name}`)", index + 1),
            None => format!("file {}", index + 1),
        };
        let file = read_file(value).map_err(|reason| format!("{named}: {reason}"))?;
        if files.iter().any(|earlier| earlier.name == file.name) {
            return Err(format!("{named}: listed twice"));
        }
        files.push(file);
    }
    if !files.iter().any(|file| file.name == artifact) {
        return Err(format!("`artifact` `{artifact}` is not among its `files`"));
    }
    let sidecars: Vec<String> = crate::loader::SIDECAR_SUFFIXES.iter().map(|suffix| format!("{artifact}{suffix}")).collect();
    if !files.iter().any(|file| sidecars.contains(&file.name)) {
        return Err(format!(
            "no file is the artifact's sidecar ({}): the loader refuses a load without one",
            sidecars.join(" or ")
        ));
    }
    Ok(CatalogEntry { id, repo, revision, artifact, files, layer })
}

/// One of an entry's files, or why it is refused.
fn read_file(value: &Value) -> Result<CatalogFile, String> {
    let Value::Object(map) = value else {
        return Err(format!("must be a map of {}", FILE_KEYS.join(", ")));
    };
    known_keys(map, &FILE_KEYS, "a file")?;
    let name = text(map, "name")?;
    if !plain_name(&name) {
        return Err(format!(
            "`{name}` is not a plain file name (a path separator, `..` or an absolute path): a catalog never writes outside the destination directory"
        ));
    }
    let bytes = match map.get("bytes") {
        Some(Value::Number(number)) => number.as_u64().ok_or_else(|| format!("`bytes` must be a whole number of bytes, got {number}"))?,
        Some(other) => return Err(format!("`bytes` must be a whole number of bytes, got {other}")),
        None => return Err("has no `bytes`".to_owned()),
    };
    if bytes == 0 {
        return Err("`bytes` is 0: an empty file pins nothing".to_owned());
    }
    let sha256 = text(map, "sha256")?;
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(format!("`sha256` must be 64 lowercase hex digits, got `{sha256}`"));
    }
    Ok(CatalogFile { name, bytes, sha256 })
}

/// Refuse a key `map` should not hold, naming it and the ones it may.
fn known_keys(map: &Map<String, Value>, keys: &[&str], what: &str) -> Result<(), String> {
    match map.keys().find(|key| !keys.contains(&key.as_str())) {
        Some(key) => Err(format!("`{key}` is not a key of {what} (the keys are {})", keys.join(", "))),
        None => Ok(()),
    }
}

/// `map[key]` as non-empty text, trimmed.
fn text(map: &Map<String, Value>, key: &str) -> Result<String, String> {
    match map.get(key) {
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(text.trim().to_owned()),
        Some(Value::String(_)) | Some(Value::Null) | None => Err(format!("`{key}` is empty or missing")),
        Some(other) => Err(format!("`{key}` must be text, got {other}")),
    }
}

/// A name that stays inside the directory it is joined to on every host:
/// no separator of either kind (`\` is an ordinary character on Linux and a
/// separator on Windows), no drive (`C:x` replaces the directory on
/// Windows), and neither `.` nor `..`.
fn plain_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec's table, written out again: what the built-in catalog must
    /// hold, value for value (`id`, `repo`, `revision`, `artifact`, then each
    /// file's name, bytes and SHA-256 in the order listed).
    #[allow(clippy::type_complexity)]
    const RELEASES: [(&str, &str, &str, &str, &[(&str, u64, &str)]); 3] = [
        (
            "qwen3.8-27b",
            "gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer",
            "e961b419b672e183aa55df8c4b975abc82006e8a",
            "qwen3_8_27b_nvfp4full-v2.ninfer",
            &[
                ("qwen3_8_27b_nvfp4full-v2.ninfer.graft.json", 6_981, "b87e3c005fb1daf7d6b208da52790f95a3f478732f767f87417a5d7fc23491d1"),
                ("qwen3_8_27b_nvfp4full-v2.ninfer", 19_406_942_468, "abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af"),
            ],
        ),
        (
            "qwen3.8-27b-abliterated",
            "gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer",
            "ef3216949e42c42fc713feffb2e05849d57f0f3b",
            "qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer",
            &[
                (
                    "qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer.graft.json",
                    41_707,
                    "09b0df892c314563192b0a4f102139939d4ef6261fcd210439d4e2636d12e6fe",
                ),
                (
                    "qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer",
                    19_406_942_468,
                    "18954280c794cb2ff1fc24ada8158de1df11bf0a0a2ea63f109045af48905ef1",
                ),
            ],
        ),
        (
            "qwen3.8-flash-next",
            "gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis",
            "8d94db83cd8e58d6be31e796723e5e35b960638c",
            "qwen3_8_flash_next_trellis_a25-v2.ninfer",
            &[
                (
                    "qwen3_8_flash_next_mtp_3p0-v2.ninfer.conversion.json",
                    179_029,
                    "a5b2c10fe0745cfe6b8e2914135e44903cfb80acb3f0db21e549e76641f14127",
                ),
                (
                    "qwen3_8_flash_next_trellis_a25-v2.ninfer.conversion.json",
                    1_183_942,
                    "9858d9fb03dd66fa2558b5dc6aa6463ee70f00024ffbd746545b231c9514f7a0",
                ),
                ("qwen3_8_flash_next_mtp_3p0-v2.ninfer", 1_037_905_920, "432fe5c0838027047c5b46c4fde526f3d3ef3d126f00fed1213921f8705988d2"),
                ("qwen3_8_flash_next_trellis_a25-v2.ninfer", 71_760_711_680, "12e3b265b0678ba935a4b05d1b612ba4b68b3a96c3026d818ad65838985fd864"),
            ],
        ),
    ];

    /// A minimal valid operator entry, YAML, for the refusal table to break
    /// one thing at a time.
    const OPERATOR: &str = "models:
  - id: acme-qwen3.8-27b-ft
    repo: acme/qwen-ft-ninfer
    revision: v1
    artifact: acme_ft.ninfer
    files:
      - { name: acme_ft.ninfer.graft.json, bytes: 48211, sha256: 9c1e000000000000000000000000000000000000000000000000000000000001 }
      - { name: acme_ft.ninfer, bytes: 19406942468, sha256: abb1000000000000000000000000000000000000000000000000000000000002 }
";

    fn operator(text: &str) -> Result<Catalog, CatalogError> {
        parse_operator(text, Format::Yaml, "acme.catalog.yaml")
    }

    /// AC 1: the built-in catalog is the data file, it parses under every
    /// rule an operator's does, and it holds exactly the spec's three
    /// releases with those values.
    #[test]
    fn the_built_in_catalog_holds_exactly_the_three_releases_with_their_pins() {
        let document = Format::Yaml.read(BUILT_IN_TEXT).expect("YAML");
        let parsed = parse(&document, "built into this binary", CatalogLayer::BuiltIn).expect("every invariant holds");
        assert_eq!(parsed, BUILT_IN_CATALOG.entries);
        assert_eq!(parsed.len(), RELEASES.len());
        for (entry, (id, repo, revision, artifact, files)) in parsed.iter().zip(RELEASES) {
            assert_eq!((entry.id.as_str(), entry.repo.as_str(), entry.revision.as_str(), entry.artifact.as_str()), (id, repo, revision, artifact));
            assert_eq!(entry.layer, CatalogLayer::BuiltIn, "{id}");
            let listed: Vec<(&str, u64, &str)> = entry.files.iter().map(|f| (f.name.as_str(), f.bytes, f.sha256.as_str())).collect();
            assert_eq!(listed, files, "{id}");
            // Listed small first, which is also the order a fetch takes.
            let by_size: Vec<&str> = entry.files_by_size().iter().map(|f| f.name.as_str()).collect();
            assert_eq!(by_size, files.iter().map(|f| f.0).collect::<Vec<_>>(), "{id}");
        }
    }

    /// The invariants the rest of the server leans on, stated once more over
    /// the built-in catalog on their own.
    #[test]
    fn the_built_in_catalog_serves_the_default_id_and_the_flash_next_companion() {
        let catalog = Catalog::built_in();
        assert_eq!(catalog.entry(crate::config::DEFAULT_MODEL).map(|e| e.artifact.as_str()), Some("qwen3_8_27b_nvfp4full-v2.ninfer"));
        assert_eq!(catalog.entry("QWEN3.8-27B").map(|e| e.id.as_str()), Some("qwen3.8-27b"), "case-insensitive");
        assert!(catalog.entry("qwen3.8-27b-instruct").is_none());
        // The loader finds Flash-Next's companion beside the main container
        // under its fixed name, so the entry must land it under that name.
        let flash = catalog.entry("qwen3.8-flash-next").expect("Flash-Next");
        assert!(flash.files.iter().any(|f| f.name == ignis_artifact::packer::MTP_ARTIFACT_FILE_NAME));
        assert_eq!(flash.artifact, ignis_artifact::packer::ARTIFACT_FILE_NAME);
        assert_eq!(format!("{:.1}", flash.gib()), "67.8", "72.8 GB, the four files together");
        for entry in catalog.entries() {
            for file in &entry.files {
                assert!(plain_name(&file.name), "{}", file.name);
            }
        }
    }

    /// AC 2: an operator catalog adds its entries after the built-in ones,
    /// read as YAML or as JSON.
    #[test]
    fn an_operator_catalog_adds_its_entries_in_either_format() {
        let from_yaml = operator(OPERATOR).expect("valid");
        let json = serde_json::to_string(&Format::Yaml.read(OPERATOR).unwrap()).unwrap();
        let from_json = parse_operator(&json, Format::Json, "acme.catalog.json").expect("valid");
        assert_eq!(from_yaml, from_json);
        assert_eq!(from_yaml.ids(), ["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next", "acme-qwen3.8-27b-ft"]);
        let acme = from_yaml.entry("acme-qwen3.8-27b-ft").unwrap();
        assert_eq!(acme.layer, CatalogLayer::Operator);
        assert_eq!(acme.revision, "v1");
        assert_eq!(acme.artifact_path(Path::new("models")), Path::new("models").join("acme_ft.ninfer"));
        assert_eq!(acme.total_bytes(), 48_211 + 19_406_942_468);
    }

    /// AC 2: every refusal *The catalog file* lists, each naming the entry
    /// or the key it is about — the built-in-id collision and the non-plain
    /// file name among them.
    #[test]
    fn each_refusal_names_the_offending_entry_or_key() {
        let entry = "acme-qwen3.8-27b-ft";
        for (from, to, says) in [
            ("    revision: v1\n", "    revision: v1\n    mirror: x\n", vec!["`acme-qwen3.8-27b-ft`", "`mirror`"]),
            ("models:\n", "endpoint: x\nmodels:\n", vec!["`endpoint`"]),
            ("bytes: 48211,", "bytes: 48211, size: 1,", vec!["`size`", "acme_ft.ninfer.graft.json"]),
            ("id: acme-qwen3.8-27b-ft", "id: \"\"", vec!["entry 1", "`id`"]),
            ("id: acme-qwen3.8-27b-ft", "id: QWEN3.8-27B", vec!["`QWEN3.8-27B`", "built-in id `qwen3.8-27b`"]),
            ("id: acme-qwen3.8-27b-ft", "id: qwen3.8-flash-next", vec!["`qwen3.8-flash-next`", "built-in"]),
            ("repo: acme/qwen-ft-ninfer", "repo: acme-qwen-ft-ninfer", vec![entry, "`repo`", "<owner>/<name>"]),
            ("repo: acme/qwen-ft-ninfer", "repo: acme/qwen/ft", vec![entry, "`repo`"]),
            ("repo: acme/qwen-ft-ninfer", "repo: /qwen", vec![entry, "`repo`"]),
            ("revision: v1", "revision: \"\"", vec![entry, "`revision`"]),
            ("artifact: acme_ft.ninfer", "artifact: acme_other.ninfer", vec![entry, "`acme_other.ninfer`", "not among"]),
            ("acme_ft.ninfer.graft.json", "acme_ft.ninfer.notes.json", vec![entry, "sidecar", "acme_ft.ninfer.graft.json"]),
            ("sha256: 9c1e0", "sha256: 9C1E0", vec![entry, "`sha256`", "lowercase"]),
            ("sha256: abb1000000000000000000000000000000000000000000000000000000000002", "sha256: abb1", vec![entry, "`sha256`", "64"]),
            ("bytes: 48211", "bytes: 0", vec![entry, "`bytes` is 0"]),
            ("bytes: 48211", "bytes: lots", vec![entry, "`bytes`"]),
        ] {
            assert!(OPERATOR.contains(from), "{from}");
            let text = OPERATOR.replacen(from, to, 1);
            let err = operator(&text).expect_err(&text).0;
            assert!(err.contains("acme.catalog.yaml"), "{err}");
            for said in says {
                assert!(err.contains(said), "{to:?}: `{said}` missing from: {err}");
            }
        }
    }

    #[test]
    fn a_duplicate_id_is_refused_case_insensitively_within_the_file() {
        let body = OPERATOR.strip_prefix("models:\n").unwrap();
        let twice = format!("{OPERATOR}{}", body.replacen("acme-qwen3.8-27b-ft", "ACME-Qwen3.8-27B-FT", 1));
        let err = operator(&twice).unwrap_err().0;
        assert!(err.contains("entry 2 (`ACME-Qwen3.8-27B-FT`)") && err.contains("`acme-qwen3.8-27b-ft`"), "{err}");
    }

    /// AC 2: a catalog must never write outside the destination directory —
    /// a separator of either kind, `..`, an absolute path or a drive.
    #[test]
    fn a_file_name_that_is_not_a_plain_name_is_refused() {
        for name in ["../escape.ninfer", "..", ".", "sub/acme_ft.ninfer", "sub\\acme_ft.ninfer", "/etc/acme.ninfer", "C:acme.ninfer", "C:\\models\\acme.ninfer"] {
            // Single-quoted: YAML reads no escape in it, so a `\` stays one.
            let text = OPERATOR.replacen("{ name: acme_ft.ninfer,", &format!("{{ name: '{name}',"), 1);
            assert_ne!(text, OPERATOR);
            let err = operator(&text).expect_err(name).0;
            assert!(err.contains("not a plain file name") && err.contains("acme-qwen3.8-27b-ft") && err.contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn a_catalog_that_is_not_a_list_of_entries_is_refused() {
        for (text, says) in [("[]", "`models` list"), ("{}", "`models`"), ("models: {}", "`models`"), ("models:\n  - just-a-name\n", "entry 1")] {
            let err = operator(text).expect_err(text).0;
            assert!(err.contains(says), "{text}: {err}");
        }
        assert!(operator("models: [").unwrap_err().0.contains("not valid"));
        assert_eq!(operator("models: []").unwrap(), *Catalog::built_in(), "an empty operator catalog adds nothing");
    }

    /// A `download.catalog` naming a file that is not there refuses, naming
    /// the path and the field.
    #[test]
    fn a_catalog_path_that_cannot_be_read_is_refused() {
        use crate::config::file::testing::MemFiles;
        let files = MemFiles::with(&[("acme.json", &serde_json::to_string(&Format::Yaml.read(OPERATOR).unwrap()).unwrap())]);
        assert_eq!(load(&files, None).unwrap(), *Catalog::built_in());
        assert!(load(&files, Some(Path::new("acme.json"))).unwrap().entry("acme-qwen3.8-27b-ft").is_some(), "read as JSON by extension");
        let err = load(&files, Some(Path::new("missing.yaml"))).unwrap_err().0;
        assert!(err.contains("missing.yaml") && err.contains("download.catalog"), "{err}");
    }
}
