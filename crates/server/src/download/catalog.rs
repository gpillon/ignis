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
//!
//! An entry may state the smallest **card class** it loads on (every built-in
//! one does), and may list **aliases**: its family's default id, or another
//! entry's served id of its family, which it answers to on a card below that
//! id's own entry (ADR 0064, ADR 0074, spec 27b-16gb/04 §6). It may state its
//! **model family**, and must where an alias involves it and neither its id
//! nor its aliases name one: the header stays the authority, checked where an
//! alias resolves. [`Catalog::answering`] is the one class-aware step a
//! requested id goes through — at start, for a switch, a request, a download
//! and `GET /v1/models` — so a model the card cannot load is refused before
//! anything is torn down or fetched.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use ignis_core::compute::{ModelFamily, ModelKind};
use serde_json::{Map, Value};

use crate::config::file::{Files, Format};
use crate::config::profile::CardClass;

/// The built-in catalog's text, as shipped.
const BUILT_IN_TEXT: &str = include_str!("catalog.yaml");

/// The built-in catalog, parsed once. A binary whose own catalog does not
/// parse is a build that must not ship: the test below parses it on every
/// `cargo test`.
pub static BUILT_IN_CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    let document = Format::Yaml.read(BUILT_IN_TEXT).expect("the built-in catalog is YAML");
    let entries = parse(&document, "built into this binary", CatalogLayer::BuiltIn).expect("the built-in catalog is valid");
    aliases_within_families(&entries).expect("the built-in catalog's aliases stay within their families");
    Catalog { entries }
});

/// The keys an entry takes; every one but `kind`, `card_class`, `family` and
/// `aliases` is required.
const ENTRY_KEYS: [&str; 9] = ["id", "repo", "revision", "artifact", "kind", "card_class", "family", "aliases", "files"];

/// The name an entry's audio encoder carries beside its decoder (`kind:
/// speech`): the speech adapter finds the encoder by this prefix.
pub const SPEECH_ENCODER_PREFIX: &str = "mmproj-";

/// Whether a file name is a speech entry's audio encoder: the `mmproj-` prefix
/// and a `.gguf` extension. The catalog and the adapter share this rule.
pub fn is_speech_encoder(name: &str) -> bool {
    name.starts_with(SPEECH_ENCODER_PREFIX) && name.ends_with(".gguf")
}

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

/// One published model (`GLOSSARY.md`, **Catalog entry**): its served id,
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
    /// What the entry is, when the catalog says it: `Some(ModelKind::Speech)`
    /// for a speech model (`kind: speech`; ADR 0067), never switched to,
    /// whose files are GGUFs with no sidecar; `Some(ModelKind::Diarizer)`
    /// for the diarizer (`kind: diarizer`; ADR 0069), one GGUF, also never
    /// switched to. `None` for the rest: a text or image model, told apart
    /// by its artifact's header.
    pub kind: Option<ModelKind>,
    /// The smallest card class it loads on (`card_class`, in GB; ADR 0064
    /// §1), when it states one. Optional in both layers, by one rule: an
    /// entry that states none is never refused by class, and the VRAM plan's
    /// refusal still guards it — so an operator catalog written before card
    /// classes still starts. Every built-in entry states one.
    pub card_class: Option<CardClass>,
    /// The model family it states (`family`, ADR 0074), by the family's own
    /// id, when it states one: required where an alias involves the entry
    /// and neither its id nor its aliases is a family's own id, and agreeing
    /// with them where they are. [`CatalogEntry::family()`] is the family the
    /// catalog tells, whichever says it.
    pub family: Option<ModelFamily>,
    /// The ids it also answers to (`aliases`, ADR 0064 §2): its family's
    /// default id, or another entry's served id of its family (ADR 0074), so
    /// that on a card class below that id's own entry a request naming it is
    /// served by this one. Usually none.
    pub aliases: Vec<String>,
    /// Every file, as the catalog lists them.
    pub files: Vec<CatalogFile>,
    /// Which catalog it came from.
    pub layer: CatalogLayer,
}

impl CatalogEntry {
    /// Whether the entry is a speech model (`kind: speech`).
    pub fn is_speech(&self) -> bool {
        self.kind == Some(ModelKind::Speech)
    }

    /// Whether the entry is a diarizer (`kind: diarizer`, spec speech/08).
    pub fn is_diarizer(&self) -> bool {
        self.kind == Some(ModelKind::Diarizer)
    }

    /// The model family the entry is of, as far as the catalog tells it
    /// (ADR 0074): the one it states, else the family whose own id it is, or
    /// whose own id it answers to as an alias — the catalog refuses an entry
    /// where these disagree. `None` for any other: no alias involves it, and
    /// the family stays the artifact's header's (ADR 0047).
    pub fn family(&self) -> Option<ModelFamily> {
        self.family
            .or_else(|| ModelFamily::of_model_id(&self.id))
            .or_else(|| self.aliases.iter().find_map(|alias| ModelFamily::of_model_id(alias)))
    }

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

    /// A catalog of exactly `entries`: for a server built in process, which
    /// reads no catalog file (the speech slot's tests, whose models are
    /// fixtures).
    pub fn from_entries(entries: Vec<CatalogEntry>) -> Catalog {
        Catalog { entries }
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

    /// The class rule (ADR 0064 §1, §3), the one place it is written: the
    /// entry `id` itself needs a larger card than one of class `card` — its
    /// refusal, naming both classes. `None` when it fits, states no class,
    /// no class is known (`None`: no total read, or one of no class — the
    /// VRAM plan's refusal still guards), or no entry has the id: an
    /// operator's own model is not the catalog's to judge.
    ///
    /// Ids are matched exactly, as a request's `model`, the switch's known
    /// models and the engine match them — not case-insensitively as
    /// [`Catalog::entry`] does for the `model` command line, whose callers
    /// take the entry's own id first.
    pub fn above_class(&self, id: &str, card: Option<CardClass>) -> Option<ClassRefusal> {
        let card = card?;
        let entry = self.entries.iter().find(|entry| entry.id == id)?;
        let needs = entry.card_class.filter(|needs| *needs > card)?;
        Some(ClassRefusal { model: entry.id.clone(), needs, card, other_family: None })
    }

    /// The class-aware step (ADR 0064 §2-3, spec 27b-16gb/04 §6) every
    /// adapter takes — the start, a switch, a request, a download, the model
    /// lists: which entry answers to `requested` on a card of class `card`
    /// in its place. `Ok(None)` when `requested` stands as it is — its own
    /// entry is not [`Catalog::above_class`]; `Ok(Some(entry))` when its own
    /// entry is, and `entry`, of its family, lists it as an alias and fits
    /// the card (or states no class); refused, naming both classes, when no
    /// entry that answers to it fits. Matched exactly, as
    /// [`Catalog::above_class`] matches. The family is the one the catalog
    /// tells of `requested`'s entry ([`CatalogEntry::family()`]): its id's,
    /// or the one it states (ADR 0074) — an entry of no family the catalog
    /// tells has none to answer to it.
    pub fn answering(&self, requested: &str, card: Option<CardClass>) -> Result<Option<&CatalogEntry>, ClassRefusal> {
        let Some(refusal) = self.above_class(requested, card) else {
            return Ok(None);
        };
        let family = self.family_of(requested);
        let answering = self.entries.iter().find(|entry| {
            entry.card_class.is_none_or(|needs| needs <= refusal.card)
                && family.is_some()
                && entry.family() == family
                && entry.aliases.iter().any(|alias| alias == requested)
        });
        answering.map(Some).ok_or(refusal)
    }

    /// The family the catalog tells of the entry whose id is exactly `id`
    /// ([`CatalogEntry::family()`]); `None` for none, or no such entry.
    fn family_of(&self, id: &str) -> Option<ModelFamily> {
        self.entries.iter().find(|entry| entry.id == id).and_then(CatalogEntry::family)
    }

    /// ADR 0064 §2 on the artifact itself, read once it is on disk: `entry`,
    /// which answers to `requested` on a card of class `card`
    /// ([`Catalog::answering`]), holds an artifact whose header names
    /// `header` (ADR 0047: the header is the authority, the catalog's family
    /// a claim it checks). An alias only ever resolves within one model
    /// family, so that must be `requested`'s family as the catalog tells it
    /// — its id's, or the one its entry states (ADR 0074); else the alias
    /// does not resolve, refused naming both families — a header that names
    /// no family is refused too, the safe side. `Ok` when no alias is in
    /// play: `requested`'s own entry is not above the card's class.
    pub fn alias_holds(&self, requested: &str, entry: &CatalogEntry, header: Option<ModelFamily>, card: Option<CardClass>) -> Result<(), ClassRefusal> {
        let Some(mut refusal) = self.above_class(requested, card) else {
            return Ok(());
        };
        let alias = self.family_of(requested);
        if header.is_some() && header == alias {
            return Ok(());
        }
        refusal.other_family = Some(OtherFamily { entry: entry.id.clone(), artifact: header, alias });
        Err(refusal)
    }
}

/// A catalog entry the card cannot load (ADR 0064 §3): refused before
/// anything is torn down or fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassRefusal {
    /// The entry's id: the one requested.
    pub model: String,
    /// The entry's `card_class`.
    pub needs: CardClass,
    /// The card's class.
    pub card: CardClass,
    /// The entry that lists `model` as an alias and fits this card, when one
    /// does but its artifact is of another family than `model`'s
    /// ([`Catalog::alias_holds`]): it does not answer to the alias.
    pub other_family: Option<OtherFamily>,
}

/// An entry that would answer to an alias but holds an artifact of another
/// family than the alias's (ADR 0064 §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtherFamily {
    /// The entry's id.
    pub entry: String,
    /// The family its artifact's header names; `None` for none.
    pub artifact: Option<ModelFamily>,
    /// The alias's family, as the catalog tells it (ADR 0074): the family
    /// whose own id the alias is, or the one its entry states.
    pub alias: Option<ModelFamily>,
}

impl std::fmt::Display for ClassRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} needs a {} card or larger (its catalog entry's card_class), and this card is of the {} class", self.model, self.needs, self.card)?;
        if let Some(other) = &self.other_family {
            let alias = other.alias.map_or("no model family", ModelFamily::name);
            let artifact = other.artifact.map_or("no model family", ModelFamily::name);
            write!(
                f,
                "; {} lists it as an alias, but its artifact is {artifact} and {} is {alias}: an alias resolves only within one model family",
                other.entry, self.model
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ClassRefusal {}

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
    parse_operator(&text, Format::of_file(path), &path.display().to_string())
}

/// An operator catalog's text, read in `format` and put after the built-in
/// entries. `origin` (the file's path) is named in every refusal.
///
/// An operator entry may answer to a built-in entry's id as an alias (ADR
/// 0064 §2: an operator's own 16 GB build of the 27B answering to
/// `qwen3.8-27b`). That is not the silent re-pin ADR 0047 refuses: the
/// built-in entry keeps its id and its pins and serves it wherever it fits,
/// the alias is consulted only on a card below its class, and the served id
/// a response reports is the operator entry's, so a client can tell which
/// model answered. An alias a built-in entry already claims is refused, as
/// within one file.
pub fn parse_operator(text: &str, format: Format, origin: &str) -> Result<Catalog, CatalogError> {
    parse_operator_over(Catalog::built_in(), text, format, origin)
}

/// [`parse_operator`] over `built_in` in the built-in catalog's place: the
/// seam a test puts a built-in entry with an alias behind, which the shipped
/// catalog does not hold yet.
fn parse_operator_over(built_in: &Catalog, text: &str, format: Format, origin: &str) -> Result<Catalog, CatalogError> {
    let document = format
        .read(text)
        .map_err(|err| CatalogError(format!("the catalog {origin} is not valid: {err}")))?;
    let operator = parse(&document, origin, CatalogLayer::Operator)?;
    let mut entries = built_in.entries.clone();
    for entry in operator {
        if let Some(official) = built_in.entry(&entry.id) {
            return Err(CatalogError(format!(
                "the catalog {origin}: entry `{}` reuses the built-in id `{}` — an operator entry never re-pins an official model (a mirror of it is `download.endpoint`, a variant of it a new id)",
                entry.id, official.id
            )));
        }
        entries.push(entry);
    }
    claimed_once(&entries).map_err(|reason| CatalogError(format!("the catalog {origin}: {reason}")))?;
    aliases_within_families(&entries).map_err(|reason| CatalogError(format!("the catalog {origin}: {reason}")))?;
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
    claimed_once(&entries).map_err(refuse)?;
    Ok(entries)
}

/// Every alias claimed by one entry at most (ADR 0064 §2): an id answers for
/// one entry, or which one served it would be the catalog's order. Checked
/// over one file's entries, and again over the merged catalog once an
/// operator's join the built-in ones. That an alias is another entry's id
/// is the point of it, not a collision.
fn claimed_once(entries: &[CatalogEntry]) -> Result<(), String> {
    for (index, entry) in entries.iter().enumerate() {
        for alias in &entry.aliases {
            if let Some(earlier) = entries[..index].iter().find(|earlier| earlier.aliases.contains(alias)) {
                return Err(format!(
                    "entry `{}`: alias `{alias}` is already `{}`'s — an id answers for one entry",
                    entry.id, earlier.id
                ));
            }
        }
    }
    Ok(())
}

/// Every alias within its entry's model family (ADR 0064 §2, ADR 0074),
/// over a whole catalog — the built-in one, and the merged one once an
/// operator's entries join it, since an operator's alias may be a built-in
/// entry's id. An alias that is an entry's served id, matched exactly as a
/// request's `model` is, names an entry of a family the catalog tells, and
/// of the aliasing entry's own: so an entry another entry's alias names
/// states `family` when its id and aliases name none. Any other alias is a
/// family's own id, which [`aliases`] held to the entry's family already.
fn aliases_within_families(entries: &[CatalogEntry]) -> Result<(), String> {
    for entry in entries {
        for alias in &entry.aliases {
            let Some(named) = entries.iter().find(|named| named.id == *alias) else {
                if ModelFamily::of_model_id(alias).is_some() {
                    continue;
                }
                return Err(format!(
                    "entry `{}`: alias `{alias}` is no model family's own id ({}) and no entry's served id (ids match exactly): an alias answers for an id on a smaller card",
                    entry.id,
                    family_ids()
                ));
            };
            match (named.family(), entry.family()) {
                (None, _) => {
                    return Err(format!(
                        "entry `{}`: alias `{alias}` names an entry whose family nothing states — its id and aliases name none, so `{}` must state `family` ({}): an alias resolves only within one model family",
                        entry.id,
                        named.id,
                        family_ids()
                    ))
                }
                (Some(theirs), Some(own)) if theirs != own => {
                    return Err(format!(
                        "entry `{}`: alias `{alias}` names an entry of {} and the entry is of {}: an alias resolves only within one model family",
                        entry.id,
                        theirs.name(),
                        own.name()
                    ))
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Every model family's own id, as a refusal lists them.
fn family_ids() -> String {
    ModelFamily::ALL.iter().map(|family| family.model_id()).collect::<Vec<_>>().join(", ")
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
    let kind = kind(map)?;
    let card_class = card_class(map)?;
    let family = family(map, &id)?;
    let aliases = aliases(map, &id, family)?;
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
    if kind == Some(ModelKind::Speech) {
        // A speech model is two GGUFs and no sidecar: the decoder is the
        // artifact, the encoder the `mmproj-` file beside it.
        if !artifact.ends_with(".gguf") {
            return Err(format!("a speech model's `artifact` is its decoder GGUF, got `{artifact}`"));
        }
        if files.iter().filter(|file| file.name != artifact && is_speech_encoder(&file.name)).count() != 1 {
            return Err(format!(
                "a speech model needs its audio encoder among its `files`, exactly one `{SPEECH_ENCODER_PREFIX}*.gguf` file beside `{artifact}`"
            ));
        }
    } else if kind == Some(ModelKind::Diarizer) {
        // A diarizer is one GGUF: no sidecar, no encoder beside it.
        if !artifact.ends_with(".gguf") {
            return Err(format!("a diarizer's `artifact` is its GGUF, got `{artifact}`"));
        }
        if files.len() != 1 {
            return Err(format!("a diarizer is one GGUF: its `files` list `{artifact}` alone, with no sidecar"));
        }
    } else {
        let sidecars: Vec<String> = crate::loader::SIDECAR_SUFFIXES.iter().map(|suffix| format!("{artifact}{suffix}")).collect();
        if !files.iter().any(|file| sidecars.contains(&file.name)) {
            return Err(format!(
                "no file is the artifact's sidecar ({}): the loader refuses a load without one",
                sidecars.join(" or ")
            ));
        }
    }
    Ok(CatalogEntry { id, repo, revision, artifact, kind, card_class, family, aliases, files, layer })
}

/// `kind`, when stated: `speech` or `diarizer`, the kinds a catalog says. A
/// text or an image model is told apart by its artifact's header (ADR 0047),
/// so stating either here would be a second word that could disagree with
/// the file.
fn kind(map: &Map<String, Value>) -> Result<Option<ModelKind>, String> {
    match map.get("kind") {
        None => Ok(None),
        Some(Value::String(kind)) if kind.trim() == "speech" => Ok(Some(ModelKind::Speech)),
        Some(Value::String(kind)) if kind.trim() == "diarizer" => Ok(Some(ModelKind::Diarizer)),
        Some(other) => Err(format!("`kind` must be `speech` or `diarizer` when stated (a text or image model's kind is read from its artifact), got {other}")),
    }
}

/// `card_class`, when stated: the GB of the smallest card class the entry
/// loads on, one of [`CardClass::ALL`]'s — `CardClass::from_gb` reads it, so
/// any other number refuses the catalog. Left out, the entry states none.
fn card_class(map: &Map<String, Value>) -> Result<Option<CardClass>, String> {
    let classes = CardClass::ALL.map(|class| class.gb().to_string()).join(", ");
    let refuse = |got: &dyn std::fmt::Display| format!("`card_class` must be {classes} (the GB of the smallest card it loads on), got {got}");
    match map.get("card_class") {
        Some(Value::Number(number)) => {
            number.as_u64().and_then(|gb| u32::try_from(gb).ok()).and_then(CardClass::from_gb).map(Some).ok_or_else(|| refuse(number))
        }
        Some(other) => Err(refuse(other)),
        None => Ok(None),
    }
}

/// `family`, when stated: a model family's own id ([`ModelFamily::ALL`]),
/// matched exactly, and the family the entry's own id is, when it is one's
/// (ADR 0074). Its aliases are held to it by [`aliases`].
fn family(map: &Map<String, Value>, id: &str) -> Result<Option<ModelFamily>, String> {
    let Some(value) = map.get("family") else {
        return Ok(None);
    };
    let Some(stated) = value.as_str().and_then(|stated| ModelFamily::of_model_id(stated.trim())) else {
        return Err(format!("`family` must be a model family's own id ({}), got {value}", family_ids()));
    };
    if let Some(own) = ModelFamily::of_model_id(id).filter(|own| *own != stated) {
        return Err(format!(
            "`family` is {} and the entry's own id `{id}` is {}'s: a stated family agrees with what the id says",
            stated.name(),
            own.name()
        ));
    }
    Ok(Some(stated))
}

/// `aliases`, when listed: none the entry's own id, none twice. One that is
/// a model family's own id is of the family the entry's own id names, when
/// it names one, of the family it states (`stated`), and of the family its
/// other such aliases are. One that is not names another entry's served id,
/// held to the entry's family over the whole catalog
/// ([`aliases_within_families`]) — so an entry listing one, whose id and
/// aliases name no family, states `family` (ADR 0074). That is what keeps an
/// alias within one family.
fn aliases(map: &Map<String, Value>, id: &str, stated: Option<ModelFamily>) -> Result<Vec<String>, String> {
    let listed = match map.get("aliases") {
        None => return Ok(Vec::new()),
        Some(Value::Array(listed)) => listed,
        Some(other) => return Err(format!("`aliases` must be a list of served ids, got {other}")),
    };
    let mut aliases: Vec<String> = Vec::new();
    for value in listed {
        let Some(alias) = value.as_str().map(str::trim).filter(|alias| !alias.is_empty()) else {
            return Err(format!("`aliases` must be a list of served ids, got {value}"));
        };
        if alias.eq_ignore_ascii_case(id) {
            return Err(format!("`aliases` lists the entry's own id `{alias}`"));
        }
        if aliases.iter().any(|earlier| earlier == alias) {
            return Err(format!("alias `{alias}` is listed twice"));
        }
        if let Some(family) = ModelFamily::of_model_id(alias) {
            if let Some(own) = ModelFamily::of_model_id(id).filter(|own| *own != family) {
                return Err(format!(
                    "alias `{alias}` is {}'s own id and the entry's own id `{id}` is {}'s: an alias resolves only within one model family",
                    family.name(),
                    own.name()
                ));
            }
            if let Some(stated) = stated.filter(|stated| *stated != family) {
                return Err(format!(
                    "alias `{alias}` is {}'s own id and the entry states `family` {}: an alias resolves only within one model family",
                    family.name(),
                    stated.name()
                ));
            }
            if let Some(first) = aliases.iter().find(|earlier| ModelFamily::of_model_id(earlier).is_some_and(|earlier| earlier != family)) {
                return Err(format!("aliases `{first}` and `{alias}` name two model families: an alias resolves only within one"));
            }
        }
        aliases.push(alias.to_owned());
    }
    let named = ModelFamily::of_model_id(id).is_some() || aliases.iter().any(|alias| ModelFamily::of_model_id(alias).is_some());
    if !aliases.is_empty() && stated.is_none() && !named {
        return Err(format!(
            "it lists aliases, and neither its id nor its aliases is a model family's own id, so it must state `family` ({}), the family it is of (ADR 0074)",
            family_ids()
        ));
    }
    Ok(aliases)
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
            "`{name}` is not a plain file name (letters, digits, `.`, `_` and `-`, not starting or ending with `.`, not a Windows device name): a catalog never writes outside the destination directory"
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

/// A name that stays inside the directory it is joined to on every host. A
/// security predicate lists the safe side, not the unsafe one: only letters,
/// digits, `.`, `_` and `-` (so no separator of either kind, no drive, no
/// control character, no space), neither starting nor ending with `.`
/// (`.`, `..`, and the trailing dot Windows silently drops), and not a
/// Windows device name (`CON`, `NUL`, `COM1`...), which Windows opens as the
/// device whatever follows its first `.`.
fn plain_name(name: &str) -> bool {
    const DEVICES: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    let stem = name.split('.').next().unwrap_or(name);
    !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        && !name.starts_with('.')
        && !name.ends_with('.')
        && !DEVICES.iter().any(|device| stem.eq_ignore_ascii_case(device))
}

/// Test fixtures other modules' tests share.
#[cfg(test)]
pub(crate) mod testing {
    /// The 16 GB 27B's entry as an operator catalog would state it: its own
    /// served id, the 16 GB class, and the 27B's default id as its alias.
    /// The built-in catalog holds the real one (GitHub #338), so this one is
    /// parsed over [`thirty_two`]: the input of the parsing tests.
    pub(crate) const SIXTEEN: &str = include_str!("../../tests/fixtures/catalog/sixteen.yaml");

    /// The built-in catalog, which holds the 16 GB 27B (GitHub #338) and the
    /// uncensored variant's 16 GB build (GitHub #423).
    pub(crate) fn built_in() -> super::Catalog {
        super::Catalog::built_in().clone()
    }

    /// The built-in catalog without its 16 GB entries: a catalog in which no
    /// entry answers to the 27B's default id, nor to the uncensored
    /// variant's, on a 16 GB card.
    pub(crate) fn thirty_two() -> super::Catalog {
        let entries = super::Catalog::built_in().entries.iter().filter(|entry| entry.card_class != Some(super::CardClass::Gb16));
        super::Catalog { entries: entries.cloned().collect() }
    }

    /// [`SIXTEEN`] as an operator's catalog over [`thirty_two`].
    pub(crate) fn sixteen_over_thirty_two(text: &str) -> Result<super::Catalog, super::CatalogError> {
        super::parse_operator_over(&thirty_two(), text, crate::config::file::Format::Yaml, "acme.catalog.yaml")
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{sixteen_over_thirty_two, thirty_two, SIXTEEN};
    use super::*;

    /// The spec's table, written out again: what the built-in catalog must
    /// hold, value for value (`id`, `repo`, `revision`, `artifact`, its
    /// `card_class` in GB, its model family, its aliases, then each file's
    /// name, bytes and SHA-256 in the order listed). Every built-in entry
    /// states its card class (ADR 0064 §1: an operator's may leave it out, a
    /// built-in one never does), the 32 GB one for each but the two 16 GB
    /// entries, the only ones with an alias: the 16 GB 27B's (GitHub #338,
    /// spec 27b-16gb/01) `qwen3.8-27b`, and its uncensored variant's build's
    /// (GitHub #423, ADR 0074) `qwen3.8-27b-abliterated`, the 32 GB entry's
    /// served id. Every one is of a family the catalog tells (ADR 0074): its
    /// own id's, its alias's, or the one it states — the abliterated 27B's
    /// and its 16 GB build's, whose ids name none.
    #[allow(clippy::type_complexity)]
    const RELEASES: [(&str, &str, &str, &str, u32, ModelFamily, &[&str], &[(&str, u64, &str)]); 6] = [
        (
            "qwen3.8-27b",
            "gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer",
            "e961b419b672e183aa55df8c4b975abc82006e8a",
            "qwen3_8_27b_nvfp4full-v2.ninfer",
            32,
            ModelFamily::Qwen38_27b,
            &[],
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
            32,
            ModelFamily::Qwen38_27b,
            &[],
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
            32,
            ModelFamily::FlashNext,
            &[],
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
        // The image model: the upload's hub commit, and the published sidecar
        // (converter.command paths replaced by placeholders).
        (
            "ignis-image-2.1-turbo",
            "gpillon/ignis-image-2.1-turbo",
            "d6912417aa47320cd71c78e19d99872faa5cb63b",
            "ignis_image_2_1_turbo-v2.ninfer",
            32,
            ModelFamily::QwenImage21,
            &[],
            &[
                ("ignis_image_2_1_turbo-v2.ninfer.conversion.json", 617_924, "a97f04912b9a6d76e828efc02b7e458d0f45d00bc00d794cca39d703ad0f33ca"),
                ("ignis_image_2_1_turbo-v2.ninfer", 16_721_644_544, "4205dd1932f0801804b0a544fe1340697e4e0a0dea4a276996bcb981006050af"),
            ],
        ),
        // The 16 GB 27B: the upload's hub commit (2026-10-23).
        (
            "qwen3.8-27b-16g",
            "gpillon/Qwen3.8-27B-trellis-3bpw-ignis",
            "f5bbd2cc14c9ca573ef5553c100ac69410f1fe58",
            "qwen3_8_27b_trellis_sc30-v2.ninfer",
            16,
            ModelFamily::Qwen38_27b,
            &["qwen3.8-27b"],
            &[
                ("qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json", 309_495, "210ab54bfe8b04e5fc4337ee3755c340ea3f21239aa625c505aa2406645a340d"),
                ("qwen3_8_27b_trellis_sc30-v2.ninfer", 13_379_877_124, "2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c"),
            ],
        ),
        // The 16 GB build of the uncensored variant: the upload's hub commit
        // (2026-10-30, GitHub #423).
        (
            "qwen3.8-27b-abliterated-16g",
            "gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis",
            "94a8bfc0f47ce86403f274be03e2c0f79500727c",
            "qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer",
            16,
            ModelFamily::Qwen38_27b,
            &["qwen3.8-27b-abliterated"],
            &[
                (
                    "qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer.conversion.json",
                    339_543,
                    "b5b61863b218bf722e4786ec8ca5e7ffde03548068f92184f10ecd95f6fd0f61",
                ),
                (
                    "qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer",
                    13_379_877_124,
                    "4929bc8e1fbc747092f96aebae6b81a0b619db452d05a56df3499fbe1afe9ac5",
                ),
            ],
        ),
    ];

    /// The speech releases (ADR 0067, spec speech/01 AC 22): `(id, repo,
    /// revision, decoder, [(file, bytes, sha256)])`, written out again from
    /// the repos' `SHA256SUMS`. A speech entry states no card class: it runs
    /// on the CPU, so no card refuses it.
    #[allow(clippy::type_complexity)]
    const SPEECH_RELEASES: [(&str, &str, &str, &str, &[(&str, u64, &str)]); 3] = [
        (
            "qwen3-asr-1.7b",
            "gpillon/Qwen3-ASR-1.7B-GGUF-ignis",
            "f8a5263e398c1b3606abedc6b9c74dcbced9f93f",
            "Qwen3-ASR-1.7B-Q4_K_M.gguf",
            &[
                ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", 355_709_472, "f325a44bdbc23b2df78ed35ad6342da3aabf1e618e1a9ffd1ef8861cb3b1b0f6"),
                ("Qwen3-ASR-1.7B-Q4_K_M.gguf", 1_282_435_136, "5c5f878af090a70b25dc7a0f1090620e734654e9eceb6f4ac4656423c243974c"),
            ],
        ),
        (
            "qwen3-asr-1.7b-q8",
            "gpillon/Qwen3-ASR-1.7B-GGUF-ignis",
            "f8a5263e398c1b3606abedc6b9c74dcbced9f93f",
            "Qwen3-ASR-1.7B-Q8_0.gguf",
            &[
                ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", 355_709_472, "f325a44bdbc23b2df78ed35ad6342da3aabf1e618e1a9ffd1ef8861cb3b1b0f6"),
                ("Qwen3-ASR-1.7B-Q8_0.gguf", 2_165_035_072, "9039c0cbbc13be5c25f2b34fcb70258f68f06b64f67a48831c7e13c72593e1e7"),
            ],
        ),
        (
            "qwen3-asr-0.6b",
            "gpillon/Qwen3-ASR-0.6B-GGUF-ignis",
            "af02916ea7a532a667a907c2a163b16a8c04b433",
            "Qwen3-ASR-0.6B-Q8_0.gguf",
            &[
                ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", 214_392_608, "ee5e87436bcf8b1c550edd4300fd0516067d55b547611734d27fbc55ecaec3bc"),
                ("Qwen3-ASR-0.6B-Q8_0.gguf", 804_749_376, "027ad3329549cd2c14e4ebd7d104e0390566e06570270c33062529474ec56015"),
            ],
        ),
    ];

    /// A minimal valid speech entry, YAML, for the refusal table.
    const SPEECH_OPERATOR: &str = "models:
  - id: acme-asr
    repo: acme/asr
    revision: v1
    artifact: acme-decoder.gguf
    kind: speech
    files:
      - { name: mmproj-acme.gguf, bytes: 10, sha256: 9c1e000000000000000000000000000000000000000000000000000000000001 }
      - { name: acme-decoder.gguf, bytes: 20, sha256: abb1000000000000000000000000000000000000000000000000000000000002 }
";

    /// A minimal valid operator entry, YAML, for the refusal table to break
    /// one thing at a time.
    const OPERATOR: &str = "models:
  - id: acme-qwen3.8-27b-ft
    repo: acme/qwen-ft-ninfer
    revision: v1
    artifact: acme_ft.ninfer
    card_class: 32
    files:
      - { name: acme_ft.ninfer.graft.json, bytes: 48211, sha256: 9c1e000000000000000000000000000000000000000000000000000000000001 }
      - { name: acme_ft.ninfer, bytes: 19406942468, sha256: abb1000000000000000000000000000000000000000000000000000000000002 }
";

    fn operator(text: &str) -> Result<Catalog, CatalogError> {
        parse_operator(text, Format::Yaml, "acme.catalog.yaml")
    }

    /// AC 1: the built-in catalog is the data file, it parses under every
    /// rule an operator's does, and it holds exactly the spec's releases
    /// with those values.
    #[test]
    fn the_built_in_catalog_holds_exactly_the_releases_with_their_pins() {
        let document = Format::Yaml.read(BUILT_IN_TEXT).expect("YAML");
        let parsed = parse(&document, "built into this binary", CatalogLayer::BuiltIn).expect("every invariant holds");
        assert_eq!(parsed, BUILT_IN_CATALOG.entries);
        assert_eq!(parsed.len(), RELEASES.len() + SPEECH_RELEASES.len() + 1, "and the diarizer");
        for (entry, (id, repo, revision, artifact, card_class, family, aliases, files)) in parsed.iter().zip(RELEASES) {
            assert_eq!(entry.kind, None, "{id}: a text or image model states no kind");
            assert_eq!((entry.id.as_str(), entry.repo.as_str(), entry.revision.as_str(), entry.artifact.as_str()), (id, repo, revision, artifact));
            assert_eq!(entry.card_class.map(CardClass::gb), Some(card_class), "{id}: every built-in entry states its class");
            assert_eq!(entry.family(), Some(family), "{id}: every built-in entry is of a family the catalog tells");
            assert_eq!(entry.aliases, aliases, "{id}");
            assert_eq!(entry.layer, CatalogLayer::BuiltIn, "{id}");
            let listed: Vec<(&str, u64, &str)> = entry.files.iter().map(|f| (f.name.as_str(), f.bytes, f.sha256.as_str())).collect();
            assert_eq!(listed, files, "{id}");
            // Listed small first, which is also the order a fetch takes.
            let by_size: Vec<&str> = entry.files_by_size().iter().map(|f| f.name.as_str()).collect();
            assert_eq!(by_size, files.iter().map(|f| f.0).collect::<Vec<_>>(), "{id}");
        }
    }

    /// Spec speech/01 AC 22: the three speech entries, pin for pin, each a
    /// `kind: speech` entry of GGUFs with no sidecar and no card class, the
    /// decoder its artifact and the encoder the `mmproj-` file beside it.
    #[test]
    fn the_built_in_catalog_holds_the_three_speech_entries_with_their_pins() {
        let catalog = Catalog::built_in();
        for (id, repo, revision, decoder, files) in SPEECH_RELEASES {
            let entry = catalog.entry(id).unwrap_or_else(|| panic!("{id} is built in"));
            assert_eq!((entry.repo.as_str(), entry.revision.as_str(), entry.artifact.as_str()), (repo, revision, decoder), "{id}");
            assert!(entry.is_speech(), "{id}");
            assert_eq!((entry.card_class, entry.aliases.as_slice(), entry.layer), (None, &[][..], CatalogLayer::BuiltIn), "{id}");
            assert_eq!(entry.family(), None, "{id}: of no family the catalog tells: its id names none, and it states none");
            let listed: Vec<(&str, u64, &str)> = entry.files.iter().map(|f| (f.name.as_str(), f.bytes, f.sha256.as_str())).collect();
            assert_eq!(listed, files, "{id}");
            assert_eq!(entry.files.iter().filter(|f| f.name.starts_with(SPEECH_ENCODER_PREFIX)).count(), 1, "{id}: one encoder");
        }
        assert_eq!(crate::config::AUTO_SPEECH_MODEL_ID, "qwen3-asr-1.7b");
        assert_eq!(catalog.entries().iter().filter(|entry| entry.is_speech()).count(), SPEECH_RELEASES.len());
        assert!(!catalog.entry("qwen3.8-27b").unwrap().is_speech());
        // The two 1.7B entries share one encoder file, byte for byte, so a
        // download of both lands it once.
        let (a, b) = (catalog.entry("qwen3-asr-1.7b").unwrap(), catalog.entry("qwen3-asr-1.7b-q8").unwrap());
        assert_eq!(a.files[0], b.files[0]);
    }

    /// `kind: speech` takes the speech rules in place of the sidecar rule,
    /// and no other kind is stated in a catalog.
    #[test]
    fn a_speech_entry_needs_its_encoder_and_no_sidecar_and_no_other_kind_is_stated() {
        let catalog = operator(SPEECH_OPERATOR).expect("a speech entry has no sidecar");
        let acme = catalog.entry("acme-asr").unwrap();
        assert_eq!(acme.kind, Some(ModelKind::Speech));
        assert!(acme.is_speech());
        for (from, to, says) in [
            ("    kind: speech\n", "    kind: text\n", vec!["acme-asr", "`kind`", "`speech`"]),
            ("    kind: speech\n", "    kind: 3\n", vec!["acme-asr", "`kind`"]),
            ("mmproj-acme.gguf", "encoder-acme.gguf", vec!["acme-asr", "audio encoder", "mmproj-"]),
            ("mmproj-acme.gguf", "mmproj-acme.bin", vec!["acme-asr", "audio encoder"]),
            ("artifact: acme-decoder.gguf", "artifact: mmproj-acme.gguf", vec!["acme-asr", "audio encoder"]),
            ("acme-decoder.gguf", "acme-decoder.ninfer", vec!["acme-asr", "decoder GGUF"]),
            ("      - { name: acme-decoder.gguf,", "      - { name: mmproj-second.gguf, bytes: 5, sha256: 9c1e000000000000000000000000000000000000000000000000000000000003 }
      - { name: acme-decoder.gguf,", vec!["acme-asr", "exactly one"]),
        ] {
            let text = SPEECH_OPERATOR.replace(from, to);
            let error = operator(&text).expect_err(to).to_string();
            for want in says {
                assert!(error.contains(want), "{to}: {error} lacks {want}");
            }
        }
        // A text entry without a kind still needs its sidecar.
        let text = OPERATOR.replace("acme_ft.ninfer.graft.json", "acme_ft.ninfer.notes.json");
        assert!(operator(&text).expect_err("sidecar").to_string().contains("sidecar"));
    }
    /// A minimal valid diarizer entry, YAML.
    const DIARIZER_OPERATOR: &str = "models:
  - id: acme-diarizer
    repo: acme/diarizer
    revision: v1
    artifact: acme-diarizer.gguf
    kind: diarizer
    files:
      - { name: acme-diarizer.gguf, bytes: 20, sha256: abb1000000000000000000000000000000000000000000000000000000000002 }
";

    /// Spec speech/08 (Q118): `kind: diarizer` is one GGUF with no sidecar
    /// and no card class; a second file, a sidecar or another artifact is
    /// refused naming the entry.
    #[test]
    fn a_diarizer_entry_is_one_gguf_and_no_sidecar() {
        let catalog = operator(DIARIZER_OPERATOR).expect("a diarizer entry has no sidecar");
        let acme = catalog.entry("acme-diarizer").unwrap();
        assert_eq!(acme.kind, Some(ModelKind::Diarizer));
        assert!(acme.is_diarizer() && !acme.is_speech());
        assert_eq!(acme.card_class, None);
        assert!(!catalog.entry("acme-diarizer").unwrap().is_speech());
        for (from, to, says) in [
            ("acme-diarizer.gguf", "acme-diarizer.bin", vec!["acme-diarizer", "diarizer's `artifact`"]),
            (
                "      - { name: acme-diarizer.gguf,",
                "      - { name: acme-diarizer.gguf.graft.json, bytes: 5, sha256: 9c1e000000000000000000000000000000000000000000000000000000000003 }
      - { name: acme-diarizer.gguf,",
                vec!["acme-diarizer", "one GGUF"],
            ),
        ] {
            let text = DIARIZER_OPERATOR.replace(from, to);
            let error = operator(&text).expect_err(to).to_string();
            for want in says {
                assert!(error.contains(want), "{to}: {error} lacks {want}");
            }
        }
        let error = operator(&DIARIZER_OPERATOR.replace("kind: diarizer", "kind: image")).expect_err("image").to_string();
        assert!(error.contains("`speech` or `diarizer`"), "{error}");
        // The built-in diarizer, pin for pin: the F16 GGUF (spec speech/08, AC 2) on its own repo.
        let built_in: Vec<&CatalogEntry> = Catalog::built_in().entries().iter().filter(|entry| entry.is_diarizer()).collect();
        let [entry] = built_in.as_slice() else { panic!("one built-in diarizer: {built_in:?}") };
        assert_eq!(
            (entry.id.as_str(), entry.repo.as_str(), entry.revision.as_str(), entry.artifact.as_str(), entry.card_class),
            ("nemotron-3-diarization", "gpillon/Nemotron-3-Diarization-GGUF-ignis", "d46d59d03be709db18e223b6bf92e8c32d14f422", "Nemotron-3-Diarization-F16.gguf", None)
        );
        assert_eq!(entry.family(), Some(ModelFamily::Nemotron3Diarization), "its id is the family's own");
        let listed: Vec<(&str, u64, &str)> = entry.files.iter().map(|f| (f.name.as_str(), f.bytes, f.sha256.as_str())).collect();
        assert_eq!(listed, [("Nemotron-3-Diarization-F16.gguf", 198_805_760, "f7a80981d178be5d683aee225166bc03806aac2d9b629f42f418d4e5fd2c8623")]);
    }

    /// The invariants the rest of the server leans on, stated once more over
    /// the built-in catalog on their own.
    #[test]
    fn the_built_in_catalog_serves_the_default_id_and_the_flash_next_companion() {
        let catalog = Catalog::built_in();
        assert_eq!(catalog.entry(crate::config::DEFAULT_MODEL).map(|e| e.artifact.as_str()), Some("qwen3_8_27b_nvfp4full-v2.ninfer"));
        assert_eq!(catalog.entry("QWEN3.8-27B").map(|e| e.id.as_str()), Some("qwen3.8-27b"), "case-insensitive");
        assert!(catalog.entry("qwen3.8-27b-instruct").is_none());
        // `model download ignis-image-2.1-turbo` (spec image-model/05): the image
        // model resolves like the others, its container beside the sidecar the
        // loader verifies it against.
        let image = catalog.entry("Ignis-Image-2.1-Turbo").expect("the image model");
        assert_eq!(image.id, "ignis-image-2.1-turbo");
        assert_eq!(image.artifact, "ignis_image_2_1_turbo-v2.ninfer");
        assert_eq!(image.files.len(), 2);
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
        assert_eq!(from_yaml.ids(), ["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next", "ignis-image-2.1-turbo", "qwen3.8-27b-16g", "qwen3.8-27b-abliterated-16g", "qwen3-asr-1.7b", "qwen3-asr-1.7b-q8", "qwen3-asr-0.6b", "nemotron-3-diarization", "acme-qwen3.8-27b-ft"]);
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
            // ADR 0064 §1: a card class stated is one of the classes there
            // are.
            ("card_class: 32", "card_class: 24", vec![entry, "`card_class`", "16, 32, 96", "got 24"]),
            ("card_class: 32", "card_class: \"32\"", vec![entry, "`card_class`", "16, 32, 96"]),
            ("card_class: 32", "card_class: -16", vec![entry, "`card_class`"]),
            // ADR 0064 §2: an alias is of one family, never the entry's own
            // id.
            ("card_class: 32", "card_class: 32\n    aliases: qwen3.8-27b", vec![entry, "`aliases` must be a list"]),
            ("card_class: 32", "card_class: 32\n    aliases: [acme-qwen3.8-27b-ft]", vec![entry, "own id"]),
            ("card_class: 32", "card_class: 32\n    aliases: [qwen3.8-27b, qwen3.8-flash-next]", vec![entry, "`qwen3.8-27b`", "`qwen3.8-flash-next`", "two model families"]),
            ("card_class: 32", "card_class: 32\n    aliases: [qwen3.8-27b, qwen3.8-27b]", vec![entry, "`qwen3.8-27b`", "twice"]),
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

    /// ADR 0074: `family` is a model family's own id and agrees with what
    /// the entry's id and aliases say; an alias is a family's own id or
    /// another entry's served id, of the entry's own family, and an entry an
    /// alias involves is of a family the catalog tells. Each refusal names
    /// the entry.
    #[test]
    fn a_family_or_an_alias_that_does_not_hold_is_refused_naming_the_entry() {
        let entry = "`acme-qwen3.8-27b-ft`";
        for (to, says) in [
            // A `family` that is no family's own id.
            ("card_class: 32\n    family: qwen3.8-27b-abliterated", vec![entry, "`family`", "qwen3.8-27b-abliterated", "qwen3.8-27b, qwen3.8-flash-next"]),
            ("card_class: 32\n    family: Qwen3.8-27B", vec![entry, "`family`", "Qwen3.8-27B"]),
            ("card_class: 32\n    family: 27", vec![entry, "`family`"]),
            // A `family` contradicting an alias that is a family's own id.
            ("card_class: 32\n    family: qwen3.8-flash-next\n    aliases: [qwen3.8-27b]", vec![entry, "`family`", "`qwen3.8-27b`", "Qwen3.8-Flash-Next", "Qwen3.8-27B"]),
            // An alias naming an entry of another family: the built-in
            // abliterated 16 GB build is a Qwen3.8-27B by the `family` it
            // states. (The abliterated 27B's own id is that build's alias,
            // which no other entry may claim.)
            ("card_class: 32\n    family: qwen3.8-flash-next\n    aliases: [qwen3.8-27b-abliterated-16g]", vec![entry, "`qwen3.8-27b-abliterated-16g`", "Qwen3.8-27B", "Qwen3.8-Flash-Next", "one model family"]),
            // An alias naming neither a family nor an entry: ids match
            // exactly, as a request's `model` does.
            ("card_class: 32\n    family: qwen3.8-27b\n    aliases: [acme-nothing]", vec![entry, "`acme-nothing`", "no entry's"]),
            ("card_class: 32\n    family: qwen3.8-27b\n    aliases: [QWEN3.8-27B-ABLITERATED]", vec![entry, "`QWEN3.8-27B-ABLITERATED`", "no entry's"]),
            // An entry listing an alias, whose id and aliases name no family,
            // and that states none.
            ("card_class: 32\n    aliases: [qwen3.8-27b-abliterated-16g]", vec![entry, "`family`", "neither its id nor its aliases"]),
        ] {
            let text = OPERATOR.replacen("card_class: 32", to, 1);
            let err = operator(&text).expect_err(to).0;
            assert!(err.contains("acme.catalog.yaml"), "{err}");
            for said in says {
                assert!(err.contains(said), "{to:?}: `{said}` missing from: {err}");
            }
        }

        // An alias naming an entry whose family nothing states: the
        // operator's own 16 GB build answering to its fine-tune, whose id
        // names none.
        let sixteen = OPERATOR
            .strip_prefix("models:\n")
            .unwrap()
            .replacen("id: acme-qwen3.8-27b-ft", "id: acme-qwen3.8-27b-ft-16g", 1)
            .replacen("card_class: 32", "card_class: 16\n    family: qwen3.8-27b\n    aliases: [acme-qwen3.8-27b-ft]", 1)
            .replace("acme_ft.ninfer", "acme_ft_16g.ninfer");
        let pair = format!("{OPERATOR}{sixteen}");
        let err = operator(&pair).unwrap_err().0;
        for said in ["acme.catalog.yaml", "`acme-qwen3.8-27b-ft-16g`", "alias `acme-qwen3.8-27b-ft`", "`family`"] {
            assert!(err.contains(said), "`{said}` missing from: {err}");
        }
        // Stated, the pair holds: the fine-tune's id is served by its 16 GB
        // build on a 16 GB card.
        let pair = operator(&pair.replacen("card_class: 32", "card_class: 32\n    family: qwen3.8-27b", 1)).expect("an alias of the operator's own entry");
        assert_eq!(resolved(&pair, "acme-qwen3.8-27b-ft", Some(CardClass::Gb16)), Ok("acme-qwen3.8-27b-ft-16g"));
        assert_eq!(resolved(&pair, "acme-qwen3.8-27b-ft", Some(CardClass::Gb32)), Ok("acme-qwen3.8-27b-ft"));

        // A `family` contradicting the entry's own id, a family's own id (a
        // built-in id, so read as the built-in layer).
        let text = OPERATOR.replacen("acme-qwen3.8-27b-ft", "qwen3.8-flash-next", 1).replacen("card_class: 32", "card_class: 32\n    family: qwen3.8-27b", 1);
        let err = parse(&Format::Yaml.read(&text).unwrap(), "test", CatalogLayer::BuiltIn).unwrap_err().0;
        for said in ["`qwen3.8-flash-next`", "`family`", "Qwen3.8-27B", "Qwen3.8-Flash-Next"] {
            assert!(err.contains(said), "`{said}` missing from: {err}");
        }
    }

    /// ADR 0074: a catalog written before `family` existed loads unchanged:
    /// an entry no alias involves, and one whose alias is a family's own id,
    /// state none and need none.
    #[test]
    fn an_operator_catalog_without_family_loads_unchanged() {
        let catalog = operator(OPERATOR).expect("an entry no alias involves");
        let acme = catalog.entry("acme-qwen3.8-27b-ft").unwrap();
        assert_eq!((acme.family, acme.family()), (None, None));
        let (_, entries) = SIXTEEN.split_once("models:\n").unwrap();
        let first = entries.split("\n  - ").next().unwrap();
        assert!(!first.contains("family:"), "{first}");
        let catalog = sixteen_over_thirty_two(&format!("models:\n{first}\n")).expect("an alias that is the family's own id");
        let sixteen = catalog.entry("qwen3.8-27b-16g").unwrap();
        assert_eq!((sixteen.family, sixteen.family()), (None, Some(ModelFamily::Qwen38_27b)));
        assert_eq!(resolved(&catalog, "qwen3.8-27b", Some(CardClass::Gb16)), Ok("qwen3.8-27b-16g"));
    }

    #[test]
    fn a_duplicate_id_is_refused_case_insensitively_within_the_file() {
        let body = OPERATOR.strip_prefix("models:\n").unwrap();
        let twice = format!("{OPERATOR}{}", body.replacen("acme-qwen3.8-27b-ft", "ACME-Qwen3.8-27B-FT", 1));
        let err = operator(&twice).unwrap_err().0;
        assert!(err.contains("entry 2 (`ACME-Qwen3.8-27B-FT`)") && err.contains("`acme-qwen3.8-27b-ft`"), "{err}");
    }

    /// ADR 0064 §1-2: an entry states the smallest card class it loads on
    /// and may list the ids it also answers to; the family the catalog can
    /// tell is the one whose own id it is or answers to, or the one it
    /// states (ADR 0074), and no other — the rest is the artifact's header's
    /// (ADR 0047).
    #[test]
    fn an_entry_states_its_card_class_and_may_list_aliases() {
        use ignis_core::compute::ModelFamily;

        let catalog = sixteen_over_thirty_two(SIXTEEN).expect("valid");
        let sixteen = catalog.entry("qwen3.8-27b-16g").expect("listed");
        assert_eq!(sixteen.card_class, Some(CardClass::Gb16));
        assert_eq!(sixteen.aliases, ["qwen3.8-27b"]);
        assert_eq!(sixteen.layer, CatalogLayer::Operator);
        assert_eq!(sixteen.family(), Some(ModelFamily::Qwen38_27b), "the family whose own id it answers to");
        assert_eq!(catalog.entry("qwen3.8-27b").unwrap().family(), Some(ModelFamily::Qwen38_27b));
        assert_eq!(catalog.entry("ignis-image-2.1-turbo").unwrap().family(), Some(ModelFamily::QwenImage21));
        let abliterated = catalog.entry("qwen3.8-27b-abliterated").unwrap();
        assert_eq!((abliterated.family, abliterated.family()), (Some(ModelFamily::Qwen38_27b), Some(ModelFamily::Qwen38_27b)), "its id names none: it states it (ADR 0074)");
        assert_eq!((sixteen.family, catalog.entry("qwen3.8-27b").unwrap().family), (None, None), "their ids and aliases say it: they state none");
        let json = serde_json::to_string(&Format::Yaml.read(SIXTEEN).unwrap()).unwrap();
        assert_eq!(parse_operator_over(&thirty_two(), &json, Format::Json, "acme.json").unwrap(), catalog);
        let built_in = Catalog::built_in().entry("qwen3.8-27b-16g").expect("the built-in 16 GB 27B");
        assert_eq!((built_in.card_class, built_in.aliases.as_slice(), built_in.layer), (Some(CardClass::Gb16), &["qwen3.8-27b".to_owned()][..], CatalogLayer::BuiltIn));
        assert_eq!(built_in.family(), Some(ModelFamily::Qwen38_27b));
    }

    /// ADR 0064 §1 as built: `card_class` is optional, by one rule — an
    /// entry that states none is never refused by class, as its own id or
    /// answering to an alias; the VRAM plan's refusal still guards it. So an
    /// operator catalog written before card classes still starts.
    #[test]
    fn an_entry_that_states_no_card_class_is_never_refused_by_class() {
        use CardClass::*;
        let catalog = operator(&OPERATOR.replacen("    card_class: 32\n", "", 1)).expect("a catalog from before card classes");
        let acme = catalog.entry("acme-qwen3.8-27b-ft").unwrap();
        assert_eq!(acme.card_class, None);
        for card in [Some(Gb16), Some(Gb32), None] {
            assert_eq!(catalog.above_class(&acme.id, card), None, "{card:?}");
            assert_eq!(resolved(&catalog, &acme.id, card), Ok(acme.id.as_str()), "{card:?}");
        }
        let classless = sixteen_over_thirty_two(&SIXTEEN.replacen("    card_class: 16\n", "", 1)).unwrap();
        assert_eq!(resolved(&classless, "qwen3.8-27b", Some(Gb16)), Ok("qwen3.8-27b-16g"), "it answers to its alias on any card");
    }

    /// ADR 0064 §2: an alias resolves only within one model family — an
    /// entry whose own id is a family's may not answer to another's.
    #[test]
    fn an_alias_of_another_family_than_the_entrys_own_id_is_refused() {
        let text = OPERATOR.replacen("acme-qwen3.8-27b-ft", "qwen3.8-flash-next", 1).replacen("card_class: 32", "card_class: 32\n    aliases: [qwen3.8-27b]", 1);
        let err = parse(&Format::Yaml.read(&text).unwrap(), "test", CatalogLayer::BuiltIn).unwrap_err().0;
        assert!(err.contains("`qwen3.8-flash-next`") && err.contains("`qwen3.8-27b`") && err.contains("one model family"), "{err}");
    }

    /// ADR 0064 §2: an id answers for one entry, or which one served it would
    /// be the catalog's order — refused within a file.
    #[test]
    fn an_alias_two_entries_claim_is_refused() {
        let (_, body) = SIXTEEN.split_once("models:\n").unwrap();
        let first = body.split("\n  - ").next().unwrap();
        let twice = format!("{SIXTEEN}\n{}\n", first.replacen("id: qwen3.8-27b-16g", "id: acme-27b-16g", 1));
        let err = sixteen_over_thirty_two(&twice).unwrap_err().0;
        assert!(
            err.contains("acme.catalog.yaml") && err.contains("`acme-27b-16g`") && err.contains("`qwen3.8-27b-16g`") && err.contains("alias `qwen3.8-27b`"),
            "{err}"
        );
    }

    /// ADR 0064 §2 across the two layers: an operator entry claiming an alias
    /// a built-in entry claims is refused by the check over the merged
    /// catalog, each file being valid on its own; an operator entry reusing
    /// the built-in 16 GB 27B's id is refused as any built-in id is.
    #[test]
    fn an_alias_a_built_in_entry_claims_is_refused_to_an_operator_entry() {
        let claiming = SIXTEEN.replacen("id: qwen3.8-27b-16g", "id: acme-27b-16g", 1);
        let err = operator(&claiming).unwrap_err().0;
        assert!(err.contains("acme.catalog.yaml") && err.contains("entry `acme-27b-16g`: alias `qwen3.8-27b` is already `qwen3.8-27b-16g`'s"), "{err}");
        assert!(operator(SIXTEEN).unwrap_err().0.contains("reuses the built-in id `qwen3.8-27b-16g`"));
        let joined = operator(OPERATOR).expect("an entry claiming no alias joins");
        assert_eq!(joined.ids().last(), Some(&"acme-qwen3.8-27b-ft"));
        assert!(joined.ids().contains(&"qwen3.8-27b-16g"));
    }

    /// The class-aware step's answer, as the id that serves `id` on a card
    /// of class `card` or the refusal's words.
    fn resolved<'a>(catalog: &'a Catalog, id: &'a str, card: Option<CardClass>) -> Result<&'a str, String> {
        match catalog.answering(id, card) {
            Ok(Some(entry)) => Ok(entry.id.as_str()),
            Ok(None) => Ok(id),
            Err(refusal) => Err(refusal.to_string()),
        }
    }

    /// Spec 27b-16gb/04 §6: an id whose own entry fits the card is served by
    /// it — on its class and on every larger one.
    #[test]
    fn an_id_resolves_to_its_own_entry_when_the_card_class_fits() {
        use CardClass::*;
        let catalog = Catalog::built_in();
        for (id, card) in [("qwen3.8-27b", Gb32), ("qwen3.8-27b", Gb96), ("qwen3.8-flash-next", Gb32), ("ignis-image-2.1-turbo", Gb96), ("qwen3.8-27b-16g", Gb16), ("qwen3.8-27b-16g", Gb32), ("qwen3.8-27b-abliterated", Gb32), ("qwen3.8-27b-abliterated-16g", Gb16)] {
            assert_eq!(catalog.above_class(id, Some(card)), None, "{id} on {card}");
            assert_eq!(catalog.answering(id, Some(card)), Ok(None), "{id} on {card}: it stands as it is");
        }
    }

    /// ADR 0064 §2: on a card below the id's own entry, the entry of its
    /// family that answers to it, and fits, serves it — each id its own
    /// answering entry: the 27B's default id the 16 GB 27B, the uncensored
    /// variant's its 16 GB build (ADR 0074).
    #[test]
    fn an_id_above_the_card_class_resolves_to_the_entry_that_answers_to_it() {
        let catalog = Catalog::built_in();
        for (id, answering) in [("qwen3.8-27b", "qwen3.8-27b-16g"), ("qwen3.8-27b-abliterated", "qwen3.8-27b-abliterated-16g")] {
            assert_eq!(resolved(&catalog, id, Some(CardClass::Gb16)), Ok(answering));
            assert!(catalog.above_class(id, Some(CardClass::Gb16)).is_some(), "{id}: its own entry is still above the class");
        }
    }

    /// ADR 0064 §3: with no entry answering to it, the id is refused naming
    /// its entry's class and the card's — the 27B's default id on a 16 GB
    /// card in a catalog without the 16 GB 27B.
    #[test]
    fn an_id_above_the_card_class_with_no_entry_answering_to_it_is_refused_naming_both_classes() {
        let err = resolved(&thirty_two(), "qwen3.8-27b", Some(CardClass::Gb16)).unwrap_err();
        assert!(err.contains("qwen3.8-27b") && err.contains("32 GB") && err.contains("16 GB"), "{err}");
        let refusal = ClassRefusal { model: "ignis-image-2.1-turbo".to_owned(), needs: CardClass::Gb32, card: CardClass::Gb16, other_family: None };
        assert_eq!(Catalog::built_in().answering("ignis-image-2.1-turbo", Some(CardClass::Gb16)), Err(refusal.clone()));
        assert_eq!(Catalog::built_in().above_class("ignis-image-2.1-turbo", Some(CardClass::Gb16)), Some(refusal));
    }

    /// ADR 0064 §2 on the artifact: the entry that answers to an alias must
    /// hold an artifact of the alias's family by its header (ADR 0047) —
    /// else the alias does not resolve, refused naming both families, and a
    /// header of no family is refused too. With no alias in play (the id's
    /// own entry fits) there is nothing to check.
    #[test]
    fn an_alias_resolves_only_to_an_artifact_of_its_family() {
        let catalog = Catalog::built_in();
        let sixteen = catalog.entry("qwen3.8-27b-16g").unwrap();
        let card = Some(CardClass::Gb16);
        assert_eq!(catalog.alias_holds("qwen3.8-27b", sixteen, Some(ModelFamily::Qwen38_27b), card), Ok(()));
        let refusal = catalog.alias_holds("qwen3.8-27b", sixteen, Some(ModelFamily::FlashNext), card).unwrap_err();
        assert_eq!(
            refusal.other_family,
            Some(OtherFamily { entry: "qwen3.8-27b-16g".to_owned(), artifact: Some(ModelFamily::FlashNext), alias: Some(ModelFamily::Qwen38_27b) })
        );
        let said = refusal.to_string();
        for named in ["qwen3.8-27b needs a 32 GB card", "16 GB class", "qwen3.8-27b-16g lists it as an alias", "Qwen3.8-Flash-Next", "Qwen3.8-27B", "one model family"] {
            assert!(said.contains(named), "{named}: {said}");
        }
        assert!(catalog.alias_holds("qwen3.8-27b", sixteen, None, card).unwrap_err().to_string().contains("artifact is no model family"));
        for card in [Some(CardClass::Gb32), None] {
            assert_eq!(catalog.alias_holds("qwen3.8-27b", sixteen, Some(ModelFamily::FlashNext), card), Ok(()), "{card:?}: no alias in play");
        }
    }

    /// ADR 0074: an alias may be another entry's served id of the same
    /// family, answered on a card below that entry's class. The built-in
    /// 16 GB build of the uncensored variant (GitHub #423) states the 27B's
    /// family and answers to `qwen3.8-27b-abliterated`, whose entry states it
    /// too, on a 16 GB card; on a 32 GB card, a 96 GB one or one of no known
    /// class the id is its own entry's.
    #[test]
    fn an_alias_may_be_another_entrys_served_id_of_the_same_family() {
        use CardClass::*;
        let catalog = Catalog::built_in();
        let entry = catalog.entry("qwen3.8-27b-abliterated-16g").expect("listed");
        assert_eq!((entry.card_class, entry.family, entry.aliases.as_slice()), (Some(Gb16), Some(ModelFamily::Qwen38_27b), &["qwen3.8-27b-abliterated".to_owned()][..]));
        assert_eq!(resolved(&catalog, "qwen3.8-27b-abliterated", Some(Gb16)), Ok("qwen3.8-27b-abliterated-16g"));
        assert_eq!(resolved(&catalog, "qwen3.8-27b", Some(Gb16)), Ok("qwen3.8-27b-16g"), "each id its own answering entry");
        for card in [Some(Gb32), Some(Gb96), None] {
            assert_eq!(catalog.answering("qwen3.8-27b-abliterated", card), Ok(None), "{card:?}: its own entry");
        }
    }

    /// ADR 0074 on the artifact: the alias's family is the one its entry
    /// states when its id names none, and the header stays the authority —
    /// the abliterated 16 GB build holding an artifact of another family does
    /// not answer to `qwen3.8-27b-abliterated`, refused naming both families.
    #[test]
    fn an_alias_of_a_stated_family_resolves_only_to_an_artifact_of_that_family() {
        let catalog = Catalog::built_in();
        let entry = catalog.entry("qwen3.8-27b-abliterated-16g").unwrap();
        let card = Some(CardClass::Gb16);
        assert_eq!(catalog.alias_holds("qwen3.8-27b-abliterated", entry, Some(ModelFamily::Qwen38_27b), card), Ok(()));
        let refusal = catalog.alias_holds("qwen3.8-27b-abliterated", entry, Some(ModelFamily::FlashNext), card).unwrap_err();
        let said = refusal.to_string();
        for named in [
            "qwen3.8-27b-abliterated needs a 32 GB card",
            "16 GB class",
            "qwen3.8-27b-abliterated-16g lists it as an alias",
            "its artifact is Qwen3.8-Flash-Next and qwen3.8-27b-abliterated is Qwen3.8-27B",
            "one model family",
        ] {
            assert!(said.contains(named), "{named}: {said}");
        }
        assert!(catalog.alias_holds("qwen3.8-27b-abliterated", entry, None, card).unwrap_err().to_string().contains("artifact is no model family"));
        assert_eq!(catalog.alias_holds("qwen3.8-27b-abliterated", entry, Some(ModelFamily::FlashNext), Some(CardClass::Gb32)), Ok(()), "no alias in play");
    }

    /// ADR 0064 §2: an entry answering to an alias never serves another
    /// family's id: on a 16 GB card, where the 27B's two 16 GB entries answer
    /// to their own aliases, Flash-Next's and the image model's are refused.
    #[test]
    fn an_alias_never_resolves_across_families() {
        let catalog = Catalog::built_in();
        for id in ["qwen3.8-flash-next", "ignis-image-2.1-turbo"] {
            let err = resolved(&catalog, id, Some(CardClass::Gb16)).unwrap_err();
            assert!(err.contains(id) && err.contains("32 GB") && err.contains("16 GB"), "{err}");
        }
    }

    /// Spec 27b-16gb/04 §6: a card of no known class refuses nothing — the
    /// VRAM plan's refusal still guards — and an id is its own entry's.
    #[test]
    fn an_unknown_card_class_refuses_nothing() {
        let catalog = Catalog::built_in();
        for id in ["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next", "ignis-image-2.1-turbo", "qwen3.8-27b-16g", "qwen3.8-27b-abliterated-16g"] {
            assert_eq!(catalog.answering(id, None), Ok(None), "{id}: the id's own entry, never the one answering to it");
            assert_eq!(catalog.above_class(id, None), None, "{id}");
        }
    }

    /// An id no entry holds is not the catalog's to judge: a known model's,
    /// or a typo the route's own check refuses by name. Matched exactly, as
    /// a request's `model` is.
    #[test]
    fn an_id_the_catalog_does_not_hold_is_not_its_to_resolve() {
        let catalog = Catalog::built_in();
        for card in [None, Some(CardClass::Gb16), Some(CardClass::Gb32)] {
            for id in ["acme-model", "QWEN3.8-27B"] {
                assert_eq!(catalog.answering(id, card), Ok(None), "{id} {card:?}");
                assert_eq!(catalog.above_class(id, card), None, "{id} {card:?}");
            }
        }
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

    /// The allowlist behind AC 2: every built-in file name passes, the
    /// shapes that escape a directory or name a Windows device do not.
    #[test]
    fn plain_name_is_an_allowlist() {
        for entry in Catalog::built_in().entries() {
            for file in &entry.files {
                assert!(plain_name(&file.name), "{}", file.name);
            }
        }
        for name in ["qwen3_8_27b_nvfp4full-v2.ninfer", "a.b.c", "x-1_2", "console.ninfer", "COM10", "LPT0"] {
            assert!(plain_name(name), "{name}");
        }
        for name in [
            "", ".", "..", "NUL", "nul", "con.ninfer", "Com1.tar.gz", "aux", "lpt9.x", "a.ninfer.", ".hidden", "a b", "a\tb", "a\u{7f}b", "../x", "/x", "C:x", "a/b", r"a\b",
            "a*b", "a?b", "a<b", "a|b", "a\"b", "caf\u{e9}",
        ] {
            assert!(!plain_name(name), "{name:?}");
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
