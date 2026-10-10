//! The config file (spec config-v2/01 §The config file, config-v2/02): one
//! document, JSON or YAML, nested exactly as the flags are named —
//! `<group>: { <field>: value, <family>: { <field>: value } }` — plus an
//! optional `profile:` naming the profile to use and a `profiles:` section
//! defining custom ones in the same shape.
//!
//! A file is read into [`Layer`]s, never straight into the settings: it is
//! one source among several, and a field it does not name is decided by the
//! sources below it. Each value is turned back into the text its kind parses
//! ([`super::field::FieldMeta::file_text`]) and goes through the same parser
//! a flag does — so `kv_host_pool_bytes: 2G` and `--reuse-kv-host-pool-bytes
//! 2G` cannot be read differently, and a wrong value in a file is refused
//! naming its key, never by a serde type error or a silent default.
//!
//! The filesystem is behind [`Files`], so resolution stays testable without
//! one: [`NoFiles`] is what the pure `config::resolve` runs with.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ignis_core::compute::ModelFamily;
use serde_json::{Map, Value};

use super::field::{family_of_scope, scope, FieldMeta, FAMILIES};
use super::schema::{self, all_fields};
use super::source::{resolve_settings, Candidate, Fit, Layer};
use super::{Config, ConfigError};

/// A config file's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `.json`.
    Json,
    /// `.yaml` / `.yml`, and the format of a file with neither.
    Yaml,
}

impl Format {
    /// The format a path's extension names: `.json`, or `.yaml`/`.yml`.
    pub fn of_path(path: &Path) -> Option<Format> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "json" => Some(Format::Json),
            "yaml" | "yml" => Some(Format::Yaml),
            _ => None,
        }
    }

    /// `json` or `yaml`, as `--format` spells it.
    pub fn parse(name: &str) -> Option<Format> {
        match name.to_ascii_lowercase().as_str() {
            "json" => Some(Format::Json),
            "yaml" | "yml" => Some(Format::Yaml),
            _ => None,
        }
    }

    /// Parse a document. A file with no recognized extension is read as YAML,
    /// which reads JSON too.
    pub fn read(self, text: &str) -> Result<Value, String> {
        match self {
            Format::Json => serde_json::from_str(text).map_err(|e| e.to_string()),
            Format::Yaml => serde_yaml::from_str::<Option<Value>>(text)
                .map(|value| value.unwrap_or(Value::Null))
                .map_err(|e| e.to_string()),
        }
    }

    /// Write a document, ending in a newline, its keys in the order the
    /// fields are declared ([`Ordered`]) rather than alphabetically.
    pub fn write(self, value: &Value) -> String {
        let ordered = Ordered { value, at: Level::Top };
        match self {
            Format::Json => {
                let mut text = serde_json::to_string_pretty(&ordered).expect("a JSON value always serializes");
                text.push('\n');
                text
            }
            Format::Yaml => serde_yaml::to_string(&ordered).expect("a JSON value always serializes as YAML"),
        }
    }
}

/// Where in a config document a map sits, which decides its keys' order.
#[derive(Clone, Copy)]
enum Level {
    /// `profile`, the groups, `profiles`.
    Top,
    /// The fields of one group, then its family sections.
    Group(usize),
    /// A family section's fields.
    Scoped(usize),
    /// The profile names, alphabetical.
    Profiles,
    /// Inside a value (a known-models map): as stored.
    Value,
}

/// A document serialized with its keys in declaration order: `profile`,
/// then the groups as `schema.rs` declares them, each with its fields in
/// order and its family sections last, then `profiles`. `serde_json`'s map
/// is sorted (this workspace does not turn on `preserve_order`, which would
/// reorder every JSON body the server writes), and a generated file read
/// top to bottom should follow `help --fields`, not the alphabet.
struct Ordered<'a> {
    value: &'a Value,
    at: Level,
}

impl serde::Serialize for Ordered<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let Value::Object(map) = self.value else {
            return self.value.serialize(serializer);
        };
        let rank = |key: &str| -> (usize, Level) {
            let groups = schema::GROUPS;
            let group_index = |name: &str| groups.iter().position(|(group, _)| *group == name);
            let field_index = |group: usize, name: &str| groups[group].1.iter().position(|meta| meta.name == name);
            match self.at {
                Level::Top => match key {
                    "profile" => (0, Level::Value),
                    "profiles" => (usize::MAX, Level::Profiles),
                    _ => group_index(key).map_or((usize::MAX - 1, Level::Value), |g| (1 + g, Level::Group(g))),
                },
                Level::Group(g) => match field_index(g, key) {
                    Some(i) => (i, Level::Value),
                    None => (1000 + family_of_scope(key).map_or(99, |f| f as usize), Level::Scoped(g)),
                },
                Level::Scoped(g) => (field_index(g, key).unwrap_or(usize::MAX), Level::Value),
                Level::Profiles => (0, Level::Top),
                Level::Value => (0, Level::Value),
            }
        };
        let mut entries: Vec<(usize, Level, &String, &Value)> =
            map.iter().map(|(key, value)| { let (r, at) = rank(key); (r, at, key, value) }).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(b.2)));
        let mut out = serializer.serialize_map(Some(entries.len()))?;
        for (_, at, key, value) in entries {
            out.serialize_entry(key, &Ordered { value, at })?;
        }
        out.end()
    }
}

/// The filesystem, as configuration sees it: the one seam between
/// resolution and the disk.
pub trait Files {
    /// The file's text.
    fn read(&self, path: &Path) -> std::io::Result<String>;
    /// Replace the file's text with `contents`.
    fn write(&self, path: &Path, contents: &str) -> std::io::Result<()>;
    /// Whether there is a file at `path` (what discovery asks of each
    /// candidate).
    fn exists(&self, path: &Path) -> bool;
}

/// The real filesystem, for `main`.
pub struct RealFiles;

impl Files for RealFiles {
    fn read(&self, path: &Path) -> std::io::Result<String> {
        std::fs::read_to_string(path)
    }

    fn write(&self, path: &Path, contents: &str) -> std::io::Result<()> {
        std::fs::write(path, contents)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

/// No filesystem at all: nothing exists, nothing can be read or written.
/// What the pure [`super::resolve`] runs with, so a test never finds a
/// config file lying in the directory it happens to run in.
pub struct NoFiles;

impl Files for NoFiles {
    fn read(&self, _: &Path) -> std::io::Result<String> {
        Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no filesystem"))
    }

    fn write(&self, _: &Path, _: &str) -> std::io::Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "no filesystem"))
    }

    fn exists(&self, _: &Path) -> bool {
        false
    }
}

/// What a config file holds, as layers.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Document {
    /// Its field values, general and per family.
    pub values: Layer,
    /// The profile it names (`profile: <name>`), if any.
    pub profile: Option<String>,
    /// The profiles it defines (`profiles: { <name>: { <group>: … } }`).
    pub profiles: BTreeMap<String, Layer>,
}

/// Read and parse the file at `path`: the document as written (what a
/// merge rewrites) and as layers (what resolution reads).
pub fn load(files: &dyn Files, path: &Path) -> Result<(Value, Document), ConfigError> {
    let text = files
        .read(path)
        .map_err(|e| ConfigError(format!("the config file {} cannot be read: {e}", path.display())))?;
    let value = Format::of_path(path)
        .unwrap_or(Format::Yaml)
        .read(&text)
        .map_err(|e| ConfigError(format!("the config file {} is not valid: {e}", path.display())))?;
    let document = read_document(&value, &path.display().to_string())?;
    Ok((value, document))
}

/// A parsed document as layers; `origin` (the file's path) is named in every
/// value's spelling, so an error says which file to open.
pub fn read_document(value: &Value, origin: &str) -> Result<Document, ConfigError> {
    let mut document = Document::default();
    let top = match value {
        Value::Null => return Ok(document),
        Value::Object(top) => top,
        _ => return Err(ConfigError(format!("the config file {origin} must be a map of groups"))),
    };
    for (key, value) in top {
        match key.as_str() {
            "profile" => match value {
                Value::String(name) => document.profile = Some(name.clone()),
                Value::Null => {}
                _ => return Err(ConfigError(format!("`profile` in {origin} must be a profile's name"))),
            },
            "profiles" => {
                let Value::Object(profiles) = value else {
                    return Err(ConfigError(format!("`profiles` in {origin} must map names to groups")));
                };
                for (name, body) in profiles {
                    let Value::Object(body) = body else {
                        return Err(ConfigError(format!("profile `{name}` in {origin} must be a map of groups")));
                    };
                    let mut layer = Layer::default();
                    for (group, fields) in body {
                        read_group(group, fields, &format!("profile {name} in {origin}"), &mut layer)?;
                    }
                    document.profiles.insert(name.clone(), layer);
                }
            }
            group => read_group(group, value, origin, &mut document.values)?,
        }
    }
    Ok(document)
}

/// One `<group>: { … }` section into `layer`.
fn read_group(group: &str, value: &Value, origin: &str, layer: &mut Layer) -> Result<(), ConfigError> {
    let Some((group, fields)) = schema::GROUPS.iter().find(|(name, _)| *name == group) else {
        let groups: Vec<&str> = schema::GROUPS.iter().map(|(name, _)| *name).collect();
        return Err(ConfigError(format!(
            "`{group}` in {origin} is not a config group (the groups are {})",
            groups.join(", ")
        )));
    };
    let entries = match value {
        Value::Null => return Ok(()),
        Value::Object(entries) => entries,
        _ => return Err(ConfigError(format!("`{group}` in {origin} must map field names to values"))),
    };
    for (key, value) in entries {
        if let Some(family) = family_of_scope(key) {
            let Value::Object(scoped) = value else {
                return Err(ConfigError(format!("`{group}.{key}` in {origin} must map field names to values")));
            };
            for (name, value) in scoped {
                let meta = field_of(fields, group, name, origin)?;
                set(layer, meta, Some(family), value, &format!("{group}.{key}.{name}"), origin)?;
            }
        } else {
            let meta = field_of(fields, group, key, origin)?;
            set(layer, meta, None, value, &format!("{group}.{key}"), origin)?;
        }
    }
    Ok(())
}

fn field_of(fields: &'static [FieldMeta], group: &str, name: &str, origin: &str) -> Result<&'static FieldMeta, ConfigError> {
    fields.iter().find(|meta| meta.name == name).ok_or_else(|| {
        ConfigError(format!("`{group}.{name}` in {origin} is not a field of `{group}` (`ignis-server help --fields` lists them)"))
    })
}

fn set(layer: &mut Layer, meta: &'static FieldMeta, family: Option<ModelFamily>, value: &Value, key: &str, origin: &str) -> Result<(), ConfigError> {
    let spelling = format!("{key} ({origin})");
    let Some(raw) = (meta.file_text)(value).map_err(|reason| ConfigError(format!("`{spelling}` {reason}")))? else {
        return Ok(());
    };
    layer.set(meta, family, Candidate { raw, spelling })
}

/// `config`'s settings as a document: every field's value for no family in
/// particular, and beside it, in each family's section, every scoped field
/// whose value differs for that family — what the sources say, written so a
/// start reading it back resolves the same values. What `config generate`
/// and `config print` write and `GET /v1/config` shows.
pub fn effective_document(config: &Config) -> Value {
    let mut general = config.settings().render();
    // A field only one family takes is never written at the general scope,
    // even when that family's own value is the one it already has there:
    // the general section is read by every family, and this one refuses it
    // outright (the same rule `config patch` already writes by, GitHub
    // #311's family-section fix). Removing it here first is what turns its
    // ordinary "does the scoped value differ from the general one" check,
    // below, into "yes" unconditionally for this field -- no second rule
    // needed.
    for meta in all_fields().filter(|meta| meta.home_family().is_some()) {
        if let Some(group) = general.get_mut(meta.group).and_then(Value::as_object_mut) {
            group.remove(meta.name);
        }
    }
    let mut changes = Vec::new();
    for family in FAMILIES {
        let Ok(resolution) = resolve_settings(config.basis.sources(), Some(family), Fit::Switch) else {
            continue;
        };
        let scoped = resolution.settings.render();
        for meta in all_fields().filter(|meta| meta.takes_scope(family)) {
            let at = |groups: &Map<String, Value>| groups.get(meta.group).and_then(|group| group.get(meta.name)).cloned();
            let value = at(&scoped);
            if value != at(&general) {
                changes.push((meta, Some(family), value.unwrap_or(Value::Null)));
            }
        }
    }
    let mut document = Value::Object(general);
    merge(&mut document, changes);
    document
}

/// One field's change, as [`merge`] writes it: the field, its scope, and the
/// value in the form the file format writes.
pub type Change = (&'static FieldMeta, Option<ModelFamily>, Value);

/// Write `changes` into `document`, leaving every other key — other fields,
/// other families, `profile`, `profiles` — as it was: the one function that
/// changes an existing config file, shared by `config patch` and `PATCH
/// /v1/config` so the two can never write a change differently.
pub fn merge(document: &mut Value, changes: impl IntoIterator<Item = Change>) {
    if !document.is_object() {
        *document = Value::Object(Map::new());
    }
    let top = document.as_object_mut().expect("an object, made one above");
    for (meta, family, value) in changes {
        let group = top.entry(meta.group.to_owned()).or_insert_with(|| Value::Object(Map::new()));
        if !group.is_object() {
            *group = Value::Object(Map::new());
        }
        let mut section = group.as_object_mut().expect("an object, made one above");
        if let Some(family) = family {
            let scoped = section.entry(scope(family).to_owned()).or_insert_with(|| Value::Object(Map::new()));
            if !scoped.is_object() {
                *scoped = Value::Object(Map::new());
            }
            section = scoped.as_object_mut().expect("an object, made one above");
        }
        section.insert(meta.name.to_owned(), value);
    }
}

/// The text of the config file at `path` with `changes` merged in (and, when
/// given, `profile:` set) — what both `config patch` and `PATCH /v1/config`
/// write, so the two can never change a file differently. Every key the
/// changes do not name is kept; the keys are written in declaration order;
/// a YAML file's comments do not survive the rewrite.
pub fn rewrite(files: &dyn Files, path: &Path, changes: Vec<Change>, profile: Option<&str>) -> Result<String, ConfigError> {
    let (mut document, _) = load(files, path)?;
    merge(&mut document, changes);
    if let Some(name) = profile {
        document.as_object_mut().expect("a merged document is a map").insert("profile".to_owned(), Value::String(name.to_owned()));
    }
    Ok(Format::of_path(path).unwrap_or(Format::Yaml).write(&document))
}

/// The changes a layer stands for: each value it sets, parsed by its field's
/// kind and written back in canonical form (`2147483648` as `2G`).
pub fn changes_of(layer: &Layer) -> Result<Vec<Change>, ConfigError> {
    layer
        .entries()
        .map(|(meta, family, candidate)| {
            (meta.canonical)(candidate.raw.trim())
                .map(|value| (meta, family, value))
                .map_err(|reason| ConfigError(format!("`{}` {reason}", candidate.spelling)))
        })
        .collect()
}

/// The conventional places a config file is looked for when none is named
/// (spec config-v2/02 §Config-file auto-discovery), in order: the working
/// directory (YAML, then JSON), then the per-user config directory —
/// `%APPDATA%\ignis\config.yaml` on Windows, `$XDG_CONFIG_HOME/ignis/
/// config.yaml` or `~/.config/ignis/config.yaml` elsewhere. Read from `env`
/// so a test controls them.
pub fn discovery_candidates(env: &dyn Fn(&str) -> Option<String>) -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from("ignis.config.yaml"), PathBuf::from("ignis.config.json")];
    let user_dir = if cfg!(windows) {
        env("APPDATA").filter(|dir| !dir.is_empty()).map(PathBuf::from)
    } else {
        env("XDG_CONFIG_HOME")
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
            .or_else(|| env("HOME").filter(|dir| !dir.is_empty()).map(|home| PathBuf::from(home).join(".config")))
    };
    if let Some(dir) = user_dir {
        candidates.push(dir.join("ignis").join("config.yaml"));
    }
    candidates
}

/// An in-memory [`Files`] for tests: what exists, what was looked at, what
/// was written.
#[cfg(test)]
pub(crate) mod testing {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use super::Files;

    /// Thread-safe, so a server's shared config state can write through it.
    #[derive(Default)]
    pub struct MemFiles {
        pub files: Mutex<BTreeMap<PathBuf, String>>,
        /// Every path `exists` was asked about, in order.
        pub looked_at: Mutex<Vec<PathBuf>>,
        /// Every path written.
        pub written: Mutex<Vec<PathBuf>>,
        /// Refuse every write, as a read-only volume would.
        pub read_only: bool,
    }

    impl MemFiles {
        pub fn with(files: &[(&str, &str)]) -> Self {
            let memory = Self::default();
            for (path, text) in files {
                memory.files.lock().unwrap().insert(PathBuf::from(path), text.to_string());
            }
            memory
        }
    }

    impl Files for MemFiles {
        fn read(&self, path: &Path) -> std::io::Result<String> {
            self.files
                .lock()
                .unwrap()
                .get(path)
                .cloned()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "not in memory"))
        }

        fn write(&self, path: &Path, contents: &str) -> std::io::Result<()> {
            if self.read_only {
                return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "read-only"));
            }
            self.written.lock().unwrap().push(path.to_owned());
            self.files.lock().unwrap().insert(path.to_owned(), contents.to_owned());
            Ok(())
        }

        fn exists(&self, path: &Path) -> bool {
            self.looked_at.lock().unwrap().push(path.to_owned());
            self.files.lock().unwrap().contains_key(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::field;
    use serde_json::json;

    #[test]
    fn a_document_reads_into_general_family_and_profile_layers() {
        let value = json!({
            "profile": "small-card",
            "reuse": { "kv_host_pool_bytes": "8G", "qwen38flashnext": { "kv_host_pool_bytes": "1G" } },
            "model": { "max_context": 131072, "artifact": null },
            "server": { "ui": false },
            "profiles": { "small-card": { "vram": { "headroom_bytes": "2G" } } }
        });
        let document = read_document(&value, "ignis.config.yaml").unwrap();
        assert_eq!(document.profile.as_deref(), Some("small-card"));
        let pool = field("reuse", "kv_host_pool_bytes").unwrap();
        assert_eq!(document.values.get(pool, None).unwrap().raw, "8G");
        let scoped = document.values.get(pool, Some(ModelFamily::FlashNext)).unwrap();
        assert_eq!(scoped.raw, "1G");
        assert_eq!(scoped.spelling, "reuse.qwen38flashnext.kv_host_pool_bytes (ignis.config.yaml)");
        assert_eq!(document.values.get(field("model", "max_context").unwrap(), None).unwrap().raw, "131072");
        assert!(document.values.get(field("model", "artifact").unwrap(), None).is_none(), "null is unset");
        assert_eq!(document.values.get(field("server", "ui").unwrap(), None).unwrap().raw, "false");
        let profile = &document.profiles["small-card"];
        assert_eq!(profile.get(field("vram", "headroom_bytes").unwrap(), None).unwrap().raw, "2G");
    }

    /// Spec config-v2/01, Testing: an unknown field or a wrong type is a
    /// `ConfigError` naming it, not a serde panic or a silent default.
    #[test]
    fn an_unknown_key_or_a_value_of_the_wrong_shape_is_refused_by_name() {
        for (value, says) in [
            (json!({ "serverr": {} }), "`serverr`"),
            (json!({ "server": { "bnd": "x" } }), "`server.bnd`"),
            (json!({ "server": { "qwen38": { "bind": "x" } } }), "server.qwen38.bind"),
            (json!({ "model": { "max_context": [1, 2] } }), "model.max_context"),
            (json!({ "model": "big" }), "`model`"),
            (json!(["a"]), "map of groups"),
            (json!({ "profiles": { "p": { "nope": {} } } }), "profile p"),
        ] {
            let err = read_document(&value, "f.yaml").expect_err(&value.to_string()).0;
            assert!(err.contains(says) && err.contains("f.yaml"), "{value}: {err}");
        }
    }

    #[test]
    fn both_formats_read_the_same_document() {
        let yaml = "reuse:\n  kv_host_pool_bytes: 8G\n  qwen38:\n    prompt: off\nmodel:\n  max_context: 131072\n";
        let json = r#"{"reuse": {"kv_host_pool_bytes": "8G", "qwen38": {"prompt": "off"}}, "model": {"max_context": 131072}}"#;
        let from_yaml = read_document(&Format::Yaml.read(yaml).unwrap(), "f").unwrap();
        let from_json = read_document(&Format::Json.read(json).unwrap(), "f").unwrap();
        assert_eq!(from_yaml, from_json);
        assert_eq!(Format::Yaml.read("").unwrap(), Value::Null, "an empty file is an empty config");
        assert_eq!(Format::of_path(Path::new("a/ignis.config.YML")), Some(Format::Yaml));
        assert_eq!(Format::of_path(Path::new("c.json")), Some(Format::Json));
        assert_eq!(Format::of_path(Path::new("c.toml")), None);
    }

    #[test]
    fn a_merge_changes_only_what_it_names() {
        let mut document = json!({
            "profile": "p",
            "reuse": { "prompt": false, "qwen38": { "retained_host": 4 } },
            "profiles": { "p": { "vram": { "headroom_bytes": "2G" } } }
        });
        let pool = field("reuse", "kv_host_pool_bytes").unwrap();
        let ttl = field("reuse", "retained_interactive_ttl").unwrap();
        merge(&mut document, [(pool, None, json!("8G")), (ttl, Some(ModelFamily::Qwen38_27b), json!(60))]);
        assert_eq!(
            document,
            json!({
                "profile": "p",
                "reuse": { "prompt": false, "kv_host_pool_bytes": "8G", "qwen38": { "retained_host": 4, "retained_interactive_ttl": 60 } },
                "profiles": { "p": { "vram": { "headroom_bytes": "2G" } } }
            })
        );
        let mut empty = Value::Null;
        merge(&mut empty, [(pool, None, json!("1G"))]);
        assert_eq!(empty, json!({ "reuse": { "kv_host_pool_bytes": "1G" } }));
    }

    #[test]
    fn changes_are_written_in_canonical_form() {
        let mut layer = Layer::default();
        let pool = field("reuse", "kv_host_pool_bytes").unwrap();
        layer.set(pool, None, Candidate { raw: "2147483648".into(), spelling: "IGNIS_REUSE_KV_HOST_POOL_BYTES".into() }).unwrap();
        let changes: Vec<_> = changes_of(&layer).unwrap().into_iter().map(|(meta, family, value)| (meta.file_key(), family, value)).collect();
        assert_eq!(changes, vec![("reuse.kv_host_pool_bytes".to_owned(), None, json!("2G"))]);
        layer.set(pool, None, Candidate { raw: "lots".into(), spelling: "IGNIS_REUSE_KV_HOST_POOL_BYTES".into() }).unwrap();
        assert!(changes_of(&layer).unwrap_err().0.contains("IGNIS_REUSE_KV_HOST_POOL_BYTES"));
    }

    #[test]
    fn discovery_looks_in_the_working_directory_then_the_user_config_directory() {
        let env = |name: &str| match name {
            "APPDATA" => Some("C:/Users/u/AppData/Roaming".to_owned()),
            "XDG_CONFIG_HOME" => Some("/home/u/.xdg".to_owned()),
            "HOME" => Some("/home/u".to_owned()),
            _ => None,
        };
        let candidates = discovery_candidates(&env);
        assert_eq!(candidates[..2], [PathBuf::from("ignis.config.yaml"), PathBuf::from("ignis.config.json")]);
        let user = if cfg!(windows) { "C:/Users/u/AppData/Roaming" } else { "/home/u/.xdg" };
        assert_eq!(candidates[2], PathBuf::from(user).join("ignis").join("config.yaml"));
        assert_eq!(discovery_candidates(&|_: &str| None).len(), 2, "no user directory known, none looked in");
    }
}
