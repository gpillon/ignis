//! Where a config value can come from, and which one wins (spec
//! config-v2/01 §Precedence resolution, extended by config-v2/02's
//! profiles and live patches).
//!
//! Every source is gathered first into a [`Layer`] of raw text per field and
//! scope — the command line, the environment, a config file, a profile, a
//! live `PATCH` — and only then is each field resolved, by
//! [`Resolver::field`], walking the layers in precedence order:
//!
//! ```text
//! patch > <family>-flag > flag > <family>-env > env > <family>-config > config
//!       > <family>-profile > profile > hardcoded default
//! ```
//!
//! The family scope is a tiebreaker *within* a source, never a source of
//! its own: the command line wins whatever its scope, and inside any one
//! source the more specific value wins. The first value found is the only
//! one parsed — a lower-precedence value that would not parse is never
//! reached, so an operator overriding a bad env var with a good flag is not
//! refused for the env var.
//!
//! Gathering the layers once and keeping them ([`Sources`], carried by every
//! resolved [`super::Config`]) is what lets one start's options be resolved
//! again later for a family the artifact names, or with a patch on top,
//! without the process environment or argv being read twice.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use ignis_core::compute::ModelFamily;

use super::field::{scope, FieldMeta, FAMILIES};
use super::kind::FieldKind;
use super::schema::{all_fields, Settings};
use super::ConfigError;

/// A source, in precedence order: the earlier variant wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    /// A live `PATCH /v1/config` (or a switch target's own artifact and id):
    /// what the running process was told after it started.
    Patch,
    /// The command line (`--<group>-<field>`, `--<family>-…`).
    Flag,
    /// The environment (`IGNIS_<GROUP>_<FIELD>`, `IGNIS_<FAMILY>_…`).
    Env,
    /// The config file (`<group>.<field>`, `<group>.<family>.<field>`).
    File,
    /// The hardware profile in use: a default, never an override.
    Profile,
    /// The field's hardcoded default.
    Default,
}

impl Source {
    /// Whether the operator wrote this value for this process — every source
    /// but a profile and the hardcoded default. Only an explicit value is
    /// refused for a family that cannot take it, and only an explicit value
    /// counts as "named" by a rule that needs one field to have been named
    /// beside another (`--vision-max-tokens` without vision).
    pub fn explicit(self) -> bool {
        matches!(self, Source::Patch | Source::Flag | Source::Env | Source::File)
    }
}

/// Where a resolved value came from: the source, and the exact spelling the
/// operator used there (`--qwen38-reuse-kv-host-pool-bytes`,
/// `IGNIS_REUSE_KV_HOST_POOL_BYTES`, `reuse.kv_host_pool_bytes` in a file) —
/// what an error names, so it points at the line to fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    /// Which source gave the value.
    pub source: Source,
    /// The exact name it was given under — a flag, an env var, a file key
    /// with its file, a profile's key.
    pub spelling: String,
    /// The value given is the field's hardcoded default. Naming the default
    /// says nothing: such a value is never refused for a family that cannot
    /// take the field, and never counts as "named" by a rule that ties one
    /// field to another — so a file `config generate` wrote with every
    /// default (`vision.enabled: false`, `server.metrics_bind:
    /// 127.0.0.1:9464`) starts either model, as no file at all would.
    pub at_default: bool,
}

/// One raw value and the spelling it arrived under.
#[derive(Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The text as given, before its kind parses it (a file's value turned
    /// back into text first).
    pub raw: String,
    /// The name it was given under, for an error to point at.
    pub spelling: String,
}

/// A field's place in a layer: `(group, name, scope)`, the scope `None` for
/// the group's general value.
type Key = (&'static str, &'static str, Option<&'static str>);

/// One source's raw values, general and per family.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Layer {
    values: BTreeMap<Key, Candidate>,
}

impl std::fmt::Debug for Layer {
    /// The spellings set, never their values: a layer holds the API key's
    /// raw text, and a `Config` dumped into a log line must not leak it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.values.values().map(|candidate| &candidate.spelling)).finish()
    }
}

impl Layer {
    /// The value set for `meta` in `family`'s scope (`None`: the general one).
    pub fn get(&self, meta: &FieldMeta, family: Option<ModelFamily>) -> Option<&Candidate> {
        self.values.get(&(meta.group, meta.name, family.map(scope)))
    }

    /// Set `meta`'s value in `family`'s scope. An empty value unsets it — an
    /// empty env var or flag value is an unset one everywhere in this config,
    /// never an empty string.
    ///
    /// A family-scoped value is refused here, whatever source it came from,
    /// when the family cannot take the field at all
    /// (`IGNIS_QWEN38_NGRAM_HOT_BYTES`) or the field has no place in that
    /// family's section ([`FieldMeta::takes_scope`]): such a value was
    /// written for a family that can never use it, so it is wrong the moment
    /// it is written. A field only one family takes may sit in that family's
    /// section, which is where `PATCH /v1/config` writes it.
    pub fn set(&mut self, meta: &'static FieldMeta, family: Option<ModelFamily>, candidate: Candidate) -> Result<(), ConfigError> {
        if let Some(family) = family {
            if !meta.applies.to(family) {
                return Err(ConfigError(format!(
                    "`{}`: {} does not take {} (a {} option)",
                    candidate.spelling,
                    family.name(),
                    meta.file_key(),
                    meta.applies.describe()
                )));
            }
            if !meta.takes_scope(family) {
                return Err(ConfigError(format!(
                    "`{}`: {} has no per-family value; set `{}` instead",
                    candidate.spelling,
                    meta.file_key(),
                    meta.file_key()
                )));
            }
        }
        let key = (meta.group, meta.name, family.map(scope));
        if candidate.raw.trim().is_empty() {
            self.values.remove(&key);
        } else {
            self.values.insert(key, candidate);
        }
        Ok(())
    }

    /// Append to a repeatable field's value (`;`-joined).
    fn append(&mut self, meta: &'static FieldMeta, family: Option<ModelFamily>, candidate: Candidate) -> Result<(), ConfigError> {
        let joined = match self.get(meta, family) {
            Some(existing) => Candidate { raw: format!("{};{}", existing.raw, candidate.raw), spelling: candidate.spelling },
            None => candidate,
        };
        self.set(meta, family, joined)
    }

    /// Whether the layer sets nothing (a patch that names no field).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Every value set, with its field and scope.
    pub fn entries(&self) -> impl Iterator<Item = (&'static FieldMeta, Option<ModelFamily>, &Candidate)> + '_ {
        self.values.iter().map(|((group, name, scope_name), candidate)| {
            let meta = super::schema::field(group, name).expect("a layer only holds declared fields");
            (meta, scope_name.and_then(super::field::family_of_scope), candidate)
        })
    }

    /// `other`'s values over this layer's.
    pub fn overlay(&mut self, other: &Layer) {
        for (key, candidate) in &other.values {
            self.values.insert(*key, candidate.clone());
        }
    }
}

/// Where the config file in use came from (spec config-v2/02 §Config-file
/// auto-discovery): named by `--config`/`IGNIS_CONFIG`, found at one of the
/// conventional paths, or none at all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FileSource {
    /// No config file: flags, env, profile and defaults only.
    #[default]
    None,
    /// The file `--config` / `IGNIS_CONFIG` named.
    Explicit(PathBuf),
    /// The file found at one of the conventional paths.
    Discovered(PathBuf),
}

impl FileSource {
    /// The path, when there is a file.
    pub fn path(&self) -> Option<&std::path::Path> {
        match self {
            FileSource::None => None,
            FileSource::Explicit(path) | FileSource::Discovered(path) => Some(path),
        }
    }

    /// `explicit`, `discovered` or `none`, as `ignis.config.source` logs it.
    pub fn kind(&self) -> &'static str {
        match self {
            FileSource::None => "none",
            FileSource::Explicit(_) => "explicit",
            FileSource::Discovered(_) => "discovered",
        }
    }
}

/// Every source a start gathered, kept so the same options can be resolved
/// again for a family, or with a patch on top.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sources {
    /// Live changes made after start (`PATCH /v1/config`, a switch target's
    /// artifact and id): above everything.
    pub patch: Layer,
    /// The command line's values.
    pub flags: Layer,
    /// The environment's values.
    pub env: Layer,
    /// The config file's values.
    pub file: Layer,
    /// Where the config file came from, if there is one: what a change is
    /// written back to, and what `ignis.config.source` reports.
    pub file_source: FileSource,
    /// The profile's values (spec config-v2/02 §`--profile`): defaults, not
    /// overrides — below the file, above the hardcoded default.
    pub profile: Layer,
    /// The profile's name.
    pub profile_name: String,
}

/// What to do with an explicit value for a field the family being
/// configured cannot take (spec config-v2/01 AC 7-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// A start: the operator named it for this load, so it is refused.
    Start,
    /// A switch: it was named for the model being switched away from, so it
    /// is dropped, and said so.
    Switch,
}

/// The result of resolving every field against a [`Sources`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// Every field's resolved value.
    pub settings: Settings,
    /// Where each field not at its hardcoded default came from.
    pub origins: BTreeMap<(&'static str, &'static str), Origin>,
    /// The flags of the fields a [`Fit::Switch`] dropped, in declaration
    /// order.
    pub dropped: Vec<String>,
}

impl Resolution {
    /// Where `<group>.<name>` came from; `None` is the hardcoded default.
    pub fn origin(&self, group: &str, name: &str) -> Option<&Origin> {
        self.origins.iter().find(|((g, n), _)| *g == group && *n == name).map(|(_, origin)| origin)
    }

    /// Whether the operator wrote `<group>.<name>` for this process
    /// ([`Source::explicit`]) as something other than its default
    /// ([`Origin::at_default`]).
    pub fn explicit(&self, group: &str, name: &str) -> bool {
        self.origin(group, name).is_some_and(|origin| origin.source.explicit() && !origin.at_default)
    }
}

/// Resolve every field of `sources` for `family` (`None`: before any
/// artifact has named one, so only general values and no applicability).
pub fn resolve_settings(sources: &Sources, family: Option<ModelFamily>, fit: Fit) -> Result<Resolution, ConfigError> {
    let resolver = Resolver { sources, family, fit, origins: RefCell::default(), dropped: RefCell::default() };
    let settings = Settings::resolve(&resolver)?;
    Ok(Resolution { settings, origins: resolver.origins.into_inner(), dropped: resolver.dropped.into_inner() })
}

/// Resolves one field at a time against a [`Sources`] — the one function
/// every field's value goes through, called by each group's
/// macro-generated `resolve`.
pub struct Resolver<'a> {
    sources: &'a Sources,
    family: Option<ModelFamily>,
    fit: Fit,
    origins: RefCell<BTreeMap<(&'static str, &'static str), Origin>>,
    dropped: RefCell<Vec<String>>,
}

impl Resolver<'_> {
    /// `meta`'s value: the first candidate in precedence order, parsed and
    /// validated by kind `K`, or `default`. The error names the spelling the
    /// winning candidate arrived under.
    pub fn field<K: FieldKind>(&self, meta: &'static FieldMeta, default: K::Value) -> Result<K::Value, ConfigError> {
        let sources = self.sources;
        let explicit = [
            (&sources.patch, Source::Patch),
            (&sources.flags, Source::Flag),
            (&sources.env, Source::Env),
            (&sources.file, Source::File),
        ];
        for (layer, source) in explicit {
            let Some(candidate) = self.candidate(layer, meta) else {
                continue;
            };
            let value = parse_candidate::<K>(meta, candidate)?;
            let at_default = value == default;
            if let Some(family) = self.family.filter(|family| !meta.applies.to(*family) && !at_default) {
                if self.fit == Fit::Start {
                    return Err(ConfigError(format!(
                        "`{} {}`: {} does not take it (a {} option)",
                        candidate.spelling,
                        candidate.raw.trim(),
                        family.name(),
                        meta.applies.describe()
                    )));
                }
                self.dropped.borrow_mut().push(meta.flag());
                return Ok(default);
            }
            self.record(meta, source, candidate, at_default);
            return Ok(value);
        }
        // A profile is a default, never an override: a value of it the
        // family cannot take is passed over in silence, as a hardcoded
        // default for a field the family ignores would be.
        if self.family.is_none_or(|family| meta.applies.to(family)) {
            if let Some(candidate) = self.candidate(&sources.profile, meta) {
                let value = parse_candidate::<K>(meta, candidate)?;
                let at_default = value == default;
                self.record(meta, Source::Profile, candidate, at_default);
                return Ok(value);
            }
        }
        Ok(default)
    }

    /// `layer`'s value for `meta`: the family-scoped one first, then the
    /// general one.
    fn candidate<'l>(&self, layer: &'l Layer, meta: &FieldMeta) -> Option<&'l Candidate> {
        let scoped = self.family.filter(|family| meta.takes_scope(*family)).and_then(|family| layer.get(meta, Some(family)));
        scoped.or_else(|| layer.get(meta, None))
    }

    fn record(&self, meta: &'static FieldMeta, source: Source, candidate: &Candidate, at_default: bool) {
        self.origins
            .borrow_mut()
            .insert((meta.group, meta.name), Origin { source, spelling: candidate.spelling.clone(), at_default });
    }
}

/// One candidate through `meta`'s validator and kind `K`.
pub fn parse_candidate<K: FieldKind>(meta: &FieldMeta, candidate: &Candidate) -> Result<K::Value, ConfigError> {
    let raw = candidate.raw.trim();
    let refuse = |reason: String| ConfigError(format!("`{}` {reason}", candidate.spelling));
    meta.validator.check_text(raw).map_err(refuse)?;
    let value = K::parse(raw).map_err(refuse)?;
    if let Some(number) = K::number(&value) {
        meta.validator.check_number(number).map_err(refuse)?;
    }
    Ok(value)
}

/// What the command line said, apart from field values.
#[derive(Debug, Default)]
pub struct ParsedArgs {
    /// The field flags, as a layer.
    pub flags: Layer,
    /// `--config <path>`.
    pub config: Option<String>,
    /// `--profile <name>`.
    pub profile: Option<String>,
}

/// Every field flag, general and scoped, by spelling.
fn flag_table() -> &'static BTreeMap<String, (&'static FieldMeta, Option<ModelFamily>)> {
    static TABLE: OnceLock<BTreeMap<String, (&'static FieldMeta, Option<ModelFamily>)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = BTreeMap::new();
        for meta in all_fields() {
            table.insert(meta.flag(), (meta, None));
            if meta.scoped {
                for family in FAMILIES {
                    table.insert(meta.scoped_flag(family), (meta, Some(family)));
                }
            }
        }
        table
    })
}

/// Parse argv (without the program name, and without any subcommand word)
/// into the flag layer. A bool's flag may stand bare (`--server-ui`) or take
/// its value (`--server-ui false`); a repeatable one accumulates; any other
/// given twice keeps the last, as it always has.
pub fn parse_args(args: &[String]) -> Result<ParsedArgs, ConfigError> {
    let table = flag_table();
    let mut parsed = ParsedArgs::default();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--config" => parsed.config = Some(take_value(args, &mut i, arg)?),
            "--profile" => parsed.profile = Some(take_value(args, &mut i, arg)?),
            _ => {
                let Some(&(meta, family)) = table.get(arg) else {
                    return Err(unrecognized(arg));
                };
                let raw = if meta.switch {
                    match args.get(i + 1) {
                        Some(next) if !next.starts_with('-') => {
                            i += 1;
                            next.clone()
                        }
                        _ => "true".to_owned(),
                    }
                } else {
                    take_value(args, &mut i, arg)?
                };
                let candidate = Candidate { raw, spelling: arg.to_owned() };
                if meta.repeatable {
                    parsed.flags.append(meta, family, candidate)?;
                } else {
                    parsed.flags.set(meta, family, candidate)?;
                }
            }
        }
        i += 1;
    }
    Ok(parsed)
}

fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, ConfigError> {
    *i += 1;
    args.get(*i).cloned().ok_or_else(|| ConfigError(format!("`{flag}` requires a value")))
}

/// An unknown flag, with the field flags that end the same way: the
/// regrouping renamed every flag (ADR 0046), and `--max-context` is most
/// likely `--model-max-context`. A hint, not an alias — the old spelling is
/// still refused.
fn unrecognized(arg: &str) -> ConfigError {
    let near: Vec<&str> = match arg.strip_prefix("--") {
        Some(rest) if !rest.is_empty() => flag_table()
            .iter()
            .filter(|(flag, (_, family))| family.is_none() && flag.ends_with(&format!("-{rest}")))
            .map(|(flag, _)| flag.as_str())
            .collect(),
        _ => Vec::new(),
    };
    match near.as_slice() {
        [] => ConfigError(format!("unrecognized flag `{arg}` (`ignis-server help --fields` lists every one)")),
        names => ConfigError(format!(
            "unrecognized flag `{arg}`: did you mean {}? (`ignis-server help --fields` lists every one)",
            names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(" or ")
        )),
    }
}

/// Every field's env var that `env` sets, general and scoped. A scoped
/// variable of a field with no family scope is refused rather than ignored:
/// it can only be a mistake.
pub fn env_layer(env: &dyn Fn(&str) -> Option<String>) -> Result<Layer, ConfigError> {
    let mut layer = Layer::default();
    for meta in all_fields() {
        let name = meta.env();
        if let Some(raw) = env(&name) {
            layer.set(meta, None, Candidate { raw, spelling: name })?;
        }
        for family in FAMILIES {
            let name = meta.scoped_env(family);
            if let Some(raw) = env(&name).filter(|raw| !raw.is_empty()) {
                layer.set(meta, Some(family), Candidate { raw, spelling: name })?;
            }
        }
    }
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::field;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn layer(meta: &'static FieldMeta, values: &[(Option<ModelFamily>, &str)]) -> Layer {
        let mut layer = Layer::default();
        for (family, raw) in values {
            let spelling = match family {
                Some(family) => meta.scoped_flag(*family),
                None => meta.flag(),
            };
            layer.set(meta, *family, Candidate { raw: raw.to_string(), spelling }).unwrap();
        }
        layer
    }

    const GIB: u64 = 1 << 30;

    /// Spec config-v2/01 AC 6 and config-v2/02 §`--profile`, as one table:
    /// for each combination of sources, the value `reuse.kv_host_pool_bytes`
    /// (a family-scoped field) resolves to for the 27B, and where from.
    #[test]
    fn precedence_is_flag_env_file_profile_default_with_the_family_first_within_each() {
        let meta = field("reuse", "kv_host_pool_bytes").unwrap();
        // Constants, so the table's slices are promoted to `'static`.
        const Q: Option<ModelFamily> = Some(ModelFamily::Qwen38_27b);
        const F: Option<ModelFamily> = Some(ModelFamily::FlashNext);
        let q = Q;
        type Row = (&'static str, &'static [(Option<ModelFamily>, &'static str)], &'static [(Option<ModelFamily>, &'static str)], &'static [(Option<ModelFamily>, &'static str)], &'static [(Option<ModelFamily>, &'static str)], u64, Source);
        let none: &[(Option<ModelFamily>, &str)] = &[];
        let rows: Vec<Row> = vec![
            ("default only", none, none, none, none, 2 * GIB, Source::Default),
            ("profile only", none, none, none, &[(None, "3G")], 3 * GIB, Source::Profile),
            ("file over profile", none, none, &[(None, "4G")], &[(None, "3G")], 4 * GIB, Source::File),
            ("env over file", none, &[(None, "5G")], &[(None, "4G")], none, 5 * GIB, Source::Env),
            ("flag over env", &[(None, "6G")], &[(None, "5G")], none, none, 6 * GIB, Source::Flag),
            ("family-flag over flag", &[(None, "6G"), (Q, "7G")], none, none, none, 7 * GIB, Source::Flag),
            ("family-env over env", none, &[(None, "5G"), (Q, "8G")], none, none, 8 * GIB, Source::Env),
            ("family-file over file", none, none, &[(None, "4G"), (Q, "9G")], none, 9 * GIB, Source::File),
            ("family-profile over profile", none, none, none, &[(None, "3G"), (Q, "1G")], GIB, Source::Profile),
            // The command line always wins, whatever its scope.
            ("flag over family-env", &[(None, "6G")], &[(Q, "8G")], none, none, 6 * GIB, Source::Flag),
            ("env over family-file", none, &[(None, "5G")], &[(Q, "9G")], none, 5 * GIB, Source::Env),
            // The other family's value is not this one's.
            ("another family's flag", &[(F, "7G")], none, &[(None, "4G")], none, 4 * GIB, Source::File),
        ];
        for (name, flags, env, file, profile, want, from) in rows {
            let sources = Sources {
                flags: layer(meta, flags),
                env: layer(meta, env),
                file: layer(meta, file),
                profile: layer(meta, profile),
                ..Sources::default()
            };
            let resolved = resolve_settings(&sources, q, Fit::Start).unwrap();
            assert_eq!(resolved.settings.reuse.kv_host_pool_bytes, want, "{name}");
            assert_eq!(resolved.origin("reuse", "kv_host_pool_bytes").map_or(Source::Default, |o| o.source), from, "{name}");
        }
    }

    /// Before an artifact names a family, only the general values count.
    #[test]
    fn with_no_family_yet_every_scoped_value_waits() {
        let meta = field("reuse", "kv_host_pool_bytes").unwrap();
        let sources = Sources { flags: layer(meta, &[(Some(ModelFamily::FlashNext), "7G")]), ..Sources::default() };
        let resolved = resolve_settings(&sources, None, Fit::Start).unwrap();
        assert_eq!(resolved.settings.reuse.kv_host_pool_bytes, 2 * GIB);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.reuse.kv_host_pool_bytes, 7 * GIB);
    }

    /// The winning candidate is the only one parsed: a bad value below a
    /// good one is never reached, and a bad winner is refused naming the
    /// spelling it came under.
    #[test]
    fn only_the_winner_is_parsed_and_its_error_names_its_spelling() {
        let meta = field("model", "prefill_chunk").unwrap();
        let mut env = Layer::default();
        env.set(meta, None, Candidate { raw: "300".into(), spelling: meta.env() }).unwrap();
        let sources = Sources { flags: layer(meta, &[(None, "2048")]), env: env.clone(), ..Sources::default() };
        assert_eq!(resolve_settings(&sources, None, Fit::Start).unwrap().settings.model.prefill_chunk, 2048);
        let err = resolve_settings(&Sources { env, ..Sources::default() }, None, Fit::Start).unwrap_err();
        assert!(err.0.starts_with("`IGNIS_MODEL_PREFILL_CHUNK` ") && err.0.contains("128") && err.0.contains("300"), "{err}");
    }

    /// Spec config-v2/01 AC 7-8, generically: an explicit value for a field
    /// only the other family has is refused at start and dropped, named, on
    /// a switch; with no family yet it is simply taken.
    #[test]
    fn a_field_the_family_cannot_take_is_refused_at_start_and_dropped_on_a_switch() {
        let meta = field("ngram", "hot_bytes").unwrap();
        let sources = Sources { flags: layer(meta, &[(None, "512M")]), ..Sources::default() };
        let qwen = Some(ModelFamily::Qwen38_27b);
        let err = resolve_settings(&sources, qwen, Fit::Start).unwrap_err();
        assert!(err.0.contains("--ngram-hot-bytes 512M") && err.0.contains("Qwen3.8-27B") && err.0.contains("Flash-Next"), "{err}");
        let switched = resolve_settings(&sources, qwen, Fit::Switch).unwrap();
        assert_eq!(switched.settings.ngram.hot_bytes, None);
        assert_eq!(switched.dropped, ["--ngram-hot-bytes"]);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert!(flash.settings.ngram.hot_bytes.is_some() && flash.dropped.is_empty());
        assert!(resolve_settings(&sources, None, Fit::Start).unwrap().settings.ngram.hot_bytes.is_some());
    }

    /// An explicit value equal to the default says nothing: not refused for
    /// a family that cannot take the field, and not "named".
    #[test]
    fn an_explicit_default_is_neither_refused_nor_named() {
        let vision = field("vision", "enabled").unwrap();
        let sources = Sources { file: layer(vision, &[(None, "false")]), ..Sources::default() };
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert!(!flash.settings.vision.enabled && flash.dropped.is_empty());
        assert!(!flash.explicit("vision", "enabled"));
        assert_eq!(flash.origin("vision", "enabled").map(|o| o.source), Some(Source::File), "its origin is still known");
        let on = Sources { file: layer(vision, &[(None, "true")]), ..Sources::default() };
        assert!(resolve_settings(&on, Some(ModelFamily::FlashNext), Fit::Start).is_err());
    }

    /// A profile value is a default: one the family cannot take is passed
    /// over, never refused.
    #[test]
    fn a_profile_value_the_family_cannot_take_is_passed_over() {
        let meta = field("spec", "decode_lanes").unwrap();
        let sources = Sources { profile: layer(meta, &[(None, "3")]), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.spec.decode_lanes, None);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.spec.decode_lanes, Some(3));
        assert!(!flash.explicit("spec", "decode_lanes"), "a profile value is not the operator's");
    }

    /// A field only one family takes may sit in that family's own section —
    /// read there for that family, never for the other — and nowhere else.
    #[test]
    fn a_family_only_field_lives_in_its_familys_section_and_no_other() {
        let lanes = field("spec", "decode_lanes").unwrap();
        let flash = Some(ModelFamily::FlashNext);
        let sources = Sources { file: layer(lanes, &[(flash, "1")]), ..Sources::default() };
        assert_eq!(resolve_settings(&sources, flash, Fit::Start).unwrap().settings.spec.decode_lanes, Some(1));
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.spec.decode_lanes, None, "the 27B never sees it, so never refuses it");
        assert_eq!(resolve_settings(&sources, None, Fit::Start).unwrap().settings.spec.decode_lanes, None);
        let err = Layer::default()
            .set(lanes, Some(ModelFamily::Qwen38_27b), Candidate { raw: "1".into(), spelling: "spec.qwen38.decode_lanes".into() })
            .unwrap_err();
        assert!(err.0.contains("Qwen3.8-27B does not take spec.decode_lanes"), "{err}");
    }

    #[test]
    fn a_scoped_value_for_an_unscoped_or_inapplicable_field_is_refused_when_written() {
        let unscoped = field("server", "bind").unwrap();
        let err = Layer::default()
            .set(unscoped, Some(ModelFamily::FlashNext), Candidate { raw: "x".into(), spelling: "IGNIS_QWEN38FLASHNEXT_SERVER_BIND".into() })
            .unwrap_err();
        assert!(err.0.contains("server.bind") && err.0.contains("per-family"), "{err}");
        let env = |name: &str| (name == "IGNIS_QWEN38_SERVER_BIND").then(|| "x".to_owned());
        assert!(env_layer(&env).is_err());
    }

    #[test]
    fn flags_take_values_bools_stand_bare_and_repeatables_accumulate() {
        let parsed = parse_args(&args(&[
            "--server-ui",
            "--server-bind",
            "b",
            "--vision-enabled",
            "false",
            "--switch-known-models",
            "a=F:/a",
            "--switch-known-models",
            "b=F:/b",
            "--qwen38flashnext-reuse-kv-host-pool-bytes",
            "1G",
            "--profile",
            "rtx5090",
            "--config",
            "c.yaml",
        ]))
        .unwrap();
        let get = |group, name, family| parsed.flags.get(field(group, name).unwrap(), family).map(|c| c.raw.as_str());
        assert_eq!(get("server", "ui", None), Some("true"));
        assert_eq!(get("server", "bind", None), Some("b"));
        assert_eq!(get("vision", "enabled", None), Some("false"));
        assert_eq!(get("switch", "known_models", None), Some("a=F:/a;b=F:/b"));
        assert_eq!(get("reuse", "kv_host_pool_bytes", Some(ModelFamily::FlashNext)), Some("1G"));
        assert_eq!(parsed.profile.as_deref(), Some("rtx5090"));
        assert_eq!(parsed.config.as_deref(), Some("c.yaml"));
    }

    #[test]
    fn an_old_flag_is_refused_with_its_new_spelling_as_a_hint() {
        let err = parse_args(&args(&["--max-context", "8192"])).unwrap_err();
        assert!(err.0.contains("`--max-context`") && err.0.contains("`--model-max-context`"), "{err}");
        let err = parse_args(&args(&["--nope"])).unwrap_err();
        assert!(err.0.contains("`--nope`") && !err.0.contains("did you mean"), "{err}");
        let err = parse_args(&args(&["--server-bind"])).unwrap_err();
        assert!(err.0.contains("`--server-bind` requires a value"), "{err}");
        assert!(parse_args(&args(&["--qwen38-server-bind", "x"])).is_err(), "an unscoped field has no scoped flag");
    }

    #[test]
    fn the_environment_is_read_by_every_fields_name_and_empty_is_unset() {
        let env = |name: &str| match name {
            "IGNIS_MODEL_MAX_CONTEXT" => Some("16384".to_owned()),
            "IGNIS_QWEN38_MODEL_MAX_CONTEXT" => Some("8192".to_owned()),
            "IGNIS_SERVER_BIND" => Some(String::new()),
            _ => None,
        };
        let layer = env_layer(&env).unwrap();
        let max_context = field("model", "max_context").unwrap();
        assert_eq!(layer.get(max_context, None).unwrap().raw, "16384");
        assert_eq!(layer.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().spelling, "IGNIS_QWEN38_MODEL_MAX_CONTEXT");
        assert!(layer.get(field("server", "bind").unwrap(), None).is_none());
    }

    #[test]
    fn a_layer_never_prints_a_value() {
        let meta = field("server", "api_key").unwrap();
        let layer = layer(meta, &[(None, "sk-secret")]);
        let shown = format!("{layer:?}");
        assert!(shown.contains("--server-api-key") && !shown.contains("sk-secret"), "{shown}");
    }
}
