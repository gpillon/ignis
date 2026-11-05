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
//! A patch is a change to the config file, so it keeps the file's own
//! tiebreak: its general value yields to the file's `<family>-config` for
//! that family, as in the rewritten file ([`Resolver::field`]).
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
    /// The card, which decided a value no operator named: the band it runs
    /// where the field's default or a profile's value is one it cannot (GitHub
    /// #477). Not explicit, so a load's adaptation may still lower it.
    Card,
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
    /// The value was given for the family resolved for, not the group: a
    /// family section of a file or a profile, a `--<family>-…` flag, an
    /// `IGNIS_<FAMILY>_…` env var, a live family value. What `GET
    /// /v1/config/origins` tells apart (spec playground/05 follow-up 2): a
    /// family's own value in a file or a patch is the operator's, which a
    /// group value does not reach and a `null` removes.
    pub scoped: bool,
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

/// One source's raw values, general and per family — and, for a live
/// patch only, the keys it removes.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Layer {
    values: BTreeMap<Key, Candidate>,
    /// The keys a patch's `null` removes, each with the spelling it was
    /// given under (spec playground/05 §Decisions). Never kept by a source:
    /// [`Layer::overlay`] and [`super::Config::general_with`] apply them.
    removed: BTreeMap<Key, String>,
}

impl std::fmt::Debug for Layer {
    /// The spellings set or removed, never their values: a layer holds the
    /// API key's raw text, and a `Config` dumped into a log line must not
    /// leak it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.values.values().map(|candidate| &candidate.spelling).chain(self.removed.values())).finish()
    }
}

/// The field a layer's key names, and its scope.
fn of_key((group, name, scope_name): &Key) -> (&'static FieldMeta, Option<ModelFamily>) {
    let meta = super::schema::field(group, name).expect("a layer only holds declared fields");
    (meta, scope_name.and_then(super::field::family_of_scope))
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
        check_scope(meta, family, &candidate.spelling)?;
        let key = (meta.group, meta.name, family.map(scope));
        self.removed.remove(&key);
        if candidate.raw.trim().is_empty() {
            self.values.remove(&key);
        } else {
            self.values.insert(key, candidate);
        }
        Ok(())
    }

    /// Remove `meta`'s value in `family`'s scope — a live patch's `null`
    /// (spec playground/05 §Decisions, JSON merge patch): overlaid on the
    /// patches before it, the key's value is gone from them, and the config
    /// file loses the key too. Its key is checked as a value's would be: a
    /// `null` where no value can be is as wrong as a value there.
    pub fn remove(&mut self, meta: &'static FieldMeta, family: Option<ModelFamily>, spelling: String) -> Result<(), ConfigError> {
        check_scope(meta, family, &spelling)?;
        let key = (meta.group, meta.name, family.map(scope));
        self.values.remove(&key);
        self.removed.insert(key, spelling);
        Ok(())
    }

    /// Take `meta`'s value in `family`'s scope out of this layer, if it has
    /// one.
    pub fn unset(&mut self, meta: &FieldMeta, family: Option<ModelFamily>) {
        self.values.remove(&(meta.group, meta.name, family.map(scope)));
    }

    /// Append to a repeatable field's value (`;`-joined).
    fn append(&mut self, meta: &'static FieldMeta, family: Option<ModelFamily>, candidate: Candidate) -> Result<(), ConfigError> {
        let joined = match self.get(meta, family) {
            Some(existing) => Candidate { raw: format!("{};{}", existing.raw, candidate.raw), spelling: candidate.spelling },
            None => candidate,
        };
        self.set(meta, family, joined)
    }

    /// Whether the layer names nothing, set or removed (a patch that names no
    /// field).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty() && self.removed.is_empty()
    }

    /// Every value set, with its field and scope.
    pub fn entries(&self) -> impl Iterator<Item = (&'static FieldMeta, Option<ModelFamily>, &Candidate)> + '_ {
        self.values.iter().map(|(key, candidate)| {
            let (meta, family) = of_key(key);
            (meta, family, candidate)
        })
    }

    /// Every key removed, with its field, scope and spelling.
    pub fn removals(&self) -> impl Iterator<Item = (&'static FieldMeta, Option<ModelFamily>, &str)> + '_ {
        self.removed.iter().map(|(key, spelling)| {
            let (meta, family) = of_key(key);
            (meta, family, spelling.as_str())
        })
    }

    /// Every key the layer names, set or removed, in key order.
    pub fn named(&self) -> impl Iterator<Item = (&'static FieldMeta, Option<ModelFamily>)> + '_ {
        let keys: std::collections::BTreeSet<&Key> = self.values.keys().chain(self.removed.keys()).collect();
        keys.into_iter().map(of_key)
    }

    /// This layer with every value and removal of a field only one family
    /// takes moved from the group's key to that family's own section
    /// ([`FieldMeta::home_family`]) — where a live patch puts it, in the
    /// running configuration as in the config file (spec playground/05
    /// follow-up 2, "a removal goes where a value goes"). Such a field means
    /// something to that family alone, so its group key and its family key
    /// are one value; homing it is what keeps a patched value from yielding
    /// to the file's section it is written over, and a removal at either key
    /// from leaving the other behind. A key the layer already names in that
    /// family's section is the more specific and stays.
    pub fn homed(&self) -> Layer {
        let mut homed = self.clone();
        let general = self.values.keys().chain(self.removed.keys()).filter(|(_, _, scope_name)| scope_name.is_none());
        for key in general {
            let Some(family) = of_key(key).0.home_family() else {
                continue;
            };
            let home = (key.0, key.1, Some(scope(family)));
            let named_there = self.values.contains_key(&home) || self.removed.contains_key(&home);
            if let Some(candidate) = homed.values.remove(key).filter(|_| !named_there) {
                homed.values.insert(home, candidate);
            }
            if let Some(spelling) = homed.removed.remove(key).filter(|_| !named_there) {
                homed.removed.insert(home, spelling);
            }
        }
        homed
    }

    /// `other`'s values over this layer's, and its removals taken out of it.
    pub fn overlay(&mut self, other: &Layer) {
        for (key, candidate) in &other.values {
            self.values.insert(*key, candidate.clone());
        }
        for key in other.removed.keys() {
            self.values.remove(key);
        }
    }
}

/// Refuse a value — or a removal — of `meta` in `family`'s section when the
/// family cannot take the field, or the field has no per-family value
/// ([`Layer::set`]).
fn check_scope(meta: &FieldMeta, family: Option<ModelFamily>, spelling: &str) -> Result<(), ConfigError> {
    let Some(family) = family else {
        return Ok(());
    };
    if !meta.applies.to(family) {
        return Err(ConfigError(format!(
            "`{spelling}`: {} does not take {} (a {} option)",
            family.name(),
            meta.file_key(),
            meta.applies.describe()
        )));
    }
    if !meta.takes_scope(family) {
        return Err(ConfigError(format!(
            "`{spelling}`: {} has no per-family value; set `{}` instead",
            meta.file_key(),
            meta.file_key()
        )));
    }
    Ok(())
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
/// again for a family, or with a patch on top: the layers, and the one
/// hardware reading the profile was chosen by (`card`).
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
    /// The profile's name — the one `auto` chose, never `auto` itself (spec
    /// 27b-16gb/04): what `GET /v1/config` and the Playground show.
    pub profile_name: String,
    /// The profile was chosen by `--profile auto` (the default) from the
    /// card's total, not named.
    pub profile_auto: bool,
    /// The card, when the start read it (`auto` reads it to choose; a named
    /// profile reads it only at a start): its total memory, name and compute
    /// capability, and what the build's kernels were compiled for; `None`
    /// when it was not read.
    pub card: Option<super::profile::CardReading>,
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

    /// The spelling `<group>.<name>`'s value came under, for an error that
    /// names it ([`Origin::spelling`]); its flag when it is at its default.
    pub fn spelled(&self, group: &str, name: &str) -> String {
        self.origin(group, name)
            .map(|origin| origin.spelling.clone())
            .unwrap_or_else(|| super::schema::field(group, name).expect("a declared field").flag())
    }

    /// Whether the operator wrote `<group>.<name>` for this process
    /// ([`Source::explicit`]) as something other than its default
    /// ([`Origin::at_default`]).
    pub fn explicit(&self, group: &str, name: &str) -> bool {
        self.origin(group, name).is_some_and(|origin| origin.source.explicit() && !origin.at_default)
    }

    /// Whether `<group>.<name>`'s value is the profile's or the hardcoded
    /// default's: what a load's **adaptation** may lower (ADR 0063 §3). By
    /// source alone -- a value the operator wrote never adapts, not even one
    /// equal to the default, since a file `config generate` wrote names the
    /// numbers of the card it was generated on.
    pub fn adapts(&self, group: &str, name: &str) -> bool {
        !self.origin(group, name).is_some_and(|origin| origin.source.explicit())
    }
}

/// Resolve every field of `sources` for `family` (`None`: before any
/// artifact has named one, so only general values and no applicability).
pub fn resolve_settings(sources: &Sources, family: Option<ModelFamily>, fit: Fit) -> Result<Resolution, ConfigError> {
    let resolver = Resolver { sources, family, fit, origins: RefCell::default(), dropped: RefCell::default() };
    let mut settings = Settings::resolve(&resolver)?;
    let mut origins = resolver.origins.into_inner();
    decide_by_card(sources, &mut settings, &mut origins);
    Ok(Resolution { settings, origins, dropped: resolver.dropped.into_inner() })
}

/// The values the card and the build decide (GitHub #477, #505, #510, #515):
/// the report says what the load runs. On an 8.x card the 27B's prompt
/// attention is the integer band wherever the 8-bit band resolved, its
/// trellis prefill the dense prefill when the field is left to its default;
/// on any card a value the build cannot run, from a default or a profile,
/// falls back ([`super::arch::prefill_attention`],
/// [`super::arch::prefill_linear`]). The origin of a value so decided is the
/// card, unless the operator named it. A named value the load cannot run is
/// refused where the config is built.
fn decide_by_card(sources: &Sources, settings: &mut Settings, origins: &mut BTreeMap<(&'static str, &'static str), Origin>) {
    use super::arch::{self, GivenBy};
    let card = sources.card.as_ref();
    let spelling = card.map_or_else(|| "this build".to_owned(), |card| format!("{} ({})", card.name, card.capability));
    let attention = &mut settings.model.prefill_attention;
    decided_by(origins, ("model", "prefill_attention"), attention, &spelling, |value, given| arch::prefill_attention(value, given, card, ""));
    let linear = &mut settings.model.prefill_linear;
    decided_by(origins, ("model", "prefill_linear"), linear, &spelling, |value, given| arch::prefill_linear(value, given, card, ""));

    /// `value`, at `key`, replaced by what `decide` makes of it for its giver;
    /// a value the operator did not name then has the card as its origin. A
    /// refusal is left to the config.
    fn decided_by<T: PartialEq + Copy>(
        origins: &mut BTreeMap<(&'static str, &'static str), Origin>,
        key: (&'static str, &'static str),
        value: &mut T,
        spelling: &str,
        decide: impl FnOnce(T, GivenBy) -> Result<T, String>,
    ) {
        let given = GivenBy::of(origins.get(&key));
        let Ok(decided) = decide(*value, given) else { return };
        if decided == *value {
            return;
        }
        *value = decided;
        if given != GivenBy::Operator {
            origins.insert(key, Origin { source: Source::Card, spelling: spelling.to_owned(), at_default: false, scoped: false });
        }
    }
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
            let found = match source {
                Source::Patch => self.patch_candidate(meta),
                _ => self.candidate(layer, meta),
            };
            let Some((candidate, scoped)) = found else {
                continue;
            };
            let value = parse_candidate::<K>(meta, candidate)?;
            let at_default = value == default;
            if let Some(family) = self.family.filter(|family| meta.applies.refused_by(*family) && !at_default) {
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
            self.record(meta, source, candidate, scoped, at_default);
            return Ok(value);
        }
        // A profile is a default, never an override: a value of it the
        // family cannot take is passed over in silence, as a hardcoded
        // default for a field the family ignores would be.
        if self.family.is_none_or(|family| !meta.applies.refused_by(family)) {
            if let Some((candidate, scoped)) = self.candidate(&sources.profile, meta) {
                let value = parse_candidate::<K>(meta, candidate)?;
                let at_default = value == default;
                self.record(meta, Source::Profile, candidate, scoped, at_default);
                return Ok(value);
            }
        }
        Ok(default)
    }

    /// `layer`'s value for `meta`: the family-scoped one first, then the
    /// general one — and whether it was the family-scoped one.
    fn candidate<'l>(&self, layer: &'l Layer, meta: &FieldMeta) -> Option<(&'l Candidate, bool)> {
        let scoped = self.family.filter(|family| meta.takes_scope(*family)).and_then(|family| layer.get(meta, Some(family)));
        match scoped {
            Some(candidate) => Some((candidate, true)),
            None => layer.get(meta, None).map(|candidate| (candidate, false)),
        }
    }

    /// The live patch's value for `meta`: as [`Resolver::candidate`] picks
    /// it, except that its general value yields to a value the config file
    /// holds in the family's own section. A patch is a change to the config
    /// file (written into it once it takes), applied ahead of every other
    /// source while the process runs; in the file a family's own section
    /// outranks the group's value, so a patched group value reaching a
    /// family the file gives its own would run a value no start from the
    /// rewritten file runs (spec playground/05 §Decisions). Whatever
    /// outranks the file's own section then decides, as at a start.
    fn patch_candidate(&self, meta: &FieldMeta) -> Option<(&Candidate, bool)> {
        let (patch, file) = (&self.sources.patch, &self.sources.file);
        if let Some(family) = self.family.filter(|family| meta.takes_scope(*family)) {
            if let Some(scoped) = patch.get(meta, Some(family)) {
                return Some((scoped, true));
            }
            if file.get(meta, Some(family)).is_some() {
                return None;
            }
        }
        patch.get(meta, None).map(|candidate| (candidate, false))
    }

    fn record(&self, meta: &'static FieldMeta, source: Source, candidate: &Candidate, scoped: bool, at_default: bool) {
        self.origins
            .borrow_mut()
            .insert((meta.group, meta.name), Origin { source, spelling: candidate.spelling.clone(), at_default, scoped });
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

/// Split argv into the words that are not a flag nor a flag's value, in the
/// order given, and the rest, which [`parse_args`] reads. It steps exactly as
/// `parse_args` does: a flag that takes a value owns the word after it, a
/// bool flag owns the next word only when that does not start with `-`, so
/// `--download-path D:/m qwen3.8-flash-next` leaves the id for the caller
/// and a bool flag directly before an id would claim it as its value.
pub fn split_positionals(args: &[String]) -> (Vec<String>, Vec<String>) {
    let table = flag_table();
    let (mut positionals, mut rest) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            positionals.push(arg.clone());
            i += 1;
            continue;
        }
        let owns_next = match arg.as_str() {
            "--config" | "--profile" => true,
            flag => match table.get(flag) {
                Some(&(meta, _)) if meta.switch => args.get(i + 1).is_some_and(|next| !next.starts_with('-')),
                Some(_) => true,
                None => false,
            },
        };
        rest.push(arg.clone());
        if owns_next {
            rest.extend(args.get(i + 1).cloned());
            i += 1;
        }
        i += 1;
    }
    (positionals, rest)
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

    /// Spec playground/05 follow-up 2: an origin says whether the value that
    /// won was the family's own or the group's, whatever the source — and a
    /// patched group value that yields to the file's own section leaves the
    /// file's, scoped, as the origin.
    #[test]
    fn an_origin_says_whether_the_familys_own_value_won() {
        let meta = field("reuse", "kv_host_pool_bytes").unwrap();
        let (q, f) = (Some(ModelFamily::Qwen38_27b), Some(ModelFamily::FlashNext));
        let origin = |sources: &Sources, family| {
            let resolved = resolve_settings(sources, family, Fit::Start).unwrap();
            resolved.origin("reuse", "kv_host_pool_bytes").map(|origin| (origin.source, origin.scoped))
        };
        let profile = Sources { profile: layer(meta, &[(None, "3G"), (q, "1G")]), ..Sources::default() };
        assert_eq!((origin(&profile, q), origin(&profile, f), origin(&profile, None)), (Some((Source::Profile, true)), Some((Source::Profile, false)), Some((Source::Profile, false))));
        let flags = Sources { flags: layer(meta, &[(f, "7G")]), ..profile.clone() };
        assert_eq!((origin(&flags, q), origin(&flags, f)), (Some((Source::Profile, true)), Some((Source::Flag, true))));
        let patched = Sources { patch: layer(meta, &[(None, "5G")]), file: layer(meta, &[(q, "4G")]), ..Sources::default() };
        assert_eq!((origin(&patched, q), origin(&patched, f)), (Some((Source::File, true)), Some((Source::Patch, false))));
        let own = Sources { patch: layer(meta, &[(q, "6G")]), ..patched.clone() };
        assert_eq!(origin(&own, q), Some((Source::Patch, true)));
        assert_eq!(origin(&Sources::default(), q), None, "the hardcoded default has no origin");
    }

    /// Spec playground/05 §Decisions: a live patch is a change to the config
    /// file, applied ahead of every other source — and in the file a
    /// family's own section outranks the group's value, so a patched group
    /// value does not reach a family the file gives its own. The families
    /// without one take it, and a patched family value still wins.
    #[test]
    fn a_patched_group_value_yields_to_the_files_own_section_for_that_family() {
        let meta = field("reuse", "kv_host_pool_bytes").unwrap();
        let (q, f) = (Some(ModelFamily::Qwen38_27b), Some(ModelFamily::FlashNext));
        let resolved = |sources: &Sources, family| resolve_settings(sources, family, Fit::Start).unwrap().settings.reuse.kv_host_pool_bytes;
        let sources = Sources { patch: layer(meta, &[(None, "3G")]), file: layer(meta, &[(q, "4G")]), ..Sources::default() };
        assert_eq!((resolved(&sources, q), resolved(&sources, f), resolved(&sources, None)), (4 * GIB, 3 * GIB, 3 * GIB));
        let own = Sources { patch: layer(meta, &[(None, "3G"), (q, "1G")]), ..sources.clone() };
        assert_eq!(resolved(&own, q), GIB, "the patch's own value for the family");
        let flagged = Sources { flags: layer(meta, &[(None, "6G")]), ..sources.clone() };
        assert_eq!(resolved(&flagged, q), 6 * GIB, "what outranks the file's own section still does");
        let scoped_flag = Sources { patch: layer(meta, &[(None, "3G")]), flags: layer(meta, &[(q, "5G")]), ..Sources::default() };
        assert_eq!(resolved(&scoped_flag, q), 3 * GIB, "with nothing in the file a patch still outranks the command line");
    }

    /// Spec playground/05 §Decisions: a patch's `null` is a removal — it
    /// names its key, checked as a value there would be, sets nothing, and
    /// overlaid on the patches before it takes their value for that key away.
    #[test]
    fn a_removal_names_its_key_sets_nothing_and_takes_an_earlier_patchs_value_away() {
        let meta = field("reuse", "kv_host_pool_bytes").unwrap();
        let q = Some(ModelFamily::Qwen38_27b);
        let mut removal = Layer::default();
        removal.remove(meta, q, "reuse.qwen38.kv_host_pool_bytes".into()).unwrap();
        assert!(!removal.is_empty() && removal.entries().next().is_none(), "a body of only removals is a patch");
        let named: Vec<_> = removal.named().map(|(meta, family)| (meta.file_key(), family)).collect();
        assert_eq!(named, [("reuse.kv_host_pool_bytes".to_owned(), q)]);
        let mut earlier = layer(meta, &[(None, "3G"), (q, "1G")]);
        earlier.overlay(&removal);
        assert!(earlier.get(meta, q).is_none() && earlier.get(meta, None).is_some(), "only the key removed: {earlier:?}");
        assert_eq!(earlier.removals().count(), 0, "a removal is applied, never kept");
        let err = Layer::default().remove(field("server", "bind").unwrap(), q, "server.qwen38.bind".into()).unwrap_err();
        assert!(err.0.contains("server.qwen38.bind") && err.0.contains("per-family"), "{err}");
    }

    /// Spec playground/05 follow-up 2: a field only one family takes is one
    /// value at its group key and in its family's section, so homing moves
    /// its value and its removal there, leaves every other field where it
    /// was named, and lets a key named in the section win over the group's.
    #[test]
    fn homing_moves_a_family_only_fields_value_and_removal_to_its_section() {
        let rows = field("spec", "draft_rows").unwrap();
        let pool = field("reuse", "kv_host_pool_bytes").unwrap();
        let flash = Some(ModelFamily::FlashNext);
        let mut patch = Layer::default();
        patch.set(rows, None, Candidate { raw: "2".into(), spelling: "spec.draft_rows".into() }).unwrap();
        patch.set(pool, None, Candidate { raw: "3G".into(), spelling: "reuse.kv_host_pool_bytes".into() }).unwrap();
        let homed = patch.homed();
        let raw = |layer: &Layer, meta, family| layer.get(meta, family).map(|candidate| candidate.raw.clone());
        assert_eq!((raw(&homed, rows, None), raw(&homed, rows, flash)), (None, Some("2".to_owned())));
        assert_eq!(homed.get(pool, None).map(|c| c.raw.as_str()), Some("3G"), "a field both families take stays at its group key");

        let mut removal = Layer::default();
        removal.remove(rows, None, "spec.draft_rows".into()).unwrap();
        let named: Vec<_> = removal.homed().named().map(|(meta, family)| (meta.file_key(), family)).collect();
        assert_eq!(named, [("spec.draft_rows".to_owned(), flash)]);

        let mut both = Layer::default();
        both.set(rows, None, Candidate { raw: "2".into(), spelling: "spec.draft_rows".into() }).unwrap();
        both.remove(rows, flash, "spec.qwen38flashnext.draft_rows".into()).unwrap();
        let homed = both.homed();
        assert_eq!((homed.entries().count(), homed.removals().count()), (0, 1), "the section's own key wins: {homed:?}");
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
        let meta = field("spec", "draft_rows").unwrap();
        let sources = Sources { profile: layer(meta, &[(None, "6")]), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.spec.draft_rows, None);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.spec.draft_rows, Some(6));
        assert!(!flash.explicit("spec", "draft_rows"), "a profile value is not the operator's");
    }

    /// GitHub #336: the decode lanes are both models', so a profile's
    /// general value would be the 27B's too -- which is why the built-in
    /// profiles scope theirs -- and each model's section is its own.
    #[test]
    fn the_decode_lanes_are_each_models_own_where_scoped_and_both_where_not() {
        let lanes = field("spec", "decode_lanes").unwrap();
        let (qwen, flash) = (Some(ModelFamily::Qwen38_27b), Some(ModelFamily::FlashNext));
        let lanes_of = |sources: &Sources, family| resolve_settings(sources, family, Fit::Start).unwrap().settings.spec.decode_lanes;
        let general = Sources { profile: layer(lanes, &[(None, "3")]), ..Sources::default() };
        assert_eq!((lanes_of(&general, qwen), lanes_of(&general, flash)), (Some(3), Some(3)));
        let scoped = Sources { profile: layer(lanes, &[(flash, "3"), (qwen, "2")]), ..Sources::default() };
        assert_eq!((lanes_of(&scoped, qwen), lanes_of(&scoped, flash), lanes_of(&scoped, None)), (Some(2), Some(3), None));
        let file = Sources { file: layer(lanes, &[(flash, "1")]), ..Sources::default() };
        assert_eq!((lanes_of(&file, qwen), lanes_of(&file, flash)), (None, Some(1)));
    }

    /// A field only one family takes may sit in that family's own section —
    /// read there for that family, never for the other — and nowhere else.
    #[test]
    fn a_family_only_field_lives_in_its_familys_section_and_no_other() {
        let rows = field("spec", "draft_rows").unwrap();
        let flash = Some(ModelFamily::FlashNext);
        let sources = Sources { file: layer(rows, &[(flash, "6")]), ..Sources::default() };
        assert_eq!(resolve_settings(&sources, flash, Fit::Start).unwrap().settings.spec.draft_rows, Some(6));
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.spec.draft_rows, None, "the 27B never sees it, so never refuses it");
        assert_eq!(resolve_settings(&sources, None, Fit::Start).unwrap().settings.spec.draft_rows, None);
        let err = Layer::default()
            .set(rows, Some(ModelFamily::Qwen38_27b), Candidate { raw: "6".into(), spelling: "spec.qwen38.draft_rows".into() })
            .unwrap_err();
        assert!(err.0.contains("Qwen3.8-27B does not take spec.draft_rows"), "{err}");
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
