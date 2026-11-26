//! What one config field is, apart from its value (ADR 0046, spec
//! config-v2/01 §The field declaration): the [`FieldMeta`] every field's
//! single declaration in `schema.rs` produces, the [`Validator`] and
//! [`Applicability`] it carries, and the two model-family scopes a field may
//! be overridden in.
//!
//! Nothing here knows any particular field. The names a field is spelled by
//! on each surface — `--<group>-<field>`, `IGNIS_<GROUP>_<FIELD>`,
//! `<group>.<field>` — are computed from its group and name by the methods
//! below rather than written down per field, so the three surfaces cannot
//! disagree about a field's name: there is only one name.

use ignis_core::compute::{ModelFamily, ModelKind};
use serde_json::Value;

/// Every model family a config value can be scoped to, in the order
/// `help --fields` and the scoped-name tables list them.
pub const FAMILIES: [ModelFamily; 2] = [ModelFamily::Qwen38_27b, ModelFamily::FlashNext];

/// The spelling of `family` as a config scope (spec config-v2/01 §Family
/// scope): `qwen38` / `qwen38flashnext` in a flag or a file key,
/// `QWEN38` / `QWEN38FLASHNEXT` in an env var.
///
/// Tied to [`ModelFamily`] by an exhaustive match rather than to a served
/// model id: a family is one of a fixed few plain identifiers, while a model
/// id (`qwen3.8-flash-next`) holds characters an env var cannot, and a third
/// family added to the enum fails to compile here until it is given a scope.
pub fn scope(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen38_27b => "qwen38",
        ModelFamily::FlashNext => "qwen38flashnext",
        // Not in FAMILIES yet: spec image-model/05 gives the image model its
        // config group and decides which fields it takes.
        ModelFamily::QwenImage21 => "qwenimage21",
        // Not in FAMILIES: the `diarizer` group's own fields are its, under
        // their plain keys (spec speech/08), and no text model's section
        // holds one.
        ModelFamily::Nemotron3Diarization => "nemotron3diarization",
    }
}

/// The family a scope spelling names (`qwen38` → the 27B), case-insensitive
/// so the env var's upper-case spelling reads back too.
pub fn family_of_scope(name: &str) -> Option<ModelFamily> {
    FAMILIES.into_iter().find(|family| scope(*family).eq_ignore_ascii_case(name))
}

/// How a field's parsed value is checked (spec config-v2/01 AC 9): one of a
/// small typed set, so "a nonzero multiple of 128" is stated as what it is
/// instead of as a pattern over the raw text.
///
/// There is no `Regex` variant although the spec names one: no field today
/// is a string with a pattern to match, and the variant would bring the
/// `regex` crate in as a third dependency of this crate for nothing. It is a
/// one-variant addition the day a field needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validator {
    /// Anything the field's kind parses is accepted.
    None,
    /// An inclusive numeric range on the parsed value; either end open.
    Range { min: Option<i64>, max: Option<i64> },
    /// An inclusive range on a decimal value ([`super::kind::Decimal`]),
    /// both ends in the thousandths it is kept as, and stated in its own
    /// units (`0..=1`).
    DecimalRange { min: i64, max: i64 },
    /// A **nonzero** multiple of the given step (`--model-prefill-chunk`'s
    /// 128). Zero is refused because the one field that uses this
    /// (`--model-prefill-chunk`) has never accepted it, and a chunk of no
    /// tokens is a typo, not a request.
    MultipleOf(u64),
    /// One of a fixed set of spellings, compared trimmed and
    /// case-insensitively against the raw text before the kind parses it —
    /// so a refusal always lists every value the field takes.
    OneOf(&'static [&'static str]),
}

impl Validator {
    /// Check a parsed numeric value. The message continues "`<spelling>` …".
    pub fn check_number(&self, value: i64) -> Result<(), String> {
        match *self {
            Validator::Range { min, max } => {
                let below = min.is_some_and(|min| value < min);
                let above = max.is_some_and(|max| value > max);
                if !below && !above {
                    return Ok(());
                }
                Err(match (min, max) {
                    (Some(min), Some(max)) => format!("must be in {min}..={max}, got {value}"),
                    (Some(min), None) => format!("must be at least {min}, got {value}"),
                    (None, Some(max)) => format!("must be at most {max}, got {value}"),
                    (None, None) => unreachable!("an open range refuses nothing"),
                })
            }
            Validator::MultipleOf(step) => {
                let step = step as i64;
                if value != 0 && value % step == 0 {
                    Ok(())
                } else {
                    Err(format!("must be a nonzero multiple of {step}, got {value}"))
                }
            }
            Validator::DecimalRange { min, max } => match (min..=max).contains(&value) {
                true => Ok(()),
                false => Err(format!("must be in {}..={}, got {}", decimal(min), decimal(max), decimal(value))),
            },
            Validator::None | Validator::OneOf(_) => Ok(()),
        }
    }

    /// Check the raw text, before parsing (only [`Validator::OneOf`] looks at
    /// it). The message continues "`<spelling>` …".
    pub fn check_text(&self, raw: &str) -> Result<(), String> {
        let Validator::OneOf(allowed) = *self else {
            return Ok(());
        };
        let wanted = raw.trim();
        if allowed.iter().any(|name| name.eq_ignore_ascii_case(wanted)) {
            Ok(())
        } else {
            Err(format!("must be one of {}, got `{raw}`", allowed.join(", ")))
        }
    }

    /// The rule as `help --fields` prints it; empty for [`Validator::None`].
    pub fn describe(&self) -> String {
        match *self {
            Validator::None => String::new(),
            Validator::Range { min: Some(min), max: Some(max) } => format!("{min}..={max}"),
            Validator::Range { min: Some(min), max: None } => format!(">= {min}"),
            Validator::Range { min: None, max: Some(max) } => format!("<= {max}"),
            Validator::Range { min: None, max: None } => String::new(),
            Validator::MultipleOf(step) => format!("nonzero multiple of {step}"),
            Validator::DecimalRange { min, max } => format!("{}..={}", decimal(min), decimal(max)),
            Validator::OneOf(allowed) => format!("one of {}", allowed.join(", ")),
        }
    }

    /// The rule as a tool reads it (`GET /v1/config/fields`, spec
    /// playground/05): `null` for [`Validator::None`], else the one variant's
    /// own shape — `{ "min", "max" }` (an open end is `null`),
    /// `{ "multiple_of" }` or `{ "one_of": [..] }`. [`Validator::describe`] is
    /// the same rule as the sentence a person reads.
    ///
    /// A [`Validator::OneOf`] lists the spellings the field *accepts*, which
    /// may name one value twice (`hq` and `hq-e8-2b`); `canonical` is the
    /// field's own ([`FieldMeta::canonical`]), and each value is listed once,
    /// as the config writes it, in the order first seen.
    pub fn to_json(&self, canonical: fn(&str) -> Result<Value, String>) -> Value {
        match *self {
            Validator::None => Value::Null,
            Validator::Range { min, max } => serde_json::json!({ "min": min, "max": max }),
            Validator::MultipleOf(step) => serde_json::json!({ "multiple_of": step }),
            Validator::DecimalRange { min, max } => serde_json::json!({ "min": min as f64 / 1000.0, "max": max as f64 / 1000.0 }),
            Validator::OneOf(allowed) => {
                let mut values: Vec<String> = Vec::new();
                for spelling in allowed {
                    let written = match canonical(spelling) {
                        Ok(Value::String(written)) => written,
                        _ => (*spelling).to_owned(),
                    };
                    if !values.contains(&written) {
                        values.push(written);
                    }
                }
                serde_json::json!({ "one_of": values })
            }
        }
    }
}

/// Which model families a field means anything for (spec config-v2/01
/// §`applicable_to`).
///
/// A field only one family has at all — Flash-Next's draft row budget, the
/// 27B's vision tower — is `Only` that family. An explicit value for it
/// while the other family is the one being configured is refused at start
/// (the operator named it for this load, so it is a mistake made now) and
/// dropped with a log line on a switch (it was named for the model being
/// switched away from, and was never wrong). This replaces the drop list
/// `fit_to_family` used to keep by hand.
///
/// The family configured is the active model's, a text model or the image
/// model, the one a switch replaces. A field only the diarizer takes (spec
/// speech/08) is the diarizer's family's to judge, not the active model's: a
/// 27B start takes `diarizer.*` as it takes `speech.*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    /// Every family takes the field.
    AllFamilies,
    /// Only these families do; an explicit value for any other is refused at
    /// start and dropped on a switch.
    Only(&'static [ModelFamily]),
}

impl Applicability {
    /// Whether a load of `family` takes the field.
    pub fn to(self, family: ModelFamily) -> bool {
        match self {
            Applicability::AllFamilies => true,
            Applicability::Only(families) => families.contains(&family),
        }
    }

    /// Whether `family`'s load judges the field at all: every field but one
    /// only a model beside the active one takes (the diarizer's), which only
    /// such a family judges.
    pub fn judged_by(self, family: ModelFamily) -> bool {
        match self {
            Applicability::AllFamilies => true,
            Applicability::Only(families) => families.iter().any(|only| beside(*only) == beside(family)),
        }
    }

    /// Whether `family`'s load refuses an explicit value of the field (at a
    /// start; drops it on a switch): it judges the field and does not take it.
    pub fn refused_by(self, family: ModelFamily) -> bool {
        self.judged_by(family) && !self.to(family)
    }

    /// As `help --fields` prints it: `all`, or the families' names.
    pub fn describe(self) -> String {
        match self {
            Applicability::AllFamilies => "all".to_owned(),
            Applicability::Only(families) => families.iter().map(|f| f.name()).collect::<Vec<_>>().join(", "),
        }
    }

    /// As `GET /v1/config/fields` states it: `null` for a field every family
    /// takes, else the scope names ([`scope`]) of the families that do — the
    /// spelling a config file's family section uses.
    pub fn to_json(self) -> Value {
        match self {
            Applicability::AllFamilies => Value::Null,
            Applicability::Only(families) => families.iter().map(|family| scope(*family)).collect::<Vec<_>>().into(),
        }
    }
}

/// The yes/no attributes a field declaration may list in its trailing
/// `[..]` (spec config-v2/02 §`applicable_to` × `visible`/`patchable`/
/// `reload_required`, plus whether a family override exists). Each is off
/// unless listed: a field nobody thought about is hidden from `GET
/// /v1/config`, refused by `PATCH`, and has no family scope — the closed
/// side of every question, so forgetting an attribute can only ever make a
/// field less exposed, never more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    /// `GET /v1/config` shows it. A whitelist: a field without it is absent
    /// from the response, not masked.
    Visible,
    /// `PATCH /v1/config` may change it.
    Patchable,
    /// A change only takes effect through a model reload.
    ReloadRequired,
    /// It has `--<family>-<group>-<field>` / `IGNIS_<FAMILY>_…` /
    /// `<group>.<family>.<field>` variants beside the general ones.
    Scoped,
}

impl Attr {
    /// Whether `attrs` lists `wanted` (`const`, so a field's flags are
    /// computed into its `FieldMeta` at compile time).
    pub const fn listed(attrs: &[Attr], wanted: Attr) -> bool {
        let mut i = 0;
        while i < attrs.len() {
            if attrs[i] as u8 == wanted as u8 {
                return true;
            }
            i += 1;
        }
        false
    }
}

/// Everything about a field except its value: what its one declaration in
/// `schema.rs` expands to beside the struct field itself. `help --fields`,
/// the flag/env/file parsers, the applicability check and `GET`/`PATCH
/// /v1/config` all read this one table, so none of them can know a field
/// the others do not.
#[derive(Debug, Clone, Copy)]
pub struct FieldMeta {
    /// The group (`reuse`), as the file key and the env var spell it.
    pub group: &'static str,
    /// The field's own name (`kv_host_pool_bytes`), as the file key spells
    /// it; `stringify!`'d from the struct field, so it cannot differ from it.
    pub name: &'static str,
    /// The kind's tag (`bytes`, `bool`, …), for `help --fields`.
    pub kind: &'static str,
    /// The field may be unset (its kind is [`super::kind::Opt`]): `null` in a
    /// file, and "auto" or "off" where a page shows it.
    pub nullable: bool,
    /// The hardcoded default as the file format would write it.
    pub default: fn() -> Value,
    /// A value read from a config file as the text the kind parses
    /// (`FieldKind::file_text`); `Ok(None)` for `null`.
    pub file_text: fn(&Value) -> Result<Option<String>, String>,
    /// Raw text parsed by the kind and written back in canonical form
    /// (`2147483648` as `2G`), for a file a change is merged into.
    pub canonical: fn(&str) -> Result<Value, String>,
    /// The field's doc comment, which is written for the operator: the same
    /// text rustdoc shows on the struct field is what `help --fields` prints.
    pub description: &'static str,
    /// The rule a parsed value must meet.
    pub validator: Validator,
    /// The model families the field means anything for.
    pub applies: Applicability,
    /// `GET /v1/config` shows it ([`Attr::Visible`]).
    pub visible: bool,
    /// `PATCH /v1/config` may change it ([`Attr::Patchable`]).
    pub patchable: bool,
    /// A change takes effect only through a model reload
    /// ([`Attr::ReloadRequired`]).
    pub reload_required: bool,
    /// It has family-scoped variants ([`Attr::Scoped`]).
    pub scoped: bool,
    /// A bool: the flag may stand bare (`--server-ui` = `--server-ui true`).
    pub switch: bool,
    /// The flag may be given more than once, each naming one more entry.
    pub repeatable: bool,
}

impl FieldMeta {
    /// `--<group>-<field>`, underscores as dashes.
    pub fn flag(&self) -> String {
        format!("--{}-{}", dashed(self.group), dashed(self.name))
    }

    /// `--<family>-<group>-<field>`.
    pub fn scoped_flag(&self, family: ModelFamily) -> String {
        format!("--{}-{}-{}", scope(family), dashed(self.group), dashed(self.name))
    }

    /// `IGNIS_<GROUP>_<FIELD>`.
    pub fn env(&self) -> String {
        format!("IGNIS_{}_{}", self.group.to_ascii_uppercase(), self.name.to_ascii_uppercase())
    }

    /// `IGNIS_<FAMILY>_<GROUP>_<FIELD>`.
    pub fn scoped_env(&self, family: ModelFamily) -> String {
        format!(
            "IGNIS_{}_{}_{}",
            scope(family).to_ascii_uppercase(),
            self.group.to_ascii_uppercase(),
            self.name.to_ascii_uppercase()
        )
    }

    /// `<group>.<field>`, as a config file nests it.
    pub fn file_key(&self) -> String {
        format!("{}.{}", self.group, self.name)
    }

    /// `<group>.<family>.<field>`.
    pub fn scoped_file_key(&self, family: ModelFamily) -> String {
        format!("{}.{}.{}", self.group, scope(family), self.name)
    }

    /// The one family a field only that family takes belongs to
    /// (`Only([family])`), when it is a family with a config scope
    /// ([`FAMILIES`]); `None` for a field more families take, and for one
    /// only the diarizer takes, which keeps its plain key (spec speech/08).
    pub fn home_family(&self) -> Option<ModelFamily> {
        match self.applies {
            Applicability::Only([family]) if FAMILIES.contains(family) => Some(*family),
            _ => None,
        }
    }

    /// GitHub #729: whether the field is a draft setting, read only by a
    /// model's drafter: one of [`super::DRAFT_FIELDS`] or Flash-Next's draft
    /// rows. A model whose drafter does not read it ignores a value named for
    /// it, neither refused nor dropped, as a load whose speculation does not
    /// run ignores every one.
    pub fn is_draft_setting(&self) -> bool {
        self.group == "spec" && (super::DRAFT_FIELDS.contains(&self.name) || self.name == "draft_rows")
    }

    /// Whether a value of the field may sit in `family`'s own section (a
    /// file's `<group>.<family>.<field>`, an env var's `IGNIS_<FAMILY>_…`):
    /// a scoped field in either family's, and a field only one family takes
    /// in that family's — where `PATCH /v1/config` writes it, so that a start
    /// of the other model from the same file never sees it.
    pub fn takes_scope(&self, family: ModelFamily) -> bool {
        self.scoped || self.home_family() == Some(family)
    }

    /// The description on one line: rustdoc's leading spaces trimmed, the
    /// lines joined.
    pub fn description_line(&self) -> String {
        self.description.lines().map(str::trim).collect::<Vec<_>>().join(" ").trim().to_owned()
    }

    /// The field's declaration as a tool reads it — the keys `help --fields
    /// --format json` and `GET /v1/config/fields` have in common, built here
    /// once so the two cannot drift (spec playground/05 §`GET
    /// /v1/config/fields`). The command line adds what only it needs (the
    /// per-family flags, `visible`: the route lists only visible fields).
    ///
    /// `rule` is machine-readable ([`Validator::to_json`]) with `rule_text`
    /// the sentence `help --fields` prints (empty when there is no rule);
    /// `families` is `null` for a field every family takes ([`Applicability::to_json`]).
    pub fn declaration(&self) -> serde_json::Map<String, Value> {
        let Value::Object(declaration) = serde_json::json!({
            "group": self.group,
            "name": self.name,
            "file_key": self.file_key(),
            "flag": self.flag(),
            "env": self.env(),
            "kind": self.kind,
            "nullable": self.nullable,
            "default": (self.default)(),
            "rule": self.validator.to_json(self.canonical),
            "rule_text": self.validator.describe(),
            "families": self.applies.to_json(),
            "scoped": self.scoped,
            "patchable": self.patchable,
            "reload_required": self.reload_required,
            "description": self.description_line(),
        }) else {
            unreachable!("a JSON object literal")
        };
        declaration
    }

    /// The description's first sentence-or-paragraph, trimmed of rustdoc's
    /// leading spaces and joined onto one line.
    pub fn summary(&self) -> String {
        self.description
            .split("\n\n")
            .next()
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A value in thousandths as the decimal it stands for: `1500` is `1.5`.
fn decimal(thousandths: i64) -> String {
    let written = format!("{}.{:03}", thousandths / 1000, (thousandths % 1000).abs());
    written.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// A family of a model beside the active one (the diarizer, ADR 0069), never
/// switched to: its fields are its own group's.
fn beside(family: ModelFamily) -> bool {
    family.kind() == ModelKind::Diarizer
}

fn dashed(name: &str) -> String {
    name.replace('_', "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_range_refuses_either_side_naming_the_bound_and_the_value() {
        let both = Validator::Range { min: Some(1), max: Some(8) };
        assert!(both.check_number(1).is_ok() && both.check_number(8).is_ok());
        let err = both.check_number(9).unwrap_err();
        assert!(err.contains("1..=8") && err.contains('9'), "{err}");
        assert!(both.check_number(0).unwrap_err().contains("1..=8"));
        let floor = Validator::Range { min: Some(1), max: None };
        assert!(floor.check_number(i64::MAX).is_ok());
        assert_eq!(floor.check_number(0).unwrap_err(), "must be at least 1, got 0");
        let ceiling = Validator::Range { min: None, max: Some(3600) };
        assert_eq!(ceiling.check_number(3601).unwrap_err(), "must be at most 3600, got 3601");
        assert_eq!(both.describe(), "1..=8");
    }

    #[test]
    fn a_decimal_range_is_stated_and_refused_in_its_own_units() {
        let rate = Validator::DecimalRange { min: 0, max: 4_000 };
        assert!(rate.check_number(0).is_ok() && rate.check_number(1_500).is_ok() && rate.check_number(4_000).is_ok());
        assert_eq!(rate.check_number(4_250).unwrap_err(), "must be in 0..=4, got 4.25");
        assert_eq!(Validator::DecimalRange { min: 1, max: 1_000 }.describe(), "0.001..=1");
        assert_eq!(rate.to_json(|_| Ok(Value::Null)), serde_json::json!({ "min": 0.0, "max": 4.0 }));
    }

    #[test]
    fn a_multiple_of_refuses_zero_and_an_unaligned_value() {
        let chunk = Validator::MultipleOf(128);
        assert!(chunk.check_number(128).is_ok() && chunk.check_number(2048).is_ok());
        let err = chunk.check_number(1000).unwrap_err();
        assert!(err.contains("128") && err.contains("1000") && err.contains("nonzero"), "{err}");
        assert!(chunk.check_number(0).unwrap_err().contains("nonzero"));
        assert_eq!(chunk.describe(), "nonzero multiple of 128");
    }

    #[test]
    fn one_of_matches_trimmed_and_case_insensitively_and_lists_every_value() {
        let formats = Validator::OneOf(&["bf16", "hq-e8-2b"]);
        assert!(formats.check_text(" BF16 ").is_ok());
        let err = formats.check_text("fp8").unwrap_err();
        assert!(err.contains("bf16, hq-e8-2b") && err.contains("`fp8`"), "{err}");
        // It says nothing about numbers, and the others nothing about text.
        assert!(formats.check_number(-5).is_ok());
        assert!(Validator::MultipleOf(128).check_text("anything").is_ok());
    }

    #[test]
    fn applicability_names_the_families_it_takes() {
        let flash = Applicability::Only(&[ModelFamily::FlashNext]);
        assert!(flash.to(ModelFamily::FlashNext) && !flash.to(ModelFamily::Qwen38_27b));
        assert!(Applicability::AllFamilies.to(ModelFamily::Qwen38_27b));
        assert_eq!(flash.describe(), "Qwen3.8-Flash-Next");
    }

    /// Spec speech/08, AC 14: a field only the diarizer's family takes is
    /// refused by every other family that judges it, and judged by none of
    /// the active model's: they take it as they take `speech.*`.
    #[test]
    fn a_diarizer_field_is_the_diarizers_family_to_judge() {
        let diarizer = Applicability::Only(&[ModelFamily::Nemotron3Diarization]);
        assert!(diarizer.to(ModelFamily::Nemotron3Diarization) && !diarizer.refused_by(ModelFamily::Nemotron3Diarization));
        for family in ModelFamily::ALL.into_iter().filter(|family| *family != ModelFamily::Nemotron3Diarization) {
            assert!(!diarizer.to(family), "{family:?} takes it");
            assert!(!diarizer.judged_by(family) && !diarizer.refused_by(family), "{family:?} judges it");
        }
        assert_eq!(diarizer.describe(), "Nemotron 3 Diarization");
        // An active model's field: the diarizer does not judge it, the other
        // active families refuse it as before (the image model included).
        let flash = Applicability::Only(&[ModelFamily::FlashNext]);
        assert!(flash.refused_by(ModelFamily::Qwen38_27b) && flash.refused_by(ModelFamily::QwenImage21));
        assert!(!flash.judged_by(ModelFamily::Nemotron3Diarization));
        assert!(Applicability::AllFamilies.judged_by(ModelFamily::Nemotron3Diarization));
    }

    #[test]
    fn the_family_scopes_are_plain_identifiers_and_read_back() {
        assert_eq!(scope(ModelFamily::Qwen38_27b), "qwen38");
        assert_eq!(scope(ModelFamily::FlashNext), "qwen38flashnext");
        for family in FAMILIES {
            assert_eq!(family_of_scope(scope(family)), Some(family));
            assert_eq!(family_of_scope(&scope(family).to_ascii_uppercase()), Some(family));
        }
        assert_eq!(family_of_scope("qwen3.8-27b"), None, "a model id is not a scope");
        // the image model has a spelling but no config scope until spec image-model/05
        assert_eq!(scope(ModelFamily::QwenImage21), "qwenimage21");
        assert_eq!(family_of_scope("qwenimage21"), None);
        // nor has the diarizer, whose fields keep their plain keys (spec speech/08)
        assert_eq!(family_of_scope(scope(ModelFamily::Nemotron3Diarization)), None);
    }

    #[test]
    fn attributes_are_off_unless_listed() {
        assert!(!Attr::listed(&[], Attr::Visible));
        assert!(Attr::listed(&[Attr::Patchable, Attr::Visible], Attr::Visible));
        assert!(!Attr::listed(&[Attr::Patchable], Attr::ReloadRequired));
    }

    #[test]
    fn a_field_is_spelled_one_way_per_surface() {
        let meta = FieldMeta {
            group: "kv_disk",
            name: "kv_host_pool_bytes",
            kind: "bytes",
            nullable: false,
            default: || Value::Null,
            file_text: |_| Ok(None),
            canonical: |_| Ok(Value::Null),
            description: " The budget.\n\n More.",
            validator: Validator::None,
            applies: Applicability::AllFamilies,
            visible: false,
            patchable: false,
            reload_required: false,
            scoped: true,
            switch: false,
            repeatable: false,
        };
        assert_eq!(meta.flag(), "--kv-disk-kv-host-pool-bytes");
        assert_eq!(meta.env(), "IGNIS_KV_DISK_KV_HOST_POOL_BYTES");
        assert_eq!(meta.file_key(), "kv_disk.kv_host_pool_bytes");
        assert_eq!(meta.scoped_flag(ModelFamily::Qwen38_27b), "--qwen38-kv-disk-kv-host-pool-bytes");
        assert_eq!(meta.scoped_env(ModelFamily::FlashNext), "IGNIS_QWEN38FLASHNEXT_KV_DISK_KV_HOST_POOL_BYTES");
        assert_eq!(meta.scoped_file_key(ModelFamily::FlashNext), "kv_disk.qwen38flashnext.kv_host_pool_bytes");
        assert_eq!(meta.summary(), "The budget.");
    }
}
