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

use ignis_core::compute::ModelFamily;
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
            Validator::OneOf(allowed) => format!("one of {}", allowed.join(", ")),
        }
    }
}

/// Which model families a field means anything for (spec config-v2/01
/// §`applicable_to`).
///
/// A field only one family has at all — Flash-Next's decode lanes, the
/// 27B's vision tower — is `Only` that family. An explicit value for it
/// while the other family is the one being configured is refused at start
/// (the operator named it for this load, so it is a mistake made now) and
/// dropped with a log line on a switch (it was named for the model being
/// switched away from, and was never wrong). This replaces the drop list
/// `fit_to_family` used to keep by hand.
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

    /// As `help --fields` prints it: `all`, or the families' names.
    pub fn describe(self) -> String {
        match self {
            Applicability::AllFamilies => "all".to_owned(),
            Applicability::Only(families) => families.iter().map(|f| f.name()).collect::<Vec<_>>().join(", "),
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

    #[test]
    fn the_family_scopes_are_plain_identifiers_and_read_back() {
        assert_eq!(scope(ModelFamily::Qwen38_27b), "qwen38");
        assert_eq!(scope(ModelFamily::FlashNext), "qwen38flashnext");
        for family in FAMILIES {
            assert_eq!(family_of_scope(scope(family)), Some(family));
            assert_eq!(family_of_scope(&scope(family).to_ascii_uppercase()), Some(family));
        }
        assert_eq!(family_of_scope("qwen3.8-27b"), None, "a model id is not a scope");
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
