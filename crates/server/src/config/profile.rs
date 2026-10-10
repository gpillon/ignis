//! Hardware profiles (spec config-v2/02 §`--profile`): a named bundle of
//! field defaults for one kind of card, slotted between the config file and
//! the hardcoded defaults — a smarter default, never an override.
//!
//! The built-in profiles are written below in the config file's own
//! `profiles:` shape and read by the same [`super::file::read_document`] a
//! file's section is, so a profile an operator defines for a card the
//! maintainers do not own resolves exactly as a built-in one does: there is
//! one code path, and a built-in profile is only the default contents of it.

use std::collections::BTreeMap;

use super::file::read_document;
use super::source::Layer;
use super::ConfigError;

/// The profile used when nothing names one: the owner's RTX 5090, whose
/// numbers are the hardcoded defaults — so an unset `--profile` changes
/// nothing on that machine.
pub const DEFAULT_PROFILE: &str = "rtx5090";

/// The built-in profiles, as a config file's `profiles:` section.
///
/// `rtx5090` names only the VRAM headroom: the one hardware-shaped field
/// whose default is a plain number. The others the spec lists are not field
/// defaults a profile can restate — the expert cache's 12 GiB floor is the
/// engine's constant, not a config field; the decode lanes and retained
/// host slots are unset by default so the model's own default applies (3
/// lanes, 16 or 8 slots), and a profile value would replace that with one
/// number for both models. No other card ships a profile yet: its numbers
/// have to be measured on the card (finding
/// 2026-10-09-vram-headroom-wddm-paging is what 1.5 GiB came from), and a
/// guessed profile would be worse than none. A config file's `profiles:`
/// defines one for any other card.
const BUILT_IN: &str = r#"{
    "rtx5090": {
        "vram": { "headroom_bytes": "1536M" }
    }
}"#;

/// Every built-in profile, by name.
pub fn built_in() -> BTreeMap<String, Layer> {
    let value = serde_json::from_str(&format!(r#"{{"profiles": {BUILT_IN}}}"#)).expect("the built-in profiles are JSON");
    read_document(&value, "the built-in profiles").expect("the built-in profiles name only declared fields").profiles
}

/// The profile called `name`: the config file's own first (an operator may
/// redefine a built-in one for their card), then the built-in ones.
pub fn lookup(name: &str, from_file: &BTreeMap<String, Layer>) -> Result<Layer, ConfigError> {
    if let Some(layer) = from_file.get(name) {
        return Ok(layer.clone());
    }
    let built_in = built_in();
    built_in.get(name).cloned().ok_or_else(|| {
        let mut known: Vec<&str> = built_in.keys().map(String::as_str).collect();
        known.extend(from_file.keys().map(String::as_str));
        known.sort_unstable();
        known.dedup();
        ConfigError(format!("no profile `{name}` (the profiles are {})", known.join(", ")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::field;

    #[test]
    fn the_default_profile_restates_the_hardcoded_default_and_nothing_else() {
        let profiles = built_in();
        let rtx5090 = &profiles[DEFAULT_PROFILE];
        let headroom = field("vram", "headroom_bytes").unwrap();
        assert_eq!(rtx5090.get(headroom, None).unwrap().raw, "1536M");
        assert_eq!(rtx5090.entries().count(), 1);
        assert_eq!(
            (headroom.default)(),
            serde_json::Value::String("1536M".into()),
            "the profile and the hardcoded default agree"
        );
    }

    #[test]
    fn a_file_profile_shadows_a_built_in_one_and_an_unknown_name_lists_the_known() {
        let headroom = field("vram", "headroom_bytes").unwrap();
        let mut mine = Layer::default();
        mine.set(headroom, None, super::super::source::Candidate { raw: "2G".into(), spelling: "x".into() }).unwrap();
        let from_file = BTreeMap::from([("rtx5090".to_owned(), mine.clone()), ("a4000".to_owned(), mine.clone())]);
        assert_eq!(lookup("rtx5090", &from_file).unwrap(), mine);
        assert_eq!(lookup("rtx5090", &BTreeMap::new()).unwrap(), built_in()["rtx5090"]);
        let err = lookup("h100", &from_file).unwrap_err().0;
        assert!(err.contains("`h100`") && err.contains("a4000, rtx5090"), "{err}");
    }
}
