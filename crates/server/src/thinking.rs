//! The `enable_thinking` / `reasoning_effort` wire contract (GitHub #68):
//! parsing the request's thinking controls, resolving them against the
//! server's configured defaults and the loaded template's probed
//! capabilities, and the resolved value ([`ThinkingOptions`]) that crosses
//! the template seam.
//!
//! Mirrors the reference field-for-field (`.scratch/server/specs/04-enable-thinking.md`):
//! `enable_thinking` accepted at the top level and under
//! `chat_template_kwargs`, `null` treated as unset, a same-value duplicate
//! accepted, a conflicting duplicate rejected, `reasoning_effort` resolved
//! against the template's probed vocabulary with a validation/capability
//! error distinction.

use serde_json::Value as JsonValue;

pub use ignis_artifact::{ReasoningEffort, ThinkingCapabilities};

/// The resolved thinking semantics for one request — constructed once,
/// after validation, and the only thing (besides the messages) that
/// crosses the template seam ([`crate::template::TemplateProvider::apply_chat_template`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingOptions {
    /// Always resolved explicitly (never "let it default further down") —
    /// this is what gets bound to the template's `enable_thinking`
    /// variable.
    pub enable_thinking: bool,
    /// Absent means "let the template's own default apply" — bound to the
    /// template's `reasoning_effort` variable only when `Some`.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Whether inbound assistant `reasoning_content` survives into the
    /// rendered prompt (default: dropped).
    pub preserve_thinking: bool,
}

impl Default for ThinkingOptions {
    /// Thinking on, template's own default effort, prior reasoning dropped
    /// — the server's own out-of-the-box stance before any env-var default
    /// or per-request override.
    fn default() -> Self {
        Self {
            enable_thinking: true,
            reasoning_effort: None,
            preserve_thinking: false,
        }
    }
}

/// The server-wide defaults (`IGNIS_ENABLE_THINKING` / `IGNIS_REASONING_EFFORT`)
/// a request's unset fields fall back to.
#[derive(Debug, Clone, Copy)]
pub struct ThinkingDefaults {
    pub enable_thinking: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl Default for ThinkingDefaults {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            reasoning_effort: None,
        }
    }
}

/// A rejected request. The two classes are deliberately distinguishable
/// (the wire contract's machine-readable `code`): a validation error is the
/// client's mistake (a bad type, a conflict, an unknown vocabulary word); a
/// capability error is the loaded model's limitation (a real vocabulary
/// word the template does not support).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingError {
    Validation(String),
    Capability(String),
}

impl ThinkingError {
    pub fn message(&self) -> &str {
        match self {
            Self::Validation(m) | Self::Capability(m) => m,
        }
    }
}

/// The raw thinking-related fields of a request, exactly as they arrived on
/// the wire (still JSON — parsed and validated by [`resolve`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkingRequestFields<'a> {
    pub enable_thinking: Option<&'a JsonValue>,
    pub reasoning_effort: Option<&'a JsonValue>,
    pub preserve_thinking: Option<&'a JsonValue>,
    pub chat_template_kwargs: Option<&'a JsonValue>,
}

/// Resolve one request's thinking controls against the server's defaults
/// and the loaded template's capabilities.
///
/// Validation runs before capability resolution (a client's own mistake is
/// reported as such, even when the loaded model also could not have
/// honoured the request).
pub fn resolve(
    fields: ThinkingRequestFields<'_>,
    defaults: &ThinkingDefaults,
    capabilities: &ThinkingCapabilities,
) -> Result<ThinkingOptions, ThinkingError> {
    let kwargs = parse_chat_template_kwargs(fields.chat_template_kwargs)?;

    let top_enable = optional_bool(fields.enable_thinking, "enable_thinking")?;
    let enable = merge_bool(top_enable, kwargs.enable_thinking, "enable_thinking")?;

    let top_preserve = optional_bool(fields.preserve_thinking, "preserve_thinking")?;
    let preserve = merge_bool(top_preserve, kwargs.preserve_thinking, "preserve_thinking")?;

    let effort = optional_effort(fields.reasoning_effort)?;

    // Cross-field agreement: an effort implies a thinking state; an
    // explicit enable_thinking that disagrees is the client's mistake,
    // checked before any capability question.
    if let (Some(effort), Some(enable)) = (effort, enable) {
        let implied = effort != ReasoningEffort::None;
        if implied != enable {
            return Err(ThinkingError::Validation(format!(
                "reasoning_effort \"{}\" conflicts with enable_thinking: {enable}",
                effort.as_str()
            )));
        }
    }

    let resolved_enable = enable
        .or_else(|| effort.map(|e| e != ReasoningEffort::None))
        .unwrap_or(defaults.enable_thinking);
    let resolved_effort = if resolved_enable {
        match effort {
            Some(ReasoningEffort::None) => None,
            Some(other) => Some(other),
            None => defaults.reasoning_effort,
        }
    } else {
        None
    };

    if !resolved_enable && !capabilities.can_disable {
        return Err(ThinkingError::Capability(
            "the loaded template cannot disable thinking".to_owned(),
        ));
    }
    if let Some(effort) = resolved_effort {
        if !capabilities.supports(effort) {
            return Err(ThinkingError::Capability(format!(
                "the loaded template does not support reasoning_effort \"{}\"",
                effort.as_str()
            )));
        }
    }

    Ok(ThinkingOptions {
        enable_thinking: resolved_enable,
        reasoning_effort: resolved_effort,
        preserve_thinking: preserve.unwrap_or(false),
    })
}

#[derive(Debug, Default)]
struct ChatTemplateKwargs {
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
}

const CHAT_TEMPLATE_KWARGS_KEYS: [&str; 2] = ["enable_thinking", "preserve_thinking"];

fn parse_chat_template_kwargs(
    value: Option<&JsonValue>,
) -> Result<ChatTemplateKwargs, ThinkingError> {
    let value = match value {
        None | Some(JsonValue::Null) => return Ok(ChatTemplateKwargs::default()),
        Some(v) => v,
    };
    let obj = value.as_object().ok_or_else(|| {
        ThinkingError::Validation("`chat_template_kwargs` must be an object".to_owned())
    })?;
    for key in obj.keys() {
        if !CHAT_TEMPLATE_KWARGS_KEYS.contains(&key.as_str()) {
            return Err(ThinkingError::Validation(format!(
                "chat_template_kwargs: unsupported key `{key}` (supported: {CHAT_TEMPLATE_KWARGS_KEYS:?})"
            )));
        }
    }
    Ok(ChatTemplateKwargs {
        enable_thinking: optional_bool(
            obj.get("enable_thinking"),
            "chat_template_kwargs.enable_thinking",
        )?,
        preserve_thinking: optional_bool(
            obj.get("preserve_thinking"),
            "chat_template_kwargs.preserve_thinking",
        )?,
    })
}

/// `None` (absent) and `Some(Null)` both mean unset; a non-boolean is a
/// validation error naming the field.
fn optional_bool(value: Option<&JsonValue>, field: &str) -> Result<Option<bool>, ThinkingError> {
    match value {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ThinkingError::Validation(format!(
            "`{field}` must be a boolean"
        ))),
    }
}

/// The top-level field and its `chat_template_kwargs` twin: absent on both
/// sides is unset; present on one side wins; present on both with the same
/// value is accepted; present on both with different values is a 400
/// naming the conflict.
fn merge_bool(
    top: Option<bool>,
    nested: Option<bool>,
    field: &str,
) -> Result<Option<bool>, ThinkingError> {
    match (top, nested) {
        (Some(a), Some(b)) if a != b => Err(ThinkingError::Validation(format!(
            "`{field}` ({a}) conflicts with `chat_template_kwargs.{field}` ({b})"
        ))),
        (Some(a), _) => Ok(Some(a)),
        (None, Some(b)) => Ok(Some(b)),
        (None, None) => Ok(None),
    }
}

fn optional_effort(value: Option<&JsonValue>) -> Result<Option<ReasoningEffort>, ThinkingError> {
    match value {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::String(s)) => ReasoningEffort::parse(s).map(Some).ok_or_else(|| {
            let accepted: Vec<&str> = ReasoningEffort::ALL.iter().map(|e| e.as_str()).collect();
            ThinkingError::Validation(format!(
                "`reasoning_effort` must be one of {accepted:?}, got {s:?}"
            ))
        }),
        Some(_) => Err(ThinkingError::Validation(
            "`reasoning_effort` must be a string".to_owned(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Server-wide defaults (`IGNIS_ENABLE_THINKING` / `IGNIS_REASONING_EFFORT`)
// ---------------------------------------------------------------------------

/// Parse `IGNIS_ENABLE_THINKING` (already defaulted to `"true"` by the
/// caller when unset). A malformed value is a startup failure, not a silent
/// fallback (matches the server's existing "refuse to start" convention for
/// a bad env default).
pub fn parse_default_enable_thinking(value: &str) -> Result<bool, String> {
    value
        .parse::<bool>()
        .map_err(|_| format!("IGNIS_ENABLE_THINKING must be \"true\" or \"false\", got {value:?}"))
}

/// Parse `IGNIS_REASONING_EFFORT` (empty = unset — the template's own
/// default applies).
pub fn parse_default_reasoning_effort(value: &str) -> Result<Option<ReasoningEffort>, String> {
    if value.is_empty() {
        return Ok(None);
    }
    ReasoningEffort::parse(value).map(Some).ok_or_else(|| {
        let accepted: Vec<&str> = ReasoningEffort::ALL.iter().map(|e| e.as_str()).collect();
        format!("IGNIS_REASONING_EFFORT must be one of {accepted:?}, got {value:?}")
    })
}

/// Refuse a default the loaded template cannot honour (a model swap must
/// not silently change behaviour — the operator finds out at startup, not
/// on the first request).
pub fn validate_defaults(
    defaults: &ThinkingDefaults,
    capabilities: &ThinkingCapabilities,
) -> Result<(), String> {
    if !defaults.enable_thinking && !capabilities.can_disable {
        return Err(
            "the loaded template cannot disable thinking, but IGNIS_ENABLE_THINKING=false"
                .to_owned(),
        );
    }
    if defaults.enable_thinking {
        if let Some(effort) = defaults.reasoning_effort {
            if !capabilities.supports(effort) {
                return Err(format!(
                    "the loaded template does not support IGNIS_REASONING_EFFORT={}",
                    effort.as_str()
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn permissive() -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn fields<'a>(
        enable_thinking: Option<&'a JsonValue>,
        reasoning_effort: Option<&'a JsonValue>,
        preserve_thinking: Option<&'a JsonValue>,
        chat_template_kwargs: Option<&'a JsonValue>,
    ) -> ThinkingRequestFields<'a> {
        ThinkingRequestFields {
            enable_thinking,
            reasoning_effort,
            preserve_thinking,
            chat_template_kwargs,
        }
    }

    #[test]
    fn no_fields_falls_back_to_the_server_default() {
        let opts = resolve(
            fields(None, None, None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert_eq!(opts, ThinkingOptions::default());
    }

    #[test]
    fn a_configured_server_default_applies_when_unset() {
        let defaults = ThinkingDefaults {
            enable_thinking: false,
            reasoning_effort: None,
        };
        let opts = resolve(fields(None, None, None, None), &defaults, &permissive())
            .expect("resolve");
        assert!(!opts.enable_thinking);
        assert_eq!(opts.reasoning_effort, None);
    }

    #[test]
    fn enable_thinking_false_disables_and_needs_no_effort() {
        let opts = resolve(
            fields(Some(&json!(false)), None, None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(!opts.enable_thinking);
        assert_eq!(opts.reasoning_effort, None);
    }

    #[test]
    fn enable_thinking_true_overrides_a_disabled_server_default() {
        let defaults = ThinkingDefaults {
            enable_thinking: false,
            reasoning_effort: None,
        };
        let opts = resolve(
            fields(Some(&json!(true)), None, None, None),
            &defaults,
            &permissive(),
        )
        .expect("resolve");
        assert!(opts.enable_thinking);
    }

    #[test]
    fn null_enable_thinking_is_treated_as_unset() {
        let opts = resolve(
            fields(Some(&json!(null)), None, None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert_eq!(opts, ThinkingOptions::default());
    }

    #[test]
    fn a_non_boolean_enable_thinking_is_a_validation_error_naming_the_field() {
        let err = resolve(
            fields(Some(&json!("nope")), None, None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Validation(_)));
        assert!(err.message().contains("enable_thinking"), "{err:?}");
    }

    #[test]
    fn chat_template_kwargs_enable_thinking_is_honoured() {
        let kwargs = json!({ "enable_thinking": false });
        let opts = resolve(
            fields(None, None, None, Some(&kwargs)),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(!opts.enable_thinking);
    }

    #[test]
    fn matching_top_level_and_kwargs_values_are_accepted() {
        let kwargs = json!({ "enable_thinking": false });
        let opts = resolve(
            fields(Some(&json!(false)), None, None, Some(&kwargs)),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(!opts.enable_thinking);
    }

    #[test]
    fn conflicting_top_level_and_kwargs_values_are_rejected() {
        let kwargs = json!({ "enable_thinking": true });
        let err = resolve(
            fields(Some(&json!(false)), None, None, Some(&kwargs)),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Validation(_)));
        assert!(err.message().contains("enable_thinking"), "{err:?}");
    }

    #[test]
    fn an_unsupported_chat_template_kwargs_key_is_rejected() {
        let kwargs = json!({ "top_p": 0.9 });
        let err = resolve(
            fields(None, None, None, Some(&kwargs)),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Validation(_)));
        assert!(err.message().contains("top_p"), "{err:?}");
    }

    #[test]
    fn reasoning_effort_low_enables_thinking_with_that_effort() {
        let opts = resolve(
            fields(None, Some(&json!("low")), None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(opts.enable_thinking);
        assert_eq!(opts.reasoning_effort, Some(ReasoningEffort::Low));
    }

    #[test]
    fn reasoning_effort_none_disables_thinking() {
        let opts = resolve(
            fields(None, Some(&json!("none")), None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(!opts.enable_thinking);
        assert_eq!(opts.reasoning_effort, None);
    }

    #[test]
    fn an_effort_outside_the_protocol_vocabulary_lists_accepted_values() {
        let err = resolve(
            fields(None, Some(&json!("super-duper")), None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Validation(_)));
        assert!(err.message().contains("xhigh"), "{err:?}");
    }

    #[test]
    fn a_disagreeing_effort_and_enable_thinking_is_rejected() {
        let err = resolve(
            fields(Some(&json!(true)), Some(&json!("none")), None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Validation(_)));
    }

    #[test]
    fn an_effort_the_template_cannot_honour_is_a_capability_error() {
        let caps = ThinkingCapabilities {
            can_disable: true,
            supported_efforts: [ReasoningEffort::Low].into_iter().collect(),
        };
        let err = resolve(
            fields(None, Some(&json!("high")), None, None),
            &ThinkingDefaults::default(),
            &caps,
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Capability(_)));
    }

    #[test]
    fn disabling_thinking_on_a_template_that_cannot_is_a_capability_error() {
        let caps = ThinkingCapabilities {
            can_disable: false,
            supported_efforts: [ReasoningEffort::Low].into_iter().collect(),
        };
        let err = resolve(
            fields(Some(&json!(false)), None, None, None),
            &ThinkingDefaults::default(),
            &caps,
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Capability(_)));
    }

    #[test]
    fn validation_errors_take_priority_over_capability_errors() {
        // A conflicting request is a client mistake even when the template
        // also could not have honoured either side.
        let caps = ThinkingCapabilities {
            can_disable: false,
            supported_efforts: Default::default(),
        };
        let err = resolve(
            fields(Some(&json!(true)), Some(&json!("none")), None, None),
            &ThinkingDefaults::default(),
            &caps,
        )
        .expect_err("must reject");
        assert!(matches!(err, ThinkingError::Validation(_)));
    }

    #[test]
    fn preserve_thinking_defaults_to_false() {
        let opts = resolve(
            fields(None, None, None, None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(!opts.preserve_thinking);
    }

    #[test]
    fn preserve_thinking_true_is_honoured() {
        let opts = resolve(
            fields(None, None, Some(&json!(true)), None),
            &ThinkingDefaults::default(),
            &permissive(),
        )
        .expect("resolve");
        assert!(opts.preserve_thinking);
    }

    #[test]
    fn env_default_enable_thinking_rejects_a_non_boolean() {
        assert!(parse_default_enable_thinking("not-a-bool").is_err());
        assert_eq!(parse_default_enable_thinking("false").unwrap(), false);
    }

    #[test]
    fn env_default_reasoning_effort_rejects_an_unknown_value() {
        assert!(parse_default_reasoning_effort("nonsense").is_err());
        assert_eq!(
            parse_default_reasoning_effort("low").unwrap(),
            Some(ReasoningEffort::Low)
        );
        assert_eq!(parse_default_reasoning_effort("").unwrap(), None);
    }

    #[test]
    fn validate_defaults_rejects_a_default_the_template_cannot_honour() {
        let caps = ThinkingCapabilities {
            can_disable: true,
            supported_efforts: [ReasoningEffort::Low].into_iter().collect(),
        };
        let bad = ThinkingDefaults {
            enable_thinking: true,
            reasoning_effort: Some(ReasoningEffort::High),
        };
        assert!(validate_defaults(&bad, &caps).is_err());
        let ok = ThinkingDefaults {
            enable_thinking: true,
            reasoning_effort: Some(ReasoningEffort::Low),
        };
        assert!(validate_defaults(&ok, &caps).is_ok());
    }
}
