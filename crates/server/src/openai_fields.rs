//! Every field of the OpenAI request bodies this server does not model on
//! its own, classified by value (spec server/09, GitHub #284): **inert** —
//! the value asks for what this server already does, so it is accepted and
//! changes nothing — or **refused**, a `400` naming the field in `param` and
//! saying what to send instead.
//!
//! One table per surface, read by both the request validator ([`check`])
//! and the OpenAPI document ([`describe`]), so what is refused and what the
//! document says is refused cannot drift.
//!
//! The enumerated side is the finite one: the fields OpenAI documents. A
//! field outside the table stays ignored, as serde's default has it, because
//! refusing the open-ended side would turn every field a gateway or a new
//! SDK adds into an outage. Fields a handler already reads (`stop`,
//! `max_completion_tokens`, `tools`, the thinking controls, ...) are not
//! here: their parse is theirs.

use serde_json::{Map, Value as JsonValue};

/// Which request body a table classifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// `POST /v1/chat/completions`.
    Chat,
    /// `POST /v1/responses`, and the WebSocket mode's `response.create`.
    Responses,
}

/// One field's row: when its value is inert, and why any other value is
/// refused.
pub struct FieldRule {
    /// The top-level request field.
    pub name: &'static str,
    /// Whether `value` (never absent: an absent field is not checked) asks
    /// for nothing this server does not already do.
    inert: fn(&JsonValue) -> bool,
    /// The inert values, as the document lists them.
    pub inert_when: &'static str,
    /// Why a value past [`FieldRule::inert_when`] is refused and what to
    /// send instead; `None` for a field that is inert at every value.
    pub refusal: Option<&'static str>,
}

/// A refused field: the `param` the 400 names, and its message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub param: &'static str,
    pub message: String,
}

impl FieldRule {
    /// Whether this row refuses some value.
    pub fn can_refuse(&self) -> bool {
        self.refusal.is_some()
    }
}

fn always(_: &JsonValue) -> bool {
    true
}

fn null(value: &JsonValue) -> bool {
    value.is_null()
}

fn null_or_one(value: &JsonValue) -> bool {
    value.is_null() || value.as_u64() == Some(1)
}

fn null_or_zero(value: &JsonValue) -> bool {
    value.is_null() || value.as_u64() == Some(0)
}

fn null_or_false(value: &JsonValue) -> bool {
    value.is_null() || value == &JsonValue::Bool(false)
}

fn null_or_true(value: &JsonValue) -> bool {
    value.is_null() || value == &JsonValue::Bool(true)
}

fn null_or_empty_object(value: &JsonValue) -> bool {
    value.is_null() || value.as_object().is_some_and(Map::is_empty)
}

fn null_or_text_format(value: &JsonValue) -> bool {
    value.is_null()
        || value
            .as_object()
            .is_some_and(|o| o.len() == 1 && o.get("type").and_then(JsonValue::as_str) == Some("text"))
}

fn null_or_standard_tier(value: &JsonValue) -> bool {
    value.is_null() || matches!(value.as_str(), Some("auto" | "default"))
}

fn null_or_text_modality(value: &JsonValue) -> bool {
    value.is_null() || value.as_array().is_some_and(|m| m.len() == 1 && m[0] == "text")
}

fn null_or_disabled(value: &JsonValue) -> bool {
    value.is_null() || value.as_str() == Some("disabled")
}

const ALWAYS: &str = "any value";


/// The Chat Completions body's fields past what the handler reads.
pub const CHAT: &[FieldRule] = &[
    FieldRule {
        name: "n",
        inert: null_or_one,
        inert_when: "`1`",
        refusal: Some("this server returns one choice per request; send n: 1 or omit it (more than one choice is GitHub #289)"),
    },
    FieldRule {
        name: "logprobs",
        inert: null_or_false,
        inert_when: "`false`",
        refusal: Some("log probabilities are not served; send logprobs: false or omit it (GitHub #288)"),
    },
    FieldRule {
        name: "top_logprobs",
        inert: null,
        inert_when: "`null`",
        refusal: Some("log probabilities are not served; omit top_logprobs (GitHub #288)"),
    },
    FieldRule {
        name: "response_format",
        inert: null_or_text_format,
        inert_when: "`{\"type\": \"text\"}`",
        refusal: Some("structured output is not served, this server returns plain text only; send {\"type\": \"text\"} or omit it (GitHub #287)"),
    },
    FieldRule {
        name: "logit_bias",
        inert: null_or_empty_object,
        inert_when: "`{}`",
        refusal: Some("logit bias is not served; send {} or omit it (GitHub #288)"),
    },
    FieldRule {
        name: "parallel_tool_calls",
        inert: null_or_true,
        inert_when: "`true`",
        refusal: Some("this template emits each tool call as it closes and cannot hold the model to one; send true or omit it"),
    },
    FieldRule {
        name: "store",
        inert: null_or_false,
        inert_when: "`false`",
        refusal: Some("this server stores no completions; send false or omit it"),
    },
    FieldRule {
        name: "service_tier",
        inert: null_or_standard_tier,
        inert_when: "`\"auto\"`, `\"default\"`",
        refusal: Some("this server runs one card at one tier; send \"auto\" or \"default\", or omit it"),
    },
    FieldRule {
        name: "modalities",
        inert: null_or_text_modality,
        inert_when: "`[\"text\"]`",
        refusal: Some("this server generates text only, no audio; send [\"text\"] or omit it"),
    },
    FieldRule { name: "metadata", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "user", inert: always, inert_when: ALWAYS, refusal: None },
    // Inert and staying inert: reuse here is by content (ADR 0029), and a key
    // that changed nothing would be a promise not kept.
    FieldRule { name: "prompt_cache_key", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "safety_identifier", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "verbosity", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule {
        name: "audio",
        inert: null,
        inert_when: "`null`",
        refusal: Some("audio output is not served; omit audio"),
    },
    FieldRule {
        name: "prediction",
        inert: null,
        inert_when: "`null`",
        refusal: Some("predicted outputs are not served; omit prediction"),
    },
    FieldRule {
        name: "web_search_options",
        inert: null,
        inert_when: "`null`",
        refusal: Some("this server runs no hosted tools, web search included; omit web_search_options"),
    },
    FieldRule {
        name: "functions",
        inert: null,
        inert_when: "`null`",
        refusal: Some("functions is the deprecated form of tools; use tools"),
    },
    FieldRule {
        name: "function_call",
        inert: null,
        inert_when: "`null`",
        refusal: Some("function_call is the deprecated form of tool_choice; use tool_choice"),
    },
    FieldRule {
        name: "best_of",
        inert: null,
        inert_when: "`null`",
        refusal: Some("best_of is a Completions field, and /v1/completions is not served; use messages"),
    },
    FieldRule {
        name: "echo",
        inert: null,
        inert_when: "`null`",
        refusal: Some("echo is a Completions field, and /v1/completions is not served; use messages"),
    },
    FieldRule {
        name: "suffix",
        inert: null,
        inert_when: "`null`",
        refusal: Some("suffix is a Completions field, and /v1/completions is not served; use messages"),
    },
    FieldRule {
        name: "prompt",
        inert: null,
        inert_when: "`null`",
        refusal: Some("prompt is a Completions field, and /v1/completions is not served; use messages"),
    },
];

/// The Responses body's fields past what the handler reads. `text.format`,
/// hosted `tools`, `background` and `previous_response_id` are refused where
/// they are read (`responses::input`).
///
/// `parallel_tool_calls`, `store` and `service_tier` are inert at every
/// value here, unlike on chat (spec server/09 acceptance 2 — a value a real
/// client sends is inert): Codex sends `parallel_tool_calls: false` and
/// `store: false`, its flex and fast modes send `service_tier: "flex"` and
/// `"priority"`, and the Responses default of `store` is `true`.
/// `parallel_tool_calls` and `store` are echoed on the response.
pub const RESPONSES: &[FieldRule] = &[
    FieldRule { name: "parallel_tool_calls", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "store", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "service_tier", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule {
        name: "top_logprobs",
        inert: null_or_zero,
        inert_when: "`0`",
        refusal: Some("log probabilities are not served; send 0 or omit top_logprobs (GitHub #288)"),
    },
    FieldRule {
        name: "truncation",
        inert: null_or_disabled,
        inert_when: "`\"disabled\"`",
        refusal: Some("this server never drops input to fit the context, a prompt over it is a 400; send \"disabled\" or omit it"),
    },
    FieldRule {
        name: "conversation",
        inert: null,
        inert_when: "`null`",
        refusal: Some("this server stores no conversations; send the whole input, or continue on the WebSocket mode"),
    },
    FieldRule {
        name: "prompt",
        inert: null,
        inert_when: "`null`",
        refusal: Some("stored prompt templates are not served; send instructions and input"),
    },
    FieldRule {
        name: "context_management",
        inert: null,
        inert_when: "`null`",
        refusal: Some("server-side compaction is not served; omit context_management"),
    },
    FieldRule { name: "metadata", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "user", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "prompt_cache_key", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "prompt_cache_retention", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "safety_identifier", inert: always, inert_when: ALWAYS, refusal: None },
    // Codex asks for `reasoning.encrypted_content`; this server has nothing
    // to encrypt and sends the reasoning as text.
    FieldRule { name: "include", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "stream_options", inert: always, inert_when: ALWAYS, refusal: None },
    // Counts hosted-tool calls, and this server runs none.
    FieldRule { name: "max_tool_calls", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "client_metadata", inert: always, inert_when: ALWAYS, refusal: None },
];

/// The table for `surface`.
pub fn table(surface: Surface) -> &'static [FieldRule] {
    match surface {
        Surface::Chat => CHAT,
        Surface::Responses => RESPONSES,
    }
}

/// The first field of `fields` whose value `surface`'s table refuses. An
/// absent field, and a field outside the table, are never refused.
pub fn check(fields: &Map<String, JsonValue>, surface: Surface) -> Result<(), Refusal> {
    for rule in table(surface) {
        let Some(value) = fields.get(rule.name) else { continue };
        if (rule.inert)(value) {
            continue;
        }
        let reason = rule.refusal.expect("a row inert at every value refuses nothing");
        return Err(Refusal { param: rule.name, message: format!("{}: {reason}", rule.name) });
    }
    Ok(())
}

/// What `surface`'s table says, as the OpenAPI operation description carries
/// it: the fields accepted and ignored, and each refusal with its reason.
pub fn describe(surface: Surface) -> String {
    let rules = table(surface);
    let always: Vec<String> =
        rules.iter().filter(|r| !r.can_refuse()).map(|r| format!("`{}`", r.name)).collect();
    let mut text = String::from(
        "Every OpenAI field is honoured, accepted as inert, or refused with a 400 naming it in `param`. A field outside this list is ignored.\n\n",
    );
    text.push_str(&format!("Inert at any value: {}.\n\n", always.join(", ")));
    text.push_str("Inert at the listed values, refused at any other (absent and `null` are always inert):\n\n");
    for rule in rules.iter().filter(|r| r.can_refuse()) {
        text.push_str(&format!(
            "- `{}`: inert at {}. Otherwise refused: {}.\n",
            rule.name,
            rule.inert_when,
            rule.refusal.expect("filtered")
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(value: JsonValue) -> Map<String, JsonValue> {
        value.as_object().expect("an object").clone()
    }

    /// Each value is inert on `surface`.
    fn accepted(surface: Surface, name: &str, values: &[JsonValue]) {
        for value in values {
            let fields = body(json!({ name: value }));
            assert_eq!(check(&fields, surface), Ok(()), "{name}: {value} must be inert on {surface:?}");
        }
    }

    /// Each value is refused on `surface`, naming `name` and saying `pointer`.
    fn refused(surface: Surface, name: &str, values: &[JsonValue], pointer: &str) {
        for value in values {
            let fields = body(json!({ name: value }));
            let refusal = check(&fields, surface).expect_err(&format!("{name}: {value} must be refused on {surface:?}"));
            assert_eq!(refusal.param, name);
            assert!(refusal.message.contains(pointer), "{name}: {value}: {}", refusal.message);
        }
    }

    #[test]
    fn n_is_inert_at_one_and_refused_above() {
        accepted(Surface::Chat, "n", &[json!(null), json!(1)]);
        refused(Surface::Chat, "n", &[json!(2), json!(8), json!("1")], "#289");
    }

    #[test]
    fn logprobs_is_inert_when_false() {
        accepted(Surface::Chat, "logprobs", &[json!(null), json!(false)]);
        refused(Surface::Chat, "logprobs", &[json!(true)], "#288");
        accepted(Surface::Chat, "top_logprobs", &[json!(null)]);
        refused(Surface::Chat, "top_logprobs", &[json!(0), json!(5)], "#288");
    }

    #[test]
    fn response_format_is_inert_only_as_text() {
        accepted(Surface::Chat, "response_format", &[json!(null), json!({ "type": "text" })]);
        refused(
            Surface::Chat,
            "response_format",
            &[
                json!({ "type": "json_object" }),
                json!({ "type": "json_schema", "json_schema": { "name": "x" } }),
                json!({ "type": "grammar" }),
                json!("text"),
            ],
            "#287",
        );
    }

    #[test]
    fn logit_bias_is_inert_empty() {
        accepted(Surface::Chat, "logit_bias", &[json!(null), json!({})]);
        refused(Surface::Chat, "logit_bias", &[json!({ "50256": -100 })], "#288");
    }

    #[test]
    fn the_defaults_an_sdk_fills_in_are_inert() {
        accepted(Surface::Chat, "parallel_tool_calls", &[json!(null), json!(true)]);
        refused(Surface::Chat, "parallel_tool_calls", &[json!(false)], "as it closes");
        accepted(Surface::Chat, "store", &[json!(null), json!(false)]);
        refused(Surface::Chat, "store", &[json!(true)], "stores no completions");
        accepted(Surface::Chat, "service_tier", &[json!(null), json!("auto"), json!("default")]);
        refused(Surface::Chat, "service_tier", &[json!("flex"), json!("priority"), json!("scale")], "one tier");
        accepted(Surface::Chat, "modalities", &[json!(null), json!(["text"])]);
        refused(Surface::Chat, "modalities", &[json!(["text", "audio"]), json!(["audio"])], "no audio");
    }

    #[test]
    fn the_always_inert_fields_accept_anything() {
        for name in ["metadata", "user", "prompt_cache_key", "safety_identifier", "verbosity"] {
            accepted(Surface::Chat, name, &[json!(null), json!("x"), json!({ "a": 1 }), json!(7)]);
        }
    }

    #[test]
    fn fields_this_server_does_not_serve_are_refused_unless_null() {
        for (name, pointer) in [
            ("audio", "audio"),
            ("prediction", "predicted"),
            ("web_search_options", "web search"),
            ("functions", "use tools"),
            ("function_call", "use tool_choice"),
            ("best_of", "use messages"),
            ("echo", "use messages"),
            ("suffix", "use messages"),
            ("prompt", "use messages"),
        ] {
            accepted(Surface::Chat, name, &[json!(null)]);
            refused(Surface::Chat, name, &[json!(true), json!({}), json!("x")], pointer);
        }
    }

    #[test]
    fn a_field_outside_the_table_is_ignored() {
        assert_eq!(check(&body(json!({ "some_future_field": { "x": 1 } })), Surface::Chat), Ok(()));
    }

    #[test]
    fn responses_takes_what_codex_sends() {
        accepted(Surface::Responses, "parallel_tool_calls", &[json!(false), json!(true)]);
        accepted(Surface::Responses, "store", &[json!(false), json!(true)]);
        accepted(Surface::Responses, "include", &[json!(["reasoning.encrypted_content"])]);
        accepted(Surface::Responses, "prompt_cache_key", &[json!("thread-1")]);
        accepted(Surface::Responses, "client_metadata", &[json!({ "a": "b" })]);
        accepted(Surface::Responses, "service_tier", &[json!("auto"), json!("flex"), json!("priority")]);
        accepted(Surface::Responses, "truncation", &[json!("disabled")]);
        accepted(Surface::Responses, "top_logprobs", &[json!(0)]);
    }

    #[test]
    fn responses_refuses_what_it_does_not_serve() {
        refused(Surface::Responses, "top_logprobs", &[json!(3)], "#288");
        refused(Surface::Responses, "truncation", &[json!("auto")], "never drops input");
        refused(Surface::Responses, "conversation", &[json!("conv_1")], "no conversations");
        refused(Surface::Responses, "prompt", &[json!({ "id": "pmpt_1" })], "prompt templates");
        refused(Surface::Responses, "context_management", &[json!([{ "type": "compaction" }])], "compaction");
    }

    #[test]
    fn the_description_names_every_refused_field_and_its_reason() {
        for surface in [Surface::Chat, Surface::Responses] {
            let text = describe(surface);
            for rule in table(surface) {
                assert!(text.contains(&format!("`{}`", rule.name)), "{surface:?}: {} missing", rule.name);
                if let Some(reason) = rule.refusal {
                    assert!(text.contains(reason), "{surface:?}: {} reason missing", rule.name);
                }
            }
        }
    }
}
