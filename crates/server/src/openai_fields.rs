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
    /// `POST /v1/images/generations`'s body and `POST /v1/images/edits`'s
    /// form (spec image-model/05), whose text parts are read as JSON values.
    Images,
    /// `POST /v1/audio/transcriptions`'s form (spec speech/02), whose text
    /// parts are read as JSON values.
    Transcriptions,
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

fn null_or_zero_number(value: &JsonValue) -> bool {
    value.is_null() || value.as_f64() == Some(0.0)
}

fn null_or_empty_array(value: &JsonValue) -> bool {
    value.is_null() || value.as_array().is_some_and(Vec::is_empty)
}



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
    // `include` is not here: the handler reads it, for
    // `reasoning.encrypted_content` (ADR 0052), and takes any other value.
    FieldRule { name: "stream_options", inert: always, inert_when: ALWAYS, refusal: None },
    // Counts hosted-tool calls, and this server runs none.
    FieldRule { name: "max_tool_calls", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "client_metadata", inert: always, inert_when: ALWAYS, refusal: None },
];

/// A member of a field the handler reads (`tools`) that is accepted at any
/// value and inert: it asks for something this server does not do yet, and
/// changes nothing it does. The template still renders it as sent.
pub struct InertMember {
    /// Where it sits in the body, as the document names it.
    pub path: &'static str,
    /// What it would ask for, and the issue that will serve it.
    pub why: &'static str,
}

/// `strict` on a function tool (spec #287): the call's arguments are the
/// model's own text, parsed as they come.
const TOOL_STRICT: &str = "a call's arguments are not held to the function's `parameters` yet (GitHub #582)";

/// The chat body's inert members.
pub const CHAT_MEMBERS: &[InertMember] = &[InertMember { path: "tools[].function.strict", why: TOOL_STRICT }];

/// The Responses body's inert members: its function tools are flat.
pub const RESPONSES_MEMBERS: &[InertMember] = &[InertMember { path: "tools[].strict", why: TOOL_STRICT }];

/// The inert members of `surface`'s body.
pub fn members(surface: Surface) -> &'static [InertMember] {
    match surface {
        Surface::Chat => CHAT_MEMBERS,
        Surface::Responses => RESPONSES_MEMBERS,
        Surface::Images | Surface::Transcriptions => &[],
    }
}

/// The Images API's fields past what the image routes read
/// (`crate::images::routes`): `response_format`, `output_format`,
/// `background`, `size`, `n` and `mask` are refused where they are read.
pub const IMAGES: &[FieldRule] = &[
    FieldRule {
        name: "partial_images",
        inert: null_or_zero,
        inert_when: "`0`",
        refusal: Some("partial images are not sent, each image comes whole in its completed event; send 0 or omit partial_images"),
    },
    // One image model, at one quality and in one style.
    FieldRule { name: "quality", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "style", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "moderation", inert: always, inert_when: ALWAYS, refusal: None },
    // PNG only, which takes no compression level.
    FieldRule { name: "output_compression", inert: always, inert_when: ALWAYS, refusal: None },
    // An edit is conditioned on its whole images, at the size the condition
    // budget leaves them.
    FieldRule { name: "input_fidelity", inert: always, inert_when: ALWAYS, refusal: None },
    FieldRule { name: "user", inert: always, inert_when: ALWAYS, refusal: None },
];

/// The transcription form's fields past what the route reads
/// (`crate::speech::route`): `file`, `model`, `prompt`, `keywords[]`,
/// `language`, `languages[]`, `response_format`, `timestamp_granularities[]`
/// and `stream` are honoured or refused where they are read. The form's text
/// parts arrive as the JSON values a body would carry, the array fields under
/// their `[]` names.
pub const TRANSCRIPTIONS: &[FieldRule] = &[
    FieldRule {
        name: "temperature",
        inert: null_or_zero_number,
        inert_when: "`0`",
        refusal: Some("this server decodes greedily; send 0 or omit temperature"),
    },
    FieldRule {
        name: "chunking_strategy",
        inert: null,
        inert_when: "`null`",
        refusal: Some("the clip is transcribed as one block, with no automatic split in this version; omit chunking_strategy or send null"),
    },
    FieldRule {
        name: "include[]",
        inert: null_or_empty_array,
        inert_when: "empty",
        refusal: Some("log probabilities are not served (`logprobs`); omit include[]"),
    },
    FieldRule {
        name: "known_speaker_names[]",
        inert: null_or_empty_array,
        inert_when: "empty",
        refusal: Some("this server does not diarize; omit known_speaker_names[]"),
    },
    FieldRule {
        name: "known_speaker_references[]",
        inert: null_or_empty_array,
        inert_when: "empty",
        refusal: Some("this server does not diarize; omit known_speaker_references[]"),
    },
];

/// The table for `surface`.
pub fn table(surface: Surface) -> &'static [FieldRule] {
    match surface {
        Surface::Chat => CHAT,
        Surface::Responses => RESPONSES,
        Surface::Images => IMAGES,
        Surface::Transcriptions => TRANSCRIPTIONS,
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
    let members = members(surface);
    if !members.is_empty() {
        text.push_str("\nInert at any value inside a field the server reads:\n\n");
        for member in members {
            text.push_str(&format!("- `{}`: {}.\n", member.path, member.why));
        }
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

    /// Review-322b L3: the image routes take what OpenAI's SDKs send and
    /// refuse partial images, which are never sent.
    #[test]
    fn images_refuse_partial_images_and_take_the_rest() {
        accepted(Surface::Images, "partial_images", &[json!(null), json!(0)]);
        refused(Surface::Images, "partial_images", &[json!(1), json!(3), json!("2")], "comes whole");
        for name in ["quality", "style", "moderation", "output_compression", "input_fidelity", "user"] {
            accepted(Surface::Images, name, &[json!(null), json!("high"), json!(80), json!({ "a": 1 })]);
        }
    }

    #[test]
    fn the_description_names_every_refused_field_and_its_reason() {
        for surface in [Surface::Chat, Surface::Responses, Surface::Images] {
            let text = describe(surface);
            for rule in table(surface) {
                assert!(text.contains(&format!("`{}`", rule.name)), "{surface:?}: {} missing", rule.name);
                if let Some(reason) = rule.refusal {
                    assert!(text.contains(reason), "{surface:?}: {} reason missing", rule.name);
                }
            }
        }
    }

    /// Spec #287: `strict` on a function tool is declared inert where the
    /// document lists the fields, on both completion surfaces, naming #582.
    #[test]
    fn the_description_declares_a_function_tools_strict_inert() {
        for (surface, path) in [(Surface::Chat, "`tools[].function.strict`"), (Surface::Responses, "`tools[].strict`")] {
            let text = describe(surface);
            let line = text.lines().find(|line| line.contains(path)).unwrap_or_else(|| panic!("{surface:?}: {text}"));
            assert!(line.contains("#582") && line.contains("`parameters`"), "{surface:?}: {line}");
        }
        assert!(!describe(Surface::Images).contains("strict"));
    }
}
