//! Frontend object extraction (artifact-02, GitHub #7): the
//! container-carried frontend set — the tokenizer, the chat template, and
//! the four auxiliary config resources.
//!
//! Confirms the `ignis-v1` §7 risk that the container carries the frontend:
//! six resource objects ride alongside the 1,319 tensors (ADR 0002 — the
//! binder consumes every object, and the frontend is read straight from
//! the container, never re-shipped with requests).
//!
//! This module is the typed extraction layer over the generic reader:
//! [`FrontendSet`] verifies all six frontend resources are present and
//! parses the two that have a format — the HuggingFace
//! [`tokenizers`](https://crates.io/crates/tokenizers)-format tokenizer
//! ([`Tokenizer`]) and the Jinja chat template ([`ChatTemplate`],
//! compiled once with minijinja) — leaving the four config resources as
//! raw host bytes.
//!
//! The API is server-agnostic: the HTTP server (server-01) consumes
//! [`FrontendSet`] through its own `TemplateProvider` seam (the dependency
//! direction is server → artifact, never the other way).

use minijinja::{Environment, Value};
use serde_json::{json, Value as JsonValue};

use crate::{fail, Object, Reader, Result};

// ---------------------------------------------------------------------------
// The six frontend resources
// ---------------------------------------------------------------------------

/// The six frontend resource base names carried in the container
/// (directory entries carry a `frontend/` scheme prefix, e.g.
/// `frontend/chat_template.jinja`).
pub const FRONTEND_RESOURCES: [&str; 6] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
];

/// The complete frontend object set carried in the `.ninfer` container:
/// a parsed [`Tokenizer`] + [`ChatTemplate`], and the four config
/// resources as raw host bytes.
pub struct FrontendSet {
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    tokenizer_config: Vec<u8>,
    generation_config: Vec<u8>,
    preprocessor_config: Vec<u8>,
    video_preprocessor_config: Vec<u8>,
}

impl std::fmt::Debug for FrontendSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The tokenizer + template are opaque (the underlying engines'
        // debug output is noise); the config byte counts are the signal.
        f.debug_struct("FrontendSet")
            .field("tokenizer_config_bytes", &self.tokenizer_config.len())
            .field("generation_config_bytes", &self.generation_config.len())
            .field("preprocessor_config_bytes", &self.preprocessor_config.len())
            .field(
                "video_preprocessor_config_bytes",
                &self.video_preprocessor_config.len(),
            )
            .finish()
    }
}

impl FrontendSet {
    /// Extract the frontend set from an open artifact.
    ///
    /// All six frontend resources must be present — a missing or
    /// ambiguous resource is a load failure (ADR 0002). The set owns
    /// everything it needs, so the reader may be dropped afterwards.
    pub fn from_reader(reader: &Reader) -> Result<Self> {
        let tokenizer_bytes = read_resource(reader, "tokenizer.json")?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
        let chat_template =
            ChatTemplate::from_bytes(&read_resource(reader, "chat_template.jinja")?)?;
        Ok(Self {
            tokenizer,
            chat_template,
            tokenizer_config: read_resource(reader, "tokenizer_config.json")?,
            generation_config: read_resource(reader, "generation_config.json")?,
            preprocessor_config: read_resource(reader, "preprocessor_config.json")?,
            video_preprocessor_config: read_resource(reader, "video_preprocessor_config.json")?,
        })
    }

    /// The parsed container-carried tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The compiled container-carried chat template.
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Raw host bytes of `tokenizer_config.json`.
    pub fn tokenizer_config(&self) -> &[u8] {
        &self.tokenizer_config
    }

    /// Raw host bytes of `generation_config.json`.
    pub fn generation_config(&self) -> &[u8] {
        &self.generation_config
    }

    /// Raw host bytes of `preprocessor_config.json`.
    pub fn preprocessor_config(&self) -> &[u8] {
        &self.preprocessor_config
    }

    /// Raw host bytes of `video_preprocessor_config.json`.
    pub fn video_preprocessor_config(&self) -> &[u8] {
        &self.video_preprocessor_config
    }
}

/// The raw bytes of the container resource with base name `base`
/// (the canonical `frontend/{base}` entry, or — when the directory uses
/// another scheme — the unique resource object named `base` or ending in
/// `/{base}`). A missing or ambiguous resource is a load failure.
fn read_resource(reader: &Reader, base: &str) -> Result<Vec<u8>> {
    if let Some(object) = reader.find(&format!("frontend/{base}")) {
        return reader.payload_at(object).map(|span| span.data.to_vec());
    }
    let matches: Vec<&Object> = reader
        .objects()
        .iter()
        .filter(|object| {
            matches!(
                object,
                Object::Resource(resource)
                    if resource.name == base
                        || resource.name.ends_with(&format!("/{base}"))
            )
        })
        .collect();
    match matches.len() {
        1 => reader.payload_at(matches[0]).map(|span| span.data.to_vec()),
        0 => Err(fail(format!(
            "frontend resource missing from artifact: {base}"
        ))),
        _ => Err(fail(format!(
            "frontend resource is ambiguous in artifact: {base}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer (HuggingFace `tokenizers` format)
// ---------------------------------------------------------------------------

/// A HuggingFace `tokenizers`-format tokenizer parsed from the
/// container's `tokenizer.json` (request-time encode/decode for the
/// server's prompt / response paths).
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Parse a tokenizer from the raw bytes of `tokenizer.json`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| fail(format!("parse tokenizer.json: {e}")))?;
        Ok(Self { inner })
    }

    /// Encode text to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let output = self
            .inner
            .encode(text, false)
            .map_err(|e| fail(format!("tokenize: {e}")))?;
        Ok(output.get_ids().to_vec())
    }

    /// Decode token ids to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| fail(format!("detokenize: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Chat template (Jinja, compiled once with minijinja)
// ---------------------------------------------------------------------------

/// The container's Jinja chat template, compiled once.
///
/// Rendering takes OpenAI-wire messages ([`ChatMessage`]) and applies the
/// template's own defaults for the variables it does not receive (tools,
/// thinking, reasoning effort, …).
pub struct ChatTemplate {
    env: Environment<'static>,
    name: &'static str,
}

impl ChatTemplate {
    const NAME: &'static str = "chat_template.jinja";

    /// Compile a chat template source. A syntax error is a load failure
    /// (the template cannot be used for any request).
    pub fn from_source(source: &str) -> Result<Self> {
        let mut env: Environment<'static> = Environment::new();
        register_template_builtins(&mut env);
        env.add_template_owned(Self::NAME, source.to_owned())
            .map_err(|e| fail(format!("compile chat template: {e}")))?;
        Ok(Self {
            env,
            name: Self::NAME,
        })
    }

    /// Compile a chat template from the raw bytes of
    /// `chat_template.jinja`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_source(
            std::str::from_utf8(bytes)
                .map_err(|_| fail("chat_template.jinja is not valid UTF-8"))?,
        )
    }

    /// Render an OpenAI-style conversation through the template.
    ///
    /// `add_generation_prompt` is set to `true` (the standard completion
    /// behavior); the remaining template variables use the template's own
    /// defaults.
    pub fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        let context = json!({
            "messages": messages.iter().map(message_to_json).collect::<Vec<_>>(),
            "add_generation_prompt": true,
        });
        let template = self
            .env
            .get_template(self.name)
            .map_err(|e| fail(format!("render chat template: {e}")))?;
        template
            .render(&context)
            .map_err(|e| fail(format!("render chat template: {e}")))
    }
}

/// The Jinja built-ins the container template relies on that minijinja
/// does not ship by default: the `raise_exception` global (the HuggingFace
/// chat-template error idiom) and the string `startswith` / `endswith`
/// methods (Jinja2 Python string methods, absent from minijinja's
/// built-in method set).
fn register_template_builtins(env: &mut Environment<'static>) {
    // The template's error idiom: abort rendering with the given message
    // (a controlled runtime abort — the template raised, not the engine).
    env.add_function(
        "raise_exception",
        |message: Value| -> std::result::Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message.as_str().unwrap_or("").to_string(),
            ))
        },
    );
    env.set_unknown_method_callback(|_state, value, method, args| match method {
        "startswith" | "endswith" => {
            let subject = value
                .as_str()
                .ok_or_else(|| minijinja::Error::from(minijinja::ErrorKind::InvalidOperation))?;
            let other = args
                .first()
                .and_then(|v| v.as_str())
                .ok_or_else(|| minijinja::Error::from(minijinja::ErrorKind::MissingArgument))?;
            let hit = match method {
                "startswith" => subject.starts_with(other),
                _ => subject.ends_with(other),
            };
            Ok(Value::from(hit))
        }
        _ => Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
    });
    // The template renders raw chat text, never HTML.
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
}

// ---------------------------------------------------------------------------
// OpenAI-wire message shapes
// ---------------------------------------------------------------------------

/// A chat message in OpenAI wire shape (what the chat template accepts).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The message role (`system`, `user`, `assistant`, `tool`).
    pub role: Role,
    /// The message content (plain text or structured content parts).
    pub content: MessageContent,
    /// Tool calls requested by an assistant message (OpenAI shape).
    pub tool_calls: Vec<ToolCall>,
    /// The model's thinking trace (assistant messages of thinking models).
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// A message with plain-text content and no tool calls / thinking.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: MessageContent::Text(content.into()),
            tool_calls: Vec::new(),
            reasoning_content: None,
        }
    }
}

/// A message role (OpenAI wire names: `system`, `user`, `assistant`,
/// `tool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// The wire name of the role.
    pub fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    /// Parse a wire-name role.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            _ => return None,
        })
    }
}

/// The content of a message: plain text, or OpenAI-style content parts
/// (text / image / video).
#[derive(Debug, Clone)]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// Structured content parts.
    Parts(Vec<ContentPart>),
}

/// A structured content part (OpenAI multimodal shape).
#[derive(Debug, Clone)]
pub enum ContentPart {
    /// A text part.
    Text(String),
    /// An image part (`image_url.url`, when the part carries a URL).
    Image { url: Option<String> },
    /// A video part (`video_url`, when the part carries a URL).
    Video { url: Option<String> },
}

/// A tool/function call of an assistant message (OpenAI shape: a
/// `function` with a name and JSON arguments).
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// The call id (OpenAI `id`), when present.
    pub id: Option<String>,
    /// The tool (function) name.
    pub name: String,
    /// The tool arguments (a JSON object, or a JSON-encoded string).
    pub arguments: JsonValue,
}

// ---------------------------------------------------------------------------
// Message → template context (Jinja sees plain JSON, OpenAI wire shape)
// ---------------------------------------------------------------------------

/// The JSON a [`ChatMessage`] presents to the template (OpenAI wire
/// shape: `role`, `content`, `reasoning_content`, `tool_calls`).
fn message_to_json(message: &ChatMessage) -> JsonValue {
    let mut object = serde_json::Map::new();
    object.insert("role".to_owned(), json!(message.role.name()));
    object.insert("content".to_owned(), content_to_json(&message.content));
    if let Some(reasoning) = &message.reasoning_content {
        object.insert("reasoning_content".to_owned(), json!(reasoning));
    }
    if !message.tool_calls.is_empty() {
        let calls: Vec<JsonValue> = message
            .tool_calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                })
            })
            .collect();
        object.insert("tool_calls".to_owned(), json!(calls));
    }
    JsonValue::Object(object)
}

/// The JSON of a message's [`MessageContent`] (plain text, or the
/// OpenAI-style content-part list the template's vision branches match
/// against: `type` + `image_url` / `video_url` keys).
fn content_to_json(content: &MessageContent) -> JsonValue {
    match content {
        MessageContent::Text(text) => json!(text),
        MessageContent::Parts(parts) => json!(parts
            .iter()
            .map(|part| match part {
                ContentPart::Text(text) => json!({
                    "type": "text",
                    "text": text,
                }),
                ContentPart::Image { url } => json!({
                    "type": "image",
                    "image_url": url,
                }),
                ContentPart::Video { url } => json!({
                    "type": "video",
                    "video_url": url,
                }),
            })
            .collect::<Vec<_>>()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::FixtureObject;

    /// A minimal valid HuggingFace `tokenizers` tokenizer JSON
    /// (WordLevel model with a fixed four-word vocab — a round-trip of
    /// that text is exact and its id stream is predictable).
    const FIXTURE_TOKENIZER_JSON: &str = r#"{"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":{"the":0,"quick":1,"brown":2,"fox":3},"unk_token":"fox"}}"#;

    /// A fixture chat template (minijinja-compatible, uses the template
    /// variables `FrontendSet::from_reader` sets).
    const FIXTURE_TEMPLATE: &str =
        "{%- for m in messages -%}{{ m.role }}={{ m.content }};{%- endfor -%}{{- \"END\" if add_generation_prompt else \"\" -}}";

    const TOKENIZER_CONFIG: &[u8] = b"{\"tokenizer_class\": \"Whitespace\"}\n";
    const GENERATION_CONFIG: &[u8] = b"{\"max_new_tokens\": 8192, \"do_sample\": false}";
    const PREPROCESSOR_CONFIG: &[u8] = b"{\"size\": {\"shortestedge\": 672}}";
    const VIDEO_PREPROCESSOR_CONFIG: &[u8] = b"{\"max_pixels\": 1048576}";

    /// A six-resource frontend fixture (payload regions are the exact
    /// resource bytes, at their cumulative offsets).
    fn frontend_fixture(omit: Option<&str>) -> (Vec<FixtureObject>, Vec<u8>) {
        let resources: [(&'static str, &[u8]); 6] = [
            ("frontend/tokenizer.json", FIXTURE_TOKENIZER_JSON.as_bytes()),
            ("frontend/tokenizer_config.json", TOKENIZER_CONFIG),
            ("frontend/chat_template.jinja", FIXTURE_TEMPLATE.as_bytes()),
            ("frontend/generation_config.json", GENERATION_CONFIG),
            ("frontend/preprocessor_config.json", PREPROCESSOR_CONFIG),
            (
                "frontend/video_preprocessor_config.json",
                VIDEO_PREPROCESSOR_CONFIG,
            ),
        ];
        let mut objects = Vec::new();
        let mut payload = Vec::new();
        for (name, bytes) in resources {
            if let Some(omit) = omit {
                if name.ends_with(omit) {
                    continue;
                }
            }
            let offset = payload.len() as u64;
            objects.push(FixtureObject::Resource {
                name,
                encoding: "raw-bytes-v1",
                offset,
                bytes: bytes.len() as u64,
            });
            payload.extend_from_slice(bytes);
        }
        (objects, payload)
    }

    /// Write a five- or six-resource frontend fixture, open it, extract
    /// the frontend set, and hand it (plus the reader) to `f` — the
    /// fixture file is removed only after the reader has released its
    /// handles (the Windows direct-I/O handle must be closed first).
    fn with_frontend_set(omit: Option<&str>, f: impl FnOnce(&Reader, &FrontendSet)) {
        let (objects, payload) = frontend_fixture(omit);
        let fixture =
            crate::fixture::write_fixture(&objects, &payload, "frontend-set").expect("fixture");
        let reader = crate::Reader::open(&fixture.path).expect("open fixture artifact");
        let set = FrontendSet::from_reader(&reader).expect("frontend set");
        f(&reader, &set);
        // Explicit drop order: the reader releases its file handles before
        // the fixture removes the file (the removal is a no-op while the
        // Windows direct-I/O handle is still open).
        drop(set);
        drop(reader);
        drop(fixture);
    }

    #[test]
    fn frontend_set_round_trips_all_six_resources() {
        with_frontend_set(None, |_, set| {
            // The four config resources round-trip byte-for-byte.
            assert_eq!(set.tokenizer_config(), TOKENIZER_CONFIG);
            assert_eq!(set.generation_config(), GENERATION_CONFIG);
            assert_eq!(set.preprocessor_config(), PREPROCESSOR_CONFIG);
            assert_eq!(set.video_preprocessor_config(), VIDEO_PREPROCESSOR_CONFIG);
        });
    }

    #[test]
    fn frontend_set_fails_when_a_resource_is_missing() {
        let (objects, payload) = frontend_fixture(Some("video_preprocessor_config.json"));
        let fixture =
            crate::fixture::write_fixture(&objects, &payload, "frontend-missing").expect("fixture");
        let reader = crate::Reader::open(&fixture.path).expect("open fixture artifact");
        let err = FrontendSet::from_reader(&reader).expect_err("missing resource");
        assert!(
            err.to_string().contains("video_preprocessor_config.json"),
            "{err}"
        );
    }

    #[test]
    fn tokenizer_round_trips_a_known_string() {
        with_frontend_set(None, |_, set| {
            let ids = set
                .tokenizer()
                .encode("the quick brown fox")
                .expect("encode");
            assert_eq!(ids.len(), 4, "word-level model: one token per word");
            let text = set.tokenizer().decode(&ids).expect("decode");
            assert_eq!(text, "the quick brown fox");
        });
    }

    #[test]
    fn template_renders_a_two_message_conversation() {
        with_frontend_set(None, |_, set| {
            let messages = [
                ChatMessage::text(Role::User, "hello world"),
                ChatMessage::text(Role::Assistant, "hi there"),
            ];
            let prompt = set.chat_template().render(&messages).expect("render");
            // Property assertions (not an exact string): every message
            // renders under its own role marker, and the generation prompt
            // closes the conversation.
            assert!(prompt.contains("user=hello world;"), "{prompt}");
            assert!(prompt.contains("assistant=hi there;"), "{prompt}");
            assert!(prompt.ends_with("END"), "{prompt}");
        });
    }

    #[test]
    fn template_renders_structured_content_parts() {
        let template = ChatTemplate::from_source(FIXTURE_TEMPLATE).expect("compile");
        let messages = [ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text("what is in this picture?".to_string()),
                ContentPart::Image {
                    url: Some("http://example/x.png".to_owned()),
                },
            ]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        }];
        let prompt = template.render(&messages).expect("render");
        assert!(prompt.contains("what is in this picture?"), "{prompt}");
        assert!(prompt.ends_with("END"), "{prompt}");
    }

    #[test]
    fn template_engine_registers_the_missing_jinja_builtins() {
        // `startswith` / `endswith` are string methods minijinja does not
        // ship; the registration makes the container template's usage
        // resolve.
        let template = ChatTemplate::from_source(
            "{% if 'abc'.startswith('ab') and 'abc'.endswith('bc') %}ok{% else %}no{% endif %}",
        )
        .expect("compile");
        assert_eq!(template.render(&[]).expect("render"), "ok");
    }

    #[test]
    fn template_engine_provides_the_tojson_filter() {
        // The container template serializes tool definitions / tool-call
        // arguments with `|tojson` — the filter is provided by the
        // `json` feature, not a minijinja default (guards the
        // Cargo.toml feature wiring).
        let template = ChatTemplate::from_source("{{ {'a': 1} | tojson }}").expect("compile");
        let out = template.render(&[]).expect("render");
        assert!(out.contains("1"), "tojson output must carry the value: {out}");
    }

    #[test]
    fn raise_exception_aborts_the_render() {
        let template =
            ChatTemplate::from_source("{{- raise_exception('No messages provided.') -}}")
                .expect("compile");
        let err = template.render(&[]).expect_err("the template must abort");
        assert!(err.to_string().contains("No messages provided."), "{err}");
    }

    #[test]
    fn a_template_syntax_error_is_a_load_failure() {
        assert!(ChatTemplate::from_source("{{ oops").is_err());
    }

    #[test]
    fn empty_conversation_renders_only_the_generation_prompt() {
        // The fixture template's contract (mirrored by the real Qwen
        // template, which *raises* on an empty message list): the
        // context always carries `messages` — even when empty — so an
        // empty conversation renders as just the generation prompt.
        let template = ChatTemplate::from_source(FIXTURE_TEMPLATE).expect("compile");
        let prompt = template.render(&[]).expect("render");
        assert_eq!(prompt, "END");
    }
}
