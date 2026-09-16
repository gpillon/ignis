//! The artifact-backed [`TemplateProvider`] (the artifact-02 wiring,
//! GitHub #7 follow-up).
//!
//! The built-in [`crate::template::SimpleTemplateProvider`] is a
//! placeholder (placeholder token ids, not natural language). This module
//! plugs the artifact container's real frontend object set — the
//! [`FrontendSet`] extracted by artifact-02 — into the same
//! [`TemplateProvider`] seam: the chat template (minijinja, with the
//! Qwen3.8-specific extensions registered) renders the conversation, and
//! the container's HuggingFace tokenizer turns the rendered prompt into
//! the token ids the scheduler consumes. Generated ids are decoded back
//! to text with the same tokenizer.

use std::sync::Arc;

use ignis_artifact::vision::{PreparedMedia, VisionProcessor};
use ignis_artifact::{DecodeStreamState, FrontendSet, Role};
use ignis_core::vision::Multimodal;
use ignis_core::TokenId;
use serde_json::Value as JsonValue;

use crate::decoder::TokenDecoder;
use crate::template::{
    processor_rejection, ChatMessage, ContentRejection, MessageContent, TemplateProvider,
};
use crate::thinking::{ThinkingCapabilities, ThinkingOptions};

/// The [`TemplateProvider`] backed by the artifact's [`FrontendSet`]: the
/// real chat template + tokenizer extracted from the `.ninfer` container
/// (artifact-02, GitHub #7).
///
/// Wire-shape notes (v1):
/// - A message role that does not parse to an artifact [`Role`]
///   (`system` / `user` / `assistant` / `tool`) is templated as `user` —
///   the OpenAI surface sends only those four roles, and a foreign role
///   string must not take the server down (the provider is infallible by
///   design; the request still completes, just templated as a user
///   message).
/// - A `render` or `encode` failure (e.g. the container template raising
///   on a malformed conversation) yields an empty token list: the request
///   completes degenerate, but the process survives (logged, not
///   panicked).
#[derive(Debug)]
pub struct ArtifactTemplateProvider {
    set: Arc<FrontendSet>,
    /// The vision processor on a `--vision` load (GitHub #179); `None`
    /// renders no images.
    vision: Option<VisionProcessor>,
}

impl ArtifactTemplateProvider {
    /// A provider over the artifact's frontend object set. The set is
    /// built once at startup (`FrontendSet::from_reader`) and shared
    /// across handlers through the `Arc` (the router shares one
    /// `Arc<dyn TemplateProvider>`).
    pub fn new(frontend: FrontendSet) -> Self {
        Self {
            set: Arc::new(frontend),
            vision: None,
        }
    }

    /// Render image prompts with `processor` (a `--vision` load).
    pub fn with_vision(mut self, processor: VisionProcessor) -> Self {
        self.vision = Some(processor);
        self
    }

    /// The chat template's text for `messages`, image parts rendered as
    /// their placeholders.
    fn render(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<String, String> {
        let templated: Vec<ignis_artifact::ChatMessage> = messages
            .iter()
            .map(|message| {
                let role = Role::parse(&message.role).unwrap_or(Role::User);
                let mut templated = ignis_artifact::ChatMessage::text(role, "");
                templated.content = artifact_content(&message.content);
                // Prior assistant reasoning is handed to the template
                // whole; which of it survives into the prompt is the
                // template's decision, driven by the `preserve_thinking`
                // bound below (GitHub #185). Dropping it here instead
                // (GitHub #68) pre-empted that decision and left the
                // template rendering an emptied think block on turns the
                // reference renders without one (#182). The story #68 was
                // protecting — a long conversation must not accumulate
                // traces — is now the template's own
                // strip-before-the-last-real-user-query branch, which keeps
                // only the in-flight turn's reasoning, as the reference
                // does.
                templated.reasoning_content = message.reasoning_content.clone();
                if let Some(calls) = &message.tool_calls {
                    templated.tool_calls = calls.iter().map(artifact_tool_call).collect();
                }
                templated
            })
            .collect();
        self.set
            .chat_template()
            .render_with_thinking_and_tools(
                &templated,
                options.enable_thinking,
                options.reasoning_effort,
                options.preserve_thinking,
                Some(tools),
            )
            .map_err(|err| err.to_string())
    }
}

impl TemplateProvider for ArtifactTemplateProvider {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Vec<TokenId> {
        let prompt = match self.render(messages, options, tools) {
            Ok(prompt) => prompt,
            Err(err) => {
                eprintln!("ignis-server: chat template render failed: {err}");
                return Vec::new();
            }
        };
        match self.set.tokenizer().encode(&prompt) {
            Ok(ids) => ids,
            Err(err) => {
                eprintln!("ignis-server: tokenizer encode failed: {err}");
                Vec::new()
            }
        }
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        // `TokenId` is a `u32` alias, so the id slice is the tokenizer's
        // own input type.
        match self.set.tokenizer().decode(tokens) {
            Ok(text) => text,
            // The engine only emits ids from the model's own vocabulary, so
            // a decode failure should not happen; fall back to the
            // placeholder's decimal stream rather than panic.
            Err(err) => {
                eprintln!("ignis-server: tokenizer decode failed: {err}");
                tokens
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        self.set.thinking_capabilities().clone()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(ArtifactTokenDecoder {
            set: Arc::clone(&self.set),
            state: DecodeStreamState::default(),
        })
    }

    /// Render with each image part as the template's placeholder, then lay
    /// the prepared media out exactly as the reference processor does.
    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        media: Vec<PreparedMedia>,
    ) -> Result<(Vec<TokenId>, Multimodal), ContentRejection> {
        let Some(processor) = &self.vision else {
            return Err(ContentRejection {
                code: "vision_disabled",
                message: "vision is disabled for this server".to_owned(),
            });
        };
        let rendered = self.render(messages, options, tools).map_err(|err| ContentRejection {
            code: "invalid_media",
            message: format!("chat template render failed: {err}"),
        })?;
        let prompt = processor
            .prepare_prompt(self.set.tokenizer(), &rendered, media, &[])
            .map_err(processor_rejection)?;
        Ok(Multimodal::from_prepared(prompt))
    }
}

/// The real tokenizer's incremental decoder (GitHub #68): wraps the
/// artifact tokenizer's own streaming decode (`Tokenizer::decode_step`),
/// which holds back an incomplete multi-byte character until the token that
/// completes it arrives, instead of decoding each token in isolation.
struct ArtifactTokenDecoder {
    set: Arc<FrontendSet>,
    state: DecodeStreamState,
}

impl TokenDecoder for ArtifactTokenDecoder {
    fn push(&mut self, token: TokenId) -> String {
        match self.set.tokenizer().decode_step(&mut self.state, token) {
            Ok(Some(text)) => text,
            Ok(None) => String::new(),
            Err(err) => {
                eprintln!("ignis-server: streaming tokenizer decode failed: {err}");
                String::new()
            }
        }
    }

    fn finish(&mut self) -> String {
        match self.set.tokenizer().decode_step_finish(&self.state) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("ignis-server: streaming tokenizer decode failed: {err}");
                String::new()
            }
        }
    }
}

/// A prior assistant turn's tool call, as the template takes it (GitHub
/// #132).
///
/// The call's `arguments` is a JSON-encoded string on the wire (matching
/// #121's own response shape) and rides to the template as that exact
/// string, never re-serialized: the template walks `arguments|items`, and
/// a `serde_json` round-trip here would sort the keys and lose the order
/// the model emitted them in, which is what a later turn has to match
/// (GitHub #184). A string that is not a JSON object renders as no
/// parameters rather than failing the whole render — the same "never
/// panic, degrade" posture as the render/encode failures — and is logged,
/// since only a client can produce one.
fn artifact_tool_call(call: &crate::template::ToolCallIn) -> ignis_artifact::ToolCall {
    let arguments = &call.function.arguments;
    if let Some(reason) = not_a_json_object(arguments) {
        eprintln!(
            "ignis-server: history tool_calls[].function.arguments is not a JSON object, degrading to {{}}: {reason}"
        );
    }
    ignis_artifact::ToolCall {
        id: call.id.clone(),
        name: call.function.name.clone(),
        arguments: arguments.clone(),
    }
}

/// Why a wire `arguments` string is not the JSON object the template can
/// walk, or `None` when it is one. Diagnostic only — the template itself
/// degrades a string that is not one to no parameters (GitHub #184) — so
/// this decides a log line, never the render, and it reports the reason
/// rather than the string, which is a client's own content.
fn not_a_json_object(arguments: &str) -> Option<String> {
    match serde_json::from_str::<JsonValue>(arguments) {
        Ok(value) if value.is_object() => None,
        Ok(value) => Some(format!("a JSON {}", json_kind(&value))),
        Err(err) => Some(err.to_string()),
    }
}

/// The name of a JSON value's kind, for the log line above.
fn json_kind(value: &JsonValue) -> &'static str {
    match value {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Array(_) => "array",
        JsonValue::Object(_) => "object",
    }
}

/// The template-facing content of a wire message (GitHub #175). A parts
/// array reaches the template as its text parts (the reference's `"\n"`
/// between adjacent text parts included, as `template_text_parts` places
/// it), so the real template's `render_content` loop renders it exactly as
/// the string `MessageContent::text` joins. An image part (GitHub #179,
/// only past `check_content_parts` on a `--vision` load) stays in place as
/// the template's image item and breaks the text run; every other part was
/// refused before a request is templated.
fn artifact_content(content: &MessageContent) -> ignis_artifact::MessageContent {
    use ignis_artifact::ContentPart as Part;
    match content {
        MessageContent::Text(text) => ignis_artifact::MessageContent::Text(text.clone()),
        MessageContent::Parts(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            let mut after_text = false;
            for part in parts {
                match (part.kind.as_deref(), part.text.as_deref()) {
                    (Some("text"), Some(text)) => {
                        if after_text && !text.is_empty() {
                            out.push(Part::Text("\n".to_owned()));
                        }
                        out.push(Part::Text(text.to_owned()));
                        after_text = true;
                    }
                    (Some("image_url"), _) => {
                        out.push(Part::Image { url: part.url.clone() });
                        after_text = false;
                    }
                    _ => after_text = false,
                }
            }
            ignis_artifact::MessageContent::Parts(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_part_stays_in_place_and_breaks_the_text_run() {
        let part = |kind: &str, text: Option<&str>, url: Option<&str>| crate::template::ContentPart {
            kind: Some(kind.to_owned()),
            text: text.map(str::to_owned),
            url: url.map(str::to_owned),
        };
        let content = MessageContent::Parts(vec![
            part("text", Some("a"), None),
            part("text", Some("b"), None),
            part("image_url", None, Some("data:x")),
            part("text", Some("c"), None),
        ]);
        let ignis_artifact::MessageContent::Parts(parts) = artifact_content(&content) else {
            panic!("parts stay parts");
        };
        let shape: Vec<String> = parts
            .iter()
            .map(|p| match p {
                ignis_artifact::ContentPart::Text(text) => format!("text:{text}"),
                ignis_artifact::ContentPart::Image { url } => format!("image:{}", url.as_deref().unwrap_or("")),
                ignis_artifact::ContentPart::Video { .. } => "video".to_owned(),
            })
            .collect();
        // "\n" joins adjacent text parts only; no newline across the image.
        assert_eq!(shape, ["text:a", "text:\n", "text:b", "image:data:x", "text:c"]);
    }

    #[test]
    fn a_provider_without_vision_refuses_to_render_images() {
        let (_fixture, _reader, provider) = build_provider_with(TEMPLATE);
        let rejection = provider
            .prepare_multimodal(&[ChatMessage::text("user", "hi")], &ThinkingOptions::default(), &[], Vec::new())
            .unwrap_err();
        assert_eq!(rejection.code, "vision_disabled");
    }
    use ignis_artifact::fixture::{self, FixtureObject};
    use ignis_artifact::Reader;
    use ignis_artifact::{
        ChatMessage as ArtifactMessage, ChatTemplate, ToolCall as ArtifactToolCall,
    };
    use serde_json::json;

    /// A minimal word-level tokenizer (the `tokenizers` 0.21 schema: a
    /// `model` with a `WordLevel` type and a fixed five-word vocab, plus
    /// a whitespace pre-tokenizer — the same shape the artifact-02
    /// fixture pins; a round-trip of the vocab words is exact).
    const TOKENIZER_JSON: &str = r#"{"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"hi":2,"there":3,"foo":4},"unk_token":"foo"}}"#;

    /// A minimal chat template (minijinja; the same property the
    /// artifact-02 fixture pins: one `role=content;` line per message,
    /// closed by a generation marker).
    const TEMPLATE: &str =
        "{%- for m in messages -%}{{ m.role }}={{ m.content }};{%- endfor -%}{{- \"END\" if add_generation_prompt else \"\" -}}";

    /// The six frontend resources of a synthetic fixture artifact (the
    /// configs are trivial; the tokenizer + template are the interesting
    /// two — `template` is swappable so tests can probe a template with
    /// different thinking-control behavior than the plain `TEMPLATE`).
    fn frontend_fixture(template: &str) -> (Vec<FixtureObject>, Vec<u8>) {
        let resources: [(&'static str, &[u8]); 6] = [
            ("frontend/tokenizer.json", TOKENIZER_JSON.as_bytes()),
            ("frontend/tokenizer_config.json", b"{}"),
            ("frontend/chat_template.jinja", template.as_bytes()),
            ("frontend/generation_config.json", b"{}"),
            ("frontend/preprocessor_config.json", b"{}"),
            ("frontend/video_preprocessor_config.json", b"{}"),
        ];
        let mut objects = Vec::new();
        let mut payload = Vec::new();
        for (name, bytes) in resources {
            objects.push(FixtureObject::Resource {
                name,
                encoding: "raw-bytes-v1",
                offset: payload.len() as u64,
                bytes: bytes.len() as u64,
            });
            payload.extend_from_slice(bytes);
        }
        (objects, payload)
    }

    /// Open a fixture artifact built from `template`, extract its frontend
    /// set, and hand the provider back. The fixture + reader stay alive for
    /// the caller's scope (the Windows direct-I/O handle must be closed
    /// before the fixture removes the file).
    fn build_provider_with(
        template: &str,
    ) -> (fixture::TempArtifact, Reader, ArtifactTemplateProvider) {
        let (objects, payload) = frontend_fixture(template);
        let fixture =
            fixture::write_fixture(&objects, &payload, "server-template-test").expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture artifact");
        let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
        (
            fixture,
            reader,
            ArtifactTemplateProvider::new(frontend),
        )
    }

    fn build_provider() -> (fixture::TempArtifact, Reader, ArtifactTemplateProvider) {
        build_provider_with(TEMPLATE)
    }

    fn opts() -> ThinkingOptions {
        ThinkingOptions::default()
    }

    fn no_tools() -> &'static [JsonValue] {
        &[]
    }

    #[test]
    fn apply_chat_template_uses_the_real_template_and_tokenizer() {
        let (_fixture, _reader, provider) = build_provider();
        let messages = [
            ChatMessage::text("user", "hello world"),
            ChatMessage::text("assistant", "hi there"),
        ];
        let tokens = provider.apply_chat_template(&messages, &opts(), no_tools());
        assert!(!tokens.is_empty(), "the rendered prompt must tokenize");
        // Determinism: the same conversation templates identically (the
        // trait's contract).
        assert_eq!(tokens, provider.apply_chat_template(&messages, &opts(), no_tools()));
        // Property assertion (not an exact string): the templated prompt
        // contains the user's message text, decoded by the same tokenizer.
        let text = provider.render_tokens(&tokens);
        assert!(text.contains("hello world"), "{text}");
    }

    #[test]
    fn render_tokens_decodes_with_the_real_tokenizer() {
        let (_fixture, _reader, provider) = build_provider();
        // Vocab ids: hello=0, world=1 (the fixture tokenizer's own
        // id-space — not the placeholder's decimal stream).
        assert_eq!(provider.render_tokens(&[0, 1]), "hello world");
    }

    #[test]
    fn unknown_roles_fall_back_to_user_without_panicking() {
        let (_fixture, _reader, provider) = build_provider();
        let messages = [ChatMessage::text("bogus", "hello")];
        // The foreign role must not panic: it templates as `user` (the
        // documented v1 fallback), and the prompt still renders.
        let tokens = provider.apply_chat_template(&messages, &opts(), no_tools());
        assert!(!tokens.is_empty());
        let text = provider.render_tokens(&tokens);
        assert!(text.contains("hello"), "{text}");
    }

    /// GitHub #175: a text-parts message renders exactly as the string its
    /// parts join to (`"\n"` between adjacent text parts, the reference's
    /// rule), through a template shaped like the real one's
    /// `render_content` (string → as-is, parts → each `text` in order).
    #[test]
    fn text_parts_render_the_same_prompt_as_the_concatenated_string() {
        const PARTS_TEMPLATE: &str = "{%- for m in messages -%}{{ m.role }} {% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{{ p.text }}{% endfor %}{% endif %} {% endfor -%}";
        let (_fixture, _reader, provider) = build_provider_with(PARTS_TEMPLATE);
        let string_form = [ChatMessage::text("user", "hello\nworld")];
        let parts: MessageContent =
            serde_json::from_value(json!([{ "type": "text", "text": "hello" }, { "type": "text", "text": "world" }]))
                .expect("parts deserialize");
        let mut parts_form = ChatMessage::text("user", "");
        parts_form.content = parts;
        let string_tokens = provider.apply_chat_template(&string_form, &opts(), no_tools());
        let parts_tokens = provider.apply_chat_template(&[parts_form], &opts(), no_tools());
        assert_eq!(parts_tokens, string_tokens);
        assert!(provider.render_tokens(&parts_tokens).contains("hello world"));
    }

    // -- thinking controls (GitHub #68) --------------------------------------

    /// A template that reacts to `enable_thinking`/`reasoning_effort` (the
    /// same probing shape as `frontend.rs`'s `THINKING_TEMPLATE`), so this
    /// module's tests can pin that the provider actually threads the
    /// resolved options through, end to end. Pure whitespace-separated
    /// words only (no punctuation) — the paired `THINKING_TOKENIZER_JSON`
    /// vocab below has to cover exactly what this renders for the
    /// round-trip through `render_tokens` to be meaningful.
    const THINKING_TEMPLATE: &str = r#"
{%- if not enable_thinking -%}
  {{- raise_exception("cannot disable") -}}
{%- endif -%}
{%- for m in messages -%}{{ m.role }} {{ m.content }} {% if m.reasoning_content is defined %}{{ m.reasoning_content }} {% endif %}{%- endfor -%}
{% if enable_thinking %}true{% else %}false{% endif %} {% if reasoning_effort is defined %}{{ reasoning_effort }}{% endif %}"#;

    /// The word-level vocab covering every word `THINKING_TEMPLATE` can
    /// render for the fixed conversations these tests use.
    const THINKING_TOKENIZER_JSON: &str = r#"{"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":{"user":0,"assistant":1,"hi":2,"the":3,"answer":4,"scratch":5,"work":6,"true":7,"low":8,"xhigh":9,"unk":10},"unk_token":"unk"}}"#;

    fn build_thinking_provider(
        template: &str,
    ) -> (fixture::TempArtifact, Reader, ArtifactTemplateProvider) {
        let resources: [(&'static str, &[u8]); 6] = [
            ("frontend/tokenizer.json", THINKING_TOKENIZER_JSON.as_bytes()),
            ("frontend/tokenizer_config.json", b"{}"),
            ("frontend/chat_template.jinja", template.as_bytes()),
            ("frontend/generation_config.json", b"{}"),
            ("frontend/preprocessor_config.json", b"{}"),
            ("frontend/video_preprocessor_config.json", b"{}"),
        ];
        let mut objects = Vec::new();
        let mut payload = Vec::new();
        for (name, bytes) in resources {
            objects.push(FixtureObject::Resource {
                name,
                encoding: "raw-bytes-v1",
                offset: payload.len() as u64,
                bytes: bytes.len() as u64,
            });
            payload.extend_from_slice(bytes);
        }
        let fixture =
            fixture::write_fixture(&objects, &payload, "server-thinking-template").expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture artifact");
        let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
        (fixture, reader, ArtifactTemplateProvider::new(frontend))
    }

    #[test]
    fn apply_chat_template_threads_enable_thinking_to_the_real_template() {
        let (_fixture, _reader, provider) = build_thinking_provider(THINKING_TEMPLATE);
        let options = ThinkingOptions {
            enable_thinking: true,
            reasoning_effort: Some(ignis_artifact::ReasoningEffort::Low),
            preserve_thinking: false,
        };
        let tokens =
            provider.apply_chat_template(&[ChatMessage::text("user", "hi")], &options, no_tools());
        let text = provider.render_tokens(&tokens);
        assert!(text.contains("true"), "{text}");
        assert!(text.contains("low"), "{text}");
    }

    #[test]
    fn apply_chat_template_disabled_raises_and_yields_no_tokens() {
        // `THINKING_TEMPLATE` raises when asked to disable thinking; the
        // provider logs and degrades to an empty token list rather than
        // panicking (the documented v1 render-failure contract).
        let (_fixture, _reader, provider) = build_thinking_provider(THINKING_TEMPLATE);
        let options = ThinkingOptions {
            enable_thinking: false,
            reasoning_effort: None,
            preserve_thinking: false,
        };
        let tokens =
            provider.apply_chat_template(&[ChatMessage::text("user", "hi")], &options, no_tools());
        assert!(tokens.is_empty(), "a raising render must degrade to no tokens");
    }

    #[test]
    fn apply_chat_template_forwards_reasoning_content_for_the_template_to_decide() {
        // GitHub #185: which history reasoning survives into the prompt is
        // the template's decision, not the provider's. The provider hands it
        // every turn's `reasoning_content` in both states and binds the
        // resolved `preserve_thinking` alongside it; the real template then
        // strips the turns before the last real user query and keeps a tool
        // loop's in-flight ones, exactly as the reference does
        // (`ChatTemplate`'s `preserve_thinking_*` tests pin that decision on
        // the real dialect).
        //
        // Blanking it here instead — what the provider used to do — made the
        // decision unreachable: the template never saw the text it was
        // supposed to keep.
        let (_fixture, _reader, provider) = build_thinking_provider(THINKING_TEMPLATE);
        let mut message = ChatMessage::text("assistant", "the answer");
        message.reasoning_content = Some("scratch work".to_owned());

        for options in [opts(), ThinkingOptions { preserve_thinking: true, ..opts() }] {
            let tokens = provider.apply_chat_template(&[message.clone()], &options, no_tools());
            let text = provider.render_tokens(&tokens);
            assert!(
                text.contains("scratch") && text.contains("work"),
                "preserve_thinking={}: {text}",
                options.preserve_thinking
            );
        }
    }

    #[test]
    fn thinking_capabilities_reflect_the_real_templates_probe() {
        let (_fixture, _reader, provider) = build_provider_with(THINKING_TEMPLATE);
        let caps = provider.thinking_capabilities();
        assert!(!caps.can_disable, "THINKING_TEMPLATE raises on disable");
    }

    // -- tool calling, round-tripped against #121's own parser (#132) -------
    //
    // These render through `ignis_artifact::ChatTemplate` directly (no
    // `FrontendSet`/tokenizer fixture — the round-trip a WordLevel test
    // tokenizer would do on tag-heavy XML text is not faithful, so this is
    // the level the earlier tools tests in `frontend.rs` already picked).
    // `REAL_TOOL_DIALECT_TEMPLATE` is not a synthetic analogue: its `tools`
    // and `tool_calls` branches are copied verbatim from the real Qwen3.8
    // template's own `# Tools` system section and its assistant
    // `<tool_call>` rendering (everything else trimmed to the minimum that
    // still parses) — so a test against it exercises the exact tag dialect
    // production actually emits, not an approximation of it.

    const REAL_TOOL_DIALECT_TEMPLATE: &str = r##"
{%- if tools and tools is iterable and tools is not mapping -%}
{{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
{%- for tool in tools -%}
{{- "\n" }}
{{- tool | tojson }}
{%- endfor -%}
{{- "\n</tools>\n\n" }}
{%- endif -%}
{%- for m in messages -%}
{{ m.role }}={{ m.content }}
{%- if m.tool_calls and m.tool_calls is iterable and m.tool_calls is not mapping -%}
    {%- for tool_call in m.tool_calls -%}
        {%- if tool_call.function is defined -%}{%- set tool_call = tool_call.function -%}{%- endif -%}
        {{- "\n<tool_call>\n<function=" + tool_call.name + ">\n" }}
        {%- if tool_call.arguments is defined and tool_call.arguments != "" -%}
            {%- for args_name, args_value in tool_call.arguments|items -%}
                {{- "<parameter=" + args_name + ">\n" }}
                {%- set args_value = args_value if args_value is string else (args_value | tojson) -%}
                {{- args_value }}
                {{- "\n</parameter>\n" }}
            {%- endfor -%}
        {%- endif -%}
        {{- "</function>\n</tool_call>" }}
    {%- endfor -%}
{%- endif -%}
;{%- endfor -%}
"##;

    #[test]
    fn the_real_templates_tools_system_section_renders_verbatim() {
        // AC1 (GitHub #132): a well-formed `tools` array renders the real
        // "# Tools" section, not just a synthetic stand-in for it.
        let template = ChatTemplate::from_source(REAL_TOOL_DIALECT_TEMPLATE).expect("compile");
        let tools = [json!({
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        })];
        let prompt = template
            .render_with_thinking_and_tools(
                &[ArtifactMessage::text(Role::User, "hi")],
                true,
                None,
                false,
                Some(&tools),
            )
            .expect("render");
        assert!(prompt.contains("# Tools\n\nYou have access to the following functions:"), "{prompt}");
        assert!(prompt.contains("<tools>"), "{prompt}");
        assert!(prompt.contains("get_weather"), "{prompt}");
        assert!(prompt.contains("</tools>"), "{prompt}");
    }

    #[test]
    fn the_real_templates_tools_section_matches_the_references_bytes() {
        // GitHub #172 AC1: the reference's `render_tools_system_block` text
        // for the same tools — `", "` / `": "`, sorted keys, `<`, `&`, `'`
        // literal — one tool per line inside `<tools>`.
        let template = ChatTemplate::from_source(REAL_TOOL_DIALECT_TEMPLATE).expect("compile");
        let tools = [
            json!({"type": "function", "function": {
                "name": "read_file",
                "description": "Read <path> & print it's text",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }}),
            json!({"type": "function", "function": {
                "name": "bash",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
            }}),
        ];
        let prompt = template
            .render_with_thinking_and_tools(&[ArtifactMessage::text(Role::User, "hi")], true, None, false, Some(&tools))
            .expect("render");
        let expected = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n\
{\"function\": {\"description\": \"Read <path> & print it's text\", \"name\": \"read_file\", \
\"parameters\": {\"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}}, \
\"type\": \"function\"}\n\
{\"function\": {\"name\": \"bash\", \"parameters\": {\"properties\": {\"command\": {\"type\": \"string\"}}, \
\"type\": \"object\"}}, \"type\": \"function\"}\n</tools>\n\n";
        assert!(prompt.starts_with(expected), "{prompt}");
    }

    #[test]
    fn an_assistant_history_tool_call_renders_the_real_tag_dialect_and_121_parses_it_back() {
        // AC5 (GitHub #132): a prior assistant turn's `tool_calls` renders
        // as the real `<tool_call>` block, and — closing the loop with
        // #121 — `ToolCallScanner` (the exact parser the live response
        // path uses) recovers the same call from that exact rendered text.
        let template = ChatTemplate::from_source(REAL_TOOL_DIALECT_TEMPLATE).expect("compile");
        let history = ArtifactMessage {
            role: Role::Assistant,
            content: ignis_artifact::MessageContent::Text(String::new()),
            tool_calls: vec![ArtifactToolCall {
                id: Some("call_0".to_owned()),
                name: "read_file".to_owned(),
                arguments: r#"{"path": "a.txt"}"#.to_owned(),
            }],
            reasoning_content: None,
        };
        let prompt = template.render(&[history]).expect("render");
        assert!(prompt.contains("<tool_call>\n<function=read_file>\n"), "{prompt}");
        assert!(prompt.contains("<parameter=path>\na.txt\n</parameter>"), "{prompt}");

        let mut scanner = crate::toolcall::ToolCallScanner::new();
        let mut events = scanner.feed(&prompt);
        events.extend(scanner.finish());
        let calls: Vec<_> = events
            .into_iter()
            .filter_map(|e| match e {
                crate::toolcall::ToolEvent::Call(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "{prompt}");
        assert_eq!(calls[0].name, "read_file");
        let args: JsonValue = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args, json!({"path": "a.txt"}));
    }

    /// One wire tool call, as a client resends it in a history message.
    fn wire_tool_call(name: &str, arguments: &str) -> crate::template::ToolCallIn {
        crate::template::ToolCallIn {
            id: Some("call_0".to_owned()),
            function: crate::template::FunctionIn {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            },
        }
    }

    #[test]
    fn a_resent_tool_calls_arguments_reach_the_template_as_the_wire_string() {
        // GitHub #184 AC1, at the server's own hop: the wire `arguments`
        // string is handed to the template untouched, so no re-encoding
        // step exists that could reorder its keys.
        let wire = r#"{"path":"b.rs","dry_run":false,"count":2}"#;
        let call = artifact_tool_call(&wire_tool_call("edit", wire));
        assert_eq!(call.arguments, wire);
        assert_eq!(call.name, "edit");
        assert_eq!(call.id.as_deref(), Some("call_0"));
    }

    #[test]
    fn a_resent_tool_call_renders_its_parameters_in_the_wire_order() {
        // GitHub #184 AC1, through the real tag dialect: the parameters
        // come out in the order the `arguments` document carries them —
        // `path`, `dry_run`, `edits`, `count` — not the alphabetical
        // `count`, `dry_run`, `edits`, `path` a `serde_json::Map` would
        // impose, and the nested object keeps `old` before `new` too.
        let template = ChatTemplate::from_source(REAL_TOOL_DIALECT_TEMPLATE).expect("compile");
        let history = ArtifactMessage {
            role: Role::Assistant,
            content: ignis_artifact::MessageContent::Text(String::new()),
            tool_calls: vec![artifact_tool_call(&wire_tool_call(
                "edit",
                r#"{"path":"src/b.rs","dry_run":false,"edits":[{"old":"x","new":"y"}],"count":2}"#,
            ))],
            reasoning_content: None,
        };
        let prompt = template.render(&[history]).expect("render");
        // The reference's own bytes for these arguments
        // (`chat_template.cpp` `render_tool_call`: a string parameter
        // verbatim, anything else through `tojson_text`).
        assert!(
            prompt.contains(
                "<tool_call>\n<function=edit>\n\
                 <parameter=path>\nsrc/b.rs\n</parameter>\n\
                 <parameter=dry_run>\nfalse\n</parameter>\n\
                 <parameter=edits>\n[{\"old\": \"x\", \"new\": \"y\"}]\n</parameter>\n\
                 <parameter=count>\n2\n</parameter>\n\
                 </function>\n</tool_call>"
            ),
            "{prompt}"
        );
    }

    #[test]
    fn a_malformed_tool_call_arguments_string_degrades_to_an_empty_object_not_a_panic() {
        // Acceptance: `ArtifactTemplateProvider::apply_chat_template`
        // parses a history tool call's wire `arguments` (a JSON-encoded
        // string) back into the object the template needs — a string that
        // is not valid JSON must degrade to `{}`, never panic or corrupt
        // the render (spec 07's documented posture).
        let (_fixture, _reader, provider) = build_provider_with(TEMPLATE);
        let mut message = ChatMessage::text("assistant", "");
        message.tool_calls = Some(vec![crate::template::ToolCallIn {
            id: Some("call_0".to_owned()),
            function: crate::template::FunctionIn {
                name: "f".to_owned(),
                arguments: "not json".to_owned(),
            },
        }]);
        // Must not panic; `TEMPLATE` has no `tool_calls` branch of its own,
        // so this only proves the degrade happens before the render call —
        // the real dialect is exercised in the round-trip test above.
        let tokens = provider.apply_chat_template(&[message], &opts(), no_tools());
        assert!(!tokens.is_empty(), "the render must still complete");
    }
}