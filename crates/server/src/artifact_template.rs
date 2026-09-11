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

use ignis_artifact::{DecodeStreamState, FrontendSet, Role};
use ignis_core::TokenId;
use serde_json::Value as JsonValue;

use crate::decoder::TokenDecoder;
use crate::template::{ChatMessage, TemplateProvider};
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
}

impl ArtifactTemplateProvider {
    /// A provider over the artifact's frontend object set. The set is
    /// built once at startup (`FrontendSet::from_reader`) and shared
    /// across handlers through the `Arc` (the router shares one
    /// `Arc<dyn TemplateProvider>`).
    pub fn new(frontend: FrontendSet) -> Self {
        Self {
            set: Arc::new(frontend),
        }
    }
}

impl TemplateProvider for ArtifactTemplateProvider {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Vec<TokenId> {
        let templated: Vec<ignis_artifact::ChatMessage> = messages
            .iter()
            .map(|message| {
                let role = Role::parse(&message.role).unwrap_or(Role::User);
                let mut templated =
                    ignis_artifact::ChatMessage::text(role, message.content.clone());
                // Prior assistant reasoning rides into the prompt only when
                // the request opted in (GitHub #68) — dropped by default so
                // a long conversation does not silently accumulate traces.
                if options.preserve_thinking {
                    templated.reasoning_content = message.reasoning_content.clone();
                }
                // A prior assistant turn's tool calls (GitHub #132): each
                // call's `arguments` is a JSON-encoded string on the wire
                // (matching #121's own response shape) and the template
                // needs the parsed object (`arguments|items`) — a call
                // whose string does not parse as a JSON object degrades to
                // an empty one rather than failing the whole render (the
                // same "never panic, degrade" posture as the render/encode
                // failures below).
                if let Some(calls) = &message.tool_calls {
                    templated.tool_calls = calls
                        .iter()
                        .map(|call| ignis_artifact::ToolCall {
                            id: call.id.clone(),
                            name: call.function.name.clone(),
                            arguments: serde_json::from_str(&call.function.arguments)
                                .unwrap_or_else(|err| {
                                    eprintln!(
                                        "ignis-server: history tool_calls[].function.arguments is not valid JSON, degrading to {{}}: {err}"
                                    );
                                    serde_json::json!({})
                                }),
                        })
                        .collect();
                }
                templated
            })
            .collect();
        let prompt = match self.set.chat_template().render_with_thinking_and_tools(
            &templated,
            options.enable_thinking,
            options.reasoning_effort,
            Some(tools),
        ) {
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn apply_chat_template_drops_reasoning_content_unless_preserved() {
        let (_fixture, _reader, provider) = build_thinking_provider(THINKING_TEMPLATE);
        let mut message = ChatMessage::text("assistant", "the answer");
        message.reasoning_content = Some("scratch work".to_owned());

        let dropped = provider.apply_chat_template(&[message.clone()], &opts(), no_tools());
        let dropped_text = provider.render_tokens(&dropped);
        assert!(!dropped_text.contains("scratch"), "{dropped_text}");
        assert!(!dropped_text.contains("work"), "{dropped_text}");

        let preserved_options = ThinkingOptions {
            preserve_thinking: true,
            ..opts()
        };
        let preserved = provider.apply_chat_template(&[message], &preserved_options, no_tools());
        let preserved_text = provider.render_tokens(&preserved);
        assert!(preserved_text.contains("scratch"), "{preserved_text}");
        assert!(preserved_text.contains("work"), "{preserved_text}");
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
                Some(&tools),
            )
            .expect("render");
        assert!(prompt.contains("# Tools\n\nYou have access to the following functions:"), "{prompt}");
        assert!(prompt.contains("<tools>"), "{prompt}");
        assert!(prompt.contains("get_weather"), "{prompt}");
        assert!(prompt.contains("</tools>"), "{prompt}");
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
                arguments: json!({"path": "a.txt"}),
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