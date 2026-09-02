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

use ignis_artifact::{FrontendSet, Role};
use ignis_core::TokenId;

use crate::template::{ChatMessage, TemplateProvider};

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
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Vec<TokenId> {
        let templated: Vec<ignis_artifact::ChatMessage> = messages
            .iter()
            .map(|message| {
                let role = Role::parse(&message.role).unwrap_or(Role::User);
                ignis_artifact::ChatMessage::text(role, message.content.clone())
            })
            .collect();
        let prompt = match self.set.chat_template().render(&templated) {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::fixture::{self, FixtureObject};
    use ignis_artifact::Reader;

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
    /// two).
    fn frontend_fixture() -> (Vec<FixtureObject>, Vec<u8>) {
        let resources: [(&'static str, &'static [u8]); 6] = [
            ("frontend/tokenizer.json", TOKENIZER_JSON.as_bytes()),
            ("frontend/tokenizer_config.json", b"{}"),
            ("frontend/chat_template.jinja", TEMPLATE.as_bytes()),
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

    /// Open the fixture artifact, extract its frontend set, and hand the
    /// provider back. The fixture + reader stay alive for the caller's
    /// scope (the Windows direct-I/O handle must be closed before the
    /// fixture removes the file).
    fn build_provider() -> (fixture::TempArtifact, Reader, ArtifactTemplateProvider) {
        let (objects, payload) = frontend_fixture();
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

    #[test]
    fn apply_chat_template_uses_the_real_template_and_tokenizer() {
        let (_fixture, _reader, provider) = build_provider();
        let messages = [
            ChatMessage {
                role: "user".into(),
                content: "hello world".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "hi there".into(),
            },
        ];
        let tokens = provider.apply_chat_template(&messages);
        assert!(!tokens.is_empty(), "the rendered prompt must tokenize");
        // Determinism: the same conversation templates identically (the
        // trait's contract).
        assert_eq!(tokens, provider.apply_chat_template(&messages));
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
        let messages = [ChatMessage {
            role: "bogus".into(),
            content: "hello".into(),
        }];
        // The foreign role must not panic: it templates as `user` (the
        // documented v1 fallback), and the prompt still renders.
        let tokens = provider.apply_chat_template(&messages);
        assert!(!tokens.is_empty());
        let text = provider.render_tokens(&tokens);
        assert!(text.contains("hello"), "{text}");
    }
}