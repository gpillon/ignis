//! The chat-template / tokenizer seam (server-01).
//!
//! The OpenAI surface talks in *text*, the engine talks in *tokens*. This
//! module is the only place where the two worlds meet: the server applies
//! the chat template to the request's messages (producing the templated
//! prompt tokens the scheduler consumes) and renders the generated tokens
//! back into the `content` / `text` fields of the responses.
//!
//! v1 ships a **minimal built-in provider** ([`SimpleTemplateProvider`]):
//! a deterministic, tokenizer-free stand-in so the endpoints stay fully
//! functional and testable without the artifact's frontend object set.
//! The artifact's real tokenizer + chat template (artifact-02, GitHub
//! #7) plug into the same seam through the [`crate::artifact_template`]
//! module (`Server::with_artifact_template`). When no artifact is
//! available the built-in placeholder is used instead, and its rendered
//! text is the token id-space (a decimal id per token), not human
//! language; clients of a dev build should not treat `content` as
//! natural text.

use ignis_core::TokenId;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::decoder::TokenDecoder;
use crate::thinking::{ThinkingCapabilities, ThinkingOptions};

/// One conversation message in OpenAI wire shape (`role` + `content`).
///
/// `content` is the plain-string form (v1: the structured content-parts
/// form is rejected at the API boundary with a 400).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The message role (`system`, `user`, `assistant`, `tool`, …).
    pub role: String,
    /// The message text.
    pub content: String,
    /// A prior assistant turn's thinking trace (GitHub #68). Dropped before
    /// rendering unless the request sets `preserve_thinking: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// A prior assistant turn's tool calls (GitHub #132) — the OpenAI wire
    /// shape, `function.arguments` a JSON-encoded string, matching what
    /// this server's own response side emits (#121). Absent on every role
    /// but a prior assistant turn that called a tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallIn>>,
    /// A `role: "tool"` message's correlation id (OpenAI wire field).
    /// Accepted and otherwise unused: the loaded chat template correlates
    /// a tool result to its call sequentially, not by id (verified against
    /// `.qwen/tmp/chat_template.jinja`'s `role == "tool"` branch) — kept
    /// only so a real client that always sends it is not rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// A message with plain content and no prior reasoning or tool calls.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// One tool call an assistant history message carries (OpenAI wire shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallIn {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionIn,
}

/// A tool call's function half: name + arguments (a JSON-encoded string on
/// the wire, matching #121's own response shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionIn {
    pub name: String,
    pub arguments: String,
}

/// The template / tokenizer seam (artifact-02 plugs the real implementation
/// in here): apply the chat template to a conversation to get the prompt
/// tokens the scheduler submits, and render generated tokens back to the
/// response text.
///
/// Implementations must be deterministic (a fixed conversation maps to the
/// same tokens on every call) and `Send + Sync` (the router shares one
/// instance across request handlers).
pub trait TemplateProvider: Send + Sync {
    /// Apply the chat template: the templated prompt tokens for
    /// `messages` (the scheduler prompt — role markers, delimiters, etc.),
    /// with `options` (GitHub #68) and `tools` (GitHub #132) the only other
    /// things that cross this seam. `tools` is the client's own OpenAI
    /// `tools` array, opaque JSON — empty means the request never
    /// mentioned tools at all.
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Vec<TokenId>;

    /// Render generated tokens to the response text (`content` / `text`) —
    /// a whole-list decode, unaware of the reasoning/content split.
    fn render_tokens(&self, tokens: &[TokenId]) -> String;

    /// The thinking controls this provider's template supports (GitHub
    /// #68). The built-in placeholder has no jinja template to probe, so it
    /// imposes no capability limits of its own.
    fn thinking_capabilities(&self) -> ThinkingCapabilities;

    /// A fresh incremental token→text decoder for one request (GitHub #68):
    /// the byte-completeness half of the streaming split — see
    /// [`crate::decoder::OutputDecoder`] for the marker-splitting half built
    /// on top of it.
    fn token_decoder(&self) -> Box<dyn TokenDecoder>;

    /// Whether [`crate::decoder::OutputDecoder`] should start a request in
    /// the reasoning channel (GitHub #68).
    ///
    /// A real thinking-aware template opens thinking-enabled generation
    /// directly in reasoning text — the prompt already primed the opening
    /// `<think>` tag, so there is no marker to wait for before the model's
    /// first generated token is genuinely reasoning. The default answers
    /// `thinking.enable_thinking`, which is correct for such a provider.
    ///
    /// A provider whose output can never actually separate into two
    /// channels (the built-in placeholder: no jinja template, no
    /// `</think>` it could ever emit) must override this to always return
    /// `false` — otherwise a thinking-enabled request would misclassify
    /// its entire output as reasoning purely because no marker can ever
    /// arrive to prove otherwise.
    fn decoder_starts_in_reasoning(&self, thinking: &ThinkingOptions) -> bool {
        thinking.enable_thinking
    }
}

/// The minimal built-in provider (v1 placeholder, replaced by artifact-02):
///
/// - **apply** — one token per whitespace-separated word of each message;
///   the token id is the FNV-1a 32-bit hash of `"{role}:{word}"` —
///   deterministic across runs (no RNG, no clocks), so tests can pin the
///   exact prompt-token stream.
/// - **render** — the token id in decimal (the tokenizer that maps ids
///   back to text lands with artifact-02).
#[derive(Debug, Default)]
pub struct SimpleTemplateProvider;

impl TemplateProvider for SimpleTemplateProvider {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        _options: &ThinkingOptions,
        _tools: &[JsonValue],
    ) -> Vec<TokenId> {
        // The placeholder has no jinja template to bind thinking variables
        // or tools into — it ignores both (a test-only `TemplateProvider`
        // that wants to observe them records them itself; see
        // `openai_http_thinking.rs`'s / `openai_http_toolcalls.rs`'s
        // recording doubles).
        messages
            .iter()
            .flat_map(|m| {
                m.content.split_whitespace().map(move |word| {
                    fnv1a32(format!("{}:{}", m.role, word).as_bytes())
                })
            })
            .collect()
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(SimpleTokenDecoder { first: true })
    }

    fn decoder_starts_in_reasoning(&self, _thinking: &ThinkingOptions) -> bool {
        // The placeholder has no jinja template and can never emit a
        // `</think>` marker — it never produces a reasoning span, so its
        // output is always content regardless of the resolved thinking
        // options (see the trait doc for why this must not default to
        // `thinking.enable_thinking` here).
        false
    }
}

/// The placeholder's incremental decoder: the same decimal-id, space-joined
/// rendering as [`SimpleTemplateProvider::render_tokens`], but one token at
/// a time. Every token is plain ASCII, so there is no multi-byte concern to
/// hold back.
struct SimpleTokenDecoder {
    first: bool,
}

impl TokenDecoder for SimpleTokenDecoder {
    fn push(&mut self, token: TokenId) -> String {
        if std::mem::replace(&mut self.first, false) {
            token.to_string()
        } else {
            format!(" {token}")
        }
    }

    fn finish(&mut self) -> String {
        String::new()
    }
}

/// FNV-1a 32-bit — a stable, dependency-free content hash (deterministic
/// token ids without a real tokenizer / RNG).
fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in bytes {
        hash ^= u32::from(*b);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::text(role, content)
    }

    fn opts() -> ThinkingOptions {
        ThinkingOptions::default()
    }

    fn no_tools() -> &'static [JsonValue] {
        &[]
    }

    #[test]
    fn template_is_deterministic() {
        let p = SimpleTemplateProvider;
        let messages = [msg("user", "hello world"), msg("assistant", "hi")];
        let a = p.apply_chat_template(&messages, &opts(), no_tools());
        let b = p.apply_chat_template(&messages, &opts(), no_tools());
        assert_eq!(a, b, "the same conversation must template identically");
    }

    #[test]
    fn one_token_per_word_and_role_scoped() {
        let p = SimpleTemplateProvider;
        let tokens = p.apply_chat_template(&[msg("user", "a b c")], &opts(), no_tools());
        assert_eq!(tokens.len(), 3, "one token per whitespace word");
        // The same word under a different role is a different token (the
        // role is part of the hashed key).
        let other = p.apply_chat_template(&[msg("assistant", "a")], &opts(), no_tools());
        let user_a = p.apply_chat_template(&[msg("user", "a")], &opts(), no_tools());
        assert_ne!(other, user_a);
    }

    #[test]
    fn empty_conversation_has_no_tokens() {
        let p = SimpleTemplateProvider;
        assert!(p.apply_chat_template(&[], &opts(), no_tools()).is_empty());
        assert!(p
            .apply_chat_template(&[msg("user", "   ")], &opts(), no_tools())
            .is_empty());
    }

    #[test]
    fn render_is_the_decimal_id_stream() {
        let p = SimpleTemplateProvider;
        assert_eq!(p.render_tokens(&[7, 42, 3]), "7 42 3");
        assert_eq!(p.render_tokens(&[]), "");
    }

    #[test]
    fn the_placeholder_accepts_every_thinking_control() {
        let caps = SimpleTemplateProvider.thinking_capabilities();
        assert!(caps.can_disable);
        assert!(caps.supports(crate::thinking::ReasoningEffort::Xhigh));
    }

    #[test]
    fn the_incremental_decoder_matches_the_whole_list_render() {
        let p = SimpleTemplateProvider;
        let mut decoder = p.token_decoder();
        let mut out = String::new();
        for &t in &[7u32, 42, 3] {
            out.push_str(&decoder.push(t));
        }
        out.push_str(&decoder.finish());
        assert_eq!(out, p.render_tokens(&[7, 42, 3]));
    }
}