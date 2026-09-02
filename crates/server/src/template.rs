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

/// One conversation message in OpenAI wire shape (`role` + `content`).
///
/// `content` is the plain-string form (v1: the structured content-parts
/// form is rejected at the API boundary with a 400).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The message role (`system`, `user`, `assistant`, …).
    pub role: String,
    /// The message text.
    pub content: String,
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
    /// `messages` (the scheduler prompt — role markers, delimiters, etc.).
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Vec<TokenId>;

    /// Render generated tokens to the response text (`content` / `text`).
    fn render_tokens(&self, tokens: &[TokenId]) -> String;
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
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Vec<TokenId> {
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
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn template_is_deterministic() {
        let p = SimpleTemplateProvider;
        let messages = [msg("user", "hello world"), msg("assistant", "hi")];
        let a = p.apply_chat_template(&messages);
        let b = p.apply_chat_template(&messages);
        assert_eq!(a, b, "the same conversation must template identically");
    }

    #[test]
    fn one_token_per_word_and_role_scoped() {
        let p = SimpleTemplateProvider;
        let tokens = p.apply_chat_template(&[msg("user", "a b c")]);
        assert_eq!(tokens.len(), 3, "one token per whitespace word");
        // The same word under a different role is a different token (the
        // role is part of the hashed key).
        let other = p.apply_chat_template(&[msg("assistant", "a")]);
        let user_a = p.apply_chat_template(&[msg("user", "a")]);
        assert_ne!(other, user_a);
    }

    #[test]
    fn empty_conversation_has_no_tokens() {
        let p = SimpleTemplateProvider;
        assert!(p.apply_chat_template(&[]).is_empty());
        assert!(p
            .apply_chat_template(&[msg("user", "   ")])
            .is_empty());
    }

    #[test]
    fn render_is_the_decimal_id_stream() {
        let p = SimpleTemplateProvider;
        assert_eq!(p.render_tokens(&[7, 42, 3]), "7 42 3");
        assert_eq!(p.render_tokens(&[]), "");
    }
}