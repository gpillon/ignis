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

use ignis_artifact::vision::{layout, MediaItem, PreparedMedia, ProcessorError, IMAGE_PAD_ID};
use ignis_core::vision::Multimodal;
use ignis_core::TokenId;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::decoder::TokenDecoder;
use crate::thinking::{ThinkingCapabilities, ThinkingOptions};

/// One conversation message in OpenAI wire shape (`role` + `content`).
///
/// `content` is a plain string or an array of OpenAI content parts
/// (GitHub #175). The parts are parsed permissively so a malformed or
/// unknown part reaches [`check_content_parts`] and is refused with a 400
/// naming it, rather than failing deserialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The message role (`system`, `user`, `assistant`, `tool`, …).
    pub role: String,
    /// The message content.
    pub content: MessageContent,
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
            content: MessageContent::Text(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// A message's `content`: the plain string, or OpenAI content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// The plain-string form.
    Text(String),
    /// The content-parts form, in wire order.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// The text the chat template renders for this content once
    /// [`check_content_parts`] has passed it: the string itself, or the
    /// parts' [`template_text_parts`] joined.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => template_text_parts(parts).concat(),
        }
    }
}

/// The text runs a parts array hands the chat template, the reference's
/// way (ninfer `serve/translate.cpp` `to_prompt_input`): each text part in
/// order, with a `"\n"` inserted before a non-empty text part that directly
/// follows another text part. So `[{"text":"a"},{"text":"b"}]` renders as
/// the string `"a\nb"`, not `"ab"`. A non-text part breaks the run (no
/// newline across it) and contributes nothing here: only text parts
/// survive [`check_content_parts`] until media is served.
pub fn template_text_parts(parts: &[ContentPart]) -> Vec<&str> {
    let mut out = Vec::with_capacity(parts.len());
    let mut after_text = false;
    for part in parts {
        match (part.kind.as_deref(), part.text.as_deref()) {
            (Some("text"), Some(text)) => {
                if after_text && !text.is_empty() {
                    out.push("\n");
                }
                out.push(text);
                after_text = true;
            }
            _ => after_text = false,
        }
    }
    out
}

/// One OpenAI content part, parsed permissively: every field is optional
/// and unknown fields are ignored (the `image_url` object's `detail` among
/// them), so what a part *is* gets decided by [`check_content_parts`] with
/// an error that names it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "JsonValue", into = "JsonValue")]
pub struct ContentPart {
    /// The part's `type`, when it is a string.
    pub kind: Option<String>,
    /// A `text` part's text, when it is a string.
    pub text: Option<String>,
    /// An `image_url` / `video_url` part's `url`, when it is a string
    /// inside the object OpenAI's wire shape nests it in.
    pub url: Option<String>,
}

impl From<ContentPart> for JsonValue {
    fn from(part: ContentPart) -> Self {
        let mut object = serde_json::Map::new();
        if let Some(text) = part.text {
            object.insert("text".to_owned(), JsonValue::String(text));
        }
        if let Some(kind) = part.kind {
            if let Some(url) = part.url {
                object.insert(kind.clone(), serde_json::json!({ "url": url }));
            }
            object.insert("type".to_owned(), JsonValue::String(kind));
        }
        JsonValue::Object(object)
    }
}

impl From<JsonValue> for ContentPart {
    fn from(value: JsonValue) -> Self {
        let string = |v: Option<&JsonValue>| v.and_then(JsonValue::as_str).map(str::to_owned);
        let kind = string(value.get("type"));
        let url = kind
            .as_deref()
            .and_then(|kind| value.get(kind))
            .and_then(|media| string(media.get("url")));
        Self {
            text: string(value.get("text")),
            kind,
            url,
        }
    }
}

/// Why a request's content parts were refused (GitHub #175): the OpenAI
/// error `code` — the reference's name — and a message naming the offending
/// message and part index. Always an HTTP 400, raised before admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentRejection {
    /// The error `code` (`vision_disabled`, `video_unsupported`,
    /// `invalid_media`, `modality_not_supported` or `invalid_request_error`).
    pub code: &'static str,
    /// The human-readable message.
    pub message: String,
}

/// Validate every message's content parts. Text parts pass; media parts are
/// recognised but not served yet, so each is refused with the reference's
/// error code:
///
/// - media in a `system` message → `invalid_media` (the reference's
///   processor refuses it as invalid input; the template cannot render
///   media there). Checked first: it is wrong on any server, with or
///   without vision.
/// - `video_url` → `video_unsupported` (images only, spec §Scope).
/// - `image_url` → `vision_disabled`, unless `vision` (a `--vision` load,
///   GitHub #179), where it passes to media acquisition.
/// - an unknown part type → `modality_not_supported` (the reference's code
///   and wording).
/// - an empty array, a part without a string `type`, a `text` part without
///   string text, or a media part without a string `url` →
///   `invalid_request_error` (the reference's schema checks; its body leaves
///   `code` empty, this server's error body always carries one).
///
/// Media parts are recognised on `user`, `assistant` and `tool` messages
/// alike — unlike the reference's OpenAI schema, which refuses non-text
/// parts on a tool message: a browser tool's screenshot result must reach
/// the model the way its text does (spec user story 6).
///
/// Checked in the reference's order, each pass over the whole conversation:
/// shape errors (its schema), then media (its vision check), then unknown
/// types (its prompt translation) — so a malformed request is told it is
/// malformed rather than that vision is off.
pub fn check_content_parts(messages: &[ChatMessage], vision: bool) -> Result<(), ContentRejection> {
    let refuse = |code, message: String| Err(ContentRejection { code, message });
    let parts = messages.iter().enumerate().filter_map(|(i, message)| match &message.content {
        MessageContent::Parts(parts) => Some((i, message, parts)),
        MessageContent::Text(_) => None,
    });
    for (i, _, parts) in parts.clone() {
        if parts.is_empty() {
            return refuse("invalid_request_error", format!("message {i} content must not be empty"));
        }
        for (j, part) in parts.iter().enumerate() {
            let at = format!("message {i} content part {j}");
            match part.kind.as_deref() {
                None => {
                    return refuse("invalid_request_error", format!("{at} must have a string 'type'"))
                }
                Some("text") if part.text.is_none() => {
                    return refuse(
                        "invalid_request_error",
                        format!("{at}: text content part must contain a string 'text'"),
                    )
                }
                Some("text") => {}
                Some(media @ ("image_url" | "video_url")) if part.url.is_none() => {
                    return refuse(
                        "invalid_request_error",
                        format!("{at}: {media} must be an object containing a string url"),
                    )
                }
                Some(_) => {}
            }
        }
    }
    for (i, message, parts) in parts.clone() {
        for (j, part) in parts.iter().enumerate() {
            let at = format!("message {i} content part {j}");
            let video = match part.kind.as_deref() {
                Some("image_url") => false,
                Some("video_url") => true,
                _ => continue,
            };
            if message.role == "system" {
                return refuse(
                    "invalid_media",
                    format!("{at}: system messages cannot contain images or videos"),
                );
            }
            if video {
                return refuse("video_unsupported", format!("{at}: video input is not supported"));
            }
            if !vision {
                return refuse("vision_disabled", format!("{at}: vision is disabled for this server"));
            }
        }
    }
    for (i, _, parts) in parts {
        for (j, part) in parts.iter().enumerate() {
            let kind = part.kind.as_deref().unwrap_or_default();
            if !matches!(kind, "text" | "image_url" | "video_url") {
                return refuse(
                    "modality_not_supported",
                    format!("message {i} content part {j}: content type '{kind}' is not supported"),
                );
            }
        }
    }
    Ok(())
}

/// One tool call an assistant history message carries (OpenAI wire shape).
/// No `type` field: OpenAI's wire shape always carries `"type": "function"`
/// here (the only tool type this protocol defines), so it is accepted and
/// ignored the same way `tool_call_id` is on a `role: "tool"` message —
/// there is nothing to branch on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallIn {
    /// The call's id (OpenAI `id`), when the client sends one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The function that was called.
    pub function: FunctionIn,
}

/// A tool call's function half: name + arguments (a JSON-encoded string on
/// the wire, matching #121's own response shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionIn {
    /// The function name.
    pub name: String,
    /// The arguments, as the JSON-encoded object string OpenAI's wire
    /// shape carries (parsed back into an object where the template needs
    /// one — see `artifact_template.rs`'s `apply_chat_template`).
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

    /// Apply the chat template to a conversation carrying images (GitHub
    /// #179): `media` are its image parts' prepared payloads in prompt
    /// order. Each image's placeholder expands to its merged-grid run, and
    /// the prompt comes back with its three-axis positions, `rope_delta` and
    /// media items. A provider that cannot render images refuses with
    /// `vision_disabled`.
    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        media: Vec<PreparedMedia>,
    ) -> Result<(Vec<TokenId>, Multimodal), ContentRejection> {
        let _ = (messages, options, tools, media);
        Err(ContentRejection {
            code: "vision_disabled",
            message: "this server's chat template cannot render images".to_owned(),
        })
    }
}

/// A prepared prompt's refusal as a content rejection: the processor's own
/// wire code (`invalid_media` or `media_budget_exceeded`).
pub fn processor_rejection(error: ProcessorError) -> ContentRejection {
    ContentRejection { code: error.code(), message: error.to_string() }
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
                let text = m.content.text();
                text.split_whitespace()
                    .map(|word| fnv1a32(format!("{}:{}", m.role, word).as_bytes()))
                    .collect::<Vec<_>>()
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

    /// The placeholder's words, with each image part replaced by its
    /// merged-grid run of `<|image_pad|>` ids, laid out by the processor's
    /// own position rules.
    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        _options: &ThinkingOptions,
        _tools: &[JsonValue],
        media: Vec<PreparedMedia>,
    ) -> Result<(Vec<TokenId>, Multimodal), ContentRejection> {
        let mut tokens = Vec::new();
        let mut images = media.iter();
        for message in messages {
            let word = |word: &str| fnv1a32(format!("{}:{}", message.role, word).as_bytes());
            match &message.content {
                MessageContent::Text(text) => tokens.extend(text.split_whitespace().map(word)),
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match (part.kind.as_deref(), &part.text) {
                            (Some("image_url"), _) => {
                                let image = images.next().ok_or_else(|| {
                                    processor_rejection(ProcessorError::PlaceholderMismatch(
                                        "more image parts than prepared media",
                                    ))
                                })?;
                                let run = image.grid.vision_tokens() as usize;
                                tokens.extend(std::iter::repeat_n(IMAGE_PAD_ID, run));
                            }
                            (_, Some(text)) => tokens.extend(text.split_whitespace().map(word)),
                            _ => {}
                        }
                    }
                }
            }
        }
        if images.next().is_some() {
            return Err(processor_rejection(ProcessorError::PlaceholderMismatch(
                "fewer image parts than prepared media",
            )));
        }
        let grids: Vec<_> = media.iter().map(|m| m.grid).collect();
        let (positions, spans, rope_delta) =
            layout::assign_positions(&layout::token_types(&tokens), &grids).map_err(processor_rejection)?;
        let media = media
            .into_iter()
            .zip(spans)
            .map(|(m, token_span)| MediaItem {
                grid: m.grid,
                token_span,
                patches: m.patches,
                content_digest: m.content_digest,
            })
            .collect();
        Ok((tokens, Multimodal { positions, rope_delta, media }))
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
    fn content_is_a_string_or_parts_and_serializes_back_to_the_wire_shape() {
        let message: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "a " },
                { "type": "image_url", "image_url": { "url": "https://x/a.png", "detail": "low" } },
                { "type": "text", "text": "b" }
            ]
        }))
        .unwrap();
        assert_eq!(message.content.text(), "a b");
        let MessageContent::Parts(parts) = &message.content else {
            panic!("expected parts: {message:?}");
        };
        assert_eq!(parts[1].kind.as_deref(), Some("image_url"));
        assert_eq!(parts[1].url.as_deref(), Some("https://x/a.png"));
        assert_eq!(
            serde_json::to_value(&parts[1]).unwrap(),
            serde_json::json!({ "type": "image_url", "image_url": { "url": "https://x/a.png" } })
        );
        let plain: ChatMessage =
            serde_json::from_value(serde_json::json!({ "role": "user", "content": "hi" })).unwrap();
        assert_eq!(plain, ChatMessage::text("user", "hi"));
        assert_eq!(serde_json::to_value(&plain).unwrap()["content"], "hi");
    }

    #[test]
    fn text_parts_join_the_references_way() {
        let t = |s: &str| serde_json::json!({ "type": "text", "text": s });
        let text = |parts: Vec<JsonValue>| {
            serde_json::from_value::<MessageContent>(JsonValue::Array(parts)).unwrap().text()
        };
        assert_eq!(text(vec![t("a"), t("b")]), "a\nb");
        assert_eq!(text(vec![t("a"), t(""), t("b")]), "a\nb");
        assert_eq!(text(vec![t(""), t("b")]), "\nb");
        assert_eq!(text(vec![t("a"), t("")]), "a");
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