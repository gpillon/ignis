//! Shared test-only helpers (GitHub #69), included via `#[path]` by
//! individual integration test binaries (`tests/*.rs`) — this file is not
//! itself a test binary (it lives under a subdirectory of `tests/`, so
//! cargo does not compile it as one).

use std::sync::Arc;

use ignis_core::TokenId;
use ignis_server::decoder::TokenDecoder;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};

/// A few deterministic scheduling turns (never a sleep — ADR 0006) to let a
/// spawned task run up to its next await point — e.g. far enough into
/// `Engine::submit` to have enqueued its command, before the test proceeds
/// to release a gate the model thread is held on.
pub async fn nudge() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}

/// A pure-delegate `TemplateProvider` over a shared `Arc<T>` (GitHub #132):
/// `Server::new` takes ownership of a boxed provider, but a test's
/// recording double (e.g. one that captures `ThinkingOptions` or `tools`
/// for later inspection) needs its own `Arc` clone to read back from after
/// the request completes. Every `TemplateProvider` method forwards to `T`
/// unchanged — this exists purely so the harness can hand the router an
/// owned `Box<dyn TemplateProvider>` while keeping a second handle to the
/// same instance.
pub struct SharedTemplate<T>(pub Arc<T>);

impl<T: TemplateProvider> TemplateProvider for SharedTemplate<T> {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[serde_json::Value],
    ) -> Vec<TokenId> {
        self.0.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        self.0.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        self.0.thinking_capabilities()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        self.0.token_decoder()
    }

    fn decoder_starts_in_reasoning(&self, options: &ThinkingOptions) -> bool {
        self.0.decoder_starts_in_reasoning(options)
    }
}
