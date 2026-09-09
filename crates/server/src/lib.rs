//! ignis-server: the OpenAI-compatible HTTP surface (localhost, no auth).
//!
//! v1 endpoints (server-01, `docs/design/ignis-v1.md` §2):
//! - `GET /v1/models` — the loaded model.
//! - `POST /v1/chat/completions` — chat completions, streaming (SSE) and
//!   non-streaming; requests route into the core scheduler and tokens
//!   stream back as they are generated.
//! - `POST /v1/responses` — the OpenAI responses API (non-streaming in v1).
//!
//! Architecture: the server owns the core [`Scheduler`] behind an
//! [`Engine`] — a dedicated model thread owning the scheduler exclusively,
//! with the async/HTTP side talking to it only through a command channel
//! (GitHub #69, `engine.rs`); the text⇄token boundary is the
//! [`TemplateProvider`] seam (`template.rs`) — v1 ships a minimal built-in
//! provider, artifact-02 (the artifact's frontend object set, GitHub #7)
//! replaces it through the same constructor-injection seam.
//!
//! The [`Compute`] backend is injected through the scheduler constructor:
//! tests use [`ignis_core::MockCompute`] (ADR 0006, CPU-only), production
//! wires the kernel-leaf adapter when it lands.

pub mod api;
pub mod artifact_template;
pub mod config;
pub mod decoder;
pub mod engine;
pub mod loader;
pub mod runtime;
pub mod telemetry;
pub mod template;
pub mod thinking;

use std::time::Duration;

use axum::Router;

use crate::engine::Engine;
use crate::template::TemplateProvider;
use crate::thinking::ReasoningEffort;

/// The server's knobs (constructor injection — the template seam is
/// pluggable here: artifact-02 swaps in the artifact-backed provider).
#[derive(Clone)]
pub struct Server {
    /// The engine: the core scheduler + per-request event routing
    /// (submit / drive / route — `engine.rs`).
    pub engine: Engine,
    /// The chat-template / tokenizer seam (artifact-02 plugs the real
    /// frontend object set in here).
    pub template: std::sync::Arc<dyn TemplateProvider>,
    /// How long a non-streaming request waits for its completion before the
    /// handler gives up with a 504 (guards a wedged engine from hanging
    /// the client forever).
    pub request_timeout: Duration,
    /// The server-wide `enable_thinking` default (`IGNIS_ENABLE_THINKING`,
    /// GitHub #68) a request's unset field falls back to.
    pub default_enable_thinking: bool,
    /// The server-wide `reasoning_effort` default (`IGNIS_REASONING_EFFORT`).
    pub default_reasoning_effort: Option<ReasoningEffort>,
}

impl Server {
    /// A server over `engine`'s scheduler with the given template provider.
    pub fn new(engine: Engine, template: Box<dyn TemplateProvider>) -> Self {
        Self {
            engine,
            template: std::sync::Arc::from(template),
            request_timeout: Duration::from_secs(30),
            default_enable_thinking: true,
            default_reasoning_effort: None,
        }
    }

    /// A server over `engine`'s scheduler with the artifact's real
    /// tokenizer + chat template (the [`FrontendSet`] extracted by
    /// artifact-02, GitHub #7) in place of the built-in placeholder: the
    /// conversation is templated by the container's chat template and
    /// tokenized by the container's HuggingFace tokenizer
    /// (`artifact_template.rs`).
    pub fn with_artifact_template(engine: Engine, frontend: ignis_artifact::FrontendSet) -> Self {
        Self::new(
            engine,
            Box::new(artifact_template::ArtifactTemplateProvider::new(frontend)),
        )
    }

    /// Set the non-streaming completion timeout (test knob; the default is
    /// 30 s).
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set the server-wide thinking defaults (`IGNIS_ENABLE_THINKING` /
    /// `IGNIS_REASONING_EFFORT`, GitHub #68). Callers that set a non-trivial
    /// default should validate it against `template.thinking_capabilities()`
    /// first (`thinking::validate_defaults`) — this setter does not, so
    /// tests can construct an out-of-band `Server` without a template to
    /// probe.
    pub fn with_thinking_defaults(
        mut self,
        enable_thinking: bool,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
        self.default_enable_thinking = enable_thinking;
        self.default_reasoning_effort = reasoning_effort;
        self
    }

    /// Route the server's telemetry through `sink` (the §5 JSONL sink — a
    /// file or stdout in production, an in-memory sink in tests).
    pub fn with_telemetry(mut self, sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>) -> Self {
        self.engine = self.engine.with_telemetry(sink);
        self
    }

    /// The axum app (build once, share across a listener; the state the
    /// router serves is an `Arc` of this server).
    pub fn app(&self) -> Router {
        let state: std::sync::Arc<Server> = std::sync::Arc::new(self.clone());
        api::router(state)
    }

    /// Bind `addr` and serve. The engine's model thread (GitHub #69) was
    /// already spawned when it was constructed — nothing to start here.
    /// Runs until [`shutdown_signal`] resolves (Ctrl+C, or SIGTERM on
    /// unix), then drains in-flight requests before returning (GitHub #80,
    /// spec §28: graceful shutdown, not an abrupt kill) — `main` is
    /// responsible for emitting `ignis.process.stopped` and flushing the
    /// logging queue after this returns, since that is the true last event.
    pub async fn serve(self, addr: String) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let app = self.app();
        axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await
    }
}

/// Emits `ignis.process.stopping` (spec §28) — split out from
/// [`shutdown_signal`] so the event's own shape is unit-testable without
/// having to actually deliver Ctrl+C/SIGTERM to a live process (not
/// portably doable from a Rust test, and Windows has no SIGTERM at all —
/// see `shutdown_tests` below).
fn emit_stopping() {
    tracing::info!(name: "ignis.process.stopping", "graceful shutdown signal received");
}

/// Waits for a graceful-shutdown signal (Ctrl+C, or SIGTERM on unix) and
/// emits `ignis.process.stopping` (spec §28) right as it resolves — passed
/// to `axum::serve(..).with_graceful_shutdown(..)` so in-flight requests get
/// a chance to finish before the listener actually stops. This is the one
/// INFO event on this path; it fires at most once per process lifetime, not
/// a hot-path concern.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("installing the Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("installing the SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    emit_stopping();
}

#[cfg(test)]
mod shutdown_tests {
    use std::sync::Arc;

    use ignis_logging::{JsonLayer, MemorySink};
    use tracing_subscriber::layer::SubscriberExt;

    use super::emit_stopping;

    /// The integration-level path (`shutdown_signal` actually waiting on a
    /// real Ctrl+C/SIGTERM) is exercised by hand, not by this suite — see
    /// `emit_stopping`'s doc comment. This covers the part that is
    /// meaningfully unit-testable: the event `shutdown_signal` emits once a
    /// signal resolves has the exact name/severity spec §28 asks for.
    #[test]
    fn emit_stopping_produces_the_spec_shaped_event() {
        let sink = Arc::new(MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, emit_stopping);

        let lines = sink.lines();
        assert_eq!(lines.len(), 1);
        let record: serde_json::Value = serde_json::from_str(&lines[0]).expect("valid json");
        assert_eq!(record["event_name"], "ignis.process.stopping");
        assert_eq!(record["severity_text"], "INFO");
    }
}
