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
//! [`Engine`] (submit / per-request event routing / a driver loop that
//! advances the engine, `engine.rs`); the text⇄token boundary is the
//! [`TemplateProvider`] seam (`template.rs`) — v1 ships a minimal built-in
//! provider, artifact-02 (the artifact's frontend object set, GitHub #7)
//! replaces it through the same constructor-injection seam.
//!
//! The [`Compute`] backend is injected through the scheduler constructor:
//! tests use [`ignis_core::MockCompute`] (ADR 0006, CPU-only), production
//! wires the kernel-leaf adapter when it lands.

pub mod api;
pub mod engine;
pub mod template;

use std::time::Duration;

use axum::Router;

use crate::engine::Engine;
use crate::template::TemplateProvider;

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
}

impl Server {
    /// A server over `engine`'s scheduler with the given template provider.
    pub fn new(engine: Engine, template: Box<dyn TemplateProvider>) -> Self {
        Self {
            engine,
            template: std::sync::Arc::from(template),
            request_timeout: Duration::from_secs(30),
        }
    }

    /// Set the non-streaming completion timeout (test knob; the default is
    /// 30 s).
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// The axum app (build once, share across a listener; the state the
    /// router serves is an `Arc` of this server).
    pub fn app(&self) -> Router {
        let state: std::sync::Arc<Server> = std::sync::Arc::new(self.clone());
        api::router(state)
    }

    /// Bind `addr` and serve (spawns the engine's driver loop — the single
    /// task that advances the engine and routes events into the request
    /// streams). Runs until the listener is closed.
    pub async fn serve(self, addr: String) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let app = self.app();
        // The driver loop: one task for the server's life (the single
        // engine advancer; see `Engine::run`).
        let driver = self.engine.clone();
        tokio::spawn(async move { driver.run().await });
        axum::serve(listener, app).await
    }
}