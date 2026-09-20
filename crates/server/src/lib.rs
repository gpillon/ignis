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
pub mod decide;
pub mod decoder;
pub mod download;
pub mod engine;
pub mod expose;
pub mod instruction;
pub mod loader;
pub mod media;
pub mod metrics;
pub mod playground;
pub mod numbers;
pub mod runtime;
pub mod telemetry;
pub mod template;
pub mod thinking;
pub mod toolcall;

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
    /// the client forever). An operator knob (`--request-timeout` /
    /// `IGNIS_REQUEST_TIMEOUT`, GitHub #95) — `main` sets it via
    /// [`Server::with_request_timeout`] after `config::resolve` validates it.
    pub request_timeout: Duration,
    /// The server-wide `enable_thinking` default (`IGNIS_ENABLE_THINKING`,
    /// GitHub #68) a request's unset field falls back to.
    pub default_enable_thinking: bool,
    /// The server-wide `reasoning_effort` default (`IGNIS_REASONING_EFFORT`).
    pub default_reasoning_effort: Option<ReasoningEffort>,
    /// The Playground's asset table when `--ui` is on (GitHub #163, ADR
    /// 0026); `None` leaves the `/ui` routes out of the router entirely.
    pub playground: Option<playground::Assets>,
    /// The Prometheus projection when `--metrics` is on (GitHub #89, ADR
    /// 0017); `None` installs neither the projection nor any route to it.
    pub metrics: Option<std::sync::Arc<metrics::Metrics>>,
    /// The key `/v1` requests must present (`--api-key` / `IGNIS_API_KEY`);
    /// `None` leaves the API open.
    pub api_key: Option<crate::config::ApiKey>,
    /// Acquires and prepares a request's images before admission (GitHub
    /// #179); `None` on a load without `--vision`.
    pub media: Option<std::sync::Arc<media::MediaAcquirer>>,
    /// Where `system` and `developer` messages go before the conversation is
    /// templated (`--system-message-policy` / `--developer-message-policy`,
    /// GitHub #209).
    pub instruction_policy: instruction::InstructionPolicy,
    /// The labels `/v1/decide` may give a decision's options (GitHub #237,
    /// #239), computed once from the loaded tokenizer.
    ///
    /// Once, because it is a property of the load and not of a request:
    /// deriving it per request would re-encode 624 candidate labels on the
    /// way to serving a prompt of 132. Empty on a provider with no real
    /// tokenizer, and `/v1/decide` refuses rather than guessing.
    pub alphabet: std::sync::Arc<ignis_core::decision::AnswerAlphabet>,
}

impl Server {
    /// A server over `engine`'s scheduler with the given template provider.
    pub fn new(engine: Engine, template: Box<dyn TemplateProvider>) -> Self {
        let template: std::sync::Arc<dyn TemplateProvider> = std::sync::Arc::from(template);
        Self {
            engine,
            alphabet: std::sync::Arc::new(template.answer_alphabet()),
            template,
            request_timeout: Duration::from_secs(crate::config::DEFAULT_REQUEST_TIMEOUT_SECS as u64),
            default_enable_thinking: true,
            default_reasoning_effort: None,
            playground: None,
            metrics: None,
            api_key: None,
            media: None,
            instruction_policy: instruction::InstructionPolicy::default(),
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

    /// Set the non-streaming completion timeout (`main` wires this to
    /// `--request-timeout`/`IGNIS_REQUEST_TIMEOUT`, GitHub #95; the default
    /// is 30 s).
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

    /// Serve the Playground from `assets` under `/ui/` (`main` passes
    /// [`playground::EMBEDDED`] when `--ui` is set; tests inject their own
    /// table, including the empty one that selects the fallback page).
    pub fn with_playground(mut self, assets: playground::Assets) -> Self {
        self.playground = Some(assets);
        self
    }

    /// Turn Prometheus metrics on (`main` calls this when `--metrics` is
    /// set): installs the projection into the engine's telemetry consumer,
    /// which alone keeps it up to date from the facts it already receives.
    /// It is served by [`Server::metrics_app`] and, with the Playground, at
    /// `/ui/metrics`.
    pub fn with_metrics(mut self) -> Self {
        let metrics = std::sync::Arc::new(metrics::Metrics::new());
        self.engine.install_metrics(std::sync::Arc::clone(&metrics));
        self.metrics = Some(metrics);
        self
    }

    /// Record what this load reserved (GitHub #216, ADR 0030
    /// §Observability): the plan's lines and the shapes they bound, written
    /// once, before the first request. Without `--metrics` there is nothing
    /// to write them into and the call does nothing; a load that built no
    /// plan (the placeholder path) never makes it.
    pub fn with_load_reservations(self, reserved: metrics::LoadReservations) -> Self {
        if let Some(metrics) = &self.metrics {
            metrics.set_load_reservations(reserved);
        }
        self
    }

    /// Require `key` as `Authorization: Bearer <key>` on every `/v1` route
    /// (`main` wires this to `--api-key` / `IGNIS_API_KEY`).
    pub fn with_api_key(mut self, key: crate::config::ApiKey) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Acquire image parts with `acquirer` (a `--vision` load, GitHub #179).
    pub fn with_media(mut self, acquirer: std::sync::Arc<media::MediaAcquirer>) -> Self {
        self.media = Some(acquirer);
        self
    }

    /// Place instruction messages under `policy` (`main` wires this to
    /// `--system-message-policy` / `--developer-message-policy`, GitHub #209;
    /// the default is merge / inplace).
    pub fn with_instruction_policy(mut self, policy: instruction::InstructionPolicy) -> Self {
        self.instruction_policy = policy;
        self
    }

    /// The axum app (build once, share across a listener; the state the
    /// router serves is an `Arc` of this server).
    pub fn app(&self) -> Router {
        let state: std::sync::Arc<Server> = std::sync::Arc::new(self.clone());
        api::router(state)
    }

    /// The metrics listener's app (`--metrics-bind`, GitHub #89, ADR 0017):
    /// `GET /metrics` and nothing else, with no API key — it is kept private
    /// by its bind address, and `--expose` never tunnels it. `None` unless
    /// [`Server::with_metrics`] turned metrics on.
    pub fn metrics_app(&self) -> Option<Router> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics::router("/metrics", std::sync::Arc::clone(metrics)))
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
        self.serve_on(listener, None).await
    }

    /// [`Server::serve`] on a listener the caller already bound — `main`
    /// binds first when `--expose` needs the bound port before serving.
    ///
    /// With `metrics_listener` (`--metrics`, GitHub #89), [`Server::metrics_app`]
    /// is served there too, and one process signal stops both.
    pub async fn serve_on(
        self,
        listener: tokio::net::TcpListener,
        metrics_listener: Option<tokio::net::TcpListener>,
    ) -> std::io::Result<()> {
        self.serve_on_until(listener, metrics_listener, shutdown_signal()).await
    }

    /// [`Server::serve_on`], stopping gracefully — both listeners, when there
    /// are two — once `shutdown` resolves instead of on a process signal
    /// (tests). A metrics listener without metrics on
    /// ([`Server::with_metrics`]) is refused with `InvalidInput` before
    /// anything is served.
    pub async fn serve_on_until(
        self,
        listener: tokio::net::TcpListener,
        metrics_listener: Option<tokio::net::TcpListener>,
        shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        use std::future::IntoFuture;

        let app = self.app();
        let (metrics_listener, metrics_app) = match (metrics_listener, self.metrics_app()) {
            (None, _) => {
                return axum::serve(listener, app).with_graceful_shutdown(shutdown).await;
            }
            (Some(metrics_listener), Some(metrics_app)) => (metrics_listener, metrics_app),
            (Some(_), None) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "a metrics listener needs metrics on (Server::with_metrics)",
                ));
            }
        };
        // One shutdown, fanned out to both listeners.
        let (stop, stopped) = tokio::sync::watch::channel(());
        let signal = async move {
            shutdown.await;
            let _ = stop.send(());
            Ok::<(), std::io::Error>(())
        };
        let wait = |mut stopped: tokio::sync::watch::Receiver<()>| async move {
            let _ = stopped.changed().await;
        };
        let api = axum::serve(listener, app).with_graceful_shutdown(wait(stopped.clone()));
        let metrics = axum::serve(metrics_listener, metrics_app).with_graceful_shutdown(wait(stopped));
        tokio::try_join!(signal, api.into_future(), metrics.into_future()).map(|_| ())
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
