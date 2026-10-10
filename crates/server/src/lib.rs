//! ignis-server: the OpenAI-compatible HTTP surface (localhost, no auth).
//!
//! v1 endpoints (server-01, `docs/design/ignis-v1.md` §2):
//! - `GET /v1/models` — the loaded model, and whether it is serving or
//!   being switched; `POST /v1/models/switch` replaces it on the running
//!   process (spec model-switch/01, `model_switch.rs`).
//! - `POST /v1/chat/completions` — chat completions, streaming (SSE) and
//!   non-streaming; requests route into the core scheduler and tokens
//!   stream back as they are generated.
//! - `POST /v1/responses` — the OpenAI Responses API, streaming or not, and
//!   `GET /v1/responses`, its WebSocket mode (GitHub #282).
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

pub mod active;
pub mod api;
pub mod artifact_template;
pub mod config;
pub mod config_http;
pub mod decide;
pub mod decoder;
pub mod download;
pub mod engine;
pub mod expose;
pub mod instruction;
pub mod load;
pub mod locate;
pub mod loader;
pub mod media;
pub mod metrics;
pub mod model_switch;
pub mod playground;
pub mod numbers;
pub mod openai_fields;
pub mod openapi;
pub mod responses;
pub mod reuse;
pub mod runtime;
pub mod scalar;
pub mod stop;
pub mod telemetry;
pub mod template;
pub mod thinking;
pub mod tokenize;
pub mod toolcall;

use std::time::Duration;

use arc_swap::ArcSwap;
use axum::Router;

pub use crate::active::{ActiveModel, ModelSource, ModelStatus};
use crate::engine::Engine;
use crate::template::TemplateProvider;
use crate::thinking::ReasoningEffort;

/// The server's knobs (constructor injection — the template seam is
/// pluggable here: artifact-02 swaps in the artifact-backed provider).
#[derive(Clone)]
pub struct Server {
    /// The loaded model — engine, template, family, calibrations, decide
    /// labels, vision acquirer — as one value (spec model-switch/01). Read
    /// with [`Server::active`]; a model switch replaces it whole, so a
    /// request that takes it once sees one model throughout. Shared by every
    /// clone of the server, so a switch reaches every handler.
    pub active: std::sync::Arc<ArcSwap<ActiveModel>>,
    /// The server-level knobs a live `PATCH /v1/config` may change without
    /// touching the model (spec config-v2/02), read with [`Server::live`].
    /// Shared by every clone, like the active model, so a change reaches
    /// every handler at once; a request handler's [`Server::pinned`] copy
    /// holds the set in force when the request began, however many times
    /// the request reads it.
    pub live: std::sync::Arc<ArcSwap<Live>>,
    /// The Playground's asset table when `--server-ui` is on (GitHub #163, ADR
    /// 0026); `None` leaves the `/ui` routes out of the router entirely.
    pub playground: Option<playground::Assets>,
    /// The Prometheus projection when `--server-metrics` is on (GitHub #89, ADR
    /// 0017); `None` installs neither the projection nor any route to it.
    pub metrics: Option<std::sync::Arc<metrics::Metrics>>,
    /// The key `/v1` requests must present (`--server-api-key` / `IGNIS_SERVER_API_KEY`);
    /// `None` leaves the API open.
    pub api_key: Option<crate::config::ApiKey>,
    /// The match keys of recent `/v1/decide` parts states at their run ends
    /// (GitHub #270): what an **observed fork** is found in. Shared by every
    /// clone of the server, as the active model is. Keys name the loaded
    /// model's KV, so a model switch empties it in place rather than carry
    /// keys the next model cannot resolve.
    pub fork_history: std::sync::Arc<std::sync::Mutex<reuse::ForkHistory>>,
    /// The next `/v1/decide` fan-out's name (GitHub #270): what its head is
    /// kept under, and given up by.
    pub next_fan_out: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// What every Responses WebSocket shares (GitHub #282): the server-wide
    /// admission queue and the open-socket count.
    pub responses: std::sync::Arc<responses::Hub>,
    /// The wall clock a response's `created_at` is read from (a fixed clock
    /// keeps a test's events byte-for-byte reproducible).
    pub wall_clock: std::sync::Arc<dyn telemetry::TelemetryClock>,
    /// The seed a request that sends none draws with: `None`, the default,
    /// is a fresh one per request (spec server/12); a fixed one keeps a
    /// test's mock token streams reproducible, as a fixed `wall_clock` keeps
    /// its events.
    pub seedless_seed: Option<u64>,
    /// Whether the `/v1` routes admit requests (GitHub #129, generalized by
    /// spec model-switch/01). [`ModelStatus::Serving`] unless
    /// [`Server::with_warm_up`] held it back — every `/v1` route answers 503
    /// `server_not_ready` until the first traversal has run, so no request
    /// pays the decode-graph capture on its first token — or a model switch
    /// is under way (`503 model_switching`). Shared by every clone, like the
    /// active model.
    pub status: std::sync::Arc<ArcSwap<ModelStatus>>,
    /// What `POST /v1/models/switch` loads models with (spec
    /// model-switch/01): the loader, the drain window, the one switch that
    /// may run at a time. `None` answers the route `501`: a server built in
    /// a test without one has no way to load anything.
    pub switcher: Option<std::sync::Arc<model_switch::Switcher>>,
    /// The running configuration, and where a change to it is written down
    /// (`GET`/`PATCH /v1/config`, spec config-v2/02). `None` answers both
    /// routes `501`: a server built in a test without one has no config to
    /// show or change.
    pub config: Option<std::sync::Arc<config_http::ConfigState>>,
}

/// The server-level knobs a live config change may touch (spec config-v2/02
/// §`PATCH`): each one read per request, so changing it needs no model
/// reload. The model's own configuration lives with the model and changes
/// only through a reload.
#[derive(Debug, Clone, PartialEq)]
pub struct Live {
    /// How long a non-streaming request waits for its completion before the
    /// handler gives up with a 504 (guards a wedged engine from hanging the
    /// client forever; `--server-request-timeout`, GitHub #95).
    pub request_timeout: Duration,
    /// The `enable_thinking` a request's unset field falls back to
    /// (`--model-enable-thinking`, GitHub #68).
    pub default_enable_thinking: bool,
    /// The `reasoning_effort` a request's unset field falls back to
    /// (`--model-reasoning-effort`).
    pub default_reasoning_effort: Option<ReasoningEffort>,
    /// The thinking budget a request's unset `thinking_budget` falls back to
    /// (`--model-thinking-budget`, 2026-09-24). `None` = no budget.
    pub default_thinking_budget: Option<u32>,
    /// Where `system` and `developer` messages go before the conversation is
    /// templated (`--server-system-message-policy` /
    /// `--server-developer-message-policy`, GitHub #209).
    pub instruction_policy: instruction::InstructionPolicy,
}

impl Default for Live {
    /// What a server built without a config runs with: the timeout's
    /// default, thinking on with no effort or budget, merge / inplace.
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(crate::config::DEFAULT_REQUEST_TIMEOUT_SECS as u64),
            default_enable_thinking: true,
            default_reasoning_effort: None,
            default_thinking_budget: None,
            instruction_policy: instruction::InstructionPolicy::default(),
        }
    }
}

impl Live {
    /// The live knobs `config` names.
    pub fn of(config: &config::Config) -> Self {
        Self {
            request_timeout: Duration::from_secs(u64::from(config.request_timeout_secs)),
            default_enable_thinking: config.enable_thinking,
            default_reasoning_effort: config.reasoning_effort,
            default_thinking_budget: config.thinking_budget,
            instruction_policy: config.instruction_policy,
        }
    }
}

/// The one token the warm-up prompts with. Id 1 is an ordinary byte-level
/// token (`"`) of the Qwen vocabulary both families load (248,320 entries,
/// ids 0 and 1 the first two printable bytes), not a special token, so it
/// needs no template and no BOS: both families' embeddings take it. What the
/// model says to it is discarded; a prompt of one token is under a KV page,
/// so it publishes no prefix and captures no checkpoint.
const WARM_UP_TOKEN: ignis_core::TokenId = 1;

/// How long the warm-up may take: the first decode captures its graphs.
const WARM_UP_TIMEOUT: Duration = Duration::from_secs(600);

/// The first traversal of `engine` (GitHub #129): one two-token request the
/// telemetry consumer drops as the server's own, run to completion, and how
/// long it took. Shared by the start ([`Server::warm_up`]) and the model
/// switch, which warms the new engine before any client can reach it. Not a
/// *warm-up request* in CONTEXT.md's sense (`RequestInput::warm_up`, a
/// prefill with no decode): this one decodes, to capture the decode graphs.
pub async fn warm_up_engine(engine: &Engine) -> Result<Duration, String> {
    let started = std::time::Instant::now();
    let input = ignis_core::RequestInput {
        decision: None,
        constrained: None,
        forced_literal: None,
        warm_up: false,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: engine.model_id(),
        tokens: vec![WARM_UP_TOKEN],
        params: ignis_core::DecodeParams { max_tokens: Some(2), ..ignis_core::DecodeParams::default() },
    };
    let (_, mut events) = engine
        .submit_with_notes(
            input,
            ignis_core::RequestClass::Interactive,
            engine::RequestNotes { internal: true, ..engine::RequestNotes::default() },
        )
        .await
        .map_err(|err| format!("the warm-up request was refused: {err:?}"))?;
    engine::collect_completion(&mut events, WARM_UP_TIMEOUT)
        .await
        .map_err(|err| format!("the warm-up request did not complete: {err:?}"))?;
    Ok(started.elapsed())
}

impl Server {
    /// A server over `engine`'s scheduler with the given template provider.
    pub fn new(engine: Engine, template: Box<dyn TemplateProvider>) -> Self {
        Self::from_active(ActiveModel::new(engine, std::sync::Arc::from(template)))
    }

    /// A server serving `model`, with every server-level knob at its
    /// default (the `with_*` setters change them). What `main` builds once
    /// the start-up load ([`crate::load::load_model`]) has produced the
    /// model.
    pub fn from_active(model: ActiveModel) -> Self {
        Self {
            active: std::sync::Arc::new(ArcSwap::from_pointee(model)),
            live: std::sync::Arc::new(ArcSwap::from_pointee(Live::default())),
            playground: None,
            metrics: None,
            api_key: None,
            fork_history: std::sync::Arc::default(),
            next_fan_out: std::sync::Arc::default(),
            responses: std::sync::Arc::default(),
            wall_clock: std::sync::Arc::new(telemetry::SystemClock),
            seedless_seed: None,
            status: std::sync::Arc::new(ArcSwap::from_pointee(ModelStatus::Serving)),
            switcher: None,
            config: None,
        }
    }

    /// The live knobs, as of now (wait-free) — on a pinned copy, as of when
    /// the request began.
    pub fn live(&self) -> std::sync::Arc<Live> {
        self.live.load_full()
    }

    /// Change the live knobs: a copy edited by `edit`, stored over them.
    pub fn update_live(&self, edit: impl FnOnce(&mut Live)) {
        let mut live = Live::clone(&self.live());
        edit(&mut live);
        self.live.store(std::sync::Arc::new(live));
    }

    /// Make `config`'s server-level values the ones requests read, without
    /// touching the model (spec config-v2/02): the live knobs, and the model
    /// switch's drain window and known models. What a live `PATCH
    /// /v1/config` applies, and what a reload applies once its model serves.
    pub fn apply_config(&self, config: &config::Config) {
        self.live.store(std::sync::Arc::new(Live::of(config)));
        if let Some(switcher) = &self.switcher {
            let known = if config.allow_model_switch {
                model_switch::known_models(&config.known_models, self.active().source.as_ref())
            } else {
                Default::default()
            };
            switcher.set_knobs(Duration::from_secs(u64::from(config.switch_drain_timeout_secs)), known);
        }
    }

    /// Show and change the running configuration over `GET`/`PATCH
    /// /v1/config` (spec config-v2/02), written down where `state` says.
    pub fn with_config(mut self, state: std::sync::Arc<config_http::ConfigState>) -> Self {
        self.config = Some(state);
        self
    }

    /// The loaded model, as of now (wait-free). Take it once per request
    /// and read every model property off the one value: a model switch that
    /// lands meanwhile then never pairs one model's template with the
    /// next's engine.
    pub fn active(&self) -> std::sync::Arc<ActiveModel> {
        self.active.load_full()
    }

    /// Change the loaded model's bundle in place — a copy edited by `edit`
    /// and stored over it. For building a server (the `with_*` setters
    /// below, tests that adjust a calibration), not while one serves: the
    /// load-edit-store is not atomic against a model switch, which replaces
    /// the bundle whole instead.
    pub fn update_active(&self, edit: impl FnOnce(&mut ActiveModel)) {
        let mut model = ActiveModel::clone(&self.active());
        edit(&mut model);
        self.active.store(std::sync::Arc::new(model));
    }

    /// This server with the loaded model pinned (spec model-switch/01): a
    /// copy whose [`Server::active`] keeps answering the model loaded now,
    /// whatever a switch publishes later. Every request handler takes one
    /// first, so a request reads its template, its engine and the id it
    /// defaults to off one model — never one model's template with the
    /// next's engine — however long it waits on media or between `/v1/decide`
    /// rounds. A request pinned to a model a switch has since torn down is
    /// refused by that engine with `503 engine_full`, which a retry answers.
    /// The live knobs are pinned with the model (spec config-v2/02): a
    /// `PATCH /v1/config` landing mid-request never gives one request two
    /// timeouts or two thinking defaults. Everything else — the status,
    /// metrics, fork history, the switcher — stays shared with the server.
    pub fn pinned(&self) -> Self {
        Self {
            active: std::sync::Arc::new(ArcSwap::new(self.active())),
            live: std::sync::Arc::new(ArcSwap::new(self.live())),
            ..self.clone()
        }
    }

    /// Whether the `/v1` routes are admitting requests, and why not.
    pub fn status(&self) -> std::sync::Arc<ModelStatus> {
        self.status.load_full()
    }

    /// Hold the API back until [`Server::warm_up`] has run (GitHub #129):
    /// the serve loops run it before the first `/v1` request is admitted.
    /// Without this a server is ready as constructed (the mock, the tests).
    pub fn with_warm_up(self) -> Self {
        self.status.store(std::sync::Arc::new(ModelStatus::WarmingUp));
        self
    }

    /// Whether the `/v1` routes admit requests ([`ModelStatus::Serving`]).
    pub fn is_ready(&self) -> bool {
        matches!(*self.status(), ModelStatus::Serving)
    }

    /// The first traversal (GitHub #129): one two-token request through the
    /// scheduler ([`warm_up_engine`]), so the prefill and decode kernels
    /// have run and the decode graphs are captured before a client's request
    /// needs them, then the API is marked ready and `ignis.process.ready`
    /// says how long it took. A warm-up that does not complete is an error
    /// and leaves the server not ready: better refused than silently cold.
    pub async fn warm_up(&self) -> Result<Duration, String> {
        let took = warm_up_engine(&self.active().engine).await?;
        self.status.store(std::sync::Arc::new(ModelStatus::Serving));
        tracing::info!(
            name: "ignis.process.ready",
            warm_up_ms = took.as_millis() as u64,
            "first traversal done: the API admits requests"
        );
        Ok(took)
    }

    /// Run [`Server::warm_up`] now if the API is held back for it, on its
    /// own task so the listener is already answering (503) meanwhile.
    fn spawn_warm_up(&self) {
        if !matches!(*self.status(), ModelStatus::WarmingUp) {
            return;
        }
        let server = self.clone();
        tokio::spawn(async move {
            if let Err(error) = server.warm_up().await {
                tracing::error!(name: "ignis.process.warm_up_failed", %error, "still not ready");
            }
        });
    }

    /// Serve `family` (see [`ActiveModel::family`]).
    pub fn with_family(self, family: ignis_core::compute::ModelFamily) -> Self {
        self.update_active(|model| model.family = family);
        self
    }

    /// The content-part check every endpoint runs before admission (GitHub
    /// #175, #179): an image is refused on a load without vision, and on
    /// Flash-Next, which has no vision tower, the refusal names the model.
    pub fn check_content_parts(
        &self,
        messages: &[template::ChatMessage],
    ) -> Result<(), template::ContentRejection> {
        let model = self.active();
        template::check_content_parts(messages, model.media.is_some()).map_err(|mut rejection| {
            if rejection.code == "vision_disabled" && !model.family.takes_images() {
                rejection.message = format!("{}: {} takes no images", rejection.message, model.family.name());
            }
            rejection
        })
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
    /// `--server-request-timeout`/`IGNIS_SERVER_REQUEST_TIMEOUT`, GitHub #95; the default
    /// is 30 s).
    pub fn with_request_timeout(self, timeout: Duration) -> Self {
        self.update_live(|live| live.request_timeout = timeout);
        self
    }

    /// Give every request that sends no `seed` this one, instead of a fresh
    /// seed per request (see [`Server::seedless_seed`]).
    pub fn with_seedless_seed(mut self, seed: u64) -> Self {
        self.seedless_seed = Some(seed);
        self
    }

    /// Set the server-wide thinking defaults (`IGNIS_MODEL_ENABLE_THINKING` /
    /// `IGNIS_MODEL_REASONING_EFFORT`, GitHub #68). Callers that set a non-trivial
    /// default should validate it against `template.thinking_capabilities()`
    /// first (`thinking::validate_defaults`) — this setter does not, so
    /// tests can construct an out-of-band `Server` without a template to
    /// probe.
    pub fn with_thinking_defaults(self, enable_thinking: bool, reasoning_effort: Option<ReasoningEffort>) -> Self {
        self.update_live(|live| {
            live.default_enable_thinking = enable_thinking;
            live.default_reasoning_effort = reasoning_effort;
        });
        self
    }

    /// Set the server-wide thinking budget (`--model-thinking-budget`).
    pub fn with_thinking_budget(self, budget: Option<u32>) -> Self {
        self.update_live(|live| live.default_thinking_budget = budget);
        self
    }

    /// Serve the Playground from `assets` under `/ui/` (`main` passes
    /// [`playground::EMBEDDED`] when `--server-ui` is set; tests inject their own
    /// table, including the empty one that selects the fallback page).
    pub fn with_playground(mut self, assets: playground::Assets) -> Self {
        self.playground = Some(assets);
        self
    }

    /// Turn Prometheus metrics on (`main` calls this when `--server-metrics` is
    /// set): installs the projection into the engine's telemetry consumer,
    /// which alone keeps it up to date from the facts it already receives.
    /// It is served by [`Server::metrics_app`] and, with the Playground, at
    /// `/ui/metrics`.
    pub fn with_metrics(mut self) -> Self {
        let metrics = std::sync::Arc::new(metrics::Metrics::new());
        self.active().engine.install_metrics(std::sync::Arc::clone(&metrics));
        self.responses.install_metrics(std::sync::Arc::clone(&metrics));
        self.metrics = Some(metrics);
        self
    }

    /// Let `POST /v1/models/switch` replace the loaded model (spec
    /// model-switch/01): `main` installs the production loader over the start
    /// options; tests install their own.
    pub fn with_switcher(mut self, switcher: model_switch::Switcher) -> Self {
        self.switcher = Some(std::sync::Arc::new(switcher));
        self
    }

    /// Point what `--server-metrics` keeps at `model`, a model a switch has just
    /// loaded (spec model-switch/01): its engine's telemetry consumer starts
    /// feeding the projection, its own counters (Flash-Next's, GitHub #301)
    /// are read, and `reserved` replaces what the previous load reserved. The
    /// start does the same through [`Server::with_metrics`] and
    /// [`Server::with_load_reservations`]. Nothing without `--server-metrics`.
    pub(crate) fn observe_load(&self, model: &ActiveModel, reserved: Option<metrics::LoadReservations>) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        model.engine.install_metrics(std::sync::Arc::clone(metrics));
        if let Some(reserved) = reserved {
            self.install_reservations(model, reserved);
        }
    }

    /// Read a response's `created_at` from `clock` instead of the wall.
    pub fn with_wall_clock(mut self, clock: std::sync::Arc<dyn telemetry::TelemetryClock>) -> Self {
        self.wall_clock = clock;
        self
    }

    /// Record what this load reserved (GitHub #216, ADR 0030
    /// §Observability): the plan's lines and the shapes they bound, written
    /// once, before the first request. Without `--server-metrics` there is nothing
    /// to write them into and the call does nothing; a load that built no
    /// plan (the placeholder path) never makes it.
    pub fn with_load_reservations(self, reserved: metrics::LoadReservations) -> Self {
        self.install_reservations(&self.active(), reserved);
        self
    }

    /// Record what `model`'s load reserved in `--server-metrics` (nothing without
    /// it): the plan's lines replace the previous load's, and a Flash-Next
    /// load's counter source — not a reservation; the telemetry consumer
    /// reads it at every tick (GitHub #301, #302) — goes to `model`'s engine.
    fn install_reservations(&self, model: &ActiveModel, mut reserved: metrics::LoadReservations) {
        if let Some(metrics) = &self.metrics {
            model.engine.install_counter_source(reserved.flash_next.take());
            metrics.set_load_reservations(reserved);
        }
    }

    /// Require `key` as `Authorization: Bearer <key>` on every `/v1` route
    /// (`main` wires this to `--server-api-key` / `IGNIS_SERVER_API_KEY`).
    pub fn with_api_key(mut self, key: crate::config::ApiKey) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Acquire image parts with `acquirer` (a `--vision-enabled` load, GitHub #179).
    pub fn with_media(self, acquirer: std::sync::Arc<media::MediaAcquirer>) -> Self {
        self.update_active(|model| model.media = Some(acquirer));
        self
    }

    /// Place instruction messages under `policy` (`main` wires this to
    /// `--server-system-message-policy` / `--server-developer-message-policy`, GitHub #209;
    /// the default is merge / inplace).
    pub fn with_instruction_policy(self, policy: instruction::InstructionPolicy) -> Self {
        self.update_live(|live| live.instruction_policy = policy);
        self
    }

    /// The axum app (build once, share across a listener; the state the
    /// router serves is an `Arc` of this server).
    pub fn app(&self) -> Router {
        let state: std::sync::Arc<Server> = std::sync::Arc::new(self.clone());
        api::router(state)
    }

    /// The metrics listener's app (`--server-metrics-bind`, GitHub #89, ADR 0017):
    /// `GET /metrics` and nothing else, with no API key — it is kept private
    /// by its bind address, and `--server-expose` never tunnels it. `None` unless
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
    /// binds first when `--server-expose` needs the bound port before serving.
    ///
    /// With `metrics_listener` (`--server-metrics`, GitHub #89), [`Server::metrics_app`]
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
        self.spawn_warm_up();
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
