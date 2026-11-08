//! ignis-server: the OpenAI-compatible HTTP surface (localhost, no auth).
//!
//! v1 endpoints (server-01, `docs/archive/ignis-v1.md` §2):
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
pub mod adaptation;
pub mod api;
pub mod artifact_template;
pub mod catalog_http;
pub mod config;
pub mod config_http;
pub mod deadline;
pub mod decide;
pub mod decoder;
pub mod download;
pub mod download_http;
pub mod engine;
pub mod expose;
pub mod idle;
pub mod image_encoder_input;
pub mod image_prompt;
pub mod images;
pub mod instruction;
pub mod load;
pub mod locate;
pub mod loop_guard;
pub mod loader;
pub mod media;
pub mod metrics;
pub mod model_switch;
pub mod playground;
pub mod numbers;
pub mod openai_fields;
pub mod openapi;
pub mod removal_http;
pub mod responses;
pub mod reuse;
pub mod runtime;
pub mod scalar;
pub mod shutdown;
pub mod speech;
pub mod starting;
pub mod stop;
pub mod telemetry;
pub mod template;
pub mod thinking;
pub mod tokenize;
pub mod toolcall;
pub mod user_docs;

use std::time::Duration;

use arc_swap::ArcSwap;
use axum::Router;

pub use crate::active::{ActiveModel, InFlight, ModelSource, ModelStatus, Runner, StatusCell};
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
    /// Where a response's id comes from: random, the default, so no two
    /// responses share one whatever load served them; counted keeps a
    /// test's events byte-for-byte reproducible, as a fixed `wall_clock`
    /// keeps their `created_at`.
    pub response_ids: std::sync::Arc<responses::ResponseIds>,
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
    /// active model. Every store wakes the requests waiting on a load
    /// ([`StatusCell`], spec model-switch/02).
    pub status: std::sync::Arc<StatusCell>,
    /// The requests using the model now, and when it was last used: what the
    /// idle clock reads to unload a model nobody uses (spec model-switch/02,
    /// [`idle`]). Shared by every clone — a pinned copy's request counts on
    /// the server's model.
    pub usage: std::sync::Arc<idle::Usage>,
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
    /// The edits decoding their condition images now, at most
    /// `image.max_condition_decodes` at once ([`images::Decodes`]).
    pub image_decodes: std::sync::Arc<images::Decodes>,
    /// The speech model beside the active one (ADR 0067, spec speech/02):
    /// its availability, its executor and the compute worker behind
    /// `POST /v1/audio/transcriptions`. Shared by every clone, like the active
    /// model; a model switch does not touch it.
    pub speech: std::sync::Arc<speech::Speech>,
    /// What `POST /v1/models/download` fetches with, and the download it
    /// runs (GitHub #376): shared by every clone, like the switcher. `None`
    /// answers the route `501`: a server built in a test without one has no
    /// endpoint to fetch from.
    pub downloads: Option<std::sync::Arc<download_http::Downloads>>,
}

/// The server-level knobs a live config change may touch (spec config-v2/02
/// §`PATCH`): each one read per request, so changing it needs no model
/// reload. The model's own configuration lives with the model and changes
/// only through a reload.
#[derive(Debug, Clone, PartialEq)]
pub struct Live {
    /// The deadline of a request answered whole, from its admission
    /// (`--server-request-timeout`, GitHub #95, ADR 0058); zero is none.
    pub request_timeout: Duration,
    /// The deadline of a request that streams, from its admission
    /// (`--server-stream-timeout`, ADR 0058); zero is none.
    pub stream_timeout: Duration,
    /// What a request past its deadline is answered
    /// (`--server-timeout-reply`, ADR 0058).
    pub timeout_reply: deadline::TimeoutReply,
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
    /// How long the loaded model may go unused before the idle clock unloads
    /// it (`--model-idle-unload`, spec model-switch/02); zero never unloads.
    /// Read by the clock at every tick, so a live change needs no restart.
    pub idle_unload: Duration,
    /// How long a Responses WebSocket may sit idle before the server closes
    /// it (`--server-socket-idle-close`, GitHub #329); zero never closes.
    /// Read by every open socket at least once a second, so a live change
    /// reaches the sockets already open.
    pub socket_idle_close: Duration,
    /// The loop guard (`--model-loop-guard`, ADR 0052): read per request.
    pub loop_guard: loop_guard::LoopGuardParams,
    /// The image model's serving (`image.*`, spec image-model/05): its
    /// queue, what one request may ask for, the condition budget, prompt
    /// enhancement — each read per request, so a live change needs no
    /// reload.
    pub image: config::schema::ImageGroup,
    /// The speech model's serving (`speech.*`, spec speech/02): which model,
    /// its compat names, the clip and queue limits — each read per request,
    /// so a live change needs no reload.
    pub speech: config::schema::SpeechGroup,
    /// The diarizer (`diarizer.*`, spec speech/08): which model, the ghost
    /// rule, the geometries and the speaker cache's constants, read when a
    /// Realtime session's speakers start.
    pub diarizer: config::schema::DiarizerGroup,
}

impl Default for Live {
    /// What a server built without a config runs with: the deadlines'
    /// defaults and their partial reply, thinking on with no effort or
    /// budget, merge / inplace — and no idle unload, no idle socket close and
    /// no loop guard: only a configured server (`main`, a live `PATCH`)
    /// unloads its model, closes an idle socket or rewinds a reasoning, so no
    /// test or in-process harness meets a cold reload, a close or a rewind it
    /// did not ask for.
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(crate::config::DEFAULT_REQUEST_TIMEOUT_SECS as u64),
            stream_timeout: Duration::from_secs(u64::from(crate::config::DEFAULT_STREAM_TIMEOUT_SECS)),
            timeout_reply: deadline::TimeoutReply::default(),
            default_enable_thinking: true,
            default_reasoning_effort: None,
            default_thinking_budget: None,
            instruction_policy: instruction::InstructionPolicy::default(),
            idle_unload: Duration::ZERO,
            socket_idle_close: Duration::ZERO,
            loop_guard: loop_guard::LoopGuardParams { mode: loop_guard::LoopGuardMode::Off, ..Default::default() },
            image: config::schema::ImageGroup::default(),
            speech: config::schema::SpeechGroup::default(),
            diarizer: config::schema::DiarizerGroup::default(),
        }
    }
}

impl Live {
    /// The live knobs `config` names.
    pub fn of(config: &config::Config) -> Self {
        Self {
            request_timeout: Duration::from_secs(u64::from(config.request_timeout_secs)),
            stream_timeout: Duration::from_secs(u64::from(config.stream_timeout_secs)),
            timeout_reply: config.timeout_reply,
            default_enable_thinking: config.enable_thinking,
            default_reasoning_effort: config.reasoning_effort,
            default_thinking_budget: config.thinking_budget,
            instruction_policy: config.instruction_policy,
            idle_unload: Duration::from_secs(u64::from(config.idle_unload_secs)),
            socket_idle_close: Duration::from_secs(u64::from(config.socket_idle_close_secs)),
            loop_guard: config.loop_guard.clone(),
            image: config.settings().image.clone(),
            speech: config.settings().speech.clone(),
            diarizer: config.settings().diarizer.clone(),
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
/// *warm-up request* in GLOSSARY.md's sense (`RequestInput::warm_up`, a
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
    let completion = engine::collect_completion(&mut events, WARM_UP_TIMEOUT)
        .await
        .map_err(|err| format!("the warm-up request did not complete: {err:?}"))?;
    // A request the engine ended — a leaf error, a shutdown under it — is
    // done too, but its traversal never ran: not a warm-up.
    if completion.reason == ignis_core::FinishReason::Error {
        return Err("the warm-up request ended with an error".into());
    }
    Ok(started.elapsed())
}

/// The first traversal of `model` before it serves: a text model's
/// [`warm_up_engine`]. The image model has no decode graphs to capture and
/// none is run (spec image-model/05): it serves once loaded.
pub async fn warm_up_model(model: &ActiveModel) -> Result<Duration, String> {
    match model.engine() {
        Some(engine) => warm_up_engine(engine).await,
        None => Ok(Duration::ZERO),
    }
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
            response_ids: std::sync::Arc::default(),
            seedless_seed: None,
            status: std::sync::Arc::new(StatusCell::new(ModelStatus::Serving)),
            usage: std::sync::Arc::default(),
            switcher: None,
            config: None,
            image_decodes: std::sync::Arc::default(),
            speech: std::sync::Arc::default(),
            downloads: None,
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
        self.speech.live_changed(&live.speech);
        self.speech.diarizer_changed(&live.diarizer);
        self.live.store(std::sync::Arc::new(live));
    }

    /// Make `config`'s server-level values the ones requests read, without
    /// touching the model (spec config-v2/02): the live knobs, and the model
    /// switch's drain window and known models (the catalog's among them, or
    /// not, with implicit switching). What a live `PATCH /v1/config`
    /// applies, and what a reload applies once its model serves.
    pub fn apply_config(&self, config: &config::Config) {
        let live = Live::of(config);
        // The speech worker's idle clock reads `speech.idle_unload` live.
        self.speech.live_changed(&live.speech);
        self.speech.diarizer_changed(&live.diarizer);
        self.live.store(std::sync::Arc::new(live));
        if let Some(switcher) = &self.switcher {
            // The explicit switch by id (spec playground/05) names these
            // whether or not implicit switching is on; only the implicit one
            // is gated by `switch.allow_implicit`.
            switcher.note_start(self.active().source.as_ref());
            let tables = switcher.tables(&config.known_models, config.allow_model_switch);
            switcher.set_listed_models(tables.listed);
            switcher.set_knobs(Duration::from_secs(u64::from(config.switch_drain_timeout_secs)), tables.known, config.allow_model_switch);
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
    /// Narrower than `GET /ready`, which also answers 200 while a model is on
    /// its way (a switch, an idle unload, the load after it).
    pub fn is_ready(&self) -> bool {
        matches!(*self.status(), ModelStatus::Serving)
    }

    /// Whether no model serves and none is on its way (GitHub #327): a failed
    /// switch that could reload nothing, [`ModelStatus::Failed`] not out of
    /// `unloaded`, with the switch's slot free. Only another switch ends it.
    /// While the failed switch is still reloading the old model its slot is
    /// held, and this is false.
    pub fn nothing_serves_or_loads(&self) -> bool {
        matches!(*self.status(), ModelStatus::Failed { from_unloaded: false, .. })
            && !self.switcher.as_ref().is_some_and(|switcher| switcher.running().is_some())
    }

    /// The first traversal (GitHub #129): one two-token request through the
    /// scheduler ([`warm_up_model`]), so the prefill and decode kernels
    /// have run and the decode graphs are captured before a client's request
    /// needs them, then the API is marked ready and `ignis.process.ready`
    /// says how long it took. A warm-up that does not complete is an error
    /// and leaves the server not ready: better refused than silently cold.
    pub async fn warm_up(&self) -> Result<Duration, String> {
        let took = warm_up_model(&self.active()).await?;
        self.status.store(std::sync::Arc::new(ModelStatus::Serving));
        tracing::info!(
            name: "ignis.process.ready",
            warm_up_ms = took.as_millis() as u64,
            "first traversal done: the API admits requests"
        );
        Ok(took)
    }

    /// Run [`Server::warm_up`] now if the API is held back for it, on its
    /// own task so the listener is already answering (503) meanwhile. The
    /// task's handle, so a serve that ends before the warm-up does can
    /// cancel it.
    fn spawn_warm_up(&self) -> Option<tokio::task::JoinHandle<()>> {
        if !matches!(*self.status(), ModelStatus::WarmingUp) {
            return None;
        }
        let server = self.clone();
        Some(tokio::spawn(async move {
            if let Err(error) = server.warm_up().await {
                tracing::error!(name: "ignis.process.warm_up_failed", %error, "still not ready");
            }
        }))
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

    /// Set the deadline of a request answered whole (`main` wires this to
    /// `--server-request-timeout`/`IGNIS_SERVER_REQUEST_TIMEOUT`, GitHub #95,
    /// ADR 0058); zero is none.
    pub fn with_request_timeout(self, timeout: Duration) -> Self {
        self.update_live(|live| live.request_timeout = timeout);
        self
    }

    /// Set the deadline of a request that streams
    /// (`--server-stream-timeout`, ADR 0058); zero is none.
    pub fn with_stream_timeout(self, timeout: Duration) -> Self {
        self.update_live(|live| live.stream_timeout = timeout);
        self
    }

    /// Answer a request past its deadline as `reply` says
    /// (`--server-timeout-reply`, ADR 0058).
    pub fn with_timeout_reply(self, reply: deadline::TimeoutReply) -> Self {
        self.update_live(|live| live.timeout_reply = reply);
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

    /// Set the loop guard (`--model-loop-guard`).
    pub fn with_loop_guard(self, params: loop_guard::LoopGuardParams) -> Self {
        self.update_live(|live| live.loop_guard = params);
        self
    }

    /// Unload the model after `after` with no request using it
    /// (`--model-idle-unload`, spec model-switch/02); zero never does.
    pub fn with_idle_unload(self, after: Duration) -> Self {
        self.update_live(|live| live.idle_unload = after);
        self
    }

    /// Close a Responses WebSocket idle for `after`
    /// (`--server-socket-idle-close`, GitHub #329); zero never does.
    pub fn with_socket_idle_close(self, after: Duration) -> Self {
        self.update_live(|live| live.socket_idle_close = after);
        self
    }

    /// Serve transcriptions with `speech` (ADR 0067): `main` passes the slot
    /// over its executor and the catalog; a server built without one has a
    /// speech model that is unavailable, saying so.
    pub fn with_speech(mut self, speech: std::sync::Arc<speech::Speech>) -> Self {
        speech.live_changed(&self.live().speech);
        speech.diarizer_changed(&self.live().diarizer);
        self.speech = speech;
        self
    }

    /// Serve the speech model as `speech` says (`speech.*`, spec speech/02):
    /// what `main` resolved at start, until a live change applies another.
    pub fn with_speech_group(self, speech: config::schema::SpeechGroup) -> Self {
        self.update_live(|live| live.speech = speech);
        self
    }

    /// Diarize as `diarizer` says (`diarizer.*`, spec speech/08): what `main`
    /// resolved at start, until a live change applies another.
    pub fn with_diarizer_group(self, diarizer: config::schema::DiarizerGroup) -> Self {
        self.update_live(|live| live.diarizer = diarizer);
        self
    }

    /// Serve images as `image` says (`image.*`, spec image-model/05): what
    /// `main` resolved at start, until a live change or a switch applies
    /// another configuration.
    pub fn with_image(self, image: config::schema::ImageGroup) -> Self {
        self.update_live(|live| live.image = image);
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
    /// `/ui/metrics`. The image model has no engine to feed it: the
    /// projection is served as the last text model left it.
    pub fn with_metrics(mut self) -> Self {
        let metrics = std::sync::Arc::new(metrics::Metrics::new());
        if let Some(engine) = self.active().engine() {
            engine.install_metrics(std::sync::Arc::clone(&metrics));
        }
        self.responses.install_metrics(std::sync::Arc::clone(&metrics));
        self.speech.attach_metrics(std::sync::Arc::clone(&metrics));
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

    /// Let `POST /v1/models/download` fetch the switcher's catalog entries
    /// as `downloads` says (GitHub #376): `main` builds it from the running
    /// configuration; tests build their own.
    pub fn with_downloads(mut self, downloads: download_http::Downloads) -> Self {
        self.downloads = Some(std::sync::Arc::new(downloads));
        self
    }

    /// Point what `--server-metrics` keeps at `model`, a model a switch has just
    /// loaded (spec model-switch/01): its engine's telemetry consumer starts
    /// feeding the projection, its own counters (Flash-Next's, GitHub #301)
    /// are read, and `reserved` replaces what the previous load reserved. The
    /// start does the same through [`Server::with_metrics`] and
    /// [`Server::with_load_reservations`]. The image model has no engine to
    /// feed them (spec image-model/05): what a text model's load reserved and
    /// occupied reads as no load, as while unloaded, then its own plan's lines
    /// replace them (GitHub #324). Nothing without `--server-metrics`.
    pub(crate) fn observe_load(&self, model: &ActiveModel, reserved: Option<metrics::LoadReservations>) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        match model.engine() {
            Some(engine) => engine.install_metrics(std::sync::Arc::clone(metrics)),
            None => metrics.set_unloaded(),
        }
        if let Some(reserved) = reserved {
            self.install_reservations(model, reserved);
        }
    }

    /// The idle clock unloaded the model (spec model-switch/02): what
    /// `--server-metrics` keeps shows nothing reserved, until a load's
    /// [`Server::observe_load`] writes its own. Nothing without
    /// `--server-metrics`.
    pub(crate) fn observe_unload(&self) {
        if let Some(metrics) = &self.metrics {
            metrics.set_unloaded();
        }
    }

    /// Read a response's `created_at` from `clock` instead of the wall.
    pub fn with_wall_clock(mut self, clock: std::sync::Arc<dyn telemetry::TelemetryClock>) -> Self {
        self.wall_clock = clock;
        self
    }

    /// Count response ids from 0 instead of drawing them (see
    /// [`Server::response_ids`]).
    pub fn with_counted_response_ids(mut self) -> Self {
        self.response_ids = std::sync::Arc::new(responses::ResponseIds::counted());
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
            if let Some(engine) = model.engine() {
                engine.install_counter_source(reserved.flash_next.take());
            }
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
    /// Runs until Ctrl+C, or SIGTERM on unix, then drains in-flight requests
    /// before returning (GitHub #80, spec §28: graceful shutdown, not an
    /// abrupt kill) — the caller emits `ignis.process.stopped` and flushes
    /// the logging queue after this returns, since that is the true last
    /// event, and on [`Served::LoadAbandoned`] ends the process at once.
    pub async fn serve(self, addr: String) -> std::io::Result<Served> {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        self.serve_on(listener, None).await
    }

    /// [`Server::serve`] on a listener the caller already bound.
    ///
    /// With `metrics_listener` (`--server-metrics`, GitHub #89), [`Server::metrics_app`]
    /// is served there too, and one process signal stops both. The signal
    /// is listened for from this call: `main`, which has a load to stop
    /// before it serves, listens from its start instead
    /// ([`shutdown::StopSignal`], GitHub #329) and calls
    /// [`Server::serve_on_until`] with it.
    pub async fn serve_on(
        self,
        listener: tokio::net::TcpListener,
        metrics_listener: Option<tokio::net::TcpListener>,
    ) -> std::io::Result<Served> {
        let stop = shutdown::StopSignal::install_serving()?;
        self.serve_on_until(listener, metrics_listener, stop.serving()).await
    }

    /// [`Server::serve_on`], stopping gracefully — both listeners, when there
    /// are two — once `shutdown` resolves: `main`'s
    /// [`shutdown::StopSignal::serving`], or a test's own. A metrics
    /// listener without metrics on ([`Server::with_metrics`]) is refused
    /// with `InvalidInput` before anything is served.
    ///
    /// After the HTTP drain, an unload still holding the switch's slot runs
    /// to its end, then the loaded model is shut down and its driver thread
    /// is joined ([`ActiveModel::shut_down`]) before this returns
    /// ([`Served::Drained`]). Dropping the `JoinHandle` without joining would
    /// detach `ignis-model`; on Linux that thread can hang in CUDA teardown
    /// and leave the process a VRAM-holding zombie under init — so serve exit
    /// must join, the same way a model switch does.
    ///
    /// A switch, a reload or a load is not waited for (GitHub #329): once
    /// `shutdown` resolves, the requests waiting for one are answered `503
    /// server_stopping` ([`model_switch::Switcher::stop_waiting`]), and one
    /// still running after the drain is returned
    /// ([`Served::LoadAbandoned`]) with nothing torn down, for the caller to
    /// end the process under it.
    pub async fn serve_on_until(
        self,
        listener: tokio::net::TcpListener,
        metrics_listener: Option<tokio::net::TcpListener>,
        shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<Served> {
        use std::future::IntoFuture;

        let switcher = self.switcher.clone();
        let shutdown = async move {
            shutdown.await;
            if let Some(switcher) = switcher {
                switcher.stop_waiting();
            }
        };
        let metrics = match (metrics_listener, self.metrics_app()) {
            (None, _) => None,
            (Some(metrics_listener), Some(metrics_app)) => Some((metrics_listener, metrics_app)),
            (Some(_), None) => {
                // Refused before anything is served, but the model is loaded:
                // it leaves the card the way it does after a serve.
                model_switch::shut_down(&self.active()).await;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "a metrics listener needs metrics on (Server::with_metrics)",
                ));
            }
        };
        let app = self.app();
        let warm_up = self.spawn_warm_up();
        // Spec model-switch/02: an unused model leaves the card while this
        // serves; the clock stops when serving does.
        let idle_clock = self.spawn_idle_clock(idle::IDLE_TICK);
        let serve_result = match metrics {
            None => axum::serve(listener, app).with_graceful_shutdown(shutdown).await,
            Some((metrics_listener, metrics_app)) => {
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
                let metrics =
                    axum::serve(metrics_listener, metrics_app).with_graceful_shutdown(wait(stopped));
                tokio::try_join!(signal, api.into_future(), metrics.into_future()).map(|_| ())
            }
        };
        // Nothing starts on the model once serving has ended. A warm-up still
        // running would be ended by the teardown below and report its failure
        // on its own task, racing `ignis.process.stopped`; the idle clock
        // could begin an unload in the middle of it. Both are stopped before
        // the teardown begins, the warm-up awaited so no poll of it is left.
        if let Some(warm_up) = warm_up {
            warm_up.abort();
            let _ = warm_up.await;
        }
        drop(idle_clock);
        // An unload under way outlives the drain and runs to its end first. A
        // switch, a reload or a load still running is left as it is: the
        // model it tears down or publishes is the process end's to free, and
        // `active` may be a model it is tearing down.
        if let Some((load, phase)) = model_switch::load_left_running(&self).await {
            return serve_result.map(|()| Served::LoadAbandoned { load, phase });
        }
        // Join before returning to `main`: `ignis.process.stopped` must stay
        // the last log line (spec §28), and that is emitted after this returns.
        model_switch::shut_down(&self.active()).await;
        serve_result.map(|()| Served::Drained)
    }
}

/// How a serve ended ([`Server::serve_on_until`]). A caller must look:
/// returning from `main` on [`Served::LoadAbandoned`] would wait for the
/// load (the runtime's drop joins its blocking thread) and leave the model
/// it publishes un-joined.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Served {
    /// HTTP drained, and the loaded model shut down with its thread joined:
    /// nothing of it is left.
    Drained,
    /// HTTP drained, with a switch, a reload or a load still running: it has
    /// no cancel, and it is not waited for (GitHub #329). Nothing was torn
    /// down: the caller ends the process now ([`shutdown::exit_now`]), which
    /// frees the GPU, and logs nothing that waits for the load.
    LoadAbandoned {
        /// The ids it moves between.
        load: model_switch::SwitchStarted,
        /// What `GET /v1/models` reports it as: `loading` for a load out of
        /// `unloaded`, else `switching`.
        phase: &'static str,
    },
}
