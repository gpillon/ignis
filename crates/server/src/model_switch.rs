//! The runtime model switch (spec model-switch/01, GitHub #305): one process
//! serves the 27B or Qwen3.8-Flash-Next, and `POST /v1/models/switch` moves
//! it from one to the other without a restart.
//!
//! A switch is a **full reload**: the old model's VRAM, pinned host memory,
//! KV-RAM and retained slots are all freed before the new model allocates
//! anything, so the two are never resident together and the switch itself
//! spends no VRAM. In order, on one spawned task ([`begin`]):
//!
//! 1. **Gate** — [`ModelStatus::Switching`] is stored, and the `/v1` routes
//!    answer `503 model_switching` (`crate::api`).
//! 2. **Prepare** — everything that can refuse the target without the GPU
//!    ([`ModelLoader::prepare`]: the file, its checksum, its model, the flags
//!    it takes). A refusal here touches nothing: the old model never stopped.
//! 3. **Drain** — the old engine's in-flight counts (the same wait-free
//!    snapshot `/metrics` reads) are polled until nothing is waiting or
//!    running, or `--switch-drain-timeout` passes.
//! 4. **Tear down** — the old model thread is told to shut down (whatever is
//!    still running ends with an error) and joined: only once the join
//!    returns has its scheduler — and every GPU buffer and pinned KV-RAM
//!    arena its leaf owns — been dropped.
//! 5. **Load** — the target is loaded on the freed card and warmed up, then
//!    published whole ([`crate::Server::active`]), and the API serves again.
//!
//! **Why tear down before loading.** The spec's first draft built the new
//! model before releasing the old one, so a failed load could fall back to a
//! model that never stopped. On the card that cannot work at all, in either
//! direction: every load sizes its VRAM plan from the memory NVML reports
//! free at its start (`crate::runtime`), which the old model still holds, so
//! the new plan would either refuse or be built around the old weights — and
//! the two models resident together is what the spec rules out. On top of
//! that, the 27B's KV-RAM arena is a process-wide singleton whose create
//! refuses while one exists ("destroy it before creating another",
//! `kernel/src/seq.cu`; Flash-Next's arena is its own instance's). So the
//! failure contract is kept the other way round: a target that can be refused
//! without the GPU is refused in step 2, before anything stops; a load that
//! fails on the GPU after the teardown reloads the **old** model from the
//! artifact it came from ([`crate::ModelSource`]), so a bad switch degrades
//! to a cold restart of what was serving, never to a stopped server. Only a
//! reload that fails too leaves [`ModelStatus::Failed`] standing, with the
//! process up and a further switch still accepted.
//!
//! **Who may begin one.** `POST /v1/models/switch`, naming the artifact; and
//! (spec §Implicit switch) a request whose own `model` names another model the
//! operator listed ([`implicit_switch`]), which waits for that same switch and
//! is then served on the model it named. Both run [`begin`]: one mechanism,
//! two callers.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::Engine;
use crate::load::LoadedModel;
use crate::{ActiveModel, ModelSource, ModelStatus, Server};

/// How often the drain reads the old engine's in-flight counts. A switch is
/// an operator's action, seconds long; 10 ms of latency on noticing the last
/// request left is nothing beside the load that follows.
const DRAIN_POLL: Duration = Duration::from_millis(10);

/// Loads models for a switch — the seam the tests replace (ADR 0006: the
/// orchestration is proven on `MockCompute`, the GPU load by its own
/// `#[ignore]`d test).
pub trait ModelLoader: Send + Sync {
    /// Everything that can refuse `target` without touching the GPU, done
    /// while the old model is still serving; the error says why. Blocking
    /// (it reads and verifies files): called on `spawn_blocking`.
    ///
    /// `options` is the configuration to load with when a reload changes it
    /// (`PATCH /v1/config`, spec config-v2/02: already patched and validated,
    /// fitted to the family it reloads); `None` is the running one, fitted
    /// to `target`'s family as a switch fits it.
    fn prepare(&self, target: &ModelSource, options: Option<&crate::config::Config>) -> Result<Box<dyn PreparedLoad>, String>;
}

/// A target [`ModelLoader::prepare`] accepted: what is left is the load
/// itself, on a card the old model has already left.
pub trait PreparedLoad: Send {
    /// Load the model and start its engine. Blocking for the length of a GPU
    /// load: called on `spawn_blocking`, inside the runtime (the engine
    /// spawns its telemetry consumer).
    fn load(self: Box<Self>) -> Result<LoadedModel, String>;

    /// The configuration the load runs with, fitted to its family: what
    /// `GET /v1/config` shows once it serves. `None` for a loader that keeps
    /// none (the tests' mock).
    fn options(&self) -> Option<&crate::config::Config> {
        None
    }
}

/// The production [`ModelLoader`]: the start-up load path
/// ([`crate::load::prepare_model`] / [`crate::load::load_model`]) on
/// the running configuration, as a restart with `--model-artifact` and
/// `--model-id` changed would run it — except that a value only the *other*
/// model takes is dropped rather than refused
/// ([`crate::config::fit_to_family`]), and said so.
///
/// The running configuration is the server's own
/// ([`crate::config_http::ConfigState`]), shared: a live `PATCH /v1/config`
/// is in it, so a later switch loads with the change, and a reload that
/// failed is not, so the model it falls back to loads as it did before.
pub struct ArtifactLoader {
    state: Arc<crate::config_http::ConfigState>,
}

impl ArtifactLoader {
    /// Load every switch target with `options`, kept by the loader alone (a
    /// test's loader; `main` shares the server's with
    /// [`ArtifactLoader::sharing`]).
    pub fn new(options: crate::config::Config) -> Self {
        Self::sharing(Arc::new(crate::config_http::ConfigState::new(options, Arc::new(crate::config::file::NoFiles))))
    }

    /// Load every switch target with the server's running configuration.
    pub fn sharing(state: Arc<crate::config_http::ConfigState>) -> Self {
        Self { state }
    }
}

impl ModelLoader for ArtifactLoader {
    fn prepare(&self, target: &ModelSource, options: Option<&crate::config::Config>) -> Result<Box<dyn PreparedLoad>, String> {
        if !target.artifact.exists() {
            return Err(format!("no such file: {}", target.artifact.display()));
        }
        // The family first: which flags this load can take depends on it.
        let family = crate::loader::artifact_family(&target.artifact)
            .map_err(|e| e.to_string())?
            .unwrap_or(ignis_core::compute::ModelFamily::Qwen38_27b);
        let options = match options {
            // A reload with a changed config: patched, validated and fitted
            // by the caller, for the model it reloads.
            Some(options) if options.basis.family() == Some(family) => options.clone(),
            given => {
                // The target's artifact and id over the running options, as
                // a patch: the options are resolved again from their
                // sources, so a value set on the struct would not survive
                // the fit.
                let base = given.cloned().unwrap_or_else(|| crate::config::Config::clone(&self.state.current()));
                let named = crate::config::target_patch(&target.artifact, &target.model)
                    .and_then(|patch| base.general_with(&patch))
                    .map_err(|e| e.to_string())?;
                let (options, dropped) = crate::config::fit_to_family(&named, family).map_err(|e| e.to_string())?;
                log_flags_dropped(&target.model, family, &dropped);
                options
            }
        };
        let prepared = crate::load::prepare_model(&options, &target.artifact).map_err(|e| e.to_string())?;
        Ok(Box::new(prepared))
    }
}

/// Say which of the running options a switch to `to`, a model of `family`,
/// left off because the family does not take them (spec config-v2/01 AC 8):
/// `ignis.model.switch_flags_dropped`, nothing when nothing was dropped.
fn log_flags_dropped(to: &str, family: ignis_core::compute::ModelFamily, dropped: &[String]) {
    if dropped.is_empty() {
        return;
    }
    tracing::info!(
        name: "ignis.model.switch_flags_dropped",
        to = %to,
        family = family.name(),
        flags = %dropped.join(","),
        "start flags this model does not take are off for this load"
    );
}

impl PreparedLoad for crate::load::PreparedModel {
    fn load(self: Box<Self>) -> Result<LoadedModel, String> {
        crate::load::load_model(*self).map_err(|e| e.to_string())
    }

    fn options(&self) -> Option<&crate::config::Config> {
        Some(self.config())
    }
}

/// What a server switches models with (`Server::with_switcher`): the loader,
/// the drain window, and the one switch that may run at a time.
pub struct Switcher {
    loader: Arc<dyn ModelLoader>,
    /// `--switch-drain-timeout`; a live `PATCH /v1/config` may change it.
    drain_timeout: Mutex<Duration>,
    /// The switch under way, if any. Switches are serialized, not queued:
    /// a second one is refused naming this one (spec model-switch/01).
    running: Mutex<Option<SwitchStarted>>,
    /// The models a request's own `model` may switch to, each with the
    /// artifact it loads from ([`implicit_switch`]). Empty unless
    /// [`Switcher::with_known_models`] named some — `main` names none under
    /// `--switch-allow-implicit false` — and then a request naming another
    /// model is refused by name, as it always was. A live `PATCH
    /// /v1/config` may change it.
    known: Mutex<BTreeMap<String, PathBuf>>,
    /// The catalog a request's `model` is looked up in when `known` does not
    /// name it ([`Switcher::with_catalog`]), and whether it is — off with
    /// `--switch-allow-implicit false`, which a live `PATCH /v1/config` may
    /// change. Kept apart from `known` on purpose: `known` is the operator's
    /// list, written back to the config file; the catalog's contribution is
    /// derived, never written anywhere.
    catalog: Option<CatalogModels>,
    catalog_on: AtomicBool,
}

/// A catalog's entries as known models (spec model-download/02 §Known
/// models): an entry whose artifact is under `dir` **when a request names
/// it**, so a `model download` into a running server's `download.path` makes
/// it switchable with no restart — and one not there is refused as an
/// unknown model is, since a switch never starts a download.
#[derive(Debug, Clone)]
pub struct CatalogModels {
    /// The merged catalog the server started with.
    pub catalog: Arc<crate::download::Catalog>,
    /// `download.path`.
    pub dir: PathBuf,
}

impl CatalogModels {
    /// The artifact a request naming `model` would switch to: the entry
    /// whose id it is, exactly as a known model's is matched, once its
    /// artifact is on disk.
    fn artifact(&self, model: &str) -> Option<PathBuf> {
        let entry = self.catalog.entries().iter().find(|entry| entry.id == model)?;
        let path = entry.artifact_path(&self.dir);
        path.is_file().then_some(path)
    }
}

impl Switcher {
    /// Switch with `loader`, giving the old model's requests `drain_timeout`
    /// to finish (`--switch-drain-timeout`).
    pub fn new(loader: Arc<dyn ModelLoader>, drain_timeout: Duration) -> Self {
        Self {
            loader,
            drain_timeout: Mutex::new(drain_timeout),
            running: Mutex::new(None),
            known: Mutex::new(BTreeMap::new()),
            catalog: None,
            catalog_on: AtomicBool::new(false),
        }
    }

    /// Let a request naming one of `known`'s ids switch to it
    /// ([`implicit_switch`]); `main` passes [`known_models`]' table.
    pub fn with_known_models(self, known: BTreeMap<String, PathBuf>) -> Self {
        *self.known.lock().expect("known models lock") = known;
        self
    }

    /// Let a request naming a catalog entry whose artifact is on disk switch
    /// to it too, when `on` (`--switch-allow-implicit`). An explicit
    /// known-models entry wins for its id.
    pub fn with_catalog(self, catalog: CatalogModels, on: bool) -> Self {
        self.catalog_on.store(on, Ordering::SeqCst);
        Self { catalog: Some(catalog), ..self }
    }

    /// Change the drain window, the known models and whether the catalog is
    /// consulted, on a running server (a live `PATCH /v1/config`, spec
    /// config-v2/02): the next switch reads them; one already under way
    /// keeps what it began with.
    pub fn set_knobs(&self, drain_timeout: Duration, known: BTreeMap<String, PathBuf>, catalog_on: bool) {
        *self.drain_timeout.lock().expect("drain timeout lock") = drain_timeout;
        *self.known.lock().expect("known models lock") = known;
        self.catalog_on.store(catalog_on, Ordering::SeqCst);
    }

    /// The drain window a switch beginning now gets.
    pub fn drain_timeout(&self) -> Duration {
        *self.drain_timeout.lock().expect("drain timeout lock")
    }

    /// Every model the operator's list and the start make switchable, with
    /// its artifact — the catalog's entries apart, since which of them are
    /// switchable is decided on disk at each request.
    pub fn known_models(&self) -> BTreeMap<String, PathBuf> {
        self.known.lock().expect("known models lock").clone()
    }

    /// The artifact a request naming `model` would switch to, if it is known:
    /// the known models' entry for it, else a catalog entry on disk.
    fn known(&self, model: &str) -> Option<PathBuf> {
        if let Some(listed) = self.known.lock().expect("known models lock").get(model) {
            return Some(listed.clone());
        }
        if !self.catalog_on.load(Ordering::SeqCst) {
            return None;
        }
        self.catalog.as_ref()?.artifact(model)
    }
}

/// The models a request may switch to by naming them: the operator's
/// (`--switch-known-models`, [`crate::config::Config::known_models`]) and the one the
/// server started on, `start` — its served id and the artifact it loaded,
/// which only its load knows. An operator's entry for the start's own id
/// gives way to the start's, which is the artifact actually loaded and so the
/// one a switch back must reload; `ignis.config.known_model_replaced` says so
/// when the two paths differ. A start built in process (the placeholder) has
/// no artifact and adds nothing.
pub fn known_models(named: &BTreeMap<String, PathBuf>, start: Option<&ModelSource>) -> BTreeMap<String, PathBuf> {
    let mut known = named.clone();
    if let Some(start) = start {
        if let Some(listed) = known.insert(start.model.clone(), start.artifact.clone()) {
            if listed != start.artifact {
                tracing::warn!(
                    name: "ignis.config.known_model_replaced",
                    model = %start.model,
                    listed = %listed.display(),
                    loaded = %start.artifact.display(),
                    "--switch-known-models names the start model at another path; a switch back reloads the one loaded"
                );
            }
        }
    }
    known
}

/// A switch that has begun: the ids it moves between, as `202` reports them
/// and as a second switch's `409` names the one under way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchStarted {
    /// The id served when it began — the model that is drained and torn
    /// down, and reloaded if the target fails on the GPU.
    pub from: String,
    /// The id it is loading: the request's `model`, which the target
    /// artifact's own model must not contradict.
    pub to: String,
}

/// Why a switch did not begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchRefusal {
    /// The server was built without a [`Switcher`].
    Unavailable,
    /// The first traversal of the loaded model has not run yet; switching
    /// under it would race its own warm-up.
    WarmingUp,
    /// Another switch is under way (`409`): this is it.
    InProgress(SwitchStarted),
    /// A reload with a changed config (`PATCH /v1/config`) was asked of a
    /// model not loaded from an artifact (the placeholder start): there is
    /// nothing to load it from again.
    NotReloadable,
}

/// How a switch ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchOutcome {
    /// The target is serving.
    Switched,
    /// The target was refused before anything was torn down: the old model
    /// never stopped serving.
    Refused {
        /// What refused it — a path with no file, a checksum that is not
        /// clean, flags the target cannot take — as the prepare step said
        /// it, and as `ignis.model.switch_failed` logs it.
        reason: String,
    },
    /// The target failed to load after the old model was torn down, and the
    /// old model was loaded again from its artifact: it is serving.
    Restored {
        /// Why the target failed on the GPU (or in its warm-up).
        reason: String,
    },
    /// The target failed, and the old model could not be loaded again
    /// either: nothing is serving until a switch succeeds.
    Failed {
        /// Why, both failures said: the target's, then the reload's.
        reason: String,
    },
}

/// Ends a switch's hold on the server when its task ends, however it ends:
/// [`Switcher::running`] is cleared so the next switch may begin, and a task
/// that ended without settling the status — a panic, or a runtime shutting
/// down under it — leaves [`ModelStatus::Failed`] rather than a gate stuck
/// at `Switching` with nobody left to open it.
struct RunningGuard {
    switcher: Arc<Switcher>,
    status: Arc<arc_swap::ArcSwap<ModelStatus>>,
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        if matches!(**self.status.load(), ModelStatus::Switching { .. }) {
            let reason = "the switch ended before it finished (see ignis.model.switch_panicked)".to_owned();
            tracing::error!(
                name: "ignis.model.switch_panicked",
                panicking = std::thread::panicking(),
                "the switch task ended mid-switch; a switch to a model that loads is the way back"
            );
            self.status.store(Arc::new(ModelStatus::Failed { reason }));
        }
        *self.switcher.running.lock().expect("switch lock") = None;
    }
}

/// Begin switching `server` to `target`: the gate closes now
/// ([`ModelStatus::Switching`]), and the rest — prepare, drain, tear down,
/// load — runs on the returned task, so an HTTP caller answers `202` without
/// waiting on a model load. Must be called inside a tokio runtime.
pub fn begin(
    server: &Server,
    target: ModelSource,
) -> Result<(SwitchStarted, tokio::task::JoinHandle<SwitchOutcome>), SwitchRefusal> {
    begin_with(server, target, None)
}

/// A reload asked for by a config change (spec config-v2/02 §`PATCH`).
struct Reconfigure {
    /// The configuration to load with: the running one with `patch` on top,
    /// validated and fitted to the loaded model's family by the caller.
    config: crate::config::Config,
    /// The change itself, written to the config file once the reload has
    /// taken.
    patch: crate::config::source::Layer,
}

/// Reload the loaded model with `config` — the running configuration with
/// `patch` applied, already validated — through the very switch [`begin`]
/// runs (spec config-v2/02 §`PATCH` step 4): gate, drain, teardown, load,
/// and the old model reloaded as it was if the new load fails. Only once
/// the reloaded model serves do `config` become the running configuration,
/// its server-level values take effect, and `patch` reach the config file —
/// never part of it before, never any of it after a failure. The task runs
/// detached; `GET /v1/models` and `GET /v1/config` report it.
pub fn reconfigure(
    server: &Server,
    config: crate::config::Config,
    patch: crate::config::source::Layer,
) -> Result<SwitchStarted, SwitchRefusal> {
    if server.switcher.is_none() {
        return Err(SwitchRefusal::Unavailable);
    }
    let target = server.active().source.clone().ok_or(SwitchRefusal::NotReloadable)?;
    begin_with(server, target, Some(Reconfigure { config, patch })).map(|(started, _task)| started)
}

/// [`begin`], with the config change a reload carries, if any.
fn begin_with(
    server: &Server,
    target: ModelSource,
    reconfigure: Option<Reconfigure>,
) -> Result<(SwitchStarted, tokio::task::JoinHandle<SwitchOutcome>), SwitchRefusal> {
    let switcher = server.switcher.clone().ok_or(SwitchRefusal::Unavailable)?;
    let (started, before) = {
        let mut running = switcher.running.lock().expect("switch lock");
        if let Some(started) = running.as_ref() {
            return Err(SwitchRefusal::InProgress(started.clone()));
        }
        let before = server.status();
        if matches!(*before, ModelStatus::WarmingUp) {
            return Err(SwitchRefusal::WarmingUp);
        }
        let started = SwitchStarted { from: server.active().engine.model_id(), to: target.model.clone() };
        *running = Some(started.clone());
        (started, before)
    };
    server
        .status
        .store(Arc::new(ModelStatus::Switching { from: started.from.clone(), to: started.to.clone() }));
    tracing::info!(
        name: "ignis.model.switch_started",
        from = %started.from,
        to = %started.to,
        artifact = %target.artifact.display(),
        "switching models: /v1 answers 503 model_switching until it is done"
    );
    let serving_before = matches!(*before, ModelStatus::Serving);
    let guard = RunningGuard { switcher, status: Arc::clone(&server.status) };
    let task = tokio::spawn(run(server.clone(), guard, target, started.clone(), serving_before, reconfigure));
    Ok((started, task))
}

/// Why a request whose `model` named another model was not served on it
/// ([`implicit_switch`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplicitRefusal {
    /// A switch was already under way when the request named its model —
    /// another request's, or one `POST /v1/models/switch` began — whatever
    /// its target, this request's own included. Not joined, not queued
    /// behind, not retargeted (spec §Implicit switch, ordering 5): answered
    /// `503 model_switching`, exactly as the gate answers a request sent a
    /// moment later.
    Switching(SwitchStarted),
    /// The switch this request began did not leave `to` serving — refused
    /// before anything stopped, or failed on the GPU with the previous model
    /// reloaded or not. Never served on another model instead: the client
    /// named one, and gets it or the reason it did not.
    Failed {
        /// The model the request named.
        to: String,
        /// The switch's own reason, as `GET /v1/models` reports a failed
        /// switch's.
        reason: String,
    },
}

/// Switch to the model a request named, and wait until it serves (spec
/// model-switch/01 §Implicit switch): what lets an OpenAI client that names
/// the other model in `model` simply be answered by it, the first such
/// request taking as long as the switch takes.
///
/// `requested` is the request's `model` with its lane tag already stripped.
/// Nothing named, the model already loaded, or one the switcher does not know
/// (switching off, or a name `--switch-known-models` never listed) is not this
/// function's to refuse: it returns at once, and the request goes on to the
/// check every endpoint already makes, which refuses a model it does not load
/// by name. A known model other than the loaded one begins the very switch
/// `POST /v1/models/switch` would ([`begin`]) — the gate closes before this
/// returns control to anything else, the old model's requests drain, it is
/// torn down, the named one loads — and only once that switch has ended does
/// this return: `Ok` with the named model serving, so the caller, taking the
/// loaded model only now, runs exactly as if it had always been loaded.
///
/// Call it on the server every handler shares, before
/// [`crate::Server::pinned`]: a pinned copy holds a model of its own, and a
/// switch published into it would reach no one.
pub async fn implicit_switch(server: &Server, requested: Option<&str>) -> Result<(), ImplicitRefusal> {
    let Some(requested) = requested.filter(|model| !model.is_empty()) else {
        return Ok(());
    };
    let Some(artifact) = server.switcher.as_ref().and_then(|switcher| switcher.known(requested)) else {
        return Ok(());
    };
    if server.active().engine.model_id() == requested {
        return Ok(());
    }
    let target = ModelSource { artifact: artifact.clone(), model: requested.to_owned() };
    let task = match begin(server, target) {
        Ok((started, task)) => {
            tracing::info!(
                name: "ignis.model.switch_requested",
                from = %started.from,
                to = %started.to,
                "a request named a known model other than the loaded one; it is served once the switch lands"
            );
            task
        }
        Err(SwitchRefusal::InProgress(running)) => return Err(ImplicitRefusal::Switching(running)),
        // None reaches a handler: there is a switcher, the gate holds every
        // request until the warm-up has run, and only a reload refuses a
        // model with no artifact. Were one to, the request's own model check
        // refuses it by name.
        Err(SwitchRefusal::Unavailable | SwitchRefusal::WarmingUp | SwitchRefusal::NotReloadable) => return Ok(()),
    };
    let failed = |reason: String| Err(ImplicitRefusal::Failed { to: requested.to_owned(), reason });
    match task.await {
        Ok(SwitchOutcome::Switched) => Ok(()),
        Ok(SwitchOutcome::Refused { reason } | SwitchOutcome::Restored { reason } | SwitchOutcome::Failed { reason }) => {
            failed(reason)
        }
        Err(ended) => failed(format!("the switch ended before it finished: {ended}")),
    }
}

/// The switch itself, steps 2 to 5 of the module doc. `serving_before` is
/// whether a model was serving when it began: a switch may also begin from
/// [`ModelStatus::Failed`], the way out of a switch that left nothing
/// serving, and a refusal then must not reopen the gate over a dead engine.
async fn run(
    server: Server,
    running: RunningGuard,
    target: ModelSource,
    started: SwitchStarted,
    serving_before: bool,
    reconfigure: Option<Reconfigure>,
) -> SwitchOutcome {
    let switcher = Arc::clone(&running.switcher);
    let began = Instant::now();
    let SwitchStarted { from, to } = started;

    let options = reconfigure.as_ref().map(|reconfigure| reconfigure.config.clone());
    let prepared = match prepare(&switcher, target.clone(), options).await {
        Ok(prepared) => prepared,
        Err(reason) => {
            tracing::error!(
                name: "ignis.model.switch_failed",
                from = %from,
                to = %to,
                stage = "prepare",
                %reason,
                "the switch was refused before anything stopped; the old model is as it was"
            );
            // Failed for one tick (`GET /v1/models` may read it), then the
            // old model, which never stopped, serves again — or, when the
            // switch began from a failed one, nothing still serves.
            server.status.store(Arc::new(ModelStatus::Failed { reason: reason.clone() }));
            if serving_before {
                server.status.store(Arc::new(ModelStatus::Serving));
            }
            return SwitchOutcome::Refused { reason };
        }
    };

    // What the configuration will be once this load serves: the patched
    // one for a reload, the one the loader fitted to the target otherwise.
    let serving_config = match &reconfigure {
        Some(reconfigure) => Some(reconfigure.config.clone()),
        None => prepared.options().cloned(),
    };
    let old = server.active();
    let drain_began = Instant::now();
    let drain_timeout = switcher.drain_timeout();
    if let Err(left) = drain(&old.engine, drain_timeout).await {
        // The window is over: what is still running is cut now, before the
        // line saying so — whoever reads it can count on the cut being under
        // way. The join below waits for it.
        old.engine.shutdown();
        tracing::warn!(
            name: "ignis.model.switch_drain_timed_out",
            from = %from,
            to = %to,
            waiting = left.waiting,
            running = left.running,
            drain_timeout_ms = drain_timeout.as_millis() as u64,
            "requests still on the old model past the drain window are cancelled"
        );
    }
    let drain_ms = drain_began.elapsed().as_millis() as u64;

    let teardown_began = Instant::now();
    shut_down(&old).await;
    let teardown_ms = teardown_began.elapsed().as_millis() as u64;

    let load_began = Instant::now();
    match load_and_warm(&server, prepared).await {
        Ok((new, warm_up)) => {
            publish(&server, new);
            // Spec config-v2/02: the configuration changes only now, with a
            // model serving under it — and is written down only now.
            if let Some(config) = serving_config {
                server.apply_config(&config);
                if let Some(state) = &server.config {
                    state.publish(config);
                    if let Some(reconfigure) = &reconfigure {
                        state.persist(&reconfigure.patch);
                    }
                }
            }
            tracing::info!(
                name: "ignis.model.switched",
                from = %from,
                to = %to,
                artifact = %target.artifact.display(),
                drain_ms,
                teardown_ms,
                load_ms = (load_began.elapsed() - warm_up).as_millis() as u64,
                warm_up_ms = warm_up.as_millis() as u64,
                took_ms = began.elapsed().as_millis() as u64,
                "model switched: the API admits requests"
            );
            server.status.store(Arc::new(ModelStatus::Serving));
            SwitchOutcome::Switched
        }
        Err(reason) => {
            tracing::error!(
                name: "ignis.model.switch_failed",
                from = %from,
                to = %to,
                stage = "load",
                %reason,
                "the target failed to load after the old model was torn down; reloading the old model"
            );
            server.status.store(Arc::new(ModelStatus::Failed { reason: reason.clone() }));
            restore(&server, &switcher, &old, reason).await
        }
    }
}

/// Step 2: [`ModelLoader::prepare`] off the async workers, with `options`
/// when a reload changes the configuration.
async fn prepare(
    switcher: &Arc<Switcher>,
    target: ModelSource,
    options: Option<crate::config::Config>,
) -> Result<Box<dyn PreparedLoad>, String> {
    let loader = Arc::clone(&switcher.loader);
    tokio::task::spawn_blocking(move || loader.prepare(&target, options.as_ref()))
        .await
        .unwrap_or_else(|panicked| Err(format!("preparing the load panicked: {panicked}")))
}

/// Step 3: wait until `engine` has nothing waiting or running, or `timeout`
/// passes — then what was still in flight.
async fn drain(engine: &Engine, timeout: Duration) -> Result<(), crate::telemetry::IntervalCounters> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let counters = engine.interval_counters();
        if counters.waiting == 0 && counters.running == 0 {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(counters);
        }
        tokio::time::sleep(DRAIN_POLL).await;
    }
}

/// Step 4: stop `model` and wait until its thread has dropped everything it
/// held ([`ActiveModel::shut_down`]), off the async workers.
async fn shut_down(model: &Arc<ActiveModel>) {
    let model = Arc::clone(model);
    if let Err(panicked) = tokio::task::spawn_blocking(move || model.shut_down()).await {
        tracing::error!(name: "ignis.model.teardown_panicked", error = %panicked, "tearing the old model down panicked");
    }
}

/// Step 5 up to publishing: load `prepared`, run its first traversal, and
/// point `--server-metrics` at it. A model whose warm-up fails is torn down again
/// and reported as a failed load: better refused than silently cold. Returns
/// the model and how long its warm-up took.
async fn load_and_warm(server: &Server, prepared: Box<dyn PreparedLoad>) -> Result<(ActiveModel, Duration), String> {
    let loaded = tokio::task::spawn_blocking(move || prepared.load())
        .await
        .unwrap_or_else(|panicked| Err(format!("the load panicked: {panicked}")))?;
    let warm_up = match crate::warm_up_engine(&loaded.model.engine).await {
        Ok(took) => took,
        Err(error) => {
            shut_down(&Arc::new(loaded.model)).await;
            return Err(error);
        }
    };
    server.observe_load(&loaded.model, loaded.reservations);
    Ok((loaded.model, warm_up))
}

/// Make `model` the one every request reads, with a fresh fork history: the
/// old one's match keys name KV the new model never computed.
fn publish(server: &Server, model: ActiveModel) {
    *server.fork_history.lock().expect("fork history lock") = crate::reuse::ForkHistory::default();
    server.active.store(Arc::new(model));
    crate::decide::log_load_heads(&server.active());
}

/// After a failed load: load `old` again from where it came from, so a bad
/// switch ends with the previous model serving. `reason` is why the target
/// failed.
async fn restore(server: &Server, switcher: &Arc<Switcher>, old: &ActiveModel, reason: String) -> SwitchOutcome {
    let Some(source) = old.source.clone() else {
        let reason = format!("{reason}; the previous model was not loaded from an artifact, so there is nothing to reload");
        tracing::error!(name: "ignis.model.switch_restore_failed", %reason, "nothing is serving until a switch succeeds");
        server.status.store(Arc::new(ModelStatus::Failed { reason: reason.clone() }));
        return SwitchOutcome::Failed { reason };
    };
    // The running configuration, unchanged: a reload that failed never
    // published its own.
    let restored = match prepare(switcher, source.clone(), None).await {
        Ok(prepared) => load_and_warm(server, prepared).await.map(|(model, _)| model),
        Err(error) => Err(error),
    };
    match restored {
        Ok(model) => {
            publish(server, model);
            tracing::warn!(
                name: "ignis.model.switch_restored",
                model = %source.model,
                artifact = %source.artifact.display(),
                %reason,
                "the switch failed; the previous model was reloaded and serves again"
            );
            server.status.store(Arc::new(ModelStatus::Serving));
            SwitchOutcome::Restored { reason }
        }
        Err(again) => {
            let reason = format!("{reason}; reloading {} failed too: {again}", source.model);
            tracing::error!(name: "ignis.model.switch_restore_failed", %reason, "nothing is serving until a switch succeeds");
            server.status.store(Arc::new(ModelStatus::Failed { reason: reason.clone() }));
            SwitchOutcome::Failed { reason }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::compute::ModelFamily;

    fn logged(run: impl FnOnce()) -> Vec<serde_json::Value> {
        use tracing_subscriber::layer::SubscriberExt;
        let sink = Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, run);
        sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect()
    }

    /// Spec config-v2/01 AC 8: a value the target family cannot take is
    /// dropped, and the switch says which, from the dropped list the fit
    /// returned — the list each field's own applicability produces.
    #[test]
    fn a_switch_names_what_it_dropped_and_says_nothing_when_nothing_was() {
        let started = match crate::config::resolve(&["--vision-enabled".to_owned()], |_| None).unwrap() {
            crate::config::ConfigOutcome::Config(config) => config,
            other => panic!("{other:?}"),
        };
        let (_, dropped) = crate::config::fit_to_family(&started, ModelFamily::FlashNext).unwrap();
        let records = logged(|| log_flags_dropped("qwen3.8-flash-next", ModelFamily::FlashNext, &dropped));
        assert_eq!(records.len(), 1, "{records:?}");
        let event = &records[0];
        assert_eq!(event["event_name"], "ignis.model.switch_flags_dropped", "{event}");
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
        assert_eq!(field("flags"), Some("--vision-enabled".into()), "{event}");
        assert_eq!(field("to"), Some("qwen3.8-flash-next".into()), "{event}");
        assert!(logged(|| log_flags_dropped("qwen3.8-27b", ModelFamily::Qwen38_27b, &[])).is_empty());
    }
}
