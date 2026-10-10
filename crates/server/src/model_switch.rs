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
    fn prepare(&self, target: &ModelSource) -> Result<Box<dyn PreparedLoad>, String>;
}

/// A target [`ModelLoader::prepare`] accepted: what is left is the load
/// itself, on a card the old model has already left.
pub trait PreparedLoad: Send {
    /// Load the model and start its engine. Blocking for the length of a GPU
    /// load: called on `spawn_blocking`, inside the runtime (the engine
    /// spawns its telemetry consumer).
    fn load(self: Box<Self>) -> Result<LoadedModel, String>;
}

/// The production [`ModelLoader`]: the start-up load path
/// ([`crate::load::prepare_model`] / [`crate::load::load_model`]) on
/// the options the server was started with, as a restart with `--model-artifact`
/// and `--model-id` changed would run it — except that a flag only the *other*
/// model takes is dropped rather than refused
/// ([`crate::config::fit_to_family`]), and said so.
pub struct ArtifactLoader {
    options: crate::config::Config,
}

impl ArtifactLoader {
    /// Load every switch target with the start options `options`.
    pub fn new(options: crate::config::Config) -> Self {
        Self { options }
    }
}

impl ModelLoader for ArtifactLoader {
    fn prepare(&self, target: &ModelSource) -> Result<Box<dyn PreparedLoad>, String> {
        if !target.artifact.exists() {
            return Err(format!("no such file: {}", target.artifact.display()));
        }
        // The family first: which flags this load can take depends on it.
        let family = crate::loader::artifact_family(&target.artifact)
            .map_err(|e| e.to_string())?
            .unwrap_or(ignis_core::compute::ModelFamily::Qwen38_27b);
        // The target's artifact and id over the start options, as a patch:
        // the options are resolved again from their sources, so a value set
        // on the struct would not survive the fit.
        let named = crate::config::target_patch(&target.artifact, &target.model)
            .and_then(|patch| self.options.with_patch(&patch))
            .map_err(|e| e.to_string())?;
        let (options, dropped) = crate::config::fit_to_family(&named, family).map_err(|e| e.to_string())?;
        if !dropped.is_empty() {
            tracing::info!(
                name: "ignis.model.switch_flags_dropped",
                to = %target.model,
                family = family.name(),
                flags = %dropped.join(","),
                "start flags this model does not take are off for this load"
            );
        }
        let prepared = crate::load::prepare_model(&options, &target.artifact).map_err(|e| e.to_string())?;
        Ok(Box::new(prepared))
    }
}

impl PreparedLoad for crate::load::PreparedModel {
    fn load(self: Box<Self>) -> Result<LoadedModel, String> {
        crate::load::load_model(*self).map_err(|e| e.to_string())
    }
}

/// What a server switches models with (`Server::with_switcher`): the loader,
/// the drain window, and the one switch that may run at a time.
pub struct Switcher {
    loader: Arc<dyn ModelLoader>,
    drain_timeout: Duration,
    /// The switch under way, if any. Switches are serialized, not queued:
    /// a second one is refused naming this one (spec model-switch/01).
    running: Mutex<Option<SwitchStarted>>,
    /// The models a request's own `model` may switch to, each with the
    /// artifact it loads from ([`implicit_switch`]). Empty unless
    /// [`Switcher::with_known_models`] named some — `main` names none under
    /// `--switch-allow-implicit false` — and then a request naming another
    /// model is refused by name, as it always was.
    known: BTreeMap<String, PathBuf>,
}

impl Switcher {
    /// Switch with `loader`, giving the old model's requests `drain_timeout`
    /// to finish (`--switch-drain-timeout`).
    pub fn new(loader: Arc<dyn ModelLoader>, drain_timeout: Duration) -> Self {
        Self { loader, drain_timeout, running: Mutex::new(None), known: BTreeMap::new() }
    }

    /// Let a request naming one of `known`'s ids switch to it
    /// ([`implicit_switch`]); `main` passes [`known_models`]' table.
    pub fn with_known_models(mut self, known: BTreeMap<String, PathBuf>) -> Self {
        self.known = known;
        self
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
    let task = tokio::spawn(run(server.clone(), guard, target, started.clone(), serving_before));
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
    let Some(artifact) = server.switcher.as_ref().and_then(|switcher| switcher.known.get(requested)) else {
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
        // Neither reaches a handler: there is a switcher, and the gate holds
        // every request until the warm-up has run. Were one to, the request's
        // own model check refuses it by name.
        Err(SwitchRefusal::Unavailable | SwitchRefusal::WarmingUp) => return Ok(()),
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
) -> SwitchOutcome {
    let switcher = Arc::clone(&running.switcher);
    let began = Instant::now();
    let SwitchStarted { from, to } = started;

    let prepared = match prepare(&switcher, target.clone()).await {
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

    let old = server.active();
    let drain_began = Instant::now();
    if let Err(left) = drain(&old.engine, switcher.drain_timeout).await {
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
            drain_timeout_ms = switcher.drain_timeout.as_millis() as u64,
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

/// Step 2: [`ModelLoader::prepare`] off the async workers.
async fn prepare(switcher: &Arc<Switcher>, target: ModelSource) -> Result<Box<dyn PreparedLoad>, String> {
    let loader = Arc::clone(&switcher.loader);
    tokio::task::spawn_blocking(move || loader.prepare(&target))
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
    let restored = match prepare(switcher, source.clone()).await {
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
