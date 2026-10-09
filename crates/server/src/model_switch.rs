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
//!    returns has its scheduler — and the process-wide pinned KV-RAM arena
//!    its leaf owns — been dropped.
//! 5. **Load** — the target is loaded on the freed card and warmed up, then
//!    published whole ([`crate::Server::active`]), and the API serves again.
//!
//! **Why tear down before loading.** The spec's first draft built the new
//! model before releasing the old one, so a failed load could fall back to a
//! model that never stopped. On the card that cannot work at all: the leaf's
//! pinned KV-RAM arena is a process-wide singleton whose create refuses
//! while one exists ("destroy it before creating another", `kernel/src/seq.cu`),
//! and every load sizes its VRAM plan from the memory free at its start
//! (NVML, `crate::runtime`), which the old model still holds. So the failure
//! contract is kept the other way round: a target that can be refused
//! without the GPU is refused in step 2, before anything stops; a load that
//! fails on the GPU after the teardown reloads the **old** model from the
//! artifact it came from ([`crate::ModelSource`]), so a bad switch degrades
//! to a cold restart of what was serving, never to a stopped server. Only a
//! reload that fails too leaves [`ModelStatus::Failed`] standing, with the
//! process up and a further switch still accepted.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::Engine;
use crate::runtime::LoadedModel;
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
/// ([`crate::runtime::prepare_model`] / [`crate::runtime::load_model`]) on
/// the options the server was started with, as a restart with `--artifact`
/// and `--model` changed would run it — except that a flag only the *other*
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
        let named = crate::config::Config {
            artifact: Some(target.artifact.clone()),
            model: target.model.clone(),
            model_named: true,
            ..self.options.clone()
        };
        let (options, dropped) = crate::config::fit_to_family(&named, family);
        if !dropped.is_empty() {
            tracing::info!(
                name: "ignis.model.switch_flags_dropped",
                to = %target.model,
                family = family.name(),
                flags = %dropped.join(","),
                "start flags this model does not take are off for this load"
            );
        }
        let prepared = crate::runtime::prepare_model(&options, &target.artifact).map_err(|e| e.to_string())?;
        Ok(Box::new(prepared))
    }
}

impl PreparedLoad for crate::runtime::PreparedModel {
    fn load(self: Box<Self>) -> Result<LoadedModel, String> {
        crate::runtime::load_model(*self).map_err(|e| e.to_string())
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
}

impl Switcher {
    /// Switch with `loader`, giving the old model's requests `drain_timeout`
    /// to finish (`--switch-drain-timeout`).
    pub fn new(loader: Arc<dyn ModelLoader>, drain_timeout: Duration) -> Self {
        Self { loader, drain_timeout, running: Mutex::new(None) }
    }
}

/// A switch that has begun: the ids it moves between, as `202` reports them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchStarted {
    /// The id served when it began.
    pub from: String,
    /// The id it is loading.
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
        /// Why.
        reason: String,
    },
    /// The target failed to load after the old model was torn down, and the
    /// old model was loaded again from its artifact: it is serving.
    Restored {
        /// Why the target failed.
        reason: String,
    },
    /// The target failed, and the old model could not be loaded again
    /// either: nothing is serving until a switch succeeds.
    Failed {
        /// Why, both failures said.
        reason: String,
    },
}

/// Clears [`Switcher::running`] when the switch task ends, however it ends
/// (a panicking task included), so one bad switch never wedges the next.
struct Running(Arc<Switcher>);

impl Drop for Running {
    fn drop(&mut self) {
        *self.0.running.lock().expect("switch lock") = None;
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
    let task = tokio::spawn(run(server.clone(), Running(switcher), target, started.clone(), serving_before));
    Ok((started, task))
}

/// The switch itself, steps 2 to 5 of the module doc. `serving_before` is
/// whether a model was serving when it began: a switch may also begin from
/// [`ModelStatus::Failed`], the way out of a switch that left nothing
/// serving, and a refusal then must not reopen the gate over a dead engine.
async fn run(
    server: Server,
    running: Running,
    target: ModelSource,
    started: SwitchStarted,
    serving_before: bool,
) -> SwitchOutcome {
    let switcher = Arc::clone(&running.0);
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
/// point `--metrics` at it. A model whose warm-up fails is torn down again
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
