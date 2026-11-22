//! The loaded model as one value, and what the API says about it (spec
//! model-switch/01, GitHub #305).
//!
//! Until the runtime model switch, every property of the loaded artifact sat
//! on [`crate::Server`] as its own field, captured once by `Server::new` and
//! never replaced. A switch has to replace all of them together — a request
//! must never pair one model's template with another model's engine — so
//! they live here, in one [`ActiveModel`] the server holds behind an
//! `ArcSwap` and swaps whole. Everything that is the *server's* rather than
//! the model's (timeouts, keys, the Playground, metrics) stays on `Server`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ignis_core::compute::{ModelFamily, ModelKind};

use crate::engine::Engine;
use crate::images::ImageModel;
use crate::media::MediaAcquirer;
use crate::template::TemplateProvider;

/// Everything on the server that is a property of the loaded artifact
/// rather than of the process: what it computes with — the engine over its
/// scheduler, or the image model's runner — its template, what its family
/// serves, the heads its content hash was calibrated with, the `/v1/decide`
/// labels its tokenizer yields, and its vision acquirer.
///
/// The template, the calibrations, the labels and the acquirer are a text
/// model's: the image model carries the placeholder template and none of
/// the rest, and no route reads them past the kind check
/// (`crate::api::check_served_kind`, spec image-model/05).
///
/// Read wait-free through [`crate::Server::active`] (an `ArcSwap` load), and
/// replaced whole by the model switch (`crate::model_switch`). Cloning is
/// cheap and shares the model thread's handle: a clone is the same loaded
/// model, never a second one.
#[derive(Clone)]
pub struct ActiveModel {
    /// What the model computes with, by its kind (spec image-model/05).
    pub runner: Runner,
    /// The chat-template / tokenizer seam (artifact-02 plugs the real
    /// frontend object set in here).
    pub template: Arc<dyn TemplateProvider>,
    /// The loaded model's family (ADR 0043). Flash-Next has no vision tower
    /// and no readouts: an image or a `/v1/decide` request to it is refused
    /// naming it (spec flash-next/04). The 27B by default.
    pub family: ModelFamily,
    /// The heads `/v1/decide` reads to answer a `point` or a `box` in one
    /// pass (GitHub #260, #263), looked up once from the calibration table
    /// by the loaded artifact's content hash: the pointing head, and the
    /// head set beside it where one was chosen — or `None` on a load nobody
    /// calibrated, and then `point` and `box` answer with the digit chain.
    pub calibration: Option<ignis_core::pointing::Calibration>,
    /// The heads `/v1/decide` votes with to answer a `locate` (GitHub #275),
    /// looked up once by the loaded artifact's content hash as `calibration`
    /// is — or `None` on a load nobody calibrated, which refuses a `locate`.
    pub locate: Option<ignis_core::locate::LocateCalibration>,
    /// The labels `/v1/decide` may give a decision's options (GitHub #237,
    /// #239), computed once from the loaded tokenizer.
    ///
    /// Once, because it is a property of the load and not of a request:
    /// deriving it per request would re-encode 624 candidate labels on the
    /// way to serving a prompt of 132. Empty on a provider with no real
    /// tokenizer, and `/v1/decide` refuses rather than guessing.
    pub alphabet: Arc<ignis_core::decision::AnswerAlphabet>,
    /// Acquires and prepares a request's images before admission (GitHub
    /// #179); `None` on a load without `--vision-enabled`.
    pub media: Option<Arc<MediaAcquirer>>,
    /// What a switch would name to load this model again: the artifact it
    /// came from and the id it is served under. `None` for a model built in
    /// process (the tests' mock engines, the placeholder start) — a switch
    /// away from one that then fails to load its target has nothing to
    /// reload, and says so instead of serving nothing silently.
    pub source: Option<ModelSource>,
    /// Every adaptation the load took to fit its VRAM plan (ADR 0063 §3), as
    /// `ignis.runtime.vram_plan`'s `adapted` spells them: empty when it took
    /// none, as on every load but the 27B's. `GET /v1/models` reports it
    /// beside the context the load runs (#340).
    pub adapted: String,
    /// The model thread's handle (GitHub #71), kept so a switch can join it:
    /// joining is what proves the old scheduler — and with it every GPU
    /// buffer and the process-wide pinned KV-RAM arena its leaf owns — was
    /// dropped before the next load creates its own. A `Mutex<Option<..>>`
    /// rather than the bare handle because the bundle sits in an `Arc` other
    /// readers may still hold when the switch takes it; `None` for an engine
    /// built without one (`Engine::new`, the tests).
    driver: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
}

/// What a loaded model computes with: its [`ModelKind`] as a value.
#[derive(Clone)]
pub enum Runner {
    /// A text model's engine: the core scheduler + per-request event
    /// routing (submit / drive / route — `engine.rs`).
    Text(Engine),
    /// The image model's runner and job count (`crate::images`).
    Image(ImageModel),
}

/// The work waiting and running on a loaded model
/// ([`ActiveModel::in_flight`]): a text model's requests not yet on a lane
/// and on one, the image model's jobs queued and the one running.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InFlight {
    pub waiting: u32,
    pub running: u32,
}

impl InFlight {
    /// Nothing waiting, nothing running.
    pub fn is_idle(&self) -> bool {
        self.waiting == 0 && self.running == 0
    }
}

/// Where a loaded model came from, as a switch request names it (the
/// `--model-artifact` / `--model-id` pair a restart would have been given).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSource {
    /// The `.ninfer` container, a path on the server's machine, with its
    /// sidecar beside it: what the load verifies, names and loads.
    pub artifact: PathBuf,
    /// The id the model is served under — what `GET /v1/models` reports and
    /// what a request must name. A known model's id must be the artifact's
    /// own (`config::Config::for_family`).
    pub model: String,
}

impl ActiveModel {
    /// The model `engine` runs with `template`: the calibrations are looked
    /// up by the engine's artifact hash and the decide labels derived from
    /// the template, as `Server::new` always did; the family is the 27B,
    /// with no media acquirer, no source and no driver handle until the
    /// `with_*` setters say otherwise.
    pub fn new(engine: Engine, template: Arc<dyn TemplateProvider>) -> Self {
        Self {
            calibration: ignis_core::pointing::calibration(engine.artifact()),
            locate: ignis_core::locate::calibration(engine.artifact()),
            alphabet: Arc::new(template.answer_alphabet()),
            runner: Runner::Text(engine),
            template,
            family: ModelFamily::Qwen38_27b,
            media: None,
            source: None,
            adapted: String::new(),
            driver: Arc::default(),
        }
    }

    /// The image model `image` (ADR 0050): its family, the placeholder
    /// template, no calibration, labels nor acquirer, and no source until
    /// [`ActiveModel::with_source`].
    pub fn image(image: ImageModel) -> Self {
        let template: Arc<dyn TemplateProvider> = Arc::new(crate::template::SimpleTemplateProvider);
        Self {
            calibration: None,
            locate: None,
            alphabet: Arc::new(template.answer_alphabet()),
            runner: Runner::Image(image),
            template,
            family: ModelFamily::QwenImage21,
            media: None,
            source: None,
            adapted: String::new(),
            driver: Arc::default(),
        }
    }

    /// What the model produces: what its runner is.
    pub fn kind(&self) -> ModelKind {
        match self.runner {
            Runner::Text(_) => ModelKind::Text,
            Runner::Image(_) => ModelKind::Image,
        }
    }

    /// The id it is served under.
    pub fn model_id(&self) -> String {
        match &self.runner {
            Runner::Text(engine) => engine.model_id(),
            Runner::Image(image) => image.model_id(),
        }
    }

    /// A text model's engine; `None` on the image model, which has none.
    pub fn engine(&self) -> Option<&Engine> {
        match &self.runner {
            Runner::Text(engine) => Some(engine),
            Runner::Image(_) => None,
        }
    }

    /// The engine of a model a text route has checked the kind of
    /// (`crate::api::check_served_kind`): every route that reads it refuses
    /// the image model first, so the image model here is a route that skipped
    /// the check, and that panics naming it rather than serving it.
    pub fn text_engine(&self) -> &Engine {
        self.engine().unwrap_or_else(|| panic!("{} is an image model and has no engine: a text route skipped the kind check", self.model_id()))
    }

    /// The image model's runner and jobs; `None` on a text model.
    pub fn image_model(&self) -> Option<&ImageModel> {
        match &self.runner {
            Runner::Text(_) => None,
            Runner::Image(image) => Some(image),
        }
    }

    /// The work waiting and running on it now: what the idle clock reads to
    /// tell it is in use, and a switch's drain to tell it is done. A text
    /// model's from its engine's wait-free counters, the image model's from
    /// its job count.
    pub fn in_flight(&self) -> InFlight {
        match &self.runner {
            Runner::Text(engine) => {
                let counters = engine.interval_counters();
                InFlight { waiting: counters.waiting, running: counters.running }
            }
            Runner::Image(image) => image.jobs.in_flight(),
        }
    }

    /// Tell the model to stop now, not waiting for it: a text model's thread
    /// ends every request still on it with an error ([`Engine::shutdown`]);
    /// the image model's queue admits nothing more and cancels its jobs
    /// between denoising steps.
    pub fn stop(&self) {
        match &self.runner {
            Runner::Text(engine) => engine.shutdown(),
            Runner::Image(image) => image.jobs.close(),
        }
    }

    /// Keep the model thread's handle (`Engine::with_clock_and_driver`) so a
    /// switch away from this model can join it.
    pub fn with_driver(self, driver: std::thread::JoinHandle<()>) -> Self {
        *self.driver.lock().expect("driver lock") = Some(driver);
        self
    }

    /// Serve `family` (see [`ActiveModel::family`]).
    pub fn with_family(mut self, family: ModelFamily) -> Self {
        self.family = family;
        self
    }

    /// Acquire image parts with `acquirer` (a `--vision-enabled` load, GitHub #179).
    pub fn with_media(mut self, acquirer: Arc<MediaAcquirer>) -> Self {
        self.media = Some(acquirer);
        self
    }

    /// Record what a switch would name to load this model again.
    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Record the adaptations the load took (see [`ActiveModel::adapted`]).
    pub fn with_adapted(mut self, adapted: String) -> Self {
        self.adapted = adapted;
        self
    }

    /// Stop this model for good and wait until it is gone: the model thread
    /// is told to shut down ([`ActiveModel::stop`] — every request still on it
    /// ends with an error), and its handle is joined, which returns only once
    /// the thread has dropped the scheduler and everything the scheduler
    /// owned. The image model's runner is dropped once the job holding it lets
    /// go ([`ImageModel::shut_down`]). Blocking: an async caller runs it on
    /// `spawn_blocking`.
    ///
    /// A model with no driver handle is told to shut down and not waited
    /// for — nothing was kept to wait on. Calling this twice is harmless.
    pub fn shut_down(&self) {
        self.stop();
        if let Runner::Image(image) = &self.runner {
            image.shut_down();
        }
        let driver = self.driver.lock().expect("driver lock").take();
        if let Some(driver) = driver {
            if driver.join().is_err() {
                tracing::error!(
                    name: "ignis.model.thread_panicked",
                    model = %self.model_id(),
                    "the model thread panicked before it shut down"
                );
            }
        }
    }

    /// Whether this model still holds its model thread's handle
    /// ([`ActiveModel::with_driver`]): `false` once [`ActiveModel::shut_down`]
    /// has taken it to join it — the join may still be running — and for a
    /// model built without one (`Engine::new`, most unit tests). A test seam:
    /// nothing in the server reads it, `tests/serve_joins_model_thread.rs`
    /// does.
    pub fn holds_driver(&self) -> bool {
        self.driver.lock().expect("driver lock").is_some()
    }
}

/// Whether the `/v1` routes admit requests, and why not when they do not
/// (generalizes the warm-up's `ready` flag, GitHub #129, to the switch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// Loaded, but its first traversal has not run: `503 server_not_ready`.
    WarmingUp,
    /// Admitting requests.
    Serving,
    /// A model switch is replacing `from` with `to`: `503 model_switching`,
    /// except on the two routes that report and drive the switch.
    Switching {
        /// The id served before the switch.
        from: String,
        /// The id the switch is loading.
        to: String,
    },
    /// The last switch failed for `reason`. Transient when the previous
    /// model could be kept or reloaded (the switch stores `Serving` again
    /// right after); it stays only when nothing could be loaded at all,
    /// and then `/v1` answers `503 server_not_ready` until a switch
    /// succeeds — or, when it was the load leaving [`ModelStatus::Unloaded`]
    /// that failed (`from_unloaded`), every request is admitted and the next
    /// one that needs the model tries the load again (spec model-switch/02
    /// AC 7).
    Failed {
        /// Why, as the switch's log line says it.
        reason: String,
        /// The run that failed left [`ModelStatus::Unloaded`]: nothing was
        /// resident before it, and nothing is to be reloaded, so a request
        /// may load the model again. False after a failed switch, which only
        /// a switch recovers from.
        from_unloaded: bool,
    },
    /// Nobody used the model for `model.idle_unload` seconds, or a client
    /// asked for it (`POST /v1/ignis/models/unload`, ADR 0105), so it was torn
    /// down the way a switch tears one down (spec model-switch/02, ADR 0049):
    /// the process and the API are up, the model's id and source are kept,
    /// and the next request that needs it loads it again.
    Unloaded,
    /// The load that leaves [`ModelStatus::Unloaded`] is running: requests
    /// that need the model wait for it rather than being refused.
    Loading,
}

impl ModelStatus {
    /// The status as `GET /v1/models` reports it.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WarmingUp => "warming_up",
            Self::Serving => "serving",
            Self::Switching { .. } => "switching",
            Self::Failed { .. } => "failed",
            Self::Unloaded => "unloaded",
            Self::Loading => "loading",
        }
    }

    /// The `503` code a `/v1` request is refused with in this state, or
    /// `None` when it is admitted: `server_not_ready` for the warm-up (the
    /// wire contract GitHub #129 set) and for a failed switch — by the time a
    /// client could see that one, the previous model is usually serving
    /// again, so it needs no code of its own — and `model_switching` while
    /// a switch runs. Unloaded, loading, or failed while leaving unloaded,
    /// every request is admitted: the one that needs the model loads it,
    /// behind the API key, and waits (`crate::model_switch::ensure_loaded`).
    pub fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Serving | Self::Unloaded | Self::Loading | Self::Failed { from_unloaded: true, .. } => None,
            Self::WarmingUp | Self::Failed { from_unloaded: false, .. } => Some("server_not_ready"),
            Self::Switching { .. } => Some("model_switching"),
        }
    }

    /// Whether nothing is resident because the model was unloaded, idle or on
    /// request (spec model-switch/02): [`ModelStatus::Unloaded`], or a load out
    /// of it that failed. A request that needs the model loads it from here,
    /// and a run of the switch's slot only loads — nothing to drain, tear
    /// down, or reload when it fails.
    pub fn is_unloaded(&self) -> bool {
        matches!(self, Self::Unloaded | Self::Failed { from_unloaded: true, .. })
    }

    /// Whether no model is on the card for the API to serve: a switch or a
    /// load failed, the model was unloaded, or a load out of that runs. The
    /// model `active` still names was torn down, so the catalog marks no row
    /// loaded.
    pub fn no_model_loaded(&self) -> bool {
        matches!(self, Self::Failed { .. } | Self::Unloaded | Self::Loading)
    }
}

/// The server's [`ModelStatus`]: read wait-free like the `ArcSwap` it wraps,
/// and every store wakes whoever waits for it to change (spec
/// model-switch/02) — the requests holding on a load that leaves
/// [`ModelStatus::Unloaded`], which need a signal, not a poll. Firing on the
/// store itself, rather than at each call site, is what keeps a new store
/// from being one nobody hears.
pub struct StatusCell {
    current: arc_swap::ArcSwap<ModelStatus>,
    changed: tokio::sync::watch::Sender<()>,
}

impl StatusCell {
    /// A cell holding `status`.
    pub fn new(status: ModelStatus) -> Self {
        Self { current: arc_swap::ArcSwap::from_pointee(status), changed: tokio::sync::watch::Sender::new(()) }
    }

    /// The status, as of now (wait-free).
    pub fn load_full(&self) -> Arc<ModelStatus> {
        self.current.load_full()
    }

    /// Replace the status and wake every waiter.
    pub fn store(&self, status: Arc<ModelStatus>) {
        self.current.store(status);
        self.wake();
    }

    /// A receiver that changes at every store from now on. Take it
    /// **before** reading the status it is to wait past, so a store between
    /// the read and the wait is not missed.
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<()> {
        self.changed.subscribe()
    }

    /// Wake every waiter without a store: what the switch's slot does when
    /// it is freed, since a waiter that found it taken waits for that too.
    pub(crate) fn wake(&self) {
        self.changed.send_replace(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec model-switch/02: a load out of `unloaded` that failed is a state
    /// the gate admits — the next request tries the load again — and the
    /// status says so itself, the way it says every other state; a failed
    /// switch is still refused.
    #[test]
    fn a_failed_load_out_of_unloaded_is_admitted_and_a_failed_switch_is_not() {
        let failed = |from_unloaded| ModelStatus::Failed { reason: "x".into(), from_unloaded };
        assert_eq!(failed(true).refusal(), None);
        assert_eq!(failed(false).refusal(), Some("server_not_ready"));
        assert_eq!((failed(true).as_str(), failed(false).as_str()), ("failed", "failed"));
        assert!(failed(true).is_unloaded() && ModelStatus::Unloaded.is_unloaded());
        assert!(!failed(false).is_unloaded() && !ModelStatus::Loading.is_unloaded() && !ModelStatus::Serving.is_unloaded());
        for status in [failed(true), failed(false), ModelStatus::Unloaded, ModelStatus::Loading] {
            assert!(status.no_model_loaded(), "{status:?}");
        }
        let switching = ModelStatus::Switching { from: "a".into(), to: "b".into() };
        for status in [ModelStatus::Serving, ModelStatus::WarmingUp, switching] {
            assert!(!status.no_model_loaded(), "{status:?}");
        }
    }
}
