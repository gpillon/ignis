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

use ignis_core::compute::ModelFamily;

use crate::engine::Engine;
use crate::media::MediaAcquirer;
use crate::template::TemplateProvider;

/// Everything on the server that is a property of the loaded artifact
/// rather than of the process: the engine over its scheduler, its template,
/// what its family serves, the heads its content hash was calibrated with,
/// the `/v1/decide` labels its tokenizer yields, and its vision acquirer.
///
/// Read wait-free through [`crate::Server::active`] (an `ArcSwap` load), and
/// replaced whole by the model switch (`crate::model_switch`). Cloning is
/// cheap and shares the model thread's handle: a clone is the same loaded
/// model, never a second one.
#[derive(Clone)]
pub struct ActiveModel {
    /// The engine: the core scheduler + per-request event routing
    /// (submit / drive / route — `engine.rs`).
    pub engine: Engine,
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
    /// #179); `None` on a load without `--vision`.
    pub media: Option<Arc<MediaAcquirer>>,
    /// What a switch would name to load this model again: the artifact it
    /// came from and the id it is served under. `None` for a model built in
    /// process (the tests' mock engines, the placeholder start) — a switch
    /// away from one that then fails to load its target has nothing to
    /// reload, and says so instead of serving nothing silently.
    pub source: Option<ModelSource>,
    /// The model thread's handle (GitHub #71), kept so a switch can join it:
    /// joining is what proves the old scheduler — and with it every GPU
    /// buffer and the process-wide pinned KV-RAM arena its leaf owns — was
    /// dropped before the next load creates its own. A `Mutex<Option<..>>`
    /// rather than the bare handle because the bundle sits in an `Arc` other
    /// readers may still hold when the switch takes it; `None` for an engine
    /// built without one (`Engine::new`, the tests).
    driver: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
}

/// Where a loaded model came from, as a switch request names it (the
/// `--artifact` / `--model` pair a restart would have been given).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSource {
    /// The `.ninfer` container, a path on the server's machine, with its
    /// sidecar beside it: what the load verifies, names and loads.
    pub artifact: PathBuf,
    /// The id the model is served under — what `GET /v1/models` reports and
    /// what a request must name. A known model's id must be the artifact's
    /// own (`config::served_model_for`).
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
            engine,
            template,
            family: ModelFamily::Qwen38_27b,
            media: None,
            source: None,
            driver: Arc::default(),
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

    /// Acquire image parts with `acquirer` (a `--vision` load, GitHub #179).
    pub fn with_media(mut self, acquirer: Arc<MediaAcquirer>) -> Self {
        self.media = Some(acquirer);
        self
    }

    /// Record what a switch would name to load this model again.
    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Stop this model for good and wait until it is gone: the model thread
    /// is told to shut down ([`Engine::shutdown`] — every request still on it
    /// ends with an error), and its handle is joined, which returns only once
    /// the thread has dropped the scheduler and everything the scheduler
    /// owned. Blocking: an async caller runs it on `spawn_blocking`.
    ///
    /// A model with no driver handle is told to shut down and not waited
    /// for — nothing was kept to wait on. Calling this twice is harmless.
    pub fn shut_down(&self) {
        self.engine.shutdown();
        let driver = self.driver.lock().expect("driver lock").take();
        if let Some(driver) = driver {
            if driver.join().is_err() {
                tracing::error!(
                    name: "ignis.model.thread_panicked",
                    model = %self.engine.model_id(),
                    "the model thread panicked before it shut down"
                );
            }
        }
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
    /// succeeds.
    Failed {
        /// Why, as the switch's log line says it.
        reason: String,
    },
}

impl ModelStatus {
    /// The status as `GET /v1/models` reports it.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WarmingUp => "warming_up",
            Self::Serving => "serving",
            Self::Switching { .. } => "switching",
            Self::Failed { .. } => "failed",
        }
    }

    /// The `503` code a `/v1` request is refused with in this state, or
    /// `None` while serving: `server_not_ready` for the warm-up (the wire
    /// contract GitHub #129 set) and for a failed switch — by the time a
    /// client could see that one, the previous model is usually serving
    /// again, so it needs no code of its own — and `model_switching` while
    /// a switch runs.
    pub fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Serving => None,
            Self::WarmingUp | Self::Failed { .. } => Some("server_not_ready"),
            Self::Switching { .. } => Some("model_switching"),
        }
    }
}
