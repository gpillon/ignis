//! Loading a model (spec model-switch/01): the sequence `main` has always
//! run at start — find, verify and name the artifact, check the start
//! options against it, build its template, then load it on the GPU and start
//! its engine — as two functions the model switch runs too.
//!
//! [`prepare_model`] is everything that can refuse a load short of the GPU;
//! [`load_model`] is the load itself. The split is what lets a switch refuse
//! a bad target while the old model is still serving. The image model (ADR
//! 0050) takes the same two steps: its artifact verified, then its runner
//! loaded (spec image-model/05). Start-up code, not the
//! per-token path: its logging is the once-per-load kind (`hotpath_lint`
//! watches `runtime.rs`, the scheduler construction it calls into).

use std::sync::Arc;

use ignis_core::compute::{ModelFamily, ModelKind};
use ignis_core::image::ImageRunner;
use ignis_core::{ConcreteScheduler, SchedulerConfig, TokenId};

use crate::runtime::EngineShape;

/// The mock-backed scheduler (ADR 0006, CPU-only): what serves when no
/// artifact is configured, or the binary was not built with `--features
/// cuda` — the entrypoint never silently blocks startup on a missing GPU
/// backend.
pub fn mock_scheduler(model: &str, default_max_tokens: u32) -> Box<dyn ignis_core::Scheduler> {
    let compute: Arc<dyn ignis_core::Compute> = Arc::new(ignis_core::mock::MockCompute::new());
    Box::new(ConcreteScheduler::with_config(
        SchedulerConfig {
            model: model.into(),
            // ADR 0045: the operator's default cap holds on the mock too.
            default_max_tokens,
            ..SchedulerConfig::default()
        },
        compute,
    ))
}

/// The one count of request ids every model this process loads draws from:
/// a switch or a reload builds a new scheduler, and its requests go on from
/// where the last load's stopped, so a `request_id` in the log names one
/// request for the process's life (ADR 0012), not one per load.
fn process_request_ids() -> ignis_core::RequestIds {
    static IDS: std::sync::OnceLock<ignis_core::RequestIds> = std::sync::OnceLock::new();
    IDS.get_or_init(ignis_core::RequestIds::default).clone()
}

/// The engine over a loaded `scheduler`, its request ids drawn from
/// [`process_request_ids`], and its model thread's handle: how every model
/// this process serves text with starts — the start-up load, a switch's, a
/// reload's ([`load_model`]), and the placeholder ([`placeholder_model`]).
fn start_engine(mut scheduler: Box<dyn ignis_core::Scheduler>) -> (crate::engine::Engine, std::thread::JoinHandle<()>) {
    scheduler.share_request_ids(process_request_ids());
    crate::engine::Engine::with_clock_and_driver(scheduler, Arc::new(crate::telemetry::SystemClock))
}

/// What a start with no artifact serves as `model`: the mock compute under
/// the plain template, its engine started as a load's is.
pub fn placeholder_model(model: &str, default_max_tokens: u32) -> crate::ActiveModel {
    let (engine, driver) = start_engine(mock_scheduler(model, default_max_tokens));
    crate::ActiveModel::new(engine, Arc::new(crate::template::SimpleTemplateProvider)).with_driver(driver)
}

/// Why loading a model refused (spec model-switch/01): each variant is one
/// of the refusals the start-up load has always logged under its own event
/// name ([`LoadError::log`], which `main` calls before it exits). The model
/// switch meets the same refusals and does not stop the process: it reports
/// one through its `Display` as the `reason` of `ignis.model.switch_failed`
/// and of `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The artifact path names no file (`ignis.artifact.missing`).
    ArtifactMissing,
    /// No sidecar to verify the artifact against
    /// (`ignis.artifact.sidecar_missing`).
    SidecarMissing(String),
    /// The artifact did not verify, or its identity could not be read
    /// (`ignis.artifact.load_failed`).
    ArtifactInvalid(String),
    /// The start options and the artifact's model disagree
    /// (`ignis.config.model_mismatch`, spec flash-next/04).
    ModelMismatch(String),
    /// A default thinking budget with no close to force
    /// (`ignis.config.thinking_budget_inert`, spec server/08).
    ThinkingBudgetInert(String),
    /// A thinking default the template cannot honour
    /// (`ignis.config.thinking_invalid`, GitHub #68).
    ThinkingInvalid(String),
    /// The artifact's vision processor does not fit the model contract
    /// (`ignis.vision.processor_invalid`, GitHub #179).
    VisionProcessorInvalid(String),
    /// Vision is on, and Flash-Next's vision companion is missing beside the
    /// artifact or is not its own (`ignis.vision.companion_refused`, GitHub
    /// #557, ADR 0087): the refusal names the file.
    VisionCompanion(ignis_core::flash_next_vision::CompanionRefused),
    /// `generation_config.json` carries no `eos_token_id`
    /// (`ignis.model.eos_missing`): a backend that can never stop on EOS
    /// would run every request to its cap.
    EosMissing,
    /// The model needs a route the card does not run
    /// (`ignis.model.route_unsupported`, GitHub #456): kernels this build
    /// compiles for its arch-specific Blackwell targets alone.
    RouteUnsupported(String),
    /// The GPU load itself failed (`ignis.model.load_failed`).
    LoadFailed(String),
    /// A host plan refused the load -- Flash-Next's, or the 27B's (ADR
    /// 0062): not enough RAM for its pinned lines and the margin
    /// (`ignis.model.load_failed`; the lines in bytes are
    /// `ignis.runtime.flash_next_host_plan_refused`'s or
    /// `ignis.runtime.qwen27b_host_plan_refused`'s).
    HostPlan(ignis_core::residency::HostPlanError),
}

impl From<crate::runtime::Qwen27bLoadError> for LoadError {
    fn from(error: crate::runtime::Qwen27bLoadError) -> Self {
        match error {
            crate::runtime::Qwen27bLoadError::HostPlan(refusal) => Self::HostPlan(refusal),
            // The VRAM plan's refusal reads as it always did: its deficit and
            // the adaptations tried (ADR 0063 §3).
            crate::runtime::Qwen27bLoadError::VramPlan(refusal) => Self::LoadFailed(refusal.to_string()),
            crate::runtime::Qwen27bLoadError::Other(message) => Self::LoadFailed(message),
        }
    }
}

impl From<crate::runtime::FlashNextLoadError> for LoadError {
    fn from(error: crate::runtime::FlashNextLoadError) -> Self {
        match error {
            crate::runtime::FlashNextLoadError::HostPlan(refusal) => Self::HostPlan(refusal),
            crate::runtime::FlashNextLoadError::PinnedExperts(refusal) => Self::LoadFailed(refusal.to_string()),
            crate::runtime::FlashNextLoadError::Other(message) => Self::LoadFailed(message),
        }
    }
}

impl LoadError {
    /// A Flash-Next load's failure under `config`: a pinned hot set it cannot
    /// hold names its budget as `config` took it (GitHub #329 item 12) -- the
    /// flag, the environment variable, the file key or the profile -- and
    /// anything else reads as [`LoadError::from`] has it.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn from_flash_next(error: crate::runtime::FlashNextLoadError, config: &crate::config::Config) -> Self {
        match error {
            crate::runtime::FlashNextLoadError::PinnedExperts(refusal) => {
                Self::LoadFailed(refusal.naming(&config.basis.resolution().spelled("vram", "pinned_expert_bytes")))
            }
            error => Self::from(error),
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArtifactMissing => write!(f, "no such file"),
            Self::EosMissing => write!(f, "generation_config.json has no eos_token_id"),
            Self::SidecarMissing(message)
            | Self::ArtifactInvalid(message)
            | Self::ModelMismatch(message)
            | Self::ThinkingBudgetInert(message)
            | Self::ThinkingInvalid(message)
            | Self::VisionProcessorInvalid(message)
            | Self::RouteUnsupported(message)
            | Self::LoadFailed(message) => write!(f, "{message}"),
            Self::VisionCompanion(refusal) => write!(f, "{refusal}"),
            Self::HostPlan(refusal) => write!(f, "{refusal}"),
        }
    }
}

impl LoadError {
    /// A missing sidecar's refusal, naming `command`, the command that
    /// fetches it, when there is one (GitHub #497,
    /// [`crate::download::sidecar_fetch_command`]). Every other refusal is
    /// returned as it is.
    pub fn naming_the_fetch(self, command: Option<String>) -> Self {
        match (self, command) {
            (Self::SidecarMissing(error), Some(command)) => {
                Self::SidecarMissing(format!("{error}; the catalog lists it: `{command}` fetches the sidecar beside it and keeps the artifact"))
            }
            (other, _) => other,
        }
    }

    /// Log the refusal of `artifact` under the event name the start-up load
    /// has always used for it, with `outcome` (what happens next: "refusing
    /// to start") as the message. Event names are static in `tracing`, so
    /// each variant names its own.
    pub fn log(&self, artifact: &std::path::Path, outcome: &str) {
        let artifact = artifact.display();
        match self {
            Self::ArtifactMissing => tracing::error!(
                name: "ignis.artifact.missing",
                %artifact,
                "no such file — {outcome}; drop --model-artifact/IGNIS_MODEL_ARTIFACT to fetch the model into --download-path instead"
            ),
            Self::SidecarMissing(error) => {
                tracing::error!(name: "ignis.artifact.sidecar_missing", %artifact, %error, "{outcome}")
            }
            Self::ArtifactInvalid(error) => {
                tracing::error!(name: "ignis.artifact.load_failed", %artifact, %error, "{outcome}")
            }
            Self::ModelMismatch(error) => tracing::error!(name: "ignis.config.model_mismatch", %error, "{outcome}"),
            Self::ThinkingBudgetInert(error) => {
                tracing::error!(name: "ignis.config.thinking_budget_inert", %error, "{outcome}")
            }
            Self::ThinkingInvalid(error) => tracing::error!(name: "ignis.config.thinking_invalid", %error, "{outcome}"),
            Self::VisionProcessorInvalid(error) => {
                tracing::error!(name: "ignis.vision.processor_invalid", %error, "{outcome}")
            }
            Self::VisionCompanion(error) => {
                tracing::error!(name: "ignis.vision.companion_refused", %artifact, %error, "{outcome}")
            }
            Self::EosMissing => tracing::error!(
                name: "ignis.model.eos_missing",
                %artifact,
                "generation_config.json has no eos_token_id — {outcome}"
            ),
            Self::RouteUnsupported(error) => {
                tracing::error!(name: "ignis.model.route_unsupported", %artifact, %error, "{outcome}")
            }
            Self::LoadFailed(error) => tracing::error!(name: "ignis.model.load_failed", %artifact, %error, "{outcome}"),
            Self::HostPlan(error) => tracing::error!(name: "ignis.model.load_failed", %artifact, %error, "{outcome}"),
        }
    }
}

/// A model load with everything that can refuse it short of the GPU already
/// done (spec model-switch/01): the artifact found, verified and named, the
/// start options checked against its model, its template, thinking close and
/// vision processor built. [`load_model`] finishes it.
///
/// The split is what lets a model switch refuse a bad target — a wrong path,
/// a checksum that is not clean, flags the target cannot take — while the
/// old model is still serving: only [`load_model`] needs the GPU the old
/// model holds.
pub struct PreparedModel {
    /// The container the load opens.
    pub artifact: std::path::PathBuf,
    /// The id the load is served under ([`crate::config::Config::for_family`]).
    pub model: String,
    /// The artifact's model (ADR 0043).
    pub family: ModelFamily,
    /// The start options the load runs with.
    config: crate::config::Config,
    /// What only a text model's load prepares; `None` for the image model,
    /// whose runner — the adapter [`cuda_image_runner`] builds — reads the
    /// tokenizer and templates itself (spec image-model/03 §Seams, 05).
    text: Option<PreparedText>,
}

/// The template, thinking close and vision processor of a text model's load.
struct PreparedText {
    provider: crate::artifact_template::ArtifactTemplateProvider,
    /// The vision processor, checked against the model contract at the
    /// configured context; [`load_model`] serves it at the context the load
    /// runs ([`serve_vision`]).
    processor: Option<ignis_artifact::vision::VisionProcessor>,
    /// The vision processor's largest item at the configured context, in
    /// tokens: what the encoder's workspace is sized for (a load that adapts
    /// its context lowers it with the envelope, `EngineShape::at`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    item_bound: Option<u64>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    thinking_close: Option<Arc<ignis_core::thinking_budget::ThinkingClose>>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    eos: Option<TokenId>,
}

impl PreparedModel {
    /// The configuration the load runs with, fitted to its family.
    pub fn config(&self) -> &crate::config::Config {
        &self.config
    }
}

/// A loaded model and what its load reserved (`/metrics`, GitHub #216):
/// `None` where nothing was planned (a build without `cuda`).
pub struct LoadedModel {
    /// The model, its engine running, its driver handle kept.
    pub model: crate::ActiveModel,
    /// What the load's VRAM plan reserved.
    pub reservations: Option<crate::metrics::LoadReservations>,
}

/// Everything the start-up load does before the GPU, for `artifact` under
/// `config` (the verified loader path, server-03): the file must exist, its
/// sidecar be present and its checksum report clean; the artifact names its
/// model, which the start options must fit; the thinking defaults must be
/// ones its template honours, and with `--vision-enabled` its processor must match
/// the model contract. Any of those is a [`LoadError`], never a silent
/// fallback to the placeholder.
pub fn prepare_model(config: &crate::config::Config, artifact: &std::path::Path) -> Result<PreparedModel, LoadError> {
    use crate::template::TemplateProvider;

    // A path that is not there at all gets its own refusal (GitHub #234):
    // the sidecar error below would otherwise name a file next to a file
    // that does not exist.
    if !artifact.exists() {
        return Err(LoadError::ArtifactMissing);
    }
    let sidecar = crate::loader::find_sidecar(artifact).map_err(|e| LoadError::SidecarMissing(e.to_string()))?;
    // GitHub #456: a model whose kernels the card does not run is refused
    // before its checksum is read. An artifact whose header cannot be read
    // is checked as the 27B, and refused below for what it is.
    let named = crate::loader::artifact_family(artifact).ok().flatten();
    // The leaf reads the variable as it does (`kernel/src/flash_next/program.cu`).
    let staged = std::env::var("IGNIS_FN_MOE_STAGED").is_ok_and(|value| value.starts_with('1'));
    let nvfp4_body = || crate::loader::has_nvfp4_body(artifact).unwrap_or(false);
    if let Some(refusal) =
        crate::config::arch::load_refusal(config.card(), named.unwrap_or(ModelFamily::Qwen38_27b), nvfp4_body, staged)
    {
        return Err(LoadError::RouteUnsupported(refusal));
    }
    // GitHub #509: a named dense prefill on the NVFP4 artifact, by the option's
    // spelling, before the leaf refuses it by number. Options the 27B cannot
    // take are refused below, where it is fitted.
    if named.unwrap_or(ModelFamily::Qwen38_27b) == ModelFamily::Qwen38_27b {
        if let Ok(the_27b) = fitted(config, ModelFamily::Qwen38_27b) {
            let spelled = the_27b.basis.resolution().spelled("model", "prefill_linear");
            if let Some(refusal) = crate::config::arch::prefill_linear_refusal(the_27b.prefill_linear, nvfp4_body, &spelled) {
                return Err(LoadError::RouteUnsupported(refusal));
            }
        }
    }
    // GitHub #557 (ADR 0087): before the checksum is read and before anything
    // loads.
    check_vision_companion(config, named, artifact)?;
    // The image model's container holds no text frontend (spec
    // image-model/01): it is verified, and its runner reads the rest.
    if named.map(ModelFamily::kind) == Some(ModelKind::Image) {
        return prepare_image_model(config, artifact, &sidecar);
    }
    let frontend =
        crate::loader::load_artifact(artifact, &sidecar).map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?;
    tracing::info!(
        name: "ignis.artifact.verified",
        artifact = %artifact.display(),
        "checksum clean — tokenizer + chat template loaded"
    );

    // ADR 0043: the artifact names its model, and the start options meet it
    // in one place, which decides the served id or refuses naming the model
    // (spec flash-next/04). An artifact of neither model is served as the
    // 27B always was.
    let family = crate::loader::artifact_family(artifact)
        .map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?
        .unwrap_or(ModelFamily::Qwen38_27b);
    let config = fitted(config, family)?;
    let model = config.model.clone();

    // The thinking budget's forced close (2026-09-24), in this model's own
    // tokens. A tokenizer that splits `</think>` leaves every budget inert:
    // with a default budget configured that is a refusal (spec server/08),
    // since the operator would believe one is active; without one it is said
    // once here rather than discovered per request.
    let thinking_close =
        crate::thinking::thinking_close(|text| frontend.tokenizer().encode(text).map_err(|e| e.to_string()));
    crate::thinking::check_default_budget_close(config.thinking_budget, &thinking_close)
        .map_err(LoadError::ThinkingBudgetInert)?;
    let thinking_close = match thinking_close {
        Ok(close) => Some(Arc::new(close)),
        Err(error) => {
            tracing::warn!(name: "ignis.model.thinking_close_unavailable", %error, "thinking budgets are inert");
            None
        }
    };

    // GitHub #179: a `--vision-enabled` load prepares images with the artifact's
    // processor and acquires them before admission. A tokenizer whose
    // placeholder ids are not the model contract's is refused. Built before
    // the load, which sizes the encoder for the processor's item bound; the
    // load serves it at the context it runs (`serve_vision`, ADR 0063 §3).
    let processor = config
        .vision
        .map(|v| crate::media::load_processor(&frontend, v, config.max_context))
        .transpose()
        .map_err(|e| LoadError::VisionProcessorInvalid(e.to_string()))?;
    // Only the GPU backend needs it; the mock never stops on EOS.
    let eos = frontend.eos_token_id();
    if cfg!(feature = "cuda") && eos.is_none() {
        return Err(LoadError::EosMissing);
    }

    let item_bound = processor.as_ref().map(|p| p.options().max_item_tokens());
    // GitHub #587, #588: a model that serves structured output compiles its
    // grammars within `model.grammar_*`.
    let provider = crate::artifact_template::ArtifactTemplateProvider::serving(frontend, family, config.grammar);

    // A default the template cannot honour is a refusal (a model swap must
    // not silently change behaviour), as a missing EOS token or an unclean
    // checksum is.
    let defaults = crate::thinking::ThinkingDefaults {
        enable_thinking: config.enable_thinking,
        reasoning_effort: config.reasoning_effort,
    };
    crate::thinking::validate_defaults(&defaults, &provider.thinking_capabilities())
        .map_err(LoadError::ThinkingInvalid)?;

    Ok(PreparedModel {
        artifact: artifact.to_path_buf(),
        model,
        family,
        config,
        text: Some(PreparedText { provider, processor, item_bound, thinking_close, eos }),
    })
}

/// GitHub #557 (ADR 0087): Flash-Next sees with its vision companion's tower,
/// so a load of `artifact`, of `family`, under `config` with vision on for
/// Flash-Next is refused, naming the file, when the companion is not beside
/// the artifact or is not its own -- held to it by its pair pin, its identity
/// and the width its merger writes, against Flash-Next's hidden width
/// ([`ignis_core::flash_next_vision::Companion::open`]). Nothing of the
/// tower is read. Any other load asks for no companion. What a start's and
/// a switch's prepare check, and what `POST /v1/models/switch` asks before
/// it begins (GitHub #563).
pub(crate) fn check_vision_companion(
    config: &crate::config::Config,
    family: Option<ModelFamily>,
    artifact: &std::path::Path,
) -> Result<(), LoadError> {
    if family != Some(ModelFamily::FlashNext) || !fitted(config, ModelFamily::FlashNext).is_ok_and(|c| c.vision.is_some()) {
        return Ok(());
    }
    let main = ignis_artifact::Reader::open(artifact).map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?;
    let geometry = ignis_artifact::flash_next::FlashNextGeometry::qwen38_flash_next();
    let hidden = ignis_core::compute::ModelConfig::flash_next_from(&geometry).hidden;
    ignis_core::flash_next_vision::Companion::open(artifact, &main, hidden).map(drop).map_err(LoadError::VisionCompanion)
}

/// The options fitted to `family` (spec config-v2/01): at start, its scoped
/// values applied and anything the family cannot take refused; a switch
/// hands over options it already fitted (`config::fit_to_family`, which
/// drops instead of refusing), and those are used as they are.
fn fitted(config: &crate::config::Config, family: ModelFamily) -> Result<crate::config::Config, LoadError> {
    match config.basis.family() {
        Some(fitted) if fitted == family => Ok(config.clone()),
        _ => config.for_family(family).map_err(|e| LoadError::ModelMismatch(e.to_string())),
    }
}

/// [`prepare_model`] for the image model (spec image-model/05): the checksum
/// against `sidecar` must report clean, and the start options must fit the
/// image family — a speculative backend or the vision tower named for it is
/// refused at start, dropped on a switch, as for any family.
fn prepare_image_model(
    config: &crate::config::Config,
    artifact: &std::path::Path,
    sidecar: &std::path::Path,
) -> Result<PreparedModel, LoadError> {
    crate::loader::verify_artifact(artifact, sidecar).map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?;
    tracing::info!(
        name: "ignis.artifact.verified",
        artifact = %artifact.display(),
        "checksum clean — the image model's runner reads its own frontend"
    );
    let family = ModelFamily::QwenImage21;
    let config = fitted(config, family)?;
    Ok(PreparedModel { artifact: artifact.to_path_buf(), model: config.model.clone(), family, config, text: None })
}

/// Finish a [`PreparedModel`]: build its scheduler — the real GPU-backed one
/// under `cuda` (GitHub #61 / P1-25, its own program and leaf on Flash-Next,
/// GitHub #302), `MockCompute` over the real template otherwise — and start
/// its engine, keeping the model thread's handle so a switch can join it.
///
/// Blocking for the length of a GPU load (tens of seconds); an async caller
/// runs it on `spawn_blocking`. It needs a tokio runtime context: the
/// engine's telemetry consumer is spawned onto it.
pub fn load_model(prepared: PreparedModel) -> Result<LoadedModel, LoadError> {
    let PreparedModel { artifact, model, family, config, text } = prepared;
    let Some(PreparedText { provider, processor, item_bound, thinking_close, eos }) = text else {
        return load_image_model(artifact, model, &config);
    };
    #[cfg(feature = "cuda")]
    let (scheduler, reservations, max_context, adapted, vision) = {
        let eos = eos.ok_or(LoadError::EosMissing)?;
        let (scheduler, reserved, shape, adapted) =
            cuda_scheduler_for(&artifact, &model, family, eos, &config, item_bound, thinking_close)?;
        (scheduler, Some(reserved), shape.max_context, adapted, shape.vision)
    };
    #[cfg(not(feature = "cuda"))]
    let (scheduler, reservations, max_context, adapted, vision) = {
        let _ = (item_bound, thinking_close, eos);
        tracing::warn!(
            name: "ignis.model.mock_compute",
            "built without --features cuda — MockCompute despite --model-artifact/IGNIS_MODEL_ARTIFACT (the templated text is real, the completions are not)"
        );
        (mock_scheduler(&model, config.default_max_tokens), None, config.max_context, String::new(), config.vision)
    };
    let (provider, acquirer) = serve_vision(provider, processor, vision, &config, max_context);
    let (engine, driver) = start_engine(scheduler);
    let mut active = crate::ActiveModel::new(engine, Arc::new(provider))
        .with_driver(driver)
        .with_family(family)
        .with_source(crate::ModelSource { artifact, model })
        .with_adapted(adapted);
    if let Some(acquirer) = acquirer {
        active = active.with_media(acquirer);
    }
    Ok(LoadedModel { model: active, reservations })
}

/// `provider` with the vision processor a `--vision-enabled` load serves,
/// and the acquirer that prepares images with it, both at `max_context`:
/// the context the load runs, which its adaptation may have lowered from
/// the configured one (ADR 0063 §3). So the processor's request budget and
/// item bound are the envelope the leaf planned, never the configured
/// context's; `vision` is the load's, its item bound what the encoder holds
/// (GitHub #557: Flash-Next's, in its prefill scratch, may be below the
/// processor's).
fn serve_vision(
    provider: crate::artifact_template::ArtifactTemplateProvider,
    processor: Option<ignis_artifact::vision::VisionProcessor>,
    vision: Option<ignis_core::Vision>,
    config: &crate::config::Config,
    max_context: u32,
) -> (crate::artifact_template::ArtifactTemplateProvider, Option<Arc<crate::media::MediaAcquirer>>) {
    let Some((processor, vision)) = processor.zip(vision) else {
        return (provider, None);
    };
    let processor = crate::media::processor_at(processor, vision, max_context);
    let acquirer = crate::media::MediaAcquirer::new(
        Arc::new(processor.clone()),
        processor.options().clone(),
        crate::media::MediaPolicy::new(config.media.allow_private_network, config.media.cache_bytes),
    );
    (provider.with_vision(processor), Some(Arc::new(acquirer)))
}

/// The shape a text model's GPU load is handed (`cuda_scheduler_for`): the
/// resolution's ([`EngineShape::from`]), its vision encoder
/// sized for the processor's item bound (`vision_item_bound`), and -- spec
/// 27b-16gb/04 AC 5 -- a drafter's head left to `auto` says what it
/// resolved to (`ignis.runtime.draft_head`, ADR 0063), once per load.
pub fn load_shape(config: &crate::config::Config, vision_item_bound: Option<u64>) -> EngineShape {
    let shape = EngineShape::from(config);
    // The encoder holds one item at a time, and the processor bounds an item
    // below the envelope (`ProcessorOptions::max_item_tokens`): the load
    // sizes the encoder's workspace for that.
    let shape = EngineShape {
        vision: shape.vision.map(|vision| match vision_item_bound {
            // Never above the processor's request budget, the u32 envelope.
            Some(bound) => vision.with_item_max_tokens(u32::try_from(bound).expect("an item bound is at most the envelope")),
            None => vision,
        }),
        ..shape
    };
    if let Some(speculation) = shape.speculation.filter(|_| config.settings().spec.draft_head.is_some_and(|head| head.is_auto())) {
        crate::runtime::log_draft_head_auto(speculation.proposal_head());
    }
    shape
}

/// The image model's load (spec image-model/05): its runner — the image
/// leaf's under `cuda` ([`cuda_image_runner`]), [`ignis_core::image::FakeImageRunner`]
/// otherwise, as a build without `cuda` serves `MockCompute`, its largest
/// target the leaf's ([`crate::images::leaf::max_target_cells`]) — and the
/// model over it, its source kept for a switch back and for the idle unload,
/// with what the leaf's plan reserved (GitHub #324; nothing on the fake). A
/// size of `image.sizes` past the largest target is not served: the load
/// warns of it.
fn load_image_model(
    artifact: std::path::PathBuf,
    model: String,
    config: &crate::config::Config,
) -> Result<LoadedModel, LoadError> {
    #[cfg(feature = "cuda")]
    let (runner, reservations) = cuda_image_runner(&artifact, config)?;
    #[cfg(not(feature = "cuda"))]
    let (runner, reservations): (Box<dyn ImageRunner>, _) = {
        tracing::warn!(
            name: "ignis.model.fake_image_runner",
            "built without --features cuda — the fake image runner despite --model-artifact (its images are deterministic noise, not the prompt's)"
        );
        let image = &config.settings().image;
        let max_target_cells = crate::images::leaf::max_target_cells(&image.sizes, image.max_target_cells);
        let runner = ignis_core::image::FakeImageRunner::new(ignis_core::image::ImagePlan {
            max_target_cells,
            ..ignis_core::image::fake_plan()
        });
        // The fake plans no device memory, as `MockCompute` plans none.
        (Box::new(runner), None)
    };
    tracing::info!(
        name: "ignis.model.loaded",
        artifact = %artifact.display(),
        kind = ModelKind::Image.as_str(),
        condition_budget_tokens = runner.plan().condition_budget_tokens,
        sample_sigmas = runner.plan().sample_sigmas.len(),
        max_target_cells = runner.plan().max_target_cells,
        "image model loaded"
    );
    // A size this load cannot make is dropped from what it serves, and said
    // so; the config keeps it, for a load that can.
    let past: Vec<String> =
        config.settings().image.sizes.iter().filter(|&&size| !runner.plan().takes(size)).map(ToString::to_string).collect();
    if !past.is_empty() {
        tracing::warn!(
            name: "ignis.image.sizes_not_served",
            sizes = %past.join(", "),
            max_target_cells = runner.plan().max_target_cells,
            "image.sizes names sizes past the largest image this load makes (image.max_target_cells): they are not served"
        );
    }
    let image = crate::images::ImageModel::new(model.clone(), runner);
    let active = crate::ActiveModel::image(image).with_source(crate::ModelSource { artifact, model });
    Ok(LoadedModel { model: active, reservations })
}

/// The image leaf's runner for `artifact` (spec image-model/04, GitHub #321):
/// the adapter in the server over the leaf ([`crate::images::leaf::LeafRunner`])
/// — the artifact's tokenizer, templates and Qwen3-VL preprocessing on the
/// host (spec image-model/03 §Seams), the leaf's calls on the device — and
/// what its plan reserved, for `/metrics` (GitHub #324). A load the leaf's
/// plan refuses names the shortfall.
#[cfg(feature = "cuda")]
fn cuda_image_runner(
    artifact: &std::path::Path,
    config: &crate::config::Config,
) -> Result<(Box<dyn ImageRunner>, Option<crate::metrics::LoadReservations>), LoadError> {
    let runner = crate::images::leaf::LeafRunner::load(artifact, config).map_err(|why| {
        LoadError::LoadFailed(format!("{} cannot load {}: {why}", ModelFamily::QwenImage21.name(), artifact.display()))
    })?;
    let reserved = crate::metrics::LoadReservations::image(runner.leaf_plan());
    Ok((Box::new(runner), Some(reserved)))
}

/// The GPU-backed scheduler for `artifact` (GitHub #61 / P1-25), logged as
/// `ignis.model.loaded` with the shape it runs: the 27B's at the values its
/// adaptation kept (ADR 0063 §3), Flash-Next's at the item bound its encoder
/// holds (GitHub #557). Beside it, what the load reserved, that shape and
/// the adaptations it took, as `ignis.runtime.vram_plan` spells them (empty
/// on Flash-Next, which adapts nothing).
#[cfg(feature = "cuda")]
fn cuda_scheduler_for(
    artifact: &std::path::Path,
    model: &str,
    family: ignis_core::compute::ModelFamily,
    eos: TokenId,
    config: &crate::config::Config,
    vision_item_bound: Option<u64>,
    thinking_close: Option<Arc<ignis_core::thinking_budget::ThinkingClose>>,
) -> Result<(Box<dyn ignis_core::Scheduler>, crate::metrics::LoadReservations, EngineShape, String), LoadError> {
    let shape = load_shape(config, vision_item_bound);
    // GitHub #302: a Flash-Next artifact runs its own program and leaf.
    let loaded = match family {
        ModelFamily::FlashNext => crate::runtime::flash_next_load(
            artifact,
            model.into(),
            eos,
            shape,
            thinking_close,
            config.ngram_cache.clone(),
            &config.kv_disk_location,
        )
        .map(|(scheduler, reserved, shape)| (scheduler, reserved, shape, Vec::new()))
        .map_err(|error| LoadError::from_flash_next(error, config)),
        ModelFamily::Qwen38_27b => crate::runtime::qwen27b_load(
            artifact,
            model.into(),
            eos,
            shape.with_family_decode_share(ModelFamily::Qwen38_27b),
            thinking_close,
            &config.kv_disk_location,
        )
        .map_err(LoadError::from),
        // The image model has no token scheduler: `load_image_model` loads
        // it, never this; nor has the diarizer, which no artifact carries.
        ModelFamily::QwenImage21 | ModelFamily::Nemotron3Diarization => Err(LoadError::LoadFailed(format!("{} has no token scheduler", family.name()))),
    };
    // The 27B's context, lanes and head are the ones its plan kept, and so
    // its rope table: what the scheduler admits and `GET /v1/models` says.
    let (scheduler, reserved, shape, adapted) = loaded?;
    // GitHub #489: the solo width as configured and as the load resolved it.
    tracing::info!(
        name: "ignis.runtime.prefill_chunk_solo",
        prefill_chunk_solo = %config.prefill_chunk_solo,
        width = shape.solo_width(),
        prefill_chunk = shape.prefill_chunk,
        "{}",
        shape.solo_text(config.prefill_chunk_solo)
    );
    tracing::info!(
        name: "ignis.model.loaded",
        artifact = %artifact.display(),
        eos,
        prefill_chunk = shape.prefill_chunk,
        prefill_chunk_solo = shape.solo_width(),
        max_context = shape.max_context,
        // GitHub #336: the 27B's are a load setting too.
        decode_lanes = shape.decode_lanes_for(family),
        kv_format = shape.kv_format.as_str(),
        speculation = %speculation_text(shape.speculation),
        vision = %shape.vision.map_or_else(
            || "off".to_owned(),
            |v| format!("max_tokens={} item_max_tokens={}", v.max_tokens(), v.item_max_tokens())
        ),
        // ADR 0057: the table the load runs, beside the setting it came
        // from (`auto`, `none` or `yarn:F...`, as configured).
        rope_scaling = %shape.rope_scaling,
        rope_scaling_setting = %config.settings().model.rope_scaling,
        "model loaded on the GPU"
    );
    Ok((Box::new(scheduler), reserved, shape, crate::adaptation::steps_text(&adapted)))
}

/// A load's speculation as `ignis.model.loaded` spells it: the backend, the
/// draft window and the proposal head, then (GitHub #576, ADR 0090) the
/// draft rule and the drafter's two settings, which a report of what was
/// measured names; `off` without one.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn speculation_text(speculation: Option<ignis_core::Speculation>) -> String {
    speculation.map_or_else(
        || "off".to_owned(),
        |s| {
            let drafter = s.drafter_sampling();
            format!(
                "{} draft_tokens={} draft_head={} draft_rule={} drafter_temperature_factor={} drafter_top_p={}",
                s.backend().as_str(),
                s.draft_tokens(),
                s.proposal_head().as_str(),
                s.draft_rule().as_str(),
                f64::from(drafter.temperature_factor_milli) / 1000.0,
                f64::from(drafter.top_p_milli) / 1000.0
            )
        },
    )
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;
    use ignis_core::image::ImageSize;

    /// A request on `engine`'s model: its id.
    async fn submitted(engine: &crate::engine::Engine) -> ignis_core::RequestId {
        let input = ignis_core::RequestInput {
            decision: None,
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            reuse_boundaries: Vec::new(),
            model: "m".into(),
            tokens: vec![1, 2, 3],
            params: ignis_core::DecodeParams { max_tokens: Some(1), ..Default::default() },
            constrained: None,
            forced_literal: None,
            grammar: None,
            warm_up: false,
        };
        engine.submit(input, ignis_core::RequestClass::Interactive).await.expect("admitted").0
    }

    /// A text model's load prepared over a fixture artifact that holds only
    /// its frontend: on a build without `cuda`, [`load_model`] serves it on
    /// the mock compute, through the path a GPU load takes.
    fn prepared(model: &str) -> (ignis_artifact::fixture::TempArtifact, PreparedModel) {
        use ignis_artifact::fixture::{self, FixtureObject};
        let tokenizer = r#"{"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":{"user":0,"unk":1},"unk_token":"unk"}}"#;
        let resources: [(&'static str, &[u8]); 6] = [
            ("frontend/tokenizer.json", tokenizer.as_bytes()),
            ("frontend/tokenizer_config.json", b"{}"),
            ("frontend/chat_template.jinja", b"{%- for m in messages -%}{{ m.role }} {{ m.content }} {%- endfor -%}"),
            ("frontend/generation_config.json", b"{}"),
            ("frontend/preprocessor_config.json", b"{}"),
            ("frontend/video_preprocessor_config.json", b"{}"),
        ];
        let (mut objects, mut payload) = (Vec::new(), Vec::new());
        for (name, bytes) in resources {
            objects.push(FixtureObject::Resource { name, encoding: "raw-bytes-v1", offset: payload.len() as u64, bytes: bytes.len() as u64 });
            payload.extend_from_slice(bytes);
        }
        let artifact = fixture::write_fixture(&objects, &payload, "load-request-ids").expect("fixture");
        let reader = ignis_artifact::Reader::open(&artifact.path).expect("open fixture artifact");
        let provider = crate::artifact_template::ArtifactTemplateProvider::new(ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set"));
        let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&[], |_| None).expect("the options resolve") else {
            panic!("a config")
        };
        let text = PreparedText { provider, processor: None, item_bound: None, thinking_close: None, eos: None };
        let prepared = PreparedModel { artifact: artifact.path.clone(), model: model.into(), family: ModelFamily::Qwen38_27b, config, text: Some(text) };
        (artifact, prepared)
    }

    /// GitHub #456: on an Ampere card of a build for it, the 27B NVFP4
    /// artifact is refused before its checksum is read, naming its W4A4
    /// GEMMs; on a Blackwell card the same artifact gets past that check, to
    /// be refused for what it lacks.
    #[test]
    fn a_load_the_card_has_no_route_for_is_refused_before_its_checksum() {
        use crate::config::{arch, profile};
        use ignis_artifact::fixture::{self, FixtureObject};
        let body = FixtureObject::Tensor {
            name: "layers.0.mlp.gate_up",
            shape: vec![128, 64],
            format: "NVFP4",
            layout: "blockscale-k16-m128x4-v1",
            offset: 0,
            bytes: 4612,
        };
        let artifact = fixture::write_fixture(&[body], &vec![0u8; 4612], "route-nvfp4").expect("fixture");
        let name = artifact.path.file_name().unwrap().to_str().unwrap().to_owned();
        // Found, and never read on the Ampere card: the refusal comes first.
        let sidecar = artifact.path.with_file_name(format!("{name}.graft.json"));
        std::fs::write(&sidecar, "{}").expect("sidecar");
        let on = |major: u32, minor: u32| {
            let card = profile::Card::Read(profile::CardReading {
                total: 24 << 30,
                name: "a test card".to_owned(),
                capability: arch::ComputeCapability::new(major, minor),
                built_for: arch::BuiltArchitectures::parse("86;120a").unwrap(),
            });
            let args: Vec<String> = ["--profile", "none"].map(String::from).to_vec();
            match crate::config::resolve_on(&args, |_| None, &crate::config::file::NoFiles, &|| card.clone()).expect("resolves") {
                crate::config::ConfigOutcome::Config(config) => config,
                other => panic!("expected a start, got {other:?}"),
            }
        };
        match prepare_model(&on(8, 6), &artifact.path) {
            Err(LoadError::RouteUnsupported(refusal)) => {
                assert!(refusal.contains("W4A4") && refusal.contains("a test card (8.6)"), "{refusal}")
            }
            Err(other) => panic!("expected the route refused, got {other}"),
            Ok(_) => panic!("expected the route refused"),
        }
        match prepare_model(&on(12, 0), &artifact.path) {
            Err(LoadError::RouteUnsupported(refusal)) => panic!("a Blackwell card runs it: {refusal}"),
            Err(_) => {}
            Ok(_) => panic!("the fixture holds no frontend"),
        }
        let _ = std::fs::remove_file(&sidecar);
    }

    /// GitHub #509 (the review of spec #503): on a Blackwell card a named
    /// dense prefill on the 27B's NVFP4 artifact is refused before its
    /// checksum is read, naming the option as the operator spelled it, by
    /// flag or by the environment; unnamed, the same load goes past that
    /// check, to be refused for what the fixture lacks.
    #[test]
    fn a_named_dense_prefill_on_the_nvfp4_artifact_is_refused_naming_the_option() {
        use crate::config::{arch, profile};
        use ignis_artifact::fixture::{self, FixtureObject};
        let body = FixtureObject::Tensor {
            name: "layers.0.mlp.gate_up",
            shape: vec![128, 64],
            format: "NVFP4",
            layout: "blockscale-k16-m128x4-v1",
            offset: 0,
            bytes: 4612,
        };
        let artifact = fixture::write_fixture(&[body], &vec![0u8; 4612], "dense-nvfp4").expect("fixture");
        let name = artifact.path.file_name().unwrap().to_str().unwrap().to_owned();
        let sidecar = artifact.path.with_file_name(format!("{name}.graft.json"));
        std::fs::write(&sidecar, "{}").expect("sidecar");
        let card = profile::Card::Read(profile::CardReading {
            total: 32 << 30,
            name: "a test card".to_owned(),
            capability: arch::ComputeCapability::new(12, 0),
            built_for: arch::BuiltArchitectures::parse("86;120a").unwrap(),
        });
        let on = |flags: &[&str], env: &'static [(&'static str, &'static str)]| {
            let args: Vec<String> = [&["--profile", "none"][..], flags].concat().into_iter().map(String::from).collect();
            let env = move |key: &str| env.iter().find(|(name, _)| *name == key).map(|(_, value)| value.to_string());
            match crate::config::resolve_on(&args, env, &crate::config::file::NoFiles, &|| card.clone()).expect("resolves") {
                crate::config::ConfigOutcome::Config(config) => config,
                other => panic!("expected a start, got {other:?}"),
            }
        };
        let named = [
            (on(&["--model-prefill-linear", "int8"], &[]), "--model-prefill-linear"),
            (on(&[], &[("IGNIS_MODEL_PREFILL_LINEAR", "int8")]), "IGNIS_MODEL_PREFILL_LINEAR"),
        ];
        for (config, spelled) in named {
            match prepare_model(&config, &artifact.path) {
                Err(LoadError::RouteUnsupported(refusal)) => {
                    assert!(refusal.starts_with(&format!("`{spelled} int8` names the dense prefill")), "{refusal}");
                    assert!(refusal.contains("NVFP4"), "{refusal}");
                }
                Err(other) => panic!("expected the dense prefill refused, got {other}"),
                Ok(_) => panic!("expected the dense prefill refused"),
            }
        }
        match prepare_model(&on(&[], &[]), &artifact.path) {
            Err(LoadError::RouteUnsupported(refusal)) => panic!("the fused prefill runs it: {refusal}"),
            Err(_) => {}
            Ok(_) => panic!("the fixture holds no frontend"),
        }
        let _ = std::fs::remove_file(&sidecar);
    }

    /// Every model the process serves text with starts its engine as a
    /// load's does: the placeholder, then two loads ([`load_model`], the
    /// start-up load's path and a switch's or a reload's), and each one's
    /// requests go on from the one before's ids instead of from 0 again.
    #[tokio::test]
    async fn every_loads_request_ids_go_on_from_the_load_before() {
        let placeholder = placeholder_model("m", 16);
        let mut ids = vec![submitted(placeholder.text_engine()).await];
        drop(placeholder);
        for _ in 0..2 {
            let (_artifact, prepared) = prepared("m");
            let loaded = load_model(prepared).expect("the mock loads");
            ids.push(submitted(loaded.model.text_engine()).await);
        }
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "{ids:?}");
    }

    /// Review-321d F2: a size of `image.sizes` past `image.max_target_cells`
    /// neither grows the load's largest target nor is served, and the load
    /// says so (`ignis.image.sizes_not_served`); the fake plans what the
    /// leaf would.
    #[test]
    fn a_size_past_the_cap_is_not_served_and_the_load_warns() {
        let args: Vec<String> = ["--image-sizes", "4096x4096;1024x1024", "--image-max-target-cells", "8000"].map(String::from).to_vec();
        let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&args, |_| None).expect("the options resolve") else {
            panic!("a config")
        };
        use tracing_subscriber::layer::SubscriberExt;
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        let loaded = tracing::subscriber::with_default(subscriber, || {
            load_image_model("image.ninfer".into(), "ignis-image-2.1-turbo".into(), &config).map_err(|e| e.to_string())
        })
        .expect("the fake loads");
        let image = loaded.model.image_model().expect("the image model").clone();
        assert_eq!(image.plan().max_target_cells, 4096, "1024x1024, the largest size within 8,000 cells");
        assert_eq!(crate::images::served_sizes(&config.settings().image.sizes, image.plan()), [ImageSize::new(1024, 1024)]);
        let records: Vec<serde_json::Value> = sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect();
        let warned = records.iter().find(|event| event["event_name"] == "ignis.image.sizes_not_served").expect("the warning");
        let field = |name: &str| warned.get(name).or_else(|| warned["attributes"].get(name)).cloned();
        assert_eq!(field("sizes"), Some(serde_json::json!("4096x4096")), "{warned}");
        assert_eq!(field("max_target_cells"), Some(serde_json::json!(4096)), "{warned}");
    }

    /// GitHub #329 item 12, second review 2026-10-18: a pinned hot set the
    /// load cannot hold is refused naming where its budget was set -- the
    /// profile, the flag -- not always the flag; any other failure reads as
    /// it did.
    #[test]
    fn a_refused_pin_names_where_its_budget_was_set() {
        use crate::runtime::FlashNextLoadError;
        let resolved = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
            let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&args, |_| None).expect("the options resolve") else {
                panic!("a config")
            };
            config
        };
        let refused = ignis_core::residency::PinRefused { asked: 2 << 30, most: 1 << 30, keep: 78, rows: 3, lanes: 3 };
        let message = |config: &crate::config::Config, error: FlashNextLoadError| LoadError::from_flash_next(error, config).to_string();
        let profiled = message(&resolved(&[]), FlashNextLoadError::PinnedExperts(refused.clone()));
        assert!(profiled.starts_with("`vram.pinned_expert_bytes (profile rtx5090 in the built-in profiles)` asks for 2147483648 bytes"), "{profiled}");
        let flagged = message(&resolved(&["--vram-pinned-expert-bytes", "2G"]), FlashNextLoadError::PinnedExperts(refused));
        assert!(flagged.starts_with("`--vram-pinned-expert-bytes` asks for"), "{flagged}");
        assert_eq!(message(&resolved(&[]), FlashNextLoadError::from("no MTP head")), "no MTP head");
    }

    /// GitHub #557 (ADR 0087): a Flash-Next start with vision on is refused
    /// before anything loads, naming the file, when its vision companion is
    /// not beside the artifact or is not its own -- pinned to another main
    /// container, or a merger of another width than Flash-Next's (the
    /// fixture's tower writes 64 rows, which the pin and the identity let
    /// through). With vision off nothing asks for the companion.
    #[test]
    fn a_flash_next_start_with_vision_refuses_a_missing_or_mismatched_companion_by_name() {
        use ignis_artifact::packer::{sidecar_path, VISION_ARTIFACT_FILE_NAME};
        use ignis_core::flash_next_vision::CompanionRefused;
        let config = |vision: bool| {
            let args: Vec<String> = ["--profile", "none"].into_iter().chain(vision.then_some("--vision-enabled")).map(String::from).collect();
            let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&args, |_| None).expect("the options resolve") else {
                panic!("a config")
            };
            config
        };
        let refusal = |vision: bool, artifact: &std::path::Path| prepare_model(&config(vision), artifact).err();
        let main = ignis_artifact::flash_next::fixture::build("load-vision-companion").expect("the main fixture");
        let off = refusal(false, &main.path);
        assert!(!matches!(off, Some(LoadError::VisionCompanion(_))), "vision off asks for no companion: {off:?}");

        let Some(LoadError::VisionCompanion(missing)) = refusal(true, &main.path) else {
            panic!("no companion is refused by it")
        };
        assert!(matches!(missing, CompanionRefused::Missing { .. }), "{missing:?}");
        assert!(missing.to_string().contains(VISION_ARTIFACT_FILE_NAME), "{missing}");

        let companion = main.with_vision_companion().expect("its vision companion");
        let Some(LoadError::VisionCompanion(width)) = refusal(true, &main.path) else { panic!("refused for its width") };
        let width = width.to_string();
        assert!(width.contains(VISION_ARTIFACT_FILE_NAME) && width.contains("vision/merger/fc2") && width.contains("2560"), "{width}");

        let sidecar = sidecar_path(&companion);
        let mut pinned: serde_json::Value = serde_json::from_slice(&std::fs::read(&sidecar).expect("the sidecar")).expect("json");
        pinned["pair"]["main"]["content_hash"] = serde_json::json!("00");
        std::fs::write(&sidecar, pinned.to_string()).expect("pin it elsewhere");
        let Some(LoadError::VisionCompanion(elsewhere)) = refusal(true, &main.path) else { panic!("refused for its pin") };
        let elsewhere = elsewhere.to_string();
        assert!(elsewhere.contains(VISION_ARTIFACT_FILE_NAME) && elsewhere.contains("pair.main.content_hash is 00"), "{elsewhere}");
    }

    /// GitHub #576: the load's log names the draft rule and the drafter's
    /// settings after what it named before, whichever rule runs.
    #[test]
    fn the_load_log_names_the_draft_rule_and_the_drafter_settings() {
        use ignis_core::{DraftRule, DrafterSampling, Speculation, SpeculativeBackend};
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        assert_eq!(
            speculation_text(Some(spec)),
            "dflash2 draft_tokens=7 draft_head=full draft_rule=fixed drafter_temperature_factor=0.9 drafter_top_p=0.95"
        );
        let sampled = spec.with_draft_rule(DraftRule::Sampled, DrafterSampling { temperature_factor_milli: 1000, top_p_milli: 875 });
        assert_eq!(
            speculation_text(Some(sampled)),
            "dflash2 draft_tokens=7 draft_head=full draft_rule=sampled drafter_temperature_factor=1 drafter_top_p=0.875"
        );
        assert_eq!(speculation_text(None), "off");
    }
}
