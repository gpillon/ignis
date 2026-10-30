//! A [`ModelLoader`] over `MockCompute` for the model-switch tests (spec
//! model-switch/01), included via `#[path]` by `model_switch.rs` and
//! `model_switch_http.rs`.
//!
//! Every model it builds — the server's first one included — runs on an
//! engine started with `Engine::with_clock_and_driver`, its driver handle
//! kept on the [`ActiveModel`], so a switch's teardown really joins the
//! model thread. And it keeps a weak handle on every compute it built: a
//! compute is alive exactly as long as the scheduler that owns it, so the
//! count of live ones when a load begins is how many models were still
//! resident then — the mock's proof that the old model was gone first.
//!
//! An id of the image family (`ignis-image-2.1-turbo`, spec image-model/05)
//! builds the image model over a `FakeImageRunner` instead, resident until
//! its runner is dropped, the prompt of every condition prefix it encodes
//! logged — and, for the server's first one, each denoising step behind a
//! [`StepGate`] if the test asks ([`MockLoader::gated_image_model`]).

#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use ignis_core::compute::ModelFamily;
use ignis_core::image::{ConditionId, FakeImageRunner, ImageCondition, ImageError, ImagePlan, ImageRunner, ImageSize, RgbaImage, TargetLatents};
use ignis_core::mock::{GateController, GatedCompute, MockCompute};
use ignis_core::{
    Compute, ConcreteScheduler, EngineMode, FinishReason, Occupancy, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, SubmitError,
};
use ignis_server::config_http::ConfigState;
use ignis_server::engine::Engine;
use ignis_server::images::ImageModel;
use ignis_server::model_switch::{load_options, LoadFailure, ModelLoader, PreparedLoad};
use ignis_server::load::LoadedModel;
use ignis_server::telemetry::SystemClock;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, ModelSource};

/// The loader's shared state: what its prepared loads also reach.
#[derive(Default)]
struct State {
    /// Every compute built, weakly.
    built: Mutex<Vec<Weak<dyn Compute>>>,
    /// Every image runner built, by a handle only the runner keeps alive.
    built_images: Mutex<Vec<Weak<AtomicUsize>>>,
    /// The prompt of every condition prefix an image runner encoded, in
    /// order.
    encoded: Arc<Mutex<Vec<String>>>,
    /// Set, the next denoising step of any image runner panics, once.
    panic_step: Arc<AtomicBool>,
    /// How many built models were still alive as each load began.
    resident_at_load: Mutex<Vec<usize>>,
    /// The ids loaded, in order.
    loads: Mutex<Vec<String>>,
    /// Ids [`ModelLoader::prepare`] refuses.
    refused: Mutex<HashSet<String>>,
    /// Ids whose load fails after prepare accepted them.
    broken: Mutex<HashSet<String>>,
    /// Ids whose load Flash-Next's host plan refuses ([`starved_plan`]).
    starved: Mutex<HashSet<String>>,
    /// Ids whose every request ends in an error ([`MockLoader::fail_requests`]).
    failing: Mutex<HashSet<String>>,
    /// Each load takes the first of these and waits for a message on it
    /// before it builds: how a test holds a switch inside its load step (a
    /// second one holds the reload that follows a failed load).
    hold: Mutex<std::collections::VecDeque<Receiver<()>>>,
    /// The configuration every prepared load reports as the one it runs
    /// with ([`PreparedLoad::options`]), as the production loader does:
    /// what makes a switch re-apply the server-level values once a model
    /// serves.
    serving: Mutex<Option<ignis_server::config::Config>>,
    /// The options each prepare was handed (spec config-v2/02: a reload
    /// with a changed config hands its patched config; a switch none).
    options: Mutex<Vec<Option<ignis_server::config::Config>>>,
    /// The artifact each prepare was asked to load, in order.
    artifacts: Mutex<Vec<std::path::PathBuf>>,
    /// The server's configuration, and the family each id's artifact is
    /// ([`MockLoader::fit_from`]): when set, every prepare fits the options
    /// its load runs with from the configuration running at that moment, by
    /// the production loader's own function.
    fitting: Mutex<Option<(Arc<ConfigState>, Vec<(String, ModelFamily)>)>>,
    /// The options each fitted load runs with, in order.
    fitted: Mutex<Vec<ignis_server::config::Config>>,
    /// What each id's load reports it reserved ([`MockLoader::reserve`]);
    /// an id not here reserved nothing, as on a build without `cuda`.
    reservations: Mutex<std::collections::HashMap<String, ignis_server::metrics::LoadReservations>>,
    /// How many times [`ModelLoader::kind_of`] was asked: what a production
    /// loader answers by opening the artifact.
    kind_of_calls: AtomicUsize,
    /// The family each id's artifact header names ([`MockLoader::header`]);
    /// an id not here is of the family whose own id it is, or of none.
    headers: Mutex<std::collections::HashMap<String, ModelFamily>>,
}

/// The mock loader. Clone the `Arc` into a `Switcher`, keep one to steer it.
#[derive(Default)]
pub struct MockLoader {
    state: Arc<State>,
}

/// Where a mock model `id` "comes from": what a switch to it names — a
/// file that is there (under this process's own temp directory), since a
/// known model whose artifact is not a file is not switchable (spec
/// model-switch/03).
pub fn source(id: &str) -> ModelSource {
    let dir = std::env::temp_dir().join(format!("ignis-mock-artifacts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the mock artifacts' directory");
    let artifact = dir.join(format!("{id}.ninfer"));
    if !artifact.is_file() {
        std::fs::write(&artifact, b"mock").expect("a mock artifact");
    }
    ModelSource { artifact, model: id.to_owned() }
}

impl MockLoader {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A model served as `id`, built as a switch would build it (driver
    /// kept, source recorded) — the server's first model.
    pub fn model(&self, id: &str) -> ActiveModel {
        build(&self.state, id, Arc::new(MockCompute::new()))
    }

    /// [`MockLoader::model`] over a gated compute (spec server/05's
    /// primitive): arm it, then drop the `Arc` unless the test needs to arm
    /// it again — a test holding it keeps the model "resident".
    pub fn gated_model(&self, id: &str) -> (ActiveModel, Arc<GatedCompute>, GateController) {
        let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
        let model = build(&self.state, id, Arc::clone(&gated) as Arc<dyn Compute>);
        (model, gated, controller)
    }

    /// [`MockLoader::model`] for the image model `id`, each denoising step
    /// of its runner waiting at the returned gate, shut.
    pub fn gated_image_model(&self, id: &str) -> (ActiveModel, Arc<StepGate>) {
        let gate = Arc::new(StepGate::default());
        let runner = FakeImageRunner::default();
        self.state.built_images.lock().unwrap().push(Arc::downgrade(&runner.held_conditions()));
        let runner = Logged { inner: runner, encoded: Arc::clone(&self.state.encoded), gate: Some(Arc::clone(&gate)), panic: Arc::clone(&self.state.panic_step) };
        (ActiveModel::image(ImageModel::new(id, Box::new(runner))).with_source(source(id)), gate)
    }

    /// [`MockLoader::model`] over `compute`, a test's own (counted as
    /// resident like every compute this loader builds).
    pub fn model_over(&self, id: &str, compute: Arc<dyn Compute>) -> ActiveModel {
        build(&self.state, id, compute)
    }

    /// Refuse `id` at prepare, as a path with no file would be.
    pub fn refuse(&self, id: &str) {
        self.state.refused.lock().unwrap().insert(id.to_owned());
    }

    /// Fail `id`'s load after prepare accepted it, as a kernel load error
    /// would.
    pub fn break_load(&self, id: &str) {
        self.state.broken.lock().unwrap().insert(id.to_owned());
    }

    /// Fail `id`'s load as Flash-Next's host plan refuses one short of RAM
    /// ([`starved_plan`]), the refusal kept whole as the real loader keeps it.
    pub fn starve(&self, id: &str) {
        self.state.starved.lock().unwrap().insert(id.to_owned());
    }

    /// End every request on a model `id` builds — its warm-up first — with
    /// [`FinishReason::Error`] at its first step, as a leaf failing every
    /// traversal would. The model loads; it never runs anything.
    pub fn fail_requests(&self, id: &str) {
        self.state.failing.lock().unwrap().insert(id.to_owned());
    }

    /// Let `id` load again ([`MockLoader::break_load`] undone): the way a
    /// test proves a later attempt is made, and can succeed.
    pub fn mend(&self, id: &str) {
        self.state.broken.lock().unwrap().remove(id);
    }

    /// Have every load of `id` report `reserved` as what its VRAM plan
    /// reserved, as the production loader does on a build with `cuda`.
    pub fn reserve(&self, id: &str, reserved: ignis_server::metrics::LoadReservations) {
        self.state.reservations.lock().unwrap().insert(id.to_owned(), reserved);
    }

    /// Make the next denoising step of an image runner panic, as a leaf
    /// fault would.
    pub fn panic_next_step(&self) {
        self.state.panic_step.store(true, Ordering::SeqCst);
    }

    /// Hold the next load until the returned sender sends (or is dropped).
    /// Called again before that load runs, it holds the one after.
    pub fn hold_next_load(&self) -> Sender<()> {
        let (release, hold) = channel();
        self.state.hold.lock().unwrap().push_back(hold);
        release
    }

    /// Have every prepared load report `config` as the one it runs with, as
    /// the production loader does.
    pub fn serve_with(&self, config: ignis_server::config::Config) {
        *self.state.serving.lock().unwrap() = Some(config);
    }

    /// Fit every load's options from `config`, the server's running
    /// configuration, as the production loader does
    /// ([`ignis_server::model_switch::load_options`]); an id `families` does
    /// not name is a 27B. What proves a value set for a family not loaded
    /// reaches that family's load, not only a fixed configuration
    /// ([`MockLoader::serve_with`]) handed back.
    pub fn fit_from(&self, config: Arc<ConfigState>, families: &[(&str, ModelFamily)]) {
        let families = families.iter().map(|(id, family)| (id.to_string(), *family)).collect();
        *self.state.fitting.lock().unwrap() = Some((config, families));
    }

    /// Have `id`'s artifact header name `family` ([`ModelLoader::family_of`]),
    /// as the production loader reads it off the file (ADR 0047).
    pub fn header(&self, id: &str, family: ModelFamily) {
        self.state.headers.lock().unwrap().insert(id.to_owned(), family);
    }

    /// The options each fitted load ran with, in order.
    pub fn fitted_options(&self) -> Vec<ignis_server::config::Config> {
        self.state.fitted.lock().unwrap().clone()
    }

    /// How many built models were alive as each load began.
    pub fn resident_at_load(&self) -> Vec<usize> {
        self.state.resident_at_load.lock().unwrap().clone()
    }

    /// How many times the loader was asked a target's kind.
    pub fn kind_of_calls(&self) -> usize {
        self.state.kind_of_calls.load(Ordering::SeqCst)
    }

    /// The ids loaded, in order.
    pub fn loads(&self) -> Vec<String> {
        self.state.loads.lock().unwrap().clone()
    }

    /// The prompt of every condition prefix the image models encoded, in
    /// order: what an image job ran on.
    pub fn encoded_prompts(&self) -> Vec<String> {
        self.state.encoded.lock().unwrap().clone()
    }

    /// How many built models are alive now.
    pub fn resident(&self) -> usize {
        resident(&self.state)
    }

    /// The artifact each prepare was asked to load, in order.
    pub fn prepared_artifacts(&self) -> Vec<std::path::PathBuf> {
        self.state.artifacts.lock().unwrap().clone()
    }

    /// The options each prepare was handed, in order.
    pub fn prepared_options(&self) -> Vec<Option<ignis_server::config::Config>> {
        self.state.options.lock().unwrap().clone()
    }
}

/// The host plan of the switch refused on 2026-10-13: 41.6 GB available,
/// too little for Flash-Next's experts and the 4 GiB margin.
pub fn starved_plan() -> ignis_core::residency::HostPlanError {
    ignis_core::residency::plan_host(&ignis_core::residency::HostPlanRequest {
        available_physical_bytes: 41_632_530_432,
        scope: "qwen38flashnext",
        token_embedding_bytes: 0,
        vision_tower_bytes: 0,
        vision_pool_bytes: 0,
        expert_pool_bytes: 37_795_446_784,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: 1 << 30,
        staging_bytes: 64 << 20,
        retained_host_slots_bytes: 1_040_115_712,
        kv_ram_arena_bytes: 2 << 30,
        margin_bytes: ignis_core::residency::DEFAULT_HOST_MARGIN_BYTES,
    })
    .expect_err("short of RAM")
}

fn resident(state: &State) -> usize {
    let images = state.built_images.lock().unwrap().iter().filter(|built| built.strong_count() > 0).count();
    images + state.built.lock().unwrap().iter().filter(|built| built.strong_count() > 0).count()
}

fn build(state: &State, id: &str, compute: Arc<dyn Compute>) -> ActiveModel {
    if ModelFamily::of_model_id(id) == Some(ModelFamily::QwenImage21) {
        let runner = FakeImageRunner::default();
        state.built_images.lock().unwrap().push(Arc::downgrade(&runner.held_conditions()));
        let runner = Logged { inner: runner, encoded: Arc::clone(&state.encoded), gate: None, panic: Arc::clone(&state.panic_step) };
        return ActiveModel::image(ImageModel::new(id, Box::new(runner))).with_source(source(id));
    }
    state.built.lock().unwrap().push(Arc::downgrade(&compute));
    let scheduler =
        ConcreteScheduler::with_config(SchedulerConfig { model: id.into(), ..SchedulerConfig::default() }, compute);
    let scheduler: Box<dyn Scheduler> = if state.failing.lock().unwrap().contains(id) {
        Box::new(Failing { inner: scheduler, taken: Vec::new() })
    } else {
        Box::new(scheduler)
    };
    let (engine, driver) = Engine::with_clock_and_driver(scheduler, Arc::new(SystemClock));
    ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver).with_source(source(id))
}

/// The mock's scheduler, except that every request it takes ends with
/// [`FinishReason::Error`] at the next step ([`MockLoader::fail_requests`]).
struct Failing {
    inner: ConcreteScheduler,
    taken: Vec<RequestId>,
}

impl Scheduler for Failing {
    fn share_request_ids(&mut self, ids: ignis_core::RequestIds) {
        self.inner.share_request_ids(ids)
    }
    fn submit(&mut self, input: RequestInput, class: RequestClass) -> Result<RequestId, SubmitError> {
        let request = self.inner.submit(input, class)?;
        self.taken.push(request);
        Ok(request)
    }
    fn cancel(&mut self, request: RequestId) -> bool {
        self.taken.retain(|taken| *taken != request);
        self.inner.cancel(request)
    }
    fn advance(&mut self) -> Vec<SchedEvent> {
        std::mem::take(&mut self.taken)
            .into_iter()
            .map(|request| {
                self.inner.cancel(request);
                SchedEvent::Done {
                    request,
                    tokens: 0,
                    reason: FinishReason::Error,
                    spec: None,
                    readout: None,
                    attention: None,
                    drawn: None,
                    thinking: None,
                    reasoning_redirected_at: None,
                }
            })
            .collect()
    }
    fn is_idle(&self) -> bool {
        self.taken.is_empty()
    }
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn max_sequence_tokens(&self) -> u32 {
        self.inner.max_sequence_tokens()
    }
    fn mode(&self) -> EngineMode {
        self.inner.mode()
    }
    fn occupancy(&self) -> Occupancy {
        self.inner.occupancy()
    }
}

impl ModelLoader for MockLoader {
    fn prepare(&self, target: &ModelSource, options: Option<&ignis_server::config::Config>) -> Result<Box<dyn PreparedLoad>, String> {
        self.state.options.lock().unwrap().push(options.cloned());
        self.state.artifacts.lock().unwrap().push(target.artifact.clone());
        if self.state.refused.lock().unwrap().contains(&target.model) {
            return Err(format!("{} refused at prepare (simulated)", target.model));
        }
        let fitted = match &*self.state.fitting.lock().unwrap() {
            Some((config, families)) => {
                let family = families.iter().find(|(id, _)| *id == target.model).map_or(ModelFamily::Qwen38_27b, |(_, family)| *family);
                let fitted = load_options(&config.current(), target, family, options)?;
                self.state.fitted.lock().unwrap().push(fitted.clone());
                Some(fitted)
            }
            None => None,
        };
        let options = fitted.or_else(|| self.state.serving.lock().unwrap().clone());
        Ok(Box::new(MockPrepared { state: Arc::clone(&self.state), id: target.model.clone(), options }))
    }

    /// The default's answer, counted: what a production loader reads off the
    /// artifact's header.
    fn kind_of(&self, target: &ModelSource) -> Option<ignis_core::compute::ModelKind> {
        self.state.kind_of_calls.fetch_add(1, Ordering::SeqCst);
        ModelFamily::of_model_id(&target.model).map(ModelFamily::kind)
    }

    /// The family [`MockLoader::header`] gave the id, else the default's.
    fn family_of(&self, target: &ModelSource) -> Option<ModelFamily> {
        self.state.headers.lock().unwrap().get(&target.model).copied().or_else(|| ModelFamily::of_model_id(&target.model))
    }
}

struct MockPrepared {
    state: Arc<State>,
    id: String,
    options: Option<ignis_server::config::Config>,
}

impl PreparedLoad for MockPrepared {
    fn load(self: Box<Self>) -> Result<LoadedModel, String> {
        let hold = self.state.hold.lock().unwrap().pop_front();
        if let Some(hold) = hold {
            let _ = hold.recv();
        }
        self.state.resident_at_load.lock().unwrap().push(resident(&self.state));
        self.state.loads.lock().unwrap().push(self.id.clone());
        if self.state.broken.lock().unwrap().contains(&self.id) {
            return Err(format!("{} failed to load (simulated kernel load error)", self.id));
        }
        if self.state.starved.lock().unwrap().contains(&self.id) {
            return Err(starved_plan().to_string());
        }
        let reservations = self.state.reservations.lock().unwrap().get(&self.id).cloned();
        Ok(LoadedModel { model: build(&self.state, &self.id, Arc::new(MockCompute::new())), reservations })
    }

    fn load_detailed(self: Box<Self>) -> Result<LoadedModel, LoadFailure> {
        let starved = self.state.starved.lock().unwrap().contains(&self.id);
        self.load().map_err(|reason| if starved { starved_plan().into() } else { reason.into() })
    }

    fn options(&self) -> Option<&ignis_server::config::Config> {
        self.options.as_ref()
    }
}

/// Shut, an image runner's denoising step waits at it; open, every step
/// passes. A step waits 30 s at most, so a failed test leaves no thread
/// behind.
#[derive(Default)]
pub struct StepGate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl StepGate {
    pub fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    fn pass(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut open = self.open.lock().unwrap();
        while !*open && Instant::now() < deadline {
            open = self.changed.wait_timeout(open, Duration::from_millis(50)).unwrap().0;
        }
    }
}

/// `FakeImageRunner`, logging the prompt of each prefix it encodes, each
/// denoising step behind `gate` when there is one, and panicking at one when
/// `panic` is set.
struct Logged {
    inner: FakeImageRunner,
    encoded: Arc<Mutex<Vec<String>>>,
    gate: Option<Arc<StepGate>>,
    panic: Arc<AtomicBool>,
}

impl ImageRunner for Logged {
    fn plan(&self) -> &ImagePlan {
        self.inner.plan()
    }

    fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ignis_core::image::ConditionUnfit> {
        self.inner.condition_tokens(prompt, images)
    }

    fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError> {
        self.encoded.lock().unwrap().push(condition.prompt.clone());
        self.inner.encode_condition(condition)
    }

    fn denoise_step(
        &mut self,
        condition: ConditionId,
        latents: &mut TargetLatents,
        sigma: f32,
        sigma_next: f32,
        velocity: Option<&mut Vec<f32>>,
    ) -> Result<(), ImageError> {
        if let Some(gate) = &self.gate {
            gate.pass();
        }
        if self.panic.swap(false, Ordering::SeqCst) {
            panic!("a leaf fault (simulated)");
        }
        self.inner.denoise_step(condition, latents, sigma, sigma_next, velocity)
    }

    fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
        self.inner.vae_decode(latents)
    }

    fn release_condition(&mut self, condition: ConditionId) {
        self.inner.release_condition(condition);
    }

    fn condition_bytes(&self) -> u64 {
        self.inner.condition_bytes()
    }
}
