//! The speech model in the server (spec speech/02, ADR 0067): Qwen3-ASR,
//! loaded beside the active model, on the CPU, at first use.
//!
//! It is a slot of its own on [`crate::Server`], not a [`crate::Runner`]: it
//! is never switched to, a switch does not tear it down, and the text
//! model's idle clock never sees it. Its state is computed where it is read,
//! from the live `speech.*` settings, the catalog and what is on disk:
//!
//! - **off**: `speech.model` is `off`;
//! - **unavailable, with a reason**: the catalog has no such entry, the host
//!   cannot run the executor (no AVX2), or its files are not under
//!   `download.path` (the reason names the command that fetches them);
//! - **available**: it can serve, whether or not it is loaded right now. The
//!   first request loads it (Q53), and the worker unloads it when idle.
//!
//! The executor is behind [`executor::SpeechExecutor`]; the route
//! ([`route`]), this slot and the worker ([`worker`]) are tested against the
//! scripted `executor::FakeExecutor` (test builds only).

pub mod audio;
pub mod cpu;
pub mod diarizing;
pub mod executor;
pub mod languages;
pub mod realtime;
pub mod route;
pub mod worker;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::config::schema::{DiarizerGroup, SpeechGroup};
use crate::download::Catalog;
use executor::{DiarizerExecutor, LoadSettings, NoDiarizer, NoExecutor, SessionOpen, SpeechExecutor, SpeechFiles, SpeechSession};
use worker::{DiarizerWorker, Full, Gauges, Job, Worker};

/// The `Retry-After` of a refused Realtime session: room frees when a person
/// ends a dictation, which no clip length estimates, so the shortest wait.
/// Not an operator threshold (spec speech/02 As built).
pub const SESSION_RETRY_AFTER_SECS: u64 = 1;

/// The catalog and the directory the speech model's files are looked for in:
/// `download.path`.
#[derive(Debug, Clone)]
pub struct SpeechModels {
    pub catalog: Arc<Catalog>,
    pub dir: PathBuf,
}

/// What `GET /v1/models` and the route say of the speech model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Availability {
    /// The id it is served under, when `speech.model` names one.
    pub id: Option<String>,
    /// Why it cannot serve; `None` when it can.
    pub reason: Option<String>,
}

impl Availability {
    pub fn available(&self) -> bool {
        self.reason.is_none()
    }
}

/// What `GET /v1/models` says of the diarizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiarizerAvailability {
    /// The id `diarizer.model` names, when it names one.
    pub id: Option<String>,
    /// Whether the weights are in memory right now.
    pub loaded: bool,
    /// Why it cannot serve; `None` when it can.
    pub reason: Option<String>,
}

/// The speech model's slot.
pub struct Speech {
    executor: Arc<dyn SpeechExecutor>,
    /// The diarizer beside it (spec speech/08), its units on a thread of its
    /// own.
    diarizer: Arc<dyn DiarizerExecutor>,
    models: Option<SpeechModels>,
    gauges: Arc<Gauges>,
    /// Started by the first request: a server that is never asked for a
    /// transcription never has the thread.
    worker: OnceLock<Worker>,
    /// The diarizer's thread, started by the first session that asks for
    /// speakers: a server never asked for them never has it.
    diarizer_worker: OnceLock<DiarizerWorker>,
    /// The next Realtime session's key: its lane on each worker.
    next_session: AtomicU64,
    /// `speech.idle_unload` in seconds as the live configuration last said:
    /// the worker's idle clock reads it ([`Speech::live_changed`]).
    idle_unload_secs: Arc<AtomicU64>,
}

impl Default for Speech {
    /// A server built without a speech model: no executor, no catalog.
    fn default() -> Self {
        Self::new(Arc::new(NoExecutor), None)
    }
}

impl Speech {
    pub fn new(executor: Arc<dyn SpeechExecutor>, models: Option<SpeechModels>) -> Self {
        Self {
            executor,
            diarizer: Arc::new(NoDiarizer),
            models,
            gauges: Arc::default(),
            worker: OnceLock::new(),
            diarizer_worker: OnceLock::new(),
            next_session: AtomicU64::new(0),
            idle_unload_secs: Arc::new(AtomicU64::new(u64::from(SpeechGroup::default().idle_unload))),
        }
    }

    /// The live `speech.*` settings changed (the server's start, a `PATCH
    /// /v1/config`, a reload): the idle clock takes `speech.idle_unload` at
    /// once, the model in use or idle.
    pub fn live_changed(&self, group: &SpeechGroup) {
        self.idle_unload_secs.store(u64::from(group.idle_unload), Ordering::SeqCst);
        if let Some(worker) = self.worker.get() {
            worker.wake();
        }
    }

    pub fn executor(&self) -> &Arc<dyn SpeechExecutor> {
        &self.executor
    }

    /// Give speakers with `diarizer` (spec speech/08). A slot built without
    /// one has a diarizer that is unavailable, saying so.
    pub fn with_diarizer(mut self, diarizer: Arc<dyn DiarizerExecutor>) -> Self {
        self.diarizer = diarizer;
        self
    }

    pub fn diarizer(&self) -> &Arc<dyn DiarizerExecutor> {
        &self.diarizer
    }

    /// The live `diarizer.*` settings changed (the server's start, a `PATCH
    /// /v1/config`, a reload): the diarizer takes `diarizer.model` for the
    /// next stream it opens.
    pub fn diarizer_changed(&self, group: &DiarizerGroup) {
        self.diarizer.follow(group);
    }

    /// What `GET /v1/models` says of the diarizer: its id, whether its
    /// weights are in memory, and why it cannot serve (spec speech/08, Q118).
    pub fn diarizer_availability(&self) -> DiarizerAvailability {
        DiarizerAvailability { id: self.diarizer.id(), loaded: self.diarizer.is_loaded(), reason: self.diarizer.availability().err() }
    }

    /// The catalog the speech model is looked up in, when the server has one.
    pub fn catalog(&self) -> Option<&Catalog> {
        self.models.as_ref().map(|models| &*models.catalog)
    }

    /// The files of the speech model the worker has loaded now, `None` while
    /// none is: those it was loaded from, whatever `speech.model` names since.
    pub fn loaded(&self) -> Option<SpeechFiles> {
        self.executor.loaded_files()
    }

    /// The GGUF the diarizer has loaded now, `None` while none is: unloaded
    /// with the speech model, whatever `diarizer.model` names since.
    pub fn diarizer_loaded(&self) -> Option<PathBuf> {
        self.diarizer.loaded_weights()
    }

    /// Publish the worker's gauges to `metrics` (`Server::with_metrics`).
    pub fn attach_metrics(&self, metrics: Arc<crate::metrics::Metrics>) {
        self.gauges.attach(metrics);
    }

    fn worker(&self) -> &Worker {
        self.worker.get_or_init(|| Worker::start_with(Arc::clone(&self.executor), Arc::clone(&self.diarizer), Arc::clone(&self.gauges), Arc::clone(&self.idle_unload_secs)))
    }

    /// The id `speech.model` names: `auto` is the 1.7B Q4_K_M entry, `off`
    /// is none.
    pub fn served_id(group: &SpeechGroup) -> Option<String> {
        match group.model.as_str() {
            "off" => None,
            "auto" => Some(crate::config::AUTO_SPEECH_MODEL_ID.to_owned()),
            id => Some(id.to_owned()),
        }
    }

    /// Whether the speech model can serve now, and if not why, in the words
    /// the Playground shows (spec speech/02 §Unavailable, Q35, Q39).
    pub fn availability(&self, group: &SpeechGroup) -> Availability {
        match self.resolve(group) {
            Ok(files) => Availability { id: Some(files.id), reason: None },
            Err(unavailable) => Availability { id: Self::served_id(group), reason: Some(unavailable.0) },
        }
    }

    /// The files to serve with, or why there is no model to serve: checked in
    /// the order a person fixes them — off, an id the catalog has as a speech
    /// model, a host that can run it, the files on disk.
    pub fn resolve(&self, group: &SpeechGroup) -> Result<SpeechFiles, Unavailable> {
        let Some(id) = Self::served_id(group) else {
            return Err(Unavailable("the speech model is turned off (`speech.model` is `off`)".to_owned()));
        };
        let entry = self.models.as_ref().and_then(|models| models.catalog.entries().iter().find(|entry| entry.id == id).map(|entry| (models, entry)));
        let Some((models, entry)) = entry else {
            return Err(Unavailable(format!("`speech.model` names `{id}`, which is not in the model catalog")));
        };
        if !entry.is_speech() {
            return Err(Unavailable(format!("`speech.model` names `{id}`, which is not a speech model: name a speech entry of the catalog")));
        }
        self.executor.availability().map_err(Unavailable)?;
        let files: Vec<PathBuf> = entry.files.iter().map(|file| models.dir.join(&file.name)).collect();
        if !files.iter().all(|file| file.is_file()) {
            return Err(Unavailable(format!(
                "no speech model files for `{id}` under download.path; fetch them with `ignis-server model download {id}`"
            )));
        }
        Ok(SpeechFiles { id, decoder: entry.artifact_path(&models.dir), files })
    }

    /// Whether a batch job would be queued now, or when room frees: the look
    /// a route takes before it decodes a clip, so a full queue costs a
    /// refusal and not a decode. [`Speech::submit`] checks again.
    pub fn admit(&self, group: &SpeechGroup) -> Result<(), Full> {
        self.worker().room(group.max_queue)
    }

    /// Queue a batch job, or say when room frees. Starts the worker.
    pub fn submit(&self, job: Box<dyn Job>, group: &SpeechGroup) -> Result<(), Full> {
        self.worker().submit(job, group.max_queue)
    }

    /// Whether a Realtime session would open now, or when room frees: the
    /// look a plain `GET /v1/realtime` takes without claiming a place.
    /// [`Speech::open_session`] checks again.
    pub fn session_room(&self, group: &SpeechGroup) -> Result<(), Full> {
        if self.gauges.sessions.load(Ordering::SeqCst) >= group.max_sessions as usize {
            return Err(Full { retry_after_secs: SESSION_RETRY_AFTER_SECS });
        }
        Ok(())
    }

    /// Open a Realtime session, unless `max_sessions` are open (spec
    /// speech/02 §Admission, 03). The guard closes it when dropped.
    pub fn open_session(self: &Arc<Self>, group: &SpeechGroup) -> Result<SessionGuard, Full> {
        let mut open = self.gauges.sessions.load(Ordering::SeqCst);
        loop {
            if open >= group.max_sessions as usize {
                return Err(Full { retry_after_secs: SESSION_RETRY_AFTER_SECS });
            }
            match self.gauges.sessions.compare_exchange(open, open + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(now) => open = now,
            }
        }
        self.gauges.publish();
        Ok(SessionGuard { speech: Arc::clone(self), key: self.next_session.fetch_add(1, Ordering::SeqCst) })
    }

    /// The Realtime sessions open now.
    pub fn sessions_open(&self) -> usize {
        self.gauges.sessions.load(Ordering::SeqCst)
    }

    /// The batch requests waiting behind the one running.
    pub fn queue_length(&self) -> usize {
        self.gauges.queue.load(Ordering::SeqCst)
    }

    /// Queue a unit of an open Realtime session's (spec speech/03): after
    /// the ones it queued before, ahead of every batch window, in turn with
    /// the other sessions'. Starts the worker.
    pub fn submit_interactive(&self, session: &SessionGuard, task: worker::Task) {
        self.worker().submit_interactive(session.key, task);
    }

    /// Queue a unit of an open Realtime session's speakers (spec speech/08)
    /// on the diarizer's own thread, in turn with the other sessions': it
    /// never waits for a unit of the speech model's, nor holds one back.
    /// Starts the thread.
    pub fn submit_diarizer(&self, session: &SessionGuard, task: worker::Task) {
        self.diarizer_worker.get_or_init(DiarizerWorker::start).submit(session.key, task);
    }
}

/// `speech.model` and `speech.compat_names` against `catalog`, the operator's
/// file among it (spec speech/02 §Which model): a model naming an entry of
/// another kind is refused, since it would be listed and served as speech,
/// and so is a compat name equal to an entry's id or alias, since a request
/// naming it would be two models' at once. Checked at start, where the
/// catalog is read, and by a `PATCH /v1/config` naming either field; the
/// config's own check covers the built-in entries and the model families.
pub fn check_names(catalog: &Catalog, group: &SpeechGroup) -> Result<(), String> {
    if let Some(id) = Speech::served_id(group) {
        if let Some(entry) = catalog.entries().iter().find(|entry| entry.id == id).filter(|entry| !entry.is_speech()) {
            return Err(format!("`speech.model` names `{}`, which is not a speech model: name a speech entry of the catalog, `auto` or `off`", entry.id));
        }
    }
    for name in &group.compat_names {
        if catalog.entries().iter().any(|entry| entry.id.eq_ignore_ascii_case(name) || entry.aliases.iter().any(|alias| alias.eq_ignore_ascii_case(name))) {
            return Err(format!("`speech.compat_names` names `{name}`, which is a catalog model's id: a compat name is for a name no served model has"));
        }
    }
    Ok(())
}

/// `diarizer.model` against `catalog`, the operator's file among it (spec
/// speech/08, Q118): an id naming an entry that is not a diarizer is refused,
/// as `check_names` refuses a wrong-kind `speech.model`. Checked at start,
/// where the catalog is read, and by a `PATCH /v1/config` naming the field;
/// the config's own check covers the built-in entries and the model families.
pub fn check_diarizer_name(catalog: &Catalog, group: &DiarizerGroup) -> Result<(), String> {
    if let Some(id) = diarizing::served_id(catalog, &group.model) {
        if let Some(entry) = catalog.entries().iter().find(|entry| entry.id == id).filter(|entry| !entry.is_diarizer()) {
            return Err(format!("`diarizer.model` names `{}`, which is not a diarizer: name a diarizer entry of the catalog, `auto` or `off`", entry.id));
        }
    }
    Ok(())
}

/// What `text` adds to `sent`, when it extends it: the only text a delta may
/// carry, since a delta only ever appends (a streamed transcription's and a
/// Realtime item's alike).
pub(crate) fn extension<'a>(sent: &str, text: &'a str) -> Option<&'a str> {
    (text.len() > sent.len() && text.starts_with(sent)).then(|| &text[sent.len()..])
}

/// Why a transcription stopped short: the model would not load, or a call to
/// it failed. The executor's own words are the log's (`reason`), never a
/// client's but for a load's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Fault {
    Load(String),
    Executor(String),
}

impl Fault {
    /// The error code and message every surface answers it with: the route's
    /// body and its `error` event, and a Realtime item's `failed`. `what`
    /// names what failed.
    pub(crate) fn error(&self, what: &str) -> (&'static str, String) {
        match self {
            Fault::Load(why) => ("speech_unavailable", format!("the speech model could not be loaded: {why}")),
            Fault::Executor(_) => engine_failed(what),
        }
    }

    /// The executor's own words, for the transcription's log line.
    pub(crate) fn reason(&self) -> &str {
        match self {
            Fault::Load(why) | Fault::Executor(why) => why,
        }
    }
}

/// The code and message of a transcription the speech model failed, or
/// whose job the worker dropped.
pub(crate) fn engine_failed(what: &str) -> (&'static str, String) {
    ("engine_error", format!("the speech model failed {what}"))
}

/// Load the model if it is not, then open a session on it: a batch job's
/// first step, and a Realtime item's opening.
pub(crate) fn open_loaded(executor: &dyn SpeechExecutor, files: &SpeechFiles, load: LoadSettings, open: SessionOpen) -> Result<Box<dyn SpeechSession>, Fault> {
    if !executor.is_loaded() {
        executor.load(files, load).map_err(|error| Fault::Load(error.0))?;
    }
    executor.open(open).map_err(|error| Fault::Executor(error.0))
}

/// An open Realtime session's place among `speech.max_sessions`, and its
/// lane on each worker.
pub struct SessionGuard {
    speech: Arc<Speech>,
    key: u64,
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.speech.gauges.sessions.fetch_sub(1, Ordering::SeqCst);
        self.speech.gauges.publish();
        // The session's end is the model's last use: the idle clock runs from
        // here (an open session holds the model, `worker`).
        if let Some(worker) = self.speech.worker.get() {
            worker.touch();
        }
    }
}

/// Why the speech model cannot serve, as the route and `GET /v1/models` say it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unavailable(pub String);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::catalog::{CatalogEntry, CatalogFile, CatalogLayer};

    /// A speech catalog entry of our own (the real ones are ticket #351's):
    /// the decoder GGUF and the encoder beside it.
    pub(crate) fn fixture_models(dir: &std::path::Path) -> SpeechModels {
        let file = |name: &str| CatalogFile { name: name.to_owned(), bytes: 1, sha256: "0".repeat(64) };
        let entry = |id: &str| CatalogEntry {
            id: id.to_owned(),
            repo: "test/asr".to_owned(),
            revision: "0".repeat(40),
            artifact: format!("{id}-decoder.gguf"),
            card_class: None,
            family: None,
            aliases: Vec::new(),
            kind: Some(ignis_core::compute::ModelKind::Speech),
            files: vec![file(&format!("{id}-encoder.gguf")), file(&format!("{id}-decoder.gguf"))],
            layer: CatalogLayer::BuiltIn,
        };
        SpeechModels { catalog: Arc::new(Catalog::from_entries(vec![entry("qwen3-asr-1.7b"), entry("qwen3-asr-0.6b")])), dir: dir.to_owned() }
    }

    pub(crate) fn put_files(dir: &std::path::Path, id: &str) {
        for name in [format!("{id}-encoder.gguf"), format!("{id}-decoder.gguf")] {
            std::fs::write(dir.join(name), b"gguf").unwrap();
        }
    }

    fn group(model: &str) -> SpeechGroup {
        SpeechGroup { model: model.to_owned(), ..SpeechGroup::default() }
    }

    fn slot(executor: Arc<dyn SpeechExecutor>, dir: &std::path::Path) -> Speech {
        Speech::new(executor, Some(fixture_models(dir)))
    }

    /// AC 1: `auto` is the 1.7B Q4_K_M entry, an id is its entry, `off` none.
    #[test]
    fn speech_model_resolves_auto_an_id_and_off() {
        assert_eq!(Speech::served_id(&group("auto")).as_deref(), Some("qwen3-asr-1.7b"));
        assert_eq!(Speech::served_id(&group("qwen3-asr-0.6b")).as_deref(), Some("qwen3-asr-0.6b"));
        assert_eq!(Speech::served_id(&group("off")), None);
    }

    /// AC 1, 2: the model is available when its files are on disk, with the
    /// files the executor loads, and nothing is fetched by asking.
    #[test]
    fn the_model_is_available_when_its_files_are_on_disk() {
        let dir = tempfile_dir("available");
        let speech = slot(Arc::new(executor::FakeExecutor::saying(&["hi"])), &dir);
        let missing = speech.availability(&group("auto"));
        assert!(!missing.available());
        put_files(&dir, "qwen3-asr-1.7b");
        let files = speech.resolve(&group("auto")).unwrap();
        assert_eq!(files.id, "qwen3-asr-1.7b");
        assert_eq!(files.decoder, dir.join("qwen3-asr-1.7b-decoder.gguf"));
        assert_eq!(files.files.len(), 2);
        assert_eq!(speech.availability(&group("auto")), Availability { id: Some("qwen3-asr-1.7b".to_owned()), reason: None });
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "asking downloaded nothing");
    }

    /// AC 2: each unavailable state reports its reason, in order of what a
    /// person fixes: off, an id the catalog lacks, a host that cannot run it,
    /// files missing, naming the command that fetches them.
    #[test]
    fn each_unavailable_state_reports_its_reason() {
        let dir = tempfile_dir("reasons");
        let fake = Arc::new(executor::FakeExecutor::saying(&["hi"]));
        let speech = slot(fake.clone(), &dir);
        let reason = |model: &str| speech.availability(&group(model)).reason.expect("unavailable");
        assert!(reason("off").contains("turned off") && reason("off").contains("speech.model"));
        assert!(reason("qwen3-asr-9b").contains("`qwen3-asr-9b`") && reason("qwen3-asr-9b").contains("not in the model catalog"));
        let files = reason("auto");
        assert!(files.contains("no speech model files") && files.contains("model download qwen3-asr-1.7b"), "{files}");
        put_files(&dir, "qwen3-asr-1.7b");
        fake.set_unavailable(Some("the CPU lacks AVX2, which the speech model needs"));
        assert_eq!(reason("auto"), "the CPU lacks AVX2, which the speech model needs");
        // The host's reason comes before the files': downloading cannot fix it.
        std::fs::remove_file(dir.join("qwen3-asr-1.7b-decoder.gguf")).unwrap();
        assert!(reason("auto").contains("AVX2"));
        assert!(Speech::default().availability(&group("auto")).reason.unwrap().contains("not in the model catalog"));
    }

    /// Spec speech/02 §Which model: `speech.model` naming a catalog entry that
    /// is not a speech model is unavailable, saying so, files on disk or not,
    /// so it is neither listed as speech nor served by the speech routes.
    #[test]
    fn a_model_of_another_kind_is_not_the_speech_model() {
        let dir = tempfile_dir("kind");
        let mut models = fixture_models(&dir);
        let mut entries = models.catalog.entries().to_vec();
        let mut text = entries[0].clone();
        (text.id, text.kind, text.artifact) = ("qwen3.8-27b".to_owned(), Some(ignis_core::compute::ModelKind::Text), "qwen3.8-27b-decoder.gguf".to_owned());
        text.files.iter_mut().zip(["qwen3.8-27b-encoder.gguf", "qwen3.8-27b-decoder.gguf"]).for_each(|(file, name)| file.name = name.to_owned());
        entries.push(text);
        models.catalog = Arc::new(Catalog::from_entries(entries));
        put_files(&dir, "qwen3.8-27b");
        let speech = Speech::new(Arc::new(executor::FakeExecutor::saying(&["hi"])), Some(models.clone()));
        let reason = speech.availability(&group("qwen3.8-27b")).reason.expect("unavailable");
        assert!(reason.contains("`qwen3.8-27b`") && reason.contains("not a speech model"), "{reason}");
        let refused = check_names(&models.catalog, &group("qwen3.8-27b")).unwrap_err();
        assert!(refused.contains("speech.model") && refused.contains("not a speech model"), "{refused}");
        assert!(check_names(&models.catalog, &group("qwen3-asr-0.6b")).is_ok());
        assert!(check_names(&models.catalog, &group("auto")).is_ok());
    }

    /// Spec speech/08 (AC 14): `diarizer.model` naming a catalog entry that is
    /// not a diarizer is refused, the operator's entries among them.
    #[test]
    fn a_diarizer_model_of_another_kind_is_refused_against_the_catalog() {
        let models = fixture_models(&tempfile_dir("diarizer-kind"));
        let named = |model: &str| DiarizerGroup { model: model.to_owned(), ..DiarizerGroup::default() };
        for id in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"] {
            let refused = check_diarizer_name(&models.catalog, &named(id)).unwrap_err();
            assert!(refused.contains("diarizer.model") && refused.contains(id) && refused.contains("not a diarizer"), "{refused}");
        }
        for id in ["auto", "off", "not-in-the-catalog"] {
            assert!(check_diarizer_name(&models.catalog, &named(id)).is_ok(), "{id}");
        }
    }

    /// The compat names against the catalog the server runs with, an
    /// operator's entries among it: one equal to an entry's id is refused.
    #[test]
    fn a_compat_name_equal_to_a_catalog_id_is_refused() {
        let models = fixture_models(&tempfile_dir("compat"));
        let named = |names: &[&str]| SpeechGroup { compat_names: names.iter().map(|name| (*name).to_owned()).collect(), ..SpeechGroup::default() };
        assert!(check_names(&models.catalog, &named(&["whisper-1"])).is_ok());
        let refused = check_names(&models.catalog, &named(&["whisper-1", "QWEN3-ASR-0.6B"])).unwrap_err();
        assert!(refused.contains("speech.compat_names") && refused.contains("QWEN3-ASR-0.6B"), "{refused}");
    }

    /// `speech.idle_unload` changed live reaches the worker's idle clock
    /// before anything is submitted again.
    #[test]
    fn a_live_idle_unload_reaches_the_idle_clock() {
        let fake = Arc::new(executor::FakeExecutor::saying(&["hi"]));
        let speech = Speech::new(fake.clone(), None);
        speech.live_changed(&SpeechGroup { idle_unload: 0, ..SpeechGroup::default() });
        // The worker starts with the first request; the model is loaded by it.
        assert!(speech.admit(&SpeechGroup::default()).is_ok());
        fake.loaded.store(true, Ordering::SeqCst);
        speech.worker().touch();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(fake.is_loaded(), "0 never unloads");
        speech.live_changed(&SpeechGroup { idle_unload: 1, ..SpeechGroup::default() });
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while fake.is_loaded() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!fake.is_loaded(), "a live 1 s did not unload the idle model");
    }

    /// AC 11: past `max_sessions` open sessions the next is refused with a
    /// bounded `Retry-After`; a closed one gives its place back.
    #[test]
    fn past_max_sessions_a_session_is_refused_and_a_closed_one_frees_its_place() {
        let speech = Arc::new(Speech::default());
        let group = SpeechGroup { max_sessions: 2, ..SpeechGroup::default() };
        let first = speech.open_session(&group).unwrap();
        let _second = speech.open_session(&group).unwrap();
        assert_eq!(speech.sessions_open(), 2);
        let full = speech.open_session(&group).err().expect("a third is refused");
        assert!((1..=worker::RETRY_AFTER_MAX_SECS).contains(&full.retry_after_secs));
        drop(first);
        assert_eq!(speech.sessions_open(), 1);
        assert!(speech.open_session(&group).is_ok());
    }

    /// A unique empty directory under the system temp dir.
    pub(crate) fn tempfile_dir(name: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!("ignis-speech-{name}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
