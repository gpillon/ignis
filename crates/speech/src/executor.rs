//! The executor the server drives (spec speech/01 §3-4): availability with
//! its reason, the model loaded at first use and unloaded when asked, the
//! thread count and the prefix cache.
//!
//! The server owns the `speech` config group and the routes (spec 02); the
//! executor takes its settings as plain values. The server's one compute
//! worker (spec 02 §Admission) decides when the model is idle, and runs one
//! unit of work at a time: each call a session takes is a unit, and a commit
//! is one per [`Commit::step`](crate::session::Commit::step), a window, the
//! decode's prefill or 16 decode steps.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::cpu::{select, Features, Isa};
use crate::decoder::Cache;
use crate::mel::Mel;
use crate::model::{LoadError, LoadReport, Model};
use crate::pool::Pool;
use crate::session::Session;
use crate::text::ContextCopy;

/// `speech.threads`: `auto` = physical cores − 1 (Q38), or a count used as given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    Auto,
    Count(usize),
}

impl Threads {
    /// The workers to run, and whether they were resolved from `auto`. `auto` counts the
    /// physical cores within the CPUs this process may run on: on Linux the physical count
    /// reads `/proc/cpuinfo`, which a container's cpuset or quota does not narrow, so it is
    /// capped by the CPUs the affinity mask and the cgroup quota allow.
    pub fn resolve(self) -> (usize, &'static str) {
        match self {
            Threads::Auto => (physical_cores().saturating_sub(1).max(1), "auto"),
            Threads::Count(n) => (n.max(1), "configured"),
        }
    }

    /// `diarizer.threads` (spec speech/08): the diarizer's own pool, beside
    /// the speech model's. `auto` is [`Threads::diarizer_auto`] of the
    /// physical cores, counted as [`Threads::resolve`] counts them; a count
    /// is used as given.
    pub fn resolve_diarizer(self) -> (usize, &'static str) {
        match self {
            Threads::Auto => (Threads::diarizer_auto(physical_cores()), "auto"),
            Threads::Count(n) => (n.max(1), "configured"),
        }
    }

    /// The diarizer's `auto` on `physical` cores: a third of them, at least
    /// 2 (3 on the i9's 10 cores, 2 on 4).
    pub fn diarizer_auto(physical: usize) -> usize {
        (physical / 3).max(2)
    }
}

/// The physical cores within the CPUs this process may run on.
fn physical_cores() -> usize {
    num_cpus::get_physical().min(num_cpus::get())
}

/// The executor's settings: the `speech` fields it reads, as plain values.
#[derive(Clone, Debug)]
pub struct Settings {
    pub threads: Threads,
    /// `speech.pinned_weights`: a private, locked copy instead of the mapped file (Q31).
    pub pinned_weights: bool,
    /// `speech.partial_interval_ms` (Q37).
    pub partial_interval_ms: u64,
    /// `speech.rollback_tokens`: the provisional tail's length (Q37).
    pub rollback_tokens: usize,
    /// The first partials fix no token: the official streaming's `unfixed_chunk_num` (2), so
    /// the language label and the opening words are decided on more than one interval of
    /// audio. Fixing at the first partial, on 2 s, labelled an Italian dictation Portuguese.
    pub unfixed_partials: usize,
    /// `speech.context_max_tokens` (Q50).
    pub context_max_tokens: usize,
    /// `speech.context_copy_words` and `speech.context_copy_edits`: what the guard takes for a
    /// copy of the context (#362, #442).
    pub context_copy: ContextCopy,
    /// `speech.max_clip_seconds`: the KV cache is sized for it (Q19, Q36).
    pub max_clip_seconds: u32,
}

impl Default for Settings {
    /// The `speech` group's defaults (spec speech/spec.md).
    fn default() -> Self {
        Settings {
            threads: Threads::Auto,
            pinned_weights: false,
            partial_interval_ms: 2000,
            rollback_tokens: 10,
            unfixed_partials: 2,
            context_max_tokens: 512,
            context_copy: ContextCopy::DEFAULT,
            max_clip_seconds: 1200,
        }
    }
}

/// What one session reads of the settings when it opens, so a live change of
/// the server's config holds from the next session (the server passes the
/// current values; `Settings` holds the defaults a plain `open` uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTuning {
    pub context_max_tokens: usize,
    pub rollback_tokens: usize,
    pub unfixed_partials: usize,
    pub context_copy: ContextCopy,
}

impl Settings {
    /// What a session reads of these settings at its opening.
    pub fn tuning(&self) -> SessionTuning {
        SessionTuning {
            context_max_tokens: self.context_max_tokens,
            rollback_tokens: self.rollback_tokens,
            unfixed_partials: self.unfixed_partials,
            context_copy: self.context_copy,
        }
    }
}

/// The GGUF pair: the decoder, and the audio encoder found beside it.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub decoder: PathBuf,
    pub encoder: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Available,
    /// Unavailable, with the reason (ADR 0067): "the CPU lacks AVX2".
    Unavailable(String),
}

#[derive(Debug)]
pub enum ExecError {
    Unavailable(String),
    Load(LoadError),
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecError::Unavailable(r) => write!(f, "the speech model is unavailable: {r}"),
            ExecError::Load(e) => write!(f, "loading the speech model: {e}"),
        }
    }
}

impl std::error::Error for ExecError {}

/// A loaded model and what runs it.
pub struct Loaded {
    pub model: Model,
    pub pool: Pool,
    pub isa: Isa,
    pub mel: Mel,
}

pub(crate) struct PrefixEntry {
    pub head: Vec<u32>,
    pub cache: Cache,
}

/// What sessions share with the executor.
pub(crate) struct Core {
    pub settings: Settings,
    /// The prefix cache's one entry: the last prompt head and its K/V (Q50).
    pub prefix: Mutex<Option<PrefixEntry>>,
    loaded: Mutex<Option<Arc<Loaded>>>,
}

impl Core {
    fn unload(&self) {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.prefix.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

pub struct Executor {
    core: Arc<Core>,
    files: ModelFiles,
    isa: Result<Isa, &'static str>,
}

impl Executor {
    /// Nothing is mapped or loaded here (Q53): the model loads at the first signal.
    pub fn new(settings: Settings, files: ModelFiles, features: Features) -> Executor {
        let core = Arc::new(Core { settings, prefix: Mutex::new(None), loaded: Mutex::new(None) });
        Executor { core, files, isa: select(features) }
    }

    pub fn settings(&self) -> &Settings {
        &self.core.settings
    }

    pub fn availability(&self) -> Availability {
        match self.isa {
            Ok(_) => Availability::Available,
            Err(r) => Availability::Unavailable(r.to_string()),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.core.loaded.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Loads the model if it is not loaded: the first signal (a session opening, a mic press,
    /// a batch request). Returns the load's report, or `None` when it was already loaded.
    pub fn load(&self) -> Result<Option<LoadReport>, ExecError> {
        self.loaded().map(|(_, r)| r)
    }

    fn loaded(&self) -> Result<(Arc<Loaded>, Option<LoadReport>), ExecError> {
        let isa = self.isa.map_err(|r| ExecError::Unavailable(r.to_string()))?;
        let mut slot = self.core.loaded.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = slot.as_ref() {
            return Ok((l.clone(), None));
        }
        let (n, source) = self.core.settings.threads.resolve();
        // hotpath-lint-allow: one line per load.
        tracing::info!(name: "ignis.speech.threads", threads = n, source, "speech threads {n} ({source})");
        let model = Model::load(&self.files.decoder, &self.files.encoder, self.core.settings.pinned_weights).map_err(ExecError::Load)?;
        let report = model.report.clone();
        // hotpath-lint-allow: one line per load.
        tracing::info!(
            name: "ignis.speech.load",
            millis = report.millis as u64,
            repacked_bytes = report.repacked_bytes,
            mapped_bytes = report.mapped_bytes,
            pinned = report.pinned,
            locked = report.locked,
            "speech model loaded in {} ms",
            report.millis
        );
        let l = Arc::new(Loaded { model, pool: Pool::new(n), isa, mel: Mel::new() });
        *slot = Some(l.clone());
        Ok((l, Some(report)))
    }

    /// Drops the model: the mapped files, the repacked weights, the pool and the prefix cache.
    /// Sessions still open keep theirs until they end. The server's worker asks for it once the
    /// model has gone unused for `speech.idle_unload` (spec 02).
    pub fn unload(&self) {
        self.core.unload();
    }

    /// Opens a transcription with `context` as the system turn, loading the model if needed.
    /// `language` forces the official prefill (Q6); `None` lets the model detect it.
    pub fn open(&self, context: &str, language: Option<&str>) -> Result<Session, ExecError> {
        self.open_tuned(context, language, self.core.settings.tuning())
    }

    /// [`open`](Self::open) with the settings a session reads at its opening passed in: the
    /// server's live `speech.context_max_tokens`, `speech.rollback_tokens` and
    /// `speech.unfixed_partials`.
    pub fn open_tuned(&self, context: &str, language: Option<&str>, tuning: SessionTuning) -> Result<Session, ExecError> {
        let (l, _) = self.loaded()?;
        Ok(Session::open(l, self.core.clone(), context, language, tuning))
    }
}

impl Drop for Executor {
    /// The model and the prefix cache go with the executor; a session still open keeps its own.
    fn drop(&mut self) {
        self.core.unload();
    }
}
