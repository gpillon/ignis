//! The seam between the server and the speech model's executor (spec
//! speech/01 §4, speech/02): the operations the server needs of it and no
//! more. The real executor is `ignis-speech` (`crates/speech`, ticket #351),
//! adapted in [`super::cpu`]; the route, the slot and the queue are built and
//! tested against `FakeExecutor` (test builds and the `fake-speech` feature
//! only) behind this trait, and the adapter is the one place that knows both.
//!
//! The diarizer (spec speech/08) has its seam beside it, [`DiarizerExecutor`]:
//! availability, a stream opened at a geometry, a step run on the diarizer's
//! own thread. The real one is
//! `ignis_speech::diarizer`, adapted in [`super::cpu`]; the Realtime session
//! is tested against `FakeDiarizer`, scripted frame by frame.

use std::path::PathBuf;
#[cfg(any(test, feature = "fake-speech"))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(any(test, feature = "fake-speech"))]
use std::sync::{Arc, Mutex};
#[cfg(any(test, feature = "fake-speech"))]
use std::time::{Duration, Instant};

/// The model's files on disk, as the slot found them: every file of the
/// catalog entry (the decoder GGUF, `decoder`, and the encoder GGUF beside
/// it) under `download.path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechFiles {
    /// The served id.
    pub id: String,
    /// The decoder GGUF, the entry's `artifact`.
    pub decoder: PathBuf,
    /// Every file of the entry, the decoder among them.
    pub files: Vec<PathBuf>,
}

/// What a load reads from the configuration (`speech.threads`,
/// `speech.pinned_weights`): the load's, so a change holds from the next one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadSettings {
    /// `None` is `auto`, the physical cores less one.
    pub threads: Option<u32>,
    pub pinned_weights: bool,
}

/// What a session opens with: the request's context and language, and the
/// `speech` settings a session reads once, when it opens (spec speech/01 §4).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionOpen {
    /// The system turn: the request's `prompt`, then its `keywords[]` as one
    /// line. Empty when there is none. The executor owns the tokenizer, so
    /// it keeps the last `context_max_tokens` tokens of it (spec speech/02
    /// §Context, Q50).
    pub context: String,
    pub context_max_tokens: u32,
    /// `speech.context_copy_words`: the words in a row of one context line a
    /// transcript that only repeats them is emptied for (#362); 0, never.
    pub context_copy_words: u32,
    /// `speech.context_copy_edits`: the words such a transcript may change,
    /// add or drop from a whole context line or keyword (#442); 0, none.
    pub context_copy_edits: u32,
    /// The language forced, by the model's English name (`Italian`), the
    /// official assistant prefill `language <Name><asr_text>`. `None`
    /// auto-detects.
    pub language: Option<String>,
    /// `speech.rollback_tokens`: the tokens of a live transcript kept
    /// provisional. Read at the opening, so a live change holds from the next
    /// session.
    pub rollback_tokens: u32,
    /// `speech.unfixed_partials`: the first live refreshes that fix no text.
    pub unfixed_partials: u32,
    /// The session keeps every sample it is pushed, so a cut can reach back
    /// into the windows it encoded: a Realtime item a speaker change may end
    /// after its audio was heard (spec speech/08).
    pub keep_audio: bool,
}

impl SessionOpen {
    /// A session's opening: its `context` and `language`, and the `speech`
    /// settings in force now.
    pub fn of(group: &crate::config::schema::SpeechGroup, context: String, language: Option<String>) -> Self {
        Self {
            context,
            context_max_tokens: group.context_max_tokens,
            context_copy_words: group.context_copy_words,
            context_copy_edits: group.context_copy_edits,
            language,
            rollback_tokens: group.rollback_tokens,
            unfixed_partials: group.unfixed_partials,
            keep_audio: false,
        }
    }
}

/// A live transcript so far (spec speech/01 §4, `partial()`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Partial {
    /// The stable text: only ever extended, never revised.
    pub stable: String,
    /// The provisional tail, which the next call may revise.
    pub tail: String,
}

/// A finished transcription (`commit()`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Transcript {
    /// The text, beginning with every stable text ever reported.
    pub text: String,
    /// The model's own label (`Italian`); `None` when it said `None`.
    pub language: Option<String>,
    /// The decoded token ids, for `verbose_json`'s segment.
    pub tokens: Vec<u32>,
}

/// Why the executor could not do what it was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechError(pub String);

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpeechError {}

/// One transcription, driven by the compute worker, one call at a time.
pub trait SpeechSession: Send {
    /// 16 kHz mono f32 samples, in order.
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError>;
    /// The transcript so far, or `None` when `stop` said so at one of the
    /// points it checks: a refresh a commit, a clear or the session's end
    /// stopped (Q79), which leaves the session as it was before the call.
    fn partial(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SpeechError>;
    /// One unit of the commit (spec speech/01 §3, Q44): the last window, a window, the
    /// decode's prefill or 16 decode steps, so the worker runs other units between two; `Some`
    /// with the transcript at its end. The session takes nothing after its first.
    fn commit_step(&mut self) -> Result<Option<Transcript>, SpeechError>;
    /// End the transcription at sample `at` of what it was pushed (spec
    /// speech/08): the audio from it on and the stable text go, so the commit
    /// decodes the audio before it afresh.
    fn cut(&mut self, at: usize) -> Result<(), SpeechError>;
}

/// The speech model's executor.
pub trait SpeechExecutor: Send + Sync {
    /// Where it runs: `cpu`, or `gpu_loan` (the `executor` label and log
    /// field).
    fn name(&self) -> &'static str;
    /// Whether this host can run it, and if not why, in a sentence the
    /// Playground shows (`the CPU lacks AVX2 ...`). Cheap: asked per call.
    fn availability(&self) -> Result<(), String>;
    /// Map and prepare the model. Called by the worker before the first
    /// session, and again after an unload.
    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError>;
    /// Give the model's host memory back.
    fn unload(&self);
    fn is_loaded(&self) -> bool;
    /// The files of the model loaded now, `None` while none is: what a
    /// removal of their catalog entry must leave alone (GitHub #380).
    fn loaded_files(&self) -> Option<SpeechFiles>;
    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError>;
}

/// A diarizer's 80 ms frame: each of its 8 speakers' probability.
pub use ignis_speech::diarizer::Probs;
/// A diarizer's streaming geometry, in its 80 ms frames.
pub use ignis_speech::diarizer::Geometry;

/// The speaker cache's constants a stream opens with (the checkpoint's
/// `streaming_config` keys), each `None` for the weights' own: `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CacheConstants {
    pub prediction_score_threshold: Option<f32>,
    pub latest_frames_score_boost: Option<f32>,
    pub strong_boost_rate: Option<f32>,
    pub weak_boost_rate: Option<f32>,
    pub min_positive_scores_rate: Option<f32>,
    pub speaker_cache_silence_frames_per_speaker: Option<usize>,
}

/// What a diarizer stream opens with: its geometry and the speaker cache's
/// constants, the `diarizer` group's when a session's speakers start; and
/// `diarizer.threads`, which only a load reads, so a change holds from the
/// next one (`None` is `auto`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamOpen {
    pub geometry: Geometry,
    pub constants: CacheConstants,
    pub threads: Option<u32>,
}

impl StreamOpen {
    /// The live geometry (`diarizer.live_*`), or the file one
    /// (`diarizer.file_*`), as the session asked, with the group's constants
    /// and threads.
    pub(crate) fn of(group: &crate::config::schema::DiarizerGroup, mode: super::realtime::config::Diarization) -> Self {
        use super::realtime::config::Diarization;
        let frames = |n: u32| n as usize;
        let geometry = match mode {
            Diarization::Live => Geometry {
                speaker_cache: frames(group.live_speaker_cache_frames),
                fifo: frames(group.live_fifo_frames),
                chunk: frames(group.live_chunk_frames),
                right_context: frames(group.live_right_context_frames),
                update_period: frames(group.live_update_period_frames),
            },
            Diarization::File => Geometry {
                speaker_cache: frames(group.file_speaker_cache_frames),
                fifo: frames(group.file_fifo_frames),
                chunk: frames(group.file_chunk_frames),
                right_context: frames(group.file_right_context_frames),
                update_period: frames(group.file_update_period_frames),
            },
        };
        let decimal = |thousandths: Option<u32>| thousandths.map(|value| value as f32 / 1000.0);
        let constants = CacheConstants {
            prediction_score_threshold: decimal(group.prediction_score_threshold),
            latest_frames_score_boost: decimal(group.latest_frames_score_boost),
            strong_boost_rate: decimal(group.strong_boost_rate),
            weak_boost_rate: decimal(group.weak_boost_rate),
            min_positive_scores_rate: decimal(group.min_positive_scores_rate),
            speaker_cache_silence_frames_per_speaker: group.speaker_cache_silence_frames_per_speaker.map(|n| n as usize),
        };
        Self { geometry, constants, threads: group.threads }
    }
}

/// A step's frames: the first's index, from the stream's first, and each
/// 80 ms frame's speakers.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Frames {
    pub first: usize,
    pub probs: Vec<Probs>,
}

impl Frames {
    /// The frame after the last.
    pub fn end(&self) -> usize {
        self.first + self.probs.len()
    }
}

/// One stream of the diarizer: a Realtime session's 16 kHz audio from its
/// first sample, scored a step of its geometry at a time on the diarizer's
/// thread.
pub trait DiarizerStream: Send {
    /// The stream's next 16 kHz samples.
    fn push(&mut self, samples: &[f32]);
    /// The samples heard, from the stream's first, that the next step waits
    /// for.
    fn samples_needed(&self) -> usize;
    /// The next step, once its samples are in; `None` before.
    fn step(&mut self) -> Result<Option<Frames>, SpeechError>;
    /// The frames heard past the last step, scored as if the stream ended
    /// now, with no look-ahead: a copy of the stream finished, the stream
    /// itself untouched. Empty when nothing is past the last step.
    fn tail(&self) -> Result<Frames, SpeechError>;
}

/// The diarizer's executor (spec speech/08): who spoke when, on a thread and
/// a pool of its own beside the speech model's worker.
pub trait DiarizerExecutor: Send + Sync {
    /// Whether the diarizer can serve, and if not why, in a sentence a client
    /// reads. Cheap: asked per session.
    fn availability(&self) -> Result<(), String>;
    /// Open a stream, loading the weights first if they are not, with a pool
    /// of `open.threads`: on the diarizer's thread, whose steps compute on
    /// that pool, never the speech model's.
    fn open(&self, open: StreamOpen) -> Result<Box<dyn DiarizerStream>, SpeechError>;
    /// Give the weights' host memory back.
    fn unload(&self);
    fn is_loaded(&self) -> bool;
    /// The id `diarizer.model` names, available or not; `None` when it names
    /// none (`off`, or `auto` with no diarizer in the catalog).
    fn id(&self) -> Option<String> {
        None
    }
    /// The GGUF the diarizer has loaded now, `None` while none is: what a
    /// removal of its catalog entry must leave alone (GitHub #380).
    fn loaded_weights(&self) -> Option<PathBuf> {
        None
    }
    /// The live `diarizer.*` settings changed: a diarizer that resolves its
    /// model from them takes the new one for the next stream.
    fn follow(&self, _: &crate::config::schema::DiarizerGroup) {}
}

/// The diarizer of a server that has none configured: unavailable, saying
/// so.
#[derive(Debug, Default)]
pub struct NoDiarizer;

/// Why a server with no diarizer cannot give speakers.
pub const NO_DIARIZER: &str = "no diarizer is configured on this server";

impl DiarizerExecutor for NoDiarizer {
    fn availability(&self) -> Result<(), String> {
        Err(NO_DIARIZER.to_owned())
    }
    fn open(&self, _: StreamOpen) -> Result<Box<dyn DiarizerStream>, SpeechError> {
        Err(SpeechError(NO_DIARIZER.to_owned()))
    }
    fn unload(&self) {}
    fn is_loaded(&self) -> bool {
        false
    }
}

/// A scripted diarizer for tests of everything around the real one: each
/// 80 ms frame's speakers are `script(frame)`, from the stream's first. A
/// stream steps as the real one does at its geometry: its first step once its
/// chunk and right context are heard, each later one a chunk on, each step's
/// frames its chunk's. Its tail scores the frames heard past the last step.
/// It records what it was opened with and every call, in order
/// ([`FakeDiarizer::calls`]): `open <chunk>`, `step <first> <frames>`,
/// `tail <first> <frames>`; each step takes `step_delay`, and when each began
/// and ended is kept ([`FakeDiarizer::steps_at`]). Test builds only.
#[cfg(any(test, feature = "fake-speech"))]
pub struct FakeDiarizer {
    /// The frames of the streams opened from now: [`FakeDiarizer::set_script`].
    pub script: Mutex<Arc<dyn Fn(usize) -> Probs + Send + Sync>>,
    pub unavailable: Mutex<Option<String>>,
    pub fail_open: AtomicBool,
    pub loaded: AtomicBool,
    /// What [`DiarizerExecutor::loaded_weights`] names while it is loaded.
    pub weights: Mutex<Option<PathBuf>>,
    pub step_delay: Arc<Mutex<Duration>>,
    pub opened: Mutex<Vec<StreamOpen>>,
    pub calls: Arc<Mutex<Vec<String>>>,
    /// When each step began and ended, over every stream, in order.
    pub steps_at: Arc<Mutex<Vec<(Instant, Instant)>>>,
    /// The steps begun so far, over every stream, the one running included.
    pub steps_begun: Arc<AtomicUsize>,
}

#[cfg(any(test, feature = "fake-speech"))]
impl FakeDiarizer {
    /// A diarizer that hears speaker `speaker(frame)` in each frame, `None`
    /// silence.
    pub fn hearing(speaker: impl Fn(usize) -> Option<usize> + Send + Sync + 'static) -> Self {
        Self::scripted(move |frame| {
            let mut probs = [0.01; ignis_speech::diarizer::SPEAKERS];
            if let Some(speaker) = speaker(frame) {
                probs[speaker] = 0.9;
            }
            probs
        })
    }

    /// A diarizer whose frames are `script`'s.
    pub fn scripted(script: impl Fn(usize) -> Probs + Send + Sync + 'static) -> Self {
        Self {
            script: Mutex::new(Arc::new(script)),
            unavailable: Mutex::new(None),
            fail_open: AtomicBool::new(false),
            loaded: AtomicBool::new(false),
            weights: Mutex::new(None),
            step_delay: Arc::new(Mutex::new(Duration::ZERO)),
            opened: Mutex::new(Vec::new()),
            calls: Arc::new(Mutex::new(Vec::new())),
            steps_at: Arc::new(Mutex::new(Vec::new())),
            steps_begun: Arc::default(),
        }
    }

    pub fn set_unavailable(&self, reason: Option<&str>) {
        *self.unavailable.lock().unwrap() = reason.map(str::to_owned);
    }

    /// The streams opened from now hear `speaker(frame)` in each frame; the
    /// ones open keep their script.
    pub fn set_script(&self, speaker: impl Fn(usize) -> Option<usize> + Send + Sync + 'static) {
        *self.script.lock().unwrap() = Self::hearing(speaker).script.into_inner().unwrap();
    }

    /// When each step so far began and ended ([`FakeDiarizer::steps_at`]).
    pub fn steps_at(&self) -> Vec<(Instant, Instant)> {
        self.steps_at.lock().unwrap().clone()
    }

    /// Every step from now takes `delay`.
    pub fn set_step_delay(&self, delay: Duration) {
        *self.step_delay.lock().unwrap() = delay;
    }

    /// The calls so far ([`FakeDiarizer::calls`]).
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[cfg(any(test, feature = "fake-speech"))]
struct FakeStream {
    geometry: Geometry,
    script: Arc<dyn Fn(usize) -> Probs + Send + Sync>,
    heard: usize,
    done: usize,
    delay: Arc<Mutex<Duration>>,
    calls: Arc<Mutex<Vec<String>>>,
    steps_at: Arc<Mutex<Vec<(Instant, Instant)>>>,
    steps_begun: Arc<AtomicUsize>,
}

#[cfg(any(test, feature = "fake-speech"))]
impl FakeStream {
    fn frames(&self, first: usize, n: usize) -> Frames {
        Frames { first, probs: (first..first + n).map(|frame| (self.script)(frame)).collect() }
    }
}

#[cfg(any(test, feature = "fake-speech"))]
impl DiarizerStream for FakeStream {
    fn push(&mut self, samples: &[f32]) {
        self.heard += samples.len();
    }

    fn samples_needed(&self) -> usize {
        (self.done + self.geometry.chunk + self.geometry.right_context) * ignis_speech::diarizer::FRAME_SAMPLES
    }

    fn step(&mut self) -> Result<Option<Frames>, SpeechError> {
        if self.heard < self.samples_needed() {
            return Ok(None);
        }
        let began = Instant::now();
        self.steps_begun.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(*self.delay.lock().unwrap());
        let frames = self.frames(self.done, self.geometry.chunk);
        self.calls.lock().unwrap().push(format!("step {} {}", frames.first, frames.probs.len()));
        self.steps_at.lock().unwrap().push((began, Instant::now()));
        self.done = frames.end();
        Ok(Some(frames))
    }

    fn tail(&self) -> Result<Frames, SpeechError> {
        let heard = self.heard.div_ceil(ignis_speech::diarizer::FRAME_SAMPLES);
        let frames = self.frames(self.done, heard.saturating_sub(self.done));
        self.calls.lock().unwrap().push(format!("tail {} {}", frames.first, frames.probs.len()));
        Ok(frames)
    }
}

#[cfg(any(test, feature = "fake-speech"))]
impl DiarizerExecutor for FakeDiarizer {
    fn availability(&self) -> Result<(), String> {
        match self.unavailable.lock().unwrap().clone() {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    fn open(&self, open: StreamOpen) -> Result<Box<dyn DiarizerStream>, SpeechError> {
        if self.fail_open.load(Ordering::SeqCst) {
            return Err(SpeechError("the fake diarizer was told to fail its open".to_owned()));
        }
        self.loaded.store(true, Ordering::SeqCst);
        self.calls.lock().unwrap().push(format!("open {}", open.geometry.chunk));
        self.opened.lock().unwrap().push(open);
        Ok(Box::new(FakeStream {
            geometry: open.geometry,
            script: Arc::clone(&self.script.lock().unwrap()),
            heard: 0,
            done: 0,
            delay: Arc::clone(&self.step_delay),
            calls: Arc::clone(&self.calls),
            steps_at: Arc::clone(&self.steps_at),
            steps_begun: Arc::clone(&self.steps_begun),
        }))
    }

    fn unload(&self) {
        self.loaded.store(false, Ordering::SeqCst);
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    fn id(&self) -> Option<String> {
        Some("fake-diarizer".to_owned())
    }

    fn loaded_weights(&self) -> Option<PathBuf> {
        self.loaded.load(Ordering::SeqCst).then(|| self.weights.lock().unwrap().clone()).flatten()
    }
}

/// The executor of a binary that has none yet: every host is unavailable,
/// saying so.
#[derive(Debug, Default)]
pub struct NoExecutor;

const NO_EXECUTOR: &str = "this build has no speech executor";

impl SpeechExecutor for NoExecutor {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn availability(&self) -> Result<(), String> {
        Err(NO_EXECUTOR.to_owned())
    }
    fn load(&self, _: &SpeechFiles, _: LoadSettings) -> Result<(), SpeechError> {
        Err(SpeechError(NO_EXECUTOR.to_owned()))
    }
    fn unload(&self) {}
    fn is_loaded(&self) -> bool {
        false
    }
    fn loaded_files(&self) -> Option<SpeechFiles> {
        None
    }
    fn open(&self, _: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        Err(SpeechError(NO_EXECUTOR.to_owned()))
    }
}

/// A scripted executor for tests of everything around the real one. A
/// session's transcript is the `words` it was scripted with: one more of them
/// is stable per 8 s window pushed (128,000 samples), the last arriving only
/// at the commit. It records what it was opened with, the samples it was
/// pushed, what it was loaded with, every call its sessions took, in order
/// ([`FakeExecutor::calls`]), when each refresh began and ended, how many stopped, and
/// the silence each session was fed first. A refresh checks its stop every
/// millisecond it takes; a commit takes `commit_steps` steps. Test builds only.
#[cfg(any(test, feature = "fake-speech"))]
#[derive(Debug)]
pub struct FakeExecutor {
    pub words: Vec<String>,
    pub language: Option<String>,
    pub name: &'static str,
    pub unavailable: Mutex<Option<String>>,
    pub loaded: AtomicBool,
    pub loads: AtomicUsize,
    pub unloads: AtomicUsize,
    pub fail_load: AtomicBool,
    /// Read by each session at its commit.
    pub fail_commit: Arc<AtomicBool>,
    /// How long each call to `push`, `partial` and `commit` takes.
    pub step_delay: Mutex<Duration>,
    /// How long each `partial` takes beyond `step_delay`, read at each:
    /// a refresh long beside the windows and the commit.
    pub partial_delay: Arc<Mutex<Duration>>,
    /// The refreshes stopped before their end (Q79).
    pub stopped: Arc<AtomicUsize>,
    /// The steps each session's commit takes, read when it opens; 1 by default.
    pub commit_steps: AtomicUsize,
    pub opened: Mutex<Vec<SessionOpen>>,
    /// The length of every `push`, in order, over every session.
    pub pushed: Arc<Mutex<Vec<usize>>>,
    pub load_settings: Mutex<Vec<LoadSettings>>,
    pub load_files: Mutex<Vec<SpeechFiles>>,
    /// Every call, in order, over every session: `open <n> <context>`,
    /// `push <n> <samples>`, `partial <n>` and `commit <n>`, `n` counting
    /// the sessions opened from 0.
    pub calls: Arc<Mutex<Vec<String>>>,
    /// When each `partial` began, over every session, in order.
    pub partials_at: Arc<Mutex<Vec<Instant>>>,
    /// When each `partial` ended, stopped or not, in the same order.
    pub partials_ended: Arc<Mutex<Vec<Instant>>>,
    /// Per session, in the order opened, the samples it was pushed before
    /// its first that is not zero: the digital silence it was fed first.
    pub silent_lead: Arc<Mutex<Vec<usize>>>,
}

#[cfg(any(test, feature = "fake-speech"))]
impl FakeExecutor {
    pub fn saying(words: &[&str]) -> Self {
        Self {
            words: words.iter().map(|word| (*word).to_owned()).collect(),
            language: Some("English".to_owned()),
            name: "cpu",
            unavailable: Mutex::new(None),
            loaded: AtomicBool::new(false),
            loads: AtomicUsize::new(0),
            unloads: AtomicUsize::new(0),
            fail_load: AtomicBool::new(false),
            fail_commit: Arc::new(AtomicBool::new(false)),
            step_delay: Mutex::new(Duration::ZERO),
            partial_delay: Arc::new(Mutex::new(Duration::ZERO)),
            stopped: Arc::new(AtomicUsize::new(0)),
            commit_steps: AtomicUsize::new(1),
            opened: Mutex::new(Vec::new()),
            pushed: Arc::new(Mutex::new(Vec::new())),
            load_settings: Mutex::new(Vec::new()),
            load_files: Mutex::new(Vec::new()),
            calls: Arc::new(Mutex::new(Vec::new())),
            partials_at: Arc::new(Mutex::new(Vec::new())),
            partials_ended: Arc::new(Mutex::new(Vec::new())),
            silent_lead: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn set_unavailable(&self, reason: Option<&str>) {
        *self.unavailable.lock().unwrap() = reason.map(str::to_owned);
    }

    pub fn set_step_delay(&self, delay: Duration) {
        *self.step_delay.lock().unwrap() = delay;
    }

    /// Every `partial` from now, of every session, takes `delay` more.
    pub fn set_partial_delay(&self, delay: Duration) {
        *self.partial_delay.lock().unwrap() = delay;
    }

    /// Every sample pushed to any session, in total.
    pub fn samples_pushed(&self) -> usize {
        self.pushed.lock().unwrap().iter().sum()
    }

    /// The calls so far ([`FakeExecutor::calls`]).
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    /// When each refresh so far began ([`FakeExecutor::partials_at`]).
    pub fn partials_at(&self) -> Vec<Instant> {
        self.partials_at.lock().unwrap().clone()
    }

    /// When each refresh so far began and ended, stopped or not.
    pub fn partial_spans(&self) -> Vec<(Instant, Instant)> {
        let began = self.partials_at.lock().unwrap().clone();
        began.into_iter().zip(self.partials_ended.lock().unwrap().iter().copied()).collect()
    }

    /// Each session's leading digital silence so far
    /// ([`FakeExecutor::silent_lead`]).
    pub fn silent_lead(&self) -> Vec<usize> {
        self.silent_lead.lock().unwrap().clone()
    }
}

/// Samples in one 8 s window at 16 kHz.
pub const WINDOW_SAMPLES: usize = 128_000;

#[cfg(any(test, feature = "fake-speech"))]
struct FakeSession {
    words: Vec<String>,
    language: Option<String>,
    delay: Duration,
    partial_delay: Arc<Mutex<Duration>>,
    stopped: Arc<AtomicUsize>,
    fail_commit: Arc<AtomicBool>,
    pushed: usize,
    stable_words: usize,
    log: Arc<Mutex<Vec<usize>>>,
    /// Its number among the sessions opened, and the executor's call log.
    index: usize,
    calls: Arc<Mutex<Vec<String>>>,
    partials_at: Arc<Mutex<Vec<Instant>>>,
    partials_ended: Arc<Mutex<Vec<Instant>>>,
    /// The executor's record of leading silence, and whether this session
    /// has been pushed a sample that is not zero.
    silent_lead: Arc<Mutex<Vec<usize>>>,
    sounded: bool,
    /// The commit's steps still to take.
    commit_steps: usize,
}

#[cfg(any(test, feature = "fake-speech"))]
impl SpeechSession for FakeSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        std::thread::sleep(self.delay);
        self.pushed += samples.len();
        self.log.lock().unwrap().push(samples.len());
        self.calls.lock().unwrap().push(format!("push {} {}", self.index, samples.len()));
        if !self.sounded {
            let silent = samples.iter().take_while(|&&sample| sample == 0.0).count();
            self.silent_lead.lock().unwrap()[self.index] += silent;
            self.sounded = silent < samples.len();
        }
        // One more stable word per window pushed, the last left to the commit.
        self.stable_words = (self.pushed / WINDOW_SAMPLES).min(self.words.len().saturating_sub(1));
        Ok(())
    }

    /// The audio past `at` goes, and the stable words with it, logged `cut <n> <at>`.
    fn cut(&mut self, at: usize) -> Result<(), SpeechError> {
        self.calls.lock().unwrap().push(format!("cut {} {at}", self.index));
        self.pushed = self.pushed.min(at);
        self.stable_words = 0;
        Ok(())
    }

    fn partial(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SpeechError> {
        self.calls.lock().unwrap().push(format!("partial {}", self.index));
        self.partials_at.lock().unwrap().push(Instant::now());
        let end = Instant::now() + self.delay + *self.partial_delay.lock().unwrap();
        loop {
            if stop() {
                self.stopped.fetch_add(1, Ordering::SeqCst);
                self.partials_ended.lock().unwrap().push(Instant::now());
                return Ok(None);
            }
            let left = end.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            std::thread::sleep(left.min(Duration::from_millis(1)));
        }
        self.partials_ended.lock().unwrap().push(Instant::now());
        let (stable, tail) = self.words.split_at(self.stable_words);
        let tail = tail.first().map(|word| format!(" {word}")).unwrap_or_default();
        Ok(Some(Partial { stable: stable.join(" "), tail }))
    }

    /// One step of `commit_steps`, each logged `commit <n>`; the last fails if `fail_commit`.
    fn commit_step(&mut self) -> Result<Option<Transcript>, SpeechError> {
        if self.commit_steps == 0 {
            return Err(SpeechError("committed already".to_owned()));
        }
        self.calls.lock().unwrap().push(format!("commit {}", self.index));
        std::thread::sleep(self.delay);
        self.commit_steps -= 1;
        if self.commit_steps > 0 {
            return Ok(None);
        }
        if self.fail_commit.load(Ordering::SeqCst) {
            return Err(SpeechError("the fake executor was told to fail".to_owned()));
        }
        Ok(Some(Transcript {
            text: self.words.join(" "),
            language: self.language.clone(),
            tokens: (0..self.words.len() as u32).collect(),
        }))
    }
}

#[cfg(any(test, feature = "fake-speech"))]
impl SpeechExecutor for FakeExecutor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn availability(&self) -> Result<(), String> {
        match self.unavailable.lock().unwrap().clone() {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError> {
        if self.fail_load.load(Ordering::SeqCst) {
            return Err(SpeechError("the fake executor was told to fail its load".to_owned()));
        }
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.load_files.lock().unwrap().push(files.clone());
        self.load_settings.lock().unwrap().push(settings);
        self.loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn unload(&self) {
        self.unloads.fetch_add(1, Ordering::SeqCst);
        self.loaded.store(false, Ordering::SeqCst);
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    /// The files of its last load while it is loaded.
    fn loaded_files(&self) -> Option<SpeechFiles> {
        self.is_loaded().then(|| self.load_files.lock().unwrap().last().cloned()).flatten()
    }

    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        if !self.is_loaded() {
            return Err(SpeechError("opened before the model was loaded".to_owned()));
        }
        // A forced language is what the model reports it heard.
        let language = open.language.clone().or_else(|| self.language.clone());
        let mut opened = self.opened.lock().unwrap();
        let index = opened.len();
        self.calls.lock().unwrap().push(format!("open {index} {}", open.context));
        self.silent_lead.lock().unwrap().push(0);
        opened.push(open);
        Ok(Box::new(FakeSession {
            words: self.words.clone(),
            language,
            delay: *self.step_delay.lock().unwrap(),
            partial_delay: Arc::clone(&self.partial_delay),
            stopped: Arc::clone(&self.stopped),
            fail_commit: Arc::clone(&self.fail_commit),
            pushed: 0,
            stable_words: 0,
            log: Arc::clone(&self.pushed),
            index,
            calls: Arc::clone(&self.calls),
            partials_at: Arc::clone(&self.partials_at),
            partials_ended: Arc::clone(&self.partials_ended),
            silent_lead: Arc::clone(&self.silent_lead),
            sounded: false,
            commit_steps: self.commit_steps.load(Ordering::SeqCst).max(1),
        }))
    }
}
