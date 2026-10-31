//! The speech model's CPU executor in the server: `ignis-speech` (spec
//! speech/01, ADR 0068) behind [`SpeechExecutor`].
//!
//! A load builds the crate's executor from the catalog entry's files and the
//! load's settings; the server's worker keeps the idle clock (spec 02) and
//! asks for the unload.
//!
//! Beside it, [`CpuDiarizer`]: the crate's diarizer (spec speech/08) behind
//! [`DiarizerExecutor`], its steps computed on a pool of its own
//! (`diarizer.threads`), never the speech model's, with the speech model's
//! ISA.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use ignis_speech::cpu::Isa;
use ignis_speech::diarizer::{Constants, Diarizer, Stream};
use ignis_speech::pool::Pool;
use ignis_speech::{ContextCopy, Features, ModelFiles, SessionTuning, Settings, Threads};

use super::executor::{
    CacheConstants, DiarizerExecutor, DiarizerStream, Frames, LoadSettings, Partial, SessionOpen, SpeechError, SpeechExecutor, SpeechFiles, SpeechSession,
    StreamOpen, Transcript,
};

/// The reason a person reads on a CPU without AVX2, FMA and F16C.
pub const NO_AVX2_REASON: &str = "the CPU lacks AVX2, which the speech model needs";

/// A loaded model: what it was loaded from and with, and the crate's executor.
struct Loaded {
    files: SpeechFiles,
    settings: LoadSettings,
    exec: ignis_speech::Executor,
}

pub struct CpuExecutor {
    features: Features,
    loaded: Mutex<Option<Loaded>>,
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuExecutor {
    pub fn new() -> CpuExecutor {
        CpuExecutor::with_features(Features::detect())
    }

    /// With features taken away from what detection reported (a test of the unavailable state).
    fn with_features(features: Features) -> CpuExecutor {
        CpuExecutor { features, loaded: Mutex::new(None) }
    }

    fn slot(&self) -> MutexGuard<'_, Option<Loaded>> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The instruction set the speech model computes with: the diarizer's
    /// too.
    fn isa(&self) -> Result<Isa, SpeechError> {
        self.availability().map_err(SpeechError)?;
        ignis_speech::cpu::select(self.features).map_err(|reason| SpeechError(reason.to_owned()))
    }
}

fn err(e: impl std::fmt::Display) -> SpeechError {
    SpeechError(e.to_string())
}

/// The encoder is the entry's file under the fixed `mmproj-` name, beside the
/// decoder (spec speech/01 §6; the catalog holds an entry to having one).
fn encoder_of(files: &SpeechFiles) -> Option<&std::path::PathBuf> {
    files.files.iter().find(|f| {
        *f != &files.decoder
            && f.file_name().and_then(|n| n.to_str()).is_some_and(crate::download::catalog::is_speech_encoder)
    })
}

impl SpeechExecutor for CpuExecutor {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn availability(&self) -> Result<(), String> {
        match ignis_speech::cpu::select(self.features) {
            Ok(_) => Ok(()),
            Err(r) if r == ignis_speech::NO_AVX2 => Err(NO_AVX2_REASON.to_owned()),
            Err(r) => Err(r.to_owned()),
        }
    }

    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError> {
        self.availability().map_err(SpeechError)?;
        let mut slot = self.slot();
        if let Some(l) = slot.as_ref() {
            if l.files == *files && l.settings == settings && l.exec.is_loaded() {
                return Ok(());
            }
        }
        let encoder = encoder_of(files)
            .ok_or_else(|| SpeechError(format!("{}: no audio encoder (mmproj-*.gguf) beside {}", files.id, files.decoder.display())))?
            .clone();
        let model = ModelFiles { decoder: files.decoder.clone(), encoder };
        let threads = settings.threads.map_or(Threads::Auto, |n| Threads::Count(n as usize));
        let s = Settings { threads, pinned_weights: settings.pinned_weights, ..Settings::default() };
        *slot = None;
        let exec = ignis_speech::Executor::new(s, model, self.features);
        exec.load().map_err(err)?;
        *slot = Some(Loaded { files: files.clone(), settings, exec });
        Ok(())
    }

    fn unload(&self) {
        self.slot().take();
    }

    fn is_loaded(&self) -> bool {
        self.slot().as_ref().is_some_and(|l| l.exec.is_loaded())
    }

    fn loaded_files(&self) -> Option<SpeechFiles> {
        self.slot().as_ref().filter(|l| l.exec.is_loaded()).map(|l| l.files.clone())
    }

    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        let slot = self.slot();
        let l = slot.as_ref().ok_or_else(|| SpeechError("the speech model is not loaded".into()))?;
        let tuning = SessionTuning {
            context_max_tokens: open.context_max_tokens as usize,
            rollback_tokens: open.rollback_tokens as usize,
            unfixed_partials: open.unfixed_partials as usize,
            context_copy: ContextCopy { run: open.context_copy_words as usize, edits: open.context_copy_edits as usize },
        };
        let mut s = l.exec.open_tuned(&open.context, open.language.as_deref(), tuning).map_err(err)?;
        if open.keep_audio {
            s.keep_audio();
        }
        Ok(Box::new(CpuSession::Open(s)))
    }
}

/// A transcription: open, then committing a unit at a time, then spent.
enum CpuSession {
    Open(ignis_speech::Session),
    Committing(ignis_speech::Commit),
    Spent,
}

const COMMITTED: &str = "the transcription is committed and takes no more audio";

impl SpeechSession for CpuSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        match self {
            CpuSession::Open(s) => s.push(samples).map_err(err),
            _ => Err(SpeechError(COMMITTED.to_owned())),
        }
    }

    fn partial(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SpeechError> {
        let CpuSession::Open(s) = self else { return Err(SpeechError(COMMITTED.to_owned())) };
        let p = s.partial_unless(stop).map_err(err)?;
        Ok(p.map(|p| Partial { stable: p.stable, tail: p.provisional }))
    }

    fn commit_step(&mut self) -> Result<Option<Transcript>, SpeechError> {
        if let CpuSession::Open(_) = self {
            let CpuSession::Open(s) = std::mem::replace(self, CpuSession::Spent) else { unreachable!() };
            *self = CpuSession::Committing(s.into_commit());
        }
        let CpuSession::Committing(commit) = self else { return Err(SpeechError(COMMITTED.to_owned())) };
        let Some(t) = commit.step().map_err(err)? else { return Ok(None) };
        *self = CpuSession::Spent;
        Ok(Some(Transcript { text: t.text, language: t.language, tokens: t.tokens }))
    }

    fn cut(&mut self, at: usize) -> Result<(), SpeechError> {
        match self {
            CpuSession::Open(s) => s.cut(at).map_err(err),
            _ => Err(SpeechError(COMMITTED.to_owned())),
        }
    }
}

/// The diarizer on the CPU (spec speech/08, ADR 0069): Nemotron 3
/// Diarization's GGUF, loaded at the first stream that opens with a pool of
/// its own, its steps computed there with `speech`'s ISA.
pub struct CpuDiarizer {
    weights: PathBuf,
    pinned: bool,
    speech: Arc<CpuExecutor>,
    loaded: Mutex<Option<Arc<Compute>>>,
}

/// The diarizer loaded: its weights, the pool of `diarizer.threads` its steps
/// compute on (the calling thread, the diarizer's, among them), created with
/// the weights and dropped with them, and the speech model's ISA.
struct Compute {
    diarizer: Diarizer,
    pool: Pool,
    isa: Isa,
}

impl CpuDiarizer {
    /// The diarizer whose weights are the GGUF at `weights`, beside `speech`.
    pub fn new(weights: PathBuf, pinned: bool, speech: Arc<CpuExecutor>) -> CpuDiarizer {
        CpuDiarizer { weights, pinned, speech, loaded: Mutex::new(None) }
    }

    fn slot(&self) -> MutexGuard<'_, Option<Arc<Compute>>> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The weights and their pool, loaded on `threads` (`None` is `auto`) if
    /// they are not, and whether they were just now.
    fn diarizer(&self, threads: Option<u32>) -> Result<(Arc<Compute>, bool), SpeechError> {
        let mut slot = self.slot();
        if let Some(d) = slot.as_ref() {
            return Ok((Arc::clone(d), false));
        }
        let isa = self.speech.isa()?;
        let diarizer = Diarizer::load(&self.weights, self.pinned).map_err(err)?;
        let (n, _) = threads.map_or(Threads::Auto, |n| Threads::Count(n as usize)).resolve_diarizer();
        let d = Arc::new(Compute { diarizer, pool: Pool::named(n, "ignis-diarizer"), isa });
        *slot = Some(Arc::clone(&d));
        Ok((d, true))
    }

    /// The threads the loaded diarizer computes on, `None` while it is not
    /// loaded.
    pub fn threads(&self) -> Option<usize> {
        self.slot().as_ref().map(|d| d.pool.threads())
    }
}

/// The speaker cache's constants: each override, else the weights' own.
fn constants(own: Constants, over: CacheConstants) -> Constants {
    Constants {
        prediction_score_threshold: over.prediction_score_threshold.unwrap_or(own.prediction_score_threshold),
        latest_frames_score_boost: over.latest_frames_score_boost.unwrap_or(own.latest_frames_score_boost),
        strong_boost_rate: over.strong_boost_rate.unwrap_or(own.strong_boost_rate),
        weak_boost_rate: over.weak_boost_rate.unwrap_or(own.weak_boost_rate),
        min_positive_scores_rate: over.min_positive_scores_rate.unwrap_or(own.min_positive_scores_rate),
        speaker_cache_silence_frames_per_speaker: over.speaker_cache_silence_frames_per_speaker.unwrap_or(own.speaker_cache_silence_frames_per_speaker),
    }
}

impl DiarizerExecutor for CpuDiarizer {
    fn availability(&self) -> Result<(), String> {
        self.speech.availability()?;
        match self.weights.is_file() {
            true => Ok(()),
            false => Err(format!("no diarizer weights at {}", self.weights.display())),
        }
    }

    fn open(&self, open: StreamOpen) -> Result<Box<dyn DiarizerStream>, SpeechError> {
        let (d, loaded) = self.diarizer(open.threads)?;
        let constants = constants(d.diarizer.weights().constants, open.constants);
        if loaded {
            // The constants in force: the `diarizer` group's overrides over the checkpoint's.
            tracing::info!(
                name: "ignis.diarizer.load",
                millis = d.diarizer.weights().report.millis as u64,
                mapped_bytes = d.diarizer.weights().report.mapped_bytes,
                threads = d.pool.threads(),
                prediction_score_threshold = constants.prediction_score_threshold,
                latest_frames_score_boost = constants.latest_frames_score_boost,
                strong_boost_rate = constants.strong_boost_rate,
                weak_boost_rate = constants.weak_boost_rate,
                min_positive_scores_rate = constants.min_positive_scores_rate,
                speaker_cache_silence_frames_per_speaker = constants.speaker_cache_silence_frames_per_speaker,
                "the diarizer loaded in {} ms, on {} threads",
                d.diarizer.weights().report.millis,
                d.pool.threads()
            );
        }
        let stream = d.diarizer.open(open.geometry, constants);
        Ok(Box::new(CpuStream { compute: d, stream }))
    }

    fn unload(&self) {
        self.slot().take();
    }

    fn is_loaded(&self) -> bool {
        self.slot().is_some()
    }
}

/// A stream of the crate's diarizer, and what computes its steps.
struct CpuStream {
    compute: Arc<Compute>,
    stream: Stream,
}

fn frames(step: ignis_speech::diarizer::Step) -> Frames {
    Frames { first: step.first, probs: step.frames }
}

impl DiarizerStream for CpuStream {
    fn push(&mut self, samples: &[f32]) {
        self.stream.push(samples);
    }

    fn samples_needed(&self) -> usize {
        self.stream.samples_needed()
    }

    fn step(&mut self) -> Result<Option<Frames>, SpeechError> {
        if !self.stream.ready() {
            return Ok(None);
        }
        let c = &self.compute;
        Ok(self.stream.step(&c.diarizer, &c.pool, c.isa).map(frames))
    }

    fn tail(&self) -> Result<Frames, SpeechError> {
        let c = &self.compute;
        let mut copy = self.stream.clone();
        let first = copy.frames_done();
        Ok(copy.finish(&c.diarizer, &c.pool, c.isa).map(frames).unwrap_or(Frames { first, probs: Vec::new() }))
    }
}

/// A session's commit stepped to its end (tests).
#[cfg(test)]
fn commit(mut s: Box<dyn SpeechSession>) -> Result<Transcript, SpeechError> {
    loop {
        if let Some(t) = s.commit_step()? {
            return Ok(t);
        }
    }
}

// Declared here, outside the inline `tests` block, so the path is relative to
// this file's directory: inside an inline module it would go through
// `cpu/tests/`, which Linux cannot resolve because it does not exist.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../speech/tests/common/synthetic.rs"]
mod synthetic;

#[cfg(test)]
mod tests {
    use super::*;

    fn files(dir: &std::path::Path, encoder: bool) -> SpeechFiles {
        let decoder = dir.join("Qwen3-ASR-1.7B-Q4_K_M.gguf");
        let mut all = vec![decoder.clone()];
        if encoder {
            all.push(dir.join("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"));
        }
        SpeechFiles { id: "qwen3-asr-1.7b".into(), decoder, files: all }
    }

    /// Spec speech/01 AC 22: each built-in speech entry, as `Speech::resolve`
    /// hands its files over, has its decoder as the artifact and exactly its
    /// `mmproj-` file as the encoder this adapter picks.
    #[test]
    fn each_built_in_speech_entry_gives_the_adapter_its_decoder_and_encoder() {
        let dir = std::path::Path::new("models");
        let catalog = crate::download::Catalog::built_in();
        for (id, decoder, encoder) in [
            ("qwen3-asr-1.7b", "Qwen3-ASR-1.7B-Q4_K_M.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"),
            ("qwen3-asr-1.7b-q8", "Qwen3-ASR-1.7B-Q8_0.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"),
            ("qwen3-asr-0.6b", "Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"),
        ] {
            let entry = catalog.entry(id).expect(id);
            let files = SpeechFiles { id: id.into(), decoder: entry.artifact_path(dir), files: entry.files.iter().map(|f| dir.join(&f.name)).collect() };
            assert_eq!(files.decoder, dir.join(decoder), "{id}");
            assert!(files.files.contains(&files.decoder), "{id}: the decoder is one of the entry's files");
            assert_eq!(encoder_of(&files), Some(&dir.join(encoder)), "{id}");
        }
    }

    #[test]
    fn a_cpu_without_avx2_is_unavailable_with_the_reason_the_guide_shows() {
        let e = CpuExecutor::with_features(Features::none());
        assert_eq!(e.name(), "cpu");
        assert_eq!(e.availability(), Err(NO_AVX2_REASON.to_owned()));
        let r = e.load(&files(&std::env::temp_dir(), true), LoadSettings::default());
        assert_eq!(r, Err(SpeechError(NO_AVX2_REASON.to_owned())));
        assert!(!e.is_loaded());
        assert!(e.open(SessionOpen::default()).is_err());
    }

    #[test]
    fn a_load_needs_the_encoder_beside_the_decoder_and_names_a_missing_file() {
        let e = CpuExecutor::new();
        if e.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the load is not reached");
            return;
        }
        let dir = std::env::temp_dir().join(format!("ignis-speech-cpu-{}", std::process::id()));
        let r = e.load(&files(&dir, false), LoadSettings::default());
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("no audio encoder")), "{r:?}");
        // A file beside the decoder that is not the encoder's fixed name is not taken for it.
        let mut odd = files(&dir, false);
        odd.files.push(dir.join("README.md"));
        let r = e.load(&odd, LoadSettings::default());
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("no audio encoder")), "{r:?}");
        let r = e.load(&files(&dir, true), LoadSettings { threads: Some(2), pinned_weights: false });
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("Qwen3-ASR-1.7B-Q4_K_M.gguf")), "{r:?}");
        assert!(!e.is_loaded());
    }

    use super::synthetic;

    /// The executor loaded with the crate's synthetic pair (its encoder under the fixed `mmproj-`
    /// name) on 2 threads: what the diarizer computes on.
    fn loaded_synthetic(tag: &str) -> CpuExecutor {
        let e = CpuExecutor::new();
        let pair = synthetic::pair(tag);
        let encoder = pair.encoder.with_file_name("mmproj-synthetic.gguf");
        std::fs::rename(&pair.encoder, &encoder).unwrap();
        let files = SpeechFiles { id: "synthetic".into(), decoder: pair.decoder.clone(), files: vec![pair.decoder.clone(), encoder] };
        e.load(&files, LoadSettings { threads: Some(2), pinned_weights: false }).unwrap();
        e
    }

    /// Spec speech/08: the diarizer is unavailable, saying why, on a host the speech model cannot
    /// run on or without its weights; with them it says nothing.
    #[test]
    fn the_diarizer_is_unavailable_without_avx2_or_its_weights() {
        let weights = std::env::temp_dir().join(format!("ignis-no-diarizer-{}.gguf", std::process::id()));
        let none = CpuDiarizer::new(weights.clone(), false, Arc::new(CpuExecutor::with_features(Features::none())));
        assert_eq!(none.availability(), Err(NO_AVX2_REASON.to_owned()));
        let speech = CpuExecutor::new();
        if speech.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the weights are not looked for");
            return;
        }
        let missing = CpuDiarizer::new(weights.clone(), false, Arc::new(speech)).availability().unwrap_err();
        assert!(missing.contains("no diarizer weights") && missing.contains(&weights.display().to_string()), "{missing}");
        std::fs::write(&weights, b"gguf").unwrap();
        assert_eq!(CpuDiarizer::new(weights.clone(), false, Arc::new(CpuExecutor::new())).availability(), Ok(()));
        let _ = std::fs::remove_file(&weights);
    }

    /// The GPU profile's serialized pass runs every `--ignored` test it does not skip; the
    /// diarizer's leg is the CPU's, on the study disk, so `scripts/gpu-profile.ps1` skips it by
    /// name.
    #[test]
    fn the_gpu_profile_skips_the_diarizers_leg() {
        let script = include_str!("../../../../scripts/gpu-profile.ps1");
        let listed = script.split("$NotGpuLegs = @(").nth(1).and_then(|rest| rest.split(')').next()).expect("gpu-profile.ps1 lists $NotGpuLegs");
        assert!(listed.contains("\"the_diarizer_adapter_steps_the_crates_stream_on_its_own_pool\""), "{listed}");
    }

    /// Spec speech/08 on the real weights (the study disk, `IGNIS_SPEECH_STUDY`, CPU only): the
    /// adapter's stream is the crate's, its steps computed on a pool of its own, of
    /// `diarizer.threads`, with the speech model's ISA: step by step its frames are those of the
    /// crate's stream run alone on a pool of the same threads. Its tail is that stream's copy
    /// finished, and leaves the stream as it was; the constants left `auto` are the checkpoint's;
    /// the speech model unloaded stops nothing, and the diarizer's unload gives its pool back.
    #[test]
    #[ignore = "reads the diarizer's weights from the study disk (CPU only)"]
    fn the_diarizer_adapter_steps_the_crates_stream_on_its_own_pool() {
        let study = PathBuf::from(std::env::var("IGNIS_SPEECH_STUDY").unwrap_or_else(|_| "H:/speech-study".into()));
        let weights = study.join("models").join("diarizer").join("Nemotron-3-Diarization-F32.gguf");
        assert!(weights.is_file(), "the diarizer's weights are not at {} (set IGNIS_SPEECH_STUDY)", weights.display());
        let speech = Arc::new(loaded_synthetic("diarizer-adapter"));
        let adapter = CpuDiarizer::new(weights.clone(), false, Arc::clone(&speech));
        assert_eq!(adapter.availability(), Ok(()));
        let open = StreamOpen { geometry: ignis_speech::diarizer::Geometry::LIVE, constants: CacheConstants::default(), threads: Some(2) };
        let mut stream = adapter.open(open).unwrap();
        assert!(adapter.is_loaded());
        assert_eq!(adapter.threads(), Some(2), "diarizer.threads");
        let d = Diarizer::load(&weights, false).unwrap();
        let isa = ignis_speech::cpu::select(Features::detect()).unwrap();
        let pool = ignis_speech::pool::Pool::new(2);
        let mut alone = d.open(ignis_speech::diarizer::Geometry::LIVE, d.weights().constants);
        let audio = synthetic::audio(12.0);
        let mut steps = 0;
        for piece in audio.chunks(16_000) {
            stream.push(piece);
            alone.push(piece);
            while let Some(frames) = stream.step().unwrap() {
                let reference = alone.step(&d, &pool, isa).expect("the crate's stream steps too");
                assert_eq!((frames.first, &frames.probs), (reference.first, &reference.frames), "step {steps}");
                steps += 1;
            }
            assert_eq!(stream.samples_needed(), alone.samples_needed());
        }
        assert_eq!(steps, 5, "12 s at the live geometry: five steps of 2 s");
        let tail = stream.tail().unwrap();
        let reference = alone.clone().finish(&d, &pool, isa).expect("frames past the last step");
        assert_eq!((tail.first, &tail.probs), (reference.first, &reference.frames));
        stream.push(&audio[..48_000]);
        alone.push(&audio[..48_000]);
        let after = stream.step().unwrap().expect("the tail left the stream able to step");
        assert_eq!(after.probs, alone.step(&d, &pool, isa).unwrap().frames, "the tail left the stream as it was");
        // The speech model unloaded, the stream steps on.
        speech.unload();
        stream.push(&audio[..32_000]);
        alone.push(&audio[..32_000]);
        let after = stream.step().unwrap().expect("a step on the diarizer's own pool");
        assert_eq!(after.probs, alone.step(&d, &pool, isa).unwrap().frames);
        adapter.unload();
        assert!(!adapter.is_loaded() && adapter.threads().is_none());
    }

    /// The speaker cache's constants: each override, the rest the checkpoint's.
    #[test]
    fn an_override_replaces_its_constant_and_auto_keeps_the_checkpoints() {
        let own = Constants {
            prediction_score_threshold: 0.25,
            latest_frames_score_boost: 0.05,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_positive_scores_rate: 0.5,
            speaker_cache_silence_frames_per_speaker: 1,
        };
        assert_eq!(constants(own, CacheConstants::default()), own);
        let over = CacheConstants { weak_boost_rate: Some(2.0), speaker_cache_silence_frames_per_speaker: Some(3), ..CacheConstants::default() };
        assert_eq!(constants(own, over), Constants { weak_boost_rate: 2.0, speaker_cache_silence_frames_per_speaker: 3, ..own });
    }

    /// Q79: the adapter hands the server's stop to the crate's session. On the crate's synthetic
    /// pair (its encoder under the fixed `mmproj-` name), a refresh whose stop says so answers
    /// `None` and leaves the session as it was: its commit is the one with no refresh tried.
    #[test]
    fn a_refresh_the_server_stops_answers_none_and_changes_nothing() {
        let e = CpuExecutor::new();
        if e.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the model is not loaded");
            return;
        }
        let pair = synthetic::pair("adapter-stop");
        let encoder = pair.encoder.with_file_name("mmproj-synthetic.gguf");
        std::fs::rename(&pair.encoder, &encoder).unwrap();
        let files = SpeechFiles { id: "synthetic".into(), decoder: pair.decoder.clone(), files: vec![pair.decoder.clone(), encoder] };
        e.load(&files, LoadSettings { threads: Some(2), pinned_weights: false }).unwrap();
        let open = SessionOpen { context_max_tokens: 512, rollback_tokens: 10, unfixed_partials: 2, ..SessionOpen::default() };
        let session = || {
            let mut s = e.open(open.clone()).unwrap();
            s.push(&synthetic::audio(9.3)).unwrap();
            s
        };
        let untried = commit(session()).unwrap();
        let mut s = session();
        assert_eq!(s.partial(&|| true).unwrap(), None);
        assert_eq!(commit(s).unwrap(), untried);
        assert!(session().partial(&|| false).unwrap().is_some(), "a refresh nothing stops answers");
    }

    /// Spec speech/01 AC 19 through the seam: the adapter's commit is the crate's, a unit a step
    /// (the last window with the audio's last frames, then the decode's prefill, then the
    /// decode), and a spent session takes no more audio and no more steps.
    #[test]
    fn the_adapters_commit_is_a_unit_a_step_and_then_spent() {
        let e = CpuExecutor::new();
        if e.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the model is not loaded");
            return;
        }
        let pair = synthetic::pair("adapter-steps");
        let encoder = pair.encoder.with_file_name("mmproj-synthetic.gguf");
        std::fs::rename(&pair.encoder, &encoder).unwrap();
        let files = SpeechFiles { id: "synthetic".into(), decoder: pair.decoder.clone(), files: vec![pair.decoder.clone(), encoder] };
        e.load(&files, LoadSettings { threads: Some(2), pinned_weights: false }).unwrap();
        let mut s = e.open(SessionOpen { context_max_tokens: 512, ..SessionOpen::default() }).unwrap();
        s.push(&synthetic::audio(9.3)).unwrap();
        assert_eq!(s.commit_step().unwrap(), None, "the last window");
        assert_eq!(s.commit_step().unwrap(), None, "the decode's prefill");
        // The synthetic decoder's one token, then EOS: one decode step ends it.
        let t = s.commit_step().unwrap().expect("the decode ends");
        assert_eq!(t.tokens.len(), 1);
        assert!(s.commit_step().is_err());
        assert!(s.push(&[0.0; 160]).is_err());
    }
}
