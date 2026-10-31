//! One transcription, incrementally (spec speech/01 §4, Q37): open with a
//! context, push 16 kHz samples, ask for partials, commit.
//!
//! 1. `open` prefills the prompt's head (system turn, user turn, `<|audio_start|>`): the
//!    prefix cache's one entry, so a transcription with the previous one's context
//!    prefills none of it (Q50).
//! 2. `push` computes each mel frame once, encodes each 8 s window once as soon as it is
//!    complete, and prefills its tokens: the audio prefix.
//! 3. `partial` encodes the incomplete window, appends it, the prompt's tail and the stable
//!    text, decodes greedily, keeps the last `rollback_tokens` of what it decoded as the
//!    provisional tail (the rest joins the stable text, never revised), and rolls the cache
//!    back to the audio prefix. The first `unfixed_partials` fix nothing, as the official
//!    streaming's first chunks do. `partial_unless` is a partial its caller can stop (Q79):
//!    it checks before each step of its work and before each token its decode computes, and
//!    a partial stopped leaves the session as it was.
//! 4. `commit` encodes the last window and decodes from the stable text to EOS, so the
//!    transcript begins with every stable text reported (Q43). [`Commit`] is the same work a
//!    unit at a time (Q44): a window, a window's worth of the decode's prefill (the prompt's
//!    tail and the stable text), or 16 decode steps per [`Commit::step`], so the server's
//!    worker runs other sessions' units between two and a long clip's decode never holds it
//!    for its whole length.
//!
//! 5. `cut` ends a transcription early (spec speech/08): the audio from a sample on and the stable
//!    text go, the windows past the cut leave the audio prefix and the one holding it is encoded
//!    again from its audio before the cut, so the commit decodes that audio afresh. A cut reaches
//!    into the windows already encoded only on a session that keeps its audio (`keep_audio`).
//!
//! The KV cache holds the head, the longest clip's audio and the tail, then a text room of
//! half as many tokens as the audio's (speech gives about a third) and 64 more. The stable
//! text never outgrows that room, so a commit always has the room its prefill needs, and a
//! decode that reaches the cache's end stops there: no clip within `max_clip_seconds` fails
//! for want of cache.
//!
//! The mel floor (the maximum − 8) is taken over the frames computed so far: a stream cannot
//! know a clip's maximum before its end. A whole clip pushed at once has every frame but its
//! last one or two computed before its first window is encoded.

use std::sync::Arc;

use crate::decoder::{self, argmax, Cache, Logits};
use crate::encoder::encode_window;
use crate::executor::{Core, Loaded, PrefixEntry, SessionTuning};
use crate::mel::{self, audio_tokens, frames_of, normalise, CHUNK_FRAMES, HOP, MIN_SAMPLES, N_FFT, N_MELS, WINDOW_CHUNKS};
use crate::text::{copies_context, parse, Prompt};

const WINDOW_FRAMES: usize = CHUNK_FRAMES * WINDOW_CHUNKS;
const WINDOW_TOKENS: usize = 104;
/// Decode steps per [`Commit::step`]: about a window's prefill, so a long decode interleaves
/// with other sessions' windows (Q44).
pub const DECODE_UNIT: usize = 16;

/// A partial result: the stable text (never revised) and the provisional tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partial {
    pub stable: String,
    pub provisional: String,
    pub language: Option<String>,
}

/// A completed transcription.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The model's language label (Q51), or the forced one.
    pub language: Option<String>,
    pub audio_seconds: f64,
    pub audio_tokens: usize,
    /// The decoded token ids, the language header included (spec 02's `verbose_json` segment).
    pub tokens: Vec<u32>,
}

/// What a session did, for acceptance 17-18.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Context tokens prefilled at open (0 when the prefix cache held the head).
    pub context_prefilled: usize,
    /// Windows encoded into the audio prefix: each window once.
    pub windows: usize,
    /// Audio tokens prefilled into the audio prefix: each once.
    pub audio_prefilled: usize,
    pub partials: usize,
    /// The stable text's tokens: what a commit's prefill carries after the prompt's tail.
    pub stable_tokens: usize,
}

#[derive(Debug)]
pub enum SessionError {
    /// The clip passed `max_clip_seconds`.
    TooLong { seconds: u32 },
    /// A commit was asked for another step after it ended.
    Committed,
    /// A cut reaches back into audio the session no longer keeps: it was opened without
    /// [`Session::keep_audio`].
    CutTooFar { at: usize },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::TooLong { seconds } => write!(f, "the audio is longer than {seconds} s"),
            SessionError::Committed => write!(f, "the transcription was already committed"),
            SessionError::CutTooFar { at } => write!(f, "a cut at sample {at} reaches audio the session no longer keeps"),
        }
    }
}

impl std::error::Error for SessionError {}

pub struct Session {
    loaded: Arc<Loaded>,
    core: Arc<Core>,
    prompt: Prompt,
    language: Option<String>,
    /// Pushed samples from absolute index `sample_base` (a multiple of the hop).
    samples: Vec<f32>,
    sample_base: usize,
    total_samples: usize,
    /// log10 mel rows from absolute frame `frame_base`; `next_frame` is the first not computed.
    frames: Vec<f32>,
    frame_base: usize,
    next_frame: usize,
    max_log: f32,
    windows: usize,
    cache: Cache,
    /// Cache positions after the head and every encoded window.
    audio_prefix: usize,
    /// The cache's length after the head, then after each encoded window: where a cut takes the
    /// audio prefix back to.
    window_ends: Vec<usize>,
    /// The largest log-mel value of each encoded window's frames: the floor a cut starts again
    /// from.
    window_max: Vec<f32>,
    /// Every pushed sample is kept, so a cut can reach any of them.
    keep_audio: bool,
    /// The positions the cache keeps for text after the longest clip's audio and the tail:
    /// the stable text stays shorter, so a commit's prefill always fits.
    text_room: usize,
    stable: Vec<u32>,
    since_partial: usize,
    stats: Stats,
    /// `speech.rollback_tokens` and `speech.unfixed_partials` as they were when this session opened.
    tuning: SessionTuning,
}

impl Session {
    pub(crate) fn open(loaded: Arc<Loaded>, core: Arc<Core>, context: &str, language: Option<&str>, tuning: SessionTuning) -> Session {
        let s = &core.settings;
        let text = &loaded.model.text;
        let prompt = Prompt::new(text, context, tuning.context_max_tokens, language);
        let dec = &loaded.model.decoder;
        let audio_cap = audio_tokens(frames_of(s.max_clip_seconds as usize * mel::SAMPLE_RATE));
        // Sized for the context cap, the longest clip's audio tokens and the decode (§2): room
        // for half as many text tokens as audio tokens, where speech gives about a third. It
        // starts at the head, a window and the tail, and grows as the transcription does.
        let text_room = audio_cap / 2 + 64;
        let max = prompt.head.len() + audio_cap + prompt.tail.len() + text_room;
        let first = prompt.head.len() + 2 * WINDOW_TOKENS + prompt.tail.len() + 64;
        let mut cache = Cache::growing(dec, first, max);
        let mut stats = Stats::default();
        {
            let mut entry = core.prefix.lock().unwrap_or_else(|e| e.into_inner());
            match entry.as_ref() {
                Some(e) if e.head == prompt.head => cache.copy_prefix(&e.cache, e.head.len()),
                _ => {
                    let x = decoder::embed(dec, &prompt.head);
                    decoder::forward(dec, &loaded.pool, loaded.isa, &mut cache, x, Logits::None, None);
                    stats.context_prefilled = prompt.context_tokens;
                    let mut keep = Cache::new(dec, prompt.head.len());
                    keep.copy_prefix(&cache, prompt.head.len());
                    *entry = Some(PrefixEntry { head: prompt.head.clone(), cache: keep });
                }
            }
        }
        let audio_prefix = cache.len;
        Session {
            loaded,
            core,
            prompt,
            language: language.map(str::to_string),
            samples: Vec::new(),
            sample_base: 0,
            total_samples: 0,
            frames: Vec::new(),
            frame_base: 0,
            next_frame: 0,
            max_log: f32::NEG_INFINITY,
            windows: 0,
            cache,
            audio_prefix,
            window_ends: vec![audio_prefix],
            window_max: Vec::new(),
            keep_audio: false,
            text_room,
            stable: Vec::new(),
            since_partial: 0,
            stats,
            tuning,
        }
    }

    /// The tuning this session opened with.
    pub fn tuning(&self) -> SessionTuning {
        self.tuning
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Audio pushed so far, in seconds.
    pub fn seconds(&self) -> f64 {
        self.total_samples as f64 / mel::SAMPLE_RATE as f64
    }

    /// True once `partial_interval_ms` of audio arrived since the last partial.
    pub fn partial_due(&self) -> bool {
        self.since_partial * 1000 >= self.core.settings.partial_interval_ms as usize * mel::SAMPLE_RATE
    }

    /// Appends mono 16 kHz samples; encodes and prefills each window that completes.
    pub fn push(&mut self, samples: &[f32]) -> Result<(), SessionError> {
        let max = self.core.settings.max_clip_seconds;
        if self.total_samples + samples.len() > max as usize * mel::SAMPLE_RATE {
            return Err(SessionError::TooLong { seconds: max });
        }
        self.samples.extend_from_slice(samples);
        self.total_samples += samples.len();
        self.since_partial += samples.len();
        // Frame t is final once samples up to 160·t + 200 have arrived.
        // Frame 0 also reflects at the start: it needs sample 200.
        let ready = if self.total_samples > N_FFT / 2 { (self.total_samples - N_FFT / 2) / HOP + 1 } else { 0 };
        if ready > self.next_frame {
            self.compute_frames(ready - self.next_frame);
        }
        while self.next_frame >= (self.windows + 1) * WINDOW_FRAMES {
            let mel = self.window_mel(self.windows * WINDOW_FRAMES, WINDOW_FRAMES, &[]);
            self.prefill_window(&mel, WINDOW_FRAMES);
        }
        Ok(())
    }

    fn compute_frames(&mut self, n: usize) {
        let l = &self.loaded;
        let first = self.next_frame - self.sample_base / HOP;
        let start = self.frames.len();
        l.mel.log_frames(&l.pool, l.isa, &self.samples, first, n, false, &mut self.frames);
        self.max_log = self.frames[start..].iter().fold(self.max_log, |m, &v| m.max(v));
        self.next_frame += n;
    }

    /// Normalised mel rows of `n` frames from absolute frame `first`: computed frames, then `extra`.
    fn window_mel(&self, first: usize, n: usize, extra: &[f32]) -> Vec<f32> {
        let max = extra.iter().fold(self.max_log, |m, &v| m.max(v));
        let have = (self.next_frame - first).min(n);
        let from = (first - self.frame_base) * N_MELS;
        let mut out: Vec<f32> = self.frames[from..from + have * N_MELS].iter().map(|&v| normalise(v, max)).collect();
        out.extend(extra.iter().take((n - have) * N_MELS).map(|&v| normalise(v, max)));
        out
    }

    /// Encodes a window into the audio prefix and drops what no later frame needs.
    fn prefill_window(&mut self, mel: &[f32], frames: usize) {
        let from = (self.windows * WINDOW_FRAMES - self.frame_base) * N_MELS;
        let top = self.frames[from..from + frames * N_MELS].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        self.window_max.push(top);
        let l = &self.loaded;
        let x = encode_window(&l.model.encoder, &l.pool, l.isa, mel, frames, None);
        let n = x.len() / l.model.decoder.hidden;
        decoder::forward(&l.model.decoder, &l.pool, l.isa, &mut self.cache, x, Logits::None, None);
        self.audio_prefix = self.cache.len;
        self.window_ends.push(self.audio_prefix);
        self.windows += 1;
        self.stats.windows += 1;
        self.stats.audio_prefilled += n;
        let keep_frame = self.windows * WINDOW_FRAMES;
        if keep_frame > self.frame_base && keep_frame <= self.next_frame {
            self.frames.drain(..(keep_frame - self.frame_base) * N_MELS);
            self.frame_base = keep_frame;
        }
        // Keep from the first sample a frame not yet computed needs (frames past the clip
        // start never reflect there), at a multiple of the hop; short clips keep everything.
        let need = (self.next_frame * HOP).saturating_sub(2 * HOP);
        if !self.keep_audio && self.total_samples > MIN_SAMPLES && need > self.sample_base {
            self.samples.drain(..need - self.sample_base);
            self.sample_base = need;
        }
    }

    /// The frames from `next_frame` to the end of the audio so far, as if the clip ended now:
    /// reflect padding at the end, zero padding to half a second.
    fn tail_frames(&self) -> (Vec<f32>, usize) {
        let mut x = self.samples.clone();
        if self.total_samples < MIN_SAMPLES {
            x.resize(MIN_SAMPLES - self.sample_base, 0.0);
        }
        let total = frames_of(self.total_samples);
        let n = total.saturating_sub(self.next_frame);
        let mut out = Vec::new();
        let l = &self.loaded;
        l.mel.log_frames(&l.pool, l.isa, &x, self.next_frame - self.sample_base / HOP, n, true, &mut out);
        (out, total)
    }

    /// The decode's prefill after the audio: the prompt's tail and the stable text.
    fn decode_prompt(&self) -> Vec<u32> {
        [&self.prompt.tail[..], &self.stable[..]].concat()
    }

    /// Prefills `ids`, a piece of the decode's prompt, and answers the next token's logits
    /// when it is the last piece. The text room holds the whole prompt (the stable text never
    /// outgrows it), so a clip within `max_clip_seconds` always has the positions.
    fn prefill_text(&mut self, ids: &[u32], last: bool) -> Option<Vec<f32>> {
        let loaded = self.loaded.clone();
        let dec = &loaded.model.decoder;
        let fits = self.cache.reserve(self.cache.len + ids.len() + 1);
        debug_assert!(fits, "the stable text outgrew the text room");
        let want = if last { Logits::Last } else { Logits::None };
        let logits = decoder::forward(dec, &loaded.pool, loaded.isa, &mut self.cache, decoder::embed(dec, ids), want, None);
        last.then_some(logits)
    }

    /// The whole decode's prefill in one call: a partial's.
    fn begin_decode(&mut self) -> Decode {
        let ids = self.decode_prompt();
        Decode { out: Vec::new(), logits: self.prefill_text(&ids, true).expect("the last piece answers logits") }
    }

    /// Up to `steps` greedy steps of `d`: `Some(true)` once it ended, at EOS or at the cache's
    /// end (§2), `Some(false)` with steps left to take, `None` once `stop` says so, which it
    /// is asked before every token's forward pass (Q79).
    fn decode_steps(&mut self, d: &mut Decode, steps: usize, stop: &dyn Fn() -> bool) -> Option<bool> {
        let loaded = self.loaded.clone();
        let dec = &loaded.model.decoder;
        for _ in 0..steps {
            let t = argmax(&d.logits);
            if loaded.model.text.eos.contains(&t) {
                return Some(true);
            }
            d.out.push(t);
            if !self.cache.reserve(self.cache.len + 1) {
                return Some(true);
            }
            if stop() {
                return None;
            }
            d.logits = decoder::forward(dec, &loaded.pool, loaded.isa, &mut self.cache, decoder::embed(dec, &[t]), Logits::Last, None);
        }
        Some(false)
    }

    /// Greedy decoding after the audio, to EOS or the cache's end, in one call: a partial's.
    /// `None` once `stop` says so, before the prefill, after it or before any token (Q79).
    fn decode_after_audio(&mut self, stop: &dyn Fn() -> bool) -> Option<Vec<u32>> {
        if stop() {
            return None;
        }
        let mut d = self.begin_decode();
        if stop() {
            return None;
        }
        self.decode_steps(&mut d, usize::MAX, stop)?;
        Some(d.out)
    }

    /// The transcript text of output ids: after the `<asr_text>` header, or all of it under a
    /// forced language. Ids that end inside the header give no text yet.
    fn body(&self, ids: &[u32]) -> (String, Option<String>) {
        let text = &self.loaded.model.text;
        if self.language.is_none() && !ids.contains(&text.asr_text) {
            return (String::new(), None);
        }
        let p = parse(&text.decode(ids), self.language.as_deref());
        if copies_context(&self.prompt.context, &p.text, self.tuning.context_copy_words, self.tuning.context_copy_edits) {
            return (String::new(), None);
        }
        (p.text, p.language)
    }

    /// A partial's incomplete window: encoded and prefilled after the audio prefix. `None`
    /// when `stop` says so between the encode and the prefill (Q79).
    fn prefill_incomplete(&mut self, stop: &dyn Fn() -> bool) -> Option<()> {
        let (tail, total) = self.tail_frames();
        let first = self.windows * WINDOW_FRAMES;
        if total > first {
            let mel = self.window_mel(first, total - first, &tail);
            let l = &self.loaded;
            let x = encode_window(&l.model.encoder, &l.pool, l.isa, &mel, total - first, None);
            if stop() {
                return None;
            }
            if self.cache.reserve(self.cache.len + x.len() / l.model.decoder.hidden) {
                decoder::forward(&l.model.decoder, &l.pool, l.isa, &mut self.cache, x, Logits::None, None);
            }
        }
        Some(())
    }

    /// Re-encodes the incomplete window, decodes, and moves all but the last `rollback_tokens`
    /// of what it decoded into the stable text. The cache is rolled back to the audio prefix.
    pub fn partial(&mut self) -> Result<Partial, SessionError> {
        Ok(self.partial_unless(&|| false)?.expect("a partial nothing stops ends"))
    }

    /// [`partial`](Self::partial), stopped once `stop` says so (Q79): it asks before the
    /// encode, before the incomplete window's prefill, before the decode's prefill and after
    /// it, and where the decode yields the unit of compute. A partial stopped returns `None`
    /// and leaves the session as it was before the call: the cache rolled back to the audio
    /// prefix, the stable text and the partials counted unchanged, so a commit next decodes
    /// from the stable text already reported.
    pub fn partial_unless(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SessionError> {
        if self.total_samples == 0 {
            self.since_partial = 0;
            self.stats.partials += 1;
            return Ok(Some(Partial { stable: String::new(), provisional: String::new(), language: self.language.clone() }));
        }
        if stop() {
            return Ok(None);
        }
        let decoded = self.prefill_incomplete(stop).and_then(|()| self.decode_after_audio(stop));
        self.cache.truncate(self.audio_prefix);
        let Some(decoded) = decoded else { return Ok(None) };
        self.since_partial = 0;
        self.stats.partials += 1;
        let r = self.tuning.rollback_tokens;
        let mut cut = if self.stats.partials <= self.tuning.unfixed_partials { 0 } else { decoded.len().saturating_sub(r) };
        // The stable text stays inside the text room, with a position for the next token.
        cut = cut.min(self.text_room.saturating_sub(self.stable.len() + 1));
        // The stable text ends on a whole word: the provisional tail starts where a word does
        // (a token that decodes with a leading space), so an identifier the audio has not
        // finished (`model_switch.rs`) stays provisional whole. A whole word also ends on a
        // whole character, which byte-level tokens may split.
        let text = &self.loaded.model.text;
        while cut > 0 && cut < decoded.len() && !text.decode(&decoded[cut..cut + 1]).starts_with(char::is_whitespace) {
            cut -= 1;
        }
        while cut > 0 && text.decode(&[&self.stable[..], &decoded[..cut]].concat()).ends_with('\u{FFFD}') {
            cut -= 1;
        }
        self.stable.extend_from_slice(&decoded[..cut]);
        self.stats.stable_tokens = self.stable.len();
        let (stable, language) = self.body(&self.stable);
        let all: Vec<u32> = [&self.stable[..], &decoded[cut..]].concat();
        let (full, _) = self.body(&all);
        let provisional = full.strip_prefix(stable.as_str()).unwrap_or(&full).to_string();
        Ok(Some(Partial { stable, provisional, language: language.or_else(|| self.language.clone()) }))
    }

    /// Keeps every sample pushed from now on, so a [`cut`](Self::cut) can reach any of them, the
    /// windows already encoded included: 4 bytes a sample, 64 KB a second.
    pub fn keep_audio(&mut self) {
        self.keep_audio = true;
    }

    /// Ends the transcription at sample `at` (16 kHz, from the first pushed): the audio from it
    /// on goes, and so does the stable text, which may hold words past it, so a commit decodes
    /// the audio before `at` afresh. The windows past the cut leave the audio prefix; the one
    /// holding it is encoded again, from its audio before the cut, when it completes or at the
    /// commit. The mel floor starts again from the windows kept and the frames recomputed. A
    /// cut at or past the audio pushed drops only the stable text.
    pub fn cut(&mut self, at: usize) -> Result<(), SessionError> {
        if at < self.total_samples {
            // The frames final with the audio before `at` alone: frame t reads up to 160·t + 200.
            let ready = if at > N_FFT / 2 { (at - N_FFT / 2) / HOP + 1 } else { 0 };
            let windows = (ready / WINDOW_FRAMES).min(self.windows);
            if windows < self.windows {
                // The windows from the cut's on leave the audio prefix, and their frames are
                // computed again: the first reads from two hops before its own (the start
                // reflects).
                let first = windows * WINDOW_FRAMES;
                if self.sample_base > 0 && self.sample_base + 2 * HOP > first * HOP {
                    return Err(SessionError::CutTooFar { at });
                }
                self.cache.truncate(self.window_ends[windows]);
                self.audio_prefix = self.cache.len;
                self.window_ends.truncate(windows + 1);
                self.window_max.truncate(windows);
                self.windows = windows;
                self.frames.clear();
                self.frame_base = first;
                self.next_frame = first;
            } else {
                // The frames computed before `ready` stand; the ones after it read past the cut.
                let keep = ready.min(self.next_frame);
                self.frames.truncate((keep - self.frame_base) * N_MELS);
                self.next_frame = keep;
            }
            self.samples.truncate(at - self.sample_base);
            self.total_samples = at;
            let kept = self.window_max.iter().chain(&self.frames);
            self.max_log = kept.fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            if ready > self.next_frame {
                self.compute_frames(ready - self.next_frame);
            }
        }
        self.stable.clear();
        self.stats.stable_tokens = 0;
        Ok(())
    }

    /// Encodes the rest of the audio and decodes to EOS from the stable text: a [`Commit`]
    /// stepped to its end.
    pub fn commit(self) -> Result<Transcript, SessionError> {
        let mut commit = self.into_commit();
        loop {
            if let Some(transcript) = commit.step()? {
                return Ok(transcript);
            }
        }
    }

    /// The commit, to be taken a unit at a time ([`Commit::step`]). Nothing is computed here.
    pub fn into_commit(self) -> Commit {
        Commit { session: self, total: None, prompt: None, decode: None, done: false }
    }
}

/// A greedy decode in progress: the tokens so far and the logits of the next.
struct Decode {
    out: Vec<u32>,
    logits: Vec<f32>,
}

/// A commit in progress (§4, Q44): the session's last windows, the decode's prefill, then the
/// decode, a unit of compute per [`step`](Commit::step).
pub struct Commit {
    session: Session,
    /// The clip's frames, once the first step folded in the last ones.
    total: Option<usize>,
    /// The decode's prompt, and how much of it is prefilled.
    prompt: Option<(Vec<u32>, usize)>,
    decode: Option<Decode>,
    done: bool,
}

impl Commit {
    /// One unit of the commit: the audio's last frames and a window of it, a window, a
    /// window's worth of tokens (104) of the decode's prefill, or [`DECODE_UNIT`] decode
    /// steps. `Some` with the transcript once the decode ends; a step after that is
    /// [`SessionError::Committed`].
    pub fn step(&mut self) -> Result<Option<Transcript>, SessionError> {
        if self.done {
            return Err(SessionError::Committed);
        }
        let s = &mut self.session;
        let total = match self.total {
            Some(total) => total,
            None => {
                let (tail, total) = s.tail_frames();
                // The tail frames are final now: fold them into the computed ones.
                s.max_log = tail.iter().fold(s.max_log, |m, &v| m.max(v));
                s.frames.extend_from_slice(&tail);
                s.next_frame = total;
                *self.total.insert(total)
            }
        };
        let Some(d) = self.decode.as_mut() else {
            if s.windows * WINDOW_FRAMES < total {
                let first = s.windows * WINDOW_FRAMES;
                let n = (total - first).min(WINDOW_FRAMES);
                let mel = s.window_mel(first, n, &[]);
                s.prefill_window(&mel, n);
                return Ok(None);
            }
            let (ids, done) = self.prompt.get_or_insert_with(|| (s.decode_prompt(), 0));
            let to = (*done + WINDOW_TOKENS).min(ids.len());
            let piece = ids[*done..to].to_vec();
            let last = to == ids.len();
            *done = to;
            if let Some(logits) = s.prefill_text(&piece, last) {
                self.decode = Some(Decode { out: Vec::new(), logits });
            }
            return Ok(None);
        };
        if !s.decode_steps(d, DECODE_UNIT, &|| false).expect("nothing stops a commit") {
            return Ok(None);
        }
        self.done = true;
        let all: Vec<u32> = [&s.stable[..], &d.out[..]].concat();
        let text = &s.loaded.model.text;
        let p = parse(&text.decode(&all), s.language.as_deref());
        // A copy of the context is no speech: an empty transcript, labelled only by a forced language.
        let (said, language) = if copies_context(&s.prompt.context, &p.text, s.tuning.context_copy_words, s.tuning.context_copy_edits) { (String::new(), s.language.clone()) } else { (p.text, p.language) };
        Ok(Some(Transcript { text: said, language, audio_seconds: s.seconds(), audio_tokens: audio_tokens(total), tokens: all }))
    }
}
