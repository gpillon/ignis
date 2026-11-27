//! A transcription request's audio, decoded in memory to what the speech
//! executor takes: 16 kHz mono f32 (spec speech/02 §`file`, Q12, Q13, Q42).
//!
//! The upload is never written to disk. WAV, FLAC, MP3 and AAC (ADTS or in an
//! MP4/M4A) are decoded by symphonia; the Ogg and Matroska/WebM containers
//! are demuxed, and what they carry is decoded when it is FLAC. Opus is
//! not decoded: libopus is a native build the owner accepts only if a
//! verification build passes (spec of GitHub #350, "Native code
//! outside `cuda`"), and until then an Opus clip is refused naming what is
//! accepted. Resampling to 16 kHz is a windowed-sinc filter of our own.

use std::io::Cursor;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// The rate the speech model takes: the executor's front end's.
pub const TARGET_RATE: u32 = ignis_speech::mel::SAMPLE_RATE as u32;

/// What the route says is accepted, in a refusal.
pub const ACCEPTED: &str = "WAV, FLAC, MP3 and AAC (ADTS or MP4/M4A), and the Ogg and Matroska/WebM containers when they carry FLAC";

/// Why a clip was not decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioError {
    /// Not a format the route accepts, or a codec it does not decode.
    Unsupported(String),
    /// Longer than `speech.max_clip_seconds`, the limit it was held to.
    TooLong { limit_seconds: u32 },
    /// Accepted, but it holds no audio, or is damaged beyond what was read.
    Unreadable(String),
}

impl AudioError {
    /// The sentence the `400` carries.
    pub fn message(&self) -> String {
        match self {
            Self::Unsupported(why) => format!("`file` is not a format this server transcribes ({why}); accepted: {ACCEPTED}"),
            Self::TooLong { limit_seconds } => {
                format!("`file` is longer than speech.max_clip_seconds ({limit_seconds} s); nothing is split automatically, send a shorter clip")
            }
            Self::Unreadable(why) => format!("`file` could not be read as audio: {why}"),
        }
    }
}

/// A decoded clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// 16 kHz mono.
    pub samples: Vec<f32>,
}

impl Clip {
    /// The clip's length, in seconds.
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(TARGET_RATE)
    }
}

/// Decode `bytes` to 16 kHz mono, refusing a clip over `max_seconds` as soon
/// as its length is known or reached. Blocking.
pub fn decode(bytes: Vec<u8>, max_seconds: u32) -> Result<Clip, AudioError> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(&Hint::new(), source, FormatOptions::default(), MetadataOptions::default())
        .map_err(|error| AudioError::Unsupported(format!("no container recognised: {error}")))?;
    let (track, params) = format
        .tracks()
        .iter()
        .find_map(|track| track.codec_params.as_ref().and_then(|params| params.audio()).map(|params| (track, params.clone())))
        .ok_or_else(|| AudioError::Unsupported("no audio track whose codec this server decodes (Opus is not decoded)".to_owned()))?;
    let track_id = track.id;
    // A length the container states is checked before a sample is decoded.
    if let (Some(frames), Some(rate)) = (track.num_frames, params.sample_rate) {
        if rate > 0 && frames / u64::from(rate) > u64::from(max_seconds) {
            return Err(AudioError::TooLong { limit_seconds: max_seconds });
        }
    }
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|error| AudioError::Unsupported(format!("the codec is not decoded (Opus is not): {error}")))?;

    let max_samples = u64::from(max_seconds) * u64::from(TARGET_RATE);
    let mut resampler: Option<Resampler> = None;
    let mut samples: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(error) => return Err(AudioError::Unreadable(error.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A damaged packet is skipped, as every player does.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(AudioError::Unreadable(error.to_string())),
        };
        let spec = decoded.spec().clone();
        let channels = spec.channels().count().max(1);
        interleaved.resize(decoded.samples_interleaved(), 0.0);
        decoded.copy_to_slice_interleaved(&mut interleaved);
        mono.clear();
        mono.extend(interleaved.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32));
        let rate = spec.rate();
        if rate == TARGET_RATE {
            samples.extend_from_slice(&mono);
        } else {
            resampler.get_or_insert_with(|| Resampler::new(rate, TARGET_RATE)).process(&mono, &mut samples);
        }
        if samples.len() as u64 > max_samples {
            return Err(AudioError::TooLong { limit_seconds: max_seconds });
        }
    }
    if let Some(resampler) = resampler.as_mut() {
        resampler.finish(&mut samples);
    }
    if samples.len() as u64 > max_samples {
        return Err(AudioError::TooLong { limit_seconds: max_seconds });
    }
    if samples.is_empty() {
        return Err(AudioError::Unreadable("it holds no audio".to_owned()));
    }
    Ok(Clip { samples })
}

/// A streaming windowed-sinc resampler: Hann-windowed, its cutoff just under
/// the lower of the two Nyquist rates, so a downsample does not alias. The
/// kernel is tabulated at 1/512 of an input sample.
pub(crate) struct Resampler {
    /// Input samples per output sample.
    step: f64,
    /// The kernel's half width, in input samples.
    half: usize,
    table: Vec<f32>,
    /// Input not yet consumed, and where the next output sits in it.
    buffer: Vec<f32>,
    position: f64,
}

const KERNEL_RESOLUTION: usize = 512;
const ZERO_CROSSINGS: f64 = 12.0;

impl Resampler {
    pub(crate) fn new(from: u32, to: u32) -> Self {
        let step = f64::from(from) / f64::from(to);
        // The cutoff, as a fraction of the input's Nyquist rate.
        let cutoff = (1.0 / step).min(1.0) * 0.97;
        let half = (ZERO_CROSSINGS / cutoff).ceil() as usize;
        let table = (0..=half * KERNEL_RESOLUTION)
            .map(|i| {
                let x = i as f64 / KERNEL_RESOLUTION as f64;
                let sinc = if x == 0.0 { 1.0 } else { (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * cutoff * x) };
                let window = 0.5 * (1.0 + (std::f64::consts::PI * x / half as f64).cos());
                (cutoff * sinc * window) as f32
            })
            .collect();
        // Pad the front so the first outputs see silence, not nothing.
        Self { step, half, table, buffer: vec![0.0; half], position: half as f64 }
    }

    fn kernel(&self, distance: f64) -> f32 {
        let at = distance.abs() * KERNEL_RESOLUTION as f64;
        let low = at as usize;
        let fraction = (at - low as f64) as f32;
        match (self.table.get(low), self.table.get(low + 1)) {
            (Some(a), Some(b)) => a + (b - a) * fraction,
            _ => 0.0,
        }
    }

    pub(crate) fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buffer.extend_from_slice(input);
        self.emit(out);
    }

    /// Output what the buffered input allows (every output whose kernel lies
    /// wholly inside it), then drop the input no later output reads.
    fn emit(&mut self, out: &mut Vec<f32>) {
        while self.position + self.half as f64 + 1.0 < self.buffer.len() as f64 {
            let centre = self.position.floor() as usize;
            let mut sum = 0.0f32;
            for k in centre + 1 - self.half..=centre + self.half {
                sum += self.buffer[k] * self.kernel(k as f64 - self.position);
            }
            out.push(sum);
            self.position += self.step;
        }
        let keep_from = (self.position.floor() as usize).saturating_sub(self.half);
        if keep_from > 0 {
            self.buffer.drain(..keep_from);
            self.position -= keep_from as f64;
        }
    }

    /// The tail: silence after the last input, as long as the kernel is wide.
    pub(crate) fn finish(&mut self, out: &mut Vec<f32>) {
        let silence = vec![0.0; self.half + 2];
        self.process(&silence, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q42: symphonia is an MPL-2.0 dependency, used unmodified, and `NOTICE`
    /// says so.
    #[test]
    fn the_notice_records_symphonia_and_its_licence() {
        let notice = include_str!("../../../../NOTICE");
        let at = notice.find("symphonia").expect("NOTICE names symphonia");
        let paragraph = &notice[at..at + notice[at..].find("\n\n").unwrap_or(notice.len() - at)];
        assert!(paragraph.contains("Mozilla Public License, Version 2.0") && paragraph.contains("unmodified"), "{paragraph}");
    }

    const WAV: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.wav");
    const WAV_44K_STEREO: &[u8] = include_bytes!("../../tests/fixtures/audio/tone-44k-stereo.wav");
    const FLAC: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.flac");
    const MP3: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.mp3");
    const AAC: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.aac");
    const M4A: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.m4a");
    const OGG_FLAC: &[u8] = include_bytes!("../../tests/fixtures/audio/tone-flac.ogg");
    const MATROSKA: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.mka");
    const OGG_OPUS: &[u8] = include_bytes!("../../tests/fixtures/audio/tone-opus.ogg");
    const WEBM_OPUS: &[u8] = include_bytes!("../../tests/fixtures/audio/tone-opus.webm");

    /// The energy of `samples` at `frequency` against its total: how much of
    /// the clip is the 440 Hz tone the fixtures hold.
    fn tone_share(samples: &[f32], frequency: f64) -> f64 {
        let (mut sin, mut cos, mut total) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &sample) in samples.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * frequency * i as f64 / f64::from(TARGET_RATE);
            sin += f64::from(sample) * phase.sin();
            cos += f64::from(sample) * phase.cos();
            total += f64::from(sample) * f64::from(sample);
        }
        let n = samples.len() as f64;
        (2.0 * (sin * sin + cos * cos) / n) / total.max(f64::MIN_POSITIVE)
    }

    /// Spec speech/02 AC 5: every accepted format comes out as about one
    /// second of 16 kHz mono that is still the 440 Hz tone it went in as,
    /// whatever its rate and channel count.
    #[test]
    fn every_accepted_format_decodes_to_the_tone_at_16_khz_mono() {
        for (name, bytes) in [
            ("wav", WAV),
            ("wav 44.1 kHz stereo", WAV_44K_STEREO),
            ("flac 48 kHz", FLAC),
            ("mp3 stereo", MP3),
            ("aac adts", AAC),
            ("m4a", M4A),
            ("flac in ogg", OGG_FLAC),
            ("flac in matroska", MATROSKA),
        ] {
            let clip = decode(bytes.to_vec(), 1200).unwrap_or_else(|error| panic!("{name}: {error:?}"));
            let seconds = clip.seconds();
            assert!((0.9..=1.15).contains(&seconds), "{name}: {seconds} s");
            // Past the codec's start-up, the middle of the clip is the tone.
            let middle = &clip.samples[clip.samples.len() / 4..clip.samples.len() * 3 / 4];
            let share = tone_share(middle, 440.0);
            assert!(share > 0.9, "{name}: {share:.3} of the energy is at 440 Hz");
            assert!(middle.iter().any(|sample| sample.abs() > 0.05), "{name}: silent");
        }
    }

    /// AC 5: Opus is demuxed but not decoded without libopus, and the refusal
    /// names the accepted set.
    #[test]
    fn opus_is_refused_naming_the_accepted_set() {
        for (name, bytes) in [("ogg", OGG_OPUS), ("webm", WEBM_OPUS)] {
            let error = decode(bytes.to_vec(), 1200).expect_err(name);
            let message = error.message();
            assert!(matches!(error, AudioError::Unsupported(_)), "{name}: {error:?}");
            assert!(message.contains("WAV") && message.contains("FLAC") && message.contains("MP3") && message.contains("AAC"), "{message}");
        }
    }

    #[test]
    fn bytes_that_are_no_audio_are_refused_naming_the_accepted_set() {
        let error = decode(b"this is not audio at all, just some text".repeat(100), 1200).unwrap_err();
        assert!(matches!(error, AudioError::Unsupported(_)), "{error:?}");
        assert!(error.message().contains("accepted:"), "{}", error.message());
        assert!(matches!(decode(Vec::new(), 1200), Err(AudioError::Unsupported(_))));
    }

    /// AC 10: a clip over the limit is refused naming it, from the length the
    /// container states before a sample is decoded, and from the samples
    /// decoded for one that states none.
    #[test]
    fn a_clip_over_the_limit_is_refused_naming_it() {
        let error = decode(WAV.to_vec(), 0).unwrap_err();
        assert_eq!(error, AudioError::TooLong { limit_seconds: 0 });
        assert!(decode(WAV.to_vec(), 1).is_ok(), "one second is not over a one-second limit");
        // A header-only length check does not apply to a raw ADTS stream.
        let error = decode(AAC.to_vec(), 0).unwrap_err();
        assert_eq!(error, AudioError::TooLong { limit_seconds: 0 });
        assert!(error.message().contains("speech.max_clip_seconds (0 s)"), "{}", error.message());
    }

    /// The resampler keeps a tone's pitch and its length through a rate it
    /// has to interpolate (44.1 to 16 kHz), and removes what the lower rate
    /// cannot hold instead of folding it back.
    #[test]
    fn the_resampler_keeps_pitch_and_length_and_does_not_alias() {
        let rate = 44_100u32;
        let tone = |frequency: f64, seconds: f64| -> Vec<f32> {
            (0..(f64::from(rate) * seconds) as usize)
                .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / f64::from(rate)).sin() as f32 * 0.5)
                .collect()
        };
        let mut out = Vec::new();
        let mut resampler = Resampler::new(rate, TARGET_RATE);
        // In uneven pieces, as packets arrive.
        for piece in tone(1000.0, 2.0).chunks(1111) {
            resampler.process(piece, &mut out);
        }
        resampler.finish(&mut out);
        assert!((out.len() as i64 - 32_000).abs() < 40, "{} samples for two seconds", out.len());
        assert!(tone_share(&out[1000..out.len() - 1000], 1000.0) > 0.99);
        // 12 kHz is above the 8 kHz Nyquist rate of the output: it must go.
        let mut out = Vec::new();
        let mut resampler = Resampler::new(rate, TARGET_RATE);
        resampler.process(&tone(12_000.0, 1.0), &mut out);
        resampler.finish(&mut out);
        let energy: f32 = out[500..out.len() - 500].iter().map(|sample| sample * sample).sum::<f32>() / (out.len() - 1000) as f32;
        assert!(energy < 0.001, "an alias is left at {energy}");
    }
}
