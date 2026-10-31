//! A Realtime transcription session's configuration (spec speech/03, §Client
//! events): what `session.update` sets, every field honoured, inert or
//! refused as the spec's table says, and the effective configuration that
//! `session.created` and `session.updated` carry.
//!
//! An update names only what it changes: a field it leaves out keeps its
//! value, `null` puts it back to its default. One refused field refuses the
//! whole update, and the session keeps the configuration it had.

use ignis_speech::vad::TurnSettings;
use super::super::audio::TARGET_RATE;
use serde_json::{json, Map, Value};

use crate::config::schema::SpeechGroup;

use super::super::executor::SessionOpen;
use super::super::route::{self, FieldRefusal, ModelRefusal};

/// OpenAI's PCM16 rate, the default.
pub(crate) const OPENAI_RATE: u32 = 24_000;
// The model's own rate, `TARGET_RATE`, is accepted as a declared extension,
// and not resampled.

/// `transcription.delay`'s values. Inert: the cadence is
/// `speech.partial_interval_ms`.
const DELAYS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];

/// `server_vad`'s fields, as a session runs them (spec speech/05).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ServerVad {
    pub threshold: f64,
    pub prefix_padding_ms: u32,
    pub silence_duration_ms: u32,
    /// No speech for this long is an item of its own; `None` never.
    pub idle_timeout_ms: Option<u32>,
    /// Inert, since they belong to conversation sessions: echoed when set.
    pub create_response: Option<bool>,
    pub interrupt_response: Option<bool>,
}

/// The server's `server_vad`, read when a session opens: `speech.vad_*`, and
/// `speech.max_clip_seconds` in milliseconds, the most any of its lengths
/// takes, since an item holds no more.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VadDefaults {
    pub threshold: f64,
    pub prefix_padding_ms: u32,
    pub silence_duration_ms: u32,
    pub max_ms: u32,
}

impl VadDefaults {
    pub fn of(group: &SpeechGroup) -> Self {
        Self {
            threshold: f64::from(group.vad_threshold) / 1000.0,
            prefix_padding_ms: group.vad_prefix_padding_ms,
            silence_duration_ms: group.vad_silence_duration_ms,
            max_ms: group.max_clip_seconds.saturating_mul(1000),
        }
    }

    fn server_vad(&self) -> ServerVad {
        ServerVad {
            threshold: self.threshold,
            prefix_padding_ms: self.prefix_padding_ms,
            silence_duration_ms: self.silence_duration_ms,
            idle_timeout_ms: None,
            create_response: None,
            interrupt_response: None,
        }
    }
}

/// `session.ignis.diarization`: the diarizer's geometry a session's speakers
/// come from (spec speech/08).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Diarization {
    /// The live geometry, for audio at real time.
    Live,
    /// The file geometry, for audio fed faster than real time.
    File,
}

impl Diarization {
    pub fn as_str(self) -> &'static str {
        match self {
            Diarization::Live => "live",
            Diarization::File => "file",
        }
    }
}

/// What a session runs with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SessionConfig {
    /// The rate its PCM16 arrives at: [`OPENAI_RATE`] or [`TARGET_RATE`].
    pub rate: u32,
    /// The speech model's served id, whatever name the client sent.
    pub model: String,
    pub prompt: Option<String>,
    pub keywords: Vec<String>,
    pub language: Option<String>,
    pub languages: Vec<String>,
    /// The language `language` or `languages` force, by the model's English
    /// name; `None` detects it.
    pub forced: Option<String>,
    pub delay: Option<String>,
    /// `session.ignis.provisional_tail`: the provisional tail is sent.
    pub provisional_tail: bool,
    /// `session.ignis.diarization`: the speakers and their geometry; `None`
    /// none. Only under `server_vad`.
    pub diarization: Option<Diarization>,
    /// `session.ignis.max_utterance_seconds`: the length guard's limit, `0`
    /// off; `None` is the server's `speech.max_utterance_seconds`.
    pub max_utterance_seconds: Option<u32>,
    /// `session.ignis.item_language`: each item's language is sent (spec
    /// speech/07).
    pub item_language: bool,
    /// `turn_detection`: `server_vad`, or `None` for manual commit.
    pub turn_detection: Option<ServerVad>,
    /// What `server_vad`'s fields take when unset or `null`.
    pub vad_defaults: VadDefaults,
}

/// An event or a field refused: what its `error` event carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub code: &'static str,
    pub message: String,
    pub param: Option<String>,
}

impl Refusal {
    /// A value outside what the field takes.
    pub fn invalid(message: impl Into<String>, param: &str) -> Self {
        Self { code: "invalid_request_error", message: message.into(), param: Some(param.to_owned()) }
    }

    /// A value OpenAI serves and this server does not.
    pub fn unsupported(message: impl Into<String>, param: &str) -> Self {
        Self { code: "unsupported_value", message: message.into(), param: Some(param.to_owned()) }
    }
}

impl From<FieldRefusal> for Refusal {
    fn from(refused: FieldRefusal) -> Self {
        Self::invalid(refused.message, &refused.param)
    }
}

const TRANSCRIPTION: &str = "session.audio.input.transcription";

/// The keys `session.ignis` takes (spec speech/08, speech/07).
const IGNIS_KEYS: [&str; 4] = ["provisional_tail", "diarization", "max_utterance_seconds", "item_language"];

const DIARIZATION: &str = "session.ignis.diarization";

impl SessionConfig {
    /// A session as it opens: 24 kHz, the speech model, no context, the
    /// language detected, no provisional tail and no item's language,
    /// `server_vad` at the server's defaults (spec speech/05, Q22).
    pub fn new(model: String, vad_defaults: VadDefaults) -> Self {
        Self {
            rate: OPENAI_RATE,
            model,
            prompt: None,
            keywords: Vec::new(),
            language: None,
            languages: Vec::new(),
            forced: None,
            delay: None,
            provisional_tail: false,
            diarization: None,
            max_utterance_seconds: None,
            item_language: false,
            turn_detection: Some(vad_defaults.server_vad()),
            vad_defaults,
        }
    }

    /// What the executor's session opens with: the context, prefilled when
    /// it opens (Q50), and the language forced.
    pub fn open(&self, group: &SpeechGroup) -> SessionOpen {
        SessionOpen::of(group, route::context_of(self.prompt.as_deref(), &self.keywords), self.forced.clone())
    }

    /// This configuration with `session.update`'s `session` applied, or the
    /// first field refused. `model` decides what a
    /// `transcription.model` reaches (spec speech/02's rule); `diarizer` says
    /// whether the diarizer can serve, and if not why, when the update asks
    /// for speakers (spec speech/08).
    pub fn updated(
        &self,
        session: &Value,
        model: impl Fn(&str) -> Result<String, ModelRefusal>,
        diarizer: impl Fn() -> Result<(), String>,
    ) -> Result<Self, Refusal> {
        let Some(session) = session.as_object() else {
            return Err(Refusal::invalid("session.update needs a `session` object", "session"));
        };
        let mut next = self.clone();
        match session.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(kind)) if kind == "transcription" => {}
            Some(Value::String(kind)) if kind == "realtime" => {
                return Err(Refusal::unsupported(
                    "Realtime conversation sessions are not served: this socket transcribes, with session.type `transcription`",
                    "session.type",
                ));
            }
            Some(_) => return Err(Refusal::invalid("session.type must be `transcription`", "session.type")),
        }
        if let Some(input) = object(session, "audio", "session.audio")?.map(|audio| object(audio, "input", "session.audio.input")).transpose()?.flatten() {
            next.format(input.get("format"))?;
            next.transcription(input.get("transcription"), &model)?;
            next.turn_detection(input.get("turn_detection"))?;
            noise_reduction(input.get("noise_reduction"))?;
        }
        include(session.get("include"))?;
        let mut asks_for_speakers = false;
        if let Some(ignis) = object(session, "ignis", "session.ignis")? {
            // ignis's own namespace lists its keys, so a misspelt opt-in is
            // refused rather than passing unnoticed.
            if let Some(unknown) = ignis.keys().find(|key| !IGNIS_KEYS.contains(&key.as_str())) {
                return Err(Refusal::invalid(
                    format!("session.ignis.{unknown} is not an ignis session field: session.ignis takes {}", IGNIS_KEYS.join(", ")),
                    &format!("session.ignis.{unknown}"),
                ));
            }
            match ignis.get("provisional_tail") {
                None => {}
                Some(Value::Null) => next.provisional_tail = false,
                Some(Value::Bool(on)) => next.provisional_tail = *on,
                Some(_) => return Err(Refusal::invalid("session.ignis.provisional_tail must be true or false", "session.ignis.provisional_tail")),
            }
            match ignis.get("item_language") {
                None => {}
                Some(Value::Null) => next.item_language = false,
                Some(Value::Bool(on)) => next.item_language = *on,
                Some(_) => return Err(Refusal::invalid("session.ignis.item_language must be true or false", "session.ignis.item_language")),
            }
            match ignis.get("diarization") {
                None => {}
                Some(Value::Null) => next.diarization = None,
                Some(Value::String(mode)) if mode == "live" || mode == "file" => {
                    next.diarization = Some(if mode == "live" { Diarization::Live } else { Diarization::File });
                    asks_for_speakers = true;
                }
                Some(_) => return Err(Refusal::invalid("session.ignis.diarization must be null, \"live\" or \"file\"", DIARIZATION)),
            }
            match ignis.get("max_utterance_seconds") {
                None => {}
                Some(Value::Null) => next.max_utterance_seconds = None,
                Some(seconds) => match seconds.as_u64().filter(|&s| s <= u64::from(self.vad_defaults.max_ms / 1000)) {
                    Some(seconds) => next.max_utterance_seconds = Some(seconds as u32),
                    None => {
                        return Err(Refusal::invalid(
                            format!(
                                "session.ignis.max_utterance_seconds must be null (the server's), 0 (off) or a whole number of seconds up to {} (speech.max_clip_seconds)",
                                self.vad_defaults.max_ms / 1000
                            ),
                            "session.ignis.max_utterance_seconds",
                        ))
                    }
                },
            }
        }
        // Speakers cut items as the VAD does: a session without it has no
        // items to cut (spec speech/08).
        if next.diarization.is_some() && next.turn_detection.is_none() {
            let message = "session.ignis.diarization needs server_vad: with turn_detection null (manual commit) the client ends its items, so no speaker change can end one; send turn_detection {\"type\": \"server_vad\"} or diarization null";
            let param = if asks_for_speakers { DIARIZATION } else { "session.audio.input.turn_detection" };
            return Err(Refusal::invalid(message, param));
        }
        if asks_for_speakers && next.diarization != self.diarization {
            if let Err(why) = diarizer() {
                return Err(Refusal {
                    code: "ignis_diarizer_unavailable",
                    message: format!("the diarizer is unavailable: {why}; the session goes on without speakers"),
                    param: Some(DIARIZATION.to_owned()),
                });
            }
        }
        next.forced = route::forced(next.language.as_deref(), &next.languages, &format!("{TRANSCRIPTION}.language"), &format!("{TRANSCRIPTION}.languages"))?;
        Ok(next)
    }

    fn format(&mut self, format: Option<&Value>) -> Result<(), Refusal> {
        const PARAM: &str = "session.audio.input.format";
        let format = match format {
            None => return Ok(()),
            Some(Value::Null) => {
                self.rate = OPENAI_RATE;
                return Ok(());
            }
            Some(Value::Object(format)) => format,
            Some(_) => return Err(Refusal::invalid(format!("{PARAM} must be an object"), PARAM)),
        };
        match format.get("type").and_then(Value::as_str) {
            None | Some("audio/pcm") => {}
            Some(g711 @ ("audio/pcmu" | "audio/pcma")) => {
                return Err(Refusal::unsupported(
                    format!("{PARAM}.type `{g711}` (G.711) is not served: send audio/pcm, 16-bit mono at 24000 Hz"),
                    &format!("{PARAM}.type"),
                ));
            }
            Some(other) => return Err(Refusal::invalid(format!("{PARAM}.type `{other}` is not audio/pcm"), &format!("{PARAM}.type"))),
        }
        self.rate = match format.get("rate") {
            None | Some(Value::Null) => OPENAI_RATE,
            Some(rate) => match rate.as_u64() {
                Some(rate) if rate == u64::from(OPENAI_RATE) || rate == u64::from(TARGET_RATE) => rate as u32,
                _ => {
                    return Err(Refusal::unsupported(
                        format!("{PARAM}.rate {rate} is not served: send 24000, OpenAI's, or 16000, the model's own"),
                        &format!("{PARAM}.rate"),
                    ))
                }
            },
        };
        Ok(())
    }

    fn transcription(&mut self, transcription: Option<&Value>, model: &impl Fn(&str) -> Result<String, ModelRefusal>) -> Result<(), Refusal> {
        let transcription = match transcription {
            None => return Ok(()),
            Some(Value::Null) => {
                let model = std::mem::take(&mut self.model);
                *self = Self {
                    rate: self.rate,
                    provisional_tail: self.provisional_tail,
                    diarization: self.diarization,
                    max_utterance_seconds: self.max_utterance_seconds,
                    item_language: self.item_language,
                    turn_detection: self.turn_detection.take(),
                    ..Self::new(model, self.vad_defaults)
                };
                return Ok(());
            }
            Some(Value::Object(transcription)) => transcription,
            Some(_) => return Err(Refusal::invalid(format!("{TRANSCRIPTION} must be an object"), TRANSCRIPTION)),
        };
        match transcription.get("model") {
            None => {}
            Some(Value::Null) => self.model = model("").map_err(|refusal| model_refusal(refusal))?,
            Some(Value::String(named)) => self.model = model(named).map_err(|refusal| model_refusal(refusal))?,
            Some(_) => return Err(Refusal::invalid(format!("{TRANSCRIPTION}.model must be a string"), &format!("{TRANSCRIPTION}.model"))),
        }
        if let Some(prompt) = optional_text(transcription, "prompt")? {
            self.prompt = prompt;
        }
        if let Some(keywords) = optional_list(transcription, "keywords")? {
            let param = format!("{TRANSCRIPTION}.keywords");
            if let Some(refused) = route::refused_keyword(&keywords, &param) {
                return Err(refused.into());
            }
            self.keywords = keywords;
        }
        if let Some(language) = optional_text(transcription, "language")? {
            self.language = language;
        }
        if let Some(languages) = optional_list(transcription, "languages")? {
            self.languages = languages;
        }
        if let Some(delay) = optional_text(transcription, "delay")? {
            if let Some(delay) = delay.as_deref().filter(|delay| !DELAYS.contains(delay)) {
                return Err(Refusal::invalid(
                    format!("{TRANSCRIPTION}.delay `{delay}` is not one of {}", DELAYS.join(", ")),
                    &format!("{TRANSCRIPTION}.delay"),
                ));
            }
            self.delay = delay;
        }
        Ok(())
    }

    /// `turn_detection`: `null` is manual commit; `server_vad` takes its
    /// fields, each left out keeping its value and `null` the server's default
    /// (a session coming from manual commit starts from the defaults);
    /// `semantic_vad` is refused (spec speech/05, Q22).
    fn turn_detection(&mut self, value: Option<&Value>) -> Result<(), Refusal> {
        const PARAM: &str = "session.audio.input.turn_detection";
        let detection = match value {
            None => return Ok(()),
            Some(Value::Null) => {
                self.turn_detection = None;
                return Ok(());
            }
            Some(Value::Object(detection)) => detection,
            Some(_) => return Err(Refusal::invalid(format!("{PARAM} must be null, or an object whose type is server_vad"), PARAM)),
        };
        match detection.get("type").and_then(Value::as_str) {
            Some("server_vad") => {}
            Some("semantic_vad") => {
                return Err(Refusal::unsupported(
                    format!("{PARAM} `semantic_vad` is not served: send server_vad, or null and commit with input_audio_buffer.commit"),
                    PARAM,
                ))
            }
            _ => return Err(Refusal::invalid(format!("{PARAM}.type must be server_vad or semantic_vad"), &format!("{PARAM}.type"))),
        }
        let defaults = self.vad_defaults;
        let mut vad = self.turn_detection.take().unwrap_or_else(|| defaults.server_vad());
        let millis = |least: u64| {
            let what = format!("a whole number of milliseconds from {least} to {} (speech.max_clip_seconds)", defaults.max_ms);
            move |value: &Value| value.as_u64().filter(|ms| (least..=u64::from(defaults.max_ms)).contains(ms)).map(|ms| ms as u32).ok_or(what.clone())
        };
        let number = |value: &Value| value.as_f64().filter(|p| (0.0..=1.0).contains(p)).ok_or("a number from 0 to 1".to_owned());
        let flag = |value: &Value| value.as_bool().ok_or("true or false".to_owned());
        if let Some(threshold) = detection_field(detection, "threshold", number)? {
            vad.threshold = threshold.unwrap_or(defaults.threshold);
        }
        if let Some(ms) = detection_field(detection, "prefix_padding_ms", millis(0))? {
            vad.prefix_padding_ms = ms.unwrap_or(defaults.prefix_padding_ms);
        }
        if let Some(ms) = detection_field(detection, "silence_duration_ms", millis(0))? {
            vad.silence_duration_ms = ms.unwrap_or(defaults.silence_duration_ms);
        }
        if let Some(ms) = detection_field(detection, "idle_timeout_ms", millis(1))? {
            vad.idle_timeout_ms = ms;
        }
        if let Some(on) = detection_field(detection, "create_response", flag)? {
            vad.create_response = on;
        }
        if let Some(on) = detection_field(detection, "interrupt_response", flag)? {
            vad.interrupt_response = on;
        }
        self.turn_detection = Some(vad);
        Ok(())
    }

    /// The `session` object of `session.created` and `session.updated`: the
    /// configuration in effect, `turn_detection` declared, `null` for manual
    /// commit (Q54).
    pub fn effective(&self, id: &str) -> Value {
        let mut transcription = Map::new();
        transcription.insert("model".to_owned(), json!(self.model));
        transcription.insert("prompt".to_owned(), json!(self.prompt.clone().unwrap_or_default()));
        transcription.insert("language".to_owned(), json!(self.language));
        if !self.keywords.is_empty() {
            transcription.insert("keywords".to_owned(), json!(self.keywords));
        }
        if !self.languages.is_empty() {
            transcription.insert("languages".to_owned(), json!(self.languages));
        }
        if let Some(delay) = &self.delay {
            transcription.insert("delay".to_owned(), json!(delay));
        }
        json!({
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": id,
            "audio": { "input": {
                "format": { "type": "audio/pcm", "rate": self.rate },
                "transcription": transcription,
                "noise_reduction": null,
                "turn_detection": self.turn_detection.as_ref().map(ServerVad::effective),
            } },
            "include": [],
            "ignis": {
                "provisional_tail": self.provisional_tail,
                "diarization": self.diarization.map(Diarization::as_str),
                "max_utterance_seconds": self.max_utterance_seconds,
                "item_language": self.item_language,
            },
        })
    }

    /// The length guard's limit in force, in seconds, the session's own or
    /// the server's `server_limit`; `None` when it is off.
    pub fn utterance_limit(&self, server_limit: u32) -> Option<u32> {
        Some(self.max_utterance_seconds.unwrap_or(server_limit)).filter(|&seconds| seconds > 0)
    }
}

fn model_refusal(refusal: ModelRefusal) -> Refusal {
    let param = Some(format!("{TRANSCRIPTION}.model"));
    match refusal {
        ModelRefusal::KindMismatch(message) => Refusal { code: "model_kind_mismatch", message, param },
        ModelRefusal::NotFound(message) => Refusal { code: "model_not_found", message, param },
    }
}

/// A field of `turn_detection`: `None` when absent, `Some(None)` when `null`,
/// or what `read` makes of it, which names what the field takes when it
/// refuses the value.
fn detection_field<T>(detection: &Map<String, Value>, name: &str, read: impl Fn(&Value) -> Result<T, String>) -> Result<Option<Option<T>>, Refusal> {
    const PARAM: &str = "session.audio.input.turn_detection";
    match detection.get(name) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(value) => read(value).map(|v| Some(Some(v))).map_err(|what| Refusal::invalid(format!("{PARAM}.{name} must be {what}"), &format!("{PARAM}.{name}"))),
    }
}

/// `parent[key]` as an object; `None` when absent or `null`.
fn object<'a>(parent: &'a Map<String, Value>, key: &str, param: &str) -> Result<Option<&'a Map<String, Value>>, Refusal> {
    match parent.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object)),
        Some(_) => Err(Refusal::invalid(format!("{param} must be an object"), param)),
    }
}

/// A text field of the transcription object: `None` when absent, `Some(None)`
/// when `null`.
fn optional_text(transcription: &Map<String, Value>, key: &str) -> Result<Option<Option<String>>, Refusal> {
    match transcription.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::String(text)) => Ok(Some(Some(text.clone()))),
        Some(_) => Err(Refusal::invalid(format!("{TRANSCRIPTION}.{key} must be a string"), &format!("{TRANSCRIPTION}.{key}"))),
    }
}

/// A list of strings of the transcription object: `None` when absent, empty
/// when `null`.
fn optional_list(transcription: &Map<String, Value>, key: &str) -> Result<Option<Vec<String>>, Refusal> {
    let refused = || Refusal::invalid(format!("{TRANSCRIPTION}.{key} must be an array of strings"), &format!("{TRANSCRIPTION}.{key}"));
    match transcription.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(Vec::new())),
        Some(Value::Array(items)) => items.iter().map(|item| item.as_str().map(str::to_owned).ok_or_else(refused)).collect::<Result<_, _>>().map(Some),
        Some(_) => Err(refused()),
    }
}

impl ServerVad {
    /// Its `turn_detection` object, the inert fields when set.
    fn effective(&self) -> Value {
        let mut detection = json!({
            "type": "server_vad",
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "idle_timeout_ms": self.idle_timeout_ms,
        });
        for (field, value) in [("create_response", self.create_response), ("interrupt_response", self.interrupt_response)] {
            if let Some(on) = value {
                detection[field] = json!(on);
            }
        }
        detection
    }

    /// What [`ignis_speech::vad::Turns`] finds turns with.
    pub fn settings(&self) -> TurnSettings {
        TurnSettings {
            threshold: self.threshold as f32,
            prefix_padding_ms: self.prefix_padding_ms,
            silence_duration_ms: self.silence_duration_ms,
            idle_timeout_ms: self.idle_timeout_ms,
        }
    }
}

/// `noise_reduction`: `null` is served; `near_field` and `far_field` are
/// refused.
fn noise_reduction(value: Option<&Value>) -> Result<(), Refusal> {
    const PARAM: &str = "session.audio.input.noise_reduction";
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(reduction)) => match reduction.get("type").and_then(Value::as_str) {
            Some(kind @ ("near_field" | "far_field")) => {
                Err(Refusal::unsupported(format!("{PARAM} `{kind}` is not served: this server reduces no noise; send null"), PARAM))
            }
            _ => Err(Refusal::invalid(format!("{PARAM}.type must be near_field or far_field"), PARAM)),
        },
        Some(_) => Err(Refusal::invalid(format!("{PARAM} must be null or an object"), PARAM)),
    }
}

/// `include`: empty is served; the transcription's logprobs are refused.
fn include(value: Option<&Value>) -> Result<(), Refusal> {
    const PARAM: &str = "session.include";
    let items = match value {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(Refusal::invalid(format!("{PARAM} must be an array"), PARAM)),
    };
    if items.iter().any(|item| item.as_str() == Some("item.input_audio_transcription.logprobs")) {
        return Err(Refusal::unsupported(
            format!("{PARAM} `item.input_audio_transcription.logprobs` is not served: this server reports no logprobs"),
            PARAM,
        ));
    }
    match items.is_empty() {
        true => Ok(()),
        false => Err(Refusal::invalid(format!("{PARAM} names nothing this server includes"), PARAM)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVED: &str = "qwen3-asr-1.7b";

    /// Spec speech/02's rule over a fixed set of names: absent, the served id
    /// or a compat name reach the speech model, a text model's id is a kind
    /// mismatch, anything else not found.
    fn model(named: &str) -> Result<String, ModelRefusal> {
        match named {
            "" | SERVED | "whisper-1" | "gpt-4o-transcribe" => Ok(SERVED.to_owned()),
            "qwen3.8-27b" => Err(ModelRefusal::KindMismatch("names qwen3.8-27b, a text model".to_owned())),
            other => Err(ModelRefusal::NotFound(format!("unknown model: {other}"))),
        }
    }

    fn update(config: &SessionConfig, session: Value) -> Result<SessionConfig, Refusal> {
        config.updated(&session, model, || Ok(()))
    }

    /// The server's `server_vad` defaults: `speech.vad_*`'s, with items held to 1,200 s.
    const DEFAULTS: VadDefaults = VadDefaults { threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500, max_ms: 1_200_000 };

    fn fresh() -> SessionConfig {
        SessionConfig::new(SERVED.to_owned(), DEFAULTS)
    }

    fn input(input: Value) -> Value {
        json!({ "type": "transcription", "audio": { "input": input } })
    }

    /// AC 2 (spec 03); spec 05's AC 4: a session opens at 24 kHz on the speech
    /// model, running `server_vad` at the server's defaults, declared, with no
    /// provisional tail.
    #[test]
    fn a_session_opens_at_24_khz_with_server_vad_declared() {
        let effective = fresh().effective("sess_1");
        assert_eq!(effective["type"], "transcription");
        assert_eq!(effective["id"], "sess_1");
        assert_eq!(effective["audio"]["input"]["format"], json!({ "type": "audio/pcm", "rate": 24000 }));
        assert_eq!(effective["audio"]["input"]["transcription"]["model"], SERVED);
        assert_eq!(
            effective["audio"]["input"]["turn_detection"],
            json!({ "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500, "idle_timeout_ms": null })
        );
        assert!(effective["audio"]["input"]["noise_reduction"].is_null());
        assert_eq!(effective["ignis"], json!({ "provisional_tail": false, "diarization": null, "max_utterance_seconds": null, "item_language": false }));
        let other = SessionConfig::new(SERVED.to_owned(), VadDefaults { threshold: 0.25, prefix_padding_ms: 100, silence_duration_ms: 800, ..DEFAULTS });
        let detection = &other.effective("s")["audio"]["input"]["turn_detection"];
        assert_eq!((detection["threshold"].as_f64(), detection["prefix_padding_ms"].as_u64(), detection["silence_duration_ms"].as_u64()), (Some(0.25), Some(100), Some(800)));
    }

    /// AC 2, 3: an update names what it changes; 16 kHz is the declared
    /// extension and the effective configuration reports it.
    #[test]
    fn an_update_changes_what_it_names_and_reports_the_effective_rate() {
        let config = update(&fresh(), input(json!({ "format": { "type": "audio/pcm", "rate": 16000 } }))).unwrap();
        assert_eq!(config.rate, 16000);
        assert_eq!(config.effective("s")["audio"]["input"]["format"]["rate"], 16000);
        let config = update(&config, input(json!({ "transcription": { "prompt": "A talk about GPUs." } }))).unwrap();
        assert_eq!(config.rate, 16000, "a field left out keeps its value");
        assert_eq!(config.prompt.as_deref(), Some("A talk about GPUs."));
        let config = update(&config, input(json!({ "format": null }))).unwrap();
        assert_eq!(config.rate, 24000, "null is the default");
        assert_eq!(config.prompt.as_deref(), Some("A talk about GPUs."));
    }

    /// AC 3: G.711 and any other rate are refused, naming the field.
    #[test]
    fn g711_and_other_rates_are_refused() {
        for (format, param) in [
            (json!({ "type": "audio/pcmu" }), "session.audio.input.format.type"),
            (json!({ "type": "audio/pcma" }), "session.audio.input.format.type"),
            (json!({ "type": "audio/pcm", "rate": 44100 }), "session.audio.input.format.rate"),
        ] {
            let refused = update(&fresh(), input(json!({ "format": format }))).unwrap_err();
            assert_eq!(refused.code, "unsupported_value", "{format}");
            assert_eq!(refused.param.as_deref(), Some(param));
        }
        assert!(update(&fresh(), input(json!({ "format": { "type": "audio/pcmu" } }))).unwrap_err().message.contains("G.711"));
    }

    /// AC 8 (spec 03); spec 05's AC 5: `turn_detection: null` is manual commit,
    /// declared null; `semantic_vad` is refused, and an object needs its type.
    #[test]
    fn manual_commit_is_served_and_semantic_vad_is_refused() {
        let manual = update(&fresh(), input(json!({ "turn_detection": null }))).unwrap();
        assert_eq!(manual.turn_detection, None);
        let declared = manual.effective("s")["audio"]["input"].clone();
        assert!(declared.as_object().unwrap().contains_key("turn_detection") && declared["turn_detection"].is_null());
        let refused = update(&fresh(), input(json!({ "turn_detection": { "type": "semantic_vad" } }))).unwrap_err();
        assert_eq!((refused.code, refused.param.as_deref()), ("unsupported_value", Some("session.audio.input.turn_detection")));
        assert!(refused.message.contains("semantic_vad") && refused.message.contains("server_vad"), "{}", refused.message);
        let refused = update(&fresh(), input(json!({ "turn_detection": { "threshold": 0.4 } }))).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.audio.input.turn_detection.type"), "the type is required");
    }

    fn vad(config: &SessionConfig) -> ServerVad {
        config.turn_detection.clone().expect("server_vad")
    }

    /// Spec 05's AC 2: `server_vad` honours `threshold`, `prefix_padding_ms` and
    /// `silence_duration_ms`; a field left out keeps its value, `null` is the
    /// server's default, and a session coming back from manual commit starts
    /// from the defaults. A value outside its range is refused, naming it.
    #[test]
    fn server_vad_takes_its_three_fields_each_by_its_own_rule() {
        let config = update(&fresh(), input(json!({ "turn_detection": { "type": "server_vad", "threshold": 0.7, "silence_duration_ms": 900 } }))).unwrap();
        assert_eq!((vad(&config).threshold, vad(&config).prefix_padding_ms, vad(&config).silence_duration_ms), (0.7, 300, 900));
        let config = update(&config, input(json!({ "turn_detection": { "type": "server_vad", "prefix_padding_ms": 0 } }))).unwrap();
        assert_eq!((vad(&config).threshold, vad(&config).prefix_padding_ms, vad(&config).silence_duration_ms), (0.7, 0, 900), "left out keeps");
        let config = update(&config, input(json!({ "turn_detection": { "type": "server_vad", "threshold": null } }))).unwrap();
        assert_eq!(vad(&config).threshold, 0.5, "null is the server's default");
        let config = update(&config, input(json!({ "transcription": { "prompt": "x" } }))).unwrap();
        assert_eq!(vad(&config).silence_duration_ms, 900, "an update without turn_detection keeps it");
        let manual = update(&config, input(json!({ "turn_detection": null }))).unwrap();
        let back = update(&manual, input(json!({ "turn_detection": { "type": "server_vad", "threshold": 0.6 } }))).unwrap();
        assert_eq!((vad(&back).threshold, vad(&back).prefix_padding_ms, vad(&back).silence_duration_ms), (0.6, 300, 500));
        assert_eq!(back.effective("s")["audio"]["input"]["turn_detection"]["threshold"], 0.6);
        let param = |field: &str| format!("session.audio.input.turn_detection.{field}");
        for (field, value) in [
            ("threshold", json!(1.5)),
            ("threshold", json!(-0.1)),
            ("threshold", json!("high")),
            ("prefix_padding_ms", json!(-1)),
            ("prefix_padding_ms", json!(1_200_001)),
            ("silence_duration_ms", json!(2.5)),
            ("idle_timeout_ms", json!(0)),
            ("idle_timeout_ms", json!(1_200_001)),
            ("create_response", json!("yes")),
        ] {
            let refused = update(&fresh(), input(json!({ "turn_detection": { "type": "server_vad", field: value } }))).unwrap_err();
            assert_eq!((refused.code, refused.param), ("invalid_request_error", Some(param(field))), "{field}: {value}");
        }
    }

    /// Spec 05: `idle_timeout_ms` is honoured; `create_response` and
    /// `interrupt_response` are inert, echoed when set.
    #[test]
    fn the_idle_timeout_is_honoured_and_the_response_fields_are_inert() {
        let config = update(
            &fresh(),
            input(json!({ "turn_detection": { "type": "server_vad", "idle_timeout_ms": 6000, "create_response": false, "interrupt_response": true } })),
        )
        .unwrap();
        assert_eq!(vad(&config).idle_timeout_ms, Some(6000));
        let detection = &config.effective("s")["audio"]["input"]["turn_detection"];
        assert_eq!(
            (detection["idle_timeout_ms"].as_u64(), detection["create_response"].as_bool(), detection["interrupt_response"].as_bool()),
            (Some(6000), Some(false), Some(true))
        );
        let config = update(&config, input(json!({ "turn_detection": { "type": "server_vad", "idle_timeout_ms": null } }))).unwrap();
        assert_eq!(vad(&config).idle_timeout_ms, None);
        assert!(!fresh().effective("s")["audio"]["input"]["turn_detection"].as_object().unwrap().contains_key("create_response"));
    }

    /// AC 8: the rest of the table: `delay` inert, `noise_reduction` null
    /// served and the two kinds refused, `include` logprobs refused, a
    /// conversation session refused.
    #[test]
    fn every_other_field_is_honoured_inert_or_refused_as_the_table_says() {
        let config = update(&fresh(), input(json!({ "transcription": { "delay": "low" }, "noise_reduction": null }))).unwrap();
        assert_eq!(config.delay.as_deref(), Some("low"));
        assert_eq!(config.open(&SpeechGroup::default()), fresh().open(&SpeechGroup::default()), "delay changes nothing the executor sees");
        assert_eq!(update(&fresh(), input(json!({ "transcription": { "delay": "soon" } }))).unwrap_err().code, "invalid_request_error");
        for kind in ["near_field", "far_field"] {
            let refused = update(&fresh(), input(json!({ "noise_reduction": { "type": kind } }))).unwrap_err();
            assert_eq!((refused.code, refused.param.as_deref()), ("unsupported_value", Some("session.audio.input.noise_reduction")));
        }
        let refused = update(&fresh(), json!({ "type": "transcription", "include": ["item.input_audio_transcription.logprobs"] })).unwrap_err();
        assert_eq!((refused.code, refused.param.as_deref()), ("unsupported_value", Some("session.include")));
        let behind = update(&fresh(), json!({ "include": ["x", "item.input_audio_transcription.logprobs"] })).unwrap_err();
        assert_eq!(behind.code, "unsupported_value", "the logprobs are refused wherever they are listed");
        assert!(update(&fresh(), json!({ "type": "transcription", "include": [] })).is_ok());
        let refused = update(&fresh(), json!({ "type": "realtime" })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.type"));
        assert!(refused.message.contains("conversation"), "{}", refused.message);
    }

    /// AC 6: the provisional tail is opt-in under `session.ignis`.
    #[test]
    fn the_provisional_tail_is_opt_in() {
        assert!(!fresh().provisional_tail);
        let on = update(&fresh(), json!({ "type": "transcription", "ignis": { "provisional_tail": true } })).unwrap();
        assert!(on.provisional_tail);
        assert_eq!(on.effective("s")["ignis"]["provisional_tail"], true);
        let refused = update(&fresh(), json!({ "ignis": { "provisional_tail": "yes" } })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.ignis.provisional_tail"));
    }

    /// Spec speech/07: each item's language is opt-in under `session.ignis`,
    /// declared in the effective configuration; `null` or `false` turns it
    /// off, a reset transcription keeps it, and anything else is refused.
    #[test]
    fn the_items_language_is_opt_in() {
        assert!(!fresh().item_language);
        let on = update(&fresh(), json!({ "type": "transcription", "ignis": { "item_language": true } })).unwrap();
        assert!(on.item_language);
        assert_eq!(on.effective("s")["ignis"]["item_language"], true);
        let kept = update(&on, input(json!({ "transcription": { "prompt": "x" } }))).unwrap();
        assert!(kept.item_language, "an update that does not name it keeps it");
        assert!(update(&on, input(json!({ "transcription": null }))).unwrap().item_language, "a reset transcription keeps it");
        for off in [json!(null), json!(false)] {
            let off = update(&on, json!({ "ignis": { "item_language": off } })).unwrap();
            assert!(!off.item_language);
            assert_eq!(off.effective("s")["ignis"]["item_language"], false);
        }
        for bad in [json!("yes"), json!(1), json!({})] {
            let refused = update(&on, json!({ "ignis": { "item_language": bad } })).unwrap_err();
            assert_eq!(refused.param.as_deref(), Some("session.ignis.item_language"), "{bad}");
        }
    }

    /// Honoured or refused (spec server/09): a key under `session.ignis` that
    /// is not one of ignis's is refused naming it and the four it takes
    /// (spec speech/08, speech/07), the update not applied.
    #[test]
    fn an_unknown_ignis_key_is_refused_naming_it() {
        let refused = update(&fresh(), json!({ "ignis": { "provisional_tial": true } })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.ignis.provisional_tial"));
        assert!(refused.message.contains("provisional_tial") && refused.message.contains("provisional_tail"), "{}", refused.message);
        assert!(refused.message.ends_with("session.ignis takes provisional_tail, diarization, max_utterance_seconds, item_language"), "{}", refused.message);
        let refused = update(&fresh(), json!({ "ignis": { "item_languages": true } })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.ignis.item_languages"));
        let refused = update(&fresh(), json!({ "ignis": { "provisional_tail": true, "extra": 1 } })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.ignis.extra"));
    }

    /// Spec speech/08, AC 9: `session.ignis.diarization` takes `null`,
    /// `"live"` or `"file"` on a `server_vad` session, declared in the
    /// effective configuration; on manual commit it is refused naming
    /// `turn_detection`, and so is a switch to manual commit while speakers
    /// are on; with the diarizer unavailable it is refused with the reason,
    /// and the session keeps what it had.
    #[test]
    fn diarization_is_taken_on_server_vad_and_refused_on_manual_commit() {
        let live = update(&fresh(), json!({ "ignis": { "diarization": "live" } })).unwrap();
        assert_eq!(live.diarization, Some(Diarization::Live));
        assert_eq!(live.effective("s")["ignis"]["diarization"], "live");
        let file = update(&live, json!({ "ignis": { "diarization": "file" } })).unwrap();
        assert_eq!(file.effective("s")["ignis"]["diarization"], "file");
        let kept = update(&file, input(json!({ "transcription": null }))).unwrap();
        assert_eq!(kept.diarization, Some(Diarization::File), "a reset transcription keeps the speakers");
        assert_eq!(update(&file, json!({ "ignis": { "diarization": null } })).unwrap().diarization, None);
        for bad in [json!("yes"), json!(true), json!({ "type": "live" })] {
            let refused = update(&fresh(), json!({ "ignis": { "diarization": bad } })).unwrap_err();
            assert_eq!(refused.param.as_deref(), Some("session.ignis.diarization"), "{bad}");
        }
        let manual = update(&fresh(), input(json!({ "turn_detection": null }))).unwrap();
        let refused = update(&manual, json!({ "ignis": { "diarization": "live" } })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.ignis.diarization"));
        assert!(refused.message.contains("turn_detection"), "{}", refused.message);
        let both = update(&fresh(), json!({ "audio": { "input": { "turn_detection": null } }, "ignis": { "diarization": "live" } })).unwrap_err();
        assert!(both.message.contains("turn_detection"), "{}", both.message);
        let refused = update(&live, input(json!({ "turn_detection": null }))).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.audio.input.turn_detection"), "speakers on: manual commit is refused");
        assert!(update(&live, json!({ "audio": { "input": { "turn_detection": null } }, "ignis": { "diarization": null } })).is_ok());
        let unavailable = || Err("no diarizer is configured".to_owned());
        let refused = fresh().updated(&json!({ "ignis": { "diarization": "live" } }), model, unavailable).unwrap_err();
        assert_eq!((refused.code, refused.param.as_deref()), ("ignis_diarizer_unavailable", Some("session.ignis.diarization")));
        assert!(refused.message.contains("no diarizer is configured"), "{}", refused.message);
        assert!(live.updated(&input(json!({ "transcription": { "prompt": "x" } })), model, unavailable).is_ok(), "only asking for speakers asks the diarizer");
        assert!(fresh().updated(&json!({ "ignis": { "diarization": null } }), model, unavailable).is_ok());
    }

    /// Spec speech/08: `session.ignis.max_utterance_seconds` is `null` (the
    /// server's), `0` (off) or a limit up to `speech.max_clip_seconds`,
    /// declared as set.
    #[test]
    fn the_sessions_utterance_limit_overrides_the_servers() {
        assert_eq!(fresh().utterance_limit(120), Some(120));
        assert_eq!(fresh().utterance_limit(0), None, "the server's off");
        let off = update(&fresh(), json!({ "ignis": { "max_utterance_seconds": 0 } })).unwrap();
        assert_eq!((off.utterance_limit(120), off.effective("s")["ignis"]["max_utterance_seconds"].as_u64()), (None, Some(0)));
        let short = update(&off, json!({ "ignis": { "max_utterance_seconds": 30 } })).unwrap();
        assert_eq!(short.utterance_limit(0), Some(30), "the session's over the server's");
        assert_eq!(update(&short, json!({ "ignis": { "max_utterance_seconds": null } })).unwrap().utterance_limit(120), Some(120));
        for bad in [json!(-1), json!(2.5), json!("30"), json!(1_201)] {
            let refused = update(&fresh(), json!({ "ignis": { "max_utterance_seconds": bad } })).unwrap_err();
            assert_eq!(refused.param.as_deref(), Some("session.ignis.max_utterance_seconds"), "{bad}");
        }
        assert!(update(&fresh(), json!({ "ignis": { "max_utterance_seconds": 1_200 } })).is_ok());
    }

    /// AC 9, 11: the prompt and keywords are the context, as spec 02 joins
    /// them; one language forces it, several detect, both fields refused; a
    /// keyword the system turn cannot carry is refused.
    #[test]
    fn the_context_and_the_language_follow_spec_02() {
        let config = update(&fresh(), input(json!({ "transcription": { "prompt": "A talk.", "keywords": ["ignis", "Qwen3"], "language": "it" } }))).unwrap();
        let open = config.open(&SpeechGroup::default());
        assert_eq!(open.context, "A talk.\nignis, Qwen3");
        assert_eq!(open.context_max_tokens, 512);
        assert_eq!(open.context_copy_words, 3, "the copy guard's run, as the server sets it");
        assert_eq!(open.context_copy_edits, 0, "the words a copy may change, as the server sets it");
        assert_eq!(open.language.as_deref(), Some("Italian"));
        assert_eq!(fresh().open(&SpeechGroup::default()).language, None, "nothing is forced unless asked");
        let detect = update(&fresh(), input(json!({ "transcription": { "languages": ["it", "en"] } }))).unwrap();
        assert_eq!(detect.forced, None);
        let one = update(&fresh(), input(json!({ "transcription": { "languages": ["de"] } }))).unwrap();
        assert_eq!(one.forced.as_deref(), Some("German"));
        let both = update(&config, input(json!({ "transcription": { "languages": ["en"] } }))).unwrap_err();
        assert_eq!(both.param.as_deref(), Some("session.audio.input.transcription.languages"));
        let unknown = update(&fresh(), input(json!({ "transcription": { "language": "xx" } }))).unwrap_err();
        assert_eq!(unknown.param.as_deref(), Some("session.audio.input.transcription.language"));
        let keyword = update(&fresh(), input(json!({ "transcription": { "keywords": ["<b>"] } }))).unwrap_err();
        assert_eq!(keyword.param.as_deref(), Some("session.audio.input.transcription.keywords"));
        // A refused update changes nothing: the caller keeps the old one.
        assert_eq!(config.forced.as_deref(), Some("Italian"));
    }

    /// AC 10: `transcription.model` follows spec 02's resolution; the served id
    /// is what the session reports.
    #[test]
    fn the_model_follows_spec_02() {
        let config = update(&fresh(), input(json!({ "transcription": { "model": "gpt-4o-transcribe" } }))).unwrap();
        assert_eq!(config.model, SERVED);
        assert_eq!(config.effective("s")["audio"]["input"]["transcription"]["model"], SERVED);
        let refused = update(&fresh(), input(json!({ "transcription": { "model": "qwen3.8-27b" } }))).unwrap_err();
        assert_eq!((refused.code, refused.param.as_deref()), ("model_kind_mismatch", Some("session.audio.input.transcription.model")));
        assert_eq!(update(&fresh(), input(json!({ "transcription": { "model": "nope" } }))).unwrap_err().code, "model_not_found");
    }
}
