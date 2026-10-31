//! `GET /v1/realtime?intent=transcription` (spec speech/03): OpenAI's
//! Realtime GA transcription session over a WebSocket, on the speech model.
//!
//! - **The socket.** The upgrade is keyed like the Responses socket's (a
//!   bearer token, or the `openai-insecure-api-key.<key>` subprotocol),
//!   sits outside the serving gate and the text model's idle clock, and is
//!   refused before it opens: a conversation session (`400` naming
//!   `intent`), a speech model that cannot serve (`503
//!   speech_unavailable`), `speech.max_sessions` open (`503` with
//!   `Retry-After`).
//! - **Items.** Audio appended at 24 kHz (16 kHz, the declared extension)
//!   goes to the current item, which a commit ends and a clear discards.
//!   Each item has its own session on the executor (spec speech/01 §4),
//!   opened before its first append, so the context is prefilled when the
//!   socket opens or a `session.update` changes it, never by a commit
//!   (Q50).
//! - **Turn detection** (spec speech/05). Under `server_vad`, the default,
//!   the VAD reads the 16 kHz audio on this socket's task and decides what
//!   reaches an item: speech starts one (`speech_started`, the prefix padding
//!   before it), the silence after it commits it (`speech_stopped`), and with
//!   an idle timeout so much audio without speech is an item of its own. The
//!   model reads each such item after `speech.vad_leading_silence_ms` of
//!   digital silence, which no event and no `usage` counts (Q77). A commit
//!   the client sends with no speech heard commits the audio appended since
//!   the last item, as OpenAI's buffer holds it. With `turn_detection: null`
//!   every sample goes to the current item and the client commits.
//! - **Units.** Every call to the executor is one unit on the speech worker,
//!   in this socket's lane: a window pushed, a partial, an open, a step of a
//!   commit (the rest pushed, a window, the decode's prefill, 16 decode
//!   steps). The sessions take turns, unit by unit, ahead of the batch
//!   windows (Q44). The executor's session lives in a slot the units share;
//!   they run one at a time on the worker's one thread, in the order queued.
//! - **Refreshes are best effort** (Q74). The live transcript is refreshed
//!   at most every `speech.partial_interval_ms`, and a refresh is skipped,
//!   never queued, while any unit of the socket's is still in the worker: a
//!   CPU that cannot keep up drops refreshes, never windows or the commit.
//!   The next refresh also waits `speech.partial_cost_factor` times the last
//!   one's compute time (Q76), so the refreshes keep to about that share of
//!   the worker as the item grows and each costs more.
//! - **A commit stops the refresh in flight** (Q79). An item's commit, its
//!   clear or the socket's end stops a refresh of its still in the worker at
//!   the next point the executor checks (spec speech/01 §4): the refresh
//!   sends nothing and leaves the executor's session as it was, so the commit
//!   runs next, from the stable text already sent. A window never stops one.
//! - **Text.** `delta` carries stable text only, so the deltas joined are the
//!   transcript `completed` carries (Q43). A session that opted in with
//!   `session.ignis.provisional_tail` also gets the tail, the text still
//!   inside the rollback zone, as `ignis.input_audio_transcription.provisional`;
//!   one that opted in with `session.ignis.item_language` gets each item's
//!   language, the model's label as its code, as
//!   `ignis.input_audio_transcription.language`, after its speaker and before
//!   `completed` (spec speech/07).
//! - **Speakers** (spec speech/08). Under `server_vad`, a session that asks
//!   with `session.ignis.diarization` feeds the diarizer the audio the VAD
//!   reads, a step of its geometry a unit in this socket's lane on the
//!   diarizer's own thread, beside the speech worker: a step never waits for
//!   the speech model's units and never stops them, and a refresh is not
//!   skipped for a step, only for the socket's speech units. A change of
//!   speaker inside an item cuts it there, the audio past the change heard
//!   already moved to the next item and the item decoded again up to it. No
//!   text is fixed before an item ends; its end sends one delta, its speaker
//!   and `completed`, once the frames cover it. A stream that stalls, no
//!   audio for as long as its next step still needs at real time and the
//!   client's last piece again, gets them from a copy of it finished.
//! - **The length guard** (spec speech/08). An item past
//!   `speech.max_utterance_seconds` (the session's
//!   `session.ignis.max_utterance_seconds` first) ends at the VAD's next pause
//!   or `speech.utterance_cut_search_ms` later, where the audio is.
//! - **The queue limit** (spec speech/08). An append that would take the
//!   audio not computed yet past `speech.max_queued_audio_seconds` is refused,
//!   `ignis_audio_queue_full`.
//! - **Limits.** An item past `speech.max_clip_seconds`, past
//!   `server.stream_timeout` from its first append, or whose transcription
//!   fails, gets `failed`: at its commit, after an `error` saying why at
//!   once, or at once if it is already committed. A socket idle past `server.socket_idle_close` is closed, by
//!   the Responses sockets' rule.
//!
//! Neither the audio nor the transcript is logged: each committed item writes
//! one `ignis.speech.transcription` line (Q13, Q45).

pub(crate) mod config;
pub(crate) mod pcm;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use ignis_speech::diarizer::speakers::Speakers;
use ignis_speech::diarizer::FRAME_SAMPLES;
use ignis_speech::vad::{TurnEvent, Turns, Vad};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::Instant;

use super::audio::TARGET_RATE;
use super::executor::{
    DiarizerExecutor, DiarizerStream, Frames, LoadSettings, Partial, SessionOpen, SpeechExecutor, SpeechFiles, SpeechSession, StreamOpen, Transcript, WINDOW_SAMPLES,
};
use super::route::{self, Ending};
use super::worker::{Full, Step};
use super::{extension, languages, open_loaded, Fault, SessionGuard};
use crate::api::{self, bad_request_param, ApiError};
use crate::deadline::Timeout;
use crate::metrics::SpeechOutcome;
use crate::responses::ids::random_suffix;
use crate::responses::socket::{close_idle, offered, CREDENTIAL_PREFIX};
use crate::Server;
use config::{Diarization, Refusal, SessionConfig, VadDefaults};
use pcm::Pcm;

/// The most base64 `audio` one append carries: OpenAI's limit, 15 MiB.
pub(crate) const MAX_APPEND_BYTES: usize = 15 << 20;

/// `GET /v1/realtime` — a Realtime transcription session.
#[utoipa::path(
    get,
    path = "/v1/realtime",
    tag = "audio",
    operation_id = "realtime_transcription",
    summary = "A Realtime transcription session",
    description = "Upgrades to a WebSocket carrying OpenAI's Realtime GA transcription session (`session.type: \"transcription\"`) on the speech model (Qwen3-ASR, ADR 0067). Open it with `?intent=transcription` and no `model` in the URL; a conversation session (no `intent=transcription`, or a `model` in the URL) is refused `400` naming `intent`. Authenticate with `Authorization: Bearer <key>` on the upgrade or, from a browser, with a `Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<key>`; the server selects the first offered entry that is not a credential (`realtime`, for OpenAI's browser clients). The socket serves through a model switch and with the text model unloaded, and opening it starts loading a text model the idle clock unloaded, while the session is served on the CPU. At most `speech.max_sessions` (4) sessions are open at once.

On open the server sends `session.created` with the effective configuration: `audio.input.format` `{type: \"audio/pcm\", rate: 24000}`, `transcription.model` the speech model's served id, `turn_detection` `{type: \"server_vad\", threshold, prefix_padding_ms, silence_duration_ms, idle_timeout_ms: null}` at the server's `speech.vad_threshold`, `speech.vad_prefix_padding_ms` and `speech.vad_silence_duration_ms` (0.5, 300, 500, OpenAI's), `noise_reduction: null`, `include: []` and `ignis` `{provisional_tail: false, diarization: null, max_utterance_seconds: null, item_language: false}`. The speech model is loaded if it is not, and the context prefilled.

Turn detection. Under `server_vad` (the default) the server finds the speech itself, with Silero VAD on the CPU, one speech probability per 32 ms of audio: the first 32 ms at or over `threshold` start an item, `input_audio_buffer.speech_started` `{audio_start_ms, item_id}`, and the item keeps `prefix_padding_ms` of the audio before it (silence, even the silence that ended the item before, never an earlier item's speech); once the speech has been silent (under `threshold`) for `silence_duration_ms`, `input_audio_buffer.speech_stopped` `{audio_end_ms, item_id}`, and the server commits the item, that silence included: `committed`, then its transcription. Audio outside an item goes nowhere. With `idle_timeout_ms`, that much audio with no speech, since the session opened, the last item or the field's change, is committed as an item of its own after `input_audio_buffer.timeout_triggered` `{audio_start_ms, audio_end_ms, item_id}`. Every time is in milliseconds of the audio the session has received. The speech model reads each of these items after `speech.vad_leading_silence_ms` of digital silence, since its transcript depends on the audio before the speech; no time and no `usage` counts it. With `turn_detection: null` the client commits (manual commit), as OpenAI's Realtime clients that set it, the Playground among them, do.

The length guard. Under `server_vad`, once an item holds `speech.max_utterance_seconds` (120; `session.ignis.max_utterance_seconds` overrides it, `0` off) of audio, the server ends it at the next 32 ms the VAD hears under `threshold`, or `speech.utterance_cut_search_ms` (10000) later at the latest, where the audio is: `speech_stopped` with `audio_end_ms` at the cut, `committed`, then `speech_started` for a new item with `audio_start_ms` at the cut, which takes no prefix padding and is fed after the leading silence. The search looks forward only.

Speakers. With `session.ignis.diarization` (`\"live\"` or `\"file\"`, the diarizer's geometry; only under `server_vad`) the diarizer, Nemotron 3 Diarization on the CPU, reads the session's audio as the VAD does and tells up to 8 speakers apart, numbered 0 to 7 in the order they first spoke. A change of speaker inside an item ends it at the change, with the events of the length guard's cut at the new speaker's first 80 ms: the item ended is decoded again from its audio up to the change, and the audio from the change on is the new item's. In such a session no text is fixed before an item ends: the provisional tail carries all of it, and the item's end sends one `delta` with the whole transcript, then `ignis.input_audio_transcription.speaker` `{item_id, content_index, speaker}`, then `completed`, once the diarizer's frames cover the item's end (when no audio comes for as long as the diarizer's next step still needs at real time, and the client's last piece again, it scores what it heard past its last step as if the audio ended there). An item with no speech gets no speaker. A diarizer that fails is an `error`, `ignis_diarizer_failed`, and the session goes on without speakers.

Client events, each a text frame of one JSON object:
- `session.update` with `session.type: \"transcription\"`, answered `session.updated` with the effective configuration. It names only what it changes; `null` puts a field back to its default; one refused field refuses the whole update. `audio.input.format.rate` is 24000 (resampled to 16 kHz) or 16000 (the model's own, an ignis extension); `audio/pcmu` and `audio/pcma` (G.711) are refused. `transcription.model` is absent, the served id or one of `speech.compat_names`; a text or image model's id is an `error` with `model_kind_mismatch`, any other `model_not_found`. `transcription.prompt` and `transcription.keywords` are the context, prefilled when the update arrives (a change while an item holds audio applies from the next item); `transcription.language` and `transcription.languages` force or detect the language as `POST /v1/audio/transcriptions` does: one language forces it, several are each checked and the model detects over all 30, since it cannot be held to a list. `transcription.delay` is inert: the live transcript refreshes at most every `speech.partial_interval_ms`. `turn_detection` takes `null` or `{type: \"server_vad\"}` with any of `threshold` (0 to 1), `prefix_padding_ms`, `silence_duration_ms` and `idle_timeout_ms` (each at most `speech.max_clip_seconds`): a field left out keeps its value, `null` is the server's default (no idle timeout); `create_response` and `interrupt_response` are inert, since they belong to conversation sessions; `semantic_vad` is refused. A switch between `null` and `server_vad` while an item holds audio holds from the next item; `server_vad`'s fields hold at once. `noise_reduction` takes `null`; `near_field` and `far_field` are refused. `include: [\"item.input_audio_transcription.logprobs\"]` is refused. `session.ignis.provisional_tail: true` opts in to the provisional tail; `session.ignis.diarization` asks for speakers (above), refused on manual commit naming `turn_detection`, and with the diarizer unavailable refused with the reason (`ignis_diarizer_unavailable`), the session going on without speakers; `session.ignis.max_utterance_seconds` is the length guard's limit (`null` the server's, `0` off); `session.ignis.item_language: true` opts in to each item's language (below); any other key under `session.ignis` is an `error` naming it and the four.
- `input_audio_buffer.append` `{audio}`: base64 16-bit little-endian mono PCM at the session's rate, up to 15 MiB of base64 an event. No reply. An append that would take the audio the session holds not computed yet (waiting for the speech model's windows or the diarizer's steps) past `speech.max_queued_audio_seconds` (120) is refused with an `error`, `ignis_audio_queue_full`, and the session stays open.
- `input_audio_buffer.commit`: ends the item. `input_audio_buffer.committed` `{item_id, previous_item_id}`, then the last stable text as a `delta`, then `completed`. An empty buffer is an `error` (`input_audio_buffer_commit_empty`). Under `server_vad` the buffer holds the audio appended since the last item ended, as OpenAI's does: a commit during speech ends that item at once, with no `speech_stopped`, and one with no speech heard makes an item of that audio (at most `speech.max_clip_seconds` of it, the leading silence included); a clear discards it.
- `input_audio_buffer.clear`: discards the buffer and the item's partial state (under `server_vad`, the speech and the padding the VAD holds); answered `input_audio_buffer.cleared`.
- `transcription_session.update` (the retired beta) is an `error` naming `session.update`; conversation events (`response.*`, `conversation.*`, `output_audio_buffer.*`) are an `error`.

Server events, each with an `event_id`: under `server_vad`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped` and `input_audio_buffer.timeout_triggered`, as above; `conversation.item.input_audio_transcription.delta` `{item_id, content_index: 0, delta}` carries stable text only, while the audio arrives (the live transcript is refreshed at most every `speech.partial_interval_ms`, best effort: a refresh is skipped while the session's audio is still being computed, and the next waits at least `speech.partial_cost_factor` times the last one's compute time; a commit or a clear stops a refresh in flight at its next check, and a stopped refresh sends nothing), so the deltas of an item joined equal its `completed.transcript`; `conversation.item.input_audio_transcription.completed` `{item_id, content_index, transcript, usage: {type: \"duration\", seconds}}`; `conversation.item.input_audio_transcription.failed` `{item_id, content_index, error}` for an item over `speech.max_clip_seconds`, one past `--server-stream-timeout` from its first append (`request_timeout`), or one the speech model failed; `error` `{error: {type, code, message, param, event_id}}` for every refused event or field, which never closes the socket. With `session.ignis.provisional_tail: true` the session also gets `ignis.input_audio_transcription.provisional` `{item_id, content_index, text}`: the whole current tail, the text the next refresh may still revise, which replaces the previous one, and `\"\"` when the item ends. A session that does not opt in never receives it. With `session.ignis.item_language: true` each item also gets `ignis.input_audio_transcription.language` `{item_id, content_index, language}`, the language the model labelled its transcript with as its ISO 639 code (`it`; under a forced language, that one), after its speaker and right before its `completed`; an item the model labelled `None` (no speech), or with a label outside its 30 languages, gets none. A session that does not opt in never receives it either.

A socket idle for `server.socket_idle_close` seconds (default 300; 0 never) -- no item committed and still being transcribed, and no text or binary frame from the client -- is closed with a normal close frame (1000) whose reason says so.",
    params(("intent" = String, Query, description = "`transcription`: the only session this route serves.")),
    responses(
        (status = 101, description = "Switching protocols: the session is open."),
        (status = 400, description = "Not a transcription session (`param: \"intent\"`): `intent` is not `transcription`, or the URL names a `model`; or a plain `GET`, not a WebSocket upgrade, to a route that could open (`upgrade_required`).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the upgrade carried no matching bearer token or `openai-insecure-api-key.<key>` subprotocol.", body = ApiError),
        (status = 503, description = "The speech model is unavailable (`speech_unavailable`, with the reason, and no `Retry-After`), or `speech.max_sessions` sessions are open (`engine_full`, with `Retry-After`). A plain `GET` without the upgrade is answered the same, so a page can ask why its upgrade was refused.", body = ApiError),
    ),
)]
pub(crate) async fn connect(
    State(server): State<Arc<Server>>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    if query.get("intent").map(String::as_str) != Some("transcription") || query.contains_key("model") {
        return bad_request_param(
            "this server serves Realtime transcription sessions only: open /v1/realtime?intent=transcription with no model in the URL; conversation sessions are not served",
            "intent",
        );
    }
    let live = server.live();
    let group = &live.speech;
    let files = match server.speech.resolve(group) {
        Ok(files) => files,
        Err(why) => return route::unavailable(why),
    };
    let refused = |full: Full| {
        route::engine_full(
            format!("the speech model has {} Realtime sessions open (speech.max_sessions); retry in {} s", group.max_sessions, full.retry_after_secs),
            full,
        )
    };
    // A plain GET is answered what an upgrade would be, so a page that cannot
    // read a refused upgrade's status can still ask (spec speech/04, AC 11):
    // it claims no place.
    let Ok(upgrade) = upgrade else {
        return match server.speech.session_room(group) {
            Err(why) => refused(why),
            Ok(()) => api::error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "upgrade_required",
                "GET /v1/realtime is the Realtime transcription session and takes a WebSocket upgrade, with ?intent=transcription; the speech model can serve now",
            ),
        };
    };
    let guard = match server.speech.open_session(group) {
        Ok(guard) => guard,
        Err(why) => return refused(why),
    };
    // The client's own names, never the credential.
    let protocols: Vec<String> = offered(&headers).filter(|protocol| !protocol.starts_with(CREDENTIAL_PREFIX)).map(str::to_owned).collect();
    upgrade
        .protocols(protocols)
        .max_message_size(api::TEXT_REQUEST_BODY_LIMIT)
        .max_frame_size(api::TEXT_REQUEST_BODY_LIMIT)
        .on_upgrade(move |socket| Connection::new(server, socket, guard, files).run())
}

/// The executor's session the units of one socket take turns at, and what
/// the item's units have computed so far.
#[derive(Default)]
struct Slot {
    session: Option<Box<dyn SpeechSession>>,
    computed: Duration,
}

/// The diarizer's stream of a session's speakers (spec speech/08), which
/// outlives the items: a lock of its own, so a step on the diarizer's thread
/// never waits for a unit of the speech model's, which holds the [`Slot`].
type StreamSlot = Arc<Mutex<Option<Box<dyn DiarizerStream>>>>;

/// One call to the executor, a window or less of work.
enum Unit {
    /// Load the model if it is not, and open the next item's session,
    /// prefilling its context.
    Open(SessionOpen, LoadSettings),
    /// Push audio: a whole window, as soon as one is in.
    Push(Vec<f32>),
    /// Push what is in, then refresh the live transcript.
    Partial(Vec<f32>),
    /// Push the rest, then the commit, a step of it a unit: the task's
    /// samples are taken by its first step.
    Commit(Vec<f32>),
    /// End the item at sample `at` of what its session was pushed: the audio
    /// from it on and the stable text go (spec speech/08).
    Cut(usize),
}

impl Unit {
    /// The 16 kHz samples it carries: audio not computed while it waits.
    fn samples(&self) -> usize {
        match self {
            Unit::Push(samples) | Unit::Partial(samples) | Unit::Commit(samples) => samples.len(),
            Unit::Open(..) | Unit::Cut(_) => 0,
        }
    }
}

/// One call to the diarizer (spec speech/08), on its own thread.
enum SpeakersUnit {
    /// Open the diarizer's stream, loading the diarizer if it is not.
    Open(StreamOpen),
    /// Push the stream's next samples, then score its next step.
    Diarize(Vec<f32>),
    /// Push the stream's next samples, then score what it heard past its
    /// last step on a copy of it, as if it ended now: the stream stalled.
    Tail(Vec<f32>),
}

impl SpeakersUnit {
    /// The 16 kHz samples it carries: audio not computed while it waits.
    fn samples(&self) -> usize {
        match self {
            SpeakersUnit::Diarize(samples) | SpeakersUnit::Tail(samples) => samples.len(),
            SpeakersUnit::Open(_) => 0,
        }
    }
}

/// Where a unit is: steps left (a commit), or out of the worker with what it
/// says, if anything.
enum Progress {
    More,
    Done(Option<Outcome>),
}

/// What a unit tells the socket.
enum Outcome {
    /// A refresh, and its compute time on the worker.
    Partial(Partial, Duration),
    Transcript(Transcript, Duration),
    Failed(Failure),
    /// The diarizer's stream opened: the samples its first step waits for.
    Opened(usize),
    /// A step's frames, the samples the next waits for, and the step's
    /// compute time on the diarizer's thread.
    Frames(Frames, usize, Duration),
    /// What the stream heard past its last step, scored as if it ended.
    Tail(Frames),
    /// The diarizer failed: the session goes on without speakers.
    Unheard(String),
}

/// A unit is out of its worker: whose it was, what it says, if anything,
/// and the samples it carried.
struct Done {
    of: Of,
    outcome: Option<Outcome>,
    samples: usize,
}

/// Whose a unit is.
#[derive(Clone, Copy)]
enum Of {
    /// The item `seq`'s, on the speech worker.
    Item(u64),
    /// The speakers' stream numbered so in the session, on the diarizer's
    /// thread: a stream turned off or replaced since hears nothing more.
    Stream(u64),
}

/// Why an item failed.
#[derive(Debug, Clone)]
enum Failure {
    /// The model would not load, or a call to it failed.
    Fault(Fault),
    /// Longer than `speech.max_clip_seconds` (the limit), counting the
    /// milliseconds of leading silence a `server_vad` item is fed after.
    TooLong(u32, u32),
    /// Past `server.stream_timeout` from its first append.
    TimedOut(Timeout),
}

impl Failure {
    /// The `failed` event's `error`.
    fn error(&self) -> Value {
        let (kind, code, message) = match self {
            Self::Fault(fault) => {
                let (code, message) = fault.error("this item's transcription");
                ("server_error", code, message)
            }
            Self::TooLong(limit, 0) => (
                "invalid_request_error",
                "invalid_request_error",
                format!("the item holds more than speech.max_clip_seconds ({limit} s) of audio; nothing is split automatically, commit sooner"),
            ),
            Self::TooLong(limit, lead_ms) => (
                "invalid_request_error",
                "invalid_request_error",
                format!(
                    "the item holds more than speech.max_clip_seconds ({limit} s) of audio, counting the {lead_ms} ms of silence (speech.vad_leading_silence_ms) the speech model reads before it; nothing is split automatically, commit sooner"
                ),
            ),
            Self::TimedOut(timeout) => ("request_timeout", "request_timeout", timeout.message()),
        };
        json!({ "type": kind, "code": code, "message": message, "param": null })
    }

    fn outcome(&self) -> SpeechOutcome {
        match self {
            Self::TimedOut(_) => SpeechOutcome::TimedOut,
            _ => SpeechOutcome::Failed,
        }
    }
}

/// One item: the audio between two commits (or clears).
struct Item {
    seq: u64,
    /// Given at its first append.
    id: Option<String>,
    /// Set when the item is cleared, failed or the socket closes: its units
    /// still queued do nothing.
    cancel: Arc<AtomicBool>,
    /// Set at its commit, its clear or the socket's end: a refresh of its in
    /// the worker stops, and one queued stops before it begins (Q79).
    stop_refresh: Arc<AtomicBool>,
    /// What its executor session was opened with.
    opened_with: SessionOpen,
    /// Samples appended, at whatever rate, and the seconds they make.
    samples_in: u64,
    seconds: f64,
    /// The milliseconds of digital silence the model reads before its audio,
    /// under `server_vad`: counted against `speech.max_clip_seconds`, as the
    /// model counts them, and nowhere else.
    lead_ms: u32,
    timeout: Timeout,
    deadline: Option<Instant>,
    /// The stable text sent as deltas, and the tail last sent.
    emitted: String,
    tail: String,
    failure: Option<Failure>,
    /// Set at its commit: the line and the count it ends with.
    ending: Option<Ending>,
    /// Audio was queued since the last partial.
    fresh_audio: bool,
    /// Under `server_vad`, its first sample on the session's clock (the
    /// VAD's), and its end once it ended.
    start: u64,
    end: Option<u64>,
    /// It has a speaker (spec speech/08): the session's speakers were on when
    /// its audio began. Its text is sent only at its end, with its speaker.
    diarized: bool,
    /// Its speaker, once the diarizer's frames cover its end: `Some(None)`
    /// when none of them is speech, or the speakers stopped.
    speaker: Option<Option<u8>>,
    /// Its transcript, until its speaker is known.
    transcript: Option<(Transcript, Duration)>,
}

struct Connection {
    server: Arc<Server>,
    socket: WebSocket,
    guard: SessionGuard,
    files: SpeechFiles,
    session_id: String,
    /// What this session's event ids begin with: a piece of its id.
    event_prefix: String,
    config: SessionConfig,
    pcm: Pcm,
    slot: Arc<Mutex<Slot>>,
    results_tx: UnboundedSender<Done>,
    results: UnboundedReceiver<Done>,
    /// The committed items still being transcribed, in order, then the
    /// current one, last: there is always one.
    items: VecDeque<Item>,
    next_seq: u64,
    /// The current item's 16 kHz samples not yet queued.
    pending: Vec<f32>,
    previous_item_id: Option<String>,
    /// When the current item's live transcript may next be refreshed.
    next_partial: Option<Instant>,
    /// When the session's last refresh was queued, and the earliest the next
    /// may be: `speech.partial_cost_factor` times its compute time later.
    refresh_queued: Instant,
    refresh_hold: Instant,
    /// This socket's speech units queued on the speech worker and not yet out
    /// of it, and the 16 kHz samples they carry; and the samples the speakers'
    /// units carry on the diarizer's thread.
    in_worker: usize,
    in_worker_samples: usize,
    speakers_in_worker_samples: usize,
    outbox: Vec<Value>,
    next_event: u64,
    /// The client's latest text or binary frame, or the end of the latest
    /// item: where the idle time counts from.
    quiet_since: Instant,
    /// The 16 kHz samples the session has heard: the VAD's clock.
    heard: u64,
    /// Under `server_vad`, the turns the VAD finds in the audio; `None` for
    /// manual commit (spec speech/05).
    turns: Option<Turns<Vad>>,
    /// Under `server_vad`, the audio heard since the last item ended.
    since_item: SinceItem,
    /// Under `server_vad`, with `session.ignis.diarization`, the speakers.
    speakers: Option<Diarized>,
    /// The speakers' streams opened so far: the next one's number.
    streams: u64,
}

/// A session's speakers (spec speech/08): the diarizer's stream, on the
/// diarizer's thread, fed the 16 kHz audio the VAD reads from sample `origin`
/// on, a step of its geometry at a time; and the frames it scored, by the
/// ghost rule.
/// The audio is read from where the speakers start, never from before: the
/// leading silence only the speech model is fed is no audio of the stream's.
struct Diarized {
    mode: Diarization,
    /// Its number in the session, which its units' answers carry.
    number: u64,
    /// The diarizer's stream, once its open unit opened it.
    stream: StreamSlot,
    /// The session's sample the stream's first is.
    origin: u64,
    /// The samples heard since `origin` that the stream has been sent, and
    /// the ones still to send.
    sent: u64,
    pending: Vec<f32>,
    /// The samples heard, from `origin`, its next step waits for; `None` until
    /// the stream is open.
    due: Option<u64>,
    /// A step is on the diarizer's thread; a tail is.
    stepping: bool,
    tailing: bool,
    /// The frames scored, and the latest tail's: the steps' with the frames
    /// heard past them, until the steps reach as far.
    frames: Speakers,
    tail: Option<Speakers>,
    /// Set when the speakers stop or the socket ends: its units still queued
    /// do nothing.
    cancel: Arc<AtomicBool>,
    /// The diarizer failed: the session goes on without speakers.
    failed: bool,
    /// What the session's log line says when its speakers end: the channels
    /// the items were given (a bit each), the steps' compute times on the
    /// diarizer's thread, and the tails run.
    heard_speakers: u32,
    steps: u32,
    step_total: Duration,
    step_max: Duration,
    tails: u32,
    /// When the audio arrived, and the most a frame waited from its first
    /// sample's arrival to its step's probabilities: the diarizer's pace
    /// (spec speech/08, AC 11).
    arrivals: Arrivals,
    lag_max: Duration,
    /// The items given a speaker.
    items: u32,
}

impl Diarized {
    /// The session's one log line about its speakers (spec speech/08, Out of
    /// Scope): how many were heard, what a step cost on the diarizer's thread, the tails
    /// run, and how long a frame waited for its probabilities, the figures a
    /// person tunes `diarizer.*` by. Numbers only.
    fn log(&self, session: &str) {
        let mean = self.step_total.checked_div(self.steps).unwrap_or_default();
        tracing::info!(
            name: "ignis.speech.diarization",
            session = %session,
            speakers = self.heard_speakers.count_ones(),
            steps = self.steps,
            step_mean_ms = mean.as_secs_f64() * 1000.0,
            step_max_ms = self.step_max.as_secs_f64() * 1000.0,
            tails = self.tails,
            items = self.items,
            lag_max_ms = self.lag_max.as_secs_f64() * 1000.0,
            "a session with speakers ended"
        );
    }

    /// The frame a session sample is in, from the stream's first: the first
    /// whole frame at or after it, for an item's start; the first after it,
    /// for an end.
    fn frame(&self, sample: u64) -> usize {
        sample.saturating_sub(self.origin).div_ceil(FRAME_SAMPLES as u64) as usize
    }

    /// The session sample frame `frame` starts at.
    fn sample(&self, frame: usize) -> u64 {
        self.origin + (frame * FRAME_SAMPLES) as u64
    }

    /// When the stream has stalled (spec speech/08): its latest audio as long
    /// ago as the samples its next step still waits for take at real time,
    /// and its latest piece's again, since a client at real-time pace sends a
    /// piece that long after the one before. `None` before the stream opens.
    fn stalls_at(&self) -> Option<Instant> {
        let due = self.due?;
        let (at, piece) = self.arrivals.latest();
        let short = due.saturating_sub(self.sent + self.pending.len() as u64);
        Some(at + Duration::from_secs_f64((short + piece) as f64 / f64::from(TARGET_RATE)))
    }
}

/// When the session had heard so many samples, an entry per piece of audio
/// heard: when a frame's first sample arrived, for the diarizer's pace (spec
/// speech/08, AC 11), and when the latest piece did, for its stall. It keeps
/// what no step has scored yet, at any geometry's latency: audio the queue
/// limit bounds, with a step.
struct Arrivals {
    heard: VecDeque<(u64, Instant)>,
    /// The samples the latest piece brought.
    piece: u64,
}

impl Arrivals {
    /// The session had heard `heard` samples at `at`, when the speakers
    /// started.
    fn new(heard: u64, at: Instant) -> Arrivals {
        Arrivals { heard: VecDeque::from([(heard, at)]), piece: 0 }
    }

    /// The session had heard `heard` samples at `at`.
    fn heard(&mut self, heard: u64, at: Instant) {
        let (before, _) = self.heard.back().expect("an arrival from the start");
        self.piece = heard - before;
        self.heard.push_back((heard, at));
    }

    /// When the session's sample `sample` had arrived, while it is kept.
    fn arrived(&self, sample: u64) -> Option<Instant> {
        self.heard.iter().find(|(heard, _)| *heard >= sample).map(|(_, at)| *at)
    }

    /// When the latest piece arrived, and the samples it brought.
    fn latest(&self) -> (Instant, u64) {
        let (_, at) = self.heard.back().expect("an arrival from the start");
        (*at, self.piece)
    }

    /// The steps scored the session's samples before `sample`: the entries
    /// before the one at or just before it go.
    fn scored(&mut self, sample: u64) {
        while self.heard.get(1).is_some_and(|(heard, _)| *heard <= sample) {
            self.heard.pop_front();
        }
    }
}

/// Under `server_vad`, the 16 kHz audio heard since the last item ended: what
/// a commit with no speech heard makes an item of, as OpenAI's buffer holds
/// everything appended since the last commit. It always ends at the VAD's
/// clock (the samples the session heard), so an item's end, a sample of that
/// clock, drops what came before it. At most `speech.max_clip_seconds` of it
/// is kept: 77 MB at the default 1,200 s, for a session that long silent.
#[derive(Default)]
struct SinceItem {
    samples: VecDeque<f32>,
    /// The clock at its first sample.
    from: u64,
}

impl SinceItem {
    /// Nothing heard since sample `at`: the VAD starts there, or a clear.
    fn reset(&mut self, at: u64) {
        self.samples.clear();
        self.from = at;
    }

    /// `samples` heard, at most the last `keep` kept.
    fn heard(&mut self, samples: &[f32], keep: usize) {
        self.samples.extend(samples);
        let over = self.samples.len().saturating_sub(keep);
        self.samples.drain(..over);
        self.from += over as u64;
    }

    /// An item ended at sample `end` of the clock.
    fn ended_at(&mut self, end: u64) {
        let gone = end.saturating_sub(self.from).min(self.samples.len() as u64);
        self.samples.drain(..gone as usize);
        self.from += gone;
    }

    /// The samples `from..to` of the clock heard, as far as they are kept.
    fn range(&self, from: u64, to: u64) -> Vec<f32> {
        let at = |sample: u64| (sample.saturating_sub(self.from) as usize).min(self.samples.len());
        self.samples.range(at(from)..at(to).max(at(from))).copied().collect()
    }

    /// The last `room` samples, for the item a commit makes of them; nothing
    /// is left.
    fn take(&mut self, room: usize) -> Vec<f32> {
        let end = self.from + self.samples.len() as u64;
        let skip = self.samples.len().saturating_sub(room);
        let audio = self.samples.iter().skip(skip).copied().collect();
        self.reset(end);
        audio
    }
}

impl Connection {
    fn new(server: Arc<Server>, socket: WebSocket, guard: SessionGuard, files: SpeechFiles) -> Self {
        let (results_tx, results) = unbounded_channel();
        let config = SessionConfig::new(files.id.clone(), VadDefaults::of(&server.live().speech));
        let suffix = random_suffix();
        Self {
            server,
            socket,
            guard,
            files,
            event_prefix: format!("event_{}", &suffix[..16]),
            session_id: format!("sess_{suffix}"),
            pcm: Pcm::new(config.rate),
            config,
            slot: Arc::default(),
            results_tx,
            results,
            items: VecDeque::new(),
            next_seq: 0,
            pending: Vec::new(),
            previous_item_id: None,
            next_partial: None,
            refresh_queued: Instant::now(),
            refresh_hold: Instant::now(),
            in_worker: 0,
            in_worker_samples: 0,
            speakers_in_worker_samples: 0,
            outbox: Vec::new(),
            next_event: 0,
            quiet_since: Instant::now(),
            heard: 0,
            turns: None,
            since_item: SinceItem::default(),
            speakers: None,
            streams: 0,
        }
    }

    async fn run(mut self) {
        // Q40: a person dictating a prompt sends it next, so a text model the
        // idle clock unloaded starts loading now; the session is served on
        // the CPU meanwhile.
        let server = Arc::clone(&self.server);
        tokio::spawn(async move {
            let _ = crate::model_switch::ensure_loaded(&server, None).await;
        });
        let session = self.config.effective(&self.session_id);
        self.send(json!({ "type": "session.created", "session": session }));
        self.open_next();
        let look = tokio::time::sleep_until(self.next_look());
        tokio::pin!(look);
        loop {
            if !self.flush().await {
                break;
            }
            look.as_mut().reset(self.next_look());
            // Biased, the client's frames first, as on the Responses socket:
            // a frame already read when a deadline falls due wins it.
            tokio::select! {
                biased;
                frame = self.socket.recv() => match frame {
                    Some(Ok(Message::Text(text))) => {
                        self.quiet_since = Instant::now();
                        self.receive(text.as_str());
                    }
                    Some(Ok(Message::Binary(_))) => {
                        self.quiet_since = Instant::now();
                        self.refuse(Refusal::invalid("a frame must be a text frame carrying one JSON event", "type"), None);
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                    Some(Ok(Message::Close(_))) => {
                        while let Some(Ok(_)) = self.socket.recv().await {}
                        break;
                    }
                    Some(Err(_)) | None => break,
                },
                Some(done) = self.results.recv() => self.done(done),
                () = &mut look => {
                    if let Some(after) = self.idle_past() {
                        close_idle(&mut self.socket, after, "realtime").await;
                        break;
                    }
                    self.tick();
                }
            }
        }
        // What is still queued does nothing; a committed item not transcribed
        // counts as cancelled (its `Ending`, dropped).
        for item in &self.items {
            item.cancel.store(true, Ordering::SeqCst);
            item.stop_refresh.store(true, Ordering::SeqCst);
        }
        if let Some(speakers) = self.speakers.take() {
            speakers.cancel.store(true, Ordering::SeqCst);
            speakers.log(&self.session_id);
        }
    }

    /// Queue a server event, with its `event_id`.
    fn send(&mut self, mut event: Value) {
        self.next_event += 1;
        event["event_id"] = json!(format!("{}{:08x}", self.event_prefix, self.next_event));
        self.outbox.push(event);
    }

    async fn flush(&mut self) -> bool {
        for event in std::mem::take(&mut self.outbox) {
            if self.socket.send(Message::Text(event.to_string().into())).await.is_err() {
                return false;
            }
        }
        true
    }

    /// An `error` event refusing a client event or a field.
    fn refuse(&mut self, refusal: Refusal, event_id: Option<String>) {
        self.send(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "code": refusal.code, "message": refusal.message, "param": refusal.param, "event_id": event_id },
        }));
    }

    fn current(&mut self) -> &mut Item {
        self.items.back_mut().expect("a socket always has a current item")
    }

    /// The `speech.context_max_tokens` and load settings in effect now. An
    /// item with speakers keeps its audio, so a speaker change found after it
    /// was heard can still cut it (spec speech/08).
    fn open_settings(&self) -> (SessionOpen, LoadSettings) {
        let live = self.server.live();
        let group = &live.speech;
        let open = SessionOpen { keep_audio: self.config.diarization.is_some(), ..self.config.open(group) };
        (open, LoadSettings { threads: group.threads, pinned_weights: group.pinned_weights })
    }

    /// Start the next item: its executor session opens now, so its context
    /// is prefilled before its first append (Q50).
    fn open_next(&mut self) {
        let (open, load) = self.open_settings();
        let item = Item {
            seq: self.next_seq,
            id: None,
            cancel: Arc::new(AtomicBool::new(false)),
            stop_refresh: Arc::new(AtomicBool::new(false)),
            opened_with: open.clone(),
            samples_in: 0,
            seconds: 0.0,
            lead_ms: 0,
            timeout: Timeout::NONE,
            deadline: None,
            emitted: String::new(),
            tail: String::new(),
            failure: None,
            ending: None,
            fresh_audio: false,
            start: self.heard,
            end: None,
            diarized: false,
            speaker: None,
            transcript: None,
        };
        self.next_seq += 1;
        self.items.push_back(item);
        self.queue(Unit::Open(open, load));
        self.pending.clear();
        self.next_partial = None;
        self.sync_turns();
        self.sync_speakers();
    }

    /// The turn detection `session.update` set, from now while the current
    /// item holds no audio; a switch between manual commit and `server_vad`
    /// otherwise waits for the item to end, as a changed context does.
    /// `server_vad`'s fields hold from the VAD's next frame either way.
    fn sync_turns(&mut self) {
        let holds_audio = self.current().samples_in > 0;
        match (&self.config.turn_detection, self.turns.as_mut()) {
            (Some(vad), Some(turns)) => turns.set_settings(vad.settings()),
            (Some(vad), None) if !holds_audio => {
                self.turns = Some(Turns::new(Vad::new(), vad.settings(), self.heard));
                self.since_item.reset(self.heard);
            }
            (None, Some(_)) if !holds_audio => self.turns = None,
            _ => {}
        }
    }

    /// Queue `unit` of the current item on the speech worker, in this socket's
    /// lane. Every unit answers when it is out of the worker, so the socket
    /// knows what it still has in it.
    fn queue(&mut self, mut unit: Unit) {
        let item = self.items.back().expect("a current item");
        let (seq, cancel, stop) = (item.seq, Arc::clone(&item.cancel), Arc::clone(&item.stop_refresh));
        let samples = unit.samples();
        let (slot, broken) = (Arc::clone(&self.slot), Arc::clone(&self.slot));
        let executor = Arc::clone(self.server.speech.executor());
        let files = self.files.clone();
        let work = move || run(&mut unit, &slot, executor.as_ref(), &files, &stop);
        // A unit that panicked leaves no session to go on with.
        let panicked = move || {
            broken.lock().unwrap_or_else(PoisonError::into_inner).session = None;
            Outcome::Failed(Failure::Fault(Fault::Executor("the speech executor panicked".to_owned())))
        };
        let task = self.task(Done { of: Of::Item(seq), outcome: None, samples }, cancel, work, panicked);
        self.in_worker += 1;
        self.in_worker_samples += samples;
        self.server.speech.submit_interactive(&self.guard, task);
    }

    /// Queue a unit of the speakers' (spec speech/08): an interactive unit of
    /// no item, on the diarizer's own thread, after the speakers' units before
    /// it, in turn with the other sessions'.
    fn queue_speakers(&mut self, mut unit: SpeakersUnit) {
        let speakers = self.speakers.as_ref().expect("the speakers are on");
        let (cancel, stream, broken) = (Arc::clone(&speakers.cancel), Arc::clone(&speakers.stream), Arc::clone(&speakers.stream));
        let samples = unit.samples();
        let diarizer = Arc::clone(self.server.speech.diarizer());
        let work = move || run_speakers(&mut unit, &stream, diarizer.as_ref());
        // A unit that panicked leaves no stream to go on with.
        let panicked = move || {
            *broken.lock().unwrap_or_else(PoisonError::into_inner) = None;
            Outcome::Unheard("the diarizer panicked".to_owned())
        };
        let task = self.task(Done { of: Of::Stream(speakers.number), outcome: None, samples }, cancel, work, panicked);
        self.speakers_in_worker_samples += samples;
        self.server.speech.submit_diarizer(&self.guard, task);
    }

    /// A unit as its worker runs it: nothing once `cancel` is set, `work`
    /// otherwise, `panicked`'s outcome if it panics; out of the worker, it
    /// answers `done` with its outcome.
    fn task(
        &self,
        done: Done,
        cancel: Arc<AtomicBool>,
        mut work: impl FnMut() -> Progress + Send + 'static,
        panicked: impl Fn() -> Outcome + Send + 'static,
    ) -> super::worker::Task {
        let results = self.results_tx.clone();
        let mut done = Some(done);
        Box::new(move || {
            let progress = match cancel.load(Ordering::SeqCst) {
                true => Progress::Done(None),
                false => std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut work)).unwrap_or_else(|_| Progress::Done(Some(panicked()))),
            };
            match progress {
                Progress::More => Step::More,
                Progress::Done(outcome) => {
                    let done = done.take().expect("a unit is done once");
                    let _ = results.send(Done { outcome, ..done });
                    Step::Done
                }
            }
        })
    }

    /// One client frame.
    fn receive(&mut self, text: &str) {
        let Ok(Value::Object(event)) = serde_json::from_str::<Value>(text) else {
            return self.refuse(Refusal::invalid("a frame must carry one JSON object event", "type"), None);
        };
        let event_id = event.get("event_id").and_then(Value::as_str).map(str::to_owned);
        match event.get("type").and_then(Value::as_str) {
            Some("session.update") => self.update(&event, event_id),
            Some("input_audio_buffer.append") => self.append(&event, event_id),
            Some("input_audio_buffer.commit") => self.commit(event_id),
            Some("input_audio_buffer.clear") => self.clear(),
            Some("transcription_session.update") => self.refuse(
                Refusal::unsupported(
                    "transcription_session.update is the retired beta's event: send session.update with session.type `transcription`",
                    "type",
                ),
                event_id,
            ),
            Some(kind) if ["response.", "conversation.", "output_audio_buffer."].iter().any(|prefix| kind.starts_with(prefix)) => self.refuse(
                Refusal::unsupported(
                    format!("`{kind}` belongs to a Realtime conversation session, which this server does not serve: this session transcribes"),
                    "type",
                ),
                event_id,
            ),
            Some(kind) => self.refuse(
                Refusal::invalid(
                    format!("unknown event type `{kind}`: a transcription session takes session.update and input_audio_buffer.append, .commit and .clear"),
                    "type",
                ),
                event_id,
            ),
            None => self.refuse(Refusal::invalid("an event needs a string type", "type"), event_id),
        }
    }

    fn update(&mut self, event: &Map<String, Value>, event_id: Option<String>) {
        let live = self.server.live();
        let served = self.files.id.clone();
        let model = |named: &str| {
            route::model_named(&self.server, &live.speech, Some(named), "session.audio.input.transcription.model").map(|()| served.clone())
        };
        let diarizer = || self.server.speech.diarizer().availability();
        let next = match self.config.updated(event.get("session").unwrap_or(&Value::Null), model, diarizer) {
            Ok(next) => next,
            Err(refusal) => return self.refuse(refusal, event_id),
        };
        if next.rate != self.pcm.rate() {
            // What the item holds at the old rate goes in first.
            let mut rest = Vec::new();
            self.pcm.set_rate(next.rate, &mut rest);
            self.stream(rest);
        }
        self.config = next;
        // A changed context is prefilled now, while the item holds no audio;
        // with audio, it applies from the next item.
        let (open, _) = self.open_settings();
        let current = self.current();
        if current.samples_in == 0 && current.opened_with != open {
            current.cancel.store(true, Ordering::SeqCst);
            self.items.pop_back();
            self.open_next();
        }
        self.sync_turns();
        self.sync_speakers();
        let session = self.config.effective(&self.session_id);
        self.send(json!({ "type": "session.updated", "session": session }));
    }

    fn append(&mut self, event: &Map<String, Value>, event_id: Option<String>) {
        let Some(audio) = event.get("audio").and_then(Value::as_str) else {
            return self.refuse(Refusal::invalid("input_audio_buffer.append needs `audio`: base64 16-bit mono PCM", "audio"), event_id);
        };
        if audio.len() > MAX_APPEND_BYTES {
            return self.refuse(Refusal::invalid("`audio` carries more than 15 MiB of base64; send it over several appends", "audio"), event_id);
        }
        let rate = self.pcm.rate();
        // Spec speech/08: past the audio a session may hold not computed yet,
        // the append is refused before the resampler reads any of it.
        let limit = self.server.live().speech.max_queued_audio_seconds;
        let held = self.uncomputed();
        if held + samples_of(audio, rate) > limit as usize * TARGET_RATE as usize {
            let message = format!(
                "the session holds {:.1} s of audio not computed yet, and this append takes it past speech.max_queued_audio_seconds ({limit} s): it is refused; send it again once the session catches up, or pace the audio nearer real time",
                held as f64 / f64::from(TARGET_RATE)
            );
            return self.refuse(Refusal { code: "ignis_audio_queue_full", message, param: Some("audio".to_owned()) }, event_id);
        }
        let mut samples = Vec::new();
        let Ok(carried) = self.pcm.append(audio, &mut samples) else {
            return self.refuse(Refusal::invalid("`audio` is not base64", "audio"), event_id);
        };
        if self.turns.is_some() {
            // Under `server_vad` the VAD decides what reaches an item.
            return self.stream(samples);
        }
        self.begin_item();
        if self.admit(carried, rate) {
            self.stream(samples);
        }
    }

    /// The current item takes `count` samples at `rate`: its length grows,
    /// and whether they go on to the executor, which they do not once the
    /// item has failed or once it holds more than `speech.max_clip_seconds`.
    fn admit(&mut self, count: usize, rate: u32) -> bool {
        let limit = self.server.live().speech.max_clip_seconds;
        let item = self.current();
        item.samples_in += count as u64;
        item.seconds += count as f64 / f64::from(rate);
        if item.failure.is_some() {
            return false;
        }
        if item.seconds + f64::from(item.lead_ms) / 1000.0 > f64::from(limit) {
            let lead_ms = item.lead_ms;
            self.fail_current(Failure::TooLong(limit, lead_ms));
            return false;
        }
        true
    }

    /// The current item's first audio, at its first append or when the VAD
    /// heard speech: its id, its deadline (`server.stream_timeout`, per item
    /// from its first audio, ADR 0058), and when its first refresh falls due:
    /// an interval on, and not before the session's last refresh allows.
    fn begin_item(&mut self) {
        let live = self.server.live();
        let interval = Duration::from_millis(u64::from(live.speech.partial_interval_ms));
        let hold = self.refresh_hold;
        let diarized = self.speakers.as_ref().is_some_and(|speakers| !speakers.failed);
        let item = self.current();
        if item.id.is_none() {
            item.diarized = diarized;
            item.id = Some(format!("item_{}", random_suffix()));
            item.timeout = Timeout::of(&live, true);
            item.deadline = Some(item.timeout.from_now());
            self.next_partial = Some((Instant::now() + interval).max(hold));
        }
    }

    /// A `server_vad` item's leading silence, before its first sample:
    /// `speech.vad_leading_silence_ms` of zeros the model reads first (Q77).
    /// Its times, its `usage` and the samples it counts are the client's
    /// audio alone.
    fn lead_in(&mut self) {
        let lead_ms = self.server.live().speech.vad_leading_silence_ms;
        self.current().lead_ms = lead_ms;
        self.take_audio(ignis_speech::vad::leading_silence(lead_ms));
    }

    /// The session's next 16 kHz samples: the current item's under manual
    /// commit, the VAD's under `server_vad`, which gives an item the turns it
    /// finds (spec speech/05). The VAD reads every sample where it arrives, on
    /// this socket's task: a 32 ms frame never waits behind a window.
    fn stream(&mut self, samples: Vec<f32>) {
        self.heard += samples.len() as u64;
        let Some(turns) = self.turns.as_mut() else {
            return self.take_audio(samples);
        };
        let mut events = Vec::new();
        turns.push(&samples, &mut events);
        self.since_item.heard(&samples, self.server.live().speech.max_clip_seconds as usize * TARGET_RATE as usize);
        // The diarizer reads what the VAD reads (spec speech/08).
        if let Some(speakers) = self.speakers.as_mut().filter(|speakers| !speakers.failed) {
            speakers.pending.extend_from_slice(&samples);
            speakers.arrivals.heard(self.heard, Instant::now());
        }
        for event in events {
            self.turn(event);
        }
        self.diarize();
    }

    /// What the VAD found, as the Realtime events and the item's audio: an
    /// item begins at `speech_started`, its padded audio follows, and it is
    /// committed at `speech_stopped`; an idle timeout commits the silence.
    fn turn(&mut self, event: TurnEvent) {
        let millis = |sample: u64| sample * 1000 / u64::from(TARGET_RATE);
        match event {
            TurnEvent::Started { start } => {
                self.begin_item();
                self.current().start = start;
                let id = self.current().id.clone();
                self.send(json!({ "type": "input_audio_buffer.speech_started", "audio_start_ms": millis(start), "item_id": id }));
                self.lead_in();
            }
            TurnEvent::Audio(samples) => {
                self.give(samples);
                self.guard(false);
            }
            TurnEvent::Pause { .. } => self.guard(true),
            TurnEvent::Stopped { end } => {
                let id = self.current().id.clone();
                self.send(json!({ "type": "input_audio_buffer.speech_stopped", "audio_end_ms": millis(end), "item_id": id }));
                self.end_item();
                self.since_item.ended_at(end);
            }
            TurnEvent::TimedOut { start, end, audio } => {
                self.begin_item();
                self.current().start = start;
                let id = self.current().id.clone();
                self.send(json!({
                    "type": "input_audio_buffer.timeout_triggered",
                    "audio_start_ms": millis(start),
                    "audio_end_ms": millis(end),
                    "item_id": id,
                }));
                self.lead_in();
                self.give(audio);
                self.end_item();
                self.since_item.ended_at(end);
            }
        }
    }

    /// Samples the VAD gave the current item, counted at 16 kHz.
    fn give(&mut self, samples: Vec<f32>) {
        if self.admit(samples.len(), TARGET_RATE) {
            self.take_audio(samples);
        }
    }

    /// The current item's new 16 kHz samples: each whole window is queued at
    /// once, the rest waits for the window to fill, a partial or the commit.
    fn take_audio(&mut self, samples: Vec<f32>) {
        if self.current().failure.is_some() {
            return;
        }
        self.pending.extend(samples);
        while self.pending.len() >= WINDOW_SAMPLES {
            let window: Vec<f32> = self.pending.drain(..WINDOW_SAMPLES).collect();
            self.queue(Unit::Push(window));
            self.current().fresh_audio = true;
        }
        if !self.pending.is_empty() {
            self.current().fresh_audio = true;
        }
    }

    fn commit(&mut self, event_id: Option<String>) {
        if let Some(turns) = self.turns.as_mut() {
            // Under `server_vad` a commit ends the turn the VAD is in; the
            // stream goes on, so the resampler keeps its state.
            if turns.speaking() {
                let rest = turns.end_turn();
                self.give(rest);
                self.end_item();
                return self.since_item.reset(self.heard);
            }
            // No speech heard: as OpenAI's, the buffer is the audio appended
            // since the last item, and the commit makes it one, after the
            // leading silence, as long as the limit allows with it. The VAD
            // starts again where the commit left the stream, as after a
            // commit during speech: no padding and no idle time reach back
            // into this item.
            if self.since_item.samples.is_empty() {
                let empty = "the input audio buffer is empty: append audio before committing".to_owned();
                return self.refuse(Refusal { code: "input_audio_buffer_commit_empty", message: empty, param: None }, event_id);
            }
            turns.clear();
            let live = self.server.live();
            let limit = live.speech.max_clip_seconds as usize * TARGET_RATE as usize;
            let audio = self.since_item.take(limit.saturating_sub(ignis_speech::vad::ms_samples(live.speech.vad_leading_silence_ms) as usize));
            self.begin_item();
            self.current().start = self.heard - audio.len() as u64;
            self.lead_in();
            self.give(audio);
            return self.end_item();
        }
        if self.current().samples_in == 0 {
            let empty = "the input audio buffer is empty: append audio before committing".to_owned();
            return self.refuse(Refusal { code: "input_audio_buffer_commit_empty", message: empty, param: None }, event_id);
        }
        let mut rest = Vec::new();
        self.pcm.finish(&mut rest);
        self.stream(rest);
        self.end_item();
    }

    /// The current item ends: `committed`, a refresh of its in the worker
    /// stopped, its transcription queued (or its `failed`), and the next item
    /// opened.
    fn end_item(&mut self) {
        self.current().stop_refresh.store(true, Ordering::SeqCst);
        let previous = self.previous_item_id.take();
        let ending = Ending::new(
            Arc::clone(&self.server),
            Arc::clone(&self.current().cancel),
            self.server.speech.executor().name(),
            self.files.id.clone(),
            self.current().seconds,
            std::time::Instant::now(),
        );
        let item = self.current();
        item.ending = Some(ending);
        item.end = Some(item.start + item.samples_in);
        let id = item.id.clone().expect("an item with audio has its id");
        self.previous_item_id = Some(id.clone());
        self.send(json!({ "type": "input_audio_buffer.committed", "item_id": id, "previous_item_id": previous }));
        if self.current().failure.is_some() {
            let last = self.items.len() - 1;
            self.end_failed(last);
        } else {
            let rest = std::mem::take(&mut self.pending);
            self.queue(Unit::Commit(rest));
        }
        self.assign_speakers();
        self.open_next();
    }

    fn clear(&mut self) {
        let vad = self.turns.is_some();
        if let Some(turns) = self.turns.as_mut() {
            // The padding the VAD keeps, and a turn on, go as well, and the
            // audio heard since the last item: the buffer.
            turns.clear();
            self.since_item.reset(self.heard);
        }
        let current = self.current();
        if current.samples_in > 0 || current.id.is_some() {
            current.cancel.store(true, Ordering::SeqCst);
            current.stop_refresh.store(true, Ordering::SeqCst);
            let mut cleared = self.items.pop_back().expect("a current item");
            self.end_tail(&mut cleared);
            if !vad {
                // Under `server_vad` the stream goes on, and the VAD's clock
                // with it: the resampler keeps its state.
                self.pcm.reset();
            }
            self.open_next();
        }
        self.send(json!({ "type": "input_audio_buffer.cleared" }));
    }

    /// The current item fails before its commit: its audio goes nowhere from
    /// now, an `error` says why at once, and its commit answers `failed`.
    fn fail_current(&mut self, failure: Failure) {
        let mut error = failure.error();
        error["event_id"] = Value::Null;
        let item = self.current();
        item.failure = Some(failure);
        item.cancel.store(true, Ordering::SeqCst);
        self.pending.clear();
        self.next_partial = None;
        self.send(json!({ "type": "error", "error": error }));
    }

    /// The committed item at `index` fails: `failed` now.
    fn fail_committed(&mut self, index: usize, failure: Failure) {
        let item = &mut self.items[index];
        item.failure = Some(failure);
        item.cancel.store(true, Ordering::SeqCst);
        self.end_failed(index);
    }

    /// Send `failed` for the committed item at `index`, end it and drop it.
    fn end_failed(&mut self, index: usize) {
        let mut item = self.items.remove(index).expect("an item at the index");
        let failure = item.failure.clone().expect("a failed item");
        self.end_tail(&mut item);
        self.send(json!({
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": item.id,
            "content_index": 0,
            "error": failure.error(),
        }));
        if let Some(mut ending) = item.ending.take() {
            match &failure {
                Failure::Fault(fault) => ending.fail(failure.outcome(), Some(fault)),
                _ => ending.end(failure.outcome(), None, None),
            }
        }
        self.quiet_since = Instant::now();
    }

    /// An item ends: a tail it showed goes, for a session that opted in.
    fn end_tail(&mut self, item: &mut Item) {
        if self.config.provisional_tail && !item.tail.is_empty() {
            item.tail.clear();
            let id = item.id.clone();
            self.send(json!({ "type": "ignis.input_audio_transcription.provisional", "item_id": id, "content_index": 0, "text": "" }));
        }
    }

    /// A unit is out of its worker.
    fn done(&mut self, done: Done) {
        let seq = match done.of {
            Of::Item(seq) => {
                self.in_worker -= 1;
                self.in_worker_samples -= done.samples;
                seq
            }
            Of::Stream(number) => {
                self.speakers_in_worker_samples -= done.samples;
                // A step of a stream turned off or replaced since, still
                // running then, says nothing of the next.
                if self.speakers.as_ref().is_some_and(|speakers| speakers.number == number) {
                    if let Some(outcome) = done.outcome {
                        self.speakers_heard(outcome);
                    }
                }
                return;
            }
        };
        let Some(outcome) = done.outcome else { return };
        if let Outcome::Partial(_, cost) = &outcome {
            self.hold_refreshes(*cost);
        }
        // An item cleared or ended since: nobody is waiting for it.
        let Some(index) = self.items.iter().position(|item| item.seq == seq) else { return };
        match outcome {
            Outcome::Partial(partial, _) => {
                let opted_in = self.config.provisional_tail;
                let item = &mut self.items[index];
                if item.failure.is_some() {
                    return;
                }
                // With speakers, no text is fixed before the item ends, which a
                // speaker change can move (spec speech/08): the tail carries
                // all of it.
                let (stable, tail) = match item.diarized {
                    true => (String::new(), format!("{}{}", partial.stable, partial.tail)),
                    false => (partial.stable, partial.tail),
                };
                let delta = extension(&item.emitted, &stable).map(str::to_owned);
                let id = item.id.clone();
                if let Some(delta) = delta {
                    item.emitted = stable;
                    self.send(json!({ "type": "conversation.item.input_audio_transcription.delta", "item_id": id, "content_index": 0, "delta": delta }));
                }
                let item = &mut self.items[index];
                if opted_in && item.tail != tail {
                    item.tail = tail.clone();
                    self.send(json!({ "type": "ignis.input_audio_transcription.provisional", "item_id": id, "content_index": 0, "text": tail }));
                }
            }
            Outcome::Transcript(transcript, computed) => {
                let item = &mut self.items[index];
                if item.failure.is_some() {
                    self.items.remove(index);
                    return;
                }
                item.transcript = Some((transcript, computed));
                // Its speaker comes from the diarizer's frames over its end,
                // which no more audio may bring.
                if item.diarized && item.speaker.is_none() {
                    self.want_tail();
                }
                self.complete_ready();
            }
            Outcome::Failed(failure) => {
                if self.items[index].failure.is_some() {
                    return;
                }
                if self.items[index].ending.is_some() {
                    self.fail_committed(index, failure);
                } else {
                    self.fail_current(failure);
                }
            }
            Outcome::Opened(_) | Outcome::Frames(..) | Outcome::Tail(_) | Outcome::Unheard(_) => unreachable!("a speech unit answers for its item"),
        }
    }

    /// What a unit of the current speakers' stream did.
    fn speakers_heard(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Opened(due) => self.speakers_opened(due),
            Outcome::Frames(frames, due, cost) => self.heard_frames(frames, due, cost),
            Outcome::Tail(frames) => self.heard_tail(frames),
            Outcome::Unheard(why) => self.speakers_failed(why),
            Outcome::Partial(..) | Outcome::Transcript(..) | Outcome::Failed(_) => unreachable!("a speakers' unit answers what the diarizer did"),
        }
    }

    /// Every committed item ready, in order: its transcript in, and its
    /// speaker known when it has one. An item waiting for its speaker holds
    /// no later one back: a later item ends later, so its speaker comes no
    /// sooner.
    fn complete_ready(&mut self) {
        let mut index = 0;
        while index < self.items.len() {
            let item = &self.items[index];
            if item.transcript.is_none() || (item.diarized && item.speaker.is_none()) {
                index += 1;
                continue;
            }
            let item = self.items.remove(index).expect("an item at the index");
            self.complete(item);
        }
    }

    /// A committed item's text and its end: what its deltas have not sent,
    /// its speaker after its last delta (spec speech/08), its language
    /// (spec speech/07), then `completed`.
    fn complete(&mut self, mut item: Item) {
        let (transcript, computed) = item.transcript.take().expect("a transcript in");
        match extension(&item.emitted, &transcript.text) {
            Some(delta) => self.send(json!({
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": item.id,
                "content_index": 0,
                "delta": delta,
            })),
            // An item taken for a copy of its context completes empty, whatever its deltas sent
            // (`speech.context_copy_words`, `speech.context_copy_edits`): no revision.
            None if !transcript.text.is_empty() && !transcript.text.starts_with(item.emitted.as_str()) => {
                // The executor revised stable text, which spec 01 §4
                // says it never does: the transcript is still its own.
                tracing::warn!(
                    name: "ignis.speech.stable_text_revised",
                    variant = %self.files.id,
                    "a transcript does not begin with the stable text its deltas sent, so they do not join to it"
                );
            }
            None => {}
        }
        self.end_tail(&mut item);
        if let Some(Some(speaker)) = item.speaker {
            self.send(json!({ "type": "ignis.input_audio_transcription.speaker", "item_id": item.id, "content_index": 0, "speaker": speaker }));
        }
        // The model's label, as the item's log line reads it, by its code:
        // none for `language None` (no speech) or a label outside the 30.
        if self.config.item_language {
            if let Some(language) = transcript.language.as_deref().and_then(languages::find) {
                self.send(json!({ "type": "ignis.input_audio_transcription.language", "item_id": item.id, "content_index": 0, "language": language.code }));
            }
        }
        self.send(json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item.id,
            "content_index": 0,
            "transcript": transcript.text,
            "usage": { "type": "duration", "seconds": (item.seconds * 1000.0).round() / 1000.0 },
        }));
        if let Some(mut ending) = item.ending.take() {
            ending.end(SpeechOutcome::Completed, Some(&transcript), Some(computed));
        }
        self.quiet_since = Instant::now();
    }

    /// The length guard (spec speech/08): once the current item holds the
    /// limit in force, the session's or the server's, it ends at the VAD's
    /// next pause, or `speech.utterance_cut_search_ms` past the limit at the
    /// latest, where the audio is; the speech goes on in a new item.
    fn guard(&mut self, paused: bool) {
        let live = self.server.live();
        let Some(limit) = self.config.utterance_limit(live.speech.max_utterance_seconds) else { return };
        let item = self.items.back().expect("a current item");
        if item.id.is_none() || item.failure.is_some() {
            return;
        }
        let held_ms = item.samples_in * 1000 / u64::from(TARGET_RATE);
        let limit_ms = u64::from(limit) * 1000;
        if held_ms >= limit_ms && (paused || held_ms >= limit_ms + u64::from(live.speech.utterance_cut_search_ms)) {
            let at = item.start + item.samples_in;
            self.cut_item(at, false);
        }
    }

    /// End the current item at sample `at` of the session's clock, as the VAD
    /// ends one at a silence, and go on with the speech in a new item from
    /// there (spec speech/08): `speech_stopped` at `at`, `committed`, then
    /// `speech_started` at `at` for the new one, which has no prefix padding
    /// (it would take the speech before) and the leading silence. At a
    /// speaker change (`change`) the audio past `at`, heard already, is the
    /// new item's, and the item is decoded again from its audio before `at`;
    /// the length guard cuts where the audio is, so none is past it.
    fn cut_item(&mut self, at: u64, change: bool) {
        let millis = |sample: u64| sample * 1000 / u64::from(TARGET_RATE);
        let item = self.items.back().expect("a current item");
        let (id, start, given, lead) = (item.id.clone(), item.start, item.samples_in, ignis_speech::vad::ms_samples(item.lead_ms) as usize);
        self.send(json!({ "type": "input_audio_buffer.speech_stopped", "audio_end_ms": millis(at), "item_id": id }));
        let moved = self.since_item.range(at, start + given);
        let kept = (at - start) as usize;
        if change {
            // What the item's session was pushed, and what still waits to be.
            let queued = (lead + given as usize).saturating_sub(self.pending.len());
            let fed = lead + kept;
            match fed.checked_sub(queued) {
                Some(more) => self.pending.truncate(more),
                None => self.pending.clear(),
            }
            self.queue(Unit::Cut(fed));
        }
        let item = self.current();
        item.samples_in = kept as u64;
        item.seconds = kept as f64 / f64::from(TARGET_RATE);
        self.since_item.ended_at(at);
        self.end_item();
        self.begin_item();
        self.current().start = at;
        let id = self.current().id.clone();
        self.send(json!({ "type": "input_audio_buffer.speech_started", "audio_start_ms": millis(at), "item_id": id }));
        self.lead_in();
        self.give(moved);
    }

    /// The speakers `session.update` set: at once while the current item
    /// holds no audio, else from the next item, as a switch of turn detection
    /// holds. Speakers that start read the `diarizer` group, and their stream
    /// starts at the audio heard so far; speakers that stop leave the items
    /// waiting for theirs without one.
    fn sync_speakers(&mut self) {
        let wanted = self.config.diarization;
        if self.speakers.as_ref().map(|speakers| speakers.mode) == wanted || self.current().samples_in > 0 {
            return;
        }
        if let Some(gone) = self.speakers.take() {
            gone.cancel.store(true, Ordering::SeqCst);
            gone.log(&self.session_id);
            self.unheard();
        }
        let Some(mode) = wanted else { return };
        let live = self.server.live();
        self.streams += 1;
        self.speakers = Some(Diarized {
            mode,
            number: self.streams,
            stream: Arc::default(),
            origin: self.heard,
            sent: 0,
            pending: Vec::new(),
            due: None,
            stepping: false,
            tailing: false,
            frames: Speakers::new(f64::from(live.diarizer.min_speaker_seconds) / 1000.0),
            tail: None,
            cancel: Arc::new(AtomicBool::new(false)),
            failed: false,
            heard_speakers: 0,
            steps: 0,
            step_total: Duration::ZERO,
            step_max: Duration::ZERO,
            tails: 0,
            arrivals: Arrivals::new(self.heard, Instant::now()),
            lag_max: Duration::ZERO,
            items: 0,
        });
        self.queue_speakers(SpeakersUnit::Open(StreamOpen::of(&live.diarizer, mode)));
    }

    /// The audio the session holds that is not computed yet, each sample
    /// counted once however many consumers hold it: the larger of the speech
    /// model's (the current item's samples not yet queued, and its units'
    /// in the speech worker) and the speakers' (their samples not yet sent,
    /// and their units' on the diarizer's thread).
    fn uncomputed(&self) -> usize {
        let speakers = self.speakers.as_ref().map_or(0, |speakers| speakers.pending.len());
        (self.pending.len() + self.in_worker_samples).max(speakers + self.speakers_in_worker_samples)
    }

    /// Queue the diarizer's next step once the audio it waits for is heard:
    /// one at a time, each answering when the next falls due.
    fn diarize(&mut self) {
        let Some(speakers) = self.speakers.as_mut() else { return };
        let heard = speakers.sent + speakers.pending.len() as u64;
        if speakers.stepping || speakers.failed || !speakers.due.is_some_and(|due| heard >= due) {
            return;
        }
        let samples = std::mem::take(&mut speakers.pending);
        speakers.sent += samples.len() as u64;
        speakers.stepping = true;
        self.queue_speakers(SpeakersUnit::Diarize(samples));
    }

    /// When the speakers' tail falls due (spec speech/08): an item's
    /// transcript is in and its speaker waits for frames past the steps' and
    /// the last tail's, no step is on the diarizer's thread (one that can run is), and
    /// the stream has stalled. `None` while none is wanted.
    fn tail_due(&self) -> Option<Instant> {
        let speakers = self.speakers.as_ref()?;
        if speakers.tailing || speakers.stepping || speakers.failed {
            return None;
        }
        let covered = speakers.frames.len().max(speakers.tail.as_ref().map_or(0, Speakers::len));
        let waits = self.items.iter().any(|item| {
            item.diarized && item.speaker.is_none() && item.transcript.is_some() && item.end.is_some_and(|end| speakers.frame(end) > covered)
        });
        waits.then(|| speakers.stalls_at()).flatten()
    }

    /// Score what the stream heard past its last step once the tail falls
    /// due: a file's end, a pause in what the client sends, a commit with no
    /// more audio (spec speech/08). In a stream that flows, the next step,
    /// with its look-ahead, covers the item's end instead. On a copy of the
    /// stream, one at a time.
    fn want_tail(&mut self) {
        if !self.tail_due().is_some_and(|at| at <= Instant::now()) {
            return;
        }
        let speakers = self.speakers.as_mut().expect("the speakers are on");
        let samples = std::mem::take(&mut speakers.pending);
        speakers.sent += samples.len() as u64;
        speakers.tailing = true;
        self.queue_speakers(SpeakersUnit::Tail(samples));
    }

    /// The diarizer's stream is open: its first step waits for `due` samples.
    fn speakers_opened(&mut self, due: usize) {
        let Some(speakers) = self.speakers.as_mut() else { return };
        speakers.due = Some(due as u64);
        self.diarize();
        self.want_tail();
    }

    /// A step's frames: a speaker change in the current item cuts it, and the
    /// items they cover the end of have their speaker.
    fn heard_frames(&mut self, frames: Frames, due: usize, cost: Duration) {
        let Some(speakers) = self.speakers.as_mut() else { return };
        speakers.stepping = false;
        if !frames.probs.is_empty() {
            speakers.steps += 1;
            speakers.step_total += cost;
            speakers.step_max = speakers.step_max.max(cost);
        }
        speakers.due = Some(due as u64);
        if !frames.probs.is_empty() {
            let first = speakers.sample(speakers.frames.len()) + 1;
            if let Some(at) = speakers.arrivals.arrived(first) {
                speakers.lag_max = speakers.lag_max.max(at.elapsed());
            }
        }
        speakers.frames.push(&frames.probs);
        let scored = speakers.sample(speakers.frames.len());
        speakers.arrivals.scored(scored);
        if speakers.tail.as_ref().is_some_and(|tail| tail.len() <= speakers.frames.len()) {
            speakers.tail = None;
        }
        self.speaker_change();
        self.assign_speakers();
        self.complete_ready();
        self.diarize();
        self.want_tail();
    }

    /// A tail's frames: scored with no look-ahead, so they cut nothing and
    /// join no later step's, but they give the items they cover the end of
    /// their speaker, those ended since among them, until the steps reach as
    /// far.
    fn heard_tail(&mut self, tail: Frames) {
        let Some(speakers) = self.speakers.as_mut() else { return };
        speakers.tailing = false;
        if !tail.probs.is_empty() {
            speakers.tails += 1;
        }
        let mut heard = speakers.frames.clone();
        heard.push(tail.probs.get(heard.len().saturating_sub(tail.first)..).unwrap_or_default());
        speakers.tail = Some(heard);
        self.assign_speakers();
        self.complete_ready();
        self.want_tail();
    }

    /// The diarizer failed: an `error` says so, and the session goes on
    /// without speakers, the items waiting for theirs completed without one.
    fn speakers_failed(&mut self, why: String) {
        let Some(speakers) = self.speakers.as_mut().filter(|speakers| !speakers.failed) else { return };
        speakers.failed = true;
        speakers.pending = Vec::new();
        speakers.cancel.store(true, Ordering::SeqCst);
        tracing::warn!(name: "ignis.speech.diarizer_failed", session = %self.session_id, reason = %why, "the diarizer failed; the session goes on without speakers");
        let message = format!("the diarizer failed ({why}); the session goes on without speakers");
        self.send(json!({ "type": "error", "error": { "type": "server_error", "code": "ignis_diarizer_failed", "message": message, "param": null, "event_id": null } }));
        self.unheard();
    }

    /// The items waiting for a speaker that no frame will give now get none.
    fn unheard(&mut self) {
        for item in self.items.iter_mut().filter(|item| item.diarized && item.speaker.is_none()) {
            item.speaker = Some(None);
        }
        self.complete_ready();
    }

    /// The committed items whose end the frames cover (the steps', else the
    /// latest tail's) take their speaker: the one holding most of their
    /// speech frames (spec speech/08).
    fn assign_speakers(&mut self) {
        let Some(speakers) = self.speakers.as_ref() else { return };
        let mut given = 0u32;
        let mut items = 0;
        for item in self.items.iter_mut().filter(|item| item.diarized && item.speaker.is_none()) {
            let Some(end) = item.end else { continue };
            let (from, to) = (speakers.frame(item.start), speakers.frame(end));
            let covering = [Some(&speakers.frames), speakers.tail.as_ref()].into_iter().flatten().find(|frames| frames.len() >= to);
            if let Some(frames) = covering {
                item.speaker = Some(frames.speaker_of(from, to));
                given |= item.speaker.flatten().map_or(0, |speaker| 1 << (speaker as u32));
                items += 1;
            }
        }
        if let Some(speakers) = self.speakers.as_mut() {
            speakers.heard_speakers |= given;
            speakers.items += items;
        }
    }

    /// A change of speaker inside the current item ends it at the change, the
    /// first frame of the new speaker's run (spec speech/08), as many times
    /// as the frames hold one.
    fn speaker_change(&mut self) {
        loop {
            let Some(speakers) = self.speakers.as_ref() else { return };
            let speaking = self.turns.as_ref().is_some_and(|turns| turns.speaking());
            let item = self.items.back().expect("a current item");
            if !speaking || !item.diarized || item.id.is_none() || item.failure.is_some() {
                return;
            }
            let Some((frame, _)) = speakers.frames.change(speakers.frame(item.start), speakers.frames.len()) else { return };
            let at = speakers.sample(frame);
            if at <= item.start || at >= item.start + item.samples_in {
                return;
            }
            self.cut_item(at, true);
        }
    }

    /// A refresh of the session's took `cost` on the worker: the next waits
    /// at least `speech.partial_cost_factor` times that from when this one was
    /// queued, never less than `speech.partial_interval_ms` (Q76), so the
    /// refreshes take at most about one part in that many of the worker
    /// however much each costs. Read live, at each refresh.
    fn hold_refreshes(&mut self, cost: Duration) {
        self.refresh_hold = self.refresh_queued + cost * self.server.live().speech.partial_cost_factor;
        if let Some(at) = self.next_partial.as_mut() {
            *at = (*at).max(self.refresh_hold);
        }
    }

    /// The timers: the live transcript's refresh, the items' deadlines, and
    /// the speakers' tail.
    fn tick(&mut self) {
        let now = Instant::now();
        if self.next_partial.is_some_and(|at| at <= now) {
            let live = self.server.live();
            self.next_partial = Some(now + Duration::from_millis(u64::from(live.speech.partial_interval_ms)));
            // Best effort (Q74): with anything of this socket's still in the
            // speech worker (a window, the last refresh, a commit) the refresh
            // is skipped, never queued behind it. A diarizer step on its own
            // thread holds nothing back (spec speech/08).
            let item = self.items.back().expect("a current item");
            if self.in_worker == 0 && item.failure.is_none() && item.fresh_audio {
                let rest = std::mem::take(&mut self.pending);
                self.queue(Unit::Partial(rest));
                self.refresh_queued = now;
                self.current().fresh_audio = false;
            }
        }
        let mut index = 0;
        while index < self.items.len() {
            let item = &self.items[index];
            if item.failure.is_none() && item.deadline.is_some_and(|deadline| deadline <= now) {
                let failure = Failure::TimedOut(item.timeout);
                if item.ending.is_some() {
                    self.fail_committed(index, failure);
                    continue;
                }
                self.fail_current(failure);
            }
            index += 1;
        }
        self.want_tail();
    }

    /// The live `server.socket_idle_close`, when the socket has been idle at
    /// least that long: no committed item being transcribed, and nothing
    /// from the client since.
    fn idle_past(&self) -> Option<Duration> {
        let after = self.server.live().socket_idle_close;
        let idle = self.items.len() == 1;
        (idle && !after.is_zero() && self.quiet_since.elapsed() >= after).then_some(after)
    }

    /// When to look at the timers next: the next refresh, the nearest
    /// deadline, the speakers' tail, the idle time, and at least once an
    /// [`IDLE_TICK`](crate::idle::IDLE_TICK), so a live change of a knob
    /// reaches an open socket within one.
    fn next_look(&self) -> Instant {
        let mut at = Instant::now() + crate::idle::IDLE_TICK;
        if let Some(partial) = self.next_partial {
            at = at.min(partial);
        }
        for item in &self.items {
            if let (None, Some(deadline)) = (&item.failure, item.deadline) {
                at = at.min(deadline);
            }
        }
        if let Some(tail) = self.tail_due() {
            at = at.min(tail);
        }
        let after = self.server.live().socket_idle_close;
        if !after.is_zero() && self.items.len() == 1 {
            at = at.min(self.quiet_since + after);
        }
        at
    }
}

/// Run a unit's next step on the speech worker's thread: `More` for a commit
/// with steps left; done with `None` when it has nothing to say: an open, a
/// push or a cut that went through, a refresh `stop` stopped (Q79), or a unit
/// after a failure that left no session (the failure already answered).
fn run(unit: &mut Unit, slot: &Mutex<Slot>, executor: &dyn SpeechExecutor, files: &SpeechFiles, stop: &AtomicBool) -> Progress {
    let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
    let failed = |slot: &mut Slot, why: String| {
        slot.session = None;
        Progress::Done(Some(Outcome::Failed(Failure::Fault(Fault::Executor(why)))))
    };
    let began = std::time::Instant::now();
    match unit {
        Unit::Open(open, load) => {
            slot.session = None;
            slot.computed = Duration::ZERO;
            match open_loaded(executor, files, *load, std::mem::take(open)) {
                Ok(session) => {
                    slot.session = Some(session);
                    Progress::Done(None)
                }
                Err(fault) => Progress::Done(Some(Outcome::Failed(Failure::Fault(fault)))),
            }
        }
        Unit::Push(samples) => {
            let Some(session) = slot.session.as_mut() else { return Progress::Done(None) };
            let pushed = session.push(samples);
            slot.computed += began.elapsed();
            match pushed {
                Ok(()) => Progress::Done(None),
                Err(error) => failed(&mut *slot, error.0),
            }
        }
        Unit::Partial(samples) => {
            let Some(session) = slot.session.as_mut() else { return Progress::Done(None) };
            // The audio goes in whatever happens: the commit has the rest.
            let partial = push(session.as_mut(), samples).and_then(|()| session.partial(&|| stop.load(Ordering::SeqCst)));
            let cost = began.elapsed();
            slot.computed += cost;
            match partial {
                Ok(partial) => Progress::Done(partial.map(|partial| Outcome::Partial(partial, cost))),
                Err(error) => failed(&mut *slot, error.0),
            }
        }
        Unit::Commit(samples) => {
            let Some(session) = slot.session.as_mut() else { return Progress::Done(None) };
            // The rest first, a step of its own, then the commit's steps.
            let step = match samples.is_empty() {
                true => session.commit_step(),
                false => push(session.as_mut(), &std::mem::take(samples)).map(|()| None),
            };
            slot.computed += began.elapsed();
            match step {
                Ok(None) => Progress::More,
                Ok(Some(transcript)) => {
                    slot.session = None;
                    Progress::Done(Some(Outcome::Transcript(transcript, slot.computed)))
                }
                Err(error) => failed(&mut *slot, error.0),
            }
        }
        Unit::Cut(at) => {
            let Some(session) = slot.session.as_mut() else { return Progress::Done(None) };
            let cut = session.cut(*at);
            slot.computed += began.elapsed();
            match cut {
                Ok(()) => Progress::Done(None),
                Err(error) => failed(&mut *slot, error.0),
            }
        }
    }
}

/// Run a speakers' unit on the diarizer's thread: it answers what the
/// diarizer did, or why it failed. Only the stream's lock is taken, never the
/// speech model's [`Slot`].
fn run_speakers(unit: &mut SpeakersUnit, stream: &Mutex<Option<Box<dyn DiarizerStream>>>, diarizer: &dyn DiarizerExecutor) -> Progress {
    let mut slot = stream.lock().unwrap_or_else(PoisonError::into_inner);
    let began = std::time::Instant::now();
    let tail = matches!(unit, SpeakersUnit::Tail(_));
    match unit {
        SpeakersUnit::Open(open) => {
            *slot = None;
            match diarizer.open(*open) {
                Ok(stream) => {
                    let due = stream.samples_needed();
                    *slot = Some(stream);
                    Progress::Done(Some(Outcome::Opened(due)))
                }
                Err(error) => Progress::Done(Some(Outcome::Unheard(error.0))),
            }
        }
        SpeakersUnit::Diarize(samples) | SpeakersUnit::Tail(samples) => {
            let Some(stream) = slot.as_mut() else {
                return Progress::Done(Some(Outcome::Unheard("the diarizer's stream is not open".to_owned())));
            };
            stream.push(&std::mem::take(samples));
            let scored = match tail {
                true => stream.tail().map(Outcome::Tail),
                false => stream.step().map(|frames| Outcome::Frames(frames.unwrap_or_default(), stream.samples_needed(), began.elapsed())),
            };
            match scored {
                Ok(outcome) => Progress::Done(Some(outcome)),
                Err(error) => {
                    *slot = None;
                    Progress::Done(Some(Outcome::Unheard(error.0)))
                }
            }
        }
    }
}

/// The 16 kHz samples base64 `audio` of 16-bit PCM at `rate` makes, read off
/// its length before it is decoded.
fn samples_of(audio: &str, rate: u32) -> usize {
    let bytes = audio.trim_end_matches('=').len() * 3 / 4;
    bytes / 2 * TARGET_RATE as usize / rate as usize
}

/// Push `samples` when there are any: a partial or a commit with nothing new
/// pushes nothing.
fn push(session: &mut dyn SpeechSession, samples: &[f32]) -> Result<(), super::executor::SpeechError> {
    if samples.is_empty() {
        return Ok(());
    }
    session.push(samples)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::Instant;

    use super::{Arrivals, SinceItem};

    /// A frame's first sample keeps its arrival until a step scores it,
    /// however long its geometry makes it wait (the file geometry's latency
    /// can be set past a minute): 90 s of audio heard, 100 ms a piece, the
    /// first piece's arrival is still there. Once the steps scored the samples
    /// before 800,000 (50 s), the entries before the one at it go: the
    /// earliest kept is that one.
    #[test]
    fn a_frames_arrival_is_kept_until_a_step_scores_it() {
        let start = Instant::now();
        let at = |piece: u64| start + Duration::from_millis(100 * piece);
        let mut arrivals = Arrivals::new(0, start);
        for piece in 1..=900 {
            arrivals.heard(piece * 1_600, at(piece));
        }
        assert_eq!(arrivals.arrived(1), Some(at(1)), "the first sample, 90 s on");
        assert_eq!(arrivals.arrived(800_001), Some(at(501)));
        arrivals.scored(800_000);
        assert_eq!(arrivals.arrived(800_001), Some(at(501)), "what no step scored is kept");
        assert_eq!(arrivals.arrived(1), Some(at(500)), "the entries before the one at the scored sample went");
    }

    /// The audio heard since the last item stays on the VAD's clock: one
    /// started at a later sample (`server_vad` switched on mid-session), an
    /// item's end, the cap and a commit's take all keep it there.
    #[test]
    fn the_audio_since_the_last_item_stays_on_the_vads_clock() {
        let mut since = SinceItem::default();
        since.reset(1_000);
        since.heard(&[1.0; 300], 10_000);
        // An item ended at sample 1,200: the 100 samples after it are what is left.
        since.ended_at(1_200);
        assert_eq!((since.from, since.samples.len()), (1_200, 100));
        since.heard(&[2.0; 50], 120);
        assert_eq!((since.from, since.samples.len()), (1_230, 120), "at most the cap, the oldest dropped");
        let taken = since.take(100);
        assert_eq!(taken.len(), 100);
        assert_eq!(taken.iter().filter(|&&sample| sample == 2.0).count(), 50, "the last of it");
        assert!(since.samples.is_empty());
        assert_eq!(since.from, 1_350, "empty from the end of what was heard");
        since.ended_at(1_300);
        assert_eq!(since.from, 1_350, "an end before it drops nothing");
    }
}
