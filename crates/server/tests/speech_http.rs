//! The speech model in the server and `POST /v1/audio/transcriptions` (spec
//! speech/02, GitHub #352) over the real router, on a scripted executor
//! ([`FakeExecutor`]) and catalog entries of our own: the model's states and
//! `GET /v1/models`, which model a request names, every field honoured,
//! inert or refused, the answers, the stream, the length limit, admission,
//! deadlines, and the route's place outside the serving gate.
//!
//! The log and metrics of a transcription are `speech_log.rs`'s: tracing
//! caches a callsite's interest process-wide, so a log test lives in a
//! binary of its own.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::speech::executor::{FakeExecutor, SpeechExecutor};
use ignis_server::speech::worker::Step;
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/mod.rs"]
mod support;
use support::speech::{self, Part, SPEECH_MODEL, TEXT_MODEL};

fn fake() -> Arc<FakeExecutor> {
    Arc::new(FakeExecutor::saying(&["hello", "from", "ignis"]))
}

fn app(server: &Server) -> axum::Router {
    server.app()
}

/// The status, headers and JSON of an answer.
async fn answer(app: &axum::Router, request: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let (status, headers) = (response.status(), response.headers().clone());
    let bytes = speech::bytes(response).await;
    (status, headers, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// A transcription of the tone as a WAV, with `extra` text parts.
async fn transcribe(app: &axum::Router, extra: &[(&str, &str)]) -> (StatusCode, axum::http::HeaderMap, Value) {
    let mut parts = vec![speech::Part::File("file", speech::WAV)];
    parts.extend(extra.iter().map(|(name, value)| Part::Text(name, value)));
    answer(app, speech::transcription(&parts)).await
}

fn models_request() -> Request<Body> {
    Request::builder().uri("/v1/models").body(Body::empty()).unwrap()
}

async fn models(app: &axum::Router) -> Value {
    answer(app, models_request()).await.2
}

// ── AC 1, 2, 3: the model, its states, `GET /v1/models` ──────────────────

/// AC 2, 3: with its files on disk the speech model is listed after
/// `data[0]`, which stays the text model, as `kind: "speech"`, `active: true`;
/// the top-level `speech` object says it is available.
#[tokio::test]
async fn an_available_speech_model_is_listed_after_the_text_model() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let list = models(&app(&server)).await;
    let data = list["data"].as_array().unwrap();
    assert_eq!(data[0]["id"], TEXT_MODEL);
    assert_eq!(data[0]["active"], true);
    assert_eq!(data[0]["kind"], "text");
    assert_eq!(data[1]["id"], SPEECH_MODEL);
    assert_eq!(data[1]["kind"], "speech");
    assert_eq!(data[1]["active"], true);
    assert_eq!(data[1]["available"], true);
    // A test server has no diarizer (`speech_diarizer_models.rs` is the member's).
    assert_eq!(list["speech"]["available"], true);
    assert!(list["speech"].get("reason").is_none());
    // The compat names are not listed.
    assert!(data.iter().all(|entry| entry["id"] != "whisper-1"));
}

/// AC 2, 3: each unavailable state is listed nowhere but the `speech` object,
/// which always carries its reason: off; files missing, naming the command
/// that fetches them; no AVX2.
#[tokio::test]
async fn an_unavailable_speech_model_is_not_listed_and_says_why() {
    let reasons = |list: &Value| (list["data"].as_array().unwrap().len(), list["speech"]["available"].clone(), list["speech"]["reason"].as_str().unwrap_or_default().to_owned());

    let (off, _dir) = speech::server(fake(), |group| group.model = "off".to_owned());
    let (len, available, reason) = reasons(&models(&app(&off)).await);
    assert_eq!((len, available), (1, json!(false)));
    assert!(reason.contains("turned off") && reason.contains("speech.model"), "{reason}");

    let (missing, dir) = speech::server(fake(), |_| {});
    std::fs::remove_file(dir.join(format!("{SPEECH_MODEL}-decoder.gguf"))).unwrap();
    let (len, available, reason) = reasons(&models(&app(&missing)).await);
    assert_eq!((len, available), (1, json!(false)));
    assert!(reason.contains(&format!("model download {SPEECH_MODEL}")), "{reason}");

    let executor = fake();
    executor.set_unavailable(Some("the CPU lacks AVX2, which the speech model needs"));
    let (no_avx2, _dir) = speech::server(executor, |_| {});
    let (len, available, reason) = reasons(&models(&app(&no_avx2)).await);
    assert_eq!((len, available), (1, json!(false)));
    assert!(reason.contains("AVX2"), "{reason}");
}

/// A server built with no speech model at all (a plain test server) carries
/// the object too: never missing, never a model nobody can run.
#[tokio::test]
async fn a_server_without_a_speech_slot_still_carries_the_speech_object() {
    let scheduler = ignis_core::ConcreteScheduler::with_config(
        ignis_core::SchedulerConfig { model: TEXT_MODEL.into(), ..Default::default() },
        Arc::new(ignis_core::MockCompute::new()),
    );
    let plain = Server::new(ignis_server::engine::Engine::new(Box::new(scheduler)), Box::new(ignis_server::template::SimpleTemplateProvider));
    let list = models(&plain.app()).await;
    assert_eq!(list["speech"]["available"], false);
    assert!(list["speech"]["reason"].is_string());
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
}

/// AC 1: nothing downloads by itself: a missing model stays missing after a
/// request for it.
#[tokio::test]
async fn nothing_is_downloaded_by_asking() {
    let (server, dir) = speech::server(fake(), |group| group.model = "qwen3-asr-0.6b".to_owned());
    let before = std::fs::read_dir(&dir).unwrap().count();
    let (status, _, body) = transcribe(&app(&server), &[]).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), before);
}

/// AC 2: an off or missing model answers `503 speech_unavailable` naming the
/// reason, and with no `Retry-After`: waiting changes nothing.
#[tokio::test]
async fn the_route_answers_an_unavailable_model_503_without_retry_after() {
    let (off, _dir) = speech::server(fake(), |group| group.model = "off".to_owned());
    let (status, headers, body) = transcribe(&app(&off), &[]).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "speech_unavailable");
    assert!(body["error"]["message"].as_str().unwrap().contains("turned off"), "{body}");
    assert!(!headers.contains_key("retry-after"));

    let (missing, dir) = speech::server(fake(), |_| {});
    std::fs::remove_file(dir.join(format!("{SPEECH_MODEL}-encoder.gguf"))).unwrap();
    let (status, headers, body) = transcribe(&app(&missing), &[]).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"]["message"].as_str().unwrap().contains("model download"), "{body}");
    assert!(!headers.contains_key("retry-after"));
}

// ── AC 4: which model a request names ───────────────────────────────────

/// AC 4: no `model`, the served id and every compat name reach the speech
/// model; the compat list is changeable.
#[tokio::test]
async fn the_speech_model_answers_to_no_model_its_id_and_the_compat_names() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let app = app(&server);
    for model in [None, Some(SPEECH_MODEL), Some("whisper-1"), Some("gpt-4o-transcribe"), Some("gpt-4o-mini-transcribe"), Some("gpt-transcribe"), Some("gpt-live-transcribe")] {
        let extra: Vec<(&str, &str)> = model.map(|model| ("model", model)).into_iter().collect();
        let (status, _, body) = transcribe(&app, &extra).await;
        assert_eq!(status, StatusCode::OK, "{model:?}: {body}");
        assert_eq!(body["text"], "hello from ignis");
    }
    let (changed, _dir) = speech::server(fake(), |group| group.compat_names = vec!["scribe".to_owned()]);
    let changed = self::app(&changed);
    assert_eq!(transcribe(&changed, &[("model", "scribe")]).await.0, StatusCode::OK);
    assert_eq!(transcribe(&changed, &[("model", "whisper-1")]).await.0, StatusCode::NOT_FOUND);
}

/// AC 4: a text model's id is `409 model_kind_mismatch` naming both kinds;
/// anything else is `404 model_not_found`.
#[tokio::test]
async fn a_text_models_id_is_a_kind_mismatch_and_an_unknown_one_is_not_found() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let app = app(&server);
    let (status, _, body) = transcribe(&app, &[("model", TEXT_MODEL)]).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_kind_mismatch");
    assert_eq!(body["error"]["param"], "model");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains(TEXT_MODEL) && message.contains("a text model") && message.contains("speech"), "{message}");
    let (status, _, body) = transcribe(&app, &[("model", "no-such-model")]).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
    // A catalog speech entry the config does not serve is not this server's.
    assert_eq!(transcribe(&app, &[("model", "qwen3-asr-0.6b")]).await.0, StatusCode::NOT_FOUND);
}

/// `/v1/audio/translations` is not served: 404, like the other routes not
/// served by choice (AC 17).
#[tokio::test]
async fn translations_are_not_served() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let mut request = speech::transcription(&[Part::File("file", speech::WAV)]);
    *request.uri_mut() = "/v1/audio/translations".parse().unwrap();
    assert_eq!(app(&server).oneshot(request).await.unwrap().status(), StatusCode::NOT_FOUND);
}

// ── AC 5, 10: the audio ─────────────────────────────────────────────────

/// AC 5: WAV, FLAC, MP3 and AAC are transcribed, and the Ogg and Matroska
/// containers demuxed; the executor is pushed 16 kHz samples, about a second
/// of them.
#[tokio::test]
async fn every_accepted_format_is_transcribed() {
    for (name, bytes) in [
        ("wav", speech::WAV),
        ("flac", speech::FLAC),
        ("mp3", speech::MP3),
        ("aac", speech::AAC),
        ("m4a", speech::M4A),
        ("ogg", speech::OGG_FLAC),
        ("matroska", speech::MATROSKA),
    ] {
        let executor = fake();
        let (server, _dir) = speech::server(executor.clone(), |_| {});
        let (status, _, body) = answer(&app(&server), speech::transcription(&[Part::File("file", bytes)])).await;
        assert_eq!(status, StatusCode::OK, "{name}: {body}");
        assert_eq!(body["text"], "hello from ignis", "{name}");
        let seconds = body["usage"]["seconds"].as_f64().unwrap();
        assert!((0.9..=1.15).contains(&seconds), "{name}: {seconds}");
        let pushed = executor.samples_pushed();
        assert!((14_000..=18_500).contains(&pushed), "{name}: {pushed} samples");
    }
}

/// AC 5: any other format is a `400` naming the accepted set (Opus, until
/// libopus is built), and no file is a `400` naming `file`.
#[tokio::test]
async fn any_other_format_is_a_400_naming_the_accepted_set() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let app = app(&server);
    for (name, bytes) in [("opus in ogg", speech::OGG_OPUS), ("opus in webm", speech::WEBM_OPUS), ("text", b"not audio".as_slice())] {
        let (status, _, body) = answer(&app, speech::transcription(&[Part::File("file", bytes)])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {body}");
        assert_eq!(body["error"]["param"], "file");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("accepted:") && message.contains("WAV") && message.contains("FLAC"), "{name}: {message}");
    }
    let (status, _, body) = answer(&app, speech::transcription(&[Part::Text("model", "whisper-1")])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "file");
}

/// AC 5: no audio byte reaches the disk: the temp directory the process
/// writes to holds no new file after a transcription of every format.
#[tokio::test]
async fn no_audio_byte_reaches_the_disk() {
    let (server, dir) = speech::server(fake(), |_| {});
    let before = std::fs::read_dir(&dir).unwrap().count();
    let (status, ..) = transcribe(&app(&server), &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), before, "download.path holds nothing new");
    let tmp = std::env::temp_dir();
    let leftovers: Vec<_> = std::fs::read_dir(&tmp)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.metadata().is_ok_and(|meta| meta.is_file() && meta.len() == speech::WAV.len() as u64))
        .collect();
    assert!(leftovers.is_empty(), "a file the size of the upload is in the temp dir: {leftovers:?}");
}

/// AC 10: a clip over `speech.max_clip_seconds` is a `400` naming the limit,
/// and nothing of it is pushed; at the limit it is served.
#[tokio::test]
async fn a_clip_over_the_limit_is_refused_naming_it_and_nothing_is_split() {
    let executor = fake();
    let (server, _dir) = speech::server(executor.clone(), |group| group.max_clip_seconds = 5);
    let app = app(&server);
    let long = speech::wav(6);
    let (status, _, body) = answer(&app, speech::transcription(&[Part::File("file", &long)])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "file");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("speech.max_clip_seconds") && message.contains("5 s"), "{message}");
    assert_eq!(executor.samples_pushed(), 0);
    let fits = speech::wav(5);
    let (status, ..) = answer(&app, speech::transcription(&[Part::File("file", &fits)])).await;
    assert_eq!(status, StatusCode::OK);
}

// ── AC 6, 7, 9: the fields and the answers ──────────────────────────────

/// AC 6: every field the table refuses is a `400` naming it in `param`, and
/// the values it leaves inert are served; no field is dropped in silence.
#[tokio::test]
async fn every_field_is_honoured_inert_or_refused_naming_it() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let app = app(&server);
    let refused: &[(&[(&str, &str)], &str)] = &[
        (&[("response_format", "srt")], "response_format"),
        (&[("response_format", "vtt")], "response_format"),
        (&[("response_format", "diarized_json")], "response_format"),
        (&[("response_format", "nonsense")], "response_format"),
        (&[("timestamp_granularities[]", "word"), ("response_format", "verbose_json")], "timestamp_granularities[]"),
        (&[("timestamp_granularities[]", "segment")], "timestamp_granularities[]"),
        (&[("temperature", "0.4")], "temperature"),
        (&[("chunking_strategy", "auto")], "chunking_strategy"),
        (&[("chunking_strategy", "{\"type\":\"server_vad\"}")], "chunking_strategy"),
        (&[("include[]", "logprobs")], "include[]"),
        (&[("known_speaker_names[]", "alice")], "known_speaker_names[]"),
        (&[("known_speaker_references[]", "data:audio/wav;base64,AAAA")], "known_speaker_references[]"),
        (&[("keywords[]", "bad<word")], "keywords[]"),
        (&[("keywords[]", "bad>word")], "keywords[]"),
        (&[("keywords[]", "two\nlines")], "keywords[]"),
        (&[("keywords[]", "cr\rword")], "keywords[]"),
        (&[("language", "xx")], "language"),
        (&[("language", "it"), ("languages[]", "en")], "languages[]"),
        (&[("languages[]", "en"), ("languages[]", "xx")], "languages[]"),
        (&[("stream", "maybe")], "stream"),
    ];
    for (fields, param) in refused {
        let (status, _, body) = transcribe(&app, fields).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{fields:?}: {body}");
        assert_eq!(body["error"]["param"], *param, "{fields:?}: {body}");
    }
    let served: &[&[(&str, &str)]] = &[
        &[("temperature", "0")],
        &[("temperature", "0.0")],
        &[("chunking_strategy", "null")],
        &[("response_format", "json")],
        &[("response_format", "text")],
        &[("response_format", "verbose_json"), ("timestamp_granularities[]", "segment")],
        &[("language", "it")],
        &[("language", "Italian")],
        &[("languages[]", "de")],
        &[("languages[]", "de"), ("languages[]", "en")],
        &[("keywords[]", "ignis"), ("keywords[]", "Qwen3")],
        &[("prompt", "A talk about GPUs.")],
        &[("stream", "false")],
        &[("some_future_field", "x")],
    ];
    for fields in served {
        let (status, _, body) = transcribe(&app, fields).await;
        assert_eq!(status, StatusCode::OK, "{fields:?}: {body}");
    }
}

/// AC 7: `json` answers `{text, usage}` with the detected language as
/// `languages: [{code}]`, and omits it when the model labelled the clip
/// `None`; `text` is the body; `verbose_json` carries the model's own label,
/// one segment spanning the clip, and omits `language` when `None`.
#[tokio::test]
async fn the_three_formats_answer_as_specified() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let app = app(&server);

    let (status, _, json) = transcribe(&app, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["text"], "hello from ignis");
    assert_eq!(json["usage"]["type"], "duration");
    assert!((json["usage"]["seconds"].as_f64().unwrap() - 1.0).abs() < 0.01);
    assert_eq!(json["languages"], json!([{ "code": "en" }]));

    let response = app.clone().oneshot(speech::transcription(&[Part::File("file", speech::WAV), Part::Text("response_format", "text")])).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()["content-type"].to_str().unwrap().starts_with("text/plain"));
    assert_eq!(speech::bytes(response).await, b"hello from ignis");

    let (_, _, verbose) = transcribe(&app, &[("response_format", "verbose_json"), ("timestamp_granularities[]", "segment")]).await;
    assert_eq!(verbose["language"], "English");
    assert_eq!(verbose["text"], "hello from ignis");
    assert!((verbose["duration"].as_f64().unwrap() - 1.0).abs() < 0.01);
    let segments = verbose["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0]["start"], 0.0);
    assert_eq!(segments[0]["end"], verbose["duration"]);
    assert_eq!(segments[0]["text"], "hello from ignis");
    assert_eq!(segments[0]["tokens"], json!([0, 1, 2]));
    assert_eq!(segments[0]["avg_logprob"], 0.0);
    assert_eq!(verbose["usage"], json!({ "type": "duration", "seconds": verbose["duration"] }));

    // The model said `None`: no language to report.
    let mut silent = FakeExecutor::saying(&["hello"]);
    silent.language = None;
    let (server, _dir) = speech::server(Arc::new(silent), |_| {});
    let app = self::app(&server);
    let (_, _, json) = transcribe(&app, &[]).await;
    assert!(json.get("languages").is_none(), "{json}");
    let (_, _, verbose) = transcribe(&app, &[("response_format", "verbose_json")]).await;
    assert!(verbose.get("language").is_none(), "{verbose}");
}

/// AC 9: `prompt` and `keywords[]` reach the executor as the context, the
/// keywords after the prompt as one line, with the cap `speech.context_max_tokens`;
/// a forced language reaches it by the model's name.
#[tokio::test]
async fn the_prompt_and_keywords_reach_the_executor_as_the_capped_context() {
    let executor = fake();
    let (server, _dir) = speech::server(executor.clone(), |group| group.context_max_tokens = 64);
    let app = app(&server);
    let (status, ..) = transcribe(&app, &[("prompt", "A talk about GPUs."), ("keywords[]", "ignis"), ("keywords[]", "Qwen3"), ("language", "it")]).await;
    assert_eq!(status, StatusCode::OK);
    transcribe(&app, &[("languages[]", "de"), ("languages[]", "en")]).await;
    transcribe(&app, &[]).await;
    let opened = executor.opened.lock().unwrap().clone();
    assert_eq!(opened.len(), 3);
    assert_eq!(opened[0].context, "A talk about GPUs.\nignis, Qwen3");
    assert_eq!(opened[0].context_max_tokens, 64);
    assert_eq!((opened[0].rollback_tokens, opened[0].unfixed_partials), (10, 2), "the group's defaults reach the executor");
    assert_eq!(opened[0].language.as_deref(), Some("Italian"));
    assert_eq!(opened[1].language, None, "several languages auto-detect");
    assert_eq!(opened[2].context, "");
}

/// The model is loaded by the first request, with the load's settings, and
/// not before; a later one does not load it again.
#[tokio::test]
async fn the_first_request_loads_the_model_with_its_settings() {
    let executor = fake();
    let (server, dir) = speech::server(executor.clone(), |group| {
        group.threads = Some(3);
        group.pinned_weights = true;
    });
    let app = app(&server);
    assert_eq!(executor.loads.load(Ordering::SeqCst), 0, "nothing is loaded at start");
    assert_eq!(models(&app).await["data"].as_array().unwrap().len(), 2, "available, though not loaded");
    transcribe(&app, &[]).await;
    transcribe(&app, &[]).await;
    assert_eq!(executor.loads.load(Ordering::SeqCst), 1);
    let settings = executor.load_settings.lock().unwrap()[0];
    assert_eq!((settings.threads, settings.pinned_weights), (Some(3), true));
    let files = executor.load_files.lock().unwrap()[0].clone();
    assert_eq!(files.id, SPEECH_MODEL);
    assert_eq!(files.decoder, dir.join(format!("{SPEECH_MODEL}-decoder.gguf")));
    assert_eq!(files.files.len(), 2);
}

/// A model that will not load is `503 speech_unavailable` with the reason; an
/// executor that fails mid-transcription is a `500`.
#[tokio::test]
async fn a_failed_load_and_a_failed_transcription_are_answered() {
    let executor = fake();
    executor.fail_load.store(true, Ordering::SeqCst);
    let (server, _dir) = speech::server(executor.clone(), |_| {});
    let app = app(&server);
    let (status, headers, body) = transcribe(&app, &[]).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "speech_unavailable");
    assert!(body["error"]["message"].as_str().unwrap().contains("could not be loaded"));
    assert!(!headers.contains_key("retry-after"));
    executor.fail_load.store(false, Ordering::SeqCst);
    executor.fail_commit.store(true, Ordering::SeqCst);
    let (status, _, body) = transcribe(&app, &[]).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert_eq!(body["error"]["code"], "engine_error");
    executor.fail_commit.store(false, Ordering::SeqCst);
    assert_eq!(transcribe(&app, &[]).await.0, StatusCode::OK, "the next one is served");
}

// ── AC 8: the stream ────────────────────────────────────────────────────

async fn stream(app: &axum::Router, wav: &[u8], extra: &[(&str, &str)]) -> (StatusCode, Vec<String>) {
    let mut parts = vec![Part::File("file", wav), Part::Text("stream", "true")];
    parts.extend(extra.iter().map(|(name, value)| Part::Text(name, value)));
    let response = app.clone().oneshot(speech::transcription(&parts)).await.unwrap();
    let status = response.status();
    if status == StatusCode::OK {
        assert!(response.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    }
    let text = String::from_utf8(speech::bytes(response).await).unwrap();
    (status, speech::data_lines(&text))
}

/// AC 8: the stream sends `transcript.text.delta` events as the text becomes
/// stable, one `transcript.text.done` whose `text` is the deltas joined and
/// whose usage is in audio seconds, then `[DONE]`.
#[tokio::test]
async fn a_stream_sends_deltas_then_done_whose_text_is_the_deltas_joined() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let long = speech::wav(17);
    let (status, lines) = stream(&app(&server), &long, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(lines.last().map(String::as_str), Some("[DONE]"));
    let events: Vec<Value> = lines[..lines.len() - 1].iter().map(|line| serde_json::from_str(line).unwrap()).collect();
    let (done, deltas) = events.split_last().unwrap();
    assert_eq!(done["type"], "transcript.text.done");
    assert!(deltas.len() >= 2, "the text grew over the 17 s clip's windows: {deltas:?}");
    assert!(deltas.iter().all(|event| event["type"] == "transcript.text.delta"));
    let joined: String = deltas.iter().map(|event| event["delta"].as_str().unwrap()).collect();
    assert_eq!(done["text"], joined.as_str());
    assert_eq!(done["text"], "hello from ignis");
    assert_eq!(done["usage"]["type"], "duration");
    assert!((done["usage"]["seconds"].as_f64().unwrap() - 17.0).abs() < 0.01);
}

/// A short clip streams too: its text arrives whole in the done event's
/// deltas, joined equal to it.
#[tokio::test]
async fn a_short_clip_streams_its_text_before_done() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let (status, lines) = stream(&app(&server), speech::WAV, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let events: Vec<Value> = lines[..lines.len() - 1].iter().map(|line| serde_json::from_str(line).unwrap()).collect();
    let joined: String = events.iter().filter(|event| event["type"] == "transcript.text.delta").map(|event| event["delta"].as_str().unwrap()).collect();
    assert_eq!(joined, "hello from ignis");
    assert_eq!(events.last().unwrap()["text"], "hello from ignis");
}

// ── AC 11: admission ────────────────────────────────────────────────────

/// Wait until `condition` holds, at most 20 s.
async fn until(condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !condition() {
        assert!(Instant::now() < deadline, "waited 20 s");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// AC 11: past `speech.max_queue` waiting requests the answer is `503
/// engine_full` with a `Retry-After` of one to fifteen seconds; the ones
/// admitted are served in order.
#[tokio::test]
async fn past_max_queue_the_answer_is_503_with_a_bounded_retry_after() {
    let executor = fake();
    executor.set_step_delay(Duration::from_millis(300));
    let (server, _dir) = speech::server(executor.clone(), |group| group.max_queue = 1);
    let app = app(&server);
    let first = tokio::spawn({
        let app = app.clone();
        async move { transcribe(&app, &[]).await }
    });
    until(|| !executor.opened.lock().unwrap().is_empty()).await;
    let second = tokio::spawn({
        let app = app.clone();
        async move { transcribe(&app, &[]).await }
    });
    until(|| server.speech.queue_length() == 1).await;
    let (status, headers, body) = transcribe(&app, &[]).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "engine_full");
    let retry: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=15).contains(&retry), "{retry}");
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    assert_eq!(second.await.unwrap().0, StatusCode::OK);
    assert_eq!(transcribe(&app, &[]).await.0, StatusCode::OK, "room again");
}

/// Admission comes before the decode: a request that finds the queue full is
/// refused `503` whatever its bytes are. A body that is no audio would be a
/// `400` had it been decoded, so the `503` shows the decode did not run; with
/// room again, the same body is decoded and refused as audio.
#[tokio::test]
async fn a_full_queue_is_answered_before_the_clip_is_decoded() {
    let executor = fake();
    executor.set_step_delay(Duration::from_millis(300));
    let (server, _dir) = speech::server(executor.clone(), |group| group.max_queue = 0);
    let app = app(&server);
    let first = tokio::spawn({
        let app = app.clone();
        async move { transcribe(&app, &[]).await }
    });
    until(|| !executor.opened.lock().unwrap().is_empty()).await;
    let not_audio = || speech::transcription(&[Part::File("file", b"not audio")]);
    let (status, headers, body) = answer(&app, not_audio()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "engine_full");
    assert!(headers.contains_key("retry-after"));
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    let (status, _, body) = answer(&app, not_audio()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "room again, so it is decoded: {body}");
}

/// `speech.max_queue` 0 runs one at a time with none waiting.
#[tokio::test]
async fn a_queue_of_zero_serves_one_at_a_time() {
    let executor = fake();
    executor.set_step_delay(Duration::from_millis(300));
    let (server, _dir) = speech::server(executor.clone(), |group| group.max_queue = 0);
    let app = app(&server);
    let first = tokio::spawn({
        let app = app.clone();
        async move { transcribe(&app, &[]).await }
    });
    until(|| !executor.opened.lock().unwrap().is_empty()).await;
    assert_eq!(transcribe(&app, &[]).await.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
}

// ── AC 12: deadlines ────────────────────────────────────────────────────

/// AC 12: a request past `server.request_timeout`, counted from its admission,
/// is `504 request_timeout`, whatever `server.timeout_reply` says: a
/// transcript has no answer that says it is incomplete.
#[tokio::test]
async fn a_request_past_its_deadline_is_a_504_whatever_the_reply_setting() {
    for reply in [ignis_server::deadline::TimeoutReply::Partial, ignis_server::deadline::TimeoutReply::Error] {
        let executor = fake();
        executor.set_step_delay(Duration::from_millis(400));
        let (server, _dir) = speech::server(executor.clone(), |_| {});
        let server = server.with_request_timeout(Duration::from_millis(100)).with_timeout_reply(reply);
        let began = Instant::now();
        let (status, headers, body) = transcribe(&server.app(), &[]).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert_eq!(body["error"]["code"], "request_timeout");
        assert!(!headers.contains_key("ignis-stopped-by"));
        assert!(began.elapsed() < Duration::from_millis(350), "answered at the deadline, not at the job's end");
    }
}

/// AC 12: a streamed one ends with an `error` event after the `200`, the
/// deadline being `server.stream_timeout`.
#[tokio::test]
async fn a_stream_past_its_deadline_ends_with_an_error_event() {
    let executor = fake();
    executor.set_step_delay(Duration::from_millis(400));
    let (server, _dir) = speech::server(executor, |_| {});
    let server = server.with_request_timeout(Duration::from_secs(60)).with_stream_timeout(Duration::from_millis(100));
    let (status, lines) = stream(&server.app(), speech::WAV, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(lines.last().map(String::as_str), Some("[DONE]"));
    let error: Value = serde_json::from_str(&lines[lines.len() - 2]).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["code"], "request_timeout");
    assert!(lines.iter().all(|line| !line.contains("transcript.text.done")));
}

// ── AC 13: beside the other model ───────────────────────────────────────

/// AC 13 (Q40): during a switch, while the first traversal runs, and with the
/// text model unloaded, a transcription is served; and none of them is a use
/// of the text model, which the idle clock would count.
#[tokio::test]
async fn transcriptions_are_served_through_a_switch_a_warm_up_and_an_unload() {
    for status in [
        ModelStatus::Switching { from: TEXT_MODEL.to_owned(), to: "another".to_owned() },
        ModelStatus::WarmingUp,
        ModelStatus::Unloaded,
        ModelStatus::Loading,
        ModelStatus::Failed { reason: "the load failed".to_owned(), from_unloaded: false },
    ] {
        let (server, _dir) = speech::server(fake(), |_| {});
        server.status.store(Arc::new(status.clone()));
        let app = app(&server);
        let (code, _, body) = transcribe(&app, &[]).await;
        assert_eq!(code, StatusCode::OK, "{status:?}: {body}");
        assert_eq!(body["text"], "hello from ignis");
        // The text model's own routes are held, as before.
        let chat = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json!({"model": TEXT_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).to_string()))
            .unwrap();
        if matches!(status, ModelStatus::Switching { .. } | ModelStatus::WarmingUp) {
            assert_eq!(answer(&app, chat).await.0, StatusCode::SERVICE_UNAVAILABLE, "{status:?}");
        }
    }
}

/// AC 13 (Q40): with the image model loaded instead of a text model, a
/// transcription is served on the CPU beside it, and the image model stays.
#[tokio::test]
async fn a_transcription_is_served_with_the_image_model_loaded() {
    let image = ignis_server::images::ImageModel::new("ignis-image-2.1-turbo", Box::new(ignis_core::image::FakeImageRunner::default()));
    let base = Server::from_active(ignis_server::ActiveModel::image(image));
    let (server, _dir) = speech::server_on(base, fake(), |_| {});
    assert_eq!(server.active().kind(), ignis_core::compute::ModelKind::Image);
    let app = app(&server);
    let (code, _, body) = transcribe(&app, &[]).await;
    assert_eq!(code, StatusCode::OK, "{body}");
    assert_eq!(body["text"], "hello from ignis");
    assert_eq!(server.active().kind(), ignis_core::compute::ModelKind::Image, "the image model is still the one loaded");
    assert_eq!(server.usage.in_flight(), 0, "a transcription is no use of the loaded model");
}

/// Spec speech/02 AC 11, 01 AC 19: a batch transcription's commit is a unit
/// of work a step, so a Realtime session's unit queued while it decodes runs
/// between two of its steps, not after the whole commit.
#[tokio::test]
async fn a_realtime_unit_runs_between_the_steps_of_a_batch_commit() {
    let executor = fake();
    executor.commit_steps.store(3, Ordering::SeqCst);
    executor.set_step_delay(Duration::from_millis(200));
    let (server, _dir) = speech::server(executor.clone(), |_| {});
    let speech = Arc::clone(&server.speech);
    let app = app(&server);
    let running = tokio::spawn({
        let app = app.clone();
        async move { transcribe(&app, &[]).await }
    });
    until(|| executor.calls().iter().any(|call| call == "commit 0")).await;
    let session = speech.open_session(&server.live().speech).unwrap();
    let calls = Arc::clone(&executor.calls);
    speech.submit_interactive(
        &session,
        Box::new(move || {
            calls.lock().unwrap().push("realtime".to_owned());
            Step::Done
        }),
    );
    assert_eq!(running.await.unwrap().0, StatusCode::OK);
    let calls: Vec<String> = executor.calls().into_iter().filter(|call| call.starts_with("commit") || call == "realtime").collect();
    assert_eq!(calls, ["commit 0", "realtime", "commit 0", "commit 0"]);
}

/// A server running a configuration resolved from `argv`, with the config
/// state a start builds, so `PATCH /v1/config` applies to it.
fn configured(server: Server, argv: &[&str]) -> Server {
    let argv: Vec<String> = argv.iter().map(|arg| arg.to_string()).collect();
    let config = match resolve_with(&argv, |_| None, &RealFiles).expect("the options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    let server = server.with_config(Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles))));
    server.apply_config(&config);
    server
}

async fn patch(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let request = Request::builder().method("PATCH").uri("/v1/config").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let (status, _, body) = answer(app, request).await;
    (status, body)
}

/// `speech.idle_unload` is read live (the guide: every field is): a `PATCH`
/// while the model is loaded and idle reaches the idle clock, from 0 (never)
/// and from a long time alike, nothing submitted since.
#[tokio::test]
async fn a_patch_of_the_idle_time_reaches_an_idle_model() {
    for before in ["0", "300"] {
        let executor = fake();
        let (server, _dir) = speech::server(executor.clone(), |_| {});
        let server = configured(server, &["--speech-idle-unload", before]);
        let app = app(&server);
        assert_eq!(transcribe(&app, &[]).await.0, StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(executor.is_loaded(), "{before}: unloaded before the patch");
        let (status, applied) = patch(&app, json!({ "speech": { "idle_unload": 1 } })).await;
        assert_eq!(status, StatusCode::OK, "{applied}");
        until(|| !executor.is_loaded()).await;
    }
}

/// Spec speech/02 §Which model: a `PATCH` naming a text model as
/// `speech.model`, or a compat name equal to an id of the catalog the server
/// runs with (an operator's entry here), is refused, and nothing applies.
#[tokio::test]
async fn a_patch_of_a_text_model_or_a_catalog_id_as_a_speech_name_is_refused() {
    let (server, dir) = speech::server(fake(), |_| {});
    let mut models = speech::models(&dir);
    let mut entries = models.catalog.entries().to_vec();
    let mut operator = entries[0].clone();
    operator.id = "operator-asr".to_owned();
    entries.push(operator);
    models.catalog = Arc::new(ignis_server::download::catalog::Catalog::from_entries(entries));
    let server = configured(server.with_speech(Arc::new(ignis_server::speech::Speech::new(fake(), Some(models)))), &[]);
    let app = app(&server);
    let (status, refused) = patch(&app, json!({ "speech": { "model": "qwen3.8-27b" } })).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("config_invalid")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("not a speech model"), "{refused}");
    let (status, refused) = patch(&app, json!({ "speech": { "compat_names": ["whisper-1", "operator-asr"] } })).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("config_invalid")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("operator-asr"), "{refused}");
    assert_eq!(server.live().speech.model, "auto");
    assert!(!server.live().speech.compat_names.iter().any(|name| name == "operator-asr"));
    let (status, applied) = patch(&app, json!({ "speech": { "compat_names": ["whisper-1", "my-transcriber"] } })).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
}

/// AC 13: a transcription in flight is not a request using the text model.
#[tokio::test]
async fn a_transcription_in_flight_is_no_use_of_the_text_model() {
    let executor = fake();
    executor.set_step_delay(Duration::from_millis(300));
    let (server, _dir) = speech::server(executor.clone(), |_| {});
    let app = app(&server);
    let running = tokio::spawn({
        let app = app.clone();
        async move { transcribe(&app, &[]).await }
    });
    until(|| !executor.opened.lock().unwrap().is_empty()).await;
    assert_eq!(server.usage.in_flight(), 0, "the idle clock sees no use while it transcribes");
    assert_eq!(running.await.unwrap().0, StatusCode::OK);
    assert_eq!(server.usage.in_flight(), 0);
}

/// The route is behind the API key, as every `/v1` route is, though it is
/// outside the serving gate.
#[tokio::test]
async fn the_route_is_behind_the_api_key() {
    let (server, _dir) = speech::server(fake(), |_| {});
    let server = server.with_api_key(ignis_server::config::ApiKey::new("sk-speech"));
    let app = app(&server);
    assert_eq!(transcribe(&app, &[]).await.0, StatusCode::UNAUTHORIZED);
    let mut request = speech::transcription(&[Part::File("file", speech::WAV)]);
    request.headers_mut().insert("authorization", "Bearer sk-speech".parse().unwrap());
    assert_eq!(answer(&app, request).await.0, StatusCode::OK);
}

/// A switch never goes to a speech model: it is not a model to switch to,
/// whether named by a request or by `POST /v1/models/switch`.
#[tokio::test]
async fn a_speech_entry_is_no_model_to_switch_to() {
    let (server, dir) = speech::server(fake(), |_| {});
    let catalog = speech::models(&dir);
    let switcher = ignis_server::model_switch::Switcher::new(
        Arc::new(NeverLoads),
        Duration::from_secs(1),
    )
    .with_catalog(ignis_server::model_switch::CatalogModels { catalog: catalog.catalog.clone(), dir: dir.clone() }, true);
    let server = server.with_switcher(switcher);
    let app = app(&server);
    // On disk, a catalog entry: but not listed as a model to switch to.
    let list = models(&app).await;
    let listed: Vec<_> = list["data"].as_array().unwrap().iter().filter(|entry| entry["active"] == false).collect();
    assert!(listed.is_empty(), "{listed:?}");
    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/switch")
        .header("content-type", "application/json")
        .body(Body::from(json!({ "model": SPEECH_MODEL }).to_string()))
        .unwrap();
    let (status, _, body) = answer(&app, request).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_kind_mismatch");
    assert!(body["error"]["message"].as_str().unwrap().contains("speech model"));
}

/// A speech entry is told by the catalog's `kind`, not by its id: one named
/// `acme-asr` is never switched to, and a text entry named like a Qwen3-ASR
/// model still is.
#[tokio::test]
async fn a_speech_entry_is_told_by_its_kind_and_not_by_its_id() {
    use ignis_core::compute::ModelKind;
    use ignis_server::download::{Catalog, CatalogEntry, CatalogFile, CatalogLayer};
    use ignis_server::model_switch::{CatalogModels, Switcher, Unswitchable};
    let dir = speech::temp_dir("kind-not-id");
    let entry = |id: &str, kind: Option<ModelKind>| CatalogEntry {
        id: id.to_owned(),
        repo: "test/asr".to_owned(),
        revision: "0".repeat(40),
        artifact: format!("{id}.gguf"),
        kind,
        card_class: None,
        family: None,
        aliases: Vec::new(),
        files: vec![CatalogFile { name: format!("{id}.gguf"), bytes: 4, sha256: "0".repeat(64) }],
        layer: CatalogLayer::Operator,
    };
    let catalog = Catalog::from_entries(vec![entry("acme-asr", Some(ModelKind::Speech)), entry("qwen3-asr-text", None)]);
    for id in ["acme-asr", "qwen3-asr-text"] {
        std::fs::write(dir.join(format!("{id}.gguf")), b"gguf").unwrap();
    }
    let switcher = Switcher::new(Arc::new(NeverLoads), Duration::from_secs(1))
        .with_catalog(CatalogModels { catalog: Arc::new(catalog), dir: dir.clone() }, true);
    assert!(switcher.is_speech("acme-asr") && !switcher.is_speech("qwen3-asr-text"));
    assert_eq!(switcher.resolve("acme-asr"), Err(Unswitchable::Speech));
    assert!(switcher.known("acme-asr").is_none());
    assert!(switcher.resolve("qwen3-asr-text").is_ok());
    assert!(switcher.known("qwen3-asr-text").is_some());
    let servable: Vec<String> = switcher.servable_models("other").into_iter().map(|servable| servable.id).collect();
    assert_eq!(servable, ["qwen3-asr-text"]);
}

/// A loader that loads nothing: the switch tests need a switcher, not a load.
struct NeverLoads;

impl ignis_server::model_switch::ModelLoader for NeverLoads {
    fn prepare(
        &self,
        _: &ignis_server::ModelSource,
        _: Option<&ignis_server::config::Config>,
    ) -> Result<Box<dyn ignis_server::model_switch::PreparedLoad>, String> {
        Err("this test never loads".to_owned())
    }
}

// ── AC 15: the metrics ──────────────────────────────────────────────────

async fn metrics_text(server: &ignis_server::Server) -> String {
    let response = server
        .metrics_app()
        .expect("metrics are on")
        .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    String::from_utf8(speech::bytes(response).await).unwrap()
}

/// AC 15: `/metrics` exports the two counters, the wait histogram and the
/// two gauges, from the server's telemetry; a completed transcription moves
/// them, and a cancelled one counts as such.
#[tokio::test(flavor = "multi_thread")]
async fn metrics_export_the_five_speech_families() {
    let executor = Arc::new(FakeExecutor::saying(&["one", "two"]));
    let (server, _dir) = speech::server(executor.clone(), |_| {});
    let server = server.with_metrics();
    assert!(!metrics_text(&server).await.contains("ignis_speech_"), "absent until the speech model is asked");
    let app = server.app();
    assert_eq!(app.clone().oneshot(speech::transcription(&[Part::File("file", speech::WAV)])).await.unwrap().status(), StatusCode::OK);
    let text = metrics_text(&server).await;
    for family in [
        "# TYPE ignis_speech_transcriptions_total counter",
        "# TYPE ignis_speech_audio_seconds_total counter",
        "# TYPE ignis_speech_commit_wait_seconds histogram",
        "# TYPE ignis_speech_sessions_open gauge",
        "# TYPE ignis_speech_queue_length gauge",
    ] {
        assert!(text.contains(family), "{family} missing:\n{text}");
    }
    assert!(text.contains("ignis_speech_transcriptions_total{executor=\"cpu\",outcome=\"completed\"} 1"), "{text}");
    assert!(text.contains("ignis_speech_transcriptions_total{executor=\"gpu_loan\",outcome=\"completed\"} 0"), "{text}");
    assert!(text.contains("ignis_speech_audio_seconds_total 1
"), "{text}");
    assert!(text.contains("ignis_speech_commit_wait_seconds_count 1"), "{text}");
    assert!(text.contains("ignis_speech_sessions_open 0") && text.contains("ignis_speech_queue_length 0"), "{text}");
}

/// A request that goes away while its transcription runs is cancelled: the
/// job stops at its next window and the outcome says so.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_that_goes_away_is_counted_cancelled() {
    let executor = Arc::new(FakeExecutor::saying(&["one", "two"]));
    executor.set_step_delay(Duration::from_millis(200));
    let (server, _dir) = speech::server(executor.clone(), |_| {});
    let server = server.with_metrics();
    let app = server.app();
    let long = speech::wav(30);
    let request = tokio::spawn({
        let app = app.clone();
        async move { app.oneshot(speech::transcription(&[Part::File("file", &long)])).await }
    });
    while executor.opened.lock().unwrap().is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    request.abort();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let text = metrics_text(&server).await;
        if text.contains("ignis_speech_transcriptions_total{executor=\"cpu\",outcome=\"cancelled\"} 1") {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "never counted cancelled:\n{text}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // The job stopped at its next window: it did not push all four.
    assert!(executor.pushed.lock().unwrap().len() < 4);
}
