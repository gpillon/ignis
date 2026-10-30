//! The speech model's test fixtures (spec speech/02, GitHub #352): a server
//! over a [`FakeExecutor`] and a catalog entry of our own with its files in a
//! temp directory, a `multipart/form-data` builder, and the audio fixtures.
//! Included by `tests/speech_http.rs` and `tests/speech_log.rs` alike.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::config::schema::{DiarizerGroup, SpeechGroup};
use ignis_server::download::catalog::{Catalog, CatalogEntry, CatalogFile, CatalogLayer};
use ignis_server::engine::Engine;
use ignis_server::speech::executor::{FakeDiarizer, FakeExecutor};
use ignis_server::speech::{Speech, SpeechModels};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use serde_json::Value;

/// The text model the mock server loads.
pub const TEXT_MODEL: &str = "test-model";
/// The speech entry's served id, what `speech.model: auto` resolves to.
pub const SPEECH_MODEL: &str = "qwen3-asr-1.7b";

pub const WAV: &[u8] = include_bytes!("../fixtures/audio/tone.wav");
pub const FLAC: &[u8] = include_bytes!("../fixtures/audio/tone.flac");
pub const MP3: &[u8] = include_bytes!("../fixtures/audio/tone.mp3");
pub const AAC: &[u8] = include_bytes!("../fixtures/audio/tone.aac");
pub const M4A: &[u8] = include_bytes!("../fixtures/audio/tone.m4a");
pub const OGG_FLAC: &[u8] = include_bytes!("../fixtures/audio/tone-flac.ogg");
pub const MATROSKA: &[u8] = include_bytes!("../fixtures/audio/tone.mka");
pub const OGG_OPUS: &[u8] = include_bytes!("../fixtures/audio/tone-opus.ogg");
pub const WEBM_OPUS: &[u8] = include_bytes!("../fixtures/audio/tone-opus.webm");

/// A unique empty directory under the system temp dir: `download.path`.
pub fn temp_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!("ignis-speech-http-{name}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Our own catalog entries, whose ids are the speech models' (the real ones
/// are ticket #351's).
pub fn models(dir: &std::path::Path) -> SpeechModels {
    let entry = |id: &str| CatalogEntry {
        id: id.to_owned(),
        repo: "test/asr".to_owned(),
        revision: "0".repeat(40),
        artifact: format!("{id}-decoder.gguf"),
        card_class: None,
        family: None,
        aliases: Vec::new(),
        kind: Some(ignis_core::compute::ModelKind::Speech),
        files: ["encoder", "decoder"]
            .iter()
            .map(|part| CatalogFile { name: format!("{id}-{part}.gguf"), bytes: 4, sha256: "0".repeat(64) })
            .collect(),
        layer: CatalogLayer::BuiltIn,
    };
    SpeechModels { catalog: Arc::new(Catalog::from_entries(vec![entry(SPEECH_MODEL), entry("qwen3-asr-0.6b")])), dir: dir.to_owned() }
}

/// Put an entry's files on disk.
pub fn put_files(dir: &std::path::Path, id: &str) {
    for part in ["encoder", "decoder"] {
        std::fs::write(dir.join(format!("{id}-{part}.gguf")), b"gguf").unwrap();
    }
}

/// A server over the mock text model with the speech model's files on disk
/// and `executor` behind the route; `edit` changes the `speech.*` settings.
pub fn server(executor: Arc<FakeExecutor>, edit: impl FnOnce(&mut SpeechGroup)) -> (Server, PathBuf) {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: TEXT_MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    server_on(Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider)), executor, edit)
}

/// [`server`] over `base`, a server of the test's own (an image model loaded, say).
pub fn server_on(base: Server, executor: Arc<FakeExecutor>, edit: impl FnOnce(&mut SpeechGroup)) -> (Server, PathBuf) {
    let dir = temp_dir("server");
    put_files(&dir, SPEECH_MODEL);
    let mut group = SpeechGroup::default();
    edit(&mut group);
    let server = base.with_speech(Arc::new(Speech::new(executor, Some(models(&dir))))).with_speech_group(group);
    (server, dir)
}

/// [`server`], with `diarizer` giving the speakers (spec speech/08) and the
/// `diarizer` group as `edit_diarizer` changed it.
pub fn server_with_diarizer(
    executor: Arc<FakeExecutor>,
    diarizer: Arc<FakeDiarizer>,
    edit: impl FnOnce(&mut SpeechGroup),
    edit_diarizer: impl FnOnce(&mut DiarizerGroup),
) -> (Server, PathBuf) {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: TEXT_MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let dir = temp_dir("server");
    put_files(&dir, SPEECH_MODEL);
    let (mut group, mut diarizer_group) = (SpeechGroup::default(), DiarizerGroup::default());
    edit(&mut group);
    edit_diarizer(&mut diarizer_group);
    let speech = Speech::new(executor, Some(models(&dir))).with_diarizer(diarizer);
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_speech(Arc::new(speech))
        .with_speech_group(group)
        .with_diarizer_group(diarizer_group);
    (server, dir)
}

/// One part of a form.
pub enum Part<'a> {
    Text(&'a str, &'a str),
    File(&'a str, &'a [u8]),
}

/// A `POST /v1/audio/transcriptions` over `parts`.
pub fn transcription(parts: &[Part<'_>]) -> Request<Body> {
    let boundary = "----ignis-speech-test-boundary";
    let mut body: Vec<u8> = Vec::new();
    for part in parts {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match part {
            Part::Text(name, value) => {
                body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
            }
            Part::File(name, bytes) => {
                body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"audio.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes(),
                );
                body.extend_from_slice(bytes);
                body.extend_from_slice(b"\r\n");
            }
        }
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap()
}

/// A body as bytes.
pub async fn bytes(response: axum::response::Response) -> Vec<u8> {
    response.into_body().collect().await.unwrap().to_bytes().to_vec()
}

/// A body as JSON.
pub async fn json(response: axum::response::Response) -> Value {
    let bytes = bytes(response).await;
    serde_json::from_slice(&bytes).unwrap_or_else(|_| panic!("not JSON: {}", String::from_utf8_lossy(&bytes)))
}

/// A server-sent stream's `data:` payloads, in order.
pub fn data_lines(sse: &str) -> Vec<String> {
    sse.lines().filter_map(|line| line.strip_prefix("data:")).map(|data| data.trim().to_owned()).collect()
}

/// `seconds` of 16 kHz mono 16-bit silence as a WAV file: a clip of any
/// length without a fixture of that size.
pub fn wav(seconds: u32) -> Vec<u8> {
    let data = seconds as usize * 16_000 * 2;
    let mut file = Vec::with_capacity(44 + data);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(36 + data as u32).to_le_bytes());
    file.extend_from_slice(b"WAVEfmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes()); // PCM
    file.extend_from_slice(&1u16.to_le_bytes()); // mono
    file.extend_from_slice(&16_000u32.to_le_bytes());
    file.extend_from_slice(&32_000u32.to_le_bytes());
    file.extend_from_slice(&2u16.to_le_bytes());
    file.extend_from_slice(&16u16.to_le_bytes());
    file.extend_from_slice(b"data");
    file.extend_from_slice(&(data as u32).to_le_bytes());
    file.resize(44 + data, 0);
    file
}
