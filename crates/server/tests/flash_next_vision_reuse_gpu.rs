//! A follow-up turn on an image reuses the prompt and the image's encoding on
//! Flash-Next, and questions fanned out over one image encode it once (GitHub
//! #559, spec #555; ADR 0029 as amended for multimodal prompts, ADR 0035),
//! through the production load path (`load::prepare_model` then
//! `load::load_model`, the vision companion beside the main container) and
//! `POST /v1/chat/completions`, on the RTX 5090 profile's settings with vision
//! on. What each request reused and encoded is read where an operator reads
//! it: `usage.prompt_tokens_details.cached_tokens` on the answer, and the
//! request log's `ignis.request.admitted` line (`media.vision_tokens`,
//! `media.encode_seconds`) off a sink this binary installs. One load, both
//! legs:
//!
//! - **The follow-up turn.** A question about the colour canary, then the
//!   conversation's next turn. The next turn resumes from the first turn's
//!   checkpoint at its generation opener, past the picture -- more prompt
//!   than the first turn holds beside the image's placeholders, which no
//!   claim stopping before the picture reaches -- so it prefills none of the
//!   image and encodes nothing; and its answer still names the colour.
//! - **The fan-out.** Four questions about the number canary, sent together,
//!   each putting its own text before the picture, so no prompt shares
//!   another's past its first tokens and each prefills the image: exactly one
//!   encodes it, the other three find its embedding in the cache. Every
//!   answer names the number.
//!
//! The canary images are the 27B's (`fixtures/vision_canary/`), each answer
//! the fact it shows, greedy with thinking off and the loop guard off
//! (testing.md, "GPU tests that compare output or load an option"); every
//! answer and every count is printed.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test flash_next_vision_reuse_gpu -- --ignored --nocapture
//! ```
//!
//! Run under `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`): it needs
//! the real artifact and its vision companion
//! (`F:/ai/models/Qwen3.8-Flash-Next-ignis/`, or `IGNIS_FLASH_NEXT_DIR`), the
//! card exclusively, and about 40 GiB of host RAM for the expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::gpu_profile;
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;

use support::live_server::SwitchingLiveServer;
use support::media::data_uri;
use support::vision_canary::fixture_dir;

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";

/// The events this binary's load emits, as JSON lines, beside the usual
/// console output.
fn events() -> &'static Arc<ignis_logging::MemorySink> {
    static SINK: OnceLock<Arc<ignis_logging::MemorySink>> = OnceLock::new();
    SINK.get_or_init(|| {
        use tracing_subscriber::layer::SubscriberExt;
        let sink = Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry()
            .with(ignis_logging::JsonLayer::new(sink.clone()))
            .with(tracing_subscriber::fmt::layer());
        tracing::subscriber::set_global_default(subscriber).expect("this binary's only subscriber");
        sink
    })
}

/// The `ignis.request.admitted` lines' attributes from line `from` of the
/// sink on, once `count` of them carry an image: the request log is written
/// after the answer, by the telemetry consumer, so it is waited for (60 s at
/// most).
fn admitted_with_media(from: usize, count: usize) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let admitted: Vec<Value> = events().lines()[from..]
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).expect("a JSON line"))
            .filter(|event| event["event_name"] == "ignis.request.admitted")
            .map(|event| event["attributes"].clone())
            .filter(|attributes| attributes["media.items"] == 1)
            .collect();
        if admitted.len() >= count {
            return admitted;
        }
        assert!(Instant::now() < deadline, "{} admitted lines with an image, wanted {count}", admitted.len());
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn number(attributes: &Value, name: &str) -> f64 {
    attributes[name].as_f64().unwrap_or_else(|| panic!("no `{name}` in {attributes}"))
}

/// `main`'s options: the RTX 5090 profile's (the built-in default on that
/// card) with vision on, a context the canaries need, two host retained
/// slots for the checkpoints and no KV-RAM or KV-disk tier past them, no idle
/// unload, the loop guard off.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = [
        "--vision-enabled",
        "--model-max-context",
        "16384",
        "--reuse-retained-host",
        "2",
        "--reuse-kv-host-pool-bytes",
        "0",
        "--kv-disk-bytes",
        "0",
        "--model-idle-unload",
        "0",
        "--model-loop-guard",
        "off",
    ]
    .map(String::from)
    .to_vec();
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        _ => unreachable!("flags without --help are a config"),
    }
}

fn canary(name: &str) -> String {
    let path = fixture_dir().join(name);
    data_uri(&std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())))
}

/// A user turn: the picture, then `question`.
fn picture_then(image: &str, question: &str) -> Value {
    json!({ "role": "user", "content": [
        { "type": "image_url", "image_url": { "url": image } },
        { "type": "text", "text": question },
    ]})
}

/// One chat completion over `messages`, answered 200: its answer, its prompt
/// tokens and its `cached_tokens`.
async fn chat(url: &str, messages: Value) -> (String, u64, u64) {
    let body = json!({
        "model": MODEL,
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0,
        "enable_thinking": false,
    });
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .expect("the completion request");
    let status = response.status().as_u16();
    let reply: Value = serde_json::from_str(&response.text().await.expect("body")).expect("json");
    assert_eq!(status, 200, "{reply}");
    let answer = reply["choices"][0]["message"]["content"].as_str().expect("an answer").to_owned();
    let usage = &reply["usage"];
    let prompt = usage["prompt_tokens"].as_u64().expect("prompt tokens");
    let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().expect("cached tokens");
    (answer, prompt, cached)
}

fn artifact() -> Option<String> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let artifact = dir.join(ARTIFACT_FILE_NAME);
    for path in [&artifact, &ignis_core::flash_next_vision::companion_path(&dir)] {
        if !path.exists() && gpu_profile::skip_or_fail(&format!("absent: {}", path.display())) {
            return None;
        }
    }
    Some(artifact.to_string_lossy().into_owned())
}

/// The follow-up turn on the colour canary.
fn a_follow_up_turn_resumes_past_the_picture(live: &SwitchingLiveServer) {
    let image = canary("colour.png");
    let question = "What colour is the square in the image? Answer in one short sentence.";
    let from = events().lines().len();
    let (first, first_prompt, first_cached) = live.block_on(chat(&live.url, json!([picture_then(&image, question)])));
    let turn = admitted_with_media(from, 1).remove(0);
    let vision_tokens = number(&turn, "media.vision_tokens") as u64;
    println!(
        "turn 1: {first_prompt} prompt tokens, {vision_tokens} of them the picture's, {first_cached} cached, encode {} s -> {first:?}",
        number(&turn, "media.encode_seconds")
    );
    assert!(first.to_lowercase().contains("red"), "turn 1 names the colour: {first:?}");
    assert!(number(&turn, "media.encode_seconds") > 0.0, "turn 1 encodes the picture: {turn}");

    let from = events().lines().len();
    let messages = json!([
        picture_then(&image, question),
        { "role": "assistant", "content": first },
        { "role": "user", "content": "Name that colour again, in one word." },
    ]);
    let (next, next_prompt, cached) = live.block_on(chat(&live.url, messages));
    let next_turn = admitted_with_media(from, 1).remove(0);
    let encode = number(&next_turn, "media.encode_seconds");
    println!("turn 2: {next_prompt} prompt tokens, {cached} cached, encode {encode} s -> {next:?}");
    // A claim never ends inside a picture, so one that stops before it holds
    // at most the tokens before it: fewer than turn 1's prompt less the
    // picture's placeholders.
    assert!(
        cached > first_prompt - vision_tokens,
        "turn 2 resumes past the picture: {cached} cached of turn 1's {first_prompt} ({vision_tokens} the picture's)"
    );
    assert!(cached <= first_prompt, "and from turn 1's state: {cached} cached of its {first_prompt}");
    assert_eq!(encode, 0.0, "turn 2 encodes nothing: {next_turn}");
    assert!(next.to_lowercase().contains("red"), "turn 2 still reads the picture: {next:?}");
}

/// Four questions over the number canary, sent together.
fn a_fan_out_encodes_the_picture_once(live: &SwitchingLiveServer) {
    let image = canary("number.png");
    let questions = [
        "Read the number in the picture below and answer with the number only.",
        "Which number does this image show? Answer with the number only.",
        "Look at the following image. What number is written in it? Answer with the number only.",
        "Tell me the number shown in the image that follows, with the number only.",
    ];
    let from = events().lines().len();
    let answers = live.block_on(futures_util::future::join_all(questions.iter().map(|question| {
        let messages = json!([{ "role": "user", "content": [
            { "type": "text", "text": question },
            { "type": "image_url", "image_url": { "url": image } },
        ]}]);
        chat(&live.url, messages)
    })));
    let admitted = admitted_with_media(from, questions.len());
    let encodes: Vec<f64> = admitted.iter().map(|attributes| number(attributes, "media.encode_seconds")).collect();
    for (question, (answer, prompt, cached)) in questions.iter().zip(&answers) {
        println!("fan-out: {question:?}: {prompt} prompt tokens, {cached} cached -> {answer:?}");
    }
    println!("fan-out: encode seconds per question, in admission order: {encodes:?}");
    assert_eq!(admitted.len(), questions.len(), "one admitted line per question: {admitted:?}");
    assert_eq!(encodes.iter().filter(|&&s| s > 0.0).count(), 1, "one question encodes the picture: {encodes:?}");
    for (question, (answer, _, _)) in questions.iter().zip(&answers) {
        assert!(answer.contains("47"), "{question:?}: the answer {answer:?} names the number");
    }
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, its vision companion, and the card exclusively"]
fn a_follow_up_turn_reuses_the_prompt_and_a_fan_out_encodes_the_picture_once() {
    events();
    let Some(artifact) = artifact() else { return };
    let Some(live) = SwitchingLiveServer::start(options(), &artifact, MODEL) else { return };
    a_follow_up_turn_resumes_past_the_picture(&live);
    a_fan_out_encodes_the_picture_once(&live);
}
