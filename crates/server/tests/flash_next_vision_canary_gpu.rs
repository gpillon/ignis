//! Flash-Next answers a small image (GitHub #557, ADR 0087): one load with
//! vision on, built the way `main` builds it (`load::prepare_model` then
//! `load::load_model`: the vision companion opened beside the artifact and
//! held to it, the tower staged from host memory under the RTX 5090
//! profile's `host` residency, the encode in the prefill scratch), answers
//! the four vision canary images through chat completions and through
//! Responses alike.
//!
//! The images are the 27B canary's (`fixtures/vision_canary/`: a number, a
//! colour, circles to count, a line of text), each small enough to stay
//! below the indexer's dense range. The fixture's recorded answers are the
//! 27B reference's greedy tokens, which do not apply to Flash-Next, so each
//! image is held to the answer it shows -- 47, red, three, `config.yaml not
//! found` -- in the greedy text, thinking off, with the loop guard off
//! (testing.md, "GPU tests that compare output or load an option"). Every
//! answer is printed beside its verdict.
//!
//! Run under `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`): it
//! needs the real artifact and its vision companion
//! (`F:/ai/models/Qwen3.8-Flash-Next-ignis/`, or `IGNIS_FLASH_NEXT_DIR`), the
//! card exclusively, and about 40 GiB of host RAM for the expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::gpu_profile;
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;

use support::live_server::SwitchingLiveServer;
use support::media::data_uri;

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";

/// One canary image, its question, and what a correct answer holds.
struct Canary {
    image: &'static str,
    question: &'static str,
    /// Every needle must appear in the lowercased answer; each entry holds
    /// the spellings any one of which counts.
    needles: &'static [&'static [&'static str]],
}

const CANARIES: [Canary; 4] = [
    Canary {
        image: "number.png",
        question: "What number is shown in the image? Answer in one short sentence.",
        needles: &[&["47"]],
    },
    Canary {
        image: "colour.png",
        question: "What colour is the square in the image? Answer in one short sentence.",
        needles: &[&["red"]],
    },
    Canary {
        image: "circles.png",
        question: "How many circles are in the image? Answer in one short sentence.",
        needles: &[&["3", "three"]],
    },
    Canary {
        image: "text.png",
        question: "What does the text in the image say? Answer in one short sentence.",
        needles: &[&["config.yaml"], &["not found"]],
    },
];

/// The start options: vision on (a built-in profile leaves it off on
/// Flash-Next), the RTX 5090 profile's `host` residency, a context the
/// canary needs, no retained host slots and no idle unload, the loop guard
/// off.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = [
        "--vision-enabled",
        "--model-max-context",
        "16384",
        "--reuse-retained-host",
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

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision_canary").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The text of a chat completion's first choice.
fn chat_text(body: &Value) -> String {
    body["choices"][0]["message"]["content"].as_str().unwrap_or_else(|| panic!("no answer in {body}")).to_owned()
}

/// The text of a response's message output.
fn responses_text(body: &Value) -> String {
    let output = body["output"].as_array().unwrap_or_else(|| panic!("no output in {body}"));
    output
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str().map(str::to_owned))
        .collect()
}

fn post(runtime: &tokio::runtime::Runtime, url: &str, body: Value) -> Value {
    runtime.block_on(async {
        let response = reqwest::Client::new()
            .post(url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .expect("the request");
        let status = response.status();
        let text = response.text().await.expect("the body");
        assert!(status.is_success(), "{url}: {status}: {text}");
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{url}: not JSON ({e}): {text}"))
    })
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, its vision companion, and the card exclusively"]
fn flash_next_answers_the_four_canary_images_on_both_apis() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let artifact = dir.join(ARTIFACT_FILE_NAME);
    let companion = ignis_core::flash_next_vision::companion_path(&dir);
    for path in [&artifact, &companion] {
        if !path.exists() && gpu_profile::skip_or_fail(&format!("absent: {}", path.display())) {
            return;
        }
    }
    let Some(live) = SwitchingLiveServer::start(options(), &artifact.to_string_lossy(), MODEL) else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().expect("a client runtime");
    let mut wrong = Vec::new();
    for canary in &CANARIES {
        let image = data_uri(&fixture(canary.image));
        let chat = post(
            &runtime,
            &format!("{}/v1/chat/completions", live.url),
            json!({
                "model": MODEL,
                "temperature": 0,
                "enable_thinking": false,
                "max_tokens": 64,
                "messages": [{
                    "role": "user",
                    "content": [
                        { "type": "text", "text": canary.question },
                        { "type": "image_url", "image_url": { "url": image } }
                    ]
                }]
            }),
        );
        let responses = post(
            &runtime,
            &format!("{}/v1/responses", live.url),
            json!({
                "model": MODEL,
                "temperature": 0,
                "enable_thinking": false,
                "max_output_tokens": 64,
                "input": [{
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": canary.question },
                        { "type": "input_image", "image_url": image }
                    ]
                }]
            }),
        );
        for (api, answer) in [("chat completions", chat_text(&chat)), ("responses", responses_text(&responses))] {
            let lower = answer.to_lowercase();
            let right = canary.needles.iter().all(|spellings| spellings.iter().any(|needle| lower.contains(needle)));
            println!("{} via {api}: {} -- {answer:?}", canary.image, if right { "right" } else { "WRONG" });
            if !right {
                wrong.push(format!("{} via {api}: {answer:?}", canary.image));
            }
        }
    }
    assert!(wrong.is_empty(), "answered wrongly: {wrong:#?}");
}
