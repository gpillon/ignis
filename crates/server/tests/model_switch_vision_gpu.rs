//! A model switch keeps vision on the card (GitHub #563, ADR 0087): one
//! process started on the 27B with vision on answers the canary images, is
//! switched to Flash-Next through `POST /v1/models/switch` and answers them
//! again, is switched back and answers a third time. Then a switch to a
//! copy of Flash-Next's artifact with no vision companion beside it answers
//! `409 vision_companion_missing` naming the file, and the 27B still answers
//! an image.
//!
//! The copy is made of hard links to the artifact and its sidecar (no 72 GB
//! copy), beside the real directory, and removed at the end; where the two
//! are not on one volume the test fails saying so.
//!
//! Each image is held to the answer it shows (47, red, three,
//! `config.yaml not found`), in the greedy text with thinking off, as
//! `flash_next_vision_canary_gpu.rs` holds Flash-Next alone. Run under
//! `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`): both artifacts,
//! Flash-Next's companion, the card exclusively. Machine-local paths: the
//! 27B at [`ARTIFACT_27B`] (or `IGNIS_ARTIFACT_27B`), Flash-Next under
//! [`FLASH_NEXT_DIR`] (or `IGNIS_FLASH_NEXT_DIR`).

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::{sidecar_path, ARTIFACT_FILE_NAME, VISION_ARTIFACT_FILE_NAME};
use ignis_core::gpu_profile;
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;

use support::live_server::SwitchingLiveServer;
use support::media::data_uri;

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL_27B: &str = "qwen3.8-27b";
const MODEL_FLASH_NEXT: &str = "qwen3.8-flash-next";

/// `(image, question, the spellings of each needle)`.
type Canary = (&'static str, &'static str, &'static [&'static [&'static str]]);

const CANARIES: [Canary; 4] = [
    ("number.png", "What number is shown in the image? Answer in one short sentence.", &[&["47"]]),
    ("colour.png", "What colour is the square in the image? Answer in one short sentence.", &[&["red"]]),
    ("circles.png", "How many circles are in the image? Answer in one short sentence.", &[&["3", "three"]]),
    ("text.png", "What does the text in the image say? Answer in one short sentence.", &[&["config.yaml"], &["not found"]]),
];

/// Vision on for both models, a context both serve, no retained host slots,
/// no idle unload, the loop guard off.
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

/// Chat completions asking `model` about one image: the status and the body.
fn ask(live: &SwitchingLiveServer, model: &str, image: &str, question: &str) -> (u16, Value) {
    let image = data_uri(&fixture(image));
    live.block_on(async {
        let response = reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", live.url))
            .header("content-type", "application/json")
            .body(
                json!({
                    "model": model,
                    "temperature": 0,
                    "enable_thinking": false,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": [
                        { "type": "text", "text": question },
                        { "type": "image_url", "image_url": { "url": image } }
                    ]}]
                })
                .to_string(),
            )
            .send()
            .await
            .expect("the request");
        let status = response.status().as_u16();
        let text = response.text().await.expect("the body");
        (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
    })
}

/// Every canary answered with what it shows by `model`; the verdicts are
/// printed, the wrong ones collected.
fn answers_all(live: &SwitchingLiveServer, model: &str, wrong: &mut Vec<String>) {
    for (image, question, needles) in CANARIES {
        let (status, body) = ask(live, model, image, question);
        let answer = body["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_lowercase();
        let right = status == 200 && needles.iter().all(|spellings| spellings.iter().any(|needle| answer.contains(needle)));
        println!("{model}: {image}: {} -- {status} {answer:?}", if right { "right" } else { "WRONG" });
        if !right {
            wrong.push(format!("{model}: {image}: {status} {body}"));
        }
    }
}

#[test]
#[ignore = "GPU profile only: the real 27B and Flash-Next artifacts, Flash-Next's companion, and the card exclusively"]
fn vision_crosses_a_switch_between_the_27b_and_flash_next_and_a_missing_companion_is_a_409() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    let artifact_27b = std::env::var("IGNIS_ARTIFACT_27B").unwrap_or_else(|_| ARTIFACT_27B.to_owned());
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let flash_next = dir.join(ARTIFACT_FILE_NAME);
    let companion = ignis_core::flash_next_vision::companion_path(&dir);
    for path in [&flash_next, &companion] {
        if !path.exists() && gpu_profile::skip_or_fail(&format!("absent: {}", path.display())) {
            return;
        }
    }
    let flash_next_s = flash_next.to_string_lossy().into_owned();

    let Some(live) = SwitchingLiveServer::start(options(), &artifact_27b, MODEL_27B) else {
        return;
    };
    let mut wrong = Vec::new();
    answers_all(&live, MODEL_27B, &mut wrong);
    println!("27B -> Flash-Next: {:?}", live.switch_to(&flash_next_s, MODEL_FLASH_NEXT));
    answers_all(&live, MODEL_FLASH_NEXT, &mut wrong);
    println!("Flash-Next -> 27B: {:?}", live.switch_to(&artifact_27b, MODEL_27B));
    answers_all(&live, MODEL_27B, &mut wrong);

    // The same artifact with no companion beside it: hard links to the
    // container and its sidecar, none to the companion.
    let name = dir.file_name().expect("a directory name").to_string_lossy().into_owned();
    let bare = dir.with_file_name(format!("{name}-no-companion-{}", std::process::id()));
    std::fs::create_dir_all(&bare).expect("the copy's directory");
    let linked = bare.join(ARTIFACT_FILE_NAME);
    for (from, to) in [(flash_next.clone(), linked.clone()), (sidecar_path(&flash_next), sidecar_path(&linked))] {
        std::fs::hard_link(&from, &to)
            .unwrap_or_else(|e| panic!("hard-link {} (one volume with the model directory?): {e}", from.display()));
    }
    let refused = live.block_on(async {
        let response = reqwest::Client::new()
            .post(format!("{}/v1/models/switch", live.url))
            .header("content-type", "application/json")
            .body(json!({ "artifact": linked.to_string_lossy(), "model": MODEL_FLASH_NEXT }).to_string())
            .send()
            .await
            .expect("the switch request");
        let status = response.status().as_u16();
        (status, serde_json::from_str::<Value>(&response.text().await.expect("the body")).unwrap_or(Value::Null))
    });
    println!("switch to Flash-Next without its companion: {refused:?}");
    let message = refused.1["error"]["message"].as_str().unwrap_or_default().to_owned();
    // The 27B was never torn down: it answers an image still.
    answers_all(&live, MODEL_27B, &mut wrong);
    let _ = std::fs::remove_dir_all(&bare);
    assert_eq!(refused.0, 409, "{refused:?}");
    assert_eq!(refused.1["error"]["code"], "vision_companion_missing", "{refused:?}");
    assert!(message.contains(VISION_ARTIFACT_FILE_NAME), "names the file: {message}");
    assert!(wrong.is_empty(), "answered wrongly: {wrong:#?}");
}
