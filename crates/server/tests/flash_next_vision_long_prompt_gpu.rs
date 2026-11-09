//! A long prompt with an image early in it, on Qwen3.8-Flash-Next (GitHub
//! #558, spec #555, ADR 0087): the image's tokens open the prompt and more
//! than the indexer's dense range of text (2,051 tokens) follows them, so the
//! last rows read the image only through the indexer's selection -- its
//! queries rotated at their three-axis positions, the image's blocks' keys at
//! their first tokens' -- and the question about the image still gets the
//! image's answer: the canary's number (47) and its square's colour (red).
//!
//! The production load path, through the HTTP surface: the server's options
//! resolved from flags as `main` resolves them (`--vision-enabled`, the
//! vision companion beside the main container), `load::prepare_model` and
//! `load::load_model`, then `POST /v1/chat/completions` with the image as a
//! `data:` URI. Greedy, thinking off, the loop guard off (testing.md: a test
//! that reads output). The images are `tests/fixtures/vision_canary`'s; their
//! answers there are the 27B's reference, so this reads the fact, not the
//! tokens.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`) with its vision companion. Explicit GPU profile
//! (ADR 0006): a missing artifact or GPU is a skip outside it, a failure
//! under it.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Duration;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;

use support::live_server::SwitchingLiveServer;
use support::media::data_uri;
use support::vision_canary::fixture_dir;

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";
/// The indexer's dense range: a row with more visible tokens reads a
/// selection of blocks (`Geometry::dense_threshold`, budget 2,048 + 3).
const DENSE_RANGE: usize = 2051;

/// `main`'s options for a vision start, at a context the prompt fits.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = [
        "--model-max-context",
        "16384",
        "--vision-enabled",
        "--model-loop-guard",
        "off",
        "--reuse-retained-host",
        "0",
        "--model-idle-unload",
        "0",
    ]
    .map(String::from)
    .to_vec();
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        _ => unreachable!("flags without --help are a config"),
    }
}

/// Text past the dense range, with no digit and no colour word in it: one
/// sentence after another from small word lists, varied so that nothing
/// repeats for long.
fn filler(sentences: usize) -> String {
    const SUBJECTS: [&str; 9] =
        ["The archivist", "A tired courier", "The night clerk", "An old surveyor", "The ferry pilot", "A quiet engineer",
         "The librarian", "A travelling cook", "The lighthouse keeper"];
    const VERBS: [&str; 8] = ["catalogued", "carried", "inspected", "described", "repaired", "measured", "copied", "folded"];
    const OBJECTS: [&str; 10] =
        ["a ledger of tide tables", "the brass compass", "a bundle of letters", "the long survey map", "a wooden crate",
         "the station logbook", "a canvas satchel", "the harbour charts", "a set of keys", "the weather notes"];
    const PLACES: [&str; 7] =
        ["before the morning train", "beside the northern pier", "under the stone arch", "in the quiet reading room",
         "near the old customs house", "while the rain kept falling", "after the market closed"];
    let mut out = String::new();
    let mut state = 0x2545F491u32;
    for _ in 0..sentences {
        let mut pick = |n: usize| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as usize % n
        };
        let sentence =
            format!("{} {} {} {}. ", SUBJECTS[pick(9)], VERBS[pick(8)], OBJECTS[pick(10)], PLACES[pick(7)]);
        out.push_str(&sentence);
    }
    out
}

fn ask(runtime: &tokio::runtime::Runtime, url: &str, image: &str, text: &str) -> (String, u64) {
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": image } },
            { "type": "text", "text": text },
        ]}],
        "max_tokens": 32,
        "temperature": 0,
        "enable_thinking": false,
    });
    runtime.block_on(async {
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
        let content = reply["choices"][0]["message"]["content"].as_str().expect("an answer").to_owned();
        (content, reply["usage"]["prompt_tokens"].as_u64().expect("prompt tokens"))
    })
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its vision companion"]
fn a_long_prompt_reads_an_image_early_in_it_past_the_dense_range() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    let artifact = path.to_string_lossy().into_owned();
    let frontend = FrontendSet::from_reader(&Reader::open(&path).expect("open the artifact")).expect("frontend");
    let long = filler(260);
    let after_image = frontend.tokenizer().encode(&long).expect("tokenize the filler").len();
    println!("text after the image: {after_image} tokens");
    assert!(after_image > DENSE_RANGE + 200, "the text after the image passes the indexer's dense range");
    let Some(live) = SwitchingLiveServer::start(options(), &artifact, MODEL) else { return };
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("client runtime");

    let cases = [
        ("number.png", "What number is shown in the image at the start of this message? Answer with the number only.", "47"),
        ("colour.png", "What colour is the square in the image at the start of this message? Answer with one word.", "red"),
    ];
    for (image, question, fact) in cases {
        let bytes = std::fs::read(fixture_dir().join(image)).expect("the canary image");
        let text = format!("Here is a long log to keep in mind.\n\n{long}\n\n{question}");
        let (answer, prompt_tokens) = ask(&runtime, &live.url, &data_uri(&bytes), &text);
        println!("{image}: {prompt_tokens} prompt tokens -> {answer:?}");
        assert!(prompt_tokens as usize > after_image, "{image}: the prompt holds the image and the text");
        assert!(answer.to_lowercase().contains(fact), "{image}: the answer {answer:?} names {fact}");
    }
}
