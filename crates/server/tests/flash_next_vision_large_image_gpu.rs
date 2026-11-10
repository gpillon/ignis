//! Flash-Next answers a screenshot-sized image, and vision takes nothing from
//! its expert cache (GitHub #558, spec #555, ADR 0087).
//!
//! An item encodes in memory a Flash-Next load already reserves and leaves
//! idle while it runs -- the prefill scratch and, on a card that cannot hold
//! every expert, the expert staging ring -- and the positions an image's
//! tokens rotate at ride in buffers a text load reserves too. Two checks, each
//! through the production load path (`load::prepare_model` then
//! `load::load_model`, the vision companion beside the main container) and
//! `POST /v1/chat/completions`, with the `ignis.runtime.flash_next_plan`
//! event read off a sink this binary installs:
//!
//! - on the RTX 5090 profile's settings a text-only start and a vision start
//!   plan the same lines beside the expert cache -- so at the same free VRAM
//!   the same expert cache (each start reads the card's free memory itself,
//!   so the comparison is the budget less the cache: every other line) -- the
//!   same KV pool and the same staging ring, and the vision start's item
//!   bound is the processor's 16,384 tokens, whose image it answers;
//! - at a 1,024-token prefill chunk, an eighth of the profile's, the bound is
//!   still 16,384 and the image is answered: what the narrow chunk's scratch
//!   does not hold of the encode, the staging ring does (the leaf's bind test
//!   prints the bound such a scratch holds on its own).
//!
//! The image is a vision canary's (`fixtures/vision_canary`) scaled to 4096 x
//! 4096 pixels, nearest neighbour: 128 x 128 merged tokens, the item bound
//! exactly. Its answer is the fact it shows (47, red), greedy, thinking off,
//! the loop guard off (testing.md, "GPU tests that compare output or load an
//! option"); every answer and plan line is printed.
//!
//! Run under `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`), one test
//! at a time (`--test-threads=1`: the tests share this binary's event sink):
//! the real artifact and its vision companion
//! (`F:/ai/models/Qwen3.8-Flash-Next-ignis/`, or `IGNIS_FLASH_NEXT_DIR`), the
//! card exclusively, and about 40 GiB of host RAM for the expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

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
/// The processor's item bound: a 4096 x 4096 image, one merged token per
/// 32 x 32 pixels.
const ITEM_BOUND: u64 = 16_384;
const SIDE: u32 = 4096;

/// The events this binary's loads emit, as JSON lines, beside the usual
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

/// The last `ignis.runtime.flash_next_plan` event, its fields flattened.
fn last_plan() -> Value {
    let line = events()
        .lines()
        .into_iter()
        .rev()
        .find(|line| line.contains("ignis.runtime.flash_next_plan"))
        .expect("the start logged its VRAM plan");
    let event: Value = serde_json::from_str(&line).expect("json");
    let mut fields = event.get("attributes").cloned().unwrap_or_else(|| json!({}));
    for (key, value) in event.as_object().expect("an object") {
        fields.as_object_mut().expect("an object").entry(key.clone()).or_insert(value.clone());
    }
    fields
}

fn field(plan: &Value, name: &str) -> u64 {
    plan[name].as_u64().unwrap_or_else(|| panic!("no `{name}` in {plan}"))
}

/// `main`'s options: the RTX 5090 profile's (the built-in default on that
/// card) with `extra`, no retained host slots, no idle unload, the loop guard
/// off.
fn options(extra: &[&str]) -> ignis_server::config::Config {
    let mut args: Vec<String> =
        ["--reuse-retained-host", "0", "--model-idle-unload", "0", "--model-loop-guard", "off"].map(String::from).to_vec();
    args.extend(extra.iter().map(|flag| flag.to_string()));
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        _ => unreachable!("flags without --help are a config"),
    }
}

/// A canary image scaled to `SIDE` x `SIDE`, nearest neighbour, as a PNG.
fn screenshot_sized(name: &str) -> Vec<u8> {
    let bytes = std::fs::read(fixture_dir().join(name)).expect("the canary image");
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().expect("a PNG");
    let mut pixels = vec![0; reader.output_buffer_size().expect("a buffer size")];
    let info = reader.next_frame(&mut pixels).expect("its pixels");
    let channels = info.color_type.samples();
    let rgb = |x: u32, y: u32, c: usize| -> u8 {
        let at = (y as usize * info.width as usize + x as usize) * channels;
        if channels < 3 { pixels[at] } else { pixels[at + c] }
    };
    let mut scaled = Vec::with_capacity((SIDE * SIDE * 3) as usize);
    for y in 0..SIDE {
        let sy = y * info.height / SIDE;
        for x in 0..SIDE {
            let sx = x * info.width / SIDE;
            scaled.extend([rgb(sx, sy, 0), rgb(sx, sy, 1), rgb(sx, sy, 2)]);
        }
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, SIDE, SIDE);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().expect("header").write_image_data(&scaled).expect("pixels");
    }
    out
}

/// The answer and the prompt's tokens.
fn ask(runtime: &tokio::runtime::Runtime, url: &str, image: &str, question: &str) -> (String, u64) {
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": image } },
            { "type": "text", "text": question },
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

/// Both canary facts at the item bound, on a running vision start.
fn answers_the_screenshot_sized_images(live: &SwitchingLiveServer) {
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("client runtime");
    let cases = [
        ("number.png", "What number is shown in the image? Answer with the number only.", "47"),
        ("colour.png", "What colour is the square in the image? Answer with one word.", "red"),
    ];
    for (image, question, fact) in cases {
        let (answer, prompt_tokens) = ask(&runtime, &live.url, &data_uri(&screenshot_sized(image)), question);
        println!("{image} at {SIDE} x {SIDE}: {prompt_tokens} prompt tokens -> {answer:?}");
        assert!(prompt_tokens > ITEM_BOUND, "{image}: the prompt holds the image's {ITEM_BOUND} tokens");
        assert!(answer.to_lowercase().contains(fact), "{image}: the answer {answer:?} names {fact}");
    }
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

/// The plan's lines a text and a vision start share, printed beside each
/// other.
const SHARED: [&str; 8] = [
    "weights_bytes",
    "program_bytes",
    "pool_bytes",
    "kv_pool_pages",
    "residency_fixed_bytes",
    "prefill_staging_bytes",
    "verify_round_bytes",
    "media_embedding_bytes",
];

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, its vision companion, and the card exclusively"]
fn vision_on_and_off_plan_the_same_expert_cache_and_a_16384_token_image_is_answered() {
    events();
    let Some(artifact) = artifact() else { return };
    let text_plan = {
        let Some(_live) = SwitchingLiveServer::start(options(&[]), &artifact, MODEL) else { return };
        last_plan()
    };
    let Some(live) = SwitchingLiveServer::start(options(&["--qwen38flashnext-vision-enabled"]), &artifact, MODEL) else {
        return;
    };
    let vision_plan = last_plan();
    for name in SHARED {
        println!("{name}: text {}, vision {}", field(&text_plan, name), field(&vision_plan, name));
    }
    let beside = |plan: &Value| field(plan, "budget_bytes") - field(plan, "expert_cache_bytes");
    println!(
        "expert cache: text {} of a {}-byte budget, vision {} of {}; item bound {} ({:?} residency)",
        field(&text_plan, "expert_cache_bytes"),
        field(&text_plan, "budget_bytes"),
        field(&vision_plan, "expert_cache_bytes"),
        field(&vision_plan, "budget_bytes"),
        field(&vision_plan, "vision_item_max_tokens"),
        vision_plan["vision_residency"],
    );
    for name in SHARED {
        assert_eq!(field(&vision_plan, name), field(&text_plan, name), "{name}: vision moves no line");
    }
    assert_eq!(beside(&vision_plan), beside(&text_plan), "every byte beside the expert cache is the text start's");
    assert_eq!(vision_plan["vision_residency"], "host", "the RTX 5090 profile's residency");
    assert_eq!(field(&vision_plan, "vision_item_max_tokens"), ITEM_BOUND, "the processor's largest item");
    answers_the_screenshot_sized_images(&live);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, its vision companion, and the card exclusively"]
fn at_a_narrow_chunk_the_encode_takes_the_staging_ring_and_the_16384_token_image_is_answered() {
    events();
    let Some(artifact) = artifact() else { return };
    let flags = ["--qwen38flashnext-vision-enabled", "--qwen38flashnext-model-prefill-chunk", "1024"];
    let Some(live) = SwitchingLiveServer::start(options(&flags), &artifact, MODEL) else { return };
    let plan = last_plan();
    println!(
        "a 1,024-token chunk: program {} bytes, staging ring {} bytes, item bound {}",
        field(&plan, "program_bytes"),
        field(&plan, "prefill_staging_bytes"),
        field(&plan, "vision_item_max_tokens")
    );
    assert!(field(&plan, "prefill_staging_bytes") > 0, "a 32 GB card stages its prefill misses");
    assert_eq!(field(&plan, "vision_item_max_tokens"), ITEM_BOUND, "the ring lends the encode what the scratch lacks");
    answers_the_screenshot_sized_images(&live);
}
