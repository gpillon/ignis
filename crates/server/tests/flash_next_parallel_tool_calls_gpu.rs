//! `parallel_tool_calls: false` returns one call per turn on the served
//! Flash-Next (GitHub #617 and #609, spec #582): on chat completions and on
//! the Responses API, a conversation whose model makes several calls in one
//! turn when it may (the control, without the field) makes them one turn at
//! a time under `false`. Flash-Next holds no tool (`strict` is inert there,
//! #581), so the turn ends at the close of its first call as the call
//! parser reads it. Every turn and every call is printed.
//!
//! Served through the production load path (`load::prepare_model` then
//! `load::load_model`) and `SwitchingLiveServer`, on the RTX 5090 profile's
//! settings, the loop guard off, no idle unload, no KV-RAM or KV-disk tier.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test flash_next_parallel_tool_calls_gpu -- --ignored --nocapture
//! ```
//!
//! Run under `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`): it needs
//! the real artifact (`F:/ai/models/Qwen3.8-Flash-Next-ignis/`, or
//! `IGNIS_FLASH_NEXT_DIR`), the card exclusively, and about 40 GiB of host
//! RAM for the expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Duration;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::gpu_profile;

#[path = "support/mod.rs"]
mod support;

#[path = "support/held_tools_corpus.rs"]
mod corpus;

use corpus::Api;
use support::live_server::SwitchingLiveServer;

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";

/// `main`'s options: the RTX 5090 profile's (the built-in default on that
/// card), a context the conversations fit, no KV-RAM or KV-disk tier, no
/// idle unload, the loop guard off.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = [
        "--model-max-context",
        "16384",
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

/// `body` posted to `path` on the server at `url`: the status and the body
/// back.
async fn send(url: String, path: &'static str, body: String) -> (u16, String) {
    let response = reqwest::Client::new()
        .post(format!("{url}{path}"))
        .header("content-type", "application/json")
        .body(body)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .expect("the request");
    let status = response.status().as_u16();
    (status, response.text().await.expect("the body"))
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and the card exclusively"]
fn with_parallel_tool_calls_false_flash_next_makes_one_call_per_turn() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let artifact = dir.join(ARTIFACT_FILE_NAME);
    if !artifact.exists() && gpu_profile::skip_or_fail(&format!("absent: {}", artifact.display())) {
        return;
    }
    let Some(live) = SwitchingLiveServer::start(options(), &artifact.to_string_lossy(), MODEL) else { return };
    let url = live.url.clone();
    let post = |path: &'static str, body: String| send(url.clone(), path, body);
    for api in [Api::Chat, Api::Responses] {
        // As a client sends them: `strict: true`, inert on Flash-Next.
        let tools: Vec<String> = corpus::PARALLEL_TOOLS.iter().map(|tool| if api == Api::Chat { (*tool).to_owned() } else { corpus::flat(tool) }).collect();
        let control = live.block_on(corpus::conversation(&post, MODEL, api, &tools, corpus::PARALLEL_PROMPT, None));
        let limited = live.block_on(corpus::conversation(&post, MODEL, api, &tools, corpus::PARALLEL_PROMPT, Some(false)));
        corpus::one_call_per_turn(&format!("Flash-Next, {api:?}"), api, &control, &limited);
    }
}
