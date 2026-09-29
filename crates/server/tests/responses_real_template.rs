//! The Responses warm-up on the **real frontend** (GitHub #282): Codex's
//! prewarm — `generate: false` with `instructions` and a tool and no input —
//! against the artifact's own chat template, which refuses to render a
//! conversation without a user query.
//!
//! CPU-only — the backend is the mock; only the tokenizer and the template
//! are real. Skips when the artifact is not at its machine-local path, the
//! convention `checkpoint_lineage.rs` follows.

#[path = "support/responses.rs"]
mod responses;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::Engine;
use ignis_server::Server;
use serde_json::json;

use responses::{live, next_event, send, socket, terminal, Socket};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// The terminal event of the response on the default stream.
async fn ends(socket: &mut Socket) -> serde_json::Value {
    loop {
        let event = next_event(socket).await;
        if terminal(&event) {
            return event;
        }
    }
}

#[tokio::test]
async fn codexs_prewarm_keeps_the_system_block_on_the_real_template() {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = ignis_artifact::Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let frontend = ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set");
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(ArtifactTemplateProvider::new(frontend)))
        .with_request_timeout(Duration::from_secs(30));
    let live = live(server).await;
    let mut socket = socket(&live).await;

    let instructions = "You are a coding agent. Work in the repository, run the tests, and report what changed.";
    let tools = json!([{ "type": "function", "name": "shell", "description": "Run a shell command.",
        "parameters": { "type": "object", "properties": { "command": { "type": "string" } }, "required": ["command"] } }]);
    send(&mut socket, json!({
        "type": "response.create", "model": MODEL, "generate": false,
        "instructions": instructions, "tools": tools, "input": [],
    }))
    .await;
    let warmed = ends(&mut socket).await;
    assert_eq!(warmed["type"], "response.completed", "{warmed}");
    assert_eq!(warmed["response"]["output"], json!([]));
    let block = warmed["response"]["usage"]["input_tokens"].as_u64().unwrap();

    send(&mut socket, json!({
        "type": "response.create", "model": MODEL, "max_output_tokens": 1,
        "previous_response_id": warmed["response"]["id"],
        "instructions": instructions, "tools": tools, "input": "list the files",
    }))
    .await;
    let turn = ends(&mut socket).await;
    assert_eq!(turn["type"], "response.incomplete", "{turn}");
    let usage = &turn["response"]["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > block, "the turn extends the warmed block: {usage}");
    let cached = usage["input_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(cached > 0 && cached <= block, "the turn stood on the warmed system block ({block} tokens): {usage}");
}
