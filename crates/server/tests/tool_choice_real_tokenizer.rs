//! The forced tool-call opening against the **real tokenizer** (GitHub #286,
//! spec server/11 acceptance 5): the ids the server forces for `tool_choice`
//! are a prefix of the ids every recorded tool call tokenizes to, so the
//! model is never made to write a split of its own dialect it would not
//! write itself (the token-healing problem).
//!
//! The calls are the chat-render fixtures' (`crates/artifact/tests/fixtures/
//! chat_render/tool_call_*.json`, #121's), the system prompt's own example
//! call included. What is forced is read off the mock's permitted sets, so
//! this checks what the server forces and not a restatement of it.
//!
//! CPU-only — the backend is the mock; the tokenizer and the template are
//! real. Skips when the artifact is not at its machine-local path, the
//! convention `checkpoint_lineage.rs` follows.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig, TokenId};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::Engine;
use ignis_server::Server;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const FIXTURES: [&str; 4] = [
    "tool_call_empty_arguments",
    "tool_call_key_order",
    "tool_call_reversed_keys",
    "tool_call_two_calls",
];

fn frontend() -> ignis_artifact::FrontendSet {
    let reader = ignis_artifact::Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set")
}

/// POST one chat request forcing `tool_choice` over a single tool `name`;
/// its status, and the ids the server forced, in order.
async fn forced(app: &axum::Router, mock: &MockCompute, name: &str, tool_choice: JsonValue, thinking: bool) -> (u16, Vec<TokenId>) {
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [{ "type": "function", "function": { "name": name, "parameters": { "type": "object", "properties": {} } } }],
        "tool_choice": tool_choice,
        "enable_thinking": thinking,
        "max_tokens": 40,
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: JsonValue = serde_json::from_slice(&bytes).unwrap();
    let Some(id) = body["id"].as_str().and_then(|id| id.strip_prefix("chatcmpl-")) else {
        return (status, Vec::new());
    };
    let id: u64 = id.parse().unwrap();
    let first = mock.prefill_calls().concat().into_iter().filter(|job| job.request == id).filter_map(|job| job.permitted);
    let rest = mock.decode_calls().concat().into_iter().filter(|job| job.request == id).filter_map(|job| job.permitted);
    (status, first.chain(rest).map(|set| set[0]).collect())
}

#[tokio::test]
async fn the_forced_openings_are_prefixes_of_every_recorded_call() {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let tokenizer_set = frontend();
    let tokenizer = tokenizer_set.tokenizer();
    let mock = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        mock.clone(),
    );
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(ArtifactTemplateProvider::new(frontend())))
        .with_request_timeout(Duration::from_secs(30))
        .app();
    let open = tokenizer.encode("<tool_call>").unwrap();
    assert_eq!(open.len(), 1, "<tool_call> is one token");
    let mut checked = 0;
    for fixture in FIXTURES {
        let path = format!("{}/../artifact/tests/fixtures/chat_render/{fixture}.json", env!("CARGO_MANIFEST_DIR"));
        let case: JsonValue = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let ids = tokenizer.encode(case["expected"]["text"].as_str().unwrap()).unwrap();
        for at in (0..ids.len()).filter(|&at| ids[at] == open[0]) {
            let after = tokenizer.decode(&ids[at..(at + 16).min(ids.len())]).unwrap();
            let Some((name, _)) = after.strip_prefix("<tool_call>\n<function=").and_then(|rest| rest.split_once('>')) else {
                continue;
            };
            let (status, required) = forced(&app, &mock, name, json!("required"), false).await;
            assert_eq!(status, 200);
            assert!(ids[at..].starts_with(&required), "{fixture}: required forces {required:?}, the call is {:?}", &ids[at..at + 12]);
            let named = json!({ "type": "function", "function": { "name": name } });
            let (status, named) = forced(&app, &mock, name, named, false).await;
            assert_eq!(status, 200);
            assert!(ids[at..].starts_with(&named), "{fixture}: {name} forces {named:?}, the call is {:?}", &ids[at..at + 12]);
            assert!(named.len() > required.len(), "the name is forced too");
            checked += 1;
        }
    }
    // Two read_file calls, two edits, a now, and the system prompt's own
    // example call (the one fixture that renders tools).
    assert_eq!(checked, 6, "every recorded call was checked");
}

/// Thinking on: the tokenizer has the single-token `</think>` the scheduler
/// sees the block close by, and the opening starts `<tool_call>`, `\n` — the
/// joiner — so the server accepts the request rather than refusing it.
#[tokio::test]
async fn with_thinking_on_the_real_tokenizer_can_force_after_the_block() {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let mock = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        mock.clone(),
    );
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(ArtifactTemplateProvider::new(frontend())))
        .with_request_timeout(Duration::from_secs(30))
        .app();
    let (status, sets) = forced(&app, &mock, "read_file", json!("required"), true).await;
    assert_eq!(status, 200);
    assert!(sets.is_empty(), "nothing is forced while the block is open: {sets:?}");
}
