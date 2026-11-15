//! Held tools on the served 27B (GitHub #611, spec #582, ADR 0094): chat
//! completions with **strict tools**, through the real router over the real
//! GPU-backed scheduler with DFlash2 on, the server's own grammars compiled
//! against the artifact's tokenizer (`ArtifactTemplateProvider::serving`).
//!
//! A corpus of agent requests whose tools are strict: free strings, an enum,
//! a pattern and a format; a long file to write; a file about tool calls,
//! whose content holds the dialect's own tags; two calls in one turn; a free
//! tool beside the strict ones; thinking on and off. Every call that comes
//! back names a declared tool, and a strict tool's arguments validate
//! against its `parameters` (#608's validator); every request makes a call;
//! the held calls' misses and repairs are counted (`type="tool"`), every
//! miss repaired; and the verified runs keep their drafts: the requests'
//! verify rounds drafted and accepted tokens. What each call wrote, and how
//! many misses each request had, is printed: the repairs are read, as the
//! ticket asks, the early close's above all.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test held_tools_gpu -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::Value as JsonValue;
use tower::ServiceExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::compute::ModelFamily;
use ignis_core::{gpu_profile, Speculation, SpeculativeBackend};
use ignis_grammar::json::Json;
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::Server;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// DFlash2-7, every built-in profile's window.
const WINDOW: u32 = 7;

const WRITE_FILE: &str = r#"{"type":"function","function":{"name":"write_file","description":"Write a text file, replacing it or appending to it.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"The file's path, relative to the project."},"content":{"type":"string","description":"The text to write."},"mode":{"type":"string","enum":["replace","append"],"description":"Whether to replace the file or append to it."}},"required":["path","content","mode"],"additionalProperties":false}}}"#;
const READ_FILE: &str = r#"{"type":"function","function":{"name":"read_file","description":"Read a text file.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"The file's path, relative to the project."}},"required":["path"],"additionalProperties":false}}}"#;
const MEETING: &str = r#"{"type":"function","function":{"name":"book_meeting","description":"Book a meeting room for a day.","strict":true,"parameters":{"type":"object","properties":{"room":{"type":"string","pattern":"^R[0-9]{3}$","description":"The room, R and three digits."},"day":{"type":"string","format":"date","description":"The day, as YYYY-MM-DD."},"title":{"type":"string","description":"The meeting's title."}},"required":["room","day","title"],"additionalProperties":false}}}"#;
/// A tool that is not strict: its arguments stay free.
const SEARCH: &str = r#"{"type":"function","function":{"name":"search","description":"Search the web.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}"#;

/// The agent requests: a prompt, the tools sent, thinking on or off.
const CASES: &[(&str, &[&str], bool)] = &[
    ("Save a file notes/todo.md with a checklist of three things to do this week.", &[WRITE_FILE, READ_FILE], false),
    ("Write a Python script primes.py that prints the first twenty prime numbers, with a docstring and comments.", &[WRITE_FILE], false),
    (
        "Write docs/dialect.md: a short explanation of the tool-call format you use, with a complete example call inside a fenced block.",
        &[WRITE_FILE],
        false,
    ),
    ("Read both README.md and Cargo.toml.", &[READ_FILE, WRITE_FILE], false),
    ("Book room R204 on 2026-12-03 for the quarterly planning meeting.", &[MEETING, SEARCH], false),
    ("Append a line saying 'release cut' to CHANGES.txt.", &[WRITE_FILE, READ_FILE, SEARCH], true),
];

/// A live server over the real scheduler with DFlash2, its grammars the
/// load's own; the router dropped before the model thread is joined, so the
/// next load never allocates beside this one.
struct Harness {
    server: Option<Server>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.server.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

fn harness() -> Option<Harness> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let shape = EngineShape {
        speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, WINDOW).expect("dflash2-7")),
        ..EngineShape::default()
    };
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok((scheduler, _reservations)) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let provider = ignis_server::artifact_template::ArtifactTemplateProvider::serving(frontend, ModelFamily::Qwen38_27b, ignis_grammar::GrammarLimits::default());
    let server = Server::new(engine, Box::new(provider)).with_metrics().with_request_timeout(Duration::from_secs(600));
    Some(Harness { server: Some(server), driver: Some(driver) })
}

async fn chat(server: &Server, prompt: &str, tools: &[&str], thinking: bool) -> JsonValue {
    let body = format!(
        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":{}}}],"tools":[{}],"enable_thinking":{thinking},"temperature":0,"max_tokens":3072}}"#,
        JsonValue::String(prompt.into()),
        tools.join(",")
    );
    let request = Request::builder().method("POST").uri("/v1/chat/completions").header("content-type", "application/json").body(Body::from(body)).unwrap();
    let response = server.app().oneshot(request).await.expect("the router answers");
    let status = response.status().as_u16();
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    assert_eq!(status, 200, "{prompt}: {text}");
    serde_json::from_str(&text).unwrap()
}

/// A counter's value in the exposition, 0 when absent.
fn counter(exposition: &str, series: &str) -> u64 {
    exposition.lines().find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok()).unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn strict_tools_on_the_27b_with_dflash2_return_valid_calls_their_repairs_counted_and_drafts_kept() {
    let Some(harness) = harness() else { return };
    let server = harness.server.as_ref().unwrap();
    let metrics = server.metrics.clone().expect("metrics on");
    for (prompt, tools, thinking) in CASES {
        let before = metrics.render();
        let answer = chat(server, prompt, tools, *thinking).await;
        let choice = &answer["choices"][0];
        let calls = choice["message"]["tool_calls"].as_array().cloned().unwrap_or_default();
        let after = metrics.render();
        let misses = counter(&after, "ignis_structured_misses_total{type=\"tool\"}") - counter(&before, "ignis_structured_misses_total{type=\"tool\"}");
        eprintln!("{prompt}\n  finish {}, {} calls, {misses} misses", choice["finish_reason"], calls.len());
        assert!(!calls.is_empty(), "{prompt}: no call: {answer}");
        assert_eq!(choice["finish_reason"], "tool_calls", "{prompt}: {answer}");
        for call in &calls {
            let (name, arguments) = (call["function"]["name"].as_str().unwrap(), call["function"]["arguments"].as_str().unwrap());
            eprintln!("  {name} {arguments}");
            let tool = tools
                .iter()
                .map(|tool| Json::parse(tool).unwrap())
                .find(|tool| tool.get("function").and_then(|f| f.get("name")).and_then(Json::as_str) == Some(name))
                .unwrap_or_else(|| panic!("{prompt}: a call to {name}, which the request does not declare"));
            let function = tool.get("function").unwrap();
            if function.get("strict") == Some(&Json::Bool(true)) {
                let parameters = function.get("parameters").unwrap();
                let arguments = Json::parse(arguments).unwrap_or_else(|e| panic!("{prompt}: {name}'s arguments are not JSON ({e}): {arguments}"));
                let violations = ignis_grammar::validate::violations(parameters, &arguments);
                assert!(violations.is_empty(), "{prompt}: {name} {arguments:?}: {violations:?}");
            }
        }
    }
    let exposition = metrics.render();
    let (requests, misses, repairs) = (
        counter(&exposition, "ignis_structured_requests_total{type=\"tool\"}"),
        counter(&exposition, "ignis_structured_misses_total{type=\"tool\"}"),
        counter(&exposition, "ignis_structured_repairs_total{type=\"tool\"}"),
    );
    let (drafted, accepted) = (counter(&exposition, "ignis_speculative_drafted_tokens_total"), counter(&exposition, "ignis_speculative_accepted_tokens_total"));
    eprintln!("tool requests {requests}, misses {misses}, repairs {repairs}; drafted {drafted}, accepted {accepted}");
    assert_eq!(requests, CASES.len() as u64, "every request was held");
    assert_eq!(repairs, misses, "every miss repaired");
    assert!(drafted > 0 && accepted > 0, "the held requests' verify rounds kept their drafts: {drafted} drafted, {accepted} accepted");
}
