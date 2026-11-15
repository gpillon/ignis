//! `parallel_tool_calls: false` (GitHub #609, spec #582): the turn ends at
//! the close of its first call, so a client gets at most one call per turn,
//! on chat completions (streamed or not) and on the Responses API (answered
//! whole, streamed, and over the WebSocket mode), on the 27B and on
//! Flash-Next alike. `true` or absent serves every call the model writes.
//!
//! Over the in-process router on `MockCompute` and a scripted template, so
//! the mock's tokens read as chosen text (ADR 0006).

#[path = "support/responses.rs"]
mod responses;

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::compute::ModelFamily;
use ignis_core::mock::GatedCompute;
use ignis_core::{Compute, MockCompute, SchedulerConfig};
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use responses::{events_until, live, metric_reaches, send, server, socket, sse_events, terminal, Script, MODEL};

const FIRST: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
/// A second call, whose argument no answer that ends at the first call
/// carries anywhere in its bytes.
const SECOND: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\nnever.txt\n</parameter>\n</function>\n</tool_call>";
const SECOND_MARK: &str = "never.txt";

/// Two calls, one a token apart: text, the first call, the newline the
/// template writes between calls, the second.
const SPLIT: &[&str] = &["Reading. ", FIRST, "\n", SECOND];
/// Both calls in the text of one token.
const ONE_TOKEN: &[&str] = &["Reading. ", "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read_file>\n<parameter=path>\nnever.txt\n</parameter>\n</function>\n</tool_call>"];

/// The cap a turn that ends at its first call never reaches.
const CAP: u32 = 4000;

#[derive(Clone, Copy, Debug)]
enum Surface {
    Chat,
    ChatStream,
    Responses,
    ResponsesStream,
    Socket,
}

const SURFACES: [Surface; 5] = [Surface::Chat, Surface::ChatStream, Surface::Responses, Surface::ResponsesStream, Surface::Socket];

/// How a turn came back, whatever the surface.
#[derive(Debug)]
struct Turn {
    /// Each call's `path` argument, in order.
    paths: Vec<String>,
    content: String,
    reasoning: String,
    /// `finish_reason` on chat, `status` on Responses.
    finish: String,
    /// `completion_tokens` / `output_tokens`.
    tokens: u64,
    /// Every byte the client received.
    raw: String,
    /// The WebSocket mode's serve, going on until the turn is dropped.
    _serving: Option<Serving>,
}

/// A server serving the WebSocket mode. A serve's end shuts its model down,
/// which would end what a turn's end left the engine to release and any
/// request sent after the turn, so a turn keeps it.
struct Serving(responses::Live);

impl std::fmt::Debug for Serving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "serving on {}", self.0.addr)
    }
}

/// A server whose request 0 writes `steps`, one a step, on `family`; with
/// `eos` it stops on its own there, without it every later token reads `?`
/// up to the cap. With `runs` each decode round commits 8 tokens, which
/// reach the server together.
fn writing(steps: &[&'static str], eos: Option<u32>, family: ModelFamily, runs: bool) -> (ignis_server::Server, Arc<MockCompute>) {
    let mock = Arc::new(if runs { MockCompute::with_runs(&[8]) } else { MockCompute::new() });
    if let Some(step) = eos {
        mock.eos_after(0, step);
    }
    let decode: HashMap<_, _> = steps.iter().enumerate().map(|(step, text)| (mock.token_for(0, step as u32), *text)).collect();
    let server = server(&Script::new(decode), SchedulerConfig::default(), mock.clone()).with_family(family);
    (server, mock)
}

fn read_file() -> JsonValue {
    json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] })
}

/// The request body on `surface`: `parallel` as `parallel_tool_calls`
/// (absent when `None`), and `extra` merged in.
fn body(surface: Surface, parallel: Option<bool>, extra: &JsonValue) -> JsonValue {
    let mut body = match surface {
        Surface::Chat | Surface::ChatStream => json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "read a.txt" }],
            "tools": [{ "type": "function", "function": { "name": "read_file", "parameters": read_file() } }],
            "max_tokens": CAP,
            "enable_thinking": false,
        }),
        Surface::Responses | Surface::ResponsesStream | Surface::Socket => json!({
            "model": MODEL,
            "input": "read a.txt",
            "tools": [{ "type": "function", "name": "read_file", "parameters": read_file() }],
            "max_output_tokens": CAP,
            "enable_thinking": false,
        }),
    };
    match surface {
        Surface::ChatStream => {
            body["stream"] = json!(true);
            body["stream_options"] = json!({ "include_usage": true });
        }
        Surface::ResponsesStream => body["stream"] = json!(true),
        _ => {}
    }
    if let Some(parallel) = parallel {
        body["parallel_tool_calls"] = json!(parallel);
    }
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

async fn post(app: &axum::Router, path: &str, body: &JsonValue) -> String {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert_eq!(status, 200, "{body}: {text}");
    text
}

/// A call's `path` argument.
fn path_of(arguments: &JsonValue) -> String {
    let arguments: JsonValue = serde_json::from_str(arguments.as_str().unwrap()).unwrap();
    arguments["path"].as_str().unwrap().to_owned()
}

/// A Responses response object as a [`Turn`].
fn response_turn(response: &JsonValue, raw: String) -> Turn {
    let mut turn = Turn {
        paths: Vec::new(),
        content: String::new(),
        reasoning: String::new(),
        finish: response["status"].as_str().unwrap().to_owned(),
        tokens: response["usage"]["output_tokens"].as_u64().unwrap(),
        raw,
        _serving: None,
    };
    for item in response["output"].as_array().unwrap() {
        match item["type"].as_str().unwrap() {
            "function_call" => turn.paths.push(path_of(&item["arguments"])),
            "message" => turn.content.push_str(item["content"][0]["text"].as_str().unwrap()),
            "reasoning" => turn.reasoning.push_str(item["content"][0]["text"].as_str().unwrap()),
            other => panic!("an output item of type {other}: {response}"),
        }
    }
    turn
}

/// One turn on `surface`, over `server`, which outlives it as a serving
/// one does: an engine with no handle left ends its model thread before the
/// step that would release what the turn's end stopped. On the WebSocket
/// mode the turn keeps its serve going ([`Serving`]).
async fn turn(server: &ignis_server::Server, surface: Surface, body: &JsonValue) -> Turn {
    match surface {
        Surface::Chat => {
            let raw = post(&server.app(), "/v1/chat/completions", body).await;
            let answer: JsonValue = serde_json::from_str(&raw).unwrap();
            let message = &answer["choices"][0]["message"];
            let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
            Turn {
                paths: calls.iter().map(|c| path_of(&c["function"]["arguments"])).collect(),
                content: message["content"].as_str().unwrap().to_owned(),
                reasoning: message["reasoning_content"].as_str().unwrap_or_default().to_owned(),
                finish: answer["choices"][0]["finish_reason"].as_str().unwrap().to_owned(),
                tokens: answer["usage"]["completion_tokens"].as_u64().unwrap(),
                raw,
                _serving: None,
            }
        }
        Surface::ChatStream => {
            let raw = post(&server.app(), "/v1/chat/completions", body).await;
            let data: Vec<&str> = raw.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
            assert_eq!(data.last(), Some(&"[DONE]"), "{raw}");
            let chunks: Vec<JsonValue> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
            let delta = |c: &JsonValue, key: &str| c["choices"][0]["delta"][key].as_str().unwrap_or_default().to_owned();
            let finishes: Vec<&JsonValue> = chunks.iter().filter(|c| !c["choices"][0]["finish_reason"].is_null()).collect();
            assert_eq!(finishes.len(), 1, "one finish chunk: {raw}");
            Turn {
                paths: chunks
                    .iter()
                    .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
                    .flatten()
                    .map(|call| path_of(&call["function"]["arguments"]))
                    .collect(),
                content: chunks.iter().map(|c| delta(c, "content")).collect(),
                reasoning: chunks.iter().map(|c| delta(c, "reasoning_content")).collect(),
                finish: finishes[0]["choices"][0]["finish_reason"].as_str().unwrap().to_owned(),
                tokens: chunks.last().unwrap()["usage"]["completion_tokens"].as_u64().expect("the usage chunk"),
                raw,
                _serving: None,
            }
        }
        Surface::Responses => {
            let raw = post(&server.app(), "/v1/responses", body).await;
            let response: JsonValue = serde_json::from_str(&raw).unwrap();
            response_turn(&response, raw)
        }
        Surface::ResponsesStream => {
            let raw = post(&server.app(), "/v1/responses", body).await;
            let events = sse_events(&raw);
            let ends: Vec<&JsonValue> = events.iter().map(|(_, e)| e).filter(|e| terminal(e)).collect();
            assert_eq!(ends.len(), 1, "one terminal event: {raw}");
            assert!(terminal(&events.last().unwrap().1), "nothing after the terminal event: {raw}");
            response_turn(&ends[0]["response"], raw)
        }
        Surface::Socket => {
            let live = live(server.clone()).await;
            let mut socket = socket(&live).await;
            let mut create = body.clone();
            create["type"] = json!("response.create");
            send(&mut socket, create).await;
            let events = events_until(&mut socket, terminal).await;
            let raw = events.iter().map(JsonValue::to_string).collect::<Vec<_>>().join("\n");
            Turn { _serving: Some(Serving(live)), ..response_turn(&events.last().unwrap()["response"], raw) }
        }
    }
}

/// Whether `surface` reports a turn that called as `tool_calls` (chat) or
/// `completed` (Responses).
fn called(surface: Surface) -> &'static str {
    match surface {
        Surface::Chat | Surface::ChatStream => "tool_calls",
        _ => "completed",
    }
}

/// The engine let request 0 go long before its cap: the turn's end released
/// it, and it did not generate on to `CAP` for nobody. `case` names the turn.
async fn released_early(mock: &MockCompute, case: &str) {
    let released = responses::released(mock, 0).await;
    let rounds = mock.decode_calls().len();
    assert!(released, "request 0 was never released, {rounds} decode rounds in: {case}");
    assert!(rounds < (CAP / 2) as usize, "{rounds} decode rounds: the request ran on past its turn's end: {case}");
}

#[tokio::test]
async fn false_ends_the_turn_at_its_first_call_on_every_surface_and_both_families() {
    for family in [ModelFamily::Qwen38_27b, ModelFamily::FlashNext] {
        for (steps, runs) in [(SPLIT, false), (SPLIT, true), (ONE_TOKEN, false)] {
            for surface in SURFACES {
                let (server, mock) = writing(steps, None, family, runs);
                let got = turn(&server, surface, &body(surface, Some(false), &json!({}))).await;
                let case = format!("{family:?} {surface:?} runs {runs} {steps:?}: {got:?}");
                assert_eq!(got.paths, ["a.txt"], "{case}");
                assert_eq!(got.content, "Reading. ", "{case}");
                assert_eq!(got.finish, called(surface), "{case}");
                assert_eq!(got.tokens, 2, "the tokens up to the call's close: {case}");
                assert!(!got.raw.contains(SECOND_MARK), "the second call reached the client: {case}");
                assert!(!got.raw.contains('?'), "a token past the close reached the client: {case}");
                released_early(&mock, &case).await;
            }
        }
    }
}

/// The model's own end right behind the close, as a real turn writes it:
/// the stop and the engine's finish race, and the request is counted once,
/// completed and never cancelled, on every surface, the finish in the same
/// decode run as the close or a round later.
#[tokio::test]
async fn a_turn_whose_model_ends_right_after_its_call_is_counted_once_as_completed() {
    for runs in [false, true] {
        for surface in SURFACES {
            let (server, _) = writing(SPLIT, Some(2), ModelFamily::Qwen38_27b, runs);
            let server = server.with_metrics();
            let metrics = server.metrics.clone().unwrap();
            let again = server.clone().app();
            let got = turn(&server, surface, &body(surface, Some(false), &json!({}))).await;
            let case = format!("{surface:?} runs {runs}: {got:?}");
            assert_eq!(got.paths, ["a.txt"], "{case}");
            assert_eq!(got.finish, called(surface), "{case}");
            assert_eq!(got.tokens, 2, "{case}");
            // A request behind it: counted, it follows every fact of the
            // first, which a second count or a cancel would have moved.
            let plain = json!({ "model": MODEL, "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 1, "enable_thinking": false });
            post(&again, "/v1/chat/completions", &plain).await;
            metric_reaches(&metrics, "ignis_requests_completed_total 2").await;
            metric_reaches(&metrics, "ignis_generated_tokens_total 3").await;
            metric_reaches(&metrics, "ignis_requests_cancelled_total 0").await;
        }
    }
}

#[tokio::test]
async fn true_or_absent_serves_every_call_the_turn_writes() {
    for family in [ModelFamily::Qwen38_27b, ModelFamily::FlashNext] {
        for (steps, runs) in [(SPLIT, false), (SPLIT, true), (ONE_TOKEN, false)] {
            for parallel in [None, Some(true)] {
                for surface in SURFACES {
                    let (server, _) = writing(steps, Some(steps.len() as u32), family, runs);
                    let got = turn(&server, surface, &body(surface, parallel, &json!({}))).await;
                    let case = format!("{family:?} {surface:?} {parallel:?} runs {runs} {steps:?}: {got:?}");
                    assert_eq!(got.paths, ["a.txt", "never.txt"], "{case}");
                    assert_eq!(got.finish, called(surface), "{case}");
                    assert_eq!(got.tokens, steps.len() as u64, "{case}");
                }
            }
        }
    }
}

#[tokio::test]
async fn a_call_closed_inside_the_reasoning_ends_nothing() {
    let thought = "plan <tool_call>\n<function=read_file>\n<parameter=path>\nthought.txt\n</parameter>\n</function>\n</tool_call> then";
    let steps: &[&'static str] = &[thought, "</think>", "Reading. ", FIRST, "\n", SECOND];
    for surface in SURFACES {
        let (server, mock) = writing(steps, None, ModelFamily::Qwen38_27b, false);
        let thinking = json!({ "enable_thinking": true, "thinking_budget": 0 });
        let got = turn(&server, surface, &body(surface, Some(false), &thinking)).await;
        let case = format!("{surface:?}: {got:?}");
        assert_eq!(got.reasoning, thought, "the reasoning is whole: {case}");
        assert_eq!(got.paths, ["a.txt"], "{case}");
        assert_eq!(got.content, "Reading. ", "{case}");
        assert_eq!(got.finish, called(surface), "{case}");
        assert_eq!(got.tokens, 4, "{case}");
        assert!(!got.raw.contains(SECOND_MARK), "{case}");
        released_early(&mock, &case).await;
    }
}

/// The turn's end releases its request at the engine's next step however
/// late that step comes. Here the turn's first token closes its call and the
/// model thread is held in the decode round its prefill goes on to, so the
/// stop waits in the engine's queue until the turn has ended, as behind a
/// slow round on a loaded machine; then the round goes on.
#[tokio::test]
async fn a_stop_that_waits_behind_a_round_releases_the_request_at_the_next_step() {
    for surface in SURFACES {
        let mock = Arc::new(MockCompute::new());
        let (gated, gate) = GatedCompute::new(mock.clone() as Arc<dyn Compute>);
        gated.arm_at(0, 0);
        let steps = [FIRST, "\n", SECOND];
        let decode: HashMap<_, _> = steps.iter().enumerate().map(|(step, text)| (mock.token_for(0, step as u32), *text)).collect();
        let server = server(&Script::new(decode), SchedulerConfig::default(), gated);
        let got = turn(&server, surface, &body(surface, Some(false), &json!({}))).await;
        let case = format!("{surface:?}: {got:?}");
        assert_eq!(got.paths, ["a.txt"], "{case}");
        assert_eq!(got.tokens, 1, "the call closes at the first token: {case}");
        gate.wait_entered();
        gate.release();
        released_early(&mock, &case).await;
    }
}

#[tokio::test]
async fn a_turn_with_no_call_is_served_to_its_end() {
    // Nothing closes, so nothing ends the turn early: `false` changes nothing.
    for surface in SURFACES {
        let steps: &[&'static str] = &["one ", "two"];
        let mut answers = Vec::new();
        for parallel in [Some(false), None] {
            let (server, _) = writing(steps, Some(2), ModelFamily::Qwen38_27b, false);
            answers.push(turn(&server, surface, &body(surface, parallel, &json!({}))).await);
        }
        let [with, without]: [Turn; 2] = answers.try_into().unwrap();
        assert_eq!((with.content.as_str(), with.tokens), ("one two", 2), "{surface:?}: {with:?}");
        assert_eq!((with.content, with.finish, with.tokens), (without.content, without.finish, without.tokens), "{surface:?}");
    }
}

#[tokio::test]
async fn a_turn_ended_at_its_call_leaves_the_conversation_resumable() {
    // The next turn, carrying the call and its output, resumes from what this
    // one left as a turn that ended on its own does.
    let words = (0..20).map(|n| format!("w{n}")).collect::<Vec<_>>().join(" ");
    let mut cached = Vec::new();
    for parallel in [Some(false), None] {
        let mock = Arc::new(MockCompute::new());
        let decode = HashMap::from([(mock.token_for(0, 0), FIRST), (mock.token_for(0, 1), "\n"), (mock.token_for(0, 2), SECOND)]);
        mock.eos_after(0, 3);
        let app = server(&Script::with_boundaries(decode), SchedulerConfig::default(), mock.clone()).app();
        let tools = json!([{ "type": "function", "function": { "name": "read_file", "parameters": read_file() } }]);
        let mut first = json!({ "model": MODEL, "messages": [{ "role": "user", "content": words }], "tools": tools, "max_tokens": CAP, "enable_thinking": false });
        if let Some(parallel) = parallel {
            first["parallel_tool_calls"] = json!(parallel);
        }
        let answer: JsonValue = serde_json::from_str(&post(&app, "/v1/chat/completions", &first).await).unwrap();
        let call = answer["choices"][0]["message"]["tool_calls"][0].clone();
        let second = json!({
            "model": MODEL, "tools": tools, "max_tokens": 1, "enable_thinking": false,
            "messages": [
                { "role": "user", "content": words },
                { "role": "assistant", "content": "", "tool_calls": [call] },
                { "role": "tool", "tool_call_id": "call_0", "content": "hello" }
            ]
        });
        let answer: JsonValue = serde_json::from_str(&post(&app, "/v1/chat/completions", &second).await).unwrap();
        cached.push(answer["usage"]["prompt_tokens_details"]["cached_tokens"].clone());
    }
    assert_eq!(cached[0], cached[1], "{cached:?}");
    assert!(cached[0].as_u64().is_some_and(|n| n > 0), "the second turn resumes: {cached:?}");
}
