//! The Responses API over HTTP (GitHub #282, spec responses-api/01, seam 1):
//! `POST /v1/responses` through the in-process router over `MockCompute`,
//! asserted as a client sees it — the event sequence, the output items, the
//! prompt a conversation renders, the 400s, the terminal states and usage.

#[path = "support/responses.rs"]
mod responses;
#[path = "support/failing_prefill.rs"]
mod failing_prefill;
use failing_prefill::FailingPrefill;

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{MockCompute, SchedulerConfig};
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use ignis_server::deadline::TimeoutReply;
use ignis_server::images::job::base64;
use ignis_server::loop_guard::{repair, LoopGuardMode, LoopGuardParams, TAG_PREFIX};
use responses::{server, sse_events, Script, MODEL};

/// POST `body` to `path`; the status, the content type and the body text.
async fn post(app: &axum::Router, path: &str, body: JsonValue) -> (u16, String, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, content_type, String::from_utf8(bytes.to_vec()).unwrap())
}

/// Request 0's first three tokens: a thought, the close and a sentence, and
/// a whole tool call.
fn agent_turn() -> HashMap<u32, &'static str> {
    let mock = MockCompute::new();
    HashMap::from([
        (mock.token_for(0, 0), "planning"),
        (mock.token_for(0, 1), "</think>Reading it. "),
        (
            mock.token_for(0, 2),
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
        ),
    ])
}

/// A server whose request 0 plays [`agent_turn`] and then stops on its own.
fn agent_app() -> axum::Router {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(0, 3);
    server(&Script::new(agent_turn()), SchedulerConfig::default(), mock).app()
}

fn agent_request(stream: bool) -> JsonValue {
    json!({
        "model": MODEL,
        "input": "read a.txt",
        "tools": [{ "type": "function", "name": "read_file", "parameters": {
            "type": "object", "properties": { "path": { "type": "string" } }
        } }],
        "max_output_tokens": 16,
        "stream": stream,
    })
}

/// Acceptance 1 and 2: the standard sequence, in order, with
/// `sequence_number` counting from 0, and each item's content events inside
/// its added/done pair.
#[tokio::test]
async fn a_streamed_response_is_the_standard_event_sequence() {
    let (status, content_type, body) = post(&agent_app(), "/v1/responses", agent_request(true)).await;
    assert_eq!(status, 200, "{body}");
    assert!(content_type.starts_with("text/event-stream"), "{content_type}");
    let events = sse_events(&body);
    for (at, (name, event)) in events.iter().enumerate() {
        assert_eq!(&event["type"], name, "the event line names the event");
        assert_eq!(event["sequence_number"], at, "{event}");
    }
    let kinds: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let event = |at: usize| &events[at].1;
    assert_eq!(event(0)["response"]["status"], "in_progress");
    assert!(event(0)["response"]["usage"].is_null());
    let first = "0".repeat(32);
    assert_eq!(event(0)["response"]["id"], format!("resp_{first}"));
    assert_eq!(event(2)["item"]["type"], "reasoning");
    assert_eq!(event(4)["delta"], "planning");
    assert_eq!(event(4)["item_id"], format!("rs_{first}"));
    assert_eq!(event(8)["item"]["type"], "message");
    assert_eq!(event(10)["delta"], "Reading it. ");
    assert_eq!(event(10)["output_index"], 1);
    // The message is done before the call is added: items follow one another.
    assert_eq!(event(13)["item"]["type"], "message");
    assert_eq!(event(13)["item"]["status"], "completed");
    assert_eq!(event(14)["item"]["type"], "function_call");
    assert_eq!(event(14)["output_index"], 2);
    assert_eq!(event(16)["name"], "read_file");
    assert_eq!(event(16)["arguments"], r#"{"path":"a.txt"}"#);
    assert_eq!(event(17)["item"]["call_id"], format!("call_{first}_0"));
    assert_eq!(event(17)["item"]["id"], format!("fc_{first}_0"));
    let done = &event(18)["response"];
    assert_eq!(done["status"], "completed");
    let types: Vec<&str> = done["output"].as_array().unwrap().iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["reasoning", "message", "function_call"]);
    assert_eq!(done["output"][0]["content"][0], json!({ "type": "reasoning_text", "text": "planning" }));
    assert_eq!(done["output"][1]["content"][0]["text"], "Reading it. ");
    // Two tokens were generated inside the thinking channel: the thought and
    // the one carrying `</think>`.
    assert_eq!(done["usage"]["output_tokens"], 3);
    assert_eq!(done["usage"]["output_tokens_details"]["reasoning_tokens"], 2);
    assert_eq!(done["usage"]["input_tokens_details"]["cached_tokens"], 0);
    assert_eq!(done["usage"]["total_tokens"], done["usage"]["input_tokens"].as_u64().unwrap() + 3);
}

/// Acceptance 1 and 2: without `stream` the body is the terminal event's
/// `response`, byte for byte in payload, items included.
#[tokio::test]
async fn the_non_streaming_body_is_the_terminal_events_response() {
    let (status, _, streamed) = post(&agent_app(), "/v1/responses", agent_request(true)).await;
    assert_eq!(status, 200, "{streamed}");
    let terminal = sse_events(&streamed).pop().unwrap().1["response"].clone();
    let (status, content_type, body) = post(&agent_app(), "/v1/responses", agent_request(false)).await;
    assert_eq!(status, 200, "{body}");
    assert!(content_type.starts_with("application/json"), "{content_type}");
    let body: JsonValue = serde_json::from_str(&body).unwrap();
    assert_eq!(body, terminal);
    assert_eq!(body["object"], "response");
    assert_eq!(body["tools"][0]["name"], "read_file", "tools echo in the shape they were sent");
}

/// Text after a call is a new message item, added after the call is done;
/// continuing from that output renders the one assistant turn chat
/// completions would.
#[tokio::test]
async fn text_after_a_call_opens_a_new_message_item() {
    let mock = MockCompute::new();
    let script = Script::new(HashMap::from([
        (mock.token_for(0, 0), "before "),
        (mock.token_for(0, 1), "<tool_call>\n<function=a>\n</function>\n</tool_call>"),
        (mock.token_for(0, 2), " after"),
    ]));
    let compute = Arc::new(MockCompute::new());
    compute.eos_after(0, 3);
    let app = server(&script, SchedulerConfig::default(), compute).app();
    let body = json!({ "model": MODEL, "input": "go", "max_output_tokens": 8, "enable_thinking": false, "stream": true });
    let (status, _, streamed) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{streamed}");
    let events = sse_events(&streamed);
    let items: Vec<(String, String)> = events
        .iter()
        .filter(|(name, _)| name.starts_with("response.output_item."))
        .map(|(name, e)| (name.trim_start_matches("response.output_item.").to_owned(), e["item"]["id"].as_str().unwrap().to_owned()))
        .collect();
    let first = "0".repeat(32);
    let (message, call, after) = (format!("msg_{first}"), format!("fc_{first}_0"), format!("msg_{first}_1"));
    let pairs = |v: &[(&str, &str)]| v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>();
    assert_eq!(
        items,
        pairs(&[("added", &message), ("done", &message), ("added", &call), ("done", &call), ("added", &after), ("done", &after)])
    );
    let output = events.last().unwrap().1["response"]["output"].clone();

    let mut conversation = vec![json!({ "role": "user", "content": "go" })];
    conversation.extend(output.as_array().unwrap().iter().cloned());
    let (chat, responses, _) = render_both(
        json!({ "model": MODEL, "max_tokens": 1, "messages": [
            { "role": "user", "content": "go" },
            { "role": "assistant", "content": "before  after",
              "tool_calls": [{ "id": format!("call_{first}_0"), "type": "function", "function": { "name": "a", "arguments": "{}" } }] }
        ] }),
        json!({ "model": MODEL, "max_output_tokens": 1, "input": conversation }),
    )
    .await;
    assert_eq!(responses, chat);
}

// ── acceptance 3: the same prompt as chat completions ────────────────────

/// Render `chat` through chat completions and `responses` through the
/// Responses API on one server; what the template received for each.
async fn render_both(chat: JsonValue, responses: JsonValue) -> (responses::Rendered, responses::Rendered, JsonValue) {
    let script = Script::new(HashMap::new());
    let app = server(&script, SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
    let (status, _, body) = post(&app, "/v1/chat/completions", chat).await;
    assert_eq!(status, 200, "{body}");
    let (status, _, body) = post(&app, "/v1/responses", responses).await;
    assert_eq!(status, 200, "{body}");
    let seen = script.seen();
    assert_eq!(seen.len(), 2);
    (seen[0].clone(), seen[1].clone(), serde_json::from_str(&body).unwrap())
}

#[tokio::test]
async fn a_responses_conversation_renders_what_chat_completions_renders() {
    let (chat, responses, body) = render_both(
        json!({
            "model": MODEL,
            "max_tokens": 1,
            "tools": [{ "type": "function", "function": {
                "name": "read_file", "description": "Read a file.",
                "parameters": { "type": "object", "properties": { "path": { "type": "string" } } }
            } }],
            "messages": [
                { "role": "system", "content": "be brief" },
                { "role": "developer", "content": "today is Tuesday" },
                { "role": "user", "content": [{ "type": "text", "text": "read" }, { "type": "text", "text": "a.txt" }] },
                { "role": "assistant", "content": "Reading it.", "reasoning_content": "planning",
                  "tool_calls": [{ "id": "call_7", "type": "function",
                                   "function": { "name": "read_file", "arguments": "{\"path\":\"a.txt\"}" } }] },
                { "role": "tool", "tool_call_id": "call_7", "content": "hello" },
                { "role": "assistant", "content": "It says hello." },
                { "role": "assistant", "content": "Anything else?" },
                { "role": "user", "content": "thanks" }
            ]
        }),
        json!({
            "model": MODEL,
            "max_output_tokens": 1,
            "instructions": "be brief",
            "tools": [{ "type": "function", "name": "read_file", "description": "Read a file.",
                        "parameters": { "type": "object", "properties": { "path": { "type": "string" } } } }],
            "input": [
                { "type": "message", "role": "developer", "content": "today is Tuesday" },
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "read" }, { "type": "input_text", "text": "a.txt" }
                ] },
                { "type": "reasoning", "summary": [], "content": [{ "type": "reasoning_text", "text": "planning" }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Reading it." }] },
                { "type": "function_call", "call_id": "call_7", "name": "read_file", "arguments": "{\"path\":\"a.txt\"}" },
                { "type": "function_call_output", "call_id": "call_7", "output": "hello" },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "It says hello." }] },
                { "role": "assistant", "content": "Anything else?" },
                { "role": "user", "content": "thanks" }
            ]
        }),
    )
    .await;
    assert_eq!(responses, chat);
    let assistants: Vec<String> =
        chat.messages.iter().filter(|m| m.role == "assistant").map(|m| m.content.text()).collect();
    assert_eq!(assistants, ["Reading it.", "It says hello.", "Anything else?"], "consecutive assistant messages stay two turns");
    assert_eq!(body["status"], "incomplete", "{body}");
}

/// History repair (ADR 0052): a rewind the loop guard tagged in an
/// assistant's reasoning reaches the template taken out, on both surfaces
/// and with the guard off, as this harness runs. Only the reasoning: a user
/// message and assistant content carrying the same tag reach it as sent.
#[tokio::test]
async fn a_rewound_reasoning_reaches_the_template_repaired_on_both_surfaces() {
    let tag = ignis_server::loop_guard::tag(4, "the loop, ");
    let reasoning = format!("planning, the loop, {tag}\n\nWait, I'm starting to repeat myself.");
    let echoed = format!("the loop, {tag}");
    let (chat, responses, _) = render_both(
        json!({
            "model": MODEL,
            "max_tokens": 1,
            "messages": [
                { "role": "user", "content": echoed },
                { "role": "assistant", "content": echoed, "reasoning_content": reasoning },
                { "role": "user", "content": "go on" }
            ]
        }),
        json!({
            "model": MODEL,
            "max_output_tokens": 1,
            "input": [
                { "role": "user", "content": echoed },
                { "type": "reasoning", "summary": [], "content": [{ "type": "reasoning_text", "text": reasoning }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": echoed }] },
                { "role": "user", "content": "go on" }
            ]
        }),
    )
    .await;
    assert_eq!(responses, chat);
    assert_eq!(chat.messages[1].reasoning_content.as_deref(), Some("planning, \n\nWait, I'm starting to repeat myself."));
    assert_eq!(chat.messages[0].content.text(), echoed, "a user message is never repaired");
    assert_eq!(chat.messages[1].content.text(), echoed, "nor an assistant's content");
}

/// The items of one turn, listed in the order this server emits them when a
/// call comes before any text, still make the one assistant message.
#[tokio::test]
async fn a_turns_items_make_one_assistant_message_in_any_order() {
    let (chat, responses, _) = render_both(
        json!({
            "model": MODEL,
            "max_tokens": 1,
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "content": "done", "reasoning_content": "think",
                  "tool_calls": [{ "id": "c1", "type": "function", "function": { "name": "a", "arguments": "{}" } },
                                 { "id": "c2", "type": "function", "function": { "name": "b", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "c1", "content": "1" }
            ]
        }),
        json!({
            "model": MODEL,
            "max_output_tokens": 1,
            "input": [
                { "role": "user", "content": "go" },
                { "type": "reasoning", "summary": [], "content": [{ "type": "reasoning_text", "text": "think" }] },
                { "type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}" },
                { "type": "message", "role": "assistant", "status": "completed", "id": "msg_1",
                  "content": [{ "type": "output_text", "text": "done", "annotations": [] }] },
                { "type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": "1" }
            ]
        }),
    )
    .await;
    assert_eq!(responses, chat);
}

// ── acceptance 4: the 400s, and what is accepted and ignored ─────────────

#[tokio::test]
async fn what_this_server_does_not_serve_is_a_400_naming_the_field() {
    let app = server(&Script::new(HashMap::new()), SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
    let base = json!({ "model": MODEL, "input": "hi", "max_output_tokens": 1 });
    let with = |field: &str, value: JsonValue| {
        let mut body = base.clone();
        body[field] = value;
        body
    };
    let cases = [
        (with("tools", json!([{ "type": "web_search" }])), "tools[0].type", "invalid_request_error"),
        (
            with("tools", json!([{ "type": "function", "name": "a" }, { "type": "code_interpreter" }])),
            "tools[1].type",
            "invalid_request_error",
        ),
        (with("text", json!({ "format": { "type": "json_schema", "name": "x" } })), "text.format", "invalid_request_error"),
        (with("text", json!({ "format": { "type": "xml" } })), "text.format", "unsupported_value"),
        (with("background", json!(true)), "background", "invalid_request_error"),
        (with("generate", json!(false)), "generate", "invalid_request_error"),
        (with("previous_response_id", json!("resp_0")), "previous_response_id", "previous_response_not_found"),
        (with("input", json!([{ "type": "item_reference", "id": "msg_1" }])), "input[0].type", "invalid_request_error"),
        (
            with("input", json!([{ "role": "user", "content": "a" }, { "type": "computer_call" }])),
            "input[1].type",
            "invalid_request_error",
        ),
        // Spec server/09: the rest of the Responses body, refused by value.
        (with("top_logprobs", json!(3)), "top_logprobs", "unsupported_value"),
        (with("truncation", json!("auto")), "truncation", "unsupported_value"),
        (with("conversation", json!("conv_1")), "conversation", "unsupported_value"),
        (with("prompt", json!({ "id": "pmpt_1" })), "prompt", "unsupported_value"),
        (with("context_management", json!([{ "type": "compaction" }])), "context_management", "unsupported_value"),
    ];
    for (body, param, code) in cases {
        let (status, _, answer) = post(&app, "/v1/responses", body.clone()).await;
        assert_eq!(status, 400, "{body}: {answer}");
        let answer: JsonValue = serde_json::from_str(&answer).unwrap();
        assert_eq!(answer["error"]["type"], "invalid_request_error", "{answer}");
        assert_eq!(answer["error"]["param"], param, "{answer}");
        assert_eq!(answer["error"]["code"], code, "{answer}");
    }
}

#[tokio::test]
async fn fields_that_do_not_change_the_answer_are_accepted() {
    let app = server(&Script::new(HashMap::new()), SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
    let body = json!({
        "model": MODEL,
        "input": "hi",
        "max_output_tokens": 1,
        "store": false,
        "metadata": { "run": "7" },
        "user": "u",
        "safety_identifier": "s",
        "prompt_cache_key": "k",
        "prompt_cache_retention": "24h",
        "service_tier": "auto",
        "truncation": "disabled",
        "stream_options": { "include_obfuscation": false },
        "client_metadata": { "a": 1 },
        "text": { "format": { "type": "text" }, "verbosity": "low" },
        "reasoning": { "effort": "low", "summary": "auto" },
        "parallel_tool_calls": false,
        "tool_choice": "auto",
        "background": false,
        "previous_response_id": null,
        "top_logprobs": 0,
        "max_tool_calls": 4,
        "conversation": null,
        "some_future_field": { "x": 1 }
    });
    let (status, _, answer) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{answer}");
    let answer: JsonValue = serde_json::from_str(&answer).unwrap();
    assert_eq!(answer["store"], false);
    assert_eq!(answer["metadata"], json!({ "run": "7" }));
    assert_eq!(answer["reasoning"]["effort"], "low");
}

// ── acceptance 5: terminal states and usage ───────────────────────────────

#[tokio::test]
async fn a_length_stop_ends_incomplete_with_its_reason() {
    let app = server(&Script::new(HashMap::new()), SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
    let body = json!({ "model": MODEL, "input": "hi", "max_output_tokens": 2, "enable_thinking": false, "stream": true });
    let (status, _, streamed) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{streamed}");
    let (name, last) = sse_events(&streamed).pop().unwrap();
    assert_eq!(name, "response.incomplete");
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(last["response"]["incomplete_details"], json!({ "reason": "max_output_tokens" }));
    assert_eq!(last["response"]["output"][0]["status"], "incomplete", "a cut message is incomplete");
    assert_eq!(last["response"]["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn an_engine_error_ends_the_response_failed() {
    let app = server(&Script::new(HashMap::new()), SchedulerConfig::default(), Arc::new(FailingPrefill(MockCompute::new()))).app();
    let body = json!({ "model": MODEL, "input": "hi", "max_output_tokens": 2, "stream": true });
    let (status, _, streamed) = post(&app, "/v1/responses", body.clone()).await;
    assert_eq!(status, 200, "{streamed}");
    let (name, last) = sse_events(&streamed).pop().unwrap();
    assert_eq!(name, "response.failed");
    assert_eq!(last["response"]["error"]["code"], "server_error");

    let mut body = body;
    body["stream"] = json!(false);
    let (status, _, answer) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, 200, "the body is the failed response: {answer}");
    let answer: JsonValue = serde_json::from_str(&answer).unwrap();
    assert_eq!(answer["status"], "failed");
    assert_eq!(answer["error"]["code"], "server_error");
}

/// `cached_tokens` is the prompt a request resumed from retained state: the
/// second turn of a conversation resumes from the checkpoint the first left
/// at its generation opener (past one whole 16-token page, which is where a
/// checkpoint can be taken).
#[tokio::test]
async fn usage_reports_the_prompt_resumed_from_retained_state() {
    let script = Script::with_boundaries(HashMap::new());
    let app = server(&script, SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
    let words = (0..20).map(|n| format!("w{n}")).collect::<Vec<_>>().join(" ");
    let first = json!({ "model": MODEL, "input": words, "max_output_tokens": 1, "enable_thinking": false });
    let (status, _, answer) = post(&app, "/v1/responses", first).await;
    assert_eq!(status, 200, "{answer}");
    let answer: JsonValue = serde_json::from_str(&answer).unwrap();
    assert_eq!(answer["usage"]["input_tokens_details"]["cached_tokens"], 0);

    let second = json!({
        "model": MODEL,
        "max_output_tokens": 1,
        "enable_thinking": false,
        "input": [
            { "role": "user", "content": words },
            { "role": "assistant", "content": "ok" },
            { "role": "user", "content": "six" }
        ]
    });
    let (status, _, answer) = post(&app, "/v1/responses", second).await;
    assert_eq!(status, 200, "{answer}");
    let answer: JsonValue = serde_json::from_str(&answer).unwrap();
    assert_eq!(answer["usage"]["input_tokens_details"]["cached_tokens"], 20, "{answer}");
}

// ── the loop guard (ADR 0052) ──────────────────────────────────────────────

/// The reasoning request 0 writes: prose, then a loop the guard rewinds.
fn looping_reasoning() -> String {
    responses::prose(3000) + &responses::looping(4000)
}

/// A server with the loop guard on, `holdback` characters held back, whose
/// request 0 reasons [`looping_reasoning`] and whose continuation (request
/// 1) reasons silently for 20 tokens and stops.
fn looping_app(holdback: u32) -> (axum::Router, Arc<Script>) {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(1, 20);
    let script = Script::encoding(responses::scripted_text(&mock, 0, &looping_reasoning()));
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, holdback_chars: holdback, ..LoopGuardParams::default() };
    (server(&script, SchedulerConfig::default(), mock).with_loop_guard(guard).app(), script)
}

fn looping_request(stream: bool, include: bool) -> JsonValue {
    let mut body = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": 4000, "stream": stream });
    if include {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }
    body
}

/// The response to `body`: its terminal object, and, streamed, the
/// reasoning its deltas carried.
async fn respond(app: &axum::Router, body: JsonValue) -> (JsonValue, Option<String>) {
    let stream = body["stream"] == true;
    let (status, _, text) = post(app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{text}");
    if !stream {
        return (serde_json::from_str(&text).unwrap(), None);
    }
    let events = sse_events(&text);
    let deltas =
        events.iter().filter(|(name, _)| name == "response.reasoning_text.delta").map(|(_, e)| e["delta"].as_str().unwrap()).collect();
    (events.last().unwrap().1["response"].clone(), Some(deltas))
}

/// A rewound reasoning around its tag: the text the client had before it,
/// the tag's `N TOKENS`, and what follows the tag.
fn split_tag(reasoning: &str) -> (&str, u32, &str) {
    let (before, rest) = reasoning.split_once(TAG_PREFIX).expect("a rewind tag");
    let (tokens, _) = rest.split_once(" TOKENS").unwrap();
    let (_, after) = rest.split_once(']').unwrap();
    (before.strip_suffix("\n\n").expect("the tag's blank line"), tokens.parse().unwrap(), after)
}

/// The loop guard on `POST /v1/responses`, streamed and not: the reasoning
/// item carries the tag and the nudge, `rewound_tokens` the tag's count,
/// and `output_tokens` every token the model generated.
#[tokio::test]
async fn a_looping_reasoning_is_rewound_over_http_and_the_usage_counts_it() {
    for stream in [false, true] {
        let (app, _) = looping_app(0);
        let (response, deltas) = respond(&app, looping_request(stream, false)).await;
        assert_eq!(response["status"], "completed", "{response}");
        let reasoning = response["output"][0]["content"][0]["text"].as_str().unwrap();
        if let Some(deltas) = deltas {
            assert_eq!(deltas, reasoning, "the deltas are the item");
        }
        let (sent, rewound, after) = split_tag(reasoning);
        assert!(looping_reasoning().starts_with(sent));
        assert!(after.trim_start_matches('\n').starts_with("Wait, I'm starting to repeat myself."), "{after:?}");
        let usage = &response["usage"];
        assert_eq!(usage["output_tokens_details"]["rewound_tokens"], rewound, "{usage}");
        // A token per 8 characters before the tag, one per character of the
        // tag and the nudge (this tokenizer's), then the continuation's 20.
        assert_eq!(sent.chars().count() % 8, 0);
        let generated = sent.chars().count() / 8 + (reasoning.chars().count() - sent.chars().count()) + 20;
        assert_eq!(usage["output_tokens"], generated, "{usage}");
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], generated);
    }
}

/// A holdback past the cut hides the rewind: no tag, the kept reasoning
/// then the nudge, and the tokens never sent still counted.
#[tokio::test]
async fn a_rewind_the_holdback_hid_is_still_counted() {
    let (app, _) = looping_app(6000);
    let (response, _) = respond(&app, looping_request(false, false)).await;
    let reasoning = response["output"][0]["content"][0]["text"].as_str().unwrap();
    assert!(!reasoning.contains(TAG_PREFIX), "{reasoning}");
    let at = reasoning.find("Wait, I'm starting to repeat myself.").expect("the nudge");
    // The prose never has two line breaks in a row: two are the nudge's.
    let kept = reasoning[..at].strip_suffix("\n\n").unwrap_or(&reasoning[..at]);
    assert!(looping_reasoning().starts_with(kept) && kept.chars().count() % 8 == 0);
    let usage = &response["usage"];
    let rewound = usage["output_tokens_details"]["rewound_tokens"].as_u64().expect("rewound tokens");
    let received = kept.chars().count() / 8 + (reasoning.chars().count() - kept.chars().count()) + 20;
    assert_eq!(usage["output_tokens"], received as u64 + rewound, "every rewound token was held back, and generated: {usage}");
}

/// The reasoning item's `encrypted_content` with the `include`: on a
/// response the guard did not rewind, the reasoning as streamed; on one it
/// did, the model's view, which the same item sent back renders in place of
/// whatever text it shows, with nothing left for history repair to do.
#[tokio::test]
async fn include_gives_the_reasoning_item_the_models_view_and_it_comes_back_as_that() {
    let mut plain = agent_request(false);
    plain["include"] = json!(["reasoning.encrypted_content"]);
    let (plain, _) = respond(&agent_app(), plain).await;
    assert_eq!(plain["output"][0]["encrypted_content"], format!("ignis:v1:{}", base64(b"planning")), "{plain}");
    let (unasked, _) = respond(&agent_app(), agent_request(false)).await;
    assert!(unasked["output"][0].get("encrypted_content").is_none(), "{unasked}");
    let mut other = agent_request(false);
    other["include"] = json!(["message.output_text.logprobs"]);
    let (other, _) = respond(&agent_app(), other).await;
    assert_eq!(other["output"], unasked["output"], "any other include changes nothing");

    let (app, script) = looping_app(0);
    let (response, _) = respond(&app, looping_request(false, true)).await;
    let item = response["output"][0].clone();
    let shown = item["content"][0]["text"].as_str().unwrap();
    let model_view = repair(shown).into_owned();
    assert!(shown.contains(TAG_PREFIX) && !model_view.contains(TAG_PREFIX));
    assert_eq!(item["encrypted_content"], format!("ignis:v1:{}", base64(model_view.as_bytes())));

    let mut sent_back = item.clone();
    sent_back["content"][0]["text"] = json!("what the client showed");
    let next = json!({
        "model": MODEL,
        "max_output_tokens": 1,
        "input": [
            { "role": "user", "content": "the whole history of Rome" },
            sent_back,
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Rome." }] },
            { "role": "user", "content": "go on" }
        ]
    });
    let (status, _, body) = post(&app, "/v1/responses", next).await;
    assert_eq!(status, 200, "{body}");
    let rendered = script.seen().last().unwrap().messages[1].reasoning_content.clone().unwrap();
    assert_eq!(rendered, model_view);
    assert!(matches!(repair(&rendered), Cow::Borrowed(_)), "nothing for history repair to do");
}

/// An `encrypted_content` this server did not write (another server's, or
/// a malformed one) renders the item's visible text, which history repair
/// sees; one of ours with no visible text (how Codex sends an item back)
/// renders from the blob alone.
#[tokio::test]
async fn a_reasoning_blob_this_server_cannot_read_falls_back_to_the_visible_text() {
    let tag = ignis_server::loop_guard::tag(4, "the loop, ");
    let visible = format!("planning, the loop, {tag}Wait.");
    let ours = format!("ignis:v1:{}", base64("the model's view".as_bytes()));
    let cases = [
        (json!([{ "type": "reasoning_text", "text": visible }]), "gAAAAABo-another-servers-blob".to_owned(), "planning, Wait."),
        (json!([{ "type": "reasoning_text", "text": visible }]), "ignis:v1:!!!".to_owned(), "planning, Wait."),
        (json!([{ "type": "reasoning_text", "text": visible }]), "ignis:v1://4=".to_owned(), "planning, Wait."),
        (json!([]), ours, "the model's view"),
    ];
    for (content, blob, rendered) in cases {
        let script = Script::new(HashMap::new());
        let app = server(&script, SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
        let body = json!({
            "model": MODEL,
            "max_output_tokens": 1,
            "input": [
                { "role": "user", "content": "q" },
                { "type": "reasoning", "summary": [], "content": content, "encrypted_content": blob },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "a" }] },
                { "role": "user", "content": "go on" }
            ]
        });
        let (status, _, answer) = post(&app, "/v1/responses", body).await;
        assert_eq!(status, 200, "{answer}");
        assert_eq!(script.seen()[0].messages[1].reasoning_content.as_deref(), Some(rendered), "{blob}");
    }
}

/// A stop sequence the continuation writes stops the continuation itself
/// (ADR 0052): the request log reports it done at its stop, and only the
/// looping request the rewind replaced is cancelled. Chat completions,
/// streamed and not. With no retries the rewind closes the reasoning, and
/// the continuation answers at once.
///
/// The looping request must still be running when the rewind cancels it, or
/// it completes and the counts below say nothing: at 4,000 tokens the mock
/// finished it first on a loaded 4-core runner (the v0.7.0 release run). It
/// gets a budget the mock takes seconds to spend, which the guard's cancel
/// never lags behind.
#[tokio::test]
async fn a_stop_sequence_after_a_rewind_stops_the_continuation_by_its_own_id() {
    const MAX_TOKENS: u32 = 1 << 20;
    let room = || SchedulerConfig {
        max_sequence_tokens: MAX_TOKENS + 1024,
        kv_capacity_pages: (ignis_core::N_DECODE_LANES as u32) * ((MAX_TOKENS + 1024) / 16),
        ..SchedulerConfig::default()
    };
    for stream in [false, true] {
        let mock = Arc::new(MockCompute::new());
        // Every token the continuation draws reads as this.
        let script = Script::encoding_or(responses::scripted_text(&mock, 0, &looping_reasoning()), "Fine! STOP ");
        let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, retries: 0, ..LoopGuardParams::default() };
        let server = server(&script, room(), mock).with_loop_guard(guard).with_metrics();
        let metrics = Arc::clone(server.metrics.as_ref().unwrap());
        let app = server.app();
        let body = json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "the whole history of Rome" }],
            "max_tokens": MAX_TOKENS,
            "stop": ["STOP"],
            "stream": stream,
        });
        let (status, _, text) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, 200, "{text}");
        assert_eq!(text.matches("Fine! ").count(), 1, "nothing after the stop sequence: {text}");
        if !stream {
            let answer: JsonValue = serde_json::from_str(&text).unwrap();
            let message = &answer["choices"][0]["message"];
            assert!(message["reasoning_content"].as_str().unwrap().contains(TAG_PREFIX), "rewound: {answer}");
            assert_eq!((&message["content"], &answer["choices"][0]["finish_reason"]), (&json!("Fine! "), &json!("stop")), "{answer}");
        }
        responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
        assert!(metrics.render().contains("\nignis_requests_cancelled_total 1\n"), "{}", metrics.render());
    }
}

/// A streamed response its timeout cuts short still gets what the holdback
/// held back (ADR 0052): its terminal event — `response.incomplete` under
/// `server.timeout_reply: partial`, `response.failed` under `error` (ADR
/// 0058) — carries it in the reasoning item, its deltas come first, and
/// `output_tokens` counts it. Here the holdback is longer than the whole
/// reasoning, so the client had none of it.
async fn check_a_timeout_lets_what_the_holdback_held_reach_the_streamed_response(reply: TimeoutReply) {
    let mock = Arc::new(MockCompute::new());
    let reasoning = responses::prose(2000);
    let script = Script::encoding(responses::scripted_text(&mock, 0, &reasoning));
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, holdback_chars: 100_000, ..LoopGuardParams::default() };
    // Room for a request that runs until its timeout.
    let config = SchedulerConfig { max_sequence_tokens: 2 << 20, kv_capacity_pages: (2 << 20) / 16 * 2, ..SchedulerConfig::default() };
    let app = server(&script, config, mock)
        .with_loop_guard(guard)
        .with_stream_timeout(std::time::Duration::from_millis(500))
        .with_timeout_reply(reply)
        .app();
    let body = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": 1_000_000, "stream": true });
    let (status, _, text) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{text}");
    let events = sse_events(&text);
    let (name, failed) = events.last().unwrap();
    match reply {
        TimeoutReply::Partial => assert_eq!(
            (name.as_str(), &failed["response"]["incomplete_details"]["reason"]),
            ("response.incomplete", &json!("max_output_tokens"))
        ),
        TimeoutReply::Error => {
            assert_eq!((name.as_str(), &failed["response"]["error"]["code"]), ("response.failed", &json!("request_timeout")))
        }
    }
    let shown = failed["response"]["output"][0]["content"][0]["text"].as_str().unwrap_or_default();
    assert!(!shown.is_empty() && reasoning.starts_with(shown), "the held reasoning reaches the item: {shown:?}");
    let deltas: String =
        events.iter().filter(|(name, _)| name == "response.reasoning_text.delta").map(|(_, e)| e["delta"].as_str().unwrap()).collect();
    assert_eq!(deltas, shown, "streamed before the end");
    let output_tokens = failed["response"]["usage"]["output_tokens"].as_u64().unwrap() as usize;
    assert!(output_tokens >= shown.chars().count() / 8, "{}", failed["response"]["usage"]);
}

#[tokio::test]
async fn a_timeout_lets_what_the_holdback_held_reach_the_streamed_response() {
    check_a_timeout_lets_what_the_holdback_held_reach_the_streamed_response(TimeoutReply::Partial).await;
}

#[tokio::test]
async fn a_timeout_answered_with_an_error_lets_what_the_holdback_held_reach_the_streamed_response() {
    check_a_timeout_lets_what_the_holdback_held_reach_the_streamed_response(TimeoutReply::Error).await;
}
