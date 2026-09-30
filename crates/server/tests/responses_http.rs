//! The Responses API over HTTP (GitHub #282, spec responses-api/01, seam 1):
//! `POST /v1/responses` through the in-process router over `MockCompute`,
//! asserted as a client sees it — the event sequence, the output items, the
//! prompt a conversation renders, the 400s, the terminal states and usage.

#[path = "support/responses.rs"]
mod responses;

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{Compute, ComputeError, MockCompute, PrefillJob, PrefillOutcome, SchedulerConfig};
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

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
    assert_eq!(event(0)["response"]["id"], "resp_0");
    assert_eq!(event(2)["item"]["type"], "reasoning");
    assert_eq!(event(4)["delta"], "planning");
    assert_eq!(event(4)["item_id"], "rs_0");
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
    assert_eq!(event(17)["item"]["call_id"], "call_0_0");
    assert_eq!(event(17)["item"]["id"], "fc_0_0");
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
    let pairs = |v: &[(&str, &str)]| v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>();
    assert_eq!(
        items,
        pairs(&[("added", "msg_0"), ("done", "msg_0"), ("added", "fc_0_0"), ("done", "fc_0_0"), ("added", "msg_0_1"), ("done", "msg_0_1")])
    );
    let output = events.last().unwrap().1["response"]["output"].clone();

    let mut conversation = vec![json!({ "role": "user", "content": "go" })];
    conversation.extend(output.as_array().unwrap().iter().cloned());
    let (chat, responses, _) = render_both(
        json!({ "model": MODEL, "max_tokens": 1, "messages": [
            { "role": "user", "content": "go" },
            { "role": "assistant", "content": "before  after",
              "tool_calls": [{ "id": "call_0_0", "type": "function", "function": { "name": "a", "arguments": "{}" } }] }
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
        (with("text", json!({ "format": { "type": "json_schema", "name": "x", "schema": {} } })), "text.format", "invalid_request_error"),
        (with("text", json!({ "format": { "type": "json_object" } })), "text.format", "invalid_request_error"),
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
        (with("service_tier", json!("priority")), "service_tier", "unsupported_value"),
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
        "include": ["reasoning.encrypted_content"],
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

/// A backend whose every prefill fails: the scheduler gives the request up
/// with `FinishReason::Error` after its retries.
struct FailingPrefill(MockCompute);

impl Compute for FailingPrefill {
    fn prefill_step(&self, _jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }
    fn decode_step(
        &self,
        jobs: &[ignis_core::DecodeJob],
    ) -> Result<Vec<ignis_core::DecodeOutcome>, ComputeError> {
        self.0.decode_step(jobs)
    }
    fn release(&self, request: ignis_core::RequestId) {
        self.0.release(request);
    }
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
