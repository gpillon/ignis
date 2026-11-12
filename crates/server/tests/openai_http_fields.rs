//! Spec server/09 (GitHub #284): every field of the chat-completions body is
//! honoured, accepted as inert, or refused naming it — asserted row by row
//! over the one table the validator and the document read — plus the three
//! it honours: `stop`, `max_completion_tokens`, and `usage`'s
//! `prompt_tokens_details.cached_tokens`.
//!
//! Driven through the in-process router over `MockCompute` and a scripted
//! template, so chosen tokens read as chosen text.

#[path = "support/responses.rs"]
mod responses;

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::mock::GatedCompute;
use ignis_core::{MockCompute, SchedulerConfig};
use ignis_server::openai_fields;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use responses::{server, Script, MODEL};

const CHAT: &str = "/v1/chat/completions";

/// POST `body` to `path`: the status and the body text.
async fn post(app: &axum::Router, path: &str, body: &JsonValue) -> (u16, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// A `200` body, parsed.
async fn served(app: &axum::Router, path: &str, body: &JsonValue) -> JsonValue {
    let (status, text) = post(app, path, body).await;
    assert_eq!(status, 200, "{body}: {text}");
    serde_json::from_str(&text).unwrap()
}

/// A `400` body, parsed.
async fn refused(app: &axum::Router, body: &JsonValue) -> JsonValue {
    let (status, text) = post(app, CHAT, body).await;
    assert_eq!(status, 400, "{body}: {text}");
    serde_json::from_str(&text).unwrap()
}

/// The chunks of a streamed chat completion, `[DONE]` checked and left out.
fn chunks(body: &str) -> Vec<JsonValue> {
    let data: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(data.last(), Some(&"[DONE]"), "{body}");
    data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect()
}

/// The streamed content, joined.
fn streamed_content(chunks: &[JsonValue]) -> String {
    chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect()
}

/// The chunk carrying `finish_reason`.
fn finish_chunk(chunks: &[JsonValue]) -> &JsonValue {
    chunks.iter().find(|c| !c["choices"][0]["finish_reason"].is_null()).expect("a finish chunk")
}

/// A fresh server whose request 0 reads `text` token by token (anything
/// past it reads `?`) and generates until its cap.
fn app_reading(text: &[&'static str]) -> axum::Router {
    let mock = MockCompute::new();
    let decode = text.iter().enumerate().map(|(step, t)| (mock.token_for(0, step as u32), *t)).collect();
    server(&Script::new(decode), SchedulerConfig::default(), Arc::new(mock)).app()
}

/// A plain request, thinking off.
fn plain(extra: JsonValue) -> JsonValue {
    let mut body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 8,
        "enable_thinking": false
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

// ── the table, row by row (acceptance 1) ─────────────────────────────────

/// One inert value and one refused value (`None`: inert at every value) for
/// each row of the chat table.
fn rows() -> Vec<(&'static str, JsonValue, Option<JsonValue>)> {
    vec![
        ("n", json!(1), Some(json!(2))),
        ("logprobs", json!(false), Some(json!(true))),
        ("top_logprobs", json!(null), Some(json!(3))),
        ("logit_bias", json!({}), Some(json!({ "7": 5 }))),
        ("parallel_tool_calls", json!(true), Some(json!(false))),
        ("store", json!(false), Some(json!(true))),
        ("service_tier", json!("auto"), Some(json!("priority"))),
        ("modalities", json!(["text"]), Some(json!(["text", "audio"]))),
        ("metadata", json!({ "run": "7" }), None),
        ("user", json!("u-1"), None),
        ("prompt_cache_key", json!("thread-1"), None),
        ("safety_identifier", json!("s-1"), None),
        ("verbosity", json!("low"), None),
        ("audio", json!(null), Some(json!({ "voice": "alloy", "format": "wav" }))),
        ("prediction", json!(null), Some(json!({ "type": "content", "content": "x" }))),
        ("web_search_options", json!(null), Some(json!({}))),
        ("functions", json!(null), Some(json!([{ "name": "f" }]))),
        ("function_call", json!(null), Some(json!("auto"))),
        ("best_of", json!(null), Some(json!(2))),
        ("echo", json!(null), Some(json!(true))),
        ("suffix", json!(null), Some(json!("x"))),
        ("prompt", json!(null), Some(json!("x"))),
    ]
}

#[test]
fn the_test_rows_are_exactly_the_tables_rows() {
    let tested: BTreeSet<&str> = rows().iter().map(|(name, ..)| *name).collect();
    let table: BTreeSet<&str> = openai_fields::CHAT.iter().map(|r| r.name).collect();
    assert_eq!(tested, table, "a row added to the table needs its values here");
    for (name, _, refused) in rows() {
        let rule = openai_fields::CHAT.iter().find(|r| r.name == name).unwrap();
        assert_eq!(rule.can_refuse(), refused.is_some(), "{name}");
    }
}

#[tokio::test]
async fn an_inert_value_is_served_as_the_body_without_it() {
    let text = ["one ", "two ", "three"];
    let without = served(&app_reading(&text), CHAT, &plain(json!({}))).await;
    for (name, inert, _) in rows() {
        let with = served(&app_reading(&text), CHAT, &plain(json!({ name: inert }))).await;
        assert_eq!(with["choices"], without["choices"], "{name}: {inert}");
        assert_eq!(with["usage"], without["usage"], "{name}: {inert}");
    }
}

#[tokio::test]
async fn a_refused_value_is_a_400_naming_the_field_and_what_to_send() {
    let app = app_reading(&[]);
    for (name, _, value) in rows() {
        let Some(value) = value else { continue };
        let answer = refused(&app, &plain(json!({ name: value }))).await;
        let rule = openai_fields::CHAT.iter().find(|r| r.name == name).unwrap();
        assert_eq!(answer["error"]["type"], "invalid_request_error", "{answer}");
        assert_eq!(answer["error"]["code"], "unsupported_value", "{answer}");
        assert_eq!(answer["error"]["param"], name, "{answer}");
        let message = answer["error"]["message"].as_str().unwrap();
        assert!(message.contains(rule.refusal.unwrap()), "{name}: {message}");
    }
}

// ── the clients this server serves (acceptances 2 and 3) ──────────────────

#[tokio::test]
async fn a_default_shaped_sdk_body_is_served_as_the_body_without_its_defaults() {
    let text = ["one ", "two ", "three"];
    let without = served(&app_reading(&text), CHAT, &plain(json!({}))).await;
    let defaults = json!({
        "n": 1,
        "store": false,
        "parallel_tool_calls": true,
        "logprobs": false,
        "response_format": { "type": "text" },
        "metadata": { "a": "b" },
        "user": "u",
        "service_tier": "auto",
        "some_future_field": { "x": 1 }
    });
    let with = served(&app_reading(&text), CHAT, &plain(defaults)).await;
    assert_eq!(with["choices"], without["choices"]);
    assert_eq!(with["usage"], without["usage"]);
}

#[tokio::test]
async fn the_bodies_recorded_from_this_servers_clients_are_served() {
    // The fields qwen-code sent across the recorded agent traces
    // (`bench/traces/*-trace.jsonl`), and the Playground's request
    // (`web/src/api/request.ts`): none of them is refused.
    let qwen_code = json!({
        "model": "test-model@agent",
        "max_tokens": 4,
        "messages": [{ "role": "system", "content": "s" }, { "role": "user", "content": "hi" }],
        "stream": true,
        "stream_options": { "include_usage": true },
        "tools": [{ "type": "function", "function": { "name": "read_file", "parameters": { "type": "object" } } }],
        "tool_choice": "auto",
        "enable_thinking": false
    });
    let playground = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "stream": true,
        "stream_options": { "include_usage": true },
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4,
        "reasoning_effort": "low",
        "thinking_budget": 64,
        "class": "interactive"
    });
    for body in [qwen_code, playground] {
        let (status, text) = post(&app_reading(&[]), CHAT, &body).await;
        assert_eq!(status, 200, "{body}: {text}");
        let chunks = chunks(&text);
        assert_eq!(finish_chunk(&chunks)["choices"][0]["finish_reason"], "length");
    }
}

// ── max_completion_tokens (acceptance 4) ─────────────────────────────────

#[tokio::test]
async fn max_completion_tokens_alone_caps_the_generation() {
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "max_completion_tokens": 3,
        "enable_thinking": false
    });
    let answer = served(&app_reading(&[]), CHAT, &body).await;
    assert_eq!(answer["choices"][0]["finish_reason"], "length");
    assert_eq!(answer["usage"]["completion_tokens"], 3);
    assert_eq!(answer["choices"][0]["message"]["content"], "???");
}

#[tokio::test]
async fn max_completion_tokens_beside_max_tokens_is_one_cap_or_a_400() {
    let same = plain(json!({ "max_tokens": 2, "max_completion_tokens": 2 }));
    assert_eq!(served(&app_reading(&[]), CHAT, &same).await["usage"]["completion_tokens"], 2);

    let different = plain(json!({ "max_tokens": 2, "max_completion_tokens": 3 }));
    let answer = refused(&app_reading(&[]), &different).await;
    assert_eq!(answer["error"]["param"], "max_completion_tokens", "{answer}");
    assert!(answer["error"]["message"].as_str().unwrap().contains("max_tokens"), "{answer}");
}

// ── stop (acceptance 5) ──────────────────────────────────────────────────

/// Request 0's text, one token per piece: the sequence `END` is split
/// across three tokens, none of them aligned to it.
const SPLIT: [&str; 5] = ["one ", "tw", "o E", "N", "D three"];

#[tokio::test]
async fn stop_ends_the_answer_before_the_sequence_which_is_never_emitted() {
    let body = plain(json!({ "max_tokens": 50, "stop": "END" }));
    let answer = served(&app_reading(&SPLIT), CHAT, &body).await;
    assert_eq!(answer["choices"][0]["message"]["content"], "one two ");
    assert_eq!(answer["choices"][0]["finish_reason"], "stop");
    // The request ended at the token that completed the sequence.
    assert_eq!(answer["usage"]["completion_tokens"], 5);
}

#[tokio::test]
async fn a_streamed_stop_split_across_deltas_leaks_no_fragment() {
    let body = plain(json!({ "max_tokens": 50, "stop": ["END"], "stream": true, "stream_options": { "include_usage": true } }));
    let (status, text) = post(&app_reading(&SPLIT), CHAT, &body).await;
    assert_eq!(status, 200, "{text}");
    let chunks = chunks(&text);
    assert_eq!(streamed_content(&chunks), "one two ");
    assert!(!text.contains("\"E") && !text.contains("three"), "{text}");
    assert_eq!(finish_chunk(&chunks)["choices"][0]["finish_reason"], "stop");
    let usage = &chunks.last().unwrap()["usage"];
    assert_eq!(usage["completion_tokens"], 5, "{text}");
}

#[tokio::test]
async fn the_engine_serves_on_after_a_stop() {
    // The stopped request is released: the next one is admitted and served.
    let app = app_reading(&SPLIT);
    let first = served(&app, CHAT, &plain(json!({ "max_tokens": 50, "stop": "END" }))).await;
    assert_eq!(first["choices"][0]["finish_reason"], "stop");
    let second = served(&app, CHAT, &plain(json!({ "max_tokens": 2 }))).await;
    assert_eq!(second["choices"][0]["finish_reason"], "length");
}

/// #518: a stop sequence the first token completes -- the token that leaves
/// with the prefill -- ends the request there, answered whole and streamed,
/// and the engine serves on.
#[tokio::test]
async fn a_stop_the_first_token_completes_ends_the_request() {
    for stream in [false, true] {
        let app = app_reading(&["END", " more"]);
        let body = match stream {
            true => plain(json!({ "max_tokens": 50, "stop": "END", "stream": true, "stream_options": { "include_usage": true } })),
            false => plain(json!({ "max_tokens": 50, "stop": "END" })),
        };
        let (status, text) = post(&app, CHAT, &body).await;
        assert_eq!(status, 200, "{text}");
        let (content, finish, usage) = if stream {
            let chunks = chunks(&text);
            (streamed_content(&chunks), finish_chunk(&chunks)["choices"][0]["finish_reason"].clone(), chunks.last().unwrap()["usage"].clone())
        } else {
            let answer: JsonValue = serde_json::from_str(&text).unwrap();
            (answer["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(), answer["choices"][0]["finish_reason"].clone(), answer["usage"].clone())
        };
        assert_eq!(content, "", "stream {stream}: {text}");
        assert!(!text.contains("more"), "stream {stream}: {text}");
        assert_eq!(finish, "stop", "stream {stream}: {text}");
        assert_eq!(usage["completion_tokens"], 1, "stream {stream}: {text}");
        let next = served(&app, CHAT, &plain(json!({ "max_tokens": 2 }))).await;
        assert_eq!(next["choices"][0]["finish_reason"], "length", "stream {stream}");
    }
}

/// #518: and it ends before the lane deal, so no decode round runs for it,
/// and with none no drafter proposal.
#[tokio::test]
async fn a_stop_the_first_token_completes_runs_no_decode_round() {
    for stream in [false, true] {
        let mock = Arc::new(MockCompute::new());
        let decode = ["END", " more"].iter().enumerate().map(|(step, t)| (mock.token_for(0, step as u32), *t)).collect();
        let app = server(&Script::new(decode), SchedulerConfig::default(), mock.clone()).app();
        let (status, text) = post(&app, CHAT, &plain(json!({ "max_tokens": 50, "stop": "END", "stream": stream }))).await;
        assert_eq!(status, 200, "{text}");
        assert!(!text.contains("more"), "stream {stream}: {text}");
        // The engine's step that drew the token has finished by the time
        // the next request is answered.
        let next = served(&app, CHAT, &plain(json!({ "max_tokens": 1 }))).await;
        assert_eq!(next["choices"][0]["finish_reason"], "length", "stream {stream}");
        assert!(mock.decode_calls().is_empty(), "stream {stream}: {:?}", mock.decode_calls());
    }
}

#[tokio::test]
async fn stop_never_fires_inside_a_tool_call() {
    let text = [
        "a ",
        "<tool_call>\n<function=write>\n<parameter=text>\nEND\n</parameter>\n</function>\n</tool_call>",
        "b END c",
    ];
    for stream in [false, true] {
        let body = plain(json!({ "max_tokens": 50, "stop": "END", "stream": stream }));
        let (status, text) = post(&app_reading(&text), CHAT, &body).await;
        assert_eq!(status, 200, "{text}");
        let (content, calls, finish) = if stream {
            let chunks = chunks(&text);
            let calls: Vec<JsonValue> = chunks
                .iter()
                .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
                .flatten()
                .cloned()
                .collect();
            (streamed_content(&chunks), calls, finish_chunk(&chunks)["choices"][0]["finish_reason"].clone())
        } else {
            let answer: JsonValue = serde_json::from_str(&text).unwrap();
            let message = &answer["choices"][0]["message"];
            (
                message["content"].as_str().unwrap().to_owned(),
                message["tool_calls"].as_array().unwrap().clone(),
                answer["choices"][0]["finish_reason"].clone(),
            )
        };
        assert_eq!(content, "a b ", "stream {stream}");
        assert_eq!(calls.len(), 1, "stream {stream}: {calls:?}");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"text":"END"}"#);
        assert_eq!(finish, "tool_calls", "a delivered call is still the turn's finish");
    }
}

#[tokio::test]
async fn a_whitespace_stop_does_not_drop_the_calls_it_precedes() {
    // The template's own whitespace before a call, and between two, is
    // content; a caller's "\n\n" must not end the answer there.
    let a = "<tool_call>\n<function=a>\n</function>\n</tool_call>";
    let b = "<tool_call>\n<function=b>\n</function>\n</tool_call>";
    let text = ["Reading it.", "\n\n", a, "\n\n", b, "\n\n", "Done."];
    for stream in [false, true] {
        let body = plain(json!({ "max_tokens": 50, "stop": "\n\n", "stream": stream }));
        let (status, text) = post(&app_reading(&text), CHAT, &body).await;
        assert_eq!(status, 200, "{text}");
        let (content, calls, finish) = if stream {
            let chunks = chunks(&text);
            let calls = chunks.iter().filter(|c| !c["choices"][0]["delta"]["tool_calls"].is_null()).count();
            (streamed_content(&chunks), calls, finish_chunk(&chunks)["choices"][0]["finish_reason"].clone())
        } else {
            let answer: JsonValue = serde_json::from_str(&text).unwrap();
            let message = &answer["choices"][0]["message"];
            let calls = message["tool_calls"].as_array().map_or(0, Vec::len);
            (message["content"].as_str().unwrap().to_owned(), calls, answer["choices"][0]["finish_reason"].clone())
        };
        assert_eq!(calls, 2, "stream {stream}");
        // The last "\n\n" is followed by text, not a call: it fires there.
        assert_eq!(content, "Reading it.\n\n\n\n", "stream {stream}");
        assert_eq!(finish, "tool_calls", "stream {stream}");
    }
}

#[tokio::test]
async fn stop_never_fires_on_reasoning() {
    let text = ["plan END more", "</think>", "answer END rest"];
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 50,
        "enable_thinking": true,
        "thinking_budget": 0,
        "stop": "END"
    });
    let answer = served(&app_reading(&text), CHAT, &body).await;
    let message = &answer["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "plan END more");
    assert_eq!(message["content"], "answer ");
    assert_eq!(answer["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn four_sequences_work_and_five_are_refused() {
    let text = ["x ", "y ", "###", " z"];
    let body = plain(json!({ "max_tokens": 50, "stop": ["STOP", "\n\n", "###", "<|end|>"] }));
    let answer = served(&app_reading(&text), CHAT, &body).await;
    assert_eq!(answer["choices"][0]["message"]["content"], "x y ");

    for bad in [json!(["a", "b", "c", "d", "e"]), json!([]), json!(""), json!(["a", 1]), json!(3)] {
        let answer = refused(&app_reading(&[]), &plain(json!({ "stop": bad }))).await;
        assert_eq!(answer["error"]["param"], "stop", "{answer}");
    }
}

#[tokio::test]
async fn a_stop_that_never_appears_changes_nothing() {
    let text = ["one ", "two ", "three"];
    for stream in [false, true] {
        let without = plain(json!({ "stream": stream }));
        let with = plain(json!({ "stream": stream, "stop": ["NEVER"] }));
        let (_, a) = post(&app_reading(&text), CHAT, &without).await;
        let (_, b) = post(&app_reading(&text), CHAT, &with).await;
        if stream {
            let (a, b) = (chunks(&a), chunks(&b));
            assert_eq!(streamed_content(&a), streamed_content(&b));
            assert_eq!(finish_chunk(&a)["choices"], finish_chunk(&b)["choices"]);
        } else {
            let (a, b): (JsonValue, JsonValue) = (serde_json::from_str(&a).unwrap(), serde_json::from_str(&b).unwrap());
            assert_eq!(a["choices"], b["choices"]);
            assert_eq!(a["usage"], b["usage"]);
        }
    }
}

// ── usage.prompt_tokens_details.cached_tokens (acceptance 6) ─────────────

/// Twenty words: past one whole 16-token page, where a checkpoint is taken.
fn words() -> String {
    (0..20).map(|n| format!("w{n}")).collect::<Vec<_>>().join(" ")
}

/// Two turns of one conversation on one server: the first's cached tokens,
/// and the second's, which resumes from the checkpoint the first left.
async fn two_turns(path: &str, stream: bool) -> (JsonValue, JsonValue) {
    let app = server(&Script::with_boundaries(HashMap::new()), SchedulerConfig::default(), Arc::new(MockCompute::new())).app();
    let words = words();
    let turn = |messages: JsonValue| match path {
        CHAT => json!({
            "model": MODEL, "messages": messages, "max_tokens": 1, "enable_thinking": false,
            "stream": stream, "stream_options": { "include_usage": true }
        }),
        _ => json!({ "model": MODEL, "input": messages, "max_output_tokens": 1, "enable_thinking": false }),
    };
    let first = turn(json!([{ "role": "user", "content": words }]));
    let second = turn(json!([
        { "role": "user", "content": words },
        { "role": "assistant", "content": "ok" },
        { "role": "user", "content": "six" }
    ]));
    let cached = |body: &str| -> JsonValue {
        match (path, stream) {
            (CHAT, true) => chunks(body).last().unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"].clone(),
            (CHAT, false) => serde_json::from_str::<JsonValue>(body).unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"].clone(),
            _ => serde_json::from_str::<JsonValue>(body).unwrap()["usage"]["input_tokens_details"]["cached_tokens"].clone(),
        }
    };
    let (status, a) = post(&app, path, &first).await;
    assert_eq!(status, 200, "{a}");
    let (status, b) = post(&app, path, &second).await;
    assert_eq!(status, 200, "{b}");
    (cached(&a), cached(&b))
}

#[tokio::test]
async fn chat_usage_reports_the_prompt_resumed_as_responses_does() {
    let responses = two_turns("/v1/responses", false).await;
    assert_eq!(responses, (json!(0), json!(20)));
    assert_eq!(two_turns(CHAT, false).await, responses, "the body");
    assert_eq!(two_turns(CHAT, true).await, responses, "the usage chunk");
}

/// `count` words from `stem`: `stem0 stem1 ...`, one token each.
fn words_of(stem: &str, count: u32) -> String {
    (0..count).map(|n| format!("{stem}{n}")).collect::<Vec<_>>().join(" ")
}

/// GitHub #445: R's `cached_tokens` when the claim it held on A's system block
/// was given up while it waited for room, and it then prefilled its whole
/// prompt, asked on `path`.
///
/// One 1,024-token sequence of 16-token pages. A, Interactive, publishes its
/// two-page system block and decodes on 23 pages, held at a decode step while
/// R and I arrive. R, an Agent opening with the same block, claims it and
/// waits: its 47 pages past it do not fit beside A. I, Interactive, needs 63
/// pages and cannot move A, which outranks it. Once A is done, A's checkpoint
/// and the link under it go for I, which is then the block's two pages
/// short: R's claim, ranked below I, goes, the block with it, and I enters.
/// R then finds nothing to resume from and prefills from its first token.
async fn cached_tokens_of_a_claim_given_up(path: &str) -> JsonValue {
    let config = SchedulerConfig {
        max_in_flight: 8,
        max_sequence_tokens: 1024,
        kv_capacity_pages: 64,
        kv_page_tokens: 16,
        host_capacity_bytes: 0,
        prompt_reuse: true,
        retained_slots: 4,
        ..SchedulerConfig::default()
    };
    let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
    let served = server(&Script::with_boundaries(HashMap::new()), config, gated.clone());
    let engine = served.active().engine().cloned().expect("a loaded engine");
    let app = served.app();
    let block = words_of("s", 32);
    let chat_body = |user: String, max_tokens: u32, class: &str| {
        json!({
            "model": MODEL,
            "messages": [{ "role": "system", "content": block }, { "role": "user", "content": user }],
            "max_tokens": max_tokens, "enable_thinking": false, "class": class
        })
    };
    // A is request 0: held at its fifth decode step, past its publish and
    // its capture.
    gated.arm_at(0, 5);
    let a = tokio::spawn({
        let (app, body) = (app.clone(), chat_body(words_of("a", 34), 300, "interactive"));
        async move { post(&app, CHAT, &body).await }
    });
    let mut controller = tokio::task::spawn_blocking(move || {
        controller.wait_entered();
        controller
    })
    .await
    .unwrap();
    let r_body = match path {
        CHAT => chat_body(words_of("r", 50), 700, "agent"),
        _ => json!({
            "model": MODEL,
            "input": [{ "role": "system", "content": block }, { "role": "user", "content": words_of("r", 50) }],
            "max_output_tokens": 700, "enable_thinking": false, "class": "agent"
        }),
    };
    let r = tokio::spawn({
        let (app, path) = (app.clone(), path.to_owned());
        async move { post(&app, &path, &r_body).await }
    });
    let i = tokio::spawn({
        let (app, body) = (app.clone(), json!({
            "model": MODEL, "messages": [{ "role": "user", "content": words_of("i", 4) }],
            "max_tokens": 1000, "enable_thinking": false
        }));
        async move { post(&app, CHAT, &body).await }
    });
    // A decodes a step at a time until R and I wait beside it: the step that
    // holds A again ran after their first look for state to resume from.
    for _ in 0..200 {
        if engine.interval_counters().waiting == 2 {
            break;
        }
        gated.arm_at(0, 0);
        controller = tokio::task::spawn_blocking(move || {
            controller.release();
            controller.wait_entered();
            controller
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(engine.interval_counters().waiting, 2, "R and I wait while A decodes");
    tokio::task::spawn_blocking(move || controller.release()).await.unwrap();
    for (name, handle) in [("A", a), ("I", i)] {
        let (status, text) = handle.await.unwrap();
        assert_eq!(status, 200, "{name}: {text}");
    }
    let (status, text) = r.await.unwrap();
    assert_eq!(status, 200, "R: {text}");
    let answer: JsonValue = serde_json::from_str(&text).unwrap();
    match path {
        CHAT => answer["usage"]["prompt_tokens_details"]["cached_tokens"].clone(),
        _ => answer["usage"]["input_tokens_details"]["cached_tokens"].clone(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_claim_given_up_before_its_prefill_reports_no_cached_tokens() {
    // GitHub #445: `cached_tokens` is the prompt the request resumed from,
    // not the prefix it once claimed and let go of.
    for path in [CHAT, "/v1/responses"] {
        assert_eq!(cached_tokens_of_a_claim_given_up(path).await, json!(0), "{path}");
    }
}

// ── unchanged output (acceptance 7) ──────────────────────────────────────

#[tokio::test]
async fn a_request_without_stop_is_served_as_before() {
    // Without `stop`, the non-streaming answer is the recorded split of the
    // generated tokens, and the stream joins to the same text.
    let text = ["<tool_call>\n<function=a>\n</function>\n</tool_call>", "one ", "two"];
    let recorded = json!({
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "one two???",
            "tool_calls": [{ "id": "call_0", "type": "function", "function": { "name": "a", "arguments": "{}" } }],
            // GitHub #587: OpenAI's `refusal`, always null here.
            "refusal": null
        },
        "finish_reason": "length"
    });
    let body = plain(json!({ "max_tokens": 6 }));
    let answer = served(&app_reading(&text), CHAT, &body).await;
    assert_eq!(answer["choices"][0], recorded);
    let (_, streamed) = post(&app_reading(&text), CHAT, &plain(json!({ "max_tokens": 6, "stream": true }))).await;
    assert_eq!(streamed_content(&chunks(&streamed)), "one two???");
}
