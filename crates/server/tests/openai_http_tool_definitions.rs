//! `tools` / `tool_choice` request wire contract (GitHub #132): end-to-end
//! tests over the real axum router, on the CPU gate (ADR 0006 —
//! `MockCompute`, no GPU).
//!
//! Mirrors `openai_http_thinking.rs`'s harness shape (prior art): a
//! recording [`TemplateProvider`] double captures every `tools` slice
//! `apply_chat_template` is called with, so a test can assert on what
//! actually reached the template seam without reaching inside the server.
//! Whether the *real* jinja template renders `tools` into the "# Tools"
//! section is `ignis_artifact::frontend`'s own concern
//! (`render_with_thinking_and_tools_binds_the_tools_array`) — this file
//! only pins the HTTP → seam wire contract: validation, `tool_choice`
//! semantics, and that a well-formed request's tools actually arrive.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";

/// A [`TemplateProvider`] double: templating delegates to the built-in
/// placeholder, but every `tools` slice `apply_chat_template` receives is
/// captured for the test to inspect afterward.
struct RecordingTemplateProvider {
    inner: ignis_server::template::SimpleTemplateProvider,
    captured_tools: Mutex<Vec<Vec<serde_json::Value>>>,
}

impl RecordingTemplateProvider {
    fn new() -> Self {
        Self {
            inner: ignis_server::template::SimpleTemplateProvider,
            captured_tools: Mutex::new(Vec::new()),
        }
    }

    fn captured_tools(&self) -> Vec<Vec<serde_json::Value>> {
        self.captured_tools.lock().unwrap().clone()
    }
}

impl TemplateProvider for RecordingTemplateProvider {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[serde_json::Value],
    ) -> Vec<TokenId> {
        self.captured_tools.lock().unwrap().push(tools.to_vec());
        self.inner.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        self.inner.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        self.inner.token_decoder()
    }
}

struct Harness {
    app: axum::Router,
    template: Arc<RecordingTemplateProvider>,
}

fn harness() -> Harness {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    let template = Arc::new(RecordingTemplateProvider::new());
    let server = Server::new(Engine::new(Box::new(scheduler)), {
        struct Shared(Arc<RecordingTemplateProvider>);
        impl TemplateProvider for Shared {
            fn apply_chat_template(
                &self,
                m: &[ChatMessage],
                o: &ThinkingOptions,
                tools: &[serde_json::Value],
            ) -> Vec<TokenId> {
                self.0.apply_chat_template(m, o, tools)
            }
            fn render_tokens(&self, t: &[TokenId]) -> String {
                self.0.render_tokens(t)
            }
            fn thinking_capabilities(&self) -> ThinkingCapabilities {
                self.0.thinking_capabilities()
            }
            fn token_decoder(&self) -> Box<dyn TokenDecoder> {
                self.0.token_decoder()
            }
        }
        Box::new(Shared(Arc::clone(&template)))
    })
    .with_request_timeout(Duration::from_secs(5));
    Harness {
        app: server.app(),
        template,
    }
}

async fn call(
    app: &axum::Router,
    body: serde_json::Value,
) -> (u16, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string().into_bytes()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn weather_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    })
}

#[tokio::test]
async fn a_well_formed_tools_array_reaches_the_template() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "tools": [weather_tool()]
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 200, "{body}");
    let captured = h.template.captured_tools();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].len(), 1);
    assert_eq!(captured[0][0]["function"]["name"], "get_weather");
}

#[tokio::test]
async fn no_tools_field_reaches_the_template_as_an_empty_slice() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(h.template.captured_tools(), vec![Vec::<serde_json::Value>::new()]);
}

#[tokio::test]
async fn a_tool_missing_type_is_a_400() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [{"function": {"name": "get_weather"}}]
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("tools[0]"), "{body}");
    assert!(h.template.captured_tools().is_empty(), "must never reach the template");
}

#[tokio::test]
async fn a_tool_with_an_empty_function_name_is_a_400() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [{"type": "function", "function": {"name": ""}}]
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn tool_choice_none_suppresses_tools_even_when_well_formed() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "tools": [weather_tool()],
        "tool_choice": "none"
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(h.template.captured_tools(), vec![Vec::<serde_json::Value>::new()]);
}

#[tokio::test]
async fn tool_choice_auto_passes_tools_through() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "tools": [weather_tool()],
        "tool_choice": "auto"
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(h.template.captured_tools()[0].len(), 1);
}

#[tokio::test]
async fn tool_choice_required_is_a_400_naming_the_reason() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [weather_tool()],
        "tool_choice": "required"
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("force"), "{body}");
}

#[tokio::test]
async fn tool_choice_naming_a_specific_function_is_a_400() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [weather_tool()],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn an_unknown_tool_choice_string_is_a_400() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tool_choice": "sometimes"
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 400, "{body}");
}

/// An assistant history message's `tool_calls` — a prior turn's call —
/// round-trips through the wire type and reaches `apply_chat_template`'s
/// `messages` (the real template's own rendering of that shape is
/// `ignis_artifact::frontend`'s concern, pinned there directly).
#[tokio::test]
async fn assistant_history_tool_calls_are_accepted_on_the_wire() {
    let h = harness();
    let req = serde_json::json!({
        "messages": [
            { "role": "user", "content": "what's the weather in Turin?" },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Turin\"}"}
                }]
            },
            { "role": "tool", "tool_call_id": "call_0", "content": "18C, cloudy" }
        ],
        "max_tokens": 1,
        "enable_thinking": false,
        "tools": [weather_tool()]
    });
    let (status, body) = call(&h.app, req).await;
    assert_eq!(status, 200, "{body}");
}
