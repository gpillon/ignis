//! End-to-end HTTP coverage for P3-04's sampling surface (GitHub #101)
//! and its model-card defaults (spec server/12, GitHub #297).
//!
//! The test compute is the CPU stand-in for the GPU boundary (ADR 0006): it
//! turns the `DecodeParams` it receives into a token id, so the response body
//! proves each wire parameter reached the scheduler independently.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use http_body_util::BodyExt;
use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, DecodeParams, FinishReason,
    PrefillJob, PrefillOutcome, RequestId, SchedulerConfig,
};
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "test-model";

#[derive(Default)]
struct SamplingEchoCompute {
    generated: Mutex<HashSet<RequestId>>,
}

impl Compute for SamplingEchoCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        Ok(PrefillOutcome::nothing_encoded(jobs.len()))
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let mut generated = self.generated.lock().unwrap();
        Ok(jobs
            .iter()
            .map(|job| {
                if !generated.insert(job.request) {
                    return DecodeOutcome::finished(FinishReason::Length);
                }
                DecodeOutcome::token(token_for(job.params))
            })
            .collect())
    }
}

fn token_for(params: DecodeParams) -> u32 {
    if params.temperature == 0.5 {
        101
    } else if params.top_p == 0.7 {
        102
    } else if params.top_k == 7 {
        103
    } else if params.presence_penalty == 0.4 {
        104
    } else if params.frequency_penalty == -0.4 {
        105
    } else if params.seed == 9 {
        106
    } else if params.seed == (-9_i64) as u64 {
        107
    } else {
        100
    }
}

fn app() -> axum::Router {
    app_over(Arc::new(SamplingEchoCompute::default()))
}

fn app_over(compute: Arc<dyn Compute>) -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    Server::new(
        Engine::new(Box::new(scheduler)),
        Box::new(SimpleTemplateProvider),
    )
    .with_request_timeout(Duration::from_secs(5))
    .app()
}

async fn call(app: &axum::Router, body: Value) -> (u16, String) {
    call_at(app, "/v1/chat/completions", body).await
}

fn request(field: &str, value: Value, stream: bool) -> Value {
    let mut body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 1,
        "stream": stream,
    });
    body[field] = value;
    body
}

fn content(body: &str, stream: bool) -> String {
    if !stream {
        let value: Value = serde_json::from_str(body).unwrap();
        return value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
    }
    body.lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

async fn assert_parameter_reaches_compute(field: &str, value: Value, expected_token: u32) {
    for stream in [false, true] {
        let mut request = request(field, value.clone(), stream);
        if matches!(
            field,
            "top_p" | "top_k" | "presence_penalty" | "frequency_penalty"
        ) {
            request["temperature"] = json!(1.0);
        }
        let (status, body) = call(&app(), request).await;
        assert_eq!(status, 200, "{field}, stream={stream}: {body}");
        assert_eq!(content(&body, stream), expected_token.to_string());
    }
}

#[tokio::test]
async fn temperature_reaches_each_chat_completion_mode() {
    assert_parameter_reaches_compute("temperature", json!(0.5), 101).await;
}

#[tokio::test]
async fn top_p_reaches_each_chat_completion_mode() {
    assert_parameter_reaches_compute("top_p", json!(0.7), 102).await;
}

#[tokio::test]
async fn top_k_extension_reaches_each_chat_completion_mode() {
    assert_parameter_reaches_compute("top_k", json!(7), 103).await;
}

#[tokio::test]
async fn presence_penalty_reaches_each_chat_completion_mode() {
    assert_parameter_reaches_compute("presence_penalty", json!(0.4), 104).await;
}

#[tokio::test]
async fn frequency_penalty_reaches_each_chat_completion_mode() {
    assert_parameter_reaches_compute("frequency_penalty", json!(-0.4), 105).await;
}

#[tokio::test]
async fn seed_reaches_each_chat_completion_mode() {
    assert_parameter_reaches_compute("seed", json!(9), 106).await;
    assert_parameter_reaches_compute("seed", json!(-9), 107).await;
}

/// The test compute for the defaults (spec server/12): it records the
/// `DecodeParams` each request's first decode job carries, so a test reads
/// back exactly what the API layer resolved.
#[derive(Default)]
struct RecordingCompute {
    seen: Mutex<Vec<DecodeParams>>,
    generated: Mutex<HashSet<RequestId>>,
}

impl Compute for RecordingCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        Ok(PrefillOutcome::nothing_encoded(jobs.len()))
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let mut generated = self.generated.lock().unwrap();
        Ok(jobs
            .iter()
            .map(|job| {
                if !generated.insert(job.request) {
                    return DecodeOutcome::finished(FinishReason::Length);
                }
                self.seen.lock().unwrap().push(job.params);
                DecodeOutcome::token(100)
            })
            .collect())
    }
}

fn recording_app() -> (axum::Router, Arc<RecordingCompute>) {
    let compute = Arc::new(RecordingCompute::default());
    (app_over(compute.clone()), compute)
}

async fn call_at(app: &axum::Router, uri: &str, body: Value) -> (u16, String) {
    let request = axum::http::Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// The three request shapes every default must hold on: chat completions
/// streaming and not, and `/v1/responses`. `fields` are merged into each.
fn surfaces(fields: Value) -> Vec<(&'static str, &'static str, Value)> {
    let chat = |stream: bool| {
        json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 1,
            "stream": stream,
        })
    };
    let responses = json!({ "model": MODEL, "input": "hello", "max_output_tokens": 1 });
    let mut out = vec![
        ("chat", "/v1/chat/completions", chat(false)),
        ("chat stream", "/v1/chat/completions", chat(true)),
        ("responses", "/v1/responses", responses),
    ];
    for (_, _, body) in &mut out {
        for (key, value) in fields.as_object().unwrap() {
            body[key] = value.clone();
        }
    }
    out
}

/// What one request resolved to, read back from the recording compute.
async fn resolved(uri: &str, body: Value) -> DecodeParams {
    let (app, compute) = recording_app();
    let (status, response) = call_at(&app, uri, body).await;
    assert_eq!(status, 200, "{uri}: {response}");
    let seen = compute.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "{uri}: one decode job");
    seen[0]
}

/// (temperature, top_p, top_k, presence_penalty, frequency_penalty)
fn sampling(params: &DecodeParams) -> (f32, f32, i32, f32, f32) {
    (
        params.temperature,
        params.top_p,
        params.top_k,
        params.presence_penalty,
        params.frequency_penalty,
    )
}

#[tokio::test]
async fn absent_sampling_takes_the_model_cards_thinking_row() {
    for (name, uri, body) in surfaces(json!({})) {
        let params = resolved(uri, body).await;
        assert_eq!(sampling(&params), (1.0, 0.95, 20, 0.0, 0.0), "{name}");
    }
}

#[tokio::test]
async fn absent_sampling_without_thinking_takes_the_model_cards_instruct_row() {
    for (name, uri, body) in surfaces(json!({ "enable_thinking": false })) {
        let params = resolved(uri, body).await;
        assert_eq!(sampling(&params), (0.7, 0.8, 20, 1.5, 0.0), "{name}");
    }
}

#[tokio::test]
async fn an_explicit_field_replaces_only_its_own_default() {
    for (fields, expected) in [
        (json!({ "temperature": 0.6 }), (0.6, 0.95, 20, 0.0, 0.0)),
        (json!({ "top_p": 0.5 }), (1.0, 0.5, 20, 0.0, 0.0)),
        (json!({ "top_k": 5 }), (1.0, 0.95, 5, 0.0, 0.0)),
        (json!({ "presence_penalty": 0.4 }), (1.0, 0.95, 20, 0.4, 0.0)),
        (
            json!({ "enable_thinking": false, "presence_penalty": 0.0 }),
            (0.7, 0.8, 20, 0.0, 0.0),
        ),
    ] {
        for (name, uri, body) in surfaces(fields.clone()) {
            let params = resolved(uri, body).await;
            assert_eq!(sampling(&params), expected, "{name}: {fields}");
        }
    }
}

#[tokio::test]
async fn an_explicit_zero_temperature_is_greedy_with_neutral_filters() {
    for thinking in [true, false] {
        let fields = json!({ "temperature": 0, "enable_thinking": thinking });
        for (name, uri, body) in surfaces(fields) {
            let params = resolved(uri, body).await;
            assert_eq!(sampling(&params), (0.0, 1.0, 0, 0.0, 0.0), "{name}, thinking={thinking}");
        }
    }
}

#[tokio::test]
async fn top_p_without_temperature_is_served() {
    // What opencode sends for a model that does not declare temperature
    // support: top_p and top_k, no temperature (spec server/12).
    for (name, uri, body) in surfaces(json!({ "top_p": 0.95, "top_k": 20 })) {
        let params = resolved(uri, body).await;
        assert_eq!(sampling(&params), (1.0, 0.95, 20, 0.0, 0.0), "{name}");
    }
}

#[tokio::test]
async fn seedless_requests_draw_seeds_of_their_own() {
    for (name, uri, body) in surfaces(json!({})) {
        let first = resolved(uri, body.clone()).await;
        let second = resolved(uri, body).await;
        assert_ne!(first.seed, second.seed, "{name}: two seedless requests share a seed");
    }
    for (name, uri, body) in surfaces(json!({ "seed": 9 })) {
        assert_eq!(resolved(uri, body).await.seed, 9, "{name}");
    }
}

#[tokio::test]
async fn values_the_engine_cannot_honour_are_clear_400_errors() {
    let cases = [
        ("temperature", json!(-0.1), "between 0 and 2"),
        ("temperature", json!(2.1), "between 0 and 2"),
        ("top_p", json!(-0.1), "between 0 and 1"),
        ("top_p", json!(1.1), "between 0 and 1"),
        ("top_k", json!(-1), "ignis extension"),
        ("top_k", json!(21), "ignis extension"),
        ("presence_penalty", json!(-2.1), "between -2 and 2"),
        ("frequency_penalty", json!(2.1), "between -2 and 2"),
    ];
    for (field, value, message) in cases {
        let (status, body) = call(&app(), request(field, value, false)).await;
        assert_eq!(status, 400, "{field}: {body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["code"], "invalid_sampling_parameter");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains(message),
            "{field}: {body}"
        );
    }
}

#[tokio::test]
async fn numeric_shapes_and_widths_use_the_sampling_error_contract() {
    for (field, value, message) in [
        ("top_k", json!(1.5), "ignis extension"),
        ("top_k", json!(u64::MAX), "ignis extension"),
        ("seed", json!(u64::MAX), "signed 64-bit integer"),
    ] {
        let (status, body) = call(&app(), request(field, value, false)).await;
        assert_eq!(status, 400, "{field}: {body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["code"], "invalid_sampling_parameter");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains(message),
            "{field}: {body}"
        );
    }
}

#[tokio::test]
async fn nonzero_values_that_underflow_f32_are_rejected_not_ignored() {
    for field in ["temperature", "presence_penalty", "frequency_penalty"] {
        let mut body = request(field, json!(1e-50), false);
        body["temperature"] = if field == "temperature" {
            json!(1e-50)
        } else {
            json!(1.0)
        };
        let (status, body) = call(&app(), body).await;
        assert_eq!(status, 400, "{field}: {body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["code"], "invalid_sampling_parameter");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("too small to be represented"),
            "{field}: {body}"
        );
    }
}

#[tokio::test]
async fn stochastic_only_settings_are_refused_when_explicit_greedy_would_ignore_them() {
    for (field, value) in [
        ("top_p", json!(0.7)),
        ("top_k", json!(7)),
        ("presence_penalty", json!(0.4)),
        ("frequency_penalty", json!(-0.4)),
    ] {
        let mut fields = json!({ "temperature": 0 });
        fields[field] = value;
        for (name, uri, body) in surfaces(fields) {
            let (status, body) = call_at(&app(), uri, body).await;
            assert_eq!(status, 400, "{name}, {field}: {body}");
            let error: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(error["error"]["code"], "invalid_sampling_parameter", "{name}, {field}");
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("temperature must be greater than 0"),
                "{name}, {field}: {body}"
            );
        }
    }
}

#[tokio::test]
async fn concurrent_requests_keep_distinct_sampling_settings() {
    let app = app();
    let left = tokio::spawn({
        let app = app.clone();
        async move {
            let mut body = request("top_k", json!(7), false);
            body["temperature"] = json!(1.0);
            call(&app, body).await
        }
    });
    let right = tokio::spawn({
        let app = app.clone();
        async move {
            let mut body = request("frequency_penalty", json!(-0.4), false);
            body["temperature"] = json!(1.0);
            call(&app, body).await
        }
    });

    let (left, right) = tokio::join!(left, right);
    let (left_status, left_body) = left.unwrap();
    let (right_status, right_body) = right.unwrap();
    assert_eq!(left_status, 200, "{left_body}");
    assert_eq!(right_status, 200, "{right_body}");
    assert_eq!(content(&left_body, false), "103");
    assert_eq!(content(&right_body, false), "105");
}
