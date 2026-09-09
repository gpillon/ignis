//! End-to-end HTTP coverage for P3-04's sampling surface (GitHub #101).
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
    PrefillJob, RequestId, SchedulerConfig,
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
    fn prefill_step(&self, _jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        Ok(())
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let mut generated = self.generated.lock().unwrap();
        Ok(jobs
            .iter()
            .map(|job| {
                if !generated.insert(job.request) {
                    return DecodeOutcome::Finished(FinishReason::Length);
                }
                DecodeOutcome::Token(token_for(job.params))
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
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        Arc::new(SamplingEchoCompute::default()),
    );
    Server::new(
        Engine::new(Box::new(scheduler)),
        Box::new(SimpleTemplateProvider),
    )
    .with_request_timeout(Duration::from_secs(5))
    .app()
}

async fn call(app: &axum::Router, body: Value) -> (u16, String) {
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
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

#[tokio::test]
async fn absent_sampling_parameters_preserve_greedy_fixed_seed_defaults() {
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 1,
    });
    let (status, response) = call(&app(), body).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(content(&response, false), "100");
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
async fn stochastic_only_settings_are_refused_when_greedy_would_ignore_them() {
    for (field, value) in [
        ("top_p", json!(0.7)),
        ("top_k", json!(7)),
        ("presence_penalty", json!(0.4)),
        ("frequency_penalty", json!(-0.4)),
    ] {
        let (status, body) = call(&app(), request(field, value, false)).await;
        assert_eq!(status, 400, "{field}: {body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["code"], "invalid_sampling_parameter");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("temperature must be greater than 0")
        );
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
