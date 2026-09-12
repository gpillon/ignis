//! P4-05 (GitHub #123): the canary sanity floor under hq-e8-2b KV.
//!
//! This is a **sanity floor, not an agreement score** — the distinction ADR
//! 0022 turns on. hq is lossy, so its greedy text is not required to match
//! BF16's token for token, and nothing here compares the two. What is
//! required is that a model served out of a quantized KV cache still answers
//! coherently and still answers the same way twice: the two properties
//! `ignis_bench::canary` already defines for the G1 self-check (ADR 0007) —
//! *sane* (non-empty, no NUL bytes, no runaway repetition) and
//! *deterministic* (greedy, same prompt, same output).
//!
//! The scoring lives in `ignis_bench::canary` rather than here so that the
//! sanity rule is one definition shared with `ignis-bench canary`, and this
//! file is only the GPU driver that produces the two completions.
//!
//! **The KV format cannot silently be the wrong one.** The pool budget below
//! is 512 MiB, which holds one 40,960-token context under hq-e8-2b (377 MB)
//! and does not under BF16 (2.5 GiB). A load that ignored `kv_format` would
//! be refused by `CudaLeafConfig::kv_pool_plan` naming the budget, the format
//! and the capacity it bought — so this test either runs on hq or fails, and
//! never passes having quietly measured the oracle format instead.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or an unavailable GPU is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_bench::canary::{CANARIES, evaluate};
use ignis_core::KvFormat;
use ignis_core::gpu_profile;
use ignis_logging::NullSink;
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::telemetry::SystemClock;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// A budget only hq can hold one configured context inside — see the module
/// doc: this is what stops the test from passing on the wrong format.
const KV_POOL_BYTES: u64 = 512 * 1024 * 1024;
/// Enough for a one-or-two-sentence answer to every canary, and short enough
/// that four prompts run twice each stays a quick leg of the GPU profile.
const MAX_TOKENS: u32 = 48;

/// The GPU-backed router plus the model thread that owns its VRAM. Dropping
/// `app` disconnects every `Engine` clone; joining `driver` then blocks until
/// the model thread has actually released the weights (the same teardown
/// `openai_http_gpu.rs` documents).
struct Harness {
    app: Option<axum::Router>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    fn app(&self) -> &axum::Router {
        self.app.as_ref().expect("app is only taken by Drop")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.app.take());
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
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));

    let shape = EngineShape {
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes: KV_POOL_BYTES,
        ..EngineShape::default()
    };
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok(scheduler) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler (hq-e8-2b): {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (engine, driver) = Engine::with_sinks_and_driver(
        Box::new(scheduler),
        Arc::new(NullSink),
        Arc::new(SystemClock),
    );
    let server = Server::with_artifact_template(engine, frontend)
        .with_request_timeout(Duration::from_secs(120));
    Some(Harness {
        app: Some(server.app()),
        driver: Some(driver),
    })
}

/// One greedy, non-streaming completion. Temperature is left unset, which the
/// API layer resolves to 0.0 — greedy sampling, the determinism check's own
/// precondition. Thinking is off so the token budget buys content rather than
/// a reasoning channel.
async fn complete(app: &axum::Router, prompt: &str) -> String {
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": MAX_TOKENS,
        "stream": false,
        "enable_thinking": false
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string().into_bytes()))
        .expect("build the request");
    let response = app.clone().oneshot(request).await.expect("route the request");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read the body");
    let text = String::from_utf8(bytes.to_vec()).expect("the body is UTF-8");
    assert_eq!(status, 200, "chat completion should be 200: {text}");
    let value: serde_json::Value = serde_json::from_str(&text).expect("the body is JSON");
    value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("a completion must carry message content: {text}"))
        .to_string()
}

#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_canary_suite_is_sane_and_deterministic_under_hq_kv() {
    let Some(h) = harness() else { return };

    let mut failures = Vec::new();
    for canary in CANARIES {
        let first = complete(h.app(), canary.prompt).await;
        let second = complete(h.app(), canary.prompt).await;
        let result = evaluate(canary.id, &first, &second);
        // Printed for every canary, pass or fail: a sanity floor is only
        // useful if the text it accepted is visible when someone later asks
        // what "coherent under hq" actually looked like.
        println!(
            "[{}] sane={} deterministic={}\n  {}",
            result.id,
            result.sane,
            result.deterministic,
            first.replace('\n', "\n  ")
        );
        if !result.consistent() {
            failures.push(format!(
                "{}: sane={} ({}), deterministic={}",
                result.id,
                result.sane,
                result.sane_reason.clone().unwrap_or_else(|| "-".into()),
                result.deterministic
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the canary suite must stay coherent and deterministic under hq-e8-2b KV \
         (a sanity floor, not an agreement score — ADR 0022): {}",
        failures.join("; ")
    );
}
