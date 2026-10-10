//! A live `ignis-server` over the real GPU-backed scheduler, for the
//! `#[ignore]`d GPU-profile tests that measure the production stack through
//! its own HTTP surface (`ttft_cell_gpu.rs`, `needle_128k_gpu.rs`).
//!
//! Those tests differ only in the engine shape they need and what they then
//! measure; standing one up is the same work every time, and getting its
//! *teardown* wrong is a GPU leak rather than a test failure — which is why
//! it lives here once instead of being copied per test file.
//!
//! Compiled only under the `cuda` feature (its callers are all
//! `#![cfg(feature = "cuda")]`).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::Server;

/// A live server: the production router over the real GPU-backed
/// scheduler, served on a random localhost port.
///
/// The teardown order matters (GitHub #71): the model thread owns the
/// scheduler's GPU-resident state and only frees it when it exits, so the
/// serve runtime is dropped first (disconnecting every `Engine` clone) and
/// the driver thread is joined afterwards. `Drop` does that, so a test only
/// has to keep the value alive.
pub struct LiveServer {
    /// The base URL to point an `HttpEndpoint` at.
    pub url: String,
    /// A second `FrontendSet` off the same artifact: the server's template
    /// provider consumes one, and this is the *instrument's* own tokenizer
    /// + chat template. Both come from the same artifact, which is the
    /// point — a prompt's claimed length is measured against the length the
    /// engine itself will count.
    pub frontend: FrontendSet,
    runtime: Option<tokio::runtime::Runtime>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Drop for LiveServer {
    fn drop(&mut self) {
        drop(self.runtime.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

impl LiveServer {
    /// Start a server on `artifact` with `shape`, or `None` when the GPU
    /// profile says to skip (a missing artifact or an unavailable GPU —
    /// outside `IGNIS_GPU_PROFILE=1` those are a skip, under it a hard
    /// failure, and that decision is `gpu_profile`'s to make, never a
    /// test's own early return).
    ///
    /// `request_timeout` is the *server's* deadline for a request; a test
    /// measuring long prefills wants it generous enough that what it
    /// catches is the measurement rather than a timeout of its own.
    pub fn start(
        artifact: &str,
        model: &str,
        shape: EngineShape,
        request_timeout: Duration,
    ) -> Option<Self> {
        let path = Path::new(artifact);
        if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {artifact}")) {
            return None;
        }
        let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
        let frontend =
            FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
        let eos = frontend
            .eos_token_id()
            .unwrap_or_else(|| panic!("{model} generation config must carry eos_token_id"));
        let instrument_frontend =
            FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));

        // GitHub #210: the reservations beside the scheduler are for
        // `/metrics`, which this harness does not read.
        let scheduler = match cuda_scheduler(path, model.into(), eos, shape) {
            Ok((scheduler, _reservations)) => scheduler,
            Err(e) => {
                if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                    return None;
                }
                unreachable!();
            }
        };
        // `Engine::with_clock_and_driver` spawns its telemetry task with
        // `tokio::spawn`, which needs a live reactor — build the runtime
        // and enter it before touching the engine (a `#[tokio::test]` gets
        // this for free; a plain `#[test]` has to do it explicitly).
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("serve runtime");
        let _guard = runtime.enter();

        let (engine, driver) =
            Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
        let app = Server::with_artifact_template(engine, frontend)
            .with_request_timeout(request_timeout)
            .app();

        let url = runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind a local port");
            let port = listener.local_addr().expect("local addr").port();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            format!("http://127.0.0.1:{port}")
        });
        Some(Self {
            url,
            frontend: instrument_frontend,
            runtime: Some(runtime),
            driver: Some(driver),
        })
    }
}

/// A live server that can switch models (spec model-switch/01): built the
/// way `main` builds one — the production load path
/// (`load::prepare_model` / `load::load_model`), the model thread's
/// handle kept on the loaded model, the production `ArtifactLoader` behind
/// `POST /v1/models/switch` — and served on a random localhost port.
///
/// Not a [`LiveServer`]: that one keeps the driver handle beside the
/// server, so a switch away from its model would tear it down without a join
/// and the next load's pinned-arena create would refuse. Here the handle
/// travels with the model, and teardown shuts down whichever model is
/// loaded when the harness drops.
pub struct SwitchingLiveServer {
    /// The base URL.
    pub url: String,
    /// The server the listener serves (its `active` model is the one loaded
    /// now).
    pub server: Server,
    runtime: Option<tokio::runtime::Runtime>,
}

impl Drop for SwitchingLiveServer {
    fn drop(&mut self) {
        drop(self.runtime.take());
        self.server.active().shut_down();
    }
}

impl SwitchingLiveServer {
    /// Load `artifact` as `model` under `options` and serve it, warm, or
    /// `None` when the GPU profile says to skip (a missing artifact or an
    /// unavailable GPU, as [`LiveServer::start`]).
    pub fn start(options: ignis_server::config::Config, artifact: &str, model: &str) -> Option<Self> {
        let path = Path::new(artifact);
        if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {artifact}")) {
            return None;
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("serve runtime");
        let _guard = runtime.enter();
        let options = ignis_server::config::Config { model: model.to_owned(), model_named: true, ..options };
        let loaded = match ignis_server::load::prepare_model(&options, path).and_then(ignis_server::load::load_model) {
            Ok(loaded) => loaded,
            Err(e) => {
                gpu_profile::skip_or_fail(&format!("load {artifact}: {e}"));
                return None;
            }
        };
        let server = Server::from_active(loaded.model).with_warm_up().with_switcher(
            ignis_server::model_switch::Switcher::new(
                Arc::new(ignis_server::model_switch::ArtifactLoader::new(options)),
                Duration::from_secs(30),
            ),
        );
        let served = server.clone();
        let url = runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind a local port");
            let port = listener.local_addr().expect("local addr").port();
            tokio::spawn(served.serve_on_until(listener, None, std::future::pending()));
            format!("http://127.0.0.1:{port}")
        });
        let live = Self { url, server, runtime: Some(runtime) };
        live.until_serving(model);
        Some(live)
    }

    /// `GET /v1/models` until it reports `serving`, which must be `model`.
    /// After a `202` the status reads `switching` until the switch ends, so
    /// `serving` under another id is a switch that failed and reloaded the
    /// previous model — a test failure, said with the body. `failed` is
    /// waited through: it is what the API says while that reload runs.
    pub fn until_serving(&self, model: &str) {
        let runtime = self.runtime.as_ref().expect("serving");
        runtime.block_on(async {
            let client = reqwest::Client::new();
            let deadline = std::time::Instant::now() + Duration::from_secs(900);
            let mut last = serde_json::Value::Null;
            loop {
                assert!(std::time::Instant::now() < deadline, "{model} never served; last: {last}");
                if let Ok(response) = client.get(format!("{}/v1/models", self.url)).send().await {
                    if response.status().is_success() {
                        last = serde_json::from_str(&response.text().await.expect("models body")).expect("models json");
                        if last["status"] == "serving" {
                            assert_eq!(last["data"][0]["id"], model, "the switch to {model} failed: {last}");
                            return;
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
    }

    /// Switch to `artifact` as `model` through `POST /v1/models/switch` and
    /// wait until `GET /v1/models` reports it serving: the wall time from the
    /// request to that answer, the measured cost of one direction.
    pub fn switch_to(&self, artifact: &str, model: &str) -> Duration {
        let started = std::time::Instant::now();
        let accepted = self.runtime.as_ref().expect("serving").block_on(async {
            reqwest::Client::new()
                .post(format!("{}/v1/models/switch", self.url))
                .header("content-type", "application/json")
                .body(serde_json::json!({ "artifact": artifact, "model": model }).to_string())
                .send()
                .await
                .expect("the switch request")
                .status()
        });
        assert_eq!(accepted.as_u16(), 202, "the switch to {model} was not accepted");
        self.until_serving(model);
        started.elapsed()
    }

    /// One short chat completion on the loaded model, answered `200`.
    pub fn completes(&self, model: &str) {
        let status = self.runtime.as_ref().expect("serving").block_on(async {
            reqwest::Client::new()
                .post(format!("{}/v1/chat/completions", self.url))
                .header("content-type", "application/json")
                .body(
                    serde_json::json!({
                        "model": model,
                        "messages": [{ "role": "user", "content": "Say hello." }],
                        "max_tokens": 8,
                    })
                    .to_string(),
                )
                .send()
                .await
                .expect("the completion request")
                .status()
        });
        assert_eq!(status.as_u16(), 200, "{model} did not complete a request");
    }
}
