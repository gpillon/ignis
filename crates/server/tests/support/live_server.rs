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
use ignis_logging::NullSink;
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

        let scheduler = match cuda_scheduler(path, model.into(), eos, shape) {
            Ok(scheduler) => scheduler,
            Err(e) => {
                if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                    return None;
                }
                unreachable!();
            }
        };
        // `Engine::with_sinks_and_driver` spawns its telemetry task with
        // `tokio::spawn`, which needs a live reactor — build the runtime
        // and enter it before touching the engine (a `#[tokio::test]` gets
        // this for free; a plain `#[test]` has to do it explicitly).
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("serve runtime");
        let _guard = runtime.enter();

        let (engine, driver) = Engine::with_sinks_and_driver(
            Box::new(scheduler),
            Arc::new(NullSink),
            Arc::new(SystemClock),
        );
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
