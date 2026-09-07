//! The G2 measurement instrument against a live `ignis-server` (P2-05,
//! GitHub #87): one TTFT cell, end to end, over real HTTP.
//!
//! The instrument's arithmetic — exact prompt lengths, per-sample prompt
//! distinctness, the median, the void rule, every gate refusal — is
//! CPU-unit-tested in `ignis-bench` (`ttft.rs`, `g2.rs`,
//! `tests/ttft_cell.rs`). What only a real engine can show is the part the
//! mock cannot fake: that a prompt generated to N post-template tokens with
//! *this* artifact's tokenizer and chat template is a prompt the real
//! server counts as N, and that the server's own usage figures come back
//! saying it computed all N. That is the smoke run this file is.
//!
//! It uses the production stack unchanged: the real GPU-backed scheduler
//! behind the real axum router, bound to a real localhost port, driven by
//! the same `ignis_bench::client::HttpEndpoint` that will drive the
//! reference at the gate (spec `02-real-prefill.md`: ignis and the
//! reference are measured by one instrument).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ignis_artifact::{FrontendSet, Reader};
use ignis_bench::client::HttpEndpoint;
use ignis_bench::ttft::{self, CellSpec, TtftConfig};
use ignis_core::gpu_profile;
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::{NullSink, SystemClock};
use ignis_server::Server;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// The smoke cell. Small enough to be a smoke run rather than a gate run
/// (the 8K and 32K cells are #88's job, on a free GPU) but large enough
/// that "the whole prompt was computed" means something.
const CELL_TOKENS: u32 = 1_024;

/// Samples for the smoke run. The statistic is not what is being checked
/// here — coldness is — so two samples plus the warmup is enough.
const CELL_SAMPLES: usize = 2;

/// A live server: the production router over the real GPU-backed
/// scheduler, served on a random localhost port.
///
/// The teardown order matters for the same reason it does in
/// `openai_http_gpu.rs` (GitHub #71): the model thread owns the
/// scheduler's GPU-resident state and only frees it when it exits, so the
/// serve runtime is dropped first (disconnecting every `Engine` clone) and
/// the driver thread is joined afterwards.
struct LiveServer {
    url: String,
    frontend: FrontendSet,
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

/// Start the live server, or `None` when the GPU profile says to skip.
fn live_server() -> Option<LiveServer> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    // A second frontend set: one is consumed by the server's template
    // provider, the other is the instrument's own tokenizer + chat
    // template. Both come from the same artifact, which is the point —
    // the cell's claimed length is measured against the engine's own.
    let instrument_frontend =
        FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));

    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, EngineShape::default()) {
        Ok(scheduler) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let (engine, driver) = Engine::with_sinks_and_driver(
        Box::new(scheduler),
        Arc::new(NullSink),
        Arc::new(SystemClock),
    );
    let app = Server::with_artifact_template(engine, frontend)
        .with_request_timeout(Duration::from_secs(600))
        .app();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("serve runtime");
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
    Some(LiveServer {
        url,
        frontend: instrument_frontend,
        runtime: Some(runtime),
        driver: Some(driver),
    })
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_ttft_cell_against_a_live_ignis_server_is_all_cold() {
    let Some(server) = live_server() else { return };
    let ep = HttpEndpoint::new(&server.url);
    let models = ep.list_models().unwrap_or_else(|e| panic!("GET /v1/models: {e}"));
    assert_eq!(models, vec![MODEL.to_string()]);

    // The acceptance criterion (GitHub #87) is a *record*, not a bare cell
    // measurement: `ttft::measure` is the same entry point a real gate run
    // uses, so this smoke run exercises the record's identity (session,
    // engine, artifact, profile, date) along with the cell itself.
    let cfg = TtftConfig {
        cells: vec![CellSpec { prompt_tokens: CELL_TOKENS, samples: CELL_SAMPLES }],
        max_tokens: ttft::DEFAULT_MAX_TOKENS,
        label: "ignis".into(),
        profile: "smoke test".into(),
        artifact: ARTIFACT.into(),
        session: "smoke".into(),
    };
    let record = ttft::measure(&ep, &server.frontend, MODEL.into(), server.url.clone(), &cfg);

    assert_eq!(record.session, "smoke");
    assert_eq!(record.engine, MODEL);
    assert_eq!(record.artifact, ARTIFACT);
    assert!(record.date.ends_with('Z'), "an RFC 3339 UTC date: {}", record.date);
    assert_eq!(record.cells.len(), 1);

    let cell = &record.cells[0];
    assert!(cell.error.is_none(), "the cell failed: {:?}", cell.error);
    assert_eq!(cell.samples.len(), CELL_SAMPLES);
    // The point of the run: every sample's prefix was cold, proved against
    // the engine's own computed-prefill-token count (ADR 0015) — not
    // assumed because the prompts looked different.
    assert!(
        cell.all_cold(),
        "every sample must be cold; void: {:?}",
        cell.void_samples()
    );
    for sample in &cell.samples {
        assert_eq!(
            sample.computed_prefill_tokens,
            Some(CELL_TOKENS),
            "the server must report computing the whole {CELL_TOKENS}-token prompt"
        );
    }
    let median = cell.median_ttft_ms.expect("a median");
    assert!(median > 0.0, "a real TTFT was measured: {median} ms");
    eprintln!("ttft {CELL_TOKENS} tokens: median {median:.1} ms over {CELL_SAMPLES} cold samples");
}
