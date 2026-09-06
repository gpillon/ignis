//! ignis-server: the OpenAI-compatible HTTP entrypoint (localhost, no
//! auth).
//!
//! v1 surface (server-01, `docs/design/ignis-v1.md` §2):
//! - `GET /v1/models` — the loaded model.
//! - `POST /v1/chat/completions` — chat completions (streaming +
//!   non-streaming); requests route into the core scheduler and tokens
//!   stream back as they are generated.
//! - `POST /v1/responses` — the OpenAI responses API (non-streaming).
//!
//! The request → engine boundary is the [`TemplateProvider`] seam
//! (`template.rs`): v1 ships a deterministic built-in provider,
//! artifact-02's artifact-backed tokenizer replaces it through the same
//! constructor injection. The compute backend is injected through the
//! scheduler: with `IGNIS_ARTIFACT` set and the binary built with
//! `--features cuda`, the entrypoint drives the real GPU-backed model
//! (`ignis_server::runtime::cuda_scheduler`, GitHub #61 / P1-25); without
//! either, it drives the deterministic `MockCompute` (CPU-only, ADR 0006).
//!
//! Configuration (environment):
//! - `IGNIS_MODEL` — the loaded model id (default `qwen3.8-27b`; what
//!   `GET /v1/models` reports and what submissions must name).
//! - `IGNIS_BIND` — the bind address (default `127.0.0.1:8000`;
//!   localhost-only by design — no network exposure, no auth, v1).
//! - `IGNIS_ARTIFACT` — the `.ninfer` container path (the real tokenizer
//!   and chat template, artifact-02); unset = the built-in placeholder
//!   template (its rendered `content` is not natural text). A configured
//!   artifact is loaded through the verified loader path (server-03):
//!   its sidecar must be present and its checksum report clean, or the
//!   server refuses to start (no silent fallback to the placeholder).
//! - `IGNIS_TELEMETRY` — the telemetry JSONL sink path (server-02, design
//!   §5): one compact line per event; unset = stdout.

use std::sync::Arc;

use ignis_core::{
    mock::MockCompute,
    Compute, ConcreteScheduler, Scheduler, SchedulerConfig,
};
use ignis_server::{
    engine::Engine,
    loader,
    template::SimpleTemplateProvider,
    telemetry::{FileSink, StdoutSink, SystemClock, TelemetrySink},
    Server,
};

/// The default loaded-model id (the v1 specialization: Qwen 3.8-27B —
/// `CONTEXT.md`).
const DEFAULT_MODEL: &str = "qwen3.8-27b";

/// The default bind address: localhost, port 8000 (OpenAI convention).
const DEFAULT_BIND: &str = "127.0.0.1:8000";

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

/// The mock-backed scheduler (ADR 0006, CPU-only): used whenever no
/// artifact is configured, or the binary was not built with
/// `--features cuda` — the entrypoint never silently blocks startup on a
/// missing GPU backend.
fn mock_scheduler(model: &str) -> Box<dyn Scheduler> {
    let compute: Arc<dyn Compute> = Arc::new(MockCompute::new());
    Box::new(ConcreteScheduler::with_config(
        SchedulerConfig {
            model: model.into(),
            ..SchedulerConfig::default()
        },
        compute,
    ))
}

/// Build the real GPU-backed scheduler for `artifact_path` (GitHub #61 /
/// P1-25). Requires `generation_config.json` to carry `eos_token_id` — a
/// backend that can never stop on EOS would silently run every request to
/// its `max_tokens` cap, so a missing one is a load failure like the
/// checksum / sidecar checks above it, not a silent default.
#[cfg(feature = "cuda")]
fn cuda_scheduler(
    artifact_path: &std::path::Path,
    model: &str,
    frontend: &ignis_artifact::FrontendSet,
) -> Box<dyn Scheduler> {
    let eos = match frontend.eos_token_id() {
        Some(eos) => eos,
        None => {
            eprintln!(
                "ignis-server: {}: generation_config.json has no eos_token_id — refusing to start",
                artifact_path.display()
            );
            std::process::exit(1);
        }
    };
    match ignis_server::runtime::cuda_scheduler(artifact_path, model.into(), eos) {
        Ok(scheduler) => {
            eprintln!("ignis-server: {} loaded on the GPU (eos={eos})", artifact_path.display());
            Box::new(scheduler)
        }
        Err(err) => {
            eprintln!("ignis-server: {}: {err} — refusing to start", artifact_path.display());
            std::process::exit(1);
        }
    }
}

#[tokio::main]
async fn main() {
    let model = env("IGNIS_MODEL", DEFAULT_MODEL);
    let bind = env("IGNIS_BIND", DEFAULT_BIND);
    let artifact = env("IGNIS_ARTIFACT", "");

    // The telemetry sink (server-02, design §5): a JSONL file named by
    // `IGNIS_TELEMETRY`, or stdout by default. One compact line per event.
    let telemetry_sink: Arc<dyn TelemetrySink> = match env("IGNIS_TELEMETRY", "") {
        path if path.is_empty() => Arc::new(StdoutSink),
        path => match FileSink::open(&path) {
            Ok(file) => {
                eprintln!("ignis-server: telemetry → {path}");
                Arc::new(file)
            }
            Err(err) => {
                eprintln!("ignis-server: telemetry: {err} (falling back to stdout)");
                Arc::new(StdoutSink)
            }
        },
    };

    let server = if artifact.is_empty() {
        eprintln!("ignis-server: no artifact (set IGNIS_ARTIFACT) — placeholder template (content is not natural text) and MockCompute");
        let engine = Engine::with_sinks(mock_scheduler(&model), telemetry_sink, Arc::new(SystemClock));
        Server::new(engine, Box::new(SimpleTemplateProvider))
    } else {
        // The loader path (server-03, GitHub #21): the `.ninfer` container
        // named by `IGNIS_ARTIFACT` is loaded through the verified loader —
        // open the reader, load the sidecar (ADR 0002), verify the checksum
        // report, and only then extract the frontend set. A missing
        // sidecar or a report that is not clean is a load failure: serving
        // a broken artifact would silently degrade to the placeholder, so
        // the server refuses to start instead.
        let artifact_path = std::path::Path::new(&artifact);
        let sidecar = match loader::find_sidecar(artifact_path) {
            Ok(path) => path,
            Err(err) => {
                eprintln!("ignis-server: {artifact}: {err} — refusing to start");
                std::process::exit(1);
            }
        };
        let frontend = match loader::load_artifact(artifact_path, &sidecar) {
            Ok(frontend) => {
                eprintln!("ignis-server: {artifact} verified (checksum clean) — tokenizer + chat template loaded");
                frontend
            }
            Err(err) => {
                eprintln!("ignis-server: {artifact}: {err} — refusing to start");
                std::process::exit(1);
            }
        };

        #[cfg(feature = "cuda")]
        let scheduler = cuda_scheduler(artifact_path, &model, &frontend);
        #[cfg(not(feature = "cuda"))]
        let scheduler = {
            eprintln!("ignis-server: built without --features cuda — MockCompute despite IGNIS_ARTIFACT (the templated text is real, the completions are not)");
            mock_scheduler(&model)
        };

        let engine = Engine::with_sinks(scheduler, telemetry_sink, Arc::new(SystemClock));
        Server::with_artifact_template(engine, frontend)
    };

    // The driver loop: the single task that advances the engine and routes
    // its per-request events into the request handlers' streams (the
    // server's `serve` spawns it; see `Server::serve`).
    eprintln!("ignis-server: model {model} on http://{bind} (localhost, no auth; OpenAI API at /v1)");
    if let Err(err) = server.serve(bind).await {
        eprintln!("ignis-server: {err}");
        std::process::exit(1);
    }
}