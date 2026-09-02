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
//! scheduler: this entrypoint drives the deterministic mock (CPU-only,
//! ADR 0006) until the kernel-leaf `Compute` adapter lands.
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
    ConcreteScheduler, SchedulerConfig,
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

#[tokio::main]
async fn main() {
    let model = env("IGNIS_MODEL", DEFAULT_MODEL);
    let bind = env("IGNIS_BIND", DEFAULT_BIND);

    // The compute seam: the kernel-leaf `Compute` adapter (C ABI, ADR
    // 0001) plugs in here when it lands; until then the deterministic
    // mock drives the scheduler (CPU-only — ADR 0006, the GPU is held by
    // the reference runner).
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: model.clone(),
            ..SchedulerConfig::default()
        },
        compute,
    );

    // The loader path (server-03, GitHub #21): the `.ninfer` container
    // named by `IGNIS_ARTIFACT` is loaded through the verified loader —
    // open the reader, load the sidecar (ADR 0002), verify the checksum
    // report, and only then extract the frontend set. A missing sidecar
    // or a report that is not clean is a load failure: serving a broken
    // artifact would silently degrade to the placeholder, so the server
    // refuses to start instead.
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
    let engine = Engine::with_sinks(Box::new(scheduler), telemetry_sink, Arc::new(SystemClock));
    let server = if artifact.is_empty() {
        eprintln!("ignis-server: no artifact (set IGNIS_ARTIFACT) — placeholder template (content is not natural text)");
        Server::new(engine, Box::new(SimpleTemplateProvider))
    } else {
        let artifact_path = std::path::Path::new(&artifact);
        let sidecar = match loader::find_sidecar(artifact_path) {
            Ok(path) => path,
            Err(err) => {
                eprintln!("ignis-server: {artifact}: {err} — refusing to start");
                std::process::exit(1);
            }
        };
        match loader::load_artifact(artifact_path, &sidecar) {
            Ok(frontend) => {
                eprintln!("ignis-server: {artifact} verified (checksum clean) — tokenizer + chat template loaded");
                Server::with_artifact_template(engine, frontend)
            }
            Err(err) => {
                eprintln!("ignis-server: {artifact}: {err} — refusing to start");
                std::process::exit(1);
            }
        }
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