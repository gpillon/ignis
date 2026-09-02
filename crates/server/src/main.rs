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
//!   template (its rendered `content` is not natural text).

use std::sync::Arc;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::{
    mock::MockCompute,
    ConcreteScheduler, SchedulerConfig,
};
use ignis_server::{
    engine::Engine,
    template::SimpleTemplateProvider,
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

    // The template seam (artifact-02 wiring, GitHub #7 follow-up): the
    // `.ninfer` container named by `IGNIS_ARTIFACT` carries the real
    // tokenizer + chat template. Without a usable artifact the server
    // falls back to the built-in placeholder (its rendered `content` is
    // the token id-space, not natural text).
    let artifact = env("IGNIS_ARTIFACT", "");
    let engine = Engine::new(Box::new(scheduler));
    let server = if artifact.is_empty() {
        eprintln!("ignis-server: no artifact (set IGNIS_ARTIFACT) — placeholder template (content is not natural text)");
        Server::new(engine, Box::new(SimpleTemplateProvider))
    } else {
        match Reader::open(std::path::Path::new(&artifact)).and_then(|reader| {
            FrontendSet::from_reader(&reader)
        }) {
            Ok(frontend) => {
                eprintln!("ignis-server: tokenizer + chat template from {artifact}");
                Server::with_artifact_template(engine, frontend)
            }
            Err(err) => {
                eprintln!("ignis-server: {artifact}: {err} — placeholder template (content is not natural text)");
                Server::new(engine, Box::new(SimpleTemplateProvider))
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