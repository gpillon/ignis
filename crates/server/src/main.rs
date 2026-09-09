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
//! Configuration (CLI flags mirror each env var one-to-one — GitHub #77;
//! run `ignis-server --help` for the full flag table. A flag overrides its
//! env var, which overrides the built-in default):
//! - `IGNIS_MODEL` / `--model`, `-m` — the loaded model id (default
//!   `qwen3.8-27b`; what `GET /v1/models` reports and what submissions must
//!   name).
//! - `IGNIS_BIND` / `--bind`, `-b` — the bind address (default
//!   `127.0.0.1:8000`; localhost-only by design — no network exposure, no
//!   auth, v1).
//! - `IGNIS_ARTIFACT` / `--artifact`, `-a` — the `.ninfer` container path
//!   (the real tokenizer and chat template, artifact-02); unset = the
//!   built-in placeholder template (its rendered `content` is not natural
//!   text). A configured artifact is loaded through the verified loader
//!   path (server-03): its sidecar must be present and its checksum report
//!   clean, or the server refuses to start (no silent fallback to the
//!   placeholder).
//! - `IGNIS_TELEMETRY` / `--telemetry`, `-t` — the telemetry JSONL sink path
//!   (server-02, design §5): one compact line per event; unset = stdout.
//! - `IGNIS_ENABLE_THINKING` / `--enable-thinking` — the server-wide default
//!   for `enable_thinking` (GitHub #68); `true` or `false`, default `true`.
//!   An unparseable value, or a `false` the loaded template cannot honour,
//!   refuses to start.
//! - `IGNIS_PREFILL_CHUNK` / `--prefill-chunk` — the prefill chunk width
//!   in tokens (default 1024; a nonzero multiple of 128 — the reference's
//!   own alignment rule). GitHub #87.
//! - `IGNIS_MAX_CONTEXT` / `--max-context` — the maximum per-sequence
//!   context in tokens (default 40960: a 32K prompt plus an 8K generation
//!   budget, so G2's largest cell is admissible without editing code). The
//!   paged-KV pool the leaf builds is derived from this value (never
//!   below it, so admission can never promise more pages than the leaf
//!   built) rather than being an independent flag.
//! - `IGNIS_REASONING_EFFORT` / `--reasoning-effort` — the server-wide
//!   default `reasoning_effort`; unset means "let the template's own
//!   default apply". An unknown value, or one the loaded template does not
//!   support, refuses to start.

use std::sync::Arc;

use ignis_core::{
    mock::MockCompute,
    Compute, ConcreteScheduler, Scheduler, SchedulerConfig,
};
use ignis_server::{
    config::{self, Config, ConfigOutcome},
    engine::Engine,
    loader,
    template::SimpleTemplateProvider,
    telemetry::{FileSink, StdoutSink, SystemClock, TelemetrySink},
    thinking::{self, ThinkingDefaults},
    Server,
};

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

/// Flush pending logging before an immediate exit (GitHub #80): every
/// "refusing to start" path calls this instead of a bare
/// `std::process::exit`. Logging is asynchronous now for every sink
/// (`ignis_logging::init`'s queued writer thread) — without this, the very
/// diagnostic that explains *why* the process is exiting could still be
/// sitting in the queue when the process terminates, and be lost.
fn exit_after_flush(logging_handle: &ignis_logging::LoggingHandle, code: i32) -> ! {
    logging_handle.flush(ignis_logging::SHUTDOWN_FLUSH_TIMEOUT);
    std::process::exit(code);
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
    shape: ignis_server::runtime::EngineShape,
    logging_handle: &ignis_logging::LoggingHandle,
) -> Box<dyn Scheduler> {
    let eos = match frontend.eos_token_id() {
        Some(eos) => eos,
        None => {
            tracing::error!(
                name: "ignis.model.eos_missing",
                artifact = %artifact_path.display(),
                "generation_config.json has no eos_token_id — refusing to start"
            );
            exit_after_flush(logging_handle, 1);
        }
    };
    match ignis_server::runtime::cuda_scheduler(artifact_path, model.into(), eos, shape) {
        Ok(scheduler) => {
            tracing::info!(
                name: "ignis.model.loaded",
                artifact = %artifact_path.display(),
                eos,
                prefill_chunk = shape.prefill_chunk,
                max_context = shape.max_context,
                kv_pool_tokens = shape.kv_pool_tokens,
                "model loaded on the GPU"
            );
            Box::new(scheduler)
        }
        Err(err) => {
            tracing::error!(
                name: "ignis.model.load_failed",
                artifact = %artifact_path.display(),
                error = %err,
                "refusing to start"
            );
            exit_after_flush(logging_handle, 1);
        }
    }
}

#[tokio::main]
async fn main() {
    // The canonical structured-logging system (GitHub #78, ADR 0011) — first
    // thing `main` does, before args/config, so the earliest possible
    // startup messages already go through it rather than a bootstrap
    // `eprintln!`. Every diagnostic site below is migrated onto it (GitHub
    // #79); only this call's own failure predates a subscriber existing, so
    // it keeps a minimal bootstrap `eprintln!`.
    // Kept alive for the rest of `main` (GitHub #80): dropping it early would
    // signal the background logging-writer thread to stop while the process
    // is still emitting events. `logging_handle.flush(..)` below, right
    // before the final `ignis.process.stopped` event, is the shutdown seam
    // that actually matters — see spec §27/28.
    let logging_handle = match ignis_logging::init(|name| std::env::var(name).ok()) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("ignis-server: logging: {err} — refusing to start");
            std::process::exit(1);
        }
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    let config = match config::resolve(&args, |name| std::env::var(name).ok()) {
        Ok(ConfigOutcome::Config(config)) => config,
        Ok(ConfigOutcome::Help(text)) => {
            println!("{text}");
            std::process::exit(0);
        }
        Ok(ConfigOutcome::Version(text)) => {
            println!("{text}");
            std::process::exit(0);
        }
        Err(err) => {
            tracing::error!(name: "ignis.config.invalid", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
    };
    // The engine shape (GitHub #87) — already validated by `config::resolve`
    // above, so nothing below this point can fail on an unaligned chunk
    // width or a pool that cannot serve the configured context.
    #[cfg(feature = "cuda")]
    let engine_shape = ignis_server::runtime::EngineShape::from(&config);
    let Config {
        model,
        bind,
        artifact,
        telemetry,
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
        prefill_chunk: _,
        max_context: _,
        kv_pool_tokens: _,
    } = config;

    // The telemetry sink (server-02, design §5): a JSONL file named by
    // `--telemetry`/`IGNIS_TELEMETRY`, or stdout by default. One compact
    // line per event.
    let telemetry_sink: Arc<dyn TelemetrySink> = match &telemetry {
        None => Arc::new(StdoutSink),
        Some(path) => match FileSink::open(path) {
            Ok(file) => {
                tracing::info!(
                    name: "ignis.telemetry.sink_selected",
                    path = %path.display(),
                    "telemetry sink selected"
                );
                Arc::new(file)
            }
            Err(err) => {
                tracing::warn!(
                    name: "ignis.telemetry.sink_failed",
                    path = %path.display(),
                    error = %err,
                    "falling back to stdout"
                );
                Arc::new(StdoutSink)
            }
        },
    };

    let server = if let Some(artifact_path) = &artifact {
        // The loader path (server-03, GitHub #21): the `.ninfer` container
        // named by `--artifact`/`IGNIS_ARTIFACT` is loaded through the
        // verified loader — open the reader, load the sidecar (ADR 0002),
        // verify the checksum report, and only then extract the frontend
        // set. A missing sidecar or a report that is not clean is a load
        // failure: serving a broken artifact would silently degrade to the
        // placeholder, so the server refuses to start instead.
        let sidecar = match loader::find_sidecar(artifact_path) {
            Ok(path) => path,
            Err(err) => {
                tracing::error!(
                    name: "ignis.artifact.sidecar_missing",
                    artifact = %artifact_path.display(),
                    error = %err,
                    "refusing to start"
                );
                exit_after_flush(&logging_handle, 1);
            }
        };
        let frontend = match loader::load_artifact(artifact_path, &sidecar) {
            Ok(frontend) => {
                tracing::info!(
                    name: "ignis.artifact.verified",
                    artifact = %artifact_path.display(),
                    "checksum clean — tokenizer + chat template loaded"
                );
                frontend
            }
            Err(err) => {
                tracing::error!(
                    name: "ignis.artifact.load_failed",
                    artifact = %artifact_path.display(),
                    error = %err,
                    "refusing to start"
                );
                exit_after_flush(&logging_handle, 1);
            }
        };

        #[cfg(feature = "cuda")]
        let scheduler = cuda_scheduler(artifact_path, &model, &frontend, engine_shape, &logging_handle);
        #[cfg(not(feature = "cuda"))]
        let scheduler = {
            tracing::warn!(
                name: "ignis.model.mock_compute",
                "built without --features cuda — MockCompute despite --artifact/IGNIS_ARTIFACT (the templated text is real, the completions are not)"
            );
            mock_scheduler(&model)
        };

        let engine = Engine::with_sinks(scheduler, telemetry_sink, Arc::new(SystemClock));
        Server::with_artifact_template(engine, frontend)
    } else {
        tracing::warn!(
            name: "ignis.model.placeholder_template",
            "no artifact (set --artifact/IGNIS_ARTIFACT) — placeholder template (content is not natural text) and MockCompute"
        );
        let engine = Engine::with_sinks(mock_scheduler(&model), telemetry_sink, Arc::new(SystemClock));
        Server::new(engine, Box::new(SimpleTemplateProvider))
    };

    // A default the loaded template cannot honour is a refused start (a
    // model swap must not silently change behaviour), matching how the
    // server already treats a missing EOS token or an unclean checksum.
    let thinking_defaults = ThinkingDefaults {
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
    };
    if let Err(err) =
        thinking::validate_defaults(&thinking_defaults, &server.template.thinking_capabilities())
    {
        tracing::error!(name: "ignis.config.thinking_invalid", error = %err, "refusing to start");
        exit_after_flush(&logging_handle, 1);
    }
    let server = server.with_thinking_defaults(default_enable_thinking, default_reasoning_effort);

    // The driver loop: the single task that advances the engine and routes
    // its per-request events into the request handlers' streams (the
    // server's `serve` spawns it; see `Server::serve`).
    tracing::info!(
        name: "ignis.process.started",
        model = %model,
        bind = %bind,
        "localhost, no auth; OpenAI API at /v1"
    );
    let serve_result = server.serve(bind).await;

    // `ignis.process.stopped` MUST be the last event emitted (spec §28) —
    // whichever branch below runs, nothing after it logs anything. The
    // trailing `flush` (bounded, GitHub #80) gives the queued writer thread
    // a chance to actually hand these last lines to the sink before the
    // process exits; it never blocks past its own timeout even if the sink
    // is stuck.
    match serve_result {
        Ok(()) => {
            tracing::info!(name: "ignis.process.stopped", "graceful shutdown complete");
            logging_handle.flush(ignis_logging::SHUTDOWN_FLUSH_TIMEOUT);
        }
        Err(err) => {
            tracing::error!(name: "ignis.server.failed", error = %err, "server exited");
            exit_after_flush(&logging_handle, 1);
        }
    }
}