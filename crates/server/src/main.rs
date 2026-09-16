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
//! - `IGNIS_REQUEST_TIMEOUT` / `--request-timeout` — how long a
//!   non-streaming completion waits before the handler gives up with a
//!   `504` (default 30 seconds, max 3600 — GitHub #95).
//! - `--ui` (flag only, no env var) — serve the Playground at `/ui/`
//!   (GitHub #163, ADR 0026); off by default.
//! - `--metrics` / `--metrics-bind <addr>` (flags only, no env var) — serve
//!   Prometheus metrics at `GET /metrics` on their own listener (default
//!   `127.0.0.1:9464`, no API key, never exposed), and at `/ui/metrics` with
//!   `--ui`, under the API key when one is set (GitHub #89, ADR 0017). Off by
//!   default, and off means neither the routes nor the projection exist.
//! - `IGNIS_API_KEY` / `--api-key` — when set, every `/v1` request must
//!   send `Authorization: Bearer <key>` or gets a `401`; unset (default)
//!   leaves the API open. `auto` generates a key at start and prints it to
//!   stdout — the only case a key is ever printed.
//! - `IGNIS_EXPOSE` / `--expose` — make the server reachable from outside
//!   this machine (ADR 0028). `cloudflare-quick` opens a Cloudflare quick
//!   tunnel once the listener is bound and prints its public URL on stdout.
//!   An exposed server always requires an API key: with none set, it
//!   behaves as `--api-key auto`.

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
    telemetry::SystemClock,
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
                kv_format = shape.kv_format.as_str(),
                kv_pool_bytes = shape.kv_pool_bytes,
                speculation = %shape.speculation.map_or_else(
                    || "off".to_owned(),
                    |s| format!("{} draft_tokens={}", s.backend().as_str(), s.draft_tokens())
                ),
                vision = %shape.vision.map_or_else(
                    || "off".to_owned(),
                    |v| format!("max_tokens={}", v.max_tokens())
                ),
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
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
        prefill_chunk: _,
        max_context,
        kv_format: _,
        kv_pool_bytes: _,
        host_pool_bytes: _,
        speculation: _,
        vision,
        media,
        request_timeout_secs,
        ui,
        metrics,
        api_key,
        expose,
    } = config;
    let api_key = match api_key {
        None => None,
        Some(ignis_server::config::ApiKeySetting::Fixed(key)) => Some(key),
        Some(ignis_server::config::ApiKeySetting::Generate) => match ignis_server::config::ApiKey::generate() {
            Ok(key) => {
                // Printed on purpose, and only for a generated key: nobody
                // else knows it. A plain line, not a log record — the
                // logger redacts credentials. `make start`/`dev-ui` pick it
                // out of the log (mk/windows/common.ps1).
                println!("ignis-server: generated API key: {}", key.as_str());
                Some(key)
            }
            Err(err) => {
                tracing::error!(name: "ignis.config.api_key_generate_failed", error = %err, "refusing to start");
                exit_after_flush(&logging_handle, 1);
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

        // GitHub #179: a `--vision` load prepares images with the artifact's
        // processor and acquires them before admission. A tokenizer whose
        // placeholder ids are not the model contract's is a refused start.
        let processor = match vision.map(|v| ignis_server::media::load_processor(&frontend, v, max_context)) {
            None => None,
            Some(Ok(processor)) => Some(processor),
            Some(Err(err)) => {
                tracing::error!(name: "ignis.vision.processor_invalid", error = %err, "refusing to start");
                exit_after_flush(&logging_handle, 1);
            }
        };
        let engine = Engine::with_clock(scheduler, Arc::new(SystemClock));
        let provider = ignis_server::artifact_template::ArtifactTemplateProvider::new(frontend);
        match processor {
            None => Server::new(engine, Box::new(provider)),
            Some(processor) => {
                let acquirer = ignis_server::media::MediaAcquirer::new(
                    Arc::new(processor.clone()),
                    processor.options().clone(),
                    ignis_server::media::MediaPolicy::new(media.allow_private_network, media.cache_bytes),
                );
                Server::new(engine, Box::new(provider.with_vision(processor))).with_media(Arc::new(acquirer))
            }
        }
    } else {
        tracing::warn!(
            name: "ignis.model.placeholder_template",
            "no artifact (set --artifact/IGNIS_ARTIFACT) — placeholder template (content is not natural text) and MockCompute"
        );
        let engine = Engine::with_clock(mock_scheduler(&model), Arc::new(SystemClock));
        Server::new(engine, Box::new(SimpleTemplateProvider))
    }
    .with_request_timeout(std::time::Duration::from_secs(request_timeout_secs as u64));

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

    // The Playground (GitHub #163, ADR 0026): whatever this binary embedded
    // — the frontend build, or nothing, which serves the fallback page.
    let server = if ui {
        if ignis_server::playground::EMBEDDED.is_empty() {
            tracing::warn!(
                name: "ignis.playground.not_built",
                "--ui set but this binary was built without web/dist — /ui/ serves the build instructions"
            );
        }
        server.with_playground(ignis_server::playground::EMBEDDED)
    } else {
        server
    };
    // Prometheus metrics (GitHub #89, ADR 0017): without `--metrics`, neither
    // the projection nor any route to it exists.
    let server = if metrics.is_some() { server.with_metrics() } else { server };
    let auth = api_key.is_some();
    let server = match api_key {
        Some(key) => server.with_api_key(key),
        None => server,
    };

    // The driver loop: the single task that advances the engine and routes
    // its per-request events into the request handlers' streams (the
    // server's `serve` spawns it; see `Server::serve`).
    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(listener) => listener,
        Err(err) => {
            tracing::error!(name: "ignis.server.failed", bind = %bind, error = %err, "server exited");
            exit_after_flush(&logging_handle, 1);
        }
    };

    // The metrics listener (ADR 0017): its own address, bound before serving
    // like the API's, and never the one `--expose` tunnels.
    let metrics_listener = match &metrics {
        None => None,
        Some(metrics_bind) => match tokio::net::TcpListener::bind(metrics_bind).await {
            Ok(listener) => Some(listener),
            Err(err) => {
                tracing::error!(name: "ignis.server.failed", bind = %metrics_bind, error = %err, "metrics listener could not bind");
                exit_after_flush(&logging_handle, 1);
            }
        },
    };

    // `--expose` (ADR 0028): opened on the bound port, before serving, so a
    // tunnel that cannot open refuses the start instead of leaving a server
    // the operator believes is reachable.
    let exposure = match expose {
        None => None,
        Some(mode) => {
            let opened = match listener.local_addr() {
                Ok(local) => ignis_server::expose::start(mode, local).await,
                Err(err) => Err(err.to_string()),
            };
            match opened {
                Ok(exposure) => {
                    // A plain stdout line like the generated key's, so the
                    // Makefile helpers can repeat it (mk/windows/common.ps1).
                    println!("ignis-server: public URL: {}", exposure.url());
                    if ui {
                        println!("ignis-server: Playground: {}/ui/", exposure.url());
                    }
                    tracing::info!(
                        name: "ignis.expose.opened",
                        mode = mode.as_str(),
                        url = %exposure.url(),
                        location = %exposure.location(),
                        "exposed beyond the bind address"
                    );
                    Some(exposure)
                }
                Err(err) => {
                    tracing::error!(
                        name: "ignis.expose.failed",
                        mode = mode.as_str(),
                        error = %err,
                        "refusing to start"
                    );
                    exit_after_flush(&logging_handle, 1);
                }
            }
        }
    };

    tracing::info!(
        name: "ignis.process.started",
        model = %model,
        bind = %bind,
        api_key_required = auth,
        metrics = metrics.as_deref().unwrap_or("off"),
        exposed = exposure.as_ref().map_or("no", |_| "yes"),
        "OpenAI API at /v1"
    );
    let serve_result = server.serve_on(listener, metrics_listener).await;

    // The tunnel outlives the drain above, so in-flight remote requests
    // finish through it; only then is it unregistered.
    if let Some(exposure) = exposure {
        if let Err(err) = exposure.shutdown().await {
            tracing::warn!(name: "ignis.expose.close_failed", error = %err, "tunnel did not close cleanly");
        }
    }

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