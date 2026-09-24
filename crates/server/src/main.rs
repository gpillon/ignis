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
//!   (the real tokenizer and chat template, artifact-02). A configured
//!   artifact is loaded through the verified loader path (server-03): it
//!   must exist, its sidecar must be present and its checksum report clean,
//!   or the server refuses to start (no silent fallback to the placeholder).
//!   Unset, the model is looked for under `--model-download-path` and
//!   fetched when it is not there (below).
//! - `IGNIS_MODEL_DOWNLOAD` / `--model-download` / `--no-model-download` —
//!   may a missing model be fetched (ADR 0033, GitHub #234)? On by default,
//!   and only ever consulted with `--artifact` unset. On a terminal the
//!   operator is asked first; without one (a container, a daemon) it
//!   downloads, since there is nobody to answer. A build without
//!   `--features cuda` never downloads: it could not run the weights.
//!   Off, or a model the registry does not know, is the built-in
//!   placeholder template (its rendered `content` is not natural text).
//! - `IGNIS_MODEL_DOWNLOAD_PATH` / `--model-download-path` — where a fetched
//!   model lands, and where one fetched earlier is found (default
//!   `./models`). Flat: the artifact and its sidecar keep the names the
//!   repo publishes them under, so `hf download … --local-dir models` and
//!   this produce the same file.
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
//! - `IGNIS_THINKING_BUDGET` / `--thinking-budget` — the server-wide
//!   thinking budget, in reasoning tokens (default 8192, `off` for none;
//!   spec server/08). Configured with a tokenizer that yields no thinking
//!   close, it refuses to start.
//! - `IGNIS_REQUEST_TIMEOUT` / `--request-timeout` — how long a
//!   non-streaming completion waits before the handler gives up with a
//!   `504` (default 30 seconds, max 3600 — GitHub #95).
//! - `--ui` / `--no-ui` / `IGNIS_UI` — serve the Playground at `/ui/`
//!   (GitHub #163, ADR 0026); **on** by default. A binary built without
//!   `web/dist` serves the page that says how to build it, so the default
//!   costs a route and nothing else.
//! - `--metrics` / `--metrics-bind <addr>` (flags only, no env var) — serve
//!   Prometheus metrics at `GET /metrics` on their own listener (default
//!   `127.0.0.1:9464`, no API key, never exposed), and at `/ui/metrics`
//!   unless `--no-ui`, under the API key when one is set (GitHub #89, ADR
//!   0017). Off by default, and off means neither the routes nor the
//!   projection exist.
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
    download,
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
    vision_item_bound: Option<u64>,
    thinking_close: Option<std::sync::Arc<ignis_core::thinking_budget::ThinkingClose>>,
    logging_handle: &ignis_logging::LoggingHandle,
) -> (Box<dyn Scheduler>, ignis_server::metrics::LoadReservations) {
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
    // The encoder holds one item at a time, and the processor bounds an item
    // below the envelope (`ProcessorOptions::max_item_tokens`): the load
    // sizes the encoder's workspace for that.
    let shape = ignis_server::runtime::EngineShape {
        vision: shape.vision.map(|vision| match vision_item_bound {
            // Never above the processor's request budget, the u32 envelope.
            Some(bound) => vision.with_item_max_tokens(u32::try_from(bound).expect("an item bound is at most the envelope")),
            None => vision,
        }),
        ..shape
    };
    match ignis_server::runtime::cuda_scheduler_with_thinking_close(artifact_path, model.into(), eos, shape, thinking_close) {
        Ok((scheduler, reserved)) => {
            tracing::info!(
                name: "ignis.model.loaded",
                artifact = %artifact_path.display(),
                eos,
                prefill_chunk = shape.prefill_chunk,
                max_context = shape.max_context,
                kv_format = shape.kv_format.as_str(),
                speculation = %shape.speculation.map_or_else(
                    || "off".to_owned(),
                    |s| format!(
                        "{} draft_tokens={} draft_head={}",
                        s.backend().as_str(),
                        s.draft_tokens(),
                        s.proposal_head().as_str()
                    )
                ),
                vision = %shape.vision.map_or_else(
                    || "off".to_owned(),
                    |v| format!("max_tokens={} item_max_tokens={}", v.max_tokens(), v.item_max_tokens())
                ),
                "model loaded on the GPU"
            );
            (Box::new(scheduler), reserved)
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
        model_download,
        model_download_path,
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
        thinking_budget: default_thinking_budget,
        prefill_chunk: _,
        max_context,
        kv_format: _,
        kv_pool_bytes: _,
        vram: _,
        host_pool_bytes,
        prompt_reuse: _,
        retained_slots: _,
        retained_interactive_ttl_secs: _,
        instruction_policy,
        speculation: _,
        vision,
        // GitHub #227: read through `EngineShape` above, like the other
        // load-shape knobs.
        rope_scaling: _,
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

    // Where the artifact comes from (GitHub #234, ADR 0033): the path the
    // operator named, one already under `--model-download-path`, one fetched
    // now, or none at all — which is the placeholder start this server has
    // always had, with a line saying which of the reasons it was.
    let source = download::artifact_source(
        &download::DownloadSettings {
            artifact: artifact.as_deref(),
            model: &model,
            enabled: model_download,
            dir: &model_download_path,
            // A binary built without `cuda` could not run the weights it
            // fetched, so it never fetches them: this is what keeps
            // `make mock`, `cargo test` and every CPU CI job off the network.
            supported: cfg!(feature = "cuda"),
        },
        |path| path.exists(),
        download::ask_on_terminal,
    );
    let artifact = match source {
        download::ArtifactSource::Use(path) => Some(path),
        download::ArtifactSource::Placeholder(reason) => {
            tracing::warn!(
                name: "ignis.model.placeholder_template",
                reason = reason.as_str(),
                model = %model,
                download_path = %model_download_path.display(),
                "no artifact — placeholder template (content is not natural text) and MockCompute"
            );
            None
        }
        download::ArtifactSource::Download { entry, dir } => {
            // Said yes, or nobody was there to ask. A transfer that does not
            // end in a verified artifact refuses the start: coming up on the
            // mock instead would be exactly the silent degradation the loader
            // path below refuses for an unclean checksum.
            let downloader = match download::Downloader::huggingface() {
                Ok(downloader) => downloader,
                Err(err) => {
                    tracing::error!(name: "ignis.model.download_failed", error = %err, "refusing to start");
                    exit_after_flush(&logging_handle, 1);
                }
            };
            match downloader.fetch(entry, &dir).await {
                Ok(path) => Some(path),
                Err(err) => {
                    tracing::error!(
                        name: "ignis.model.download_failed",
                        model = entry.model,
                        repo = entry.repo,
                        error = %err,
                        "refusing to start"
                    );
                    exit_after_flush(&logging_handle, 1);
                }
            }
        }
    };

    // GitHub #216 (ADR 0030 §Observability): what the load's VRAM plan
    // reserved, kept past the load so `/metrics` can name it. `None` on the
    // placeholder path, which loads no model and plans no device memory.
    // `mut` only under `cuda`: the placeholder path never assigns it.
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut load_reservations: Option<ignis_server::metrics::LoadReservations> = None;
    let server = if let Some(artifact_path) = &artifact {
        // The loader path (server-03, GitHub #21): the `.ninfer` container
        // named by `--artifact`/`IGNIS_ARTIFACT` is loaded through the
        // verified loader — open the reader, load the sidecar (ADR 0002),
        // verify the checksum report, and only then extract the frontend
        // set. A missing sidecar or a report that is not clean is a load
        // failure: serving a broken artifact would silently degrade to the
        // placeholder, so the server refuses to start instead.
        //
        // A path that is not there at all gets its own line (GitHub #234):
        // the sidecar error below would otherwise name a file next to a file
        // that does not exist, and say nothing about the download that could
        // have produced it.
        if !artifact_path.exists() {
            tracing::error!(
                name: "ignis.artifact.missing",
                artifact = %artifact_path.display(),
                model = %model,
                "no such file — refusing to start; drop --artifact/IGNIS_ARTIFACT to fetch the model into --model-download-path instead"
            );
            exit_after_flush(&logging_handle, 1);
        }
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

        // The thinking budget's forced close (2026-09-24), in this model's
        // own tokens. A tokenizer that splits `</think>` leaves every budget
        // inert: with a default budget configured that is a refused start
        // (spec server/08), since the operator would believe one is active;
        // without one it is said once here rather than discovered per
        // request.
        let thinking_close = thinking::thinking_close(|text| {
            frontend.tokenizer().encode(text).map_err(|e| e.to_string())
        });
        if let Err(err) = thinking::check_default_budget_close(default_thinking_budget, &thinking_close) {
            tracing::error!(name: "ignis.config.thinking_budget_inert", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
        let thinking_close = match thinking_close {
            Ok(close) => Some(Arc::new(close)),
            Err(error) => {
                tracing::warn!(name: "ignis.model.thinking_close_unavailable", %error, "thinking budgets are inert");
                None
            }
        };
        // Only the GPU backend forces the close; the mock never reasons.
        #[cfg(not(feature = "cuda"))]
        let _ = thinking_close;

        // GitHub #179: a `--vision` load prepares images with the artifact's
        // processor and acquires them before admission. A tokenizer whose
        // placeholder ids are not the model contract's is a refused start.
        // Built before the load, which sizes the encoder for the processor's
        // item bound.
        let processor = match vision.map(|v| ignis_server::media::load_processor(&frontend, v, max_context)) {
            None => None,
            Some(Ok(processor)) => Some(processor),
            Some(Err(err)) => {
                tracing::error!(name: "ignis.vision.processor_invalid", error = %err, "refusing to start");
                exit_after_flush(&logging_handle, 1);
            }
        };

        #[cfg(feature = "cuda")]
        let scheduler = {
            let item_bound = processor.as_ref().map(|p| p.options().max_item_tokens());
            let (scheduler, reserved) = cuda_scheduler(
                artifact_path,
                &model,
                &frontend,
                engine_shape,
                item_bound,
                thinking_close,
                &logging_handle,
            );
            // GitHub #216: what the plan reserved leaves the load here, so
            // the exposition can name it. The placeholder path below builds
            // no plan and leaves this `None`.
            load_reservations = Some(reserved);
            scheduler
        };
        #[cfg(not(feature = "cuda"))]
        let scheduler = {
            tracing::warn!(
                name: "ignis.model.mock_compute",
                "built without --features cuda — MockCompute despite --artifact/IGNIS_ARTIFACT (the templated text is real, the completions are not)"
            );
            mock_scheduler(&model)
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
        // Why there is no artifact was said once, with its reason, where the
        // decision was made (`ignis.model.placeholder_template` above).
        let engine = Engine::with_clock(mock_scheduler(&model), Arc::new(SystemClock));
        Server::new(engine, Box::new(SimpleTemplateProvider))
    }
    .with_request_timeout(std::time::Duration::from_secs(request_timeout_secs as u64))
    .with_instruction_policy(instruction_policy);

    // GitHub #209: joining or gathering developer messages trades prefix
    // reuse for fewer system blocks; the operator is told once, at start.
    if instruction_policy.developer.rerenders_history() {
        tracing::warn!(
            name: "ignis.config.developer_policy_rerenders",
            developer_message_policy = instruction_policy.developer.as_str(),
            "re-renders history when a developer message arrives mid-conversation; prefix reuse is lost from that point"
        );
    }

    // GitHub #260, #263: how a `point` with no `method` will be answered, and
    // whether a `box` can be asked for `head`, said once at load and before
    // the first request. The heads are keyed to the artifact's content hash,
    // so a load nobody calibrated says "chain" here instead of being found
    // out from its answers — and a load without `--vision` says neither,
    // since it takes no image to point on.
    let methods = ignis_server::decide::load_methods(server.calibration, server.media.is_some());
    tracing::info!(
        name: "ignis.decide.pointing_head",
        point_method = methods.point,
        box_methods = methods.box_methods,
        box_default = methods.box_default,
        head = server.calibration.map(|calibration| calibration.head.to_string()),
        set_heads = server.calibration.and_then(|calibration| calibration.set).map(|set| set.heads.len()),
        artifact = %server.engine.artifact(),
        "{}",
        methods.summary
    );

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
    let server = server
        .with_thinking_defaults(default_enable_thinking, default_reasoning_effort)
        .with_thinking_budget(default_thinking_budget);

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
    let server = match load_reservations {
        Some(reserved) => server.with_load_reservations(reserved),
        None => server,
    };
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
        // GitHub #216: the pinned host arena is locked in RAM from start
        // (ADR 0030), and until now no line anywhere said how large it is.
        kv_host_pool_bytes = host_pool_bytes,
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