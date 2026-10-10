//! ignis-server: the OpenAI-compatible HTTP entrypoint (localhost, no
//! auth).
//!
//! v1 surface (server-01, `docs/design/ignis-v1.md` §2):
//! - `GET /v1/models` — the loaded model; `POST /v1/models/switch` replaces
//!   it without a restart (spec model-switch/01).
//! - `POST /v1/chat/completions` — chat completions (streaming +
//!   non-streaming); requests route into the core scheduler and tokens
//!   stream back as they are generated.
//! - `POST /v1/responses` — the OpenAI responses API (non-streaming).
//!
//! The request → engine boundary is the [`TemplateProvider`] seam
//! (`template.rs`): v1 ships a deterministic built-in provider,
//! artifact-02's artifact-backed tokenizer replaces it through the same
//! constructor injection. The compute backend is injected through the
//! scheduler: with `IGNIS_MODEL_ARTIFACT` set and the binary built with
//! `--features cuda`, the entrypoint drives the real GPU-backed model
//! (`ignis_server::runtime::cuda_scheduler`, GitHub #61 / P1-25); without
//! either, it drives the deterministic `MockCompute` (CPU-only, ADR 0006).
//!
//! Configuration (CLI flags mirror each env var one-to-one — GitHub #77;
//! run `ignis-server --help` for the full flag table. A flag overrides its
//! env var, which overrides the built-in default):
//! - `IGNIS_MODEL_ID` / `--model-id`, `-m` — the loaded model id (default
//!   `qwen3.8-27b`; what `GET /v1/models` reports and what submissions must
//!   name).
//! - `IGNIS_SERVER_BIND` / `--server-bind`, `-b` — the bind address (default
//!   `127.0.0.1:8000`; localhost-only by design — no network exposure, no
//!   auth, v1).
//! - `IGNIS_MODEL_ARTIFACT` / `--model-artifact`, `-a` — the `.ninfer` container path
//!   (the real tokenizer and chat template, artifact-02). A configured
//!   artifact is loaded through the verified loader path (server-03): it
//!   must exist, its sidecar must be present and its checksum report clean,
//!   or the server refuses to start (no silent fallback to the placeholder).
//!   Unset, the model is looked for under `--download-path` and
//!   fetched when it is not there (below).
//! - `IGNIS_DOWNLOAD_ENABLED` / `--download-enabled` / `--download-enabled false` —
//!   may a missing model be fetched (ADR 0033, GitHub #234)? On by default,
//!   and only ever consulted with `--model-artifact` unset. On a terminal the
//!   operator is asked first; without one (a container, a daemon) it
//!   downloads, since there is nobody to answer. A build without
//!   `--features cuda` never downloads: it could not run the weights.
//!   Off, or a model the registry does not know, is the built-in
//!   placeholder template (its rendered `content` is not natural text).
//! - `IGNIS_DOWNLOAD_PATH` / `--download-path` — where a fetched
//!   model lands, and where one fetched earlier is found (default
//!   `./models`). Flat: the artifact and its sidecar keep the names the
//!   repo publishes them under, so `hf download … --local-dir models` and
//!   this produce the same file.
//! - `IGNIS_MODEL_ENABLE_THINKING` / `--model-enable-thinking` — the server-wide default
//!   for `enable_thinking` (GitHub #68); `true` or `false`, default `true`.
//!   An unparseable value, or a `false` the loaded template cannot honour,
//!   refuses to start.
//! - `IGNIS_MODEL_PREFILL_CHUNK` / `--model-prefill-chunk` — the prefill chunk width
//!   in tokens (default 1024; a nonzero multiple of 128 — the reference's
//!   own alignment rule). GitHub #87.
//! - `IGNIS_MODEL_MAX_CONTEXT` / `--model-max-context` — the maximum per-sequence
//!   context in tokens (default 40960: a 32K prompt plus an 8K generation
//!   budget, so G2's largest cell is admissible without editing code). The
//!   paged-KV pool the leaf builds is derived from this value (never
//!   below it, so admission can never promise more pages than the leaf
//!   built) rather than being an independent flag.
//! - `IGNIS_MODEL_REASONING_EFFORT` / `--model-reasoning-effort` — the server-wide
//!   default `reasoning_effort`; unset means "let the template's own
//!   default apply". An unknown value, or one the loaded template does not
//!   support, refuses to start.
//! - `IGNIS_MODEL_THINKING_BUDGET` / `--model-thinking-budget` — the server-wide
//!   thinking budget, in reasoning tokens (default 32768, `off` for none;
//!   spec server/08). Configured with a tokenizer that yields no thinking
//!   close, it refuses to start.
//! - `IGNIS_SERVER_REQUEST_TIMEOUT` / `--server-request-timeout` — how long a
//!   non-streaming completion waits before the handler gives up with a
//!   `504` (default 30 seconds, max 3600 — GitHub #95).
//! - `IGNIS_SWITCH_DRAIN_TIMEOUT` / `--switch-drain-timeout` — how long a
//!   model switch lets the old model's running requests finish before it
//!   cancels them (default 30 seconds, max 3600, 0 cancels at once — spec
//!   model-switch/01). A switch loads its target on the flags the server
//!   started with, leaving off the ones only the other model takes.
//! - `IGNIS_SWITCH_ALLOW_IMPLICIT` / `--switch-allow-implicit <true|false>` and
//!   `IGNIS_SWITCH_KNOWN_MODELS` / `--switch-known-models <id>=<path>` (repeatable; the env
//!   var takes `;`-separated pairs) — a request whose `model` names another
//!   listed model switches the server to it and is then served on it (default
//!   on; spec model-switch/01 §Implicit switch). The model the server starts
//!   on is always listed; off, or unlisted, such a request is a `404
//!   model_not_found`, as it always was.
//! - `--server-ui` / `--server-ui false` / `IGNIS_SERVER_UI` — serve the Playground at `/ui/`
//!   (GitHub #163, ADR 0026); **on** by default. A binary built without
//!   `web/dist` serves the page that says how to build it, so the default
//!   costs a route and nothing else.
//! - `--server-metrics` / `--server-metrics-bind <addr>` (flags only, no env var) — serve
//!   Prometheus metrics at `GET /metrics` on their own listener (default
//!   `127.0.0.1:9464`, no API key, never exposed), and at `/ui/metrics`
//!   unless `--server-ui false`, under the API key when one is set (GitHub #89, ADR
//!   0017). Off by default, and off means neither the routes nor the
//!   projection exist.
//! - `IGNIS_SERVER_API_KEY` / `--server-api-key` — when set, every `/v1` request must
//!   send `Authorization: Bearer <key>` or gets a `401`; unset (default)
//!   leaves the API open. `auto` generates a key at start and prints it to
//!   stdout — the only case a key is ever printed.
//! - `IGNIS_SERVER_EXPOSE` / `--server-expose` — make the server reachable from outside
//!   this machine (ADR 0028). `cloudflare-quick` opens a Cloudflare quick
//!   tunnel once the listener is bound and prints its public URL on stdout.
//!   An exposed server always requires an API key: with none set, it
//!   behaves as `--server-api-key auto`.

use std::sync::Arc;

use ignis_server::{
    config::{self, Config, ConfigOutcome},
    download,
    engine::Engine,
    load::mock_scheduler,
    template::SimpleTemplateProvider,
    telemetry::SystemClock,
    thinking::{self, ThinkingDefaults},
    ActiveModel, Server,
};

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
    let config = match config::resolve_with(&args, |name| std::env::var(name).ok(), &config::file::RealFiles) {
        Ok(ConfigOutcome::Config(config)) => config,
        Ok(ConfigOutcome::Help(text)) => {
            println!("{text}");
            std::process::exit(0);
        }
        Ok(ConfigOutcome::Version(text)) | Ok(ConfigOutcome::Print(text)) => {
            print!("{}", if text.ends_with('\n') { text } else { format!("{text}\n") });
            std::process::exit(0);
        }
        // `config generate --out` / `config patch` (spec config-v2/02): the
        // one place a config file is written, resolution having checked it.
        Ok(ConfigOutcome::Write { path, contents }) => match std::fs::write(&path, contents) {
            Ok(()) => {
                println!("ignis-server: wrote {}", path.display());
                std::process::exit(0);
            }
            Err(err) => {
                tracing::error!(name: "ignis.config.write_failed", path = %path.display(), error = %err, "config file not written");
                exit_after_flush(&logging_handle, 1);
            }
        },
        Err(err) => {
            tracing::error!(name: "ignis.config.invalid", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
    };
    // Spec config-v2/02 AC 12: where the configuration came from, said
    // before anything is loaded, so it is in the log even when the load then
    // fails.
    config::log_source(&config);
    // The start options a load reads (`load::prepare_model`), which fits
    // them to the artifact's family once it names one (`Config::for_family`,
    // spec flash-next/04 and config-v2/01): a value scoped to that family is
    // the one the load runs with, and the engine shape (GitHub #87) is read
    // from the fitted config. A model switch loads every later model from
    // the same options, fitted to its own family.
    let start_options = config.clone();
    let Config {
        model,
        model_named: _,
        bind,
        artifact,
        model_download,
        model_download_path,
        // Read by the load, through `start_options`.
        ngram_cache: _,
        // GitHub #306: read through `EngineShape`, like the other load-shape
        // knobs.
        ngram_hot_bytes: _,
        // Spec vram-budget/03: the budget through `EngineShape`, the
        // location handed to the loader through `start_options`.
        kv_disk_bytes: _,
        kv_disk_location: _,
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
        thinking_budget: default_thinking_budget,
        prefill_chunk: _,
        decode_share_percent: _,
        max_context: _,
        default_max_tokens,
        kv_format: _,
        kv_pool: _,
        vram: _,
        allow_expert_cache_below_floor: _,
        host_pool_bytes,
        prompt_reuse: _,
        retained_device_slots: _,
        retained_host_slots: _,
        retained_host_named: _,
        retained_interactive_ttl_secs: _,
        instruction_policy,
        speculation: _,
        speculation_off: _,
        draft_rows: _,
        decode_lanes: _,
        // Read by the load, through `start_options`.
        vision: _,
        // GitHub #227: read through `EngineShape`, like the other load-shape
        // knobs.
        rope_scaling: _,
        media: _,
        request_timeout_secs,
        switch_drain_timeout_secs,
        allow_model_switch,
        known_models,
        ui,
        metrics,
        api_key,
        expose,
        // What the options were resolved from: the load fits them again.
        basis: _,
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
    // operator named, one already under `--download-path`, one fetched
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
    // placeholder path, which loads no model and plans no device memory, and
    // on a build without `cuda`, which plans none either.
    let mut load_reservations: Option<ignis_server::metrics::LoadReservations> = None;
    let (server, running) = if let Some(artifact_path) = &artifact {
        // The loader path (server-03, GitHub #21), shared with the model
        // switch (spec model-switch/01): the `.ninfer` container named by
        // `--model-artifact`/`IGNIS_MODEL_ARTIFACT` is verified — sidecar present,
        // checksum report clean — named, checked against the start options,
        // and only then loaded. Any refusal stops the start: serving a broken
        // artifact would silently degrade to the placeholder.
        // The running configuration is the start options fitted to the
        // artifact's family (spec config-v2/01): what `GET /v1/config` shows.
        let loaded = ignis_server::load::prepare_model(&start_options, artifact_path).and_then(|prepared| {
            let running = prepared.config().clone();
            ignis_server::load::load_model(prepared).map(|loaded| (loaded, running))
        });
        let (loaded, running) = match loaded {
            Ok(loaded) => loaded,
            Err(err) => {
                err.log(artifact_path, "refusing to start");
                exit_after_flush(&logging_handle, 1);
            }
        };
        load_reservations = loaded.reservations;
        // GitHub #129: a loaded model is ready only after its first traversal.
        (Server::from_active(loaded.model).with_warm_up(), running)
    } else {
        // Why there is no artifact was said once, with its reason, where the
        // decision was made (`ignis.model.placeholder_template` above).
        let (engine, driver) =
            Engine::with_clock_and_driver(mock_scheduler(&model, default_max_tokens), Arc::new(SystemClock));
        (Server::from_active(ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver)), start_options.clone())
    };
    let server = server
        .with_request_timeout(std::time::Duration::from_secs(request_timeout_secs as u64))
        .with_instruction_policy(instruction_policy);
    // Spec config-v2/02: the running configuration, shown and changed over
    // `GET`/`PATCH /v1/config` and written back to the config file in use;
    // the model loader shares it, so a later switch loads with a live change
    // and a failed reload falls back to the configuration still running.
    let config_state = Arc::new(ignis_server::config_http::ConfigState::new(
        running,
        Arc::new(ignis_server::config::file::RealFiles),
    ));
    // Spec model-switch/01: `POST /v1/models/switch` loads every later model
    // through the same path, on the same start options — and so does a
    // request naming a known model (§Implicit switch), the start model among
    // them once its load has said which id and file it is.
    let known = if allow_model_switch {
        ignis_server::model_switch::known_models(&known_models, server.active().source.as_ref())
    } else {
        Default::default()
    };
    tracing::info!(
        name: "ignis.config.model_switch",
        allow_model_switch,
        known_models = %known.keys().cloned().collect::<Vec<_>>().join(","),
        "a request naming one of these models, other than the loaded one, switches the server to it"
    );
    let server = server.with_switcher(
        ignis_server::model_switch::Switcher::new(
            Arc::new(ignis_server::model_switch::ArtifactLoader::sharing(Arc::clone(&config_state))),
            std::time::Duration::from_secs(u64::from(switch_drain_timeout_secs)),
        )
        .with_known_models(known),
    )
    .with_config(config_state);

    // GitHub #209: joining or gathering developer messages trades prefix
    // reuse for fewer system blocks; the operator is told once, at start.
    if instruction_policy.developer.rerenders_history() {
        tracing::warn!(
            name: "ignis.config.developer_policy_rerenders",
            developer_message_policy = instruction_policy.developer.as_str(),
            "re-renders history when a developer message arrives mid-conversation; prefix reuse is lost from that point"
        );
    }

    // GitHub #260, #263, #275, #278: how `/v1/decide` will answer a point,
    // a box and a locate on this load, said once before the first request
    // (and again by every model switch).
    ignis_server::decide::log_load_heads(&server.active());

    // A default the loaded template cannot honour is a refused start (a
    // model swap must not silently change behaviour), matching how the
    // server already treats a missing EOS token or an unclean checksum.
    let thinking_defaults = ThinkingDefaults {
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
    };
    if let Err(err) =
        thinking::validate_defaults(&thinking_defaults, &server.active().template.thinking_capabilities())
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
                "--server-ui set but this binary was built without web/dist — /ui/ serves the build instructions"
            );
        }
        server.with_playground(ignis_server::playground::EMBEDDED)
    } else {
        server
    };
    // Prometheus metrics (GitHub #89, ADR 0017): without `--server-metrics`, neither
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
    // like the API's, and never the one `--server-expose` tunnels.
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

    // `--server-expose` (ADR 0028): opened on the bound port, before serving, so a
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
        // The id the load is served under (`config::served_model_for`): a
        // Flash-Next artifact started without `--model-id` is its own.
        model = %server.active().engine.model_id(),
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