//! ignis-server: the OpenAI-compatible HTTP entrypoint (localhost, no
//! auth).
//!
//! v1 surface (server-01, `docs/archive/ignis-v1.md` §2):
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
//! Configuration (ADR 0046, `ignis_server::config`): every field is declared
//! once and has a flag (`--<group>-<field>`), an env var
//! (`IGNIS_<GROUP>_<FIELD>`) and a config-file key (`<group>.<field>`), resolved
//! flag > env > config file > profile > default, with a family-scoped
//! variant (`--qwen38-…`, `--qwen38flashnext-…`) winning within each source.
//! `ignis-server help --fields` prints every one from the same table, so this
//! comment does not repeat it. `main` resolves once, builds the load from the
//! options fitted to the artifact's family, and serves `GET`/`PATCH
//! /v1/config` over the running configuration.

use std::sync::Arc;

use ignis_server::{
    config::{self, Config, ConfigOutcome},
    download,
    load::placeholder_model,
    thinking::{self, ThinkingDefaults},
    Served, Server,
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

/// The card `--profile auto` chooses by (ADR 0063): NVML's total memory of
/// device 0 on a CUDA build — a total it cannot read refuses the start — and
/// no card on a build without CUDA, where `auto` resolves as `none`. Read
/// here, never in the config module, which runs on machines with no NVML.
/// With it, what the card is and what the leaf was built for, which a start
/// checks before anything else (GitHub #457); none of it creates a context.
#[cfg(feature = "cuda")]
fn card_probe() -> config::profile::Card {
    use config::{arch, profile};
    let read = || -> Result<profile::CardReading, String> {
        let (_, total) = ignis_artifact::CudaDevice::nvml_memory(0).map_err(|error| error.to_string())?;
        let (name, (major, minor)) = ignis_artifact::CudaDevice::identity(0).map_err(|error| error.to_string())?;
        let built_for = arch::BuiltArchitectures::parse(&ignis_artifact::built_cuda_architectures())?;
        Ok(profile::CardReading { total, name, capability: arch::ComputeCapability::new(major, minor), built_for })
    };
    match read() {
        Ok(reading) => profile::Card::Read(reading),
        Err(error) => profile::Card::Unreadable(error),
    }
}

/// A build without CUDA has no card: `auto` resolves as `none`.
#[cfg(not(feature = "cuda"))]
fn card_probe() -> config::profile::Card {
    config::profile::Card::Absent
}

/// Refuse the start (ADR 0064, ADR 0079): the model it would load with no
/// artifact named is a catalog entry this card cannot load, by its class or
/// its generation, and no entry of its family that answers to it can — or
/// the one that answers holds an artifact of another family. Logged under
/// the refusal's code: `ignis.model.above_gpu_generation` when the generation
/// refuses it, `ignis.model.above_card_class` for the class alone.
fn refuse_on_card(logging_handle: &ignis_logging::LoggingHandle, refusal: &download::CardRefusal) -> ! {
    let card_class = refusal.class.as_ref().map(|class| class.needs.gb());
    let card = refusal.class.as_ref().map(|class| class.card.gb());
    let fix = "refusing to start: name a model this card loads (`--model-id`), or its artifact (`--model-artifact`)";
    match &refusal.generation {
        None => tracing::error!(name: "ignis.model.above_card_class", model = %refusal.model, card_class, card, error = %refusal, "{fix}"),
        Some(generation) => tracing::error!(
            name: "ignis.model.above_gpu_generation",
            model = %refusal.model,
            generation = generation.needs.as_str(),
            card_generation = generation.card.as_str(),
            compute_capability = %refusal.card.capability,
            card_class,
            card,
            error = %refusal,
            "{fix}"
        ),
    }
    exit_after_flush(logging_handle, 1);
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
    let config = match config::resolve_on(&args, |name| std::env::var(name).ok(), &config::file::RealFiles, &card_probe) {
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
        // `model download` / `model list` (spec model-download/02): the
        // artifact paths or the listing on stdout, progress on stderr — a
        // line at a time, never through the log, which shares stdout.
        Ok(ConfigOutcome::Model(command)) => {
            let progress: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(|line: &str| eprintln!("{line}"));
            let code = download::command::run(command, &mut std::io::stdout(), progress).await;
            exit_after_flush(&logging_handle, code);
        }
        Err(err) => {
            tracing::error!(name: "ignis.config.invalid", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
    };
    // GitHub #329: the stop signal is listened for from here, before
    // anything slow. A signal during the download or the load ends the
    // process at once (`shutdown::StopSignal`), one while serving stops the
    // serve, and a second one during that shutdown ends the process at once
    // too. The kernel drops a signal a PID 1 has no handler for, so with
    // the handler installed only once serving began, a container's SIGTERM
    // during the load did nothing.
    let logging_handle = Arc::new(logging_handle);
    let stop = match ignis_server::shutdown::StopSignal::install({
        let logging_handle = Arc::clone(&logging_handle);
        move || {
            logging_handle.flush(ignis_logging::SHUTDOWN_FLUSH_TIMEOUT);
            ignis_server::shutdown::exit_now(0)
        }
    }) {
        Ok(stop) => stop,
        Err(err) => {
            tracing::error!(name: "ignis.process.signal_failed", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
    };
    // Spec config-v2/02 AC 12: where the configuration came from, said
    // before anything is loaded, so it is in the log even when the load then
    // fails.
    config::log_source(&config);
    // GitHub #557: which models a vision or media option named for every
    // model applies to, when one is named.
    config::log_vision_options(&config);
    // The start options a load reads (`load::prepare_model`), which fits
    // them to the artifact's family once it names one (`Config::for_family`,
    // spec flash-next/04 and config-v2/01): a value scoped to that family is
    // the one the load runs with, and the engine shape (GitHub #87) is read
    // from the fitted config. A model switch loads every later model from
    // the same options, fitted to its own family.
    let start_options = config.clone();
    // The card the start read (ADR 0063, ADR 0064, ADR 0079): what the
    // catalog's entries are fitted to at start, for a switch, a request and a
    // listing — by the class of the total and the generation of the compute
    // capability `card_probe` read at resolution. `None` without a card,
    // which refuses nothing; an axis of nothing known refuses nothing on it.
    let card = config.card().cloned();
    let Config {
        model,
        model_named: _,
        bind,
        artifact,
        model_download,
        model_download_path,
        download_endpoint,
        download_token,
        download_catalog,
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
        describe_schema,
        loop_guard,
        prefill_chunk: _,
        prefill_chunk_solo: _,
        decode_share_percent: _,
        max_context: _,
        default_max_tokens,
        kv_format: _,
        kv_pool: _,
        vram: _,
        allow_expert_cache_below_floor: _,
        // Read through `EngineShape`, like the other load-shape knobs.
        pinned_expert_bytes: _,
        host_pool_bytes,
        // Read through `EngineShape`, like the other load-shape knobs.
        host_margin_bytes: _,
        // Read by the load, for its template's grammars.
        grammar: _,
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
        prefill_attention: _,
        prefill_linear: _,
        exception_weights: _,
        media: _,
        request_timeout_secs,
        stream_timeout_secs,
        timeout_reply,
        switch_drain_timeout_secs,
        idle_unload_secs,
        socket_idle_close_secs,
        allow_model_switch,
        known_models,
        ui,
        metrics,
        metrics_bind_named,
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

    // GitHub #329: the API listener is bound before anything slow and
    // answers through the start — `/health` 200, everything else 503
    // `server_not_ready` (`starting::Starting`) — so a liveness probe meets a
    // live process during the download and the load, never a refused
    // connection. The server takes it over once it serves.
    let starting = match tokio::net::TcpListener::bind(&bind).await.and_then(ignis_server::starting::Starting::answer) {
        Ok(starting) => starting,
        Err(err) => {
            tracing::error!(name: "ignis.server.failed", bind = %bind, error = %err, "server exited");
            exit_after_flush(&logging_handle, 1);
        }
    };
    tracing::info!(
        name: "ignis.server.listening",
        bind = %bind,
        address = %starting.local_addr(),
        "the API listener answers while the server starts: /health 200, /ready and /v1 503 until the model serves"
    );

    // The catalog (ADR 0047): the built-in entries, and the operator's file
    // when `--download-catalog` names one — read now, so a catalog that is
    // missing or malformed refuses the start whatever the start then needs.
    let catalog = match download::catalog::load(&config::file::RealFiles, download_catalog.as_deref()) {
        Ok(catalog) => Arc::new(catalog),
        Err(err) => {
            tracing::error!(name: "ignis.config.catalog_invalid", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
    };
    // Spec speech/02: the speech names against that catalog, an operator's
    // file among it; the config's own check had the built-in entries only.
    if let Err(why) = ignis_server::speech::check_names(&catalog, &start_options.settings().speech) {
        tracing::error!(name: "ignis.config.speech_invalid", error = %why, "refusing to start");
        exit_after_flush(&logging_handle, 1);
    }
    // Spec speech/08: `diarizer.model` against it too.
    if let Err(why) = ignis_server::speech::check_diarizer_name(&catalog, &start_options.settings().diarizer) {
        tracing::error!(name: "ignis.config.diarizer_invalid", error = %why, "refusing to start");
        exit_after_flush(&logging_handle, 1);
    }

    // ADR 0064, ADR 0079 (spec 27b-16gb/04 §6): with no artifact named, the
    // model is the catalog's, fitted to the card's class and generation before
    // it is looked for or fetched — on a card that cannot load its own entry
    // the entry that answers to it there, served under that entry's own id;
    // one this card cannot load refuses the start before anything is fetched.
    // A named artifact is the operator's word, which only the VRAM plan and
    // the route checks guard.
    let served = match artifact {
        Some(_) => model.clone(),
        None => match download::start_model(&catalog, &model, card.as_ref()) {
            Ok(served) => served,
            Err(refusal) => refuse_on_card(&logging_handle, &refusal),
        },
    };

    // Where the artifact comes from (GitHub #234, ADR 0033): the path the
    // operator named, one already under `--download-path`, one fetched
    // now, or none at all — which is the placeholder start this server has
    // always had, with a line saying which of the reasons it was.
    let source = download::artifact_source(
        &download::DownloadSettings {
            artifact: artifact.as_deref(),
            model: &served,
            enabled: model_download,
            dir: &model_download_path,
            // A binary built without `cuda` could not run the weights it
            // fetched, so it never fetches them: this is what keeps
            // `make mock`, `cargo test` and every CPU CI job off the network.
            supported: cfg!(feature = "cuda"),
            catalog: &catalog,
        },
        |path| path.exists(),
        |entry, path| download::ask_on_terminal(entry, path, &download_endpoint),
    );
    let artifact = match source {
        download::ArtifactSource::Use(path) => Some(path),
        download::ArtifactSource::Placeholder(reason) => {
            tracing::warn!(
                name: "ignis.model.placeholder_template",
                reason = reason.as_str(),
                model = %served,
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
            // GitHub #378: so does a models folder without room for what is
            // still missing plus download.free_margin, before a byte is
            // written, as the Server view's Download and `model download`.
            // `fetch_at_start` logs every refusal; the exit is ours.
            let token = download::bearer_token(&download_endpoint, download_token.as_ref(), std::env::var("HF_TOKEN").ok().as_deref());
            stop.starting("downloading");
            match download::fetch_at_start(&entry, &dir, &download_endpoint, token, start_options.settings().download.free_margin).await {
                Ok(path) => Some(path),
                Err(_) => exit_after_flush(&logging_handle, 1),
            }
        }
    };

    // The entry the card chose in `model`'s place is served under its
    // own id (ADR 0064 §2), as a switch to it would serve it — once its
    // artifact's header says it is of the configured id's family.
    let artifact_family = |path: &std::path::Path| ignis_server::loader::artifact_family(path).ok().flatten();
    let start_options = match download::served_start_options(start_options, &served, artifact.as_deref(), &catalog, card.as_ref(), &artifact_family) {
        Ok(options) => options,
        Err(download::StartRefusal::RefusedOnCard(refusal)) => refuse_on_card(&logging_handle, &refusal),
        Err(download::StartRefusal::Config(err)) => {
            tracing::error!(name: "ignis.config.invalid", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
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
        // On its own thread (GitHub #329), as a switch's load is: `stop`'s
        // watcher ends the process from a runtime worker whatever the load is
        // doing, and the start keeps answering on the listener meanwhile.
        stop.starting("loading");
        tracing::info!(
            name: "ignis.model.loading",
            artifact = %artifact_path.display(),
            "the start-up load begins: /ready answers 503 until the model serves"
        );
        let (options, path) = (start_options.clone(), artifact_path.clone());
        let loaded = tokio::task::spawn_blocking(move || {
            ignis_server::load::prepare_model(&options, &path).and_then(|prepared| {
                let running = prepared.config().clone();
                ignis_server::load::load_model(prepared).map(|loaded| (loaded, running))
            })
        })
        .await
        .unwrap_or_else(|panicked| std::panic::resume_unwind(panicked.into_panic()));
        stop.starting("starting");
        let (loaded, running) = match loaded {
            Ok(loaded) => loaded,
            Err(err) => {
                // GitHub #497: a catalog entry's artifact placed without its
                // sidecar names the command that fetches it.
                let fetch = download::sidecar_fetch_command(&catalog, artifact_path, download_catalog.as_deref());
                err.naming_the_fetch(fetch).log(artifact_path, "refusing to start");
                exit_after_flush(&logging_handle, 1);
            }
        };
        load_reservations = loaded.reservations;
        // GitHub #129: a loaded model is ready only after its first traversal.
        (Server::from_active(loaded.model).with_warm_up(), running)
    } else {
        // Why there is no artifact was said once, with its reason, where the
        // decision was made (`ignis.model.placeholder_template` above).
        // Under the id the start resolved (ADR 0064 §2), the one the catalog
        // and the switcher know it by.
        (Server::from_active(placeholder_model(&served, default_max_tokens)), start_options.clone())
    };
    let server = server
        .with_request_timeout(std::time::Duration::from_secs(request_timeout_secs as u64))
        .with_stream_timeout(std::time::Duration::from_secs(u64::from(stream_timeout_secs)))
        .with_timeout_reply(timeout_reply)
        .with_instruction_policy(instruction_policy)
        .with_idle_unload(std::time::Duration::from_secs(u64::from(idle_unload_secs)))
        .with_socket_idle_close(std::time::Duration::from_secs(u64::from(socket_idle_close_secs)))
        .with_image(running.settings().image.clone())
        .with_speech_group(running.settings().speech.clone())
        .with_diarizer_group(running.settings().diarizer.clone());
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
    // The explicit switch by id (spec playground/05) names them whatever
    // `--switch-allow-implicit` says; only the implicit one is gated by it.
    let switcher = ignis_server::model_switch::Switcher::new(
        Arc::new(ignis_server::model_switch::ArtifactLoader::sharing(Arc::clone(&config_state))),
        std::time::Duration::from_secs(u64::from(switch_drain_timeout_secs)),
    );
    switcher.note_start(server.active().source.as_ref());
    let tables = switcher.tables(&known_models, allow_model_switch);
    tracing::info!(
        name: "ignis.config.model_switch",
        allow_model_switch,
        known_models = %tables.known.keys().cloned().collect::<Vec<_>>().join(","),
        "a request naming one of these models, other than the loaded one, switches the server to it"
    );
    // ADR 0067, spec speech/02: the speech model beside the active one. Its
    // files are looked for in the same catalog and directory; its executor is
    // `ignis-speech` on the CPU (spec speech/01), unavailable with its reason
    // on a CPU without AVX2.
    // The diarizer (spec speech/08) is the catalog's too, computed on a pool
    // of its own with the same executor's ISA: loaded by the first session
    // that asks for speakers, unloaded with the speech model.
    let speech_cpu = Arc::new(ignis_server::speech::cpu::CpuExecutor::new());
    let speech_models = ignis_server::speech::SpeechModels { catalog: Arc::clone(&catalog), dir: model_download_path.clone() };
    let speech = ignis_server::speech::Speech::new(speech_cpu.clone(), Some(speech_models.clone()))
        .with_diarizer(Arc::new(ignis_server::speech::diarizing::CatalogDiarizer::new(speech_models, speech_cpu)));
    // Spec model-download/02 §Known models: a catalog entry whose artifact is
    // under `--download-path` when a request names it is switchable too, so
    // a `model download` into this directory needs no restart.
    let catalog_models = ignis_server::model_switch::CatalogModels { catalog, dir: model_download_path.clone() };
    // GitHub #376 (ADR 0071): a catalog entry downloaded in the background
    // through `POST /v1/models/download`, from the endpoint and with the
    // token the start-time fetch uses, `HF_TOKEN` read now as it reads it.
    let downloads = ignis_server::download_http::Downloads::of(&config_state.current(), std::env::var("HF_TOKEN").ok().as_deref());
    let server = server
        .with_switcher(
            switcher
                .with_known_models(tables.known)
                .with_listed_models(tables.listed)
                .with_catalog(catalog_models, allow_model_switch)
                .with_card(card),
        )
        .with_config(config_state)
        .with_downloads(downloads)
        .with_speech(Arc::new(speech));

    // Spec image-model/05 §Prompt enhancement: an enhancer no switch reaches
    // leaves every enhanced request on its prompt as sent. A `PATCH` naming
    // one is refused; the start is not, and says so once.
    let enhancer = server.live().image.prompt_enhancer.clone();
    if let Some(reason) = ignis_server::images::enhance::unknown_enhancer(&server, &enhancer, &Default::default()) {
        tracing::warn!(
            name: "ignis.config.prompt_enhancer_unknown",
            enhancer = %enhancer,
            %reason,
            "prompt enhancement uses each prompt as sent until image.prompt_enhancer names a text model this server can switch to"
        );
    }

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
    // server already treats a missing EOS token or an unclean checksum. The
    // image model thinks nothing: a text model's load judges them.
    let thinking_defaults = ThinkingDefaults {
        enable_thinking: default_enable_thinking,
        reasoning_effort: default_reasoning_effort,
    };
    if server.active().engine().is_some() {
        if let Err(err) =
            thinking::validate_defaults(&thinking_defaults, &server.active().template.thinking_capabilities())
        {
            tracing::error!(name: "ignis.config.thinking_invalid", error = %err, "refusing to start");
            exit_after_flush(&logging_handle, 1);
        }
    }
    let server = server
        .with_thinking_defaults(default_enable_thinking, default_reasoning_effort)
        .with_thinking_budget(default_thinking_budget)
        .with_describe_schema(describe_schema)
        .with_loop_guard(loop_guard);

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
    // Prometheus metrics (GitHub #89, ADR 0017), on by default: with
    // `--server-metrics false`, neither the projection nor any route to it
    // exists.
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

    // The metrics listener (ADR 0017): its own address, bound before serving
    // like the API's, and never the one `--server-expose` tunnels. Metrics are
    // on by default, so the default address may be a second server's: that
    // costs this start its listener, never the start (`/ui/metrics` still
    // serves). An address the operator named is theirs, and refuses it.
    let metrics_listener = match &metrics {
        None => None,
        Some(metrics_bind) => match tokio::net::TcpListener::bind(metrics_bind).await {
            Ok(listener) => Some(listener),
            Err(err) if !metrics_bind_named => {
                tracing::warn!(
                    name: "ignis.server.metrics_unavailable",
                    bind = %metrics_bind,
                    error = %err,
                    "the metrics listener's default address could not be bound (another server on this host holds it?): serving without it; --server-metrics-bind names another address, --server-metrics false turns metrics off"
                );
                None
            }
            Err(err) => {
                tracing::error!(name: "ignis.server.failed", bind = %metrics_bind, error = %err, "metrics listener could not bind");
                exit_after_flush(&logging_handle, 1);
            }
        },
    };
    // Where the metrics listener is, for the start line: the port a `:0`
    // bind was given, `off` when metrics are off, or `unavailable` when the
    // default address was taken (`/ui/metrics` still serves).
    let metrics_at = match metrics_listener.as_ref().map(tokio::net::TcpListener::local_addr) {
        Some(Ok(local)) => local.to_string(),
        Some(Err(_)) => metrics.clone().unwrap_or_default(),
        None if metrics.is_some() => "unavailable".to_owned(),
        None => "off".to_owned(),
    };

    // `--server-expose` (ADR 0028): opened on the bound port, before serving, so a
    // tunnel that cannot open refuses the start instead of leaving a server
    // the operator believes is reachable.
    let exposure = match expose {
        None => None,
        Some(mode) => {
            match ignis_server::expose::start(mode, starting.local_addr()).await {
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

    // The start's answers end here: the listener, still bound, is the
    // server's from its first accept. From here a signal stops the serve,
    // not the process: taken before `started` is logged, so a signal can no
    // longer end a loaded process as if it were still starting.
    let listener = starting.finish().await;
    let stopped = stop.serving();
    tracing::info!(
        name: "ignis.process.started",
        // The id the load is served under (`config::Config::for_family`): a
        // Flash-Next artifact started without `--model-id` is its own.
        model = %server.active().model_id(),
        bind = %bind,
        api_key_required = auth,
        metrics = %metrics_at,
        // GitHub #216: the pinned host arena is locked in RAM from start
        // (ADR 0030), and until now no line anywhere said how large it is.
        kv_host_pool_bytes = host_pool_bytes,
        exposed = exposure.as_ref().map_or("no", |_| "yes"),
        "OpenAI API at /v1"
    );
    // The serve drains HTTP, then joins the model thread before returning
    // (`Server::serve_on_until`): `ignis-model` is never left detached. A
    // switch or a load still running is not waited for: it is returned, and
    // the process ends under it below.
    let serve_result = server.serve_on_until(listener, metrics_listener, stopped).await;

    // The tunnel outlives the drain above, so in-flight remote requests
    // finish through it; only then is it unregistered. Not with a load
    // abandoned: unregistering takes up to ~32 s (the vendored crate's 30 s
    // grace, which it does not let a caller shorten), past a pod's grace
    // period, and the process ends at once below. The tunnel is in-process
    // and dies with it; the edge drops the connection, and a quick tunnel's
    // URL is never reused. Kept, not dropped, until then: a drop starts the
    // unregistering on detached tasks, whose lines would follow `stopped`.
    let _tunnel = match exposure {
        Some(exposure) if !matches!(serve_result, Ok(Served::LoadAbandoned { .. })) => {
            if let Err(err) = exposure.shutdown().await {
                tracing::warn!(name: "ignis.expose.close_failed", error = %err, "tunnel did not close cleanly");
            }
            None
        }
        kept => kept,
    };

    // `ignis.process.stopped` MUST be the last event emitted (spec §28) —
    // whichever branch below runs, nothing after it logs anything. The
    // trailing `flush` (bounded, GitHub #80) gives the queued writer thread
    // a chance to actually hand these last lines to the sink before the
    // process exits; it never blocks past its own timeout even if the sink
    // is stuck. A second signal that got there first logged the last line
    // itself and is ending the process: nothing is left to do but let it.
    if !stop.ending() {
        std::future::pending::<()>().await;
    }
    match serve_result {
        Ok(Served::Drained) => {
            tracing::info!(name: "ignis.process.stopped", "graceful shutdown complete");
            logging_handle.flush(ignis_logging::SHUTDOWN_FLUSH_TIMEOUT);
        }
        // GitHub #329: the load has no cancel, and returning from `main`
        // would wait for it (the runtime's drop joins its blocking thread):
        // the process ends now, as it does for a stop during the start-up
        // load, which frees the GPU.
        Ok(Served::LoadAbandoned { load, phase }) => {
            ignis_server::shutdown::emit_stopped_abandoning(&load, phase);
            logging_handle.flush(ignis_logging::SHUTDOWN_FLUSH_TIMEOUT);
            ignis_server::shutdown::exit_now(0)
        }
        Err(err) => {
            tracing::error!(name: "ignis.server.failed", error = %err, "server exited");
            exit_after_flush(&logging_handle, 1);
        }
    }
}