//! `ignis-logging`: the one canonical structured-event model the whole
//! application logs through (GitHub #78, ADR 0011), backed by `tracing` +
//! `tracing-subscriber`.
//!
//! Application code instruments with ordinary `tracing` macros
//! (`tracing::info!`, `warn!`, `error!`, `debug!`, `trace!`) — this crate
//! does not define a parallel macro surface or an intermediate event struct
//! call sites construct by hand. The canonical event *is* `tracing::Event` +
//! `Metadata` ([`record::LogRecord`], built once per event); [`JsonLayer`]
//! and [`PrettyLayer`] are the only things that observe and render it, so
//! changing which layer is active never changes what attributes/severity/
//! event_name exist, only how they're displayed.
//!
//! Call sites that want a stable, greppable name pass one explicitly:
//! `tracing::info!(name: "ignis.model.loaded", model = %id, "model ready")`
//! — `event_name` in the rendered record comes straight from
//! `Metadata::name()`, nothing here invents or rewrites it.
//!
//! Named `logging`, not `telemetry`: `ignis_server::telemetry` already owns
//! that name for an unrelated, pre-existing concern (the scheduler
//! interval-counter JSONL stream behind `IGNIS_TELEMETRY`/`--telemetry`,
//! GitHub #77). The two event models do not merge here — GitHub #108 only
//! moved telemetry's line-writing sink onto [`sink::LineSink`]/
//! [`sink::FileSink`]/[`sink::NullSink`] (see `sink.rs`'s module doc);
//! telemetry's facts, counters, and JSONL shape stay its own.
//!
//! [`init`] is the one production entrypoint: resolves [`config::LogConfig`]
//! from `IGNIS_LOG_FORMAT`/`IGNIS_LOG_LEVEL` (`auto` format picks pretty on
//! an interactive stdout, JSON otherwise) and installs the matching layer as
//! the global default subscriber. Nothing in this crate migrates an
//! existing `println!`/`eprintln!` call site — that is Phase 2 (GitHub #79).

pub mod config;
pub mod hotpath_lint;
pub mod json_layer;
pub mod pretty_layer;
pub mod queue;
pub mod record;
pub mod sink;
pub mod trace_context;

use std::io::IsTerminal;
use std::sync::Arc;
use std::time::Duration;

use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::Registry;

pub use config::{LogConfig, LogConfigError, LogConfigOverride, LogFormat, LogLevel};
pub use json_layer::JsonLayer;
pub use pretty_layer::PrettyLayer;
pub use queue::{QueueConfig, QueueWorkerGuard, QueuedSink};
pub use record::LogRecord;
pub use sink::{FileSink, LineSink, MemorySink, NullSink, StderrSink, StdoutSink};

/// [`init`] failed: either the env/format config was invalid, or a global
/// subscriber was already installed (each process may install exactly one —
/// `main` calling this more than once is a programming error, not a runtime
/// condition to recover from at every call site).
#[derive(Debug)]
pub enum InitError {
    Config(LogConfigError),
    AlreadyInitialized,
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::Config(err) => write!(f, "{err}"),
            InitError::AlreadyInitialized => {
                write!(f, "logging::init was already called once for this process")
            }
        }
    }
}

/// Build the (unboxed-as-one-type) subscriber for `config`, writing through
/// `sink` — a pure construction step, separated from [`init`]'s
/// process-global side effect so it can be built and timed in tests without
/// installing it.
fn build_subscriber(
    config: LogConfig,
    sink: Arc<dyn LineSink>,
) -> impl tracing::Subscriber + Send + Sync {
    let level_filter = LevelFilter::from_level(config.level.as_tracing_level());
    let layer: Box<dyn Layer<Registry> + Send + Sync> = match config.format {
        LogFormat::Json => JsonLayer::new(sink).with_filter(level_filter).boxed(),
        // `config.color` was already decided in `config::resolve` from the
        // same injected TTY check `format`'s `auto` branch uses — an
        // explicit `IGNIS_LOG_FORMAT=pretty` in a non-interactive pipe still
        // gets color off, matching "gated on stdout being an interactive
        // terminal" rather than on the format choice itself. This function
        // stays pure: no second, real `is_terminal()` probe here.
        LogFormat::Pretty => {
            // GitHub #81: reuse `IGNIS_LOG_LEVEL` as the "verbose" signal
            // spec §19 allows for a pretty debug mode showing trace/span
            // ids, rather than adding a second config surface — `Debug`/
            // `Trace` show them, `Info`/`Warn`/`Error` (the default) don't.
            let show_trace = config.level <= LogLevel::Debug;
            PrettyLayer::new(sink, config.color, show_trace).with_filter(level_filter).boxed()
        }
    };
    Registry::default().with(layer)
}

/// The shutdown flush budget every binary's `main` uses (GitHub #80, spec
/// §28's "bounded timeout"): shared here, once, rather than the same
/// `Duration::from_millis(500)` literal re-typed at every call site
/// (`ignis-server`'s several "refusing to start" exits, its graceful-serve
/// exit, and `vendor-ninfer`'s one exit) — a single number a reviewer can
/// find and change in one place. Long enough for a healthy sink to drain a
/// handful of lines, short enough that a stalled one never meaningfully
/// delays process exit.
pub const SHUTDOWN_FLUSH_TIMEOUT: Duration = Duration::from_millis(500);

/// The handle [`init`]/[`init_with_sink`] return: keeps the background
/// logging-writer thread alive ([`QueueWorkerGuard`], GitHub #80) and gives
/// the caller the one shutdown seam that matters — [`LoggingHandle::flush`]
/// — without exposing the queue's internals. Drop it only once, at the very
/// end of `main` (dropping it early lets the writer thread stop draining
/// while the process is still logging).
pub struct LoggingHandle {
    sink: Arc<QueuedSink>,
    _guard: QueueWorkerGuard,
}

impl std::fmt::Debug for LoggingHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoggingHandle").finish_non_exhaustive()
    }
}

impl LoggingHandle {
    /// Wait up to `timeout` for every INFO/WARN/ERROR event enqueued before
    /// this call to have reached the sink — the shutdown pattern spec §28
    /// describes: emit `ignis.process.stopping`, do any other pending
    /// cleanup, emit `ignis.process.stopped` last, then call this once more
    /// so the final event itself is confirmed flushed before the process
    /// exits. Never blocks past `timeout`, even against a stalled sink.
    pub fn flush(&self, timeout: Duration) -> bool {
        self.sink.flush(timeout)
    }
}

/// Resolve `env` (real process env in production, injected in tests) and
/// install the matching layer as the global default subscriber, writing
/// through a queued wrapper around `sink` (GitHub #80: event creation and
/// physical I/O are decoupled here, once, for every layer/format — neither
/// `JsonLayer` nor `PrettyLayer` nor `build_subscriber` needs to know
/// queueing exists). Call once, at the very top of `main`, before any other
/// startup work — a minimal bootstrap `eprintln!` fallback before this call
/// is fine (spec §31) and should stay as small as possible.
///
/// [`init`] is the production entrypoint for a long-running service (`ignis
/// serve` keeps its whole event stream on stdout, GitHub #79); a one-shot
/// command that reserves stdout for its result output calls this directly
/// with [`StderrSink`] instead — and, because logging is now asynchronous
/// for every sink, MUST call [`LoggingHandle::flush`] before exiting, or a
/// diagnostic emitted just before `main` returns can be lost (see
/// `crates/vendor/src/main.rs`'s `run`/`main` split).
pub fn init_with_sink(
    env: impl Fn(&str) -> Option<String>,
    sink: Arc<dyn LineSink>,
) -> Result<LoggingHandle, InitError> {
    let config = config::resolve(
        env,
        || std::io::stdout().is_terminal(),
        LogConfigOverride::default(),
    )
    .map_err(InitError::Config)?;
    let (queued, guard) = QueuedSink::new(sink, QueueConfig::default());
    let subscriber = build_subscriber(config, queued.clone());
    tracing::subscriber::set_global_default(subscriber).map_err(|_| InitError::AlreadyInitialized)?;
    Ok(LoggingHandle { sink: queued, _guard: guard })
}

/// [`init_with_sink`] with the production long-running-service sink
/// ([`StdoutSink`]).
pub fn init(env: impl Fn(&str) -> Option<String>) -> Result<LoggingHandle, InitError> {
    init_with_sink(env, Arc::new(StdoutSink))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::config::LogConfigOverride;
    use crate::sink::MemorySink;

    /// Testing Decisions: "pretty and JSON layers are driven from the same
    /// `tracing::Event`" — install both layers together and confirm the
    /// JSON side's event_name/severity/attributes match what the pretty
    /// side rendered, i.e. only rendering differs.
    #[test]
    fn json_and_pretty_layers_observe_the_same_event() {
        let json_sink = Arc::new(MemorySink::new());
        let pretty_sink = Arc::new(MemorySink::new());
        let subscriber = Registry::default()
            .with(JsonLayer::new(json_sink.clone()))
            .with(PrettyLayer::new(pretty_sink.clone(), false, false));

        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(name: "ignis.dual.check", attempt = 3i64, ok = false, "checked twice");
        });

        let json_line = json_sink.lines().remove(0);
        let pretty_line = pretty_sink.lines().remove(0);
        let json: serde_json::Value = serde_json::from_str(&json_line).expect("valid json");

        assert_eq!(json["event_name"], "ignis.dual.check");
        assert_eq!(json["severity_text"], "WARN");
        assert_eq!(json["body"], "checked twice");
        assert_eq!(json["attributes"]["attempt"], 3);
        assert_eq!(json["attributes"]["ok"], false);

        assert!(pretty_line.contains("ignis.dual.check"), "{pretty_line}");
        assert!(pretty_line.contains("WARN"), "{pretty_line}");
        assert!(pretty_line.contains("checked twice"), "{pretty_line}");
        assert!(pretty_line.contains("attempt=3"), "{pretty_line}");
        assert!(pretty_line.contains("ok=false"), "{pretty_line}");
    }

    /// CPU-only sanity timing check (Testing Decisions): resolving config
    /// and constructing the subscriber must not add meaningful latency to
    /// server startup. This is a deliberate proxy for `init()`'s own cost
    /// rather than timing `init()` directly: `init()` installs the
    /// process-wide global subscriber, which can only happen once per test
    /// binary, so it can't be called (let alone timed) more than once
    /// without colliding with every other test here. Everything `init()`
    /// does beyond `resolve` + `build_subscriber` is exactly one
    /// `tracing::subscriber::set_global_default` call — a fixed, one-time,
    /// non-looping cost this sanity check isn't trying to catch regressions
    /// in. Not a G4 gate — nothing here touches the inference path; this
    /// only guards against something pathological (e.g. accidental blocking
    /// I/O) sneaking into the construction path.
    #[test]
    fn resolving_config_and_building_the_subscriber_is_fast() {
        let start = Instant::now();
        let config = config::resolve(|_| None, || false, LogConfigOverride::default())
            .expect("resolve");
        let _subscriber = build_subscriber(config, Arc::new(MemorySink::new()));
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(50),
            "logging init construction took {elapsed:?}, expected well under 50ms"
        );
    }

    /// `build_subscriber` must actually route through both `Json` and
    /// `Pretty` branches (not just have the branches exist) — the dual-layer
    /// test above builds `JsonLayer`/`PrettyLayer` directly, which never
    /// exercises `build_subscriber`'s own `match config.format`.
    #[test]
    fn build_subscriber_routes_json_format_to_the_json_layer() {
        let sink = Arc::new(MemorySink::new());
        let config = LogConfig { format: LogFormat::Json, level: LogLevel::Info, color: false };
        let subscriber = build_subscriber(config, sink.clone());

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(name: "ignis.build_subscriber.json_route", "via json");
        });

        let line = sink.lines().remove(0);
        let json: serde_json::Value = serde_json::from_str(&line).expect("json layer emits valid json");
        assert_eq!(json["event_name"], "ignis.build_subscriber.json_route");
    }

    #[test]
    fn build_subscriber_routes_pretty_format_to_the_pretty_layer() {
        let sink = Arc::new(MemorySink::new());
        let config = LogConfig { format: LogFormat::Pretty, level: LogLevel::Info, color: false };
        let subscriber = build_subscriber(config, sink.clone());

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(name: "ignis.build_subscriber.pretty_route", "via pretty");
        });

        let line = sink.lines().remove(0);
        // The pretty layer's output is not JSON — this is what distinguishes
        // it from the json-route test above without duplicating
        // `pretty_layer.rs`'s own rendering assertions.
        assert!(
            serde_json::from_str::<serde_json::Value>(&line).is_err(),
            "pretty layer output should not itself be a JSON line: {line}"
        );
        assert!(line.contains("ignis.build_subscriber.pretty_route"), "{line}");
        assert!(line.contains("via pretty"), "{line}");
    }

    #[test]
    fn an_invalid_env_value_surfaces_as_a_config_error() {
        let err = init(|k| {
            if k == "IGNIS_LOG_FORMAT" {
                Some("nonsense".to_owned())
            } else {
                None
            }
        })
        .expect_err("must reject");
        assert!(matches!(err, InitError::Config(_)));
    }
}
