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
//! interval-counter / request-lifecycle JSONL stream behind
//! `IGNIS_TELEMETRY`/`--telemetry`, GitHub #77). The two do not merge here.
//!
//! [`init`] is the one production entrypoint: resolves [`config::LogConfig`]
//! from `IGNIS_LOG_FORMAT`/`IGNIS_LOG_LEVEL` (`auto` format picks pretty on
//! an interactive stdout, JSON otherwise) and installs the matching layer as
//! the global default subscriber. Nothing in this crate migrates an
//! existing `println!`/`eprintln!` call site — that is Phase 2 (GitHub #79).

pub mod config;
pub mod json_layer;
pub mod pretty_layer;
pub mod record;
pub mod sink;

use std::io::IsTerminal;
use std::sync::Arc;

use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::Registry;

pub use config::{LogConfig, LogConfigError, LogConfigOverride, LogFormat, LogLevel};
pub use json_layer::JsonLayer;
pub use pretty_layer::PrettyLayer;
pub use record::LogRecord;
pub use sink::{LineSink, MemorySink, StdoutSink};

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
        LogFormat::Pretty => {
            // `auto` already resolved against the real TTY check in
            // `config::resolve`; an explicit `IGNIS_LOG_FORMAT=pretty` in a
            // non-interactive pipe still gets color off, matching "gated on
            // stdout being an interactive terminal" rather than on the
            // format choice itself.
            let color = std::io::stdout().is_terminal();
            PrettyLayer::new(sink, color).with_filter(level_filter).boxed()
        }
    };
    Registry::default().with(layer)
}

/// Resolve `env` (real process env in production, injected in tests) and
/// install the matching layer as the global default subscriber. Call once,
/// at the very top of `main`, before any other startup work — a minimal
/// bootstrap `eprintln!` fallback before this call is fine (spec §31) and
/// should stay as small as possible.
pub fn init(env: impl Fn(&str) -> Option<String>) -> Result<(), InitError> {
    let config = config::resolve(
        env,
        || std::io::stdout().is_terminal(),
        LogConfigOverride::default(),
    )
    .map_err(InitError::Config)?;
    let subscriber = build_subscriber(config, Arc::new(StdoutSink));
    tracing::subscriber::set_global_default(subscriber).map_err(|_| InitError::AlreadyInitialized)
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
            .with(PrettyLayer::new(pretty_sink.clone(), false));

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
    /// server startup. Not a G4 gate — nothing here touches the inference
    /// path; this only guards against something pathological (e.g.
    /// accidental blocking I/O) sneaking into the construction path.
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
