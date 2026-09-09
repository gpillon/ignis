//! `IGNIS_LOG_FORMAT` / `IGNIS_LOG_LEVEL` resolution (GitHub #78).
//!
//! [`resolve`] is the one seam for this crate's config: pure (no
//! `std::env`, no TTY syscall — both are injected), so format/level
//! precedence and the `auto` TTY-detection branch are covered by fast unit
//! tests instead of only end-to-end runs, mirroring
//! `ignis_server::config::resolve`'s shape. [`init`] (`lib.rs`) is the only
//! caller that plugs in the real environment and the real TTY check.

/// `IGNIS_LOG_FORMAT`: which layer renders events. `Auto` is the default —
/// resolved to `Pretty`/`Json` by [`resolve`] via the injected TTY check,
/// same as every other value here explicit input always wins over
/// autodetection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormatSetting {
    Auto,
    Pretty,
    Json,
}

/// The format a [`LogConfig`] actually renders with, after `auto` has been
/// resolved against the TTY check. Only two layers exist (`json_layer.rs`,
/// `pretty_layer.rs`); `auto` is never a runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Pretty,
    Json,
}

/// `IGNIS_LOG_LEVEL`: the minimum severity emitted. Maps 1:1 onto
/// [`tracing::Level`] — kept as its own type (rather than re-exporting
/// `tracing::Level` directly) so parsing/validation stays in this crate's
/// seam instead of leaking `tracing`'s own `FromStr` error type into
/// [`LogConfigError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_tracing_level(self) -> tracing::Level {
        match self {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

/// The resolved config `init` needs: a concrete format (never `Auto` — see
/// [`LogFormat`]), a minimum level, and whether the real stdout is an
/// interactive terminal (`color`) — decided once here from the same
/// injected `is_terminal` check `format`'s `auto` resolution uses, so
/// [`crate::build_subscriber`] never re-probes the real TTY itself. An
/// explicit `IGNIS_LOG_FORMAT=pretty` on a non-interactive pipe still gets
/// `color: false` — color tracks the real terminal, not the chosen format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogConfig {
    pub format: LogFormat,
    pub level: LogLevel,
    pub color: bool,
}

/// A future CLI flag's override slot (e.g. `--log-format`/`--log-level`,
/// once a CLI parser exists — GitHub #77 or a later issue). Empty for now:
/// [`resolve`] already takes it so wiring a real flag in later does not
/// change this function's signature or its existing tests, matching the
/// issue's explicit "config resolution function is shaped so a future CLI
/// flag can override it without rework".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogConfigOverride {}

/// An unrecognized `IGNIS_LOG_FORMAT`/`IGNIS_LOG_LEVEL` value — the message
/// is suitable for a bootstrap `eprintln!` before `init` gives up and the
/// caller falls back to its pre-`logging::init` bootstrap path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogConfigError(pub String);

impl std::fmt::Display for LogConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Resolve `env` (injected so tests never touch the real process
/// environment) and `is_terminal` (injected so `auto` mode is testable
/// without a real TTY) into a [`LogConfig`]. `override_` is currently
/// always a no-op (see [`LogConfigOverride`]).
pub fn resolve(
    env: impl Fn(&str) -> Option<String>,
    is_terminal: impl Fn() -> bool,
    _override: LogConfigOverride,
) -> Result<LogConfig, LogConfigError> {
    let is_tty = is_terminal();

    let format_setting = match env("IGNIS_LOG_FORMAT") {
        None => LogFormatSetting::Auto,
        Some(raw) => parse_format(&raw)?,
    };
    let format = match format_setting {
        LogFormatSetting::Auto => {
            if is_tty {
                LogFormat::Pretty
            } else {
                LogFormat::Json
            }
        }
        LogFormatSetting::Pretty => LogFormat::Pretty,
        LogFormatSetting::Json => LogFormat::Json,
    };

    let level = match env("IGNIS_LOG_LEVEL") {
        None => LogLevel::Info,
        Some(raw) => parse_level(&raw)?,
    };

    Ok(LogConfig { format, level, color: is_tty })
}

fn parse_format(raw: &str) -> Result<LogFormatSetting, LogConfigError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(LogFormatSetting::Auto),
        "pretty" => Ok(LogFormatSetting::Pretty),
        "json" => Ok(LogFormatSetting::Json),
        other => Err(LogConfigError(format!(
            "IGNIS_LOG_FORMAT must be one of auto, pretty, json — got `{other}`"
        ))),
    }
}

fn parse_level(raw: &str) -> Result<LogLevel, LogConfigError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "trace" => Ok(LogLevel::Trace),
        "debug" => Ok(LogLevel::Debug),
        "info" => Ok(LogLevel::Info),
        "warn" => Ok(LogLevel::Warn),
        "error" => Ok(LogLevel::Error),
        other => Err(LogConfigError(format!(
            "IGNIS_LOG_LEVEL must be one of trace, debug, info, warn, error — got `{other}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env_map(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    }

    fn tty() -> bool {
        true
    }

    fn no_tty() -> bool {
        false
    }

    #[test]
    fn defaults_are_auto_and_info() {
        let config = resolve(no_env, no_tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Json, "auto + no TTY resolves to json");
        assert_eq!(config.level, LogLevel::Info);
    }

    #[test]
    fn auto_picks_pretty_for_a_simulated_tty() {
        let config = resolve(no_env, tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Pretty);
    }

    #[test]
    fn auto_picks_json_when_not_a_tty() {
        let config = resolve(no_env, no_tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Json);
    }

    #[test]
    fn an_explicit_format_overrides_tty_autodetection_pretty_forced_json() {
        let env = env_map(&[("IGNIS_LOG_FORMAT", "json")]);
        // TTY says "pretty" (interactive), but the explicit env var wins.
        let config = resolve(env, tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Json);
    }

    #[test]
    fn an_explicit_format_overrides_tty_autodetection_forced_pretty() {
        let env = env_map(&[("IGNIS_LOG_FORMAT", "pretty")]);
        // TTY says "json" (non-interactive), but the explicit env var wins.
        let config = resolve(env, no_tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Pretty);
    }

    #[test]
    fn format_values_are_case_insensitive_and_trimmed() {
        let env = env_map(&[("IGNIS_LOG_FORMAT", " JSON ")]);
        let config = resolve(env, tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Json);
    }

    #[test]
    fn an_invalid_format_is_a_config_error() {
        let env = env_map(&[("IGNIS_LOG_FORMAT", "yaml")]);
        let err = resolve(env, tty, LogConfigOverride::default()).expect_err("must reject");
        assert!(err.0.contains("IGNIS_LOG_FORMAT"), "{err}");
        assert!(err.0.contains("yaml"), "{err}");
    }

    #[test]
    fn every_level_parses() {
        for (raw, expected) in [
            ("trace", LogLevel::Trace),
            ("debug", LogLevel::Debug),
            ("info", LogLevel::Info),
            ("warn", LogLevel::Warn),
            ("error", LogLevel::Error),
        ] {
            let config = resolve(
                move |k| {
                    if k == "IGNIS_LOG_LEVEL" {
                        Some(raw.to_string())
                    } else {
                        None
                    }
                },
                no_tty,
                LogConfigOverride::default(),
            )
            .expect("resolve");
            assert_eq!(config.level, expected, "{raw}");
        }
    }

    #[test]
    fn level_values_are_case_insensitive_and_trimmed() {
        let env = env_map(&[("IGNIS_LOG_LEVEL", " WARN ")]);
        let config = resolve(env, no_tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.level, LogLevel::Warn);
    }

    #[test]
    fn an_invalid_level_is_a_config_error() {
        let env = env_map(&[("IGNIS_LOG_LEVEL", "verbose")]);
        let err = resolve(env, tty, LogConfigOverride::default()).expect_err("must reject");
        assert!(err.0.contains("IGNIS_LOG_LEVEL"), "{err}");
        assert!(err.0.contains("verbose"), "{err}");
    }

    #[test]
    fn color_tracks_the_real_tty_not_the_chosen_format() {
        // Auto + TTY: pretty and color both follow the same real terminal.
        let config = resolve(no_env, tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Pretty);
        assert!(config.color);

        // Explicit `pretty` forced on a non-interactive pipe: format is
        // pretty, but color stays off since the real stdout isn't a TTY.
        let env = env_map(&[("IGNIS_LOG_FORMAT", "pretty")]);
        let config = resolve(env, no_tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Pretty);
        assert!(!config.color, "color must not follow a forced format, only the real TTY");

        // Explicit `json` forced on an interactive TTY: format is json, but
        // color is still recorded true since the real stdout is a TTY.
        let env = env_map(&[("IGNIS_LOG_FORMAT", "json")]);
        let config = resolve(env, tty, LogConfigOverride::default()).expect("resolve");
        assert_eq!(config.format, LogFormat::Json);
        assert!(config.color);
    }
}
